#!/usr/bin/env python3
"""Export individual patch-cosine panels from two epoch-200 iBOT ViT-S/16 teachers.

Run in the existing training environment on one Slurm GPU:
  sbatch --gres=gpu:1 --wrap="srun ./.conda-env/bin/python -u patch_cosine_similarity.py"

All settings are hard-coded below. No arguments, repository imports, downloads,
assembled figures, or output subfolders. Dependencies: torch >= 2, numpy,
Pillow, matplotlib. Inputs: five upright PNGs beside this script and the same
ImageNet validation ImageFolder as imagenet_visualizations.py.
Each image yields *_original.png, *_reference.png, *_region.png and raw cosine
arrays in *_cosine.npz. Query locations and provenance are in manifest.json.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# -------------------------- Hard-coded settings --------------------------
ROOT = Path(__file__).resolve().parent
IMAGENET_VAL = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/datasets/imagenet/val")
OUTPUT_DIR = ROOT / "output/patch_cosine_similarity_1000"
CHECKPOINTS = {
    "reference": ROOT / "output/long_ibot_vit_small_reference/85535_3/checkpoint_source1000_continuation0200.pth",
    "region": ROOT / "output/long_ibot_vit_small/85535_0/checkpoint_source1000_continuation0200.pth",
}
CHECKPOINT_KEY = "teacher"
RESOLUTION = 1024                    # Square input; must be divisible by 16.
N_IMAGENET_IMAGES = 1000
SEED = 1                            # Reproducible image and query sampling.
DEVICE = "cuda"
COLOR_MIN, COLOR_MAX = 0.0, 1.0       # Fixed viridis scale for BOTH models.
OVERWRITE = False                   # Change intentionally to replace exports.
ARXIV = "https://arxiv.org/html/2508.10104v1/images/gram/comparison_w_wo/"
# Names, published source URLs, and zero-based query (row, column) at grid 64.
# Locations recovered from Figure 10 red cells; originals are already upright.
EXAMPLES = [
    ("flower", ARXIV + "P_20250105_134132.jpg_croped.lr.jpg", (32, 32)),
    ("geese", ARXIV + "IMG_3727.HEIC_croped.lr.jpg", (48, 48)),
    ("meal", ARXIV + "IMG_3730.HEIC_croped.lr.jpg", (16, 48)),
    ("seals", ARXIV + "IMG_20190618_192715.jpg_croped.lr.jpg", (48, 48)),
    ("sheep", ARXIV + "P_20240705_151013.jpg_croped.lr.jpg", (48, 48)),
]
# ------------------------------------------------------------------------


def write_manifest(manifest):
    temp = OUTPUT_DIR / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(OUTPUT_DIR / "manifest.json")


def validate_configuration():
    if len(sys.argv) != 1:
        raise ValueError("No arguments: edit the hard-coded settings above.")
    if type(RESOLUTION) is not int or RESOLUTION < 16 or RESOLUTION % 16:
        raise ValueError("RESOLUTION must be a positive multiple of 16")
    if type(N_IMAGENET_IMAGES) is not int or N_IMAGENET_IMAGES < 1:
        raise ValueError("N_IMAGENET_IMAGES must be positive")
    if not math.isfinite(COLOR_MIN) or not math.isfinite(COLOR_MAX) or COLOR_MAX <= COLOR_MIN:
        raise ValueError("Color limits must be finite and increasing")
    if CHECKPOINT_KEY not in ("teacher", "student") or list(CHECKPOINTS) != ["reference", "region"]:
        raise ValueError("Specify teacher/student and exactly reference, region checkpoints")
    missing = [str(p) for p in CHECKPOINTS.values() if not p.is_file()]
    missing += [str(ROOT / f"{name}.png") for name, _, _ in EXAMPLES if not (ROOT / f"{name}.png").is_file()]
    if not IMAGENET_VAL.is_dir():
        missing.append(str(IMAGENET_VAL))
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not OVERWRITE:
        raise FileExistsError(f"Output folder is nonempty: {OUTPUT_DIR}; set OVERWRITE=True intentionally.")


def imagenet_samples():
    """Enumerate the same sorted class ImageFolder as torchvision.ImageFolder.

    Avoid importing torchvision only to list files. Class names, directory
    traversal and extensions follow its default ImageFolder enumeration.
    """
    extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
    classes = sorted(p.name for p in IMAGENET_VAL.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"Expected ImageNet class subfolders in {IMAGENET_VAL}")
    samples = []
    for class_index, class_name in enumerate(classes):
        for directory, _, filenames in sorted(os.walk(IMAGENET_VAL / class_name, followlinks=True)):
            for filename in sorted(filenames):
                if filename.lower().endswith(extensions):
                    samples.append((Path(directory) / filename, class_index, class_name))
    if len(samples) < N_IMAGENET_IMAGES:
        raise ValueError(f"Requested {N_IMAGENET_IMAGES} ImageNet images but found {len(samples)}")
    return samples


def select_images():
    import numpy as np
    grid = RESOLUTION // 16
    selected = []
    for number, (name, url, (row, col)) in enumerate(EXAMPLES, 1):
        normalized_xy = [(col + 0.5) / 64, (row + 0.5) / 64]
        query = [min(grid - 1, int(normalized_xy[1] * grid)),
                 min(grid - 1, int(normalized_xy[0] * grid))]
        selected.append({"name": f"fig10_{number:02d}_{name}", "source_path": str(ROOT / f"{name}.png"),
                         "source_url": url, "preprocessing": "square_resize",
                         "query_rc": query, "published_query_rc_grid64": [row, col],
                         "published_normalized_xy": normalized_xy})
    samples = imagenet_samples()
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(samples), size=N_IMAGENET_IMAGES, replace=False)
    for number, index in enumerate(indices, 1):
        path, class_index, class_name = samples[int(index)]
        patch = int(rng.integers(0, grid * grid))
        selected.append({"name": f"imagenet_{number:03d}_{class_name}_{path.stem}",
                         "source_path": str(path.resolve()), "preprocessing": "resize_short_side_center_crop",
                         "dataset_index": int(index), "class_index": class_index, "class_name": class_name,
                         "query_rc": [patch // grid, patch % grid]})
    return selected


def make_backbone(position_grid=14, register_count=0, layer_scale=False):
    """Inference-only standard iBOT ViT-S/16; state names match training.

    Adapted from the iBOT/DINO ViT (https://github.com/bytedance/ibot).
    SDPA computes the same attention without constructing a saved attention map.
    Dropout/stochastic depth are identities during inference, hence omitted.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(384, 1536)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(1536, 384)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(384, 1152, bias=True)
            self.proj = nn.Linear(384, 384)

        def forward(self, x):
            batch, tokens, dim = x.shape
            qkv = self.qkv(x).reshape(batch, tokens, 3, 6, 64).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            return self.proj(x.transpose(1, 2).reshape(batch, tokens, dim))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(384, eps=1e-6)
            self.attn = Attention()
            self.norm2 = nn.LayerNorm(384, eps=1e-6)
            self.mlp = Mlp()
            if layer_scale:
                self.gamma_1 = nn.Parameter(torch.ones(384))
                self.gamma_2 = nn.Parameter(torch.ones(384))

        def forward(self, x):
            y = self.attn(self.norm1(x))
            x = x + (self.gamma_1 * y if layer_scale else y)
            y = self.mlp(self.norm2(x))
            return x + (self.gamma_2 * y if layer_scale else y)

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 384, kernel_size=16, stride=16)

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    class ViTSmall(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
            self.pos_embed = nn.Parameter(torch.zeros(1, position_grid ** 2 + 1, 384))
            if register_count:
                self.register_tokens = nn.Parameter(torch.zeros(1, register_count, 384))
            self.blocks = nn.ModuleList([Block() for _ in range(12)])
            self.norm = nn.LayerNorm(384, eps=1e-6)

        def forward(self, image):
            x = self.patch_embed(image)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            height, width = image.shape[-2:]
            gh, gw = height // 16, width // 16
            if gh == gw == position_grid:
                pos = self.pos_embed
            else:
                patch_pos = self.pos_embed[:, 1:].reshape(1, position_grid, position_grid, 384).permute(0, 3, 1, 2)
                # Match original iBOT bicubic interpolation, including +0.1.
                patch_pos = F.interpolate(patch_pos, scale_factor=((gh + 0.1) / position_grid, (gw + 0.1) / position_grid), mode="bicubic", align_corners=False)
                if patch_pos.shape[-2:] != (gh, gw):
                    raise ValueError("Unexpected positional interpolation dimensions")
                pos = torch.cat((self.pos_embed[:, :1], patch_pos.flatten(2).transpose(1, 2)), dim=1)
            x = x + pos
            if register_count:
                x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
            for block in self.blocks:
                x = block(x)
            return self.norm(x)[:, 1 + register_count:]

    return ViTSmall()


def load_backbone(path, key):
    import torch
    # These are the user's own full training checkpoints, with serialized args.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint: {path}")
    if key in checkpoint:
        raw = checkpoint[key]
    elif isinstance(checkpoint.get("state_dict"), dict):
        raw = checkpoint["state_dict"]
    elif checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        raw = checkpoint
    else:
        raise ValueError(f"No {key} weights in {path}; available entries: {list(checkpoint)[:15]}")
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint entry {key} is not a state dictionary")
    state = {}
    for name, value in raw.items():
        if not torch.is_tensor(value):
            continue
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name.startswith("backbone."):
            name = name[len("backbone."):]
        if name in state:
            raise ValueError(f"Duplicate canonical checkpoint weight: {name}")
        state[name] = value
    if "cls_token" not in state or tuple(state["cls_token"].shape) != (1, 1, 384):
        raise ValueError("This script expects a ViT-S backbone with hidden dimension 384")
    weight = state.get("patch_embed.proj.weight")
    if weight is None or tuple(weight.shape) != (384, 3, 16, 16):
        raise ValueError("This script expects RGB, patch-size-16 iBOT weights")
    pos = state.get("pos_embed")
    if pos is None or pos.ndim != 3 or pos.shape[0] != 1 or pos.shape[-1] != 384:
        raise ValueError("Missing or malformed positional embedding")
    grid = math.isqrt(pos.shape[1] - 1)
    if grid ** 2 != pos.shape[1] - 1:
        raise ValueError("Expected square patch-position grid plus CLS position")
    registers = state.get("register_tokens")
    if registers is not None and (registers.ndim != 3 or registers.shape[0] != 1 or registers.shape[2] != 384):
        raise ValueError("Malformed register_tokens")
    count = 0 if registers is None else registers.shape[1]
    model = make_backbone(grid, count, "blocks.0.gamma_1" in state)
    expected = set(model.state_dict())
    extra_backbone = sorted(k for k in state if k not in expected and k.startswith(("blocks.", "patch_embed.", "norm.", "fc_norm.", "reg_token")))
    if extra_backbone:
        raise ValueError(f"Unsupported backbone components: {extra_backbone[:10]}")
    missing = sorted(expected - set(state))
    if missing:
        raise ValueError(f"Missing backbone weights: {missing[:10]}")
    # Heads, masked_embed and optimizer/loss state are not used for inference.
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    model.eval().requires_grad_(False)
    stat = path.stat()
    metadata = {"checkpoint": str(path), "checkpoint_key": key, "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "position_grid": grid, "register_count": count,
                "architecture": "vit_small", "patch_size": 16}
    return model, metadata


def prepare_image(record):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(record["source_path"]) as source:
        image = source.convert("RGB")
        record["source_size_wh"] = list(image.size)
    if record["preprocessing"] == "square_resize":
        image = image.resize((RESOLUTION, RESOLUTION), Image.Resampling.BICUBIC)
    else:
        # Match T.Resize(integer, BICUBIC) + T.CenterCrop used in
        # imagenet_visualizations.py, preserving aspect ratio for ImageNet.
        width, height = image.size
        if width <= height:
            target = (RESOLUTION, int(RESOLUTION * height / width))
        else:
            target = (int(RESOLUTION * width / height), RESOLUTION)
        if image.size != target:
            image = image.resize(target, Image.Resampling.BICUBIC)
        left = round((image.width - RESOLUTION) / 2)
        top = round((image.height - RESOLUTION) / 2)
        image = image.crop((left, top, left + RESOLUTION, top + RESOLUTION))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return image, ((tensor - mean) / std).unsqueeze(0)


def cosine_map(features, query_rc):
    import torch
    import torch.nn.functional as F
    grid = RESOLUTION // 16
    if tuple(features.shape) != (1, grid * grid, 384):
        raise ValueError(f"Unexpected patch features: {tuple(features.shape)}")
    features = features[0].float()
    if not torch.isfinite(features).all() or (features.norm(dim=-1) <= 1e-12).any():
        raise ValueError("Non-finite or zero patch features")
    features = F.normalize(features, dim=-1)
    row, col = query_rc
    return (features @ features[row * grid + col]).reshape(grid, grid).cpu().numpy()


def save_heatmap(path, values, query_rc):
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image, ImageDraw
    # No interpolation of features or similarities, no per-image normalization.
    colors = colormaps["viridis"](np.clip((values - COLOR_MIN) / (COLOR_MAX - COLOR_MIN), 0, 1), bytes=True)[..., :3]
    image = Image.fromarray(colors).resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
    row, col = query_rc
    ImageDraw.Draw(image).rectangle((col * 16, row * 16, (col + 1) * 16 - 1, (row + 1) * 16 - 1), fill="red")
    image.save(path, dpi=(300, 300))


def run():
    import numpy as np
    import torch
    from matplotlib import colormaps
    colormaps["viridis"]  # Check dependencies before creating output files.
    device = torch.device(DEVICE)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    records = select_images()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__), "device": str(device),
        "continuation_epoch": 200, "checkpoint_key": CHECKPOINT_KEY,
        "resolution": RESOLUTION, "patch_size": 16, "seed": SEED,
        "imagenet_val": str(IMAGENET_VAL), "n_imagenet_images": N_IMAGENET_IMAGES,
        "feature_definition": "final-block backbone spatial output after LayerNorm, L2-normalized; cosine similarity, not 1-cosine distance",
        "color_limits": [COLOR_MIN, COLOR_MAX], "colormap": "viridis",
        "upsampling": "nearest; patch cells retained; no smoothing",
        "source_note": "DINOv3 Figure 10 public crops are 512x512; flower/sheep rotated clockwise once to match the paper. Query cells recovered from published maps, not the original plotting script.",
        "raw_values_note": "NPZ values are unmarked and unclipped; display clips to color_limits. Same query and preprocessing for both models.",
        "checkpoints": {}, "images": records,
    }
    write_manifest(manifest)
    # Save the exact preprocessed model inputs as row 1. Never assemble panels.
    for record in records:
        image, _ = prepare_image(record)
        filename = f"{record['name']}_original.png"
        image.save(OUTPUT_DIR / filename, dpi=(300, 300))
        record["outputs"] = {"original": filename}
        record["statistics"] = {}
    write_manifest(manifest)
    reference_maps = {}
    for method, checkpoint in CHECKPOINTS.items():
        print(f"Loading {method}, epoch 200: {checkpoint}", flush=True)
        model, metadata = load_backbone(checkpoint, CHECKPOINT_KEY)
        model.to(device)
        manifest["checkpoints"][method] = metadata
        with torch.inference_mode():
            for number, record in enumerate(records, 1):
                print(f"[{method} {number}/{len(records)}] {record['name']}, patch {record['query_rc']}", flush=True)
                _, tensor = prepare_image(record)
                features = model(tensor.to(device))
                values = cosine_map(features, record["query_rc"])
                del features, tensor
                filename = f"{record['name']}_{method}.png"
                save_heatmap(OUTPUT_DIR / filename, values, record["query_rc"])
                record["outputs"][method] = filename
                record["statistics"][method] = {
                    "min_cosine": float(values.min()), "max_cosine": float(values.max()),
                    "fraction_clipped_low": float((values < COLOR_MIN).mean()),
                    "fraction_clipped_high": float((values > COLOR_MAX).mean()),
                }
                if method == "reference":
                    reference_maps[record["name"]] = values
                else:
                    raw_filename = f"{record['name']}_cosine.npz"
                    np.savez_compressed(OUTPUT_DIR / raw_filename,
                                        reference=reference_maps.pop(record["name"]), region=values,
                                        query_rc=np.asarray(record["query_rc"], dtype=np.int64))
                    record["outputs"]["raw_cosine"] = raw_filename
                write_manifest(manifest)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest)
    print(f"Complete: {len(records)} images, {3 * len(records)} individual PNGs in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        validate_configuration()
        run()
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
