#!/usr/bin/env python3
"""Export individual DINOv3-style iBOT ViT-S/16 patch similarity panels.

Run inside an allocated Slurm GPU job: srun python export_patch_similarity.py
All configuration is hard-coded below; no arguments or external config needed.
Dependencies: torch >= 2.0, numpy, Pillow, matplotlib (training environment).
No assembled figures are generated. Inputs are already upright PNGs.
The standalone backbone matches the original iBOT/DINO ViT and this repository's
model/vision_transformer.py, including positional interpolation and registers.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


# ======================== HARD-CODED CONFIGURATION ========================
ROOT = Path(__file__).resolve().parent
CONFIG = SimpleNamespace(
    checkpoint_root=ROOT,  # Training repo on the server; relative paths below.
    checkpoint_key="teacher",
    epochs=[0, 50, 100, 150, 200],
    methods=["reference", "region"],
    images=["sheep", "flower", "fruits"],
    image_dir=ROOT,
    output_dir=ROOT / "output/patch_similarity",
    probe_size=512,
    fruit_size=1024,  # Use 512 for less memory; must be divisible by 16.
    device="cuda",
    precision="float32",  # Alternatively "float16" or "bfloat16" on CUDA.
    color_min=0.0,
    color_max=1.0,  # Fixed limits for every checkpoint; no panel-wise rescaling.
    overlay_alpha=0.8,  # Fruit heatmap opacity; pure heatmaps also exported.
    overwrite=False,  # Set True to intentionally replace previous exports.
)
# Epoch 0 uses the shared initialization from config/train.yaml.
# Change its path here if the server stores it elsewhere.
CHECKPOINTS = {
    "region": {
        0: "checkpoints/ibot_vit_small.pth",
        50: "output/long_ibot_vit_small/85535_0/checkpoint_source0850_continuation0050.pth",
        100: "output/long_ibot_vit_small/85535_0/checkpoint_source0900_continuation0100.pth",
        150: "output/long_ibot_vit_small/85535_0/checkpoint_source0950_continuation0150.pth",
        200: "output/long_ibot_vit_small/85535_0/checkpoint_source1000_continuation0200.pth",
    },
    "reference": {
        0: "checkpoints/ibot_vit_small.pth",
        50: "output/long_ibot_vit_small_reference/85535_3/checkpoint_source0850_continuation0050.pth",
        100: "output/long_ibot_vit_small_reference/85535_3/checkpoint_source0900_continuation0100.pth",
        150: "output/long_ibot_vit_small_reference/85535_3/checkpoint_source0950_continuation0150.pth",
        200: "output/long_ibot_vit_small_reference/85535_3/checkpoint_source1000_continuation0200.pth",
    },
}
ARXIV = "https://arxiv.org/html/2508.10104v1/"
SOURCE_IMAGES = {
    "sheep": ("sheep.png", ARXIV + "images/evolution_cosine_u1/P_20240709_135144.jpg_croped.lr.jpg"),
    "flower": ("flower.png", ARXIV + "images/evolution_cosine_u1/P_20250105_134132.jpg_croped.lr.jpg"),
    "fruits": ("fruits.png", ARXIV + "figures/introduction/clutter_scene/market_cropped_coloredit2_faceblur.lr.jpeg"),
}
# DINOv3 Figure 3 cells, recovered from red crosses in the original 256x256
# grid. At lower resolution, normalized centers locate the containing patch.
FRUIT_QUERIES = [
    ("top_1", 15, 104), ("top_2", 73, 149),
    ("top_3", 64, 134), ("top_4", 132, 152),
    ("middle_left", 87, 85), ("middle_right", 104, 213),
    ("bottom_left", 128, 51), ("bottom_right", 216, 232),
]
# ========================================================================


def validate_configuration():
    if len(sys.argv) != 1:
        raise ValueError("This script takes no arguments. Edit the hard-coded CONFIG/CHECKPOINTS instead.")
    for size in [CONFIG.probe_size, CONFIG.fruit_size]:
        if size < 16 or size % 16:
            raise ValueError("Input sizes must be positive and divisible by 16")
    if not math.isfinite(CONFIG.color_min) or not math.isfinite(CONFIG.color_max) or CONFIG.color_max <= CONFIG.color_min:
        raise ValueError("Color limits must be finite and increasing")
    if not 0 <= CONFIG.overlay_alpha <= 1:
        raise ValueError("overlay_alpha must be between 0 and 1")
    if CONFIG.precision not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported inference precision")
    if CONFIG.checkpoint_key not in ("teacher", "student"):
        raise ValueError("checkpoint_key must be teacher or student")
    if len(CONFIG.epochs) != len(set(CONFIG.epochs)) or any(e < 0 for e in CONFIG.epochs):
        raise ValueError("Epochs must be distinct nonnegative integers")
    if len(CONFIG.methods) != len(set(CONFIG.methods)) or len(CONFIG.images) != len(set(CONFIG.images)):
        raise ValueError("Methods and images must not contain duplicates")
    if not CONFIG.methods or not CONFIG.images or not CONFIG.epochs:
        raise ValueError("At least one method, image, and epoch is required")
    for method in CONFIG.methods:
        for epoch in CONFIG.epochs:
            if method not in CHECKPOINTS or epoch not in CHECKPOINTS[method]:
                raise ValueError(f"Missing hard-coded checkpoint for {method}, epoch {epoch}")
    if any(name not in SOURCE_IMAGES for name in CONFIG.images):
        raise ValueError("Unknown image in CONFIG.images")
    CONFIG.epochs.sort()
    for name in ["checkpoint_root", "image_dir", "output_dir"]:
        setattr(CONFIG, name, Path(getattr(CONFIG, name)).expanduser().resolve())


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def source_images(image_dir, names):
    from PIL import Image
    metadata = {}
    for name in names:
        filename, url = SOURCE_IMAGES[name]
        path = image_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing input image: {path}")
        with Image.open(path) as image:
            image.load()
            size = list(image.size)
        metadata[name] = {
            "path": str(path), "source_url": url, "downloaded_size_wh": size,
            "rotation_clockwise_degrees": 0,
            "orientation": "Prepared upright; no additional rotation required",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return metadata


def checkpoint_paths(args):
    result = {}
    missing = []
    for method in args.methods:
        result[method] = {}
        for epoch in args.epochs:
            path = Path(CHECKPOINTS[method][epoch]).expanduser()
            if not path.is_absolute():
                path = args.checkpoint_root / path
            path = path.resolve()
            result[method][epoch] = path
            if not path.is_file():
                missing.append(f"  {method}, epoch {epoch}: {path}")
    if missing:
        raise FileNotFoundError("Missing checkpoints (nothing inferred or skipped):\n" + "\n".join(missing))
    return result


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


def prepare_image(metadata, size):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(metadata["path"]) as source:
        image = source.convert("RGB")
    if metadata["rotation_clockwise_degrees"]:
        image = image.transpose(Image.Transpose.ROTATE_270)
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return image, ((tensor - mean) / std).unsqueeze(0)


def queries_for(name, grid):
    source = FRUIT_QUERIES if name == "fruits" else [("center", 16, 16)]
    source_grid = 256 if name == "fruits" else 32
    queries = []
    for label, row, col in source:
        u, v = (col + 0.5) / source_grid, (row + 0.5) / source_grid
        r, c = min(grid - 1, int(v * grid)), min(grid - 1, int(u * grid))
        queries.append({"label": label, "source_grid": source_grid, "source_rc": [row, col],
                        "normalized_xy": [u, v], "rc": [r, c], "flat_index": r * grid + c})
    return queries


def cosine_maps(features, grid, queries):
    import torch
    import torch.nn.functional as F
    if tuple(features.shape) != (1, grid * grid, 384):
        raise ValueError(f"Expected {grid * grid} spatial features, got {tuple(features.shape)}")
    if not torch.isfinite(features).all() or (features.float().norm(dim=-1) <= 1e-12).any():
        raise ValueError("Backbone produced non-finite or zero patch features")
    features = F.normalize(features[0].float(), dim=-1)
    indices = torch.tensor([q["flat_index"] for q in queries], device=features.device)
    maps = features[indices] @ features.T
    return maps.reshape(len(queries), grid, grid).cpu().numpy()


def add_marker(image, query, grid):
    from PIL import ImageDraw
    image = image.copy()
    draw = ImageDraw.Draw(image)
    row, col = query["rc"]
    x, y = (col + 0.5) * image.width / grid, (row + 0.5) * image.height / grid
    if query["source_grid"] == 32:
        # Figure 6: mark one reference cell in red.
        draw.rectangle((int(col * image.width / grid), int(row * image.height / grid),
                        int((col + 1) * image.width / grid) - 1, int((row + 1) * image.height / grid) - 1), fill="red")
    else:
        # Figure 3: small red cross, readable even on a dense grid.
        radius = max(4, round(image.width * 0.009))
        line = max(2, round(image.width * 0.003))
        draw.line((x - radius, y, x + radius, y), fill="red", width=line)
        draw.line((x, y - radius, x, y + radius), fill="red", width=line)
    return image


def save_panels(directory, name, original, maps, queries, args):
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    grid = maps.shape[-1]
    output = []
    for index, (values, query) in enumerate(zip(maps, queries)):
        colors = colormaps["viridis"](np.clip((values - args.color_min) / (args.color_max - args.color_min), 0, 1), bytes=True)[..., :3]
        pure = Image.fromarray(colors).resize(original.size, Image.Resampling.NEAREST)
        stem = name if name != "fruits" else f"fruits_query_{index:02d}"
        heatmap = add_marker(pure, query, grid)
        heatmap.save(directory / f"{stem}_heatmap.png", dpi=(300, 300))
        panel = Image.blend(original, pure, args.overlay_alpha) if name == "fruits" else pure
        add_marker(panel, query, grid).save(directory / f"{stem}.png", dpi=(300, 300))
        output.append({"panel": f"{stem}.png", "heatmap": f"{stem}_heatmap.png", "query": query,
                       "min_cosine": float(values.min()), "max_cosine": float(values.max()),
                       "fraction_clipped_low": float((values < args.color_min).mean()),
                       "fraction_clipped_high": float((values > args.color_max).mean())})
    # Raw cosine values, not clipped/normalized/overlaid or marked red.
    np.savez_compressed(directory / f"{name}_similarities.npz", cosine=maps,
                        query_rc=np.array([q["rc"] for q in queries]),
                        normalized_xy=np.array([q["normalized_xy"] for q in queries]))
    return output


def run(args):
    images = source_images(args.image_dir, args.images)
    paths = checkpoint_paths(args)
    import numpy as np
    import torch
    # Resolve all inference dependencies before creating any output.
    from matplotlib import colormaps
    colormaps["viridis"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "cpu" and args.precision != "float32":
        raise ValueError("Use float32 on CPU; half precision is supported on CUDA")
    if args.precision == "bfloat16" and device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support bfloat16")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is nonempty: {args.output_dir}. Use a new output_dir or set CONFIG.overwrite = True.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    prepared = {}
    original_dir = args.output_dir / "originals"
    original_dir.mkdir(exist_ok=True)
    for name, meta in images.items():
        size = args.fruit_size if name == "fruits" else args.probe_size
        original, tensor = prepare_image(meta, size)
        queries = queries_for(name, size // 16)
        prepared[name] = (original, tensor, queries)
        original.save(original_dir / f"{name}.png", dpi=(300, 300))
        marked = original
        for query in queries:
            marked = add_marker(marked, query, size // 16)
        marked.save(original_dir / f"{name}_queries.png", dpi=(300, 300))
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__), "device": str(device),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_images": images, "feature_definition": "last-block backbone output after LayerNorm; spatial tokens only; L2-normalized cosine similarity",
        "color_limits": [args.color_min, args.color_max], "runs": [],
        "fruit_coordinate_provenance": "Recovered from DINOv3 Figure 3 red crosses, not original plotting code; mapped by normalized centers to the selected grid.",
        "fruit_resolution_note": "Public source JPEG is 1854x1854; the paper used 4096x4096. Reduced-resolution outputs preserve the location, not the original patch footprint.",
    }
    json_write(args.output_dir / "manifest.json", manifest)
    # Shared initialization is evaluated once and reused for both epoch-0 rows.
    cached_path = None
    cached_results = None
    work = [(m, e, p) for e in args.epochs for m in args.methods for p in [paths[m][e]]]
    for step, (method, epoch, path) in enumerate(work, 1):
        print(f"[{step}/{len(work)}] {method}, epoch {epoch}: {path}", flush=True)
        if path != cached_path:
            model, checkpoint_meta = load_backbone(path, args.checkpoint_key)
            model.to(device)
            results = {}
            with torch.inference_mode():
                for name, (_, tensor, queries) in prepared.items():
                    print(f"  {name}: {tensor.shape[-1]}x{tensor.shape[-1]}, {len(queries)} query/queries", flush=True)
                    context = nullcontext() if args.precision == "float32" else torch.autocast("cuda", dtype=getattr(torch, args.precision))
                    with context:
                        features = model(tensor.to(device))
                    results[name] = cosine_maps(features, tensor.shape[-1] // 16, queries)
                    del features
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cached_path, cached_results = path, (checkpoint_meta, results)
        else:
            print("  Reusing identical checkpoint outputs", flush=True)
        checkpoint_meta, results = cached_results
        directory = args.output_dir / method / f"epoch_{epoch:03d}"
        record = {"method": method, "continuation_epoch": epoch, "checkpoint": checkpoint_meta, "images": {}}
        for name, (original, _, queries) in prepared.items():
            record["images"][name] = save_panels(directory, name, original, results[name], queries, args)
        json_write(directory / "metadata.json", record)
        manifest["runs"].append(record)
        json_write(args.output_dir / "manifest.json", manifest)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    json_write(args.output_dir / "manifest.json", manifest)
    print(f"Complete. Individual images and similarity panels: {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        validate_configuration()
        run(CONFIG)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
