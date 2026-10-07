"""Frozen patch correspondence using the released Probe3D data splits.

The MIT-licensed Probe3D dataset readers are vendored under evaluation/vendor.
Matching and aggregation are implemented here to avoid its FAISS-GPU dependency.
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.utils.common import (
    base_parser, evaluation_identity, load_backbone, prepare_paths, print_progress,
    utc_now, write_json,
)


SPAIR_BINS = ("0", "1", "2", "all")
NAVI_EDGES = (0, 30, 60, 90, 120)
SCANNET_EDGES = (0, 15, 30, 60, 180)
PROBE3D_COMMIT = "a1f14640076e38b8bc07b66d0fe2d01d15691e9d"
PROBE3D_SOURCE = f"https://github.com/mbanani/probe3d/tree/{PROBE3D_COMMIT}"


@torch.no_grad()
def patch_features(backbone, images, patch_size, num_register_tokens=0):
    """Probe3D iBOT adapter: raw final block, before final LayerNorm.

    SPair normalizes these vectors before keypoint sampling; NAVI normalizes
    only after bicubic interpolation to the geometry grid. These operations
    do not commute, so return raw patch vectors here.
    """
    if hasattr(backbone, "extract_patch_map"):
        return backbone.extract_patch_map(images)
    if images.shape[-2] % patch_size or images.shape[-1] % patch_size:
        raise ValueError("Correspondence image dimensions must be patch aligned")
    tokens = backbone.prepare_tokens(images)
    for block in backbone.blocks:
        tokens = block(tokens)
    height, width = images.shape[-2] // patch_size, images.shape[-1] // patch_size
    tokens = tokens[:, 1 + num_register_tokens:]
    if tokens.shape[1] != height * width:
        raise ValueError("Backbone token grid does not match correspondence image size")
    features = tokens.float().transpose(1, 2).reshape(len(images), -1, height, width)
    return features


def nearest_ratio_matches(source, target, max_matches=1000, chunk_size=1024):
    """Probe3D cosine 2-NN ratio ranking, in bounded query chunks."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Correspondence features must have compatible [points, channels] shapes")
    if len(source) == 0 or len(target) < 2:
        raise ValueError("Correspondence pair needs valid source points and at least two target points")
    source = F.normalize(source.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    weights, neighbors = [], []
    for query in source.split(chunk_size):
        similarity, index = (query @ target.T).topk(2, dim=1)
        distance = (1 - similarity).clamp_min(1e-9)
        weights.append(1 - distance[:, 0] / distance[:, 1])
        neighbors.append(index[:, 0])
    weights, neighbors = torch.cat(weights), torch.cat(neighbors)
    _, source_indices = weights.topk(min(max_matches, len(weights)))
    return source_indices, neighbors[source_indices]


def relative_rotation_degrees(matrix):
    trace = torch.trace(matrix[:3, :3].float())
    return math.degrees(torch.acos(((trace - 1) / 2).clamp(-1, 1)).item())


def binned_pair_recall(rows, edges):
    """Mean pair recall in each half-open angle interval, as in Probe3D."""
    result = {}
    for left, right in zip(edges[:-1], edges[1:]):
        selected = [score for angle, score in rows if left <= angle < right]
        if not selected:
            raise ValueError(f"No correspondence pairs in rotation bin [{left}, {right})")
        result[f"{left}-{right}"] = 100.0 * float(np.mean(selected))
    return result


def _spair_errors(features, source_keypoints, target_keypoints, bbox_scale, image_size):
    """Probe3D PCK errors for keypoints annotated in both views."""
    features = F.normalize(features.float(), dim=1)
    source, target = features
    source_keypoints = source_keypoints.float()
    target_keypoints = target_keypoints.float()
    valid = (source_keypoints[:, 2] == 1) & (target_keypoints[:, 2] == 1)
    if not valid.any():
        return torch.empty(0)
    source_points = source_keypoints[valid, :2] / image_size
    grid = (2 * source_points - 1).reshape(1, 1, -1, 2).to(source.device)
    query = F.grid_sample(source[None], grid, mode="bilinear", align_corners=True)[0, :, 0].T
    similarity = query @ target.flatten(1)
    target_height, target_width = target.shape[-2:]
    locations = similarity.argmax(1)
    predicted = torch.stack((locations % target_width, locations // target_width), dim=1).float()
    predicted[:, 0] /= target_width
    predicted[:, 1] /= target_height
    truth = target_keypoints[valid, :2].to(source.device) / image_size
    return (predicted - truth).norm(dim=1).cpu() / float(bbox_scale)


def _spair_dataset_class():
    """Read shared annotations once without changing upstream subset sampling."""
    from evaluation.vendor.probe3d.evals.datasets.spair import SPairDataset

    class CachedSPairDataset(SPairDataset):
        pair_annotations = None
        image_annotations_cache = None

        def get_pair_annotations(self):
            if type(self).pair_annotations is None:
                type(self).pair_annotations = super().get_pair_annotations()
            # Each dataset must have its own list for upstream seed-20 shuffle.
            return list(type(self).pair_annotations)

        def get_image_annotations(self):
            if type(self).image_annotations_cache is None:
                type(self).image_annotations_cache = super().get_image_annotations()
            return type(self).image_annotations_cache

    return CachedSPairDataset


def run_spair(backbone, metadata, datasets_root, device):
    from evaluation.vendor.probe3d.evals.datasets.spair import CLASS_IDS
    SPairDataset = _spair_dataset_class()
    root = datasets_root / "SPair-71k"
    if not root.is_dir():
        raise FileNotFoundError(f"Probe3D SPair-71k directory missing: {root}")
    per_category, pair_ids = {}, {}
    for class_name in CLASS_IDS:
        category, category_pairs = {}, {}
        for difficulty in (0, 1, 2, None):
            dataset = SPairDataset(
                str(root), "test", image_size=800, image_mean="imagenet",
                use_bbox=False, class_name=class_name, num_instances=200, vp_diff=difficulty,
            )
            category_pairs[str(difficulty) if difficulty is not None else "all"] = [
                item["filename"] for item in dataset.instances
            ]
            errors = []
            for index in range(len(dataset)):
                src, _, src_kp, dst, _, dst_kp, bbox_scale, _ = dataset[index]
                feats = patch_features(
                    backbone, torch.stack((src, dst)).to(device),
                    metadata["patch_size"], metadata["num_register_tokens"],
                )
                errors.extend(_spair_errors(feats, src_kp, dst_kp, bbox_scale, 800).tolist())
                print_progress(f"SPair {class_name}/{difficulty}", index + 1, len(dataset))
            category[str(difficulty) if difficulty is not None else "all"] = (
                None if not errors else 100.0 * float(np.mean(np.asarray(errors) < 0.1))
            )
        per_category[class_name] = category
        pair_ids[class_name] = category_pairs
    scores = {}
    for difficulty in SPAIR_BINS:
        valid = [row[difficulty] for row in per_category.values() if row[difficulty] is not None]
        if not valid:
            raise ValueError(f"SPair-71k has no evaluated keypoints for viewpoint {difficulty}")
        scores[f"d{difficulty}" if difficulty != "all" else "all"] = float(np.mean(valid))
    return scores, {"root": str(root), "per_category": per_category, "pairs_per_category_and_difficulty": 200,
                    "pair_ids": pair_ids, "source": PROBE3D_SOURCE,
                    "feature_layer": "raw final block before final LayerNorm",
                    "feature_normalization": "L2 before bilinear keypoint sampling",
                    "aggregation": "keypoint mean per category, then category mean",
                    "sampling": "Probe3D seed 20 per category/viewpoint subset", "pck_alpha_bbox": 0.1,
                    "image_size": 800, "bbox_crop": False}


def _valid_flattened_features(feature_map, geometry_grid):
    """Bicubic feature interpolation and valid 3D grid extraction for NAVI."""
    height, width = geometry_grid.shape[-2:]
    resized = F.interpolate(feature_map[None], size=(height, width), mode="bicubic", align_corners=False)[0]
    valid = geometry_grid[2] > 0
    vectors = resized.permute(1, 2, 0)[valid]
    xyz = geometry_grid.permute(1, 2, 0)[valid]
    return vectors, xyz


def run_navi(backbone, metadata, datasets_root, device):
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from evaluation.vendor.probe3d.evals.datasets.navi import NAVI
    root = datasets_root / "navi_v1"
    if not root.is_dir():
        raise FileNotFoundError(f"Probe3D NAVI directory missing: {root}")
    dataset = NAVI(path=str(root), split="test", model="all", image_mean="imagenet",
                   augment_train=False, rotateflip=False, bbox_crop=True,
                   pair_dataset=True, max_angle=120)
    if len(dataset) == 0:
        raise ValueError("NAVI in-the-wild test split has no pairs")
    rows, pair_results = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        images = torch.stack((item["image_0"], item["image_1"])).to(device)
        feats = patch_features(backbone, images, metadata["patch_size"], metadata["num_register_tokens"])
        points = []
        for view in (0, 1):
            xyz = F.interpolate(item[f"xyz_grid_{view}"][None], scale_factor=0.25, mode="nearest")[0].to(device)
            points.append(_valid_flattened_features(feats[view], xyz))
        src_idx, dst_idx = nearest_ratio_matches(points[0][0], points[1][0])
        source_xyz = points[0][1][src_idx]
        target_xyz = points[1][1][dst_idx]
        transform = item["Rt_01"].to(device).float()[:3, :4]
        transformed = source_xyz @ transform[:, :3].T + transform[:, 3]
        error = (transformed - target_xyz).norm(dim=1)
        rows.append((relative_rotation_degrees(transform.cpu()), float((error < 0.02).float().mean())))
        obj, scene, _ = dataset.instances[index]
        pair_results.append({"object": obj, "scene": scene, "pair_id": item["pair_id"],
                             "rotation_degrees": rows[-1][0], "recall_2cm": rows[-1][1],
                             "matches": len(src_idx)})
        print_progress("NAVI correspondence", index + 1, len(dataset))
    return binned_pair_recall(rows, NAVI_EDGES), {"root": str(root), "pairs": len(rows),
        "pair_results": pair_results, "source": PROBE3D_SOURCE,
        "feature_layer": "raw final block before final LayerNorm",
        "feature_normalization": "L2 after bicubic geometry-grid interpolation",
        "aggregation": "mean pair recall per half-open rotation bin",
        "split": "wild/all", "image_size": [512, 512], "bbox_crop": True,
        "geometry_scale": 0.25, "max_correspondences": 1000, "3d_threshold_m": 0.02,
        "pairing": "released Probe3D NAVI test pair selection, seed 8"}


def _pixel_grid(height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32) + 0.5,
                          torch.arange(width, device=device, dtype=torch.float32) + 0.5, indexing="ij")
    return torch.stack((x, y, torch.ones_like(x)), dim=-1).reshape(-1, 3)


def _depth_points(feature, depth, intrinsics):
    height, width = depth.shape[-2:]
    grid = _pixel_grid(height, width, feature.device)
    depth_values = depth.reshape(-1).to(feature.device)
    valid = depth_values > 0
    xyz = (grid[valid] @ torch.inverse(intrinsics.to(feature.device)).T) * depth_values[valid, None]
    coordinates = grid[valid, :2]
    normalized = coordinates.clone()
    normalized[:, 0] = normalized[:, 0] / width * 2 - 1
    normalized[:, 1] = normalized[:, 1] / height * 2 - 1
    vectors = F.grid_sample(feature[None], normalized[None, None], align_corners=False)[0, :, 0].T
    return vectors, xyz


def _project(xyz, intrinsics):
    homogeneous = xyz @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-9)


def run_scannet(backbone, metadata, datasets_root, device):
    from evaluation.vendor.probe3d.evals.datasets.scannet_pairs import ScanNetPairsDataset
    root = datasets_root / "scannet_test_1500"
    if not root.is_dir():
        raise FileNotFoundError(f"Probe3D ScanNet test pairs missing: {root}")
    # The released reader hard-codes its historical path. Override only the root;
    # its exact test.npz pairs, intrinsics and preprocessing remain unchanged.
    class LocatedScanNet(ScanNetPairsDataset):
        def get_instances(self, root_path):
            self.root = str(root)
            return super().get_instances(self.root)
    dataset = LocatedScanNet()
    if len(dataset) == 0:
        raise ValueError("ScanNet released test split has no pairs")
    rows, pair_results = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        images = torch.stack((item["rgb_0"], item["rgb_1"])).to(device)
        feats = patch_features(backbone, images, metadata["patch_size"], metadata["num_register_tokens"])
        depths = [F.interpolate(item[f"depth_{view}"][None], scale_factor=0.25,
                                mode="nearest")[0].to(device) for view in (0, 1)]
        intrinsics = item["K"].to(device).float().clone()
        intrinsics[:2] *= 0.25
        src, xyz0 = _depth_points(feats[0], depths[0], intrinsics)
        dst, xyz1 = _depth_points(feats[1], depths[1], intrinsics)
        src_idx, dst_idx = nearest_ratio_matches(src, dst)
        transform = item["Rt_1"].to(device).float()[:3, :4]
        transformed = xyz0[src_idx] @ transform[:, :3].T + transform[:, 3]
        uv0, uv1 = _project(transformed, intrinsics), _project(xyz1[dst_idx], intrinsics)
        error = (uv0 - uv1).norm(dim=1)
        rows.append((relative_rotation_degrees(transform.cpu()), float((error < 10).float().mean())))
        pair_results.append({"scene": item["sequence_id"], "source_frame": item["frame_0"],
                             "target_frame": item["frame_1"], "rotation_degrees": rows[-1][0],
                             "recall_10px": rows[-1][1], "matches": len(src_idx)})
        print_progress("ScanNet correspondence", index + 1, len(dataset))
    return binned_pair_recall(rows, SCANNET_EDGES), {"root": str(root), "pairs": len(rows),
        "pair_results": pair_results, "source": PROBE3D_SOURCE,
        "feature_layer": "raw final block before final LayerNorm",
        "feature_normalization": "L2 after bilinear geometry-point sampling",
        "image_size": [480, 640], "geometry_scale": 0.25, "max_correspondences": 1000,
        "reprojection_threshold_px_at_geometry_scale": 10,
        "split": "Probe3D scannet_test_1500/test.npz"}


def main(dataset_name):
    evaluation_name = f"{dataset_name}_correspondence"
    args = prepare_paths(base_parser(f"Probe3D {dataset_name} frozen correspondence").parse_args(), evaluation_name)
    if not torch.cuda.is_available():
        raise RuntimeError("Correspondence evaluation requires an NVIDIA GPU")
    started, start_time = utc_now(), time.monotonic()
    backbone, metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    device = torch.device("cuda:0")
    backbone.to(device).eval()
    from evaluation.utils.correspondence_features import (
        CorrespondenceFeatures, fit_voc_standardizer, load_patch_projection,
    )
    variant = args.correspondence_feature_variant
    descriptions = {
        "raw_final": "raw final block before final LayerNorm",
        "final_norm_standardized": "final block after final LayerNorm and frozen VOC-train StandardScaler",
        "projection": "trained teacher patch-head output logits, after final backbone LayerNorm",
        "projection_softmax": "softmax of trained teacher patch-head output logits, after final backbone LayerNorm",
        "concat_4_6_8_12": "concatenated raw outputs of blocks 4, 6, 8 and 12, without final LayerNorm",
    }
    feature_protocol = {"variant": variant, "feature_layer": descriptions[variant],
                        "official_probe3d_feature_recipe": variant == "raw_final",
                        "projection_head": None, "standardization": None}
    projection = None
    if variant.startswith("projection"):
        projection, head_metadata = load_patch_projection(args.checkpoint, args.checkpoint_key)
        projection.to(device).eval()
        feature_protocol["projection_head"] = head_metadata
        if variant == "projection_softmax":
            feature_protocol.update(softmax_temperature=args.correspondence_softmax_temperature,
                                    centering=False, sinkhorn=False)
    extractor = CorrespondenceFeatures(backbone, variant, metadata["patch_size"],
                                       metadata["num_register_tokens"], projection,
                                       args.correspondence_softmax_temperature).eval()
    if variant == "final_norm_standardized":
        feature_protocol["standardization"] = fit_voc_standardizer(
            extractor, args.datasets_root, args.output_dir, device, args.seed, args.num_workers)
    write_json(args.output_dir / "feature_protocol.json", feature_protocol)
    print(f"Correspondence features: {feature_protocol}", flush=True)
    runners = {"spair": run_spair, "navi": run_navi, "scannet": run_scannet}
    with torch.inference_mode():
        metrics, dataset_metadata = runners[dataset_name](extractor, metadata, args.datasets_root, device)
    dataset_metadata["feature_layer"] = descriptions[variant]
    dataset_metadata["feature_variant_protocol"] = feature_protocol
    write_json(args.result_json, {
        "evaluation": evaluation_name, "task": "semantic_correspondence" if dataset_name == "spair" else "geometric_correspondence",
        "dataset": {"spair": "SPair-71k", "navi": "NAVI", "scannet": "ScanNet"}[dataset_name],
        "status": "completed", "started_at": started, "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start_time,
        "model": metadata, "evaluation_identity": evaluation_identity(args),
        "dataset_protocol": dataset_metadata, "metrics": metrics,
    })
    print(f"Completed {evaluation_name}: {metrics}", flush=True)
