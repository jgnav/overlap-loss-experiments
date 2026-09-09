from collections.abc import Mapping
from pathlib import Path

import torch


RESUME_COMPATIBILITY_KEYS = (
    "arch",
    "patch_size",
    "out_dim",
    "patch_out_dim",
    "shared_head",
    "shared_head_teacher",
    "norm_in_head",
    "act_in_head",
    "norm_last_layer",
    "use_masked_im_modeling",
    "pred_ratio",
    "pred_ratio_var",
    "pred_shape",
    "pred_start_epoch",
    "pred_aspect_ratio",
    "global_crops_number",
    "global_crop_size",
    "global_crops_scale",
    "local_crops_number",
    "local_crop_size",
    "local_crops_scale",
    "student_temp",
    "centering",
    "teacher_target_version",
    "center_momentum",
    "center_momentum2",
    "warmup_teacher_temp",
    "teacher_temp",
    "warmup_teacher_patch_temp",
    "teacher_patch_temp",
    "warmup_teacher_temp_epochs",
    "lambda1",
    "lambda2",
    "lambda3",
    "region_min_area",
    "momentum_teacher",
    "epochs",
    "batch_size_per_gpu",
    "reference_batch_size",
    "gpu_count",
    "world_size",
    "effective_batch_size",
    "optimizer",
    "lr_schedule",
    "lr",
    "min_lr",
    "warmup_epochs",
    "weight_decay",
    "weight_decay_end",
    "precision",
    "use_fp16",
    "seed",
)

RESUME_TOPOLOGY_KEYS = {
    "batch_size_per_gpu",
    "gpu_count",
    "world_size",
}


def _read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("The checkpoint root must be a mapping")
    return checkpoint


def read_pretrained_checkpoint(args):
    checkpoint = _read_checkpoint(args.initial_checkpoint)

    required = {"student", "teacher", "ibot_loss"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            "The iBOT checkpoint is missing required training state: "
            f"{missing}. Use the official full checkpoint, not a "
            "backbone-only checkpoint."
        )
    if not isinstance(checkpoint["ibot_loss"], Mapping):
        raise ValueError("The checkpoint key 'ibot_loss' must be a mapping")
    missing_centers = sorted(
        {"center", "center2"} - set(checkpoint["ibot_loss"])
    )
    if missing_centers:
        raise ValueError(
            "The iBOT loss state is missing pretrained centers: "
            f"{missing_centers}"
        )
    if "epoch" in checkpoint and not isinstance(checkpoint["epoch"], int):
        raise ValueError("The checkpoint epoch must be an integer")
    if "optimizer" in checkpoint and not isinstance(checkpoint["optimizer"], Mapping):
        raise ValueError("The checkpoint key 'optimizer' must be a mapping")
    if "optimizer" in checkpoint:
        missing_optimizer_state = sorted(
            {"state", "param_groups"} - set(checkpoint["optimizer"])
        )
        if missing_optimizer_state:
            raise ValueError(
                "The checkpoint optimizer state is incomplete: "
                f"{missing_optimizer_state}"
            )

    # A completed continuation checkpoint stores its resume position in
    # `epoch` and its absolute provenance in `source_equivalent_epoch`.
    args.source_checkpoint_epoch = checkpoint.get(
        "source_equivalent_epoch", checkpoint.get("epoch")
    )
    args.source_optimizer_available = "optimizer" in checkpoint
    args.source_fp16_scaler_available = "fp16_scaler" in checkpoint
    args.source_args_available = "args" in checkpoint
    return checkpoint


def _checkpoint_argument(checkpoint, name):
    saved_args = checkpoint.get("args")
    if isinstance(saved_args, Mapping):
        return saved_args.get(name)
    return getattr(saved_args, name, None)


def _validate_resume_compatibility(checkpoint, args):
    saved_effective_batch_size = _checkpoint_argument(
        checkpoint, "effective_batch_size"
    )
    configured_effective_batch_size = getattr(
        args, "effective_batch_size", None
    )
    preserve_effective_batch_size = (
        saved_effective_batch_size is not None
        and configured_effective_batch_size is not None
        and saved_effective_batch_size == configured_effective_batch_size
    )

    mismatches = []
    for key in RESUME_COMPATIBILITY_KEYS:
        saved_value = _checkpoint_argument(checkpoint, key)
        if key == "centering" and saved_value is None:
            # Checkpoints written before this option always used centering.
            saved_value = "centering"
        if key == "teacher_target_version" and saved_value is None:
            # Old SK runs normalized CLS/iBOT with SK too. They cannot be
            # resumed exactly under the new overlap-only normalization.
            saved_value = (
                1 if _checkpoint_argument(checkpoint, "centering") == "sinkhorn_knopp"
                else 2
            )
        if saved_value is None or not hasattr(args, key):
            continue
        configured_value = getattr(args, key)
        if configured_value != saved_value:
            if key in RESUME_TOPOLOGY_KEYS and preserve_effective_batch_size:
                continue
            if key in {"precision", "use_fp16"} and getattr(
                args, "resume_allow_precision_change", False
            ):
                continue
            mismatches.append(
                f"{key}: checkpoint={saved_value!r}, config={configured_value!r}"
            )
    if mismatches:
        formatted = "\n  - ".join(mismatches)
        raise ValueError(
            "Resume configuration does not match the saved training state:\n"
            f"  - {formatted}"
        )


def read_resume_checkpoint(args):
    checkpoint = _read_checkpoint(args.resume_checkpoint)
    required = {
        "student",
        "teacher",
        "optimizer",
        "epoch",
        "args",
        "ibot_loss",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            "The resume checkpoint is missing required training state: "
            f"{missing}"
        )
    if not isinstance(checkpoint["epoch"], int):
        raise ValueError("The resume checkpoint epoch must be an integer")
    if not 0 <= checkpoint["epoch"] <= args.epochs:
        raise ValueError(
            "The resume checkpoint epoch must be between 0 and the configured "
            f"training length ({args.epochs}); got {checkpoint['epoch']}"
        )
    if args.use_fp16 and "fp16_scaler" not in checkpoint:
        raise ValueError(
            "The resume checkpoint has no FP16 scaler state, but use_fp16 is enabled"
        )
    _validate_resume_compatibility(checkpoint, args)
    args.resume_epoch = checkpoint["epoch"]
    args.source_checkpoint_epoch = checkpoint.get(
        "source_checkpoint_epoch",
        _checkpoint_argument(checkpoint, "source_checkpoint_epoch"),
    )
    return checkpoint


def load_pretrained_state(
    checkpoint,
    student,
    teacher,
    ibot_loss,
):
    objects = {
        "student": student,
        "teacher": teacher,
    }
    for key, value in objects.items():
        incompatible = value.load_state_dict(checkpoint[key], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"Checkpoint key '{key}' is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    center_state = {
        key: checkpoint["ibot_loss"][key] for key in ("center", "center2")
    }
    incompatible = ibot_loss.load_state_dict(center_state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if key in {"center", "center2"}
    ]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint iBOT centers are incompatible: "
            f"missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def load_resume_state(
    checkpoint,
    student,
    teacher,
    ibot_loss,
    optimizer,
    fp16_scaler,
):
    load_pretrained_state(checkpoint, student, teacher, ibot_loss)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if fp16_scaler is not None:
        fp16_scaler.load_state_dict(checkpoint["fp16_scaler"])
    return checkpoint["epoch"]


def load_continuation_state(
    checkpoint,
    student,
    teacher,
    ibot_loss,
    optimizer,
    fp16_scaler=None,
):
    """Load an external source checkpoint for additional training.

    Unlike an exact resume, continuation starts at continuation epoch zero.
    Model, teacher, centers, and Adam moments are restored when available. A
    source FP16 scaler is deliberately irrelevant to BF16/FP32 continuation.
    """
    load_pretrained_state(checkpoint, student, teacher, ibot_loss)
    optimizer_restored = "optimizer" in checkpoint
    if optimizer_restored:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if fp16_scaler is not None:
        if "fp16_scaler" not in checkpoint:
            raise ValueError(
                "FP16 continuation requires the source FP16 scaler state"
            )
        fp16_scaler.load_state_dict(checkpoint["fp16_scaler"])
    return optimizer_restored


def source_equivalent_epoch(source_checkpoint_epoch, continuation_epoch):
    if source_checkpoint_epoch is None:
        return None
    return source_checkpoint_epoch + continuation_epoch


def continuation_provenance(
    source_checkpoint_epoch,
    continuation_epoch,
    additional_epochs,
    precision,
):
    """Return the unambiguous epoch/precision fields stored in checkpoints."""
    return {
        # `epoch` is retained for compatibility with exact-resume code.
        "epoch": continuation_epoch,
        "continuation_epoch": continuation_epoch,
        "additional_epochs": additional_epochs,
        "source_checkpoint_epoch": source_checkpoint_epoch,
        "source_equivalent_epoch": source_equivalent_epoch(
            source_checkpoint_epoch, continuation_epoch
        ),
        "precision": precision,
    }


def milestone_checkpoint_name(source_checkpoint_epoch, continuation_epoch):
    """Name a permanent checkpoint by both continuation and source epoch."""
    if continuation_epoch <= 0:
        raise ValueError("continuation_epoch must be positive")
    source_epoch = source_equivalent_epoch(
        source_checkpoint_epoch, continuation_epoch
    )
    continuation = f"continuation{continuation_epoch:04d}"
    if source_epoch is None:
        return f"checkpoint_{continuation}.pth"
    return f"checkpoint_source{source_epoch:04d}_{continuation}.pth"
