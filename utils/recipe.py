COMMON_IBOT_RECIPE = {
    "patch_size": 16,
    "out_dim": 8192,
    "patch_out_dim": 8192,
    # Reuse the final prototype layer for CLS and patch tokens by default.
    "shared_head": True,
    "norm_in_head": None,
    "act_in_head": "gelu",
    "use_masked_im_modeling": True,
    "pred_ratio": [0.0, 0.3],
    "pred_ratio_var": [0.0, 0.2],
    "pred_shape": "block",
    "pred_start_epoch": 0,
    "pred_aspect_ratio": [0.3, 1 / 0.3],
    "global_crops_number": 2,
    "global_crop_size": 224,
    "local_crops_number": 10,
    "local_crop_size": 96,
    "student_temp": 0.1,
    "region_patch_threshold": 0.5,
    "region_temp": 0.1,
    "region_normalization": "softmax",
    "center_momentum": 0.9,
    "center_momentum2": 0.9,
    # A pretrained iBOT checkpoint has already completed temperature warmup.
    "warmup_teacher_temp": 0.07,
    "warmup_teacher_patch_temp": 0.07,
    "warmup_teacher_temp_epochs": 0,
    "reference_batch_size": 256,
    "distributed_backend": "nccl",
    "dist_url": "env://",
    "saveckp_freq": 40,
    "print_freq": 10,
    "diagnostic_feature_batches": 1,
    "diagnostic_max_patch_features_per_batch": 4096,
    "diagnostic_prototype_chunk_size": 256,
    # Fixed online representation probes are opt-in because they require the
    # ImageNet and VOC datasets in addition to the pre-training data.
    "online_probes_enabled": False,
    "online_probe_frequency": 10,
    "online_probe_datasets_root": "dataset",
    "online_probe_imagenet_train_size": 10000,
    "online_probe_imagenet_val_size": 5000,
    "online_probe_voc_train_size": 400,
    "online_probe_voc_val_size": 200,
    "online_probe_k": 20,
    "online_probe_batch_size": 256,
    "online_probe_num_workers": 0,
    "online_probe_max_concurrent_jobs": 2,
    "online_probe_wait_at_exit": False,
    "online_probe_gpu": None,
}


ARCHITECTURE_RECIPES = {
    "vit_small": {
        "norm_last_layer": False,
        "drop_path": 0.1,
        "global_crops_scale": [0.25, 1.0],
        "local_crops_scale": [0.05, 0.25],
        "clip_grad": 3.0,
        "freeze_last_layer": 1,
        "lr": 0.0005,
        "min_lr": 0.000001,
        "weight_decay_end": 0.4,
    },
    "vit_base": {
        "norm_last_layer": True,
        "drop_path": 0.1,
        "global_crops_scale": [0.32, 1.0],
        "local_crops_scale": [0.05, 0.32],
        "clip_grad": 0.3,
        "freeze_last_layer": 3,
        "lr": 0.00075,
        "min_lr": 0.000002,
        "weight_decay_end": 0.4,
    },
    "vit_large": {
        "norm_last_layer": True,
        "drop_path": 0.2,
        "global_crops_scale": [0.25, 1.0],
        "local_crops_scale": [0.05, 0.25],
        "clip_grad": 0.3,
        "freeze_last_layer": 3,
        "lr": 0.0005,
        "min_lr": 0.0002,
        "weight_decay_end": 0.48,
    },
}


def get_ibot_recipe(architecture):
    try:
        architecture_recipe = ARCHITECTURE_RECIPES[architecture]
    except KeyError as error:
        choices = ", ".join(ARCHITECTURE_RECIPES)
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose from: {choices}"
        ) from error
    return {**COMMON_IBOT_RECIPE, **architecture_recipe}
