"""Continue iBOT pretraining from a full source checkpoint using one YAML config."""

import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
import yaml
from tensorboardX import SummaryWriter

from data import DataAugmentationiBOT, ImageFolderMask
from losses import iBOTLoss
from model import create_model, iBOTHead
from utils import training as utils
from utils.checkpoint import (
    continuation_provenance,
    load_continuation_state,
    load_resume_state,
    milestone_checkpoint_name,
    read_pretrained_checkpoint,
    read_resume_checkpoint,
    source_equivalent_epoch,
)
from utils.recipe import get_ibot_recipe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML training configuration")
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle)

    config = {**get_ibot_recipe(user_config["arch"]), **user_config}
    if "additional_epochs" in user_config and "epochs" in user_config:
        raise ValueError(
            "Configure training length with additional_epochs, not both keys"
        )
    if "additional_epochs" in user_config:
        config["epochs"] = user_config["additional_epochs"]
    elif "epochs" in user_config:
        config["additional_epochs"] = user_config["epochs"]
    else:
        raise ValueError("The configuration must define additional_epochs")
    if config.get("warmup_epochs", 0) != 0:
        raise ValueError(
            "Continuation training must not restart the iBOT learning-rate warm-up; "
            "set warmup_epochs to 0"
        )
    config.setdefault("resume_checkpoint", None)
    config.setdefault("resume_allow_precision_change", False)
    requested_precision = config.get("precision")
    if requested_precision is None:
        requested_precision = "fp16" if config.get("use_fp16", False) else "fp32"
    if "precision" in user_config and "use_fp16" in user_config:
        raise ValueError("Configure precision with either precision or use_fp16, not both")
    precision_override = os.environ.get("IBOT_PRECISION_OVERRIDE")
    if precision_override:
        requested_precision = precision_override
    if requested_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of: fp32, fp16, bf16")
    integer_overrides = {
        "IBOT_BATCH_SIZE_PER_GPU_OVERRIDE": "batch_size_per_gpu",
        "IBOT_GPU_COUNT_OVERRIDE": "gpu_count",
    }
    for environment_key, config_key in integer_overrides.items():
        value = os.environ.get(environment_key)
        if value:
            value = int(value)
            if value <= 0:
                raise ValueError(f"{environment_key} must be positive")
            config[config_key] = value
    config["precision"] = requested_precision
    config["use_fp16"] = requested_precision == "fp16"
    for key in ("data_path", "initial_checkpoint", "output_dir"):
        config[key] = os.path.expandvars(os.path.expanduser(config[key]))
    config["initial_checkpoint"] = Path(config["initial_checkpoint"])
    if config["resume_checkpoint"] is not None:
        config["resume_checkpoint"] = Path(
            os.path.expandvars(os.path.expanduser(config["resume_checkpoint"]))
        )
    return SimpleNamespace(**config)


def assign_run_output_directory(args):
    output_root = Path(args.output_dir)
    run_id = (
        os.environ.get("IBOT_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
    )
    if not run_id or run_id.lower() == "none":
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError(
                "Distributed training requires a shared IBOT_RUN_ID when "
                "Slurm and torchrun do not provide one"
            )
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S-%f")
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)
    args.output_root = str(output_root)
    args.run_id = run_id
    args.output_dir = str(output_root / run_id)
    args.wandb_run_id = os.environ.get("WANDB_RUN_ID") or None
    args.wandb_resume = os.environ.get("WANDB_RESUME") or None
    if args.wandb_resume and not args.wandb_run_id:
        raise ValueError("WANDB_RESUME requires WANDB_RUN_ID")
    if args.wandb_run_id and not args.wandb_resume:
        args.wandb_resume = "must"


def init_wandb(args):
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=args.output_dir,
        config=config,
        job_type="pretraining-continuation",
        id=args.wandb_run_id,
        resume=args.wandb_resume,
    )


def train_ibot(args, wandb_run=None):
    utils.init_distributed_mode(args)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 training is not supported by the allocated GPU")
    utils.fix_random_seeds(args.seed)
    args.effective_batch_size = args.batch_size_per_gpu * utils.get_world_size()
    if args.resume_checkpoint is None:
        checkpoint = read_pretrained_checkpoint(args)
        start_epoch = 0
    else:
        checkpoint = read_resume_checkpoint(args)
        start_epoch = checkpoint["epoch"]
    args.source_equivalent_final_epoch = source_equivalent_epoch(
        args.source_checkpoint_epoch, args.epochs
    )
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "effective_batch_size": args.effective_batch_size,
                "continuation_start_epoch": start_epoch,
                "additional_epochs": args.epochs,
                "source_checkpoint_epoch": args.source_checkpoint_epoch,
                "source_equivalent_final_epoch": args.source_equivalent_final_epoch,
            },
            allow_val_change=True,
        )
    print("\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())))
    cudnn.benchmark = True

    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
        args.global_crop_size,
        args.local_crop_size,
    )
    dataset = ImageFolderMask(
        args.data_path,
        transform=transform,
        patch_size=args.patch_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=args.pred_aspect_ratio,
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if len(data_loader) == 0:
        raise ValueError("The distributed data loader is empty")
    print(f"Data loaded: there are {len(dataset)} images.")

    student = create_model(
        args.arch,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
        return_all_tokens=True,
        masked_im_modeling=args.use_masked_im_modeling,
    )
    teacher = create_model(
        args.arch,
        patch_size=args.patch_size,
        return_all_tokens=True,
    )
    embed_dim = student.embed_dim

    student = utils.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu]
        )
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
        center_momentum2=args.center_momentum2,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        region_min_area=args.region_min_area,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    writer = None
    if utils.is_main_process():
        writer = SummaryWriter(logdir=os.path.join(args.output_dir, "tf_logs"))

    params_groups = utils.get_params_groups(student)
    if args.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    optimizer = torch.optim.AdamW(params_groups)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.precision == "fp16" else None

    if args.resume_checkpoint is None:
        optimizer_restored = load_continuation_state(
            checkpoint,
            student,
            teacher,
            ibot_loss,
            optimizer,
            fp16_scaler,
        )
    else:
        optimizer_restored = True
        start_epoch = load_resume_state(
            checkpoint,
            student,
            teacher,
            ibot_loss,
            optimizer,
            fp16_scaler,
        )

    source_has_scaler = "fp16_scaler" in checkpoint
    source_has_args = "args" in checkpoint
    del checkpoint

    source_epoch = args.source_checkpoint_epoch
    source_description = (
        str(source_epoch) if source_epoch is not None else "unknown"
    )
    print(
        "Continuation provenance: "
        f"source checkpoint epoch={source_description}; "
        f"continuation epoch={start_epoch}/{args.epochs}; "
        "source-equivalent epoch="
        f"{source_equivalent_epoch(source_epoch, start_epoch)}/"
        f"{args.source_equivalent_final_epoch}."
    )
    print(
        "Source checkpoint state: "
        f"optimizer={'restored' if optimizer_restored else 'absent/reconstructed'}; "
        f"FP16 scaler={'present' if source_has_scaler else 'absent'}; "
        f"original args={'present' if source_has_args else 'absent'}."
    )
    print(
        f"Precision transition: continuation uses {args.precision.upper()} from "
        f"the first iteration; GradScaler is "
        f"{'enabled' if fp16_scaler is not None else 'disabled'}."
    )
    if args.resume_checkpoint is None:
        if optimizer_restored:
            print(
                "Loaded student, teacher, projection heads, iBOT centers, and "
                "AdamW first/second moments from the continuation source."
            )
        else:
            print(
                "WARNING: the continuation source has no optimizer state; AdamW "
                "was reconstructed with zero first/second moments."
            )
    else:
        restored_state = "student, teacher, iBOT centers, and optimizer"
        if fp16_scaler is not None:
            restored_state += ", including the FP16 scaler"
        print(
            f"Restored {restored_state} from {args.resume_checkpoint} at "
            f"continuation epoch {start_epoch}/{args.epochs}."
        )

    if args.lr_schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {args.lr_schedule}")
    lr_schedule = utils.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * utils.get_world_size())
        / args.reference_batch_size,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=0,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    if start_epoch < args.epochs:
        first_schedule_iteration = start_epoch * len(data_loader)
        print(
            "Continuation schedulers ready: no LR warm-up; first optimizer-step "
            f"lr={lr_schedule[first_schedule_iteration]:.12g}; "
            f"weight_decay={wd_schedule[first_schedule_iteration]:.12g}; "
            f"teacher_momentum={momentum_schedule[first_schedule_iteration]:.12g}."
        )
    else:
        print("Continuation checkpoint is already at the configured final epoch.")
    print(
        "Teacher EMA schedule position is reconstructed because the source "
        "checkpoint does not store scheduler state or original args."
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        data_epoch = source_equivalent_epoch(args.source_checkpoint_epoch, epoch)
        if data_epoch is None:
            data_epoch = epoch
        data_loader.sampler.set_epoch(data_epoch)
        data_loader.dataset.set_epoch(data_epoch)

        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            ibot_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        completed_continuation_epoch = epoch + 1
        completed_source_epoch = source_equivalent_epoch(
            args.source_checkpoint_epoch, completed_continuation_epoch
        )
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            **continuation_provenance(
                args.source_checkpoint_epoch,
                completed_continuation_epoch,
                args.epochs,
                args.precision,
            ),
            "args": args,
            "ibot_loss": ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(
            save_dict, os.path.join(args.output_dir, "checkpoint.pth")
        )
        if (
            args.saveckp_freq
            and completed_continuation_epoch % args.saveckp_freq == 0
        ):
            utils.save_on_master(
                save_dict,
                os.path.join(
                    args.output_dir,
                    milestone_checkpoint_name(
                        args.source_checkpoint_epoch,
                        completed_continuation_epoch,
                    ),
                ),
            )
        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            "epoch": epoch,
            "continuation_epoch": completed_continuation_epoch,
            "additional_epochs": args.epochs,
            "source_checkpoint_epoch": args.source_checkpoint_epoch,
            "source_equivalent_epoch": completed_source_epoch,
        }
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as handle:
                handle.write(json.dumps(log_stats) + "\n")
            for key, value in train_stats.items():
                writer.add_scalar(key, value, epoch)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": completed_continuation_epoch,
                        "state/continuation_epoch": completed_continuation_epoch,
                        "state/source_checkpoint_epoch": args.source_checkpoint_epoch,
                        "state/source_equivalent_epoch": completed_source_epoch,
                        "state/global_step": completed_continuation_epoch
                        * len(data_loader),
                        **{
                            f"train/{key}": value
                            for key, value in train_stats.items()
                        },
                    },
                    step=epoch + 1,
                )

    if writer is not None:
        writer.close()
    total_time = time.time() - start_time
    total_time_string = str(datetime.timedelta(seconds=int(total_time)))
    if wandb_run is not None:
        wandb_run.summary["state/final_continuation_epoch"] = args.epochs
        wandb_run.summary["state/final_source_equivalent_epoch"] = (
            args.source_equivalent_final_epoch
        )
        wandb_run.summary["state/training_time_seconds"] = total_time
        wandb_run.summary["state/checkpoint"] = str(
            Path(args.output_dir) / "checkpoint.pth"
        )
    print(f"Training time {total_time_string}")


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    ibot_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    source_epoch = source_equivalent_epoch(args.source_checkpoint_epoch, epoch)
    source_final = source_equivalent_epoch(
        args.source_checkpoint_epoch, args.epochs
    )
    header = (
        f"Continuation epoch: [{epoch}/{args.epochs}]  "
        f"Source-equivalent epoch: [{source_epoch}/{source_final}]"
    )

    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [
        param_q
        for name_q, param_q in zip(names_q, params_q)
        if name_q in names_common
    ]
    params_k = [
        param_k
        for name_k, param_k in zip(names_k, params_k)
        if name_k in names_common
    ]

    for iteration, (images, _labels, masks, crop_boxes) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        schedule_iteration = len(data_loader) * epoch + iteration
        for index, parameter_group in enumerate(optimizer.param_groups):
            parameter_group["lr"] = lr_schedule[schedule_iteration]
            if index == 0:
                parameter_group["weight_decay"] = wd_schedule[schedule_iteration]

        images = [image.cuda(non_blocking=True) for image in images]
        masks = [mask.cuda(non_blocking=True) for mask in masks]
        crop_boxes = (
            crop_boxes.cuda(non_blocking=True) if ibot_loss.lambda3 != 0 else None
        )

        autocast_enabled = args.precision in {"fp16", "bf16"}
        autocast_dtype = (
            torch.bfloat16 if args.precision == "bf16" else torch.float16
        )
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            teacher_output = teacher(images[: args.global_crops_number])
            student_output = student(
                images[: args.global_crops_number],
                mask=masks[: args.global_crops_number],
            )

            student.module.backbone.masked_im_modeling = False
            student_local_cls = (
                student(images[args.global_crops_number :])[0]
                if len(images) > args.global_crops_number
                else None
            )
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(
                student_output,
                teacher_output,
                student_local_cls,
                masks,
                crop_boxes,
                epoch,
            )
            loss = all_loss.pop("loss")

        if not math.isfinite(loss.item()):
            component_values = {
                name: value.detach().float().item()
                for name, value in all_loss.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            non_finite_components = {
                name: value
                for name, value in component_values.items()
                if not math.isfinite(value)
            }
            print(
                f"Loss is {loss.item()} on rank {utils.get_rank()}, stopping "
                f"training. Non-finite components: {non_finite_components}; "
                f"all components: {component_values}",
                force=True,
            )
            sys.exit(1)

        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1])
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        accuracy = (pred1 == pred2).sum() / pred1.size(0)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(
                source_epoch if source_epoch is not None else epoch,
                student,
                args.freeze_last_layer,
            )
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(
                source_epoch if source_epoch is not None else epoch,
                student,
                args.freeze_last_layer,
            )
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            momentum = momentum_schedule[schedule_iteration]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(momentum).add_(
                    (1 - momentum) * param_q.detach().data
                )

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=accuracy)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def main():
    cli_args = parse_args()
    args = load_config(cli_args.config)
    assign_run_output_directory(args)
    output_checkpoint = Path(args.output_dir) / "checkpoint.pth"
    if (
        args.resume_checkpoint is None
        and args.initial_checkpoint.resolve() == output_checkpoint.resolve()
    ):
        raise ValueError("The output checkpoint must not overwrite the input checkpoint")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if int(os.environ.get("RANK", "0")) == 0:
        wandb_run = init_wandb(args)
    exit_code = 0
    try:
        train_ibot(args, wandb_run)
    except BaseException:
        exit_code = 1
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)
        utils.destroy_process_group()


if __name__ == "__main__":
    main()
