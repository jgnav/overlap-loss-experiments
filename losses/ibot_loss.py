import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .region_loss import RegionLoss


class iBOTLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        nlcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        region_min_area=0.05,
        mim_start_epoch=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.region_loss = RegionLoss(region_min_area)

        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp,
                    teacher_temp,
                    warmup_teacher_temp_epochs,
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_temp2_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_temp2,
                        teacher_temp2,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_temp2,
                    np.linspace(
                        warmup_teacher_temp2,
                        teacher_temp2,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(
                        nepochs
                        - warmup_teacher_temp_epochs
                        - mim_start_epoch
                    )
                    * teacher_temp2,
                )
            )
        )

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_output, teacher_temp, teacher_patch_temp
    ):
        """Build detached CLS and dense patch targets using the current centers."""
        teacher_cls, teacher_patch = teacher_output
        return (
            F.softmax((teacher_cls - self.center) / teacher_temp, dim=-1),
            F.softmax((teacher_patch - self.center2) / teacher_patch_temp, dim=-1),
        )

    @staticmethod
    @torch.no_grad()
    def _distribution_diagnostics(distributions, are_probabilities):
        """Summarize patch distributions without retaining a large graph."""
        entropy_sum = None
        maximum_sum = None
        prototype_sum = None
        token_count = 0
        for distribution in distributions:
            rows = distribution.detach().flatten(0, 1)
            for chunk in rows.split(1024):
                if are_probabilities:
                    probabilities = chunk.float()
                    log_probabilities = probabilities.clamp_min(1e-12).log()
                else:
                    log_probabilities = F.log_softmax(chunk.float(), dim=-1)
                    probabilities = log_probabilities.exp()
                chunk_entropy = -(
                    probabilities * log_probabilities
                ).sum(dim=-1).sum()
                chunk_maximum = probabilities.amax(dim=-1).sum()
                chunk_prototype_sum = probabilities.sum(dim=0)
                entropy_sum = (
                    chunk_entropy
                    if entropy_sum is None
                    else entropy_sum + chunk_entropy
                )
                maximum_sum = (
                    chunk_maximum
                    if maximum_sum is None
                    else maximum_sum + chunk_maximum
                )
                prototype_sum = (
                    chunk_prototype_sum
                    if prototype_sum is None
                    else prototype_sum + chunk_prototype_sum
                )
                token_count += len(chunk)

        mean_probability = prototype_sum / token_count
        usage_entropy = -(
            mean_probability * mean_probability.clamp_min(1e-12).log()
        ).sum()
        return {
            "entropy": entropy_sum / token_count,
            "max_probability": maximum_sum / token_count,
            "effective_prototypes": usage_entropy.exp(),
        }

    @staticmethod
    @torch.no_grad()
    def _masked_overlap_diagnostics(
        patch_cross_entropies,
        student_mask,
        patch_weights,
        valid,
    ):
        """Split masked-patch CE inside/outside valid shared regions.

        Boundary patches contribute fractionally according to their covered
        area. Samples below the overlap threshold are excluded from both
        conditional diagnostics.
        """
        patch_count = patch_cross_entropies[0].shape[-1]
        inside_sum = patch_cross_entropies[0].new_zeros((), dtype=torch.float32)
        outside_sum = inside_sum.clone()
        inside_count = inside_sum.clone()
        outside_count = inside_sum.clone()
        for view, cross_entropy in enumerate(patch_cross_entropies):
            mask = student_mask[view].flatten(-2, -1).float()
            coverage = (patch_weights[:, view].float() * patch_count).clamp(0, 1)
            eligible = valid[:, None].float()
            inside_weight = mask * coverage * eligible
            outside_weight = mask * (1.0 - coverage) * eligible
            cross_entropy = cross_entropy.detach().float()
            inside_sum += (cross_entropy * inside_weight).sum()
            outside_sum += (cross_entropy * outside_weight).sum()
            inside_count += inside_weight.sum()
            outside_count += outside_weight.sum()

        return {
            "inside": inside_sum / inside_count.clamp_min(1.0),
            "outside": outside_sum / outside_count.clamp_min(1.0),
        }

    def forward(
        self,
        student_output,
        teacher_targets,
        student_local_cls,
        student_mask,
        crop_boxes,
        *,
        teacher_overlap_targets=None,
    ):
        """Use centered CLS/patch targets and optional independent overlap targets."""
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_targets

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        teacher_cls_c = teacher_cls.detach().chunk(self.ngcrops)
        teacher_patch_c = teacher_patch.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        patch_cross_entropies = []
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q]
                        * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    patch_cross_entropies.append(loss2)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(
                        dim=-1
                    ).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(
                        -teacher_cls_c[q]
                        * F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        raw_patch_loss = total_loss2 / n_loss_terms2
        total_loss2 = raw_patch_loss * self.lambda2
        region_weight = float(self.lambda3)
        zero = total_loss2.detach().float().new_zeros(())
        if self.lambda3 == 0:
            # Pure iBOT control: do not compute intersections, patch coverage,
            # aggregated distributions, or region cross-entropy.
            region_raw = zero
            total_loss3 = zero
            region_valid_ratio = zero
            region_intersection_area = zero
            patch_inside_overlap = zero
            patch_outside_overlap = zero
            region_active = zero
            objective = total_loss1 + total_loss2
        else:
            region_stats = self.region_loss(
                student_patch_c,
                teacher_patch_c,
                crop_boxes,
                teacher_overlap_targets=teacher_overlap_targets,
            )
            region_raw = region_stats["loss"]
            total_loss3 = region_raw * region_weight
            region_valid_ratio = region_stats["valid_ratio"]
            region_intersection_area = region_stats["intersection_area"]
            overlap_diagnostics = self._masked_overlap_diagnostics(
                patch_cross_entropies,
                student_mask,
                region_stats["patch_weights"],
                region_stats["valid"],
            )
            patch_inside_overlap = overlap_diagnostics["inside"]
            patch_outside_overlap = overlap_diagnostics["outside"]
            region_active = zero.new_ones(())
            objective = total_loss1 + total_loss2 + total_loss3

        student_diagnostics = self._distribution_diagnostics(
            student_patch_c,
            are_probabilities=False,
        )
        teacher_diagnostics = self._distribution_diagnostics(
            teacher_patch_c,
            are_probabilities=True,
        )
        total_loss = {
            "cls": total_loss1,
            "patch": total_loss2,
            "patch_masked": raw_patch_loss.detach().float(),
            "patch_masked_inside_overlap": patch_inside_overlap,
            "patch_masked_outside_overlap": patch_outside_overlap,
            "student_patch_entropy": student_diagnostics["entropy"],
            "teacher_patch_entropy": teacher_diagnostics["entropy"],
            "student_patch_max_probability": student_diagnostics[
                "max_probability"
            ],
            "teacher_patch_max_probability": teacher_diagnostics[
                "max_probability"
            ],
            "student_patch_effective_prototypes": student_diagnostics[
                "effective_prototypes"
            ],
            "teacher_patch_effective_prototypes": teacher_diagnostics[
                "effective_prototypes"
            ],
            "region": total_loss3,
            "region_raw": region_raw,
            "region_weight": total_loss3.new_tensor(region_weight),
            "region_active": region_active,
            "region_valid_ratio": region_valid_ratio,
            "region_intersection_area": region_intersection_area,
            "loss": objective,
        }
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        if distributed:
            dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * world_size)
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        if distributed:
            dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * world_size)
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
        )
