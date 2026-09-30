"""Register-only adaptation without changing DDP's registered parameter set."""

import numpy as np


REGISTER_PARAMETER = "backbone.register_tokens"


def clear_non_register_gradients(student):
    """Freeze pretrained weights, AdamW moments, and weight decay during warmup.

    Retain the full backward/DDP reduction, then discard frozen gradients before
    clipping or GradScaler's overflow check. This supports the existing multiple
    crop forwards without changing DDP or optimizer parameter groups on resume.
    """
    parameters = dict(student.named_parameters())
    if REGISTER_PARAMETER not in parameters:
        raise ValueError("Register warmup requires backbone.register_tokens")
    for name, parameter in parameters.items():
        if name != REGISTER_PARAMETER:
            parameter.grad = None


def teacher_ema_pairs(student, teacher, register_only=False):
    """Match EMA parameters by name and keep pretrained teacher weights frozen."""
    teacher_parameters = dict(teacher.named_parameters())
    return [
        (parameter, teacher_parameters[name])
        for name, parameter in student.named_parameters()
        if name in teacher_parameters and (not register_only or name == REGISTER_PARAMETER)
    ]


def prepend_register_warmup(schedule, warmup_epochs, iterations_per_epoch):
    """Hold the first value during adaptation, then run the full normal schedule."""
    if warmup_epochs == 0:
        return schedule
    return np.concatenate((
        np.full(warmup_epochs * iterations_per_epoch, schedule[0], dtype=schedule.dtype),
        schedule,
    ))


def completed_normal_training_epochs(completed_epochs, warmup_epochs):
    return max(0, completed_epochs - warmup_epochs)
