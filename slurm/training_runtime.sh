#!/usr/bin/env bash

# Shared helpers for Slurm training launchers. This file is sourced, not run.

configure_training_compatibility() {
    # Keep Slurm's GPU visibility and allocation intact. Avoid inherited
    # Python/library overrides from the submission shell.
    unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
    export PYTHONNOUSERSITE=1
    # Leave NCCL transport selection unchanged by default. Set this only for
    # a targeted communication workaround on a known-good CUDA allocation.
    if [[ "${IBOT_NCCL_COMPATIBILITY:-0}" == "1" ]]; then
        export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
        echo "NCCL compatibility: P2P and InfiniBand disabled"
    fi
}

prepare_training_config() {
    local python_path="$1"
    local base_config="$2"
    local output_config="$3"
    local run_dir="$4"
    "$python_path" slurm/prepare_training_config.py "$base_config" "$output_config" \
        --run-dir "$run_dir"
}
