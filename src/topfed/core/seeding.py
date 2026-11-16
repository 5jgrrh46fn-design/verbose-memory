"""Deterministic seeding and GPU determinism controls for reproducible runs.

``set_all_seeds`` is the canonical seeding entry point. ``configure_determinism``
additionally pins the GPU numerics so that one (config, seed) pair reproduces
bit-identical results on the same machine and software stack:

- cuDNN and all other torch ops are restricted to deterministic kernels
  (an op without one raises instead of silently diverging);
- cuDNN autotuning is off, so kernel choice cannot vary between runs.

TF32 precision is left at the torch defaults (off for matmul, on for cuDNN
convolution on GPUs that support it): TF32 kernels are themselves
deterministic, and the defaults keep the arithmetic identical to runs made
without these controls. The effective TF32 flags are still reported.

None of these settings consume random numbers, so enabling them leaves every
seeded stream (sampling, partitioning, augmentation, poisoning) unchanged.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# cuBLAS needs a fixed workspace configuration for deterministic matmuls. It is
# read when the cuBLAS handle is created, so it must be in the environment
# before the first CUDA matmul.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(deterministic: bool) -> dict[str, object]:
    """Apply (or lift) the GPU determinism controls and report the effective state.

    Call before any CUDA work in the process. With ``deterministic=False`` the
    torch defaults are restored (nondeterministic kernels, autotuning off).
    The returned mapping is meant for the run manifest.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    return {
        "deterministic": deterministic,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def derive_round_seed(run_seed: int, round_num: int) -> int:
    return (run_seed * 1_000_003 + round_num) & 0x7FFFFFFF
