"""Out-of-distribution image sets for server-side indicator tasks.

A detector that plants an auxiliary task into the global model needs images
that are not from the training distribution. Following the pairing used by
BackdoorIndicator's reference implementation, CIFAR-10 is served CIFAR-100
and FEMNIST is served CIFAR-10. The images are brought into the host
dataset's input format (channel count, spatial size, normalisation, constant
border) so the model consumes them exactly like training data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from topfed.data.loaders import pad_images


def _raw_images(source: str, root: str | Path, download: bool) -> np.ndarray:
    """``(N, H, W, 3)`` uint8 training images of ``source``."""
    if source == "cifar100":
        return datasets.CIFAR100(str(root), train=True, download=download).data
    if source == "cifar10":
        return datasets.CIFAR10(str(root), train=True, download=download).data
    raise ValueError(f"unknown OOD source {source!r}; expected 'cifar100' or 'cifar10'")


def load_ood_images(
    source: str,
    root: str | Path,
    num_samples: int,
    seed: int,
    *,
    grayscale: bool,
    image_size: int,
    mean: tuple[float, ...] | float,
    std: tuple[float, ...] | float,
    pad_border: int = 0,
    download: bool = True,
) -> torch.Tensor:
    """Sample ``num_samples`` images of ``source`` in the host dataset's format.

    ``grayscale`` averages the three channels (for single-channel hosts),
    ``image_size`` resizes bilinearly when it differs from the source's 32,
    ``mean`` / ``std`` normalise per channel (a float applies to every
    channel), and ``pad_border`` adds the host's constant border after
    normalisation. Sampling is without replacement from a generator seeded
    with ``seed``.
    """
    data = _raw_images(source, root, download)
    if num_samples > data.shape[0]:
        raise ValueError(f"asked for {num_samples} samples; {source} has {data.shape[0]}")
    idx = np.random.default_rng(seed).choice(data.shape[0], size=num_samples, replace=False)
    x = torch.from_numpy(data[idx]).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    if grayscale:
        x = x.mean(dim=1, keepdim=True)
    if image_size != x.shape[-1]:
        x = torch.nn.functional.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    c = x.shape[1]
    mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
    if mean_t.numel() not in (1, c) or std_t.numel() not in (1, c):
        raise ValueError(f"mean/std must have 1 or {c} entries")
    x = (x - mean_t.view(1, -1, 1, 1)) / std_t.view(1, -1, 1, 1)
    return pad_images(x, pad_border)
