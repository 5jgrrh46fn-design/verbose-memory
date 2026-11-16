"""Augmentation keeps a constant padded border constant."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from topfed.attacks.data.backdoor import trigger_mask
from topfed.data.loaders import build_client_loaders, pad_images


def test_augmentation_never_moves_content_into_the_border() -> None:
    torch.manual_seed(0)
    base = torch.randn(1, 3, 32, 32)
    padded = pad_images(base, 4)
    loaders = build_client_loaders(
        padded,
        torch.tensor([0]),
        [np.array([0], dtype=np.int64)],
        batch_size=1,
        augment=True,
        shuffle=False,
        pad_border=4,
    )
    ring = trigger_mask(40, 40, trigger_size=4, trigger_position="border")
    seen = set()
    for _ in range(30):
        for x, _ in loaders["c0"]:
            assert x.shape == (1, 3, 40, 40)
            assert torch.all(x[:, :, ring] == 0.0)
            seen.add(float(x.sum().item()))
    # Interior is still being augmented (different crops/flips).
    assert len(seen) > 1


def test_augmentation_without_pad_border_is_unchanged() -> None:
    # pad_border=0 must route through the original 32x32 crop-and-flip.
    torch.manual_seed(0)
    base = torch.randn(1, 3, 32, 32)
    loaders = build_client_loaders(
        base, torch.tensor([0]), [np.array([0], dtype=np.int64)],
        batch_size=1, augment=True, shuffle=False,
    )
    x, _ = next(iter(loaders["c0"]))
    assert x.shape == (1, 3, 32, 32)


def test_negative_pad_border_rejected() -> None:
    with pytest.raises(ValueError):
        build_client_loaders(
            torch.zeros(1, 3, 32, 32), torch.tensor([0]),
            [np.array([0], dtype=np.int64)], batch_size=1, pad_border=-1,
        )
