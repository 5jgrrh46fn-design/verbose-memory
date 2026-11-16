"""Per-client DataLoader construction respects the partition."""

from __future__ import annotations

import numpy as np
import torch

from topfed.data.loaders import build_client_loaders


def test_loaders_carry_only_their_partition_indices() -> None:
    train_x = torch.arange(40 * 3 * 32 * 32, dtype=torch.float32).view(40, 3, 32, 32)
    train_y = torch.arange(40, dtype=torch.long) % 10
    partition = [
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([4, 5, 6, 7, 8, 9], dtype=np.int64),
    ] + [np.array([], dtype=np.int64)] * 8
    loaders = build_client_loaders(
        train_x,
        train_y,
        partition,
        batch_size=2,
        augment=False,
        shuffle=False,
    )

    assert set(loaders.keys()) == {f"c{i}" for i in range(10)}
    assert sum(1 for _ in loaders["c0"]) == 2  # 4 samples / batch=2
    assert sum(1 for _ in loaders["c1"]) == 3  # 6 samples / batch=2
    # Empty shard yields zero batches.
    assert sum(1 for _ in loaders["c2"]) == 0


def test_augmentation_produces_diverse_outputs() -> None:
    # A natural image (gradient on each axis) should yield distinct crops
    # under random crop + flip across repeated draws.
    base = torch.zeros(1, 3, 32, 32)
    grid = torch.linspace(0.0, 1.0, 32)
    base[0, 0] = grid.view(32, 1).expand(32, 32)
    base[0, 1] = grid.view(1, 32).expand(32, 32)
    train_y = torch.tensor([0])
    partition = [np.array([0], dtype=np.int64)]
    loaders = build_client_loaders(
        base,
        train_y,
        partition,
        batch_size=1,
        augment=True,
        shuffle=False,
    )

    torch.manual_seed(0)
    fingerprints = set()
    for _ in range(20):
        for x, _ in loaders["c0"]:
            fingerprints.add(float(x.sum().item()))
    assert len(fingerprints) > 1


def test_augmentation_off_returns_input_verbatim() -> None:
    train_x = torch.randn(2, 3, 32, 32)
    train_y = torch.tensor([0, 1])
    partition = [np.array([0, 1], dtype=np.int64)]
    loaders = build_client_loaders(
        train_x,
        train_y,
        partition,
        batch_size=2,
        augment=False,
        shuffle=False,
    )
    for x, _ in loaders["c0"]:
        torch.testing.assert_close(x, train_x)
