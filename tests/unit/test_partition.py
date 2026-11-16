"""Dirichlet partitioning correctness and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.data.partition import dirichlet_partition


def _balanced_labels(num_classes: int, per_class: int) -> np.ndarray:
    return np.repeat(np.arange(num_classes), per_class)


def test_partition_covers_all_indices_exactly_once() -> None:
    labels = _balanced_labels(num_classes=10, per_class=50)
    shards = dirichlet_partition(labels, num_clients=20, alpha=0.5, seed=0)
    assigned = sorted(int(i) for s in shards for i in s)
    assert assigned == list(range(len(labels)))


def test_partition_is_deterministic() -> None:
    labels = _balanced_labels(10, 50)
    a = dirichlet_partition(labels, num_clients=20, alpha=0.5, seed=42)
    b = dirichlet_partition(labels, num_clients=20, alpha=0.5, seed=42)
    for sa, sb in zip(a, b, strict=True):
        np.testing.assert_array_equal(sa, sb)


def test_partition_changes_with_seed() -> None:
    labels = _balanced_labels(10, 50)
    a = dirichlet_partition(labels, num_clients=20, alpha=0.5, seed=1)
    b = dirichlet_partition(labels, num_clients=20, alpha=0.5, seed=2)
    # At least one shard should differ.
    assert any(not np.array_equal(sa, sb) for sa, sb in zip(a, b, strict=True))


def test_low_alpha_is_more_skewed_than_high_alpha() -> None:
    labels = _balanced_labels(10, 100)
    skewed = dirichlet_partition(labels, num_clients=10, alpha=0.05, seed=0)
    uniform = dirichlet_partition(labels, num_clients=10, alpha=100.0, seed=0)

    def shard_class_entropy(shard: np.ndarray) -> float:
        if len(shard) == 0:
            return 0.0
        counts = np.bincount(labels[shard], minlength=10).astype(float)
        p = counts / counts.sum()
        nz = p[p > 0]
        return float(-np.sum(nz * np.log(nz)))

    avg_skewed = np.mean([shard_class_entropy(s) for s in skewed if len(s) > 0])
    avg_uniform = np.mean([shard_class_entropy(s) for s in uniform if len(s) > 0])
    # alpha=100 -> nearly uniform per-class distribution per shard; alpha=0.05
    # collapses each shard onto a few classes. Entropy is much higher for the
    # uniform case.
    assert avg_uniform > avg_skewed + 0.5


def test_partition_validates_inputs() -> None:
    labels = _balanced_labels(10, 10)
    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=0, alpha=0.5, seed=0)
    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=10, alpha=0.0, seed=0)
    with pytest.raises(ValueError):
        dirichlet_partition(labels.reshape(10, 10), num_clients=10, alpha=0.5, seed=0)
