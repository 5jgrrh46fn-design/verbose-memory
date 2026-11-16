"""IID partitioning and the kind-dispatching partition builder."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.data.partition import build_partition, iid_partition


def test_iid_partition_covers_every_index_once_in_equal_shards() -> None:
    parts = iid_partition(num_samples=103, num_clients=10, seed=0)
    assert len(parts) == 10
    sizes = sorted(len(p) for p in parts)
    assert sizes[0] >= 10 and sizes[-1] <= 11
    assert sorted(np.concatenate(parts).tolist()) == list(range(103))


def test_iid_partition_is_deterministic_and_seed_sensitive() -> None:
    a = iid_partition(50, 5, seed=1)
    b = iid_partition(50, 5, seed=1)
    c = iid_partition(50, 5, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert any(not np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_iid_partition_is_label_balanced_in_expectation() -> None:
    labels = np.repeat(np.arange(10), 100)
    parts = iid_partition(len(labels), 10, seed=0)
    for shard in parts:
        counts = np.bincount(labels[shard], minlength=10)
        assert counts.min() >= 3  # no class starved under IID


def test_build_partition_dispatches_on_kind() -> None:
    labels = np.repeat(np.arange(4), 25)
    writers = np.repeat(np.arange(5), 20)
    d = build_partition("dirichlet", labels=labels, num_clients=5, seed=0, alpha=0.5)
    i = build_partition("iid", labels=labels, num_clients=5, seed=0)
    n = build_partition("natural", labels=labels, num_clients=5, seed=0, writer_ids=writers)
    for parts in (d, i, n):
        assert len(parts) == 5
        assert sorted(np.concatenate(parts).tolist()) == list(range(100))
    # Natural keeps one writer per client; dirichlet pools and re-splits.
    assert all(len(set(writers[s].tolist())) == 1 for s in n)
    assert any(len(set(writers[s].tolist())) > 1 for s in d if len(s))


def test_pooled_femnist_style_dirichlet_ignores_writers() -> None:
    # Re-splitting a naturally partitioned pool by Dirichlet is a pure
    # function of (labels, alpha, seed): writer ids play no part.
    labels = np.repeat(np.arange(6), 30)
    a = build_partition("dirichlet", labels=labels, num_clients=6, seed=3, alpha=1.0,
                        writer_ids=np.arange(180))
    b = build_partition("dirichlet", labels=labels, num_clients=6, seed=3, alpha=1.0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_build_partition_validates_requirements() -> None:
    labels = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError):
        build_partition("dirichlet", labels=labels, num_clients=2, seed=0)
    with pytest.raises(ValueError):
        build_partition("natural", labels=labels, num_clients=2, seed=0)
    with pytest.raises(ValueError):
        build_partition("shards", labels=labels, num_clients=2, seed=0)
