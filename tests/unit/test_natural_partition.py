"""Unit tests for natural (per-writer) partitioning."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.data.partition import natural_partition


def test_groups_indices_by_writer() -> None:
    writers = np.array([5, 5, 7, 7, 7, 2])
    parts = natural_partition(writers, seed=0)
    assert len(parts) == 3  # three distinct writers
    assert sorted(np.concatenate(parts).tolist()) == [0, 1, 2, 3, 4, 5]
    for shard in parts:
        assert len(set(writers[shard].tolist())) == 1  # one writer per client


def test_num_clients_caps_writer_count() -> None:
    writers = np.array([1, 2, 3, 4])
    assert len(natural_partition(writers, seed=1, num_clients=2)) == 2


def test_is_deterministic_in_seed() -> None:
    writers = np.array([1, 1, 2, 3, 3, 3])
    a = natural_partition(writers, seed=0)
    b = natural_partition(writers, seed=0)
    assert [s.tolist() for s in a] == [s.tolist() for s in b]


def test_cap_exceeding_writers_raises() -> None:
    with pytest.raises(ValueError):
        natural_partition(np.array([1, 2]), seed=0, num_clients=5)
