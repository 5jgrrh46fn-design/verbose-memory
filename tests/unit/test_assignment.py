"""Determinism and sizing of the malicious-client assignment."""

from __future__ import annotations

import pytest

from topfed.attacks.assignment import deterministic_malicious_set


def test_assignment_is_deterministic_across_calls() -> None:
    a = deterministic_malicious_set(num_clients=100, fraction=0.1, seed=42)
    b = deterministic_malicious_set(num_clients=100, fraction=0.1, seed=42)
    assert a == b


def test_assignment_changes_with_seed() -> None:
    a = deterministic_malicious_set(num_clients=100, fraction=0.1, seed=1)
    b = deterministic_malicious_set(num_clients=100, fraction=0.1, seed=2)
    assert a != b


def test_assignment_size_is_ceiling() -> None:
    # 7 * 0.10 = 0.7  -> 1 malicious client
    s = deterministic_malicious_set(num_clients=7, fraction=0.10, seed=0)
    assert len(s) == 1
    # 100 * 0.30 = 30
    s = deterministic_malicious_set(num_clients=100, fraction=0.30, seed=0)
    assert len(s) == 30


def test_assignment_zero_fraction() -> None:
    s = deterministic_malicious_set(num_clients=100, fraction=0.0, seed=0)
    assert s == frozenset()


def test_assignment_validates_inputs() -> None:
    with pytest.raises(ValueError):
        deterministic_malicious_set(num_clients=10, fraction=1.5, seed=0)
    with pytest.raises(ValueError):
        deterministic_malicious_set(num_clients=0, fraction=0.5, seed=0)
