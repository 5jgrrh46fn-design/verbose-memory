"""Uniform and forced co-selection client sampling."""

from __future__ import annotations

import random

import pytest

from topfed.federation.sampling import forced_co_selection_sample, uniform_sample

IDS = tuple(f"c{i}" for i in range(20))
MALICIOUS = frozenset({"c1", "c5", "c9", "c13", "c17"})


def test_uniform_sample_matches_random_sample_on_the_same_rng() -> None:
    # The server previously called ``rng.sample`` directly; the helper must
    # draw the identical cohort so existing seeds remain reproducible.
    a = uniform_sample(random.Random(7), IDS, 5)
    b = tuple(random.Random(7).sample(IDS, 5))
    assert a == b
    assert len(set(a)) == 5


@pytest.mark.parametrize("m", [0, 1, 3, 5])
def test_forced_sample_has_exactly_m_malicious(m: int) -> None:
    rng = random.Random(0)
    for _ in range(50):
        cohort = forced_co_selection_sample(rng, IDS, MALICIOUS, k=8, m=m)
        assert len(cohort) == 8
        assert len(set(cohort)) == 8
        assert sum(c in MALICIOUS for c in cohort) == m


def test_forced_sample_is_deterministic_in_rng() -> None:
    a = [
        forced_co_selection_sample(random.Random(3), IDS, MALICIOUS, k=6, m=2)
        for _ in range(3)
    ]
    assert a[0] == a[1] == a[2]
    b = forced_co_selection_sample(random.Random(4), IDS, MALICIOUS, k=6, m=2)
    assert b != a[0]


def test_forced_sample_shuffles_roles_across_positions() -> None:
    # Malicious members must not always occupy the leading slots.
    rng = random.Random(1)
    first_slot_malicious = [
        forced_co_selection_sample(rng, IDS, MALICIOUS, k=6, m=2)[0] in MALICIOUS
        for _ in range(100)
    ]
    assert 0 < sum(first_slot_malicious) < 100


def test_forced_sample_rejects_impossible_counts() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError):
        forced_co_selection_sample(rng, IDS, MALICIOUS, k=6, m=7)  # m > k
    with pytest.raises(ValueError):
        forced_co_selection_sample(rng, IDS, MALICIOUS, k=6, m=6)  # only 5 exist
    with pytest.raises(ValueError):
        forced_co_selection_sample(rng, IDS, MALICIOUS, k=18, m=1)  # 17 benign > 15
    with pytest.raises(ValueError):
        forced_co_selection_sample(rng, IDS, MALICIOUS, k=6, m=-1)
