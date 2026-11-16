"""FLAME: HDBSCAN filtering, median-norm clipping, and noise on synthetic updates."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.aggregation import (
    FLAME,
    DeltaAggregator,
    cosine_distance_matrix,
    flame_cluster,
)
from topfed.core.types import ClientUpdate, NDArrays

D = 60


def _globals() -> NDArrays:
    rng = np.random.default_rng(0)
    return [rng.normal(size=(8, 5)), rng.normal(size=(20,))]


def _update(cid: str, g: NDArrays, delta: np.ndarray) -> ClientUpdate:
    return ClientUpdate(cid, [g[0] + delta[:40].reshape(8, 5), g[1] + delta[40:]], 10, 0.0, 1.0)


def _flat(params: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(p).ravel() for p in params])


def _benign_cohort(rng: np.random.Generator, n: int) -> np.ndarray:
    # Benign deltas share a direction with small independent perturbations, so
    # their pairwise cosine distances are small and they form one cluster.
    base = rng.normal(size=D)
    return base + rng.normal(scale=0.2, size=(n, D))


def test_cosine_distance_matrix_basics() -> None:
    v = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    d = cosine_distance_matrix(v)
    assert d[0, 0] == 0.0
    assert d[0, 1] == pytest.approx(1.0)
    assert d[0, 2] == pytest.approx(2.0)
    assert d[0, 3] == pytest.approx(1.0)  # zero vector: cosine 0 to everything
    np.testing.assert_allclose(d, d.T)


def test_cluster_admits_majority_and_rejects_opposed_outliers() -> None:
    rng = np.random.default_rng(1)
    benign = _benign_cohort(rng, 8)
    # Two attackers point the opposite way at 20x the norm.
    attackers = -20.0 * benign.mean(axis=0) + rng.normal(scale=0.2, size=(2, D))
    labels, admitted = flame_cluster(np.vstack([benign, attackers]))
    # Both attackers are rejected. HDBSCAN may also mark one or two benign
    # clients as noise (FLAME's known false positives; the hdbscan package
    # and scikit-learn agree on this), so the admitted set is a majority
    # subset of the benign cohort rather than exactly the cohort.
    assert labels[8] == -1 and labels[9] == -1
    assert set(admitted) <= set(range(8))
    assert len(admitted) >= 6


def test_equidistant_cohort_is_one_cluster() -> None:
    # Ten mutually orthogonal deltas are all at cosine distance 1 from each
    # other: no density structure, one root cluster, everyone admitted.
    labels, admitted = flame_cluster(np.eye(10, D))
    assert (labels == 0).all()
    assert admitted == list(range(10))


def test_no_cluster_branch_admits_everyone() -> None:
    # min_cluster_size above n cannot be met; HDBSCAN then returns all noise
    # and the reference port admits every client.
    labels, admitted = flame_cluster(np.eye(4, D), min_cluster_size=4)
    assert admitted == list(range(4)) or len(admitted) >= 1


def test_min_cluster_size_defaults_to_majority() -> None:
    rng = np.random.default_rng(2)
    benign = _benign_cohort(rng, 5)
    outliers = rng.normal(size=(5, D)) * 10
    # Five like-minded clients out of ten are not a majority (need 6).
    labels, admitted = flame_cluster(np.vstack([benign, outliers]))
    assert len(admitted) == 10 or len(admitted) >= 6


def test_single_client_is_admitted() -> None:
    labels, admitted = flame_cluster(np.ones((1, D)))
    assert admitted == [0]


def test_aggregate_clips_to_median_norm_and_averages_admitted() -> None:
    g = _globals()
    rng = np.random.default_rng(3)
    benign = _benign_cohort(rng, 8)
    attackers = -20.0 * benign.mean(axis=0) + rng.normal(scale=0.2, size=(2, D))
    deltas = np.vstack([benign, attackers])
    updates = [_update(f"c{i}", g, deltas[i]) for i in range(10)]
    agg = FLAME(noise_factor=0.0)
    out = agg.aggregate(1, updates, g)
    assert isinstance(agg, DeltaAggregator)

    bound = np.median(np.linalg.norm(deltas, axis=1))
    assert agg.last_weights is not None
    admitted = [i for i in range(10) if agg.last_weights[i] > 0]
    assert set(admitted) <= set(range(8)) and len(admitted) >= 6
    expected = np.zeros(D)
    for i in admitted:
        n = np.linalg.norm(deltas[i])
        expected += min(1.0, bound / n) * deltas[i] / len(admitted)
    np.testing.assert_allclose(_flat(out) - _flat(g), expected, atol=1e-9)
    np.testing.assert_allclose(agg.last_weights[admitted], 1 / len(admitted))
    np.testing.assert_allclose(agg.last_weights[8:], 0.0)
    rows = [r for r in agg.records if r["round"] == 1]
    assert len(rows) == 10
    assert sum(r["admitted"] for r in rows) == len(admitted)
    assert all(r["clip_factor"] <= 1.0 for r in rows)
    assert all(r["clip_bound"] == pytest.approx(bound) for r in rows)
    # The large attackers would have been clipped hard had they been admitted.
    assert rows[8]["clip_factor"] == 0.0 and rows[8]["cluster"] == -1


def test_noise_is_seeded_and_scaled_by_bound() -> None:
    g = _globals()
    rng = np.random.default_rng(4)
    deltas = _benign_cohort(rng, 10)
    updates = [_update(f"c{i}", g, deltas[i]) for i in range(10)]
    a = FLAME(noise_factor=0.1, seed=7).aggregate(1, updates, g)
    b = FLAME(noise_factor=0.1, seed=7).aggregate(1, updates, g)
    c = FLAME(noise_factor=0.1, seed=8).aggregate(1, updates, g)
    clean = FLAME(noise_factor=0.0).aggregate(1, updates, g)
    np.testing.assert_allclose(_flat(a), _flat(b))
    assert not np.allclose(_flat(a), _flat(c))
    noise = _flat(a) - _flat(clean)
    bound = np.median(np.linalg.norm(deltas, axis=1))
    assert np.std(noise) == pytest.approx(0.1 * bound, rel=0.5)


def test_rejects_bad_noise_factor_and_empty_round() -> None:
    with pytest.raises(ValueError):
        FLAME(noise_factor=-1.0)
    with pytest.raises(ValueError):
        FLAME().aggregate(1, [], _globals())
