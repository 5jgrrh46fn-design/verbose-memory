"""MESAS: metric extraction, significance cascade, and pruning on synthetic updates."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.aggregation import (
    MESAS,
    DeltaAggregator,
    mesas_metrics,
    mesas_prune,
    series_is_significant,
    smaller_cluster,
)
from topfed.aggregation.mesas import METRIC_NAMES
from topfed.core.types import ClientUpdate, NDArrays

D = 60


def _globals() -> NDArrays:
    rng = np.random.default_rng(0)
    return [rng.normal(size=(8, 5)), rng.normal(size=(20,))]


def _update(cid: str, g: NDArrays, delta: np.ndarray) -> ClientUpdate:
    return ClientUpdate(cid, [g[0] + delta[:40].reshape(8, 5), g[1] + delta[40:]], 10, 0.0, 1.0)


def _flat(params: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(p).ravel() for p in params])


def _cohort(rng: np.random.Generator, n: int, scale: float = 0.05) -> np.ndarray:
    return rng.normal(scale=scale, size=(n, D))


def test_metrics_shape_and_whole_model_values() -> None:
    g = _globals()
    delta = np.zeros(D)
    delta[:3] = [0.5, -0.25, 1.0]
    u = _update("a", g, delta)
    m = mesas_metrics([u], g)
    assert m.shape == (1, 6 * (1 + len(g)))
    whole = dict(zip(METRIC_NAMES, m[0, :6], strict=True))
    assert whole["eucl"] == pytest.approx(np.linalg.norm(delta))
    assert whole["count"] == 2  # two parameters increased
    assert whole["min"] == pytest.approx(0.25)
    assert whole["max"] == pytest.approx(1.0)
    assert whole["var"] == pytest.approx(np.var(_flat(u.parameters)))
    assert 0.0 <= whole["cos"] <= 2.0
    # Second array is unchanged: its block has eucl 0, count 0, min 0, max 0.
    block2 = dict(zip(METRIC_NAMES, m[0, 12:18], strict=True))
    assert block2["eucl"] == 0.0 and block2["count"] == 0 and block2["max"] == 0.0


def test_series_not_significant_when_constant() -> None:
    sig, p = series_is_significant(np.full(10, 3.0), alpha=0.03)
    assert not sig and p == {}


def test_three_sigma_cannot_fire_below_eleven_values() -> None:
    # max z-score of n values is (n - 1) / sqrt(n); at n = 10 that is 2.85.
    v = np.r_[np.ones(9), 1e6]
    sig, p = series_is_significant(v, alpha=1e-9)  # alpha so small only sigma could fire
    assert p["sigma"] == 0.0
    assert not sig
    v20 = np.r_[1.0 + np.random.default_rng(0).normal(scale=0.05, size=19), 25.0]
    sig, p = series_is_significant(v20, alpha=1e-9)
    assert p["sigma"] == 1.0 and sig


def test_balanced_series_tests_run_and_are_not_significant() -> None:
    rng = np.random.default_rng(1)
    sig, p = series_is_significant(rng.normal(size=10), alpha=0.0001)
    assert set(p) == {"t", "levene", "ks", "sigma"}
    assert not sig


def test_center_mean_isolates_two_large_outliers_median_does_not() -> None:
    v = np.r_[1.0 + np.random.default_rng(2).normal(scale=0.05, size=8), 40.0, 42.0]
    sig_med, _ = series_is_significant(v, alpha=0.03, center="median")
    sig_mean, p = series_is_significant(v, alpha=0.03, center="mean")
    assert not sig_med
    assert sig_mean and p["t"] < 0.03
    with pytest.raises(ValueError):
        series_is_significant(v, alpha=0.03, center="mode")


def test_smaller_cluster_picks_minority_and_ties_prune_nothing() -> None:
    v = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01, 25.0, 26.0])
    assert sorted(smaller_cluster(v)) == [8, 9]
    assert smaller_cluster(np.array([0.0, 0.0, 10.0, 10.0])) == []
    assert smaller_cluster(np.full(5, 2.0)) == []


def _scenario(n_benign: int, n_att: int, scale: float, seed: int):
    g = _globals()
    rng = np.random.default_rng(seed)
    deltas = np.vstack([_cohort(rng, n_benign), scale * _cohort(rng, n_att)])
    updates = [_update(f"c{i}", g, deltas[i]) for i in range(n_benign + n_att)]
    return g, deltas, updates


def test_prune_ten_clients_median_center_keeps_everyone() -> None:
    # Two 40x attackers among ten: the median split puts both into a
    # five-element list whose variance hides them, and 3-sigma cannot fire.
    g, _, updates = _scenario(8, 2, 40.0, 802)
    retained, trace = mesas_prune(mesas_metrics(updates, g), alpha=0.03, center="median")
    assert retained == list(range(10))
    assert trace == []


def test_prune_ten_clients_mean_center_removes_attackers() -> None:
    g, _, updates = _scenario(8, 2, 40.0, 802)
    retained, trace = mesas_prune(mesas_metrics(updates, g), alpha=0.03, center="mean")
    assert 8 not in retained and 9 not in retained
    assert len(retained) >= 2
    assert trace and min(e["pass"] for e in trace) == 1


def test_prune_twenty_clients_removes_attackers_with_either_center() -> None:
    for center in ("median", "mean"):
        g, _, updates = _scenario(12, 8, 40.0, 1208)
        retained, _ = mesas_prune(mesas_metrics(updates, g), alpha=0.03, center=center)
        assert set(retained) <= set(range(12)), center
        assert len(retained) >= 2


def test_aggregate_averages_retained_and_reports_weights() -> None:
    g, deltas, updates = _scenario(8, 2, 40.0, 802)
    agg = MESAS(alpha=0.03, center="mean")
    assert isinstance(agg, DeltaAggregator)
    out = agg.aggregate(1, updates, g)
    assert agg.last_weights is not None
    retained = [i for i in range(10) if agg.last_weights[i] > 0]
    assert 8 not in retained and 9 not in retained
    expected = _flat(g) + deltas[retained].mean(axis=0)
    np.testing.assert_allclose(_flat(out), expected, atol=1e-9)
    np.testing.assert_allclose(agg.last_weights[retained], 1 / len(retained))
    rows = [r for r in agg.records if r["round"] == 1]
    assert len(rows) == 10
    assert all({f"m_{n}" for n in METRIC_NAMES} <= set(r) for r in rows)
    assert rows[8]["retained"] is False and rows[8]["pruned_pass"] >= 1
    assert all(r["pruned_pass"] == 0 for r in rows if r["retained"])


def test_rejects_bad_alpha_and_empty_round() -> None:
    with pytest.raises(ValueError):
        MESAS(alpha=0.0)
    with pytest.raises(ValueError):
        MESAS(alpha=1.0)
    with pytest.raises(ValueError):
        MESAS(center="mode")
    with pytest.raises(ValueError):
        MESAS().aggregate(1, [], _globals())
