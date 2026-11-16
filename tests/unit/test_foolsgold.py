"""FoolsGold: history-similarity down-weighting on synthetic updates."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.aggregation import DeltaAggregator, FoolsGold, foolsgold_weights
from topfed.core.types import ClientUpdate, NDArrays

D = 60


def _update(cid: str, g: NDArrays, delta: np.ndarray) -> ClientUpdate:
    return ClientUpdate(cid, [g[0] + delta[:40].reshape(8, 5), g[1] + delta[40:]], 10, 0.0, 1.0)


def _globals() -> NDArrays:
    rng = np.random.default_rng(0)
    return [rng.normal(size=(8, 5)), rng.normal(size=(20,))]


def _flat_delta(params: NDArrays, g: NDArrays) -> np.ndarray:
    return np.concatenate([(p - gg).ravel() for p, gg in zip(params, g, strict=True)])


def test_mutually_orthogonal_clients_all_get_full_weight() -> None:
    alpha, max_cs = foolsgold_weights(np.eye(5))
    np.testing.assert_allclose(alpha, 1.0)
    np.testing.assert_allclose(max_cs, 0.0)


def test_identical_clients_get_zero_and_honest_keep_full_weight() -> None:
    rng = np.random.default_rng(1)
    honest = rng.normal(size=(6, D))
    sybil = rng.normal(size=D)
    histories = np.vstack([honest, np.tile(sybil, (4, 1))])
    alpha, max_cs = foolsgold_weights(histories)
    np.testing.assert_allclose(alpha[6:], 0.0)
    np.testing.assert_allclose(max_cs[6:], 1.0)
    assert (alpha[:6] > 0.9).all()


def test_similar_but_not_identical_bloc_is_down_weighted() -> None:
    rng = np.random.default_rng(2)
    honest = rng.normal(size=(6, D))
    base = rng.normal(size=D)
    bloc = base + rng.normal(scale=0.1, size=(4, D))
    alpha, _ = foolsgold_weights(np.vstack([honest, bloc]))
    assert alpha[6:].max() < alpha[:6].min()
    assert alpha[6:].max() < 0.1


def test_pardoning_spares_an_honest_client_that_resembles_one_sybil() -> None:
    # Client 0 is moderately similar to the Sybil pair (1, 2), which are
    # identical. Pardoning scales cs[0, j] by max_0 / max_j = 0.6 / 1.
    s = np.zeros(D)
    s[0] = 1.0
    other = np.zeros(D)
    other[1] = 1.0
    honest_like = 0.6 * s + 0.8 * other
    histories = np.vstack([honest_like, s, s, np.eye(D)[5], np.eye(D)[6]])
    alpha, max_cs = foolsgold_weights(histories)
    assert max_cs[0] == pytest.approx(0.6)
    assert alpha[1] == alpha[2] == 0.0
    # 1 - 0.6 * 0.6 = 0.64 before rescale/logit, so client 0 keeps positive weight.
    assert alpha[0] > 0.5


def test_all_identical_gives_all_zero() -> None:
    alpha, _ = foolsgold_weights(np.tile(np.arange(1.0, 5.0), (3, 1)))
    np.testing.assert_array_equal(alpha, 0.0)


def test_single_client_keeps_full_weight() -> None:
    alpha, _ = foolsgold_weights(np.ones((1, 4)))
    np.testing.assert_array_equal(alpha, [1.0])


def test_logit_is_scaled_by_kappa_and_clipped() -> None:
    rng = np.random.default_rng(3)
    h = rng.normal(size=(6, D))
    a1, _ = foolsgold_weights(h, kappa=1.0)
    a2, _ = foolsgold_weights(h, kappa=0.1)
    assert ((a1 >= 0) & (a1 <= 1)).all()
    assert a2.sum() <= a1.sum()


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        foolsgold_weights(np.zeros(3))
    with pytest.raises(ValueError):
        FoolsGold(kappa=0.0)


def test_aggregate_down_weights_identical_clients_and_follows_the_honest_mean() -> None:
    g = _globals()
    rng = np.random.default_rng(4)
    honest = [rng.normal(size=D) for _ in range(6)]
    poison = 5.0 * rng.normal(size=D)
    updates = [_update(f"h{i}", g, d) for i, d in enumerate(honest)]
    updates += [_update(f"m{i}", g, poison) for i in range(4)]
    fg = FoolsGold()
    assert isinstance(fg, DeltaAggregator)
    out = fg.aggregate(1, updates, g)
    assert fg.last_weights is not None
    np.testing.assert_allclose(fg.last_weights[6:], 0.0)
    assert fg.last_weights.sum() == pytest.approx(1.0)
    expected = np.sum([w * d for w, d in zip(fg.last_weights[:6], honest, strict=True)], axis=0)
    np.testing.assert_allclose(_flat_delta(out, g), expected)
    # The Sybil direction is not in the aggregate.
    step = _flat_delta(out, g)
    assert abs(np.dot(step, poison)) / (np.linalg.norm(step) * np.linalg.norm(poison)) < 0.5


def test_all_dissimilar_round_is_the_plain_mean_of_deltas() -> None:
    g = _globals()
    deltas = [np.eye(D)[i] * (i + 1) for i in range(5)]
    fg = FoolsGold()
    out = fg.aggregate(1, [_update(f"c{i}", g, d) for i, d in enumerate(deltas)], g)
    np.testing.assert_allclose(_flat_delta(out, g), np.mean(deltas, axis=0))


def test_history_accumulates_across_rounds_and_catches_a_bloc_that_diverges_per_round() -> None:
    # Each round the bloc's deltas share only a common poison component on top
    # of per-round noise; the accumulated histories converge toward it.
    g = _globals()
    rng = np.random.default_rng(5)
    poison = rng.normal(size=D)
    fg = FoolsGold()
    for r in range(1, 21):
        updates = [_update(f"h{i}", g, rng.normal(size=D)) for i in range(6)]
        updates += [_update(f"m{i}", g, poison + rng.normal(scale=1.0, size=D)) for i in range(4)]
        fg.aggregate(r, updates, g)
    first = [row for row in fg.records if row["round"] == 1]
    last = [row for row in fg.records if row["round"] == 20]
    mal_first = np.mean([r["alpha"] for r in first if r["client_id"].startswith("m")])
    mal_last = np.mean([r["alpha"] for r in last if r["client_id"].startswith("m")])
    honest_last = np.mean([r["alpha"] for r in last if r["client_id"].startswith("h")])
    assert mal_last < mal_first
    assert mal_last < 0.1 < honest_last


def test_history_is_per_client_and_only_grows_when_selected() -> None:
    g = _globals()
    fg = FoolsGold()
    d = np.eye(D)
    fg.aggregate(1, [_update("a", g, d[0]), _update("b", g, d[1])], g)
    fg.aggregate(2, [_update("a", g, d[2]), _update("c", g, d[3])], g)
    np.testing.assert_allclose(fg._history["a"], d[0] + d[2])
    np.testing.assert_allclose(fg._history["b"], d[1])
    np.testing.assert_allclose(fg._history["c"], d[3])


def test_all_zero_alpha_keeps_the_global_model() -> None:
    g = _globals()
    v = np.arange(D, dtype=float)
    fg = FoolsGold()
    out = fg.aggregate(1, [_update(f"c{i}", g, v) for i in range(3)], g)
    for a, b in zip(out, g, strict=True):
        np.testing.assert_array_equal(a, b)
    assert fg.last_weights is not None and fg.last_weights.sum() == 0.0


def test_records_one_row_per_client_per_round() -> None:
    g = _globals()
    fg = FoolsGold()
    fg.aggregate(3, [_update(f"c{i}", g, np.eye(D)[i]) for i in range(4)], g)
    assert [r["round"] for r in fg.records] == [3] * 4
    assert {"client_id", "max_similarity", "alpha", "weight"} <= set(fg.records[0])
