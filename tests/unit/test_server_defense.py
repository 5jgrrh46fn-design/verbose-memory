"""Unit tests for the server's defended aggregation path (no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.core.types import (
    ClientId,
    ClientUpdate,
    FederationConfig,
    NDArrays,
    RoundContext,
)
from topfed.aggregation import coordinate_median_aggregate
from topfed.integration.mechanisms import HardFilter, ScoreWeight
from topfed.federation.server import (
    _weighted_sample,
    fltrust_aggregate,
    run_federation,
    weighted_aggregate,
    weighted_coordinate_median,
)


def test_fltrust_trusts_aligned_and_drops_opposed() -> None:
    # Server delta = +1 (norm 2). An aligned client (cos=1) is kept and rescaled
    # to the server's norm; an opposed client (cos<0) is ReLU-zeroed and dropped.
    g: NDArrays = [np.zeros((2, 2))]
    server = _update("s", 1.0).parameters  # server params -> delta +1 everywhere
    aligned = _update("a", 3.0)  # same direction, larger magnitude
    opposed = _update("b", -5.0)  # opposite direction -> trust 0
    out = fltrust_aggregate([aligned, opposed], g, server)
    assert out[0] == pytest.approx(np.full((2, 2), 1.0))


def test_fltrust_keeps_global_when_nobody_trusted() -> None:
    g: NDArrays = [np.full((2, 2), 7.0)]
    server = _update("s", 8.0).parameters  # server delta +1
    opposed = _update("b", 6.0)  # delta -1 -> cos<0 -> trust 0
    out = fltrust_aggregate([opposed], g, server)
    assert out[0] == pytest.approx(np.full((2, 2), 7.0))  # unchanged


def test_weighted_coordinate_median_equal_weights_is_median() -> None:
    # Odd client count, equal weights -> the middle value (plain median).
    updates = [_update("a", 1.0), _update("b", 5.0), _update("c", 100.0)]
    agg = weighted_coordinate_median(updates, np.array([0.3, 0.3, 0.3]))
    assert agg[0] == pytest.approx(np.full((2, 2), 5.0))


def test_weighted_coordinate_median_bounds_over_weighted_minority() -> None:
    # An extreme client (value 1000) with the *largest* weight but still a
    # minority of total weight cannot pull the weighted median to its value.
    updates = [_update("a", 1.0), _update("b", 2.0), _update("c", 3.0), _update("d", 1000.0)]
    agg = weighted_coordinate_median(updates, np.array([0.2, 0.2, 0.2, 0.3]))
    assert agg[0] == pytest.approx(np.full((2, 2), 3.0))  # bounded, not 1000


def test_weighted_coordinate_median_majority_weight_controls() -> None:
    # A client holding a weight majority does control the median (expected: the
    # cap only bounds non-majority clients).
    updates = [_update("a", 1.0), _update("b", 2.0), _update("c", 9.0)]
    agg = weighted_coordinate_median(updates, np.array([0.1, 0.1, 0.9]))
    assert agg[0] == pytest.approx(np.full((2, 2), 9.0))


def test_weighted_coordinate_median_downweights_low_trust() -> None:
    # A low-weight client at an extreme value is pulled away from.
    updates = [_update("m", 0.0), _update("a", 5.0), _update("b", 5.0), _update("c", 5.0)]
    agg = weighted_coordinate_median(updates, np.array([0.01, 0.3, 0.3, 0.3]))
    assert agg[0] == pytest.approx(np.full((2, 2), 5.0))


def _update(cid: str, value: float) -> ClientUpdate:
    return ClientUpdate(
        client_id=cid,
        parameters=[np.full((2, 2), value, dtype=np.float64)],
        num_examples=10,
        train_loss=0.1,
        train_accuracy=0.9,
    )


def test_weighted_aggregate_excludes_zero_weight() -> None:
    updates = [_update("a", 1.0), _update("b", 2.0), _update("c", 9.0)]
    agg = weighted_aggregate(updates, np.array([0.5, 0.5, 0.0]))
    assert agg[0] == pytest.approx(np.full((2, 2), 1.5))


def test_weighted_aggregate_rejects_zero_total() -> None:
    with pytest.raises(ValueError):
        weighted_aggregate([_update("a", 1.0)], np.array([0.0]))


def test_weighted_sample_favours_high_probability_clients() -> None:
    ids = tuple(f"c{i}" for i in range(6))
    probs = {"c0": 5.0, "c1": 5.0, "c2": 0.01, "c3": 0.01, "c4": 0.01, "c5": 0.01}
    rng = np.random.default_rng(0)
    selected = _weighted_sample(ids, 2, probs, rng)
    assert set(selected) == {"c0", "c1"}


class _FakeClient:
    def __init__(self, client_id: ClientId, value: float) -> None:
        self.client_id = client_id
        self._value = value

    def local_train(
        self, ctx: RoundContext, config: FederationConfig
    ) -> ClientUpdate:
        return _update(self.client_id, self._value)

    def evaluate(
        self, parameters: NDArrays, config: FederationConfig
    ) -> tuple[float, int, dict[str, float]]:
        return 0.0, 0, {}


class _FakeScorer:
    def __init__(self, score_map: dict[str, float]) -> None:
        self._score_map = score_map

    def score_round(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> tuple[tuple[ClientId, ...], np.ndarray]:
        ids = tuple(u.client_id for u in updates)
        scores = np.array([self._score_map[c] for c in ids], dtype=np.float64)
        return ids, scores


def _config() -> FederationConfig:
    return FederationConfig(
        num_clients=4,
        clients_per_round=4,
        num_rounds=1,
        local_epochs=1,
        local_batch_size=8,
        local_learning_rate=0.1,
        local_momentum=0.0,
        seed=0,
    )


def test_defended_loop_hard_filter_drops_low_scorers() -> None:
    values = {"c0": 1.0, "c1": 2.0, "c2": 9.0, "c3": 9.0}
    factory = lambda cid: _FakeClient(cid, values[cid])  # noqa: E731
    scorer = _FakeScorer({"c0": 0.9, "c1": 0.9, "c2": 0.1, "c3": 0.1})
    outcomes = run_federation(
        _config(),
        factory,
        initial_parameters=[np.zeros((2, 2))],
        evaluator=None,
        scorer=scorer,
        integrator=HardFilter(threshold=0.5),
        verbose=False,
    )
    # Only c0 (1.0) and c1 (2.0) survive the filter -> mean 1.5.
    assert outcomes[-1].aggregated_parameters[0] == pytest.approx(
        np.full((2, 2), 1.5)
    )


def test_defended_loop_score_weight_blends_by_trust() -> None:
    values = {"c0": 0.0, "c1": 4.0, "c2": 0.0, "c3": 4.0}
    factory = lambda cid: _FakeClient(cid, values[cid])  # noqa: E731
    scorer = _FakeScorer({"c0": 0.0, "c1": 0.75, "c2": 0.0, "c3": 0.25})
    outcomes = run_federation(
        _config(),
        factory,
        initial_parameters=[np.zeros((2, 2))],
        evaluator=None,
        scorer=scorer,
        integrator=ScoreWeight(),
        verbose=False,
    )
    # Weights normalise to c1=0.75, c3=0.25 (both value 4.0) -> 4.0.
    assert outcomes[-1].aggregated_parameters[0] == pytest.approx(
        np.full((2, 2), 4.0)
    )


def test_robust_aggregator_path_uses_the_rule() -> None:
    values = {"c0": 1.0, "c1": 2.0, "c2": 9.0, "c3": 9.0}
    factory = lambda cid: _FakeClient(cid, values[cid])  # noqa: E731
    outcomes = run_federation(
        _config(),
        factory,
        initial_parameters=[np.zeros((2, 2))],
        evaluator=None,
        aggregator=coordinate_median_aggregate,
        verbose=False,
    )
    # Coordinate-wise median of {1, 2, 9, 9} = 5.5 (mean of the two middles).
    assert outcomes[-1].aggregated_parameters[0] == pytest.approx(
        np.full((2, 2), 5.5)
    )


def test_defended_median_base_takes_median_of_survivors() -> None:
    # HardFilter keeps c0,c1,c2 (>0.5) and drops c3; the median base combines the
    # survivors by coordinate median (= 2.0), not their mean (which would be 4.0).
    values = {"c0": 1.0, "c1": 2.0, "c2": 9.0, "c3": 100.0}
    factory = lambda cid: _FakeClient(cid, values[cid])  # noqa: E731
    scorer = _FakeScorer({"c0": 0.9, "c1": 0.9, "c2": 0.9, "c3": 0.1})
    outcomes = run_federation(
        _config(),
        factory,
        initial_parameters=[np.zeros((2, 2))],
        evaluator=None,
        scorer=scorer,
        integrator=HardFilter(threshold=0.5),
        defended_base="median",
        verbose=False,
    )
    assert outcomes[-1].aggregated_parameters[0] == pytest.approx(
        np.full((2, 2), 2.0)
    )


def test_defended_median_base_bounds_a_high_trust_outlier() -> None:
    # Even with every client included (all high-trust), a single extreme update
    # cannot move the coordinate median: median{1,2,3,1000} = 2.5, not the
    # mean (~251.5). This is the property that makes the base backfire-proof.
    values = {"c0": 1000.0, "c1": 1.0, "c2": 2.0, "c3": 3.0}
    factory = lambda cid: _FakeClient(cid, values[cid])  # noqa: E731
    scorer = _FakeScorer({c: 0.9 for c in values})
    outcomes = run_federation(
        _config(),
        factory,
        initial_parameters=[np.zeros((2, 2))],
        evaluator=None,
        scorer=scorer,
        integrator=HardFilter(threshold=0.5),
        defended_base="median",
        verbose=False,
    )
    assert outcomes[-1].aggregated_parameters[0] == pytest.approx(
        np.full((2, 2), 2.5)
    )


def test_defended_base_rejects_invalid_value() -> None:
    factory = lambda cid: _FakeClient(cid, 1.0)  # noqa: E731
    with pytest.raises(ValueError):
        run_federation(
            _config(),
            factory,
            initial_parameters=[np.zeros((2, 2))],
            scorer=_FakeScorer({f"c{i}": 0.5 for i in range(4)}),
            integrator=HardFilter(),
            defended_base="trimmed",
            verbose=False,
        )


def test_aggregator_and_integrator_are_mutually_exclusive() -> None:
    factory = lambda cid: _FakeClient(cid, 1.0)  # noqa: E731
    with pytest.raises(ValueError):
        run_federation(
            _config(),
            factory,
            initial_parameters=[np.zeros((2, 2))],
            scorer=_FakeScorer({f"c{i}": 0.5 for i in range(4)}),
            integrator=HardFilter(),
            aggregator=coordinate_median_aggregate,
            verbose=False,
        )


def test_fltrust_median_cap_stops_a_single_trusted_client_from_deciding() -> None:
    from topfed.federation.server import fltrust_median_aggregate

    # Server delta +1 everywhere (norm 2). One client is perfectly aligned
    # (trust 1); four others have cosine about 0.30 (trust 0.30 each), so the
    # aligned client holds 45% of the raw trust. After FLTrust's rescaling to
    # the server norm the aligned delta is (1, 1, 1, 1) and each other delta
    # is a rotation of (1.951, -0.254, -0.254, -0.254). Uncapped, the weighted
    # median crosses half the mass at the aligned client on every coordinate;
    # capped at an equal share (0.2 before renormalisation) it crosses at the
    # three others sitting at -0.254.
    g: NDArrays = [np.zeros((4,))]
    server = [np.array([1.0, 1.0, 1.0, 1.0])]
    aligned = ClientUpdate("a", [np.array([5.0, 5.0, 5.0, 5.0])], 10, 0.0, 1.0)
    base = np.array([1.0, -0.13, -0.13, -0.13])
    others = [
        ClientUpdate(f"b{i}", [np.roll(base, i)], 10, 0.0, 1.0) for i in range(4)
    ]
    updates = [aligned] + others
    uncapped = fltrust_median_aggregate(updates, g, server)
    capped = fltrust_median_aggregate(updates, g, server, cap=1.0)
    assert uncapped[0] == pytest.approx(np.ones(4), abs=1e-6)
    expected = -0.13 * 2.0 / float(np.linalg.norm(base))
    assert capped[0] == pytest.approx(np.full(4, expected), abs=1e-6)
    with pytest.raises(ValueError):
        fltrust_median_aggregate(updates, g, server, cap=0.0)


def test_weighted_sample_fills_uniformly_when_too_few_clients_have_a_positive_bias() -> None:
    ids = tuple(f"c{i}" for i in range(8))
    # Only two clients keep a positive bias; a cohort of four takes both and
    # two of the zero-bias clients.
    probs = {c: 0.0 for c in ids}
    probs["c2"], probs["c5"] = 0.7, 0.1
    counts = dict.fromkeys(ids, 0)
    for seed in range(200):
        selected = _weighted_sample(ids, 4, probs, np.random.default_rng(seed))
        assert len(set(selected)) == 4
        assert {"c2", "c5"} <= set(selected)
        for c in selected:
            counts[c] += 1
    assert counts["c2"] == counts["c5"] == 200
    # The fill is uniform over the six others: each about 200 * 2 / 6 times.
    others = [counts[c] for c in ids if c not in ("c2", "c5")]
    assert min(others) > 40 and max(others) < 95


def test_weighted_sample_is_unchanged_when_enough_clients_have_a_positive_bias() -> None:
    ids = tuple(f"c{i}" for i in range(6))
    probs = {"c0": 0.5, "c1": 0.2, "c2": 0.2, "c3": 0.1, "c4": 0.0, "c5": 0.0}
    selected = _weighted_sample(ids, 4, probs, np.random.default_rng(3))
    p = np.array([0.5, 0.2, 0.2, 0.1, 0.0, 0.0])
    expected = np.random.default_rng(3).choice(6, size=4, replace=False, p=p)
    assert selected == tuple(ids[i] for i in expected)
