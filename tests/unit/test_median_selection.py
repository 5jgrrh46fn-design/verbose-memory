"""Coordinate-level median selection diagnostic (no torch)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from topfed.aggregation import coordinate_median_aggregate
from topfed.core.types import (
    ClientId,
    ClientUpdate,
    FederationConfig,
    NDArrays,
    RoundContext,
)
from topfed.federation.median_selection import (
    MedianSelectionRecorder,
    weighted_median_selection,
)
from topfed.federation.server import (
    _weighted_median_rows,
    run_federation,
    weighted_coordinate_median,
)
from topfed.integration.mechanisms import CappedScoreWeight, HardFilter, ScoreWeight

G: NDArrays = [np.zeros(4), np.zeros((2, 2))]


def _u(cid: str, value: float) -> ClientUpdate:
    return ClientUpdate(
        cid, [np.full(4, value), np.full((2, 2), value)], 1, 0.0, 1.0
    )


def _record(
    updates: list[ClientUpdate],
    malicious: frozenset[ClientId],
    weights: np.ndarray,
    rule: str = "weighted_median",
) -> MedianSelectionRecorder:
    if rule == "weighted_median":
        agg = weighted_coordinate_median(updates, weights)
    else:
        agg = coordinate_median_aggregate([u for u, w in zip(updates, weights) if w > 0])
    rec = MedianSelectionRecorder(malicious)
    rec.record(1, updates, G, agg, weights, rule)  # type: ignore[arg-type]
    return rec


def test_selection_matches_the_weighted_median_values() -> None:
    rng = np.random.default_rng(0)
    flat = rng.normal(size=(7, 50))
    w = rng.uniform(0.1, 1.0, size=7)
    sel = weighted_median_selection(flat, w, float(w.sum()))
    np.testing.assert_array_equal(
        flat[sel, np.arange(50)], _weighted_median_rows(flat, w, float(w.sum()))
    )


def test_equal_weights_select_the_lower_middle_row() -> None:
    flat = np.array([[3.0], [1.0], [4.0], [2.0]])  # sorted: 1 (row1), 2 (row3), 3, 4
    sel = weighted_median_selection(flat, np.ones(4), 4.0)
    assert sel.tolist() == [3]
    sel = weighted_median_selection(flat[:3], np.ones(3), 3.0)
    assert sel.tolist() == [0]


def test_attacker_in_the_middle_supplies_every_coordinate() -> None:
    # Honest 0 and 2, malicious 1, equal weights: the attacker's value is the
    # median although it lies inside the honest range.
    updates = [_u("h0", 0.0), _u("m0", 1.0), _u("h1", 2.0)]
    rec = _record(updates, frozenset({"m0"}), np.full(3, 1 / 3))
    (row,) = rec.round_rows
    assert row["attacker_selection_fraction"] == pytest.approx(1.0)
    assert row["weight_share"] == pytest.approx(1 / 3)
    assert row["equal_share"] == pytest.approx(1 / 3)
    assert row["num_coordinates"] == 8
    # Honest-only median (0 and 2 at renormalised equal weight) is 0, so the
    # aggregate sits 1 away on every coordinate, along the attacker's delta.
    assert row["displacement_norm"] == pytest.approx(math.sqrt(8.0))
    assert row["displacement_cosine"] == pytest.approx(1.0)
    by_client = {r["client_id"]: r for r in rec.client_rows}
    assert by_client["m0"]["selection_fraction"] == 1.0
    assert by_client["m0"]["is_malicious"] is True
    assert by_client["h0"]["selection_fraction"] == 0.0
    assert by_client["h1"]["selection_fraction"] == 0.0
    assert all(r["weight"] == pytest.approx(1 / 3) for r in rec.client_rows)


def test_attacker_at_an_extreme_supplies_nothing_and_leaves_the_median() -> None:
    updates = [_u("h0", 0.0), _u("h1", 1.0), _u("h2", 2.0), _u("m0", 3.0)]
    rec = _record(updates, frozenset({"m0"}), np.full(4, 0.25))
    (row,) = rec.round_rows
    assert row["attacker_selection_fraction"] == 0.0
    assert row["weight_share"] == pytest.approx(0.25)
    assert row["displacement_norm"] == 0.0
    assert math.isnan(row["displacement_cosine"])


def test_attacker_can_displace_the_median_by_rank_without_supplying_it() -> None:
    # Two honest rows and a low extreme attacker: the lower weighted median of
    # the three is the lower honest value, the honest-only median the lower
    # of two, which is the same row here; the upper extreme instead moves the
    # half-weight point onto the upper honest row.
    low = [_u("m0", -5.0), _u("h0", 0.0), _u("h1", 1.0)]
    (row,) = _record(low, frozenset({"m0"}), np.full(3, 1 / 3)).round_rows
    assert row["attacker_selection_fraction"] == 0.0
    assert row["displacement_norm"] == 0.0
    high = [_u("h0", 0.0), _u("h1", 1.0), _u("m0", 5.0)]
    (row,) = _record(high, frozenset({"m0"}), np.full(3, 1 / 3)).round_rows
    assert row["attacker_selection_fraction"] == 0.0
    assert row["displacement_norm"] == pytest.approx(math.sqrt(8.0))
    assert row["displacement_cosine"] == pytest.approx(1.0)


def test_round_without_attackers_has_zero_displacement() -> None:
    updates = [_u("h0", 0.0), _u("h1", 1.0), _u("h2", 5.0)]
    rec = _record(updates, frozenset({"m0"}), np.array([0.5, 0.3, 0.2]))
    (row,) = rec.round_rows
    assert row["num_malicious"] == 0
    assert row["attacker_selection_fraction"] == 0.0
    assert row["weight_share"] == 0.0
    assert row["equal_share"] == 0.0
    assert row["displacement_norm"] == 0.0
    assert math.isnan(row["displacement_cosine"])


def test_weights_decide_the_selection_and_the_share() -> None:
    # The attacker holds 0.6 of the weight: it is selected everywhere and its
    # weight share exceeds its equal share.
    updates = [_u("h0", 0.0), _u("h1", 1.0), _u("m0", 2.0)]
    (row,) = _record(updates, frozenset({"m0"}), np.array([0.2, 0.2, 0.6])).round_rows
    assert row["attacker_selection_fraction"] == 1.0
    assert row["weight_share"] == pytest.approx(0.6)
    assert row["equal_share"] == pytest.approx(1 / 3)


def test_plain_median_attributes_the_lower_middle_and_equal_weights() -> None:
    # Even cohort: numpy's median averages 1 and 2; the lower middle (the
    # attacker's 1) is credited, and the weight share is m / K.
    updates = [_u("h0", 0.0), _u("m0", 1.0), _u("h1", 2.0), _u("h2", 3.0)]
    rec = _record(updates, frozenset({"m0"}), np.ones(4), rule="median")
    (row,) = rec.round_rows
    assert row["attacker_selection_fraction"] == 1.0
    assert row["weight_share"] == pytest.approx(0.25)
    # Aggregate 1.5 against the honest-only median of {0, 2, 3} = 2.
    assert row["displacement_norm"] == pytest.approx(math.sqrt(8 * 0.25))
    assert row["displacement_cosine"] == pytest.approx(-1.0)
    assert all(r["weight"] == pytest.approx(0.25) for r in rec.client_rows)


def test_plain_median_ignores_excluded_rows() -> None:
    updates = [_u("m0", 9.0), _u("h0", 0.0), _u("h1", 1.0), _u("h2", 2.0)]
    rec = _record(updates, frozenset({"m0"}), np.array([0.0, 1.0, 1.0, 1.0]), rule="median")
    (row,) = rec.round_rows
    assert row["attacker_selection_fraction"] == 0.0
    assert row["weight_share"] == 0.0
    assert row["displacement_norm"] == 0.0
    by_client = {r["client_id"]: r for r in rec.client_rows}
    assert by_client["m0"]["weight"] == 0.0
    assert by_client["h1"]["selection_fraction"] == 1.0


def test_rejects_misaligned_weights() -> None:
    rec = MedianSelectionRecorder(frozenset())
    with pytest.raises(ValueError, match="weights shape"):
        rec.record(1, [_u("a", 0.0)], G, G, np.ones(2), "weighted_median")


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


VALUES = {f"c{i}": [float(i), float((i * 7) % 5), 1.0 - i / 8] for i in range(8)}
MALICIOUS = frozenset({"c1", "c4"})


class _Client:
    def __init__(self, cid: ClientId) -> None:
        self.client_id = cid

    def local_train(self, ctx: RoundContext, config: FederationConfig) -> ClientUpdate:
        return ClientUpdate(
            self.client_id,
            [ctx.global_parameters[0] + np.asarray(VALUES[self.client_id])],
            1 + int(self.client_id[1:]),
            0.0,
            1.0,
        )

    def evaluate(self, parameters: NDArrays, config: FederationConfig):
        return 0.0, 0, {}


class _Scorer:
    def score_round(self, round_num, updates, global_parameters):
        ids = tuple(u.client_id for u in updates)
        return ids, np.linspace(0.2, 1.0, len(ids))


def _config(**kw: Any) -> FederationConfig:
    base: dict[str, Any] = dict(
        num_clients=8, clients_per_round=4, num_rounds=3, local_epochs=1,
        local_batch_size=1, local_learning_rate=0.1, local_momentum=0.0, seed=0,
    )
    base.update(kw)
    return FederationConfig(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scorer=_Scorer(), integrator=CappedScoreWeight(), defended_base="weighted_median"),
        dict(scorer=_Scorer(), integrator=HardFilter(threshold=0.5), defended_base="median"),
        dict(aggregator=coordinate_median_aggregate),
    ],
)
def test_server_records_every_round_without_changing_the_aggregate(kwargs) -> None:
    plain = run_federation(_config(), _Client, [np.zeros(3)], verbose=False, **kwargs)
    rec = MedianSelectionRecorder(MALICIOUS)
    traced = run_federation(
        _config(), _Client, [np.zeros(3)], verbose=False, median_selection=rec, **kwargs
    )
    for a, b in zip(plain, traced, strict=True):
        np.testing.assert_array_equal(a.aggregated_parameters[0], b.aggregated_parameters[0])
    assert [r["round"] for r in rec.round_rows] == [1, 2, 3]
    assert len(rec.client_rows) == 3 * 4
    for outcome, row in zip(traced, rec.round_rows, strict=True):
        ids = {u.client_id for u in outcome.client_updates}
        assert row["num_malicious"] == len(ids & MALICIOUS)
        assert row["num_coordinates"] == 3
        is_mal = np.array([u.client_id in MALICIOUS for u in outcome.client_updates])
        if outcome.weights is None:  # the plain median has no per-client weights
            assert row["weight_share"] == pytest.approx(row["equal_share"])
        else:
            assert row["weight_share"] == pytest.approx(float(outcome.weights[is_mal].sum()))
        rows = [r for r in rec.client_rows if r["round"] == outcome.round_num]
        assert sum(r["selection_fraction"] for r in rows) == pytest.approx(1.0)
        assert row["attacker_selection_fraction"] == pytest.approx(
            sum(r["selection_fraction"] for r in rows if r["is_malicious"])
        )


def test_server_rejects_the_recorder_on_a_mean_rule() -> None:
    with pytest.raises(ValueError, match="median rule"):
        run_federation(
            _config(), _Client, [np.zeros(3)], verbose=False,
            median_selection=MedianSelectionRecorder(MALICIOUS),
        )
    with pytest.raises(ValueError, match="median rule"):
        run_federation(
            _config(), _Client, [np.zeros(3)], verbose=False,
            scorer=_Scorer(), integrator=ScoreWeight(),
            median_selection=MedianSelectionRecorder(MALICIOUS),
        )
