"""Capture trace and per-round aggregation weights on the server."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from topfed.aggregation import FoolsGold, coordinate_median_aggregate
from topfed.core.types import (
    ClientId,
    ClientUpdate,
    FederationConfig,
    NDArrays,
    RoundContext,
)
from topfed.federation.instrumentation import TRACE_KEYS, capture_trace
from topfed.federation.server import fltrust_trust, run_federation
from topfed.integration.mechanisms import ScoreWeight
from topfed.runtime.manifest import write_metrics


def _u(cid: str, values: list[float], n: int = 1) -> ClientUpdate:
    return ClientUpdate(cid, [np.asarray(values, dtype=np.float64)], n, 0.0, 1.0)


G: NDArrays = [np.zeros(3)]


def test_share_uses_the_weights_and_is_nan_without_them() -> None:
    updates = [_u("m0", [1, 1, 1]), _u("h0", [0, 0, 0]), _u("h1", [0, 0, 0])]
    mal = frozenset({"m0"})
    agg: NDArrays = [np.zeros(3)]
    t = capture_trace(updates, G, agg, mal, np.array([0.5, 0.25, 0.25]))
    assert t["trace_malicious_share"] == pytest.approx(0.5)
    assert math.isnan(capture_trace(updates, G, agg, mal, None)["trace_malicious_share"])
    assert math.isnan(capture_trace(updates, G, agg, mal, np.zeros(3))["trace_malicious_share"])


def test_captured_fraction_counts_bloc_set_coordinates_only() -> None:
    # Coordinate 0: aggregate equals the bloc's value, far outside the honest IQR.
    # Coordinate 1: aggregate at the honest median. Coordinate 2: outside the
    # honest IQR but also outside the bloc's range.
    mal = frozenset({"m0", "m1"})
    updates = [
        _u("m0", [5.0, 5.0, 5.0]),
        _u("m1", [5.0, 6.0, 5.0]),
        _u("h0", [0.0, 0.0, 0.0]),
        _u("h1", [1.0, 1.0, 1.0]),
        _u("h2", [2.0, 2.0, 2.0]),
    ]
    agg: NDArrays = [np.array([5.0, 1.0, -3.0])]
    t = capture_trace(updates, G, agg, mal, None)
    assert t["trace_captured_fraction"] == pytest.approx(1 / 3)


def test_projection_is_along_the_malicious_mean_direction() -> None:
    mal = frozenset({"m0", "m1"})
    updates = [_u("m0", [2, 0, 0]), _u("m1", [4, 0, 0]), _u("h0", [0, 1, 0]), _u("h1", [1, -1, 0])]
    agg: NDArrays = [np.array([1.5, 7.0, 0.0])]
    t = capture_trace(updates, G, agg, mal, None)
    assert t["trace_malicious_norm"] == pytest.approx(3.0)
    assert t["trace_projection"] == pytest.approx(1.5)
    assert t["trace_honest_projection"] == pytest.approx(0.5)


def test_round_without_attackers_reports_nan_except_the_share() -> None:
    updates = [_u("h0", [1, 0, 0]), _u("h1", [0, 1, 0])]
    t = capture_trace(updates, G, [np.zeros(3)], frozenset({"m0"}), np.array([0.5, 0.5]))
    assert t["trace_malicious_share"] == 0.0
    assert all(math.isnan(t[k]) for k in TRACE_KEYS if k != "trace_malicious_share")


def test_all_malicious_round_has_no_honest_quantities() -> None:
    updates = [_u("m0", [1, 0, 0]), _u("m1", [1, 0, 0])]
    t = capture_trace(updates, G, [np.array([1.0, 0, 0])], frozenset({"m0", "m1"}), None)
    assert math.isnan(t["trace_captured_fraction"])
    assert math.isnan(t["trace_honest_projection"])
    assert t["trace_projection"] == pytest.approx(1.0)


def test_fltrust_trust_is_relu_cosine_to_the_server_delta() -> None:
    updates = [_u("a", [1, 0, 0]), _u("b", [-1, 0, 0]), _u("c", [1, 1, 0])]
    trust = fltrust_trust(updates, G, [np.array([2.0, 0.0, 0.0])])
    np.testing.assert_allclose(trust, [1.0, 0.0, 1 / np.sqrt(2)])
    np.testing.assert_array_equal(fltrust_trust(updates, G, G), 0.0)


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


VALUES = {f"c{i}": [float(i), float(i % 3), 1.0] for i in range(8)}
MALICIOUS = frozenset({"c1", "c4"})


class _Client:
    def __init__(self, cid: ClientId) -> None:
        self.client_id = cid

    def local_train(self, ctx: RoundContext, config: FederationConfig) -> ClientUpdate:
        n = 1 + int(self.client_id[1:])
        return ClientUpdate(self.client_id, [ctx.global_parameters[0] + np.asarray(VALUES[self.client_id])], n, 0.0, 1.0)

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


def test_fedavg_weights_are_sample_shares() -> None:
    outcomes = run_federation(_config(), _Client, [np.zeros(3)], verbose=False)
    for o in outcomes:
        n = np.array([u.num_examples for u in o.client_updates], dtype=float)
        assert o.weights is not None
        np.testing.assert_allclose(o.weights, n / n.sum())
        assert o.trace == {}


def test_defense_weights_are_the_realised_integrator_weights() -> None:
    outcomes = run_federation(
        _config(), _Client, [np.zeros(3)], scorer=_Scorer(), integrator=ScoreWeight(),
        verbose=False,
    )
    s = np.linspace(0.2, 1.0, 4)
    for o in outcomes:
        assert o.weights is not None
        np.testing.assert_allclose(o.weights, s / s.sum())


def test_stateless_rule_has_no_weights_and_delta_aggregator_reports_its_own() -> None:
    median = run_federation(_config(), _Client, [np.zeros(3)], aggregator=coordinate_median_aggregate, verbose=False)
    assert all(o.weights is None for o in median)
    fg = FoolsGold()
    outcomes = run_federation(_config(), _Client, [np.zeros(3)], aggregator=fg, verbose=False)
    for o in outcomes:
        assert o.weights is not None and o.weights.sum() == pytest.approx(1.0)
    assert len(fg.records) == 3 * 4
    assert {r["round"] for r in fg.records} == {1, 2, 3}


def test_krum_selection_reaches_the_baseline_ledger() -> None:
    from experiments.run import _baseline_ledger
    from topfed.aggregation.robust import Krum

    rule = Krum(num_byzantine=1)
    outcomes = run_federation(_config(), _Client, [np.zeros(3)], aggregator=rule, verbose=False)
    malicious = frozenset({outcomes[0].client_updates[0].client_id})
    rows = _baseline_ledger(outcomes, rule, malicious)
    assert len(rows) == 3 * 4
    for o in outcomes:
        mine = [r for r in rows if r["round"] == o.round_num]
        assert [r["client_id"] for r in mine] == [u.client_id for u in o.client_updates]
        assert sorted(r["weight"] for r in mine) == [0.0, 0.0, 0.0, 1.0]
        picked = next(r for r in mine if r["weight"] == 1.0)
        assert picked["is_malicious"] == (picked["client_id"] in malicious)


def test_trace_requires_the_malicious_set() -> None:
    with pytest.raises(ValueError, match="trace_capture"):
        run_federation(_config(trace_capture=True), _Client, [np.zeros(3)], verbose=False)


def test_trace_lands_in_every_round_and_in_metrics_json(tmp_path) -> None:  # noqa: ANN001
    outcomes = run_federation(
        _config(trace_capture=True, force_malicious_per_round=2), _Client, [np.zeros(3)],
        malicious_clients=MALICIOUS, verbose=False,
    )
    for o in outcomes:
        assert set(o.trace) == set(TRACE_KEYS)
        n = np.array([u.num_examples for u in o.client_updates], dtype=float)
        mal = np.array([u.client_id in MALICIOUS for u in o.client_updates])
        assert o.trace["trace_malicious_share"] == pytest.approx(n[mal].sum() / n.sum())
    import json

    rows = json.loads(write_metrics(tmp_path, outcomes).read_text())
    assert all("trace_captured_fraction" in r for r in rows)


def test_rooted_rule_weights_come_from_trust_only_when_tracing() -> None:
    def server_fn(ctx, cfg):  # noqa: ANN001
        return [ctx.global_parameters[0] + np.array([1.0, 0.0, 0.0])]

    plain = run_federation(_config(), _Client, [np.zeros(3)], server_update_fn=server_fn, verbose=False)
    assert all(o.weights is None for o in plain)
    traced = run_federation(
        _config(trace_capture=True), _Client, [np.zeros(3)], server_update_fn=server_fn,
        malicious_clients=MALICIOUS, verbose=False,
    )
    for o in traced:
        assert o.weights is not None
        assert o.weights.sum() == pytest.approx(1.0)
    # Round 1 from the zero model: trust is the cosine to e1.
    u = list(traced[0].client_updates)
    expected = fltrust_trust(u, [np.zeros(3)], [np.array([1.0, 0.0, 0.0])])
    np.testing.assert_allclose(traced[0].weights, expected / expected.sum())
