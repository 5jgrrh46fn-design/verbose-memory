"""Server-rooted pieces: the sixth TOPSIS criterion (H2), FLTrust on a
weighted-median base (H3), and the server loop's evidence plumbing (no torch)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from topfed.core.geometry import server_cosines
from topfed.core.types import (
    ClientId,
    ClientUpdate,
    FederationConfig,
    NDArrays,
    RoundContext,
    RoundEvidence,
)
from topfed.federation.scoring_bridge import ServerRootedMetricScorer
from topfed.federation.server import (
    fltrust_aggregate,
    fltrust_median_aggregate,
    run_federation,
)
from topfed.integration.gated import ServerGatedCappedScoreWeight
from topfed.integration.mechanisms import CappedScoreWeight, ScoreWeight
from topfed.scoring.criteria import (
    METRIC_DIRECTION,
    ROOTED_TRUST_METRICS,
    SERVER_COSINE,
    TRUST_METRICS,
    benefit_mask,
)
from topfed.scoring.topsis import TopsisScorer, topsis_closeness
from topfed.scoring.weighting import critic_weights


def _update(cid: str, delta: list[float], base: float = 0.0) -> ClientUpdate:
    return ClientUpdate(cid, [base + np.array(delta, dtype=np.float64)], 1, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Server cosine
# ----------------------------------------------------------------------------


def test_server_cosines_match_definition_and_handle_degenerate_cases() -> None:
    g: NDArrays = [np.full(3, 2.0)]
    server: NDArrays = [np.array([3.0, 2.0, 2.0])]  # delta e1
    updates = [
        _update("a", [4.0, 0.0, 0.0], 2.0),
        _update("b", [-1.0, 0.0, 0.0], 2.0),
        _update("c", [1.0, 1.0, 0.0], 2.0),
        _update("z", [0.0, 0.0, 0.0], 2.0),
    ]
    assert server_cosines(updates, g, server) == pytest.approx([1.0, -1.0, 2**-0.5, 0.0])
    assert server_cosines(updates, g, g) == pytest.approx([0.0] * 4)


# ----------------------------------------------------------------------------
# H2: sixth TOPSIS criterion
# ----------------------------------------------------------------------------


def test_server_cosine_is_a_benefit_criterion() -> None:
    assert METRIC_DIRECTION[SERVER_COSINE] == +1
    assert ROOTED_TRUST_METRICS == TRUST_METRICS + (SERVER_COSINE,)
    mask = benefit_mask(ROOTED_TRUST_METRICS)
    assert mask.tolist() == [True, False, False, False, False, True]


def test_higher_server_cosine_scores_higher_all_else_equal() -> None:
    # Identical on M1-M5; only the server cosine differs.
    n = 4
    matrix = np.tile(np.array([0.5, 1.0, 0.1, 0.2, 0.3, 0.0]), (n, 1))
    matrix[:, 5] = [0.9, 0.1, -0.4, 0.5]
    scores = topsis_closeness(
        matrix, np.full(6, 1.0 / 6), benefit_mask(ROOTED_TRUST_METRICS)
    )
    assert np.argsort(scores).tolist() == [2, 1, 3, 0]


def test_critic_gives_the_server_cosine_column_weight_when_it_varies() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0.1, 1.0, size=(10, 6))
    w = critic_weights(matrix)
    assert w.shape == (6,) and w[5] > 0.0 and w.sum() == pytest.approx(1.0)


class _StubExtractor:
    """Stands in for MetricsExtractor: identical M1-M5 for every client."""

    def extract(self, round_num, updates, global_parameters, malicious):  # noqa: ANN001
        return [
            {"client_id": u.client_id, "m1_cosine_to_mean": 0.5, "m2_norm_ratio": 1.0,
             "m3_temporal_cv": 0.1, "m4_val_loss_delta": 0.2, "m5_max_layer_deviation": 0.3}
            for u in updates
        ]


def test_rooted_scorer_ranks_by_server_cosine_and_logs_it() -> None:
    g: NDArrays = [np.zeros(3)]
    server: NDArrays = [np.array([1.0, 0.0, 0.0])]
    updates = (
        _update("c0", [1.0, 0.0, 0.0]),     # cos 1
        _update("c1", [-1.0, 0.2, 0.0]),    # cos < 0
        _update("c2", [0.5, 0.5, 0.0]),     # cos 0.71
    )
    scorer = ServerRootedMetricScorer(_StubExtractor(), TopsisScorer(weighting=critic_weights))  # type: ignore[arg-type]
    scorer.observe(RoundEvidence(1, updates, g, server_parameters=server))
    ids, scores = scorer.score_round(1, updates, g)
    assert ids == ("c0", "c1", "c2")
    assert scores[0] > scores[2] > scores[1]
    assert scorer.last_batch is not None
    assert scorer.last_batch.metric_names == ROOTED_TRUST_METRICS
    assert [r[SERVER_COSINE] for r in scorer.records] == pytest.approx(
        server_cosines(updates, g, server).tolist()
    )


def test_rooted_scorer_requires_the_rounds_server_update() -> None:
    g: NDArrays = [np.zeros(3)]
    updates = (_update("c0", [1.0, 0.0, 0.0]),)
    scorer = ServerRootedMetricScorer(_StubExtractor(), TopsisScorer())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="server update"):
        scorer.score_round(1, updates, g)
    scorer.observe(RoundEvidence(1, updates, g))
    with pytest.raises(RuntimeError, match="server update"):
        scorer.score_round(1, updates, g)


# ----------------------------------------------------------------------------
# H3: FLTrust with a weighted-median base
# ----------------------------------------------------------------------------


def test_fltrust_median_with_equal_trust_is_the_median_of_rescaled_deltas() -> None:
    g: NDArrays = [np.zeros(3)]
    server: NDArrays = [np.array([2.0, 2.0, 2.0])]  # norm 2*sqrt(3), cos 1 for +x
    updates = [
        _update("a", [1.0, 1.0, 1.0]),
        _update("b", [3.0, 3.0, 3.0]),
        _update("c", [0.5, 0.5, 0.5]),
    ]
    # All aligned (trust 1) and rescaled to the server norm -> all equal 2.
    out = fltrust_median_aggregate(updates, g, server)
    assert out[0] == pytest.approx([2.0, 2.0, 2.0])


def test_fltrust_median_bounds_a_trusted_minority_that_the_mean_follows() -> None:
    # Server delta e1+e2 direction. Three honest clients along it; one client
    # is trusted (cos > 0) but concentrates its rescaled mass on coordinate 3.
    g: NDArrays = [np.zeros(3)]
    server: NDArrays = [np.array([1.0, 1.0, 0.0])]
    honest = [_update(f"h{i}", [1.0 + 0.1 * i, 1.0 - 0.1 * i, 0.0]) for i in range(3)]
    attacker = _update("m", [0.5, 0.5, 3.0])
    updates = honest + [attacker]
    med = fltrust_median_aggregate(updates, g, server)[0]
    mean = fltrust_aggregate(updates, g, server)[0]
    assert abs(med[2]) == pytest.approx(0.0)  # median ignores the lone spike
    assert mean[2] > 0.05                     # the trust-weighted mean follows it
    honest_scaled = np.stack(
        [np.linalg.norm(server[0]) * (u.parameters[0] / np.linalg.norm(u.parameters[0])) for u in honest]
    )
    assert (honest_scaled.min(axis=0)[:2] - 1e-12 <= med[:2]).all()
    assert (med[:2] <= honest_scaled.max(axis=0)[:2] + 1e-12).all()


def test_fltrust_median_never_selects_an_untrusted_client() -> None:
    g: NDArrays = [np.zeros(2)]
    server: NDArrays = [np.array([1.0, 0.0])]
    updates = [
        _update("a", [1.0, 0.1]),
        _update("m1", [-5.0, 9.0]),   # cos < 0 -> trust 0
        _update("m2", [-5.0, 9.0]),
        _update("m3", [-5.0, 9.0]),
    ]
    out = fltrust_median_aggregate(updates, g, server)[0]
    expected = np.array([1.0, 0.1]) / np.linalg.norm([1.0, 0.1])
    assert out == pytest.approx(expected)


def test_fltrust_median_trust_weight_majority_controls() -> None:
    g: NDArrays = [np.zeros(2)]
    server: NDArrays = [np.array([1.0, 0.0])]
    updates = [
        _update("a", [1.0, 0.0]),      # cos 1
        _update("b", [1.0, 0.0]),      # cos 1
        _update("c", [0.2, 1.0]),      # cos ~0.196
    ]
    out = fltrust_median_aggregate(updates, g, server)[0]
    assert out == pytest.approx([1.0, 0.0])


def test_fltrust_median_keeps_global_when_nobody_trusted_or_server_still() -> None:
    g: NDArrays = [np.full(2, 7.0)]
    server: NDArrays = [np.array([8.0, 7.0])]
    opposed = [_update("m", [-1.0, 0.0], 7.0)]
    assert fltrust_median_aggregate(opposed, g, server)[0] == pytest.approx([7.0, 7.0])
    aligned = [_update("a", [1.0, 0.0], 7.0)]
    assert fltrust_median_aggregate(aligned, g, g)[0] == pytest.approx([7.0, 7.0])


# ----------------------------------------------------------------------------
# Server loop plumbing
# ----------------------------------------------------------------------------


class _Client:
    def __init__(self, cid: ClientId, delta: list[float]) -> None:
        self.client_id = cid
        self._delta = np.array(delta, dtype=np.float64)

    def local_train(self, ctx: RoundContext, config: FederationConfig) -> ClientUpdate:
        return ClientUpdate(
            self.client_id, [ctx.global_parameters[0] + self._delta], 1, 0.0, 1.0
        )

    def evaluate(self, parameters: NDArrays, config: FederationConfig):
        return 0.0, 0, {}


class _Scorer:
    def __init__(self) -> None:
        self.last_batch: Any = "batch-sentinel"

    def score_round(self, round_num, updates, global_parameters):  # noqa: ANN001
        ids = tuple(u.client_id for u in updates)
        return ids, np.full(len(ids), 0.5)


class _RecordingIntegrator(ScoreWeight):
    requires_server_update = True

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[RoundEvidence] = []

    def observe(self, evidence: RoundEvidence) -> None:
        self.seen.append(evidence)


DELTAS = {"c0": [1.0, 0.0], "c1": [0.0, 1.0], "c2": [-1.0, 0.0]}


def _config(rounds: int = 2) -> FederationConfig:
    return FederationConfig(
        num_clients=3, clients_per_round=3, num_rounds=rounds, local_epochs=1,
        local_batch_size=1, local_learning_rate=0.1, local_momentum=0.0, seed=0,
    )


def _factory(cid: ClientId) -> _Client:
    return _Client(cid, DELTAS[cid])


def _server_fn(delta: list[float]):
    calls: list[int] = []

    def fn(ctx: RoundContext, config: FederationConfig) -> NDArrays:
        calls.append(ctx.round_num)
        return [ctx.global_parameters[0] + np.array(delta)]

    return fn, calls


def test_integrator_observes_server_update_and_scorer_batch() -> None:
    fn, calls = _server_fn([1.0, 0.0])
    integ = _RecordingIntegrator()
    outcomes = run_federation(
        _config(), _factory, [np.zeros(2)], scorer=_Scorer(), integrator=integ,
        server_update_fn=fn, verbose=False,
    )
    assert calls == [1, 2]
    assert [e.round_num for e in integ.seen] == [1, 2]
    first = integ.seen[0]
    assert first.server_parameters is not None
    assert first.server_parameters[0] == pytest.approx([1.0, 0.0])
    assert first.metrics == "batch-sentinel"
    assert {u.client_id for u in first.updates} == set(DELTAS)
    assert all(o.timings["server_train_s"] > 0.0 for o in outcomes)


def test_server_gated_defense_end_to_end_drops_misaligned_clients() -> None:
    fn, _ = _server_fn([1.0, 0.0])
    outcomes = run_federation(
        _config(rounds=1), _factory, [np.zeros(2)], scorer=_Scorer(),
        integrator=ServerGatedCappedScoreWeight(), defended_base="weighted_median",
        server_update_fn=fn, verbose=False,
    )
    # Only c0 (cos 1) survives the gate; c1 (cos 0) and c2 (cos -1) are zeroed.
    assert outcomes[0].aggregated_parameters[0] == pytest.approx([1.0, 0.0])


def test_keep_global_carries_the_model_forward() -> None:
    # Server delta -e2: c0 and c2 are orthogonal (cos 0), c1 is opposed.
    fn, _ = _server_fn([0.0, -1.0])
    factory = lambda cid: _Client(cid, [1.0, 0.0] if cid != "c1" else [0.0, 1.0])  # noqa: E731
    outcomes = run_federation(
        _config(rounds=1), factory, [np.full(2, 3.0)], scorer=_Scorer(),
        integrator=ServerGatedCappedScoreWeight(), defended_base="weighted_median",
        server_update_fn=fn, verbose=False,
    )
    assert outcomes[0].aggregated_parameters[0] == pytest.approx([3.0, 3.0])


def test_defense_needing_server_update_requires_server_update_fn() -> None:
    with pytest.raises(ValueError, match="server_update_fn"):
        run_federation(
            _config(), _factory, [np.zeros(2)], scorer=_Scorer(),
            integrator=ServerGatedCappedScoreWeight(), verbose=False,
        )


def test_server_update_fn_with_plain_defense_is_still_rejected() -> None:
    fn, _ = _server_fn([1.0, 0.0])
    with pytest.raises(ValueError, match="only for components"):
        run_federation(
            _config(), _factory, [np.zeros(2)], scorer=_Scorer(),
            integrator=CappedScoreWeight(), server_update_fn=fn, verbose=False,
        )


def test_plain_defense_does_no_server_training() -> None:
    outcomes = run_federation(
        _config(), _factory, [np.zeros(2)], scorer=_Scorer(),
        integrator=CappedScoreWeight(), defended_base="weighted_median", verbose=False,
    )
    assert all(o.timings["server_train_s"] == 0.0 for o in outcomes)


def test_rooted_aggregator_selects_the_server_rooted_rule() -> None:
    fn, _ = _server_fn([1.0, 0.0])
    seen: list[str] = []

    def spy(updates, global_parameters, server_parameters):  # noqa: ANN001
        seen.append("spy")
        return fltrust_median_aggregate(updates, global_parameters, server_parameters)

    outcomes = run_federation(
        _config(rounds=1), _factory, [np.zeros(2)], server_update_fn=fn,
        rooted_aggregator=spy, verbose=False,
    )
    assert seen == ["spy"]
    # Only c0 is trusted -> the server norm along e1.
    assert outcomes[0].aggregated_parameters[0] == pytest.approx([1.0, 0.0])


def test_default_rooted_aggregator_is_fltrust() -> None:
    fn, _ = _server_fn([1.0, 1.0])
    factory = lambda cid: _Client(cid, {"c0": [1.0, 0.0], "c1": [0.0, 1.0], "c2": [1.0, 1.0]}[cid])  # noqa: E731
    outcomes = run_federation(
        _config(rounds=1), factory, [np.zeros(2)], server_update_fn=fn, verbose=False,
    )
    updates = list(outcomes[0].client_updates)
    expected = fltrust_aggregate(updates, [np.zeros(2)], [np.array([1.0, 1.0])])
    assert outcomes[0].aggregated_parameters[0] == pytest.approx(expected[0])
