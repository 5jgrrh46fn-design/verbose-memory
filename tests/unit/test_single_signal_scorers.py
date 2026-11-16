"""Single-signal round scorers: FLTrust's server cosine and FoolsGold's weight."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.aggregation import foolsgold_weights
from topfed.core.types import ClientUpdate, RoundEvidence
from topfed.federation.scoring_bridge import (
    HistorySimilarityScorer,
    ServerCosineScorer,
)
from topfed.integration import HardFilter, Quarantine


def _u(cid: str, delta: list[float]) -> ClientUpdate:
    return ClientUpdate(cid, [np.asarray(delta, dtype=np.float64)], 1, 0.0, 1.0)


GLOBAL = [np.zeros(3)]


def _observe(scorer: ServerCosineScorer, round_num: int, updates, server) -> None:  # noqa: ANN001
    scorer.observe(
        RoundEvidence(
            round_num=round_num,
            updates=tuple(updates),
            global_parameters=GLOBAL,
            server_parameters=[np.asarray(server, dtype=np.float64)],
        )
    )


def test_server_cosine_is_the_relu_of_the_cosine_to_the_server_update() -> None:
    updates = [
        _u("aligned", [2.0, 0.0, 0.0]),
        _u("half", [1.0, 1.0, 0.0]),
        _u("orthogonal", [0.0, 3.0, 0.0]),
        _u("opposed", [-1.0, 0.0, 0.0]),
        _u("zero", [0.0, 0.0, 0.0]),
    ]
    scorer = ServerCosineScorer()
    _observe(scorer, 1, updates, [1.0, 0.0, 0.0])
    ids, scores = scorer.score_round(1, updates, GLOBAL)
    assert ids == ("aligned", "half", "orthogonal", "opposed", "zero")
    np.testing.assert_allclose(scores, [1.0, 1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0], atol=1e-12)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    raw = [r["m6_server_cosine"] for r in scorer.records]
    np.testing.assert_allclose(raw, [1.0, 1.0 / np.sqrt(2.0), 0.0, -1.0, 0.0], atol=1e-12)


def test_server_cosine_ignores_the_update_norm() -> None:
    updates = [_u("small", [0.01, 0.01, 0.0]), _u("large", [50.0, 50.0, 0.0])]
    scorer = ServerCosineScorer()
    _observe(scorer, 4, updates, [1.0, 0.0, 0.0])
    _, scores = scorer.score_round(4, updates, GLOBAL)
    assert scores[0] == pytest.approx(scores[1])


def test_server_cosine_scores_zero_when_the_server_did_not_move() -> None:
    updates = [_u("a", [1.0, 0.0, 0.0]), _u("b", [0.0, 1.0, 0.0])]
    scorer = ServerCosineScorer()
    _observe(scorer, 1, updates, [0.0, 0.0, 0.0])
    _, scores = scorer.score_round(1, updates, GLOBAL)
    np.testing.assert_array_equal(scores, [0.0, 0.0])


def test_server_cosine_needs_this_rounds_server_update() -> None:
    updates = [_u("a", [1.0, 0.0, 0.0])]
    scorer = ServerCosineScorer()
    assert scorer.requires_server_update
    with pytest.raises(RuntimeError, match="server update"):
        scorer.score_round(1, updates, GLOBAL)
    _observe(scorer, 1, updates, [1.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="server update"):
        scorer.score_round(2, updates, GLOBAL)
    # The evidence is consumed by the round it was delivered for.
    _observe(scorer, 3, updates, [1.0, 0.0, 0.0])
    scorer.score_round(3, updates, GLOBAL)
    with pytest.raises(RuntimeError, match="server update"):
        scorer.score_round(3, updates, GLOBAL)


def test_history_similarity_matches_foolsgold_on_the_accumulated_histories() -> None:
    rng = np.random.default_rng(0)
    round1 = {c: rng.normal(size=6) for c in ("a", "b", "c", "d")}
    round2 = {c: rng.normal(size=6) for c in ("a", "b", "e")}
    g = [np.zeros(6)]
    scorer = HistorySimilarityScorer(kappa=1.0)

    ids, s1 = scorer.score_round(1, [_u(c, list(d)) for c, d in round1.items()], g)
    assert ids == ("a", "b", "c", "d")
    expected1, max_cs1 = foolsgold_weights(np.stack(list(round1.values())), 1.0)
    np.testing.assert_allclose(s1, expected1, atol=1e-6)

    # Round 2: a and b carry both rounds' deltas, e only its own; c and d,
    # not selected, keep their histories and are not scored.
    ids, s2 = scorer.score_round(2, [_u(c, list(d)) for c, d in round2.items()], g)
    assert ids == ("a", "b", "e")
    histories = np.stack(
        [round1["a"] + round2["a"], round1["b"] + round2["b"], round2["e"]]
    )
    expected2, _ = foolsgold_weights(histories.astype(np.float32), 1.0)
    np.testing.assert_allclose(s2, expected2, atol=1e-6)
    assert [r["round"] for r in scorer.records] == [1] * 4 + [2] * 3
    np.testing.assert_allclose(
        [r["max_similarity"] for r in scorer.records[:4]], max_cs1, atol=1e-6
    )


def test_history_similarity_scores_look_alikes_low_and_loners_high() -> None:
    base = np.array([1.0, 2.0, -1.0, 0.5])
    updates = [
        _u("sybil1", list(base)),
        _u("sybil2", list(base * 1.01)),
        _u("loner1", [1.0, -1.0, 0.0, 0.0]),
        _u("loner2", [0.0, 0.0, 1.0, -3.0]),
    ]
    scorer = HistorySimilarityScorer()
    _, scores = scorer.score_round(1, updates, [np.zeros(4)])
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert scores[0] == pytest.approx(0.0, abs=1e-6)
    assert scores[1] == pytest.approx(0.0, abs=1e-6)
    assert scores[2] == pytest.approx(1.0) and scores[3] == pytest.approx(1.0)


def test_history_similarity_single_client_is_fully_trusted() -> None:
    scorer = HistorySimilarityScorer()
    _, scores = scorer.score_round(1, [_u("only", [1.0, 2.0, 3.0])], GLOBAL)
    np.testing.assert_allclose(scores, [1.0])


def test_history_similarity_rejects_nonpositive_kappa() -> None:
    with pytest.raises(ValueError, match="kappa"):
        HistorySimilarityScorer(kappa=0.0)


def test_hard_filter_cohort_median_keeps_the_upper_half() -> None:
    ids = ("a", "b", "c", "d")
    # Every score is above the fixed threshold: the fixed rule keeps all, the
    # cohort-median rule still drops the lower half.
    scores = np.array([0.9, 0.8, 0.95, 0.99])
    fixed = HardFilter(threshold=0.5).integrate(1, ids, scores)
    np.testing.assert_allclose(fixed.weights, [0.25] * 4)
    hf = HardFilter(threshold=0.5, cohort_median=True)
    relative = hf.integrate(1, ids, scores)
    np.testing.assert_allclose(relative.weights, [0.0, 0.0, 0.5, 0.5])
    assert all(r["threshold"] == pytest.approx(0.925) for r in hf.records)


def test_hard_filter_cohort_median_keeps_everyone_on_a_tie() -> None:
    hf = HardFilter(cohort_median=True)
    result = hf.integrate(1, ("a", "b", "c"), np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(result.weights, [1 / 3] * 3)


def test_fixed_threshold_rules_log_no_threshold_column() -> None:
    hf = HardFilter(threshold=0.5)
    hf.integrate(1, ("a", "b"), np.array([0.2, 0.8]))
    q = Quarantine(threshold=0.5)
    q.integrate(1, ("a", "b"), np.array([0.2, 0.8]))
    assert all("threshold" not in r for r in hf.records + q.records)


def test_quarantine_cohort_median_counts_streaks_below_the_round_median() -> None:
    ids = ("low", "mid", "high")
    scores = np.array([0.7, 0.8, 0.9])
    fixed = Quarantine(threshold=0.5, quarantine_limit=2)
    relative = Quarantine(threshold=0.5, quarantine_limit=2, cohort_median=True)
    for r in (1, 2):
        out_fixed = fixed.integrate(r, ids, scores)
        out_relative = relative.integrate(r, ids, scores)
    np.testing.assert_allclose(out_fixed.weights, [1 / 3] * 3)
    # "low" sat below the cohort median (0.8) twice in a row.
    np.testing.assert_allclose(out_relative.weights, [0.0, 0.5, 0.5])
    assert relative.records[-1]["threshold"] == pytest.approx(0.8)
