"""Unit tests for the score-integration mechanisms."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.integration.mechanisms import (
    AdaptiveParticipation,
    CappedScoreWeight,
    HardFilter,
    Quarantine,
    ScoreWeight,
    TemporalReputation,
)
from topfed.integration.reputation import EmaReputation


def test_capped_score_weight_bounds_over_trust() -> None:
    # An over-trusted client that plain score-weighting amplifies above its equal
    # share is held at (about) the fair-share ceiling cap/n.
    ids = tuple(f"c{i}" for i in range(10))
    scores = np.array([0.95] + [0.55] * 9)
    sw = ScoreWeight().integrate(0, ids, scores).weights
    capped = CappedScoreWeight(cap=1.0).integrate(0, ids, scores).weights
    fair = 1.0 / 10
    assert sw[0] > fair  # uncapped amplifies the over-trusted client above fair share
    assert capped[0] < sw[0]  # capping reduces the amplification
    assert abs(capped[0] - fair) < abs(sw[0] - fair)  # ...pulling it toward fair share
    assert capped.sum() == pytest.approx(1.0)


def test_capped_score_weight_preserves_suppression() -> None:
    # A low-trust client stays suppressed (weight well below fair share).
    ids = tuple(f"c{i}" for i in range(10))
    scores = np.array([0.05] + [0.6] * 9)
    capped = CappedScoreWeight(cap=1.0).integrate(0, ids, scores).weights
    assert capped[0] < 0.5 / 10


def test_capped_score_weight_large_cap_recovers_score_weight() -> None:
    ids = tuple(f"c{i}" for i in range(5))
    scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    sw = ScoreWeight().integrate(0, ids, scores).weights
    capped = CappedScoreWeight(cap=100.0).integrate(0, ids, scores).weights
    assert capped == pytest.approx(sw)


def test_capped_score_weight_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError):
        CappedScoreWeight(cap=0.0)


def test_ema_reputation_smooths_history() -> None:
    rep = EmaReputation(alpha=0.5)
    first = rep.update(("a",), np.array([1.0]))
    assert first[0] == pytest.approx(1.0)  # seeds with first score
    second = rep.update(("a",), np.array([0.0]))
    assert second[0] == pytest.approx(0.5)  # 0.5*0 + 0.5*1


def test_ema_alpha_one_is_instantaneous() -> None:
    rep = EmaReputation(alpha=1.0)
    rep.update(("a",), np.array([0.9]))
    assert rep.update(("a",), np.array([0.2]))[0] == pytest.approx(0.2)


def test_ema_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        EmaReputation(alpha=0.0)
    with pytest.raises(ValueError):
        EmaReputation(alpha=1.5)


def test_hard_filter_excludes_below_threshold() -> None:
    integ = HardFilter(threshold=0.5)
    result = integ.integrate(1, ("a", "b", "c"), np.array([0.9, 0.8, 0.1]))
    assert result.weights == pytest.approx([0.5, 0.5, 0.0])
    assert result.weights.sum() == pytest.approx(1.0)
    assert [r["included"] for r in integ.records] == [True, True, False]


def test_hard_filter_keeps_all_when_none_pass() -> None:
    integ = HardFilter(threshold=0.9)
    result = integ.integrate(1, ("a", "b"), np.array([0.1, 0.2]))
    assert result.weights == pytest.approx([0.5, 0.5])


def test_score_weight_is_proportional() -> None:
    integ = ScoreWeight()
    result = integ.integrate(1, ("a", "b", "c"), np.array([0.2, 0.6, 0.2]))
    assert result.weights == pytest.approx([0.2, 0.6, 0.2])


def test_score_weight_uniform_when_all_zero() -> None:
    integ = ScoreWeight()
    result = integ.integrate(1, ("a", "b"), np.array([0.0, 0.0]))
    assert result.weights == pytest.approx([0.5, 0.5])


def test_quarantine_excludes_after_consecutive_low_rounds() -> None:
    integ = Quarantine(threshold=0.5, quarantine_limit=2)
    # Round 1: below threshold -> streak 1, still included.
    r1 = integ.integrate(1, ("a", "b"), np.array([0.1, 0.9]))
    assert r1.weights == pytest.approx([0.5, 0.5])
    # Round 2: below again -> streak 2 == limit -> excluded.
    r2 = integ.integrate(2, ("a", "b"), np.array([0.1, 0.9]))
    assert r2.weights == pytest.approx([0.0, 1.0])
    # Round 3: recovers -> streak resets, included again.
    r3 = integ.integrate(3, ("a", "b"), np.array([0.9, 0.9]))
    assert r3.weights == pytest.approx([0.5, 0.5])


def test_temporal_reputation_smooths_before_weighting() -> None:
    integ = TemporalReputation(alpha=0.5)
    integ.integrate(1, ("a", "b"), np.array([0.2, 0.8]))
    # Round 2 reputations: a=0.5*0.8+0.5*0.2=0.5, b=0.5*0.2+0.5*0.8=0.5 -> equal.
    result = integ.integrate(2, ("a", "b"), np.array([0.8, 0.2]))
    assert result.weights == pytest.approx([0.5, 0.5])


def test_adaptive_participation_emits_sampling_probs() -> None:
    integ = AdaptiveParticipation()
    result = integ.integrate(1, ("a", "b"), np.array([0.3, 0.9]))
    assert result.weights == pytest.approx([0.5, 0.5])  # equal this round
    assert result.sampling_probs == {"a": pytest.approx(0.3), "b": pytest.approx(0.9)}
