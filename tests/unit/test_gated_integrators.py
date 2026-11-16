"""Gated capped score-weighting: server-cosine gate (H1) and M4-band control (H4)."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.core.types import ClientUpdate, MetricBatch, NDArrays, RoundEvidence
from topfed.integration.gated import (
    M4BandCappedScoreWeight,
    ServerGatedCappedScoreWeight,
)
from topfed.integration.mechanisms import CappedScoreWeight
from topfed.scoring.criteria import TRUST_METRICS

GLOBAL: NDArrays = [np.zeros(3)]
SERVER: NDArrays = [np.array([1.0, 0.0, 0.0])]


def _update(cid: str, delta: list[float]) -> ClientUpdate:
    return ClientUpdate(cid, [np.array(delta, dtype=np.float64)], 1, 0.0, 1.0)


def _ids(n: int) -> tuple[str, ...]:
    return tuple(f"c{i}" for i in range(n))


def _deltas_with_cosines(cosines: np.ndarray, rng: np.random.Generator) -> list[list[float]]:
    """3-d deltas with the given cosines to the server direction e1, random norms."""
    out = []
    for c in cosines:
        norm = float(rng.uniform(0.5, 3.0))
        out.append([norm * c, norm * float(np.sqrt(1.0 - c * c)), 0.0])
    return out


def _server_evidence(round_num: int, deltas: list[list[float]]) -> RoundEvidence:
    updates = tuple(_update(cid, d) for cid, d in zip(_ids(len(deltas)), deltas, strict=True))
    return RoundEvidence(round_num, updates, GLOBAL, server_parameters=SERVER)


def _run_h1(scores: np.ndarray, deltas: list[list[float]], cap: float = 1.0):
    integ = ServerGatedCappedScoreWeight(cap=cap)
    integ.observe(_server_evidence(1, deltas))
    result = integ.integrate(1, _ids(len(scores)), scores)
    return integ, result


# ----------------------------------------------------------------------------
# H1: server-cosine gate
# ----------------------------------------------------------------------------


def test_h1_with_fully_aligned_cohort_equals_capped_score_weight() -> None:
    scores = np.array([0.9, 0.2, 0.5, 0.7, 0.05])
    deltas = [[2.0, 0.0, 0.0]] * 5  # cosine 1 to the server delta
    _, result = _run_h1(scores, deltas)
    expected = CappedScoreWeight(cap=1.0).integrate(1, _ids(5), scores).weights
    assert result.weights == pytest.approx(expected)
    assert not result.keep_global


def test_h1_is_down_weight_only() -> None:
    # For any scores and cosines: the gate lies in [0, 1], each client's vote
    # before normalisation is its capped weight times the gate (never above
    # the capped weight), and the returned weights are those votes normalised.
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 12))
        scores = rng.uniform(0.0, 1.0, n)
        cosines = rng.uniform(-1.0, 1.0, n)
        integ, result = _run_h1(scores, _deltas_with_cosines(cosines, rng))
        rows = integ.records
        capped = np.array([r["capped_weight"] for r in rows])
        gate = np.array([r["gate"] for r in rows])
        assert capped == pytest.approx(
            CappedScoreWeight(cap=1.0).integrate(1, _ids(n), scores).weights
        )
        assert ((gate >= 0.0) & (gate <= 1.0)).all()
        assert gate == pytest.approx(np.clip(cosines, 0.0, None), abs=1e-9)
        votes = capped * gate
        assert (votes <= capped + 1e-15).all()
        if votes.sum() > 0:
            assert result.weights == pytest.approx(votes / votes.sum())


def test_h1_opposed_client_is_excluded_and_aligned_keeps_capped_share() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    # c0 opposed (cos -1), c1 orthogonal (cos 0), c2 and c3 aligned (cos 1).
    deltas = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    integ, result = _run_h1(scores, deltas)
    assert result.weights == pytest.approx([0.0, 0.0, 0.5, 0.5])
    assert [r["included"] for r in integ.records] == [False, False, True, True]
    assert [r["server_cosine"] for r in integ.records] == pytest.approx([-1.0, 0.0, 1.0, 1.0])


def test_h1_raising_a_clients_cosine_never_lowers_its_weight() -> None:
    scores = np.array([0.8, 0.4, 0.6])
    base = np.array([0.3, 0.5, 0.2])
    rng = np.random.default_rng(1)
    prev = -1.0
    for c in np.linspace(-0.5, 1.0, 16):
        cos = base.copy()
        cos[1] = c
        _, result = _run_h1(scores, _deltas_with_cosines(cos, rng))
        assert result.weights[1] >= prev - 1e-12
        prev = float(result.weights[1])


def test_h1_high_trust_client_is_still_capped_at_a_fair_share() -> None:
    # c0 scores 1.0 against nine clients at 0.1: by trust share alone it would
    # outweigh each of them 10 to 1. Its share (1/1.9) is capped at 1/n = 0.1
    # while theirs (0.1/1.9) is under the cap, so with every gate at 1 the
    # ratio falls to 0.1 / (0.1/1.9) = 1.9.
    scores = np.array([1.0] + [0.1] * 9)
    deltas = [[1.0, 0.0, 0.0]] * 10
    _, result = _run_h1(scores, deltas)
    assert result.weights[0] / result.weights[1] == pytest.approx(1.9)


def test_h1_keeps_global_when_no_client_is_aligned() -> None:
    scores = np.array([0.9, 0.9, 0.9])
    deltas = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.5, 0.0]]
    integ, result = _run_h1(scores, deltas)
    assert result.keep_global
    assert result.weights == pytest.approx([0.0, 0.0, 0.0])
    assert not any(r["included"] for r in integ.records)


def test_h1_requires_evidence_for_the_same_round() -> None:
    integ = ServerGatedCappedScoreWeight()
    with pytest.raises(RuntimeError, match="observe"):
        integ.integrate(1, _ids(2), np.array([0.5, 0.5]))
    integ.observe(_server_evidence(1, [[1.0, 0.0, 0.0]] * 2))
    with pytest.raises(RuntimeError, match="observe"):
        integ.integrate(2, _ids(2), np.array([0.5, 0.5]))


def test_h1_rejects_evidence_without_server_update() -> None:
    integ = ServerGatedCappedScoreWeight()
    updates = (_update("c0", [1.0, 0.0, 0.0]), _update("c1", [1.0, 0.0, 0.0]))
    integ.observe(RoundEvidence(1, updates, GLOBAL))
    with pytest.raises(ValueError, match="server update"):
        integ.integrate(1, _ids(2), np.array([0.5, 0.5]))


def test_h1_declares_it_needs_the_server_update() -> None:
    assert ServerGatedCappedScoreWeight.requires_server_update is True
    assert M4BandCappedScoreWeight.requires_server_update is False


# ----------------------------------------------------------------------------
# H4: M4-band control
# ----------------------------------------------------------------------------


def _m4_evidence(m4: np.ndarray, round_num: int = 1) -> RoundEvidence:
    n = len(m4)
    matrix = np.zeros((n, len(TRUST_METRICS)))
    matrix[:, TRUST_METRICS.index("m4_val_loss_delta")] = m4
    batch = MetricBatch(_ids(n), TRUST_METRICS, matrix)
    updates = tuple(_update(cid, [1.0, 0.0, 0.0]) for cid in _ids(n))
    return RoundEvidence(round_num, updates, GLOBAL, metrics=batch)


def test_h4_gates_out_clients_above_the_cohort_quantile() -> None:
    m4 = np.array([0.1, 0.5, 0.2, 3.0, 0.3, 0.05, 0.4, 0.15, 0.25, 2.0])
    integ = M4BandCappedScoreWeight(quantile=0.75)
    integ.observe(_m4_evidence(m4))
    result = integ.integrate(1, _ids(10), np.full(10, 0.5))
    band = float(np.quantile(m4, 0.75))
    kept = m4 <= band
    assert kept.sum() == 7  # linear quantile over 10 values keeps 7
    assert (result.weights[~kept] == 0.0).all()
    assert result.weights[kept] == pytest.approx(np.full(7, 1.0 / 7))
    assert all(r["m4_band"] == pytest.approx(band) for r in integ.records)


def test_h4_is_down_weight_only() -> None:
    rng = np.random.default_rng(2)
    for _ in range(100):
        n = int(rng.integers(3, 12))
        scores = rng.uniform(0.0, 1.0, n)
        integ = M4BandCappedScoreWeight()
        integ.observe(_m4_evidence(rng.normal(size=n)))
        integ.integrate(1, _ids(n), scores)
        capped = np.array([r["capped_weight"] for r in integ.records])
        gate = np.array([r["gate"] for r in integ.records])
        assert set(np.unique(gate)) <= {0.0, 1.0}
        assert (capped * gate <= capped + 1e-15).all()


def test_h4_requires_the_m4_column() -> None:
    integ = M4BandCappedScoreWeight()
    updates = (_update("c0", [1.0, 0.0, 0.0]),)
    integ.observe(RoundEvidence(1, updates, GLOBAL))
    with pytest.raises(ValueError, match="m4_val_loss_delta"):
        integ.integrate(1, ("c0",), np.array([0.5]))


def test_h4_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError):
        M4BandCappedScoreWeight(quantile=0.0)
