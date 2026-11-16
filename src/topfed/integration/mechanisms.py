"""Score-integration mechanisms.

Each mechanism turns per-client trust scores into an aggregation decision,
ordered here from most to least aggressive:

* :class:`HardFilter` — drop clients whose reputation is below a threshold.
* :class:`ScoreWeight` — weight each client's update by its reputation.
* :class:`Quarantine` — exclude clients that stay below threshold for a number
  of consecutive rounds.
* :class:`TemporalReputation` — score-weighting on an EMA-smoothed reputation,
  damping transient score dips.
* :class:`AdaptiveParticipation` — keep the current round equal-weighted but
  bias which clients the server samples next round.

All mechanisms maintain an :class:`~topfed.integration.reputation.EmaReputation`
(``alpha = 1`` means no smoothing) and append one row per client per round to
``records`` for offline analysis of the defense's decisions. None of them sees
ground-truth labels; the malicious set is joined in only when reporting.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from topfed.core.types import ClientId, IntegrationResult, TrustScores
from topfed.integration.reputation import EmaReputation

_EPS = 1e-12


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _round_threshold(fixed: float, cohort_median: bool, reputations: np.ndarray) -> float:
    """The round's exclusion threshold: ``fixed``, or the cohort's median reputation."""
    return float(np.median(reputations)) if cohort_median else fixed


def _threshold_extras(
    cohort_median: bool, threshold: float, n: int
) -> dict[str, np.ndarray] | None:
    """A ``threshold`` ledger column, only when the threshold varies by round."""
    return {"threshold": np.full(n, threshold)} if cohort_median else None


class _BaseIntegrator:
    """Shared reputation tracking and per-round decision logging."""

    def __init__(self, alpha: float = 1.0) -> None:
        self._reputation = EmaReputation(alpha)
        self.records: list[dict[str, Any]] = []

    def _log(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
        reputations: np.ndarray,
        weights: np.ndarray,
        extras: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Append one record per client; ``extras`` adds aligned columns."""
        for i, (cid, score, rep, weight) in enumerate(
            zip(client_ids, scores, reputations, weights, strict=True)
        ):
            record: dict[str, Any] = {
                "round": round_num,
                "client_id": cid,
                "score": float(score),
                "reputation": float(rep),
                "weight": float(weight),
                "included": bool(weight > 0.0),
            }
            for name, column in (extras or {}).items():
                record[name] = float(column[i])
            self.records.append(record)


class HardFilter(_BaseIntegrator):
    """Exclude below-threshold clients; equal-weight the rest.

    With ``cohort_median`` the threshold of a round is the median reputation
    of that round's clients instead of the fixed ``threshold``, so the rule
    stays selective for a score whose values sit on one side of the fixed
    threshold; the ledger then carries the round's threshold.
    """

    def __init__(
        self, threshold: float = 0.5, alpha: float = 1.0, cohort_median: bool = False
    ) -> None:
        super().__init__(alpha)
        self.threshold = threshold
        self.cohort_median = cohort_median

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult:
        reputations = self._reputation.update(client_ids, scores)
        threshold = _round_threshold(self.threshold, self.cohort_median, reputations)
        keep = reputations >= threshold
        if not keep.any():
            # Never drop everyone — fall back to including all this round.
            keep = np.ones_like(keep)
        weights = keep.astype(np.float64)
        weights /= weights.sum()
        self._log(
            round_num, client_ids, scores, reputations, weights,
            _threshold_extras(self.cohort_median, threshold, len(client_ids)),
        )
        return IntegrationResult(weights=weights)


class ScoreWeight(_BaseIntegrator):
    """Weight each client's update by its (non-negative) reputation."""

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult:
        reputations = self._reputation.update(client_ids, scores)
        clipped = np.clip(reputations, 0.0, None)
        total = clipped.sum()
        weights = clipped / total if total > _EPS else _uniform(len(client_ids))
        self._log(round_num, client_ids, scores, reputations, weights)
        return IntegrationResult(weights=weights)


class CappedScoreWeight(_BaseIntegrator):
    """Score-weighting with a per-client influence ceiling.

    Weights start proportional to reputation (as in :class:`ScoreWeight`), then
    each is capped at ``cap`` times an equal share (ceiling ``cap / n``) and the
    vector is renormalized. Down-weighting of low-trust clients is preserved, so
    a detectably anomalous client is still suppressed, and no client outweighs
    another client that also reaches the cap, so a stealthy update that the
    scorer over-trusts gets no more weight than the other capped clients.

    The cap bounds the share before renormalization, not the final weight:
    after renormalization every capped client holds the same, largest weight,
    which exceeds ``cap / n`` by the mass removed from the down-weighted
    clients. When all other scores collapse, one client can still end with
    most of the weight. Larger ``cap`` values permit proportionally more
    reweighting, recovering plain score-weighting in the limit.
    """

    def __init__(self, cap: float = 1.0, alpha: float = 1.0) -> None:
        super().__init__(alpha)
        if cap <= 0:
            raise ValueError(f"cap must be positive; got {cap}")
        self.cap = cap

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult:
        reputations = self._reputation.update(client_ids, scores)
        weights = self._capped_weights(reputations)
        self._log(round_num, client_ids, scores, reputations, weights)
        return IntegrationResult(weights=weights)

    def _capped_weights(self, reputations: np.ndarray) -> np.ndarray:
        """Reputation shares, each capped at ``cap / n``, renormalized."""
        clipped = np.clip(reputations, 0.0, None)
        n = len(reputations)
        total = clipped.sum()
        weights = clipped / total if total > _EPS else _uniform(n)
        weights = np.minimum(weights, self.cap / n)
        renorm = weights.sum()
        return weights / renorm if renorm > _EPS else _uniform(n)


class TemporalReputation(ScoreWeight):
    """Score-weighting on an EMA-smoothed reputation (``alpha < 1``)."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__(alpha)


class Quarantine(_BaseIntegrator):
    """Exclude clients below threshold for ``quarantine_limit`` rounds running.

    ``cohort_median`` replaces the fixed ``threshold`` by the round's median
    reputation, as in :class:`HardFilter`.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        quarantine_limit: int = 3,
        alpha: float = 1.0,
        cohort_median: bool = False,
    ) -> None:
        super().__init__(alpha)
        self.threshold = threshold
        self.quarantine_limit = quarantine_limit
        self.cohort_median = cohort_median
        self._below_streak: dict[ClientId, int] = {}

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult:
        reputations = self._reputation.update(client_ids, scores)
        threshold = _round_threshold(self.threshold, self.cohort_median, reputations)
        included = np.ones(len(client_ids), dtype=bool)
        for i, cid in enumerate(client_ids):
            if reputations[i] < threshold:
                self._below_streak[cid] = self._below_streak.get(cid, 0) + 1
            else:
                self._below_streak[cid] = 0
            if self._below_streak[cid] >= self.quarantine_limit:
                included[i] = False
        if not included.any():
            included = np.ones_like(included)
        weights = included.astype(np.float64)
        weights /= weights.sum()
        self._log(
            round_num, client_ids, scores, reputations, weights,
            _threshold_extras(self.cohort_median, threshold, len(client_ids)),
        )
        return IntegrationResult(weights=weights)


class AdaptiveParticipation(_BaseIntegrator):
    """Equal-weight this round; bias next round's sampling by reputation."""

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult:
        reputations = self._reputation.update(client_ids, scores)
        weights = _uniform(len(client_ids))
        self._log(round_num, client_ids, scores, reputations, weights)
        # Sampling bias spans every client seen so far; the server fills in a
        # default for clients not yet observed.
        probs = self._reputation.as_dict()
        return IntegrationResult(weights=weights, sampling_probs=probs)
