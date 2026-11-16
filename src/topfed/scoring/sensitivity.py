"""One-at-a-time (OAT) weight sensitivity for TOPSIS rankings.

The weight space is low-dimensional (five metrics), so a one-at-a-time sweep
is sufficient: each metric weight is perturbed in isolation by a set of
relative amounts, the perturbation is absorbed proportionally by the remaining
weights, and the resulting TOPSIS ranking is compared against the unperturbed
ranking via Spearman's rho and Kendall's tau. The output identifies which
single metric weight most strongly drives ranking changes under a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from topfed.scoring.topsis import topsis_closeness

DEFAULT_LEVELS: tuple[float, ...] = (0.10, 0.20, 0.30)


def perturb_weight(weights: np.ndarray, index: int, delta: float) -> np.ndarray:
    """Scale ``weights[index]`` by ``(1 + delta)``, renormalising the rest.

    The remaining weights keep their relative proportions and absorb whatever
    mass is needed to hold the total at one. ``weights`` is assumed to sum to
    one and to be non-negative.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    m = w.size
    base = w[index]
    new = base * (1.0 + delta)
    remaining = w.sum() - base
    w[index] = new
    if remaining > 1e-12:
        scale = (1.0 - new) / remaining
        for k in range(m):
            if k != index:
                w[k] = w[k] * scale
    return w


@dataclass(frozen=True)
class OATResult:
    """Rank-stability grid for one weighting strategy.

    ``spearman`` and ``kendall`` are ``(m, len(levels))`` arrays: entry
    ``[j, l]`` is the rank correlation between the unperturbed ranking and the
    ranking after perturbing weight ``j`` at level ``levels[l]``, averaged over
    the ``+`` and ``-`` perturbation directions.
    """

    levels: tuple[float, ...]
    spearman: np.ndarray
    kendall: np.ndarray


def _ranking(matrix: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    return topsis_closeness(matrix, weights, benefit)


def oat_sensitivity(
    matrix: np.ndarray,
    weights: np.ndarray,
    benefit: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> OATResult:
    """Spearman/Kendall rank stability of a TOPSIS ranking under OAT weight perturbation.

    For each metric and each level the score is averaged over a ``+level`` and
    a ``-level`` perturbation. Correlations are computed against the baseline
    ranking from the unperturbed ``weights``. A degenerate baseline (fewer than
    two clients, or no rank variation) yields ``nan`` entries.
    """
    base_scores = _ranking(matrix, weights, benefit)
    m = np.asarray(weights).size
    spearman = np.full((m, len(levels)), np.nan, dtype=np.float64)
    kendall = np.full((m, len(levels)), np.nan, dtype=np.float64)

    if base_scores.size < 2:
        return OATResult(levels=levels, spearman=spearman, kendall=kendall)

    for j in range(m):
        for li, level in enumerate(levels):
            rhos: list[float] = []
            taus: list[float] = []
            for signed in (level, -level):
                perturbed = perturb_weight(weights, j, signed)
                scores = _ranking(matrix, perturbed, benefit)
                rho = stats.spearmanr(base_scores, scores).correlation
                tau = stats.kendalltau(base_scores, scores).correlation
                if not np.isnan(rho):
                    rhos.append(float(rho))
                if not np.isnan(tau):
                    taus.append(float(tau))
            if rhos:
                spearman[j, li] = float(np.mean(rhos))
            if taus:
                kendall[j, li] = float(np.mean(taus))

    return OATResult(levels=levels, spearman=spearman, kendall=kendall)
