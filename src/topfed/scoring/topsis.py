"""TOPSIS trustworthiness scoring.

TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) ranks
clients by how close their behavioural profile sits to an ideal (all-benign)
point and how far from the anti-ideal. Given an ``n``-client by ``m``-metric
decision matrix and a weight vector, it returns one closeness coefficient per
client in ``[0, 1]``, higher meaning more trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topfed.core.types import MetricBatch, TrustScores
from topfed.scoring.criteria import benefit_mask
from topfed.scoring.weighting import WeightingStrategy, equal_weights


def topsis_closeness(
    matrix: np.ndarray,
    weights: np.ndarray,
    benefit: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Closeness coefficients for an ``(n, m)`` decision matrix.

    ``weights`` sums to one over the ``m`` columns; ``benefit`` is a boolean
    mask, ``True`` where a higher value is more trustworthy. A column with zero
    norm (all clients identical on that metric) contributes the constant
    ``1/sqrt(n)`` to every row, so it cannot separate clients. When a client is
    equidistant from both ideal points (e.g. a single-client round) its score
    is the neutral ``0.5``.
    """
    x = np.asarray(matrix, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    w = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(benefit, dtype=bool)

    col_norm = np.sqrt(np.sum(x**2, axis=0))
    col_norm_safe = np.where(col_norm > eps, col_norm, 1.0)
    r = np.where(col_norm > eps, x / col_norm_safe, 1.0 / np.sqrt(n))
    v = w * r

    col_max = v.max(axis=0)
    col_min = v.min(axis=0)
    ideal = np.where(mask, col_max, col_min)
    anti_ideal = np.where(mask, col_min, col_max)

    d_plus = np.sqrt(np.sum((v - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((v - anti_ideal) ** 2, axis=1))
    denom = d_plus + d_minus
    return np.where(denom > eps, d_minus / np.where(denom > eps, denom, 1.0), 0.5)


@dataclass(frozen=True)
class TopsisScorer:
    """Scores clients by TOPSIS closeness under a chosen weighting strategy.

    The weighting strategy is applied to each round's decision matrix, so the
    data-derived strategies (entropy, CRITIC) adapt their weights round by
    round. Benefit/cost orientation is read from the batch's metric names.
    """

    weighting: WeightingStrategy = equal_weights
    eps: float = 1e-12

    def score(self, batch: MetricBatch) -> TrustScores:
        mask = benefit_mask(batch.metric_names)
        weights = self.weighting(batch.matrix)
        return topsis_closeness(batch.matrix, weights, mask, self.eps)
