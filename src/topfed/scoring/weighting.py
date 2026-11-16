"""Weighting strategies for the trust metrics.

A strategy maps the per-round decision matrix to a weight vector that sums to
one, in the column order of :data:`topfed.scoring.criteria.TRUST_METRICS`.
Fixed strategies (equal-sum, the three expert templates and the single-metric
``m4_only``) ignore the matrix;
data-derived strategies (entropy, CRITIC) read it so the weights track which
metrics actually separate clients in the round at hand.

Expert templates are defined per metric name and resolved against the
canonical metric order, so they assume a matrix whose columns are
``TRUST_METRICS``. Equal, entropy, and CRITIC depend only on the matrix shape
and its values, not on column identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from topfed.scoring.criteria import TRUST_METRICS

WeightingStrategy = Callable[[np.ndarray], np.ndarray]

_EPS = 1e-12

# Each template encodes one defensive priority over the five metrics.
_EXPERT_TEMPLATES: Mapping[str, Mapping[str, float]] = {
    "consistency": {
        "m1_cosine_to_mean": 0.30,
        "m2_norm_ratio": 0.10,
        "m3_temporal_cv": 0.10,
        "m4_val_loss_delta": 0.30,
        "m5_max_layer_deviation": 0.20,
    },
    "magnitude": {
        "m1_cosine_to_mean": 0.10,
        "m2_norm_ratio": 0.30,
        "m3_temporal_cv": 0.10,
        "m4_val_loss_delta": 0.20,
        "m5_max_layer_deviation": 0.30,
    },
    "stability": {
        "m1_cosine_to_mean": 0.20,
        "m2_norm_ratio": 0.10,
        "m3_temporal_cv": 0.30,
        "m4_val_loss_delta": 0.25,
        "m5_max_layer_deviation": 0.15,
    },
}


def equal_weights(matrix: np.ndarray) -> np.ndarray:
    """Uniform weights ``1/m``."""
    m = matrix.shape[1]
    return np.full(m, 1.0 / m, dtype=np.float64)


def _template_weights(name: str) -> np.ndarray:
    template = _EXPERT_TEMPLATES[name]
    return np.array([template[k] for k in TRUST_METRICS], dtype=np.float64)


def consistency_weights(matrix: np.ndarray) -> np.ndarray:
    """Expert template prioritising directional and functional signals."""
    return _template_weights("consistency")


def magnitude_weights(matrix: np.ndarray) -> np.ndarray:
    """Expert template prioritising scale-based anomalies."""
    return _template_weights("magnitude")


def stability_weights(matrix: np.ndarray) -> np.ndarray:
    """Expert template prioritising temporal behavioural patterns."""
    return _template_weights("stability")


def m4_only_weights(matrix: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """All weight on the validation-loss change; the other metrics count for nothing.

    With one weighted cost column the closeness score is that column's
    min-max rank inside the round: 1 for the client whose update lowers the
    validation loss most, 0 for the one that raises it most. Like the expert
    templates it assumes the canonical metric columns.
    """
    return np.array(
        [1.0 if k == "m4_val_loss_delta" else 0.0 for k in TRUST_METRICS], dtype=np.float64
    )


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Shannon-entropy weights from per-column min-max-normalised values.

    Columns whose clients differ more (lower entropy) receive higher weight.
    A constant column carries no information and collapses to zero weight;
    when no column discriminates, weights fall back to uniform.
    """
    x = np.asarray(matrix, dtype=np.float64)
    n, m = x.shape
    if n < 2:
        return np.full(m, 1.0 / m, dtype=np.float64)

    col_min = x.min(axis=0)
    col_max = x.max(axis=0)
    span = col_max - col_min
    span_safe = np.where(span > _EPS, span, 1.0)
    # Constant columns map to all-ones, yielding a uniform p and maximal
    # entropy (zero weight) rather than a spurious low-entropy spike.
    scaled = np.where(span > _EPS, (x - col_min) / span_safe, 1.0)

    col_sum = scaled.sum(axis=0)
    col_sum_safe = np.where(col_sum > _EPS, col_sum, 1.0)
    p = scaled / col_sum_safe
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > _EPS, p * np.log(p), 0.0)
    k = 1.0 / np.log(n)
    entropy = -k * plogp.sum(axis=0)  # in [0, 1]

    diversification = 1.0 - entropy
    total = diversification.sum()
    if total <= _EPS:
        return np.full(m, 1.0 / m, dtype=np.float64)
    return diversification / total


def critic_weights(matrix: np.ndarray) -> np.ndarray:
    """CRITIC weights from per-column dispersion and inter-column correlation.

    A column's weight grows with its standard deviation (information content)
    and with how uncorrelated it is from the others (low redundancy). Constant
    columns have zero dispersion and therefore zero weight; correlations that
    are undefined (against a constant column) are treated as zero.
    """
    x = np.asarray(matrix, dtype=np.float64)
    n, m = x.shape
    if n < 2:
        return np.full(m, 1.0 / m, dtype=np.float64)

    sigma = x.std(axis=0, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
    if corr.shape != (m, m):
        corr = np.eye(m)

    information = sigma * np.sum(1.0 - corr, axis=1)
    total = information.sum()
    if total <= _EPS:
        return np.full(m, 1.0 / m, dtype=np.float64)
    return information / total


WEIGHTING_STRATEGIES: dict[str, WeightingStrategy] = {
    "equal": equal_weights,
    "consistency": consistency_weights,
    "magnitude": magnitude_weights,
    "stability": stability_weights,
    "m4_only": m4_only_weights,
    "entropy": entropy_weights,
    "critic": critic_weights,
}
