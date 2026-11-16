"""Trust scoring: TOPSIS, weighting strategies, and sensitivity analysis.

The decision-matrix conventions live in :mod:`topfed.scoring.criteria`; the
TOPSIS scorer in :mod:`topfed.scoring.topsis`; the six weighting strategies in
:mod:`topfed.scoring.weighting`; and one-at-a-time weight sensitivity in
:mod:`topfed.scoring.sensitivity`.
"""

from topfed.scoring.criteria import (
    METRIC_DIRECTION,
    TRUST_METRICS,
    benefit_mask,
    build_batch,
)
from topfed.scoring.ml import (
    IsolationForestScorer,
    SupervisedScorer,
    fit_isolation_forest,
    fit_logistic,
    fit_random_forest,
)
from topfed.scoring.sensitivity import OATResult, oat_sensitivity, perturb_weight
from topfed.scoring.topsis import TopsisScorer, topsis_closeness
from topfed.scoring.weighting import (
    WEIGHTING_STRATEGIES,
    WeightingStrategy,
    critic_weights,
    entropy_weights,
    equal_weights,
)

__all__ = [
    "METRIC_DIRECTION",
    "TRUST_METRICS",
    "benefit_mask",
    "build_batch",
    "IsolationForestScorer",
    "SupervisedScorer",
    "fit_isolation_forest",
    "fit_logistic",
    "fit_random_forest",
    "OATResult",
    "oat_sensitivity",
    "perturb_weight",
    "TopsisScorer",
    "topsis_closeness",
    "WEIGHTING_STRATEGIES",
    "WeightingStrategy",
    "critic_weights",
    "entropy_weights",
    "equal_weights",
]
