"""Byzantine-robust aggregation rules used as defense baselines.

These combine client updates from their geometry alone (no per-client trust
score), kept separate from the score-integration mechanisms in
:mod:`topfed.integration`.
"""

from topfed.aggregation.flame import FLAME, cosine_distance_matrix, flame_cluster
from topfed.aggregation.mesas import (
    MESAS,
    mesas_metrics,
    mesas_prune,
    series_is_significant,
    smaller_cluster,
)
from topfed.aggregation.robust import (
    ClippedMean,
    DeltaAggregator,
    FoolsGold,
    Krum,
    coordinate_median_aggregate,
    foolsgold_weights,
    history_similarity,
    krum_aggregate,
    krum_select,
    trimmed_mean_aggregate,
)

__all__ = [
    "FLAME",
    "MESAS",
    "ClippedMean",
    "DeltaAggregator",
    "FoolsGold",
    "Krum",
    "coordinate_median_aggregate",
    "cosine_distance_matrix",
    "flame_cluster",
    "foolsgold_weights",
    "history_similarity",
    "krum_aggregate",
    "krum_select",
    "mesas_metrics",
    "mesas_prune",
    "series_is_significant",
    "smaller_cluster",
    "trimmed_mean_aggregate",
]
