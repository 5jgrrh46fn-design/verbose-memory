"""Canonical trust-metric set and per-metric optimisation direction.

The five behavioural metrics are scored together by TOPSIS and the
learning-based detectors. Their column order is fixed here so every
component — decision-matrix construction, weighting, scoring — agrees on
which column is which metric and on whether higher values are good
(benefit) or suspicious (cost).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from topfed.core.types import MetricBatch

TRUST_METRICS: tuple[str, ...] = (
    "m1_cosine_to_mean",
    "m2_norm_ratio",
    "m3_temporal_cv",
    "m4_val_loss_delta",
    "m5_max_layer_deviation",
)

# Optional server-rooted criterion: cosine of the client's delta to the
# server's own delta from training on its clean root set. Only available when
# the server trains on a root set each round, so it is not part of the
# canonical five.
SERVER_COSINE = "m6_server_cosine"
ROOTED_TRUST_METRICS: tuple[str, ...] = TRUST_METRICS + (SERVER_COSINE,)

# Optional cross-client agreement criterion: the largest cosine similarity
# between the client's accumulated update history and that of any other
# client in the round. Colluders and Sybils that pursue one objective
# accumulate aligned histories; honest non-IID clients do not.
PAIRWISE_AGREEMENT = "m6_pairwise_agreement"
AGREEMENT_TRUST_METRICS: tuple[str, ...] = TRUST_METRICS + (PAIRWISE_AGREEMENT,)

# +1: benefit criterion (higher is more trustworthy); -1: cost criterion
# (higher is more suspicious). Cosine-to-mean and cosine-to-server rise as an
# update aligns with the cohort or the server; the remaining metrics rise as an
# update looks anomalous (pairwise agreement: as it looks coordinated).
METRIC_DIRECTION: Mapping[str, int] = {
    "m1_cosine_to_mean": +1,
    "m2_norm_ratio": -1,
    "m3_temporal_cv": -1,
    "m4_val_loss_delta": -1,
    "m5_max_layer_deviation": -1,
    SERVER_COSINE: +1,
    PAIRWISE_AGREEMENT: -1,
}


def benefit_mask(metric_names: Sequence[str]) -> np.ndarray:
    """Boolean benefit/cost mask aligned to ``metric_names`` (True = benefit)."""
    return np.array(
        [METRIC_DIRECTION[name] == +1 for name in metric_names], dtype=bool
    )


def build_batch(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str] = TRUST_METRICS,
) -> MetricBatch:
    """Assemble a :class:`MetricBatch` from per-client metric records.

    Each row must carry a ``client_id`` and every name in ``metric_names``.
    Rows are taken in the given order; the resulting matrix is ``float64``.
    """
    names = tuple(metric_names)
    client_ids = tuple(str(r["client_id"]) for r in rows)
    if rows:
        matrix = np.array(
            [[float(r[name]) for name in names] for r in rows],
            dtype=np.float64,
        )
    else:
        matrix = np.empty((0, len(names)), dtype=np.float64)
    return MetricBatch(client_ids=client_ids, metric_names=names, matrix=matrix)
