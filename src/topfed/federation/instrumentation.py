"""Per-round capture trace: how far the malicious bloc moved the aggregate.

A diagnostic that reads ground truth after aggregation, so it never feeds back
into any defense. For each round it records:

* ``trace_malicious_share`` -- the aggregation weight held by malicious
  clients (``NaN`` when the rule has no per-client weights, e.g. a plain
  coordinate median, or when the round kept the global model);
* ``trace_captured_fraction`` -- the fraction of coordinates where the
  aggregate delta lies inside the malicious clients' value range but outside
  the honest clients' interquartile range, i.e. where the bloc, not the
  honest majority, set the value;
* ``trace_projection`` -- the aggregate delta projected on the unit direction
  of the malicious mean delta, next to ``trace_honest_projection`` (the honest
  mean delta on the same direction) and ``trace_malicious_norm`` (the length
  of the malicious mean delta), so a projection near the malicious norm means
  the aggregate moved as far along the attack as the bloc itself.

Quantities that need both groups are ``NaN`` in a round without them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from topfed.core.types import ClientId, ClientUpdate, NDArrays

TRACE_KEYS = (
    "trace_malicious_share",
    "trace_captured_fraction",
    "trace_projection",
    "trace_honest_projection",
    "trace_malicious_norm",
)


def _flat(arrays: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def capture_trace(
    updates: Sequence[ClientUpdate],
    global_parameters: NDArrays,
    aggregated: NDArrays,
    malicious_clients: frozenset[ClientId],
    weights: np.ndarray | None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """The round's capture trace; see the module docstring for each key."""
    out = {k: math.nan for k in TRACE_KEYS}
    is_mal = np.array([u.client_id in malicious_clients for u in updates], dtype=bool)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        total = float(w.sum())
        if total > eps:
            out["trace_malicious_share"] = float(w[is_mal].sum()) / total
    if not is_mal.any():
        return out

    g = _flat(global_parameters)
    deltas = np.stack([_flat(u.parameters) - g for u in updates])
    agg = _flat(aggregated) - g
    mal = deltas[is_mal]
    honest = deltas[~is_mal]

    mal_mean = mal.mean(axis=0)
    mal_norm = float(np.linalg.norm(mal_mean))
    out["trace_malicious_norm"] = mal_norm
    if mal_norm > eps:
        direction = mal_mean / mal_norm
        out["trace_projection"] = float(agg @ direction)
        if honest.shape[0]:
            out["trace_honest_projection"] = float(honest.mean(axis=0) @ direction)

    if honest.shape[0]:
        inside = (agg >= mal.min(axis=0)) & (agg <= mal.max(axis=0))
        q25, q75 = np.percentile(honest, [25.0, 75.0], axis=0)
        outside = (agg < q25) | (agg > q75)
        out["trace_captured_fraction"] = float(np.mean(inside & outside))
    return out
