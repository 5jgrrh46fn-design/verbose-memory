"""Coordinate-level diagnostic for median aggregation rules.

A per-coordinate median picks, in every coordinate, the value of exactly one
client. This module records which client that was, so a run can show whether
the compromised clients supply more of the aggregate than their share of the
cohort or of the weight would predict. Like the capture trace it reads ground
truth after aggregation and never feeds back into any rule.

:func:`weighted_median_selection` is the selection step of the weighted
coordinate median: for an ``(n, m)`` array of client rows it returns, per
column, the index of the row whose value the rule selects. The median rules
in :mod:`topfed.federation.server` take their values through it, so the
diagnostic and the aggregate see one and the same choice. With equal weights
it is the lower middle row, so for an even ``n`` the plain coordinate median
(which averages the two middle values) is attributed to the lower one.

:class:`MedianSelectionRecorder` keeps two tables for the run:

* one row per round and sampled client: the client's renormalised
  aggregation weight and the fraction of coordinates whose selected value
  came from it (``selection_fraction``);
* one row per round: the attackers' combined selection fraction
  (``attacker_selection_fraction``), their equal share ``m / K``
  (``equal_share``), their renormalised weight share ``W_A``
  (``weight_share``), the L2 norm of the aggregate's displacement from the
  honest-only median (the same rule over the honest rows with their weights
  renormalised, ``displacement_norm``) and the cosine of that displacement
  to the attackers' mean delta (``displacement_cosine``, ``NaN`` when either
  vector is zero or a group is empty).

Selection fraction 0 does not imply displacement 0: an attacker's row shifts
the rank of every honest value on the other side of it, so removing it can
move the half-weight point to a neighbouring honest value. The displacement
measures that too. Server-rooted median rules (FLTrust on a median base)
are not covered: their weights and rescaled rows stay inside the rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from topfed.core.types import ClientId, ClientUpdate, NDArrays

MedianRule = Literal["weighted_median", "median"]

CLIENT_KEYS = ("round", "client_id", "is_malicious", "weight", "selection_fraction")
ROUND_KEYS = (
    "round",
    "num_clients",
    "num_malicious",
    "num_coordinates",
    "attacker_selection_fraction",
    "equal_share",
    "weight_share",
    "displacement_norm",
    "displacement_cosine",
)


def weighted_median_selection(
    flat: np.ndarray, w: np.ndarray, total: float
) -> np.ndarray:
    """Row selected by the weighted median down each column of ``flat``.

    Rows are ordered by value per column and the selected row is the one at
    which the cumulative weight first reaches half of ``total``. Returns an
    integer array of length ``flat.shape[1]``; a zero-weight row is never
    selected. ``flat[selection, arange(m)]`` is the weighted median itself.
    """
    half = total / 2.0
    order = np.argsort(flat, axis=0)
    cumw = np.cumsum(w[order], axis=0)
    idx = np.argmax(cumw >= half - 1e-12, axis=0)
    return order[idx, np.arange(flat.shape[1])]


def _flat_layer(updates: Sequence[ClientUpdate], layer: int) -> np.ndarray:
    return np.stack([np.asarray(u.parameters[layer]) for u in updates], axis=0).reshape(
        len(updates), -1
    )


class MedianSelectionRecorder:
    """Per-client and per-round selection statistics of a median rule.

    ``malicious_clients`` is the ground-truth set the statistics are split
    on. :meth:`record` is called once per aggregated round with the inputs the
    rule combined and the aggregate it produced; the rule's own output is
    never touched.
    """

    def __init__(self, malicious_clients: frozenset[ClientId]) -> None:
        self.malicious_clients = malicious_clients
        self.client_rows: list[dict[str, Any]] = []
        self.round_rows: list[dict[str, Any]] = []

    def record(
        self,
        round_num: int,
        updates: Sequence[ClientUpdate],
        global_parameters: NDArrays,
        aggregated: NDArrays,
        weights: np.ndarray,
        rule: MedianRule,
    ) -> None:
        """Record one round.

        ``weights`` is aligned to ``updates``. For ``"weighted_median"`` it is
        the weight vector the rule combined with (the selection depends only
        on the ratios); for ``"median"`` a positive entry marks an included
        row and the included rows take equal weight, as the rule does. When
        no entry is positive every row is included.
        """
        n = len(updates)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights shape {w.shape} does not match {n} updates")
        if rule == "median":
            included = w > 0.0
            if not included.any():
                included = np.ones(n, dtype=bool)
            w = included.astype(np.float64)
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("median selection requires positive total weight")
        realised = w / total
        is_mal = np.array([u.client_id in self.malicious_clients for u in updates], dtype=bool)
        honest = (~is_mal) & (w > 0.0)

        counts = np.zeros(n, dtype=np.int64)
        num_coords = 0
        displacement_parts: list[np.ndarray] = []
        mal_delta_parts: list[np.ndarray] = []
        for layer in range(len(updates[0].parameters)):
            flat = _flat_layer(updates, layer)
            if rule == "weighted_median":
                selected = weighted_median_selection(flat, w, total)
            else:
                rows = np.flatnonzero(w > 0.0)
                sub = flat[rows]
                selected = rows[
                    weighted_median_selection(
                        sub, np.ones(len(rows), dtype=np.float64), float(len(rows))
                    )
                ]
            counts += np.bincount(selected, minlength=n)
            num_coords += flat.shape[1]

            agg = np.asarray(aggregated[layer], dtype=np.float64).ravel()
            if honest.any():
                sub = flat[honest]
                if rule == "weighted_median":
                    wh = w[honest] / float(w[honest].sum())
                    cols = np.arange(sub.shape[1])
                    honest_med = sub[weighted_median_selection(sub, wh, 1.0), cols]
                else:
                    honest_med = np.median(sub, axis=0)
                displacement_parts.append(agg - np.asarray(honest_med, dtype=np.float64))
            if is_mal.any():
                g = np.asarray(global_parameters[layer], dtype=np.float64).ravel()
                mal_delta_parts.append(flat[is_mal].astype(np.float64).mean(axis=0) - g)

        fractions = counts / float(num_coords)
        for u, m_flag, wt, frac in zip(updates, is_mal, realised, fractions, strict=True):
            self.client_rows.append(
                {
                    "round": round_num,
                    "client_id": u.client_id,
                    "is_malicious": bool(m_flag),
                    "weight": float(wt),
                    "selection_fraction": float(frac),
                }
            )

        displacement_norm = math.nan
        cosine = math.nan
        if displacement_parts:
            displacement = np.concatenate(displacement_parts)
            displacement_norm = float(np.linalg.norm(displacement))
            if mal_delta_parts:
                mal_mean = np.concatenate(mal_delta_parts)
                mal_norm = float(np.linalg.norm(mal_mean))
                if displacement_norm > 0.0 and mal_norm > 0.0:
                    cosine = float(displacement @ mal_mean / (displacement_norm * mal_norm))
        self.round_rows.append(
            {
                "round": round_num,
                "num_clients": n,
                "num_malicious": int(is_mal.sum()),
                "num_coordinates": num_coords,
                "attacker_selection_fraction": float(fractions[is_mal].sum()),
                "equal_share": float(is_mal.sum()) / n,
                "weight_share": float(realised[is_mal].sum()),
                "displacement_norm": displacement_norm,
                "displacement_cosine": cosine,
            }
        )

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        """The two tables, ready for JSON (``NaN`` is kept as ``NaN``)."""
        return {"clients": list(self.client_rows), "rounds": list(self.round_rows)}
