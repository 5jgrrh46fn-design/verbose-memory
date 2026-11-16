"""FLAME: cluster, clip and noise (Nguyen et al., USENIX Security 2022).

Provenance. FLAME has no official code release. This module follows the
BackFed benchmark's port (github.com/thinh-dao/BackFed,
``backfed/servers/anomaly_detection/flame_server.py`` at commit
``c851ce90373e``, 2026-03-29) step for step, re-expressed in NumPy over this
repository's ``ClientUpdate`` type:

* clustering runs on the flattened client *deltas* ``W_i - G`` (the paper's
  Algorithm 1 writes the cosine distance between the local models ``W_i``;
  since every ``W_i`` shares ``G`` the delta form is what the text describes
  and what public ports use);
* HDBSCAN with ``min_cluster_size = n // 2 + 1``, ``min_samples = 1`` and a
  single cluster allowed, on the cosine distance matrix;
* the largest cluster is admitted; when HDBSCAN finds no cluster every
  client is admitted (the port's choice; the paper is silent);
* ``S_t`` is the median L2 norm over *all* ``n`` deltas, admitted or not;
  each admitted delta is scaled by ``min(1, S_t / e_i)``;
* the admitted clipped deltas are averaged with equal weight and Gaussian
  noise ``N(0, (lambda * S_t)^2)`` is added to every coordinate of the new
  global model.

Two deviations from the port, both deliberate: HDBSCAN comes from
scikit-learn (``sklearn.cluster.HDBSCAN``) rather than the ``hdbscan``
package, and the noise is drawn from a seeded NumPy generator so runs are
reproducible. The paper's noise factor ``lambda = 0.001`` for image
classification (derived from ``epsilon = 3705``) is the default.

The models here carry no normalisation buffers (the CNNs have no
normalisation layers and the ResNet uses GroupNorm, whose state dict holds
weights and biases only), so the port's "trainable parameters only" filter is
a no-op; every state-dict array is included.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN

from topfed.core.types import ClientUpdate, NDArrays

NO_CLUSTER = -1


def _flatten(parameters: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parameters])


def cosine_distance_matrix(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise cosine distance ``1 - cos`` with a zero diagonal.

    A zero vector has distance 1 to everything (cosine 0), so a client that
    submitted no change never joins a cluster by accident.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] == 0:
        raise ValueError(f"vectors must be a non-empty 2-D array; got shape {v.shape}")
    norms = np.linalg.norm(v, axis=1)
    unit = np.where(norms[:, None] > eps, v / np.maximum(norms, eps)[:, None], 0.0)
    dist = 1.0 - unit @ unit.T
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, 2.0)


def flame_cluster(
    deltas: np.ndarray, min_cluster_size: int | None = None
) -> tuple[np.ndarray, list[int]]:
    """HDBSCAN labels over the deltas and the indices of the admitted clients.

    ``min_cluster_size`` defaults to ``n // 2 + 1``, so at most one cluster
    can form and it must hold a majority. Returns ``(labels, admitted)``;
    ``labels`` is HDBSCAN's output (``-1`` is noise). ``admitted`` is the
    largest cluster, or every client when no cluster forms.
    """
    n = deltas.shape[0]
    if n == 1:
        return np.array([0]), [0]
    size = n // 2 + 1 if min_cluster_size is None else int(min_cluster_size)
    size = max(2, min(size, n))
    dist = cosine_distance_matrix(deltas)
    clusterer = HDBSCAN(
        min_cluster_size=size,
        min_samples=1,
        metric="precomputed",
        allow_single_cluster=True,
        copy=True,
    )
    labels = np.asarray(clusterer.fit_predict(dist), dtype=int)
    if labels.max() < 0:
        return labels, list(range(n))
    members, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest = int(members[np.argmax(counts)])
    return labels, [i for i, lab in enumerate(labels) if lab == largest]


class FLAME:
    """FLAME aggregation: HDBSCAN filtering, adaptive clipping, adaptive noise.

    Stateless across rounds apart from the noise generator. ``last_weights``
    holds the round's aggregation weights (``1 / L`` for each of the ``L``
    admitted clients, 0 for the rejected) so the server loop can log them
    like any other rule. ``records`` holds one row per client per round
    (round, client id, cluster label, delta norm, clip factor, admitted) for
    offline analysis.
    """

    def __init__(
        self,
        noise_factor: float = 0.001,
        min_cluster_size: int | None = None,
        seed: int = 0,
    ) -> None:
        if noise_factor < 0:
            raise ValueError(f"noise_factor must be non-negative; got {noise_factor}")
        self.noise_factor = float(noise_factor)
        self.min_cluster_size = min_cluster_size
        self._rng = np.random.default_rng(seed)
        self.last_weights: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []

    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> NDArrays:
        if not updates:
            raise ValueError("FLAME requires at least one ClientUpdate")
        flat_global = _flatten(global_parameters)
        deltas = np.stack([_flatten(u.parameters) - flat_global for u in updates])
        n = deltas.shape[0]

        labels, admitted = flame_cluster(deltas, self.min_cluster_size)
        norms = np.linalg.norm(deltas, axis=1)
        bound = float(np.median(norms))
        with np.errstate(divide="ignore", invalid="ignore"):
            factors = np.where(norms > 0, np.minimum(1.0, bound / norms), 1.0)

        weights = np.zeros(n, dtype=np.float64)
        weights[admitted] = 1.0 / len(admitted)
        self.last_weights = weights
        admitted_set = set(admitted)
        for i, u in enumerate(updates):
            self.records.append(
                {
                    "round": round_num,
                    "client_id": u.client_id,
                    "cluster": int(labels[i]),
                    "delta_norm": float(norms[i]),
                    "clip_factor": float(factors[i]) if i in admitted_set else 0.0,
                    "admitted": i in admitted_set,
                    "clip_bound": bound,
                }
            )

        step = np.zeros_like(flat_global)
        for i in admitted:
            step += weights[i] * factors[i] * deltas[i]
        sigma = self.noise_factor * bound
        if sigma > 0:
            step += self._rng.normal(0.0, sigma, size=step.shape)

        out: NDArrays = []
        offset = 0
        for g in global_parameters:
            size = g.size
            out.append(
                (np.asarray(g) + step[offset : offset + size].reshape(g.shape)).astype(
                    np.asarray(g).dtype
                )
            )
            offset += size
        return out
