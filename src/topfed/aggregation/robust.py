"""Byzantine-robust aggregation rules.

Fixed-rule alternatives to sample-weighted averaging that resist a minority of
corrupted updates by construction, used here as defense baselines:

* :func:`krum_aggregate` — select the update closest to its nearest
  neighbours (or average the best few, "multi-Krum"); :class:`Krum` is the
  same rule and also reports the selection.
* :func:`coordinate_median_aggregate` — per-coordinate median across clients.
* :func:`trimmed_mean_aggregate` — per-coordinate mean after discarding the
  most extreme values at each end.
* :class:`FoolsGold` — down-weight clients whose accumulated update histories
  point the same way (Fung, Yoon and Beschastnikh, RAID 2020), the standard
  defense against Sybils and colluders.
* :class:`ClippedMean` — clip every delta to an L2 bound and average with
  equal weight (Sun et al. 2019), the cheapest magnitude bound and the
  control for whether bounding a mean already suffices.

These are aggregation rules, distinct from trust-score integration: they decide
how to combine updates from their geometry alone, without any per-client score.
The stateless rules take the round's client updates and return the aggregated
parameters; FoolsGold keeps per-client history across rounds and also needs the
round's global parameters (see :class:`DeltaAggregator`).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from topfed.core.types import ClientId, ClientUpdate, NDArrays


def _flatten(parameters: NDArrays) -> np.ndarray:
    return np.concatenate([p.ravel() for p in parameters])


def _mean_parameters(updates: list[ClientUpdate]) -> NDArrays:
    num_layers = len(updates[0].parameters)
    return [
        np.mean(
            np.stack([u.parameters[layer] for u in updates], axis=0), axis=0
        )
        for layer in range(num_layers)
    ]


def krum_select(
    updates: list[ClientUpdate],
    num_byzantine: int = 1,
    num_selected: int = 1,
) -> np.ndarray:
    """Indices of the updates Krum / multi-Krum selects, best first.

    Each update is scored by the sum of squared distances to its
    ``n - num_byzantine - 2`` nearest other updates; the ``num_selected``
    lowest-scoring updates are selected. Requires
    ``n - num_byzantine - 2 >= 1``.
    """
    n = len(updates)
    if n == 0:
        raise ValueError("krum_aggregate requires at least one ClientUpdate")
    neighbours = n - num_byzantine - 2
    if neighbours < 1:
        raise ValueError(
            f"Krum needs n - num_byzantine - 2 >= 1; got n={n}, "
            f"num_byzantine={num_byzantine}"
        )
    if not 1 <= num_selected <= n:
        raise ValueError(f"num_selected must be in [1, {n}]; got {num_selected}")

    vectors = [_flatten(u.parameters) for u in updates]
    sq_dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((vectors[i] - vectors[j]) ** 2))
            sq_dist[i, j] = d
            sq_dist[j, i] = d

    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        ordered = np.sort(sq_dist[i])
        # Index 0 is the zero self-distance; sum the closest `neighbours` others.
        scores[i] = float(np.sum(ordered[1 : 1 + neighbours]))

    return np.argsort(scores)[:num_selected]


def krum_aggregate(
    updates: list[ClientUpdate],
    num_byzantine: int = 1,
    num_selected: int = 1,
) -> NDArrays:
    """Krum / multi-Krum aggregation over flattened client updates.

    The updates :func:`krum_select` selects are averaged (``num_selected = 1``
    is classic Krum, returning a single client's parameters).
    """
    chosen = krum_select(updates, num_byzantine, num_selected)
    if num_selected == 1:
        return [p.copy() for p in updates[int(chosen[0])].parameters]
    return _mean_parameters([updates[int(i)] for i in chosen])


class Krum:
    """Krum / multi-Krum that reports which updates it selected.

    The aggregate is :func:`krum_aggregate`'s. ``last_weights`` is one-hot on
    the selected update for classic Krum and uniform over the selected updates
    for multi-Krum, so a run's ledger shows whose update each round took.
    """

    def __init__(self, num_byzantine: int = 1, num_selected: int = 1) -> None:
        self.num_byzantine = int(num_byzantine)
        self.num_selected = int(num_selected)
        self.last_weights: np.ndarray | None = None

    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> NDArrays:
        chosen = krum_select(updates, self.num_byzantine, self.num_selected)
        weights = np.zeros(len(updates), dtype=np.float64)
        weights[chosen] = 1.0 / len(chosen)
        self.last_weights = weights
        if self.num_selected == 1:
            return [p.copy() for p in updates[int(chosen[0])].parameters]
        return _mean_parameters([updates[int(i)] for i in chosen])


def coordinate_median_aggregate(updates: list[ClientUpdate]) -> NDArrays:
    """Per-coordinate median across client updates."""
    if not updates:
        raise ValueError("coordinate_median_aggregate requires updates")
    num_layers = len(updates[0].parameters)
    return [
        np.median(
            np.stack([u.parameters[layer] for u in updates], axis=0), axis=0
        )
        for layer in range(num_layers)
    ]


def trimmed_mean_aggregate(
    updates: list[ClientUpdate], trim_ratio: float = 0.1
) -> NDArrays:
    """Per-coordinate mean after trimming the extremes at each end.

    ``trim_ratio`` of the clients is dropped from each tail (floored to an
    integer count, and capped so at least one value always remains).
    """
    if not updates:
        raise ValueError("trimmed_mean_aggregate requires updates")
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError(f"trim_ratio must be in [0, 0.5); got {trim_ratio}")
    n = len(updates)
    k = min(int(np.floor(trim_ratio * n)), (n - 1) // 2)
    num_layers = len(updates[0].parameters)
    out: NDArrays = []
    for layer in range(num_layers):
        stacked = np.sort(
            np.stack([u.parameters[layer] for u in updates], axis=0), axis=0
        )
        kept = stacked[k : n - k] if k > 0 else stacked
        out.append(np.mean(kept, axis=0))
    return out


@runtime_checkable
class DeltaAggregator(Protocol):
    """An aggregation rule that reads client deltas and may keep state.

    Unlike the stateless rules, which combine the submitted parameters alone,
    it receives the round number and the global parameters the updates were
    trained from. ``last_weights`` holds the per-client aggregation weights of
    the most recent round (aligned to its updates, summing to one, all zero
    when the round kept the global model).
    """

    last_weights: np.ndarray | None

    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> NDArrays: ...


def history_similarity(
    histories: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine similarity of update histories, and each row's maximum.

    Returns ``(cs, max_cs)``: ``cs`` has a zero diagonal, a zero history has
    similarity 0 to everyone, and ``max_cs[i] = max_j cs[i, j]`` includes the
    diagonal, so it is never negative.
    """
    h = np.asarray(histories, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] == 0:
        raise ValueError(f"histories must be a non-empty 2-D array; got shape {h.shape}")
    norms = np.linalg.norm(h, axis=1)
    unit = np.where(norms[:, None] > eps, h / np.maximum(norms, eps)[:, None], 0.0)
    cs = unit @ unit.T
    np.fill_diagonal(cs, 0.0)
    return cs, cs.max(axis=1)


def foolsgold_weights(
    histories: np.ndarray, kappa: float = 1.0, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """FoolsGold per-client learning rates from accumulated update histories.

    ``histories`` has one row per client: the sum of every delta that client
    has submitted so far. Returns ``(alpha, max_similarity)``, both aligned to
    the rows. Clients whose histories are cosine-similar to another client's
    get a low ``alpha``; the steps are:

    1. pairwise cosine similarity ``cs`` of the histories with the diagonal set
       to 0, and ``v_i = max_j cs[i, j]`` (so ``v_i >= 0``);
    2. *pardoning*: when ``v_i < v_j``, ``cs[i, j]`` is scaled by ``v_i / v_j``,
       so an honest client that happens to resemble a Sybil is not punished as
       hard as the Sybil;
    3. ``alpha_i = 1 - max_j cs[i, j]``, clipped to ``[0, 1]`` and divided by
       the largest ``alpha``;
    4. a logit, ``kappa * (ln(alpha / (1 - alpha)) + 0.5)``, with 1 mapped to
       0.99 first, clipped to ``[0, 1]``.

    The diagonal stays in every row maximum, as in the authors' reference
    implementation. A zero history has similarity 0 to everyone. When every
    client is perfectly similar to another, all ``alpha`` are 0.
    """
    cs, max_cs = history_similarity(histories, eps)
    n = cs.shape[0]
    pardoned = cs.copy()
    for i in range(n):
        for j in range(n):
            if i != j and max_cs[i] < max_cs[j]:
                pardoned[i, j] = cs[i, j] * max_cs[i] / max_cs[j]
    alpha = np.clip(1.0 - pardoned.max(axis=1), 0.0, 1.0)
    top = float(alpha.max())
    if top <= eps:
        return np.zeros(n), max_cs
    alpha = alpha / top
    alpha[alpha >= 1.0] = 0.99
    with np.errstate(divide="ignore"):
        logit = kappa * (np.log(alpha / (1.0 - alpha)) + 0.5)
    return np.clip(np.nan_to_num(logit, neginf=0.0), 0.0, 1.0), max_cs


class FoolsGold:
    """FoolsGold aggregation (Fung, Yoon and Beschastnikh, RAID 2020).

    Keeps, for every client, the sum of all deltas it has submitted in the
    rounds it was selected (the current round included). Each round the
    selected clients' histories are compared pairwise and turned into
    per-client learning rates by :func:`foolsgold_weights`; the new global
    model is ``global + sum(alpha_i * delta_i) / sum(alpha_i)``, so a round of
    mutually dissimilar clients (every ``alpha`` = 1) is a plain mean of their
    deltas. When every ``alpha`` is 0 the global model is kept.

    Histories are kept in float32: one vector per client ever selected, so
    memory grows to ``num_clients x num_parameters`` floats.

    Clients that have always submitted similar updates, the signature of
    Sybils and of colluders sharing one objective, end up with ``alpha`` near
    0. A single attacker has no one to resemble and is not down-weighted.

    ``records`` holds one row per client per round (round, client id, maximum
    history similarity before pardoning, ``alpha``, and the normalised
    aggregation weight) for offline analysis.
    """

    def __init__(self, kappa: float = 1.0, eps: float = 1e-12) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive; got {kappa}")
        self.kappa = float(kappa)
        self.eps = float(eps)
        self._history: dict[ClientId, np.ndarray] = {}
        self.last_weights: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []

    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> NDArrays:
        if not updates:
            raise ValueError("FoolsGold requires at least one ClientUpdate")
        flat_global = _flatten(global_parameters)
        deltas = [_flatten(u.parameters) - flat_global for u in updates]
        for u, d in zip(updates, deltas, strict=True):
            prev = self._history.get(u.client_id)
            self._history[u.client_id] = d.astype(np.float32) if prev is None else prev + d
        histories = np.stack([self._history[u.client_id] for u in updates])
        alpha, max_cs = foolsgold_weights(histories, self.kappa, self.eps)
        total = float(alpha.sum())
        weights = alpha / total if total > self.eps else np.zeros_like(alpha)
        self.last_weights = weights
        for u, s, a, w in zip(updates, max_cs, alpha, weights, strict=True):
            self.records.append(
                {
                    "round": round_num,
                    "client_id": u.client_id,
                    "max_similarity": float(s),
                    "alpha": float(a),
                    "weight": float(w),
                }
            )
        if total <= self.eps:
            return [np.array(g) for g in global_parameters]
        step = np.zeros_like(deltas[0], dtype=np.float64)
        for w, d in zip(weights, deltas, strict=True):
            step += w * d
        out: NDArrays = []
        offset = 0
        for g in global_parameters:
            size = g.size
            out.append(g + step[offset : offset + size].reshape(g.shape).astype(g.dtype))
            offset += size
        return out


class ClippedMean:
    """Norm-clipped equal-weight mean of client deltas (Sun et al. 2019).

    Each delta ``local - global`` is scaled by ``min(1, bound / ||delta||)``
    and the clipped deltas are averaged with equal weight. ``bound=None``
    uses the round's median delta norm, so the bound adapts as updates shrink
    over training (the clipping stage of FLAME uses the same statistic); a
    number fixes the bound for the whole run. No client is ever excluded and
    no trust score is involved, so this rule isolates the effect of bounding
    update magnitude inside a mean.

    ``last_weights`` is uniform over the round's clients; ``records`` holds one
    row per client per round (round, client id, delta norm, clip factor, the
    bound used) for offline analysis.
    """

    def __init__(self, bound: float | None = None) -> None:
        if bound is not None and bound <= 0:
            raise ValueError(f"bound must be positive; got {bound}")
        self.bound = None if bound is None else float(bound)
        self.last_weights: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []

    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> NDArrays:
        if not updates:
            raise ValueError("ClippedMean requires at least one ClientUpdate")
        flat_global = _flatten(global_parameters)
        deltas = np.stack([_flatten(u.parameters) - flat_global for u in updates])
        norms = np.linalg.norm(deltas, axis=1)
        bound = float(np.median(norms)) if self.bound is None else self.bound
        with np.errstate(divide="ignore", invalid="ignore"):
            factors = np.where(norms > 0, np.minimum(1.0, bound / norms), 1.0)
        n = len(updates)
        self.last_weights = np.full(n, 1.0 / n)
        for u, nrm, f in zip(updates, norms, factors, strict=True):
            self.records.append(
                {
                    "round": round_num,
                    "client_id": u.client_id,
                    "delta_norm": float(nrm),
                    "clip_factor": float(f),
                    "clip_bound": bound,
                }
            )
        step = (factors[:, None] * deltas).mean(axis=0)
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
