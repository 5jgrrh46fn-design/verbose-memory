"""Federation server: FedAvg, a trust-scored defense, or robust aggregation.

Each round samples K clients, broadcasts the current global parameters,
collects `ClientUpdate`s, and aggregates in one of four mutually-exclusive
modes:

* **FedAvg** (default) — sample-weighted average per McMahan et al. 2017.
* **Trust-scored defense** (``scorer`` + ``integrator``) — extract behavioural
  scores, turn them into per-client aggregation weights (and optionally a
  next-round sampling bias), then weighted-average.
* **Robust aggregation** (``aggregator``) — a fixed rule (Krum, median,
  trimmed mean) that combines updates from their geometry alone, or a
  history-keeping rule over client deltas (FoolsGold).
* **Server-rooted aggregation** (``server_update_fn``) — the server trains on a
  clean root set and combines client updates against its own delta (FLTrust,
  or FLTrust's trust on a weighted-median base).

A trust-scored defense can also read the server's root-set update: when its
scorer or integrator asks for it, the server trains on the root set each round
and hands the result over before scoring and integration.

Training terminates at the round cap or when an early-stop monitor on a named
evaluator metric plateaus. The loop is synchronous and in-process; Flower's
simulation runtime is not used — we go straight to the aggregation primitive
so the smoke test stays deterministic.

Every round also records a wall-clock breakdown (client training, scoring,
aggregation, server-side training, evaluation) on the ``RoundOutcome``; on a
CUDA device each block is bracketed by a device synchronisation so the
numbers reflect finished GPU work rather than kernel launch time. When
``FederationConfig.profile_memory`` is set, the server-side work of each round
(scoring, aggregation, server training) also records its peak memory. When
``FederationConfig.trace_capture`` is set, each round also records the capture
trace of :mod:`topfed.federation.instrumentation`, outside the timed blocks.
A :class:`~topfed.federation.median_selection.MedianSelectionRecorder` passed
to :func:`run_federation` likewise records, after each round of a median
rule, which client supplied each coordinate of the aggregate.
"""

from __future__ import annotations

import math
import random
import time
import tracemalloc
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from flwr.server.strategy.aggregate import aggregate

from topfed.aggregation import DeltaAggregator, coordinate_median_aggregate
from topfed.core.seeding import derive_round_seed, set_all_seeds
from topfed.core.types import (
    ClientFactory,
    ClientId,
    ClientUpdate,
    EvidenceObserver,
    FederationConfig,
    Integrator,
    NDArrays,
    RoundContext,
    RoundEvidence,
    RoundOutcome,
    RoundScorer,
    ServerEvaluator,
)
from topfed.federation.instrumentation import capture_trace
from topfed.federation.median_selection import (
    MedianRule,
    MedianSelectionRecorder,
    weighted_median_selection,
)
from topfed.federation.sampling import forced_co_selection_sample, uniform_sample


def _device_sync() -> None:
    """Block until queued GPU work has finished, if a CUDA device is in use.

    Imported lazily so this module stays importable without torch on the
    path of pure-numpy unit tests.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Stopwatch:
    """Accumulates named wall-clock blocks for one round."""

    def __init__(self) -> None:
        self.elapsed: dict[str, float] = {}

    def measure(self, name: str) -> "_Timer":
        return _Timer(self, name)


class _Timer:
    def __init__(self, watch: _Stopwatch, name: str) -> None:
        self._watch = watch
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        _device_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        _device_sync()
        self._watch.elapsed[self._name] = (
            self._watch.elapsed.get(self._name, 0.0)
            + time.perf_counter()
            - self._start
        )


class _PeakMemory:
    """Peak memory allocated inside a block, above the level at entry, in MiB.

    Host memory is traced with ``tracemalloc``, which sees NumPy buffers and
    Python objects but not PyTorch's CPU allocator. Device memory is the CUDA
    caching allocator's peak and is recorded only when a CUDA device is in use.
    A disabled probe records nothing and costs nothing.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.peaks: dict[str, float] = {}
        self._host_base = 0
        self._device_base = 0
        self._was_tracing = False
        self._cuda: Any = None

    def __enter__(self) -> "_PeakMemory":
        if not self.enabled:
            return self
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency
            pass
        else:
            if torch.cuda.is_available():
                self._cuda = torch.cuda
                torch.cuda.synchronize()
                self._device_base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
        self._was_tracing = tracemalloc.is_tracing()
        if not self._was_tracing:
            tracemalloc.start()
        self._host_base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        return self

    def __exit__(self, *exc: object) -> None:
        if not self.enabled:
            return
        host_peak = tracemalloc.get_traced_memory()[1]
        if not self._was_tracing:
            tracemalloc.stop()
        self.peaks["server_peak_host_mb"] = max(host_peak - self._host_base, 0) / 2**20
        if self._cuda is not None:
            self._cuda.synchronize()
            device_peak = self._cuda.max_memory_allocated()
            self.peaks["server_peak_device_mb"] = (
                max(device_peak - self._device_base, 0) / 2**20
            )


MEMORY_KEYS = ("server_peak_host_mb", "server_peak_device_mb")

TIMING_KEYS = (
    "client_train_s",
    "scoring_s",
    "aggregation_s",
    "server_train_s",
    "evaluation_s",
    "round_s",
)


def fedavg_aggregate(updates: list[ClientUpdate]) -> NDArrays:
    """Sample-weighted average over client parameters."""
    if not updates:
        raise ValueError("fedavg_aggregate requires at least one ClientUpdate")
    return aggregate([(u.parameters, u.num_examples) for u in updates])


def weighted_aggregate(
    updates: list[ClientUpdate], weights: np.ndarray
) -> NDArrays:
    """Average client parameters under explicit per-client weights.

    ``weights`` is aligned to ``updates`` and need not be normalised; a zero
    weight excludes that client. Unlike :func:`fedavg_aggregate`, sample counts
    are not used — the integrator's weights fully determine each client's
    contribution.
    """
    if not updates:
        raise ValueError("weighted_aggregate requires at least one ClientUpdate")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(updates),):
        raise ValueError(
            f"weights shape {w.shape} does not match {len(updates)} updates"
        )
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weighted_aggregate requires positive total weight")
    w = w / total
    num_layers = len(updates[0].parameters)
    return [
        sum(w[i] * updates[i].parameters[layer] for i in range(len(updates)))
        for layer in range(num_layers)
    ]


def weighted_coordinate_median(
    updates: list[ClientUpdate], weights: np.ndarray
) -> NDArrays:
    """Per-coordinate weighted median across clients.

    For each coordinate, clients are ordered by value and the result is the
    value at which the cumulative client weight first reaches half of the total.
    Down-weighting a client pulls the median away from it, but a client cannot
    move the median by an extreme value alone — only by holding (with everything
    on its side of the order) a majority of the weight — so an over-trusted
    client with a benign-magnitude update stays bounded. With equal weights this
    is the (lower) coordinate-wise median.
    """
    if not updates:
        raise ValueError(
            "weighted_coordinate_median requires at least one ClientUpdate"
        )
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(updates),):
        raise ValueError(
            f"weights shape {w.shape} does not match {len(updates)} updates"
        )
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError(
            "weighted_coordinate_median requires positive total weight"
        )
    out: NDArrays = []
    for layer in range(len(updates[0].parameters)):
        stack = np.stack([u.parameters[layer] for u in updates], axis=0)
        shape = stack.shape
        med = _weighted_median_rows(stack.reshape(shape[0], -1), w, total)
        out.append(med.reshape(shape[1:]))
    return out


def _weighted_median_rows(
    flat: np.ndarray, w: np.ndarray, total: float
) -> np.ndarray:
    """Weighted median down the rows of an ``(n, m)`` array, per column.

    The result is the value at which the cumulative weight, in value order,
    first reaches half of ``total``. A zero-weight row is never selected. The
    row chosen per column is :func:`weighted_median_selection`'s.
    """
    return flat[weighted_median_selection(flat, w, total), np.arange(flat.shape[1])]


def _weighted_sample(
    client_ids: tuple[ClientId, ...],
    k: int,
    probs: dict[ClientId, float],
    rng: np.random.Generator,
) -> tuple[ClientId, ...]:
    """Sample ``k`` clients without replacement, biased by ``probs``.

    Clients absent from ``probs`` (not yet observed) take the mean of the
    observed biases so they keep a baseline chance of selection. When fewer
    than ``k`` clients have a positive bias (a score with exact zeros leaves
    a client at zero until it is sampled again), every one of them is
    selected and the rest of the cohort is drawn uniformly from the others.
    """
    observed = list(probs.values())
    default = float(np.mean(observed)) if observed else 1.0
    raw = np.array(
        [max(probs.get(c, default), 0.0) for c in client_ids], dtype=np.float64
    )
    if raw.sum() <= 0.0:
        raw = np.ones(len(client_ids), dtype=np.float64)
    positive = np.flatnonzero(raw > 0.0)
    if positive.size < k:
        others = np.setdiff1d(np.arange(len(client_ids)), positive)
        fill = rng.choice(others, size=k - positive.size, replace=False)
        return tuple(client_ids[i] for i in (*positive, *fill))
    p = raw / raw.sum()
    idx = rng.choice(len(client_ids), size=k, replace=False, p=p)
    return tuple(client_ids[i] for i in idx)


def _fltrust_normalised(
    updates: list[ClientUpdate],
    global_parameters: NDArrays,
    server_parameters: NDArrays,
    eps: float,
) -> tuple[float, list[float], list[np.ndarray]] | None:
    """FLTrust's per-client trust and magnitude-normalised deltas.

    Returns ``(server_norm, trust, scaled)`` where ``trust[i]`` is the ReLU of
    the cosine between client ``i``'s delta and the server delta and
    ``scaled[i]`` is that delta rescaled to the server delta's norm; ``None``
    when the server did not move.
    """
    def _flat(arrs: NDArrays) -> np.ndarray:
        return np.concatenate([np.asarray(a).ravel() for a in arrs])

    g0 = _flat(
        [sp - gp for sp, gp in zip(server_parameters, global_parameters, strict=True)]
    )
    n0 = float(np.linalg.norm(g0))
    if n0 <= eps:
        return None
    trust: list[float] = []
    scaled: list[np.ndarray] = []
    for u in updates:
        di = _flat(
            [w - g for w, g in zip(u.parameters, global_parameters, strict=True)]
        )
        ni = float(np.linalg.norm(di))
        cos = float(np.dot(di, g0) / (ni * n0)) if ni > eps else 0.0
        trust.append(max(cos, 0.0))
        scaled.append((n0 / ni) * di if ni > eps else np.zeros_like(di))
    return n0, trust, scaled


def _add_flat(global_parameters: NDArrays, flat_delta: np.ndarray) -> NDArrays:
    out: NDArrays = []
    idx = 0
    for gp in global_parameters:
        size = gp.size
        out.append(gp + flat_delta[idx : idx + size].reshape(gp.shape))
        idx += size
    return out


def fltrust_trust(
    updates: list[ClientUpdate],
    global_parameters: NDArrays,
    server_parameters: NDArrays,
    eps: float = 1e-12,
) -> np.ndarray:
    """FLTrust's per-client trust: ReLU of the cosine to the server delta.

    All zero when the server did not move. Used to log the weights a
    server-rooted rule gave each client; the rules compute it themselves.
    """
    def _flat(arrs: NDArrays) -> np.ndarray:
        return np.concatenate([np.asarray(a).ravel() for a in arrs])

    g0 = _flat(
        [sp - gp for sp, gp in zip(server_parameters, global_parameters, strict=True)]
    )
    n0 = float(np.linalg.norm(g0))
    trust = np.zeros(len(updates), dtype=np.float64)
    if n0 <= eps:
        return trust
    for i, u in enumerate(updates):
        di = _flat(
            [w - g for w, g in zip(u.parameters, global_parameters, strict=True)]
        )
        ni = float(np.linalg.norm(di))
        if ni > eps:
            trust[i] = max(float(np.dot(di, g0) / (ni * n0)), 0.0)
    return trust


def _normalised(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    return w / total if total > 0.0 else np.zeros_like(w)


def weighted_geometric_median(
    updates: list[ClientUpdate],
    weights: np.ndarray,
    nu: float = 1e-6,
    max_iter: int = 10,
    ftol: float = 1e-6,
) -> NDArrays:
    """Weighted geometric median of the client updates, by smoothed Weiszfeld.

    The geometric median minimises the weighted sum of Euclidean distances to
    the clients' parameter vectors; unlike the coordinate-wise median it
    treats an update as one point. Starting from the weighted mean, each
    iteration re-weights client ``i`` by ``w_i / max(nu, ||z - x_i||)`` and
    takes the weighted mean again, for at most ``max_iter`` iterations or
    until the objective changes by less than ``ftol`` of its value.

    This is the aggregation of RFA (Pillutla, Kakade and Harchaoui, "Robust
    Aggregation for Federated Learning", IEEE TSP 2022), following
    ``ServerModel.geometric_median_update`` in the authors' code
    (https://github.com/krishnap25/RFA, commit
    40945d27a5246b2c82fd37bd453db4f5bc049952, ``models/model.py``,
    BSD-2-Clause): the same initial point, smoothing, re-weighting and
    stopping rule, with the client weights supplied by the caller instead of
    the sample counts.
    """
    if not updates:
        raise ValueError("weighted_geometric_median requires at least one ClientUpdate")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(updates),):
        raise ValueError(f"weights shape {w.shape} does not match {len(updates)} updates")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weighted_geometric_median requires positive total weight")
    if nu <= 0.0 or max_iter < 0:
        raise ValueError(f"need nu > 0 and max_iter >= 0; got nu={nu}, max_iter={max_iter}")
    alphas = w / total
    points = np.stack(
        [
            np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in u.parameters])
            for u in updates
        ]
    )

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        dist = np.linalg.norm(points - z, axis=1)
        return float(alphas @ dist), dist

    median = alphas @ points
    value, dist = objective(median)
    for _ in range(max_iter):
        beta = alphas / np.maximum(nu, dist)
        beta = beta / beta.sum()
        median = beta @ points
        previous = value
        value, dist = objective(median)
        if abs(previous - value) < ftol * value:
            break
    out: NDArrays = []
    offset = 0
    for ref in updates[0].parameters:
        ref_arr = np.asarray(ref)
        out.append(
            median[offset : offset + ref_arr.size].reshape(ref_arr.shape).astype(ref_arr.dtype)
        )
        offset += ref_arr.size
    return out


def _realised_weights(weights: np.ndarray, defended_base: str) -> np.ndarray:
    """The per-client weights a defended base actually aggregates with.

    The ``median`` base only uses the integrator to exclude clients and then
    takes an equal-weight median over the rest.
    """
    if defended_base == "median":
        included = (np.asarray(weights) > 0.0).astype(np.float64)
        if not included.any():
            included = np.ones_like(included)
        return _normalised(included)
    return _normalised(weights)


def fltrust_aggregate(
    updates: list[ClientUpdate],
    global_parameters: NDArrays,
    server_parameters: NDArrays,
    eps: float = 1e-12,
) -> NDArrays:
    """FLTrust aggregation (Cao et al. 2021).

    Uses the server's own update, trained on a small clean root set, as a trusted
    reference direction. Each client is trusted by the ReLU of the cosine
    similarity between its delta and the server delta, and its magnitude is
    rescaled to the server delta's norm; the new global model is the
    trust-weighted average of the rescaled deltas. When no client is trusted (or
    the server did not move), the global model is left unchanged.
    """
    found = _fltrust_normalised(updates, global_parameters, server_parameters, eps)
    if found is None:
        return [np.array(g) for g in global_parameters]
    _, trust, scaled = found
    tsum = float(sum(trust))
    if tsum <= eps:
        return [np.array(g) for g in global_parameters]
    agg = np.zeros_like(scaled[0])
    for t, sv in zip(trust, scaled, strict=True):
        agg += t * sv
    agg /= tsum
    return _add_flat(global_parameters, agg)


def fltrust_median_aggregate(
    updates: list[ClientUpdate],
    global_parameters: NDArrays,
    server_parameters: NDArrays,
    eps: float = 1e-12,
    cap: float | None = None,
) -> NDArrays:
    """FLTrust's trust and magnitude normalisation on a weighted-median base.

    Same per-client trust (ReLU of the cosine to the server delta) and the same
    rescaling of every delta to the server delta's norm as
    :func:`fltrust_aggregate`, but the rescaled deltas are combined by a
    per-coordinate trust-weighted median instead of a trust-weighted mean, so a
    trusted client moves a coordinate only together with a weight majority.
    Untrusted clients (trust 0) are never selected. When no client is trusted
    (or the server did not move), the global model is left unchanged.

    With ``cap`` set, the trust shares are first normalised to sum to one,
    each is then ceiled at ``cap / n`` (``cap = 1`` is an equal share) and the
    vector is renormalised before the median, so a trusted client holds a
    weight majority on its own only when fewer than two others are trusted.
    ``None`` keeps the raw trust weights.
    """
    found = _fltrust_normalised(updates, global_parameters, server_parameters, eps)
    if found is None:
        return [np.array(g) for g in global_parameters]
    _, trust, scaled = found
    w = np.asarray(trust, dtype=np.float64)
    tsum = float(w.sum())
    if tsum <= eps:
        return [np.array(g) for g in global_parameters]
    if cap is not None:
        if cap <= 0:
            raise ValueError(f"cap must be positive; got {cap}")
        w = np.minimum(w / tsum, cap / len(w))
        tsum = float(w.sum())
        if tsum <= eps:
            return [np.array(g) for g in global_parameters]
    med = _weighted_median_rows(np.stack(scaled, axis=0), w, tsum)
    return _add_flat(global_parameters, med)


RootedAggregator = Callable[[list[ClientUpdate], NDArrays, NDArrays], NDArrays]


def _defended_aggregate(
    updates: list[ClientUpdate],
    weights: np.ndarray,
    defended_base: str,
    geometric_median_options: Mapping[str, Any] | None = None,
) -> NDArrays:
    """Realise an integrator's weights under the configured defended base."""
    if defended_base == "geometric_median":
        # Trust as a bounded weight inside the other robust estimator: one
        # point per client, a weighted sum of distances instead of a
        # per-coordinate order statistic. Zero-weight clients drop out.
        keep = [i for i, w in enumerate(weights) if w > 0.0]
        if keep:
            kept = np.asarray([weights[i] for i in keep], dtype=np.float64)
        else:
            keep = list(range(len(updates)))
            kept = np.ones(len(keep), dtype=np.float64)
        options = dict(geometric_median_options or {})
        return weighted_geometric_median(
            [updates[i] for i in keep],
            kept,
            nu=float(options.get("nu", 1e-6)),
            max_iter=int(options.get("max_iter", 10)),
        )
    if defended_base == "median":
        # Trust used subtractively: drop the integrator's zero-weighted
        # clients, then combine the survivors by a coordinate-wise median so
        # no single included client can be amplified beyond its structural
        # share.
        included = [
            u for u, w in zip(updates, weights, strict=True) if w > 0.0
        ]
        if not included:
            included = updates
        return coordinate_median_aggregate(included)
    if defended_base == "weighted_median":
        # Trust used as a soft, bounded down-weight inside a robust estimator:
        # every client is retained (no cohort shrinkage), but low-trust
        # clients pull the per-coordinate median less. An over-trusted client
        # still cannot move the median by value alone without holding a
        # weight majority.
        return weighted_coordinate_median(updates, weights)
    return weighted_aggregate(updates, weights)


def run_federation(
    config: FederationConfig,
    client_factory: ClientFactory,
    initial_parameters: NDArrays,
    evaluator: ServerEvaluator | None = None,
    *,
    scorer: RoundScorer | None = None,
    integrator: Integrator | None = None,
    defended_base: str = "mean",
    geometric_median_options: Mapping[str, Any] | None = None,
    aggregator: Callable[[list[ClientUpdate]], NDArrays] | DeltaAggregator | None = None,
    server_update_fn: Callable[[RoundContext, FederationConfig], NDArrays]
    | None = None,
    rooted_aggregator: RootedAggregator = fltrust_aggregate,
    malicious_clients: frozenset[ClientId] | None = None,
    verbose: bool = True,
    on_round: Callable[[RoundOutcome], None] | None = None,
    retain_parameters: bool = True,
    median_selection: MedianSelectionRecorder | None = None,
) -> list[RoundOutcome]:
    """Execute the server loop in one of four mutually-exclusive modes.

    Plain FedAvg by default; a trust-scored defense when ``scorer`` and
    ``integrator`` are given; or a fixed robust-aggregation rule when
    ``aggregator`` is given.

    For a trust-scored defense, ``defended_base`` selects how the integrator's
    decision is realised: ``"mean"`` takes the integrator's weighted mean of all
    updates; ``"median"`` uses the integrator only to *exclude* clients (those it
    zero-weights) and combines the survivors by a bounded-influence coordinate
    median, so trust can suppress but never amplify a client.

    ``server_update_fn`` trains the server's own model on its clean root set
    for one round. Without a defense it drives a server-rooted aggregation
    rule, ``rooted_aggregator`` (FLTrust by default), which combines the
    updates against the server's delta. With a defense it is required exactly
    when the scorer or the integrator is an
    :class:`~topfed.core.types.EvidenceObserver` that asks for the server
    update, and it is then delivered to them in the round's evidence. An
    integrator can decline to trust anyone (``keep_global``), in which case the
    round carries the global model forward unchanged.

    ``aggregator`` is either a stateless rule over the round's updates or a
    :class:`~topfed.aggregation.DeltaAggregator`, which also receives the
    round number and the global parameters and reports its weights.

    ``malicious_clients`` is consulted only by the forced co-selection sampler
    (``config.force_malicious_per_round``) and by the capture trace
    (``config.trace_capture``), which runs after aggregation; the aggregation
    path never sees it, so a defense cannot accidentally read ground truth
    through the server.

    Every outcome carries the per-client weights the round aggregated with
    when the rule has them (FedAvg's sample shares, a defense's realised
    weights, a delta aggregator's weights). A server-rooted rule's trust is
    only computed for this when the run traces capture, since it costs an
    extra pass over the parameters.

    ``on_round`` is called with each round's outcome as soon as the round
    completes, before the next round starts. With ``retain_parameters=False``
    the returned outcomes drop their parameter arrays (the aggregated model and
    every client update's parameters become empty lists) once ``on_round`` has
    seen them, so a long run's memory no longer grows with every round's
    updates; consume the parameters in ``on_round`` instead.

    ``median_selection`` records, after every aggregated round, which client
    supplied each coordinate of the aggregate. It needs a median rule: the
    ``"median"`` or ``"weighted_median"`` defended base, or the plain
    coordinate median as ``aggregator``. It only reads the round's inputs and
    output, so the aggregate is the same with and without it.
    """
    if config.clients_per_round > config.num_clients:
        raise ValueError("clients_per_round cannot exceed num_clients")
    forced_m = config.force_malicious_per_round
    if forced_m is not None and malicious_clients is None:
        raise ValueError(
            "force_malicious_per_round requires the malicious client set"
        )
    if config.trace_capture and malicious_clients is None:
        raise ValueError("trace_capture requires the malicious client set")
    if defended_base not in ("mean", "median", "weighted_median", "geometric_median"):
        raise ValueError(
            "defended_base must be 'mean', 'median', 'weighted_median' or "
            f"'geometric_median'; got {defended_base}"
        )
    defended = scorer is not None and integrator is not None
    scorer_observer = scorer if isinstance(scorer, EvidenceObserver) else None
    integrator_observer = (
        integrator if isinstance(integrator, EvidenceObserver) else None
    )
    needs_server_update = defended and any(
        o is not None and o.requires_server_update
        for o in (scorer_observer, integrator_observer)
    )
    if needs_server_update and server_update_fn is None:
        raise ValueError(
            "the defense needs the server's root-set update; pass server_update_fn"
        )
    if aggregator is not None and defended:
        raise ValueError(
            "a fixed aggregation rule is mutually exclusive with scorer/integrator"
        )
    if server_update_fn is not None and defended and not needs_server_update:
        raise ValueError(
            "server_update_fn with a defense is only for components that need "
            "the server update"
        )
    if aggregator is not None and server_update_fn is not None:
        raise ValueError("aggregator and server_update_fn are mutually exclusive")
    median_rule: MedianRule | None = None
    if median_selection is not None:
        if defended and defended_base in ("median", "weighted_median"):
            median_rule = "median" if defended_base == "median" else "weighted_median"
        elif aggregator is coordinate_median_aggregate:
            median_rule = "median"
        else:
            raise ValueError(
                "median_selection needs a median rule: the 'median' or "
                "'weighted_median' defended base, or the coordinate median"
            )

    set_all_seeds(config.seed)
    sampler_rng = random.Random(config.seed)
    sampling_rng = np.random.default_rng(config.seed)

    all_client_ids: tuple[ClientId, ...] = tuple(
        f"c{i}" for i in range(config.num_clients)
    )
    global_parameters = initial_parameters
    outcomes: list[RoundOutcome] = []
    best_monitor = -math.inf
    rounds_since_improvement = 0
    next_sampling_probs: dict[ClientId, float] | None = None

    for round_num in range(1, config.num_rounds + 1):
        round_start = time.perf_counter()
        watch = _Stopwatch()
        if forced_m is not None:
            if next_sampling_probs is not None:
                raise ValueError(
                    "forced co-selection cannot be combined with an integrator "
                    "that biases next-round sampling"
                )
            assert malicious_clients is not None
            selected = forced_co_selection_sample(
                sampler_rng,
                all_client_ids,
                malicious_clients,
                config.clients_per_round,
                forced_m,
            )
        elif next_sampling_probs is None:
            selected = uniform_sample(
                sampler_rng, all_client_ids, config.clients_per_round
            )
        else:
            selected = _weighted_sample(
                all_client_ids,
                config.clients_per_round,
                next_sampling_probs,
                sampling_rng,
            )
        # A rule with a pre-round step (BackdoorIndicator plants its indicator
        # task) rewrites the model the clients receive; the round's base for
        # deltas and for the aggregate is that broadcast model.
        prepare_round = getattr(aggregator, "prepare_round", None)
        if callable(prepare_round):
            with watch.measure("server_train_s"):
                global_parameters = prepare_round(round_num, global_parameters)

        ctx = RoundContext(
            round_num=round_num,
            total_rounds=config.num_rounds,
            global_parameters=global_parameters,
            selected_clients=selected,
            seed=derive_round_seed(config.seed, round_num),
        )

        updates: list[ClientUpdate] = []
        with watch.measure("client_train_s"):
            for cid in selected:
                client = client_factory(cid)
                updates.append(client.local_train(ctx, config))

        # Server-side work only: client training and evaluation are common to
        # every scheme and stay outside the probe.
        round_weights: np.ndarray | None = None
        with _PeakMemory(config.profile_memory) as probe:
            if defended:
                assert scorer is not None and integrator is not None
                server_params: NDArrays | None = None
                if needs_server_update:
                    assert server_update_fn is not None
                    with watch.measure("server_train_s"):
                        server_params = server_update_fn(ctx, config)
                with watch.measure("scoring_s"):
                    if scorer_observer is not None:
                        scorer_observer.observe(
                            RoundEvidence(
                                round_num=round_num,
                                updates=tuple(updates),
                                global_parameters=global_parameters,
                                server_parameters=server_params,
                            )
                        )
                    client_ids, scores = scorer.score_round(
                        round_num, updates, global_parameters
                    )
                    if client_ids != tuple(u.client_id for u in updates):
                        raise ValueError(
                            "scorer returned client ids out of update order"
                        )
                    if integrator_observer is not None:
                        integrator_observer.observe(
                            RoundEvidence(
                                round_num=round_num,
                                updates=tuple(updates),
                                global_parameters=global_parameters,
                                server_parameters=server_params,
                                metrics=getattr(scorer, "last_batch", None),
                            )
                        )
                    result = integrator.integrate(round_num, client_ids, scores)
                with watch.measure("aggregation_s"):
                    if result.keep_global:
                        aggregated = [np.array(g) for g in global_parameters]
                    else:
                        aggregated = _defended_aggregate(
                            updates, result.weights, defended_base, geometric_median_options
                        )
                next_sampling_probs = result.sampling_probs
                round_weights = (
                    np.zeros(len(updates))
                    if result.keep_global
                    else _realised_weights(result.weights, defended_base)
                )
            elif server_update_fn is not None:
                # Server-rooted rule (FLTrust and variants): the server
                # computes its own update on a clean root set and uses it as
                # the trusted reference direction.
                with watch.measure("server_train_s"):
                    rooted_params = server_update_fn(ctx, config)
                with watch.measure("aggregation_s"):
                    aggregated = rooted_aggregator(
                        updates, global_parameters, rooted_params
                    )
            elif isinstance(aggregator, DeltaAggregator):
                with watch.measure("aggregation_s"):
                    aggregated = aggregator.aggregate(
                        round_num, updates, global_parameters
                    )
                if aggregator.last_weights is not None:
                    round_weights = np.asarray(aggregator.last_weights, dtype=np.float64)
            elif aggregator is not None:
                with watch.measure("aggregation_s"):
                    aggregated = aggregator(updates)
            else:
                with watch.measure("aggregation_s"):
                    aggregated = fedavg_aggregate(updates)
                round_weights = _normalised(
                    np.array([u.num_examples for u in updates], dtype=np.float64)
                )

        if median_rule is not None:
            assert median_selection is not None
            if defended:
                if not result.keep_global:
                    median_selection.record(
                        round_num, updates, global_parameters, aggregated,
                        result.weights, median_rule,
                    )
            else:
                median_selection.record(
                    round_num, updates, global_parameters, aggregated,
                    np.ones(len(updates), dtype=np.float64), median_rule,
                )

        trace: dict[str, float] = {}
        if config.trace_capture:
            assert malicious_clients is not None
            if server_update_fn is not None and not defended:
                round_weights = _normalised(
                    fltrust_trust(updates, global_parameters, rooted_params)
                )
            trace = capture_trace(
                updates, global_parameters, aggregated, malicious_clients, round_weights
            )

        with watch.measure("evaluation_s"):
            metrics = (
                evaluator.evaluate(aggregated) if evaluator is not None else {}
            )

        watch.elapsed["round_s"] = time.perf_counter() - round_start
        timings = {k: float(watch.elapsed.get(k, 0.0)) for k in TIMING_KEYS}
        outcome = RoundOutcome(
            round_num=round_num,
            aggregated_parameters=aggregated,
            metrics=metrics,
            client_updates=tuple(updates),
            timings=timings,
            memory=dict(probe.peaks),
            weights=round_weights,
            trace=trace,
        )
        if on_round is not None:
            on_round(outcome)
        if not retain_parameters:
            outcome = replace(
                outcome,
                aggregated_parameters=[],
                client_updates=tuple(
                    replace(u, parameters=[]) for u in outcome.client_updates
                ),
            )
        outcomes.append(outcome)
        global_parameters = aggregated

        if verbose:
            elapsed = timings["round_s"]
            train_loss = sum(
                u.train_loss * u.num_examples for u in updates
            ) / max(sum(u.num_examples for u in updates), 1)
            train_acc = sum(
                u.train_accuracy * u.num_examples for u in updates
            ) / max(sum(u.num_examples for u in updates), 1)
            parts = [
                f"[round {round_num:>3}/{config.num_rounds}]",
                f"{elapsed:5.1f}s",
                f"train_loss={train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
            ]
            parts.extend(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            print(" ".join(parts), flush=True)

        if config.early_stop_patience > 0:
            monitor = metrics.get(config.early_stop_metric)
            if monitor is not None:
                if monitor > best_monitor:
                    best_monitor = monitor
                    rounds_since_improvement = 0
                else:
                    rounds_since_improvement += 1
                if rounds_since_improvement >= config.early_stop_patience:
                    if verbose:
                        print(
                            f"[early-stop] {config.early_stop_metric} has not "
                            f"improved for {rounds_since_improvement} rounds "
                            f"(best={best_monitor:.4f}); stopping at round "
                            f"{round_num}.",
                            flush=True,
                        )
                    break

    return outcomes
