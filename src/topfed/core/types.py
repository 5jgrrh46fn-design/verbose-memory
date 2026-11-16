"""Core types for the federated learning loop.

These types form the contract between server, client, and scoring /
integration components.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

NDArrays = list[np.ndarray]
ClientId = str
# One trust score per client, in [0, 1]; higher means more trustworthy.
TrustScores = np.ndarray


@dataclass(frozen=True, slots=True)
class ClientUpdate:
    """One client's contribution for a single round."""

    client_id: ClientId
    parameters: NDArrays
    num_examples: int
    train_loss: float
    train_accuracy: float

    def delta(self, global_parameters: NDArrays) -> NDArrays:
        return [w - g for w, g in zip(self.parameters, global_parameters, strict=True)]


@dataclass(frozen=True, slots=True)
class RoundContext:
    """Per-round state visible to clients and server hooks.

    TODO: add history of past rounds when the temporal CV metric (M3) needs it.
    """

    round_num: int
    total_rounds: int
    global_parameters: NDArrays
    selected_clients: tuple[ClientId, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """Server-side artifact for one completed round.

    ``metrics`` carries whatever the configured ``ServerEvaluator`` produced
    after aggregation — typically ``val_loss`` and ``val_accuracy`` on a
    held-out clean validation set, plus attack-specific quantities such as
    ``asr`` (attack success rate on a triggered evaluation set).

    ``timings`` holds the server's wall-clock breakdown of the round in
    seconds (``client_train_s``, ``scoring_s``, ``aggregation_s``,
    ``server_train_s``, ``evaluation_s``, ``round_s``). It is kept apart from
    ``metrics`` so evaluator outputs stay a pure model-quality record; the
    metrics writer merges both into the per-round ledger.

    ``memory`` holds the server-side peak memory in MiB
    (``server_peak_host_mb``, and ``server_peak_device_mb`` on a CUDA device)
    when the run profiles memory; it is empty otherwise.

    ``weights`` is the per-client aggregation weight the round actually used,
    aligned to ``client_updates`` and summing to one (all zero when the round
    kept the global model), or ``None`` when the rule has no per-client
    weights. ``trace`` holds the round's capture trace when the run traces
    capture; it is empty otherwise.

    TODO: persist `client_updates` to a parquet ledger instead of keeping them
    in-memory.
    """

    round_num: int
    aggregated_parameters: NDArrays
    metrics: dict[str, float]
    client_updates: tuple[ClientUpdate, ...]
    timings: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    weights: np.ndarray | None = None
    trace: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricBatch:
    """One round's behavioural metrics, one row per scored client.

    ``matrix`` has shape ``(len(client_ids), len(metric_names))``; row ``i``
    holds the metric values for ``client_ids[i]`` in the column order given by
    ``metric_names``. This is the only input a metric-based scorer receives,
    keeping scorers independent of how the metrics were produced or persisted.
    """

    client_ids: tuple[ClientId, ...]
    metric_names: tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D; got {self.matrix.ndim}-D")
        n, m = self.matrix.shape
        if n != len(self.client_ids):
            raise ValueError(
                f"matrix has {n} rows but {len(self.client_ids)} client ids"
            )
        if m != len(self.metric_names):
            raise ValueError(
                f"matrix has {m} columns but {len(self.metric_names)} names"
            )


@dataclass(frozen=True, slots=True)
class FederationConfig:
    """Top-level run config.

    ``num_rounds`` is a hard cap. When ``early_stop_patience`` is
    positive, training also stops once ``early_stop_metric`` (a key in
    the server evaluator's output) has failed to set a new maximum for
    that many consecutive rounds.

    ``force_malicious_per_round`` is a diagnostic sampling knob: when set,
    every round's cohort contains exactly that many malicious clients (the
    rest drawn uniformly from the benign population) instead of the plain
    uniform draw over all clients. ``None`` leaves sampling unchanged.

    ``profile_memory`` records, per round, the peak host and device memory
    allocated by the server-side work (scoring, aggregation, server training).
    Host tracing slows allocation-heavy code, so wall-clock timings from a run
    with it switched on are not representative.

    ``trace_capture`` records, per round, how far the malicious clients moved
    the aggregate (their weight share, the fraction of coordinates they
    captured, and the aggregate's projection on their mean direction). It
    reads the ground-truth malicious set after aggregation and costs an extra
    pass over the parameters, so it is off by default.

    TODO: make Hydra-instantiable.
    """

    num_clients: int
    clients_per_round: int
    num_rounds: int
    local_epochs: int
    local_batch_size: int
    local_learning_rate: float
    local_momentum: float
    seed: int = 0
    early_stop_patience: int = 0
    early_stop_metric: str = "val_accuracy"
    force_malicious_per_round: int | None = None
    profile_memory: bool = False
    trace_capture: bool = False


class FederatedClient(Protocol):
    """Behavioural contract for a participating client.

    TODO: implement adversarial clients against this same protocol so the
    server loop remains agnostic to client identity.
    """

    client_id: ClientId

    def local_train(
        self, ctx: RoundContext, config: FederationConfig
    ) -> ClientUpdate: ...

    def evaluate(
        self, parameters: NDArrays, config: FederationConfig
    ) -> tuple[float, int, dict[str, float]]: ...


class FederatedServer(Protocol):
    """Orchestrator contract.

    TODO: add a TOPSIS-aware impl satisfying the same protocol once metrics
    and integration land.
    """

    def run(self) -> list[RoundOutcome]: ...


class ServerEvaluator(Protocol):
    """Evaluates the aggregated global model on the server's held-out data.

    Implementations return a flat ``str -> float`` dict; the contract is
    that any of ``val_loss``, ``val_accuracy``, ``asr`` (attack success
    rate) may appear, depending on what the experiment is measuring.
    """

    def evaluate(self, parameters: NDArrays) -> dict[str, float]: ...


class Scorer(Protocol):
    """Maps one round's client metrics to per-client trust scores.

    Returns a 1-D array aligned to ``batch.client_ids``; each score lies in
    ``[0, 1]`` with higher meaning more trustworthy. Interpretable (TOPSIS)
    and learning-based detectors implement this same contract so the server
    loop and the offline evaluation stay agnostic to the scoring method.
    """

    def score(self, batch: MetricBatch) -> TrustScores: ...


class RoundScorer(Protocol):
    """Produces per-client trust scores for one round of collected updates.

    Bridges metric extraction and scoring for the online server loop: given
    the round's updates and the parameters they trained from, it returns the
    participating client ids and a trust score per client (aligned, in
    ``[0, 1]``, higher = more trustworthy). The server stays agnostic to how
    the scores are computed.
    """

    def score_round(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> tuple[tuple[ClientId, ...], TrustScores]: ...


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """One round's aggregation decision from an integrator.

    ``weights`` is aligned to the round's client order and sums to one; a zero
    weight means the client is excluded from this round's aggregate.
    ``sampling_probs`` is set only by participation-control mechanisms: a
    per-client bias the server applies when sampling the *next* round (``None``
    leaves sampling unchanged). ``keep_global`` means the integrator trusts no
    client this round: the server skips aggregation and carries the current
    global model forward, and ``weights`` is all zeros.
    """

    weights: np.ndarray
    sampling_probs: dict[ClientId, float] | None = None
    keep_global: bool = False


@dataclass(frozen=True, slots=True)
class RoundEvidence:
    """What the server observed in one round, beyond the trust scores.

    Handed to scorers and integrators that need more than the per-client
    scores (see :class:`EvidenceObserver`). ``server_parameters`` is the
    server's own model after training on its clean root set this round, or
    ``None`` when no component asked for it. ``metrics`` is the round's
    behavioural metric batch when the scorer exposes one; it is ``None`` when
    the evidence is delivered to the scorer itself, before scoring.
    """

    round_num: int
    updates: tuple[ClientUpdate, ...]
    global_parameters: NDArrays
    server_parameters: NDArrays | None = None
    metrics: MetricBatch | None = None


@runtime_checkable
class EvidenceObserver(Protocol):
    """A scorer or integrator that reads the round's evidence before acting.

    The server calls :meth:`observe` once per round, on the scorer before it
    scores and on the integrator before it integrates. When
    ``requires_server_update`` is true the server trains its own model on the
    root set every round and includes it in the evidence; a run whose
    components never ask for it does no server-side training.
    """

    requires_server_update: bool

    def observe(self, evidence: RoundEvidence) -> None: ...


class Integrator(Protocol):
    """Turns per-client trust scores into an aggregation decision.

    Stateful across rounds (e.g. an EMA reputation, quarantine counters). Kept
    separate from both the scorer and the aggregation rule: an integrator
    decides *which* clients contribute and with *what* weight, never how the
    weighted updates are combined.
    """

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult: ...


ClientFactory = Callable[[ClientId], FederatedClient]
