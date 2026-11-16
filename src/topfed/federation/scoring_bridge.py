"""Bridge from collected updates to per-client trust scores, online.

Wraps the stateful :class:`~topfed.metrics.MetricsExtractor` and a
:class:`~topfed.core.types.Scorer` so the server can score a round in one
call. No ground-truth labels are used — extraction is invoked with an empty
malicious set, since the defense must decide from behaviour alone.

:class:`ServerRootedMetricScorer` adds a sixth, server-rooted benefit
criterion (the cosine of each client's delta to the server's root-set delta)
to the decision matrix, so the scorer and its weighting strategy decide how
much it counts. :class:`AgreementMetricScorer` instead adds a cross-client
cost criterion: how closely each client's accumulated update history agrees
with any other client's in the round.

Two further round scorers use a single signal instead of the metric matrix,
so the same integration mechanisms can consume a score from another family:
:class:`ServerCosineScorer` (server-anchored: FLTrust's trust score) and
:class:`HistorySimilarityScorer` (peer-relative and history-based:
FoolsGold's per-client weight). Both already live in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from topfed.aggregation import foolsgold_weights, history_similarity
from topfed.core.geometry import server_cosines
from topfed.core.types import (
    ClientId,
    ClientUpdate,
    MetricBatch,
    NDArrays,
    RoundEvidence,
    Scorer,
    TrustScores,
)
from topfed.metrics import MetricsExtractor
from topfed.scoring.criteria import (
    AGREEMENT_TRUST_METRICS,
    PAIRWISE_AGREEMENT,
    ROOTED_TRUST_METRICS,
    SERVER_COSINE,
    TRUST_METRICS,
    build_batch,
)


class MetricScorer:
    """Extract M1-M5 for a round and turn them into trust scores.

    ``last_batch`` holds the most recent round's decision matrix so an
    integrator that gates on a metric can read it without re-extracting.
    """

    def __init__(self, extractor: MetricsExtractor, scorer: Scorer) -> None:
        self._extractor = extractor
        self._scorer = scorer
        self.last_batch: MetricBatch | None = None

    def _rows(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> list[dict[str, Any]]:
        rows = self._extractor.extract(
            round_num=round_num,
            updates=updates,
            global_parameters=global_parameters,
            malicious=frozenset(),
        )
        return [dict(r) for r in rows]

    def score_round(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> tuple[tuple[ClientId, ...], TrustScores]:
        rows = self._rows(round_num, updates, global_parameters)
        batch = build_batch(rows, TRUST_METRICS)
        self.last_batch = batch
        return batch.client_ids, self._scorer.score(batch)


class ServerRootedMetricScorer(MetricScorer):
    """M1-M5 plus the cosine to the server's root-set delta as a benefit column.

    The server delivers its update through :meth:`observe` before each round is
    scored. ``records`` keeps one row per client per round with the server
    cosine, for offline analysis of the new criterion.
    """

    requires_server_update = True

    def __init__(self, extractor: MetricsExtractor, scorer: Scorer) -> None:
        super().__init__(extractor, scorer)
        self._evidence: RoundEvidence | None = None
        self.records: list[dict[str, Any]] = []

    def observe(self, evidence: RoundEvidence) -> None:
        self._evidence = evidence

    def score_round(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> tuple[tuple[ClientId, ...], TrustScores]:
        evidence = self._evidence
        if (
            evidence is None
            or evidence.round_num != round_num
            or evidence.server_parameters is None
        ):
            raise RuntimeError(
                "server-rooted scoring needs the round's server update; call "
                "observe() before score_round()"
            )
        self._evidence = None
        cosines = server_cosines(
            list(updates), global_parameters, evidence.server_parameters
        )
        rows = self._rows(round_num, updates, global_parameters)
        for row, cos in zip(rows, cosines, strict=True):
            row[SERVER_COSINE] = float(cos)
            self.records.append(
                {"round": round_num, "client_id": row["client_id"], SERVER_COSINE: float(cos)}
            )
        batch = build_batch(rows, ROOTED_TRUST_METRICS)
        self.last_batch = batch
        scores = self._scorer.score(batch)
        return batch.client_ids, np.asarray(scores, dtype=np.float64)


class AgreementMetricScorer(MetricScorer):
    """M1-M5 plus pairwise history agreement as a cost column.

    Keeps, per client, the sum of every delta it has submitted in the rounds
    it was selected (the current round included), as FoolsGold does. The
    agreement of client ``i`` is the largest cosine similarity between its
    history and the history of any other client in the round, floored at 0.
    A bloc pursuing one objective accumulates aligned histories and scores
    high; honest clients on heterogeneous data do not. A single attacker has
    no one to agree with. ``records`` keeps one row per client per round with
    the agreement, for offline analysis.

    Histories are float32 vectors, one per client ever selected.
    """

    def __init__(self, extractor: MetricsExtractor, scorer: Scorer) -> None:
        super().__init__(extractor, scorer)
        self._history: dict[ClientId, np.ndarray] = {}
        self.records: list[dict[str, Any]] = []

    def _agreement(
        self,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> np.ndarray:
        flat_global = np.concatenate([np.ravel(g) for g in global_parameters])
        for u in updates:
            d = np.concatenate([np.ravel(p) for p in u.parameters]) - flat_global
            prev = self._history.get(u.client_id)
            self._history[u.client_id] = (
                d.astype(np.float32) if prev is None else prev + d
            )
        _, max_cs = history_similarity(
            np.stack([self._history[u.client_id] for u in updates])
        )
        return max_cs

    def score_round(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> tuple[tuple[ClientId, ...], TrustScores]:
        agreement = self._agreement(updates, global_parameters)
        rows = self._rows(round_num, updates, global_parameters)
        for row, a in zip(rows, agreement, strict=True):
            row[PAIRWISE_AGREEMENT] = float(a)
            self.records.append(
                {"round": round_num, "client_id": row["client_id"], PAIRWISE_AGREEMENT: float(a)}
            )
        batch = build_batch(rows, AGREEMENT_TRUST_METRICS)
        self.last_batch = batch
        scores = self._scorer.score(batch)
        return batch.client_ids, np.asarray(scores, dtype=np.float64)


def _accumulate_histories(
    history: dict[ClientId, np.ndarray],
    updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
    global_parameters: NDArrays,
) -> np.ndarray:
    """Add each client's delta to its float32 history; return the round's rows."""
    flat_global = np.concatenate([np.ravel(g) for g in global_parameters])
    for u in updates:
        d = np.concatenate([np.ravel(p) for p in u.parameters]) - flat_global
        prev = history.get(u.client_id)
        history[u.client_id] = d.astype(np.float32) if prev is None else prev + d
    return np.stack([history[u.client_id] for u in updates])


class ServerCosineScorer:
    """FLTrust's trust score as a round scorer: ``max(0, cos(delta_i, delta_0))``.

    ``delta_0`` is the server's own update from training on its clean root set
    this round, delivered through :meth:`observe` before the round is scored
    (Cao et al., "FLTrust", NDSS 2021, the trust score of Algorithm 2). The
    score is server-anchored and uses one signal; it lies in ``[0, 1]`` by
    construction, so it is passed to the integrator unscaled. A client whose
    delta points away from the server's, a zero delta and a round in which
    the server did not move all score 0.

    ``records`` keeps the raw (signed) cosine per client per round.
    """

    requires_server_update = True

    def __init__(self) -> None:
        self._evidence: RoundEvidence | None = None
        self.last_batch: MetricBatch | None = None
        self.records: list[dict[str, Any]] = []

    def observe(self, evidence: RoundEvidence) -> None:
        self._evidence = evidence

    def score_round(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> tuple[tuple[ClientId, ...], TrustScores]:
        evidence = self._evidence
        if (
            evidence is None
            or evidence.round_num != round_num
            or evidence.server_parameters is None
        ):
            raise RuntimeError(
                "server-cosine scoring needs the round's server update; call "
                "observe() before score_round()"
            )
        self._evidence = None
        cosines = server_cosines(
            list(updates), global_parameters, evidence.server_parameters
        )
        for u, cos in zip(updates, cosines, strict=True):
            self.records.append(
                {"round": round_num, "client_id": u.client_id, SERVER_COSINE: float(cos)}
            )
        scores = np.clip(cosines, 0.0, 1.0).astype(np.float64)
        return tuple(u.client_id for u in updates), scores


class HistorySimilarityScorer:
    """FoolsGold's per-client weight as a round scorer.

    Keeps, per client, the sum of every delta it has submitted in the rounds
    it was selected (the current round included) and scores the round's
    clients with :func:`~topfed.aggregation.foolsgold_weights` on those
    histories: pairwise cosine similarity, pardoning, ``1 - max similarity``
    rescaled by its maximum, then the logit with confidence ``kappa``
    (Fung, Yoon and Beschastnikh, "The Limitations of Federated Learning in
    Sybil Settings", RAID 2020). The score is peer-relative and
    history-based; it lies in ``[0, 1]`` by construction, so it is passed to
    the integrator unscaled. Clients whose histories resemble another
    client's score near 0; a client with no look-alike scores 1.

    Histories are float32 vectors, one per client ever selected. ``records``
    keeps each client's maximum history similarity (before pardoning) per
    round.
    """

    def __init__(self, kappa: float = 1.0, eps: float = 1e-12) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive; got {kappa}")
        self.kappa = float(kappa)
        self.eps = float(eps)
        self._history: dict[ClientId, np.ndarray] = {}
        self.last_batch: MetricBatch | None = None
        self.records: list[dict[str, Any]] = []

    def score_round(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> tuple[tuple[ClientId, ...], TrustScores]:
        histories = _accumulate_histories(self._history, updates, global_parameters)
        alpha, max_cs = foolsgold_weights(histories, self.kappa, self.eps)
        for u, s in zip(updates, max_cs, strict=True):
            self.records.append(
                {"round": round_num, "client_id": u.client_id, "max_similarity": float(s)}
            )
        scores = np.clip(alpha, 0.0, 1.0).astype(np.float64)
        return tuple(u.client_id for u in updates), scores
