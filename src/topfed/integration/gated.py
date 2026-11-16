"""Gated capped score-weighting: a fair-share cap times a per-client gate.

Both integrators here start from :class:`CappedScoreWeight`'s weights (trust
shares, each capped at ``cap / n``) and multiply each client's weight by a gate
in ``[0, 1]`` read from the round's evidence:

* :class:`ServerGatedCappedScoreWeight` gates by ``ReLU(cos(delta_i,
  delta_server))``, the cosine of the client's delta to the server's own
  root-set delta. The attacker cannot observe the server delta.
* :class:`M4BandCappedScoreWeight` gates by ``1[M4_i <= q]``, where ``q`` is a
  quantile of the round's validation-loss deltas across the cohort. It uses no
  data beyond the validation slice the scorer already reads, and serves as the
  control for whether the server-rooted signal matters or gating alone does.

A gate never exceeds one, so a client's pre-normalisation weight never exceeds
its capped weight: gating can only down-weight. When every gated weight is
zero the integrator trusts no client and asks the server to keep the current
global model (``keep_global``).
"""

from __future__ import annotations

import numpy as np

from topfed.core.geometry import server_cosines
from topfed.core.types import (
    ClientId,
    IntegrationResult,
    RoundEvidence,
    TrustScores,
)
from topfed.integration.mechanisms import _EPS, CappedScoreWeight

M4_METRIC = "m4_val_loss_delta"


class _GatedCappedScoreWeight(CappedScoreWeight):
    """Capped score-weighting multiplied by a per-client gate in ``[0, 1]``."""

    requires_server_update: bool = False

    def __init__(self, cap: float = 1.0, alpha: float = 1.0) -> None:
        super().__init__(cap=cap, alpha=alpha)
        self._evidence: RoundEvidence | None = None

    def observe(self, evidence: RoundEvidence) -> None:
        self._evidence = evidence

    def _gate(
        self, evidence: RoundEvidence, client_ids: tuple[ClientId, ...]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return the gate and any per-client columns to log beside it."""
        raise NotImplementedError

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult:
        evidence = self._evidence
        if evidence is None or evidence.round_num != round_num:
            raise RuntimeError(
                f"{type(self).__name__} needs the round's evidence; call "
                "observe() before integrate()"
            )
        if tuple(u.client_id for u in evidence.updates) != client_ids:
            raise ValueError("evidence updates are not in the scored client order")
        self._evidence = None

        reputations = self._reputation.update(client_ids, scores)
        capped = self._capped_weights(reputations)
        gate, extras = self._gate(evidence, client_ids)
        if gate.shape != capped.shape or (gate < 0.0).any() or (gate > 1.0).any():
            raise ValueError("gate must be aligned to the cohort and lie in [0, 1]")
        gated = capped * gate
        total = float(gated.sum())
        keep_global = total <= _EPS
        weights = np.zeros_like(gated) if keep_global else gated / total
        self._log(
            round_num,
            client_ids,
            scores,
            reputations,
            weights,
            {"capped_weight": capped, "gate": gate, **extras},
        )
        return IntegrationResult(weights=weights, keep_global=keep_global)


class ServerGatedCappedScoreWeight(_GatedCappedScoreWeight):
    """Capped score-weighting gated by ReLU of the cosine to the server delta.

    ``w_i = min(cap / n, t_i) * ReLU(cos(delta_i, delta_server))``, normalized,
    where ``t_i`` is the client's trust share. A client pointing away from the
    server's clean-data direction is excluded; one aligned with it keeps at
    most its capped share.
    """

    requires_server_update = True

    def _gate(
        self, evidence: RoundEvidence, client_ids: tuple[ClientId, ...]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if evidence.server_parameters is None:
            raise ValueError("server-gated integration needs the server update")
        cosines = server_cosines(
            evidence.updates, evidence.global_parameters, evidence.server_parameters
        )
        return np.clip(cosines, 0.0, 1.0), {"server_cosine": cosines}


class M4BandCappedScoreWeight(_GatedCappedScoreWeight):
    """Capped score-weighting gated by a validation-loss-delta band.

    ``w_i = min(cap / n, t_i) * 1[M4_i <= q]`` with ``q`` the ``quantile`` of
    the round's M4 values across the cohort (attackers included; the server
    cannot tell them apart). Clients whose update raises the validation loss
    the most are excluded every round.
    """

    def __init__(
        self, cap: float = 1.0, alpha: float = 1.0, quantile: float = 0.75
    ) -> None:
        super().__init__(cap=cap, alpha=alpha)
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must lie in (0, 1]; got {quantile}")
        self.quantile = quantile

    def _gate(
        self, evidence: RoundEvidence, client_ids: tuple[ClientId, ...]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        batch = evidence.metrics
        if batch is None or M4_METRIC not in batch.metric_names:
            raise ValueError(f"M4-band integration needs the {M4_METRIC} column")
        if batch.client_ids != client_ids:
            raise ValueError("metric batch is not in the scored client order")
        m4 = batch.matrix[:, batch.metric_names.index(M4_METRIC)]
        band = float(np.quantile(m4, self.quantile))
        gate = (m4 <= band).astype(np.float64)
        return gate, {"m4": m4, "m4_band": np.full_like(m4, band)}
