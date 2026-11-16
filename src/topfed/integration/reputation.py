"""Exponential-moving-average client reputation.

Every integration mechanism tracks a reputation per client: an EMA of its
trust scores across the rounds it participated in,
``R_t = alpha * s_t + (1 - alpha) * R_{t-1}``. A client's reputation is
seeded with its first observed score. ``alpha = 1`` disables smoothing, so
the reputation equals the current round's raw score; smaller ``alpha`` gives
more weight to history.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from topfed.core.types import ClientId, TrustScores


class EmaReputation:
    """Per-client EMA of trust scores, updated once per round."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = alpha
        self._reputation: dict[ClientId, float] = {}

    def update(
        self, client_ids: Sequence[ClientId], scores: TrustScores
    ) -> np.ndarray:
        """Update and return the reputations for ``client_ids`` (aligned)."""
        out = np.empty(len(client_ids), dtype=np.float64)
        for i, (cid, score) in enumerate(zip(client_ids, scores, strict=True)):
            s = float(score)
            prev = self._reputation.get(cid, s)
            r = self.alpha * s + (1.0 - self.alpha) * prev
            self._reputation[cid] = r
            out[i] = r
        return out

    def get(self, client_id: ClientId) -> float | None:
        return self._reputation.get(client_id)

    def as_dict(self) -> dict[ClientId, float]:
        return dict(self._reputation)
