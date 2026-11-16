"""Oracle integrator: ground-truth exclusion for aggregation diagnostics.

``OracleFilter`` zero-weights every client in a known malicious set and
weights the rest uniformly. It ignores the trust scores it is handed (they are
still logged so the ledger shows what the scorer would have said). This is
not a defense: it answers the question "if detection were perfect, would the
aggregation rule underneath still be captured?" A run whose oracle-filtered
aggregate still admits the attack has an aggregation-side failure that no
scoring improvement can fix.
"""

from __future__ import annotations

import numpy as np

from topfed.core.types import ClientId, IntegrationResult, TrustScores
from topfed.integration.mechanisms import _BaseIntegrator


class OracleFilter(_BaseIntegrator):
    """Weight known attackers 0 and everyone else equally."""

    def __init__(self, malicious_clients: frozenset[ClientId]) -> None:
        super().__init__(alpha=1.0)
        self.malicious_clients = frozenset(malicious_clients)

    def integrate(
        self,
        round_num: int,
        client_ids: tuple[ClientId, ...],
        scores: TrustScores,
    ) -> IntegrationResult:
        reputations = self._reputation.update(client_ids, scores)
        keep = np.array(
            [cid not in self.malicious_clients for cid in client_ids],
            dtype=bool,
        )
        if not keep.any():
            # An all-malicious cohort has nothing honest to aggregate; fall
            # back to equal weights so the round still produces a model.
            keep = np.ones_like(keep)
        weights = keep.astype(np.float64)
        weights /= weights.sum()
        self._log(round_num, client_ids, scores, reputations, weights)
        return IntegrationResult(weights=weights)
