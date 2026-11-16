"""Score integration: trust scores -> per-round aggregation decisions.

Mechanisms live in :mod:`topfed.integration.mechanisms`; the shared EMA
reputation in :mod:`topfed.integration.reputation`; the ground-truth
diagnostic in :mod:`topfed.integration.oracle`; gated capped weighting (a
fair-share cap times a server-cosine or validation-loss gate) in
:mod:`topfed.integration.gated`. Integrators are kept separate
from the aggregation rule that combines the weighted updates.
"""

from topfed.integration.mechanisms import (
    AdaptiveParticipation,
    CappedScoreWeight,
    HardFilter,
    Quarantine,
    ScoreWeight,
    TemporalReputation,
)
from topfed.integration.gated import (
    M4BandCappedScoreWeight,
    ServerGatedCappedScoreWeight,
)
from topfed.integration.oracle import OracleFilter
from topfed.integration.reputation import EmaReputation

__all__ = [
    "AdaptiveParticipation",
    "CappedScoreWeight",
    "HardFilter",
    "M4BandCappedScoreWeight",
    "OracleFilter",
    "Quarantine",
    "ScoreWeight",
    "ServerGatedCappedScoreWeight",
    "TemporalReputation",
    "EmaReputation",
]
