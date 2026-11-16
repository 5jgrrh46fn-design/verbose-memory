"""Attack schedules: when a malicious client actually misbehaves.

* :class:`AlwaysAttack` — active on every round the client is sampled.
* :class:`WindowedAttack` — active only on rounds inside a closed interval
  ``[start_round, end_round]``; outside it the client trains benignly. A late
  ``start_round`` tests whether a defense that has converged on honest data
  resists a fresh attack; an early ``end_round`` tests whether an implanted
  attack persists once its sources go quiet.

TODO: add ``IntermittentAttack`` (active on a random subset of rounds) once
the intermittent attack row is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from topfed.core.types import RoundContext


@dataclass(frozen=True, slots=True)
class AlwaysAttack:
    """Active on every round the client is sampled."""

    def is_active(self, ctx: RoundContext) -> bool:  # noqa: ARG002
        return True


@dataclass(frozen=True, slots=True)
class WindowedAttack:
    """Active on rounds ``start_round <= round <= end_round`` (1-based).

    ``end_round=None`` leaves the window open-ended, so ``WindowedAttack(50)``
    is a delayed start and ``WindowedAttack(1, 100)`` an early stop.
    """

    start_round: int = 1
    end_round: int | None = None

    def __post_init__(self) -> None:
        if self.start_round < 1:
            raise ValueError(
                f"start_round must be >= 1; got {self.start_round}"
            )
        if self.end_round is not None and self.end_round < self.start_round:
            raise ValueError(
                f"end_round ({self.end_round}) must not precede start_round "
                f"({self.start_round})"
            )

    def is_active(self, ctx: RoundContext) -> bool:
        if ctx.round_num < self.start_round:
            return False
        return self.end_round is None or ctx.round_num <= self.end_round
