"""Adversarial client building blocks.

The attack surface is composed from three orthogonal hooks plus a schedule:

- ``Poisoner``           — transforms ``(x, y)`` batches before local training.
- ``LossAugmenter``      — adds a term to the per-batch loss (e.g. anomaly
                            evasion regularizer).
- ``UpdateManipulator``  — transforms the finished ``ClientUpdate`` before it
                            is returned to the server (e.g. delta boosting).
- ``AttackSchedule``     — decides whether the malicious behaviour is active
                            on a given round.

A single ``TorchClient`` composes these hooks; a benign client is the same
class with all hooks set to ``None``.
"""

from topfed.attacks.assignment import deterministic_malicious_set
from topfed.attacks.protocols import (
    AttackSchedule,
    LossAugmenter,
    Poisoner,
    UpdateManipulator,
)
from topfed.attacks.schedule import AlwaysAttack, WindowedAttack

__all__ = [
    "AlwaysAttack",
    "AttackSchedule",
    "LossAugmenter",
    "Poisoner",
    "UpdateManipulator",
    "WindowedAttack",
    "deterministic_malicious_set",
]
