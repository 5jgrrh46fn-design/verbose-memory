"""Protocols for the three orthogonal attack hooks plus the round schedule.

These are kept deliberately narrow so that a single training-loop class can
be configured to behave benignly (no hooks) or adversarially (any subset of
hooks set) without conditional branches in the loop body.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from topfed.core.types import ClientUpdate, NDArrays, RoundContext


class Poisoner(Protocol):
    """Transforms a training batch in-place semantically.

    Implementations must be pure functions of the inputs (modulo their own
    RNG state). They MUST NOT mutate the caller's tensors; they return new
    tensors instead so the original loader remains reusable.
    """

    def poison_batch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class LossAugmenter(Protocol):
    """Combines the per-batch classification loss with an attack-specific term.

    Implementations receive the model and the unweighted cross-entropy loss
    and return the total scalar to call ``backward()`` on. They may close
    over the round's frozen reference parameters (e.g. the global model at
    round start) to compute a regularizer.
    """

    def combine(
        self, model: nn.Module, classification_loss: torch.Tensor
    ) -> torch.Tensor: ...


class UpdateManipulator(Protocol):
    """Transforms a finished ``ClientUpdate`` before it leaves the client.

    Used for model-side attacks that operate on the submitted delta rather
    than on training (gradient scaling, sign flipping, model replacement).
    The ``ctx`` argument lets implementations react to round-level state —
    e.g. dynamic scaling that depends on how many adversaries the server
    happens to have selected this round.
    """

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,
    ) -> ClientUpdate: ...


class AttackSchedule(Protocol):
    """Decides whether the attack is active on a given round."""

    def is_active(self, ctx: RoundContext) -> bool: ...
