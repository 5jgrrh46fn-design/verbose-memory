"""Constrain-and-scale loss augmenter.

Adds an L2 anomaly term that pulls the local model toward the round-start
global parameters, balancing attack utility against detectability:

    L_total = alpha * L_class + (1 - alpha) * ||theta_local - theta_global||^2

The reference parameters are captured at round start and held as buffers on
the augmenter so that gradients flow through ``theta_local`` only.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from topfed.core.types import NDArrays


class ConstrainAndScaleAugmenter:
    """Penalize drift from a frozen reference parameter snapshot."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = alpha
        self._reference: list[torch.Tensor] | None = None
        self._reference_device: torch.device | None = None

    def set_reference(self, global_parameters: NDArrays) -> None:
        """Capture the round-start global parameters as the anomaly anchor."""
        self._reference = [
            torch.from_numpy(np.asarray(p)).detach() for p in global_parameters
        ]
        self._reference_device = self._reference[0].device

    def combine(
        self, model: nn.Module, classification_loss: torch.Tensor
    ) -> torch.Tensor:
        if self._reference is None:
            raise RuntimeError(
                "ConstrainAndScaleAugmenter.set_reference must be called "
                "before combine()"
            )
        # The model's device is only known after .to(device) is called by
        # the client; lazily migrate the reference tensors on first use so
        # the augmenter follows the model rather than the other way around.
        device = next(model.parameters()).device
        if self._reference_device != device:
            self._reference = [r.to(device) for r in self._reference]
            self._reference_device = device
        # Distance is computed against the *trainable* parameters only —
        # buffers such as BN running stats are skipped, since the anomaly
        # term is meant to constrain learnable drift.
        ref_by_name = dict(
            zip(model.state_dict().keys(), self._reference, strict=True)
        )
        sq_dist = torch.zeros((), device=device)
        for name, param in model.named_parameters():
            ref = ref_by_name[name]
            sq_dist = sq_dist + torch.sum((param - ref) ** 2)
        return self.alpha * classification_loss + (1.0 - self.alpha) * sq_dist
