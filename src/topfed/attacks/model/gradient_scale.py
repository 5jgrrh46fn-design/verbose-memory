"""Gradient-scaling model poisoning.

Returns ``global + factor * (local - global)``, amplifying the local delta
without involving a poisoner or anomaly-evasion loss. The intent is to
amplify a malicious client's pull on the FedAvg average; ``factor`` is the
perturbation-magnitude knob varied across the attack-budget sweep.

This is mathematically the same transform that ``ModelReplacementManipulator``
applies, but is kept as a separate class because:

- the two attacks are evaluated under different budget ranges,
- a gradient-scaling client carries no poisoner / loss augmenter,
- future variants (e.g. per-layer scaling) may diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topfed.core.types import ClientUpdate, NDArrays, RoundContext


@dataclass(frozen=True, slots=True)
class GradientScalingManipulator:
    factor: float = 5.0

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError(f"factor must be positive; got {self.factor}")

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,  # noqa: ARG002
    ) -> ClientUpdate:
        scaled = [
            g + self.factor * (w - g)
            for w, g in zip(update.parameters, global_parameters, strict=True)
        ]
        return ClientUpdate(
            client_id=update.client_id,
            parameters=[np.asarray(s) for s in scaled],
            num_examples=update.num_examples,
            train_loss=update.train_loss,
            train_accuracy=update.train_accuracy,
        )
