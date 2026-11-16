"""Sign-flipping model poisoning.

Returns ``global - eta * (local - global)``, inverting the direction of the
local delta so the submitted update opposes benign convergence. ``eta``
controls magnitude: ``eta=1`` flips the sign without scaling; larger values
amplify the inverted pull, matching the perturbation-magnitude axis of the
attack-budget sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topfed.core.types import ClientUpdate, NDArrays, RoundContext


@dataclass(frozen=True, slots=True)
class SignFlippingManipulator:
    eta: float = 1.0

    def __post_init__(self) -> None:
        if self.eta <= 0:
            raise ValueError(f"eta must be positive; got {self.eta}")

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,  # noqa: ARG002
    ) -> ClientUpdate:
        flipped = [
            g - self.eta * (w - g)
            for w, g in zip(update.parameters, global_parameters, strict=True)
        ]
        return ClientUpdate(
            client_id=update.client_id,
            parameters=[np.asarray(f) for f in flipped],
            num_examples=update.num_examples,
            train_loss=update.train_loss,
            train_accuracy=update.train_accuracy,
        )
