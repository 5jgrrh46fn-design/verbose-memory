"""Symmetric random label-flip poisoner.

A malicious client replaces a fraction of each batch's labels with values
drawn uniformly at random from the remaining classes (never the original).
This degrades the global model's clean accuracy without altering image
content, so it is the simplest untargeted data-poisoning attack: there is
no trigger, no perturbation, just label noise injected by adversarial
clients.

The poisoner operates on the batch prefix so that the poisoned-vs-clean
indices are deterministic for tests; callers are expected to shuffle
upstream so the prefix is not always the same samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_NUM_CLASSES = 10


def _random_other_class(
    original: torch.Tensor, num_classes: int, generator: torch.Generator | None
) -> torch.Tensor:
    """Draw a label uniformly from ``[0, num_classes) \\ {original}`` per element.

    Uses the rejection-free trick: pick an offset in ``[1, num_classes)``
    uniformly and add modulo ``num_classes``. This avoids a rejection loop
    while preserving the uniform distribution over the (C-1) alternatives.
    """
    offsets = torch.randint(
        low=1, high=num_classes, size=original.shape,
        generator=generator, device=original.device,
    )
    return (original + offsets) % num_classes


@dataclass(frozen=True, slots=True)
class DirtyLabelPoisoner:
    """Per-batch random label flip.

    A fraction ``poisoning_rate`` of each batch (the prefix; callers
    shuffle upstream) has its label replaced with a uniformly-drawn class
    other than the original. The inputs ``x`` are untouched.

    ``num_classes`` defaults to CIFAR-10; override for other datasets.
    """

    num_classes: int = DEFAULT_NUM_CLASSES
    poisoning_rate: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.poisoning_rate <= 1.0:
            raise ValueError(
                f"poisoning_rate must be in [0, 1]; got {self.poisoning_rate}"
            )
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2; got {self.num_classes}"
            )

    def poison_batch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = y.size(0)
        num_poisoned = int(round(batch_size * self.poisoning_rate))
        if num_poisoned == 0:
            return x, y

        y_out = y.clone()
        y_out[:num_poisoned] = _random_other_class(
            y[:num_poisoned], self.num_classes, generator=None
        )
        return x, y_out
