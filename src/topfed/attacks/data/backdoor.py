"""Trigger-pattern backdoor poisoner.

Stamps a fixed pixel pattern onto a batch and rewrites the affected
labels to a target class. The same trigger function is exposed for use
at evaluation time (computing attack success rate) so the training and
evaluation triggers stay byte-identical.

Two batch-composition modes are supported, matching the two
conventions in the federated-backdoor literature:

* ``replace`` (default): a fraction ``poisoning_rate`` of each batch is
  overwritten with triggered + relabeled samples; the rest stays clean.
  The total batch size is unchanged.
* ``augment``: the batch is duplicated; the original half is kept
  intact and the new half is the entire input with the trigger applied
  and labels rewritten. The total batch size doubles. This matches the
  Bagdasaryan et al. canonical attack: the malicious client trains on
  ``[clean, triggered]`` pairs so the local model learns the backdoor
  *while preserving clean-task accuracy*, which is what keeps the
  scaled-replacement update from collapsing global clean accuracy.

The default trigger is a 4x4 white square in the bottom-right corner;
the intensity is controlled by ``trigger_amplitude`` (the value
written into the patched pixels, applied to all channels). Where the
trigger sits is controlled by ``trigger_position``:

* ``bottom_right`` (default) — a square patch in the bottom-right corner.
* ``center`` — the same square patch centred on the image, so it overlaps
  the region where the natural content (for example, handwritten ink)
  lives.
* ``border`` — a ring ``trigger_size`` pixels wide around the whole image
  perimeter. Combined with a constant padding border on the data this
  places the trigger in pixels that no natural sample ever varies.

Trigger placement decides which input region carries the backdoor, and
therefore whether honest training data ever exercises the weights that
implement it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DEFAULT_TRIGGER_SIZE = 4
DEFAULT_TARGET_CLASS = 9
BatchMode = Literal["replace", "augment"]
TriggerPosition = Literal["bottom_right", "center", "border"]
TRIGGER_POSITIONS: tuple[str, ...] = ("bottom_right", "center", "border")


def trigger_mask(
    height: int,
    width: int,
    *,
    trigger_size: int = DEFAULT_TRIGGER_SIZE,
    trigger_position: str = "bottom_right",
) -> torch.Tensor:
    """Boolean ``(H, W)`` mask of the pixels the trigger overwrites."""
    if trigger_size <= 0:
        raise ValueError(f"trigger_size must be positive; got {trigger_size}")
    if trigger_size > height or trigger_size > width:
        raise ValueError(
            f"trigger_size={trigger_size} exceeds spatial dims "
            f"{(height, width)}"
        )
    mask = torch.zeros(height, width, dtype=torch.bool)
    if trigger_position == "bottom_right":
        mask[height - trigger_size :, width - trigger_size :] = True
    elif trigger_position == "center":
        top = (height - trigger_size) // 2
        left = (width - trigger_size) // 2
        mask[top : top + trigger_size, left : left + trigger_size] = True
    elif trigger_position == "border":
        if 2 * trigger_size >= min(height, width):
            raise ValueError(
                f"border trigger of width {trigger_size} leaves no interior "
                f"on a {(height, width)} image"
            )
        mask[:] = True
        mask[trigger_size:-trigger_size, trigger_size:-trigger_size] = False
    else:
        raise ValueError(
            f"trigger_position must be one of {TRIGGER_POSITIONS}; "
            f"got {trigger_position!r}"
        )
    return mask


def apply_trigger(
    x: torch.Tensor,
    *,
    trigger_size: int = DEFAULT_TRIGGER_SIZE,
    trigger_amplitude: float = 1.0,
    trigger_position: str = "bottom_right",
) -> torch.Tensor:
    """Stamp the trigger onto every sample in ``x``.

    ``x`` is expected to be a ``(N, C, H, W)`` float tensor. A copy is
    returned; the input is not mutated.
    """
    if x.dim() != 4:
        raise ValueError(f"expected (N, C, H, W) tensor; got shape {tuple(x.shape)}")
    h, w = x.shape[-2:]
    mask = trigger_mask(
        int(h),
        int(w),
        trigger_size=trigger_size,
        trigger_position=trigger_position,
    ).to(x.device)
    out = x.clone()
    out[..., mask] = trigger_amplitude
    return out


@dataclass(frozen=True, slots=True)
class BackdoorPoisoner:
    """Per-batch trigger injection.

    Behaviour depends on ``batch_mode``:

    * ``replace`` — a fraction ``poisoning_rate`` of each batch (the
      prefix; callers shuffle upstream) is overwritten with triggered
      samples relabelled to ``target_class``. Batch size unchanged.
      Deterministic poisoned-vs-clean indices for tests.
    * ``augment`` — the batch is concatenated with a fully-triggered,
      fully-relabelled copy of itself. Batch size doubles. The
      ``poisoning_rate`` field is ignored in this mode (the effective
      mix is 1:1 by construction). This is the canonical Bagdasaryan
      training setup and is what preserves the local model's clean
      accuracy when the malicious client trains on backdoor data.
    """

    target_class: int = DEFAULT_TARGET_CLASS
    trigger_size: int = DEFAULT_TRIGGER_SIZE
    trigger_amplitude: float = 1.0
    poisoning_rate: float = 0.5
    batch_mode: BatchMode = "replace"
    trigger_position: str = "bottom_right"

    def __post_init__(self) -> None:
        if not 0.0 <= self.poisoning_rate <= 1.0:
            raise ValueError(
                f"poisoning_rate must be in [0, 1]; got {self.poisoning_rate}"
            )
        if self.trigger_size <= 0:
            raise ValueError(
                f"trigger_size must be positive; got {self.trigger_size}"
            )
        if self.batch_mode not in ("replace", "augment"):
            raise ValueError(
                f"batch_mode must be 'replace' or 'augment'; "
                f"got {self.batch_mode!r}"
            )
        if self.trigger_position not in TRIGGER_POSITIONS:
            raise ValueError(
                f"trigger_position must be one of {TRIGGER_POSITIONS}; "
                f"got {self.trigger_position!r}"
            )

    def poison_batch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_mode == "augment":
            triggered_x = apply_trigger(
                x,
                trigger_size=self.trigger_size,
                trigger_amplitude=self.trigger_amplitude,
                trigger_position=self.trigger_position,
            )
            triggered_y = torch.full_like(y, self.target_class)
            return (
                torch.cat([x, triggered_x], dim=0),
                torch.cat([y, triggered_y], dim=0),
            )

        # replace mode
        batch_size = y.size(0)
        num_poisoned = int(round(batch_size * self.poisoning_rate))
        if num_poisoned == 0:
            return x, y

        x_out = x.clone()
        y_out = y.clone()
        x_out[:num_poisoned] = apply_trigger(
            x[:num_poisoned],
            trigger_size=self.trigger_size,
            trigger_amplitude=self.trigger_amplitude,
            trigger_position=self.trigger_position,
        )
        y_out[:num_poisoned] = self.target_class
        return x_out, y_out
