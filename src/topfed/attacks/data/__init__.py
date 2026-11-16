"""Data-side poisoners (batch-level transformations)."""

from topfed.attacks.data.backdoor import (
    BackdoorPoisoner,
    apply_trigger,
    trigger_mask,
)
from topfed.attacks.data.dirty_label import DirtyLabelPoisoner

__all__ = ["BackdoorPoisoner", "DirtyLabelPoisoner", "apply_trigger", "trigger_mask"]
