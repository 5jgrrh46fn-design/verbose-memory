"""``DirtyLabelPoisoner`` flips a fraction of labels to a different class."""

from __future__ import annotations

import pytest
import torch

from topfed.attacks.data.dirty_label import DirtyLabelPoisoner

NUM_CLASSES = 10


def _batch(batch_size: int = 8, label_value: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(batch_size, 3, 32, 32)
    y = torch.full((batch_size,), label_value, dtype=torch.long)
    return x, y


def test_zero_rate_is_identity() -> None:
    torch.manual_seed(0)
    poisoner = DirtyLabelPoisoner(num_classes=NUM_CLASSES, poisoning_rate=0.0)
    x, y = _batch()
    x_out, y_out = poisoner.poison_batch(x, y)
    assert torch.equal(x_out, x)
    assert torch.equal(y_out, y)


def test_inputs_are_never_modified() -> None:
    torch.manual_seed(0)
    poisoner = DirtyLabelPoisoner(num_classes=NUM_CLASSES, poisoning_rate=1.0)
    x, y = _batch(batch_size=4)
    x_snap = x.clone()
    y_snap = y.clone()
    poisoner.poison_batch(x, y)
    assert torch.equal(x, x_snap)
    assert torch.equal(y, y_snap)


def test_full_rate_flips_every_label_away_from_original() -> None:
    torch.manual_seed(0)
    poisoner = DirtyLabelPoisoner(num_classes=NUM_CLASSES, poisoning_rate=1.0)
    x, y = _batch(batch_size=64, label_value=3)
    _, y_out = poisoner.poison_batch(x, y)
    # Every poisoned label differs from the original — uniform draws over
    # [0, C) \ {original} guarantee this with probability 1.
    assert torch.all(y_out != y)
    # And remains in-range.
    assert torch.all(y_out >= 0)
    assert torch.all(y_out < NUM_CLASSES)


def test_partial_rate_flips_only_the_prefix() -> None:
    torch.manual_seed(0)
    poisoner = DirtyLabelPoisoner(num_classes=NUM_CLASSES, poisoning_rate=0.5)
    x, y = _batch(batch_size=8, label_value=3)
    _, y_out = poisoner.poison_batch(x, y)
    # Prefix flipped, tail untouched.
    assert torch.all(y_out[:4] != 3)
    assert torch.all(y_out[4:] == 3)


def test_x_passes_through_untouched() -> None:
    torch.manual_seed(0)
    poisoner = DirtyLabelPoisoner(num_classes=NUM_CLASSES, poisoning_rate=1.0)
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    x_out, _ = poisoner.poison_batch(x, y)
    assert torch.equal(x_out, x)


def test_full_rate_covers_every_alternative_class_over_many_draws() -> None:
    """The flip distribution is uniform over C-1 other classes."""
    torch.manual_seed(0)
    poisoner = DirtyLabelPoisoner(num_classes=NUM_CLASSES, poisoning_rate=1.0)
    # 10000 samples all labelled 0; expect roughly uniform draws over {1..9}.
    y = torch.zeros(10000, dtype=torch.long)
    x = torch.zeros(10000, 3, 4, 4)
    _, y_out = poisoner.poison_batch(x, y)
    # No original label leaks through.
    assert torch.all(y_out != 0)
    # Every alternative class appears.
    seen = set(y_out.tolist())
    assert seen == set(range(1, NUM_CLASSES))
    # Counts are within a few sd of the uniform expectation (1000 each).
    counts = torch.bincount(y_out, minlength=NUM_CLASSES)[1:]
    assert counts.min().item() > 800
    assert counts.max().item() < 1200


def test_validates_rate() -> None:
    with pytest.raises(ValueError):
        DirtyLabelPoisoner(poisoning_rate=-0.1)
    with pytest.raises(ValueError):
        DirtyLabelPoisoner(poisoning_rate=1.5)


def test_validates_num_classes() -> None:
    with pytest.raises(ValueError):
        DirtyLabelPoisoner(num_classes=1)


def test_two_class_problem_flips_to_the_other() -> None:
    torch.manual_seed(0)
    poisoner = DirtyLabelPoisoner(num_classes=2, poisoning_rate=1.0)
    x = torch.zeros(8, 1)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    _, y_out = poisoner.poison_batch(x, y)
    # Binary case: only one alternative per label.
    assert torch.equal(y_out, 1 - y)
