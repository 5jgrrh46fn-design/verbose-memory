"""``ConstrainAndScaleAugmenter`` correctly weights CE and L2 anomaly term."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from topfed.attacks.model.constrain import ConstrainAndScaleAugmenter
from topfed.federation.client import get_parameters


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


def test_combine_zero_distance_returns_alpha_times_ce() -> None:
    model = _TinyModel()
    aug = ConstrainAndScaleAugmenter(alpha=0.4)
    aug.set_reference(get_parameters(model))

    ce = torch.tensor(2.0, requires_grad=False)
    total = aug.combine(model, ce)
    # No drift yet, so the L2 term is 0; total = 0.4 * 2.0.
    assert torch.allclose(total, torch.tensor(0.8))


def test_combine_penalizes_drift_from_reference() -> None:
    model = _TinyModel()
    aug = ConstrainAndScaleAugmenter(alpha=0.5)
    aug.set_reference(get_parameters(model))

    # Move the model away from the reference.
    with torch.no_grad():
        model.linear.weight.add_(torch.ones_like(model.linear.weight))

    ce = torch.tensor(0.0)
    total = aug.combine(model, ce)
    # 4 weights, each shifted by 1 -> sq_dist = 4. (1 - 0.5) * 4 = 2.0.
    assert torch.allclose(total, torch.tensor(2.0))


def test_combine_requires_reference() -> None:
    aug = ConstrainAndScaleAugmenter()
    with pytest.raises(RuntimeError):
        aug.combine(_TinyModel(), torch.tensor(1.0))


def test_alpha_validation() -> None:
    with pytest.raises(ValueError):
        ConstrainAndScaleAugmenter(alpha=0.0)
    with pytest.raises(ValueError):
        ConstrainAndScaleAugmenter(alpha=1.5)
