"""Backbones accept a padded input size without changing the default net."""

from __future__ import annotations

import pytest
import torch

from topfed.models.cifar_cnn import CifarCNN
from topfed.models.femnist_cnn import FemnistCNN


def test_default_input_size_reproduces_reference_flatten_dims() -> None:
    assert CifarCNN().fc1.in_features == 64 * 5 * 5
    assert FemnistCNN().fc1.in_features == 64 * 4 * 4


def test_padded_cifar_input_forwards() -> None:
    model = CifarCNN(input_size=40)
    assert model.fc1.in_features == 64 * 7 * 7
    assert model(torch.zeros(2, 3, 40, 40)).shape == (2, 10)


def test_default_model_is_identical_to_pre_knob_construction() -> None:
    # Same seed -> same parameters, so existing runs stay reproducible.
    torch.manual_seed(0)
    a = CifarCNN()
    torch.manual_seed(0)
    b = CifarCNN(input_size=32)
    for pa, pb in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        assert torch.equal(pa, pb)


def test_too_small_input_rejected() -> None:
    with pytest.raises(ValueError):
        CifarCNN(input_size=12)
