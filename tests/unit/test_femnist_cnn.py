"""Unit tests for the FEMNIST CNN backbone."""

from __future__ import annotations

import torch

from topfed.models.femnist_cnn import FemnistCNN


def test_forward_shape_62_classes() -> None:
    model = FemnistCNN()
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 62)


def test_custom_num_classes() -> None:
    model = FemnistCNN(num_classes=10)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)
