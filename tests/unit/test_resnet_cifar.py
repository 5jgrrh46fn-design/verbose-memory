"""Unit tests for the CIFAR ResNet-18 with GroupNorm."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from topfed.federation.client import get_parameters, set_parameters
from topfed.models.resnet_cifar import ResNet18GN

RESNET18_CIFAR_PARAMS = 11_173_962


def test_forward_shape() -> None:
    model = ResNet18GN()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert ResNet18GN(num_classes=7)(torch.randn(1, 3, 32, 32)).shape == (1, 7)


def test_parameter_count_is_resnet18() -> None:
    assert sum(p.numel() for p in ResNet18GN().parameters()) == RESNET18_CIFAR_PARAMS
    # GroupNorm has the same (weight, bias) shape whatever the group count.
    assert sum(p.numel() for p in ResNet18GN(groups=4).parameters()) == RESNET18_CIFAR_PARAMS


def test_state_dict_has_no_buffers() -> None:
    model = ResNet18GN()
    assert list(model.buffers()) == []
    assert len(model.state_dict()) == sum(1 for _ in model.parameters())


def test_groupnorm_present_batchnorm_absent() -> None:
    model = ResNet18GN(groups=2)
    norms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert norms and all(m.num_groups == 2 for m in norms)
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_padded_input_keeps_parameter_count() -> None:
    model = ResNet18GN(input_size=40)
    assert sum(p.numel() for p in model.parameters()) == RESNET18_CIFAR_PARAMS
    assert model(torch.zeros(2, 3, 40, 40)).shape == (2, 10)


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError):
        ResNet18GN(input_size=4)
    with pytest.raises(ValueError):
        ResNet18GN(groups=3)


def test_eval_and_train_mode_agree() -> None:
    # No running statistics: the forward pass does not depend on the mode.
    model = ResNet18GN()
    x = torch.randn(3, 3, 32, 32)
    model.train()
    with torch.no_grad():
        a = model(x)
    model.eval()
    with torch.no_grad():
        b = model(x)
    assert torch.equal(a, b)


def test_parameters_round_trip_through_client_helpers() -> None:
    torch.manual_seed(0)
    src = ResNet18GN()
    arrays = get_parameters(src)
    assert len(arrays) == len(src.state_dict())
    torch.manual_seed(1)
    dst = ResNet18GN()
    set_parameters(dst, arrays)
    for a, b in zip(src.state_dict().values(), dst.state_dict().values(), strict=True):
        assert torch.equal(a, b)
