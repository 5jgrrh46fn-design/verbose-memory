"""``GradientScalingManipulator`` amplifies the delta about the global anchor."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.attacks.model.gradient_scale import GradientScalingManipulator
from topfed.core.types import ClientUpdate, RoundContext


_CTX = RoundContext(
    round_num=1,
    total_rounds=1,
    global_parameters=[],
    selected_clients=("c0",),
    seed=0,
)


def _make_update(parameters: list[np.ndarray]) -> ClientUpdate:
    return ClientUpdate(
        client_id="c0",
        parameters=parameters,
        num_examples=10,
        train_loss=0.5,
        train_accuracy=0.8,
    )


def test_scaling_amplifies_delta() -> None:
    global_params = [np.array([1.0, 2.0]), np.array([0.0])]
    local_params = [np.array([1.5, 2.5]), np.array([0.1])]
    manipulator = GradientScalingManipulator(factor=5.0)

    out = manipulator.manipulate(_make_update(local_params), global_params, _CTX)

    # global + 5 * (local - global) = 1 + 5*0.5 = 3.5, etc.
    np.testing.assert_allclose(out.parameters[0], np.array([3.5, 4.5]))
    np.testing.assert_allclose(out.parameters[1], np.array([0.5]))


def test_factor_one_is_identity() -> None:
    global_params = [np.array([1.0, 2.0])]
    local_params = [np.array([1.5, 2.5])]
    out = GradientScalingManipulator(factor=1.0).manipulate(
        _make_update(local_params), global_params, _CTX
    )
    np.testing.assert_allclose(out.parameters[0], local_params[0])


def test_metadata_preserved() -> None:
    update = _make_update([np.ones(2)])
    out = GradientScalingManipulator(factor=2.0).manipulate(update, [np.zeros(2)], _CTX)
    assert out.client_id == update.client_id
    assert out.num_examples == update.num_examples
    assert out.train_loss == update.train_loss
    assert out.train_accuracy == update.train_accuracy


def test_factor_must_be_positive() -> None:
    with pytest.raises(ValueError):
        GradientScalingManipulator(factor=0.0)
    with pytest.raises(ValueError):
        GradientScalingManipulator(factor=-1.0)
