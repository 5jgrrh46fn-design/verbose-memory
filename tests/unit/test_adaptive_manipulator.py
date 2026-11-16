"""Unit tests for the adaptive (norm-bounded) update manipulator."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.attacks.model.adaptive import (
    DirectionAlignedManipulator,
    NormBoundedManipulator,
)
from topfed.core.types import ClientUpdate, NDArrays, RoundContext


def _ctx() -> RoundContext:
    return RoundContext(
        round_num=1,
        total_rounds=10,
        global_parameters=[np.zeros((2, 2))],
        selected_clients=("c0",),
        seed=0,
    )


def _flat_norm(params: NDArrays, global_params: NDArrays) -> float:
    delta = np.concatenate(
        [(w - g).ravel() for w, g in zip(params, global_params, strict=True)]
    )
    return float(np.linalg.norm(delta))


def _update(params: NDArrays) -> ClientUpdate:
    return ClientUpdate("c0", params, num_examples=10, train_loss=0.1, train_accuracy=0.9)


def test_projects_delta_to_target_norm() -> None:
    g: NDArrays = [np.zeros((2, 2))]
    update = _update([np.full((2, 2), 5.0)])  # delta norm = sqrt(4*25) = 10
    out = NormBoundedManipulator(target_norm=2.0).manipulate(update, g, _ctx())
    assert _flat_norm(out.parameters, g) == pytest.approx(2.0)


def test_preserves_direction() -> None:
    g: NDArrays = [np.zeros((2, 2)), np.zeros((3,))]
    update = _update([np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([5.0, 6.0, 7.0])])
    orig = np.concatenate([d.ravel() for d in update.parameters])
    out = NormBoundedManipulator(target_norm=1.0).manipulate(update, g, _ctx())
    new = np.concatenate([d.ravel() for d in out.parameters])
    cos = float(np.dot(orig, new) / (np.linalg.norm(orig) * np.linalg.norm(new)))
    assert cos == pytest.approx(1.0)


def test_upscales_small_delta_to_budget() -> None:
    g: NDArrays = [np.zeros((2, 2))]
    update = _update([np.full((2, 2), 0.1)])  # delta norm = 0.2
    out = NormBoundedManipulator(target_norm=5.0).manipulate(update, g, _ctx())
    assert _flat_norm(out.parameters, g) == pytest.approx(5.0)


def test_zero_delta_unchanged() -> None:
    g: NDArrays = [np.zeros((2, 2))]
    update = _update([np.zeros((2, 2))])
    out = NormBoundedManipulator(target_norm=3.0).manipulate(update, g, _ctx())
    assert out is update


def test_rejects_nonpositive_target() -> None:
    with pytest.raises(ValueError):
        NormBoundedManipulator(target_norm=0.0)


def _delta(params: NDArrays, global_params: NDArrays) -> np.ndarray:
    return np.concatenate(
        [(w - g).ravel() for w, g in zip(params, global_params, strict=True)]
    )


def test_direction_aligned_hits_cosine_and_reference_norm() -> None:
    # reference delta along axis 0; poison adds an orthogonal component on axis
    # 2. With cosine_target 0.6 the output is 0.6*r_hat + 0.8*orth_hat at the
    # reference norm (1.0) -> [0.6, 0, 0.8, 0].
    g: NDArrays = [np.zeros(4)]
    man = DirectionAlignedManipulator(cosine_target=0.6)
    man.set_reference_delta([np.array([1.0, 0.0, 0.0, 0.0])])
    out = man.manipulate(_update([np.array([1.0, 0.0, 2.0, 0.0])]), g, _ctx())
    u = _delta(out.parameters, g)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    cos = float(np.dot(u, r) / (np.linalg.norm(u) * np.linalg.norm(r)))
    assert cos == pytest.approx(0.6)
    assert np.linalg.norm(u) == pytest.approx(1.0)  # reference norm
    assert u == pytest.approx(np.array([0.6, 0.0, 0.8, 0.0]))


def test_direction_aligned_retains_orthogonal_backdoor() -> None:
    # The orthogonal poison direction (axis 2) survives with positive weight.
    g: NDArrays = [np.zeros(4)]
    man = DirectionAlignedManipulator(cosine_target=0.9)
    man.set_reference_delta([np.array([1.0, 0.0, 0.0, 0.0])])
    out = man.manipulate(_update([np.array([1.0, 0.0, 5.0, 0.0])]), g, _ctx())
    u = _delta(out.parameters, g)
    assert u[2] > 0.0


def test_direction_aligned_passthrough_without_reference() -> None:
    g: NDArrays = [np.zeros(4)]
    update = _update([np.array([1.0, 0.0, 2.0, 0.0])])
    out = DirectionAlignedManipulator().manipulate(update, g, _ctx())
    assert out is update


def test_direction_aligned_passthrough_when_poison_parallel() -> None:
    # Poison lies entirely along the benign reference -> nothing orthogonal to
    # inject -> the update is returned unchanged.
    g: NDArrays = [np.zeros(4)]
    man = DirectionAlignedManipulator(cosine_target=0.8)
    man.set_reference_delta([np.array([1.0, 0.0, 0.0, 0.0])])
    update = _update([np.array([3.0, 0.0, 0.0, 0.0])])  # delta parallel to ref
    out = man.manipulate(update, g, _ctx())
    assert out is update


def test_direction_aligned_rejects_invalid_cosine() -> None:
    with pytest.raises(ValueError):
        DirectionAlignedManipulator(cosine_target=1.0)
    with pytest.raises(ValueError):
        DirectionAlignedManipulator(cosine_target=0.0)
