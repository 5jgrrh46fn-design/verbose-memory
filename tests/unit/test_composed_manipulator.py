"""ComposedManipulator: direction-aligned then norm-bounded."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.attacks.model.adaptive import (
    ComposedManipulator,
    DirectionAlignedManipulator,
    NormBoundedManipulator,
)
from topfed.core.types import ClientUpdate, NDArrays, RoundContext


def _ctx(g: NDArrays) -> RoundContext:
    return RoundContext(1, 10, g, ("c0",), 0)


def _flat(arrs: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(a).ravel() for a in arrs])


def _delta(update: ClientUpdate, g: NDArrays) -> np.ndarray:
    return _flat([w - gg for w, gg in zip(update.parameters, g, strict=True)])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _setup(seed: int) -> tuple[NDArrays, NDArrays, ClientUpdate]:
    rng = np.random.default_rng(seed)
    g: NDArrays = [rng.normal(size=(4, 3)), rng.normal(size=(5,))]
    reference: NDArrays = [rng.normal(size=(4, 3)), rng.normal(size=(5,))]
    poisoned = [gg + rng.normal(scale=2.0, size=gg.shape) for gg in g]
    update = ClientUpdate("c0", poisoned, 10, 0.1, 0.9)
    return g, reference, update


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(("cosine", "norm"), [(0.9, 0.6), (0.5, 1.0), (0.95, 3.0)])
def test_output_satisfies_both_constraints(seed: int, cosine: float, norm: float) -> None:
    g, reference, update = _setup(seed)
    composed = ComposedManipulator(
        [DirectionAlignedManipulator(cosine_target=cosine), NormBoundedManipulator(target_norm=norm)]
    )
    composed.set_reference_delta(reference)
    out = composed.manipulate(update, g, _ctx(g))
    delta = _delta(out, g)
    assert _cos(delta, _flat(reference)) == pytest.approx(cosine, abs=1e-9)
    assert float(np.linalg.norm(delta)) == pytest.approx(norm, rel=1e-9)
    assert [p.shape for p in out.parameters] == [p.shape for p in g]
    assert out.client_id == update.client_id and out.num_examples == update.num_examples


def test_the_order_is_load_bearing() -> None:
    # Norm-bounding first and aligning second resets the magnitude to the
    # reference's norm, so the norm constraint no longer holds.
    g, reference, update = _setup(0)
    reversed_ = ComposedManipulator(
        [NormBoundedManipulator(target_norm=0.6), DirectionAlignedManipulator(cosine_target=0.9)]
    )
    reversed_.set_reference_delta(reference)
    delta = _delta(reversed_.manipulate(update, g, _ctx(g)), g)
    assert float(np.linalg.norm(delta)) == pytest.approx(float(np.linalg.norm(_flat(reference))))
    assert float(np.linalg.norm(delta)) != pytest.approx(0.6)


def test_equals_applying_the_stages_by_hand() -> None:
    g, reference, update = _setup(3)
    da = DirectionAlignedManipulator(cosine_target=0.7)
    nb = NormBoundedManipulator(target_norm=1.5)
    da.set_reference_delta(reference)
    by_hand = nb.manipulate(da.manipulate(update, g, _ctx(g)), g, _ctx(g))

    composed = ComposedManipulator(
        [DirectionAlignedManipulator(cosine_target=0.7), NormBoundedManipulator(target_norm=1.5)]
    )
    composed.set_reference_delta(reference)
    out = composed.manipulate(update, g, _ctx(g))
    for a, b in zip(out.parameters, by_hand.parameters, strict=True):
        assert a == pytest.approx(b)


def test_reference_is_forwarded_only_to_stages_that_take_it() -> None:
    g, reference, update = _setup(1)
    da = DirectionAlignedManipulator(cosine_target=0.9)
    composed = ComposedManipulator([da, NormBoundedManipulator(target_norm=0.6)])
    composed.set_reference_delta(reference)
    assert da._reference_delta is not None
    assert _flat(da._reference_delta) == pytest.approx(_flat(reference))


def test_without_a_reference_only_the_norm_bound_applies() -> None:
    g, _, update = _setup(2)
    composed = ComposedManipulator(
        [DirectionAlignedManipulator(cosine_target=0.9), NormBoundedManipulator(target_norm=0.6)]
    )
    out = composed.manipulate(update, g, _ctx(g))
    delta = _delta(out, g)
    assert float(np.linalg.norm(delta)) == pytest.approx(0.6)
    assert _cos(delta, _delta(update, g)) == pytest.approx(1.0)


def test_rejects_an_empty_composition() -> None:
    with pytest.raises(ValueError):
        ComposedManipulator([])


def test_composed_config_instantiates_both_stages_with_their_knobs() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=composed_backdoor_cifar10_f20",
                "attack.update_manipulator.manipulators.0.cosine_target=0.5",
                "attack.update_manipulator.manipulators.1.target_norm=1.0",
            ],
        )
    m = instantiate(cfg.attack.update_manipulator)
    assert isinstance(m, ComposedManipulator)
    da, nb = m.manipulators
    assert isinstance(da, DirectionAlignedManipulator) and da.cosine_target == 0.5
    assert isinstance(nb, NormBoundedManipulator) and nb.target_norm == 1.0
    assert cfg.attack.loss_augmenter.alpha == 0.5
    assert cfg.attack.assignment.fraction == 0.2
