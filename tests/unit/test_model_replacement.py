"""``ModelReplacementManipulator`` boosts the delta but leaves global anchor."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.attacks.model.scale import ModelReplacementManipulator
from topfed.core.types import ClientUpdate, RoundContext


def _make_update(parameters: list[np.ndarray]) -> ClientUpdate:
    return ClientUpdate(
        client_id="c0",
        parameters=parameters,
        num_examples=10,
        train_loss=0.5,
        train_accuracy=0.8,
    )


def _ctx(selected: tuple[str, ...] = ("c0",)) -> RoundContext:
    return RoundContext(
        round_num=1,
        total_rounds=1,
        global_parameters=[],
        selected_clients=selected,
        seed=0,
    )


def test_boost_scales_delta() -> None:
    global_params = [np.array([1.0, 2.0]), np.array([0.0])]
    local_params = [np.array([1.5, 2.5]), np.array([0.1])]
    update = _make_update(local_params)
    manipulator = ModelReplacementManipulator(gamma=10.0)

    boosted = manipulator.manipulate(update, global_params, _ctx())

    # global + 10 * (local - global) = 1 + 10*0.5 = 6, etc.
    np.testing.assert_allclose(boosted.parameters[0], np.array([6.0, 7.0]))
    np.testing.assert_allclose(boosted.parameters[1], np.array([1.0]))


def test_gamma_one_is_identity() -> None:
    global_params = [np.array([1.0, 2.0])]
    local_params = [np.array([1.5, 2.5])]
    manipulator = ModelReplacementManipulator(gamma=1.0)
    boosted = manipulator.manipulate(_make_update(local_params), global_params, _ctx())
    np.testing.assert_allclose(boosted.parameters[0], local_params[0])


def test_metadata_preserved() -> None:
    global_params = [np.zeros(2)]
    update = _make_update([np.ones(2)])
    boosted = ModelReplacementManipulator(gamma=5.0).manipulate(
        update, global_params, _ctx()
    )
    assert boosted.client_id == update.client_id
    assert boosted.num_examples == update.num_examples
    assert boosted.train_loss == update.train_loss
    assert boosted.train_accuracy == update.train_accuracy


def test_gamma_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ModelReplacementManipulator(gamma=0.0)


def test_dynamic_gamma_uses_per_round_adversary_count() -> None:
    # K=10 clients per round; malicious set = {c0, c1, c2}; this round
    # picked two of them (c0, c1) plus eight benigns, so f_t = 2 and
    # gamma_t should be K/f_t = 5.
    global_params = [np.array([0.0, 0.0])]
    local_params = [np.array([1.0, 2.0])]
    selected = ("c0", "c1") + tuple(f"c{i}" for i in range(3, 11))
    manipulator = ModelReplacementManipulator(
        gamma=None,
        clients_per_round=10,
        malicious_clients=frozenset({"c0", "c1", "c2"}),
    )

    boosted = manipulator.manipulate(_make_update(local_params), global_params, _ctx(selected))

    np.testing.assert_allclose(boosted.parameters[0], np.array([5.0, 10.0]))


def test_dynamic_gamma_treats_zero_adversaries_as_one() -> None:
    # f_t = 0 collapses to a no-op only if the manipulator is never called
    # (the client wouldn't be selected). But guard against the impossible:
    # division-by-zero must not crash; max(1, f_t) keeps gamma well-defined.
    manipulator = ModelReplacementManipulator(
        gamma=None,
        clients_per_round=10,
        malicious_clients=frozenset({"c0"}),
    )
    selected = tuple(f"c{i}" for i in range(1, 11))  # no malicious selected
    out = manipulator.manipulate(
        _make_update([np.array([1.0])]),
        [np.array([0.0])],
        _ctx(selected),
    )
    # gamma = 10/max(1, 0) = 10
    np.testing.assert_allclose(out.parameters[0], np.array([10.0]))


def test_dynamic_gamma_requires_population_args() -> None:
    with pytest.raises(ValueError):
        ModelReplacementManipulator(gamma=None)
    with pytest.raises(ValueError):
        ModelReplacementManipulator(gamma=None, clients_per_round=10)


def test_dynamic_gamma_rejects_nonpositive_K() -> None:
    with pytest.raises(ValueError):
        ModelReplacementManipulator(
            gamma=None, clients_per_round=0, malicious_clients=frozenset({"c0"})
        )
