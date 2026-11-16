"""Unit tests for Byzantine-robust aggregation rules."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.aggregation.robust import (
    DeltaAggregator,
    Krum,
    coordinate_median_aggregate,
    krum_aggregate,
    krum_select,
    trimmed_mean_aggregate,
)
from topfed.core.types import ClientUpdate, NDArrays


def _update(cid: str, value: float, shape: tuple[int, ...] = (2, 2)) -> ClientUpdate:
    return ClientUpdate(
        client_id=cid,
        parameters=[np.full(shape, value, dtype=np.float64)],
        num_examples=10,
        train_loss=0.1,
        train_accuracy=0.9,
    )


def test_coordinate_median_ignores_outlier() -> None:
    updates = [_update("a", 1.0), _update("b", 2.0), _update("c", 100.0)]
    agg = coordinate_median_aggregate(updates)
    assert agg[0] == pytest.approx(np.full((2, 2), 2.0))


def test_trimmed_mean_drops_extremes() -> None:
    updates = [_update(c, v) for c, v in zip("abcde", [1.0, 2.0, 3.0, 4.0, 100.0])]
    # trim_ratio 0.2 over 5 clients -> drop 1 from each end -> mean(2,3,4) = 3.
    agg = trimmed_mean_aggregate(updates, trim_ratio=0.2)
    assert agg[0] == pytest.approx(np.full((2, 2), 3.0))


def test_trimmed_mean_zero_trim_is_plain_mean() -> None:
    updates = [_update(c, v) for c, v in zip("abc", [1.0, 2.0, 6.0])]
    agg = trimmed_mean_aggregate(updates, trim_ratio=0.0)
    assert agg[0] == pytest.approx(np.full((2, 2), 3.0))


def test_trimmed_mean_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError):
        trimmed_mean_aggregate([_update("a", 1.0)], trim_ratio=0.5)


def test_krum_selects_clustered_update_over_outlier() -> None:
    # Four tightly clustered updates and one far outlier; Krum should return a
    # cluster member's parameters, never the outlier's.
    updates = [
        _update("a", 1.0),
        _update("b", 1.1),
        _update("c", 0.9),
        _update("d", 1.05),
        _update("outlier", 50.0),
    ]
    agg = krum_aggregate(updates, num_byzantine=1)
    value = float(agg[0].flat[0])
    assert value < 5.0  # a cluster member, not the 50.0 outlier


def test_krum_multi_averages_selected() -> None:
    updates = [_update(c, v) for c, v in zip("abcde", [1.0, 1.0, 1.0, 1.0, 50.0])]
    agg = krum_aggregate(updates, num_byzantine=1, num_selected=4)
    # The four clustered 1.0 updates are selected and averaged.
    assert agg[0] == pytest.approx(np.full((2, 2), 1.0))


def test_krum_requires_enough_clients() -> None:
    updates = [_update("a", 1.0), _update("b", 2.0)]
    with pytest.raises(ValueError):
        krum_aggregate(updates, num_byzantine=1)  # n - f - 2 = -1


def test_rules_preserve_shapes_multilayer() -> None:
    def multi(cid: str, v: float) -> ClientUpdate:
        params: NDArrays = [np.full((2, 2), v), np.full((3,), v)]
        return ClientUpdate(cid, params, 10, 0.1, 0.9)

    updates = [multi(c, v) for c, v in zip("abc", [1.0, 2.0, 3.0])]
    for agg in (
        coordinate_median_aggregate(updates),
        trimmed_mean_aggregate(updates, trim_ratio=0.0),
    ):
        assert agg[0].shape == (2, 2)
        assert agg[1].shape == (3,)


def test_krum_select_reports_the_selected_index() -> None:
    updates = [
        _update("outlier", 50.0),
        _update("a", 1.0),
        _update("b", 1.1),
        _update("c", 0.9),
        _update("d", 1.05),
    ]
    chosen = krum_select(updates, num_byzantine=1)
    assert chosen.shape == (1,)
    agg = krum_aggregate(updates, num_byzantine=1)
    assert np.array_equal(agg[0], updates[int(chosen[0])].parameters[0])
    assert updates[int(chosen[0])].client_id != "outlier"


@pytest.mark.parametrize("num_selected", [1, 3])
def test_krum_rule_matches_the_function_and_reports_weights(num_selected: int) -> None:
    rng = np.random.default_rng(0)
    updates = [
        ClientUpdate(
            client_id=f"c{i}",
            parameters=[rng.normal(size=(3, 4)), rng.normal(size=(5,))],
            num_examples=10,
            train_loss=0.1,
            train_accuracy=0.9,
        )
        for i in range(8)
    ]
    global_parameters: NDArrays = [np.zeros((3, 4)), np.zeros((5,))]
    rule = Krum(num_byzantine=2, num_selected=num_selected)
    assert isinstance(rule, DeltaAggregator)
    agg = rule.aggregate(1, updates, global_parameters)
    expected = krum_aggregate(updates, num_byzantine=2, num_selected=num_selected)
    for got, want in zip(agg, expected, strict=True):
        assert np.array_equal(got, want)
    chosen = krum_select(updates, num_byzantine=2, num_selected=num_selected)
    weights = rule.last_weights
    assert weights is not None
    assert weights.sum() == pytest.approx(1.0)
    assert set(np.flatnonzero(weights)) == {int(i) for i in chosen}
    assert weights[chosen] == pytest.approx(1.0 / num_selected)
