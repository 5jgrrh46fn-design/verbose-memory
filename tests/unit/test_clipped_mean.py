"""Norm-clipped mean on synthetic updates."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.aggregation import ClippedMean, DeltaAggregator
from topfed.core.types import ClientUpdate, NDArrays


def _globals() -> NDArrays:
    return [np.zeros((2, 2)), np.zeros((3,))]


def _update(cid: str, value: float) -> ClientUpdate:
    return ClientUpdate(cid, [np.full((2, 2), value), np.full((3,), value)], 10, 0.0, 1.0)


def test_median_bound_clips_the_outlier_and_averages() -> None:
    g = _globals()
    updates = [_update("a", 1.0), _update("b", 1.0), _update("c", 1.0), _update("d", 100.0)]
    agg = ClippedMean()
    assert isinstance(agg, DeltaAggregator)
    out = agg.aggregate(1, updates, g)
    # Norms: three at sqrt(7), one at 100 sqrt(7); median bound = sqrt(7), so
    # the outlier is scaled to the bound and the mean is exactly 1.0 everywhere.
    assert out[0] == pytest.approx(np.full((2, 2), 1.0))
    assert out[1] == pytest.approx(np.full((3,), 1.0))
    assert agg.last_weights is not None
    np.testing.assert_allclose(agg.last_weights, 0.25)
    rows = agg.records
    assert len(rows) == 4
    assert rows[3]["clip_factor"] == pytest.approx(0.01)
    assert rows[0]["clip_factor"] == 1.0
    assert rows[0]["clip_bound"] == pytest.approx(np.sqrt(7.0))


def test_fixed_bound_and_validation() -> None:
    g = _globals()
    updates = [_update("a", 1.0), _update("b", 3.0)]
    out = ClippedMean(bound=np.sqrt(7.0)).aggregate(1, updates, g)
    # Both deltas clipped to norm sqrt(7): the second is scaled by 1/3.
    assert out[0] == pytest.approx(np.full((2, 2), 1.0))
    with pytest.raises(ValueError):
        ClippedMean(bound=0.0)
    with pytest.raises(ValueError):
        ClippedMean().aggregate(1, [], g)


def test_zero_delta_is_left_alone() -> None:
    g = _globals()
    updates = [_update("a", 0.0), _update("b", 2.0)]
    agg = ClippedMean()
    out = agg.aggregate(1, updates, g)
    # Median norm is the midpoint of 0 and 2 sqrt(7); the zero delta has
    # factor 1 by convention and contributes nothing.
    assert agg.records[0]["clip_factor"] == 1.0
    assert (out[0] > 0).all()
