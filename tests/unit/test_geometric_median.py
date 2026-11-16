"""Weighted geometric median (smoothed Weiszfeld) as a defended base."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.core.types import ClientUpdate
from topfed.federation.server import _defended_aggregate, weighted_geometric_median


def _u(cid: str, *arrays: list[float]) -> ClientUpdate:
    return ClientUpdate(cid, [np.asarray(a, dtype=np.float64) for a in arrays], 1, 0.0, 1.0)


def _objective(z: np.ndarray, points: np.ndarray, w: np.ndarray) -> float:
    return float((w / w.sum()) @ np.linalg.norm(points - z, axis=1))


def test_collinear_points_give_the_weighted_median_point() -> None:
    # On a line the geometric median is the weighted median: 0, 1, 10 with
    # equal weights -> 1, wherever the outlier sits.
    updates = [_u("a", [0.0, 0.0]), _u("b", [1.0, 0.0]), _u("c", [10.0, 0.0])]
    out = weighted_geometric_median(updates, np.ones(3), max_iter=200, ftol=0.0)
    np.testing.assert_allclose(out[0], [1.0, 0.0], atol=1e-3)
    far = [_u("a", [0.0, 0.0]), _u("b", [1.0, 0.0]), _u("c", [1e6, 0.0])]
    np.testing.assert_allclose(
        weighted_geometric_median(far, np.ones(3), max_iter=200, ftol=0.0)[0], [1.0, 0.0], atol=1e-2
    )


def test_iterations_never_increase_the_objective_and_beat_the_mean() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(9, 40))
    points[0] += 50.0  # one outlier
    w = rng.uniform(0.5, 1.5, size=9)
    updates = [_u(f"c{i}", list(p)) for i, p in enumerate(points)]
    values = []
    for it in range(0, 11):
        z = weighted_geometric_median(updates, w, max_iter=it, ftol=0.0)[0]
        values.append(_objective(z, points, w))
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
    assert values[-1] < values[0]  # max_iter=0 is the weighted mean
    mean = (w / w.sum()) @ points
    np.testing.assert_allclose(
        weighted_geometric_median(updates, w, max_iter=0)[0], mean, atol=1e-12
    )
    # The outlier moves the geometric median far less than it moves the mean.
    clean = points[1:].mean(axis=0)
    z = weighted_geometric_median(updates, np.ones(9))[0]
    assert np.linalg.norm(z - clean) < 0.2 * np.linalg.norm(points.mean(axis=0) - clean)


def test_a_weight_majority_at_one_point_pins_the_median_there() -> None:
    updates = [_u("heavy", [3.0, -1.0]), _u("b", [0.0, 0.0]), _u("c", [9.0, 9.0])]
    out = weighted_geometric_median(updates, np.array([0.6, 0.2, 0.2]), max_iter=500, ftol=0.0)
    np.testing.assert_allclose(out[0], [3.0, -1.0], atol=1e-3)


def test_shapes_dtypes_and_multiple_tensors_are_preserved() -> None:
    a = ClientUpdate("a", [np.zeros((2, 2), dtype=np.float32), np.array([1.0], dtype=np.float32)], 1, 0.0, 1.0)
    b = ClientUpdate("b", [np.ones((2, 2), dtype=np.float32), np.array([3.0], dtype=np.float32)], 1, 0.0, 1.0)
    out = weighted_geometric_median([a, b], np.array([1.0, 1.0]))
    assert out[0].shape == (2, 2) and out[0].dtype == np.float32
    assert out[1].shape == (1,) and out[1].dtype == np.float32
    # Two points with equal weight: every point of the segment is a median;
    # the iteration starts at the midpoint and stays there.
    np.testing.assert_allclose(out[0], 0.5)
    np.testing.assert_allclose(out[1], [2.0])


def test_identical_updates_return_that_update_despite_zero_distances() -> None:
    updates = [_u(f"c{i}", [1.5, -2.0, 0.25]) for i in range(4)]
    out = weighted_geometric_median(updates, np.ones(4))
    np.testing.assert_allclose(out[0], [1.5, -2.0, 0.25])
    assert np.all(np.isfinite(out[0]))


def test_invalid_inputs_are_rejected() -> None:
    updates = [_u("a", [0.0]), _u("b", [1.0])]
    with pytest.raises(ValueError, match="at least one"):
        weighted_geometric_median([], np.array([]))
    with pytest.raises(ValueError, match="shape"):
        weighted_geometric_median(updates, np.ones(3))
    with pytest.raises(ValueError, match="positive total"):
        weighted_geometric_median(updates, np.zeros(2))
    with pytest.raises(ValueError, match="nu"):
        weighted_geometric_median(updates, np.ones(2), nu=0.0)


def test_defended_base_drops_zero_weight_clients_and_reads_its_options() -> None:
    updates = [_u("a", [0.0, 0.0]), _u("b", [1.0, 0.0]), _u("attacker", [100.0, 100.0])]
    weights = np.array([0.5, 0.5, 0.0])
    out = _defended_aggregate(updates, weights, "geometric_median")
    np.testing.assert_allclose(out[0], [0.5, 0.0], atol=1e-9)
    # Every weight zero: fall back to the equal-weight geometric median.
    all_zero = _defended_aggregate(updates, np.zeros(3), "geometric_median", {"max_iter": 300})
    assert np.linalg.norm(all_zero[0] - np.array([1.0, 0.0])) < 1.0
    # max_iter = 0 is the weighted mean.
    mean = _defended_aggregate(updates, np.array([1.0, 1.0, 2.0]), "geometric_median", {"max_iter": 0})
    np.testing.assert_allclose(mean[0], [50.25, 50.0])
