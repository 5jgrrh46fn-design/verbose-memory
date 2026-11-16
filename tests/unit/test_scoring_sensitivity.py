"""Unit tests for OAT weight sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.scoring.sensitivity import (
    DEFAULT_LEVELS,
    oat_sensitivity,
    perturb_weight,
)


def test_perturb_weight_preserves_sum_and_scales_target() -> None:
    w = np.full(5, 0.2)
    out = perturb_weight(w, index=2, delta=0.3)
    assert out.sum() == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.2 * 1.3)
    # Untouched weights stay equal to each other after proportional rescale.
    others = np.delete(out, 2)
    assert np.allclose(others, others[0])


def test_perturb_weight_negative_delta() -> None:
    w = np.array([0.30, 0.10, 0.10, 0.30, 0.20])
    out = perturb_weight(w, index=0, delta=-0.2)
    assert out.sum() == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.30 * 0.8)


def test_oat_baseline_is_self_consistent() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(10, 5))
    weights = np.full(5, 0.2)
    benefit = np.array([True, False, False, False, False])
    result = oat_sensitivity(matrix, weights, benefit)
    assert result.spearman.shape == (5, len(DEFAULT_LEVELS))
    assert result.kendall.shape == (5, len(DEFAULT_LEVELS))
    finite = result.spearman[~np.isnan(result.spearman)]
    assert finite.size > 0
    # Rank correlations never exceed 1.
    assert np.all(np.abs(finite) <= 1.0 + 1e-9)


def test_oat_degenerate_single_client_is_nan() -> None:
    matrix = np.array([[0.4, 0.6, 0.1, 0.2, 0.3]])
    weights = np.full(5, 0.2)
    benefit = np.array([True, False, False, False, False])
    result = oat_sensitivity(matrix, weights, benefit)
    assert np.all(np.isnan(result.spearman))
    assert np.all(np.isnan(result.kendall))
