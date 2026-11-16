"""Unit tests for the trust-metric weighting strategies."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.scoring.weighting import (
    WEIGHTING_STRATEGIES,
    consistency_weights,
    critic_weights,
    entropy_weights,
    equal_weights,
    magnitude_weights,
    stability_weights,
)


def _sample_matrix() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(size=(10, 5))


@pytest.mark.parametrize("name", list(WEIGHTING_STRATEGIES))
def test_weights_are_valid_simplex(name: str) -> None:
    w = WEIGHTING_STRATEGIES[name](_sample_matrix())
    assert w.shape == (5,)
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0)


def test_equal_weights() -> None:
    w = equal_weights(_sample_matrix())
    assert np.allclose(w, 0.2)


def test_expert_templates_match_specification() -> None:
    m = _sample_matrix()
    assert np.allclose(consistency_weights(m), [0.30, 0.10, 0.10, 0.30, 0.20])
    assert np.allclose(magnitude_weights(m), [0.10, 0.30, 0.10, 0.20, 0.30])
    assert np.allclose(stability_weights(m), [0.20, 0.10, 0.30, 0.25, 0.15])


def test_expert_templates_ignore_data() -> None:
    a = consistency_weights(_sample_matrix())
    b = consistency_weights(np.zeros((4, 5)))
    assert np.allclose(a, b)


def test_entropy_downweights_constant_column() -> None:
    # Column 0 varies, column 1 is constant: the constant column carries no
    # information and should receive near-zero weight.
    matrix = np.array([[0.0, 7.0], [1.0, 7.0], [5.0, 7.0], [9.0, 7.0]])
    w = entropy_weights(matrix)
    assert w[0] > w[1]
    assert w[1] == pytest.approx(0.0, abs=1e-9)


def test_critic_downweights_constant_column() -> None:
    # A constant column has zero dispersion and must get zero weight.
    matrix = np.array([[1.0, 2.0], [4.0, 2.0], [9.0, 2.0]])
    w = critic_weights(matrix)
    assert w[1] == pytest.approx(0.0, abs=1e-9)
    assert w[0] == pytest.approx(1.0, abs=1e-9)


def test_critic_rewards_uncorrelated_high_variance() -> None:
    # Columns 0 and 1 are perfectly correlated (redundant); column 2 is
    # independent. The independent column should not be starved of weight.
    rng = np.random.default_rng(7)
    base = rng.normal(size=20)
    independent = rng.normal(size=20)
    matrix = np.column_stack([base, 2.0 * base, independent])
    w = critic_weights(matrix)
    assert w[2] > 0.0
    assert w.sum() == pytest.approx(1.0)


def test_data_derived_fall_back_to_equal_for_single_client() -> None:
    one = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    assert np.allclose(entropy_weights(one), 0.2)
    assert np.allclose(critic_weights(one), 0.2)


def test_m4_only_puts_all_weight_on_the_validation_loss_change() -> None:
    from topfed.scoring import TopsisScorer
    from topfed.scoring.criteria import TRUST_METRICS, build_batch
    from topfed.scoring.weighting import WEIGHTING_STRATEGIES, m4_only_weights

    w = m4_only_weights(np.zeros((3, 5)))
    assert w.sum() == 1.0 and w[TRUST_METRICS.index("m4_val_loss_delta")] == 1.0
    assert WEIGHTING_STRATEGIES["m4_only"] is m4_only_weights
    # The score is the min-max rank of M4 in the round (a cost: lower is better),
    # whatever the other metrics say.
    rows = [
        {"client_id": f"c{i}", "m1_cosine_to_mean": m1, "m2_norm_ratio": 1.0 + i,
         "m3_temporal_cv": 0.1 * i, "m4_val_loss_delta": m4, "m5_max_layer_deviation": 5.0 - i}
        for i, (m1, m4) in enumerate([(0.9, 0.4), (-0.5, -0.2), (0.1, 0.1), (0.3, 1.0)])
    ]
    scores = TopsisScorer(weighting=m4_only_weights).score(build_batch(rows, TRUST_METRICS))
    np.testing.assert_allclose(scores, [0.5, 1.0, 0.75, 0.0], atol=1e-12)
