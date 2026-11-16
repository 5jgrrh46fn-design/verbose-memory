"""Unit tests for TOPSIS closeness scoring."""

from __future__ import annotations

import numpy as np

from topfed.core.types import MetricBatch
from topfed.scoring.criteria import TRUST_METRICS, build_batch
from topfed.scoring.topsis import TopsisScorer, topsis_closeness
from topfed.scoring.weighting import equal_weights


def test_benefit_only_closed_form() -> None:
    # One benefit metric, two clients at the extremes: the high client is the
    # ideal (C=1), the low client is the anti-ideal (C=0).
    matrix = np.array([[1.0], [0.0]])
    scores = topsis_closeness(
        matrix, weights=np.array([1.0]), benefit=np.array([True])
    )
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_cost_orientation_flips_preference() -> None:
    # One cost metric: the lower value is now the trustworthy one.
    matrix = np.array([[1.0], [0.0]])
    scores = topsis_closeness(
        matrix, weights=np.array([1.0]), benefit=np.array([False])
    )
    assert scores[1] > scores[0]
    assert scores[0] == 0.0
    assert scores[1] == 1.0


def test_scores_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(10, 5))
    weights = np.full(5, 0.2)
    benefit = np.array([True, False, False, False, False])
    scores = topsis_closeness(matrix, weights, benefit)
    assert scores.shape == (10,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_ideal_client_ranks_highest() -> None:
    # Client 0 is best on every criterion: highest benefit, lowest costs.
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # ideal
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 1.0, 1.0, 1.0],  # anti-ideal
        ]
    )
    benefit = np.array([True, False, False, False, False])
    scores = topsis_closeness(matrix, np.full(5, 0.2), benefit)
    assert np.argmax(scores) == 0
    assert np.argmin(scores) == 2


def test_constant_column_is_finite_and_neutral() -> None:
    # A column where every client is identical cannot separate anyone; the
    # scorer must stay finite.
    matrix = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    benefit = np.array([True, False])
    scores = topsis_closeness(matrix, np.array([0.5, 0.5]), benefit)
    assert np.all(np.isfinite(scores))


def test_single_client_is_neutral() -> None:
    matrix = np.array([[0.4, 0.6, 0.1, 0.2, 0.3]])
    benefit = np.array([True, False, False, False, False])
    scores = topsis_closeness(matrix, np.full(5, 0.2), benefit)
    assert scores.shape == (1,)
    assert scores[0] == 0.5


def test_empty_matrix_returns_empty() -> None:
    scores = topsis_closeness(
        np.empty((0, 5)), np.full(5, 0.2), np.array([True] * 5)
    )
    assert scores.shape == (0,)


def test_scorer_uses_metric_orientation_from_batch() -> None:
    # Benign client aligned with the cohort (high m1, low costs) should score
    # above a client that is anti-aligned and anomalous.
    rows = [
        {
            "client_id": "benign",
            "m1_cosine_to_mean": 0.95,
            "m2_norm_ratio": 1.0,
            "m3_temporal_cv": 0.05,
            "m4_val_loss_delta": 0.0,
            "m5_max_layer_deviation": 0.5,
        },
        {
            "client_id": "malicious",
            "m1_cosine_to_mean": -0.5,
            "m2_norm_ratio": 5.0,
            "m3_temporal_cv": 0.9,
            "m4_val_loss_delta": 2.0,
            "m5_max_layer_deviation": 8.0,
        },
    ]
    batch = build_batch(rows, TRUST_METRICS)
    scores = TopsisScorer(weighting=equal_weights).score(batch)
    by_id = dict(zip(batch.client_ids, scores))
    assert by_id["benign"] > by_id["malicious"]


def test_build_batch_shapes_and_order() -> None:
    rows = [
        {"client_id": "a", **{m: float(i) for i, m in enumerate(TRUST_METRICS)}},
        {"client_id": "b", **{m: float(i + 1) for i, m in enumerate(TRUST_METRICS)}},
    ]
    batch = build_batch(rows, TRUST_METRICS)
    assert isinstance(batch, MetricBatch)
    assert batch.client_ids == ("a", "b")
    assert batch.matrix.shape == (2, 5)
    assert batch.matrix[0, 0] == 0.0
    assert batch.matrix[1, 0] == 1.0
