"""Unit tests for the learning-based client scorers."""

from __future__ import annotations

import numpy as np

from topfed.core.types import MetricBatch
from topfed.scoring.criteria import TRUST_METRICS
from topfed.scoring.ml import (
    fit_isolation_forest,
    fit_logistic,
    fit_random_forest,
)


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    # Benign cluster near 0; malicious cluster shifted away on every feature.
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 0.3, size=(60, 5))
    malicious = rng.normal(3.0, 0.3, size=(20, 5))
    x = np.vstack([benign, malicious])
    y = np.array([False] * 60 + [True] * 20)
    return x, y


def _batch(x: np.ndarray) -> MetricBatch:
    return MetricBatch(
        client_ids=tuple(str(i) for i in range(x.shape[0])),
        metric_names=TRUST_METRICS,
        matrix=x,
    )


def test_logistic_conforms_and_orients_trust() -> None:
    x, y = _toy_data()
    scorer = fit_logistic(x, y, seed=0)
    trust = scorer.score(_batch(x))
    assert trust.shape == (80,)
    assert np.all(trust >= 0.0) and np.all(trust <= 1.0)
    # Benign clients earn higher trust than malicious ones.
    assert trust[:60].mean() > trust[60:].mean()


def test_random_forest_orients_trust() -> None:
    x, y = _toy_data()
    scorer = fit_random_forest(x, y, n_estimators=50, seed=0)
    trust = scorer.score(_batch(x))
    assert trust[:60].mean() > trust[60:].mean()


def test_isolation_forest_flags_outliers_as_low_trust() -> None:
    x, _ = _toy_data()
    scorer = fit_isolation_forest(x, seed=0)
    trust = scorer.score(_batch(x))
    assert np.all(trust >= 0.0) and np.all(trust <= 1.0)
    # The shifted (malicious) cluster is anomalous -> lower trust.
    assert trust[:60].mean() > trust[60:].mean()


def test_empty_batch_returns_empty() -> None:
    x, y = _toy_data()
    scorer = fit_logistic(x, y, seed=0)
    empty = MetricBatch(
        client_ids=(), metric_names=TRUST_METRICS, matrix=np.empty((0, 5))
    )
    assert scorer.score(empty).shape == (0,)
