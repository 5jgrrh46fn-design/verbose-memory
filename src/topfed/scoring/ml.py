"""Learning-based client scorers that conform to the same scoring contract.

These wrap lightweight scikit-learn models so they can stand in for TOPSIS
wherever a :class:`~topfed.core.types.Scorer` is expected: a supervised
classifier (logistic regression or random forest) trained on labelled
behavioural metrics, and an unsupervised isolation-forest anomaly detector.
Each returns a trust score in ``[0, 1]`` (higher = more trustworthy), so the
malicious-ness of a client is ``1 - trust``.

Unlike TOPSIS, these models score each client independently of the others in
the batch — there is no per-round normalisation — so a batch may hold any set
of clients. The supervised variants must be fitted on labelled data before
use; the isolation forest is fitted on (unlabelled) feature rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from topfed.core.types import MetricBatch, TrustScores


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass(frozen=True)
class SupervisedScorer:
    """Trust scorer backed by a fitted probabilistic binary classifier.

    ``trust = 1 - P(malicious)``. ``pos_index`` is the column of
    ``predict_proba`` corresponding to the malicious (positive) class.
    """

    model: Pipeline
    pos_index: int

    def score(self, batch: MetricBatch) -> TrustScores:
        if batch.matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        proba = self.model.predict_proba(batch.matrix)
        malicious = proba[:, self.pos_index]
        return 1.0 - malicious


@dataclass(frozen=True)
class IsolationForestScorer:
    """Trust scorer backed by a fitted isolation forest.

    ``decision_function`` is positive for inliers and negative for outliers;
    a sigmoid maps it monotonically to a ``[0, 1]`` trust score, so more
    anomalous clients receive lower trust.
    """

    model: Pipeline

    def score(self, batch: MetricBatch) -> TrustScores:
        if batch.matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        return _sigmoid(self.model.decision_function(batch.matrix))


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, *, seed: int = 0
) -> SupervisedScorer:
    """Standardised logistic regression with balanced class weights."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
    )
    model.fit(features, labels)
    pos_index = int(np.where(model.classes_)[0][0])
    return SupervisedScorer(model=model, pos_index=pos_index)


def fit_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_estimators: int = 100,
    seed: int = 0,
) -> SupervisedScorer:
    """Random-forest classifier with balanced class weights."""
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    )
    model.fit(features, labels)
    pos_index = int(np.where(model.classes_)[0][0])
    return SupervisedScorer(model=model, pos_index=pos_index)


def fit_isolation_forest(
    features: np.ndarray,
    *,
    contamination: float | str = "auto",
    seed: int = 0,
) -> IsolationForestScorer:
    """Unsupervised isolation forest over standardised features."""
    model = make_pipeline(
        StandardScaler(),
        IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1),
    )
    model.fit(features)
    return IsolationForestScorer(model=model)
