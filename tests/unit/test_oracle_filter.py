"""Ground-truth oracle integrator."""

from __future__ import annotations

import numpy as np
import pytest

from topfed.integration.oracle import OracleFilter


def test_oracle_zero_weights_known_attackers_regardless_of_score() -> None:
    integ = OracleFilter(malicious_clients=frozenset({"c1", "c3"}))
    ids = ("c0", "c1", "c2", "c3")
    # Attackers get the *highest* scores; the oracle must still drop them.
    scores = np.array([0.1, 0.99, 0.2, 0.95])
    result = integ.integrate(1, ids, scores)
    assert result.weights == pytest.approx([0.5, 0.0, 0.5, 0.0])
    assert result.sampling_probs is None


def test_oracle_logs_scores_but_uses_none_of_them() -> None:
    integ = OracleFilter(malicious_clients=frozenset({"c1"}))
    integ.integrate(1, ("c0", "c1"), np.array([0.3, 0.9]))
    rows = {r["client_id"]: r for r in integ.records}
    assert rows["c1"]["score"] == pytest.approx(0.9)
    assert rows["c1"]["weight"] == 0.0 and not rows["c1"]["included"]
    assert rows["c0"]["weight"] == pytest.approx(1.0) and rows["c0"]["included"]


def test_oracle_all_malicious_cohort_falls_back_to_equal_weights() -> None:
    integ = OracleFilter(malicious_clients=frozenset({"c0", "c1"}))
    result = integ.integrate(1, ("c0", "c1"), np.array([0.5, 0.5]))
    assert result.weights == pytest.approx([0.5, 0.5])


def test_oracle_is_memoryless_across_rounds() -> None:
    integ = OracleFilter(malicious_clients=frozenset({"c9"}))
    integ.integrate(1, ("c0", "c9"), np.array([0.0, 1.0]))
    result = integ.integrate(2, ("c0", "c2"), np.array([0.0, 0.0]))
    assert result.weights == pytest.approx([0.5, 0.5])
