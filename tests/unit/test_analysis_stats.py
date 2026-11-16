"""Unit tests for the non-parametric statistics toolkit."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from topfed.analysis.stats import (
    auc_separability,
    bca_ci,
    detector_summary,
    holm_bonferroni,
    wilcoxon_vs_baseline,
)


def test_wilcoxon_matches_scipy_and_floors_at_2_over_2n() -> None:
    # Five seed-matched pairs, attack always below baseline.
    attack = [0.70, 0.71, 0.69, 0.72, 0.68]
    baseline = [0.75, 0.76, 0.74, 0.77, 0.73]
    res = wilcoxon_vs_baseline(attack, baseline)

    direct = stats.wilcoxon(
        attack, baseline, alternative="two-sided", method="auto"
    )
    assert res.statistic == pytest.approx(direct.statistic)
    assert res.p_value == pytest.approx(direct.pvalue)
    # All differences same sign, n=5: exact two-sided p == 2/2**5.
    assert res.p_value == pytest.approx(2 / 2**5)
    assert res.n == 5


def test_rank_biserial_is_signed_and_extreme_when_uniform() -> None:
    # Attack uniformly worse -> baseline uniformly higher -> r = +1
    # (first arg tends to exceed: here baseline does, so attack-baseline
    # is all negative -> r = -1).
    res = wilcoxon_vs_baseline(
        [0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]
    )
    assert res.rank_biserial == pytest.approx(-1.0)
    assert res.effect_label == "large"

    flipped = wilcoxon_vs_baseline(
        [0.6, 0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert flipped.rank_biserial == pytest.approx(1.0)


def test_wilcoxon_all_zero_differences_is_neutral() -> None:
    res = wilcoxon_vs_baseline([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert res.n == 0
    assert res.p_value == 1.0
    assert res.rank_biserial == 0.0
    assert res.effect_label == "small"


def test_wilcoxon_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        wilcoxon_vs_baseline([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        wilcoxon_vs_baseline([[1.0, 2.0]], [[1.0, 2.0]])


def test_holm_bonferroni_known_example() -> None:
    adj, reject = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # Sorted: 0.005,0.01,0.03,0.04 -> scaled 0.02,0.03,0.06,0.06 (monotone).
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert reject == [True, False, False, True]


def test_holm_bonferroni_empty() -> None:
    assert holm_bonferroni([]) == ([], [])


def test_holm_is_monotone_nondecreasing_in_sorted_order() -> None:
    p = [0.001, 0.2, 0.04, 0.5, 0.009]
    adj, _ = holm_bonferroni(p)
    in_sorted = [adj[i] for i in np.argsort(p)]
    assert all(a <= b + 1e-12 for a, b in zip(in_sorted, in_sorted[1:]))
    assert max(adj) <= 1.0


def test_bca_ci_degenerate_sample_is_zero_width() -> None:
    assert bca_ci([0.4, 0.4, 0.4, 0.4]) == (0.4, 0.4)
    assert bca_ci([0.7]) == (0.7, 0.7)


def test_bca_ci_brackets_mean_and_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    sample = rng.normal(0.75, 0.05, size=200).tolist()
    lo, hi = bca_ci(sample, seed=42)
    assert lo < float(np.mean(sample)) < hi
    assert (lo, hi) == bca_ci(sample, seed=42)  # reproducible


def test_auc_separability_extremes_and_overlap() -> None:
    assert auc_separability([5, 6, 7, 8], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert auc_separability([1, 2, 3, 4], [5, 6, 7, 8]) == pytest.approx(0.0)
    # Identical distributions -> chance-level.
    mid = auc_separability([1, 2, 3, 4], [1, 2, 3, 4])
    assert mid == pytest.approx(0.5)
    assert np.isnan(auc_separability([], [1, 2, 3]))


def test_detector_summary_perfect_separation_higher_positive() -> None:
    res = detector_summary(positive=[5, 6, 7, 8], negative=[1, 2, 3, 4])
    assert res.auc == pytest.approx(1.0)
    assert res.precision == pytest.approx(1.0)
    assert res.recall == pytest.approx(1.0)
    assert res.f1 == pytest.approx(1.0)
    assert res.higher_is_positive is True
    # Threshold in the gap between the two groups.
    assert 4.0 < res.threshold < 5.0


def test_detector_summary_perfect_inverse_auto_flips_direction() -> None:
    # Lower scores are the malicious ones; detector should still report
    # F1 = 1.0 and quote the threshold in the original orientation.
    res = detector_summary(positive=[1, 2, 3, 4], negative=[5, 6, 7, 8])
    assert res.auc == pytest.approx(1.0)
    assert res.f1 == pytest.approx(1.0)
    assert res.higher_is_positive is False
    # In the original orientation, the rule is score <= threshold.
    # Reported threshold sits in the gap.
    assert 4.0 < res.threshold < 5.0


def test_detector_summary_no_separation_returns_low_f1() -> None:
    res = detector_summary(positive=[1, 2, 3, 4], negative=[1, 2, 3, 4])
    assert res.auc == pytest.approx(0.5)
    # With identical distributions, F1 maxes out at predict-all-positive
    # (recall=1, precision=0.5, F1≈0.67).
    assert res.f1 == pytest.approx(2 / 3, abs=1e-9)


def test_detector_summary_overlapping_distributions() -> None:
    # Two overlapping groups: malicious skewed higher.
    res = detector_summary(
        positive=[3, 4, 5, 6, 7], negative=[1, 2, 3, 4, 5]
    )
    assert 0.5 < res.auc < 1.0
    assert 0.0 < res.f1 < 1.0
    assert res.precision >= res.recall * 0.5  # not pathological
    assert res.higher_is_positive is True


def test_detector_summary_empty_groups_returns_degenerate() -> None:
    res = detector_summary(positive=[], negative=[1, 2, 3])
    assert np.isnan(res.auc)
    assert np.isnan(res.precision)
    assert res.f1 == 0.0


def test_detector_summary_constant_signal_is_degenerate() -> None:
    res = detector_summary(positive=[2.0, 2.0, 2.0], negative=[2.0, 2.0])
    assert res.f1 == 0.0
    # AUC degenerate (Mann-Whitney on ties) but not NaN — both groups present.
    assert not np.isnan(res.auc)


def test_detector_summary_threshold_recovers_predict_all_positive_when_optimal() -> None:
    # All positives equal to all negatives' max -> the best detector is
    # threshold strictly above all data, which gives F1=0; OR predict-all
    # gives precision=0.5 recall=1.0 F1=2/3. The latter wins.
    res = detector_summary(positive=[5, 5, 5], negative=[1, 2, 5])
    assert res.f1 >= 2 / 3 - 1e-9
