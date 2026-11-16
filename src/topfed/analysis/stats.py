"""Non-parametric statistics for small seed-replicated experiments.

Each experimental cell is repeated with a handful of independent random
seeds, so sample sizes are tiny and distributional assumptions cannot be
verified. Every routine here is therefore distribution-free.

The toolkit:

* :func:`wilcoxon_vs_baseline` -- two-sided paired Wilcoxon signed-rank
  test over seed-matched outcomes, returned together with the matched-pairs
  rank-biserial correlation as an effect size.
* :func:`bca_ci` -- bias-corrected and accelerated bootstrap confidence
  interval, for metrics with enough per-round samples to bootstrap finely.
* :func:`holm_bonferroni` -- Holm sequential family-wise error control,
  uniformly more powerful than classical Bonferroni.
* :func:`auc_separability` -- AUC of a single behavioural signal as a
  univariate adversarial-vs-benign discriminator, via the Mann-Whitney U
  identity (no scikit-learn dependency).

A note on power: with ``S`` seed-matched pairs all differing in the same
direction, the exact two-sided signed-rank p-value bottoms out at
``2 / 2**S``. For ``S = 5`` that floor is ``0.0625`` -- already above
``alpha = 0.05`` -- so seed-level significance is unreachable at five seeds
and the rank-biserial effect size plus bootstrap intervals carry the
practical interpretation. Callers should surface this rather than read a
non-significant p-value as "no effect".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def _effect_label(r: float) -> str:
    """Standard small/medium/large bands for an absolute correlation."""
    a = abs(r)
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


@dataclass(frozen=True)
class PairedTest:
    """Outcome of one paired Wilcoxon signed-rank comparison.

    ``n`` is the number of non-zero seed-matched differences actually used
    (zero differences are discarded by the signed-rank test). ``p_value``
    is two-sided. ``rank_biserial`` is in ``[-1, 1]``: positive means the
    first sample tended to exceed the baseline.
    """

    n: int
    statistic: float
    p_value: float
    rank_biserial: float
    effect_label: str


def wilcoxon_vs_baseline(
    attack: Sequence[float], baseline: Sequence[float]
) -> PairedTest:
    """Two-sided paired Wilcoxon signed-rank test with a rank-biserial effect size.

    ``attack`` and ``baseline`` must be seed-matched and equal length. The
    exact null distribution is used for small samples (scipy's default
    ``method="auto"``). If every paired difference is zero the test is
    undefined; a neutral non-significant result is returned instead of
    raising.
    """
    a = np.asarray(attack, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"attack/baseline shape mismatch: {a.shape} vs {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError("inputs must be 1-D seed-level vectors")

    diff = a - b
    nonzero = diff[diff != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return PairedTest(
            n=0,
            statistic=0.0,
            p_value=1.0,
            rank_biserial=0.0,
            effect_label=_effect_label(0.0),
        )

    result = stats.wilcoxon(
        a,
        b,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )

    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    total = w_plus + w_minus  # == n(n+1)/2 since zeros are dropped
    rank_biserial = (w_plus - w_minus) / total if total > 0 else 0.0

    return PairedTest(
        n=n,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        rank_biserial=rank_biserial,
        effect_label=_effect_label(rank_biserial),
    )


def bca_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI for a 1-D sample.

    Returns ``(low, high)``. A degenerate sample (fewer than two points, or
    all values identical) has no bootstrap variation; the point estimate is
    returned as a zero-width interval rather than raising.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2 or bool(np.allclose(v, v.flat[0])):
        point = float(statistic(v)) if v.size else float("nan")
        return (point, point)

    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (v,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="BCa",
        random_state=rng,
    )
    ci = res.confidence_interval
    return (float(ci.low), float(ci.high))


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """Holm sequential family-wise correction.

    Returns ``(adjusted_pvalues, reject)`` aligned to the input order.
    Adjusted p-values are the step-down ``(k - i) * p`` values made
    monotone non-decreasing and clipped to ``1``; ``reject[j]`` is
    ``adjusted_pvalues[j] < alpha``.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    k = p.size
    if k == 0:
        return [], []

    order = np.argsort(p, kind="stable")
    adjusted = np.empty(k, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (k - rank) * float(p[idx])
        running = max(running, scaled)
        adjusted[idx] = min(running, 1.0)

    reject = [bool(adjusted[j] < alpha) for j in range(k)]
    return adjusted.tolist(), reject


def auc_separability(
    positive: Sequence[float], negative: Sequence[float]
) -> float:
    """AUC of a scalar signal separating a positive from a negative group.

    Computed from the Mann-Whitney U statistic: ``AUC = U / (n_pos *
    n_neg)``. ``0.5`` means no separation; values toward ``1`` mean the
    positive (adversarial) group scores higher, toward ``0`` lower. Returns
    NaN if either group is empty.
    """
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u) / (pos.size * neg.size)


@dataclass(frozen=True)
class DiscriminationSummary:
    """Operational detector performance at the F1-optimal threshold.

    A behavioural signal is treated as a univariate binary classifier
    over (malicious, benign) client rows. The decision rule is
    ``score >= threshold`` when ``higher_is_positive`` is ``True``,
    flipped otherwise. The reported threshold maximizes F1 over the
    union of observed score values; ties broken by lower threshold
    (more conservative).

    ``auc`` is the standard Mann-Whitney AUC of the signal in the
    canonical (positive >= threshold) orientation; it equals 1 - raw
    AUC when the signal turned out to be a "lower-is-positive" detector
    and was internally flipped.
    """

    auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    higher_is_positive: bool


def detector_summary(
    positive: Sequence[float], negative: Sequence[float]
) -> DiscriminationSummary:
    """Threshold-optimized precision / recall / F1 for a univariate signal.

    The direction (whether higher scores predict the positive class) is
    inferred from the raw AUC: if the signal is anti-correlated with
    the positive class (AUC < 0.5), the score is internally negated so
    that the reported threshold is always quoted in the original
    ``score >= threshold`` orientation.

    The threshold sweep runs over the midpoints between consecutive
    unique observed scores plus the extremes, so every distinct
    confusion matrix is enumerated exactly once.

    Returns a degenerate summary (NaNs, F1 = 0, threshold from the
    centre of the range) if either group is empty or if both groups are
    identical singletons.
    """
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return DiscriminationSummary(
            auc=float("nan"),
            threshold=float("nan"),
            precision=float("nan"),
            recall=float("nan"),
            f1=0.0,
            higher_is_positive=True,
        )

    raw_auc = auc_separability(pos.tolist(), neg.tolist())
    higher_is_positive = raw_auc >= 0.5
    if not higher_is_positive:
        pos = -pos
        neg = -neg

    all_scores = np.concatenate([pos, neg])
    uniq = np.unique(all_scores)
    if uniq.size == 1:
        # Constant signal cannot discriminate.
        return DiscriminationSummary(
            auc=max(raw_auc, 1.0 - raw_auc),
            threshold=float(uniq[0]),
            precision=float("nan"),
            recall=float("nan"),
            f1=0.0,
            higher_is_positive=higher_is_positive,
        )

    # Candidate thresholds: midpoints between consecutive unique
    # scores, plus one below the minimum (predict-all-positive) and one
    # above the maximum (predict-none-positive — included only to
    # bookend the sweep; F1 = 0 there).
    midpoints = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate(([uniq[0] - 1.0], midpoints))

    best_f1 = -1.0
    best_threshold = float(candidates[0])
    best_precision = float("nan")
    best_recall = float("nan")
    n_pos = pos.size
    for t in candidates:
        tp = int(np.sum(pos >= t))
        fp = int(np.sum(neg >= t))
        if tp == 0:
            f1 = 0.0
            precision = 0.0 if (tp + fp) > 0 else float("nan")
            recall = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / n_pos
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_precision = float(precision)
            best_recall = float(recall)

    # Quote the threshold in the original orientation.
    quoted_threshold = (
        best_threshold if higher_is_positive else -best_threshold
    )
    return DiscriminationSummary(
        auc=max(raw_auc, 1.0 - raw_auc),
        threshold=quoted_threshold,
        precision=best_precision,
        recall=best_recall,
        f1=best_f1,
        higher_is_positive=higher_is_positive,
    )
