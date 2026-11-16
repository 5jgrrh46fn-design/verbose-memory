"""MESAS: metric cascades with statistical pruning (Krauss and Dmitrienko, CCS 2023).

IMPLEMENTED FROM THE PAPER. MESAS has no public code release, so this module
is a from-scratch reading of the paper's Section 4.3 and Appendix E. Every
choice the paper leaves open is listed here and should be reported alongside
any result that uses it:

* **Metrics** (Eq. 6 to 11), per client against the round's global model
  ``G``: COS, the cosine distance between the flattened local model and
  ``G`` (Eq. 6 is written over ``L - G``, the text and Fig. 4 describe the
  angle between ``L`` and ``G``; we follow the text); EUCL, the L2 norm of
  ``L - G``; COUNT, the number of parameters that increased; VAR, the
  variance of the flattened local model; MIN, the smallest non-zero
  ``|L - G|``; MAX, the largest ``|L - G|``. Each metric is computed for the
  whole model and for every parameter array separately (the paper's "each
  layer individually"), giving ``6 * (1 + number of arrays)`` series.
* **Significance test** per series over the still-retained clients: split
  the values around a centre into ``l1`` (above) and ``l2`` (below) as
  absolute distances to the centre; run Welch's t-test, Levene's test and
  the two-sample Kolmogorov-Smirnov test on ``l1`` vs ``l2``; the series is
  significant when any p-value is below ``alpha`` or when any value lies
  outside mean +- 3 sd of the series. A test that needs two values per
  list is skipped when a list is shorter.
* **Pruning**: for every significant series, agglomerative clustering (two
  clusters, Euclidean, Ward linkage via scikit-learn's default) on the
  one-dimensional values; the smaller cluster's clients are marked. All
  clients marked in a pass are removed together and the pass repeats on
  the survivors until no series is significant. No series removes its
  larger cluster, and a tie removes nothing, but the union over series in
  one pass can exceed half the cohort; the loop stops before fewer than
  two clients remain.
* **Aggregation**: equal-weight average of the retained local models
  (the paper uses equal weights, footnote 1, and a global learning rate 1).
* **Centre** of the split: Section 4.3 says the median twice ("the
  statistical tests first extract the median value, which is considered as
  benign due to the majority assumption"); the caption of Fig. 6 says the
  mean. ``center="median"`` is the default; ``"mean"`` is available because
  the choice decides whether a few large outliers are caught: with ten
  clients and two attackers, a median split leaves both in a five-element
  list whose variance hides them, a mean split isolates them.
* **alpha** defaults to 0.03, the paper's value for inter-client non-IID
  data (0.001 intra-client non-IID, 0.0001 IID).

Values equal to the centre belong to neither list. Zero-variance series
(every retained value identical) are never significant. With ``n`` values
the 3-sigma rule cannot fire below ``n = 11`` (the largest possible z-score
is ``(n - 1) / sqrt(n)``), so at ten clients per round only the three
two-sample tests act.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats
from sklearn.cluster import AgglomerativeClustering

from topfed.core.types import ClientUpdate, NDArrays

METRIC_NAMES = ("cos", "eucl", "count", "var", "min", "max")


def _flatten(parameters: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parameters])


def _six_metrics(local: np.ndarray, global_: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """The six MESAS metrics of one flattened local model against the global."""
    diff = local - global_
    denom = float(np.linalg.norm(local) * np.linalg.norm(global_))
    cos = 1.0 - float(local @ global_) / denom if denom > eps else 1.0
    absdiff = np.abs(diff)
    nonzero = absdiff[absdiff > 0]
    return np.array(
        [
            cos,
            float(np.linalg.norm(diff)),
            float(np.count_nonzero(diff > 0)),
            float(np.var(local)),
            float(nonzero.min()) if nonzero.size else 0.0,
            float(absdiff.max()) if absdiff.size else 0.0,
        ]
    )


def mesas_metrics(updates: list[ClientUpdate], global_parameters: NDArrays) -> np.ndarray:
    """Metric matrix of shape ``(n_clients, 6 * (1 + n_arrays))``.

    Column block 0 is the whole model; block ``k >= 1`` is parameter array
    ``k - 1``. Within a block the order is :data:`METRIC_NAMES`.
    """
    flat_global = _flatten(global_parameters)
    rows = []
    for u in updates:
        flat_local = _flatten(u.parameters)
        blocks = [_six_metrics(flat_local, flat_global)]
        for p, g in zip(u.parameters, global_parameters, strict=True):
            blocks.append(
                _six_metrics(
                    np.asarray(p, dtype=np.float64).ravel(),
                    np.asarray(g, dtype=np.float64).ravel(),
                )
            )
        rows.append(np.concatenate(blocks))
    return np.stack(rows)


CENTERS = ("median", "mean")


def series_is_significant(
    values: np.ndarray, alpha: float, center: str = "median"
) -> tuple[bool, dict[str, float]]:
    """MESAS's significance analysis of one metric series.

    Returns ``(significant, p)`` where ``p`` holds the p-values that were
    computed (``t``, ``levene``, ``ks``) and ``sigma`` = the number of values
    outside mean +- 3 sd. ``center`` is the split point, ``"median"`` or
    ``"mean"`` (see the module docstring).
    """
    if center not in CENTERS:
        raise ValueError(f"center must be one of {CENTERS}; got {center!r}")
    v = np.asarray(values, dtype=np.float64)
    out: dict[str, float] = {}
    if v.size < 2 or np.ptp(v) == 0.0:
        return False, out
    med = float(np.median(v)) if center == "median" else float(v.mean())
    l1 = np.abs(v[v > med] - med)
    l2 = np.abs(v[v < med] - med)
    significant = False
    if l1.size >= 2 and l2.size >= 2:
        for name, test in (
            ("t", lambda a, b: stats.ttest_ind(a, b, equal_var=False)),
            ("levene", stats.levene),
            ("ks", stats.ks_2samp),
        ):
            # Near-identical lists make scipy warn about precision loss or a
            # zero denominator; the resulting nan p-value is read as "not
            # significant", which is the right call for a flat series.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = float(test(l1, l2).pvalue)
            if not np.isfinite(p):
                p = 1.0
            out[name] = p
            significant = significant or p < alpha
    sd = float(v.std())
    outliers = int(np.count_nonzero(np.abs(v - v.mean()) > 3.0 * sd)) if sd > 0 else 0
    out["sigma"] = float(outliers)
    return significant or outliers > 0, out


def smaller_cluster(values: np.ndarray) -> list[int]:
    """Indices of the smaller of two agglomerative clusters over 1-D values.

    Returns an empty list when the clusters tie, so a tie never prunes.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if v.shape[0] < 2 or np.ptp(v) == 0.0:
        return []
    labels = AgglomerativeClustering(n_clusters=2).fit_predict(v)
    n1 = int(np.count_nonzero(labels == 1))
    n0 = labels.shape[0] - n1
    if n0 == n1:
        return []
    minority = 1 if n1 < n0 else 0
    return [int(i) for i in np.flatnonzero(labels == minority)]


def mesas_prune(
    metrics: np.ndarray, alpha: float, center: str = "median"
) -> tuple[list[int], list[dict[str, Any]]]:
    """Run the pruning loop; return the retained client indices and a trace.

    ``metrics`` is the output of :func:`mesas_metrics`. The trace has one
    entry per (pass, significant series) with the clients it marked.
    """
    n = metrics.shape[0]
    retained = list(range(n))
    trace: list[dict[str, Any]] = []
    pass_num = 0
    while len(retained) >= 3:
        pass_num += 1
        marked: set[int] = set()
        for col in range(metrics.shape[1]):
            values = metrics[retained, col]
            significant, p = series_is_significant(values, alpha, center)
            if not significant:
                continue
            local_idx = smaller_cluster(values)
            clients = [retained[i] for i in local_idx]
            marked.update(clients)
            trace.append({"pass": pass_num, "series": col, "clients": clients, **p})
        if not marked:
            break
        survivors = [i for i in retained if i not in marked]
        if len(survivors) < 2:
            break
        retained = survivors
    return retained, trace


class MESAS:
    """MESAS aggregation: six metrics, statistical cascade, prune, average.

    ``last_weights`` holds the round's aggregation weights (``1 / L`` for the
    ``L`` retained clients, 0 for the pruned). ``records`` holds one row per
    client per round (round, client id, the six whole-model metrics, the pass
    in which the client was pruned or 0, retained flag) for offline analysis.
    """

    def __init__(self, alpha: float = 0.03, center: str = "median") -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if center not in CENTERS:
            raise ValueError(f"center must be one of {CENTERS}; got {center!r}")
        self.alpha = float(alpha)
        self.center = center
        self.last_weights: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []

    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> NDArrays:
        if not updates:
            raise ValueError("MESAS requires at least one ClientUpdate")
        n = len(updates)
        metrics = mesas_metrics(updates, global_parameters)
        retained, trace = mesas_prune(metrics, self.alpha, self.center)

        pruned_in: dict[int, int] = {}
        for entry in trace:
            for c in entry["clients"]:
                pruned_in.setdefault(int(c), int(entry["pass"]))
        weights = np.zeros(n, dtype=np.float64)
        weights[retained] = 1.0 / len(retained)
        self.last_weights = weights
        retained_set = set(retained)
        for i, u in enumerate(updates):
            row: dict[str, Any] = {
                "round": round_num,
                "client_id": u.client_id,
                "retained": i in retained_set,
                "pruned_pass": pruned_in.get(i, 0) if i not in retained_set else 0,
            }
            for name, value in zip(METRIC_NAMES, metrics[i, :6], strict=True):
                row[f"m_{name}"] = float(value)
            self.records.append(row)

        out: NDArrays = []
        for k, g in enumerate(global_parameters):
            stacked = np.stack([np.asarray(updates[i].parameters[k]) for i in retained])
            out.append(stacked.mean(axis=0).astype(np.asarray(g).dtype))
        return out
