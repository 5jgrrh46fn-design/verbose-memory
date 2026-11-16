"""Benchmark TOPSIS against lightweight ML detectors on identical metrics.

Reads the recorded client-metric ledgers under ``results/`` and compares the
TOPSIS composite score against learning-based detectors trained on the same
five behavioural metrics, along two axes:

* **In-distribution (stratified k-fold).** The supervised models are trained
  and tested on folds of the *same* attack — an upper bound on what a
  classifier achieves when it has seen the attack it faces.
* **Novel attack (leave-one-attack-out).** The supervised models are trained
  on every *other* attack and tested on the held-out one — the honest measure
  of generalisation to an attack absent from the training data.

TOPSIS (with the CRITIC weighting) is reported alongside as the reference: it
uses no labels and no training, so it has a single column that applies in both
regimes. The isolation forest is unsupervised; it is fitted on the held-out
attack's own features (no labels), so it has one column.

This module only reads ``results/``; ``--out`` is rejected inside the results
tree.

Usage::

    uv run python -m topfed.analysis.ml_benchmark --out tables/ml-benchmark.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from topfed.analysis.aggregate import SeedRun, discover_runs
from topfed.analysis.scoring import _windowed_rows, topsis_pooled
from topfed.analysis.stats import DiscriminationSummary, detector_summary
from topfed.core.types import MetricBatch, Scorer
from topfed.scoring.criteria import TRUST_METRICS
from topfed.scoring.ml import (
    SupervisedScorer,
    fit_isolation_forest,
    fit_logistic,
    fit_random_forest,
)
from topfed.scoring.topsis import TopsisScorer
from topfed.scoring.weighting import critic_weights

SEED = 0


def build_feature_sets(
    runs: dict[str, list[SeedRun]], baseline_name: str, window: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-attack ``(features, labels)`` over complete client-metric rows."""
    sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rs in runs.items():
        if name == baseline_name:
            continue
        attack_runs = [r for r in rs if r.has_malicious]
        if not attack_runs:
            continue
        feats: list[list[float]] = []
        labels: list[bool] = []
        for run in attack_runs:
            for row in _windowed_rows(run, window):
                if any(row.get(m) is None for m in TRUST_METRICS):
                    continue
                feats.append([float(row[m]) for m in TRUST_METRICS])
                labels.append(bool(row["is_malicious"]))
        x = np.asarray(feats, dtype=np.float64)
        y = np.asarray(labels, dtype=bool)
        if x.size and y.any() and (~y).any():
            sets[name] = (x, y)
    return sets


def _summary_from_scores(
    malicious_score: np.ndarray, labels: np.ndarray
) -> DiscriminationSummary:
    return detector_summary(
        malicious_score[labels].tolist(), malicious_score[~labels].tolist()
    )


def _malicious_score(scorer: Scorer, x: np.ndarray) -> np.ndarray:
    batch = MetricBatch(
        client_ids=tuple(str(i) for i in range(x.shape[0])),
        metric_names=TRUST_METRICS,
        matrix=x,
    )
    trust = np.asarray(scorer.score(batch), dtype=np.float64)
    return 1.0 - trust


def logistic_kfold(x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> DiscriminationSummary:
    return _supervised_kfold(x, y, fit_logistic, n_splits)


def forest_kfold(x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> DiscriminationSummary:
    return _supervised_kfold(x, y, fit_random_forest, n_splits)


def _supervised_kfold(
    x: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[..., SupervisedScorer],
    n_splits: int,
) -> DiscriminationSummary:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = np.empty(x.shape[0], dtype=np.float64)
    for train_idx, test_idx in splitter.split(x, y):
        scorer: SupervisedScorer = fit_fn(x[train_idx], y[train_idx], seed=SEED)
        scores[test_idx] = _malicious_score(scorer, x[test_idx])
    return _summary_from_scores(scores, y)


def isolation_kfold(x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> DiscriminationSummary:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = np.empty(x.shape[0], dtype=np.float64)
    for train_idx, test_idx in splitter.split(x, y):
        scorer = fit_isolation_forest(x[train_idx], seed=SEED)
        scores[test_idx] = _malicious_score(scorer, x[test_idx])
    return _summary_from_scores(scores, y)


def supervised_loao(
    sets: dict[str, tuple[np.ndarray, np.ndarray]],
    held_out: str,
    fit_fn: Callable[..., SupervisedScorer],
) -> DiscriminationSummary:
    others = [name for name in sets if name != held_out]
    x_train = np.vstack([sets[name][0] for name in others])
    y_train = np.concatenate([sets[name][1] for name in others])
    x_test, y_test = sets[held_out]
    if not (y_train.any() and (~y_train).any()):
        return detector_summary([], [])
    scorer: SupervisedScorer = fit_fn(x_train, y_train, seed=SEED)
    return _summary_from_scores(_malicious_score(scorer, x_test), y_test)


def topsis_critic_auc(runs: list[SeedRun], window: int) -> DiscriminationSummary:
    mal, ben = topsis_pooled(runs, TopsisScorer(weighting=critic_weights), window)
    return detector_summary(mal, ben)


@dataclass(frozen=True)
class BenchmarkRow:
    experiment: str
    topsis_critic: float
    logreg_kfold: float
    logreg_loao: float
    forest_kfold: float
    forest_loao: float
    isolation: float


def build_rows(
    runs: dict[str, list[SeedRun]], baseline_name: str, window: int
) -> list[BenchmarkRow]:
    sets = build_feature_sets(runs, baseline_name, window)
    rows: list[BenchmarkRow] = []
    for name in sorted(sets):
        x, y = sets[name]
        attack_runs = [r for r in runs[name] if r.has_malicious]
        can_loao = len(sets) > 1
        rows.append(
            BenchmarkRow(
                experiment=name,
                topsis_critic=topsis_critic_auc(attack_runs, window).auc,
                logreg_kfold=logistic_kfold(x, y).auc,
                logreg_loao=(
                    supervised_loao(sets, name, fit_logistic).auc
                    if can_loao
                    else float("nan")
                ),
                forest_kfold=forest_kfold(x, y).auc,
                forest_loao=(
                    supervised_loao(sets, name, fit_random_forest).auc
                    if can_loao
                    else float("nan")
                ),
                isolation=isolation_kfold(x, y).auc,
            )
        )
    return rows


def family_of(experiment: str) -> str:
    """Collapse an experiment name to its attack family."""
    if "backdoor" in experiment:
        return "backdoor"
    if experiment.startswith("gradient_scaling"):
        return "gradient_scaling"
    if experiment.startswith("sign_flip"):
        return "sign_flip"
    if experiment.startswith("dirty_label"):
        return "dirty_label"
    if experiment.startswith("clean_label"):
        return "clean_label"
    return experiment


def build_family_sets(
    runs: dict[str, list[SeedRun]], baseline_name: str, window: int
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[SeedRun]]]:
    """Per-family ``(features, labels)`` plus the runs grouped by family."""
    feats: dict[str, list[list[float]]] = defaultdict(list)
    labels: dict[str, list[bool]] = defaultdict(list)
    runs_by_family: dict[str, list[SeedRun]] = defaultdict(list)
    for name, rs in runs.items():
        if name == baseline_name:
            continue
        attack_runs = [r for r in rs if r.has_malicious]
        if not attack_runs:
            continue
        family = family_of(name)
        runs_by_family[family].extend(attack_runs)
        for run in attack_runs:
            for row in _windowed_rows(run, window):
                if any(row.get(m) is None for m in TRUST_METRICS):
                    continue
                feats[family].append([float(row[m]) for m in TRUST_METRICS])
                labels[family].append(bool(row["is_malicious"]))
    sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family in feats:
        x = np.asarray(feats[family], dtype=np.float64)
        y = np.asarray(labels[family], dtype=bool)
        if x.size and y.any() and (~y).any():
            sets[family] = (x, y)
    return sets, runs_by_family


@dataclass(frozen=True)
class FamilyRow:
    family: str
    topsis_critic: float
    logreg_lofo: float
    forest_lofo: float


def build_family_rows(
    runs: dict[str, list[SeedRun]], baseline_name: str, window: int
) -> list[FamilyRow]:
    sets, runs_by_family = build_family_sets(runs, baseline_name, window)
    rows: list[FamilyRow] = []
    for family in sorted(sets):
        can_lofo = len(sets) > 1
        rows.append(
            FamilyRow(
                family=family,
                topsis_critic=topsis_critic_auc(runs_by_family[family], window).auc,
                logreg_lofo=(
                    supervised_loao(sets, family, fit_logistic).auc
                    if can_lofo
                    else float("nan")
                ),
                forest_lofo=(
                    supervised_loao(sets, family, fit_random_forest).auc
                    if can_lofo
                    else float("nan")
                ),
            )
        )
    return rows


def _fmt(value: float) -> str:
    return f"{value:.3f}" if not np.isnan(value) else "n/a"


def build_report(
    runs: dict[str, list[SeedRun]], baseline_name: str, window: int
) -> str:
    rows = build_rows(runs, baseline_name, window)
    pooled_desc = f"last {window} rounds" if window > 0 else "all rounds"
    lines: list[str] = []
    lines.append("# TOPSIS vs. ML detector benchmark\n")
    lines.append(
        f"Detector AUC over the {pooled_desc}, pooled across seeds. All models "
        "use the same five behavioural metrics. **TOPSIS-critic** uses no "
        "labels and no training. **k-fold** is in-distribution (the supervised "
        "model has seen the attack). **LOAO** (leave-one-attack-out) trains on "
        "every other attack and tests on this one — the novel-attack measure. "
        "**IsoForest** is unsupervised (fitted on this attack's features).\n"
    )
    header = (
        "| experiment | TOPSIS-critic | LogReg k-fold | LogReg LOAO | "
        "RF k-fold | RF LOAO | IsoForest |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 7)
    for row in rows:
        lines.append(
            f"| `{row.experiment}` | {_fmt(row.topsis_critic)} | "
            f"{_fmt(row.logreg_kfold)} | {_fmt(row.logreg_loao)} | "
            f"{_fmt(row.forest_kfold)} | {_fmt(row.forest_loao)} | "
            f"{_fmt(row.isolation)} |"
        )
    lines.append("")
    lines.append(
        "> The gap between each model's k-fold and LOAO columns is its "
        "generalisation cost to an unseen attack; TOPSIS-critic pays no such "
        "cost because it is not trained. Note that LOAO leaves out a single "
        "configuration while sibling configs of the same family remain in "
        "training, so it understates true novelty — see the family table.\n"
    )

    # --- Leave-one-family-out (true novel-attack test) -------------------
    family_rows = build_family_rows(runs, baseline_name, window)
    lines.append("## Leave-one-family-out (true novel-attack test)\n")
    lines.append(
        "Detector AUC on a held-out attack *family* whose every configuration "
        "is removed from the supervised training set — the honest measure of "
        "generalisation to an attack type never seen before. TOPSIS-critic is "
        "the same untrained reference (pooled over the family).\n"
    )
    lines.append("| held-out family | TOPSIS-critic | LogReg LOFO | RF LOFO |")
    lines.append("|" + "---|" * 4)
    for frow in family_rows:
        lines.append(
            f"| `{frow.family}` | {_fmt(frow.topsis_critic)} | "
            f"{_fmt(frow.logreg_lofo)} | {_fmt(frow.forest_lofo)} |"
        )
    lines.append("")
    lines.append(
        "> When a whole family is held out, the supervised models can only "
        "transfer signatures learned from *other* families. Where their LOFO "
        "AUC falls toward TOPSIS-critic (or below it), the labelled-data "
        "advantage does not survive a genuinely novel attack.\n"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--baseline", default="benign_cifar10")
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=0,
        help="pool only the last N rounds of each run (0 = all rounds)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional Markdown destination; must be outside results-root",
    )
    args = parser.parse_args(argv)

    results_root: Path = args.results_root.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")

    if args.out is not None:
        out = args.out.resolve()
        if results_root == out or results_root in out.parents:
            raise SystemExit(f"refusing to write inside the results tree: {out}")

    runs = discover_runs(results_root)
    if not runs:
        raise SystemExit(f"no runs discovered under {results_root}")

    report = build_report(runs, args.baseline, args.convergence_window)
    if args.out is not None:
        args.out.write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
