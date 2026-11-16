"""Offline TOPSIS scoring evaluation over recorded client-metric ledgers.

Reads the per-run ledgers under ``results/`` (the same ones the attack-potency
report consumes) and asks the central question of the scoring phase: does a
multi-criteria TOPSIS score separate malicious from benign clients better than
the best single behavioural metric, which weighting strategy does best, and
how sensitive are the resulting rankings to the weights?

For every attack experiment, and for each weighting strategy, the per-round
decision matrix is rebuilt from the recorded metrics, scored with TOPSIS, and
the resulting closeness coefficients are pooled across rounds and seeds. Each
pooled score set is then treated as a univariate malicious-vs-benign detector
(reusing the same AUC / F1 machinery as the single-metric analysis), so TOPSIS
and the individual metrics are compared on an identical footing. A weight
sensitivity section perturbs each metric weight one at a time and reports how
stable the TOPSIS ranking is under each strategy.

This module only reads ``results/``; it never writes there. ``--out`` is
rejected if it points inside the results tree.

Usage::

    uv run python -m topfed.analysis.scoring
    uv run python -m topfed.analysis.scoring --convergence-window 50 \\
        --out tables/scoring-eval.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from topfed.analysis.aggregate import SeedRun, discover_runs
from topfed.analysis.stats import detector_summary
from topfed.scoring.criteria import TRUST_METRICS, benefit_mask, build_batch
from topfed.scoring.sensitivity import DEFAULT_LEVELS, oat_sensitivity
from topfed.scoring.topsis import TopsisScorer
from topfed.scoring.weighting import WEIGHTING_STRATEGIES

STRATEGY_ORDER: tuple[str, ...] = (
    "equal",
    "consistency",
    "magnitude",
    "stability",
    "entropy",
    "critic",
)


def _windowed_rows(run: SeedRun, window: int) -> list[dict]:
    """Client rows from the last ``window`` rounds of a run (all if window<=0)."""
    if window <= 0 or not run.client_rows:
        return run.client_rows
    max_round = max(int(r["round"]) for r in run.client_rows)
    lower = max_round - window
    return [r for r in run.client_rows if int(r["round"]) > lower]


def _grouped_windowed(run: SeedRun, window: int) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in _windowed_rows(run, window):
        grouped[int(row["round"])].append(row)
    return grouped


def _complete_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if all(r.get(m) is not None for m in TRUST_METRICS)]


def topsis_pooled(
    runs: list[SeedRun], scorer: TopsisScorer, window: int = 0
) -> tuple[list[float], list[float]]:
    """Pool per-round TOPSIS closeness into (malicious, benign) score lists.

    Each round's participating clients form one decision matrix; the scorer
    runs once per round per seed and the per-client closeness values are
    collected by ground-truth label.
    """
    malicious: list[float] = []
    benign: list[float] = []
    for run in runs:
        for rows in _grouped_windowed(run, window).values():
            usable = _complete_rows(rows)
            if not usable:
                continue
            scores = scorer.score(build_batch(usable, TRUST_METRICS))
            for row, score in zip(usable, scores, strict=True):
                (malicious if row["is_malicious"] else benign).append(float(score))
    return malicious, benign


def _single_metric_pooled(
    runs: list[SeedRun], metric: str, window: int = 0
) -> tuple[list[float], list[float]]:
    malicious: list[float] = []
    benign: list[float] = []
    for run in runs:
        for row in _windowed_rows(run, window):
            value = row.get(metric)
            if value is None:
                continue
            (malicious if row["is_malicious"] else benign).append(float(value))
    return malicious, benign


@dataclass(frozen=True)
class ScoringRow:
    experiment: str
    best_single_metric: str
    best_single_auc: float
    best_single_f1: float
    topsis_auc: dict[str, float]
    topsis_f1: dict[str, float]


def _best_single(runs: list[SeedRun], window: int) -> tuple[str, float, float]:
    # Seed at chance (AUC 0.5, distance 0) so any informative metric wins.
    best_name = ""
    best_auc = 0.5
    best_f1 = -1.0
    for metric in TRUST_METRICS:
        mal, ben = _single_metric_pooled(runs, metric, window)
        summary = detector_summary(mal, ben)
        if np.isnan(summary.auc):
            continue
        # Rank single metrics by AUC distance from chance, breaking ties by F1.
        if abs(summary.auc - 0.5) > abs(best_auc - 0.5) or (
            np.isclose(summary.auc, best_auc) and summary.f1 > best_f1
        ):
            best_name = metric
            best_auc = summary.auc
            best_f1 = summary.f1
    return best_name, best_auc, best_f1


def build_scoring_rows(
    runs: dict[str, list[SeedRun]], baseline_name: str, window: int
) -> list[ScoringRow]:
    attacks = sorted(
        name
        for name, rs in runs.items()
        if name != baseline_name and any(r.has_malicious for r in rs)
    )
    rows: list[ScoringRow] = []
    for name in attacks:
        rs = [r for r in runs[name] if r.has_malicious]
        best_name, best_auc, best_f1 = _best_single(rs, window)
        topsis_auc: dict[str, float] = {}
        topsis_f1: dict[str, float] = {}
        for strategy in STRATEGY_ORDER:
            scorer = TopsisScorer(weighting=WEIGHTING_STRATEGIES[strategy])
            mal, ben = topsis_pooled(rs, scorer, window)
            summary = detector_summary(mal, ben)
            topsis_auc[strategy] = summary.auc
            topsis_f1[strategy] = summary.f1
        rows.append(
            ScoringRow(
                experiment=name,
                best_single_metric=best_name,
                best_single_auc=best_auc,
                best_single_f1=best_f1,
                topsis_auc=topsis_auc,
                topsis_f1=topsis_f1,
            )
        )
    return rows


def oat_mean_spearman(
    runs: list[SeedRun], window: int, levels: tuple[float, ...] = DEFAULT_LEVELS
) -> dict[str, np.ndarray]:
    """Per-strategy mean Spearman rank stability under OAT weight perturbation.

    Returns a strategy -> ``(m, len(levels))`` array, the per-round OAT grids
    averaged over every round and seed of ``runs``. For the data-derived
    strategies the perturbed baseline is that round's own computed weights.
    """
    benefit = benefit_mask(TRUST_METRICS)
    out: dict[str, np.ndarray] = {}
    for strategy in STRATEGY_ORDER:
        weighting = WEIGHTING_STRATEGIES[strategy]
        grids: list[np.ndarray] = []
        for run in runs:
            for rows in _grouped_windowed(run, window).values():
                usable = _complete_rows(rows)
                if len(usable) < 2:
                    continue
                matrix = build_batch(usable, TRUST_METRICS).matrix
                weights = weighting(matrix)
                grids.append(
                    oat_sensitivity(matrix, weights, benefit, levels).spearman
                )
        if grids:
            with np.errstate(invalid="ignore"):
                out[strategy] = np.nanmean(np.stack(grids), axis=0)
        else:
            out[strategy] = np.full((len(TRUST_METRICS), len(levels)), np.nan)
    return out


def _fmt(value: float) -> str:
    return f"{value:.3f}" if not np.isnan(value) else "n/a"


def _emphasise_best(values: dict[str, float], cells: dict[str, str]) -> None:
    finite = {k: v for k, v in values.items() if not np.isnan(v)}
    if not finite:
        return
    best = max(finite, key=lambda k: finite[k])
    cells[best] = f"**{cells[best]}**"


def build_report(
    runs: dict[str, list[SeedRun]],
    baseline_name: str,
    window: int,
    oat_experiment: str,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> str:
    rows = build_scoring_rows(runs, baseline_name, window)
    lines: list[str] = []
    lines.append("# TOPSIS scoring evaluation\n")
    pooled_desc = (
        f"last {window} rounds" if window > 0 else "all rounds"
    )
    lines.append(
        f"Per-round TOPSIS closeness pooled over the {pooled_desc} and across "
        "seeds, then scored as a malicious-vs-benign detector. The "
        "single-metric column is the most separating individual metric (AUC "
        "distance from 0.5, ties broken by F1) over the same client rows. Best "
        "TOPSIS strategy per attack is bold.\n"
    )

    # --- AUC table --------------------------------------------------------
    lines.append("## Detector AUC\n")
    lines.append(
        "| experiment | best single (AUC) | " + " | ".join(STRATEGY_ORDER) + " |"
    )
    lines.append("|" + "---|" * (len(STRATEGY_ORDER) + 2))
    for row in rows:
        cells = {s: _fmt(row.topsis_auc[s]) for s in STRATEGY_ORDER}
        _emphasise_best(row.topsis_auc, cells)
        single = f"{row.best_single_metric} ({_fmt(row.best_single_auc)})"
        lines.append(
            f"| `{row.experiment}` | {single} | "
            + " | ".join(cells[s] for s in STRATEGY_ORDER)
            + " |"
        )
    lines.append("")

    # --- F1 table ---------------------------------------------------------
    lines.append("## Detector F1 (threshold-optimal)\n")
    lines.append(
        "| experiment | best single (F1) | " + " | ".join(STRATEGY_ORDER) + " |"
    )
    lines.append("|" + "---|" * (len(STRATEGY_ORDER) + 2))
    for row in rows:
        cells = {s: _fmt(row.topsis_f1[s]) for s in STRATEGY_ORDER}
        _emphasise_best(row.topsis_f1, cells)
        single = f"{row.best_single_metric} ({_fmt(row.best_single_f1)})"
        lines.append(
            f"| `{row.experiment}` | {single} | "
            + " | ".join(cells[s] for s in STRATEGY_ORDER)
            + " |"
        )
    lines.append("")
    lines.append(
        "> AUC is folded to the canonical orientation (>= 0.5), so 0.5 means "
        "no separation regardless of sign. A composite that only matches the "
        "best single metric indicates the metrics are redundant for that "
        "attack; a composite that beats it indicates complementary coverage.\n"
    )

    # --- OAT weight sensitivity ------------------------------------------
    lines.append("## Weight sensitivity (OAT)\n")
    if oat_experiment in runs:
        oat_runs = [r for r in runs[oat_experiment] if r.has_malicious]
        grids = oat_mean_spearman(oat_runs, window, levels)
        top_level = levels[-1]
        lines.append(
            f"Mean Spearman rank correlation between the unperturbed TOPSIS "
            f"ranking and the ranking after perturbing each metric weight by "
            f"+/-{int(top_level * 100)}%, averaged over rounds/seeds of "
            f"`{oat_experiment}`. Lower = the ranking is more sensitive to that "
            f"metric's weight; the most sensitive metric per strategy is bold.\n"
        )
        lines.append("| strategy | " + " | ".join(TRUST_METRICS) + " |")
        lines.append("|" + "---|" * (len(TRUST_METRICS) + 1))
        for strategy in STRATEGY_ORDER:
            col = grids[strategy][:, -1]
            oat_cells = [_fmt(v) for v in col]
            finite_idx = [i for i, v in enumerate(col) if not np.isnan(v)]
            if finite_idx:
                most = min(finite_idx, key=lambda i: col[i])
                oat_cells[most] = f"**{oat_cells[most]}**"
            lines.append(f"| {strategy} | " + " | ".join(oat_cells) + " |")
        lines.append("")
    else:
        lines.append(f"(reference experiment `{oat_experiment}` not found)\n")

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
        "--oat-experiment",
        default="backdoor_cifar10_f10",
        help="reference experiment for the weight-sensitivity section",
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

    report = build_report(
        runs, args.baseline, args.convergence_window, args.oat_experiment
    )
    if args.out is not None:
        args.out.write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
