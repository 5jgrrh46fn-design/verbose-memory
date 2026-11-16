"""End-to-end defense evaluation: defended vs. undefended runs.

Pairs each defended run (output label ``{attack}__{Mechanism}_{weighting}``)
with its undefended counterpart (``{attack}``) and reports, over the converged
window and across seeds:

* **Backdoor suppression** — converged ASR undefended vs. defended, with a
  paired Wilcoxon test over seeds.
* **Clean accuracy** — converged test accuracy undefended vs. defended (a good
  defense suppresses the attack without sacrificing accuracy).
* **Defense behaviour** — from each run's ``defense_ledger.json``: the
  false-exclusion rate (benign client-rows excluded from aggregation) and the
  mean aggregation weight given to malicious vs. benign clients.

Read-only over ``results/``; ``--out`` is rejected inside the results tree.

Usage::

    uv run python -m topfed.analysis.defense --out tables/defense-eval.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from topfed.analysis.aggregate import SeedRun, discover_runs
from topfed.analysis.stats import bca_ci, wilcoxon_vs_baseline


def parse_label(name: str) -> tuple[str, str] | None:
    """Split a defended run label into ``(base_attack, mechanism_tag)``."""
    if "__" not in name:
        return None
    base, tag = name.split("__", 1)
    return base, tag


def _converged(run: SeedRun, key: str, window: int) -> float | None:
    series = run.tail_series(key, window)
    return float(np.mean(series)) if series.size else None


def _ledger_path(results_root: Path, run: SeedRun) -> Path:
    return (
        results_root
        / run.experiment
        / run.config_hash
        / f"seed_{run.seed}"
        / "defense_ledger.json"
    )


@dataclass(frozen=True)
class LedgerStats:
    false_exclusion: float  # fraction of benign rows excluded (weight == 0)
    mean_weight_malicious: float
    mean_weight_benign: float


def _ledger_stats(results_root: Path, runs: list[SeedRun]) -> LedgerStats | None:
    mal_w: list[float] = []
    ben_w: list[float] = []
    ben_excluded = 0
    ben_total = 0
    found = False
    for run in runs:
        path = _ledger_path(results_root, run)
        if not path.exists():
            continue
        found = True
        for row in json.loads(path.read_text()):
            if row["is_malicious"]:
                mal_w.append(float(row["weight"]))
            else:
                ben_w.append(float(row["weight"]))
                ben_total += 1
                if not row["included"]:
                    ben_excluded += 1
    if not found:
        return None
    return LedgerStats(
        false_exclusion=ben_excluded / ben_total if ben_total else float("nan"),
        mean_weight_malicious=float(np.mean(mal_w)) if mal_w else float("nan"),
        mean_weight_benign=float(np.mean(ben_w)) if ben_w else float("nan"),
    )


@dataclass(frozen=True)
class DefenseRow:
    label: str
    base: str
    seeds: list[int]
    asr_undef: float
    asr_def: float
    asr_p: float
    asr_effect: float
    acc_undef: float
    acc_def: float
    ledger: LedgerStats | None


def build_rows(
    runs: dict[str, list[SeedRun]], results_root: Path, window: int
) -> list[DefenseRow]:
    rows: list[DefenseRow] = []
    for name in sorted(runs):
        parsed = parse_label(name)
        if parsed is None:
            continue
        base, _tag = parsed
        if base not in runs:
            continue
        defended = {r.seed: r for r in runs[name]}
        undef = {r.seed: r for r in runs[base]}
        seeds = sorted(set(defended) & set(undef))
        if not seeds:
            continue

        has_asr = all("asr_accuracy" in defended[s].final for s in seeds) and all(
            "asr_accuracy" in undef[s].final for s in seeds
        )
        if has_asr:
            asr_def = [_converged(defended[s], "asr_accuracy", window) for s in seeds]
            asr_und = [_converged(undef[s], "asr_accuracy", window) for s in seeds]
            asr_def_v = [v for v in asr_def if v is not None]
            asr_und_v = [v for v in asr_und if v is not None]
            test = wilcoxon_vs_baseline(asr_def_v, asr_und_v)
            asr_undef_mean = float(np.mean(asr_und_v))
            asr_def_mean = float(np.mean(asr_def_v))
            asr_p = test.p_value
            asr_effect = test.rank_biserial
        else:
            asr_undef_mean = asr_def_mean = float("nan")
            asr_p = asr_effect = float("nan")

        acc_und = [_converged(undef[s], "test_accuracy", window) for s in seeds]
        acc_def = [_converged(defended[s], "test_accuracy", window) for s in seeds]
        rows.append(
            DefenseRow(
                label=name,
                base=base,
                seeds=seeds,
                asr_undef=asr_undef_mean,
                asr_def=asr_def_mean,
                asr_p=asr_p,
                asr_effect=asr_effect,
                acc_undef=float(np.mean([v for v in acc_und if v is not None])),
                acc_def=float(np.mean([v for v in acc_def if v is not None])),
                ledger=_ledger_stats(results_root, runs[name]),
            )
        )
    return rows


def _fmt(value: float) -> str:
    return f"{value:.3f}" if not np.isnan(value) else "n/a"


def build_report(
    runs: dict[str, list[SeedRun]], results_root: Path, window: int
) -> str:
    rows = build_rows(runs, results_root, window)
    lines: list[str] = []
    lines.append("# Defense evaluation (defended vs. undefended)\n")
    lines.append(
        f"Converged over the last {window} rounds, paired across seeds. ASR "
        "columns apply to targeted (backdoor) attacks; for untargeted attacks "
        "read the accuracy columns (a defense should recover clean accuracy). "
        "`weight mal/ben` is the mean aggregation weight given to malicious vs. "
        "benign clients online.\n"
    )
    lines.append(
        "| defended run | seeds | ASR undef -> def | Wilcoxon p (r) | "
        "acc undef -> def | false-excl | weight mal/ben |"
    )
    lines.append("|" + "---|" * 7)
    for row in rows:
        asr = f"{_fmt(row.asr_undef)} -> {_fmt(row.asr_def)}"
        wil = f"{_fmt(row.asr_p)} ({_fmt(row.asr_effect)})"
        acc = f"{_fmt(row.acc_undef)} -> {_fmt(row.acc_def)}"
        if row.ledger is not None:
            fexcl = _fmt(row.ledger.false_exclusion)
            wts = (
                f"{_fmt(row.ledger.mean_weight_malicious)}/"
                f"{_fmt(row.ledger.mean_weight_benign)}"
            )
        else:
            fexcl = "n/a"
            wts = "n/a"
        lines.append(
            f"| `{row.label}` | {len(row.seeds)} | {asr} | {wil} | {acc} | "
            f"{fexcl} | {wts} |"
        )
    lines.append("")
    lines.append(
        "> With S=5 seeds the two-sided Wilcoxon p-value floors at 0.0625, so "
        "read the effect size (rank-biserial r) alongside it. A successful "
        "targeted defense lowers ASR while holding accuracy; a successful "
        "untargeted defense raises accuracy toward the benign baseline.\n"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--convergence-window", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results_root: Path = args.results_root.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    if args.out is not None:
        out = args.out.resolve()
        if results_root == out or results_root in out.parents:
            raise SystemExit(f"refusing to write inside the results tree: {out}")

    runs = discover_runs(results_root)
    rows = build_rows(runs, results_root, args.convergence_window)
    if not rows:
        raise SystemExit(
            "no defended runs found (expected labels like "
            "'<attack>__ScoreWeight_critic' under results/)"
        )

    report = build_report(runs, results_root, args.convergence_window)
    if args.out is not None:
        args.out.write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
