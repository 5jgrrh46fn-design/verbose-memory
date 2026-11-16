"""Aggregate seed-replicated runs into an attack-potency report.

Reads the per-run ledgers under ``results/`` (``manifest.json``,
``metrics.json``, ``client_metrics.json``) and emits a Markdown report
covering, for every attack experiment relative to the benign baseline:

* **Global accuracy degradation** -- final-round clean test accuracy, the
  per-seed drop versus the benign baseline, a paired Wilcoxon signed-rank
  test (Holm-corrected across attacks) with a rank-biserial effect size,
  and a BCa bootstrap interval over the per-round accuracy trace.
* **Backdoor success rate** -- final-round attack success rate for targeted
  attacks that report one.
* **Attack visibility** -- how separable adversarial client updates are
  from benign ones on each behavioural signal, as a Mann-Whitney AUC.

This module only reads ``results/``; it never writes there. ``--out`` is
rejected if it points inside the results tree.

Usage::

    uv run python -m topfed.analysis.aggregate
    uv run python -m topfed.analysis.aggregate --results-root results \\
        --baseline benign_cifar10 --out tables/potency.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from topfed.analysis.stats import (
    auc_separability,
    bca_ci,
    detector_summary,
    holm_bonferroni,
    wilcoxon_vs_baseline,
)

BEHAVIOURAL_SIGNALS = (
    "m1_cosine_to_mean",
    "m2_norm_ratio",
    "m3_temporal_cv",
    "m4_val_loss_delta",
    "m5_max_layer_deviation",
    "update_norm",
)


@dataclass(frozen=True)
class SeedRun:
    experiment: str
    seed: int
    per_round: list[dict]
    client_rows: list[dict]
    config_hash: str

    @property
    def final(self) -> dict:
        return self.per_round[-1]

    def round_series(self, key: str) -> np.ndarray:
        return np.array(
            [r[key] for r in self.per_round if key in r], dtype=float
        )

    def tail_series(self, key: str, window: int) -> np.ndarray:
        """The last ``window`` per-round values of ``key`` (all if fewer).

        Used so bootstrap intervals describe the *converged* metric rather
        than an average dragged down by the early training ramp.
        """
        series = self.round_series(key)
        return series[-window:] if window > 0 else series

    def signal_auc(self, signal: str) -> float:
        """AUC separating malicious from benign client rows on ``signal``."""
        mal = [
            r[signal]
            for r in self.client_rows
            if r["is_malicious"] and r.get(signal) is not None
        ]
        ben = [
            r[signal]
            for r in self.client_rows
            if not r["is_malicious"] and r.get(signal) is not None
        ]
        return auc_separability(mal, ben)

    @property
    def has_malicious(self) -> bool:
        return any(r["is_malicious"] for r in self.client_rows)


def discover_runs(results_root: Path) -> dict[str, list[SeedRun]]:
    """Group runs by experiment, choosing the config hash with most seeds.

    A warning is printed to stderr if an experiment has more than one
    config hash (a sign of config drift between sweeps); only the
    best-covered hash is used so the report compares like with like.
    """
    by_experiment: dict[str, list[SeedRun]] = {}
    for exp_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        per_hash: dict[str, list[SeedRun]] = defaultdict(list)
        for manifest_path in sorted(exp_dir.glob("*/seed_*/manifest.json")):
            seed_dir = manifest_path.parent
            manifest = json.loads(manifest_path.read_text())
            metrics_path = seed_dir / "metrics.json"
            client_path = seed_dir / "client_metrics.json"
            if not metrics_path.exists():
                continue
            client_rows = (
                json.loads(client_path.read_text())
                if client_path.exists()
                else []
            )
            per_hash[seed_dir.parent.name].append(
                SeedRun(
                    experiment=exp_dir.name,
                    seed=int(manifest["seed"]),
                    per_round=json.loads(metrics_path.read_text()),
                    client_rows=client_rows,
                    config_hash=seed_dir.parent.name,
                )
            )
        if not per_hash:
            continue
        if len(per_hash) > 1:
            print(
                f"warning: {exp_dir.name} has {len(per_hash)} config hashes "
                f"{sorted(per_hash)}; using the best-covered one.",
                file=sys.stderr,
            )
        best = max(per_hash.values(), key=len)
        by_experiment[exp_dir.name] = sorted(best, key=lambda r: r.seed)
    return by_experiment


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    return f"[{lo:.4f}, {hi:.4f}]"


def _mean_sd(xs: list[float]) -> str:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return "n/a"
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return f"{a.mean():.4f} ± {sd:.4f}"


def build_report(
    runs: dict[str, list[SeedRun]],
    baseline_name: str,
    convergence_window: int = 50,
) -> str:
    if baseline_name not in runs:
        raise SystemExit(
            f"baseline experiment '{baseline_name}' not found under results; "
            f"have {sorted(runs)}"
        )

    baseline = {r.seed: r for r in runs[baseline_name]}
    attacks = sorted(name for name in runs if name != baseline_name)

    lines: list[str] = []
    lines.append("# Attack-potency report\n")
    lines.append(
        f"Baseline: `{baseline_name}` "
        f"({len(baseline)} seeds: {sorted(baseline)}).\n"
    )
    lines.append(
        "> Wilcoxon is two-sided paired over seed-matched runs. With "
        "`S` seeds the exact two-sided p-value cannot fall below "
        "`2 / 2**S` (= 0.0625 at S=5), so read the rank-biserial effect "
        "size and the bootstrap interval, not only the p-value.\n"
    )
    lines.append(
        f"> BCa intervals bootstrap the per-round value pooled over the "
        f"last {convergence_window} rounds of every seed, so they describe "
        f"the converged metric, not the training ramp.\n"
    )

    # --- Global accuracy degradation -------------------------------------
    acc_rows: list[tuple[str, dict]] = []
    raw_pvalues: list[float] = []
    for name in attacks:
        seeds = sorted(set(r.seed for r in runs[name]) & set(baseline))
        att = {r.seed: r for r in runs[name]}
        att_acc = [att[s].final["test_accuracy"] for s in seeds]
        base_acc = [baseline[s].final["test_accuracy"] for s in seeds]
        drop = [b - a for a, b in zip(att_acc, base_acc)]
        test = wilcoxon_vs_baseline(att_acc, base_acc)
        pooled = np.concatenate(
            [
                att[s].tail_series("test_accuracy", convergence_window)
                for s in seeds
            ]
        )
        ci = bca_ci(pooled.tolist())
        raw_pvalues.append(test.p_value)
        acc_rows.append(
            (
                name,
                {
                    "seeds": seeds,
                    "final": _mean_sd(att_acc),
                    "drop": float(np.mean(drop)) if drop else float("nan"),
                    "test": test,
                    "ci": ci,
                },
            )
        )

    adj_p, reject = (
        holm_bonferroni(raw_pvalues) if raw_pvalues else ([], [])
    )

    lines.append("## Global accuracy degradation\n")
    lines.append(
        f"| experiment | seeds | final test acc | Δ vs benign | "
        f"Wilcoxon p | Holm p | effect (r) | "
        f"converged acc 95% BCa (last {convergence_window}r) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    base_final = [baseline[s].final["test_accuracy"] for s in sorted(baseline)]
    base_tail = np.concatenate(
        [
            baseline[s].tail_series("test_accuracy", convergence_window)
            for s in sorted(baseline)
        ]
    )
    lines.append(
        f"| `{baseline_name}` | {len(baseline)} | {_mean_sd(base_final)} "
        f"| -- | -- | -- | -- | {_fmt_ci(bca_ci(base_tail.tolist()))} |"
    )
    for (name, info), padj, rej in zip(acc_rows, adj_p, reject):
        t = info["test"]
        star = " *" if rej else ""
        lines.append(
            f"| `{name}` | {len(info['seeds'])} | {info['final']} "
            f"| {info['drop']:+.4f} | {t.p_value:.4f} | {padj:.4f}{star} "
            f"| {t.rank_biserial:+.3f} ({t.effect_label}) "
            f"| {_fmt_ci(info['ci'])} |"
        )
    lines.append(
        "\n`*` = rejected at family-wise alpha=0.05 after Holm correction. "
        "Δ vs benign is the per-seed mean accuracy the attack removes "
        "(higher = more potent).\n"
    )

    # --- Backdoor success rate -------------------------------------------
    asr_attacks = [
        name
        for name in attacks
        if all("asr_accuracy" in r.final for r in runs[name])
    ]
    if asr_attacks:
        lines.append("## Backdoor success rate\n")
        lines.append(
            f"| experiment | seeds | final-round ASR | "
            f"converged ASR 95% BCa (last {convergence_window}r) |"
        )
        lines.append("|---|---|---|---|")
        for name in asr_attacks:
            rs = sorted(runs[name], key=lambda r: r.seed)
            asr = [r.final["asr_accuracy"] for r in rs]
            pooled = np.concatenate(
                [
                    r.tail_series("asr_accuracy", convergence_window)
                    for r in rs
                ]
            )
            lines.append(
                f"| `{name}` | {len(rs)} | {_mean_sd(asr)} "
                f"| {_fmt_ci(bca_ci(pooled.tolist()))} |"
            )
        lines.append("")

    # --- Attack visibility ------------------------------------------------
    vis_attacks = [
        name
        for name in attacks
        if any(r.has_malicious for r in runs[name])
    ]
    if vis_attacks:
        lines.append("## Attack visibility (malicious-vs-benign AUC)\n")
        lines.append(
            "AUC of each behavioural signal as a univariate discriminator, "
            "mean over seeds. 0.5 = invisible; distance from 0.5 = "
            "detectable. The most separating signal is bold.\n"
        )
        header = "| experiment | " + " | ".join(BEHAVIOURAL_SIGNALS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(BEHAVIOURAL_SIGNALS) + 1))
        for name in vis_attacks:
            rs = [r for r in runs[name] if r.has_malicious]
            cells: list[str] = []
            seps: list[float] = []
            for sig in BEHAVIOURAL_SIGNALS:
                aucs = [r.signal_auc(sig) for r in rs]
                aucs = [a for a in aucs if not np.isnan(a)]
                if not aucs:
                    cells.append("n/a")
                    seps.append(-1.0)
                    continue
                m = float(np.mean(aucs))
                cells.append(f"{m:.3f}")
                seps.append(abs(m - 0.5))
            best = int(np.argmax(seps)) if seps else -1
            if best >= 0 and seps[best] >= 0:
                cells[best] = f"**{cells[best]}**"
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")

    # --- Per-metric detection performance ---------------------------------
    if vis_attacks:
        lines.append(
            "## Per-metric detection performance (F1-optimal threshold)\n"
        )
        lines.append(
            "Treating each behavioural signal as a univariate binary "
            "detector over (malicious, benign) client rows pooled across "
            "seeds. The threshold is swept to maximize F1; the direction "
            "(higher score = malicious, or the inverse) is inferred from "
            "the AUC. The best-F1 signal per attack is bold.\n"
        )
        header = (
            "| experiment | signal | AUC | dir | threshold | "
            "precision | recall | F1 |"
        )
        lines.append(header)
        lines.append("|" + "---|" * 8)
        for name in vis_attacks:
            rs = [r for r in runs[name] if r.has_malicious]
            rows: list[tuple[str, float, bool, float, float, float, float]] = []
            for sig in BEHAVIOURAL_SIGNALS:
                pos: list[float] = []
                neg: list[float] = []
                for r in rs:
                    for row in r.client_rows:
                        v = row.get(sig)
                        if v is None:
                            continue
                        (pos if row["is_malicious"] else neg).append(float(v))
                summary = detector_summary(pos, neg)
                rows.append(
                    (
                        sig,
                        summary.auc,
                        summary.higher_is_positive,
                        summary.threshold,
                        summary.precision,
                        summary.recall,
                        summary.f1,
                    )
                )
            best_idx = max(range(len(rows)), key=lambda i: rows[i][6])
            for i, (sig, auc, hi, thr, p, rec, f1) in enumerate(rows):
                fmt_auc = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
                fmt_thr = f"{thr:.3g}" if not np.isnan(thr) else "n/a"
                fmt_p = f"{p:.3f}" if not np.isnan(p) else "n/a"
                fmt_r = f"{rec:.3f}" if not np.isnan(rec) else "n/a"
                fmt_f1 = f"{f1:.3f}"
                direction = ">=" if hi else "<="
                if i == best_idx and f1 > 0:
                    fmt_f1 = f"**{fmt_f1}**"
                lines.append(
                    f"| `{name}` | {sig} | {fmt_auc} | {direction} | "
                    f"{fmt_thr} | {fmt_p} | {fmt_r} | {fmt_f1} |"
                )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--baseline", default="benign_cifar10")
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=50,
        help="rounds from the end of each run used for BCa intervals",
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
            raise SystemExit(
                f"refusing to write inside the results tree: {out}"
            )

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
