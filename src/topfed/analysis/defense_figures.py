"""Figure: defended vs. undefended outcomes per attack.

Two panels — backdoor attack-success-rate (targeted) and clean test accuracy
(all attacks) — each comparing the undefended run with the TOPSIS score-weight
defense, converged over the last rounds and averaged across seeds. A benign
reference line marks the attainable clean-accuracy ceiling. Read-only over
``results/`` (``--out-dir`` is rejected inside the results tree).

Usage::

    uv run python -m topfed.analysis.defense_figures --out-dir figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from topfed.analysis.aggregate import discover_runs
from topfed.analysis.defense import build_rows

_UNDEF = "#95a5a6"
_DEF = "#c0392b"


def _short(label: str) -> str:
    return label.split("__")[0].replace("_cifar10", "")


def _benign_accuracy(runs: dict, baseline: str, window: int) -> float | None:
    if baseline not in runs:
        return None
    vals = [
        float(np.mean(r.tail_series("test_accuracy", window)))
        for r in runs[baseline]
        if r.tail_series("test_accuracy", window).size
    ]
    return float(np.mean(vals)) if vals else None


def defense_figure(
    runs: dict, results_root: Path, baseline: str, window: int
) -> Figure:
    rows = sorted(build_rows(runs, results_root, window), key=lambda r: r.base)
    benign = _benign_accuracy(runs, baseline, window)

    fig, (ax_asr, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Backdoor ASR -----------------------------------------------------
    asr_rows = [r for r in rows if not np.isnan(r.asr_undef)]
    labels = [_short(r.label) for r in asr_rows]
    x = np.arange(len(labels))
    w = 0.38
    ax_asr.bar(x - w / 2, [r.asr_undef for r in asr_rows], w, label="undefended", color=_UNDEF)
    ax_asr.bar(x + w / 2, [r.asr_def for r in asr_rows], w, label="defended", color=_DEF)
    ax_asr.set_ylabel("backdoor ASR (converged)")
    ax_asr.set_ylim(0, 1.0)
    ax_asr.set_title("Backdoor suppression")
    ax_asr.set_xticks(x)
    ax_asr.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_asr.legend()

    # --- Clean accuracy ---------------------------------------------------
    labels2 = [_short(r.label) for r in rows]
    x2 = np.arange(len(labels2))
    ax_acc.bar(x2 - w / 2, [r.acc_undef for r in rows], w, label="undefended", color=_UNDEF)
    ax_acc.bar(x2 + w / 2, [r.acc_def for r in rows], w, label="defended", color="#27ae60")
    if benign is not None:
        ax_acc.axhline(benign, color="black", linestyle=":", linewidth=1, label=f"benign ({benign:.2f})")
    ax_acc.set_ylabel("clean test accuracy (converged)")
    ax_acc.set_ylim(0, 0.85)
    ax_acc.set_title("Clean-accuracy preservation / recovery")
    ax_acc.set_xticks(x2)
    ax_acc.set_xticklabels(labels2, rotation=30, ha="right", fontsize=9)
    ax_acc.legend(fontsize=8)

    fig.suptitle("TOPSIS score-weight defense vs. no defense (5 seeds)")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--baseline", default="benign_cifar10")
    parser.add_argument("--convergence-window", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)

    results_root: Path = args.results_root.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    out_dir: Path = args.out_dir.resolve()
    if results_root == out_dir or results_root in out_dir.parents:
        raise SystemExit(f"refusing to write inside the results tree: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_root)
    fig = defense_figure(runs, results_root, args.baseline, args.convergence_window)
    path = out_dir / "defense-eval.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
