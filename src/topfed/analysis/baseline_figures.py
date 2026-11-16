"""Figure: TOPSIS-defense vs. robust-aggregation baselines (trade-off).

For each backdoor setting, a scatter of clean accuracy (y) against backdoor
ASR (x): the ideal corner is top-left (low ASR, high accuracy). TOPSIS
integration mechanisms and fixed robust-aggregation rules are drawn with
distinct markers, the undefended run as a star, and the benign accuracy as a
reference line — making visible that the robust-aggregation rules each fail one
axis while TOPSIS occupies the good corner.

Read-only over ``results/`` (``--out-dir`` is rejected inside it).

Usage::

    uv run python -m topfed.analysis.baseline_figures --out-dir figures
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

_BASELINES = {"krum", "coordinate_median", "trimmed_mean"}
_SHORT = {
    "ScoreWeight": "score-wt",
    "TemporalReputation": "temporal",
    "AdaptiveParticipation": "adaptive",
    "HardFilter": "hard-filter",
    "Quarantine": "quarantine",
    "krum": "Krum",
    "coordinate_median": "median",
    "trimmed_mean": "trimmed",
}


def _tag(label: str) -> str:
    return label.split("__", 1)[1]


def _name(tag: str) -> str:
    base = tag[:-7] if tag.endswith("_critic") else tag
    return _SHORT.get(base, base)


def _is_baseline(tag: str) -> bool:
    return tag in _BASELINES


def _panel(ax, rows, attack: str, benign: float | None) -> None:
    attack_rows = [r for r in rows if r.base == attack and not np.isnan(r.asr_def)]
    if not attack_rows:
        ax.set_title(f"{attack}\n(no data)")
        return
    undef = attack_rows[0]
    ax.scatter([undef.asr_undef], [undef.acc_undef], marker="*", s=220,
               color="black", zorder=5, label="undefended")
    ax.annotate("undefended", (undef.asr_undef, undef.acc_undef),
                fontsize=8, xytext=(4, 4), textcoords="offset points")
    for r in attack_rows:
        tag = _tag(r.label)
        baseline = _is_baseline(tag)
        ax.scatter([r.asr_def], [r.acc_def],
                   marker="s" if baseline else "o", s=70,
                   color="#2e86c1" if baseline else "#c0392b", zorder=4)
        ax.annotate(_name(tag), (r.asr_def, r.acc_def), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    if benign is not None:
        ax.axhline(benign, color="gray", ls=":", lw=1)
    ax.set_xlabel("backdoor ASR (converged) — lower better")
    ax.set_ylabel("clean test accuracy — higher better")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0.15, 0.82)
    ax.set_title(attack)


def baseline_figure(runs: dict, results_root: Path, window: int) -> Figure:
    rows = build_rows(runs, results_root, window)
    benign = None
    if "benign_cifar10" in runs:
        accs = [
            float(np.mean(r.tail_series("test_accuracy", window)))
            for r in runs["benign_cifar10"]
            if r.tail_series("test_accuracy", window).size
        ]
        benign = float(np.mean(accs)) if accs else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _panel(ax1, rows, "backdoor_cifar10_f10", benign)
    _panel(ax2, rows, "backdoor_cifar10_f20", benign)
    # Shared legend: circle = TOPSIS, square = robust-agg, star = undefended.
    handles = [
        plt.Line2D([], [], marker="o", color="#c0392b", ls="", label="TOPSIS defense"),
        plt.Line2D([], [], marker="s", color="#2e86c1", ls="", label="robust aggregation"),
        plt.Line2D([], [], marker="*", color="black", ls="", label="undefended"),
        plt.Line2D([], [], color="gray", ls=":", label="benign accuracy"),
    ]
    ax1.legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle("Accuracy vs. backdoor suppression — TOPSIS defense vs. robust aggregation (5 seeds)")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
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
    fig = baseline_figure(runs, results_root, args.convergence_window)
    path = out_dir / "baseline-comparison.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
