"""Render presentation-ready figures from the seed-replicated ledgers.

Read-only consumer of ``results/``: it loads the same per-run ledgers as
the Markdown aggregator and emits PNGs to a destination *outside* the
results tree (``--out-dir`` is rejected if it points inside it).

Four figures:

* ``potency`` -- per-condition clean-accuracy loss versus the benign
  baseline, mean over seeds with a standard-deviation whisker.
* ``visibility`` -- behavioural-signal-by-attack heatmap of the
  malicious-vs-benign separability AUC.
* ``convergence`` -- per-round clean test accuracy, benign with a
  spread band against each attack condition.
* ``asr`` -- backdoor success rate, final-round point estimate against
  the converged bootstrap interval, with the acceptance band shaded.

Usage::

    uv run python -m topfed.analysis.figures
    uv run python -m topfed.analysis.figures --out-dir notes \\
        --baseline benign_cifar10
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

from topfed.analysis.aggregate import (
    BEHAVIOURAL_SIGNALS,
    SeedRun,
    discover_runs,
)
from topfed.analysis.stats import bca_ci

# Stable family colour / fraction line-style mapping so every figure
# tells the same visual story.
_FAMILY_COLOUR = {
    "backdoor": "#c0392b",
    "gradient_scaling": "#2e86c1",
    "sign_flip": "#27ae60",
    "dirty_label": "#8e44ad",
    "clean_label": "#e67e22",
}
_FRACTION_STYLE = {"f10": "-", "f20": "--"}


def _family(name: str) -> str:
    for fam in _FAMILY_COLOUR:
        if name.startswith(fam):
            return fam
    return name


def _fraction(name: str) -> str:
    if name.endswith("_f20"):
        return "f20"
    if name.endswith("_f10"):
        return "f10"
    return ""


def _pretty(name: str) -> str:
    return name.replace("_cifar10", "").replace("_", " ")


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average; per-round accuracy is jumpy under attack.

    The signal is edge-padded before convolution so the first and last
    rounds reflect their true values rather than drooping toward zero.
    """
    if window <= 1 or y.size < window:
        return y
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _colour(name: str) -> str:
    return _FAMILY_COLOUR.get(_family(name), "#7f8c8d")


def fig_potency(
    runs: dict[str, list[SeedRun]], baseline: str
) -> Figure:
    """Bar chart: clean accuracy removed by each attack vs the baseline."""
    base = {r.seed: r.final["test_accuracy"] for r in runs[baseline]}
    rows: list[tuple[str, float, float]] = []
    for name in runs:
        if name == baseline:
            continue
        seeds = sorted(set(r.seed for r in runs[name]) & set(base))
        att = {r.seed: r.final["test_accuracy"] for r in runs[name]}
        drop = np.array([base[s] - att[s] for s in seeds], dtype=float)
        sd = float(drop.std(ddof=1)) if drop.size > 1 else 0.0
        rows.append((name, float(drop.mean()), sd))
    rows.sort(key=lambda r: r[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [_pretty(n) for n, _, _ in rows]
    means = [m for _, m, _ in rows]
    sds = [s for _, _, s in rows]
    colours = [_colour(n) for n, _, _ in rows]
    bars = ax.bar(labels, means, yerr=sds, capsize=4, color=colours)
    for b, m, s in zip(bars, means, sds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            m + s + 0.015,
            f"{m:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("clean accuracy removed vs benign")
    ax.set_title("Attack potency (mean ± sd over 5 seeds)")
    ax.set_ylim(0, max(m + s for m, s in zip(means, sds)) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def fig_visibility(
    runs: dict[str, list[SeedRun]], baseline: str
) -> Figure:
    """Heatmap: per-signal malicious-vs-benign AUC for each attack."""
    attacks = sorted(
        n
        for n in runs
        if n != baseline and any(r.has_malicious for r in runs[n])
    )
    grid = np.full((len(attacks), len(BEHAVIOURAL_SIGNALS)), np.nan)
    for i, name in enumerate(attacks):
        rs = [r for r in runs[name] if r.has_malicious]
        for j, sig in enumerate(BEHAVIOURAL_SIGNALS):
            aucs = [r.signal_auc(sig) for r in rs]
            aucs = [a for a in aucs if not np.isnan(a)]
            if aucs:
                grid[i, j] = float(np.mean(aucs))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(BEHAVIOURAL_SIGNALS)))
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in BEHAVIOURAL_SIGNALS], fontsize=8
    )
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([_pretty(a) for a in attacks])
    for i in range(len(attacks)):
        for j in range(len(BEHAVIOURAL_SIGNALS)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(v - 0.5) > 0.32 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("AUC (0.5 = invisible)")
    ax.set_title("Attack visibility per behavioural signal")
    fig.tight_layout()
    return fig


def fig_convergence(
    runs: dict[str, list[SeedRun]], baseline: str
) -> Figure:
    """Per-round clean test accuracy: benign band vs each attack."""
    fig, ax = plt.subplots(figsize=(8.5, 5))

    def stacked(name: str) -> np.ndarray:
        series = [
            r.round_series("test_accuracy")
            for r in sorted(runs[name], key=lambda r: r.seed)
        ]
        n = min(len(s) for s in series)
        return np.vstack([s[:n] for s in series])

    bmat = stacked(baseline)
    x = np.arange(1, bmat.shape[1] + 1)
    bmean = _smooth(bmat.mean(axis=0))
    bsd = _smooth(bmat.std(axis=0, ddof=1))
    ax.plot(x, bmean, color="black", lw=2.2, label="benign")
    ax.fill_between(
        x, bmean - bsd, bmean + bsd, color="black", alpha=0.15
    )

    for name in sorted(n for n in runs if n != baseline):
        mat = stacked(name)
        xa = np.arange(1, mat.shape[1] + 1)
        ax.plot(
            xa,
            _smooth(mat.mean(axis=0)),
            color=_colour(name),
            linestyle=_FRACTION_STYLE.get(_fraction(name), "-"),
            lw=1.7,
            label=_pretty(name),
        )
    ax.set_xlabel("communication round")
    ax.set_ylabel("clean test accuracy")
    ax.set_title(
        "Training trajectory: benign vs attacks "
        "(5-seed mean, 5-round smoothing)"
    )
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout()
    return fig


def fig_asr(
    runs: dict[str, list[SeedRun]],
    baseline: str,
    convergence_window: int = 50,
) -> Figure:
    """Backdoor ASR: final-round point vs converged bootstrap interval."""
    targets = sorted(
        n
        for n in runs
        if n != baseline
        and all("asr_accuracy" in r.final for r in runs[n])
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhspan(0.75, 0.85, color="gold", alpha=0.25, label="target band")

    for k, name in enumerate(targets):
        rs = sorted(runs[name], key=lambda r: r.seed)
        final = np.array([r.final["asr_accuracy"] for r in rs], dtype=float)
        fmean = float(final.mean())
        fsd = float(final.std(ddof=1)) if final.size > 1 else 0.0
        pooled = np.concatenate(
            [r.tail_series("asr_accuracy", convergence_window) for r in rs]
        )
        lo, hi = bca_ci(pooled.tolist())

        xf, xc = k - 0.13, k + 0.13
        ax.errorbar(
            xf, fmean, yerr=fsd, fmt="o", color=_colour(name),
            capsize=5, ms=9, label="final-round (±sd)" if k == 0 else None,
        )
        ax.errorbar(
            xc,
            (lo + hi) / 2,
            yerr=[[(lo + hi) / 2 - lo], [hi - (lo + hi) / 2]],
            fmt="s",
            color=_colour(name),
            mfc="white",
            capsize=5,
            ms=8,
            label="converged 95% BCa" if k == 0 else None,
        )
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([_pretty(t) for t in targets])
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Backdoor success rate vs acceptance band")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--baseline", default="benign_cifar10")
    parser.add_argument("--prefix", default="fig")
    parser.add_argument("--convergence-window", type=int, default=50)
    parser.add_argument(
        "--exclude",
        default="",
        help=(
            "comma-separated condition names to omit from every figure "
            "(e.g. tuning ablations or diagnostics); the baseline is always kept"
        ),
    )
    args = parser.parse_args(argv)

    results_root: Path = args.results_root.resolve()
    out_dir: Path = args.out_dir.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    if results_root == out_dir or results_root in out_dir.parents:
        raise SystemExit(f"refusing to write inside the results tree: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_root)
    if args.baseline not in runs:
        raise SystemExit(
            f"baseline '{args.baseline}' not found; have {sorted(runs)}"
        )

    excluded = {name.strip() for name in args.exclude.split(",") if name.strip()}
    excluded.discard(args.baseline)
    if excluded:
        runs = {name: rs for name, rs in runs.items() if name not in excluded}

    builders = {
        "potency": lambda: fig_potency(runs, args.baseline),
        "visibility": lambda: fig_visibility(runs, args.baseline),
        "convergence": lambda: fig_convergence(runs, args.baseline),
        "asr": lambda: fig_asr(
            runs, args.baseline, args.convergence_window
        ),
    }
    for stem, build in builders.items():
        fig = build()
        dest = out_dir / f"{args.prefix}-{stem}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
