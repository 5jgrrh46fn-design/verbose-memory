"""Figure: TOPSIS composite vs. best single metric, per attack.

Renders the detection-quality comparison as a grouped bar chart — for each
attack, the best single behavioural metric's AUC beside the TOPSIS-CRITIC
composite AUC, with the 0.5 chance line marked. Read-only over ``results/``
(``--out-dir`` is rejected inside the results tree). Defended runs (labels
containing ``__``) and the benign baseline are excluded.

Usage::

    uv run python -m topfed.analysis.scoring_figures --out-dir figures
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
from topfed.analysis.scoring import build_scoring_rows


def scoring_auc_figure(runs: dict, baseline: str, window: int) -> Figure:
    # Exclude defended runs and the baseline so only attack conditions remain.
    attack_runs = {
        name: rs for name, rs in runs.items() if "__" not in name and name != baseline
    }
    rows = build_scoring_rows(attack_runs, baseline, window)
    rows.sort(key=lambda r: r.experiment)

    labels = [r.experiment for r in rows]
    single = [r.best_single_auc for r in rows]
    critic = [r.topsis_auc["critic"] for r in rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar(x - width / 2, single, width, label="best single metric", color="#95a5a6")
    ax.bar(x + width / 2, critic, width, label="TOPSIS (CRITIC)", color="#c0392b")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="chance")
    ax.set_ylabel("malicious-vs-benign AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        f"Detection quality: TOPSIS composite vs. best single metric "
        f"(last {window} rounds)"
        if window > 0
        else "Detection quality: TOPSIS composite vs. best single metric"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
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
    fig = scoring_auc_figure(runs, args.baseline, args.convergence_window)
    path = out_dir / "scoring-auc.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
