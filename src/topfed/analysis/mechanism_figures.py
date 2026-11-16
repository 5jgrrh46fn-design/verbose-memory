"""Figure: integration-mechanism comparison (suppression vs. collateral).

Two panels over the attacks swept across all five mechanisms:

* backdoor attack-success-rate by mechanism at f10 and f20, against the
  undefended reference — how hard each mechanism suppresses the backdoor.
* false-exclusion rate by mechanism — the collateral cost (benign clients
  dropped from aggregation), which separates the aggressive mechanisms from
  the zero-collateral ones.

Read-only over ``results/`` (``--out-dir`` is rejected inside it).

Usage::

    uv run python -m topfed.analysis.mechanism_figures --out-dir figures
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

# Display order (soft -> aggressive) and short labels.
_ORDER = [
    "ScoreWeight",
    "TemporalReputation",
    "AdaptiveParticipation",
    "HardFilter",
    "Quarantine",
]
_SHORT = {
    "ScoreWeight": "score-wt",
    "TemporalReputation": "temporal",
    "AdaptiveParticipation": "adaptive",
    "HardFilter": "hard-filter",
    "Quarantine": "quarantine",
}


def _mechanism(label: str) -> str:
    return label.split("__", 1)[1].rsplit("_", 1)[0]


def mechanism_figure(runs: dict, results_root: Path, window: int) -> Figure:
    rows = build_rows(runs, results_root, window)
    by_attack: dict[str, dict[str, object]] = {}
    for r in rows:
        by_attack.setdefault(r.base, {})[_mechanism(r.label)] = r

    def present(mechs: dict) -> bool:
        return all(m in mechs for m in _ORDER)

    f10 = by_attack.get("backdoor_cifar10_f10", {})
    f20 = by_attack.get("backdoor_cifar10_f20", {})
    x = np.arange(len(_ORDER))
    w = 0.38

    fig, (ax_asr, ax_fx) = plt.subplots(1, 2, figsize=(13, 5))

    if present(f10) and present(f20):
        asr10 = [f10[m].asr_def for m in _ORDER]  # type: ignore[attr-defined]
        asr20 = [f20[m].asr_def for m in _ORDER]  # type: ignore[attr-defined]
        ax_asr.bar(x - w / 2, asr10, w, label="f10", color="#e59866")
        ax_asr.bar(x + w / 2, asr20, w, label="f20", color="#c0392b")
        ax_asr.axhline(f10[next(iter(f10))].asr_undef, color="#e59866", ls=":", lw=1)  # type: ignore[attr-defined]
        ax_asr.axhline(f20[next(iter(f20))].asr_undef, color="#c0392b", ls=":", lw=1)  # type: ignore[attr-defined]
    ax_asr.set_ylabel("backdoor ASR (converged)")
    ax_asr.set_ylim(0, 1.0)
    ax_asr.set_title("Backdoor suppression by mechanism\n(dotted = undefended)")
    ax_asr.set_xticks(x)
    ax_asr.set_xticklabels([_SHORT[m] for m in _ORDER], rotation=20, ha="right")
    ax_asr.legend()

    # False-exclusion (collateral) — average f10/f20 where available.
    fx = []
    for m in _ORDER:
        vals = [
            d[m].ledger.false_exclusion  # type: ignore[attr-defined]
            for d in (f10, f20)
            if m in d and d[m].ledger is not None  # type: ignore[attr-defined]
        ]
        fx.append(float(np.mean(vals)) if vals else 0.0)
    ax_fx.bar(x, fx, color="#2e86c1")
    ax_fx.set_ylabel("false-exclusion rate (benign dropped)")
    ax_fx.set_title("Collateral cost by mechanism")
    ax_fx.set_xticks(x)
    ax_fx.set_xticklabels([_SHORT[m] for m in _ORDER], rotation=20, ha="right")

    fig.suptitle("Integration-mechanism comparison — TOPSIS-CRITIC (5 seeds)")
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
    fig = mechanism_figure(runs, results_root, args.convergence_window)
    path = out_dir / "mechanism-comparison.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
