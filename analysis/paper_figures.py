"""Build the paper figures from results/ (local campaign) as vector PDFs.

Outputs (into --out-dir, default figures):
  backfire.pdf            Figure 1: adaptive backdoor ASR per trust-score
                          integration vs undefended, f=10% and 20%, n=10
  sweep-cifar10-asr.pdf   Figure 2: last-50-round ASR vs compromised fraction,
                          standard and norm-bounded backdoor, every rule
  sweep-cifar10-acc.pdf   appendix: clean accuracy vs fraction, same cells

Usage (from views/main, see analysis/reproduce.sh):
  /path/to/.venv/bin/python analysis/paper_figures.py

Cells are keyed by the resolved config in each run's manifest (fraction,
attack kind and norm budget, defense block), restricted to the reference
setting (N=100, K=10, 100 validation samples per class, 6x6 bottom-right
trigger, attack every round). A seed that exists under several run labels is
taken from the label with the most completed seeds, as in topfed.analysis.e2.
Converged ASR and accuracy are the mean of the last 50 of 200 rounds; bands
and bars are +-1 sd over seeds. Colours are Okabe-Ito; Middleweight-Peer is
vermillion in every figure and the undefended line is black dashed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAST = 50
ROUNDS = 200
BUDGET = 0.6  # the sweep's norm budget and the best response against MW-Peer
FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

# Okabe-Ito
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7",
)
GREY = "#7F7F7F"

STYLE = {  # name -> (colour, linestyle, marker)
    "Undefended": (BLACK, "--", "x"),
    "Median": (GREY, "-", "s"),
    "Trimmed mean": (GREY, ":", "^"),
    "Clipped mean": (SKY, "--", "d"),
    "Krum": (BLACK, "-.", "v"),
    "FoolsGold": (GREEN, ":", "D"),
    "FLTrust": (BLUE, "-", "o"),
    "FLAME": (ORANGE, "-", "P"),
    "MESAS": (BLUE, ":", "X"),
    "BackdoorIndicator": (PURPLE, ":", "*"),
    "MW-Peer": (VERMILLION, "-", "o"),
}
SWEEP_ORDER = list(STYLE)
MECHANISMS = ["ScoreWeight", "TemporalReputation", "HardFilter", "Quarantine",
              "AdaptiveParticipation"]


def _name(cfg: dict) -> str | None:
    """Map a resolved config's defense/baseline block to a paper name."""
    base = cfg.get("baseline")
    dfn = cfg.get("defense")
    if base is None and dfn is None:
        return "Undefended"
    if base is not None:
        b = base["name"]
        table = {
            "coordinate_median": "Median", "krum": "Krum", "foolsgold": "FoolsGold",
            "fltrust": "FLTrust", "flame": "FLAME", "backdoor_indicator": "BackdoorIndicator",
            "fltrust_median": "MW-Root",
            "clipped_mean": "Clipped mean",
        }
        if b == "trimmed_mean":
            return "Trimmed mean" if abs(base.get("trim_ratio", 0.2) - 0.2) < 1e-9 else None
        if b == "mesas":
            return "MESAS" if base.get("center", "median") == "median" else None
        if b == "backdoor_indicator":
            ok = base.get("retrain_epochs") == 10 and base.get("start_round", 1) == 1
            return "BackdoorIndicator" if ok else None
        return table.get(b)
    scorer = dfn.get("scorer") or {}
    integ = dfn.get("integrator") or {}
    target = str(integ.get("_target_", "")).rsplit(".", 1)[-1]
    if scorer != {"weighting": "critic"}:
        return None  # rooted / agreement scorers are not in these figures
    if target == "CappedScoreWeight" and dfn.get("base") == "weighted_median":
        ok = abs(float(integ.get("cap", 1.0)) - 1.0) < 1e-9 and abs(float(integ.get("alpha", 1.0)) - 1.0) < 1e-9
        return "MW-Peer" if ok else None
    if target in MECHANISMS and dfn.get("base") in (None, "mean"):
        return target
    return None


def _attack(cfg: dict) -> tuple[str, float | None] | None:
    atk = cfg["attack"]
    if atk.get("kind") != "backdoor":
        return None
    p = atk.get("poisoner") or {}
    if p.get("trigger_size") != 6 or p.get("trigger_position", "bottom_right") != "bottom_right":
        return None
    if abs(float(p.get("trigger_amplitude", 1.0)) - 1.0) > 1e-9 or p.get("batch_mode") != "augment":
        return None
    if str((atk.get("schedule") or {}).get("_target_", "")).rsplit(".", 1)[-1] != "AlwaysAttack":
        return None
    um = atk.get("update_manipulator") or {}
    t = str(um.get("_target_", "")).rsplit(".", 1)[-1]
    if t == "ModelReplacementManipulator":
        return ("standard", None) if um.get("gamma") is None else None
    if t == "NormBoundedManipulator":
        la = str((atk.get("loss_augmenter") or {}).get("_target_", "")).rsplit(".", 1)[-1]
        if la != "ConstrainAndScaleAugmenter":
            return None
        return ("adaptive", round(float(um["target_norm"]), 4))
    return None


def _reference(cfg: dict) -> bool:
    fed = cfg["federation"]
    return (
        cfg["dataset"]["name"] == "cifar10"
        and cfg["dataset"].get("val_per_class") == 100
        and fed["num_clients"] == 100 and fed["clients_per_round"] == 10
        and fed["num_rounds"] == ROUNDS
        and fed.get("force_malicious_per_round") is None
        and int(cfg["dataset"].get("pad_border", 0) or 0) == 0
        and cfg["partition"].get("kind") == "dirichlet" and abs(cfg["partition"].get("alpha", 0) - 0.5) < 1e-9
    )


def load(results: Path) -> dict[tuple, dict[int, tuple[float, float]]]:
    """key (attack, budget, fraction, name) -> {seed: (asr, acc)}."""
    cands: dict[tuple, list[Path]] = defaultdict(list)
    completed: dict[Path, int] = defaultdict(int)
    for mp in sorted(results.glob("*/*/seed_*/manifest.json")):
        run = mp.parent
        if not (run / "metrics.json").exists():
            continue
        man = json.loads(mp.read_text())
        cfg = man["config"]
        if not _reference(cfg):
            continue
        atk = _attack(cfg)
        name = _name(cfg)
        if atk is None or name is None:
            continue
        key = (atk[0], atk[1], round(float(cfg["attack"]["assignment"]["fraction"]), 4), name)
        completed[run.parent] += 1
        cands[(key, int(man["seed"]))].append(run)
    cells: dict[tuple, dict[int, tuple[float, float]]] = defaultdict(dict)
    for (key, seed), runs in cands.items():
        run = sorted(runs, key=lambda d: (-completed[d.parent], str(d)))[0]
        metrics = json.loads((run / "metrics.json").read_text())
        if len(metrics) != ROUNDS:
            continue
        asr = float(np.mean([m["asr_accuracy"] for m in metrics[-LAST:]]))
        acc = float(np.mean([m["test_accuracy"] for m in metrics[-LAST:]]))
        cells[key][seed] = (asr, acc)
    return cells


def stats(cell: dict[int, tuple[float, float]], i: int) -> tuple[float, float, int]:
    v = np.array([x[i] for x in cell.values()]) * 100.0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def _rc() -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.compression": 9,
    })


def fig_backfire(cells, out: Path) -> None:
    rows = MECHANISMS + ["MW-Peer"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    width = 0.38
    x = np.arange(len(rows))
    for j, (f, colour) in enumerate(((0.10, BLUE), (0.20, ORANGE))):
        means, sds = [], []
        for r in rows:
            m, s, n = stats(cells[("adaptive", BUDGET, f, r)], 0)
            means.append(m); sds.append(s)
            print(f"backfire f={f:.2f} {r:22s} {m:5.1f} +- {s:4.1f} (n={n})")
        um, us, un = stats(cells[("adaptive", BUDGET, f, "Undefended")], 0)
        print(f"backfire f={f:.2f} {'Undefended':22s} {um:5.1f} +- {us:4.1f} (n={un})")
        offs = (j - 0.5) * width
        bars = ax.bar(x + offs, means, width, yerr=sds, capsize=2, color=colour,
                      edgecolor="none", error_kw={"lw": 0.8}, label=f"defended, $f={int(f*100)}\\%$")
        bars[-1].set_color(VERMILLION)
        bars[-1].set_alpha(1.0 if j == 1 else 0.6)
        ax.axhline(um, color=colour, ls="--", lw=1.0)
        ax.axhspan(um - us, um + us, color=colour, alpha=0.12, lw=0)
        ax.text(len(rows) - 0.55, um + 1.5, f"undefended, $f={int(f*100)}\\%$", color=colour,
                ha="right", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Score-\nWeight", "Temporal\nReputation", "Hard\nFilter", "Quarantine",
                        "Adaptive\nParticipation", "Middle-\nweight"], fontsize=6.5)
    ax.set_ylabel("last-50-round ASR (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "backfire.pdf")
    plt.close(fig)


def _sweep(cells, attack: str, budget, metric: int, ax, ylabel: str, top_axis: bool = True) -> None:
    for name in SWEEP_ORDER:
        xs, ms, ss = [], [], []
        for f in FRACTIONS:
            cell = cells.get((attack, budget, f, name))
            if not cell:
                continue
            m, s, n = stats(cell, metric)
            xs.append(f * 100); ms.append(m); ss.append(s)
            if metric == 0:
                print(f"sweep {attack:8s} f={f:.2f} {name:18s} {m:5.1f} +- {s:4.1f} (n={n})")
        if not xs:
            continue
        colour, ls, marker = STYLE[name]
        lw = 1.8 if name == "MW-Peer" else 1.0
        ax.plot(xs, ms, color=colour, ls=ls, marker=marker, ms=3.5 if name != "MW-Peer" else 4.5,
                lw=lw, label=name, zorder=3 if name == "MW-Peer" else 2)
        ax.fill_between(xs, np.array(ms) - np.array(ss), np.array(ms) + np.array(ss),
                        color=colour, alpha=0.10, lw=0)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 40])
    ax.set_xlabel("compromised clients (%)")
    ax.set_ylabel(ylabel)
    if not top_axis:
        return
    sec = ax.secondary_xaxis("top", functions=(lambda f: f / 10.0, lambda m: m * 10.0))
    sec.set_xlabel("expected attackers per round")
    sec.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 4])
    sec.tick_params(labelsize=7)
    for s in ("top",):
        sec.spines[s].set_visible(True)


def fig_sweep(cells, out: Path, metric: int, fname: str, ylabel: str, stacked: bool = False) -> None:
    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.1), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=True)
    _sweep(cells, "standard", None, metric, axes[0], ylabel)
    _sweep(cells, "adaptive", BUDGET, metric, axes[1], ylabel if stacked else "", top_axis=not stacked)
    axes[0].set_title("standard backdoor", loc="left")
    axes[1].set_title("norm-bounded adaptive backdoor", loc="left")
    if metric == 0:
        for ax in axes:
            ax.set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    if stacked:
        axes[0].set_xlabel("")
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=0.7, fontsize=6.8)
        fig.tight_layout(rect=(0, 0.17, 1, 1), pad=0.3, h_pad=0.8)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
                   bbox_to_anchor=(0.5, -0.01), handlelength=2.4, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.3, w_pad=1.0)
    fig.savefig(out / fname)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    a = ap.parse_args(argv)
    _rc()
    cells = load(a.results)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig_backfire(cells, a.out_dir)
    fig_sweep(cells, a.out_dir, 0, "sweep-cifar10-asr.pdf", "last-50-round ASR (%)", stacked=True)
    fig_sweep(cells, a.out_dir, 1, "sweep-cifar10-acc.pdf", "clean accuracy (%)", stacked=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
