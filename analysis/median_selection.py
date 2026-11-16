"""Coordinate-level median selection statistics for the diagnostic cells.

Reads every ``median_selection.json`` under ``results/diag_*/*/seed_*/``
(written by runs with ``+diagnostics.median_selection=true``, see
``experiments/sweep_diagnostic.sh``) and prints one Markdown row per cell
(label directory, the hash with most seeds):

* the mean attacker selection fraction (the share of the aggregate's
  coordinates whose value came from a compromised client) over all rounds
  and over the last 50, counting the rounds in which at least one attacker
  was sampled;
* that mean against the attackers' equal share m/K and their renormalised
  weight share W_A (ratios of the round means);
* the fraction of rounds in which the attackers supplied a larger share of
  the coordinates than their weight share;
* the mean L2 norm of the aggregate's displacement from the honest-only
  median and the mean cosine of that displacement to the attackers' mean
  delta (rounds where the cosine is defined);
* the cell's last-50-round ASR from ``metrics.json``, averaged over seeds.

Usage (from views/main, see analysis/reproduce.sh):

    uv run python analysis/median_selection.py [--results-root results]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def _mean(values: list[float]) -> float:
    return st.mean(values) if values else math.nan


def best_hash(label_dir: Path) -> Path | None:
    hashes = [h for h in label_dir.iterdir() if h.is_dir()]
    if not hashes:
        return None
    return max(hashes, key=lambda h: len(list(h.glob("seed_*/median_selection.json"))))


def last50_asr(h: Path) -> float:
    vals = []
    for p in sorted(h.glob("seed_*/metrics.json")):
        rows = json.load(open(p))[-50:]
        a = [r["asr_accuracy"] for r in rows if "asr_accuracy" in r]
        if a:
            vals.append(st.mean(a))
    return _mean(vals)


def cell_stats(h: Path) -> dict[str, float]:
    sel: list[float] = []
    sel_late: list[float] = []
    equal: list[float] = []
    weight: list[float] = []
    disp: list[float] = []
    cos: list[float] = []
    n_over = n_minority = n_joint = n_sel_half = 0
    seeds = 0
    for p in sorted(h.glob("seed_*/median_selection.json")):
        seeds += 1
        rounds = json.load(open(p))["rounds"]
        if not rounds:
            continue
        last = max(r["round"] for r in rounds)
        for r in rounds:
            if r["num_malicious"] == 0:
                continue
            sel.append(r["attacker_selection_fraction"])
            if r["round"] > last - 50:
                sel_late.append(r["attacker_selection_fraction"])
            equal.append(r["equal_share"])
            weight.append(r["weight_share"])
            over = r["attacker_selection_fraction"] > r["weight_share"]
            minority = r["weight_share"] < 0.5
            n_over += over
            n_minority += minority
            n_joint += over and minority
            n_sel_half += r["attacker_selection_fraction"] >= 0.5
            if not math.isnan(r["displacement_norm"]):
                disp.append(r["displacement_norm"])
            if not math.isnan(r["displacement_cosine"]):
                cos.append(r["displacement_cosine"])
    mean_sel = _mean(sel)
    return {
        "seeds": seeds,
        "rounds": len(sel),
        "sel": mean_sel,
        "sel_last50": _mean(sel_late),
        "ratio_equal": mean_sel / _mean(equal) if equal else math.nan,
        "ratio_weight": mean_sel / _mean(weight) if weight and _mean(weight) > 0 else math.nan,
        "frac_over_weight": n_over / len(sel) if sel else math.nan,
        "frac_minority": n_minority / len(sel) if sel else math.nan,
        "frac_joint": n_joint / len(sel) if sel else math.nan,
        "frac_sel_ge_half": n_sel_half / len(sel) if sel else math.nan,
        "disp": _mean(disp),
        "cos": _mean(cos),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--glob", default="diag_*")
    args = ap.parse_args()
    root = Path(args.results_root)
    print(
        "| cell | seeds | rounds with attacker | sel. frac (all) | sel. frac (last 50) | "
        "sel / (m/K) | sel / W_A | rounds sel > W_A | rounds W_A < 1/2 | "
        "rounds W_A < 1/2 and sel > W_A | rounds sel >= 1/2 | mean displacement | "
        "mean cosine | ASR last 50 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda x: "--" if math.isnan(x) else f"{x:.3f}"  # noqa: E731
    for label_dir in sorted(p for p in root.glob(args.glob) if p.is_dir()):
        h = best_hash(label_dir)
        if h is None:
            continue
        s = cell_stats(h)
        if s["seeds"] == 0:
            continue
        print(
            f"| {label_dir.name} | {s['seeds']} | {s['rounds']} | {f(s['sel'])} | "
            f"{f(s['sel_last50'])} | {f(s['ratio_equal'])} | {f(s['ratio_weight'])} | "
            f"{f(s['frac_over_weight'])} | {f(s['frac_minority'])} | {f(s['frac_joint'])} | "
            f"{f(s['frac_sel_ge_half'])} | {f(s['disp'])} | {f(s['cos'])} | "
            f"{f(last50_asr(h))} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
