"""Mechanical PROBE decisions of the follow-up runs.

``d``: block D. Seed 0, undefended, last-50-round ASR per knob for the
direction-aligned and composed attackers on FEMNIST and CIFAR-100. The rule
is: (the smallest probed budget whose
undefended ASR reaches 0.5), read for these knobs as: the *weakest* probed
knob that implants. Direction-aligned: the highest cosine with ASR >= 0.5.
Composed: the smallest budget with ASR >= 0.5 at any probed cosine, then the
highest such cosine. No knob reaching 0.5 leaves the cell unset (D_MAIN skips
it and the status note reports it).

``e``: block E. Seed 0, f10, norm-bounded backdoor: the share of honest
client-rounds each threshold rule (HardFilter, Quarantine) gives zero weight,
per score. A rule that excludes more than half or none at all also runs with
the threshold at the cohort median (E_MEDTHR). "None at all" is read to
include a rule whose threshold no client passes in more than half of the
rounds: the rule then keeps everyone in those rounds (its fallback), so it
does not act as a filter there even if a few rounds exclude someone.

``p3``: knob selection for the held-out protocol. From this box's scan on
seeds 0 to 4 (block P3_SCAN) each rule (Peer, Root, plain median, FLTrust)
gets, per attack and fraction, the knob the paper's rule selects: another
value replaces the attack's default (budget 0.6, cosine 0.9, composed
(0.6, 0.9)) only when it raises the ASR on at least three of the five seeds
with a bias-corrected bootstrap interval of the paired difference above
zero; among such values the one with the highest mean. A rule whose scan is
incomplete gets no knob (P3_EVAL skips it). ``--campaign`` and ``--prefix``
name the box whose scan is read (defaults: this box, ``gpu_b_det`` and
``fin_``; the second box runs with ``--campaign local --prefix fin_k_``), and
``--knobs-out`` also writes the decision to a file that travels with the
repository, so the tables can be built on the other box.

All print a markdown table and, with ``--write``, merge the decision into
``outputs/final_queue/env.json`` (read by sweep_extra.py at collect time).

Usage (repository root; see analysis/reproduce.sh)::

    uv run python analysis/final_queue_probe.py d [--write]
    uv run python analysis/final_queue_probe.py e [--write]
    uv run python analysis/final_queue_probe.py p3 [--write]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path.cwd()
RESULTS = REPO / "results"
ENV = REPO / "outputs" / "final_queue" / "env.json"
THRESHOLD = 0.5
CAMPAIGN = "gpu_b_det"
PREFIX = "fin_"


def _seed0(label: str) -> tuple[float, float, Path] | None:
    """(ASR, clean accuracy, run dir) of the label's finished seed-0 run."""
    for sd in sorted((RESULTS / label).glob("*/seed_0")):
        me = sd / "metrics.json"
        if not me.exists():
            continue
        rows = json.loads(me.read_text())
        if len(rows) < 200:
            continue
        return (
            st.mean(float(r["asr_accuracy"]) for r in rows[-50:]),
            st.mean(float(r["test_accuracy"]) for r in rows[-50:]),
            sd,
        )
    return None


def probe_d() -> dict[str, str]:
    env: dict[str, str] = {}
    print("| dataset | f | attack | knob | ASR seed 0 | clean acc | selected |")
    print("|---|---|---|---|---|---|---|")
    for ds, da_tag, comp_tag, budgets in (
        ("FEMNIST", "fin_femnist_fltrust_backdoor_{f}", "fin_femnist_composed_backdoor_{f}",
         ("0.2", "0.6", "1.0")),
        ("CIFAR100", "fin_fltrust_backdoor_cifar100_{f}", "fin_composed_backdoor_cifar100_{f}",
         ("0.6", "1.0", "3.0")),
    ):
        for f in ("f10", "f20"):
            # Weakest first: cosine descending.
            rows = [(c, _seed0(f"{da_tag.format(f=f)}_c{c}")) for c in ("0.95", "0.9", "0.7", "0.5")]
            pick = next((c for c, r in rows if r and r[0] >= THRESHOLD), None)
            if pick and all(r for _, r in rows):
                env[f"D_DA_{ds}_{f.upper()}"] = pick
            for c, r in rows:
                cells = ("–", "–") if r is None else (f"{100 * r[0]:.1f}", f"{100 * r[1]:.1f}")
                print(f"| {ds} | {f} | direction-aligned | c={c} | {cells[0]} | {cells[1]} | "
                      f"{'**yes**' if c == pick else ''} |")
            # Weakest first: budget ascending, then cosine descending.
            crow = [((g, c), _seed0(f"{comp_tag.format(f=f)}_g{g}_c{c}"))
                    for g in budgets for c in ("0.9", "0.5")]
            cpick = next((k for k, r in crow if r and r[0] >= THRESHOLD), None)
            if cpick and all(r for _, r in crow):
                env[f"D_COMP_{ds}_{f.upper()}"] = f"{cpick[0]},{cpick[1]}"
            for (g, c), r in crow:
                cells = ("–", "–") if r is None else (f"{100 * r[0]:.1f}", f"{100 * r[1]:.1f}")
                print(f"| {ds} | {f} | composed | budget={g}, c={c} | {cells[0]} | {cells[1]} | "
                      f"{'**yes**' if (g, c) == cpick else ''} |")
    return env


def probe_e() -> dict[str, str]:
    pairs: list[str] = []
    tag = "fin_adaptive_backdoor_cifar10_f10"
    integrations = [
        ("topsis_score_weight", "ScoreWeight_{k}"),
        ("topsis_temporal_reputation", "TemporalReputation_{k}"),
        ("topsis_hard_filter", "HardFilter_{k}"),
        ("topsis_quarantine", "Quarantine_{k}"),
        ("topsis_adaptive", "AdaptiveParticipation_{k}"),
        ("topsis_capped_weighted_median", "CappedScoreWeight_{k}_wmedian"),
    ]
    und = _seed0(tag)
    print(f"Undefended seed 0: ASR {100 * und[0]:.1f}, clean acc {100 * und[1]:.1f}." if und
          else "Undefended seed 0: not finished.")
    print()
    print("| score | integration | ASR seed 0 | clean acc | honest excluded % | attacker excluded % "
          "| honest score mean | attacker score mean | attacker top % | rounds nobody >= 0.5 % "
          "| median-threshold variant |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for kind in ("server_cosine", "history_similarity"):
        for name, suffix in integrations:
            r = _seed0(f"{tag}__{suffix.format(k=kind)}")
            if r is None:
                print(f"| {kind} | {name} | – | – | – | – | – | – | – | – | |")
                continue
            ledger = json.loads((r[2] / "defense_ledger.json").read_text())
            hon = [x for x in ledger if not x["is_malicious"]]
            mal = [x for x in ledger if x["is_malicious"]]
            h_x = sum(float(x["weight"]) <= 0.0 for x in hon) / len(hon)
            m_x = sum(float(x["weight"]) <= 0.0 for x in mal) / len(mal) if mal else float("nan")
            by_round: dict[int, list[dict]] = {}
            for x in ledger:
                by_round.setdefault(int(x["round"]), []).append(x)
            att_rounds = [rs for rs in by_round.values() if any(x["is_malicious"] for x in rs)]
            top = sum(
                max(float(x["score"]) for x in rs if x["is_malicious"])
                >= max((float(x["score"]) for x in rs if not x["is_malicious"]), default=-1.0)
                for rs in att_rounds
            ) / len(att_rounds)
            nobody = sum(
                all(float(x["reputation"]) < THRESHOLD for x in rs) for rs in by_round.values()
            ) / len(by_round)
            variant = ""
            if name in ("topsis_hard_filter", "topsis_quarantine"):
                if h_x > 0.5:
                    variant = "**yes** (more than half excluded)"
                elif h_x == 0.0:
                    variant = "**yes** (none excluded)"
                elif nobody > 0.5:
                    variant = "**yes** (no client passes the threshold in most rounds)"
                else:
                    variant = "no"
                if variant != "no":
                    pairs.append(f"{kind}:{name}")
            print(f"| {kind} | {name} | {100 * r[0]:.1f} | {100 * r[1]:.1f} | {100 * h_x:.1f} | "
                  f"{100 * m_x:.1f} | {st.mean(float(x['score']) for x in hon):.3f} | "
                  f"{st.mean(float(x['score']) for x in mal):.3f} | {100 * top:.1f} | "
                  f"{100 * nobody:.1f} | {variant} |")
    return {"E_MEDTHR": " ".join(pairs)}


def _cell_asr(label: str, seeds: tuple[int, ...]) -> dict[int, float]:
    """Last-50 ASR per seed of this campaign's finished runs under ``label``."""
    out: dict[int, float] = {}
    for sd in sorted((RESULTS / label).glob("*/seed_*")):
        seed = int(sd.name.removeprefix("seed_"))
        mf, me = sd / "manifest.json", sd / "metrics.json"
        if seed not in seeds or not (mf.exists() and me.exists()):
            continue
        cfg = json.loads(mf.read_text())["config"]
        if (cfg.get("reproducibility") or {}).get("campaign") != CAMPAIGN:
            continue
        rows = json.loads(me.read_text())
        if len(rows) == int(cfg["federation"]["num_rounds"]):
            out[seed] = st.mean(float(r["asr_accuracy"]) for r in rows[-50:])
    return out


def probe_p3() -> dict[str, str]:
    import numpy as np

    from topfed.analysis.stats import bca_ci

    seeds = (0, 1, 2, 3, 4)
    rules = {"PEER": "CappedScoreWeight_critic_wmedian", "ROOT": "fltrust_median",
             "MEDIAN": "coordinate_median", "FLTRUST": "fltrust"}
    grids = {
        "NB": ("0.6", ["0.5", "0.6", "0.75", "1.0", "1.5", "2.0"],
               lambda f, k: f"{PREFIX}adaptive_backdoor_cifar10_{f}" + ("" if k == "0.6" else f"_g{k}")),
        "DA": ("0.9", ["0.5", "0.7", "0.9", "0.95"],
               lambda f, k: f"{PREFIX}fltrust_backdoor_cifar10_{f}_c{k}"),
        "COMP": ("0.6,0.9", ["0.6,0.5", "0.6,0.9", "1.0,0.5", "1.0,0.9"],
                 lambda f, k: f"{PREFIX}composed_backdoor_cifar10_{f}_g{k.split(',')[0]}_c{k.split(',')[1]}"),
    }
    env: dict[str, str] = {}
    print("| attack | f | rule | mean ASR on seeds 0-4 by knob (n) | selected |")
    print("|---|---|---|---|---|")
    for attack, (default, knobs, tag) in grids.items():
        for f in ("f10", "f20"):
            for rule, suffix in rules.items():
                cells = {k: _cell_asr(f"{tag(f, k)}__{suffix}", seeds) for k in knobs}
                desc = "; ".join(
                    f"{k}: {100 * st.mean(v.values()):.1f} ({len(v)})" if v else f"{k}: – (0)"
                    for k, v in cells.items()
                )
                complete = all(len(v) == len(seeds) for v in cells.values())
                pick = ""
                if complete:
                    pick = default
                    base = cells[default]
                    best = st.mean(base.values())
                    for k, v in cells.items():
                        if k == default:
                            continue
                        d = np.array([v[s_] - base[s_] for s_ in seeds])
                        low, _ = bca_ci(d)
                        mean = st.mean(v.values())
                        if int((d > 0).sum()) >= 3 and low > 0 and mean > best:
                            pick, best = k, mean
                    env[f"P3_{attack}_{rule}_{f.upper()}"] = pick
                print(f"| {attack} | {f} | {rule} | {desc} | {pick or 'scan incomplete'} |")
    return env


def main() -> int:
    global CAMPAIGN, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["d", "e", "p3"])
    ap.add_argument("--write", action="store_true", help="merge the decision into env.json")
    ap.add_argument("--campaign", default=CAMPAIGN, help="campaign of the runs read by p3")
    ap.add_argument("--prefix", default=PREFIX, help="tag prefix of the runs read by p3")
    ap.add_argument("--knobs-out", type=Path, help="also write the decision to this JSON file")
    args = ap.parse_args()
    CAMPAIGN, PREFIX = args.campaign, args.prefix
    env = {"d": probe_d, "e": probe_e, "p3": probe_p3}[args.probe]()
    print()
    print("decision:", json.dumps(env))
    if args.write:
        current = json.loads(ENV.read_text()) if ENV.exists() else {}
        current.update(env)
        tmp = ENV.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=1))
        tmp.replace(ENV)
        print(f"wrote {ENV}")
    if args.knobs_out:
        args.knobs_out.write_text(json.dumps(
            {"campaign": CAMPAIGN, "prefix": PREFIX, "knobs": env}, indent=1) + "\n")
        print(f"wrote {args.knobs_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
