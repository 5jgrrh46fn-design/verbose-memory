"""Recompute the paper cells that exist only in run directories.

Run from views/main (see analysis/reproduce.sh):

    uv run python analysis/paper_supplementary_cells.py   # after saving this block there

Every number is a mean over the last 50 of 200 rounds per seed (last 10 where
stated), read from results/<label>/<hash>/seed_k/metrics.json. Where a label
has several hash directories the one with the most completed seeds is used
and its hash is printed. Paired tests use topfed.analysis.stats (two-sided
Wilcoxon signed-rank on seed-matched differences, BCa 95% interval, Holm).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("results")
LAST = 50
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures as pf  # noqa: E402  (the Figure 1 to 3 loader)
from topfed.analysis.stats import bca_ci, holm_bonferroni, wilcoxon_vs_baseline  # noqa: E402


def pick_hash(label: str) -> Path:
    dirs = [d for d in (ROOT / label).iterdir() if d.is_dir()]
    done = {d: sum((s / "metrics.json").exists() for s in d.glob("seed_*")) for d in dirs}
    best = max(dirs, key=lambda d: (done[d], str(d)))
    return best


def load(label: str, metric: str = "asr_accuracy", last: int = LAST) -> dict[int, float]:
    out = {}
    for sd in sorted(pick_hash(label).glob("seed_*")):
        mf = sd / "metrics.json"
        if not mf.exists():
            continue
        rounds = json.loads(mf.read_text())
        out[int(sd.name.split("_")[1])] = float(np.mean([r[metric] for r in rounds[-last:]]))
    return out


def msd(d: dict[int, float]) -> str:
    v = np.array(list(d.values())) * 100
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.1f} ± {sd:.1f} ({len(v)})"


def paired(name: str, a: dict, b: dict):
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    t = wilcoxon_vs_baseline(list(x), list(y))
    lo, hi = bca_ci(x - y)
    return name, len(seeds), float((x - y).mean()), lo, hi, t.p_value, t.rank_biserial


def family(title: str, rows: list) -> None:
    res = [paired(*r) for r in rows]
    adj, rej = holm_bonferroni([r[5] for r in res])
    print(f"\n### {title}\n")
    print("| comparison (a − b) | pairs | mean Δ (pts) | BCa 95% | p | Holm p | reject | r |")
    print("|---|---|---|---|---|---|---|---|")
    for (name, n, d, lo, hi, p, r), ap, rj in zip(res, adj, rej):
        print(f"| {name} | {n} | {100*d:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {p:.4f} | {ap:.4f} | {'yes' if rj else 'no'} | {r:+.2f} |")


# ---------------------------------------------------------------- A. Table 1
print("## A. Table 1: integration mechanisms under the norm-bounded backdoor (CIFAR-10, budget 0.6, results/)\n")
print("Cells from the Figure 1 loader (`paper_figures.load`, reference setting, hash with most seeds).\n")
cells = pf.load(ROOT)
mechs = ["ScoreWeight", "TemporalReputation", "HardFilter", "Quarantine", "AdaptiveParticipation"]
print("| rule | f=10% ASR | f=10% ACC | f=20% ASR | f=20% ACC |")
print("|---|---|---|---|---|")
asr = {}
for name in ["Undefended"] + mechs + ["MW-Peer"]:
    row = [name]
    for f in (0.10, 0.20):
        c = cells[("adaptive", 0.6, f, name)]
        asr[(f, name)] = {s: v[0] for s, v in c.items()}
        acc = {s: v[1] for s, v in c.items()}
        row += [msd(asr[(f, name)]), msd(acc)]
    print("| " + " | ".join(row) + " |")
family("Backfire family, Holm over the ten comparisons (paired Wilcoxon, ten seeds)",
       [(f"{m} − Undefended, f={int(f*100)}%", asr[(f, m)], asr[(f, "Undefended")])
        for f in (0.10, 0.20) for m in mechs])
family("MW-Peer against undefended (not part of the Table 1 family)",
       [(f"MW-Peer − Undefended, f={int(f*100)}%", asr[(f, "MW-Peer")], asr[(f, "Undefended")]) for f in (0.10, 0.20)])

# ---------------------------------------------------------------- B. ablations, local campaign
print("\n## B. Cap and median ablations at f=10% under the norm-bounded backdoor (results/, local campaign)\n")
print("Ten-seed cells of the RTX 5070 Ti box.\n")
ab = {
    "capped weighted median (MW-Peer)": "adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian",
    "uncapped trust-weighted median": "adaptive_backdoor_cifar10_f10__ScoreWeight_critic_wmedian",
    "cap on a mean (CappedScoreWeight)": "adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic",
    "undefended": "adaptive_backdoor_cifar10_f10",
    "uncapped weighted median, standard backdoor": "backdoor_cifar10_f10__ScoreWeight_critic_wmedian",
    "capped weighted median, standard backdoor": "backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian",
}
abv = {}
print("| cell | label/hash | ASR | ACC |")
print("|---|---|---|---|")
for k, lab in ab.items():
    abv[k] = load(lab)
    print(f"| {k} | `{lab}/{pick_hash(lab).name}` | {msd(abv[k])} | {msd(load(lab, 'test_accuracy'))} |")
family("Ablation family (Holm k=3)", [
    ("uncapped wmed − capped wmed", abv["uncapped trust-weighted median"], abv["capped weighted median (MW-Peer)"]),
    ("cap on mean − undefended", abv["cap on a mean (CappedScoreWeight)"], abv["undefended"]),
    ("uncapped wmed − undefended", abv["uncapped trust-weighted median"], abv["undefended"]),
])

# ---------------------------------------------------------------- C. HardFilter ledger
print("\n## C. HardFilter ledger: honest client-rounds excluded and attacker share (norm-bounded backdoor, CIFAR-10, results/)\n")
print("| f | hash | seeds | honest rows excluded | attacker rows excluded | attacker weight share in rounds with an attacker | equal share conditioned on m ≥ 1 |")
print("|---|---|---|---|---|---|---|")
for f, lab in ((0.10, "adaptive_backdoor_cifar10_f10__HardFilter_critic"), (0.20, "adaptive_backdoor_cifar10_f20__HardFilter_critic")):
    h = pick_hash(lab)
    hon_ex, mal_ex, shares, eq = [], [], [], []
    for sd in sorted(h.glob("seed_*")):
        lf = sd / "defense_ledger.json"
        if not lf.exists():
            continue
        rows = json.loads(lf.read_text())
        hon = [r for r in rows if not r["is_malicious"]]; mal = [r for r in rows if r["is_malicious"]]
        hon_ex.append(np.mean([not r["included"] for r in hon])); mal_ex.append(np.mean([not r["included"] for r in mal]))
        byr = {}
        for r in rows:
            byr.setdefault(r["round"], []).append(r)
        s, e = [], []
        for rr in byr.values():
            m = sum(r["is_malicious"] for r in rr)
            if m:
                s.append(sum(r["weight"] for r in rr if r["is_malicious"])); e.append(m / len(rr))
        shares.append(np.mean(s)); eq.append(np.mean(e))
    print(f"| {f:.2f} | `{h.name}` | {len(hon_ex)} | {100*np.mean(hon_ex):.1f}% | {100*np.mean(mal_ex):.1f}% | {100*np.mean(shares):.1f}% | {100*np.mean(eq):.1f}% |")

# ---------------------------------------------------------------- D. FEMNIST budget probe
print("\n## D. FEMNIST norm-bounded budget probe (f=10%, undefended, seed 0)\n")
print("| budget | label/hash | last-50 ASR | last-50 ACC | final-round ASR |")
print("|---|---|---|---|---|")
for b, lab in ((0.1, "sp_femnist_adaptive_backdoor_f10_g0.1"), (0.2, "sp_femnist_adaptive_backdoor_f10_g0.2"),
               (0.3, "sp_femnist_adaptive_backdoor_f10_g0.3"), (0.6, "sp_femnist_adaptive_backdoor_f10"),
               (1.0, "sp_femnist_adaptive_backdoor_f10_g1.0"), (1.5, "sp_femnist_adaptive_backdoor_f10_g1.5"),
               (3.0, "sp_femnist_adaptive_backdoor_f10_g3.0")):
    h = pick_hash(lab)
    m = json.loads((h / "seed_0" / "metrics.json").read_text())
    tn = json.loads((h / "seed_0" / "manifest.json").read_text())["config"]["attack"]["update_manipulator"]["target_norm"]
    assert abs(float(tn) - b) < 1e-9, (lab, tn)
    print(f"| {b} | `{lab}/{h.name}` | {100*np.mean([r['asr_accuracy'] for r in m[-50:]]):.1f} | {100*np.mean([r['test_accuracy'] for r in m[-50:]]):.1f} | {100*m[-1]['asr_accuracy']:.1f} |")

# ---------------------------------------------------------------- E. per-seed Root on FEMNIST
print("\n## E. MW-Root (fltrust_median) on FEMNIST f=10%, per-seed last-50 ASR\n")
print("| cell | label/hash | per-seed ASR (seeds 0..9) | mean ± sd | seeds < 20 | seeds > 60 |")
print("|---|---|---|---|---|---|")
for k, lab in (("standard backdoor, uncapped", "e4_femnist_backdoor_f10__fltrust_median"),
               ("standard backdoor, capped", "e4_femnist_backdoor_f10__fltrust_median_capped"),
               ("norm-bounded 0.2, uncapped", "sp_femnist_adaptive_backdoor_f10_g0.2__fltrust_median"),
               ("norm-bounded 0.2, capped", "sp_femnist_adaptive_backdoor_f10_g0.2__fltrust_median_capped")):
    d = load(lab); v = np.array([d[s] for s in sorted(d)]) * 100
    print(f"| {k} | `{lab}/{pick_hash(lab).name}` | {', '.join(f'{x:.1f}' for x in v)} | {msd(d)} | {(v < 20).sum()} | {(v > 60).sum()} |")

# ---------------------------------------------------------------- F. FLAME full participation
print("\n## F. FLAME with all 100 clients per round (CIFAR-10, standard backdoor, f=20%, seed 0)\n")
print("| cell | label/hash | K | last-50 ASR | last-50 ACC |")
print("|---|---|---|---|---|")
for k, lab in (("undefended", "sp_flame_full"), ("FLAME", "sp_flame_full__flame")):
    h = pick_hash(lab); man = json.loads((h / "seed_0" / "manifest.json").read_text())["config"]
    print(f"| {k} | `{lab}/{h.name}` | {man['federation']['clients_per_round']} (f={man['attack']['assignment']['fraction']}) | {100*load(lab)[0]:.1f} | {100*load(lab, 'test_accuracy')[0]:.1f} |")

# ---------------------------------------------------------------- G. BackdoorIndicator probes
print("\n## G. BackdoorIndicator probes (seed 0): injection epochs × start round, CIFAR-10 and FEMNIST\n")
print("Drop shares count ledger rows whose `included`/`admitted` flag is false (weight 0 where neither exists), from the start round on.\n")
print("| label/hash | experiment | epochs | start | last-10 ACC | last-10 ASR | last-50 ACC | last-50 ASR | honest rows dropped | attacker rows dropped |")
print("|---|---|---|---|---|---|---|---|---|---|")
for lab in sorted(p.name for p in ROOT.glob("probe_indicator_*")):
    for h in sorted((ROOT / lab).iterdir()):
        sd = h / "seed_0"
        if not (sd / "metrics.json").exists():
            continue
        cfg = json.loads((sd / "manifest.json").read_text())["config"]
        m = json.loads((sd / "metrics.json").read_text())
        start = cfg["baseline"].get("start_round", 1) or 1
        rows = [r for r in json.loads((sd / "defense_ledger.json").read_text()) if r["round"] >= start]
        def dropped(r: dict) -> bool:  # baseline ledgers carry a weight, not an inclusion flag
            for key in ("included", "admitted"):
                if key in r:
                    return not r[key]
            return float(r["weight"]) == 0.0
        hon = [dropped(r) for r in rows if not r["is_malicious"]]; mal = [dropped(r) for r in rows if r["is_malicious"]]
        a10 = 100 * np.mean([r["test_accuracy"] for r in m[-10:]]); a50 = 100 * np.mean([r["test_accuracy"] for r in m[-50:]])
        s10 = 100 * np.mean([r.get("asr_accuracy", np.nan) for r in m[-10:]]); s50 = 100 * np.mean([r.get("asr_accuracy", np.nan) for r in m[-50:]])
        print(f"| `{lab}/{h.name}` | {cfg['experiment']} | {cfg['baseline']['retrain_epochs']} | {start} | {a10:.1f} | {s10:.1f} | {a50:.1f} | {s50:.1f} | {100*np.mean(hon):.1f}% | {100*np.mean(mal):.1f}% |" if mal else
              f"| `{lab}/{h.name}` | {cfg['experiment']} | {cfg['baseline']['retrain_epochs']} | {start} | {a10:.1f} | – | {a50:.1f} | – | {100*np.mean(hon):.1f}% | – |")
print("\nUndefended seed 0 references: ", end="")
for lab in ("backdoor_cifar10_f10", "benign_cifar10", "femnist_backdoor_f10", "femnist_benign"):
    m = json.loads((pick_hash(lab) / "seed_0" / "metrics.json").read_text())
    print(f"{lab} last-10 ACC {100*np.mean([r['test_accuracy'] for r in m[-10:]]):.1f}, last-50 ACC {100*np.mean([r['test_accuracy'] for r in m[-50:]]):.1f}; ", end="")
print()

# ---------------------------------------------------------------- H. baseline knob scans
print("\n## H. Norm-budget scans of the published baselines (CIFAR-10, last-50 ASR, mean (n))\n")
print("| defense | f | 0.6 | 1.0 | 1.5 | 10 |")
print("|---|---|---|---|---|---|")
for base in ("flame", "foolsgold", "backdoor_indicator", "trimmed_mean", "clipped_mean", "krum", "mesas"):
    for f in ("10", "20"):
        row = [base, f"{f}%"]
        for g, suffix in (("0.6", ""), ("1.0", "_g1.0"), ("1.5", "_g1.5"), ("10", "_g10.0")):
            lab = f"sp_adaptive_backdoor_cifar10_f{f}{suffix}__{base}"
            if (ROOT / lab).exists():
                d = load(lab); row.append(f"{100*np.mean(list(d.values())):.1f} ({len(d)})")
            else:
                row.append("–")
        print("| " + " | ".join(row) + " |")

# ---------------------------------------------------------------- I. Krum accuracy gap in the sweep
print("\n## I. Krum clean accuracy against the next-lowest rule, Figure 3 cells (CIFAR-10)\n")
print("| attack | f | Krum ACC | lowest other rule (ACC) | gap (pts) |")
print("|---|---|---|---|---|")
for attack, budget in (("standard", None), ("adaptive", 0.6)):
    for f in pf.FRACTIONS:
        k = cells.get((attack, budget, f, "Krum"))
        if not k:
            continue
        kacc = 100 * np.mean([v[1] for v in k.values()])
        others = {n: 100 * np.mean([v[1] for v in c.values()]) for (a, b, ff, n), c in cells.items()
                  if a == attack and b == budget and abs(ff - f) < 1e-9 and n != "Krum"}
        n, o = min(others.items(), key=lambda kv: kv[1])
        print(f"| {attack} | {int(f*100)}% | {kacc:.1f} | {n} ({o:.1f}) | {o - kacc:.1f} |")

# ---------------------------------------------------------------- J. sweep cells behind Figures 2 and 3
print("\n## J. Figure 2 and Figure 3 cells: last-50 ASR / ACC by compromised fraction (CIFAR-10, every rule)\n")
for attack, budget in (("standard", None), ("adaptive", 0.6)):
    print(f"\n### {attack} backdoor" + (f" (budget {budget})" if budget else "") + "\n")
    print("| rule | " + " | ".join(f"f={int(f*100)}%" for f in pf.FRACTIONS) + " |")
    print("|---|" + "---|" * len(pf.FRACTIONS))
    for name in pf.SWEEP_ORDER:
        row = [name]
        for f in pf.FRACTIONS:
            c = cells.get((attack, budget, f, name))
            if not c:
                row.append("–"); continue
            a = np.array([v[0] for v in c.values()]) * 100; k = np.array([v[1] for v in c.values()]) * 100
            row.append(f"{a.mean():.1f} ± {a.std(ddof=1) if len(a) > 1 else 0:.1f} / {k.mean():.1f} ({len(a)})")
        print("| " + " | ".join(row) + " |")

# ---------------------------------------------------------------- K. honest update norm behind "budget 0.6 is about a third"
print("\n## K. Honest update norms (CIFAR-10, f=10%, rounds 151 to 200, client_metrics.json)\n")
print("| cell | label/hash | seeds | median honest ‖δ‖ | median attacker ‖δ‖ | budget / honest median |")
print("|---|---|---|---|---|---|")
for k, lab, budget in (("norm-bounded 0.6 vs MW-Peer", "adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian", 0.6),
                       ("norm-bounded 0.6, undefended", "adaptive_backdoor_cifar10_f10", 0.6),
                       ("standard backdoor, undefended", "backdoor_cifar10_f10", None)):
    h = pick_hash(lab); hon, mal = [], []; n = 0
    for sd in sorted(h.glob("seed_*")):
        cf = sd / "client_metrics.json"
        if not cf.exists():
            continue
        n += 1
        for r in json.loads(cf.read_text()):
            if r["round"] > 150:
                (mal if r["is_malicious"] else hon).append(r["update_norm"])
    hm, mm = float(np.median(hon)), float(np.median(mal))
    print(f"| {k} | `{lab}/{h.name}` | {n} | {hm:.2f} | {mm:.2f} | {'–' if budget is None else f'{budget / hm:.2f}'} |")
