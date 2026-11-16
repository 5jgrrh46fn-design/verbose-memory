"""Numbers for the last runs on the RTX 5070 Ti box (blocks A1-A3).

Usage (from views/main, see analysis/reproduce.sh): uv run python analysis/last_runs_analysis.py
Prints markdown. ASR and accuracy are means over the last 50 rounds, the
convention of ``topfed.analysis.e3``.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from topfed.analysis.e3 import CONVERGENCE_WINDOW as W
from topfed.analysis.stats import wilcoxon_vs_baseline

R = Path("results")


def seeds(label: str, hash_prefix: str = "") -> dict[int, Path]:
    return {int(sd.name[5:]): sd for sd in sorted((R / label).glob(f"{hash_prefix}*/seed_*"))
            if (sd / "metrics.json").exists()}


def tail(sd: Path, key: str) -> float:
    m = json.loads((sd / "metrics.json").read_text())
    assert len(m) == 200, sd
    return st.mean(r[key] for r in m[-W:])


def cell(labels: list[str]) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for lab in labels:
        for s, sd in seeds(lab).items():
            assert s not in out, (lab, s)
            out[s] = (tail(sd, "asr_accuracy"), tail(sd, "test_accuracy"))
    return out


def fmt(c: dict[int, tuple[float, float]]) -> str:
    a = [100 * v[0] for v in c.values()]
    b = [100 * v[1] for v in c.values()]
    if not a:
        return "no runs | | |"
    sd = lambda x: st.stdev(x) if len(x) > 1 else float("nan")  # noqa: E731
    return (f"{st.mean(a):.1f} | {sd(a):.1f} | {st.mean(b):.1f} | {sd(b):.1f} | "
            f"{len(a)} ({','.join(map(str, sorted(c)))})")


print("### A1. Undefended under the composed attacker (0.6, 0.9)\n")
print("| fraction | ASR mean | ASR sd | acc mean | acc sd | seeds |\n|---|---|---|---|---|---|")
a1 = {}
for f in ("f10", "f20"):
    a1[f] = cell([f"e4_composed_backdoor_cifar10_{f}_g0.6_c0.9"])
    print(f"| {f} | {fmt(a1[f])} |")
print("\n| fraction | capped wmed ASR | undefended ASR | Δ (pp) | capped above on | Wilcoxon W | p (two-sided) | rank-biserial |\n|---|---|---|---|---|---|---|---|")
for f in ("f10", "f20"):
    d = cell([f"e4_composed_backdoor_cifar10_{f}_g0.6_c0.9__CappedScoreWeight_critic_wmedian"])
    s = sorted(set(d) & set(a1[f]))
    x = [d[k][0] for k in s]
    y = [a1[f][k][0] for k in s]
    t = wilcoxon_vs_baseline(x, y)
    print(f"| {f} | {100*st.mean(x):.1f} | {100*st.mean(y):.1f} | {100*(st.mean(x)-st.mean(y)):+.1f} | "
          f"{sum(p > q for p, q in zip(x, y))}/{len(s)} | {t.statistic:g} | {t.p_value:.4f} | {t.rank_biserial:+.2f} |")

print("\n### A2. Undefended under the norm-bounded attacker\n")
print("| fraction | budget | ASR mean | ASR sd | acc mean | acc sd | seeds | labels |\n|---|---|---|---|---|---|---|---|")
for f in ("f10", "f20"):
    e = f"adaptive_backdoor_cifar10_{f}"
    und06 = {"f10": "d7655368", "f20": "0e7cb95c"}[f]
    rows = {
        "0.6": None,
        "0.75": [f"sp_{e}_g0.75"],
        "1.0": [f"e4_{e}_g1.0", f"sp_{e}_g1.0"],
        "1.5": [f"sp_{e}_g1.5"],
        "2.0": [f"sp_{e}_g2.0"],
    }
    for g, labs in rows.items():
        if labs is None:
            c = {s: (tail(sd, "asr_accuracy"), tail(sd, "test_accuracy")) for s, sd in seeds(e, und06).items()}
            labs = [f"{e}/{und06}"]
        else:
            c = cell(labs)
        print(f"| {f} | {g} | {fmt(c)} | {', '.join(labs)} |")


def rounds(sd: Path):
    """Per round: list of (client_id, weight, is_malicious), and update norms."""
    led = defaultdict(list)
    for r in json.loads((sd / "defense_ledger.json").read_text()):
        led[r["round"]].append((r["client_id"], r["weight"], r["is_malicious"]))
    norms = {(r["round"], r["client_id"]): r["update_norm"]
             for r in json.loads((sd / "client_metrics.json").read_text())}
    return led, norms


print("\n### A3. Krum's selections\n")
print("| cell | seeds | rounds with a sampled attacker | attacker selected (share of those rounds) | attackers' head-count share in those rounds | selected / head-count | attacker selected, per-seed min-max |\n|---|---|---|---|---|---|---|")
for lab in [f"diag_{a}_cifar10_{f}{g}__krum" for a, g in (("adaptive_backdoor", "_g0.6"), ("backdoor", "")) for f in ("f10", "f20")]:
    n_r = n_sel = 0
    head = []
    per = []
    ss = seeds(lab)
    for s, sd in ss.items():
        led, _ = rounds(sd)
        assert len(led) == 200, (sd, len(led))
        a = b = 0
        for rows in led.values():
            k = sum(m for _, _, m in rows)
            if k == 0:
                continue
            a += 1
            b += any(m and w > 0 for _, w, m in rows)
            head.append(k / len(rows))
        n_r += a
        n_sel += b
        per.append(b / a)
    if not ss:
        print(f"| {lab} | none | | | | | |")
        continue
    print(f"| {lab} | {len(ss)} | {n_r} of {200*len(ss)} | {100*n_sel/n_r:.1f}% | {100*np.mean(head):.1f}% | "
          f"{(n_sel/n_r)/np.mean(head):.2f} | {100*min(per):.1f}-{100*max(per):.1f}% |")

print("\n### A3. FLTrust's weights\n")
print("| cell | seeds | rounds with a sampled attacker | an attacker's trust strictly above every honest client's | attackers' weight share | head-count share | weight / head-count | rounds with all trust zero | attacker norm (before rescaling) | honest norm |\n|---|---|---|---|---|---|---|---|---|---|")
for lab in ["diag_femnist_adaptive_backdoor_f10_g0.2__fltrust", "diag_femnist_backdoor_f10__fltrust",
            "diag_adaptive_backdoor_cifar10_f10_g0.6__fltrust", "diag_adaptive_backdoor_cifar10_f20_g0.6__fltrust"]:
    ss = seeds(lab)
    n_r = n_top = n_zero = 0
    wshare, head, an, hn = [], [], [], []
    for s, sd in ss.items():
        led, norms = rounds(sd)
        assert len(led) == 200, (sd, len(led))
        for rnd, rows in led.items():
            for cid, _, m in rows:
                (an if m else hn).append(norms[(rnd, cid)])
            k = sum(m for _, _, m in rows)
            if k == 0:
                continue
            n_r += 1
            mal = [w for _, w, m in rows if m]
            hon = [w for _, w, m in rows if not m]
            n_top += (not hon) or max(mal) > max(hon)
            tot = sum(mal) + sum(hon)
            n_zero += tot == 0
            wshare.append(sum(mal) / tot if tot > 0 else 0.0)
            head.append(k / len(rows))
    if not ss:
        print(f"| {lab} | none | | | | | | | | |")
        continue
    print(f"| {lab} | {len(ss)} | {n_r} of {200*len(ss)} | {100*n_top/n_r:.1f}% | {100*np.mean(wshare):.1f}% | "
          f"{100*np.mean(head):.1f}% | {np.mean(wshare)/np.mean(head):.2f} | {n_zero} | {np.mean(an):.3f} | {np.mean(hn):.3f} |")
