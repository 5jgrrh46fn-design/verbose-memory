import pickle, sys, json, statistics as st
from pathlib import Path
import numpy as np
from topfed.analysis import e3
from topfed.analysis.stats import holm_bonferroni, wilcoxon_vs_baseline
cells = pickle.load(open(sys.argv[1], "rb"))
RULES = ["median","trimmed_mean","clipped-mean","krum","foolsgold","fltrust","flame","mesas","indicator"]
NAMES = {"median":"Median","trimmed_mean":"Trimmed mean","clipped-mean":"Clipped mean","krum":"Krum","foolsgold":"FoolsGold","fltrust":"FLTrust","flame":"FLAME","mesas":"MESAS","indicator":"BackdoorIndicator"}
GRID = [("cifar10","adaptive","CIFAR-10 NB"),("cifar10","direction_aligned","CIFAR-10 DA"),("femnist","adaptive","FEMNIST NB"),("cifar100","adaptive","CIFAR-100 NB")]
out = []
for ds, atk, name in GRID:
    for f in (0.1, 0.2):
        su = e3.select_cell(cells, ds, atk, f, "undefended")
        cu = e3._candidates(cells, ds, atk, f, "undefended", e3.REFERENCE_VAL[ds])
        print(f"undefended {name} f{int(f*100)}: knob used {su.strength}; by knob " + ", ".join(f"{k:g}: {100*m:.1f} ({n})" for k, (m, n) in sorted(su.by_strength.items())), file=sys.stderr)
        col = []
        for r in RULES:
            sr = e3.select_cell(cells, ds, atk, f, r)
            if sr is None: continue
            a = e3.paired("x","x","asr_final", sr.runs, su.runs)
            same = cu.get(sr.strength)
            b = e3.paired("x","x","asr_final", sr.runs, same) if same else None
            acc = e3.paired("x","x","acc_final", sr.runs, su.runs)
            us = {x.seed: x.asr_final for x in su.runs}
            wins = sum(1 for x in sr.runs if x.seed in us and x.asr_final > us[x.seed])
            col.append(dict(col=f"{name} {int(f*100)}%", rule=NAMES[r], k=sr.strength, ku=su.strength, m=100*np.mean([x.asr_final for x in sr.runs]),
                            mu=100*np.mean([x.asr_final for x in su.runs]), d=100*a.mean_diff, n=a.n, wins=wins, p=a.p_value,
                            db=(100*b.mean_diff if b else None), pb=(b.p_value if b else None), nb=(b.n if b else None), dacc=100*acc.mean_diff))
        h, _ = holm_bonferroni([c["p"] for c in col])
        ib = [i for i, c in enumerate(col) if c["pb"] is not None]
        hb, _ = holm_bonferroni([col[i]["pb"] for i in ib]) if ib else ([], None)
        for c, hp in zip(col, h): c["holm"] = hp
        for i, hp in zip(ib, hb): col[i]["holmb"] = hp
        out += col
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("| column | rule | rule ASR | undefended | Δ | seeds above | p | Holm p (column) | Δ at the rule's knob | p | Holm p | Δ clean acc |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in out:
    if c["d"] <= 0 and (c["db"] is None or c["db"] <= 0): continue
    b = f"{c['db']:+.1f} | {c['pb']:.4f} | {c['holmb']:.3f}" if c["db"] is not None else "not run | – | –"
    print(f"| {c['col']} | {c['rule']} (knob {c['k']:g}; und. {c['ku']:g}) | {c['m']:.1f} | {c['mu']:.1f} | {c['d']:+.1f} | {c['wins']}/{c['n']} | {c['p']:.4f} | {c['holm']:.3f} | {b} | {c['dacc']:+.1f} |")
print("sig above (column Holm<0.05, best response):", [(c["col"], c["rule"]) for c in out if c["d"] > 0 and c["holm"] < 0.05])
print("sig above (same knob):", [(c["col"], c["rule"]) for c in out if c["db"] is not None and c["db"] > 0 and c["holmb"] < 0.05])
# composed cross-box
R = Path(__file__).resolve().parents[1] / "results"
def load(label):
    o = {}
    for sd in sorted((R/label).glob("*/seed_*")):
        M = json.loads((sd/"metrics.json").read_text())
        if len(M) == 200: o[int(sd.name[5:])] = st.mean(r["asr_accuracy"] for r in M[-50:])
    return o
for f in ("f10","f20"):
    peer = load(f"e4_composed_backdoor_cifar10_{f}_g0.6_c0.9__CappedScoreWeight_critic_wmedian")
    for und_lab in (f"e4_composed_backdoor_cifar10_{f}_g0.6_c0.9", f"fin_composed_backdoor_cifar10_{f}_g0.6_c0.9"):
        und = load(und_lab); s = sorted(set(peer) & set(und))
        a = [peer[x] for x in s]; b = [und[x] for x in s]; t = wilcoxon_vs_baseline(a, b)
        print(f"composed (0.6,0.9) {f}: Peer (5070 Ti) vs {und_lab}: n={len(s)} {100*st.mean(a):.1f} vs {100*st.mean(b):.1f}, Δ={100*(st.mean(a)-st.mean(b)):+.1f}, above on {sum(x>y for x,y in zip(a,b))}/{len(s)}, p={t.p_value:.4f}")
