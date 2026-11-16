import re, json, statistics as st, itertools, sys
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from topfed.analysis.stats import wilcoxon_vs_baseline
BASE = Path(__file__).resolve().parents[1]
text = (BASE/"tables/final-queue.md").read_text()
GATES = ["ScoreWeight","TemporalReputation","HardFilter","Quarantine","AdaptiveParticipation"]
CAP = "Capped weighted median"
def sections():
    out = {}; title = None
    for line in text.splitlines():
        if line.startswith("### "):
            title = line[4:].strip() if line.startswith("### Score:") else None
            if title: out[title] = {}
            continue
        if title is None or not line.startswith("|"): continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) < 16 or c[0] not in GATES + [CAP]: continue
        for f, d, p, h, t in ((10, 2, 3, 4, 7), (20, 10, 11, 12, 15)):
            dm = re.match(r"([+-][0-9.]+)", c[d]); tm = re.match(r"[0-9.]+ / ([0-9.]+)", c[t])
            if dm and tm:
                out[title][(c[0], f)] = dict(delta=float(dm.group(1)), p=float(c[p]), holm=(float(c[h]) if re.match(r"[0-9.]+$", c[h]) else None), top=float(tm.group(1)))
    return out
sec = sections()
for k in sec: print("section:", k, len(sec[k]), file=sys.stderr)
m4 = next(v for k, v in sec.items() if "M4 alone" in k)
fg_nb = next(v for k, v in sec.items() if "FoolsGold" in k and "norm-bounded" in k)
fg_c9 = next(v for k, v in sec.items() if "FoolsGold" in k and "c = 0.9" in k)
fg_c5 = next(v for k, v in sec.items() if "FoolsGold" in k and "c = 0.5" in k)
cos = {c: next(v for k, v in sec.items() if "direction-aligned" in k and f"c = {c}" in k) for c in ("0.5","0.7","0.9","0.95")}
blk = text.split("### Cosine score: the attacker's best response per integration")[1].split("###")[0]
best = {}
for line in blk.splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) >= 9 and c[0] in GATES + [CAP]: best[(c[0], 10)], best[(c[0], 20)] = c[1], c[5]
cosine = {k: cos[c][k] for k, c in best.items()}
FIVE = {("ScoreWeight", 10): (17.4, 55.2), ("TemporalReputation", 10): (15.6, 55.8), ("HardFilter", 10): (20.4, 57.0), ("Quarantine", 10): (8.5, 60.3),
        ("AdaptiveParticipation", 10): (22.2, 57.7), (CAP, 10): (-11.9, 54.1), ("ScoreWeight", 20): (11.7, 59.4), ("TemporalReputation", 20): (10.8, 59.6),
        ("HardFilter", 20): (12.2, 61.4), ("Quarantine", 20): (6.7, 59.8), ("AdaptiveParticipation", 20): (15.5, 64.1), (CAP, 20): (-7.3, 50.2)}
# Unadjusted paired Wilcoxon p against undefended, ten seeds (supplementary cells, section A).
FIVE_P = {("Quarantine", 10): 0.0371, ("Quarantine", 20): 0.0039, (CAP, 10): 0.0039}
five = {k: dict(delta=d, top=t, p=FIVE_P.get(k, 0.0020), holm=None) for k, (d, t) in FIVE.items()}
STD = {("ScoreWeight",10):(-66.7,0.0,0.0020),("TemporalReputation",10):(-59.1,0.0,0.0625),("HardFilter",10):(-74.1,0.0,0.0625),("Quarantine",10):(-31.4,0.2,0.0625),("AdaptiveParticipation",10):(-77.7,0.0,0.0625),(CAP,10):(-77.1,0.0,0.0020),
       ("ScoreWeight",20):(-31.7,1.0,0.0020),("TemporalReputation",20):(-20.0,1.7,0.0625),("HardFilter",20):(-69.3,0.3,0.0625),("Quarantine",20):(-6.5,4.4,0.3125),("AdaptiveParticipation",20):(-75.2,0.0,0.0625),(CAP,20):(-73.0,0.3,0.0020)}
std = {k: dict(delta=d, top=t, p=p, holm=None) for k, (d, t, p) in STD.items()}
groups = [("five metrics", "norm-bounded", five), ("M4 alone", "norm-bounded", m4), ("server cosine", "direction-aligned", cosine),
          ("FoolsGold weight", "norm-bounded", fg_nb), ("FoolsGold weight", "composed (0.6, 0.9)", fg_c9), ("FoolsGold weight", "composed (0.6, 0.5)", fg_c5),
          ("five metrics", "standard backdoor", std)]
pts = [dict(score=s, atk=a, integ=i, f=f, **v) for s, a, g in groups for (i, f), v in g.items()]
json.dump(pts, open(sys.argv[1], "w"), indent=1)
def corr(ps, name):
    x = [p["top"] for p in ps]; y = [p["delta"] for p in ps]
    r = spearmanr(x, y); k = kendalltau(x, y)
    print(f"{name}: n={len(ps)} Spearman rho={r.statistic:.2f} (p={r.pvalue:.1e}); Kendall tau={k.statistic:.2f} (p={k.pvalue:.1e})")
gates = [p for p in pts if p["integ"] != CAP]
corr(gates, "gates, mean weights, sampling: all points")
corr([p for p in gates if p["atk"] != "standard backdoor"], "same without the standard-backdoor cells")
corr([p for p in pts if p["integ"] == CAP], "capped weighted median only")
corr(pts, "every point incl. capped median")
# cutoff: excess := delta > 0 (mean) ; and excess_sig := delta>0 and p<0.05
def fit(train, key):
    xs = sorted({p["top"] for p in train}); cands = [(a + b) / 2 for a, b in zip(xs, xs[1:])]
    bestacc, taus = -1, []
    for t in cands:
        acc = sum((p["top"] >= t) == key(p) for p in train) / len(train)
        if acc > bestacc + 1e-12: bestacc, taus = acc, [t]
        elif abs(acc - bestacc) < 1e-12: taus.append(t)
    return bestacc, min(taus), max(taus)
for name, key in (("excess in mean (Δ > 0)", lambda p: p["delta"] > 0), ("excess with unadjusted p < 0.05", lambda p: p["delta"] > 0 and p["p"] < 0.05)):
    print("\ncutoff, definition:", name)
    acc, lo, hi = fit(gates, key); print(f"  all four scores: best cutoff anywhere in [{lo:.1f}, {hi:.1f}] %, accuracy {acc*100:.0f}% of {len(gates)} cells")
    for held in ("five metrics", "M4 alone", "server cosine", "FoolsGold weight"):
        tr = [p for p in gates if p["score"] != held]; te = [p for p in gates if p["score"] == held]
        acc, lo, hi = fit(tr, key); mid = (lo + hi) / 2
        res = [sum((p["top"] >= t) == key(p) for p in te) for t in (lo, mid, hi)]
        print(f"  hold out {held:17s}: cutoff from the other three in [{lo:.1f}, {hi:.1f}] (train acc {acc*100:.0f}%); held-out correct {res[0]}/{len(te)} at low end, {res[1]}/{len(te)} at midpoint, {res[2]}/{len(te)} at high end; positives in held-out {sum(key(p) for p in te)}")
# table 4: largest excess per integration x score
print("\n| integration | five metrics | M4 alone | server cosine | FoolsGold weight |\n|---|---|---|---|---|")
for i in GATES + [CAP]:
    row = []
    for s in ("five metrics", "M4 alone", "server cosine", "FoolsGold weight"):
        c = [p for p in pts if p["integ"] == i and p["score"] == s and p["atk"] != "standard backdoor"]
        b = max(c, key=lambda p: p["delta"])
        row.append(f"{b['delta']:+.1f} (f{b['f']}, {b['atk']}, p={b['p']:.3f}, top {b['top']:.0f}%)")
    print(f"| {i} | " + " | ".join(row) + " |")
