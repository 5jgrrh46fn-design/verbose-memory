import json, sys, statistics as st
from pathlib import Path
from collections import defaultdict
from topfed.analysis.stats import wilcoxon_vs_baseline
R = Path("results")
def load(label):
    # One configuration per label: the hash directory with the most finished
    # seeds (other hashes under a label are single-seed ablations).
    out = {}
    hs = sorted((R/label).iterdir()) if (R/label).is_dir() else []
    hs = sorted(hs, key=lambda h: -len(list(h.glob("seed_*/metrics.json"))))[:1]
    for h in hs:
        for sd in sorted(h.glob("seed_*")):
            me = sd/"metrics.json"
            if not me.exists(): continue
            rows = json.loads(me.read_text())
            if len(rows) != 200: continue
            out[int(sd.name[5:])] = (st.mean(float(r["asr_accuracy"]) for r in rows[-50:]), sd)
    return out
def top(dirs):
    rounds = sole = ties = 0; hx = hn = 0
    for sd in dirs:
        p = sd/"defense_ledger.json"
        if not p.exists(): return None
        by = defaultdict(list)
        for r in json.loads(p.read_text()): by[int(r["round"])].append(r)
        for rs in by.values():
            for r in rs:
                if not r["is_malicious"]: hn += 1; hx += float(r["weight"]) <= 0
            m = [float(r["score"]) for r in rs if r["is_malicious"]]
            h = [float(r["score"]) for r in rs if not r["is_malicious"]]
            if not m: continue
            rounds += 1; sole += max(m) > max(h, default=-1e9); ties += max(m) >= max(h, default=-1e9)
    return 100*sole/rounds, 100*ties/rounds, 100*hx/hn
INT = ["ScoreWeight_critic","TemporalReputation_critic","HardFilter_critic","Quarantine_critic","AdaptiveParticipation_critic","CappedScoreWeight_critic_wmedian"]
for f in ("f10","f20"):
    und = None
    for cand in (f"backdoor_cifar10_{f}", f"sp_backdoor_cifar10_{f}", f"backdoor_cifar10_{f}__undefended", f"sp_backdoor_cifar10_{f}__undefended", f"backdoor_cifar10_{f}__fedavg"):
        u = load(cand)
        if u: print("undefended:", cand, len(u), f"{100*st.mean(v[0] for v in u.values()):.1f}"); und = und or u
    for i in INT:
        best = None
        for pre in ("", "sp_", "e2_", "e3_"):
            c = load(f"{pre}backdoor_cifar10_{f}__{i}")
            if c and (best is None or len(c) > len(best[1])): best = (pre, c)
        if not best: print(f, i, "none"); continue
        pre, c = best
        seeds = sorted(set(c) & set(und))
        a = [c[s][0] for s in seeds]; b = [und[s][0] for s in seeds]
        t = wilcoxon_vs_baseline(a, b)
        tp = top([c[s][1] for s in seeds])
        print(f"{f} {pre}{i:36s} n={len(c)} pairs={len(seeds)} ASR={100*st.mean(a):.1f} und={100*st.mean(b):.1f} delta={100*(st.mean(a)-st.mean(b)):+.1f} p={t.p_value:.4f} top(strict/ties/honest excl)={tp}")
