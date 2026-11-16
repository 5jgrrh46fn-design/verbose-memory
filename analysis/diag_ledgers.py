import sys, json, re, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import last_runs_c100 as L
L.RESULTS = Path("results")
R = L.RESULTS
CELLS = [
 ("krum","adaptive_backdoor_cifar10_f10_g0.6"),("krum","adaptive_backdoor_cifar10_f20_g0.6"),
 ("krum","backdoor_cifar10_f10"),("krum","backdoor_cifar10_f20"),
 ("fltrust","adaptive_backdoor_cifar10_f10_g0.6"),("fltrust","adaptive_backdoor_cifar10_f20_g0.6"),
 ("fltrust","femnist_adaptive_backdoor_f10_g0.2"),("fltrust","femnist_backdoor_f10"),
 ("krum","adaptive_backdoor_cifar100_f20_g1.0"),("krum","backdoor_cifar100_f20"),
 ("fltrust","adaptive_backdoor_cifar100_f10_g1.0"),("fltrust","adaptive_backdoor_cifar100_f20_g1.0"),
 ("fltrust","backdoor_cifar100_f10"),("fltrust","backdoor_cifar100_f20"),
]
def ref_match(exp, rule, diag):
    base = re.sub(r"_g[0-9.]+$", "", exp)
    cands = [p for p in R.glob(f"*{base}*__{rule}/*/seed_*") if not p.parts[1].startswith("diag_")]
    n = 0; labels = set()
    for s, d in diag.items():
        for c in cands:
            if c.name == f"seed_{s}" and (c/"metrics.json").exists() and L.same_metrics(d, c):
                n += 1; labels.add(c.parts[1]); break
    return n, labels
for rule, exp in CELLS:
    diag = L.seeds_of(f"diag_{exp}__{rule}")
    n, labels = ref_match(exp, rule, diag)
    asr = [L.tail(r, "asr_accuracy") for r in diag.values()]
    fn = L.krum_stats if rule == "krum" else L.fltrust_stats
    per = [fn(r) for r in diag.values()]
    col = lambda k: [p[k] for p in per]
    line = f"{rule:8s} {exp:40s} n={len(per)} ASR {L.msd(asr)} rounds {st.fmean(col('rounds')):.0f} head {L.msd(col('head'))} "
    if rule == "krum":
        line += f"selects-attacker {L.msd(col('selected'))} "
    else:
        line += f"top {L.msd(col('top'))} share {L.msd(col('share'))} ratio {L.msd(col('ratio'),1,2)} "
    line += f"norm mal {L.msd(col('norm_mal'),1,3)} hon {L.msd(col('norm_ben'),1,3)} | identical {n}/{len(diag)} {sorted(labels)}"
    print(line)
