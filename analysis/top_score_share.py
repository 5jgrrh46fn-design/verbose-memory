"""Share of attacker rounds in which an attacker holds the strictly top score.

Five-metric CRITIC score under the norm-bounded backdoor at budget 0.6,
CIFAR-10, per integration and compromised fraction. A round counts when at
least one compromised client is sampled; it is a top-score round when the
highest attacker score is strictly above every honest score. These are the
horizontal coordinates of the five-metric markers in the gameability figure
(the ``FIVE`` table of ``paper_figures_final.py``).

Run from views/main (reads ``results/`` relative to the working directory).
"""
import json
from collections import defaultdict
from pathlib import Path

R = Path("results")
INTEGRATIONS = ["ScoreWeight", "TemporalReputation", "HardFilter", "Quarantine",
                "AdaptiveParticipation", "CappedScoreWeight"]

print("| fraction | integration | config hash | seeds | top-score share (%) |")
print("|---|---|---|---|---|")
for f in (10, 20):
    for integ in INTEGRATIONS:
        label = (f"adaptive_backdoor_cifar10_f{f}__{integ}_critic"
                 + ("_wmedian" if integ.startswith("Capped") else ""))
        h = max(sorted((R / label).iterdir()),
                key=lambda d: len(list(d.glob("seed_*/metrics.json"))))
        rounds = top = seeds = 0
        for sd in sorted(h.glob("seed_*")):
            p = sd / "defense_ledger.json"
            if not p.exists():
                continue
            seeds += 1
            by = defaultdict(list)
            for r in json.loads(p.read_text()):
                by[int(r["round"])].append(r)
            for rs in by.values():
                mal = [float(r["score"]) for r in rs if r["is_malicious"]]
                hon = [float(r["score"]) for r in rs if not r["is_malicious"]]
                if not mal:
                    continue
                rounds += 1
                top += max(mal) > max(hon, default=-1e9)
        print(f"| {f}% | {integ} | {h.name} | {seeds} | {100 * top / rounds:.1f} |")
