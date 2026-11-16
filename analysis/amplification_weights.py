"""Trust-score and malicious-weight diagnostics for the capped weighted median.

For each Peer cell the defense ledger records, per round and per sampled
client, the TOPSIS score, the reputation, the renormalised aggregation weight
and the ground-truth label. This script reduces those ledgers to the
quantities the amplification question needs:

* W_A: the attackers' combined renormalised weight in a round (the lemma's
  condition is W_A < 1/2);
* the attackers' weight relative to an equal share m/K (above 1 means the
  score up-weights them);
* the fraction of rounds in which W_A >= 1/2, and in which the attackers
  are up-weighted;
* the fraction of rounds in which the highest score, and the largest
  renormalised weight, among the sampled clients belongs to an attacker
  (the "scored as the most trustworthy client" statistic);
* the detector AUC of the score (benign higher = trusted, so AUC is
  P(score_benign > score_malicious));
* the undefended and Peer last-50-round ASR for the same cell, from
  metrics.json.

Cells are chosen by label; the hash with most seeds is used, and the
manifest's target_norm is printed so the budget is explicit.

Usage (from views/main, see analysis/reproduce.sh):

    uv run python analysis/amplification_weights.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from topfed.analysis.stats import auc_separability

RESULTS = Path("results")

# (name, Peer label, undefended label)
CELLS = [
    ("CIFAR-10 CNN, standard f10", "backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian", "backdoor_cifar10_f10"),
    ("CIFAR-10 CNN, adaptive 0.6 f10", "adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian", "adaptive_backdoor_cifar10_f10"),
    ("CIFAR-10 CNN, adaptive 0.6 f20", "adaptive_backdoor_cifar10_f20__CappedScoreWeight_critic_wmedian", "adaptive_backdoor_cifar10_f20"),
    ("CIFAR-10 CNN, composed f20", "e4_composed_backdoor_cifar10_f20_g0.6_c0.9__CappedScoreWeight_critic_wmedian", "e4_composed_backdoor_cifar10_f20_g0.6_c0.9"),
    ("CIFAR-10 CNN, adaptive f30", "e2_adaptive_backdoor_cifar10_f30__CappedScoreWeight_critic_wmedian", "e2_adaptive_backdoor_cifar10_f30"),
    ("CIFAR-10 ResNet-18, standard f10", "rn_backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian", "rn_backdoor_cifar10_f10"),
    ("CIFAR-10 ResNet-18, standard f20", "rn_backdoor_cifar10_f20__CappedScoreWeight_critic_wmedian", "rn_backdoor_cifar10_f20"),
    ("CIFAR-10 ResNet-18, adaptive f10", "rn_adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian", "rn_adaptive_backdoor_cifar10_f10"),
    ("CIFAR-10 ResNet-18, adaptive f20", "rn_adaptive_backdoor_cifar10_f20__CappedScoreWeight_critic_wmedian", "rn_adaptive_backdoor_cifar10_f20"),
    ("FEMNIST, standard f10", "femnist_backdoor_f10__CappedScoreWeight_critic_wmedian", "femnist_backdoor_f10"),
    ("FEMNIST, adaptive 0.2 f10", "sp_femnist_adaptive_backdoor_f10_g0.2__CappedScoreWeight_critic_wmedian", "sp_femnist_adaptive_backdoor_f10_g0.2"),
    ("CIFAR-100, standard f10", "c100_backdoor_cifar100_f10__CappedScoreWeight_critic_wmedian", "c100_backdoor_cifar100_f10"),
    ("CIFAR-100, standard f20", "c100_backdoor_cifar100_f20__CappedScoreWeight_critic_wmedian", "c100_backdoor_cifar100_f20"),
    ("CIFAR-100, adaptive 3.0 f10", "c100_adaptive_backdoor_cifar100_f10_g3.0__CappedScoreWeight_critic_wmedian", "c100_adaptive_backdoor_cifar100_f10_g3.0"),
    ("CIFAR-100, adaptive 1.0 f20", "c100_adaptive_backdoor_cifar100_f20_g1.0__CappedScoreWeight_critic_wmedian", "c100_adaptive_backdoor_cifar100_f20_g1.0"),
]


def best_hash(label: str) -> Path | None:
    d = RESULTS / label
    if not d.is_dir():
        return None
    hashes = [h for h in d.iterdir() if h.is_dir()]
    if not hashes:
        return None
    return max(hashes, key=lambda h: len(list(h.glob("seed_*/metrics.json"))))


def last50_asr(label: str) -> tuple[float, int] | None:
    h = best_hash(label)
    if h is None:
        return None
    vals = []
    for p in sorted(h.glob("seed_*/metrics.json")):
        rows = json.load(open(p))[-50:]
        a = [r["asr_accuracy"] for r in rows if "asr_accuracy" in r]
        if a:
            vals.append(st.mean(a))
    return (st.mean(vals), len(vals)) if vals else None


def ledger_stats(h: Path) -> dict[str, float]:
    shares, rel, auc_m, auc_b = [], [], [], []
    n_half = n_up = n_rounds = n_top_score = n_top_weight = 0
    late_shares = []
    for p in sorted(h.glob("seed_*/defense_ledger.json")):
        rows = json.load(open(p))
        by_round: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_round[int(r["round"])].append(r)
        max_round = max(by_round)
        for rnd, rs in by_round.items():
            inc = [r for r in rs if r.get("included", True)]
            mal = [r for r in inc if r["is_malicious"]]
            if not mal:
                continue
            total = sum(r["weight"] for r in inc)
            if total <= 0:
                continue
            w_a = sum(r["weight"] for r in mal) / total
            m, k = len(mal), len(inc)
            shares.append(w_a)
            rel.append(w_a / (m / k))
            n_rounds += 1
            n_half += w_a >= 0.5
            n_up += w_a > m / k
            n_top_score += max(inc, key=lambda r: r["score"])["is_malicious"]
            n_top_weight += max(inc, key=lambda r: r["weight"])["is_malicious"]
            if rnd > max_round - 50:
                late_shares.append(w_a)
            ben = [r["score"] for r in inc if not r["is_malicious"]]
            ms = [r["score"] for r in mal]
            if ben and ms:
                auc_b.append(auc_separability(ben, ms))
    return {
        "seeds": len(list(h.glob("seed_*/defense_ledger.json"))),
        "rounds_with_attacker": n_rounds,
        "mean_W_A": st.mean(shares) if shares else float("nan"),
        "mean_W_A_last50": st.mean(late_shares) if late_shares else float("nan"),
        "mean_rel_share": st.mean(rel) if rel else float("nan"),
        "frac_W_A_ge_half": n_half / n_rounds if n_rounds else float("nan"),
        "frac_upweighted": n_up / n_rounds if n_rounds else float("nan"),
        "frac_top_score": n_top_score / n_rounds if n_rounds else float("nan"),
        "frac_top_weight": n_top_weight / n_rounds if n_rounds else float("nan"),
        "auc_benign_over_mal": st.mean(auc_b) if auc_b else float("nan"),
    }


def main() -> int:
    print("| cell | budget | seeds | undefended ASR | Peer ASR | mean W_A | W_A last 50 | W_A / (m/K) | rounds W_A>=1/2 | rounds up-weighted | rounds attacker top score | rounds attacker top weight | AUC benign>mal |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, peer, und in CELLS:
        h = best_hash(peer)
        if h is None:
            print(f"| {name} | -- | 0 | | | | | | | | |")
            continue
        man = json.load(open(next(h.glob("seed_*/manifest.json"))))
        budget = (man.get("config", {}).get("attack", {}).get("update_manipulator") or {}).get("target_norm", "-")
        s = ledger_stats(h)
        u = last50_asr(und)
        p = last50_asr(peer)
        f = lambda x: f"{x:.3f}"
        print(
            f"| {name} | {budget} | {s['seeds']} | "
            f"{f(u[0]) + f' ({u[1]})' if u else '--'} | {f(p[0]) + f' ({p[1]})' if p else '--'} | "
            f"{f(s['mean_W_A'])} | {f(s['mean_W_A_last50'])} | {f(s['mean_rel_share'])} | "
            f"{f(s['frac_W_A_ge_half'])} | {f(s['frac_upweighted'])} | {f(s['frac_top_score'])} | "
            f"{f(s['frac_top_weight'])} | {f(s['auc_benign_over_mal'])} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
