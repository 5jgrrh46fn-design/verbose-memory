"""Both appendix AUC tables from one pooling.

The paper's weighting table (detector AUC of the TOPSIS score under each
criterion weighting, M4 alone for reference) and its leave-one-metric-out
table used to come from two analyses that pooled rounds differently, so the
all-five composite disagreed between them. This script computes both from
the same runs, the same rounds and the same pooling: every participating
client of every round of the undefended CIFAR-10 standard-backdoor runs,
scores pooled over rounds and seeds, AUC with the malicious class positive.

The leave-one-out table uses the method's own weighting (CRITIC), so its
"all five" entry equals the CRITIC column of the weighting table by
construction.

Usage (from views/main, see analysis/reproduce.sh):

    uv run python analysis/auc_tables.py
    uv run python analysis/auc_tables.py --window 50 --loo-weighting equal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topfed.analysis.aggregate import SeedRun
from topfed.analysis.scoring import STRATEGY_ORDER, _grouped_windowed, _single_metric_pooled
from topfed.analysis.stats import detector_summary
from topfed.scoring.criteria import TRUST_METRICS, build_batch
from topfed.scoring.topsis import TopsisScorer
from topfed.scoring.weighting import WEIGHTING_STRATEGIES

LABELS = {"10": "backdoor_cifar10_f10", "20": "backdoor_cifar10_f20"}
SHORT = {
    "m1_cosine_to_mean": "M1",
    "m2_norm_ratio": "M2",
    "m3_temporal_cv": "M3",
    "m4_val_loss_delta": "M4",
    "m5_max_layer_deviation": "M5",
}


def load_runs(results_root: Path, label: str) -> list[SeedRun]:
    runs: list[SeedRun] = []
    exp_dir = results_root / label
    per_hash: dict[str, list[SeedRun]] = {}
    for manifest_path in sorted(exp_dir.glob("*/seed_*/manifest.json")):
        seed_dir = manifest_path.parent
        if not (seed_dir / "metrics.json").exists():
            continue
        client_path = seed_dir / "client_metrics.json"
        if not client_path.exists():
            continue
        run = SeedRun(
            experiment=label,
            seed=int(json.loads(manifest_path.read_text())["seed"]),
            per_round=json.loads((seed_dir / "metrics.json").read_text()),
            client_rows=json.loads(client_path.read_text()),
            config_hash=seed_dir.parent.name,
        )
        per_hash.setdefault(run.config_hash, []).append(run)
    if not per_hash:
        raise SystemExit(f"no runs with client metrics under {exp_dir}")
    runs = max(per_hash.values(), key=len)
    return sorted(runs, key=lambda r: r.seed)


def pooled_auc(
    runs: list[SeedRun], metrics: list[str], weighting: str, window: int
) -> float:
    scorer = TopsisScorer(weighting=WEIGHTING_STRATEGIES[weighting])
    malicious: list[float] = []
    benign: list[float] = []
    for run in runs:
        for rows in _grouped_windowed(run, window).values():
            usable = [r for r in rows if all(r.get(m) is not None for m in metrics)]
            if len(usable) < 2:
                continue
            scores = scorer.score(build_batch(usable, metrics))
            for row, score in zip(usable, scores, strict=True):
                (malicious if row["is_malicious"] else benign).append(float(score))
    return float(detector_summary(malicious, benign).auc)


def single_auc(runs: list[SeedRun], metric: str, window: int) -> float:
    mal, ben = _single_metric_pooled(runs, metric, window)
    auc = float(detector_summary(mal, ben).auc)
    # Report the orientation that treats the metric as a malicious-positive
    # detector, as the TOPSIS score is; a cost metric reads below one half.
    return auc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--window", type=int, default=0, help="last N rounds (0 = all)")
    parser.add_argument("--loo-weighting", default="critic", choices=sorted(WEIGHTING_STRATEGIES))
    args = parser.parse_args()

    runs = {f: load_runs(args.results_root, lab) for f, lab in LABELS.items()}
    for f, rs in runs.items():
        print(f"f={f}%: {len(rs)} seeds, {sum(len(r.client_rows) for r in rs)} client rows")

    print("\n## Detector AUC by criterion weighting (CIFAR-10 standard backdoor, undefended runs)")
    head = ["f", "M4 alone"] + [s for s in STRATEGY_ORDER]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for f, rs in runs.items():
        cells = [f"{f}%", f"{single_auc(rs, 'm4_val_loss_delta', args.window):.3f}"]
        cells += [f"{pooled_auc(rs, list(TRUST_METRICS), s, args.window):.3f}" for s in STRATEGY_ORDER]
        print("| " + " | ".join(cells) + " |")

    print(f"\n## Single-metric AUC (malicious positive; cost metrics read below 0.5)")
    print("| f | " + " | ".join(SHORT[m] for m in TRUST_METRICS) + " |")
    print("|" + "---|" * (len(TRUST_METRICS) + 1))
    for f, rs in runs.items():
        print(f"| {f}% | " + " | ".join(f"{single_auc(rs, m, args.window):.3f}" for m in TRUST_METRICS) + " |")

    print(f"\n## Leave-one-metric-out AUC ({args.loo_weighting} weighting)")
    head = ["f", "All five"] + [f"-{SHORT[m]}" for m in TRUST_METRICS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for f, rs in runs.items():
        cells = [f"{f}%", f"{pooled_auc(rs, list(TRUST_METRICS), args.loo_weighting, args.window):.3f}"]
        for m in TRUST_METRICS:
            kept = [k for k in TRUST_METRICS if k != m]
            cells.append(f"{pooled_auc(rs, kept, args.loo_weighting, args.window):.3f}")
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
