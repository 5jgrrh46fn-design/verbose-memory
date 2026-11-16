#!/usr/bin/env bash
# Resumable sweep over (experiment x override), with skip_existing=true so a
# re-run after an interruption skips every (config, seed) whose run already
# completed and executes only what is missing. A crashed run leaves no
# metrics.json, so it is correctly re-run rather than skipped.
#
# KIND selects which config group is varied:
#   defense  -> +defense=<value>  (TOPSIS integration mechanisms; default)
#   baseline -> +baseline=<value> (Byzantine-robust aggregation rules)
#
# Examples:
#   experiments/sweep.sh                        # mechanism comparison (defense)
#   KIND=baseline experiments/sweep.sh          # robust-agg baselines
#   EXPERIMENTS="backdoor_cifar10_f20" VALUES="krum" KIND=baseline experiments/sweep.sh
#   SEEDS="0,1,2" experiments/sweep.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KIND="${KIND:-defense}"
EXPERIMENTS="${EXPERIMENTS:-backdoor_cifar10_f10 backdoor_cifar10_f20 gradient_scaling_cifar10_f20}"
if [ "$KIND" = "defense" ]; then
  VALUES="${VALUES:-topsis_hard_filter topsis_quarantine topsis_temporal_reputation topsis_adaptive}"
else
  VALUES="${VALUES:-krum coordinate_median trimmed_mean}"
fi
SEEDS="${SEEDS:-0,1,2,3,4}"

for exp in $EXPERIMENTS; do
  for val in $VALUES; do
    echo "=========== ${exp} / ${KIND}=${val} $(date -u +%FT%TZ) ==========="
    uv run python experiments/run.py --multirun \
      +experiment="$exp" "+${KIND}=$val" federation.seed="$SEEDS" +skip_existing=true \
      2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
  done
done
echo "########### SWEEP COMPLETE ###########"
