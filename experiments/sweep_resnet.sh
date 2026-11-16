#!/usr/bin/env bash
# ResNet-18 (GroupNorm) appendix sweep on CIFAR-10. Every cell carries
# +model=resnet18_gn and +skip_existing=true, so it is resumable and never
# shares a config hash with a small-CNN cell. Tags are rn_<experiment>
# [_g<budget>] [_lr<lr>], so the runs also never share a label directory with
# a CNN cell.
#
# Blocks (select with BLOCKS="PROBE" or BLOCKS="MAIN"):
#   PROBE  seed PROBE_SEED only, undefended: benign_cifar10 and
#          backdoor_cifar10_f20 at the default LR and at PROBE_LR (lr 0.05,
#          the usual ResNet-18 local rate), adaptive_backdoor_cifar10_f20 at
#          the attack config's budget, and adaptive_backdoor_cifar10_f20 at
#          target_norm in PROBE_BUDGETS. The budget 0.6 was calibrated on the
#          small CNN; ResNet-18 deltas live on another norm scale, so the probe
#          finds the budget at which the adaptive attack still implants with
#          no defense. The probe decides LR and BUDGET for MAIN.   10 runs
#   MAIN   {backdoor_cifar10_<FRACTION>, adaptive_backdoor_cifar10_<FRACTION>
#          at BUDGET} (FRACTION f20 by default, f10 to test the regime where
#          the small-CNN defenses hold) x
#          {undefended, capped-wmed, FLTrust-median, FLAME, median} x SEEDS.
#                                                                   50 runs
#
# Variables:
#   LR      federation.local_learning_rate for every cell (default 0.01, the
#           CNN's; set LR=0.05 for MAIN if the probe says so). Non-default
#           values are tagged _lr<LR>.
#   BUDGET  the adaptive attacker's target_norm for MAIN (default 0.6, the
#           attack config's; set from the probe). Tagged _g<BUDGET> whenever
#           it differs from the attack config's value.
#
# Examples:
#   BLOCKS=PROBE experiments/sweep_resnet.sh
#   BLOCKS=MAIN LR=0.05 BUDGET=1.5 experiments/sweep_resnet.sh
#   DRY_RUN=1 experiments/sweep_resnet.sh                 # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-PROBE MAIN}"
SEEDS="${SEEDS:-0,1,2,3,4}"
PROBE_SEED="${PROBE_SEED:-0}"
LR="${LR:-0.01}"
PROBE_LR="${PROBE_LR:-0.05}"
BUDGET="${BUDGET:-0.6}"
# Compromise level for MAIN: f20 (the default) or f10 (the regime where
# the small-CNN defenses hold; run it to see whether the ordering transfers).
FRACTION="${FRACTION:-f20}"
PROBE_BUDGETS="${PROBE_BUDGETS:-0.6 1.0 1.5 3.0}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_LR="0.01"
DEFAULT_BUDGET="0.6"
MODEL="+model=resnet18_gn"
CAPPED="+defense=topsis_capped_weighted_median"
H3="+baseline=fltrust_median"
FLAME="+baseline=flame"
MEDIAN="+baseline=coordinate_median"
DEFENSES=(none "$CAPPED" "$H3" "$FLAME" "$MEDIAN")

# run_cell <tag> <base experiment> <seeds> [override ...]
run_cell() {
  local tag="$1" base="$2" seeds="$3"
  shift 3
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY ${tag} ${base} [${MODEL} $*] seeds=${seeds}"
    return 0
  fi
  echo "=========== ${tag} [$*] seeds=${seeds} $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" "$MODEL" \
    federation.seed="$seeds" +skip_existing=true "$@" \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

# run_defended <tag> <base> <seeds> <lr> <defense override or "none"> [override ...]
run_defended() {
  local tag="$1" base="$2" seeds="$3" lr="$4" defense="$5"
  shift 5
  local -a extra=("federation.local_learning_rate=${lr}")
  [ "$lr" != "$DEFAULT_LR" ] && tag="${tag}_lr${lr}"
  [ "$defense" != "none" ] && extra+=("$defense")
  run_cell "$tag" "$base" "$seeds" "$@" "${extra[@]}"
}

# budget_tag <experiment> <budget>: rn_<experiment>[_g<budget>]
budget_tag() {
  local exp="$1" budget="$2"
  if [ "$budget" = "$DEFAULT_BUDGET" ]; then
    echo "rn_${exp}"
  else
    echo "rn_${exp}_g${budget}"
  fi
}

block_PROBE() {
  local s="$PROBE_SEED"
  for lr in "$DEFAULT_LR" "$PROBE_LR"; do
    run_defended "rn_benign_cifar10" "benign_cifar10" "$s" "$lr" none
    run_defended "rn_backdoor_cifar10_f20" "backdoor_cifar10_f20" "$s" "$lr" none
  done
  local exp="adaptive_backdoor_cifar10_f20"
  for g in $PROBE_BUDGETS; do
    run_defended "$(budget_tag "$exp" "$g")" "$exp" "$s" "$DEFAULT_LR" none \
      "attack.update_manipulator.target_norm=${g}"
  done
}

block_MAIN() {
  local d
  for d in "${DEFENSES[@]}"; do
    run_defended "rn_backdoor_cifar10_${FRACTION}" "backdoor_cifar10_${FRACTION}" "$SEEDS" "$LR" "$d"
  done
  local exp="adaptive_backdoor_cifar10_${FRACTION}"
  for d in "${DEFENSES[@]}"; do
    run_defended "$(budget_tag "$exp" "$BUDGET")" "$exp" "$SEEDS" "$LR" "$d" \
      "attack.update_manipulator.target_norm=${BUDGET}"
  done
}

for block in $BLOCKS; do
  case "$block" in
    PROBE) block_PROBE ;;
    MAIN) block_MAIN ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "sweep_resnet done $(date -u +%FT%TZ)"
