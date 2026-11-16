#!/usr/bin/env bash
# Last runs: undefended reference rows under the adaptive attackers and
# diagnostic re-runs that record which clients Krum selects and how FLTrust
# weights them. Resumable; every cell is a Hydra multirun with
# +skip_existing=true.
#
# Blocks (select with BLOCKS="A1 A2"):
#   A1      undefended under the composed backdoor at (budget, cosine) =
#           (0.6, 0.9), f10/f20, A1_SEEDS (default 5-9), under the existing
#           tag e4_composed_backdoor_cifar10_<f>_g0.6_c0.9.          10 runs
#   A2      undefended under the norm-bounded backdoor at every budget of
#           A2_BUDGETS (default 1.5 2.0), f10/f20, SEEDS, tags
#           sp_adaptive_backdoor_cifar10_<f>_g<budget>.     20 runs per budget
#   A3      diagnostic re-runs, tags diag_<experiment>[_g<budget>], SEEDS:
#           Krum (its ledger holds a one-hot weight per client-round) on the
#           norm-bounded backdoor at 0.6 and the standard backdoor, f10/f20;
#           FLTrust with federation.trace_capture=true (its ledger holds the
#           normalised trust scores) on the FEMNIST norm-bounded backdoor at
#           0.2 and the FEMNIST standard backdoor, f10, and on the CIFAR-10
#           norm-bounded backdoor at 0.6, f10/f20.                   80 runs
#   A3_FIRST  the A3 cells wanted first, A3_FIRST_SEEDS (default 0-4): FLTrust
#           on the FEMNIST norm-bounded backdoor at 0.2, then Krum on the
#           CIFAR-10 norm-bounded backdoor at 0.6, f10/f20.          15 runs
#
# Examples:
#   experiments/sweep_last.sh
#   BLOCKS=A2 A2_BUDGETS=0.75 experiments/sweep_last.sh
#   DRY_RUN=1 experiments/sweep_last.sh                  # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-A1 A2 A3}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
A1_SEEDS="${A1_SEEDS:-5,6,7,8,9}"
A2_BUDGETS="${A2_BUDGETS:-1.5 2.0}"
A3_FIRST_SEEDS="${A3_FIRST_SEEDS:-0,1,2,3,4}"
DRY_RUN="${DRY_RUN:-0}"

# run_cell <tag> <base experiment> <seeds> [override ...]
run_cell() {
  local tag="$1" base="$2" seeds="$3"
  shift 3
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY ${tag} ${base} [$*] seeds=${seeds}"
    return 0
  fi
  echo "=========== ${tag} [$*] seeds=${seeds} $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" \
    federation.seed="$seeds" +skip_existing=true "$@" \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

block_A1() {
  for f in f10 f20; do
    run_cell "e4_composed_backdoor_cifar10_${f}_g0.6_c0.9" "composed_backdoor_cifar10_${f}" "$A1_SEEDS" \
      "attack.update_manipulator.manipulators.0.cosine_target=0.9" \
      "attack.update_manipulator.manipulators.1.target_norm=0.6"
  done
}

block_A2() {
  for g in $A2_BUDGETS; do
    for f in f10 f20; do
      local exp="adaptive_backdoor_cifar10_${f}"
      run_cell "sp_${exp}_g${g}" "$exp" "$SEEDS" "attack.update_manipulator.target_norm=${g}"
    done
  done
}

block_A3() {
  local trace="+federation.trace_capture=true"
  for f in f10 f20; do
    run_cell "diag_adaptive_backdoor_cifar10_${f}_g0.6" "adaptive_backdoor_cifar10_${f}" "$SEEDS" \
      "attack.update_manipulator.target_norm=0.6" "+baseline=krum"
    run_cell "diag_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$SEEDS" "+baseline=krum"
    run_cell "diag_adaptive_backdoor_cifar10_${f}_g0.6" "adaptive_backdoor_cifar10_${f}" "$SEEDS" \
      "attack.update_manipulator.target_norm=0.6" "+baseline=fltrust" "$trace"
  done
  run_cell "diag_femnist_adaptive_backdoor_f10_g0.2" "femnist_adaptive_backdoor_f10" "$SEEDS" \
    "attack.update_manipulator.target_norm=0.2" "+baseline=fltrust" "$trace"
  run_cell "diag_femnist_backdoor_f10" "femnist_backdoor_f10" "$SEEDS" "+baseline=fltrust" "$trace"
}

block_A3_FIRST() {
  run_cell "diag_femnist_adaptive_backdoor_f10_g0.2" "femnist_adaptive_backdoor_f10" "$A3_FIRST_SEEDS" \
    "attack.update_manipulator.target_norm=0.2" "+baseline=fltrust" "+federation.trace_capture=true"
  for f in f10 f20; do
    run_cell "diag_adaptive_backdoor_cifar10_${f}_g0.6" "adaptive_backdoor_cifar10_${f}" "$A3_FIRST_SEEDS" \
      "attack.update_manipulator.target_norm=0.6" "+baseline=krum"
  done
}

for block in $BLOCKS; do
  case "$block" in
    A1|A2|A3|A3_FIRST) "block_${block}" ;;
    *) echo "unknown block: $block" >&2; exit 2 ;;
  esac
done
echo "########### LAST SWEEP COMPLETE (${BLOCKS}) ###########"
