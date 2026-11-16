#!/usr/bin/env bash
# CIFAR-100 sweep: the third dataset's probe and its attack-by-defense grid.
# Every cell carries +skip_existing=true, so it is resumable, and the dataset
# override dataset=$DATASET, so one script serves the 100-class variant
# (DATASET=cifar100, tags c100_<experiment>[_g<budget>]) and the 20-superclass
# variant (DATASET=cifar100_coarse, tags c100c_<experiment>[_g<budget>]). The
# tag keeps the two variants in separate label directories; their config
# hashes differ anyway (dataset.name, num_classes, val_per_class, coarse).
#
# Blocks (select with BLOCKS="PROBE" or BLOCKS="MAIN"):
#   PROBE  seed PROBE_SEED only, undefended: benign_cifar100,
#          backdoor_cifar100_f20, and adaptive_backdoor_cifar100_f20 at
#          target_norm in PROBE_BUDGETS. The probe decides two things for
#          MAIN: the dataset variant and the adaptive budget.      7 runs
#   MAIN   {backdoor, adaptive_backdoor at BUDGET} x {f10, f20} x
#          {undefended, capped-wmed, FLTrust-median, FLTrust, median, FLAME,
#          clipped mean} x SEEDS.                    28 runs per seed
#   UNDEF  undefended adaptive_backdoor at the (experiment suffix, budget)
#          pairs of UNDEF_CELLS x SEEDS, under the PROBE and MAIN tags, so
#          the undefended reference exists at budgets above MAIN's.
#                                                    30 runs by default
#   DIAG   diagnostic re-runs under diag_<experiment>[_g<budget>] tags, so
#          nothing shares a directory with a reported cell: FLTrust with
#          +federation.trace_capture=true (writes its trust weights) on
#          {backdoor, adaptive_backdoor at BUDGET} x {f10, f20}, and Krum
#          (writes its selection) on the same two attacks at f20, x SEEDS.
#                                                    6 runs per seed
#
# Decision rule, read from the PROBE runs (last-50-round means of
# results/<tag>/*/seed_0/metrics.json, test_accuracy and asr_accuracy):
#   1. Accuracy ceiling. If the benign probe's last-50-round clean accuracy
#      is below 0.30, the 100-class task is too hard for the small CNN to
#      read attack effects: rerun PROBE with DATASET=cifar100_coarse and run
#      MAIN on the coarse variant (DATASET=cifar100_coarse). If the coarse
#      probe is also below 0.30, stop: GTSRB is the next fallback and that
#      switch is made by hand.
#   2. Budget. BUDGET for MAIN is the smallest probed target_norm whose
#      undefended last-50-round ASR is at least 0.5. If no probed budget
#      reaches 0.5, stop and report; do not run MAIN.
#
# Variables:
#   DATASET  cifar100 (default) or cifar100_coarse; applied as a Hydra
#            override to every run of both blocks.
#   BUDGET   the adaptive attacker's target_norm for MAIN. No default: set
#            it from the probe. Tagged _g<BUDGET> whenever it differs from
#            the attack config's value (0.6).
#   SEEDS    comma-separated seeds for MAIN (default 0-9). Run MAIN in
#            chunks (SEEDS=0,1 then SEEDS=2,3 ...) with a backup between them.
#
# Examples:
#   BLOCKS=PROBE experiments/sweep_cifar100.sh
#   BLOCKS=PROBE DATASET=cifar100_coarse experiments/sweep_cifar100.sh
#   BLOCKS=MAIN BUDGET=0.6 SEEDS=0,1 experiments/sweep_cifar100.sh
#   BLOCKS=MAIN BUDGET=1.0 DATASET=cifar100_coarse SEEDS=2,3 experiments/sweep_cifar100.sh
#   DRY_RUN=1 BLOCKS=MAIN BUDGET=0.6 experiments/sweep_cifar100.sh   # list cells
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-PROBE}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
PROBE_SEED="${PROBE_SEED:-0}"
DATASET="${DATASET:-cifar100}"
BUDGET="${BUDGET:-}"
PROBE_BUDGETS="${PROBE_BUDGETS:-0.3 0.6 1.0 1.5 3.0}"
DRY_RUN="${DRY_RUN:-0}"
UNDEF_CELLS="${UNDEF_CELLS:-f20:1.5 f20:3.0 f10:1.5}"

DEFAULT_BUDGET="0.6"
case "$DATASET" in
  cifar100) PREFIX="c100" ;;
  cifar100_coarse) PREFIX="c100c" ;;
  *) echo "DATASET must be cifar100 or cifar100_coarse; got ${DATASET}" >&2; exit 2 ;;
esac
CAPPED="+defense=topsis_capped_weighted_median"
H3="+baseline=fltrust_median"
FLTRUST="+baseline=fltrust"
MEDIAN="+baseline=coordinate_median"
FLAME="+baseline=flame"
CLIPPED="+baseline=clipped_mean"
DEFENSES=(none "$CAPPED" "$H3" "$FLTRUST" "$MEDIAN" "$FLAME" "$CLIPPED")

# run_cell <tag> <base experiment> <seeds> [override ...]
run_cell() {
  local tag="$1" base="$2" seeds="$3"
  shift 3
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY ${tag} ${base} [dataset=${DATASET} $*] seeds=${seeds}"
    return 0
  fi
  echo "=========== ${tag} [dataset=${DATASET} $*] seeds=${seeds} $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" "dataset=${DATASET}" \
    federation.seed="$seeds" +skip_existing=true "$@" \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

# run_defended <tag> <base> <seeds> <defense override or "none"> [override ...]
run_defended() {
  local tag="$1" base="$2" seeds="$3" defense="$4"
  shift 4
  if [ "$defense" = "none" ]; then
    run_cell "$tag" "$base" "$seeds" "$@"
  else
    run_cell "$tag" "$base" "$seeds" "$@" "$defense"
  fi
}

# budget_tag <experiment> <budget>: <PREFIX>_<experiment>[_g<budget>]
budget_tag() {
  local exp="$1" budget="$2"
  if [ "$budget" = "$DEFAULT_BUDGET" ]; then
    echo "${PREFIX}_${exp}"
  else
    echo "${PREFIX}_${exp}_g${budget}"
  fi
}

block_PROBE() {
  local s="$PROBE_SEED"
  run_defended "${PREFIX}_benign_cifar100" "benign_cifar100" "$s" none
  run_defended "${PREFIX}_backdoor_cifar100_f20" "backdoor_cifar100_f20" "$s" none
  local exp="adaptive_backdoor_cifar100_f20"
  for g in $PROBE_BUDGETS; do
    run_defended "$(budget_tag "$exp" "$g")" "$exp" "$s" none \
      "attack.update_manipulator.target_norm=${g}"
  done
}

block_MAIN() {
  if [ -z "$BUDGET" ]; then
    echo "MAIN needs BUDGET=<target_norm chosen from the probe>" >&2
    exit 2
  fi
  local f d exp
  for f in f10 f20; do
    for d in "${DEFENSES[@]}"; do
      run_defended "${PREFIX}_backdoor_cifar100_${f}" "backdoor_cifar100_${f}" "$SEEDS" "$d"
    done
    exp="adaptive_backdoor_cifar100_${f}"
    for d in "${DEFENSES[@]}"; do
      run_defended "$(budget_tag "$exp" "$BUDGET")" "$exp" "$SEEDS" "$d" \
        "attack.update_manipulator.target_norm=${BUDGET}"
    done
  done
}

block_UNDEF() {
  local cell f g exp
  for cell in $UNDEF_CELLS; do
    f="${cell%%:*}"
    g="${cell##*:}"
    exp="adaptive_backdoor_cifar100_${f}"
    run_defended "$(budget_tag "$exp" "$g")" "$exp" "$SEEDS" none \
      "attack.update_manipulator.target_norm=${g}"
  done
}

block_DIAG() {
  if [ -z "$BUDGET" ]; then
    echo "DIAG needs BUDGET=<the adaptive attacker's target_norm>" >&2
    exit 2
  fi
  local f exp gtag=""
  [ "$BUDGET" = "$DEFAULT_BUDGET" ] || gtag="_g${BUDGET}"
  for f in f10 f20; do
    exp="adaptive_backdoor_cifar100_${f}"
    run_cell "diag_${exp}${gtag}" "$exp" "$SEEDS" \
      "attack.update_manipulator.target_norm=${BUDGET}" "$FLTRUST" +federation.trace_capture=true
    run_cell "diag_backdoor_cifar100_${f}" "backdoor_cifar100_${f}" "$SEEDS" \
      "$FLTRUST" +federation.trace_capture=true
  done
  exp="adaptive_backdoor_cifar100_f20"
  run_cell "diag_${exp}${gtag}" "$exp" "$SEEDS" \
    "attack.update_manipulator.target_norm=${BUDGET}" "+baseline=krum"
  run_cell "diag_backdoor_cifar100_f20" "backdoor_cifar100_f20" "$SEEDS" "+baseline=krum"
}

for block in $BLOCKS; do
  case "$block" in
    PROBE) block_PROBE ;;
    MAIN) block_MAIN ;;
    UNDEF) block_UNDEF ;;
    DIAG) block_DIAG ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "sweep_cifar100 done $(date -u +%FT%TZ)"
