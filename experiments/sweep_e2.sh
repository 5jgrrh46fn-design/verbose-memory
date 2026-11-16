#!/usr/bin/env bash
# E2: compromise-rate sensitivity. Resumable sweep over new compromise levels.
#
# Every cell is a Hydra multirun over SEEDS with +skip_existing=true, so a
# re-run after an interruption only executes the (config, seed) pairs that
# have no metrics.json yet. Each cell carries an explicit run tag
# (+tag=e2_<experiment>[_g<budget>]) so its runs never share a label directory
# with the frozen f=0.10 / f=0.20 cells; those are read from their canonical
# directories by the analysis, not re-run here.
#
# Blocks (select with BLOCKS="MAIN ORACLE"):
#   MAIN      CIFAR-10, f in {0.05,0.15,0.25,0.30,0.40} x {vanilla backdoor,
#             adaptive budget 0.6} x {undefended, ScoreWeight, capped-wmed,
#             median, FLTrust}                                      250 runs
#   GAMMA     adaptive budget rescan {0.6,1.0,1.5} at f in {0.30,0.40} vs
#             capped-wmed and ScoreWeight, seed 0; the 0.6 cells are the MAIN
#             seed-0 runs, so 8 of the 12 cells are new               8 runs
#   ORACLE    ground-truth weights on the weighted-median base, both attacks,
#             every f including 0.10 / 0.20 (the frozen grid has no oracle
#             cell there)                                            70 runs
#   FEMNIST   f in {0.05,0.15} x {undefended, wmed, capped-wmed, median,
#             ScoreWeight}                                           50 runs
#   BUMP      seeds 5-9 at the breaking f (first f where mean ASR >= 0.5) for
#             capped-wmed and FLTrust, both attacks; read from the analysis
#             after MAIN, so it runs last                         <= 20 runs
#   UNTARGETED (optional, off by default) gradient_scaling at the new f x
#             {undefended, capped-wmed}, plus the capped-wmed f=0.20 cell the
#             frozen grid lacks                                      55 runs
#
# Tables and figures:
#   uv run python -m topfed.analysis.e2 --out tables/e2-tables.md \
#       --fig-dir figures
#
# Examples:
#   experiments/sweep_e2.sh                          # MAIN ORACLE GAMMA FEMNIST BUMP
#   BLOCKS="FEMNIST" experiments/sweep_e2.sh         # one block
#   BLOCKS="UNTARGETED" experiments/sweep_e2.sh      # the optional block
#   SEEDS="0" BLOCKS="ORACLE" experiments/sweep_e2.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-MAIN ORACLE GAMMA FEMNIST BUMP}"
SEEDS="${SEEDS:-0,1,2,3,4}"
BUMP_SEEDS="${BUMP_SEEDS:-5,6,7,8,9}"

NEW_F="f05 f15 f25 f30 f40"
ALL_F="f05 f10 f15 f20 f25 f30 f40"
CAPPED="+defense=topsis_capped_weighted_median"
SCOREW="+defense=topsis_score_weight"
WMED="+defense=topsis_weighted_median"
MEDIAN="+baseline=coordinate_median"
FLTRUST="+baseline=fltrust"
ORACLE="+defense=oracle_filter_wmedian"
# "none" is the undefended cell (no defense or baseline override).
CIFAR_DEFENSES="none ${SCOREW} ${CAPPED} ${MEDIAN} ${FLTRUST}"
FEMNIST_DEFENSES="none ${WMED} ${CAPPED} ${MEDIAN} ${SCOREW}"

# run_cell <tag> <base experiment> [override ...]
# The tag becomes the run label prefix: results/<tag>[__<defense>]/<hash>/seed_*/
run_cell() {
  local tag="$1" base="$2"
  shift 2
  echo "=========== ${tag} [$*] seeds=${SEEDS} $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" \
    federation.seed="$SEEDS" +skip_existing=true "$@" \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

# run_defended <tag> <base experiment> <defense override or "none"> [override ...]
run_defended() {
  local tag="$1" base="$2" defense="$3"
  shift 3
  if [ "$defense" = "none" ]; then
    run_cell "$tag" "$base" "$@"
  else
    run_cell "$tag" "$base" "$@" "$defense"
  fi
}

block_MAIN() {
  # Where each defense breaks as f rises, for the scaled and the adaptive
  # backdoor. f=0.10 / 0.20 come from the frozen n=10 cells.
  for f in $NEW_F; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_${f}"
      for d in $CIFAR_DEFENSES; do
        run_defended "e2_${exp}" "$exp" "$d"
      done
    done
  done
}

block_GAMMA() {
  # The 0.6 budget was calibrated at f=0.10 and transferred to f=0.20 without
  # a rescan. At the two highest levels, rescan it against the two trust-scored
  # defenses; the analysis reports the max over budgets. Budget 0.6 reuses the
  # MAIN tag (same config, so the MAIN seed-0 run is skipped, or MAIN skips it).
  local SEEDS="0"
  for f in f30 f40; do
    local exp="adaptive_backdoor_cifar10_${f}"
    for d in $CAPPED $SCOREW; do
      run_cell "e2_${exp}" "$exp" "$d"
      for g in 1.0 1.5; do
        run_cell "e2_${exp}_g${g}" "$exp" "attack.update_manipulator.target_norm=${g}" "$d"
      done
    done
  done
}

block_ORACLE() {
  # Detection-vs-aggregation split. If oracle-wmed fails at some f, the
  # weighted-median base is the limit there regardless of detection; if the
  # oracle holds and capped-wmed fails, detection is the limit.
  for f in $ALL_F; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_${f}"
      run_cell "e2_${exp}" "$exp" "$ORACLE"
    done
  done
}

block_FEMNIST() {
  # Locate the FEMNIST cliff between f=0.10 (capped-wmed holds) and f=0.20
  # (every defense fails).
  for f in f05 f15; do
    local exp="femnist_backdoor_${f}"
    for d in $FEMNIST_DEFENSES; do
      run_defended "e2_${exp}" "$exp" "$d"
    done
  done
}

block_BUMP() {
  # Five more seeds at each (attack, defense)'s breaking f. The analysis
  # prints one "<experiment> <override>" line per cell that needs them and
  # nothing when the break is at a frozen n=10 level or never happens.
  local lines
  if ! lines="$(uv run python -m topfed.analysis.e2 --breaking-f)"; then
    echo "breaking-f analysis failed; run MAIN first" >&2
    return 1
  fi
  if [ -z "$lines" ]; then
    echo "=========== BUMP: no new-level breaking f to extend ==========="
    return 0
  fi
  local SEEDS="$BUMP_SEEDS"
  while read -r exp override; do
    run_cell "e2_${exp}" "$exp" "$override"
  done <<< "$lines"
}

block_UNTARGETED() {
  # Optional. Shows whether the breakdown is specific to the backdoor.
  for f in $NEW_F; do
    local exp="gradient_scaling_cifar10_${f}"
    run_defended "e2_${exp}" "$exp" none
    run_defended "e2_${exp}" "$exp" "$CAPPED"
  done
  run_cell "e2_gradient_scaling_cifar10_f20" "gradient_scaling_cifar10_f20" "$CAPPED"
}

for block in $BLOCKS; do
  case "$block" in
    MAIN|GAMMA|ORACLE|FEMNIST|BUMP|UNTARGETED) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### E2 SWEEP COMPLETE (${BLOCKS}) ###########"
