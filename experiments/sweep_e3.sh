#!/usr/bin/env bash
# E3: systematic FLTrust comparison. Resumable sweep over the cells the
# attack-by-defense table is missing, the root-set sensitivity, and the
# server-overhead runs.
#
# Every cell is a Hydra multirun over SEEDS with +skip_existing=true, so a
# re-run after an interruption only executes the (config, seed) pairs that
# have no metrics.json yet. Each cell carries an explicit run tag
# (+tag=e3_<experiment>[_c<cosine>]) so its runs never share a label directory
# with a frozen cell. The one exception is the capped-wmed gradient_scaling
# f=0.20 cell, which is also the last cell of E2's optional UNTARGETED block:
# it runs under E2's tag so the two sweeps share one set of runs.
#
# Blocks (select with BLOCKS="FLTRUST MEDIAN"):
#   FLTRUST   FLTrust vs {gradient_scaling, sign_flip, dirty_label,
#             clean_label} x f in {0.10,0.20}, CIFAR-10               40 runs
#   CAPPED    capped-wmed at f=0.20 x the same 4 attacks (f=0.10
#             exists)                                                20 runs
#   MEDIAN    plain median at f=0.10 x the 4 attacks, and at f=0.20 x
#             {sign_flip, dirty_label, clean_label}                  35 runs
#   DIRALIGN  direction-aligned (FLTrust-tailored) backdoor at f=0.20 vs
#             {undefended, ScoreWeight, capped-wmed}, cosine in {0.5, 0.9}
#                                                                    30 runs
#   FEMNIST   FLTrust vs {backdoor, gradient_scaling, dirty_label} x
#             f in {0.10,0.20}                               30 FEMNIST runs
#   ROOTSET   val_per_class in {1, 10} (100 exists) x {vanilla, adaptive}
#             backdoor at f=0.10 x {FLTrust, capped-wmed}. The validation
#             set is FLTrust's root set and M4's reference, so both move 40 runs
#   OVERHEAD  seed 0, 50 rounds, {undefended, ScoreWeight, capped-wmed,
#             median, trimmed-mean, Krum, FLTrust} x {CIFAR-10, FEMNIST};
#             per-round timing columns                       14 short runs
#   MEMORY    the same 14 cells for 5 rounds with server peak-memory
#             profiling on (kept apart: host tracing slows the timings)
#                                                            14 short runs
#   FILL      (optional, off by default) cells the table shows as gaps that
#             not scheduled at first: median vs the direction-aligned
#             backdoor at f in {0.10,0.20} x cosine in {0.5,0.9} (20 CIFAR),
#             and {capped-wmed, median} vs FEMNIST {gradient_scaling,
#             dirty_label} x f in {0.10,0.20} (40 FEMNIST)
#
# OVERHEAD measures wall-clock time, so it must have the device to itself.
# It refuses to start while nvidia-smi reports another compute process;
# FORCE_OVERHEAD=1 overrides that check.
#
# Tables:
#   uv run python -m topfed.analysis.e3 --out tables/e3-tables.md
#
# Examples:
#   experiments/sweep_e3.sh                          # every block except FILL
#   BLOCKS="OVERHEAD MEMORY" experiments/sweep_e3.sh # overhead only, idle GPU
#   BLOCKS="FILL" experiments/sweep_e3.sh            # the optional block
#   SEEDS="0" BLOCKS="ROOTSET" experiments/sweep_e3.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-FLTRUST CAPPED MEDIAN DIRALIGN FEMNIST ROOTSET OVERHEAD MEMORY}"
SEEDS="${SEEDS:-0,1,2,3,4}"
FORCE_OVERHEAD="${FORCE_OVERHEAD:-0}"

UNTARGETED="gradient_scaling sign_flip dirty_label clean_label"
CAPPED="+defense=topsis_capped_weighted_median"
SCOREW="+defense=topsis_score_weight"
MEDIAN="+baseline=coordinate_median"
TRIMMED="+baseline=trimmed_mean"
KRUM="+baseline=krum"
FLTRUST="+baseline=fltrust"
# "none" is the undefended cell (no defense or baseline override).
OVERHEAD_DEFENSES="none ${SCOREW} ${CAPPED} ${MEDIAN} ${TRIMMED} ${KRUM} ${FLTRUST}"

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

block_FLTRUST() {
  # FLTrust has only ever been run on the backdoor families.
  for f in f10 f20; do
    for attack in $UNTARGETED; do
      local exp="${attack}_cifar10_${f}"
      run_cell "e3_${exp}" "$exp" "$FLTRUST"
    done
  done
}

block_CAPPED() {
  for attack in $UNTARGETED; do
    local exp="${attack}_cifar10_f20"
    local tag="e3_${exp}"
    # Shared with E2's UNTARGETED block (same config, same tag).
    [ "$attack" = "gradient_scaling" ] && tag="e2_${exp}"
    run_cell "$tag" "$exp" "$CAPPED"
  done
}

block_MEDIAN() {
  # gradient_scaling f=0.20 vs median already exists.
  for attack in $UNTARGETED; do
    run_cell "e3_${attack}_cifar10_f10" "${attack}_cifar10_f10" "$MEDIAN"
  done
  for attack in sign_flip dirty_label clean_label; do
    run_cell "e3_${attack}_cifar10_f20" "${attack}_cifar10_f20" "$MEDIAN"
  done
}

block_DIRALIGN() {
  # f=0.10 has these three defenses at cosine 0.5 / 0.9 (n=5) and FLTrust at
  # both levels; f=0.20 has FLTrust only.
  local exp="fltrust_backdoor_cifar10_f20"
  for c in 0.5 0.9; do
    for d in none "$SCOREW" "$CAPPED"; do
      run_defended "e3_${exp}_c${c}" "$exp" "$d" "attack.update_manipulator.cosine_target=${c}"
    done
  done
}

block_FEMNIST() {
  for f in f10 f20; do
    for exp in "femnist_backdoor_${f}" "femnist_gradient_scaling_${f}" "femnist_dirty_label_${f}"; do
      run_cell "e3_${exp}" "$exp" "$FLTRUST"
    done
  done
}

block_ROOTSET() {
  # 100 per class is the frozen n=10 grid; undefended does not read the
  # validation set, so it has no root-set cell.
  for v in 1 10; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_f10_root${v}"
      for d in "$FLTRUST" "$CAPPED"; do
        run_cell "e3_${exp}" "$exp" "$d"
      done
    done
  done
}

device_is_busy() {
  command -v nvidia-smi > /dev/null 2>&1 || return 1
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}

block_OVERHEAD() {
  local SEEDS="0"
  for ds in cifar10 femnist; do
    local exp="overhead_${ds}"
    for d in $OVERHEAD_DEFENSES; do
      if [ "$FORCE_OVERHEAD" != "1" ] && device_is_busy; then
        echo "OVERHEAD: another process is using the GPU; skipping ${exp} [${d}]" \
          "(re-run on an idle device, or FORCE_OVERHEAD=1)" >&2
        continue
      fi
      run_defended "e3_${exp}" "$exp" "$d"
    done
  done
}

block_MEMORY() {
  local SEEDS="0"
  for ds in cifar10 femnist; do
    local exp="overhead_memory_${ds}"
    for d in $OVERHEAD_DEFENSES; do
      run_defended "e3_${exp}" "$exp" "$d"
    done
  done
}

block_FILL() {
  for f in f10 f20; do
    local exp="fltrust_backdoor_cifar10_${f}"
    for c in 0.5 0.9; do
      run_cell "e3_${exp}_c${c}" "$exp" "$MEDIAN" "attack.update_manipulator.cosine_target=${c}"
    done
    for attack in gradient_scaling dirty_label; do
      local fexp="femnist_${attack}_${f}"
      run_cell "e3_${fexp}" "$fexp" "$CAPPED"
      run_cell "e3_${fexp}" "$fexp" "$MEDIAN"
    done
  done
}

for block in $BLOCKS; do
  case "$block" in
    FLTRUST|CAPPED|MEDIAN|DIRALIGN|FEMNIST|ROOTSET|OVERHEAD|MEMORY|FILL) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### E3 SWEEP COMPLETE (${BLOCKS}) ###########"
