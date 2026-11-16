#!/usr/bin/env bash
# New-baseline sweep: run one or more robust-aggregation baselines over the
# attack-by-compromise grid that the frozen defenses already cover, so the
# comparison table gains a column per baseline. Resumable.
#
# Every cell is a Hydra multirun over SEEDS with +skip_existing=true, tagged
# +tag=sp_<experiment>[_c<cosine>] so its runs never share a label directory
# with a frozen cell. Analyses pair cells by (label, config hash, seed).
#
# Baselines (select with BASELINES="flame mesas"): any name under
# configs/baseline/. Default: flame.
#
# Blocks (select with BLOCKS="MAIN FEMNIST"):
#   MAIN      CIFAR-10, f in {0.10, 0.20} x {backdoor, adaptive_backdoor
#             (gamma 0.6), clean_label, dirty_label, gradient_scaling,
#             sign_flip}                        12 cells x |SEEDS| per baseline
#   DIRALIGN  direction-aligned backdoor (fltrust_backdoor_cifar10_f10/f20)
#             at cosine_target in {0.5, 0.9}     4 cells x |SEEDS| per baseline
#   FEMNIST   FEMNIST, f in {0.10, 0.20} x {backdoor, gradient_scaling,
#             dirty_label}                       6 cells x |SEEDS| per baseline
#   FRACTION  CIFAR-10 compromise sweep, f in {0.05, 0.15, 0.30, 0.40} x
#             {backdoor, adaptive_backdoor}, FRACTION_SEEDS
#                                         8 cells x |FRACTION_SEEDS| per baseline
#   RESCAN    the norm-bounded adaptive attacker's budget rescan against each
#             baseline, so a new defense faces a tuned attacker like the
#             frozen ones did: target_norm in {1.0, 1.5} x f in {0.10, 0.20},
#             FRACTION_SEEDS, tagged sp_<experiment>_g<budget>
#                                         4 cells x |FRACTION_SEEDS| per baseline
#   BUMP      seed upgrades for knob values that would win a table cell once
#             they match the reference seed count, as listed by
#             `topfed.analysis.e3 --knob-bumps --defenses <labels>`; run RESCAN
#             (and MAIN) first. Off by default.
#   MESAS_MEAN (off by default; MESAS only) the split-centre sensitivity cell:
#             baseline.center=mean on CIFAR-10 {backdoor, adaptive_backdoor} x
#             f in {0.10, 0.20}, FRACTION_SEEDS, tagged sp_<experiment>_mean
#                                                   4 cells x |FRACTION_SEEDS|
#
# With the defaults (SEEDS 0-9, FRACTION_SEEDS 0-4) one baseline is 280 runs,
# 60 of them FEMNIST.
#
# Examples:
#   experiments/sweep_baselines.sh                                # flame, all blocks
#   BASELINES="flame mesas" BLOCKS="MAIN" experiments/sweep_baselines.sh
#   SEEDS="0" BLOCKS="MAIN" experiments/sweep_baselines.sh        # a quick look
#   DRY_RUN=1 experiments/sweep_baselines.sh                      # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASELINES="${BASELINES:-flame}"
BLOCKS="${BLOCKS:-MAIN DIRALIGN FEMNIST FRACTION RESCAN}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
FRACTION_SEEDS="${FRACTION_SEEDS:-0,1,2,3,4}"
DRY_RUN="${DRY_RUN:-0}"

MAIN_ATTACKS="backdoor adaptive_backdoor clean_label dirty_label gradient_scaling sign_flip"
FEMNIST_ATTACKS="backdoor gradient_scaling dirty_label"
FRACTION_F="f05 f15 f30 f40"

for b in $BASELINES; do
  if [ ! -f "configs/baseline/${b}.yaml" ]; then
    echo "unknown baseline: ${b} (no configs/baseline/${b}.yaml)" >&2
    exit 2
  fi
done

# run_cell <seeds> <tag> <base experiment> [override ...]
run_cell() {
  local seeds="$1" tag="$2" base="$3"
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

block_MAIN() {
  local b="$1"
  for f in f10 f20; do
    for attack in $MAIN_ATTACKS; do
      local exp="${attack}_cifar10_${f}"
      run_cell "$SEEDS" "sp_${exp}" "$exp" "+baseline=${b}"
    done
  done
}

block_DIRALIGN() {
  local b="$1"
  for f in f10 f20; do
    local exp="fltrust_backdoor_cifar10_${f}"
    for c in 0.5 0.9; do
      run_cell "$SEEDS" "sp_${exp}_c${c}" "$exp" \
        "attack.update_manipulator.cosine_target=${c}" "+baseline=${b}"
    done
  done
}

block_FEMNIST() {
  local b="$1"
  for f in f10 f20; do
    for attack in $FEMNIST_ATTACKS; do
      local exp="femnist_${attack}_${f}"
      run_cell "$SEEDS" "sp_${exp}" "$exp" "+baseline=${b}"
    done
  done
}

block_RESCAN() {
  local b="$1"
  for f in f10 f20; do
    local exp="adaptive_backdoor_cifar10_${f}"
    for g in 1.0 1.5; do
      run_cell "$FRACTION_SEEDS" "sp_${exp}_g${g}" "$exp" \
        "attack.update_manipulator.target_norm=${g}" "+baseline=${b}"
    done
  done
}

# The analysis label of a baseline config name (what --knob-bumps expects).
defense_label() {
  case "$1" in
    coordinate_median) echo "median" ;;
    fltrust_median) echo "fltrust-median" ;;
    backdoor_indicator) echo "indicator" ;;
    *) echo "$1" ;;
  esac
}

block_BUMP() {
  local b="$1" lines
  if ! lines="$(uv run python -m topfed.analysis.e3 --knob-bumps --defenses "$(defense_label "$b")" 2>/dev/null)"; then
    echo "knob-bump listing failed for ${b}" >&2
    return 1
  fi
  [ -z "$lines" ] && { echo "no knob bumps for ${b}"; return 0; }
  local tag exp seeds rest
  while IFS=$'\t' read -r tag exp seeds rest; do
    [ -z "$tag" ] && continue
    local -a overrides=()
    [ -n "$rest" ] && IFS=$'\t' read -r -a overrides <<< "$rest"
    run_cell "$seeds" "$tag" "$exp" ${overrides[@]+"${overrides[@]}"}
  done <<< "$lines"
}

block_MESAS_MEAN() {
  local b="$1"
  if [ "$b" != "mesas" ]; then
    echo "MESAS_MEAN applies to the mesas baseline only; skipping for ${b}"
    return 0
  fi
  for f in f10 f20; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_${f}"
      run_cell "$FRACTION_SEEDS" "sp_${exp}_mean" "$exp" "+baseline=${b}" "baseline.center=mean"
    done
  done
}

block_FRACTION() {
  local b="$1"
  for f in $FRACTION_F; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_${f}"
      run_cell "$FRACTION_SEEDS" "sp_${exp}" "$exp" "+baseline=${b}"
    done
  done
}

for b in $BASELINES; do
  echo "########### BASELINE ${b} ###########"
  for block in $BLOCKS; do
    case "$block" in
      MAIN|DIRALIGN|FEMNIST|FRACTION|RESCAN|BUMP|MESAS_MEAN) "block_${block}" "$b" ;;
      *) echo "unknown block: ${block}" >&2; exit 2 ;;
    esac
  done
done
echo "########### BASELINE SWEEP COMPLETE (${BASELINES}: ${BLOCKS}) ###########"
