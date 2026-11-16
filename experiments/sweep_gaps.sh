#!/usr/bin/env bash
# Gap sweep for the paper draft: the capped FLTrust-median variant, the
# server-overhead rows the comparison table lacks, and a FLAME sanity cell in
# FLAME's own participation regime. Resumable (+skip_existing).
#
# Blocks (select with BLOCKS="MWROOT_CAP OVERHEAD"):
#   MWROOT_CAP  +baseline=fltrust_median_capped on every cell the uncapped
#               fltrust_median has: CIFAR-10 f10/f20 x {backdoor, adaptive at
#               gamma 0.6 and 1.0, direction-aligned at c 0.5 and 0.9,
#               composed (0.6, 0.9), clean_label, dirty_label,
#               gradient_scaling, sign_flip}, SEEDS; FEMNIST f10/f20 x
#               {backdoor, gradient_scaling, dirty_label}, SEEDS; root-set
#               cells (val_per_class 1 and 10 at f10, backdoor and adaptive),
#               FIVE_SEEDS. Tags mirror the uncapped cells' tags (e4_/e3_/sp_)
#               so the pairs sit side by side.            about 300 runs
#   OVERHEAD    50-round timing runs (overhead_cifar10, overhead_femnist) for
#               flame, mesas, fltrust_median, fltrust_median_capped, and
#               backdoor_indicator (CIFAR only), seed 0, tag e3_overhead_<ds>.
#               Needs the device to itself; refuses to start while nvidia-smi
#               shows another compute process unless FORCE_OVERHEAD=1.
#   FA_PROBE    FEMNIST adaptive backdoor, undefended, seed 0, target_norm in
#               FA_BUDGETS (default 0.3 0.6 1.0 1.5 3.0) at f10: pick the
#               smallest budget whose last-50-round ASR reaches 0.5. 5 runs
#   FA_MAIN     femnist_adaptive_backdoor_f10/f20 at FA_BUDGET x {undefended,
#               capped-wmed, fltrust_median_capped, fltrust, median, flame},
#               SEEDS. Tags sp_femnist_adaptive_backdoor_<f>[_g<budget>]; the
#               budget suffix is omitted when FA_BUDGET is the CIFAR default
#               0.6.                                             120 runs
#   FLAME_FULL  FLAME in its own regime: every client participates each round
#               (federation.clients_per_round=100) on backdoor_cifar10_f20,
#               undefended and +baseline=flame, seed 0, tag sp_flame_full.
#               Each run is about ten times a normal run.        2 long runs
#
# Examples:
#   experiments/sweep_gaps.sh                       # all blocks
#   BLOCKS=MWROOT_CAP experiments/sweep_gaps.sh
#   DRY_RUN=1 experiments/sweep_gaps.sh             # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-MWROOT_CAP OVERHEAD FLAME_FULL}"
FA_BUDGETS="${FA_BUDGETS:-0.3 0.6 1.0 1.5 3.0}"
FA_BUDGET="${FA_BUDGET:-0.6}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
FIVE_SEEDS="${FIVE_SEEDS:-0,1,2,3,4}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_OVERHEAD="${FORCE_OVERHEAD:-0}"

CAPPED_ROOT="+baseline=fltrust_median_capped"
UNTARGETED="clean_label dirty_label gradient_scaling sign_flip"
FEMNIST_ATTACKS="backdoor gradient_scaling dirty_label"

# run_cell <tag or ""> <base experiment> <seeds> [override ...]
run_cell() {
  local tag="$1" base="$2" seeds="$3"
  shift 3
  local -a tag_arg=()
  [ -n "$tag" ] && tag_arg=("+tag=${tag}")
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY ${tag:-<no tag>} ${base} [$*] seeds=${seeds}"
    return 0
  fi
  echo "=========== ${tag:-${base}} [$*] seeds=${seeds} $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" ${tag_arg[@]+"${tag_arg[@]}"} \
    federation.seed="$seeds" +skip_existing=true "$@" \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|round 50/|Error|Traceback" | tail -10
}

device_is_busy() {
  command -v nvidia-smi > /dev/null 2>&1 || return 1
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}

block_MWROOT_CAP() {
  for f in f10 f20; do
    run_cell "e4_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$SEEDS" "$CAPPED_ROOT"
    for g in 0.6 1.0; do
      run_cell "e4_adaptive_backdoor_cifar10_${f}_g${g}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" \
        "attack.update_manipulator.target_norm=${g}" "$CAPPED_ROOT"
    done
    for c in 0.5 0.9; do
      run_cell "e4_fltrust_backdoor_cifar10_${f}_c${c}" "fltrust_backdoor_cifar10_${f}" "$SEEDS" \
        "attack.update_manipulator.cosine_target=${c}" "$CAPPED_ROOT"
    done
    run_cell "e4_composed_backdoor_cifar10_${f}_g0.6_c0.9" "composed_backdoor_cifar10_${f}" "$SEEDS" \
      "attack.update_manipulator.manipulators.0.cosine_target=0.9" \
      "attack.update_manipulator.manipulators.1.target_norm=0.6" "$CAPPED_ROOT"
    for attack in $UNTARGETED; do
      local exp="${attack}_cifar10_${f}"
      if [ "$f" = "f10" ]; then
        run_cell "e4_${exp}" "$exp" "$SEEDS" "$CAPPED_ROOT"
      else
        run_cell "sp_${exp}" "$exp" "$SEEDS" "$CAPPED_ROOT"
      fi
    done
    for attack in $FEMNIST_ATTACKS; do
      local fe="femnist_${attack}_${f}"
      if [ "$attack" = "backdoor" ]; then
        run_cell "e4_${fe}" "$fe" "$SEEDS" "$CAPPED_ROOT"
      else
        run_cell "sp_${fe}" "$fe" "$SEEDS" "$CAPPED_ROOT"
      fi
    done
  done
  for v in 1 10; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_f10_root${v}"
      run_cell "e3_${exp}" "$exp" "$FIVE_SEEDS" "$CAPPED_ROOT"
    done
  done
}

block_OVERHEAD() {
  local -a cifar=("+baseline=flame" "+baseline=mesas" "+baseline=fltrust_median" "+baseline=fltrust_median_capped" "+baseline=backdoor_indicator")
  local -a femnist=("+baseline=flame" "+baseline=mesas" "+baseline=fltrust_median" "+baseline=fltrust_median_capped")
  local d
  for d in "${cifar[@]}"; do
    if [ "$FORCE_OVERHEAD" != "1" ] && device_is_busy; then
      echo "OVERHEAD: another process is using the GPU; skipping overhead_cifar10 [${d}]" >&2
      continue
    fi
    run_cell "e3_overhead_cifar10" "overhead_cifar10" "0" "$d"
  done
  for d in "${femnist[@]}"; do
    if [ "$FORCE_OVERHEAD" != "1" ] && device_is_busy; then
      echo "OVERHEAD: another process is using the GPU; skipping overhead_femnist [${d}]" >&2
      continue
    fi
    run_cell "e3_overhead_femnist" "overhead_femnist" "0" "$d"
  done
}

fa_tag() {
  local exp="$1" budget="$2"
  if [ "$budget" = "0.6" ]; then echo "sp_${exp}"; else echo "sp_${exp}_g${budget}"; fi
}

block_FA_PROBE() {
  local exp="femnist_adaptive_backdoor_f10"
  for g in $FA_BUDGETS; do
    run_cell "$(fa_tag "$exp" "$g")" "$exp" "0" "attack.update_manipulator.target_norm=${g}"
  done
}

block_FA_MAIN() {
  local -a defenses=(none "+defense=topsis_capped_weighted_median" "+baseline=fltrust_median_capped" "+baseline=fltrust" "+baseline=coordinate_median" "+baseline=flame")
  local d
  for f in f10 f20; do
    local exp="femnist_adaptive_backdoor_${f}"
    for d in "${defenses[@]}"; do
      if [ "$d" = "none" ]; then
        run_cell "$(fa_tag "$exp" "$FA_BUDGET")" "$exp" "$SEEDS" "attack.update_manipulator.target_norm=${FA_BUDGET}"
      else
        run_cell "$(fa_tag "$exp" "$FA_BUDGET")" "$exp" "$SEEDS" "attack.update_manipulator.target_norm=${FA_BUDGET}" "$d"
      fi
    done
  done
}

block_FLAME_FULL() {
  run_cell "sp_flame_full" "backdoor_cifar10_f20" "0" "federation.clients_per_round=100"
  run_cell "sp_flame_full" "backdoor_cifar10_f20" "0" "federation.clients_per_round=100" "+baseline=flame"
}

for block in $BLOCKS; do
  case "$block" in
    MWROOT_CAP|OVERHEAD|FLAME_FULL|FA_PROBE|FA_MAIN) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### GAP SWEEP COMPLETE (${BLOCKS}) ###########"
