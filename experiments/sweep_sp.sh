#!/usr/bin/env bash
# Paper sweep: the cells the comparison tables still miss, and the seed
# upgrades that bring paper-facing n=5 cells to n=10. Resumable.
#
# Every cell is a Hydra multirun with +skip_existing=true. Cells that extend
# an existing tagged cell reuse its tag (so the new seeds land in the same
# directory and the old ones are skipped); cells that are new carry
# +tag=sp_<experiment>[...]. The analyses pair runs by resolved config and
# seed, not by directory, so a cell split across tags is still one cell.
#
# Blocks (select with BLOCKS="H3 UPGRADE"):
#   H3        FLTrust-median (H3) to n=10 on every cell E4 defined for it
#             (headline, regression, FEMNIST; tags from topfed.analysis.e4),
#             plus the composed-attack reference defenses (FLTrust,
#             capped-wmed), plus new H3 cells: CIFAR f=0.20 untargeted
#             attacks, FEMNIST gradient scaling and dirty label, the
#             direction-aligned c=0.5 and adaptive gamma=0.6 and composed
#             c=0.5 knob values that E4 rescanned on one seed.
#   H3_ROOT   FLTrust-median root-set sensitivity: val_per_class 1 and 10 x
#             {backdoor, adaptive} at f=0.10 (E3's ROOTSET tags), FIVE_SEEDS.
#   UPGRADE   Seeds UPGRADE_SEEDS (default 5-9) for the n=5 cells in the main
#             tables: CIFAR untargeted x {undefended, ScoreWeight, capped-wmed,
#             median, FLTrust} x f10/f20; direction-aligned x {undefended,
#             ScoreWeight, capped-wmed} x c {0.5, 0.9} x f10/f20; FEMNIST
#             backdoor x {undefended, ScoreWeight, wmed, capped-wmed, median}
#             x f10/f20; FEMNIST FLTrust x {backdoor, gradient scaling, dirty
#             label} x f10/f20; FEMNIST untargeted x {undefended, ScoreWeight}.
#   FILL      Cells that do not exist at all: median vs direction-aligned
#             (c 0.5/0.9, f10/f20); {capped-wmed, median} vs FEMNIST
#             {gradient scaling, dirty label} x f10/f20. SEEDS.
#   BUMP      Seed upgrades for knob values that would win a table cell once
#             they match the reference seed count, for the frozen defenses
#             (undefended, ScoreWeight, capped-wmed, median, FLTrust,
#             FLTrust-median), as listed by `topfed.analysis.e3 --knob-bumps`.
#             Run last: the listing is computed from results/.
#
# The new published baselines and the classic ones (krum, trimmed_mean,
# foolsgold) run through experiments/sweep_baselines.sh, not here.
#
# Examples:
#   experiments/sweep_sp.sh                              # all blocks
#   BLOCKS="H3 H3_ROOT" experiments/sweep_sp.sh
#   DRY_RUN=1 experiments/sweep_sp.sh                    # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-H3 H3_ROOT UPGRADE FILL BUMP}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
FIVE_SEEDS="${FIVE_SEEDS:-0,1,2,3,4}"
UPGRADE_SEEDS="${UPGRADE_SEEDS:-5,6,7,8,9}"
DRY_RUN="${DRY_RUN:-0}"

CAPPED="+defense=topsis_capped_weighted_median"
SCOREW="+defense=topsis_score_weight"
WMED="+defense=topsis_weighted_median"
MEDIAN="+baseline=coordinate_median"
FLTRUST="+baseline=fltrust"
H3="+baseline=fltrust_median"
UNTARGETED="clean_label dirty_label gradient_scaling sign_flip"
FEMNIST_UNTARGETED="gradient_scaling dirty_label"

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
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

# run_defended <tag or ""> <base> <seeds> <defense override or "none"> [override ...]
run_defended() {
  local tag="$1" base="$2" seeds="$3" defense="$4"
  shift 4
  if [ "$defense" = "none" ]; then
    run_cell "$tag" "$base" "$seeds" "$@"
  else
    run_cell "$tag" "$base" "$seeds" "$@" "$defense"
  fi
}

# run_e4_cells <e4 block> <variants> <seeds>: the cells topfed.analysis.e4 lists.
run_e4_cells() {
  local block="$1" variants="$2" seeds="$3" cells
  if ! cells="$(uv run python -m topfed.analysis.e4 --cells "$block" --variants "$variants" 2>/dev/null)"; then
    echo "could not list E4 cells for ${block} (${variants})" >&2
    return 1
  fi
  local tag exp seedset rest
  while IFS=$'\t' read -r tag exp seedset rest; do
    [ -z "$tag" ] && continue
    local -a overrides=()
    [ -n "$rest" ] && IFS=$'\t' read -r -a overrides <<< "$rest"
    run_cell "$tag" "$exp" "$seeds" ${overrides[@]+"${overrides[@]}"}
  done <<< "$cells"
}

block_H3() {
  run_e4_cells headline H3 "$SEEDS"
  run_e4_cells regression H3 "$SEEDS"
  run_e4_cells femnist H3 "$SEEDS"
  run_e4_cells refs H1 "$SEEDS"
  # Knob values E4 rescanned on one seed, to n=10 under E4's tags.
  for f in f10 f20; do
    run_cell "e4_adaptive_backdoor_cifar10_${f}_g0.6" "adaptive_backdoor_cifar10_${f}" "$SEEDS" \
      "attack.update_manipulator.target_norm=0.6" "$H3"
    run_cell "e4_fltrust_backdoor_cifar10_${f}_c0.5" "fltrust_backdoor_cifar10_${f}" "$SEEDS" \
      "attack.update_manipulator.cosine_target=0.5" "$H3"
    run_cell "e4_composed_backdoor_cifar10_${f}_g0.6_c0.5" "composed_backdoor_cifar10_${f}" "$SEEDS" \
      "attack.update_manipulator.manipulators.0.cosine_target=0.5" \
      "attack.update_manipulator.manipulators.1.target_norm=0.6" "$H3"
  done
  # New H3 cells.
  for attack in $UNTARGETED; do
    run_cell "sp_${attack}_cifar10_f20" "${attack}_cifar10_f20" "$SEEDS" "$H3"
  done
  for f in f10 f20; do
    for attack in $FEMNIST_UNTARGETED; do
      run_cell "sp_femnist_${attack}_${f}" "femnist_${attack}_${f}" "$SEEDS" "$H3"
    done
  done
}

block_H3_ROOT() {
  for v in 1 10; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_f10_root${v}"
      run_cell "e3_${exp}" "$exp" "$FIVE_SEEDS" "$H3"
    done
  done
}

block_UPGRADE() {
  local s="$UPGRADE_SEEDS"
  for f in f10 f20; do
    for attack in $UNTARGETED; do
      local exp="${attack}_cifar10_${f}"
      for d in none "$SCOREW" "$CAPPED" "$MEDIAN" "$FLTRUST"; do
        run_defended "sp_${exp}" "$exp" "$s" "$d"
      done
    done
    local da="fltrust_backdoor_cifar10_${f}"
    for c in 0.5 0.9; do
      for d in none "$SCOREW" "$CAPPED"; do
        run_defended "sp_${da}_c${c}" "$da" "$s" "$d" "attack.update_manipulator.cosine_target=${c}"
      done
    done
    local fb="femnist_backdoor_${f}"
    for d in none "$SCOREW" "$WMED" "$CAPPED" "$MEDIAN" "$FLTRUST"; do
      run_defended "sp_${fb}" "$fb" "$s" "$d"
    done
    for attack in $FEMNIST_UNTARGETED; do
      local fe="femnist_${attack}_${f}"
      for d in none "$SCOREW" "$FLTRUST"; do
        run_defended "sp_${fe}" "$fe" "$s" "$d"
      done
    done
  done
}

block_FILL() {
  for f in f10 f20; do
    local da="fltrust_backdoor_cifar10_${f}"
    for c in 0.5 0.9; do
      run_cell "sp_${da}_c${c}" "$da" "$SEEDS" "attack.update_manipulator.cosine_target=${c}" "$MEDIAN"
    done
    for attack in $FEMNIST_UNTARGETED; do
      local fe="femnist_${attack}_${f}"
      for d in "$CAPPED" "$MEDIAN"; do
        run_cell "sp_${fe}" "$fe" "$SEEDS" "$d"
      done
    done
  done
}

block_BUMP() {
  local lines
  if ! lines="$(uv run python -m topfed.analysis.e3 --knob-bumps \
      --defenses undefended,ScoreWeight,capped-wmed,median,fltrust,fltrust-median 2>/dev/null)"; then
    echo "knob-bump listing failed" >&2
    return 1
  fi
  [ -z "$lines" ] && { echo "no knob bumps"; return 0; }
  local tag exp seeds rest
  while IFS=$'\t' read -r tag exp seeds rest; do
    [ -z "$tag" ] && continue
    local -a overrides=()
    [ -n "$rest" ] && IFS=$'\t' read -r -a overrides <<< "$rest"
    run_cell "$tag" "$exp" "$seeds" ${overrides[@]+"${overrides[@]}"}
  done <<< "$lines"
}

for block in $BLOCKS; do
  case "$block" in
    H3|H3_ROOT|UPGRADE|FILL|BUMP) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### SP SWEEP COMPLETE (${BLOCKS}) ###########"
