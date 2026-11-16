#!/usr/bin/env bash
# E4: bounded influence plus a server-rooted signal. Resumable sweep over the
# four hybrid variants against the vanilla, norm-bounded, direction-aligned and
# composed backdoors, with best-response rescans first.
#
# Variants: H1 server-gated capped-wmed, H2 server cosine as a sixth TOPSIS
# criterion, H3 FLTrust on a weighted-median base, H4 the M4-band control.
#
# The cell lists come from `python -m topfed.analysis.e4 --cells <block>`, the
# same code that scores the results, so the sweep and the report agree on the
# grid, the tags and the best-response knob. Every cell is a Hydra multirun
# with +skip_existing=true and an explicit run tag
# (+tag=e4_<experiment>[_g<budget>][_c<cosine>]), so a re-run only executes
# missing (config, seed) pairs and no E4 run shares a label with a frozen cell.
#
# Blocks (select with BLOCKS="RESCAN_A HEADLINE_A"):
#   Stage A (run first):
#   RESCAN_A    H1, H4: seed 0 at every best-response grid point (NB gamma
#               {0.6,1.0}; DA c {0.9,0.5}; composed gamma x c) x f10/f20
#                                                                32 runs
#   HEADLINE_A  H1, H4 x {vanilla, NB, DA, composed} x f10/f20, n=5, adaptive
#               attacks at each variant's seed-0 worst point, plus undefended
#               at every such config that lacks 5 seeds         80 + refs
#   Full (after Stage A):
#   RESCAN_B    H2, H3, as RESCAN_A                              32 runs
#   HEADLINE_B  H2, H3, as HEADLINE_A                            80 + refs
#   REGRESSION  H1-H3 x {gradient_scaling, sign_flip, dirty_label,
#               clean_label} x f10                               60 runs
#   FEMNIST     H1-H3 x FEMNIST backdoor x f10/f20       30 FEMNIST runs
#   Optional (off by default):
#   REFS        FLTrust and capped-wmed vs the composed attack (default
#               point) x f10/f20, plus undefended                20-30 runs
#
# BLOCKS aliases: STAGE_A = "RESCAN_A HEADLINE_A" (the default); FULL = Stage A
# followed by the full blocks (Stage A cells already done are skipped).
# HEADLINE_* must run after the matching RESCAN_*: a headline cell whose
# rescan is incomplete is not scheduled (the analysis says so on stderr).
#
# Tables:
#   uv run python -m topfed.analysis.e4 --out tables/e4-tables.md
#
# Examples:
#   experiments/sweep_e4.sh                            # Stage A
#   BLOCKS="FULL" experiments/sweep_e4.sh              # everything
#   BLOCKS="REFS" experiments/sweep_e4.sh              # the optional block
#   SEEDS="0,1" BLOCKS="HEADLINE_A" experiments/sweep_e4.sh
#   DRY_RUN=1 BLOCKS="FULL REFS" experiments/sweep_e4.sh  # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-STAGE_A}"
SEEDS="${SEEDS:-0,1,2,3,4}"
RESCAN_SEEDS="${RESCAN_SEEDS:-0}"
DRY_RUN="${DRY_RUN:-0}"

BLOCKS="${BLOCKS//FULL/RESCAN_A HEADLINE_A RESCAN_B HEADLINE_B REGRESSION FEMNIST}"
BLOCKS="${BLOCKS//STAGE_A/RESCAN_A HEADLINE_A}"

# run_cell <tag> <base experiment> <seeds> [override ...]
run_cell() {
  local tag="$1" base="$2" seeds="$3"
  shift 3
  echo "=========== ${tag} [$*] seeds=${seeds} $(date -u +%FT%TZ) ==========="
  [ "$DRY_RUN" = "1" ] && return 0
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" \
    federation.seed="$seeds" +skip_existing=true "$@" \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

# run_block <analysis block> <variants>
run_block() {
  local block="$1" variants="$2" cells
  if ! cells="$(uv run python -m topfed.analysis.e4 --cells "$block" --variants "$variants")"; then
    echo "E4: could not list cells for ${block} (${variants})" >&2
    return 1
  fi
  local tag exp seeds rest seed_arg
  while IFS=$'\t' read -r tag exp seeds rest; do
    [ -z "$tag" ] && continue
    case "$seeds" in
      rescan) seed_arg="$RESCAN_SEEDS" ;;
      all) seed_arg="$SEEDS" ;;
      *) echo "E4: unknown seed set ${seeds}" >&2; return 1 ;;
    esac
    local -a overrides=()
    [ -n "$rest" ] && IFS=$'\t' read -r -a overrides <<< "$rest"
    run_cell "$tag" "$exp" "$seed_arg" ${overrides[@]+"${overrides[@]}"}
  done <<< "$cells"
}

for block in $BLOCKS; do
  case "$block" in
    RESCAN_A) run_block rescan H1,H4 ;;
    HEADLINE_A) run_block headline H1,H4 ;;
    RESCAN_B) run_block rescan H2,H3 ;;
    HEADLINE_B) run_block headline H2,H3 ;;
    REGRESSION) run_block regression H1,H2,H3 ;;
    FEMNIST) run_block femnist H1,H2,H3 ;;
    REFS) run_block refs H1 ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### E4 SWEEP COMPLETE (${BLOCKS}) ###########"
