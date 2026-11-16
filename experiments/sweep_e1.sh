#!/usr/bin/env bash
# E1: colluding attackers vs the peer-relative metrics and capped-wmed.
# Resumable sweep.
#
# Every cell is CIFAR-10, Dirichlet(0.5), N=100 / K=10, population f=0.30,
# with exactly m attackers forced into every round's cohort and the capture
# trace on (experiment configs collusion_{backdoor,adaptive_backdoor}_cifar10_f30).
# Each cell is a Hydra multirun over SEEDS with +skip_existing=true, tagged
# +tag=e1_<experiment>_<strategy>_m<m>. The tag carries no stage, so a Stage B
# cell that repeats a Stage A config reuses its seeds 0-2 for free.
#
# Strategies: independent (no coordination, the control), clone (S1),
# budget_split (S2), reference_shift (S3), coordinate_capture (S4). At m=1
# every strategy is the independent attacker, so m=1 runs once, as the
# control.
#
# Blocks (select with BLOCKS="A_CONTROL A_COLLUDE"):
#   A_CONTROL  independent attackers, m in {1..5} x 2 attacks x 4 defenses
#              (undefended, ScoreWeight, capped-wmed, median)        120 runs
#   A_COLLUDE  S1 and S3, m in {2..5} x 2 attacks x 4 defenses      192 runs
#   B          Stage B, n=5: STAGE_B_M x STAGE_B_STRATEGIES x 2 attacks x
#              6 defenses (Stage A's four plus FLTrust and FoolsGold).
#              S2 runs with the adaptive attack only (under round-aware
#              model replacement it submits the same update as S1).
#              Seeds 0-2 of the four Stage A defenses are reused.
#   B_AGREE    (optional, off by default) the Stage B grid against capped-wmed
#              with the sixth TOPSIS criterion, pairwise history agreement,
#              n=5; the same STAGE_B_M / STAGE_B_STRATEGIES as B.
#
# Tables and figures:
#   uv run python -m topfed.analysis.e1 --out tables/e1-tables.md \
#       --fig-dir figures
# Stage B suggestion (breaking m and neighbours per strategy and attack):
#   uv run python -m topfed.analysis.e1 --stage-b
#
# Examples:
#   experiments/sweep_e1.sh                                 # Stage A (both blocks)
#   BLOCKS="A_CONTROL" SEEDS="0" experiments/sweep_e1.sh    # a quick look
#   STAGE_B_M="3 4 5" STAGE_B_STRATEGIES="reference_shift budget_split" \
#       BLOCKS="B" experiments/sweep_e1.sh
#   DRY_RUN=1 experiments/sweep_e1.sh                       # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-A_CONTROL A_COLLUDE}"
SEEDS="${SEEDS:-}"
DRY_RUN="${DRY_RUN:-0}"

# TODO: set after Stage A. STAGE_B_M is the breaking m (first m whose mean
# converged ASR reaches 0.5 for the defense of interest) and its two
# neighbours, e.g. "3 4 5"; STAGE_B_STRATEGIES is the worst strategy from
# Stage A, plus budget_split / coordinate_capture if Stage A shows they matter.
# `uv run python -m topfed.analysis.e1 --stage-b` prints a suggestion.
STAGE_B_M="${STAGE_B_M:-}"
STAGE_B_STRATEGIES="${STAGE_B_STRATEGIES:-}"

ATTACKS="collusion_backdoor_cifar10_f30 collusion_adaptive_backdoor_cifar10_f30"
STAGE_A_SEEDS="0,1,2"
STAGE_B_SEEDS="0,1,2,3,4"
STAGE_A_DEFENSES="undefended +defense=topsis_score_weight +defense=topsis_capped_weighted_median +baseline=coordinate_median"
STAGE_B_DEFENSES="$STAGE_A_DEFENSES +baseline=fltrust +baseline=foolsgold"

# run_cell <seeds> <experiment> <strategy> <m> <defense override | undefended>
# The tag becomes the run label prefix: results/<tag>[__<defense>]/<hash>/seed_*/
run_cell() {
  local seeds="$1" exp="$2" strategy="$3" m="$4" defense="$5"
  local tag="e1_${exp}_${strategy}_m${m}"
  local args=("+experiment=${exp}" "+tag=${tag}" "federation.force_malicious_per_round=${m}")
  if [ "$strategy" != "independent" ]; then
    args+=("+attack/collusion=${strategy}")
  fi
  if [ "$defense" != "undefended" ]; then
    args+=("$defense")
  fi
  echo "=========== ${tag} [${defense}] seeds=${seeds} $(date -u +%FT%TZ) ==========="
  if [ "$DRY_RUN" = "1" ]; then
    return
  fi
  uv run python experiments/run.py --multirun "${args[@]}" \
    federation.seed="$seeds" +skip_existing=true \
    2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

block_A_CONTROL() {
  # Independent attackers at each forced m: what co-selection alone does.
  local seeds="${SEEDS:-$STAGE_A_SEEDS}"
  for exp in $ATTACKS; do
    for m in 1 2 3 4 5; do
      for d in $STAGE_A_DEFENSES; do
        run_cell "$seeds" "$exp" independent "$m" "$d"
      done
    done
  done
}

block_A_COLLUDE() {
  # S1 clone and S3 reference shift. Prediction: capped-wmed holds to m=5
  # under clone but breaks earlier under reference shift.
  local seeds="${SEEDS:-$STAGE_A_SEEDS}"
  for exp in $ATTACKS; do
    for strategy in clone reference_shift; do
      for m in 2 3 4 5; do
        for d in $STAGE_A_DEFENSES; do
          run_cell "$seeds" "$exp" "$strategy" "$m" "$d"
        done
      done
    done
  done
}

require_stage_b() {
  if [ -z "$STAGE_B_M" ] || [ -z "$STAGE_B_STRATEGIES" ]; then
    echo "Stage B needs STAGE_B_M and STAGE_B_STRATEGIES (see the TODO at the top;" \
      "uv run python -m topfed.analysis.e1 --stage-b suggests them)" >&2
    exit 2
  fi
}

# stage_b_grid <defense overrides>
stage_b_grid() {
  local defenses="$1"
  local seeds="${SEEDS:-$STAGE_B_SEEDS}"
  for exp in $ATTACKS; do
    for strategy in $STAGE_B_STRATEGIES; do
      if [ "$strategy" = "budget_split" ] && [ "$exp" = "collusion_backdoor_cifar10_f30" ]; then
        continue
      fi
      for m in $STAGE_B_M; do
        local s="$strategy"
        if [ "$m" = "1" ]; then
          s="independent"
        fi
        for d in $defenses; do
          run_cell "$seeds" "$exp" "$s" "$m" "$d"
        done
      done
    done
  done
}

block_B() {
  require_stage_b
  stage_b_grid "$STAGE_B_DEFENSES"
}

block_B_AGREE() {
  require_stage_b
  stage_b_grid "+defense=topsis_agreement_capped_weighted_median"
}

for block in $BLOCKS; do
  case "$block" in
    A_CONTROL|A_COLLUDE|B|B_AGREE) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### E1 SWEEP COMPLETE (${BLOCKS}) ###########"
