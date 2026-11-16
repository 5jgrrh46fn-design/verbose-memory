#!/usr/bin/env bash
# E5: why every defense fails on FEMNIST at f=20%. Resumable diagnostic sweep.
#
# Every cell is a Hydra multirun over SEEDS with +skip_existing=true, so a
# re-run after an interruption only executes the (config, seed) pairs that
# have no metrics.json yet. Each cell carries an explicit run tag
# (+tag=e5<block>_...) so its runs never share a label directory with the frozen
# reference cells; the reference cells themselves (natural sampling, K=10,
# corner trigger, natural partition) are read from their canonical
# directories by the analysis, not re-run here.
#
# Blocks, in priority order (select with BLOCKS="D3 D8"):
#   D3  forced co-selection: m in {2..6} attackers per round, f=0.20,
#       capped-wmed and plain median, FEMNIST and CIFAR-10          100 runs
#   D8  trigger placement: FEMNIST centre trigger; CIFAR-10 trigger in a
#       constant 4-pixel padded border (ring, and size-matched corner
#       patch), undefended and capped-wmed                            30 runs
#   D7  persistence: attack active rounds 1-100 then stops              10 runs
#   D9  sampling-tail levers: K in {20, 30}; TemporalReputation on the
#       weighted-median base                                          15 runs
#   D1  oracle detection on weighted-median and mean bases             10 runs
#   D4  heterogeneity dial: Dirichlet-FEMNIST alpha in {IID,1,0.5,0.1};
#       CIFAR-10 alpha=0.1                                            50 runs
#   D6  attack-strength dial at f20 vs capped-wmed                     30 runs
#   D2  (optional, off by default) delayed start from round 50 / 100   20 runs
#
# Tables: uv run python -m topfed.analysis.e5 --out tables/e5-tables.md
#
# Examples:
#   experiments/sweep_e5.sh                       # D3 D8 D7 D9 D1 D4 D6
#   BLOCKS="D3" experiments/sweep_e5.sh           # one block
#   BLOCKS="D2" experiments/sweep_e5.sh           # the optional block
#   SEEDS="0" BLOCKS="D1" experiments/sweep_e5.sh # one seed for a quick look
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-D3 D8 D7 D9 D1 D4 D6}"
SEEDS="${SEEDS:-0,1,2,3,4}"

FEM="femnist_backdoor_f20"
CIF="backdoor_cifar10_f20"
CAPPED="+defense=topsis_capped_weighted_median"
MEDIAN="+baseline=coordinate_median"
WINDOW="attack.schedule._target_=topfed.attacks.schedule.WindowedAttack"

# run_cell <tag> <base experiment> [override ...]
# The tag becomes the run label prefix: results/<tag>[__<defense>]/<hash>/seed_*/
run_cell() {
  local tag="$1" base="$2"
  shift 2
  echo "=========== ${tag} [$*] $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" \
    federation.seed="$SEEDS" +skip_existing=true "$@" \
    2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

block_D3() {
  # Forced co-selection. Prediction: capped-wmed holds at m <= 4 and is
  # captured at m >= 5 whenever the malicious weight share reaches 0.5; a
  # captured CIFAR-10 model recovers, a captured FEMNIST model does not.
  for m in 2 3 4 5 6; do
    run_cell "e5d3_${FEM}_m${m}" "$FEM" "+federation.force_malicious_per_round=${m}" "$CAPPED"
    run_cell "e5d3_${FEM}_m${m}" "$FEM" "+federation.force_malicious_per_round=${m}" "$MEDIAN"
    run_cell "e5d3_${CIF}_m${m}" "$CIF" "+federation.force_malicious_per_round=${m}" "$CAPPED"
    run_cell "e5d3_${CIF}_m${m}" "$CIF" "+federation.force_malicious_per_round=${m}" "$MEDIAN"
  done
}

block_D8() {
  # Trigger placement. Prediction: centre-trigger FEMNIST recovers like
  # CIFAR-10; border-trigger CIFAR-10 becomes permanent like FEMNIST.
  # FEMNIST: same 6x6 trigger and -5.0 amplitude, moved to the ink region.
  run_cell "e5d8_${FEM}_center" "$FEM" "+attack.poisoner.trigger_position=center"
  run_cell "e5d8_${FEM}_center" "$FEM" "+attack.poisoner.trigger_position=center" "$CAPPED"
  # CIFAR-10: images padded with a constant 4-pixel border (40x40 input, the
  # model's flatten size follows). Two placements inside that border:
  #   ring  - the trigger fills the whole 4-pixel ring 
  #   corner - a size-matched 4x4 patch in the padded bottom-right corner
  # Both sit entirely in pixels no natural sample ever varies.
  local pad="+dataset.pad_border=4 attack.poisoner.trigger_size=4"
  run_cell "e5d8_${CIF}_border_ring" "$CIF" $pad "+attack.poisoner.trigger_position=border"
  run_cell "e5d8_${CIF}_border_ring" "$CIF" $pad "+attack.poisoner.trigger_position=border" "$CAPPED"
  run_cell "e5d8_${CIF}_border_corner" "$CIF" $pad "+attack.poisoner.trigger_position=bottom_right"
  run_cell "e5d8_${CIF}_border_corner" "$CIF" $pad "+attack.poisoner.trigger_position=bottom_right" "$CAPPED"
}

block_D7() {
  # Persistence: attackers active rounds 1-100 only, then train benignly.
  # Measures whether the median base ever erodes an implanted backdoor.
  local win="$WINDOW +attack.schedule.start_round=1 +attack.schedule.end_round=100"
  run_cell "e5d7_${FEM}_w1-100" "$FEM" $win
  run_cell "e5d7_${FEM}_w1-100" "$FEM" $win "$CAPPED"
}

block_D9() {
  # Sampling-tail levers at f20 vs capped-wmed. Larger K shrinks the
  # hypergeometric tail (breakdown needs m >= K/2); carried reputation keeps a
  # fresh bloc below equal share on first co-selection.
  for k in 20 30; do
    run_cell "e5d9_${FEM}_k${k}" "$FEM" "federation.clients_per_round=${k}" "$CAPPED"
  done
  run_cell "e5d9_${FEM}_temporal" "$FEM" "+defense=topsis_temporal_reputation_wmedian"
}

block_D1() {
  # Oracle detection: true attackers weight 0. If ASR stays high with
  # perfect detection, aggregation is the bottleneck.
  run_cell "e5d1_${FEM}" "$FEM" "+defense=oracle_filter_wmedian"
  run_cell "e5d1_${FEM}" "$FEM" "+defense=oracle_filter"
}

block_D4() {
  # Heterogeneity dial: pool the FEMNIST writers and re-split synthetically.
  run_cell "e5d4_${FEM}_iid" "$FEM" "partition=iid"
  run_cell "e5d4_${FEM}_iid" "$FEM" "partition=iid" "$CAPPED"
  for a in 1.0 0.5 0.1; do
    run_cell "e5d4_${FEM}_a${a}" "$FEM" "partition=dirichlet_alpha_${a}"
    run_cell "e5d4_${FEM}_a${a}" "$FEM" "partition=dirichlet_alpha_${a}" "$CAPPED"
  done
  run_cell "e5d4_${CIF}_a0.1" "$CIF" "partition=dirichlet_alpha_0.1"
  run_cell "e5d4_${CIF}_a0.1" "$CIF" "partition=dirichlet_alpha_0.1" "$CAPPED"
}

block_D6() {
  # Attack-strength dial vs capped-wmed. Reference cell is scaling on,
  # amplitude -5.0, augmented batches (rate ignored).
  run_cell "e5d6_${FEM}_noscale" "$FEM" "attack.update_manipulator=null" "$CAPPED"
  run_cell "e5d6_${FEM}_amp-2" "$FEM" "attack.poisoner.trigger_amplitude=-2.0" "$CAPPED"
  run_cell "e5d6_${FEM}_noscale_amp-2" "$FEM" "attack.update_manipulator=null" "attack.poisoner.trigger_amplitude=-2.0" "$CAPPED"
  run_cell "e5d6_${FEM}_replace0.5" "$FEM" "attack.poisoner.batch_mode=replace" "attack.poisoner.poisoning_rate=0.5" "$CAPPED"
  run_cell "e5d6_${FEM}_replace1.0" "$FEM" "attack.poisoner.batch_mode=replace" "attack.poisoner.poisoning_rate=1.0" "$CAPPED"
  run_cell "e5d6_${FEM}_replace0.5_amp-2" "$FEM" "attack.poisoner.batch_mode=replace" "attack.poisoner.poisoning_rate=0.5" "attack.poisoner.trigger_amplitude=-2.0" "$CAPPED"
}

block_D2() {
  # Optional. Delayed start: attackers idle until round 50 / 100.
  for s in 50 100; do
    local win="$WINDOW +attack.schedule.start_round=${s}"
    run_cell "e5d2_${FEM}_start${s}" "$FEM" $win
    run_cell "e5d2_${FEM}_start${s}" "$FEM" $win "$CAPPED"
  done
}

for block in $BLOCKS; do
  case "$block" in
    D1|D2|D3|D4|D6|D7|D8|D9) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### E5 SWEEP COMPLETE (${BLOCKS}) ###########"
