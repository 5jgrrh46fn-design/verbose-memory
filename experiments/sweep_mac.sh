#!/usr/bin/env bash
# CPU sweep for the cells the paper still lacks, sized for a laptop: the
# norm-clipped mean on the headline CIFAR-10 and FEMNIST cells, and the
# adaptive backdoor on FEMNIST. Resumable (+skip_existing), so it can be
# stopped and restarted at any time; a run that was killed mid-way leaves no
# metrics.json and is simply redone.
#
# Runs pair with the GPU campaign by (label, config hash, seed): the hash
# excludes the device and the determinism flags, so a CPU run fills the same
# cell. The manifest records device=cpu and the platform, which is how the
# analysis can tell the two apart.
#
# Blocks (select with BLOCKS="CLIP_MAIN FA_MAIN"), in priority order:
#   CLIP_MAIN     +baseline=clipped_mean on CIFAR-10 {backdoor,
#                 adaptive_backdoor (gamma 0.6)} x f in {0.10, 0.20}, SEEDS,
#                 tags sp_<experiment> (the tags sweep_baselines.sh MAIN uses,
#                 so the cells land next to the other baselines).    40 runs
#   FA_MAIN       femnist_adaptive_backdoor_f10 at target_norm FA_BUDGET
#                 (default 0.3, the smallest probed budget that implants
#                 undefended) x {undefended, capped weighted median,
#                 fltrust_median, coordinate_median}, FA_SEEDS, tag
#                 sp_femnist_adaptive_backdoor_f10_g<budget> (suffix omitted
#                 at the CIFAR default 0.6; same rule as sweep_gaps.sh).
#                                                                      20 runs
#   CLIP_FEMNIST  +baseline=clipped_mean on FEMNIST backdoor f10/f20, SEEDS,
#                 tags sp_femnist_backdoor_<f>.                        20 runs
#   CLIP_RESCAN   the adaptive attacker's budget rescan against clipped_mean:
#                 target_norm in {1.0, 1.5} x f in {0.10, 0.20}, FA_SEEDS,
#                 tags sp_adaptive_backdoor_cifar10_<f>_g<budget>.    20 runs
#   FA_EXTRA      FA_MAIN's cell against +baseline=fltrust and +baseline=flame,
#                 FA_SEEDS. Off by default.                            10 runs
#
# Timing on an Apple M2 (CPU, one process): about 8.5 s per CIFAR-10 round and
# 5 s per FEMNIST round, so 28 and 17 minutes per 200-round run. The default
# four blocks are about 40 hours sequential.
#
# WORKERS=n runs n Hydra multiruns per cell at once, each over a disjoint
# subset of the seeds, with THREADS torch threads each (default 2 workers x
# 4 threads on an 8-core machine). Keep the laptop awake with caffeinate:
#   caffeinate -i experiments/sweep_mac.sh
#
# Examples:
#   experiments/sweep_mac.sh                          # default blocks
#   BLOCKS="FA_MAIN" FA_BUDGET=0.2 experiments/sweep_mac.sh
#   WORKERS=1 BLOCKS="CLIP_MAIN" experiments/sweep_mac.sh
#   DRY_RUN=1 experiments/sweep_mac.sh                # list cells only
#   EXTRA="federation.num_rounds=2 results_root=/tmp/smoke" BLOCKS=FA_MAIN FA_SEEDS=0,1 experiments/sweep_mac.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-CLIP_MAIN FA_MAIN CLIP_FEMNIST CLIP_RESCAN}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
FA_SEEDS="${FA_SEEDS:-0,1,2,3,4}"
FA_BUDGET="${FA_BUDGET:-0.3}"
WORKERS="${WORKERS:-2}"
THREADS="${THREADS:-4}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-results/_sweep_mac_logs}"
# EXTRA holds extra Hydra overrides for every run (smoke tests:
# EXTRA="federation.num_rounds=2 results_root=/tmp/x").
EXTRA="${EXTRA:-}"

CLIP="+baseline=clipped_mean"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

# split_seeds "0,1,2,3" 2 -> prints one comma list per worker (round robin)
split_seeds() {
  local seeds="$1" n="$2"
  python3 - "$seeds" "$n" <<'EOF'
import sys
seeds = sys.argv[1].split(","); n = int(sys.argv[2])
parts = [seeds[i::n] for i in range(n)]
for p in parts:
    if p: print(",".join(p))
EOF
}

# run_cell <tag> <base experiment> <seeds> [override ...]
run_cell() {
  local tag="$1" base="$2" seeds="$3"
  shift 3
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY ${tag} ${base} [$*] seeds=${seeds}"
    return 0
  fi
  mkdir -p "$LOG_DIR"
  echo "=========== ${tag} [$*] seeds=${seeds} workers=${WORKERS} $(date -u +%FT%TZ) ==========="
  local -a pids=() logs=()
  local part i=0
  while IFS= read -r part; do
    local log="${LOG_DIR}/${tag}$(printf '%s' "$*" | tr -c 'A-Za-z0-9=._-' '_')_w${i}.log"
    uv run python experiments/run.py --multirun \
      "+experiment=${base}" "+tag=${tag}" \
      federation.seed="$part" +skip_existing=true "$@" $EXTRA \
      < /dev/null > "$log" 2>&1 &
    pids+=("$!"); logs+=("$log")
    i=$((i + 1))
  done < <(split_seeds "$seeds" "$WORKERS")
  local rc=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  grep -h -E "^\[run\] (done|skip)|Error|Traceback" "${logs[@]}" | tail -6
  [ "$rc" = "0" ] || echo "!!! ${tag}: a worker exited non-zero; see ${LOG_DIR}" >&2
  return 0
}

fa_tag() {
  local exp="$1" budget="$2"
  if [ "$budget" = "0.6" ]; then echo "sp_${exp}"; else echo "sp_${exp}_g${budget}"; fi
}

block_CLIP_MAIN() {
  for f in f10 f20; do
    for attack in backdoor adaptive_backdoor; do
      local exp="${attack}_cifar10_${f}"
      run_cell "sp_${exp}" "$exp" "$SEEDS" "$CLIP"
    done
  done
}

block_FA_MAIN() {
  local exp="femnist_adaptive_backdoor_f10"
  local tag; tag="$(fa_tag "$exp" "$FA_BUDGET")"
  local budget="attack.update_manipulator.target_norm=${FA_BUDGET}"
  run_cell "$tag" "$exp" "$FA_SEEDS" "$budget"
  run_cell "$tag" "$exp" "$FA_SEEDS" "$budget" "+defense=topsis_capped_weighted_median"
  run_cell "$tag" "$exp" "$FA_SEEDS" "$budget" "+baseline=fltrust_median"
  run_cell "$tag" "$exp" "$FA_SEEDS" "$budget" "+baseline=coordinate_median"
}

block_FA_EXTRA() {
  local exp="femnist_adaptive_backdoor_f10"
  local tag; tag="$(fa_tag "$exp" "$FA_BUDGET")"
  local budget="attack.update_manipulator.target_norm=${FA_BUDGET}"
  run_cell "$tag" "$exp" "$FA_SEEDS" "$budget" "+baseline=fltrust"
  run_cell "$tag" "$exp" "$FA_SEEDS" "$budget" "+baseline=flame"
}

block_CLIP_FEMNIST() {
  for f in f10 f20; do
    local exp="femnist_backdoor_${f}"
    run_cell "sp_${exp}" "$exp" "$SEEDS" "$CLIP"
  done
}

block_CLIP_RESCAN() {
  for f in f10 f20; do
    local exp="adaptive_backdoor_cifar10_${f}"
    for g in 1.0 1.5; do
      run_cell "sp_${exp}_g${g}" "$exp" "$FA_SEEDS" \
        "attack.update_manipulator.target_norm=${g}" "$CLIP"
    done
  done
}

for block in $BLOCKS; do
  case "$block" in
    CLIP_MAIN|FA_MAIN|FA_EXTRA|CLIP_FEMNIST|CLIP_RESCAN) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### MAC SWEEP COMPLETE (${BLOCKS}) ###########"
