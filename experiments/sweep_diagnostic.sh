#!/usr/bin/env bash
# Coordinate-level median diagnostic: which client supplies each coordinate
# of a median rule's aggregate. Every run carries
# +diagnostics.median_selection=true, which writes median_selection.json
# beside defense_ledger.json and changes no number of the run. The flag is
# excluded from the config hash, so these runs carry the tag
# diag_<experiment> and never share a directory with the frozen cells.
#
# Cells: {backdoor_cifar10_f20, adaptive_backdoor_cifar10_f20 at the attack
# config's budget 0.6} x {capped weighted median (Peer), plain coordinate
# median} x SEEDS (default 0,1,2).                               12 runs
#
# Resumable (+skip_existing=true keys on metrics.json). Read the output with
#   uv run python analysis/median_selection.py
#
# Examples:
#   experiments/sweep_diagnostic.sh
#   SEEDS=0 experiments/sweep_diagnostic.sh
#   DRY_RUN=1 experiments/sweep_diagnostic.sh      # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS="${SEEDS:-0,1,2}"
DRY_RUN="${DRY_RUN:-0}"

EXPERIMENTS="backdoor_cifar10_f20 adaptive_backdoor_cifar10_f20"
RULES=("+defense=topsis_capped_weighted_median" "+baseline=coordinate_median")

# run_cell <base experiment> <rule override>
run_cell() {
  local base="$1" rule="$2" tag="diag_$1"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY ${tag} ${base} [${rule}] seeds=${SEEDS}"
    return 0
  fi
  echo "=========== ${tag} [${rule}] seeds=${SEEDS} $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" "${rule}" \
    +diagnostics.median_selection=true \
    federation.seed="$SEEDS" +skip_existing=true \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|Error|Traceback" | tail -10
}

for exp in $EXPERIMENTS; do
  for rule in "${RULES[@]}"; do
    run_cell "$exp" "$rule"
  done
done
echo "=========== sweep_diagnostic done $(date -u +%FT%TZ) ==========="
