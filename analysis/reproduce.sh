#!/usr/bin/env bash
# Regenerate every table and figure of the paper from results/ and
# reconstruction/. Run from the repository root after
# scripts/fetch_results.sh. Writes tables/ and figures/. About ten minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
# Fixed timestamp in the figure files, so that reruns are byte-identical.
export SOURCE_DATE_EPOCH=1794700800
mkdir -p tables figures
uv run python analysis/make_views.py
M=views/main
py() { uv run --project "$ROOT" python "$@"; }

# Main grid (CIFAR-10, FEMNIST, CIFAR-100): ASR, clean accuracy, paired
# tests, detection rates, server overhead, best-response knobs.
py -m topfed.analysis.e3 --results-root $M/results --out tables/e3-tables.md
# Composed attacker and the hybrid variants.
py -m topfed.analysis.e4 --results-root $M/results --out tables/e4-tables.md
# Compromised-fraction sweep, 5 to 40 percent.
py -m topfed.analysis.e2 --results-root views/main_no_c100/results --out tables/e2-tables.md
# Follow-up runs: other scores, in-range attacks, held-out knob selection,
# geometric median, server cost.
py analysis/final_queue.py --main-seeds 10 > tables/final-queue.md
# Reconstruction SSIM by images per client.
py experiments/gradient_inversion_summary.py --root reconstruction > tables/reconstruction.md

cd $M
py "$ROOT/analysis/paper_supplementary_cells.py" > "$ROOT/tables/supplementary-cells.md"
py "$ROOT/analysis/top_score_share.py" > "$ROOT/tables/top-score-share.md"
py "$ROOT/analysis/standard_backdoor_points.py" > "$ROOT/tables/standard-backdoor-points.md"
py "$ROOT/analysis/gameability_stats.py" "$ROOT/tables/gameability-points.json" > "$ROOT/tables/gameability-stats.md"
py "$ROOT/analysis/load_cells.py" "$ROOT/tables/cells.pkl" > /dev/null
py "$ROOT/analysis/published_vs_undefended.py" "$ROOT/tables/cells.pkl" "$ROOT/tables/published-vs-undefended.json" > "$ROOT/tables/published-vs-undefended.md"
rm -f "$ROOT/tables/cells.pkl"
py "$ROOT/analysis/diag_ledgers.py" > "$ROOT/tables/diag-ledgers.md"
py "$ROOT/analysis/last_runs_c100.py" > "$ROOT/tables/cifar100-budgets-and-ledgers.md"
py "$ROOT/analysis/last_runs_analysis.py" > "$ROOT/tables/composed-and-knob-scans.md"
py "$ROOT/analysis/amplification_weights.py" > "$ROOT/tables/amplification-weights.md"
py "$ROOT/analysis/median_selection.py" > "$ROOT/tables/median-selection.md"
py "$ROOT/analysis/auc_tables.py" > "$ROOT/tables/auc-tables.md"
py "$ROOT/analysis/paper_figures.py" --out-dir "$ROOT/figures" > "$ROOT/tables/figure-data.md"
cd "$ROOT"
py analysis/paper_figures_final.py --out-dir figures > tables/figure-data-final.md
echo "done: tables/ and figures/"
