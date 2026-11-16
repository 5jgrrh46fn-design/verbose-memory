#!/usr/bin/env bash
# Final queue: other fusions of the five-metric score through the six
# integrations, held-out evaluation seeds, and the reference cells those
# comparisons need on the machine that runs them. Every cell writes to a new
# tag (prefix fin_) and is resumable (+skip_existing), so no existing cell is
# touched.
#
# Blocks (select with BLOCKS="REF_NB A"):
#   REF_NB   reference rows on CIFAR-10 f10/f20 under the norm-bounded
#            backdoor at budget 0.6: undefended, capped weighted median, plain
#            median, SEEDS. Tag fin_adaptive_backdoor_cifar10_<f>.     60 runs
#   REF_DA   the same three rows under the direction-aligned backdoor at
#            cosine 0.9. Tag fin_fltrust_backdoor_cifar10_<f>_c0.9.    60 runs
#   REF_STD  the same three rows under the standard backdoor. Tag
#            fin_backdoor_cifar10_<f>.                                 60 runs
#   A        the six integrations with defense.scorer.weighting in WEIGHTINGS
#            (default equal entropy) under the norm-bounded backdoor at 0.6,
#            f10/f20, SEEDS. Tag fin_adaptive_backdoor_cifar10_<f>; the label
#            carries the integrator and the weighting.                240 runs
#   B        held-out seeds (HOLDOUT_SEEDS, default 10..19) at the knob
#            already selected against each rule on seeds 0..9: undefended,
#            capped weighted median, fltrust_median, plain median, FLTrust and
#            ScoreWeight on the standard, norm-bounded, direction-aligned and
#            composed backdoors at f10/f20, plus the five score integrations
#            and the undefended row at budget 0.6. Tags
#            fin_holdout_<experiment>[_g<budget>][_c<cosine>].        580 runs
#   C_UP     ten-seed cells for rows that have five elsewhere: FLTrust under
#            the direction-aligned backdoor at cosine 0.9 (tag of REF_DA), and
#            ScoreWeight with its three reference rows under the clean-label
#            backdoor (tag fin_clean_label_cifar10_<f>), f10/f20.     100 runs
#   C_F3040  undefended, capped weighted median, plain median and
#            fltrust_median at f30/f40 under the standard and the norm-bounded
#            (0.6) backdoor, SEEDS. Tags fin_[adaptive_]backdoor_cifar10_<f>.
#                                                                      160 runs
#   C_ROOT   fltrust_median against the root-set size: dataset.val_per_class
#            in {1, 10, 100} at f10/f20, standard and norm-bounded (0.6),
#            SEEDS. Tags fin_[adaptive_]backdoor_cifar10_<f>[_root<v>]; the
#            100-per-class cells carry no suffix.                     120 runs
#   C_H500   500 rounds (federation.num_rounds=500): undefended, capped
#            weighted median and plain median, standard and norm-bounded
#            (0.6), f10/f20, FIVE_SEEDS. Tags fin_h500_<experiment>.
#                                                    60 runs of 2.5x length
#   D_PROBE  direction-aligned and composed backdoors on FEMNIST and
#            CIFAR-100, undefended, seed 0, f10/f20: cosine in D_COSINES
#            (default 0.5 0.7 0.9 0.95); composed pairs of a budget (the
#            dataset's own norm-bounded budget, 0.6 and 1.0) and a cosine in
#            D_COMPOSED_COSINES (default 0.5 0.9). The selected knob is the
#            weakest one (highest cosine; smallest budget, then highest
#            cosine) whose last-50-round ASR reaches 0.5.              40 runs
#   D_MAIN   the two attackers at the selected knobs x {undefended, capped
#            weighted median, fltrust_median, plain median, FLTrust},
#            f10/f20, SEEDS. Knobs come from the environment, one per dataset
#            and fraction: D_DA_FEMNIST_F10=<cosine>,
#            D_COMP_FEMNIST_F10=<budget>,<cosine>, and the same for F20 and
#            for CIFAR100; a cell whose knob is unset is skipped. Tags
#            fin_femnist_fltrust_backdoor_<f>_c<c>,
#            fin_femnist_composed_backdoor_<f>_g<b>_c<c>,
#            fin_fltrust_backdoor_cifar100_<f>_c<c>,
#            fin_composed_backdoor_cifar100_<f>_g<b>_c<c>.      up to 400 runs
#   E_PROBE  two single-signal trust scores (defense.scorer.kind in E_KINDS,
#            default server_cosine history_similarity) through the six
#            integrations, norm-bounded backdoor at 0.6, f10, seed 0. The
#            ledgers give each threshold rule's honest-exclusion rate. 12 runs
#   E_NB     the same scores and integrations under the norm-bounded backdoor
#            at 0.6, f10/f20, SEEDS (tag of REF_NB).                  240 runs
#   E_DA     ... under the direction-aligned backdoor at cosine 0.9 (tag of
#            REF_DA).                                                 240 runs
#   E_STD    ... under the standard backdoor (tag of REF_STD).        240 runs
#            E_MEDTHR lists "<kind>:<integration>" pairs whose threshold rule
#            also runs with the threshold at the cohort's median score
#            (defense.integrator.cohort_median=true), in every E block, under
#            the block's tag plus _medthr.
#   P1_COS   the server_cosine score through the six integrations under the
#            direction-aligned backdoor at every cosine in P1_COSINES
#            (default 0.5 0.7 0.9 0.95), f10/f20, SEEDS, with the undefended
#            and plain-median rows at the same cosine. Tags
#            fin_fltrust_backdoor_cifar10_<f>_c<c>; E_MEDTHR adds the
#            cohort-median threshold variants.  56 cells (+ variants) x seeds
#   P1_M4    the validation-loss-change metric alone as the score
#            (defense.scorer.weighting=m4_only) through the six integrations
#            under the norm-bounded backdoor at 0.6, f10/f20, SEEDS (tag of
#            REF_NB).                                         12 cells x seeds
#   P1_FGC   the history_similarity score through the six integrations under
#            the composed backdoor at every "<budget>,<cosine>" pair of
#            P1_FGC_KNOBS (default 0.6,0.9 0.6,0.5), f10/f20, SEEDS, with the
#            undefended and plain-median rows at the same knob. Tags
#            fin_composed_backdoor_cifar10_<f>_g<b>_c<c>; E_MEDTHR adds the
#            cohort-median threshold variants.   16 cells per knob x seeds
#   P3_SCAN  knob scan for the held-out protocol: capped weighted median,
#            fltrust_median, plain median and FLTrust at every knob of the
#            grids (norm budget in P3_BUDGETS, default 0.5 0.6 0.75 1.0 1.5
#            2.0; cosine in P1_COSINES; composed pairs of P3_COMPOSED_BUDGETS
#            x P3_COMPOSED_COSINES, default 0.6 1.0 x 0.5 0.9), f10/f20,
#            SEEDS (meant to be 0..4). Tags fin_<experiment>[_g<b>][_c<c>]
#            (budget 0.6 carries no suffix, like REF_NB).   112 cells x seeds
#   P3_EVAL  the same four rules and the undefended row on HOLDOUT_SEEDS at
#            the knob selected from the scan. One variable per attack, rule
#            and fraction, e.g. P3_NB_PEER_F10=0.6, P3_DA_FLTRUST_F20=0.9,
#            P3_COMP_ROOT_F10=0.6,0.9 (rules PEER ROOT MEDIAN FLTRUST); a
#            cell whose variable is unset is skipped. The undefended row runs
#            at every knob selected in its attack and fraction. Tags
#            fin_holdout_<experiment>_g<b> / _c<c> / _g<b>_c<c>.
#   P2       in-range backdoors: the ALIE backdoor at num_std in P2_ZS
#            (default 0.5 1.0 1.5) and at the paper's z_max for the
#            population counts (0.11 at f10, 0.29 at f20: 100 clients with 10
#            or 20 attackers; the bound is not positive for a ten-client
#            cohort), and the Min-Max backdoor, x {undefended, plain median,
#            trimmed mean, FLTrust, capped weighted median, fltrust_median},
#            f10/f20, SEEDS. Tags fin_alie_backdoor_cifar10_<f>_z<z>,
#            fin_minmax_backdoor_cifar10_<f>.                 60 cells x seeds
#            P2_ALIE_ONLY=1 keeps only the ALIE cells at the values of P2_ZS
#            (no z_max cell, no Min-Max).
#   P4       the capped weighted geometric median
#            (+defense=topsis_capped_weighted_geomedian) on the standard,
#            norm-bounded (0.6 and 1.0), direction-aligned (0.5 and 0.9) and
#            clean-label backdoors, its uncapped ablation
#            (topsis_weighted_geomedian) on the standard and norm-bounded
#            (0.6) backdoors, and the reference rows of those comparisons
#            (REF_STD; undefended, capped weighted median and plain median on
#            clean-label, at budget 1.0 and at cosines 0.5 and 0.9; REF_NB is
#            expected to exist), f10/f20, SEEDS.              46 cells x seeds
#   P4_COST  50-round timing runs (overhead_cifar10) for undefended, plain
#            median, capped weighted median and capped weighted geometric
#            median, seed 0, tag fin_overhead_cifar10. Needs the device to
#            itself.                                                    4 runs
#   G_PROBE  the untargeted attacks aimed at a median (LIE and the Fang
#            median / trimmed-mean attack), f10/f20, undefended and plain
#            median, seed 0, with benign runs of both rules as the accuracy
#            reference. Tags fin_<attack>_cifar10_<f>, fin_benign_cifar10.
#                                                                       10 runs
#   G_MAIN   the attacks in G_ATTACKS (default lie fang) x f10/f20 x
#            {undefended, capped weighted median, fltrust_median, plain
#            median, trimmed mean, FLTrust}, FIVE_SEEDS, and the benign run of
#            each rule; the metric is clean accuracy.         up to 150 runs
#
# Examples:
#   BLOCKS="REF_NB A" experiments/sweep_final.sh
#   DRY_RUN=1 experiments/sweep_final.sh             # list cells only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLOCKS="${BLOCKS:-REF_NB A B REF_DA REF_STD}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
HOLDOUT_SEEDS="${HOLDOUT_SEEDS:-10,11,12,13,14,15,16,17,18,19}"
WEIGHTINGS="${WEIGHTINGS:-equal entropy}"
FIVE_SEEDS="${FIVE_SEEDS:-0,1,2,3,4}"
E_KINDS="${E_KINDS:-server_cosine history_similarity}"
G_ATTACKS="${G_ATTACKS:-lie fang}"
P1_COSINES="${P1_COSINES:-0.5 0.7 0.9 0.95}"
P1_FGC_KNOBS="${P1_FGC_KNOBS:-0.6,0.9 0.6,0.5}"
P3_BUDGETS="${P3_BUDGETS:-0.5 0.6 0.75 1.0 1.5 2.0}"
P2_ZS="${P2_ZS:-0.5 1.0 1.5}"
P2_ALIE_ONLY="${P2_ALIE_ONLY:-}"
P3_COMPOSED_BUDGETS="${P3_COMPOSED_BUDGETS:-0.6 1.0}"
P3_COMPOSED_COSINES="${P3_COMPOSED_COSINES:-0.5 0.9}"
E_MEDTHR="${E_MEDTHR:-}"
FIN_PREFIX="${FIN_PREFIX:-fin_}"
D_COSINES="${D_COSINES:-0.5 0.7 0.9 0.95}"
D_COMPOSED_COSINES="${D_COMPOSED_COSINES:-0.5 0.9}"
D_BUDGETS_FEMNIST="${D_BUDGETS_FEMNIST:-0.2 0.6 1.0}"
D_BUDGETS_CIFAR100="${D_BUDGETS_CIFAR100:-0.6 1.0 3.0}"
DRY_RUN="${DRY_RUN:-0}"

PEER="+defense=topsis_capped_weighted_median"
MEDIAN="+baseline=coordinate_median"
ROOT="+baseline=fltrust_median"
FLTRUST="+baseline=fltrust"
SCOREWEIGHT="+defense=topsis_score_weight"
INTEGRATIONS="topsis_score_weight topsis_temporal_reputation topsis_hard_filter topsis_quarantine topsis_adaptive topsis_capped_weighted_median"

# run_cell <tag> <base experiment> <seeds> [override ...]; the override "none"
# stands for the undefended rule and is dropped. FIN_PREFIX (default fin_)
# replaces the leading fin_ of every tag, so a second machine that runs part
# of the queue writes to labels of its own and its rows never mix with this
# machine's.
run_cell() {
  local tag="${FIN_PREFIX}${1#fin_}" base="$2" seeds="$3"
  shift 3
  local -a extra=()
  local o
  for o in "$@"; do
    [ "$o" = "none" ] || extra+=("$o")
  done
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY ${tag} ${base} [${extra[*]+${extra[*]}}] seeds=${seeds}"
    return 0
  fi
  echo "=========== ${tag} [${extra[*]+${extra[*]}}] seeds=${seeds} $(date -u +%FT%TZ) ==========="
  uv run python experiments/run.py --multirun \
    "+experiment=${base}" "+tag=${tag}" \
    federation.seed="$seeds" +skip_existing=true ${extra[@]+"${extra[@]}"} \
    < /dev/null 2>&1 | grep -E "^\[run\]|round 200/|round 500/|Error|Traceback" | tail -10
}

budget() { echo "attack.update_manipulator.target_norm=$1"; }
cosine() { echo "attack.update_manipulator.cosine_target=$1"; }

block_REF_NB() {
  local f d
  for f in f10 f20; do
    for d in none "$PEER" "$MEDIAN"; do
      run_cell "fin_adaptive_backdoor_cifar10_${f}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" "$(budget 0.6)" "$d"
    done
  done
}

block_REF_DA() {
  local f d
  for f in f10 f20; do
    for d in none "$PEER" "$MEDIAN"; do
      run_cell "fin_fltrust_backdoor_cifar10_${f}_c0.9" "fltrust_backdoor_cifar10_${f}" "$SEEDS" "$(cosine 0.9)" "$d"
    done
  done
}

block_REF_STD() {
  local f d
  for f in f10 f20; do
    for d in none "$PEER" "$MEDIAN"; do
      run_cell "fin_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$SEEDS" "$d"
    done
  done
}

block_A() {
  local w f d
  for w in $WEIGHTINGS; do
    for f in f10 f20; do
      for d in $INTEGRATIONS; do
        run_cell "fin_adaptive_backdoor_cifar10_${f}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" \
          "$(budget 0.6)" "+defense=${d}" "defense.scorer.weighting=${w}"
      done
    done
  done
}

# holdout_nb <fraction> <budget> <rule override>
holdout_nb() {
  run_cell "fin_holdout_adaptive_backdoor_cifar10_$1_g$2" "adaptive_backdoor_cifar10_$1" "$HOLDOUT_SEEDS" \
    "$(budget "$2")" "$3"
}

# holdout_da <fraction> <cosine> <rule override>
holdout_da() {
  run_cell "fin_holdout_fltrust_backdoor_cifar10_$1_c$2" "fltrust_backdoor_cifar10_$1" "$HOLDOUT_SEEDS" \
    "$(cosine "$2")" "$3"
}

block_B() {
  local f d
  local -a rules=(none "$PEER" "$ROOT" "$MEDIAN" "$FLTRUST" "$SCOREWEIGHT")
  for f in f10 f20; do
    # The score integrations and their undefended reference at budget 0.6.
    holdout_nb "$f" 0.6 none
    for d in $INTEGRATIONS; do
      holdout_nb "$f" 0.6 "+defense=${d}"
    done
    # Norm-bounded backdoor, each rule at the budget selected against it.
    holdout_nb "$f" 1.0 none
    holdout_nb "$f" 0.6 "$FLTRUST"
    if [ "$f" = "f10" ]; then
      holdout_nb "$f" 0.6 "$ROOT"
      holdout_nb "$f" 2.0 "$MEDIAN"
    else
      holdout_nb "$f" 1.0 "$ROOT"
      holdout_nb "$f" 0.6 "$MEDIAN"
    fi
    # Direction-aligned backdoor, each rule at its selected cosine.
    for d in none "$PEER" "$MEDIAN" "$SCOREWEIGHT"; do
      holdout_da "$f" 0.5 "$d"
    done
    for d in "$ROOT" "$FLTRUST"; do
      holdout_da "$f" 0.9 "$d"
    done
    for d in "${rules[@]}"; do
      run_cell "fin_holdout_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$HOLDOUT_SEEDS" "$d"
      run_cell "fin_holdout_composed_backdoor_cifar10_${f}_g0.6_c0.9" "composed_backdoor_cifar10_${f}" "$HOLDOUT_SEEDS" \
        "attack.update_manipulator.manipulators.0.cosine_target=0.9" \
        "attack.update_manipulator.manipulators.1.target_norm=0.6" "$d"
    done
  done
}

block_C_UP() {
  local f d
  for f in f10 f20; do
    run_cell "fin_fltrust_backdoor_cifar10_${f}_c0.9" "fltrust_backdoor_cifar10_${f}" "$SEEDS" "$(cosine 0.9)" "$FLTRUST"
    for d in "$SCOREWEIGHT" none "$PEER" "$MEDIAN"; do
      run_cell "fin_clean_label_cifar10_${f}" "clean_label_cifar10_${f}" "$SEEDS" "$d"
    done
  done
}

block_C_F3040() {
  local f d
  for f in f30 f40; do
    for d in none "$PEER" "$MEDIAN" "$ROOT"; do
      run_cell "fin_adaptive_backdoor_cifar10_${f}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" "$(budget 0.6)" "$d"
      run_cell "fin_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$SEEDS" "$d"
    done
  done
}

block_C_ROOT() {
  local f v suffix
  for f in f10 f20; do
    for v in 100 1 10; do
      suffix="_root${v}"
      [ "$v" = "100" ] && suffix=""
      run_cell "fin_backdoor_cifar10_${f}${suffix}" "backdoor_cifar10_${f}" "$SEEDS" \
        "dataset.val_per_class=${v}" "$ROOT"
      run_cell "fin_adaptive_backdoor_cifar10_${f}${suffix}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" \
        "$(budget 0.6)" "dataset.val_per_class=${v}" "$ROOT"
    done
  done
}

block_C_H500() {
  local f d
  for f in f10 f20; do
    for d in none "$PEER" "$MEDIAN"; do
      run_cell "fin_h500_adaptive_backdoor_cifar10_${f}" "adaptive_backdoor_cifar10_${f}" "$FIVE_SEEDS" \
        "$(budget 0.6)" "federation.num_rounds=500" "$d"
      run_cell "fin_h500_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$FIVE_SEEDS" \
        "federation.num_rounds=500" "$d"
    done
  done
}

# d_da <dataset> <fraction> <cosine> <seeds> <rule override>
d_da() {
  local exp="fltrust_backdoor_cifar100_$2"
  [ "$1" = "femnist" ] && exp="femnist_fltrust_backdoor_$2"
  run_cell "fin_${exp}_c$3" "$exp" "$4" "$(cosine "$3")" "$5"
}

# d_composed <dataset> <fraction> <budget> <cosine> <seeds> <rule override>
d_composed() {
  local exp="composed_backdoor_cifar100_$2"
  [ "$1" = "femnist" ] && exp="femnist_composed_backdoor_$2"
  run_cell "fin_${exp}_g$3_c$4" "$exp" "$5" \
    "attack.update_manipulator.manipulators.0.cosine_target=$4" \
    "attack.update_manipulator.manipulators.1.target_norm=$3" "$6"
}

block_D_PROBE() {
  local ds f c g budgets
  for ds in femnist cifar100; do
    budgets="$D_BUDGETS_CIFAR100"
    [ "$ds" = "femnist" ] && budgets="$D_BUDGETS_FEMNIST"
    for f in f10 f20; do
      for c in $D_COSINES; do
        d_da "$ds" "$f" "$c" "0" none
      done
      for g in $budgets; do
        for c in $D_COMPOSED_COSINES; do
          d_composed "$ds" "$f" "$g" "$c" "0" none
        done
      done
    done
  done
}

block_D_MAIN() {
  local ds f d var knob
  local -a rules=(none "$PEER" "$ROOT" "$MEDIAN" "$FLTRUST")
  for ds in femnist cifar100; do
    for f in f10 f20; do
      var="D_DA_${ds^^}_${f^^}"
      knob="${!var:-}"
      if [ -n "$knob" ]; then
        for d in "${rules[@]}"; do
          d_da "$ds" "$f" "$knob" "$SEEDS" "$d"
        done
      else
        echo "D_MAIN: ${var} unset; skipping the direction-aligned cells of ${ds} ${f}" >&2
      fi
      var="D_COMP_${ds^^}_${f^^}"
      knob="${!var:-}"
      if [ -n "$knob" ]; then
        for d in "${rules[@]}"; do
          d_composed "$ds" "$f" "${knob%,*}" "${knob#*,}" "$SEEDS" "$d"
        done
      else
        echo "D_MAIN: ${var} unset; skipping the composed cells of ${ds} ${f}" >&2
      fi
    done
  done
}

# e_cells <tag> <base experiment> <seeds> <fractions> [attack override ...]
e_cells() {
  local tag="$1" base="$2" seeds="$3" fractions="$4"
  shift 4
  local kind f d pair
  for kind in $E_KINDS; do
    for f in $fractions; do
      for d in $INTEGRATIONS; do
        run_cell "${tag/@F@/$f}" "${base/@F@/$f}" "$seeds" "$@" \
          "+defense=${d}" "+defense.scorer.kind=${kind}"
      done
      for pair in $E_MEDTHR; do
        [ "${pair%%:*}" = "$kind" ] || continue
        run_cell "${tag/@F@/$f}_medthr" "${base/@F@/$f}" "$seeds" "$@" \
          "+defense=${pair#*:}" "+defense.scorer.kind=${kind}" \
          "+defense.integrator.cohort_median=true"
      done
    done
  done
}

block_E_PROBE() {
  e_cells "fin_adaptive_backdoor_cifar10_@F@" "adaptive_backdoor_cifar10_@F@" "0" "f10" "$(budget 0.6)"
}

block_E_NB() {
  e_cells "fin_adaptive_backdoor_cifar10_@F@" "adaptive_backdoor_cifar10_@F@" "$SEEDS" "f10 f20" "$(budget 0.6)"
}

block_E_DA() {
  e_cells "fin_fltrust_backdoor_cifar10_@F@_c0.9" "fltrust_backdoor_cifar10_@F@" "$SEEDS" "f10 f20" "$(cosine 0.9)"
}

block_E_STD() {
  e_cells "fin_backdoor_cifar10_@F@" "backdoor_cifar10_@F@" "$SEEDS" "f10 f20"
}

block_P1_COS() {
  local E_KINDS="server_cosine"
  local c f d
  for c in $P1_COSINES; do
    for f in f10 f20; do
      for d in none "$MEDIAN"; do
        run_cell "fin_fltrust_backdoor_cifar10_${f}_c${c}" "fltrust_backdoor_cifar10_${f}" "$SEEDS" "$(cosine "$c")" "$d"
      done
    done
    e_cells "fin_fltrust_backdoor_cifar10_@F@_c${c}" "fltrust_backdoor_cifar10_@F@" "$SEEDS" "f10 f20" "$(cosine "$c")"
  done
}

block_P1_M4() {
  local f d
  for f in f10 f20; do
    for d in $INTEGRATIONS; do
      run_cell "fin_adaptive_backdoor_cifar10_${f}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" \
        "$(budget 0.6)" "+defense=${d}" "defense.scorer.weighting=m4_only"
    done
  done
}

block_P2() {
  local f d z zmax
  local -a rules=(none "$MEDIAN" "+baseline=trimmed_mean" "$FLTRUST" "$PEER" "$ROOT")
  for f in f10 f20; do
    zmax="0.11"
    [ "$f" = "f20" ] && zmax="0.29"
    [ -n "$P2_ALIE_ONLY" ] && zmax=""
    for d in "${rules[@]}"; do
      [ -n "$P2_ALIE_ONLY" ] ||
        run_cell "fin_minmax_backdoor_cifar10_${f}" "minmax_backdoor_cifar10_${f}" "$SEEDS" "$d"
      for z in $P2_ZS $zmax; do
        run_cell "fin_alie_backdoor_cifar10_${f}_z${z}" "alie_backdoor_cifar10_${f}" "$SEEDS" \
          "attack.update_manipulator.num_std=${z}" "$d"
      done
    done
  done
}

block_P4() {
  local GEO="+defense=topsis_capped_weighted_geomedian"
  local GEO_UNCAPPED="+defense=topsis_weighted_geomedian"
  local f d c
  block_REF_STD
  for f in f10 f20; do
    run_cell "fin_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$SEEDS" "$GEO"
    run_cell "fin_backdoor_cifar10_${f}" "backdoor_cifar10_${f}" "$SEEDS" "$GEO_UNCAPPED"
    run_cell "fin_adaptive_backdoor_cifar10_${f}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" "$(budget 0.6)" "$GEO"
    run_cell "fin_adaptive_backdoor_cifar10_${f}" "adaptive_backdoor_cifar10_${f}" "$SEEDS" "$(budget 0.6)" "$GEO_UNCAPPED"
    for d in "$GEO" none "$PEER" "$MEDIAN"; do
      run_cell "fin_adaptive_backdoor_cifar10_${f}_g1.0" "adaptive_backdoor_cifar10_${f}" "$SEEDS" "$(budget 1.0)" "$d"
      for c in 0.5 0.9; do
        run_cell "fin_fltrust_backdoor_cifar10_${f}_c${c}" "fltrust_backdoor_cifar10_${f}" "$SEEDS" "$(cosine "$c")" "$d"
      done
    done
    for d in "$GEO" none "$PEER" "$MEDIAN"; do
      run_cell "fin_clean_label_cifar10_${f}" "clean_label_cifar10_${f}" "$SEEDS" "$d"
    done
  done
}

block_P4_COST() {
  local d
  for d in none "$MEDIAN" "$PEER" "+defense=topsis_capped_weighted_geomedian"; do
    run_cell "fin_overhead_cifar10" "overhead_cifar10" "0" "$d"
  done
}

# composed_overrides <budget> <cosine>
composed_overrides() {
  echo "attack.update_manipulator.manipulators.0.cosine_target=$2"
  echo "attack.update_manipulator.manipulators.1.target_norm=$1"
}

block_P1_FGC() {
  local E_KINDS="history_similarity"
  local knob g c f d tag
  for knob in $P1_FGC_KNOBS; do
    g="${knob%,*}"
    c="${knob#*,}"
    tag="fin_composed_backdoor_cifar10_@F@_g${g}_c${c}"
    for f in f10 f20; do
      for d in none "$MEDIAN"; do
        # shellcheck disable=SC2046
        run_cell "${tag/@F@/$f}" "composed_backdoor_cifar10_${f}" "$SEEDS" \
          $(composed_overrides "$g" "$c") "$d"
      done
    done
    # shellcheck disable=SC2046
    e_cells "$tag" "composed_backdoor_cifar10_@F@" "$SEEDS" "f10 f20" \
      $(composed_overrides "$g" "$c")
  done
}

block_P3_SCAN() {
  local f d g c tag
  local -a rules=("$PEER" "$ROOT" "$MEDIAN" "$FLTRUST")
  for f in f10 f20; do
    for d in "${rules[@]}"; do
      for g in $P3_BUDGETS; do
        tag="fin_adaptive_backdoor_cifar10_${f}_g${g}"
        [ "$g" = "0.6" ] && tag="fin_adaptive_backdoor_cifar10_${f}"
        run_cell "$tag" "adaptive_backdoor_cifar10_${f}" "$SEEDS" "$(budget "$g")" "$d"
      done
      for c in $P1_COSINES; do
        run_cell "fin_fltrust_backdoor_cifar10_${f}_c${c}" "fltrust_backdoor_cifar10_${f}" "$SEEDS" "$(cosine "$c")" "$d"
      done
      for g in $P3_COMPOSED_BUDGETS; do
        for c in $P3_COMPOSED_COSINES; do
          # shellcheck disable=SC2046
          run_cell "fin_composed_backdoor_cifar10_${f}_g${g}_c${c}" "composed_backdoor_cifar10_${f}" "$SEEDS" \
            $(composed_overrides "$g" "$c") "$d"
        done
      done
    done
  done
}

# p3_eval_cell <attack NB|DA|COMP> <fraction> <knob> <rule override>
p3_eval_cell() {
  local attack="$1" f="$2" knob="$3" d="$4"
  case "$attack" in
    NB) run_cell "fin_holdout_adaptive_backdoor_cifar10_${f}_g${knob}" "adaptive_backdoor_cifar10_${f}" "$HOLDOUT_SEEDS" "$(budget "$knob")" "$d" ;;
    DA) run_cell "fin_holdout_fltrust_backdoor_cifar10_${f}_c${knob}" "fltrust_backdoor_cifar10_${f}" "$HOLDOUT_SEEDS" "$(cosine "$knob")" "$d" ;;
    COMP)
      # shellcheck disable=SC2046
      run_cell "fin_holdout_composed_backdoor_cifar10_${f}_g${knob%,*}_c${knob#*,}" "composed_backdoor_cifar10_${f}" "$HOLDOUT_SEEDS" \
        $(composed_overrides "${knob%,*}" "${knob#*,}") "$d" ;;
  esac
}

block_P3_EVAL() {
  local attack f rule var knob seen
  for attack in NB DA COMP; do
    for f in f10 f20; do
      seen=" "
      for rule in PEER ROOT MEDIAN FLTRUST; do
        var="P3_${attack}_${rule}_${f^^}"
        knob="${!var:-}"
        if [ -z "$knob" ]; then
          echo "P3_EVAL: ${var} unset; skipping" >&2
          continue
        fi
        p3_eval_cell "$attack" "$f" "$knob" "${!rule}"
        case "$seen" in
          *" ${knob} "*) ;;
          *) seen="${seen}${knob} "; p3_eval_cell "$attack" "$f" "$knob" none ;;
        esac
      done
    done
  done
}

block_G_PROBE() {
  local a f d
  for d in none "$MEDIAN"; do
    run_cell "fin_benign_cifar10" "benign_cifar10" "0" "$d"
    for a in lie fang; do
      for f in f10 f20; do
        run_cell "fin_${a}_cifar10_${f}" "${a}_cifar10_${f}" "0" "$d"
      done
    done
  done
}

block_G_MAIN() {
  local a f d
  for d in none "$PEER" "$ROOT" "$MEDIAN" "+baseline=trimmed_mean" "$FLTRUST"; do
    run_cell "fin_benign_cifar10" "benign_cifar10" "$FIVE_SEEDS" "$d"
    for a in $G_ATTACKS; do
      for f in f10 f20; do
        run_cell "fin_${a}_cifar10_${f}" "${a}_cifar10_${f}" "$FIVE_SEEDS" "$d"
      done
    done
  done
}

for block in $BLOCKS; do
  case "$block" in
    REF_NB|REF_DA|REF_STD|A|B|C_UP|C_F3040|C_ROOT|C_H500|D_PROBE|D_MAIN|E_PROBE|E_NB|E_DA|E_STD|G_PROBE|G_MAIN|P1_COS|P1_M4|P1_FGC|P2|P3_SCAN|P3_EVAL|P4|P4_COST) "block_${block}" ;;
    *) echo "unknown block: ${block}" >&2; exit 2 ;;
  esac
done
echo "########### FINAL SWEEP COMPLETE (${BLOCKS}) ###########"
