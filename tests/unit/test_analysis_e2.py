"""E2 analysis: cell keys, curve quantities, the budget rule, and the report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from topfed.analysis.e2 import (
    CIFAR_FRACTIONS,
    FEMNIST_FRACTIONS,
    CellKey,
    RunSummary,
    breaking_fraction,
    budget_rescan,
    build_report,
    bump_targets,
    cell_key,
    experiment_name,
    hypergeom_pmf,
    load_cells,
    suppression_by_m,
    main,
    oracle_verdict,
    sampling_tail,
    summarise_run,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
N_ROUNDS = 200

CAPPED = {"scorer": {"weighting": "critic"},
          "integrator": {"_target_": "topfed.integration.mechanisms.CappedScoreWeight",
                         "cap": 1.0, "alpha": 1.0},
          "base": "weighted_median"}
ORACLE = {"scorer": {"weighting": "critic"},
          "integrator": {"_target_": "topfed.integration.oracle.OracleFilter"},
          "base": "weighted_median"}
ADAPTIVE = {"_target_": "topfed.attacks.model.adaptive.NormBoundedManipulator", "target_norm": 0.6}


def _config(**over):
    cfg = {
        "dataset": {"name": "cifar10", "augment": True},
        "partition": {"kind": "dirichlet", "alpha": 0.5, "seed": 0},
        "federation": {"num_clients": 100, "clients_per_round": 10, "num_rounds": N_ROUNDS},
        "attack": {
            "kind": "backdoor",
            "poisoner": {
                "_target_": "topfed.attacks.data.backdoor.BackdoorPoisoner",
                "target_class": 9,
                "trigger_size": 6,
                "trigger_amplitude": 1.0,
                "batch_mode": "augment",
                "poisoning_rate": 0.5,
            },
            "loss_augmenter": {"_target_": "x.ConstrainAndScaleAugmenter", "alpha": 0.5},
            "update_manipulator": {"_target_": "x.ModelReplacementManipulator", "gamma": None},
            "schedule": {"_target_": "topfed.attacks.schedule.AlwaysAttack"},
            "assignment": {"fraction": 0.2},
        },
        "experiment": "backdoor_cifar10_f20",
    }
    for dotted, value in over.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def _femnist(**over):
    return _config(**{"dataset.name": "femnist", "partition": {"kind": "natural", "seed": 0},
                      "attack.poisoner.trigger_amplitude": -5.0, **over})


def _key(cfg) -> CellKey | None:
    return cell_key({"seed": 0, "config": cfg})


def _defense(cfg) -> str:
    key = _key(cfg)
    assert key is not None
    return key.defense


# ----------------------------------------------------------------------------
# Cell keys
# ----------------------------------------------------------------------------


def test_cell_key_names_attacker_fraction_and_defense() -> None:
    assert _key(_config()) == CellKey("cifar10", "backdoor", None, 0.2, "undefended")
    adaptive = _key(_config(**{"attack.update_manipulator": ADAPTIVE,
                               "attack.assignment.fraction": 0.3, "defense": CAPPED}))
    assert adaptive == CellKey("cifar10", "adaptive", 0.6, 0.3, "capped-wmed")
    assert _defense(_config(defense=ORACLE)) == "oracle-wmed"
    assert _defense(_config(baseline={"name": "fltrust"})) == "fltrust"
    assert _defense(_config(baseline={"name": "coordinate_median"})) == "median"
    assert _key(_femnist()) == CellKey("femnist", "backdoor", None, 0.2, "undefended")
    gs = _config()
    gs["attack"] = {"kind": "gradient_scaling", "poisoner": None,
                    "update_manipulator": {"_target_": "x.GradientScalingManipulator", "factor": 5.0},
                    "schedule": {"_target_": "topfed.attacks.schedule.AlwaysAttack"},
                    "assignment": {"fraction": 0.05}}
    assert _key(gs) == CellKey("cifar10", "gradient_scaling", 5.0, 0.05, "undefended")


@pytest.mark.parametrize(
    "over",
    [
        {"federation.force_malicious_per_round": 5},
        {"federation.clients_per_round": 20},
        {"federation.num_rounds": 8},
        {"partition": {"kind": "iid", "seed": 0}},
        {"dataset.pad_border": 4},
        {"dataset.val_per_class": 10},
        {"attack.poisoner.trigger_position": "center"},
        {"attack.poisoner.trigger_size": 4},
        {"attack.poisoner.trigger_amplitude": -2.0},
        {"attack.poisoner.batch_mode": "replace"},
        {"attack.update_manipulator": None},
        {"attack.update_manipulator": {"_target_": "x.ModelReplacementManipulator", "gamma": 10.0}},
        {"attack.schedule": {"_target_": "topfed.attacks.schedule.WindowedAttack",
                             "start_round": 1, "end_round": 100}},
        {"attack.kind": "sign_flip"},
    ],
)
def test_diagnostic_variants_are_out_of_scope(over) -> None:
    assert _key(_config(**over)) is None


def test_defense_sweeps_do_not_merge_into_the_curve() -> None:
    swept = {**CAPPED, "integrator": {"_target_": "topfed.integration.mechanisms.CappedScoreWeight",
                                      "cap": 0.5, "alpha": 1.0}}
    assert _defense(_config(defense=swept)) == "capped-wmed(cap=0.5)"


# ----------------------------------------------------------------------------
# Per-run summary
# ----------------------------------------------------------------------------


def _client_rows(mal_by_round: dict[int, int], n: int, k: int = 10, flip_after: int = 25):
    """Attackers score high on every metric through ``flip_after``, low after."""
    rows = []
    for r in range(1, n + 1):
        m = mal_by_round.get(r, 1)
        for i in range(k):
            mal = i < m
            high = (r <= flip_after) == mal
            value = 1.0 + i * 0.01 if high else -1.0 - i * 0.01
            rows.append({"round": r, "client_id": f"c{i}", "is_malicious": mal,
                         **{f"m{j}_{name}": value for j, name in (
                             (1, "cosine_to_mean"), (2, "norm_ratio"), (3, "temporal_cv"),
                             (4, "val_loss_delta"), (5, "max_layer_deviation"))}})
    return rows


def _metrics(asr: np.ndarray | None, acc: float = 0.7):
    out = []
    for r in range(N_ROUNDS):
        row = {"round": r + 1, "test_accuracy": acc}
        if asr is not None:
            row["asr_accuracy"] = float(asr[r])
        out.append(row)
    return out


def test_summarise_run_counts_windows_and_capture() -> None:
    asr = np.full(N_ROUNDS, 0.05)
    asr[39:] = 0.95
    rows = _client_rows({40: 6, 41: 5}, N_ROUNDS)
    s = summarise_run(0, 20, _metrics(asr), rows, None)
    assert s.asr_final == pytest.approx(0.95)
    assert s.acc_final == pytest.approx(0.7)
    assert s.implant_round == 40
    assert s.first_majority_round == 40  # 6/10 with equal weights
    assert s.max_malicious_share == pytest.approx(0.6)
    assert s.malicious_count_hist[1] == N_ROUNDS - 2
    assert s.malicious_count_hist[5] == 1 and s.malicious_count_hist[6] == 1
    assert sum(s.malicious_count_hist) == N_ROUNDS
    assert s.auc["early"]["m4_val_loss_delta"] == pytest.approx(1.0)
    assert s.auc["late"]["m4_val_loss_delta"] == pytest.approx(0.0)


def test_share_by_m_and_suppression_ratio() -> None:
    asr = np.zeros(N_ROUNDS)
    rows = _client_rows({r: 2 for r in range(1, 11)}, N_ROUNDS)  # m=2 in rounds 1-10, else 1
    ledger = []
    for r in rows:
        # Attackers get half their equal weight: share = 0.5 * m/K.
        ledger.append({"round": r["round"], "client_id": r["client_id"],
                       "weight": 0.05 if r["is_malicious"] else 0.1, "is_malicious": r["is_malicious"]})
    s = summarise_run(0, 20, _metrics(asr), rows, ledger)
    assert len(s.share_sum_by_m) == 11
    assert s.share_sum_by_m[2] == pytest.approx(10 * 0.1 / (0.1 + 0.8))
    ratios = suppression_by_m([s])
    assert ratios[1] == pytest.approx((0.05 / 0.95) / 0.1)
    assert ratios[2] == pytest.approx((0.1 / 0.9) / 0.2)
    assert 3 not in ratios
    no_ledger = summarise_run(0, 20, _metrics(asr), rows, None)
    assert no_ledger.share_sum_by_m == [] and suppression_by_m([no_ledger]) == {}


def test_summarise_run_untargeted_has_no_asr() -> None:
    s = summarise_run(0, 20, _metrics(None, acc=0.4), _client_rows({}, N_ROUNDS), None)
    assert s.asr_final is None and s.implant_round is None
    assert s.acc_final == pytest.approx(0.4)


# ----------------------------------------------------------------------------
# Curve quantities
# ----------------------------------------------------------------------------


def _summary(seed: int, asr: float | None, acc: float = 0.7, hist=None) -> RunSummary:
    return RunSummary(
        seed=seed, num_malicious=0, asr_final=asr, acc_final=acc, implant_round=None,
        first_majority_round=None, max_malicious_share=0.0,
        malicious_count_hist=hist or [N_ROUNDS] + [0] * 10,
    )


def _grid(asr_by_f: dict[float, float], attack="backdoor", strength=None, defense="capped-wmed",
          seeds=range(5)):
    return {
        CellKey("cifar10", attack, strength, round(f, 4), defense): [_summary(s, a) for s in seeds]
        for f, a in asr_by_f.items()
    }


def test_breaking_fraction_first_crossing_and_completeness() -> None:
    full = dict(zip(CIFAR_FRACTIONS, (0.02, 0.04, 0.3, 0.45, 0.6, 0.8, 0.9)))
    cells = _grid(full)
    assert breaking_fraction(cells, "cifar10", "backdoor", None, "capped-wmed", CIFAR_FRACTIONS) == (0.25, True)
    gappy = {f: a for f, a in full.items() if f != 0.05}
    assert breaking_fraction(_grid(gappy), "cifar10", "backdoor", None, "capped-wmed",
                             CIFAR_FRACTIONS) == (0.25, False)
    never = _grid({f: 0.1 for f in CIFAR_FRACTIONS})
    assert breaking_fraction(never, "cifar10", "backdoor", None, "capped-wmed", CIFAR_FRACTIONS) == (None, True)


def test_bump_targets_only_new_complete_underseeded_levels() -> None:
    new_break = dict(zip(CIFAR_FRACTIONS, (0.02, 0.04, 0.3, 0.45, 0.6, 0.8, 0.9)))
    frozen_break = dict(zip(CIFAR_FRACTIONS, (0.02, 0.04, 0.3, 0.6, 0.7, 0.8, 0.9)))
    cells = {
        **_grid(new_break, defense="capped-wmed"),
        **_grid(frozen_break, defense="fltrust"),
        **_grid(new_break, attack="adaptive", strength=0.6, defense="capped-wmed", seeds=range(10)),
        **_grid({f: a for f, a in new_break.items() if f != 0.15}, attack="adaptive",
                strength=0.6, defense="fltrust"),
    }
    assert bump_targets(cells) == [
        ("backdoor_cifar10_f25", "+defense=topsis_capped_weighted_median"),
    ]


def test_experiment_names_exist_for_every_grid_level() -> None:
    for f in CIFAR_FRACTIONS:
        for attack in ("backdoor", "adaptive", "gradient_scaling"):
            assert (CONFIG_DIR / "experiment" / f"{experiment_name('cifar10', attack, f)}.yaml").exists()
    for f in FEMNIST_FRACTIONS:
        assert (CONFIG_DIR / "experiment" / f"{experiment_name('femnist', 'backdoor', f)}.yaml").exists()
    assert experiment_name("cifar10", "adaptive", 0.05) == "adaptive_backdoor_cifar10_f05"


def test_budget_rescan_max_rule() -> None:
    def cells_for(asr_by_budget):
        return {CellKey("cifar10", "adaptive", g, 0.3, "capped-wmed"): [_summary(0, a), _summary(1, 0.0)]
                for g, a in asr_by_budget.items()}

    beaten = budget_rescan(cells_for({0.6: 0.4, 1.0: 0.7, 1.5: 0.5}), 0.3, "capped-wmed")
    assert beaten.best_budget == 1.0 and beaten.best_asr == pytest.approx(0.7)
    assert beaten.understated
    tie = budget_rescan(cells_for({0.6: 0.5, 1.0: 0.5, 1.5: 0.2}), 0.3, "capped-wmed")
    assert tie.best_budget == 0.6 and not tie.understated
    missing_ref = budget_rescan(cells_for({1.0: 0.3}), 0.3, "capped-wmed")
    assert missing_ref.asr_by_budget[0.6] is None and missing_ref.understated
    empty = budget_rescan({}, 0.3, "capped-wmed")
    assert empty.best_budget is None and not empty.understated


def test_oracle_verdict() -> None:
    assert oracle_verdict(0.8, 0.9) == "aggregation-limited"
    assert oracle_verdict(0.1, 0.9) == "detection-limited"
    assert oracle_verdict(0.1, 0.2) == "holds"
    assert oracle_verdict(None, 0.9) == "–"
    assert oracle_verdict(0.1, None) == "–"


def test_hypergeometric_tail_and_observed_counts() -> None:
    assert sum(hypergeom_pmf(100, 20, 10, m) for m in range(11)) == pytest.approx(1.0)
    hist_a = [100, 90, 0, 0, 0, 10] + [0] * 5
    hist_b = [150, 50] + [0] * 9
    s_a = _summary(0, 0.1, hist=hist_a)
    s_b = _summary(1, 0.1, hist=hist_b)
    s_a.num_malicious = s_b.num_malicious = 20
    # Seed 0 appears in two cells; it is counted once.
    cells = {
        CellKey("cifar10", "backdoor", None, 0.2, "undefended"): [s_a, s_b],
        CellKey("cifar10", "backdoor", None, 0.2, "capped-wmed"): [s_a],
    }
    (row,) = sampling_tail(cells, "cifar10", (0.2,))
    assert row.num_malicious == 20
    assert row.expected_per_round == pytest.approx(2.0)
    assert row.p_round == pytest.approx(0.0255, abs=5e-4)
    assert row.p_any == pytest.approx(1 - (1 - row.p_round) ** N_ROUNDS)
    assert row.observed_round_frac == pytest.approx(10 / 400)
    assert row.observed_seeds_with_majority == (1, 2)
    assert row.max_m_observed == 5
    assert row.observed_pmf[0] == pytest.approx(250 / 400)
    (empty,) = sampling_tail({}, "cifar10", (0.15,))
    assert empty.num_malicious == 15 and empty.observed_round_frac is None


# ----------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------


def _write_run(root: Path, label: str, cfg_hash: str, seed: int, cfg: dict, asr_value: float,
               mal_by_round: dict[int, int] | None = None, ledger: bool = False) -> None:
    d = root / label / cfg_hash / f"seed_{seed}"
    d.mkdir(parents=True)
    fraction = cfg["attack"]["assignment"]["fraction"]
    (d / "manifest.json").write_text(json.dumps(
        {"seed": seed, "config": cfg, "num_malicious": math.ceil(100 * fraction - 1e-9)}))
    has_asr = cfg["attack"]["kind"] == "backdoor"
    (d / "metrics.json").write_text(json.dumps(
        _metrics(np.full(N_ROUNDS, asr_value) if has_asr else None)))
    rows = _client_rows(mal_by_round or {}, N_ROUNDS)
    (d / "client_metrics.json").write_text(json.dumps(rows))
    if ledger:
        (d / "defense_ledger.json").write_text(json.dumps(
            [{"round": r["round"], "client_id": r["client_id"], "weight": 0.1,
              "is_malicious": r["is_malicious"]} for r in rows]))


def _populate(results: Path) -> None:
    for f, asr in ((0.05, 0.02), (0.10, 0.04), (0.15, 0.3), (0.20, 0.45), (0.25, 0.6),
                   (0.30, 0.8), (0.40, 0.9)):
        level = f"f{round(f * 100):02d}"
        tagged = f not in (0.10, 0.20)
        prefix = "e2_" if tagged else ""
        cfg = _config(**{"attack.assignment.fraction": f, "defense": CAPPED})
        if tagged:
            cfg["tag"] = f"e2_backdoor_cifar10_{level}"
        for seed in range(10 if not tagged else 5):
            _write_run(results, f"{prefix}backdoor_cifar10_{level}__CappedScoreWeight_critic_wmedian",
                       "aaaa", seed, cfg, asr, {30: 6} if f >= 0.2 else None, ledger=True)
        oracle_cfg = _config(**{"attack.assignment.fraction": f, "defense": ORACLE,
                                "tag": f"e2_backdoor_cifar10_{level}"})
        for seed in range(2):
            _write_run(results, f"e2_backdoor_cifar10_{level}__OracleFilter_critic_wmedian",
                       "oooo", seed, oracle_cfg, 0.9 if f >= 0.4 else 0.03, ledger=True)
    # Budget rescan at f=0.30: 1.0 beats the main-grid 0.6 on seed 0.
    for g, asr in ((0.6, 0.5), (1.0, 0.75), (1.5, 0.2)):
        cfg = _config(**{"attack.assignment.fraction": 0.3, "defense": CAPPED,
                         "attack.update_manipulator": {**ADAPTIVE, "target_norm": g}})
        _write_run(results, f"e2_adaptive_backdoor_cifar10_f30_g{g}__CappedScoreWeight_critic_wmedian",
                   "gggg", 0, cfg, asr, ledger=True)
    # The same frozen config under a second, smaller directory: not double-counted.
    dup = _config(**{"attack.assignment.fraction": 0.1, "defense": CAPPED, "experiment": "ablation"})
    _write_run(results, "ablation__CappedScoreWeight_critic_wmedian", "dddd", 0, dup, 0.99, ledger=True)
    # FEMNIST cliff cell and an out-of-scope forced-m run.
    for seed in range(5):
        _write_run(results, "e2_femnist_backdoor_f15__CappedScoreWeight_critic_wmedian", "ffff", seed,
                   _femnist(**{"attack.assignment.fraction": 0.15, "defense": CAPPED}),
                   0.6, {50: 6}, ledger=True)
    _write_run(results, "e5d3_x", "eeee", 0,
               _config(**{"federation.force_malicious_per_round": 5}), 0.99)
    # A crashed run (no metrics.json) is skipped.
    crashed = results / "e2_crashed" / "cccc" / "seed_0"
    crashed.mkdir(parents=True)
    (crashed / "manifest.json").write_text(json.dumps({"seed": 0, "config": _config()}))


def test_load_cells_report_and_cli(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _populate(results)
    cells = load_cells(results)

    frozen = cells[CellKey("cifar10", "backdoor", None, 0.1, "capped-wmed")]
    assert len(frozen) == 10
    assert frozen[0].asr_final == pytest.approx(0.04)  # the 10-seed directory wins
    assert CellKey("femnist", "backdoor", None, 0.15, "capped-wmed") in cells
    assert all(k.defense != "undefended" or k.fraction != 0.2 for k in cells)  # forced-m excluded

    report = build_report(cells)
    assert "## Sampling tail" in report
    assert "## Suppression vs co-selected attackers" in report
    assert "| cifar10 | vanilla backdoor | capped-wmed | 0.2 |" in report
    assert "| capped-wmed | 0.020 ± 0.000 (n=5) | 0.040 ± 0.000 (n=10) |" in report
    assert "| 0.25 |" in report and "breaking f" in report
    assert "0.800 ± 0.000 (n=5) | 0.900 ± 0.000 (n=5) | 0.25 |" in report
    # Oracle holds below 0.4 while capped-wmed fails at 0.25 and 0.30.
    assert "| vanilla backdoor | 0.25 | 2.5 |" in report
    assert "detection-limited" in report and "aggregation-limited" in report
    assert "| 0.3 | capped-wmed | 0.500 | 0.750 | 0.200 | 1 | 0.750 |" in report
    assert "| capped-wmed | 0.15 | 5 | 0.600 ± 0.000 |" in report

    out = tmp_path / "e2.md"
    dump = tmp_path / "e2.json"
    figs = tmp_path / "figs"
    assert main(["--results-root", str(results), "--out", str(out), "--json", str(dump),
                 "--fig-dir", str(figs)]) == 0
    assert out.read_text() == report
    records = json.loads(dump.read_text())
    assert len(records) == sum(len(v) for v in cells.values())
    assert (figs / "e2-oracle-vs-capped.png").exists()
    assert (figs / "e2-asr-vs-f-cifar10-backdoor.png").exists()
    assert (figs / "e2-malicious-count-hist.png").exists()
    assert (figs / "e2-suppression-vs-m.png").exists()


def test_breaking_f_cli_prints_sweep_lines(tmp_path: Path, capsys) -> None:
    results = tmp_path / "results"
    _populate(results)
    assert main(["--results-root", str(results), "--breaking-f"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["backdoor_cifar10_f25 +defense=topsis_capped_weighted_median"]


def test_main_refuses_to_write_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for flag in ("--out", "--json", "--fig-dir"):
        with pytest.raises(SystemExit):
            main(["--results-root", str(results), flag, str(results / "x")])


@pytest.mark.parametrize(
    ("experiment", "overrides", "expected"),
    [
        ("backdoor_cifar10_f05", ["+baseline=fltrust"],
         CellKey("cifar10", "backdoor", None, 0.05, "fltrust")),
        ("adaptive_backdoor_cifar10_f40", ["+defense=topsis_capped_weighted_median",
                                           "attack.update_manipulator.target_norm=1.5"],
         CellKey("cifar10", "adaptive", 1.5, 0.4, "capped-wmed")),
        ("adaptive_backdoor_cifar10_f25", ["+defense=oracle_filter_wmedian"],
         CellKey("cifar10", "adaptive", 0.6, 0.25, "oracle-wmed")),
        ("femnist_backdoor_f15", ["+defense=topsis_weighted_median"],
         CellKey("femnist", "backdoor", None, 0.15, "wmed")),
        ("gradient_scaling_cifar10_f30", [],
         CellKey("cifar10", "gradient_scaling", 5.0, 0.3, "undefended")),
    ],
)
def test_sweep_configs_compose_into_the_expected_cells(experiment, overrides, expected) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config",
                      overrides=[f"+experiment={experiment}", f"+tag=e2_{experiment}", *overrides])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert cell_key({"seed": 0, "config": resolved}) == expected
