"""E5 analysis: cell inference, capture/recovery detection, and the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topfed.analysis.e5 import (
    CellKey,
    build_report,
    cell_key,
    hypergeom_tail,
    load_cells,
    main,
    summarise_run,
    sustained_implant,
)


def _config(**over):
    cfg = {
        "dataset": {"name": "femnist", "val_per_class": 30, "augment": False},
        "partition": {"kind": "natural", "seed": 0},
        "federation": {"num_clients": 100, "clients_per_round": 10, "num_rounds": 200},
        "attack": {
            "kind": "backdoor",
            "poisoner": {
                "_target_": "topfed.attacks.data.backdoor.BackdoorPoisoner",
                "target_class": 9,
                "trigger_size": 6,
                "trigger_amplitude": -5.0,
                "batch_mode": "augment",
                "poisoning_rate": 0.5,
            },
            "update_manipulator": {"_target_": "x.ModelReplacementManipulator", "gamma": None},
            "schedule": {"_target_": "topfed.attacks.schedule.AlwaysAttack"},
            "assignment": {"fraction": 0.2},
        },
        "experiment": "femnist_backdoor_f20",
    }
    for dotted, value in over.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def _key(**over) -> CellKey:
    key = cell_key({"seed": 0, "config": _config(**over)})
    assert key is not None
    return key


def test_reference_cell_is_ref_block() -> None:
    assert _key().block == "REF"
    assert _key().defense == "undefended"
    cifar = _key(**{"dataset.name": "cifar10", "partition.kind": "dirichlet",
                    "partition.alpha": 0.5, "attack.poisoner.trigger_amplitude": 1.0})
    assert cifar.block == "REF"


def test_block_inference_from_config_knobs() -> None:
    assert _key(**{"federation.force_malicious_per_round": 5}).block == "D3"
    assert _key(**{"attack.poisoner.trigger_position": "center"}).block == "D8"
    assert _key(**{"dataset.pad_border": 4, "attack.poisoner.trigger_size": 4,
                   "attack.poisoner.trigger_position": "border"}).block == "D8"
    assert _key(**{"attack.schedule": {"_target_": "topfed.attacks.schedule.WindowedAttack",
                                       "start_round": 1, "end_round": 100}}).block == "D7"
    assert _key(**{"attack.schedule": {"_target_": "topfed.attacks.schedule.WindowedAttack",
                                       "start_round": 50}}).block == "D2"
    assert _key(**{"federation.clients_per_round": 20}).block == "D9"
    assert _key(**{"defense": {"scorer": {"weighting": "critic"},
                               "integrator": {"_target_": "topfed.integration.mechanisms.TemporalReputation"},
                               "base": "weighted_median"}}).block == "D9"
    oracle = _key(**{"defense": {"scorer": {"weighting": "critic"},
                                 "integrator": {"_target_": "topfed.integration.oracle.OracleFilter"},
                                 "base": "weighted_median"}})
    assert oracle.block == "D1" and oracle.defense == "oracle-wmed"
    assert _key(**{"partition.kind": "iid"}).block == "D4"
    assert _key(**{"partition.kind": "dirichlet", "partition.alpha": 0.1}).partition == "dirichlet(0.1)"
    assert _key(**{"attack.update_manipulator": None}).block == "D6"
    assert _key(**{"attack.poisoner.trigger_amplitude": -2.0}).block == "D6"
    assert _key(**{"attack.poisoner.batch_mode": "replace"}).block == "D6"
    capped = _key(**{"defense": {"scorer": {"weighting": "critic"},
                                 "integrator": {"_target_": "topfed.integration.mechanisms.CappedScoreWeight", "cap": 1.0},
                                 "base": "weighted_median"}})
    assert capped.defense == "capped-wmed" and capped.block == "REF"
    assert _key(**{"baseline": {"name": "coordinate_median"}}).defense == "median"


def test_sweeps_and_adaptive_attackers_do_not_merge_into_reference() -> None:
    capped = {"scorer": {"weighting": "critic"},
              "integrator": {"_target_": "topfed.integration.mechanisms.CappedScoreWeight",
                             "cap": 0.5, "alpha": 1.0},
              "base": "weighted_median"}
    swept = _key(defense=capped)
    assert swept.defense == "capped-wmed(cap=0.5)"
    adaptive = _key(**{"attack.update_manipulator": {
        "_target_": "topfed.attacks.model.adaptive.NormBoundedManipulator", "target_norm": 0.6}})
    assert adaptive.attack_variant == "NormBoundedManipulator(target_norm=0.6)"
    assert not adaptive.is_vanilla_attack and adaptive.scaling
    vanilla = _key()
    assert vanilla.attack_variant == "model_replacement" and vanilla.is_vanilla_attack
    off = _key(**{"attack.update_manipulator": None})
    assert off.attack_variant == "none" and off.is_vanilla_attack and not off.scaling


def test_non_backdoor_runs_are_ignored() -> None:
    cfg = _config()
    cfg["attack"] = {"kind": "gradient_scaling", "poisoner": None, "assignment": {"fraction": 0.2}}
    assert cell_key({"seed": 0, "config": cfg}) is None


def test_sustained_implant_ignores_short_spikes() -> None:
    asr = np.zeros(40)
    asr[10:15] = 1.0  # 5-round spike: not sustained
    asr[20:] = 0.9
    assert sustained_implant(asr) == 21
    assert sustained_implant(np.zeros(40)) is None


def _rows(mal_by_round: dict[int, int], n: int, k: int = 10):
    rows = []
    for r in range(1, n + 1):
        m = mal_by_round.get(r, 0)
        for i in range(k):
            rows.append({"round": r, "client_id": f"c{i}", "is_malicious": i < m})
    return rows


def test_summarise_run_detects_one_round_capture_and_no_recovery() -> None:
    n = 60
    asr = np.full(n, 0.05)
    asr[8:] = 0.98  # captured in round 9, never recovers
    metrics = [{"round": r + 1, "asr_accuracy": float(asr[r]), "test_accuracy": 0.8,
                "client_train_s": 1.0, "round_s": 1.5} for r in range(n)]
    rows = _rows({9: 6, 30: 2}, n)
    # Ledger: round 9 the six attackers hold 0.58 of the weight.
    ledger = []
    for r in range(1, n + 1):
        m = 6 if r == 9 else (2 if r == 30 else 0)
        for i in range(10):
            mal = i < m
            w = (0.58 / 6 if mal else 0.42 / 4) if r == 9 else 0.1
            ledger.append({"round": r, "client_id": f"c{i}", "weight": w, "is_malicious": mal})
    s = summarise_run(0, metrics, rows, ledger, k=10, window=(1, None))
    assert s.implant_round == 9
    assert s.first_cross == 9
    assert s.m_at_implant == 6
    assert s.share_at_implant == pytest.approx(0.58)
    assert s.first_majority_round == 9 and s.majority_rounds == 1
    assert s.max_malicious_share == pytest.approx(0.58)
    assert s.recovered is False
    assert s.asr_final == pytest.approx(0.98)
    assert s.timings == {"client_train_s": 1.0, "round_s": 1.5}


def test_summarise_run_detects_recovery_and_equal_share_fallback() -> None:
    n = 60
    asr = np.full(n, 0.05)
    asr[8:12] = 0.7  # short capture, then recovers
    metrics = [{"round": r + 1, "asr_accuracy": float(asr[r]), "test_accuracy": 0.7} for r in range(n)]
    rows = _rows({9: 5}, n)
    s = summarise_run(0, metrics, rows, None, k=10, window=(1, 100))
    assert s.implant_round is None
    assert s.first_cross == 9
    assert s.recovered is True
    assert s.asr_after_cross == pytest.approx(0.05)
    assert s.first_majority_round == 9  # 5/10 = 0.5 with equal weights
    assert s.max_m == 5
    assert "asr_at_window_end" not in s.asr_at  # window end beyond the run


def test_summarise_run_window_markers() -> None:
    n = 120
    asr = np.zeros(n)
    asr[99] = 0.9
    metrics = [{"round": r + 1, "asr_accuracy": float(asr[r]), "test_accuracy": 0.7} for r in range(n)]
    s = summarise_run(0, metrics, _rows({}, n), None, k=10, window=(1, 100))
    assert s.asr_at["asr_at_window_end"] == pytest.approx(0.9)
    assert s.asr_at["asr_window_end_plus_50"] == pytest.approx(0.0)  # clamps to last round
    s2 = summarise_run(0, metrics, _rows({}, n), None, k=10, window=(50, None))
    assert s2.asr_at["asr_before_start"] == pytest.approx(0.0)


def test_hypergeom_tail_matches_known_values() -> None:
    assert hypergeom_tail(100, 20, 10, 5) == pytest.approx(0.0255, abs=5e-4)
    assert hypergeom_tail(100, 10, 10, 5) == pytest.approx(0.0007, abs=2e-4)
    assert hypergeom_tail(100, 20, 10, 11) == 0.0


def _write_run(root: Path, label: str, cfg_hash: str, seed: int, cfg: dict, asr: np.ndarray,
               mal_by_round: dict[int, int], ledger: bool) -> None:
    d = root / label / cfg_hash / f"seed_{seed}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"seed": seed, "config": cfg}))
    (d / "metrics.json").write_text(json.dumps(
        [{"round": r + 1, "asr_accuracy": float(asr[r]), "test_accuracy": 0.8,
          **{k: 0.1 for k in ("client_train_s", "scoring_s", "aggregation_s",
                              "server_train_s", "evaluation_s", "round_s")}}
         for r in range(len(asr))]))
    rows = _rows(mal_by_round, len(asr))
    (d / "client_metrics.json").write_text(json.dumps(rows))
    if ledger:
        (d / "defense_ledger.json").write_text(json.dumps(
            [{**r, "weight": 0.1} for r in rows]))


def test_load_cells_and_report_end_to_end(tmp_path: Path) -> None:
    results = tmp_path / "results"
    n = 30
    captured = np.concatenate([np.zeros(10), np.ones(n - 10)])
    clean = np.zeros(n)
    capped = {"scorer": {"weighting": "critic"},
              "integrator": {"_target_": "topfed.integration.mechanisms.CappedScoreWeight", "cap": 1.0},
              "base": "weighted_median"}
    # Reference capped-wmed, two seeds.
    for seed in (0, 1):
        _write_run(results, "femnist_backdoor_f20__CappedScoreWeight_critic_wmedian", "aaaa", seed,
                   _config(defense=capped), captured, {11: 6}, ledger=True)
    # Forced m=5 capped-wmed, one seed, tagged.
    _write_run(results, "e5d3_femnist_backdoor_f20_m5__CappedScoreWeight_critic_wmedian", "bbbb", 0,
               _config(defense=capped, **{"federation.force_malicious_per_round": 5, "tag": "e5d3_x"}),
               captured, {r: 5 for r in range(1, n + 1)}, ledger=True)
    # Forced m=2 plain median, holds.
    _write_run(results, "e5d3_femnist_backdoor_f20_m2__coordinate_median", "cccc", 0,
               _config(baseline={"name": "coordinate_median"},
                       **{"federation.force_malicious_per_round": 2}),
               clean, {r: 2 for r in range(1, n + 1)}, ledger=False)
    # A crashed run (no metrics.json) must be skipped.
    (results / "e5d1_x" / "dddd" / "seed_0").mkdir(parents=True)
    (results / "e5d1_x" / "dddd" / "seed_0" / "manifest.json").write_text(
        json.dumps({"seed": 0, "config": _config()}))
    # A non-backdoor run must be ignored.
    other = _config()
    other["attack"] = {"kind": "sign_flip", "poisoner": None, "assignment": {"fraction": 0.2}}
    _write_run(results, "sign_flip", "eeee", 0, other, clean, {}, ledger=False)

    cells = load_cells(results)
    blocks = sorted((k.block, k.defense, k.forced_m, len(v)) for k, v in cells.items())
    assert blocks == [
        ("D3", "capped-wmed", 5, 1),
        ("D3", "median", 2, 1),
        ("REF", "capped-wmed", None, 2),
    ]

    report = build_report(cells, 0.2)
    assert "## D3 Forced co-selection" in report
    assert "| femnist | capped-wmed | 5 | capture | capture |" in report
    assert "| femnist | median | 2 | hold | hold |" in report
    assert "| femnist | capped-wmed | sampled | – | capture |" in report
    assert "## Timing" in report
    assert "_no runs yet_" in report  # D8 has nothing

    out = tmp_path / "e5.md"
    dump = tmp_path / "e5.json"
    assert main(["--results-root", str(results), "--out", str(out), "--json", str(dump)]) == 0
    assert out.read_text() == report
    records = json.loads(dump.read_text())
    assert len(records) == 4 and {r["block"] for r in records} == {"D3", "REF"}


def test_main_refuses_to_write_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    with pytest.raises(SystemExit):
        main(["--results-root", str(results), "--out", str(results / "e5.md")])
