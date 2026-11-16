"""E1 analysis: cell keys, per-run summaries, breaking m, KS shift, report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topfed.analysis import e5
from topfed.analysis.e1 import (
    CellKey,
    RunSummary,
    breaking_m,
    build_report,
    cell_key,
    find,
    ks_shift,
    load_cells,
    main,
    neighbours,
    stage_b_suggestion,
    summarise_run,
)


def _config(strategy: str | None = None, m: int = 3, adaptive: bool = False, **over):
    manip = (
        {"_target_": "topfed.attacks.model.adaptive.NormBoundedManipulator", "target_norm": 0.6}
        if adaptive
        else {"_target_": "topfed.attacks.model.scale.ModelReplacementManipulator", "gamma": None}
    )
    cfg = {
        "dataset": {"name": "cifar10", "val_per_class": 100, "augment": True},
        "partition": {"kind": "dirichlet", "alpha": 0.5, "seed": 0},
        "federation": {"num_clients": 100, "clients_per_round": 10, "num_rounds": 200,
                       "force_malicious_per_round": m, "trace_capture": True},
        "attack": {
            "kind": "backdoor",
            "poisoner": {"_target_": "x.BackdoorPoisoner", "target_class": 9, "trigger_size": 6,
                         "trigger_amplitude": 1.0, "batch_mode": "augment", "poisoning_rate": 0.5},
            "update_manipulator": manip,
            "schedule": {"_target_": "topfed.attacks.schedule.AlwaysAttack"},
            "assignment": {"fraction": 0.3},
        },
        "experiment": "collusion_backdoor_cifar10_f30",
    }
    if strategy is not None:
        cfg["attack"]["collusion"] = {"strategy": strategy}
    cfg.update(over)
    return cfg


CAPPED = {"scorer": {"weighting": "critic"},
          "integrator": {"_target_": "topfed.integration.mechanisms.CappedScoreWeight", "cap": 1.0, "alpha": 1.0},
          "base": "weighted_median"}


def test_cell_key_reads_strategy_m_attack_and_defense() -> None:
    key = cell_key({"config": _config("reference_shift", m=4, adaptive=True, defense=CAPPED)})
    assert key == CellKey("adaptive", 0.6, "reference_shift", None, 4, "capped-wmed")
    assert cell_key({"config": _config(None, m=2)}).strategy == "independent"  # type: ignore[union-attr]
    cfg = _config("coordinate_capture")
    cfg["attack"]["collusion"]["top_fraction"] = 0.05
    assert cell_key({"config": cfg}).top_fraction == 0.05  # type: ignore[union-attr]
    assert cell_key({"config": _config("clone", baseline={"name": "foolsgold", "kappa": 1.0})}).defense == "foolsgold"  # type: ignore[union-attr]


def test_out_of_scope_runs_are_skipped() -> None:
    no_force = _config()
    no_force["federation"]["force_malicious_per_round"] = None
    assert cell_key({"config": no_force}) is None
    f20 = _config()
    f20["attack"]["assignment"]["fraction"] = 0.2
    assert cell_key({"config": f20}) is None
    femnist = _config()
    femnist["dataset"]["name"] = "femnist"
    assert cell_key({"config": femnist}) is None


def test_e5_skips_collusion_runs() -> None:
    assert e5.cell_key({"config": _config("clone")}) is None
    assert e5.cell_key({"config": _config(None)}) is not None


def _metrics(asr: float, share: float, rounds: int = 200) -> list[dict]:
    return [
        {"round": r, "asr_accuracy": asr, "test_accuracy": 0.7,
         "trace_malicious_share": share, "trace_captured_fraction": 0.1,
         "trace_projection": 1.0, "trace_malicious_norm": 2.0}
        for r in range(1, rounds + 1)
    ]


def _clients(m: int, benign_m1: float, rounds: int = 3) -> list[dict]:
    rows = []
    for r in range(1, rounds + 1):
        for i in range(10):
            mal = i < m
            rows.append({"round": r, "client_id": f"c{i}", "is_malicious": mal,
                         "m1_cosine_to_mean": 0.9 if mal else benign_m1 + 0.01 * i,
                         "m2_norm_ratio": 1.0, "m5_max_layer_deviation": 2.0})
    return rows


def test_summary_share_trace_and_trust_ranks() -> None:
    clients = _clients(3, 0.3)
    ledger = [
        {"round": r, "client_id": f"c{i}", "score": float(10 - i), "weight": 0.1,
         "is_malicious": i < 3}
        for r in (1, 2) for i in range(10)
    ]
    coll = [{"round": 1, "leader": "c2", "colluders": ["c2", "c0", "c1"], "strategy": "reference_shift"},
            {"round": 2, "leader": "c0", "colluders": ["c0", "c1", "c2"], "strategy": "reference_shift"}]
    s = summarise_run(0, _metrics(0.8, 0.6), clients, ledger, coll)
    assert s.asr_final == pytest.approx(0.8)
    assert s.mean_share == pytest.approx(0.6)
    assert s.majority_rounds == pytest.approx(1.0)
    assert s.mean_captured == pytest.approx(0.1)
    assert s.mean_projection_ratio == pytest.approx(0.5)
    # Scores 10..1: c0 most trusted (rank 0), c1 1/9, c2 2/9.
    assert s.malicious_rank == pytest.approx(1 / 9)
    assert s.leader_rank == pytest.approx((2 / 9 + 0) / 2)
    assert s.follower_rank == pytest.approx((0 + 1 / 9 + 1 / 9 + 2 / 9) / 4)
    assert len(s.benign["m1_cosine_to_mean"]) == 7 * 3


def test_summary_without_trace_falls_back_to_ledger_then_nan() -> None:
    metrics = [{"round": r, "asr_accuracy": 0.1, "test_accuracy": 0.7} for r in range(1, 4)]
    clients = _clients(2, 0.3)
    ledger = [{"round": r, "client_id": f"c{i}", "weight": 0.25 if i < 2 else 0.5 / 8,
               "is_malicious": i < 2} for r in range(1, 4) for i in range(10)]
    s = summarise_run(0, metrics, clients, ledger, None)
    assert s.mean_share == pytest.approx(0.5)
    assert s.leader_rank is None
    bare = summarise_run(0, metrics, clients, None, None)
    assert bare.mean_share is None and bare.malicious_rank is None


def _run(asr: float) -> RunSummary:
    return RunSummary(0, asr, 0.7, None, [0.2], [0.1], [0.5], {"m1_cosine_to_mean": [0.3]}, None, None, None)


def test_breaking_m_uses_independent_m1_and_flags_gaps() -> None:
    cells = {
        CellKey("backdoor", None, "independent", None, 1, "capped-wmed"): [_run(0.1)],
        CellKey("backdoor", None, "clone", None, 2, "capped-wmed"): [_run(0.2)],
        CellKey("backdoor", None, "clone", None, 3, "capped-wmed"): [_run(0.7)],
        CellKey("backdoor", None, "reference_shift", None, 3, "capped-wmed"): [_run(0.9)],
    }
    assert find(cells, "backdoor", None, "clone", 1, "capped-wmed")[0].asr_final == 0.1
    assert breaking_m(cells, "backdoor", None, "clone", "capped-wmed") == (3, True)
    assert breaking_m(cells, "backdoor", None, "reference_shift", "capped-wmed") == (3, False)
    assert breaking_m(cells, "backdoor", None, "clone", "median") == (None, False)


def test_neighbours_stay_inside_the_grid() -> None:
    assert neighbours(3) == (2, 3, 4)
    assert neighbours(1) == (1, 2, 3)
    assert neighbours(5) == (3, 4, 5)
    assert neighbours(None) == (3, 4, 5)


def test_stage_b_picks_the_strategy_that_breaks_earliest() -> None:
    cells = {CellKey("backdoor", None, "independent", None, 1, "capped-wmed"): [_run(0.1)]}
    for m in (2, 3, 4, 5):
        cells[CellKey("backdoor", None, "clone", None, m, "capped-wmed")] = [_run(0.6 if m >= 5 else 0.1)]
        cells[CellKey("backdoor", None, "reference_shift", None, m, "capped-wmed")] = [_run(0.6 if m >= 3 else 0.1)]
    s = stage_b_suggestion(cells)[0]
    assert (s.strategy, s.breaking, s.ms, s.complete) == ("reference_shift", 3, (2, 3, 4), True)


def test_ks_shift_detects_a_moved_benign_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = [RunSummary(0, 0.1, 0.7, None, [], [], [], {"m1_cosine_to_mean": list(rng.normal(0.3, 0.05, 500))}, None, None, None)]
    same = [RunSummary(1, 0.1, 0.7, None, [], [], [], {"m1_cosine_to_mean": list(rng.normal(0.3, 0.05, 500))}, None, None, None)]
    moved = [RunSummary(2, 0.1, 0.7, None, [], [], [], {"m1_cosine_to_mean": list(rng.normal(0.1, 0.05, 500))}, None, None, None)]
    d_same, _ = ks_shift(same, ref, "m1_cosine_to_mean")  # type: ignore[misc]
    d_moved, shift = ks_shift(moved, ref, "m1_cosine_to_mean")  # type: ignore[misc]
    assert d_same < 0.15 < 0.9 < d_moved
    assert shift == pytest.approx(-0.2, abs=0.02)
    assert ks_shift(moved, [], "m1_cosine_to_mean") is None


def _write_run(root: Path, label: str, seed: int, cfg: dict, asr: float, share: float, m: int) -> None:
    d = root / label / "hash" / f"seed_{seed}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"seed": seed, "config": cfg}))
    (d / "metrics.json").write_text(json.dumps(_metrics(asr, share)))
    (d / "client_metrics.json").write_text(json.dumps(_clients(m, 0.3 - 0.05 * m)))


def test_load_report_and_cli(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for seed in (0, 1):
        _write_run(root, "e1_ind_m1", seed, _config(None, m=1, defense=CAPPED), 0.05, 0.1, 1)
        for m in (2, 3, 4, 5):
            _write_run(root, f"e1_clone_m{m}", seed, _config("clone", m=m, defense=CAPPED), 0.1 * m, 0.1 * m, m)
            _write_run(root, f"e1_rs_m{m}", seed, _config("reference_shift", m=m, defense=CAPPED), 0.2 * m, 0.12 * m, m)
    # An incomplete run is ignored.
    d = root / "e1_partial" / "h" / "seed_0"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"seed": 0, "config": _config("clone", m=2)}))
    (d / "metrics.json").write_text(json.dumps(_metrics(0.5, 0.5, rounds=10)))

    cells = load_cells(root)
    assert len(cells[CellKey("backdoor", None, "clone", None, 3, "capped-wmed")]) == 2
    assert CellKey("backdoor", None, "clone", None, 2, "undefended") not in cells
    report = build_report(cells)
    assert "Breaking m" in report and "S3 reference shift" in report
    assert "Benign metric shift" in report
    assert "`reference_shift`" in report  # RS breaks at m=3 (0.6) before clone at m=5

    out = tmp_path / "e1.md"
    fig_dir = tmp_path / "figs"
    assert main(["--results-root", str(root), "--out", str(out), "--fig-dir", str(fig_dir),
                 "--json", str(tmp_path / "e1.json")]) == 0
    assert out.read_text() == report
    assert (fig_dir / "e1-asr-vs-m-backdoor.png").exists()
    assert (fig_dir / "e1-share-backdoor-reference_shift.png").exists()
    rows = json.loads((tmp_path / "e1.json").read_text())
    assert all("benign" not in r for r in rows)
    with pytest.raises(SystemExit):
        main(["--results-root", str(root), "--out", str(root / "x.md")])


def test_stage_b_cli_prints_sweep_variables(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    root = tmp_path / "results"
    _write_run(root, "e1_ind_m1", 0, _config(None, m=1, defense=CAPPED), 0.05, 0.1, 1)
    for m in (2, 3, 4, 5):
        _write_run(root, f"e1_rs_m{m}", 0, _config("reference_shift", m=m, defense=CAPPED), 0.9 if m >= 4 else 0.1, 0.3, m)
    assert main(["--results-root", str(root), "--stage-b"]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith('STAGE_B_M="') and "reference_shift" in line
    assert '"3 4 5"' in line


def test_diverged_run_has_undefined_asr_and_chance_accuracy() -> None:
    metrics = _metrics(0.9, 0.5)
    for row in metrics[26:]:
        row["test_loss"] = float("nan")
        row["asr_accuracy"] = 0.0
        row["test_accuracy"] = 0.1
    for row in metrics[:26]:
        row["test_loss"] = 1.0
    s = summarise_run(1, metrics, _clients(3, 0.3), None, None)
    assert s.diverged and s.asr_final is None
    assert s.acc_final == pytest.approx(0.1)
    healthy = summarise_run(0, _metrics(0.9, 0.5), _clients(3, 0.3), None, None)
    assert not healthy.diverged and healthy.asr_final == pytest.approx(0.9)
    cells = {CellKey("adaptive", 0.6, "reference_shift", None, 5, "undefended"): [healthy, s],
             CellKey("adaptive", 0.6, "independent", None, 1, "undefended"): [healthy]}
    report = build_report(cells)
    assert "0.900 ± 0.000 (2, 1 diverged)" in report
    assert "(diverged: 1)" in report


def test_pre_implant_window_stops_at_the_implant_round() -> None:
    metrics = [{"round": r, "asr_accuracy": 0.9 if r >= 40 else 0.0, "test_accuracy": 0.7, "test_loss": 1.0}
               for r in range(1, 201)]
    clients = _clients(2, 0.3, rounds=60)
    s = summarise_run(0, metrics, clients, None, None)
    assert s.implant_round == 40
    assert len(s.benign_pre["m1_cosine_to_mean"]) == 8 * 39
    assert len(s.benign["m1_cosine_to_mean"]) == 8 * 60
    ref = [summarise_run(0, _metrics(0.1, 0.1), _clients(1, 0.3, rounds=60), None, None)]
    assert ks_shift([s], ref, "m1_cosine_to_mean", "pre") is not None
