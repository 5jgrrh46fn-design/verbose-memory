"""E3 analysis: cell keys, best-response selection, paired stats, overhead, report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from topfed.analysis.e3 import (
    CLIENTS_PER_ROUND,
    NUM_ROUNDS,
    OVERHEAD_ROUNDS,
    WARMUP_ROUNDS,
    CellKey,
    RunSummary,
    build_report,
    cell_key,
    communication_per_round,
    holm,
    load_cells,
    main,
    model_state_bytes,
    overhead_row,
    paired,
    root_comparisons,
    select_cell,
    summarise_run,
    table_comparisons,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"

CAPPED = {"scorer": {"weighting": "critic"},
          "integrator": {"_target_": "topfed.integration.mechanisms.CappedScoreWeight",
                         "cap": 1.0, "alpha": 1.0},
          "base": "weighted_median"}
SCOREW = {"scorer": {"weighting": "critic"},
          "integrator": {"_target_": "topfed.integration.mechanisms.ScoreWeight", "alpha": 1.0}}
FLTRUST = {"name": "fltrust"}
MEDIAN = {"name": "coordinate_median"}
TRIGGER = {"_target_": "topfed.attacks.data.backdoor.BackdoorPoisoner", "target_class": 9,
           "trigger_size": 6, "trigger_amplitude": 1.0, "batch_mode": "augment",
           "poisoning_rate": 0.5}


def _attack(kind: str, strength: float | None = None) -> dict[str, Any]:
    schedule = {"_target_": "topfed.attacks.schedule.AlwaysAttack"}
    base: dict[str, Any] = {"schedule": schedule, "assignment": {"fraction": 0.1}, "loss_augmenter": None}
    if kind == "backdoor":
        return {**base, "kind": "backdoor", "poisoner": dict(TRIGGER),
                "update_manipulator": {"_target_": "x.ModelReplacementManipulator", "gamma": None}}
    if kind == "adaptive":
        return {**base, "kind": "backdoor", "poisoner": dict(TRIGGER),
                "update_manipulator": {"_target_": "x.NormBoundedManipulator",
                                       "target_norm": 0.6 if strength is None else strength}}
    if kind == "direction_aligned":
        return {**base, "kind": "backdoor", "poisoner": dict(TRIGGER),
                "update_manipulator": {"_target_": "x.DirectionAlignedManipulator",
                                       "cosine_target": 0.9 if strength is None else strength}}
    if kind == "clean_label":
        return {**base, "kind": "clean_label", "update_manipulator": None,
                "poisoner": {"_target_": "x.CleanLabelPoisoner", "target_class": 9,
                             "trigger_size": 6, "trigger_amplitude": 1.0, "pgd_eps": 0.125}}
    if kind == "gradient_scaling":
        return {**base, "kind": "gradient_scaling", "poisoner": None,
                "update_manipulator": {"_target_": "x.GradientScalingManipulator", "factor": 5.0}}
    if kind == "sign_flip":
        return {**base, "kind": "sign_flip", "poisoner": None,
                "update_manipulator": {"_target_": "x.SignFlippingManipulator", "eta": 1.0}}
    if kind == "dirty_label":
        return {**base, "kind": "dirty_label", "update_manipulator": None,
                "poisoner": {"_target_": "x.DirtyLabelPoisoner", "num_classes": 10,
                             "poisoning_rate": 1.0}}
    assert kind == "none"
    return {**base, "kind": "none", "poisoner": None, "update_manipulator": None,
            "schedule": None, "assignment": {"fraction": 0.0}}


def _config(kind: str = "backdoor", strength: float | None = None, /, **over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "dataset": {"name": "cifar10", "val_per_class": 100, "augment": True},
        "partition": {"kind": "dirichlet", "alpha": 0.5, "seed": 0},
        "federation": {"num_clients": 100, "clients_per_round": 10, "num_rounds": NUM_ROUNDS},
        "attack": _attack(kind, strength),
        "experiment": "x",
    }
    for dotted, value in over.items():
        node: Any = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def _femnist(kind: str = "backdoor", /, **over: Any) -> dict[str, Any]:
    cfg = _config(kind, **{"dataset": {"name": "femnist", "val_per_class": 30, "augment": False},
                           "partition": {"kind": "natural", "seed": 0}, **over})
    if cfg["attack"].get("poisoner") and "trigger_amplitude" in cfg["attack"]["poisoner"]:
        cfg["attack"]["poisoner"]["trigger_amplitude"] = -5.0
    return cfg


def _key(cfg) -> CellKey | None:
    return cell_key({"seed": 0, "config": cfg})


def _ck(attack="backdoor", strength=None, fraction=0.1, defense="undefended", dataset="cifar10",
        val=None, rounds=NUM_ROUNDS, memory=False) -> CellKey:
    v = {"cifar10": 100, "femnist": 30}[dataset] if val is None else val
    return CellKey(dataset, attack, strength, fraction, defense, v, rounds, memory)


# ----------------------------------------------------------------------------
# Cell keys
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "strength", "expected"),
    [
        ("backdoor", None, ("backdoor", None)),
        ("adaptive", 2.0, ("adaptive", 2.0)),
        ("direction_aligned", 0.5, ("direction_aligned", 0.5)),
        ("clean_label", None, ("clean_label", None)),
        ("gradient_scaling", None, ("gradient_scaling", 5.0)),
        ("sign_flip", None, ("sign_flip", 1.0)),
        ("dirty_label", None, ("dirty_label", 1.0)),
    ],
)
def test_cell_key_names_every_attacker(kind, strength, expected) -> None:
    key = _key(_config(kind, strength))
    assert key is not None
    assert (key.attack, key.strength) == expected


def test_cell_key_carries_defense_setting_and_benign() -> None:
    assert _key(_config(baseline=FLTRUST)) == _ck(defense="fltrust")
    assert _key(_config(defense=CAPPED, **{"dataset.val_per_class": 1})) == _ck(
        defense="capped-wmed", val=1)
    assert _key(_config(baseline={"name": "krum", "num_byzantine": 2})) == _ck(defense="krum")
    over = {"federation.num_rounds": OVERHEAD_ROUNDS}
    assert _key(_config(**over)) == _ck(rounds=OVERHEAD_ROUNDS)
    mem = {"federation.num_rounds": 5, "federation.profile_memory": True}
    assert _key(_config(baseline=MEDIAN, **mem)) == _ck(defense="median", rounds=5, memory=True)
    assert _key(_config("none")) == _ck("benign", fraction=0.0)
    assert _key(_femnist(baseline=FLTRUST)) == _ck(dataset="femnist", defense="fltrust")


@pytest.mark.parametrize(
    "over",
    [
        {"federation.force_malicious_per_round": 5},
        {"federation.clients_per_round": 20},
        {"partition": {"kind": "iid", "seed": 0}},
        {"dataset.pad_border": 4},
        {"attack.poisoner.trigger_position": "center"},
        {"attack.poisoner.trigger_size": 4},
        {"attack.poisoner.trigger_amplitude": -2.0},
        {"attack.poisoner.batch_mode": "replace"},
        {"attack.update_manipulator": None},
        {"attack.update_manipulator": {"_target_": "x.ModelReplacementManipulator", "gamma": 10.0}},
        {"attack.schedule": {"_target_": "topfed.attacks.schedule.WindowedAttack",
                             "start_round": 1, "end_round": 100}},
    ],
)
def test_diagnostic_variants_are_out_of_scope(over) -> None:
    assert _key(_config(**over)) is None


# ----------------------------------------------------------------------------
# Per-run summary
# ----------------------------------------------------------------------------


def _metrics(n: int, asr: float | None = None, acc: float = 0.7, timing: bool = False,
             memory: bool = False) -> list[dict]:
    rows = []
    for r in range(1, n + 1):
        row: dict = {"round": r, "test_accuracy": acc}
        if asr is not None:
            row["asr_accuracy"] = asr
        if timing:
            slow = 10.0 if r <= WARMUP_ROUNDS else 1.0
            row.update({"client_train_s": 2.0 * slow, "scoring_s": 0.3, "aggregation_s": 0.1,
                        "server_train_s": 0.0, "evaluation_s": 0.5, "round_s": 3.0 * slow})
        if memory:
            row.update({"server_peak_host_mb": float(r), "server_peak_device_mb": 2.0 * r})
        rows.append(row)
    return rows


def test_summarise_run_drops_warmup_and_takes_memory_peak() -> None:
    s = summarise_run(0, _metrics(OVERHEAD_ROUNDS, timing=True, memory=True))
    assert s.asr_final is None and s.acc_final == pytest.approx(0.7)
    assert len(s.timing_rounds["round_s"]) == OVERHEAD_ROUNDS - WARMUP_ROUNDS
    assert set(s.timing_rounds["round_s"]) == {3.0}
    assert s.memory_peak == {"server_peak_host_mb": 50.0, "server_peak_device_mb": 100.0}
    frozen = summarise_run(0, _metrics(NUM_ROUNDS, asr=0.4))
    assert frozen.asr_final == pytest.approx(0.4)
    assert frozen.timing_rounds == {} and frozen.memory_peak == {}


# ----------------------------------------------------------------------------
# Best-response selection
# ----------------------------------------------------------------------------


def _runs(values, seeds=None, acc=0.7) -> list[RunSummary]:
    seeds = range(len(values)) if seeds is None else seeds
    return [RunSummary(seed=s, asr_final=v, acc_final=acc) for s, v in zip(seeds, values)]


def test_best_response_needs_a_seed_matched_gain() -> None:
    ref = [0.20, 0.25, 0.22, 0.24, 0.21, 0.23, 0.2, 0.2, 0.2, 0.2]
    noisy = [0.21, 0.24, 0.23, 0.23, 0.22]  # same level, mixed signs
    clear = [v + 0.1 for v in ref[:5]]
    cells = {
        _ck("adaptive", 0.6, defense="median"): _runs(ref),
        _ck("adaptive", 0.5, defense="median"): _runs(noisy),
        _ck("adaptive", 2.0, defense="median"): _runs(clear),
        _ck("adaptive", 5.0, defense="median"): _runs([0.99]),  # one seed: not eligible
    }
    # A rival with fewer seeds than the reference never displaces it, even
    # when it raises ASR; it is reported for a seed upgrade instead.
    from topfed.analysis.e3 import knob_bumps

    sel = select_cell(cells, "cifar10", "adaptive", 0.1, "median")
    assert sel is not None and sel.strength == 0.6 and not sel.best_response
    assert set(sel.by_strength) == {0.6, 0.5, 2.0, 5.0}
    bumps = knob_bumps(cells, "cifar10", ("median",))
    assert [(b.strength, b.missing_seeds) for b in bumps] == [(2.0, (5, 6, 7, 8, 9))]
    assert bumps[0].sweep_line() == (
        "sp_adaptive_backdoor_cifar10_f10_g2.0\tadaptive_backdoor_cifar10_f10\t5,6,7,8,9"
        "\tattack.update_manipulator.target_norm=2.0\t+baseline=coordinate_median"
    )
    # With matching seeds the rival wins.
    cells[_ck("adaptive", 2.0, defense="median")] = _runs([v + 0.1 for v in ref])
    sel = select_cell(cells, "cifar10", "adaptive", 0.1, "median")
    assert sel is not None and sel.strength == 2.0 and sel.best_response
    assert len(sel.runs) == 10
    assert knob_bumps(cells, "cifar10", ("median",)) == []
    del cells[_ck("adaptive", 2.0, defense="median")]
    kept = select_cell(cells, "cifar10", "adaptive", 0.1, "median")
    assert kept is not None and kept.strength == 0.6 and not kept.best_response
    fixed = select_cell(cells, "cifar10", "adaptive", 0.1, "median", best_response=False)
    assert fixed is not None and fixed.strength == 0.6


def test_probe_only_reference_is_treated_as_absent() -> None:
    # The attack config's budget has one probe seed on FEMNIST; the tuned
    # budget has ten. A one-seed reference cannot anchor the table, so the
    # choice falls to the eligible maximum and is marked best-response.
    cells = {
        _ck("adaptive", 0.6, dataset="femnist"): _runs([0.997]),
        _ck("adaptive", 0.3, dataset="femnist"): _runs([0.986]),
        _ck("adaptive", 0.2, dataset="femnist"): _runs([0.89] * 10),
        _ck("adaptive", 0.2, dataset="femnist", defense="median"): _runs([0.65] * 10),
    }
    sel = select_cell(cells, "femnist", "adaptive", 0.1, "undefended")
    assert sel is not None and sel.strength == 0.2 and sel.best_response
    assert len(sel.runs) == 10
    sel = select_cell(cells, "femnist", "adaptive", 0.1, "median")
    assert sel is not None and sel.strength == 0.2 and len(sel.runs) == 10
    # Without best-response the attack config's value is still reported as is.
    fixed = select_cell(cells, "femnist", "adaptive", 0.1, "undefended", best_response=False)
    assert fixed is not None and fixed.strength == 0.6 and len(fixed.runs) == 1


def test_best_response_without_reference_and_non_knob_attacks() -> None:
    cells = {
        _ck("direction_aligned", 0.5, fraction=0.2): _runs([0.8] * 5),
        _ck("direction_aligned", 0.7, fraction=0.2): _runs([0.6] * 5),
        _ck("gradient_scaling", 5.0): _runs([None] * 5, acc=0.6),
        _ck("gradient_scaling", 10.0): _runs([None] * 5, acc=0.1),
    }
    sel = select_cell(cells, "cifar10", "direction_aligned", 0.2, "undefended")
    assert sel is not None and sel.strength == 0.5 and sel.best_response
    gs = select_cell(cells, "cifar10", "gradient_scaling", 0.1, "undefended")
    assert gs is not None and gs.strength == 5.0
    assert select_cell(cells, "cifar10", "sign_flip", 0.1, "undefended") is None
    assert select_cell({}, "cifar10", "adaptive", 0.1, "undefended") is None


# ----------------------------------------------------------------------------
# Paired statistics
# ----------------------------------------------------------------------------


def test_paired_matches_seeds_and_holm_is_per_family() -> None:
    a = _runs([0.1] * 10)
    b = _runs([0.3 + 0.01 * i for i in range(10)])
    c = paired("fam1", "a − b", "asr_final", a, b)
    assert c is not None and c.n == 10
    assert c.mean_diff == pytest.approx(-0.245)
    assert c.ci_high < 0 and c.p_value == pytest.approx(2 / 2**10)
    assert c.rank_biserial == pytest.approx(-1.0)
    short = paired("fam2", "x", "asr_final", a[:5], b[:5])
    assert short is not None and short.p_value == pytest.approx(0.0625)
    assert paired("fam3", "y", "asr_final", a[:1], b[:1]) is None
    # Disjoint seeds pair nothing.
    assert paired("fam3", "z", "asr_final", _runs([0.1] * 3), _runs([0.2] * 3, seeds=[7, 8, 9])) is None
    adjusted = {x.family: x for x in holm([c, short])}
    assert adjusted["fam1"].holm_p == pytest.approx(c.p_value)  # one test in its family
    assert adjusted["fam1"].reject and not adjusted["fam2"].reject


def test_table_comparisons_use_primary_metric() -> None:
    cells = {
        _ck("sign_flip", 1.0, defense="fltrust"): _runs([None] * 6, acc=0.72),
        _ck("sign_flip", 1.0, defense="capped-wmed"): _runs([None] * 6, acc=0.70),
    }
    (comp,) = table_comparisons(cells, "cifar10")
    assert comp.metric == "acc_final"
    assert comp.mean_diff == pytest.approx(-0.02)  # capped-wmed − fltrust
    assert comp.family == "cifar10 f=0.1 sign_flip"


def test_root_comparisons_pair_against_the_reference_size() -> None:
    cells = {}
    for v, asr in ((1, 0.5), (10, 0.2), (100, 0.06)):
        cells[_ck("backdoor", defense="fltrust", val=v)] = _runs([asr] * 5)
        cells[_ck("backdoor", defense="capped-wmed", val=v)] = _runs([0.04] * 5)
    comps = root_comparisons(cells)
    labels = {c.label: c for c in comps}
    assert labels["vanilla backdoor fltrust: 1/class − 100/class"].mean_diff == pytest.approx(0.44)
    assert labels["vanilla backdoor 1/class: capped-wmed − fltrust"].mean_diff == pytest.approx(-0.46)
    assert all(c.family == "root backdoor" for c in comps)


# ----------------------------------------------------------------------------
# Overhead and communication
# ----------------------------------------------------------------------------


def test_overhead_row_pools_timing_runs_and_memory_runs() -> None:
    timing = [summarise_run(s, _metrics(OVERHEAD_ROUNDS, timing=True)) for s in (0, 1)]
    memory = [summarise_run(0, _metrics(5, memory=True))]
    cells = {
        _ck(defense="fltrust", rounds=OVERHEAD_ROUNDS): timing,
        _ck(defense="fltrust", rounds=5, memory=True): memory,
        _ck(defense="fltrust"): [summarise_run(0, _metrics(NUM_ROUNDS, asr=0.1, timing=True))],
    }
    row = overhead_row(cells, "cifar10", "fltrust")
    assert row.rounds == 2 * (OVERHEAD_ROUNDS - WARMUP_ROUNDS)
    assert row.mean_s["round_s"] == pytest.approx(3.0)
    assert row.server_s == pytest.approx(0.4)
    assert row.memory_mb == {"server_peak_host_mb": 5.0, "server_peak_device_mb": 10.0}
    empty = overhead_row(cells, "cifar10", "krum")
    assert empty.server_s is None and empty.memory_mb == {}


def test_communication_is_two_k_state_dicts() -> None:
    from topfed.models.cifar_cnn import CifarCNN

    numel, nbytes = model_state_bytes("cifar10")
    assert numel == sum(t.numel() for t in CifarCNN().state_dict().values())
    assert nbytes == 4 * numel  # every value is float32
    c = communication_per_round("femnist")
    assert c["upload_bytes"] == c["download_bytes"] == CLIENTS_PER_ROUND * c["state_bytes"]
    assert c["total_bytes"] == 2 * CLIENTS_PER_ROUND * c["state_bytes"]


# ----------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------


def _write_run(root: Path, label: str, cfg_hash: str, seed: int, cfg: dict, asr: float | None,
               acc: float = 0.7, timing: bool = False, memory: bool = False) -> None:
    d = root / label / cfg_hash / f"seed_{seed}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"seed": seed, "config": cfg}))
    n = int(cfg["federation"]["num_rounds"])
    (d / "metrics.json").write_text(json.dumps(_metrics(n, asr, acc, timing, memory)))


def _populate(results: Path) -> None:
    for seed in range(10):
        _write_run(results, "backdoor_cifar10_f10__fltrust", "a1", seed, _config(baseline=FLTRUST), 0.06)
        _write_run(results, "backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian", "a2", seed,
                   _config(defense=CAPPED), 0.04 - 0.002 * seed)
    for seed in range(5):
        _write_run(results, "benign_cifar10", "b0", seed, _config("none"), None, acc=0.75)
        _write_run(results, "e3_sign_flip_cifar10_f20__fltrust", "s1", seed,
                   _config("sign_flip", baseline=FLTRUST, **{"attack.assignment.fraction": 0.2}),
                   None, acc=0.70)
        _write_run(results, "e3_backdoor_cifar10_f10_root1__fltrust", "r1", seed,
                   _config(baseline=FLTRUST, **{"dataset.val_per_class": 1}), 0.5)
        for c in (0.5, 0.9):
            cfg = _config("direction_aligned", c, defense=CAPPED, **{"attack.assignment.fraction": 0.2})
            _write_run(results, f"e3_fltrust_backdoor_cifar10_f20_c{c}__CappedScoreWeight_critic_wmedian",
                       f"d{c}", seed, cfg, 0.3 if c == 0.5 else 0.1)
    for label, cfg in (("fltrust", _config(baseline=FLTRUST)), ("undefended", _config())):
        suffix = "__fltrust" if label == "fltrust" else ""
        timed = {**cfg, "federation": {**cfg["federation"], "num_rounds": OVERHEAD_ROUNDS}}
        _write_run(results, f"e3_overhead_cifar10{suffix}", "o1", 0, timed, 0.5, timing=True)
        mem = {**cfg, "federation": {**cfg["federation"], "num_rounds": 5, "profile_memory": True}}
        _write_run(results, f"e3_overhead_memory_cifar10{suffix}", "m1", 0, mem, 0.5, memory=True)
    # The same config under a second, smaller directory is not double-counted.
    _write_run(results, "dup__fltrust", "zz", 0, _config(baseline=FLTRUST), 0.99)
    # A crashed run (no metrics.json) and a truncated run are skipped.
    crashed = results / "e3_crashed" / "cccc" / "seed_0"
    crashed.mkdir(parents=True)
    (crashed / "manifest.json").write_text(json.dumps({"seed": 0, "config": _config()}))
    short = results / "e3_short" / "tttt" / "seed_0"
    short.mkdir(parents=True)
    (short / "manifest.json").write_text(json.dumps({"seed": 0, "config": _config(defense=SCOREW)}))
    (short / "metrics.json").write_text(json.dumps(_metrics(10, 0.1)))


def test_load_cells_report_and_cli(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _populate(results)
    cells = load_cells(results)

    fl = cells[_ck(defense="fltrust")]
    assert len(fl) == 10 and fl[0].asr_final == pytest.approx(0.06)
    assert _ck(defense="ScoreWeight") not in cells  # truncated run dropped
    assert _ck("direction_aligned", 0.5, fraction=0.2, defense="capped-wmed") in cells

    report = build_report(cells)
    assert "0.060 ± 0.000 (10) | 0.700 ± 0.000 (10) | 0.031 ± 0.006 (10)" in report
    # Direction-aligned best response against capped-wmed is cosine 0.5.
    assert "0.300 ± 0.000 (5) †0.5" in report
    # Untargeted rows report accuracy against the benign FedAvg reference.
    assert "benign FedAvg accuracy 0.750, n=5" in report
    assert re.search(r"\| sign flip \| Δacc \| -0\.050 ± 0\.000 \(5\) \| 0\.700", report)
    assert "f=0.1 vanilla backdoor: capped-wmed − fltrust | ASR | 10 | -0.029" in report
    assert "vanilla backdoor fltrust: 1/class − 100/class | ASR | 5 | +0.440" in report
    assert "n<6: p floor 0.062" in report
    # Overhead: warm-up dropped, server = scoring + aggregation + server training.
    assert "| fltrust | 45 | 2.000 ± 0.000 | 0.300 ± 0.000 | 0.100 ± 0.000 | 0.000 ± 0.000 | 0.400 | +0.000 |" in report
    assert "0.400 s | 5.0 + 10.0 MiB" in report
    assert "| cifar10 | 878,538 |" in report

    out = tmp_path / "e3.md"
    dump = tmp_path / "e3.json"
    assert main(["--results-root", str(results), "--out", str(out), "--json", str(dump)]) == 0
    assert out.read_text() == report
    records = json.loads(dump.read_text())
    assert len(records) == sum(len(v) for v in cells.values())
    assert any(r.get("timing_mean", {}).get("round_s") == pytest.approx(3.0) for r in records)


def test_main_refuses_to_write_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for flag in ("--out", "--json"):
        with pytest.raises(SystemExit):
            main(["--results-root", str(results), flag, str(results / "x")])


# ----------------------------------------------------------------------------
# Sweep configs
# ----------------------------------------------------------------------------


def _compose(experiment: str, overrides: list[str]) -> dict:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config",
                      overrides=[f"+experiment={experiment}", f"+tag=e3_{experiment}", *overrides])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


@pytest.mark.parametrize(
    ("experiment", "overrides", "expected"),
    [
        ("clean_label_cifar10_f20", ["+baseline=fltrust"],
         _ck("clean_label", None, 0.2, "fltrust")),
        ("dirty_label_cifar10_f10", ["+baseline=coordinate_median"],
         _ck("dirty_label", 1.0, 0.1, "median")),
        ("sign_flip_cifar10_f20", ["+defense=topsis_capped_weighted_median"],
         _ck("sign_flip", 1.0, 0.2, "capped-wmed")),
        ("fltrust_backdoor_cifar10_f20", ["+defense=topsis_score_weight",
                                          "attack.update_manipulator.cosine_target=0.5"],
         _ck("direction_aligned", 0.5, 0.2, "ScoreWeight")),
        ("femnist_gradient_scaling_f20", ["+baseline=fltrust"],
         _ck("gradient_scaling", 5.0, 0.2, "fltrust", dataset="femnist")),
        ("adaptive_backdoor_cifar10_f10_root1", ["+baseline=fltrust"],
         _ck("adaptive", 0.6, 0.1, "fltrust", val=1)),
        ("backdoor_cifar10_f10_root10", ["+defense=topsis_capped_weighted_median"],
         _ck("backdoor", None, 0.1, "capped-wmed", val=10)),
        ("overhead_femnist", ["+baseline=krum"],
         _ck("backdoor", None, 0.1, "krum", dataset="femnist", rounds=OVERHEAD_ROUNDS)),
        ("overhead_memory_cifar10", ["+baseline=trimmed_mean"],
         _ck("backdoor", None, 0.1, "trimmed_mean", rounds=5, memory=True)),
    ],
)
def test_sweep_configs_compose_into_the_expected_cells(experiment, overrides, expected) -> None:
    assert cell_key({"seed": 0, "config": _compose(experiment, overrides)}) == expected


def test_root_and_overhead_configs_differ_from_their_base_only_in_the_knob() -> None:
    for experiment, knob in (("backdoor_cifar10_f10_root1", ("dataset", "val_per_class")),
                             ("overhead_cifar10", ("federation", "num_rounds"))):
        base = _compose("backdoor_cifar10_f10", [])
        variant = _compose(experiment, [])
        for cfg in (base, variant):
            cfg.pop("experiment")
            cfg.pop("tag")
        group, key = knob
        assert variant[group][key] != base[group][key]
        variant[group][key] = base[group][key]
        assert variant == base
    femnist = _compose("femnist_backdoor_f10", [])
    memory = _compose("overhead_memory_femnist", [])
    for cfg in (femnist, memory):
        cfg.pop("experiment")
        cfg.pop("tag")
    assert memory["federation"].pop("profile_memory") is True
    memory["federation"]["num_rounds"] = femnist["federation"]["num_rounds"]
    assert memory == femnist


def test_sweep_script_experiments_exist() -> None:
    script = (REPO / "experiments" / "sweep_e3.sh").read_text()
    expected = [f"fltrust_backdoor_cifar10_{f}" for f in ("f10", "f20")]
    expected += [f"{a}_cifar10_{f}" for f in ("f10", "f20")
                 for a in ("gradient_scaling", "sign_flip", "dirty_label", "clean_label")]
    expected += [f"femnist_{a}_{f}" for f in ("f10", "f20")
                 for a in ("backdoor", "gradient_scaling", "dirty_label")]
    expected += [f"{a}_cifar10_f10_root{v}" for v in (1, 10) for a in ("backdoor", "adaptive_backdoor")]
    expected += [f"overhead_{d}" for d in ("cifar10", "femnist")]
    expected += [f"overhead_memory_{d}" for d in ("cifar10", "femnist")]
    for name in expected:
        assert (CONFIG_DIR / "experiment" / f"{name}.yaml").exists(), name
    for prefix in ("overhead_", "overhead_memory_", "_root", "fltrust_backdoor_cifar10_"):
        assert prefix in script


# ----------------------------------------------------------------------------
# SOTA baselines and detection rates
# ----------------------------------------------------------------------------


def test_new_baseline_labels_and_parameter_deviations() -> None:
    assert _key(_config(baseline={"name": "flame", "noise_factor": 0.001})) == _ck(defense="flame")
    assert _key(_config(baseline={"name": "mesas", "alpha": 0.03, "center": "median"})) == _ck(defense="mesas")
    assert _key(_config(baseline={"name": "mesas", "alpha": 0.03, "center": "mean"})) == _ck(
        defense="mesas(center=mean)"
    )
    assert _key(_config(baseline={"name": "backdoor_indicator", "eta": 1.0})) == _ck(defense="indicator")
    assert _key(_config(baseline={"name": "backdoor_indicator", "eta": 0.5})) == _ck(
        defense="indicator(eta=0.5)"
    )


def _ledger(rounds: int) -> list[dict]:
    rows = []
    for r in range(1, rounds + 1):
        # One attacker (zero weight in even rounds), three benign (one dropped
        # every round). Weights renormalise over the admitted clients.
        dropped_mal = r % 2 == 0
        admitted = 2 if dropped_mal else 3
        rows.append({"round": r, "client_id": "m", "weight": 0.0 if dropped_mal else 1 / admitted,
                     "is_malicious": True})
        rows.append({"round": r, "client_id": "b0", "weight": 0.0, "is_malicious": False})
        rows.append({"round": r, "client_id": "b1", "weight": 1 / admitted, "is_malicious": False})
        rows.append({"round": r, "client_id": "b2", "weight": 1 / admitted, "is_malicious": False})
    return rows


def test_detection_rates_from_ledger() -> None:
    from topfed.analysis.e3 import detection_rates

    tpr, fpr, share = detection_rates(_ledger(4))
    assert tpr == pytest.approx(0.5)
    assert fpr == pytest.approx(1 / 3)
    # Odd rounds: attacker holds 1/3 of the weight; even rounds: 0.
    assert share == pytest.approx((1 / 3 + 0.0) / 2)
    assert detection_rates(None) == (None, None, None)
    assert detection_rates([{"round": 1, "client_id": "x", "is_malicious": False}]) == (None, None, None)
    no_mal = [r for r in _ledger(2) if not r["is_malicious"]]
    tpr, fpr, share = detection_rates(no_mal)
    assert tpr is None and fpr == pytest.approx(1 / 3) and share is None


def test_load_cells_reads_defense_ledger(tmp_path: Path) -> None:
    results = tmp_path / "results"
    cfg = _config(baseline={"name": "flame", "noise_factor": 0.001})
    _write_run(results, "sp_backdoor_cifar10_f10__flame", "f1", 0, cfg, 0.2)
    run_dir = results / "sp_backdoor_cifar10_f10__flame" / "f1" / "seed_0"
    (run_dir / "defense_ledger.json").write_text(json.dumps(_ledger(NUM_ROUNDS)))
    _write_run(results, "backdoor_cifar10_f10", "u0", 0, _config(), 0.8)
    cells = load_cells(results)
    flame = cells[_ck(defense="flame")]
    assert flame[0].drop_tpr == pytest.approx(0.5)
    assert flame[0].drop_fpr == pytest.approx(1 / 3)
    undefended = cells[_ck()]
    assert undefended[0].drop_tpr is None and undefended[0].mal_share is None


def test_sota_panel_and_detection_table_render(tmp_path: Path) -> None:
    from topfed.analysis.e3 import SOTA_PANEL, table_detection, table_panel

    results = tmp_path / "results"
    for seed in range(5):
        _write_run(results, "sp_backdoor_cifar10_f10__flame", "f1", seed,
                   _config(baseline={"name": "flame", "noise_factor": 0.001}), 0.2)
        run_dir = results / "sp_backdoor_cifar10_f10__flame" / "f1" / f"seed_{seed}"
        (run_dir / "defense_ledger.json").write_text(json.dumps(_ledger(NUM_ROUNDS)))
        _write_run(results, "backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian", "a2", seed,
                   _config(defense=CAPPED), 0.04)
    cells = load_cells(results)
    panel = table_panel(cells, "cifar10", 0.1, SOTA_PANEL)
    header = panel.splitlines()[0]
    assert "flame ASR/Δacc" in header and "mesas ASR/Δacc" in header
    vanilla = next(line for line in panel.splitlines() if line.startswith("| vanilla backdoor"))
    assert "0.200 ± 0.000 (5)" in vanilla
    det = table_detection(cells, "cifar10", 0.1)
    assert "flame TPR" in det.splitlines()[0]
    vanilla_det = next(line for line in det.splitlines() if line.startswith("| vanilla backdoor"))
    assert "0.500 ± 0.000 (5)" in vanilla_det
    report = build_report(cells)
    assert "#### SOTA baselines, f = 0.1" in report and "#### Detection, f = 0.1" in report


# ----------------------------------------------------------------------------
# Alternative backbone (optional ``model`` block)
# ----------------------------------------------------------------------------

RESNET = {"name": "resnet18_gn", "groups": 2}


def test_model_block_and_learning_rate_give_their_own_cells() -> None:
    assert _key(_config()).model == "cnn"
    assert _key(_config()).local_lr == pytest.approx(0.01)
    rn = _key(_config(model=RESNET))
    assert rn is not None and rn.model == "resnet18_gn"
    assert rn != _key(_config())
    assert _key(_config(model={"name": "resnet18_gn", "groups": 4})).model == "resnet18_gn(groups=4)"
    fast = _key(_config(**{"federation.local_learning_rate": 0.05}))
    assert fast is not None and fast.local_lr == pytest.approx(0.05) and fast != _key(_config())
    # Default-valued fields keep the positional constructor the tests use.
    assert _ck() == CellKey("cifar10", "backdoor", None, 0.1, "undefended", 100, NUM_ROUNDS, False)


def test_resnet_sweep_cells_compose_apart_from_cnn_cells() -> None:
    cnn = cell_key({"seed": 0, "config": _compose("backdoor_cifar10_f20", ["+baseline=flame"])})
    rn = cell_key({"seed": 0, "config": _compose(
        "backdoor_cifar10_f20", ["+baseline=flame", "+model=resnet18_gn"])})
    assert cnn is not None and rn is not None
    assert cnn.model == "cnn" and rn.model == "resnet18_gn"
    assert cnn.defense == rn.defense == "flame" and cnn != rn
    adaptive = cell_key({"seed": 0, "config": _compose(
        "adaptive_backdoor_cifar10_f20",
        ["+model=resnet18_gn", "+baseline=fltrust_median",
         "attack.update_manipulator.target_norm=1.5", "federation.local_learning_rate=0.05"])})
    assert adaptive == CellKey("cifar10", "adaptive", 1.5, 0.2, "fltrust-median", 100, NUM_ROUNDS,
                               False, "resnet18_gn", 0.05)


def test_resnet_sweep_script_cells() -> None:
    script = (REPO / "experiments" / "sweep_resnet.sh").read_text()
    for name in ("benign_cifar10", "backdoor_cifar10_f20", "adaptive_backdoor_cifar10_f20"):
        assert (CONFIG_DIR / "experiment" / f"{name}.yaml").exists(), name
        assert name in script
    assert (CONFIG_DIR / "model" / "resnet18_gn.yaml").exists()
    for needle in ("+model=resnet18_gn", "+skip_existing=true", "rn_", "+baseline=fltrust_median",
                   "+baseline=flame", "+baseline=coordinate_median",
                   "+defense=topsis_capped_weighted_median", "target_norm", "local_learning_rate"):
        assert needle in script, needle


def test_model_table_keeps_resnet_out_of_the_cnn_tables(tmp_path: Path) -> None:
    from topfed.analysis.e3 import build_model_report, table_model, table_panel

    results = tmp_path / "results"
    f20 = {"attack.assignment.fraction": 0.2}
    for seed in range(5):
        # CNN cells at f=0.2 and the same cells under the ResNet, different numbers.
        _write_run(results, "backdoor_cifar10_f20", "c1", seed, _config(**f20), 0.80, acc=0.70)
        _write_run(results, "rn_backdoor_cifar10_f20", "r1", seed, _config(model=RESNET, **f20),
                   0.60, acc=0.80)
        _write_run(results, "rn_backdoor_cifar10_f20__flame", "r2", seed,
                   _config(model=RESNET, baseline={"name": "flame", "noise_factor": 0.001}, **f20),
                   0.30, acc=0.78)
        # The adaptive attacker only at the probe's budget 1.5 under the ResNet.
        _write_run(results, "rn_adaptive_backdoor_cifar10_f20_g1.5", "r3", seed,
                   _config("adaptive", 1.5, model=RESNET, **f20), 0.50, acc=0.79)
        _write_run(results, "rn_benign_cifar10", "r0", seed, _config("none", model=RESNET), None, acc=0.82)
    # A one-seed probe cell at another budget must not displace the five-seed cell.
    _write_run(results, "rn_adaptive_backdoor_cifar10_f20_g3.0", "r4", 0,
               _config("adaptive", 3.0, model=RESNET, **f20), 0.95, acc=0.70)
    # A ResNet run at another learning rate is a cell of its own.
    _write_run(results, "rn_backdoor_cifar10_f20_lr0.05", "r5", 0,
               _config(model=RESNET, **{**f20, "federation.local_learning_rate": 0.05}), 0.10, acc=0.85)

    cells = load_cells(results)
    cnn_panel = table_panel(cells, "cifar10", 0.2)
    vanilla = next(line for line in cnn_panel.splitlines() if line.startswith("| vanilla backdoor"))
    assert "0.800 ± 0.000 (5)" in vanilla and "0.600" not in vanilla

    table = table_model(cells, "resnet18_gn")
    header = table.splitlines()[0]
    assert "undefended ASR" in header and "flame clean acc" in header
    vanilla_rn = next(line for line in table.splitlines() if line.startswith("| vanilla backdoor"))
    assert "0.600 ± 0.000 (5) | 0.800 ± 0.000 (5)" in vanilla_rn
    assert "0.300 ± 0.000 (5) | 0.780 ± 0.000 (5)" in vanilla_rn
    adaptive = next(line for line in table.splitlines() if line.startswith("| norm-bounded adaptive"))
    assert "0.500 ± 0.000 (5) †1.5 | 0.790 ± 0.000 (5)" in adaptive
    assert "0.950" not in adaptive
    benign = next(line for line in table.splitlines() if line.startswith("| benign"))
    assert "0.820 (n=5)" in benign

    fast = table_model(cells, "resnet18_gn", local_lr=0.05)
    vanilla_fast = next(line for line in fast.splitlines() if line.startswith("| vanilla backdoor"))
    assert "0.100 ± 0.000 (1) | 0.850 ± 0.000 (1)" in vanilla_fast

    report = build_model_report(cells, "resnet18_gn")
    assert report.startswith("# resnet18_gn (local lr 0.01)")
    assert "Runs found: 21 in 5 cells." in report
    out = tmp_path / "rn.md"
    assert main(["--results-root", str(results), "--model", "resnet18_gn", "--out", str(out)]) == 0
    assert out.read_text() == report
    assert "## f = 0.2" in report and "## f = 0.1" not in report

    # A second compromise level gets a panel of its own.
    _write_run(results, "rn_backdoor_cifar10_f10", "r6", 0,
               _config(model=RESNET, **{"attack.assignment.fraction": 0.1}), 0.20, acc=0.81)
    report = build_model_report(load_cells(results), "resnet18_gn")
    f10, f20_panel = report.split("## f = 0.1")[1].split("## f = 0.2")
    assert "0.200 ± 0.000 (1) | 0.810 ± 0.000 (1)" in f10
    assert "0.600 ± 0.000 (5) | 0.800 ± 0.000 (5)" in f20_panel


def test_capped_root_label_and_panel() -> None:
    from topfed.analysis.e3 import DEFENSE_OVERRIDES, ROOT_PANEL

    assert _key(_config(baseline={"name": "fltrust_median_capped", "cap": 1.0})) == _ck(
        defense="fltrust-median-capped"
    )
    assert _key(_config(baseline={"name": "fltrust_median_capped", "cap": 2.0})) == _ck(
        defense="fltrust-median-capped(cap=2.0)"
    )
    assert DEFENSE_OVERRIDES["fltrust-median-capped"] == "+baseline=fltrust_median_capped"
    assert ROOT_PANEL == ("capped-wmed", "fltrust", "fltrust-median", "fltrust-median-capped")


# ----------------------------------------------------------------------------
# CIFAR-100 (fine and superclass variants)
# ----------------------------------------------------------------------------


def _cifar100(kind: str = "backdoor", strength: float | None = None, /, coarse: bool = False,
              **over: Any) -> dict[str, Any]:
    dataset = ({"name": "cifar100_coarse", "num_classes": 20, "val_per_class": 50, "coarse": True}
               if coarse else
               {"name": "cifar100", "num_classes": 100, "val_per_class": 10, "coarse": False})
    return _config(kind, strength, **{"dataset": {**dataset, "augment": True}, **over})


def test_cifar100_variants_are_separate_reference_cells() -> None:
    fine = _key(_cifar100())
    coarse = _key(_cifar100(coarse=True))
    assert fine == CellKey("cifar100", "backdoor", None, 0.1, "undefended", 10, NUM_ROUNDS, False)
    assert coarse == CellKey("cifar100_coarse", "backdoor", None, 0.1, "undefended", 50, NUM_ROUNDS,
                             False)
    adaptive = _key(_cifar100("adaptive", 1.0, coarse=True, baseline=MEDIAN,
                              **{"attack.assignment.fraction": 0.2}))
    assert adaptive == CellKey("cifar100_coarse", "adaptive", 1.0, 0.2, "median", 50, NUM_ROUNDS, False)
    # A non-reference partition stays out of scope for the new datasets too.
    assert _key(_cifar100(**{"partition.alpha": 0.1})) is None


@pytest.mark.parametrize("dataset", ["cifar100", "cifar100_coarse"])
@pytest.mark.parametrize(
    ("experiment", "overrides", "attack", "strength", "fraction", "defense"),
    [
        ("benign_cifar100", [], "benign", None, 0.0, "undefended"),
        ("backdoor_cifar100_f10", ["+baseline=clipped_mean"], "backdoor", None, 0.1, "clipped-mean"),
        ("backdoor_cifar100_f20", ["+baseline=fltrust_median"], "backdoor", None, 0.2, "fltrust-median"),
        ("adaptive_backdoor_cifar100_f10", ["+defense=topsis_capped_weighted_median",
                                            "attack.update_manipulator.target_norm=1.5"],
         "adaptive", 1.5, 0.1, "capped-wmed"),
        ("adaptive_backdoor_cifar100_f20", ["+baseline=flame"], "adaptive", 0.6, 0.2, "flame"),
    ],
)
def test_cifar100_sweep_configs_compose_into_their_cells(
    dataset, experiment, overrides, attack, strength, fraction, defense
) -> None:
    from topfed.analysis.e3 import REFERENCE_VAL

    cfg = _compose(experiment, [f"dataset={dataset}", *overrides])
    assert cell_key({"seed": 0, "config": cfg}) == CellKey(
        dataset, attack, strength, fraction, defense, REFERENCE_VAL[dataset], NUM_ROUNDS, False)


def test_cifar100_sweep_script_cells() -> None:
    script = (REPO / "experiments" / "sweep_cifar100.sh").read_text()
    for name in ("benign_cifar100", "backdoor_cifar100_f10", "backdoor_cifar100_f20",
                 "adaptive_backdoor_cifar100_f10", "adaptive_backdoor_cifar100_f20"):
        assert (CONFIG_DIR / "experiment" / f"{name}.yaml").exists(), name
    # The grid spells the compromise level as a loop variable.
    for needle in ("benign_cifar100", "backdoor_cifar100_${f}", "adaptive_backdoor_cifar100_${f}",
                   "for f in f10 f20"):
        assert needle in script, needle
    for name in ("cifar100", "cifar100_coarse"):
        assert (CONFIG_DIR / "dataset" / f"{name}.yaml").exists(), name
    for needle in ("dataset=${DATASET}", "+skip_existing=true", "c100", "c100c",
                   "+defense=topsis_capped_weighted_median", "+baseline=fltrust_median",
                   "+baseline=fltrust", "+baseline=coordinate_median", "+baseline=flame",
                   "+baseline=clipped_mean", "target_norm", "0.3 0.6 1.0 1.5 3.0"):
        assert needle in script, needle


def test_cifar100_communication_rows_use_the_variant_head() -> None:
    from topfed.analysis.e3 import table_communication
    from topfed.models.cifar_cnn import CifarCNN

    for dataset, classes in (("cifar100", 100), ("cifar100_coarse", 20)):
        numel, nbytes = model_state_bytes(dataset)
        assert numel == sum(t.numel() for t in CifarCNN(num_classes=classes).state_dict().values())
        assert nbytes == 4 * numel
    assert model_state_bytes("cifar100")[0] > model_state_bytes("cifar100_coarse")[0] > \
        model_state_bytes("cifar10")[0]
    table = table_communication()
    assert "| cifar100 |" in table and "| cifar100_coarse |" in table and "| cifar10 |" in table


def test_cifar100_cells_render_in_their_own_report_sections(tmp_path: Path) -> None:
    from topfed.analysis.e3 import report_datasets, table_panel

    results = tmp_path / "results"
    _populate(results)
    # Without CIFAR-100 runs the report keeps its two core sections.
    cells = load_cells(results)
    assert report_datasets(cells) == ("cifar10", "femnist")
    report = build_report(cells)
    assert "## cifar10 (CIFAR-10): attack by defense" in report
    assert "## femnist (FEMNIST): attack by defense" in report
    assert "cifar100" not in report.split("## Server overhead")[0]

    f20 = {"attack.assignment.fraction": 0.2}
    for seed in range(5):
        _write_run(results, "c100_benign_cifar100", "h0", seed, _cifar100("none"), None, acc=0.33)
        _write_run(results, "c100_backdoor_cifar100_f20", "h1", seed, _cifar100(**f20), 0.90, acc=0.30)
        _write_run(results, "c100_backdoor_cifar100_f20__flame", "h2", seed,
                   _cifar100(baseline={"name": "flame", "noise_factor": 0.001}, **f20), 0.40, acc=0.28)
        _write_run(results, "c100_adaptive_backdoor_cifar100_f20_g1.0", "h3", seed,
                   _cifar100("adaptive", 1.0, **f20), 0.70, acc=0.31)
        _write_run(results, "c100c_benign_cifar100", "k0", seed, _cifar100("none", coarse=True),
                   None, acc=0.50)
        _write_run(results, "c100c_backdoor_cifar100_f20__coordinate_median", "k1", seed,
                   _cifar100(baseline=MEDIAN, coarse=True, **f20), 0.20, acc=0.48)
    cells = load_cells(results)
    assert report_datasets(cells) == ("cifar10", "femnist", "cifar100", "cifar100_coarse")

    fine = table_panel(cells, "cifar100", 0.2, ("undefended", "flame"))
    vanilla = next(line for line in fine.splitlines() if line.startswith("| vanilla backdoor"))
    assert "0.900 ± 0.000 (5) | 0.300 ± 0.000 (5) | 0.400 ± 0.000 (5) | 0.280 ± 0.000 (5)" in vanilla
    adaptive = next(line for line in fine.splitlines() if line.startswith("| norm-bounded adaptive"))
    # The only budget run is the table's best response and is marked as such.
    assert "0.700 ± 0.000 (5) †1 | 0.310 ± 0.000 (5)" in adaptive
    assert [line.split("|")[1].strip() for line in fine.splitlines()[2:-1]] == [
        "vanilla backdoor", "norm-bounded adaptive"]

    coarse = table_panel(cells, "cifar100_coarse", 0.2, ("undefended", "median"))
    vanilla_c = next(line for line in coarse.splitlines() if line.startswith("| vanilla backdoor"))
    assert "| – | – | 0.200 ± 0.000 (5) | 0.480 ± 0.000 (5)" in vanilla_c

    report = build_report(cells)
    fine_section = report.split("## cifar100 (CIFAR-100 (100 fine classes)): attack by defense")[1]
    assert "benign FedAvg accuracy 0.330, n=5" in fine_section.split("## cifar100_coarse")[0]
    coarse_section = report.split("## cifar100_coarse (CIFAR-100 (20 superclasses)): attack by defense")[1]
    assert "benign FedAvg accuracy 0.500, n=5" in coarse_section.split("## Best-response")[0]
    assert "### cifar100_coarse" in report.split("## Paired comparisons")[1]
