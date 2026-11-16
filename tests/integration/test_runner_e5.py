"""End-to-end runner tests for the E5 diagnostic knobs on stubbed datasets.

Covers, through Hydra composition and ``run_from_cfg``: forced co-selection,
the oracle integrator on the weighted-median base, a windowed attack, a
centred trigger, a pooled Dirichlet split of a writer-partitioned dataset,
the padded-border trigger on CIFAR-shaped data, a run tag, the timing
columns in ``metrics.json``, and the E3 memory-profiling and root-set configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from experiments.run import run_from_cfg
from topfed.data.cifar10 import CifarBundle
from topfed.data.femnist import FemnistBundle

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
NUM_CLIENTS = 10


def _femnist_bundle() -> FemnistBundle:
    g = torch.Generator().manual_seed(0)
    n = 120
    return FemnistBundle(
        train_x=torch.randn(n, 1, 28, 28, generator=g),
        train_y=torch.randint(0, 62, (n,), generator=g),
        train_writers=np.repeat(np.arange(NUM_CLIENTS), n // NUM_CLIENTS),
        val_x=torch.randn(20, 1, 28, 28, generator=g),
        val_y=torch.arange(20) % 62,
        test_x=torch.randn(40, 1, 28, 28, generator=g),
        test_y=torch.arange(40) % 62,
    )


def _cifar_bundle() -> CifarBundle:
    g = torch.Generator().manual_seed(0)
    return CifarBundle(
        train_x=torch.randn(80, 3, 32, 32, generator=g),
        train_y=torch.randint(0, 10, (80,), generator=g),
        val_x=torch.randn(20, 3, 32, 32, generator=g),
        val_y=torch.arange(20) % 10,
        test_x=torch.randn(40, 3, 32, 32, generator=g),
        test_y=torch.arange(40) % 10,
    )


def _compose(overrides: list[str], tmp_path: Path):
    common = [
        f"federation.num_clients={NUM_CLIENTS}",
        "federation.clients_per_round=4",
        "federation.num_rounds=3",
        "federation.local_epochs=1",
        "federation.local_batch_size=8",
        f"data_root={tmp_path / 'data'}",
        f"results_root={tmp_path / 'results'}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides + common)


def test_femnist_forced_oracle_window_center_dirichlet(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_femnist", lambda *a, **kw: _femnist_bundle())
    cfg = _compose(
        [
            "+experiment=femnist_backdoor_f20",
            "+tag=e5test_femnist",
            "+defense=oracle_filter_wmedian",
            "+federation.force_malicious_per_round=2",
            "partition=dirichlet_alpha_0.5",
            "partition.alpha=1.0",
            "+attack.poisoner.trigger_position=center",
            "attack.schedule._target_=topfed.attacks.schedule.WindowedAttack",
            "+attack.schedule.start_round=2",
            "+attack.schedule.end_round=2",
        ],
        tmp_path,
    )
    out_dir = run_from_cfg(cfg)

    # The tag replaces the experiment name in the label; the defense suffix stays.
    assert out_dir.parts[-3] == "e5test_femnist__OracleFilter_critic_wmedian"

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["num_malicious"] == 2  # ceil(10 * 0.2)
    cfg_out = manifest["config"]
    assert cfg_out["federation"]["force_malicious_per_round"] == 2
    assert cfg_out["partition"] == {"kind": "dirichlet", "alpha": 1.0, "seed": 0}
    assert cfg_out["attack"]["poisoner"]["trigger_position"] == "center"
    assert cfg_out["attack"]["schedule"]["end_round"] == 2

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 3
    for row in metrics:
        assert {"asr_accuracy", "test_accuracy", "client_train_s", "scoring_s",
                "aggregation_s", "server_train_s", "evaluation_s", "round_s"} <= set(row)
        assert row["round_s"] >= row["client_train_s"] + row["scoring_s"]
        assert row["scoring_s"] > 0.0

    # Exactly two attackers in every 4-client round.
    rows = json.loads((out_dir / "client_metrics.json").read_text())
    per_round = {}
    for r in rows:
        per_round[r["round"]] = per_round.get(r["round"], 0) + int(r["is_malicious"])
    assert per_round == {1: 2, 2: 2, 3: 2}

    # The oracle zero-weights the attackers every round and includes the rest.
    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    assert all((e["weight"] == 0.0) == e["is_malicious"] for e in ledger)
    assert all(e["included"] == (not e["is_malicious"]) for e in ledger)


def test_femnist_natural_partition_still_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_femnist", lambda *a, **kw: _femnist_bundle())
    cfg = _compose(["+experiment=femnist_backdoor_f20", "+defense=topsis_capped_weighted_median"], tmp_path)
    out_dir = run_from_cfg(cfg)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["partition"]["kind"] == "natural"
    assert "force_malicious_per_round" not in manifest["config"]["federation"]
    assert out_dir.parts[-3] == "femnist_backdoor_f20__CappedScoreWeight_critic_wmedian"


def test_cifar_padded_border_trigger(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        [
            "+experiment=backdoor_cifar10_f20",
            "+tag=e5test_cifar_border",
            "+defense=topsis_capped_weighted_median",
            "+dataset.pad_border=4",
            "attack.poisoner.trigger_size=4",
            "+attack.poisoner.trigger_position=border",
        ],
        tmp_path,
    )
    out_dir = run_from_cfg(cfg)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 3 and "asr_accuracy" in metrics[-1]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["dataset"]["pad_border"] == 4
    assert (out_dir / "defense_ledger.json").exists()


def test_cifar_padded_corner_patch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        [
            "+experiment=backdoor_cifar10_f20",
            "+dataset.pad_border=4",
            "attack.poisoner.trigger_size=4",
        ],
        tmp_path,
    )
    out_dir = run_from_cfg(cfg)
    assert len(json.loads((out_dir / "metrics.json").read_text())) == 3


def test_cifar_memory_profiling_and_root_set_configs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(["+experiment=overhead_memory_cifar10", "+tag=e3test_mem", "+baseline=fltrust"],
                   tmp_path)
    out_dir = run_from_cfg(cfg)
    assert out_dir.parts[-3] == "e3test_mem__fltrust"
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 3
    assert all(row["server_peak_host_mb"] > 0.0 for row in metrics)
    assert all(row["server_train_s"] > 0.0 for row in metrics)

    cfg = _compose(["+experiment=adaptive_backdoor_cifar10_f10_root1", "+tag=e3test_root",
                    "+defense=topsis_capped_weighted_median"], tmp_path)
    out_dir = run_from_cfg(cfg)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["dataset"]["val_per_class"] == 1
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "server_peak_host_mb" not in metrics[-1]
