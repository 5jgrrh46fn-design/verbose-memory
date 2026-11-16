"""End-to-end runner tests for the LIE and Fang attacks on a stubbed dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from experiments.run import run_from_cfg
from tests.integration.test_runner_e5 import _cifar_bundle

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _compose(overrides: list[str], tmp_path: Path):
    common = [
        "federation.num_clients=10",
        "federation.clients_per_round=4",
        "federation.num_rounds=2",
        "federation.local_epochs=1",
        "federation.local_batch_size=8",
        f"data_root={tmp_path / 'data'}",
        f"results_root={tmp_path / 'results'}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides + common)


@pytest.mark.parametrize("attack", ["lie", "fang"])
@pytest.mark.parametrize(
    ("rule", "label"),
    [
        ([], ""),
        (["+baseline=coordinate_median"], "__coordinate_median"),
        (["+baseline=trimmed_mean"], "__trimmed_mean"),
        (["+defense=topsis_capped_weighted_median"], "__CappedScoreWeight_critic_wmedian"),
    ],
)
def test_statistical_attacks_run_against_the_median_rules(
    monkeypatch, tmp_path, attack, rule, label
) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose([f"+experiment={attack}_cifar10_f20", "+tag=stattest", *rule], tmp_path)
    out_dir = run_from_cfg(cfg)
    assert out_dir.parts[-3] == f"stattest{label}"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["attack"]["kind"] == attack
    assert manifest["num_malicious"] == 2
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 2
    assert all(0.0 <= m["test_accuracy"] <= 1.0 for m in metrics)
    # Untargeted: no trigger, so no attack-success metric.
    assert "asr_accuracy" not in metrics[-1]


@pytest.mark.parametrize(
    ("experiment", "extra"),
    [
        ("alie_backdoor_cifar10_f20", ["attack.update_manipulator.num_std=0.5"]),
        ("minmax_backdoor_cifar10_f20", []),
    ],
)
@pytest.mark.parametrize(
    ("rule", "label"),
    [
        ([], ""),
        (["+baseline=coordinate_median"], "__coordinate_median"),
        (["+baseline=fltrust_median"], "__fltrust_median"),
        (["+defense=topsis_capped_weighted_median"], "__CappedScoreWeight_critic_wmedian"),
    ],
)
def test_in_range_backdoors_run_against_the_median_rules(
    monkeypatch, tmp_path, experiment, extra, rule, label
) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose([f"+experiment={experiment}", "+tag=inrange", *extra, *rule], tmp_path)
    out_dir = run_from_cfg(cfg)
    assert out_dir.parts[-3] == f"inrange{label}"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["attack"]["kind"] == "backdoor"
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 2
    # A backdoor: the attack-success metric is reported.
    assert all(0.0 <= m["asr_accuracy"] <= 1.0 for m in metrics)
