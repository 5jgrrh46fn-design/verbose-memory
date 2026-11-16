"""End-to-end runner tests for the E4 hybrid defenses on a stubbed dataset.

Covers, through Hydra composition and ``run_from_cfg``: the server-gated
capped weighted median, the server-cosine sixth criterion, FLTrust on a
weighted-median base, the M4-band control, and the composed
(direction-aligned then norm-bounded) attack.
"""

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
        "federation.num_rounds=3",
        "federation.local_epochs=1",
        "federation.local_batch_size=8",
        f"data_root={tmp_path / 'data'}",
        f"results_root={tmp_path / 'results'}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides + common)


COMPOSED = [
    "+experiment=composed_backdoor_cifar10_f20",
    "attack.update_manipulator.manipulators.0.cosine_target=0.5",
    "attack.update_manipulator.manipulators.1.target_norm=1.0",
]


@pytest.mark.parametrize(
    ("override", "label", "server_trains", "ledger_keys"),
    [
        ("+defense=topsis_server_gated_capped_wmedian",
         "ServerGatedCappedScoreWeight_critic_wmedian", True,
         {"server_cosine", "gate", "capped_weight"}),
        ("+defense=topsis_rooted_capped_weighted_median",
         "CappedScoreWeight_critic_rooted_wmedian", True, {"m6_server_cosine"}),
        ("+defense=topsis_m4band_capped_wmedian",
         "M4BandCappedScoreWeight_critic_wmedian", False, {"m4", "m4_band", "gate"}),
        ("+baseline=fltrust_median", "fltrust_median", True, None),
    ],
)
def test_e4_variants_run_against_the_composed_attack(
    monkeypatch, tmp_path, override, label, server_trains, ledger_keys
) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose([*COMPOSED, "+tag=e4test", override], tmp_path)
    out_dir = run_from_cfg(cfg)
    assert out_dir.parts[-3] == f"e4test__{label}"

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 3 and "asr_accuracy" in metrics[-1]
    assert all((row["server_train_s"] > 0.0) == server_trains for row in metrics)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    stages = manifest["config"]["attack"]["update_manipulator"]["manipulators"]
    assert stages[0]["cosine_target"] == 0.5 and stages[1]["target_norm"] == 1.0

    ledger_path = out_dir / "defense_ledger.json"
    if ledger_keys is None:
        assert not ledger_path.exists()
    else:
        ledger = json.loads(ledger_path.read_text())
        assert len(ledger) == 3 * 4
        assert all(ledger_keys <= set(row) for row in ledger)


def test_rooted_criterion_rejects_a_fixed_template_weighting(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        ["+experiment=backdoor_cifar10_f10", "+defense=topsis_rooted_capped_weighted_median",
         "defense.scorer.weighting=consistency"],
        tmp_path,
    )
    with pytest.raises(SystemExit, match="server-cosine"):
        run_from_cfg(cfg)
