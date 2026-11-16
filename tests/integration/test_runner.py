"""End-to-end runner test using a stubbed CIFAR-10 bundle.

Exercises Hydra config composition, partitioning, malicious-set assignment,
client/evaluator construction, the FedAvg loop, and manifest/metrics
persistence without touching the network or the real dataset.
"""

from __future__ import annotations

import json

import pytest
import torch
from hydra import compose, initialize_config_dir

from experiments.run import run_from_cfg
from topfed.data.cifar10 import CifarBundle


CONFIG_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[2] / "configs"
)


def _tiny_bundle() -> CifarBundle:
    g = torch.Generator().manual_seed(0)
    train_x = torch.randn(80, 3, 32, 32, generator=g)
    train_y = torch.randint(0, 10, (80,), generator=g)
    val_x = torch.randn(20, 3, 32, 32, generator=g)
    val_y = torch.arange(20) % 10
    test_x = torch.randn(40, 3, 32, 32, generator=g)
    test_y = torch.arange(40) % 10
    return CifarBundle(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )


@pytest.mark.parametrize(
    "experiment, expected_malicious",
    [
        ("backdoor_cifar10_f10", 1),  # ceil(10 * 0.10) = 1
        ("backdoor_cifar10_f20", 2),  # ceil(10 * 0.20) = 2
    ],
)
def test_runner_e2e_smoke(
    monkeypatch, tmp_path, experiment: str, expected_malicious: int
) -> None:
    monkeypatch.setattr(
        "experiments.run.load_cifar10", lambda *a, **kw: _tiny_bundle()
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+experiment={experiment}",
                "federation.num_clients=10",
                "federation.clients_per_round=4",
                "federation.num_rounds=2",
                "federation.local_epochs=1",
                "federation.local_batch_size=8",
                f"data_root={tmp_path / 'data'}",
                f"results_root={tmp_path / 'results'}",
            ],
        )

    out_dir = run_from_cfg(cfg)

    manifest_path = out_dir / "manifest.json"
    metrics_path = out_dir / "metrics.json"
    client_metrics_path = out_dir / "client_metrics.json"
    assert manifest_path.exists()
    assert metrics_path.exists()
    assert client_metrics_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 0
    assert manifest["num_malicious"] == expected_malicious
    assert manifest["config"]["experiment"] == experiment

    metrics = json.loads(metrics_path.read_text())
    assert len(metrics) == 2
    keys = set(metrics[-1].keys())
    assert {"round", "val_loss", "val_accuracy", "asr_accuracy"} <= keys

    client_rows = json.loads(client_metrics_path.read_text())
    # 2 rounds * 4 clients sampled = 8 rows.
    assert len(client_rows) == 2 * 4
    expected_cols = {
        "round",
        "client_id",
        "is_malicious",
        "update_norm",
        "m1_cosine_to_mean",
        "m2_norm_ratio",
        "m3_temporal_cv",
        "m4_val_loss_delta",
        "m5_max_layer_deviation",
    }
    assert expected_cols <= set(client_rows[0].keys())
    # Malicious flag matches the manifest's malicious_ids list.
    malicious_ids = set(manifest["malicious_ids"])
    for row in client_rows:
        assert row["is_malicious"] == (row["client_id"] in malicious_ids)


def test_runner_benign_baseline(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "experiments.run.load_cifar10", lambda *a, **kw: _tiny_bundle()
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=benign_cifar10",
                "federation.num_clients=10",
                "federation.clients_per_round=4",
                "federation.num_rounds=2",
                "federation.local_epochs=1",
                "federation.local_batch_size=8",
                f"data_root={tmp_path / 'data'}",
                f"results_root={tmp_path / 'results'}",
            ],
        )

    out_dir = run_from_cfg(cfg)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["num_malicious"] == 0
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # Benign run has no asr key.
    assert "asr_accuracy" not in metrics[-1]
    assert "val_accuracy" in metrics[-1]
