"""End-to-end runner smoke for the FLAME, MESAS and BackdoorIndicator baselines.

A tiny synthetic CIFAR bundle replaces the dataset and random tensors replace
the indicator's OOD images, so the test checks the Hydra composition, the
baseline construction in the runner, the server loop (including the
pre-round hook) and the ledger output, not the defenses' accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from experiments.run import run_from_cfg
from topfed.data.cifar10 import CifarBundle

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _tiny_bundle() -> CifarBundle:
    g = torch.Generator().manual_seed(0)
    return CifarBundle(
        train_x=torch.randn(80, 3, 32, 32, generator=g),
        train_y=torch.randint(0, 10, (80,), generator=g),
        val_x=torch.randn(20, 3, 32, 32, generator=g),
        val_y=torch.arange(20) % 10,
        test_x=torch.randn(40, 3, 32, 32, generator=g),
        test_y=torch.arange(40) % 10,
    )


def _fake_ood(*args, **kwargs) -> torch.Tensor:
    return torch.randn(64, 3, 32, 32, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize(
    ("baseline", "extra", "ledger_keys"),
    [
        ("flame", [], {"cluster", "clip_factor", "admitted", "clip_bound"}),
        ("mesas", [], {"retained", "pruned_pass", "m_cos", "m_eucl", "m_count"}),
        (
            "backdoor_indicator",
            ["baseline.retrain_epochs=2", "baseline.batch_size=16"],
            {"indicator_score", "indicator_class", "admitted"},
        ),
        ("fltrust_median_capped", [], set()),
        ("clipped_mean", [], {"delta_norm", "clip_factor", "clip_bound"}),
    ],
)
def test_runner_new_baselines(monkeypatch, tmp_path, baseline, extra, ledger_keys) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _tiny_bundle())
    monkeypatch.setattr("experiments.run.load_ood_images", _fake_ood)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=backdoor_cifar10_f20",
                f"+baseline={baseline}",
                "federation.num_clients=10",
                "federation.clients_per_round=4",
                "federation.num_rounds=2",
                "federation.local_epochs=1",
                "federation.local_batch_size=8",
                f"data_root={tmp_path / 'data'}",
                f"results_root={tmp_path / 'results'}",
                *extra,
            ],
        )

    out_dir = run_from_cfg(cfg)
    assert out_dir.parent.parent.name == f"backdoor_cifar10_f20__{baseline}"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["baseline"]["name"] == baseline
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 2
    if baseline == "backdoor_indicator":
        # The pre-round injection is timed as server training.
        assert all(m["server_train_s"] > 0 for m in metrics)

    if baseline == "fltrust_median_capped":
        # Server-rooted rules write a ledger only when the run traces capture.
        return
    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    assert len(ledger) == 2 * 4
    assert ledger_keys | {"round", "client_id", "weight", "is_malicious"} <= set(ledger[0])
    for rnd in (1, 2):
        rows = [r for r in ledger if r["round"] == rnd]
        total = sum(r["weight"] for r in rows)
        assert total == pytest.approx(1.0) or total == 0.0
