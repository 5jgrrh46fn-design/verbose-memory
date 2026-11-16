"""End-to-end runner smoke for the optional ResNet-18 (GroupNorm) model.

A tiny synthetic CIFAR bundle replaces the dataset, so the test checks that
``+model=resnet18_gn`` composes, that the runner builds the ResNet for the
clients, the evaluator and the server, and that the defenses and baselines
run on its state dict; it says nothing about accuracy. It also pins the
config hash of a frozen cell, which the optional model block must not move.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.run import _load_dataset, run_from_cfg
from topfed.core.types import FederationConfig
from topfed.data.cifar10 import CifarBundle
from topfed.models.resnet_cifar import ResNet18GN
from topfed.runtime.paths import config_hash

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SMALL = [
    "federation.num_clients=10",
    "federation.clients_per_round=4",
    "federation.num_rounds=2",
    "federation.local_epochs=1",
    "federation.local_batch_size=8",
]


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


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides)


def _hash(cfg) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(c, dict)
    c["federation"].pop("seed", None)
    c.pop("skip_existing", None)
    c.pop("reproducibility", None)
    return config_hash(OmegaConf.to_yaml(c))


def test_frozen_cell_hash_unchanged_without_model_override() -> None:
    # The hash of a frozen cell as it exists under results/; the optional
    # model group must leave it alone.
    assert _hash(_compose(["+experiment=backdoor_cifar10_f20"])) == "e32dce1ee2a661fa"
    assert _hash(_compose(["+experiment=benign_cifar10"])) == "9f894f17f32b62c5"
    with_model = _compose(["+experiment=backdoor_cifar10_f20", "+model=resnet18_gn"])
    assert _hash(with_model) != "e32dce1ee2a661fa"
    assert OmegaConf.to_container(with_model.model) == {"name": "resnet18_gn", "groups": 2}


def test_model_block_selects_resnet_for_cifar_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _tiny_bundle())
    cfg = _compose(["+experiment=benign_cifar10", "+model=resnet18_gn", "model.groups=4", *SMALL])
    fed = FederationConfig(**{k: v for k, v in OmegaConf.to_container(cfg.federation).items()
                              if k != "_target_"})
    model = _load_dataset(cfg, fed).model_factory()
    assert isinstance(model, ResNet18GN) and model.groups == 4

    plain = _load_dataset(_compose(["+experiment=benign_cifar10", *SMALL]), fed).model_factory()
    assert not isinstance(plain, ResNet18GN)

    femnist = _compose(["+experiment=femnist_benign", "+model=resnet18_gn", *SMALL])
    with pytest.raises(SystemExit, match="cifar10 only"):
        _load_dataset(femnist, fed)
    unknown = _compose(["+experiment=benign_cifar10", "+model=resnet18_gn", "model.name=vgg", *SMALL])
    with pytest.raises(SystemExit, match="unknown model"):
        _load_dataset(unknown, fed)


@pytest.mark.parametrize(
    ("overrides", "suffix"),
    [
        ([], ""),
        (["+defense=topsis_capped_weighted_median"], "__CappedScoreWeight_critic_wmedian"),
        (["+baseline=flame"], "__flame"),
    ],
)
def test_runner_two_rounds_with_resnet(monkeypatch, tmp_path, overrides, suffix) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _tiny_bundle())
    cfg = _compose([
        "+experiment=backdoor_cifar10_f20",
        "+model=resnet18_gn",
        *overrides,
        *SMALL,
        f"data_root={tmp_path / 'data'}",
        f"results_root={tmp_path / 'results'}",
    ])
    out_dir = run_from_cfg(cfg)
    assert out_dir.parent.parent.name == f"backdoor_cifar10_f20{suffix}"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["model"] == {"name": "resnet18_gn", "groups": 2}
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 2
    assert all(0.0 <= m["test_accuracy"] <= 1.0 for m in metrics)
    if overrides:
        ledger = json.loads((out_dir / "defense_ledger.json").read_text())
        assert len(ledger) == 2 * 4
