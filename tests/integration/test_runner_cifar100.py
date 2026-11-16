"""End-to-end runner smoke for the CIFAR-100 dataset in both label variants.

A tiny synthetic bundle with the variant's label range replaces the dataset,
so the test checks the Hydra composition of the CIFAR-100 experiments (with
and without the ``dataset=cifar100_coarse`` override), the class count of the
CNN the runner builds, the attack's ASR loader on a 100- and 20-class label
space, the defense and baseline paths, and the ledger output; it says nothing
about accuracy.
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

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
EXPERIMENTS = (
    "benign_cifar100",
    "backdoor_cifar100_f10",
    "backdoor_cifar100_f20",
    "adaptive_backdoor_cifar100_f10",
    "adaptive_backdoor_cifar100_f20",
)
SMALL = [
    "federation.num_clients=10",
    "federation.clients_per_round=4",
    "federation.num_rounds=2",
    "federation.local_epochs=1",
    "federation.local_batch_size=8",
]


def _tiny_bundle(num_classes: int) -> CifarBundle:
    g = torch.Generator().manual_seed(0)
    return CifarBundle(
        train_x=torch.randn(80, 3, 32, 32, generator=g),
        train_y=torch.randint(0, num_classes, (80,), generator=g),
        val_x=torch.randn(40, 3, 32, 32, generator=g),
        val_y=torch.arange(40) % num_classes,
        test_x=torch.randn(120, 3, 32, 32, generator=g),
        test_y=torch.arange(120) % num_classes,
    )


def _fake_loader(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stub ``load_cifar100`` and record the keyword arguments it received."""
    calls: list[dict] = []

    def fake(*args, **kwargs):  # noqa: ANN001, ANN202
        calls.append(kwargs)
        return _tiny_bundle(20 if kwargs["coarse"] else 100)

    monkeypatch.setattr("experiments.run.load_cifar100", fake)
    return calls


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides)


def _fed(cfg) -> FederationConfig:  # noqa: ANN001
    return FederationConfig(**{k: v for k, v in OmegaConf.to_container(cfg.federation).items()
                               if k != "_target_"})


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_takes_the_coarse_override(monkeypatch, experiment: str) -> None:
    calls = _fake_loader(monkeypatch)
    fine = _compose([f"+experiment={experiment}", *SMALL])
    coarse = _compose([f"+experiment={experiment}", "dataset=cifar100_coarse", *SMALL])
    assert (fine.dataset.name, fine.dataset.num_classes, fine.dataset.val_per_class,
            fine.dataset.coarse) == ("cifar100", 100, 10, False)
    assert (coarse.dataset.name, coarse.dataset.num_classes, coarse.dataset.val_per_class,
            coarse.dataset.coarse) == ("cifar100_coarse", 20, 50, True)
    assert fine.experiment == coarse.experiment == experiment
    if fine.attack.poisoner is not None:
        # The backdoor target stays a valid label in both variants.
        assert 0 <= fine.attack.poisoner.target_class < 20

    for cfg, width in ((fine, 100), (coarse, 20)):
        model = _load_dataset(cfg, _fed(cfg)).model_factory()
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, width)
    assert [c["coarse"] for c in calls] == [False, True]
    assert [c["val_per_class"] for c in calls] == [10, 50]
    assert all(c["seed"] == 0 for c in calls)


def test_mismatched_head_is_refused(monkeypatch) -> None:
    _fake_loader(monkeypatch)
    cfg = _compose(["+experiment=benign_cifar100", "dataset.num_classes=20", *SMALL])
    with pytest.raises(SystemExit, match="num_classes=20"):
        _load_dataset(cfg, _fed(cfg))


def test_resnet_is_not_offered_for_cifar100(monkeypatch) -> None:
    _fake_loader(monkeypatch)
    cfg = _compose(["+experiment=benign_cifar100", "+model=resnet18_gn", *SMALL])
    with pytest.raises(SystemExit, match="cifar10 only"):
        _load_dataset(cfg, _fed(cfg))


@pytest.mark.parametrize(
    ("dataset", "overrides", "suffix", "ledger_rows"),
    [
        ("cifar100", [], "", 0),
        ("cifar100_coarse", [], "", 0),
        ("cifar100", ["+defense=topsis_capped_weighted_median"],
         "__CappedScoreWeight_critic_wmedian", 2 * 4),
        ("cifar100_coarse", ["+baseline=flame"], "__flame", 2 * 4),
        ("cifar100", ["+baseline=fltrust_median"], "__fltrust_median", 0),
        ("cifar100_coarse", ["+baseline=clipped_mean"], "__clipped_mean", 2 * 4),
    ],
)
def test_runner_two_rounds(monkeypatch, tmp_path, dataset, overrides, suffix, ledger_rows) -> None:
    _fake_loader(monkeypatch)
    cfg = _compose([
        "+experiment=backdoor_cifar100_f20",
        f"dataset={dataset}",
        "+tag=c100test",
        *overrides,
        *SMALL,
        f"data_root={tmp_path / 'data'}",
        f"results_root={tmp_path / 'results'}",
    ])
    out_dir = run_from_cfg(cfg)
    assert out_dir.parent.parent.name == f"c100test{suffix}"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["dataset"]["name"] == dataset
    assert manifest["config"]["dataset"]["coarse"] is (dataset == "cifar100_coarse")
    assert manifest["num_malicious"] == 2
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 2
    assert all(0.0 <= m["test_accuracy"] <= 1.0 for m in metrics)
    assert all(0.0 <= m["asr_accuracy"] <= 1.0 for m in metrics)
    ledger = tmp_path / "results" / f"c100test{suffix}" / out_dir.parent.name / out_dir.name / "defense_ledger.json"
    if ledger_rows:
        assert len(json.loads(ledger.read_text())) == ledger_rows
    else:
        assert not ledger.exists()


def test_fine_and_coarse_cells_hash_apart(monkeypatch, tmp_path) -> None:
    from topfed.runtime.paths import config_hash

    def h(cfg) -> str:  # noqa: ANN001
        c = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(c, dict)
        c["federation"].pop("seed", None)
        c.pop("skip_existing", None)
        c.pop("reproducibility", None)
        return config_hash(OmegaConf.to_yaml(c))

    fine = _compose(["+experiment=backdoor_cifar100_f20"])
    coarse = _compose(["+experiment=backdoor_cifar100_f20", "dataset=cifar100_coarse"])
    cifar10 = _compose(["+experiment=backdoor_cifar10_f20"])
    assert len({h(fine), h(coarse), h(cifar10)}) == 3
    assert h(cifar10) == "e32dce1ee2a661fa"
