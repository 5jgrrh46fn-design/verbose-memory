"""Smoke test: a single malicious client implants a pixel-pattern backdoor.

Three benign clients and one malicious client (c0) train for a few rounds
on small synthetic CIFAR-shaped data. The malicious client uses the full
attack stack — trigger poisoner, constrain-and-scale loss augmenter, and
model-replacement manipulator. After training the attack-success-rate
(ASR) on a triggered evaluation set should be substantially above the
1/num_classes random baseline.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from topfed.attacks.data.backdoor import BackdoorPoisoner, apply_trigger
from topfed.attacks.model.constrain import ConstrainAndScaleAugmenter
from topfed.attacks.model.scale import ModelReplacementManipulator
from topfed.attacks.schedule import AlwaysAttack
from topfed.core.seeding import set_all_seeds
from topfed.core.types import FederationConfig
from topfed.federation.client import (
    BenignTorchClient,
    TorchClient,
    get_parameters,
)
from topfed.federation.evaluation import TorchEvaluator
from topfed.federation.server import run_federation
from topfed.models.cifar_cnn import CifarCNN

NUM_CLASSES = 10
TARGET_CLASS = 9
SAMPLES_PER_CLIENT = 16


def _make_loader(seed: int, batch_size: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(SAMPLES_PER_CLIENT, 3, 32, 32, generator=g)
    y = torch.randint(0, NUM_CLASSES, (SAMPLES_PER_CLIENT,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_backdoor_implants_trigger_above_random() -> None:
    set_all_seeds(0)
    config = FederationConfig(
        num_clients=4,
        clients_per_round=4,
        num_rounds=5,
        local_epochs=3,
        local_batch_size=8,
        local_learning_rate=0.01,
        local_momentum=0.9,
        seed=0,
    )
    malicious_ids = {"c0"}

    loaders = {
        f"c{i}": _make_loader(seed=100 + i, batch_size=config.local_batch_size)
        for i in range(config.num_clients)
    }

    poisoner = BackdoorPoisoner(
        target_class=TARGET_CLASS,
        trigger_size=4,
        trigger_amplitude=1.0,
        poisoning_rate=0.5,
    )
    augmenter = ConstrainAndScaleAugmenter(alpha=0.5)
    manipulator = ModelReplacementManipulator(gamma=10.0)
    schedule = AlwaysAttack()

    def client_factory(cid: str):
        if cid in malicious_ids:
            return TorchClient(
                client_id=cid,
                model_factory=CifarCNN,
                train_loader=loaders[cid],
                poisoner=poisoner,
                loss_augmenter=augmenter,
                update_manipulator=manipulator,
                schedule=schedule,
            )
        return BenignTorchClient(
            client_id=cid,
            model_factory=CifarCNN,
            train_loader=loaders[cid],
        )

    eval_g = torch.Generator().manual_seed(999)
    eval_x = torch.randn(40, 3, 32, 32, generator=eval_g)
    eval_y = torch.randint(0, NUM_CLASSES, (40,), generator=eval_g)
    triggered_x = apply_trigger(eval_x, trigger_size=4, trigger_amplitude=1.0)
    asr_y = torch.full((40,), TARGET_CLASS, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=40)
    asr_loader = DataLoader(TensorDataset(triggered_x, asr_y), batch_size=40)

    evaluator = TorchEvaluator(
        model_factory=CifarCNN,
        loaders={"val": val_loader, "asr": asr_loader},
    )

    initial_parameters = get_parameters(CifarCNN())
    outcomes = run_federation(
        config, client_factory, initial_parameters, evaluator
    )

    assert len(outcomes) == config.num_rounds
    final = outcomes[-1].metrics
    assert "val_loss" in final
    assert "val_accuracy" in final
    assert "asr_accuracy" in final

    asr = final["asr_accuracy"]
    # Random baseline is 1/NUM_CLASSES = 0.1; with model-replacement boosting
    # of a 4x4 trigger and a target class, the implant should comfortably
    # clear 0.5 within 5 rounds.
    assert asr > 0.5, f"expected backdoor ASR > 0.5; got {asr:.3f}"
