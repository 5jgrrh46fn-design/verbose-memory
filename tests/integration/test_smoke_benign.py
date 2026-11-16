"""Smoke test: 3 rounds of benign FedAvg on a tiny synthetic CIFAR-shaped set.

Verifies that the FL loop is wired correctly end-to-end. Two clients each get
10 random RGB-32x32 samples with random labels; we assert that the average
training loss in the final round is meaningfully lower than in the first.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from topfed.core.seeding import set_all_seeds
from topfed.core.types import FederationConfig
from topfed.federation.client import BenignTorchClient, get_parameters
from topfed.federation.server import run_federation
from topfed.models.cifar_cnn import CifarCNN

SAMPLES_PER_CLIENT = 10
NUM_CLASSES = 10


def _make_synthetic_loader(seed: int, batch_size: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(SAMPLES_PER_CLIENT, 3, 32, 32, generator=g)
    y = torch.randint(0, NUM_CLASSES, (SAMPLES_PER_CLIENT,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_benign_fedavg_loss_decreases() -> None:
    set_all_seeds(0)
    config = FederationConfig(
        num_clients=2,
        clients_per_round=2,
        num_rounds=3,
        local_epochs=5,
        local_batch_size=10,
        local_learning_rate=0.01,
        local_momentum=0.9,
        seed=0,
    )

    loaders = {
        f"c{i}": _make_synthetic_loader(seed=100 + i, batch_size=config.local_batch_size)
        for i in range(config.num_clients)
    }

    def client_factory(cid: str) -> BenignTorchClient:
        return BenignTorchClient(
            client_id=cid,
            model_factory=CifarCNN,
            train_loader=loaders[cid],
            device="cpu",
        )

    initial_parameters = get_parameters(CifarCNN())

    outcomes = run_federation(config, client_factory, initial_parameters)

    assert len(outcomes) == config.num_rounds
    for o in outcomes:
        assert len(o.client_updates) == config.clients_per_round
        assert o.aggregated_parameters is not None

    first_loss = sum(u.train_loss for u in outcomes[0].client_updates) / len(
        outcomes[0].client_updates
    )
    last_loss = sum(u.train_loss for u in outcomes[-1].client_updates) / len(
        outcomes[-1].client_updates
    )
    assert last_loss < first_loss - 0.1, (
        f"expected loss to decrease across {config.num_rounds} rounds; "
        f"got first={first_loss:.4f}, last={last_loss:.4f}"
    )
