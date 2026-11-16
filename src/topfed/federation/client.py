"""Federated client.

A single ``TorchClient`` runs SGD-with-momentum on a local dataset and
returns a ``ClientUpdate``. Adversarial behavior is opt-in via four hooks:

- ``poisoner``           — transforms each training batch.
- ``loss_augmenter``     — combines the classification loss with an extra
                            term (e.g. anomaly-evasion regularizer).
- ``update_manipulator`` — transforms the finished update before it leaves
                            the client (e.g. delta boosting).
- ``schedule``           — gates whether the hooks fire on a given round.

When all hooks are ``None``, the client is benign. The convenience
constructor ``BenignTorchClient`` exists for that case so callers don't
have to spell out the optional arguments.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from topfed.attacks.protocols import (
    AttackSchedule,
    LossAugmenter,
    Poisoner,
    UpdateManipulator,
)
from topfed.core.types import (
    ClientId,
    ClientUpdate,
    FederationConfig,
    NDArrays,
    RoundContext,
)

ModelFactory = Callable[[], nn.Module]


def get_parameters(model: nn.Module) -> NDArrays:
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"parameter count mismatch: model has {len(keys)} tensors, "
            f"received {len(parameters)}"
        )
    state_dict = {
        k: torch.from_numpy(np.asarray(v))
        for k, v in zip(keys, parameters, strict=True)
    }
    model.load_state_dict(state_dict, strict=True)


class TorchClient:
    """Local-training client with optional adversarial hooks."""

    def __init__(
        self,
        client_id: ClientId,
        model_factory: ModelFactory,
        train_loader: DataLoader,
        device: torch.device | str = "cpu",
        poisoner: Poisoner | None = None,
        loss_augmenter: LossAugmenter | None = None,
        update_manipulator: UpdateManipulator | None = None,
        schedule: AttackSchedule | None = None,
    ) -> None:
        self.client_id = client_id
        self._model_factory = model_factory
        self._train_loader = train_loader
        self._device = torch.device(device)
        self._poisoner = poisoner
        self._loss_augmenter = loss_augmenter
        self._update_manipulator = update_manipulator
        self._schedule = schedule

    def _is_attack_active(self, ctx: RoundContext) -> bool:
        if self._schedule is None:
            return any(
                h is not None
                for h in (
                    self._poisoner,
                    self._loss_augmenter,
                    self._update_manipulator,
                )
            )
        return self._schedule.is_active(ctx)

    def local_train(
        self, ctx: RoundContext, config: FederationConfig
    ) -> ClientUpdate:
        attack_active = self._is_attack_active(ctx)

        if attack_active and self._loss_augmenter is not None:
            set_ref = getattr(self._loss_augmenter, "set_reference", None)
            if callable(set_ref):
                set_ref(ctx.global_parameters)
        if attack_active and self._poisoner is not None:
            set_ref = getattr(self._poisoner, "set_reference", None)
            if callable(set_ref):
                set_ref(ctx.global_parameters)

        # A direction-aligning manipulator needs a benign reference update to
        # point at; produce one by retraining on the local data with the attack
        # hooks disabled — the attacker's own clean update, a white-box proxy
        # for the server's clean-data direction. A manipulator that needs the
        # clean update only in some rounds or for some clients says so through
        # an optional ``wants_reference_delta(client_id, ctx)``.
        if attack_active and self._update_manipulator is not None:
            set_ref_delta = getattr(
                self._update_manipulator, "set_reference_delta", None
            )
            wants = getattr(self._update_manipulator, "wants_reference_delta", None)
            if callable(set_ref_delta) and (
                not callable(wants) or wants(self.client_id, ctx)
            ):
                benign = self._train_from_global(ctx, config, apply_hooks=False)
                reference_delta = [
                    b - g
                    for b, g in zip(
                        benign.parameters, ctx.global_parameters, strict=True
                    )
                ]
                set_ref_delta(reference_delta)

        # A manipulator that estimates benign-update statistics asks, through
        # ``reference_shards``, for that many clean updates trained on disjoint
        # shards of the local data: the spread of updates this client can
        # simulate without seeing anyone else's.
        if attack_active and self._update_manipulator is not None:
            shards = int(getattr(self._update_manipulator, "reference_shards", 0))
            set_ref_deltas = getattr(
                self._update_manipulator, "set_reference_deltas", None
            )
            if shards > 0 and callable(set_ref_deltas):
                set_ref_deltas(self._shard_reference_deltas(ctx, config, shards))

        update = self._train_from_global(
            ctx, config, apply_hooks=attack_active
        )
        if attack_active and self._update_manipulator is not None:
            update = self._update_manipulator.manipulate(
                update, ctx.global_parameters, ctx
            )
        return update

    def _shard_reference_deltas(
        self, ctx: RoundContext, config: FederationConfig, shards: int
    ) -> list[NDArrays]:
        """Clean deltas trained on disjoint random shards of the local data.

        The data is split into ``shards`` near-equal parts by a fresh
        permutation each round; a client with fewer samples than shards gets
        one part per sample. Empty local data yields no deltas.
        """
        dataset = self._train_loader.dataset
        n = len(dataset)  # type: ignore[arg-type]
        if n == 0:
            return []
        order = torch.randperm(n)
        deltas: list[NDArrays] = []
        for part in torch.tensor_split(order, min(shards, n)):
            loader: DataLoader = DataLoader(
                Subset(dataset, part.tolist()),
                batch_size=self._train_loader.batch_size,
                shuffle=True,
                drop_last=False,
            )
            clean = self._train_from_global(ctx, config, apply_hooks=False, loader=loader)
            deltas.append(
                [
                    c - g
                    for c, g in zip(clean.parameters, ctx.global_parameters, strict=True)
                ]
            )
        return deltas

    def _train_from_global(
        self,
        ctx: RoundContext,
        config: FederationConfig,
        apply_hooks: bool,
        loader: DataLoader | None = None,
    ) -> ClientUpdate:
        """Run local SGD from the round's global parameters.

        When ``apply_hooks`` is false the poisoner and loss augmenter are
        skipped, yielding a purely benign update regardless of how the client
        is configured; this backs both the ordinary training pass and the
        clean reference pass a direction-aligning manipulator requires.
        ``loader`` replaces the client's own loader for a pass over part of
        its data.
        """
        train_loader = self._train_loader if loader is None else loader
        model = self._model_factory().to(self._device)
        set_parameters(model, ctx.global_parameters)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.local_learning_rate,
            momentum=config.local_momentum,
        )
        criterion = nn.CrossEntropyLoss()
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for _ in range(config.local_epochs):
            for x, y in train_loader:
                if apply_hooks and self._poisoner is not None:
                    x, y = self._poisoner.poison_batch(x, y)
                x = x.to(self._device)
                y = y.to(self._device)
                optimizer.zero_grad()
                logits = model(x)
                ce = criterion(logits, y)
                if apply_hooks and self._loss_augmenter is not None:
                    loss = self._loss_augmenter.combine(model, ce)
                else:
                    loss = ce
                loss.backward()
                optimizer.step()
                batch = y.size(0)
                # Reported metrics track the classification loss so that
                # benign and adversarial clients are comparable.
                total_loss += ce.item() * batch
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total_seen += batch

        avg_loss = total_loss / max(total_seen, 1)
        avg_acc = total_correct / max(total_seen, 1)
        num_examples = len(train_loader.dataset)  # type: ignore[arg-type]

        return ClientUpdate(
            client_id=self.client_id,
            parameters=get_parameters(model),
            num_examples=num_examples,
            train_loss=avg_loss,
            train_accuracy=avg_acc,
        )

    def evaluate(
        self, parameters: NDArrays, config: FederationConfig  # noqa: ARG002
    ) -> tuple[float, int, dict[str, float]]:
        model = self._model_factory().to(self._device)
        set_parameters(model, parameters)
        criterion = nn.CrossEntropyLoss()
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for x, y in self._train_loader:
                x = x.to(self._device)
                y = y.to(self._device)
                logits = model(x)
                loss = criterion(logits, y)
                batch = y.size(0)
                total_loss += loss.item() * batch
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total_seen += batch

        if total_seen == 0:
            return 0.0, 0, {"accuracy": 0.0}
        return (
            total_loss / total_seen,
            total_seen,
            {"accuracy": total_correct / total_seen},
        )


def BenignTorchClient(  # noqa: N802 — kept callable-as-class for back compat
    client_id: ClientId,
    model_factory: ModelFactory,
    train_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> TorchClient:
    """Convenience constructor: ``TorchClient`` with no adversarial hooks."""
    return TorchClient(
        client_id=client_id,
        model_factory=model_factory,
        train_loader=train_loader,
        device=device,
    )
