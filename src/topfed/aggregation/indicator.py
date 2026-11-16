"""BackdoorIndicator: proactive backdoor detection with OOD data (Li and Dai, USENIX Security 2024).

Provenance. Port of the official implementation,
github.com/ybdai7/Backdoor-indicator-defense (MIT) at commit ``93eb75466e38``
(2025-09-10): ``participants/servers/IndicatorServer.py`` (``pre_process``,
``_global_watermark_injection``, ``_indicator``,
``_global_watermarking_test_sub``, ``aggregation``) and
``dataloader/WMFLDataloader.py`` (``_get_ood_dataloader``). Defaults are the
repository's ``utils/yamls/indicator/params_vanilla_Indicator.yaml``.

The method, per round:

1. **Indicator injection** (before broadcast). The server fine-tunes the
   global model on a fixed set of out-of-distribution images carrying labels
   assigned uniformly over the task's classes (shuffled once), with SGD
   (lr 0.005, momentum 0.9, weight decay 5e-4, MultiStepLR gamma 0.8) for
   ``retrain_epochs`` passes, minimising cross-entropy plus
   ``mu / 2 * || theta - theta_0 ||_2`` (the L2 norm, not its square, as in
   the reference). The resulting model is what clients receive.
2. **Detection** (after upload). Each returned model is evaluated on the same
   OOD set; its per-class accuracy on the assigned labels is computed and the
   maximum over classes is the indicator score. Benign local training erodes
   the indicator on every class, so the score drops; a backdoor task pins
   OOD-like inputs to its target label, so one class stays high. A model with
   score ``>= threshold`` (95, in percent) is flagged.
3. **Aggregation**. The admitted models' deltas from the broadcast
   (indicator-carrying) model are averaged with equal weight and applied with
   server learning rate ``eta``. The reference uses ``eta = 0.5`` for every
   server it runs; here ``eta`` defaults to 1.0 so the baseline shares the
   server step of the other rules in this repository. When nothing is
   admitted the broadcast model is kept.

Deviations and choices:

* The reference saves the BatchNorm running statistics before injection and
  restores them into the broadcast model and into each evaluated client
  model. The models here carry no normalisation buffers (the CNNs have no
  normalisation layers, the ResNet uses GroupNorm), so that step has no
  effect and is omitted; a model with BatchNorm would need it.
* The reference's optional norm clipping (``norm_clip``) is off in its
  default configuration and is not ported.
* The OOD set is sampled once with a seeded generator and kept for the whole
  run (the reference samples once at start-up with Python's global
  ``random``).
* Which images serve as OOD data is decided by the caller (see
  ``topfed.data.ood``): CIFAR-100 for CIFAR-10 and CIFAR-10 for FEMNIST, the
  reference's pairings, with the image preprocessing of the host dataset.
* ``start_round``. The reference never injects into a model training from
  scratch: its configuration resumes a global model after 1,200 clean rounds
  and starts injecting at round 1,100 of that schedule (poisoning starts at
  1,210). Injecting from round 1 into a randomly initialised small model
  erases the main task faster than the clients rebuild it. Rounds before
  ``start_round`` therefore run plain equal-weight FedAvg with no
  injection and no filtering (the reference's behaviour outside its
  watermarking rounds); the default of 1 keeps the from-scratch behaviour.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from topfed.core.types import ClientUpdate, NDArrays
from topfed.federation.client import get_parameters, set_parameters

ModelFactory = Callable[[], nn.Module]

DEFAULT_MILESTONES = (10, 20, 30, 160, 200, 240, 280, 320, 360)


def assign_indicator_labels(n: int, num_classes: int, seed: int) -> np.ndarray:
    """Labels for ``n`` OOD samples: every class equally often, then shuffled.

    Mirrors the reference's ``_get_ood_dataloader``: ``range(num_classes)``
    repeated ``n // num_classes`` times plus the first ``n % num_classes``
    classes, permuted once.
    """
    base = list(range(num_classes)) * (n // num_classes) + list(range(n % num_classes))
    labels = np.array(base, dtype=np.int64)
    np.random.default_rng(seed).shuffle(labels)
    return labels


def _flat_distance(model: nn.Module, reference: list[torch.Tensor]) -> torch.Tensor:
    """L2 norm of ``theta - theta_0`` over the model's trainable parameters."""
    parts = [
        (p - r).reshape(-1)
        for p, r in zip(model.parameters(), reference, strict=True)
    ]
    return torch.norm(torch.cat(parts), p=2)


class BackdoorIndicator:
    """BackdoorIndicator as a stateful aggregation rule with a pre-round step.

    ``prepare_round`` injects the indicator into the global parameters and
    returns the model to broadcast; ``aggregate`` evaluates the returned
    models, drops the flagged ones and averages the rest. ``last_weights``
    and ``records`` follow the other rules in this package; a record holds
    the client's indicator score (max per-class accuracy, percent), the
    class that attained it, and whether the client was admitted.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        ood_x: torch.Tensor,
        num_classes: int,
        device: torch.device | str = "cpu",
        *,
        threshold: float = 95.0,
        retrain_epochs: int = 200,
        lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        milestones: tuple[int, ...] = DEFAULT_MILESTONES,
        lr_gamma: float = 0.8,
        mu: float = 0.4,
        batch_size: int = 64,
        eta: float = 1.0,
        start_round: int = 1,
        seed: int = 0,
    ) -> None:
        if ood_x.dim() != 4 or ood_x.shape[0] < batch_size:
            raise ValueError(
                f"ood_x must be (N, C, H, W) with N >= batch_size; got {tuple(ood_x.shape)}"
            )
        if not 0.0 < threshold <= 100.0:
            raise ValueError(f"threshold is a percentage in (0, 100]; got {threshold}")
        if retrain_epochs < 1:
            raise ValueError(f"retrain_epochs must be >= 1; got {retrain_epochs}")
        if start_round < 1:
            raise ValueError(f"start_round must be >= 1; got {start_round}")
        self._factory = model_factory
        self._device = torch.device(device)
        # drop_last, as in the reference: whole batches only.
        n = (ood_x.shape[0] // batch_size) * batch_size
        self.ood_x = ood_x[:n].to(self._device)
        self.ood_y = torch.from_numpy(assign_indicator_labels(n, num_classes, seed)).to(
            self._device
        )
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.retrain_epochs = int(retrain_epochs)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.milestones = tuple(int(m) for m in milestones)
        self.lr_gamma = float(lr_gamma)
        self.mu = float(mu)
        self.batch_size = int(batch_size)
        self.eta = float(eta)
        self.start_round = int(start_round)
        self.last_weights: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []
        self.injection_log: list[dict[str, float]] = []
        self._model = model_factory().to(self._device)

    # ----------------------------------------------------------------- scoring
    @torch.no_grad()
    def indicator_score(self, parameters: NDArrays) -> tuple[float, int, list[float]]:
        """``(max per-class accuracy in percent, its class, all per-class accuracies)``."""
        model = self._model
        set_parameters(model, parameters)
        model.eval()
        correct = torch.zeros(self.num_classes, device=self._device)
        total = torch.zeros(self.num_classes, device=self._device)
        for start in range(0, self.ood_x.shape[0], self.batch_size):
            x = self.ood_x[start : start + self.batch_size]
            y = self.ood_y[start : start + self.batch_size]
            pred = model(x).argmax(dim=1)
            total += torch.bincount(y, minlength=self.num_classes).float()
            correct += torch.bincount(y[pred == y], minlength=self.num_classes).float()
        acc = torch.where(total > 0, 100.0 * correct / total.clamp(min=1), torch.zeros_like(total))
        per_class = [float(a) for a in acc.tolist()]
        best = int(torch.argmax(acc).item())
        return per_class[best], best, per_class

    # ---------------------------------------------------------------- pre-round
    def active(self, round_num: int) -> bool:
        """Whether the indicator is injected and checked in ``round_num``."""
        return round_num >= self.start_round

    def prepare_round(self, round_num: int, global_parameters: NDArrays) -> NDArrays:
        """Inject the indicator task into the global model; return what to broadcast.

        Before ``start_round`` the global model is broadcast unchanged.
        """
        if not self.active(round_num):
            return global_parameters
        model = self._model
        set_parameters(model, global_parameters)
        reference = [p.detach().clone() for p in model.parameters()]
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(self.milestones), gamma=self.lr_gamma
        )
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.retrain_epochs):
            for start in range(0, self.ood_x.shape[0], self.batch_size):
                x = self.ood_x[start : start + self.batch_size]
                y = self.ood_y[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(model(x), y) + (self.mu / 2.0) * _flat_distance(model, reference)
                loss.backward()
                optimizer.step()
            scheduler.step()
        with torch.no_grad():
            norm = float(_flat_distance(model, reference))
        prepared = get_parameters(model)
        score, _, _ = self.indicator_score(prepared)
        self.injection_log.append(
            {"round": float(round_num), "injection_norm": norm, "indicator_score": score}
        )
        return prepared

    # -------------------------------------------------------------- aggregation
    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_parameters: NDArrays,
    ) -> NDArrays:
        if not updates:
            raise ValueError("BackdoorIndicator requires at least one ClientUpdate")
        n = len(updates)
        admitted: list[int] = []
        active = self.active(round_num)
        for i, u in enumerate(updates):
            if active:
                score, cls, _ = self.indicator_score(u.parameters)
                flagged = score >= self.threshold
            else:
                score, cls, flagged = float("nan"), -1, False
            if not flagged:
                admitted.append(i)
            self.records.append(
                {
                    "round": round_num,
                    "client_id": u.client_id,
                    "indicator_score": score,
                    "indicator_class": cls,
                    "admitted": not flagged,
                    "indicator_active": active,
                }
            )
        weights = np.zeros(n, dtype=np.float64)
        if admitted:
            weights[admitted] = 1.0 / len(admitted)
        self.last_weights = weights
        if not admitted:
            return [np.array(g) for g in global_parameters]
        out: NDArrays = []
        for k, g in enumerate(global_parameters):
            g64 = np.asarray(g, dtype=np.float64)
            step = np.zeros_like(g64)
            for i in admitted:
                step += np.asarray(updates[i].parameters[k], dtype=np.float64) - g64
            out.append((g64 + self.eta * step / len(admitted)).astype(np.asarray(g).dtype))
        return out
