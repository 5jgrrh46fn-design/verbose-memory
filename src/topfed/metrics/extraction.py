"""Per-client metric extraction.

For each (round, selected_client) we compute five behavioural metrics
plus a few raw quantities useful for downstream analysis:

- ``m1_cosine_to_mean``: cosine similarity of the client's full-update
  delta against the cohort mean delta.
- ``m2_norm_ratio``: client delta norm divided by the cohort median norm.
- ``m3_temporal_cv``: coefficient of variation of the client's delta
  norms over the rolling window ``[round - kappa, round]``. ``0`` when
  the client has fewer than two recorded participations in the window.
- ``m4_val_loss_delta``: change in validation cross-entropy when the
  client's full update replaces the global parameters (``eta = 1``).
- ``m5_max_layer_deviation``: maximum, over state-dict tensors, of
  ``|delta_i^l - mean_delta^l| / (|mean_delta^l| + eps)``.

The mean delta over selected clients is contaminated by adversarial
contributions; this is accepted at moderate compromise fractions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from topfed.core.types import ClientId, ClientUpdate, NDArrays
from topfed.federation.client import set_parameters

ModelFactory = Callable[[], nn.Module]


class RoundMetricRow(TypedDict):
    round: int
    client_id: str
    is_malicious: bool
    num_examples: int
    train_loss: float
    train_accuracy: float
    update_norm: float
    m1_cosine_to_mean: float
    m2_norm_ratio: float
    m3_temporal_cv: float
    m4_val_loss_delta: float
    m5_max_layer_deviation: float


@dataclass
class _History:
    # (round_num, norm) entries; M3 filters this to a rolling window.
    entries: list[tuple[int, float]]


def _flatten(arrays: NDArrays) -> np.ndarray:
    return np.concatenate([a.ravel() for a in arrays])


def _cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean == 0.0:
        return 0.0
    std = float(arr.std(ddof=0))
    return std / abs(mean)


class MetricsExtractor:
    """Stateful extractor: rolling per-client norm history feeds M3.

    Construct once per run; call ``extract`` for each round in order.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        model_factory: ModelFactory,
        device: torch.device | str = "cpu",
        kappa: int = 10,
        eps: float = 1e-12,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive; got {kappa}")
        self._val_loader = val_loader
        self._model_factory = model_factory
        self._device = torch.device(device)
        self._kappa = kappa
        self._eps = eps
        self._history: dict[ClientId, _History] = {}
        self._scratch_model = model_factory().to(self._device)
        self._criterion = nn.CrossEntropyLoss()

    def _val_loss(self, parameters: NDArrays) -> float:
        set_parameters(self._scratch_model, parameters)
        self._scratch_model.eval()
        total_loss = 0.0
        total_seen = 0
        with torch.no_grad():
            for x, y in self._val_loader:
                x = x.to(self._device)
                y = y.to(self._device)
                logits = self._scratch_model(x)
                batch = y.size(0)
                total_loss += float(self._criterion(logits, y).item()) * batch
                total_seen += batch
        return total_loss / max(total_seen, 1)

    def _record(self, client_id: ClientId, round_num: int, norm: float) -> None:
        h = self._history.setdefault(client_id, _History(entries=[]))
        h.entries.append((round_num, norm))

    def _window_norms(self, client_id: ClientId, round_num: int) -> list[float]:
        lower = round_num - self._kappa
        h = self._history.get(client_id)
        if h is None:
            return []
        return [n for (r, n) in h.entries if lower <= r <= round_num]

    def extract(
        self,
        round_num: int,
        updates: tuple[ClientUpdate, ...] | list[ClientUpdate],
        global_parameters: NDArrays,
        malicious: frozenset[ClientId] | set[ClientId],
    ) -> list[RoundMetricRow]:
        if not updates:
            return []

        deltas: list[NDArrays] = [u.delta(global_parameters) for u in updates]
        flat_deltas = [_flatten(d) for d in deltas]
        norms = [float(np.linalg.norm(fd)) for fd in flat_deltas]
        mean_flat = np.mean(np.stack(flat_deltas, axis=0), axis=0)
        median_norm = float(np.median(np.asarray(norms)))

        # Per-tensor mean delta for M5.
        num_tensors = len(deltas[0])
        mean_per_tensor: list[np.ndarray] = [
            np.mean(
                np.stack([d[t] for d in deltas], axis=0), axis=0
            )
            for t in range(num_tensors)
        ]
        mean_per_tensor_norm = [
            float(np.linalg.norm(m)) for m in mean_per_tensor
        ]

        baseline_loss = self._val_loss(global_parameters)

        rows: list[RoundMetricRow] = []
        for i, update in enumerate(updates):
            cid = update.client_id

            # M1.
            m1 = _cosine(flat_deltas[i], mean_flat, self._eps)
            # M2.
            m2 = norms[i] / median_norm if median_norm > self._eps else 0.0
            # M3 — record current round first so the window includes it.
            self._record(cid, round_num, norms[i])
            window = self._window_norms(cid, round_num)
            m3 = _coefficient_of_variation(window)
            # M4.
            loss = self._val_loss(update.parameters)
            m4 = loss - baseline_loss
            # M5.
            m5 = 0.0
            for t in range(num_tensors):
                diff = deltas[i][t] - mean_per_tensor[t]
                dev = float(np.linalg.norm(diff)) / (
                    mean_per_tensor_norm[t] + self._eps
                )
                if dev > m5:
                    m5 = dev

            rows.append(
                RoundMetricRow(
                    round=round_num,
                    client_id=cid,
                    is_malicious=cid in malicious,
                    num_examples=update.num_examples,
                    train_loss=update.train_loss,
                    train_accuracy=update.train_accuracy,
                    update_norm=norms[i],
                    m1_cosine_to_mean=m1,
                    m2_norm_ratio=m2,
                    m3_temporal_cv=m3,
                    m4_val_loss_delta=m4,
                    m5_max_layer_deviation=m5,
                )
            )
        return rows
