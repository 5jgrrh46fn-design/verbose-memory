"""Server-side evaluator on held-out data.

The evaluator runs the aggregated global model over one or more named
DataLoaders and emits ``{name}_loss`` / ``{name}_accuracy`` per loader.

For a benign run the typical configuration is a single ``"val"`` loader,
producing ``val_loss`` and ``val_accuracy``. For a backdoor experiment the
caller adds an ``"asr"`` loader whose samples have the trigger applied and
labels remapped to the target class — accuracy on that loader is the
attack success rate (emitted as ``asr_accuracy``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from topfed.core.types import NDArrays
from topfed.federation.client import set_parameters

ModelFactory = Callable[[], nn.Module]


class TorchEvaluator:
    def __init__(
        self,
        model_factory: ModelFactory,
        loaders: Mapping[str, DataLoader],
        device: torch.device | str = "cpu",
    ) -> None:
        if not loaders:
            raise ValueError("TorchEvaluator requires at least one loader")
        self._model_factory = model_factory
        self._loaders = dict(loaders)
        self._device = torch.device(device)

    def evaluate(self, parameters: NDArrays) -> dict[str, float]:
        model = self._model_factory().to(self._device)
        set_parameters(model, parameters)
        criterion = nn.CrossEntropyLoss()
        model.eval()

        out: dict[str, float] = {}
        with torch.no_grad():
            for name, loader in self._loaders.items():
                total_loss = 0.0
                total_correct = 0
                total_seen = 0
                for x, y in loader:
                    x = x.to(self._device)
                    y = y.to(self._device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    batch = y.size(0)
                    total_loss += loss.item() * batch
                    total_correct += int((logits.argmax(dim=1) == y).sum().item())
                    total_seen += batch
                if total_seen == 0:
                    out[f"{name}_loss"] = 0.0
                    out[f"{name}_accuracy"] = 0.0
                else:
                    out[f"{name}_loss"] = total_loss / total_seen
                    out[f"{name}_accuracy"] = total_correct / total_seen
        return out
