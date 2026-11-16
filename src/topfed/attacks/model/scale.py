"""Model-replacement manipulator: scale the submitted delta by gamma.

Takes ``parameters_after_training`` and rewrites them as
``global + gamma * (local - global)`` before submission. Used to amplify
an attacker's influence on the FedAvg average.

Two scaling modes are supported:

* **Static** (``gamma=<float>``): a fixed multiplier. Useful for tests and
  for ablations that decouple scaling from the per-round adversary count.
* **Dynamic** (``gamma=None``, plus ``clients_per_round`` and
  ``malicious_clients``): each round computes
  ``gamma_t = clients_per_round / max(1, f_t)`` where ``f_t`` is the
  number of malicious clients actually selected that round. With this
  rule the aggregate amplification per round equals ``clients_per_round``
  regardless of selection variance, which is the canonical
  model-replacement budget for FedAvg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topfed.core.types import ClientId, ClientUpdate, NDArrays, RoundContext


@dataclass(frozen=True, slots=True)
class ModelReplacementManipulator:
    gamma: float | None = 10.0
    clients_per_round: int | None = None
    malicious_clients: frozenset[ClientId] | None = None

    def __post_init__(self) -> None:
        if self.gamma is None:
            if self.clients_per_round is None or self.malicious_clients is None:
                raise ValueError(
                    "dynamic gamma (gamma=None) requires both "
                    "clients_per_round and malicious_clients to be set"
                )
            if self.clients_per_round <= 0:
                raise ValueError(
                    f"clients_per_round must be positive; "
                    f"got {self.clients_per_round}"
                )
        else:
            if self.gamma <= 0:
                raise ValueError(f"gamma must be positive; got {self.gamma}")

    def _gamma_for(self, ctx: RoundContext) -> float:
        if self.gamma is not None:
            return self.gamma
        assert self.clients_per_round is not None  # mypy
        assert self.malicious_clients is not None
        f_t = sum(1 for c in ctx.selected_clients if c in self.malicious_clients)
        return self.clients_per_round / max(1, f_t)

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,
    ) -> ClientUpdate:
        gamma = self._gamma_for(ctx)
        boosted = [
            g + gamma * (w - g)
            for w, g in zip(update.parameters, global_parameters, strict=True)
        ]
        return ClientUpdate(
            client_id=update.client_id,
            parameters=[np.asarray(b) for b in boosted],
            num_examples=update.num_examples,
            train_loss=update.train_loss,
            train_accuracy=update.train_accuracy,
        )
