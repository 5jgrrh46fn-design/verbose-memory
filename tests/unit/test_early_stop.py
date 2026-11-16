"""Early stopping fires after `early_stop_patience` rounds without a new max."""

from __future__ import annotations

import numpy as np

from topfed.core.types import (
    ClientUpdate,
    FederationConfig,
    NDArrays,
    RoundContext,
)
from topfed.federation.server import run_federation


class _ConstantClient:
    """Returns the broadcast parameters unchanged; never improves anything."""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id

    def local_train(
        self, ctx: RoundContext, config: FederationConfig
    ) -> ClientUpdate:
        return ClientUpdate(
            client_id=self.client_id,
            parameters=[p.copy() for p in ctx.global_parameters],
            num_examples=1,
            train_loss=0.0,
            train_accuracy=0.0,
        )


class _MonotonicEvaluator:
    """Emits a val_accuracy that climbs for `peak_round` rounds then plateaus."""

    def __init__(self, peak_round: int) -> None:
        self._peak_round = peak_round
        self._calls = 0

    def evaluate(self, parameters: NDArrays) -> dict[str, float]:  # noqa: ARG002
        self._calls += 1
        # Accuracy = min(call_index, peak_round) / peak_round.
        # After peak_round calls, accuracy is pinned at 1.0 and no longer improves.
        return {
            "val_accuracy": min(self._calls, self._peak_round) / self._peak_round
        }


def _initial_params() -> NDArrays:
    return [np.zeros((2, 2), dtype=np.float32)]


def test_early_stop_triggers_after_plateau() -> None:
    peak = 5
    patience = 3
    config = FederationConfig(
        num_clients=2,
        clients_per_round=2,
        num_rounds=50,
        local_epochs=1,
        local_batch_size=1,
        local_learning_rate=0.01,
        local_momentum=0.0,
        seed=0,
        early_stop_patience=patience,
        early_stop_metric="val_accuracy",
    )
    evaluator = _MonotonicEvaluator(peak_round=peak)

    outcomes = run_federation(
        config,
        client_factory=_ConstantClient,
        initial_parameters=_initial_params(),
        evaluator=evaluator,
        verbose=False,
    )

    # New maximum on rounds 1..peak; first plateau round is peak+1. Patience
    # is hit after `patience` consecutive non-improving rounds, so the loop
    # ends on round peak + patience.
    assert len(outcomes) == peak + patience


def test_early_stop_disabled_runs_full_budget() -> None:
    config = FederationConfig(
        num_clients=2,
        clients_per_round=2,
        num_rounds=7,
        local_epochs=1,
        local_batch_size=1,
        local_learning_rate=0.01,
        local_momentum=0.0,
        seed=0,
        early_stop_patience=0,
    )
    evaluator = _MonotonicEvaluator(peak_round=2)

    outcomes = run_federation(
        config,
        client_factory=_ConstantClient,
        initial_parameters=_initial_params(),
        evaluator=evaluator,
        verbose=False,
    )
    assert len(outcomes) == config.num_rounds
