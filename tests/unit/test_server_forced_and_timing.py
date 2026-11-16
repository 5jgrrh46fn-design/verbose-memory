"""Server loop: forced co-selection and per-round timing (no torch)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from topfed.core.types import (
    ClientId,
    ClientUpdate,
    FederationConfig,
    NDArrays,
    RoundContext,
    RoundOutcome,
)
from topfed.federation.server import MEMORY_KEYS, TIMING_KEYS, run_federation
from topfed.integration.mechanisms import AdaptiveParticipation, ScoreWeight
from topfed.integration.oracle import OracleFilter
from topfed.runtime.manifest import write_metrics


class _Client:
    def __init__(self, cid: ClientId, value: float) -> None:
        self.client_id = cid
        self._value = value

    def local_train(self, ctx: RoundContext, config: FederationConfig) -> ClientUpdate:
        return ClientUpdate(
            client_id=self.client_id,
            parameters=[np.full((2,), self._value)],
            num_examples=1,
            train_loss=0.0,
            train_accuracy=1.0,
        )

    def evaluate(self, parameters: NDArrays, config: FederationConfig):
        return 0.0, 0, {}


class _Scorer:
    def score_round(self, round_num, updates, global_parameters):
        ids = tuple(u.client_id for u in updates)
        return ids, np.full(len(ids), 0.5)


def _config(**kw: Any) -> FederationConfig:
    base: dict[str, Any] = dict(
        num_clients=20, clients_per_round=6, num_rounds=8, local_epochs=1,
        local_batch_size=1, local_learning_rate=0.1, local_momentum=0.0, seed=0,
    )
    base.update(kw)
    return FederationConfig(**base)


MALICIOUS = frozenset({"c1", "c5", "c9", "c13", "c17"})


def _factory(cid: ClientId) -> _Client:
    return _Client(cid, 10.0 if cid in MALICIOUS else 1.0)


def test_forced_co_selection_fixes_malicious_count_every_round() -> None:
    outcomes = run_federation(
        _config(force_malicious_per_round=3), _factory, [np.zeros(2)],
        malicious_clients=MALICIOUS, verbose=False,
    )
    for o in outcomes:
        ids = [u.client_id for u in o.client_updates]
        assert len(ids) == 6
        assert sum(c in MALICIOUS for c in ids) == 3


def test_forced_co_selection_requires_the_malicious_set() -> None:
    with pytest.raises(ValueError):
        run_federation(
            _config(force_malicious_per_round=2), _factory, [np.zeros(2)], verbose=False
        )


def test_forced_co_selection_rejects_sampling_bias_integrators() -> None:
    with pytest.raises(ValueError):
        run_federation(
            _config(force_malicious_per_round=2, num_rounds=3), _factory, [np.zeros(2)],
            scorer=_Scorer(), integrator=AdaptiveParticipation(),
            malicious_clients=MALICIOUS, verbose=False,
        )


def test_unforced_sampling_is_unchanged_by_the_new_argument() -> None:
    a = run_federation(_config(), _factory, [np.zeros(2)], verbose=False)
    b = run_federation(
        _config(), _factory, [np.zeros(2)], malicious_clients=MALICIOUS, verbose=False
    )
    for oa, ob in zip(a, b, strict=True):
        assert [u.client_id for u in oa.client_updates] == [
            u.client_id for u in ob.client_updates
        ]


def test_oracle_on_weighted_median_base_excludes_forced_attackers() -> None:
    # 3 attackers (value 10) forced into every 6-client round; the oracle
    # zero-weights them so the weighted median sees only the honest value.
    outcomes = run_federation(
        _config(force_malicious_per_round=3, num_rounds=2), _factory, [np.zeros(2)],
        scorer=_Scorer(), integrator=OracleFilter(MALICIOUS),
        defended_base="weighted_median", malicious_clients=MALICIOUS, verbose=False,
    )
    assert outcomes[-1].aggregated_parameters[0] == pytest.approx(np.full(2, 1.0))


def test_round_outcomes_carry_timing_breakdown() -> None:
    outcomes = run_federation(
        _config(num_rounds=2), _factory, [np.zeros(2)],
        scorer=_Scorer(), integrator=ScoreWeight(), verbose=False,
    )
    for o in outcomes:
        assert set(o.timings) == set(TIMING_KEYS)
        assert all(v >= 0.0 for v in o.timings.values())
        assert o.timings["round_s"] >= o.timings["client_train_s"]
        assert o.timings["server_train_s"] == 0.0  # no FLTrust here


def test_fltrust_path_times_server_training() -> None:
    def server_update_fn(ctx, cfg):
        return [np.full(2, 1.0)]

    outcomes = run_federation(
        _config(num_rounds=1), _factory, [np.zeros(2)],
        server_update_fn=server_update_fn, verbose=False,
    )
    assert outcomes[0].timings["server_train_s"] >= 0.0
    assert outcomes[0].timings["scoring_s"] == 0.0


def test_write_metrics_merges_timings_into_rows(tmp_path) -> None:
    outcome = RoundOutcome(
        round_num=1, aggregated_parameters=[np.zeros(1)],
        metrics={"val_accuracy": 0.5}, client_updates=(),
        timings={"client_train_s": 1.5, "round_s": 2.0},
    )
    path = write_metrics(tmp_path, [outcome])
    rows = json.loads(path.read_text())
    assert rows == [{"round": 1, "val_accuracy": 0.5, "client_train_s": 1.5, "round_s": 2.0}]


def test_memory_probe_off_by_default() -> None:
    outcomes = run_federation(_config(num_rounds=2), _factory, [np.zeros(2)], verbose=False)
    assert all(o.memory == {} for o in outcomes)


def test_memory_probe_records_server_side_host_peak() -> None:
    import tracemalloc

    big = 1 << 18  # 2 MiB of float64 per client update

    class _BigClient(_Client):
        def local_train(self, ctx: RoundContext, config: FederationConfig) -> ClientUpdate:
            u = super().local_train(ctx, config)
            return ClientUpdate(u.client_id, [np.full(big, self._value)], 1, 0.0, 1.0)

    def factory(cid: ClientId) -> _BigClient:
        return _BigClient(cid, 1.0)

    def stack_mean(updates: list[ClientUpdate]) -> NDArrays:
        return [np.stack([u.parameters[0] for u in updates]).mean(axis=0)]

    outcomes = run_federation(
        _config(num_rounds=2, profile_memory=True), factory, [np.zeros(big)],
        aggregator=stack_mean, verbose=False,
    )
    for o in outcomes:
        assert set(o.memory) <= set(MEMORY_KEYS)
        # The K x d stack (6 x 2 MiB) is allocated inside the probe.
        assert o.memory["server_peak_host_mb"] >= 12.0
    assert not tracemalloc.is_tracing()

    # A caller that is already tracing keeps its trace.
    tracemalloc.start()
    try:
        run_federation(
            _config(num_rounds=1, profile_memory=True), factory, [np.zeros(big)],
            aggregator=stack_mean, verbose=False,
        )
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_write_metrics_merges_memory_into_rows(tmp_path) -> None:
    outcome = RoundOutcome(
        round_num=1, aggregated_parameters=[np.zeros(1)], metrics={}, client_updates=(),
        timings={"round_s": 2.0}, memory={"server_peak_host_mb": 3.5},
    )
    rows = json.loads(write_metrics(tmp_path, [outcome]).read_text())
    assert rows == [{"round": 1, "round_s": 2.0, "server_peak_host_mb": 3.5}]
