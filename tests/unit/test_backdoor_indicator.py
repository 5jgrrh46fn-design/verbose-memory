"""BackdoorIndicator: indicator injection, scoring, pruning, and the pre-round hook."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from topfed.aggregation import DeltaAggregator
from topfed.aggregation.indicator import BackdoorIndicator, assign_indicator_labels
from topfed.core.types import ClientUpdate, FederationConfig, NDArrays, RoundContext
from topfed.federation.client import get_parameters
from topfed.federation.server import run_federation

D, C, N = 6, 4, 64


def _factory() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(D * 1 * 1, 16), nn.ReLU(), nn.Linear(16, C))


def _ood() -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(1).normal(size=(N, D, 1, 1))).float()


def _indicator(**kw) -> BackdoorIndicator:
    defaults = dict(retrain_epochs=40, batch_size=16, lr=0.05, mu=0.0, seed=0)
    defaults.update(kw)
    return BackdoorIndicator(_factory, _ood(), C, "cpu", **defaults)


def _update(cid: str, params: NDArrays) -> ClientUpdate:
    return ClientUpdate(cid, [np.array(p) for p in params], 10, 0.0, 1.0)


def test_labels_cover_classes_uniformly_and_are_seeded() -> None:
    a = assign_indicator_labels(64, C, seed=3)
    b = assign_indicator_labels(64, C, seed=3)
    np.testing.assert_array_equal(a, b)
    assert np.bincount(a, minlength=C).tolist() == [16] * C
    odd = assign_indicator_labels(10, C, seed=0)
    assert sorted(np.bincount(odd, minlength=C).tolist()) == [2, 2, 3, 3]
    assert not np.array_equal(a, assign_indicator_labels(64, C, seed=4))


def test_prepare_round_plants_the_indicator() -> None:
    ind = _indicator()
    g = get_parameters(_factory())
    before, _, _ = ind.indicator_score(g)
    prepared = ind.prepare_round(1, g)
    after, _, per_class = ind.indicator_score(prepared)
    # A 16-unit MLP memorising random labels on 64 points in 40 epochs: the
    # score rises well above chance (25) on every class.
    assert after > before
    assert after >= 60.0
    assert all(p > 25.0 for p in per_class)
    assert any(not np.allclose(a, b) for a, b in zip(prepared, g, strict=True))
    assert ind.injection_log[-1]["injection_norm"] > 0


def test_mu_regularises_the_injection_norm() -> None:
    g = get_parameters(_factory())
    free = _indicator(mu=0.0).prepare_round(1, g)
    tight = _indicator(mu=50.0).prepare_round(1, g)

    def dist(p: NDArrays) -> float:
        return float(np.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(p, g, strict=True))))

    assert dist(tight) < dist(free)


def _constant_class_model(k: int) -> NDArrays:
    params = get_parameters(_factory())
    params[-1] = np.full_like(params[-1], -10.0)
    params[-1][k] = 10.0
    return params


def test_aggregate_flags_models_pinned_to_one_class_and_averages_the_rest() -> None:
    ind = _indicator(eta=1.0)
    g = ind.prepare_round(1, get_parameters(_factory()))
    rng = np.random.default_rng(5)
    # Benign stand-ins: fresh random models forget the indicator on every class.
    benign = []
    for i in range(3):
        torch.manual_seed(100 + i)
        p = [np.array(t.detach().numpy()) for t in _factory().parameters()]
        benign.append(_update(f"b{i}", p))
    poisoned = _update("p0", _constant_class_model(2))
    updates = benign + [poisoned]
    assert isinstance(ind, DeltaAggregator)
    out = ind.aggregate(1, updates, g)

    rows = [r for r in ind.records if r["round"] == 1]
    assert [r["admitted"] for r in rows] == [True, True, True, False]
    assert rows[3]["indicator_score"] == 100.0 and rows[3]["indicator_class"] == 2
    assert ind.last_weights is not None
    np.testing.assert_allclose(ind.last_weights, [1 / 3, 1 / 3, 1 / 3, 0.0])
    for k, gk in enumerate(g):
        expected = gk + np.mean([u.parameters[k] - gk for u in benign], axis=0)
        np.testing.assert_allclose(out[k], expected, atol=1e-6)
    _ = rng


def test_eta_scales_the_server_step_and_nothing_admitted_keeps_global() -> None:
    ind = _indicator(eta=0.5)
    g = ind.prepare_round(1, get_parameters(_factory()))
    torch.manual_seed(7)
    p = [np.array(t.detach().numpy()) for t in _factory().parameters()]
    out = ind.aggregate(1, [_update("b", p)], g)
    for k, gk in enumerate(g):
        np.testing.assert_allclose(out[k], gk + 0.5 * (p[k] - gk), atol=1e-6)
    kept = ind.aggregate(2, [_update("p", _constant_class_model(0))], g)
    for k, gk in enumerate(g):
        np.testing.assert_allclose(kept[k], gk)
    assert ind.last_weights is not None and ind.last_weights.sum() == 0.0


def test_start_round_keeps_fedavg_before_injection() -> None:
    ind = _indicator(start_round=3)
    g = get_parameters(_factory())
    assert ind.prepare_round(1, g) is g
    assert ind.injection_log == []
    # Before start_round nothing is flagged, not even a one-class model.
    p = _constant_class_model(1)
    out = ind.aggregate(2, [_update("p", p)], g)
    for k, gk in enumerate(g):
        np.testing.assert_allclose(out[k], p[k], atol=1e-6)
    row = ind.records[-1]
    assert row["admitted"] is True and row["indicator_active"] is False
    assert row["indicator_class"] == -1 and np.isnan(row["indicator_score"])
    # From start_round the indicator is injected and the same model is flagged.
    prepared = ind.prepare_round(3, g)
    assert ind.injection_log and ind.injection_log[-1]["round"] == 3.0
    ind.aggregate(3, [_update("p", p)], prepared)
    assert ind.records[-1]["admitted"] is False and ind.records[-1]["indicator_active"] is True
    with pytest.raises(ValueError):
        _indicator(start_round=0)


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        BackdoorIndicator(_factory, _ood()[:8], C, batch_size=16)
    with pytest.raises(ValueError):
        _indicator(threshold=0.0)
    with pytest.raises(ValueError):
        _indicator(retrain_epochs=0)
    with pytest.raises(ValueError):
        _indicator().aggregate(1, [], get_parameters(_factory()))


# ------------------------------------------------------------ server-loop hook
class _Prepared:
    """A delta aggregator whose pre-round step shifts the broadcast model by +1."""

    last_weights = None

    def __init__(self) -> None:
        self.seen_bases: list[float] = []

    def prepare_round(self, round_num: int, global_parameters: NDArrays) -> NDArrays:
        return [g + 1.0 for g in global_parameters]

    def aggregate(self, round_num: int, updates: list[ClientUpdate], g: NDArrays) -> NDArrays:
        self.seen_bases.append(float(g[0].mean()))
        return [np.mean([u.parameters[0] for u in updates], axis=0)]


class _EchoClient:
    """Returns the broadcast parameters plus its own constant."""

    def __init__(self, cid: str, value: float) -> None:
        self.client_id = cid
        self.value = value

    def local_train(self, ctx: RoundContext, config: FederationConfig) -> ClientUpdate:
        return ClientUpdate(
            self.client_id, [ctx.global_parameters[0] + self.value], 10, 0.0, 1.0
        )


def test_server_calls_prepare_round_before_broadcast() -> None:
    agg = _Prepared()
    cfg = FederationConfig(
        num_clients=2, clients_per_round=2, num_rounds=2, local_epochs=1,
        local_batch_size=8, local_learning_rate=0.1, local_momentum=0.0, seed=0,
    )
    outcomes = run_federation(
        cfg,
        lambda cid: _EchoClient(cid, 0.5),
        initial_parameters=[np.zeros((2, 2))],
        evaluator=None,
        aggregator=agg,
        verbose=False,
    )
    # Round 1: broadcast 0 + 1 = 1; clients return 1.5; aggregate 1.5.
    # Round 2: broadcast 1.5 + 1 = 2.5; clients return 3.0.
    assert agg.seen_bases == [1.0, 2.5]
    assert outcomes[-1].aggregated_parameters[0] == pytest.approx(np.full((2, 2), 3.0))
    assert outcomes[0].timings["server_train_s"] >= 0.0
