"""Unit tests for the gradient-inversion probe's score-driven rules and attackers.

Covers the two score-driven means (``scoreweight``, ``hardfilter``) and both
attackers against them (the standard one, which removes its own update
assuming equal shares, and the oracle, which is handed the true weights),
and the attacker that estimates the capped trust weights of the weighted
median for itself.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.gradient_inversion import (
    HARD_FILTER_THRESHOLD,
    TRUST_MEAN_RULES,
    AwareSimulator,
    _aggregates,
    _estimate_topfed_weights,
    _flat,
    _local_steps,
    _peer_target,
    _resolve_protocols,
    invert,
)
from topfed.core.types import ClientUpdate
from topfed.federation.client import get_parameters
from topfed.integration.mechanisms import CappedScoreWeight, HardFilter, ScoreWeight
from topfed.models.cifar_cnn import CifarCNN

K = 6
DIM = 40
ATTACKER = 0
PEERS = list(range(1, K))


def _updates(rng: np.random.Generator) -> tuple[list[ClientUpdate], list[np.ndarray]]:
    """Synthetic single-layer updates around a zero global model."""
    deltas = [rng.normal(size=DIM) for _ in range(K)]
    updates = [ClientUpdate(f"c{i}", [d.copy()], 1, 0.0, 0.0) for i, d in enumerate(deltas)]
    return updates, deltas


def _weighted(deltas: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    return np.sum([wi * d for wi, d in zip(w, deltas, strict=True)], axis=0)


@pytest.mark.parametrize("rule", TRUST_MEAN_RULES)
def test_standard_attacker_removes_own_update_at_an_equal_share(rule: str) -> None:
    updates, deltas = _updates(np.random.default_rng(0))
    true_w = np.array([0.30, 0.05, 0.20, 0.15, 0.10, 0.20])
    agg = _weighted(deltas, true_w)

    target, mask, info = _peer_target(rule, agg, updates, deltas, ATTACKER)

    assert mask is None
    assert info["assumed_weights"] == "equal"
    assert target == pytest.approx((K * agg - deltas[ATTACKER]) / (K - 1))
    # The equal-share guess is wrong here, so part of the attacker's own
    # update stays in the target and it is not the peers' weighted mean.
    peers_only = _weighted([deltas[i] for i in PEERS], true_w[PEERS])
    assert not np.allclose(target, peers_only)


@pytest.mark.parametrize("rule", TRUST_MEAN_RULES)
def test_standard_attacker_is_exact_when_the_weights_are_equal(rule: str) -> None:
    updates, deltas = _updates(np.random.default_rng(1))
    agg = _weighted(deltas, np.ones(K))

    target, _, _ = _peer_target(rule, agg, updates, deltas, ATTACKER)

    assert target == pytest.approx(np.mean([deltas[i] for i in PEERS], axis=0))


def test_oracle_recovers_the_peers_weighted_mean_under_scoreweight() -> None:
    updates, deltas = _updates(np.random.default_rng(2))
    true_w = np.array([0.30, 0.05, 0.20, 0.15, 0.10, 0.20])
    agg = _weighted(deltas, true_w)

    target, mask, info = _peer_target(
        "scoreweight", agg, updates, deltas, ATTACKER, weights=true_w
    )

    assert mask is None
    assert info["assumed_weights"] == "true"
    assert target == pytest.approx(_weighted([deltas[i] for i in PEERS], true_w[PEERS]))


def test_oracle_sees_only_the_kept_peers_under_hardfilter() -> None:
    updates, deltas = _updates(np.random.default_rng(3))
    scores = np.array([0.8, 0.2, 0.7, 0.6, 0.4, 0.9])
    true_w = HardFilter(threshold=HARD_FILTER_THRESHOLD).integrate(
        1, tuple(u.client_id for u in updates), scores
    ).weights
    assert (true_w > 0).tolist() == [True, False, True, True, False, True]
    agg = _weighted(deltas, true_w)

    target, mask, _ = _peer_target("hardfilter", agg, updates, deltas, ATTACKER, weights=true_w)

    assert mask is None
    assert target == pytest.approx(np.mean([deltas[i] for i in (2, 3, 5)], axis=0))


def test_oracle_target_is_skipped_when_every_peer_was_dropped() -> None:
    updates, deltas = _updates(np.random.default_rng(4))
    true_w = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    agg = _weighted(deltas, true_w)

    _, mask, _ = _peer_target("hardfilter", agg, updates, deltas, ATTACKER, weights=true_w)

    assert mask is not None and not mask.any()


def test_weighted_mean_simulator_applies_the_true_weights() -> None:
    rng = np.random.default_rng(5)
    step = 0.01
    grads = [torch.tensor(rng.normal(size=DIM), requires_grad=True) for _ in PEERS]
    own = torch.tensor(rng.normal(size=DIM))
    # Peers first, the attacker (already removed from the target) last.
    w = torch.tensor([0.05, 0.20, 0.0, 0.10, 0.20, 0.0], dtype=torch.float64)
    sim = AwareSimulator("weighted_mean", own, tuple((i, i + 1) for i in range(K - 1)), step, weights=w)

    out = sim.aggregate(list(grads))

    expected = sum(w[i] * (-step * grads[i]) for i in range(K - 1)) / w.sum()
    assert torch.allclose(out, expected)
    out.sum().backward()
    assert grads[2].grad is not None and not grads[2].grad.any()  # zero weight: no signal
    assert grads[0].grad is not None and grads[0].grad.any()


def _tiny_round(seed: int, clients: int = 4) -> tuple[
    CifarCNN, list[ClientUpdate], list[np.ndarray], torch.Tensor, torch.Tensor
]:
    """One real SGD step (lr 0.01) of ``clients`` clients on one image each."""
    torch.manual_seed(seed)
    model = CifarCNN(num_classes=10, input_size=32)
    global_parameters = get_parameters(model)
    flat_global = _flat(global_parameters)
    images = torch.randn(clients, 3, 32, 32)
    labels = torch.arange(clients) % 10
    params = list(model.parameters())
    updates: list[ClientUpdate] = []
    deltas: list[np.ndarray] = []
    for i in range(clients):
        loss = F.cross_entropy(model(images[i : i + 1]), labels[i : i + 1])
        grads = torch.autograd.grad(loss, params)
        local = [g_p - 0.01 * g.numpy() for g_p, g in zip(global_parameters, grads, strict=True)]
        updates.append(ClientUpdate(f"c{i}", local, 1, float(loss.detach()), 0.0))
        deltas.append(_flat(local) - flat_global)
    return model, updates, deltas, images, labels


def test_oracle_simulation_of_the_true_images_reproduces_its_target() -> None:
    model, updates, deltas, images, labels = _tiny_round(0)
    true_w = np.array([0.4, 0.1, 0.0, 0.5])
    agg = _weighted(deltas, true_w)
    target, _, _ = _peer_target("scoreweight", agg, updates, deltas, ATTACKER, weights=true_w)
    peer_w = torch.tensor([0.1, 0.0, 0.5, 0.0], dtype=torch.float32)
    own = torch.from_numpy(deltas[ATTACKER].astype(np.float32))
    sim = AwareSimulator("weighted_mean", own, ((0, 1), (1, 2), (2, 3)), 0.01, weights=peer_w)

    params = list(model.parameters())
    peer_grads = []
    for i in (1, 2, 3):
        loss = F.cross_entropy(model(images[i : i + 1]), labels[i : i + 1])
        grads = torch.autograd.grad(loss, params)
        peer_grads.append(torch.cat([g.reshape(-1) for g in grads]))

    simulated = sim.aggregate(peer_grads).numpy()
    assert simulated == pytest.approx(target, rel=1e-4, abs=1e-7)


@pytest.mark.parametrize("oracle", [False, True])
def test_invert_runs_against_a_trust_weighted_mean(oracle: bool) -> None:
    model, updates, deltas, _, labels = _tiny_round(1)
    true_w = np.array([0.25, 0.4, 0.0, 0.35])
    agg = _weighted(deltas, true_w)
    target, mask, _ = _peer_target(
        "hardfilter", agg, updates, deltas, ATTACKER, weights=true_w if oracle else None
    )
    sim = None
    if oracle:
        own = torch.from_numpy(deltas[ATTACKER].astype(np.float32))
        peer_w = torch.tensor([0.4, 0.0, 0.35, 0.0], dtype=torch.float32)
        sim = AwareSimulator("weighted_mean", own, ((0, 1), (1, 2), (2, 3)), 0.01, weights=peer_w)
    bound = torch.full((1, 3, 1, 1), 3.0)

    res = invert(
        model,
        target,
        mask,
        labels[1:],
        bounds=(-bound, bound),
        iterations=3,
        restarts=1,
        tv_weight=1e-2,
        lr=0.1,
        seed=0,
        device=torch.device("cpu"),
        simulator=sim,
    )

    assert res.images.shape == (3, 3, 32, 32)
    assert np.isfinite(res.rec_loss)
    assert res.masked_fraction == 0.0


def test_aggregates_feed_one_set_of_scores_to_the_three_scored_rules() -> None:
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    factory = partial(CifarCNN, num_classes=10, input_size=32)
    global_parameters = get_parameters(factory())
    clients = 8
    updates = [
        ClientUpdate(
            f"c{i}",
            [g + rng.normal(scale=1e-3 * (1 + i), size=g.shape).astype(g.dtype) for g in global_parameters],
            1,
            0.0,
            0.0,
        )
        for i in range(clients)
    ]
    val = DataLoader(
        TensorDataset(torch.randn(16, 3, 32, 32), torch.arange(16) % 10), batch_size=16
    )

    aggs, info = _aggregates(
        updates, global_parameters, round_num=1, val_loader=val, model_factory=factory, device="cpu"
    )

    ids = tuple(u.client_id for u in updates)
    scores = np.asarray(info["trust_scores"])
    deltas = [_flat(u.parameters) - _flat(global_parameters) for u in updates]
    assert info["topfed_weights"] == pytest.approx(
        CappedScoreWeight(cap=1.0).integrate(1, ids, scores).weights
    )
    sw = ScoreWeight().integrate(1, ids, scores).weights
    assert info["scoreweight_weights"] == pytest.approx(sw)
    assert aggs["scoreweight"] == pytest.approx(_weighted(deltas, sw), abs=1e-7)
    kept = scores >= HARD_FILTER_THRESHOLD
    assert 0 < kept.sum() < clients  # the filter must bite for the check to mean anything
    assert (np.asarray(info["hardfilter_weights"]) > 0).tolist() == kept.tolist()
    assert aggs["hardfilter"] == pytest.approx(
        np.mean([d for d, k in zip(deltas, kept, strict=True) if k], axis=0), abs=1e-7
    )


def test_equal_weights_make_the_weighted_median_the_lower_median() -> None:
    rng = np.random.default_rng(6)
    grads = [torch.tensor(rng.normal(size=DIM)) for _ in range(9)]
    own = torch.tensor(rng.normal(size=DIM))
    groups = tuple((i, i + 1) for i in range(9))
    equal = AwareSimulator("weighted_median", own, groups, 0.01, weights=torch.ones(10))
    lower = AwareSimulator("lower_median", own, groups, 0.01)

    assert torch.equal(equal.aggregate(grads), lower.aggregate(grads))


def test_estimated_weights_match_the_rule_given_the_true_peer_updates() -> None:
    # Fed the peers' true deltas and the scorer's own validation data, the
    # attacker's estimate is the rule's weights, peers first, attacker last.
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    factory = partial(CifarCNN, num_classes=10, input_size=32)
    global_parameters = get_parameters(factory())
    clients = 6
    updates = [
        ClientUpdate(
            f"c{i}",
            [g + rng.normal(scale=1e-3 * (1 + i), size=g.shape).astype(g.dtype) for g in global_parameters],
            1,
            0.0,
            0.0,
        )
        for i in range(clients)
    ]
    val = DataLoader(
        TensorDataset(torch.randn(16, 3, 32, 32), torch.arange(16) % 10), batch_size=16
    )
    scoring = {"round_num": 1, "val_loader": val, "model_factory": factory, "device": "cpu"}
    _, info = _aggregates(updates, global_parameters, **scoring)
    true_w = np.asarray(info["topfed_weights"])
    assert true_w.max() - true_w.min() > 1e-3  # unequal, so the ordering is tested
    flat_global = _flat(global_parameters)
    peer_deltas = np.stack([_flat(u.parameters) - flat_global for u in updates[1:]])

    estimate = _estimate_topfed_weights(
        peer_deltas, updates[ATTACKER], global_parameters, 1, **scoring
    )

    assert estimate == pytest.approx(np.concatenate([true_w[1:], true_w[:1]]), abs=1e-5)


def test_invert_reestimates_the_weights_on_schedule() -> None:
    model, updates, deltas, _, labels = _tiny_round(4)
    agg = _weighted(deltas, np.ones(4))  # any target of the right size
    own = torch.from_numpy(deltas[ATTACKER].astype(np.float32))
    sim = AwareSimulator(
        "weighted_median", own, ((0, 1), (1, 2), (2, 3)), 0.01, weights=torch.ones(4)
    )
    seen: list[tuple[int, ...]] = []

    def reweight(peer_deltas: np.ndarray) -> np.ndarray:
        seen.append(peer_deltas.shape)
        return np.array([0.1, 0.2, 0.3, 0.4]) * len(seen)

    bound = torch.full((1, 3, 1, 1), 3.0)
    res = invert(
        model,
        agg,
        None,
        labels[1:],
        bounds=(-bound, bound),
        iterations=5,
        restarts=1,
        tv_weight=1e-2,
        lr=0.1,
        seed=0,
        device=torch.device("cpu"),
        simulator=sim,
        reweight=reweight,
        reweight_every=2,
    )

    # Iterations 2 and 4; never before the first block of iterations.
    assert seen == [(3, deltas[0].size)] * 2
    assert res.weights == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert sim.weights is not None and torch.equal(sim.weights, torch.ones(4))


def test_ad_hoc_protocol_names_follow_the_named_ones() -> None:
    protocols = _resolve_protocols(["i32_e2_b8", "1step_b4", "i8_e1_b32", "1step_b1"])

    assert [p.name for p in protocols] == ["1step_b1", "1step_b4", "i32_e2_b8", "i8_e1_b32"]
    assert (protocols[2].images, protocols[2].local_epochs, protocols[2].batch_size) == (32, 2, 8)
    assert [_local_steps(p) for p in protocols] == [1, 1, 8, 1]


@pytest.mark.parametrize("name", ["i0_e1_b8", "i8_e1", "2step_b1", ""])
def test_malformed_protocol_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        _resolve_protocols([name])


def test_whole_shard_protocol_has_no_step_count() -> None:
    (paper,) = _resolve_protocols(["paper"])
    assert _local_steps(paper) is None


def test_summary_counts_a_repeated_seed_once() -> None:
    from experiments.gradient_inversion_summary import summarise

    base = {
        "dataset": "cifar10",
        "state": "warm",
        "protocol": "1step_b1",
        "observer": "peer",
        "rule": "mean",
        "psnr_mean": 20.0,
        "rec_loss": 0.1,
    }
    rows = [
        {**base, "seed": 0, "ssim_mean": 0.2},
        {**base, "seed": 0, "ssim_mean": 0.4},  # the same seed, run again
        {**base, "seed": 1, "ssim_mean": 0.5},
    ]

    (line,) = [ln for ln in summarise(rows).splitlines() if ln.startswith("| peer")]

    # Seed 0 enters as 0.3, so the mean is 0.4 over two seeds, not 0.367 over three rows.
    assert "| 0.400 ± 0.141 |" in line
    assert line.endswith("| 2 (3 runs, averaged within seed) |")
