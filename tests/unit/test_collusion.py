"""Collusion coordinator and the four collusion strategies (S1-S4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from topfed.attacks.model.adaptive import (
    DirectionAlignedManipulator,
    NormBoundedManipulator,
)
from topfed.attacks.model.collusion import (
    STRATEGIES,
    BudgetSplitManipulator,
    CloneManipulator,
    CollusionCoordinator,
    CollusionPayload,
    CoordinateCaptureManipulator,
    ReferenceShiftManipulator,
    build_collusion_manipulator,
)
from topfed.attacks.model.scale import ModelReplacementManipulator
from topfed.core.types import ClientUpdate, FederationConfig, NDArrays, RoundContext
from topfed.federation.client import TorchClient

K = 10
MALICIOUS = frozenset({"m0", "m1", "m2", "m3", "m4"})


def _flat(arrs: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(a).ravel() for a in arrs])


def _delta(update: ClientUpdate, g: NDArrays) -> np.ndarray:
    return _flat([w - gg for w, gg in zip(update.parameters, g, strict=True)])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _ctx(g: NDArrays, m: int, round_num: int = 1) -> RoundContext:
    # Colluders interleaved with honest clients; cohort order is training order.
    bloc = [f"m{i}" for i in range(m)]
    honest = [f"h{i}" for i in range(K - m)]
    cohort: list[str] = []
    while bloc or honest:
        if honest:
            cohort.append(honest.pop(0))
        if bloc:
            cohort.append(bloc.pop(0))
    return RoundContext(round_num, 200, g, tuple(cohort), 0)


def _globals(seed: int = 0) -> NDArrays:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(6, 5)), rng.normal(size=(8,))]


def _trained(g: NDArrays, cid: str, seed: int, scale: float = 1.0) -> ClientUpdate:
    rng = np.random.default_rng(seed)
    params = [gg + rng.normal(scale=scale, size=gg.shape) for gg in g]
    return ClientUpdate(cid, params, 50 + seed, 0.3, 0.8)


def _clean(g: NDArrays, seed: int) -> NDArrays:
    rng = np.random.default_rng(1000 + seed)
    return [rng.normal(scale=0.5, size=gg.shape) for gg in g]


def _vanilla() -> ModelReplacementManipulator:
    return ModelReplacementManipulator(
        gamma=None, clients_per_round=K, malicious_clients=MALICIOUS
    )


def _run_bloc(manip, g: NDArrays, ctx: RoundContext) -> dict[str, ClientUpdate]:
    """Run every colluder in cohort order, feeding clean deltas on request."""
    out: dict[str, ClientUpdate] = {}
    for cid in ctx.selected_clients:
        if cid not in MALICIOUS:
            continue
        seed = int(cid[1:])
        if manip.wants_reference_delta(cid, ctx):
            manip.set_reference_delta(_clean(g, seed))
        out[cid] = manip.manipulate(_trained(g, cid, seed), g, ctx)
    return out


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def test_coordinator_orders_colluders_by_cohort_and_picks_the_first_as_leader() -> None:
    coord = CollusionCoordinator(MALICIOUS)
    ctx = RoundContext(1, 200, [], ("h0", "m3", "h1", "m0", "m4"), 0)
    assert coord.colluders(ctx) == ("m3", "m0", "m4")
    assert coord.leader(ctx) == "m3"
    assert coord.is_leader("m3", ctx) and not coord.is_leader("m0", ctx)


def test_lone_context_drops_the_other_colluders_only() -> None:
    coord = CollusionCoordinator(MALICIOUS)
    ctx = RoundContext(1, 200, [], ("h0", "m3", "h1", "m0", "m4"), 0)
    assert coord.lone_context("m0", ctx).selected_clients == ("h0", "h1", "m0")


def test_payload_before_the_leader_posts_raises() -> None:
    coord = CollusionCoordinator(MALICIOUS)
    g = _globals()
    ctx = _ctx(g, 3)
    with pytest.raises(RuntimeError, match="has not submitted"):
        coord.payload(ctx)
    manip = CloneManipulator(coord, _vanilla())
    # A non-leader querying first cannot see a payload.
    with pytest.raises(RuntimeError, match="has not submitted"):
        manip.manipulate(_trained(g, "m1", 1), g, ctx)


def test_only_the_leader_may_post() -> None:
    coord = CollusionCoordinator(MALICIOUS)
    g = _globals()
    ctx = _ctx(g, 3)
    payload = CollusionPayload(1, "m1", g, g, g, None, {})
    with pytest.raises(ValueError, match="not the leader"):
        coord.post(payload, ctx, "clone")


def test_payload_from_an_earlier_round_is_not_reused() -> None:
    coord = CollusionCoordinator(MALICIOUS)
    manip = CloneManipulator(coord, _vanilla())
    g = _globals()
    _run_bloc(manip, g, _ctx(g, 3, round_num=1))
    with pytest.raises(RuntimeError):
        manip.manipulate(_trained(g, "m1", 1), g, _ctx(g, 3, round_num=2))
    assert [r["round"] for r in coord.records] == [1]
    assert coord.records[0]["colluders"] == ["m0", "m1", "m2"]


def test_non_colluder_is_rejected() -> None:
    coord = CollusionCoordinator(MALICIOUS)
    manip = CloneManipulator(coord, _vanilla())
    g = _globals()
    with pytest.raises(ValueError, match="not a selected colluder"):
        manip.manipulate(_trained(g, "h0", 0), g, _ctx(g, 3))


def test_base_that_needs_its_own_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="clean reference"):
        CloneManipulator(CollusionCoordinator(MALICIOUS), DirectionAlignedManipulator())


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown collusion strategy"):
        build_collusion_manipulator("mimic", CollusionCoordinator(MALICIOUS), None)


# ---------------------------------------------------------------------------
# m = 1 is the independent attacker for every strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
@pytest.mark.parametrize("base", ["vanilla", "adaptive"])
def test_single_colluder_submits_exactly_the_base_attack(strategy: str, base: str) -> None:
    make = _vanilla if base == "vanilla" else (lambda: NormBoundedManipulator(0.6))
    manip = build_collusion_manipulator(strategy, CollusionCoordinator(MALICIOUS), make())
    g = _globals()
    ctx = _ctx(g, 1)
    assert not manip.wants_reference_delta("m0", ctx)
    update = _trained(g, "m0", 0)
    got = manip.manipulate(update, g, ctx)
    expected = make().manipulate(update, g, ctx)
    for a, b in zip(got.parameters, expected.parameters, strict=True):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# S1 clone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [2, 3, 5])
def test_clone_submits_identical_updates_equal_to_the_leaders_base_attack(m: int) -> None:
    g = _globals()
    ctx = _ctx(g, m)
    manip = CloneManipulator(CollusionCoordinator(MALICIOUS), _vanilla())
    out = _run_bloc(manip, g, ctx)
    deltas = [_delta(u, g) for u in out.values()]
    for d in deltas[1:]:
        np.testing.assert_array_equal(d, deltas[0])
    # Leader's delta scaled by the round-aware factor K/m.
    leader_raw = _delta(_trained(g, "m0", 0), g)
    np.testing.assert_allclose(deltas[0], (K / m) * leader_raw)
    # Identity fields stay the submitting client's own.
    assert {u.client_id for u in out.values()} == set(out)
    assert out["m1"].num_examples == 51
    assert not any(manip.wants_reference_delta(c, ctx) for c in out)


def test_clone_under_a_norm_bound_keeps_the_budget() -> None:
    g = _globals()
    out = _run_bloc(CloneManipulator(CollusionCoordinator(MALICIOUS), NormBoundedManipulator(0.6)), g, _ctx(g, 4))
    for u in out.values():
        assert float(np.linalg.norm(_delta(u, g))) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# S2 budget split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [2, 4, 5])
def test_budget_split_bloc_total_equals_one_lone_attacker(m: int) -> None:
    g = _globals()
    ctx = _ctx(g, m)
    out = _run_bloc(BudgetSplitManipulator(CollusionCoordinator(MALICIOUS), NormBoundedManipulator(0.6)), g, ctx)
    deltas = [_delta(u, g) for u in out.values()]
    for d in deltas:
        assert float(np.linalg.norm(d)) == pytest.approx(0.6 / m)
    assert float(np.linalg.norm(np.sum(deltas, axis=0))) == pytest.approx(0.6)


def test_budget_split_under_round_aware_replacement_coincides_with_clone() -> None:
    g = _globals()
    ctx = _ctx(g, 3)
    split = _run_bloc(BudgetSplitManipulator(CollusionCoordinator(MALICIOUS), _vanilla()), g, ctx)
    clone = _run_bloc(CloneManipulator(CollusionCoordinator(MALICIOUS), _vanilla()), g, ctx)
    for cid in split:
        np.testing.assert_allclose(_delta(split[cid], g), _delta(clone[cid], g))
    # Lone budget is K * delta, so each member carries K/m of it.
    np.testing.assert_allclose(_delta(split["m2"], g), (K / 3) * _delta(_trained(g, "m0", 0), g))


# ---------------------------------------------------------------------------
# S3 reference shift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [2, 3, 5])
def test_reference_shift_carrier_is_a_lone_attacker_and_decoys_follow_it(m: int) -> None:
    g = _globals()
    ctx = _ctx(g, m)
    manip = ReferenceShiftManipulator(CollusionCoordinator(MALICIOUS), _vanilla())
    assert not manip.wants_reference_delta("m0", ctx)
    assert all(manip.wants_reference_delta(f"m{i}", ctx) for i in range(1, m))
    out = _run_bloc(manip, g, ctx)
    carrier = _delta(out["m0"], g)
    # Sole carrier: the full replacement budget K, not K/m.
    np.testing.assert_allclose(carrier, K * _delta(_trained(g, "m0", 0), g))
    for i in range(1, m):
        decoy = _delta(out[f"m{i}"], g)
        assert _cos(decoy, carrier) == pytest.approx(1.0)
        assert float(np.linalg.norm(decoy)) == pytest.approx(
            float(np.linalg.norm(_flat(_clean(g, i))))
        )


def test_reference_shift_decoys_pull_the_cohort_mean_toward_the_carrier() -> None:
    # Honest updates share a common component; the carrier points elsewhere.
    # Decoys along the carrier raise its cosine to the cohort mean and lower
    # the honest clients'.
    rng = np.random.default_rng(7)
    g: NDArrays = [np.zeros(200)]
    common = rng.normal(size=200)
    honest = [common + rng.normal(scale=1.0, size=200) for _ in range(K - 4)]
    carrier_dir = rng.normal(size=200)

    def cohort_cos(mal: list[np.ndarray]) -> tuple[float, float]:
        mean = np.mean(honest + mal, axis=0)
        return _cos(mal[0], mean), float(np.mean([_cos(h, mean) for h in honest]))

    independent = [carrier_dir] + [common + rng.normal(scale=1.0, size=200) for _ in range(3)]
    ctx = RoundContext(1, 200, g, tuple([f"h{i}" for i in range(K - 4)] + ["m0", "m1", "m2", "m3"]), 0)
    manip = ReferenceShiftManipulator(CollusionCoordinator(MALICIOUS), NormBoundedManipulator(float(np.linalg.norm(carrier_dir))))
    colluded = []
    for i, cid in enumerate(["m0", "m1", "m2", "m3"]):
        if manip.wants_reference_delta(cid, ctx):
            manip.set_reference_delta([independent[i]])
        raw = carrier_dir if i == 0 else independent[i]
        colluded.append(_flat(manip.manipulate(ClientUpdate(cid, [raw], 1, 0.0, 1.0), g, ctx).parameters))
    carrier_ind, honest_ind = cohort_cos(independent)
    carrier_col, honest_col = cohort_cos(colluded)
    assert carrier_col > carrier_ind
    assert honest_col < honest_ind


# ---------------------------------------------------------------------------
# S4 coordinate capture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_fraction", [0.05, 0.2])
def test_coordinate_capture_agrees_on_top_backdoor_coordinates_only(top_fraction: float) -> None:
    g = _globals()
    m = 4
    ctx = _ctx(g, m)
    manip = CoordinateCaptureManipulator(CollusionCoordinator(MALICIOUS), _vanilla(), top_fraction=top_fraction)
    assert all(manip.wants_reference_delta(f"m{i}", ctx) for i in range(m))
    out = _run_bloc(manip, g, ctx)

    d = sum(gg.size for gg in g)
    k = math.ceil(top_fraction * d)
    direction = _delta(_trained(g, "m0", 0), g) - _flat(_clean(g, 0))
    expected = set(np.argsort(-np.abs(direction))[:k].tolist())
    captured = np.array(sorted(expected))
    rest = np.setdiff1d(np.arange(d), captured)
    leader_submitted = (K / m) * _delta(_trained(g, "m0", 0), g)

    for i in range(m):
        delta = _delta(out[f"m{i}"], g)
        np.testing.assert_allclose(delta[captured], leader_submitted[captured])
        np.testing.assert_allclose(delta[rest], _flat(_clean(g, i))[rest])
    # Identical on the captured set across the bloc, different elsewhere.
    a, b = _delta(out["m1"], g), _delta(out["m2"], g)
    np.testing.assert_array_equal(a[captured], b[captured])
    assert not np.allclose(a[rest], b[rest])


def test_coordinate_capture_wins_the_median_on_captured_coordinates() -> None:
    # A majority bloc submitting identical values on the captured set makes
    # the per-coordinate median equal that value exactly there.
    g = _globals()
    m = 6
    malicious = frozenset(f"m{i}" for i in range(m))
    manip = CoordinateCaptureManipulator(
        CollusionCoordinator(malicious),
        ModelReplacementManipulator(gamma=None, clients_per_round=K, malicious_clients=malicious),
        top_fraction=0.1,
    )
    ctx = _ctx(g, m)
    out: dict[str, ClientUpdate] = {}
    for cid in ctx.selected_clients:
        if cid in malicious:
            manip.set_reference_delta(_clean(g, int(cid[1:])))
            out[cid] = manip.manipulate(_trained(g, cid, int(cid[1:])), g, ctx)
    honest = [_trained(g, cid, 100 + j, scale=0.5) for j, cid in enumerate(c for c in ctx.selected_clients if c not in malicious)]
    stack = np.stack([_delta(u, g) for u in list(out.values()) + honest])
    median = np.median(stack, axis=0)
    captured = manip.coordinator.payload(ctx).shared["captured"]
    np.testing.assert_allclose(median[captured], _delta(out["m0"], g)[captured])


def test_coordinate_capture_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="top_fraction"):
        CoordinateCaptureManipulator(CollusionCoordinator(MALICIOUS), None, top_fraction=0.0)


# ---------------------------------------------------------------------------
# Client hook: the clean pass runs only when the strategy asks for it
# ---------------------------------------------------------------------------


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _Counting(BudgetSplitManipulator):
    """Budget split that counts clean passes delivered to it."""

    def __init__(self, coordinator: CollusionCoordinator, wants: bool) -> None:
        super().__init__(coordinator, None)
        self._wants = wants
        self.delivered = 0

    def wants_reference_delta(self, client_id, ctx):  # noqa: ANN001
        return self._wants

    def set_reference_delta(self, reference_delta: NDArrays) -> None:
        self.delivered += 1
        super().set_reference_delta(reference_delta)


@pytest.mark.parametrize("wants", [False, True])
def test_client_runs_the_clean_pass_only_when_asked(wants: bool) -> None:
    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,))), batch_size=4)
    coord = CollusionCoordinator(frozenset({"m0"}))
    manip = _Counting(coord, wants)
    client = TorchClient("m0", _Tiny, loader, update_manipulator=manip)
    g = [p.detach().numpy().copy() for p in _Tiny().state_dict().values()]
    ctx = RoundContext(1, 1, g, ("h0", "m0"), 0)
    cfg = FederationConfig(2, 2, 1, 1, 4, 0.1, 0.0)
    client.local_train(ctx, cfg)
    assert manip.delivered == (1 if wants else 0)
