"""LIE and the Fang median / trimmed-mean attack from shard statistics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from topfed.attacks.model.statistical import (
    FangTrimmedMeanManipulator,
    LittleIsEnoughManipulator,
)
from topfed.core.types import ClientUpdate, FederationConfig, RoundContext
from topfed.federation.client import TorchClient, get_parameters


def _ctx(round_num: int = 1, seed: int = 0, global_parameters=None) -> RoundContext:  # noqa: ANN001
    return RoundContext(
        round_num=round_num,
        total_rounds=5,
        global_parameters=global_parameters or [],
        selected_clients=("c0",),
        seed=seed,
    )


def _update(parameters: list[np.ndarray], cid: str = "c0") -> ClientUpdate:
    return ClientUpdate(cid, parameters, 10, 0.5, 0.8)


GLOBAL = [np.array([1.0, -1.0, 0.5]), np.array([[2.0]])]
SHARDS = [
    [np.array([0.2, -0.4, 0.0]), np.array([[1.0]])],
    [np.array([0.4, -0.2, 0.0]), np.array([[3.0]])],
    [np.array([0.6, -0.6, 0.0]), np.array([[2.0]])],
]
MEAN = [np.array([0.4, -0.4, 0.0]), np.array([[2.0]])]


def test_lie_submits_mean_minus_z_population_std() -> None:
    m = LittleIsEnoughManipulator(num_std=1.5, reference_shards=3)
    m.set_reference_deltas(SHARDS)
    out = m.manipulate(_update([g + 9.0 for g in GLOBAL]), GLOBAL, _ctx())
    std0 = np.std(np.stack([s[0] for s in SHARDS]), axis=0)  # population (ddof 0)
    std1 = np.std(np.stack([s[1] for s in SHARDS]), axis=0)
    np.testing.assert_allclose(out.parameters[0] - GLOBAL[0], MEAN[0] - 1.5 * std0)
    np.testing.assert_allclose(out.parameters[1] - GLOBAL[1], MEAN[1] - 1.5 * std1)
    # The trained update itself is discarded; its metadata is kept.
    assert out.client_id == "c0" and out.num_examples == 10


def test_lie_stays_within_z_deviations_of_the_estimated_mean() -> None:
    rng = np.random.default_rng(0)
    shards = [[rng.normal(size=50)] for _ in range(4)]
    m = LittleIsEnoughManipulator(num_std=0.5, reference_shards=4)
    m.set_reference_deltas(shards)
    g = [np.zeros(50)]
    out = m.manipulate(_update([np.ones(50)]), g, _ctx())
    stack = np.stack([s[0] for s in shards])
    z = (stack.mean(axis=0) - out.parameters[0]) / stack.std(axis=0)
    np.testing.assert_allclose(z, 0.5)


def test_fang_draws_three_to_four_deviations_against_the_benign_direction() -> None:
    m = FangTrimmedMeanManipulator(reference_shards=3)
    m.set_reference_deltas(SHARDS)
    out = m.manipulate(_update(GLOBAL), GLOBAL, _ctx())
    delta0 = out.parameters[0] - GLOBAL[0]
    delta1 = out.parameters[1] - GLOBAL[1]
    std0 = np.std(np.stack([s[0] for s in SHARDS]), axis=0, ddof=1)  # sample std
    std1 = np.std(np.stack([s[1] for s in SHARDS]), axis=0, ddof=1)
    # Coordinate 0: benign direction positive, so the value lies below the mean.
    assert MEAN[0][0] - 4 * std0[0] <= delta0[0] <= MEAN[0][0] - 3 * std0[0]
    # Coordinate 1: benign direction negative, so the value lies above the mean.
    assert MEAN[0][1] + 3 * std0[1] <= delta0[1] <= MEAN[0][1] + 4 * std0[1]
    # Coordinate 2: no benign movement and no spread: nothing to exploit.
    assert delta0[2] == pytest.approx(0.0)
    assert MEAN[1][0, 0] - 4 * std1[0, 0] <= delta1[0, 0] <= MEAN[1][0, 0] - 3 * std1[0, 0]


def test_fang_opposes_the_benign_direction_coordinate_by_coordinate() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=400)
    shards = [[base + 0.05 * rng.normal(size=400)] for _ in range(4)]
    m = FangTrimmedMeanManipulator(reference_shards=4)
    m.set_reference_deltas(shards)
    out = m.manipulate(_update([np.zeros(400)]), [np.zeros(400)], _ctx())
    mean = np.stack([s[0] for s in shards]).mean(axis=0)
    crafted = out.parameters[0]
    assert np.all((crafted - mean) * np.sign(mean) < 0)


def test_fang_draw_is_reproducible_and_differs_by_round_client_and_seed() -> None:
    def craft(round_num: int = 1, cid: str = "c0", seed: int = 0) -> np.ndarray:
        m = FangTrimmedMeanManipulator(reference_shards=3)
        m.set_reference_deltas(SHARDS)
        return m.manipulate(_update(GLOBAL, cid), GLOBAL, _ctx(round_num, seed)).parameters[0]

    np.testing.assert_array_equal(craft(), craft())
    assert not np.array_equal(craft(), craft(round_num=2))
    assert not np.array_equal(craft(), craft(cid="c7"))
    assert not np.array_equal(craft(), craft(seed=3))


@pytest.mark.parametrize("cls", [LittleIsEnoughManipulator, FangTrimmedMeanManipulator])
def test_passes_through_without_two_shard_updates(cls) -> None:  # noqa: ANN001
    m = cls(reference_shards=4)
    update = _update([g + 1.0 for g in GLOBAL])
    assert m.manipulate(update, GLOBAL, _ctx()) is update
    m.set_reference_deltas(SHARDS[:1])
    assert m.manipulate(update, GLOBAL, _ctx()) is update


@pytest.mark.parametrize("cls", [LittleIsEnoughManipulator, FangTrimmedMeanManipulator])
def test_integer_buffers_are_left_alone(cls) -> None:  # noqa: ANN001
    g = [np.array([0.0, 0.0]), np.array([7], dtype=np.int64)]
    shards = [[np.array([0.1, 0.2]), np.array([0.0])], [np.array([0.3, 0.1]), np.array([0.0])]]
    m = cls(reference_shards=2)
    m.set_reference_deltas(shards)
    out = m.manipulate(_update([np.array([1.0, 1.0]), np.array([9], dtype=np.int64)]), g, _ctx())
    assert out.parameters[1].dtype == np.int64 and out.parameters[1][0] == 9
    assert out.parameters[0].dtype == np.float64


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="reference_shards"):
        LittleIsEnoughManipulator(reference_shards=1)
    with pytest.raises(ValueError, match="num_std"):
        LittleIsEnoughManipulator(num_std=0.0)
    with pytest.raises(ValueError, match="low"):
        FangTrimmedMeanManipulator(low=5.0, high=4.0)


class _Recorder(LittleIsEnoughManipulator):
    def __init__(self, reference_shards: int) -> None:
        super().__init__(reference_shards=reference_shards)
        self.seen: list = []

    def set_reference_deltas(self, reference_deltas) -> None:  # noqa: ANN001
        self.seen.append(reference_deltas)
        super().set_reference_deltas(reference_deltas)


def _client(n: int, manipulator) -> tuple[TorchClient, list[np.ndarray]]:  # noqa: ANN001
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=n > 0)

    def factory() -> nn.Module:
        return nn.Linear(4, 2)

    return (
        TorchClient("c0", factory, loader, update_manipulator=manipulator),
        get_parameters(factory()),
    )


CONFIG = FederationConfig(
    num_clients=1, clients_per_round=1, num_rounds=1, local_epochs=1,
    local_batch_size=4, local_learning_rate=0.1, local_momentum=0.0, seed=0,
)


def test_client_trains_one_clean_update_per_shard() -> None:
    recorder = _Recorder(reference_shards=3)
    client, g = _client(12, recorder)
    out = client.local_train(_ctx(global_parameters=g), CONFIG)
    (deltas,) = recorder.seen
    assert len(deltas) == 3
    assert all(len(d) == len(g) for d in deltas)
    # Shards hold different samples, so their clean updates differ.
    assert not np.allclose(deltas[0][0], deltas[1][0])
    # The submitted update is the crafted one, not the client's own training.
    mean = np.mean([d[0] for d in deltas], axis=0)
    std = np.std([d[0] for d in deltas], axis=0)
    np.testing.assert_allclose(out.parameters[0] - g[0], mean - 1.5 * std, atol=1e-6)
    # The full local dataset is still what the client reports.
    assert out.num_examples == 12


def test_client_with_fewer_samples_than_shards_uses_one_shard_per_sample() -> None:
    recorder = _Recorder(reference_shards=4)
    client, g = _client(2, recorder)
    client.local_train(_ctx(global_parameters=g), CONFIG)
    assert len(recorder.seen[0]) == 2


def test_client_without_data_submits_its_own_update() -> None:
    recorder = _Recorder(reference_shards=4)
    client, g = _client(0, recorder)
    out = client.local_train(_ctx(global_parameters=g), CONFIG)
    assert recorder.seen == [[]]
    np.testing.assert_allclose(out.parameters[0], g[0])


# ---------------------------------------------------------------- backdoors


def test_alie_z_max_matches_the_papers_formula() -> None:
    from topfed.attacks.model.statistical import alie_z_max

    # n = 50, m = 12: s = 26 - 12 = 14, quantile (50 - 12 - 14) / 38.
    assert alie_z_max(50, 12) == pytest.approx(0.3360, abs=1e-3)
    assert alie_z_max(100, 10) == pytest.approx(0.1116, abs=1e-3)
    assert alie_z_max(100, 20) == pytest.approx(0.2858, abs=1e-3)
    # A ten-client cohort with one or two attackers leaves no positive shift.
    assert alie_z_max(10, 1) < 0.0
    assert alie_z_max(10, 2) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="num_attackers"):
        alie_z_max(10, 10)


def test_in_range_backdoor_clips_each_coordinate_into_z_deviations() -> None:
    from topfed.attacks.model.statistical import InRangeBackdoorManipulator

    m = InRangeBackdoorManipulator(num_std=1.0, reference_shards=3)
    m.set_reference_deltas(SHARDS)
    std0 = np.std(np.stack([s[0] for s in SHARDS]), axis=0)
    std1 = np.std(np.stack([s[1] for s in SHARDS]), axis=0)
    # Poisoned delta: far above, inside, and far below the benign range.
    poisoned = [np.array([5.0, -0.45, -3.0]), np.array([[2.1]])]
    out = m.manipulate(_update([g + d for g, d in zip(GLOBAL, poisoned)]), GLOBAL, _ctx())
    delta0 = out.parameters[0] - GLOBAL[0]
    assert delta0[0] == pytest.approx(MEAN[0][0] + std0[0])      # clipped from above
    assert delta0[1] == pytest.approx(-0.45)                      # inside: untouched
    assert delta0[2] == pytest.approx(0.0)                        # zero spread: the mean
    assert (out.parameters[1] - GLOBAL[1])[0, 0] == pytest.approx(2.1)
    assert abs(2.1 - MEAN[1][0, 0]) < std1[0, 0]


def test_in_range_backdoor_with_zero_z_submits_the_benign_mean() -> None:
    from topfed.attacks.model.statistical import InRangeBackdoorManipulator

    m = InRangeBackdoorManipulator(num_std=0.0, reference_shards=3)
    m.set_reference_deltas(SHARDS)
    out = m.manipulate(_update([g + 7.0 for g in GLOBAL]), GLOBAL, _ctx())
    np.testing.assert_allclose(out.parameters[0] - GLOBAL[0], MEAN[0])
    with pytest.raises(ValueError, match="num_std"):
        InRangeBackdoorManipulator(num_std=-0.1)


def _min_max_case(seed: int = 0, k: int = 4, d: int = 30):
    rng = np.random.default_rng(seed)
    benign = [[rng.normal(size=d)] for _ in range(k)]
    g = [np.zeros(d)]
    poisoned = [np.mean([b[0] for b in benign], axis=0) + 10.0 * rng.normal(size=d)]
    return benign, g, poisoned


def test_min_max_sits_exactly_on_the_distance_bound() -> None:
    from topfed.attacks.model.statistical import MinMaxBackdoorManipulator

    benign, g, poisoned = _min_max_case()
    m = MinMaxBackdoorManipulator(reference_shards=4)
    m.set_reference_deltas(benign)
    out = m.manipulate(_update(poisoned), g, _ctx())
    crafted = out.parameters[0]
    b = np.stack([x[0] for x in benign])
    max_pair = max(np.linalg.norm(x - y) for x in b for y in b)
    max_to_benign = max(np.linalg.norm(crafted - x) for x in b)
    assert max_to_benign == pytest.approx(max_pair, rel=1e-9)
    # The crafted update lies on the ray from the benign mean to the poison.
    mu = b.mean(axis=0)
    ray, poison_dir = crafted - mu, poisoned[0] - mu
    cos = ray @ poison_dir / (np.linalg.norm(ray) * np.linalg.norm(poison_dir))
    assert cos == pytest.approx(1.0)
    # A slightly longer step along the same ray would break the bound.
    longer = mu + 1.01 * ray
    assert max(np.linalg.norm(longer - x) for x in b) > max_pair


def test_min_max_does_not_depend_on_how_far_the_poison_is() -> None:
    from topfed.attacks.model.statistical import MinMaxBackdoorManipulator

    benign, g, poisoned = _min_max_case(seed=2)
    mu = np.mean([b[0] for b in benign], axis=0)
    outs = []
    for scale in (0.01, 1.0, 100.0):
        m = MinMaxBackdoorManipulator(reference_shards=4)
        m.set_reference_deltas(benign)
        outs.append(m.manipulate(_update([mu + scale * (poisoned[0] - mu)]), g, _ctx()).parameters[0])
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-9)
    np.testing.assert_allclose(outs[1], outs[2], atol=1e-9)


def test_min_max_passes_through_when_there_is_no_poison_direction() -> None:
    from topfed.attacks.model.statistical import MinMaxBackdoorManipulator

    benign, g, _ = _min_max_case(seed=3)
    mu = np.mean([b[0] for b in benign], axis=0)
    m = MinMaxBackdoorManipulator(reference_shards=4)
    m.set_reference_deltas(benign)
    update = _update([mu])
    assert m.manipulate(update, g, _ctx()) is update
    fresh = MinMaxBackdoorManipulator(reference_shards=4)
    assert fresh.manipulate(update, g, _ctx()) is update
