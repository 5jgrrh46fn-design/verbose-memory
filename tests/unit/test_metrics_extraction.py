"""Unit tests for the per-client M1-M5 extractor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from topfed.core.types import ClientUpdate, NDArrays
from topfed.metrics.extraction import MetricsExtractor


class _TinyModel(nn.Module):
    """Two parameter tensors so M5 has multiple layers to choose from."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def _model_factory() -> nn.Module:
    return _TinyModel()


def _val_loader() -> DataLoader:
    # Non-zero inputs so the linear-without-bias model produces non-constant
    # logits across different parameter snapshots.
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=torch.float32
    )
    y = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def _global() -> NDArrays:
    return [np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]


def _update(cid: str, params: NDArrays) -> ClientUpdate:
    return ClientUpdate(
        client_id=cid,
        parameters=[p.astype(np.float32) for p in params],
        num_examples=10,
        train_loss=0.1,
        train_accuracy=0.9,
    )


def _make_extractor(kappa: int = 10) -> MetricsExtractor:
    return MetricsExtractor(
        val_loader=_val_loader(),
        model_factory=_model_factory,
        device="cpu",
        kappa=kappa,
    )


def test_m1_aligned_with_mean_is_high_unaligned_is_low() -> None:
    g = _global()
    benign_a = _update("a", [np.full((2, 2), 1.0), np.zeros((2, 2))])
    benign_b = _update("b", [np.full((2, 2), 1.0), np.zeros((2, 2))])
    inverted = _update("c", [np.full((2, 2), -1.0), np.zeros((2, 2))])

    rows = _make_extractor().extract(
        round_num=1,
        updates=[benign_a, benign_b, inverted],
        global_parameters=g,
        malicious={"c"},
    )

    by_cid = {r["client_id"]: r for r in rows}
    assert by_cid["a"]["m1_cosine_to_mean"] == pytest.approx(1.0, abs=1e-6)
    # Mean delta = (1, 1, -1) / 3 = (1/3, 1/3, -1/3). Inverted has cos = -1
    # against itself; against the mean (which it pulls down) it should be
    # strongly negative.
    assert by_cid["c"]["m1_cosine_to_mean"] < 0


def test_m2_norm_ratio_is_relative_to_median() -> None:
    g = _global()
    # Three updates with flat-delta norms sqrt(4), sqrt(4*9), sqrt(4*25):
    # 2, 6, 10. Median = 6.
    small = _update("a", [np.full((2, 2), 1.0 / np.sqrt(2)), np.zeros((2, 2))])
    mid = _update("b", [np.full((2, 2), 3.0 / np.sqrt(2)), np.zeros((2, 2))])
    big = _update("c", [np.full((2, 2), 5.0 / np.sqrt(2)), np.zeros((2, 2))])
    rows = _make_extractor().extract(
        round_num=1,
        updates=[small, mid, big],
        global_parameters=g,
        malicious=set(),
    )
    by_cid = {r["client_id"]: r for r in rows}
    # Norms are 2, 6, 10; median is 6.
    assert by_cid["a"]["m2_norm_ratio"] == pytest.approx(2.0 / 6.0, rel=1e-5)
    assert by_cid["b"]["m2_norm_ratio"] == pytest.approx(1.0, rel=1e-5)
    assert by_cid["c"]["m2_norm_ratio"] == pytest.approx(10.0 / 6.0, rel=1e-5)


def test_m3_is_zero_until_window_has_two_samples_then_nonzero_if_varied() -> None:
    extractor = _make_extractor(kappa=10)
    g = _global()
    # Round 1: client "a" alone — M3 default 0.
    rows = extractor.extract(
        1, [_update("a", [np.full((2, 2), 1.0), np.zeros((2, 2))])], g, set()
    )
    assert rows[0]["m3_temporal_cv"] == 0.0

    # Round 2: bigger delta → CV > 0.
    rows = extractor.extract(
        2, [_update("a", [np.full((2, 2), 4.0), np.zeros((2, 2))])], g, set()
    )
    assert rows[0]["m3_temporal_cv"] > 0.0

    # Round 100: history out of window → back to default.
    rows = extractor.extract(
        100, [_update("a", [np.full((2, 2), 4.0), np.zeros((2, 2))])], g, set()
    )
    assert rows[0]["m3_temporal_cv"] == 0.0


def test_m4_positive_when_update_increases_val_loss() -> None:
    # Initialize model parameters so the global state has *low* loss on the
    # constant-zero validation set, and a non-trivial update increases loss.
    g: NDArrays = [
        np.array([[5.0, 0.0], [-5.0, 0.0]], dtype=np.float32),  # fc1
        np.array([[5.0, 0.0], [-5.0, 0.0]], dtype=np.float32),  # fc2
    ]
    # Malicious update flips fc2 so class 0 logit is now negative.
    malicious_params: NDArrays = [
        g[0].copy(),
        np.array([[-5.0, 5.0], [5.0, -5.0]], dtype=np.float32),
    ]
    # Benign-ish update: tiny perturbation.
    benign_params: NDArrays = [g[0] + 0.001, g[1] + 0.001]

    rows = _make_extractor().extract(
        1,
        [_update("mal", malicious_params), _update("ben", benign_params)],
        g,
        {"mal"},
    )
    by_cid = {r["client_id"]: r for r in rows}
    assert by_cid["mal"]["m4_val_loss_delta"] > by_cid["ben"]["m4_val_loss_delta"]


def test_m5_detects_layer_concentrated_anomaly() -> None:
    g = _global()
    # Two benign clients with uniform per-layer deltas.
    a = _update("a", [np.full((2, 2), 1.0), np.full((2, 2), 1.0)])
    b = _update("b", [np.full((2, 2), 1.0), np.full((2, 2), 1.0)])
    # Outlier: matches a/b on tensor 0 but spikes tensor 1.
    c = _update("c", [np.full((2, 2), 1.0), np.full((2, 2), 100.0)])
    rows = _make_extractor().extract(1, [a, b, c], g, {"c"})
    by_cid = {r["client_id"]: r for r in rows}
    assert by_cid["c"]["m5_max_layer_deviation"] > by_cid["a"]["m5_max_layer_deviation"]
    assert by_cid["c"]["m5_max_layer_deviation"] > 1.0


def test_row_schema_complete_and_typed() -> None:
    g = _global()
    rows = _make_extractor().extract(
        7,
        [_update("a", [np.ones((2, 2)), np.zeros((2, 2))])],
        g,
        {"a"},
    )
    r = rows[0]
    assert r["round"] == 7
    assert r["client_id"] == "a"
    assert r["is_malicious"] is True
    assert r["num_examples"] == 10
    for key in (
        "train_loss",
        "train_accuracy",
        "update_norm",
        "m1_cosine_to_mean",
        "m2_norm_ratio",
        "m3_temporal_cv",
        "m4_val_loss_delta",
        "m5_max_layer_deviation",
    ):
        assert key in r
        assert isinstance(r[key], float)
