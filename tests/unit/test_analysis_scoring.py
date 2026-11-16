"""Unit tests for the offline TOPSIS scoring harness."""

from __future__ import annotations

import numpy as np

from topfed.analysis.aggregate import SeedRun
from topfed.analysis.scoring import (
    _best_single,
    _windowed_rows,
    build_scoring_rows,
    topsis_pooled,
)
from topfed.scoring.topsis import TopsisScorer


def _row(round_num: int, cid: str, malicious: bool) -> dict:
    # Benign clients align with the cohort (high m1, low costs); the malicious
    # client is anti-aligned and anomalous on every cost metric.
    if malicious:
        vals = {
            "m1_cosine_to_mean": -0.6,
            "m2_norm_ratio": 6.0,
            "m3_temporal_cv": 0.8,
            "m4_val_loss_delta": 1.5,
            "m5_max_layer_deviation": 7.0,
        }
    else:
        vals = {
            "m1_cosine_to_mean": 0.9,
            "m2_norm_ratio": 1.0,
            "m3_temporal_cv": 0.05,
            "m4_val_loss_delta": -0.01,
            "m5_max_layer_deviation": 0.4,
        }
    return {"round": round_num, "client_id": cid, "is_malicious": malicious, **vals}


def _run(experiment: str, seed: int, n_rounds: int) -> SeedRun:
    rows: list[dict] = []
    for rnd in range(1, n_rounds + 1):
        rows.append(_row(rnd, "c0", malicious=True))
        rows.append(_row(rnd, "c1", malicious=False))
        rows.append(_row(rnd, "c2", malicious=False))
    return SeedRun(
        experiment=experiment,
        seed=seed,
        per_round=[{"round": r} for r in range(1, n_rounds + 1)],
        client_rows=rows,
        config_hash="hash",
    )


def test_windowed_rows_keeps_only_recent_rounds() -> None:
    run = _run("attack", 0, n_rounds=10)
    recent = _windowed_rows(run, window=3)
    kept = {int(r["round"]) for r in recent}
    assert kept == {8, 9, 10}
    # window <= 0 keeps everything.
    assert len(_windowed_rows(run, window=0)) == len(run.client_rows)


def test_topsis_pooled_separates_malicious_from_benign() -> None:
    runs = [_run("attack", s, n_rounds=5) for s in range(3)]
    mal, ben = topsis_pooled(runs, TopsisScorer(), window=0)
    # One malicious + two benign per round, three seeds, five rounds.
    assert len(mal) == 3 * 5
    assert len(ben) == 2 * 3 * 5
    # Malicious clients should score below benign clients on closeness.
    assert max(mal) < min(ben)


def test_best_single_identifies_an_informative_metric() -> None:
    runs = [_run("attack", s, n_rounds=4) for s in range(2)]
    name, auc, f1 = _best_single(runs, window=0)
    assert name in {
        "m1_cosine_to_mean",
        "m2_norm_ratio",
        "m3_temporal_cv",
        "m4_val_loss_delta",
        "m5_max_layer_deviation",
    }
    assert auc > 0.9  # perfectly separable synthetic data


def test_build_scoring_rows_covers_every_strategy() -> None:
    runs = {
        "benign_cifar10": [_run("benign_cifar10", 0, n_rounds=2)],
        "attack": [_run("attack", s, n_rounds=3) for s in range(2)],
    }
    # Mark the benign baseline as all-benign so it is excluded as an attack.
    for run in runs["benign_cifar10"]:
        for row in run.client_rows:
            row["is_malicious"] = False
    rows = build_scoring_rows(runs, "benign_cifar10", window=0)
    assert len(rows) == 1
    row = rows[0]
    assert row.experiment == "attack"
    assert set(row.topsis_auc) == {
        "equal",
        "consistency",
        "magnitude",
        "stability",
        "entropy",
        "critic",
    }
    assert all(0.0 <= v <= 1.0 for v in row.topsis_auc.values())
