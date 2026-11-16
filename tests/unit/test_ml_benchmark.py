"""Unit tests for the TOPSIS-vs-ML benchmark harness."""

from __future__ import annotations

import numpy as np

from topfed.analysis.aggregate import SeedRun
from topfed.analysis.ml_benchmark import (
    build_family_rows,
    build_family_sets,
    build_feature_sets,
    build_rows,
    family_of,
    logistic_kfold,
    supervised_loao,
)
from topfed.scoring.ml import fit_logistic


def _row(round_num: int, cid: str, malicious: bool) -> dict:
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


def _run(experiment: str, seed: int, n_rounds: int, malicious: bool = True) -> SeedRun:
    rows: list[dict] = []
    for rnd in range(1, n_rounds + 1):
        rows.append(_row(rnd, "c0", malicious=malicious))
        rows.append(_row(rnd, "c1", malicious=False))
        rows.append(_row(rnd, "c2", malicious=False))
    return SeedRun(
        experiment=experiment,
        seed=seed,
        per_round=[{"round": r} for r in range(1, n_rounds + 1)],
        client_rows=rows,
        config_hash="hash",
    )


def test_build_feature_sets_excludes_baseline_and_keeps_both_classes() -> None:
    runs = {
        "benign_cifar10": [_run("benign_cifar10", 0, 3, malicious=False)],
        "attack_a": [_run("attack_a", s, 5) for s in range(2)],
    }
    sets = build_feature_sets(runs, "benign_cifar10", window=0)
    assert "benign_cifar10" not in sets
    assert "attack_a" in sets
    x, y = sets["attack_a"]
    assert x.shape[1] == 5
    assert y.any() and (~y).any()


def test_logistic_kfold_separates_clean_synthetic() -> None:
    runs = {"attack_a": [_run("attack_a", s, 6) for s in range(2)]}
    x, y = build_feature_sets(runs, "benign_cifar10", window=0)["attack_a"]
    assert logistic_kfold(x, y).auc > 0.95


def test_loao_generalises_across_identical_attacks() -> None:
    sets = build_feature_sets(
        {
            "attack_a": [_run("attack_a", 0, 5)],
            "attack_b": [_run("attack_b", 0, 5)],
        },
        "benign_cifar10",
        window=0,
    )
    summary = supervised_loao(sets, "attack_a", fit_logistic)
    # The two synthetic attacks share a signature, so training on one
    # transfers to the other.
    assert summary.auc > 0.95


def test_family_of_groups_configurations() -> None:
    assert family_of("backdoor_cifar10_f10") == "backdoor"
    assert family_of("ablation_backdoor_aug_t4_f10") == "backdoor"
    assert family_of("backdoor_cifar10_f50_diagnostic") == "backdoor"
    assert family_of("gradient_scaling_cifar10_f20") == "gradient_scaling"
    assert family_of("sign_flip_cifar10_f10") == "sign_flip"
    assert family_of("dirty_label_cifar10_f20") == "dirty_label"
    assert family_of("clean_label_cifar10_f10") == "clean_label"


def test_build_family_sets_merges_configs_of_a_family() -> None:
    runs = {
        "benign_cifar10": [_run("benign_cifar10", 0, 2, malicious=False)],
        "backdoor_cifar10_f10": [_run("backdoor_cifar10_f10", 0, 3)],
        "backdoor_cifar10_f20": [_run("backdoor_cifar10_f20", 0, 3)],
        "sign_flip_cifar10_f10": [_run("sign_flip_cifar10_f10", 0, 3)],
    }
    sets, runs_by_family = build_family_sets(runs, "benign_cifar10", window=0)
    assert set(sets) == {"backdoor", "sign_flip"}
    # Both backdoor configs are merged into one family training set.
    assert len(runs_by_family["backdoor"]) == 2
    rows = build_family_rows(runs, "benign_cifar10", window=0)
    assert {r.family for r in rows} == {"backdoor", "sign_flip"}
    for r in rows:
        assert 0.0 <= r.logreg_lofo <= 1.0


def test_build_rows_shape_and_bounds() -> None:
    runs = {
        "benign_cifar10": [_run("benign_cifar10", 0, 2, malicious=False)],
        "attack_a": [_run("attack_a", s, 4) for s in range(2)],
        "attack_b": [_run("attack_b", s, 4) for s in range(2)],
    }
    rows = build_rows(runs, "benign_cifar10", window=0)
    assert {r.experiment for r in rows} == {"attack_a", "attack_b"}
    for row in rows:
        for value in (
            row.topsis_critic,
            row.logreg_kfold,
            row.logreg_loao,
            row.forest_kfold,
            row.forest_loao,
            row.isolation,
        ):
            assert 0.0 <= value <= 1.0
