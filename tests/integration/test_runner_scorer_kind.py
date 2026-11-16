"""End-to-end runner tests for ``defense.scorer.kind`` on a stubbed dataset.

The single-signal scores run through every integration mechanism, the label
names the score, and a config without the key keeps the hash it had before
the key existed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.run import run_from_cfg
from tests.integration.test_runner_e5 import _cifar_bundle
from topfed.runtime.paths import config_hash

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

INTEGRATIONS = {
    "topsis_score_weight": ("ScoreWeight", ""),
    "topsis_temporal_reputation": ("TemporalReputation", ""),
    "topsis_hard_filter": ("HardFilter", ""),
    "topsis_quarantine": ("Quarantine", ""),
    "topsis_adaptive": ("AdaptiveParticipation", ""),
    "topsis_capped_weighted_median": ("CappedScoreWeight", "_wmedian"),
}

# Hashes of cells that existed before ``defense.scorer.kind``: the default
# must leave them where they are.
FROZEN_HASHES = {
    ("adaptive_backdoor_cifar10_f10", "topsis_score_weight"): "ad70142c429ac1af",
    ("adaptive_backdoor_cifar10_f10", "topsis_temporal_reputation"): "be9f1b05092dc5ed",
    ("adaptive_backdoor_cifar10_f10", "topsis_hard_filter"): "02f3488e8608e670",
    ("adaptive_backdoor_cifar10_f10", "topsis_quarantine"): "21d1e0602c5ca3b6",
    ("adaptive_backdoor_cifar10_f10", "topsis_adaptive"): "97d5167ef94bacf0",
    ("adaptive_backdoor_cifar10_f10", "topsis_capped_weighted_median"): "33b6313e3610b4e2",
    ("backdoor_cifar10_f20", "topsis_score_weight"): "6d04d052480aadae",
    ("backdoor_cifar10_f20", "topsis_temporal_reputation"): "da3f0a9d97eb21c9",
    ("backdoor_cifar10_f20", "topsis_hard_filter"): "97eb5060fa1c98b4",
    ("backdoor_cifar10_f20", "topsis_quarantine"): "60048d3e446736dc",
    ("backdoor_cifar10_f20", "topsis_adaptive"): "2aef772df198d721",
    ("backdoor_cifar10_f20", "topsis_capped_weighted_median"): "450859f547a5328a",
}


def _compose_raw(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides)


def _compose(overrides: list[str], tmp_path: Path):
    common = [
        "federation.num_clients=10",
        "federation.clients_per_round=4",
        "federation.num_rounds=3",
        "federation.local_epochs=1",
        "federation.local_batch_size=8",
        f"data_root={tmp_path / 'data'}",
        f"results_root={tmp_path / 'results'}",
    ]
    return _compose_raw(overrides + common)


def _hash(cfg) -> str:  # noqa: ANN001
    c = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(c, dict)
    c["federation"].pop("seed", None)
    c.pop("skip_existing", None)
    c.pop("reproducibility", None)
    c.pop("diagnostics", None)
    return config_hash(OmegaConf.to_yaml(c))


@pytest.mark.parametrize(("cell", "expected"), sorted(FROZEN_HASHES.items()))
def test_default_scorer_kind_keeps_existing_hashes(cell, expected) -> None:  # noqa: ANN001
    experiment, defense = cell
    cfg = _compose_raw([f"+experiment={experiment}", f"+defense={defense}"])
    assert "kind" not in cfg.defense.scorer
    assert _hash(cfg) == expected


def test_scorer_kind_changes_the_hash() -> None:
    base = ["+experiment=adaptive_backdoor_cifar10_f10", "+defense=topsis_hard_filter"]
    hashes = {
        _hash(_compose_raw(base)),
        _hash(_compose_raw([*base, "+defense.scorer.kind=server_cosine"])),
        _hash(_compose_raw([*base, "+defense.scorer.kind=history_similarity"])),
        _hash(_compose_raw([
            *base, "+defense.scorer.kind=server_cosine",
            "+defense.integrator.cohort_median=true",
        ])),
    }
    assert len(hashes) == 4


@pytest.mark.parametrize("kind", ["server_cosine", "history_similarity"])
@pytest.mark.parametrize("defense", sorted(INTEGRATIONS))
def test_single_signal_scores_run_through_every_integration(
    monkeypatch, tmp_path, kind, defense
) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        ["+experiment=adaptive_backdoor_cifar10_f20", "+tag=kindtest",
         f"+defense={defense}", f"+defense.scorer.kind={kind}"],
        tmp_path,
    )
    out_dir = run_from_cfg(cfg)
    integrator, base = INTEGRATIONS[defense]
    assert out_dir.parts[-3] == f"kindtest__{integrator}_{kind}{base}"

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 3 and "asr_accuracy" in metrics[-1]
    # Only the server-anchored score trains the server on its root set.
    assert all((row["server_train_s"] > 0.0) == (kind == "server_cosine") for row in metrics)

    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    assert len(ledger) == 3 * 4
    assert all(0.0 <= row["score"] <= 1.0 for row in ledger)
    assert all(sum(r["weight"] for r in ledger if r["round"] == n) == pytest.approx(1.0)
               for n in (1, 2, 3))
    column = "m6_server_cosine" if kind == "server_cosine" else "max_similarity"
    assert all(column in row and "is_malicious" in row for row in ledger)
    if kind == "server_cosine":
        assert all(row["score"] == pytest.approx(max(row[column], 0.0)) for row in ledger)


def test_cohort_median_threshold_through_hydra(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        ["+experiment=adaptive_backdoor_cifar10_f20", "+tag=kindtest_medthr",
         "+defense=topsis_hard_filter", "+defense.scorer.kind=history_similarity",
         "+defense.integrator.cohort_median=true"],
        tmp_path,
    )
    out_dir = run_from_cfg(cfg)
    assert out_dir.parts[-3] == "kindtest_medthr__HardFilter_history_similarity"
    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    for n in (1, 2, 3):
        rows = [r for r in ledger if r["round"] == n]
        threshold = rows[0]["threshold"]
        assert all(r["included"] == (r["reputation"] >= threshold) for r in rows)


def test_unknown_scorer_kind_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        ["+experiment=adaptive_backdoor_cifar10_f20", "+defense=topsis_hard_filter",
         "+defense.scorer.kind=nonsense"],
        tmp_path,
    )
    with pytest.raises(SystemExit, match="scorer.kind"):
        run_from_cfg(cfg)


def test_single_signal_kind_refuses_a_sixth_criterion(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        ["+experiment=adaptive_backdoor_cifar10_f20",
         "+defense=topsis_rooted_capped_weighted_median",
         "+defense.scorer.kind=history_similarity"],
        tmp_path,
    )
    with pytest.raises(SystemExit, match="single-signal"):
        run_from_cfg(cfg)


@pytest.mark.parametrize(
    ("defense", "label"),
    [
        ("topsis_capped_weighted_geomedian", "CappedScoreWeight_critic_gmedian"),
        ("topsis_weighted_geomedian", "ScoreWeight_critic_gmedian"),
    ],
)
def test_geometric_median_base_through_hydra(monkeypatch, tmp_path, defense, label) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        ["+experiment=adaptive_backdoor_cifar10_f20", "+tag=gmtest", f"+defense={defense}"],
        tmp_path,
    )
    out_dir = run_from_cfg(cfg)
    assert out_dir.parts[-3] == f"gmtest__{label}"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config"]["defense"]["geometric_median"] == {"nu": 1.0e-6, "max_iter": 10}
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics) == 3 and all(0.0 <= m["test_accuracy"] <= 1.0 for m in metrics)
    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    assert len(ledger) == 3 * 4
