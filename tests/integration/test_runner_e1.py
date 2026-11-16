"""End-to-end runner tests for the E1 collusion cells on a stubbed CIFAR-10.

Covers, through Hydra composition and ``run_from_cfg``: the collusion
experiment configs (forced co-selection at f=0.30, capture trace on), each
collusion strategy, the FoolsGold and traced FLTrust weight ledgers, and the
collusion ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from experiments.run import run_from_cfg
from topfed.data.cifar10 import CifarBundle
from topfed.federation.instrumentation import TRACE_KEYS

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
NUM_CLIENTS = 10


def _cifar_bundle() -> CifarBundle:
    g = torch.Generator().manual_seed(0)
    return CifarBundle(
        train_x=torch.randn(80, 3, 32, 32, generator=g),
        train_y=torch.randint(0, 10, (80,), generator=g),
        val_x=torch.randn(20, 3, 32, 32, generator=g),
        val_y=torch.arange(20) % 10,
        test_x=torch.randn(40, 3, 32, 32, generator=g),
        test_y=torch.arange(40) % 10,
    )


def _run(overrides: list[str], tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    common = [
        f"federation.num_clients={NUM_CLIENTS}",
        "federation.clients_per_round=6",
        "federation.num_rounds=2",
        "federation.local_epochs=1",
        "federation.local_batch_size=8",
        f"data_root={tmp_path / 'data'}",
        f"results_root={tmp_path / 'results'}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides + common)
    return run_from_cfg(cfg)


def _deltas_by_round(out_dir: Path) -> dict[int, dict[str, float]]:
    rows = json.loads((out_dir / "client_metrics.json").read_text())
    out: dict[int, dict[str, float]] = {}
    for r in rows:
        if r["is_malicious"]:
            out.setdefault(r["round"], {})[r["client_id"]] = r["update_norm"]
    return out


@pytest.mark.parametrize(
    "strategy", ["clone", "budget_split", "reference_shift", "coordinate_capture"]
)
def test_collusion_cell_runs_with_forced_m_trace_and_ledger(strategy, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    out_dir = _run(
        [
            "+experiment=collusion_adaptive_backdoor_cifar10_f30",
            f"+tag=e1test_{strategy}_m3",
            "federation.force_malicious_per_round=3",
            f"+attack/collusion={strategy}",
            "+defense=topsis_capped_weighted_median",
        ],
        tmp_path,
        monkeypatch,
    )
    assert out_dir.parts[-3] == f"e1test_{strategy}_m3__CappedScoreWeight_critic_wmedian"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["num_malicious"] == 3  # ceil(10 * 0.3)
    assert manifest["config"]["attack"]["collusion"]["strategy"] == strategy
    assert manifest["config"]["federation"]["trace_capture"] is True

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert all(set(TRACE_KEYS) <= set(row) for row in metrics)

    norms = _deltas_by_round(out_dir)
    assert all(len(v) == 3 for v in norms.values())
    if strategy == "clone":
        for v in norms.values():
            assert np.allclose(list(v.values()), 0.6)
    if strategy == "budget_split":
        for v in norms.values():
            assert np.allclose(list(v.values()), 0.2)

    ledger = json.loads((out_dir / "collusion_ledger.json").read_text())
    assert [r["round"] for r in ledger] == [1, 2]
    assert all(r["strategy"] == strategy and len(r["colluders"]) == 3 for r in ledger)
    assert all(r["leader"] == r["colluders"][0] for r in ledger)


def test_independent_cell_has_no_collusion_ledger(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    out_dir = _run(
        ["+experiment=collusion_backdoor_cifar10_f30", "+tag=e1test_independent_m2",
         "federation.force_malicious_per_round=2", "+baseline=coordinate_median"],
        tmp_path, monkeypatch,
    )
    assert not (out_dir / "collusion_ledger.json").exists()
    assert not (out_dir / "defense_ledger.json").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # A plain median has no per-client weights.
    assert all(row["trace_malicious_share"] is None or np.isnan(row["trace_malicious_share"]) for row in metrics)


def test_foolsgold_baseline_writes_a_weight_ledger(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    out_dir = _run(
        ["+experiment=collusion_backdoor_cifar10_f30", "+tag=e1test_clone_m3",
         "federation.force_malicious_per_round=3", "+attack/collusion=clone",
         "+baseline=foolsgold"],
        tmp_path, monkeypatch,
    )
    assert out_dir.parts[-3] == "e1test_clone_m3__foolsgold"
    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    assert len(ledger) == 2 * 6
    assert {"alpha", "max_similarity", "weight", "is_malicious"} <= set(ledger[0])
    # Clones have identical first-round histories: FoolsGold zeroes them.
    first = [r for r in ledger if r["round"] == 1]
    assert all(r["alpha"] == 0.0 for r in first if r["is_malicious"])
    assert sum(r["weight"] for r in first) == pytest.approx(1.0)


def test_traced_fltrust_writes_a_trust_ledger(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    out_dir = _run(
        ["+experiment=collusion_backdoor_cifar10_f30", "+tag=e1test_clone_m2",
         "federation.force_malicious_per_round=2", "+attack/collusion=clone",
         "+baseline=fltrust"],
        tmp_path, monkeypatch,
    )
    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    assert len(ledger) == 2 * 6
    by_round: dict[int, float] = {}
    for r in ledger:
        by_round[r["round"]] = by_round.get(r["round"], 0.0) + r["weight"]
    assert all(v == pytest.approx(1.0) or v == 0.0 for v in by_round.values())


def test_agreement_defense_logs_the_sixth_criterion(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    out_dir = _run(
        ["+experiment=collusion_backdoor_cifar10_f30", "+tag=e1test_clone_m3",
         "federation.force_malicious_per_round=3", "+attack/collusion=clone",
         "+defense=topsis_agreement_capped_weighted_median"],
        tmp_path, monkeypatch,
    )
    assert out_dir.parts[-3] == "e1test_clone_m3__CappedScoreWeight_critic_agree_wmedian"
    ledger = json.loads((out_dir / "defense_ledger.json").read_text())
    assert all("m6_pairwise_agreement" in r for r in ledger)
    # Clones submit identical deltas, so their histories agree perfectly.
    first = [r for r in ledger if r["round"] == 1 and r["is_malicious"]]
    assert all(r["m6_pairwise_agreement"] == pytest.approx(1.0, abs=1e-5) for r in first)
