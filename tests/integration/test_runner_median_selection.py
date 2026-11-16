"""End-to-end runner test for the ``diagnostics.median_selection`` flag.

A tiny synthetic CIFAR bundle replaces the dataset. The test checks that the
flag writes ``median_selection.json`` beside the defense ledger for the
capped weighted median and the plain coordinate median, that it leaves the
config hash and every number of the run unchanged, and that it is refused on
a mean-based rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from experiments.run import run_from_cfg
from tests.integration.test_runner_e5 import _cifar_bundle

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
FLAG = "+diagnostics.median_selection=true"


TIMING_KEYS = {"client_train_s", "scoring_s", "aggregation_s", "server_train_s", "evaluation_s", "round_s"}


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
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides + common)


@pytest.mark.parametrize(
    "rule", ["+defense=topsis_capped_weighted_median", "+baseline=coordinate_median"]
)
def test_flag_writes_the_diagnostic_and_changes_nothing_else(
    monkeypatch, tmp_path, rule
) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    base = ["+experiment=backdoor_cifar10_f20", "+tag=diagtest", rule]

    def _numbers(out_dir: Path) -> list[dict]:
        rows = json.loads((out_dir / "metrics.json").read_text())
        return [{k: v for k, v in r.items() if k not in TIMING_KEYS} for r in rows]

    plain_dir = run_from_cfg(_compose(base, tmp_path))
    plain = _numbers(plain_dir)
    assert not (plain_dir / "median_selection.json").exists()
    # The flag is excluded from the hash, so the second run lands in the same
    # directory, and the diagnostic only reads, so every number is the same.
    diag_dir = run_from_cfg(_compose([*base, FLAG], tmp_path))
    assert diag_dir == plain_dir
    assert _numbers(diag_dir) == plain
    manifest = json.loads((diag_dir / "manifest.json").read_text())
    assert manifest["config"]["diagnostics"]["median_selection"] is True

    diag = json.loads((diag_dir / "median_selection.json").read_text())
    assert set(diag) == {"clients", "rounds"}
    assert [r["round"] for r in diag["rounds"]] == [1, 2, 3]
    assert len(diag["clients"]) == 3 * 4
    malicious = set(manifest["malicious_ids"])
    for row in diag["rounds"]:
        clients = [c for c in diag["clients"] if c["round"] == row["round"]]
        assert row["num_clients"] == 4
        assert row["num_malicious"] == sum(c["client_id"] in malicious for c in clients)
        assert row["equal_share"] == pytest.approx(row["num_malicious"] / 4)
        assert sum(c["selection_fraction"] for c in clients) == pytest.approx(1.0)
        assert sum(c["weight"] for c in clients) == pytest.approx(1.0)
        assert 0.0 <= row["attacker_selection_fraction"] <= 1.0
        assert row["displacement_norm"] >= 0.0
    if rule.startswith("+baseline"):
        assert all(c["weight"] == pytest.approx(0.25) for c in diag["clients"])


def test_flag_is_refused_on_a_mean_rule(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("experiments.run.load_cifar10", lambda *a, **kw: _cifar_bundle())
    cfg = _compose(
        ["+experiment=backdoor_cifar10_f20", "+defense=topsis_score_weight", FLAG], tmp_path
    )
    with pytest.raises(ValueError, match="median rule"):
        run_from_cfg(cfg)
