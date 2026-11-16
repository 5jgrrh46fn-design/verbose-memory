"""Unit tests for the defended-vs-undefended evaluation."""

from __future__ import annotations

import pytest

from topfed.analysis.aggregate import SeedRun
from topfed.analysis.defense import build_rows, parse_label


def _run(name: str, seed: int, asr: float, acc: float, n: int = 4) -> SeedRun:
    per_round = [
        {"round": r, "asr_accuracy": asr, "test_accuracy": acc}
        for r in range(1, n + 1)
    ]
    return SeedRun(
        experiment=name,
        seed=seed,
        per_round=per_round,
        client_rows=[],
        config_hash="h",
    )


def test_parse_label_splits_defended_runs() -> None:
    assert parse_label("backdoor_cifar10_f10__ScoreWeight_critic") == (
        "backdoor_cifar10_f10",
        "ScoreWeight_critic",
    )
    assert parse_label("backdoor_cifar10_f10") is None


def test_build_rows_pairs_and_measures(tmp_path) -> None:
    runs = {
        "atk": [_run("atk", s, asr=0.80, acc=0.70) for s in range(3)],
        "atk__ScoreWeight_critic": [
            _run("atk__ScoreWeight_critic", s, asr=0.20, acc=0.73)
            for s in range(3)
        ],
    }
    rows = build_rows(runs, tmp_path, window=50)
    assert len(rows) == 1
    row = rows[0]
    assert row.base == "atk"
    assert row.seeds == [0, 1, 2]
    assert row.asr_undef == pytest.approx(0.80)
    assert row.asr_def == pytest.approx(0.20)
    assert row.acc_undef == pytest.approx(0.70)
    assert row.acc_def == pytest.approx(0.73)
    # No ledger files on disk -> ledger stats are absent, not an error.
    assert row.ledger is None


def test_build_rows_ignores_unpaired_defended(tmp_path) -> None:
    # Defended run whose base attack is missing is skipped, not crashed.
    runs = {
        "orphan__ScoreWeight_critic": [
            _run("orphan__ScoreWeight_critic", 0, asr=0.2, acc=0.7)
        ],
    }
    assert build_rows(runs, tmp_path, window=50) == []
