"""Output paths are content-addressed and deterministic."""

from __future__ import annotations

from topfed.runtime.paths import config_hash, output_dir


def test_config_hash_is_stable_for_identical_input() -> None:
    a = config_hash("federation:\n  num_rounds: 300\n")
    b = config_hash("federation:\n  num_rounds: 300\n")
    assert a == b
    assert len(a) == 16


def test_config_hash_changes_when_content_changes() -> None:
    a = config_hash("federation:\n  num_rounds: 300\n")
    b = config_hash("federation:\n  num_rounds: 500\n")
    assert a != b


def test_output_dir_is_created_and_correctly_shaped(tmp_path) -> None:
    p = output_dir(tmp_path, "exp1", "deadbeefdeadbeef", seed=7)
    assert p.exists() and p.is_dir()
    assert p.parts[-3:] == ("exp1", "deadbeefdeadbeef", "seed_7")
