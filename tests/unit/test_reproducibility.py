"""Determinism controls and manifest provenance."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from topfed.core.seeding import configure_determinism
from topfed.runtime.manifest import environment_info, write_manifest


@pytest.fixture
def restore_torch_defaults() -> Iterator[None]:
    yield
    configure_determinism(False)


def test_deterministic_mode_pins_kernels_and_keeps_default_tf32(
    restore_torch_defaults: None,
) -> None:
    state = configure_determinism(True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32
    assert state["deterministic"] is True
    assert state["cublas_workspace_config"] is not None


def test_nondeterministic_mode_restores_torch_defaults(restore_torch_defaults: None) -> None:
    configure_determinism(True)
    state = configure_determinism(False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert state["deterministic"] is False


def test_deterministic_mode_reproduces_seeded_training(restore_torch_defaults: None) -> None:
    configure_determinism(True)

    def train() -> torch.Tensor:
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.Flatten(),
            torch.nn.Linear(4 * 6 * 6, 2),
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        x, y = torch.randn(8, 3, 8, 8), torch.randint(0, 2, (8,))
        for _ in range(3):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    assert torch.equal(train(), train())


def test_manifest_records_environment_and_git_state(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, resolved_config={}, cfg_hash="h", seed=0)
    manifest = json.loads(path.read_text())
    env = manifest["environment"]
    for key in ("torch", "numpy", "cuda_runtime", "cudnn", "uv_lock_blake2", "hostname"):
        assert key in env
    assert env["torch"] == torch.__version__
    assert manifest["git_dirty"] in (True, False, None)


def test_environment_info_without_lockfile(tmp_path: Path) -> None:
    assert environment_info(tmp_path)["uv_lock_blake2"] is None
