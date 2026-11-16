"""Run manifests and per-round metric persistence.

The manifest captures everything needed to reproduce a run: the resolved
config, the git SHA of the code (and whether the working tree was dirty), and
the software and hardware environment the numbers came from (library and
CUDA/cuDNN versions, GPU, driver, lockfile digest).
Per-round metrics land alongside as a JSON list. Parquet replacement is
left as a future optimisation once a tabular library is added.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from topfed.core.types import RoundOutcome

_REPO_ROOT = Path(__file__).resolve().parents[3]


def git_sha(repo_root: str | Path | None = None) -> str:
    """Return the current ``HEAD`` commit SHA, or ``"unknown"`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty(repo_root: str | Path | None = None) -> bool | None:
    """Whether tracked files differ from ``HEAD``; ``None`` if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip().splitlines()[0] or None
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def environment_info(repo_root: str | Path = _REPO_ROOT) -> dict[str, Any]:
    """Software and hardware provenance for a run."""
    lock = Path(repo_root) / "uv.lock"
    env: dict[str, Any] = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "flwr": _package_version("flwr"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "uv_lock_blake2": (
            hashlib.blake2b(lock.read_bytes(), digest_size=16).hexdigest()
            if lock.exists()
            else None
        ),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        env["nvidia_driver"] = _nvidia_driver()
    return env


def write_manifest(
    output_dir: str | Path,
    *,
    resolved_config: dict[str, Any],
    cfg_hash: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_hash": cfg_hash,
        "seed": seed,
        "git_sha": git_sha(_REPO_ROOT),
        "git_dirty": git_dirty(_REPO_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
        "config": resolved_config,
    }
    if extra:
        manifest.update(extra)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def write_metrics(
    output_dir: str | Path, outcomes: list[RoundOutcome]
) -> Path:
    """Write a flat per-round metrics record.

    Each row carries the evaluator's metrics plus the server's wall-clock
    timing breakdown for the round (``client_train_s``, ``scoring_s``, ...)
    and, when the run profiles memory, the server-side peak memory, and when
    it traces capture, the round's capture trace.
    ``client_updates`` and ``aggregated_parameters`` are intentionally
    omitted — those are persisted separately via ``write_client_metrics``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"round": o.round_num, **o.metrics, **o.timings, **o.memory, **o.trace}
        for o in outcomes
    ]
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(rows, indent=2))
    return path


def write_client_metrics(
    output_dir: str | Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    """Persist per-round per-client behavioural metrics as JSON.

    Each entry corresponds to one (round, selected client) pair. The
    schema is whatever the caller supplies; the metric-extraction module
    produces rows matching ``RoundMetricRow``. Switch to parquet when a
    tabular library is added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "client_metrics.json"
    path.write_text(json.dumps(rows, indent=2, default=float))
    return path
