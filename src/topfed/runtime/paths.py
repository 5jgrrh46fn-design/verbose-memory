"""Content-addressed output directories.

Each run writes to ``results/{experiment}/{config_hash}/{seed}/`` so that
re-running an identical config / seed combination overwrites the same
directory rather than scattering near-duplicate copies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def config_hash(resolved_config_yaml: str) -> str:
    """Stable 16-hex-character hash of a fully-resolved config YAML string."""
    digest = hashlib.blake2b(resolved_config_yaml.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def output_dir(
    results_root: str | Path,
    experiment: str,
    cfg_hash: str,
    seed: int,
) -> Path:
    path = Path(results_root) / experiment / cfg_hash / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path
