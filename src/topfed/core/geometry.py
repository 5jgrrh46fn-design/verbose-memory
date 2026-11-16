"""Update geometry shared by server-rooted scoring, integration and aggregation.

The server-rooted signal is the cosine between each client's delta and the
server's own delta, both taken from the same pre-round global parameters. The
server delta comes from training on a small clean root set, so an attacker
cannot observe it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from topfed.core.types import ClientUpdate, NDArrays

_EPS = 1e-12


def flatten(arrays: NDArrays) -> np.ndarray:
    """Concatenate a parameter list into one flat vector."""
    return np.concatenate([np.asarray(a).ravel() for a in arrays])


def flat_delta(parameters: NDArrays, global_parameters: NDArrays) -> np.ndarray:
    """``parameters - global_parameters`` as one flat vector."""
    return flatten(
        [p - g for p, g in zip(parameters, global_parameters, strict=True)]
    )


def server_cosines(
    updates: Sequence[ClientUpdate],
    global_parameters: NDArrays,
    server_parameters: NDArrays,
    eps: float = _EPS,
) -> np.ndarray:
    """Cosine of each client's delta to the server delta, aligned to ``updates``.

    A client whose delta is numerically zero, or a round in which the server
    did not move, gets cosine 0 (no evidence either way).
    """
    g0 = flat_delta(server_parameters, global_parameters)
    n0 = float(np.linalg.norm(g0))
    out = np.zeros(len(updates), dtype=np.float64)
    if n0 <= eps:
        return out
    for i, u in enumerate(updates):
        di = flat_delta(u.parameters, global_parameters)
        ni = float(np.linalg.norm(di))
        if ni > eps:
            out[i] = float(np.dot(di, g0) / (ni * n0))
    return out
