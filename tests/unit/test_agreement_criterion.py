"""Pairwise history agreement as a sixth TOPSIS criterion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from topfed.aggregation import foolsgold_weights, history_similarity
from topfed.analysis.e5 import _defense_label
from topfed.core.types import ClientUpdate, NDArrays
from topfed.federation.scoring_bridge import AgreementMetricScorer
from topfed.scoring import TopsisScorer, critic_weights
from topfed.scoring.criteria import (
    AGREEMENT_TRUST_METRICS,
    METRIC_DIRECTION,
    PAIRWISE_AGREEMENT,
    TRUST_METRICS,
    benefit_mask,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class _StubExtractor:
    """Identical M1-M5 for every client, so only the agreement column differs."""

    def extract(self, round_num, updates, global_parameters, malicious):  # noqa: ANN001
        return [
            {"client_id": u.client_id, "m1_cosine_to_mean": 0.5, "m2_norm_ratio": 1.0,
             "m3_temporal_cv": 0.1, "m4_val_loss_delta": 0.2, "m5_max_layer_deviation": 0.3}
            for u in updates
        ]


def _u(cid: str, delta: np.ndarray) -> ClientUpdate:
    return ClientUpdate(cid, [delta.copy()], 1, 0.0, 1.0)


def test_agreement_is_a_cost_column_after_the_canonical_five() -> None:
    assert AGREEMENT_TRUST_METRICS == TRUST_METRICS + (PAIRWISE_AGREEMENT,)
    assert METRIC_DIRECTION[PAIRWISE_AGREEMENT] == -1
    assert benefit_mask(AGREEMENT_TRUST_METRICS).tolist() == [True, False, False, False, False, False]


def test_history_similarity_matches_foolsgold_max_similarity() -> None:
    rng = np.random.default_rng(0)
    h = rng.normal(size=(7, 30))
    cs, max_cs = history_similarity(h)
    assert np.allclose(np.diag(cs), 0.0)
    assert (max_cs >= 0).all()
    np.testing.assert_allclose(max_cs, foolsgold_weights(h)[1])


def test_scorer_ranks_a_colluding_bloc_below_honest_clients_and_logs_it() -> None:
    rng = np.random.default_rng(1)
    g: NDArrays = [np.zeros(50)]
    poison = rng.normal(size=50)
    scorer = AgreementMetricScorer(_StubExtractor(), TopsisScorer(weighting=critic_weights))  # type: ignore[arg-type]
    for r in range(1, 6):
        updates = [_u(f"h{i}", rng.normal(size=50)) for i in range(6)]
        updates += [_u(f"m{i}", poison + 0.3 * rng.normal(size=50)) for i in range(4)]
        ids, scores = scorer.score_round(r, updates, g)
    assert scorer.last_batch is not None
    assert scorer.last_batch.metric_names == AGREEMENT_TRUST_METRICS
    honest = scores[:6]
    bloc = scores[6:]
    assert bloc.max() < honest.min()
    last = [row for row in scorer.records if row["round"] == 5]
    assert all(row[PAIRWISE_AGREEMENT] > 0.8 for row in last if row["client_id"].startswith("m"))
    assert all(row[PAIRWISE_AGREEMENT] < 0.6 for row in last if row["client_id"].startswith("h"))


def test_history_persists_across_rounds_per_client() -> None:
    g: NDArrays = [np.zeros(4)]
    scorer = AgreementMetricScorer(_StubExtractor(), TopsisScorer())  # type: ignore[arg-type]
    e = np.eye(4)
    scorer.score_round(1, [_u("a", e[0]), _u("b", e[1])], g)
    scorer.score_round(2, [_u("a", e[1]), _u("b", e[1])], g)
    # a = e0 + e1, b = 2 e1 -> cosine 1/sqrt(2)
    row = [r for r in scorer.records if r["round"] == 2 and r["client_id"] == "a"][0]
    assert row[PAIRWISE_AGREEMENT] == pytest.approx(1 / np.sqrt(2))


def test_defense_label_and_config_compose() -> None:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose("config", overrides=[
            "+experiment=collusion_backdoor_cifar10_f30",
            "+defense=topsis_agreement_capped_weighted_median",
        ])
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(cfg, resolve=True)
    assert _defense_label(container) == "agree-capped-wmed"  # type: ignore[arg-type]
