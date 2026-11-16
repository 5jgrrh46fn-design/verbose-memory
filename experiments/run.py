"""Hydra entry-point for a single run (FedAvg, a defense or a baseline).

Usage::

    uv run python experiments/run.py +experiment=backdoor_cifar10_f10
    uv run python experiments/run.py +experiment=backdoor_cifar10_f20

Sweep seeds via Hydra multirun::

    uv run python experiments/run.py --multirun \\
        +experiment=backdoor_cifar10_f10 federation.seed=0,1,2,3,4

Outputs land in ``${results_root}/${experiment}/${config_hash}/seed_${seed}/``;
``+tag=<name>`` substitutes ``<name>`` for ``${experiment}`` in that path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from topfed.aggregation import (
    FLAME,
    MESAS,
    ClippedMean,
    FoolsGold,
    Krum,
    coordinate_median_aggregate,
    trimmed_mean_aggregate,
)
from topfed.attacks.assignment import deterministic_malicious_set
from topfed.attacks.data.clean_label import CleanLabelPoisoner
from topfed.attacks.model.collusion import (
    CollusionCoordinator,
    build_collusion_manipulator,
)
from topfed.attacks.model.scale import ModelReplacementManipulator
from topfed.core.seeding import configure_determinism, set_all_seeds
from topfed.core.types import (
    ClientId,
    EvidenceObserver,
    FederationConfig,
    RoundOutcome,
    RoundScorer,
)
from topfed.data import (
    build_asr_loader,
    build_client_loaders,
    build_partition,
    load_cifar10,
    load_cifar100,
    load_femnist,
    pad_images,
)
from topfed.federation.client import (
    BenignTorchClient,
    TorchClient,
    get_parameters,
)
from topfed.federation.evaluation import TorchEvaluator
from topfed.federation.median_selection import MedianSelectionRecorder
from topfed.federation.scoring_bridge import (
    AgreementMetricScorer,
    HistorySimilarityScorer,
    MetricScorer,
    ServerCosineScorer,
    ServerRootedMetricScorer,
)
from topfed.federation.server import (
    fltrust_aggregate,
    fltrust_median_aggregate,
    run_federation,
)
from topfed.integration.oracle import OracleFilter
from topfed.metrics import MetricsExtractor, RoundMetricRow
from topfed.aggregation.indicator import BackdoorIndicator
from topfed.data.cifar10 import CIFAR10_NORMALIZE_MEAN, CIFAR10_NORMALIZE_STD
from topfed.data.cifar100 import CIFAR100_NUM_CLASSES, CIFAR100_NUM_COARSE_CLASSES
from topfed.data.ood import load_ood_images
from topfed.models.cifar_cnn import CIFAR10_NUM_CLASSES, CifarCNN
from topfed.models.femnist_cnn import FEMNIST_NUM_CLASSES, FemnistCNN
from topfed.models.resnet_cifar import ResNet18GN
from topfed.scoring import WEIGHTING_STRATEGIES, TopsisScorer
from topfed.runtime import (
    config_hash,
    output_dir,
    write_client_metrics,
    write_manifest,
    write_metrics,
)
from torch.utils.data import DataLoader, TensorDataset


def _build_poisoner(cfg: DictConfig, device: str, model_factory):
    """Hydra-instantiate the configured poisoner.

    ``CleanLabelPoisoner`` needs runtime values (model factory, device)
    that Hydra cannot supply from YAML, so it is constructed directly
    with those injected; YAML-supplied fields (target_class,
    trigger_size, PGD config) flow through verbatim.
    """
    block = cfg.attack.poisoner
    if block is None:
        return None
    target = block.get("_target_") if hasattr(block, "get") else None
    if target == "topfed.attacks.data.clean_label.CleanLabelPoisoner":
        kwargs = {
            k: v for k, v in block.items() if k != "_target_"
        }
        return CleanLabelPoisoner(
            model_factory=model_factory,
            device=device,
            **kwargs,
        )
    return instantiate(block)


def _build_manipulator(
    cfg: DictConfig,
    federation_cfg: FederationConfig,
    malicious: frozenset[ClientId],
):
    """Hydra-instantiate the configured update manipulator.

    ``ModelReplacementManipulator`` with ``gamma=null`` is a sentinel for
    round-aware scaling: the manipulator needs the clients-per-round budget
    K and the malicious-client set, neither of which Hydra can supply from
    YAML, so we build it directly with those runtime values injected.

    An ``attack.collusion`` block (``strategy`` plus the strategy's knobs)
    wraps the configured manipulator in that collusion strategy, sharing one
    round-level coordinator that knows the malicious set.
    """
    block = cfg.attack.update_manipulator
    base = None
    if block is not None:
        target = block.get("_target_") if hasattr(block, "get") else None
        if (
            target == "topfed.attacks.model.scale.ModelReplacementManipulator"
            and block.get("gamma") is None
        ):
            base = ModelReplacementManipulator(
                gamma=None,
                clients_per_round=federation_cfg.clients_per_round,
                malicious_clients=malicious,
            )
        else:
            base = instantiate(block)
    collusion = cfg.attack.get("collusion", None)
    if collusion is None:
        return base
    knobs = {k: v for k, v in collusion.items() if k != "strategy"}
    return build_collusion_manipulator(
        str(collusion.strategy), CollusionCoordinator(malicious), base, **knobs
    )


def _build_val_loader(bundle) -> DataLoader:
    return DataLoader(
        TensorDataset(bundle.val_x, bundle.val_y),
        batch_size=256,
        shuffle=False,
    )


def _build_evaluator(
    cfg: DictConfig, bundle, val_loader: DataLoader, device: str, model_factory
) -> TorchEvaluator:
    loaders: dict[str, DataLoader] = {"val": val_loader}
    test_loader = DataLoader(
        TensorDataset(bundle.test_x, bundle.test_y),
        batch_size=256,
        shuffle=False,
    )
    loaders["test"] = test_loader

    if cfg.attack.kind in ("backdoor", "clean_label"):
        asr_loader = build_asr_loader(
            bundle.test_x,
            bundle.test_y,
            target_class=cfg.attack.poisoner.target_class,
            trigger_size=cfg.attack.poisoner.trigger_size,
            trigger_amplitude=cfg.attack.poisoner.trigger_amplitude,
            trigger_position=str(
                cfg.attack.poisoner.get("trigger_position", "bottom_right")
            ),
        )
        loaders["asr"] = asr_loader

    return TorchEvaluator(
        model_factory=model_factory, loaders=loaders, device=device
    )


def _build_client_factory(
    cfg: DictConfig,
    train_loaders: dict[ClientId, DataLoader],
    malicious: frozenset[ClientId],
    device: str,
    model_factory,
    manipulator,
):
    poisoner = _build_poisoner(cfg, device, model_factory)
    augmenter = (
        instantiate(cfg.attack.loss_augmenter)
        if cfg.attack.loss_augmenter is not None
        else None
    )
    schedule = (
        instantiate(cfg.attack.schedule) if cfg.attack.schedule is not None else None
    )

    def factory(cid: ClientId):
        if cid in malicious:
            return TorchClient(
                client_id=cid,
                model_factory=model_factory,
                train_loader=train_loaders[cid],
                device=device,
                poisoner=poisoner,
                loss_augmenter=augmenter,
                update_manipulator=manipulator,
                schedule=schedule,
            )
        return BenignTorchClient(
            client_id=cid,
            model_factory=model_factory,
            train_loader=train_loaders[cid],
            device=device,
        )

    return factory


def _build_integrator(defense_cfg: DictConfig, malicious: frozenset[ClientId]):
    """Hydra-instantiate the configured integrator.

    ``OracleFilter`` is a diagnostic that needs the ground-truth malicious
    set, which Hydra cannot supply from YAML, so it is built directly with
    that runtime value injected. Every other integrator is instantiated
    verbatim and never sees the set.
    """
    block = defense_cfg.integrator
    target = block.get("_target_") if hasattr(block, "get") else None
    if target == "topfed.integration.oracle.OracleFilter":
        return OracleFilter(malicious_clients=malicious)
    return instantiate(block)


@dataclass(frozen=True)
class _Loaded:
    """Dataset tensors plus the pieces the rest of the run derives from them."""

    bundle: Any
    model_factory: Callable[[], torch.nn.Module]
    partition: list


def _select_model(
    cfg: DictConfig, dataset_name: str, default: Callable[..., torch.nn.Module]
) -> Callable[..., torch.nn.Module]:
    """Resolve the optional ``model`` block to a backbone constructor.

    Without a ``model`` block the dataset's default CNN is used. The block
    is optional so that configs written before it existed keep their
    content-addressed hash. ``resnet18_gn`` is the only alternative and is
    defined for CIFAR-10 only.
    """
    model_cfg = cfg.get("model", None)
    if model_cfg is None:
        return default
    name = str(model_cfg.get("name", ""))
    if name != "resnet18_gn":
        raise SystemExit(f"unknown model: {name!r} (known: resnet18_gn)")
    if dataset_name != "cifar10":
        raise SystemExit(
            f"model {name} is defined for cifar10 only; dataset is {dataset_name}"
        )
    return partial(ResNet18GN, groups=int(model_cfg.get("groups", 2)))


def _load_dataset(cfg: DictConfig, federation_cfg: FederationConfig) -> _Loaded:
    """Load the dataset, apply the optional constant border, and partition it.

    The partition kind comes from ``cfg.partition.kind`` for every dataset:
    ``natural`` needs per-sample writer ids (FEMNIST only), ``dirichlet``
    and ``iid`` pool all training samples and re-split them, so the same
    dataset can be run under synthetic heterogeneity by config alone.
    """
    dataset_name = cfg.dataset.name
    pad_border = int(cfg.dataset.get("pad_border", 0))
    writer_ids = None
    bundle: Any
    if dataset_name == "cifar10":
        bundle = load_cifar10(
            root=cfg.data_root,
            val_per_class=cfg.dataset.val_per_class,
            seed=cfg.partition.seed,
            download=True,
        )
        base_size = 32
        model_cls: Callable[..., torch.nn.Module] = CifarCNN
    elif dataset_name in ("cifar100", "cifar100_coarse"):
        # The fine (100-class) and superclass (20-class) variants share the
        # loader; ``coarse`` picks the label set and ``num_classes`` must
        # agree with it, so a config cannot pair the wrong head with it.
        coarse = bool(cfg.dataset.get("coarse", False))
        num_classes = int(cfg.dataset.num_classes)
        expected = CIFAR100_NUM_COARSE_CLASSES if coarse else CIFAR100_NUM_CLASSES
        if num_classes != expected:
            raise SystemExit(
                f"dataset {dataset_name}: num_classes={num_classes} but "
                f"coarse={coarse} gives {expected} classes"
            )
        bundle = load_cifar100(
            root=cfg.data_root,
            val_per_class=cfg.dataset.val_per_class,
            coarse=coarse,
            seed=cfg.partition.seed,
            download=True,
        )
        base_size = 32
        model_cls = partial(CifarCNN, num_classes=num_classes)
    elif dataset_name == "femnist":
        bundle = load_femnist(
            root=cfg.data_root,
            num_clients=federation_cfg.num_clients,
            val_per_class=cfg.dataset.val_per_class,
            seed=cfg.partition.seed,
            download=True,
        )
        base_size = 28
        model_cls = FemnistCNN
        writer_ids = bundle.train_writers
    else:
        raise SystemExit(f"unknown dataset: {dataset_name}")

    if pad_border > 0:
        bundle = replace(
            bundle,
            train_x=pad_images(bundle.train_x, pad_border),
            val_x=pad_images(bundle.val_x, pad_border),
            test_x=pad_images(bundle.test_x, pad_border),
        )
    input_size = base_size + 2 * pad_border
    model_cls = _select_model(cfg, dataset_name, model_cls)
    model_factory = partial(model_cls, input_size=input_size)

    kind = str(cfg.partition.kind)
    if kind == "natural" and writer_ids is None:
        raise SystemExit(
            f"partition kind 'natural' needs per-sample writer ids, which "
            f"{dataset_name} does not provide"
        )
    partition = build_partition(
        kind,
        labels=bundle.train_y.numpy(),
        num_clients=federation_cfg.num_clients,
        seed=cfg.partition.seed,
        alpha=cfg.partition.get("alpha", None),
        writer_ids=writer_ids,
    )
    return _Loaded(bundle=bundle, model_factory=model_factory, partition=partition)


def _build_aggregator(block, seed: int = 0):
    """Build a robust-aggregation rule from the ``baseline`` config block.

    ``seed`` feeds the rules that draw randomness (FLAME's noise), so a run
    is reproducible from its config and seed alone.
    """
    name = block.name
    if name == "krum":
        return Krum(
            num_byzantine=int(block.get("num_byzantine", 2)),
            num_selected=int(block.get("num_selected", 1)),
        )
    if name == "coordinate_median":
        return coordinate_median_aggregate
    if name == "trimmed_mean":
        return partial(
            trimmed_mean_aggregate, trim_ratio=float(block.get("trim_ratio", 0.2))
        )
    if name == "foolsgold":
        return FoolsGold(kappa=float(block.get("kappa", 1.0)))
    if name == "flame":
        size = block.get("min_cluster_size", None)
        return FLAME(
            noise_factor=float(block.get("noise_factor", 0.001)),
            min_cluster_size=None if size is None else int(size),
            seed=seed,
        )
    if name == "clipped_mean":
        bound = block.get("bound", None)
        return ClippedMean(bound=None if bound is None else float(bound))
    if name == "mesas":
        return MESAS(
            alpha=float(block.get("alpha", 0.03)),
            center=str(block.get("center", "median")),
        )
    raise SystemExit(f"unknown baseline: {name}")


def _build_indicator(block, cfg: DictConfig, bundle, model_factory, device: str, seed: int):
    """BackdoorIndicator with the OOD set matched to the host dataset.

    CIFAR-10 is served CIFAR-100; FEMNIST is served CIFAR-10 in grayscale at
    FEMNIST's resolution. Sample counts follow the reference (800 for
    CIFAR-10, 2000 for the EMNIST family) unless ``ood_samples`` is set.
    """
    dataset_name = str(cfg.dataset.name)
    pad_border = int(cfg.dataset.get("pad_border", 0))
    requested = block.get("ood_samples", None)
    if dataset_name == "cifar10":
        source, grayscale, size, num_classes = "cifar100", False, 32, CIFAR10_NUM_CLASSES
        mean: tuple[float, ...] | float = CIFAR10_NORMALIZE_MEAN
        std: tuple[float, ...] | float = CIFAR10_NORMALIZE_STD
        default_samples = 800
    elif dataset_name == "femnist":
        source, grayscale, size, num_classes = "cifar10", True, 28, FEMNIST_NUM_CLASSES
        mean, std = float(bundle.norm_mean), float(bundle.norm_std)
        default_samples = 2000
    else:
        raise SystemExit(f"backdoor_indicator has no OOD source for dataset {dataset_name}")
    ood_x = load_ood_images(
        source,
        cfg.data_root,
        int(requested) if requested is not None else default_samples,
        seed=seed,
        grayscale=grayscale,
        image_size=size,
        mean=mean,
        std=std,
        pad_border=pad_border,
    )
    return BackdoorIndicator(
        model_factory,
        ood_x,
        num_classes,
        device,
        threshold=float(block.get("threshold", 95.0)),
        retrain_epochs=int(block.get("retrain_epochs", 200)),
        lr=float(block.get("lr", 0.005)),
        momentum=float(block.get("momentum", 0.9)),
        weight_decay=float(block.get("weight_decay", 5e-4)),
        lr_gamma=float(block.get("lr_gamma", 0.8)),
        mu=float(block.get("mu", 0.4)),
        batch_size=int(block.get("batch_size", 64)),
        eta=float(block.get("eta", 1.0)),
        start_round=int(block.get("start_round", 1)),
        seed=seed,
    )


def _baseline_ledger(outcomes, aggregator, malicious: frozenset[ClientId]) -> list[dict]:
    """Per-client aggregation weights of a baseline rule, one row per round.

    Rows exist for the rounds whose rule reported weights (FoolsGold and Krum
    always, the server-rooted rules when the run traces capture). A rule's own
    per-client records (FoolsGold's similarity and learning rate) are joined in.
    """
    extras = {
        (r["round"], r["client_id"]): r for r in getattr(aggregator, "records", [])
    }
    rows: list[dict] = []
    for outcome in outcomes:
        if outcome.weights is None:
            continue
        for update, weight in zip(outcome.client_updates, outcome.weights, strict=True):
            key = (outcome.round_num, update.client_id)
            rows.append(
                {
                    **extras.get(key, {}),
                    "round": outcome.round_num,
                    "client_id": update.client_id,
                    "weight": float(weight),
                    "is_malicious": update.client_id in malicious,
                }
            )
    return rows


# Server-rooted aggregation rules, by baseline name; each entry builds the
# rule from the ``baseline`` config block.
_ROOTED_BASELINES: dict[str, Callable[[Any], Any]] = {
    "fltrust": lambda block: fltrust_aggregate,
    "fltrust_median": lambda block: fltrust_median_aggregate,
    "fltrust_median_capped": lambda block: partial(
        fltrust_median_aggregate, cap=float(block.get("cap", 1.0))
    ),
}

# Weighting strategies that do not assume the canonical five-column order, so
# they also apply to a decision matrix with a sixth column added.
_COLUMN_AGNOSTIC_WEIGHTINGS = ("equal", "entropy", "critic")
_SCORER_KINDS = ("metrics", "server_cosine", "history_similarity")


def _build_server_update_fn(model_factory, val_loader: DataLoader, device: str):
    """The server's own round update, trained on the clean validation (root) set."""
    server_client = BenignTorchClient("__server__", model_factory, val_loader, device)

    def server_update_fn(ctx, fed_cfg):  # noqa: ANN001
        return server_client.local_train(ctx, fed_cfg).parameters

    return server_update_fn


def _needs_server_update(*components: object) -> bool:
    return any(
        isinstance(c, EvidenceObserver) and c.requires_server_update
        for c in components
    )


def run_from_cfg(cfg: DictConfig) -> Path:
    """Execute one run from a fully-resolved Hydra config and return its output dir."""
    federation_cfg: FederationConfig = instantiate(cfg.federation)
    determinism = configure_determinism(
        bool(cfg.reproducibility.deterministic)
    )
    set_all_seeds(federation_cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = _load_dataset(cfg, federation_cfg)
    bundle = loaded.bundle
    model_factory = loaded.model_factory
    partition = loaded.partition
    pad_border = int(cfg.dataset.get("pad_border", 0))

    train_loaders = build_client_loaders(
        bundle.train_x,
        bundle.train_y,
        partition,
        batch_size=federation_cfg.local_batch_size,
        augment=cfg.dataset.augment,
        pad_border=pad_border,
    )

    malicious = deterministic_malicious_set(
        num_clients=federation_cfg.num_clients,
        fraction=cfg.attack.assignment.fraction,
        seed=federation_cfg.seed,
    )

    manipulator = _build_manipulator(cfg, federation_cfg, malicious)
    client_factory = _build_client_factory(
        cfg, train_loaders, malicious, device, model_factory, manipulator
    )
    val_loader = _build_val_loader(bundle)
    evaluator = _build_evaluator(cfg, bundle, val_loader, device, model_factory)
    extractor = MetricsExtractor(
        val_loader=val_loader,
        model_factory=model_factory,
        device=device,
    )

    # Optional online defense: TOPSIS scoring + an integration mechanism wired
    # into the aggregation step. Absent ``defense`` leaves a plain FedAvg run
    # (and an unchanged config hash).
    defense_cfg = cfg.get("defense", None)
    scorer_bridge: RoundScorer | None = None
    integrator = None
    defended_base = "mean"
    geometric_median_options: dict[str, Any] | None = None
    # An optional ``+tag=<name>`` replaces the experiment name in the output
    # label so a diagnostic variant of an experiment (forced co-selection,
    # moved trigger, ...) never shares a label directory with the canonical
    # cell. The tag is part of the hashed config.
    run_label = str(cfg.get("tag") or cfg.experiment)
    server_update_fn = None
    if defense_cfg is not None:
        weighting_name = defense_cfg.scorer.weighting
        topsis = TopsisScorer(weighting=WEIGHTING_STRATEGIES[weighting_name])
        defense_extractor = MetricsExtractor(
            val_loader=val_loader, model_factory=model_factory, device=device
        )
        # ``defense.scorer.kind`` selects the trust score the integrator
        # consumes: ``metrics`` (the default, also when the key is absent) is
        # the TOPSIS score over the behavioural metrics; ``server_cosine`` and
        # ``history_similarity`` are single-signal scores that ignore the
        # weighting and the sixth-criterion switches.
        scorer_kind = str(defense_cfg.scorer.get("kind", "metrics"))
        if scorer_kind not in _SCORER_KINDS:
            raise SystemExit(
                f"unknown defense.scorer.kind {scorer_kind!r}; expected one of {_SCORER_KINDS}"
            )
        # ``defense.scorer.server_cosine: true`` adds the cosine to the
        # server's root-set delta as a sixth benefit criterion.
        # ``defense.scorer.pairwise_agreement: true`` instead adds the
        # cross-client history agreement as a sixth cost criterion.
        server_cosine = bool(defense_cfg.scorer.get("server_cosine", False))
        agreement = bool(defense_cfg.scorer.get("pairwise_agreement", False))
        if server_cosine and agreement:
            raise SystemExit(
                "server_cosine and pairwise_agreement are alternative sixth "
                "criteria; enable one"
            )
        if (server_cosine or agreement) and weighting_name not in _COLUMN_AGNOSTIC_WEIGHTINGS:
            raise SystemExit(
                f"weighting {weighting_name!r} assumes the five canonical "
                f"metrics; the {'server-cosine' if server_cosine else 'pairwise-agreement'} "
                f"criterion needs one of {_COLUMN_AGNOSTIC_WEIGHTINGS}"
            )
        if scorer_kind != "metrics" and (server_cosine or agreement):
            raise SystemExit(
                f"defense.scorer.kind={scorer_kind} is a single-signal score; "
                "server_cosine and pairwise_agreement add a criterion to the "
                "metric score only"
            )
        if scorer_kind == "server_cosine":
            scorer_bridge = ServerCosineScorer()
        elif scorer_kind == "history_similarity":
            scorer_bridge = HistorySimilarityScorer(
                kappa=float(defense_cfg.scorer.get("kappa", 1.0))
            )
        elif server_cosine:
            scorer_bridge = ServerRootedMetricScorer(defense_extractor, topsis)
        elif agreement:
            scorer_bridge = AgreementMetricScorer(defense_extractor, topsis)
        else:
            scorer_bridge = MetricScorer(defense_extractor, topsis)
        integrator = _build_integrator(defense_cfg, malicious)
        defended_base = str(defense_cfg.get("base", "mean"))
        base_tag = {
            "mean": "",
            "median": "_median",
            "weighted_median": "_wmedian",
            "geometric_median": "_gmedian",
        }[defended_base]
        # ``defense.geometric_median`` carries the Weiszfeld settings of the
        # geometric-median base (``nu``, ``max_iter``).
        gm_cfg = defense_cfg.get("geometric_median", None)
        if gm_cfg is not None:
            geometric_median_options = {str(k): gm_cfg[k] for k in gm_cfg}
        scorer_tag = "_rooted" if server_cosine else ("_agree" if agreement else "")
        # A single-signal score names itself in the label where the metric
        # score names its weighting.
        score_name = (
            f"{weighting_name}{scorer_tag}" if scorer_kind == "metrics" else scorer_kind
        )
        run_label = f"{run_label}__{type(integrator).__name__}_{score_name}{base_tag}"
        if _needs_server_update(scorer_bridge, integrator):
            server_update_fn = _build_server_update_fn(
                model_factory, val_loader, device
            )

    # Optional robust-aggregation baseline (no trust scoring), mutually
    # exclusive with the defense.
    baseline_cfg = cfg.get("baseline", None)
    aggregator = None
    rooted_aggregator = fltrust_aggregate
    if baseline_cfg is not None:
        if defense_cfg is not None:
            raise SystemExit("set either defense or baseline, not both")
        run_label = f"{run_label}__{baseline_cfg.name}"
        if baseline_cfg.name in _ROOTED_BASELINES:
            # FLTrust and its variants train a server model on the clean
            # validation (root) set each round; its update is the trusted
            # reference direction.
            server_update_fn = _build_server_update_fn(
                model_factory, val_loader, device
            )
            rooted_aggregator = _ROOTED_BASELINES[baseline_cfg.name](baseline_cfg)
        elif baseline_cfg.name == "backdoor_indicator":
            aggregator = _build_indicator(
                baseline_cfg, cfg, bundle, model_factory, device, int(federation_cfg.seed)
            )
        else:
            aggregator = _build_aggregator(baseline_cfg, seed=int(federation_cfg.seed))

    initial_parameters = get_parameters(model_factory())

    # Exclude the seed from the hash so all seeds of one config share a
    # parent directory (results/{experiment}/{hash}/seed_{seed}/).
    hash_container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(hash_container, dict)
    hash_container["federation"].pop("seed", None)
    # Operational flags must not change the content-addressed hash.
    hash_container.pop("skip_existing", None)
    # Determinism controls change kernels, not the experiment: they are
    # recorded in the manifest, not hashed, so a cell keeps one identity
    # across machines and determinism settings.
    hash_container.pop("reproducibility", None)
    # Diagnostics only read the round; the cell they describe keeps its hash.
    hash_container.pop("diagnostics", None)
    cfg_hash = config_hash(OmegaConf.to_yaml(hash_container))
    out_dir = output_dir(
        cfg.results_root, run_label, cfg_hash, federation_cfg.seed
    )
    # Resumable sweeps: skip a (config, seed) whose run already completed. A
    # crashed run leaves no metrics.json, so it is correctly re-run.
    if bool(cfg.get("skip_existing", False)) and (
        Path(out_dir) / "metrics.json"
    ).exists():
        print(f"[run] skip (already complete): {out_dir}")
        return out_dir
    write_manifest(
        out_dir,
        resolved_config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        cfg_hash=cfg_hash,
        seed=federation_cfg.seed,
        extra={
            "device": device,
            "determinism": determinism,
            "run_label": run_label,
            "num_malicious": len(malicious),
            "malicious_ids": sorted(malicious),
        },
    )

    print(
        f"[run] experiment={run_label} hash={cfg_hash} "
        f"seed={federation_cfg.seed} device={device} "
        f"malicious={len(malicious)}/{federation_cfg.num_clients} "
        f"-> {out_dir}"
    )

    # Per-client metrics are extracted as each round completes, so the run need
    # not keep every round's updates in memory until the end. Extraction
    # iterates a DataLoader, which draws from the global torch RNG; forking the
    # RNG keeps the training streams identical to extracting after the run.
    client_rows: list[RoundMetricRow] = []
    prev_global = initial_parameters
    rng_devices = [torch.cuda.current_device()] if device == "cuda" else []

    def extract_round(outcome: RoundOutcome) -> None:
        nonlocal prev_global
        with torch.random.fork_rng(devices=rng_devices):
            client_rows.extend(
                extractor.extract(
                    round_num=outcome.round_num,
                    updates=outcome.client_updates,
                    global_parameters=prev_global,
                    malicious=malicious,
                )
            )
        prev_global = outcome.aggregated_parameters

    # ``+diagnostics.median_selection=true`` records which client supplied
    # each coordinate of a median rule's aggregate; it reads the rounds and
    # writes ``median_selection.json`` beside the defense ledger.
    diagnostics_cfg = cfg.get("diagnostics", None)
    median_selection: MedianSelectionRecorder | None = None
    if diagnostics_cfg is not None and bool(diagnostics_cfg.get("median_selection", False)):
        median_selection = MedianSelectionRecorder(malicious)

    outcomes = run_federation(
        federation_cfg,
        client_factory,
        initial_parameters,
        evaluator,
        scorer=scorer_bridge,
        integrator=integrator,
        defended_base=defended_base,
        geometric_median_options=geometric_median_options,
        aggregator=aggregator,
        server_update_fn=server_update_fn,
        rooted_aggregator=rooted_aggregator,
        malicious_clients=malicious,
        on_round=extract_round,
        retain_parameters=False,
        median_selection=median_selection,
    )

    write_metrics(out_dir, outcomes)
    write_client_metrics(out_dir, client_rows)

    if integrator is not None:
        # A scorer with a sixth criterion logs the value it scored with; join
        # it onto the integrator's rows.
        scorer_records = getattr(scorer_bridge, "records", [])
        scored = {(r["round"], r["client_id"]): r for r in scorer_records}
        ledger = [
            {
                **record,
                **{
                    k: v
                    for k, v in scored.get(
                        (record["round"], record["client_id"]), {}
                    ).items()
                    if k not in ("round", "client_id")
                },
                "is_malicious": record["client_id"] in malicious,
            }
            for record in integrator.records
        ]
        (Path(out_dir) / "defense_ledger.json").write_text(
            json.dumps(ledger, indent=2, default=float)
        )

    if integrator is None and baseline_cfg is not None:
        baseline_rows = _baseline_ledger(outcomes, aggregator, malicious)
        if baseline_rows:
            (Path(out_dir) / "defense_ledger.json").write_text(
                json.dumps(baseline_rows, indent=2, default=float)
            )

    if median_selection is not None:
        (Path(out_dir) / "median_selection.json").write_text(
            json.dumps(median_selection.as_dict(), indent=2, default=float)
        )

    coordinator = getattr(manipulator, "coordinator", None)
    if coordinator is not None:
        (Path(out_dir) / "collusion_ledger.json").write_text(
            json.dumps(coordinator.records, indent=2)
        )

    final = outcomes[-1].metrics
    summary = " ".join(f"{k}={v:.4f}" for k, v in sorted(final.items()))
    print(f"[run] done. final-round metrics: {summary}")
    print(f"[run] artifacts in {Path(out_dir).resolve()}")
    return out_dir


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
