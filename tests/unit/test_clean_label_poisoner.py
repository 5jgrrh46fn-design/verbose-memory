"""PGD perturbation, trigger stamping, label preservation for ``CleanLabelPoisoner``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from topfed.attacks.data.clean_label import CleanLabelPoisoner
from topfed.models.cifar_cnn import CifarCNN


def _mixed_batch(
    batch_size: int = 8, target_class: int = 9, num_target: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a batch with a controlled number of target-class samples."""
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.zeros(batch_size, dtype=torch.long)
    y[:num_target] = target_class
    # Non-target samples get a different label so we can verify they
    # aren't touched.
    y[num_target:] = 0
    return x, y


def _global_params(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]


def _make_poisoner(**overrides) -> CleanLabelPoisoner:
    defaults = dict(
        model_factory=CifarCNN,
        device="cpu",
        target_class=9,
        trigger_size=6,
        trigger_amplitude=1.0,
        pgd_eps=0.125,
        pgd_alpha=0.025,
        pgd_steps=2,  # keep tests fast; convergence checked separately
    )
    defaults.update(overrides)
    return CleanLabelPoisoner(**defaults)


def test_poison_batch_preserves_all_labels() -> None:
    poisoner = _make_poisoner()
    poisoner.set_reference(_global_params(CifarCNN()))
    x, y = _mixed_batch(batch_size=8, target_class=9, num_target=4)
    _, y_out = poisoner.poison_batch(x, y)
    assert torch.equal(y_out, y), "labels must be unchanged (label-consistent attack)"


def test_poison_batch_only_touches_target_class_samples() -> None:
    poisoner = _make_poisoner()
    poisoner.set_reference(_global_params(CifarCNN()))
    x, y = _mixed_batch(batch_size=8, target_class=9, num_target=4)
    x_out, _ = poisoner.poison_batch(x, y)
    # Non-target samples (indices 4..7) are byte-identical to input.
    assert torch.equal(x_out[4:], x[4:]), "non-target samples must be untouched"
    # Target samples (indices 0..3) are modified.
    assert not torch.equal(x_out[:4], x[:4]), "target samples must be perturbed"


def test_poison_batch_applies_trigger_to_target_samples() -> None:
    poisoner = _make_poisoner(trigger_size=6, trigger_amplitude=1.0)
    poisoner.set_reference(_global_params(CifarCNN()))
    x, y = _mixed_batch(batch_size=4, target_class=9, num_target=4)
    x_out, _ = poisoner.poison_batch(x, y)
    # The trigger is stamped after PGD, so the bottom-right 6x6 patch
    # on every target sample equals trigger_amplitude exactly.
    patch = x_out[:, :, -6:, -6:]
    expected = torch.full_like(patch, 1.0)
    assert torch.allclose(patch, expected)


def test_pgd_perturbation_respects_eps_bound() -> None:
    eps = 0.125
    poisoner = _make_poisoner(pgd_eps=eps, pgd_alpha=0.05, pgd_steps=5)
    poisoner.set_reference(_global_params(CifarCNN()))
    # All-target batch lets us check every output sample.
    x = torch.randn(4, 3, 32, 32)
    y = torch.full((4,), 9, dtype=torch.long)
    x_out, _ = poisoner.poison_batch(x, y)
    # Compare on the un-triggered region (everything outside the 6x6
    # bottom-right patch) — the trigger overwrites that patch, so an
    # L_inf check there would not measure PGD's bound.
    delta = (x_out - x)[:, :, :-6, :-6]
    assert delta.abs().max().item() <= eps + 1e-6


def test_no_target_samples_is_identity() -> None:
    poisoner = _make_poisoner(target_class=9)
    poisoner.set_reference(_global_params(CifarCNN()))
    x = torch.randn(4, 3, 32, 32)
    y = torch.zeros(4, dtype=torch.long)  # nothing in target class
    x_out, y_out = poisoner.poison_batch(x, y)
    assert torch.equal(x_out, x)
    assert torch.equal(y_out, y)


def test_set_reference_required_before_poison() -> None:
    poisoner = _make_poisoner()
    x, y = _mixed_batch()
    with pytest.raises(RuntimeError, match="set_reference"):
        poisoner.poison_batch(x, y)


def test_model_factory_required() -> None:
    poisoner = CleanLabelPoisoner(model_factory=None, device="cpu")
    poisoner.set_reference(_global_params(CifarCNN()))
    x, y = _mixed_batch()
    with pytest.raises(RuntimeError, match="model_factory"):
        poisoner.poison_batch(x, y)


def test_set_reference_invalidates_cached_surrogate() -> None:
    poisoner = _make_poisoner()
    poisoner.set_reference(_global_params(CifarCNN()))
    # Force surrogate construction.
    x, y = _mixed_batch()
    poisoner.poison_batch(x, y)
    assert poisoner._surrogate is not None
    # Setting a new reference must invalidate the cache.
    poisoner.set_reference(_global_params(CifarCNN()))
    assert poisoner._surrogate is None


def test_invalid_hyperparams_rejected() -> None:
    with pytest.raises(ValueError, match="trigger_size"):
        CleanLabelPoisoner(model_factory=CifarCNN, trigger_size=0)
    with pytest.raises(ValueError, match="pgd_eps"):
        CleanLabelPoisoner(model_factory=CifarCNN, pgd_eps=0)
    with pytest.raises(ValueError, match="pgd_alpha"):
        CleanLabelPoisoner(model_factory=CifarCNN, pgd_alpha=-0.1)
    with pytest.raises(ValueError, match="pgd_steps"):
        CleanLabelPoisoner(model_factory=CifarCNN, pgd_steps=0)
