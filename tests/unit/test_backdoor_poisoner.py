"""Trigger geometry, label rewriting, and PMR honor for ``BackdoorPoisoner``."""

from __future__ import annotations

import pytest
import torch

from topfed.attacks.data.backdoor import BackdoorPoisoner, apply_trigger


def _zero_batch(batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(batch_size, 3, 32, 32)
    y = torch.zeros(batch_size, dtype=torch.long)
    return x, y


def test_apply_trigger_stamps_bottom_right_corner() -> None:
    x, _ = _zero_batch(batch_size=2)
    out = apply_trigger(x, trigger_size=4, trigger_amplitude=0.5)
    # Bottom-right 4x4 patch is set on every channel.
    assert torch.allclose(out[..., -4:, -4:], torch.full((2, 3, 4, 4), 0.5))
    # Everything else is untouched.
    assert torch.allclose(out[..., :-4, :], torch.zeros_like(out[..., :-4, :]))
    assert torch.allclose(out[..., -4:, :-4], torch.zeros_like(out[..., -4:, :-4]))


def test_apply_trigger_does_not_mutate_input() -> None:
    x, _ = _zero_batch()
    snapshot = x.clone()
    apply_trigger(x, trigger_size=4, trigger_amplitude=1.0)
    assert torch.equal(x, snapshot)


def test_poisoner_relabels_prefix_to_target() -> None:
    poisoner = BackdoorPoisoner(
        target_class=9, trigger_size=4, trigger_amplitude=1.0, poisoning_rate=0.5
    )
    x, y = _zero_batch(batch_size=8)
    y[:] = 3  # original labels
    x_out, y_out = poisoner.poison_batch(x, y)
    # First 4 samples are poisoned + relabeled.
    assert torch.all(y_out[:4] == 9)
    assert torch.all(y_out[4:] == 3)
    # Trigger appears only on poisoned samples.
    assert torch.all(x_out[:4, :, -4:, -4:] == 1.0)
    assert torch.all(x_out[4:, :, -4:, -4:] == 0.0)


def test_poisoner_zero_rate_is_identity() -> None:
    poisoner = BackdoorPoisoner(poisoning_rate=0.0)
    x, y = _zero_batch()
    y[:] = 7
    x_out, y_out = poisoner.poison_batch(x, y)
    assert torch.equal(x_out, x)
    assert torch.equal(y_out, y)


def test_poisoner_full_rate_relabels_everything() -> None:
    poisoner = BackdoorPoisoner(target_class=9, poisoning_rate=1.0)
    _, y = _zero_batch()
    x = torch.randn(8, 3, 32, 32)
    _, y_out = poisoner.poison_batch(x, y)
    assert torch.all(y_out == 9)


def test_poisoner_validates_rate() -> None:
    with pytest.raises(ValueError):
        BackdoorPoisoner(poisoning_rate=1.5)


def test_augment_mode_doubles_batch_with_clean_plus_triggered() -> None:
    poisoner = BackdoorPoisoner(
        target_class=9, trigger_size=4, trigger_amplitude=1.0,
        batch_mode="augment",
    )
    x, y = _zero_batch(batch_size=8)
    y[:] = 3
    x_out, y_out = poisoner.poison_batch(x, y)

    # Batch doubles: first half = original (clean), second half = triggered.
    assert x_out.shape == (16, 3, 32, 32)
    assert y_out.shape == (16,)
    # First half: original labels and untriggered images.
    assert torch.all(y_out[:8] == 3)
    assert torch.all(x_out[:8] == 0.0)
    # Second half: relabeled to target_class and triggered.
    assert torch.all(y_out[8:] == 9)
    assert torch.all(x_out[8:, :, -4:, -4:] == 1.0)
    # Clean portion of triggered images stays zero.
    assert torch.all(x_out[8:, :, :-4, :] == 0.0)


def test_augment_mode_ignores_poisoning_rate() -> None:
    # Augment is rate-independent; verify the contract.
    p_low = BackdoorPoisoner(batch_mode="augment", poisoning_rate=0.1)
    p_high = BackdoorPoisoner(batch_mode="augment", poisoning_rate=0.9)
    x, y = _zero_batch(batch_size=4)
    x_low, y_low = p_low.poison_batch(x, y)
    x_high, y_high = p_high.poison_batch(x, y)
    assert x_low.shape == x_high.shape == (8, 3, 32, 32)
    assert torch.equal(y_low, y_high)


def test_invalid_batch_mode_rejected() -> None:
    with pytest.raises(ValueError):
        BackdoorPoisoner(batch_mode="invalid")  # type: ignore[arg-type]
