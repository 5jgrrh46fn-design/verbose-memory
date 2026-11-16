"""Trigger placement (corner, centre, border) and the padded-border pipeline."""

from __future__ import annotations

import pytest
import torch

from topfed.attacks.data.backdoor import (
    BackdoorPoisoner,
    apply_trigger,
    trigger_mask,
)
from topfed.data.cifar10 import build_asr_loader
from topfed.data.loaders import pad_images


def test_bottom_right_mask_matches_legacy_geometry() -> None:
    mask = trigger_mask(32, 32, trigger_size=4)
    assert mask.sum().item() == 16
    assert mask[-4:, -4:].all()
    assert not mask[:-4, :].any() and not mask[-4:, :-4].any()


def test_center_mask_is_centred_and_same_size() -> None:
    mask = trigger_mask(28, 28, trigger_size=6, trigger_position="center")
    assert mask.sum().item() == 36
    assert mask[11:17, 11:17].all()
    assert not mask[:11, :].any() and not mask[17:, :].any()
    assert not mask[:, :11].any() and not mask[:, 17:].any()


def test_border_mask_is_a_ring_of_the_given_width() -> None:
    mask = trigger_mask(40, 40, trigger_size=4, trigger_position="border")
    assert mask.sum().item() == 40 * 40 - 32 * 32
    assert mask[:4, :].all() and mask[-4:, :].all()
    assert mask[:, :4].all() and mask[:, -4:].all()
    assert not mask[4:-4, 4:-4].any()


def test_border_mask_needs_an_interior() -> None:
    with pytest.raises(ValueError):
        trigger_mask(8, 8, trigger_size=4, trigger_position="border")


def test_unknown_position_rejected() -> None:
    with pytest.raises(ValueError):
        trigger_mask(32, 32, trigger_size=4, trigger_position="top_left")
    with pytest.raises(ValueError):
        BackdoorPoisoner(trigger_position="top_left")


def test_apply_trigger_center_writes_only_the_centre_patch() -> None:
    x = torch.zeros(2, 1, 28, 28)
    out = apply_trigger(
        x, trigger_size=6, trigger_amplitude=-5.0, trigger_position="center"
    )
    assert torch.all(out[:, :, 11:17, 11:17] == -5.0)
    assert out.abs().sum().item() == pytest.approx(2 * 36 * 5.0)
    assert torch.equal(x, torch.zeros_like(x))  # input untouched


def test_poisoner_uses_its_position_in_both_batch_modes() -> None:
    x = torch.zeros(4, 3, 40, 40)
    y = torch.zeros(4, dtype=torch.long)
    ring = trigger_mask(40, 40, trigger_size=4, trigger_position="border")
    aug = BackdoorPoisoner(
        trigger_size=4, trigger_amplitude=1.0, batch_mode="augment",
        trigger_position="border",
    )
    xa, ya = aug.poison_batch(x, y)
    assert torch.all(xa[4:][:, :, ring] == 1.0)
    assert torch.all(xa[4:][:, :, ~ring] == 0.0)
    assert torch.all(xa[:4] == 0.0)
    rep = BackdoorPoisoner(
        trigger_size=4, trigger_amplitude=1.0, poisoning_rate=0.5,
        batch_mode="replace", trigger_position="border",
    )
    xr, yr = rep.poison_batch(x, y)
    assert torch.all(xr[:2][:, :, ring] == 1.0)
    assert torch.all(xr[2:] == 0.0)


def test_asr_loader_uses_the_same_position_as_the_poisoner() -> None:
    test_x = torch.zeros(6, 1, 28, 28)
    test_y = torch.tensor([0, 1, 2, 9, 4, 5])
    loader = build_asr_loader(
        test_x, test_y, target_class=9, trigger_size=6,
        trigger_amplitude=-5.0, trigger_position="center",
    )
    xb, yb = next(iter(loader))
    assert xb.shape[0] == 5  # target-class sample excluded
    assert torch.all(yb == 9)
    assert torch.all(xb[:, :, 11:17, 11:17] == -5.0)
    assert torch.all(xb[:, :, -4:, -4:] == 0.0)


def test_pad_images_adds_a_constant_border_and_keeps_content() -> None:
    x = torch.randn(3, 3, 32, 32)
    padded = pad_images(x, 4)
    assert padded.shape == (3, 3, 40, 40)
    assert torch.equal(padded[:, :, 4:-4, 4:-4], x)
    ring = trigger_mask(40, 40, trigger_size=4, trigger_position="border")
    assert torch.all(padded[:, :, ring] == 0.0)
    assert pad_images(x, 0) is x
    with pytest.raises(ValueError):
        pad_images(x, -1)


def test_border_trigger_on_padded_image_never_touches_content() -> None:
    # The whole point of the padded border: the trigger ring and the natural
    # content occupy disjoint pixels, so the triggered image keeps the
    # original content byte-for-byte.
    x = torch.randn(2, 3, 32, 32)
    padded = pad_images(x, 4)
    triggered = apply_trigger(
        padded, trigger_size=4, trigger_amplitude=1.0, trigger_position="border"
    )
    assert torch.equal(triggered[:, :, 4:-4, 4:-4], x)
    # A size-matched corner patch inside the padding also stays clear of it.
    corner = apply_trigger(padded, trigger_size=4, trigger_amplitude=1.0)
    assert torch.equal(corner[:, :, 4:-4, 4:-4], x)
