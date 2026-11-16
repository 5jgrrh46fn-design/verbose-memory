"""Label-consistent backdoor poisoner.

The malicious client perturbs target-class images with a PGD attack
against a surrogate classifier (the current global model), then stamps
the same trigger used by the dirty backdoor. Labels are preserved.

The intended effect: the natural features of the target class become
harder for the model to learn, leaving the trigger as the strongest
correlate of the target label. At evaluation time, ASR is measured
exactly like the dirty backdoor (trigger-stamped non-target inputs
classified as target), so the existing ASR pipeline carries over.

Mechanism per ``poison_batch``:

1. Filter the batch to samples whose label equals ``target_class``.
   Non-target samples are not poisoned, since stamping the trigger on
   them with their true labels would make the trigger ambiguous
   between classes.
2. Run untargeted L_inf PGD against the surrogate model to maximise
   the cross-entropy loss on the true (target) class — i.e. push the
   samples away from the target-class decision region.
3. Stamp the trigger pattern (same ``apply_trigger`` used by the dirty
   backdoor) on the perturbed samples.
4. Return the batch with target-class samples replaced by their
   perturbed-and-triggered counterparts; labels unchanged.

The surrogate is the per-round global model. ``set_reference`` is
invoked by the client at round start with the global parameters; the
surrogate is built lazily on first ``poison_batch`` call so the
poisoner can move tensors to the client's device.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from topfed.attacks.data.backdoor import apply_trigger
from topfed.core.types import NDArrays

DEFAULT_TARGET_CLASS = 9
DEFAULT_TRIGGER_SIZE = 6
DEFAULT_TRIGGER_AMPLITUDE = 1.0
DEFAULT_PGD_EPS = 0.125
DEFAULT_PGD_ALPHA = 0.025
DEFAULT_PGD_STEPS = 3


class CleanLabelPoisoner:
    """Label-consistent backdoor: PGD perturbation + trigger on target class.

    ``pgd_eps`` and ``pgd_alpha`` are expressed in normalised input
    space (the same space as ``trigger_amplitude``). For CIFAR-10 with
    the standard per-channel std of ~0.25, ``pgd_eps=0.125`` is
    approximately 8/255 in pixel space, matching the common L_inf
    perturbation budget in the adversarial-examples literature.

    ``model_factory`` and ``device`` are injected at experiment-runner
    time (not from Hydra YAML) because the surrogate model needs to
    match the per-run architecture and CUDA/CPU placement.
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module] | None = None,
        device: torch.device | str = "cpu",
        target_class: int = DEFAULT_TARGET_CLASS,
        trigger_size: int = DEFAULT_TRIGGER_SIZE,
        trigger_amplitude: float = DEFAULT_TRIGGER_AMPLITUDE,
        pgd_eps: float = DEFAULT_PGD_EPS,
        pgd_alpha: float = DEFAULT_PGD_ALPHA,
        pgd_steps: int = DEFAULT_PGD_STEPS,
    ) -> None:
        if trigger_size <= 0:
            raise ValueError(f"trigger_size must be positive; got {trigger_size}")
        if pgd_eps <= 0:
            raise ValueError(f"pgd_eps must be positive; got {pgd_eps}")
        if pgd_alpha <= 0:
            raise ValueError(f"pgd_alpha must be positive; got {pgd_alpha}")
        if pgd_steps <= 0:
            raise ValueError(f"pgd_steps must be positive; got {pgd_steps}")

        self._model_factory = model_factory
        self._device = torch.device(device)
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_amplitude = trigger_amplitude
        self.pgd_eps = pgd_eps
        self.pgd_alpha = pgd_alpha
        self.pgd_steps = pgd_steps

        self._reference: NDArrays | None = None
        self._surrogate: nn.Module | None = None

    def set_reference(self, global_parameters: NDArrays) -> None:
        """Capture round-start global parameters as the PGD surrogate.

        Invalidates any cached surrogate so the next ``poison_batch``
        rebuilds it with the new parameters.
        """
        self._reference = [np.asarray(p).copy() for p in global_parameters]
        self._surrogate = None

    def _ensure_surrogate(self) -> nn.Module:
        if self._surrogate is not None:
            return self._surrogate
        if self._reference is None:
            raise RuntimeError(
                "CleanLabelPoisoner.set_reference must be called before "
                "poison_batch()"
            )
        if self._model_factory is None:
            raise RuntimeError(
                "CleanLabelPoisoner requires model_factory to build the "
                "PGD surrogate; the experiment runner injects this"
            )
        model = self._model_factory().to(self._device)
        keys = list(model.state_dict().keys())
        if len(keys) != len(self._reference):
            raise ValueError(
                f"surrogate parameter mismatch: factory produces "
                f"{len(keys)} tensors, reference has {len(self._reference)}"
            )
        state_dict = {
            k: torch.from_numpy(np.asarray(v))
            for k, v in zip(keys, self._reference, strict=True)
        }
        model.load_state_dict(state_dict, strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self._surrogate = model
        return model

    def _pgd_untargeted(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """L_inf PGD that maximises CE(surrogate(x + delta), y)."""
        surrogate = self._ensure_surrogate()
        criterion = nn.CrossEntropyLoss()

        x = x.to(self._device)
        y = y.to(self._device)
        delta = torch.zeros_like(x, device=self._device).uniform_(
            -self.pgd_eps, self.pgd_eps
        )
        for _ in range(self.pgd_steps):
            delta = delta.detach().requires_grad_(True)
            logits = surrogate(x + delta)
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta + self.pgd_alpha * grad.sign()).detach()
            delta = delta.clamp(-self.pgd_eps, self.pgd_eps)
        return (x + delta).detach()

    def poison_batch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_mask = y == self.target_class
        if not target_mask.any():
            return x, y

        target_idx = torch.nonzero(target_mask, as_tuple=True)[0]
        x_target = x[target_idx]
        y_target = y[target_idx]

        x_perturbed = self._pgd_untargeted(x_target, y_target)
        x_perturbed = apply_trigger(
            x_perturbed,
            trigger_size=self.trigger_size,
            trigger_amplitude=self.trigger_amplitude,
        )

        x_out = x.clone()
        x_out[target_idx] = x_perturbed.to(x.device)
        return x_out, y
