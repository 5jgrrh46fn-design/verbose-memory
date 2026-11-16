"""ASR loader excludes target-class samples and stamps the trigger."""

from __future__ import annotations

import torch

from topfed.data.cifar10 import build_asr_loader


def test_asr_loader_excludes_target_class_and_relabels() -> None:
    test_x = torch.zeros(20, 3, 32, 32)
    test_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2, dtype=torch.long)

    loader = build_asr_loader(
        test_x,
        test_y,
        target_class=9,
        trigger_size=4,
        trigger_amplitude=1.0,
        batch_size=8,
    )

    all_x = []
    all_y = []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
    x_cat = torch.cat(all_x, dim=0)
    y_cat = torch.cat(all_y, dim=0)

    # 2 of 20 samples are class 9 — excluded.
    assert x_cat.size(0) == 18
    # Every retained sample's label is rewritten to 9.
    assert torch.all(y_cat == 9)
    # Every retained sample has the trigger stamped in the bottom-right corner.
    assert torch.all(x_cat[:, :, -4:, -4:] == 1.0)
    # Pixels outside the patch are still zero (untouched).
    assert torch.all(x_cat[:, :, :-4, :] == 0.0)
