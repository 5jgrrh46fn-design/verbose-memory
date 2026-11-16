"""Attack schedules: always-on and windowed activation."""

from __future__ import annotations

import pytest

from topfed.attacks.schedule import AlwaysAttack, WindowedAttack
from topfed.core.types import RoundContext


def _ctx(round_num: int) -> RoundContext:
    return RoundContext(
        round_num=round_num,
        total_rounds=200,
        global_parameters=[],
        selected_clients=(),
        seed=0,
    )


def test_always_attack_is_active_every_round() -> None:
    assert all(AlwaysAttack().is_active(_ctx(r)) for r in (1, 50, 200))


def test_window_is_closed_on_both_ends() -> None:
    w = WindowedAttack(start_round=1, end_round=100)
    assert w.is_active(_ctx(1))
    assert w.is_active(_ctx(100))
    assert not w.is_active(_ctx(101))
    assert not w.is_active(_ctx(200))


def test_delayed_start_is_open_ended() -> None:
    w = WindowedAttack(start_round=50)
    assert not w.is_active(_ctx(49))
    assert w.is_active(_ctx(50))
    assert w.is_active(_ctx(200))


def test_window_validation() -> None:
    with pytest.raises(ValueError):
        WindowedAttack(start_round=0)
    with pytest.raises(ValueError):
        WindowedAttack(start_round=10, end_round=9)
    assert WindowedAttack(start_round=10, end_round=10).is_active(_ctx(10))
