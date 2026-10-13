# -*- coding: utf-8 -*-
"""Управление: куда кликать — из найденного на кадре, с поправкой на калибровку."""

import pytest

from aggregator import Aggregator
from control import CHOKE_MAX, CHOKE_MIN, CHOKE_STEP, Controller


class FakeHid:
    name = "тест"
    connected = True
    rtt_ms = None
    info = {}

    def __init__(self):
        self.clicks = []

    def click(self, x, y):
        self.clicks.append((x, y))
        return {}


def groups(at=(1630, 640)):
    return {"button": [
        {"id": "btn_start", "label": "ПУСК", "at": list(at), "size": [130, 52]},
        {"id": "sp_choke_dn", "label": "−", "at": [1630, 864], "size": [52, 48]},
        {"id": "sp_choke_up", "label": "+", "at": [1846, 864], "size": [52, 48]}]}


def test_click_goes_to_found_button_through_calibration():
    hid = FakeHid()
    ctl = Controller(hid, Aggregator())
    # щит вписан в кадр трансляции с масштабом 0,5 и сдвигом (10, 40)
    ctl.on_reading(groups(), (0.5, 10.0, 40.0))
    where = ctl.press("btn_start")
    cx, cy = 1630 + 65, 640 + 26
    assert where["panel"] == [cx, cy]
    assert hid.clicks == [pytest.approx((10 + cx * 0.5, 40 + cy * 0.5))]


def test_button_moved_with_block_is_followed():
    hid = FakeHid()
    ctl = Controller(hid, Aggregator())
    ctl.on_reading(groups(at=(200, 300)), None)        # компоновка переставила пульт
    ctl.press("btn_start")
    assert hid.clicks == [(265.0, 326.0)]


def test_missing_button_is_an_error():
    ctl = Controller(FakeHid(), Aggregator())
    ctl.on_reading({}, None)
    with pytest.raises(RuntimeError):
        ctl.press("btn_start")


def test_choke_homes_to_stop_then_steps_up():
    hid = FakeHid()
    ctl = Controller(hid, Aggregator())
    ctl._wait_frames = lambda n, timeout: None          # кадров в тесте нет
    ctl.on_reading(groups(), None)
    ok, _ = ctl._set_choke(60)
    down = (CHOKE_MAX - CHOKE_MIN) // CHOKE_STEP + 1
    up = (60 - CHOKE_MIN) // CHOKE_STEP
    assert ok
    assert len(hid.clicks) == down + up
    assert all(x == 1630 + 26 for x, _ in hid.clicks[:down])
    assert all(x == 1846 + 26 for x, _ in hid.clicks[down:])


def test_estop_requires_confirmation():
    ctl = Controller(FakeHid(), Aggregator())
    with pytest.raises(ValueError):
        ctl.submit("estop")
