# -*- coding: utf-8 -*-
"""Агрегатор: фильтр выбросов, уставки с гистерезисом, расчёт, мигание."""

from aggregator import STABLE_S, Aggregator


def g(**tags):
    """Группы как у распознавания: значения манометров и цифровых индикаторов."""
    out = {"gauge": [], "sevenseg_digit": []}
    for tid, v in tags.items():
        cls = "gauge" if tid.startswith("gauge") else "sevenseg_digit"
        out[cls].append({"id": tid, "label": tid, "unit": "", "value": v})
    return out


def test_single_glitch_filtered():
    a = Aggregator()
    for i, v in enumerate([80, 80, 199, 80, 80]):       # одиночный сбой распознавания
        a.update(100.0 + i, g(gauge_t=v))
    assert a.value("gauge_t") == 80
    assert not [e for e in a.events if e["tag"] == "gauge_t"]


def test_no_value_before_three_readings():
    a = Aggregator()
    a.update(1.0, g(gauge_t=150))
    a.update(2.0, g(gauge_t=150))
    assert a.value("gauge_t") is None
    a.update(3.0, g(gauge_t=150))
    assert a.value("gauge_t") == 150


def test_limit_with_hysteresis():
    a = Aggregator()
    t = 0.0
    for v in [90] * 3 + [125] * 4:                      # выше уставки аварии 120
        t += 1
        a.update(t, g(gauge_t=v))
    assert a.levels["gauge_t"] == "alarm"
    for v in [119] * 4:                                 # чуть ниже уставки — внутри гистерезиса
        t += 1
        a.update(t, g(gauge_t=v))
    assert a.levels["gauge_t"] == "alarm"
    for v in [80] * 4:
        t += 1
        a.update(t, g(gauge_t=v))
    assert a.levels.get("gauge_t") is None
    assert any("норму" in e["text"] for e in a.events)


def test_oil_rate_and_cross_check():
    a = Aggregator()
    for t in range(1, 8):
        a.update(float(t), g(gauge_q=60.0, disp_q=40.0, disp_wc=25.0))
    d = {x["id"]: x for x in a.derived()}
    assert d["q_oil"]["value"] == 45.0                  # 60 × (1 − 0,25)
    assert d["q_check"]["level"] == "warn"              # стрелка и цифры разошлись дольше 3 с


def test_blinking_annunciator_is_one_state():
    a = Aggregator()
    t = 0.0
    for i in range(12):
        t += 0.3
        a.update(t, {"annunciator": [{"id": "ann_temp", "label": "ПЕРЕГРЕВ ПЭД",
                                      "state": "on" if i % 2 else "off"}]})
    assert a.state("ann_temp") == "blink"
    flips = [e for e in a.events if e["tag"] == "ann_temp"]
    assert len(flips) <= 1


def test_state_change_needs_stability():
    a = Aggregator()
    lamp = lambda s: {"lamp": [{"id": "lamp_run", "label": "НАСОС В РАБОТЕ", "state": s}]}  # noqa: E731
    a.update(0.0, lamp("off"))
    a.update(STABLE_S + 0.1, lamp("off"))
    assert a.state("lamp_run") == "off"
    a.update(STABLE_S + 0.2, lamp("on"))
    assert a.state("lamp_run") == "off"                 # ещё не продержалось
    a.update(2 * STABLE_S + 0.4, lamp("on"))
    assert a.state("lamp_run") == "on"
