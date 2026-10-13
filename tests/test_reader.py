# -*- coding: utf-8 -*-
"""Чтение щита на 14 контрольных режимах: точность не ниже исходной."""

import numpy as np
import pytest

import match_templates as MT
from fast_reader import FastReader
from conftest import TEMPLATES


@pytest.fixture(scope="module")
def results(validation):
    fr = FastReader(str(TEMPLATES), register_every=1, cache=False)
    out = []
    for scen, img, truth in validation:
        rows = fr.read(img, force_register=True)["rows"]
        out.append((scen, rows, truth))
    fr.close()
    return out


def test_overall_accuracy(results):
    ok = total = 0
    for _, rows, truth in results:
        _, st = MT.compare(rows, truth, None)
        ok += sum(v[0] for v in st.values())
        total += sum(v[1] for v in st.values())
    # исходный match_templates.py на этих же кадрах: 919 из 924
    assert ok / total >= 0.994, f"{ok}/{total}"


def test_gauges_by_needle(results):
    errs = []
    for _, rows, truth in results:
        gt = {o["id"]: o for o in truth["objects"]}
        for r in rows:
            if r["cls"] != "gauge":
                continue
            g = gt[r["object_id"]]
            exp = min(max(g["value"], g["min"]), g["max"])
            errs.append(100 * abs(exp - r["value"]) / (g["max"] - g["min"]))
            assert r["method"] == "needle"
    assert len(errs) == 84
    assert max(errs) < 1.0 and np.mean(errs) < 0.3


def test_sevenseg_decimal_point(results):
    for _, rows, truth in results:
        gt = {o["id"]: o for o in truth["objects"]}
        r = next(x for x in rows if x["object_id"] == "disp_wc")
        if r.get("reliable"):
            assert r["value"] == pytest.approx(float(gt["disp_wc"]["text"]), abs=0.05)


def test_layout_block_not_counted_as_moved(reference_frame):
    fr = FastReader(str(TEMPLATES))
    fr.read(reference_frame, force_register=True)
    assert "log" not in fr.bank.blocks                  # блок без приборов не привязывается
    assert fr.layout_name()["moved"] == 0
    fr.close()
