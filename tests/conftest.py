# -*- coding: utf-8 -*-
"""Общее для тестов: пути к модулям и контрольные кадры щита."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "bridge", ROOT / "live", ROOT / "cv_templates"):
    sys.path.insert(0, str(p))

TEMPLATES = ROOT / "cv_templates"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


@pytest.fixture(scope="session")
def validation():
    """14 контрольных режимов: (имя, кадр BGR, истина модели)."""
    idx = json.loads((TEMPLATES / "validation" / "index.json").read_text(encoding="utf-8"))
    return [(f["scenario"], imread(TEMPLATES / f["image"]),
             json.loads((TEMPLATES / f["truth"]).read_text(encoding="utf-8")))
            for f in idx["frames"]]


@pytest.fixture(scope="session")
def reference_frame():
    return imread(TEMPLATES / "reference" / "frame.png")
