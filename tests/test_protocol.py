# -*- coding: utf-8 -*-
"""Заголовок кадра и кодеки: что ушло, то и пришло."""

import numpy as np
import pytest

from capture import jpeg_size
from protocol import HDR_SIZE, JPEG, PNG, RAW, Frame, decode, encode, make_frame


def test_header_roundtrip():
    f = Frame(7, 123.5, JPEG, 1920, 1080, b"\xff\xd8payload\xff\xd9")
    buf = f.to_bytes()
    g = Frame.from_bytes(buf)
    assert (g.seq, g.codec, g.w, g.h) == (7, JPEG, 1920, 1080)
    assert g.t_cap == pytest.approx(123.5)
    assert bytes(g.payload) == b"\xff\xd8payload\xff\xd9"
    assert len(buf) == HDR_SIZE + len(f.payload)


def test_bad_magic_rejected():
    with pytest.raises(ValueError):
        Frame.from_bytes(b"XXXX" + bytes(HDR_SIZE))


@pytest.mark.parametrize("codec", [RAW, PNG])
def test_lossless_codecs(reference_frame, codec):
    img = reference_frame[:200, :300].copy()
    out = decode(codec, encode(img, codec), img.shape[1], img.shape[0])
    assert np.array_equal(out, img)


def test_jpeg_close_and_size(reference_frame):
    f = make_frame(1, img=reference_frame, codec=JPEG, quality=85)
    assert jpeg_size(bytes(f.payload)) == (1920, 1080)
    out = f.image()
    assert out.shape == reference_frame.shape
    assert np.abs(out.astype(int) - reference_frame).mean() < 4
