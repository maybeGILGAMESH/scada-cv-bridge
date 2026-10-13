# -*- coding: utf-8 -*-
"""Каждый транспорт доносит кадр целым: источник и приёмник в одном процессе."""

import itertools
import threading
import time

import pytest

from protocol import JPEG, Frame
from transports import TRANSPORTS, make_receiver, make_sender

_ports = itertools.count(18800 + (int(time.time()) % 50) * 20)


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_frame_arrives_intact(kind):
    port = next(_ports)
    payload = bytes(range(256)) * 1200            # ~300 КБ — как JPEG-кадр щита
    snd = make_sender(kind, "127.0.0.1", port)
    stop = threading.Event()

    def pump():                                     # источник шлёт, пока приёмник не поймает
        seq = 0
        while not stop.is_set():
            snd.send(Frame(seq, time.perf_counter(), JPEG, 1920, 1080, payload))
            seq += 1
            time.sleep(0.02)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    rcv = make_receiver(kind, "127.0.0.1", port, connect_timeout=10)
    try:
        got = None
        t_end = time.time() + 10
        while got is None and time.time() < t_end:
            got = rcv.recv(timeout=1.0)
        assert got is not None, f"{kind}: кадр не пришёл за 10 с"
        assert bytes(got.payload) == payload
        assert (got.w, got.h, got.codec) == (1920, 1080, JPEG)
        assert got.latency_ms < 1000
    finally:
        stop.set()
        th.join(timeout=2)
        rcv.close()
        snd.close()
