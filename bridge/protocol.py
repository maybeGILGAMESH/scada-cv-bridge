# -*- coding: utf-8 -*-
"""Формат кадра на «HDMI»-канале и команд на обратном канале.

Видеоканал несёт пакеты: заголовок фиксированной длины плюс полезная нагрузка.
Заголовок одинаков для всех транспортов — так их можно честно сравнивать
и подменять друг другом, не трогая ни захват, ни распознавание.

    magic  4s   b"SCF1"
    seq    u32  номер кадра у источника — по разрывам считаются потери
    t_cap  f64  момент захвата кадра        (clock() источника)
    t_send f64  момент отправки в транспорт (clock() источника)
    codec  u8   0 — сырой BGR, 1 — JPEG, 2 — PNG
    w, h   u16  размер кадра
    length u32  длина нагрузки

Часы — perf_counter: на одной машине он общий для всех процессов
(QueryPerformanceCounter в Windows, CLOCK_MONOTONIC в Linux), поэтому
задержку источник → приёмник можно мерить вычитанием. time.time() для этого
не годится: в Windows у него шаг 15,6 мс. Между разными машинами часы не
совпадают, там задержка оценивается половиной RTT обратного канала.

Обратный канал — строки JSON (по одной команде на строку), см. control.py.
"""

import struct
import time

import cv2
import numpy as np

MAGIC = b"SCF1"
HDR = struct.Struct("<4sIddBHHI")
HDR_SIZE = HDR.size

RAW, JPEG, PNG = 0, 1, 2
CODEC_NAME = {RAW: "raw", JPEG: "jpeg", PNG: "png"}
CODEC_ID = {v: k for k, v in CODEC_NAME.items()}

clock = time.perf_counter


class Frame:
    """Кадр в пути: заголовок плюс закодированная нагрузка."""

    __slots__ = ("seq", "t_cap", "t_send", "codec", "w", "h", "payload", "t_recv")

    def __init__(self, seq, t_cap, codec, w, h, payload, t_send=0.0, t_recv=0.0):
        self.seq, self.t_cap, self.t_send = seq, t_cap, t_send
        self.codec, self.w, self.h = codec, w, h
        self.payload = payload
        self.t_recv = t_recv

    def pack(self):
        """Заголовок с отметкой отправки и нагрузка — двумя кусками, чтобы
        не копировать мегабайты сырого кадра ради склейки."""
        self.t_send = clock()
        hdr = HDR.pack(MAGIC, self.seq & 0xFFFFFFFF, self.t_cap, self.t_send,
                       self.codec, self.w, self.h, len(self.payload))
        return hdr, self.payload

    def to_bytes(self):
        hdr, body = self.pack()
        return hdr + bytes(body)

    @classmethod
    def parse_header(cls, buf):
        magic, seq, t_cap, t_send, codec, w, h, n = HDR.unpack_from(buf, 0)
        if magic != MAGIC:
            raise ValueError("не кадр: неверная сигнатура")
        return cls(seq, t_cap, codec, w, h, None, t_send), n

    @classmethod
    def from_bytes(cls, buf):
        f, n = cls.parse_header(buf)
        f.payload = memoryview(buf)[HDR_SIZE:HDR_SIZE + n]
        f.t_recv = clock()
        return f

    def image(self):
        return decode(self.codec, self.payload, self.w, self.h)

    @property
    def latency_ms(self):
        return (self.t_recv - self.t_send) * 1000.0

    @property
    def age_ms(self):
        """Сколько прошло с захвата — вместе с кодированием и транспортом."""
        return (self.t_recv - self.t_cap) * 1000.0


def encode(img, codec, quality=80):
    """BGR → байты нагрузки. Для JPEG quality — качество, для PNG — сжатие 0–9."""
    if codec == RAW:
        return np.ascontiguousarray(img).tobytes()
    if codec == JPEG:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    elif codec == PNG:
        # PNG в реальном времени имеет смысл только с быстрым сжатием
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    else:
        raise ValueError(f"неизвестный кодек {codec}")
    if not ok:
        raise RuntimeError("кодирование кадра не удалось")
    return buf.tobytes()


def decode(codec, payload, w, h):
    if codec == RAW:
        a = np.frombuffer(payload, np.uint8)
        if a.size != w * h * 3:
            return None
        return a.reshape(h, w, 3)
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


def make_frame(seq, img=None, codec=JPEG, quality=80, encoded=None, size=None, t_cap=None):
    """Кадр из массива BGR либо из уже закодированных байтов (захват, который
    сам отдаёт JPEG — CDP screencast, canvas.toDataURL, — перекодировать незачем)."""
    t_cap = clock() if t_cap is None else t_cap
    if encoded is not None:
        w, h = size
        return Frame(seq, t_cap, codec, w, h, encoded)
    h, w = img.shape[:2]
    return Frame(seq, t_cap, codec, w, h, encode(img, codec, quality))
