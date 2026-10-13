# -*- coding: utf-8 -*-
"""Источники кадров для распознавания щита.

Щит живёт на другом компьютере: картинка уходит через KVM (Pi Zero 2)
и приходит сюда видеопотоком. Отсюда три рабочих источника и один
отладочный:

    --source http://pikvm.local/streamer/stream   MJPEG-поток KVM
    --source snapshot:http://pikvm.local/api/...   периодический снимок JPEG
    --source 0                                    USB-устройство захвата (UVC)
    --source кадр.png | папка/                    файл или папка — для отладки

Ни один из них ничего не знает про модель щита: наружу отдаётся только
кадр в виде массива BGR, ровно то, что видно на трансляции.
"""

import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request

import cv2
import numpy as np

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


class Source:
    """Общий интерфейс: read() отдаёт кадр BGR либо None."""

    name = "источник"

    def read(self):
        raise NotImplementedError

    def close(self):
        pass


class MjpegSource(Source):
    """Разбор multipart/x-mixed-replace: так отдают поток PiKVM и mjpg-streamer.

    Границы частей не разбираем — ищем маркеры начала и конца JPEG. Это
    устойчивее: некоторые стримеры шлют boundary с разным регистром и
    произвольными заголовками, а SOI/EOI одинаковы у всех.
    """

    def __init__(self, url, timeout=10.0):
        self.url = url
        self.name = f"MJPEG {url}"
        self.timeout = timeout
        self.buf = b""
        self.stream = None
        self._open()

    def _open(self):
        req = Request(self.url, headers={"User-Agent": "scada-cv/1.0"})
        self.stream = urlopen(req, timeout=self.timeout)
        self.buf = b""

    def read(self):
        while True:
            i = self.buf.find(JPEG_SOI)
            j = self.buf.find(JPEG_EOI, i + 2) if i >= 0 else -1
            if i >= 0 and j > 0:
                jpg = self.buf[i:j + 2]
                self.buf = self.buf[j + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
                continue
            chunk = self.stream.read(65536)
            if not chunk:
                self._open()            # поток оборвался — переподключаемся
                return None
            self.buf += chunk
            if len(self.buf) > 8 << 20:  # мусор вместо потока
                self.buf = self.buf[-(1 << 20):]

    def close(self):
        try:
            self.stream.close()
        except Exception:
            pass


class SnapshotSource(Source):
    """Периодический GET одиночного JPEG — если стример отдаёт только снимок."""

    def __init__(self, url, timeout=5.0):
        self.url = url
        self.name = f"снимок {url}"
        self.timeout = timeout

    def read(self):
        req = Request(self.url, headers={"User-Agent": "scada-cv/1.0"})
        with urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class DeviceSource(Source):
    """USB-устройство захвата. Просим MJPG и полный кадр: многие карты
    иначе отдают 640x480, и щит после масштабирования уже нечитаем."""

    def __init__(self, index, width=1920, height=1080, fps=30):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(int(index), backend)
        if not self.cap.isOpened():
            sys.exit(f"Не открывается устройство захвата {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.name = f"устройство {index} ({w}x{h})"

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class FileSource(Source):
    """Кадр или папка кадров — чтобы отлаживать распознавание без железа."""

    def __init__(self, path, loop=True, delay=0.15):
        p = Path(path)
        self.files = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")) if p.is_dir() else [p]
        if not self.files:
            sys.exit(f"Нет кадров в {path}")
        self.name = f"файлы {path} ({len(self.files)} шт.)"
        self.i, self.loop, self.delay = 0, loop, delay

    def read(self):
        if self.i >= len(self.files):
            if not self.loop:
                return None
            self.i = 0
        f = self.files[self.i]
        self.i += 1
        if self.delay:
            time.sleep(self.delay)
        return cv2.imread(str(f), cv2.IMREAD_COLOR)


def open_source(spec, width=1920, height=1080):
    """Разбор значения --source."""
    s = str(spec)
    if s.startswith("snapshot:"):
        return SnapshotSource(s[len("snapshot:"):])
    if s.startswith("http://") or s.startswith("https://"):
        return MjpegSource(s)
    if s.isdigit():
        return DeviceSource(int(s), width, height)
    return FileSource(s)
