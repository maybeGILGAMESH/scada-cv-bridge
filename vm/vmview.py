# -*- coding: utf-8 -*-
"""Окно с экраном ВМ — то, что приходит по «HDMI» (MJPEG от vmhdmi.py).

Показывает ровно тот кадр, который получает распознавание, и передаёт в ВМ
мышь и клавиатуру через обратный канал — как KVM.

    python vmview.py                 # окно 1280×720
    python vmview.py --fullscreen    # сразу на весь экран
    python vmview.py --view-only     # только смотреть

F11 — полный экран вкл/выкл, Esc — выйти из полного экрана (в ВМ не уходит,
чтобы случайно не закрыть щит), закрыть окно — крестиком или Ctrl+C.
"""

import argparse
import json
import queue
import socket
import threading
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

WIN = "SCADA VM"


class Stream(threading.Thread):
    """MJPEG: ищем маркеры начала и конца JPEG, держим только последний кадр."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.frame = None
        self.n = 0
        self.error = None

    def run(self):
        while True:
            try:
                r = urlopen(Request(self.url), timeout=5)
                buf = b""
                self.error = None
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
                    j = buf.rfind(b"\xff\xd9")
                    if j < 0:
                        continue
                    i = buf.rfind(b"\xff\xd8", 0, j)
                    if i >= 0:
                        img = cv2.imdecode(np.frombuffer(buf[i:j + 2], np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            self.frame = img
                            self.n += 1
                    buf = buf[j + 2:]
            except OSError as e:
                self.error = str(e)
                time.sleep(1)


class Hid(threading.Thread):
    """Очередь событий ввода → обратный канал vmhdmi. Движения мыши
    схлопываются: важна последняя точка, а не путь."""

    def __init__(self, addr):
        super().__init__(daemon=True)
        host, _, port = addr.rpartition(":")
        self.addr = (host or "127.0.0.1", int(port))
        self.q = queue.Queue()
        self.f = None

    def send(self, **m):
        self.q.put(m)

    def _call(self, m):
        if self.f is None:
            s = socket.create_connection(self.addr, timeout=3)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.f = s.makefile("rwb")
        self.f.write((json.dumps(m) + "\n").encode())
        self.f.flush()
        self.f.readline()

    def run(self):
        while True:
            m = self.q.get()
            if m.get("action") == "move":
                while not self.q.empty():
                    nxt = self.q.queue[0]
                    if nxt.get("action") != "move":
                        break
                    m = self.q.get()
            try:
                self._call(m)
            except OSError:
                self.f = None


# waitKeyEx отдаёт X11 keysym для служебных клавиш (F11 = 0xFFC8) и код символа для прочих
SPECIAL = {0xFF1B: "Escape", 0xFF0D: "Enter", 0xFF8D: "Enter", 0xFF09: "Tab", 0xFF08: "Backspace",
           0xFFFF: "Delete", 0xFF51: "ArrowLeft", 0xFF52: "ArrowUp", 0xFF53: "ArrowRight",
           0xFF54: "ArrowDown", 0xFF50: "Home", 0xFF57: "End", 0xFF55: "PageUp", 0xFF56: "PageDown",
           **{0xFFBE + i: f"F{i + 1}" for i in range(12)},
           27: "Escape", 13: "Enter", 10: "Enter", 9: "Tab", 8: "Backspace"}


def key_name(code):
    if code in SPECIAL:
        return SPECIAL[code]
    if 32 <= code < 127:
        return " " if code == 32 else chr(code).lower()
    return None


def main():
    ap = argparse.ArgumentParser(description="Окно с экраном ВМ")
    ap.add_argument("--url", default="http://127.0.0.1:8781/stream")
    ap.add_argument("--control", default="127.0.0.1:8790")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--view-only", action="store_true")
    ap.add_argument("--debug-keys", action="store_true", help="печатать коды клавиш")
    a = ap.parse_args()

    st = Stream(a.url)
    st.start()
    hid = None if a.view_only else Hid(a.control)
    if hid:
        hid.start()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    full = a.fullscreen

    def set_full(on):
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

    set_full(full)

    def on_mouse(ev, x, y, flags, _):
        if not hid:
            return
        if ev == cv2.EVENT_MOUSEMOVE:
            hid.send(op="mouse", action="move", x=x, y=y)
        elif ev == cv2.EVENT_LBUTTONDOWN:
            hid.send(op="mouse", action="down", x=x, y=y)
        elif ev == cv2.EVENT_LBUTTONUP:
            hid.send(op="mouse", action="up", x=x, y=y)
        elif ev == cv2.EVENT_MOUSEWHEEL:
            hid.send(op="wheel", x=x, y=y, dy=-1 if flags > 0 else 1)

    cv2.setMouseCallback(WIN, on_mouse)
    blank = np.zeros((1080, 1920, 3), np.uint8)
    cv2.putText(blank, "NO SIGNAL", (800, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (90, 90, 90), 3)
    shown = -1
    while True:
        if st.n != shown:
            cv2.imshow(WIN, st.frame if st.frame is not None else blank)
            shown = st.n
        k = cv2.waitKeyEx(10)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k < 0:
            continue
        if a.debug_keys:
            print(f"клавиша {k:#x} → {key_name(k)}", flush=True)
        name = key_name(k)
        if name == "F11":
            full = not full
            set_full(full)
        elif name == "Escape":
            if full:
                full = False
                set_full(False)
        elif name and hid:
            hid.send(op="key", key=name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
