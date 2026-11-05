# -*- coding: utf-8 -*-
"""Программные «HDMI + KVM» для ВМ со щитом.

ВМ работает в фоне без окна; её экран QEMU отдаёт по VNC. Эта служба —
то же, что связка HDMI → PiKVM в схеме стенда:

    ВМ (Chromium + щит) ──VNC──▶ vmhdmi ──MJPEG http://127.0.0.1:8781/stream──▶ receiver.py / live/run.py / vmview.py
                         ◀─мышь/клавиатура── vmhdmi ◀── обратный канал tcp://127.0.0.1:8790 (JSON-строки)

* видео: кадр снимается с экрана ВМ снаружи (как карта захвата с HDMI),
  отдаётся MJPEG-потоком в формате PiKVM — его читают live/run.py и
  bridge/receiver.py --source;
* ввод: обратный канал совместим с bridge/emulator.py (hello, ping, click,
  key, scenario, truth), клик и клавиша уходят в ВМ как USB-мышь и
  клавиатура — так же, как HID у KVM;
* тестовый канал (scenario, truth) идёт в Chromium ВМ через DevTools по
  ssh-туннелю — только для проверок, распознавание им не пользуется.

    python vmhdmi.py                    # VNC 127.0.0.1:5900 → :8781 (видео) и :8790 (управление)
    python vmhdmi.py --fps 15 --quality 70

Адреса:  /stream — MJPEG,  /snapshot — один JPEG,  /snapshot.png — без потерь,
         /stats — счётчики JSON,  / — страница.
"""

import argparse
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
clock = time.perf_counter


# ───────────────────────────────────────────────────────────── VNC (RFB 3.8)

class Vnc:
    """Минимальный клиент RFB: сырые прямоугольники, 32 бит BGRX.

    Кодировка Raw выбрана намеренно: сервер QEMU на той же машине, сжимать
    незачем, а разбор Raw — это одна запись в массив без декодера."""

    RAW, DESKTOP_SIZE = 0, -223

    def __init__(self, host="127.0.0.1", port=5900):
        self.addr = (host, port)
        self.wlock = threading.Lock()
        self.fb_lock = threading.Lock()
        self.cond = threading.Condition()
        self.seq = 0               # номер изменения экрана
        self.t_change = 0.0
        self.updates = 0
        self.bytes = 0
        self.buttons = 0
        self.connected = False
        self._connect()

    def _recv(self, n):
        buf = bytearray(n)
        self._recv_into(memoryview(buf))
        return bytes(buf)

    def _recv_into(self, mv):
        got = 0
        while got < len(mv):
            k = self.s.recv_into(mv[got:])
            if not k:
                raise ConnectionError("VNC: соединение закрыто")
            got += k
        self.bytes += got

    def _connect(self):
        s = socket.create_connection(self.addr, timeout=10)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
        self.s = s
        if not self._recv(12).startswith(b"RFB 003."):
            raise ConnectionError("VNC: не RFB")
        s.sendall(b"RFB 003.008\n")
        types = self._recv(self._recv(1)[0])
        if 1 not in types:
            raise ConnectionError(f"VNC: сервер требует пароль (типы {list(types)})")
        s.sendall(b"\x01")
        if struct.unpack(">I", self._recv(4))[0] != 0:
            raise ConnectionError("VNC: отказ в подключении")
        s.sendall(b"\x01")                                  # shared: не выгонять других клиентов
        w, h = struct.unpack(">HH", self._recv(4))
        self._recv(16)
        self.name = self._recv(struct.unpack(">I", self._recv(4))[0]).decode("utf-8", "replace")
        # 32 бит, глубина 24, little-endian, true colour, R<<16 G<<8 B → в памяти B,G,R,X
        pf = struct.pack(">BBBBHHHBBB3x", 32, 24, 0, 1, 255, 255, 255, 16, 8, 0)
        s.sendall(struct.pack(">B3x", 0) + pf)
        encs = [self.RAW, self.DESKTOP_SIZE]
        s.sendall(struct.pack(">BxH", 2, len(encs)) + struct.pack(f">{len(encs)}i", *encs))
        s.settimeout(None)
        self._resize(w, h)
        self.connected = True

    def _resize(self, w, h):
        with self.fb_lock:
            self.w, self.h = w, h
            self.fb = np.zeros((h, w, 4), np.uint8)

    def _request(self, incremental):
        with self.wlock:
            self.s.sendall(struct.pack(">BBHHHH", 3, incremental, 0, 0, self.w, self.h))

    def run(self):
        """Цикл приёма: следующий запрос уходит сразу после ответа — частоту
        задаёт сам QEMU (он шлёт только изменившиеся области экрана)."""
        self._request(0)
        while True:
            t = self._recv(1)[0]
            if t == 0:                                       # FramebufferUpdate
                self._recv(1)
                n = struct.unpack(">H", self._recv(2))[0]
                full = False
                for _ in range(n):
                    x, y, w, h, enc = struct.unpack(">HHHHi", self._recv(12))
                    if enc == self.RAW:
                        tile = np.empty((h, w, 4), np.uint8)
                        self._recv_into(memoryview(tile).cast("B"))
                        with self.fb_lock:
                            self.fb[y:y + h, x:x + w] = tile
                    elif enc == self.DESKTOP_SIZE:
                        self._resize(w, h)
                        full = True
                    else:
                        raise ConnectionError(f"VNC: неожиданная кодировка {enc}")
                self.updates += 1
                with self.cond:
                    self.seq += 1
                    self.t_change = clock()
                    self.cond.notify_all()
                self._request(0 if full else 1)
            elif t == 1:                                     # SetColourMapEntries
                self._recv(1)
                _, k = struct.unpack(">HH", self._recv(4))
                self._recv(6 * k)
            elif t == 2:                                     # Bell
                pass
            elif t == 3:                                     # ServerCutText
                self._recv(3)
                self._recv(struct.unpack(">I", self._recv(4))[0])
            else:
                raise ConnectionError(f"VNC: неизвестное сообщение {t}")

    def frame(self):
        """Копия экрана BGR и номер изменения."""
        with self.fb_lock:
            return self.fb[:, :, :3].copy(), self.seq

    # ── ввод: то, что у KVM делает USB HID ──
    def pointer(self, x, y, buttons=None):
        if buttons is not None:
            self.buttons = buttons
        x = int(min(max(x, 0), self.w - 1))
        y = int(min(max(y, 0), self.h - 1))
        with self.wlock:
            self.s.sendall(struct.pack(">BBHH", 5, self.buttons, x, y))

    def key(self, keysym, down):
        with self.wlock:
            self.s.sendall(struct.pack(">BBxxI", 4, 1 if down else 0, keysym))

    def click(self, x, y, hold=0.06):
        # сначала довести курсор: если USB-планшет гостя спал, первое событие
        # уходит на пробуждение и теряется — нажатие пришло бы в старую точку
        self.pointer(x + 1, y, 0)
        time.sleep(0.03)
        self.pointer(x, y, 0)
        time.sleep(0.03)
        self.pointer(x, y, 1)
        time.sleep(hold)
        self.pointer(x, y, 0)

    def tap(self, keysym, hold=0.03):
        self.key(keysym, True)
        time.sleep(hold)
        self.key(keysym, False)


# Имена клавиш как у Playwright (bridge/control.py шлёт их же) → X11 keysym
KEYSYM = {
    "Escape": 0xFF1B, "Enter": 0xFF0D, "Tab": 0xFF09, "Backspace": 0xFF08, "Delete": 0xFFFF,
    "Space": 0x20, " ": 0x20, "ArrowLeft": 0xFF51, "ArrowUp": 0xFF52, "ArrowRight": 0xFF53,
    "ArrowDown": 0xFF54, "Home": 0xFF50, "End": 0xFF57, "PageUp": 0xFF55, "PageDown": 0xFF56,
    "Shift": 0xFFE1, "Control": 0xFFE3, "Alt": 0xFFE9,
    **{f"F{i}": 0xFFBD + i for i in range(1, 13)},
}


def keysym_of(name):
    name = str(name)
    if name in KEYSYM:
        return KEYSYM[name]
    if name.startswith("Key") and len(name) == 4:         # KeyB → b
        name = name[3].lower()
    elif name.startswith("Digit") and len(name) == 6:
        name = name[5]
    if len(name) == 1:
        c = ord(name)
        return c if c < 0x100 else 0x01000000 + c          # кириллица — Unicode keysym
    raise ValueError(f"неизвестная клавиша {name!r}")


# ───────────────────────────────────────────────────────────── кодирование MJPEG

class Encoder:
    """Берёт свежий экран ВМ, жмёт в JPEG не чаще --fps. Экран не менялся —
    раз в секунду повторяет последний кадр, чтобы приёмник не считал поток мёртвым."""

    def __init__(self, vnc, fps, quality):
        self.vnc, self.period, self.quality = vnc, 1.0 / fps, quality
        self.cond = threading.Condition()
        self.jpeg, self.n, self.t_cap = b"", 0, 0.0
        self.enc_ms = []
        self.sent_frames = 0

    def run(self):
        last_seq, last_t = -1, 0.0
        while True:
            with self.vnc.cond:
                self.vnc.cond.wait_for(lambda: self.vnc.seq != last_seq, timeout=1.0)
            wait = last_t + self.period - clock()
            if wait > 0:
                time.sleep(wait)
            t0 = clock()
            img, last_seq = self.vnc.frame()
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            t1 = clock()
            if not ok:
                continue
            self.enc_ms.append((t1 - t0) * 1000)
            del self.enc_ms[:-200]
            last_t = t0
            with self.cond:
                self.jpeg, self.t_cap = buf.tobytes(), t0
                self.n += 1
                self.cond.notify_all()

    def wait(self, n_seen, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.n != n_seen, timeout=timeout)
            return self.jpeg, self.n, self.t_cap


# ───────────────────────────────────────────────────────────── DevTools (тестовый канал)

# Щиты разные: у нового (neft-hmi-cv) — cmdRepair/syncUI/setLayout, у старого
# (neft-scada) — repair/sync и компоновок нет. Отсутствующее — ошибка с понятным текстом.
_SYNC = "(typeof syncUI === 'function' ? syncUI : sync)()"
SCENARIO_JS = {       # тот же список, что в bridge/emulator.py: только эти действия, без произвольного JS
    "fault": ("k => inject(k)", ("clog", "skin", "leak", "iso", "water")),
    "layout": ("id => { if (typeof setLayout !== 'function') throw new Error('у этого щита нет компоновок'); setLayout(id); }", None),
    "speed": ("v => { speed = +v; }", None),
    "pause": (f"v => {{ paused = !!v; {_SYNC}; }}", None),
    "start": ("() => cmdStart()", None),
    "stop": ("() => cmdStop()", None),
    "repair": ("() => (typeof cmdRepair === 'function' ? cmdRepair : repair)()", None),
    "freq": (f"v => {{ ui.f = Math.max(35, Math.min(65, +v)); {_SYNC}; }}", None),
    # генератор случайных щитов (random-scada.html): новая станция по зерну
    "regen": ("s => { if (typeof regen !== 'function') throw new Error('это не генератор случайных щитов'); regen(+s || Math.floor(Math.random() * 1e6)); }", None),
}
TRUTH_JS = """() => ({run: S.run, tripped: S.tripped, trip_reason: S.tripReason,
  f: ui.f, choke: ui.choke, auto: ui.auto, apv: ui.apv, inh: ui.inh,
  layout: LAY, speed, paused, t_model: S.t})"""


class Cdp:
    """Runtime.evaluate в странице щита внутри ВМ. Порт 9222 Chromium слушает
    только на localhost ВМ, поэтому до него — ssh-туннель."""

    def __init__(self, ssh_port=2222, local_port=9222):
        self.ssh_port, self.port = ssh_port, local_port
        self.tunnel = None
        self.ws = None
        self.lock = threading.Lock()
        self.id = 0

    def _tunnel(self):
        if self.tunnel and self.tunnel.poll() is None:
            return
        self.tunnel = subprocess.Popen(
            ["ssh", "-N", "-i", str(HERE / "ssh_key"), "-p", str(self.ssh_port),
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "LogLevel=ERROR", "-o", "ExitOnForwardFailure=yes",
             "-L", f"127.0.0.1:{self.port}:127.0.0.1:9222", "scada@127.0.0.1"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise ConnectionError("ssh-туннель к DevTools ВМ не поднялся")

    def _connect(self):
        from websockets.sync.client import connect
        self._tunnel()
        tabs = json.load(urlopen(f"http://127.0.0.1:{self.port}/json", timeout=3))
        page = next(t for t in tabs if t.get("type") == "page")
        self.ws = connect(page["webSocketDebuggerUrl"], max_size=None, open_timeout=5)

    def evaluate(self, js, *arg):
        expr = f"({js})({json.dumps(arg[0])})" if arg else f"({js})()"
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.ws is None:
                        self._connect()
                    self.id += 1
                    self.ws.send(json.dumps({"id": self.id, "method": "Runtime.evaluate", "params": {
                        "expression": expr, "returnByValue": True, "awaitPromise": True}}))
                    while True:
                        r = json.loads(self.ws.recv(timeout=5))
                        if r.get("id") == self.id:
                            break
                    res = r.get("result", {})
                    if "exceptionDetails" in res:
                        ex = res["exceptionDetails"]
                        desc = (ex.get("exception") or {}).get("description") or ex.get("text") or "ошибка JS"
                        raise RuntimeError(desc.splitlines()[0].replace("Error: ", ""))
                    return res.get("result", {}).get("value")
                except (OSError, ConnectionError, StopIteration, TimeoutError) as e:
                    self.ws = None              # страницу перезагрузили, щит перезапустили
                    if attempt:
                        raise ConnectionError(f"DevTools ВМ: {e}") from e
                except Exception as e:
                    if type(e).__module__.startswith("websockets") and not attempt:
                        self.ws = None
                        continue
                    raise

    def close(self):
        if self.tunnel:
            self.tunnel.terminate()


# ───────────────────────────────────────────────────────────── обратный канал

class Control:
    """JSON-строки по TCP — тот же протокол, что у bridge/emulator.py."""

    def __init__(self, vnc, cdp, stream_url):
        self.vnc, self.cdp, self.stream_url = vnc, cdp, stream_url
        self.cmds = self.clicks = 0

    def serve(self, host, port):
        srv = socket.create_server((host, port), reuse_port=False)
        while True:
            c, _ = srv.accept()
            threading.Thread(target=self.client, args=(c,), daemon=True).start()

    def client(self, c):
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = c.makefile("rwb")
        try:
            for line in f:
                msg = {}
                try:
                    msg = json.loads(line)
                    res = self.dispatch(msg)
                    res.setdefault("ok", True)
                except Exception as e:
                    res = {"ok": False, "error": str(e)}
                res["id"], res["op"] = msg.get("id"), msg.get("op")
                f.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()
        except OSError:
            pass
        finally:
            c.close()

    def dispatch(self, m):
        op = m.get("op")
        self.cmds += 1
        v = self.vnc
        if op == "hello":
            return {"capture": f"VNC QEMU «{v.name}»", "frame_size": [v.w, v.h], "region": None,
                    "transport": f"mjpeg {self.stream_url}", "strip": 0, "os_level": True}
        if op == "ping":
            return {"t": m.get("t"), "t_emul": clock()}
        if op == "click":
            x, y = float(m["x"]), float(m["y"])
            v.click(x, y)
            self.clicks += 1
            return {"page": [round(x, 1), round(y, 1)]}
        if op == "mouse":          # для окна просмотра: перетаскивание ползунков
            mask = {"down": 1, "up": 0}.get(m.get("action"))
            v.pointer(float(m["x"]), float(m["y"]), mask)
            return {}
        if op == "wheel":
            x, y, b = float(m["x"]), float(m["y"]), 8 if m.get("dy", 0) < 0 else 16
            v.pointer(x, y, v.buttons | b)
            v.pointer(x, y, v.buttons & ~b)
            return {}
        if op == "key":
            v.tap(keysym_of(m["key"]))
            return {}
        if op == "info":
            return {"shown": False}            # полосы подсказок на экране ВМ нет
        if op == "scenario":
            act = m.get("action")
            if act not in SCENARIO_JS:
                raise ValueError(f"нет такого действия: {act}")
            js, allowed = SCENARIO_JS[act]
            arg = m.get("arg")
            if allowed is not None and arg not in allowed:
                raise ValueError(f"недопустимый аргумент {arg!r}")
            if js.startswith("()"):
                self.cdp.evaluate(js)
            else:
                self.cdp.evaluate(js, arg)
            return {"done": act}
        if op == "truth":
            return {"truth": self.cdp.evaluate(TRUTH_JS)}
        raise ValueError(f"неизвестная команда {op}")


# ───────────────────────────────────────────────────────────── HTTP

PAGE = """<!doctype html><meta charset="utf-8"><title>Экран ВМ</title>
<style>body{margin:0;background:#000;color:#9fb6c9;font:12px monospace}
img{display:block;width:100vw;height:calc(100vh - 22px);object-fit:contain}
#s{height:22px;line-height:22px;padding:0 8px}</style>
<img src="/stream"><div id="s"></div>
<script>setInterval(async()=>{try{const s=await (await fetch('/stats')).json();
document.getElementById('s').textContent=`VNC ${s.vnc_updates_per_s} обн/с · ${s.vnc_mb_per_s} МБ/с · MJPEG ${s.jpeg_fps} к/с · `
+`${s.jpeg_kb} КБ · кодирование ${s.encode_ms_p50} мс · клиентов ${s.clients}`}catch(e){}},1000)</script>"""


def make_handler(vnc, enc, stats):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/stream", "/streamer/stream"):
                return self.stream()
            if p in ("/snapshot", "/snapshot.jpg"):
                jpeg, _, _ = enc.wait(-1, 0)
                return self._send(200, "image/jpeg", jpeg)
            if p == "/snapshot.png":          # без потерь — для сверки распознавания
                ok, buf = cv2.imencode(".png", vnc.frame()[0], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                return self._send(200, "image/png", buf.tobytes())
            if p == "/stats":
                return self._send(200, "application/json", json.dumps(stats()).encode())
            if p == "/":
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            self._send(404, "text/plain", b"not found")

        def stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            enc.clients += 1
            seen = -1
            try:
                while True:
                    jpeg, seen, t_cap = enc.wait(seen)
                    if not jpeg:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\nX-Timestamp: "
                                     + f"{t_cap:.6f}".encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()
                    enc.sent_frames += 1
            except OSError:
                pass
            finally:
                enc.clients -= 1
    return H


def main():
    ap = argparse.ArgumentParser(description="Экран ВМ → MJPEG, мышь/клавиатура ← обратный канал")
    ap.add_argument("--vnc", default="127.0.0.1:5900")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8781, help="порт MJPEG")
    ap.add_argument("--control-port", type=int, default=8790, help="порт обратного канала")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--ssh-port", type=int, default=2222, help="ssh ВМ (туннель к DevTools)")
    ap.add_argument("--cdp-port", type=int, default=9222, help="локальный порт туннеля к DevTools ВМ")
    a = ap.parse_args()

    vh, _, vp = a.vnc.rpartition(":")
    for attempt in range(60):
        try:
            vnc = Vnc(vh or "127.0.0.1", int(vp))
            break
        except OSError as e:
            if attempt == 0:
                print(f"жду VNC ВМ {a.vnc}: {e}", flush=True)
            time.sleep(1)
    else:
        sys.exit("VNC ВМ недоступен — ВМ запущена? (vm/run-vm.sh)")
    enc = Encoder(vnc, a.fps, a.quality)
    enc.clients = 0
    cdp = Cdp(ssh_port=a.ssh_port, local_port=a.cdp_port)
    url = f"http://{a.host}:{a.port}/stream"
    ctl = Control(vnc, cdp, url)

    prev = {"t": clock(), "u": 0, "b": 0, "n": 0, "s": 0}
    rates = {}

    def stats():
        now = clock()
        dt = now - prev["t"]
        if dt >= 0.5:
            rates.update(vnc_updates_per_s=round((vnc.updates - prev["u"]) / dt, 1),
                         vnc_mb_per_s=round((vnc.bytes - prev["b"]) / dt / 1e6, 1),
                         jpeg_fps=round((enc.n - prev["n"]) / dt, 1),
                         sent_fps=round((enc.sent_frames - prev["s"]) / dt, 1))
            prev.update(t=now, u=vnc.updates, b=vnc.bytes, n=enc.n, s=enc.sent_frames)
        e = sorted(enc.enc_ms) or [0]
        return dict(rates, screen=[vnc.w, vnc.h], jpeg_kb=len(enc.jpeg) // 1024,
                    encode_ms_p50=round(e[len(e) // 2], 1), encode_ms_p95=round(e[int(len(e) * .95)], 1),
                    clients=enc.clients, quality=a.quality, fps_cap=a.fps,
                    commands=ctl.cmds, clicks=ctl.clicks, vnc_total_updates=vnc.updates)

    threading.Thread(target=vnc.run, daemon=True, name="vnc").start()
    threading.Thread(target=enc.run, daemon=True, name="jpeg").start()
    threading.Thread(target=ctl.serve, args=(a.host, a.control_port), daemon=True).start()
    http = ThreadingHTTPServer((a.host, a.port), make_handler(vnc, enc, stats))
    http.daemon_threads = True
    print(f"Экран ВМ : VNC {a.vnc} «{vnc.name}» {vnc.w}×{vnc.h}")
    print(f"Видео    : {url}  (снимок /snapshot, счётчики /stats, страница /)")
    print(f"Управление: tcp://{a.host}:{a.control_port} (JSON-строки, как у bridge/emulator.py)")
    print(f"JPEG     : до {a.fps:g} к/с, качество {a.quality}", flush=True)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cdp.close()


if __name__ == "__main__":
    main()
