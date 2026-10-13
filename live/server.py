# -*- coding: utf-8 -*-
"""Локальный сайт, на который уходят распознанные показания.

Отдельный от щита сайт: щит живёт на другом компьютере и сюда приходит
только картинкой. Всё, что показывает этот сайт, получено из пикселей
трансляции.

Только стандартная библиотека: HTTP + Server-Sent Events. WebSocket здесь
не нужен — поток односторонний, а SSE сам переподключается при обрыве.

    /                текущая страница
    /events          поток обновлений (text/event-stream)
    /api/state       последнее состояние одним JSON
    /frame.jpg       последний нормализованный кадр
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class State:
    """Последнее состояние плюс уведомление подписчиков."""

    def __init__(self):
        self._lock = threading.Condition()
        self._data = {"status": "ожидание кадра"}
        self._frame = b""
        self._seq = 0

    def publish(self, data, frame_jpeg=None):
        with self._lock:
            self._data = data
            if frame_jpeg is not None:
                self._frame = frame_jpeg
            self._seq += 1
            self._lock.notify_all()

    def snapshot(self):
        with self._lock:
            return self._seq, self._data

    def frame(self):
        with self._lock:
            return self._frame

    def wait(self, seq, timeout=15.0):
        """Ждёт состояние новее seq. Возвращает (seq, data) либо (seq, None)
        по таймауту — тогда наружу уходит пустой комментарий, чтобы соединение
        не закрыл прокси."""
        with self._lock:
            if self._seq > seq:
                return self._seq, self._data
            self._lock.wait(timeout)
            if self._seq > seq:
                return self._seq, self._data
            return self._seq, None


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "scada-cv"

        def log_message(self, *a):        # тишина: лог пишет сам цикл распознавания
            pass

        def _head(self, code, ctype, length=None, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body = (HERE / "dashboard.html").read_bytes()
                self._head(200, "text/html; charset=utf-8", len(body))
                self.wfile.write(body)
            elif path == "/api/state":
                _, data = state.snapshot()
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._head(200, "application/json; charset=utf-8", len(body))
                self.wfile.write(body)
            elif path == "/frame.jpg":
                body = state.frame()
                self._head(200, "image/jpeg", len(body))
                self.wfile.write(body)
            elif path == "/events":
                self._events()
            else:
                self._head(404, "text/plain; charset=utf-8", 0)

        def _events(self):
            self._head(200, "text/event-stream; charset=utf-8",
                       extra={"Connection": "keep-alive",
                              "X-Accel-Buffering": "no"})
            seq, data = state.snapshot()
            try:
                self._send_event(data)
                while True:
                    seq, data = state.wait(seq)
                    if data is None:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        self._send_event(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _send_event(self, data):
            payload = json.dumps(data, ensure_ascii=False)
            self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")

    return Handler


def serve(state: State, host="127.0.0.1", port=8770):
    srv = ThreadingHTTPServer((host, port), make_handler(state))
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv
