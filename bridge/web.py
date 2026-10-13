# -*- coding: utf-8 -*-
"""Сайт приёмника: показания, агрегаты, события и пульт.

Только стандартная библиотека. Показания идут вниз потоком Server-Sent
Events (пять раз в секунду), команды — вверх обычными POST: двусторонний
WebSocket тут не нужен, команды редки, а SSE сам переподключается.

    GET  /                  страница
    GET  /events            поток состояния (text/event-stream)
    GET  /api/state         последнее состояние одним JSON
    GET  /frame.jpg         последний кадр после нормализации
    GET  /api/history?tag=  история тега [[t, v], …]
    GET  /api/history.csv   вся история (Excel, «;», UTF-8 с BOM)
    POST /api/command       {"cmd": "start", "args": {...}}   → команда через HID
    POST /api/rule          {"id": "overheat", "enabled": true}
    POST /api/assist        {"on": true}                      — подсказки на экран щита
    POST /api/scenario      {"action": "fault", "arg": "clog"} — тестовый канал эмулятора

Сайт слушает 127.0.0.1. Если открыть его в сеть (--web-host 0.0.0.0), пульт
станет доступен всем в этой сети — авторизации здесь нет.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent


class Web:
    def __init__(self, rx, host, port):
        self.rx, self.host, self.port = rx, host, port

    def start(self):
        srv = ThreadingHTTPServer((self.host, self.port), make_handler(self.rx))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def make_handler(rx):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "scada-bridge"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            p = u.path
            if p in ("/", "/index.html"):
                self._send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif p == "/api/state":
                self._send(200, rx.state)
            elif p == "/frame.jpg":
                self._send(200, rx.thumb, "image/jpeg")
            elif p == "/api/history":
                tag = parse_qs(u.query).get("tag", [""])[0]
                self._send(200, rx.agg.history(tag))
            elif p == "/api/history.csv":
                self._send(200, rx.agg.history_csv(), "text/csv; charset=utf-8",
                           {"Content-Disposition": "attachment; filename=scada_history.csv"})
            elif p == "/events":
                self._events()
            else:
                self._send(404, {"error": "нет такого адреса"})

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_POST(self):
            p = urlparse(self.path).path
            try:
                d = self._body()
                if p == "/api/command":
                    if not rx.ctl:
                        raise RuntimeError("обратного канала нет — приёмник запущен только на чтение")
                    job = rx.ctl.submit(d.get("cmd"), d.get("args") or {}, source="оператор (сайт)")
                    self._send(200, {"ok": True, "job": job})
                elif p == "/api/rule":
                    ok = rx.auto.set_enabled(d.get("id"), d.get("enabled")) if rx.auto else False
                    self._send(200, {"ok": ok})
                elif p == "/api/assist":
                    rx.assist_on = bool(d.get("on"))
                    self._send(200, {"ok": True, "on": rx.assist_on})
                elif p == "/api/scenario":
                    if not rx.hid:
                        raise RuntimeError("обратного канала нет")
                    res = rx.hid.scenario(d.get("action"), d.get("arg"))
                    rx.agg.event(time.time(), "info",
                                 f"Тестовый канал: {d.get('action')} {d.get('arg') or ''}".strip(),
                                 "scenario")
                    self._send(200, {"ok": True, "res": res})
                else:
                    self._send(404, {"error": "нет такого адреса"})
            except Exception as e:
                self._send(400, {"ok": False, "error": str(e)})

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last = None
            idle = 0
            try:
                while True:
                    st = rx.state
                    ts = st.get("ts")
                    if ts != last:
                        last = ts
                        idle = 0
                        payload = json.dumps(st, ensure_ascii=False)
                        self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                        self.wfile.flush()
                    else:
                        idle += 1
                        if idle % 75 == 0:            # раз в 15 с — чтобы прокси не рвал
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return H
