# -*- coding: utf-8 -*-
"""Сторона щита (ПК-1): поднимает эмулятор, отдаёт его картинку в «HDMI»-канал
и исполняет команды обратного канала.

    ПК-1: Chrome + neft-hmi-cv.html ──захват──▶ транспорт ──▶ ПК-2: receiver.py
            ▲                                                      │
            └──── клик мышью / клавиша / подсказки (JSON по TCP) ◀─┘

Картинка — только пиксели, как и с KVM. Обратный канал по умолчанию умеет то
же, что HID у KVM: подвести мышь и нажать, нажать клавишу. Куда нажимать,
решает распознавание по картинке; эмулятор лишь переводит координаты кадра в
координаты страницы.

Сверх HID есть ещё две вещи, которых у настоящего KVM нет:
  * полоса подсказок под щитом (--strip, по умолчанию 120 px): приёмник
    выводит туда то, что насчитал сам — агрегаты, прогнозы, события. Монитор
    становится 1920×1200, щит остаётся 1920×1080 сверху, поэтому распознавание
    полоса не задевает: калибровка вырезает щит по шапке;
  * тестовый канал (op=scenario / op=truth): впрыснуть отказ, сменить
    компоновку, ускорить время, подсмотреть истинное состояние модели. Нужен
    только для проверок — распознавание и управление им не пользуются.

Примеры:
    python bridge/emulator.py                                   # tcp:8780, screencast, управление :8790
    python bridge/emulator.py --transport ws --capture canvas --fps 10
    python bridge/emulator.py --transport mjpeg --port 8781     # поток как у PiKVM — читается и live/run.py
    python bridge/emulator.py --headed --capture screen --strip 0   # честный захват экрана монитора
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture import CAPTURES, make_capture, os_click   # noqa: E402
from protocol import CODEC_ID, JPEG, clock, make_frame  # noqa: E402
from transports import TRANSPORTS, make_sender          # noqa: E402

ROOT = HERE.parent
PANEL = ROOT / "neft-hmi-cv.html"

KIOSK_CSS = """
html,body{overflow:hidden;background:#000;margin:0}
#wrap{max-width:none;padding:0;margin:0}
#panelBox{border:0;border-radius:0;justify-content:flex-start}
#panel{max-width:none;max-height:none;width:1920px;height:1080px}
.bar,#prog,#note{display:none!important}
#cvStrip{position:fixed;left:0;top:1080px;width:1920px;height:%(h)dpx;box-sizing:border-box;
  background:#060a0d;border-top:2px solid #1d3a4f;color:#c8d8e4;display:flex;gap:14px;
  padding:10px 16px;font:13px/1.3 "Segoe UI",Roboto,Arial,sans-serif;overflow:hidden}
#cvStrip .hd{min-width:250px;border-right:1px solid #1b2836;padding-right:14px}
#cvStrip .hd b{display:block;font-size:11px;letter-spacing:.14em;color:#38bdf8}
#cvStrip .hd span{display:block;font-size:11px;color:#6b8296;margin-top:3px}
#cvStrip .kp{display:flex;gap:8px;flex-wrap:wrap;align-content:flex-start;flex:1}
#cvStrip .t{background:#0d151d;border:1px solid #1e2d3a;border-radius:4px;padding:5px 9px;min-width:132px}
#cvStrip .t i{display:block;font-style:normal;font-size:10px;letter-spacing:.06em;color:#6b8296}
#cvStrip .t em{font-style:normal;font:600 16px Consolas,monospace;color:#e6f0f7}
#cvStrip .t.warn{border-color:#f7b731}#cvStrip .t.warn em{color:#f7b731}
#cvStrip .t.alarm{border-color:#ff4757}#cvStrip .t.alarm em{color:#ff4757}
#cvStrip .ev{width:520px;border-left:1px solid #1b2836;padding-left:14px;font-size:11.5px;color:#9fb6c9}
#cvStrip .ev div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#cvStrip .ev .alarm{color:#ff6b78}#cvStrip .ev .warn{color:#f7b731}#cvStrip .ev .ok{color:#00e5a0}
"""

STRIP_JS = """
(() => {
  const el = document.createElement('div'); el.id = 'cvStrip';
  el.innerHTML = '<div class="hd"><b>CV-АССИСТЕНТ</b><span>ожидание приёмника…</span></div>'
               + '<div class="kp"></div><div class="ev"></div>';
  document.body.appendChild(el);
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  window.__cvStrip = d => {
    el.querySelector('.hd').innerHTML = '<b>' + esc(d.title || 'CV-АССИСТЕНТ') + '</b>'
      + (d.lines || []).map(x => '<span>' + esc(x) + '</span>').join('');
    el.querySelector('.kp').innerHTML = (d.items || []).map(t =>
      '<div class="t ' + esc(t.level || '') + '"><i>' + esc(t.label) + '</i><em>' + esc(t.value)
      + (t.unit ? ' <small>' + esc(t.unit) + '</small>' : '') + '</em></div>').join('');
    el.querySelector('.ev').innerHTML = (d.events || []).slice(0, 5).map(e =>
      '<div class="' + esc(e.level || '') + '">' + esc(e.text) + '</div>').join('');
  };
})();
"""

# Тестовый канал: только перечисленные действия, без произвольного JS.
SCENARIO_JS = {
    "fault": ("k => inject(k)", ("clog", "skin", "leak", "iso", "water")),
    "layout": ("id => setLayout(id)", None),
    "speed": ("v => { speed = +v; }", None),
    "pause": ("v => { paused = !!v; syncUI(); }", None),
    "start": ("() => cmdStart()", None),
    "stop": ("() => cmdStop()", None),
    "repair": ("() => cmdRepair()", None),
    "freq": ("v => { ui.f = Math.max(35, Math.min(65, +v)); syncUI(); }", None),
}

TRUTH_JS = """() => ({run: S.run, tripped: S.tripped, trip_reason: S.tripReason,
  f: ui.f, choke: ui.choke, auto: ui.auto, apv: ui.apv, inh: ui.inh,
  layout: LAY, speed, paused, t_model: S.t})"""


async def launch(pw, headed):
    last = None
    for ch in ("chrome", "msedge", None):
        try:
            args = ["--start-fullscreen"] if headed else []
            kw = {"channel": ch} if ch else {}
            return await pw.chromium.launch(headless=not headed, args=args, **kw)
        except Exception as e:          # нет такого браузера — пробуем следующий
            last = e
    raise RuntimeError(f"не удалось запустить браузер: {last}")


class Emulator:
    def __init__(self, a):
        self.a = a
        self.page = None
        self.capture = None
        self.sender = None
        self.clicks = 0
        self.cmds = 0
        self.t0 = time.time()

    async def setup(self, pw):
        a = self.a
        self.browser = await launch(pw, a.headed)
        if a.capture == "screen":
            ctx = await self.browser.new_context(no_viewport=True)
        else:
            ctx = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080 + a.strip}, device_scale_factor=1)
        self.page = await ctx.new_page()
        await self.page.goto(PANEL.resolve().as_uri())
        if a.capture == "screen":
            # на экране щит вписывается в окно — масштаб снимет калибровка приёмника
            await self.page.add_style_tag(content=(KIOSK_CSS % {"h": max(a.strip, 1)}).replace(
                "width:1920px;height:1080px", "width:100vw;height:auto;max-height:100vh"))
        else:
            await self.page.add_style_tag(content=KIOSK_CSS % {"h": max(a.strip, 1)})
        if a.strip:
            await self.page.evaluate(STRIP_JS)
        if a.speed:
            await self.page.evaluate("v => { speed = +v; }", a.speed)
        if a.layout:
            await self.page.evaluate("id => setLayout(id)", a.layout)
        if a.start_pump:
            await self.page.evaluate("() => cmdStart()")
        await self.page.wait_for_timeout(300)

        self.capture = make_capture(a.capture, self.page, fps=a.fps, quality=a.quality)
        await self.capture.start()
        kw = {"mtu": a.mtu} if a.transport == "udp" and a.mtu else {}
        self.sender = make_sender(a.transport, a.host, a.port, **kw)

    # ── видеоканал ──────────────────────────────────────────────────
    async def pump_frames(self):
        codec = CODEC_ID[self.a.codec]
        seq = 0
        last_print = time.time()
        n_print = 0
        async for c in self.capture.frames():
            if c.jpeg is not None and codec == JPEG:
                f = make_frame(seq, encoded=c.jpeg, codec=JPEG, size=(c.w, c.h), t_cap=c.t_cap)
            else:
                img = c.img
                if img is None:
                    import cv2
                    import numpy as np
                    img = cv2.imdecode(np.frombuffer(c.jpeg, np.uint8), cv2.IMREAD_COLOR)
                f = make_frame(seq, img=img, codec=codec, quality=self.a.quality, t_cap=c.t_cap)
            self.sender.send(f)
            seq += 1
            n_print += 1
            now = time.time()
            if now - last_print >= 1.0:
                print(f"\rзахват {self.capture.name}: {n_print / (now - last_print):5.1f} к/с  "
                      f"кадр {c.w}×{c.h} {len(f.payload) // 1024:4d} КБ  "
                      f"приёмников {self.sender.clients()}  выброшено {self.sender.dropped}  "
                      f"команд {self.cmds} (кликов {self.clicks})   ", end="", flush=True)
                last_print, n_print = now, 0

    # ── обратный канал ──────────────────────────────────────────────
    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        print(f"\nобратный канал: подключился {peer}")
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                    res = await self.dispatch(msg)
                    res.setdefault("ok", True)
                except Exception as e:
                    msg = msg if isinstance(msg, dict) else {}
                    res = {"ok": False, "error": str(e)}
                res["id"] = msg.get("id")
                res["op"] = msg.get("op")
                writer.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            print(f"\nобратный канал: отключился {peer}")
            writer.close()

    async def dispatch(self, m):
        op = m.get("op")
        self.cmds += 1
        if op == "hello":
            return {"capture": self.capture.name, "frame_size": self.capture.frame_size,
                    "region": self.capture.region, "transport": self.sender.name,
                    "strip": self.a.strip, "os_level": self.capture.os_level}
        if op == "ping":
            return {"t": m.get("t"), "t_emul": clock()}
        if op == "click":
            x, y = float(m["x"]), float(m["y"])
            px, py = self.capture.to_page(x, y)
            if self.capture.os_level:
                os_click(px, py)
            else:
                await self.page.mouse.move(px, py, steps=2)
                await self.page.mouse.down()
                await asyncio.sleep(0.04)
                await self.page.mouse.up()
            self.clicks += 1
            return {"page": [round(px, 1), round(py, 1)]}
        if op == "key":
            await self.page.keyboard.press(str(m["key"]))
            return {}
        if op == "info":
            if self.a.strip:
                await self.page.evaluate("d => window.__cvStrip && window.__cvStrip(d)",
                                         m.get("data") or {})
            return {"shown": bool(self.a.strip)}
        if op == "scenario":
            act = m.get("action")
            if act not in SCENARIO_JS:
                raise ValueError(f"нет такого действия: {act}")
            js, allowed = SCENARIO_JS[act]
            arg = m.get("arg")
            if allowed is not None and arg not in allowed:
                raise ValueError(f"недопустимый аргумент {arg!r}")
            if "=>" in js and js.startswith("()"):
                await self.page.evaluate(js)
            else:
                await self.page.evaluate(js, arg)
            return {"done": act}
        if op == "truth":
            return {"truth": await self.page.evaluate(TRUTH_JS)}
        raise ValueError(f"неизвестная команда {op}")


async def amain(a):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        emu = Emulator(a)
        await emu.setup(pw)
        srv = await asyncio.start_server(emu.handle, a.host, a.control_port)
        print(f"Щит      : {PANEL}")
        print(f"Захват   : {a.capture}, {a.fps} к/с, JPEG {a.quality}"
              + (f", полоса подсказок {a.strip} px" if a.strip else ""))
        print(f"Видео    : {emu.sender.name}")
        print(f"Управление: tcp://{a.host}:{a.control_port} (JSON-строки)")
        print("Остановка: Ctrl+C\n")
        try:
            await emu.pump_frames()
        finally:
            srv.close()
            await emu.capture.stop()
            emu.sender.close()


def main():
    ap = argparse.ArgumentParser(description="Эмулятор щита: картинка в «HDMI»-канал, "
                                             "команды из обратного канала")
    ap.add_argument("--transport", default="tcp", choices=TRANSPORTS)
    ap.add_argument("--host", default="127.0.0.1",
                    help="на каком адресе слушать (0.0.0.0 — для приёмника на другой машине)")
    ap.add_argument("--port", type=int, default=8780, help="порт видеоканала")
    ap.add_argument("--control-port", type=int, default=8790, help="порт обратного канала")
    ap.add_argument("--capture", default="screencast", choices=CAPTURES)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--quality", type=int, default=80, help="качество JPEG")
    ap.add_argument("--codec", default="jpeg", choices=["jpeg", "png", "raw"],
                    help="что уходит в канал; raw/png — перекодирование из JPEG захвата")
    ap.add_argument("--mtu", type=int, default=0, help="для udp: размер датаграммы")
    ap.add_argument("--strip", type=int, default=120,
                    help="высота полосы подсказок под щитом, px (0 — без неё, кадр 1920×1080)")
    ap.add_argument("--speed", type=float, default=0, help="ускорение времени модели (по умолчанию ×30)")
    ap.add_argument("--layout", default=None, help="компоновка щита: A, B, C, D")
    ap.add_argument("--start-pump", action="store_true", help="сразу запустить насос")
    ap.add_argument("--headed", action="store_true", help="показать окно браузера")
    a = ap.parse_args()
    if a.capture == "screen" and not a.headed:
        sys.exit("захват экрана требует видимого окна: добавьте --headed")
    try:
        asyncio.run(amain(a))
    except KeyboardInterrupt:
        print("\nостановлено")


if __name__ == "__main__":
    main()
