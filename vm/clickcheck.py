# -*- coding: utf-8 -*-
"""Проверка щелчков «Щита в окне»: каждая кнопка каждого щита — настоящим
щелчком мыши в браузере, там, где кнопка видна на экране, при нескольких
размерах окна.

Щит в ВМ на время проверки только записывает, куда пришёл щелчок и какая
кнопка сработала бы (обработчик в фазе перехвата на window), а сам щелчок
дальше не пускает — «АВАР. СТОП», «ТКРС», имитация отказов не срабатывают.

  python vm/clickcheck.py                     все системы
  python vm/clickcheck.py --sys newsys --sizes 1280x720
Отчёт: vm/results/clickcheck.md. Нужны дашборд (./cvdash.sh) и google-chrome.
"""

import argparse
import json
import warnings
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

warnings.filterwarnings("ignore", category=DeprecationWarning)
# где щит нажимается: отдельное окно и вкладка «Системы» дашборда (id картинки)
PAGES = [("окно", "/system?sys={}", "live"), ("вкладка", "/?tab=systems&sys={}", "sysLive")]
HERE = Path(__file__).resolve().parent
DASH = "http://127.0.0.1:8775"

# цели на щите: у нового щита (canvas) — зоны нажатия HIT, у старого — элементы <button>
TARGETS_JS = """() => {
  if (typeof HIT !== 'undefined' && HIT.length)
    return HIT.map((h, i) => ({i, x: h.x, y: h.y, w: h.w, h: h.h,
      name: (h.fn.name || String(h.fn).replace(/\\s+/g, ' ').slice(0, 48))}));
  return [...document.querySelectorAll('button')].map((b, i) => {
    const r = b.getBoundingClientRect();
    return {i, x: r.x, y: r.y, w: r.width, h: r.height, name: b.textContent.trim()};
  }).filter(t => t.w > 4 && t.h > 4 && t.name && !document.querySelectorAll('button')[t.i].disabled);   // недоступная кнопка щелчков не принимает
}"""
DRY_ON = """() => {
  window.__clk = [];
  if (!window.__dry) {
    window.__dry = e => {
      if (e.type === 'click') {
        let hit = null;
        if (typeof HIT !== 'undefined' && HIT.length) {
          const k = HIT.findIndex(h => e.clientX >= h.x && e.clientX <= h.x + h.w && e.clientY >= h.y && e.clientY <= h.y + h.h);
          hit = k >= 0 ? k : null;
        } else {
          const el = document.elementFromPoint(e.clientX, e.clientY), b = el && el.closest('button');
          hit = b ? b.textContent.trim() : null;
        }
        window.__clk.push({x: e.clientX, y: e.clientY, hit});
      }
      e.stopImmediatePropagation(); e.preventDefault();
    };
    for (const t of ['click', 'mousedown', 'mouseup', 'pointerdown', 'pointerup']) addEventListener(t, window.__dry, true);
  }
  return true;
}"""
DRY_OFF = """() => { if (window.__dry) { for (const t of ['click', 'mousedown', 'mouseup', 'pointerdown', 'pointerup'])
  removeEventListener(t, window.__dry, true); delete window.__dry; } return true; }"""


class Page:
    def __init__(self, ws_url):
        self.ws, self.n = connect(ws_url, max_size=2**24), 0

    def cmd(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            r = json.loads(self.ws.recv(timeout=20))
            if r.get("id") == self.n:
                return r.get("result", {})

    def ev(self, js):
        r = self.cmd("Runtime.evaluate", expression=js, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("text"))
        return r["result"].get("value")


def panel_page(port):
    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=5))
    return Page(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])


def screen_point(geo, fx, fy):
    """Где точка кадра видна на экране: как браузер рисует картинку (object-fit)."""
    x, y, w, h, nw, nh, fit = geo
    if fit == "fill":
        return x + fx * w / nw, y + fy * h / nh
    s = min(w / nw, h / nh)
    return x + (w - nw * s) / 2 + fx * s, y + (h - nh * s) / 2 + fy * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys")
    ap.add_argument("--sizes", default="1600x1000,1280x720,1920x1080,1100x900")
    a = ap.parse_args()
    reg = json.loads((HERE / "systems.json").read_text(encoding="utf-8"))["systems"]
    want = set(a.sys.split(",")) if a.sys else None
    prof = tempfile.mkdtemp()
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--password-store=basic", "--no-first-run",
                               f"--user-data-dir={prof}", "--remote-debugging-port=9398", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows, total, good = [], 0, 0
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9398/json"))
                break
            except Exception:
                time.sleep(0.2)
        br = Page(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
        for s in reg:
            if want and s["id"] not in want:
                continue
            pp = panel_page(s["ports"]["cdp_dash"])
            targets = pp.ev(f"({TARGETS_JS})()")
            pp.ev(f"({DRY_ON})()")
            try:
                for (pname, url, img_id), size in [(pg, sz) for pg in PAGES for sz in a.sizes.split(",")]:
                    W, H = map(int, size.split("x"))
                    br.cmd("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
                    br.cmd("Page.navigate", url=DASH + url.format(s["id"]))
                    for _ in range(40):
                        time.sleep(0.25)
                        if br.ev(f"(() => {{ const i = document.getElementById('{img_id}'); return !!i && i.naturalWidth > 0; }})()"):
                            break
                    # кнопка должна быть видна в окне: вкладку прокручиваем к щиту
                    br.ev(f"document.getElementById('{img_id}').scrollIntoView({{block: 'center'}})")
                    time.sleep(0.3)
                    geo = br.ev(f"""(() => {{ const i = document.getElementById('{img_id}'), r = i.getBoundingClientRect();
                      return [r.x, r.y, r.width, r.height, i.naturalWidth, i.naturalHeight, getComputedStyle(i).objectFit]; }})()""")
                    misses = []
                    for t in targets:
                        cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
                        px, py = screen_point(geo, cx, cy)
                        pp.ev("(() => { window.__clk = []; return 1; })()")
                        for typ in ("mouseMoved", "mousePressed", "mouseReleased"):
                            br.cmd("Input.dispatchMouseEvent", type=typ, x=px, y=py, button="left" if typ != "mouseMoved" else "none",
                                   clickCount=1)
                        got = None
                        for _ in range(30):
                            time.sleep(0.05)
                            got = pp.ev("window.__clk.length ? window.__clk[window.__clk.length - 1] : null")
                            if got:
                                break
                        expect = t["i"] if got and isinstance(got.get("hit"), int) else t["name"]
                        ok = bool(got) and got.get("hit") == expect
                        total += 1
                        good += ok
                        if not ok:
                            misses.append(f"{t['name'][:28]} (центр {cx:.0f},{cy:.0f}) → "
                                          + (f"пришло в {got['x']:.0f},{got['y']:.0f}: {got.get('hit')!r}" if got else "щелчок не пришёл"))
                    rows.append((s["id"], f"{pname} {size}", len(targets), len(targets) - len(misses), geo[6], misses))
                    print(f"{s['id']:9s} {pname:7s} {size:9s} кнопок {len(targets):2d}: попало {len(targets) - len(misses):2d}"
                          f"  (картинка {geo[2]:.0f}×{geo[3]:.0f}, object-fit {geo[6]})", flush=True)
                    for m in misses[:6]:
                        print("    ✗", m, flush=True)
            finally:
                pp.ev(f"({DRY_OFF})()")
    finally:
        chrome.terminate()
    lines = ["# Проверка щелчков «Щита в окне»", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · каждая кнопка — настоящим щелчком в браузере, щит в ВМ только записывает попадание", "",
             "| система | окно | кнопок | попало | промахи |", "|---|---|---|---|---|"]
    for sid, size, n, ok, fit, misses in rows:
        lines.append(f"| {sid} | {size} | {n} | {ok} | {'<br>'.join(misses) or '—'} |")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "clickcheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nвсего {good} из {total} щелчков попали в свою кнопку · отчёт vm/results/clickcheck.md")


if __name__ == "__main__":
    main()
