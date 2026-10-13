# -*- coding: utf-8 -*-
"""Сравнение способов снять картинку с эмулятора щита.

    python bridge/bench_capture.py
    python bridge/bench_capture.py --seconds 8 --fps 60

Для каждого способа (screencast, screenshot, canvas) поднимается свежий
Chrome со щитом, кадры снимаются заданное время и считаются: сколько кадров
в секунду получилось, сколько стоил один кадр, какой размер JPEG. Захват
экрана монитора (screen) здесь не меряется: ему нужно видимое окно
браузера поверх рабочего стола.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture import make_capture   # noqa: E402
from emulator import KIOSK_CSS, PANEL, launch   # noqa: E402
from protocol import clock          # noqa: E402


async def run_one(kind, seconds, fps, quality):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        br = await launch(pw, headed=False)
        page = await br.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        await page.goto(PANEL.resolve().as_uri())
        await page.add_style_tag(content=KIOSK_CSS % {"h": 1})
        await page.evaluate("() => cmdStart()")      # щит в работе: стрелки и потоки движутся
        await page.wait_for_timeout(500)
        cap = make_capture(kind, page, fps=fps, quality=quality)
        await cap.start()
        n, size, gaps = 0, 0, []
        t0 = clock()
        prev = t0
        async for c in cap.frames():
            now = clock()
            gaps.append(now - prev)
            prev = now
            n += 1
            size += len(c.jpeg or b"")
            if now - t0 >= seconds:
                break
        await cap.stop()
        await br.close()
    span = prev - t0
    gaps = sorted(gaps[1:]) or [0.0]
    return {"capture": kind, "frames": n, "fps": round(n / span, 1),
            "interval_ms_p50": round(1000 * gaps[len(gaps) // 2], 1),
            "interval_ms_p95": round(1000 * gaps[int(len(gaps) * 0.95)], 1),
            "kb": round(size / max(1, n) / 1024, 1)}


def main():
    ap = argparse.ArgumentParser(description="Сравнение способов захвата эмулятора")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=60, help="потолок частоты; 0 — без ограничения")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--out", default=str(HERE / "results" / "capture"))
    a = ap.parse_args()
    rows = []
    for kind in ("screencast", "screenshot", "canvas"):
        print(f"{kind:<11} …", end=" ", flush=True)
        r = asyncio.run(run_one(kind, a.seconds, a.fps, a.quality))
        rows.append(r)
        print(f"{r['fps']:>5} к/с  интервал p50 {r['interval_ms_p50']} мс, p95 {r['interval_ms_p95']} мс, "
              f"{r['kb']} КБ")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Захват картинки с эмулятора — замер", "",
         f"Дата: {time.strftime('%Y-%m-%d %H:%M')}  ",
         f"Chrome headless, щит 1920×1080 в режиме киоска, насос в работе, JPEG {a.quality}, "
         f"потолок {a.fps:g} к/с, {a.seconds:g} с на способ", "",
         "| способ | к/с | интервал p50, мс | p95, мс | кадр, КБ |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['capture']} | {r['fps']} | {r['interval_ms_p50']} | {r['interval_ms_p95']} | {r['kb']} |")
    L += ["", "* `screencast` — CDP Page.startScreencast: браузер кодирует JPEG сам после отрисовки и проталкивает кадр",
          "* `screenshot` — CDP Page.captureScreenshot: каждый кадр отдельным запросом",
          "* `canvas` — canvas.toDataURL внутри страницы, кадр через JS и base64"]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Отчёт:", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
