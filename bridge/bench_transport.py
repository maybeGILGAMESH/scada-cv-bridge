# -*- coding: utf-8 -*-
"""Сравнение транспортов «HDMI»-канала на настоящих кадрах щита.

Источник работает в отдельном процессе — как эмулятор на другой стороне
кабеля. Кадры (14 контрольных режимов) кодируются один раз заранее, чтобы
мерить именно транспорт; стоимость кодеков меряется отдельной таблицей.

    python bridge/bench_transport.py                       # всё: 6 транспортов × 3 кодека × 2 режима
    python bridge/bench_transport.py --transports tcp,ws --codecs jpeg --rates 30
    python bridge/bench_transport.py --seconds 3 --out bridge/results/transport

Режимы частоты: 30 — обычное видео, 0 — без ограничения (сколько пролезет).
Что мерится на приёмнике:
    fps        принятых кадров в секунду
    lat p50/95 задержка от отправки до полного приёма, мс
    loss       потерянные кадры (разрывы номеров) — у UDP это потери пакетов,
               у остальных — кадры, вытесненные свежими, пока приёмник был занят
    cpu        процессорное время приёмника и источника, % одного ядра
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import CODEC_ID, JPEG, PNG, RAW, Frame, clock, decode, encode  # noqa: E402
from transports import (TRANSPORT_TITLE, TRANSPORTS, LinkStats,           # noqa: E402
                        make_receiver, make_sender)

ROOT = HERE.parent
VALIDATION = ROOT / "cv_templates" / "validation"


def load_frames(limit=14):
    files = sorted(VALIDATION.glob("*.png"))[:limit]
    out = []
    for p in files:
        a = cv2.imdecode(__import__("numpy").fromfile(str(p), "uint8"), cv2.IMREAD_COLOR)
        if a is not None:
            out.append(a)
    if not out:
        sys.exit(f"нет кадров в {VALIDATION}")
    return out


def hires_timer():
    """В Windows sleep() по умолчанию квантуется по 15,6 мс — ровные 30 к/с
    без этого не выдержать."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


# ───────────────────────────────────────────────────────────── источник

def run_sender(a):
    hires_timer()
    frames = load_frames()
    codec = CODEC_ID[a.codec]
    payloads = [encode(f, codec, a.quality) for f in frames]
    h, w = frames[0].shape[:2]
    kw = {"mtu": a.mtu} if a.transport == "udp" and a.mtu else {}
    snd = make_sender(a.transport, a.host, a.port, **kw)
    print(json.dumps({"ready": True}), flush=True)

    t_wait = time.time() + 15
    while snd.clients() == 0 and time.time() < t_wait:
        time.sleep(0.01)
    time.sleep(0.2)                   # приёмник успевает встать в recv

    period = 1.0 / a.fps if a.fps > 0 else 0.0
    cpu0, wall0 = time.process_time(), clock()
    t_next = clock()
    seq = 0
    while clock() - wall0 < a.seconds:
        f = Frame(seq, clock(), codec, w, h, payloads[seq % len(payloads)])
        snd.send(f)
        seq += 1
        if period:
            t_next += period
            d = t_next - clock()
            if d > 0.002:
                time.sleep(d - 0.0015)
            while clock() < t_next:
                pass
        else:
            # Без ограничения частоты: следующий кадр — как только транспорт
            # забрал предыдущий. У udp и shm обратной связи нет, отправка
            # синхронная — они честно показывают, сколько теряют при перегрузе.
            snd.wait_idle(1.0)
    wall = clock() - wall0
    time.sleep(0.5)                   # хвост успевает дойти
    print(json.dumps({"sent": seq, "dropped_at_source": snd.dropped,
                      "cpu_pct": round(100 * (time.process_time() - cpu0) / wall, 1)}),
          flush=True)
    time.sleep(0.3)
    snd.close()


# ───────────────────────────────────────────────────────────── один прогон

def run_case(transport, codec, fps, seconds, port, quality, mtu=0):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--role", "sender",
           "--transport", transport, "--codec", codec, "--fps", str(fps),
           "--seconds", str(seconds), "--port", str(port), "--quality", str(quality)]
    if mtu:
        cmd += ["--mtu", str(mtu)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8")
    line = p.stdout.readline()
    if not line:
        err = p.stderr.read()
        return {"error": err.strip().splitlines()[-1] if err else "источник не поднялся"}

    kw = {}
    rcv = make_receiver(transport, "127.0.0.1", port, **kw)
    st = LinkStats(window=100000)
    cpu0, wall0 = time.process_time(), clock()
    t_end = time.time() + seconds + 3
    idle = 0
    try:
        while time.time() < t_end:
            f = rcv.recv(timeout=0.5)
            if f is None:
                idle += 1
                if st.frames and idle >= 3:
                    break
                continue
            idle = 0
            st.note(f)
    except (ConnectionError, OSError, EOFError):
        pass
    wall = clock() - wall0
    cpu = time.process_time() - cpu0
    rcv.close()
    tail = p.stdout.readline()
    p.wait(timeout=10)
    res = st.summary()
    try:
        res.update(json.loads(tail))
    except Exception:
        pass
    res["cpu_rx_pct"] = round(100 * cpu / wall, 1)
    if "sent" in res:
        res["delivered_pct"] = round(100.0 * st.frames / max(1, res["sent"]), 1)
    extra = getattr(rcv, "incomplete", None)
    if extra:
        res["udp_incomplete"] = extra
    return res


# ───────────────────────────────────────────────────────────── кодеки

def bench_codecs(frames, quality, reps=3):
    rows = []
    for name, codec, q in (("jpeg q%d" % quality, JPEG, quality), ("jpeg q95", JPEG, 95),
                           ("png (быстрое)", PNG, 1), ("raw BGR", RAW, 0)):
        enc_t, dec_t, size = [], [], []
        for _ in range(reps):
            for f in frames:
                t = clock(); b = encode(f, codec, q); enc_t.append(clock() - t)
                t = clock(); decode(codec, b, f.shape[1], f.shape[0]); dec_t.append(clock() - t)
                size.append(len(b))
        rows.append({"codec": name,
                     "kb": round(sum(size) / len(size) / 1024, 1),
                     "enc_ms": round(1000 * sorted(enc_t)[len(enc_t) // 2], 2),
                     "dec_ms": round(1000 * sorted(dec_t)[len(dec_t) // 2], 2)})
    return rows


# ───────────────────────────────────────────────────────────── отчёт

def write_report(out, codecs, cases, meta):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "codecs": codecs, "cases": cases}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    L = ["# Транспорты «HDMI»-канала — замер", "",
         f"Дата: {meta['date']}  ",
         f"Машина: {meta['machine']}  ",
         f"Кадр: {meta['frame']} — 14 контрольных режимов щита по кругу  ",
         f"Длительность прогона: {meta['seconds']} с на случай", "",
         "## Кодеки (одно ядро, медиана)", "",
         "| кодек | размер кадра, КБ | кодирование, мс | декодирование, мс |",
         "|---|---:|---:|---:|"]
    for c in codecs:
        L.append(f"| {c['codec']} | {c['kb']} | {c['enc_ms']} | {c['dec_ms']} |")
    for rate in sorted({c["fps_target"] for c in cases}, reverse=True):
        title = "без ограничения частоты" if rate == 0 else f"{rate} к/с"
        L += ["", f"## Режим: {title}", "",
              "| транспорт | кодек | к/с | задержка p50, мс | p95, мс | доставлено | Мбит/с | CPU приём / источник |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for c in cases:
            if c["fps_target"] != rate:
                continue
            r = c["result"]
            if "error" in r or not r.get("frames"):
                L.append(f"| {c['transport']} | {c['codec']} | — | — | — | — | — | "
                         f"{r.get('error', 'нет кадров')} |")
                continue
            L.append(f"| {c['transport']} | {c['codec']} | {r['fps']} | {r['lat_ms_p50']} | "
                     f"{r['lat_ms_p95']} | {r.get('delivered_pct', '—')} % | {r['mbit_s']} | "
                     f"{r['cpu_rx_pct']} / {r.get('cpu_pct', '—')} % |")
    L += ["", "Транспорты:", ""]
    for t in TRANSPORTS:
        L.append(f"* `{t}` — {TRANSPORT_TITLE[t]}")
    L.append("* `udp:1400` — тот же UDP, но датаграммы по 1400 байт, как в сети с MTU 1500")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Сравнение транспортов кадра")
    ap.add_argument("--role", default="bench", choices=["bench", "sender"])
    ap.add_argument("--transport", default="tcp")
    ap.add_argument("--transports", default=",".join(TRANSPORTS[:3] + ["udp:1400"] + TRANSPORTS[3:]),
                    help="через запятую; udp:1400 — UDP с датаграммами размером с MTU сети")
    ap.add_argument("--codec", default="jpeg")
    ap.add_argument("--codecs", default="jpeg,png,raw")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--rates", default="30,0")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--mtu", type=int, default=0, help="для udp: размер датаграммы как в сети")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--out", default=str(HERE / "results" / "transport"))
    a = ap.parse_args()

    if a.role == "sender":
        run_sender(a)
        return

    import platform
    frames = load_frames()
    print("Кодеки…")
    codecs = bench_codecs(frames, a.quality)
    for c in codecs:
        print(f"  {c['codec']:<14} {c['kb']:>8} КБ  кодир. {c['enc_ms']:>6} мс  декод. {c['dec_ms']:>6} мс")

    cases = []
    port = a.port
    for rate in [float(r) for r in a.rates.split(",")]:
        for codec in a.codecs.split(","):
            for t in a.transports.split(","):
                port += 1                # свежий порт: не ждать освобождения TIME_WAIT
                print(f"{t:<6} {codec:<5} {('max' if rate == 0 else int(rate)):>4} к/с … ",
                      end="", flush=True)
                kind, _, mtu = t.partition(":")      # udp:1400 — датаграммы размером с сетевой MTU
                mtu = int(mtu) if mtu else (a.mtu if kind == "udp" else 0)
                r = run_case(kind, codec, rate, a.seconds, port, a.quality, mtu)
                cases.append({"transport": t, "codec": codec, "fps_target": int(rate), "result": r})
                if "error" in r or not r.get("frames"):
                    print("ошибка:", r.get("error", "нет кадров"))
                else:
                    print(f"{r['fps']:>7} к/с  p50 {r['lat_ms_p50']:>7} мс  p95 {r['lat_ms_p95']:>7} мс  "
                          f"доставлено {r.get('delivered_pct', '?')} %  "
                          f"CPU {r['cpu_rx_pct']}/{r.get('cpu_pct', '?')} %")
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"),
            "machine": f"{platform.processor() or platform.machine()}, "
                       f"{__import__('os').cpu_count()} лог. ядер, {platform.system()} {platform.release()}, "
                       f"Python {platform.python_version()}",
            "frame": f"{frames[0].shape[1]}×{frames[0].shape[0]}",
            "seconds": a.seconds}
    write_report(Path(a.out), codecs, cases, meta)
    print("\nОтчёт:", Path(a.out).with_suffix(".md"))


if __name__ == "__main__":
    main()
