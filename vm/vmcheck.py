# -*- coding: utf-8 -*-
"""Проверка стенда на ВМ: скорость «HDMI»-канала и правильность работы.

    python vmcheck.py speed       # канал: к/с, задержки, ввод → пиксели, нагрузка
    python vmcheck.py accuracy    # распознавание кадров с ВМ против истины модели
    python vmcheck.py e2e         # сквозной контур receiver.py ↔ ВМ: чтение + команды
    python vmcheck.py all

Нужны запущенные ВМ (run-vm.sh) и vmhdmi.py. Истина модели берётся из
страницы щита через DevTools — распознавание её не видит, она только для оценки.
Отчёт: vm/results/vmcheck.md (+ .json).
"""

import argparse
import json
import os
import socket
import statistics as stat
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# щит: в репозитории — рядом с vm/, в локальной раскладке — в extracted/
ROOT = HERE.parent if (HERE.parent / "neft-hmi-cv.html").exists() else HERE.parent / "extracted" / "scada-cv-bridge-main"
TPL = ROOT / "cv_templates"
for p in (HERE, ROOT / "bridge", ROOT / "live", TPL):
    sys.path.insert(0, str(p))

from vmhdmi import Cdp      # noqa: E402

BASE = "http://127.0.0.1:8781"
CTL = ("127.0.0.1", 8790)
SSH = ["ssh", "-q", "-i", str(HERE / "ssh_key"), "-p", "2222", "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "scada@127.0.0.1"]
clock = time.perf_counter
REPORT = []


def say(s=""):
    print(s, flush=True)
    REPORT.append(s)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else None


def rnd(v, n=None):
    return None if v is None else round(v, n)


class Ctl:
    def __init__(self):
        s = socket.create_connection(CTL, timeout=10)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = s.makefile("rwb")

    def call(self, op, **kw):
        t = clock()
        self.f.write((json.dumps(dict(kw, op=op)) + "\n").encode())
        self.f.flush()
        r = json.loads(self.f.readline())
        if not r.get("ok", True):
            raise RuntimeError(f"{op}: {r.get('error')}")
        return r, (clock() - t) * 1000


def get_json(path):
    return json.loads(urlopen(BASE + path, timeout=10).read())


def get_img(path):
    data = urlopen(BASE + path, timeout=10).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), len(data)


class StreamWatch(threading.Thread):
    """Клиент MJPEG, как у распознавания: каждый кадр — время прихода,
    размер, метка кодирования (X-Timestamp, те же часы perf_counter)
    и средние по контрольным областям."""

    def __init__(self, rois):
        super().__init__(daemon=True)
        self.rois = rois
        self.log = []                 # (t_recv, bytes, t_enc, dec_ms)
        self.last = {}                # имя области → (t_recv, среднее по области)
        self.cond = threading.Condition()
        self.stop = False

    def run(self):
        r = urlopen(Request(BASE + "/stream"), timeout=10)
        while not self.stop:
            line = r.readline()
            if not line.startswith(b"--frame"):
                continue
            hdr = {}
            while True:
                line = r.readline().strip()
                if not line:
                    break
                k, _, v = line.decode().partition(":")
                hdr[k.strip().lower()] = v.strip()
            data = r.read(int(hdr["content-length"]))
            t = clock()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            dec = (clock() - t) * 1000
            with self.cond:
                self.log.append((t, len(data), float(hdr.get("x-timestamp", "nan")), dec))
                for name, (x, y, w, h) in self.rois.items():
                    self.last[name] = (t, img[y:y + h, x:x + w].astype(np.float32))
                self.cond.notify_all()

    def roi(self, name):
        with self.cond:
            return self.last[name][1]

    def wait_change(self, name, base, t0, thr, timeout=3.0):
        """Ждать кадр, пришедший после t0, где область отличается от
        снятой до команды (base) больше порога. Возвращает задержку в мс."""
        with self.cond:
            end = time.time() + timeout
            while time.time() < end:
                self.cond.wait(0.2)
                t, roi = self.last[name]
                if t > t0 and float(np.abs(roi - base).mean()) > thr:
                    return (t - t0) * 1000
        return None


def cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


def pid_of(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def vm_cpu(seconds=5):
    """Загрузка ВМ изнутри: доля занятого времени всех vCPU и кто ест."""
    script = (f"head -1 /proc/stat; sleep {seconds}; head -1 /proc/stat; "
              "nproc; free -m | awk '/Mem/{print $3\"/\"$2}'; "
              "top -bn1 -o %CPU | sed -n '8,12p'")
    out = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=seconds + 20).stdout.splitlines()
    a = list(map(int, out[0].split()[1:]))
    b = list(map(int, out[1].split()[1:]))
    busy = lambda v: sum(v) - v[3] - v[4]          # без idle и iowait
    load = 100.0 * (busy(b) - busy(a)) / max(1, sum(b) - sum(a))
    return {"cpu_pct": round(load, 1), "ncpu": int(out[2]), "mem_mb": out[3], "top": out[4:]}


# ───────────────────────────────────────────────────────────── скорость

def speed(res, seconds=10):
    say("## 1. Скорость «HDMI»-канала ВМ")
    say()
    ctl = Ctl()
    cdp = Cdp(local_port=9223)   # свой туннель: у vmhdmi — 9222
    cdp.evaluate("() => { if (typeof showBox !== 'undefined' && showBox) { showBox = false; syncUI(); } }")
    rois = {"markup": (930, 62, 90, 16),          # подпись разметки над манометрами (клавиша B)
            "setpoint": (1796, 831, 16, 20)}      # последняя цифра задания ЧРП (кнопки «+» / «−»)
    w = StreamWatch(rois)
    w.start()
    time.sleep(1.5)

    qemu, hdmi = pid_of(r"[q]emu-system-x86_64 -name"), pid_of(r"[v]mhdmi.py")
    get_json("/stats")
    c0 = {p: cpu_seconds(p) for p in (qemu, hdmi) if p}
    n0, t0 = len(w.log), clock()
    vm = vm_cpu(seconds)
    dt = clock() - t0
    st = get_json("/stats")
    cpu = {p: 100 * (cpu_seconds(p) - c0[p]) / dt for p in c0}
    frames = w.log[n0:]
    gaps = [(b[0] - a[0]) * 1000 for a, b in zip(frames, frames[1:])]
    ages = [(t - te) * 1000 for t, _, te, _ in frames]
    kb = [b / 1024 for _, b, _, _ in frames]
    fps = len(frames) / dt
    page_fps = cdp.evaluate("""() => new Promise(r => { let n = 0; const t0 = performance.now();
        const f = () => { n++; performance.now() - t0 < 2000 ? requestAnimationFrame(f)
                           : r(n / ((performance.now() - t0) / 1000)); }; requestAnimationFrame(f); })""")

    rtt = [ctl.call("ping", t=0)[1] for _ in range(200)]
    tr = [ctl.call("truth")[1] for _ in range(20)]

    # ввод → пиксели: клавиша B (разметка) и клик «+»/«−» на пульте
    key_lat, click_lat = [], []
    for i in range(10):
        for _ in range(2):
            b, t = w.roi("markup"), clock()
            ctl.call("key", key="b")
            key_lat.append(w.wait_change("markup", b, t, 12))
            time.sleep(0.25)
        for x in (1871, 1655):                    # «+», затем «−» — задание возвращается
            b, t = w.roi("setpoint"), clock()
            ctl.call("click", x=x, y=832)
            click_lat.append(w.wait_change("setpoint", b, t, 6))
            time.sleep(0.25)
    w.stop = True
    lost = sum(v is None for v in key_lat + click_lat)
    key_lat = [v for v in key_lat if v is not None]
    click_lat = [v for v in click_lat if v is not None]

    m = {"screen_updates_per_s": st.get("vnc_updates_per_s"), "vnc_mb_per_s": st.get("vnc_mb_per_s"),
         "page_raf_fps": round(page_fps, 1), "jpeg_encode_ms_p50": st["encode_ms_p50"],
         "jpeg_encode_ms_p95": st["encode_ms_p95"], "mjpeg_fps": round(fps, 1),
         "kb_per_frame": round(stat.mean(kb), 1), "mbit_per_s": round(fps * stat.mean(kb) * 8 / 1024, 1),
         "gap_ms_p50": rnd(pct(gaps, 50), 1), "gap_ms_p95": rnd(pct(gaps, 95), 1),
         "gap_ms_max": round(max(gaps), 1), "age_ms_p50": rnd(pct(ages, 50), 1),
         "age_ms_p95": rnd(pct(ages, 95), 1),
         "decode_ms_p50": rnd(pct([f[3] for f in frames], 50), 1),
         "ctl_rtt_ms_p50": rnd(pct(rtt, 50), 2), "ctl_rtt_ms_p95": rnd(pct(rtt, 95), 2),
         "truth_ms_p50": rnd(pct(tr, 50), 1),
         "key_to_pixels_ms_p50": rnd(pct(key_lat, 50)), "key_to_pixels_ms_p95": rnd(pct(key_lat, 95)),
         "click_to_pixels_ms_p50": rnd(pct(click_lat, 50)),
         "click_to_pixels_ms_p95": rnd(pct(click_lat, 95)), "input_lost": lost,
         "qemu_cpu_pct": round(cpu.get(qemu, 0)), "vmhdmi_cpu_pct": round(cpu.get(hdmi, 0)),
         "vm_cpu_pct": vm["cpu_pct"], "vm_ncpu": vm["ncpu"], "vm_mem_mb": vm["mem_mb"]}
    res["speed"] = m
    say("| метрика | значение |")
    say("|---|---:|")
    for k, v in (
            ("щит в ВМ рисует (requestAnimationFrame), к/с", m["page_raf_fps"]),
            ("экран ВМ меняется (обновлений VNC), в с", m["screen_updates_per_s"]),
            ("поток с экрана ВМ (VNC, сырые пиксели), МБ/с", m["vnc_mb_per_s"]),
            ("кодирование JPEG q80 p50 / p95, мс", f"{m['jpeg_encode_ms_p50']} / {m['jpeg_encode_ms_p95']}"),
            ("MJPEG у клиента, к/с", m["mjpeg_fps"]),
            ("кадр, КБ  ·  поток, Мбит/с", f"{m['kb_per_frame']}  ·  {m['mbit_per_s']}"),
            ("интервал между кадрами p50 / p95 / макс, мс",
             f"{m['gap_ms_p50']} / {m['gap_ms_p95']} / {m['gap_ms_max']}"),
            ("от кодирования до клиента p50 / p95, мс", f"{m['age_ms_p50']} / {m['age_ms_p95']}"),
            ("декодирование JPEG у клиента, мс", m["decode_ms_p50"]),
            ("RTT обратного канала p50 / p95, мс", f"{m['ctl_rtt_ms_p50']} / {m['ctl_rtt_ms_p95']}"),
            ("истина модели через DevTools, мс", m["truth_ms_p50"]),
            ("**клавиша → изменение в потоке** p50 / p95, мс",
             f"**{m['key_to_pixels_ms_p50']} / {m['key_to_pixels_ms_p95']}**"),
            ("**клик мышью → изменение в потоке** p50 / p95, мс",
             f"**{m['click_to_pixels_ms_p50']} / {m['click_to_pixels_ms_p95']}**"),
            ("потеряно команд ввода (нет реакции за 3 с)", f"{lost} из 40"),
            ("CPU хоста: QEMU / vmhdmi, % одного ядра", f"{m['qemu_cpu_pct']} / {m['vmhdmi_cpu_pct']}"),
            (f"CPU ВМ (все {m['vm_ncpu']} vCPU), %", m["vm_cpu_pct"]),
            ("память ВМ занято/всего, МБ", m["vm_mem_mb"])):
        say(f"| {k} | {v} |")
    say()
    say("Кто ест CPU внутри ВМ (top):")
    say("```")
    for line in vm["top"]:
        say(line)
    say("```")
    say()
    cdp.close()


# ───────────────────────────────────────────────────────────── точность

SCENARIOS = [
    # имя, JS подготовки (выполняется до паузы), сколько секунд дать модели
    ("стоп, исходное", "() => { initState(); setLayout('A'); speed = 30; syncUI(); }", 1),
    ("работа 50 Гц", "() => { speed = 600; cmdStart(); }", 6),
    ("40 Гц, штуцер 60 %", "() => { ui.f = 40; ui.choke = 60; syncUI(); }", 4),
    ("62 Гц, штуцер 100 %", "() => { ui.f = 62; ui.choke = 100; syncUI(); }", 4),
    ("отказ: АСПО", "() => inject('clog')", 5),
    ("отказ: обводнение", "() => inject('water')", 5),
    ("отказ: изоляция", "() => inject('iso')", 6),
    ("ремонт, пуск", "() => { initState(); speed = 600; cmdStart(); syncUI(); }", 6),
    ("компоновка B", "() => setLayout('B')", 2),
    ("компоновка C", "() => setLayout('C')", 2),
    ("компоновка D", "() => setLayout('D')", 2),
    ("D, 45 Гц, стоп", "() => { ui.f = 45; syncUI(); cmdStop(); }", 4),
]


def accuracy(res):
    import match_templates as MT
    from fast_reader import FastReader

    say("## 2. Распознавание кадров с ВМ против истины модели")
    say()
    cdp = Cdp(local_port=9223)   # свой туннель: у vmhdmi — 9222
    readers = {"png": FastReader(str(TPL), register_every=1, cache=False),
               "jpeg": FastReader(str(TPL), register_every=1, cache=False)}
    stats = {k: defaultdict(lambda: [0, 0]) for k in readers}
    gerr = {k: [] for k in readers}
    fails = []
    per = []
    try:
        cdp.evaluate("() => { showBox = false; paused = false; syncUI(); }")
        for name, js, wait in SCENARIOS:
            cdp.evaluate(js)
            time.sleep(wait)
            cdp.evaluate("() => { paused = true; syncUI(); }")
            time.sleep(1.3)                    # кадр после паузы успел дойти до JPEG
            truth = cdp.evaluate("() => frameAnnotation(null, null)")
            imgs = {"png": get_img("/snapshot.png")[0], "jpeg": get_img("/snapshot")[0]}
            truth2 = cdp.evaluate("() => frameAnnotation(null, null)")
            same = truth["sim_time_s"] == truth2["sim_time_s"]
            cdp.evaluate("() => { paused = false; syncUI(); }")
            row = {"scenario": name, "layout": truth["layout_id"], "frozen": same}
            for k, img in imgs.items():
                t = clock()
                out = readers[k].read(img, force_register=True)
                ms = (clock() - t) * 1000
                cmp_rows, st = MT.compare(out["rows"], truth, None)
                ok = sum(v[0] for v in st.values())
                tot = sum(v[1] for v in st.values())
                for c, (a, b) in st.items():
                    stats[k][c][0] += a
                    stats[k][c][1] += b
                for r, exp, got, good in cmp_rows:
                    if good is False:
                        fails.append((k, name, r["object_id"], MT.fmt(got), MT.fmt(exp)))
                    if r["cls"] == "gauge" and good is not None and r.get("value") is not None:
                        g = next(o for o in truth["objects"] if o["id"] == r["object_id"])
                        e = min(max(g["value"], g["min"]), g["max"])
                        gerr[k].append(100 * abs(e - r["value"]) / (g["max"] - g["min"]))
                row[k] = f"{ok}/{tot}"
                row[k + "_ms"] = round(ms)
            per.append(row)
            print(f"  {name:22s} {row['png']:>7s} (без потерь)  {row['jpeg']:>7s} (JPEG)", flush=True)
    finally:
        cdp.evaluate("() => { initState(); setLayout('A'); speed = 30; paused = false; syncUI(); }")
        cdp.close()

    def total(k):
        ok = sum(v[0] for v in stats[k].values())
        tot = sum(v[1] for v in stats[k].values())
        return ok, tot, 100.0 * ok / max(1, tot)

    say(f"{len(SCENARIOS)} режимов модели (пуск, частоты, штуцер, отказы, все 4 компоновки). "
        "На каждом щит ставится на паузу, снимается кадр с экрана ВМ и истина модели; "
        "читает `bridge/fast_reader.py` (манометры по стрелке), правила «верно» — из `match_templates.py`.")
    say()
    say("| вход | верно | точность | манометры: ср. / макс. ошибка, % шкалы |")
    say("|---|---:|---:|---:|")
    for k, title in (("png", "экран ВМ без потерь"), ("jpeg", "MJPEG-поток, JPEG q80")):
        ok, tot, acc = total(k)
        ge = gerr[k]
        say(f"| {title} | {ok}/{tot} | **{acc:.2f} %** | "
            f"{np.mean(ge):.2f} / {np.max(ge):.2f} |" if ge else f"| {title} | {ok}/{tot} | {acc:.2f} % | — |")
    say()
    say("По классам (MJPEG-поток):")
    say()
    say("| класс | верно |")
    say("|---|---:|")
    for c, (a, b) in sorted(stats["jpeg"].items()):
        say(f"| {c} | {a}/{b} |")
    say()
    say("| режим | компоновка | без потерь | JPEG | чтение, мс |")
    say("|---|---|---:|---:|---:|")
    for r in per:
        say(f"| {r['scenario']} | {r['layout']} | {r['png']} | {r['jpeg']} | {r['jpeg_ms']} |")
    say()
    if fails:
        say("Расхождения:")
        say()
        say("| вход | режим | объект | прочитано | истина |")
        say("|---|---|---|---:|---:|")
        for f in fails:
            say(f"| {f[0]} | {f[1]} | {f[2]} | {f[3]} | {f[4]} |")
        say()
    res["accuracy"] = {k: dict(zip(("ok", "total", "acc"), total(k))) for k in readers}
    res["accuracy"]["scenarios"] = per
    res["accuracy"]["fails"] = fails


# ───────────────────────────────────────────────────────────── сквозной контур

def e2e(res, web_port=8772):
    say("## 3. Сквозной контур: receiver.py ↔ ВМ")
    say()
    say("Штатный приёмник стенда, источник — MJPEG с экрана ВМ, HID — мышь ВМ через обратный канал. "
        "Команды выполняет распознавание по картинке; истина модели — только для оценки.")
    say()
    log = open(HERE / "results" / "e2e_receiver.log", "w", encoding="utf-8")
    rx = subprocess.Popen([sys.executable, str(ROOT / "bridge" / "receiver.py"),
                           "--source", BASE + "/stream", "--control", f"{CTL[0]}:{CTL[1]}",
                           "--no-assist", "--web-port", str(web_port)],
                          stdout=log, stderr=subprocess.STDOUT,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    base = f"http://127.0.0.1:{web_port}"
    ctl = Ctl()
    checks = []

    def get(p):
        return json.loads(urlopen(base + p, timeout=10).read())

    def post(p, d):
        r = Request(base + p, data=json.dumps(d).encode(), method="POST",
                    headers={"Content-Type": "application/json"})
        return json.loads(urlopen(r, timeout=10).read())

    def truth():
        return ctl.call("truth")[0]["truth"]

    def reading():
        s = get("/api/state")
        return s, {i["id"]: i for g in s.get("groups", {}).values() for i in g}

    def command(cmd, args=None, timeout=60):
        job = post("/api/command", {"cmd": cmd, "args": args or {}})["job"]
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(0.5)
            for j in get("/api/state")["control"]["jobs"]:
                if j["id"] == job["id"] and j["status"] not in ("в очереди", "выполняется"):
                    return j
        return {"status": "таймаут", "detail": "", "ms": None}

    def check(name, cond, detail=""):
        checks.append({"name": name, "ok": bool(cond), "detail": str(detail)})
        say(f"| {name} | {'✔' if cond else '✘'} | {detail} |")

    try:
        ctl.call("scenario", action="layout", arg="A")
        ctl.call("scenario", action="stop")
        end = time.time() + 120
        while time.time() < end:
            try:
                s = get("/api/state")
                if s.get("cv") and s["cv"]["frames"] > 3:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("приёмник не начал читать щит за 120 с")
        ctl.call("scenario", action="start")
        time.sleep(10)
        s, g = reading()
        L, C = s["link"], s["cv"]
        say("| метрика | значение |")
        say("|---|---:|")
        for k, v in (("канал, к/с", L["fps"]), ("распознавание, к/с", C["fps"]),
                     ("чтение кадра, мс", C["read_ms"]),
                     ("перечитано / из кэша", f"{C['read_objects']} / {C['cached_objects']}"),
                     ("от кадра до показаний, мс", C["glass_ms"]),
                     ("RTT обратного канала, мс", s["control"]["rtt_ms"]),
                     ("кадр трансляции", "×".join(map(str, s["raw_size"]))),
                     ("калибровка", s["calibration"])):
            say(f"| {k} | {v} |")
        say()
        say("| проверка | итог | подробности |")
        say("|---|---|---|")
        tr = truth()
        check("лампа «НАСОС В РАБОТЕ»", (g["lamp_run"].get("state") == "on") == tr["run"],
              f"прочитано {g['lamp_run'].get('state')}, модель run={tr['run']}")
        time.sleep(4)
        s, g = reading()
        tr = truth()
        f_read = g["disp_f"].get("value")
        check("частота ЧРП (индикатор)", f_read is not None and abs(f_read - tr["f"]) <= 0.15,
              f"прочитано {f_read}, модель {tr['f']}")
        j = command("set_freq", {"value": 45})
        tr = truth()
        check("задание ЧРП 45 Гц", j["status"] == "выполнено" and abs(tr["f"] - 45) < 0.01,
              f"{j['status']}: {j['detail']}; модель f={tr['f']}; {j.get('ms')} мс")
        j = command("set_choke", {"value": 60})
        tr = truth()
        check("штуцер 60 %", tr["choke"] == 60, f"{j['status']}: {j['detail']}; модель {tr['choke']} %")
        j = command("stop")
        tr = truth()
        check("останов", j["status"] == "выполнено" and not tr["run"],
              f"{j['status']}: {j['detail']}; модель run={tr['run']}; {j.get('ms')} мс")
        for lay in ("B", "C", "D"):
            ctl.call("scenario", action="layout", arg=lay)
            time.sleep(3)
            j = command("start")
            tr = truth()
            s, _ = reading()
            moved = sum(1 for b in s["blocks"].values() if b["dx"] or b["dy"])
            check(f"компоновка {lay}: пуск", j["status"] == "выполнено" and tr["run"],
                  f"блоков сдвинуто {moved}; {j['status']}: {j['detail']}")
            j = command("stop")
            tr = truth()
            check(f"компоновка {lay}: останов", j["status"] == "выполнено" and not tr["run"],
                  f"{j['status']}: {j['detail']}")
        time.sleep(2)
        s, _ = reading()
        check("события копятся", len(s["agg"]["events"]) > 5, f"событий: {len(s['agg']['events'])}")
    finally:
        try:
            ctl.call("scenario", action="layout", arg="A")
            ctl.call("scenario", action="freq", arg=50)
        except Exception:
            pass
        rx.terminate()
        try:
            rx.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rx.kill()
    ok = all(c["ok"] for c in checks)
    say()
    say(f"**Итог контура: {'всё прошло' if ok else 'есть провалы'}** "
        f"({sum(c['ok'] for c in checks)}/{len(checks)})")
    say()
    res["e2e"] = {"metrics": {"link_fps": L["fps"], "cv_fps": C["fps"], "read_ms": C["read_ms"],
                              "glass_ms": C["glass_ms"]}, "checks": checks, "ok": ok}


def main():
    ap = argparse.ArgumentParser(description="Проверка стенда на ВМ")
    ap.add_argument("what", nargs="?", default="all", choices=["speed", "accuracy", "e2e", "all"])
    ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    res = {"date": time.strftime("%Y-%m-%d %H:%M")}
    say(f"# Стенд на ВМ — скорость и правильность ({res['date']})")
    say()
    try:
        get_json("/stats")
    except OSError:
        sys.exit("vmhdmi не отвечает на :8781 — запустите vm/start.sh")
    if a.what in ("speed", "all"):
        speed(res, a.seconds)
    if a.what in ("accuracy", "all"):
        accuracy(res)
    if a.what in ("e2e", "all"):
        e2e(res)
    name = "vmcheck" if a.what == "all" else f"vmcheck_{a.what}"
    out = HERE / "results" / name
    out.with_suffix(".md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nОтчёт:", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
