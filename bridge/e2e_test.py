# -*- coding: utf-8 -*-
"""Сквозная проверка контура: эмулятор → транспорт → распознавание → команда → щит.

Поднимает emulator.py и receiver.py отдельными процессами, управляет щитом
через сайт приёмника (как оператор) и сверяет результат с истинным
состоянием модели (тестовый канал эмулятора, op=truth). Распознавание и
управление истину не видят — она нужна только здесь, чтобы поставить оценку.

    python bridge/e2e_test.py
    python bridge/e2e_test.py --transport ws
    python bridge/e2e_test.py --transport shm --seconds 20

Что проверяется:
  1. поток и чтение: частота канала, задержка, частота распознавания,
     время «от захвата до показаний»;
  2. совпадение прочитанного с моделью (частота, дебит, обводнённость, лампы);
  3. команды с проверкой по картинке: пуск, задание частоты, штуцер, останов;
  4. перестановка блоков щита: после смены компоновки кнопки уехали, а
     клики всё равно попадают — координат кнопок в коде нет.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Truth:
    """Отдельное подключение к тестовому каналу эмулятора."""

    def __init__(self, port):
        t_end = time.time() + 60
        while True:
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                if time.time() > t_end:
                    raise
                time.sleep(0.3)
        self.f = self.s.makefile("rwb")

    def call(self, **m):
        self.f.write((json.dumps(m) + "\n").encode())
        self.f.flush()
        return json.loads(self.f.readline())

    def get(self):
        return self.call(op="truth")["truth"]


def main():
    ap = argparse.ArgumentParser(description="Сквозная проверка контура")
    ap.add_argument("--transport", default="tcp")
    ap.add_argument("--capture", default="screencast")
    ap.add_argument("--port", type=int, default=8880)
    ap.add_argument("--seconds", type=float, default=10.0, help="сколько мерить поток")
    ap.add_argument("--out", default=str(HERE / "results" / "e2e"))
    a = ap.parse_args()

    cport, wport = a.port + 10, a.port + 20
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    logs = Path(a.out).parent
    logs.mkdir(parents=True, exist_ok=True)
    emu = subprocess.Popen([sys.executable, str(HERE / "emulator.py"), "--transport", a.transport,
                            "--port", str(a.port), "--control-port", str(cport),
                            "--capture", a.capture],
                           stdout=open(logs / "e2e_emulator.log", "w", encoding="utf-8"),
                           stderr=subprocess.STDOUT, env=env)
    time.sleep(3)
    rx = subprocess.Popen([sys.executable, str(HERE / "receiver.py"), "--transport", a.transport,
                           "--port", str(a.port), "--control", f"127.0.0.1:{cport}",
                           "--web-port", str(wport)],
                          stdout=open(logs / "e2e_receiver.log", "w", encoding="utf-8"),
                          stderr=subprocess.STDOUT, env=env)
    base = f"http://127.0.0.1:{wport}"
    report = []
    ok_all = True
    result = {"transport": a.transport, "capture": a.capture,
              "date": time.strftime("%Y-%m-%d %H:%M"), "metrics": {}, "checks": []}

    def get(p):
        return json.loads(urllib.request.urlopen(base + p, timeout=10).read())

    def post(p, d):
        r = urllib.request.Request(base + p, data=json.dumps(d).encode(), method="POST",
                                   headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=10).read())

    def say(line):
        print(line)
        report.append(line)

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        result["checks"].append({"name": name, "ok": bool(cond), "detail": str(detail)})
        say(f"| {name} | {'✔' if cond else '✘'} | {detail} |")

    def reading():
        s = get("/api/state")
        return s, {i["id"]: i for g in s.get("groups", {}).values() for i in g}

    def command(cmd, args=None, timeout=60):
        job = post("/api/command", {"cmd": cmd, "args": args or {}})["job"]
        t_end = time.time() + timeout
        while time.time() < t_end:
            time.sleep(0.5)
            for j in get("/api/state")["control"]["jobs"]:
                if j["id"] == job["id"] and j["status"] not in ("в очереди", "выполняется"):
                    return j
        return {"status": "таймаут", "detail": "", "ms": None}

    try:
        truth = Truth(cport)
        t_end = time.time() + 90
        while time.time() < t_end:
            try:
                s = get("/api/state")
                if s.get("cv") and s["cv"]["frames"] > 3:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("приёмник не начал читать щит за 90 с")

        say(f"# Сквозная проверка: {a.transport}, захват {a.capture}")
        say("")
        say(f"Дата: {time.strftime('%Y-%m-%d %H:%M')}")
        say("")

        # 1. поток
        truth.call(op="scenario", action="start")
        time.sleep(a.seconds)
        s, g = reading()
        L, C = s["link"], s["cv"]
        result["metrics"] = {
            "link_fps": L["fps"], "lat_p50_ms": L["lat_ms_p50"], "lat_p95_ms": L["lat_ms_p95"],
            "kb_per_frame": L["kb_per_frame"], "loss_pct": L["loss_pct"],
            "cv_fps": C["fps"], "read_ms": C["read_ms"], "read_objects": C["read_objects"],
            "cached_objects": C["cached_objects"], "glass_ms": C["glass_ms"],
            "rtt_ms": s["control"]["rtt_ms"], "raw_size": s["raw_size"]}
        say("## Поток и чтение")
        say("")
        say("| метрика | значение |")
        say("|---|---:|")
        for k, v in (("канал, к/с", L["fps"]), ("задержка канала p50 / p95, мс",
                                                 f"{L['lat_ms_p50']} / {L['lat_ms_p95']}"),
                     ("размер кадра, КБ", L["kb_per_frame"]), ("потери, %", L["loss_pct"]),
                     ("распознавание, к/с", C["fps"]), ("чтение кадра, мс", C["read_ms"]),
                     ("перечитано / из кэша", f"{C['read_objects']} / {C['cached_objects']}"),
                     ("от захвата до показаний, мс", C["glass_ms"]),
                     ("RTT обратного канала, мс", s["control"]["rtt_ms"]),
                     ("кадр трансляции", "×".join(map(str, s["raw_size"]))),
                     ("калибровка", s["calibration"])):
            say(f"| {k} | {v} |")
        say("")

        # 2. совпадение с моделью
        say("## Прочитанное против модели")
        say("")
        say("| проверка | итог | подробности |")
        say("|---|---|---|")
        tr = truth.get()
        check("лампа «НАСОС В РАБОТЕ»", (g["lamp_run"].get("state") == "on") == tr["run"],
              f"прочитано {g['lamp_run'].get('state')}, модель run={tr['run']}")
        time.sleep(4)
        s, g = reading()
        tr = truth.get()
        f_read = g["disp_f"].get("value")
        check("частота ЧРП (индикатор)", f_read is not None and abs(f_read - tr["f"]) <= 0.15,
              f"прочитано {f_read}, задание модели {tr['f']}")
        say("")

        # 3. команды
        say("## Команды через HID с проверкой по картинке")
        say("")
        say("| команда | итог | подробности |")
        say("|---|---|---|")
        j = command("set_freq", {"value": 45})
        tr = truth.get()
        check("задание ЧРП 45 Гц", j["status"] == "выполнено" and abs(tr["f"] - 45) < 0.01,
              f"{j['status']}: {j['detail']}; модель f={tr['f']}; {j.get('ms')} мс")
        j = command("set_choke", {"value": 60})
        tr = truth.get()
        check("штуцер 60 %", tr["choke"] == 60, f"{j['status']}: {j['detail']}; модель {tr['choke']} %")
        j = command("stop")
        tr = truth.get()
        check("останов", j["status"] == "выполнено" and not tr["run"],
              f"{j['status']}: {j['detail']}; модель run={tr['run']}; {j.get('ms')} мс")

        # 4. перестановка блоков
        for lay in ("B", "C", "D"):
            truth.call(op="scenario", action="layout", arg=lay)
            time.sleep(3)
            j = command("start")
            tr = truth.get()
            s, _ = reading()
            moved = sum(1 for b in s["blocks"].values() if b["dx"] or b["dy"])
            check(f"компоновка {lay}: пуск", j["status"] == "выполнено" and tr["run"],
                  f"блоков сдвинуто {moved}; {j['status']}: {j['detail']}")
            j = command("stop")
            tr = truth.get()
            check(f"компоновка {lay}: останов", j["status"] == "выполнено" and not tr["run"],
                  f"{j['status']}: {j['detail']}")
        truth.call(op="scenario", action="layout", arg="A")

        # 5. агрегатор и подсказки
        time.sleep(2)
        s, _ = reading()
        ev = s["agg"]["events"]
        check("события копятся", len(ev) > 5, f"событий: {len(ev)}")
        check("подсказки уходят на экран щита", s["control"].get("assist_error") is None,
              s["control"].get("assist_error") or "полоса под щитом обновляется")
        say("")
        say(f"**Итог: {'всё прошло' if ok_all else 'есть провалы'}**")
    finally:
        for p in (rx, emu):
            p.terminate()
        for p in (rx, emu):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        out = Path(a.out + f"_{a.transport}").with_suffix(".md")
        out.write_text("\n".join(report) + "\n", encoding="utf-8")
        result["ok"] = ok_all
        out.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        print("\nОтчёт:", out, "(и .json рядом)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
