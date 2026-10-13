# -*- coding: utf-8 -*-
"""Сверка способов чтения щита: точность на контрольном наборе и скорость.

    python bridge/bench_reader.py
    python bridge/bench_reader.py --degrade            # ещё и через «KVM»: 1280×800, JPEG q70

Варианты:
    template  исходный конвейер live/recognize.py — манометры перебором шаблонов
    needle    манометры по стрелке, остальное как было, один поток
    needle+mt то же, объекты читаются параллельно
    full      needle+mt плюс кэш неизменившихся областей (на «живом» потоке,
              где соседние кадры похожи; на наборе из 14 разных режимов кэш
              почти не срабатывает — это честно видно в отчёте)

Истина — JSON рядом с каждым кадром: числа, из которых кадр и был нарисован.
Правила «верно» — те же, что в cv_templates/match_templates.py.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "live"))
sys.path.insert(0, str(ROOT / "cv_templates"))

import match_templates as MT          # noqa: E402
from fast_reader import FastReader    # noqa: E402

TPL = ROOT / "cv_templates"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def kvm_degrade(img, cap=(1280, 800), q=70):
    """Как кадр приходит через KVM: щит вписан в захват, поля, JPEG."""
    cw, ch = cap
    k = min(cw / img.shape[1], ch / img.shape[0])
    w, h = int(img.shape[1] * k), int(img.shape[0] * k)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ch, cw, 3), np.uint8)
    y0, x0 = (ch - h) // 2, (cw - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = small
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_set():
    idx = json.loads((TPL / "validation" / "index.json").read_text(encoding="utf-8"))
    out = []
    for fr in idx["frames"]:
        img = imread(TPL / fr["image"])
        truth = json.loads((TPL / fr["truth"]).read_text(encoding="utf-8"))
        out.append((fr["scenario"], img, truth))
    return out


def gauge_errors(rows, truth):
    gt = {o["id"]: o for o in truth["objects"]}
    errs = []
    for r in rows:
        if r["cls"] != "gauge":
            continue
        g = gt.get(r["object_id"])
        if g is None or r.get("value") is None:
            continue
        exp = min(max(g["value"], g["min"]), g["max"])
        errs.append(100.0 * abs(exp - r["value"]) / (g["max"] - g["min"]))
    return errs


def run_variant(name, frames, make, degrade, normalizer=None):
    reader = make()
    stats = defaultdict(lambda: [0, 0])
    times, gerr, fails = [], [], []
    first = None
    for scen, img, truth in frames:
        frame = img
        if degrade:
            frame = normalizer(kvm_degrade(img))
        t = time.perf_counter()
        rows = reader(frame)
        dt = time.perf_counter() - t
        if first is None:
            first = dt
        else:
            times.append(dt)
        cmp_rows, st = MT.compare(rows, truth, None)
        for k, (ok, tot) in st.items():
            stats[k][0] += ok
            stats[k][1] += tot
        for r, exp, got, ok in cmp_rows:
            if ok is False:
                fails.append((scen, r["object_id"], MT.fmt(got), MT.fmt(exp)))
        gerr += gauge_errors(rows, truth)
    ok = sum(v[0] for v in stats.values())
    tot = sum(v[1] for v in stats.values())
    return {"variant": name, "ok": ok, "total": tot,
            "acc": round(100.0 * ok / max(1, tot), 2),
            "gauge_ok": stats["gauge"][0], "gauge_total": stats["gauge"][1],
            "gauge_err_mean": round(float(np.mean(gerr)), 3) if gerr else None,
            "gauge_err_max": round(float(np.max(gerr)), 3) if gerr else None,
            "first_ms": round(first * 1000), "ms_median": round(1000 * float(np.median(times))),
            "fails": fails}


def live_template():
    """Исходный live/recognize.Reader как есть: манометры перебором шаблонов
    с подсказкой угла по прошлому кадру."""
    from recognize import Reader
    r = Reader(str(TPL), register_every=8)

    def read(frame):
        out = r.read(frame)
        return [dict(it, object_id=it["id"], cls=cls)
                for cls, items in out["groups"].items() for it in items]
    return read


def template_nohint():
    """Ядро match_templates.py без подсказок: полный перебор углов на каждом кадре."""
    bank = MT.Bank(TPL)

    def read(frame):
        bank.set_offsets(MT.offsets_only(MT.register_blocks(frame, bank)))
        return MT.recognise(frame, bank, None, None, 36)
    return read


def fast(gauge_method, workers, cache):
    def make():
        fr = FastReader(str(TPL), register_every=8, gauge_method=gauge_method,
                        workers=workers, cache=cache)
        return lambda frame: fr.read(frame)["rows"]
    return make


def live_stream_speed(frames, seconds=6.0):
    """Скорость на «живом» потоке: один и тот же режим, кадры чуть разные
    (как на трансляции, где между кадрами меняются часы и мигалки)."""
    out = {}
    base = frames[0][1]
    seq = []
    for i in range(30):
        f = base.copy()
        cv2.putText(f, f"{i:02d}", (1760, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        seq.append(f)
    for name, make in (("needle+mt", fast("needle", None, False)),
                       ("full", fast("needle", None, True))):
        rd = make()
        rd(seq[0])
        ts = []
        t_end = time.time() + seconds
        i = 0
        while time.time() < t_end:
            t = time.perf_counter()
            rd(seq[i % len(seq)])
            ts.append(time.perf_counter() - t)
            i += 1
        out[name] = round(1000 * float(np.median(ts)), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Сверка способов чтения щита")
    ap.add_argument("--degrade", action="store_true", help="ещё и через имитацию KVM")
    ap.add_argument("--out", default=str(HERE / "results" / "reader"))
    a = ap.parse_args()

    frames = load_set()
    print(f"кадров: {len(frames)}")
    variants = [("template без подсказок (match_templates)", template_nohint),
                ("template с подсказкой (исходный live)", live_template),
                ("needle, 1 поток", fast("needle", 1, False)),
                ("needle, параллельно", fast("needle", None, False))]
    results = []
    for degrade in ([False, True] if a.degrade else [False]):
        normalizer = None
        if degrade:
            sys.path.insert(0, str(ROOT / "live"))
            import normalize as N
            bank = MT.Bank(TPL)
            normalizer = N.Normalizer(bank.ref())
        for name, make in variants:
            print(f"  {'KVM ' if degrade else ''}{name} …", end=" ", flush=True)
            r = run_variant(name, frames, make, degrade, normalizer)
            r["degraded"] = degrade
            results.append(r)
            print(f"{r['ok']}/{r['total']} = {r['acc']} %  манометры {r['gauge_ok']}/{r['gauge_total']}"
                  f" (ср. {r['gauge_err_mean']} %, макс {r['gauge_err_max']} % шкалы)  "
                  f"кадр {r['ms_median']} мс, первый {r['first_ms']} мс")
    print("  живой поток (похожие кадры) …", end=" ", flush=True)
    live = live_stream_speed(frames)
    print(live)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"results": results, "live_ms": live},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Способы чтения щита — точность и скорость", "",
         f"Дата: {time.strftime('%Y-%m-%d %H:%M')}  ",
         f"Набор: {len(frames)} контрольных режимов, истина — JSON модели на каждом кадре", "",
         "| вход | вариант | верно | точность | манометры: ср. / макс. ошибка, % шкалы | кадр, мс | первый кадр, мс |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        L.append(f"| {'KVM 1280×800, JPEG 70' if r['degraded'] else 'кадр 1:1'} | {r['variant']} | "
                 f"{r['ok']}/{r['total']} | {r['acc']} % | {r['gauge_err_mean']} / {r['gauge_err_max']} | "
                 f"{r['ms_median']} | {r['first_ms']} |")
    L += ["", "## Живой поток", "",
          "Один режим, соседние кадры почти одинаковы (меняются часы и мигалки) — "
          "так выглядит трансляция между событиями.", "",
          "| вариант | кадр, мс (медиана) |", "|---|---:|"]
    for k, v in live.items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Расхождения", "", "| вход | вариант | кадр | объект | прочитано | истина |",
          "|---|---|---|---|---:|---:|"]
    for r in results:
        for f in r["fails"]:
            L.append(f"| {'KVM' if r['degraded'] else '1:1'} | {r['variant']} | {f[0]} | {f[1]} | {f[2]} | {f[3]} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Отчёт:", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
