# -*- coding: utf-8 -*-
"""Дашборд «что видит система»: распознавание щита ВМ в реальном времени.

Берёт тот же MJPEG, что и распознавание (vmhdmi.py), читает каждый кадр тем же
FastReader, что receiver.py, и показывает в браузере:

  * кадр с наложением — найденные блоки, рамки приборов, стрелки манометров,
    прочитанное и истина модели, цвет — совпало / нет / не уверен;
  * по щелчку на приборе — промежуточные шаги OpenCV: развёртка cv2.remap и
    профиль «красноты» стрелки, кандидаты-глифы индикатора, сравнение
    состояний лампы, профиль заливки шкалы, карта matchTemplate блока;
  * точность по классам, время по этапам, историю за последние минуты.

Истина модели берётся из страницы щита через DevTools — только для оценки,
распознавание её не видит. Модель идёт ускоренно, поэтому сверка «вживую»
допускает совпадение с истиной до или после чтения кадра; точная сверка —
кнопкой «Точная проверка» (модель на паузе).

    python cvdash.py          # http://127.0.0.1:8775/
"""

import argparse
import base64
import json
import socket
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# щит: в репозитории — рядом с vm/, в локальной раскладке — в extracted/
ROOT = HERE.parent if (HERE.parent / "neft-hmi-cv.html").exists() else HERE.parent / "extracted" / "scada-cv-bridge-main"
# эталоны, нарезанные с щита в ВМ (regen_templates.py); нет их — исходные
TPL = HERE / "cv_templates_vm" if (HERE / "cv_templates_vm" / "manifest.json").exists() else ROOT / "cv_templates"
for p in (HERE, ROOT / "bridge", ROOT / "live", ROOT / "cv_templates"):
    sys.path.insert(0, str(p))

import match_templates as MT                           # noqa: E402
from fast_reader import CLASS_TITLE, FastReader        # noqa: E402
from vmhdmi import Cdp                                 # noqa: E402
from extra_reader import ExtraReader                   # noqa: E402
from glyph_ocr import line_ink                         # noqa: E402

clock = time.perf_counter
TRUTH_JS = """() => Object.assign(frameAnnotation(null, null), {
  hist: {p: hist.p, q: hist.q, e: hist.e}, speed: paused ? 0 : speed,
  log2: log.slice(0, 2).map(e => ({t: e.t.slice(0, 8), tag: e.tag, msg: e.msg}))})"""
# Сверка текста журнала: кириллица и латиница одинакового начертания (А/A, С/C…)
# в этом шрифте совпадают пиксель в пиксель — по картинке их не различить; разные
# тире и точки-разделители OCR тоже сводит к одному. Читаемое при этом не меняется.
HOMO = str.maketrans("АВЕКМНОРСТХаеорсух", "ABEKMHOPCTXaeopcyx")


def norm_text(t):
    import re
    t = (t or "").translate(HOMO)
    t = re.sub(r"\s*[·•.\-–—]+\s*", " - ", t)
    return " ".join(t.split())


TREND_KEY = {"trend_p": "p", "trend_q": "q", "trend_e": "e"}
TITLES = dict(CLASS_TITLE, text_readout="Текстовые табло", trend="Тренды", log="Журнал", clock="Часы модели")
BLOCK_TITLE = {"mimic": "Мнемосхема", "gauges": "Манометры", "displays": "Индикаторы",
               "right": "Сигнализация", "console": "Пульт", "bars": "Столбчатые",
               "trends": "Тренды", "log": "Журнал"}


def png64(img, scale=1.0, interp=cv2.INTER_NEAREST):
    if img is None or img.size == 0:
        return None
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


# последовательная шкала (синяя, от фона к яркому) — для тепловых карт OpenCV
_RAMP = ["#0d1a2b", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]


def _lut():
    pts = [tuple(int(h[i:i + 2], 16) for i in (5, 3, 1)) for h in _RAMP]     # BGR
    lut = np.zeros((256, 1, 3), np.uint8)
    for v in range(256):
        f = v / 255 * (len(pts) - 1)
        i = min(int(f), len(pts) - 2)
        t = f - i
        lut[v, 0] = [round(pts[i][c] * (1 - t) + pts[i + 1][c] * t) for c in range(3)]
    return lut


LUT = _lut()


def heat(a, lo=None, hi=None):
    a = a.astype(np.float32)
    lo = float(a.min()) if lo is None else lo
    hi = float(a.max()) if hi is None else hi
    g = np.clip((a - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, LUT)


def crop(frame, x, y, w, h, pad=0):
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
    x1, y1 = min(W, int(x + w + pad)), min(H, int(y + h + pad))
    return frame[y0:y1, x0:x1].copy(), (x0, y0)


# ───────────────────────────────────────────────────────────── источник

class Mjpeg(threading.Thread):
    """MJPEG vmhdmi с меткой кодирования (X-Timestamp — те же часы perf_counter)."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cond = threading.Condition()
        self.img, self.seq, self.t_enc, self.t_recv = None, 0, 0.0, 0.0
        self.error = None

    def run(self):
        while True:
            try:
                r = urlopen(Request(self.url), timeout=10)
                self.error = None
                while True:
                    line = r.readline()
                    if not line:
                        break
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
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    with self.cond:
                        self.img, self.t_recv = img, clock()
                        self.t_enc = float(hdr.get("x-timestamp", self.t_recv))
                        self.seq += 1
                        self.cond.notify_all()
            except OSError as e:
                self.error = str(e)
                time.sleep(1)

    def next_after(self, t, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.img is not None and self.t_enc >= t, timeout)
            return self.img, self.t_enc, self.t_recv


class Ctl:
    """Обратный канал vmhdmi — команды щиту и тестовый канал."""

    def __init__(self, addr=("127.0.0.1", 8790)):
        self.addr, self.f, self.lock = addr, None, threading.Lock()

    def call(self, op, **kw):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.f is None:
                        s = socket.create_connection(self.addr, timeout=10)
                        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        self.f = s.makefile("rwb")
                    self.f.write((json.dumps(dict(kw, op=op)) + "\n").encode())
                    self.f.flush()
                    r = json.loads(self.f.readline())
                    if not r.get("ok", True):
                        raise RuntimeError(r.get("error"))
                    return r
                except OSError:
                    self.f = None
                    if attempt:
                        raise


# ───────────────────────────────────────────────────────────── зрение

def fmt_read(r):
    if r["cls"] == "gauge":
        v = r.get("value")
        return None if v is None else f"{v:.2f}"
    if r["cls"] == "sevenseg_digit":
        return r.get("display") or r.get("text")
    if r["cls"] == "vessel_level":
        v = r.get("level_pct")
        return None if v is None else f"{v:.1f} %"
    if r["cls"] == "bargraph":
        v = r.get("value")
        return None if v is None else f"{v:.1f}"
    if "state" in r:
        return r.get("state")
    return None


class Vision(threading.Thread):
    def __init__(self, src, cdp, ctl, hist_s=180):
        super().__init__(daemon=True)
        self.src, self.cdp, self.ctl = src, cdp, ctl
        self.reader = FastReader(str(TPL), register_every=8, cache=True)
        self.bank = self.reader.bank
        self.extra = ExtraReader(TPL)
        orig = self.reader._read_one

        def timed(frame, oid, recs):
            t = clock()
            r = orig(frame, oid, recs)
            r["_ms"] = (clock() - t) * 1000
            return r
        self.reader._read_one = timed
        self.lock = threading.Lock()
        self.state = {"ready": False}
        self.frames = {}                       # seq → (jpeg, raw, offsets)
        self.hist = deque(maxlen=hist_s * 12)
        self.roll = deque(maxlen=60)           # по кадрам: {класс: [ok, всего]}
        self.gerr = deque(maxlen=600)
        self.errors = deque(maxlen=40)
        self.events = deque(maxlen=40)
        self.seq = 0
        self.t_fps = deque(maxlen=30)
        self.paused_vision = False
        self.last_layout = None
        self.prev_offsets = None

    def _truth(self):
        t = clock()
        tr = self.cdp.evaluate(TRUTH_JS)
        return tr, (clock() - t) * 1000

    def run(self):
        while True:
            try:
                self.step()
            except Exception as e:                 # ВМ перезапустили, страницу перезагрузили
                with self.lock:
                    self.state["error"] = f"{type(e).__name__}: {e}"
                time.sleep(1)

    def step(self):
        if self.paused_vision:
            time.sleep(0.2)
            return
        tr_a, ms_a = self._truth()
        t_a = clock()
        img, t_enc, t_recv = self.src.next_after(t_a - 0.005)
        if img is None:
            time.sleep(0.2)
            return
        t0 = clock()
        out = self.reader.read(img)
        t_read = clock() - t0
        t1 = clock()
        ex = self.extra.read(img, dict(self.bank.offsets))
        t_extra = clock() - t1
        tr_b, ms_b = self._truth()
        t_done = clock()

        cmp_a, _ = MT.compare(out["rows"], tr_a, None)
        cmp_b, _ = MT.compare(out["rows"], tr_b, None)
        ta = {o["id"]: o for o in tr_a["objects"]}
        tb = {o["id"]: o for o in tr_b["objects"]}
        offsets = dict(self.bank.offsets)
        objs, cls_stat = [], defaultdict(lambda: [0, 0])
        cls_ms = defaultdict(float)
        for (r, exp_a, got, ok_a), (_, exp_b, _, ok_b) in zip(cmp_a, cmp_b):
            oid, cls = r["object_id"], r["cls"]
            ok = None if ok_a is None and ok_b is None else bool(ok_a or ok_b)
            exp = exp_a if ok_a and not ok_b else exp_b
            if ok is not None:
                cls_stat[cls][1] += 1
                cls_stat[cls][0] += int(ok)
            if not r.get("cached"):
                cls_ms[cls] += r.get("_ms", 0.0)
            g = tb.get(oid) or ta.get(oid) or {}
            o = {"id": oid, "cls": cls, "label": r.get("label") or g.get("label") or oid,
                 "at": [int(v) for v in r["at"]], "size": [int(v) for v in r["size"]],
                 "read": fmt_read(r), "truth": MT.fmt(exp) if exp is not None else None,
                 "ok": ok, "score": round(float(r.get("score") or 0), 3),
                 "margin": None if r.get("margin") is None else round(float(r["margin"]), 4),
                 "reliable": r.get("reliable"), "cached": bool(r.get("cached")),
                 "method": r.get("method"), "ms": round(r.get("_ms", 0.0), 1),
                 "block": self.bank.block_of.get(oid), "truth_bbox": g.get("bbox")}
            if cls == "gauge":
                ng = self.reader.needles.get(oid)
                dx, dy = offsets.get(self.bank.block_of.get(oid), (0, 0))
                if ng:
                    o["center"] = [ng.center[0] + dx, ng.center[1] + dy]
                    o["radius"] = [float(np.hypot(ng.dx[0, 0], ng.dy[0, 0])),
                                   float(np.hypot(ng.dx[0, -1], ng.dy[0, -1]))]
                o["angle"] = r.get("angle_deg")
                if ok is not None and r.get("value") is not None and g:
                    e = min(max(g["value"], g["min"]), g["max"])
                    err = 100 * abs(e - r["value"]) / (g["max"] - g["min"])
                    o["err_pct"] = round(err, 2)
                    self.gerr.append(err)
            if ok is False:
                if not self.errors or self.errors[0]["id"] != oid or self.errors[0]["read"] != o["read"]:
                    self.errors.appendleft({"t": time.strftime("%H:%M:%S"), "id": oid, "cls": cls,
                                            "label": o["label"], "read": o["read"], "truth": o["truth"]})
            objs.append(o)

        trends = self._merge_extra(objs, ex, tr_a, tr_b, cls_stat)
        for o in objs:
            if o["cls"] in ("text_readout", "trend", "log", "clock") and o.get("ok") is False:
                if not any(e["id"] == o["id"] and e["read"] == o["read"] for e in list(self.errors)[:6]):
                    self.errors.appendleft({"t": time.strftime("%H:%M:%S"), "id": o["id"], "cls": o["cls"],
                                            "label": o.get("label") or o["id"], "read": o["read"], "truth": o["truth"]})

        blocks = []
        tblocks = tr_b.get("blocks") or {}
        for name, v in (out["registration"] or {}).items():
            rect = self.bank.blocks.get(name)
            if not rect:
                continue
            x, y, w, h = [int(c) for c in rect]
            tb_rect = tblocks.get(name)
            blocks.append({"name": name, "title": BLOCK_TITLE.get(name, name),
                           "rect": [x + v[0], y + v[1], w, h], "dx": v[0], "dy": v[1],
                           "score": round(v[2], 3),
                           "truth_rect": [round(c) for c in tb_rect] if isinstance(tb_rect, list) else None})

        self.roll.append(dict(cls_stat))
        roll = defaultdict(lambda: [0, 0])
        for fr in self.roll:
            for c, (a, b) in fr.items():
                roll[c][0] += a
                roll[c][1] += b
        ok_n = sum(v[0] for v in cls_stat.values())
        tot_n = sum(v[1] for v in cls_stat.values())
        r_ok = sum(v[0] for v in roll.values())
        r_tot = sum(v[1] for v in roll.values())
        self.t_fps.append(t_done)
        fps = (len(self.t_fps) - 1) / (self.t_fps[-1] - self.t_fps[0]) if len(self.t_fps) > 2 else 0.0
        lay = tr_b.get("layout_id")
        if lay != self.last_layout:
            if self.last_layout is not None:
                self.events.appendleft({"t": time.strftime("%H:%M:%S"),
                                        "text": f"компоновка {self.last_layout} → {lay}"})
            self.last_layout = lay
        if out["registered_this_frame"]:
            moved = [BLOCK_TITLE.get(k, k) for k, v in offsets.items()
                     if self.prev_offsets is not None and tuple(v) != tuple(self.prev_offsets.get(k, (0, 0)))]
            if moved:
                self.events.appendleft({"t": time.strftime("%H:%M:%S"),
                                        "text": f"matchTemplate нашёл блоки на новых местах: {', '.join(moved)} "
                                                f"(балл {out['registration_quality']:.2f})"})
            self.prev_offsets = dict(offsets)
        ok_jpg, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.seq += 1
        timing = {"read_ms": round(t_read * 1000, 1), "registration_ms": out["timing_ms"]["registration"],
                  "extra_ms": round(t_extra * 1000, 1), "ocr_calls": self.extra.tess.calls,
                  "truth_ms": round((ms_a + ms_b) / 2, 1),
                  "age_ms": round((t_done - t_enc) * 1000, 1),
                  "transport_ms": round((t_recv - t_enc) * 1000, 1),
                  "per_class_ms": {c: round(v, 1) for c, v in sorted(cls_ms.items(), key=lambda z: -z[1])}}
        hist_row = {"t": time.time(), "acc": round(100 * ok_n / max(1, tot_n), 2),
                    "read_ms": timing["read_ms"], "fps": round(fps, 2),
                    "quality": out["registration_quality"], "age_ms": timing["age_ms"],
                    "reread": out["read_objects"]}
        with self.lock:
            self.frames[self.seq] = (jpg.tobytes(), img, offsets, ex)
            for k in [k for k in self.frames if k < self.seq - 4]:
                del self.frames[k]
            self.hist.append(hist_row)
            ge = list(self.gerr)
            self.state = {
                "ready": True, "seq": self.seq, "t": time.time(),
                "frame_size": [img.shape[1], img.shape[0]],
                "objects": objs, "blocks": blocks, "class_titles": TITLES, "trends": trends,
                "log_lines": [None if l is None else {k: l[k] for k in ("time", "tag", "msg")}
                              for l in (ex["log"] or {}).get("lines", [])],
                "log_truth": tr_b.get("log2"),
                "registered": out["registered_this_frame"],
                "quality": out["registration_quality"],
                "read_objects": out["read_objects"], "cached_objects": out["cached_objects"],
                "fps": round(fps, 2), "timing": timing,
                "frame": {"ok": ok_n, "total": tot_n},
                "rolling": {"ok": r_ok, "total": r_tot, "frames": len(self.roll),
                            "by_class": {c: {"ok": a, "total": b, "title": TITLES.get(c, c)}
                                         for c, (a, b) in roll.items()}},
                "gauge_err": {"mean": round(float(np.mean(ge)), 3) if ge else None,
                              "p95": round(float(np.percentile(ge, 95)), 3) if ge else None,
                              "hist": np.histogram(np.clip(ge, 0, 2), bins=10, range=(0, 2))[0].tolist()
                              if ge else []},
                "model": {"layout": lay, "layout_name": tr_b.get("layout_name"),
                          "clock": tr_b.get("sim_clock"), "plant": tr_b.get("plant_state")},
                "errors": list(self.errors), "events": list(self.events),
                "source_error": self.src.error, "error": None,
            }

    def snapshot(self):
        with self.lock:
            s = dict(self.state)
            s["history"] = list(self.hist)
            return s

    def frame(self, seq=None):
        with self.lock:
            if not self.frames:
                return None
            k = seq if seq in self.frames else max(self.frames)
            return self.frames[k]

    # ── точная проверка: модель на паузе ─────────────────────────────
    def exact(self):
        self.paused_vision = True
        try:
            self.cdp.evaluate("() => { paused = true; syncUI(); }")
            time.sleep(1.3)
            truth = self.cdp.evaluate(TRUTH_JS)
            img, _, _ = self.src.next_after(clock())
            rd = FastReader(str(TPL), register_every=1, cache=False)
            out = rd.read(img, force_register=True)
            rows, st = MT.compare(out["rows"], truth, None)
            rd.close()
            ex = self.extra.read(img, MT.offsets_only(out["registration"]))
            objs = [{"id": r["object_id"], "cls": r["cls"], "at": [0, 0], "size": [0, 0]} for r in out["rows"]]
            st2 = defaultdict(lambda: [0, 0])
            self._merge_extra(objs, ex, truth, truth, st2)
            for c, (a, b) in st2.items():
                st[c][0] += a
                st[c][1] += b
            extra_fails = [{"id": o["id"], "cls": o["cls"], "read": o.get("read"), "truth": o.get("truth")}
                           for o in objs if o.get("ok") is False]
        finally:
            self.cdp.evaluate("() => { paused = false; syncUI(); }")
            self.paused_vision = False
        ok = sum(v[0] for v in st.values())
        tot = sum(v[1] for v in st.values())
        return {"ok": ok, "total": tot, "acc": round(100 * ok / max(1, tot), 2),
                "layout": truth.get("layout_id"), "clock": truth.get("sim_clock"),
                "fails": [{"id": r["object_id"], "cls": r["cls"], "read": MT.fmt(g), "truth": MT.fmt(e)}
                          for r, e, g, k in rows if k is False] + extra_fails}

    # ── шаги OpenCV для одного объекта ───────────────────────────────
    def detail(self, oid):
        fr = self.frame()
        if fr is None:
            return {"error": "нет кадра"}
        _, img, offsets, ex = fr
        if oid in ex["readouts"] or oid in ex["trends"] or oid in ("log_banner", "hdr_clock"):
            return self.detail_extra(oid, img, ex)
        bank = self.bank
        recs = bank.by_obj.get(oid)
        if not recs:
            return {"error": f"нет объекта {oid}"}
        cls = bank.classes[oid]
        dx, dy = offsets.get(bank.block_of.get(oid), (0, 0))

        def anchor(rec):
            return (rec["anchor"][0] + dx, rec["anchor"][1] + dy)

        kind = MT.CLASS_READER.get(cls, "locate")
        d = {"id": oid, "cls": cls, "class_title": CLASS_TITLE.get(cls, cls), "kind": kind,
             "block": bank.block_of.get(oid), "shift": [dx, dy], "templates": len(recs)}
        if kind == "gauge" and oid in self.reader.needles:
            d.update(self._gauge(img, oid, recs, dx, dy))
        elif kind == "sevenseg":
            d.update(self._sevenseg(img, oid, recs, dx, dy))
        elif kind in ("bargraph", "vessel"):
            d.update(self._fill(img, recs, anchor, kind))
        elif kind == "discrete":
            d.update(self._discrete(img, recs, anchor))
        else:
            d.update(self._locate(img, recs[0], anchor))
        return d

    def _gauge(self, img, oid, recs, dx, dy):
        ng = self.reader.needles[oid]
        cx, cy = ng.center[0] + dx, ng.center[1] + dy
        polar = cv2.remap(img, ng.dx + np.float32(cx), ng.dy + np.float32(cy),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        p = polar.astype(np.int16)
        red = p[:, :, 2] - np.maximum(p[:, :, 0], p[:, :, 1])
        hit = red > 70
        prof = np.convolve(hit.mean(axis=1), np.ones(3) / 3.0, mode="same")
        k = int(np.argmax(prof))
        peak = float(prof[k])
        lo = hi = k
        while lo > 0 and prof[lo - 1] >= peak * 0.5:
            lo -= 1
        while hi < len(prof) - 1 and prof[hi + 1] >= peak * 0.5:
            hi += 1
        w = prof[lo:hi + 1]
        ang = float(np.sum(ng.angles[lo:hi + 1] * w) / max(1e-9, np.sum(w)))
        mask = np.ones_like(prof, bool)
        mask[max(0, lo - 10):hi + 11] = False
        second = float(prof[mask].max()) if mask.any() else 0.0
        r_in = float(np.hypot(ng.dx[0, 0], ng.dy[0, 0]))
        r_out = float(np.hypot(ng.dx[0, -1], ng.dy[0, -1]))
        R = int(r_out / 0.72) + 6
        c, (x0, y0) = crop(img, cx - R, cy - R, 2 * R, 2 * R)
        ov = c.copy()
        ctr = (int(round(cx - x0)), int(round(cy - y0)))
        # кольцо, где ищется стрелка, и найденный луч
        cv2.ellipse(ov, ctr, (int(r_in), int(r_in)), 0, ng.angles[0], ng.angles[-1], (229, 135, 57), 1, cv2.LINE_AA)
        cv2.ellipse(ov, ctr, (int(r_out), int(r_out)), 0, ng.angles[0], ng.angles[-1], (229, 135, 57), 1, cv2.LINE_AA)
        for a in (ng.angles[lo], ng.angles[hi]):
            t = np.deg2rad(a)
            cv2.line(ov, (int(ctr[0] + r_in * np.cos(t)), int(ctr[1] + r_in * np.sin(t))),
                     (int(ctr[0] + r_out * np.cos(t)), int(ctr[1] + r_out * np.sin(t))),
                     (129, 81, 213), 1, cv2.LINE_AA)
        t = np.deg2rad(ang)
        cv2.line(ov, ctr, (int(ctr[0] + r_out * 1.05 * np.cos(t)), int(ctr[1] + r_out * 1.05 * np.sin(t))),
                 (129, 81, 213), 2, cv2.LINE_AA)
        # развёртка: по горизонтали угол, по вертикали радиус (центр снизу)
        unwrap = np.ascontiguousarray(polar.transpose(1, 0, 2)[::-1])
        redmap = heat(np.clip(red, 0, 255).T[::-1], 0, 255)
        hitmap = np.where(hit.T[::-1, :, None], np.array([57, 135, 229], np.uint8),
                          np.array([27, 26, 26], np.uint8)).astype(np.uint8)
        rec = recs[0]
        sc = self.bank.man["gauge_scale"]
        frac = float(np.clip((ang - sc["start_deg"]) / sc["sweep_deg"], 0, 1))
        accepted = peak >= 0.25 and second < peak * 0.6
        return {"images": {"crop": png64(c, 2), "overlay": png64(ov, 2, cv2.INTER_LINEAR),
                           "unwrap": png64(unwrap, 1.0), "redmap": png64(redmap, 1.0),
                           "hitmap": png64(hitmap, 1.0)},
                "profile": {"angles": [round(float(a), 1) for a in ng.angles],
                            "values": [round(float(v), 4) for v in prof],
                            "lo": float(ng.angles[lo]), "hi": float(ng.angles[hi]),
                            "peak": round(peak, 3), "second": round(second, 3),
                            "threshold": 0.25, "second_limit": round(peak * 0.6, 3)},
                "result": {"angle": round(ang, 2), "frac": round(frac, 4),
                           "value": round(rec["min"] + (rec["max"] - rec["min"]) * frac, 3),
                           "min": rec["min"], "max": rec["max"], "unit": rec.get("unit"),
                           "start_deg": sc["start_deg"], "sweep_deg": sc["sweep_deg"],
                           "accepted": accepted,
                           "radii": [round(r_in, 1), round(r_out, 1)], "rays": len(ng.angles)}}

    def _sevenseg(self, img, oid, recs, dx, dy):
        glyphs = {r["char"]: r for r in recs}
        cells = [[c[0] + dx, c[1] + dy, c[2], c[3]] for c in self.bank.cells.get(oid, [])]
        out = []
        for cell in cells:
            cand = [g for g in glyphs.values() if abs(g["w"] - cell[2]) <= 3 and abs(g["h"] - cell[3]) <= 3] \
                or list(glyphs.values())
            sc = sorted(((MT.match(img, self.bank.img(g), None, (cell[0], cell[1]), 8)[0], g)
                         for g in cand), key=lambda z: -z[0])
            cc, _ = crop(img, cell[0], cell[1], cell[2], cell[3], 2)
            margin = sc[0][0] - sc[1][0] if len(sc) > 1 else 1.0
            out.append({"crop": png64(cc, 3), "best": sc[0][1]["char"],
                        "accepted": margin >= MT.MARGIN_MIN, "margin": round(margin, 4),
                        "cands": [{"char": g["char"], "score": round(s, 4), "img": png64(self.bank.img(g), 3)}
                                  for s, g in sc[:4]]})
        x0 = min(c[0] for c in cells) if cells else 0
        y0 = min(c[1] for c in cells) if cells else 0
        x1 = max(c[0] + c[2] for c in cells) if cells else 0
        y1 = max(c[1] + c[3] for c in cells) if cells else 0
        whole, _ = crop(img, x0, y0, x1 - x0, y1 - y0, 10)
        return {"images": {"crop": png64(whole, 2)}, "cells": out,
                "result": {"text": "".join(c["best"] if c["accepted"] else "?" for c in out),
                           "margin_min": MT.MARGIN_MIN, "glyphs": sorted(glyphs)}}

    def _discrete(self, img, recs, anchor):
        loc = None
        for r in recs:
            s, x, y = MT.match(img, self.bank.img(r), self.bank.mask(r), anchor(r), 26)
            if loc is None or s > loc[0]:
                loc = (s, x, y, r)
        s_loc, x, y, r_loc = loc
        scored = []
        for r in recs:
            tpl = self.bank.img(r)
            c = MT._crop_like(img, x, y, tpl)
            dm = np.abs(c.astype(np.int16) - tpl.astype(np.int16)).mean(axis=2)
            msk = self.bank.mask(r)
            sel = None
            if msk is not None and msk.shape[:2] == tpl.shape[:2]:
                sel = msk[:, :, 0] > 127
            d = dm[sel] if sel is not None and sel.any() else dm
            scored.append((1.0 - float(d.mean()) / 255.0, r, dm))
        scored.sort(key=lambda z: -z[0])
        best = scored[0]
        margin = best[0] - scored[1][0] if len(scored) > 1 else 1.0
        tpl = self.bank.img(best[1])
        c = MT._crop_like(img, x, y, tpl)
        sc = max(1.0, min(4.0, 160 / max(tpl.shape[:2])))
        return {"images": {"crop": png64(c, sc), "diff": png64(heat(best[2], 0, 60), sc)},
                "states": [{"state": r.get("state"), "sim": round(s, 4), "img": png64(self.bank.img(r), sc)}
                           for s, r, _ in scored],
                "result": {"state": best[1].get("state") if margin >= 0.004 else None,
                           "margin": round(margin, 4), "margin_min": 0.004,
                           "locate_score": round(s_loc, 3), "at": [int(x), int(y)]}}

    def _fill(self, img, recs, anchor, kind):
        if kind == "bargraph":
            empty = next((r for r in recs if r.get("fill_frac") == 0), recs[0])
            full = next((r for r in recs if r.get("fill_frac") == 1), empty)
            rect = full.get("fill_rect_rel") or [10, 26, full["w"] - 20, full["h"] - 38]
            axis, from_end = 0, False
        else:
            full = next((r for r in recs if (r.get("level_pct") or 0) > 50), recs[0])
            empty = next((r for r in recs if (r.get("level_pct") or 0) <= 50), recs[0])
            rect = full.get("measure_rect_rel") or full.get("liquid_rect_rel")
            axis, from_end = 1, True
        ei, fi = self.bank.img(empty), self.bank.img(full)
        inv = MT.invariant_mask(ei, fi)
        if inv is None:
            s, x, y = MT.match(img, ei, None, anchor(empty), 20)
        else:
            s, x, y = MT.match_sqdiff(img, ei, inv, anchor(empty), 20)
        c = MT._crop_like(img, x, y, ei)
        ref = MT.diff_profile(fi, ei, rect, axis)
        cur = MT.diff_profile(c, ei, rect, axis)
        frac = MT.fill_fraction(c, ei, fi, rect, axis, from_end) if rect else None
        lvl = float(np.median(ref[ref > 1])) if ref is not None and np.any(ref > 1) else 0.0
        ov = c.copy()
        if rect:
            rx, ry, rw, rh = [int(v) for v in rect]
            cv2.rectangle(ov, (rx, ry), (rx + rw - 1, ry + rh - 1), (229, 135, 57), 1)
        sc = max(1.0, min(3.0, 260 / max(ei.shape[:2])))
        return {"images": {"crop": png64(ov, sc), "empty": png64(ei, sc), "full": png64(fi, sc),
                           "mask": png64(inv, sc) if inv is not None else None,
                           "diff": png64(heat(np.abs(c.astype(np.int16) - ei.astype(np.int16)).max(axis=2), 0, 120), sc)},
                "profile": {"ref": [round(float(v), 1) for v in ref] if ref is not None else [],
                            "cur": [round(float(v), 1) for v in cur] if cur is not None else [],
                            "lit": round(lvl * 0.35, 1), "significant": round(lvl * 0.5, 1),
                            "axis": "по столбцам →" if axis == 0 else "по строкам ↓", "from_end": from_end},
                "result": {"frac": None if frac is None else round(float(frac), 4),
                           "locate_score": round(float(s), 3), "masked": inv is not None}}

    def _locate(self, img, rec, anchor):
        tpl = self.bank.img(rec)
        ax, ay = anchor(rec)
        pad = 36
        roi, (x0, y0) = crop(img, ax - pad, ay - pad, tpl.shape[1] + 2 * pad, tpl.shape[0] + 2 * pad)
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        hm = heat(res, -0.2, 1.0)
        cv2.drawMarker(hm, loc, (255, 255, 255), cv2.MARKER_CROSS, 9, 1)
        sc = max(1.0, min(3.0, 260 / max(tpl.shape[:2])))
        return {"images": {"crop": png64(roi, sc / 1.5), "template": png64(tpl, sc / 1.5), "heat": png64(hm, 4)},
                "result": {"score": round(float(mx), 3), "found": [x0 + loc[0], y0 + loc[1]],
                           "expected": [int(ax), int(ay)]}}

    # ── текст и графики: сверка с моделью ────────────────────────────
    @staticmethod
    def _clock_sec(t):
        import re
        m = re.search(r"(\d{2}):(\d{2}):(\d{2})\s*\(сут\s*(\d+)\)", t or "")
        if not m:
            return None
        h, mi, sec, d = map(int, m.groups())
        return d * 86400 + h * 3600 + mi * 60 + sec

    def _merge_extra(self, objs, ex, tr_a, tr_b, cls_stat):
        by = {o["id"]: o for o in objs}
        ta = {o["id"]: o for o in tr_a["objects"]}
        tb = {o["id"]: o for o in tr_b["objects"]}

        def put(oid, cls, read, truth, ok, at=None, size=None, **kw):
            o = by.get(oid)
            if o is None:
                g = tb.get(oid) or {}
                bb = g.get("bbox") or [0, 0, 0, 0]
                o = {"id": oid, "cls": cls, "label": g.get("label") or oid, "score": 1.0, "margin": None,
                     "reliable": None, "cached": False, "method": None, "ms": 0.0,
                     "block": self.extra.block_of.get(oid), "truth_bbox": g.get("bbox"),
                     "at": [int(bb[0]), int(bb[1])], "size": [int(bb[2]), int(bb[3])]}
                objs.append(o)
                by[oid] = o
            o["cls"] = cls
            if at is not None:
                o["at"], o["size"] = at, size
            o.update(read=read, truth=truth, ok=ok, **kw)
            if ok is not None:
                cls_stat[cls][1] += 1
                cls_stat[cls][0] += int(ok)

        # табло: текст как на щите, без округления
        for oid, r in ex["readouts"].items():
            t1, t2 = (ta.get(oid) or {}).get("text"), (tb.get(oid) or {}).get("text")
            ok = None if t2 is None else (r["text"] is not None and r["text"] in (t1, t2))
            truth = t1 if (ok and r["text"] == t1) else t2
            chars = r.get("chars") or []
            put(oid, "text_readout", r["text"], truth, ok, method="glyph", value_box=r.get("box"),
                score=round(min([c[1] for c in chars] or [0]), 3),
                margin=round(min([c[2] for c in chars] or [0]), 3))

        # часы: модель идёт ускоренно — время кадра между двумя снимками истины
        c = ex.get("clock")
        if c:
            sa = self._clock_sec((ta.get("hdr_clock") or {}).get("text"))
            sb = self._clock_sec((tb.get("hdr_clock") or {}).get("text"))
            sr = self._clock_sec(c.get("text"))
            # кадр приходит с задержкой захвата (до ~1 с): при ускорении модели это десятки
            # модельных секунд, поэтому окно сверки расширяется на скорость модели
            lag = 3 + 1.0 * float(tr_b.get("speed") or 0)
            ok = None if sb is None else (sr is not None and min(sa or sb, sb) - lag <= sr <= max(sa or sb, sb) + 3)
            bb = c["box"]
            put("hdr_clock", "clock", c.get("text"), (tb.get("hdr_clock") or {}).get("text"), ok,
                at=bb[:2], size=bb[2:], method="glyph")

        # журнал: верхняя строка «ИСТОЧНИК: сообщение»
        lg = ex.get("log")
        if lg and lg["lines"] and lg["lines"][0]:
            l0 = lg["lines"][0]
            read = f"{l0['tag']}: {l0['msg']}"
            norm = norm_text
            t1 = (ta.get("log_banner") or {}).get("text")
            t2 = (tb.get("log_banner") or {}).get("text")
            ok = None if not t2 else norm(read) in (norm(t1), norm(t2))
            bb = lg["box"]
            put("log_banner", "log", read, t2, ok, at=bb[:2], size=bb[2:], method="tesseract")

        # тренды: последние значения и клон всей кривой
        trends = {}
        for tid, T in ex["trends"].items():
            key = TREND_KEY.get(tid)
            ha = (tr_a.get("hist") or {}).get(key) or []
            hb = (tr_b.get("hist") or {}).get(key) or []
            g = tb.get(tid) or {}
            ymin, ymax = g.get("y_min", 0), g.get("y_max", 1)
            rng = (ymax - ymin) or 1
            last = [s["last"] for s in T["series"]]
            tl_a = [min(max(v, ymin), ymax) for v in ha[-1]] if ha else []
            tl_b = [min(max(v, ymin), ymax) for v in hb[-1]] if hb else []
            def errs(tl):
                return [None if (v is None or i >= len(tl)) else 100 * abs(v - tl[i]) / rng
                        for i, v in enumerate(last)]
            ea, eb = errs(tl_a), errs(tl_b)
            err = [min([e for e in (a, b) if e is not None], default=None) for a, b in zip(ea, eb)]
            ok = None if not hb else (len(last) == len(tl_b) and all(e is not None and e <= 2.0 for e in err))
            # кривая: столбец поля графика ↔ доля истории; ошибка — по столбцам, где кривая видна
            n = len(hb)
            curves, truthc, rms = [], [], []
            for si, s in enumerate(T["series"]):
                pts, es = [], []
                cv_ = s["curve"]
                W = len(cv_) - 1
                for cx in range(0, len(cv_), 2):
                    v = cv_[cx]
                    pts.append(None if v is None else round(v, 2))
                    if v is not None and n > 1:
                        f = cx / max(1, W) * (n - 1)
                        i0 = int(f)
                        i1 = min(n - 1, i0 + 1)
                        tv = hb[i0][si] + (hb[i1][si] - hb[i0][si]) * (f - i0)
                        es.append(100 * abs(v - min(max(tv, ymin), ymax)) / rng)
                curves.append(pts)
                rms.append(round(float(np.sqrt(np.mean(np.square(es)))), 2) if es else None)
            for si in range(len(T["series"])):
                truthc.append([round(min(max(r[si], ymin), ymax), 2) for r in hb])
            hexes = ["#%02x%02x%02x" % (c[2], c[1], c[0]) for c in (s["color"] for s in T["series"])]
            trends[tid] = {"title": g.get("label") or tid, "names": T.get("names"), "colors": hexes,
                           "y_min": T["y_min"], "y_max": T["y_max"], "plot": T["plot"],
                           "curve": curves, "truth": truthc, "last": [None if v is None else round(v, 2) for v in last],
                           "truth_last": [round(v, 2) for v in tl_b], "err_last_pct": [None if e is None else round(e, 2) for e in err],
                           "curve_rms_pct": rms, "coverage": [round(s["coverage"], 3) for s in T["series"]]}
            fmt = lambda arr: " · ".join("—" if v is None else f"{v:.1f}" for v in arr)
            put(tid, "trend", fmt(last), fmt(tl_b), ok, method="color",
                err_pct=max([e for e in err if e is not None], default=None))
        return trends

    def detail_extra(self, oid, img, ex):
        d = {"id": oid}
        if oid in ex["readouts"]:
            r = ex["readouts"][oid]
            d.update(kind="readout", class_title="Текстовые табло")
            g = self.extra.ref.get(oid) or {}
            d["label"] = g.get("label")
            if r.get("_val") is None:
                d["images"] = {"crop": png64(r.get("_crop"), 3)}
                d["result"] = {"text": None}
                return d
            v, m = line_ink(r["_val"])
            text, segs = self.extra.glyphs.explain(r["_val"], allowed=set("0123456789.-"))
            val = cv2.resize(r["_val"], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
            for sg in segs:
                a, b = sg["seg"]
                cv2.rectangle(val, (a * 6, 0), (b * 6 - 1, val.shape[0] - 1), (229, 135, 57), 1)
            d["images"] = {"crop": png64(r["_crop"], 4), "value": png64(val, 1.0),
                           "ink": png64(heat((v * 255).astype(np.uint8), 0, 255), 6) if v is not None else None}
            d["chars"] = [{"glyph": png64((sg["glyph"] * 255).astype(np.uint8), 6),
                           "cands": [{"char": ch, "score": sc, "img": png64((t * 255).astype(np.uint8), 6)}
                                     for ch, sc, t in sg["cands"]]} for sg in segs]
            d["result"] = {"text": text.replace(" ", ""), "box": r.get("box"), "gray": r.get("locked")}
        elif oid in ex["trends"]:
            T = ex["trends"][oid]
            L, Tp, PW, PH = T["plot"]
            crop = img[Tp - 20:Tp + PH + 30, L - 44:L + PW + 14].copy()
            masks = np.zeros_like(crop)
            for s in T["series"]:
                m = s["_mask"]
                col = np.array(s["color"], np.uint8)
                sub = masks[17:17 + m.shape[0], 44:44 + m.shape[1]]
                sub[m[:sub.shape[0], :sub.shape[1]]] = col
            top = img[Tp - 5:Tp + 6, L - 40:L - 3]
            bot = img[Tp + PH - 5:Tp + PH + 6, L - 40:L - 3]
            d.update(kind="trend", class_title="Тренды",
                     images={"crop": png64(crop, 1.4, cv2.INTER_LINEAR), "masks": png64(masks, 1.4),
                             "top": png64(top, 5), "bot": png64(bot, 5)},
                     result={"y_min": T["y_min"], "y_max": T["y_max"], "plot": T["plot"],
                             "colors": ["#%02x%02x%02x" % (c[2], c[1], c[0]) for c in (s["color"] for s in T["series"])],
                             "names": T.get("names"), "last": [s["last"] for s in T["series"]],
                             "coverage": [round(s["coverage"], 3) for s in T["series"]]})
        elif oid == "log_banner":
            lg = ex.get("log") or {"lines": []}
            from extra_reader import ink
            lines = []
            for l in lg["lines"]:
                if not l:
                    lines.append(None)
                    continue
                im = l["_img"]
                lines.append({"img": png64(im, 2.5, cv2.INTER_LINEAR),
                              "ink": png64(ink(im[:, 126:], 3), 0.6, cv2.INTER_AREA),
                              "time": l["time"], "tag": l["tag"], "msg": l["msg"]})
            d.update(kind="log", class_title="Журнал", lines=lines)
        elif oid == "hdr_clock":
            c = ex.get("clock") or {}
            crop = c.get("_crop")
            text, segs = self.extra.glyphs.explain(crop) if crop is not None else ("", [])
            vis = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST) if crop is not None else None
            for sg in segs:
                a, b = sg["seg"]
                cv2.rectangle(vis, (a * 3, 0), (b * 3 - 1, vis.shape[0] - 1), (229, 135, 57), 1)
            d.update(kind="clock", class_title="Часы модели", images={"crop": png64(vis, 1.0)},
                     result={"text": c.get("text"), "raw": c.get("raw")},
                     chars=[{"glyph": png64((sg["glyph"] * 255).astype(np.uint8), 4),
                             "cands": [{"char": ch, "score": sc, "img": png64((t * 255).astype(np.uint8), 4)}
                                       for ch, sc, t in sg["cands"][:3]]} for sg in segs])
        return d

    # ── блок: как OpenCV находит его в кадре ─────────────────────────
    def block(self, name):
        fr = self.frame()
        if fr is None:
            return {"error": "нет кадра"}
        _, img, offsets, _ = fr
        rect = self.bank.blocks.get(name)
        if not rect:
            return {"error": f"нет блока {name}"}
        x, y, w, h = [int(v) for v in rect]
        tpl = self.bank.ref()[y:y + h, x:x + w]
        k = 4
        f = cv2.resize(img, (img.shape[1] // k, img.shape[0] // k), interpolation=cv2.INTER_AREA)
        t = cv2.resize(tpl, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
        t0 = clock()
        res = cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED)
        ms = (clock() - t0) * 1000
        _, mx, _, loc = cv2.minMaxLoc(res)
        sup = res.copy()
        rr = max(3, min(t.shape[:2]) // 2)
        cv2.rectangle(sup, (loc[0] - rr, loc[1] - rr), (loc[0] + rr, loc[1] + rr), -1.0, -1)
        second = float(sup.max())
        full = heat(res, -0.3, 1.0)
        cv2.drawMarker(full, loc, (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        small = f.copy()
        cv2.rectangle(small, loc, (loc[0] + t.shape[1], loc[1] + t.shape[0]), (229, 135, 57), 2)
        dx, dy = offsets.get(name, (0, 0))
        return {"name": name, "title": BLOCK_TITLE.get(name, name),
                "images": {"template": png64(t, 1.0), "frame": png64(small, 1.0), "heat": png64(full, 1.0)},
                "result": {"score": round(float(mx), 3), "second": round(second, 3),
                           "margin": round(float(mx) - second, 3), "coarse_ms": round(ms, 1),
                           "coarse_at": [loc[0] * k, loc[1] * k], "ref_at": [x, y],
                           "offset": [dx, dy], "scale": k, "tpl_size": [w, h]}}


# ───────────────────────────────────────────────────────────── HTTP

ACTIONS = {"start", "stop", "repair", "fault", "layout", "freq", "speed"}


def make_handler(vision, ctl):
    page = HERE / "cvdash.html"

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False,
                                  default=lambda v: v.item() if hasattr(v, "item") else str(v)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/vision":
                    return self._send(200, vision.snapshot())
                if u.path == "/frame.jpg":
                    fr = vision.frame(int(q["seq"]) if q.get("seq", "").isdigit() else None)
                    return self._send(200 if fr else 404, fr[0] if fr else b"", "image/jpeg")
                if u.path == "/api/detail":
                    return self._send(200, vision.detail(q.get("id", "")))
                if u.path == "/api/block":
                    return self._send(200, vision.block(q.get("name", "")))
                self._send(404, {"error": "нет такого адреса"})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                m = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/exact":
                    return self._send(200, vision.exact())
                if self.path == "/api/action":
                    act = m.get("action")
                    if act not in ACTIONS:
                        return self._send(400, {"error": f"нет действия {act}"})
                    if act == "speed":
                        ctl.call("scenario", action="speed", arg=float(m.get("arg")))
                    else:
                        ctl.call("scenario", action=act, arg=m.get("arg"))
                    return self._send(200, {"ok": True})
                if self.path == "/api/cache":
                    vision.reader.cache_on = bool(m.get("on"))
                    return self._send(200, {"cache": vision.reader.cache_on})
                self._send(404, {"error": "нет такого адреса"})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def main():
    ap = argparse.ArgumentParser(description="Дашборд распознавания щита ВМ")
    ap.add_argument("--source", default="http://127.0.0.1:8781/stream")
    ap.add_argument("--control", default="127.0.0.1:8790")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8775)
    a = ap.parse_args()
    src = Mjpeg(a.source)
    src.start()
    h, _, p = a.control.rpartition(":")
    ctl = Ctl((h or "127.0.0.1", int(p)))
    cdp = Cdp(local_port=9224)                 # свой туннель к DevTools ВМ
    vision = Vision(src, cdp, ctl)
    vision.start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(vision, ctl))
    srv.daemon_threads = True
    print(f"Дашборд распознавания: http://{a.host}:{a.port}/   (источник {a.source})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cdp.close()


if __name__ == "__main__":
    main()
