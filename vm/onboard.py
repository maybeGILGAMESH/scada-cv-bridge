# -*- coding: utf-8 -*-
"""Новая система: щит, для которого нет профиля распознавания.

Что можно сказать о незнакомом экране по одним пикселям, — и чего не хватает,
чтобы начать его читать:

  * сигнал есть, разрешение, кадр стабилен (не мерцает, не прокручивается);
  * не совпадает ли с известным профилем (шапка контрольного кадра);
  * автопоиск: панели (прямоугольники с рамкой), круглые приборы
    (HoughCircles), цветные числа (насыщенный цвет + форма строки), кнопки
    (заливка цветом), надписи (Tesseract, русский + английский);
  * чек-лист готовности и что делать дальше.
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def known_profiles():
    """Профили распознавания на диске: папки с manifest.json и контрольным кадром."""
    out = []
    for d in sorted(HERE.glob("cv_templates*")) + [HERE.parent / "cv_templates",
                                                   HERE.parent / "extracted" / "scada-cv-bridge-main" / "cv_templates"]:
        if (d / "manifest.json").exists() and (d / "reference" / "frame.png").exists():
            out.append(d)
    return out


def match_profiles(frame):
    """Шапка контрольного кадра каждого профиля → лучший балл matchTemplate."""
    res = []
    for d in known_profiles():
        ref = cv2.imread(str(d / "reference" / "frame.png"))
        if ref is None:
            continue
        head = ref[4:60, 8:620]
        r = cv2.matchTemplate(cv2.resize(frame, None, fx=0.5, fy=0.5),
                              cv2.resize(head, None, fx=0.5, fy=0.5), cv2.TM_CCOEFF_NORMED)
        res.append({"profile": d.name if d.parent == HERE else f"{d.parent.name}/{d.name}",
                    "score": round(float(r.max()), 3)})
    res.sort(key=lambda z: -z["score"])
    return res


def discover(frame, tess=None):
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = []
    # панели: прямоугольные контуры с рамкой
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 6000 or w > W * 0.98 or h > H * 0.98:
            continue
        if cv2.contourArea(c) / float(w * h) < 0.75:
            continue
        if any(abs(x - b[0]) < 6 and abs(y - b[1]) < 6 and abs(w - b[2]) < 12 and abs(h - b[3]) < 12 for b in boxes):
            continue
        boxes.append((x, y, w, h))
    for b in sorted(boxes, key=lambda b: -b[2] * b[3])[:40]:
        out.append({"type": "panel", "bbox": list(map(int, b)), "conf": 0.6})
    # круглые приборы
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.5, minDist=60,
                               param1=120, param2=60, minRadius=28, maxRadius=160)
    if circles is not None:
        for x, y, r in circles[0][:24]:
            out.append({"type": "gauge?", "bbox": [int(x - r), int(y - r), int(2 * r), int(2 * r)], "conf": 0.5})
    # цветные числа и кнопки: насыщенные области
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 120)).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(sat, np.ones((3, 7), np.uint8)))
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 40 or h < 6:
            continue
        fill = a / float(w * h)
        if 8 <= h <= 34 and w >= 1.2 * h and fill < 0.65:
            out.append({"type": "value?", "bbox": [int(x), int(y), int(w), int(h)], "conf": 0.45})
        elif 22 <= h <= 70 and 50 <= w <= 220 and fill >= 0.65:
            out.append({"type": "button?", "bbox": [int(x), int(y), int(w), int(h)], "conf": 0.45})
    texts = []
    if tess is not None:
        small = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        g = 255 - g                                      # светлый текст на тёмном → тёмный на светлом
        tsv = tess(g, "rus+eng", 11, tsv=True)
        for ln in tsv.splitlines()[1:]:
            p = ln.split("\t")
            if len(p) < 12 or not p[11].strip():
                continue
            try:
                conf = float(p[10])
            except ValueError:
                continue
            if conf < 60 or len(p[11].strip()) < 2:
                continue
            x, y, w, h = [int(int(v) / 1.5) for v in p[6:10]]
            texts.append({"text": p[11].strip(), "bbox": [x, y, w, h], "conf": round(conf / 100, 2)})
    return out, texts


class Onboard(threading.Thread):
    """Анализ незнакомого щита в фоне: раз в несколько секунд — быстрые
    признаки, раз в полминуты — автопоиск объектов и надписей."""

    def __init__(self, system, src, tess):
        super().__init__(daemon=True)
        self.system, self.src, self.tess = system, src, tess
        self.lock = threading.Lock()
        self.state = {"ready": False, "kind": "onboarding"}
        self.frames = {}
        self.seq = 0
        self.prev = None
        self.cands, self.texts, self.t_disc = [], [], 0.0
        self.profiles = []
        self.force = False

    def run(self):
        while not getattr(self, "stopped", False):
            try:
                self.step()
            except Exception as e:
                with self.lock:
                    self.state["error"] = f"{type(e).__name__}: {e}"
            time.sleep(1.0)

    def step(self):
        img, t_enc, t_recv = self.src.next_after(time.perf_counter() - 2)
        if img is None:
            with self.lock:
                self.state.update(ready=False, source_error=self.src.error or "нет кадров")
            return
        H, W = img.shape[:2]
        diff = float(np.abs(img.astype(np.int16) - self.prev.astype(np.int16)).mean()) if self.prev is not None and self.prev.shape == img.shape else None
        self.prev = img
        now = time.time()
        if self.force or now - self.t_disc > 30 or not self.profiles:
            self.profiles = match_profiles(img)
            self.cands, self.texts = discover(img, self.tess)
            self.t_disc, self.force = now, False
        best = self.profiles[0] if self.profiles else None
        known = best is not None and best["score"] > 0.8
        counts = {}
        for c in self.cands:
            counts[c["type"]] = counts.get(c["type"], 0) + 1
        checklist = [
            {"text": "Видеосигнал с экрана ВМ", "status": "done", "detail": f"{W}×{H}"},
            {"text": "Кадр стабилен (не мерцает, не прокручивается)",
             "status": "done" if diff is not None and diff < 6 else ("pending" if diff is None else "fail"),
             "detail": None if diff is None else f"среднее отличие соседних кадров {diff:.1f}"},
            {"text": "Совпадает с известным профилем распознавания",
             "status": "done" if known else "fail",
             "detail": f"лучший: {best['profile']} — балл {best['score']:.2f} (нужно > 0,8)" if best else "профилей нет"},
            {"text": "Найдены области щита (панели, приборы, числа, кнопки)",
             "status": "done" if len(self.cands) >= 8 else "pending",
             "detail": ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "—"},
            {"text": "Прочитаны надписи (названия параметров)",
             "status": "done" if len(self.texts) >= 10 else "pending", "detail": f"{len(self.texts)} надписей"},
            {"text": "Профиль распознавания: эталоны объектов", "status": "fail",
             "detail": "нет — без эталонов показания не читаются"},
            {"text": "Проверка на контрольных кадрах", "status": "blocked", "detail": "после эталонов"},
        ]
        done = sum(c["status"] == "done" for c in checklist)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.seq += 1
        objs = [{"id": f"cand{i}", "cls": c["type"], "label": c["type"], "at": c["bbox"][:2], "size": c["bbox"][2:],
                 "read": None, "truth": None, "ok": None, "score": c["conf"], "candidate": True}
                for i, c in enumerate(self.cands)]
        with self.lock:
            self.frames[self.seq] = (jpg.tobytes(), img, {}, None)
            for k in [k for k in self.frames if k < self.seq - 3]:
                del self.frames[k]
            self.state = {
                "ready": True, "kind": "onboarding", "seq": self.seq, "t": now, "frame_size": [W, H],
                "objects": objs, "texts": self.texts[:300], "blocks": [], "trends": {},
                "onboarding": {"checklist": checklist, "progress": [done, len(checklist)],
                               "profiles": self.profiles, "counts": counts,
                               "summary": f"Профиля распознавания для этого щита нет. Готово {done} из {len(checklist)} "
                                          f"шагов подготовки; найдено областей {len(self.cands)}, надписей {len(self.texts)}."},
                "fps": None, "timing": {}, "error": None, "source_error": self.src.error,
            }

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def frame(self, seq=None):
        with self.lock:
            if not self.frames:
                return None
            return self.frames.get(seq) or self.frames[max(self.frames)]
