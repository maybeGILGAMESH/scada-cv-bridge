# -*- coding: utf-8 -*-
"""Чтение щита в реальном времени: те же приёмы, что в match_templates.py,
плюс три ускорения, без которых живой контур не замыкается.

Исходный конвейер (live/recognize.py) читает кадр за 1,2–1,4 с, первый —
за 7 с. Почти всё это время уходит на шесть манометров: каждый ищется
перебором 55 шаблонов по углу, и сопоставление идёт с маской, а с маской
OpenCV не умеет считать через БПФ. Остальные 76 объектов вместе стоят
около 0,45 с.

1. Манометр по стрелке, а не по шаблону. Центр циферблата известен из
   контрольного кадра и сдвигается туда, где нашёлся блок. Из центра
   пускаются лучи с шагом 0,5°, вдоль каждого считается «краснота»
   стрелки (R − max(G, B)) на радиусах 25–72 % — внутри шкалы, но до
   красной зоны и цифр. Угол стрелки — центр тяжести пика. Это одна
   cv2.remap на прибор вместо 55 сопоставлений. Если пик не уверенный
   (стрелку чем-то закрыло), честный откат на перебор шаблонов.

2. Неизменившиеся объекты не перечитываются. Для каждого объекта хранится
   уменьшенный отпечаток его области; если область не изменилась и блок
   не сдвинулся, берётся прошлый результат. Раз в refresh_every кадров
   всё перечитывается целиком, чтобы кэш не закостенел.

3. Объекты читаются параллельно. cv2.matchTemplate и remap отпускают GIL,
   поэтому пул потоков реально занимает ядра.

Результат по каждому объекту — как у match_templates.recognise, плюс
"at" (где объект найден в кадре) — это нужно управлению, чтобы знать,
куда нажимать.
"""

import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cv_templates"))

import match_templates as MT      # noqa: E402

CLASS_TITLE = {
    "gauge": "Манометры", "sevenseg_digit": "Цифровые индикаторы",
    "bargraph": "Столбчатые индикаторы", "vessel_level": "Уровни в аппаратах",
    "lamp": "Сигнальные лампы", "annunciator": "Щит тревог", "valve": "Арматура",
    "pump": "Насосы", "apparatus": "Блочные аппараты", "button": "Кнопки пульта",
    "selector": "Переключатели", "status_text": "Состояние",
    "text_readout": "Текстовые табло", "trend": "Тренды",
}
GROUP_ORDER = ["gauge", "sevenseg_digit", "bargraph", "vessel_level", "annunciator",
               "lamp", "status_text", "valve", "pump", "apparatus",
               "text_readout", "selector", "button", "trend"]


# ───────────────────────────────────────────────────── стрелка манометра

class NeedleGauge:
    """Угол стрелки по лучевому профилю красноты."""

    def __init__(self, center, radius, start_deg=135.0, sweep_deg=270.0,
                 step=0.5, r_lo=0.25, r_hi=0.72, margin=8.0):
        self.center = (float(center[0]), float(center[1]))
        self.start, self.sweep = float(start_deg), float(sweep_deg)
        # Немного за края шкалы: стрелка у упора должна давать пик целиком,
        # иначе центр тяжести пика съезжает внутрь шкалы.
        self.angles = np.arange(start_deg - margin, start_deg + sweep_deg + margin + 1e-9, step)
        radii = np.arange(r_lo * radius, r_hi * radius, 1.0)
        th = np.deg2rad(self.angles)[:, None]
        self.dx = (np.cos(th) * radii[None, :]).astype(np.float32)
        self.dy = (np.sin(th) * radii[None, :]).astype(np.float32)
        self.step = step

    def read(self, frame, shift=(0, 0)):
        cx = self.center[0] + shift[0]
        cy = self.center[1] + shift[1]
        polar = cv2.remap(frame, self.dx + np.float32(cx), self.dy + np.float32(cy),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        p = polar.astype(np.int16)
        red = p[:, :, 2] - np.maximum(p[:, :, 0], p[:, :, 1])
        # доля «красных» точек вдоль луча — устойчивее среднего к бликам
        prof = (red > 70).mean(axis=1)
        prof = np.convolve(prof, np.ones(3) / 3.0, mode="same")
        k = int(np.argmax(prof))
        peak = float(prof[k])
        if peak < 0.25:
            return None
        # центр тяжести пика по точкам выше половины максимума
        lo = k
        while lo > 0 and prof[lo - 1] >= peak * 0.5:
            lo -= 1
        hi = k
        while hi < len(prof) - 1 and prof[hi + 1] >= peak * 0.5:
            hi += 1
        w = prof[lo:hi + 1]
        ang = float(np.sum(self.angles[lo:hi + 1] * w) / np.sum(w))
        # второй пик далеко от первого — значит красное не только стрелка
        mask = np.ones_like(prof, bool)
        mask[max(0, lo - 10):hi + 11] = False
        second = float(prof[mask].max()) if mask.any() else 0.0
        width_deg = (hi - lo + 1) * self.step
        return {"angle_deg": ang, "peak": peak, "second": second, "width_deg": width_deg}


# ───────────────────────────────────────────────────── чтение кадра

class FastReader:
    def __init__(self, templates_dir, only=None, register_every=8,
                 gauge_method="needle", cache=True, refresh_every=10,
                 workers=None, change_thr=1.2):
        self.bank = MT.Bank(Path(templates_dir))
        self.only = set(only.split(",")) if isinstance(only, str) else (set(only) if only else None)
        self.register_every = max(1, int(register_every))
        self.gauge_method = gauge_method
        self.cache_on = bool(cache)
        self.refresh_every = max(1, int(refresh_every))
        self.change_thr = float(change_thr)
        self.pool = ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 4))

        self._reg = None
        self._since = 0
        self._watch = 0
        self._frame_no = 0
        self._hints = {}
        self._cache = {}               # объект → (отпечаток, сдвиг блока, результат)
        self.layouts = self.bank.man.get("layouts") or {}
        self.block_titles = self.bank.man.get("block_titles") or {}

        # Блоки без приборов (журнал событий) не привязываются: читать в них
        # нечего, а текст в них меняется и только портил бы балл привязки.
        # (принадлежность к блокам берётся из разметки, а в ней есть и объекты
        # без эталонов — например, баннер журнала; считаем только читаемые)
        used = {self.bank.block_of[o] for o in self.bank.by_obj if o in self.bank.block_of}
        self.bank.blocks = {k: v for k, v in self.bank.blocks.items() if k in used}

        # Десятичная точка индикатора — свойство прибора, а не показания:
        # её место берётся из разметки контрольного кадра.
        self.decimals = {}
        for o in self.bank.ref_ann.get("objects", []):
            if o.get("type") == "sevenseg" and "." in (o.get("text") or ""):
                self.decimals[o["id"]] = len(o["text"].split(".", 1)[1])

        # геометрия циферблатов из контрольного кадра
        self.needles = {}
        scale = self.bank.man.get("gauge_scale") or {"start_deg": 135, "sweep_deg": 270}
        for o in self.bank.ref_ann.get("objects", []):
            if o.get("type") == "gauge" and o.get("center") and o.get("radius"):
                self.needles[o["id"]] = NeedleGauge(o["center"], o["radius"],
                                                    scale["start_deg"], scale["sweep_deg"])

        # область каждого объекта в эталонной компоновке — для отпечатков
        self.rois = {}
        for oid, recs in self.bank.by_obj.items():
            ax, ay = recs[0]["anchor"]
            w = max(r["w"] for r in recs)
            h = max(r["h"] for r in recs)
            self.rois[oid] = (int(ax), int(ay), int(w), int(h))

        # прогрев: шаблоны грузятся с диска лениво, в пуле это дало бы гонки
        for recs in self.bank.by_obj.values():
            for r in recs:
                self.bank.img(r)
                self.bank.mask(r)
        self.bank.ref()

    # ── привязка блоков ─────────────────────────────────────────────
    def _register(self, frame, force):
        """Как в live/recognize.py, с двумя поправками для живого потока.

        Качество привязки — медиана по блокам, а не худший блок. У столбчатых
        балл против контрольного кадра падает до 0,18, когда все шкалы полные
        (блок почти целиком — заливка), и по худшему блоку поиск запускался бы
        на каждом кадре, хотя блоки стоят на месте.

        Перестановку выдаёт сдвиг при проверке, а не низкий балл: проверяемый
        кусок ищется в окрестности ±8 px, и переставленный блок даёт либо
        сдвиг, либо мусорный балл ниже 0,25.
        """
        need = force or self._reg is None or self._since >= self.register_every - 1
        if not need:
            names = sorted(self._reg)
            name = names[self._watch % len(names)]
            self._watch += 1
            check = MT.verify_block(frame, self.bank, name, self._reg[name][:2])
            if check is None or check[0] < 0.25 or abs(check[1]) > 2 or abs(check[2]) > 2:
                need = True
        if need:
            self._reg, self._since = MT.register_blocks(frame, self.bank), 0
        else:
            self._since += 1
        self.bank.set_offsets(MT.offsets_only(self._reg))
        scores = sorted(v[2] for v in self._reg.values())
        quality = scores[len(scores) // 2] if scores else 0.0
        return need, quality

    # ── отпечаток области ───────────────────────────────────────────
    def _thumb(self, frame, oid):
        x, y, w, h = self.rois[oid]
        dx, dy = self.bank.shift(oid)
        x, y = x + dx, y + dy
        H, W = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = frame[y0:y1, x0:x1]
        return cv2.resize(crop, (max(1, (x1 - x0) // 4), max(1, (y1 - y0) // 4)),
                          interpolation=cv2.INTER_AREA).astype(np.int16)

    # ── один объект ─────────────────────────────────────────────────
    def _read_one(self, frame, oid, recs):
        cls = self.bank.classes[oid]
        kind = MT.CLASS_READER.get(cls, "locate")
        if kind == "gauge":
            r = self._read_gauge(frame, oid, recs)
        elif kind == "sevenseg":
            r = MT.read_sevenseg(frame, self.bank, recs, self.bank.cells_of(oid))
            dec = self.decimals.get(oid)
            if dec and r.get("value") is not None and r["text"].isdigit():
                r["value"] = int(r["text"]) / 10 ** dec
                r["display"] = r["text"][:-dec] + "." + r["text"][-dec:]
        elif kind == "bargraph":
            r = MT.read_bargraph(frame, self.bank, recs)
        elif kind == "vessel":
            r = MT.read_vessel(frame, self.bank, recs)
        elif kind == "discrete":
            r = MT.read_discrete(frame, self.bank, recs)
        else:
            s, x, y = MT.match(frame, self.bank.img(recs[0]), None, self.bank.anchor(recs[0]), 36)
            r = {"score": s, "at": (x, y)}
        r.update(object_id=oid, cls=cls, label=recs[0].get("label"))
        if "at" not in r:
            ax, ay = self.bank.anchor(recs[0])
            r["at"] = (int(ax), int(ay))
        r["size"] = (int(recs[0]["w"]), int(recs[0]["h"]))
        return r

    def _read_gauge(self, frame, oid, recs):
        ng = self.needles.get(oid)
        if self.gauge_method == "needle" and ng is not None:
            res = ng.read(frame, self.bank.shift(oid))
            if res and res["second"] < res["peak"] * 0.6:
                rec = recs[0]
                sc = self.bank.man["gauge_scale"]
                frac = float(np.clip((res["angle_deg"] - sc["start_deg"]) / sc["sweep_deg"], 0, 1))
                return {"value": rec["min"] + (rec["max"] - rec["min"]) * frac,
                        "angle_deg": res["angle_deg"], "score": res["peak"],
                        "margin": res["peak"] - res["second"], "unit": rec.get("unit"),
                        "method": "needle"}
        r = MT.read_gauge(frame, self.bank, recs, hint=self._hints.get(oid))
        r["method"] = "template"
        return r

    # ── кадр целиком ────────────────────────────────────────────────
    def read(self, frame, force_register=False):
        t0 = time.perf_counter()
        self._frame_no += 1
        registered, quality = self._register(frame, force_register)
        t_reg = time.perf_counter() - t0

        # Кэш освежается по кругу: на каждом кадре заново читается своя десятая
        # часть объектов, даже если их область не менялась. Перечитывать всё
        # разом раз в десять кадров значило бы пик в полсекунды. Поиск блоков
        # кэш не сбрасывает: сдвиг блока входит в ключ кэша.
        todo, rows, thumbs = [], [], {}
        for k, (oid, recs) in enumerate(self.bank.by_obj.items()):
            full = not self.cache_on or (k + self._frame_no) % self.refresh_every == 0
            cls = self.bank.classes[oid]
            kind = MT.CLASS_READER.get(cls, "locate")
            if self.only and cls not in self.only and kind not in self.only:
                continue
            if self.cache_on:
                th = self._thumb(frame, oid)
                thumbs[oid] = th
                c = self._cache.get(oid)
                if (not full and c is not None and th is not None and c[0] is not None
                        and c[0].shape == th.shape and c[1] == self.bank.shift(oid)
                        and float(np.abs(th - c[0]).mean()) < self.change_thr):
                    rows.append(dict(c[2], cached=True))
                    continue
            todo.append((oid, recs))

        fresh = list(self.pool.map(lambda t: self._read_one(frame, *t), todo))
        for r in fresh:
            oid = r["object_id"]
            if r["cls"] == "gauge" and r.get("angle_deg") is not None:
                self._hints[oid] = r["angle_deg"]
            if self.cache_on:
                self._cache[oid] = (thumbs.get(oid), self.bank.shift(oid), r)
        rows.extend(fresh)
        rows.sort(key=lambda z: (z["cls"], z["object_id"]))
        t_all = time.perf_counter() - t0
        return {
            "rows": rows,
            "registration": self._reg,
            "registered_this_frame": bool(registered),
            "registration_quality": round(float(quality), 4),
            "read_objects": len(fresh), "cached_objects": len(rows) - len(fresh),
            "timing_ms": {"registration": round(t_reg * 1000), "total": round(t_all * 1000)},
        }

    # ── как называется компоновка (для сайта) ───────────────────────
    def layout_name(self, tol=24):
        """Сколько блоков переставлено. Сдвиг до tol px — не перестановка:
        у столбчатых при полных шкалах поиск блока ошибается на 10–15 px
        (блок почти целиком — заливка), приборы внутри при этом читаются
        верно, их окрестность поиска шире."""
        reg = self._reg or {}
        if not reg:
            return {"id": None, "name": "блоки не найдены", "moved": 0}
        moved = sum(1 for v in reg.values() if abs(v[0]) > tol or abs(v[1]) > tol)
        return {"id": None, "moved": moved,
                "name": "базовая" if not moved else f"переставлено блоков: {moved}"}

    def close(self):
        self.pool.shutdown(wait=False)


def pack_rows(rows, bank=None):
    """Строки результата → группы для сайта и агрегатора (JSON-совместимо)."""
    groups = {}
    for r in rows:
        item = {"id": r["object_id"], "label": r.get("label") or r["object_id"],
                "score": round(float(r.get("score") or 0.0), 4)}
        if bank is not None:
            item["block"] = bank.block_of.get(r["object_id"])
        for k in ("value", "text", "state", "level_pct", "unit", "angle_deg", "fill_frac",
                  "margin", "reliable", "opening_pct", "method", "cached", "display"):
            v = r.get(k)
            if v is not None:
                if isinstance(v, (float, np.floating)):
                    v = round(float(v), 3)
                item[k] = v
        if r.get("at") is not None:
            item["at"] = [int(r["at"][0]), int(r["at"][1])]
            item["size"] = list(r.get("size") or (0, 0))
        groups.setdefault(r["cls"], []).append(item)
    for lst in groups.values():
        lst.sort(key=lambda z: z["id"])
    return groups
