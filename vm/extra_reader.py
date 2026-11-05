# -*- coding: utf-8 -*-
"""То, чего FastReader не читает: текст и графики.

FastReader у текстовых табло и трендов только находит место (класс "locate"),
а журнал событий не трогает вовсе. Здесь — чтение по пикселям:

  * табло (text_readout): значение — насыщенно-цветные цифры моноширинного
    шрифта, подписи и единицы серые. Маска по насыщенности вырезает значение,
    Tesseract читает одну строку из цифр;
  * журнал: две последние строки — время, источник, сообщение (Tesseract,
    русский и английский);
  * часы модели в шапке;
  * тренды: шкала Y — по подписям оси (Tesseract), кривые — по цвету серии
    из легенды, столбец за столбцом. Получается клон графика: вся кривая и
    последнее значение (точка на правом краю).

Геометрия (где табло, где поле графика, цвета серий из легенды) берётся из
контрольного кадра банка эталонов и сдвигается вместе с найденным блоком —
так же, как центры манометров в FastReader. Модель щита не используется.
"""

import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TESSDATA = HERE / "tessdata"


# ───────────────────────────────────────────────────────────── OCR

class Tess:
    """Tesseract из командной строки, с кэшем по содержимому картинки:
    табло меняются редко, повторный кадр не стоит процесса."""

    def __init__(self, cache_size=4000):
        # русская модель — в vm/tessdata (fetch_tessdata.sh); нет её — системные модели
        self.env = dict(os.environ, OMP_THREAD_LIMIT="1")
        if (TESSDATA / "rus.traineddata").exists():
            self.env["TESSDATA_PREFIX"] = str(TESSDATA)
        self.cache = {}
        self.lock = threading.Lock()
        self.cache_size = cache_size
        self.calls = 0

    def __call__(self, img, lang="eng", psm=7, whitelist=None, tsv=False):
        ok, png = cv2.imencode(".png", img)
        key = hashlib.md5(png.tobytes() + f"{lang}{psm}{whitelist}{tsv}".encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        cmd = ["tesseract", "stdin", "stdout", "-l", lang, "--psm", str(psm), "--dpi", "300"]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        if tsv:
            cmd += ["-c", "tessedit_create_tsv=1"]
        out = subprocess.run(cmd, input=png.tobytes(), capture_output=True, env=self.env, timeout=10)
        text = out.stdout.decode("utf-8", "replace")
        text = text if tsv else text.strip()
        with self.lock:
            self.calls += 1
            if len(self.cache) > self.cache_size:
                self.cache.clear()
            self.cache[key] = text
        return text


def ink(crop, scale=4):
    """Светлый текст на тёмном → чёрный текст на белом, увеличенный.
    Яркость — максимум по каналам: цветные цифры не теряются в сером."""
    g = crop.max(axis=2) if crop.ndim == 3 else crop
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = 255 - b
    return cv2.copyMakeBorder(b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def color_mask(crop, s_min=90, v_min=150):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min)


def bbox_of(mask, pad=2):
    ys, xs = np.nonzero(mask)
    if xs.size < 6:
        return None
    return (max(0, xs.min() - pad), max(0, ys.min() - pad), xs.max() + pad + 1, ys.max() + pad + 1)


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def colored_text(frame, rect, gray_ok=False):
    """Насыщенно-цветная надпись внутри прямоугольника → (вырезка, рамка в кадре)."""
    x, y, w, h = [int(round(v)) for v in rect]
    area = frame[max(0, y):y + h, max(0, x):x + w]
    if area.size == 0:
        return None, None
    bb = bbox_of(color_mask(area))
    if bb is None and gray_ok:
        bb = bbox_of(area.max(axis=2) > 90)
    if bb is None:
        return None, None
    bx0, by0, bx1, by1 = bb
    bright = (area[by0:by1].max(axis=2) > 70).any(axis=0)
    while bx0 > 0 and bright[max(0, bx0 - 5):bx0].any():
        bx0 -= 1
    while bx1 < len(bright) and bright[bx1:bx1 + 5].any():
        bx1 += 1
    return area[by0:by1, bx0:bx1], [max(0, x) + bx0, max(0, y) + by0, bx1 - bx0, by1 - by0]


def bar_value_rect(bbox):
    """Число справа от шкалы: правое поле шириной 84 px, строка у нижнего края."""
    x, y, w, h = bbox
    return [x + w - 90, y + h - 27, 84, 23]          # левее — край заливки шкалы


def well_geometry(o):
    """Ствол скважины из разметки контрольного кадра: устье (глубина 0) и забой
    по вертикали, место риски «Ндин» по горизонтали."""
    lx, ly, lw, lh = o["liquid_bbox"]
    bot = ly + lh
    lvl = max(1e-3, o["level_pct"] / 100.0)
    top = bot - lh / lvl
    x_mark0, x_mark1 = lx - 2 - 12, lx + lw + 2 + 12
    return {"top": top, "bot": bot, "x0": x_mark0, "x1": x_mark1, "label_x": lx + lw + 2 + 14,
            "depth": o["value"] / max(1e-3, 1 - lvl)}


def value_crop(frame, bbox):
    """Значение табло: нижние две трети без рамки, по насыщенному цвету.
    → (вырезка значения, её прямоугольник в кадре, «серое» — уставка в АВТО)"""
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0 = x + 2, y + 2 + (h - 4) // 3
    lower = frame[max(0, y0):y + h - 2, max(0, x + 2):x + w - 2]
    if lower.size == 0:
        return None, None, False
    bb = bbox_of(color_mask(lower))
    gray = False
    if bb is None:                       # уставка в «АВТО» — значение серое, единица справа
        m = lower.max(axis=2) > 70
        m[:, int(lower.shape[1] * 0.80):] = False
        bb = bbox_of(m)
        gray = True
    if bb is None:
        return None, None, gray
    bx0, by0, bx1, by1 = bb
    # В MJPEG цвет края символа бледнеет (4:2:0), и маска по насыщенности теряет
    # крайние цифры. Расширяем вырезку по яркости в той же строке: соседние
    # яркие столбцы с разрывом до 4 px — это ещё значение, дальше — единица.
    bright = (lower[by0:by1].max(axis=2) > 70).any(axis=0)
    while bx0 > 0 and bright[max(0, bx0 - 5):bx0].any():
        bx0 -= 1
    while bx1 < len(bright) and bright[bx1:bx1 + 5].any() and not gray:
        bx1 += 1
    return lower[by0:by1, bx0:bx1], [x0 + bx0, y0 + by0, bx1 - bx0, by1 - by0], gray


# ───────────────────────────────────────────────────────────── чтение

class ExtraReader:
    def __init__(self, templates_dir, workers=6):
        ref = json.loads((Path(templates_dir) / "reference" / "frame.json").read_text(encoding="utf-8"))
        self.ref = {o["id"]: o for o in ref["objects"]}
        self.block_of = {o["id"]: o.get("block") for o in ref["objects"]}
        self.tess = Tess()
        from glyph_ocr import GlyphBank
        self.glyphs = GlyphBank(Path(templates_dir) / "glyphs.npz")
        self.tags = []                 # эталоны источников журнала (glyph_live.py)
        tj = Path(templates_dir) / "log_tags.json"
        if tj.exists():
            from glyph_ocr import cut
            for t in json.loads(tj.read_text(encoding="utf-8")):
                v = self._tag_ink(cv2.imread(str(Path(templates_dir) / t["image"])))
                if v is not None:
                    self.tags.append((t["tag"], v))
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.readouts = [o for o in ref["objects"] if o["type"] == "text_readout"]
        self.bars = [o for o in ref["objects"] if o["type"] == "bargraph"]
        self.well = self.ref.get("mimic_well")
        self.trends = [o for o in ref["objects"] if o["type"] == "trend"]
        # Заголовки блоков — неизменные надписи в левом верхнем углу. У блоков с
        # повторяющимся узором (шесть одинаковых шкал, сетка табло) сопоставление
        # блока целиком неоднозначно и сдвигается на шаг узора; заголовок — нет.
        import json as _j
        man = _j.loads((Path(templates_dir) / "manifest.json").read_text(encoding="utf-8"))
        self.block_rects = man.get("blocks") or {}
        self.titles = {}
        self.legend = {}               # тренд → цвета серий (BGR), снятые с легенды контрольного кадра
        ref_img = cv2.imread(str(Path(templates_dir) / "reference" / "frame.png"))
        # журнал FastReader не привязывает (в полосе нет эталонов приборов), а при
        # перестановке компоновки полоса переезжает. Опора — неизменная левая часть
        # полосы с надписью «ЖУРНАЛ»: ищется matchTemplate, как блоки.
        lb = self.ref.get("log_banner")
        self.log_anchor = None
        self.log_off = (0, 0)
        if lb is not None:
            x, y, w, h = [int(v) for v in lb["bbox"]]
            self.log_anchor = (x, y, ref_img[y:y + h, x:x + 66].copy())
        for t in self.trends:
            self.legend[t["id"]] = self._legend_colors(ref_img, t)
        for name in ("bars", "right", "console", "mimic", "trends"):
            r = self.block_rects.get(name)
            if r:
                x, y = int(r[0]), int(r[1])
                self.titles[name] = (x + 8, y + 6, ref_img[y + 6:y + 28, x + 8:x + 8 + min(220, int(r[2]) - 16)].copy())

    # сдвиг объекта = сдвиг его блока
    def _shift(self, oid, offsets):
        return offsets.get(self.block_of.get(oid), (0, 0))

    def _legend_colors(self, img, t):
        """Легенда под графиком: цветные штрихи 11×3 px. Цвет серии — медиана
        насыщенных пикселей каждого штриха, слева направо."""
        x, y, w, h = [int(v) for v in t["bbox"]]
        strip = img[y + h - 16:y + h - 6, x + 40:x + w - 4]
        m = color_mask(strip, 80, 120)
        cols = np.where(m.any(axis=0))[0]
        groups, cur = [], []
        for c in cols:
            if cur and c - cur[-1] > 2:
                groups.append(cur)
                cur = []
            cur.append(c)
        if cur:
            groups.append(cur)
        out = []
        for g in groups:
            if len(g) < 6:           # буквы подписей не цветные; штрих — 11 px
                continue
            px = strip[:, g[0]:g[-1] + 1][m[:, g[0]:g[-1] + 1]]
            out.append(np.median(px, axis=0).astype(int).tolist())
        return out[:len(t.get("series") or out)]

    # ── табло ───────────────────────────────────────────────────────
    def read_readout(self, frame, o, offsets):
        dx, dy = self._shift(o["id"], offsets)
        x, y, w, h = [int(round(v)) for v in o["bbox"]]
        x, y = x + dx, y + dy
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        val, box, gray = value_crop(frame, [x, y, w, h])
        if val is None:
            return {"id": o["id"], "text": None, "value": None, "box": None, "_crop": crop}
        text, info, segs, _ = self.glyphs.read(val, allowed=set("0123456789.-"))
        ok = bool(text) and NUM.fullmatch(text.replace(" ", "")) is not None
        text = text.replace(" ", "") if ok else None
        return {"id": o["id"], "text": text, "value": float(text) if text else None,
                "chars": info, "box": box, "locked": gray,
                "_crop": crop, "_val": val, "_segs": segs}

    def anchor_blocks(self, frame, reg):
        """Поправить сдвиги блоков по заголовкам. reg: {блок: (dx, dy, балл)} — правится на месте.
        → список исправленных блоков."""
        fixed = []
        for name, (tx, ty, tpl) in self.titles.items():
            if name not in reg or tpl.size == 0:
                continue
            dx, dy = reg[name][0], reg[name][1]
            th, tw = tpl.shape[:2]
            x0, y0 = max(0, tx + dx - 40), max(0, ty + dy - 150)
            roi = frame[y0:ty + dy + th + 150, x0:tx + dx + tw + 40]
            if roi.shape[0] < th or roi.shape[1] < tw:
                continue
            r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            ndx, ndy = x0 + loc[0] - tx, y0 + loc[1] - ty
            if mx >= 0.75 and (abs(ndx - dx) > 3 or abs(ndy - dy) > 3):
                reg[name] = (int(ndx), int(ndy), max(float(reg[name][2]), float(mx)))
                fixed.append(name)
        return fixed

    # ── шкалы: число справа — точнее заливки и видно за пределом шкалы ─
    def read_bar(self, frame, o, offsets):
        dx, dy = self._shift(o["id"], offsets)
        x, y, w, h = [int(round(v)) for v in o["bbox"]]
        val, box = colored_text(frame, bar_value_rect([x + dx, y + dy, w, h]))
        if val is None:
            return {"id": o["id"], "text": None, "value": None, "box": None}
        text, info, _, _ = self.glyphs.read(val, allowed=set("0123456789.-"))
        text = text.replace(" ", "")
        ok = NUM.fullmatch(text) is not None
        return {"id": o["id"], "text": text if ok else None, "value": float(text) if ok else None,
                "box": box, "chars": info, "_val": val}

    # ── ствол скважины: уровень по риске «Ндин» и подписи ───────────
    def read_well(self, frame, offsets):
        o = self.well
        if not o or "liquid_bbox" not in o:
            return None
        g = well_geometry(o)
        dx, dy = self._shift("mimic_well", offsets)
        top, bot = int(g["top"] + dy), int(g["bot"] + dy)
        x0, x1 = int(g["x0"] + dx), int(g["x1"] + dx)
        # полоса шире ствола вправо: там подпись «Ндин N м» — второй признак риски
        band = frame[max(0, top - 6):bot + 6, max(0, x0):x1 + 80]
        if band.size == 0:
            return None
        W0 = x1 - max(0, x0)
        # Риска — жёлтая горизонталь (#f7b731), выходящая за обсадную колонну на
        # 12 px с обеих сторон. Именно концы снаружи колонны её и выдают: внутри
        # ствола есть и другое жёлтое (жидкость, подписи). Порог мягкий — в JPEG
        # тонкая линия бледнеет.
        # как у трендов: яркость — из кадра, оттенок — из сглаженного цвета (MJPEG 4:2:0)
        ycc = cv2.cvtColor(band, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ch = cv2.blur(ycc[:, :, 1:], (3, 3)) - 128.0
        ref = cv2.cvtColor(np.uint8([[[49, 183, 247]]]), cv2.COLOR_BGR2YCrCb)[0, 0, 1:].astype(np.float32) - 128.0
        cos = (ch @ ref) / (np.linalg.norm(ch, axis=2) * np.linalg.norm(ref) + 1e-3)
        amber = (ycc[:, :, 0] > 110) & (cos > 0.85) & (np.linalg.norm(ch, axis=2) > 10)
        ends = np.minimum(amber[:, :10].mean(axis=1), amber[:, W0 - 10:W0].mean(axis=1))
        # Стрелки притока у забоя тоже жёлтые и выходят за колонну, но подписи справа
        # у них нет; у риски — «Ндин N м» в пределах ±7 строк.
        lab = amber[:, W0 + 3:W0 + 75].sum(axis=1).astype(np.float32)
        lab = np.convolve(lab, np.ones(15, np.float32), mode="same")
        ends = ends * np.where(lab >= 25, 1.0, 0.3)
        k = int(np.argmax(ends))
        if ends[k] < 0.2:                                           # ложные (стрелки притока у забоя) — 0,1
            return {"level_pct": None, "mark_y": None, "box": [x0, top, x1 - x0, bot - top]}
        sel = np.nonzero(ends >= ends[k] * 0.5)[0]
        sel = sel[np.abs(sel - k) <= 2]
        w_ = ends[sel]
        mark_y = max(0, top - 6) + float((sel * w_).sum() / w_.sum())
        level = 100.0 * (bot - mark_y) / max(1.0, bot - top)
        # Подпись «Ндин N м» лежит поверх шкалы глубины — для чтения ненадёжна.
        # Глубину даёт сама риска: шкала ствола линейна от устья до забоя.
        h_din = g["depth"] * (1 - level / 100.0)
        return {"level_pct": float(np.clip(level, 0, 100)), "mark_y": mark_y,
                "h_din": h_din, "depth": g["depth"], "box": [x0, top, x1 - x0, bot - top]}

    # ── журнал ──────────────────────────────────────────────────────
    def locate_log(self, frame):
        """Сдвиг полосы журнала: сначала прежнее место (±6 px), иначе — весь кадр."""
        if self.log_anchor is None:
            return (0, 0), 0.0
        x, y, tpl = self.log_anchor
        th, tw = tpl.shape[:2]
        ox, oy = self.log_off
        x0, y0 = max(0, x + ox - 6), max(0, y + oy - 6)
        roi = frame[y0:y0 + th + 12, x0:x0 + tw + 12]
        if roi.shape[0] >= th and roi.shape[1] >= tw:
            r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx >= 0.8:
                self.log_off = (x0 + loc[0] - x, y0 + loc[1] - y)
                return self.log_off, float(mx)
        r = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx >= 0.6:
            self.log_off = (loc[0] - x, loc[1] - y)
        return self.log_off, float(mx)

    def read_log(self, frame, offsets):
        o = self.ref.get("log_banner")
        if not o:
            return None
        (dx, dy), score = self.locate_log(frame)
        x, y, w, h = [int(v) for v in o["bbox"]]
        x, y = x + dx, y + dy
        lines = []
        for i in range(2):
            yy = y + 16 + i * 15 - 11                 # строка высотой ~15 px
            t_img = frame[yy:yy + 15, x + 72:x + 138]
            tag_img = frame[yy:yy + 15, x + 138:x + 198]
            msg_img = frame[yy:yy + 15, x + 198:x + w - 6]
            if msg_img.size == 0 or float(msg_img.max()) < 60:
                lines.append(None)
                continue
            t = self.glyphs.read(t_img, allowed=set("0123456789:"))[0].replace(" ", "")
            tag = self.tag_by_template(tag_img) or self.tag(ink(tag_img, 3))
            msg = self.tess(ink(msg_img, 3), "rus+eng", 7)
            lines.append({"time": t, "tag": tag, "msg": msg,
                          "_img": frame[yy:yy + 15, x + 72:x + w - 6]})
        return {"lines": lines, "box": [x, y, w, h], "anchor_score": round(score, 3), "shift": [dx, dy]}

    @staticmethod
    def _tag_ink(crop):
        from glyph_ocr import line_ink
        v, m = line_ink(crop)
        if v is None:
            return None
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            return None
        return v[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def tag_by_template(self, crop, min_score=0.85):
        """Источник — сравнением с эталонами (яркость без цвета: цвет — уровень записи)."""
        from glyph_ocr import dist
        g = self._tag_ink(crop)
        if g is None or not self.tags:
            return None
        best = min(((dist(g, t), tag) for tag, t in self.tags if abs(t.shape[1] - g.shape[1]) <= 3),
                   default=(1.0, None))
        return best[1] if 1 - best[0] >= min_score else None

    def tag(self, img):
        """Источник записи — короткое слово заглавными, латиницей (CV) или
        кириллицей (СИСТ). Смесь моделей путает C/С и V/У, поэтому читаем
        каждой моделью отдельно и берём ту, в которой Tesseract увереннее."""
        best = ("", -1.0)
        UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789-"
        for lang in ("rus+eng",):
            tsv = self.tess(img, lang, 7, UP, tsv=True)
            words = [ln.split("\t") for ln in tsv.splitlines()[1:]]
            words = [(w[11], float(w[10])) for w in words if len(w) >= 12 and w[11].strip() and float(w[10]) >= 0]
            if words:
                conf = sum(c for _, c in words) / len(words)
                if conf > best[1]:
                    best = (" ".join(t for t, _ in words), conf)
        return best[0].upper()

    # ── часы модели ─────────────────────────────────────────────────
    def read_clock(self, frame):
        o = self.ref.get("hdr_clock")
        if not o:
            return None
        x, y, w, h = [int(v) for v in o["bbox"]]
        crop = frame[y:y + h, x:x + w]
        t, info, _, _ = self.glyphs.read(crop)
        m = re.search(r"(\d{2}:\d{2}:\d{2})\s*\(сут\s*(\d+)\)", t)
        return {"raw": t, "text": f"{m.group(1)} (сут {m.group(2)})" if m else None,
                "box": [x, y, w, h], "_crop": crop}

    # ── тренды ──────────────────────────────────────────────────────
    def read_trend(self, frame, t, offsets):
        dx, dy = self._shift(t["id"], offsets)
        L, T, PW, PH = [int(round(v)) for v in t["plot_bbox"]]
        L, T = L + dx, T + dy
        # шкала: подписи у верхней и нижней линии сетки, слева от поля
        top = frame[T - 5:T + 6, L - 40:L - 3]
        bot = frame[T + PH - 5:T + PH + 6, L - 40:L - 3]
        y_max = NUM.search(self.glyphs.read(top, allowed=set("0123456789."))[0].replace(" ", ""))
        y_min = NUM.search(self.glyphs.read(bot, allowed=set("0123456789."))[0].replace(" ", ""))
        y_max = float(y_max.group(0)) if y_max else None
        y_min = float(y_min.group(0)) if y_min else 0.0
        plot = frame[T - 3:T + PH + 4, L:L + PW + 4]
        # Линии тонкие (1,7 px), а MJPEG хранит цвет с половинным разрешением (4:2:0):
        # точный цвет линии в кадре не сохраняется. Поэтому точка линии отбирается
        # по яркости (она в JPEG цела), а серия — по ближайшему оттенку в
        # сглаженном цветовом канале.
        ycc = cv2.cvtColor(plot, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        Y = ycc[:, :, 0]
        chroma = cv2.blur(ycc[:, :, 1:], (3, 3)) - 128.0
        bg = float(np.median(Y))
        line = Y > bg + 28
        cols = self.legend.get(t["id"], [])
        ref = []
        for col in cols:
            c = cv2.cvtColor(np.uint8([[col]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(np.float32)
            ref.append(c[1:] - 128.0)
        ref = np.array(ref) if ref else np.zeros((0, 2), np.float32)
        if len(ref):
            # направление оттенка, а не его сила: смешанные с фоном точки бледнее, но того же тона
            cn = chroma / (np.linalg.norm(chroma, axis=2, keepdims=True) + 1e-3)
            rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-3)
            sim = np.einsum("hwc,kc->hwk", cn, rn)
            who = np.argmax(sim, axis=2)
            best = np.max(sim, axis=2)
            strong = np.linalg.norm(chroma, axis=2) > 6
        series = []
        for si, col in enumerate(cols):
            m = line & strong & (who == si) & (best > 0.8)
            rows = np.where(m, np.arange(m.shape[0], dtype=np.float32)[:, None], np.nan)
            with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                ys = np.nanmedian(rows, axis=0)          # медиана: точки соседней линии и сетки не тянут центр
            # последнее значение — центр точки на правом краю поля
            edge = m[:, PW - 3:PW + 4]
            er = np.nonzero(edge.any(axis=1))[0]
            last_y = float(np.median(er)) if er.size else np.nan
            def to_val(py):
                if np.isnan(py) or y_max is None:
                    return None
                return y_min + (y_max - y_min) * (1 - (py - 3) / PH)
            if y_max is None:
                curve = [None] * (PW + 1)
            else:
                vv = y_min + (y_max - y_min) * (1 - (ys[:PW + 1] - 3) / PH)
                curve = [None if np.isnan(v) else float(v) for v in vv]
            series.append({"color": col, "last": to_val(last_y), "curve": curve,
                           "coverage": float(np.mean([c is not None for c in curve])),
                           "_mask": m})
        # Серия без единой видимой точки, когда её штрих есть в легенде, — не ошибка
        # чтения, а перекрытие: серии рисуются по порядку легенды, и более поздняя
        # с тем же значением ложится сверху (типично — несколько серий на нуле).
        for si, sr in enumerate(series):
            later = [j for j in range(si + 1, len(series)) if series[j]["coverage"] > 0.3]
            if sr["coverage"] < 0.03:
                sr["occluded_by"] = later
            elif sr["last"] is None:
                # Концевая точка ушла под более позднюю серию с близким значением —
                # берём самую правую видимую точку линии у края поля (последние 15 px).
                tail = [(i, v) for i, v in enumerate(sr["curve"]) if v is not None and i >= len(sr["curve"]) - 16]
                if tail:
                    sr["last"] = tail[-1][1]
                    sr["last_from"] = "правая видимая точка"
                elif later:
                    sr["occluded_by"] = later
        return {"id": t["id"], "y_min": y_min, "y_max": y_max, "series": series,
                "plot": [L, T, PW, PH], "names": t.get("series")}

    # ── всё вместе ──────────────────────────────────────────────────
    def read(self, frame, offsets):
        fut_r = [self.pool.submit(self.read_readout, frame, o, offsets) for o in self.readouts]
        fut_t = [self.pool.submit(self.read_trend, frame, t, offsets) for t in self.trends]
        fut_l = self.pool.submit(self.read_log, frame, offsets)
        fut_c = self.pool.submit(self.read_clock, frame)
        fut_b = [self.pool.submit(self.read_bar, frame, o, offsets) for o in self.bars]
        fut_w = self.pool.submit(self.read_well, frame, offsets)
        return {"readouts": {r["id"]: r for r in (f.result() for f in fut_r) if r},
                "bars": {r["id"]: r for r in (f.result() for f in fut_b)},
                "well": fut_w.result(),
                "trends": {t["id"]: t for t in (f.result() for f in fut_t)},
                "log": fut_l.result(), "clock": fut_c.result()}
