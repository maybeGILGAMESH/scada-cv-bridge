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
        self.trends = [o for o in ref["objects"] if o["type"] == "trend"]
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
        return {"id": t["id"], "y_min": y_min, "y_max": y_max, "series": series,
                "plot": [L, T, PW, PH], "names": t.get("series")}

    # ── всё вместе ──────────────────────────────────────────────────
    def read(self, frame, offsets):
        fut_r = [self.pool.submit(self.read_readout, frame, o, offsets) for o in self.readouts]
        fut_t = [self.pool.submit(self.read_trend, frame, t, offsets) for t in self.trends]
        fut_l = self.pool.submit(self.read_log, frame, offsets)
        fut_c = self.pool.submit(self.read_clock, frame)
        return {"readouts": {r["id"]: r for r in (f.result() for f in fut_r) if r},
                "trends": {t["id"]: t for t in (f.result() for f in fut_t)},
                "log": fut_l.result(), "clock": fut_c.result()}
