# -*- coding: utf-8 -*-
"""Распознавание незнакомого щита SCADA без профиля — «с нуля», по пикселям.

Ни разметки, ни эталонов, ни DevTools: только кадр. Знание — общее, как выглядит
SCADA вообще:

  * строки текста — морфологический градиент → строки → Tesseract по каждой
    строке (полярность у каждой своя: светлое на тёмном и наоборот); распознанное
    кэшируется по двоичной картинке строки, на следующем кадре Tesseract
    вызывается только для изменившихся строк (живые значения);
  * стрелочный прибор — круг (HoughCircles), на котором есть числа шкалы:
    стрелка — луч от центра, сильнее всего отличающийся от циферблата; шкала —
    угол каждого числа → линейная зависимость «угол → значение»;
  * лампа — маленький круг; горит — если цвет насыщенный и яркий;
  * кнопка — прямоугольник с одной короткой надписью; нажата — заливка
    насыщенная и яркая;
  * полоса и резервуар — вытянутый прямоугольник с заливкой: доля заливки;
  * числовое табло — строка с числом: единица — после числа, подпись — строка
    над числом или слева от него.

Проверка (только для стенда): истина генератора случайных щитов
(random-scada.html, window.__truth) — что где нарисовано. Распознавание её не видит.

    python zero_vision.py shots/*.jpg       # оценка по снимкам с *.json рядом
"""

import hashlib
import json
import math
import re
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NUM = re.compile(r"(?<![A-Za-zА-Яа-яЁё\d])[-−]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?(?![\d])|(?<![A-Za-zА-Яа-яЁё\d])[-−]?\d+(?:[.,]\d+)?")
UNITS = ["%НКПР", "м³/ч", "м3/ч", "т/сут", "об/мин", "мм/с", "кПа", "МПа", "бар", "°C", "°С", "кВт", "Гц", "м³", "%", "А", "В", "м", "pH"]
HOMO = str.maketrans("ABEKMHOPCTXaeopcyx", "АВЕКМНОРСТХаеорсух")


def norm(t):
    return re.sub(r"[\s·•.,:;\-–—()]+", "", (t or "").lower().translate(str.maketrans("abekmhopctxyз", "авекмнорстхуз")))


def similar(a, b):
    import difflib
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


# ── строки текста ───────────────────────────────────────────────────
def text_lines(img):
    """Кандидаты строк: градиент яркости → порог → склейка по горизонтали."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 8 or h > 70 or w < 6 or w > 900:
            continue
        fill = a / float(w * h)
        if fill < 0.18 or fill > 0.97:
            continue
        if w < h * 0.5:
            continue
        out.append([int(x), int(y), int(w), int(h)])
    # куски одной строки: одна высота, общая строка, разрыв меньше половины высоты
    out.sort(key=lambda b: (b[1], b[0]))
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(len(out)):
                if i == j or out[i] is None or out[j] is None:
                    continue
                a, b = out[i], out[j]
                if b[0] < a[0]:
                    continue
                ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                hr = max(a[3], b[3]) / max(1, min(a[3], b[3]))
                gap = b[0] - (a[0] + a[2])
                if ov >= 0.6 * min(a[3], b[3]) and hr <= 1.35 and -2 <= gap <= 0.45 * max(a[3], b[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    out[i] = [x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0]
                    out[j] = None
                    merged = True
        out = [b for b in out if b]
    return out


def normalize_crop(c, target_h=44):
    """Строка → чёрный текст на белом, высота ~44 px: полярность по меньшинству."""
    g = c.max(axis=2) if c.ndim == 3 else c
    # цветной текст на цветном фоне: берём канал с наибольшим контрастом
    if c.ndim == 3:
        best = max([c[:, :, k] for k in range(3)] + [g], key=lambda ch: float(ch.std()))
        g = best
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (b > 0).mean() > 0.5:                  # белого больше — текст тёмный: оставить
        txt_dark = True
    else:
        txt_dark = False
    gg = g if txt_dark else 255 - g
    s = max(1.0, target_h / max(1, c.shape[0]))
    gg = cv2.resize(gg, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    gg = cv2.normalize(gg, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.copyMakeBorder(gg, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=int(np.percentile(gg, 90)))


class OCR:
    """Tesseract по строкам с кэшем по двоичной картинке строки."""

    def __init__(self, tess, workers=4):
        self.tess = tess
        self.memo = {}
        self.pool = ThreadPoolExecutor(workers)
        self.calls = 0

    def key(self, c):
        """Ключ — двоичная картинка строки в полном разрешении (Otsu): у разных надписей
        одного размера грубый ключ совпадал и подпись бралась из кэша чужая."""
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return hashlib.md5(np.packbits(b).tobytes() + bytes(str(b.shape), "ascii")).hexdigest()

    def read(self, img, boxes):
        jobs, res = {}, [None] * len(boxes)
        for i, (x, y, w, h) in enumerate(boxes):
            c = img[max(0, y - 2):y + h + 2, max(0, x - 3):x + w + 3]
            k = self.key(c)
            if k in self.memo:
                res[i] = self.memo[k]
            else:
                jobs[i] = (k, self.pool.submit(self._one, c))
        for i, (k, f) in jobs.items():
            t = f.result()
            self.memo[k] = res[i] = t
        if len(self.memo) > 20000:
            self.memo.clear()
        return res

    def _one(self, c):
        self.calls += 1
        txt = self.tess(normalize_crop(c), "rus+eng", 7).strip()
        return re.sub(r"\s+", " ", txt)


# ── формы ───────────────────────────────────────────────────────────
def hsv_of(img, mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mask is not None:
        px = hsv[mask > 0]
    else:
        px = hsv.reshape(-1, 3)
    if len(px) == 0:
        return 0, 0, 0
    return [float(np.median(px[:, k])) for k in range(3)]


def find_circles(img):
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    big = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120, param1=110, param2=55, minRadius=55, maxRadius=170)
    small = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=18, param1=100, param2=14, minRadius=6, maxRadius=16)
    big = [] if big is None else [tuple(map(float, c)) for c in big[0]]
    small = [] if small is None else [tuple(map(float, c)) for c in small[0]]
    return big, small


def find_rects(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 25, 70)
    e = cv2.dilate(e, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 12 or h < 10 or w > 1500 or h > 900:
            continue
        if cv2.contourArea(c) / float(w * h) < 0.82:
            continue
        if any(abs(x - r[0]) <= 4 and abs(y - r[1]) <= 4 and abs(w - r[2]) <= 8 and abs(h - r[3]) <= 8 for r in out):
            continue
        out.append([int(x), int(y), int(w), int(h)])
    return out


def inside(p, r, pad=0):
    return r[0] - pad <= p[0] <= r[0] + r[2] + pad and r[1] - pad <= p[1] <= r[1] + r[3] + pad


def center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def has_letters(t):
    return re.search(r"[A-Za-zА-Яа-яЁё]{2,}", t or "") is not None


def is_numeric(t):
    """Строка — это число (с единицей): «52 %», «14.49 бар», «1739 м³/ч», а не «Насос 1»."""
    s = (t or "").strip()
    m = NUM.search(s.replace(",", "."))
    if not m:
        return False
    rest = (s[:m.start()] + s[m.end():]).strip()
    return not rest or is_unit(rest)


def is_unit(t):
    s = (t or "").strip().lower().replace(" ", "").replace("3", "³")
    return bool(s) and any(s == u.lower().replace(" ", "") or s.startswith(u.lower()) and len(s) <= len(u) + 2 for u in UNITS)


def parse_num(t):
    m = NUM.search((t or "").replace("−", "-").replace(",", "."))
    return float(m.group(0).replace(" ", "")) if m else None


def split_value(t):
    """«Расход воды 298.5 м³/ч» → (подпись, число-текст, единица).
    «Положение задвижки 4», «Насос 1» — номер в подписи, а не значение."""
    t = (t or "").strip()
    m = list(NUM.finditer(t))
    if not m:
        return t, None, None
    last = m[-1]
    if has_letters(t[:last.start()]) and not t[last.end():].strip() and re.fullmatch(r"\d{1,2}", last.group(0)):
        return t, None, None
    k = m[0]
    # число — то, после которого идёт единица или конец строки
    for mm in m:
        rest = t[mm.end():].strip()
        if not rest or any(rest.lower().startswith(u.lower()) for u in UNITS):
            k = mm
    label, rest = t[:k.start()].strip(" :"), t[k.end():].strip()
    unit = next((u for u in UNITS if rest.lower().startswith(u.lower())), rest[:8] or None)
    return label, k.group(0).replace(",", ".").replace(" ", ""), unit


# ── прибор ──────────────────────────────────────────────────────────
def scale_tokens(img, cx, cy, r, ocr):
    """Числа шкалы циферблата: строки текста здесь склеиваются с рисками и стрелкой,
    поэтому — свой разбор: «чернила» циферблата (контраст с его фоном, без
    насыщенного цвета стрелки), риски — тонкие длинные штрихи — отбрасываются,
    цифры собираются в числа и читаются Tesseract только цифрами."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(W, cx + r)), int(min(H, cy + r))
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    L = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.int16)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    ring = (d > 0.38 * r) & (d < 0.52 * r)
    Ld = np.median(L[ring]) if ring.any() else np.median(L)
    needle = (S >= 110) & (V >= 120)                          # стрелка — яркая и насыщенная
    ink = ((np.abs(L - Ld) > 45) & ~needle & (d > 0.25 * r) & (d < 0.9 * r)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 5 or h < 6 or h > 26 or w > 26:
            continue
        pts = np.column_stack(np.nonzero(lab == i)).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        if min(rw, rh) <= 3.2 and max(rw, rh) >= 6:         # риска шкалы
            continue
        comps.append([x, y, w, h])
    comps.sort(key=lambda b: (b[1] // 8, b[0]))
    groups = []
    for b in sorted(comps, key=lambda b: b[0]):
        for g in groups:
            gx, gy, gw, gh = g
            if b[0] - (gx + gw) <= 4 and b[0] >= gx - 2 and min(b[1] + b[3], gy + gh) - max(b[1], gy) >= 0.5 * min(b[3], gh):
                nx, ny = min(gx, b[0]), min(gy, b[1])
                g[:] = [nx, ny, max(gx + gw, b[0] + b[2]) - nx, max(gy + gh, b[1] + b[3]) - ny]
                break
        else:
            groups.append(list(b))
    out = []
    for gx, gy, gw, gh in groups:
        sl = (slice(max(0, gy - 2), gy + gh + 2), slice(max(0, gx - 2), gx + gw + 2))
        m = ink[sl]
        g = L[sl].astype(np.float32)
        g[needle[sl]] = Ld                                    # стрелку закрасить фоном
        if g.mean() < Ld - 1 or np.median(L[ring]) < 128:     # полярность: цифры тёмные на светлом
            pass
        crop = np.clip(255 - np.abs(g - Ld) * 255.0 / max(1.0, np.abs(g - Ld).max()), 0, 255).astype(np.uint8)
        k = "sc" + hashlib.md5(np.packbits(m).tobytes() + bytes(str(m.shape), "ascii")).hexdigest()
        txt = ocr.memo.get(k)
        if txt is None:
            s = 40.0 / max(1, crop.shape[0])
            big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            big = cv2.copyMakeBorder(big, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=255)
            txt = ocr.tess(big, "eng", 7, whitelist="0123456789.-").strip()
            ocr.calls += 1
            ocr.memo[k] = txt
        if re.fullmatch(r"-?\d+(\.\d+)?", txt or ""):
            out.append({"box": [x0 + gx, y0 + gy, gw, gh], "text": txt})
    return out


def read_gauge(img, cx, cy, r, toks):
    """Стрелка — луч от центра, самый непохожий на циферблат; шкала — числа на нём."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = img.shape[:2]
    # цвет циферблата — медиана кольца 0.35..0.55 r
    ring = []
    for a in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        for k in (0.4, 0.5):
            x, y = int(cx + math.cos(a) * r * k), int(cy + math.sin(a) * r * k)
            if 0 <= x < W and 0 <= y < H:
                ring.append(lab[y, x])
    dial = np.median(np.array(ring), axis=0)
    best, best_a = -1, None
    prof_a = np.zeros(360)
    for deg in range(0, 360):
        a = math.radians(deg)
        s = 0.0
        for k in np.linspace(0.18, 0.62, 12):
            x, y = int(round(cx + math.cos(a) * r * k)), int(round(cy + math.sin(a) * r * k))
            if 0 <= x < W and 0 <= y < H:
                d = np.linalg.norm(lab[y, x] - dial)
                s += min(d, 80)
        prof_a[deg] = s
        if s > best:
            best, best_a = s, deg
    if best_a is not None:                      # уточнение: центр масс пика ±4°
        ds = np.arange(-4, 5)
        wts = np.array([prof_a[(best_a + d) % 360] for d in ds])
        wts = np.clip(wts - wts.min(), 0, None)
        if wts.sum() > 0:
            best_a = (best_a + float((ds * wts).sum() / wts.sum())) % 360
    # числа шкалы: внутри круга, на 0.45..0.95 r
    pts = []
    for t in toks:
        v = parse_num(t["text"])
        if v is None or not re.fullmatch(r"[-−]?\d+(?:[.,]\d+)?", t["text"].strip()):
            continue
        px, py = center(t["box"])
        d = math.hypot(px - cx, py - cy)
        if 0.52 * r <= d <= 0.95 * r:            # кольцо шкалы; цифровое табло прибора — ближе к центру
            pts.append((math.degrees(math.atan2(py - cy, px - cx)) % 360, v))
    if len(pts) < 3 or best_a is None:
        return None, len(pts)
    # «развернуть» углы: начало шкалы неизвестно — перебираем каждое число шкалы как начало
    # (по самому большому разрыву не выходит: при 4 делениях на 300° разрыв между
    # делениями больше пустой части циферблата); затем RANSAC по парам: неверно
    # прочитанное число шкалы и цифровое табло прибора — выбросы
    span_v = (max(v for _, v in pts) - min(v for _, v in pts)) or 1
    best_fit = None
    for base in sorted({a for a, _ in pts}):
        un = sorted(((a - base) % 360, v) for a, v in pts)
        for i in range(len(un)):
            for j in range(i + 1, len(un)):
                (a1, v1), (a2, v2) = un[i], un[j]
                if abs(a2 - a1) < 5 or v1 == v2:
                    continue
                kk = (v2 - v1) / (a2 - a1)
                if kk <= 0:
                    continue
                bb = v1 - kk * a1
                inl = [q for q in range(len(un)) if abs(kk * un[q][0] + bb - un[q][1]) <= 0.025 * span_v]
                score = (len({un[q][1] for q in inl}), -sum(abs(kk * un[q][0] + bb - un[q][1]) for q in inl))
                if best_fit is None or score > best_fit[0]:
                    best_fit = (score, inl, base, un)
    if best_fit is None or best_fit[0][0] < 3:
        return None, len(pts)
    _, keep, base, un = best_fit
    A = np.array([[un[q][0], 1.0] for q in keep])
    yv = np.array([un[q][1] for q in keep])
    k, b = np.linalg.lstsq(A, yv, rcond=None)[0]
    resid = float(np.abs(A @ np.array([k, b]) - yv).max())
    pts = [(un[i][0] + base, un[i][1]) for i in keep]
    pts.sort(key=lambda p: p[1])
    un = sorted(((a - base) % 360, v) for a, v in pts)
    na = (best_a - base) % 360
    gap_mid = un[-1][0] + (360 - un[-1][0] + un[0][0]) / 2
    if na > gap_mid:                            # стрелка «до нуля» — по другую сторону разрыва шкалы
        na -= 360
    val = k * na + b
    return {"value": float(val), "angle": best_a, "scale": [p[1] for p in pts], "fit_err": float(resid),
            "min": pts[0][1], "max": pts[-1][1]}, len(pts)


def refine_bar(img, x, y, w, h):
    """Горизонтальная шкала: рамка дорожки в тон фону и в Canny почти не видна,
    находится только залитая или только пустая часть. От найденного куска — влево и
    вправо по той же строке, пока цвет столбцов отличается от фона панели; заливка —
    доля столбцов более насыщенного из двух цветов дорожки."""
    H, W = img.shape[:2]
    y0, y1 = y + max(2, h // 4), y + h - max(2, h // 4)
    band = img[y0:y1].astype(np.float32).mean(axis=0)            # цвет каждого столбца
    above = img[max(0, y - 4):max(1, y - 2), max(0, x - 10):min(W, x + w + 10)].reshape(-1, 3).astype(np.float32)
    bg = np.median(above, axis=0) if len(above) else band[max(0, x - 6)]
    dist = np.linalg.norm(band - bg, axis=1)
    a, b = x + 2, x + w - 3
    while a > 0 and dist[a - 1] > 14:
        a -= 1
    while b < W - 1 and dist[b + 1] > 14:
        b += 1
    if b - a < 40:
        return x, w, None
    cols = band[a + 3:b - 2]
    hsv = cv2.cvtColor(cols.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[0].astype(np.float32)
    sv = hsv[:, 1] * hsv[:, 2] / 255.0
    lab = cv2.cvtColor(cols.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    # два цвета дорожки: левый край (если заливка есть, она слева) и правый
    left, right = lab[:4].mean(axis=0), lab[-4:].mean(axis=0)
    if np.linalg.norm(left - right) < 12:                     # одного цвета: пусто или полностью залито
        frac = 1.0 if sv[:4].mean() > 60 else 0.0
    else:
        frac = float((np.linalg.norm(lab - left, axis=1) < np.linalg.norm(lab - right, axis=1)).mean())
    return a, b - a + 1, frac


# ── щит целиком ─────────────────────────────────────────────────────
def analyze(img, ocr):
    t0 = time.perf_counter()
    lines = text_lines(img)
    texts = ocr.read(img, lines)
    toks = [{"box": b, "text": t} for b, t in zip(lines, texts) if t and len(t.strip()) >= 1]
    t_ocr = time.perf_counter() - t0
    big, small = find_circles(img)
    rects = find_rects(img)
    used, items = set(), []

    def take(tok):
        used.add(id(tok))

    # стрелочные приборы
    gauges = []
    for cx, cy, r in big:
        inner = [t for t in toks if math.hypot(*(np.subtract(center(t["box"]), (cx, cy)))) < r * 1.02]
        g, nscale = read_gauge(img, cx, cy, r, scale_tokens(img, cx, cy, r, ocr))
        if g is None:
            continue
        for t in inner:
            take(t)
        unit = next((t["text"] for t in inner if parse_num(t["text"]) is None and len(t["text"]) <= 8), None)
        digital = next((t for t in inner if re.fullmatch(r"[-−]?\d+[.,]\d+", t["text"].strip())
                        and math.hypot(*(np.subtract(center(t["box"]), (cx, cy)))) < 0.6 * r and center(t["box"])[1] > cy), None)
        below = [t for t in toks if id(t) not in used and abs(center(t["box"])[0] - cx) < r and 0 < t["box"][1] - (cy + r) < 50
                 and parse_num(t["text"]) is None]
        label = min(below, key=lambda t: t["box"][1])["text"] if below else None
        for t in below[:1]:
            take(t)
        box = [int(cx - r), int(cy - r), int(2 * r), int(2 * r)]
        val = g["value"]
        if digital is not None:
            dv = parse_num(digital["text"])
            if dv is not None and abs(dv - val) < 0.08 * (g["max"] - g["min"] or 1):
                val = dv                                          # цифровое табло на приборе — точнее стрелки
        items.append({"type": "gauge", "box": box, "label": label, "value": round(val, 2), "unit": unit,
                      "detail": {"angle": g["angle"], "scale": g["scale"], "fit_err": round(g["fit_err"], 2)}})
        gauges.append((cx, cy, r))
    in_gauge = lambda p: any(math.hypot(p[0] - cx, p[1] - cy) < r * 1.05 for cx, cy, r in gauges)

    # лампы — маленькие круги вне приборов
    for cx, cy, r in small:
        if in_gauge((cx, cy)):
            continue
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(m, (int(cx), int(cy)), max(2, int(r * 0.6)), 255, -1)
        h, s, v = hsv_of(img, m)
        right = [t for t in toks if id(t) not in used and 0 <= t["box"][0] - (cx + r) < 40 and abs(center(t["box"])[1] - cy) < 12]
        if not right:
            continue
        lab_t = min(right, key=lambda t: t["box"][0])
        take(lab_t)
        on = s > 110 and v > 150
        items.append({"type": "lamp", "box": [int(cx - r), int(cy - r), int(2 * r), int(2 * r)], "label": lab_t["text"],
                      "state": "on" if on else "off", "detail": {"h": h, "s": s, "v": v}})

    # прямоугольники: кнопки, полосы, резервуары. Крупные — первыми: заливка и пустая
    # часть шкалы — тоже прямоугольники, но внутри уже взятой шкалы
    taken = []
    contained = lambda r: any(r[0] >= q[0] - 3 and r[1] >= q[1] - 3 and r[0] + r[2] <= q[0] + q[2] + 3
                              and r[1] + r[3] <= q[1] + q[3] + 3 for q in taken)
    for x, y, w, h in sorted(rects, key=lambda r: -r[2] * r[3]):
        if in_gauge((x + w / 2, y + h / 2)) or contained([x, y, w, h]):
            continue
        inner = [t for t in toks if inside(center(t["box"]), [x, y, w, h]) and t["box"][2] < w]
        roi = img[y + 4:y + h - 4, x + 4:x + w - 4]
        if roi.size == 0:
            continue
        # кнопка: одна строка-надпись без числа, рамка ~ 28..70 px высотой
        if 26 <= h <= 72 and 80 <= w <= 330 and len(inner) == 1 and has_letters(inner[0]["text"]) \
                and not is_numeric(inner[0]["text"]) and id(inner[0]) not in used:
            t = inner[0]
            mask = np.full(roi.shape[:2], 255, np.uint8)
            hh, s, v = hsv_of(roi, mask)
            take(t)
            taken.append([x, y, w, h])
            items.append({"type": "button", "box": [x, y, w, h], "label": t["text"],
                          "state": "on" if (s > 115 and v > 115) else "off", "detail": {"s": s, "v": v}})
            continue
        # полоса: вытянута по горизонтали, без текста внутри
        horiz = w >= 5 * h and 10 <= h <= 34 and w >= 120 and not inner
        vert = h >= 1.6 * w and 30 <= w <= 130 and h >= 90 and not inner
        if horiz:
            x, w, frac_b = refine_bar(img, x, y, w, h)
            roi = img[y + 4:y + h - 4, x + 4:x + w - 4]
            if contained([x, y, w, h]):
                continue
        if horiz or vert:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            prof = (hsv[:, :, 1].astype(np.float32) * hsv[:, :, 2] / 255.0)
            prof = prof.mean(axis=0) if horiz else prof.mean(axis=1)[::-1]
            thr = (prof.max() + prof.min()) / 2
            if prof.max() - prof.min() < 25:
                frac = 1.0 if prof.mean() > 80 else 0.0
            else:
                frac = float((prof > thr).mean())   # заливка сплошная — доля заполненного
            if horiz and frac_b is not None:
                frac = frac_b
            kind = "bar" if horiz else "tank"
            taken.append([x, y, w, h])
            # подпись: над полосой (полоса) / под резервуаром; число «97 %» — рядом
            if horiz:
                cand = [t for t in toks if id(t) not in used and 0 <= y - (t["box"][1] + t["box"][3]) < 22 and abs(t["box"][0] - x) < 30 and not is_numeric(t["text"])]
                numc = [t for t in toks if id(t) not in used and 0 <= t["box"][1] - (y + h) < 22 and abs(t["box"][0] - x) < 30 and is_numeric(t["text"])]
            else:
                below = sorted([t for t in toks if id(t) not in used and 0 <= t["box"][1] - (y + h) < 70
                                and abs(center(t["box"])[0] - (x + w / 2)) < 70], key=lambda t: t["box"][1])
                cand = [t for t in below if not is_numeric(t["text"])][:1]
                numc = [t for t in below if is_numeric(t["text"])][:1]
            label = cand[0]["text"] if cand else None
            for t in cand[:1] + numc[:1]:
                take(t)
            val = round(frac * 100, 1)
            nv = parse_num(numc[0]["text"]) if numc else None
            if nv is not None and abs(nv - val) <= 6:
                val = nv
            items.append({"type": kind, "box": [x, y, w, h], "label": label, "value": val, "unit": "%",
                          "detail": {"fill": round(frac, 3), "text": numc[0]["text"] if numc else None}})

    # числовые табло: строки с числом, ещё не взятые
    rest = [t for t in toks if id(t) not in used]
    for t in rest:
        lab, num, unit = split_value(t["text"])
        if num is None or id(t) in used:
            continue
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", t["text"].strip()):
            continue                                              # часы и время журнала
        x, y, w, h = t["box"]
        # рамка табло — самый маленький прямоугольник вокруг числа: подпись ищем в ней
        frame = min((r for r in rects if inside((x + 2, y + 2), r) and inside((x + w - 2, y + h - 2), r)
                     and r[2] * r[3] < 900 * 300), key=lambda r: r[2] * r[3], default=None)
        same = (lambda o: inside(center(o["box"]), frame)) if frame else (lambda o: True)
        if not lab or len(norm(lab)) < 3:
            above = [o for o in rest if id(o) not in used and o is not t and 0 <= y - (o["box"][1] + o["box"][3]) < 1.6 * h
                     and o["box"][0] < x + w and o["box"][0] + o["box"][2] > x - 20 and has_letters(o["text"]) and not is_numeric(o["text"])
                     and not is_unit(o["text"]) and same(o)]
            left = [o for o in rest if id(o) not in used and o is not t and 0 <= x - (o["box"][0] + o["box"][2]) < 60
                    and abs(center(o["box"])[1] - center(t["box"])[1]) < h * 0.8 and has_letters(o["text"]) and not is_numeric(o["text"])
                    and not is_unit(o["text"]) and same(o)]
            src = max(left, key=lambda o: o["box"][0]) if left else (min(above, key=lambda o: y - o["box"][1]) if above else None)
            if src is None:
                continue
            lab = src["text"]
            take(src)
            if unit is None or not unit:
                unit = None
        take(t)
        bb = t["box"]
        items.append({"type": "value", "box": bb, "label": lab, "value": float(num), "text": num, "unit": unit})
    return items, {"lines": len(lines), "ocr_s": round(t_ocr, 2), "total_s": round(time.perf_counter() - t0, 2),
                   "tess_calls": ocr.calls, "circles": len(big), "lamps": len(small), "rects": len(rects)}


# ── сверка с истиной генератора (только стенд) ───────────────────────
TOL = {"gauge": 0.04, "bar": 0.04, "tank": 0.04, "trend": 0.06}


def evaluate(items, truth):
    res = defaultdict(lambda: {"n": 0, "found": 0, "label": 0, "value": 0})
    matched = []
    for w in truth["widgets"]:
        typ = w["type"]
        if typ in ("log", "trend"):
            continue
        r = res[typ]
        r["n"] += 1
        bb = w["bbox"]
        cand = [it for it in items if it["type"] == typ and inside(center(it["box"]), bb, 12)]
        if not cand:
            matched.append((w, None))
            continue
        it = min(cand, key=lambda it: math.hypot(*(np.subtract(center(it["box"]), center(bb)))))
        r["found"] += 1
        r["label"] += similar(it.get("label"), w.get("label"))
        if typ in ("lamp", "button"):
            r["value"] += it.get("state") == w.get("state")
        elif typ == "value":
            r["value"] += it.get("text") == w.get("text") or (it.get("value") is not None and abs(it["value"] - w["value"]) < 1e-6)
        else:
            span = (w.get("max", 100) - w.get("min", 0)) or 1
            r["value"] += it.get("value") is not None and abs(it["value"] - w["value"]) <= TOL[typ] * span
        matched.append((w, it))
    return dict(res), matched


TITLES = {"gauge": "Стрелочные приборы", "value": "Числовые табло", "bar": "Шкалы-полосы", "tank": "Резервуары",
          "lamp": "Лампы", "button": "Кнопки"}
STATE_RU = {"on": "вкл", "off": "выкл"}


def png64(img, scale=1.0):
    import base64
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ok, b = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(b.tobytes()).decode()


class ZeroVision(threading.Thread):
    """Рабочий дашборда для щита без профиля: тот же контракт, что у профильных
    (snapshot / frame / press / exact / detail), только знание — общее."""

    def __init__(self, system, src, cdp, ctl, tess):
        super().__init__(daemon=True)
        self.system, self.src, self.cdp, self.ctl = system, src, cdp, ctl
        self.ocr = OCR(tess)
        self.lock = threading.Lock()
        self.state = {"ready": False, "kind": "vision", "reader": "zero"}
        self.frames, self.items = {}, []
        self.hist = deque(maxlen=600)
        self.roll = deque(maxlen=20)
        self.persist = defaultdict(lambda: deque(maxlen=12))
        self.errors, self.events = deque(maxlen=40), deque(maxlen=40)
        self.seq, self.t_fps = 0, deque(maxlen=12)
        self.active, self.t_last = False, 0.0
        self.truth_down = False
        self.paused_vision = False
        self.force_reg = False
        self.cache_off = set()
        self.inventory = set()
        self.seed = None

    def run(self):
        while True:
            try:
                self.step()
            except Exception as e:
                with self.lock:
                    self.state["error"] = f"{type(e).__name__}: {e}"
                time.sleep(1)

    def _truth(self):
        try:
            tr = self.cdp.evaluate("() => (window.__truth ? window.__truth() : null)")
            self.truth_down = tr is None
            return tr
        except Exception:
            self.truth_down = True
            return None

    def step(self):
        if self.paused_vision:
            time.sleep(0.2)
            return
        if not self.active and time.time() > getattr(self, "kvm_until", 0):
            w = self.t_last + 3.0 - time.time()                  # фоновая система — раз в ~3 с (OCR тяжёлый)
            if w > 0:
                time.sleep(w)
        self.t_last = time.time()
        clock = time.perf_counter
        truth_a = self._truth()                                  # истина до кадра…
        img, t_enc, t_recv = self.src.next_after(clock() - 0.005)
        truth_b = self._truth()                                  # …и сразу после: значения живут раз в секунду
        age = clock() - t_enc if img is not None else None
        if img is None or age > 5:
            with self.lock:
                self.state["health"] = {"stream_dead": True, "stream_age_s": None if age is None else round(age)}
            time.sleep(0.5)
            return
        if self.force_reg:                                   # «сбросить кэш»: распознать заново
            self.ocr.memo.clear()
            self.force_reg = False
        t0 = clock()
        items, info = analyze(img, self.ocr)
        t_read = clock() - t0
        truth = truth_b or truth_a
        alt = {w["i"]: w for w in (truth_a or {}).get("widgets", [])} if truth_a and truth_b and truth_a.get("seed") == truth_b.get("seed") else {}
        if truth and truth.get("seed") != self.seed:
            if self.seed is not None:
                self.events.appendleft({"t": time.strftime("%H:%M:%S"), "text": f"новая станция (зерно {truth.get('seed')})"})
                self.persist.clear()
                self.roll.clear()
            self.seed = truth.get("seed")
        objs, stat = [], defaultdict(lambda: [0, 0])
        found = set()
        for it in items:
            cx, cy = center(it["box"])
            oid = f"{it['type']}@{int(cx) // 8 * 8},{int(cy) // 8 * 8}"
            it["id"] = oid
            read = it.get("state") if it["type"] in ("lamp", "button") else it.get("value")
            disp = STATE_RU.get(read, read) if it["type"] in ("lamp", "button") else (
                None if read is None else f"{read:g}" + (f" {it['unit']}" if it.get("unit") else ""))
            tw, ok, truth_disp, label_ok = None, None, None, None
            if truth:
                cand = [w for w in truth["widgets"] if w["type"] == it["type"] and inside((cx, cy), w["bbox"], 12)]
                if cand:
                    tw = min(cand, key=lambda w: math.hypot(cx - center(w["bbox"])[0], cy - center(w["bbox"])[1]))
                    found.add(tw["i"])
                    label_ok = similar(it.get("label"), tw.get("label"))
                    def vok(w):
                        if it["type"] in ("lamp", "button"):
                            return it.get("state") == w.get("state")
                        if it["type"] == "value":
                            return it.get("text") == w.get("text") or (read is not None and abs(read - w["value"]) < 1e-6)
                        span = (w.get("max", 100) - w.get("min", 0)) or 1
                        return read is not None and abs(read - w["value"]) <= TOL.get(it["type"], 0.04) * span
                    v_ok = vok(tw) or (tw["i"] in alt and vok(alt[tw["i"]]))   # кадр — между двумя снимками истины
                    truth_disp = STATE_RU.get(tw.get("state")) if it["type"] in ("lamp", "button") else (
                        tw.get("text") if it["type"] == "value" else f"{tw['value']:g}")
                    ok = bool(v_ok and label_ok)
                    stat[it["type"]][1] += 1
                    stat[it["type"]][0] += int(ok)
            objs.append({"id": oid, "cls": it["type"], "label": it.get("label") or "—", "at": it["box"][:2], "size": it["box"][2:],
                         "read": disp, "truth": (f"{tw.get('label')}: {truth_disp}" if tw else None), "ok": ok,
                         "label_ok": label_ok, "score": None, "method": "без профиля", "cached": False, "ms": 0,
                         "block": None, "hidden": False, "reliable": None, "value_box": it["box"] if it["type"] == "value" else None})
        missed = []
        if truth:
            for w in truth["widgets"]:
                if w["type"] in ("log", "trend") or w["i"] in found:
                    continue
                missed.append(w)
                stat[w["type"]][1] += 1                               # не найден — тоже ошибка
                objs.append({"id": f"missed@{w['i']}", "cls": w["type"], "label": w.get("label") or "—", "at": w["bbox"][:2],
                             "size": w["bbox"][2:], "read": None, "truth": f"{w.get('label')}: {w.get('value', w.get('state'))}",
                             "ok": False, "missed": True, "score": None, "method": "не найден", "cached": False, "ms": 0,
                             "block": None, "hidden": False, "reliable": None})
        persistent = []
        for ob in objs:
            if ob["ok"] is not None:
                self.persist[ob["id"]].append(ob["ok"])
            pq = self.persist.get(ob["id"])
            if pq and len(pq) >= 6 and pq.count(False) >= 0.7 * len(pq) and ob["ok"] is False:
                persistent.append({"id": ob["id"], "cls": ob["cls"], "label": ob["label"], "read": ob["read"], "truth": ob["truth"]})
            if ob["ok"] is False and not any(e["id"] == ob["id"] for e in list(self.errors)[:10]):
                self.errors.appendleft({"t": time.strftime("%H:%M:%S"), "id": ob["id"], "cls": ob["cls"], "label": ob["label"],
                                        "read": ob["read"], "truth": ob["truth"]})
        self.roll.append(dict(stat))
        roll = defaultdict(lambda: [0, 0])
        for fr in self.roll:
            for c, (a, b) in fr.items():
                roll[c][0] += a
                roll[c][1] += b
        ok_n, tot_n = sum(v[0] for v in stat.values()), sum(v[1] for v in stat.values())
        t_done = clock()
        self.t_fps.append(t_done)
        fps = (len(self.t_fps) - 1) / (self.t_fps[-1] - self.t_fps[0]) if len(self.t_fps) > 2 else 0.0
        ok_jpg, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.seq += 1
        timing = {"read_ms": round(t_read * 1000, 1), "registration_ms": 0, "extra_ms": 0, "truth_ms": 0,
                  "age_ms": round((t_done - t_enc) * 1000, 1), "transport_ms": round((t_recv - t_enc) * 1000, 1),
                  "per_class_ms": {"OCR строк": round(info["ocr_s"] * 1000, 1), "формы и смысл": round((info["total_s"] - info["ocr_s"]) * 1000, 1)}}
        health = {"scene": "ok", "lost_blocks": [], "quality": 1.0, "persistent": persistent, "occluded": [],
                  "truth_down": self.truth_down, "model_paused": False, "fps": round(fps, 2), "active": self.active,
                  "stream_dead": False, "stream_age_s": round(age, 1)}
        with self.lock:
            self.items = items
            self.inventory = {it["id"] for it in items}
            self.frames[self.seq] = (jpg.tobytes(), img, {}, None)
            for k in [k for k in self.frames if k < self.seq - 3]:
                del self.frames[k]
            self.hist.append({"t": time.time(), "acc": round(100 * ok_n / max(1, tot_n), 2), "read_ms": timing["read_ms"],
                              "fps": round(fps, 2), "quality": 1.0, "age_ms": timing["age_ms"], "reread": info["tess_calls"]})
            self.state = {
                "ready": True, "kind": "vision", "reader": "zero", "seq": self.seq, "t": time.time(),
                "frame_size": [img.shape[1], img.shape[0]], "objects": objs, "blocks": [], "class_titles": TITLES,
                "trends": {}, "log_lines": [], "log_truth": None, "registered": False, "quality": 1.0,
                "read_objects": len(items), "cached_objects": 0, "fps": round(fps, 2), "timing": timing,
                "frame": {"ok": ok_n, "total": tot_n},
                "rolling": {"ok": sum(v[0] for v in roll.values()), "total": sum(v[1] for v in roll.values()), "frames": len(self.roll),
                            "by_class": {c: {"ok": a, "total": b, "title": TITLES.get(c, c)} for c, (a, b) in roll.items()}},
                "gauge_err": {"mean": None, "p95": None, "hist": []},
                "model": {"layout": None, "layout_name": None, "clock": None, "plant": None, "state": None,
                          "seed": self.seed},
                "zero": {"found": len(items), "missed": len(missed), "lines": info["lines"], "tess_calls": info["tess_calls"],
                         "seed": self.seed, "truth_widgets": len(truth["widgets"]) if truth else None},
                "errors": list(self.errors), "events": list(self.events), "health": health,
                "source_error": self.src.error, "error": None, "system": self.system["id"], "cache_off": [],
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
            return self.frames.get(seq) or self.frames[max(self.frames)]

    def item(self, oid):
        with self.lock:
            return next((dict(it) for it in self.items if it.get("id") == oid), None)

    def press(self, obj, times=1):
        it = self.item(obj)
        if it is None:
            raise RuntimeError(f"«{obj}» на кадре сейчас не найден")
        x, y, w, h = it["box"]
        for _ in range(int(times)):
            self.ctl.call("click", x=x + w / 2, y=y + h / 2)
            time.sleep(0.35)
        return f"нажато «{it.get('label')}» ×{times} в точке {int(x + w / 2)}, {int(y + h / 2)}"

    def exact(self):
        """Генератор останавливается (paused), кадр и истина — один момент."""
        try:
            self.cdp.evaluate("() => { window.paused = true; }")
            time.sleep(1.2)
            was, self.active = self.active, True
            n0 = self.seq
            t_end = time.time() + 30
            while self.seq < n0 + 2 and time.time() < t_end:
                time.sleep(0.2)
            st = self.snapshot()
            self.active = was
        finally:
            self.cdp.evaluate("() => { window.paused = false; }")
        fails = [{"id": o["id"], "cls": o["cls"], "read": o["read"], "truth": o["truth"]} for o in st.get("objects", []) if o.get("ok") is False]
        ok, tot = st["frame"]["ok"], st["frame"]["total"]
        return {"ok": ok, "total": tot, "acc": round(100 * ok / max(1, tot), 2), "layout": f"зерно {self.seed}",
                "clock": "—", "fails": fails}

    def detail(self, oid):
        fr = self.frame()
        it = self.item(oid)
        if fr is None or it is None:
            return {"error": "объект не найден на последнем кадре" if fr is not None else "нет кадра"}
        img = fr[1]
        x, y, w, h = it["box"]
        pad = 30 if it["type"] in ("gauge", "tank") else 12
        H, W = img.shape[:2]
        crop = img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad - (40 if it["type"] == "value" else 0)):min(W, x + w + pad)].copy()
        steps = {
            "gauge": ["Круг найден HoughCircles (радиус 55–170 px).",
                      "Числа шкалы — «чернила» циферблата: контраст с его фоном, без яркой насыщенной стрелки; риски (тонкие длинные штрихи) отброшены; цифры собраны в числа и прочитаны Tesseract только цифрами.",
                      "Угол каждого числа → значение: прямая RANSAC по парам (неверно прочитанное число и цифровое табло — выбросы).",
                      "Стрелка — луч от центра, сильнее всего отличающийся по цвету (Lab) от циферблата на 0,18–0,62 радиуса; угол уточнён центром масс пика.",
                      "Подпись — строка под кругом; если на приборе есть цифровое табло и оно близко к стрелке — берётся оно."],
            "bar": ["Прямоугольник шкалы (Canny + контур); рамка дорожки в тон фону — дорожка дотянута по цвету столбцов до фона панели.",
                    "Заливка — доля столбцов цвета левого края (заливка идёт слева).",
                    "Подпись — строка над шкалой, число «NN %» под ней — если близко к заливке, берётся оно."],
            "tank": ["Вертикальный прямоугольник резервуара, заливка — доля заполненных строк (насыщенность × яркость).",
                     "Подпись и число — строки под резервуаром."],
            "lamp": ["Маленький круг (HoughCircles, радиус 6–16 px) вне приборов, справа от него — подпись.",
                     "Горит — если цвет внутри насыщенный и яркий (HSV: S > 110, V > 150)."],
            "button": ["Прямоугольник 26–72 px высотой с одной короткой надписью внутри.",
                       "Включена — если заливка насыщенная и яркая (HSV: S > 115, V > 115)."],
            "value": ["Строка с числом: строки текста найдены морфологическим градиентом, каждая прочитана Tesseract (rus+eng) со своей полярностью.",
                      "Число — то, после которого единица или конец строки; «Насос 1», «CH4» — номер в подписи, а не значение.",
                      "Подпись — строка слева в той же рамке табло, иначе — над числом."],
        }[it["type"]]
        return {"id": oid, "kind": "zero", "class_title": TITLES.get(it["type"], it["type"]), "block": None,
                "images": {"crop": png64(crop, 2.0 if max(crop.shape[:2]) < 260 else 1.0)},
                "steps": steps, "result": {k: v for k, v in it.items() if k not in ("box", "id")}}


def main():
    from extra_reader import Tess
    ocr = OCR(Tess())
    tot = defaultdict(lambda: {"n": 0, "found": 0, "label": 0, "value": 0})
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        truth = json.loads(Path(p).with_suffix(".json").read_text())
        items, info = analyze(img, ocr)
        res, matched = evaluate(items, truth)
        print(f"{Path(p).name}: {info}")
        for typ, r in sorted(res.items()):
            print(f"   {typ:7s} найдено {r['found']}/{r['n']}  подпись {r['label']}/{r['found']}  значение/состояние {r['value']}/{r['found']}")
            for k in r:
                tot[typ][k] += r[k]
        if "-v" in sys.argv[0:1] or "--verbose" in sys.argv:
            for w, it in matched:
                print("      ", w["type"], w.get("label"), w.get("value", w.get("state")), "→", it and (it.get("label"), it.get("value", it.get("state"))))
    print("ИТОГО:")
    for typ, r in sorted(tot.items()):
        print(f"   {typ:7s} найдено {r['found']}/{r['n']} ({100 * r['found'] / max(1, r['n']):.0f} %)  подпись {r['label']}/{r['found']}  "
              f"значение/состояние {r['value']}/{r['found']}")


if __name__ == "__main__":
    main()
