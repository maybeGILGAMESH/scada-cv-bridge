# -*- coding: utf-8 -*-
"""Приведение кадра трансляции к холсту щита 1920x1080.

Через KVM приходит не тот кадр, что рисует браузер: он отмасштабирован под
разрешение захвата, обрезан или дополнен полями, пережат JPEG. matchTemplate
не прощает масштаба, поэтому перед распознаванием кадр надо вернуть к
исходному размеру щита.

Масштаб и положение ищутся по изображению — многомасштабным сопоставлением
неизменного куска шапки щита (название куста и скважины). Шапка не входит ни
в один переставляемый блок, поэтому годится как опора при любой компоновке.

Найденное преобразование держится, пока держится балл: пересчёт запускается
только когда сопоставление начинает сыпаться (сменилось разрешение захвата,
окно передвинули, поток перезапустился).
"""

import cv2
import numpy as np

PANEL_W, PANEL_H = 1920, 1080

# Неизменный кусок шапки: «АСУ ТП ДОБЫЧИ · КУСТ 12 · СКВАЖИНА 1247…» и строка
# НГДУ под ним. Ни часы, ни строка состояния сюда не попадают.
HEADER_RECT = (12, 6, 640, 50)


class Calibration:
    """Найденное преобразование кадра к холсту щита."""

    def __init__(self, scale, x, y, score):
        self.scale = float(scale)
        self.x = float(x)              # левый верхний угол щита в кадре
        self.y = float(y)
        self.score = float(score)

    @property
    def identity(self):
        return abs(self.scale - 1.0) < 1e-3 and abs(self.x) < 0.5 and abs(self.y) < 0.5

    def __repr__(self):
        return (f"масштаб {self.scale:.3f}, начало щита ({self.x:.0f},{self.y:.0f}), "
                f"балл {self.score:.3f}")


def _match_at(frame, tpl, scale):
    h, w = tpl.shape[:2]
    tw, th = int(round(w * scale)), int(round(h * scale))
    if tw < 24 or th < 8 or tw > frame.shape[1] or th > frame.shape[0]:
        return -1.0, 0, 0
    t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA
                   if scale < 1 else cv2.INTER_CUBIC)
    res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), loc[0], loc[1]


def calibrate(frame, ref, rect=HEADER_RECT, lo=0.45, hi=1.35):
    """Ищет масштаб и положение щита в кадре. Возвращает Calibration или None.

    Два прохода: грубый с шагом 0,04 по всему диапазону, затем уточняющий
    с шагом 0,004 вокруг найденного масштаба.
    """
    x, y, w, h = rect
    tpl = ref[y:y + h, x:x + w]
    if tpl.size == 0:
        return None

    # Грубый проход идёт на половинном разрешении: перебор масштабов —
    # самая дорогая часть, а на половине кадра он вчетверо дешевле и той же
    # точности хватает, чтобы задать окно для уточнения.
    half = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2),
                      interpolation=cv2.INTER_AREA)
    coarse = (-1.0, 1.0)
    for sc in np.arange(lo, hi + 1e-9, 0.04):
        s, _, _ = _match_at(half, tpl, sc / 2.0)
        if s > coarse[0]:
            coarse = (s, float(sc))

    best = (-1.0, 1.0, 0, 0)
    c = coarse[1]
    for sc in np.arange(max(lo, c - 0.05), min(hi, c + 0.05) + 1e-9, 0.004):
        s, mx, my = _match_at(frame, tpl, sc)
        if s > best[0]:
            best = (s, float(sc), mx, my)
    if best[0] < 0:
        return None

    score, scale, mx, my = best
    return Calibration(scale, mx - x * scale, my - y * scale, score)


def apply(frame, cal):
    """Вырезает щит и растягивает его обратно к 1920x1080.

    Если часть щита вышла за границу кадра (обрезали при захвате), недостающее
    добивается чёрным: лучше честно отдать неполный щит и получить низкий балл
    на потерянных блоках, чем незаметно сдвинуть всю картинку.
    """
    if cal is None:
        return frame
    if cal.identity and frame.shape[1] == PANEL_W and frame.shape[0] == PANEL_H:
        return frame

    w = int(round(PANEL_W * cal.scale))
    h = int(round(PANEL_H * cal.scale))
    x0, y0 = int(round(cal.x)), int(round(cal.y))
    canvas = np.zeros((h, w, 3), np.uint8)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]

    if (w, h) == (PANEL_W, PANEL_H):
        return canvas
    interp = cv2.INTER_AREA if cal.scale > 1 else cv2.INTER_CUBIC
    return cv2.resize(canvas, (PANEL_W, PANEL_H), interpolation=interp)


class Normalizer:
    """Держит калибровку и пересчитывает её, когда распознавание поплыло."""

    def __init__(self, ref, rect=HEADER_RECT, min_score=0.45, patience=12):
        self.ref = ref
        self.rect = rect
        self.min_score = min_score
        self.patience = patience
        self.cal = None
        self.bad = 0
        self.recalibrations = 0

    def force(self):
        self.cal = None

    def note_quality(self, quality):
        """Балл привязки блоков за кадр. Долгая просадка — повод пересчитать."""
        if quality < self.min_score:
            self.bad += 1
            if self.bad >= self.patience:
                self.cal = None
                self.bad = 0
        else:
            self.bad = 0

    def __call__(self, frame):
        if frame is None:
            return None
        if self.cal is None:
            self.cal = calibrate(frame, self.ref, self.rect)
            self.recalibrations += 1
        return apply(frame, self.cal)
