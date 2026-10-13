#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Распознавание щита АСУ ТП по эталонным образцам через cv2.matchTemplate
и сверка результата с истинными значениями модели.

    python match_templates.py                          # весь контрольный набор режимов
    python match_templates.py --reference               # только тот кадр, откуда нарезаны эталоны
    python match_templates.py --frame кадр.png --truth кадр.json
    python match_templates.py --dataset ../scada_cv_dataset --limit 50
    python match_templates.py --only gauge,sevenseg    # выборочно
    python match_templates.py --save-overlay out.png   # картинка с рамками

По умолчанию проверка идёт по папке validation/ — это 14 кадров разных режимов
(работа, останов, авария, срыв подачи, дросселирование, крайние точки шкал),
каждый со своим JSON истинных значений. Сверять только на reference/frame.png
бессмысленно: именно из него нарезаны эталоны.

Эталоны и manifest.json выгружаются кнопкой «Выгрузить эталоны» в панели
neft-hmi-cv.html. Шаблоны вырезаны из кадра один к одному, поэтому кадры
для сверки должны быть в том же масштабе (аугментации отключить либо
оставить только фотометрические — поворот и масштаб matchTemplate не прощает).
"""

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread на Windows не открывает пути с кириллицей: внутри он зовёт
    однобайтовый fopen. Читаем файл сами и распаковываем из памяти — имена
    вроде templates/selector/sw_mode_ручн.png так читаются на любой системе."""
    p = Path(path)
    if not p.exists():
        return None
    data = np.frombuffer(p.read_bytes(), np.uint8)
    return cv2.imdecode(data, flags)


def imwrite(path, img):
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if ok:
        p.write_bytes(buf.tobytes())
    return ok

# Щит разложен по блокам; компоновка переставляет блоки целиком, не меняя ни
# габарита прибора, ни его отрисовки. Габариты блоков базовой компоновки нужны
# на случай манифеста, выгруженного до появления поля "blocks".
BLOCK_RECT_BASE = {
    "mimic":    [12,   70,  900, 700],
    "gauges":   [928,  76,  684, 456],
    "displays": [928, 534,  680, 234],
    "right":    [1620, 70,  288, 700],
    "console":  [1620, 780, 288, 288],
    "bars":     [12,  780,  460, 288],
    "trends":   [482, 780, 1126, 240],
    "log":      [482, 1026, 1126, 42],
}


# ────────────────────────────────────────────────────────────── загрузка

class Bank:
    """Эталоны, разложенные по объектам."""

    def __init__(self, root: Path):
        self.root = root
        mf = root / "manifest.json"
        if not mf.exists():
            sys.exit(f"Не найден {mf}\n"
                     f"Откройте neft-hmi-cv.html и нажмите «Выгрузить эталоны», "
                     f"указав папку {root}")
        self.man = json.loads(mf.read_text(encoding="utf-8"))
        self.by_obj = defaultdict(list)
        self._img, self._msk = {}, {}
        for t in self.man["templates"]:
            self.by_obj[t["object_id"]].append(t)
        self.classes = {}
        for oid, lst in self.by_obj.items():
            self.classes[oid] = lst[0]["class"]
        self.signature = self.man.get("layout_signature")
        self.layout_id = self.man.get("layout_id")

        # Контрольный кадр и его разметка: из кадра вырезаются опорные куски
        # блоков, из разметки берутся знакоместа индикаторов. И то и другое —
        # свойства эталонной компоновки, а не проверяемого кадра.
        self.ref_path = root / "reference" / "frame.png"
        self._ref = None
        self.ref_ann = {}
        rj = root / "reference" / "frame.json"
        if rj.exists():
            self.ref_ann = json.loads(rj.read_text(encoding="utf-8"))

        self.blocks = self.man.get("blocks") or dict(BLOCK_RECT_BASE)
        self.block_of = {}
        for t in self.man["templates"]:
            b = t.get("block")
            if b:
                self.block_of[t["object_id"]] = b
        if not self.block_of:                 # манифест без поля block
            self.block_of = self._blocks_from_reference()

        # Знакоместа семисегментных индикаторов в эталонной компоновке.
        self.cells = {}
        for o in self.ref_ann.get("objects", []):
            if o.get("type") == "sevenseg":
                self.cells[o["id"]] = [d["bbox"] for d in o.get("digits", [])
                                       if d.get("char") != "."]

        self.offsets = {}                     # блок → (dx, dy) в текущем кадре

    def _blocks_from_reference(self):
        """Принадлежность прибора блоку — по попаданию его габарита внутрь."""
        out = {}
        for o in self.ref_ann.get("objects", []):
            bb = o.get("bbox")
            if not bb:
                continue
            for name, r in self.blocks.items():
                if (bb[0] >= r[0] - 3 and bb[1] >= r[1] - 3
                        and bb[0] + bb[2] <= r[0] + r[2] + 3
                        and bb[1] + bb[3] <= r[1] + r[3] + 3):
                    out[o["id"]] = name
                    break
        return out

    def ref(self):
        if self._ref is None:
            self._ref = imread(self.ref_path, cv2.IMREAD_COLOR)
            if self._ref is None:
                sys.exit(f"Не читается контрольный кадр {self.ref_path}")
        return self._ref

    def set_offsets(self, offsets):
        """Сдвиги блоков, найденные в текущем кадре."""
        self.offsets = offsets or {}

    def shift(self, object_id):
        return self.offsets.get(self.block_of.get(object_id), (0, 0))

    def anchor(self, rec):
        """Якорь образца с поправкой на то, куда в этом кадре уехал его блок."""
        dx, dy = self.shift(rec["object_id"])
        return (rec["anchor"][0] + dx, rec["anchor"][1] + dy)

    def cells_of(self, object_id):
        """Знакоместа индикатора в текущем кадре: эталонные плюс сдвиг блока."""
        dx, dy = self.shift(object_id)
        return [[c[0] + dx, c[1] + dy, c[2], c[3]] for c in self.cells.get(object_id, [])]

    def check_layout(self, truth):
        """Эталоны, нарезанные с другой версии щита, дают тихую деградацию
        точности вместо явной ошибки — поэтому сверяем отпечаток компоновки."""
        got = (truth or {}).get("layout_signature")
        if self.signature and got and self.signature != got:
            print("ВНИМАНИЕ: кадр снят с другой компоновки щита.\n"
                  f"  эталоны  : {self.signature}\n"
                  f"  кадр     : {got}\n"
                  "  Перевыгрузите эталоны кнопкой «Выгрузить эталоны».\n")
            return False
        return True

    def img(self, rec):
        p = rec["file"]
        if p not in self._img:
            a = imread(self.root / p, cv2.IMREAD_COLOR)
            if a is None:
                sys.exit(f"Не читается шаблон {p}")
            self._img[p] = a
        return self._img[p]

    def mask(self, rec):
        p = rec.get("mask")
        if not p:
            return None
        if p not in self._msk:
            m = imread(self.root / p, cv2.IMREAD_GRAYSCALE)
            self._msk[p] = None if m is None else cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        return self._msk[p]


# ─────────────────────────────────────────────────── ядро сопоставления

def match(frame, tpl, mask=None, anchor=None, pad=36):
    """Лучшее совпадение шаблона в окрестности якоря. Возвращает (score, x, y)."""
    H, W = frame.shape[:2]
    th, tw = tpl.shape[:2]
    if anchor is not None:
        x0 = max(0, int(anchor[0]) - pad)
        y0 = max(0, int(anchor[1]) - pad)
        x1 = min(W, int(anchor[0]) + tw + pad)
        y1 = min(H, int(anchor[1]) + th + pad)
    else:
        x0 = y0 = 0
        x1, y1 = W, H
    roi = frame[y0:y1, x0:x1]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return -1.0, 0, 0

    if mask is not None and mask.shape[:2] == tpl.shape[:2]:
        # с маской OpenCV поддерживает только CCORR_NORMED и SQDIFF
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCORR_NORMED, mask=mask)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    else:
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    px, py = x0 + loc[0], y0 + loc[1]
    # На однородном участке (прибор чем-то перекрыт) знаменатель корреляции
    # вырождается и OpenCV возвращает 1.0 любому шаблону. Такой балл — мусор.
    patch = frame[py:py + th, px:px + tw]
    if patch.size and float(patch.std()) < 3.0:
        return 0.0, px, py
    return float(mx), px, py


def best_variant(frame, bank, recs, pad=36):
    """Из нескольких состояний одного прибора выбирает набравшее больший балл."""
    best = None
    for r in recs:
        s, x, y = match(frame, bank.img(r), bank.mask(r), bank.anchor(r), pad)
        if best is None or s > best[0]:
            best = (s, x, y, r)
    return best


def match_sqdiff(frame, tpl, mask, anchor, pad):
    """Поиск с маской: OpenCV поддерживает маску только для SQDIFF и CCORR_NORMED."""
    H, W = frame.shape[:2]
    th, tw = tpl.shape[:2]
    x0 = max(0, int(anchor[0]) - pad); y0 = max(0, int(anchor[1]) - pad)
    x1 = min(W, int(anchor[0]) + tw + pad); y1 = min(H, int(anchor[1]) + th + pad)
    roi = frame[y0:y1, x0:x1]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return 0.0, int(anchor[0]), int(anchor[1])
    res = cv2.matchTemplate(roi, tpl, cv2.TM_SQDIFF, mask=mask)
    res = np.nan_to_num(res, nan=np.inf, posinf=np.inf, neginf=np.inf)
    mn, _, loc, _ = cv2.minMaxLoc(res)
    npx = float(np.count_nonzero(mask[:, :, 0])) * 3 * 255.0 ** 2
    score = max(0.0, 1.0 - mn / npx) if npx > 0 else 0.0
    return float(score), x0 + loc[0], y0 + loc[1]


def invariant_mask(a, b, tol=10):
    """Пиксели, одинаковые у двух состояний прибора, — это неизменная часть
    корпуса. Именно по ней надёжно ищется положение прибора, у которого
    показание меняет внешний вид (заливка шкалы, столб жидкости, цифры)."""
    d = cv2.absdiff(a, b).max(axis=2)
    m = (d <= tol).astype(np.uint8) * 255
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    if np.count_nonzero(m) < 0.05 * m.size:      # почти всё меняется — маска бесполезна
        return None
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def diff_profile(a, b, rect, axis):
    """Профиль отличия двух картинок в пределах прямоугольника.
    axis=0 — по столбцам (шкала), axis=1 — по строкам (столб жидкости)."""
    x, y, w, h = [int(v) for v in rect]
    pa = a[y:y + h, x:x + w].astype(np.int16)
    pb = b[y:y + h, x:x + w].astype(np.int16)
    if pa.shape != pb.shape or pa.size == 0:
        return None
    d = np.abs(pa - pb).max(axis=2)          # отличие на пиксель
    return np.percentile(d, 75, axis=axis)   # устойчиво к отдельным штрихам шкалы


def fill_fraction(frame_crop, empty_tpl, full_tpl, rect, axis, from_end):
    """Доля заполнения меряется как отличие от эталона «пусто», а не по яркости.

    Яркостный порог ломался, когда цвет заливки менялся по тревоге: красная полоса
    темнее синей, и калиброванный по синему эталону порог давал ноль. Отличие от
    пустой шкалы срабатывает при любом цвете."""
    ref = diff_profile(full_tpl, empty_tpl, rect, axis)     # каким бывает «занято»
    cur = diff_profile(frame_crop, empty_tpl, rect, axis)   # что видим сейчас
    if ref is None or cur is None or len(cur) == 0:
        return None
    n = len(cur)
    lvl = float(np.median(ref[ref > 1])) if np.any(ref > 1) else 0.0
    if lvl < 8:
        return None

    # Полосы, где «пусто» и «полно» выглядят одинаково, ничего не решают: у аппарата
    # это неснижаемый остаток на дне, у шкалы — краевой пиксель. Их нельзя измерять,
    # но и обрывать на них отсчёт нельзя — они заведомо относятся к заполненной части.
    # Порог значимости намеренно высокий: полосы со слабым отличием эталонов
    # (блик на поверхности, кромка аппарата) давали ложную границу вплотную ко дну.
    idx = np.flatnonzero(ref > lvl * 0.50)
    if idx.size == 0:
        return None
    lit = cur > lvl * 0.35

    # Идём только по значимым полосам. Пропуская незначимые, мы заодно проходим
    # сквозь наложенные врезки с показаниями: они выглядят одинаково у обоих
    # эталонов, поэтому разрывают заливку, но границу уровня не определяют.
    order = idx[::-1] if from_end else idx
    boundary = None
    for k in order:
        if not lit[k]:
            boundary = int(k)
            break
    if boundary is None:
        return 1.0
    return (n - 1 - boundary) / n if from_end else boundary / n


# ──────────────────────────────────────────────── привязка компоновки

def _pyr_match(frame, tpl, k):
    """Грубое совпадение на уменьшенной в k раз паре. Возвращает (score, x, y)
    в координатах полного кадра."""
    fh, fw = frame.shape[:2]
    th, tw = tpl.shape[:2]
    f = cv2.resize(frame, (fw // k, fh // k), interpolation=cv2.INTER_AREA)
    t = cv2.resize(tpl, (max(1, tw // k), max(1, th // k)), interpolation=cv2.INTER_AREA)
    if f.shape[0] < t.shape[0] or f.shape[1] < t.shape[1]:
        return -1.0, 0, 0
    res = cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), loc[0] * k, loc[1] * k


def register_blocks(frame, bank, refine=14, verbose=False):
    """Находит в кадре каждый блок щита и возвращает его сдвиг.

    Блок ищется как изображение: из контрольного кадра вырезается кусок
    блока целиком и сопоставляется со всем кадром. Ни одна координата
    прибора при этом не используется — именно поэтому перестановка блоков
    распознавание не ломает, а показания читаются в найденном месте.

    Поиск двухступенчатый: грубо на уменьшенной вчетверо паре (там
    matchTemplate идёт через БПФ и стоит недорого), затем уточнение на
    полном разрешении в окрестности найденной точки.
    """
    ref = bank.ref()
    H, W = frame.shape[:2]
    out = {}
    for name, r in bank.blocks.items():
        x, y, w, h = [int(v) for v in r]
        if x < 0 or y < 0 or x + w > ref.shape[1] or y + h > ref.shape[0]:
            continue
        tpl = ref[y:y + h, x:x + w]
        if tpl.shape[0] > H or tpl.shape[1] > W:
            continue
        sc, cx, cy = _pyr_match(frame, tpl, 4)
        if sc < 0:
            continue
        # уточнение: полное разрешение, окно вокруг грубой точки
        x0 = max(0, cx - refine); y0 = max(0, cy - refine)
        x1 = min(W, cx + tpl.shape[1] + refine); y1 = min(H, cy + tpl.shape[0] + refine)
        roi = frame[y0:y1, x0:x1]
        if roi.shape[0] >= tpl.shape[0] and roi.shape[1] >= tpl.shape[1]:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            sc, cx, cy = float(mx), x0 + loc[0], y0 + loc[1]
        out[name] = (int(cx - x), int(cy - y), float(sc))
        if verbose:
            print(f"  блок {name:<9} сдвиг {cx - x:+5d},{cy - y:+5d}  балл {sc:.3f}")
    return out


def verify_block(frame, bank, name, off, pad=8, patch=(280, 160)):
    """Проверяет, что блок остался там, где его нашли в прошлый раз.

    Дешёвая замена полному поиску: небольшой кусок блока сопоставляется в
    окрестности ожидаемого места. Нужна, когда поиск блоков идёт не на каждом
    кадре: без такой проверки перестановка блоков осталась бы незамеченной, а
    приборы продолжили бы читаться по устаревшим сдвигам — то есть выдавали бы
    уверенные, но неверные числа.

    Возвращает (балл, отклонение по X, отклонение по Y) либо None.
    """
    r = bank.blocks.get(name)
    if not r:
        return None
    ref = bank.ref()
    x, y, w, h = [int(v) for v in r]
    pw, ph = min(patch[0], w), min(patch[1], h)
    tpl = ref[y:y + ph, x:x + pw]
    ax, ay = x + off[0], y + off[1]
    s, fx, fy = match(frame, tpl, None, (ax, ay), pad)
    return float(s), int(fx - ax), int(fy - ay)


def offsets_only(reg):
    return {k: (v[0], v[1]) for k, v in reg.items()}


def layout_of(reg, min_score=0.5):
    """Краткая сводка: сколько блоков переехало и насколько уверенно найдены."""
    moved = sum(1 for v in reg.values() if v[0] or v[1])
    worst = min([v[2] for v in reg.values()], default=0.0)
    return moved, len(reg), worst


# ──────────────────────────────────────────────── чтение по типам приборов

def read_gauge(frame, bank, recs, pad=30, stride=1,
               hint=None, hint_span=4, hint_min=0.88):
    """Перебор шаблонов по углу + параболическое уточнение по трём точкам.

    hint — угол, прочитанный на предыдущем кадре. Стрелка между соседними
    кадрами далеко не уходит, поэтому сначала проверяются несколько углов
    вокруг подсказки. Если там нашёлся уверенный максимум и он не на краю
    проверенного окна, дальше искать нечего: вместо 55 сопоставлений выходит
    девять. Иначе идёт полный перебор — подсказка ничего не решает сама по
    себе, она только сокращает работу, когда оказалась верной.

    stride>1 — грубый проход по каждому stride-му углу с уточнением вокруг
    лучшего. Втрое быстрее, но на краях шкалы даёт промахи (проверено:
    932/938 против 933/938), поэтому по умолчанию выключен.
    """
    recs = sorted(recs, key=lambda r: r["angle_deg"])
    n = len(recs)
    scores = [None] * n

    def score_at(k):
        if scores[k] is None:
            s, _x, _y = match(frame, bank.img(recs[k]), bank.mask(recs[k]),
                              bank.anchor(recs[k]), pad)
            scores[k] = s
        return scores[k]

    if hint is not None and n > 2 * hint_span + 2:
        j = min(range(n), key=lambda k: abs(recs[k]["angle_deg"] - hint))
        lo = max(0, j - hint_span)
        hi = min(n - 1, j + hint_span)
        win = list(range(lo, hi + 1))
        b = max(win, key=score_at)
        # максимум на краю окна означает, что настоящий пик может быть дальше
        if scores[b] >= hint_min and lo < b < hi:
            for k in (b - 1, b + 1):
                if 0 <= k < n:
                    score_at(k)
            sc = [-1.0 if v is None else v for v in scores]
            return _gauge_value(bank, recs, b, sc)

    if stride > 1 and n > 3 * stride:
        coarse = list(range(0, n, stride))
        if coarse[-1] != n - 1:
            coarse.append(n - 1)
        c = max(coarse, key=score_at)
        for k in range(max(0, c - stride), min(n, c + stride + 1)):
            score_at(k)
    else:
        for k in range(n):
            score_at(k)

    known = [k for k in range(n) if scores[k] is not None]
    i = max(known, key=lambda k: scores[k])
    # соседи нужны для параболического уточнения — досчитываем, если пропущены
    for k in (i - 1, i + 1):
        if 0 <= k < n:
            score_at(k)
    scores = [-1.0 if v is None else v for v in scores]
    return _gauge_value(bank, recs, i, scores)


def _gauge_value(bank, recs, i, scores):
    """Угол лидера, уточнённый параболой по трём соседним баллам, и значение."""
    r = recs[i]
    ang = r["angle_deg"]
    if 0 < i < len(recs) - 1 and scores[i - 1] >= 0 and scores[i + 1] >= 0:
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            step = recs[i + 1]["angle_deg"] - recs[i]["angle_deg"]
            ang += 0.5 * (a - c) / den * step
    frac = (ang - bank.man["gauge_scale"]["start_deg"]) / bank.man["gauge_scale"]["sweep_deg"]
    frac = float(np.clip(frac, 0.0, 1.0))
    val = r["min"] + (r["max"] - r["min"]) * frac
    return {"value": val, "angle_deg": ang, "score": scores[i], "unit": r.get("unit")}


MARGIN_MIN = 0.01          # ниже этого отрыва выбор лидера ничего не значит


def read_sevenseg(frame, bank, recs, cells, pad=8):
    """Каждое знакоместо сравнивается со всеми цифрами своего индикатора.

    Решающий признак достоверности — не сам балл, а отрыв лидера от второго
    места. Когда прибор чем-то перекрыт, корреляция вырождается и OpenCV
    выдаёт 1.0 сразу всем шаблонам: балл идеальный, а информации ноль."""
    glyphs = {r["char"]: r for r in recs}
    out, conf, margins = [], [], []
    for cell in cells:
        anchor = (cell[0], cell[1])
        cand = [g for g in glyphs.values()
                if abs(g["w"] - cell[2]) <= 3 and abs(g["h"] - cell[3]) <= 3]
        if not cand:
            cand = list(glyphs.values())
        sc = sorted(((match(frame, bank.img(g), None, anchor, pad)[0], g["char"])
                     for g in cand), key=lambda z: -z[0])
        margin = sc[0][0] - sc[1][0] if len(sc) > 1 else 1.0
        out.append(sc[0][1] if margin >= MARGIN_MIN else "?")
        conf.append(sc[0][0])
        margins.append(margin)
    text = "".join(out)
    try:
        value = float(text)
    except ValueError:
        value = None
    return {"text": text, "value": value,
            "score": float(np.mean(conf)) if conf else 0.0,
            "margin": float(min(margins)) if margins else 0.0,
            "reliable": all(mg >= MARGIN_MIN for mg in margins)}


def read_discrete(frame, bank, recs, pad=26):
    """Состояние определяется в два приёма.

    Сначала прибор находится корреляцией — она устойчива к яркости и шуму.
    Затем состояние выбирается прямым сравнением пикселей в найденной точке:
    нормированная корреляция плохо различает два похожих тёмных символа
    (открытый и закрытый обратный клапан различались на 0,004), а средняя
    абсолютная разность разводит их на порядок увереннее."""
    loc = None
    for r in recs:                       # позицию берём у самого уверенного варианта
        s, x, y = match(frame, bank.img(r), bank.mask(r), bank.anchor(r), pad)
        if loc is None or s > loc[0]:
            loc = (s, x, y)
    _, x, y = loc

    scored = []
    for r in recs:
        tpl = bank.img(r)
        crop = _crop_like(frame, x, y, tpl)
        d = np.abs(crop.astype(np.int16) - tpl.astype(np.int16)).mean(axis=2)
        msk = bank.mask(r)
        if msk is not None and msk.shape[:2] == tpl.shape[:2]:
            sel = msk[:, :, 0] > 127
            d = d[sel] if sel.any() else d
        scored.append((1.0 - float(d.mean()) / 255.0, r))
    scored.sort(key=lambda z: -z[0])
    s, r = scored[0]
    margin = s - scored[1][0] if len(scored) > 1 else 1.0
    ok = margin >= 0.004
    return {"state": r.get("state") if ok else None, "score": s, "at": (x, y),
            "margin": float(margin), "reliable": ok,
            "opening_pct": r.get("opening_pct") if ok else None}


def _locate_variable(frame, bank, empty, full, pad):
    """Прибор, чей вид зависит от показания, ищется по неизменной части корпуса."""
    ei, fi = bank.img(empty), bank.img(full)
    m = invariant_mask(ei, fi)
    if m is None:
        return match(frame, ei, None, bank.anchor(empty), pad)
    return match_sqdiff(frame, ei, m, bank.anchor(empty), pad)


def _crop_like(frame, x, y, tpl):
    """Кусок кадра ровно того же размера, что шаблон, — для попиксельного вычитания."""
    th, tw = tpl.shape[:2]
    H, W = frame.shape[:2]
    if x < 0 or y < 0 or x + tw > W or y + th > H:
        out = np.zeros_like(tpl)
        xs, ys = max(0, x), max(0, y)
        xe, ye = min(W, x + tw), min(H, y + th)
        if xe > xs and ye > ys:
            out[ys - y:ye - y, xs - x:xe - x] = frame[ys:ye, xs:xe]
        return out
    return frame[y:y + th, x:x + tw]


def read_bargraph(frame, bank, recs, pad=20):
    """Корпус ищется по неизменной части, длина заливки — по отличию от «пусто»."""
    empty = next((r for r in recs if r.get("fill_frac") == 0), recs[0])
    full = next((r for r in recs if r.get("fill_frac") == 1), empty)
    rect = full.get("fill_rect_rel") or [10, 26, full["w"] - 20, full["h"] - 38]
    s, x, y = _locate_variable(frame, bank, empty, full, pad)
    ei, fi = bank.img(empty), bank.img(full)
    frac = fill_fraction(_crop_like(frame, x, y, ei), ei, fi, rect, axis=0, from_end=False)
    if frac is None:
        return {"value": None, "score": s}
    frac = float(np.clip(frac, 0.0, 1.0)) * float(full.get("fill_frac") or 1.0)
    val = full["min"] + (full["max"] - full["min"]) * frac
    return {"value": val, "fill_frac": frac, "score": s, "unit": full.get("unit"),
            "at": (x, y)}


def read_vessel(frame, bank, recs, pad=20):
    """Уровень — верхняя граница непрерывного столба жидкости от дна аппарата."""
    full = next((r for r in recs if (r.get("level_pct") or 0) > 50), recs[0])
    empty = next((r for r in recs if (r.get("level_pct") or 0) <= 50), recs[0])
    # Область измерения задаёт панель. Для сепаратора это смотровое стекло:
    # внутри аппарата мешают раздел фаз, пузырьки и градиентная заливка.
    rect = full.get("measure_rect_rel") or full.get("liquid_rect_rel")
    if not rect:
        return {"level_pct": None, "score": 0.0}
    s, x, y = _locate_variable(frame, bank, empty, full, pad)
    ei, fi = bank.img(empty), bank.img(full)
    frac = fill_fraction(_crop_like(frame, x, y, ei), ei, fi, rect, axis=1, from_end=True)
    if frac is None:
        return {"level_pct": None, "score": s}
    # Какой доле шкалы соответствует область: у столба жидкости эталон «полный»
    # снят на 95 %, у смотрового стекла — весь диапазон.
    span = float(full.get("measure_span_pct") or full.get("level_pct") or 100.0)
    return {"level_pct": float(np.clip(frac, 0, 1)) * span, "score": s, "at": (x, y)}


# ──────────────────────────────────────────────────────── прогон по кадру

CLASS_READER = {
    "gauge": "gauge", "sevenseg_digit": "sevenseg", "bargraph": "bargraph",
    "vessel_level": "vessel", "lamp": "discrete", "annunciator": "discrete",
    "button": "discrete", "valve": "discrete", "pump": "discrete",
    "apparatus": "discrete", "status_text": "discrete", "selector": "discrete",
    "trend": "locate",
    "text_readout": "locate",
}


def recognise(frame, bank, truth=None, only=None, pad=36, use_truth_cells=False,
              gauge_stride=1, hints=None):
    """Возвращает список результатов по каждому объекту щита.

    Знакоместа индикаторов берутся из разметки контрольного кадра и
    сдвигаются туда, где блок нашёлся в этом кадре. Истинная разметка
    проверяемого кадра для чтения не нужна — она только для сверки.
    """
    digits_by_obj = {}
    if use_truth_cells and truth:
        for o in truth.get("objects", []):
            if o["type"] == "sevenseg":
                digits_by_obj[o["id"]] = [d["bbox"] for d in o["digits"] if d["char"] != "."]

    rows = []
    for oid, recs in bank.by_obj.items():
        cls = bank.classes[oid]
        kind = CLASS_READER.get(cls, "locate")
        if only and cls not in only and kind not in only:
            continue

        if kind == "gauge":
            r = read_gauge(frame, bank, recs, stride=gauge_stride,
                           hint=(hints or {}).get(oid))
        elif kind == "sevenseg":
            cells = digits_by_obj.get(oid) or bank.cells_of(oid)
            if not cells:                       # без эталонного JSON берём якоря шаблонов
                w = recs[0]["w"]
                ax, ay = bank.anchor(recs[0])
                cells = [[ax + i * (w + w * 0.24), ay, w, recs[0]["h"]] for i in range(1)]
            r = read_sevenseg(frame, bank, recs, cells)
        elif kind == "bargraph":
            r = read_bargraph(frame, bank, recs)
        elif kind == "vessel":
            r = read_vessel(frame, bank, recs)
        elif kind == "discrete":
            r = read_discrete(frame, bank, recs)
        else:
            s, x, y = match(frame, bank.img(recs[0]), None, bank.anchor(recs[0]), pad)
            r = {"score": s, "at": (x, y)}
        r.update(object_id=oid, cls=cls, label=recs[0].get("label"))
        rows.append(r)
    rows.sort(key=lambda z: (z["cls"], z["object_id"]))
    return rows


# ──────────────────────────────────────────────────────────────── сверка

def compare(rows, truth, bank_texts=None):
    """Сопоставляет распознанное с истиной из аннотации кадра."""
    bank_texts = bank_texts or {}
    gt = {o["id"]: o for o in truth.get("objects", [])}
    stats = defaultdict(lambda: [0, 0])
    out = []
    for r in rows:
        g = gt.get(r["object_id"])
        exp = got = None
        ok = None
        if g is None:
            pass
        elif r["cls"] == "gauge":
            # Прибор физически не может показать больше своей шкалы: стрелка
            # упирается в предел. Сравниваем с тем, что он в состоянии показать.
            exp = min(max(g["value"], g["min"]), g["max"])
            got = r["value"]
            tol = 0.02 * (g["max"] - g["min"])
            ok = abs(exp - got) <= tol
        elif r["cls"] == "sevenseg_digit":
            exp, got = g["text"].replace(".", ""), (r["text"] or "")
            ok = exp == got
        elif r["cls"] == "bargraph":
            exp = min(max(g["value"], g["min"]), g["max"])
            got = r["value"]
            ok = got is not None and abs(exp - got) <= 0.06 * (g["max"] - g["min"])
        elif r["cls"] == "vessel_level":
            exp, got = g["level_pct"], r["level_pct"]
            ok = got is not None and abs(exp - got) <= 6.0
        elif r["cls"] == "selector":
            exp, got = g.get("value"), r.get("state")
            ok = exp == got
        elif r["cls"] == "status_text":
            recs = {t.get("state"): t.get("text") for t in bank_texts.get(r["object_id"], [])}
            exp, got = g.get("text"), recs.get(r.get("state"))
            ok = None if exp is None or got is None else exp.strip() == got.strip()
        elif "state" in r and "state" in g:
            exp, got = g["state"], r["state"]
            if r["cls"] in ("annunciator", "lamp"):  # мигание визуально совпадает с «включено»
                exp = "on" if g["is_lit"] else "off"
            ok = exp == got
        if ok is not None:
            s = stats[r["cls"]]
            s[1] += 1
            s[0] += int(ok)
        out.append((r, exp, got, ok))
    return out, stats


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description="Сверка щита АСУ ТП с эталонами через cv2.matchTemplate")
    ap.add_argument("--templates", default=str(ROOT), help="папка с manifest.json")
    ap.add_argument("--frame", default=None, help="кадр PNG/JPG")
    ap.add_argument("--truth", default=None, help="JSON с истинными значениями")
    ap.add_argument("--dataset", default=None, help="папка датасета (images/ + annotations/)")
    ap.add_argument("--limit", type=int, default=0, help="сколько кадров датасета взять")
    ap.add_argument("--only", default=None, help="классы через запятую")
    ap.add_argument("--pad", type=int, default=36, help="запас поиска вокруг якоря, px")
    ap.add_argument("--free", action="store_true",
                    help="не доверять координатам: сначала найти блоки щита в кадре, "
                         "потом читать приборы там, где блок нашёлся")
    ap.add_argument("--gauge-stride", type=int, default=1,
                    help="шаг грубого прохода по углам манометра: 1 — все углы подряд, "
                         "3 — втрое быстрее при той же точности")
    ap.add_argument("--anchored", action="store_true",
                    help="принудительно старый режим: якоря как есть, без поиска блоков")
    ap.add_argument("--save-overlay", default=None, help="сохранить кадр с рамками")
    ap.add_argument("--reference", action="store_true",
                    help="проверить на контрольном кадре, а не на наборе режимов")
    ap.add_argument("--report", default="report", metavar="ИМЯ",
                    help="куда сохранить отчёт: ИМЯ.csv и ИМЯ.md (пусто — не сохранять)")
    a = ap.parse_args()

    bank = Bank(Path(a.templates))
    only = set(a.only.split(",")) if a.only else None

    tasks, names = [], []
    val_index = Path(a.templates) / "validation" / "index.json"
    if a.dataset:
        d = Path(a.dataset)
        imgs = sorted((d / "images").glob("*.*"))
        if a.limit:
            imgs = imgs[: a.limit]
        for p in imgs:
            j = d / "annotations" / (p.stem + ".json")
            tasks.append((p, j if j.exists() else None))
            names.append(p.stem)
    elif a.frame:
        f = Path(a.frame)
        t = Path(a.truth) if a.truth else f.with_suffix(".json")
        tasks.append((f, t if t.exists() else None))
        names.append(f.stem)
    elif val_index.exists() and not a.reference:
        # По умолчанию проверяем на контрольном НАБОРЕ режимов, а не на том
        # единственном кадре, из которого нарезаны эталоны.
        vi = json.loads(val_index.read_text(encoding="utf-8"))
        for fr in vi["frames"]:
            tasks.append((Path(a.templates) / fr["image"], Path(a.templates) / fr["truth"]))
            names.append(fr["scenario"])
        if a.limit:
            tasks, names = tasks[: a.limit], names[: a.limit]
    else:
        f = Path(a.templates) / "reference" / "frame.png"
        tasks.append((f, f.with_suffix(".json")))
        names.append("reference")

    total = defaultdict(lambda: [0, 0])
    reg_log = []
    fails = []
    log_rows = []                      # построчный отчёт для выгрузки в файл
    t0 = time.time()

    for idx, (imgp, jsonp) in enumerate(tasks):
        frame = imread(imgp, cv2.IMREAD_COLOR)
        if frame is None:
            sys.exit(f"Не читается кадр {imgp}")
        truth = json.loads(Path(jsonp).read_text(encoding="utf-8")) if jsonp else None

        # Компоновка кадра может отличаться от той, в которой резались эталоны.
        # По умолчанию это выясняется по кадру: блоки ищутся изображением.
        # Старое поведение (читать по якорю) остаётся под --anchored.
        free = a.free or not a.anchored
        if free:
            reg = register_blocks(frame, bank)
            bank.set_offsets(offsets_only(reg))
            moved, nblk, worst = layout_of(reg)
            reg_log.append((names[idx], moved, nblk, worst))
            if idx == 0 or moved:
                lay = (truth or {}).get("layout_id")
                print(f"  компоновка: переехало блоков {moved} из {nblk}, "
                      f"худший балл привязки {worst:.3f}"
                      + (f", в кадре заявлена «{lay}»" if lay else ""))
        else:
            bank.set_offsets({})
            if idx == 0:
                bank.check_layout(truth)
        rows = recognise(frame, bank, truth, only, a.pad,
                         gauge_stride=a.gauge_stride)

        if truth:
            res, st = compare(rows, truth, bank.by_obj)
            for k, v in st.items():
                total[k][0] += v[0]
                total[k][1] += v[1]
        else:
            res = [(r, None, None, None) for r in rows]

        for r, exp, got, ok in res:
            shown = r.get("text") or r.get("state")
            if shown is None:
                shown = r.get("value", r.get("level_pct"))
            err = None
            if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
                err = round(float(got) - float(exp), 3)
            log_rows.append({
                "сценарий": names[idx], "класс": r["cls"], "объект": r["object_id"],
                "подпись": r.get("label") or "", "распознано": shown,
                "истина": exp, "ошибка": err,
                "балл": round(float(r.get("score") or 0), 4),
                "отрыв": None if r.get("margin") is None else round(float(r["margin"]), 4),
                "достоверно": "" if r.get("reliable") is None else ("да" if r["reliable"] else "нет"),
                "итог": "" if ok is None else ("верно" if ok else "ошибка"),
            })

        if len(tasks) == 1:
            print(f"\nКадр: {imgp.name}   объектов: {len(rows)}\n")
            print(f"{'КЛАСС':<15}{'ОБЪЕКТ':<16}{'РАСПОЗНАНО':>16}{'ЭТАЛОН':>16}"
                  f"{'БАЛЛ':>8}{'ОТРЫВ':>8}   ")
            print("─" * 86)
            for r, exp, got, ok in res:
                shown = r.get("text") or r.get("state")
                if shown is None:
                    shown = r.get("value", r.get("level_pct"))
                mark = "" if ok is None else ("  OK" if ok else "  ✗")
                if r.get("reliable") is False:
                    mark += "  НЕДОСТОВЕРНО"
                mg = r.get("margin")
                print(f"{r['cls']:<15}{r['object_id']:<16}{fmt(shown):>16}"
                      f"{fmt(exp):>16}{r.get('score', 0):>8.3f}"
                      f"{('—' if mg is None else f'{mg:.3f}'):>8}{mark}")
        else:
            nok = sum(1 for _, _, _, o in res if o)
            ntot = sum(1 for _, _, _, o in res if o is not None)
            bad = [r["object_id"] for r, _, _, o in res if o is False]
            if idx == 0:
                print(f"\n{'СЦЕНАРИЙ':<36}{'ВЕРНО':>10}   расхождения")
                print("─" * 92)
            print(f"  {names[idx]:<34}{nok:>4} / {ntot:<4}   "
                  + (", ".join(bad[:5]) + ("…" if len(bad) > 5 else "") if bad else "—"))
            for r, exp, got, o in res:
                if o is False:
                    fails.append((names[idx], r["cls"], r["object_id"], exp, got,
                                  r.get("margin"), r.get("reliable")))

        if a.save_overlay and idx == 0:
            vis = frame.copy()
            for r in rows:
                recs = bank.by_obj[r["object_id"]]
                x, y = r.get("at", bank.anchor(recs[0]))
                w, h = recs[0]["w"], recs[0]["h"]
                cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 220, 160), 1)
                cv2.putText(vis, r["object_id"], (int(x), int(y) - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 220, 160), 1, cv2.LINE_AA)
            imwrite(a.save_overlay, vis)
            print(f"  разметка сохранена: {a.save_overlay}")

    if total:
        print("\nТочность распознавания")
        print("─" * 46)
        ok = sum(v[0] for v in total.values())
        n = sum(v[1] for v in total.values())
        for k in sorted(total):
            c, t = total[k]
            print(f"  {k:<18}{c:>6} / {t:<6} {100 * c / max(t, 1):>6.1f} %")
        print("─" * 46)
        print(f"  {'ИТОГО':<18}{ok:>6} / {n:<6} {100 * ok / max(n, 1):>6.1f} %")
        if fails:
            print("\nРазбор расхождений")
            print("─" * 92)
            print(f"  {'СЦЕНАРИЙ':<32}{'ОБЪЕКТ':<15}{'РАСПОЗНАНО':>14}{'ИСТИНА':>14}{'ОТРЫВ':>8}")
            for sc, cls, oid, exp, got, mg, rel in fails[:30]:
                note = "  недостоверно" if rel is False else ""
                print(f"  {sc:<32}{oid:<15}{fmt(got):>14}{fmt(exp):>14}"
                      f"{('—' if mg is None else f'{mg:.3f}'):>8}{note}")
            if len(fails) > 30:
                print(f"  … и ещё {len(fails) - 30}")

    if a.report and log_rows:
        write_report(Path(a.report), log_rows, total, fails, names, bank, t0, len(tasks))
        print(f"\nотчёт сохранён: {a.report}.csv  и  {a.report}.md")
    print(f"\nвремя: {time.time() - t0:.1f} с на {len(tasks)} кадр(ов)")


def write_report(base, rows, total, fails, names, bank, t0, nframes):
    """Построчный CSV со всеми объектами и читаемая сводка в Markdown."""
    import csv
    cols = list(rows[0].keys())
    with open(str(base) + ".csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    ok = sum(v[0] for v in total.values())
    n = sum(v[1] for v in total.values())
    per_frame = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["итог"]:
            per_frame[r["сценарий"]][1] += 1
            per_frame[r["сценарий"]][0] += int(r["итог"] == "верно")

    L = ["# Отчёт распознавания щита АСУ ТП", "",
         f"Дата: {time.strftime('%Y-%m-%d %H:%M')}  ",
         f"Эталоны: {len(bank.man['templates'])} образцов, отпечаток компоновки "
         f"`{bank.signature}`  ",
         f"Кадров проверено: {nframes}  ",
         f"**Итог: {ok} / {n} — {100 * ok / max(n, 1):.1f} %**", "",
         "## По классам приборов", "",
         "| класс | верно | всего | доля |", "|---|---:|---:|---:|"]
    for k in sorted(total):
        c, t = total[k]
        L.append(f"| {k} | {c} | {t} | {100 * c / max(t, 1):.1f} % |")
    L += ["", "## По кадрам", "", "| сценарий | верно | всего | доля |", "|---|---:|---:|---:|"]
    for nm in names:
        c, t = per_frame.get(nm, [0, 0])
        L.append(f"| {nm} | {c} | {t} | {100 * c / max(t, 1):.1f} % |")

    if fails:
        L += ["", "## Расхождения", "",
              "| сценарий | объект | распознано | истина | отрыв | достоверно |",
              "|---|---|---:|---:|---:|---|"]
        for sc, cls, oid, exp, got, mg, rel in fails:
            L.append(f"| {sc} | {oid} | {fmt(got)} | {fmt(exp)} | "
                     f"{'—' if mg is None else f'{mg:.3f}'} | "
                     f"{'—' if rel is None else ('да' if rel else 'нет')} |")

    num = [r for r in rows if isinstance(r["ошибка"], (int, float))]
    if num:
        L += ["", "## Отклонения числовых показаний", "",
              "| класс | среднее |отклонение| | максимум | объект максимума |",
              "|---|---:|---:|---|"]
        by_cls = defaultdict(list)
        for r in num:
            by_cls[r["класс"]].append((abs(r["ошибка"]), r["объект"], r["сценарий"]))
        for k in sorted(by_cls):
            v = by_cls[k]
            mx = max(v)
            L.append(f"| {k} | {sum(z[0] for z in v) / len(v):.3f} | {mx[0]:.3f} | "
                     f"{mx[1]} ({mx[2]}) |")
    L += ["", "Полная построчная выгрузка — в файле рядом с расширением `.csv` "
              "(разделитель «;», кодировка UTF-8 с BOM, открывается Excel без плясок)."]
    Path(str(base) + ".md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
