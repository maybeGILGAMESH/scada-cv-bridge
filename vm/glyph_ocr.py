# -*- coding: utf-8 -*-
"""Чтение моноширинных цифр по эталонам глифов.

Числа на щите (табло, часы, подписи осей трендов, время в журнале) написаны
моноширинным шрифтом. В ВМ это DejaVu Sans Mono, у которого ноль с точкой
внутри, и Tesseract путает его с 8, 9 и 6. Здесь цифры читаются тем же
способом, что семисегментные индикаторы в проекте: строка режется на символы
по пустым столбцам, каждый символ сравнивается со всеми глифами своего шрифта.

Глифы нарезаются автоматически из контрольных кадров банка эталонов: для
каждого кадра известен истинный текст табло, часов и подписей осей. Если
число найденных символов совпадает с длиной текста — символы подписываются.
Шрифт группы — по высоте строки: глифы одного размера взаимозаменяемы, цвет
не важен (сравнивается нормированная яркость).

    python glyph_ocr.py          # собрать vm/cv_templates_vm/glyphs.npz
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE / "cv_templates_vm"


def line_ink(crop):
    """Нормированная яркость строки 0..1 (фон 0, текст 1) и маска текста."""
    g = crop.max(axis=2).astype(np.float32) if crop.ndim == 3 else crop.astype(np.float32)
    bg, fg = np.percentile(g, 12), np.percentile(g, 99.5)
    if fg - bg < 25:
        return None, None
    v = np.clip((g - bg) / (fg - bg), 0, 1)
    return v, v > 0.35


def segments(mask, min_gap=1):
    """Символы — столбцы с чернилами, разделённые пустыми столбцами."""
    cols = mask.any(axis=0)
    out, x0 = [], None
    for x, c in enumerate(cols):
        if c and x0 is None:
            x0 = x
        elif not c and x0 is not None:
            out.append([x0, x])
            x0 = None
    if x0 is not None:
        out.append([x0, len(cols)])
    # слить разрывы уже min_gap (антиалиасинг рвёт тонкие штрихи)
    merged = []
    for s in out:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def cut(crop, pitch=None):
    """Строка → (символы как массивы 0..1, их x, ширины, высота строки)."""
    v, m = line_ink(crop)
    if v is None:
        return None
    # строка — самая «чернильная» полоса подряд идущих строк: случайные точки
    # соседних элементов над и под текстом в неё не попадают
    r = m.sum(axis=1)
    bands, cur = [], None
    for y, n in enumerate(r):
        if n and cur is None:
            cur = [y, y + 1, 0]
        if n:
            cur[1], cur[2] = y + 1, cur[2] + n
        elif cur is not None and y - cur[1] >= 1:
            bands.append(cur)
            cur = None
    if cur is not None:
        bands.append(cur)
    if not bands:
        return None
    y0, y1, _ = max(bands, key=lambda b: b[2])
    v, m = v[y0:y1], m[y0:y1]
    segs = split_wide(segments(m), pitch)
    return [v[:, a:b] for a, b in segs], segs, y1 - y0, (y0, y1)


def split_wide(segs, pitch=None):
    """Моноширинный шрифт: слипшиеся символы («10») дают сегмент шириной в
    несколько шагов — режем его поровну."""
    ws = [b - a for a, b in segs]
    if not ws:
        return segs
    if pitch is None:
        big = [w for w in ws if w >= 0.5 * max(ws)]
        pitch = float(np.median(big))
    out = []
    for (a, b), w in zip(segs, ws):
        n = int(round(w / pitch)) if w > 1.4 * pitch else 1
        for k in range(n):
            out.append([a + round(k * w / n), a + round((k + 1) * w / n)])
    return out


def key_of(h):
    return int(h)


def dist(g, t):
    """Средняя разность двух глифов при лучшем сдвиге ±1 px: на мелком шрифте
    дробная позиция текста и JPEG сдвигают глиф на пиксель."""
    H, W = max(g.shape[0], t.shape[0]) + 2, max(g.shape[1], t.shape[1]) + 2
    T = np.zeros((H, W), np.float32)
    T[1:1 + t.shape[0], 1:1 + t.shape[1]] = t
    best = 9.0
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            G = np.zeros((H, W), np.float32)
            h, w = min(g.shape[0], H - dy), min(g.shape[1], W - dx)
            G[dy:dy + h, dx:dx + w] = g[:h, :w]
            best = min(best, float(np.abs(G - T).mean()))
    return best


def fit(a, H, W):
    out = np.zeros((H, W), np.float32)
    h, w = min(H, a.shape[0]), min(W, a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


class GlyphBank:
    def __init__(self, path=BANK / "glyphs.npz"):
        self.sets = {}            # высота строки → [(символ, глиф)]
        self.path = Path(path)
        self._stacks, self._memo = {}, {}
        if self.path.exists():
            d = np.load(self.path, allow_pickle=True)
            self.sets = d["sets"].item()

    def save(self):
        np.savez_compressed(self.path, sets=np.array(self.sets, dtype=object))

    def _stack(self, cands):
        """Кандидаты одним массивом (N, H, W), глиф — со сдвигом (1, 1):
        сравнение со всеми и при 9 сдвигах — одна операция numpy."""
        key = id(cands), len(cands)
        st = self._stacks.get(key)
        if st is None:
            H = max(t.shape[0] for _, t in cands) + 4
            W = max(t.shape[1] for _, t in cands) + 4
            arr = np.zeros((len(cands), H, W), np.float32)
            for i, (_, t) in enumerate(cands):
                arr[i, 1:1 + t.shape[0], 1:1 + t.shape[1]] = t
            st = self._stacks[key] = (arr, [c for c, _ in cands])
        return st

    def scores(self, g, cands):
        """1 − средняя разность при лучшем сдвиге ±1 px, по всем кандидатам сразу."""
        arr, chars = self._stack(cands)
        _, H, W = arr.shape
        G = np.zeros((9, H, W), np.float32)
        k = 0
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                h, w = min(g.shape[0], H - dy), min(g.shape[1], W - dx)
                G[k, dy:dy + h, dx:dx + w] = g[:h, :w]
                k += 1
        d = np.abs(G[:, None] - arr[None]).mean(axis=(2, 3)).min(axis=0)
        return 1.0 - d * (H * W) / max(1, (g.shape[0] + 2) * (g.shape[1] + 2)), chars

    def _candidates(self, h):
        for dh in (0, -1, 1, -2, 2):
            if key_of(h + dh) in self.sets:
                return self.sets[key_of(h + dh)]
        return []

    def explain(self, crop, allowed=None, k=4):
        """Для дашборда: каждый символ и лучшие кандидаты-глифы с баллами."""
        text, info, segs, yr = self.read(crop, allowed)
        c = cut(crop)
        if c is None or not segs:
            return text, []
        cands = self._candidates(c[2])
        if allowed:
            cands = [z for z in cands if z[0] in allowed]
        y0, y1 = yr
        v, _ = line_ink(crop)
        out = []
        for a, b in segs:
            g = v[y0:y1, a:b]
            sc = {}
            vals, chars = self.scores(g, cands)
            for (ch, t), d in zip(cands, vals):
                if d > sc.get(ch, (-1, None))[0]:
                    sc[ch] = (float(d), t)
            top = sorted(sc.items(), key=lambda z: -z[1][0])[:k]
            out.append({"seg": [int(a), int(b)], "glyph": g,
                        "cands": [(ch, round(s, 3), t) for ch, (s, t) in top]})
        return text, out

    def read(self, crop, allowed=None):
        """→ (текст, [(символ, балл, отрыв)], сегменты, строка y0..y1)"""
        key = (crop.shape, crop.tobytes(), tuple(sorted(allowed)) if allowed else None)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        res = self._read(crop, allowed)
        if len(self._memo) > 5000:
            self._memo.clear()
        self._memo[key] = res
        return res

    def _read(self, crop, allowed=None):
        c = cut(crop)
        if c is None:
            return "", [], [], None
        cands = self._candidates(c[2])
        digits = [t.shape[1] for ch, t in cands if ch.isdigit()]
        if digits:                       # шаг шрифта — из банка: слипшиеся цифры режутся по нему
            c = cut(crop, float(np.median(digits)))
        glyphs, segs, h, yr = c
        if allowed:
            cands = [z for z in cands if z[0] in allowed]
        if not cands:
            return "", [], segs, yr
        text, info = "", []
        # пробел — разрыв шире полутора средних межсимвольных промежутков
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        sp = (np.median(gaps) * 2.2 + 1) if gaps else 99
        for i, g in enumerate(glyphs):
            if i and gaps[i - 1] > sp:
                text += " "
            sc, chars = self.scores(g, cands)
            order = np.argsort(-sc)
            ch = chars[order[0]]
            second = next((sc[j] for j in order if chars[j] != ch), 0.0)
            text += ch
            info.append((ch, round(float(sc[order[0]]), 3), round(float(sc[order[0]] - second), 3)))
        return text, info, segs, yr


# ───────────────────────────────────────────────────────────── сборка банка

def text_crops(img, ann):
    """Из кадра с разметкой — (вырезка строки, истинный текст) для всех
    моноширинных надписей с известным текстом."""
    from extra_reader import value_crop
    out = []
    for o in ann["objects"]:
        if o["type"] == "text_readout" and o.get("text"):
            val, _, _ = value_crop(img, o["bbox"])
            if val is not None:
                out.append((val, o["text"]))
        elif o["id"] == "hdr_clock" and o.get("text"):
            x, y, w, h = [int(v) for v in o["bbox"]]
            out.append((img[y:y + h, x:x + w], o["text"]))
        elif o["type"] == "trend":
            L, T, PW, PH = [int(round(v)) for v in o["plot_bbox"]]
            for i in range(5):
                yy = int(round(T + PH * i / 4))
                lab = f"{o['y_max'] - (o['y_max'] - o['y_min']) * i / 4:.0f}"
                out.append((img[yy - 5:yy + 6, L - 40:L - 3], lab))
    return out


def build(frames, extra=()):
    """frames: [(кадр BGR, разметка)], extra: [(вырезка, текст)] — например,
    время в журнале, снятое с живого щита."""
    acc = defaultdict(lambda: defaultdict(list))
    used = skipped = 0
    samples = list(extra)
    for img, ann in frames:
        samples += text_crops(img, ann)
    for crop, text in samples:
        c = cut(crop)
        if c is None:
            skipped += 1
            continue
        glyphs, segs, h, _ = c
        chars = text.replace(" ", "")
        if len(glyphs) != len(chars):
            skipped += 1
            continue
        for g, ch in zip(glyphs, chars):
            acc[key_of(h)][ch].append(g)
        used += 1
    sets = {}
    for k, d in acc.items():
        lst = []
        for ch, gs in d.items():
            # до 4 вариантов на символ: дробная позиция текста даёт разную растеризацию
            uniq = []
            for g in gs:
                if all(g.shape != u.shape or float(np.abs(g - u).mean()) > 0.04 for u in uniq):
                    uniq.append(g)
                if len(uniq) >= 8:
                    break
            lst += [(ch, u) for u in uniq]
        sets[k] = lst
    return sets, used, skipped


def main():
    sys.path.insert(0, str(HERE))
    idx = json.loads((BANK / "validation" / "index.json").read_text(encoding="utf-8"))
    frames = [(cv2.imread(str(BANK / f["image"])), json.loads((BANK / f["truth"]).read_text(encoding="utf-8")))
              for f in idx["frames"]]
    ref = json.loads((BANK / "reference" / "frame.json").read_text(encoding="utf-8"))
    frames.append((cv2.imread(str(BANK / "reference" / "frame.png")), ref))
    extra = []
    live = BANK / "glyphs_live.json"            # время журнала с живого щита (см. cvdash/regen)
    if live.exists():
        for s in json.loads(live.read_text(encoding="utf-8")):
            extra.append((cv2.imread(str(BANK / s["image"])), s["text"]))
    # те же кадры, пережатые как в MJPEG-потоке (JPEG q80): мелкие серые подписи
    # оси в JPEG выглядят иначе, глифы нужны в обоих видах
    def jpg(img, q=80):
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    frames += [(jpg(img), ann) for img, ann in frames]
    sets, used, skipped = build(frames, extra)
    gb = GlyphBank()
    gb.sets = sets
    gb.save()
    print(f"глифы: {sum(len(v) for v in sets.values())} по {len(sets)} шрифтам "
          f"(высоты строк {sorted(sets)}); надписей использовано {used}, пропущено {skipped}")
    for k in sorted(sets):
        print(f"  строка {k} px: {''.join(sorted(set(c for c, _ in sets[k])))}")

    # проверка на тех же кадрах (оставляем по кадру на компоновку вне сборки было бы честнее —
    # но глифы — это свойство шрифта, а не кадра)
    ok = tot = 0
    bad = []
    for img, ann in frames:
        for crop, text in text_crops(img, ann):
            got = gb.read(crop)[0].replace(" ", "")
            tot += 1
            ok += got == text.replace(" ", "")
            if got != text.replace(" ", "") and len(bad) < 8:
                bad.append((got, text))
    print(f"сверка: {ok}/{tot} надписей прочитаны точно; примеры ошибок: {bad}")


if __name__ == "__main__":
    main()
