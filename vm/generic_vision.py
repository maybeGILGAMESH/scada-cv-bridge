# -*- coding: utf-8 -*-
"""Распознавание щита по профилю, снятому со страницы (dom_profile.py).

Тот же контракт, что у Vision в cvdash.py (snapshot / frame / detail / exact /
press), поэтому дашборд, плейбуки и самовосстановление работают с ним без
изменений. Читаются только пиксели; истина страницы — для оценки.

  * привязка — шапка щита и заголовки панелей (matchTemplate), по ним же видно,
    какая панель закрыта;
  * числа и короткие надписи — глифы этого щита (glyph_ocr);
  * значки состояния и плашка в шапке — сравнение с эталонами фраз;
  * кнопки — эталоны состояний (включена / выключена / недоступна);
  * уровень табло («норма / внимание / авария») — цвет рамки;
  * журнал — Tesseract (русский + английский).
"""

import difflib
import hashlib
import json
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from glyph_ocr import GlyphBank, line_ink, dist
from dom_profile import HARVEST_JS

clock = time.perf_counter
TITLES = {"value": "Значения", "badge": "Значки состояния", "pill": "Плашка состояния", "button": "Кнопки",
          "level": "Уровни табло", "log": "Журнал", "trend": "Тренды (только место)"}
HOMO = str.maketrans("АВЕКМНОРСТХаеорсух", "ABEKMHOPCTXaeopcyx")


NUMS = re.compile(r"\d+(?:[.,]\d+)?")


def same_value(read, truth):
    """Значение сверяется по числам: «%», «м³», «·» шрифт рисует так, что глифов
    для них не набирается, а смысл несёт число."""
    if truth and ":" not in truth:
        read = (read or "").replace(":", ".")          # «:» и «.» в этом шрифте путаются
    a, b = NUMS.findall((read or "").replace(" ", "")), NUMS.findall((truth or "").replace(" ", ""))
    return bool(b) and [x.replace(",", ".") for x in a] == b if b else norm(read) == norm(truth)


def similar(a, b, thr=0.92):
    import difflib
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= thr


def norm(t):
    t = (t or "").translate(HOMO).lower()
    return re.sub(r"[\s·•.,:;\-–—]+", "", t)


def trim(v, thr=0.35):
    if v is None:
        return None
    m = v > thr
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return v[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def png64(img, scale=1.0, interp=cv2.INTER_NEAREST):
    import base64
    if img is None or img.size == 0:
        return None
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


class GenericVision(threading.Thread):
    def __init__(self, system, src, cdp, ctl, tess, prof_dir, hist_s=180):
        super().__init__(daemon=True)
        self.system, self.src, self.cdp, self.ctl, self.tess = system, src, cdp, ctl, tess
        self.dir = Path(prof_dir)
        self.load()
        self.lock = threading.Lock()
        self.state = {"ready": False, "kind": "vision"}
        self.frames = {}
        self.hist = deque(maxlen=hist_s * 12)
        self.roll = deque(maxlen=60)
        self.persist = defaultdict(lambda: deque(maxlen=30))
        self.errors, self.events = deque(maxlen=40), deque(maxlen=40)
        self.seq, self.t_fps = 0, deque(maxlen=30)
        self.active, self.t_last = False, 0.0
        self.truth_down = False
        self.card_ema = {}
        self.offset = (0, 0)
        self.force_reg = False
        self.cache_off = set()
        self.paused_vision = False
        self.setup_done = True
        self._tmemo = {}
        self.kw_ttl = 1.0            # не чаще раза в N с на объект (самопроверка ставит 0)

    # ── профиль ─────────────────────────────────────────────────────
    def load(self):
        p = json.loads((self.dir / "profile.json").read_text(encoding="utf-8"))
        self.prof = p
        self.objects = p["objects"]
        self.inventory = {o["id"] for o in self.objects}
        self.banks = [GlyphBank(self.dir / b["file"], min_gap=b["gap"], thr=b["thr"])
                      for b in p.get("banks") or [{"file": "glyphs.npz", "gap": 1, "thr": 0.35}]
                      if (self.dir / b["file"]).exists()]
        self.glyphs = self.banks[0]
        self._own = {}
        z = np.load(self.dir / "phrases.npz", allow_pickle=True)
        self.phrases = z["phrases"].item()          # текст → цветные эталоны (цвет — тоже состояние)
        self.btn_tpl = defaultdict(dict)
        for k, vs in z["buttons"].item().items():
            oid, st = k.rsplit("|", 1)
            self.btn_tpl[oid][st] = vs
        ref = cv2.imread(str(self.dir / "reference.png"))
        self.ref = ref
        bx, by, bw, bh = p["brand"]
        self.brand = (bx, by, ref[by:by + bh, bx:bx + min(bw, 420)].copy())
        self.cards = []
        for i, c in enumerate(p["cards"]):
            hx, hy, hw, hh = c["head"]
            self.cards.append({"name": f"card{i}", "title": c["title"], "rect": c["rect"],
                               "tpl": ref[hy:hy + hh, hx:hx + min(hw, 260)].copy(), "at": (hx, hy)})
        self.card_of = {}
        for o in self.objects:
            x, y = o["rect"][0], o["rect"][1]
            for c in self.cards:
                cx, cy, cw, ch = c["rect"]
                if cx <= x <= cx + cw and cy <= y <= cy + ch:
                    self.card_of[o["id"]] = c["name"]
                    break

    # ── чтение ──────────────────────────────────────────────────────
    def _gb(self, o):
        """Банк глифов, который самопроверка выбрала для этого объекта: общий
        (один из вариантов) или свой — собранный только по образцам этого поля
        (свой шрифт: «0» с точкой в 10 px не путается с «9» другого шрифта)."""
        own = o.get("own_bank")
        if own and o.get("method") == "own":
            gb = self._own.get(o["id"])
            if gb is None:
                gb = self._own[o["id"]] = GlyphBank(self.dir / own["file"], min_gap=own["gap"], thr=own["thr"])
            return gb
        i = o.get("bank", 0)
        return self.banks[i] if i < len(self.banks) else self.glyphs

    def _align(self, img):
        bx, by, tpl = self.brand
        x0, y0 = max(0, bx - 40), max(0, by - 40)
        roi = img[y0:by + tpl.shape[0] + 40, x0:bx + tpl.shape[1] + 40]
        r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > 0.6:
            self.offset = (x0 + loc[0] - bx, y0 + loc[1] - by)
        return float(mx)

    def _cards(self, img):
        dx, dy = self.offset
        out, lost = [], []
        for c in self.cards:
            hx, hy = c["at"]
            tpl = c["tpl"]
            x0, y0 = max(0, hx + dx - 12), max(0, hy + dy - 12)
            roi = img[y0:hy + dy + tpl.shape[0] + 12, x0:hx + dx + tpl.shape[1] + 12]
            sc = -1.0
            if roi.shape[0] >= tpl.shape[0] and roi.shape[1] >= tpl.shape[1]:
                sc = float(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED).max())
            ema = self.card_ema.get(c["name"])
            if ema is None:
                self.card_ema[c["name"]] = sc
            elif sc < max(0.3, 0.5 * ema):
                lost.append(c["name"])
            else:
                self.card_ema[c["name"]] = 0.95 * ema + 0.05 * sc
            x, y, w, h = c["rect"]
            out.append({"name": c["name"], "title": c["title"], "rect": [x + dx, y + dy, w, h], "dx": dx, "dy": dy,
                        "score": round(sc, 3)})
        return out, lost

    def _rect(self, r):
        return [r[0] + self.offset[0], r[1] + self.offset[1], r[2], r[3]]

    @staticmethod
    def _crop(img, r):
        x, y, w, h = r
        return img[max(0, y):y + h, max(0, x):x + w]

    def read_value(self, img, o):
        m = o.get("method") or "glyph"
        if m.startswith("tess"):
            r = self._read_tess(img, o, int(m[4:]))
            if r.get("read") and not r.get("unsure") and not o.get("floating"):
                # мелкий шрифт: Tesseract путает 0 и 9, глифы — 0 и 8; если способы
                # расходятся в числах — «не уверено», а не уверенная ошибка
                c = self._crop(img, self._rect(o["read_rect"]))
                gt, ginfo, _, _ = self._gb(o).read(c)
                a = [x.replace(",", ".") for x in NUMS.findall(r["read"])]
                b = [x.replace(",", ".") for x in NUMS.findall((gt or "").replace(":", "."))]
                if b and a != b:
                    r["unsure"], r["glyph"] = True, gt.strip()
            return r
        if o.get("floating"):
            return self._read_floating(img, o)
        c = self._crop(img, self._rect(o["read_rect"]))
        text, info, segs, _ = self._gb(o).read(c)
        # символ, у которого лучший глиф почти не отличим от второго, — чтение не уверено
        unsure = any(i[2] < 0.012 and i[0].isdigit() for i in info)
        return {"read": text.strip() or None, "score": round(min([i[1] for i in info] or [0]), 3), "chars": info,
                "unsure": unsure, "margin": round(min([i[2] for i in info] or [0]), 4),
                "method": "свой банк поля" if m == "own" else "glyph"}

    NUM_RX = r"\d+(?:[.,]\d+)?"

    def _read_tess(self, img, o, scale):
        """Мелкий текст (SVG 10 px, значки 15 px), для которого банк глифов не
        набирает образцов: Tesseract по увеличенной области. Числа берутся по
        шаблону объекта: «м3» — это «м³», а не число; лишнее число слева («0» вместо
        «Q») отбрасывается. Если чисел меньше, чем в шаблоне, — «не уверено».
        Распознанное запоминается по двоичной картинке надписи: Tesseract
        вызывается только когда текст на экране изменился."""
        if o.get("floating"):
            boxes = self._read_floating(img, o).get("cands") or []
        else:
            boxes = [self._rect(o["rect"])]
        last = {"read": None, "score": 0.0, "unsure": True, "method": f"tesseract ×{scale}"}
        for box in boxes:
            r = self._tess_box(img, o, box, scale)
            if not r["unsure"]:
                return r
            last = r if last["read"] is None else last
        if o.get("floating"):
            r = self._tess_keyword(img, o)
            if r:
                return r
        return last

    def _tess_keyword(self, img, o):
        """Плавающая подпись слиплась с соседями («14:08 · сут 46», «Ндин» с линией
        уровня): вся область её положений читается Tesseract построчно, число
        берётся сразу за ключевым словом шаблона. Значения такие меняются медленно —
        не чаще раза в 3 с на объект."""
        alt = (o.get("pattern") or "").split("|")[0]
        pieces = alt.split(self.NUM_RX)
        lit = [re.sub(r"\\(.)", r"\1", p_) for p_ in pieces]
        key_w = lit[0].translate(HOMO).lower()
        if len(pieces) != 2 or len(key_w) < 2:
            return None
        memo = self._tmemo.get(("kw", o["id"]))
        if memo and time.time() - memo[0] < self.kw_ttl:
            return memo[1]
        x, y, w, h = self._rect(o["read_rect"])
        from extra_reader import ink
        txt = self.tess(ink(self._crop(img, [x - 3, y - 3, w + 6, h + 6]), 3), "rus", 11)
        s = txt.translate(HOMO).lower()
        m = re.search(re.escape(key_w) + r"\s*(\d+(?:[.,]\d+)?)", s)
        r = None
        if m:
            r = {"read": lit[0] + m.group(1).replace(",", ".") + lit[1], "score": 1.0, "unsure": False,
                 "method": "tesseract по ключевому слову", "raw": txt.strip()}
        self._tmemo[("kw", o["id"])] = (time.time(), r)
        return r

    def _tess_box(self, img, o, box, scale):
        c = self._crop(img, [box[0] - 3, box[1] - 2, box[2] + 6, box[3] + 4])
        if c.size == 0:
            return {"read": None, "score": 0.0, "unsure": True}
        g = c.max(axis=2)
        key = (o["id"], scale, hashlib.md5(np.packbits(g > np.percentile(g, 30) + 45).tobytes() + str(g.shape[:2]).encode()).hexdigest())
        txt = self._tmemo.get(key)
        if txt is None:
            from extra_reader import ink
            txt = self.tess(ink(c, scale), "rus", 7).strip()
            if len(self._tmemo) > 3000:
                self._tmemo.clear()
            self._tmemo[key] = txt
        alt = (o.get("pattern") or "").split("|")[0]
        pieces = alt.split(self.NUM_RX)
        n = len(pieces) - 1
        clean = re.sub(r"(?<=[^\W\d_])[\d³²]", "", txt)          # «м3», «м?» — единица, не число
        nums = [x.replace(",", ".") for x in NUMS.findall(clean)]
        if n and len(nums) > n:
            nums = nums[-n:]
        lit = [re.sub(r"\\(.)", r"\1", p_) for p_ in pieces]
        # буквенная часть должна совпасть с шаблоном («сут», «Ндин»): иначе это соседняя надпись
        letters = lambda s_: re.sub(r"[^а-яёa-z]", "", s_.lower().translate(HOMO))
        want, got = letters("".join(lit)), letters(clean)
        if want and (not got or difflib.SequenceMatcher(None, want, got).ratio() < 0.5):
            n = -1
        if not n or len(nums) != n:
            return {"read": txt or None, "score": 0.0, "unsure": True, "method": f"tesseract ×{scale}", "raw": txt, "found": box}
        read = lit[0] + "".join(v + lit[i + 1] for i, v in enumerate(nums))
        return {"read": read, "score": 1.0, "unsure": False, "method": f"tesseract ×{scale}", "raw": txt, "found": box}

    def _read_floating(self, img, o):
        """Текст, чьё место плавает: в области всех его положений — группы надписей
        (слитые по горизонтали), каждая читается глифами; берётся та, что
        подходит под шаблон текста этого объекта."""
        x, y, w, h = self._rect(o["read_rect"])
        area = self._crop(img, [x - 2, y - 2, w + 4, h + 4])
        g = area.max(axis=2).astype(np.float32)
        bg = np.percentile(g, 30)
        m = (g > bg + 45).astype(np.uint8)
        m = cv2.dilate(m, np.ones((max(3, o.get("h", 12) // 3), 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        rx = re.compile(r"^(?:" + o["pattern"] + r")$") if o.get("pattern") else None
        cands = []
        for i in range(1, n):
            bx, by, bw, bh, a = st[i]
            if bh < 0.6 * o.get("h", 10) or bh > 2.2 * o.get("h", 10):
                continue
            txt, info, _, _ = self._gb(o).read(area[by:by + bh, bx:bx + bw])
            txt = txt.strip()
            if not txt:
                continue
            sc = min([q[1] for q in info] or [0])
            fit = rx is not None and rx.match(txt.replace(" ", "")) is not None or (rx and rx.match(txt))
            key = (bool(fit), -abs(bw - o.get("w", bw)), sc)
            cands.append((key, txt, sc, [x - 2 + bx, y - 2 + by, bw, bh]))
        cands.sort(key=lambda c_: c_[0], reverse=True)
        if not cands:
            return {"read": None, "score": 0.0, "cands": []}
        best = cands[0]
        # ни одна надпись не подошла под шаблон — это не прочтение, а догадка
        return {"read": best[1], "score": round(best[2], 3), "found": best[3], "fits": best[0][0],
                "unsure": not best[0][0], "cands": [c_[3] for c_ in cands[:5]]}

    def _phrase_anchored(self, img, o):
        """Элемент с неподвижным краем (плашка прижата вправо): каждый эталон — на своём
        месте: от края на свою ширину. Сравнивается только надпись (нормированная
        яркость), а не цветной фон — по фону все фразы одинаковы."""
        x, y, w, h = self._rect(o["rect"])
        right, left = x + w, x
        best = []
        for text, vs in self.phrases.items():
            for t in vs:
                th, tw = t.shape[:2]
                if abs(th - h) > 3:
                    continue
                x0 = right - tw if o["anchor"] == "right" else left
                reg = self._crop(img, [x0, y, tw, th])
                if reg.shape[:2] != t.shape[:2]:
                    continue
                a_, _ = line_ink(reg)
                b_, _ = line_ink(t)
                if a_ is None or b_ is None:
                    continue
                # край области: у верного эталона там край плашки, у короткого — середина плашки
                edge = float(np.abs(reg[:, :3].astype(np.float32) - t[:, :3].astype(np.float32)).mean()) / 255 \
                    if o["anchor"] == "right" else \
                    float(np.abs(reg[:, -3:].astype(np.float32) - t[:, -3:].astype(np.float32)).mean()) / 255
                best.append((dist(a_, b_) + 0.5 * edge, text))
        if not best:
            return None
        best.sort()
        d0, text = best[0]
        second = next((d for d, t2 in best if t2 != text), 1.0)
        return {"read": text, "score": round(1 - d0, 3), "margin": round(second - d0, 3), "method": "phrase",
                "cands": [(t2, round(1 - d, 3)) for d, t2 in best[:4]]}

    def read_phrase(self, img, o):
        if o.get("anchor"):
            r = self._phrase_anchored(img, o)
            if r and r["score"] >= 0.85:
                return r
        x, y, w, h = self._rect(o["read_rect"])
        c = self._crop(img, [x - 4, y - 4, w + 8, h + 8])
        real_w = None
        best = []
        for text, vs in self.phrases.items():
            sc = -1.0
            for t in vs:
                if real_w is not None and abs(t.shape[1] - real_w) > 6:      # «Остановлен» внутри «Ожидание АПВ» — нет
                    continue
                if t.shape[0] <= c.shape[0] and t.shape[1] <= c.shape[1]:
                    sc = max(sc, float(cv2.matchTemplate(c, t, cv2.TM_CCOEFF_NORMED).max()))
            if sc > -1:
                best.append((sc, text))
        best.sort(reverse=True)
        cands = [(t, round(sc, 3)) for sc, t in best[:4]]
        if best and best[0][0] >= 0.9:
            text = best[0][1]
            margin = best[0][0] - (best[1][0] if len(best) > 1 else 0)
            if re.search(r"\d", text):
                # фраза с числом («ЧРП 50.0 Гц»): похожий эталон с другой цифрой даёт тот же балл —
                # число перечитываем глифами
                g, info, _, _ = self._gb(o).read(self._crop(img, [x, y, w, h]))
                if g and re.sub(r"[\d.\s]", "", g) == re.sub(r"[\d.\s]", "", text):
                    return {"read": g.strip(), "score": round(best[0][0], 3), "margin": round(margin, 3),
                            "method": "phrase+glyph", "cands": cands}
            return {"read": text, "score": round(best[0][0], 3), "margin": round(margin, 3), "method": "phrase", "cands": cands}
        g, info, _, _ = self._gb(o).read(self._crop(img, [x, y, w, h]))
        if g and min([i[1] for i in info] or [0]) > 0.8:
            return {"read": g.strip(), "score": 0.8, "method": "glyph", "cands": cands}
        from extra_reader import ink
        txt = self.tess(ink(self._crop(img, [x, y, w, h]), 3), "rus+eng", 7).strip()
        return {"read": txt or None, "score": 0.5, "method": "tesseract", "cands": cands}

    def _width(self, img, o):
        """Ширина плашки/значка в кадре — от неподвижного края по строке над текстом,
        пока цвет совпадает с цветом самой плашки (там только её фон)."""
        if not o.get("anchor"):
            return None
        x, y, w, h = self._rect(o["rect"])
        ry = y + 3
        if ry >= img.shape[0]:
            return None
        row = img[ry].astype(np.int32)
        if o["anchor"] == "right":
            xr = x + w - 4
            col = row[xr]
            xx = xr
            while xx > 0 and np.abs(row[xx - 1] - col).sum() < 60:
                xx -= 1
            return int(xr - xx + 8)
        xl = x + 4
        col = row[xl]
        xx = xl
        while xx < len(row) - 1 and np.abs(row[xx + 1] - col).sum() < 60:
            xx += 1
        return int(xx - xl + 8)

    def read_button(self, img, o):
        c = self._crop(img, self._rect(o["rect"])).astype(np.float32)
        sc = []
        for st, vs in self.btn_tpl.get(o["id"], {}).items():
            for t in vs:
                if t.shape == c.shape:
                    sc.append((float(np.abs(c - t).mean()), st))
        if not sc:
            return {"read": None, "score": 0.0}
        sc.sort()
        second = next((d for d, s in sc if s != sc[0][1]), 255.0)
        return {"read": sc[0][1], "score": round(1 - sc[0][0] / 255, 3), "margin": round((second - sc[0][0]) / 255, 4),
                "cands": [(s, round(1 - d / 255, 3)) for d, s in sc[:4]]}

    def read_level(self, img, o):
        from dom_profile import ring_color
        col = ring_color(img, self._rect(o["tile"]))
        if col is None:
            return None
        d = sorted((float(np.linalg.norm(np.array(col) - np.array(c))), k) for k, c in self.prof["levels"].items())
        return {"read": d[0][1], "color": col, "dists": d}

    def read_log(self, img, o):
        """Время — глифами (ноль с точкой Tesseract путает), источник и сообщение — Tesseract."""
        from extra_reader import ink
        c = self._crop(img, self._rect(o["read_rect"]))
        if c.size == 0 or float(c.max()) < 60:
            return {"read": None}
        parts = o.get("parts") or []
        if len(parts) >= 2:
            t = self._rect(parts[0])
            tm = self._gb(o).read(self._crop(img, [t[0] - 1, t[1] - 1, t[2] + 2, t[3] + 2]), allowed=set("0123456789:"))[0]
            r0 = self._rect(o["read_rect"])
            x1 = self._rect(parts[1])[0]
            rest = self._crop(img, [x1 - 2, r0[1], r0[0] + r0[2] - x1 + 2, r0[3]])
            txt = self.tess(ink(rest, 3), "rus", 7).strip()          # сообщения — русские; смесь моделей уходит в латиницу
            return {"read": f"{tm.replace(' ', '')} {txt}".strip() or None, "method": "glyph+tesseract"}
        return {"read": self.tess(ink(c, 3), "rus+eng", 7).strip() or None, "method": "tesseract"}

    def _truth(self):
        try:
            h = self.cdp.evaluate(HARVEST_JS)
            self.truth_down = False
            return {o["id"]: o for o in h["objects"]}, h.get("model") or {}
        except Exception:
            self.truth_down = True
            return {}, {}

    def run(self):
        while True:
            try:
                self.step()
            except Exception as e:
                with self.lock:
                    self.state["error"] = f"{type(e).__name__}: {e}"
                time.sleep(1)

    def step(self):
        if self.paused_vision:
            time.sleep(0.2)
            return
        if not self.active and time.time() > getattr(self, "kvm_until", 0):   # фоновая система (и без открытого пульта) — около 2 к/с
            w = self.t_last + 0.5 - time.time()
            if w > 0:
                time.sleep(w)
        self.t_last = time.time()
        ta, model = self._truth()
        t_a = clock()
        img, t_enc, t_recv = self.src.next_after(t_a - 0.005)
        age = clock() - t_enc if img is not None else None
        if img is None or age > 5:
            with self.lock:
                self.state["health"] = {"stream_dead": True, "stream_age_s": None if age is None else round(age)}
            time.sleep(0.5)
            return
        # модель идёт ускоренно (×30: модельная минута — 2 с): кадр снят между двумя
        # снимками истины, совпадение с любым из них засчитывается
        try:
            tb, _ = self._truth()
        except Exception:
            tb = {}
        t0 = clock()
        q = self._align(img)
        blocks, lost = self._cards(img)
        scene = "absent" if (q < 0.5 or len(lost) >= max(3, int(0.7 * len(self.cards)))) else ("obstructed" if lost else "ok")
        objs, stat, cls_ms = [], defaultdict(lambda: [0, 0]), defaultdict(float)
        for o in self.objects:
            t1 = clock()
            typ = o["type"]
            r = {}
            if typ == "value":
                r = self.read_value(img, o)
            elif typ in ("badge", "pill"):
                r = self.read_phrase(img, o)
            elif typ == "button":
                r = self.read_button(img, o)
            elif typ == "log":
                r = self.read_log(img, o)
            g = ta.get(o["id"]) or {}
            truth = g.get("state") if typ == "button" else g.get("text")
            read = r.get("read")
            ok = None

            def match(tr):
                if typ == "button":
                    return read == tr
                if typ == "value":
                    return read is not None and same_value(read, tr)
                if typ == "log":
                    return read is not None and similar(read, tr)        # свободный текст — по похожести
                return read is not None and norm(read) == norm(tr)
            if g and typ != "trend":
                ok = match(truth)
                gb_ = (tb or {}).get(o["id"]) or {}
                if not ok and gb_:
                    t2 = gb_.get("state") if typ == "button" else gb_.get("text")
                    if t2 != truth and match(t2):
                        ok, truth = True, t2
            hidden = scene != "ok" and self.card_of.get(o["id"]) in lost
            unreliable = (o.get("calib_acc") is not None and o["calib_acc"] < 0.8) or bool(r.get("unsure"))
            if hidden or unreliable:
                ok = None
            rr = self._rect(o["rect"])
            ob = {"id": o["id"], "cls": typ, "label": o["label"], "at": rr[:2], "size": rr[2:], "read": read,
                  "truth": truth, "ok": ok, "score": r.get("score"), "margin": r.get("margin"),
                  "method": r.get("method") or ("glyph" if typ == "value" else "template" if typ == "button" else None),
                  "cached": False, "ms": round((clock() - t1) * 1000, 1), "block": self.card_of.get(o["id"]),
                  "hidden": hidden, "value_box": self._rect(o["read_rect"]) if typ == "value" else None,
                  "reliable": None if not unreliable else False, "calib_acc": o.get("calib_acc")}
            if ok is not None:
                stat[typ][1] += 1
                stat[typ][0] += int(ok)
            cls_ms[typ] += ob["ms"]
            objs.append(ob)
            # уровень табло — отдельный объект
            if typ == "value" and o.get("tile"):
                lv = self.read_level(img, o)
                tl = g.get("level")
                okl = None if (not g or hidden) else (lv["read"] == tl)
                objs.append({"id": "lvl:" + o["label"], "cls": "level", "label": "уровень · " + o["label"],
                             "at": self._rect(o["tile"])[:2], "size": o["tile"][2:], "read": lv["read"], "truth": tl,
                             "ok": okl, "score": None, "method": "color", "cached": False, "ms": 0, "hidden": hidden,
                             "block": self.card_of.get(o["id"]), "quiet": True})
                if okl is not None:
                    stat["level"][1] += 1
                    stat["level"][0] += int(okl)
        t_read = clock() - t0
        persistent = []
        for ob in objs:
            if ob["ok"] is not None:
                self.persist[ob["id"]].append(ob["ok"])
            pq = self.persist.get(ob["id"])
            if pq and len(pq) >= 10 and pq.count(False) >= 0.6 * len(pq) and ob["ok"] is False:
                persistent.append({"id": ob["id"], "cls": ob["cls"], "label": ob["label"], "read": ob["read"], "truth": ob["truth"]})
            if ob["ok"] is False and ob["cls"] != "log":
                if not any(e["id"] == ob["id"] and e["read"] == ob["read"] for e in list(self.errors)[:8]):
                    self.errors.appendleft({"t": time.strftime("%H:%M:%S"), "id": ob["id"], "cls": ob["cls"],
                                            "label": ob["label"], "read": ob["read"], "truth": ob["truth"]})
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
        lost_titles = [c["title"] for c in blocks if c["name"] in lost]
        health = {"scene": scene, "lost_blocks": lost_titles, "quality": round(q, 3), "persistent": persistent,
                  "occluded": [], "truth_down": self.truth_down, "model_paused": False, "fps": round(fps, 2),
                  "active": self.active, "stream_dead": False, "stream_age_s": round(age, 1)}
        timing = {"read_ms": round(t_read * 1000, 1), "registration_ms": 0, "extra_ms": 0,
                  "truth_ms": 0, "age_ms": round((t_done - t_enc) * 1000, 1), "transport_ms": round((t_recv - t_enc) * 1000, 1),
                  "per_class_ms": {c: round(v, 1) for c, v in sorted(cls_ms.items(), key=lambda z: -z[1])}}
        with self.lock:
            self.frames[self.seq] = (jpg.tobytes(), img, {}, None)
            for k in [k for k in self.frames if k < self.seq - 4]:
                del self.frames[k]
            self.hist.append({"t": time.time(), "acc": round(100 * ok_n / max(1, tot_n), 2), "read_ms": timing["read_ms"],
                              "fps": round(fps, 2), "quality": round(q, 3), "age_ms": timing["age_ms"], "reread": len(objs)})
            pill = next((o for o in objs if o["id"] == "pill"), {})
            self.state = {
                "ready": True, "kind": "vision", "reader": "dom", "seq": self.seq, "t": time.time(),
                "frame_size": [img.shape[1], img.shape[0]], "objects": objs, "blocks": blocks,
                "class_titles": TITLES, "trends": {}, "log_lines": [], "log_truth": None,
                "registered": False, "quality": round(q, 3), "read_objects": len(objs), "cached_objects": 0,
                "fps": round(fps, 2), "timing": timing, "frame": {"ok": ok_n, "total": tot_n},
                "rolling": {"ok": sum(v[0] for v in roll.values()), "total": sum(v[1] for v in roll.values()),
                            "frames": len(self.roll),
                            "by_class": {c: {"ok": a, "total": b, "title": TITLES.get(c, c)} for c, (a, b) in roll.items()}},
                "gauge_err": {"mean": None, "p95": None, "hist": []},
                "model": {"layout": None, "layout_name": None,
                          "clock": next((o["read"] for o in objs if o["id"] == "clock"), None),
                          "plant": None, "state": pill.get("read")},
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

    # ── действия ────────────────────────────────────────────────────
    def press(self, obj, times=1):
        o = next((o for o in self.objects if o["id"] == obj), None)
        if o is None:
            raise RuntimeError(f"«{obj}» нет в профиле этого щита")
        x, y, w, h = self._rect(o["rect"])
        for _ in range(int(times)):
            self.ctl.call("click", x=x + w / 2, y=y + h / 2)
            time.sleep(0.35)
        return f"нажато «{o['label']}» ×{times} в точке {int(x + w / 2)}, {int(y + h / 2)}"

    def exact(self):
        self.paused_vision = True
        try:
            self.cdp.evaluate("() => { paused = true; }")
            time.sleep(1.3)
            was = self.active
            self.active = True
            self.paused_vision = False
            n0 = self.seq
            t_end = time.time() + 6
            while self.seq < n0 + 2 and time.time() < t_end:
                time.sleep(0.2)
            st = self.snapshot()
            self.active = was
        finally:
            self.cdp.evaluate("() => { paused = false; }")
        fails = [{"id": o["id"], "cls": o["cls"], "read": o["read"], "truth": o["truth"]}
                 for o in st.get("objects", []) if o.get("ok") is False]
        ok, tot = st["frame"]["ok"], st["frame"]["total"]
        return {"ok": ok, "total": tot, "acc": round(100 * ok / max(1, tot), 2), "layout": "—",
                "clock": st["model"].get("clock"), "fails": fails}

    def detail(self, oid):
        fr = self.frame()
        if fr is None:
            return {"error": "нет кадра"}
        img = fr[1]
        oid2 = oid[4:] if oid.startswith("lvl:") else oid
        o = next((x for x in self.objects if x["id"] == oid2 or x["label"] == oid2), None)
        if o is None:
            return {"error": f"нет объекта {oid}"}
        base = {"id": oid, "class_title": TITLES.get(o["type"], o["type"]), "block": self.card_of.get(o["id"])}
        if oid.startswith("lvl:"):
            lv = self.read_level(img, o)
            sw = lambda c: np.full((28, 60, 3), np.clip(c, 0, 255), np.uint8)
            return dict(base, kind="discrete", images={"crop": png64(self._crop(img, self._rect(o["tile"])), 1.5),
                                                       "diff": png64(sw(lv["color"]), 1)},
                        states=[{"state": k, "sim": round(1 - d / 255, 4), "img": png64(sw(self.prof["levels"][k]), 1)}
                                for d, k in lv["dists"]],
                        result={"state": lv["read"], "margin": round((lv["dists"][1][0] - lv["dists"][0][0]) / 255, 4) if len(lv["dists"]) > 1 else 1,
                                "margin_min": 0.0, "locate_score": 1.0, "at": self._rect(o["tile"])[:2]})
        c = self._crop(img, self._rect(o["read_rect"] if o["type"] != "button" else o["rect"]))
        if o["type"] == "value":
            v, _ = line_ink(c)
            text, segs = self._gb(o).explain(c)
            big = cv2.resize(c, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            for sg in segs:
                a, b = sg["seg"]
                cv2.rectangle(big, (a * 4, 0), (b * 4 - 1, big.shape[0] - 1), (229, 135, 57), 1)
            return dict(base, kind="readout", label=o["label"],
                        images={"crop": png64(c, 3), "value": png64(big, 1), "ink": None},
                        chars=[{"glyph": png64((sg["glyph"] * 255).astype(np.uint8), 5),
                                "cands": [{"char": ch, "score": sc, "img": png64((t * 255).astype(np.uint8), 5)}
                                          for ch, sc, t in sg["cands"]]} for sg in segs],
                        result={"text": text, "box": self._rect(o["read_rect"]), "gray": False})
        if o["type"] in ("badge", "pill", "button"):
            r = self.read_phrase(img, o) if o["type"] != "button" else self.read_button(img, o)
            imgs = {}
            for name, sc in (r.get("cands") or []):
                if o["type"] == "button":
                    t = (self.btn_tpl[o["id"]].get(name) or [None])[0]
                    imgs[name] = png64(t, 1.5) if t is not None else None
                else:
                    t = (self.phrases.get(name) or [None])[0]
                    imgs[name] = png64((t * 255).astype(np.uint8), 2) if t is not None else None
            return dict(base, kind="discrete", images={"crop": png64(c, 2), "diff": None},
                        states=[{"state": n, "sim": s, "img": imgs.get(n)} for n, s in (r.get("cands") or [])],
                        result={"state": r.get("read"), "margin": r.get("margin") or 0, "margin_min": 0.004,
                                "locate_score": r.get("score") or 0, "at": self._rect(o["rect"])[:2]})
        return dict(base, kind="locate", images={"crop": png64(c, 1), "template": None, "heat": None},
                    result={"score": 0, "found": self._rect(o["rect"])[:2], "expected": o["rect"][:2]})
