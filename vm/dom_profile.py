# -*- coding: utf-8 -*-
"""Профиль распознавания для щита на HTML/SVG (без своей CV-разметки).

Щит neft-hmi-cv сам режет эталоны («Выгрузить эталоны»). Старый щит
neft-scada.html этого не умеет, но он свёрстан элементами страницы: где на
экране табло, значок, кнопка, плашка состояния — знает сама страница. Здесь
это снимается через DevTools в ВМ, щит прогоняется через набор состояний
(пуск, аварии, ожидание АПВ, переключатели, ремонт), и на каждом кадре
(без потерь и JPEG) режутся образцы с истинным текстом:

  * глифы цифр и букв — для чисел (табло, мнемосхема, часы, уставки);
  * фразы — значки состояния оборудования, плашка в шапке;
  * состояния кнопок (включена / выключена / недоступна);
  * цвет рамки табло для уровней «норма / внимание / авария»;
  * опоры — заголовки панелей, по ним кадр привязывается и видно, что закрыто.

Распознавание (generic_vision.py) использует только этот профиль и пиксели.

    python dom_profile.py newsys          # собрать vm/profiles/newsys/
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fleet                      # noqa: E402
from vmhdmi import Cdp            # noqa: E402
from glyph_ocr import build, GlyphBank, line_ink   # noqa: E402

HARVEST_JS = r"""() => {
  const R = e => { const b = e.getBoundingClientRect(); return [Math.round(b.x), Math.round(b.y), Math.round(b.width), Math.round(b.height)]; };
  const out = [];
  document.querySelectorAll('.disp').forEach((d, i) => {
    const v = d.querySelector('.val'), lbl = d.querySelector('.lbl').textContent.trim();
    out.push({id: 'd:' + lbl, type: 'value', label: lbl, rect: R(v), text: v.textContent.trim(),
              num: v.querySelector('span').textContent.trim(), unit: (v.querySelector('small') || {}).textContent || '',
              tile: R(d), level: d.classList.contains('crit') ? 'crit' : d.classList.contains('warn') ? 'warn' : 'normal'});
  });
  document.querySelectorAll('#eqState .kv').forEach(k => {
    const lbl = k.children[0].textContent.trim(), v = k.children[1];
    if (v.classList.contains('badge'))
      out.push({id: 'eq:' + lbl, type: 'badge', label: lbl, rect: R(v), text: v.textContent.trim(),
                state: [...v.classList].find(c => c.startsWith('b-')) || ''});
    else out.push({id: 'eq:' + lbl, type: 'value', label: lbl, rect: R(v), text: v.textContent.trim(), num: v.textContent.trim()});
  });
  const p = document.getElementById('hState');
  out.push({id: 'pill', type: 'pill', label: 'Состояние объекта', rect: R(p), text: p.textContent.trim(),
            state: (p.className.match(/st-\w+/) || [''])[0]});
  const c = document.getElementById('hClock'), dd = document.getElementById('hDays');
  out.push({id: 'clock', type: 'value', label: 'Модельное время', rect: R(c), text: c.textContent.trim(), num: c.textContent.trim()});
  if (dd) out.push({id: 'days', type: 'value', label: 'Сутки модели', rect: R(dd), text: dd.textContent.trim(), num: dd.textContent.trim()});
  [['vFreq', 'Уставка частоты ЧРП'], ['vChoke', 'Штуцер, %'], ['vGasV', 'Клапан затруб. газа, %'], ['vSub', 'Уставка погружения, м']].forEach(([id, l]) => {
    const e = document.getElementById(id); if (e) out.push({id: 'sp:' + id, type: 'value', label: l, rect: R(e), text: e.textContent.trim(), num: e.textContent.trim()}); });
  [['svgLevelTxt', 'Динамический уровень'], ['svgMotorT', 'Температура ПЭД (схема)'], ['svgPwh', 'Р буферное (схема)'],
   ['svgPcas', 'Р затрубное (схема)'], ['svgQ', 'Дебит (схема)'], ['svgPr', 'Р пластовое (схема)'], ['svgChoke', 'Штуцер (схема)']].forEach(([id, l]) => {
    const e = document.getElementById(id); if (e) out.push({id: 'svg:' + id, type: 'value', label: l, rect: R(e), text: e.textContent.trim(), num: e.textContent.trim()}); });
  document.querySelectorAll('button').forEach(b => {
    const t = b.textContent.trim();
    out.push({id: 'btn:' + t.replace(/^⏻\s*/, ''), type: 'button', label: t.replace(/^⏻\s*/, ''), rect: R(b), text: t,
              state: b.disabled ? 'disabled' : b.classList.contains('on') ? 'on' : 'off'});
  });
  [...document.querySelectorAll('#alarms > *')].slice(0, 2).forEach((a, i) => {
    const ch = [...a.children];
    out.push({id: 'log' + i, type: 'log', label: 'Журнал, строка ' + (i + 1), rect: R(a), text: a.textContent.trim(),
              parts: ch.map(x => ({rect: R(x), text: x.textContent.trim()}))});
  });
  document.querySelectorAll('canvas').forEach(cv => out.push({id: 'trend:' + cv.id, type: 'trend', label: 'Тренд ' + cv.id, rect: R(cv), text: null}));
  const cards = [...document.querySelectorAll('.card')].map(cd => {
    const h = cd.querySelector('h2, h3, .top, .ttl, header') || cd.firstElementChild;
    return {title: (h ? h.textContent.trim() : '').slice(0, 40), rect: R(cd), head: h ? R(h) : null};
  });
  const brand = document.querySelector('header .brand') || document.querySelector('header');
  return {objects: out, cards, brand: R(brand), model: {run: S.run, tripped: S.tripped, reason: S.tripReason}};
}"""

# состояния щита для калибровки: JS выполняется, модель немного считается, затем пауза
STATES = [
    ("исходное", "() => { paused = true; }", 0),
    ("пуск, ×600", "() => { S.run = true; S.starting = 1; speed = 600; paused = false; }", 3),
    ("частота 43,5, штуцер 72", "() => { ui.f = 43.5; ui.choke = 72; sync(); }", 2),
    ("переключатели вкл", "() => { tgl('auto'); tgl('inh'); tgl('heat'); }", 1),
    ("переключатели выкл", "() => { tgl('auto'); tgl('inh'); tgl('heat'); tgl('apv'); }", 1),
    ("АПВ вкл, частота 58,5", "() => { tgl('apv'); ui.f = 58.5; ui.choke = 39; sync(); }", 2),
    ("предупреждения", "() => { S.iso = 19; S.vib = 10.6; S.tMot = 112; S.gas = 0.34; }", 1),
    ("авария: изоляция", "() => { S.iso = 3.7; }", 1),
    ("ремонт", "() => { repair(); }", 1),
    ("пуск после ремонта", "() => { cmdStart(); }", 2),
    ("авария: перегрев", "() => { S.tMot = 141; }", 1),
    ("квитировать", "() => { ackAll(); }", 1),
    ("аварийный останов", "() => { cmdStart(); cmdEstop(); }", 1),
    ("ожидание АПВ", "() => { ackAll(); repair(); cmdStart(); S.starting = 1; S.subm = 10; S.h = 2080; }", 1),
    ("ток и вибрация", "() => { restartAt = null; S.restartAt = null; cmdStart(); S.I = 41; S.vib = 15; }", 1),
    ("сброс модели, ×30", "() => { resetAll(); speed = 30; ui.f = 47.5; ui.choke = 86; sync(); }", 1),
    ("пуск ×120", "() => { cmdStart(); speed = 120; }", 3),
]


# «прогон цифр»: в 17 состояниях встречаются не все цифры (засорение всегда около 1.00,
# время журнала — одни нули), и банк глифов о недостающих гадает. Здесь отрисовка щита
# на паузе замораживается, а в каждом числовом поле цифра на позиции j заменяется на
# (k + j) mod 10 — за 10 кадров каждая позиция каждого поля проходит все цифры 0–9.
SWEEP_FREEZE = """() => { if (!window.__rend) { window.__rend = [renderHeader, renderDisplays, renderMnemo, renderEq];
  renderHeader = renderDisplays = renderMnemo = renderEq = () => {}; } }"""
SWEEP_JS = r"""k => {
  const els = [...document.querySelectorAll('.disp .val span, #eqState .kv > :nth-child(2):not(.badge), #hClock, #hDays,'
    + ' #vFreq, #vChoke, #vGasV, #vSub, #svgLevelTxt, #svgMotorT, #svgPwh, #svgPcas, #svgQ, #svgPr, #svgChoke')];
  [...document.querySelectorAll('#alarms > *')].slice(0, 2).forEach(a => a.firstElementChild && els.push(a.firstElementChild));
  for (const e of els) {
    if (!e.dataset.orig) e.dataset.orig = e.textContent;
    const w = document.createTreeWalker(e, NodeFilter.SHOW_TEXT); let n, j = 0;
    while ((n = w.nextNode())) n.textContent = n.textContent.replace(/\d/g, () => String((k + j++) % 10));
  }
  return els.length;
}"""
SWEEP_THAW = """() => { if (window.__rend) { [renderHeader, renderDisplays, renderMnemo, renderEq] = window.__rend; delete window.__rend; }
  document.querySelectorAll('[data-orig]').forEach(e => delete e.dataset.orig); }"""


def text_pattern(t):
    """«Ндин 1403 м» → Ндин\\d+(?:[.,]\\d+)?м : числа — любые, пробелы не важны."""
    out = ""
    for tok in re.split(r"(\d+(?:\.\d+)?)", t):
        if not tok:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            out += r"\d+(?:[.,]\d+)?"
        else:
            out += re.escape(tok.replace(" ", ""))
    return out


def snapshots(s):
    out = {}
    for kind, path in (("png", "snapshot.png"), ("jpg", "snapshot")):
        data = urlopen(f"http://127.0.0.1:{s['ports']['mjpeg']}/{path}", timeout=10).read()
        out[kind] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return out


def union(rects, pad_r=0, pad=1):
    x0 = min(r[0] for r in rects) - pad
    y0 = min(r[1] for r in rects) - pad
    x1 = max(r[0] + r[2] for r in rects) + pad + pad_r
    y1 = max(r[1] + r[3] for r in rects) + pad
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def crop(img, r):
    x, y, w, h = r
    return img[max(0, y):y + h, max(0, x):x + w]


def ring_color(img, r):
    """Средний цвет рамки табло (кольцо 3 px по краю)."""
    c = crop(img, r).astype(np.float32)
    if c.size == 0:
        return None
    m = np.zeros(c.shape[:2], bool)
    m[:3], m[-3:], m[:, :3], m[:, -3:] = True, True, True, True
    return c[m].mean(axis=0).tolist()


def main():
    sid = next((a for a in sys.argv[1:] if not a.startswith("--")), "newsys")
    if "--rebuild" in sys.argv:
        return rebuild(sid)
    s = fleet.system(sid)
    out = HERE / "profiles" / sid
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cdp = Cdp(ssh_port=s["ports"]["ssh"], local_port=s["ports"]["cdp"] + 9)
    frames = []
    try:
        # всплывающие окна поверх щита попали бы в эталоны — закрыть до съёмки
        import socket
        with socket.create_connection(("127.0.0.1", s["ports"]["control"]), timeout=5) as k:
            k.sendall(b'{"op": "key", "key": "Escape"}\n')
            k.recv(1024)
        time.sleep(0.5)
        for i, (name, js, run_s) in enumerate(STATES):
            cdp.evaluate(js)
            if run_s:
                cdp.evaluate("() => { paused = false; }")
                time.sleep(run_s)
            cdp.evaluate("() => { paused = true; }")
            time.sleep(0.9)
            h = cdp.evaluate(HARVEST_JS)
            sn = snapshots(s)
            for kind, img in sn.items():
                cv2.imwrite(str(out / "frames" / f"{i:02d}_{kind}.png"), img)
            frames.append({"name": name, "harvest": h, "files": {k: f"frames/{i:02d}_{k}.png" for k in sn}})
            print(f"  {i + 1:2d}/{len(STATES)} {name}: {h['model']}", flush=True)
        if "--no-sweep" not in sys.argv:
            cdp.evaluate("() => { resetAll(); cmdStart(); speed = 600; paused = false; }")
            time.sleep(2)
            cdp.evaluate("() => { paused = true; }")
            cdp.evaluate(SWEEP_FREEZE)
            for k in range(10):
                i = len(frames)
                n = cdp.evaluate(SWEEP_JS, k)
                time.sleep(0.6)
                h = cdp.evaluate(HARVEST_JS)
                sn = snapshots(s)
                for kind, img in sn.items():
                    cv2.imwrite(str(out / "frames" / f"{i:02d}_{kind}.png"), img)
                frames.append({"name": f"прогон цифр {k}", "harvest": h, "files": {kk: f"frames/{i:02d}_{kk}.png" for kk in sn}})
                print(f"  прогон цифр {k + 1}/10: полей {n}", flush=True)
    finally:
        try:
            cdp.evaluate(SWEEP_THAW)
        except Exception:
            pass
        cdp.evaluate("() => { resetAll(); speed = 30; paused = false; }")
        cdp.close()
    (out / "calibration.json").write_text(json.dumps(frames, ensure_ascii=False), encoding="utf-8")
    build_profile(sid, s, out, frames)


def build_profile(sid, s, out, frames):

    # ── объекты профиля: место чтения — объединение рамок по всем состояниям ─
    by = defaultdict(list)
    for f in frames:
        for o in f["harvest"]["objects"]:
            by[o["id"]].append(o)
    objects = []
    for oid, lst in by.items():
        o0 = lst[0]
        rects = [o["rect"] for o in lst]
        # «Плавающий» объект: место меняется от состояния к состоянию (часы в шапке
        # сдвигает плашка состояния, подпись «Ндин» ездит за уровнем)
        floating = (max(r[0] for r in rects) - min(r[0] for r in rects) > 4 and
                    max(r[0] + r[2] for r in rects) - min(r[0] + r[2] for r in rects) > 4) or \
                   (max(r[1] for r in rects) - min(r[1] for r in rects) > 4)
        texts = sorted({o.get("text") or "" for o in lst})
        pat = None
        if o0["type"] == "value":
            # шаблон текста из истинных значений: цифры → \d+, пробелы — по желанию
            pats = {text_pattern(t) for t in texts if t}
            pat = "|".join(sorted(pats))
        ob = {"id": oid, "type": o0["type"], "label": o0["label"],
              "read_rect": union(rects, pad_r=0),
              "rect": o0["rect"], "tile": o0.get("tile"), "unit": o0.get("unit"),
              "floating": bool(floating), "pattern": pat,
              # какой край элемента неподвижен (плашка в шапке прижата вправо)
              "anchor": "right" if max(r[0] + r[2] for r in rects) - min(r[0] + r[2] for r in rects) <= 2 and len({r[2] for r in rects}) > 1
                        else "left" if max(r[0] for r in rects) - min(r[0] for r in rects) <= 2 and len({r[2] for r in rects}) > 1 else None,
              "h": int(np.median([r[3] for r in rects])), "w": int(np.median([r[2] for r in rects]))}
        if o0["type"] == "log" and o0.get("parts"):
            ob["parts"] = [p["rect"] for p in o0["parts"]]         # время, источник, сообщение
        objects.append(ob)
    ref = frames[0]
    cards = [c for c in ref["harvest"]["cards"] if c["head"] and c["title"]]
    ref_img = cv2.imread(str(out / ref["files"]["png"]))
    cv2.imwrite(str(out / "reference.png"), ref_img)

    samples = []
    phrases = defaultdict(list)          # текст → цветные эталоны (значки, плашка): цвет — тоже состояние
    buttons = defaultdict(list)
    levels = defaultdict(list)
    badge_cols = defaultdict(list)
    for f in frames:
        for kind, rel in f["files"].items():
            img = cv2.imread(str(out / rel))
            for o in f["harvest"]["objects"]:
                r = o["rect"]
                if o["type"] == "value" and o.get("text"):
                    samples.append((crop(img, [r[0] - 1, r[1] - 1, r[2] + 2, r[3] + 2]), o["text"]))
                    if o.get("tile"):
                        levels[o["level"]].append(ring_color(img, o["tile"]))
                elif o["type"] in ("badge", "pill"):
                    phrases[o["text"]].append(crop(img, r).copy())
                    samples.append((crop(img, r), o["text"]))          # фразы с числом — глифами
                    if o["type"] == "badge":
                        c = crop(img, r).reshape(-1, 3).astype(np.float32)
                        badge_cols[o["state"]].append(np.median(c, axis=0).tolist())
                elif o["type"] == "button":
                    buttons[f"{o['id']}|{o['state']}"].append(crop(img, r).astype(np.uint8))
                elif o["type"] == "log" and o.get("parts"):
                    t = o["parts"][0]
                    samples.append((crop(img, [t["rect"][0] - 1, t["rect"][1] - 1, t["rect"][2] + 2, t["rect"][3] + 2]), t["text"]))
    # Два банка глифов: без склейки разрывов и со склейкой разрыва в 1 px (JPEG рвёт
    # перемычку «м» крупного шрифта, а у мелкого склейка сливает символы). Какой
    # банк читает объект лучше — решает самопроверка, отдельно для каждого объекта.
    banks = [{"file": "glyphs.npz", "gap": 1, "thr": 0.35}, {"file": "glyphs_g2.npz", "gap": 2, "thr": 0.35},
             {"file": "glyphs_t55.npz", "gap": 1, "thr": 0.55}, {"file": "glyphs_g2t55.npz", "gap": 2, "thr": 0.55}]
    for b in banks:
        sets, used, skipped = build([], samples, min_gap=b["gap"], thr=b["thr"])
        gb = GlyphBank(out / b["file"], min_gap=b["gap"], thr=b["thr"])
        gb.sets = sets
        gb.save()
        b["glyphs"], b["samples"] = sum(len(v) for v in sets.values()), used

    def dedup(lst, n=4):
        u = []
        for a in lst:
            if all(a.shape != b.shape or float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()) > 4 for b in u):
                u.append(a)
            if len(u) >= n:
                break
        return u
    np.savez_compressed(out / "phrases.npz", phrases=np.array({k: dedup(v) for k, v in phrases.items()}, dtype=object),
                        buttons=np.array({k: dedup(v, 3) for k, v in buttons.items()}, dtype=object))
    prof = {"system": sid, "page": s["page"], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frames": len(frames), "objects": objects, "banks": banks,
            "cards": [{"title": c["title"], "rect": c["rect"], "head": c["head"]} for c in cards],
            "brand": ref["harvest"]["brand"],
            "levels": {k: np.mean(np.array([c for c in v if c]), axis=0).tolist() for k, v in levels.items() if v},
            "badge_colors": {k: np.mean(np.array(v), axis=0).tolist() for k, v in badge_cols.items() if v},
            "calibration": [{"name": f["name"], "files": f["files"], "harvest": f["harvest"]} for f in frames]}
    (out / "profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    # самопроверка: каждый объект — на всех калибровочных кадрах; ниже 80 % — «ненадёжно»
    # какой банк читает объект лучше — для каждого объекта отдельно
    accs = [self_check(out, prof, bank=i) for i in range(len(banks))]
    acc = {}
    for o in prof["objects"]:
        vals = [a.get(o["id"]) for a in accs]
        if vals[0] is None:
            continue
        best = max(range(len(vals)), key=lambda i: (vals[i] or 0, -i))
        o["bank"] = best
        o["calib_acc"] = acc[o["id"]] = vals[best]
    (out / "profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    # мелкий текст банк глифов может не набрать (буквы «сут», «Ндин», «.» против «:») —
    # для чисел проверяется и Tesseract по увеличенной области; способ выбирается по точности,
    # при равенстве — глифы (быстрее)
    for m in ("tess3", "tess4"):
        ta = self_check(out, prof, bank=None, method=m)
        for o in prof["objects"]:
            if o["type"] == "value" and ta.get(o["id"], 0) > acc.get(o["id"], 0) + 0.02:
                o["method"] = m
                o["calib_acc"] = acc[o["id"]] = ta[o["id"]]
    (out / "profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    # свой банк для каждого числового поля: только его образцы (прогон цифр даёт все цифры
    # в каждой позиции). Точность — на отложенных кадрах (учим на чётных, проверяем на
    # нечётных), иначе банк «проверял бы сам себя». Для чтения — банк по всем кадрам.
    (out / "fields").mkdir(exist_ok=True)
    nums = lambda s_: re.findall(r"\d+(?:\.\d+)?", (s_ or "").replace(" ", "").replace(":", "."))
    for i_, o in enumerate(prof["objects"]):
        if o["type"] != "value" or o.get("floating"):
            continue
        smp = []
        for f in frames:
            T = {x["id"]: x for x in f["harvest"]["objects"]}
            if o["id"] not in T or not T[o["id"]].get("text"):
                continue
            r = T[o["id"]]["rect"]
            for kind, rel in f["files"].items():
                img = cv2.imread(str(out / rel))
                smp.append((crop(img, [r[0] - 1, r[1] - 1, r[2] + 2, r[3] + 2]), T[o["id"]]["text"]))
        if len(smp) < 8:
            continue
        best = None
        for gap, thr in ((1, 0.35), (2, 0.35), (1, 0.55), (2, 0.55)):
            osets, _, _ = build([], smp[::2], min_gap=gap, thr=thr)
            gb = GlyphBank(out / "fields" / "_tmp.npz", min_gap=gap, thr=thr)
            gb.sets = osets
            a = sum(nums(gb.read(c)[0]) == nums(tx) for c, tx in smp[1::2]) / max(1, len(smp[1::2]))
            if best is None or a > best[0]:
                best = (a, gap, thr)
        a, gap, thr = best
        fn = f"fields/f{i_:03d}.npz"
        osets, _, _ = build([], smp, min_gap=gap, thr=thr)
        gb = GlyphBank(out / fn, min_gap=gap, thr=thr)
        gb.sets = osets
        gb.save()
        o["own_bank"] = {"file": fn, "gap": gap, "thr": thr, "heldout_acc": round(a, 3)}
        if a >= acc.get(o["id"], 0):                 # при равенстве — свой: быстрый и по своему шрифту
            o["method"] = "own"
            o["calib_acc"] = acc[o["id"]] = round(a, 3)
    (out / "fields" / "_tmp.npz").unlink(missing_ok=True)
    (out / "profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    meth = defaultdict(int)
    for o in prof["objects"]:
        if o["type"] == "value":
            meth[o.get("method", "glyph")] += 1
    print(f"  способ чтения чисел: {dict(meth)}")
    weak = sorted(k for k, v in acc.items() if v < 0.8)
    print(f"  самопроверка: надёжно читаются {sum(v >= 0.8 for v in acc.values())} из {len(acc)}; ненадёжно: {weak}")
    cnt = defaultdict(int)
    for o in objects:
        cnt[o["type"]] += 1
    print(f"профиль: {out}  объектов {len(objects)} ({dict(cnt)}), опор-панелей {len(cards)}, "
          f"глифов {sum(len(v) for v in sets.values())} (надписей {used}, пропущено {skipped}), "
          f"фраз {len(phrases)}, состояний кнопок {len(buttons)}, уровни табло {sorted(prof['levels'])}")


def self_check(out, prof, bank=0, method="glyph"):
    from generic_vision import GenericVision, norm, same_value, similar
    from extra_reader import Tess

    class _Src:
        error = None
    g = GenericVision({"id": prof["system"]}, _Src(), None, None, Tess(), out)
    g.kw_ttl = 0
    for o in g.objects:
        if bank is not None:
            o["bank"] = bank
        o["method"] = method
    hit = defaultdict(lambda: [0, 0])
    for f in prof["calibration"]:
        T = {o["id"]: o for o in f["harvest"]["objects"]}
        for kind in ("png", "jpg"):
            img = cv2.imread(str(out / f["files"][kind]))
            g._align(img)
            for o in g.objects:
                t, typ = T.get(o["id"]), o["type"]
                if not t or typ == "trend" or (method != "glyph" and typ != "value"):
                    continue
                rd = {"value": g.read_value, "badge": g.read_phrase, "pill": g.read_phrase,
                      "button": g.read_button, "log": g.read_log}[typ](img, o)["read"]
                truth = t["state"] if typ == "button" else t["text"]
                ok = (rd == truth if typ == "button" else same_value(rd, truth) if typ == "value"
                      else similar(rd, truth) if typ == "log" else (rd is not None and norm(rd) == norm(truth)))
                hit[o["id"]][1] += 1
                hit[o["id"]][0] += bool(ok)
    return {k: round(a / b, 3) for k, (a, b) in hit.items()}


def rebuild(sid):
    """Пересобрать профиль из уже снятых кадров (без щита)."""
    out = HERE / "profiles" / sid
    cal = out / "calibration.json"
    frames = json.loads(cal.read_text(encoding="utf-8")) if cal.exists() else \
        [{"name": c["name"], "files": c["files"], "harvest": c["harvest"]}
         for c in json.loads((out / "profile.json").read_text(encoding="utf-8"))["calibration"]]
    build_profile(sid, fleet.system(sid), out, frames)


if __name__ == "__main__":
    main()
