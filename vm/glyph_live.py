# -*- coding: utf-8 -*-
"""Живая калибровка глифов: надписи, которых нет в контрольных кадрах.

В контрольных кадрах задание ЧРП и штуцера почти не меняется, а время в
журнале не размечено вовсе. Здесь щит в ВМ ставится на паузу, через DevTools
выставляются разные значения (частота, штуцер, время модели, запись в журнал),
кадр снимается с экрана ВМ, а вырезки подписываются истинным текстом.
Результат — vm/cv_templates_vm/glyphs_live/ и glyphs_live.json; затем
glyph_ocr.py собирает банк глифов уже вместе с ними.

    python glyph_live.py && python glyph_ocr.py
"""

import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vmhdmi import Cdp                     # noqa: E402
from extra_reader import ExtraReader, value_crop    # noqa: E402

BANK = HERE / "cv_templates_vm"
OUT = BANK / "glyphs_live"

# частоты и штуцер так, чтобы встретились все цифры
FREQS = [35.5, 36.0, 47.5, 58.5, 39.0, 64.5, 41.0, 52.5, 43.5, 61.0, 57.0, 49.5]
CHOKES = [20, 47, 83, 29, 96, 71, 34, 58, 65, 100, 92, 41]


def snap():
    data = urlopen("http://127.0.0.1:8781/snapshot.png", timeout=10).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def panel_tags():
    """Источники записей журнала — строки в вызовах alarm() в исходнике щита."""
    import re
    root = HERE.parent if (HERE.parent / "neft-hmi-cv.html").exists() else HERE.parent / "extracted" / "scada-cv-bridge-main"
    src = (root / "neft-hmi-cv-fullscreen.html").read_text(encoding="utf-8")
    return sorted(set(re.findall(r'alarm\("[a-z]+",\s*"([^"]+)"', src)))


def calibrate_tags(cdp, log_o):
    """Эталон каждого источника: запись в журнал → вырезка из кадра (без
    потерь и JPEG). Источник всегда рисуется в одном месте одним шрифтом,
    поэтому сравнение с эталоном надёжнее OCR (C/С, V/У)."""
    from glyph_ocr import line_ink
    x, y, w, h = [int(v) for v in log_o["bbox"]]
    out = []
    (BANK / "log_tags").mkdir(exist_ok=True)
    for i, tag in enumerate(panel_tags()):
        cdp.evaluate("t => { alarm('info', t, 'калибровка источника'); syncUI(); render(); }", tag)
        time.sleep(0.6)
        for kind, url in (("png", "http://127.0.0.1:8781/snapshot.png"), ("jpg", "http://127.0.0.1:8781/snapshot")):
            img = cv2.imdecode(np.frombuffer(urlopen(url, timeout=10).read(), np.uint8), cv2.IMREAD_COLOR)
            crop = img[y + 5:y + 20, x + 138:x + 198]
            p = f"log_tags/{i:02d}_{kind}.png"
            cv2.imwrite(str(BANK / p), crop)
            out.append({"image": p, "tag": tag})
    (BANK / "log_tags.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"эталоны источников журнала: {len(out) // 2} ({', '.join(panel_tags())})")


def main():
    OUT.mkdir(exist_ok=True)
    er = ExtraReader(BANK)
    sp = {o["id"]: o for o in er.readouts}
    log_o = er.ref["log_banner"]
    cdp = Cdp(local_port=9227)
    samples = []
    rng = np.random.default_rng(7)
    try:
        cdp.evaluate("() => { setLayout('A'); paused = true; syncUI(); }")
        for i, (f, ch) in enumerate(zip(FREQS, CHOKES)):
            t_model = int(rng.integers(0, 9 * 86400))
            truth = cdp.evaluate("""a => { ui.f = a[0]; ui.choke = a[1]; S.t = a[2];
                alarm("info", "CV", "калибровка глифов " + a[3]); syncUI(); render();
                return { log: log.slice(0, 2).map(e => e.t.slice(0, 8)),
                         ann: frameAnnotation(null, null).objects.filter(o => o.id === "sp_freq" || o.id === "sp_choke")
                                .map(o => [o.id, o.text]) }; }""", [f, ch, t_model, i])
            time.sleep(0.6)
            img = snap()
            name = f"live_{i:02d}.png"
            cv2.imwrite(str(OUT / name), img)
            for oid, text in truth["ann"]:
                crop, _, _ = value_crop(img, sp[oid]["bbox"])
                if crop is not None:
                    p = f"glyphs_live/{oid}_{i:02d}.png"
                    cv2.imwrite(str(BANK / p), crop)
                    samples.append({"image": p, "text": text})
            x, y, w, h = [int(v) for v in log_o["bbox"]]
            for k, t in enumerate(truth["log"]):
                yy = y + 5 + k * 15
                p = f"glyphs_live/logtime_{i:02d}_{k}.png"
                cv2.imwrite(str(BANK / p), img[yy:yy + 15, x + 72:x + 138])
                samples.append({"image": p, "text": t})
            print(f"  {i + 1}/{len(FREQS)}: ЧРП {f}, штуцер {ch}, время журнала {truth['log']}", flush=True)
        calibrate_tags(cdp, log_o)
    finally:
        cdp.evaluate("() => { ui.f = 50; ui.choke = 100; paused = false; syncUI(); }")
        cdp.close()
    (BANK / "glyphs_live.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"надписей: {len(samples)} → {BANK / 'glyphs_live.json'}")


if __name__ == "__main__":
    main()
