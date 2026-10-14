# -*- coding: utf-8 -*-
"""Перенарезать банк эталонов с щита, который работает в ВМ.

Исходный cv_templates/ нарезан в Chrome под Windows и с более ранней версии
щита: шрифты другие, а часть табло (давления у скважины) с тех пор переехала
и поменяла размер. Эталоны и контрольные кадры режет сам щит — кнопкой
«Выгрузить эталоны». Здесь та же функция вызывается через DevTools в
Chromium ВМ, архив перехватывается и раскладывается в vm/cv_templates_vm/ —
исходный банк не трогается.

    python regen_templates.py              # контрольные кадры во всех компоновках
    python regen_templates.py --cur        # только текущая компоновка (быстрее)
"""

import argparse
import base64
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vmhdmi import Cdp      # noqa: E402

OUT = HERE / "cv_templates_vm"

EXPORT_JS = """async (val) => {
  window.__tplZip = null;
  window.showDirectoryPicker = undefined;          // без выбора папки щит пишет zip
  const orig = download;
  download = (name, bytes) => { window.__tplZip = bytes; };
  $("selAng").value = "5";
  $("selVal").value = val;
  try { await exportTemplates(); } finally { download = orig; }
  return window.__tplZip ? window.__tplZip.length : 0;
}"""

CHUNK_JS = """(a) => { const b = window.__tplZip.subarray(a[0], a[1]); let s = "";
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s); }"""


def fix_manifest(man):
    """Поправка к выгрузке щита: у аппарата, где область измерения — сам столб
    жидкости (КУ-1), щит пишет охват шкалы 100 %, хотя эталон «полно» снят на
    своём уровне (84 %). Чтение тогда завышает уровень в 100/84 раза. У
    сепаратора область — смотровое стекло на всю шкалу, там 100 верно."""
    for t in man["templates"]:
        if (t.get("class") == "vessel_level" and t.get("measure_span_pct") == 100
                and t.get("measure_rect_rel") and t.get("measure_rect_rel") == t.get("liquid_rect_rel")
                and 0 < (t.get("level_pct") or 0) < 100):
            t["measure_span_pct"] = t["level_pct"]
            print(f"  поправка охвата шкалы: {t['object_id']} → {t['level_pct']} %")
    return man


def main():
    ap = argparse.ArgumentParser(description="Эталоны с щита в ВМ")
    ap.add_argument("--cur", action="store_true", help="контрольные кадры только в текущей компоновке")
    a = ap.parse_args()
    cdp = Cdp(local_port=9226)
    try:
        cdp.evaluate("() => { setLayout('A'); syncUI(); }")
        t = time.time()
        print("щит режет эталоны…", flush=True)
        n = cdp.evaluate(EXPORT_JS, "cur" if a.cur else "all")
        if not n:
            sys.exit("щит не отдал архив")
        print(f"архив {n / 1e6:.1f} МБ за {time.time() - t:.0f} с, забираю…", flush=True)
        buf = bytearray()
        step = 4 << 20
        for i in range(0, n, step):
            buf += base64.b64decode(cdp.evaluate(CHUNK_JS, [i, min(n, i + step)]))
        cdp.evaluate("() => { window.__tplZip = null; }")
    finally:
        cdp.close()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    with zipfile.ZipFile(io.BytesIO(bytes(buf))) as z:
        z.extractall(OUT)
    man = fix_manifest(json.loads((OUT / "manifest.json").read_text(encoding="utf-8")))
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"готово: {OUT}  эталонов {len(man['templates'])}, "
          f"контрольных кадров {len(list((OUT / 'validation').rglob('*.png')))}")


if __name__ == "__main__":
    main()
