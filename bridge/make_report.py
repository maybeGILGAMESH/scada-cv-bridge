# -*- coding: utf-8 -*-
"""Наглядный отчёт по замерам: results/*.json → results/report.html и results/charts/*.png.

    python bridge/make_report.py

Читает то, что пишут замеры:
    results/transport.json   bench_transport.py — транспорты и кодеки
    results/capture.json     bench_capture.py   — способы захвата
    results/reader.json      bench_reader.py    — способы чтения
    results/e2e_<транспорт>.json (или .md)     e2e_test.py — весь контур

report.html самодостаточен: данные и графики внутри файла, ни библиотек, ни
интернета не нужно — открывается двойным щелчком. PNG — те же графики для
README, записки или презентации. Замер отсутствует — его раздел пропускается.
"""

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

TR_TITLE = {"tcp": "TCP", "ws": "WebSocket", "udp": "UDP", "udp:1400": "UDP, MTU 1400",
            "pipe": "Локальный сокет", "shm": "Общая память", "mjpeg": "MJPEG (PiKVM)"}
TR_ORDER = ["shm", "pipe", "udp", "tcp", "ws", "mjpeg", "udp:1400"]
READER_TITLE = {"template без подсказок (match_templates)": "Шаблоны без подсказки",
                "template с подсказкой (исходный live)": "Шаблоны с подсказкой (live/)",
                "needle, 1 поток": "Стрелка, 1 поток",
                "needle, параллельно": "Стрелка, параллельно"}

# цвета графиков — те же, что в HTML для светлой темы
BLUE, ORANGE, AQUA, MUTED, CRIT, GRID, INK, INK2 = (
    "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1", "#d03b3b", "#e6e5e0", "#0b0b0b", "#52514e")


# ───────────────────────────────────────────────────────────── данные

def load_json(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def e2e_from_md(p):
    """Старые прогоны e2e_test.py писали только markdown — разбираем его."""
    s = p.read_text(encoding="utf-8")

    def g(k):
        m = re.search(r"\| " + re.escape(k) + r" \| ([^|]+) \|", s)
        return m.group(1).strip() if m else None
    lat = (g("задержка канала p50 / p95, мс") or "0 / 0").split("/")
    checks = [{"name": m.group(1).strip(), "ok": m.group(2) == "✔", "detail": m.group(3).strip()}
              for m in re.finditer(r"^\| ([^|]+) \| (✔|✘) \| ([^|]*) \|$", s, re.M)]
    date = re.search(r"Дата: (.+)", s)
    return {"transport": p.stem.split("_", 1)[1], "date": date.group(1).strip() if date else "",
            "metrics": {"link_fps": float(g("канал, к/с") or 0), "lat_p50_ms": float(lat[0]),
                        "lat_p95_ms": float(lat[1]), "read_ms": float(g("чтение кадра, мс") or 0),
                        "glass_ms": float(g("от захвата до показаний, мс") or 0),
                        "rtt_ms": float(g("RTT обратного канала, мс") or 0),
                        "cv_fps": float(g("распознавание, к/с") or 0)},
            "checks": checks, "ok": all(c["ok"] for c in checks)}


def collect():
    data = {"generated": time.strftime("%Y-%m-%d %H:%M")}
    t = load_json("transport.json")
    if t:
        data["meta"] = t.get("meta", {})
        data["codecs"] = t["codecs"]
        data["transport"] = [{
            "transport": c["transport"], "codec": c["codec"],
            "mode": "30" if c["fps_target"] else "max",
            **{k: c["result"].get(k) for k in ("fps", "lat_ms_p50", "lat_ms_p95", "delivered_pct",
                                               "mbit_s", "cpu_rx_pct", "cpu_pct")}}
            for c in t["cases"] if c["result"].get("frames")]
    data["capture"] = load_json("capture.json")
    r = load_json("reader.json")
    if r:
        data["reader"] = [dict(x, variant=READER_TITLE.get(x["variant"], x["variant"]))
                          for x in r["results"]]
        data["reader_live"] = r.get("live_ms", {})
    e2e = {}
    for p in sorted(RES.glob("e2e_*.json")):
        e2e[p.stem.split("_", 1)[1]] = json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(RES.glob("e2e_*.md")):
        e2e.setdefault(p.stem.split("_", 1)[1], e2e_from_md(p))
    data["e2e"] = [e2e[k] for k in sorted(e2e, key=lambda k: TR_ORDER.index(k) if k in TR_ORDER else 99)]
    return data


def budget(d):
    """Куда уходит время кадра: кодек, транспорт, чтение — по TCP и JPEG q80."""
    try:
        c = next(x for x in d["codecs"] if x["codec"].startswith("jpeg q80"))
        t = next(x for x in d["transport"] if x["transport"] == "tcp" and x["codec"] == "jpeg"
                 and x["mode"] == "30")
        e = next(x for x in d["e2e"] if x["transport"] == "tcp")
    except (KeyError, StopIteration):
        return None
    rows = [("Кодирование JPEG", c["enc_ms"]), ("Транспорт TCP", t["lat_ms_p50"]),
            ("Декодирование JPEG", c["dec_ms"]), ("Чтение кадра", e["metrics"]["read_ms"])]
    s = sum(v for _, v in rows)
    return [{"stage": k, "ms": v, "share": v / s} for k, v in rows]


# ───────────────────────────────────────────────────────────── PNG

def charts(d, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("matplotlib не установлен — PNG пропущены (pip install matplotlib)")
        return []
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK, "axes.titlesize": 12,
                         "axes.titleweight": "medium", "axes.titlelocation": "left",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    made = []
    ru = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

    def hbar(name, title, note, labels, values, colors, unit, log=False, fmt=lambda v: f"{v:g}"):
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.3), dpi=150)
        y = range(len(labels))[::-1]
        ax.barh(list(y), values, color=colors, height=0.56)
        for yi, v in zip(y, values):
            ax.text(v * (1.12 if log else 1) + (0 if log else max(values) * 0.01), yi,
                    (fmt(v) + " " + unit).replace(".", ","), va="center", color=INK2, fontsize=9)
        ax.set_yticks(list(y), labels)
        if log:
            ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ru)
        ax.set_xlim(right=max(values) * (6 if log else 1.22))
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, pad=14)
        fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p = out / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        made.append(p)

    def dots(name, title, note, labels, lo, hi, colors, unit):
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.3), dpi=150)
        y = list(range(len(labels)))[::-1]
        for yi, a, b, c in zip(y, lo, hi, colors):
            ax.plot([a, b], [yi, yi], color=c, lw=2, solid_capstyle="round")
            ax.plot([b, b], [yi - 0.18, yi + 0.18], color=c, lw=2)
            ax.plot(a, yi, "o", color=c, ms=7, mec="white", mew=1.5)
            ax.text(b + max(hi) * 0.02, yi, f"{a:g} / {b:g} {unit}".replace(".", ","),
                    va="center", color=INK2, fontsize=9)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, max(hi) * 1.3)
        ax.xaxis.set_major_formatter(ru)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, pad=14)
        fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p = out / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        made.append(p)

    b = budget(d)
    if b:
        hbar("budget", "Куда уходит время кадра", "мс, медианы; TCP и JPEG q80 при 30 к/с, чтение — сквозной прогон",
             [r["stage"] for r in b], [r["ms"] for r in b],
             [BLUE if r["stage"].startswith("Транспорт") else MUTED for r in b], "мс",
             fmt=lambda v: f"{v:.2f}" if v < 10 else f"{v:.0f}")
    if d.get("transport"):
        for codec, title in (("jpeg", "JPEG q80"), ("raw", "сырой кадр 6 МБ")):
            rows = [next((x for x in d["transport"] if x["transport"] == t and x["codec"] == codec
                          and x["mode"] == "30"), None) for t in TR_ORDER]
            rows = [r for r in rows if r]
            if rows:
                dots(f"latency_{codec}", f"Задержка транспортов при 30 к/с — {title}",
                     "точка — медиана, черта — 95-й процентиль, мс",
                     [TR_TITLE[r["transport"]] for r in rows], [r["lat_ms_p50"] for r in rows],
                     [r["lat_ms_p95"] for r in rows],
                     [ORANGE if r["transport"] == "ws" else BLUE for r in rows], "мс")
            rows = [next((x for x in d["transport"] if x["transport"] == t and x["codec"] == codec
                          and x["mode"] == "max"), None) for t in TR_ORDER]
            rows = [r for r in rows if r and r["fps"]]
            if rows:
                hbar(f"throughput_{codec}", f"Пропускная способность без ограничения — {title}",
                     "кадров в секунду, логарифмическая шкала",
                     [TR_TITLE[r["transport"]] for r in rows], [max(0.1, r["fps"]) for r in rows],
                     [ORANGE if r["transport"] == "ws" else BLUE for r in rows], "к/с", log=True,
                     fmt=lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}")
    if d.get("capture"):
        c = d["capture"]
        hbar("capture", "Захват с эмулятора", "кадров в секунду, Chrome headless, JPEG 80",
             [x["capture"] for x in c], [x["fps"] for x in c],
             [BLUE if x["capture"] == "screencast" else MUTED for x in c], "к/с", fmt=lambda v: f"{v:.1f}")
    if d.get("reader"):
        rows = d["reader"]
        lab = [f"{r['variant']} · {'KVM' if r['degraded'] else '1:1'}" for r in rows]
        val = [r["ms_median"] for r in rows]
        col = [BLUE if r["degraded"] else MUTED for r in rows]
        for k, v in (d.get("reader_live") or {}).items():
            lab.append({"needle+mt": "Стрелка, параллельно", "full": "Стрелка + кэш"}.get(k, k) + " · живой поток")
            val.append(v)
            col.append(AQUA)
        hbar("reader_time", "Время чтения кадра", "мс, медиана; серый — кадр 1:1, синий — через KVM, зелёный — живой поток",
             lab, val, col, "мс", log=True, fmt=lambda v: f"{v:.0f}")
        one = [r for r in rows if not r["degraded"]]
        dots("gauge_error", "Ошибка манометров, кадр 1:1", "% шкалы: точка — средняя, черта — наибольшая",
             [r["variant"] for r in one], [r["gauge_err_mean"] for r in one], [r["gauge_err_max"] for r in one],
             [CRIT if r["gauge_err_max"] > 2 else BLUE for r in one], "%")
    return made


# ───────────────────────────────────────────────────────────── HTML

HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Замеры SCADA-моста</title>
<style>
/* Отчёт стенда: сводка сверху, затем разделы по звеньям контура — канал, захват, чтение, весь контур. */
:root{
  --bg:#f3f2ee; --panel:#fcfcfb; --line:#dfded8; --fg:#121211; --fg2:#52514e; --muted:#7a7972;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --quiet:#c4c3bb; --crit:#d03b3b; --ok:#0b8f0b; --grid:#e8e7e2;
  --sans:"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; --serif:Georgia,"Times New Roman",serif;
  --mono:Consolas,"Cascadia Mono",monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#121312; --panel:#1b1c1a; --line:#30312e; --fg:#f0efec; --fg2:#c3c2b7; --muted:#8f8e86;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --quiet:#4a4b46; --crit:#e05555; --ok:#2fb52f; --grid:#2a2b28;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#121312; --panel:#1b1c1a; --line:#30312e; --fg:#f0efec; --fg2:#c3c2b7; --muted:#8f8e86;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --quiet:#4a4b46; --crit:#e05555; --ok:#2fb52f; --grid:#2a2b28;
  color-scheme:dark;
}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 var(--sans)}
.wrap{max-width:1240px;margin:0 auto;padding-inline:16px;padding-block:28px 48px;display:flex;flex-direction:column;gap:28px}
h1{font:400 30px/1.15 var(--serif);margin:0;text-wrap:balance}
h2{font-size:17px;font-weight:600;margin:0;text-wrap:balance}
h3{font-size:14px;font-weight:600;margin:0}
p{margin:0}
.lead{color:var(--fg2);max-width:72ch;margin-top:8px;font-size:15px}
.lead b{color:var(--fg);font-weight:600}
.meta{color:var(--muted);font-size:12px;margin-top:6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,13rem),1fr));gap:12px}
.kpi,.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;min-width:0}
.kpi{display:flex;flex-direction:column;gap:4px}
.kpi .l{color:var(--muted);font-size:12px}
.kpi .n{font:400 30px/1.1 var(--serif);font-variant-numeric:tabular-nums}
.kpi .n small{font:13px var(--sans);color:var(--muted);margin-left:4px}
section{display:flex;flex-direction:column;gap:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,27rem),1fr));gap:12px}
.card{display:flex;flex-direction:column;gap:8px}
.note{color:var(--muted);font-size:12px}
.chart{position:relative;min-height:120px}
.chart svg{display:block;width:100%;overflow:visible}
.chart text{font:12px var(--sans)}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;
     box-shadow:0 4px 14px rgba(0,0,0,.14);padding:5px 9px;font-size:12px;white-space:nowrap;z-index:3}
.tip .h{color:var(--fg2)}
.seg{display:inline-flex;gap:2px;background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:2px;align-self:flex-start;flex-wrap:wrap}
.seg button{font:inherit;font-size:12px;border:0;background:transparent;color:var(--fg2);padding:5px 10px;border-radius:5px;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--panel);color:var(--fg);box-shadow:0 0 0 1px var(--line)}
.seg button:focus-visible{outline:2px solid var(--s1);outline-offset:1px}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600}
td.r,th.r{text-align:right}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--fg2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.okc{color:var(--ok);font-weight:600}.badc{color:var(--crit);font-weight:600}
.checks{columns:2 22rem;column-gap:24px;font-size:12.5px}
.checks div{break-inside:avoid;padding:3px 0;border-bottom:1px solid var(--line);color:var(--fg2)}
.checks b{font-weight:600;color:var(--fg)}
footer{color:var(--muted);font-size:12px}
@media (prefers-reduced-motion: no-preference){ .card{animation:in .35s ease-out both} @keyframes in{from{opacity:.4}to{opacity:1}} }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Замеры SCADA-моста</h1>
    <p class="lead" id="takeaway"></p>
    <p class="meta" id="meta"></p>
  </header>

  <div class="kpis" id="kpis"></div>

  <section id="s-budget">
    <div class="card">
      <h2>Куда уходит время кадра</h2>
      <p class="note">Медианы, мс: кодирование и декодирование JPEG q80, задержка TCP при 30 к/с, чтение кадра в сквозном прогоне по TCP.</p>
      <div class="chart" id="c-budget"></div>
    </div>
  </section>

  <section id="s-transport">
    <h2>Транспорты «HDMI»-канала</h2>
    <div class="seg" role="group" aria-label="Кодек кадра" id="codec-seg"></div>
    <div class="grid">
      <div class="card">
        <h3>Задержка при 30 к/с</h3>
        <p class="note">От отправки до полного приёма. Точка — медиана, черта — 95-й процентиль. WebSocket выделен.</p>
        <div class="chart" id="c-lat"></div>
      </div>
      <div class="card">
        <h3>Пропускная способность без ограничения</h3>
        <p class="note">Кадров в секунду, логарифмическая шкала. Следующий кадр уходит, как только транспорт забрал предыдущий.</p>
        <div class="chart" id="c-thr"></div>
      </div>
    </div>
    <div class="card">
      <h3>Все случаи для выбранного кодека</h3>
      <p class="note">Источник и приёмник — разные процессы на одной машине. «Доставлено» — доля кадров, дошедших целиком.</p>
      <div class="tw"><table id="t-transport"><thead><tr>
        <th>Транспорт</th><th>Кодек</th><th>Режим</th><th class="r">к/с</th><th class="r">p50, мс</th><th class="r">p95, мс</th>
        <th class="r">Доставлено, %</th><th class="r">Мбит/с</th><th class="r">CPU приём, %</th><th class="r">CPU источник, %</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <section id="s-pipe">
    <div class="grid">
      <div class="card">
        <h3>Кодеки</h3>
        <p class="note">Кадр щита 1920×1080, одно ядро, медиана по 14 режимам.</p>
        <div class="tw"><table id="t-codecs"><thead><tr>
          <th>Кодек</th><th class="r">Размер, КБ</th><th class="r">Кодирование, мс</th><th class="r">Декодирование, мс</th>
        </tr></thead><tbody></tbody></table></div>
      </div>
      <div class="card">
        <h3>Захват с эмулятора</h3>
        <p class="note">Chrome headless, щит в работе, JPEG 80. Потолок задаёт страница: щит рисуется около 30 раз в секунду.</p>
        <div class="chart" id="c-cap"></div>
      </div>
    </div>
  </section>

  <section id="s-reader">
    <h2>Способы чтения щита</h2>
    <div class="grid">
      <div class="card">
        <h3>Время чтения кадра</h3>
        <p class="note">Медиана, мс, логарифмическая шкала.</p>
        <div class="legend"><span><i style="background:var(--quiet)"></i>кадр 1:1</span><span><i style="background:var(--s1)"></i>через KVM: 1280×800, JPEG 70</span><span><i style="background:var(--s3)"></i>живой поток</span></div>
        <div class="chart" id="c-rt"></div>
      </div>
      <div class="card">
        <h3>Ошибка манометров</h3>
        <p class="note">% шкалы, кадр 1:1. Точка — средняя, черта — наибольшая. Подсказка угла в live/ цепляется за ложный пик, когда стрелка прыгает.</p>
        <div class="chart" id="c-ge"></div>
      </div>
    </div>
    <div class="card">
      <h3>Точность по вариантам</h3>
      <p class="note">Верно прочитанных объектов на 14 контрольных режимах; правила сверки — как в match_templates.py.</p>
      <div class="tw"><table id="t-reader"><thead><tr>
        <th>Вариант</th><th>Вход</th><th class="r">Верно</th><th class="r">Точность, %</th><th class="r">Манометры: ср. ошибка, %</th>
        <th class="r">макс. ошибка, %</th><th class="r">Кадр, мс</th><th class="r">Первый кадр, мс</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <section id="s-e2e">
    <h2>Весь контур</h2>
    <div class="card">
      <p class="note">Эмулятор → транспорт → распознавание → команда через HID → щит; итог сверяется с истинным состоянием модели. Цифры — срез одного прогона.</p>
      <div class="tw"><table id="t-e2e"><thead><tr>
        <th>Транспорт</th><th>Дата</th><th class="r">Канал, к/с</th><th class="r">p50, мс</th><th class="r">Чтение кадра, мс</th>
        <th class="r">Захват → показания, мс</th><th class="r">RTT обратного канала, мс</th><th class="r">Проверки</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="card">
      <h3>Проверки</h3>
      <div class="checks" id="checks"></div>
    </div>
  </section>

  <footer id="foot"></footer>
</div>
<div class="tip" id="tip" hidden></div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const NF = {};
const fmt = (v, d = 1) => (v == null || isNaN(v)) ? '—'
  : (NF[d] || (NF[d] = new Intl.NumberFormat('ru-RU', {maximumFractionDigits: d}))).format(v);
const TR = {tcp:'TCP', ws:'WebSocket', udp:'UDP', 'udp:1400':'UDP, MTU 1400', pipe:'Локальный сокет', shm:'Общая память', mjpeg:'MJPEG (PiKVM)'};
const ORDER = ['shm','pipe','udp','tcp','ws','mjpeg','udp:1400'];
const SVGNS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const txt = (parent, x, y, s, o = {}) => { const t = el('text', {x, y, 'dominant-baseline':'central', 'text-anchor':o.anchor || 'start'}, parent); t.style.fill = o.color || 'var(--fg2)'; t.textContent = s; return t; };
const hide = () => { $('tip').hidden = true; };
function tip(ev, head, line) {
  const t = $('tip'); t.replaceChildren();
  const h = document.createElement('div'); h.className = 'h'; h.textContent = head; t.appendChild(h);
  const l = document.createElement('div'); l.textContent = line; t.appendChild(l);
  t.hidden = false;
  const w = t.offsetWidth, x = ev.pageX + 14 + w > document.documentElement.scrollWidth ? ev.pageX - w - 14 : ev.pageX + 14;
  t.style.left = x + 'px'; t.style.top = (ev.pageY - 12) + 'px';
}
document.addEventListener('scroll', hide, {passive: true});

function measure(svg, labels) {
  const t = el('text', {x: 0, y: -100}, svg); let w = 0;
  labels.forEach(s => { t.textContent = s; w = Math.max(w, t.getComputedTextLength()); });
  t.remove(); return Math.ceil(w);
}
function scale(o, x0, x1) {
  if (o.log) { const a = Math.log10(o.min), b = Math.log10(o.max); return v => x0 + (Math.log10(Math.max(v, o.min)) - a) / (b - a) * (x1 - x0); }
  return v => x0 + v / o.max * (x1 - x0);
}
function ticks(o) {
  if (o.log) { const out = []; for (let p = Math.ceil(Math.log10(o.min)); 10 ** p <= o.max; p++) out.push(10 ** p); return out; }
  const raw = o.max / 5, m = 10 ** Math.floor(Math.log10(raw)), st = [1, 2, 2.5, 5, 10].map(k => k * m).find(k => k >= raw);
  const out = []; for (let v = 0; v <= o.max + 1e-9; v += st) out.push(+v.toFixed(6)); return out;
}

// Строки с подписью слева: столбцы (kind 'bar') или точка с чертой (kind 'range').
function rowsChart(box, rows, o) {
  box.replaceChildren();
  const W = Math.max(300, box.clientWidth), rh = 30, top = 4, plotH = rows.length * rh, H = top + plotH + 26;
  const narrow = W < 560;              // на узком экране подпись короче, подробности — в подсказке
  const vt = r => o.valueText(r, narrow);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.aria}, box);
  const lw = measure(svg, rows.map(o.label)) + 14;
  const vw = measure(svg, rows.map(vt)) + 14;
  const hi = Math.max(...rows.map(o.kind === 'range' ? o.hi : o.value));
  const sc = {log: !!o.log, min: o.min || 1, max: o.log ? 10 ** Math.ceil(Math.log10(hi * 1.05)) : hi * 1.08};
  const x = scale(sc, lw, W - vw);
  const g0 = el('g', {}, svg);
  let lastRight = -Infinity;
  ticks(sc).forEach(v => {
    const l = el('line', {x1: x(v), x2: x(v), y1: top, y2: top + plotH}, g0); l.style.stroke = 'var(--grid)';
    const t = txt(g0, x(v), top + plotH + 14, o.log ? (v >= 1000 ? fmt(v / 1000, 0) + ' тыс.' : fmt(v, 1)) : fmt(v, 2), {anchor: 'middle', color: 'var(--muted)'});
    // подпись, налезающая на предыдущую, не рисуется — линия сетки остаётся
    const w = t.getComputedTextLength();
    if (x(v) - w / 2 < lastRight + 8) t.remove(); else lastRight = x(v) + w / 2;
  });
  rows.forEach((r, i) => {
    const cy = top + i * rh + rh / 2, g = el('g', {}, svg);
    txt(g, lw - 10, cy, o.label(r), {anchor: 'end', color: 'var(--fg)'});
    const c = o.color(r);
    if (o.kind === 'range') {
      const a = x(o.lo(r)), b = x(o.hi(r));
      const ln = el('line', {x1: a, x2: b, y1: cy, y2: cy, 'stroke-width': 2, 'stroke-linecap': 'round'}, g); ln.style.stroke = c;
      const cap = el('line', {x1: b, x2: b, y1: cy - 6, y2: cy + 6, 'stroke-width': 2}, g); cap.style.stroke = c;
      const dot = el('circle', {cx: a, cy, r: 5, 'stroke-width': 2}, g); dot.style.fill = c; dot.style.stroke = 'var(--panel)';
      txt(g, b + 8, cy, vt(r));
    } else {
      const x0 = x(sc.log ? sc.min : 0), w = Math.max(2, x(o.value(r)) - x0);
      const bar = el('rect', {x: x0, y: cy - 9, width: w, height: 18, rx: 4}, g); bar.style.fill = c;
      txt(g, x0 + w + 6, cy, vt(r));
    }
    const hit = el('rect', {x: 0, y: cy - rh / 2, width: W, height: rh}, g); hit.style.fill = 'transparent';
    hit.addEventListener('mousemove', ev => tip(ev, o.label(r), o.tip(r)));
    hit.addEventListener('mouseleave', hide);
  });
}

function table(id, rows, cells) {
  const tb = $(id).querySelector('tbody'); tb.replaceChildren();
  rows.forEach(r => {
    const tr = document.createElement('tr');
    cells.forEach(([v, right, cls]) => {
      const td = document.createElement('td'); td.textContent = v(r);
      if (right) td.className = 'r';
      if (cls && cls(r)) td.classList.add(cls(r));
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
}

// ── сводка ──
const T = D.transport || [], R = D.reader || [], E = D.e2e || [];
const find = (rows, f) => rows.find(f) || null;
const tcp = find(T, r => r.transport === 'tcp' && r.codec === 'jpeg' && r.mode === '30');
const fast = find(R, r => r.variant === 'Стрелка, параллельно' && !r.degraded);
const fastK = find(R, r => r.variant === 'Стрелка, параллельно' && r.degraded);
const tpl = find(R, r => r.variant === 'Шаблоны без подсказки' && !r.degraded);
const codecQ = find(D.codecs || [], r => r.codec.startsWith('jpeg q80'));
const e2tcp = find(E, r => r.transport === 'tcp');
let budget = null;
if (tcp && codecQ && e2tcp) {
  budget = [['Кодирование JPEG', codecQ.enc_ms], ['Транспорт TCP', tcp.lat_ms_p50], ['Декодирование JPEG', codecQ.dec_ms], ['Чтение кадра', e2tcp.metrics.read_ms]]
    .map(([stage, ms]) => ({stage, ms}));
  const s = budget.reduce((a, r) => a + r.ms, 0); budget.forEach(r => r.share = r.ms / s);
}
const passed = E.reduce((a, e) => a + e.checks.filter(c => c.ok).length, 0), total = E.reduce((a, e) => a + e.checks.length, 0);
const lead = $('takeaway');
const add = (s, b) => { const n = b ? document.createElement('b') : document.createTextNode(s); if (b) n.textContent = s; lead.appendChild(n); };
if (budget) { add('Транспорт занимает '); add(fmt(budget[1].share * 100, 1) + ' %', true); add(' времени кадра — выбирать его стоит по удобству, а не по скорости. '); }
if (fast && tpl) { add('Узкое место — распознавание: стрелочный метод читает кадр за '); add(fmt(fast.ms_median, 0) + ' мс', true); add(' вместо '); add(fmt(tpl.ms_median / 1000, 1) + ' с', true); add(` (в ${fmt(tpl.ms_median / fast.ms_median, 0)} раз быстрее) с той же точностью.`); }
$('meta').textContent = `${(D.meta && D.meta.machine) || ''} · отчёт собран ${D.generated}`;

const K = [
  tcp && ['TCP, задержка p50 при 30 к/с, JPEG', fmt(tcp.lat_ms_p50, 2), 'мс', 'бюджет кадра при 30 к/с — 33 мс'],
  fast && ['Чтение кадра, стрелка, параллельно', fmt(fast.ms_median, 0), 'мс', tpl ? `шаблоны без подсказки: ${fmt(tpl.ms_median, 0)} мс` : ''],
  fast && ['Точность на 14 контрольных режимах', fmt(fast.acc, 2), '%', fastK ? `через KVM: ${fmt(fastK.acc, 2)} %` : ''],
  total && ['Сквозные проверки', fmt(passed, 0), `из ${fmt(total, 0)}`, `транспорты: ${E.map(e => TR[e.transport] || e.transport).join(', ')}`],
].filter(Boolean);
$('kpis').replaceChildren(...K.map(([l, n, u, s]) => {
  const d = document.createElement('div'); d.className = 'kpi';
  d.innerHTML = '<span class="l"></span><span class="n"><span></span><small></small></span><span class="l"></span>';
  const sp = d.querySelectorAll('span'); sp[0].textContent = l; sp[2].textContent = n; d.querySelector('small').textContent = u; sp[3].textContent = s;
  return d;
}));

// ── переключатель кодека ──
let codec = 'jpeg';
const CODECS = [['jpeg', 'JPEG q80'], ['png', 'PNG'], ['raw', 'Сырой BGR · 6 МБ']].filter(([c]) => T.some(r => r.codec === c));
$('codec-seg').replaceChildren(...CODECS.map(([c, name]) => {
  const b = document.createElement('button'); b.type = 'button'; b.id = 'cd-' + c; b.textContent = name;
  b.setAttribute('aria-pressed', String(c === codec));
  b.addEventListener('click', () => { codec = c; document.querySelectorAll('#codec-seg button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw(); });
  return b;
}));

function draw() {
  const S1 = 'var(--s1)', S2 = 'var(--s2)', S3 = 'var(--s3)', Q = 'var(--quiet)', CR = 'var(--crit)';
  if (budget) rowsChart($('c-budget'), budget, {aria: 'Бюджет времени кадра', label: r => r.stage, value: r => r.ms,
    valueText: r => fmt(r.ms, r.ms < 10 ? 2 : 0) + ' мс', color: r => r.stage === 'Транспорт TCP' ? S1 : Q,
    tip: r => `${fmt(r.ms, 2)} мс · ${fmt(r.share * 100, 1)} % суммы`});
  else $('s-budget').hidden = true;

  if (T.length) {
    const pick = mode => ORDER.map(t => find(T, r => r.transport === t && r.codec === codec && r.mode === mode)).filter(Boolean);
    rowsChart($('c-lat'), pick('30'), {aria: 'Задержка транспортов', kind: 'range', label: r => TR[r.transport] || r.transport,
      lo: r => r.lat_ms_p50, hi: r => r.lat_ms_p95,
      valueText: (r, short) => short ? `${fmt(r.lat_ms_p50, 2)} мс` : `${fmt(r.lat_ms_p50, 2)} / ${fmt(r.lat_ms_p95, 2)} мс` + (r.delivered_pct < 100 ? ` · ${fmt(r.delivered_pct, 1)} % кадров` : (r.fps < 25 ? ` · ${fmt(r.fps, 0)} к/с` : '')),
      color: r => r.transport === 'ws' ? S2 : S1,
      tip: r => `p50 ${fmt(r.lat_ms_p50, 2)} мс · p95 ${fmt(r.lat_ms_p95, 2)} мс · ${fmt(r.fps, 1)} к/с · доставлено ${fmt(r.delivered_pct, 1)} %`});
    rowsChart($('c-thr'), pick('max').filter(r => r.fps > 0), {aria: 'Пропускная способность', log: true, min: 0.1, label: r => TR[r.transport] || r.transport,
      value: r => Math.max(0.1, r.fps), valueText: r => fmt(r.fps, r.fps < 10 ? 1 : 0) + ' к/с', color: r => r.transport === 'ws' ? S2 : S1,
      tip: r => `${fmt(r.fps, 0)} к/с · p50 ${fmt(r.lat_ms_p50, 2)} мс · доставлено ${fmt(r.delivered_pct, 1)} % · CPU ${fmt(r.cpu_rx_pct, 0)} / ${fmt(r.cpu_pct, 0)} %`});
    const all = ['30', 'max'].flatMap(m => ORDER.map(t => find(T, r => r.transport === t && r.codec === codec && r.mode === m)).filter(Boolean));
    table('t-transport', all, [[r => TR[r.transport] || r.transport], [r => r.codec], [r => r.mode === '30' ? '30 к/с' : 'без огр.'],
      [r => fmt(r.fps, 1), 1], [r => fmt(r.lat_ms_p50, 2), 1], [r => fmt(r.lat_ms_p95, 2), 1],
      [r => fmt(r.delivered_pct, 1), 1, r => r.delivered_pct < 99 ? 'badc' : ''], [r => fmt(r.mbit_s, 0), 1],
      [r => fmt(r.cpu_rx_pct, 1), 1], [r => fmt(r.cpu_pct, 1), 1]]);
  } else $('s-transport').hidden = true;

  if (D.capture) rowsChart($('c-cap'), D.capture, {aria: 'Захват с эмулятора', label: r => r.capture, value: r => r.fps,
    valueText: r => fmt(r.fps, 1) + ' к/с', color: r => r.capture === 'screencast' ? S1 : Q,
    tip: r => `${fmt(r.fps, 1)} к/с · интервал p50 ${fmt(r.interval_ms_p50, 0)} мс, p95 ${fmt(r.interval_ms_p95, 0)} мс · ${fmt(r.kb, 0)} КБ`});

  if (R.length) {
    const rows = R.map(r => ({...r, lab: `${r.variant} · ${r.degraded ? 'KVM' : '1:1'}`, kind: r.degraded ? 'kvm' : 'one'}));
    const LIVE = {'needle+mt': 'Стрелка, параллельно', full: 'Стрелка + кэш'};
    Object.entries(D.reader_live || {}).forEach(([k, v]) => rows.push({lab: `${LIVE[k] || k} · живой поток`, ms_median: v, kind: 'live'}));
    rowsChart($('c-rt'), rows, {aria: 'Время чтения кадра', log: true, min: 10, label: r => r.lab, value: r => r.ms_median,
      valueText: r => r.ms_median >= 1000 ? fmt(r.ms_median / 1000, 1) + ' с' : fmt(r.ms_median, 0) + ' мс',
      color: r => r.kind === 'kvm' ? S1 : (r.kind === 'live' ? S3 : Q),
      tip: r => `кадр ${fmt(r.ms_median, 0)} мс` + (r.first_ms ? ` · первый ${fmt(r.first_ms, 0)} мс · точность ${fmt(r.acc, 2)} %` : '')});
    rowsChart($('c-ge'), R.filter(r => !r.degraded), {aria: 'Ошибка манометров', kind: 'range', label: r => r.variant,
      lo: r => r.gauge_err_mean, hi: r => r.gauge_err_max,
      valueText: (r, short) => short ? `до ${fmt(r.gauge_err_max, 2)} %` : `${fmt(r.gauge_err_mean, 2)} / ${fmt(r.gauge_err_max, 2)} %`,
      color: r => r.gauge_err_max > 2 ? CR : S1, tip: r => `манометры ${r.gauge_ok} из ${r.gauge_total} · средняя ${fmt(r.gauge_err_mean, 3)} %, наибольшая ${fmt(r.gauge_err_max, 3)} % шкалы`});
  } else $('s-reader').hidden = true;
}

// ── таблицы ──
if (D.codecs) table('t-codecs', D.codecs, [[r => r.codec], [r => fmt(r.kb, 0), 1], [r => fmt(r.enc_ms, 2), 1], [r => fmt(r.dec_ms, 2), 1]]);
if (R.length) table('t-reader', R, [[r => r.variant], [r => r.degraded ? 'через KVM' : 'кадр 1:1'], [r => `${r.ok} из ${r.total}`, 1],
  [r => fmt(r.acc, 2), 1], [r => fmt(r.gauge_err_mean, 3), 1], [r => fmt(r.gauge_err_max, 2), 1, r => r.gauge_err_max > 2 ? 'badc' : ''],
  [r => fmt(r.ms_median, 0), 1], [r => fmt(r.first_ms, 0), 1]]);
if (E.length) {
  table('t-e2e', E, [[r => TR[r.transport] || r.transport], [r => r.date || ''], [r => fmt(r.metrics.link_fps, 1), 1], [r => fmt(r.metrics.lat_p50_ms, 2), 1],
    [r => fmt(r.metrics.read_ms, 0), 1], [r => fmt(r.metrics.glass_ms, 0), 1], [r => fmt(r.metrics.rtt_ms, 2), 1],
    [r => `${r.checks.filter(c => c.ok).length} из ${r.checks.length}`, 1, r => r.checks.every(c => c.ok) ? 'okc' : 'badc']]);
  const names = [...new Set(E.flatMap(e => e.checks.map(c => c.name)))];
  $('checks').replaceChildren(...names.map(n => {
    const d = document.createElement('div');
    const b = document.createElement('b'); b.textContent = n; d.appendChild(b);
    E.forEach(e => { const c = e.checks.find(x => x.name === n); if (!c) return;
      const s = document.createElement('span'); s.className = c.ok ? 'okc' : 'badc';
      s.textContent = `  ${c.ok ? '✔' : '✘'} ${TR[e.transport] || e.transport}`; s.title = c.detail; d.appendChild(s); });
    return d;
  }));
} else $('s-e2e').hidden = true;
$('foot').textContent = 'Собрано bridge/make_report.py из bridge/results/*.json. Пересобрать после новых замеров: python bridge/make_report.py';

draw();
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
})();
</script>
</body>
</html>
"""


def main():
    d = collect()
    if not any(d.get(k) for k in ("transport", "capture", "reader", "e2e")):
        sys.exit(f"нет результатов в {RES}: сначала запустите bench_*.py или e2e_test.py")
    # «</» внутри JSON закрыл бы тег <script> раньше времени
    blob = json.dumps(d, ensure_ascii=False).replace("</", "<\\/")
    out = RES / "report.html"
    out.write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    print("Отчёт  :", out)
    for p in charts(d, RES / "charts"):
        print("График :", p)


if __name__ == "__main__":
    main()
