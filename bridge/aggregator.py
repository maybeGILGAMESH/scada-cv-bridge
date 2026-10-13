# -*- coding: utf-8 -*-
"""Агрегация прочитанного: из мгновенных показаний — то, чего на щите нет.

Распознавание отдаёт снимок: 82 объекта на одном кадре. Агрегатор копит
снимки во времени и добавляет сверху:

  * фильтр выбросов — медиана по трём последним чтениям: одиночный сбой
    распознавания не должен порождать ни события, ни команду;
  * производные величины — дебит нефти из дебита жидкости и обводнённости,
    скорость изменения каждого параметра, время до уставки при текущем тренде
    (когда перегреется ПЭД, когда заполнится РВС);
  * перекрёстную проверку — дебит показан и стрелкой, и цифрами: расхождение
    больше допуска означает, что врёт прибор или распознавание;
  * события — смена состояний ламп, тревог, насосов, задвижек и выход
    параметров за уставки с гистерезисом;
  * сводку по времени — готовность (доля времени в работе), число пусков
    и аварийных остановов.

Уставки здесь опережающие: ниже тех, на которых модель щита отключает насос
сама (LIM в neft-hmi-cv.html), — смысл в том, чтобы успеть вмешаться.
"""

import csv
import io
import time
from collections import deque

# тег → (уставка «внимание», уставка «авария», направление: +1 — опасен рост, −1 — падение)
LIMITS = {
    "gauge_t":   (100.0, 120.0, +1),     # ТЕМП. ПЭД, отключение моделью при 130 °C
    "gauge_i":   (36.0, 42.0, +1),       # ТОК ПЭД, отключение при 46 А
    "bar_sub":   (150.0, 60.0, -1),      # ПОГРУЖЕНИЕ, срыв подачи при 30 м
    "bar_iso":   (40.0, 12.0, -1),       # СОПР. ИЗОЛЯЦИИ, отключение при 5 кОм
    "bar_vib":   (7.0, 11.0, +1),        # ВИБРАЦИЯ
    "tank_oil":  (85.0, 92.0, +1),       # РВС-1 уровень
    "sep_level": (80.0, 90.0, +1),       # НГС-1 уровень
    "disp_wc":   (85.0, 95.0, +1),       # ОБВОДНЁННОСТЬ
}
HYST = 0.03                               # гистерезис — доля уставки
STABLE_S = 0.8                            # сколько держится состояние, чтобы его принять

NUMERIC = {"gauge": "value", "sevenseg_digit": "value", "bargraph": "value",
           "vessel_level": "level_pct"}
DISCRETE = ("lamp", "annunciator", "pump", "valve", "apparatus", "selector", "status_text")

STATE_RU = {"on": "вкл", "off": "выкл", "running": "работа", "stopped": "стоп",
            "open": "открыта", "closed": "закрыта", "partial": "прикрыта",
            "pressed": "нажата", "released": "отпущена", "blink": "мигает",
            "throttled": "дросселирует", "run": "в работе", "stop": "остановлен", "trip": "авария"}


class Tag:
    __slots__ = ("id", "label", "unit", "cls", "raw", "hist", "value", "t")

    def __init__(self, id_, label, unit, cls, maxlen):
        self.id, self.label, self.unit, self.cls = id_, label, unit, cls
        self.raw = deque(maxlen=3)
        self.hist = deque(maxlen=maxlen)      # (t, отфильтрованное значение)
        self.value = None
        self.t = None

    def push(self, t, v):
        self.raw.append(v)
        f = sorted(self.raw)[len(self.raw) // 2]    # медиана из трёх
        self.value, self.t = f, t
        self.hist.append((t, f))
        return f

    def rate_per_min(self, span=60.0):
        """Наклон по методу наименьших квадратов за последние span секунд."""
        if len(self.hist) < 4:
            return None
        t_end = self.hist[-1][0]
        pts = [(t, v) for t, v in self.hist if t >= t_end - span]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < span * 0.25:
            return None
        n = len(pts)
        mt = sum(p[0] for p in pts) / n
        mv = sum(p[1] for p in pts) / n
        den = sum((p[0] - mt) ** 2 for p in pts)
        if den <= 0:
            return None
        return 60.0 * sum((p[0] - mt) * (p[1] - mv) for p in pts) / den


class Aggregator:
    def __init__(self, history_s=3600, limits=None, rate_span=60.0):
        self.limits = dict(LIMITS if limits is None else limits)
        self.rate_span = rate_span
        self.maxlen = 20000
        self.history_s = history_s
        self.tags = {}
        self.states = {}               # дискретный объект → (состояние, подпись)
        self.levels = {}               # тег → текущий уровень уставки: None / warn / alarm
        self.events = deque(maxlen=300)
        self.t0 = None
        self.run_time = 0.0
        self.seen_time = 0.0
        self.starts = 0
        self.trips = 0
        self._last_t = None
        self._running = None
        self._deb = {}
        self.mismatch_since = None
        self.mismatch_told = False

    # ── события ─────────────────────────────────────────────────────
    def event(self, t, level, text, tag=None):
        self.events.appendleft({"t": t, "clock": time.strftime("%H:%M:%S", time.localtime(t)),
                                "level": level, "text": text, "tag": tag})

    # ── обновление по кадру ─────────────────────────────────────────
    def update(self, t, groups):
        if self.t0 is None:
            self.t0 = t
        for cls, key in NUMERIC.items():
            for it in groups.get(cls, []):
                v = it.get(key)
                if v is None or it.get("reliable") is False:
                    continue                    # недостоверное в историю не пишем
                tag = self.tags.get(it["id"])
                if tag is None:
                    tag = self.tags[it["id"]] = Tag(it["id"], it.get("label"), it.get("unit"),
                                                    cls, self.maxlen)
                tag.push(t, float(v))
                while tag.hist and tag.hist[0][0] < t - self.history_s:
                    tag.hist.popleft()
        for cls in DISCRETE:
            for it in groups.get(cls, []):
                if it.get("state") is not None:
                    self._discrete(t, cls, it)
        self._limits(t)
        self._time_stats(t)
        self._cross_check(t)

    def _discrete(self, t, cls, it):
        """Состояние принимается, только когда продержалось STABLE_S секунд.
        Мигающая тревога (неквитированная) иначе сыпала бы «вкл/выкл» на
        каждом кадре — её переключения считаются и превращаются в «мигает»."""
        oid, raw = it["id"], it["state"]
        d = self._deb.setdefault(oid, {"cand": raw, "since": t, "flips": deque(maxlen=8)})
        if raw != d["cand"]:
            d["cand"], d["since"] = raw, t
            d["flips"].append(t)
        flips = [x for x in d["flips"] if t - x < 4.0]
        st = "blink" if cls in ("annunciator", "lamp") and len(flips) >= 3 else (
            raw if t - d["since"] >= STABLE_S else None)
        if st is None:
            return
        prev = self.states.get(oid)
        self.states[oid] = (st, it.get("label") or oid, cls)
        if prev is None or prev[0] == st:
            return
        lvl = "info"
        if cls == "annunciator" and st in ("on", "blink") and prev[0] == "off":
            lvl = "alarm"
        elif oid == "lamp_trip" and st in ("on", "blink"):
            lvl = "alarm"
        elif cls == "annunciator" and st == "off":
            lvl = "ok"
        elif cls == "annunciator":
            return                      # мигание ↔ горит после квитирования — не событие
        self.event(t, lvl, f"{it.get('label')}: {STATE_RU.get(prev[0], prev[0])} → "
                           f"{STATE_RU.get(st, st)}", oid)

    def _limits(self, t):
        for tid, (warn, alarm, sign) in self.limits.items():
            v = self.value(tid)
            if v is None:
                continue
            tag = self.tags[tid]
            cur = self.levels.get(tid)

            def over(th, back=False):
                h = abs(th) * HYST
                return (v > th - (h if back else 0) if sign > 0
                        else v < th + (h if back else 0))
            if over(alarm, cur == "alarm"):
                new = "alarm"
            elif over(warn, cur in ("warn", "alarm")):
                new = "warn"
            else:
                new = None
            if new != cur:
                self.levels[tid] = new
                unit = tag.unit or ""
                if new is None:
                    self.event(t, "ok", f"{tag.label}: вернулся в норму ({v:.1f} {unit})", tid)
                else:
                    th = alarm if new == "alarm" else warn
                    word = "выше" if sign > 0 else "ниже"
                    self.event(t, "alarm" if new == "alarm" else "warn",
                               f"{tag.label}: {v:.1f} {unit} — {word} уставки "
                               f"{'аварии' if new == 'alarm' else 'внимания'} {th:g}", tid)

    def _time_stats(self, t):
        run = self.state("lamp_run") == "on"
        if self._last_t is not None:
            dt = min(5.0, t - self._last_t)
            self.seen_time += dt
            if run:
                self.run_time += dt
        if self._running is not None and run and not self._running:
            self.starts += 1
        if self._running and not run and self.state("lamp_trip") == "on":
            self.trips += 1
        self._running = run
        self._last_t = t

    def _cross_check(self, t):
        a, b = self.value("gauge_q"), self.value("disp_q")
        if a is None or b is None:
            return
        bad = abs(a - b) > max(2.0, 0.05 * max(abs(a), abs(b)))
        if not bad:
            if self.mismatch_told:
                self.event(t, "ok", "Дебит: стрелка и цифры снова сходятся", "gauge_q")
            self.mismatch_since, self.mismatch_told = None, False
            return
        if self.mismatch_since is None:
            self.mismatch_since = t
        # Стрелка и цифры обновляются не одновременно, на переходных режимах
        # короткое расхождение нормально. Сообщаем, если держится дольше 3 с.
        if not self.mismatch_told and t - self.mismatch_since >= 3.0:
            self.mismatch_told = True
            self.event(t, "warn", f"Дебит: стрелка {a:.1f}, цифры {b:.1f} м3/сут — "
                                  f"расхождение приборов дольше 3 с", "gauge_q")

    # ── доступ ──────────────────────────────────────────────────────
    def value(self, tid):
        """Отфильтрованное значение — только когда накопилось три чтения.
        Первые кадры после подключения бывают переходными (щит ещё дорисовывается),
        и одиночное чтение не должно ни поднимать тревогу, ни запускать правило."""
        tag = self.tags.get(tid)
        return None if tag is None or len(tag.raw) < 3 else tag.value

    def state(self, oid):
        s = self.states.get(oid)
        return None if s is None else s[0]

    # ── производные величины ────────────────────────────────────────
    def derived(self):
        out = []
        q = self.value("gauge_q")
        if q is None:
            q = self.value("disp_q")
        wc = self.value("disp_wc")
        if q is not None and wc is not None:
            out.append({"id": "q_oil", "label": "Дебит нефти (расчёт)", "unit": "м3/сут",
                        "value": round(q * (1 - wc / 100.0), 2),
                        "note": "дебит жидкости × (1 − обводнённость)"})
        a, b = self.value("gauge_q"), self.value("disp_q")
        if a is not None and b is not None:
            out.append({"id": "q_check", "label": "Дебит: стрелка − цифры", "unit": "м3/сут",
                        "value": round(a - b, 2),
                        "level": "warn" if self.mismatch_told else None,
                        "note": "перекрёстная проверка двух приборов"})
        for tid, target, title in (("gauge_t", 130.0, "До отключения по перегреву"),
                                   ("tank_oil", 92.0, "До заполнения РВС-1"),
                                   ("bar_sub", 30.0, "До срыва подачи")):
            eta = self.eta(tid, target)
            if eta is not None:
                out.append({"id": f"eta_{tid}", "label": title, "unit": "мин",
                            "value": round(eta, 1),
                            "level": "alarm" if eta < 2 else ("warn" if eta < 10 else None),
                            "note": "при нынешнем тренде, минут реального времени"})
        if self.seen_time > 0:
            out.append({"id": "availability", "label": "В работе за сеанс", "unit": "%",
                        "value": round(100.0 * self.run_time / self.seen_time, 1)})
        out.append({"id": "starts", "label": "Пусков / аварийных остановов", "unit": "",
                    "value": f"{self.starts} / {self.trips}"})
        return out

    def eta(self, tid, target):
        """Через сколько минут параметр дойдёт до target при нынешнем тренде."""
        tag = self.tags.get(tid)
        if tag is None or tag.value is None:
            return None
        r = tag.rate_per_min(self.rate_span)
        if r is None or abs(r) < 1e-6:
            return None
        m = (target - tag.value) / r
        return m if 0 < m < 24 * 60 else None

    def rates(self):
        out = {}
        for tid, tag in self.tags.items():
            r = tag.rate_per_min(self.rate_span)
            if r is not None:
                out[tid] = round(r, 3)
        return out

    def snapshot(self):
        return {
            "derived": self.derived(),
            "rates_per_min": self.rates(),
            "levels": {k: v for k, v in self.levels.items() if v},
            "events": list(self.events)[:60],
            "tags": {tid: {"label": t.label, "unit": t.unit, "cls": t.cls,
                           "value": None if t.value is None else round(t.value, 3)}
                     for tid, t in self.tags.items()},
            "states": {k: v[0] for k, v in self.states.items()},
            "since": self.t0,
        }

    def history(self, tid, max_points=600):
        tag = self.tags.get(tid)
        if tag is None:
            return []
        h = list(tag.hist)
        if len(h) > max_points:
            step = len(h) / max_points
            h = [h[int(i * step)] for i in range(max_points)]
        return [[round(t, 2), round(v, 3)] for t, v in h]

    def history_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";")
        ids = sorted(self.tags)
        w.writerow(["время"] + [f"{self.tags[i].label} ({self.tags[i].unit or ''})" for i in ids])
        times = sorted({round(t, 1) for tag in self.tags.values() for t, _ in tag.hist})
        idx = {i: {round(t, 1): v for t, v in self.tags[i].hist} for i in ids}
        for t in times:
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))]
                       + [("" if idx[i].get(t) is None else f"{idx[i][t]:.3f}".replace(".", ","))
                          for i in ids])
        return "﻿" + buf.getvalue()
