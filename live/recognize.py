# -*- coding: utf-8 -*-
"""Чтение щита с кадра и упаковка результата для сайта.

Обёртка над cv_templates/match_templates.py: тот же банк эталонов, тот же
поиск блоков, те же приёмы чтения приборов. Здесь только то, что нужно для
работы в реальном времени: кадр на входе, словарь показаний на выходе.

Ни одна координата прибора заранее не известна: сначала в кадре ищутся
блоки щита, потом приборы читаются там, где блок нашёлся.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cv_templates"))

import match_templates as MT      # noqa: E402

CLASS_TITLE = {
    "gauge": "Манометры",
    "sevenseg_digit": "Цифровые индикаторы",
    "bargraph": "Столбчатые индикаторы",
    "vessel_level": "Уровни в аппаратах",
    "lamp": "Сигнальные лампы",
    "annunciator": "Щит тревог",
    "valve": "Арматура",
    "pump": "Насосы",
    "apparatus": "Блочные аппараты",
    "button": "Кнопки пульта",
    "selector": "Переключатели",
    "status_text": "Состояние",
    "text_readout": "Текстовые табло",
    "trend": "Тренды",
}

# Порядок групп на сайте: сначала то, за чем следит оператор.
GROUP_ORDER = ["gauge", "sevenseg_digit", "bargraph", "vessel_level", "annunciator",
               "lamp", "status_text", "valve", "pump", "apparatus",
               "text_readout", "selector", "button", "trend"]


class Reader:
    def __init__(self, templates_dir, only=None, pad=36,
                 register_every=1, gauge_stride=1):
        self.bank = MT.Bank(Path(templates_dir))
        self.only = set(only.split(",")) if only else None
        self.pad = pad
        # Компоновка меняется редко, а поиск блоков — самая дорогая часть после
        # чтения приборов. Поэтому он идёт не каждый кадр, но обязательно
        # повторяется, если предыдущая привязка стала хуже.
        self.register_every = max(1, int(register_every))
        self.gauge_stride = max(1, int(gauge_stride))
        self._reg = None
        self._since = 0
        self._watch = 0            # какой блок проверять на этом кадре
        # Угол каждой стрелки с предыдущего кадра. Между кадрами стрелка далеко
        # не уходит, поэтому поиск начинается рядом с прошлым значением и при
        # неудаче честно откатывается на полный перебор.
        self._hints = {}
        self.layouts = self.bank.man.get("layouts") or {}
        self.block_titles = self.bank.man.get("block_titles") or {}

    # ── компоновка ──────────────────────────────────────────────────────
    def name_layout(self, reg):
        """Как называется найденная компоновка.

        Сдвиги блоков найдены по изображению; таблица компоновок нужна лишь
        для того, чтобы назвать результат словом, а не парой чисел. Если
        сдвиги не совпали ни с одной известной — так и пишем.
        """
        if not reg:
            return {"id": None, "name": "блоки не найдены", "deviation": None}
        if not self.layouts:
            # Манифест выгружен до появления таблицы компоновок. Блоки найдены,
            # приборы читаются — назвать компоновку словом просто нечем.
            moved = sum(1 for v in reg.values() if v[0] or v[1])
            return {"id": None, "deviation": None, "moved": moved,
                    "name": ("базовая" if not moved
                             else "переставлено блоков: %d" % moved),
                    "note": "в манифесте нет таблицы компоновок — "
                            "перевыгрузите эталоны, чтобы называть её по имени"}
        found = {}
        for name, (dx, dy, _s) in reg.items():
            base = self.bank.blocks.get(name)
            if base:
                found[name] = (base[0] + dx, base[1] + dy)
        best, best_dev = None, None
        for lid, spec in self.layouts.items():
            blocks = spec.get("blocks") or {}
            dev = 0
            n = 0
            for name, (fx, fy) in found.items():
                r = blocks.get(name)
                if not r:
                    continue
                dev = max(dev, abs(fx - r[0]), abs(fy - r[1]))
                n += 1
            if n and (best_dev is None or dev < best_dev):
                best, best_dev = lid, dev
        if best is None or best_dev is None or best_dev > 6:
            return {"id": None, "name": "не из известных компоновок",
                    "deviation": best_dev}
        return {"id": best, "name": self.layouts[best].get("name", best),
                "note": self.layouts[best].get("note"), "deviation": best_dev}

    # ── чтение кадра ────────────────────────────────────────────────────
    def read(self, frame, force_register=False):
        t0 = time.time()
        need = (force_register or self._reg is None
                or self._since >= self.register_every - 1)
        check = None
        if not need:
            # Поиск блоков идёт не каждый кадр, поэтому каждый кадр проверяется
            # один блок по очереди: если он сдвинулся или перестал узнаваться,
            # компоновку переставили и искать надо заново, не дожидаясь очереди.
            names = sorted(self._reg)
            name = names[self._watch % len(names)]
            self._watch += 1
            check = MT.verify_block(frame, self.bank, name, self._reg[name][:2])
            if check is None or check[0] < 0.45 or abs(check[1]) > 2 or abs(check[2]) > 2:
                need = True

        if need:
            reg = MT.register_blocks(frame, self.bank)
            self._reg, self._since = reg, 0
            check = None
        else:
            reg = self._reg
            self._since += 1
        self.bank.set_offsets(MT.offsets_only(reg))
        t_reg = time.time() - t0

        rows = MT.recognise(frame, self.bank, None, self.only, self.pad,
                            gauge_stride=self.gauge_stride, hints=self._hints)
        for r in rows:
            if r["cls"] == "gauge" and r.get("angle_deg") is not None:
                self._hints[r["object_id"]] = r["angle_deg"]
        t_all = time.time() - t0

        # Балл привязки: при полном поиске — худший из блоков, при пропуске —
        # балл проверенного блока. Старый балл выдавать за свежий нельзя.
        quality = (check[0] if check is not None
                   else min([v[2] for v in reg.values()], default=0.0))
        groups = {}
        for r in rows:
            cls = r["cls"]
            item = {
                "id": r["object_id"],
                "label": r.get("label") or r["object_id"],
                "score": round(float(r.get("score") or 0.0), 4),
                "block": self.bank.block_of.get(r["object_id"]),
            }
            for k in ("value", "text", "state", "level_pct", "unit", "angle_deg",
                      "fill_frac", "margin", "reliable", "opening_pct"):
                if r.get(k) is not None:
                    item[k] = (round(float(r[k]), 3)
                               if isinstance(r[k], float) else r[k])
            groups.setdefault(cls, []).append(item)
        for lst in groups.values():
            lst.sort(key=lambda z: z["id"])

        blocks = {name: {"dx": dx, "dy": dy, "score": round(sc, 4),
                         "title": self.block_titles.get(name, name)}
                  for name, (dx, dy, sc) in sorted(reg.items())}

        return {
            "layout": self.name_layout(reg),
            "blocks": blocks,
            "registration_quality": round(quality, 4),
            "groups": groups,
            "group_order": [g for g in GROUP_ORDER if g in groups],
            "group_titles": {g: CLASS_TITLE.get(g, g) for g in groups},
            "registered_this_frame": bool(need),
            "timing_ms": {"registration": round(t_reg * 1000),
                          "total": round(t_all * 1000)},
        }
