# -*- coding: utf-8 -*-
"""Плейбуки: что делать — собираются из того, что система видит.

Источник — только распознанное с экрана (табло, лампы, показания), здоровье
конвейера распознавания и состояние инфраструктуры. Истина модели сюда не
попадает: по ней лишь отмечается, где распознавание расходится с щитом.

Плейбук собирается под конкретную систему: шаг с органом управления, которого
нет на этом щите (в профиле распознавания нет такой кнопки), помечается
недоступным, а не предлагается вслепую. Система без профиля получает план
подготовки к работе.

Виды:
  plant      — технологический процесс (по табло и лампам щита)
  vision     — распознавание (чинится само, если включено самовосстановление)
  infra      — ВМ, «HDMI»-служба, канал сверки
  onboarding — подготовка новой системы

Шаг:
  check  — условие на распознанном (выполнен, когда условие видно на экране)
  action — нажатие органа управления щита через HID с проверкой по картинке
  fix    — исправление распознавания/инфраструктуры (безопасно, может само)
  test   — тестовый канал стенда (только для модели, не для настоящего щита)
  info   — пояснение
"""

import re

SEV_ORDER = {"critical": 0, "serious": 1, "warning": 2, "info": 3, "ok": 4}
SEV_TITLE = {"critical": "авария", "serious": "серьёзно", "warning": "внимание", "info": "сведения", "ok": "норма"}


class View:
    """Распознанное состояние щита одной системы."""

    def __init__(self, state, inventory):
        self.s = state or {}
        self.inv = set(inventory or ())
        self.objs = {o["id"]: o for o in self.s.get("objects", [])}

    def has(self, oid):
        return oid in self.inv

    def read(self, oid):
        """Прочитанное; «не уверено» (reliable = False) — как не прочитано: на
        догадке плейбук не строится и шаг ею не подтверждается."""
        o = self.objs.get(oid)
        return None if o is None or o.get("reliable") is False else o.get("read")

    def on(self, oid):
        o = self.objs.get(oid) or {}
        if "on_recent" in o:                      # мигающее табло — горит
            return bool(o["on_recent"])
        return o.get("read") in ("on", "blink", "pressed")

    def val(self, oid):
        r = self.read(oid)
        if r is None:
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", str(r))
        return float(m.group(0)) if m else None

    def label(self, oid):
        o = self.objs.get(oid)
        return (o or {}).get("label") or oid

    def check(self, cond):
        """cond: {"obj": id, "op": "on|off|>=|<=|==", "val": x} → True/False/None"""
        oid, op = cond["obj"], cond["op"]
        if oid not in self.objs:
            return None
        if op == "on":
            return self.on(oid)
        if op == "text":                         # надпись (значок, плашка) содержит текст
            r = self.read(oid)
            return None if r is None else norm_t(cond["val"]) in norm_t(r)
        if op == "off":
            r = self.read(oid)
            return None if r is None else not self.on(oid)
        v = self.val(oid)
        if v is None:
            return None
        x = cond["val"]
        return {">=": v >= x, "<=": v <= x, ">": v > x, "<": v < x, "==": abs(v - x) < 1e-6}[op]


def norm_t(t):
    return re.sub(r"\s+", " ", str(t or "")).strip().lower()


def cond_text(v, cond):
    lab = v.label(cond["obj"])
    op = cond["op"]
    if op == "text":
        return f"«{lab}» показывает «{cond['val']}»"
    if op == "on":
        return f"«{lab}» горит"
    if op == "off":
        return f"«{lab}» погашено"
    return f"{lab} {op} {cond['val']:g}"


# ───────────────────────────────────────────────────────────── шаги

def press(obj, text, expect=None, times=1, wait=20):
    return {"kind": "action", "text": text, "action": {"name": "press", "args": {"obj": obj, "times": times}},
            "control": obj, "expect": expect, "wait": wait}


def check(text, cond):
    return {"kind": "check", "text": text, "expect": cond}


def info(text):
    return {"kind": "info", "text": text}


def fix(text, name, auto=True, **args):
    return {"kind": "fix", "text": text, "action": {"name": name, "args": args}, "auto": auto}


def test(text, name, **args):
    return {"kind": "test", "text": text, "action": {"name": name, "args": args}}


# ───────────────────────────────────────────────────────────── технологический процесс

# ───────────────────────────────────────────────────────────── щит без CV-разметки (neft-scada)

def legacy_library(v):
    """Плейбуки для щита, распознанного по профилю со страницы (dom_profile).
    Источник — плашка состояния в шапке, значки «Состояние оборудования»,
    уровни табло (цвет рамки). Органы — кнопки этого щита."""
    out = []
    pill = v.read("pill") or ""
    lvl = lambda name: (v.read("lvl:" + name) or "normal")
    T = "btn:ТКРС — устранить все"
    sbros = press("btn:Сброс защит", "Сбросить защиту — «Сброс защит»", {"obj": "eq:Защита ЗСП", "op": "text", "val": "НОРМА"})
    pusk = press("btn:Пуск", "Пуск — «Пуск»", {"obj": "pill", "op": "text", "val": "В работе"})
    zsp = v.read("eq:Защита ЗСП") or ""
    # «Ингибитор» — переключатель: если дозатор уже в работе, повторное нажатие его выключит
    inh_on = norm_t(v.read("eq:Дозатор ингибитора")) == "работа"
    # «Авторегул. уровня» — тоже переключатель: включённый повторным нажатием выключится
    auto_on = norm_t(v.read("eq:Авторегулятор")) == "авто"
    auto_step = lambda text: (check("Авторегулятор уровня уже в АВТО", {"obj": "eq:Авторегулятор", "op": "text", "val": "АВТО"})
                              if auto_on else
                              press("btn:Авторегул. уровня", text, {"obj": "eq:Авторегулятор", "op": "text", "val": "АВТО"}))
    inh_step = lambda: (check("Дозатор ингибитора уже в работе", {"obj": "eq:Дозатор ингибитора", "op": "text", "val": "РАБОТА"})
                        if inh_on else
                        press("btn:Ингибитор", "Дозатор ингибитора — «Ингибитор»", {"obj": "eq:Дозатор ингибитора", "op": "text", "val": "РАБОТА"}))
    tripped = pill.lower().startswith("авария") or (zsp and zsp.upper() != "НОРМА")
    if tripped:
        # причину берём со значка «Защита ЗСП»: там она одна и короткая, читается надёжнее,
        # чем длинная плашка в шапке; плашка — запасной источник
        reason = zsp if zsp and zsp.upper() != "НОРМА" else (pill.split("·", 1)[1].strip() if "·" in pill else "?")
        r = reason.lower()
        why = f"Значок «Защита ЗСП» — {zsp or '?'}; плашка в шапке: «{pill}»."
        if "изоляц" in r:
            steps = [info("Не пускать насос: при пробое изоляции повторный пуск — риск пробоя кабеля."),
                     press(T, "Текущий ремонт — «ТКРС — устранить все»", {"obj": "d:Сопр. изоляции", "op": ">=", "val": 100}),
                     sbros, pusk]
            title = "Отключение по пробою изоляции"
        elif "перегрев" in r:
            steps = [check("Дождаться остывания ПЭД ниже 60 °C", {"obj": "d:Темп. ПЭД", "op": "<=", "val": 60}),
                     auto_step("Включить авторегулятор уровня — меньше риск сухого хода"),
                     sbros, pusk]
            title = "Отключение по перегреву ПЭД"
        elif "перегруз" in r or "ток" in r:
            steps = [press("btn:Промывка", "Промывка НКТ — «Промывка»"),
                     inh_step(),
                     sbros, pusk]
            title = "Отключение по перегрузу тока"
        else:
            steps = [info(f"Причина на щите: «{reason}». Проверить журнал под трендами."), sbros, pusk]
            title = f"Аварийное отключение: {reason}"
        out.append({"id": "plant.trip", "severity": "critical", "title": title, "why": why,
                    "evidence": ["eq:Защита ЗСП", "pill", "eq:ЭЦН / ПЭД"], "steps": steps})
    elif "ожидание апв" in pill.lower():
        out.append({"id": "plant.apv", "severity": "serious", "title": "Срыв подачи — ожидание АПВ",
                    "why": f"Плашка «{pill}», значок АПВ — {v.read('eq:АПВ') or '?'}: насос остановлен защитой, повторный пуск по таймеру.",
                    "evidence": ["pill", "eq:АПВ", "d:Погружение"],
                    "steps": [info("Дождаться автоматического повторного пуска (значок АПВ показывает отсчёт)."),
                              auto_step("Включить авторегулятор уровня — после пуска не сорвётся снова")]})
    tripped = bool(out)                  # авария или ожидание АПВ: предупреждения ниже — без повторов
    clog = v.val("eq:Коэф. засорения НКТ")
    if clog is not None and clog > 2.2 and not tripped:
        out.append({"id": "plant.asp", "severity": "warning", "title": "Отложения АСПО / солей в НКТ",
                    "why": f"Значок «Коэф. засорения НКТ» — {clog:g} (предупреждение выше 2,2): гидравлические потери "
                           "растут, ток и загрузка ПЭД пойдут вверх. Рамки табло об этом не говорят — только значок и журнал.",
                    "evidence": ["eq:Коэф. засорения НКТ", "d:Ток ПЭД", "log0"],
                    "steps": [press("btn:Промывка", "Промывка НКТ горячей нефтью — «Промывка»",
                                    {"obj": "eq:Коэф. засорения НКТ", "op": "<=", "val": 2.2}),
                              inh_step()]})
    warns = [(n, lvl(n)) for n in ("Сопр. изоляции", "Темп. ПЭД", "Ток ПЭД", "Вибрация", "Газ на приёме", "Погружение", "Р затрубное")
             if lvl(n) in ("warn", "crit")]
    for name, lv in warns:
        sev = "serious" if lv == "crit" else "warning"
        val = v.read("d:" + name)
        base = {"severity": sev, "evidence": ["d:" + name, "lvl:" + name], "kind": "plant",
                "why": f"Табло «{name}» в {'красной' if lv == 'crit' else 'жёлтой'} рамке, на экране — {val}."}
        if name == "Сопр. изоляции" and not tripped:
            out.append(dict(base, id="plant.iso_warn", title="Снижение изоляции кабеля",
                            steps=[info("Изоляция снижается — до отключения (5 кОм) запланировать ремонт."),
                                   press(T, "Текущий ремонт — «ТКРС — устранить все»", {"obj": "d:Сопр. изоляции", "op": ">=", "val": 100})]))
        elif name == "Ток ПЭД":
            out.append(dict(base, id="plant.cur_warn", title="Повышенный ток ПЭД",
                            steps=[press("btn:Промывка", "Промывка НКТ — «Промывка»"),
                                   inh_step()]))
        elif name == "Вибрация":
            out.append(dict(base, id="plant.vib_warn", title="Повышенная вибрация",
                            steps=[press("btn:Промывка", "Промывка — отложения дают дисбаланс")]))
        elif name in ("Темп. ПЭД", "Газ на приёме", "Погружение") and not tripped:
            out.append(dict(base, id=f"plant.{name}_warn", title={"Темп. ПЭД": "Повышенная температура ПЭД",
                                                                  "Газ на приёме": "Высокое газосодержание на приёме",
                                                                  "Погружение": "Низкое погружение насоса"}[name],
                            steps=[auto_step("Включить авторегулятор уровня")]))
        elif name == "Р затрубное":
            out.append(dict(base, id="plant.cas_warn", title="Высокое давление в затрубье",
                            steps=[info("Стравить газ из затрубья на кусте — клапан затрубного газа (ползунок на щите).")]))
    starts = any(st.get("control") == "btn:Пуск" for p in out for st in p["steps"])
    if "в работе" in pill.lower():
        if not out:
            out.append({"id": "plant.normal", "severity": "ok", "title": "Режим в норме",
                        "why": "Плашка «В работе», табло без жёлтых и красных рамок.", "evidence": ["pill"],
                        "steps": [check("Защита в норме", {"obj": "eq:Защита ЗСП", "op": "text", "val": "НОРМА"})]})
    elif "остановлен" in pill.lower() and not tripped and not starts:
        out.append({"id": "plant.stopped", "severity": "info", "title": "Насос остановлен",
                    "why": "Плашка «Остановлен», защиты нет." + (" Предупреждения выше — пустить можно, затем выполнить их плейбуки." if out else ""),
                    "evidence": ["pill"], "steps": [pusk]})
    return out


def plant_library(v):
    out = []
    # защита сработала: табло «ЗСП ОТКЛЮЧЕНИЕ» горит ровно, лампа «АВАРИЯ ЗСП» — мигает
    trip = v.on("ann_trip") or v.on("lamp_trip")
    iso, temp, cur = v.val("bar_iso"), v.val("gauge_t"), v.val("gauge_i")
    # «ИНГИБИТОР» — переключатель: включённый дозатор повторным нажатием выключится
    # (по текущему кадру, без защёлки мигания: погасшая лампа «горит» ещё 10 кадров)
    inh = (check("Дозатор ингибитора уже включён", {"obj": "lamp_inh", "op": "on"}) if v.read("lamp_inh") == "on" else
           press("btn_inh", "Включить дозатор ингибитора — «ИНГИБИТОР»", {"obj": "lamp_inh", "op": "on"}))

    if v.on("ann_iso") or (trip and iso is not None and iso <= 5):
        out.append({"id": "plant.iso", "severity": "critical", "title": "Отключение по низкой изоляции кабеля",
                    "why": f"Табло «НИЗКАЯ ИЗОЛЯЦИЯ» горит, сопротивление изоляции на экране — {iso if iso is not None else '?'} кОм "
                           f"(отключение ниже 5 кОм){', горит «ЗСП ОТКЛЮЧЕНИЕ»' if trip else ''}.",
                    "evidence": ["ann_iso", "bar_iso", "ann_trip"],
                    "steps": [
                        info("Не пускать насос повторно: при изоляции ниже порога пуск заблокирован, а попытки — риск пробоя кабеля."),
                        press("btn_repair", "Текущий ремонт скважины — кнопка «ТКРС» (замена кабеля и НКТ)",
                              {"obj": "bar_iso", "op": ">=", "val": 100}),
                        press("btn_ack", "Квитировать защиту — «КВИТ.»", {"obj": "ann_trip", "op": "off"}),
                        press("btn_start", "Пуск — «ПУСК»", {"obj": "lamp_run", "op": "on"}),
                        check("Изоляция в норме после ремонта", {"obj": "bar_iso", "op": ">=", "val": 30}),
                    ]})
    if v.on("ann_temp") or (temp is not None and temp >= 130):
        out.append({"id": "plant.temp", "severity": "critical" if trip else "serious", "title": "Перегрев ПЭД",
                    "why": f"Табло «ПЕРЕГРЕВ ПЭД», температура ПЭД на манометре — {temp if temp is not None else '?'} °C "
                           "(отключение выше 130 °C). Обычно — слабое охлаждение: мало жидкости или газ на приёме.",
                    "evidence": ["ann_temp", "gauge_t", "bar_gas", "bar_sub"],
                    "steps": [
                        check("Дождаться остывания ПЭД ниже 60 °C", {"obj": "gauge_t", "op": "<=", "val": 60}),
                        press("sp_freq_dn", "Снизить частоту ЧРП на 2 Гц (4 нажатия «−»)", None, times=4),
                        press("btn_ack", "Квитировать защиту — «КВИТ.»", {"obj": "ann_trip", "op": "off"}),
                        press("btn_start", "Пуск — «ПУСК»", {"obj": "lamp_run", "op": "on"}),
                        info("Если перегрев повторится — «ТКРС»: износ насоса или отложения."),
                    ]})
    if v.on("ann_cur"):
        out.append({"id": "plant.current", "severity": "critical" if trip else "serious", "title": "Перегруз по току",
                    "why": f"Табло «ПЕРЕГРУЗ ТОКА», ток ПЭД — {cur if cur is not None else '?'} А. Частая причина — АСПО и соли в НКТ.",
                    "evidence": ["ann_cur", "gauge_i", "bar_load", "ann_asp"],
                    "steps": [
                        press("btn_flush", "Промывка НКТ горячей нефтью — «ПРОМЫВКА»"),
                        inh,
                        press("sp_freq_dn", "Снизить частоту ЧРП на 2 Гц", None, times=4),
                        press("btn_ack", "Квитировать защиту — «КВИТ.»", {"obj": "ann_trip", "op": "off"}),
                        press("btn_start", "Пуск — «ПУСК»", {"obj": "lamp_run", "op": "on"}),
                    ]})
    if v.on("ann_asp") and not v.on("ann_cur"):
        out.append({"id": "plant.asp", "severity": "warning", "title": "Отложения АСПО в НКТ",
                    "why": "Табло «АСПО В НКТ»: растут потери в лифте, ток и загрузка ПЭД пойдут вверх.",
                    "evidence": ["ann_asp", "bar_load", "gauge_i"],
                    "steps": [
                        press("btn_flush", "Промывка — «ПРОМЫВКА»"),
                        inh,
                        check("Табло «АСПО В НКТ» погасло", {"obj": "ann_asp", "op": "off"}),
                    ]})
    if v.on("ann_sub"):
        out.append({"id": "plant.subm", "severity": "serious", "title": "Низкое погружение насоса",
                    "why": f"Табло «НИЗКОЕ ПОГРУЖЕНИЕ», погружение на шкале — {v.val('bar_sub')} м: насос откачивает быстрее притока, близко к срыву подачи.",
                    "evidence": ["ann_sub", "bar_sub", "mimic_well"],
                    "steps": [
                        press("sw_mode", "Включить автоматический регулятор уровня — переключатель «РЕЖИМ ЧРП» в АВТО",
                              {"obj": "lamp_auto", "op": "on"}),
                        press("sp_freq_dn", "Либо вручную снизить частоту ЧРП на 2 Гц", None, times=4),
                        press("btn_apv", "Убедиться, что АПВ включено (повторный пуск после срыва)", {"obj": "lamp_apv", "op": "on"}),
                    ]})
    if v.on("ann_gas"):
        out.append({"id": "plant.gas", "severity": "warning", "title": "Высокое содержание газа на приёме",
                    "why": f"Табло «ВЫСОКИЙ ГАЗ», газ на приёме — {v.val('bar_gas')} %: риск срыва подачи и перегрева.",
                    "evidence": ["ann_gas", "bar_gas"],
                    "steps": [press("sp_freq_dn", "Снизить частоту ЧРП на 2 Гц — погружение вырастет", None, times=4),
                              check("Табло «ВЫСОКИЙ ГАЗ» погасло", {"obj": "ann_gas", "op": "off"})]})
    if v.on("ann_vib"):
        out.append({"id": "plant.vib", "severity": "warning", "title": "Повышенная вибрация",
                    "why": f"Табло «ВИБРАЦИЯ», вибрация — {v.val('bar_vib')} мм/с.",
                    "evidence": ["ann_vib", "bar_vib"],
                    "steps": [press("sp_freq_dn", "Снизить частоту ЧРП на 2 Гц", None, times=4),
                              press("btn_flush", "Промывка — отложения дают дисбаланс"),
                              check("Табло «ВИБРАЦИЯ» погасло", {"obj": "ann_vib", "op": "off"})]})
    for oid, title, text in (("ann_wc", "Рост обводнённости", "Сообщить технологу: возможен прорыв пластовой воды; органа управления на щите нет."),
                             ("ann_cas", "Высокое давление в затрубье", "Стравить газ из затрубного пространства на кусте — с щита не управляется."),
                             ("ann_seph", "Высокий уровень в сепараторе НГС", "Передать диспетчеру: увеличить откачку из НГС."),
                             ("ann_tank", "Резервуар РВС заполнен", "Передать диспетчеру товарного парка: откачка в напорный нефтепровод.")):
        if v.on(oid):
            out.append({"id": f"plant.{oid}", "severity": "info", "title": title,
                        "why": f"Горит табло «{v.label(oid)}».", "evidence": [oid], "steps": [info(text)]})
    # защита отключила по недогрузу или срыву подачи и ждёт АПВ: шапка «ОЖИДАНИЕ АПВ»,
    # лампа «АПВ ГОТОВ» мигает. Пускать вручную не нужно — АПВ пустит сам
    apv_wait = not trip and (v.read("hdr_state") == "apv" or v.read("lamp_apv") == "blink")
    if apv_wait:
        log = v.read("log_banner") or ""
        auto = (check("Регулятор уровня уже в АВТО", {"obj": "lamp_auto", "op": "on"}) if v.read("lamp_auto") == "on" else
                press("sw_mode", "Включить регулятор уровня — «РЕЖИМ ЧРП» в АВТО, чтобы после АПВ не сорвалось снова",
                      {"obj": "lamp_auto", "op": "on"}))
        out.append({"id": "plant.apv", "severity": "serious", "title": "Отключение по недогрузу / срыву подачи — ожидание АПВ",
                    "why": "В шапке «ОЖИДАНИЕ АПВ», лампа «АПВ ГОТОВ» мигает: насос отключён защитой, повторный пуск — по таймеру."
                           + (f" Журнал: «{log}»." if "АПВ" in log or "недогруз" in log.lower() or "срыв" in log.lower() else ""),
                    "evidence": ["hdr_state", "lamp_apv", "log_banner", "bar_sub"],
                    "steps": [info("Ручной «ПУСК» не нужен: АПВ пустит насос сам. Причина — мало жидкости на приёме."),
                              auto,
                              check("Насос снова в работе после АПВ", {"obj": "lamp_run", "op": "on"})]})
    if trip and not any(p["id"] in ("plant.iso", "plant.temp", "plant.current") for p in out):
        out.append({"id": "plant.trip", "severity": "critical", "title": "Отключение защитой ЗСП",
                    "why": "Горит «АВАРИЯ ЗСП»; конкретное табло причины не горит — причина в журнале под трендами.",
                    "evidence": ["lamp_trip", "log_banner"],
                    "steps": [info("Прочитать причину в журнале (строка «ЗСП: Отключение …»)."),
                              press("btn_ack", "Квитировать — «КВИТ.»", {"obj": "ann_trip", "op": "off"}),
                              press("btn_start", "Пуск — «ПУСК»", {"obj": "lamp_run", "op": "on"})]})
    # насос стоит без защиты — «Пуск» нужен, даже если горят предупреждения (АСПО, вибрация…),
    # кроме случая, когда пуск уже есть шагом другого плейбука
    starts = any(st.get("control") == "btn_start" for p in out for st in p["steps"])
    if v.on("lamp_run"):
        if not out:
            out.append({"id": "plant.normal", "severity": "ok", "title": "Режим в норме",
                        "why": "Насос в работе, ни одно табло тревоги не горит.", "evidence": ["lamp_run"],
                        "steps": [check("Нет отключения защитой", {"obj": "ann_trip", "op": "off"}),
                                  check("Изоляция выше порога предупреждения", {"obj": "bar_iso", "op": ">=", "val": 30}),
                                  check("Температура ПЭД ниже предупреждения", {"obj": "gauge_t", "op": "<=", "val": 105})]})
    elif v.on("lamp_stop") and not trip and not starts and not apv_wait:
        out.append({"id": "plant.stopped", "severity": "info", "title": "Насос остановлен",
                    "why": "Горит «ОСТАНОВЛЕН», защиты нет." + (" Предупреждения выше — пустить можно, затем выполнить их плейбуки." if out else ""),
                    "evidence": ["lamp_stop"],
                    "steps": [press("btn_start", "Пуск — «ПУСК»", {"obj": "lamp_run", "op": "on"})]})
    return out


# ───────────────────────────────────────────────────────────── распознавание и инфраструктура

def vision_library(st, health, system=None):
    out = []
    h = health or {}
    if h.get("stream_dead"):
        out.append({"id": "infra.stream", "kind": "infra", "severity": "critical", "title": "Нет видеопотока с экрана ВМ",
                    "why": f"Новых кадров нет {h.get('stream_age_s', '?')} с. Без потока распознавание стоит.",
                    "steps": [fix("Перезапустить «HDMI»-службу (vmhdmi)", "restart_vmhdmi"),
                              fix("Если ВМ выключена — запустить её", "start_vm", auto=False)]})
    if h.get("vm_down"):
        out.append({"id": "infra.vm", "kind": "infra", "severity": "critical", "title": "ВМ со щитом не запущена",
                    "why": "Процесс QEMU этой системы не найден.",
                    "steps": [fix("Запустить ВМ", "start_vm", auto=False), fix("Запустить «HDMI»-службу", "restart_vmhdmi")]})
    if h.get("scene") == "obstructed":
        out.append({"id": "vision.obstructed", "kind": "vision", "severity": "serious", "title": "Часть щита чем-то закрыта",
                    "why": f"Блоки не находятся на своих местах: {', '.join(h.get('lost_blocks') or [])}. "
                           "Похоже на всплывающее окно или меню поверх щита.",
                    "steps": [fix("Нажать Esc в ВМ — закрыть всплывающее окно", "key", key="Escape"),
                              fix("Перепривязать блоки", "reregister"),
                              fix("Если не помогло — перезапустить щит в ВМ", "restart_panel", auto=False)]})
    if h.get("scene") == "absent":
        out.append({"id": "vision.absent", "kind": "vision", "severity": "critical", "title": "Щит не виден в кадре",
                    "why": "На экране ВМ нет ни одного блока щита (балл привязки "
                           f"{h.get('quality', 0):.2f}): щит закрыли, браузер упал или открыта консоль.",
                    "steps": [fix("Перезапустить щит в ВМ (X + Chromium)", "restart_panel"),
                              fix("Перепривязать блоки", "reregister")]})
    bad = h.get("persistent") or []
    if bad:
        cls = sorted({b["cls"] for b in bad})
        steps = [fix("Сбросить кэш распознавания и перепривязать блоки", "flush")]
        if (system or {}).get("reader") == "dom":         # профиль со страницы — пересобрать его калибровку
            steps.append(fix("Пересобрать профиль со страницы щита (калибровка по 17 состояниям, 2–3 мин)",
                             "build_profile", auto=False))
            cls = []
        if "sevenseg_digit" in cls:
            steps.append(fix("Читать цифровые индикаторы без кэша (смена цифры меньше порога кэша)", "cache_off", cls="sevenseg_digit"))
        if {"text_readout", "clock", "log"} & set(cls):
            steps.append(fix("Пересобрать глифы шрифта с этого щита", "recalibrate", auto=False))
        if {"gauge", "lamp", "annunciator", "button", "valve", "pump", "apparatus", "bargraph", "vessel_level"} & set(cls):
            steps.append(fix("Перенарезать банк эталонов с этого щита", "regen_templates", auto=False))
        out.append({"id": "vision.persistent", "kind": "vision", "severity": "serious",
                    "title": f"Распознавание стабильно расходится со щитом ({len(bad)})",
                    "why": "Расходится больше чем в половине последних кадров: " +
                           "; ".join(f"{b['label']} — прочитано {b['read']}, на щите {b['truth']}" for b in bad[:5]),
                    "evidence": [b["id"] for b in bad], "steps": steps})
    if h.get("quality") is not None and 0.2 < h["quality"] < 0.5 and h.get("scene") == "ok":
        out.append({"id": "vision.quality", "kind": "vision", "severity": "warning", "title": "Низкий балл привязки блоков",
                    "why": f"Медианный балл {h['quality']:.2f} (обычно 0,7–0,9).",
                    "steps": [fix("Перепривязать блоки", "reregister")]})
    occ = h.get("occluded") or []
    if occ:
        out.append({"id": "vision.occluded", "kind": "vision", "severity": "info", "title": "На трендах есть скрытые серии",
                    "why": "Серия не видна, потому что под ней лежит более поздняя с тем же значением "
                           "(серии рисуются в порядке легенды): " + "; ".join(occ) + ". Это не ошибка чтения.",
                    "steps": [info("Значение скрытой серии совпадает с одной из серий поверх — по картинке не различить.")]})
    if h.get("truth_down"):
        out.append({"id": "infra.truth", "kind": "infra", "severity": "warning", "title": "Сверка с моделью недоступна",
                    "why": "Нет ответа от DevTools щита в ВМ: распознавание работает, но оценить его точность нечем.",
                    "steps": [fix("Переподключить канал сверки (ssh-туннель)", "reconnect_truth")]})
    if h.get("model_paused"):
        out.append({"id": "test.paused", "kind": "vision", "severity": "info", "title": "Модель щита на паузе",
                    "why": "Часы модели не идут (тестовый канал стенда).",
                    "steps": [test("Снять паузу модели", "unpause")]})
    if h.get("fps") is not None and 0 < h["fps"] < 3 and h.get("active"):
        out.append({"id": "vision.slow", "kind": "vision", "severity": "warning", "title": "Распознавание медленное",
                    "why": f"{h['fps']:.1f} к/с — менее 3 кадров в секунду.",
                    "steps": [info("Фоновые системы читаются реже; закройте лишние окна просмотра и проверьте загрузку CPU на вкладке «На чём работает».")]})
    return out


def onboarding_library(ob):
    if not ob:
        return []
    steps = []
    for c in ob.get("checklist", []):
        steps.append({"kind": "check", "text": c["text"], "status": c["status"], "detail": c.get("detail")})
    steps += [fix("Запустить автопоиск объектов на экране", "discover"),
              fix("Сравнить с известными профилями", "match_profiles"),
              fix("Собрать профиль со страницы щита (DevTools): разметка, калибровка по 17 состояниям, самопроверка — 2–3 мин",
                  "build_profile", auto=False),
              {"kind": "info", "text": "Дальше нужен источник эталонов: если щит умеет выгружать разметку (как neft-hmi-cv) — "
                                       "«Перенарезать банк эталонов»; иначе — подтвердить найденные области и снять эталоны "
                                       "по нескольким режимам работы."}]
    return [{"id": "onboarding", "kind": "onboarding", "severity": "info",
             "title": "Подготовка новой системы к работе",
             "why": ob.get("summary") or "Для этого щита нет профиля распознавания.", "steps": steps}]


# ───────────────────────────────────────────────────────────── сборка

ALARM_WORDS = ("авария", "перегрев", "утечка", "защита", "пожар", "загазован", "обрыв", "отказ", "превышен")
ACK_WORDS = ("квит", "сброс")


def zero_library(v):
    """Незнакомый щит, распознанный без профиля (zero_vision): смысла кнопок система
    не знает, поэтому — только то, что видно по общему виду SCADA:
      * горит лампа с тревожной подписью → квитировать, если на щите есть «Квитировать»/«Сброс»;
      * стрелка или уровень у края шкалы → внимание оператора."""
    out = []
    objs = list(v.objs.values())
    lamps_on = [o for o in objs if o.get("cls") == "lamp" and o.get("read") == "вкл" and not o.get("missed")
                and any(w in (o.get("label") or "").lower() for w in ALARM_WORDS)]
    ack = next((o for o in objs if o.get("cls") == "button" and not o.get("missed")
                and any((o.get("label") or "").lower().startswith(w) for w in ACK_WORDS)), None)
    if lamps_on:
        names = ", ".join(f"«{o['label']}»" for o in lamps_on[:4])
        steps = [info("Щит незнакомый: смысл сигнала система не знает — только то, что лампа с тревожной подписью горит.")]
        if ack:
            steps.append({"kind": "action", "text": f"Квитировать — «{ack['label']}» (без проверки результата: что изменится, на незнакомом щите неизвестно)",
                          "action": {"name": "press", "args": {"obj": ack["id"], "times": 1}}, "control": ack["id"], "expect": None, "wait": 5})
        steps.append(info("Сообщить оператору: проверить причину сигнала на объекте."))
        out.append({"id": "zero.alarm", "severity": "warning", "title": f"Горит сигнал тревоги: {names}",
                    "why": f"Лампы {names} горят; по подписи — тревожные." + (f" На щите есть кнопка «{ack['label']}»." if ack else ""),
                    "evidence": [o["id"] for o in lamps_on[:4]], "steps": steps})
    edge = []
    for o in objs:
        if o.get("cls") in ("gauge", "tank", "bar") and not o.get("missed") and o.get("read"):
            val = v.val(o["id"])
            if val is None:
                continue
            if o["cls"] in ("tank", "bar") and (val >= 95 or val <= 3):
                edge.append(f"«{o.get('label')}» — {o['read']}")
    if edge:
        out.append({"id": "zero.edge", "severity": "info", "title": "Показания у края шкалы",
                    "why": "; ".join(edge[:5]) + ".", "evidence": [],
                    "steps": [info("Проверить: возможно, перелив, сухой ход или упор задвижки.")]})
    return out


def generate(system, state, inventory, health, onboarding=None, jobs=None):
    v = View(state, inventory)
    pbs = []
    if system.get("reader") == "zero" and state and state.get("objects"):
        for p in zero_library(v):
            p["kind"] = "plant"
            pbs.append(p)
    elif system.get("profile") and state and state.get("objects"):
        lib = legacy_library if system.get("reader") == "dom" else plant_library
        for p in lib(v):
            p["kind"] = "plant"
            pbs.append(p)
    for p in vision_library(state, health, system):
        pbs.append(p)
    pbs += onboarding_library(onboarding)
    jobs = jobs or []
    for p in pbs:
        p.setdefault("kind", "vision")
        p.setdefault("evidence", [])
        p["severity_title"] = SEV_TITLE[p["severity"]]
        for i, s in enumerate(p["steps"]):
            s["id"] = f"{p['id']}#{i}"
            if s.get("control") and not v.has(s["control"]):
                s["available"] = False
                s["reason"] = "на этом щите нет такого органа управления"
            elif s["kind"] == "action" and s.get("control") and s["control"] not in v.objs:
                s["available"] = False
                s["reason"] = "кнопка сейчас не найдена в кадре"
            else:
                s["available"] = s["kind"] != "info"
            if s.get("expect"):
                ok = v.check(s["expect"])
                s["expect_text"] = cond_text(v, s["expect"])
                s.setdefault("status", "done" if ok else ("unknown" if ok is None else "pending"))
            j = next((j for j in reversed(jobs) if j.get("step") == s["id"]), None)
            if j:
                s["job"] = {k: j.get(k) for k in ("status", "msg", "t")}
                if s["kind"] != "check" and j["status"] == "выполнено" and s.get("status") != "done":
                    s["status"] = "done" if not s.get("expect") else s.get("status")
            s.setdefault("status", "pending" if s["kind"] in ("action", "fix", "test") else "info")
        p["auto"] = all(s["kind"] in ("fix", "info", "check") for s in p["steps"]) and \
            any(s["kind"] == "fix" and s.get("auto", True) for s in p["steps"])
    pbs.sort(key=lambda p: (SEV_ORDER[p["severity"]], p["kind"] != "plant"))
    return pbs
