# -*- coding: utf-8 -*-
"""Управление щитом по картинке: найти орган управления в кадре, нажать,
убедиться по следующим кадрам, что щит отреагировал.

Здесь нет ни одной заранее известной координаты кнопки. Кнопка — такой же
объект распознавания, как манометр: её положение в кадре берётся из
последнего чтения (поле "at"), затем переводится из координат щита
1920×1080 в координаты кадра трансляции (калибровка нормализатора) и
отправляется в HID. Если кнопку переставили вместе с блоком, клик уйдёт
туда, где она нашлась.

Каждая команда проверяется по картинке, а не верится на слово: «ПУСК»
считается выполненным, когда лампа «НАСОС В РАБОТЕ» загорелась на кадрах
после нажатия. Не загорелась — команда помечается невыполненной, и в журнал
пишется, что видно на щите (например, горит «АВАРИЯ ЗСП» — пуск запрещён).

HID-каналы:
    EmulatorHid  обратный канал emulator.py: JSON-строки по TCP
    PiKvmHid     настоящий PiKVM: HTTP API kvmd (/api/hid/events/...).
                 Написан по документации kvmd, на живом PiKVM не проверялся.
"""

import json
import socket
import ssl
import threading
import time
import urllib.parse
import urllib.request
from collections import deque

from protocol import clock


# ───────────────────────────────────────────────────────────── HID

class EmulatorHid:
    """Клиент обратного канала эмулятора. Переподключается сам."""

    name = "эмулятор"

    def __init__(self, host, port, timeout=5.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock = None
        self.file = None
        self.lock = threading.Lock()
        self.rtt_ms = None
        self.info = {}
        self._next_id = 0
        self._retry_at = 0.0

    @property
    def connected(self):
        return self.sock is not None

    def _connect(self):
        if time.time() < self._retry_at:
            raise ConnectionError("обратный канал недоступен")
        try:
            self.sock = socket.create_connection(self.addr, timeout=2.0)
        except OSError as e:
            self._retry_at = time.time() + 2.0
            self.sock = None
            raise ConnectionError(f"обратный канал {self.addr[0]}:{self.addr[1]}: {e}") from e
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(self.timeout)
        self.file = self.sock.makefile("rwb")
        self.info = self._call_locked({"op": "hello"})

    def _call_locked(self, msg):
        self._next_id += 1
        msg["id"] = self._next_id
        self.file.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        self.file.flush()
        while True:
            line = self.file.readline()
            if not line:
                raise ConnectionError("обратный канал закрыт")
            res = json.loads(line)
            if res.get("id") == msg["id"]:
                return res

    def call(self, op, **kw):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.sock is None:
                        self._connect()
                    res = self._call_locked(dict(kw, op=op))
                    if not res.get("ok", True):
                        raise RuntimeError(res.get("error") or "ошибка эмулятора")
                    return res
                except (OSError, ConnectionError, ValueError) as e:
                    self.close()
                    if attempt:
                        raise ConnectionError(str(e)) from e

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        self.sock = self.file = None

    def click(self, x, y):
        return self.call("click", x=round(x, 1), y=round(y, 1))

    def key(self, k):
        return self.call("key", key=k)

    def show(self, data):
        return self.call("info", data=data)

    def scenario(self, action, arg=None):
        return self.call("scenario", action=action, arg=arg)

    def truth(self):
        return self.call("truth").get("truth")

    def ping(self):
        t = clock()
        self.call("ping", t=t)
        self.rtt_ms = (clock() - t) * 1000.0
        return self.rtt_ms


class PiKvmHid:
    """Мышь и клавиатура настоящего PiKVM через HTTP API kvmd.

    Абсолютные координаты мыши у kvmd — от −32768 до 32767 по каждой оси
    экрана ПК-1, поэтому нужен размер кадра захвата (frame_size): клик в точку
    кадра (x, y) переводится в долю экрана. Сертификат PiKVM самоподписанный,
    проверка отключена — это сеть стенда, а не интернет.

    Не проверялся на живом PiKVM: написан по описанию API kvmd.
    """

    name = "PiKVM"

    def __init__(self, url, user="admin", passwd="admin", frame_size=(1920, 1080)):
        self.url = url.rstrip("/")
        self.headers = {"X-KVMD-User": user, "X-KVMD-Passwd": passwd}
        self.frame_size = frame_size
        self.ctx = ssl._create_unverified_context()
        self.rtt_ms = None
        self.info = {"pikvm": self.url}

    connected = True

    def _post(self, path, **params):
        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{self.url}{path}?{q}", method="POST", headers=self.headers)
        with urllib.request.urlopen(req, timeout=3, context=self.ctx) as r:
            return r.read()

    def click(self, x, y):
        w, h = self.frame_size
        ax = int(round(x / w * 65535 - 32768))
        ay = int(round(y / h * 65535 - 32768))
        ax = max(-32768, min(32767, ax))
        ay = max(-32768, min(32767, ay))
        self._post("/api/hid/events/send_mouse_move", to_x=ax, to_y=ay)
        self._post("/api/hid/events/send_mouse_button", button="left", state="true")
        time.sleep(0.05)
        self._post("/api/hid/events/send_mouse_button", button="left", state="false")
        return {"abs": [ax, ay]}

    def key(self, k):
        self._post("/api/hid/events/send_key", key=k)
        return {}

    def show(self, data):
        return {"shown": False}            # на экран ПК-1 через KVM ничего не дорисовать

    def scenario(self, action, arg=None):
        raise RuntimeError("тестового канала у настоящего щита нет")

    def truth(self):
        return None

    def ping(self):
        t = clock()
        req = urllib.request.Request(f"{self.url}/api/auth/check", headers=self.headers)
        with urllib.request.urlopen(req, timeout=3, context=self.ctx):
            pass
        self.rtt_ms = (clock() - t) * 1000.0
        return self.rtt_ms

    def close(self):
        pass


# ───────────────────────────────────────────────────────────── команды

# команда → (кнопка, проверка: (объект, ожидаемое состояние) либо None, подпись)
SIMPLE = {
    "start":  ("btn_start", ("lamp_run", "on"), "Пуск УЭЦН"),
    "stop":   ("btn_stop", ("lamp_stop", "on"), "Останов УЭЦН"),
    "estop":  ("btn_estop", ("lamp_trip", "on"), "Аварийный останов"),
    "ack":    ("btn_ack", None, "Квитирование"),
    "flush":  ("btn_flush", None, "Промывка НКТ"),
    "repair": ("btn_repair", None, "ТКРС"),
}
TOGGLE = {                                  # переключатели: проверка — лампа сменила состояние
    "mode": ("sw_mode", "lamp_auto", "Режим ЧРП РУЧН/АВТО"),
    "apv":  ("btn_apv", "lamp_apv", "АПВ вкл/выкл"),
    "inh":  ("btn_inh", "lamp_inh", "Дозатор ингибитора"),
}
STEP = {                                    # «больше / меньше»
    "freq_up": ("sp_freq_up", "Задание ЧРП +0,5 Гц"),
    "freq_dn": ("sp_freq_dn", "Задание ЧРП −0,5 Гц"),
    "choke_up": ("sp_choke_up", "Штуцер +5 %"),
    "choke_dn": ("sp_choke_dn", "Штуцер −5 %"),
}
FREQ_STEP, CHOKE_STEP, CHOKE_MIN, CHOKE_MAX = 0.5, 5, 20, 100
COMMANDS = list(SIMPLE) + list(TOGGLE) + list(STEP) + ["set_freq", "set_choke"]


class Controller:
    """Очередь команд, исполняемых по одной, с проверкой по картинке."""

    def __init__(self, hid, agg, verify_timeout=8.0):
        self.hid = hid
        self.agg = agg
        self.verify_timeout = verify_timeout
        self.cv = threading.Condition()
        self.groups = {}
        self.cal = None                  # (масштаб, x, y): щит 1920×1080 → кадр трансляции
        self.frame_no = 0
        self.jobs = deque(maxlen=60)
        self.queue = deque()
        self._id = 0
        threading.Thread(target=self._worker, daemon=True).start()

    # ── свежие данные от распознавания ──────────────────────────────
    def on_reading(self, groups, cal):
        with self.cv:
            self.groups = groups
            self.cal = cal
            self.frame_no += 1
            self.cv.notify_all()

    def _wait_frames(self, n, timeout):
        """Дождаться n новых чтений: состояние после клика видно не сразу."""
        with self.cv:
            target = self.frame_no + n
            self.cv.wait_for(lambda: self.frame_no >= target, timeout)

    def _wait_until(self, pred, timeout):
        t_end = time.time() + timeout
        with self.cv:
            while time.time() < t_end:
                if pred():
                    return True
                self.cv.wait(min(0.5, t_end - time.time()))
        return pred()

    # ── где нажимать ────────────────────────────────────────────────
    def locate(self, oid):
        with self.cv:
            for items in self.groups.values():
                for it in items:
                    if it["id"] == oid and it.get("at"):
                        x, y = it["at"]
                        w, h = it.get("size") or (0, 0)
                        return x + w / 2.0, y + h / 2.0
        return None

    def to_frame(self, px, py):
        cal = self.cal
        if not cal:
            return px, py
        s, x0, y0 = cal
        return x0 + px * s, y0 + py * s

    def press(self, oid, times=1, gap=0.12):
        p = self.locate(oid)
        if p is None:
            raise RuntimeError(f"«{oid}» не найден в кадре — нажимать некуда")
        fx, fy = self.to_frame(*p)
        for i in range(times):
            self.hid.click(fx, fy)
            if i + 1 < times:
                time.sleep(gap)
        return {"panel": [round(p[0]), round(p[1])], "frame": [round(fx), round(fy)]}

    # ── очередь ─────────────────────────────────────────────────────
    def submit(self, cmd, args=None, source="оператор"):
        if cmd not in COMMANDS:
            raise ValueError(f"нет такой команды: {cmd}")
        if cmd == "estop" and not (args or {}).get("confirm"):
            raise ValueError("аварийный останов требует подтверждения (confirm)")
        with self.cv:
            self._id += 1
            job = {"id": self._id, "cmd": cmd, "args": args or {}, "source": source,
                   "status": "в очереди", "t": time.time(), "detail": ""}
            self.jobs.appendleft(job)
            self.queue.append(job)
            self.cv.notify_all()
        return job

    def busy(self):
        with self.cv:
            return bool(self.queue) or any(j["status"] == "выполняется" for j in self.jobs)

    def _worker(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: bool(self.queue))
                job = self.queue.popleft()
                job["status"] = "выполняется"
            t0 = time.time()
            try:
                ok, detail = self._run(job["cmd"], job["args"])
                job["status"] = "выполнено" if ok else "не подтвердилось"
                job["detail"] = detail
            except Exception as e:
                job["status"] = "ошибка"
                job["detail"] = str(e)
            job["ms"] = round((time.time() - t0) * 1000)
            lvl = {"выполнено": "ok", "не подтвердилось": "warn"}.get(job["status"], "alarm")
            self.agg.event(time.time(), lvl,
                           f"Команда «{self.title(job['cmd'])}» ({job['source']}): "
                           f"{job['status']}" + (f" — {job['detail']}" if job["detail"] else ""),
                           "control")

    @staticmethod
    def title(cmd):
        for table in (SIMPLE, TOGGLE):
            if cmd in table:
                return table[cmd][-1]
        if cmd in STEP:
            return STEP[cmd][1]
        return {"set_freq": "Задание ЧРП", "set_choke": "Штуцер"}.get(cmd, cmd)

    # ── исполнение ──────────────────────────────────────────────────
    def _run(self, cmd, args):
        st = self.agg.state
        if cmd in SIMPLE:
            btn, check, _ = SIMPLE[cmd]
            where = self.press(btn)
            if check is None:
                self._wait_frames(2, 3.0)
                return True, f"нажато в точке кадра {where['frame']}"
            oid, want = check
            if self._wait_until(lambda: st(oid) == want, self.verify_timeout):
                return True, f"на щите: {oid} = {want}"
            hint = ""
            if cmd == "start" and st("lamp_trip") in ("on", "blink"):
                hint = " — горит «АВАРИЯ ЗСП», пуск запрещён защитой (нужно квитировать)"
            return False, f"{oid} = {st(oid)}, ожидалось {want}{hint}"

        if cmd in TOGGLE:
            btn, lamp, _ = TOGGLE[cmd]
            before = st(lamp)
            self.press(btn)
            if self._wait_until(lambda: st(lamp) not in (None, before), self.verify_timeout):
                return True, f"{lamp}: {before} → {st(lamp)}"
            return False, f"{lamp} осталась {before}"

        if cmd in STEP:
            n = max(1, min(20, int(args.get("n", 1))))
            btn, _ = STEP[cmd]
            if cmd.startswith("freq") and st("lamp_auto") == "on":
                return False, "задание ЧРП заблокировано: регулятор в режиме АВТО"
            where = self.press(btn, times=n)
            self._wait_frames(2, 3.0)
            return True, f"нажато {n} раз в точке кадра {where['frame']}"

        if cmd == "set_freq":
            return self._set_freq(float(args["value"]))
        if cmd == "set_choke":
            return self._set_choke(float(args["value"]))
        raise ValueError(cmd)

    def _stable_value(self, tid, span=1.5, tol=0.15, timeout=20.0):
        """Значение, которое перестало меняться: частота после пуска разгоняется,
        мерить её посреди разгона бессмысленно."""
        t_end = time.time() + timeout
        last, since = None, time.time()
        while time.time() < t_end:
            self._wait_frames(1, 2.0)
            v = self.agg.value(tid)
            if v is None:
                continue
            if last is None or abs(v - last) > tol:
                last, since = v, time.time()
            elif time.time() - since >= span:
                return v
        return last

    def _set_freq(self, target):
        """Замкнутый контур: задание ЧРП само не читается, зато фактическая
        частота на цифровом индикаторе в ручном режиме выходит на задание."""
        st = self.agg.state
        target = max(35.0, min(65.0, round(target * 2) / 2))
        if st("lamp_run") != "on":
            return False, "насос стоит: фактическая частота 0, задание не проверить"
        if st("lamp_auto") == "on":
            return False, "регулятор в режиме АВТО — частоту ведёт он"
        for attempt in range(4):
            cur = self._stable_value("disp_f")
            if cur is None:
                return False, "частота ЧРП не читается"
            n = int(round((target - cur) / FREQ_STEP))
            if n == 0:
                return True, f"частота {cur:.1f} Гц (попыток: {attempt})"
            self.press("sp_freq_up" if n > 0 else "sp_freq_dn", times=min(abs(n), 30), gap=0.08)
        cur = self._stable_value("disp_f")
        ok = cur is not None and abs(cur - target) < FREQ_STEP / 2 + 0.05
        return ok, f"частота {cur} Гц, цель {target}"

    def _set_choke(self, target):
        """Открытый контур: положение штуцера на щите цифрой не читается.
        Поэтому как у шагового двигателя без датчика — сначала в упор
        (минимум 20 %), затем нужное число шагов вверх."""
        target = int(max(CHOKE_MIN, min(CHOKE_MAX, round(target / CHOKE_STEP) * CHOKE_STEP)))
        self.press("sp_choke_dn", times=(CHOKE_MAX - CHOKE_MIN) // CHOKE_STEP + 1, gap=0.06)
        up = (target - CHOKE_MIN) // CHOKE_STEP
        if up:
            time.sleep(0.15)
            self.press("sp_choke_up", times=up, gap=0.06)
        self._wait_frames(2, 3.0)
        return True, f"штуцер выставлен на {target} % (от упора, без обратной связи)"


# ───────────────────────────────────────────────────────────── автоматика

class Rule:
    def __init__(self, rid, title, cond, cmd, args, cooldown=30.0, enabled=False):
        self.id, self.title, self.cond = rid, title, cond
        self.cmd, self.args, self.cooldown = cmd, args, cooldown
        self.enabled = enabled
        self.last = 0.0
        self.fired = 0


def _running_manual(a):
    return a.state("lamp_run") == "on" and a.state("lamp_auto") != "on"


def default_rules():
    def ge(tid, th):
        return lambda a: _running_manual(a) and (a.value(tid) or 0) >= th

    def le(tid, th):
        return lambda a: _running_manual(a) and a.value(tid) is not None and a.value(tid) <= th
    return [
        Rule("overheat", "ТЕМП. ПЭД ≥ 115 °C → задание ЧРП −1 Гц", ge("gauge_t", 115.0),
             "freq_dn", {"n": 2}),
        Rule("overcurrent", "ТОК ПЭД ≥ 42 А → задание ЧРП −1 Гц", ge("gauge_i", 42.0),
             "freq_dn", {"n": 2}),
        Rule("low_subm", "ПОГРУЖЕНИЕ ≤ 100 м → задание ЧРП −1 Гц", le("bar_sub", 100.0),
             "freq_dn", {"n": 2}),
    ]


class Automation:
    """Правила «условие на прочитанном → команда через HID». По умолчанию
    выключены: включаются на сайте. Между срабатываниями правила — пауза,
    чтобы параметр успел отреагировать на прошлое вмешательство."""

    def __init__(self, ctl, agg, rules=None):
        self.ctl, self.agg = ctl, agg
        self.rules = rules or default_rules()

    def set_enabled(self, rid, on):
        for r in self.rules:
            if r.id == rid:
                r.enabled = bool(on)
                return True
        return False

    def step(self):
        now = time.time()
        for r in self.rules:
            if not r.enabled or now - r.last < r.cooldown or self.ctl.busy():
                continue
            try:
                hit = r.cond(self.agg)
            except Exception:
                hit = False
            if hit:
                r.last = now
                r.fired += 1
                self.ctl.submit(r.cmd, r.args, source=f"автоматика: {r.title}")

    def describe(self):
        return [{"id": r.id, "title": r.title, "enabled": r.enabled, "fired": r.fired,
                 "cooldown": r.cooldown} for r in self.rules]
