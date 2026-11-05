# -*- coding: utf-8 -*-
"""Автоматический режим системы: жизнеспособность и работоспособность.

Режим задаётся для каждой системы: «ручной» (как раньше — технологические
действия только по кнопке оператора) или «автомат».

Жизнеспособность — сквозная проверка цепочки, на которой держится зрение:
ВМ → гостевая ОС → щит открыт → видео («HDMI») идёт → кадры свежие → зрение
читает → точность → щит виден целиком → HID-канал отвечает → канал сверки.
У каждой проверки — ремонт (запустить ВМ, перезапустить щит, «HDMI»-службу,
сбросить кэш, Esc, переподключить сверку). Её можно вызвать кнопкой в любом
режиме; в автомате она идёт сама раз в минуту и чинит то, что сломалось.

Работоспособность — в автомате система сама выполняет технологические
плейбуки (авария → ремонт, сброс, пуск; насос стоит → пуск; АСПО → промывка…):

  * каждый шаг-действие — нажатие на щите через HID с подтверждением по картинке;
  * шаг не подтвердился — плейбук останавливается и откладывается на 5 минут;
  * две неудачи одного плейбука за 30 минут — «нужен оператор»: плейбук больше
    сам не выполняется, пока оператор не снимет эскалацию;
  * после успешного плейбука — пауза 3 минуты на этот плейбук, чтобы щит пришёл
    в новое состояние, а не «дёргался».
"""

import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import fleet

HERE = Path(__file__).resolve().parent
STATE = HERE / "autopilot.json"
SEV_RANK = {"critical": 0, "serious": 1, "warning": 2, "info": 3}


class Autopilot(threading.Thread):
    VITALITY_EVERY = 60
    PB_COOLDOWN_OK = 180
    PB_COOLDOWN_FAIL = 300
    ESCALATE_AFTER = 2
    ESCALATE_WINDOW = 1800
    REPAIR_COOLDOWN = 120

    def __init__(self, F):
        super().__init__(daemon=True)
        self.F = F
        try:
            saved = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
        self.mode = {sid: saved.get(sid, "manual") for sid in F.order}
        self.log = defaultdict(lambda: deque(maxlen=60))
        self.vital = {}
        self.t_vital = {}
        self.pb_cool = {}                      # (sid, pb) → до какого времени не трогать
        self.fails = defaultdict(list)         # (sid, pb) → времена неудач
        self.escalated = {}                    # (sid, pb) → {t, why}
        self.repair_cool = {}
        self.running = {}                      # sid → id выполняемого плейбука
        self._ssh = {}                         # sid → (t, ok, chromium)

    # ── режим ────────────────────────────────────────────────────────
    def set_mode(self, sid, mode):
        if mode not in ("manual", "auto"):
            raise ValueError("режим — manual или auto")
        self.mode[sid] = mode
        if mode == "auto":
            self.F.autoheal[sid] = True        # исправления распознавания — тоже сами
        try:
            STATE.write_text(json.dumps(self.mode, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
        self.say(sid, "режим: АВТОМАТ — система сама держит щит в работе" if mode == "auto"
                 else "режим: ручной — технологические действия по кнопке оператора", "info")

    def say(self, sid, text, status="info"):
        self.log[sid].appendleft({"t": time.strftime("%H:%M:%S"), "text": text, "status": status})

    def clear_escalation(self, sid, pb=None):
        for k in [k for k in self.escalated if k[0] == sid and (pb is None or k[1] == pb)]:
            del self.escalated[k]
            self.fails.pop(k, None)
            self.pb_cool.pop(k, None)
        self.say(sid, "эскалация снята оператором — автомат снова может выполнять плейбук", "info")

    # ── жизнеспособность ─────────────────────────────────────────────
    def _guest(self, s):
        t, ok, chrome = self._ssh.get(s["id"], (0, None, None))
        if time.time() - t < 30:
            return ok, chrome
        code, out = fleet.ssh(s, "pgrep -c chromium || true", timeout=8)
        ok = code == 0
        try:
            chrome = int(out.split()[-1]) if ok and out else 0
        except ValueError:
            chrome = 0
        self._ssh[s["id"]] = (time.time(), ok, chrome)
        return ok, chrome

    def vitality(self, sid):
        s, w = self.F.systems[sid], self.F.workers[sid]
        st = w.snapshot()
        checks = []

        def add(cid, text, ok, detail, repair=None, warn=False):
            checks.append({"id": cid, "text": text, "status": "ok" if ok else ("warn" if warn else "fail"),
                           "detail": detail, "repair": None if ok else repair})

        vm_pid = fleet.vm_pid(s)
        add("vm", "ВМ запущена", bool(vm_pid), f"QEMU pid {vm_pid}" if vm_pid else "процесса QEMU нет", "start_vm")
        gok, chrome = self._guest(s) if vm_pid else (False, 0)
        add("guest", "Гостевая ОС отвечает", bool(gok), "ssh — ок" if gok else "ssh не отвечает", "start_vm" if not vm_pid else None)
        add("panel", "Щит открыт в ВМ", bool(chrome), f"Chromium — процессов {chrome}" if chrome else "Chromium не запущен",
            "restart_panel" if gok else None)
        ss = fleet.stream_stats(s) or {}
        vfps = ss.get("jpeg_fps") or 0
        add("video", "Видео («HDMI») идёт", vfps > 0, f"{vfps} к/с, кадр {ss.get('jpeg_kb', '?')} КБ" if vfps else "поток не отвечает",
            "restart_vmhdmi")
        age = (st.get("timing") or {}).get("age_ms")
        fresh = bool(st.get("ready")) and age is not None and age < 3000
        add("fresh", "Кадры свежие", fresh, f"от экрана до результата {age:.0f} мс" if age is not None else "результата нет",
            "restart_vmhdmi" if vfps == 0 else "flush")
        fps = st.get("fps") or 0
        kind = st.get("kind", "vision")
        add("vision", "Зрение читает", fps >= 1 or kind == "onboarding", f"{fps:.1f} к/с" if fps else "кадров нет", "flush")
        r = st.get("rolling") or {}
        acc = 100 * r["ok"] / r["total"] if r.get("total") else None
        if acc is not None:
            add("acc", "Точность распознавания ≥ 95 %", acc >= 95, f"{acc:.1f} % за 60 кадров", "flush", warn=acc >= 85)
        scene = (st.get("health") or {}).get("scene")
        if scene:
            add("scene", "Щит виден целиком", scene == "ok",
                "ничего не закрывает" if scene == "ok" else ("часть щита закрыта окном" if scene == "obstructed" else "щита на экране нет"),
                "esc" if scene == "obstructed" else "restart_panel")
        try:
            t0 = time.perf_counter()
            self.F._ctl(sid).call("ping", t=t0)
            add("hid", "HID-канал («KVM») отвечает", True, f"ответ за {(time.perf_counter() - t0) * 1000:.0f} мс")
        except Exception as e:
            add("hid", "HID-канал («KVM») отвечает", False, f"{type(e).__name__}: {e}", "restart_vmhdmi")
        if hasattr(w, "truth_down"):
            add("truth", "Канал сверки (стенд) работает", not w.truth_down,
                "истина модели читается" if not w.truth_down else "DevTools не отвечает", "reconnect_truth", warn=True)
        jobs = [j for j in self.F.jobs[sid] if j.get("action") == "press" and j["status"] != "выполняется"]
        if jobs:
            last = jobs[-1]
            add("press", "Последнее нажатие на щите подтверждено", last["status"] == "выполнено",
                f"{last['t']} · {last['title']} — {last['status']}", None, warn=True)
        ok = sum(c["status"] == "ok" for c in checks)
        v = {"t": time.strftime("%H:%M:%S"), "ts": time.time(), "ok": ok, "total": len(checks), "checks": checks,
             "verdict": "жизнеспособна" if all(c["status"] != "fail" for c in checks) else "нужен ремонт"}
        self.vital[sid] = v
        self.t_vital[sid] = time.time()
        return v

    def _repair(self, sid, v):
        for c in v["checks"]:
            if c["status"] != "fail" or not c.get("repair"):
                continue
            key = (sid, c["id"])
            if time.time() - self.repair_cool.get(key, 0) < self.REPAIR_COOLDOWN:
                continue
            self.repair_cool[key] = time.time()
            act = c["repair"]
            args = {}
            if act == "esc":
                act, args = "key", {"key": "Escape"}
            self.say(sid, f"жизнеспособность: «{c['text']}» — нет ({c['detail']}); ремонт: {act}", "run")
            self.F.run(sid, act, args, step=f"vital.{c['id']}", auto=True, title=f"автомат, ремонт: {c['text']}")
            return True                         # по одному ремонту за раз
        return False

    # ── работоспособность: технологические плейбуки ─────────────────
    def _candidates(self, sid):
        out = []
        for p in self.F.playbooks(sid):
            if p.get("kind") != "plant":
                continue
            if p["severity"] not in SEV_RANK or (p["severity"] == "info" and p["id"] != "plant.stopped"):
                continue
            if not any(st["kind"] == "action" and st.get("available") for st in p["steps"]):
                continue
            key = (sid, p["id"])
            if key in self.escalated or time.time() < self.pb_cool.get(key, 0):
                continue
            out.append(p)
        out.sort(key=lambda p: SEV_RANK[p["severity"]])
        return out

    def _wait_job(self, job, timeout=150):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if job["status"] != "выполняется":
                return job
            time.sleep(0.5)
        return job

    def _run_pb(self, sid, p):
        import playbooks as PB
        key = (sid, p["id"])
        self.running[sid] = p["id"]
        self.say(sid, f"плейбук «{p['title']}» — выполняю сам. Почему: {p['why']}", "run")
        ok = True
        try:
            for st in p["steps"]:
                if st["kind"] == "info":
                    self.say(sid, f"· {st['text']}", "info")
                    continue
                if st["kind"] == "check":
                    w = self.F.workers[sid]
                    seen = None
                    for _ in range(40):
                        seen = PB.View(w.snapshot(), getattr(w, "inventory", ())).check(st["expect"])
                        if seen:
                            break
                        time.sleep(0.5)
                    self.say(sid, f"· проверка «{st['text']}»: " + ("да, видно на щите" if seen else "пока нет — щит ещё меняется"),
                             "ok" if seen else "info")
                    continue
                if st["kind"] not in ("action", "fix") or not st.get("available"):
                    continue
                job = self.F.run(sid, st["action"]["name"], st["action"]["args"], step=st["id"], expect=st.get("expect"),
                                 wait=st.get("wait", 25), auto=True, title=f"автомат: {st['text']}")
                j = self._wait_job(job)
                if j["status"] != "выполнено":
                    self.say(sid, f"· шаг «{st['text']}» — {j['status']}: {j['msg']}. Плейбук остановлен.", "fail")
                    ok = False
                    break
                self.say(sid, f"· {st['text']} — {j['msg']}", "ok")
        except Exception as e:
            ok = False
            self.say(sid, f"ошибка автомата: {type(e).__name__}: {e}", "fail")
        finally:
            self.running.pop(sid, None)
        if ok:
            self.pb_cool[key] = time.time() + self.PB_COOLDOWN_OK
            self.say(sid, f"плейбук «{p['title']}» выполнен, все шаги подтверждены по картинке", "ok")
            return
        now = time.time()
        self.fails[key] = [t for t in self.fails[key] if now - t < self.ESCALATE_WINDOW] + [now]
        self.pb_cool[key] = now + self.PB_COOLDOWN_FAIL
        if len(self.fails[key]) >= self.ESCALATE_AFTER:
            self.escalated[key] = {"t": time.strftime("%H:%M:%S"), "title": p["title"],
                                   "why": f"{len(self.fails[key])} неудачи за {self.ESCALATE_WINDOW // 60} мин"}
            self.say(sid, f"НУЖЕН ОПЕРАТОР: «{p['title']}» не удаётся выполнить автоматически — "
                          "автомат его больше не трогает, пока оператор не снимет эскалацию", "fail")
        else:
            self.say(sid, f"плейбук «{p['title']}» отложен на {self.PB_COOLDOWN_FAIL // 60} мин", "info")

    # ── цикл ─────────────────────────────────────────────────────────
    def run(self):
        while True:
            time.sleep(2)
            for sid in self.F.order:
                if self.mode.get(sid) != "auto" or sid in self.running:
                    continue
                if getattr(self.F, "demo", None) and self.F.demo.get("running") and self.F.demo.get("sys") == sid:
                    continue
                if any(j["status"] == "выполняется" for j in self.F.jobs[sid]):
                    continue
                try:
                    if time.time() - self.t_vital.get(sid, 0) > self.VITALITY_EVERY:
                        v = self.vitality(sid)
                        if v["verdict"] != "жизнеспособна":
                            if self._repair(sid, v):
                                continue
                    v = self.vital.get(sid)
                    # пока цепочка сломана (нет видео, ВМ), технологию не трогаем — не видно результата
                    if v and any(c["status"] == "fail" and c["id"] in ("vm", "video", "fresh", "hid", "scene")
                                 for c in v["checks"]):
                        continue
                    cands = self._candidates(sid)
                    if cands:
                        threading.Thread(target=self._run_pb, args=(sid, cands[0]), daemon=True).start()
                        self.running[sid] = cands[0]["id"]
                except Exception as e:
                    self.say(sid, f"ошибка автомата: {type(e).__name__}: {e}", "fail")

    def snapshot(self, sid):
        return {"mode": self.mode.get(sid, "manual"), "running": self.running.get(sid),
                "vitality": self.vital.get(sid), "log": list(self.log[sid])[:40],
                "escalated": [dict(v, playbook=k[1]) for k, v in self.escalated.items() if k[0] == sid],
                "cooldown": {k[1]: round(v - time.time()) for k, v in self.pb_cool.items() if k[0] == sid and v > time.time()}}
