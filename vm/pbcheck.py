# -*- coding: utf-8 -*-
"""Проверка плейбуков на живых системах — тем же путём, что кнопки дашборда.

Для каждой системы и сценария:
  1. тестовым каналом (DevTools страницы щита) создаётся ситуация: насос
     остановлен / пробой изоляции / отложения АСПО;
  2. ждём, пока распознавание само увидит её на экране и дашборд соберёт
     плейбук (время до плейбука — в отчёт);
  3. шаги-действия выполняются через POST /api/fix — как «Выполнить по шагам»:
     нажатие по месту на кадре через HID, подтверждение по картинке;
  4. итог сверяется с истиной модели щита (а не с распознаванием).

  python vm/pbcheck.py                       все системы, все сценарии
  python vm/pbcheck.py --sys newsys --scen trip
Отчёт: vm/results/pbcheck.md. Дашборд (./cvdash.sh) должен работать.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

HERE = Path(__file__).resolve().parent
DASH = "http://127.0.0.1:8775"

# ситуация на щите: JS для обоих щитов (новый — syncUI, старый — sync)
SYNC = "(typeof syncUI === 'function' ? syncUI : sync)()"
SCEN = {
    "stopped": {
        "title": "Насос остановлен → «Пуск»",
        "setup": f"() => {{ if (S.tripped) {{ S.tripped = false; S.tripReason = ''; }} if (!S.run) cmdStart(); cmdStop(); {SYNC}; }}",
        "want": lambda p: p["id"] == "plant.stopped",
        "done": "S.run && !S.tripped",
        "done_text": "насос в работе",
    },
    "trip": {
        "title": "Пробой изоляции → ремонт, сброс, пуск",
        "setup": f"() => {{ if (!S.run && !S.tripped) cmdStart(); S.iso = 4; {SYNC}; }}",
        "want": lambda p: p["kind"] == "plant" and p["severity"] == "critical",
        "done": "S.run && !S.tripped && S.iso >= 100",
        "done_text": "насос в работе, защиты нет, изоляция ≥ 100 кОм",
    },
    "asp": {
        "title": "Отложения АСПО → промывка, ингибитор",
        "setup": f"() => {{ if (!S.run && !S.tripped) cmdStart(); S.clog = 4.5; if (typeof ui.inh !== 'undefined') ui.inh = false; {SYNC}; }}",
        "want": lambda p: p["kind"] == "plant" and p["severity"] in ("warning", "serious") and
                any(s["kind"] == "action" for s in p["steps"]),
        "done": "S.run && !S.tripped && ui.inh",
        "done_text": "насос в работе, дозатор ингибитора включён",
    },
}


def http(path, body=None):
    req = urllib.request.Request(DASH + path, data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST")
    return json.load(urllib.request.urlopen(req, timeout=15))


class Page:
    """Runtime.evaluate в странице щита через туннель DevTools дашборда."""

    def __init__(self, port):
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=5))
        self.ws = connect(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"], max_size=2**24)
        self.n = 0

    def __call__(self, expr):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": "Runtime.evaluate",
                                 "params": {"expression": expr, "returnByValue": True}}))
        while True:
            r = json.loads(self.ws.recv(timeout=10))
            if r.get("id") == self.n:
                res = r["result"]
                if "exceptionDetails" in res:
                    raise RuntimeError(res["exceptionDetails"].get("text"))
                return res["result"].get("value")


def wait(fn, timeout, step=1.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(step)
    return None


def playbooks(sid):
    return http(f"/api/playbooks?sys={sid}")


def run_case(s, name, log):
    sc, sid = SCEN[name], s["id"]
    page = Page(s["ports"]["cdp_dash"])
    res = {"sys": sid, "scen": name, "title": sc["title"], "steps": [], "ok": False}
    page(f"({sc['setup']})()")
    t0 = time.time()
    pb = wait(lambda: next((p for p in playbooks(sid)["playbooks"] if sc["want"](p)), None), 90)
    if not pb:
        res["note"] = "плейбук не появился за 90 с: " + ", ".join(p["title"] for p in playbooks(sid)["playbooks"])
        return res
    res["pb"], res["t_pb"] = pb["title"], round(time.time() - t0, 1)
    log(f"  плейбук «{pb['title']}» через {res['t_pb']} с")
    for st in pb["steps"]:
        if st["kind"] not in ("action", "fix"):
            continue
        if not st.get("available"):
            res["steps"].append({"text": st["text"], "status": "недоступен"})
            continue
        job = http("/api/fix", {"sys": sid, "action": st["action"]["name"], "args": st["action"]["args"],
                                "step": st["id"], "expect": st.get("expect"), "wait": st.get("wait", 20),
                                "title": "проверка: " + st["text"]})["job"]
        j = wait(lambda: next((x for x in playbooks(sid)["jobs"] if x["id"] == job["id"] and x["status"] != "выполняется"), None),
                 120, 0.5)
        j = j or {"status": "нет ответа", "msg": ""}
        res["steps"].append({"text": st["text"], "status": j["status"], "msg": j["msg"]})
        log(f"  · {st['text']}: {j['status']} — {j['msg']}")
        if j["status"] != "выполнено":
            break
    res["truth_ok"] = bool(wait(lambda: page(sc["done"]), 30))
    res["truth"] = page("({run: S.run, tripped: S.tripped, reason: S.tripReason, iso: Math.round(S.iso), inh: !!ui.inh})")
    left = [p["title"] for p in playbooks(sid)["playbooks"] if sc["want"](p)] if name != "asp" else []
    res["left"] = left
    res["ok"] = res["truth_ok"] and all(x["status"] == "выполнено" for x in res["steps"]) and not left
    res["t_total"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", help="через запятую; по умолчанию все")
    ap.add_argument("--scen", default="stopped,trip,asp")
    a = ap.parse_args()
    reg = json.loads((HERE / "systems.json").read_text(encoding="utf-8"))["systems"]
    want = set(a.sys.split(",")) if a.sys else None
    out = []
    for s in reg:
        if want and s["id"] not in want:
            continue
        for name in a.scen.split(","):
            print(f"{s['id']} · {SCEN[name]['title']}", flush=True)
            try:
                r = run_case(s, name, lambda m: print(m, flush=True))
            except Exception as e:
                r = {"sys": s["id"], "scen": name, "title": SCEN[name]["title"], "ok": False, "steps": [],
                     "note": f"{type(e).__name__}: {e}"}
            print(f"  → {'ДА' if r['ok'] else 'НЕТ'} {r.get('note', '')} истина: {r.get('truth')}", flush=True)
            out.append(r)
    lines = ["# Проверка плейбуков на живых системах", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · шаги через /api/fix (HID по месту на кадре, подтверждение по картинке), итог — по истине модели", "",
             "| система | сценарий | плейбук (через, с) | шаги | итог по истине | годен |", "|---|---|---|---|---|---|"]
    for r in out:
        steps = "<br>".join(f"{x['text']}: {x['status']}" for x in r["steps"]) or "—"
        lines.append(f"| {r['sys']} | {r['title']} | {r.get('pb', r.get('note', '—'))} ({r.get('t_pb', '—')}) | {steps} | "
                     f"{json.dumps(r.get('truth'), ensure_ascii=False)} | {'✓' if r['ok'] else '✗'} |")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "pbcheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{sum(r['ok'] for r in out)} из {len(out)} сценариев прошли · отчёт vm/results/pbcheck.md")


if __name__ == "__main__":
    main()
