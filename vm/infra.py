# -*- coding: utf-8 -*-
"""«На чём работает»: хост, ВМ, гостевые системы, процессы, стек распознавания."""

import json
import os
import platform
import re
import subprocess
import threading
import time
from pathlib import Path

import fleet

HERE = Path(__file__).resolve().parent
TICK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def _run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def host_static():
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    osr = {}
    try:
        for l in open("/etc/os-release"):
            k, _, v = l.strip().partition("=")
            osr[k] = v.strip('"')
    except Exception:
        pass
    import cv2
    import numpy
    try:
        import websockets
        ws = websockets.__version__
    except Exception:
        ws = None
    tess = _run(["tesseract", "--version"]).splitlines()
    langs = sorted({p.stem for p in (HERE / "tessdata").glob("*.traineddata")}) or \
        _run(["tesseract", "--list-langs"]).splitlines()[1:]
    return {"cpu": cpu, "cores": os.cpu_count(), "os": osr.get("PRETTY_NAME"), "kernel": platform.release(),
            "qemu": (_run(["qemu-system-x86_64", "--version"]).splitlines() or [""])[0],
            "kvm": os.path.exists("/dev/kvm"), "python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": numpy.__version__, "websockets": ws, "tesseract": tess[0] if tess else None,
            "tess_langs": langs, "hostname": platform.node()}


def meminfo():
    m = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":")
        m[k] = int(v.split()[0]) * 1024
    return {"total_gb": round(m["MemTotal"] / 2**30, 1),
            "used_gb": round((m["MemTotal"] - m["MemAvailable"]) / 2**30, 1)}


class Proc:
    """CPU% и память процесса по /proc — между двумя замерами."""

    def __init__(self):
        self.prev = {}

    def stat(self, pid):
        if not pid:
            return None
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            cpu = (int(f[11]) + int(f[12])) / TICK
            rss = int(f[21]) * PAGE
            start = int(f[19]) / TICK
        except Exception:
            return None
        now = time.time()
        p = self.prev.get(pid)
        self.prev[pid] = (now, cpu)
        pct = None if not p else round(100 * (cpu - p[1]) / max(1e-3, now - p[0]), 1)
        up = float(open("/proc/uptime").read().split()[0]) - start
        return {"pid": pid, "cpu_pct": pct, "rss_mb": round(rss / 2**20), "uptime_s": int(up)}


GUEST_CMD = (". /etc/os-release; echo \"os=$PRETTY_NAME\"; echo kernel=$(uname -r); "
             "echo chromium=$(chromium --version 2>/dev/null); echo up=$(cut -d' ' -f1 /proc/uptime); "
             "echo load=$(cut -d' ' -f1-3 /proc/loadavg); echo mem=$(free -m | awk '/Mem/{print $3\"/\"$2}'); "
             "echo cpus=$(nproc); echo xorg=$(pgrep -c Xorg); echo chrome=$(pgrep -c chromium); "
             "echo mode=$(DISPLAY=:0 xrandr 2>/dev/null | awk '/\\*/{print $1; exit}'); echo page=$(pgrep -a chromium | grep -o 'file://[^ ]*' | head -1)")


class Infra(threading.Thread):
    def __init__(self, workers):
        super().__init__(daemon=True)
        self.workers = workers
        self.proc = Proc()
        self.static = host_static()
        self.guest = {}
        self.t_guest = {}
        self.state = {}
        self.lock = threading.Lock()

    def run(self):
        while True:
            try:
                self.tick()
            except Exception as e:
                with self.lock:
                    self.state["error"] = str(e)
            time.sleep(3)

    def _guest(self, s):
        if time.time() - self.t_guest.get(s["id"], 0) < 60:
            return
        self.t_guest[s["id"]] = time.time()
        code, out = fleet.ssh(s, GUEST_CMD, timeout=10)
        g = {"reachable": code == 0}
        for l in out.splitlines():
            k, _, v = l.partition("=")
            if k and v:
                g[k] = v
        self.guest[s["id"]] = g

    def tick(self):
        systems = []
        for s in fleet.registry():
            st = fleet.status(s)
            threading.Thread(target=self._guest, args=(s,), daemon=True).start()
            w = self.workers.get(s["id"])
            ws = w.snapshot() if w else {}
            prof = None
            if s.get("profile") and s.get("reader") == "dom":
                d = HERE / s["profile"]
                try:
                    pj = json.loads((d / "profile.json").read_text(encoding="utf-8"))
                    acc = [o.get("calib_acc") for o in pj["objects"] if o.get("calib_acc") is not None]
                    prof = {"path": f"vm/{s['profile']}", "templates": len(pj["objects"]), "created": pj.get("created"),
                            "kind": "со страницы щита", "glyphs": (d / "glyphs.npz").exists(), "log_tags": True,
                            "validation": pj.get("frames"), "reliable": sum(a >= 0.8 for a in acc), "checked": len(acc)}
                except Exception:
                    prof = {"path": f"vm/{s['profile']}", "error": "не читается"}
            elif s.get("profile"):
                d = HERE / s["profile"]
                try:
                    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                    prof = {"path": f"vm/{s['profile']}", "templates": len(man["templates"]),
                            "created": man.get("created"), "layout": man.get("layout_id"),
                            "glyphs": (d / "glyphs.npz").exists(),
                            "log_tags": (d / "log_tags.json").exists(),
                            "validation": len(list((d / "validation").rglob("*.png"))) if (d / "validation").exists() else 0}
                except Exception:
                    prof = {"path": f"vm/{s['profile']}", "error": "не читается"}
            disk = fleet.vm_dir(s) / ("scada-vm.qcow2" if s["vm"] == "main" else "disk.qcow2")
            systems.append({
                "id": s["id"], "name": s["name"], "vm": s["vm"], "page": s["page"], "ports": s["ports"],
                "vm_proc": self.proc.stat(st["vm_pid"]), "vmhdmi_proc": self.proc.stat(st["vmhdmi_pid"]),
                "vm_cfg": {"vcpu": 4 if s["vm"] == "main" else 2, "ram_gb": 4 if s["vm"] == "main" else round(s.get("ram_mb", 2048) / 1024, 1),
                           "disk": str(disk.relative_to(HERE.parent)) if disk.exists() else None,
                           "disk_mb": round(disk.stat().st_size / 2**20) if disk.exists() else None},
                "stream": fleet.stream_stats(s), "guest": self.guest.get(s["id"]),
                "vision": {"kind": ws.get("kind", "vision"), "fps": ws.get("fps"), "timing": ws.get("timing"),
                           "quality": ws.get("quality"), "active": getattr(w, "active", None)},
                "profile": prof})
        me = self.proc.stat(os.getpid())
        with self.lock:
            self.state = {"host": dict(self.static, **meminfo(),
                                       load=open("/proc/loadavg").read().split()[:3],
                                       uptime_h=round(float(open("/proc/uptime").read().split()[0]) / 3600, 1)),
                          "dashboard_proc": me, "systems": systems, "t": time.time()}

    def snapshot(self):
        with self.lock:
            return dict(self.state)
