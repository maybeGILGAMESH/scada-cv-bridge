# -*- coding: utf-8 -*-
"""Парк систем: несколько ВМ со щитами и «HDMI»-служба для каждой.

Реестр — vm/systems.json. Основная ВМ (vm=main) — та, что поднимает
run-vm.sh; остальные живут в vm/instances/<vm>/: свой диск поверх образа
Debian, свой cloud-init (какую страницу щита открыть), свои порты.

    python fleet.py up                 # поднять все системы (ВМ + vmhdmi)
    python fleet.py up well1248        # одну
    python fleet.py status             # JSON: ВМ, служба, поток
    python fleet.py down newsys        # выключить
    python fleet.py restart-panel ID   # перезапустить щит в ВМ (X + Chromium)

Те же функции вызывает дашборд, когда чинит систему.
"""

import json
import os
import shlex
import subprocess
import threading
import sys
import time
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "neft-hmi-cv.html").exists() else HERE.parent / "extracted" / "scada-cv-bridge-main"
BASE = HERE / "debian-13-generic-amd64.qcow2"
INST = HERE / "instances"
PY = str(HERE.parent / ".venv" / "bin" / "python") if (HERE.parent / ".venv" / "bin" / "python").exists() else sys.executable


def registry():
    return json.loads((HERE / "systems.json").read_text(encoding="utf-8"))["systems"]


def system(sid):
    for s in registry():
        if s["id"] == sid:
            return s
    raise KeyError(f"нет системы {sid}")


def vm_dir(s):
    return HERE if s["vm"] == "main" else INST / s["vm"]


def _alive(pidfile):
    try:
        pid = int(Path(pidfile).read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def vm_pid(s):
    return _alive(vm_dir(s) / "vm.pid")


def _cmdlines():
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                yield int(p.name), (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                pass


def vmhdmi_pid(s):
    port = s["ports"]["mjpeg"]
    for pid, cmd in _cmdlines():
        if "vmhdmi.py" in cmd and (f"--port {port}" in cmd or (port == 8781 and "--port " not in cmd)):
            return pid
    return None


def stream_stats(s, timeout=1.5):
    try:
        return json.load(urlopen(f"http://127.0.0.1:{s['ports']['mjpeg']}/stats", timeout=timeout))
    except Exception:
        return None


def ssh(s, cmd, timeout=15):
    """Команда в ВМ по ssh. → (код, вывод)"""
    args = ["ssh", "-q", "-i", str(HERE / "ssh_key"), "-p", str(s["ports"]["ssh"]),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=4", "-o", "BatchMode=yes", "scada@127.0.0.1", cmd]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "таймаут ssh"


class GuestPointer:
    """Мышь оператора внутри гостя (XTEST через xdotool) — постоянный ssh-канал.

    Абсолютные устройства QEMU (USB-планшет, vmmouse) Chrome в X11 считает пером
    (pointerType «pen»), а пером ползунок <input type=range> не тянется — кнопки
    при этом нажимаются. Поэтому перетаскивание оператора на «Системах» идёт как
    обычная мышь X прямо в госте. Нажатия плейбуков (зрение, KVM) — по-прежнему
    через VNC."""

    def __init__(self, s):
        self.s, self.p = s, None
        self.lock = threading.Lock()

    def _open(self):
        if self.p and self.p.poll() is None:
            return
        self.p = subprocess.Popen(
            ["ssh", "-q", "-i", str(HERE / "ssh_key"), "-p", str(self.s["ports"]["ssh"]),
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "ConnectTimeout=4", "-o", "BatchMode=yes", "scada@127.0.0.1",
             "export DISPLAY=:0; while read -r a; do xdotool $a; done"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(self, *cmds):
        with self.lock:
            for attempt in (0, 1):
                try:
                    self._open()
                    for c in cmds:
                        self.p.stdin.write(c + "\n")
                    self.p.stdin.flush()
                    return
                except (OSError, ValueError):
                    self.p = None
                    if attempt:
                        raise


# ───────────────────────────────────────────────────────────── ВМ

def _make_instance(s):
    d = vm_dir(s)
    d.mkdir(parents=True, exist_ok=True)
    disk = d / "disk.qcow2"
    if not disk.exists():
        if not BASE.exists():
            raise RuntimeError(f"нет образа {BASE} — запустите vm/setup-vm.sh")
        subprocess.run(["qemu-img", "create", "-q", "-f", "qcow2", "-F", "qcow2", "-b", str(BASE),
                        str(disk), "20G"], check=True)
        key = (HERE / "ssh_key.pub").read_text().strip()
        ud = (HERE / "user-data.tmpl").read_text(encoding="utf-8")
        ud = ud.replace("__SSH_KEY__", key).replace("__HOST__", f"scada-{s['vm']}").replace("__PAGE__", s["page"])
        (d / "user-data").write_text(ud, encoding="utf-8")
        (d / "meta-data").write_text(f"instance-id: scada-{s['vm']}-1\nlocal-hostname: scada-{s['vm']}\n")
        subprocess.run(["cloud-localds", str(d / "seed.iso"), str(d / "user-data"), str(d / "meta-data")], check=True)


def _clean_env():
    env = dict(os.environ)
    if env.get("SNAP"):                         # терминал VS Code из snap: его GTK ломает qemu
        for k in ("GTK_PATH", "GTK_EXE_PREFIX", "GTK_IM_MODULE_FILE", "GIO_MODULE_DIR", "GDK_PIXBUF_MODULE_FILE",
                  "GDK_PIXBUF_MODULEDIR", "GSETTINGS_SCHEMA_DIR", "LOCPATH"):
            env.pop(k, None)
    return env


def start_vm(s):
    if vm_pid(s):
        return "уже запущена"
    if s["vm"] == "main":
        r = subprocess.run([str(HERE / "run-vm.sh")], capture_output=True, text=True, env=_clean_env())
        return (r.stdout + r.stderr).strip()
    _make_instance(s)
    d, p = vm_dir(s), s["ports"]
    cmd = ["qemu-system-x86_64", "-name", f"SCADA {s['vm']}", "-enable-kvm", "-machine", "q35", "-cpu", "host",
           "-smp", "2", "-m", str(s.get("ram_mb", 2048)),
           "-drive", f"file={d / 'disk.qcow2'},if=virtio",
           "-drive", f"file={d / 'seed.iso'},if=virtio,format=raw,readonly=on",
           "-netdev", f"user,id=n0,hostfwd=tcp:127.0.0.1:{p['ssh']}-:22", "-device", "virtio-net-pci,netdev=n0",
           "-virtfs", f"local,path={ROOT},mount_tag=scada,security_model=none,readonly=on",
           "-device", "virtio-vga,xres=1920,yres=1080",
           "-device", "qemu-xhci", "-device", "usb-tablet", "-device", "usb-kbd",
           "-vnc", f"127.0.0.1:{p['vnc'] - 5900}",
           "-qmp", f"unix:{d / 'qmp.sock'},server=on,wait=off",
           "-pidfile", str(d / "vm.pid"), "-audio", "none",
           "-serial", f"file:{d / 'serial.log'}", "-display", "none", "-daemonize"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_clean_env())
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return f"ВМ {s['vm']} запущена (pid {vm_pid(s)})"


def stop_vm(s, wait=60):
    pid = vm_pid(s)
    if not pid:
        return "не запущена"
    import socket
    try:
        c = socket.socket(socket.AF_UNIX)
        c.settimeout(3)
        c.connect(str(vm_dir(s) / "qmp.sock"))
        c.sendall(b'{"execute":"qmp_capabilities"}\n{"execute":"system_powerdown"}\n')
        c.recv(4096)
        c.close()
    except OSError:
        pass
    for _ in range(wait):
        if not vm_pid(s):
            return "выключена"
        time.sleep(1)
    os.kill(pid, 15)
    return "погашена (не ответила на выключение)"


# ───────────────────────────────────────────────────────────── «HDMI»-служба

def start_vmhdmi(s):
    if vmhdmi_pid(s):
        return "уже работает"
    p = s["ports"]
    log = open(vm_dir(s) / "vmhdmi.log", "w")
    subprocess.Popen([PY, "-u", str(HERE / "vmhdmi.py"), "--vnc", f"127.0.0.1:{p['vnc']}",
                      "--port", str(p["mjpeg"]), "--control-port", str(p["control"]),
                      "--ssh-port", str(p["ssh"]), "--cdp-port", str(p["cdp"])],
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(40):
        if stream_stats(s):
            return f"vmhdmi :{p['mjpeg']} работает"
        time.sleep(0.5)
    return "vmhdmi запущен, поток ещё не отвечает"


def stop_vmhdmi(s):
    pid = vmhdmi_pid(s)
    if pid:
        os.kill(pid, 15)
        time.sleep(0.5)
        return "остановлен"
    return "не работал"


def restart_vmhdmi(s):
    stop_vmhdmi(s)
    time.sleep(0.5)
    return start_vmhdmi(s)


def restart_panel(s):
    """Щит в ВМ закрыли или он завис: закрыть Chromium и перезапустить сеанс
    автологина — .bash_profile снова поднимет X и щит."""
    code, out = ssh(s, "pkill -x chromium; sleep 1; sudo systemctl restart getty@tty1; echo ok")
    return "щит в ВМ перезапущен" if code == 0 else f"ssh: {out or code}"


def up(s):
    msg = [start_vm(s)]
    msg.append(start_vmhdmi(s))
    return "; ".join(msg)


def status(s):
    st = stream_stats(s)
    return {"id": s["id"], "vm_pid": vm_pid(s), "vmhdmi_pid": vmhdmi_pid(s),
            "stream": bool(st), "stream_fps": (st or {}).get("jpeg_fps"),
            "vm_dir": str(vm_dir(s))}


def provisioned(s):
    """cloud-init закончил и щит открыт (для новых ВМ первый запуск — минуты)."""
    if s["vm"] == "main":
        return True
    log = vm_dir(s) / "serial.log"
    try:
        t = log.read_text(errors="replace")
    except OSError:
        return False
    return t.count("Reached target") > 0 and "Cloud-init v." in t and "finished" in t


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, ids = sys.argv[1], sys.argv[2:]
    systems = [s for s in registry() if not ids or s["id"] in ids]
    for s in systems:
        if cmd == "up":
            print(f"{s['id']}: {up(s)}", flush=True)
        elif cmd == "down":
            print(f"{s['id']}: vmhdmi {stop_vmhdmi(s)}; ВМ {stop_vm(s)}", flush=True)
        elif cmd == "status":
            print(json.dumps(status(s), ensure_ascii=False))
        elif cmd == "restart-panel":
            print(f"{s['id']}: {restart_panel(s)}")
        elif cmd == "restart-vmhdmi":
            print(f"{s['id']}: {restart_vmhdmi(s)}")
        else:
            sys.exit(f"неизвестная команда {cmd}")


if __name__ == "__main__":
    main()
