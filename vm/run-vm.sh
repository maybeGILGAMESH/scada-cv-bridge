#!/bin/bash
# ВМ со щитом.
#   ./run-vm.sh            — запустить в фоне, без окна (экран: VNC 127.0.0.1:5900)
#   ./run-vm.sh --window   — запустить с окном QEMU (Ctrl+Alt+F полный экран, Ctrl+Alt+G отпустить мышь)
#   ./run-vm.sh stop       — выключить (ACPI, как кнопка питания)
#   ./run-vm.sh status     — запущена ли
# Смотреть и управлять, когда ВМ в фоне: ../vmview.sh   ssh: ssh -i vm/ssh_key -p 2222 scada@127.0.0.1
set -euo pipefail
cd "$(dirname "$0")"
# папка со щитом: в репозитории — на уровень выше vm/, в локальной раскладке — extracted/
if [ -f ../neft-hmi-cv-fullscreen.html ]; then SHARE="$(cd .. && pwd)"; else SHARE="$(cd ../extracted/scada-cv-bridge-main && pwd)"; fi
PID=vm.pid

running() { [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; }

case "${1:-}" in
  status)
    if running; then echo "ВМ запущена (pid $(cat $PID))"; else echo "ВМ не запущена"; fi; exit 0 ;;
  stop)
    running || { echo "ВМ не запущена"; exit 0; }
    printf '{"execute":"qmp_capabilities"}\n{"execute":"system_powerdown"}\n' \
      | timeout 3 python3 -I -c 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect("qmp.sock"); s.sendall(sys.stdin.buffer.read()); s.recv(4096)' || true
    for _ in $(seq 60); do running || { echo "ВМ выключена"; exit 0; }; sleep 1; done
    echo "ВМ не выключилась за 60 с — гашу"; kill "$(cat $PID)"; exit 0 ;;
esac

running && { echo "ВМ уже запущена (pid $(cat $PID))"; exit 0; }

# терминал VS Code из snap подсовывает свои GTK/GIO-библиотеки — qemu на них падает
if [ -n "${SNAP:-}" ]; then
  unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GDK_PIXBUF_MODULE_FILE \
        GDK_PIXBUF_MODULEDIR GSETTINGS_SCHEMA_DIR LOCPATH
  [ -n "${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-}" ] && export XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG"
  [ -n "${XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG:-}" ] && export XDG_CONFIG_DIRS="$XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG"
fi

if [ "${1:-}" = --window ]; then
  DISPLAY_ARGS=(-display gtk,zoom-to-fit=on)
else
  DISPLAY_ARGS=(-display none -daemonize)
fi

qemu-system-x86_64 -name "SCADA VM" \
  -enable-kvm -machine q35 -cpu host -smp 4 -m 4096 \
  -drive file=scada-vm.qcow2,if=virtio \
  -drive file=seed.iso,if=virtio,format=raw,readonly=on \
  -netdev user,id=n0,hostfwd=tcp:127.0.0.1:2222-:22 -device virtio-net-pci,netdev=n0 \
  -virtfs local,path="$SHARE",mount_tag=scada,security_model=none,readonly=on \
  -device virtio-vga,xres=1920,yres=1080 \
  -device qemu-xhci -device usb-tablet -device usb-kbd \
  -vnc 127.0.0.1:0 \
  -qmp unix:qmp.sock,server=on,wait=off \
  -pidfile $PID \
  -audio none \
  -serial file:serial.log \
  "${DISPLAY_ARGS[@]}"
running && echo "ВМ запущена в фоне (pid $(cat $PID)). Экран: VNC 127.0.0.1:5900"
