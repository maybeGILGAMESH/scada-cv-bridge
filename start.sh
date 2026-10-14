#!/bin/bash
# Поднять стенд в фоне: ВМ со щитом (без окна) + vmhdmi (экран ВМ → MJPEG, ввод → ВМ).
#   ./start.sh      затем по желанию ./vmview.sh — окно с экраном ВМ (F11 — полный экран)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"; [ -x "$PY" ] || PY=python3
BR=bridge; [ -d "$DIR/bridge" ] || BR=extracted/scada-cv-bridge-main/bridge
"$DIR/vm/run-vm.sh"
if pgrep -f "[v]mhdmi.py" >/dev/null; then
  echo "vmhdmi уже работает"
else
  nohup "$PY" -u "$DIR/vm/vmhdmi.py" "$@" > "$DIR/vm/vmhdmi.log" 2>&1 &
  for _ in $(seq 120); do curl -sf http://127.0.0.1:8781/stats >/dev/null && break; sleep 1; done
  curl -sf http://127.0.0.1:8781/stats >/dev/null || { echo "vmhdmi не поднялся, см. vm/vmhdmi.log"; exit 1; }
  echo "vmhdmi запущен (лог vm/vmhdmi.log)"
fi
cat <<TXT
  видео (MJPEG, как PiKVM) : http://127.0.0.1:8781/stream   страница: http://127.0.0.1:8781/
  управление (как emulator): tcp://127.0.0.1:8790
  окно с экраном ВМ        : ./vmview.sh   (F11 — полный экран)
  распознавание            : $PY $BR/receiver.py \\
                               --source http://127.0.0.1:8781/stream --control 127.0.0.1:8790 --no-assist --no-normalize
  остановить всё           : ./stop.sh
TXT
