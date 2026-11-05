#!/bin/bash
# Поднять стенд в фоне: все системы из vm/systems.json — ВМ со щитом (без окна) + vmhdmi каждой.
#   ./start.sh              все системы;  ./start.sh well1247  — только одну
#   затем ./cvdash.sh — дашборд, ./vmview.sh — окно с экраном основной ВМ (F11 — полный экран)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"; [ -x "$PY" ] || PY=python3
BR=bridge; [ -d "$DIR/bridge" ] || BR=extracted/scada-cv-bridge-main/bridge
# все системы из vm/systems.json: ВМ (основная — run-vm.sh, остальные — vm/instances/) + «HDMI»-служба каждой
"$PY" "$DIR/vm/fleet.py" up "$@"
cat <<TXT
  видео (MJPEG, как PiKVM) : http://127.0.0.1:8781/stream   страница: http://127.0.0.1:8781/
  управление (как emulator): tcp://127.0.0.1:8790
  окно с экраном ВМ        : ./vmview.sh   (F11 — полный экран)
  распознавание            : $PY $BR/receiver.py \\
                               --source http://127.0.0.1:8781/stream --control 127.0.0.1:8790 --no-assist --no-normalize
  остановить всё           : ./stop.sh
TXT
