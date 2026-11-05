#!/bin/bash
# Остановить стенд: дашборд, окно просмотра, все системы (vmhdmi + ВМ).
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"; [ -x "$PY" ] || PY=python3
pkill -f "[v]m/cvdash.py" && echo "дашборд остановлен"
pkill -f "[v]m/vmview.py" && echo "окно просмотра закрыто"
"$PY" "$DIR/vm/fleet.py" down "$@"
