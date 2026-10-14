#!/bin/bash
# Дашборд «что видит система»: поднимает стенд (если не поднят), дашборд и открывает его в браузере.
#   ./cvdash.sh            http://127.0.0.1:8775/
#   ./cvdash.sh --no-open  не открывать браузер
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"; [ -x "$PY" ] || PY=python3
"$DIR/start.sh" >/dev/null
if ! pgrep -f "[v]m/cvdash.py" >/dev/null; then
  nohup "$PY" -u "$DIR/vm/cvdash.py" > "$DIR/vm/cvdash.log" 2>&1 &
  for _ in $(seq 60); do curl -sf http://127.0.0.1:8775/api/vision >/dev/null && break; sleep 1; done
fi
echo "Дашборд: http://localhost:8775/   (лог vm/cvdash.log, остановить: ./stop.sh)"
[ "${1:-}" = --no-open ] || xdg-open http://localhost:8775/ >/dev/null 2>&1 || true
