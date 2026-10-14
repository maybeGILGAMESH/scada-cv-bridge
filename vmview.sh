#!/bin/bash
# Окно с экраном ВМ (то, что уходит по «HDMI»). F11 — полный экран, Esc — выйти из него.
#   ./vmview.sh [--fullscreen] [--view-only]
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"; [ -x "$PY" ] || PY=python3
if [ -n "${SNAP:-}" ]; then        # терминал VS Code из snap: убрать его GTK/Qt-окружение
  unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GDK_PIXBUF_MODULE_FILE \
        GDK_PIXBUF_MODULEDIR GSETTINGS_SCHEMA_DIR LOCPATH
fi
exec "$PY" "$DIR/vm/vmview.py" "$@"
