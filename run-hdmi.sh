#!/bin/bash
# Щит на весь экран (киоск). Закрыть: кнопка «Закрыть» (появляется при движении мыши) или Esc.
DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL="$DIR/neft-hmi-cv-fullscreen.html"
[ -f "$PANEL" ] || PANEL="$DIR/extracted/scada-cv-bridge-main/neft-hmi-cv-fullscreen.html"
exec google-chrome --kiosk --new-window \
  --user-data-dir="$DIR/.chrome-kiosk" \
  --no-first-run --no-default-browser-check --noerrdialogs \
  --disable-session-crashed-bubble --disable-infobars \
  --disable-features=Translate,TranslateUI --lang=ru \
  "file://$PANEL"
