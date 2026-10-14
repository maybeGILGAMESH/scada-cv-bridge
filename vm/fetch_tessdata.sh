#!/bin/bash
# Модели Tesseract для чтения журнала: русская (tessdata_best) + английская.
# Кладутся в vm/tessdata/, sudo не нужен.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tessdata
for l in rus eng; do
  [ -f tessdata/$l.traineddata ] || curl -fsSL -o tessdata/$l.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata
done
ls -la tessdata
