# -*- coding: utf-8 -*-
"""Распознавание щита с трансляции в реальном времени и вывод на локальный сайт.

Схема работы:

    ПК-1 (симулятор) ──HDMI──▶ KVM (Pi Zero 2) ──USB/сеть──▶ ПК-2 (этот скрипт)
                                                                  │
                                                     http://127.0.0.1:8770

Скрипт видит только картинку с трансляции. Из неё он:
  1) находит щит и возвращает кадр к масштабу 1920x1080;
  2) находит в кадре блоки щита — сопоставлением изображения, не по координатам;
  3) читает приборы там, где блок нашёлся;
  4) публикует показания на локальный сайт.

Примеры:
    python live/run.py --source http://pikvm.local/streamer/stream
    python live/run.py --source 0                       # USB-карта захвата
    python live/run.py --source cv_templates/validation/B --port 8770
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import source as src_mod            # noqa: E402
import normalize as norm_mod        # noqa: E402
from recognize import Reader        # noqa: E402
from server import State, serve     # noqa: E402

ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser(
        description="Чтение щита АСУ ТП с KVM-трансляции и вывод на локальный сайт")
    ap.add_argument("--source", required=True,
                    help="MJPEG-URL, snapshot:URL, номер устройства захвата "
                         "или путь к кадру/папке")
    ap.add_argument("--templates", default=str(ROOT / "cv_templates"),
                    help="папка с manifest.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--only", default=None,
                    help="читать только эти классы приборов, через запятую")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="минимальная пауза между распознаваниями, с")
    ap.add_argument("--thumb", type=int, default=760,
                    help="ширина кадра, отдаваемого на сайт (0 — не отдавать)")
    ap.add_argument("--register-every", type=int, default=8,
                    help="как часто заново искать блоки в кадре, в кадрах; "
                         "при просадке балла поиск запускается и вне очереди")
    ap.add_argument("--gauge-stride", type=int, default=1,
                    help="шаг грубого прохода по углам манометра (1 — все углы)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="кадр уже 1920x1080 один к одному — не искать масштаб")
    ap.add_argument("--dump", default=None,
                    help="писать каждое состояние в этот JSON-файл")
    a = ap.parse_args()

    print("Эталоны   :", a.templates)
    reader = Reader(a.templates, only=a.only,
                    register_every=a.register_every, gauge_stride=a.gauge_stride)
    print("Компоновка эталонов:", reader.bank.layout_id or "не указана",
          "| блоков:", len(reader.bank.blocks),
          "| приборов:", len(reader.bank.by_obj))

    source = src_mod.open_source(a.source)
    print("Источник  :", source.name)

    normalizer = None if a.no_normalize else norm_mod.Normalizer(reader.bank.ref())

    state = State()
    serve(state, a.host, a.port)
    print(f"Сайт      : http://{a.host}:{a.port}/")
    print("Остановка : Ctrl+C\n")

    times = deque(maxlen=20)
    force_reg = True
    frames = 0
    try:
        while True:
            t0 = time.time()
            raw = source.read()
            if raw is None:
                time.sleep(0.2)
                continue

            frame = normalizer(raw) if normalizer else raw
            if frame is None or frame.shape[0] < 200:
                time.sleep(0.2)
                continue

            data = reader.read(frame, force_register=force_reg)
            force_reg = data["registration_quality"] < 0.45
            if normalizer:
                normalizer.note_quality(data["registration_quality"])
                cal = normalizer.cal
                data["calibration"] = (None if cal is None else
                                       {"scale": round(cal.scale, 4),
                                        "x": round(cal.x, 1), "y": round(cal.y, 1),
                                        "score": round(cal.score, 4),
                                        "recalibrations": normalizer.recalibrations})
            else:
                data["calibration"] = None

            times.append(time.time() - t0)
            frames += 1
            data.update(ts=time.time(), frames=frames,
                        fps=round(len(times) / max(sum(times), 1e-6), 2),
                        source=source.name,
                        raw_size=[int(raw.shape[1]), int(raw.shape[0])])

            thumb = None
            if a.thumb:
                k = a.thumb / frame.shape[1]
                small = cv2.resize(frame, (a.thumb, int(frame.shape[0] * k)),
                                   interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                thumb = buf.tobytes() if ok else None

            state.publish(data, thumb)
            if a.dump:
                Path(a.dump).write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                        encoding="utf-8")

            lay = data["layout"]
            print(f"\rкадр {frames:>6}  {data['fps']:>5.2f} к/с  "
                  f"компоновка {lay.get('name','?'):<22} "
                  f"привязка {data['registration_quality']:.3f}  "
                  f"{data['timing_ms']['total']:>4} мс", end="", flush=True)

            if a.interval:
                left = a.interval - (time.time() - t0)
                if left > 0:
                    time.sleep(left)
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        source.close()


if __name__ == "__main__":
    main()
