# -*- coding: utf-8 -*-
"""Сторона распознавания (ПК-2): принимает «HDMI»-поток, читает щит по
картинке, копит и обогащает показания, управляет щитом через HID.

    транспорт ─▶ приём (поток) ─▶ ящик «последний кадр» ─▶ распознавание (поток)
                                                              │
               ┌──────────────────────────────────────────────┤
               ▼                     ▼                         ▼
          агрегатор            управление/автоматика       сайт :8771
     (история, расчёт,        (клик по найденной кнопке,   (показания, события,
      события, уставки)        проверка по кадрам)          команды, отчёт)
                                     │
                                     └──▶ обратный канал (HID) ──▶ эмулятор / PiKVM

Приём и распознавание — разные потоки: транспорт отдаёт 30 к/с, а чтение
кадра стоит от 15 мс до 0,5 с. Распознавание всегда берёт самый свежий кадр,
промежуточные выбрасываются — задержка не копится.

Примеры:
    python bridge/receiver.py                                      # tcp 127.0.0.1:8780, HID 127.0.0.1:8790
    python bridge/receiver.py --transport ws --port 8780
    python bridge/receiver.py --source http://pikvm.local/streamer/stream \\
                              --pikvm https://pikvm.local --pikvm-user admin --pikvm-pass admin
    python bridge/receiver.py --source ../cv_templates/validation   # без железа, по файлам
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "live"))

import normalize as N                                    # noqa: E402
from aggregator import Aggregator                        # noqa: E402
from control import Automation, Controller, EmulatorHid, PiKvmHid  # noqa: E402
from fast_reader import CLASS_TITLE, GROUP_ORDER, FastReader, pack_rows  # noqa: E402
from protocol import clock                               # noqa: E402
from transports import TRANSPORTS, LinkStats, Mailbox, make_receiver  # noqa: E402
from web import Web                                      # noqa: E402


class Normalizer(N.Normalizer):
    """Нормализатор live/ с двумя поправками для программного канала.

    1. Сначала пробуется «один к одному»: шапка щита на своём месте в кадре —
       значит масштаб 1 и калибровка не нужна (она стоит секунды).
    2. Найденный масштаб в пределах 1 % от единицы округляется до единицы:
       иначе каждый кадр зря растягивался бы на доли процента с размытием.
    """

    def _quick(self, frame):
        x, y, w, h = self.rect
        tpl = self.ref[y:y + h, x:x + w]
        pad = 6
        roi = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if roi.shape[0] < h or roi.shape[1] < w:
            return None
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        # Порог не 0,9: headless-Chrome растеризует шрифт шапки чуть иначе,
        # чем браузер, в котором нарезан эталон, — на живом кадре выходит 0,89.
        if mx < 0.8:
            return None
        ox = max(0, x - pad) + loc[0] - x
        oy = max(0, y - pad) + loc[1] - y
        if abs(ox) <= 1 and abs(oy) <= 1:      # пиксель — это разница растеризации, не сдвиг
            ox = oy = 0
        return N.Calibration(1.0, ox, oy, mx)

    def __call__(self, frame):
        if frame is None:
            return None
        if self.cal is None:
            cal = self._quick(frame) or N.calibrate(frame, self.ref, self.rect)
            if cal is not None and abs(cal.scale - 1.0) < 0.01:
                cal.scale, cal.x, cal.y = 1.0, float(round(cal.x)), float(round(cal.y))
            self.cal = cal
            self.recalibrations += 1
        return N.apply(frame, self.cal)


class LegacySource:
    """Источники исходного конвейера live/: MJPEG PiKVM, карта захвата, файлы."""

    def __init__(self, spec):
        import source as S
        self.src = S.open_source(spec)
        self.name = self.src.name
        self.seq = 0

    def recv(self, timeout=1.0):
        from protocol import Frame
        img = self.src.read()
        if img is None:
            return None
        self.seq += 1
        t = clock()
        f = Frame(self.seq, t, 0, img.shape[1], img.shape[0], b"", t_send=t, t_recv=t)
        f.payload = img          # уже декодирован
        return f

    def close(self):
        self.src.close()


class Receiver:
    def __init__(self, a):
        self.a = a
        self.link_stats = LinkStats()
        self.box = Mailbox()
        self.reader = FastReader(a.templates, only=a.only, register_every=a.register_every,
                                 gauge_method=a.gauge_method, cache=not a.no_cache)
        self.normalizer = None if a.no_normalize else Normalizer(self.reader.bank.ref())
        self.agg = Aggregator()

        if a.pikvm:
            self.hid = PiKvmHid(a.pikvm, a.pikvm_user, a.pikvm_pass)
        elif a.control and a.control != "none":
            host, _, port = a.control.rpartition(":")
            self.hid = EmulatorHid(host or "127.0.0.1", int(port))
        else:
            self.hid = None
        self.ctl = Controller(self.hid, self.agg) if self.hid else None
        self.auto = Automation(self.ctl, self.agg) if self.ctl else None
        if self.auto and a.auto:
            for r in self.auto.rules:
                r.enabled = True

        self.state = {"status": "ожидание кадра"}
        self.cv_times = []
        self.frames_cv = 0
        self.thumb = b""
        self.assist_on = not a.no_assist
        self.assist_err = None
        self.web = Web(self, a.web_host, a.web_port)
        self.src = None

    # ── приём ───────────────────────────────────────────────────────
    def open_source(self):
        a = self.a
        if a.source:
            return LegacySource(a.source)
        return make_receiver(a.transport, a.host, a.port, connect_timeout=3600)

    def rx_loop(self):
        while True:
            try:
                if self.src is None:
                    self.state["status"] = "подключение к источнику…"
                    self.src = self.open_source()
                    self.state["status"] = "поток идёт"
                f = self.src.recv(timeout=1.0)
                if f is None:
                    continue
                self.link_stats.note(f)
                self.box.put(f)
            except (ConnectionError, OSError, EOFError) as e:
                self.state["status"] = f"нет потока: {e}"
                try:
                    if self.src:
                        self.src.close()
                except Exception:
                    pass
                self.src = None
                time.sleep(1.0)

    # ── распознавание ───────────────────────────────────────────────
    def cv_loop(self):
        force = True
        last_pub = 0.0
        last_thumb = 0.0
        while True:
            f = self.box.get(1.0)
            if f is None:
                continue
            t0 = clock()
            raw = f.payload if isinstance(f.payload, np.ndarray) else f.image()
            if raw is None:
                continue
            t_dec = clock()
            frame = self.normalizer(raw) if self.normalizer else raw
            if frame is None or frame.shape[0] < 200:
                continue
            out = self.reader.read(frame, force_register=force)
            force = False                   # дальше ридер сам решает, когда искать блоки
            if self.normalizer:
                self.normalizer.note_quality(out["registration_quality"])
            t_done = clock()
            now = time.time()

            groups = pack_rows(out["rows"], self.reader.bank)
            self.agg.update(now, groups)
            cal = self.normalizer.cal if self.normalizer else None
            if self.ctl:
                self.ctl.on_reading(groups, None if cal is None else (cal.scale, cal.x, cal.y))
                self.auto.step()

            self.frames_cv += 1
            self.cv_times.append(t_done - t0)
            self.cv_times = self.cv_times[-50:]
            if now - last_thumb > 0.25:
                last_thumb = now
                k = 760 / frame.shape[1]
                small = cv2.resize(frame, (760, int(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok:
                    self.thumb = buf.tobytes()
            if now - last_pub > 0.2:          # сайту хватает пяти обновлений в секунду
                last_pub = now
                self.publish(f, raw, groups, out, cal, t0, t_dec, t_done)

    def publish(self, f, raw, groups, out, cal, t0, t_dec, t_done):
        cvt = self.cv_times
        self.state = {
            "status": "поток идёт",
            "ts": time.time(),
            "source": self.src.name if self.src else None,
            "transport": self.a.transport if not self.a.source else "источник live/",
            "link": self.link_stats.summary(),
            "cv": {"fps": round(len(cvt) / max(sum(cvt), 1e-6), 1),
                   "ms_last": round((t_done - t0) * 1000, 1),
                   "decode_ms": round((t_dec - t0) * 1000, 1),
                   "read_ms": out["timing_ms"]["total"],
                   "read_objects": out["read_objects"], "cached_objects": out["cached_objects"],
                   "frames": self.frames_cv,
                   # от захвата на эмуляторе до готового чтения: кодек + транспорт + очередь + CV
                   "glass_ms": round((t_done - f.t_cap) * 1000, 1) if f.t_cap else None,
                   "gauge_method": self.a.gauge_method, "cache": not self.a.no_cache},
            "raw_size": [int(raw.shape[1]), int(raw.shape[0])],
            "calibration": None if cal is None else {
                "scale": round(cal.scale, 4), "x": round(cal.x, 1), "y": round(cal.y, 1),
                "score": round(cal.score, 3),
                "recalibrations": self.normalizer.recalibrations},
            "registration_quality": out["registration_quality"],
            "blocks": {k: {"dx": v[0], "dy": v[1], "score": round(v[2], 3)}
                       for k, v in sorted((out["registration"] or {}).items())},
            "layout": self.reader.layout_name(),
            "groups": groups,
            "group_order": [g for g in GROUP_ORDER if g in groups],
            "group_titles": {g: CLASS_TITLE.get(g, g) for g in groups},
            "agg": self.agg.snapshot(),
            "control": self.control_state(),
        }

    def control_state(self):
        if not self.ctl:
            return {"available": False}
        h = self.hid
        return {"available": True, "hid": h.name, "connected": h.connected,
                "rtt_ms": None if h.rtt_ms is None else round(h.rtt_ms, 2),
                "info": h.info, "jobs": list(self.ctl.jobs)[:25],
                "rules": self.auto.describe(), "assist": self.assist_on,
                "assist_error": self.assist_err}

    # ── подсказки на экран щита ─────────────────────────────────────
    def assist_loop(self):
        """Раз в секунду — что насчитал приёмник, в полосу под щитом."""
        while True:
            time.sleep(self.a.assist_every)
            if not self.hid:
                continue
            try:
                self.hid.ping()
            except Exception:
                continue
            if not self.assist_on or not self.state.get("link"):
                continue
            st, link = self.state, self.state.get("link", {})
            items = []
            for d in self.agg.derived():
                if d["id"] in ("q_oil", "q_check", "availability") or d["id"].startswith("eta_"):
                    items.append({"label": d["label"], "value": d["value"], "unit": d["unit"],
                                  "level": d.get("level") or ""})
            for tid, lvl in self.agg.levels.items():
                if lvl:
                    tag = self.agg.tags[tid]
                    items.append({"label": tag.label, "value": f"{tag.value:.1f}",
                                  "unit": tag.unit or "", "level": lvl})
            cv = st.get("cv", {})
            data = {"title": "CV-АССИСТЕНТ · данные прочитаны с картинки",
                    "lines": [f"канал {st.get('transport')}: {link.get('fps', 0)} к/с, "
                              f"задержка {link.get('lat_ms_p50', '—')} мс",
                              f"распознавание {cv.get('fps', 0)} к/с, "
                              f"от захвата до показаний {cv.get('glass_ms', '—')} мс",
                              f"обратный канал: RTT {self.hid.rtt_ms:.1f} мс"
                              if self.hid.rtt_ms is not None else "обратный канал: —"],
                    "items": items[:9],
                    "events": [{"text": f"{e['clock']} {e['text']}", "level": e["level"]}
                               for e in list(self.agg.events)[:5]]}
            try:
                self.hid.show(data)
                self.assist_err = None
            except Exception as e:
                self.assist_err = str(e)

    def run(self):
        threading.Thread(target=self.rx_loop, daemon=True).start()
        threading.Thread(target=self.cv_loop, daemon=True).start()
        threading.Thread(target=self.assist_loop, daemon=True).start()
        self.web.start()
        a = self.a
        print(f"Источник : {a.source or f'{a.transport}://{a.host}:{a.port}'}")
        print(f"HID      : {self.hid.name if self.hid else 'нет (только чтение)'}"
              + (f" {a.control}" if self.hid and not a.pikvm else ""))
        print(f"Чтение   : манометры — {a.gauge_method}, кэш {'выкл' if a.no_cache else 'вкл'}")
        print(f"Сайт     : http://{a.web_host}:{a.web_port}/")
        print("Остановка: Ctrl+C\n")
        try:
            while True:
                time.sleep(1.0)
                st = self.state
                link, cv = st.get("link") or {}, st.get("cv") or {}
                print(f"\r{st.get('status', ''):<14} канал {link.get('fps', 0):>5} к/с "
                      f"p50 {link.get('lat_ms_p50', '—'):>6} мс | CV {cv.get('fps', 0):>5} к/с "
                      f"{cv.get('ms_last', '—'):>6} мс (читано {cv.get('read_objects', '—')}, "
                      f"кэш {cv.get('cached_objects', '—')}) | захват→показания "
                      f"{cv.get('glass_ms', '—')} мс   ", end="", flush=True)
        except KeyboardInterrupt:
            print("\nостановлено")


def main():
    ap = argparse.ArgumentParser(description="Приёмник: распознавание щита, агрегация, управление")
    ap.add_argument("--transport", default="tcp", choices=TRANSPORTS)
    ap.add_argument("--host", default="127.0.0.1", help="адрес эмулятора (видеоканал)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--source", default=None,
                    help="вместо транспорта — источник live/: MJPEG-URL PiKVM, номер карты "
                         "захвата, файл или папка")
    ap.add_argument("--control", default="127.0.0.1:8790",
                    help="обратный канал эмулятора host:port; none — только чтение")
    ap.add_argument("--pikvm", default=None, help="HID настоящего PiKVM: https://pikvm.local")
    ap.add_argument("--pikvm-user", default="admin")
    ap.add_argument("--pikvm-pass", default="admin")
    ap.add_argument("--templates", default=str(ROOT / "cv_templates"))
    ap.add_argument("--only", default=None, help="читать только эти классы, через запятую")
    ap.add_argument("--gauge-method", default="needle", choices=["needle", "template"])
    ap.add_argument("--no-cache", action="store_true", help="перечитывать все объекты каждый кадр")
    ap.add_argument("--register-every", type=int, default=30,
                    help="полный поиск блоков раз в столько кадров; перестановку и между ними "
                         "ловит покадровая проверка одного блока")
    ap.add_argument("--no-normalize", action="store_true",
                    help="кадр уже 1920×1080 один к одному — не искать масштаб")
    ap.add_argument("--auto", action="store_true", help="сразу включить правила автоматики")
    ap.add_argument("--no-assist", action="store_true", help="не выводить подсказки на экран щита")
    ap.add_argument("--assist-every", type=float, default=1.0)
    ap.add_argument("--web-host", default="127.0.0.1")
    ap.add_argument("--web-port", type=int, default=8771)
    a = ap.parse_args()
    Receiver(a).run()


if __name__ == "__main__":
    main()
