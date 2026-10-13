# -*- coding: utf-8 -*-
"""Способы снять картинку с поднятого эмулятора щита.

Эмулятор — Chrome с neft-hmi-cv.html, им управляет Playwright. «HDMI-выход»
можно снять по-разному, и это заметно влияет на частоту и задержку:

    screencast  CDP Page.startScreencast: браузер сам кодирует JPEG после
                каждой отрисовки и проталкивает кадр — ближе всего к видеовыходу
    screenshot  CDP Page.captureScreenshot по запросу — каждый кадр отдельной командой
    canvas      canvas.toDataURL('image/jpeg') внутри страницы — только щит, без
                того, что поверх него; кадр гоняется через JS и base64
    screen      захват экрана монитора (mss) — буквально то, что ушло бы в HDMI:
                весь рабочий стол, браузер должен быть видим (--headed)

Каждый способ отдаёт кадры одинаково: (jpeg-байты или массив BGR, w, h, t_cap)
и описывает, как координаты кадра переводятся в координаты страницы —
это нужно обратному каналу, чтобы клик попал туда, куда целилось распознавание.
"""

import asyncio
import base64
import struct
import sys

import numpy as np

from protocol import clock

CAPTURES = ["screencast", "screenshot", "canvas", "screen"]


def jpeg_size(b):
    """Размер JPEG по маркеру SOF — без декодирования всего кадра."""
    i = 2
    n = len(b)
    while i + 9 < n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seg = struct.unpack(">H", b[i + 2:i + 4])[0]
        i += 2 + seg
    return None


class Captured:
    __slots__ = ("jpeg", "img", "w", "h", "t_cap")

    def __init__(self, jpeg=None, img=None, w=0, h=0, t_cap=0.0):
        self.jpeg, self.img, self.w, self.h, self.t_cap = jpeg, img, w, h, t_cap


class Capture:
    name = "?"
    # область страницы (CSS px), которую покрывает кадр: x0, y0, ширина, высота
    region = (0.0, 0.0, 1920.0, 1080.0)
    os_level = False          # True — клик надо делать мышью ОС, а не через CDP

    def __init__(self, page, fps=30, quality=80):
        self.page, self.fps, self.quality = page, fps, quality
        self.frame_size = None

    async def start(self):
        pass

    async def frames(self):
        raise NotImplementedError
        yield

    async def stop(self):
        pass

    def to_page(self, x, y):
        """Координаты в кадре → координаты страницы (CSS px)."""
        fw, fh = self.frame_size or (self.region[2], self.region[3])
        x0, y0, rw, rh = self.region
        return x0 + x * rw / fw, y0 + y * rh / fh


async def _viewport(page):
    vs = page.viewport_size
    if vs:
        return 0.0, 0.0, float(vs["width"]), float(vs["height"])
    w, h = await page.evaluate("[innerWidth, innerHeight]")
    return 0.0, 0.0, float(w), float(h)


class ScreencastCapture(Capture):
    name = "screencast"

    async def start(self):
        self.region = await _viewport(self.page)
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.q = asyncio.Queue(maxsize=1)
        self._last = 0.0
        self.skipped = 0
        self.cdp.on("Page.screencastFrame", self._on_frame)
        await self.cdp.send("Page.startScreencast", {
            "format": "jpeg", "quality": int(self.quality),
            "maxWidth": int(self.region[2]), "maxHeight": int(self.region[3]),
            "everyNthFrame": 1})

    def _on_frame(self, ev):
        t = clock()
        asyncio.ensure_future(self.cdp.send("Page.screencastFrameAck",
                                            {"sessionId": ev["sessionId"]}))
        if self.fps and t - self._last < 1.0 / self.fps * 0.9:
            self.skipped += 1
            return
        self._last = t
        jpg = base64.b64decode(ev["data"])
        if self.q.full():
            try:
                self.q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.q.put_nowait((jpg, t))

    async def frames(self):
        while True:
            jpg, t = await self.q.get()
            size = jpeg_size(jpg) or (int(self.region[2]), int(self.region[3]))
            self.frame_size = size
            yield Captured(jpeg=jpg, w=size[0], h=size[1], t_cap=t)

    async def stop(self):
        try:
            await self.cdp.send("Page.stopScreencast")
        except Exception:
            pass


class ScreenshotCapture(Capture):
    name = "screenshot"

    async def start(self):
        self.region = await _viewport(self.page)
        self.cdp = await self.page.context.new_cdp_session(self.page)

    async def frames(self):
        period = 1.0 / self.fps if self.fps else 0.0
        while True:
            t = clock()
            r = await self.cdp.send("Page.captureScreenshot", {
                "format": "jpeg", "quality": int(self.quality), "optimizeForSpeed": True})
            jpg = base64.b64decode(r["data"])
            size = jpeg_size(jpg) or (int(self.region[2]), int(self.region[3]))
            self.frame_size = size
            yield Captured(jpeg=jpg, w=size[0], h=size[1], t_cap=t)
            left = period - (clock() - t)
            if left > 0:
                await asyncio.sleep(left)


class CanvasCapture(Capture):
    """Только холст щита. Наложенное поверх (полоса подсказок) сюда не попадает."""
    name = "canvas"

    async def start(self):
        r = await self.page.evaluate(
            "(()=>{const r=document.getElementById('panel').getBoundingClientRect();"
            "return [r.x,r.y,r.width,r.height]})()")
        self.region = tuple(float(v) for v in r)
        self.frame_size = (1920, 1080)

    async def frames(self):
        period = 1.0 / self.fps if self.fps else 0.0
        js = f"document.getElementById('panel').toDataURL('image/jpeg',{self.quality / 100:.2f})"
        while True:
            t = clock()
            url = await self.page.evaluate(js)
            jpg = base64.b64decode(url.split(",", 1)[1])
            size = jpeg_size(jpg) or (1920, 1080)
            self.frame_size = size
            yield Captured(jpeg=jpg, w=size[0], h=size[1], t_cap=t)
            left = period - (clock() - t)
            if left > 0:
                await asyncio.sleep(left)


class ScreenCapture(Capture):
    """Экран монитора целиком, как его увидела бы карта захвата HDMI.
    Клики в этом режиме — мышью операционной системы (как HID у KVM)."""
    name = "screen"
    os_level = True

    def __init__(self, page, fps=30, quality=80, monitor=1):
        super().__init__(page, fps, quality)
        self.monitor = monitor

    async def start(self):
        import mss
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
        self.sct = mss.mss()
        mon = self.sct.monitors[self.monitor]
        self.mon = mon
        self.region = (float(mon["left"]), float(mon["top"]),
                       float(mon["width"]), float(mon["height"]))
        self.frame_size = (mon["width"], mon["height"])

    def _grab(self):
        a = np.asarray(self.sct.grab(self.mon))
        return np.ascontiguousarray(a[:, :, :3])

    async def frames(self):
        period = 1.0 / self.fps if self.fps else 0.0
        loop = asyncio.get_running_loop()
        while True:
            t = clock()
            # mss держит контекст устройства в своём потоке — грабим здесь же
            img = self._grab()
            yield Captured(img=img, w=img.shape[1], h=img.shape[0], t_cap=t)
            left = period - (clock() - t)
            await asyncio.sleep(max(0.0, left))

    def to_page(self, x, y):
        # в экранных координатах: начало монитора плюс точка кадра
        return self.region[0] + x, self.region[1] + y


def make_capture(kind, page, fps=30, quality=80, **kw):
    cls = {"screencast": ScreencastCapture, "screenshot": ScreenshotCapture,
           "canvas": CanvasCapture, "screen": ScreenCapture}.get(kind)
    if cls is None:
        raise ValueError(f"неизвестный захват {kind}; есть: {', '.join(CAPTURES)}")
    return cls(page, fps, quality, **kw)


def os_click(x, y):
    """Клик мышью ОС в экранных координатах — то, что сделал бы HID KVM."""
    if not sys.platform.startswith("win"):
        raise RuntimeError("клик мышью ОС реализован только для Windows")
    import ctypes
    u = ctypes.windll.user32
    u.SetCursorPos(int(x), int(y))
    u.mouse_event(0x0002, 0, 0, 0, 0)      # левая вниз
    u.mouse_event(0x0004, 0, 0, 0, 0)      # левая вверх
