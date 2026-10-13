# -*- coding: utf-8 -*-
"""Транспорты «HDMI»-канала: как кадр попадает от эмулятора к распознаванию.

В исходном варианте кадр шёл по настоящему HDMI в KVM (Pi Zero 2), а оттуда
MJPEG-потоком по сети. Здесь кабеля нет — эмулятор и приёмник связаны
программно, и способов связать их несколько. Все они реализуют один
интерфейс, поэтому подменяются ключом --transport и сравниваются
bench_transport.py на одних и тех же кадрах.

    tcp    TCP, пакеты с длиной в заголовке               — «локальный сокет», сеть
    ws     WebSocket, одно бинарное сообщение на кадр     — то же через HTTP-рукопожатие
    udp    UDP, кадр режется на датаграммы и собирается    — пакетная передача без гарантий
    pipe   именованный канал Windows / AF_UNIX в Linux    — локальный сокет без сетевого стека
    shm    общая память + «звонок» датаграммой            — без копирования через ядро
    mjpeg  HTTP multipart/x-mixed-replace                  — как отдаёт PiKVM, совместим с live/run.py

Направление у всех одно: источник (эмулятор) слушает, приёмник подключается —
как у KVM, где поток раздаёт само устройство захвата.

Живое видео не должно копить очередь: если приёмник не успевает, старый кадр
выбрасывается и уходит свежий. Поэтому у каждого подключения отправителя —
почтовый ящик на один кадр; число выброшенных кадров видно в статистике.
"""

import socket
import struct
import sys
import threading
import time
from collections import deque

from protocol import Frame, HDR_SIZE, clock

TRANSPORTS = ["tcp", "ws", "udp", "pipe", "shm", "mjpeg"]

TRANSPORT_TITLE = {
    "tcp": "TCP, пакеты с длиной",
    "ws": "WebSocket, бинарные сообщения",
    "udp": "UDP, датаграммы с фрагментацией",
    "pipe": "локальный сокет (named pipe / AF_UNIX)",
    "shm": "общая память + звонок",
    "mjpeg": "HTTP MJPEG (как PiKVM)",
}


# ───────────────────────────────────────────────────────────── общее

class Mailbox:
    """Ящик на один предмет: новый вытесняет старый. Для живого видео
    свежесть важнее полноты — очередь кадров означала бы растущую задержку."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None
        self.replaced = 0
        self.closed = False

    def put(self, item):
        with self._cv:
            if self._item is not None:
                self.replaced += 1
            self._item = item
            self._cv.notify()

    def get(self, timeout=None):
        with self._cv:
            if self._item is None and not self.closed:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            self._cv.notify_all()
            return item

    def wait_empty(self, timeout=None):
        with self._cv:
            if self._item is not None and not self.closed:
                self._cv.wait_for(lambda: self._item is None or self.closed, timeout)
            return self._item is None

    def close(self):
        with self._cv:
            self.closed = True
            self._cv.notify_all()


class Sender:
    name = "?"

    def __init__(self):
        self.sent = 0
        self._boxes = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # Кадр раздаётся по ящикам подключений; отправка идёт в их потоках.
    def send(self, frame):
        with self._lock:
            boxes = list(self._boxes)
        for b in boxes:
            b.put(frame)
        self.sent += 1

    def clients(self):
        with self._lock:
            return len(self._boxes)

    def wait_idle(self, timeout=1.0):
        """Ждёт, пока все подключения заберут свой кадр. Нужна для замера
        пропускной способности: без неё источник в пустом цикле душит GIL
        потокам отправки и мерит сам себя, а не транспорт."""
        with self._lock:
            boxes = list(self._boxes)
        return all(b.wait_empty(timeout) for b in boxes)

    @property
    def dropped(self):
        with self._lock:
            return sum(b.replaced for b in self._boxes)

    def _add_box(self):
        b = Mailbox()
        with self._lock:
            self._boxes.append(b)
        return b

    def _drop_box(self, b):
        b.close()
        with self._lock:
            if b in self._boxes:
                self._boxes.remove(b)

    def close(self):
        self._stop.set()
        with self._lock:
            for b in self._boxes:
                b.close()


class Receiver:
    name = "?"

    def recv(self, timeout=1.0):
        """Следующий кадр (Frame) либо None по таймауту."""
        raise NotImplementedError

    def close(self):
        pass


def _recv_exact(sock, view):
    got, n = 0, len(view)
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("соединение закрыто")
        got += k


def _tune(sock, buf=2 << 20):
    """Буферы сокета. Для потоковых транспортов — умеренные: глубокий буфер
    при отставшем приёмнике копит секунды старых кадров, то есть задержку.
    Датаграммам, наоборот, нужен большой — иначе ядро их просто теряет."""
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, buf)
        except OSError:
            pass


# ───────────────────────────────────────────────────────────── TCP

class TcpSender(Sender):
    """Поток байтов TCP, кадры разделены длиной из заголовка."""

    def __init__(self, host, port):
        super().__init__()
        self.name = f"tcp://{host}:{port}"
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(4)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _tune(c)
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        box = self._add_box()
        try:
            while not self._stop.is_set():
                f = box.get(0.5)
                if f is None:
                    continue
                hdr, body = f.pack()
                if len(body) < 256 << 10:
                    c.sendall(hdr + bytes(body))      # мелкий кадр — одним сегментом
                else:
                    c.sendall(hdr)
                    c.sendall(memoryview(body))
        except OSError:
            pass
        finally:
            self._drop_box(box)
            c.close()

    def close(self):
        super().close()
        self.srv.close()


class TcpReceiver(Receiver):
    def __init__(self, host, port, connect_timeout=10.0):
        self.name = f"tcp://{host}:{port}"
        self.sock = _connect_retry(host, port, connect_timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _tune(self.sock)
        self.hdr = bytearray(HDR_SIZE)
        self.buf = bytearray(1 << 20)

    def recv(self, timeout=1.0):
        self.sock.settimeout(timeout)
        try:
            _recv_exact(self.sock, memoryview(self.hdr))
        except socket.timeout:
            return None
        self.sock.settimeout(None)
        f, n = Frame.parse_header(self.hdr)
        if n > len(self.buf):
            self.buf = bytearray(n)
        _recv_exact(self.sock, memoryview(self.buf)[:n])
        f.payload = bytes(memoryview(self.buf)[:n])
        f.t_recv = clock()
        return f

    def close(self):
        self.sock.close()


def _connect_retry(host, port, timeout):
    t_end = time.time() + timeout
    while True:
        try:
            return socket.create_connection((host, port), timeout=2.0)
        except OSError:
            if time.time() > t_end:
                raise
            time.sleep(0.1)


# ───────────────────────────────────────────────────────────── WebSocket

class WsSender(Sender):
    """Один кадр — одно бинарное сообщение. Сжатие permessage-deflate
    выключено: по умолчанию оно включено и на сыром кадре съедает процессор
    без всякого выигрыша на JPEG."""

    def __init__(self, host, port):
        super().__init__()
        from websockets.sync.server import serve
        self.name = f"ws://{host}:{port}/"
        self.srv = serve(self._serve, host, port, max_size=None, compression=None,
                         ping_interval=None)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def _serve(self, ws):
        box = self._add_box()
        try:
            while not self._stop.is_set():
                f = box.get(0.5)
                if f is None:
                    continue
                hdr, body = f.pack()
                ws.send(hdr + bytes(body))
        except Exception:
            pass
        finally:
            self._drop_box(box)

    def close(self):
        super().close()
        self.srv.shutdown()


class WsReceiver(Receiver):
    def __init__(self, host, port, connect_timeout=10.0):
        from websockets.sync.client import connect
        self.name = f"ws://{host}:{port}/"
        t_end = time.time() + connect_timeout
        while True:
            try:
                self.ws = connect(f"ws://{host}:{port}/", max_size=None,
                                  compression=None, ping_interval=None)
                break
            except OSError:
                if time.time() > t_end:
                    raise
                time.sleep(0.1)

    def recv(self, timeout=1.0):
        from websockets.exceptions import ConnectionClosed
        try:
            msg = self.ws.recv(timeout=timeout)
        except TimeoutError:
            return None
        except ConnectionClosed as e:
            raise ConnectionError("WebSocket закрыт") from e
        return Frame.from_bytes(msg)

    def close(self):
        self.ws.close()


# ───────────────────────────────────────────────────────────── UDP

FRAG = struct.Struct("<4sIHH")          # b"SCU1", seq, номер куска, всего кусков
FRAG_MAGIC = b"SCU1"
HELLO = b"SCU1HELLO"


class UdpSender(Sender):
    """Кадр режется на датаграммы. Гарантий нет: потерялся кусок — пропал
    кадр целиком, и это честно видно в статистике потерь.

    mtu=0 — датаграммы по ~60 КБ (на localhost это работает: петля не режет
    пакеты). В настоящей сети датаграмму больше MTU режет уже IP, и потеря
    любого фрагмента губит всю датаграмму, — поэтому есть mtu=1400."""

    def __init__(self, host, port, mtu=0):
        super().__init__()
        self.name = f"udp://{host}:{port}" + (f" mtu {mtu}" if mtu else "")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _tune(self.sock, 16 << 20)
        self.sock.bind((host, port))
        self.chunk = (mtu - 28 - FRAG.size) if mtu else 60000
        self.subs = {}                  # адрес → время последнего «привет»
        self.dropped_udp = 0
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(64)
            except OSError:
                if self._stop.is_set():
                    return
                continue                # Windows: ICMP «порт недоступен» от ушедшего приёмника
            if data == HELLO:
                with self._lock:
                    self.subs[addr] = time.time()

    def clients(self):
        now = time.time()
        with self._lock:
            for a in [a for a, t in self.subs.items() if now - t > 5]:
                del self.subs[a]
            return len(self.subs)

    def send(self, frame):
        if not self.clients():
            self.sent += 1
            return
        hdr, body = frame.pack()
        data = memoryview(hdr + bytes(body))
        n = (len(data) + self.chunk - 1) // self.chunk
        with self._lock:
            subs = list(self.subs)
        for i in range(n):
            part = FRAG.pack(FRAG_MAGIC, frame.seq, i, n) + data[i * self.chunk:(i + 1) * self.chunk]
            for a in subs:
                try:
                    self.sock.sendto(part, a)
                except OSError:
                    self.dropped_udp += 1
        self.sent += 1

    @property
    def dropped(self):
        return self.dropped_udp

    def wait_idle(self, timeout=1.0):
        return True

    def close(self):
        super().close()
        self.sock.close()


class UdpReceiver(Receiver):
    def __init__(self, host, port, **_):
        self.name = f"udp://{host}:{port}"
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _tune(self.sock, 32 << 20)
        self.sock.bind(("0.0.0.0" if host not in ("127.0.0.1", "localhost") else host, 0))
        self.parts = {}                 # seq → [всего, {номер: кусок}]
        self.incomplete = 0             # кадры, у которых потерялся кусок
        self._hello_at = 0.0
        self._hello()

    def _hello(self):
        self.sock.sendto(HELLO, self.addr)
        self._hello_at = time.time()

    def recv(self, timeout=1.0):
        t_end = time.time() + timeout
        while True:
            if time.time() - self._hello_at > 1.0:
                self._hello()           # подписка живёт 5 с — продлеваем
            left = t_end - time.time()
            if left <= 0:
                return None
            self.sock.settimeout(min(left, 1.0))
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue                # Windows: ICMP от источника, который ещё не поднялся
            if len(data) < FRAG.size or data[:4] != FRAG_MAGIC:
                continue
            _, seq, i, n = FRAG.unpack_from(data)
            slot = self.parts.get(seq)
            if slot is None:
                # Пришёл кусок нового кадра — недособранные старые уже не придут.
                for old in [s for s in self.parts if (seq - s) & 0xFFFFFFFF < 1 << 31 and s != seq]:
                    del self.parts[old]
                    self.incomplete += 1
                slot = self.parts[seq] = [n, {}]
            slot[1][i] = data[FRAG.size:]
            if len(slot[1]) == n:
                del self.parts[seq]
                buf = b"".join(slot[1][k] for k in range(n))
                return Frame.from_bytes(buf)

    def close(self):
        self.sock.close()


# ───────────────────────────────────────────────────────────── локальный сокет

def _pipe_address(port):
    if sys.platform.startswith("win"):
        return rf"\\.\pipe\scada-hdmi-{port}", "AF_PIPE"
    return f"/tmp/scada-hdmi-{port}.sock", "AF_UNIX"


class PipeSender(Sender):
    """Именованный канал Windows (в Linux — сокет AF_UNIX). Сетевого стека нет
    вовсе: ни TCP, ни петли, ни контрольных сумм — только буфер ядра."""

    def __init__(self, host, port):
        super().__init__()
        from multiprocessing.connection import Listener
        addr, fam = _pipe_address(port)
        if fam == "AF_UNIX":
            import os
            if os.path.exists(addr):
                os.unlink(addr)
        self.name = f"pipe {addr}"
        self.lst = Listener(addr, family=fam)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                c = self.lst.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        box = self._add_box()
        try:
            while not self._stop.is_set():
                f = box.get(0.5)
                if f is None:
                    continue
                hdr, body = f.pack()
                c.send_bytes(hdr)
                c.send_bytes(body)
        except (OSError, EOFError):
            pass
        finally:
            self._drop_box(box)
            c.close()

    def close(self):
        super().close()
        try:
            self.lst.close()
        except OSError:
            pass


class PipeReceiver(Receiver):
    def __init__(self, host, port, connect_timeout=10.0):
        from multiprocessing.connection import Client
        addr, fam = _pipe_address(port)
        self.name = f"pipe {addr}"
        t_end = time.time() + connect_timeout
        while True:
            try:
                self.c = Client(addr, family=fam)
                break
            except (OSError, FileNotFoundError):
                if time.time() > t_end:
                    raise
                time.sleep(0.1)
        self.buf = bytearray(1 << 20)

    def recv(self, timeout=1.0):
        if not self.c.poll(timeout):
            return None
        hdr = self.c.recv_bytes()
        f, n = Frame.parse_header(hdr)
        if n > len(self.buf):
            self.buf = bytearray(n)
        k = self.c.recv_bytes_into(self.buf)
        f.payload = bytes(memoryview(self.buf)[:k])
        f.t_recv = clock()
        return f

    def close(self):
        self.c.close()


# ───────────────────────────────────────────────────────────── общая память

SHM_META = struct.Struct("<Q")          # размер слота: сегмент в Windows округляется до страницы
SHM_CTRL = struct.Struct("<QIIQ")       # seq, слот, длина, seq ещё раз — признак целой записи
SHM_BASE = SHM_META.size + SHM_CTRL.size


class ShmSender(Sender):
    """Кадр кладётся в общую память, приёмнику уходит лишь «звонок» —
    датаграмма с номером кадра. Данные через ядро не копируются вовсе.

    Два слота попеременно: пока приёмник копирует кадр из одного, источник
    пишет следующий в другой. Если за время копирования источник успел
    записать два кадра, копия считается порванной и выбрасывается.

    Будить приёмник датаграммой приходится потому, что в Windows у Python 3.10
    sleep() не короче миллисекунды, а опрос в цикле съедает ядро процессора."""

    def __init__(self, host, port, max_bytes=1920 * 1080 * 3 + 4096):
        super().__init__()
        from multiprocessing import shared_memory
        self.slot = max_bytes + HDR_SIZE
        size = SHM_BASE + 2 * self.slot
        self.shm_name = f"scada_hdmi_{port}"
        try:
            self.shm = shared_memory.SharedMemory(self.shm_name, create=True, size=size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(self.shm_name)
        SHM_META.pack_into(self.shm.buf, 0, self.slot)
        self.name = f"shm {self.shm_name} + udp://{host}:{port}"
        self.k = 0
        self.bell = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bell.bind((host, port))
        self.subs = {}
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while not self._stop.is_set():
            try:
                data, addr = self.bell.recvfrom(64)
            except OSError:
                if self._stop.is_set():
                    return
                continue
            if data == HELLO:
                with self._lock:
                    self.subs[addr] = time.time()

    def clients(self):
        now = time.time()
        with self._lock:
            for a in [a for a, t in self.subs.items() if now - t > 5]:
                del self.subs[a]
            return len(self.subs)

    def send(self, frame):
        hdr, body = frame.pack()
        n = HDR_SIZE + len(body)
        if n > self.slot:
            raise ValueError(f"кадр {n} байт не влезает в слот {self.slot}")
        self.k += 1
        slot = self.k & 1
        off = SHM_BASE + slot * self.slot
        buf = self.shm.buf
        buf[off:off + HDR_SIZE] = hdr
        buf[off + HDR_SIZE:off + n] = body
        # На x86 записи видны другим ядрам в порядке выполнения, поэтому
        # заголовок управления пишется последним и публикует кадр целиком.
        SHM_CTRL.pack_into(buf, SHM_META.size, self.k, slot, n, self.k)
        ring = struct.pack("<Q", self.k)
        with self._lock:
            subs = list(self.subs)
        for a in subs:
            try:
                self.bell.sendto(ring, a)
            except OSError:
                pass
        self.sent += 1

    @property
    def dropped(self):
        return 0

    def wait_idle(self, timeout=1.0):
        return True

    def close(self):
        super().close()
        self.bell.close()
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass


class ShmReceiver(Receiver):
    def __init__(self, host, port, connect_timeout=10.0):
        from multiprocessing import shared_memory
        name = f"scada_hdmi_{port}"
        self.name = f"shm {name}"
        t_end = time.time() + connect_timeout
        while True:
            try:
                self.shm = shared_memory.SharedMemory(name)
                break
            except FileNotFoundError:
                if time.time() > t_end:
                    raise
                time.sleep(0.1)
        self.addr = (host, port)
        self.bell = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bell.bind((host if host in ("127.0.0.1", "localhost") else "0.0.0.0", 0))
        self.last = 0
        self.torn = 0
        self._hello_at = 0.0
        self._hello()

    def _hello(self):
        self.bell.sendto(HELLO, self.addr)
        self._hello_at = time.time()

    def recv(self, timeout=1.0):
        t_end = time.time() + timeout
        buf = self.shm.buf
        while True:
            if time.time() - self._hello_at > 1.0:
                self._hello()
            seq, slot, n, seq2 = SHM_CTRL.unpack_from(buf, SHM_META.size)
            slot_size = SHM_META.unpack_from(buf, 0)[0]
            if seq == seq2 and seq > self.last:
                self.last = seq
                # Заголовок управления читается не атомарно: при перегрузе
                # можно поймать его на середине записи. Такие копии отбрасываем.
                if slot > 1 or not HDR_SIZE <= n <= slot_size:
                    self.torn += 1
                    continue
                off = SHM_BASE + slot * slot_size
                data = bytes(buf[off:off + n])
                cur = SHM_CTRL.unpack_from(buf, SHM_META.size)[0]
                if cur - seq >= 2:          # источник успел переписать этот слот
                    self.torn += 1
                    continue
                try:
                    return Frame.from_bytes(data)
                except (ValueError, struct.error):
                    self.torn += 1
                    continue
            left = t_end - time.time()
            if left <= 0:
                return None
            self.bell.settimeout(min(left, 1.0))
            try:
                self.bell.recv(64)
            except (socket.timeout, ConnectionResetError):
                pass

    def close(self):
        self.bell.close()
        try:
            self.shm.close()
        except Exception:
            pass


# ───────────────────────────────────────────────────────────── HTTP MJPEG

class MjpegSender(Sender):
    """multipart/x-mixed-replace — так отдают поток PiKVM и mjpg-streamer.
    Каждая часть несёт обычный JPEG, а в заголовках части — номер и метки
    времени. Поэтому поток читается и нашим приёмником, и старым
    live/run.py (тот просто ищет маркеры JPEG и заголовков не видит)."""

    def __init__(self, host, port):
        super().__init__()
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from protocol import CODEC_NAME
        owner = self
        self.name = f"http://{host}:{port}/stream"

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.split("?")[0] not in ("/stream", "/streamer/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                box = owner._add_box()
                try:
                    while not owner._stop.is_set():
                        f = box.get(0.5)
                        if f is None:
                            continue
                        hdr, body = f.pack()
                        ctype = {1: "image/jpeg", 2: "image/png"}.get(f.codec,
                                                                      "application/octet-stream")
                        head = (f"--frame\r\nContent-Type: {ctype}\r\n"
                                f"Content-Length: {len(body)}\r\n"
                                f"X-Scada-Header: {hdr.hex()}\r\n"
                                f"X-Codec: {CODEC_NAME.get(f.codec)}\r\n\r\n").encode()
                        self.wfile.write(head)
                        self.wfile.write(body)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except OSError:
                    pass
                finally:
                    owner._drop_box(box)

        self.srv = ThreadingHTTPServer((host, port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        super().close()
        self.srv.shutdown()


class MjpegReceiver(Receiver):
    """Разбор multipart вручную, со своим буфером: таймаут посреди чтения
    тогда не портит состояние, как у буферизованного файла поверх сокета."""

    def __init__(self, host, port, connect_timeout=10.0):
        self.name = f"http://{host}:{port}/stream"
        self.sock = _connect_retry(host, port, connect_timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _tune(self.sock)
        self.sock.sendall(f"GET /stream HTTP/1.1\r\nHost: {host}:{port}\r\n"
                          f"Connection: keep-alive\r\n\r\n".encode())
        self.buf = bytearray()
        self._fill_until(b"\r\n\r\n", 10.0)          # заголовок ответа HTTP
        i = self.buf.find(b"\r\n\r\n")
        if not self.buf.startswith(b"HTTP/1.") or b" 200 " not in self.buf[:i]:
            raise ConnectionError("источник ответил не 200")
        del self.buf[:i + 4]

    def _more(self, timeout):
        self.sock.settimeout(timeout)
        chunk = self.sock.recv(1 << 20)
        if not chunk:
            raise ConnectionError("поток оборвался")
        self.buf += chunk

    def _fill_until(self, marker, timeout):
        while self.buf.find(marker) < 0:
            self._more(timeout)

    def recv(self, timeout=1.0):
        t_end = time.time() + timeout
        try:
            while True:
                i = self.buf.find(b"--frame")
                j = self.buf.find(b"\r\n\r\n", i) if i >= 0 else -1
                if j >= 0:
                    break
                self._more(max(0.01, t_end - time.time()))
            headers = {}
            for ln in bytes(self.buf[i:j]).decode("latin-1").split("\r\n")[1:]:
                k, _, v = ln.partition(":")
                headers[k.strip().lower()] = v.strip()
            n = int(headers["content-length"])
            start = j + 4
            while len(self.buf) < start + n:
                # начатый кадр дочитываем, даже если таймаут вышел
                self._more(max(0.01, t_end - time.time()) + 5.0)
        except socket.timeout:
            return None
        body = bytes(self.buf[start:start + n])
        del self.buf[:start + n]
        f, _ = Frame.parse_header(bytes.fromhex(headers["x-scada-header"]))
        f.payload = body
        f.t_recv = clock()
        return f

    def close(self):
        self.sock.close()


# ───────────────────────────────────────────────────────────── фабрики

_SENDERS = {"tcp": TcpSender, "ws": WsSender, "udp": UdpSender,
            "pipe": PipeSender, "shm": ShmSender, "mjpeg": MjpegSender}
_RECEIVERS = {"tcp": TcpReceiver, "ws": WsReceiver, "udp": UdpReceiver,
              "pipe": PipeReceiver, "shm": ShmReceiver, "mjpeg": MjpegReceiver}


def make_sender(kind, host="127.0.0.1", port=8780, **kw):
    if kind not in _SENDERS:
        raise ValueError(f"неизвестный транспорт {kind}; есть: {', '.join(TRANSPORTS)}")
    return _SENDERS[kind](host, port, **kw)


def make_receiver(kind, host="127.0.0.1", port=8780, **kw):
    if kind not in _RECEIVERS:
        raise ValueError(f"неизвестный транспорт {kind}; есть: {', '.join(TRANSPORTS)}")
    return _RECEIVERS[kind](host, port, **kw)


# ───────────────────────────────────────────────────────────── статистика

class LinkStats:
    """Что известно о канале на стороне приёмника: частота, задержка,
    потери по разрывам номеров кадров, поток байтов."""

    def __init__(self, window=300):
        self.lat = deque(maxlen=window)
        self.age = deque(maxlen=window)
        self.times = deque(maxlen=window)
        self.sizes = deque(maxlen=window)
        self.frames = 0
        self.lost = 0
        self.last_seq = None

    def note(self, f):
        self.frames += 1
        if self.last_seq is not None:
            gap = (f.seq - self.last_seq) & 0xFFFFFFFF
            if 1 < gap < 1 << 31:
                self.lost += gap - 1
        self.last_seq = f.seq
        self.lat.append(f.latency_ms)
        self.age.append(f.age_ms)
        self.times.append(f.t_recv)
        self.sizes.append(len(f.payload))

    def summary(self):
        if not self.lat:
            return {"frames": 0}
        lat = sorted(self.lat)
        span = (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

        def pct(p):
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 2)

        return {
            "frames": self.frames,
            "fps": round((len(self.times) - 1) / span, 2) if span > 0 else 0.0,
            "lat_ms_p50": pct(0.5), "lat_ms_p95": pct(0.95), "lat_ms_max": round(lat[-1], 2),
            "age_ms_p50": round(sorted(self.age)[len(self.age) // 2], 2),
            "kb_per_frame": round(sum(self.sizes) / len(self.sizes) / 1024, 1),
            "mbit_s": round(sum(self.sizes) * 8 / span / 1e6, 1) if span > 0 else 0.0,
            "lost": self.lost,
            "loss_pct": round(100.0 * self.lost / max(1, self.lost + self.frames), 2),
        }
