from PySide6.QtCore import Qt, QTimer, QRect, QStandardPaths, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QInputDialog
import sys
from pathlib import Path


_W = 10
_H = 20
_F = "WolfCat.dat"

_A = {
    "a": (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
    ),
    "b": (
        ((1, 0), (2, 0), (1, 1), (2, 1)),
    ),
    "c": (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "d": (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
    ),
    "e": (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
    ),
    "f": (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    "g": (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
}

_C = {
    "a": QColor(80, 220, 235),
    "b": QColor(245, 210, 70),
    "c": QColor(170, 80, 220),
    "d": QColor(90, 200, 110),
    "e": QColor(225, 80, 80),
    "f": QColor(70, 100, 220),
    "g": QColor(235, 145, 60),
}

_K = b"W0lfC4t::silent-window::2026"


class _Q:
    __slots__ = ("k", "r", "x", "y")

    def __init__(self, k, r=0, x=3, y=-1):
        self.k = k
        self.r = r
        self.x = x
        self.y = y

    @property
    def p(self):
        return _A[self.k][self.r % len(_A[self.k])]


class _R:
    def __init__(self):
        self.s = 0x6D2B79F5

    def n(self):
        x = self.s & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s

    def pick(self, seq):
        return seq[self.n() % len(seq)]


_rng = _R()


def _hx(b):
    z = "0123456789abcdef"
    out = []
    for v in b:
        out.append(z[(v >> 4) & 15])
        out.append(z[v & 15])
    return "".join(out)


def _uhx(s):
    z = "0123456789abcdef"
    s = "".join(ch for ch in s.lower() if ch in z)
    if len(s) & 1:
        s = s[:-1]
    out = bytearray()
    for i in range(0, len(s), 2):
        out.append((z.index(s[i]) << 4) | z.index(s[i + 1]))
    return bytes(out)


def _mix(b):
    o = bytearray(len(b))
    lk = len(_K)
    for i, v in enumerate(b):
        o[i] = v ^ _K[(i * 7 + 3) % lk] ^ ((i * 29 + 17) & 0xFF)
    return bytes(o)


def _file():
    root = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
    return Path(root) / _F if root else None


def _read():
    try:
        with open(_file(), "r", encoding="ascii") as f:
            raw = f.read().strip()
        if not raw:
            return []
        data = _mix(_uhx(raw)).decode("utf-8")
        out = []
        for line in data.splitlines():
            if "|" not in line:
                continue
            n, s = line.rsplit("|", 1)
            try:
                out.append((n[:24], int(s)))
            except Exception:
                pass
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:10]
    except Exception:
        return []


def _write(rows):
    try:
        rows = sorted(rows, key=lambda x: x[1], reverse=True)[:10]
        text = "\n".join(f"{n.replace(chr(10), ' ').replace('|', '/')[:24]}|{int(s)}" for n, s in rows)
        enc = _hx(_mix(text.encode("utf-8")))
        path = _file()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="ascii") as f:
            f.write(enc)
    except Exception:
        pass


class WolfCat(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(300, 600)

        self.u = QTimer(self)
        self.u.timeout.connect(self._z)

        self.m = []
        self.q = None
        self.v = None
        self.s = 0
        self.l = 0
        self.n = 1
        self.h = False
        self.o = False
        self.j = False
        self._done = False

        self.reset()

    def reset(self):
        self.m = [[None for _ in range(_W)] for _ in range(_H)]
        self.s = 0
        self.l = 0
        self.n = 1
        self.h = False
        self.o = False
        self.j = False
        self._done = False
        self.q = self._new()
        self.v = self._new()
        self._clock()
        self.setFocus()
        self.update()

    def _new(self):
        return _Q(_rng.pick(tuple(_A.keys())))

    def _clock(self):
        self.u.start(max(90, 650 - (self.n - 1) * 55))

    def _ok(self, a, dx=0, dy=0, rr=None):
        r = a.r if rr is None else rr
        pts = _A[a.k][r % len(_A[a.k])]
        for cx, cy in pts:
            x = a.x + dx + cx
            y = a.y + dy + cy
            if x < 0 or x >= _W or y >= _H:
                return False
            if y >= 0 and self.m[y][x] is not None:
                return False
        return True

    def _mv(self, dx, dy):
        if self.o or self.h:
            return False
        if self._ok(self.q, dx, dy):
            self.q.x += dx
            self.q.y += dy
            self.update()
            return True
        return False

    def _rt(self):
        if self.o or self.h:
            return
        rr = (self.q.r + 1) % len(_A[self.q.k])
        for d in (0, -1, 1, -2, 2):
            if self._ok(self.q, d, 0, rr):
                self.q.x += d
                self.q.r = rr
                self.update()
                return

    def _drop(self):
        if self.o or self.h:
            return
        c = 0
        while self._mv(0, 1):
            c += 1
        self.s += c * 2
        self._seal()

    def _z(self):
        if self.o or self.h:
            return
        if not self._mv(0, 1):
            self._seal()

    def _seal(self):
        for cx, cy in self.q.p:
            x = self.q.x + cx
            y = self.q.y + cy
            if y < 0:
                self.o = True
                self.u.stop()
                self._finish()
                self.update()
                return
            self.m[y][x] = self.q.k

        self._trim()
        self.q = self.v
        self.q.x, self.q.y, self.q.r = 3, -1, 0
        self.v = self._new()

        if not self._ok(self.q):
            self.o = True
            self.u.stop()
            self._finish()

        self.update()

    def _trim(self):
        keep = [row for row in self.m if any(v is None for v in row)]
        c = _H - len(keep)
        if not c:
            return

        for _ in range(c):
            keep.insert(0, [None for _ in range(_W)])

        self.m = keep
        self.s += {1: 100, 2: 300, 3: 500, 4: 800}.get(c, 0) * self.n
        self.l += c

        old = self.n
        self.n = self.l // 10 + 1
        if old != self.n:
            self._clock()

    def _finish(self):
        if self._done:
            return
        self._done = True

        rows = _read()
        pos = len(rows) < 10 or self.s > rows[-1][1]
        if not pos:
            return

        name, ok = QInputDialog.getText(self, "Result", "Name:")
        if not ok:
            name = "anonymous"
        name = name.strip() or "anonymous"

        rows.append((name, self.s))
        _write(rows)

    def pause(self):
        if self.o:
            return
        self.h = not self.h
        if self.h:
            self.u.stop()
        else:
            self._clock()
        self.update()

    def scores(self):
        return _read()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Left:
            self._mv(-1, 0)
        elif k == Qt.Key_Right:
            self._mv(1, 0)
        elif k == Qt.Key_Down:
            if self._mv(0, 1):
                self.s += 1
        elif k == Qt.Key_Up:
            self._rt()
        elif k == Qt.Key_Space:
            self._drop()
        elif k == Qt.Key_P:
            self.pause()
        elif k == Qt.Key_R:
            self.reset()
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(22, 22, 26))

        mg = 14
        top = 42
        bot = 30
        aw = self.width() - mg * 2
        ah = self.height() - mg * 2 - top - bot
        c = max(8, min(aw // _W, ah // _H))
        bw = c * _W
        bh = c * _H
        ox = (self.width() - bw) // 2
        oy = mg + top

        p.setPen(QColor(215, 215, 220))
        f = QFont()
        f.setPointSize(10)
        p.setFont(f)
        p.drawText(
            QRect(mg, 8, self.width() - mg * 2, 28),
            Qt.AlignCenter,
            f"{self.s}    {self.l}    {self.n}",
        )

        p.fillRect(ox, oy, bw, bh, QColor(10, 10, 13))
        p.setPen(QPen(QColor(36, 36, 42), 1))

        for x in range(_W + 1):
            px = ox + x * c
            p.drawLine(px, oy, px, oy + bh)
        for y in range(_H + 1):
            py = oy + y * c
            p.drawLine(ox, py, ox + bw, py)

        for y, row in enumerate(self.m):
            for x, k in enumerate(row):
                if k:
                    self._cell(p, ox, oy, x, y, c, k)

        if self.q and not self.o:
            yy = self.q.y
            while True:
                old = self.q.y
                self.q.y = yy
                ok = self._ok(self.q, 0, 1)
                self.q.y = old
                if not ok:
                    break
                yy += 1

            for cx, cy in self.q.p:
                gx = self.q.x + cx
                gy = yy + cy
                if gy >= 0:
                    r = QRect(ox + gx * c + 2, oy + gy * c + 2, c - 4, c - 4)
                    q = QColor(_C[self.q.k])
                    q.setAlpha(55)
                    p.fillRect(r, q)

            for cx, cy in self.q.p:
                x = self.q.x + cx
                y = self.q.y + cy
                if y >= 0:
                    self._cell(p, ox, oy, x, y, c, self.q.k)

        if self.h or self.o:
            p.fillRect(ox, oy, bw, bh, QColor(0, 0, 0, 150))
            p.setPen(QColor(245, 245, 245))
            f = QFont()
            f.setPointSize(18)
            f.setBold(True)
            p.setFont(f)
            p.drawText(
                QRect(ox, oy, bw, bh),
                Qt.AlignCenter,
                "PAUSE" if self.h else "END\nR",
            )

        rows = _read()
        if rows:
            f = QFont()
            f.setPointSize(8)
            p.setFont(f)
            p.setPen(QColor(165, 165, 170))
            txt = "   ".join(f"{i+1}:{n} {s}" for i, (n, s) in enumerate(rows[:3]))
            p.drawText(
                QRect(mg, oy + bh + 3, self.width() - mg * 2, 24),
                Qt.AlignCenter,
                txt,
            )

    def _cell(self, p, ox, oy, x, y, c, k):
        col = _C[k]
        r = QRect(ox + x * c + 1, oy + y * c + 1, c - 2, c - 2)
        p.fillRect(r, col)

        hi = QColor(
            min(255, col.red() + 45),
            min(255, col.green() + 45),
            min(255, col.blue() + 45),
        )
        lo = QColor(
            max(0, col.red() - 55),
            max(0, col.green() - 55),
            max(0, col.blue() - 55),
        )

        p.setPen(hi)
        p.drawLine(r.topLeft(), r.topRight())
        p.drawLine(r.topLeft(), r.bottomLeft())
        p.setPen(lo)
        p.drawLine(r.bottomLeft(), r.bottomRight())
        p.drawLine(r.topRight(), r.bottomRight())


class WolfCatWindow(QMainWindow):
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("WolfCat")
        self.setCentralWidget(WolfCat(self))
        self.resize(420, 760)

    def closeEvent(self, event):
        self.centralWidget().u.stop()
        self.closed.emit()
        super().closeEvent(event)


def open_wolfcat(parent=None):
    app = QApplication.instance()
    own = app is None

    if own:
        app = QApplication(sys.argv)
        app.setApplicationName("WolfCat")

    owner = parent if parent is not None else app
    w = getattr(owner, "_wc_window", None)
    if w is None:
        w = WolfCatWindow(parent)
        owner._wc_window = w

        def release():
            if getattr(owner, "_wc_window", None) is w:
                owner._wc_window = None

        w.closed.connect(release)
        w.destroyed.connect(release)

    if w.isMinimized():
        w.showNormal()
    else:
        w.show()
    w.raise_()
    w.activateWindow()
    w.centralWidget().setFocus()

    if own:
        return app.exec()

    return w


if __name__ == "__main__":
    open_wolfcat()
