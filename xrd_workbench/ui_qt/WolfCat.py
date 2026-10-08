from PySide6.QtCore import Qt, QTimer, QRect, QStandardPaths, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QInputDialog
import sys
from pathlib import Path


_W = 18
_H = 26
_F = "WolfCat.dat"
_K = b"W0lfC4t::silent-window::2026"


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
        text = "\n".join(
            f"{n.replace(chr(10), ' ').replace('|', '/')[:24]}|{int(s)}"
            for n, s in rows
        )
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
        self.q = (1, 0)
        self.v = (1, 0)
        self.a = None
        self.s = 0
        self.l = 0
        self.n = 1
        self.h = False
        self.o = False
        self.j = False
        self._done = False

        self.reset()

    def reset(self):
        cx = _W // 2
        cy = _H // 2
        self.m = [(cx - i, cy) for i in range(5)]
        self.q = (1, 0)
        self.v = (1, 0)
        self.a = None
        self.s = 0
        self.l = 0
        self.n = 1
        self.h = False
        self.o = False
        self.j = False
        self._done = False
        self._new()
        self._clock()
        self.setFocus()
        self.update()

    def _new(self):
        occupied = set(self.m)
        free = [
            (x, y)
            for y in range(_H)
            for x in range(_W)
            if (x, y) not in occupied
        ]
        if not free:
            self.o = True
            self.u.stop()
            self._finish()
            return
        self.a = _rng.pick(free)

    def _clock(self):
        self.u.start(max(65, 175 - (self.n - 1) * 12))

    def _turn(self, dx, dy):
        if self.o or self.h:
            return
        if (dx, dy) == (-self.q[0], -self.q[1]):
            return
        self.v = (dx, dy)

    def _z(self):
        if self.o or self.h:
            return

        self.q = self.v
        hx, hy = self.m[0]
        nx = hx + self.q[0]
        ny = hy + self.q[1]
        head = (nx, ny)
        grow = head == self.a
        body = self.m if grow else self.m[:-1]

        if nx < 0 or nx >= _W or ny < 0 or ny >= _H or head in body:
            self.o = True
            self.u.stop()
            self._finish()
            self.update()
            return

        self.m.insert(0, head)
        if grow:
            self.l += 1
            old = self.n
            self.n = self.l // 5 + 1
            self.s += 10 * self.n
            self._new()
            if self.n != old:
                self._clock()
        else:
            self.m.pop()

        self.update()

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
            self._turn(-1, 0)
        elif k == Qt.Key_Right:
            self._turn(1, 0)
        elif k == Qt.Key_Up:
            self._turn(0, -1)
        elif k == Qt.Key_Down:
            self._turn(0, 1)
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
        p.setPen(QPen(QColor(31, 31, 37), 1))
        for x in range(_W + 1):
            px = ox + x * c
            p.drawLine(px, oy, px, oy + bh)
        for y in range(_H + 1):
            py = oy + y * c
            p.drawLine(ox, py, ox + bw, py)

        if self.a is not None:
            self._mark(p, ox, oy, self.a[0], self.a[1], c)

        for i, (x, y) in enumerate(reversed(self.m)):
            self._cell(p, ox, oy, x, y, c, i == len(self.m) - 1)

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

    def _cell(self, p, ox, oy, x, y, c, head=False):
        col = QColor(86, 188, 132) if head else QColor(54, 132, 96)
        r = QRect(ox + x * c + 2, oy + y * c + 2, c - 4, c - 4)
        p.fillRect(r, col)

        p.setPen(QColor(122, 221, 163) if head else QColor(78, 164, 119))
        p.drawLine(r.topLeft(), r.topRight())
        p.drawLine(r.topLeft(), r.bottomLeft())
        p.setPen(QColor(34, 88, 65))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        p.drawLine(r.topRight(), r.bottomRight())

        if head:
            d = max(2, c // 6)
            ey = r.top() + max(2, c // 5)
            if self.q[0] != 0:
                ex = r.right() - d - 1 if self.q[0] > 0 else r.left() + 2
                p.fillRect(QRect(ex, ey, d, d), QColor(240, 240, 235))
                p.fillRect(QRect(ex, r.bottom() - d - max(2, c // 5), d, d), QColor(240, 240, 235))
            else:
                ex1 = r.left() + max(2, c // 5)
                ex2 = r.right() - d - max(2, c // 5)
                yy = r.bottom() - d - 1 if self.q[1] > 0 else r.top() + 2
                p.fillRect(QRect(ex1, yy, d, d), QColor(240, 240, 235))
                p.fillRect(QRect(ex2, yy, d, d), QColor(240, 240, 235))

    def _mark(self, p, ox, oy, x, y, c):
        cx = ox + x * c + c // 2
        cy = oy + y * c + c // 2
        r = max(3, c // 3)
        col = QColor(208, 118, 214)
        p.setPen(QPen(QColor(244, 184, 248), 1))
        p.setBrush(col)
        p.drawPolygon([
            type(QRect().topLeft())(cx, cy - r),
            type(QRect().topLeft())(cx + r, cy),
            type(QRect().topLeft())(cx, cy + r),
            type(QRect().topLeft())(cx - r, cy),
        ])


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
