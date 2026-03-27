from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget


class BoxSelectOverlay(QWidget):
    """框选时在 glwidget 上绘制拖拽矩形的透明覆盖层"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._start = None
        self._end = None

    def set_rect(self, start, end):
        self._start = start
        self._end = end
        self.update()

    def clear_rect(self):
        self._start = None
        self._end = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._start is None or self._end is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 120, 255), 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        x1, y1 = self._start
        x2, y2 = self._end
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        painter.drawRect(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        painter.end()

