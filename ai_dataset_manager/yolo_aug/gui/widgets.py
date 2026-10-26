"""공용 위젯: 파라미터 폼, 이미지 뷰, 분할 비율 바, 백그라운드 작업 스레드."""
from __future__ import annotations

import traceback

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QLabel,
                               QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from ..ops_spec import Param


class ParamForm(QWidget):
    changed = Signal()

    def __init__(self, params: list[Param], values: dict | None = None, parent=None):
        super().__init__(parent)
        self.params = params
        self.widgets: dict[str, QWidget] = {}
        lay = QFormLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for p in params:
            w = self._make(p)
            if p.tip:
                w.setToolTip(p.tip)
            self.widgets[p.key] = w
            lab = QLabel(p.label)
            if p.tip:
                lab.setToolTip(p.tip)
            lay.addRow(lab, w)
        if values:
            self.set_values(values)

    def _make(self, p: Param) -> QWidget:
        if p.kind == "float":
            w = QDoubleSpinBox()
            step = p.step or 0.1
            dec = max(0, min(6, len(f"{step:.10f}".rstrip("0").split(".")[1]) if "." in f"{step:.10f}".rstrip("0") else 0))
            w.setDecimals(max(dec, 1))
            w.setRange(p.min if p.min is not None else -1e9, p.max if p.max is not None else 1e9)
            w.setSingleStep(step)
            w.setValue(float(p.default))
            if p.suffix:
                w.setSuffix(" " + p.suffix)
            w.valueChanged.connect(self.changed)
        elif p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.min if p.min is not None else -10**9), int(p.max if p.max is not None else 10**9))
            w.setSingleStep(int(p.step or 1))
            w.setValue(int(p.default))
            if p.suffix:
                w.setSuffix(" " + p.suffix)
            w.valueChanged.connect(self.changed)
        elif p.kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in p.choices])
            w.setCurrentText(str(p.default))
            w.currentTextChanged.connect(self.changed)
        else:
            w = QCheckBox()
            w.setChecked(bool(p.default))
            w.toggled.connect(self.changed)
        w.setMinimumWidth(130)
        return w

    def values(self) -> dict:
        out = {}
        for p in self.params:
            w = self.widgets[p.key]
            if isinstance(w, QDoubleSpinBox):
                out[p.key] = float(w.value())
            elif isinstance(w, QSpinBox):
                out[p.key] = int(w.value())
            elif isinstance(w, QComboBox):
                out[p.key] = w.currentText()
            else:
                out[p.key] = bool(w.isChecked())
        return out

    def set_values(self, vals: dict):
        for k, v in vals.items():
            w = self.widgets.get(k)
            if w is None:
                continue
            w.blockSignals(True)
            if isinstance(w, QDoubleSpinBox):
                w.setValue(float(v))
            elif isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(v))
            else:
                w.setChecked(bool(v))
            w.blockSignals(False)
        self.changed.emit()


def np_to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class ImageView(QLabel):
    """비율을 유지하며 크기에 맞게 이미지를 보여주는 라벨."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("imageView")
        self.setText(title or "미리보기 없음")

    def set_image(self, img: np.ndarray | None):
        if img is None:
            self._pix = None
            self.setText("미리보기 없음")
            return
        self._pix = np_to_pixmap(img)
        self._update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update()

    def _update(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size() * 0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class CaptionedImage(QWidget):
    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.view = ImageView()
        self.caption = QLabel(caption)
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setObjectName("caption")
        lay.addWidget(self.view, 1)
        lay.addWidget(self.caption)

    def set(self, img, caption: str | None = None):
        self.view.set_image(img)
        if caption is not None:
            self.caption.setText(caption)


class RatioBar(QWidget):
    COLORS = [QColor("#3b82f6"), QColor("#f59e0b"), QColor("#10b981")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = [70, 20, 10]
        self.counts = [0, 0, 0]
        self.setMinimumHeight(38)

    def set(self, values, counts):
        self.values, self.counts = list(values), list(counts)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        total = max(sum(self.values), 1)
        x = r.left()
        names = ["train", "val", "test"]
        for i, v in enumerate(self.values):
            w = int(r.width() * v / total) if i < 2 else r.right() - x
            if v <= 0:
                continue
            p.setBrush(self.COLORS[i])
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(x, r.top(), w, r.height(), 4, 4)
            p.setPen(Qt.white)
            txt = f"{names[i]} {v}% ({self.counts[i]})"
            if p.fontMetrics().horizontalAdvance(txt) > w - 6:
                txt = f"{v}%"
            p.drawText(x, r.top(), w, r.height(), Qt.AlignCenter, txt)
            x += w
        p.end()


class HLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setObjectName("hline")


class FuncThread(QThread):
    """임의 함수를 백그라운드에서 실행."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, **self.kw))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
