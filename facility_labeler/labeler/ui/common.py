"""화면 여러 곳에서 같이 쓰는 작은 부품들."""
import os
import subprocess
import sys

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)


class Worker(QThread):
    """오래 걸리는 작업(추출, 학습, 내보내기)을 화면이 멈추지 않게 뒤에서 실행합니다.

    fn 은 progress(현재, 전체, 메시지), log(문장), should_stop() 를 인자로 받는 함수입니다.
    """
    progress = Signal(int, int, str)
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.stop_requested = False

    def run(self):
        try:
            result = self.fn(progress=lambda d, t, m="": self.progress.emit(int(d), int(t), m),
                             log=self.log.emit, should_stop=lambda: self.stop_requested)
            self.done.emit(result)
        except Exception as e:  # 오류는 화면 로그에 보여줌
            import traceback
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))


class RunPanel(QWidget):
    """진행 막대 + 중지 버튼 + 로그 창."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setFormat("%v / %m")
        self.msg = QLabel("")
        self.stop_btn = QPushButton("중지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.bar, 1)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        lay.addWidget(self.msg)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)
        self.worker = None

    def start(self, fn, on_done=None):
        """fn 을 뒤에서 실행하고 진행 상황을 이 패널에 표시합니다."""
        self.worker = Worker(fn)
        self.worker.progress.connect(self._progress)
        self.worker.log.connect(self.text.appendPlainText)
        self.worker.failed.connect(lambda m: self.text.appendPlainText(f"[오류] {m}"))
        self.worker.finished.connect(lambda: self.stop_btn.setEnabled(False))
        if on_done:
            self.worker.done.connect(on_done)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _progress(self, done, total, msg):
        """진행 막대 + 남은 시간 예상 (지금까지 속도로 계산, 단계가 바뀌면 다시 계산)."""
        import time
        now = time.time()
        if total != getattr(self, "_total", None) or done < getattr(self, "_done", 0):
            self._t0, self._d0 = now, done          # 새 단계 시작 (예: 박스 AI → 키포인트 AI)
        self._total, self._done = total, done
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(done)
        left = ""
        if done > self._d0 and now - self._t0 > 5:
            sec = (now - self._t0) / (done - self._d0) * (total - done)
            left = f"   (이 단계 남은 시간 약 {sec / 60:.0f}분)" if sec >= 60 else f"   (이 단계 남은 시간 약 {sec:.0f}초)"
        self.msg.setText(msg + left)

    def _stop(self):
        if self.worker:
            self.worker.stop_requested = True
            self.text.appendPlainText("중지 요청됨 (현재 단계가 끝나면 정지)")


def text_label(text="", rich=True):
    """줄바꿈되는 글자. 칸이 좁아져도 글자가 잘리지 않게 세로 크기를 지킵니다."""
    lb = QLabel(text)
    lb.setWordWrap(True)
    if rich:
        lb.setTextFormat(Qt.RichText)
    lb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    return lb


def note(text):
    """회색 안내 문구."""
    lb = text_label(text)
    lb.setObjectName("note")
    return lb


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)])


STYLE = """
QWidget { font-size: 10pt; }
QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 6px; margin-top: 12px; padding: 10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel#note { color: #666; font-weight: normal; }
QLabel#big { font-size: 20pt; font-weight: bold; }
QPushButton { padding: 6px 12px; border: 1px solid #b0b0b0; border-radius: 4px; background: #f7f7f7; }
QPushButton:hover { background: #e3f2fd; }
QPushButton:disabled { color: #999; }
QPushButton#primary { background: #1976d2; color: white; border-color: #0d47a1; font-weight: bold; }
QPushButton#primary:disabled { background: #90caf9; }
QPushButton:checked { background: #1976d2; color: white; border-color: #0d47a1; font-weight: bold; }
"""
