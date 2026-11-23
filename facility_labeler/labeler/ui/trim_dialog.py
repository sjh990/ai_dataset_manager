"""영상 구간 자르기 창.

영상을 보면서 쓸 구간의 시작과 끝을 정합니다. 앞뒤 필요 없는 부분(이륙, 착륙, 이동 등)은 뽑지 않습니다.
영상 파일은 수정하지 않고, 프레임을 뽑을 때 이 구간만 사용합니다.
"""
import shutil

import cv2
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from ..frames import _ascii_path, video_info
from .common import note, text_label


def mmss(sec):
    """초 → '분:초.1' 글자 (예: 125.5 → 2:05.5)."""
    sec = round(sec, 1)
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


class RangeBar(QWidget):
    """영상 전체 막대. 파란 부분 = 쓸 구간, 빨간 선 = 지금 위치. 클릭하거나 끌면 그 위치로 이동."""
    moved = Signal(int)

    def __init__(self, n_frames):
        super().__init__()
        self.n = max(1, n_frames)
        self.pos, self.start, self.end = 0, 0, self.n - 1
        self.setMinimumHeight(34)
        self.setCursor(Qt.PointingHandCursor)

    def x_of(self, f):
        return 8 + (self.width() - 16) * f / max(1, self.n - 1)

    def paintEvent(self, _):
        p = QPainter(self)
        h = self.height()
        p.fillRect(QRectF(8, h / 2 - 6, self.width() - 16, 12), QColor("#d0d0d0"))
        x1, x2 = self.x_of(self.start), self.x_of(self.end)
        p.fillRect(QRectF(x1, h / 2 - 6, max(2, x2 - x1), 12), QColor("#1e88e5"))
        x = self.x_of(self.pos)
        p.fillRect(QRectF(x - 1.5, 2, 3, h - 4), QColor("#e53935"))

    def _seek(self, e):
        f = round((e.position().x() - 8) / max(1, self.width() - 16) * (self.n - 1))
        self.moved.emit(min(max(f, 0), self.n - 1))

    def mousePressEvent(self, e):
        self._seek(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._seek(e)


class TrimDialog(QDialog):
    """사용법: dlg = TrimDialog(영상경로, 시작초, 끝초); if dlg.exec(): dlg.start_sec, dlg.end_sec

    every_sec=None: 영상 추론용 (몇 장 대신 프레임 수 표시)
    """

    def __init__(self, video_path, start_sec=0.0, end_sec=None, every_sec=0.5, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"구간 자르기: {video_path}")
        self.resize(1200, 820)
        info = video_info(video_path)
        if info is None:
            raise RuntimeError(f"영상 열기 실패: {video_path}")
        self.fps, self.n, self.every = info["fps"], info["frames"], every_sec
        path, self.tmp = _ascii_path(video_path)
        self.cap = cv2.VideoCapture(path)
        self.cur = -1                     # 지금 보이는 프레임 번호
        self.start = int(round(start_sec * self.fps))
        self.end = self.n - 1 if end_sec is None else min(self.n - 1, int(round(end_sec * self.fps)))

        lay = QVBoxLayout(self)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setStyleSheet("background:#222;")
        lay.addWidget(self.view, 1)

        self.bar = RangeBar(self.n)
        self.bar.start, self.bar.end = self.start, self.end
        self.bar.moved.connect(self.seek_later)
        lay.addWidget(self.bar)
        self.time = QLabel("")
        lay.addWidget(self.time)

        row = QHBoxLayout()
        self.play_btn = QPushButton("재생 (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        row.addWidget(self.play_btn)
        for text, sec in (("◀ 10초", -10), ("◀ 1초", -1), ("1초 ▶", 1), ("10초 ▶", 10)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=sec: self.show_frame(self.cur + round(s * self.fps)))
            row.addWidget(b)
        row.addSpacing(20)
        b_s = QPushButton("여기를 시작으로 (S)")
        b_s.clicked.connect(self.set_start)
        b_e = QPushButton("여기를 끝으로 (E)")
        b_e.clicked.connect(self.set_end)
        b_all = QPushButton("전체로 되돌리기")
        b_all.clicked.connect(self.reset)
        for b in (b_s, b_e, b_all):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        for i in range(row.count()):                 # 버튼이 Space 키를 가로채지 않게 (Space = 재생)
            w = row.itemAt(i).widget()
            if w:
                w.setFocusPolicy(Qt.NoFocus)

        self.range_text = text_label("")
        lay.addWidget(self.range_text)
        lay.addWidget(note("<b>이동</b>: 막대 클릭 또는 끌기<br>"
                           "<b>구간 지정</b>: 쓸 구간 처음에서 '여기를 시작으로', 마지막에서 '여기를 끝으로'<br>"
                           "<b>단축키</b>: Space 재생/정지, ← → 1초, Shift + ← → 10초, S 시작, E 끝<br>"
                           + ("<b>결과</b>: 프레임 추출 때 파란 구간만 사용 (영상 파일은 그대로)" if every_sec else
                              "<b>결과</b>: 영상 추론 때 파란 구간만 추론 (영상 파일은 그대로)")))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("이 구간으로 정하기")
        buttons.button(QDialogButtonBox.Cancel).setText("취소")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_step)
        self.seek_timer = QTimer(self)          # 막대를 끄는 동안 너무 자주 읽지 않게 잠깐 기다렸다 읽음
        self.seek_timer.setSingleShot(True)
        self.seek_timer.timeout.connect(lambda: self.show_frame(self.pending))
        self.pending = 0
        self.update_range()
        QTimer.singleShot(0, lambda: self.show_frame(self.start))

    # ---------- 결과 ----------
    @property
    def start_sec(self):
        return self.start / self.fps

    @property
    def end_sec(self):
        return None if self.end >= self.n - 1 else self.end / self.fps    # 끝까지면 None

    # ---------- 화면 ----------
    def seek_later(self, f):
        self.pending = f
        self.bar.pos = f
        self.bar.update()
        self.seek_timer.start(120)

    def show_frame(self, f, img=None):
        f = min(max(int(f), 0), self.n - 1)
        if img is None:
            if f != self.cur + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = self.cap.read()
            if not ok:
                return False
        self.cur = f
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.view.setPixmap(pix.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.bar.pos = f
        self.bar.update()
        self.time.setText(f"지금 위치 <b>{mmss(f / self.fps)}</b> / {mmss(self.n / self.fps)}"
                          f"  (프레임 {f} / {self.n - 1})")
        return True

    def update_range(self):
        self.bar.start, self.bar.end = self.start, self.end
        self.bar.update()
        length = (self.end - self.start) / self.fps
        if self.every:
            step = max(1, round(self.every * self.fps))
            count = len(range(-(-self.start // step) * step, self.end + 1, step))
            amount = f"{self.every}초 간격이면 약 <b>{count}장</b>"
        else:
            amount = f"<b>{self.end - self.start + 1}프레임</b> 추론"
        self.range_text.setText(f"쓸 구간: <b>{mmss(self.start / self.fps)} ~ {mmss(self.end / self.fps)}</b>  "
                                f"(길이 {length:.1f}초, {amount})  "
                                f"<span style='color:#666'>영상 전체 {self.n / self.fps:.1f}초</span>")

    # ---------- 버튼 ----------
    def set_start(self):
        self.start = self.cur
        if self.end <= self.start:
            self.end = self.n - 1
        self.update_range()

    def set_end(self):
        self.end = self.cur
        if self.start >= self.end:
            self.start = 0
        self.update_range()

    def reset(self):
        self.start, self.end = 0, self.n - 1
        self.update_range()

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("재생 (Space)")
        else:
            self.play_timer.start(max(10, int(1000 / self.fps)))
            self.play_btn.setText("정지 (Space)")

    def play_step(self):
        ok, img = self.cap.read()                # 다음 프레임을 차례로 읽음
        if not ok or not self.show_frame(self.cur + 1, img):
            self.toggle_play()

    def keyPressEvent(self, e):
        k = e.key()
        big = e.modifiers() & Qt.ShiftModifier
        if k == Qt.Key_Space:
            self.toggle_play()
        elif k in (Qt.Key_Left, Qt.Key_Right):
            sec = (10 if big else 1) * (1 if k == Qt.Key_Right else -1)
            self.show_frame(self.cur + round(sec * self.fps))
        elif k == Qt.Key_S:
            self.set_start()
        elif k == Qt.Key_E:
            self.set_end()
        else:
            super().keyPressEvent(e)

    def done(self, r):
        self.play_timer.stop()
        self.cap.release()
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
        super().done(r)
