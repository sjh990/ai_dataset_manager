"""라벨링 기준 예시 이미지 창.

예시 이미지는 클래스별 폴더에 저장: examples/<클래스 키>/*.png  (예: examples/sign/example_1.png)
"""
import shutil
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy,
                               QVBoxLayout)

from .. import config as C
from .common import note

EXAMPLES_DIR = C.APP_DIR / "examples"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def example_files(cls):
    d = EXAMPLES_DIR / C.CLASS_KEYS[cls]
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS) if d.exists() else []


class ExampleDialog(QDialog):
    """클래스별 예시 이미지 보기, 추가(파일 / 현재 화면), 삭제."""

    def __init__(self, cls, canvas=None, parent=None):
        super().__init__(parent)
        self.cls, self.canvas, self.index = cls, canvas, 0
        c = C.CLASSES[cls]
        self.setWindowTitle(f"라벨링 예시: {c['name']}")
        self.resize(900, 820)
        lay = QVBoxLayout(self)
        self.title = QLabel("")
        lay.addWidget(self.title)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setStyleSheet("background:#222;")
        lay.addWidget(self.view, 1)
        nav = QHBoxLayout()
        b_prev, b_next = QPushButton("◀ 이전"), QPushButton("다음 ▶")
        b_prev.clicked.connect(lambda: self.move(-1))
        b_next.clicked.connect(lambda: self.move(1))
        b_file = QPushButton("이미지 파일 추가...")
        b_file.clicked.connect(self.add_file)
        b_cap = QPushButton("지금 라벨링 화면 추가")
        b_cap.clicked.connect(self.add_capture)
        b_cap.setEnabled(canvas is not None)
        b_del = QPushButton("이 예시 삭제")
        b_del.clicked.connect(self.delete)
        for b in (b_prev, b_next):
            nav.addWidget(b)
        nav.addStretch(1)
        for b in (b_file, b_cap, b_del):
            nav.addWidget(b)
        lay.addLayout(nav)
        lay.addWidget(note(f"<b>저장 위치</b>: examples\\{c['key']}\\ (팀원과 폴더째 공유 가능)<br>"
                           "<b>지금 라벨링 화면 추가</b>: 라벨링 화면에 보이는 그대로 (박스, 키포인트 포함) 저장"))
        self.refresh()

    def refresh(self):
        files = example_files(self.cls)
        name = C.CLASSES[self.cls]["name"]
        if not files:
            self.title.setText(f"<b>{name}</b> 예시 없음: '이미지 파일 추가' 또는 '지금 라벨링 화면 추가'")
            self.view.clear()
            return
        self.index %= len(files)
        f = files[self.index]
        self.title.setText(f"<b>{name}</b> 예시 {self.index + 1} / {len(files)}  <span style='color:#666'>{f.name}</span>")
        pix = QPixmap(str(f))
        self.view.setPixmap(pix.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()

    def move(self, step):
        self.index += step
        self.refresh()

    def _new_path(self, suffix=".png"):
        d = EXAMPLES_DIR / C.CLASS_KEYS[self.cls]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"example_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"

    def add_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "예시 이미지 선택", "", "이미지 (*.png *.jpg *.jpeg *.bmp)")
        for i, f in enumerate(files):
            dst = self._new_path(Path(f).suffix.lower())
            dst = dst.with_name(dst.stem + f"_{i + 1}" + dst.suffix)
            shutil.copy2(f, dst)
        if files:
            self.index = len(example_files(self.cls)) - 1
            self.refresh()

    def add_capture(self):
        if self.canvas is None:
            return
        self.canvas.grab().save(str(self._new_path()))
        self.index = len(example_files(self.cls)) - 1
        self.refresh()

    def delete(self):
        files = example_files(self.cls)
        if not files:
            return
        f = files[self.index]
        if QMessageBox.question(self, "예시 삭제", f"이 예시 이미지를 삭제할까요?\n{f.name}") == QMessageBox.Yes:
            f.unlink()
            self.refresh()
