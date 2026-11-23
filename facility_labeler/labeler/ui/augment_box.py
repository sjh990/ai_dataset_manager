"""증강 설정 칸 (3번 AI 학습, 4번 내보내기 탭 공용) + 미리보기 창."""
import random

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QSpinBox, QVBoxLayout)

from .. import augment as A
from .. import config as C
from ..frames import read_image
from .common import note, text_label


class AugmentBox(QGroupBox):
    """증강 장수, 프리셋(약하게/보통/강하게/직접 설정), 옵션별 켜기와 세기, 미리보기."""

    def __init__(self, main, title="증강", copies=1, copies_note="원본 1장당 추가로 만들 장수"):
        super().__init__(title)
        self.main = main
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.copies = QSpinBox()
        self.copies.setRange(0, 10)
        self.copies.setValue(copies)
        self.copies.setSuffix(" 장")
        self.copies.setToolTip(copies_note)
        self.preset = QComboBox()
        for name in A.PRESETS:
            self.preset.addItem(name)
        self.preset.addItem("직접 설정")
        self.preset.setCurrentText("보통")
        self.preset.currentTextChanged.connect(self.apply_preset)
        b_prev = QPushButton("미리보기...")
        b_prev.clicked.connect(self.preview)
        top.addWidget(QLabel("증강 장수"))
        top.addWidget(self.copies)
        top.addSpacing(12)
        top.addWidget(QLabel("세기"))
        top.addWidget(self.preset, 1)
        top.addWidget(b_prev)
        lay.addLayout(top)

        grid = QGridLayout()
        self.rows = {}
        for r, (key, (name, why, unit, hi, _)) in enumerate(A.OPTIONS.items()):
            cb = QCheckBox(name)
            sp = QSpinBox()
            sp.setRange(1, hi)
            if unit.startswith("±"):                  # 예: ±20 %
                sp.setPrefix("±")
                sp.setSuffix(f" {unit[1:]}")
            elif unit.startswith("최대"):              # 예: 최대 9 px
                sp.setPrefix("최대 ")
                sp.setSuffix(f" {unit[3:]}")
            elif unit != "켜기 1":
                sp.setSuffix(f" {unit}")
            cb.toggled.connect(sp.setEnabled)
            cb.toggled.connect(self._to_custom)
            sp.valueChanged.connect(self._to_custom)
            lb = QLabel(why)
            lb.setObjectName("note")
            grid.addWidget(cb, r // 2, (r % 2) * 3)
            grid.addWidget(sp, r // 2, (r % 2) * 3 + 1)
            grid.addWidget(lb, r // 2, (r % 2) * 3 + 2)
            if key == "flip":
                sp.hide()
            self.rows[key] = (cb, sp)
        lay.addLayout(grid)
        lay.addWidget(note("<b>증강 장수</b>: 원본 1장당 추가로 만드는 증강 이미지 수 (0 = 증강 없음)<br>"
                           "<b>세기</b>: 약하게 / 보통 / 강하게, 옵션을 직접 바꾸면 '직접 설정'<br>"
                           "<b>색, 화질 효과</b> (모션 블러, 흐림, 안개, 그림자, 노이즈): 장마다 50% 확률로 적용<br>"
                           "<b>위아래 반전</b>: 사용 안 함 (실제 드론 영상에 없는 장면)"))
        self._loading = False
        self.apply_preset("보통")

    # ---------- 설정 ----------
    def apply_preset(self, name):
        if name not in A.PRESETS:
            return
        self._loading = True
        for key, v in A.preset(name).items():
            cb, sp = self.rows[key]
            cb.setChecked(v > 0)
            sp.setEnabled(v > 0)
            if v > 0:
                sp.setValue(v)
        self._loading = False

    def _to_custom(self, *_):
        if not self._loading and self.preset.currentText() != "직접 설정":
            self.preset.blockSignals(True)
            self.preset.setCurrentText("직접 설정")
            self.preset.blockSignals(False)

    def settings(self):
        return {k: (sp.value() if k != "flip" else 1) if cb.isChecked() else 0 for k, (cb, sp) in self.rows.items()}

    # ---------- 미리보기 ----------
    def preview(self):
        p = self.main.project
        items = [it for it in (p.human_images() if p else []) if it[1]["objects"]]
        if not items:
            QMessageBox.information(self, "미리보기", "시설물이 있는 '사람 검수 완료' 이미지가 있어야 미리보기 가능")
            return
        PreviewDialog(items, self.settings(), self.copies.value(), self).exec()


def draw(img, objects):
    out = img.copy()
    for o in objects:
        h = C.CLASSES[o["cls"]]["color"].lstrip("#")
        color = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
        x1, y1, x2, y2 = [int(v) for v in o["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 6)
        pts = [(int(x), int(y)) for x, y, v in o["kpts"] if v > 0]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 235, 255), 5)
        for q in pts:
            cv2.circle(out, q, 12, (40, 40, 220), -1)
    if objects:                                   # 시설물 주변만 크게 보여 줌 (키포인트 확인용)
        xs = [v for o in objects for v in (o["box"][0], o["box"][2])]
        ys = [v for o in objects for v in (o["box"][1], o["box"][3])]
        m = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.8 + 80
        x1, y1 = int(max(min(xs) - m, 0)), int(max(min(ys) - m, 0))
        x2, y2 = int(min(max(xs) + m, out.shape[1])), int(min(max(ys) + m, out.shape[0]))
        out = out[y1:y2, x1:x2]
    return out


class PreviewDialog(QDialog):
    """원본 1장 + 증강 예시 5장을 박스, 키포인트와 함께 보여 줌. '다른 이미지'로 바꿔 보기.

    예시 장수(5장)는 설정한 증강 장수와 상관없이 고정 (무작위라 여러 장을 봐야 바뀌는 범위를 알 수 있음).
    """

    def __init__(self, items, settings, copies, parent=None):
        super().__init__(parent)
        self.setWindowTitle("증강 미리보기 (박스, 키포인트도 같이 바뀌는지 확인)")
        self.resize(1300, 760)
        self.items, self.settings = items, settings
        lay = QVBoxLayout(self)
        real = (f"실제 학습, 내보내기: 원본 1장당 증강 <b>{copies}장</b> (아래 예시처럼 매번 무작위)" if copies else
                "실제 학습, 내보내기: 증강 <b>0장</b> (증강하지 않음, 아래는 설정 확인용 예시)")
        lay.addWidget(text_label(f"<b>증강 예시 5장</b> (설정한 장수와 상관없이 항상 5장, 바뀌는 범위 확인용) / {real}"))
        self.grid = QGridLayout()
        lay.addLayout(self.grid)
        self.labels = []
        for i in range(6):
            lb = QLabel()
            lb.setAlignment(Qt.AlignCenter)
            cap = QLabel("원본" if i == 0 else f"증강 예시 {i}")
            cap.setAlignment(Qt.AlignCenter)
            box = QVBoxLayout()
            box.addWidget(lb, 1)
            box.addWidget(cap)
            self.grid.addLayout(box, i // 3, i % 3)
            self.labels.append(lb)
        lay.addWidget(note(A.describe(settings)))
        row = QHBoxLayout()
        b_other = QPushButton("다른 이미지")
        b_other.clicked.connect(self.show_random)
        b_again = QPushButton("다시 증강")
        b_again.clicked.connect(lambda: self.show_image(self.current))
        b_close = QPushButton("닫기")
        b_close.clicked.connect(self.accept)
        row.addWidget(b_other)
        row.addWidget(b_again)
        row.addStretch(1)
        row.addWidget(b_close)
        lay.addLayout(row)
        self.show_random()

    def show_random(self):
        self.current = random.choice(self.items)
        self.show_image(self.current)

    def show_image(self, item):
        img_path, lab = item
        img = read_image(img_path)
        rng = random.Random()
        shots = [draw(img, lab["objects"])]
        for _ in range(5):
            a_img, a_objs = A.augment(img, lab["objects"], self.settings, rng)
            shots.append(draw(a_img, a_objs))
        for lb, im in zip(self.labels, shots):
            scale = min(400 / im.shape[1], 300 / im.shape[0])
            small = cv2.resize(im, (max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))))
            small = __import__("numpy").ascontiguousarray(small)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            lb.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                                                  QImage.Format_RGB888).copy()))
