"""데이터셋 생성 페이지: 프로젝트(태스크·키포인트·분류 스키마) · 텍스트→이미지 · 라벨링(박스/폴리곤/키포인트/분류) · 내보내기."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsLineItem,
                               QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..genai.project import FORMATS, KPT_PRESETS, ClassDef, Project, flip_idx_from_names
from ..io_utils import imread
from ..labels import Annotation, box_corners
from ..visualize import class_color
from .genai_tab import JobThread, RunBar, _dir_row, _dspin, _ispin, _scroll, open_folder
from .widgets import CaptionedImage, FuncThread, np_to_pixmap

DEFAULT_CLASSES = [("traffic_light", "traffic light"), ("median_barrier", "road median barrier"),
                   ("street_light", "street lamp pole"), ("traffic_sign", "traffic sign"), ("person", "person")]


# ============================================================================ 1. 프로젝트
FMT_NOTES = {
    "detect": "박스만 필요합니다. 폴리곤으로 그려도 내보낼 때 외접 박스로 변환됩니다.",
    "segment": "인스턴스 세그멘테이션: 폴리곤 라벨(수동 폴리곤 또는 자동 라벨의 SAM)이 필요합니다.",
    "obb": "폴리곤/박스를 최소 외접 회전 사각형으로 변환합니다. 폴리곤 라벨일 때 정확합니다.",
    "pose": "자세 추정: 프로젝트의 '키포인트 스키마'가 필요합니다. 대상 클래스 객체마다 박스 + 키포인트를 찍습니다 "
            "(사람 17점은 ViTPose 자동 라벨 지원).",
    "classify": "이미지 분류: '이미지 분류 클래스'를 정하고 이미지마다 클래스 하나를 지정합니다 "
                "(CLIP 자동 분류 지원). 객체 라벨은 필요 없습니다.",
}
IMG_CLASS_EXAMPLE = [("normal", "a drone photo of intact buildings and roads"),
                     ("damaged", "a drone photo of damaged collapsed buildings and debris"),
                     ("flooded", "a drone photo of a flooded area"), ("fire", "a drone photo of a fire with smoke")]


def _class_table(headers):
    t = QTableWidget(0, 2)
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.setMinimumHeight(140)
    return t


def _table_rows(tbl) -> list[ClassDef]:
    out = []
    for r in range(tbl.rowCount()):
        n = (tbl.item(r, 0).text() if tbl.item(r, 0) else "").strip()
        p = (tbl.item(r, 1).text() if tbl.item(r, 1) else "").strip()
        if n:
            out.append(ClassDef(n.replace(" ", "_"), p or n.replace("_", " ")))
    return out


def _fill(tbl, classes):
    tbl.setRowCount(len(classes))
    for r, c in enumerate(classes):
        tbl.setItem(r, 0, QTableWidgetItem(c.name))
        tbl.setItem(r, 1, QTableWidgetItem(c.prompt))


def _parse_skeleton(text: str) -> list[list[int]]:
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.replace(" ", "").split("-")
        out.append([int(a), int(b)])
    return out


class ProjectPage(QWidget):
    def __init__(self, win, tab):
        super().__init__()
        self.win, self.tab = win, tab
        outer = QVBoxLayout(self)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 6, 0)

        g = QGroupBox("프로젝트")
        f = QFormLayout(g)
        w, self.ed_root = _dir_row("프로젝트 폴더 (새 폴더 또는 기존 프로젝트)", "프로젝트 폴더", self)
        row = QHBoxLayout()
        row.addWidget(w, 1)
        b_new = QPushButton("새로 만들기")
        b_new.setObjectName("primary")
        b_new.clicked.connect(self.create)
        b_open = QPushButton("열기")
        b_open.clicked.connect(self.open)
        row.addWidget(b_new)
        row.addWidget(b_open)
        f.addRow("폴더", row)
        self.cb_fmt = QComboBox()
        for k, t in FORMATS.items():
            self.cb_fmt.addItem(t, k)
        self.cb_fmt.currentIndexChanged.connect(self._fmt_changed)
        f.addRow("YOLO 태스크 (내보낼 포맷)", self.cb_fmt)
        self.lb_fmt = QLabel("")
        self.lb_fmt.setWordWrap(True)
        self.lb_fmt.setObjectName("note")
        f.addRow(self.lb_fmt)
        self.lb_stats = QLabel("-")
        self.lb_stats.setWordWrap(True)
        f.addRow("현황", self.lb_stats)
        root.addWidget(g)

        g2 = QGroupBox("원본 이미지 가져오기")
        g2l = QHBoxLayout(g2)
        w2, self.ed_src = _dir_row("원본 이미지 폴더 (하위 폴더 포함, 옆에 YOLO .txt가 있으면 함께 가져옴)", "원본 폴더", self)
        b_imp = QPushButton("가져오기")
        b_imp.clicked.connect(self.import_images)
        g2l.addWidget(w2, 1)
        g2l.addWidget(b_imp)
        root.addWidget(g2)
        from .video_box import VideoBox
        self.video = VideoBox(self)
        root.addWidget(self.video)

        # 객체 클래스
        self.g_obj = QGroupBox("객체 클래스 — Detect / Segment / OBB / Pose  (자동 라벨 문구는 영어 권장)")
        g3l = QVBoxLayout(self.g_obj)
        self.tbl = _class_table(["클래스 이름 (라벨에 저장)", "자동 라벨 문구 (검출기 입력)"])
        g3l.addWidget(self.tbl)
        r = QHBoxLayout()
        for t, fn in (("추가", lambda: self.tbl.insertRow(self.tbl.rowCount())), ("삭제", self._del_row),
                      ("예시 채우기 (도로 시설물)", self._example), ("예시: 사람", self._example_person)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            r.addWidget(b)
        r.addStretch(1)
        g3l.addLayout(r)
        root.addWidget(self.g_obj)

        # 키포인트 스키마
        self.g_kpt = QGroupBox("키포인트 스키마 — Pose (자세 추정)")
        gk = QFormLayout(self.g_kpt)
        self.cb_preset = QComboBox()
        for k, d in KPT_PRESETS.items():
            self.cb_preset.addItem(d["title"], k)
        self.cb_preset.currentIndexChanged.connect(self._preset_changed)
        gk.addRow("프리셋", self.cb_preset)
        self.ed_kn = QPlainTextEdit()
        self.ed_kn.setPlaceholderText("키포인트 이름을 쉼표로 구분 (순서 = 라벨 순서). 예) base, middle, top")
        self.ed_kn.setMaximumHeight(62)
        self.ed_kn.textChanged.connect(self._kpt_text_changed)
        gk.addRow("키포인트 이름", self.ed_kn)
        self.ed_sk = QLineEdit()
        self.ed_sk.setPlaceholderText("연결선 (시각화용). 예) 0-1, 1-2")
        gk.addRow("연결 (skeleton)", self.ed_sk)
        self.kpt_cls = QListWidget()
        self.kpt_cls.setMaximumHeight(110)
        gk.addRow("키포인트 대상 클래스", self.kpt_cls)
        self.lb_flip = QLabel("")
        self.lb_flip.setWordWrap(True)
        self.lb_flip.setObjectName("note")
        gk.addRow("flip_idx (자동)", self.lb_flip)
        note = QLabel("YOLO pose는 데이터셋 전체가 하나의 kpt_shape를 공유합니다. 대상 클래스 객체에만 키포인트를 찍으며, "
                      "Pose로 내보낼 때는 대상 클래스(와 키포인트가 달린 클래스)만 포함됩니다. "
                      "left_/right_ 이름 짝은 좌우 반전 시 자동으로 교환됩니다.")
        note.setWordWrap(True)
        note.setObjectName("note")
        gk.addRow(note)
        root.addWidget(self.g_kpt)

        # 이미지 분류 클래스
        self.g_img = QGroupBox("이미지 분류 클래스 — Classification  (CLIP 자동 분류 문구는 영어 문장 권장)")
        gil = QVBoxLayout(self.g_img)
        self.tbl_img = _class_table(["분류 클래스 이름 (폴더명)", "자동 분류 문구 (CLIP)"])
        gil.addWidget(self.tbl_img)
        r2 = QHBoxLayout()
        for t, fn in (("추가", lambda: self.tbl_img.insertRow(self.tbl_img.rowCount())),
                      ("삭제", lambda: self.tbl_img.removeRow(self.tbl_img.currentRow())),
                      ("예시 채우기 (재난)", self._example_img)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            r2.addWidget(b)
        r2.addStretch(1)
        gil.addLayout(r2)
        root.addWidget(self.g_img)

        sv = QHBoxLayout()
        b_save = QPushButton("설정 저장")
        b_save.setObjectName("primary")
        b_save.clicked.connect(self.save)
        sv.addStretch(1)
        sv.addWidget(b_save)
        root.addLayout(sv)
        self.run = RunBar("")
        self.run.btn_run.hide()
        self.run.btn_open.clicked.connect(lambda: self.tab.project and open_folder(self.tab.project.root))
        self.run.btn_cancel.clicked.connect(lambda: self._th and self._th.cancel.set())
        self.run.setMinimumHeight(150)
        root.addWidget(self.run)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.NoFrame)
        sc.setWidget(inner)
        outer.addWidget(sc)
        self._th = None
        self._fill_kpt_classes([], [])
        self._fmt_changed()

    # ---------------------------------------------------------------- tables
    def _del_row(self):
        r = self.tbl.currentRow()
        if r >= 0:
            if self.tab.project and r < len(self.tab.project.class_names):
                if QMessageBox.question(self, "클래스 삭제", "이미 라벨에 쓰인 클래스를 지우면 id가 밀립니다.\n"
                                        "해당 클래스를 쓰지 않을 때만 삭제하세요. 계속할까요?") != QMessageBox.Yes:
                    return
            self.tbl.removeRow(r)
            self._fill_kpt_classes(None, None)

    def _append(self, tbl, rows):
        cur = {c.name for c in _table_rows(tbl)}
        for n, pr in rows:
            if n not in cur:
                r = tbl.rowCount()
                tbl.insertRow(r)
                tbl.setItem(r, 0, QTableWidgetItem(n))
                tbl.setItem(r, 1, QTableWidgetItem(pr))

    def _example(self):
        self._append(self.tbl, DEFAULT_CLASSES)
        self._fill_kpt_classes(None, None)

    def _example_person(self):
        self._append(self.tbl, [("person", "person")])
        self._fill_kpt_classes(None, self._kpt_checked() + ["person"])
        if self.cb_preset.currentData() == "none":
            self.cb_preset.setCurrentIndex(self.cb_preset.findData("coco17"))

    def _example_img(self):
        self._append(self.tbl_img, IMG_CLASS_EXAMPLE)

    # ---------------------------------------------------------------- keypoints
    def _kpt_checked(self) -> list[str]:
        return [self.kpt_cls.item(i).text() for i in range(self.kpt_cls.count())
                if self.kpt_cls.item(i).checkState() == Qt.Checked]

    def _fill_kpt_classes(self, names=None, checked=None):
        names = names if names is not None else [c.name for c in _table_rows(self.tbl)]
        checked = set(checked if checked is not None else self._kpt_checked())
        self.kpt_cls.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if n in checked else Qt.Unchecked)
            self.kpt_cls.addItem(it)

    def _kpt_names(self) -> list[str]:
        t = self.ed_kn.toPlainText().replace("\n", ",")
        return [x.strip().replace(" ", "_") for x in t.split(",") if x.strip()]

    def _preset_changed(self):
        k = self.cb_preset.currentData()
        d = KPT_PRESETS[k]
        if k != "custom":
            self.ed_kn.blockSignals(True)
            self.ed_kn.setPlainText(", ".join(d["names"]))
            self.ed_kn.blockSignals(False)
            self.ed_sk.setText(", ".join(f"{a}-{b}" for a, b in d["skeleton"]))
        on = k != "none"
        for w_ in (self.ed_kn, self.ed_sk, self.kpt_cls):
            w_.setEnabled(on)
        if on and not self._kpt_checked():  # 추천 대상 클래스
            want = {"coco17": ("person", "worker", "pedestrian", "firefighter"), "person_simple": ("person",),
                    "pole4": ("traffic_light", "street_light", "pole", "traffic_sign"),
                    "vehicle4": ("car", "vehicle", "truck")}.get(k, ())
            for i in range(self.kpt_cls.count()):
                it = self.kpt_cls.item(i)
                if any(x in it.text().lower() for x in want):
                    it.setCheckState(Qt.Checked)
        self._kpt_text_changed()

    def _kpt_text_changed(self):
        names = self._kpt_names()
        if not names:
            self.lb_flip.setText("-")
            return
        fi = flip_idx_from_names(names)
        swaps = [f"{names[i]}↔{names[j]}" for i, j in enumerate(fi) if j > i]
        self.lb_flip.setText(f"K={len(names)} · kpt_shape=[{len(names)}, 3] · 교환: " + (", ".join(swaps) or "없음 (좌우 대칭 없음)"))

    # ---------------------------------------------------------------- project
    def _root(self):
        t = self.ed_root.text().strip()
        if not t:
            QMessageBox.warning(self, "프로젝트", "프로젝트 폴더를 지정하세요.")
            return None
        return Path(t)

    def create(self):
        root = self._root()
        if root is None:
            return
        if Project.exists(root):
            QMessageBox.information(self, "프로젝트", "이미 프로젝트가 있는 폴더입니다 — 열기로 불러옵니다.")
            return self.open()
        p = Project.create(root, self.cb_fmt.currentData(), _table_rows(self.tbl))
        if not self._apply(p):
            return
        p.save()
        self.tab.set_project(p)
        self._show(p)

    def open(self):
        root = self._root()
        if root is None:
            return
        if not Project.exists(root):
            QMessageBox.warning(self, "프로젝트", "project.json이 없습니다. '새로 만들기'를 누르세요.")
            return
        p = Project.load(root)
        self.tab.set_project(p)
        self._show(p)

    def _show(self, p: Project):
        self.ed_root.setText(str(p.root))
        self.cb_fmt.blockSignals(True)
        self.cb_fmt.setCurrentIndex(max(0, self.cb_fmt.findData(p.fmt)))
        self.cb_fmt.blockSignals(False)
        _fill(self.tbl, p.classes)
        _fill(self.tbl_img, p.image_classes)
        k = p.kpt
        self.cb_preset.blockSignals(True)
        self.cb_preset.setCurrentIndex(max(0, self.cb_preset.findData(k.get("preset", "none"))))
        self.cb_preset.blockSignals(False)
        self.ed_kn.blockSignals(True)
        self.ed_kn.setPlainText(", ".join(k.get("names", [])))
        self.ed_kn.blockSignals(False)
        self.ed_sk.setText(", ".join(f"{a}-{b}" for a, b in k.get("skeleton", [])))
        self._fill_kpt_classes(p.class_names, k.get("classes", []))
        on = k.get("preset", "none") != "none"
        for w_ in (self.ed_kn, self.ed_sk, self.kpt_cls):
            w_.setEnabled(on)
        self._kpt_text_changed()
        self._fmt_changed(save=False)
        self._stats()

    def _stats(self):
        p = self.tab.project
        if p is None:
            self.lb_stats.setText("-")
            return
        s = p.stats()
        t = (f"원본 {s['images']}장 · 라벨 완료 {s['labeled']}장 · 객체 {s['objects']}개 · 생성 실행 {s['runs']}회 · "
             f"클래스 {len(p.class_names)}개")
        if p.kpt_names:
            t += f" · 키포인트 객체 {s['kpt_objects']}개 (K={len(p.kpt_names)})"
        if p.image_classes:
            t += f" · 분류 지정 {s['classified']}장"
        if s["auto"]:
            t += f" · <span style='color:#b45309'>미확정 자동 라벨 {s['auto']}개</span>"
        self.lb_stats.setText(t)

    def _fmt_changed(self, *_, save=True):
        fmt = self.cb_fmt.currentData()
        self.lb_fmt.setText(FMT_NOTES.get(fmt, "") + "  (태스크는 나중에 '내보내기'에서 바꿔도 됩니다 — 라벨은 모두 보관됨)")
        # 해당 태스크에 필요한 설정 그룹을 강조
        for gb, need in ((self.g_kpt, fmt == "pose"), (self.g_img, fmt == "classify"),
                         (self.g_obj, fmt != "classify")):
            f = gb.font()
            f.setBold(need)
            gb.setFont(f)
        p = self.tab.project
        if p is not None and save:
            p.fmt = fmt
            p.save()
            self.tab.set_project(p)

    def _apply(self, p: Project) -> bool:
        """표 → 프로젝트. 검증 실패 시 False."""
        p.classes = _table_rows(self.tbl)
        p.image_classes = _table_rows(self.tbl_img)
        p.fmt = self.cb_fmt.currentData()
        preset = self.cb_preset.currentData()
        names = self._kpt_names() if preset != "none" else []
        try:
            sk = _parse_skeleton(self.ed_sk.text()) if names else []
        except ValueError:
            QMessageBox.warning(self, "키포인트", "연결(skeleton)은 '0-1, 1-2' 형식이어야 합니다.")
            return False
        if any(a >= len(names) or b >= len(names) for a, b in sk):
            QMessageBox.warning(self, "키포인트", f"연결 번호는 0~{len(names) - 1} 범위여야 합니다.")
            return False
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "키포인트", "키포인트 이름이 중복되었습니다.")
            return False
        if names and names != p.kpt_names and p.kpt_names and p.stats()["kpt_objects"]:
            if len(names) != len(p.kpt_names) and QMessageBox.question(
                    self, "키포인트", "키포인트 개수가 바뀌었습니다. 이미 찍은 키포인트는 새 개수에 맞게 "
                    "잘리거나 비어 있는 점으로 채워집니다. 계속할까요?") != QMessageBox.Yes:
                return False
        cls_names = [c.name for c in p.classes]
        targets = [n for n in self._kpt_checked() if n in cls_names]
        if names and not targets:
            QMessageBox.information(self, "키포인트", "키포인트를 찍을 대상 클래스를 하나 이상 체크하세요 "
                                    "(예: person). 대상이 없으면 Pose 내보내기에 객체가 포함되지 않습니다.")
        p.set_keypoints(preset if names else "none", names, sk, targets)
        return True

    def save(self):
        p = self.tab.project
        if p is None:
            QMessageBox.information(self, "프로젝트", "먼저 프로젝트를 만드세요.")
            return
        if not self._apply(p):
            return
        p.save()
        self._fill_kpt_classes(p.class_names, p.kpt.get("classes", []))
        self.tab.set_project(p)
        self._stats()
        self.win.statusBar().showMessage("프로젝트 설정을 저장했습니다.", 3000)

    def import_images(self):
        p = self.tab.project
        if p is None:
            QMessageBox.information(self, "프로젝트", "먼저 프로젝트를 만드세요.")
            return
        src = self.ed_src.text().strip()
        if not src or not Path(src).is_dir():
            QMessageBox.warning(self, "원본", "원본 이미지 폴더를 선택하세요.")
            return
        self.save()
        self.run.running(True)
        self._th = JobThread(lambda progress=None, log=print, cancel=None: p.import_images(src, True, progress, log,
                                                                                           cancel))
        self._th.progress.connect(lambda d, t, m: (self.run.bar.setMaximum(t), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(lambda n: (self.run.running(False), self._stats(), self.tab.set_project(p)))
        self._th.failed.connect(lambda m: (self.run.running(False), self.run.log.appendPlainText(m)))
        self._th.start()

    def showEvent(self, e):
        super().showEvent(e)
        self._stats()


# ============================================================================ 3. 라벨링 — 주석 편집기
def _qcol(i):
    b, g, r = class_color(i)
    return QColor(r, g, b)


class KptItem(QGraphicsEllipseItem):
    """키포인트 점 (AnnItem의 자식 → 객체를 옮기면 함께 이동). v: 1=가려짐, 2=보임."""
    R = 5

    def __init__(self, parent, idx: int, v: int, name: str):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R, parent)
        self.idx, self.v = idx, v
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIgnoresTransformations |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(5)
        self.setToolTip(f"{idx}: {name}")
        self.set_v(v)

    def set_v(self, v):
        self.v = v
        col = _qcol(self.idx + 3)
        pen = QPen(QColor("#111827"), 1.5)
        self.setPen(pen)
        self.setBrush(QBrush(col if v >= 2 else QColor(160, 160, 160)))
        if v < 2:
            self.setPen(QPen(col, 2.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and isinstance(self.parentItem(), AnnItem):
            self.parentItem().update_skeleton()
            sc = self.scene()
            if sc is not None and sc.views() and not getattr(sc.views()[0], "loading", False):
                sc.views()[0].changed.emit()
        return super().itemChange(change, value)


class AnnItem(QGraphicsPolygonItem):
    def __init__(self, poly: QPolygonF, cls: int, name: str, auto: bool = False, score: float | None = None):
        super().__init__(poly)
        self.cls = cls
        self.auto = auto
        self.score = score
        self.kitems: dict[int, KptItem] = {}
        self.lines: list = []
        self.skeleton: list = []
        self.kpt_names: list[str] = []
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.text = QGraphicsSimpleTextItem(self)
        self.set_class(cls, name)

    def set_class(self, cls, name):
        self.cls = cls
        col = _qcol(cls)
        pen = QPen(col, 2)
        pen.setCosmetic(True)
        if self.auto:
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        fill = QColor(col)
        fill.setAlpha(45)
        self.setBrush(QBrush(fill))
        self.text.setText(name + (f" {self.score:.2f}" if self.score is not None else "") + (" ·자동" if self.auto else ""))
        self.text.setBrush(QBrush(col))
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        self.text.setFont(f)
        self.text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        br = self.polygon().boundingRect()
        self.text.setPos(br.left(), br.top() - 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            pen = self.pen()
            pen.setWidth(4 if value else 2)
            self.setPen(pen)
        return super().itemChange(change, value)

    def scene_points(self) -> np.ndarray:
        return np.array([[p.x(), p.y()] for p in (self.mapToScene(self.polygon()))], np.float64)

    # ---------------------------------------------------------------- keypoints
    def setup_kpts(self, names, skeleton, kpts: np.ndarray | None):
        self.kpt_names, self.skeleton = list(names), list(skeleton)
        for k in list(self.kitems.values()):
            k.setParentItem(None)
            if k.scene():
                k.scene().removeItem(k)
        self.kitems = {}
        if kpts is not None:
            for i, (x, y, v) in enumerate(kpts[: len(names)]):
                if v > 0:
                    self.set_kpt(i, QPointF(float(x), float(y)), int(v), scene_pos=True)
        self.update_skeleton()

    def set_kpt(self, i, pos: QPointF, v: int = 2, scene_pos=True):
        it = self.kitems.get(i)
        if it is None:
            it = KptItem(self, i, v, self.kpt_names[i] if i < len(self.kpt_names) else str(i))
            self.kitems[i] = it
        it.set_v(v)
        it.setPos(self.mapFromScene(pos) if scene_pos else pos)
        self.update_skeleton()

    def remove_kpt(self, i):
        it = self.kitems.pop(i, None)
        if it is not None:
            it.setParentItem(None)
            if it.scene():
                it.scene().removeItem(it)
        self.update_skeleton()

    def update_skeleton(self):
        for ln in self.lines:
            ln.setParentItem(None)
            if ln.scene():
                ln.scene().removeItem(ln)
        self.lines = []
        pen = QPen(QColor(255, 255, 255, 220), 1.5)
        pen.setCosmetic(True)
        for a, b in self.skeleton:
            if a in self.kitems and b in self.kitems:
                ln = QGraphicsLineItem(QLineF(self.kitems[a].pos(), self.kitems[b].pos()), self)
                ln.setPen(pen)
                ln.setZValue(4)
                self.lines.append(ln)

    def kpts_array(self, k: int) -> np.ndarray | None:
        if k == 0 or not self.kitems:
            return None
        out = np.zeros((k, 3), np.float64)
        for i, it in self.kitems.items():
            if i < k:
                sp = it.scenePos()
                out[i] = (sp.x(), sp.y(), it.v)
        return out


class Canvas(QGraphicsView):
    changed = Signal()
    selection = Signal()
    kpt_state = Signal()     # 키포인트 패널 갱신
    message = Signal(str)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setBackgroundBrush(QColor("#1f2937"))
        self.mode = "select"
        self.cur_class = 0
        self.names: list[str] = []
        self.kpt_names: list[str] = []
        self.skeleton: list = []
        self.pose_ids: set[int] = set()
        self.kpt_next = 0
        self.loading = False
        self.pix = None
        self.size_wh = (0, 0)
        self._start = None
        self._temp = None
        self._poly: list[QPointF] = []
        self.scene().selectionChanged.connect(self._sel)

    def _sel(self):
        self.selection.emit()
        self.kpt_state.emit()

    # ---------------------------------------------------------------- data
    def set_image(self, bgr, anns: list[Annotation], names, kpt_names=(), skeleton=(), pose_ids=()):
        sc = self.scene()
        sc.clear()
        self._temp = None
        self._poly = []
        self.names = names
        self.kpt_names, self.skeleton, self.pose_ids = list(kpt_names), list(skeleton), set(pose_ids)
        self.kpt_next = 0
        self.pix = QGraphicsPixmapItem(np_to_pixmap(bgr))
        self.pix.setZValue(-1)
        sc.addItem(self.pix)
        h, w = bgr.shape[:2]
        self.size_wh = (w, h)
        sc.setSceneRect(QRectF(0, 0, w, h))
        self.loading = True
        try:
            for a in anns:
                self.add_ann(a, a.auto, a.score)
        finally:
            self.loading = False
        self.fit()
        self.kpt_state.emit()

    def add_ann(self, a: Annotation, auto=False, score=None):
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in a.pts])
        name = self.names[a.cls] if 0 <= a.cls < len(self.names) else str(a.cls)
        it = AnnItem(poly, a.cls, name, auto, score)
        self.scene().addItem(it)
        it.setup_kpts(self.kpt_names, self.skeleton, a.kpts)
        return it

    def items_ann(self) -> list[AnnItem]:
        return [i for i in self.scene().items() if isinstance(i, AnnItem)]

    def anns(self) -> list[Annotation]:
        w, h = self.size_wh
        k = len(self.kpt_names)
        out = []
        for it in reversed(self.items_ann()):
            pts = it.scene_points()
            pts[:, 0] = np.clip(pts[:, 0], 0, w)
            pts[:, 1] = np.clip(pts[:, 1], 0, h)
            if len(pts) >= 3 and np.ptp(pts[:, 0]) >= 1 and np.ptp(pts[:, 1]) >= 1:
                kp = it.kpts_array(k)
                if kp is not None:
                    inside = (kp[:, 0] >= 0) & (kp[:, 0] <= w) & (kp[:, 1] >= 0) & (kp[:, 1] <= h)
                    kp[~inside] = 0
                out.append(Annotation(it.cls, pts, kp, it.auto, it.score))
        return out

    def fit(self):
        if self.pix is not None:
            self.fitInView(self.pix, Qt.KeepAspectRatio)

    def delete_selected(self):
        for it in self.scene().selectedItems():
            if isinstance(it, AnnItem):
                self.scene().removeItem(it)
        self.changed.emit()
        self.kpt_state.emit()

    def set_class_selected(self, cls):
        n = 0
        for it in self.scene().selectedItems():
            if isinstance(it, AnnItem):
                it.auto = False
                it.score = None
                it.set_class(cls, self.names[cls] if cls < len(self.names) else str(cls))
                n += 1
        if n:
            self.changed.emit()
        return n

    def target(self) -> AnnItem | None:
        sel = [i for i in self.scene().selectedItems() if isinstance(i, AnnItem)]
        return sel[0] if len(sel) == 1 else None

    def set_mode(self, m):
        self.mode = m
        self._cancel_draw()
        self.setDragMode(QGraphicsView.RubberBandDrag if m == "select" else QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.ArrowCursor if m == "select" else Qt.CrossCursor)
        self.kpt_state.emit()

    def _cancel_draw(self):
        if self._temp is not None:
            self.scene().removeItem(self._temp)
        self._temp = None
        self._poly = []
        self._start = None

    def advance_kpt(self, it: AnnItem | None):
        k = len(self.kpt_names)
        if not k:
            return
        if it is None:
            self.kpt_next = 0
        else:
            for d in range(1, k + 1):
                j = (self.kpt_next + d) % k
                if j not in it.kitems:
                    self.kpt_next = j
                    break
        self.kpt_state.emit()

    # ---------------------------------------------------------------- events
    def wheelEvent(self, e):
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.scale(f, f)

    def _clamp(self, p: QPointF):
        w, h = self.size_wh
        return QPointF(min(max(p.x(), 0), w), min(max(p.y(), 0), h))

    def _temp_pen(self):
        pen = QPen(_qcol(self.cur_class), 2, Qt.DashLine)
        pen.setCosmetic(True)
        return pen

    def _ann_at(self, p: QPointF) -> AnnItem | None:
        for it in self.scene().items(p):
            if isinstance(it, KptItem):
                return it.parentItem()
            if isinstance(it, AnnItem):
                return it
        return None

    def _kpt_press(self, e, p):
        if not self.kpt_names:
            self.message.emit("키포인트 스키마가 없습니다 — '1. 프로젝트 · 원본'에서 정의하세요.")
            return
        hit = [i for i in self.scene().items(p) if isinstance(i, KptItem)]
        if e.button() == Qt.RightButton:  # 점 삭제
            if hit:
                par = hit[0].parentItem()
                self.kpt_next = hit[0].idx
                par.remove_kpt(hit[0].idx)
                self.changed.emit()
                self.kpt_state.emit()
            return
        if e.button() != Qt.LeftButton:
            return
        tgt = self.target()
        if e.modifiers() & Qt.ControlModifier or tgt is None:  # 대상 객체 선택
            a = self._ann_at(p)
            if a is None:
                self.message.emit("키포인트를 찍을 객체(박스)를 먼저 그리거나 클릭해 선택하세요.")
                return
            if a is not tgt:
                self.scene().clearSelection()
                a.setSelected(True)
                self.kpt_next = next((i for i in range(len(self.kpt_names)) if i not in a.kitems), 0)
                self.kpt_state.emit()
                if e.modifiers() & Qt.ControlModifier or tgt is not None:
                    return
            tgt = a
        if tgt.cls not in self.pose_ids and self.pose_ids:
            self.message.emit(f"'{self.names[tgt.cls] if tgt.cls < len(self.names) else tgt.cls}'는 키포인트 대상 "
                              "클래스가 아닙니다 (프로젝트 설정에서 대상 클래스를 체크하세요). 그래도 저장은 됩니다.")
        v = 1 if e.modifiers() & Qt.ShiftModifier else 2
        tgt.set_kpt(self.kpt_next, p, v)
        tgt.auto = False
        tgt.set_class(tgt.cls, self.names[tgt.cls] if tgt.cls < len(self.names) else str(tgt.cls))
        self.advance_kpt(tgt)
        self.changed.emit()

    def mousePressEvent(self, e):
        if self.pix is None:
            return
        if e.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            return super().mousePressEvent(e)
        p = self._clamp(self.mapToScene(e.position().toPoint()))
        if self.mode == "kpt":
            self._kpt_press(e, p)
            return
        if self.mode == "box" and e.button() == Qt.LeftButton:
            self._start = p
            self._temp = self.scene().addRect(QRectF(p, p), self._temp_pen())
            return
        if self.mode == "poly":
            if e.button() == Qt.RightButton and self._poly:
                self._poly.pop()
            elif e.button() == Qt.LeftButton:
                self._poly.append(p)
            self._draw_poly_temp()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.mode == "box" and self._start is not None and self._temp is not None:
            p = self._clamp(self.mapToScene(e.position().toPoint()))
            self._temp.setRect(QRectF(self._start, p).normalized())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            super().mouseReleaseEvent(e)
            self.setDragMode(QGraphicsView.RubberBandDrag if self.mode == "select" else QGraphicsView.NoDrag)
            return
        if self.mode == "box" and self._start is not None:
            r = self._temp.rect() if self._temp else QRectF()
            self._cancel_draw()
            if r.width() >= 3 and r.height() >= 3:
                it = self.add_ann(Annotation(self.cur_class, box_corners(r.left(), r.top(), r.right(), r.bottom())))
                self.scene().clearSelection()
                it.setSelected(True)  # 바로 K 모드로 키포인트를 찍을 수 있게
                self.changed.emit()
            return
        if self.mode == "kpt":
            return
        was_select = self.mode == "select"
        super().mouseReleaseEvent(e)
        if was_select:
            self.changed.emit()  # 이동 반영

    def mouseDoubleClickEvent(self, e):
        if self.mode == "poly":
            self.finish_poly()
            return
        if self.mode == "kpt":
            return
        super().mouseDoubleClickEvent(e)

    def finish_poly(self):
        pts = self._poly[:]
        while len(pts) >= 2 and (pts[-1] - pts[-2]).manhattanLength() < 2:
            pts.pop()
        self._cancel_draw()
        if len(pts) >= 3:
            it = self.add_ann(Annotation(self.cur_class, np.array([[p.x(), p.y()] for p in pts], np.float64)))
            self.scene().clearSelection()
            it.setSelected(True)
            self.changed.emit()

    def _draw_poly_temp(self):
        if self._temp is not None:
            self.scene().removeItem(self._temp)
            self._temp = None
        if self._poly:
            self._temp = self.scene().addPolygon(QPolygonF(self._poly), self._temp_pen())

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and self.mode == "poly":
            self.finish_poly()
            return
        if e.key() == Qt.Key_Escape:
            self._cancel_draw()
            self.scene().clearSelection()
            return
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            return
        super().keyPressEvent(e)


class LabelPage(QWidget):
    def __init__(self, win, tab):
        super().__init__()
        self.win, self.tab = win, tab
        self.cur: Path | None = None
        self.cur_wh = (0, 0)
        self.dirty = False
        self._labelers: dict = {}
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)

        # 이미지 목록
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.cb_filter = QComboBox()
        self.cb_filter.addItems(["전체", "라벨 없음", "라벨 있음", "미확정 자동 라벨", "AI 생성 이미지"])
        self.cb_filter.addItem("AI assist: least confident first")
        self.cb_filter.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_filter.setMinimumContentsLength(8)
        self.cb_filter.currentIndexChanged.connect(self.refresh)
        ll.addWidget(self.cb_filter)
        self.lst = QListWidget()
        self.lst.setMinimumWidth(150)
        self.lst.currentItemChanged.connect(self._pick)
        ll.addWidget(self.lst, 1)
        self.lb_count = QLabel("")
        self.lb_count.setObjectName("note")
        self.lb_count.setWordWrap(True)
        ll.addWidget(self.lb_count)
        split.addWidget(left)

        # 캔버스 + 도구
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.btn_modes = {}
        for key, text, sc in (("select", "선택/이동 (V)", "V"), ("box", "박스 (B)", "B"), ("poly", "폴리곤 (P)", "P"),
                              ("kpt", "키포인트 (K)", "K")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            QShortcut(QKeySequence(sc), self, activated=lambda k=key: self.set_mode(k))
            self.btn_modes[key] = b
            tools.addWidget(b)
        acts = []
        for text, fn, sc in (("◀ 이전 (A)", lambda: self.step(-1), "A"), ("다음 ▶ (D)", lambda: self.step(1), "D"),
                             ("삭제 (Del)", lambda: self.canvas.delete_selected(), None),
                             ("맞춤 (F)", lambda: self.canvas.fit(), "F")):
            b = QPushButton(text)
            b.clicked.connect(fn)
            if sc:
                QShortcut(QKeySequence(sc), self, activated=fn)
            acts.append(b)
        tools.addStretch(1)
        ml.addLayout(tools)
        crow = QHBoxLayout()
        for b in acts:
            crow.addWidget(b)
        self.lb_imgcls = QLabel("이미지 분류:")
        self.cb_imgcls = QComboBox()
        self.cb_imgcls.setMinimumWidth(130)
        self.cb_imgcls.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_imgcls.setMinimumContentsLength(10)
        self.cb_imgcls.currentIndexChanged.connect(self._imgcls_changed)
        crow.addWidget(self.lb_imgcls)
        crow.addWidget(self.cb_imgcls)
        self.lb_imgcls_note = QLabel("")
        self.lb_imgcls_note.setObjectName("note")
        self.lb_imgcls_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        crow.addWidget(self.lb_imgcls_note, 1)
        ml.addLayout(crow)
        self.canvas = Canvas()
        self.canvas.changed.connect(self._changed)
        self.canvas.selection.connect(self._sel_changed)
        self.canvas.kpt_state.connect(self._kpt_panel)
        self.canvas.message.connect(lambda m: self.win.statusBar().showMessage(m, 5000))
        ml.addWidget(self.canvas, 1)
        help_ = QLabel("박스: 드래그 · 폴리곤: 클릭으로 점 추가, 더블클릭/Enter 완성, 우클릭 되돌리기 · "
                       "키포인트(K): 객체 선택 후 클릭=보임, Shift+클릭=가려짐, 우클릭=삭제, Ctrl+클릭=대상 객체 변경 · "
                       "휠 확대 · 가운데 버튼 이동 · 1~9 클래스 변경 · 점선=미확정 자동 라벨")
        help_.setObjectName("note")
        help_.setWordWrap(True)
        ml.addWidget(help_)
        split.addWidget(mid)

        # 클래스 · 키포인트 · 자동 라벨
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        gc = QGroupBox("객체 클래스 (선택 = 그리기 클래스 / 선택 객체 변경)")
        gcl = QVBoxLayout(gc)
        self.cls = QListWidget()
        self.cls.setMinimumHeight(120)
        self.cls.currentRowChanged.connect(self._cls_changed)
        gcl.addWidget(self.cls)
        rl.addWidget(gc, 1)
        self.g_kpt = QGroupBox("키포인트")
        gkl = QVBoxLayout(self.g_kpt)
        self.kpt_list = QListWidget()
        self.kpt_list.setMinimumHeight(170)
        self.kpt_list.currentRowChanged.connect(self._kpt_pick)
        gkl.addWidget(self.kpt_list)
        self.lb_kpt = QLabel("")
        self.lb_kpt.setWordWrap(True)
        self.lb_kpt.setObjectName("note")
        gkl.addWidget(self.lb_kpt)
        b_clr = QPushButton("선택 객체 키포인트 모두 지우기")
        b_clr.clicked.connect(self._clear_kpts)
        gkl.addWidget(b_clr)
        rl.addWidget(self.g_kpt, 1)

        ga = QGroupBox("자동 라벨")
        gal = QFormLayout(ga)
        self.chk_det = QCheckBox("객체 검출")
        self.chk_det.setChecked(True)
        self.cb_det = QComboBox()
        self.cb_det.addItem("OWLv2", "owlv2")
        self.cb_det.addItem("Grounding DINO", "grounding_dino")
        dr = QHBoxLayout()
        dr.addWidget(self.chk_det)
        dr.addWidget(self.cb_det, 1)
        gal.addRow(dr)
        self.sp_thr = _dspin(0.20, 0.01, 0.95, 0.01)
        gal.addRow("검출 임계값", self.sp_thr)
        self.chk_sam = QCheckBox("SAM으로 폴리곤 생성 (Segment/OBB)")
        self.chk_sam.setChecked(True)
        self.chk_pose = QCheckBox("자세 추정 (ViTPose · 사람 17점)")
        self.chk_pose.setChecked(True)
        self.chk_clf = QCheckBox("이미지 분류 (CLIP)")
        self.chk_clf.setChecked(True)
        self.chk_replace = QCheckBox("기존 자동 라벨(점선) 교체")
        self.chk_replace.setChecked(True)
        for c in (self.chk_sam, self.chk_pose, self.chk_clf, self.chk_replace):
            gal.addRow(c)
        self.b_auto = QPushButton("현재 이미지 자동 라벨")
        self.b_auto.setObjectName("primary")
        self.b_auto.clicked.connect(self.auto_current)
        self.b_auto_all = QPushButton("라벨 없는 이미지 전체 자동 라벨")
        self.b_auto_all.clicked.connect(self.auto_all)
        self.b_accept = QPushButton("자동 라벨 확정 (점선 → 실선)")
        self.b_accept.clicked.connect(self.accept_auto)
        gal.addRow(self.b_auto)
        gal.addRow(self.b_auto_all)
        gal.addRow(self.b_accept)
        self.lb_auto = QLabel("")
        self.lb_auto.setWordWrap(True)
        self.lb_auto.setObjectName("note")
        gal.addRow(self.lb_auto)
        rl.addWidget(ga)
        self.g_assist = QGroupBox("Project model (AI assist)")
        gs = QFormLayout(self.g_assist)
        self.cb_assist = QComboBox()
        self.cb_assist.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_assist.setMinimumContentsLength(12)
        self.sp_assist_thr = _dspin(0.35, 0.01, 0.99, 0.05)
        self.b_assist = QPushButton("Run on this image")
        self.b_assist.clicked.connect(self.assist_current)
        gs.addRow("Model", self.cb_assist)
        gs.addRow("Threshold", self.sp_assist_thr)
        gs.addRow(self.b_assist)
        rl.addWidget(self.g_assist)
        self._assist_pred = None
        self._assist_key = None
        rsc = _scroll(right)
        rsc.setMinimumWidth(300)
        split.addWidget(rsc)
        split.setStretchFactor(1, 1)
        split.setSizes([220, 860, 340])
        root.addWidget(split)
        for i in range(9):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda i=i: self._num_class(i))
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(600)
        self.save_timer.timeout.connect(self.save_current)
        tab.project_changed.connect(self.refresh)
        self._th = None
        self.set_mode("box")

    # ---------------------------------------------------------------- list
    def refresh(self):
        from ..genai import annstore

        p = self.tab.project
        cur = self.cur
        self.lst.blockSignals(True)
        self.lst.clear()
        self.cls.blockSignals(True)
        self.cls.clear()
        if p is None:
            self.lst.blockSignals(False)
            self.cls.blockSignals(False)
            return
        pose_ids = set(p.pose_class_ids)
        for i, n in enumerate(p.class_names):
            it = QListWidgetItem(f"{i + 1}. {n}" + ("  [키포인트]" if i in pose_ids else ""))
            it.setForeground(_qcol(i))
            self.cls.addItem(it)
        self.cls.blockSignals(False)
        if self.cls.count():
            self.cls.setCurrentRow(min(self.canvas.cur_class, self.cls.count() - 1))
        # 이미지 분류 콤보
        self.cb_imgcls.blockSignals(True)
        self.cb_imgcls.clear()
        self.cb_imgcls.addItem("(지정 안 함)", None)
        for n in p.image_class_names:
            self.cb_imgcls.addItem(n, n)
        self.cb_imgcls.blockSignals(False)
        has_ic = bool(p.image_class_names)
        for w_ in (self.lb_imgcls, self.cb_imgcls, self.lb_imgcls_note):
            w_.setVisible(has_ic)
        self.chk_clf.setEnabled(has_ic)
        K = len(p.kpt_names)
        self.g_kpt.setVisible(K > 0)
        self.btn_modes["kpt"].setEnabled(K > 0)
        self.chk_pose.setEnabled(p.kpt.get("preset") == "coco17")
        self.chk_det.setEnabled(bool(p.classes))
        mode = self.cb_filter.currentIndex()
        n_lab = 0
        imgs = p.images()
        if mode == 5:
            from ..assist.core import load_scores
            sc = load_scores(p).get("items", {})
            imgs = sorted(imgs, key=lambda x: -sc.get(x.name, -1.0))
        self._refresh_assist_models()
        for img in imgs:
            ia = annstore.load(p.label_path(img))
            has = bool(ia and (ia.objects or ia.image_class))
            auto = bool(ia and (any(a.auto for a in ia.objects) or ia.image_class_auto))
            n_lab += has
            if ((mode == 1 and has) or (mode == 2 and not has) or (mode in (3, 5) and not auto)
                    or (mode == 4 and not img.name.startswith("gen_"))):
                continue
            mark = "◌ " if auto else ("✓ " if has else "   ")
            tag = f"  [{ia.image_class}]" if ia and ia.image_class else ""
            it = QListWidgetItem(mark + img.name + tag)
            it.setData(Qt.UserRole, str(img))
            self.lst.addItem(it)
        self.lst.blockSignals(False)
        self.lb_count.setText(f"라벨 {n_lab} / {len(imgs)}장 · ✓ 확정 · ◌ 미확정 자동 라벨 포함")
        if cur is not None:
            for r in range(self.lst.count()):
                if self.lst.item(r).data(Qt.UserRole) == str(cur):
                    self.lst.setCurrentRow(r)
                    return
        if self.lst.count() and self.cur is None:
            self.lst.setCurrentRow(0)

    def _pick(self, item, prev=None):
        if item is None:
            return
        self.save_current()
        self.load(Path(item.data(Qt.UserRole)))

    def load(self, img_path: Path):
        p = self.tab.project
        img = imread(img_path)
        if img is None:
            return
        h, w = img.shape[:2]
        self.cur = img_path
        self.cur_wh = (w, h)
        ia = p.load_image_ann(img_path, w, h)
        from ..genai.annstore import fit_kpts
        fit_kpts(ia.objects, len(p.kpt_names))
        self.canvas.set_image(img, ia.objects, p.class_names, p.kpt_names, p.kpt_skeleton, p.pose_class_ids)
        self.cb_imgcls.blockSignals(True)
        i = self.cb_imgcls.findData(ia.image_class) if ia.image_class else 0
        if ia.image_class and i < 0:  # 프로젝트에 없는 분류명 (예: 변형 재매핑)
            self.cb_imgcls.addItem(ia.image_class, ia.image_class)
            i = self.cb_imgcls.count() - 1
        self.cb_imgcls.setCurrentIndex(max(0, i))
        self.cb_imgcls.blockSignals(False)
        self._ic_auto, self._ic_score = ia.image_class_auto, ia.image_class_score
        self.lb_imgcls_note.setText(f"AI suggestion ({self._ic_score or 0:.2f}) - confirm or change"
                                    if self._ic_auto else "")
        self.dirty = False
        self.lb_auto.setText(f"{img_path.name} · {w}×{h}")

    def step(self, d):
        r = self.lst.currentRow() + d
        if 0 <= r < self.lst.count():
            self.lst.setCurrentRow(r)

    # ---------------------------------------------------------------- edit
    def set_mode(self, m):
        if m == "kpt" and not self.btn_modes["kpt"].isEnabled():
            self.win.statusBar().showMessage("키포인트 스키마가 없습니다 — 1. 프로젝트에서 정의하세요.", 4000)
            m = "select"
        for k, b in self.btn_modes.items():
            b.setChecked(k == m)
        self.canvas.set_mode(m)

    def _cls_changed(self, row):
        if row < 0:
            return
        self.canvas.cur_class = row
        self.canvas.set_class_selected(row)

    def _num_class(self, i):
        if i < self.cls.count():
            self.cls.setCurrentRow(i)
            self.canvas.cur_class = i
            self.canvas.set_class_selected(i)

    def _sel_changed(self):
        sel = [i for i in self.canvas.scene().selectedItems() if isinstance(i, AnnItem)]
        if len(sel) == 1:
            self.cls.blockSignals(True)
            self.cls.setCurrentRow(sel[0].cls)
            self.cls.blockSignals(False)

    def _imgcls_changed(self):
        self._ic_auto = False
        self._ic_score = None
        self.lb_imgcls_note.setText("")
        self._changed()

    def _kpt_panel(self):
        names = self.canvas.kpt_names
        if not names:
            return
        tgt = self.canvas.target()
        self.kpt_list.blockSignals(True)
        self.kpt_list.clear()
        for i, n in enumerate(names):
            st = "  "
            if tgt is not None and i in tgt.kitems:
                st = "● " if tgt.kitems[i].v >= 2 else "◐ "
            else:
                st = "○ "
            it = QListWidgetItem(f"{st}{i}. {n}")
            it.setForeground(_qcol(i + 3))
            self.kpt_list.addItem(it)
        self.kpt_list.setCurrentRow(min(self.canvas.kpt_next, len(names) - 1))
        self.kpt_list.blockSignals(False)
        if tgt is None:
            self.lb_kpt.setText("대상 객체 없음 — 박스를 그리거나 객체를 클릭해 선택하세요.")
        else:
            nm = self.canvas.names[tgt.cls] if tgt.cls < len(self.canvas.names) else str(tgt.cls)
            done = len(tgt.kitems)
            self.lb_kpt.setText(f"대상: {nm} · {done}/{len(names)}점 · 다음: <b>{names[self.canvas.kpt_next]}</b>"
                                + ("" if self.canvas.mode == "kpt" else " · K를 눌러 찍기"))

    def _kpt_pick(self, row):
        if row >= 0:
            self.canvas.kpt_next = row
            self._kpt_panel()

    def _clear_kpts(self):
        tgt = self.canvas.target()
        if tgt is not None:
            for i in list(tgt.kitems):
                tgt.remove_kpt(i)
            self.canvas.kpt_next = 0
            self._changed()
            self._kpt_panel()

    def _changed(self):
        self.dirty = True
        self.save_timer.start()

    def current_ann(self):
        from ..genai.annstore import ImageAnn

        w, h = self.cur_wh
        ic = self.cb_imgcls.currentData()
        auto = bool(ic) and getattr(self, "_ic_auto", False)
        return ImageAnn(w, h, self.canvas.anns(), ic, auto, getattr(self, "_ic_score", None) if auto else None)

    def save_current(self):
        p = self.tab.project
        if p is None or self.cur is None or not self.dirty:
            return
        ia = self.current_ann()
        p.save_image_ann(self.cur, ia)
        if ia.image_class:
            p.add_image_class(ia.image_class)
        self.dirty = False
        it = self.lst.currentItem()
        if it is not None and it.data(Qt.UserRole) == str(self.cur):
            has = bool(ia.objects or ia.image_class)
            auto = any(a.auto for a in ia.objects) or ia.image_class_auto
            it.setText(("◌ " if auto else ("✓ " if has else "   ")) + self.cur.name +
                       (f"  [{ia.image_class}]" if ia.image_class else ""))

    def accept_auto(self):
        for it in self.canvas.items_ann():
            if it.auto:
                it.auto = False
                it.score = None
                it.set_class(it.cls, self.canvas.names[it.cls] if it.cls < len(self.canvas.names) else str(it.cls))
        self._ic_auto = False
        self._ic_score = None
        self.lb_imgcls_note.setText("")
        self._changed()

    # ---------------------------------------------------------------- auto label
    def _get_labelers(self):
        """선택한 옵션에 맞는 (det, pose, clf). 모델이 없으면 설명과 함께 RuntimeError."""
        from ..genai import models as M
        from ..genai.autolabel import AutoLabeler, ImageClassifier, PoseLabeler, available

        p = self.tab.project
        root = self.tab.models.models_dir()
        det = pose = clf = None
        miss = []
        if self.chk_det.isChecked() and self.chk_det.isEnabled() and p.classes:
            dk = self.cb_det.currentData()
            ok, why = available(root, dk)
            if ok:
                key = ("det", root, dk, self.chk_sam.isChecked())
                if key not in self._labelers:
                    self._labelers[key] = AutoLabeler(Path(root), dk, self.chk_sam.isChecked(), log=lambda m: None)
                det = self._labelers[key]
            else:
                miss.append(f"객체 검출: {why}")
        if self.chk_pose.isChecked() and self.chk_pose.isEnabled():
            if M.is_installed("vitpose_base", Path(root)):
                key = ("pose", root)
                if key not in self._labelers:
                    self._labelers[key] = PoseLabeler(Path(root))
                pose = self._labelers[key]
            else:
                miss.append("자세 추정: ViTPose 모델 미설치")
        if self.chk_clf.isChecked() and self.chk_clf.isEnabled():
            if M.is_installed("clip_b32", Path(root)):
                key = ("clf", root)
                if key not in self._labelers:
                    self._labelers[key] = ImageClassifier(Path(root))
                clf = self._labelers[key]
            else:
                miss.append("이미지 분류: CLIP 모델 미설치")
        if det is None and pose is None and clf is None:
            raise RuntimeError("실행할 자동 라벨이 없습니다.\n" + "\n".join(miss) +
                               "\n\n'모델 · GPU'에서 필요한 모델을 받으세요. 수동 라벨링은 모델 없이 가능합니다.")
        return det, pose, clf, miss

    def auto_current(self):
        if self.tab.project is None or self.cur is None:
            return
        try:
            det, pose, clf, miss = self._get_labelers()
        except Exception as e:
            QMessageBox.information(self, "자동 라벨", str(e))
            return
        from ..genai.autolabel import label_image_full

        p = self.tab.project
        img = imread(self.cur)
        thr, want_poly, keep = self.sp_thr.value(), self.chk_sam.isChecked(), not self.chk_replace.isChecked()
        existing = self.current_ann()
        self.b_auto.setEnabled(False)
        self.lb_auto.setText("자동 라벨 중… (첫 실행은 모델 로딩)")
        cur = self.cur

        def job():
            return label_image_full(img, p, det, pose, clf, thr, want_poly, existing, keep_auto=keep,
                                    reclassify=clf is not None)

        self._th = FuncThread(job)
        self._th.done.connect(lambda r: self._apply_auto(cur, r, existing, miss))
        self._th.failed.connect(lambda m: (self.b_auto.setEnabled(True), self.lb_auto.setText(m.splitlines()[0])))
        self._th.start()

    def _apply_auto(self, img_path, ia, before, miss):
        self.b_auto.setEnabled(True)
        if img_path != self.cur:
            return
        p = self.tab.project
        img = imread(img_path)
        n_new = sum(1 for a in ia.objects if a.auto)
        self.canvas.set_image(img, ia.objects, p.class_names, p.kpt_names, p.kpt_skeleton, p.pose_class_ids)
        if ia.image_class and ia.image_class != before.image_class:
            i = self.cb_imgcls.findData(ia.image_class)
            self.cb_imgcls.blockSignals(True)
            self.cb_imgcls.setCurrentIndex(max(0, i))
            self.cb_imgcls.blockSignals(False)
            self.lb_imgcls_note.setText(f"← CLIP 자동 분류: {ia.image_class} (확인하세요)")
            self._ic_auto, self._ic_score = True, None
        kp = sum(1 for a in ia.objects if a.auto and a.kpts is not None and (a.kpts[:, 2] > 0).any())
        msg = f"자동 라벨 {n_new}개 (점선)" + (f", 자세 {kp}명" if kp else "") + ". 틀린 것은 삭제·수정 후 [확정]하세요."
        if miss:
            msg += "<br><span style='color:#b45309'>생략: " + " / ".join(miss) + "</span>"
        self.lb_auto.setText(msg)
        self._changed()

    def auto_all(self):
        p = self.tab.project
        if p is None:
            return
        try:
            det, pose, clf, miss = self._get_labelers()
        except Exception as e:
            QMessageBox.information(self, "자동 라벨", str(e))
            return
        from ..genai.autolabel import label_image_full

        self.save_current()
        thr, want_poly = self.sp_thr.value(), self.chk_sam.isChecked()
        todo = [i for i in p.images() if not p.is_labeled(i)]
        if not todo:
            QMessageBox.information(self, "자동 라벨", "라벨 없는 이미지가 없습니다.")
            return

        def job(progress=None, log=print, cancel=None):
            n = 0
            for k, ip in enumerate(todo):
                if cancel.is_set():
                    break
                img = imread(ip)
                if img is None:
                    continue
                ia = label_image_full(img, p, det, pose, clf, thr, want_poly, p.load_image_ann(ip))
                p.save_image_ann(ip, ia)
                n += len(ia.objects)
                progress(k + 1, len(todo), ip.name)
            p.save()
            return n

        self.b_auto_all.setEnabled(False)
        self._th = JobThread(job)
        self._th.progress.connect(lambda d, t, m: self.lb_auto.setText(f"자동 라벨 {d}/{t} · {m}"))
        self._th.done.connect(lambda n: (self.b_auto_all.setEnabled(True), self.lb_auto.setText(
            f"완료: 객체 {n}개 — '미확정 자동 라벨 있음' 필터로 확인·수정하세요."), self.refresh()))
        self._th.failed.connect(lambda m: (self.b_auto_all.setEnabled(True), self.lb_auto.setText(m.splitlines()[0])))
        self._th.start()

    def _refresh_assist_models(self):
        from ..assist.core import list_models

        p = self.tab.project
        cur = self.cb_assist.currentData()
        self.cb_assist.blockSignals(True)
        self.cb_assist.clear()
        for m in (list_models(p) if p is not None else []):
            self.cb_assist.addItem(f"{m['task']} · {m['engine']} · {m['created']}", m)
        idx = next((i for i in range(self.cb_assist.count())
                    if cur and self.cb_assist.itemData(i)["name"] == cur.get("name")), 0)
        self.cb_assist.setCurrentIndex(idx)
        self.cb_assist.blockSignals(False)
        self.g_assist.setVisible(self.cb_assist.count() > 0)

    def assist_current(self):
        from ..assist.core import Predictor

        p = self.tab.project
        meta = self.cb_assist.currentData()
        if p is None or self.cur is None or not meta:
            return
        self.save_current()
        img = imread(self.cur)
        existing = self.current_ann()
        thr = self.sp_assist_thr.value()
        models_dir = self.tab.models.models_dir()
        key = meta["name"]
        cur = self.cur
        self.b_assist.setEnabled(False)
        self.lb_auto.setText("AI assist running...")

        def job():
            if self._assist_key != key:
                self._assist_pred = Predictor(p, meta, models_dir)
                self._assist_key = key
            return self._assist_pred.predict(img, existing, thr, True)

        self._th = FuncThread(job)
        self._th.done.connect(lambda r: self._apply_assist(cur, r))
        self._th.failed.connect(lambda m: (self.b_assist.setEnabled(True), self.lb_auto.setText(m.splitlines()[0])))
        self._th.start()

    def _apply_assist(self, img_path, res):
        self.b_assist.setEnabled(True)
        if img_path != self.cur:
            return
        ia, unc, added = res
        p = self.tab.project
        img = imread(img_path)
        self.canvas.set_image(img, ia.objects, p.class_names, p.kpt_names, p.kpt_skeleton, p.pose_class_ids)
        if ia.image_class_auto and ia.image_class:
            if self.cb_imgcls.findData(ia.image_class) < 0:
                self.cb_imgcls.addItem(ia.image_class, ia.image_class)
            self.cb_imgcls.blockSignals(True)
            self.cb_imgcls.setCurrentIndex(self.cb_imgcls.findData(ia.image_class))
            self.cb_imgcls.blockSignals(False)
            self._ic_auto, self._ic_score = True, ia.image_class_score
            self.lb_imgcls_note.setText(f"AI suggestion ({ia.image_class_score or 0:.2f}) - confirm or change")
        self.lb_auto.setText(f"AI assist: {added} suggestion(s), uncertainty {unc:.2f}. Fix and press confirm.")
        self._changed()

    def hideEvent(self, e):
        self.save_current()
        super().hideEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self.canvas.fit)


# ============================================================================ 2. 텍스트 → 이미지
T2I_EXAMPLES = {
    "aerial": [
        "damaged :: aerial drone photo of collapsed residential buildings after an earthquake, rubble, dust",
        "damaged :: top-down drone view of a road with a collapsed section and cracks after a landslide",
        "flooded :: aerial drone photo of a flooded town, muddy water covering streets and cars",
        "fire :: drone photo of a burning warehouse with thick black smoke, firefighters and fire trucks",
        "normal :: aerial drone photo of a city intersection with traffic lights and median barriers, daytime",
    ],
    "ground": [
        "damaged :: street-level photo of a fallen traffic light pole lying on the road after a storm",
        "damaged :: photo of a bent and broken road median barrier after a car crash",
        "fire :: firefighters in protective gear spraying water at a burning building, night",
        "normal :: street-level photo of pedestrians crossing at an intersection with traffic lights",
    ],
}


class Txt2ImgPage(QWidget):
    """SDXL로 원본 이미지를 새로 만들고(자동 라벨 포함) 프로젝트 원본에 추가."""

    def __init__(self, win, tab):
        super().__init__()
        from ..genai.prompts import NEGATIVE
        from ..genai.txt2img import SIZES

        self.win, self.tab = win, tab
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        L = QVBoxLayout(left)
        L.setContentsMargins(0, 0, 6, 0)

        g = QGroupBox("프롬프트 — 한 줄에 하나 · '분류클래스 :: 프롬프트'로 쓰면 이미지 분류 라벨도 지정")
        gl = QVBoxLayout(g)
        self.ed_prompts = QPlainTextEdit()
        self.ed_prompts.setPlaceholderText(
            "영어 권장. 예)\n"
            "damaged :: aerial drone photo of collapsed buildings after an earthquake, rubble\n"
            "aerial drone photo of a highway bridge over a river, daytime\n"
            "# 으로 시작하는 줄은 무시")
        self.ed_prompts.setMinimumHeight(150)
        self.ed_prompts.textChanged.connect(self._estimate)
        gl.addWidget(self.ed_prompts)
        r = QHBoxLayout()
        b_ex = QPushButton("예시 넣기")
        b_ex.clicked.connect(self._example)
        r.addWidget(b_ex)
        r.addStretch(1)
        gl.addLayout(r)
        self.ed_neg = QPlainTextEdit(NEGATIVE)
        self.ed_neg.setMaximumHeight(56)
        gl.addWidget(QLabel("네거티브 프롬프트"))
        gl.addWidget(self.ed_neg)
        L.addWidget(g, 1)

        g2 = QGroupBox("생성 설정")
        f = QFormLayout(g2)
        self.cb_domain = QComboBox()
        self.cb_domain.addItem("드론 항공영상 (접두 문구 자동 추가)", "aerial")
        self.cb_domain.addItem("지상 촬영", "ground")
        self.cb_domain.addItem("접두 문구 없음", "none")
        self.cb_size = QComboBox()
        for k, (w_, h_) in SIZES.items():
            self.cb_size.addItem(k, (w_, h_))
        self.sp_n = _ispin(4, 1, 200)
        self.sp_n.valueChanged.connect(self._estimate)
        self.sp_steps = _ispin(30, 4, 80)
        self.sp_steps.valueChanged.connect(self._estimate)
        self.sp_guid = _dspin(6.5, 1.0, 15.0, 0.5, 1)
        self.sp_seed = _ispin(2026, 0, 2 ** 31 - 1)
        self.ed_prefix = QLineEdit("gen")
        f.addRow("도메인", self.cb_domain)
        f.addRow("해상도", self.cb_size)
        f.addRow("프롬프트당 장수", self.sp_n)
        f.addRow("스텝", self.sp_steps)
        f.addRow("가이던스", self.sp_guid)
        f.addRow("시드", self.sp_seed)
        f.addRow("파일명 접두어", self.ed_prefix)
        L.addWidget(g2)

        g3 = QGroupBox("자동 라벨 (생성 직후, 점선=미확정 → 3. 라벨링에서 확인)")
        f3 = QFormLayout(g3)
        self.chk_auto = QCheckBox("자동 라벨 사용")
        self.chk_auto.setChecked(True)
        self.cb_det = QComboBox()
        self.cb_det.addItem("OWLv2", "owlv2")
        self.cb_det.addItem("Grounding DINO", "grounding_dino")
        self.sp_thr = _dspin(0.20, 0.01, 0.95, 0.01)
        self.chk_sam = QCheckBox("SAM 폴리곤")
        self.chk_sam.setChecked(True)
        self.chk_pose = QCheckBox("자세 추정 (ViTPose, 스키마가 사람 17점일 때)")
        self.chk_pose.setChecked(True)
        self.chk_clf = QCheckBox("이미지 분류 (CLIP, '::' 지정이 없는 줄)")
        self.chk_clf.setChecked(True)
        f3.addRow(self.chk_auto)
        f3.addRow("검출기", self.cb_det)
        f3.addRow("임계값", self.sp_thr)
        for c in (self.chk_sam, self.chk_pose, self.chk_clf):
            f3.addRow(c)
        L.addWidget(g3)
        self.lb_est = QLabel("")
        self.lb_est.setWordWrap(True)
        self.lb_est.setObjectName("note")
        L.addWidget(self.lb_est)
        split.addWidget(_scroll(left))

        right = QWidget()
        R = QVBoxLayout(right)
        R.setContentsMargins(6, 0, 0, 0)
        self.view = CaptionedImage("미리보기 — 첫 번째 줄(또는 커서가 있는 줄) 1장 생성")
        R.addWidget(self.view, 3)
        pr = QHBoxLayout()
        self.b_prev = QPushButton("미리보기 (1장)")
        self.b_prev.clicked.connect(self.preview)
        pr.addWidget(self.b_prev)
        pr.addStretch(1)
        R.addLayout(pr)
        self.run = RunBar("▶  생성 → 프로젝트 원본에 추가")
        self.run.btn_run.clicked.connect(self.start)
        self.run.btn_cancel.clicked.connect(lambda: self._th and self._th.cancel.set())
        self.run.btn_open.clicked.connect(lambda: self.tab.project and open_folder(self.tab.project.work_images))
        R.addWidget(self.run, 2)
        split.addWidget(right)
        split.setSizes([520, 700])
        root.addWidget(split)
        self._th = self._pth = None
        self._estimate()

    def _example(self):
        lines = T2I_EXAMPLES.get(self.cb_domain.currentData(), T2I_EXAMPLES["aerial"])
        cur = self.ed_prompts.toPlainText().strip()
        self.ed_prompts.setPlainText((cur + "\n" if cur else "") + "\n".join(lines))

    def _lines(self):
        return [ln.strip() for ln in self.ed_prompts.toPlainText().splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    def _estimate(self):
        n = len(self._lines()) * self.sp_n.value()
        sec = self.sp_steps.value() * 0.25  # RTX 4060급 1024px 대략치 (스텝당 초)
        self.lb_est.setText(f"총 {n}장 생성 예정 · 대략 {n * sec / 60:.0f}분 (RTX 4060·1024px 기준 추정, "
                            "VRAM 8GB는 CPU 오프로드로 더 느림) · 생성본은 work/images 에 gen_ 접두어로 저장")

    def cfg(self):
        from ..genai.prompts import DOMAIN_PREFIX
        from ..genai.txt2img import Txt2ImgConfig

        w_, h_ = self.cb_size.currentData()
        dom = self.cb_domain.currentData()
        return Txt2ImgConfig(
            project_dir=str(self.tab.project.root) if self.tab.project else "", prompts=self._lines(),
            negative=self.ed_neg.toPlainText().strip(), domain=dom if dom in DOMAIN_PREFIX else "none",
            width=w_, height=h_, n_per_prompt=self.sp_n.value(), steps=self.sp_steps.value(),
            guidance=self.sp_guid.value(), seed=self.sp_seed.value(), vram_mode=self.tab.models.vram_mode(),
            models_dir=self.tab.models.models_dir(), autolabel=self.chk_auto.isChecked(),
            detector=self.cb_det.currentData(), threshold=self.sp_thr.value(), use_sam=self.chk_sam.isChecked(),
            auto_pose=self.chk_pose.isChecked(), auto_classify=self.chk_clf.isChecked(),
            name_prefix=self.ed_prefix.text().strip() or "gen")

    def _ready(self) -> bool:
        from ..genai import models as M

        if self.tab.project is None:
            QMessageBox.information(self, "프로젝트", "먼저 프로젝트를 만드세요.")
            return False
        if not self._lines():
            QMessageBox.information(self, "프롬프트", "프롬프트를 한 줄 이상 입력하세요 ([예시 넣기] 참고).")
            return False
        if not M.diffusers_available():
            QMessageBox.warning(self, "생성형 기능", "PyTorch/diffusers가 설치되지 않았습니다.\n"
                                "windows\\install_genai.bat 또는 ubuntu/install_genai.sh를 실행하세요.")
            return False
        if not M.is_installed("sdxl_base", Path(self.tab.models.models_dir())):
            QMessageBox.warning(self, "모델 없음", "SDXL Base 모델이 없습니다. '모델 · GPU'에서 받거나 오프라인 번들로 설치하세요.")
            return False
        return True

    def preview(self):
        if not self._ready():
            return
        from ..genai.prompts import DOMAIN_PREFIX
        from ..genai.transform import TransformConfig, get_engine
        from ..genai.txt2img import parse_prompt_line

        cfg = self.cfg()
        lines = self._lines()
        cur_line = self.ed_prompts.textCursor().block().text().strip()
        ln = cur_line if cur_line in lines else lines[0]
        img_cls, prompt = parse_prompt_line(ln)
        full = DOMAIN_PREFIX.get(cfg.domain, "") + prompt
        self.b_prev.setEnabled(False)
        self.view.set(None, "생성 중… (첫 실행은 모델 로딩으로 1~2분)")

        def job():
            eng = get_engine(TransformConfig(models_dir=cfg.models_dir, vram_mode=cfg.vram_mode), lambda m: None)
            return eng.txt2img(full, cfg.negative, cfg.width, cfg.height, cfg.steps, cfg.guidance, cfg.seed)

        self._pth = FuncThread(job)
        self._pth.done.connect(lambda im: (self.b_prev.setEnabled(True), self.view.set(
            im, f"{im.shape[1]}×{im.shape[0]} · seed {cfg.seed}" + (f" · 분류 {img_cls}" if img_cls else "") +
            " — 미리보기는 저장되지 않습니다")))
        self._pth.failed.connect(lambda m: (self.b_prev.setEnabled(True), self.view.set(None, m.splitlines()[0])))
        self._pth.start()

    def start(self):
        if not self._ready():
            return
        from ..genai.txt2img import run_txt2img

        cfg = self.cfg()
        n = len(cfg.prompts) * cfg.n_per_prompt
        if n > 20 and QMessageBox.question(self, "생성", f"{n}장을 생성합니다. 계속할까요?") != QMessageBox.Yes:
            return
        self.run.log.clear()
        self.run.running(True)

        def job(cfg, progress=None, log=print, cancel=None):
            return run_txt2img(cfg, progress, log, cancel)

        self._th = JobThread(job, cfg)
        self._th.progress.connect(lambda d, t, m: (self.run.bar.setMaximum(t), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(self._done)
        self._th.failed.connect(lambda m: (self.run.running(False), self.run.log.appendPlainText(m)))
        self._th.start()

    def _done(self, rep):
        self.run.running(False)
        p = Project.load(self.tab.project.root)
        self.tab.set_project(p)
        st = rep["stats"]
        if st["generated"] and QMessageBox.question(
                self, "완료", f"신규 원본 {st['generated']}장이 추가되었습니다 (자동 라벨 객체 {st['objects']}개).\n"
                "라벨링 단계에서 확인·수정할까요?") == QMessageBox.Yes:
            self.tab.label.cb_filter.setCurrentIndex(4)
            self.tab.setCurrentWidget(self.tab.label)


# ============================================================================ 7. 내보내기
class ExportPage(QWidget):
    def __init__(self, win, tab):
        super().__init__()
        self.win, self.tab = win, tab
        root = QVBoxLayout(self)
        g = QGroupBox("YOLO 데이터셋으로 내보내기")
        f = QFormLayout(g)
        self.cb_fmt = QComboBox()
        for k, t in FORMATS.items():
            self.cb_fmt.addItem(t, k)
        self.cb_fmt.currentIndexChanged.connect(self._fmt_changed)
        f.addRow("YOLO 태스크", self.cb_fmt)
        self.lb_fmt = QLabel("")
        self.lb_fmt.setWordWrap(True)
        self.lb_fmt.setObjectName("note")
        f.addRow(self.lb_fmt)
        self.chk_orig = QCheckBox("원본(정상) 이미지 포함 — 라벨된 원본 (AI 생성 원본 포함)")
        self.chk_orig.setChecked(True)
        self.chk_unlab = QCheckBox("라벨 없는 원본도 배경 이미지로 포함")
        f.addRow(self.chk_orig)
        f.addRow(self.chk_unlab)
        self.chk_pose_cls = QCheckBox("Pose: 키포인트 대상 클래스만 포함 (권장 — YOLO pose는 모든 객체가 키포인트를 가짐)")
        self.chk_pose_cls.setChecked(True)
        self.chk_pose_req = QCheckBox("Pose: 키포인트가 하나도 없는 객체 제외")
        f.addRow(self.chk_pose_cls)
        f.addRow(self.chk_pose_req)
        self.runs = QListWidget()
        self.runs.setMaximumHeight(170)
        f.addRow("포함할 생성 실행 (검수 통과본만)", self.runs)
        w, self.ed_out = _dir_row("내보낼 폴더 (기본: 프로젝트/export/태스크_날짜)", "내보내기 폴더", self)
        f.addRow("출력 폴더", w)
        root.addWidget(g)
        row = QHBoxLayout()
        self.b_exp = QPushButton("내보내기")
        self.b_exp.setObjectName("primary")
        self.b_exp.setMinimumHeight(34)
        self.b_exp.clicked.connect(self.start)
        self.b_send = QPushButton("② 데이터셋 증강 탭으로 보내기 ▶")
        self.b_send.setMinimumHeight(34)
        self.b_send.setEnabled(False)
        self.b_send.clicked.connect(self.send)
        row.addWidget(self.b_exp, 1)
        row.addWidget(self.b_send, 1)
        root.addLayout(row)
        self.lb = QLabel("")
        self.lb.setObjectName("summary")
        self.lb.setWordWrap(True)
        root.addWidget(self.lb)
        self.run = RunBar("")
        self.run.btn_run.hide()
        self.run.btn_open.clicked.connect(lambda: open_folder(self.last or self.ed_out.text()))
        root.addWidget(self.run, 1)
        tab.project_changed.connect(self.refresh)
        self._th = None
        self.last = None
        self._fmt_changed()

    def _fmt_changed(self):
        fmt = self.cb_fmt.currentData()
        pose = fmt == "pose"
        self.chk_pose_cls.setVisible(pose)
        self.chk_pose_req.setVisible(pose)
        self.chk_unlab.setEnabled(fmt != "classify")
        p = self.tab.project
        note = FMT_NOTES.get(fmt, "")
        if p is not None:
            s = p.stats()
            if pose:
                note += (f"<br>키포인트 스키마 K={len(p.kpt_names)} · 대상 클래스: "
                         f"{', '.join(p.kpt.get('classes', [])) or '(없음)'} · 키포인트 객체 {s['kpt_objects']}개")
            if fmt == "classify":
                note += f"<br>분류가 지정된 이미지 {s['classified']}장만 포함됩니다 (<클래스>/이미지 폴더 구조)."
            if s["auto"]:
                note += f"<br><span style='color:#b45309'>미확정 자동 라벨 {s['auto']}개도 그대로 내보내집니다.</span>"
        self.lb_fmt.setText(note)

    def refresh(self):
        p = self.tab.project
        self.runs.clear()
        if p is None:
            return
        self.cb_fmt.blockSignals(True)
        self.cb_fmt.setCurrentIndex(max(0, self.cb_fmt.findData(p.fmt)))
        self.cb_fmt.blockSignals(False)
        self._fmt_changed()
        import json
        for r in p.runs():
            try:
                rep = json.loads((r / "gen_report.json").read_text(encoding="utf-8"))
                n = sum(1 for _ in (r / "images").glob("*"))
                effs = sorted({x.get("effect", "") for x in rep.get("records", [])})
                txt = f"{r.name} — 통과 {n}장 · {', '.join(effs)[:80]}"
            except Exception:
                txt = r.name
            it = QListWidgetItem(txt)
            it.setData(Qt.UserRole, str(r))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.runs.addItem(it)

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()

    def start(self):
        import time

        from ..genai.export import export_dataset

        p = self.tab.project
        if p is None:
            QMessageBox.information(self, "내보내기", "먼저 프로젝트를 여세요.")
            return
        fmt = self.cb_fmt.currentData()
        if fmt == "pose" and not p.kpt_names:
            QMessageBox.warning(self, "Pose", "키포인트 스키마가 없습니다. 1. 프로젝트 · 원본에서 먼저 정의하세요.")
            return
        if fmt == "classify" and not p.stats()["classified"]:
            QMessageBox.warning(self, "Classification", "이미지 분류가 지정된 이미지가 없습니다. "
                                "라벨링 단계에서 이미지 분류를 지정하거나 CLIP 자동 분류를 실행하세요.")
            return
        out = self.ed_out.text().strip() or str(p.root / "export" / f"{fmt}_{time.strftime('%Y%m%d_%H%M%S')}")
        self.ed_out.setText(out)
        runs = [self.runs.item(i).data(Qt.UserRole) for i in range(self.runs.count())
                if self.runs.item(i).checkState() == Qt.Checked]
        orig, unlab = self.chk_orig.isChecked(), self.chk_unlab.isChecked() and fmt != "classify"
        pcls, preq = self.chk_pose_cls.isChecked(), self.chk_pose_req.isChecked()
        self.b_exp.setEnabled(False)

        def job(progress=None, log=print, cancel=None):
            return export_dataset(p, out, fmt, orig, unlab, runs, progress, log, cancel,
                                  pose_classes_only=pcls, pose_require_kpts=preq)

        self._th = JobThread(job)
        self._th.progress.connect(lambda d, t, m: (self.run.bar.setMaximum(t), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(self._done)
        self._th.failed.connect(lambda m: (self.b_exp.setEnabled(True), self.run.log.appendPlainText(m)))
        self._th.start()

    def _done(self, s):
        self.b_exp.setEnabled(True)
        self.b_send.setEnabled(True)
        self.last = self.ed_out.text()
        pc = ", ".join(f"{k} {v}" for k, v in s["per_class"].items())
        unit = "장" if s["format"] == "classify" else "개"
        extra = f" · 분류 없음 {s['skipped']}장 제외" if s.get("skipped") else ""
        self.lb.setText(f"<b>{FORMATS[s['format']]}</b> · 원본 {s['orig']}장 + 생성 {s['generated']}장"
                        + ("" if s["format"] == "classify" else f" · 객체 {s['objects']}개") + extra +
                        f"<br>클래스별({unit}): {pc}<br>{self.last}")
        self.tab.set_project(self.tab.project)
        self.ed_out.clear()

    def send(self):
        if self.last:
            self.win.send_to_augment(self.last)
