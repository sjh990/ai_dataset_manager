"""마법사 각 단계 페이지."""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from ..dataset import analyze_dataset, describe, scan_dataset
from ..labels import TASK_LABELS, TASKS
from ..ops_spec import (AUGMENT_OPS, COMMON_AUG_PARAMS, PREPROCESS_OPS, defaults)
from ..splitter import split_samples
from ..versions import (VERSION_NAMES, VERSIONS, is_compatible, offline_recommended_preset, online_overlap_ops,
                        trainer_hyp_preset)
from ..visualize import draw
from .widgets import CaptionedImage, FuncThread, ParamForm, RatioBar


def _title(text: str, sub: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 6)
    t = QLabel(text)
    t.setObjectName("pageTitle")
    lay.addWidget(t)
    if sub:
        s = QLabel(sub)
        s.setObjectName("pageSub")
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


def _scroll(inner: QWidget) -> QScrollArea:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(inner)
    sc.setFrameShape(QScrollArea.NoFrame)
    return sc


class BasePage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win

    def enter(self):
        pass

    def leave(self) -> bool:
        return True


# =============================================================================== 0. 데이터셋 · 모델
class DatasetPage(BasePage):
    analyzed = Signal()

    def __init__(self, win):
        super().__init__(win)
        root = QVBoxLayout(self)
        root.addWidget(_title("데이터셋 · 모델 선택",
                              "이미지와 YOLO 라벨(.txt)이 들어있는 폴더를 선택하면 폴더 구조, 태스크(Detect/Seg/Pose/OBB, 또는 클래스 폴더 구조면 Classify), "
                              "클래스, 해상도, 모델 버전 단서를 자동으로 분석합니다."))
        g = QGroupBox("입력 데이터셋")
        gl = QHBoxLayout(g)
        self.ed_ds = QLineEdit()
        self.ed_ds.setPlaceholderText("예) D:/datasets/my_drone  (Ultralytics · Roboflow · 평면 구조 모두 지원)")
        b = QPushButton("찾아보기…")
        b.clicked.connect(self._browse_ds)
        self.btn_an = QPushButton("분석")
        self.btn_an.setObjectName("primary")
        self.btn_an.clicked.connect(self.run_analysis)
        gl.addWidget(self.ed_ds, 1)
        gl.addWidget(b)
        gl.addWidget(self.btn_an)
        root.addWidget(g)

        split = QSplitter(Qt.Horizontal)
        self.info_view = QTextBrowser()
        self.info_view.setOpenExternalLinks(False)
        self.info_view.setHtml("<span style='color:#888'>폴더를 선택하고 [분석]을 누르세요.</span>")
        split.addWidget(self.info_view)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        gm = QGroupBox("모델 · 태스크")
        gml = QGridLayout(gm)
        self.cb_task = QComboBox()
        for t in TASKS:
            self.cb_task.addItem(TASK_LABELS[t], t)
        self.cb_task.currentIndexChanged.connect(self._on_task_version)
        self.cb_ver = QComboBox()
        self.cb_ver.addItem("자동 추천", "auto")
        for v in VERSION_NAMES:
            self.cb_ver.addItem(v, v)
        self.cb_ver.currentIndexChanged.connect(self._on_task_version)
        self.lb_ver_note = QLabel()
        self.lb_ver_note.setWordWrap(True)
        self.lb_ver_note.setObjectName("note")
        gml.addWidget(QLabel("태스크"), 0, 0)
        gml.addWidget(self.cb_task, 0, 1)
        gml.addWidget(QLabel("YOLO 버전"), 1, 0)
        gml.addWidget(self.cb_ver, 1, 1)
        gml.addWidget(self.lb_ver_note, 2, 0, 1, 2)
        self.pose_box = QWidget()
        pl = QGridLayout(self.pose_box)
        pl.setContentsMargins(0, 0, 0, 0)
        self.ed_kpt = QLineEdit()
        self.ed_kpt.setPlaceholderText("예) 17,3")
        self.ed_flip = QLineEdit()
        self.ed_flip.setPlaceholderText("예) 0,2,1,4,3 (좌우 반전 시 교환 인덱스)")
        pl.addWidget(QLabel("kpt_shape"), 0, 0)
        pl.addWidget(self.ed_kpt, 0, 1)
        pl.addWidget(QLabel("flip_idx"), 1, 0)
        pl.addWidget(self.ed_flip, 1, 1)
        gml.addWidget(self.pose_box, 3, 0, 1, 2)
        rl.addWidget(gm)

        gc = QGroupBox("클래스 (이름 더블클릭으로 수정)")
        gcl = QVBoxLayout(gc)
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["ID", "이름", "객체 수"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        gcl.addWidget(self.tbl)
        rl.addWidget(gc, 1)
        split.addWidget(right)
        split.setSizes([560, 420])
        root.addWidget(split, 1)

        go = QGroupBox("출력 폴더")
        gol = QHBoxLayout(go)
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("증강 결과가 저장될 새 폴더")
        b2 = QPushButton("찾아보기…")
        b2.clicked.connect(self._browse_out)
        gol.addWidget(self.ed_out, 1)
        gol.addWidget(b2)
        root.addWidget(go)
        self.pose_box.setVisible(False)
        self._thread = None

    # -- actions
    def _browse_ds(self):
        d = QFileDialog.getExistingDirectory(self, "데이터셋 폴더 선택", self.ed_ds.text() or str(Path.home()))
        if d:
            self.ed_ds.setText(d)
            self.run_analysis()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "출력 폴더 선택", self.ed_out.text() or self.ed_ds.text())
        if d:
            self.ed_out.setText(d)

    def run_analysis(self):
        path = self.ed_ds.text().strip()
        if not path or not Path(path).is_dir():
            QMessageBox.warning(self, "경로 오류", "존재하는 데이터셋 폴더를 선택하세요.")
            return
        self.btn_an.setEnabled(False)
        self.info_view.setHtml("분석 중… (이미지·라벨 수에 따라 수 초~수십 초)")
        out = self.ed_out.text().strip() or None
        self._thread = FuncThread(lambda: analyze_dataset(scan_dataset(path, exclude=out)))
        self._thread.done.connect(self._on_analyzed)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _on_failed(self, msg):
        self.btn_an.setEnabled(True)
        self.info_view.setPlainText("분석 실패\n" + msg)

    def _on_analyzed(self, info):
        self.btn_an.setEnabled(True)
        self.win.info = info
        self.info_view.setHtml(describe(info))
        # 태스크 콤보: 자동 판별 표시
        for i, t in enumerate(TASKS):
            self.cb_task.setItemText(i, TASK_LABELS[t] + ("  (자동)" if t == info.task else ""))
        self.cb_task.setCurrentIndex(TASKS.index(info.task))
        self.cb_ver.setItemText(0, f"자동 추천 → {info.recommended_version}")
        self.cb_ver.setCurrentIndex(0)
        self.ed_kpt.setText(",".join(map(str, info.kpt_shape)) if info.kpt_shape else "")
        self.ed_flip.setText(",".join(map(str, info.flip_idx)) if info.flip_idx else "")
        self.tbl.setHorizontalHeaderLabels(["ID", "이름", "이미지 수" if info.task == "classify" else "객체 수"])
        self.tbl.setRowCount(len(info.class_names))
        for i, n in enumerate(info.class_names):
            it = QTableWidgetItem(str(i))
            it.setFlags(Qt.ItemIsEnabled)
            self.tbl.setItem(i, 0, it)
            nm = QTableWidgetItem(n)
            if info.task == "classify":  # 분류는 폴더 이름이 곧 클래스
                nm.setFlags(Qt.ItemIsEnabled)
                nm.setToolTip("Classification은 폴더 이름이 클래스입니다 (이름을 바꾸려면 폴더명을 변경).")
            self.tbl.setItem(i, 1, nm)
            c = QTableWidgetItem(str(info.class_counts.get(i, 0)))
            c.setFlags(Qt.ItemIsEnabled)
            c.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.tbl.setItem(i, 2, c)
        self.tbl.resizeColumnToContents(0)
        if not self.ed_out.text().strip():
            self.ed_out.setText(str(info.root.parent / f"{info.root.name}_augmented"))
        self._on_task_version()
        self.analyzed.emit()

    def task(self) -> str:
        return self.cb_task.currentData()

    def version(self) -> str:
        v = self.cb_ver.currentData()
        if v == "auto":
            return self.win.info.recommended_version if self.win.info else "YOLO11"
        return v

    def _on_task_version(self):
        t, v = self.task(), self.version()
        self.pose_box.setVisible(t == "pose")
        note = f"<b>{v}</b>: {VERSIONS[v]['note']}"
        if not is_compatible(v, t):
            ok = [n for n in VERSION_NAMES if is_compatible(n, t)]
            note += f"<br><span style='color:#c0392b'>⚠ {v}는 {TASK_LABELS[t]}를 공식 지원하지 않습니다. 호환: {', '.join(ok)}</span>"
        self.lb_ver_note.setText(note)

    def class_names(self) -> list[str]:
        return [(self.tbl.item(i, 1).text().strip() or f"class_{i}") for i in range(self.tbl.rowCount())]

    def _parse_ints(self, text):
        text = text.strip()
        if not text:
            return None
        return [int(x) for x in text.replace(" ", "").replace("[", "").replace("]", "").split(",") if x]

    def leave(self) -> bool:
        info = self.win.info
        if info is None:
            QMessageBox.information(self, "분석 필요", "먼저 데이터셋을 선택하고 [분석]을 실행하세요.")
            return False
        if info.n_images == 0:
            QMessageBox.warning(self, "이미지 없음", "선택한 폴더에서 이미지를 찾지 못했습니다.")
            return False
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "출력 폴더", "출력 폴더를 지정하세요.")
            return False
        if Path(out).resolve() == info.root:
            QMessageBox.warning(self, "출력 폴더", "출력 폴더는 입력 데이터셋과 달라야 합니다.")
            return False
        cfg = self.win.cfg
        cfg.dataset_dir = str(info.root)
        cfg.output_dir = out
        cfg.task = self.task()
        cfg.yolo_version = self.version()
        cfg.class_names = self.class_names()
        try:
            cfg.kpt_shape = self._parse_ints(self.ed_kpt.text()) if cfg.task == "pose" else None
            cfg.flip_idx = self._parse_ints(self.ed_flip.text()) if cfg.task == "pose" else None
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "kpt_shape / flip_idx는 쉼표로 구분된 정수여야 합니다.")
            return False
        if cfg.task == "pose":
            if not cfg.kpt_shape or len(cfg.kpt_shape) != 2:
                QMessageBox.warning(self, "Pose 설정", "kpt_shape를 '키포인트수,차원(2|3)' 형식으로 입력하세요.")
                return False
            if cfg.flip_idx and len(cfg.flip_idx) != cfg.kpt_shape[0]:
                QMessageBox.warning(self, "Pose 설정", "flip_idx 길이가 키포인트 수와 같아야 합니다.")
                return False
        if self.win.max_step < 4:  # 아직 증강 단계를 보지 않았다면 태스크/버전에 맞춰 추천 증강 갱신
            self.win.p_aug_sel._offline()
            self.win.aug_selected = self.win.p_aug_sel.selected()
        return True

    def load_cfg(self, cfg):
        self.ed_ds.setText(cfg.dataset_dir)
        self.ed_out.setText(cfg.output_dir)


# =============================================================================== 1. 분할
class SplitPage(BasePage):
    def __init__(self, win):
        super().__init__(win)
        root = QVBoxLayout(self)
        root.addWidget(_title("0. 데이터셋 분할 비율",
                              "train / val / test 비율을 정합니다. 증강은 기본적으로 train에만 적용되어 "
                              "검증·테스트 성능이 부풀려지지 않도록 합니다."))
        g = QGroupBox("비율 (%)")
        gl = QGridLayout(g)
        self.sp_train = QSpinBox()
        self.sp_train.setRange(1, 100)
        self.sp_val = QSpinBox()
        self.sp_val.setRange(0, 99)
        self.lb_test = QLabel()
        self.lb_test.setObjectName("big")
        for s in (self.sp_train, self.sp_val):
            s.setSuffix(" %")
            s.valueChanged.connect(self._update)
        gl.addWidget(QLabel("train"), 0, 0)
        gl.addWidget(self.sp_train, 0, 1)
        gl.addWidget(QLabel("val"), 0, 2)
        gl.addWidget(self.sp_val, 0, 3)
        gl.addWidget(QLabel("test (나머지)"), 0, 4)
        gl.addWidget(self.lb_test, 0, 5)
        pr = QHBoxLayout()
        for a, b, c in ((70, 20, 10), (80, 10, 10), (80, 20, 0), (60, 20, 20), (90, 10, 0)):
            bt = QPushButton(f"{a}/{b}/{c}")
            bt.clicked.connect(lambda _=False, a=a, b=b: self._preset(a, b))
            pr.addWidget(bt)
        pr.addStretch(1)
        gl.addLayout(pr, 1, 0, 1, 6)
        self.bar = RatioBar()
        gl.addWidget(self.bar, 2, 0, 1, 6)
        root.addWidget(g)

        g2 = QGroupBox("옵션")
        g2l = QGridLayout(g2)
        self.chk_strat = QCheckBox("클래스 층화 분할 (각 이미지의 주 클래스 비율을 split마다 유사하게 유지)")
        self.chk_keep = QCheckBox("기존 분할 유지 (폴더에 train/val/test가 있으면 그대로 사용, 없는 이미지만 위 비율로 배정)")
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2**31 - 1)
        self.chk_strat.toggled.connect(self._update)
        self.chk_keep.toggled.connect(self._update)
        self.sp_seed.valueChanged.connect(self._update)
        g2l.addWidget(self.chk_strat, 0, 0, 1, 3)
        g2l.addWidget(self.chk_keep, 1, 0, 1, 3)
        g2l.addWidget(QLabel("랜덤 시드"), 2, 0)
        g2l.addWidget(self.sp_seed, 2, 1)
        g2l.setColumnStretch(2, 1)
        root.addWidget(g2)
        self.lb_detail = QLabel()
        self.lb_detail.setObjectName("note")
        self.lb_detail.setWordWrap(True)
        root.addWidget(self.lb_detail)
        root.addStretch(1)

    def _preset(self, a, b):
        self.sp_train.setValue(a)
        self.sp_val.setValue(b)

    def _update(self):
        tr = self.sp_train.value()
        self.sp_val.blockSignals(True)
        self.sp_val.setMaximum(100 - tr)
        self.sp_val.blockSignals(False)
        va = self.sp_val.value()
        te = 100 - tr - va
        self.lb_test.setText(f"{te} %")
        info = self.win.info
        counts = [0, 0, 0]
        detail = ""
        if info and info.samples:
            sp = split_samples(info.samples, tr, va, te, self.sp_seed.value(), self.chk_strat.isChecked(),
                               self.chk_keep.isChecked())
            counts = [len(sp["train"]), len(sp["val"]), len(sp["test"])]
            detail = f"원본 {info.n_images}장 → train {counts[0]} · val {counts[1]} · test {counts[2]}"
            if va == 0:
                detail += "<br><span style='color:#c0392b'>⚠ val이 0%이면 data.yaml의 val이 train을 가리키게 됩니다.</span>"
        self.bar.set([tr, va, te], counts)
        self.lb_detail.setText(detail)

    def enter(self):
        info = self.win.info
        self.chk_keep.setEnabled(bool(info and info.has_splits))
        if not (info and info.has_splits):
            self.chk_keep.setChecked(False)
        self._update()

    def load_cfg(self, cfg):
        s = cfg.split
        self.sp_train.setValue(int(s.get("train", 70)))
        self.sp_val.setValue(int(s.get("val", 20)))
        self.sp_seed.setValue(int(s.get("seed", 42)))
        self.chk_strat.setChecked(bool(s.get("stratify", True)))
        self.chk_keep.setChecked(bool(s.get("keep_existing", False)))
        self._update()

    def leave(self):
        tr, va = self.sp_train.value(), self.sp_val.value()
        self.win.cfg.split = dict(train=tr, val=va, test=100 - tr - va, seed=self.sp_seed.value(),
                                  stratify=self.chk_strat.isChecked(), keep_existing=self.chk_keep.isChecked())
        self.win.cfg.seed = self.sp_seed.value()
        return True


# =============================================================================== 공용: 연산 선택 페이지
class OpSelectPage(BasePage):
    def __init__(self, win, title, sub, ops, kind):
        super().__init__(win)
        self.ops, self.kind = ops, kind
        root = QVBoxLayout(self)
        root.addWidget(_title(title, sub))
        self.top = QHBoxLayout()
        root.addLayout(self.top)
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setObjectName("banner")
        self.banner.setVisible(False)
        root.addWidget(self.banner)
        inner = QWidget()
        il = QVBoxLayout(inner)
        self.checks: dict[str, QCheckBox] = {}
        self.tags: dict[str, QLabel] = {}
        groups: OrderedDict[str, list] = OrderedDict()
        for o in ops:
            groups.setdefault(o.group, []).append(o)
        for gname, lst in groups.items():
            gb = QGroupBox(gname)
            gl = QGridLayout(gb)
            gl.setColumnStretch(1, 1)
            for r, o in enumerate(lst):
                cb = QCheckBox(o.name)
                cb.setObjectName("opCheck")
                cb.toggled.connect(self._changed)
                d = QLabel(o.desc)
                d.setWordWrap(True)
                d.setObjectName("opDesc")
                tag = QLabel("")
                tag.setObjectName("tag")
                self.checks[o.key] = cb
                self.tags[o.key] = tag
                gl.addWidget(cb, r, 0, Qt.AlignTop)
                gl.addWidget(d, r, 1)
                gl.addWidget(tag, r, 2, Qt.AlignTop)
            il.addWidget(gb)
        il.addStretch(1)
        root.addWidget(_scroll(inner), 1)
        self.lb_count = QLabel()
        self.lb_count.setObjectName("note")
        root.addWidget(self.lb_count)
        self._changed()

    def add_button(self, text, fn, tip=""):
        b = QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(fn)
        self.top.addWidget(b)
        return b

    def _changed(self):
        n = sum(cb.isChecked() for cb in self.checks.values())
        self.lb_count.setText(f"선택됨: {n}개 — 다음 단계에서 각 항목의 수치를 입력합니다.")

    def selected(self) -> list[str]:
        return [o.key for o in self.ops if self.checks[o.key].isChecked()]

    def set_selected(self, keys):
        for k, cb in self.checks.items():
            cb.setChecked(k in keys)


class PreSelectPage(OpSelectPage):
    def __init__(self, win):
        super().__init__(win, "1. 전처리 종류 선택",
                         "전처리는 train/val/test 모든 split에 동일하게 적용됩니다 (학습·평가 조건 일치).",
                         PREPROCESS_OPS, "pre")
        self.add_button("권장 조합 적용", self._recommend, "데이터셋 해상도에 맞춘 권장 전처리")
        self.add_button("모두 해제", lambda: self.set_selected([]))
        self.top.addStretch(1)

    def _recommend(self):
        info = self.win.info
        keys = ["resize", "filter_small"]
        if info and info.size_stats and max(info.size_stats["max_w"], info.size_stats["max_h"]) >= 1920:
            keys = ["tile", "resize", "filter_small", "drop_empty"]
        self.set_selected(keys)

    def enter(self):
        info = self.win.info
        if info and info.size_stats:
            s = info.size_stats
            self.banner.setText(f"데이터셋 해상도: {s['min_w']}×{s['min_h']} ~ {s['max_w']}×{s['max_h']} "
                                f"(중앙값 {s['med_w']}×{s['med_h']}). YOLO 학습 기본 입력은 640입니다.")
            self.banner.setVisible(True)

    def leave(self):
        self.win.pre_selected = self.selected()
        return True


class AugSelectPage(OpSelectPage):
    def __init__(self, win):
        super().__init__(win, "3. 증강 종류 선택",
                         "선택한 split(기본 train)에 대해, 원본 1장당 '증강 배수'만큼 새로운 이미지를 생성합니다.",
                         AUGMENT_OPS, "aug")
        self.add_button("오프라인 권장 (학습기와 중복 최소화)", self._offline,
                        "학습기가 이미 하는 HSV/scale/flip/mosaic은 빼고 블러·노이즈·압축 등 실환경 열화 위주")
        self.add_button("학습기 기본 hyp 재현", self._trainer, "선택한 YOLO 버전의 학습 기본 증강을 그대로 오프라인 적용")
        self.add_button("모두 해제", lambda: self.set_selected([]))
        self.top.addStretch(1)

    def _apply_preset(self, preset: dict):
        self.set_selected(list(preset))
        for k, prm in preset.items():
            base = dict(self.win.aug_params.get(k) or {})
            base.update(prm)
            self.win.aug_params[k] = base

    def _offline(self):
        self._apply_preset(offline_recommended_preset(self.win.cfg.yolo_version, self.win.cfg.task))

    def _trainer(self):
        self._apply_preset(trainer_hyp_preset(self.win.cfg.yolo_version))

    def enter(self):
        v = self.win.cfg.yolo_version
        h = VERSIONS[v]["hyp"]
        overlap = set(online_overlap_ops(v))
        self.banner.setText(
            f"<b>{v}</b> 학습기는 학습 중 이미 다음 증강을 매 epoch 자동 적용합니다: "
            f"HSV(h {h['hsv_h']}, s {h['hsv_s']}, v {h['hsv_v']}), translate ±{h['translate']}, "
            f"scale ±{h['scale']}, 좌우반전 {h['fliplr']}, mosaic {h['mosaic']}"
            + (f", mixup {h['mixup']}" if h['mixup'] else "") +
            ". <b>⟳</b> 표시 항목을 오프라인으로도 강하게 적용하면 과증강될 수 있습니다.")
        self.banner.setVisible(True)
        for k, tag in self.tags.items():
            tag.setText("⟳ 학습기 기본" if k in overlap else "")

    def leave(self):
        self.win.aug_selected = self.selected()
        return True


# =============================================================================== 공용: 수치 입력 + 미리보기 페이지
class ParamPage(BasePage):
    def __init__(self, win, title, sub, ops, kind):
        super().__init__(win)
        self.ops = {o.key: o for o in ops}
        self.kind = kind
        self.forms: dict[str, ParamForm] = {}
        self.boxes: dict[str, QGroupBox] = {}
        self.sample_idx = 0
        self.shuffle_seed = 0
        root = QVBoxLayout(self)
        root.addWidget(_title(title, sub))
        split = QSplitter(Qt.Horizontal)
        inner = QWidget()
        self.form_lay = QVBoxLayout(inner)
        self.empty = QLabel("선택된 항목이 없습니다. 이전 단계에서 선택하거나 그대로 진행하세요.")
        self.empty.setObjectName("note")
        self.form_lay.addWidget(self.empty)
        for o in ops:
            gb = QGroupBox(o.name)
            gl = QVBoxLayout(gb)
            if o.params:
                f = ParamForm(o.params)
                f.changed.connect(self._schedule)
                self.forms[o.key] = f
                gl.addWidget(f)
            else:
                gl.addWidget(QLabel("입력할 수치가 없는 항목입니다."))
            self.boxes[o.key] = gb
            self.form_lay.addWidget(gb)
        if kind == "aug":
            gb = QGroupBox("공통 설정 (모든 기하 변환에 적용)")
            gl = QVBoxLayout(gb)
            self.common_form = ParamForm(COMMON_AUG_PARAMS)
            self.common_form.changed.connect(self._schedule)
            gl.addWidget(self.common_form)
            self.form_lay.addWidget(gb)
        self.form_lay.addStretch(1)
        left = _scroll(inner)
        left.setMinimumWidth(360)
        split.addWidget(left)

        prev = QWidget()
        pl = QVBoxLayout(prev)
        pl.setContentsMargins(6, 0, 0, 0)
        bar = QHBoxLayout()
        self.lb_prev = QLabel("미리보기")
        self.lb_prev.setObjectName("sectionTitle")
        bar.addWidget(self.lb_prev)
        bar.addStretch(1)
        b_other = QPushButton("다른 샘플")
        b_other.clicked.connect(self._next_sample)
        bar.addWidget(b_other)
        if kind == "aug":
            b_sh = QPushButton("다시 섞기")
            b_sh.clicked.connect(self._reshuffle)
            bar.addWidget(b_sh)
        pl.addLayout(bar)
        grid = QGridLayout()
        grid.setSpacing(6)
        if kind == "pre":
            self.views = [CaptionedImage("원본"), CaptionedImage("전처리 결과")]
            grid.addWidget(self.views[0], 0, 0)
            grid.addWidget(self.views[1], 0, 1)
        else:
            self.views = [CaptionedImage("전처리된 원본")] + [CaptionedImage(f"증강 #{i + 1}") for i in range(5)]
            for i, v in enumerate(self.views):
                grid.addWidget(v, i // 3, i % 3)
        pl.addLayout(grid, 1)
        self.lb_stat = QLabel()
        self.lb_stat.setObjectName("note")
        self.lb_stat.setWordWrap(True)
        pl.addWidget(self.lb_stat)
        split.addWidget(prev)
        split.setSizes([380, 700])
        root.addWidget(split, 1)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.refresh_preview)

    # --- state
    def selected(self):
        return self.win.pre_selected if self.kind == "pre" else self.win.aug_selected

    def store(self):
        return self.win.pre_params if self.kind == "pre" else self.win.aug_params

    def enter(self):
        sel = set(self.selected())
        st = self.store()
        for k, gb in self.boxes.items():
            gb.setVisible(k in sel)
            if k in sel and k in self.forms:
                vals = dict(defaults(self.ops[k]))
                vals.update(st.get(k) or {})
                self.forms[k].set_values(vals)
        self.empty.setVisible(not sel)
        if self.kind == "aug":
            self.common_form.set_values(self.win.cfg.aug_common)
        self.refresh_preview()

    def collect(self) -> dict:
        out = {}
        for k in self.selected():
            out[k] = self.forms[k].values() if k in self.forms else {}
        return out

    def leave(self):
        vals = self.collect()
        self.store().update(vals)
        if self.kind == "pre":
            self.win.cfg.preprocess = vals
        else:
            if "scale" in vals and vals["scale"]["min_scale"] > vals["scale"]["max_scale"]:
                QMessageBox.warning(self, "입력 오류", "확대/축소: 최소 배율이 최대 배율보다 큽니다.")
                return False
            if "gamma" in vals and vals["gamma"]["min_gamma"] > vals["gamma"]["max_gamma"]:
                QMessageBox.warning(self, "입력 오류", "감마: 최소값이 최대값보다 큽니다.")
                return False
            self.win.cfg.augment = vals
            self.win.cfg.aug_common = self.common_form.values()
        return True

    # --- preview
    def _schedule(self):
        self.timer.start()

    def _next_sample(self):
        self.sample_idx += 1
        self.refresh_preview()

    def _reshuffle(self):
        self.shuffle_seed += 1
        self.refresh_preview()

    def _pick_sample(self):
        info = self.win.info
        labeled = [s for s in info.samples if s.label is not None]
        pool = labeled or info.samples
        rng = np.random.default_rng(1234)
        order = rng.permutation(len(pool))
        return pool[int(order[self.sample_idx % len(pool)])], pool

    def refresh_preview(self):
        from ..pipeline import preview_samples

        info = self.win.info
        if not info or not info.samples or not self.isVisible():
            return
        cfg = self.win.preview_cfg()
        if self.kind == "pre":
            cfg.preprocess = self.collect()
            cfg.augment = {}
        else:
            cfg.augment = self.collect()
            cfg.aug_common = self.common_form.values()
        sample, pool = self._pick_sample()
        try:
            res = preview_samples(cfg, sample, n_aug=0 if self.kind == "pre" else 5, seed=self.shuffle_seed,
                                  pool=pool)
        except Exception as e:  # 잘못된 조합에도 GUI는 살아있게
            self.lb_stat.setText(f"<span style='color:#c0392b'>미리보기 오류: {e}</span>")
            return
        if res is None:
            self.lb_stat.setText("이미지를 읽을 수 없습니다.")
            return
        (oimg, oanns), pieces, base, augs = res
        names, task = cfg.class_names, cfg.task
        h0, w0 = oimg.shape[:2]
        ic = sample.cls_name if task == "classify" else None

        def dr(im, an, tk, nm):  # classify: 이미지 클래스 표시
            return draw(im, an, tk, nm, None, ic)
        if self.kind == "pre":
            self.views[0].set(dr(oimg, oanns, task, names), f"원본 {w0}×{h0} · 객체 {len(oanns)}")
            bimg = base[1]
            self.views[1].set(dr(bimg, base[2], task, names),
                              f"전처리 결과 {bimg.shape[1]}×{bimg.shape[0]} · 객체 {len(base[2])}")
            msg = f"샘플: {sample.image.name}"
            if len(pieces) > 1:
                msg += f" · 타일 {len(pieces)}개 생성 (라벨이 가장 많은 타일 표시)"
            self.lb_stat.setText(msg)
        else:
            bimg = base[1]
            self.views[0].set(dr(bimg, base[2], task, names), f"전처리된 원본 · 객체 {len(base[2])}")
            for v, (aimg, aanns) in zip(self.views[1:], augs):
                v.set(dr(aimg, aanns, task, names), f"증강 · 객체 {len(aanns)}")
            self.lb_stat.setText(f"샘플: {sample.image.name} · 각 칸은 서로 다른 랜덤 증강 결과입니다. "
                                 "라벨(박스/폴리곤/키포인트)이 객체와 잘 맞는지 확인하세요.")

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self.refresh_preview)

    def load_params(self, params: dict):
        self.store().update(params)


class PreParamPage(ParamPage):
    def __init__(self, win):
        super().__init__(win, "2. 전처리 수치 입력",
                         "선택한 전처리의 수치를 입력하세요. 오른쪽에서 결과를 바로 확인할 수 있습니다.",
                         PREPROCESS_OPS, "pre")


class AugParamPage(ParamPage):
    def __init__(self, win):
        super().__init__(win, "4. 증강 수치 입력",
                         "각 증강의 '적용 확률'과 강도를 입력하세요. 증강본 1장마다 항목별로 확률에 따라 독립적으로 적용됩니다.",
                         AUGMENT_OPS, "aug")


# =============================================================================== 5. 배수 · 실행
class RunPage(BasePage):
    def __init__(self, win):
        super().__init__(win)
        root = QVBoxLayout(self)
        root.addWidget(_title("5. 증강 배수 선택 · 실행", "원본 1장당 몇 장의 증강본을 만들지 정하고 실행합니다."))
        g = QGroupBox("증강 배수")
        gl = QGridLayout(g)
        self.sp_mult = QSpinBox()
        self.sp_mult.setRange(0, 100)
        self.sp_mult.setValue(3)
        self.sp_mult.setSuffix(" 배")
        self.sp_mult.valueChanged.connect(self._estimate)
        self.chk_orig = QCheckBox("원본(전처리본)도 함께 저장")
        self.chk_orig.setChecked(True)
        self.chk_orig.toggled.connect(self._estimate)
        gl.addWidget(QLabel("원본 1장당 증강본"), 0, 0)
        gl.addWidget(self.sp_mult, 0, 1)
        gl.addWidget(self.chk_orig, 0, 2)
        quick = QHBoxLayout()
        for m in (1, 2, 3, 5, 10):
            b = QPushButton(f"×{m}")
            b.clicked.connect(lambda _=False, m=m: self.sp_mult.setValue(m))
            quick.addWidget(b)
        quick.addStretch(1)
        gl.addLayout(quick, 1, 0, 1, 3)
        gl.addWidget(QLabel("증강 적용 split"), 2, 0)
        sl = QHBoxLayout()
        self.chk_split = {s: QCheckBox(s) for s in ("train", "val", "test")}
        self.chk_split["train"].setChecked(True)
        for c in self.chk_split.values():
            c.toggled.connect(self._estimate)
            sl.addWidget(c)
        sl.addWidget(QLabel("  (val/test 증강은 평가 신뢰도를 떨어뜨리므로 권장하지 않음)"))
        sl.addStretch(1)
        gl.addLayout(sl, 2, 1, 1, 2)
        gl.setColumnStretch(2, 1)
        root.addWidget(g)

        g2 = QGroupBox("저장 옵션")
        g2l = QHBoxLayout(g2)
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItem("원본 확장자 유지", "keep")
        self.cb_fmt.addItem("JPG", "jpg")
        self.cb_fmt.addItem("PNG", "png")
        self.sp_q = QSpinBox()
        self.sp_q.setRange(50, 100)
        self.sp_q.setValue(95)
        self.sp_w = QSpinBox()
        self.sp_w.setRange(1, 64)
        self.sp_w.setValue(max(1, min(8, (os.cpu_count() or 2))))
        for lab, w in (("이미지 형식", self.cb_fmt), ("JPG 품질", self.sp_q), ("작업 스레드", self.sp_w)):
            g2l.addWidget(QLabel(lab))
            g2l.addWidget(w)
            g2l.addSpacing(12)
        g2l.addStretch(1)
        root.addWidget(g2)

        self.lb_sum = QLabel()
        self.lb_sum.setObjectName("summary")
        self.lb_sum.setWordWrap(True)
        self.lb_sum.setTextFormat(Qt.RichText)
        root.addWidget(self.lb_sum)

        bl = QHBoxLayout()
        self.btn_run = QPushButton("▶  실행")
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self.win.start_run)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.win.cancel_run)
        self.btn_open = QPushButton("출력 폴더 열기")
        self.btn_open.clicked.connect(self.win.open_output)
        self.btn_save = QPushButton("설정 저장…")
        self.btn_save.clicked.connect(self.win.save_config)
        bl.addWidget(self.btn_run, 2)
        bl.addWidget(self.btn_cancel)
        bl.addWidget(self.btn_open)
        bl.addWidget(self.btn_save)
        root.addLayout(bl)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m  (%p%)")
        root.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        root.addWidget(self.log, 1)

    def _estimate(self):
        cfg = self.win.cfg
        info = self.win.info
        if not info:
            return
        sp = cfg.split
        parts = split_samples(info.samples, sp["train"], sp["val"], sp["test"], int(sp.get("seed", 42)),
                              bool(sp.get("stratify", True)), bool(sp.get("keep_existing", False)))
        m = self.sp_mult.value()
        orig = self.chk_orig.isChecked()
        has_aug = bool(cfg.augment) and m > 0
        rows = []
        total = 0
        for s in ("train", "val", "test"):
            n = len(parts[s])
            if self.chk_split[s].isChecked() and has_aug:
                k = n * (m + (1 if orig else 0))
                rows.append(f"{s}: {n} → <b>{k}</b>장 (원본{'+' if orig else ' 제외, '}증강 ×{m})")
            else:
                k = n
                rows.append(f"{s}: {n} → <b>{k}</b>장")
            total += k
        pre = ", ".join(cfg.preprocess) or "없음"
        aug = ", ".join(cfg.augment) or "없음"
        warn = ""
        if "tile" in cfg.preprocess:
            warn += "<br>ℹ 타일 분할을 사용하므로 실제 이미지 수는 타일 개수만큼 늘어납니다 (빈 타일 조정 전)."
        if not cfg.augment:
            warn += "<br>⚠ 선택된 증강이 없어 분할·전처리만 수행합니다."
        self.lb_sum.setText(
            f"<b>대상</b>: {cfg.task} · {cfg.yolo_version} · 클래스 {len(cfg.class_names)}개<br>"
            f"<b>전처리</b>: {pre}<br><b>증강</b>: {aug}<br>"
            f"<b>예상 출력</b>: {' · '.join(rows)} &nbsp;= 총 {total}장{warn}<br>"
            f"<b>저장 위치</b>: {cfg.output_dir}")

    def enter(self):
        self._estimate()

    def apply_to_cfg(self):
        cfg = self.win.cfg
        cfg.multiplier = self.sp_mult.value()
        cfg.include_original = self.chk_orig.isChecked()
        cfg.augment_splits = [s for s, c in self.chk_split.items() if c.isChecked()]
        cfg.image_format = self.cb_fmt.currentData()
        cfg.jpeg_quality = self.sp_q.value()
        cfg.workers = self.sp_w.value()

    def load_cfg(self, cfg):
        self.sp_mult.setValue(int(cfg.multiplier))
        self.chk_orig.setChecked(bool(cfg.include_original))
        for s, c in self.chk_split.items():
            c.setChecked(s in cfg.augment_splits)
        i = self.cb_fmt.findData(cfg.image_format)
        self.cb_fmt.setCurrentIndex(max(i, 0))
        self.sp_q.setValue(int(cfg.jpeg_quality))
        self.sp_w.setValue(int(cfg.workers))

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
