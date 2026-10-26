"""'생성형 AI' 탭: 모델·GPU / 이미지 변형 / 객체 합성 / 검수."""
from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..dataset import analyze_dataset, scan_dataset
from ..genai import models as M
from ..genai.prompts import DOMAIN_PREFIX, EFFECTS, EFFECTS_BY_KEY, NEGATIVE
from ..genai.quality import QualityConfig
from ..io_utils import imread
from ..genai import annstore
from ..labels import parse_labels
from ..visualize import draw
from .widgets import CaptionedImage, FuncThread, ImageView


# ============================================================================ helpers
def _dir_row(placeholder: str, title: str, parent) -> tuple[QWidget, QLineEdit]:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    ed = QLineEdit()
    ed.setPlaceholderText(placeholder)
    b = QPushButton("찾기")
    b.setFixedWidth(52)
    b.clicked.connect(lambda: _pick_dir(parent, ed, title))
    lay.addWidget(ed, 1)
    lay.addWidget(b)
    return w, ed


def _pick_dir(parent, ed: QLineEdit, title: str):
    d = QFileDialog.getExistingDirectory(parent, title, ed.text() or str(Path.home()))
    if d:
        ed.setText(d)


def _dspin(v, lo, hi, step, dec=2):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(dec)
    s.setValue(v)
    return s


def _ispin(v, lo, hi, step=1):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(v)
    return s


def _scroll(inner):
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sc.setWidget(inner)
    sc.setFrameShape(QScrollArea.NoFrame)
    return sc


def _np_to_qimage(bgr):
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class JobThread(QThread):
    progress = Signal(int, int, str)
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.cancel = threading.Event()

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda d, t, m: self.progress.emit(d, t, m), log=self.log.emit,
                          cancel=self.cancel)
            self.done.emit(res)
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class RunBar(QWidget):
    """실행/취소 버튼 + 진행률 + 로그."""

    def __init__(self, run_text="▶  실행"):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.btn_run = QPushButton(run_text)
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumHeight(34)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_open = QPushButton("출력 폴더 열기")
        row.addWidget(self.btn_run, 2)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_open)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setFormat("%v / %m  (%p%)")
        lay.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setMaximumHeight(150)
        lay.addWidget(self.log)

    def running(self, on):
        self.btn_run.setEnabled(not on)
        self.btn_cancel.setEnabled(on)


def open_folder(path):
    import subprocess
    import sys

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    if not path or not Path(path).exists():
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class DatasetBox(QGroupBox):
    """입력 데이터셋 + 출력 폴더 + 분석(태스크/클래스)."""
    analyzed = Signal(object)

    def __init__(self, win, out_suffix: str):
        super().__init__("입력 · 출력")
        self.win, self.out_suffix = win, out_suffix
        self.info = None
        f = QFormLayout(self)
        w1, self.ed_ds = _dir_row("YOLO 데이터셋 폴더 (이미지+라벨)", "데이터셋 선택", self)
        row = QHBoxLayout()
        row.addWidget(w1, 1)
        b_from = QPushButton("증강 탭 데이터셋")
        b_from.setToolTip("① 데이터셋 증강 탭에서 분석한 데이터셋을 그대로 사용")
        b_from.clicked.connect(self._from_aug)
        self.b_an = QPushButton("분석")
        self.b_an.clicked.connect(self.analyze)
        row.addWidget(b_from)
        row.addWidget(self.b_an)
        f.addRow("데이터셋", row)
        w2, self.ed_out = _dir_row("결과 저장 폴더", "출력 폴더", self)
        f.addRow("출력 폴더", w2)
        self.lb = QLabel("데이터셋을 선택하고 [분석]을 누르세요.")
        self.lb.setObjectName("note")
        self.lb.setWordWrap(True)
        f.addRow(self.lb)
        self._th = None

    def _from_aug(self):
        info = getattr(self.win, "info", None)
        if info is None:
            QMessageBox.information(self, "증강 탭", "증강 탭에서 먼저 데이터셋을 분석하세요.")
            return
        self.ed_ds.setText(str(info.root))
        self._set(info)

    def analyze(self):
        p = self.ed_ds.text().strip()
        if not p or not Path(p).is_dir():
            QMessageBox.warning(self, "경로", "데이터셋 폴더를 선택하세요.")
            return
        self.b_an.setEnabled(False)
        self.lb.setText("분석 중…")
        self._th = FuncThread(lambda: analyze_dataset(scan_dataset(p)))
        self._th.done.connect(self._set)
        self._th.failed.connect(lambda m: (self.b_an.setEnabled(True), self.lb.setText("분석 실패: " + m[:200])))
        self._th.start()

    def _set(self, info):
        self.b_an.setEnabled(True)
        self.info = info
        self.lb.setText(f"{info.n_images}장 · 태스크 {info.task} · 클래스 {len(info.class_names)}개 "
                        f"({', '.join(info.class_names[:8])}{'…' if len(info.class_names) > 8 else ''})")
        if not self.ed_out.text().strip():
            self.ed_out.setText(str(info.root.parent / f"{info.root.name}_{self.out_suffix}"))
        self.analyzed.emit(info)

    def samples(self):
        if not self.info:
            return []
        lab = [s for s in self.info.samples if s.label is not None]
        return lab or self.info.samples


class ProjectBox(QGroupBox):
    """현재 프로젝트(원본 이미지 + 라벨)를 입력으로 사용. DatasetBox와 같은 인터페이스."""
    analyzed = Signal(object)

    def __init__(self, tab, kind: str):
        super().__init__("입력 (현재 프로젝트)")
        self.tab, self.kind = tab, kind
        self.info = None
        lay = QHBoxLayout(self)
        self.lb = QLabel("프로젝트를 먼저 만들거나 여세요 (1. 프로젝트 · 원본).")
        self.lb.setWordWrap(True)
        self.lb.setObjectName("note")
        b = QPushButton("새로고침")
        b.clicked.connect(self.refresh)
        lay.addWidget(self.lb, 1)
        lay.addWidget(b)
        self.ed_ds = QLineEdit()
        self.ed_out = QLineEdit()
        tab.project_changed.connect(self.refresh)

    def refresh(self):
        from collections import Counter

        from ..dataset import DatasetInfo
        from ..genai import annstore

        p = self.tab.project
        if p is None:
            self.info = None
            self.lb.setText("프로젝트를 먼저 만들거나 여세요 (1. 프로젝트 · 원본).")
            return
        info = DatasetInfo(root=p.work, samples=p.samples())
        info.class_names = list(p.class_names)
        info.task = "segment"  # 프로젝트 라벨은 폴리곤(+키포인트) JSON
        cnt, n_lab, n_kp, img_cls = Counter(), 0, 0, Counter()
        for smp in info.samples:
            ia = annstore.load(smp.label) if smp.label else None
            if not ia or not (ia.objects or ia.image_class):
                continue
            n_lab += 1
            for a in ia.objects:
                cnt[a.cls] += 1
                n_kp += a.kpts is not None and bool((a.kpts[:, 2] > 0).any())
            if ia.image_class:
                img_cls[ia.image_class] += 1
        info.class_counts = cnt
        info.n_labeled = n_lab
        self.info = info
        self.ed_ds.setText(str(p.work))
        extra = []
        if n_kp:
            extra.append(f"키포인트 객체 {n_kp}")
        if img_cls:
            extra.append("분류 " + ", ".join(f"{k} {v}" for k, v in img_cls.most_common(4)))
        self.lb.setText(f"<b>{p.root.name}</b> · 원본 {info.n_images}장 (라벨 {n_lab}장) · 클래스 "
                        f"{', '.join(info.class_names) or '-'}" + (" · " + " · ".join(extra) if extra else "") +
                        f"<br>결과는 실행마다 generated/{self.kind}_날짜시각 폴더에 저장 (라벨: 폴리곤·키포인트·분류 유지)")
        self.analyzed.emit(info)

    def prepare_run(self):
        if self.tab.project is not None:
            self.ed_out.setText(str(self.tab.project.new_run_dir(self.kind)))

    def samples(self):
        from ..genai import annstore

        if not self.info:
            return []
        return [smp for smp in self.info.samples if smp.label is not None and annstore.has_content(smp.label)]

    def project_dir(self) -> str:
        return str(self.tab.project.root) if self.tab.project is not None else ""

    def skeleton(self):
        return self.tab.project.kpt_skeleton if self.tab.project is not None else None


class ClassList(QListWidget):
    def __init__(self):
        super().__init__()
        self.setMaximumHeight(120)

    def set_names(self, names, checked_all=False, counts=None):
        self.clear()
        for i, n in enumerate(names):
            c = f" ({counts.get(i, 0)})" if counts is not None else ""
            it = QListWidgetItem(f"{i}: {n}{c}")
            it.setData(Qt.UserRole, i)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if checked_all else Qt.Unchecked)
            self.addItem(it)

    def checked(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked]

    def set_checked(self, ids):
        for i in range(self.count()):
            it = self.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ids else Qt.Unchecked)


# ============================================================================ 1. 모델 · GPU
class ModelsPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        g = QGroupBox("실행 환경")
        gl = QGridLayout(g)
        self.lb_gpu = QLabel("GPU 정보 확인 중…")
        self.lb_gpu.setWordWrap(True)
        self.lb_gpu.setTextFormat(Qt.RichText)
        b = QPushButton("다시 확인")
        b.clicked.connect(self.refresh_gpu)
        gl.addWidget(self.lb_gpu, 0, 0, 1, 3)
        gl.addWidget(b, 0, 3, Qt.AlignTop)
        st = M.load_settings()
        w, self.ed_dir = _dir_row("모델 폴더", "모델 폴더 선택", self)
        self.ed_dir.setText(st["models_dir"])
        self.cb_vram = QComboBox()
        for k, t in (("auto", "자동 (12GB 미만 → 저VRAM)"), ("low", "저VRAM: 모델 CPU 오프로드 (8GB 권장)"),
                     ("sequential", "초저VRAM: 순차 오프로드 (6GB, 매우 느림)"), ("high", "고VRAM: 전부 GPU (12GB+)")):
            self.cb_vram.addItem(t, k)
        self.cb_vram.setCurrentIndex(max(0, self.cb_vram.findData(st.get("vram_mode", "auto"))))
        bs = QPushButton("저장")
        bs.clicked.connect(self.save)
        gl.addWidget(QLabel("모델 폴더"), 1, 0)
        gl.addWidget(w, 1, 1, 1, 2)
        gl.addWidget(bs, 1, 3)
        gl.addWidget(QLabel("VRAM 모드"), 2, 0)
        gl.addWidget(self.cb_vram, 2, 1, 1, 2)
        gl.setColumnStretch(1, 1)
        root.addWidget(g)

        g2 = QGroupBox("Models (MIT / Apache-2.0; AGPL entries are optional and for internal use only)")
        g2l = QVBoxLayout(g2)
        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["선택", "상태", "모델", "용도", "라이선스", "크기"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        g2l.addWidget(self.tbl)
        row = QHBoxLayout()
        b1 = QPushButton("선택 모델 다운로드 (인터넷 PC)")
        b1.setObjectName("primary")
        b1.clicked.connect(self.download)
        b2 = QPushButton("상태 새로고침")
        b2.clicked.connect(self.refresh_models)
        b3 = QPushButton("모델 폴더 열기")
        b3.clicked.connect(lambda: open_folder(self.ed_dir.text()))
        b4 = QPushButton("폐쇄망 오프라인 번들 만들기…")
        b4.clicked.connect(self.bundle)
        for x in (b1, b2, b3, b4):
            row.addWidget(x)
        row.addStretch(1)
        g2l.addLayout(row)
        root.addWidget(g2, 1)
        note = QLabel(
            "Required (about 2.3 GB): OWLv2 + SAM + ViTPose + CLIP for auto labeling, RF-DETR Nano / Seg Nano as "
            "base weights for project model training. Add Small/Medium for higher accuracy.<br>"
            "Ultralytics YOLO26 weights are optional and AGPL-3.0: internal labeling only, do not ship them.<br>"
            "Manual labeling, copy-paste and review work without any model.")
        note.setWordWrap(True)
        note.setObjectName("banner")
        root.addWidget(note)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setMaximumHeight(120)
        root.addWidget(self.log)
        self._th = None
        self.refresh_models()
        self.refresh_gpu()

    def save(self):
        st = M.load_settings()
        st["models_dir"] = self.ed_dir.text().strip() or st["models_dir"]
        st["vram_mode"] = self.cb_vram.currentData()
        M.save_settings(st)
        self.refresh_models()
        self.log.appendPlainText("설정 저장됨.")

    def models_dir(self):
        return self.ed_dir.text().strip()

    def vram_mode(self):
        return self.cb_vram.currentData()

    def refresh_gpu(self):
        self._gth = FuncThread(lambda: (M.gpu_info(), M.assist_status()))
        self._gth.done.connect(lambda r: self._show_gpu(*r))
        self._gth.start()

    def _show_gpu(self, i, has_diffusers=False):
        self.gpu = i
        if not i["torch"]:
            txt = f"<span style='color:#b45309'>⚠ {i['note']}</span>"
        elif i["cuda"]:
            mode = M.resolve_vram_mode(self.vram_mode(), i["vram_gb"])
            txt = (f"<b>{i['name']}</b> · VRAM {i['vram_gb']} GB · {i['capability']} · PyTorch {i['torch_version']} "
                   f"(CUDA {i['cuda_version']}) → 실행 모드 <b>{mode}</b>")
            if i["note"]:
                txt += f"<br><span style='color:#c0392b'>⚠ {i['note']}</span>"
        else:
            txt = f"PyTorch {i['torch_version']} · <span style='color:#b45309'>⚠ {i['note']}</span>"
        self.lb_gpu.setText(txt + "<br>" + (has_diffusers or ""))

    def refresh_models(self):
        root = Path(self.models_dir()) if self.models_dir() else None
        rows = M.status_table(root)
        self.tbl.setRowCount(len(rows))
        for r, m in enumerate(rows):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if (m["phase"] == 1 and not m["installed"] and m["required"]) else Qt.Unchecked)
            chk.setData(Qt.UserRole, m["key"])
            self.tbl.setItem(r, 0, chk)
            st = QTableWidgetItem("✓ 설치됨" if m["installed"] else ("필수·미설치" if m["required"] else "미설치"))
            st.setForeground(QColor("#059669" if m["installed"] else ("#c0392b" if m["required"] else "#6b7280")))
            self.tbl.setItem(r, 1, st)
            self.tbl.setItem(r, 2, QTableWidgetItem(m["name"] + ("  [2차]" if m["phase"] == 2 else "")))
            self.tbl.setItem(r, 3, QTableWidgetItem(m["purpose"]))
            self.tbl.setItem(r, 4, QTableWidgetItem(m["license"]))
            self.tbl.setItem(r, 5, QTableWidgetItem(f"{m['size_gb']} GB"))
        self.tbl.resizeColumnsToContents()
        self.tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

    def selected(self):
        return [self.tbl.item(r, 0).data(Qt.UserRole) for r in range(self.tbl.rowCount())
                if self.tbl.item(r, 0).checkState() == Qt.Checked]

    def download(self):
        keys = self.selected()
        if not keys:
            QMessageBox.information(self, "다운로드", "다운로드할 모델을 선택하세요.")
            return
        root = Path(self.models_dir())

        def job(progress=None, log=print, cancel=None):
            for i, k in enumerate(keys):
                if cancel.is_set():
                    break
                progress(i, len(keys), k)
                M.download_model(k, root, log)
            progress(len(keys), len(keys), "")
            return True

        self._th = JobThread(job)
        self._th.log.connect(self.log.appendPlainText)
        self._th.done.connect(lambda _: (self.refresh_models(), self.log.appendPlainText("다운로드 완료")))
        self._th.failed.connect(lambda m: (self.log.appendPlainText(m),
                                           QMessageBox.warning(self, "다운로드 실패", m.splitlines()[0] +
                                                               "\n\n인터넷 연결(HuggingFace 접속)을 확인하세요.")))
        self._th.start()

    def bundle(self):
        dlg = BundleDialog(self, self.selected())
        dlg.exec()


class BundleDialog(QDialog):
    def __init__(self, parent, preselect):
        super().__init__(parent)
        self.setWindowTitle("폐쇄망 오프라인 설치 번들 만들기")
        self.resize(720, 560)
        lay = QVBoxLayout(self)
        info = QLabel("인터넷이 되는 PC에서 실행합니다. 대상 PC(OS/Python)에 맞는 패키지(wheel), PyTorch(CUDA), 모델, 앱 소스를 "
                      "한 폴더에 모읍니다. 이 폴더를 USB 등으로 옮긴 뒤 install_offline 스크립트를 실행하면 됩니다.")
        info.setWordWrap(True)
        lay.addWidget(info)
        f = QFormLayout()
        self.cb_os = QComboBox()
        self.cb_os.addItem("Windows 10/11 (x64)", "win")
        self.cb_os.addItem("Ubuntu (x86_64)", "linux")
        self.cb_py = QComboBox()
        self.cb_py.addItems(["3.11", "3.10", "3.12"])
        self.cb_cuda = QComboBox()
        self.cb_cuda.addItem("CUDA 12.8 (RTX 30/40/50 공용, 드라이버 570+)", "cu128")
        self.cb_cuda.addItem("CUDA 12.6 (RTX 50 미지원)", "cu126")
        self.cb_cuda.addItem("CPU 전용", "cpu")
        self.cb_cuda.addItem("PyTorch 제외", "none")
        self.chk_models = QListWidget()
        for m in M.MODELS:
            it = QListWidgetItem(f"{m.name}  ({m.size_gb} GB, {m.license})")
            it.setData(Qt.UserRole, m.key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if (m.key in preselect or m.required) else Qt.Unchecked)
            self.chk_models.addItem(it)
        self.chk_models.setMaximumHeight(150)
        self.cb_ubuntu = QComboBox()
        self.cb_ubuntu.addItems(["22.04", "24.04", "20.04"])
        self.chk_pyinst = QCheckBox("Windows용 Python 설치 파일 포함")
        self.chk_pyinst.setChecked(True)
        w, self.ed_out = _dir_row("번들 저장 폴더 (20~30GB 여유 필요)", "번들 폴더", self)
        f.addRow("대상 OS", self.cb_os)
        f.addRow("대상 Python", self.cb_py)
        f.addRow("Ubuntu 버전 (Linux 대상)", self.cb_ubuntu)
        f.addRow("PyTorch", self.cb_cuda)
        f.addRow("모델", self.chk_models)
        f.addRow("", self.chk_pyinst)
        f.addRow("저장 폴더", w)
        lay.addLayout(f)
        self.run = RunBar("번들 생성")
        self.run.btn_open.clicked.connect(lambda: open_folder(self.ed_out.text()))
        self.run.btn_run.clicked.connect(self.start)
        self.run.btn_cancel.clicked.connect(lambda: self._th and self._th.cancel.set())
        lay.addWidget(self.run, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._th = None

    def start(self):
        from ..genai.offline_bundle import BundleConfig, build_bundle

        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "번들", "저장 폴더를 지정하세요.")
            return
        keys = [self.chk_models.item(i).data(Qt.UserRole) for i in range(self.chk_models.count())
                if self.chk_models.item(i).checkState() == Qt.Checked]
        cfg = BundleConfig(output_dir=out, target_os=self.cb_os.currentData(), python=self.cb_py.currentText(),
                           torch_variant=self.cb_cuda.currentData(), models=keys,
                           include_python_installer=self.chk_pyinst.isChecked(),
                           ubuntu=self.cb_ubuntu.currentText())
        self.run.running(True)
        self._th = JobThread(build_bundle, cfg)
        self._th.progress.connect(lambda d, t, m: (self.run.bar.setMaximum(t), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(lambda r: (self.run.running(False), self.run.log.appendPlainText(f"완료: {r}")))
        self._th.failed.connect(lambda m: (self.run.running(False), self.run.log.appendPlainText(m)))
        self._th.start()


# ============================================================================ 2. 이미지 변형
class PromptDialog(QDialog):
    def __init__(self, parent, custom: dict, negative: str, domain: str):
        super().__init__(parent)
        self.setWindowTitle("프롬프트 편집 (생성형 SDXL)")
        self.resize(820, 620)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"앞에 자동으로 붙는 도메인 문구: <i>{DOMAIN_PREFIX.get(domain, '')}</i>"))
        inner = QWidget()
        f = QFormLayout(inner)
        self.eds = {}
        for e in EFFECTS:
            ed = QPlainTextEdit(custom.get(e.key) or e.prompt)
            ed.setMaximumHeight(62)
            self.eds[e.key] = ed
            f.addRow(f"{e.group} · {e.name}", ed)
        self.neg = QPlainTextEdit(negative)
        self.neg.setMaximumHeight(62)
        f.addRow("네거티브", self.neg)
        lay.addWidget(_scroll(inner), 1)
        row = QHBoxLayout()
        br = QPushButton("기본값으로")
        br.clicked.connect(self._reset)
        row.addWidget(br)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        lay.addLayout(row)

    def _reset(self):
        for k, ed in self.eds.items():
            ed.setPlainText(EFFECTS_BY_KEY[k].prompt)
        self.neg.setPlainText(NEGATIVE)

    def values(self):
        custom = {k: ed.toPlainText().strip() for k, ed in self.eds.items()
                  if ed.toPlainText().strip() != EFFECTS_BY_KEY[k].prompt}
        return custom, self.neg.toPlainText().strip()


class UserEffectDialog(QDialog):
    """사용자 효과: 이름 · 적용 방식 · 프롬프트 · 네거티브 · 강도 · 대상 클래스 · 변형 후 클래스명."""

    def __init__(self, parent, d: dict | None, class_names: list[str]):
        from ..genai.prompts import SCOPE_LABELS
        super().__init__(parent)
        self.setWindowTitle("사용자 효과 " + ("편집" if d else "추가"))
        self.resize(720, 640)
        self.d = dict(d or {})
        f = QFormLayout(self)
        self.ed_name = QLineEdit(self.d.get("name", ""))
        self.ed_name.setPlaceholderText("예) 가드레일 파손")
        self.cb_scope = QComboBox()
        for k, t in SCOPE_LABELS.items():
            self.cb_scope.addItem(t, k)
        self.cb_scope.setCurrentIndex(max(0, self.cb_scope.findData(self.d.get("scope", "region"))))
        self.ed_prompt = QPlainTextEdit(self.d.get("prompt", ""))
        self.ed_prompt.setPlaceholderText("영어 권장. 예) bent and broken metal guardrail, crash damage, scratched paint, "
                                          "photorealistic\n(기울어짐/쓰러짐은 객체가 빠진 자리 배경 복원용 문구 — 예: empty "
                                          "asphalt road surface)")
        self.ed_prompt.setMinimumHeight(90)
        self.ed_neg = QPlainTextEdit(self.d.get("negative", ""))
        self.ed_neg.setPlaceholderText("비우면 공통 네거티브 사용 (cartoon, text, watermark, blurry …)")
        self.ed_neg.setMaximumHeight(60)
        self.sp_str = _dspin(float(self.d.get("strength", 0.7)), 0.05, 1.0, 0.05)
        self.sp_cn = _dspin(float(self.d.get("control_scale", 0.45)), 0.0, 2.0, 0.05)
        self.targets = ClassList()
        self.targets.set_names(class_names)
        self.targets.set_checked([i for i, n in enumerate(class_names) if n in self.d.get("targets", [])])
        self.ed_remap = QLineEdit(self.d.get("remap", ""))
        self.ed_remap.setPlaceholderText("예) guardrail_damaged — 새 이름이면 클래스가 추가됩니다 (비우면 유지)")
        self.ed_img_remap = QLineEdit(self.d.get("image_remap", ""))
        self.ed_img_remap.setPlaceholderText("선택) 변형된 이미지의 '이미지 분류' 라벨. 예) damaged (Classification용)")
        self.ed_clip = QLineEdit(self.d.get("clip_text", ""))
        self.ed_clip.setPlaceholderText("선택) 품질 필터 CLIP 검사 문장. 예) a photo of a damaged guardrail")
        self.names = class_names
        f.addRow("이름", self.ed_name)
        f.addRow("적용 방식", self.cb_scope)
        f.addRow("프롬프트", self.ed_prompt)
        f.addRow("네거티브", self.ed_neg)
        f.addRow("변형 강도 (denoise)", self.sp_str)
        f.addRow("윤곽 유지 (전체 변형 시)", self.sp_cn)
        f.addRow("대상 클래스", self.targets)
        f.addRow("변형 후 클래스명", self.ed_remap)
        f.addRow("변형 후 이미지 분류", self.ed_img_remap)
        f.addRow("CLIP 검사 문장", self.ed_clip)
        note = QLabel("• 전체 변형: 윤곽선을 유지한 채 이미지 전체를 프롬프트대로 바꿈 (라벨 유지)<br>"
                      "• 대상 객체 영역: 선택한 클래스 객체 주변만 잘라 인페인팅 (라벨 위치 유지, 클래스명 변경 가능)<br>"
                      "• 기울어짐/쓰러짐: 객체를 기하 변형하고 라벨을 다시 계산 (프롬프트는 빈 자리 배경 복원에 사용)")
        note.setWordWrap(True)
        note.setObjectName("note")
        f.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _ok(self):
        if not self.ed_name.text().strip():
            QMessageBox.warning(self, "입력", "이름을 입력하세요.")
            return
        if self.cb_scope.currentData() in ("global", "region") and not self.ed_prompt.toPlainText().strip():
            QMessageBox.warning(self, "입력", "프롬프트를 입력하세요.")
            return
        if self.cb_scope.currentData() != "global" and not self.targets.checked():
            QMessageBox.warning(self, "입력", "대상 클래스를 하나 이상 선택하세요.")
            return
        self.accept()

    def value(self) -> dict:
        from ..genai.prompts import new_effect_key
        name = self.ed_name.text().strip()
        return dict(key=self.d.get("key") or new_effect_key(name), name=name, scope=self.cb_scope.currentData(),
                    prompt=self.ed_prompt.toPlainText().strip(), negative=self.ed_neg.toPlainText().strip(),
                    strength=self.sp_str.value(), control_scale=self.sp_cn.value(),
                    targets=[self.names[i] for i in self.targets.checked()], remap=self.ed_remap.text().strip(),
                    image_remap=self.ed_img_remap.text().strip(), clip_text=self.ed_clip.text().strip())


class TransformPage(QWidget):
    def __init__(self, win, tab):
        super().__init__()
        self.win, self.tab = win, tab
        self.custom_prompts: dict = {}
        self.negative = NEGATIVE
        self.sample_idx = 0
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        inner = QWidget()
        L = QVBoxLayout(inner)
        self.ds = ProjectBox(tab, "transform")
        self.ds.analyzed.connect(self._on_info)
        L.addWidget(self.ds)

        g = QGroupBox("방식 · 효과")
        gl = QGridLayout(g)
        self.cb_backend = QComboBox()
        self.cb_backend.addItem("생성형 SDXL (GPU, 사실적)", "diffusion")
        self.cb_backend.addItem("절차적 (GPU 불필요, 빠름, 라벨 100% 보존)", "procedural")
        self.cb_backend.currentIndexChanged.connect(self._backend_changed)
        self.cb_domain = QComboBox()
        self.cb_domain.addItem("드론 항공영상", "aerial")
        self.cb_domain.addItem("지상 촬영", "ground")
        self.cb_domain.addItem("도메인 문구 없음", "none")
        gl.addWidget(QLabel("방식"), 0, 0)
        gl.addWidget(self.cb_backend, 0, 1, 1, 2)
        gl.addWidget(QLabel("도메인"), 1, 0)
        gl.addWidget(self.cb_domain, 1, 1)
        bp = QPushButton("프롬프트 편집…")
        bp.clicked.connect(self._edit_prompts)
        gl.addWidget(bp, 1, 2)
        self.eff_checks: dict[str, QCheckBox] = {}
        r = 2
        groups: dict[str, list] = {}
        for e in EFFECTS:
            groups.setdefault(e.group, []).append(e)
        for gname, lst in groups.items():
            gl.addWidget(QLabel(f"<b>{gname}</b>"), r, 0)
            box = QHBoxLayout()
            for e in lst:
                cb = QCheckBox(e.name)
                cb.setToolTip(e.desc)
                cb.toggled.connect(self._estimate)
                self.eff_checks[e.key] = cb
                box.addWidget(cb)
            box.addStretch(1)
            gl.addLayout(box, r, 1, 1, 2)
            r += 1
        for k in ("rain", "night"):
            self.eff_checks[k].setChecked(True)
        L.addWidget(g)

        gu = QGroupBox("사용자 효과 (직접 입력한 프롬프트 · SDXL)")
        gul = QVBoxLayout(gu)
        self.user_list = QListWidget()
        self.user_list.setMaximumHeight(130)
        self.user_list.itemChanged.connect(lambda *_: self._estimate())
        self.user_list.itemDoubleClicked.connect(lambda it: self._edit_user())
        gul.addWidget(self.user_list)
        ur = QHBoxLayout()
        for text, fn in (("추가…", self._add_user), ("편집…", self._edit_user), ("삭제", self._del_user),
                         ("라이브러리에서 가져오기…", self._import_user)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            ur.addWidget(b)
        ur.addStretch(1)
        gul.addLayout(ur)
        hint = QLabel("예) 이름 '가드레일 파손' · 적용 방식 '대상 객체 영역' · 대상 guardrail · 프롬프트 'bent and broken metal "
                      "guardrail, crash damage' · 변형 후 클래스 guardrail_damaged")
        hint.setWordWrap(True)
        hint.setObjectName("note")
        gul.addWidget(hint)
        L.addWidget(gu)

        g3 = QGroupBox("객체 파손 · 구조물 변형 대상 (선택한 클래스만 변형)")
        g3l = QFormLayout(g3)
        self.cls_list = ClassList()
        g3l.addRow("대상 클래스", self.cls_list)
        self.ed_remap = QLineEdit()
        self.ed_remap.setPlaceholderText("예) damaged_building, traffic_light_fallen  (비우면 클래스 유지)")
        g3l.addRow("파손 후 클래스명", self.ed_remap)
        self.ed_img_remap = QLineEdit()
        self.ed_img_remap.setPlaceholderText("선택) 파손이 적용된 이미지의 분류 라벨. 예) damaged  (비우면 유지)")
        g3l.addRow("파손 후 이미지 분류", self.ed_img_remap)
        self.sp_regions = _ispin(3, 1, 20)
        g3l.addRow("이미지당 최대 대상 수", self.sp_regions)
        self.chk_whole = QCheckBox("대상 객체가 없으면 이미지 전체에 적용")
        g3l.addRow(self.chk_whole)
        L.addWidget(g3)

        g4 = QGroupBox("생성 파라미터")
        g4l = QFormLayout(g4)
        self.chk_def_strength = QCheckBox("효과별 기본 강도 사용 (비·눈 0.45, 야간 0.55, 파손 0.7~0.75)")
        self.chk_def_strength.setChecked(True)
        self.sp_strength = _dspin(0.45, 0.05, 1.0, 0.05)
        self.sp_strength.setEnabled(False)
        self.chk_def_strength.toggled.connect(lambda c: self.sp_strength.setEnabled(not c))
        g4l.addRow(self.chk_def_strength)
        g4l.addRow("변형 강도 (denoise)", self.sp_strength)
        self.chk_def_cn = QCheckBox("효과별 기본 윤곽 유지 강도 사용")
        self.chk_def_cn.setChecked(True)
        self.sp_cn = _dspin(0.65, 0.0, 2.0, 0.05)
        self.sp_cn.setEnabled(False)
        self.chk_def_cn.toggled.connect(lambda c: self.sp_cn.setEnabled(not c))
        g4l.addRow(self.chk_def_cn)
        g4l.addRow("윤곽 유지 (ControlNet)", self.sp_cn)
        self.sp_steps = _ispin(25, 4, 80)
        self.sp_guid = _dspin(6.0, 1.0, 15.0, 0.5, 1)
        self.sp_long = _ispin(1024, 512, 1536, 64)
        self.sp_detail = _dspin(0.35, 0.0, 1.0, 0.05)
        self.sp_detail.setToolTip("고해상도 원본을 1024px로 줄여 생성 후 되돌릴 때, 원본의 미세 디테일(작은 객체)을 섞는 비율")
        self.sp_blend = _dspin(0.0, 0.0, 1.0, 0.05)
        self.sp_blend.setToolTip("라벨 영역에 원본 픽셀을 섞는 비율 (객체 소실 방지). 0=섞지 않음")
        self.sp_int = _dspin(0.6, 0.1, 1.0, 0.05)
        g4l.addRow("스텝 수", self.sp_steps)
        g4l.addRow("CFG (프롬프트 충실도)", self.sp_guid)
        g4l.addRow("생성 해상도 (긴 변)", self.sp_long)
        g4l.addRow("원본 디테일 보존", self.sp_detail)
        g4l.addRow("라벨 영역 원본 보존", self.sp_blend)
        g4l.addRow("절차적 효과 강도", self.sp_int)
        self.sp_var = _ispin(1, 1, 20)
        self.sp_max = _ispin(0, 0, 1000000)
        self.sp_max.setSpecialValueText("전체")
        self.sp_max.valueChanged.connect(self._estimate)
        self.sp_var.valueChanged.connect(self._estimate)
        self.sp_seed = _ispin(1234, 0, 2**31 - 1)
        g4l.addRow("효과당 생성 수", self.sp_var)
        g4l.addRow("최대 원본 이미지 수", self.sp_max)
        g4l.addRow("시드", self.sp_seed)
        L.addWidget(g4)

        g5 = QGroupBox("품질 자동 필터")
        g5l = QFormLayout(g5)
        q = QualityConfig()
        self.q_struct = _dspin(q.min_structure, 0.0, 1.0, 0.05)
        self.q_obj = _dspin(q.min_object, 0.0, 1.0, 0.05)
        self.q_action = QComboBox()
        self.q_action.addItem("해당 객체 라벨만 제거", "drop_label")
        self.q_action.addItem("이미지 전체 거부", "reject_image")
        self.q_action.addItem("무시", "keep")
        self.q_change = _dspin(q.min_change, 0.0, 0.5, 0.005, 3)
        self.q_clip = QCheckBox("CLIP으로 효과 적용 여부 검사 (clip_b32 모델 필요)")
        self.q_clip_gain = _dspin(0.0, -0.2, 0.2, 0.005, 3)
        self.q_save_rej = QCheckBox("거부된 결과도 _rejected 폴더에 저장 (검수 탭에서 복원 가능)")
        self.q_save_rej.setChecked(True)
        g5l.addRow("구조 유지 최소", self.q_struct)
        g5l.addRow("객체 유지 최소", self.q_obj)
        g5l.addRow("객체 소실 시", self.q_action)
        g5l.addRow("최소 변화량", self.q_change)
        g5l.addRow(self.q_clip)
        g5l.addRow("CLIP 최소 증가량", self.q_clip_gain)
        g5l.addRow(self.q_save_rej)
        L.addWidget(g5)
        L.addStretch(1)
        left = _scroll(inner)
        left.setMinimumWidth(500)
        split.addWidget(left)

        right = QWidget()
        R = QVBoxLayout(right)
        top = QHBoxLayout()
        t = QLabel("미리보기")
        t.setObjectName("sectionTitle")
        top.addWidget(t)
        self.cb_prev_eff = QComboBox()
        for e in EFFECTS:
            self.cb_prev_eff.addItem(f"{e.group} · {e.name}", e.key)
        top.addStretch(1)
        top.addWidget(self.cb_prev_eff)
        b_s = QPushButton("다른 샘플")
        b_s.clicked.connect(self._next)
        self.b_prev = QPushButton("미리보기 실행")
        self.b_prev.setObjectName("primary")
        self.b_prev.clicked.connect(self.preview)
        top.addWidget(b_s)
        top.addWidget(self.b_prev)
        R.addLayout(top)
        grid = QHBoxLayout()
        self.v_orig = CaptionedImage("원본")
        self.v_gen = CaptionedImage("결과")
        grid.addWidget(self.v_orig)
        grid.addWidget(self.v_gen)
        R.addLayout(grid, 1)
        self.lb_q = QLabel("")
        self.lb_q.setWordWrap(True)
        self.lb_q.setObjectName("note")
        R.addWidget(self.lb_q)
        self.lb_est = QLabel("")
        self.lb_est.setObjectName("summary")
        self.lb_est.setWordWrap(True)
        R.addWidget(self.lb_est)
        self.run = RunBar()
        self.run.btn_run.clicked.connect(self.start)
        self.run.btn_cancel.clicked.connect(self.cancel)
        self.run.btn_open.clicked.connect(lambda: open_folder(self.ds.ed_out.text()))
        R.addWidget(self.run)
        split.addWidget(right)
        split.setSizes([540, 700])
        root.addWidget(split)
        self._th = None
        self._pth = None
        self._backend_changed()

    # ---------------------------------------------------------------- state
    def _on_info(self, info):
        self._refresh_user()
        self.cls_list.set_names(info.class_names, counts=info.class_counts)
        cand = [i for i, n in enumerate(info.class_names)
                if any(k in n.lower() for k in ("build", "house", "road", "bridge", "건물", "도로", "교량", "structure", "light", "signal", "sign", "pole", "barrier", "median", "fence", "신호", "표지", "분리대", "가로등"))]
        self.cls_list.set_checked(cand)
        self.sample_idx = 0
        self._estimate()

    def _backend_changed(self):
        diff = self.cb_backend.currentData() == "diffusion"
        for w in (self.sp_steps, self.sp_guid, self.sp_long, self.sp_detail, self.chk_def_strength, self.chk_def_cn):
            w.setEnabled(diff)
        self.sp_strength.setEnabled(diff and not self.chk_def_strength.isChecked())
        self.sp_cn.setEnabled(diff and not self.chk_def_cn.isChecked())
        self.sp_int.setEnabled(not diff)
        self._estimate()

    def _edit_prompts(self):
        d = PromptDialog(self, self.custom_prompts, self.negative, self.cb_domain.currentData())
        if d.exec():
            self.custom_prompts, self.negative = d.values()

    def effects(self):
        out = [k for k, cb in self.eff_checks.items() if cb.isChecked()]
        if hasattr(self, "user_list"):
            out += [self.user_list.item(i).data(Qt.UserRole) for i in range(self.user_list.count())
                    if self.user_list.item(i).checkState() == Qt.Checked]
        return out

    def cfg(self):
        from ..genai.transform import TransformConfig

        info = self.ds.info
        eff = self.effects()
        customs = self.user_effects()
        strength = {} if self.chk_def_strength.isChecked() else {k: self.sp_strength.value() for k in
                                                                 list(EFFECTS_BY_KEY) + [c["key"] for c in customs]}
        return TransformConfig(
            project_dir=self.ds.project_dir(), output_dir=self.ds.ed_out.text().strip(),
            image_remap=self.ed_img_remap.text().strip(),
            flip_idx=self.tab.project.flip_idx if self.tab.project is not None and self.tab.project.kpt_names else None,
            task=info.task if info else "detect", class_names=list(info.class_names) if info else [],
            kpt_shape=list(info.kpt_shape) if info and info.kpt_shape else None,
            backend=self.cb_backend.currentData(), effects=eff, custom_prompts=dict(self.custom_prompts),
            negative_prompt=self.negative, domain=self.cb_domain.currentData(), intensity=self.sp_int.value(),
            strength=strength, control_scale=None if self.chk_def_cn.isChecked() else self.sp_cn.value(),
            steps=self.sp_steps.value(), guidance=self.sp_guid.value(), long_side=self.sp_long.value(),
            detail_keep=self.sp_detail.value(), blend_objects=self.sp_blend.value(), variants=self.sp_var.value(),
            max_images=self.sp_max.value(), seed=self.sp_seed.value(), region_classes=self.cls_list.checked(),
            region_remap=self.ed_remap.text().strip(), region_whole_if_none=self.chk_whole.isChecked(),
            max_regions=self.sp_regions.value(),
            quality=asdict(QualityConfig(self.q_struct.value(), self.q_obj.value(), self.q_action.currentData(),
                                         self.q_change.value(), self.q_clip.isChecked(), self.q_clip_gain.value())),
            save_rejected=self.q_save_rej.isChecked(), vram_mode=self.tab.models.vram_mode(),
            models_dir=self.tab.models.models_dir(), custom_effects=customs)

    # ---------------------------------------------------------------- user effects
    def user_effects(self) -> list[dict]:
        p = self.tab.project
        return list(p.custom_effects) if p is not None else []

    def _refresh_user(self):
        checked = {self.user_list.item(i).data(Qt.UserRole) for i in range(self.user_list.count())
                   if self.user_list.item(i).checkState() == Qt.Checked}
        self.user_list.blockSignals(True)
        self.user_list.clear()
        from ..genai.prompts import SCOPE_LABELS
        for d in self.user_effects():
            tg = ", ".join(d.get("targets", [])) or "공통 대상"
            it = QListWidgetItem(f"{d['name']}  —  {SCOPE_LABELS.get(d.get('scope'), '')} · 대상: {tg}"
                                 + (f" → {d['remap']}" if d.get("remap") else ""))
            it.setData(Qt.UserRole, d["key"])
            it.setToolTip(d.get("prompt", ""))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if d["key"] in checked or d.get("_new") else Qt.Unchecked)
            d.pop("_new", None)
            self.user_list.addItem(it)
        self.user_list.blockSignals(False)
        cur = self.cb_prev_eff.currentData()
        self.cb_prev_eff.blockSignals(True)
        self.cb_prev_eff.clear()
        for e in EFFECTS:
            self.cb_prev_eff.addItem(f"{e.group} · {e.name}", e.key)
        for d in self.user_effects():
            self.cb_prev_eff.addItem(f"사용자 · {d['name']}", d["key"])
        i = self.cb_prev_eff.findData(cur)
        self.cb_prev_eff.setCurrentIndex(max(0, i))
        self.cb_prev_eff.blockSignals(False)
        self._estimate()

    def _need_project(self):
        if self.tab.project is None:
            QMessageBox.information(self, "프로젝트", "먼저 1. 프로젝트 · 원본에서 프로젝트를 만들거나 여세요.")
            return False
        return True

    def _add_user(self):
        if not self._need_project():
            return
        d = UserEffectDialog(self, None, self.tab.project.class_names)
        if d.exec():
            v = d.value()
            v["_new"] = True
            self.tab.project.custom_effects = self.user_effects() + [v]
            self.tab.project.save()
            from ..genai.prompts import load_library, save_library
            lib = [x for x in load_library() if x.get("key") != v["key"]]
            save_library(lib + [{k: x for k, x in v.items() if k != "_new"}])
            self._refresh_user()

    def _cur_user_key(self):
        it = self.user_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _edit_user(self):
        key = self._cur_user_key()
        if not key or not self._need_project():
            return
        lst = self.user_effects()
        cur = next(x for x in lst if x["key"] == key)
        d = UserEffectDialog(self, cur, self.tab.project.class_names)
        if d.exec():
            v = d.value()
            self.tab.project.custom_effects = [v if x["key"] == key else x for x in lst]
            self.tab.project.save()
            from ..genai.prompts import load_library, save_library
            save_library([v if x.get("key") == key else x for x in load_library()])
            self._refresh_user()

    def _del_user(self):
        key = self._cur_user_key()
        if not key:
            return
        self.tab.project.custom_effects = [x for x in self.user_effects() if x["key"] != key]
        self.tab.project.save()
        self._refresh_user()

    def _import_user(self):
        if not self._need_project():
            return
        from ..genai.prompts import load_library
        lib = load_library()
        have = {x["key"] for x in self.user_effects()}
        cand = [x for x in lib if x.get("key") not in have]
        if not cand:
            QMessageBox.information(self, "라이브러리", "가져올 수 있는 저장된 사용자 효과가 없습니다.\n"
                                    "(효과를 추가하면 다른 프로젝트에서도 쓸 수 있게 자동 저장됩니다)")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("사용자 효과 라이브러리")
        lay = QVBoxLayout(dlg)
        lw = QListWidget()
        for x in cand:
            it = QListWidgetItem(f"{x['name']} — {x.get('prompt', '')[:60]}")
            it.setData(Qt.UserRole, x["key"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            lw.addItem(it)
        lay.addWidget(lw)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec():
            keys = {lw.item(i).data(Qt.UserRole) for i in range(lw.count()) if lw.item(i).checkState() == Qt.Checked}
            add = [dict(x, _new=True) for x in cand if x["key"] in keys]
            self.tab.project.custom_effects = self.user_effects() + add
            self.tab.project.save()
            self._refresh_user()

    def _estimate(self):
        if not hasattr(self, "lb_est"):
            return
        info = self.ds.info
        if not info:
            self.lb_est.setText("데이터셋을 분석하면 예상 생성량이 표시됩니다.")
            return
        n = len(self.ds.samples())
        if self.sp_max.value():
            n = min(n, self.sp_max.value())
        eff = self.effects()
        total = n * len(eff) * self.sp_var.value()
        txt = f"원본 {n}장 × 효과 {len(eff)}개 × {self.sp_var.value()} = <b>최대 {total}장</b> 생성"
        if self.cb_backend.currentData() == "diffusion" and total:
            lo, hi = total * 15 / 3600, total * 40 / 3600
            txt += f" · 8GB GPU 기준 예상 약 {lo:.1f}~{hi:.1f}시간 (1024px, 25 step)"
        self.lb_est.setText(txt)

    def _check_ready(self, need_out=True):
        if not self.ds.info:
            QMessageBox.information(self, "데이터셋", "먼저 데이터셋을 분석하세요.")
            return False
        if not self.effects():
            QMessageBox.information(self, "효과", "적용할 효과를 선택하세요.")
            return False
        if need_out and not self.ds.ed_out.text().strip():
            QMessageBox.information(self, "출력", "출력 폴더를 지정하세요.")
            return False
        from ..genai.prompts import get_effect
        customs = [k for k in self.effects() if get_effect(k, self.user_effects()).custom
                   and get_effect(k, self.user_effects()).kind != "structural"]
        if customs and self.cb_backend.currentData() != "diffusion":
            QMessageBox.information(self, "사용자 효과", "프롬프트로 동작하는 사용자 효과는 '생성형 SDXL' 방식에서만 "
                                    "실행됩니다. 방식을 바꾸거나 해당 효과의 선택을 해제하세요.")
            return False
        if self.cb_backend.currentData() == "diffusion":
            if not M.diffusers_available():
                QMessageBox.warning(self, "생성형 기능", "PyTorch/diffusers가 설치되지 않았습니다.\n"
                                    "windows\\install_genai.bat 또는 ubuntu/install_genai.sh를 실행하거나 "
                                    "방식을 '절차적'으로 바꾸세요.")
                return False
            root = Path(self.tab.models.models_dir())
            miss = [k for k in ("sdxl_base",) if not M.is_installed(k, root)]
            if miss:
                QMessageBox.warning(self, "모델 없음", "SDXL Base 모델이 없습니다. '모델·GPU' 탭에서 다운로드하거나 "
                                    "오프라인 번들로 설치하세요.")
                return False
        return True

    # ---------------------------------------------------------------- preview
    def _next(self):
        self.sample_idx += 1
        self._show_sample_only()

    def _pick(self):
        smp = self.ds.samples()
        if not smp:
            return None
        order = np.random.default_rng(99).permutation(len(smp))
        return smp[int(order[self.sample_idx % len(smp)])]

    def _load(self, s, cfg):
        from ..genai import annstore

        img = imread(s.image)
        h, w = img.shape[:2]
        anns, self._img_cls = annstore.load_any(s.label, cfg.task, w, h,
                                                tuple(cfg.kpt_shape) if cfg.kpt_shape else None)
        return img, anns

    def _draw(self, img, anns, cfg, names=None, image_class=None):
        return draw(img, anns, cfg.task, names or cfg.class_names, self.ds.skeleton(), image_class)

    def _show_sample_only(self):
        if not self.ds.info:
            return
        cfg = self.cfg()
        s = self._pick()
        img, anns = self._load(s, cfg)
        self.v_orig.set(self._draw(img, anns, cfg, None, self._img_cls), f"원본 · {s.image.name} · 객체 {len(anns)}")
        self.v_gen.set(None, "결과 — [미리보기 실행]")
        self.lb_q.setText("")

    def preview(self):
        if not self._check_ready(need_out=False):
            return
        cfg = self.cfg()
        key = self.cb_prev_eff.currentData()
        s = self._pick()
        img, anns = self._load(s, cfg)
        img_cls = self._img_cls
        self.v_orig.set(self._draw(img, anns, cfg, None, self._img_cls), f"원본 · {s.image.name} · 객체 {len(anns)}")
        self.v_gen.set(None, "생성 중…" + (" (첫 실행은 모델 로딩으로 1~2분)" if cfg.backend == "diffusion" else ""))
        self.b_prev.setEnabled(False)

        def job():
            from ..genai.transform import get_clip, transform_one

            clip = None
            if cfg.quality.get("use_clip"):
                try:
                    clip = get_clip(cfg)
                except Exception:
                    clip = None
            names = list(cfg.class_names)
            return transform_one(img, anns, key, cfg, cfg.seed + self.sample_idx, None, clip, names,
                                 log=lambda m: None, image_class=img_cls), names

        self._pth = FuncThread(job)
        self._pth.done.connect(lambda r: self._show_result(r, cfg))
        self._pth.failed.connect(self._prev_failed)
        self._pth.start()

    def _prev_failed(self, m):
        self.b_prev.setEnabled(True)
        self.v_gen.set(None, "실패")
        self.lb_q.setText(f"<span style='color:#c0392b'>{m.splitlines()[0]}</span>")

    def _show_result(self, r, cfg):
        self.b_prev.setEnabled(True)
        res, names = r
        if res is None:
            self.v_gen.set(None, "대상 클래스 객체가 없는 샘플입니다 — [다른 샘플]")
            return
        q = res["quality"]
        self.v_gen.set(self._draw(res["image"], res["anns"], cfg, names, res.get("image_class")),
                       f"{'✓ 통과' if q.passed else '✗ 거부'} · 객체 {len(res['anns'])}")
        d = q.to_dict()
        txt = (f"구조 유지 {d['structure']:.2f} · 변화량 {d['change']:.3f} · 객체 유지 최소 "
               f"{d['object_min'] if d['object_min'] is not None else '-'}")
        if d["clip_gain"] is not None:
            txt += f" · CLIP {d['clip_gain']:+.3f}"
        if d["dropped_labels"]:
            txt += f" · 라벨 제거 {len(d['dropped_labels'])}"
        if d["reasons"]:
            txt += "<br><span style='color:#c0392b'>" + " / ".join(d["reasons"]) + "</span>"
        if res.get("region_idx"):
            txt += f"<br>파손 적용 객체 {len(res['region_idx'])}개"
        self.lb_q.setText(txt)

    # ---------------------------------------------------------------- run
    def start(self):
        if not self.ds.info:
            self._check_ready()
            return
        self.ds.prepare_run()
        if not self._check_ready():
            return
        from ..genai.transform import run_transform

        cfg = self.cfg()
        self.run.log.clear()
        self.run.running(True)
        self._th = JobThread(run_transform, cfg)
        self._th.progress.connect(lambda d, t, m: (self.run.bar.setMaximum(t), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(self._done)
        self._th.failed.connect(lambda m: (self.run.running(False), self.run.log.appendPlainText(m)))
        self._th.start()

    def cancel(self):
        if self._th:
            self._th.cancel.set()
            self.run.log.appendPlainText("취소 요청됨 (현재 이미지 완료 후 중단)…")

    def _done(self, rep):
        self.run.running(False)
        s = rep["stats"]
        self.run.log.appendPlainText(f"통과 {s['passed']} · 거부 {s['rejected']} · 라벨 제거 {s['labels_dropped']}")
        self.tab.review.ed_dir.setText(self.ds.ed_out.text())
        if QMessageBox.question(self, "완료", f"통과 {s['passed']}장 · 거부 {s['rejected']}장\n검수 탭에서 결과를 확인할까요?") \
                == QMessageBox.Yes:
            self.tab.show_review(self.ds.ed_out.text())


# ============================================================================ 3. 객체 합성
class CopyPastePage(QWidget):
    def __init__(self, win, tab):
        super().__init__()
        self.win, self.tab = win, tab
        self.bank = None
        self.bank_key = None
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        inner = QWidget()
        L = QVBoxLayout(inner)
        self.ds = ProjectBox(tab, "copypaste")
        self.ds.analyzed.connect(self._on_info)
        L.addWidget(self.ds)
        g = QGroupBox("객체 · 배경 출처")
        f = QFormLayout(g)
        self.chk_ds_obj = QCheckBox("프로젝트 라벨에서 객체 추출 (폴리곤 라벨=정밀, 박스 라벨=GrabCut)")
        self.chk_ds_obj.setChecked(True)
        f.addRow(self.chk_ds_obj)
        w, self.ed_cut = _dir_row("선택) 투명 PNG 컷아웃 폴더: <폴더>/<클래스명>/*.png", "컷아웃 폴더", self)
        f.addRow("컷아웃 폴더", w)
        w2, self.ed_bg = _dir_row("선택) 배경 원본 폴더 — 비우면 프로젝트 이미지(기존 라벨 유지)", "배경 폴더", self)
        f.addRow("배경 폴더", w2)
        self.cls_list = ClassList()
        f.addRow("합성할 클래스", self.cls_list)
        L.addWidget(g)
        g2 = QGroupBox("합성 설정")
        f2 = QFormLayout(g2)
        self.sp_n = _ispin(200, 1, 100000, 10)
        self.sp_min = _ispin(1, 1, 50)
        self.sp_max = _ispin(5, 1, 100)
        self.cb_scale = QComboBox()
        self.cb_scale.addItem("데이터셋 크기 분포 따르기 (고도 일관)", "match")
        self.cb_scale.addItem("배경 대비 비율 범위", "relative")
        self.sp_smin = _dspin(0.03, 0.005, 0.8, 0.005, 3)
        self.sp_smax = _dspin(0.12, 0.005, 0.9, 0.005, 3)
        self.sp_jit = _dspin(0.2, 0.0, 0.8, 0.05)
        self.cb_rot = QComboBox()
        self.cb_rot.addItem("자유 회전 0~360° (항공 뷰)", "free")
        self.cb_rot.addItem("소폭 ±15° (지상 뷰)", "small")
        self.cb_rot.addItem("회전 없음", "none")
        self.chk_flip = QCheckBox("좌우 반전")
        self.chk_flip.setChecked(True)
        self.sp_color = _dspin(0.6, 0.0, 1.0, 0.05)
        self.cb_blend = QComboBox()
        self.cb_blend.addItem("페더 (부드러운 경계)", "feather")
        self.cb_blend.addItem("포아송 (seamless clone)", "poisson")
        self.cb_blend.addItem("알파 (경계 그대로)", "alpha")
        self.sp_shadow = _dspin(0.3, 0.0, 0.9, 0.05)
        self.sp_ov = _dspin(0.05, 0.0, 1.0, 0.05)
        self.sp_occ = _dspin(0.6, 0.1, 1.0, 0.05)
        self.chk_gc = QCheckBox("Detect 박스는 GrabCut으로 전경만 추출")
        self.chk_gc.setChecked(True)
        self.sp_seed = _ispin(7, 0, 2**31 - 1)
        for lab, wdg in (("생성 이미지 수", self.sp_n), ("이미지당 최소 객체", self.sp_min), ("이미지당 최대 객체", self.sp_max),
                         ("크기 방식", self.cb_scale), ("비율 최소", self.sp_smin), ("비율 최대", self.sp_smax),
                         ("크기 흔들림(±)", self.sp_jit), ("회전", self.cb_rot), ("", self.chk_flip),
                         ("색 보정 강도", self.sp_color), ("경계 합성", self.cb_blend), ("그림자 농도", self.sp_shadow),
                         ("최대 겹침(IoU)", self.sp_ov), ("가려진 기존 라벨 제거 기준", self.sp_occ), ("", self.chk_gc),
                         ("시드", self.sp_seed)):
            f2.addRow(lab, wdg)
        L.addWidget(g2)
        L.addStretch(1)
        left = _scroll(inner)
        left.setMinimumWidth(500)
        split.addWidget(left)

        right = QWidget()
        R = QVBoxLayout(right)
        top = QHBoxLayout()
        t = QLabel("미리보기")
        t.setObjectName("sectionTitle")
        top.addWidget(t)
        top.addStretch(1)
        self.b_prev = QPushButton("미리보기 (4장)")
        self.b_prev.setObjectName("primary")
        self.b_prev.clicked.connect(self.preview)
        top.addWidget(self.b_prev)
        R.addLayout(top)
        grid = QGridLayout()
        self.views = [CaptionedImage(f"합성 #{i + 1}") for i in range(4)]
        for i, v in enumerate(self.views):
            grid.addWidget(v, i // 2, i % 2)
        R.addLayout(grid, 1)
        self.lb = QLabel("")
        self.lb.setObjectName("note")
        self.lb.setWordWrap(True)
        R.addWidget(self.lb)
        self.run = RunBar()
        self.run.btn_run.clicked.connect(self.start)
        self.run.btn_cancel.clicked.connect(lambda: self._th and self._th.cancel.set())
        self.run.btn_open.clicked.connect(lambda: open_folder(self.ds.ed_out.text()))
        R.addWidget(self.run)
        split.addWidget(right)
        split.setSizes([540, 700])
        root.addWidget(split)
        self._th = self._pth = None

    def _on_info(self, info):
        self.cls_list.set_names(info.class_names, checked_all=True, counts=info.class_counts)
        self.bank = None

    def cfg(self):
        from ..genai.copypaste import CopyPasteConfig

        info = self.ds.info
        return CopyPasteConfig(
            project_dir=self.ds.project_dir(),
            flip_idx=self.tab.project.flip_idx if self.tab.project is not None and self.tab.project.kpt_names else None,
            cutout_dir=self.ed_cut.text().strip(), background_dir=self.ed_bg.text().strip(),
            output_dir=self.ds.ed_out.text().strip(), task=info.task if info else "detect",
            class_names=list(info.class_names) if info else [], classes=self.cls_list.checked(),
            n_images=self.sp_n.value(), objs_min=self.sp_min.value(), objs_max=max(self.sp_min.value(), self.sp_max.value()),
            scale_mode=self.cb_scale.currentData(), scale_min=self.sp_smin.value(), scale_max=self.sp_smax.value(),
            scale_jitter=self.sp_jit.value(), rotate=self.cb_rot.currentData(), flip=self.chk_flip.isChecked(),
            color_match=self.sp_color.value(), blend=self.cb_blend.currentData(), shadow=self.sp_shadow.value(),
            max_overlap=self.sp_ov.value(), occlusion_drop=self.sp_occ.value(), grabcut=self.chk_gc.isChecked(),
            seed=self.sp_seed.value())

    def _get_bank(self, cfg):
        from ..genai.copypaste import build_bank

        key = (cfg.project_dir if self.chk_ds_obj.isChecked() else "", cfg.cutout_dir, tuple(cfg.classes), cfg.grabcut,
               cfg.task)
        if self.bank is None or self.bank_key != key:
            c2 = cfg
            if not self.chk_ds_obj.isChecked():
                from dataclasses import replace
                c2 = replace(cfg, dataset_dir="", project_dir="")
            self.bank = build_bank(c2, log=lambda m: None)
            self.bank_key = key
        return self.bank

    def _ready(self):
        if not self.ds.info:
            QMessageBox.information(self, "데이터셋", "먼저 데이터셋을 분석하세요.")
            return False
        return True

    def preview(self):
        if not self._ready():
            return
        cfg = self.cfg()
        self.b_prev.setEnabled(False)
        self.lb.setText("객체 뱅크 구성 중… (처음 1회, 데이터셋 크기에 따라 수 초~수십 초)")

        def job():
            from ..genai import annstore
            from ..genai.copypaste import paste_objects

            bank = self._get_bank(cfg)
            rng = np.random.default_rng(cfg.seed + 11)
            if cfg.background_dir:
                from ..genai.copypaste import _list_images
                bgs = [(p, None) for p in _list_images(cfg.background_dir)]
            else:
                bgs = [(s.image, s.label) for s in self.ds.info.samples]
            outs = []
            for _ in range(4):
                p, lab = bgs[int(rng.integers(len(bgs)))]
                bg = imread(p)
                h, w = bg.shape[:2]
                a = annstore.load_any(lab, cfg.task, w, h)[0] if lab else []
                img, anns, meta = paste_objects(bg, a, bank, cfg, rng, cfg.task)
                outs.append((img, anns, meta, len(a)))
            return outs, bank.count()

        self._pth = FuncThread(job)
        self._pth.done.connect(lambda r: self._show(r, cfg))
        self._pth.failed.connect(lambda m: (self.b_prev.setEnabled(True), self.lb.setText(
            f"<span style='color:#c0392b'>{m.splitlines()[0]}</span>")))
        self._pth.start()

    def _show(self, r, cfg):
        self.b_prev.setEnabled(True)
        outs, cnt = r
        for v, (img, anns, meta, n0) in zip(self.views, outs):
            v.set(draw(img, anns, cfg.task, cfg.class_names, self.ds.skeleton()),
                  f"합성 {meta['pasted']}개 · 기존 {n0 - meta['occluded_removed']}개 유지")
        names = cfg.class_names
        self.lb.setText("객체 뱅크: " + ", ".join(f"{names[c] if c < len(names) else c} {n}개" for c, n in sorted(cnt.items())))

    def start(self):
        if not self._ready():
            return
        self.ds.prepare_run()
        from ..genai.copypaste import run_copypaste

        cfg = self.cfg()
        bank = None
        try:
            bank = self._get_bank(cfg)
        except Exception:
            bank = None
        self.run.log.clear()
        self.run.running(True)

        def job(cfg, progress=None, log=print, cancel=None):
            return run_copypaste(cfg, progress, log, cancel, bank)

        self._th = JobThread(job, cfg)
        self._th.progress.connect(lambda d, t, m: (self.run.bar.setMaximum(t), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(self._done)
        self._th.failed.connect(lambda m: (self.run.running(False), self.run.log.appendPlainText(m)))
        self._th.start()

    def _done(self, rep):
        self.run.running(False)
        self.tab.review.ed_dir.setText(self.ds.ed_out.text())
        if QMessageBox.question(self, "완료", f"{rep['stats']['written']}장 생성. 검수 탭에서 확인할까요?") == QMessageBox.Yes:
            self.tab.show_review(self.ds.ed_out.text())


# ============================================================================ 4. 검수
class ThumbThread(QThread):
    thumb = Signal(int, QImage)

    def __init__(self, root, items, task, names, size=240, skeleton=None):
        super().__init__()
        self.root, self.items, self.task, self.names, self.size = Path(root), items, task, names, size
        self.skeleton = skeleton
        self.stop = False

    def run(self):
        import cv2

        for i, it in enumerate(self.items):
            if self.stop:
                return
            img = imread(it.image_path(self.root))
            if img is None:
                continue
            h, w = img.shape[:2]
            anns, ic = annstore.load_any(it.label_path(self.root), self.task, w, h)
            vis = draw(img, anns, self.task, self.names, self.skeleton, ic)
            r = self.size / max(h, w)
            vis = cv2.resize(vis, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA)
            self.thumb.emit(i, _np_to_qimage(vis))


class ReviewPage(QWidget):
    def __init__(self, win, tab):
        super().__init__()
        self.win, self.tab = win, tab
        self.items = []
        self.report = {}
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        w, self.ed_dir = _dir_row("생성 결과 폴더 (gen_report.json 포함)", "결과 폴더", self)
        b = QPushButton("불러오기")
        b.setObjectName("primary")
        b.clicked.connect(self.load)
        self.cb_filter = QComboBox()
        self.cb_filter.addItems(["전체", "통과만", "거부만"])
        self.cb_filter.currentIndexChanged.connect(self._apply_filter)
        self.cb_run = QComboBox()
        self.cb_run.setMinimumWidth(260)
        self.cb_run.activated.connect(self._pick_run)
        top.addWidget(QLabel("실행"))
        top.addWidget(self.cb_run)
        top.addWidget(QLabel("폴더"))
        top.addWidget(w, 1)
        top.addWidget(b)
        top.addWidget(self.cb_filter)
        root.addLayout(top)
        tab.project_changed.connect(self.refresh_runs)
        split = QSplitter(Qt.Horizontal)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(220, 165))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.setSpacing(6)
        self.grid.setUniformItemSizes(True)
        self.grid.currentItemChanged.connect(self._detail)
        self.grid.itemDoubleClicked.connect(lambda it: self._toggle([it]))
        split.addWidget(self.grid)
        det = QWidget()
        dl = QVBoxLayout(det)
        self.big = ImageView()
        dl.addWidget(self.big, 1)
        self.info = QTextBrowser()
        self.info.setMaximumHeight(180)
        dl.addWidget(self.info)
        split.addWidget(det)
        split.setSizes([760, 460])
        root.addWidget(split, 1)
        row = QHBoxLayout()
        ba = QPushButton("선택 → 통과")
        ba.clicked.connect(lambda: self._set_sel(True))
        br = QPushButton("선택 → 거부")
        br.clicked.connect(lambda: self._set_sel(False))
        self.sp_thr = _dspin(0.4, 0.0, 1.0, 0.05)
        bt = QPushButton("구조 점수 미만 모두 거부")
        bt.clicked.connect(self._reject_below)
        bo = QPushButton("폴더 열기")
        bo.clicked.connect(lambda: open_folder(self.ed_dir.text()))
        self.lb = QLabel("")
        self.lb.setObjectName("note")
        for x in (ba, br, QLabel("   기준"), self.sp_thr, bt, bo):
            row.addWidget(x)
        row.addStretch(1)
        row.addWidget(self.lb)
        root.addLayout(row)
        hint = QLabel("더블클릭: 통과/거부 전환 · 거부된 파일은 _rejected 폴더로 이동(삭제 안 함) · 통과 파일만 학습·증강에 사용됩니다.")
        hint.setObjectName("note")
        root.addWidget(hint)
        self._tt = None

    def refresh_runs(self):
        self.cb_run.clear()
        p = self.tab.project
        if p is None:
            return
        for r in reversed(p.runs()):
            self.cb_run.addItem(r.name, str(r))
        if self.cb_run.count() and not self.ed_dir.text().strip():
            self.ed_dir.setText(self.cb_run.itemData(0))

    def _pick_run(self, i):
        self.ed_dir.setText(self.cb_run.itemData(i))
        self.load()

    def showEvent(self, e):
        super().showEvent(e)
        cur = self.ed_dir.text()
        self.refresh_runs()
        if cur:
            self.ed_dir.setText(cur)

    def load(self):
        from ..genai.review import load_review

        p = self.ed_dir.text().strip()
        if not p or not Path(p).is_dir():
            QMessageBox.warning(self, "검수", "결과 폴더를 선택하세요.")
            return
        self.items, self.report = load_review(p)
        cfg = self.report.get("config", {})
        self.task = cfg.get("task", "detect")
        self.names = self.report.get("class_names", cfg.get("class_names", []))
        if not self.report.get("records"):
            try:
                info = analyze_dataset(scan_dataset(p))
                self.task, self.names = info.task, info.class_names
            except Exception:
                pass
        self.grid.clear()
        ph = QPixmap(220, 165)
        ph.fill(QColor("#374151"))
        for i, it in enumerate(self.items):
            li = QListWidgetItem(QIcon(ph), self._label(it))
            li.setData(Qt.UserRole, i)
            self._color(li, it)
            self.grid.addItem(li)
        if self._tt:
            self._tt.stop = True
            self._tt.wait(2000)
        proj = self.tab.project
        self._tt = ThumbThread(p, self.items, self.task, self.names,
                               skeleton=proj.kpt_skeleton if proj is not None else None)
        self._tt.thumb.connect(self._set_thumb)
        self._tt.start()
        self._count()
        self._apply_filter()

    def _label(self, it):
        r = it.record
        q = r.get("quality") or {}
        s = f"{'✓' if it.accepted else '✗'} {r.get('effect', '')}"
        if q.get("structure") is not None:
            s += f"  구조 {q['structure']:.2f}"
        return s

    def _color(self, li, it):
        li.setForeground(QColor("#059669" if it.accepted else "#c0392b"))

    def _set_thumb(self, i, qimg):
        for r in range(self.grid.count()):
            li = self.grid.item(r)
            if li.data(Qt.UserRole) == i:
                li.setIcon(QIcon(QPixmap.fromImage(qimg)))
                break

    def _count(self):
        a = sum(1 for i in self.items if i.accepted)
        self.lb.setText(f"통과 {a} · 거부 {len(self.items) - a} · 전체 {len(self.items)}")

    def _apply_filter(self):
        mode = self.cb_filter.currentIndex()
        for r in range(self.grid.count()):
            li = self.grid.item(r)
            it = self.items[li.data(Qt.UserRole)]
            li.setHidden((mode == 1 and not it.accepted) or (mode == 2 and it.accepted))

    def _detail(self, cur, prev=None):
        if cur is None:
            return
        it = self.items[cur.data(Qt.UserRole)]
        root = Path(self.ed_dir.text())
        img = imread(it.image_path(root))
        if img is not None:
            h, w = img.shape[:2]
            anns, ic = annstore.load_any(it.label_path(root), self.task, w, h)
            proj = self.tab.project
            self.big.set_image(draw(img, anns, self.task, self.names,
                                    proj.kpt_skeleton if proj is not None else None, ic))
        r = it.record
        q = r.get("quality") or {}
        lines = [f"<b>{it.file}</b> — {'통과' if it.accepted else '거부'}",
                 f"효과: {r.get('effect', '-')} · 방식: {r.get('backend', '-')} · 시드: {r.get('seed', '-')}",
                 f"원본: {r.get('source', '-')}" + (f" · 이미지 분류: {r['image_class']}" if r.get("image_class") else "")]
        if q:
            lines.append(f"구조 {q.get('structure')} · 변화량 {q.get('change')} · 객체 최소 {q.get('object_min')} · "
                         f"CLIP {q.get('clip_gain')}")
            if q.get("reasons"):
                lines.append("<span style='color:#c0392b'>" + " / ".join(q["reasons"]) + "</span>")
        if r.get("prompt"):
            lines.append(f"<span style='color:#6b7280'>prompt: {r['prompt']}</span>")
        self.info.setHtml("<br>".join(lines))

    def _toggle(self, lis):
        from ..genai.review import save_decisions, set_accepted

        root = Path(self.ed_dir.text())
        for li in lis:
            it = self.items[li.data(Qt.UserRole)]
            set_accepted(root, it, not it.accepted)
            li.setText(self._label(it))
            self._color(li, it)
        save_decisions(root, self.items)
        self._count()
        self._apply_filter()

    def _set_sel(self, accepted):
        lis = [li for li in self.grid.selectedItems() if self.items[li.data(Qt.UserRole)].accepted != accepted]
        self._toggle(lis)

    def _reject_below(self):
        thr = self.sp_thr.value()
        lis = []
        for r in range(self.grid.count()):
            li = self.grid.item(r)
            it = self.items[li.data(Qt.UserRole)]
            s = (it.record.get("quality") or {}).get("structure")
            if it.accepted and s is not None and s < thr:
                lis.append(li)
        self._toggle(lis)


# ============================================================================ tab
GEN_STEPS = [
    ("proj", "1. 프로젝트 · 원본", "프로젝트 · 원본 이미지",
     "원본(raw) 이미지에서 시작합니다. YOLO 태스크(Detect / Segment / OBB / Pose / Classify)와 클래스, "
     "키포인트 스키마(자세 추정), 이미지 분류 클래스를 정하고 원본 이미지·동영상 프레임을 가져옵니다."),
    ("label", "2. 라벨링", "라벨링 (박스 · 폴리곤 · 키포인트 · 이미지 분류)",
     "자동 라벨(OWLv2/Grounding DINO + SAM, ViTPose, CLIP)로 초안을 만들고 편집기로 수정합니다. 수동 라벨링은 모델 없이 가능합니다."),
    ("assist", "3. AI Label Assist", "AI Label Assist (project model)",
     "Train a model only on this project's confirmed labels, pre-label the remaining images, "
     "correct them in Labeling and repeat."),
    ("copypaste", "4. 객체 합성", "객체 합성 (Copy-Paste)",
     "라벨된 객체(키포인트 포함)나 투명 PNG 컷아웃을 배경 이미지에 합성하고 라벨을 자동으로 만듭니다."),
    ("review", "5. 검수", "생성 결과 검수",
     "실행별 결과를 확인하고 통과/거부를 정합니다. 거부된 파일은 _rejected 폴더로 옮겨지며 언제든 복원할 수 있습니다."),
    ("export", "6. 내보내기", "YOLO 데이터셋으로 내보내기",
     "원본과 통과된 생성본을 선택한 YOLO 태스크 형식으로 묶어 저장하고 ② 데이터셋 증강으로 넘깁니다."),
    ("models", "모델 · GPU", "모델 · GPU 설정",
     "GPU 상태 확인, 자동 라벨·프로젝트 학습 모델 다운로드, 폐쇄망 설치 번들 생성."),
]


def _page_header(title: str, sub: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 6)
    t = QLabel(title)
    t.setObjectName("pageTitle")
    s = QLabel(sub)
    s.setObjectName("pageSub")
    s.setWordWrap(True)
    lay.addWidget(t)
    lay.addWidget(s)
    return w


class GenAITab(QWidget):
    """① 데이터셋 생성 — ② 증강과 같은 레이아웃(왼쪽 단계 목록 + 페이지 + 이전/다음)."""
    project_changed = Signal()

    def __init__(self, win):
        super().__init__()
        from PySide6.QtWidgets import QStackedWidget

        from .. import __version__
        from .assist_page import AssistPage
        from .create_pages import ExportPage, LabelPage, ProjectPage

        self.win = win
        self.project = None
        self.models = ModelsPage(win)
        self.proj = ProjectPage(win, self)
        self.label = LabelPage(win, self)
        self.assist = AssistPage(win, self)
        self.copypaste = CopyPastePage(win, self)
        self.review = ReviewPage(win, self)
        self.export = ExportPage(win, self)
        self.pages = [getattr(self, k) for k, *_ in GEN_STEPS]

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        side = QWidget()
        side.setObjectName("side")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        logo = QLabel("YOLO\nAugmentor")
        logo.setObjectName("logo")
        sl.addWidget(logo)
        sec = QLabel("데이터셋 생성")
        sec.setObjectName("sideSection")
        sl.addWidget(sec)
        self.steps = QListWidget()
        self.steps.setObjectName("steps")
        for _, label, *_ in GEN_STEPS:
            self.steps.addItem(QListWidgetItem(label))
        self.steps.currentRowChanged.connect(self._on_step)
        sl.addWidget(self.steps, 1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("ver")
        sl.addWidget(ver)
        h.addWidget(side)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        self.stack = QStackedWidget()
        for (key, _, title, sub), page in zip(GEN_STEPS, self.pages):
            holder = QWidget()
            hl = QVBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(_page_header(title, sub))
            if page.layout() is not None:
                page.layout().setContentsMargins(0, 0, 0, 0)
            hl.addWidget(page, 1)
            self.stack.addWidget(holder)
        bl.addWidget(self.stack, 1)
        nav = QHBoxLayout()
        self.lb_proj = QLabel("")
        self.lb_proj.setObjectName("note")
        nav.addWidget(self.lb_proj)
        nav.addStretch(1)
        self.btn_prev = QPushButton("◀  이전")
        self.btn_next = QPushButton("다음  ▶")
        self.btn_next.setObjectName("primary")
        self.btn_prev.clicked.connect(lambda: self.go(self.stack.currentIndex() - 1))
        self.btn_next.clicked.connect(lambda: self.go(self.stack.currentIndex() + 1))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        bl.addLayout(nav)
        h.addWidget(body, 1)
        self.project_changed.connect(self._update_steps)
        self.steps.blockSignals(True)
        self.steps.setCurrentRow(0)
        self.steps.blockSignals(False)
        self._update_steps()

    # ---------------------------------------------------------------- navigation
    def _on_step(self, row):
        self.go(row)

    def go(self, idx):
        if idx < 0 or idx >= len(self.pages):
            return
        key = GEN_STEPS[idx][0]
        if self.project is None and key not in ("proj", "models"):
            self.steps.blockSignals(True)
            self.steps.setCurrentRow(self.stack.currentIndex())
            self.steps.blockSignals(False)
            self.win.statusBar().showMessage("먼저 '1. 프로젝트 · 원본'에서 프로젝트를 만들거나 여세요.", 4000)
            return
        cur = self.pages[self.stack.currentIndex()]
        if cur is self.label:
            self.label.save_current()
        self.stack.setCurrentIndex(idx)
        self.steps.blockSignals(True)
        self.steps.setCurrentRow(idx)
        self.steps.blockSignals(False)
        self._update_steps()

    def setCurrentWidget(self, page):
        if page in self.pages:
            self.go(self.pages.index(page))

    def currentWidget(self):
        return self.pages[self.stack.currentIndex()]

    def _update_steps(self):
        i = self.stack.currentIndex()
        last_main = len(GEN_STEPS) - 2
        self.btn_prev.setEnabled(0 < i <= last_main)
        self.btn_next.setVisible(i < last_main)
        self.btn_prev.setVisible(i <= last_main)
        done = self._done_flags()
        for r, (key, label, *_) in enumerate(GEN_STEPS):
            self.steps.item(r).setText(("✓ " if done.get(key) and r != i else "") + label)
        p = self.project
        if p is None:
            self.lb_proj.setText("프로젝트 없음")
        else:
            s = p.stats()
            self.lb_proj.setText(f"프로젝트 {p.root.name} · 원본 {s['images']}장 · 라벨 {s['labeled']}장 · "
                                 f"생성 실행 {s['runs']}회 · 태스크 {p.fmt}")

    def _done_flags(self) -> dict:
        p = self.project
        if p is None:
            return {}
        s = p.stats()
        runs = [r.name for r in p.runs()]
        exp = p.root / "export"
        return dict(proj=s["images"] > 0, label=s["labeled"] > 0,
                    assist=(p.root / "assist").exists() and any((p.root / "assist").glob("*/assist_model.json")),
                    copypaste=any(r.startswith("copypaste") for r in runs),
                    review=any((r / "gen_report.json").exists() and "reviewed_accepted" in
                               (r / "gen_report.json").read_text(encoding="utf-8") for r in p.runs()),
                    export=exp.exists() and any(exp.iterdir()))

    def set_project(self, project):
        self.project = project
        self.project_changed.emit()

    def show_review(self, path):
        self.review.ed_dir.setText(path)
        self.setCurrentWidget(self.review)
        self.review.load()
