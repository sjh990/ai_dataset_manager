"""메인 윈도우: 단계 목록 + 페이지 스택 + 실행 제어."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton, QStackedWidget, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import APP_NAME, __version__
from ..ops_spec import AUGMENT_BY_KEY, PREPROCESS_BY_KEY, defaults
from ..pipeline import PipelineConfig, run_pipeline
from .pages import (AugParamPage, AugSelectPage, DatasetPage, PreParamPage, PreSelectPage, RunPage, SplitPage)
from .style import STYLE

STEPS = ["데이터셋 · 모델", "0. 분할 비율", "1. 전처리 선택", "2. 전처리 수치",
         "3. 증강 선택", "4. 증강 수치", "5. 증강 배수 · 실행"]


class RunThread(QThread):
    progress = Signal(int, int, str)
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg
        self.cancel = threading.Event()

    def run(self):
        try:
            res = run_pipeline(self.cfg, lambda d, t, m: self.progress.emit(d, t, m), self.log.emit, self.cancel)
            self.finished_ok.emit(res)
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(1280, 820)
        self.info = None
        self.cfg = PipelineConfig()
        self.pre_selected: list[str] = []
        self.pre_params: dict = {}
        self.aug_selected: list[str] = []
        self.aug_params: dict = {}
        self.max_step = 0
        self.run_thread: RunThread | None = None

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.setCentralWidget(self.tabs)
        central = QWidget()
        from .genai_tab import GenAITab
        self.genai = GenAITab(self)
        self.tabs.addTab(self.genai, "① 데이터셋 생성")
        self.tabs.addTab(central, "② 데이터셋 증강")
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QWidget()
        side.setObjectName("side")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        logo = QLabel("YOLO\nAugmentor")
        logo.setObjectName("logo")
        sl.addWidget(logo)
        sec = QLabel("데이터셋 증강")
        sec.setObjectName("sideSection")
        sl.addWidget(sec)
        self.steps = QListWidget()
        self.steps.setObjectName("steps")
        for s in STEPS:
            self.steps.addItem(QListWidgetItem(s))
        self.steps.currentRowChanged.connect(self._on_step_clicked)
        sl.addWidget(self.steps, 1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("ver")
        sl.addWidget(ver)
        h.addWidget(side)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        self.stack = QStackedWidget()
        self.p_ds = DatasetPage(self)
        self.p_split = SplitPage(self)
        self.p_pre_sel = PreSelectPage(self)
        self.p_pre_par = PreParamPage(self)
        self.p_aug_sel = AugSelectPage(self)
        self.p_aug_par = AugParamPage(self)
        self.p_run = RunPage(self)
        self.pages = [self.p_ds, self.p_split, self.p_pre_sel, self.p_pre_par, self.p_aug_sel, self.p_aug_par,
                      self.p_run]
        for p in self.pages:
            self.stack.addWidget(p)
        bl.addWidget(self.stack, 1)
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀  이전")
        self.btn_next = QPushButton("다음  ▶")
        self.btn_next.setObjectName("primary")
        self.btn_prev.clicked.connect(lambda: self.go(self.stack.currentIndex() - 1))
        self.btn_next.clicked.connect(lambda: self.go(self.stack.currentIndex() + 1))
        nav.addStretch(1)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        bl.addLayout(nav)
        h.addWidget(body, 1)

        self._menu()
        self.p_split.load_cfg(self.cfg)
        self.p_aug_sel._offline()  # 기본 추천 증강 선택
        self.p_pre_sel.set_selected(["resize"])
        self.pre_selected = ["resize"]
        self.aug_selected = self.p_aug_sel.selected()
        self.steps.blockSignals(True)
        self.steps.setCurrentRow(0)
        self.steps.blockSignals(False)
        self._update_nav()
        self.statusBar().showMessage("데이터셋 폴더를 선택하세요.")

    # ------------------------------------------------------------------ menu / config
    def _menu(self):
        m = self.menuBar().addMenu("파일")
        a1 = QAction("설정 불러오기…", self)
        a1.triggered.connect(self.load_config)
        a2 = QAction("설정 저장…", self)
        a2.triggered.connect(self.save_config)
        a3 = QAction("종료", self)
        a3.triggered.connect(self.close)
        m.addActions([a1, a2])
        m.addSeparator()
        m.addAction(a3)
        hm = self.menuBar().addMenu("도움말")
        a4 = QAction("사용 방법", self)
        a4.triggered.connect(self._help)
        hm.addAction(a4)

    def _help(self):
        QMessageBox.information(self, "사용 방법", (
            "① 데이터셋 생성 (원본 이미지에서 시작)\n"
            "  1 프로젝트: 태스크 · 클래스 · 키포인트 스키마 · 이미지 분류 클래스, 원본 이미지 / 동영상 프레임 가져오기\n"
            "  2 라벨링: 박스(B) · 폴리곤(P) · 키포인트(K) · 이미지 분류, 자동 라벨(OWLv2/DINO+SAM, ViTPose, CLIP)\n"
            "  3 AI Label Assist: train on confirmed labels -> pre-label the rest -> fix -> repeat\n"
            "  4 객체 합성 · 5 검수 · 6 내보내기(태스크 선택)\n"
            "  → [② 데이터셋 증강 탭으로 보내기]\n\n"
            "② 데이터셋 증강: 분할(0) → 전처리(1·2) → 증강(3·4) → 배수·실행(5)  — Classify는 <클래스>/이미지 폴더 구조\n\n"
            "설정 파일(json)은 CLI로도 재실행할 수 있습니다: python main.py --cli --config my_config.json"))

    def sync_all(self):
        """모든 페이지 값을 cfg로 모은다 (검증 없이)."""
        for p in (self.p_split,):
            p.leave()
        self.pre_selected = self.p_pre_sel.selected()
        self.aug_selected = self.p_aug_sel.selected()
        self.cfg.preprocess = {k: {**defaults(PREPROCESS_BY_KEY[k]), **(self.pre_params.get(k) or {})}
                               for k in self.pre_selected}
        self.cfg.augment = {k: {**defaults(AUGMENT_BY_KEY[k]), **(self.aug_params.get(k) or {})}
                            for k in self.aug_selected}
        if self.stack.currentWidget() in (self.p_pre_par, self.p_aug_par):
            self.stack.currentWidget().leave()
        self.p_run.apply_to_cfg()

    def save_config(self):
        cur = self.stack.currentWidget()
        if cur is self.p_ds:
            self.p_ds.leave()
        self.sync_all()
        path, _ = QFileDialog.getSaveFileName(self, "설정 저장", str(Path.home() / "augment_config.json"),
                                              "JSON (*.json)")
        if path:
            Path(path).write_text(self.cfg.to_json(), encoding="utf-8")
            self.statusBar().showMessage(f"설정 저장: {path}", 5000)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "설정 불러오기", str(Path.home()), "JSON (*.json)")
        if not path:
            return
        try:
            cfg = PipelineConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception as e:
            QMessageBox.warning(self, "불러오기 실패", str(e))
            return
        self.cfg = cfg
        self.p_ds.load_cfg(cfg)
        self.p_split.load_cfg(cfg)
        self.pre_selected = list(cfg.preprocess)
        self.pre_params.update(copy.deepcopy(cfg.preprocess))
        self.aug_selected = list(cfg.augment)
        self.aug_params.update(copy.deepcopy(cfg.augment))
        self.p_pre_sel.set_selected(self.pre_selected)
        self.p_aug_sel.set_selected(self.aug_selected)
        self.p_run.load_cfg(cfg)
        self.go(0)
        if cfg.dataset_dir and Path(cfg.dataset_dir).is_dir():
            self.p_ds.run_analysis()
        self.statusBar().showMessage(f"설정 불러옴: {path} — 데이터셋 분석 후 단계를 진행하세요.", 8000)

    def preview_cfg(self) -> PipelineConfig:
        c = copy.deepcopy(self.cfg)
        c.preprocess = {k: {**defaults(PREPROCESS_BY_KEY[k]), **(self.pre_params.get(k) or {})}
                        for k in self.pre_selected}
        if self.stack.currentWidget() is self.p_pre_par:
            c.preprocess = self.p_pre_par.collect()
        return c

    # ------------------------------------------------------------------ navigation
    def _on_step_clicked(self, row):
        if row == self.stack.currentIndex():
            return
        if row > self.max_step:
            self.steps.blockSignals(True)
            self.steps.setCurrentRow(self.stack.currentIndex())
            self.steps.blockSignals(False)
            self.statusBar().showMessage("아직 진행하지 않은 단계입니다. [다음]으로 진행하세요.", 3000)
            return
        self.go(row)

    def go(self, idx):
        cur = self.stack.currentIndex()
        if idx < 0 or idx >= len(self.pages):
            return
        if idx > cur:
            for i in range(cur, idx):  # 앞으로 이동 시 중간 단계 검증
                if i != cur:
                    self.pages[i].enter()
                if not self.pages[i].leave():
                    idx = i
                    break
        else:
            self.pages[cur].leave() if cur not in (0,) else None
        self.stack.setCurrentIndex(idx)
        self.pages[idx].enter()
        self.max_step = max(self.max_step, idx)
        self.steps.blockSignals(True)
        self.steps.setCurrentRow(idx)
        self.steps.blockSignals(False)
        self._update_nav()

    def _update_nav(self):
        i = self.stack.currentIndex()
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setVisible(i < len(self.pages) - 1)
        for r in range(self.steps.count()):
            it = self.steps.item(r)
            base = STEPS[r]
            it.setText(("✓ " if r < self.max_step and r != i else "") + base)

    # ------------------------------------------------------------------ run
    def start_run(self):
        self.p_run.apply_to_cfg()
        cfg = copy.deepcopy(self.cfg)
        out = Path(cfg.output_dir)
        if out.exists() and any(out.iterdir()):
            r = QMessageBox.question(self, "출력 폴더", f"출력 폴더가 비어있지 않습니다.\n{out}\n\n"
                                     "같은 이름의 파일은 덮어씁니다. 계속할까요?")
            if r != QMessageBox.Yes:
                return
        self.p_run.log.clear()
        self.p_run.progress.setValue(0)
        self.p_run.set_running(True)
        self.run_thread = RunThread(cfg)
        self.run_thread.progress.connect(self._on_progress)
        self.run_thread.log.connect(self.p_run.log.appendPlainText)
        self.run_thread.finished_ok.connect(self._on_done)
        self.run_thread.failed.connect(self._on_failed)
        self.run_thread.start()

    def cancel_run(self):
        if self.run_thread:
            self.run_thread.cancel.set()
            self.p_run.log.appendPlainText("취소 요청됨…")

    def _on_progress(self, d, t, m):
        self.p_run.progress.setMaximum(t)
        self.p_run.progress.setValue(d)
        self.statusBar().showMessage(f"처리 중 {d}/{t}: {m}")

    def _on_done(self, res):
        self.p_run.set_running(False)
        w = res["written"]
        msg = (("취소됨 — 일부만 저장되었습니다.\n" if res["cancelled"] else "완료!\n") +
               f"train {w.get('train', 0)} · val {w.get('val', 0)} · test {w.get('test', 0)} 장 저장 "
               f"({res['seconds']}초)\n오류 {res['n_errors']}건\n\n{res['output_dir']}")
        rm = res.get("removed", {})
        if rm:
            self.p_run.log.appendPlainText("제거 통계: " + ", ".join(f"{k}={v}" for k, v in rm.items()))
        for s, c in res.get("instances", {}).items():
            self.p_run.log.appendPlainText(f"[{s}] 객체 수: " + ", ".join(f"{k} {v}" for k, v in c.items()))
        self.statusBar().showMessage("완료", 8000)
        QMessageBox.information(self, "결과", msg)

    def _on_failed(self, msg):
        self.p_run.set_running(False)
        self.p_run.log.appendPlainText(msg)
        QMessageBox.critical(self, "실행 실패", msg.splitlines()[0])

    def send_to_augment(self, path: str):
        """① 데이터셋 생성에서 내보낸 폴더를 ② 증강 탭 입력으로 넘긴다."""
        self.tabs.setCurrentIndex(1)
        self.go(0)
        self.p_ds.ed_ds.setText(path)
        self.p_ds.ed_out.setText(str(Path(path).parent / f"{Path(path).name}_augmented"))
        self.p_ds.run_analysis()
        self.statusBar().showMessage(f"생성 데이터셋을 불러왔습니다: {path}", 8000)

    def open_output(self):
        p = self.cfg.output_dir
        if not p or not Path(p).exists():
            QMessageBox.information(self, "출력 폴더", "아직 출력 폴더가 없습니다.")
            return
        if sys.platform.startswith("win"):
            os.startfile(p)  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def closeEvent(self, e):
        if self.run_thread and self.run_thread.isRunning():
            if QMessageBox.question(self, "종료", "작업이 진행 중입니다. 취소하고 종료할까요?") != QMessageBox.Yes:
                e.ignore()
                return
            self.run_thread.cancel.set()
            self.run_thread.wait(10000)
        e.accept()


def _resource(rel: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base / rel


def run_gui():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    f = QFont()
    f.setFamilies(["Malgun Gothic", "맑은 고딕", "Noto Sans CJK KR", "NanumGothic", "Segoe UI", "Ubuntu", "sans-serif"])
    f.setPointSize(10)
    app.setFont(f)
    app.setStyleSheet(STYLE)
    icon = _resource("assets/icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    w = MainWindow()
    w.show()
    return app.exec()
