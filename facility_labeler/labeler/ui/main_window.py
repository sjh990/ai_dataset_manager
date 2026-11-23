"""메인 창: 탭 5개 (1 데이터셋 / 2 라벨링 / 3 AI 학습 / 4 내보내기 / 5 YOLO).

코드 안의 Project = 화면에서 부르는 '데이터셋' (폴더 이름 datasets, 파일 project.json 은 호환을 위해 그대로 둠).
"""
import json
import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget

from .. import config as C
from ..project import Project
from .ai_tab import AiTab
from .dataset_tab import DatasetTab
from .export_tab import ExportTab
from .label_tab import LabelTab
from .yolo_tab import YoloTab

LAST_PROJECT_FILE = C.APP_DIR / "last_project.json"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("시설물 라벨링 툴 (도로표지판 / 중앙분리대 / 무단횡단방지펜스)")
        self.resize(1500, 900)
        self.project = None
        self.tabs = QTabWidget()
        self.dataset_tab = DatasetTab(self)
        self.label_tab = LabelTab(self)
        self.ai_tab = AiTab(self)
        self.export_tab = ExportTab(self)
        self.yolo_tab = YoloTab(self)
        self.tabs.addTab(self.dataset_tab, "1. 데이터셋 / 영상")
        self.tabs.addTab(self.label_tab, "2. 라벨링")
        self.tabs.addTab(self.ai_tab, "3. AI 학습")
        self.tabs.addTab(self.export_tab, "4. 내보내기")
        self.tabs.addTab(self.yolo_tab, "5. YOLO 학습 / 추론")
        self.tabs.currentChanged.connect(self.on_tab)
        self.setCentralWidget(self.tabs)
        self.status = QLabel("")
        self.statusBar().addWidget(self.status)
        b_reload = QPushButton("새로고침 (F5)")
        b_reload.setToolTip("툴 다시 시작: 바뀐 코드 반영. 지금 데이터셋, 탭, 이미지 위치 유지")
        b_reload.clicked.connect(self.restart)
        self.statusBar().addPermanentWidget(b_reload)
        sc = QShortcut(QKeySequence("F5"), self)
        sc.setContext(Qt.ApplicationShortcut)
        sc.activated.connect(self.restart)
        self.open_last_project()

    def open_last_project(self):
        try:
            last = json.loads(LAST_PROJECT_FILE.read_text(encoding="utf-8"))
            self.set_project(Project(last["path"]))
        except Exception:
            self.status.setText("데이터셋 열기 또는 새로 만들기 필요 (1. 데이터셋 / 영상 탭)")
            return
        place = last.get("restart")                 # 새로고침 직전 위치로 되돌림 (탭, 보기, 이미지)
        if place:
            self.label_tab.filter.setCurrentIndex(place.get("filter", 0))
            self.label_tab.select_image(place.get("image"))
            self.tabs.setCurrentIndex(place.get("tab", 0))
            LAST_PROJECT_FILE.write_text(json.dumps({"path": last["path"]}, ensure_ascii=False), encoding="utf-8")

    def busy_jobs(self, skip=None):
        """지금 돌고 있는 작업 이름 목록 (skip 탭은 빼고)."""
        jobs = (("프레임 추출, 구간 변경", self.dataset_tab, self.dataset_tab.run.busy()),
                ("AI 학습", self.ai_tab, self.ai_tab.run.busy()),
                ("AI 자동 라벨링", self.label_tab, self.label_tab.batch_busy()),
                ("내보내기", self.export_tab, self.export_tab.run.busy()),
                ("YOLO 학습 / 추론", self.yolo_tab, self.yolo_tab.busy()))
        return [name for name, tab, b in jobs if b and tab is not skip]

    def restart(self):
        """새로고침: 지금 상태 저장 → 툴을 새로 실행 (바뀐 코드 반영) → 지금 창 닫기."""
        busy = self.busy_jobs()
        if busy:
            QMessageBox.information(self, "새로고침 불가", f"{', '.join(busy)} 진행 중: 끝난 뒤 새로고침")
            return
        self.label_tab.save_current()
        if self.project is not None:
            lt = self.label_tab
            place = {"tab": self.tabs.currentIndex(), "filter": lt.filter.currentIndex(),
                     "image": lt.images[lt.index].name if 0 <= lt.index < len(lt.images) else None}
            LAST_PROJECT_FILE.write_text(json.dumps({"path": str(self.project.root), "restart": place},
                                                    ensure_ascii=False), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.Popen([sys.executable, str(C.APP_DIR / "main.py")], cwd=str(C.APP_DIR), creationflags=flags)
        self.close()

    def set_project(self, project):
        self.label_tab.save_current()
        self.project = project
        LAST_PROJECT_FILE.write_text(json.dumps({"path": str(project.root)}, ensure_ascii=False), encoding="utf-8")
        self.refresh_all()

    def refresh_all(self):
        self.dataset_tab.refresh()
        self.label_tab.reload_list()
        self.ai_tab.refresh()
        self.export_tab.refresh()
        self.yolo_tab.refresh()
        self.refresh_status()

    def refresh_status(self):
        if self.project is None:
            return
        s = self.project.summary()
        self.status.setText(f"{self.project.info['name']}  |  전체 {s['total']}장  /  사람 검수 {s['human']}장  /  "
                            f"AI 예측 {s['ai']}장  /  미작업 {s['todo']}장")

    def on_tab(self, i):
        self.label_tab.save_current()
        w = self.tabs.widget(i)
        if w is not self.label_tab and hasattr(w, "refresh"):
            w.refresh()

    def closeEvent(self, e):
        busy = self.busy_jobs()
        if busy:                          # 작업 중에 닫으면 파일이 반쯤 쓰인 채로 남을 수 있음
            QMessageBox.information(self, "닫기 불가", f"{', '.join(busy)} 진행 중: 끝나거나 중지한 뒤 닫기")
            e.ignore()
            return
        self.label_tab.save_current()
        super().closeEvent(e)
