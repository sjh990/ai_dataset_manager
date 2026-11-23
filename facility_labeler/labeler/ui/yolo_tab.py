"""5. YOLO 학습 / 영상 추론 화면 (드론에 넣을 YOLO11-pose 모델)."""
from pathlib import Path

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .. import yolo
from .common import RunPanel, note, open_folder, text_label

STEPS = ("<b>사용 순서</b><ol>"
         "<li><b>YOLO 데이터셋 준비</b>: 4번 탭에서 내보내기, 또는 받은 데이터셋의 data.yaml</li>"
         "<li><b>YOLO 학습</b>: 학습 데이터 선택, 모델 크기, epoch, 시작</li>"
         "<li><b>영상 추론</b>: 학습한 모델, 영상 선택, 시작 → 박스, 키포인트를 그린 결과 영상</li></ol>")


def spin(value, lo, hi, step=1):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    return s


class YoloTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.video = None
        self.last_video_out = None
        lay = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        lay.addLayout(left, 1)
        lay.addLayout(right, 1)
        left.addWidget(text_label(STEPS))

        # ---- 1. YOLO 학습 ----
        g = QGroupBox("1. YOLO11-pose 학습")
        f = QFormLayout(g)
        drow = QHBoxLayout()
        self.data = QComboBox()
        b_data = QPushButton("다른 data.yaml 선택...")
        b_data.clicked.connect(self.pick_data)
        drow.addWidget(self.data, 1)
        drow.addWidget(b_data)
        self.size = QComboBox()
        for k, text in yolo.MODEL_SIZES.items():
            self.size.addItem(text, k)
        self.epochs = spin(100, 1, 1000)
        self.imgsz = spin(640, 320, 1920, 32)
        self.batch = spin(8, 1, 64)
        self.device = QComboBox()
        for key, text in (("auto", "auto (GPU 있으면 GPU)"), ("cuda", "GPU (cuda)"), ("cpu", "CPU")):
            self.device.addItem(text, key)
        self.btn_train = QPushButton("YOLO 학습 시작")
        self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self.train)
        f.addRow("학습 데이터", drow)
        f.addRow("모델 크기", self.size)
        f.addRow("epoch", self.epochs)
        f.addRow("imgsz", self.imgsz)
        f.addRow("batch", self.batch)
        f.addRow("device", self.device)
        f.addRow(note("<b>학습 데이터</b>: 4번 탭에서 내보낸 폴더 (최신 순), 또는 받은 데이터셋의 data.yaml<br>"
                      "<b>모델 크기</b>: n = yolo11n-pose (드론 실시간 추천). s, m 은 처음 쓸 때 시작 가중치 자동 다운로드<br>"
                      "<b>imgsz</b>: 학습, 추론 이미지 크기 (기본 640)<br>"
                      "<b>batch</b>: GPU 6GB 기준 yolo11n 은 batch 8, 메모리 부족 오류 시 batch 4<br>"
                      "<b>결과</b>: &lt;데이터셋&gt;/yolo/&lt;날짜&gt;/weights/best.pt (가장 좋은 epoch)"))
        f.addRow(self.btn_train)
        left.addWidget(g)

        # ---- 2. 영상 추론 ----
        g = QGroupBox("2. 영상 추론")
        f = QFormLayout(g)
        mrow = QHBoxLayout()
        self.model = QComboBox()
        b_model = QPushButton("다른 가중치(.pt) 선택...")
        b_model.clicked.connect(self.pick_weights)
        mrow.addWidget(self.model, 1)
        mrow.addWidget(b_model)
        vrow = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setReadOnly(True)
        b_vid = QPushButton("영상 선택...")
        b_vid.clicked.connect(self.pick_video)
        vrow.addWidget(self.video_edit, 1)
        vrow.addWidget(b_vid)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.btn_pred = QPushButton("영상 추론 시작")
        self.btn_pred.setObjectName("primary")
        self.btn_pred.clicked.connect(self.predict)
        self.btn_play = QPushButton("결과 영상 열기")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(lambda: self.last_video_out and open_folder(self.last_video_out))
        prow = QHBoxLayout()
        prow.addWidget(self.btn_pred, 1)
        prow.addWidget(self.btn_play)
        f.addRow("모델", mrow)
        f.addRow("영상", vrow)
        f.addRow("conf threshold", self.conf)
        f.addRow(note("<b>conf threshold</b>: 점수가 이 값 이상인 객체만 표시 (놓치면 낮춤, 엉뚱한 탐지가 많으면 높임)<br>"
                      "<b>결과</b>: &lt;데이터셋&gt;/yolo_predict/ 에 결과 영상(.mp4)과 프레임별 결과(.json)"))
        f.addRow(prow)
        left.addWidget(g)
        left.addStretch(1)

        # ---- 오른쪽: 학습한 모델 표, 진행 상황 ----
        g = QGroupBox("학습한 YOLO 모델")
        gv = QVBoxLayout(g)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["이름", "모델", "클래스", "epoch (최고)", "박스 mAP50", "키포인트 mAP50",
                                              "batch / 시간"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        b_open = QPushButton("YOLO 폴더 열기")
        b_open.clicked.connect(lambda: self.main.project and open_folder(self.out_root()))
        gv.addWidget(self.table)
        gv.addWidget(note("<b>박스 mAP50</b>: 박스 위치 정확도 / <b>키포인트 mAP50</b>: 키포인트 위치 정확도 "
                          "(둘 다 1에 가까울수록 좋음, 검증(val) 이미지 기준)"))
        gv.addWidget(b_open)
        right.addWidget(g)
        self.run = RunPanel()
        right.addWidget(self.run, 1)

    # ---------- 경로 ----------
    def out_root(self):
        return self.main.project.root / "yolo"

    # ---------- 화면 갱신 ----------
    def refresh(self):
        p = self.main.project
        if p is None:
            return
        keep = self.data.currentData()
        self.data.clear()
        exports = sorted((d for d in p.exports_dir.glob("*") if (d / "data.yaml").exists()), reverse=True) \
            if p.exports_dir.exists() else []
        for d in exports:
            self.data.addItem(f"내보내기 {d.name}", str(d / "data.yaml"))
        if keep and self.data.findData(keep) < 0:
            self.data.addItem(keep, keep)
        if keep:
            self.data.setCurrentIndex(self.data.findData(keep))
        if self.data.count() == 0:
            self.data.addItem("없음 (4번 탭에서 내보내기 또는 data.yaml 선택)", None)

        keep_m = self.model.currentData()
        self.model.clear()
        models = yolo.list_models(self.out_root())
        for m in models:
            self.model.addItem(f"{m['name']}  (yolo11{m['size']}-pose, 키포인트 mAP50 {m['pose_mAP50']})", m["weights"])
        if keep_m and self.model.findData(keep_m) < 0:
            self.model.addItem(keep_m, keep_m)
        if keep_m:
            self.model.setCurrentIndex(self.model.findData(keep_m))
        if self.model.count() == 0:
            self.model.addItem("없음 (YOLO 학습 또는 가중치 선택)", None)

        self.table.setRowCount(len(models))
        for r, m in enumerate(models):
            names = ", ".join(f"{k}: {v}" for k, v in (m.get("names") or {}).items())
            vals = [m["name"], f"yolo11{m['size']}-pose", names, f"{m['epochs']} (최고 {m['best_epoch']})",
                    str(m["box_mAP50"]), str(m["pose_mAP50"]), f"{m['batch']} / {m['minutes']}분"]
            for c, t in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(t))
        self.table.resizeColumnsToContents()

    # ---------- 선택 ----------
    def pick_data(self):
        file, _ = QFileDialog.getOpenFileName(self, "YOLO 데이터셋의 data.yaml 선택", "", "data.yaml (*.yaml *.yml)")
        if file:
            self.data.addItem(file, file)
            self.data.setCurrentIndex(self.data.count() - 1)

    def pick_weights(self):
        file, _ = QFileDialog.getOpenFileName(self, "YOLO 가중치 선택 (best.pt)", "", "YOLO 가중치 (*.pt)")
        if file:
            self.model.addItem(file, file)
            self.model.setCurrentIndex(self.model.count() - 1)

    def pick_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "추론할 영상 선택", "", "영상 (*.mp4 *.mov *.avi *.mkv *.ts)")
        if file:
            self.video = file
            self.video_edit.setText(Path(file).name)

    # ---------- 실행 ----------
    def _busy(self, on):
        self.btn_train.setEnabled(not on)
        self.btn_pred.setEnabled(not on)

    def train(self):
        p, data = self.main.project, self.data.currentData()
        if p is None or self.run.busy():
            return
        if not data:
            QMessageBox.information(self, "학습 데이터 없음", "학습 데이터 없음: 4번 탭에서 내보내기 또는 data.yaml 선택 필요")
            return
        args = dict(size=self.size.currentData(), epochs=self.epochs.value(), imgsz=self.imgsz.value(),
                    batch=self.batch.value(), device=self.device.currentData())
        out_root = self.out_root()
        self._busy(True)
        self.run.text.clear()
        self.run.start(lambda progress, log, should_stop: yolo.train(data, out_root, progress=progress, log=log,
                                                                    should_stop=should_stop, **args),
                       lambda _: (self._busy(False), self.refresh()))
        self.run.worker.failed.connect(lambda _: self._busy(False))

    def predict(self):
        p, weights = self.main.project, self.model.currentData()
        if p is None or self.run.busy():
            return
        if not weights or not self.video:
            QMessageBox.information(self, "선택 필요", "모델과 영상 선택 필요")
            return
        out = p.root / "yolo_predict" / f"{Path(self.video).stem}_{Path(weights).parent.parent.name}.mp4"
        args = dict(conf=self.conf.value(), imgsz=self.imgsz.value(), device=self.device.currentData())
        video = self.video
        self._busy(True)
        self.run.text.clear()

        def done(path):
            self._busy(False)
            self.last_video_out = path
            self.btn_play.setEnabled(True)

        self.run.start(lambda progress, log, should_stop: yolo.predict_video(weights, video, out, progress=progress,
                                                                            log=log, should_stop=should_stop, **args),
                       done)
        self.run.worker.failed.connect(lambda _: self._busy(False))

    def busy(self):
        return self.run.busy()
