"""4. 내보내기 화면 (YOLO Pose 데이터셋 + 증강)."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from .. import config as C
from ..export import DEFAULT_CLASS_IDS, check_class_ids, export
from .augment_box import AugmentBox
from .common import RunPanel, note, open_folder, text_label


class ExportTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.last_out = None
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        lay.addLayout(left, 1)

        g = QGroupBox("YOLO Pose 데이터셋으로 내보내기")
        f = QFormLayout(g)
        self.data = QLabel("-")
        self.val = QDoubleSpinBox()
        self.val.setRange(0.05, 0.5)
        self.val.setSingleStep(0.05)
        self.val.setValue(0.15)
        self.bg = QSpinBox()
        self.bg.setRange(0, 50)
        self.bg.setValue(10)
        self.bg.setSuffix(" %")
        # 클래스 번호: 클래스마다 직접 지정 (0, 1, 2 또는 내보내지 않음)
        self.id_boxes = []
        for i, c in enumerate(C.CLASSES):
            cb = QComboBox()
            for n in range(len(C.CLASSES)):
                cb.addItem(str(n), n)
            cb.addItem("내보내지 않음", None)
            cb.setCurrentIndex(DEFAULT_CLASS_IDS[i])
            cb.currentIndexChanged.connect(self.update_ids)
            self.id_boxes.append(cb)
        self.id_result = text_label("")
        self.btn = QPushButton("내보내기 시작")
        self.btn.setObjectName("primary")
        self.btn.clicked.connect(self.start)
        b_open = QPushButton("결과 폴더 열기")
        b_open.clicked.connect(self.open_out)
        f.addRow("내보낼 데이터", self.data)
        f.addRow("검증(val) 비율", self.val)
        f.addRow("배경 이미지 최대 비율", self.bg)
        self.id_labels = []
        for c, cb in zip(C.CLASSES, self.id_boxes):
            lb = QLabel(f"클래스 번호: {c['name']} ({c['key']})")
            self.id_labels.append(lb)
            f.addRow(lb, cb)
        f.addRow("", self.id_result)
        f.addRow(note("<b>클래스 번호</b>: 클래스마다 직접 지정. 팀원 학습 모델과 반드시 동일<br>"
                      "&nbsp;- 규칙: 번호 겹침 불가, 0부터 빈 번호 없이 (예: 도로표지판만 내보내면 0)<br>"
                      "&nbsp;- 내보내지 않음: 그 클래스 라벨은 빼고, 그 클래스만 있는 이미지는 제외<br>"
                      "<b>라벨 형식</b>: 기존 툴과 동일 (좌표 계산, 숫자 모양, kpt_shape [4, 3], flip_idx, kpt_names)<br>"
                      "<b>원본 이미지</b>: 다시 압축하지 않고 그대로 복사"))
        f.addRow(note("<b>내보낼 이미지</b>: 사람 검수 완료 이미지만<br>"
                      "<b>증강</b>: 아래 '증강' 칸에서 장수, 세기, 옵션 지정 (train 이미지만, val 은 원본)<br>"
                      "<b>배경 이미지 최대 비율</b>: 시설물 없는 이미지를 전체의 이 비율까지만 (오탐 감소용)<br>"
                      "<b>결과</b>: exports/&lt;날짜&gt;/ 안에 images, labels, data.yaml (kpt_shape [4, 3]), report.json"))
        self.aug_box = AugmentBox(main, "증강 (train 이미지만)", copies=2)
        f.addRow(self.aug_box)
        row = QHBoxLayout()
        row.addWidget(self.btn)
        row.addWidget(b_open)
        f.addRow(row)
        left.addWidget(g)
        self.run = RunPanel()
        left.addWidget(self.run, 1)
        info = QLabel("<b>학습 예시</b> (Ultralytics 설치된 환경에서)<br>"
                      "<code>yolo pose train data=&lt;결과폴더&gt;/data.yaml model=yolo11n-pose.pt</code>")
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        left.addWidget(info)

    def refresh(self):
        p = self.main.project
        if p is None:
            return
        s = p.summary()
        self.data.setText(f"사람 검수 완료 {s['human']}장 (시설물 있음 {s['human_with_objects']}장, "
                          f"없음 {s['human_background']}장)")
        for c, lb in zip(C.CLASSES, self.id_labels):           # 클래스 옆에 검수 완료 라벨 수 표시
            lb.setText(f"클래스 번호: {c['name']} ({c['key']}, 라벨 {s['objects'][c['name']]}개)")
        self.update_ids()

    def class_ids(self):
        return {i: cb.currentData() for i, cb in enumerate(self.id_boxes)}

    def update_ids(self):
        """지정한 번호 확인 결과와 data.yaml 에 들어갈 내용 미리 보기."""
        ids = self.class_ids()
        problem = check_class_ids(ids)
        if problem:
            self.id_result.setText(f"<b style='color:#d32f2f'>확인 필요: {problem}</b>")
        else:
            used = sorted((v, C.CLASS_KEYS[k]) for k, v in ids.items() if v is not None)
            self.id_result.setText("<b style='color:#2e7d32'>data.yaml</b>: nc = " + str(len(used)) + ", names = "
                                   + ", ".join(f"{v}: {k}" for v, k in used))
        self.btn.setEnabled(problem is None and not self.run.busy())

    def start(self):
        p = self.main.project
        if p is None or self.run.busy():
            return
        self.main.label_tab.save_current()
        val, aug, bg, ids = self.val.value(), self.aug_box.copies.value(), self.bg.value(), self.class_ids()
        aug_set = self.aug_box.settings()
        self.btn.setEnabled(False)
        self.run.text.clear()

        def done(out):
            self.btn.setEnabled(True)
            self.last_out = out

        self.run.start(lambda progress, log, should_stop: export(p, val, aug, bg, ids, progress, log, aug_set), done)
        self.run.worker.failed.connect(lambda _: self.btn.setEnabled(True))

    def open_out(self):
        p = self.main.project
        if self.last_out:
            open_folder(self.last_out)
        elif p:
            open_folder(p.exports_dir)
