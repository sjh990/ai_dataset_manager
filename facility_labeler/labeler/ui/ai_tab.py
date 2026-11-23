"""3. AI 학습 화면. (여러 장 AI 자동 라벨링은 2. 라벨링 화면에 있음)"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .. import assist
from ..project import Project
from .augment_box import AugmentBox
from .common import RunPanel, note, open_folder, text_label
from .dataset_tab import find_datasets

STEPS = ("<b>사용 순서</b><ol>"
         "<li><b>직접 라벨링</b>: 라벨링 화면에서 이미지 몇십 장 라벨링, 확정</li>"
         "<li><b>AI 학습</b>: 사람 검수 완료 라벨로 박스 AI, 키포인트 AI 학습 (둘 다 또는 하나만)</li>"
         "<li><b>AI 자동 라벨링</b>: <b>2. 라벨링 탭</b> 'AI 자동 라벨링'에서 미작업 이미지 여러 장 한 번에 (점선 = 확인 필요)</li>"
         "<li><b>AI 예측 확인</b>: 라벨링 화면 'AI 예측' 보기, 점수 낮은 것부터 수정, 확정</li>"
         "<li><b>재학습</b>: 확정이 늘면 다시 학습 (반복할수록 수정할 것 감소)</li></ol>")


def spin(value, lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def epoch_text(m, part):
    """모델 표의 epoch 칸: '50 (최고 37)' / '기존 모델: 이름' / '-'."""
    source = m.get("box_from" if part == "box" else "keypoint_from")
    epochs = m.get("box_epochs" if part == "box" else "kpt_epochs", 0)
    if source not in (None, "새로 학습"):
        return "-" if source == "없음" else f"기존: {source}"
    if not epochs:
        return "-"
    best = m.get("box_detail" if part == "box" else "kpt_detail", {}).get("best_epoch")
    return f"{epochs} (최고 {best})" if best else str(epochs)


def fill_reuse(combo, models, none_text, text_fn):
    """'함께 쓸 AI' 목록 채우기. 고른 것은 유지하고, 처음이면 가장 최근 모델을 기본으로."""
    before = combo.currentText()
    combo.clear()
    combo.addItem(none_text, None)
    for m in models:
        combo.addItem(text_fn(m), m)
    i = combo.findText(before)
    combo.setCurrentIndex(i if i >= 0 else min(1, combo.count() - 1))


def labeled(text, widget):
    """'이름 + 입력칸' 한 줄 (줄 전체를 숨기거나 보이기 쉽게 묶음)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lb = QLabel(text)
    lb.setMinimumWidth(120)
    lay.addWidget(lb)
    lay.addWidget(widget, 1)
    return w


class AiTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        lay.addLayout(left, 1)
        lay.addLayout(right, 1)

        steps = QLabel(STEPS)
        steps.setTextFormat(Qt.RichText)
        steps.setWordWrap(True)
        left.addWidget(steps)

        g = QGroupBox("AI 학습")
        gv = QVBoxLayout(g)
        f = QFormLayout()
        self.data = text_label("-")
        self.ds_list = QListWidget()             # 함께 학습할 데이터셋 (체크)
        self.ds_list.setMaximumHeight(110)
        self.ds_list.itemChanged.connect(lambda _: self.update_data())
        self.mode = QComboBox()
        for key, text in assist.TRAIN_MODES.items():
            self.mode.addItem(text, key)
        self.mode.currentIndexChanged.connect(self.update_mode)
        f.addRow("학습할 데이터셋", self.ds_list)
        f.addRow("학습 데이터", self.data)
        f.addRow("무엇을 학습할까요", self.mode)
        gv.addLayout(f)

        # 박스 AI 칸
        box_g = QGroupBox("박스 AI  (내가 그린 박스로 학습 / RF-DETR Nano)")
        bf = QFormLayout(box_g)
        self.box_state = QLabel("")
        self.box_epochs = spin(50, 1, 500)
        self.box_reuse = QComboBox()
        bf.addRow(self.box_state)
        self.box_epochs_row = labeled("epoch", self.box_epochs)
        self.box_reuse_row = labeled("함께 쓸 박스 AI", self.box_reuse)
        bf.addRow(self.box_epochs_row)
        bf.addRow(self.box_reuse_row)
        gv.addWidget(box_g)

        # 키포인트 AI 칸
        kpt_g = QGroupBox("키포인트 AI  (내가 찍은 키포인트로 학습 / ViTPose)")
        kf = QFormLayout(kpt_g)
        self.kpt_state = QLabel("")
        self.kpt_epochs = spin(50, 1, 500)
        self.kpt_reuse = QComboBox()
        kf.addRow(self.kpt_state)
        self.kpt_epochs_row = labeled("epoch", self.kpt_epochs)
        self.kpt_reuse_row = labeled("함께 쓸 키포인트 AI", self.kpt_reuse)
        kf.addRow(self.kpt_epochs_row)
        kf.addRow(self.kpt_reuse_row)
        gv.addWidget(kpt_g)

        # 공통 설정
        f2 = QFormLayout()
        self.batch = spin(4, 1, 16)        # 6GB 노트북 측정 결과: 4가 속도와 메모리 여유의 균형이 가장 좋음
        self.device = QComboBox()
        for key, text in (("auto", "auto (GPU 있으면 GPU)"), ("cuda", "GPU (cuda)"), ("cpu", "CPU")):
            self.device.addItem(text, key)
        self.auto_after = QCheckBox("학습이 끝나면 미작업 이미지 자동 라벨링까지 하기")
        self.auto_after.setChecked(True)
        f2.addRow("batch", self.batch)
        f2.addRow("device", self.device)
        f2.addRow(self.auto_after)
        gv.addLayout(f2)
        self.aug = AugmentBox(main, "증강 (학습할 때)", copies=1)
        gv.addWidget(self.aug)
        gv.addWidget(note("<b>학습하지 않는 쪽</b>: '함께 쓸 AI'로 고른 기존 모델을 붙여 하나의 모델로 저장 "
                          "(예: 키포인트 AI가 이미 잘 맞으면 '박스 AI만 학습'으로 시간 절약)<br>"
                          "<b>학습 데이터</b>: '사람 검수 완료' 이미지만 사용. 배경(시설물 없음)은 시설물 이미지 수만큼만<br>"
                          "<b>batch 값</b>: 박스 AI = batch 값 그대로, 키포인트 AI = batch 값 × 4. "
                          "GPU 6GB 기준 batch 4 권장, 메모리 부족 오류 시 batch 2"))
        self.btn_train = QPushButton("AI 학습 시작")
        self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self.train)
        gv.addWidget(self.btn_train)
        left.addWidget(g)

        left.addStretch(1)

        g = QGroupBox("학습한 모델")
        gv = QVBoxLayout(g)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["이름", "종류", "학습 이미지", "박스 mAP50", "키포인트 PCK",
                                              "박스 epoch", "키포인트 epoch", "batch / 시간"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        b_open = QPushButton("모델 폴더 열기")
        b_open.clicked.connect(lambda: self.main.project and open_folder(self.main.project.models_dir))
        b_import = QPushButton("다른 데이터셋 모델 가져오기...")
        b_import.clicked.connect(lambda: self.import_model())
        row = QHBoxLayout()
        row.addWidget(b_import)
        row.addWidget(b_open)
        gv.addWidget(self.table)
        gv.addWidget(note("<b>박스 mAP50</b>: 박스 위치 정확도 (1에 가까울수록 좋음)<br>"
                          "<b>키포인트 PCK</b>: 정답에서 박스 크기 10% 안에 들어온 키포인트 비율 (1에 가까울수록 좋음)<br>"
                          "<b>다른 데이터셋 모델 가져오기</b>: 다른 데이터셋의 models 안 날짜 폴더 선택 "
                          "(예: gumi_DJI_0342 \\ models \\ 20261121_113649), 이 데이터셋으로 복사 (원본은 그대로)"))
        gv.addLayout(row)
        right.addWidget(g)
        self.run = RunPanel()
        right.addWidget(self.run, 1)

    def refresh(self):
        p = self.main.project
        if p is None:
            return
        self.fill_datasets()
        self.update_data()
        models = assist.list_models(p)
        # 학습하지 않는 쪽에 붙일 기존 모델 목록 (가장 최근 것이 기본)
        fill_reuse(self.box_reuse, [m for m in models if assist.has_box(m)], "없음 (박스는 직접 그림)",
                   lambda m: f"{m['name']}  (mAP50 {m['box_val_mAP50']})")
        fill_reuse(self.kpt_reuse, [m for m in models if assist.has_keypoint(m)], "없음 (키포인트는 직접 찍음)",
                   lambda m: f"{m['name']}  (PCK {m['kpt_val_PCK']})")
        self.update_mode()
        self.table.setRowCount(len(models))
        for r, m in enumerate(models):
            box = m["box_val_mAP50"] if assist.has_box(m) else "-"
            kpt = m["kpt_val_PCK"] if assist.has_keypoint(m) else "-"
            vals = [m["name"], assist.model_kind(m), f"{m['images']}장 (시설물 {m['images_with_objects']} + 배경 {m['images'] - m['images_with_objects']})"
                    + (f", 데이터셋 {len(m['datasets'])}개" if len(m.get("datasets", [])) > 1 else ""), str(box), str(kpt),
                    epoch_text(m, "box"), epoch_text(m, "kpt"), f"{m.get('batch', '-')} / {m['minutes']}분"]
            for c, t in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(t))
        self.table.resizeColumnsToContents()

    def import_model(self, parent=None):
        """다른 데이터셋 모델 가져오기 (3번 탭, 2번 라벨링 탭 버튼 공용). 가져온 모델 정보를 돌려줌."""
        parent = parent or self
        p = self.main.project
        if p is None:
            QMessageBox.information(parent, "데이터셋 필요", "데이터셋을 먼저 열기 (1. 데이터셋 / 영상 탭)")
            return None
        start = str(Path(p.root).parent)          # datasets 폴더
        folder = QFileDialog.getExistingDirectory(
            parent, "가져올 모델 폴더 선택 (다른 데이터셋\\models\\날짜 폴더)", start)
        if not folder:
            return None
        try:
            info = assist.import_model(p, folder)
        except Exception as e:
            QMessageBox.warning(parent, "가져올 수 없음", str(e))
            return None
        QMessageBox.information(parent, "가져오기 완료",
                                f"{info['name']}  ({assist.model_kind(info)})\n"
                                f"박스 mAP50 {info['box_val_mAP50']}, 키포인트 PCK {info['kpt_val_PCK']}\n\n"
                                "사용 위치: 2. 라벨링 탭 'AI 자동 라벨링'")
        self.main.refresh_all()
        return info

    def update_mode(self):
        """'무엇을 학습할까요' 선택에 따라 박스/키포인트 칸 내용을 바꿉니다."""
        mode = self.mode.currentData()
        train_box, train_kpt = mode in ("both", "box"), mode in ("both", "keypoint")
        for train, state, epochs_row, reuse_row in ((train_box, self.box_state, self.box_epochs_row, self.box_reuse_row),
                                                    (train_kpt, self.kpt_state, self.kpt_epochs_row, self.kpt_reuse_row)):
            state.setText("<b style='color:#1976d2'>새로 학습</b>" if train else
                          "<b style='color:#666'>학습 안 함</b> (아래 모델을 함께 사용)")
            epochs_row.setVisible(train)
            reuse_row.setVisible(not train)

    def _busy(self, on):
        self.btn_train.setEnabled(not on)

    # ---------- 함께 학습할 데이터셋 ----------
    def fill_datasets(self):
        """datasets 폴더의 데이터셋 목록 (체크한 것 유지). 지금 연 데이터셋은 항상 포함."""
        p = self.main.project
        checked = {self.ds_list.item(i).data(Qt.UserRole) for i in range(self.ds_list.count())
                   if self.ds_list.item(i).checkState() == Qt.Checked}
        self.ds_list.blockSignals(True)
        self.ds_list.clear()
        self.ds_info = {}
        for d in find_datasets():
            pr = Project(d)
            s = pr.summary()
            self.ds_info[str(d)] = (pr, s)
            current = d.resolve() == p.root.resolve()
            it = QListWidgetItem(f"{pr.info['name']}  (사람 검수 {s['human']}장, 시설물 {s['human_with_objects']}장)"
                                 + ("  [지금 연 데이터셋, 항상 포함]" if current else ""))
            it.setData(Qt.UserRole, str(d))
            if current:
                it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsEnabled)
                it.setCheckState(Qt.Checked)
            else:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if str(d) in checked else Qt.Unchecked)
            self.ds_list.addItem(it)
        self.ds_list.blockSignals(False)

    def selected_extra(self):
        """체크한 다른 데이터셋들 (지금 연 데이터셋 제외)."""
        p = self.main.project
        out = []
        for i in range(self.ds_list.count()):
            it = self.ds_list.item(i)
            path = it.data(Qt.UserRole)
            if it.checkState() == Qt.Checked and Path(path).resolve() != p.root.resolve():
                out.append(self.ds_info[path][0])
        return out

    def update_data(self):
        """체크한 데이터셋 합계로 학습 데이터 표시."""
        p = self.main.project
        if p is None:
            return
        sums = [self.ds_info[str(pr.root)][1] for pr in self.selected_extra()]
        s0 = p.summary()
        obj = s0["human_with_objects"] + sum(s["human_with_objects"] for s in sums)
        bgs = s0["human_background"] + sum(s["human_background"] for s in sums)
        bg = min(bgs, max(obj, 10))   # assist.training_items 와 같은 규칙
        n = 1 + len(sums)
        self.data.setText(f"데이터셋 {n}개, 사람 검수 완료: 시설물 있음 {obj}장 / 시설물 없음 {bgs}장<br>"
                          f"학습에 쓰는 것: 시설물 {obj}장 + 배경 {bg}장 "
                          "<br><span style='color:#666'>배경을 시설물 수만큼만 쓰는 이유: 배경이 너무 많을 때 '아무것도 없음' 쪽으로 치우친 학습 방지, "
                          "비슷한 장면 반복으로 늘어나는 학습 시간 절약. 키포인트 AI는 시설물 이미지만 사용 (배경에는 찍을 키포인트 없음)<br>"
                          "여러 영상을 함께 학습하면 처음 보는 영상에서도 잘 맞음 (같은 프레임이 겹치면 한 번만 사용)</span>")

    def train(self):
        p = self.main.project
        if p is None or self.run.busy():
            return
        self.main.label_tab.save_current()
        args = dict(mode=self.mode.currentData(), box_epochs=self.box_epochs.value(),
                    kpt_epochs=self.kpt_epochs.value(), batch=self.batch.value(), device=self.device.currentData(),
                    reuse_box=self.box_reuse.currentData(), reuse_keypoint=self.kpt_reuse.currentData(),
                    extra_projects=self.selected_extra(), aug_copies=self.aug.copies.value(),
                    aug_settings=self.aug.settings())
        auto_after, thr = self.auto_after.isChecked(), self.main.label_tab.threshold.value()   # 박스 threshold 는 라벨링 탭 값
        self._busy(True)
        self.run.text.clear()

        def job(progress, log, should_stop):
            info = assist.train_all(p, progress=progress, log=log, should_stop=should_stop, **args)
            if auto_after and not should_stop():
                log("\n이어서 미작업 이미지 AI 자동 라벨링 시작")
                assist.prelabel(p, info, "todo", thr, args["device"], progress, log, should_stop)
            return info

        self.run.start(job, lambda _: self._done())
        self.run.worker.failed.connect(lambda _: self._busy(False))

    def _done(self):
        self._busy(False)
        self.main.refresh_all()
