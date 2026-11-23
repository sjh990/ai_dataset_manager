"""2. 라벨링 화면."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget)

from .. import assist
from .. import config as C
from ..frames import read_image
from .canvas import Canvas, score_text
from .common import Worker, note, text_label

def object_counts(objects):
    """객체 목록 → 클래스별 개수 글자. 예: '객체 2 (표지판 1, 분리대 1)'."""
    if not objects:
        return "객체 0"
    n = [0] * len(C.CLASSES)
    for o in objects:
        n[o["cls"]] += 1
    parts = ", ".join(f"{c['short']} {k}" for c, k in zip(C.CLASSES, n) if k)
    return f"객체 {len(objects)} ({parts})"


FILTERS = [("all", "전체 이미지"), ("todo", "미작업"), ("ai", "AI 예측 확인 필요 (자신 없는 것부터)"),
           ("human", "사람 검수 완료"), ("objects", "시설물 있는 이미지")] + \
          [(f"cls{i}", f"{c['name']} 있는 이미지") for i, c in enumerate(C.CLASSES)]   # 클래스별 보기

# 단축키 표: (구분, [(키, 기능), ...])
SHORTCUTS = [
    ("그리기", [("B", "박스 그리기"), ("V", "선택 / 수정"), ("1 2 3", "클래스 선택"),
              ("S", "키포인트 건너뛰기 (안 보임)"), ("R", "키포인트 다시 찍기"), ("L", "키포인트 2, 3 일직선 정렬"), ("Delete", "선택 객체 삭제"), ("Shift+Delete", "모든 객체 삭제"),
              ("Ctrl+Z", "되돌리기")]),
    ("확정 / 이동", [("Space", "확정하고 다음"), ("N", "시설물 없음으로 확정"), ("U", "확정 취소"),
                 ("A", "이전 이미지"),
                 ("D", "다음 이미지")]),
    ("화면", [("휠", "확대 / 축소"), ("우클릭", "누른 채 끌어서 화면 이동"), ("F", "화면 맞춤"), ("F5", "새로고침 (코드 변경 반영, 위치 유지)")]),
]


def shortcuts_html():
    key_style = ("background:#eeeeee; border:1px solid #bbbbbb; font-family:Consolas; "
                 "font-weight:bold; padding:2px 6px;")
    rows = ""
    for group, items in SHORTCUTS:
        rows += f"<tr><td colspan=2 style='color:#1976d2; font-weight:bold; padding-top:6px;'>{group}</td></tr>"
        for key, action in items:
            rows += (f"<tr><td align=center width=80 style='{key_style}'>{key}</td>"
                     f"<td style='padding-left:8px;'>{action}</td></tr>")
    return f"<table cellspacing=3 cellpadding=2 width='100%'>{rows}</table>"


class LabelTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.images = []        # 목록에 보이는 이미지 경로
        self.index = -1
        self.dirty = False

        # 왼쪽: 이미지 목록
        left = QWidget()
        lv = QVBoxLayout(left)
        self.filter = QComboBox()
        for key, text in FILTERS:
            self.filter.addItem(text, key)
        self.filter.currentIndexChanged.connect(self.reload_list)
        self.count = QLabel("")
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.open_index)
        lv.addWidget(QLabel("보기"))
        lv.addWidget(self.filter)
        lv.addWidget(self.count)
        lv.addWidget(self.list, 1)

        # 가운데: 도구 + 그림판
        center = QWidget()
        cv = QVBoxLayout(center)
        tools = QHBoxLayout()
        self.class_group = QButtonGroup(self)
        for i, c in enumerate(C.CLASSES):
            b = QPushButton(f"{i + 1}. {c['name']}")
            b.setCheckable(True)
            b.setStyleSheet(f"QPushButton {{ color: {c['color']}; font-weight: bold; }}"
                            f"QPushButton:checked {{ background: {c['color']}; color: white; border-color: #333; }}")
            self.class_group.addButton(b, i)
            tools.addWidget(b)
        self.class_group.button(0).setChecked(True)
        self.class_group.idClicked.connect(self.set_class)
        tools.addSpacing(20)
        self.btn_box = QPushButton("박스 그리기 (B)")
        self.btn_edit = QPushButton("선택/수정 (V)")
        for b in (self.btn_box, self.btn_edit):
            b.setCheckable(True)
            tools.addWidget(b)
        self.btn_box.clicked.connect(lambda: self.canvas.set_mode("box"))
        self.btn_edit.clicked.connect(lambda: self.canvas.set_mode("edit"))
        tools.addStretch(1)
        self.canvas = Canvas()
        self.canvas.changed.connect(self.on_changed)
        self.canvas.hint_changed.connect(self.on_hint)
        self.hint = QLabel("")
        self.hint.setStyleSheet("background:#fff8e1; padding:6px; font-weight:bold;")
        cv.addLayout(tools)
        cv.addWidget(self.hint)
        cv.addWidget(self.canvas, 1)

        # 오른쪽: 현재 이미지 정보 + 기준
        right = QWidget()
        rv = QVBoxLayout(right)
        self.g1 = g1 = QGroupBox("현재 이미지")
        g1v = QVBoxLayout(g1)
        self.name = text_label("-", rich=False)
        self.status = text_label("-")
        self.objects = QListWidget()
        self.objects.currentRowChanged.connect(lambda r: r >= 0 and self.canvas.select(r))
        b_del = QPushButton("선택 객체 삭제 (Delete)")
        b_del.clicked.connect(self.canvas.delete_selected)
        b_del_all = QPushButton("모든 객체 삭제 (Shift+Delete)")
        b_del_all.clicked.connect(self.delete_all)
        b_line = QPushButton("선택 객체 키포인트 2, 3 일직선 정렬 (L)")
        b_line.clicked.connect(self.straighten)
        b_redo = QPushButton("선택 객체 키포인트 다시 찍기 (R)")
        b_redo.clicked.connect(self.canvas.redo_points)
        self.btn_confirm = QPushButton("확정하고 다음 (Space)")
        self.btn_confirm.setObjectName("primary")
        self.btn_confirm.clicked.connect(self.confirm)
        b_bg = QPushButton("시설물 없음으로 확정 (N)")
        b_bg.clicked.connect(self.confirm_background)
        b_undo = QPushButton("확정 취소 (U)")
        b_undo.clicked.connect(self.unconfirm)
        nav = QHBoxLayout()
        b_prev, b_next = QPushButton("◀ 이전 (A)"), QPushButton("다음 (D) ▶")
        b_prev.clicked.connect(lambda: self.go(-1))
        b_next.clicked.connect(lambda: self.go(1))
        nav.addWidget(b_prev)
        nav.addWidget(b_next)
        for w in (self.name, self.status, QLabel("객체 목록"), self.objects, b_del, b_del_all, b_redo, b_line, self.btn_confirm, b_bg,
                  b_undo):
            g1v.addWidget(w)
        g1v.addLayout(nav)
        g1v.addWidget(note("<b>확정</b>: '사람 검수 완료'로 변경. AI 점선 라벨은 수정 후 확정<br>"
                           "<b>자동 저장</b>: 다른 이미지로 이동 시 수정 내용 저장 (상태 변경은 확정할 때만)"))
        self.guide = text_label("")
        g2 = QGroupBox("라벨링 기준")
        g2v = QVBoxLayout(g2)
        g2v.addWidget(self.guide)
        self.b_example = QPushButton("예시 이미지 보기")
        self.b_example.clicked.connect(self.show_examples)
        g2v.addWidget(self.b_example)
        g_ai = QGroupBox("AI 자동 라벨링")
        gav = QVBoxLayout(g_ai)
        self.ai_model = QComboBox()
        self.ai_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ai_model.setMinimumContentsLength(15)     # 모델 이름이 길어도 칸이 넓어지지 않게
        self.auto_points = QCheckBox("박스를 그리면 키포인트 AI 자동 라벨링")
        self.auto_points.setChecked(False)      # 기본: 꺼짐 (박스를 그려도 AI가 키포인트를 찍지 않음)
        self.b_points = b_pts = QPushButton("AI 자동 라벨링: 키포인트만 (현재 이미지에서 박스만 있는 객체)")
        b_pts.clicked.connect(self.ai_fill_points)
        self.b_boxes = QPushButton("AI 자동 라벨링: 박스 + 키포인트 (현재 이미지 1장)")
        self.b_boxes.clicked.connect(self.ai_add_boxes)
        self.ai_model.currentIndexChanged.connect(self._model_changed)
        # 여러 장 한 번에
        self.target = QComboBox()
        self.target.addItem("미작업 이미지 전체", "todo")
        self.target.addItem("미작업 + 기존 AI 예측 다시", "todo_ai")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.95)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.4)
        self.b_batch = QPushButton("AI 자동 라벨링: 여러 장 한 번에 시작")
        self.b_batch.setObjectName("primary")
        self.b_batch.clicked.connect(self.ai_batch)
        self.b_clear_ai = QPushButton("AI 예측 라벨 전체 지우기 (이 데이터셋)")
        self.b_clear_ai.clicked.connect(self.clear_ai_labels)
        self.batch_bar = QProgressBar()
        self.batch_bar.setFormat("%v / %m")
        self.batch_stop = QPushButton("중지")
        self.batch_stop.setEnabled(False)
        self.batch_stop.clicked.connect(lambda: self.worker and setattr(self.worker, "stop_requested", True))
        self.batch_msg = text_label("", rich=False)
        self.worker = None
        bar_row = QHBoxLayout()
        bar_row.addWidget(self.batch_bar, 1)
        bar_row.addWidget(self.batch_stop)
        batch_form = QFormLayout()
        batch_form.addRow("대상", self.target)
        batch_form.addRow("박스 threshold", self.threshold)

        b_import = QPushButton("다른 데이터셋 모델 가져오기...")
        b_import.clicked.connect(self.import_model)
        gav.addWidget(QLabel("사용할 모델"))
        gav.addWidget(self.ai_model)
        gav.addWidget(b_import)
        gav.addWidget(text_label("<b>현재 이미지 1장</b>"))
        for w in (self.auto_points, b_pts, self.b_boxes):
            gav.addWidget(w)
        gav.addWidget(text_label("<b>여러 장 한 번에</b>"))
        gav.addLayout(batch_form)
        gav.addWidget(self.b_batch)
        gav.addWidget(self.b_clear_ai)
        gav.addLayout(bar_row)
        gav.addWidget(self.batch_msg)
        gav.addWidget(note("<b>박스 threshold</b>: 박스 점수가 박스 threshold 이상인 박스만 생성 "
                           "(놓치는 박스가 많으면 낮춤, 엉뚱한 박스가 많으면 높임)<br>"
                           "<b>대상</b>: 사람 검수 완료 이미지는 변경 없음. 끝나면 보기 'AI 예측'으로 자동 전환<br>"
                           "<b>확인 필수</b>: AI가 만든 박스, 키포인트는 눈으로 확인, 틀린 것은 끌어서 수정 후 확정<br>"
                           "<b>모델 준비</b>: '다른 데이터셋 모델 가져오기' 또는 3번 탭에서 학습"))
        self.canvas.auto_points = self._auto_points
        self._img = (None, None)

        g3 = QGroupBox("단축키")
        QVBoxLayout(g3).addWidget(text_label(shortcuts_html()))
        rv.addWidget(g2)          # 라벨링 기준 (맨 위)
        rv.addWidget(g1)
        rv.addWidget(g_ai)
        rv.addWidget(g3)
        rv.addStretch(1)
        scroll = QScrollArea()          # 화면이 작아도 오른쪽 칸이 잘리지 않게 스크롤
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(right)

        split = QSplitter(Qt.Horizontal)
        for w in (left, center, scroll):
            split.addWidget(w)
        split.setSizes([220, 820, 460])     # 오른쪽 칸: 버튼 글자가 잘리지 않는 너비
        QHBoxLayout(self).addWidget(split)
        self._shortcuts()
        self.set_class(0)

    def _shortcuts(self):
        keys = {"B": lambda: self.canvas.set_mode("box"), "V": lambda: self.canvas.set_mode("edit"),
                "S": self.canvas.skip_point, "R": self.canvas.redo_points, "L": self.straighten, "Delete": self.canvas.delete_selected, "Shift+Delete": self.delete_all,
                "Space": self.confirm, "N": self.confirm_background, "U": self.unconfirm, "A": lambda: self.go(-1),
                "D": lambda: self.go(1), "Ctrl+Z": self.canvas.undo, "F": self.canvas.fit,
                "1": lambda: self.set_class(0), "2": lambda: self.set_class(1), "3": lambda: self.set_class(2)}
        for k, fn in keys.items():
            sc = QShortcut(QKeySequence(k), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda f=fn: None if self.batch_busy() else f())   # 여러 장 자동 라벨링 중에는 막음

    # ---------- 목록 ----------
    def reload_list(self):
        self.save_current()
        p = self.main.project
        self.list.blockSignals(True)
        self.list.clear()
        self.images, self.index = [], -1
        if p is None:
            self.list.blockSignals(False)
            return
        self.refresh_models()
        key = self.filter.currentData()
        rows = []
        for img in p.images():
            lab = p.load_label(img) or {"status": C.STATUS_TODO, "objects": [], "ai_score": None}
            st = lab["status"]
            if key in (C.STATUS_TODO, C.STATUS_AI, C.STATUS_HUMAN) and st != key:
                continue
            if key == "objects" and not lab["objects"]:
                continue
            if key.startswith("cls") and not any(o["cls"] == int(key[3:]) for o in lab["objects"]):
                continue
            rows.append((img, lab))
        if key == C.STATUS_AI:   # 자신 없는 것(점수 낮은 것)부터, AI가 아무것도 못 찾은 이미지는 맨 뒤
            rows.sort(key=lambda r: r[1].get("ai_score") if r[1].get("ai_score") is not None else 9)
        for img, lab in rows:
            it = QListWidgetItem(f"● {img.stem}  {object_counts(lab['objects'])}")
            it.setForeground(QColor(C.STATUS_COLOR[lab["status"]]))
            self.list.addItem(it)
            self.images.append(img)
        self.list.blockSignals(False)
        s = p.summary()
        view = [o for _, lab in rows for o in lab["objects"]]
        human = ", ".join(f"{c['short']} {s['objects'][c['name']]}" for c in C.CLASSES)
        self.count.setText(f"이 보기: {len(rows)}장, {object_counts(view)}<br>"
                           f"<b>사람 검수 {s['human']} / 전체 {s['total']}장</b><br>"
                           f"검수 완료 객체: {human}")
        self.count.setTextFormat(Qt.RichText)
        if self.images:
            self.list.setCurrentRow(0)
        else:
            self.canvas.set_image(None, None)
            self.canvas.update()
            self.name.setText("이 보기에 이미지 없음")
            self.status.setText("-")
            self.objects.clear()

    def select_image(self, name):
        """이미지 이름으로 목록에서 선택 (새로고침 후 위치 되돌리기)."""
        for i, img in enumerate(self.images):
            if img.name == name:
                self.list.setCurrentRow(i)
                return

    def batch_busy(self):
        return self.worker is not None and self.worker.isRunning()

    def open_index(self, row):
        self.save_current()
        if not (0 <= row < len(self.images)):
            return
        self.index = row
        p, img = self.main.project, self.images[row]
        pix = QPixmap(str(img))
        lab = p.load_label(img)
        if lab is None:
            from ..project import empty_label
            lab = empty_label(img.name, pix.width(), pix.height())
        self.canvas.set_image(pix, lab)
        self.dirty = False
        self.refresh_info()

    def save_current(self):
        if self.dirty and self.canvas.label is not None and 0 <= self.index < len(self.images):
            self.main.project.save_label(self.images[self.index], self.canvas.label)
            self.dirty = False

    def go(self, step):
        if self.batch_busy():
            return
        row = self.list.currentRow() + step
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)

    # ---------- 확정 ----------
    def confirm(self):
        if self.batch_busy():
            return
        lab = self.canvas.label
        if lab is None:
            return
        if not lab["objects"]:
            return self.confirm_background()
        missing = [f"{i + 1}번 {C.CLASS_NAMES[o['cls']]}: {sum(1 for k in o['kpts'] if k[2] <= 0)}개"
                   for i, o in enumerate(lab["objects"]) if any(k[2] <= 0 for k in o["kpts"])]
        if missing:
            ans = QMessageBox.question(self, "비어 있는 키포인트", "찍히지 않은 키포인트가 있는 객체:\n"
                                       + "\n".join(missing) + "\n\n가려짐 또는 화면 밖이라 비워 둔 것이면 '예' (그대로 확정)")
            if ans != QMessageBox.Yes:
                return
        self._set_status(background=False)

    def confirm_background(self):
        if self.batch_busy():
            return
        lab = self.canvas.label
        if lab is None:
            return
        if lab["objects"]:
            if QMessageBox.question(self, "시설물 없음", "객체를 모두 지우고 '시설물 없음'으로 확정할까요?") != QMessageBox.Yes:
                return
            lab["objects"] = []
        self._set_status(background=True)

    def _set_status(self, background):
        lab = self.canvas.label
        lab["status"] = C.STATUS_HUMAN
        lab["background"] = background
        lab["ai_score"] = None
        for o in lab["objects"]:                 # 사람이 확정하면 AI 점수는 지움
            for key in ("score", "box_score", "kpt_score"):
                o.pop(key, None)
        self.dirty = True
        self.save_current()
        it = self.list.item(self.index)
        it.setText(f"● {self.images[self.index].stem}  {object_counts(lab['objects'])}")
        it.setForeground(QColor(C.STATUS_COLOR[C.STATUS_HUMAN]))
        self.main.refresh_status()
        self.go(1)

    def unconfirm(self):
        if self.batch_busy():
            return
        """확정 취소: '사람 검수 완료'를 '미작업'으로 되돌립니다. 그려 둔 박스와 점은 그대로 남습니다."""
        lab = self.canvas.label
        if lab is None or lab["status"] != C.STATUS_HUMAN:
            self.hint.setText("확정 취소 불가: '사람 검수 완료' 이미지에서만 가능")
            return
        lab["status"] = C.STATUS_TODO
        lab["background"] = False
        self.dirty = True
        self.save_current()
        it = self.list.item(self.index)
        it.setForeground(QColor(C.STATUS_COLOR[C.STATUS_TODO]))
        self.refresh_info()
        self.main.refresh_status()
        self.hint.setText("확정 취소 완료 (미작업으로 변경). 박스, 키포인트는 그대로 유지, 수정 후 다시 확정")

    # ---------- AI 도움 ----------
    def refresh_models(self):
        p = self.main.project
        self.ai_model.blockSignals(True)
        self.ai_model.clear()
        for m in (assist.list_models(p) if p else []):
            kind = assist.model_kind(m)
            self.ai_model.addItem(f"{m['name']}  ({kind})", m)
        if self.ai_model.count() == 0:
            self.ai_model.addItem("모델 없음 (아래 '다른 데이터셋 모델 가져오기' 또는 3번 탭에서 학습)", None)
        self.ai_model.blockSignals(False)
        self._model_changed()

    def import_model(self):
        """다른 데이터셋에서 학습한 모델을 가져와 바로 선택."""
        info = self.main.ai_tab.import_model(self)
        if info:
            for i in range(self.ai_model.count()):
                m = self.ai_model.itemData(i)
                if m and m["name"] == info["name"]:
                    self.ai_model.setCurrentIndex(i)

    def _model_changed(self):
        m = self.ai_model.currentData()
        self.b_boxes.setEnabled(assist.has_box(m))
        self.b_points.setEnabled(assist.has_keypoint(m))

    def _current_image(self):
        path = self.images[self.index] if 0 <= self.index < len(self.images) else None
        if self._img[0] != path:
            self._img = (path, read_image(path) if path else None)
        return self._img[1]

    def _load_ai(self):
        m = self.ai_model.currentData()
        if m is None:
            return None, None
        QApplication.setOverrideCursor(Qt.WaitCursor)     # 처음 불러올 때 몇 초 걸림
        try:
            return assist.load_ai(m)
        except Exception as e:
            QMessageBox.warning(self, "모델을 불러올 수 없음", str(e))
            return None, None
        finally:
            QApplication.restoreOverrideCursor()

    def _auto_points(self, obj):
        """박스를 그린 직후 호출됨. AI가 키포인트를 자동 라벨링했으면 True."""
        if not self.auto_points.isChecked() or self.ai_model.currentData() is None:
            return False
        _, kpt_ai = self._load_ai()
        if kpt_ai is None:
            return False
        obj["kpts"], conf = kpt_ai.predict(self._current_image(), obj["box"])
        obj.pop("box_score", None)              # 사람이 그린 박스: 키포인트 점수만
        assist.set_scores(obj, kpt=conf)
        return True

    def show_examples(self):
        """지금 고른 클래스의 라벨링 예시 이미지 (보기, 추가, 삭제)."""
        from .examples import ExampleDialog
        ExampleDialog(self.canvas.current_cls, self.canvas if self.canvas.pixmap else None, self).exec()
        self.set_class(self.canvas.current_cls)

    def straighten(self):
        """선택 객체 키포인트 2, 3을 1-4 직선 3등분으로 (L)."""
        if self.canvas.label is not None:
            self.hint.setText(self.canvas.straighten_selected())

    def delete_all(self):
        """현재 이미지의 모든 객체 삭제 (Shift+Delete). Ctrl+Z 로 되돌릴 수 있음."""
        n = len(self.canvas.label["objects"]) if self.canvas.label else 0
        if self.canvas.delete_all():
            self.hint.setText(f"객체 {n}개 모두 삭제 완료 (되돌리기: Ctrl+Z)")

    def ai_batch(self):
        """여러 장 AI 자동 라벨링 (미작업 이미지 전체). 도는 동안 목록과 그림판은 잠금."""
        p, m = self.main.project, self.ai_model.currentData()
        if p is None or (self.worker and self.worker.isRunning()):
            return
        if m is None:
            QMessageBox.information(self, "모델 없음", "모델 없음: 3번 탭에서 학습 또는 다른 데이터셋 모델 가져오기 먼저 필요")
            return
        self.save_current()
        target, thr = self.target.currentData(), self.threshold.value()
        self.worker = Worker(lambda progress, log, should_stop: assist.prelabel(
            p, m, target, thr, "auto", progress, log, should_stop))
        self.worker.progress.connect(self._batch_progress)
        self.worker.log.connect(lambda t: self.batch_msg.setText(t.strip().splitlines()[-1] if t.strip() else ""))
        self.worker.finished.connect(self._batch_done)
        self._lock(True)
        self.worker.start()

    def _batch_progress(self, done, total, msg):
        self.batch_bar.setMaximum(max(total, 1))
        self.batch_bar.setValue(done)
        self.batch_msg.setText(msg)

    def _batch_done(self):
        self._lock(False)
        self.main.refresh_all()
        self.filter.setCurrentIndex(2)          # 'AI 예측' 보기로 바로 확인

    def clear_ai_labels(self):
        """'AI 예측' 상태인 이미지의 객체를 모두 지우고 미작업으로 (사람 검수 완료는 그대로).
        지우기 전 라벨은 removed/<날짜>_ai_labels/ 에 복사해 둠 (되돌리기용)."""
        import json
        import shutil
        import time
        p = self.main.project
        if p is None or self.batch_busy():
            return
        self.save_current()
        targets = []
        for img in p.images():
            lab = p.load_label(img)
            if lab and lab["status"] == C.STATUS_AI:
                targets.append((img, lab))
        if not targets:
            QMessageBox.information(self, "AI 예측 없음", "'AI 예측' 상태 이미지 없음")
            return
        n_obj = sum(len(lab["objects"]) for _, lab in targets)
        if QMessageBox.question(self, "AI 예측 라벨 전체 지우기",
                                f"'AI 예측' 이미지 {len(targets)}장의 객체 {n_obj}개를 모두 지우고 미작업으로 바꿀까요?\n\n"
                                "사람 검수 완료 이미지는 변경 없음\n"
                                f"지우기 전 라벨은 {p.root.name}\\removed\\ 에 복사 (되돌리기용)") != QMessageBox.Yes:
            return
        backup = p.root / "removed" / (time.strftime("%Y%m%d_%H%M%S") + "_ai_labels")
        backup.mkdir(parents=True)
        for img, lab in targets:
            shutil.copy2(p.label_path(img), backup / p.label_path(img).name)
            lab.update(objects=[], status=C.STATUS_TODO, ai_score=None, background=False)
            p.save_label(img, lab)
        self.main.refresh_all()
        self.hint.setText(f"AI 예측 {len(targets)}장의 객체 {n_obj}개 삭제, 미작업으로 변경 (이전 라벨: removed\\{backup.name})")

    def _lock(self, on):
        """여러 장 자동 라벨링 중에는 라벨 수정 잠금 (같은 파일을 동시에 고치지 않게)."""
        for w in (self.list, self.canvas, self.filter, self.g1, self.b_points, self.b_boxes, self.b_batch,
                  self.b_clear_ai, self.ai_model):
            w.setEnabled(not on)
        self.batch_stop.setEnabled(on)
        if not on:
            self._model_changed()

    def ai_fill_points(self):
        if self.canvas.label is None:
            return
        _, kpt_ai = self._load_ai()
        if kpt_ai is None:
            QMessageBox.information(self, "모델 없음", "모델 선택 필요")
            return
        self.canvas.save_undo()
        n = assist.fill_points(self._current_image(), self.canvas.label, kpt_ai)
        self.hint.setText(f"키포인트 AI 자동 라벨링 완료: 객체 {n}개. 확인, 수정 후 확정" if n
                          else "키포인트가 빈 객체 없음 (박스 먼저 그리기)")
        self.canvas.set_mode("edit")
        self.on_changed()
        self.canvas.update()

    def ai_add_boxes(self):
        if self.canvas.label is None:
            return
        box_ai, kpt_ai = self._load_ai()
        if box_ai is None:
            return
        self.canvas.save_undo()
        n = assist.add_boxes(self._current_image(), self.canvas.label, box_ai, kpt_ai, self.threshold.value())
        self.hint.setText(f"박스 + 키포인트 AI 자동 라벨링 완료: 객체 {n}개. 확인, 수정 후 확정" if n
                          else "새로 찾은 객체 없음")
        self.canvas.set_mode("edit")
        self.on_changed()
        self.canvas.update()

    # ---------- 화면 갱신 ----------
    def set_class(self, i):
        self.class_group.button(i).setChecked(True)
        self.canvas.current_cls = i
        c = C.CLASSES[i]
        pts = "".join(f"<li>{n}</li>" for n in c["points"])
        rules = "".join(f"<li>{g}</li>" for g in c["guide"])
        self.guide.setText(f"<b style='color:{c['color']}'>{c['name']}</b><br>키포인트 순서<ol>{pts}</ol>규칙<ul>{rules}</ul>")
        from .examples import example_files
        self.b_example.setText(f"{c['name']} 예시 이미지 보기 ({len(example_files(i))}장)")
        if self.canvas.mode == "box":
            self.on_hint(self.canvas.hint())

    def on_changed(self):
        self.dirty = True
        self.refresh_info()
        it = self.list.item(self.index) if 0 <= self.index < self.list.count() else None
        if it is not None and self.canvas.label is not None:   # 목록의 객체 수도 바로 반영
            it.setText(f"● {self.images[self.index].stem}  {object_counts(self.canvas.label['objects'])}")

    def on_hint(self, text):
        self.hint.setText(text)
        self.btn_box.setChecked(self.canvas.mode == "box")
        self.btn_edit.setChecked(self.canvas.mode == "edit")

    def refresh_info(self):
        lab = self.canvas.label
        if lab is None:
            return
        self.name.setText(lab["image"])
        st = C.STATUS_TEXT[lab["status"]]
        if lab["status"] == C.STATUS_HUMAN and lab["background"]:
            st += " (시설물 없음)"
        if lab.get("source"):
            st += f"<br><span style='color:#666'>출처: {lab['source']}</span>"
        self.status.setText(f"상태: <b style='color:{C.STATUS_COLOR[lab['status']]}'>{st}</b>")
        self.objects.blockSignals(True)
        self.objects.clear()
        for i, o in enumerate(lab["objects"]):
            n = sum(1 for k in o["kpts"] if k[2] > 0)
            sc = f"  ({score_text(o)})" if score_text(o) else ""
            self.objects.addItem(f"{i + 1}. {C.CLASS_NAMES[o['cls']]}  키포인트 {n}/4{sc}")
        self.objects.setCurrentRow(self.canvas.selected)
        self.objects.blockSignals(False)
