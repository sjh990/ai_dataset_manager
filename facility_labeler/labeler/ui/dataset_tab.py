"""1. 데이터셋 / 영상 화면.

데이터셋 = 영상에서 뽑은 이미지 + 라벨 + 학습한 AI 를 모아 둔 폴더 하나 (datasets\\<이름>).
이 화면에서 하는 일
    - 이어서 작업하기: 이미 있는 데이터셋 열기
    - 프레임 추출: 영상 선택, 구간 자르기, 간격 → 새 데이터셋 폴더(이름 자동) 또는 지금 연 데이터셋에 저장
"""
import os
import re
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import config as C
from ..add_video import add_video, change_range, find_video, range_preview
from ..frames import plan_frames, video_info
from ..project import Project
from .common import RunPanel, note, text_label
from .trim_dialog import TrimDialog, mmss

DATASETS_DIR = C.DATASETS_DIR                                  # 데이터셋 폴더들이 모이는 곳 (datasets)

# 작업 순서 (화면 오른쪽 위 표). (순서, 탭, 작업)
STEPS_CONTINUE = [
    ("1", "1. 데이터셋", "<b>데이터셋 열기</b>: 왼쪽 목록에서 데이터셋 더블클릭"),
    ("2", "2. 라벨링", "<b>이어서 라벨링</b>: 필터 '미작업' 또는 'AI 예측', 수정 후 확정 (Space)"),
    ("3", "3. AI 학습", "<b>재학습</b>: 확정이 쌓이면 다시 학습"),
    ("4", "2. 라벨링", "<b>AI 자동 라벨링 (여러 장)</b>: 미작업 이미지 한 번에, 'AI 예측' 보기에서 확인 후 확정"),
    ("5", "4. 내보내기", "<b>YOLO 데이터셋 내보내기</b>: 사람 검수 완료 이미지만 저장"),
]
STEPS_NEW = [
    ("1", "1. 데이터셋", "<b>영상 선택, 구간 자르기</b>: 앞뒤 필요 없는 부분 빼기, 추출 간격 0.5초"),
    ("2", "1. 데이터셋", "<b>프레임 추출</b>: 저장 위치 '새 데이터셋 폴더' (이름 자동: 영상_시작_끝_간격)"),
    ("3", "3. AI 학습", "<b>다른 데이터셋 모델 가져오기</b>: 학습해 둔 모델 선택 ('박스 + 키포인트' 추천)"),
    ("4", "2. 라벨링", "<b>AI 자동 라벨링 (여러 장)</b>: 가져온 모델 선택, 대상 '미작업 이미지 전체', 시작"),
    ("5", "2. 라벨링", "<b>AI 예측 확인</b>: 필터 'AI 예측' (점수 낮은 순), 수정 후 확정 (Space), "
                     "시설물 없는 이미지는 시설물 없음 (N)"),
    ("6", "3. AI 학습", "<b>재학습</b>: 확정이 쌓이면 이 데이터셋 라벨로 다시 학습 (반복)"),
    ("7", "4. 내보내기", "<b>YOLO 데이터셋 내보내기</b>: 사람 검수 완료 이미지만 저장"),
]


def steps_html(title, steps):
    rows = "".join(f"<tr><td align=center><b>{n}</b></td><td style='color:#1565c0'>{tab}</td><td>{what}</td></tr>"
                   for n, tab, what in steps)
    return (f"<b>{title}</b><table border=1 cellspacing=0 cellpadding=4 width=100%>"
            "<tr style='background:#eeeeee'><th>순서</th><th>탭</th><th>작업</th></tr>" + rows + "</table>")


def find_datasets():
    """datasets 폴더 안의 데이터셋 목록 (최근 수정 순)."""
    out = []
    if DATASETS_DIR.exists():
        for d in DATASETS_DIR.iterdir():
            if (d / "project.json").exists():
                out.append(d)
    return sorted(out, key=lambda d: (d / "labels").stat().st_mtime if (d / "labels").exists() else 0, reverse=True)


def m_s(sec):
    """초 → 폴더 이름용 글자 (예: 65 → 1m05s)."""
    sec = int(round(sec))
    return f"{sec // 60}m{sec % 60:02d}s"


def snake(text):
    """폴더 이름용으로: 영문 소문자, 숫자, _ 만 (한글, 띄어쓰기, 점, - 는 _ 로 바꾸고 겹친 _ 는 하나로)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "dataset"


def auto_name(video, start, end, full, every):
    """설정이 이름에 다 보이는 데이터셋 폴더 이름. (영문 소문자, 숫자, _ 만).

    형식: 영상_시작_끝_간격   예: dji_0343_1m00s_9m00s_500ms   (구간 1분 0초 ~ 9분 0초, 0.5초(500ms)마다 1장)
                              dji_0343_full_1s           (영상 전체, 1초마다 1장)
    장수는 구간과 간격으로 정해지므로 이름에서 뺌 (화면 목록에 표시)
    점(.)은 확장자로 오해받을 수 있어 쓰지 않음 → 간격은 정수 초면 s, 아니면 ms
    """
    rng = "full" if full else f"{m_s(start)}_{m_s(end)}"
    ms = round(every * 1000)
    gap = f"{ms // 1000}s" if ms % 1000 == 0 else f"{ms}ms"         # 1초 → 1s, 0.5초 → 500ms
    name = snake(f"{Path(video).stem}_{rng}_{gap}")
    unique, i = name, 2
    while (DATASETS_DIR / unique).exists():            # 같은 이름이 있으면 _2, _3 ...
        unique, i = f"{name}_{i}", i + 1
    return unique


class DatasetTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.video, self.info = None, None
        self.start, self.end = 0.0, None        # 쓸 구간 (초). end None = 영상 끝까지
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        lay.addLayout(left, 1)

        # ---- 지금 연 데이터셋 ----
        self.current = text_label("열린 데이터셋 없음: 아래 목록에서 열기, 또는 영상 선택 후 새 데이터셋 만들기")
        left.addWidget(self.current)

        # ---- 이어서 작업하기 ----
        g = QGroupBox("이어서 작업하기 (이미 있는 데이터셋)")
        gv = QVBoxLayout(g)
        self.list = QTableWidget(0, 5)
        self.list.setHorizontalHeaderLabels(["데이터셋", "전체 이미지", "사람 검수", "영상", "마지막 작업"])
        self.list.verticalHeader().setVisible(False)
        self.list.horizontalHeader().setStretchLastSection(True)
        self.list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.list.setMaximumHeight(170)
        self.list.cellDoubleClicked.connect(lambda r, _c: self.open_row(r))
        row = QHBoxLayout()
        b_open = QPushButton("열기")
        b_open.setObjectName("primary")
        b_open.clicked.connect(lambda: self.open_row(self.list.currentRow()))
        b_other = QPushButton("다른 위치에서 열기...")
        b_other.clicked.connect(self.open_other)
        b_reload = QPushButton("목록 새로고침")
        b_reload.clicked.connect(self.fill_list)
        row.addWidget(b_open)
        row.addWidget(b_other)
        row.addStretch(1)
        row.addWidget(b_reload)
        gv.addWidget(self.list)
        gv.addLayout(row)
        gv.addWidget(note("더블클릭으로도 열기 가능"))
        left.addWidget(g)

        # ---- 프레임 추출 ----
        g = QGroupBox("영상에서 프레임 추출")
        f = QFormLayout(g)
        vrow = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setReadOnly(True)
        b_vid = QPushButton("영상 선택...")
        b_vid.clicked.connect(self.pick_video)
        self.btn_trim = QPushButton("구간 자르기...")
        self.btn_trim.setEnabled(False)
        self.btn_trim.clicked.connect(self.trim_video)
        vrow.addWidget(self.video_edit, 1)
        vrow.addWidget(b_vid)
        vrow.addWidget(self.btn_trim)
        self.range_label = QLabel("-")
        self.every = QDoubleSpinBox()
        self.every.setRange(0.04, 60)
        self.every.setSingleStep(0.1)
        self.every.setValue(0.5)
        self.every.setSuffix(" 초마다 1장")
        self.every.valueChanged.connect(self.update_estimate)
        self.estimate = QLabel("-")

        # 저장 위치: 새 데이터셋 폴더 (기본) / 지금 연 데이터셋
        self.to_new = QRadioButton("새 데이터셋 폴더 만들기 (추천)")
        self.to_current = QRadioButton("지금 연 데이터셋에 추가")
        self.to_new.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.to_new)
        grp.addButton(self.to_current)
        self.to_new.toggled.connect(self.update_estimate)
        drow = QHBoxLayout()
        drow.addWidget(self.to_new)
        drow.addWidget(self.to_current)
        drow.addStretch(1)
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("영상 선택 시 자동 생성")
        self.new_name.textEdited.connect(lambda _: setattr(self, "name_edited", True))
        self.name_edited = False             # 사람이 이름을 직접 고쳤으면 자동 이름으로 덮어쓰지 않음

        self.btn_add = QPushButton("프레임 추출 시작")
        self.btn_add.setObjectName("primary")
        self.btn_add.clicked.connect(self.start_add)
        f.addRow("영상", vrow)
        f.addRow("쓸 구간", self.range_label)
        f.addRow("추출 간격", self.every)
        f.addRow("예상 장수", self.estimate)
        f.addRow("저장 위치", drow)
        f.addRow("새 폴더 이름", self.new_name)
        f.addRow(note("<b>구간 자르기</b>: 영상을 보며 앞뒤 필요 없는 부분 제외, 쓸 구간만 지정 (영상 파일은 그대로)<br>"
                      "<b>새 데이터셋 폴더</b>: 구간이나 간격이 다르면 별도 폴더. 이름 = <b>영상_시작_끝_간격</b> (영문 소문자, 숫자, _ 만)<br>"
                      "&nbsp;&nbsp;예: dji_0343_1m00s_9m00s_500ms = 구간 1분 0초 ~ 9분 0초, 0.5초(500ms)마다 1장 / 영상 전체면 dji_0343_full_500ms<br>"
                      "<b>지금 연 데이터셋에 추가</b>: 다른 영상을 같은 데이터셋에 넣어 함께 학습할 때"))
        f.addRow(self.btn_add)
        left.addWidget(g)
        self.run = RunPanel()
        left.addWidget(self.run, 1)

        # ---- 오른쪽: 작업 순서, 진행 현황, 추가한 영상 (스크롤) ----
        right_box = QWidget()
        right = QVBoxLayout(right_box)
        right.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(right_box)
        lay.addWidget(scroll, 1)

        g = QGroupBox("작업 순서")
        gv = QVBoxLayout(g)
        gv.addWidget(text_label(steps_html("A. 이어서 작업하기", STEPS_CONTINUE)))
        gv.addWidget(text_label(steps_html("B. 새 영상으로 시작하기", STEPS_NEW)))
        gv.addWidget(note("학습한 모델이 아직 하나도 없는 경우: B의 3, 4 대신 직접 몇십 장 라벨링, 확정 후 AI 학습"))
        right.addWidget(g)

        g = QGroupBox("진행 현황 (지금 연 데이터셋)")
        grid = QGridLayout(g)
        self.nums = {}
        for i, (key, text) in enumerate((("total", "전체 이미지"), ("human", "사람 검수 완료"),
                                         ("ai", "AI 예측 (확인 필요)"), ("todo", "미작업"))):
            big = QLabel("-")
            big.setObjectName("big")
            big.setAlignment(Qt.AlignCenter)
            grid.addWidget(big, 0, i)
            t = QLabel(text)
            t.setAlignment(Qt.AlignCenter)
            grid.addWidget(t, 1, i)
            self.nums[key] = big
        self.detail = text_label("")
        grid.addWidget(self.detail, 2, 0, 1, 4)
        right.addWidget(g)

        g = QGroupBox("추가한 영상 (지금 연 데이터셋)")
        gv = QVBoxLayout(g)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["영상", "길이", "쓴 구간", "영상 프레임", "간격", "저장 장수"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(140)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.btn_range = QPushButton("선택한 영상 구간 변경...")
        self.btn_range.clicked.connect(self.change_video_range)
        gv.addWidget(self.table)
        gv.addWidget(self.btn_range)
        gv.addWidget(note("<b>구간 변경</b>: 영상을 보며 새 구간 지정 → 구간 밖 이미지, 라벨은 &lt;데이터셋&gt;/removed/ 로 이동 "
                          "(삭제 아님, 복구 가능), 새 구간에서 없는 프레임은 같은 간격으로 추가 추출. "
                          "영상이 1개인 데이터셋은 폴더 이름도 새 구간으로 변경 가능"))
        right.addWidget(g)
        right.addStretch(1)
        self.fill_list()

    # ---------- 데이터셋 고르기 ----------
    def fill_list(self):
        self.rows = find_datasets()
        self.list.setRowCount(len(self.rows))
        for r, d in enumerate(self.rows):
            p = Project(d)
            s = p.summary()
            labels = d / "labels"
            last = time.strftime("%m-%d %H:%M", time.localtime(labels.stat().st_mtime)) if labels.exists() else "-"
            vals = [p.info["name"], f"{s['total']}장", f"{s['human']}장", str(len(p.info.get("videos", []))), last]
            for c, t in enumerate(vals):
                self.list.setItem(r, c, QTableWidgetItem(t))
        self.list.resizeColumnsToContents()

    def open_row(self, r):
        if 0 <= r < len(self.rows):
            self.open_dataset(self.rows[r])

    def open_other(self):
        folder = QFileDialog.getExistingDirectory(self, "데이터셋 폴더 선택 (project.json 이 있는 곳)", str(DATASETS_DIR))
        if folder:
            self.open_dataset(Path(folder))

    def open_dataset(self, folder):
        if self.run.busy():
            QMessageBox.information(self, "작업 중", "프레임 추출 중: 끝난 뒤 다른 데이터셋 열기 가능")
            return
        if not (folder / "project.json").exists():
            QMessageBox.warning(self, "데이터셋 아님", "데이터셋 아님 (project.json 없음)")
            return
        self.main.set_project(Project(folder))

    # ---------- 영상, 구간, 간격 ----------
    def pick_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "영상 (*.mp4 *.mov *.avi *.mkv *.ts)")
        if not file:
            return
        info = video_info(file)
        if info is None:
            QMessageBox.warning(self, "영상 열기 실패", f"영상 열기 실패: {file}")
            return
        self.video, self.info = file, info
        self.start, self.end = 0.0, None
        self.name_edited = False
        self.video_edit.setText(Path(file).name)
        self.btn_trim.setEnabled(True)
        self.update_estimate()

    def trim_video(self):
        try:
            dlg = TrimDialog(self.video, self.start, self.end, self.every.value(), self)
        except Exception as e:
            QMessageBox.warning(self, "영상 열기 실패", str(e))
            return
        if dlg.exec():
            self.start, self.end = dlg.start_sec, dlg.end_sec
            self.update_estimate()

    def settings(self):
        """지금 고른 추출 설정: (시작초, 끝초, 영상 전체인지, 간격, 예상 장수)."""
        info, every = self.info, self.every.value()
        end = info["seconds"] if self.end is None else self.end
        full = self.start <= 0.05 and end >= info["seconds"] - 0.05
        step = max(1, round(every * info["fps"]))
        count = len(plan_frames(info["frames"], step, round(self.start * info["fps"]), round(end * info["fps"])))
        return self.start, end, full, every, count

    def update_estimate(self):
        has_project = self.main.project is not None
        self.to_current.setEnabled(has_project)
        self.to_current.setText(f"지금 연 데이터셋에 추가 ({self.main.project.info['name']})" if has_project
                                else "지금 연 데이터셋에 추가 (열린 데이터셋 없음)")
        if not has_project:
            self.to_new.setChecked(True)
        self.new_name.setEnabled(self.to_new.isChecked())
        if self.video is None:
            return
        start, end, full, every, count = self.settings()
        self.range_label.setText(f"전체 ({mmss(end)})" if full else
                                 f"{mmss(start)} ~ {mmss(end)}  (길이 {end - start:.1f}초, 영상 전체 {mmss(self.info['seconds'])})")
        self.estimate.setText(f"약 {count}장")
        if not self.name_edited:
            self.new_name.setText(auto_name(self.video, start, end, full, every))

    # ---------- 추출 ----------
    def start_add(self):
        if self.video is None or self.run.busy():
            return
        start, end, full, every, count = self.settings()
        end_sec = None if full else end
        if self.to_new.isChecked():
            name = self.new_name.text().strip()
            if name != snake(name):
                QMessageBox.warning(self, "이름 확인", f"폴더 이름: 영문 소문자, 숫자, _ 만 가능\n추천: {snake(name)}")
                return
            if (DATASETS_DIR / name).exists():
                QMessageBox.warning(self, "이미 있음", f"'{name}' 폴더 이미 있음: 이름 변경 필요")
                return
            target, new = DATASETS_DIR / name, True
            where = f"새 데이터셋 폴더\n{target}"
        else:
            p = self.main.project
            same = [v for v in p.info.get("videos", []) if Path(v["file"]).stem == Path(self.video).stem]
            if same:
                old = ", ".join(f"구간 {mmss(v['range_sec'][0])} ~ {mmss(v['range_sec'][1])}, 간격 {v['every_sec']}초"
                                if v.get("range_sec") else f"전체, 간격 {v['every_sec']}초" for v in same)
                QMessageBox.information(
                    self, "같은 영상이 이미 있음",
                    f"'{p.info['name']}' 데이터셋에 같은 영상 있음 ({old})\n\n"
                    "구간을 바꾸려면: 오른쪽 '추가한 영상' 표 아래 '선택한 영상 구간 변경'\n"
                    "구간이나 간격이 다른 데이터를 따로 만들려면: 저장 위치 '새 데이터셋 폴더'")
                return
            target, new = p.root, False
            where = f"지금 연 데이터셋 '{p.info['name']}'"
        rng = "전체" if full else f"{mmss(start)} ~ {mmss(end)}"
        if QMessageBox.question(self, "프레임 추출",
                                f"영상: {Path(self.video).name}\n구간: {rng}\n간격: {every}초  (약 {count}장)\n\n"
                                f"저장 위치: {where}") != QMessageBox.Yes:
            return

        project = Project.create(target, target.name) if new else self.main.project
        video = self.video
        self.btn_add.setEnabled(False)

        def job(progress, log, should_stop):
            add_video(project, video, every, progress, should_stop, log, start, end_sec)

        def finished(_):
            self.btn_add.setEnabled(True)
            self.name_edited = False
            self.main.set_project(project)          # 추출한 데이터셋을 바로 엶

        self.run.start(job, finished)
        self.run.worker.failed.connect(lambda _: (self.btn_add.setEnabled(True), self.fill_list()))

    # ---------- 이미 추가한 영상의 구간 바꾸기 ----------
    def change_video_range(self):
        p = self.main.project
        if p is None or self.run.busy():
            return
        busy = self.main.busy_jobs(skip=self)
        if busy:                               # 이미지를 옮기는 작업이라 다른 작업과 동시에 하지 않음
            QMessageBox.information(self, "구간 변경 불가", f"{', '.join(busy)} 진행 중: 끝난 뒤 구간 변경")
            return
        vids = p.info.get("videos", [])
        row = self.table.currentRow() if len(vids) > 1 else 0
        if not vids or row < 0:
            QMessageBox.information(self, "영상 선택", "아래 표에서 구간을 바꿀 영상 선택 필요")
            return
        v = vids[row]
        video = find_video(v["file"])
        if video is None:                       # 영상을 옮긴 경우: 직접 위치 지정
            file, _ = QFileDialog.getOpenFileName(self, f"영상 위치 지정: {Path(v['file']).name}", str(C.APP_DIR),
                                                  "영상 (*.mp4 *.mov *.avi *.mkv *.ts)")
            if not file:
                return
            video = Path(file)
        start, end = v.get("range_sec") or [0.0, v["video_seconds"]]
        try:
            dlg = TrimDialog(str(video), start, None if end >= v["video_seconds"] - 0.1 else end, v["every_sec"], self)
        except Exception as e:
            QMessageBox.warning(self, "영상 열기 실패", str(e))
            return
        if not dlg.exec():
            return
        s, e = dlg.start_sec, dlg.end_sec
        pv = range_preview(p, row, video, s, e)
        r = pv["removed"]
        e_txt = mmss(e) if e is not None else f"끝 ({mmss(v['video_seconds'])})"
        msg = (f"새 구간: {mmss(s)} ~ {e_txt}  (간격 {v['every_sec']}초)\n\n"
               f"구간 밖으로 정리할 이미지: {pv['remove_total']}장\n"
               f"  - 사람 검수 완료 {r[C.STATUS_HUMAN]}장, AI 예측 {r[C.STATUS_AI]}장, 미작업 {r[C.STATUS_TODO]}장\n"
               f"  - 이동 위치: {p.root.name}\\removed\\ (삭제 아님, 복구 가능)\n"
               f"새로 뽑을 이미지: {pv['add']}장\n\n진행할까요?")
        if QMessageBox.question(self, "구간 변경", msg) != QMessageBox.Yes:
            return
        self.main.label_tab.save_current()
        self.btn_range.setEnabled(False)

        def finished(_):
            self.btn_range.setEnabled(True)
            self.main.refresh_all()            # 옮겨진 이미지는 목록에서 빠짐
            self.offer_rename(video, s, e, v["every_sec"])

        self.run.text.clear()
        self.run.start(lambda progress, log, should_stop: change_range(p, row, video, s, e, progress, should_stop, log),
                       finished)
        self.run.worker.failed.connect(lambda _: self.btn_range.setEnabled(True))

    def offer_rename(self, video, start, end, every):
        """영상이 1개인 데이터셋: 폴더 이름을 새 구간 기준 이름으로 바꿀지 물어봄."""
        p = self.main.project
        if p is None or len(p.info.get("videos", [])) != 1:
            return
        info = video_info(str(video))
        e = info["seconds"] if end is None else end
        full = start <= 0.05 and e >= info["seconds"] - 0.05
        new = auto_name(str(video), start, e, full, every)
        if new.rstrip("_0123456789") == p.root.name.rstrip("_0123456789") or new == p.root.name:
            return
        if QMessageBox.question(self, "폴더 이름 변경",
                                f"구간이 바뀌어 폴더 이름도 바꿀까요?\n\n{p.root.name}\n→ {new}") != QMessageBox.Yes:
            return
        self.rename_dataset(new)

    def rename_dataset(self, new_name):
        p = self.main.project
        busy = self.main.busy_jobs()
        if busy:
            QMessageBox.information(self, "이름 변경 불가", f"{', '.join(busy)} 진행 중: 끝난 뒤 이름 변경")
            return
        self.main.label_tab.save_current()
        new_root = p.root.parent / new_name
        try:
            os.rename(p.root, new_root)
        except OSError as e:
            QMessageBox.warning(self, "이름 변경 실패", f"폴더 이름 변경 실패 (다른 프로그램이 폴더를 쓰는 중일 수 있음)\n{e}")
            return
        q = Project(new_root)
        q.info["name"] = new_name
        q.save_info()
        self.main.set_project(q)

    # ---------- 현황 ----------
    def refresh(self):
        p = self.main.project
        self.update_estimate()
        if p is None:
            return
        self.current.setText(f"지금 작업 중인 데이터셋: <b style='font-size:13pt'>{p.info['name']}</b>"
                             f"<br><span style='color:#666'>경로: {p.root}</span>")
        s = p.summary()
        for k, w in self.nums.items():
            w.setText(str(s[k]))
        pct = 100 * s["human"] / max(1, s["total"])
        objs = ", ".join(f"{k} {v}개" for k, v in s["objects"].items())
        self.detail.setText(
            f"사람 검수 비율 <b>{pct:.1f}%</b> (전체 {s['total']}장 중 {s['human']}장)<br>"
            f"&nbsp;- 시설물 있음 {s['human_with_objects']}장 / 시설물 없음 {s['human_background']}장<br>"
            f"검수 완료된 객체: {objs}")
        vids = p.info.get("videos", [])
        self.table.setRowCount(len(vids))
        for r, v in enumerate(vids):
            rng = v.get("range_sec")
            vals = [Path(v["file"]).name, f"{v['video_seconds']}초",
                    f"{mmss(rng[0])} ~ {mmss(rng[1])}" if rng else "전체", str(v["video_frames"]),
                    f"{v['every_sec']}초", str(v["frames_saved"])]
            for c, t in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(t))
        self.table.resizeColumnsToContents()
        self.fill_list()
        for r, d in enumerate(self.rows):           # 목록에서 지금 연 데이터셋 표시
            if d.resolve() == p.root.resolve():
                self.list.selectRow(r)
