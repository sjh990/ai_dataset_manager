"""라벨링 그림판: 이미지 위에 박스와 키포인트를 그리고 고칩니다.

모드
    box   : 마우스로 끌어서 박스 그리기 → 끝나면 자동으로 점 찍기 모드
    point : 클릭할 때마다 다음 키포인트를 찍음 (S 키 = 이 점 건너뛰기)
    edit  : 점/박스 모서리를 끌어서 고치기, 박스 안을 끌면 객체 전체 이동
화면 이동/확대: 휠 = 확대/축소, 오른쪽 버튼 끌기 = 이동
"""
import copy

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .. import config as C

GRAB = 9  # 이만큼(화면 픽셀) 가까우면 점/모서리를 잡은 것으로 봄


def score_text(o):
    """AI 점수 글자. 박스 + 키포인트를 AI가 했으면 둘 다, 하나만 했으면 그것만."""
    parts = []
    if o.get("box_score") is not None:
        parts.append(f"박스 {o['box_score']:.2f}")
    if o.get("kpt_score") is not None:
        parts.append(f"키포인트 {o['kpt_score']:.2f}")
    if not parts and o.get("score") is not None:      # 예전에 저장된 라벨 (점수 1개)
        parts.append(f"AI {o['score']:.2f}")
    return " / ".join(parts)


def straighten(o):
    """키포인트 2, 3을 1-4 직선 위 3등분 위치로 (1, 4가 둘 다 있을 때). 바꿨으면 True."""
    k = o["kpts"]
    if k[0][2] <= 0 or k[3][2] <= 0:
        return False
    (x1, y1), (x4, y4) = k[0][:2], k[3][:2]
    k[1] = [x1 + (x4 - x1) / 3, y1 + (y4 - y1) / 3, 2]
    k[2] = [x1 + (x4 - x1) * 2 / 3, y1 + (y4 - y1) * 2 / 3, 2]
    return True


def new_object(cls, box):
    return {"cls": cls, "box": box, "kpts": [[0.0, 0.0, 0] for _ in range(C.NUM_KEYPOINTS)], "score": None}


class Canvas(QWidget):
    changed = Signal()          # 라벨이 바뀜
    hint_changed = Signal(str)  # 안내 문구 바뀜

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.pixmap = None
        self.label = None
        self.mode = "box"
        self.current_cls = 0
        self.selected = -1          # 선택된 객체 번호
        self.next_point = 0         # point 모드에서 다음에 찍을 점 번호
        self.scale, self.offset = 1.0, QPointF(0, 0)
        self.drag = None            # 끌고 있는 대상 정보
        self.undo_stack = []
        self.auto_points = None     # 박스를 그린 뒤 AI로 점을 찍는 함수 (라벨링 화면이 넣어 줌)

    # ---------- 바깥에서 부르는 함수 ----------
    def set_image(self, pixmap, label):
        self.pixmap, self.label = pixmap, label
        self.selected, self.drag, self.undo_stack = -1, None, []
        self.set_mode("box")
        self.fit()

    def set_mode(self, mode):
        self.mode = mode
        self.update()
        self.hint_changed.emit(self.hint())

    def hint(self):
        if self.mode == "box":
            return (f"[박스 그리기] '{C.CLASS_NAMES[self.current_cls]}' 박스: 빈 곳을 끌어서 그리기  "
                    "/ 키포인트, 선택한 박스 모서리: 바로 끌어서 수정")
        if self.mode == "point" and self.selected >= 0:
            cls = self.label["objects"][self.selected]["cls"]
            names = C.CLASSES[cls]["points"]
            if C.lower_half(cls):          # 중앙분리대, 펜스: 1, 4만 클릭 → 2, 3 자동
                if self.next_point == 0:
                    return "[키포인트 찍기] 1번 '아래 끝' 클릭 → 다음 4번 '위 끝' 클릭하면 2, 3은 일직선 3등분에 자동"
                return "[키포인트 찍기] 4번 '위 끝' 클릭 (기준선 근처를 누르면 기준선에 맞춤, 시설물이 기준선 전에 끝나면 끝나는 곳) → 2, 3 자동"
            return f"[키포인트 찍기] {self.next_point + 1}/4: '{names[self.next_point]}' 위치 클릭  (안 보이면 S = 건너뛰기)"
        return "[선택/수정] 키포인트, 박스 모서리: 끌어서 수정 / 박스 안: 끌어서 통째로 이동"

    def fit(self):
        if not self.pixmap:
            return
        s = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        self.scale = s
        self.offset = QPointF((self.width() - self.pixmap.width() * s) / 2, (self.height() - self.pixmap.height() * s) / 2)
        self.update()

    def save_undo(self):
        self.undo_stack.append(copy.deepcopy(self.label["objects"]))
        self.undo_stack = self.undo_stack[-50:]

    def undo(self):
        if self.undo_stack:
            self.label["objects"] = self.undo_stack.pop()
            self.selected = -1
            self.set_mode("edit")
            self.changed.emit()

    def delete_selected(self):
        if 0 <= self.selected < len(self.label["objects"]):
            self.save_undo()
            del self.label["objects"][self.selected]
            self.selected = -1
            self.set_mode("box")
            self.changed.emit()

    def delete_all(self):
        """이미지 안의 모든 객체 삭제 (Ctrl+Z 로 되돌리기 가능)."""
        if self.label and self.label["objects"]:
            self.save_undo()
            self.label["objects"] = []
            self.selected = -1
            self.set_mode("box")
            self.changed.emit()
            return True
        return False

    def straighten_selected(self):
        """선택 객체의 키포인트 2, 3을 1-4 직선 3등분으로 정렬 (L). 결과 안내 글자를 돌려줌."""
        if not (0 <= self.selected < len(self.label["objects"])):
            return "정렬할 객체를 먼저 선택 (V 선택/수정 모드에서 클릭)"
        o = self.label["objects"][self.selected]
        self.save_undo()
        if not straighten(o):
            self.undo_stack.pop()
            return "키포인트 1, 4가 둘 다 있어야 정렬 가능"
        self.changed.emit()
        self.update()
        return "키포인트 2, 3을 1-4 직선 3등분으로 정렬 (되돌리기: Ctrl+Z)"

    def skip_point(self):
        """지금 찍을 점이 안 보일 때: 비워 두고 다음 점으로."""
        if self.mode == "point" and self.selected >= 0:
            self.save_undo()
            self.label["objects"][self.selected]["kpts"][self.next_point] = [0.0, 0.0, 0]
            self._advance_point()

    def redo_points(self):
        """선택한 객체의 점을 처음부터 다시 찍기."""
        if self.selected >= 0:
            self.save_undo()
            self.label["objects"][self.selected]["kpts"] = [[0.0, 0.0, 0] for _ in range(C.NUM_KEYPOINTS)]
            self.next_point = 0
            self.set_mode("point")
            self.changed.emit()

    def select(self, index):
        self.selected = index
        self.set_mode("edit")

    # ---------- 좌표 변환 ----------
    def to_image(self, p):
        return QPointF((p.x() - self.offset.x()) / self.scale, (p.y() - self.offset.y()) / self.scale)

    def to_screen(self, x, y):
        return QPointF(x * self.scale + self.offset.x(), y * self.scale + self.offset.y())

    # ---------- 그리기 ----------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#303030"))
        if not self.pixmap:
            p.setPen(Qt.white)
            p.drawText(self.rect(), Qt.AlignCenter, "왼쪽 목록에서 이미지 선택")
            return
        p.setRenderHint(QPainter.Antialiasing)
        p.drawPixmap(QRectF(self.offset.x(), self.offset.y(), self.pixmap.width() * self.scale,
                            self.pixmap.height() * self.scale), self.pixmap, QRectF(self.pixmap.rect()))
        self._draw_center_line(p)      # 화면에만 그림 (이미지 파일에는 저장되지 않음)
        is_ai = self.label["status"] == C.STATUS_AI
        for i, o in enumerate(self.label["objects"]):
            self._draw_object(p, o, selected=(i == self.selected), dashed=is_ai)
        if self.drag and self.drag["type"] == "new_box":
            a, b = self.drag["start"], self.drag["now"]
            p.setPen(QPen(QColor(C.CLASSES[self.current_cls]["color"]), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)          # 키포인트 원을 칠한 색이 남아 박스가 채워지지 않게
            p.drawRect(QRectF(self.to_screen(a.x(), a.y()), self.to_screen(b.x(), b.y())))

    def _draw_center_line(self, p):
        """기준선 (화면 위에서 2/3 높이). 화면에 겹쳐 그리기만 하고 이미지 파일, 라벨, 내보내기에는 들어가지 않음."""
        y = self.guide_y()
        a, b = self.to_screen(0, y), self.to_screen(self.pixmap.width(), y)
        p.setPen(QPen(QColor("#ffeb3b"), 2, Qt.DashLine))
        p.drawLine(a, b)
        p.setFont(QFont("Malgun Gothic", 9, QFont.Bold))
        p.drawText(a + QPointF(6, -6), "기준선 (이 선 아래만 라벨링)")

    def clamp(self, x, y):
        """좌표를 이미지 안으로 (이미지 밖 좌표는 YOLO 가 그 이미지를 통째로 버림)."""
        w, h = self.pixmap.width(), self.pixmap.height()
        return min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)

    def clamp_object(self, o):
        x1, y1 = self.clamp(o["box"][0], o["box"][1])
        x2, y2 = self.clamp(o["box"][2], o["box"][3])
        o["box"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        for k in o["kpts"]:
            if k[2] > 0:
                k[0], k[1] = self.clamp(k[0], k[1])

    def guide_y(self):
        return self.pixmap.height() * C.GUIDE_LINE_RATIO

    def clip_lower(self, o):
        """중앙분리대, 펜스 박스의 위쪽을 기준선에 맞춤. 박스가 기준선 위에만 있으면 False."""
        if not C.lower_half(o["cls"]):
            return True
        mid = self.guide_y()
        if o["box"][3] <= mid + 2:
            return False
        o["box"][1] = max(o["box"][1], mid)
        return True

    def _draw_object(self, p, o, selected, dashed):
        color = QColor(C.CLASSES[o["cls"]]["color"])
        x1, y1, x2, y2 = o["box"]
        pen = QPen(color, 3 if selected else 2, Qt.DashLine if dashed else Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(self.to_screen(x1, y1), self.to_screen(x2, y2)))
        text = f"{C.CLASS_NAMES[o['cls']]}  {score_text(o)}".strip()
        p.setFont(QFont("Malgun Gothic", 9, QFont.Bold))
        p.drawText(self.to_screen(x1, y1) + QPointF(2, -4), text)
        k = o["kpts"]
        p.setPen(QPen(QColor("#ffeb3b"), 2))
        for a, b in C.SKELETON:
            if k[a][2] > 0 and k[b][2] > 0:
                p.drawLine(self.to_screen(k[a][0], k[a][1]), self.to_screen(k[b][0], k[b][1]))
        for j, (x, y, v) in enumerate(k):
            if v <= 0:
                continue
            c = self.to_screen(x, y)
            p.setPen(QPen(Qt.black, 1))
            p.setBrush(color)
            p.drawEllipse(c, 7, 7)
            p.setPen(Qt.white)
            p.drawText(QRectF(c.x() - 7, c.y() - 7, 14, 14), Qt.AlignCenter, str(j + 1))
        if selected:   # 선택된 박스 모서리 손잡이
            p.setBrush(Qt.white)
            p.setPen(QPen(color, 1))
            for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                p.drawRect(QRectF(self.to_screen(cx, cy) - QPointF(4, 4), self.to_screen(cx, cy) + QPointF(4, 4)))

    # ---------- 무엇을 잡았는지 찾기 ----------
    def _hit(self, pos):
        best = None
        for i, o in enumerate(self.label["objects"]):
            for j, (x, y, v) in enumerate(o["kpts"]):
                if v > 0 and (self.to_screen(x, y) - pos).manhattanLength() < GRAB:
                    return {"type": "point", "obj": i, "kpt": j}
            x1, y1, x2, y2 = o["box"]
            for c, (cx, cy) in enumerate(((x1, y1), (x2, y1), (x2, y2), (x1, y2))):
                if i == self.selected and (self.to_screen(cx, cy) - pos).manhattanLength() < GRAB:
                    return {"type": "corner", "obj": i, "corner": c}
            ip = self.to_image(pos)
            if x1 <= ip.x() <= x2 and y1 <= ip.y() <= y2 and best is None:
                best = {"type": "move", "obj": i}
        return best

    # ---------- 마우스 ----------
    def mousePressEvent(self, e):
        if not self.pixmap:
            return
        self.setFocus()
        pos = e.position()
        if e.button() == Qt.RightButton:
            self.drag = {"type": "pan", "last": pos}
            return
        if e.button() != Qt.LeftButton:
            return
        ip = self.to_image(pos)
        ip = QPointF(*self.clamp(ip.x(), ip.y()))
        if self.mode == "point" and self.selected >= 0:
            self.save_undo()
            o = self.label["objects"][self.selected]
            y = ip.y()
            if C.lower_half(o["cls"]) and self.next_point == 3 and abs(y - self.guide_y()) * self.scale < 25:
                y = self.guide_y()          # 기준선 근처(화면 25px 안)를 클릭했을 때만 기준선에 맞춤
            o["kpts"][self.next_point] = [ip.x(), y, 2]
            self._advance_point()
            return
        if self.mode == "edit":
            hit = self._hit(pos)
            if hit:
                self.save_undo()
                self.selected = hit["obj"]
                hit["last"] = ip
                self.drag = hit
                self.update()
                return
            self.selected = -1
            self.update()
            return
        if self.mode == "box":
            hit = self._hit(pos)
            if hit and hit["type"] in ("point", "corner"):   # 박스 그리기 중에도 키포인트, 모서리는 바로 끌어서 수정
                self.save_undo()
                self.selected = hit["obj"]
                hit["last"] = ip
                self.drag = hit
                self.update()
                return
            self.drag = {"type": "new_box", "start": ip, "now": ip}

    def mouseMoveEvent(self, e):
        if not self.drag:
            return
        pos = e.position()
        if self.drag["type"] == "pan":
            self.offset += pos - self.drag["last"]
            self.drag["last"] = pos
            self.update()
            return
        ip = self.to_image(pos)
        if self.drag["type"] != "move":
            ip = QPointF(*self.clamp(ip.x(), ip.y()))
        if self.drag["type"] == "new_box":
            self.drag["now"] = ip
        else:
            o = self.label["objects"][self.drag["obj"]]
            if self.drag["type"] == "point":
                o["kpts"][self.drag["kpt"]][:2] = [ip.x(), ip.y()]
            elif self.drag["type"] == "corner":
                b = o["box"]
                c = self.drag["corner"]
                if c in (0, 3):
                    b[0] = ip.x()
                else:
                    b[2] = ip.x()
                if c in (0, 1):
                    b[1] = ip.y()
                else:
                    b[3] = ip.y()
            elif self.drag["type"] == "move":
                dx, dy = ip.x() - self.drag["last"].x(), ip.y() - self.drag["last"].y()
                o["box"] = [o["box"][0] + dx, o["box"][1] + dy, o["box"][2] + dx, o["box"][3] + dy]
                for k in o["kpts"]:
                    if k[2] > 0:
                        k[0] += dx
                        k[1] += dy
                self.drag["last"] = ip
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.drag:
            return
        d, self.drag = self.drag, None
        if d["type"] == "new_box":
            a, b = d["start"], d["now"]
            if abs(a.x() - b.x()) * self.scale > 5 and abs(a.y() - b.y()) * self.scale > 5:
                self.save_undo()
                box = [min(a.x(), b.x()), min(a.y(), b.y()), max(a.x(), b.x()), max(a.y(), b.y())]
                obj = new_object(self.current_cls, box)
                if not self.clip_lower(obj):
                    self.hint_changed.emit(f"[박스 그리기] '{C.CLASS_NAMES[self.current_cls]}': "
                                           "기준선 아래 부분에 그리기")
                    self.update()
                    return
                self.label["objects"].append(obj)
                self.selected = len(self.label["objects"]) - 1
                self.next_point = 0
                # 'AI가 점 자동으로 찍기'가 켜져 있으면 AI가 점 4개를 찍고 수정 모드로
                if self.auto_points and self.auto_points(obj):
                    self.set_mode("edit")
                else:
                    self.set_mode("point")     # 아니면 직접 점 찍기로
                self.changed.emit()
        elif d["type"] in ("point", "corner", "move"):
            o = self.label["objects"][d["obj"]]
            x1, y1, x2, y2 = o["box"]
            o["box"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            self.clamp_object(o)
            if not self.clip_lower(o):         # 중앙분리대, 펜스 박스를 기준선 위로 옮기면 되돌림
                self.undo()
                self.hint_changed.emit("중앙분리대, 펜스 박스는 기준선 아래에만 (되돌림)")
                self.update()
                return
            self.changed.emit()
        self.update()

    def wheelEvent(self, e):
        if not self.pixmap:
            return
        pos = e.position()
        before = self.to_image(pos)
        self.scale *= 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.offset = pos - QPointF(before.x() * self.scale, before.y() * self.scale)
        self.update()

    def resizeEvent(self, e):
        self.fit()

    def _advance_point(self):
        o = self.label["objects"][self.selected]
        if C.lower_half(o["cls"]) and self.next_point == 0:
            self.next_point = 3                       # 1번 다음은 4번 (2, 3은 자동)
        elif C.lower_half(o["cls"]) and self.next_point == 3:
            straighten(o)                             # 2, 3 = 1-4 직선 3등분
            self.next_point = C.NUM_KEYPOINTS
        else:
            self.next_point += 1
        if self.next_point >= C.NUM_KEYPOINTS:
            self.set_mode("box")           # 4점 다 찍으면 다음 박스 그리기로
        else:
            self.hint_changed.emit(self.hint())
        self.changed.emit()
        self.update()
