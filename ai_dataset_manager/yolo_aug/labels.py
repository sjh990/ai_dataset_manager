"""YOLO 라벨 표현·파싱·저장·기하변환.

모든 태스크의 라벨을 '픽셀 좌표 점 집합'으로 통일해서 다룬다.
  - detect : pts = bbox 4꼭짓점
  - segment: pts = 폴리곤
  - obb    : pts = 회전 박스 4꼭짓점
  - pose   : pts = bbox 4꼭짓점, kpts = (K,3) [x, y, v]
기하 변환은 3x3 행렬 하나로 점에 적용한 뒤 태스크별 규칙으로 클리핑한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

TASKS = ("detect", "segment", "pose", "obb", "classify")
TASK_LABELS = {
    "detect": "Detection (bbox)",
    "segment": "Segmentation (polygon)",
    "pose": "Pose (keypoints)",
    "obb": "OBB (회전 박스)",
    "classify": "Classification (이미지 분류)",
}


@dataclass
class Annotation:
    cls: int
    pts: np.ndarray  # (N,2) float64, pixel
    kpts: np.ndarray | None = None  # (K,3) pixel x,y,v
    auto: bool = False               # 자동 라벨(미확정) 여부
    score: float | None = None

    def copy(self) -> "Annotation":
        return Annotation(self.cls, self.pts.copy(), None if self.kpts is None else self.kpts.copy(),
                          self.auto, self.score)

    def bbox(self) -> tuple[float, float, float, float]:
        x1, y1 = self.pts.min(axis=0)
        x2, y2 = self.pts.max(axis=0)
        return float(x1), float(y1), float(x2), float(y2)


@dataclass
class ParseStats:
    lines: int = 0
    bad: int = 0
    messages: list[str] = field(default_factory=list)


def box_corners(x1, y1, x2, y2) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


# ----------------------------------------------------------------------------- parse / write
def parse_line_values(line: str) -> list[float] | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        return [float(v) for v in parts]
    except ValueError:
        return []


def parse_labels(
    path: Path | None,
    task: str,
    w: int,
    h: int,
    kpt_shape: tuple[int, int] | None = None,
    stats: ParseStats | None = None,
) -> list[Annotation]:
    anns: list[Annotation] = []
    if path is None or not Path(path).exists():
        return anns
    stats = stats or ParseStats()
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        vals = parse_line_values(raw)
        if vals is None:
            continue
        stats.lines += 1
        ann = _values_to_ann(vals, task, w, h, kpt_shape)
        if ann is None:
            stats.bad += 1
            if len(stats.messages) < 5:
                stats.messages.append(f"{Path(path).name}: 해석 불가 라인 '{raw.strip()[:60]}'")
            continue
        anns.append(ann)
    return anns


def _values_to_ann(vals, task, w, h, kpt_shape) -> Annotation | None:
    if len(vals) < 5:
        return None
    cls = int(vals[0])
    v = np.asarray(vals[1:], dtype=np.float64)
    scale = np.array([w, h], dtype=np.float64)

    def xywh_box(a):
        cx, cy, bw, bh = a[:4]
        return box_corners((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)

    if task == "detect":
        if len(v) == 4:
            return Annotation(cls, xywh_box(v))
        if len(v) % 2 == 0 and len(v) >= 6:  # 폴리곤이 섞여 있으면 bbox로
            pts = v.reshape(-1, 2) * scale
            return Annotation(cls, box_corners(*pts.min(0), *pts.max(0)))
        return Annotation(cls, xywh_box(v))
    if task == "segment":
        if len(v) == 4:
            return Annotation(cls, xywh_box(v))
        if len(v) % 2 != 0 or len(v) < 6:
            return None
        return Annotation(cls, v.reshape(-1, 2) * scale)
    if task == "obb":
        if len(v) == 8:
            return Annotation(cls, v.reshape(4, 2) * scale)
        if len(v) == 4:
            return Annotation(cls, xywh_box(v))
        return None
    if task == "pose":
        k, d = kpt_shape if kpt_shape else (0, 3)
        if k == 0:
            rest = len(v) - 4
            d = 3 if rest % 3 == 0 else 2
            k = rest // d
        if len(v) != 4 + k * d:
            return None
        box = xywh_box(v)
        kp = v[4:].reshape(k, d)
        out = np.zeros((k, 3), dtype=np.float64)
        out[:, 0] = kp[:, 0] * w
        out[:, 1] = kp[:, 1] * h
        if d == 3:
            out[:, 2] = kp[:, 2]
        else:
            out[:, 2] = np.where((kp[:, 0] == 0) & (kp[:, 1] == 0), 0, 2)
        out[out[:, 2] <= 0] = 0
        return Annotation(cls, box, out)
    return None


def format_labels(anns: list[Annotation], task: str, w: int, h: int, kpt_dim: int = 3) -> str:
    lines = []
    for a in anns:
        if task in ("detect", "pose"):
            x1, y1, x2, y2 = a.bbox()
            vals = [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h]
            if task == "pose" and a.kpts is not None:
                for x, y, vis in a.kpts:
                    if vis <= 0:
                        vals += [0.0, 0.0] + ([0.0] if kpt_dim == 3 else [])
                    else:
                        vals += [x / w, y / h] + ([float(vis)] if kpt_dim == 3 else [])
        else:
            p = a.pts / np.array([w, h])
            vals = list(np.clip(p, 0, 1).reshape(-1))
        lines.append(f"{a.cls} " + " ".join(_fmt(x) for x in vals))
    return "\n".join(lines) + ("\n" if lines else "")


def _fmt(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


# ----------------------------------------------------------------------------- geometry
def warp_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    wz = hom[:, 2:3].copy()
    wz[np.abs(wz) < 1e-9] = 1e-9
    return hom[:, :2] / wz


def polygon_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def clip_polygon(poly: np.ndarray, rect) -> np.ndarray:
    """Sutherland–Hodgman: 폴리곤을 축정렬 사각형으로 자른다."""
    x0, y0, x1, y1 = rect
    pts = [tuple(p) for p in poly]

    def clip(points, axis, bound, keep_greater):
        if not points:
            return []

        def inside(p):
            return p[axis] >= bound if keep_greater else p[axis] <= bound

        def inter(p, q):
            t = (bound - p[axis]) / (q[axis] - p[axis])
            r = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
            r[axis] = bound
            return tuple(r)

        out = []
        prev = points[-1]
        for cur in points:
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
            prev = cur
        return out

    pts = clip(pts, 0, x0, True)
    pts = clip(pts, 0, x1, False)
    pts = clip(pts, 1, y0, True)
    pts = clip(pts, 1, y1, False)
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def transform_annotations(
    anns: list[Annotation],
    M: np.ndarray,
    rect,
    task: str,
    min_visibility: float = 0.25,
    min_size: float = 2.0,
    flip_idx: list[int] | None = None,
) -> tuple[list[Annotation], int]:
    """행렬 M 적용 후 rect(x0,y0,x1,y1)로 클리핑. (결과, 제거된 개수) 반환."""
    x0, y0, x1, y1 = rect
    reflect = np.linalg.det(M[:2, :2]) < 0
    out: list[Annotation] = []
    dropped = 0
    for a in anns:
        pts = warp_points(a.pts, M)
        if not np.all(np.isfinite(pts)):
            dropped += 1
            continue
        res: Annotation | None = None
        if task in ("detect", "pose"):
            bx1, by1 = pts.min(0)
            bx2, by2 = pts.max(0)
            area0 = max((bx2 - bx1) * (by2 - by1), 1e-9)
            cx1, cy1 = max(bx1, x0), max(by1, y0)
            cx2, cy2 = min(bx2, x1), min(by2, y1)
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw >= min_size and ch >= min_size and cw * ch >= min_visibility * area0:
                res = Annotation(a.cls, box_corners(cx1, cy1, cx2, cy2))
                if task == "pose" and a.kpts is not None:
                    k = a.kpts.copy()
                    vis = k[:, 2] > 0
                    if vis.any():
                        k[vis, :2] = warp_points(k[vis, :2], M)
                    inside = (k[:, 0] >= x0) & (k[:, 0] <= x1) & (k[:, 1] >= y0) & (k[:, 1] <= y1)
                    k[~(vis & inside)] = 0
                    if reflect and flip_idx and len(flip_idx) == len(k):
                        k = k[flip_idx]
                    res.kpts = k
        else:
            area0 = polygon_area(pts)
            if area0 <= 1e-6:
                dropped += 1
                continue
            clipped = clip_polygon(pts, rect)
            area1 = polygon_area(clipped)
            if len(clipped) >= 3 and area1 >= min_visibility * area0:
                if task == "obb" and (len(clipped) != 4 or area1 < 0.999 * area0):
                    (cx, cy), (rw, rh), ang = cv2.minAreaRect(clipped.astype(np.float32))
                    clipped = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float64)
                    clipped[:, 0] = np.clip(clipped[:, 0], x0, x1)
                    clipped[:, 1] = np.clip(clipped[:, 1], y0, y1)
                bw = clipped[:, 0].max() - clipped[:, 0].min()
                bh = clipped[:, 1].max() - clipped[:, 1].min()
                if bw >= min_size and bh >= min_size:
                    res = Annotation(a.cls, clipped)
        if res is None:
            dropped += 1
        else:
            out.append(res)
    return out, dropped


def filter_small(anns: list[Annotation], min_w: float, min_h: float) -> tuple[list[Annotation], int]:
    keep = []
    for a in anns:
        x1, y1, x2, y2 = a.bbox()
        if (x2 - x1) >= min_w and (y2 - y1) >= min_h:
            keep.append(a)
    return keep, len(anns) - len(keep)
