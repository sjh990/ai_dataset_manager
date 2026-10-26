"""미리보기용 라벨 시각화."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .labels import Annotation


def class_color(c: int) -> tuple[int, int, int]:
    h = (c * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _ascii(label: str, fallback: str) -> str:
    return label if label.isascii() else fallback  # OpenCV 기본 폰트는 한글 미지원


def draw_kpts(out: np.ndarray, kpts: np.ndarray, t: int, skeleton=None) -> None:
    if skeleton:
        for a, b in skeleton:
            if a < len(kpts) and b < len(kpts) and kpts[a, 2] > 0 and kpts[b, 2] > 0:
                cv2.line(out, (int(kpts[a, 0]), int(kpts[a, 1])), (int(kpts[b, 0]), int(kpts[b, 1])),
                         (255, 255, 255), max(1, t), cv2.LINE_AA)
    for i, (x, y, v) in enumerate(kpts):
        if v > 0:
            c = class_color(i + 3)
            cv2.circle(out, (int(x), int(y)), t + 2, c if v >= 2 else (128, 128, 128), -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), t + 2, (0, 0, 0), 1, cv2.LINE_AA)


def draw(img: np.ndarray, anns: list[Annotation], task: str, names: list[str] | None = None,
         skeleton_pairs=None, image_class: str | None = None) -> np.ndarray:
    """task='classify'면 이미지 클래스만, 그 외엔 태스크 모양대로 라벨을 그린다.
    키포인트가 있는 객체는 태스크와 무관하게 점(+skeleton)을 그린다 (생성 프로젝트 JSON 라벨)."""
    out = img.copy()
    h, w = out.shape[:2]
    t = max(1, round(max(h, w) / 400))
    if image_class:
        lab = _ascii(image_class, "class")
        fs = 0.6 * t
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (40, 40, 40), -1)
        cv2.putText(out, lab, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 255, 255), 2, cv2.LINE_AA)
    if task == "classify":
        return out
    if task == "segment" and anns:
        overlay = out.copy()
        for a in anns:
            cv2.fillPoly(overlay, [np.round(a.pts).astype(np.int32)], class_color(a.cls))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
    for a in anns:
        col = class_color(a.cls)
        pts = np.round(a.pts).astype(np.int32)
        if task == "segment":
            cv2.polylines(out, [pts], True, col, t, cv2.LINE_AA)
        elif task == "obb":
            cv2.polylines(out, [pts], True, col, t + 1, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = map(int, map(round, a.bbox()))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, t + 1, cv2.LINE_AA)
        if a.kpts is not None:
            draw_kpts(out, a.kpts, t, skeleton_pairs)
        label = names[a.cls] if names and 0 <= a.cls < len(names) else str(a.cls)
        label = _ascii(label, f"{a.cls}")
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        fs = 0.4 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x, max(0, y - th - 4)), (x + tw + 4, max(th + 4, y)), col, -1)
        cv2.putText(out, label, (x + 2, max(th + 1, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return out
