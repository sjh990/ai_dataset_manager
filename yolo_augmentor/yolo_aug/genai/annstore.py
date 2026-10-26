"""데이터셋 생성용 내부 라벨 저장 형식 (이미지당 JSON 1개).

YOLO txt는 태스크마다 형식이 달라(박스/폴리곤/키포인트/분류 폴더) 한 번에 담을 수 없으므로,
생성 단계에서는 아래 JSON에 모두 보관하고 '내보내기'에서 선택한 YOLO 태스크로 변환한다.

{
  "w": 1280, "h": 720,
  "image_class": "damaged" | null,          # 이미지 분류(classify) 라벨
  "objects": [
    {"cls": 0,
     "pts":  [[x, y], ...],                  # 정규화(0~1) 폴리곤 (박스는 4점)
     "kpts": [[x, y, v], ...] | null,        # 정규화 키포인트, v: 0=없음 1=가려짐 2=보임
     "auto": false, "score": null}
  ]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..labels import Annotation, parse_labels


@dataclass
class ImageAnn:
    w: int
    h: int
    objects: list[Annotation] = field(default_factory=list)
    image_class: str | None = None
    image_class_auto: bool = False
    image_class_score: float | None = None


def to_dict(ia: ImageAnn) -> dict:
    s = np.array([ia.w, ia.h], np.float64)
    objs = []
    for a in ia.objects:
        o = {"cls": int(a.cls), "pts": np.round(a.pts / s, 6).tolist(), "kpts": None,
             "auto": bool(a.auto), "score": None if a.score is None else round(float(a.score), 4)}
        if a.kpts is not None:
            k = a.kpts.astype(np.float64).copy()
            k[:, 0] /= ia.w
            k[:, 1] /= ia.h
            k[k[:, 2] <= 0, :2] = 0
            o["kpts"] = np.round(k, 6).tolist()
        objs.append(o)
    d = {"w": int(ia.w), "h": int(ia.h), "image_class": ia.image_class, "objects": objs}
    if ia.image_class and ia.image_class_auto:
        d["image_class_auto"] = True
        if ia.image_class_score is not None:
            d["image_class_score"] = round(float(ia.image_class_score), 4)
    return d


def from_dict(d: dict, w: int | None = None, h: int | None = None) -> ImageAnn:
    W = int(w or d.get("w", 1))
    H = int(h or d.get("h", 1))
    s = np.array([W, H], np.float64)
    objs = []
    for o in d.get("objects", []):
        pts = np.asarray(o.get("pts", []), np.float64).reshape(-1, 2) * s
        if len(pts) < 3:
            continue
        k = o.get("kpts")
        kp = None
        if k is not None:
            kp = np.asarray(k, np.float64).reshape(-1, 3)
            kp[:, 0] *= W
            kp[:, 1] *= H
        objs.append(Annotation(int(o["cls"]), pts, kp, bool(o.get("auto", False)), o.get("score")))
    return ImageAnn(W, H, objs, d.get("image_class"), bool(d.get("image_class_auto", False)),
                    d.get("image_class_score"))


def load(path, w: int | None = None, h: int | None = None) -> ImageAnn | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return from_dict(json.loads(p.read_text(encoding="utf-8")), w, h)
    except Exception:
        return None


def save(path, ia: ImageAnn) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_dict(ia), ensure_ascii=False), encoding="utf-8")


def has_content(path) -> bool:
    ia = load(path)
    return bool(ia and (ia.objects or ia.image_class))


def load_any(label_path, task: str, w: int, h: int, kpt_shape=None) -> tuple[list[Annotation], str | None]:
    """JSON(생성 프로젝트) 또는 YOLO txt 라벨을 읽는다. 반환 (objects, image_class)."""
    if label_path is None:
        return [], None
    p = Path(label_path)
    if p.suffix.lower() == ".json":
        ia = load(p, w, h)
        return (ia.objects, ia.image_class) if ia else ([], None)
    return parse_labels(p, task, w, h, kpt_shape), None


def fit_kpts(anns: list[Annotation], k: int) -> None:
    """키포인트 개수를 스키마(k)에 맞춤 (부족하면 v=0으로 채우고, 넘치면 자름)."""
    for a in anns:
        if a.kpts is None:
            continue
        if len(a.kpts) < k:
            a.kpts = np.vstack([a.kpts, np.zeros((k - len(a.kpts), 3))])
        elif len(a.kpts) > k:
            a.kpts = a.kpts[:k]
