"""프로젝트 → YOLO 데이터셋 내보내기 (태스크는 여기서 결정).

  detect   : images/ + labels/ (cls cx cy w h)
  segment  : images/ + labels/ (cls x1 y1 ... 폴리곤)         — 인스턴스 세그멘테이션
  obb      : images/ + labels/ (cls 4꼭짓점, 최소 외접 회전 사각형)
  pose     : images/ + labels/ (cls cx cy w h + K×(x y v)), data.yaml에 kpt_shape·flip_idx
  classify : <클래스>/<이미지> 폴더 구조 (Ultralytics classify)

- 원본(라벨된 work 이미지) + 선택한 생성 실행의 '통과' 결과를 하나로 합침
- 실행마다 클래스 목록이 늘었을 수 있으므로(파손 재매핑 등) 이름 기준으로 id 통일
- 결과는 분할 전 평면 구조 → ② 증강 탭에서 train/val/test 분할·증강
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..io_utils import image_size
from ..labels import Annotation, box_corners, format_labels
from . import annstore
from .project import FORMATS, Project


def _obb(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    r = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))).astype(np.float64)
    tol = 1.0
    if r[:, 0].min() < -tol or r[:, 1].min() < -tol or r[:, 0].max() > w + tol or r[:, 1].max() > h + tol:
        # 회전 사각형이 이미지 밖으로 나가면 잘렸을 때 직사각형이 깨지므로 축정렬 박스로 대체
        x1, y1 = np.clip(pts.min(0), 0, [w, h])
        x2, y2 = np.clip(pts.max(0), 0, [w, h])
        return box_corners(x1, y1, x2, y2)
    return r


def convert(anns: list[Annotation], fmt: str, kpt_k: int = 0, w: int = 0, h: int = 0) -> list[Annotation]:
    out = []
    for a in anns:
        if len(a.pts) < 3:
            continue
        if fmt in ("detect", "pose"):
            b = Annotation(a.cls, box_corners(*a.bbox()))
            if fmt == "pose":
                k = a.kpts.copy() if a.kpts is not None else np.zeros((kpt_k, 3))
                if len(k) < kpt_k:
                    k = np.vstack([k, np.zeros((kpt_k - len(k), 3))])
                b.kpts = k[:kpt_k]
            out.append(b)
        elif fmt == "obb":  # OBB는 회전 '직사각형'이어야 하므로 항상 최소 외접 회전 사각형으로 변환
            out.append(Annotation(a.cls, _obb(a.pts, w, h) if w else
                                  cv2.boxPoints(cv2.minAreaRect(a.pts.astype(np.float32))).astype(np.float64)))
        else:
            out.append(Annotation(a.cls, a.pts.copy()))
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "unknown"


def export_dataset(project: Project, out_dir, fmt: str | None = None, include_originals: bool = True,
                   include_unlabeled: bool = False, runs: list | None = None, progress=None, log=print,
                   cancel=None, pose_classes_only: bool = True, pose_require_kpts: bool = False) -> dict:
    """pose_classes_only: pose 내보내기 시 키포인트 스키마 대상 클래스만 포함.
    pose_require_kpts: 키포인트가 하나도 없는 객체는 제외."""
    fmt = fmt or project.fmt
    if fmt not in FORMATS:
        raise ValueError(f"지원하지 않는 포맷: {fmt}")
    progress = progress or (lambda *a: None)
    out = Path(out_dir)
    names = list(project.class_names)
    img_names = list(project.image_class_names)
    K = len(project.kpt_names)
    if fmt == "pose" and K == 0:
        raise RuntimeError("Pose로 내보내려면 프로젝트에 키포인트 스키마를 먼저 정의하세요 (1. 프로젝트 · 원본).")
    pose_ids = None
    pose_names = {names[i] for i in project.pose_class_ids}

    def remap_fn(run_names):
        m = {}
        for i, n in enumerate(run_names):
            if n not in names:
                names.append(n)
            m[i] = names.index(n)
        return m

    jobs = []
    if include_originals:
        for s in project.samples():
            if (s.label and annstore.has_content(s.label)) or include_unlabeled:
                jobs.append(("orig", s.image, s.label, None))
    run_dirs = [Path(r) for r in (runs if runs is not None else project.runs())]
    for rd in run_dirs:
        rep = json.loads((rd / "gen_report.json").read_text(encoding="utf-8"))
        m = remap_fn(rep.get("class_names", names))
        for p in sorted((rd / "images").glob("*")):
            if p.is_file():
                lab = rd / "ann" / f"{p.stem}.json"
                if not lab.exists():
                    lab = rd / "labels" / f"{p.stem}.txt"
                jobs.append((rd.name, p, lab, m))

    if fmt == "pose" and pose_classes_only:
        # 스키마 대상 클래스 + 키포인트가 실제로 달린 클래스(예: 변형 후 재매핑된 traffic_light_fallen)
        for src, img, lab, m in jobs:
            if not lab or Path(lab).suffix != ".json" or not Path(lab).exists():
                continue
            ia = annstore.load(lab)
            for a in (ia.objects if ia else []):
                if a.kpts is not None and (a.kpts[:, 2] > 0).any():
                    cid = m.get(a.cls, a.cls) if m else a.cls
                    if cid < len(names):
                        pose_names.add(names[cid])
        if pose_names:
            pose_ids = {names.index(n) for n in pose_names}
    if fmt == "classify":
        out.mkdir(parents=True, exist_ok=True)
    else:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
    counts = {"orig": 0, "generated": 0, "objects": 0, "skipped": 0}
    per_class: dict[str, int] = {}
    for k, (src, img, lab, m) in enumerate(jobs):
        if cancel is not None and cancel.is_set():
            break
        sz = image_size(img)
        if not sz:
            continue
        w, h = sz
        anns, img_cls = annstore.load_any(lab, "segment", w, h) if lab and Path(lab).exists() else ([], None)
        stem = img.stem if src == "orig" else f"{src}__{img.stem}"
        if fmt == "classify":
            if not img_cls:
                counts["skipped"] += 1
                continue
            if img_cls not in img_names:
                img_names.append(img_cls)
            d = out / _safe(img_cls)
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, d / f"{stem}{img.suffix.lower()}")
            per_class[img_cls] = per_class.get(img_cls, 0) + 1
        else:
            if m:
                for a in anns:
                    a.cls = m.get(a.cls, a.cls)
            if pose_ids is not None:
                anns = [a for a in anns if a.cls in pose_ids]
            if fmt == "pose" and pose_require_kpts:
                anns = [a for a in anns if a.kpts is not None and (a.kpts[:, 2] > 0).any()]
            anns = convert(anns, fmt, K, w, h)
            shutil.copy2(img, out / "images" / f"{stem}{img.suffix.lower()}")
            (out / "labels" / f"{stem}.txt").write_text(format_labels(anns, fmt, w, h, 3), encoding="utf-8")
            counts["objects"] += len(anns)
            for a in anns:
                nm = names[a.cls] if a.cls < len(names) else str(a.cls)
                per_class[nm] = per_class.get(nm, 0) + 1
        counts["orig" if src == "orig" else "generated"] += 1
        progress(k + 1, len(jobs), img.name)

    if fmt == "classify":
        (out / "classes.txt").write_text("\n".join(sorted(per_class)) + "\n", encoding="utf-8")
    else:
        used = names
        if pose_ids is not None:  # pose: 대상 클래스만 id 재배열
            keep = sorted(pose_ids)
            rem = {old: new for new, old in enumerate(keep)}
            for f in (out / "labels").glob("*.txt"):
                lines = []
                for ln in f.read_text(encoding="utf-8").splitlines():
                    parts = ln.split()
                    if parts and int(parts[0]) in rem:
                        parts[0] = str(rem[int(parts[0])])
                        lines.append(" ".join(parts))
                f.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            used = [names[i] for i in keep]
        d = {"path": str(out.resolve()).replace("\\", "/"), "train": "images", "val": "images",
             "nc": len(used), "names": {i: n for i, n in enumerate(used)}}
        if fmt == "pose":
            d["kpt_shape"] = [K, 3]
            d["flip_idx"] = list(project.flip_idx)
            d["kpt_names"] = {0: project.kpt_names}
        (out / "data.yaml").write_text(
            f"# Exported by YOLO Dataset Augmentor — dataset creation ({fmt})\n"
            "# ② 데이터셋 증강 탭에서 train/val/test 분할과 증강을 진행하세요.\n"
            + yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (out / "classes.txt").write_text("\n".join(used) + "\n", encoding="utf-8")
        names_out = used
    summary = dict(format=fmt, created=time.strftime("%Y-%m-%d %H:%M"), project=str(project.root),
                   runs=[r.name for r in run_dirs], class_names=names if fmt != "classify" else sorted(per_class),
                   per_class=per_class, **counts)
    if fmt != "classify":
        summary["class_names"] = names_out
    (out / "export_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for n in names:
        project.class_id(n, add=True)
    for n in img_names:
        project.add_image_class(n)
    project.save()
    extra = f", 분류 라벨 없음 {counts['skipped']}장 제외" if counts["skipped"] else ""
    log(f"내보내기 완료({fmt}): 원본 {counts['orig']} + 생성 {counts['generated']}장, 객체 {counts['objects']}개{extra} → {out}")
    return summary
