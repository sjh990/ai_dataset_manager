"""YOLO Pose 데이터셋으로 내보내기 (+ 증강).

- 사람이 검수 완료한 이미지만 내보냅니다.
- 결과: exports/<날짜>/images/{train,val}, labels/{train,val}, data.yaml, report.json
- 증강은 train 에만 적용합니다. 좌우 반전은 하지 않습니다 (점 순서 기준 유지).

YOLO Pose 라벨 한 줄: 클래스 cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4   (모두 0~1 비율)
"""
import json
import random
import shutil
import time

import cv2
import numpy as np
import yaml

from . import config as C
from .frames import read_image, save_image


def num(x):
    """숫자를 기존 툴과 똑같은 모양으로: 소수 6자리, 끝의 0 제거 (0.500000 → 0.5, 2.0 → 2)."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def yolo_line(obj, w, h, class_id):
    x1, y1, x2, y2 = obj["box"]
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))     # 이미지 밖 좌표는 YOLO 가 이미지를 버리므로 안으로
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    vals = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
    for x, y, v in obj["kpts"]:
        inside = 0 <= x <= w and 0 <= y <= h
        vals += [x / w, y / h, 2] if v > 0 and inside else [0, 0, 0]
    return f"{class_id} " + " ".join(num(v) for v in vals)


# 클래스 번호: 사람이 직접 지정. {툴의 클래스 순서: 내보낼 번호 또는 None(내보내지 않음)}
DEFAULT_CLASS_IDS = {i: i for i in range(len(C.CLASSES))}      # sign = 0, median = 1, fence = 2


def check_class_ids(class_ids):
    """번호 확인: 겹치지 않고 0부터 빈 번호 없이 이어져야 함 (YOLO 규칙). 문제 있으면 설명 글자, 없으면 None."""
    ids = [v for v in class_ids.values() if v is not None]
    if not ids:
        return "내보낼 클래스 없음 (하나 이상 번호 지정 필요)"
    if len(ids) != len(set(ids)):
        return "같은 번호가 2개 이상 (번호는 클래스마다 달라야 함)"
    if sorted(ids) != list(range(len(ids))):
        return f"번호는 0부터 빈 번호 없이 지정 (지금: {', '.join(map(str, sorted(ids)))} → 0 ~ {len(ids) - 1} 필요)"
    return None

# 기존 툴 data.yaml 의 kpt_names 와 같은 이름 (영문)
KPT_NAMES = {"sign": ["base", "middle", "top", "tip"], "median": ["start", "mid1", "mid2", "end"],
             "fence": ["start", "mid1", "mid2", "end"]}




def export(project, val_ratio=0.15, aug_copies=2, max_background_percent=10, class_ids=None, progress=None,
           log=print, aug_settings=None):
    """aug_settings: 증강 설정 (labeler/augment.py, None = '보통')."""
    from . import augment as A
    aug_settings = aug_settings or A.preset("보통")
    class_ids = DEFAULT_CLASS_IDS if class_ids is None else class_ids
    problem = check_class_ids(class_ids)
    if problem:
        raise RuntimeError(f"클래스 번호 확인: {problem}")
    rng = random.Random(0)
    items = []
    skipped = 0
    for img, lab in project.human_images():
        kept = [o for o in lab["objects"] if class_ids.get(o["cls"]) is not None]
        if lab["objects"] and not kept:        # 내보내지 않는 클래스만 있는 이미지: 배경으로 넣지 않고 제외
            skipped += 1
            continue
        items.append((img, {**lab, "objects": kept}))
    with_obj = [it for it in items if it[1]["objects"]]
    background = [it for it in items if not it[1]["objects"]]
    rng.shuffle(background)
    max_bg = int(len(with_obj) * max_background_percent / max(1, 100 - max_background_percent))
    background = background[:max_bg]
    items = with_obj + background
    if not with_obj:
        raise RuntimeError("시설물이 있는 '사람 검수 완료' 이미지 없음")
    if len(with_obj) + len(background) < 2:
        raise RuntimeError("내보내려면 '사람 검수 완료' 이미지 2장 이상 필요 (train, val 각 1장 이상)")
    rng.shuffle(items)
    n_val = max(1, round(len(items) * val_ratio))
    splits = {"val": items[:n_val], "train": items[n_val:]}

    out = project.exports_dir / time.strftime("%Y%m%d_%H%M%S")
    for s in splits:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    cmap = {cls: i for cls, i in class_ids.items() if i is not None}
    names = [C.CLASS_KEYS[cls] for cls, _ in sorted(cmap.items(), key=lambda kv: kv[1])]   # 번호 순서

    def write_label(path, objs, w, h):
        lines = [yolo_line(o, w, h, cmap[o["cls"]]) for o in objs]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    counts = {"train": 0, "val": 0, "train_augmented": 0}
    done, total = 0, len(items)
    for split, part in splits.items():
        for img_path, lab in part:
            # 원본은 다시 압축하지 않고 파일 그대로 복사 (이미지 값이 데이터셋 원본과 완전히 같게)
            shutil.copy2(img_path, out / "images" / split / img_path.name)
            write_label(out / "labels" / split / f"{img_path.stem}.txt", lab["objects"], lab["width"], lab["height"])
            if split == "train" and aug_copies:
                img = read_image(img_path)
                for k in range(aug_copies):
                    a_img, a_objs = A.augment(img, lab["objects"], aug_settings, rng)
                    stem = f"{img_path.stem}_aug{k + 1}"
                    save_image(out / "images" / split / f"{stem}.jpg", a_img)
                    write_label(out / "labels" / split / f"{stem}.txt", a_objs, a_img.shape[1], a_img.shape[0])
            counts[split] += 1
            if split == "train":
                counts["train_augmented"] += aug_copies
            done += 1
            if progress:
                progress(done, total, f"내보내기 {done}/{total}")

    # data.yaml: 기존 툴과 같은 항목 (nc, names, kpt_shape, flip_idx, kpt_names)
    # path 를 넣지 않음 → YOLO 가 data.yaml 이 있는 폴더 기준으로 찾음 (폴더를 옮기거나 팀원에게 줘도 동작)
    data = {"train": "images/train", "val": "images/val",
            "nc": len(names), "names": dict(enumerate(names)),
            "kpt_shape": [C.NUM_KEYPOINTS, 3], "flip_idx": list(range(C.NUM_KEYPOINTS)),
            "kpt_names": {i: KPT_NAMES[n] for i, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    report = {"created": time.strftime("%Y-%m-%d %H:%M"), "images_with_objects": len(with_obj),
              "background_images": len(background), **counts, "aug_copies_per_image": aug_copies, "augment": aug_settings,
              "class_ids": dict(enumerate(names)), "skipped_images_only_excluded_classes": skipped,
              "classes": dict(zip(C.CLASS_KEYS, C.CLASS_NAMES))}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"내보내기 완료: {out}\n train 원본 {counts['train']}장 + 증강 {counts['train_augmented']}장, "
        f"val {counts['val']}장 (시설물 {len(with_obj)}장, 배경 {len(background)}장)\n"
        f" 클래스 번호: " + ", ".join(f"{i} = {n}" for i, n in enumerate(names)) + f" (nc = {len(names)})"
        + (f"\n 내보내지 않는 클래스만 있어 제외한 이미지: {skipped}장" if skipped else ""))
    return out
