from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..genai import annstore
from ..genai.export import convert
from ..labels import format_labels


@dataclass
class Item:
    image: Path
    ann: annstore.ImageAnn


@dataclass
class TrainSet:
    root: Path
    names: list[str]
    n_train: int
    n_val: int
    n_objects: int
    per_class: dict = field(default_factory=dict)
    kpt_shape: list | None = None


def _has_kpts(a) -> bool:
    return a.kpts is not None and bool((a.kpts[:, 2] > 0).any())


def confirmed_items(project, task: str) -> list[Item]:
    out = []
    pose_ids = set(project.pose_class_ids)
    for img in project.images():
        ia = annstore.load(project.label_path(img))
        if ia is None:
            continue
        if task == "classify":
            if ia.image_class and not ia.image_class_auto:
                out.append(Item(img, ia))
            continue
        if not ia.objects or any(a.auto for a in ia.objects):
            continue
        if task == "pose" and not any(a.cls in pose_ids and _has_kpts(a) for a in ia.objects):
            continue
        out.append(Item(img, ia))
    return out


def summary(project) -> dict:
    res = {}
    names = project.class_names
    for task in ("detect", "pose", "classify"):
        items = confirmed_items(project, task)
        cnt = {}
        objs = 0
        for it in items:
            if task == "classify":
                cnt[it.ann.image_class] = cnt.get(it.ann.image_class, 0) + 1
                continue
            for a in it.ann.objects:
                if task == "pose" and not (a.cls in set(project.pose_class_ids) and _has_kpts(a)):
                    continue
                n = names[a.cls] if a.cls < len(names) else str(a.cls)
                cnt[n] = cnt.get(n, 0) + 1
                objs += 1
        res[task] = dict(images=len(items), objects=objs, per_class=cnt)
    res["segment"] = dict(res["detect"])
    res["segment"]["polygons"] = sum(
        1 for it in confirmed_items(project, "segment") for a in it.ann.objects if len(a.pts) > 4)
    res["unlabeled"] = sum(1 for img in project.images() if not project.is_labeled(img))
    res["total"] = len(project.images())
    return res


def split_items(items: list[Item], val_ratio: float, seed: int) -> tuple[list[Item], list[Item]]:
    if len(items) < 2:
        raise RuntimeError("At least 2 confirmed images are needed for training.")
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(items) * val_ratio)))
    n_val = min(n_val, len(items) - 1)
    val = [items[i] for i in idx[:n_val]]
    train = [items[i] for i in idx[n_val:]]
    return train, val


def _link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def build_yolo_set(project, items: list[Item], out: Path, task: str, val_ratio: float = 0.15, seed: int = 0,
                   log=print) -> TrainSet:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    pose_ids = sorted(set(project.pose_class_ids))
    K = len(project.kpt_names)
    used = set()
    for it in items:
        for a in it.ann.objects:
            if task == "pose" and a.cls not in pose_ids:
                continue
            used.add(a.cls)
    if task == "pose":
        used |= set(pose_ids)
    keep = sorted(used)
    if not keep:
        raise RuntimeError("No labeled objects for this task.")
    remap = {c: i for i, c in enumerate(keep)}
    names = [project.class_names[c] if c < len(project.class_names) else f"class_{c}" for c in keep]
    fmt = {"detect": "detect", "segment": "segment", "pose": "pose", "obb": "obb"}[task]
    train, val = split_items(items, val_ratio, seed)
    n_obj = 0
    per = {}
    for split, lst in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for it in lst:
            w, h = it.ann.w, it.ann.h
            anns = [a.copy() for a in it.ann.objects if a.cls in remap]
            for a in anns:
                a.cls = remap[a.cls]
            anns = convert(anns, fmt, K, w, h)
            _link(it.image, out / "images" / split / it.image.name)
            (out / "labels" / split / f"{it.image.stem}.txt").write_text(
                format_labels(anns, fmt, w, h, 3), encoding="utf-8")
            if split == "train":
                n_obj += len(anns)
                for a in anns:
                    per[names[a.cls]] = per.get(names[a.cls], 0) + 1
    d = {"path": str(out.resolve()).replace("\\", "/"), "train": "images/train", "val": "images/val",
         "nc": len(names), "names": {i: n for i, n in enumerate(names)}}
    kshape = None
    if task == "pose":
        kshape = [K, 3]
        d["kpt_shape"] = kshape
        d["flip_idx"] = list(project.flip_idx)
    (out / "data.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log(f"dataset: train {len(train)} / val {len(val)} images, {n_obj} objects, classes {names}")
    return TrainSet(out, names, len(train), len(val), n_obj, per, kshape)


def pose_samples(project, items: list[Item]) -> list[tuple[Path, tuple, np.ndarray]]:
    pose_ids = set(project.pose_class_ids)
    K = len(project.kpt_names)
    out = []
    for it in items:
        for a in it.ann.objects:
            if a.cls in pose_ids and _has_kpts(a):
                k = a.kpts
                if len(k) < K:
                    k = np.vstack([k, np.zeros((K - len(k), 3))])
                out.append((it.image, a.bbox(), k[:K].copy()))
    return out
