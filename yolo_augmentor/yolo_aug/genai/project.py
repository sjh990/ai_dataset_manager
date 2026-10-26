"""데이터셋 생성 프로젝트: 원본 이미지(가져오기 또는 텍스트→이미지 생성) → 라벨링 → 변형·합성 → 내보내기.

폴더 구조
  <project>/project.json          객체 클래스, 이미지 분류 클래스, 키포인트 스키마, 내보낼 태스크, 사용자 효과
  <project>/work/images/          원본 이미지 (가져온 것 + AI로 생성한 것)
  <project>/work/ann/<stem>.json  라벨 (폴리곤 + 키포인트 + 이미지 분류) → annstore 참고
  <project>/generated/<run>/      변형·합성 실행 결과 (images, ann, _rejected, gen_report.json)
  <project>/export/<name>/        내보낸 YOLO 데이터셋 (→ ② 증강 탭 입력)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ..dataset import Sample
from ..io_utils import IMG_EXTS, image_size
from ..labels import Annotation, parse_labels
from . import annstore
from .annstore import ImageAnn

FORMATS = {
    "detect": "Detection (bbox)",
    "segment": "Instance Segmentation (polygon)",
    "obb": "OBB (회전 박스)",
    "pose": "Pose (자세 추정 · 키포인트)",
    "classify": "Classification (이미지 분류)",
}
PROJECT_FILE = "project.json"

# 키포인트 스키마 프리셋 (YOLO pose: 데이터셋 전체가 하나의 kpt_shape를 공유)
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle"]
COCO17_SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                   [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
KPT_PRESETS = {
    "none": dict(title="사용 안 함", names=[], skeleton=[]),
    "coco17": dict(title="사람 17점 (COCO) — 자동 라벨(ViTPose) 지원", names=COCO17, skeleton=COCO17_SKELETON),
    "person_simple": dict(title="사람 간이 5점 (머리·양손·양발)",
                          names=["head", "left_hand", "right_hand", "left_foot", "right_foot"],
                          skeleton=[[0, 1], [0, 2], [0, 3], [0, 4]]),
    "pole4": dict(title="기둥/구조물 4점 (밑둥·중간·꼭대기·끝단)",
                  names=["base", "middle", "top", "tip"], skeleton=[[0, 1], [1, 2], [2, 3]]),
    "vehicle4": dict(title="차량 4점 (앞왼·앞오·뒤왼·뒤오 바퀴)",
                     names=["front_left", "front_right", "rear_left", "rear_right"],
                     skeleton=[[0, 1], [1, 3], [3, 2], [2, 0]]),
    "custom": dict(title="직접 정의", names=[], skeleton=[]),
}


def flip_idx_from_names(names: list[str]) -> list[int]:
    """left_/right_ 짝을 자동으로 찾아 좌우 반전 교환 인덱스 생성."""
    idx = list(range(len(names)))
    for i, n in enumerate(names):
        for a, b in (("left", "right"), ("right", "left"), ("l_", "r_"), ("r_", "l_")):
            if a in n:
                m = n.replace(a, b, 1)
                if m in names:
                    idx[i] = names.index(m)
                    break
    return idx


@dataclass
class ClassDef:
    name: str
    prompt: str = ""   # 자동 라벨링용 영어 문구 (예: "traffic light")


class Project:
    def __init__(self, root: Path, data: dict):
        self.root = Path(root)
        self.data = data

    # ------------------------------------------------------------------ io
    @classmethod
    def create(cls, root, fmt: str = "detect", classes: list[ClassDef] | None = None) -> "Project":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        data = dict(version=2, created=time.strftime("%Y-%m-%d %H:%M"), format=fmt, sources=[],
                    classes=[dict(name=c.name, prompt=c.prompt) for c in (classes or [])], custom_effects=[],
                    image_classes=[], keypoints=dict(preset="none", names=[], skeleton=[], classes=[]))
        p = cls(root, data)
        p.work_images.mkdir(parents=True, exist_ok=True)
        p.work_ann.mkdir(parents=True, exist_ok=True)
        p.save()
        return p

    @classmethod
    def load(cls, root) -> "Project":
        root = Path(root)
        data = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
        for k, v in (("custom_effects", []), ("classes", []), ("image_classes", []),
                     ("keypoints", dict(preset="none", names=[], skeleton=[], classes=[]))):
            data.setdefault(k, v)
        p = cls(root, data)
        p._migrate_txt()
        return p

    @staticmethod
    def exists(root) -> bool:
        return (Path(root) / PROJECT_FILE).exists()

    def save(self):
        (self.root / PROJECT_FILE).write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _migrate_txt(self):
        """v1.2 프로젝트(work/labels/*.txt 폴리곤) → JSON."""
        old = self.work / "labels"
        if not old.is_dir():
            return
        for img in self.images():
            t = old / f"{img.stem}.txt"
            if t.exists() and not self.label_path(img).exists():
                sz = image_size(img)
                if sz:
                    self.save_image_ann(img, ImageAnn(sz[0], sz[1], parse_labels(t, "segment", *sz)))
        shutil.move(str(old), str(self.work / "labels_v12_backup"))
        self.data["version"] = 2
        self.save()

    # ------------------------------------------------------------------ props
    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def work_images(self) -> Path:
        return self.work / "images"

    @property
    def work_ann(self) -> Path:
        return self.work / "ann"

    @property
    def fmt(self) -> str:
        return self.data.get("format", "detect")

    @fmt.setter
    def fmt(self, v):
        self.data["format"] = v

    @property
    def classes(self) -> list[ClassDef]:
        return [ClassDef(c["name"], c.get("prompt", "")) for c in self.data["classes"]]

    @classes.setter
    def classes(self, lst: list[ClassDef]):
        self.data["classes"] = [dict(name=c.name, prompt=c.prompt) for c in lst]

    @property
    def class_names(self) -> list[str]:
        return [c["name"] for c in self.data["classes"]]

    def class_id(self, name: str, add: bool = True) -> int:
        names = self.class_names
        if name in names:
            return names.index(name)
        if not add:
            return -1
        self.data["classes"].append(dict(name=name, prompt=""))
        return len(names)

    # 이미지 분류 클래스
    @property
    def image_classes(self) -> list[ClassDef]:
        return [ClassDef(c["name"], c.get("prompt", "")) for c in self.data["image_classes"]]

    @image_classes.setter
    def image_classes(self, lst: list[ClassDef]):
        self.data["image_classes"] = [dict(name=c.name, prompt=c.prompt) for c in lst]

    @property
    def image_class_names(self) -> list[str]:
        return [c["name"] for c in self.data["image_classes"]]

    def add_image_class(self, name: str):
        if name and name not in self.image_class_names:
            self.data["image_classes"].append(dict(name=name, prompt=name.replace("_", " ")))

    # 키포인트 스키마
    @property
    def kpt(self) -> dict:
        return self.data["keypoints"]

    @property
    def kpt_names(self) -> list[str]:
        return list(self.kpt.get("names", []))

    @property
    def kpt_skeleton(self) -> list:
        return list(self.kpt.get("skeleton", []))

    @property
    def pose_class_ids(self) -> list[int]:
        names = self.class_names
        return [names.index(n) for n in self.kpt.get("classes", []) if n in names]

    @property
    def flip_idx(self) -> list[int]:
        return self.kpt.get("flip_idx") or flip_idx_from_names(self.kpt_names)

    def set_keypoints(self, preset: str, names: list[str], skeleton: list, classes: list[str],
                      flip_idx: list[int] | None = None):
        self.data["keypoints"] = dict(preset=preset, names=list(names), skeleton=[list(e) for e in skeleton],
                                      classes=list(classes), flip_idx=flip_idx or flip_idx_from_names(names))

    # ------------------------------------------------------------------ images
    def import_images(self, src, recursive=True, progress=None, log=print, cancel=None) -> int:
        """원본 이미지 폴더를 work/images로 복사 (같은 파일명은 접미사로 구분). 옆에 YOLO 라벨(txt)이
        있으면 함께 가져온다. 하위 폴더 이름이 이미지 분류 클래스처럼 보이면(옵션) 분류 라벨로 사용."""
        src = Path(src)
        files = [p for p in (src.rglob("*") if recursive else src.iterdir())
                 if p.is_file() and p.suffix.lower() in IMG_EXTS and self.root not in p.parents]
        existing = {p.stem for p in self.work_images.iterdir()} if self.work_images.exists() else set()
        n = 0
        for i, f in enumerate(sorted(files)):
            if cancel is not None and cancel.is_set():
                break
            stem = f.stem
            k = 1
            while stem in existing:
                stem = f"{f.stem}_{k}"
                k += 1
            existing.add(stem)
            dst = self.work_images / f"{stem}{f.suffix.lower()}"
            try:
                os.link(f, dst)  # 같은 드라이브면 하드링크(용량 절약)
            except Exception:
                shutil.copy2(f, dst)
            for cand in (f.with_suffix(".txt"), Path(str(f.parent).replace("images", "labels")) / f"{f.stem}.txt"):
                if cand.exists():
                    self._import_label(cand, dst)
                    break
            n += 1
            if progress:
                progress(i + 1, len(files), f.name)
        self.data["sources"] = sorted(set(self.data.get("sources", [])) | {str(src)})
        self.save()
        log(f"이미지 {n}장 가져옴 → {self.work_images}")
        return n

    def _import_label(self, lab: Path, img: Path):
        sz = image_size(img)
        if not sz:
            return
        w, h = sz
        lines = [ln.split() for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            return
        lens = {len(v) for v in lines}
        task = "detect" if lens == {5} else ("obb" if lens == {9} else "segment")
        self.save_image_ann(img, ImageAnn(w, h, parse_labels(lab, task, w, h)))

    def add_generated_image(self, bgr, ia: ImageAnn, stem: str, ext: str = ".jpg") -> Path:
        from ..io_utils import imwrite

        self.work_images.mkdir(parents=True, exist_ok=True)
        existing = {p.stem for p in self.work_images.iterdir()}
        s, k = stem, 1
        while s in existing:
            s = f"{stem}_{k}"
            k += 1
        dst = self.work_images / f"{s}{ext}"
        imwrite(dst, bgr, 95)
        self.save_image_ann(dst, ia)
        return dst

    def images(self) -> list[Path]:
        if not self.work_images.exists():
            return []
        return sorted(p for p in self.work_images.iterdir() if p.suffix.lower() in IMG_EXTS)

    def label_path(self, img: Path) -> Path:
        return self.work_ann / f"{Path(img).stem}.json"

    def load_image_ann(self, img: Path, w: int | None = None, h: int | None = None) -> ImageAnn:
        ia = annstore.load(self.label_path(img), w, h)
        if ia is None:
            if w is None:
                sz = image_size(img) or (1, 1)
                w, h = sz
            ia = ImageAnn(w, h)
        return ia

    def save_image_ann(self, img, ia: ImageAnn):
        annstore.save(self.label_path(img), ia)

    # 호환용 (객체 라벨만)
    def load_anns(self, img: Path, w: int, h: int) -> list[Annotation]:
        return self.load_image_ann(img, w, h).objects

    def save_anns(self, stem_or_img, anns: list[Annotation], w: int, h: int):
        img = self.work_images / Path(stem_or_img).name
        ia = self.load_image_ann(img, w, h) if self.label_path(img).exists() else ImageAnn(w, h)
        ia.w, ia.h, ia.objects = w, h, list(anns)
        self.save_image_ann(img, ia)

    def is_labeled(self, img: Path) -> bool:
        return annstore.has_content(self.label_path(img))

    def samples(self, labeled_only: bool = False) -> list[Sample]:
        out = []
        for img in self.images():
            lp = self.label_path(img)
            if labeled_only and not annstore.has_content(lp):
                continue
            out.append(Sample(img, lp if lp.exists() else None, None))
        return out

    def stats(self) -> dict:
        imgs = self.images()
        labeled = objs = kp = cls = auto = 0
        for p in imgs:
            ia = annstore.load(self.label_path(p))
            if not ia:
                continue
            if ia.objects or ia.image_class:
                labeled += 1
            objs += len(ia.objects)
            kp += sum(1 for a in ia.objects if a.kpts is not None and (a.kpts[:, 2] > 0).any())
            cls += ia.image_class is not None
            auto += sum(1 for a in ia.objects if a.auto)
        return dict(images=len(imgs), labeled=labeled, objects=objs, kpt_objects=kp, classified=cls,
                    auto=auto, runs=len(self.runs()))

    # ------------------------------------------------------------------ runs
    def new_run_dir(self, kind: str) -> Path:
        d = self.root / "generated" / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def runs(self) -> list[Path]:
        g = self.root / "generated"
        if not g.exists():
            return []
        return sorted(d for d in g.iterdir() if d.is_dir() and (d / "gen_report.json").exists())

    # ------------------------------------------------------------------ custom effects
    @property
    def custom_effects(self) -> list[dict]:
        return self.data.setdefault("custom_effects", [])

    @custom_effects.setter
    def custom_effects(self, lst):
        self.data["custom_effects"] = list(lst)
