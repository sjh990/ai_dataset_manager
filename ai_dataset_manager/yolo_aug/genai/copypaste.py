"""객체 합성 (Copy-Paste): 객체 컷아웃을 배경에 합성하고 라벨을 자동 생성.

객체 출처: ① 입력 데이터셋 라벨(Seg/OBB는 폴리곤 마스크, Detect는 GrabCut으로 전경 추출)
          ② 컷아웃 폴더(투명 PNG, <폴더>/<클래스명>/*.png)
크기: 'match' = 데이터셋 내 같은 클래스의 상대 크기 분포를 따름 (드론 고도 일관성)
합성: 색 보정(LAB 통계 전이) + 경계 페더/포아송 블렌딩 + 그림자
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..dataset import analyze_dataset, scan_dataset
from ..io_utils import IMG_EXTS, imread, imwrite
from ..labels import Annotation, box_corners, format_labels, parse_labels
from . import annstore
from .quality import sanity
from .transform import write_yaml


@dataclass
class CopyPasteConfig:
    dataset_dir: str = ""        # 객체 출처(+기본 배경) 데이터셋
    project_dir: str = ""        # 설정되면 생성 프로젝트(JSON 라벨) 모드
    flip_idx: list | None = None # 키포인트 좌우 교환 인덱스
    cutout_dir: str = ""         # 투명 PNG 컷아웃 폴더 (선택)
    background_dir: str = ""     # 배경 이미지 폴더 (비우면 데이터셋 이미지 + 기존 라벨 유지)
    output_dir: str = ""
    task: str = "detect"
    class_names: list = field(default_factory=list)
    classes: list = field(default_factory=list)   # 합성할 클래스 id (빈 값 = 전체)
    n_images: int = 100
    objs_min: int = 1
    objs_max: int = 5
    scale_mode: str = "match"    # match | relative
    scale_min: float = 0.03      # relative: 배경 긴 변 대비
    scale_max: float = 0.12
    scale_jitter: float = 0.2    # match 모드 크기 흔들림
    rotate: str = "free"         # free | small | none
    flip: bool = True
    color_match: float = 0.6
    blend: str = "feather"       # feather | poisson | alpha
    shadow: float = 0.3
    max_overlap: float = 0.05
    occlusion_drop: float = 0.6
    min_obj_px: int = 12
    grabcut: bool = True
    max_bank: int = 400          # 클래스별 최대 보관 객체 수
    seed: int = 7
    image_ext: str = ".jpg"
    jpeg_quality: int = 95

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class ObjCut:
    cls: int
    img: np.ndarray    # BGR crop
    mask: np.ndarray   # float 0~1
    src: str = ""
    kpts: np.ndarray | None = None  # crop 좌표계 키포인트 (K,3)


# ----------------------------------------------------------------------------- object bank
def _grabcut_mask(crop: np.ndarray, pad: int) -> np.ndarray:
    h, w = crop.shape[:2]
    rect = (pad, pad, max(1, w - 2 * pad), max(1, h - 2 * pad))
    m = np.zeros((h, w), np.uint8)
    try:
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, m, rect, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
        fgm = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except cv2.error:
        fgm = np.zeros((h, w), np.float32)
    box_area = rect[2] * rect[3]
    if fgm.sum() < 0.15 * box_area:  # 추출 실패 → 타원 소프트 마스크로 대체
        fgm = np.zeros((h, w), np.float32)
        cv2.ellipse(fgm, (w // 2, h // 2), (max(1, rect[2] // 2), max(1, rect[3] // 2)), 0, 0, 360, 1.0, -1)
    return fgm


class ObjectBank:
    def __init__(self):
        self.objs: dict[int, list[ObjCut]] = defaultdict(list)
        self.rel_sizes: dict[int, list[float]] = defaultdict(list)

    def classes(self):
        return [c for c, v in self.objs.items() if v]

    def count(self):
        return {c: len(v) for c, v in self.objs.items()}

    def add_from_dataset(self, samples, task, classes, min_px, grabcut, max_per_class, rng, kpt_shape=None,
                         log=print):
        order = rng.permutation(len(samples))
        want = set(classes) if classes else None
        for idx in order:
            s = samples[int(idx)]
            if s.label is None or not Path(s.label).exists():
                continue
            for_this = []
            if not Path(s.label).read_text(encoding="utf-8", errors="ignore").strip():
                continue
            img = imread(s.image)
            if img is None:
                continue
            h, w = img.shape[:2]
            anns, _ = annstore.load_any(s.label, task, w, h, kpt_shape)
            for a in anns:
                if want is not None and a.cls not in want:
                    continue
                x1, y1, x2, y2 = a.bbox()
                self.rel_sizes[a.cls].append(max(x2 - x1, y2 - y1) / max(w, h))
                if len(self.objs[a.cls]) >= max_per_class or min(x2 - x1, y2 - y1) < min_px:
                    continue
                for_this.append(a)
            for a in for_this:
                cut = self._cut(img, a, task, grabcut)
                if cut is not None:
                    cut.src = s.image.name
                    self.objs[a.cls].append(cut)
            if want is not None and all(len(self.objs[c]) >= max_per_class for c in want):
                break
        log("객체 뱅크(데이터셋): " + ", ".join(f"{c}:{len(v)}" for c, v in sorted(self.objs.items())))

    @staticmethod
    def _cut(img, a: Annotation, task, grabcut) -> ObjCut | None:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = a.bbox()
        pad = int(max(2, 0.08 * max(x2 - x1, y2 - y1)))
        X1, Y1 = int(max(0, np.floor(x1) - pad)), int(max(0, np.floor(y1) - pad))
        X2, Y2 = int(min(w, np.ceil(x2) + pad)), int(min(h, np.ceil(y2) + pad))
        crop = img[Y1:Y2, X1:X2].copy()
        if crop.size == 0:
            return None
        if task in ("segment", "obb"):
            m = np.zeros(crop.shape[:2], np.float32)
            cv2.fillPoly(m, [np.round(a.pts - [X1, Y1]).astype(np.int32)], 1.0)
        elif grabcut:
            m = _grabcut_mask(crop, pad)
        else:
            m = np.zeros(crop.shape[:2], np.float32)
            m[int(y1 - Y1):int(y2 - Y1), int(x1 - X1):int(x2 - X1)] = 1.0
        if m.sum() < 16:
            return None
        kp = None
        if a.kpts is not None:
            kp = a.kpts.astype(np.float64).copy()
            vis = kp[:, 2] > 0
            kp[vis, 0] -= X1
            kp[vis, 1] -= Y1
        return ObjCut(a.cls, crop, m, kpts=kp)

    def add_from_cutouts(self, folder: Path, class_names: list, log=print):
        folder = Path(folder)
        n = 0
        for p in sorted(folder.rglob("*.png")):
            cname = p.parent.name if p.parent != folder else p.stem.split("_")[0]
            if cname not in class_names:
                class_names.append(cname)
            cid = class_names.index(cname)
            data = np.fromfile(str(p), np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                continue
            ys, xs = np.nonzero(im[..., 3] > 10)
            if len(xs) == 0:
                continue
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            self.objs[cid].append(ObjCut(cid, im[..., :3].copy(), im[..., 3].astype(np.float32) / 255, p.name))
            n += 1
        log(f"객체 뱅크(컷아웃): {n}개 로드")


# ----------------------------------------------------------------------------- compositing
def _rotate(img, mask, deg):
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return (cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0), M)


def _color_transfer(obj, mask, bg_patch, alpha):
    if alpha <= 0:
        return obj
    o = cv2.cvtColor(obj, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = mask > 0.5
    if sel.sum() < 10:
        return obj
    om, os_ = o[sel].mean(0), o[sel].std(0) + 1e-3
    bm, bs = b.reshape(-1, 3).mean(0), b.reshape(-1, 3).std(0) + 1e-3
    # 밝기(L)는 평균만 부분 이동, 색(a,b)은 평균·분산 이동 → 객체 고유 색은 일부 유지
    tgt_m = np.array([om[0] + (bm[0] - om[0]) * 0.6, bm[1], bm[2]])
    tgt_s = np.array([os_[0], (os_[1] + bs[1]) / 2, (os_[2] + bs[2]) / 2])
    t = (o - om) / os_ * tgt_s + tgt_m
    o2 = o * (1 - alpha) + t * alpha
    return cv2.cvtColor(o2.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def mask_to_ann(mask_full: np.ndarray, cls: int, task: str) -> Annotation | None:
    m = (mask_full > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    if task == "segment":
        c = max(cnts, key=cv2.contourArea)
        eps = 0.004 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(poly) < 3:
            return None
        return Annotation(cls, poly)
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)
    if task == "obb":
        return Annotation(cls, cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64))
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    return Annotation(cls, box_corners(x, y, x + w, y + h))


def paste_objects(bg, bg_anns, bank: ObjectBank, cfg: CopyPasteConfig, rng, task) -> tuple[np.ndarray, list, dict]:
    H, W = bg.shape[:2]
    L = max(H, W)
    out = bg.copy()
    anns = [a.copy() for a in bg_anns]
    boxes = [a.bbox() for a in anns]
    classes = [c for c in (cfg.classes or bank.classes()) if bank.objs.get(c)]
    if not classes:
        raise RuntimeError("합성할 객체가 없습니다 (클래스 선택/객체 출처 확인).")
    n = int(rng.integers(cfg.objs_min, cfg.objs_max + 1))
    sh_ang = rng.uniform(0, 2 * np.pi)
    pasted = 0
    occ = np.zeros((H, W), np.float32)
    for _ in range(n):
        c = int(rng.choice(classes))
        obj = bank.objs[c][int(rng.integers(len(bank.objs[c])))]
        im, m = obj.img, obj.mask
        oh, ow = m.shape
        if cfg.scale_mode == "match" and bank.rel_sizes.get(c):
            rel = float(rng.choice(bank.rel_sizes[c])) * rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
        else:
            rel = rng.uniform(cfg.scale_min, cfg.scale_max)
        target = max(6.0, rel * L)
        ys, xs = np.nonzero(m > 0.5)
        cur = max(np.ptp(xs) + 1, np.ptp(ys) + 1) if len(xs) else max(oh, ow)
        s = target / cur
        nw, nh = max(2, int(ow * s)), max(2, int(oh * s))
        if nw >= W or nh >= H:
            continue
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_LINEAR)
        kp = None
        if obj.kpts is not None:
            kp = obj.kpts.copy()
            vis = kp[:, 2] > 0
            kp[vis, 0] *= nw / ow
            kp[vis, 1] *= nh / oh
        if cfg.flip and rng.random() < 0.5:
            im, m = cv2.flip(im, 1), cv2.flip(m, 1)
            if kp is not None:
                vis = kp[:, 2] > 0
                kp[vis, 0] = nw - kp[vis, 0]
                if cfg.flip_idx and len(cfg.flip_idx) == len(kp):
                    kp = kp[cfg.flip_idx]
        if cfg.rotate != "none":
            deg = rng.uniform(0, 360) if cfg.rotate == "free" else rng.uniform(-15, 15)
            im, m, Mr = _rotate(im, m, deg)
            if kp is not None:
                vis = kp[:, 2] > 0
                if vis.any():
                    kp[vis, :2] = np.hstack([kp[vis, :2], np.ones((vis.sum(), 1))]) @ Mr.T
        ph, pw = m.shape
        if pw >= W - 2 or ph >= H - 2:
            continue
        placed = None
        for _try in range(40):
            x0, y0 = int(rng.integers(1, W - pw - 1)), int(rng.integers(1, H - ph - 1))
            ys, xs = np.nonzero(m > 0.5)
            if len(xs) == 0:
                break
            box = (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)
            if all(_iou(box, b) <= cfg.max_overlap for b in boxes):
                placed = (x0, y0, box)
                break
        if placed is None:
            continue
        x0, y0, box = placed
        patch = out[y0:y0 + ph, x0:x0 + pw]
        im = _color_transfer(im, m, patch, cfg.color_match)
        # 그림자
        if cfg.shadow > 0:
            d = max(2, int(max(pw, ph) * 0.08))
            dx, dy = int(np.cos(sh_ang) * d), int(np.sin(sh_ang) * d)
            sm = np.zeros((H, W), np.float32)
            sx0, sy0 = x0 + dx, y0 + dy
            xa, ya, xb, yb = max(0, sx0), max(0, sy0), min(W, sx0 + pw), min(H, sy0 + ph)
            if xb > xa and yb > ya:
                sm[ya:yb, xa:xb] = m[ya - sy0:yb - sy0, xa - sx0:xb - sx0]
                sm = cv2.GaussianBlur(sm, (0, 0), max(1, d / 2)) * cfg.shadow
                out = (out * (1 - sm[..., None])).astype(np.uint8)
                patch = out[y0:y0 + ph, x0:x0 + pw]
        if cfg.blend == "poisson" and min(pw, ph) > 8:
            mu8 = (m > 0.3).astype(np.uint8) * 255
            try:
                center = (x0 + pw // 2, y0 + ph // 2)
                out = cv2.seamlessClone(im, out, mu8, center, cv2.NORMAL_CLONE)
            except cv2.error:
                a = m[..., None]
                out[y0:y0 + ph, x0:x0 + pw] = (patch * (1 - a) + im * a).astype(np.uint8)
        else:
            a = m if cfg.blend == "alpha" else cv2.GaussianBlur(m, (0, 0), max(0.6, min(pw, ph) / 60))
            a = a[..., None]
            out[y0:y0 + ph, x0:x0 + pw] = (patch * (1 - a) + im * a).clip(0, 255).astype(np.uint8)
        full = np.zeros((H, W), np.float32)
        full[y0:y0 + ph, x0:x0 + pw] = m
        ann = mask_to_ann(full, c, task)
        if ann is None:
            continue
        if kp is not None:
            vis = kp[:, 2] > 0
            kp[vis, 0] += x0
            kp[vis, 1] += y0
            inside = (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H)
            kp[~(vis & inside)] = 0
            ann.kpts = kp
        occ = np.maximum(occ, full)
        anns.append(ann)
        boxes.append(ann.bbox())
        pasted += 1
    # 가려진 기존 객체 라벨 제거
    kept, removed = [], 0
    n_orig = len(bg_anns)
    for i, a in enumerate(anns):
        if i < n_orig:
            x1, y1, x2, y2 = map(int, map(round, a.bbox()))
            region = occ[max(0, y1):y2, max(0, x1):x2]
            if region.size and (region > 0.5).mean() > cfg.occlusion_drop:
                removed += 1
                continue
        kept.append(a)
    return out, kept, dict(pasted=pasted, occluded_removed=removed)


def _list_images(folder):
    return [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMG_EXTS]


def build_bank(cfg: CopyPasteConfig, log=print, samples=None) -> ObjectBank:
    rng = np.random.default_rng(cfg.seed)
    bank = ObjectBank()
    if cfg.project_dir or cfg.dataset_dir:
        if samples is None:
            if cfg.project_dir:
                from .project import Project
                samples = Project.load(cfg.project_dir).samples(labeled_only=True)
            else:
                samples = scan_dataset(cfg.dataset_dir,
                                       exclude=Path(cfg.output_dir) if cfg.output_dir else None).samples
        bank.add_from_dataset(samples, "segment" if cfg.project_dir else cfg.task, cfg.classes, cfg.min_obj_px,
                              cfg.grabcut, cfg.max_bank, rng, log=log)
    if cfg.cutout_dir:
        bank.add_from_cutouts(Path(cfg.cutout_dir), cfg.class_names, log)
    return bank


def run_copypaste(cfg: CopyPasteConfig, progress=None, log=print, cancel: threading.Event | None = None,
                  bank: ObjectBank | None = None) -> dict:
    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    t0 = time.time()
    out = Path(cfg.output_dir)
    proj = None
    if cfg.project_dir:
        from .project import Project
        proj = Project.load(cfg.project_dir)
        cfg.task = "segment"
        if not cfg.class_names:
            cfg.class_names = proj.class_names
        if cfg.flip_idx is None and proj.kpt_names:
            cfg.flip_idx = proj.flip_idx
        src_samples = proj.samples()
    else:
        info = scan_dataset(cfg.dataset_dir, exclude=out) if cfg.dataset_dir else None
        if info is not None and not cfg.class_names:
            analyze_dataset(info)
            cfg.class_names, cfg.task = info.class_names, info.task
        src_samples = info.samples if info else []
    bank = bank or build_bank(cfg, log, [s for s in src_samples if s.label is not None])
    if cfg.background_dir:
        backgrounds = [(p, None) for p in _list_images(cfg.background_dir)]
    else:
        backgrounds = [(s.image, s.label) for s in src_samples]
    if not backgrounds:
        raise RuntimeError("배경 이미지가 없습니다.")
    lab_dir = "ann" if proj is not None else "labels"
    for sub in ("images", lab_dir):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 1)
    records, stats = [], dict(written=0, pasted=0, occluded_removed=0, failed=0)
    for k in range(int(cfg.n_images)):
        if cancel.is_set():
            break
        p, lab = backgrounds[int(rng.integers(len(backgrounds)))]
        bg = imread(p)
        if bg is None:
            stats["failed"] += 1
            continue
        h, w = bg.shape[:2]
        bg_anns, bg_cls = annstore.load_any(lab, cfg.task, w, h) if lab else ([], None)
        img, anns, meta = paste_objects(bg, bg_anns, bank, cfg, rng, cfg.task)
        ok, _ = sanity(img)
        if not ok or meta["pasted"] == 0:
            stats["failed"] += 1
            progress(k + 1, cfg.n_images, p.name)
            continue
        stem = f"cp_{k + 1:05d}_{p.stem}"
        imwrite(out / "images" / f"{stem}{cfg.image_ext}", img, cfg.jpeg_quality)
        if proj is not None:
            annstore.save(out / "ann" / f"{stem}.json", annstore.ImageAnn(w, h, anns, bg_cls))
        else:
            (out / "labels" / f"{stem}.txt").write_text(format_labels(anns, cfg.task, w, h), encoding="utf-8")
        stats["written"] += 1
        stats["pasted"] += meta["pasted"]
        stats["occluded_removed"] += meta["occluded_removed"]
        records.append(dict(file=f"{stem}{cfg.image_ext}", source=str(p), passed=True, n_labels=len(anns),
                            effect="copy_paste", backend="copy_paste", **meta))
        progress(k + 1, cfg.n_images, p.name)
    if proj is None:
        write_yaml(out, cfg.class_names, cfg.task)
    report = dict(config=json.loads(cfg.to_json()), stats=stats, seconds=round(time.time() - t0, 1),
                  class_names=cfg.class_names, label_format="json" if proj else "yolo",
                  records=records, cancelled=cancel.is_set(),
                  bank=bank.count())
    (out / "gen_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"완료: {stats['written']}장 생성 · 합성 객체 {stats['pasted']}개 · 가림 제거 {stats['occluded_removed']} "
        f"({report['seconds']}s)")
    return report
