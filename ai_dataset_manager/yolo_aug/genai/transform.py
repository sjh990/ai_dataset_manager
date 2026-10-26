"""이미지 변형 배치 파이프라인 (날씨·시간대·객체 파손) — 원본 라벨 유지/재매핑 + 품질 필터."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..dataset import analyze_dataset, scan_dataset
from ..io_utils import imread, imwrite
from ..labels import Annotation, format_labels, parse_labels
from . import annstore
from .effects import apply_procedural
from .prompts import NEGATIVE, build_prompt, get_effect
from .quality import ClipScorer, QualityConfig, evaluate


@dataclass
class TransformConfig:
    dataset_dir: str = ""
    project_dir: str = ""                # 설정되면 생성 프로젝트(JSON 라벨: 폴리곤·키포인트·분류) 모드
    output_dir: str = ""
    task: str = "detect"
    class_names: list = field(default_factory=list)
    kpt_shape: list | None = None
    backend: str = "procedural"          # procedural | diffusion
    effects: list = field(default_factory=lambda: ["rain", "night"])
    custom_prompts: dict = field(default_factory=dict)
    negative_prompt: str = NEGATIVE
    domain: str = "aerial"
    intensity: float = 0.6               # 절차적 강도
    strength: dict = field(default_factory=dict)       # effect -> img2img strength (없으면 프리셋)
    control_scale: float | None = None   # None → 효과 프리셋
    steps: int = 25
    guidance: float = 6.0
    canny_low: int = 100
    canny_high: int = 200
    long_side: int = 1024
    detail_keep: float = 0.35            # 업스케일 시 원본 고주파 디테일 보존율
    blend_objects: float = 0.0           # 라벨 영역 원본 보존율 (전역 효과)
    variants: int = 1
    max_images: int = 0                  # 0 = 전체
    seed: int = 1234
    region_classes: list = field(default_factory=list)   # 파손/구조물 변형 대상 클래스 id
    pole_classes: list = field(default_factory=list)     # 항상 '기둥형'(회전)으로 처리할 클래스 id
    custom_effects: list = field(default_factory=list)   # 사용자 효과 정의(dict) 목록 → prompts.custom_effect
    region_remap: str = ""               # 파손 후 바꿀 클래스 이름 ('' = 유지)
    image_remap: str = ""                # 대상 변형이 적용된 이미지의 '이미지 분류' 클래스 ('' = 유지)
    flip_idx: list | None = None
    region_whole_if_none: bool = False
    max_regions: int = 3
    quality: dict = field(default_factory=lambda: asdict(QualityConfig()))
    save_rejected: bool = True
    vram_mode: str = "auto"
    models_dir: str = ""
    image_ext: str = ".jpg"
    jpeg_quality: int = 95

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------- helpers
def ann_mask(shape, anns: list[Annotation], task: str, idx: list[int] | None = None) -> np.ndarray:
    m = np.zeros(shape[:2], np.float32)
    for i, a in enumerate(anns):
        if idx is not None and i not in idx:
            continue
        if task in ("segment", "obb"):
            cv2.fillPoly(m, [np.round(a.pts).astype(np.int32)], 1.0)
        else:
            x1, y1, x2, y2 = map(int, map(round, a.bbox()))
            m[max(0, y1):y2, max(0, x1):x2] = 1.0
    return m


def detail_blend(orig, gen, keep: float, sigma: float):
    if keep <= 0 or sigma < 0.6:
        return gen
    o, g = orig.astype(np.float32), gen.astype(np.float32)
    lo_o, lo_g = cv2.GaussianBlur(o, (0, 0), sigma), cv2.GaussianBlur(g, (0, 0), sigma)
    out = lo_g + keep * (o - lo_o) + (1 - keep) * (g - lo_g)
    return out.clip(0, 255).astype(np.uint8)


def blend_back(orig, gen, mask, alpha):
    if alpha <= 0 or mask.max() <= 0:
        return gen
    f = max(2, int(min(orig.shape[:2]) * 0.005))
    m = cv2.GaussianBlur(mask, (0, 0), f)[..., None] * alpha
    return (gen * (1 - m) + orig * m).clip(0, 255).astype(np.uint8)


class _EngineCache:
    engine = None
    key = None
    clip = None
    clip_key = None


def get_engine(cfg: TransformConfig, log=print):
    from .diffusion import DiffusionEngine

    key = (cfg.models_dir, cfg.vram_mode)
    if _EngineCache.engine is None or _EngineCache.key != key:
        _EngineCache.engine = DiffusionEngine(Path(cfg.models_dir) if cfg.models_dir else None, cfg.vram_mode, log)
        _EngineCache.key = key
    _EngineCache.engine.log = log
    return _EngineCache.engine


def get_clip(cfg: TransformConfig):
    from .models import is_installed, model_path

    root = Path(cfg.models_dir) if cfg.models_dir else None
    if not is_installed("clip_b32", root):
        return None
    if _EngineCache.clip is None or _EngineCache.clip_key != cfg.models_dir:
        _EngineCache.clip = ClipScorer(model_path("clip_b32", root))
        _EngineCache.clip_key = cfg.models_dir
    return _EngineCache.clip


def release_models():
    if _EngineCache.engine is not None:
        _EngineCache.engine.unload()
    _EngineCache.engine = None
    _EngineCache.clip = None


# ----------------------------------------------------------------------------- one image
def transform_one(img, anns, effect_key: str, cfg: TransformConfig, seed: int, engine=None, clip=None,
                  class_names: list | None = None, log=print, image_class: str | None = None):
    """반환: dict(image, anns, quality, meta, region_idx, image_class) 또는 None(대상 없음)"""
    res = _transform_one(img, anns, effect_key, cfg, seed, engine, clip, class_names, log)
    if res is None:
        return None
    eff = get_effect(effect_key, cfg.custom_effects)
    new_cls = image_class
    if eff.image_remap.strip():
        new_cls = eff.image_remap.strip()
    elif cfg.image_remap.strip() and res.get("region_idx"):
        new_cls = cfg.image_remap.strip()
    res["image_class"] = new_cls
    return res


def _transform_one(img, anns, effect_key: str, cfg: TransformConfig, seed: int, engine=None, clip=None,
                   class_names: list | None = None, log=print):
    eff = get_effect(effect_key, cfg.custom_effects)
    rng = np.random.default_rng(seed)
    names = class_names if class_names is not None else list(cfg.class_names)
    # 효과별 대상/재매핑(사용자 효과) → 없으면 공통 설정
    if eff.targets:
        targets = {names.index(n) for n in eff.targets if n in names}
    else:
        targets = set(cfg.region_classes)
    remap = (eff.remap or cfg.region_remap).strip()
    negative = eff.negative.strip() or cfg.negative_prompt
    qcfg = QualityConfig(**cfg.quality)
    qcfg.min_structure *= eff.structure_relax
    prompt = build_prompt(eff, cfg.domain, cfg.custom_prompts.get(effect_key))
    meta = dict(effect=effect_key, backend=cfg.backend, seed=int(seed))
    region_idx: list[int] = []
    new_anns = [a.copy() for a in anns]
    if eff.mode == "region":
        region_idx = [i for i, a in enumerate(anns) if a.cls in targets]
        if not region_idx and not cfg.region_whole_if_none:
            return None
        # 큰 객체부터 최대 N개
        region_idx = sorted(region_idx, key=lambda i: -np.prod(np.ptp(anns[i].pts, axis=0)))[:max(1, cfg.max_regions)]

    if eff.kind == "structural":
        from .structural import deform_object

        if not region_idx:
            return None
        eng = None
        if cfg.backend == "diffusion":
            eng = engine or get_engine(cfg, log)
        # 형태에 따라 변형 방식 자동 선택: 세로로 긴 기둥형 → 회전, 가로로 넓은 판넬형 → 앞뒤로 눕힘
        gen = img
        new_anns = [a.copy() for a in anns]
        for i in region_idx:
            x1, y1, x2, y2 = anns[i].bbox()
            tall = (y2 - y1) > 1.3 * (x2 - x1) or anns[i].cls in set(cfg.pole_classes)
            if eff.struct_action == "tilt":
                mode, rngs = ("tilt", (15, 30)) if tall else ("lean", (35, 55))
            else:
                mode, rngs = ("fall", (55, 75)) if tall else ("knockdown", (10, 10))
            gen, upd, done = deform_object(gen, new_anns, [i], cfg.task, mode, rng, rngs, eng,
                                           build_prompt(eff, "none", cfg.custom_prompts.get(effect_key)),
                                           negative, seed, long_side=cfg.long_side, steps=cfg.steps)
            new_anns = upd
        meta.update(regions=len(region_idx))
        rmask = ann_mask(img.shape, anns, cfg.task, region_idx)
        q = evaluate(img, gen, anns, qcfg, clip, eff.clip_text, region_only_idx=region_idx, region_mask=rmask)
        q.dropped_labels = [i for i in q.dropped_labels if i not in region_idx]
        if remap:
            nm = remap
            if nm not in names:
                names.append(nm)
            for i in region_idx:
                new_anns[i].cls = names.index(nm)
        if q.dropped_labels:
            drop = set(q.dropped_labels)
            new_anns = [a for k, a in enumerate(new_anns) if k not in drop]
        return dict(image=gen, anns=new_anns, quality=q, meta=meta, region_idx=region_idx)

    if cfg.backend == "procedural":
        if eff.custom:
            raise RuntimeError(f"사용자 프롬프트 효과 '{eff.name}'는 생성형(SDXL) 방식에서만 동작합니다.")
        if eff.mode == "region":
            mask = ann_mask(img.shape, anns, cfg.task, region_idx) if region_idx else None
            gen = apply_procedural(img, effect_key, rng, cfg.intensity, mask)
        else:
            gen = apply_procedural(img, effect_key, rng, cfg.intensity)
        meta["intensity"] = cfg.intensity
    else:
        engine = engine or get_engine(cfg, log)
        strength = float(cfg.strength.get(effect_key, eff.strength))
        cscale = eff.control_scale if cfg.control_scale is None else float(cfg.control_scale)
        meta.update(prompt=prompt, strength=strength, steps=cfg.steps, guidance=cfg.guidance)
        if eff.mode == "region" and region_idx:
            gen = img.copy()
            for j, i in enumerate(region_idx):
                one = ann_mask(img.shape, anns, cfg.task, [i])
                one = cv2.dilate(one, np.ones((5, 5), np.uint8))
                gen = engine.inpaint_region(gen, one, anns[i].bbox(), prompt, negative, strength,
                                            cfg.steps, cfg.guidance + 1.0, seed + j, cfg.long_side)
            meta["regions"] = len(region_idx)
        else:
            gen = engine.img2img(img, prompt, negative, strength, cfg.steps, cfg.guidance, seed,
                                 cscale, cfg.canny_low, cfg.canny_high, cfg.long_side)
            meta["control_scale"] = cscale
            h, w = img.shape[:2]
            gen = detail_blend(img, gen, cfg.detail_keep, sigma=max(w, h) / cfg.long_side * 0.8)
    if eff.mode != "region" and cfg.blend_objects > 0 and anns:
        gen = blend_back(img, gen, ann_mask(img.shape, anns, cfg.task), cfg.blend_objects)

    rmask = ann_mask(img.shape, anns, cfg.task, region_idx) if region_idx else None
    q = evaluate(img, gen, anns, qcfg, clip, eff.clip_text, region_only_idx=region_idx, region_mask=rmask)
    # 파손 클래스 재매핑
    if region_idx and remap:
        nm = remap
        if nm not in names:
            names.append(nm)
        cid = names.index(nm)
        for i in region_idx:
            new_anns[i].cls = cid
    if q.dropped_labels:
        drop = set(q.dropped_labels)
        new_anns = [a for i, a in enumerate(new_anns) if i not in drop]
    return dict(image=gen, anns=new_anns, quality=q, meta=meta, region_idx=region_idx)


# ----------------------------------------------------------------------------- batch
def run_transform(cfg: TransformConfig, progress=None, log=print, cancel: threading.Event | None = None) -> dict:
    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    t0 = time.time()
    out = Path(cfg.output_dir)
    proj = None
    if cfg.project_dir:
        from .project import Project
        proj = Project.load(cfg.project_dir)
        cfg.task = "segment"  # 프로젝트 내부 라벨은 폴리곤
        if not cfg.class_names:
            cfg.class_names = proj.class_names
        if cfg.flip_idx is None and proj.kpt_names:
            cfg.flip_idx = proj.flip_idx
        samples = proj.samples(labeled_only=True) or proj.samples()
    else:
        info = scan_dataset(cfg.dataset_dir, exclude=out)
        if not cfg.class_names:
            analyze_dataset(info)
            cfg.class_names, cfg.task = info.class_names, info.task
            cfg.kpt_shape = list(info.kpt_shape) if info.kpt_shape else None
        samples = [s for s in info.samples if s.label is not None] or info.samples
    rng = np.random.default_rng(cfg.seed)
    if cfg.max_images and len(samples) > cfg.max_images:
        samples = [samples[i] for i in sorted(rng.choice(len(samples), cfg.max_images, replace=False))]
    lab_dir = "ann" if proj is not None else "labels"
    for sub in ("images", lab_dir):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if cfg.save_rejected:
        for sub in ("images", lab_dir):
            (out / "_rejected" / sub).mkdir(parents=True, exist_ok=True)

    engine = clip = None
    if cfg.backend == "diffusion":
        engine = get_engine(cfg, log)
        if engine.device != "cuda":
            log("⚠ CUDA GPU가 없어 CPU로 실행합니다 (장당 수 분 이상 소요).")
    if QualityConfig(**cfg.quality).use_clip:
        try:
            clip = get_clip(cfg)
            if clip is None:
                log("ℹ CLIP 모델 미설치 → CLIP 필터 생략")
        except Exception as e:
            log(f"ℹ CLIP 로드 실패({e}) → CLIP 필터 생략")
    engine_cancel = cancel
    if engine is not None:
        engine.cancel_flag = engine_cancel

    names = list(cfg.class_names)
    kdim = cfg.kpt_shape[1] if cfg.kpt_shape else 3
    # 같은 파이프라인(img2img ↔ inpaint)을 쓰는 효과끼리 묶어 모델 재로딩(수십 초)을 최소화
    E = {k: get_effect(k, cfg.custom_effects) for k in cfg.effects}
    effs = sorted(cfg.effects, key=lambda k: (E[k].mode != "global", E[k].kind))
    jobs = [(s, e, v) for e in effs for s in samples for v in range(cfg.variants)]
    total = len(jobs)
    records, stats = [], dict(passed=0, rejected=0, skipped=0, errors=0, labels_dropped=0)
    for n, (smp, eff, v) in enumerate(jobs):
        if cancel.is_set():
            break
        img = imread(smp.image)
        if img is None:
            stats["errors"] += 1
            continue
        h, w = img.shape[:2]
        anns, img_cls = annstore.load_any(smp.label, cfg.task, w, h, tuple(cfg.kpt_shape) if cfg.kpt_shape else None)
        seed = int(cfg.seed + n * 7919)
        try:
            res = transform_one(img, anns, eff, cfg, seed, engine, clip, names, log, image_class=img_cls)
        except Exception as e:
            from .diffusion import Cancelled
            if isinstance(e, Cancelled):
                break
            stats["errors"] += 1
            log(f"⚠ {smp.image.name} [{eff}]: {type(e).__name__}: {e}")
            progress(n + 1, total, smp.image.name)
            continue
        if res is None:
            stats["skipped"] += 1
            progress(n + 1, total, smp.image.name)
            continue
        q = res["quality"]
        stem = f"{smp.image.stem}_{eff}_{v + 1}"
        base = out if q.passed else out / "_rejected"
        if q.passed or cfg.save_rejected:
            imwrite(base / "images" / f"{stem}{cfg.image_ext}", res["image"], cfg.jpeg_quality)
            if proj is not None:
                annstore.save(base / "ann" / f"{stem}.json",
                              annstore.ImageAnn(w, h, res["anns"], res.get("image_class")))
            else:
                (base / "labels" / f"{stem}.txt").write_text(format_labels(res["anns"], cfg.task, w, h, kdim),
                                                             encoding="utf-8")
        stats["passed" if q.passed else "rejected"] += 1
        stats["labels_dropped"] += len(q.dropped_labels)
        records.append(dict(file=f"{stem}{cfg.image_ext}", source=str(smp.image), passed=q.passed,
                            n_labels=len(res["anns"]), image_class=res.get("image_class"),
                            quality=q.to_dict(), **res["meta"]))
        progress(n + 1, total, smp.image.name)

    if proj is None:
        write_yaml(out, names, cfg.task, cfg.kpt_shape)
    img_classes = sorted({r["image_class"] for r in records if r.get("image_class")})
    report = dict(config=json.loads(cfg.to_json()), stats=stats, seconds=round(time.time() - t0, 1),
                  class_names=names, image_classes=img_classes, label_format="json" if proj else "yolo",
                  records=records, cancelled=cancel.is_set())
    (out / "gen_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"완료: 통과 {stats['passed']} · 거부 {stats['rejected']} · 대상없음 {stats['skipped']} · 오류 {stats['errors']} "
        f"({report['seconds']}s)")
    return report


def write_yaml(out: Path, names: list, task: str, kpt_shape=None):
    d = {"path": str(out.resolve()).replace("\\", "/"), "train": "images", "val": "images",
         "nc": len(names), "names": {i: n for i, n in enumerate(names)}}
    if task == "pose" and kpt_shape:
        d["kpt_shape"] = list(kpt_shape)
    (out / "data.yaml").write_text(
        f"# Generated by YOLO Dataset Augmentor - GenAI ({task})\n"
        "# 이 폴더를 '증강' 탭에 입력하면 분할·추가 증강을 할 수 있습니다.\n"
        + yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
