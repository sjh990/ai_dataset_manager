"""텍스트→이미지 신규 원본 생성 (SDXL) + 자동 라벨 → 프로젝트 원본(work/images)에 추가.

프롬프트는 한 줄에 하나. '분류클래스 :: 프롬프트' 형식이면 이미지 분류 라벨을 함께 지정한다.
  예) damaged :: aerial drone photo of a collapsed building after earthquake, rubble
생성된 이미지는 자동 라벨(점선, 미확정)과 함께 추가되며, 3. 라벨링에서 확인·수정한다.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .prompts import DOMAIN_PREFIX, NEGATIVE
from .quality import sanity

SIZES = {"1024x1024": (1024, 1024), "1152x896 (가로)": (1152, 896), "1216x832 (가로)": (1216, 832),
         "1344x768 (와이드)": (1344, 768), "896x1152 (세로)": (896, 1152), "768x768 (빠름)": (768, 768)}


@dataclass
class Txt2ImgConfig:
    project_dir: str = ""
    prompts: list = field(default_factory=list)
    negative: str = NEGATIVE
    domain: str = "aerial"
    width: int = 1024
    height: int = 1024
    n_per_prompt: int = 4
    steps: int = 30
    guidance: float = 6.5
    seed: int = 2026
    vram_mode: str = "auto"
    models_dir: str = ""
    autolabel: bool = True
    detector: str = "owlv2"
    threshold: float = 0.2
    use_sam: bool = True
    auto_pose: bool = True
    auto_classify: bool = True
    name_prefix: str = "gen"

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def parse_prompt_line(line: str) -> tuple[str | None, str]:
    if "::" in line:
        c, p = line.split("::", 1)
        return (c.strip() or None), p.strip()
    return None, line.strip()


def run_txt2img(cfg: Txt2ImgConfig, progress=None, log=print, cancel: threading.Event | None = None,
                engine=None) -> dict:
    from .annstore import ImageAnn
    from .project import Project
    from .transform import TransformConfig, get_engine

    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    t0 = time.time()
    proj = Project.load(cfg.project_dir)
    lines = [ln for ln in (x.strip() for x in cfg.prompts) if ln and not ln.startswith("#")]
    if not lines:
        raise RuntimeError("프롬프트를 한 줄 이상 입력하세요.")
    engine = engine or get_engine(TransformConfig(models_dir=cfg.models_dir, vram_mode=cfg.vram_mode), log)
    engine.cancel_flag = cancel
    det = pose = clf = None
    root = Path(cfg.models_dir) if cfg.models_dir else None
    if cfg.autolabel:
        from .autolabel import AutoLabeler, ImageClassifier, PoseLabeler, available
        ok, why = available(cfg.models_dir, cfg.detector)
        if ok and proj.classes:
            det = AutoLabeler(root, cfg.detector, cfg.use_sam, log=log)
        elif proj.classes:
            log(f"ℹ 객체 자동 라벨 생략: {why}")
        if cfg.auto_pose and proj.kpt.get("preset") == "coco17":
            try:
                pose = PoseLabeler(root)
            except Exception as e:
                log(f"ℹ 자세 자동 라벨 생략: {e}")
        if cfg.auto_classify and proj.image_classes:
            try:
                clf = ImageClassifier(root)
            except Exception as e:
                log(f"ℹ 이미지 분류 자동 라벨 생략: {e}")
    jobs = [(i, ln, k) for i, ln in enumerate(lines) for k in range(cfg.n_per_prompt)]
    prefix = DOMAIN_PREFIX.get(cfg.domain, "")
    records, stats = [], dict(generated=0, rejected=0, objects=0)
    for n, (i, ln, k) in enumerate(jobs):
        if cancel.is_set():
            break
        img_cls, prompt = parse_prompt_line(ln)
        seed = int(cfg.seed + i * 1000 + k)
        try:
            bgr = engine.txt2img(prefix + prompt, cfg.negative, cfg.width, cfg.height, cfg.steps, cfg.guidance, seed)
        except Exception as e:
            from .diffusion import Cancelled
            if isinstance(e, Cancelled):
                break
            log(f"⚠ 생성 실패 [{prompt[:40]}]: {e}")
            progress(n + 1, len(jobs), prompt[:30])
            continue
        ok, why = sanity(bgr)
        if not ok:
            stats["rejected"] += 1
            log(f"⚠ 품질 거부({why}): {prompt[:40]}")
            progress(n + 1, len(jobs), prompt[:30])
            continue
        h, w = bgr.shape[:2]
        ia = ImageAnn(w, h, [], img_cls)
        if img_cls:
            proj.add_image_class(img_cls)
        if det or pose or clf:
            from .autolabel import label_image_full
            ia = label_image_full(bgr, proj, det, pose, clf, cfg.threshold, cfg.use_sam, ia)
        dst = proj.add_generated_image(bgr, ia, f"{cfg.name_prefix}_{i + 1:03d}_{k + 1:02d}")
        stats["generated"] += 1
        stats["objects"] += len(ia.objects)
        records.append(dict(file=dst.name, prompt=prefix + prompt, seed=seed, image_class=ia.image_class,
                            n_labels=len(ia.objects)))
        progress(n + 1, len(jobs), prompt[:30])
    proj.save()
    logs = proj.root / "logs"
    logs.mkdir(exist_ok=True)
    rep = dict(config=json.loads(cfg.to_json()), stats=stats, records=records, seconds=round(time.time() - t0, 1))
    (logs / f"txt2img_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rep, ensure_ascii=False,
                                                                                     indent=2), encoding="utf-8")
    log(f"완료: 신규 원본 {stats['generated']}장 (자동 라벨 객체 {stats['objects']}개) · 거부 {stats['rejected']} "
        f"({rep['seconds']}s) → 3. 라벨링에서 확인하세요")
    return rep
