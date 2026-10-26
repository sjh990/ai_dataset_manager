"""생성 결과 품질 필터.

1) 구조 유지(structure): 원본↔결과 그래디언트 맵 상관 — 낮으면 장면이 바뀌어 라벨 무효
2) 객체 유지(object): 라벨 영역별 그래디언트 상관 — 객체가 사라졌는지 검사 (라벨 제거/이미지 거부)
3) 변화량(change): 효과가 거의 적용되지 않았는지
4) 정상성(sanity): 검은 화면·과포화 등 생성 실패
5) CLIP 효과 점수(선택): 'a photo in heavy rain' 유사도가 원본 대비 증가했는지
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..labels import Annotation


def _gray_small(img, long_side=512):
    h, w = img.shape[:2]
    r = long_side / max(h, w)
    if r < 1:
        img = cv2.resize(img, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), min(r, 1.0)


def _grad(g):
    # 빗줄기·눈송이·노이즈 같은 미세 입자는 구조가 아니므로 먼저 억제
    g = cv2.medianBlur(g.astype(np.uint8), 5).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2)


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def structure_score(orig, gen) -> float:
    go, _ = _gray_small(orig)
    gg, _ = _gray_small(gen)
    if go.shape != gg.shape:
        gg = cv2.resize(gg, (go.shape[1], go.shape[0]))
    return _corr(_grad(go), _grad(gg))


def object_scores(orig, gen, anns: list[Annotation], long_side=1024) -> list[float]:
    go, r = _gray_small(orig, long_side)
    gg, _ = _gray_small(gen, long_side)
    if go.shape != gg.shape:
        gg = cv2.resize(gg, (go.shape[1], go.shape[0]))
    Go, Gg = _grad(go), _grad(gg)
    H, W = go.shape
    out = []
    for a in anns:
        x1, y1, x2, y2 = (np.array(a.bbox()) * r)
        px, py = (x2 - x1) * 0.1 + 2, (y2 - y1) * 0.1 + 2
        xa, ya = int(max(0, x1 - px)), int(max(0, y1 - py))
        xb, yb = int(min(W, x2 + px)), int(min(H, y2 + py))
        if xb - xa < 4 or yb - ya < 4:
            out.append(1.0)
            continue
        out.append(_corr(Go[ya:yb, xa:xb], Gg[ya:yb, xa:xb]))
    return out


def change_score(orig, gen, mask=None) -> float:
    if mask is not None and mask.max() > 0:  # 영역 변형: 영역 내부 변화량
        m = mask > 0.5
        d = np.abs(cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
                   - cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32))[m]
        return float(d.mean() / 255.0) if d.size else 0.0
    a, _ = _gray_small(orig, 256)
    b, _ = _gray_small(gen, 256)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ca = cv2.cvtColor(cv2.resize(orig, (a.shape[1], a.shape[0])), cv2.COLOR_BGR2LAB).astype(np.float32)
    cb = cv2.cvtColor(cv2.resize(gen, (a.shape[1], a.shape[0])), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.abs(ca - cb).mean() / 255.0)


def sanity(gen) -> tuple[bool, str]:
    if gen is None or gen.size == 0:
        return False, "빈 이미지"
    m = gen.mean()
    if m < 3:
        return False, "검은 이미지(fp16 VAE 오류 가능)"
    if m > 252:
        return False, "흰 이미지"
    sat = (gen.max(axis=2) >= 254).mean() + (gen.max(axis=2) <= 1).mean()
    if sat > 0.6:
        return False, f"과포화/클리핑 {sat:.0%}"
    if gen.std() < 4:
        return False, "평탄한 이미지"
    return True, ""


class ClipScorer:
    """로컬 CLIP(openai/clip-vit-base-patch32)로 효과 문장 유사도 계산."""

    def __init__(self, path: Path, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(str(path), local_files_only=True).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(str(path), local_files_only=True)

    def sims(self, images_bgr: list, text: str) -> list[float]:
        from PIL import Image

        ims = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in images_bgr]
        with self.torch.no_grad():
            inp = self.proc(text=[text], images=ims, return_tensors="pt", padding=True).to(self.device)
            out = self.model(**inp)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            return (ie @ te.T).squeeze(-1).float().cpu().tolist()

    def effect_gain(self, orig, gen, text) -> float:
        s = self.sims([orig, gen], text)
        return float(s[1] - s[0])


@dataclass
class QualityConfig:
    min_structure: float = 0.35
    min_object: float = 0.20
    object_action: str = "drop_label"   # drop_label | reject_image | keep
    min_change: float = 0.015
    use_clip: bool = False
    min_clip_gain: float = 0.0


@dataclass
class QualityResult:
    passed: bool
    structure: float
    change: float
    object_scores: list[float] = field(default_factory=list)
    dropped_labels: list[int] = field(default_factory=list)
    clip_gain: float | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self):
        return dict(passed=self.passed, structure=round(self.structure, 4), change=round(self.change, 4),
                    object_min=round(min(self.object_scores), 4) if self.object_scores else None,
                    dropped_labels=self.dropped_labels,
                    clip_gain=None if self.clip_gain is None else round(self.clip_gain, 4),
                    reasons=self.reasons)


def evaluate(orig, gen, anns, cfg: QualityConfig, clip: ClipScorer | None = None, clip_text: str = "",
             region_only_idx: list[int] | None = None, region_mask=None) -> QualityResult:
    """region_only_idx: 영역 변형(파손)일 때 의도적으로 바뀐 객체 인덱스 → 객체 유지 검사 제외."""
    ok, why = sanity(gen)
    st = structure_score(orig, gen)
    ch = change_score(orig, gen, region_mask)
    obj = object_scores(orig, gen, anns) if anns else []
    res = QualityResult(True, st, ch, obj)
    if not ok:
        res.passed = False
        res.reasons.append(why)
    if st < cfg.min_structure:
        res.passed = False
        res.reasons.append(f"구조 유지 {st:.2f} < {cfg.min_structure:.2f}")
    if ch < cfg.min_change:
        res.passed = False
        res.reasons.append(f"변화량 부족 {ch:.3f}")
    skip = set(region_only_idx or [])
    bad = [i for i, s in enumerate(obj) if s < cfg.min_object and i not in skip]
    if bad and cfg.object_action == "reject_image":
        res.passed = False
        res.reasons.append(f"객체 소실 의심 {len(bad)}개")
    elif bad and cfg.object_action == "drop_label":
        res.dropped_labels = bad
    if clip is not None and cfg.use_clip and clip_text:
        g = clip.effect_gain(orig, gen, clip_text)
        res.clip_gain = g
        if g < cfg.min_clip_gain:
            res.passed = False
            res.reasons.append(f"CLIP 효과 점수 {g:+.3f} < {cfg.min_clip_gain}")
    return res
