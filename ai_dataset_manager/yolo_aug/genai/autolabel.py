"""자동 라벨러: 텍스트(클래스 문구)로 객체를 찾는 오픈어휘 검출기 + (선택) SAM 폴리곤.

  검출기: OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0) 또는
          Grounding DINO Tiny (IDEA-Research/grounding-dino-tiny, Apache-2.0)
  폴리곤: SAM ViT-B (facebook/sam-vit-base, Apache-2.0) — 박스 → 마스크 → 폴리곤
결과는 사람이 라벨링 탭에서 확인·수정한다(오검출이 섞일 수 있음).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..labels import Annotation, box_corners
from .models import is_installed, model_path


@dataclass
class Detection:
    cls: int
    box: tuple[float, float, float, float]
    score: float
    poly: np.ndarray | None = None


def _nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    from .copypaste import _iou

    keep: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.score):
        if all(_iou(d.box, k.box) < iou_thr or k.cls != d.cls for k in keep):
            keep.append(d)
    return keep


class AutoLabeler:
    def __init__(self, models_root: Path | None = None, detector: str = "owlv2", use_sam: bool = True,
                 device: str | None = None, log=print):
        import torch

        self.torch = torch
        self.root = Path(models_root) if models_root else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log = log
        self.detector = detector
        self._det = None
        self._sam = None
        self.use_sam = use_sam and is_installed("sam_base", self.root)

    # ------------------------------------------------------------------ loading
    def _load_det(self):
        if self._det is not None:
            return self._det
        key = "owlv2" if self.detector == "owlv2" else "grounding_dino"
        if not is_installed(key, self.root):
            raise RuntimeError(f"자동 라벨 모델({key})이 설치되지 않았습니다. '모델 · GPU' 탭에서 받으세요.")
        p = str(model_path(key, self.root))
        self.log(f"검출기 로딩: {key} ({self.device})")
        if key == "owlv2":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            proc = Owlv2Processor.from_pretrained(p, local_files_only=True)
            model = Owlv2ForObjectDetection.from_pretrained(p, local_files_only=True)
        else:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            proc = AutoProcessor.from_pretrained(p, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(p, local_files_only=True)
        self._det = (proc, model.to(self.device).eval())
        return self._det

    def _load_sam(self):
        if self._sam is None:
            from transformers import SamModel, SamProcessor
            p = str(model_path("sam_base", self.root))
            self.log(f"SAM 로딩 ({self.device})")
            self._sam = (SamProcessor.from_pretrained(p, local_files_only=True),
                         SamModel.from_pretrained(p, local_files_only=True).to(self.device).eval())
        return self._sam

    def unload(self):
        self._det = self._sam = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    # ------------------------------------------------------------------ detection
    def detect(self, bgr, prompts: list[str], threshold: float = 0.2, text_threshold: float = 0.25,
               nms: float = 0.5, min_size: int = 4) -> list[Detection]:
        """prompts[i] = 클래스 i의 문구 (빈 문자열이면 그 클래스는 건너뜀)."""
        from PIL import Image

        proc, model = self._load_det()
        h, w = bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        valid = [(i, p.strip()) for i, p in enumerate(prompts) if p and p.strip()]
        if not valid:
            return []
        dets: list[Detection] = []
        torch = self.torch
        with torch.no_grad():
            if self.detector == "owlv2":
                texts = [[p for _, p in valid]]
                inputs = proc(text=texts, images=pil, return_tensors="pt", padding="max_length",
                              truncation=True).to(self.device)
                out = model(**inputs)
                side = max(h, w)  # OWLv2는 정사각 패딩 기준 좌표
                ts = torch.tensor([[side, side]], device=self.device)
                fn = getattr(proc, "post_process_grounded_object_detection", None) or proc.post_process_object_detection
                res = fn(outputs=out, threshold=threshold, target_sizes=ts)[0]
                for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
                    dets.append(Detection(valid[int(lab)][0], tuple(box), float(score)))
            else:
                text = ". ".join(p.lower() for _, p in valid) + "."
                inputs = proc(images=pil, text=text, return_tensors="pt").to(self.device)
                out = model(**inputs)
                kw = dict(target_sizes=[(h, w)], text_threshold=text_threshold)
                try:
                    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=threshold, **kw)[0]
                except TypeError:
                    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=threshold,
                                                                      **kw)[0]
                labels = res.get("text_labels", res.get("labels"))
                for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
                    lab = str(lab).lower()
                    cid = next((i for i, p in valid if p.lower() in lab or lab in p.lower()), None)
                    if cid is not None:
                        dets.append(Detection(cid, tuple(box), float(score)))
        out_d = []
        for d in dets:
            x1, y1, x2, y2 = d.box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 >= min_size and y2 - y1 >= min_size:
                out_d.append(Detection(d.cls, (x1, y1, x2, y2), d.score))
        return _nms(out_d, nms)

    # ------------------------------------------------------------------ SAM
    def segment(self, bgr, dets: list[Detection]) -> list[Detection]:
        if not dets or not self.use_sam:
            return dets
        from PIL import Image

        proc, model = self._load_sam()
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes = [[list(map(float, d.box)) for d in dets]]
        with self.torch.no_grad():
            inputs = proc(pil, input_boxes=boxes, return_tensors="pt").to(self.device)
            out = model(**inputs, multimask_output=False)
            masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                            inputs["reshaped_input_sizes"].cpu())[0]
        for d, m in zip(dets, masks):
            mm = m[0].numpy().astype(np.uint8)
            cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 16:
                    poly = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
                    if len(poly) >= 3:
                        d.poly = poly.astype(np.float64)
        return dets

    def label(self, bgr, prompts, threshold=0.2, want_polygons=True, **kw) -> tuple[list[Annotation], list[float]]:
        dets = self.detect(bgr, prompts, threshold, **kw)
        if want_polygons:
            dets = self.segment(bgr, dets)
        anns, scores = [], []
        for d in dets:
            pts = d.poly if d.poly is not None else box_corners(*d.box)
            anns.append(Annotation(d.cls, pts))
            scores.append(d.score)
        return anns, scores


class PoseLabeler:
    """ViTPose(top-down): 사람 박스 → COCO 17 키포인트. 프로젝트 스키마가 'coco17'일 때 사용."""

    def __init__(self, models_root: Path | None = None, device: str | None = None, kpt_thr: float = 0.3):
        import torch

        if not is_installed("vitpose_base", Path(models_root) if models_root else None):
            raise RuntimeError("ViTPose 모델(vitpose_base)이 설치되지 않았습니다. '모델 · GPU'에서 받으세요.")
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        p = str(model_path("vitpose_base", Path(models_root) if models_root else None))
        self.proc = AutoProcessor.from_pretrained(p, local_files_only=True)
        self.model = VitPoseForPoseEstimation.from_pretrained(p, local_files_only=True).to(self.device).eval()
        self.thr = kpt_thr
        self.k = int(getattr(self.model.config, "num_labels", 17) or 17)

    def estimate(self, bgr, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        from PIL import Image

        if not boxes:
            return []
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes]
        with self.torch.no_grad():
            inputs = self.proc(pil, boxes=[xywh], return_tensors="pt").to(self.device)
            kw = {}
            if getattr(self.model.config, "backbone_config", None) is not None and \
                    getattr(self.model.config.backbone_config, "num_experts", 1) > 1:
                kw["dataset_index"] = self.torch.zeros(len(xywh), dtype=self.torch.long, device=self.device)
            out = self.model(**inputs, **kw)
        res = self.proc.post_process_pose_estimation(out, boxes=[xywh])[0]
        kpts = []
        for r in res:
            k = r["keypoints"].cpu().numpy().astype(np.float64)
            s = r["scores"].cpu().numpy().astype(np.float64)
            v = np.where(s >= self.thr, 2.0, 0.0)
            kk = np.zeros((len(k), 3))
            kk[:, :2] = k
            kk[:, 2] = v
            H, W = bgr.shape[:2]
            outside = (kk[:, 0] < 0) | (kk[:, 1] < 0) | (kk[:, 0] >= W) | (kk[:, 1] >= H)
            kk[(v == 0) | outside] = 0
            kpts.append(kk)
        return kpts


class ImageClassifier:
    """CLIP 제로샷 이미지 분류: 클래스 문구 중 가장 유사한 것 선택."""

    def __init__(self, models_root: Path | None = None, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        root = Path(models_root) if models_root else None
        if not is_installed("clip_b32", root):
            raise RuntimeError("CLIP 모델(clip_b32)이 설치되지 않았습니다. '모델 · GPU'에서 받으세요.")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        p = str(model_path("clip_b32", root))
        self.model = CLIPModel.from_pretrained(p, local_files_only=True).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(p, local_files_only=True)

    def classify(self, bgr, prompts: list[str], template: str = "a photo of {}") -> tuple[int, list[float]]:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        texts = [template.format(p) for p in prompts]
        with self.torch.no_grad():
            inp = self.proc(text=texts, images=pil, return_tensors="pt", padding=True, truncation=True).to(self.device)
            logits = self.model(**inp).logits_per_image[0]
            probs = logits.softmax(-1).float().cpu().numpy()
        return int(probs.argmax()), probs.tolist()


def available(models_root=None, detector="owlv2") -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False, "PyTorch/transformers 미설치"
    key = "owlv2" if detector == "owlv2" else "grounding_dino"
    if not is_installed(key, Path(models_root) if models_root else None):
        return False, f"{key} 모델 미설치"
    return True, ""


def label_image_full(bgr, project, det: "AutoLabeler | None" = None, pose: "PoseLabeler | None" = None,
                     clf: "ImageClassifier | None" = None, threshold: float = 0.2, want_polygons: bool = True,
                     existing=None, keep_auto: bool = False, reclassify: bool = False):
    """객체 검출(+SAM) → 자세(ViTPose, 스키마 coco17 + 대상 클래스) → 이미지 분류(CLIP).
    existing(ImageAnn)이 있으면 확정 라벨은 유지하고 자동 라벨만 교체한다. 반환 ImageAnn."""
    from .annstore import ImageAnn

    h, w = bgr.shape[:2]
    old = existing.objects if existing else []
    # 검출을 다시 돌릴 때만 기존 자동 라벨(점선)을 교체한다
    ia = ImageAnn(w, h, [a for a in old if det is None or keep_auto or not a.auto],
                  existing.image_class if existing else None)
    if det is not None:
        prompts = [c.prompt or c.name.replace("_", " ") for c in project.classes]
        anns, scores = det.label(bgr, prompts, threshold, want_polygons=want_polygons)
        for a, s in zip(anns, scores):
            a.auto, a.score = True, float(s)
        ia.objects += anns
    if pose is not None and project.kpt.get("preset") == "coco17":
        ids = set(project.pose_class_ids)
        targets = [a for a in ia.objects if a.cls in ids and (a.kpts is None or not (a.kpts[:, 2] > 0).any())]
        if targets:
            kps = pose.estimate(bgr, [a.bbox() for a in targets])
            for a, k in zip(targets, kps):
                a.kpts = k
                a.auto = True
    if clf is not None and project.image_classes and (reclassify or not ia.image_class):
        prompts = [c.prompt or c.name.replace("_", " ") for c in project.image_classes]
        i, _ = clf.classify(bgr, prompts)
        ia.image_class = project.image_classes[i].name
    return ia
