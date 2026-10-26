from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..genai.models import is_installed, model_path
from ..io_utils import imread
from .engines import resolve_device


def pick_exemplars(project, items: list, per_class: int = 3, min_px: int = 12) -> list[dict]:
    cands: dict[int, list] = {}
    for it in items:
        for a in it.ann.objects:
            x1, y1, x2, y2 = a.bbox()
            if x2 - x1 < min_px or y2 - y1 < min_px:
                continue
            cands.setdefault(a.cls, []).append(((x2 - x1) * (y2 - y1), it.image.name, [x1, y1, x2, y2]))
    out = []
    names = project.class_names
    for c, lst in cands.items():
        lst.sort(key=lambda t: -t[0])
        name = names[c] if c < len(names) else str(c)
        chosen, imgs = [], set()
        for t in lst:
            if t[1] not in imgs:
                chosen.append(t)
                imgs.add(t[1])
            if len(chosen) >= per_class:
                break
        for t in lst:
            if len(chosen) >= per_class:
                break
            if t not in chosen:
                chosen.append(t)
        out.extend(dict(cls_name=name, image=t[1], box=t[2]) for t in chosen)
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class ExemplarDetector:
    def __init__(self, project, exemplars: list[dict], models_root, device: str = "auto"):
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        root = Path(models_root) if models_root else None
        if not is_installed("owlv2", root):
            raise RuntimeError("OWLv2 model is required (Models / GPU -> owlv2).")
        p = str(model_path("owlv2", root))
        self.torch = torch
        self.dev = resolve_device(device)
        self.proc = Owlv2Processor.from_pretrained(p, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(p, local_files_only=True).to(self.dev).eval()
        self.queries = []
        for e in exemplars:
            img = imread(project.work_images / e["image"])
            if img is None:
                continue
            x1, y1, x2, y2 = e["box"]
            pw, ph = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            h, w = img.shape[:2]
            crop = img[int(max(0, y1 - ph)):int(min(h, y2 + ph)), int(max(0, x1 - pw)):int(min(w, x2 + pw))]
            if crop.size == 0:
                continue
            self.queries.append((e["cls_name"], Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        if not self.queries:
            raise RuntimeError("No usable exemplars.")

    def __call__(self, bgr: np.ndarray, threshold: float) -> list[tuple[str, np.ndarray, float, None]]:
        from PIL import Image

        h, w = bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        side = max(h, w)
        dets = []
        with self.torch.no_grad():
            for name, q in self.queries:
                inp = self.proc(images=pil, query_images=q, return_tensors="pt").to(self.dev)
                out = self.model.image_guided_detection(**inp)
                ts = self.torch.tensor([[side, side]], device=self.dev)
                res = self.proc.post_process_image_guided_detection(outputs=out, threshold=threshold,
                                                                     nms_threshold=0.3, target_sizes=ts)[0]
                for box, sc in zip(res["boxes"].tolist(), res["scores"].tolist()):
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
                        dets.append((name, (x1, y1, x2, y2), float(sc)))
        keep = []
        for d in sorted(dets, key=lambda t: -t[2]):
            if all(_iou(d[1], k[1]) < 0.5 for k in keep):
                keep.append(d)
        out = []
        for name, (x1, y1, x2, y2), sc in keep:
            out.append((name, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64), sc, None))
        return out


def save_exemplars(out_dir: Path, exemplars: list[dict]):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "exemplars.json").write_text(json.dumps(exemplars, indent=2), encoding="utf-8")


def load_exemplars(out_dir: Path) -> list[dict]:
    return json.loads((Path(out_dir) / "exemplars.json").read_text(encoding="utf-8"))
