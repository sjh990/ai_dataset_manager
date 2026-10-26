from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from ..genai.models import weight_file

RFDETR_DET = {"nano": ("RFDETRNano", "rfdetr_nano"), "small": ("RFDETRSmall", "rfdetr_small"),
              "medium": ("RFDETRMedium", "rfdetr_medium")}
RFDETR_SEG = {"nano": ("RFDETRSegNano", "rfdetr_seg_nano"), "small": ("RFDETRSegSmall", "rfdetr_seg_small"),
              "medium": ("RFDETRSegMedium", "rfdetr_seg_medium")}
ULTRA = {("detect", "nano"): "ultra_yolo26n", ("segment", "nano"): "ultra_yolo26n_seg",
         ("pose", "nano"): "ultra_yolo26n_pose", ("detect", "small"): "ultra_yolo26s",
         ("segment", "small"): "ultra_yolo26s_seg", ("pose", "small"): "ultra_yolo26s_pose"}
ULTRA_NAME = {"detect": "yolo26{}.pt", "segment": "yolo26{}-seg.pt", "pose": "yolo26{}-pose.pt"}


class Cancelled(Exception):
    pass


def resolve_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _has(mod: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def rfdetr_available() -> tuple[bool, str]:
    if _has("rfdetr") and _has("pytorch_lightning") and _has("torchvision"):
        return True, ""
    return False, "rfdetr[train] / torchvision is not installed. Run install_genai."


def ultralytics_available() -> tuple[bool, str]:
    if _has("ultralytics"):
        return True, ""
    return False, "ultralytics is not installed. pip install -r requirements-ultralytics.txt (AGPL-3.0)."


def mask_to_polygon(mask: np.ndarray, min_area: float = 16.0) -> np.ndarray | None:
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    poly = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    return poly.astype(np.float64) if len(poly) >= 3 else None


def train_rfdetr(ds_root: Path, out_dir: Path, task: str, size: str, epochs: int, batch: int, device: str,
                 models_root, progress, log, cancel: threading.Event) -> Path:
    import rfdetr
    import rfdetr.training as T
    from pytorch_lightning import Callback

    table = RFDETR_SEG if task == "segment" else RFDETR_DET
    cls_name, key = table.get(size, table["nano"])
    w = weight_file(key, Path(models_root) if models_root else None)
    if w is None:
        log(f"[WARN] pretrained weights '{key}' not found in the model folder -> training from scratch "
            "(needs much more data/epochs). Download it in Models / GPU.")
    model = getattr(rfdetr, cls_name)(pretrain_weights=str(w) if w else None)

    class _CB(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch_, batch_idx):
            nb = max(1, int(trainer.num_training_batches or 1))
            progress(trainer.current_epoch * nb + batch_idx + 1, epochs * nb,
                     f"epoch {trainer.current_epoch + 1}/{epochs}")
            if cancel.is_set():
                trainer.should_stop = True

        def on_train_epoch_end(self, trainer, pl_module):
            m = trainer.callback_metrics
            loss = float(m.get("loss", m.get("train/loss", float("nan"))))
            ap = m.get("val/segm_mAP_50" if task == "segment" else "val/mAP_50")
            ap = float(ap) if ap is not None else float("nan")
            log(f"epoch {trainer.current_epoch + 1}/{epochs}  loss {loss:.3f}  val mAP50 {ap:.3f}")

    orig = T.build_trainer

    def patched(*a, **k):
        tr = orig(*a, **k)
        tr.callbacks.append(_CB())
        return tr

    T.build_trainer = patched
    out_dir = Path(out_dir)
    try:
        model.train(dataset_dir=str(ds_root), dataset_file="yolo", epochs=int(epochs), batch_size=int(batch),
                    output_dir=str(out_dir), num_workers=0 if sys.platform == "win32" else 2,
                    device=resolve_device(device), tensorboard=False, progress_bar=None,
                    checkpoint_interval=max(1, int(epochs)))
    finally:
        T.build_trainer = orig
    if cancel.is_set():
        raise Cancelled()
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "last_ema.pth", "checkpoint_best_regular.pth"):
        if (out_dir / name).exists():
            return out_dir / name
    raise RuntimeError("RF-DETR training finished without a checkpoint.")


class RFDETRPredictor:
    def __init__(self, ckpt: Path, device: str = "auto"):
        import rfdetr
        self.model = rfdetr.from_checkpoint(str(ckpt), device=resolve_device(device))
        try:
            import torch
            if resolve_device(device) == "cuda":
                self.model.optimize_for_inference(dtype=torch.float16)
        except Exception:
            pass

    def __call__(self, bgr: np.ndarray, threshold: float) -> list[tuple[str, np.ndarray, float, np.ndarray | None]]:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        d = self.model.predict(rgb, threshold=threshold, include_source_image=False)
        names = d.data.get("class_name") if hasattr(d, "data") else None
        out = []
        for i in range(len(d)):
            x1, y1, x2, y2 = map(float, d.xyxy[i])
            name = str(names[i]) if names is not None else str(int(d.class_id[i]))
            poly = None
            if getattr(d, "mask", None) is not None:
                poly = mask_to_polygon(d.mask[i])
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
            out.append((name, poly if poly is not None else box, float(d.confidence[i]), None))
        return out


def train_ultralytics(ds_root: Path, out_dir: Path, task: str, size: str, epochs: int, batch: int, imgsz: int,
                      device: str, models_root, progress, log, cancel: threading.Event) -> Path:
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO

    s = "s" if size in ("small", "medium") else "n"
    key = ULTRA.get((task, "small" if s == "s" else "nano"))
    w = weight_file(key, Path(models_root) if models_root else None) if key else None
    src = str(w) if w else ULTRA_NAME[task].format(s)
    if w is None:
        log(f"[WARN] {src} not found in the model folder; Ultralytics will try to download it.")
    model = YOLO(src)
    state = {"epoch": 0}

    def on_batch(trainer):
        nb = max(1, len(getattr(trainer, "train_loader", []) or [1]))
        state["i"] = state.get("i", 0) + 1
        progress(min(trainer.epoch * nb + (state["i"] % nb or nb), epochs * nb), epochs * nb,
                 f"epoch {trainer.epoch + 1}/{epochs}")
        if cancel.is_set():
            raise Cancelled()

    def on_epoch(trainer):
        state["i"] = 0
        if trainer.epoch + 1 > epochs or state.get("last") == trainer.epoch:
            return
        state["last"] = trainer.epoch
        m = getattr(trainer, "metrics", {}) or {}
        ap = m.get("metrics/mAP50(M)", m.get("metrics/mAP50(P)", m.get("metrics/mAP50(B)", float("nan"))))
        log(f"epoch {trainer.epoch + 1}/{epochs}  val mAP50 {float(ap):.3f}")

    model.add_callback("on_train_batch_end", on_batch)
    model.add_callback("on_fit_epoch_end", on_epoch)
    dev = resolve_device(device)
    model.train(data=str(Path(ds_root) / "data.yaml"), epochs=int(epochs), imgsz=int(imgsz), batch=int(batch),
                device=0 if dev == "cuda" else "cpu", project=str(out_dir), name="train", exist_ok=True,
                plots=False, verbose=False, workers=0 if sys.platform == "win32" else 2, amp=dev == "cuda")
    wdir = Path(out_dir) / "train" / "weights"
    for name in ("best.pt", "last.pt"):
        if (wdir / name).exists():
            return wdir / name
    raise RuntimeError("Ultralytics training finished without weights.")


class UltralyticsPredictor:
    def __init__(self, weights: Path, device: str = "auto"):
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = 0 if resolve_device(device) == "cuda" else "cpu"

    def __call__(self, bgr: np.ndarray, threshold: float) -> list[tuple[str, np.ndarray, float, np.ndarray | None]]:
        r = self.model.predict(bgr, conf=threshold, verbose=False, device=self.device)[0]
        names = r.names
        out = []
        if r.boxes is None:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        polys = r.masks.xy if getattr(r, "masks", None) is not None else None
        kpts = r.keypoints.data.cpu().numpy() if getattr(r, "keypoints", None) is not None else None
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
            if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                pts = np.asarray(polys[i], np.float64)
            k = None
            if kpts is not None and i < len(kpts):
                k = np.asarray(kpts[i], np.float64)
                if k.shape[1] == 2:
                    k = np.hstack([k, np.full((len(k), 1), 1.0)])
            out.append((str(names[int(cls[i])]), pts, float(conf[i]), k))
        return out
