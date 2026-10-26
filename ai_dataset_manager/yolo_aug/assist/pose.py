from __future__ import annotations

import json
import math
import random
import threading
from pathlib import Path

import cv2
import numpy as np

from ..genai.models import is_installed, model_path
from ..io_utils import imread
from .engines import Cancelled, resolve_device

IN_H, IN_W = 256, 192
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def input_size(model) -> tuple[int, int]:
    bc = getattr(model.config, "backbone_config", None)
    isz = getattr(bc, "image_size", None) if bc is not None else None
    if isinstance(isz, (list, tuple)) and len(isz) == 2:
        return int(isz[0]), int(isz[1])
    return IN_H, IN_W


def _box_cs(box, sz, scale: float = 1.25):
    IN_H, IN_W = sz
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    ar = IN_W / IN_H
    if w > ar * h:
        h = w / ar
    else:
        w = h * ar
    return np.array([cx, cy], np.float64), np.array([w * scale, h * scale], np.float64)


def _affine(center, size, sz, rot_deg: float = 0.0, inv: bool = False) -> np.ndarray:
    IN_H, IN_W = sz
    s = IN_W / size[0]
    r = math.radians(rot_deg)
    c, sn = math.cos(r) * s, math.sin(r) * s
    M = np.array([[c, sn, 0.0], [-sn, c, 0.0]], np.float64)
    M[:, 2] = np.array([IN_W / 2, IN_H / 2]) - M[:, :2] @ center
    if inv:
        return cv2.invertAffineTransform(M)
    return M


def _to_tensor(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def _heatmaps(kpts_crop: np.ndarray, vis: np.ndarray, hm_h: int, hm_w: int, sz, sigma: float = 2.0):
    IN_H, IN_W = sz
    K = len(kpts_crop)
    hm = np.zeros((K, hm_h, hm_w), np.float32)
    wt = np.zeros((K,), np.float32)
    sx, sy = hm_w / IN_W, hm_h / IN_H
    ys, xs = np.mgrid[0:hm_h, 0:hm_w]
    for k in range(K):
        if vis[k] <= 0:
            continue
        x, y = kpts_crop[k, 0] * sx, kpts_crop[k, 1] * sy
        if not (0 <= x < hm_w and 0 <= y < hm_h):
            continue
        hm[k] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
        wt[k] = 1.0
    return hm, wt


def _decode(hm: np.ndarray, sz):
    IN_H, IN_W = sz
    K, H, W = hm.shape
    flat = hm.reshape(K, -1)
    idx = flat.argmax(1)
    conf = flat.max(1)
    ys, xs = (idx // W).astype(np.float64), (idx % W).astype(np.float64)
    for k in range(K):
        x, y = int(xs[k]), int(ys[k])
        if 0 < x < W - 1:
            xs[k] += 0.25 * np.sign(hm[k, y, x + 1] - hm[k, y, x - 1])
        if 0 < y < H - 1:
            ys[k] += 0.25 * np.sign(hm[k, y + 1, x] - hm[k, y - 1, x])
    pts = np.stack([xs * IN_W / W, ys * IN_H / H], 1)
    return pts, conf


class _Cache:
    def __init__(self, cap: int = 64):
        self.cap, self.d = cap, {}

    def get(self, p):
        if p not in self.d:
            if len(self.d) >= self.cap:
                self.d.pop(next(iter(self.d)))
            self.d[p] = imread(p)
        return self.d[p]


def _sample(cache, item, flip_idx, train: bool, rng: random.Random, sz):
    IN_H, IN_W = sz
    img_path, box, kpts = item
    img = cache.get(img_path)
    center, size = _box_cs(box, sz)
    rot = 0.0
    k = kpts.copy()
    if train:
        size = size * rng.uniform(0.8, 1.2)
        center = center + np.array([rng.uniform(-0.08, 0.08) * size[0], rng.uniform(-0.08, 0.08) * size[1]])
        if rng.random() < 0.5:
            rot = rng.uniform(-30, 30)
    M = _affine(center, size, sz, rot)
    crop = cv2.warpAffine(img, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    pc = np.hstack([k[:, :2], np.ones((len(k), 1))]) @ M.T
    vis = k[:, 2].copy()
    if train and rng.random() < 0.5:
        crop = crop[:, ::-1].copy()
        pc[:, 0] = IN_W - 1 - pc[:, 0]
        if flip_idx and len(flip_idx) == len(pc):
            pc, vis = pc[flip_idx], vis[flip_idx]
    if train:
        a = rng.uniform(0.75, 1.25)
        b = rng.uniform(-25, 25)
        crop = np.clip(crop.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return _to_tensor(crop), pc, vis


def _load_model(src: str, K: int, device: str):
    from transformers import VitPoseForPoseEstimation
    try:
        lab = {i: f"kpt_{i}" for i in range(K)}
        m = VitPoseForPoseEstimation.from_pretrained(src, num_labels=K, id2label=lab,
                                                     label2id={v: k for k, v in lab.items()},
                                                     ignore_mismatched_sizes=True, local_files_only=True)
    except TypeError:
        m = VitPoseForPoseEstimation.from_pretrained(src, num_labels=K, local_files_only=True)
    return m.to(device)


def _forward(model, x, torch):
    kw = {}
    bc = getattr(model.config, "backbone_config", None)
    if bc is not None and getattr(bc, "num_experts", 1) > 1:
        kw["dataset_index"] = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    return model(pixel_values=x, **kw).heatmaps


def train_pose(project, samples: list, out_dir: Path, epochs: int, batch: int, device: str, models_root,
               progress, log, cancel: threading.Event, val_ratio: float = 0.1, seed: int = 0, lr: float = 2e-4) -> dict:
    import torch

    root = Path(models_root) if models_root else None
    if not is_installed("vitpose_base", root):
        raise RuntimeError("ViTPose base model is required (Models / GPU -> vitpose_base).")
    K = len(project.kpt_names)
    if K == 0:
        raise RuntimeError("The project has no keypoint schema.")
    if len(samples) < 4:
        raise RuntimeError("At least 4 objects with keypoints are needed.")
    dev = resolve_device(device)
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(samples) * val_ratio)))
    val = [samples[i] for i in idx[:n_val]]
    train = [samples[i] for i in idx[n_val:]]
    model = _load_model(str(model_path("vitpose_base", root)), K, dev)
    sz = input_size(model)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per = max(1, math.ceil(len(train) / batch))
    total = steps_per * int(epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    flip_idx = list(project.flip_idx)
    cache = _Cache()
    step = 0
    best = -1.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(int(epochs)):
        rng.shuffle(train)
        model.train()
        tot = 0.0
        for b0 in range(0, len(train), batch):
            if cancel.is_set():
                raise Cancelled()
            xs, hms, wts = [], [], []
            for it in train[b0:b0 + batch]:
                x, pc, vis = _sample(cache, it, flip_idx, True, rng, sz)
                xs.append(x)
                hms.append(pc)
                wts.append(vis)
            x = torch.from_numpy(np.stack(xs)).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = _forward(model, x, torch)
            hh, ww = pred.shape[-2:]
            tg, tw = zip(*[_heatmaps(pc, vis, hh, ww, sz) for pc, vis in zip(hms, wts)])
            tg = torch.from_numpy(np.stack(tg)).to(dev)
            tw = torch.from_numpy(np.stack(tw)).to(dev)[:, :, None, None]
            loss = (((pred.float() - tg) ** 2) * tw).mean() * 1000
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tot += loss.item()
            progress(step, total, f"epoch {ep + 1}/{epochs}")
        pck = _pck(model, val, cache, torch, dev, sz)
        log(f"epoch {ep + 1}/{epochs}  loss {tot / steps_per:.4f}  val PCK@0.1 {pck:.3f}")
        if pck >= best:
            best = pck
            model.save_pretrained(str(out_dir / "pose_model"))
    meta = dict(K=K, input=list(sz), val_pck=round(best, 4), n_train=len(train), n_val=len(val),
                kpt_names=project.kpt_names, flip_idx=flip_idx)
    (out_dir / "pose_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _pck(model, val, cache, torch, dev, sz) -> float:
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for it in val:
            img = cache.get(it[0])
            pts, conf = _predict_one(model, img, it[1], torch, dev, sz)
            k = it[2]
            x1, y1, x2, y2 = it[1]
            thr = 0.1 * max(x2 - x1, y2 - y1)
            for j in range(len(k)):
                if k[j, 2] > 0:
                    tot += 1
                    ok += float(np.hypot(*(pts[j] - k[j, :2]))) <= thr
    model.train()
    return ok / max(tot, 1)


def _predict_one(model, img, box, torch, dev, sz):
    IN_H, IN_W = sz
    center, size = _box_cs(box, sz)
    M = _affine(center, size, sz)
    crop = cv2.warpAffine(img, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    x = torch.from_numpy(_to_tensor(crop)[None]).to(dev)
    hm = _forward(model, x, torch).float().cpu().numpy()[0]
    pts, conf = _decode(hm, sz)
    Mi = _affine(center, size, sz, inv=True)
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ Mi.T
    return pts, conf


class PosePredictor:
    def __init__(self, model_dir: Path, device: str = "auto"):
        import torch
        from transformers import VitPoseForPoseEstimation

        self.torch = torch
        self.dev = resolve_device(device)
        self.model = VitPoseForPoseEstimation.from_pretrained(str(Path(model_dir) / "pose_model"),
                                                              local_files_only=True).to(self.dev).eval()
        self.sz = input_size(self.model)

    def __call__(self, bgr: np.ndarray, boxes: list, kpt_thr: float = 0.3) -> list[tuple[np.ndarray, float]]:
        out = []
        h, w = bgr.shape[:2]
        with self.torch.no_grad():
            for box in boxes:
                pts, conf = _predict_one(self.model, bgr, box, self.torch, self.dev, self.sz)
                k = np.zeros((len(pts), 3))
                k[:, :2] = pts
                inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
                good = (conf >= kpt_thr) & inside
                k[good, 2] = 2
                k[~good] = 0
                out.append((k, float(np.mean(conf))))
        return out
