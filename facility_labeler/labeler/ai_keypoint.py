"""키포인트 AI (ViTPose).

동작 방식
    1. 박스를 1.25배 넓혀 잘라낸 뒤 256x192 크기로 맞춥니다.
    2. ViTPose 가 점 4개 각각의 '있을 법한 위치 지도(heatmap)'를 만듭니다.
    3. 지도에서 가장 진한 곳을 키포인트로 씁니다.
그래서 박스가 먼저 있어야 키포인트를 예측할 수 있습니다.
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .frames import read_image

IN_H, IN_W = 256, 192
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def get_device(choice="auto"):
    import torch
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


# ---------- 잘라내기 (박스 → 모델 입력) ----------
def crop_matrix(box, scale=1.25, rotate_deg=0.0, inverse=False):
    """박스를 모델 입력 크기(192x256)로 옮기는 변환 행렬."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    # 모델 입력 비율(3:4)에 맞게 박스를 늘림
    if w > h * IN_W / IN_H:
        h = w * IN_H / IN_W
    else:
        w = h * IN_W / IN_H
    s = IN_W / (w * scale)
    r = math.radians(rotate_deg)
    a, b = math.cos(r) * s, math.sin(r) * s
    M = np.array([[a, b, 0.0], [-b, a, 0.0]])
    M[:, 2] = np.array([IN_W / 2, IN_H / 2]) - M[:, :2] @ np.array([cx, cy])
    return cv2.invertAffineTransform(M) if inverse else M


def to_tensor(crop_bgr):
    rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def make_heatmaps(points, visible, hm_h, hm_w, sigma=2.0):
    """정답 점 위치에 동그란 '언덕'을 그린 지도 (학습 정답)."""
    hm = np.zeros((len(points), hm_h, hm_w), np.float32)
    weight = np.zeros(len(points), np.float32)
    ys, xs = np.mgrid[0:hm_h, 0:hm_w]
    for k, (x, y) in enumerate(points):
        x, y = x * hm_w / IN_W, y * hm_h / IN_H
        if visible[k] and 0 <= x < hm_w and 0 <= y < hm_h:
            hm[k] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
            weight[k] = 1.0
    return hm, weight


def read_heatmaps(hm):
    """지도에서 가장 진한 곳 → 점 위치(모델 입력 좌표)와 신뢰도."""
    K, H, W = hm.shape
    flat = hm.reshape(K, -1)
    idx, conf = flat.argmax(1), flat.max(1)
    xs, ys = (idx % W) * IN_W / W, (idx // W) * IN_H / H
    return np.stack([xs, ys], 1), conf


# ---------- 모델 ----------
def load_model(path, device, new_head=False):
    from transformers import VitPoseForPoseEstimation
    if new_head:   # 처음 학습: 출력 점 개수를 4개로 바꿔서 불러옴
        labels = {i: f"kpt_{i}" for i in range(C.NUM_KEYPOINTS)}
        model = VitPoseForPoseEstimation.from_pretrained(
            str(path), num_labels=C.NUM_KEYPOINTS, id2label=labels, label2id={v: k for k, v in labels.items()},
            ignore_mismatched_sizes=True, local_files_only=True)
    else:
        model = VitPoseForPoseEstimation.from_pretrained(str(path), local_files_only=True)
    return model.to(device)


def run_model(model, x):
    return model(pixel_values=x).heatmaps


# ---------- 학습 ----------
def collect_samples(items):
    """(이미지경로, 박스, 키포인트) 목록. 키포인트가 1개 이상 있는 객체만."""
    samples = []
    for img, lab in items:
        for o in lab["objects"]:
            if any(k[2] > 0 for k in o["kpts"]):
                samples.append((img, o["box"], np.array(o["kpts"], np.float32)))
    return samples


def train(items, out_dir, epochs=50, batch=8, device="auto", val_ratio=0.15, progress=None, log=print,
          should_stop=None, aug_copies=0, aug_settings=None):
    """aug_copies: epoch 마다 학습 객체를 (1 + aug_copies)번씩 매번 다르게 증강해서 사용."""
    import torch
    samples = collect_samples(items)
    if len(samples) < 4:
        raise RuntimeError("키포인트가 있는 객체 4개 이상 필요")
    device = get_device(device)
    rng = random.Random(0)
    rng.shuffle(samples)
    n_val = max(1, round(len(samples) * val_ratio))
    val, trn = samples[:n_val], samples[n_val:]
    log(f"[키포인트 AI] 학습 객체 {len(trn)}개 x {1 + aug_copies} (증강) / 검증 객체 {len(val)}개, device {device}")
    trn = trn * (1 + aug_copies)

    model = load_model(C.KEYPOINT_BASE_MODEL, device, new_head=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    steps = max(1, math.ceil(len(trn) / batch)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cache = {}
    best, best_epoch, step = -1.0, 0, 0
    history = []      # epoch 마다 [epoch, loss, val PCK]
    for ep in range(epochs):
        model.train()
        rng.shuffle(trn)
        total_loss = 0.0
        for i in range(0, len(trn), batch):
            if should_stop and should_stop():
                raise RuntimeError("사용자 중지")
            xs, targets, weights = [], [], []
            for img_path, box, kpts in trn[i:i + batch]:
                x, pts, vis = _training_crop(cache, img_path, box, kpts, rng, aug_settings)
                xs.append(x)
                targets.append(pts)
                weights.append(vis)
            x = torch.from_numpy(np.stack(xs)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = run_model(model, x)
            hh, ww = pred.shape[-2:]
            hms, wts = zip(*[make_heatmaps(p, v, hh, ww) for p, v in zip(targets, weights)])
            tg = torch.from_numpy(np.stack(hms)).to(device)
            wt = torch.from_numpy(np.stack(wts)).to(device)[:, :, None, None]
            loss = (((pred.float() - tg) ** 2) * wt).mean() * 1000
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            total_loss += loss.item()
            if progress:
                progress(step, steps, f"키포인트 AI epoch {ep + 1}/{epochs}")
        pck = evaluate(model, val, cache, device)
        log(f"[키포인트 AI] epoch {ep + 1}/{epochs}  loss {total_loss:.3f}  val PCK@0.1 {pck:.3f}")
        history.append([ep + 1, round(total_loss, 3), round(pck, 3)])
        if pck >= best:          # 가장 점수가 좋은 epoch 의 모델을 저장
            best, best_epoch = pck, ep + 1
            model.save_pretrained(str(Path(out_dir)))
    return {"val_pck": round(best, 3), "best_epoch": best_epoch, "history": history, "n_train": len(trn),
            "n_val": len(val), "batch": batch, "device": device}


def _image(cache, path):
    if path not in cache:
        if len(cache) > 64:
            cache.pop(next(iter(cache)))
        cache[path] = read_image(path)
    return cache[path]


def _training_crop(cache, img_path, box, kpts, rng, aug=None):
    """학습용 잘라내기 + 증강 (크기, 위치, 회전은 박스 기준으로, 색, 화질은 augment.py 와 같게).
    aug=None 이면 예전 기본값 (크기 ±15%, 위치 ±8%, 회전 ±20도, 밝기)."""
    from . import augment as A
    if aug is None:
        aug = {"scale": 15, "shift": 8, "rotate": 20, "brightness": 25}
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    sh = aug.get("shift", 0) / 100
    dx, dy = rng.uniform(-sh, sh) * bw, rng.uniform(-sh, sh) * bh
    rot = rng.uniform(-1, 1) * aug.get("rotate", 0)
    sc = aug.get("scale", 0) / 100
    M = crop_matrix((x1 + dx, y1 + dy, x2 + dx, y2 + dy), scale=1.25 * (1 + rng.uniform(-sc, sc)), rotate_deg=rot)
    crop = cv2.warpAffine(_image(cache, img_path), M, (IN_W, IN_H))
    crop = A.photometric(crop, {k: v for k, v in aug.items() if k not in ("scale", "shift", "rotate", "perspective", "flip")}, rng)
    pts = np.hstack([kpts[:, :2], np.ones((len(kpts), 1))]) @ M.T
    return to_tensor(crop), pts, kpts[:, 2] > 0


def predict_one(model, img, box, device):
    import torch
    M = crop_matrix(box)
    crop = cv2.warpAffine(img, M, (IN_W, IN_H))
    x = torch.from_numpy(to_tensor(crop)[None]).to(device)
    with torch.no_grad():
        hm = run_model(model, x).float().cpu().numpy()[0]
    pts, conf = read_heatmaps(hm)
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ crop_matrix(box, inverse=True).T
    return pts, conf


def evaluate(model, val, cache, device):
    """PCK@0.1: 예측 점이 정답에서 박스 긴 변의 10% 안에 들어온 비율."""
    model.eval()
    ok = total = 0
    for img_path, box, kpts in val:
        pts, _ = predict_one(model, _image(cache, img_path), box, device)
        limit = 0.1 * max(box[2] - box[0], box[3] - box[1])
        for j in range(len(kpts)):
            if kpts[j, 2] > 0:
                total += 1
                ok += float(np.hypot(*(pts[j] - kpts[j, :2]))) <= limit
    return ok / max(total, 1)


class KeypointAI:
    """학습된 키포인트 AI로 박스 안의 점 4개를 예측합니다."""

    def __init__(self, model_dir, device="auto"):
        self.device = get_device(device)
        self.model = load_model(model_dir, self.device).eval()

    def predict(self, img, box, min_conf=0.3):
        pts, conf = predict_one(self.model, img, box, self.device)
        h, w = img.shape[:2]
        kpts = []
        for (x, y), c in zip(pts, conf):
            ok = c >= min_conf and 0 <= x < w and 0 <= y < h
            kpts.append([float(x), float(y), 2] if ok else [0.0, 0.0, 0])
        return kpts, float(np.mean(conf))
