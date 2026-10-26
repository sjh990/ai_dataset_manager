"""증강: 모자이크 → 기하 변환(단일 행렬) → 색상/화질."""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .labels import Annotation, transform_annotations

Sample = tuple[np.ndarray, list[Annotation]]


# ----------------------------------------------------------------------------- mosaic
def mosaic4(items: list[Sample], out_w: int, out_h: int, rng, task, fill=114,
            min_vis=0.25, min_size=2.0, flip_idx=None) -> tuple[np.ndarray, list[Annotation], int]:
    sw, sh = out_w, out_h
    canvas = np.full((2 * sh, 2 * sw, 3), fill, dtype=np.uint8)
    xc = int(rng.uniform(0.5 * sw, 1.5 * sw))
    yc = int(rng.uniform(0.5 * sh, 1.5 * sh))
    all_anns: list[Annotation] = []
    dropped = 0
    for i, (img, anns) in enumerate(items[:4]):
        h, w = img.shape[:2]
        r = min(sw / w, sh / h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        if i == 0:
            ox, oy = xc - nw, yc - nh
            x1a, y1a, x2a, y2a = max(ox, 0), max(oy, 0), xc, yc
        elif i == 1:
            ox, oy = xc, yc - nh
            x1a, y1a, x2a, y2a = xc, max(oy, 0), min(xc + nw, 2 * sw), yc
        elif i == 2:
            ox, oy = xc - nw, yc
            x1a, y1a, x2a, y2a = max(ox, 0), yc, xc, min(yc + nh, 2 * sh)
        else:
            ox, oy = xc, yc
            x1a, y1a, x2a, y2a = xc, yc, min(xc + nw, 2 * sw), min(yc + nh, 2 * sh)
        if x2a <= x1a or y2a <= y1a:
            dropped += len(anns)
            continue
        canvas[y1a:y2a, x1a:x2a] = im[y1a - oy:y2a - oy, x1a - ox:x2a - ox]
        M = np.array([[nw / w, 0, ox], [0, nh / h, oy], [0, 0, 1]], dtype=np.float64)
        a2, d = transform_annotations(anns, M, (x1a, y1a, x2a, y2a), task, min_vis, min_size * 2, flip_idx)
        all_anns += a2
        dropped += d
    out = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)
    Mh = np.diag([0.5, 0.5, 1.0])
    all_anns, d = transform_annotations(all_anns, Mh, (0, 0, sw, sh), task, 0.0, min_size, flip_idx)
    return out, all_anns, dropped + d


# ----------------------------------------------------------------------------- geometric
def geometric(img, anns, ops: dict, rng, task, fill=114, min_vis=0.25, min_size=2.0, flip_idx=None):
    h, w = img.shape[:2]
    M_total = np.eye(3)

    def on(k):
        return k in ops and rng.random() < ops[k]["p"]

    # 무손실 연산 (flip, rot90) — 이미지는 정확히, 점은 행렬로
    if on("hflip"):
        img = cv2.flip(img, 1)
        M_total = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float64) @ M_total
    if on("vflip"):
        img = cv2.flip(img, 0)
        M_total = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float64) @ M_total
    if on("rot90"):
        k = int(rng.integers(1, 4))
        for _ in range(k):  # 시계방향 90°: (x, y) -> (H - y, x)
            ch, cw = img.shape[:2]
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            M_total = np.array([[0, -1, ch], [1, 0, 0], [0, 0, 1]], dtype=np.float64) @ M_total
    h, w = img.shape[:2]

    # 연속 변환 — 하나의 행렬로 합성해 1회만 보간
    C = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], dtype=np.float64)
    Pm = np.eye(3)
    use_persp = False
    if on("perspective"):
        s = ops["perspective"]["strength"]
        Pm[2, 0] = rng.uniform(-s, s)
        Pm[2, 1] = rng.uniform(-s, s)
        use_persp = True
    angle, sc = 0.0, 1.0
    if on("rotate"):
        md = ops["rotate"]["max_deg"]
        angle = rng.uniform(-md, md)
    if on("scale"):
        sc = rng.uniform(ops["scale"]["min_scale"], ops["scale"]["max_scale"])
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D((0, 0), angle, sc)
    S = np.eye(3)
    if on("shear"):
        md = ops["shear"]["max_deg"]
        S[0, 1] = math.tan(math.radians(rng.uniform(-md, md)))
        S[1, 0] = math.tan(math.radians(rng.uniform(-md, md)))
    T = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], dtype=np.float64)
    if on("translate"):
        f = ops["translate"]["fraction"]
        T[0, 2] += rng.uniform(-f, f) * w
        T[1, 2] += rng.uniform(-f, f) * h
    M = T @ S @ R @ Pm @ C
    if on("random_crop"):
        a = rng.uniform(ops["random_crop"]["min_area"], 1.0)
        cw, ch = w * math.sqrt(a), h * math.sqrt(a)
        x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
        Cr = np.array([[w / cw, 0, -x0 * w / cw], [0, h / ch, -y0 * h / ch], [0, 0, 1]], dtype=np.float64)
        M = Cr @ M
    if not np.allclose(M, np.eye(3)):
        border = (int(fill),) * 3
        if use_persp:
            img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)
        else:
            img = cv2.warpAffine(img, M[:2], (w, h), flags=cv2.INTER_LINEAR, borderValue=border)
        M_total = M @ M_total
    if np.allclose(M_total, np.eye(3)):
        return img, anns, 0
    new, dropped = transform_annotations(anns, M_total, (0, 0, w, h), task, min_vis, min_size, flip_idx)
    return img, new, dropped


# ----------------------------------------------------------------------------- photometric
def _lut(img, table):
    return cv2.LUT(img, np.clip(table, 0, 255).astype(np.uint8))


def _hsv(img, prm, rng):
    r = rng.uniform(-1, 1, 3) * [prm["h_gain"], prm["s_gain"], prm["v_gain"]] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float64)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), _lut(sat, x * r[1]), _lut(val, x * r[2])))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _fog(img, intensity, rng):
    h, w = img.shape[:2]
    small = rng.random((max(2, h // 64), max(2, w // 64))).astype(np.float32)
    mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = cv2.GaussianBlur(mask, (0, 0), max(h, w) / 30)
    mask = (mask - mask.min()) / (np.ptp(mask) + 1e-6)
    mask = (0.4 + 0.6 * mask) * intensity
    fog = np.full_like(img, 225, dtype=np.float32)
    out = img.astype(np.float32) * (1 - mask[..., None]) + fog * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _motion_kernel(k, angle):
    ker = np.zeros((k, k), dtype=np.float32)
    ker[k // 2, :] = 1.0
    Mr = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    ker = cv2.warpAffine(ker, Mr, (k, k))
    return ker / max(ker.sum(), 1e-6)


def _odd(rng, lo, hi):
    hi = max(lo, int(hi))
    vals = list(range(lo, hi + 1, 2)) or [lo]
    return int(rng.choice(vals))


def photometric(img, anns, ops: dict, rng, fill=114):
    def on(k):
        return k in ops and rng.random() < ops[k]["p"]

    if on("hsv"):
        img = _hsv(img, ops["hsv"], rng)
    if on("brightness_contrast"):
        p = ops["brightness_contrast"]
        alpha = 1 + rng.uniform(-p["contrast"], p["contrast"])
        beta = rng.uniform(-p["brightness"], p["brightness"]) * 255
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if on("gamma"):
        g = rng.uniform(ops["gamma"]["min_gamma"], ops["gamma"]["max_gamma"])
        img = _lut(img, 255.0 * (np.arange(256) / 255.0) ** (1.0 / g))
    if on("clahe_aug"):
        from .preprocess import clahe
        img = clahe(img, ops["clahe_aug"]["clip_limit"], 8)
    if on("to_gray"):
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if on("sharpen"):
        a = rng.uniform(0.1, ops["sharpen"]["amount"])
        blur = cv2.GaussianBlur(img, (0, 0), 2)
        img = cv2.addWeighted(img, 1 + a, blur, -a, 0)
    if on("fog"):
        img = _fog(img, rng.uniform(0.05, ops["fog"]["max_intensity"]), rng)
    if on("gaussian_blur"):
        k = _odd(rng, 3, ops["gaussian_blur"]["max_kernel"])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if on("motion_blur"):
        k = _odd(rng, 3, ops["motion_blur"]["max_kernel"])
        img = cv2.filter2D(img, -1, _motion_kernel(k, rng.uniform(0, 180)))
    if on("gaussian_noise"):
        sigma = rng.uniform(1, ops["gaussian_noise"]["max_sigma"])
        noise = rng.normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if on("cutout"):
        img, anns = _cutout(img, anns, ops["cutout"], rng)
    if on("jpeg"):
        q = int(rng.integers(int(ops["jpeg"]["min_quality"]), 101))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img, anns


def _cutout(img, anns, prm, rng):
    img = img.copy()
    h, w = img.shape[:2]
    n = int(rng.integers(1, int(prm["max_holes"]) + 1))
    holes = []
    for _ in range(n):
        s = rng.uniform(0.02, prm["max_size"])
        hw, hh = int(w * s), int(h * s)
        x, y = int(rng.integers(0, max(1, w - hw))), int(rng.integers(0, max(1, h - hh)))
        img[y:y + hh, x:x + hw] = rng.integers(64, 192, 3)
        holes.append((x, y, x + hw, y + hh))
    keep = []
    for a in anns:
        x1, y1, x2, y2 = a.bbox()
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        covered = 0.0
        for hx1, hy1, hx2, hy2 in holes:
            iw = max(0, min(x2, hx2) - max(x1, hx1))
            ih = max(0, min(y2, hy2) - max(y1, hy1))
            covered = max(covered, iw * ih / area)
        if covered < 0.6:
            keep.append(a)
    return img, keep


# ----------------------------------------------------------------------------- full
def augment(img, anns, ops: dict, common: dict, rng, task, flip_idx=None,
            mosaic_provider: Callable[[int], list[Sample]] | None = None):
    """ops: 활성화된 증강 {key: params(p 포함)}. 반환 (이미지, 라벨, 제거된 라벨 수)."""
    fill, mv, ms = int(common["fill"]), float(common["min_visibility"]), float(common["min_size"])
    dropped = 0
    anns = [a.copy() for a in anns]
    if "mosaic" in ops and mosaic_provider is not None and rng.random() < ops["mosaic"]["p"]:
        others = mosaic_provider(3)
        if len(others) == 3:
            h, w = img.shape[:2]
            order = [(img, anns)] + others
            perm = rng.permutation(4)
            img, anns, d = mosaic4([order[i] for i in perm], w, h, rng, task, fill, mv, ms, flip_idx)
            dropped += d
    img, anns, d = geometric(img, anns, ops, rng, task, fill, mv, ms, flip_idx)
    dropped += d
    img, anns = photometric(img, anns, ops, rng, fill)
    return img, anns, dropped
