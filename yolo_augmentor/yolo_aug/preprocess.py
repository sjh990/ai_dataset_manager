"""전처리: 한 이미지 → (타일 분할 시) 여러 조각. 모든 split에 동일하게 적용."""
from __future__ import annotations

import cv2
import numpy as np

from .labels import Annotation, filter_small, transform_annotations
from .ops_spec import PREPROCESS_ORDER

Piece = tuple[str, np.ndarray, list[Annotation]]


def _tile_positions(size: int, tile: int, overlap: float) -> list[int]:
    if size <= tile:
        return [0]
    stride = max(1, int(round(tile * (1 - overlap))))
    pos = list(range(0, size - tile + 1, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def _tile(img, anns, prm, task, flip_idx) -> list[Piece]:
    h, w = img.shape[:2]
    tw, th = int(prm["tile_w"]), int(prm["tile_h"])
    if w <= tw and h <= th:
        return [("", img, anns)]
    out: list[Piece] = []
    for yi, y in enumerate(_tile_positions(h, th, prm["overlap"])):
        for xi, x in enumerate(_tile_positions(w, tw, prm["overlap"])):
            cw, ch = min(tw, w - x), min(th, h - y)
            M = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
            t_anns, _ = transform_annotations(anns, M, (0, 0, cw, ch), task,
                                              min_visibility=prm["min_visibility"], min_size=2)
            out.append((f"_t{yi}_{xi}", img[y:y + ch, x:x + cw].copy(), t_anns))
    return out


def resize_image(img, anns, prm, task):
    h, w = img.shape[:2]
    mode = prm["mode"]
    W, H = int(prm["width"]), int(prm["height"])
    if mode == "stretch":
        sx, sy = W / w, H / h
        interp = cv2.INTER_AREA if sx * sy < 1 else cv2.INTER_LINEAR
        out = cv2.resize(img, (W, H), interpolation=interp)
        M = np.diag([sx, sy, 1.0])
        rect = (0, 0, W, H)
    elif mode == "fit_long":
        r = W / max(w, h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        M = np.diag([nw / w, nh / h, 1.0])
        rect = (0, 0, nw, nh)
    else:  # letterbox
        r = min(W / w, H / h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        left, top = (W - nw) // 2, (H - nh) // 2
        out = np.full((H, W, 3), int(prm["pad_value"]), dtype=np.uint8)
        out[top:top + nh, left:left + nw] = rs
        M = np.array([[nw / w, 0, left], [0, nh / h, top], [0, 0, 1]], dtype=np.float64)
        rect = (0, 0, W, H)
    new, _ = transform_annotations(anns, M, rect, task, min_visibility=0.0, min_size=0.0)
    return out, new


def auto_contrast(img, cutoff_pct):
    out = np.empty_like(img)
    for c in range(3):
        ch = img[..., c]
        lo, hi = np.percentile(ch, [cutoff_pct, 100 - cutoff_pct])
        if hi - lo < 1:
            out[..., c] = ch
            continue
        lut = np.clip((np.arange(256) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        out[..., c] = cv2.LUT(ch, lut)
    return out


def clahe(img, clip, grid):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(img, method, strength):
    k = int(strength)
    k = k if k % 2 == 1 else k + 1
    if method == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(img, k, 25 + 5 * k, 25 + 5 * k)
    if method == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img, None, 2 + k, 2 + k, 7, 21)
    return cv2.medianBlur(img, k)


def to_gray3(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def preprocess(img: np.ndarray, anns: list[Annotation], ops: dict[str, dict], task: str,
               rng: np.random.Generator, flip_idx=None, stats: dict | None = None) -> list[Piece]:
    """ops: 활성화된 전처리 {key: params}. 반환: [(접미사, 이미지, 라벨)]"""
    stats = stats if stats is not None else {}
    pieces: list[Piece] = [("", img, anns)]
    if "tile" in ops:
        pieces = _tile(img, anns, ops["tile"], task, flip_idx)
    result: list[Piece] = []
    for suffix, im, an in pieces:
        for key in PREPROCESS_ORDER:
            if key not in ops or key in ("tile", "drop_empty"):
                continue
            prm = ops[key]
            if key == "resize":
                im, an = resize_image(im, an, prm, task)
            elif key == "auto_contrast":
                im = auto_contrast(im, prm["cutoff"])
            elif key == "clahe":
                im = clahe(im, prm["clip_limit"], prm["grid"])
            elif key == "denoise":
                im = denoise(im, prm["method"], prm["strength"])
            elif key == "grayscale":
                im = to_gray3(im)
            elif key == "filter_small":
                an, d = filter_small(an, prm["min_w"], prm["min_h"])
                stats["small_removed"] = stats.get("small_removed", 0) + d
        if "drop_empty" in ops and not an:
            if rng.random() >= ops["drop_empty"]["keep_ratio"]:
                stats["empty_dropped"] = stats.get("empty_dropped", 0) + 1
                continue
        result.append((suffix, im, an))
    return result
