"""구조물 기하 변형: 기울어짐(tilt) · 쓰러짐(fall) · 넘어져 눕기(knockdown).

신호등·표지판·가로등·중앙분리대 같은 '세워진 구조물'은 SDXL(ControlNet Canny)로는
윤곽선이 고정되어 자세를 바꿀 수 없으므로, 객체를 잘라내 기하 변형한 뒤 합성한다.
  1) 객체 마스크: Seg/OBB 폴리곤, Detect는 GrabCut
  2) 빈 자리 복원: SDXL Inpainting(있으면) 또는 OpenCV inpaint + 주변 질감 노이즈
  3) 밑둥(pivot) 기준 회전 / 눕히기 → 접지 그림자 · 파편 추가
  4) 라벨 재계산: 변형된 마스크로 bbox/폴리곤/OBB 생성, 클래스 재매핑 가능
"""
from __future__ import annotations

import cv2
import numpy as np

from ..labels import Annotation


def object_mask(img, ann: Annotation, task: str) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    if task in ("segment", "obb"):
        cv2.fillPoly(m, [np.round(ann.pts).astype(np.int32)], 1.0)
        return m
    from .copypaste import _grabcut_mask

    x1, y1, x2, y2 = map(int, map(round, ann.bbox()))
    pad = max(2, int(0.06 * max(x2 - x1, y2 - y1)))
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    m[Y1:Y2, X1:X2] = _grabcut_mask(img[Y1:Y2, X1:X2].copy(), pad)
    return m


def fill_background(img, mask, rng, engine=None, prompt="", negative="", seed=0, avoid=None,
                    long_side=1024, steps=25) -> np.ndarray:
    """객체가 빠진 자리를 복원."""
    mu8 = (cv2.dilate((mask > 0.3).astype(np.uint8), np.ones((9, 9), np.uint8)) * 255)
    if engine is not None:
        ys, xs = np.nonzero(mu8)
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        return engine.inpaint_region(img, mu8.astype(np.float32) / 255, box,
                                     prompt or "empty background, road surface, sidewalk, photorealistic",
                                     negative, strength=0.99, steps=steps, guidance=6.0, seed=seed,
                                     long_side=long_side)
    return patch_fill(img, mu8, avoid=avoid)


def patch_fill(img, hole, tile: int = 0, avoid=None):
    """간단한 예시 기반(exemplar) 채우기: 구멍을 타일 단위로 나눠, 주변 경계와 가장 잘 맞는
    이웃 위치의 픽셀을 복사한다. 경계가 많이 알려진 타일부터 채워 나감(onion order).
    도로·하늘·건물처럼 반복 질감이 많은 배경에서 OpenCV inpaint보다 뭉개짐이 적다."""
    h, w = hole.shape
    out = img.copy()
    todo = hole > 0
    if not todo.any():
        return out
    # 다른 객체(같은 종류의 이웃 판넬 등)를 복사해 오지 않도록 출처에서 제외
    forbid = todo.copy() if avoid is None else (todo | (cv2.dilate(avoid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    ys, xs = np.nonzero(todo)
    T = tile or int(np.clip(min(np.ptp(xs) + 1, np.ptp(ys) + 1) * 0.6, 16, 48))
    offs = [(dx * T // 2, dy * T // 2) for dx in range(-6, 7) for dy in range(-4, 5)
            if (dx, dy) != (0, 0) and abs(dx) + abs(dy) >= 2]
    k9 = np.ones((9, 9), np.uint8)
    tiles = {(y // T, x // T) for y, x in zip(ys, xs)}
    src = out.astype(np.float32)
    while tiles:
        known = ~todo
        # 경계가 가장 많이 알려진 타일 선택
        best_t, best_score = None, -1
        for ty, tx in tiles:
            y0, x0 = ty * T, tx * T
            sub = todo[y0:y0 + T, x0:x0 + T]
            if not sub.any():
                best_t, best_score = (ty, tx), 2
                break
            ya, xa = max(0, y0 - 4), max(0, x0 - 4)
            ring = known[ya:y0 + T + 4, xa:x0 + T + 4]
            sc = ring.mean()
            if sc > best_score:
                best_t, best_score = (ty, tx), sc
        tiles.discard(best_t)
        ty, tx = best_t
        y0, x0 = ty * T, tx * T
        th = np.zeros((h, w), np.uint8)
        th[y0:y0 + T, x0:x0 + T] = todo[y0:y0 + T, x0:x0 + T]
        if not th.any():
            continue
        ring = (cv2.dilate(th, k9) > 0) & known
        py, px = np.nonzero(th)
        ry, rx = np.nonzero(ring)
        if len(ry) < 4:
            ry, rx = np.nonzero((cv2.dilate(th, np.ones((21, 21), np.uint8)) > 0) & known)
        best, bd = None, None
        for dx, dy in offs:
            sy, sx = py + dy, px + dx
            if sy.min() < 0 or sx.min() < 0 or sy.max() >= h or sx.max() >= w:
                continue
            if todo[sy, sx].any() or forbid[sy, sx].any():
                continue
            qy, qx = ry + dy, rx + dx
            ok = (qy >= 0) & (qx >= 0) & (qy < h) & (qx < w)
            if ok.mean() < 0.8:
                continue
            d = float(((src[ry[ok], rx[ok]] - src[qy[ok], qx[ok]]) ** 2).mean()) if ok.any() else 1e9
            d += 0.02 * (abs(dx) + abs(dy))  # 가까운 출처 선호
            if bd is None or d < bd:
                best, bd = (dx, dy), d
        if best is None:
            patch = cv2.inpaint(out, th * 255, 5, cv2.INPAINT_TELEA)
            out[py, px] = patch[py, px]
        else:
            dx, dy = best
            out[py, px] = out[py + dy, px + dx]
        src[py, px] = out[py, px]
        todo[py, px] = False
    # 이음새 완화
    edge = cv2.dilate(hole, np.ones((3, 3), np.uint8)) & ~cv2.erode(hole, np.ones((3, 3), np.uint8))
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    e = cv2.GaussianBlur(edge.astype(np.float32) / 255, (0, 0), 1.0)[..., None] * 0.8
    return (out * (1 - e) + blur * e).clip(0, 255).astype(np.uint8)


def _pivot(mask) -> tuple[float, float]:
    ys, xs = np.nonzero(mask > 0.5)
    y2 = ys.max()
    base = xs[ys >= y2 - max(2, (y2 - ys.min()) * 0.03)]
    return float(np.median(base)), float(y2)


def deform_matrix(mode: str, mask, angle: float, rng) -> np.ndarray:
    """2x3 affine (밑둥 기준).
    tilt / fall : 화면 평면 회전 (신호등·가로등·표지판 기둥이 옆으로 기울거나 쓰러짐)
    lean        : 밑둥 선을 축으로 앞/뒤로 기울어짐 → 높이 cos(θ) 압축 + 약한 전단 (중앙분리대 판넬)
    knockdown   : 완전히 넘어져 바닥에 누움 → 높이 15~25%로 압축 + 전단
    """
    px, py = _pivot(mask)
    if mode in ("tilt", "fall"):
        return cv2.getRotationMatrix2D((px, py), angle, 1.0)
    th = np.radians(abs(angle))
    if mode == "lean":
        squash, sh = float(np.cos(th)), float(np.sign(angle) * np.sin(th) * 0.35)
    else:
        squash, sh = float(rng.uniform(0.15, 0.25)), float(np.sign(angle) * rng.uniform(0.3, 0.8))
    # y' = py + (y - py) * squash,  x' = x + (py - y) * sh * squash
    return np.array([[1.0, -sh * squash, sh * squash * py], [0.0, squash, py * (1 - squash)]], np.float64)


def _shadow(out, mask_w, pivot, strength, rng):
    h, w = mask_w.shape
    px, py = pivot
    sh = np.zeros((h, w), np.float32)
    ys, xs = np.nonzero(mask_w > 0.5)
    if len(xs) == 0:
        return out
    size = max(np.ptp(xs), np.ptp(ys)) + 1
    # 접지 그림자(밑둥 타원) + 객체 아래 드롭 섀도
    cv2.ellipse(sh, (int(px), int(py)), (max(3, int(size * 0.08)), max(2, int(size * 0.025))), 0, 0, 360, 1.0, -1)
    M = np.float32([[1, 0, size * 0.02], [0, 1, size * 0.03]])
    sh = np.maximum(sh, cv2.warpAffine(mask_w, M, (w, h)) * 0.6)
    sh = cv2.GaussianBlur(sh, (0, 0), max(1.5, size * 0.015)) * strength * (1 - mask_w)
    return (out.astype(np.float32) * (1 - sh[..., None])).clip(0, 255).astype(np.uint8)


def _debris(out, mask_w, pivot, rng, amount=1.0):
    h, w = mask_w.shape
    ys, xs = np.nonzero(mask_w > 0.5)
    if len(xs) == 0:
        return out
    size = max(np.ptp(xs), np.ptp(ys)) + 1
    out = out.copy()
    n = int(40 * amount + size * 0.3 * amount)
    # 밑둥 주변 + 쓰러진 몸체를 따라 파편
    idx = rng.integers(0, len(xs), n)
    cx = np.concatenate([np.full(n // 2, pivot[0]), xs[idx[: n - n // 2]]])
    cy = np.concatenate([np.full(n // 2, pivot[1]), ys[idx[: n - n // 2]]])
    spread = size * 0.06
    for x, y in zip(cx + rng.normal(0, spread, len(cx)), cy + rng.normal(0, spread * 0.4, len(cy))):
        r = max(1, int(rng.uniform(0.004, 0.012) * size))
        g = int(rng.uniform(40, 170))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (int(x), int(y)), r, (g, g, g + int(rng.integers(-8, 8))), -1, cv2.LINE_AA)
    # 밑둥 균열/파손 자국
    px, py = map(int, pivot)
    for _ in range(int(3 + 4 * amount)):
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(0.05, 0.15) * size
        ex, ey = int(px + np.cos(ang) * L), int(py + np.sin(ang) * L * 0.35)
        cv2.line(out, (px, py), (ex, ey), (35, 35, 35), max(1, int(size * 0.004)), cv2.LINE_AA)
    return out


def deform_object(img, anns: list[Annotation], idx: list[int], task: str, mode: str, rng,
                  angle_range=(15, 30), engine=None, bg_prompt="", negative="", seed=0,
                  debris: float = 1.0, shadow: float = 0.45, direction: int | None = None,
                  long_side: int = 1024, steps: int = 25):
    """anns[idx] 객체들을 변형. 반환 (이미지, 새 라벨 리스트, 변형된 인덱스)."""
    from .copypaste import mask_to_ann

    h, w = img.shape[:2]
    masks = [object_mask(img, anns[i], task) for i in idx]
    union = np.clip(np.sum(masks, axis=0), 0, 1) if masks else np.zeros((h, w), np.float32)
    others = np.zeros((h, w), np.uint8)
    for j, a in enumerate(anns):
        if j not in idx:
            if task in ("segment", "obb"):
                cv2.fillPoly(others, [np.round(a.pts).astype(np.int32)], 1)
            else:
                x1, y1, x2, y2 = map(int, map(round, a.bbox()))
                others[max(0, y1):y2, max(0, x1):x2] = 1
    bg = fill_background(img, union, rng, engine, bg_prompt, negative, seed, avoid=others, long_side=long_side,
                         steps=steps) if masks else img.copy()
    out = bg
    new_anns = [a.copy() for a in anns]
    done = []
    sign = direction if direction in (-1, 1) else rng.choice([-1, 1])  # 여러 칸은 같은 방향으로
    for i, m in zip(idx, masks):
        if m.sum() < 16:
            continue
        s_i = sign
        if direction is None and mode in ("tilt", "fall"):
            # 무게중심(가로 암·신호등 헤드) 쪽으로 기울거나 쓰러짐. 오른쪽이면 시계 방향(음수 각)
            px, _ = _pivot(m)
            cx = float(np.nonzero(m > 0.5)[1].mean())
            if abs(cx - px) > 2:
                s_i = -1 if cx > px else 1
        ang = s_i * rng.uniform(*angle_range)
        M = deform_matrix(mode, m, ang, rng)
        obj = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mw = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        if mode in ("knockdown", "lean"):  # 눕혀진 면은 하늘을 향해 약간 밝아짐
            obj = cv2.convertScaleAbs(obj, alpha=1.05, beta=6)
        pv = _pivot(m)
        out = _shadow(out, mw, pv, shadow * (0.8 if mode in ("tilt", "lean") else 1.2), rng)
        a = cv2.GaussianBlur(mw, (0, 0), 0.8)[..., None]
        out = (out * (1 - a) + obj * a).clip(0, 255).astype(np.uint8)
        if mode in ("fall", "knockdown") and debris > 0:
            out = _debris(out, mw, pv, rng, debris)
        na = mask_to_ann(mw, anns[i].cls, task)
        if na is not None:
            na.kpts = warp_kpts(anns[i].kpts, M, w, h)
            new_anns[i] = na
            done.append(i)
    return out, new_anns, done


def warp_kpts(kpts, M, w, h):
    """2x3 또는 3x3 행렬로 키포인트 변환. 이미지 밖으로 나가면 v=0."""
    if kpts is None:
        return None
    k = kpts.astype(np.float64).copy()
    vis = k[:, 2] > 0
    if vis.any():
        M3 = np.vstack([M, [0, 0, 1]]) if M.shape[0] == 2 else M
        p = np.hstack([k[vis, :2], np.ones((vis.sum(), 1))]) @ M3.T
        k[vis, :2] = p[:, :2] / p[:, 2:3]
    inside = (k[:, 0] >= 0) & (k[:, 0] < w) & (k[:, 1] >= 0) & (k[:, 1] < h)
    k[~(vis & inside)] = 0
    return k
