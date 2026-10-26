"""도로 장면 합성기 (테스트·시연용): 신호등 + 중앙분리대 판넬, YOLO Segment 라벨 포함.

사용: python tools/make_road_scene.py <출력폴더> --n 4
클래스: 0 traffic_light, 1 median_barrier
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_aug.io_utils import imwrite  # noqa: E402
from yolo_aug.labels import Annotation, format_labels  # noqa: E402

NAMES = ["traffic_light", "median_barrier"]


def _noise(h, w, rng, scale, amp):
    s = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return (cv2.resize(s, (w, h), interpolation=cv2.INTER_CUBIC) - 0.5) * amp


def render(W=1280, H=720, seed=0, day=True):
    rng = np.random.default_rng(seed)
    vx, vy = W * rng.uniform(0.46, 0.54), H * 0.40
    img = np.zeros((H, W, 3), np.float32)
    # 하늘
    yy = np.linspace(0, 1, int(vy) + 1)[:, None]
    top, bot = np.array([200, 150, 95]), np.array([235, 222, 205])
    img[: int(vy) + 1] = (top * (1 - yy) + bot * yy)[:, None, :]
    cloud = np.clip(_noise(int(vy) + 1, W, rng, 90, 2.2) + _noise(int(vy) + 1, W, rng, 30, 0.8), 0, 1)
    img[: int(vy) + 1] += cloud[..., None] * 45
    # 원경 건물
    x = 0
    while x < W:
        bw = rng.uniform(40, 130)
        bh = rng.uniform(40, 170) * (0.6 + 0.8 * abs(x - vx) / W)
        col = rng.uniform(120, 175) + rng.uniform(-10, 10, 3)
        x1, x2 = int(x), int(min(W, x + bw))
        y1 = int(vy - bh)
        img[y1:int(vy), x1:x2] = col
        for wy in range(y1 + 6, int(vy) - 6, 12):
            for wx in range(x1 + 5, x2 - 5, 11):
                if rng.random() < 0.8:
                    img[wy:wy + 5, wx:wx + 5] = col * rng.uniform(0.7, 0.95)
        x += bw + rng.uniform(0, 12)
    img[: int(vy)] = cv2.GaussianBlur(img[: int(vy)], (0, 0), 1.2)
    # 지면 좌표계: t(0=소실점, 1=화면 하단), u(-1~1 도로 폭)
    half = W * 0.75

    def P(u, t):
        return vx + u * half * t, vy + (H - vy) * t

    ys = np.arange(int(vy), H)
    t_row = (ys - vy) / (H - vy)
    xs = np.arange(W)[None, :]
    u_map = (xs - vx) / (half * np.maximum(t_row[:, None], 1e-3))
    road = np.abs(u_map) <= 1.0
    asphalt = 88 + _noise(H - int(vy), W, rng, 6, 18) + _noise(H - int(vy), W, rng, 60, 20)
    side = 150 + _noise(H - int(vy), W, rng, 4, 14)
    g = np.where(road, asphalt, side)
    # 보도블록 줄눈
    tiles = ((np.floor(u_map * 6) + np.floor(np.log(np.maximum(t_row[:, None], 1e-3)) * 9)) % 2) * 6
    g = np.where(road, g, g - tiles)
    img[int(vy):] = np.stack([g, g * 1.01, g * 1.03], -1)
    # 연석
    for s in (-1, 1):
        pts = np.array([P(s * 1.0, 0.02), P(s * 1.0, 1.0), P(s * 1.03, 1.0), P(s * 1.03, 0.02)], np.int32)
        cv2.fillPoly(img, [pts], (175, 178, 180))
    # 차선
    for u in (0.35, 0.72):
        k = 0.02
        while k < 1.0:
            t1, t2 = k, min(1.0, k * 1.18)
            q = np.array([P(u - 0.012, t1), P(u - 0.012, t2), P(u + 0.012, t2), P(u + 0.012, t1)], np.int32)
            cv2.fillPoly(img, [q], (225, 228, 230))
            k *= 1.40
    # 중앙선(황색 실선) 위에 중앙분리대
    um = -0.42
    for off in (-0.02, 0.0):
        q = np.array([P(um + off, 0.02), P(um + off, 1.0), P(um + off + 0.012, 1.0), P(um + off + 0.012, 0.02)],
                     np.int32)
        cv2.fillPoly(img, [q], (40, 170, 215))
    anns: list[Annotation] = []
    posts = [1.05]
    while posts[-1] > 0.07:
        posts.append(posts[-1] / 1.28)
    HP = 0.16 * H  # 최근접 판넬 높이
    panels = []
    for k in range(len(posts) - 1, 0, -1):  # 원경부터 그림
        t1, t2 = posts[k], posts[k - 1]
        (x1, y1), (x2, y2) = P(um, t1), P(um, t2)
        h1, h2 = HP * t1, HP * t2
        quad = np.array([[x1, y1], [x2, y2], [x2, y2 - h2], [x1, y1 - h1]], np.float64)
        panels.append((k, quad, t1, t2))
    for k, quad, t1, t2 in panels:
        q = quad.astype(np.int32)
        cv2.fillPoly(img, [q], (212, 214, 216))
        # 음영(아래쪽 어둡게) + 반사띠(노랑/검정)
        for f, col in ((0.18, (30, 190, 235)), (0.28, (35, 35, 35)), (0.38, (30, 190, 235))):
            a, b = quad[0] + (quad[3] - quad[0]) * f, quad[1] + (quad[2] - quad[1]) * f
            c, d = quad[1] + (quad[2] - quad[1]) * (f + 0.07), quad[0] + (quad[3] - quad[0]) * (f + 0.07)
            cv2.fillPoly(img, [np.array([a, b, c, d], np.int32)], col)
        a, b = quad[0] + (quad[3] - quad[0]) * 0.85, quad[1] + (quad[2] - quad[1]) * 0.85
        cv2.fillPoly(img, [np.array([a, b, quad[2], quad[3]], np.int32)], (60, 60, 64))  # 상단 레일
        for pt, hh, tt in ((quad[0], quad[0][1] - quad[3][1], t1), (quad[1], quad[1][1] - quad[2][1], t2)):
            pw = max(2, int(10 * tt))
            cv2.rectangle(img, (int(pt[0] - pw / 2), int(pt[1] - hh * 1.08)), (int(pt[0] + pw / 2), int(pt[1])),
                          (235, 236, 238), -1)
        if t2 > 0.28:  # 가까운 판넬만 라벨
            anns.append(Annotation(1, quad.copy()))
    # 신호등 (보도 위 기둥 + 도로 위 가로 암 + 수평형 신호등)
    def traffic_light(u, t, side):
        bx, by = P(u, t)
        m = np.zeros((H, W), np.uint8)
        ph = 0.72 * H * t
        pw = max(4, int(0.022 * W * t))
        top = by - ph
        arm_len = 0.42 * W * t
        ax2 = bx - side * arm_len
        hw, hh = 0.16 * W * t, 0.045 * W * t
        hx1, hx2 = sorted([ax2 - side * hw * 0.1, ax2 + side * hw * 0.9])
        hy1 = top + ph * 0.02
        # 기둥 (원통 음영)
        for i in range(pw):
            c = 70 + 60 * np.sin(np.pi * (i + 0.5) / pw)
            cv2.line(img, (int(bx - pw / 2 + i), int(by)), (int(bx - pw / 2 + i), int(top)), (c, c, c + 4), 1)
        cv2.rectangle(m, (int(bx - pw / 2), int(top)), (int(bx + pw / 2), int(by)), 255, -1)
        cv2.rectangle(img, (int(bx - pw), int(by - pw * 1.2)), (int(bx + pw), int(by)), (95, 95, 100), -1)  # 받침
        cv2.rectangle(m, (int(bx - pw), int(by - pw * 1.2)), (int(bx + pw), int(by)), 255, -1)
        at = max(3, int(pw * 0.6))
        cv2.line(img, (int(bx), int(top + at)), (int(ax2), int(top + at)), (105, 105, 110), at, cv2.LINE_AA)
        cv2.line(m, (int(bx), int(top + at)), (int(ax2), int(top + at)), 255, at)
        cv2.rectangle(img, (int(hx1), int(hy1)), (int(hx2), int(hy1 + hh)), (28, 28, 30), -1)
        cv2.rectangle(m, (int(hx1), int(hy1)), (int(hx2), int(hy1 + hh)), 255, -1)
        r = int(hh * 0.33)
        cols = [(40, 40, 230), (30, 120, 150), (60, 110, 60)] if side > 0 else [(60, 110, 60), (30, 120, 150), (40, 40, 230)]
        lit = 0
        for j, col in enumerate(cols):
            cx = int(hx1 + (hx2 - hx1) * (j + 0.5) / 3)
            cy = int(hy1 + hh / 2)
            cv2.circle(img, (cx, cy), r, col if j == lit else tuple(c * 0.35 for c in col), -1, cv2.LINE_AA)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float64)
        anns.append(Annotation(0, poly))

    traffic_light(0.92, 0.60, +1)
    if rng.random() < 0.8:
        traffic_light(-1.10, 0.42, -1)
    # 조명·센서 느낌
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1 - 0.18 * np.clip(r - 0.5, 0, 1))[..., None]
    img += rng.normal(0, 2.5, img.shape)
    out = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    for i in range(a.n):
        img, anns = render(seed=i)
        imwrite(out / "images" / f"road_{i:03d}.jpg", img)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        (out / "labels" / f"road_{i:03d}.txt").write_text(format_labels(anns, "segment", img.shape[1], img.shape[0]))
    (out / "data.yaml").write_text("names: [traffic_light, median_barrier]\n")
    print("생성:", out)


if __name__ == "__main__":
    main()
