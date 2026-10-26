"""절차적(Procedural) 변형 효과 — GPU 없이 즉시 동작, 구도·객체 위치를 100% 유지.

생성형(SDXL) 결과보다 사실감은 낮지만 빠르고 라벨이 완벽히 보존되어
① GPU 없는 PC에서의 대량 생성 ② SDXL 결과와 섞어 쓰는 용도로 쓴다.
"""
from __future__ import annotations

import cv2
import numpy as np


def _f(img):
    return img.astype(np.float32) / 255.0


def _u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _smooth_noise(h, w, rng, scale=64):
    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    n = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), max(h, w) / 40)
    return (n - n.min()) / (np.ptp(n) + 1e-6)


def rain(img, rng, intensity=0.6):
    h, w = img.shape[:2]
    x = _f(img)
    # 흐린 날 톤: 채도↓ 밝기↓ 약간 푸른빛
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1 - 0.45 * intensity
    x = _f(cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR))
    x = x * (1 - 0.25 * intensity) + np.array([0.04, 0.02, 0.0], np.float32) * intensity
    # 젖은 노면: 국소 대비↑ + 반사광 얼룩
    wet = _smooth_noise(h, w, rng, 48)[..., None]
    x = x + (x - cv2.GaussianBlur(x, (0, 0), 3)) * 0.8 * intensity * wet
    # 빗줄기
    n = int(w * h / 900 * intensity)
    streak = np.zeros((h, w), np.float32)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    streak[ys, xs] = rng.uniform(0.5, 1.0, n)
    L = max(9, int(min(h, w) * 0.04)) | 1
    ang = rng.uniform(-15, 15)
    ker = np.zeros((L, L), np.float32)
    ker[:, L // 2] = 1
    M = cv2.getRotationMatrix2D((L / 2 - .5, L / 2 - .5), ang, 1)
    ker = cv2.warpAffine(ker, M, (L, L))
    streak = cv2.filter2D(streak, -1, ker / max(ker.sum(), 1e-6) * 3)
    streak = cv2.GaussianBlur(streak, (0, 0), 0.7)
    x = x + streak[..., None] * 0.55 * intensity
    x = cv2.GaussianBlur(x, (0, 0), 0.6 * intensity)
    return _u8(x)


def snow(img, rng, intensity=0.6):
    h, w = img.shape[:2]
    x = _f(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    # 적설: 밝고 평평한(수평면) 영역일수록 흰색으로 덮음
    flat = 1 - np.clip(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 2), cv2.CV_32F, ksize=3) ** 2 * 400, 0, 1)
    cover = np.clip((flat * 0.7 + _smooth_noise(h, w, rng, 32) * 0.5) - (1 - intensity), 0, 1)
    cover = cv2.GaussianBlur(cover, (0, 0), 2)[..., None]
    snow_col = np.array([0.97, 0.96, 0.94], np.float32)
    x = x * (1 - cover * 0.85) + snow_col * cover * 0.85
    # 차가운 색감 + 탈채도
    x = x * 0.9 + x.mean(axis=2, keepdims=True) * 0.1
    x = x + np.array([0.03, 0.01, -0.01], np.float32) * intensity
    # 눈송이 (크기 2단계)
    for size, dens in ((1.0, 1 / 500), (2.2, 1 / 3000)):
        n = int(w * h * dens * intensity)
        flakes = np.zeros((h, w), np.float32)
        flakes[rng.integers(0, h, n), rng.integers(0, w, n)] = 1
        flakes = cv2.GaussianBlur(flakes, (0, 0), size) * (size * 6)
        x = x + np.clip(flakes, 0, 1)[..., None] * 0.8
    return _u8(x * (1 - 0.1 * intensity) + 0.1 * intensity)


def fog(img, rng, intensity=0.6):
    h, w = img.shape[:2]
    x = _f(img)
    m = (0.55 + 0.45 * _smooth_noise(h, w, rng, 80)) * intensity
    x = cv2.GaussianBlur(x, (0, 0), 1.5 * intensity) * (1 - m[..., None]) + 0.82 * m[..., None]
    return _u8(x)


def night(img, rng, intensity=0.7):
    h, w = img.shape[:2]
    x = _f(img)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    # 암부 강조 감마 + 청색 틴트(푸르킨예 효과)
    x = np.power(x, 1 + 1.8 * intensity) * (1 - 0.55 * intensity)
    x = x * np.array([1.15, 0.95, 0.75], np.float32)
    # 인공 광원: 밝은 영역 일부를 주황 광원으로
    bright = (lum > np.percentile(lum, 99.3)).astype(np.float32)
    k = int(rng.integers(0, 3))
    if bright.sum() > 0:
        glow = cv2.GaussianBlur(bright, (0, 0), max(h, w) / 120) * (3 + k)
        x = x + np.clip(glow, 0, 1)[..., None] * np.array([0.25, 0.55, 0.85], np.float32) * intensity
    # 비네팅 + 고감도 노이즈
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    x = x * (1 - 0.35 * intensity * np.clip(r - 0.4, 0, 1))[..., None]
    x = x + rng.normal(0, 0.035 * intensity, x.shape).astype(np.float32)
    x = x + rng.normal(0, 0.02 * intensity, (h, w, 1)).astype(np.float32)
    return _u8(x)


def dusk(img, rng, intensity=0.6):
    h, w = img.shape[:2]
    x = _f(img)
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = ((xx / w - .5) * np.cos(ang) + (yy / h - .5) * np.sin(ang)) + .5
    g = np.clip(g, 0, 1)[..., None]
    warm = np.array([0.35, 0.62, 1.0], np.float32)  # BGR 주황
    x = x * (1 - 0.35 * intensity) * (1 - 0.3 * intensity * (1 - g))
    x = x * (1 - 0.4 * intensity) + x * warm * 0.4 * intensity * 1.6
    x = x + warm * 0.12 * intensity * g
    return _u8(x)


def backlight(img, rng, intensity=0.6):
    h, w = img.shape[:2]
    x = _f(img)
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(-0.15, 0.25) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(h, w)
    glare = np.exp(-(d / 0.35) ** 2)[..., None]
    # 실루엣화: 전경 대비↓ 어둡게, 광원 주변 뿌옇게
    x = (x - 0.5) * (1 - 0.35 * intensity) + 0.5 - 0.18 * intensity
    x = x * (1 - glare * intensity) + np.array([0.9, 0.97, 1.0], np.float32) * glare * intensity
    # 렌즈 플레어 고스트
    for i in range(1, 4):
        t = i * 0.28
        gx, gy = cx + (w / 2 - cx) * 2 * t, cy + (h / 2 - cy) * 2 * t
        rr = max(h, w) * rng.uniform(0.02, 0.06)
        gd = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
        x = x + (np.exp(-(gd / rr) ** 2) * 0.18 * intensity)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)
    return _u8(x)


# --- 파손 (대상 영역 mask 안에서만)
def _cracks(h, w, rng, n):
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(8, 30))):
            ang += rng.normal(0, 0.5)
            step = rng.uniform(2, 8) * max(h, w) / 300
            nx, ny = x + np.cos(ang) * step, y + np.sin(ang) * step
            cv2.line(m, (int(x), int(y)), (int(nx), int(ny)), 255, int(rng.integers(1, 3)), cv2.LINE_AA)
            if rng.random() < 0.15:  # 가지
                bx, by, ba = nx, ny, ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
                for _ in range(int(rng.integers(3, 8))):
                    ex, ey = bx + np.cos(ba) * step, by + np.sin(ba) * step
                    cv2.line(m, (int(bx), int(by)), (int(ex), int(ey)), 255, 1, cv2.LINE_AA)
                    bx, by = ex, ey
            x, y = nx, ny
    return m.astype(np.float32) / 255


def _fbm(h, w, rng, base=8, octaves=4):
    """멀티 옥타브 노이즈 (0~1) — 경계가 불규칙한 얼룩/그을음 생성용."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cells = base * (2 ** o)
        small = rng.random((max(2, h * cells // max(h, w)), max(2, w * cells // max(h, w)))).astype(np.float32)
        out += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def damage(img, rng, mask, kind="damage_building", intensity=0.7):
    """mask: (H,W) float 0~1, 1=파손 적용 영역. 영역 경계는 노이즈로 불규칙하게 만든다."""
    h, w = img.shape[:2]
    x = _f(img)
    n1 = _fbm(h, w, rng, 6)
    n2 = _fbm(h, w, rng, 24)
    # 객체별 크기에 비례한 부드러운 경계 (작은 객체도 내부는 충분히 변형)
    mu8 = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(mu8, cv2.DIST_L2, 5)
    nlab, lab = cv2.connectedComponents(mu8)
    soft = np.zeros((h, w), np.float32)
    for i in range(1, nlab):
        sel = lab == i
        f = max(1.5, float(dist[sel].max()) * 0.35)
        soft[sel] = np.clip(dist[sel] / f, 0, 1)
    region = soft * np.clip((n1 - (0.55 - 0.45 * intensity)) * 4, 0, 1)
    m = cv2.GaussianBlur(region, (0, 0), 1.5)[..., None]
    gray = x.mean(axis=2, keepdims=True)
    if kind == "damage_fire":
        soot = np.clip((n2 - 0.35) * 2.5, 0, 1)[..., None]
        char = np.array([0.03, 0.035, 0.045], np.float32)
        y = x * (1 - 0.9 * soot) + char * soot
        y = y * 0.75 + gray * 0.25 * np.array([0.8, 0.9, 1.1], np.float32)  # 탄 갈색 톤
        y = y + (x - cv2.GaussianBlur(x, (0, 0), 2)) * 0.8  # 표면 질감 유지
    elif kind == "flood_road":
        water = np.array([0.30, 0.38, 0.42], np.float32)  # 흙탕물 BGR
        ripple = (n2 - 0.5) * 0.06
        refl = cv2.GaussianBlur(cv2.flip(x, 0), (0, 0), 6)
        y = water * 0.65 + refl * 0.25 + gray * 0.1 + ripple[..., None]
        y = y + (n2[..., None] > 0.92) * 0.15  # 수면 반사광
    else:
        cr = _cracks(h, w, rng, max(4, int(25 * intensity)))
        cr = cv2.GaussianBlur(cr, (0, 0), 0.6)
        debris = (n2 > 0.62).astype(np.float32) * (rng.random((h, w)) < 0.5)
        debris = cv2.GaussianBlur(debris, (0, 0), 0.7)
        dust = np.array([0.55, 0.58, 0.60], np.float32)
        y = x * (1 - 0.45 * np.clip((n2 - 0.4) * 2, 0, 1))[..., None]      # 불규칙 어두운 얼룩(그림자·구멍)
        y = y * (1 - cr[..., None] * 0.9)                                   # 균열
        y = y * (1 - debris[..., None] * 0.5) + dust * debris[..., None] * 0.35  # 잔해
        y = y * 0.7 + (gray * 0.7 + dust * 0.3) * 0.3                        # 먼지·탈채도
    return _u8(x * (1 - m) + y * m)


GLOBAL_FUNCS = {"rain": rain, "snow": snow, "fog": fog, "night": night, "dusk": dusk, "backlight": backlight}
REGION_KEYS = {"damage_building", "damage_fire", "damage_road", "flood_road"}


def apply_procedural(img, key: str, rng, intensity: float = 0.6, mask=None):
    if key in GLOBAL_FUNCS:
        return GLOBAL_FUNCS[key](img, rng, intensity)
    if key in REGION_KEYS:
        if mask is None:
            mask = np.ones(img.shape[:2], np.float32)
        return damage(img, rng, mask, key, intensity)
    raise KeyError(key)
