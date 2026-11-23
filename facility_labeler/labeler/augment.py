"""증강 (드론 도로 영상용). 3번 탭 AI 학습, 4번 탭 내보내기에서 같이 사용.

드론 영상에서 실제로 생기는 변화를 흉내 냄
    크기       : 비행 고도 변화 → 시설물이 크게/작게 보임
    이동       : 비행 경로가 조금씩 달라짐
    회전       : 바람에 의한 기체 흔들림 (roll)
    원근       : 짐벌 각도(아래로 기울인 정도) 변화
    밝기, 대비 : 시간대, 구름, 역광
    색감       : 날씨, 카메라 화이트밸런스
    모션 블러  : 비행 중 흔들림, 빠른 이동
    초점 흐림  : 초점이 살짝 안 맞음
    안개       : 먼 곳(화면 위쪽)일수록 뿌옇게
    그림자     : 건물, 구조물, 구름 그림자
    노이즈, 압축 : 영상 압축(H.264), 어두운 장면 노이즈
    좌우 반전  : 기본 꺼짐 (우측통행 도로 모습이 바뀜, 같은 구간 실증에는 불필요)
위아래 반전은 하지 않음 (하늘과 땅이 바뀌는 장면은 실제로 없음).
"""
import math

import cv2
import numpy as np

# 옵션: 키 → (이름, 설명, 단위, 최댓값, 프리셋 값 {약하게, 보통, 강하게})
OPTIONS = {
    "scale":      ("크기", "비행 고도 변화", "±%", 50, {"약하게": 10, "보통": 20, "강하게": 35}),
    "shift":      ("이동", "비행 경로 차이", "±%", 20, {"약하게": 3, "보통": 6, "강하게": 10}),
    "rotate":     ("회전", "기체 흔들림", "±도", 20, {"약하게": 3, "보통": 6, "강하게": 10}),
    "perspective": ("원근", "짐벌 각도 변화", "%", 15, {"약하게": 0, "보통": 3, "강하게": 6}),
    "brightness": ("밝기, 대비", "시간대, 구름, 역광", "±%", 50, {"약하게": 15, "보통": 25, "강하게": 40}),
    "color":      ("색감", "날씨, 화이트밸런스", "±%", 30, {"약하게": 5, "보통": 10, "강하게": 15}),
    "motion_blur": ("모션 블러", "비행 중 흔들림", "최대 px", 25, {"약하게": 5, "보통": 9, "강하게": 15}),
    "blur":       ("초점 흐림", "초점 어긋남", "최대 px", 9, {"약하게": 0, "보통": 3, "강하게": 5}),
    "haze":       ("안개", "먼 곳일수록 뿌옇게", "최대 %", 60, {"약하게": 0, "보통": 15, "강하게": 30}),
    "shadow":     ("그림자", "건물, 구름 그림자", "최대 %", 60, {"약하게": 0, "보통": 25, "강하게": 40}),
    "noise":      ("노이즈, 압축", "영상 압축, 어두운 장면", "단계", 3, {"약하게": 1, "보통": 1, "강하게": 2}),
    "flip":       ("좌우 반전", "우측통행 모습이 바뀜 (기본 꺼짐)", "켜기 1", 1, {"약하게": 0, "보통": 0, "강하게": 0}),
}
PRESETS = ["약하게", "보통", "강하게"]
APPLY_PROB = 0.5        # 색, 화질 효과(모션 블러, 흐림, 안개, 그림자, 노이즈)는 장마다 50% 확률로


def preset(name="보통"):
    """프리셋 → 설정 {옵션 키: 세기 (0 = 끔)}."""
    return {k: v[4][name] for k, v in OPTIONS.items()}


def describe(settings, copies=None):
    """설정을 사람이 읽는 글자로. 예: '원본 1장당 2장 / 크기 ±20%, 회전 ±6도, ...'"""
    def one(k):
        unit = OPTIONS[k][2]
        if k == "flip":
            return OPTIONS[k][0]
        sign = "±" if unit.startswith("±") else ("최대 " if unit.startswith("최대") else "")
        return f"{OPTIONS[k][0]} {sign}{settings[k]}{unit.replace('±', '').replace('최대 ', '')}"
    on = [one(k) for k in OPTIONS if settings.get(k)]
    head = f"원본 1장당 {copies}장 / " if copies is not None else ""
    return head + (", ".join(on) if on else "증강 없음")


# ---------- 위치 변화 (박스, 키포인트도 같이 옮김) ----------
def geometry_matrix(w, h, settings, rng):
    """크기, 이동, 회전, 원근을 합친 3x3 변환 행렬."""
    s = 1 + rng.uniform(-1, 1) * settings.get("scale", 0) / 100
    a = math.radians(rng.uniform(-1, 1) * settings.get("rotate", 0))
    tx = rng.uniform(-1, 1) * settings.get("shift", 0) / 100 * w
    ty = rng.uniform(-1, 1) * settings.get("shift", 0) / 100 * h
    cx, cy = w / 2, h / 2
    M = np.array([[s * math.cos(a), s * math.sin(a), 0], [-s * math.sin(a), s * math.cos(a), 0], [0, 0, 1.0]])
    M[0, 2] = cx - M[0, 0] * cx - M[0, 1] * cy + tx
    M[1, 2] = cy - M[1, 0] * cx - M[1, 1] * cy + ty
    p = settings.get("perspective", 0) / 100
    if p:                                   # 위쪽 가로폭을 넓히거나 좁힘 (짐벌을 더/덜 숙인 느낌)
        d = rng.uniform(-1, 1) * p * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[-d, 0], [w + d, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, dst) @ M
    if settings.get("flip") and rng.random() < 0.5:
        M = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1.0]]) @ M
    return M


def move_points(points, M):
    p = np.hstack([np.asarray(points, np.float64), np.ones((len(points), 1))]) @ M.T
    return p[:, :2] / p[:, 2:3]


def move_objects(objects, M, w, h):
    """박스와 키포인트를 같은 변환으로 옮김. 화면 밖으로 많이(60% 넘게) 나간 객체는 뺌."""
    out = []
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        c = move_points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], M)
        nx1, ny1 = c.min(0)
        nx2, ny2 = c.max(0)
        full = max((nx2 - nx1) * (ny2 - ny1), 1e-6)
        bx1, by1, bx2, by2 = max(nx1, 0), max(ny1, 0), min(nx2, w - 1), min(ny2, h - 1)
        if bx2 <= bx1 or by2 <= by1 or (bx2 - bx1) * (by2 - by1) < 0.4 * full:
            continue
        kpts = []
        moved = move_points([k[:2] for k in o["kpts"]], M)
        for (x, y), (_, _, v) in zip(moved, o["kpts"]):
            kpts.append([float(x), float(y), 2] if v > 0 and 0 <= x < w and 0 <= y < h else [0.0, 0.0, 0])
        out.append({**o, "box": [float(bx1), float(by1), float(bx2), float(by2)], "kpts": kpts})
    return out


# ---------- 색, 화질 변화 (위치는 그대로) ----------
def photometric(img, settings, rng):
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    b = settings.get("brightness", 0) / 100
    if b:
        out = out * (1 + rng.uniform(-b, b)) + rng.uniform(-b, b) * 60
    c = settings.get("color", 0) / 100
    if c:
        out = out * np.array([1 + rng.uniform(-c, c) for _ in range(3)], np.float32)
    out = np.clip(out, 0, 255)
    shadow = settings.get("shadow", 0) / 100
    if shadow and rng.random() < APPLY_PROB:          # 큰 사각형 그림자 (기울어진 띠)
        mask = np.zeros((h, w), np.float32)
        x0 = rng.uniform(0, w)
        pts = np.int32([[x0, 0], [x0 + rng.uniform(0.2, 0.6) * w, 0],
                        [x0 + rng.uniform(-0.3, 0.9) * w, h], [x0 + rng.uniform(-0.6, 0.3) * w, h]])
        cv2.fillPoly(mask, [pts], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), w * 0.01)
        out = out * (1 - mask[..., None] * rng.uniform(0.3, 1.0) * shadow)
    haze = settings.get("haze", 0) / 100
    if haze and rng.random() < APPLY_PROB:            # 위쪽(먼 곳)일수록 진하게
        depth = np.linspace(1.0, 0.2, h, dtype=np.float32)[:, None, None]
        a = depth * rng.uniform(0.3, 1.0) * haze
        out = out * (1 - a) + 200 * a
    k = settings.get("motion_blur", 0)
    if k >= 3 and rng.random() < APPLY_PROB:          # 한 방향으로 번짐
        size = int(rng.uniform(3, k)) | 1
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2, :] = 1.0 / size
        rot = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, rot, (size, size))
        out = cv2.filter2D(out, -1, kernel / max(kernel.sum(), 1e-6))
    g = settings.get("blur", 0)
    if g >= 1 and rng.random() < APPLY_PROB:
        size = int(rng.uniform(1, g)) * 2 + 1
        out = cv2.GaussianBlur(out, (size, size), 0)
    n = settings.get("noise", 0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if n and rng.random() < APPLY_PROB:
        noise = np.random.default_rng(rng.randint(0, 10 ** 6)).normal(0, 3 * n, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        q = int(rng.uniform(95 - 20 * n, 95))         # 영상 압축 흉내 (JPEG 화질 낮추기)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def augment(img, objects, settings, rng):
    """이미지 1장 증강 → (증강 이미지, 옮긴 객체 목록)."""
    h, w = img.shape[:2]
    M = geometry_matrix(w, h, settings, rng)
    # 빈 곳은 회색으로 채움 (반사로 채우면 라벨 없는 시설물이 복사되어 생김)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return photometric(out, settings, rng), move_objects(objects, M, w, h)
