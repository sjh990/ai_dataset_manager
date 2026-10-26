"""이미지 입출력 (Windows 한글 경로 대응)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def imread(path) -> np.ndarray | None:
    """cv2.imread는 Windows에서 비ASCII 경로를 못 읽으므로 imdecode 사용."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img: np.ndarray, jpeg_quality: int = 95) -> None:
    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(jpeg_quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"이미지 인코딩 실패: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def image_size(path) -> tuple[int, int] | None:
    """(w, h)를 헤더만 읽어 빠르게 반환."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        img = imread(path)
        if img is None:
            return None
        return img.shape[1], img.shape[0]
