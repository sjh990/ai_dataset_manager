from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..io_utils import IMG_EXTS, imwrite

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts", ".mts", ".m2ts", ".wmv", ".flv", ".webm", ".mpg", ".mpeg"}


@dataclass
class FrameConfig:
    mode: str = "seconds"
    interval: float = 1.0
    start_sec: float = 0.0
    end_sec: float = 0.0
    max_frames: int = 0
    long_side: int = 0
    skip_similar: float = 0.0
    skip_blur: float = 0.0
    image_format: str = "jpg"
    jpeg_quality: int = 95
    prefix: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def list_videos(src) -> list[Path]:
    p = Path(src)
    if p.is_file():
        return [p] if p.suffix.lower() in VIDEO_EXTS else []
    if not p.is_dir():
        return []
    return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS)


def _short_path(path: Path) -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 32768)
        if n and buf.value.isascii():
            return buf.value
    except Exception:
        pass
    return None


class _Opened:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = None
        self.cap = None

    def __enter__(self):
        s = str(self.path)
        if sys.platform == "win32" and not s.isascii():
            s = _short_path(self.path) or s
            if not s.isascii():
                self.tmp = tempfile.mkdtemp(prefix="yoloaug_vid_")
                dst = os.path.join(self.tmp, "video" + self.path.suffix.lower())
                shutil.copyfile(self.path, dst)
                s = dst
        self.cap = cv2.VideoCapture(s)
        return self.cap

    def __exit__(self, *exc):
        if self.cap is not None:
            self.cap.release()
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)


def probe(path) -> dict:
    with _Opened(Path(path)) as cap:
        if not cap.isOpened():
            return dict(ok=False, fps=0.0, frames=0, width=0, height=0, seconds=0.0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = fps if 0 < fps < 1000 else 30.0
    return dict(ok=True, fps=float(fps), frames=n, width=w, height=h, seconds=n / fps if n > 0 else 0.0)


def estimate_count(info: dict, cfg: FrameConfig) -> int:
    if not info.get("ok") or info["frames"] <= 0:
        return 0
    fps, n = info["fps"], info["frames"]
    a = int(round(cfg.start_sec * fps))
    b = int(round(cfg.end_sec * fps)) if cfg.end_sec > 0 else n
    b = min(b, n)
    if b <= a:
        return 0
    step = _step(cfg, fps)
    k = (b - a + step - 1) // step
    return min(k, cfg.max_frames) if cfg.max_frames > 0 else k


def _step(cfg: FrameConfig, fps: float) -> int:
    if cfg.mode == "frames":
        return max(1, int(round(cfg.interval)))
    return max(1, int(round(cfg.interval * fps)))


def _resize(img: np.ndarray, long_side: int) -> np.ndarray:
    if long_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    r = long_side / m
    return cv2.resize(img, (max(1, int(round(w * r))), max(1, int(round(h * r)))), interpolation=cv2.INTER_AREA)


def _thumb(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def _sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = 640 / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _unique_stem(out_dir: Path, stem: str, used: set) -> str:
    s, k = stem, 1
    while s in used:
        s = f"{stem}_{k}"
        k += 1
    used.add(s)
    return s


def extract_frames(video, out_dir, cfg: FrameConfig, progress=None, log=print, cancel: threading.Event | None = None,
                   used_stems: set | None = None) -> dict:
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    used = used_stems if used_stems is not None else {p.stem for p in out_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    stats = dict(video=video.name, saved=0, skipped_similar=0, skipped_blur=0, read=0, files=[])
    with _Opened(video) as cap:
        if not cap.isOpened():
            log(f"[WARN] cannot open video: {video}")
            stats["error"] = "open_failed"
            return stats
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = fps if 0 < fps < 1000 else 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = int(round(cfg.start_sec * fps))
        end = int(round(cfg.end_sec * fps)) if cfg.end_sec > 0 else (n if n > 0 else 1 << 62)
        step = _step(cfg, fps)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if pos != start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                pos = 0
        else:
            pos = 0
        ext = ".png" if cfg.image_format == "png" else ".jpg"
        prefix = cfg.prefix or video.stem
        last = None
        total = max(1, (min(end, n) if n > 0 else end) - start)
        while pos < end:
            if cancel.is_set():
                break
            want = pos >= start and (pos - start) % step == 0
            if want:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None
            if not ok:
                break
            if want and frame is not None:
                stats["read"] += 1
                img = _resize(frame, cfg.long_side)
                keep = True
                if cfg.skip_blur > 0 and _sharpness(img) < cfg.skip_blur:
                    stats["skipped_blur"] += 1
                    keep = False
                if keep and cfg.skip_similar > 0:
                    t = _thumb(img)
                    if last is not None and float(np.mean(np.abs(t - last))) < cfg.skip_similar:
                        stats["skipped_similar"] += 1
                        keep = False
                    else:
                        last = t
                if keep:
                    stem = _unique_stem(out_dir, f"{prefix}_f{pos:06d}", used)
                    dst = out_dir / f"{stem}{ext}"
                    imwrite(dst, img, cfg.jpeg_quality)
                    stats["saved"] += 1
                    stats["files"].append(dst.name)
                    if cfg.max_frames > 0 and stats["saved"] >= cfg.max_frames:
                        break
                if stats["read"] % 5 == 0:
                    progress(min(pos - start, total), total, video.name)
            pos += 1
    progress(total, total, video.name)
    log(f"{video.name}: saved {stats['saved']} frames (similar skipped {stats['skipped_similar']}, "
        f"blur skipped {stats['skipped_blur']})")
    return stats


def extract_many(videos: list, out_dir, cfg: FrameConfig, progress=None, log=print,
                 cancel: threading.Event | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    used = {p.stem for p in out_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    results = []
    vids = [Path(v) for v in videos]
    for i, v in enumerate(vids):
        if cancel.is_set():
            break

        def prog(d, t, m, i=i):
            progress(i * 1000 + int(1000 * d / max(t, 1)), len(vids) * 1000, f"[{i + 1}/{len(vids)}] {m}")

        results.append(extract_frames(v, out_dir, cfg, prog, log, cancel, used))
    saved = sum(r["saved"] for r in results)
    log(f"done: {saved} frames from {len(results)} video(s) -> {out_dir}")
    return dict(videos=len(results), saved=saved, results=results, config=cfg.to_dict(), out_dir=str(out_dir))
