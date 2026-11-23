"""동영상에서 프레임(이미지)을 뽑습니다.

- 간격(초)마다 1장씩 저장합니다. 파일 이름: <영상이름>_f<프레임번호 6자리>.jpg
- 시작초 ~ 끝초 구간만 뽑을 수 있습니다 (앞뒤 필요 없는 부분 자르기).
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".ts", ".m4v"}


def _ascii_path(path):
    """OpenCV는 윈도우에서 한글 경로 영상을 못 여는 경우가 있어 짧은 영문 경로로 바꿉니다."""
    s = str(path)
    if s.isascii() or sys.platform != "win32":
        return s, None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(s, buf, 32768) and buf.value.isascii():
            return buf.value, None
    except Exception:
        pass
    # 짧은 경로도 안 되면 임시 폴더로 복사해서 엽니다.
    tmp = tempfile.mkdtemp(prefix="labeler_video_")
    dst = os.path.join(tmp, "video" + Path(s).suffix.lower())
    shutil.copyfile(s, dst)
    return dst, tmp


def video_info(path):
    """영상 길이, 프레임 수, fps, 해상도."""
    s, tmp = _ascii_path(path)
    cap = cv2.VideoCapture(s)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"fps": fps, "frames": n, "seconds": n / fps,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def plan_frames(n_frames, step, first=0, last=None):
    """저장할 프레임 번호 목록. first ~ last 프레임 구간에서 step 마다.

    간격 위치는 항상 0번 프레임 기준이라, 구간을 잘라도 같은 영상이면 같은 프레임 번호가 나옵니다.
    """
    last = n_frames - 1 if last is None else min(last, n_frames - 1)
    return list(range(-(-first // step) * step, last + 1, step))      # first 이상인 첫 간격 위치부터


def missing_frames(planned, existing, step):
    """planned 중 근처(간격의 절반 이내)에 이미 있는 프레임이 없는 것만."""
    have = sorted(existing)
    if not have:
        return list(planned)
    import bisect
    half = step // 2
    out = []
    for f in planned:
        i = bisect.bisect_left(have, f - half)
        if not (i < len(have) and have[i] <= f + half):
            out.append(f)
    return out


def save_image(path, img, quality=95):
    """한글 경로에도 저장되도록 imencode 사용."""
    ok, buf = cv2.imencode(Path(path).suffix, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return ok


def read_image(path):
    """한글 경로에서도 읽히도록 imdecode 사용."""
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def extract(video_path, out_dir, every_sec=0.5, progress=None, should_stop=None, start_sec=0.0, end_sec=None,
            skip_near=()):
    """프레임을 뽑아 out_dir 에 저장하고, 저장한 (프레임번호, 파일경로) 목록을 돌려줍니다.

    start_sec ~ end_sec: 이 구간만 뽑음 (None 이면 영상 끝까지).
    skip_near: 이미 있는 프레임 번호들. 간격 위치 근처(간격의 절반 이내)에 이미 있으면 새로 뽑지 않음
               (예전 작업에서 간격 위치 옆 프레임을 쓴 경우 거의 같은 이미지가 2장 생기지 않게).
    """
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(video_path)
    if info is None:
        raise RuntimeError(f"영상 열기 실패: {video_path}")
    step = max(1, round(every_sec * info["fps"]))
    first = max(0, int(round(start_sec * info["fps"])))
    last_frame = None if end_sec is None else int(round(end_sec * info["fps"]))
    wanted = missing_frames(plan_frames(info["frames"], step, first, last_frame), skip_near, step)
    wanted_set = set(wanted)
    last = wanted[-1] if wanted else -1

    s, tmp = _ascii_path(video_path)
    cap = cv2.VideoCapture(s)
    saved = []
    try:
        frame_no = 0
        if wanted and wanted[0] > 0:                 # 구간 시작 전은 건너뜀
            cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_no != wanted[0]:                # 이동이 정확하지 않으면 처음부터 grab
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_no = 0
        while frame_no <= last:
            if should_stop and should_stop():
                break
            if frame_no in wanted_set:
                ok, img = cap.read()          # 필요한 프레임만 실제로 디코딩
                if not ok:
                    break
                dst = out_dir / f"{video_path.stem}_f{frame_no:06d}.jpg"
                if not dst.exists():
                    save_image(dst, img)
                saved.append((frame_no, dst))
                if progress:
                    progress(len(saved), len(wanted), f"{video_path.name}: {len(saved)}/{len(wanted)}장")
            elif not cap.grab():              # 건너뛸 프레임은 grab 으로 빠르게 넘김
                break
            frame_no += 1
    finally:
        cap.release()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return saved, info, step
