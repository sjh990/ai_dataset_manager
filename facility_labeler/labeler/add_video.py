"""영상 1개를 데이터셋에 추가합니다: 정한 구간에서 프레임 추출 + 빈 라벨 만들기.
이미 추가한 영상의 구간 바꾸기도 여기 있습니다 (change_range)."""
import re
import shutil
import time
from pathlib import Path

from . import config as C
from . import frames
from .project import empty_label


def add_video(project, video_path, every_sec=0.5, progress=None, should_stop=None, log=print,
              start_sec=0.0, end_sec=None):
    """start_sec ~ end_sec: 영상에서 쓸 구간 (앞뒤 필요 없는 부분 자르기). end_sec None = 영상 끝까지."""
    video_path = Path(video_path)
    have = [int(m.group(1)) for img in project.images()
            if (m := re.fullmatch(re.escape(video_path.stem) + r"_f(\d+)", img.stem))]
    saved, info, step = frames.extract(video_path, project.images_dir, every_sec, progress, should_stop,
                                       start_sec, end_sec, skip_near=have)
    end = info["seconds"] if end_sec is None else min(end_sec, info["seconds"])
    log(f"{video_path.name}: 길이 {info['seconds']:.1f}초 중 {start_sec:.1f}~{end:.1f}초 구간, "
        f"{step}프레임({every_sec}초)마다 1장 → {len(saved)}장 저장")

    for _, img_path in saved:
        if not project.label_path(img_path).exists():   # 이미 라벨이 있으면 그대로 둠
            project.save_label(img_path, empty_label(img_path.name, info["width"], info["height"]))

    project.info["videos"].append({"file": str(video_path), "every_sec": every_sec, "frames_saved": len(saved),
                                   "video_frames": info["frames"], "video_seconds": round(info["seconds"], 1),
                                   "range_sec": [round(start_sec, 1), round(end, 1)],
                                   "added": time.strftime("%Y-%m-%d %H:%M")})
    project.save_info()
    return {"saved": len(saved), "info": info}


# ---------- 이미 추가한 영상의 구간 바꾸기 ----------
def find_video(path):
    """영상 파일 찾기: 저장된 경로 → 없으면 툴 폴더의 raw_data 안 같은 이름. 못 찾으면 None."""
    path = Path(path)
    if path.exists():
        return path
    local = C.APP_DIR / "raw_data" / path.name
    return local if local.exists() else None


def _frame_images(project, stem):
    """이 영상에서 뽑은 이미지들: {프레임번호: 이미지 경로}."""
    out = {}
    for img in project.images():
        m = re.fullmatch(re.escape(stem) + r"_f(\d+)", img.stem)
        if m:
            out[int(m.group(1))] = img
    return out


def range_preview(project, video_index, video_path, start_sec, end_sec):
    """구간을 바꾸면 정리될 이미지 수(상태별)와 새로 뽑을 이미지 수."""
    v = project.info["videos"][video_index]
    info = frames.video_info(video_path)
    first = int(round(start_sec * info["fps"]))
    last = info["frames"] - 1 if end_sec is None else int(round(end_sec * info["fps"]))
    have = _frame_images(project, Path(v["file"]).stem)
    removed = {C.STATUS_HUMAN: 0, C.STATUS_AI: 0, C.STATUS_TODO: 0}
    for f, img in have.items():
        if f < first or f > last:
            lab = project.load_label(img)
            removed[lab["status"] if lab else C.STATUS_TODO] += 1
    step = max(1, round(v["every_sec"] * info["fps"]))
    kept = [f for f in have if first <= f <= last]        # 구간 정리 후 남는 프레임 기준 (change_range 와 같게)
    new = frames.missing_frames(frames.plan_frames(info["frames"], step, first, last), kept, step)
    return {"removed": removed, "remove_total": sum(removed.values()), "add": len(new)}


def change_range(project, video_index, video_path, start_sec, end_sec, progress=None, should_stop=None, log=print):
    """구간 바꾸기: 새 구간 밖 이미지와 라벨은 removed/<날짜>/ 로 옮기고 (지우지 않음, 복구 가능),
    새 구간에서 아직 없는 프레임은 같은 간격으로 새로 뽑습니다."""
    v = project.info["videos"][video_index]
    stem = Path(v["file"]).stem
    info = frames.video_info(video_path)
    first = int(round(start_sec * info["fps"]))
    last = info["frames"] - 1 if end_sec is None else int(round(end_sec * info["fps"]))
    trash = project.root / "removed" / time.strftime("%Y%m%d_%H%M%S")
    moved = 0
    for f, img in _frame_images(project, stem).items():
        if f < first or f > last:
            (trash / "images").mkdir(parents=True, exist_ok=True)
            (trash / "labels").mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), trash / "images" / img.name)
            lab = project.label_path(img)
            if lab.exists():
                shutil.move(str(lab), trash / "labels" / lab.name)
            moved += 1
    log(f"구간 밖 이미지 {moved}장 → {trash if moved else '(없음)'}")
    have = _frame_images(project, stem).keys()          # 남은 프레임: 이 근처는 새로 뽑지 않음
    saved, info, step = frames.extract(video_path, project.images_dir, v["every_sec"], progress, should_stop,
                                       start_sec, end_sec, skip_near=list(have))
    added = 0
    for _, img_path in saved:
        if not project.label_path(img_path).exists():
            project.save_label(img_path, empty_label(img_path.name, info["width"], info["height"]))
            added += 1
    end = info["seconds"] if end_sec is None else min(end_sec, info["seconds"])
    v.update(file=str(video_path), range_sec=[round(start_sec, 1), round(end, 1)],
             frames_saved=len(_frame_images(project, stem)), range_changed=time.strftime("%Y-%m-%d %H:%M"))
    project.save_info()
    log(f"구간 변경 완료: {start_sec:.1f}~{end:.1f}초, 새로 뽑은 이미지 {added}장, 지금 {v['frames_saved']}장")
    return {"moved": moved, "added": added, "trash": str(trash) if moved else None}
