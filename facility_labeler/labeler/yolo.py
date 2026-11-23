"""YOLO11-pose 학습과 영상 추론 (드론에 넣을 파손 탐지 모델).

학습: 4번 탭에서 내보낸 YOLO 데이터셋(data.yaml)으로 YOLO11-pose 학습 → <데이터셋>/yolo/<날짜>/weights/best.pt
추론: 학습한 모델(best.pt)로 영상을 한 프레임씩 추론 → 박스, 키포인트를 그린 결과 영상(mp4) 저장

ultralytics (AGPL-3.0) 사용. 시작 가중치: models/yolo11n-pose.pt (없으면 ultralytics 가 내려받음)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

from . import config as C
from .frames import _ascii_path, video_info

MODEL_SIZES = {"n": "yolo11n-pose.pt (가장 빠름, 드론 실시간 추천)", "s": "yolo11s-pose.pt (조금 더 정확, 느림)",
               "m": "yolo11m-pose.pt (더 정확, 많이 느림)"}


def base_weights(size):
    """시작 가중치 경로. models 폴더에 있으면 그것, 없으면 이름만 (ultralytics 가 내려받음)."""
    name = f"yolo11{size}-pose.pt"
    local = C.MODELS_DIR / name
    return str(local) if local.exists() else name


def _utf8_console():
    """ultralytics 진행 표시 특수문자가 윈도우 한글 콘솔(cp949)에서 멈추지 않게."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------- 학습 ----------
def train(data_yaml, out_root, size="n", epochs=100, imgsz=640, batch=8, device="auto", progress=None, log=print,
          should_stop=None):
    """YOLO11-pose 학습. 결과: out_root/<날짜>/ (weights/best.pt, yolo_info.json, train_log.txt)."""
    from ultralytics import YOLO
    _utf8_console()
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    name = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / name
    lines = []

    def say(text):
        lines.append(f"[{time.strftime('%H:%M:%S')}] {text}")
        log(text)

    dev = 0 if device in ("auto", "cuda") else "cpu"
    if device == "auto":
        import torch
        dev = 0 if torch.cuda.is_available() else "cpu"
    say(f"[YOLO] 데이터: {data_yaml}  (nc = {data.get('nc')}, names = {data.get('names')})")
    say(f"[YOLO] 모델 yolo11{size}-pose, epoch {epochs}, imgsz {imgsz}, batch {batch}, device {dev}")
    model = YOLO(base_weights(size))
    state = {"batches": 0, "per_epoch": None, "history": [], "t0": time.time()}

    def on_batch_end(trainer):
        state["batches"] += 1
        if state["per_epoch"] is None:
            state["per_epoch"] = max(1, len(trainer.train_loader))
        if progress:
            progress(state["batches"], epochs * state["per_epoch"], f"YOLO 학습 epoch {trainer.epoch + 1}/{epochs}")
        if should_stop and should_stop():
            trainer.stop = True

    def on_fit_epoch_end(trainer):
        if trainer.epoch + 1 > epochs or any(h[0] == trainer.epoch + 1 for h in state["history"]):
            return                       # 학습 끝난 뒤 마지막 확인(검증)은 기록하지 않음
        m = trainer.metrics or {}
        box, pose = m.get("metrics/mAP50(B)"), m.get("metrics/mAP50(P)")
        row = [trainer.epoch + 1, round(float(box), 3) if box is not None else None,
               round(float(pose), 3) if pose is not None else None]
        state["history"].append(row)
        say(f"[YOLO] epoch {row[0]}/{epochs}  박스 mAP50 {row[1]}  키포인트 mAP50 {row[2]}")

    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    try:
        model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=dev,
                    project=str(Path(out_root)), name=name, exist_ok=True,
                    workers=0 if sys.platform == "win32" else 2,
                    amp=False,          # GTX 16xx 는 FP16 학습이 불안정 + AMP 확인용 가중치 추가 다운로드 방지
                    plots=False, verbose=False, fliplr=0.0)   # 좌우 반전 없음 (키포인트 순서 기준 유지)
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        say(f"[YOLO] 학습 실패 또는 중지: {e}")
        (out / "train_log.txt").write_text("\n".join(lines), encoding="utf-8")
        raise
    if should_stop and should_stop():
        say("[YOLO] 사용자 중지 (그때까지의 best.pt 저장됨)")
    best = out / "weights" / "best.pt"
    scored = [h for h in state["history"] if h[2] is not None]
    top = max(scored, key=lambda h: h[2]) if scored else [None, None, None]
    info = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "data": str(data_yaml),
            "names": data.get("names"), "size": size, "epochs": epochs, "imgsz": imgsz, "batch": batch,
            "device": str(dev), "minutes": round((time.time() - state["t0"]) / 60, 1),
            "best_epoch": top[0], "box_mAP50": top[1], "pose_mAP50": top[2], "history": state["history"],
            "weights": str(best)}
    (out / "yolo_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"[YOLO] 학습 완료 ({info['minutes']}분): 가장 좋은 epoch {top[0]}, 키포인트 mAP50 {top[2]}  →  {best}")
    (out / "train_log.txt").write_text("\n".join(lines), encoding="utf-8")
    return info


def list_models(out_root):
    """학습한 YOLO 모델 목록 (최신 순)."""
    out = []
    root = Path(out_root)
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True):
            f = d / "yolo_info.json"
            if f.exists() and (d / "weights" / "best.pt").exists():
                info = json.loads(f.read_text(encoding="utf-8"))
                info["weights"] = str(d / "weights" / "best.pt")      # 저장된 경로 대신 지금 폴더 기준 (폴더 이름이 바뀌어도 동작)
                out.append(info)
    return out


# ---------- 영상 추론 ----------
def _color(name):
    """클래스 이름(sign, median, fence) → 툴과 같은 색 (BGR)."""
    for c in C.CLASSES:
        if c["key"] == name:
            h = c["color"].lstrip("#")
            return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)
    return 0, 255, 255


def predict_video(weights, video, out_path, conf=0.25, imgsz=640, device="auto", progress=None, log=print,
                  should_stop=None, start_sec=0.0, end_sec=None):
    """영상을 추론해서 박스, 키포인트를 그린 결과 영상 저장. 프레임별 결과는 같은 이름의 .json 으로.

    start_sec, end_sec: 추론할 구간 (초, end_sec=None 이면 영상 끝까지). json 의 frame 은 원본 영상 기준 번호.
    """
    from ultralytics import YOLO
    _utf8_console()
    info = video_info(video)
    if info is None:
        raise RuntimeError(f"영상 열기 실패: {video}")
    model = YOLO(str(weights))
    dev = None if device == "auto" else (0 if device == "cuda" else "cpu")
    src, tmp = _ascii_path(video)
    cap = cv2.VideoCapture(src)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name("_writing.mp4")          # 한글 경로 대비: 영문 임시 이름으로 쓰고 나중에 이름 변경
    writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                             (info["width"], info["height"]))
    first = min(max(0, int(round(start_sec * info["fps"]))), info["frames"] - 1)
    last = info["frames"] - 1 if end_sec is None else min(info["frames"] - 1, int(round(end_sec * info["fps"])))
    total = max(1, last - first + 1)
    if first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rng = "전체" if first == 0 and last == info["frames"] - 1 else         f"구간 {first / info['fps']:.1f}초 ~ {last / info['fps']:.1f}초"
    log(f"[추론] {Path(video).name}: {rng}, {total}프레임 (영상 전체 {info['frames']}프레임), conf {conf}, imgsz {imgsz}")
    frames, n, t0 = [], 0, time.time()
    try:
        while n < total:
            if should_stop and should_stop():
                log("[추론] 사용자 중지 (여기까지 저장)")
                break
            ok, img = cap.read()
            if not ok:
                break
            r = model.predict(img, conf=conf, imgsz=imgsz, device=dev, verbose=False)[0]
            objs = _draw(img, r)
            writer.write(img)
            frames.append({"frame": first + n, "objects": objs})
            n += 1
            if progress and (n % 5 == 0 or n == total):
                progress(n, total, f"추론 {n}/{total} 프레임 ({n / max(time.time() - t0, 1e-6):.1f} fps)")
    finally:
        cap.release()
        writer.release()
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    if out_path.exists():
        out_path.unlink()
    tmp_out.rename(out_path)
    found = sum(1 for f in frames if f["objects"])
    out_path.with_suffix(".json").write_text(json.dumps({"video": str(video), "weights": str(weights), "conf": conf,
                                                         "range_sec": [round(first / info["fps"], 2),
                                                                       round(last / info["fps"], 2)],
                                                         "frames": frames}, ensure_ascii=False), encoding="utf-8")
    log(f"[추론] 완료: {n}프레임 중 {found}프레임에서 탐지, 평균 {n / max(time.time() - t0, 1e-6):.1f} fps\n → {out_path}")
    return out_path


def _draw(img, r):
    """박스, 클래스 이름, 점수, 키포인트(번호, 연결선)를 이미지에 그리고 결과 목록을 돌려줌."""
    objs = []
    if r.boxes is None:
        return objs
    kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
    for i, b in enumerate(r.boxes):
        name = r.names[int(b.cls)]
        color = _color(name)
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        score = float(b.conf)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{name} {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        pts = []
        if kpts is not None:
            pts = [[float(x), float(y), float(c)] for x, y, c in kpts[i]]
            for a, bb in C.SKELETON:
                if pts[a][2] > 0.5 and pts[bb][2] > 0.5:
                    cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[bb][0]), int(pts[bb][1])), (0, 235, 255), 3)
            for k, (x, y, c) in enumerate(pts):
                if c > 0.5:
                    cv2.circle(img, (int(x), int(y)), 9, (40, 40, 220), -1)
                    cv2.putText(img, str(k + 1), (int(x) - 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        objs.append({"cls": name, "score": round(score, 3), "box": [x1, y1, x2, y2],
                     "kpts": [[round(x, 1), round(y, 1), round(c, 3)] for x, y, c in pts]})
    return objs
