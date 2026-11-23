"""파손 시험 영상 만들기: 실제 영상의 도로표지판을 도로 쪽으로 기울여 '차로 침범' 상황을 합성합니다.

사용법 (ai_label 환경):
    python tools/make_damage_video.py

만드는 방법 (구간마다)
  1. 기존 라벨(사람이 찍은 박스/점)이 있는 프레임 사이를 이어서 매 프레임 표지판 박스 위치를 추정
  2. 키포인트 AI로 매 프레임 점 4개(밑동, 중간, 꼭대기, 팔 끝)를 정확히 찾고, 떨림을 줄이려고 부드럽게 평균
  3. 기준 프레임 1장에서 표지판 모양(기둥 + 팔 + 판)을 SAM(물체 오려내기 AI)으로 따고, 매 프레임 점 위치에 맞춰 옮김
  4. 원래 표지판 자리를 지우고(inpaint), 밑동을 축으로 도로 쪽으로 기울인 표지판을 그려 넣음

침범 % 정의 (이 시험 영상 기준)
  기둥 꼭대기가 원래 위치에서 도로 쪽으로 '차로 폭'의 몇 % 만큼 넘어갔는지.
  차로 폭(px) = lane_ratio x 기둥 길이(px). lane_ratio 는 구간마다 화면의 차선 간격을 보고 정한 값(근사).

결과 (test_videos/<날짜>/)
  damage_test.mp4       모델에 넣을 영상 (표시 없음)
  damage_test_answer.mp4 정답(박스, 점, 침범 %) 표시한 확인용 영상
  answer.json           프레임별 정답 (파손 표지판 박스, 점 4개, 침범 %, 기운 각도)
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeler.ai_keypoint import KeypointAI   # noqa: E402
from labeler.frames import _ascii_path       # noqa: E402

from labeler import config as C              # noqa: E402

VIDEO = str(C.APP_DIR / "raw_data" / "DJI_0342.MP4")
OLD_ANN = Path(r"C:\work\YOLOAugmentor\dataset_0342\work\ann")          # 읽기만 함
# 키포인트 모델: datasets 안에서 가장 최근 학습한 키포인트 AI (없으면 실행 시 오류 안내)
_kpts = sorted(C.DATASETS_DIR.glob("*/models/*/keypoint/config.json"), key=lambda p: p.stat().st_mtime)
KPT_MODEL = _kpts[-1].parent if _kpts else C.DATASETS_DIR / "(키포인트 모델 없음)"
OUT_ROOT = Path(__file__).resolve().parent.parent / "test_videos"

# (시작 프레임, 끝 프레임, 침범 비율, 차로 폭 / 기둥 길이)
SEGMENTS = [
    (2359, 2713, 0.10, 0.73),   # 노란 어린이보호구역 표지판 (1:19 ~ 1:31)
    (3575, 3769, 0.30, 0.75),   # 가로형(오버행) 표지판 (1:59 ~ 2:06)
    (5026, 5169, 0.50, 0.55),   # 파란 안내 표지판 (2:48 ~ 2:52)
]


# ---------- 1. 기존 라벨 읽기 ----------
def load_old_labels(first, last):
    """{프레임: (박스[x1,y1,x2,y2], 점 4개 [[x,y],...])} 픽셀 좌표. 점 4개가 다 있는 것만."""
    out = {}
    for f in OLD_ANN.glob("DJI_0342_f*.json"):
        n = int(f.stem[-6:])
        if not first <= n <= last:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if len(d["objects"]) != 1:
            continue
        o = d["objects"][0]
        k = o.get("kpts") or []
        if o.get("auto") or len(k) != 4 or any(v[2] <= 0 for v in k):
            continue
        w, h = d["w"], d["h"]
        xs, ys = [p[0] * w for p in o["pts"]], [p[1] * h for p in o["pts"]]
        out[n] = (np.array([min(xs), min(ys), max(xs), max(ys)]), np.array([[x * w, y * h] for x, y, _ in k]))
    return out


def interpolate_boxes(labels, frames):
    """라벨 없는 프레임의 박스는 앞뒤 라벨 프레임 사이를 직선으로 이어서 추정."""
    keys = sorted(labels)
    boxes = {}
    for f in frames:
        if f in labels:
            boxes[f] = labels[f][0]
            continue
        prev = max([k for k in keys if k < f], default=keys[0])
        nxt = min([k for k in keys if k > f], default=keys[-1])
        t = 0 if nxt == prev else (f - prev) / (nxt - prev)
        boxes[f] = labels[prev][0] * (1 - t) + labels[nxt][0] * t
    return boxes


def smooth(points_by_frame, frames, window=9):
    """점 위치를 앞뒤 몇 프레임 평균으로 부드럽게 (떨림 줄이기)."""
    arr = np.stack([points_by_frame[f] for f in frames])
    out = {}
    for i, f in enumerate(frames):
        a, b = max(0, i - window // 2), min(len(frames), i + window // 2 + 1)
        out[f] = arr[a:b].mean(0)
    return out


# ---------- 2. 표지판 모양(마스크) ----------
SAM_MODEL = Path(__file__).resolve().parent.parent / "models" / "sam_base"
_sam = {}


def sam_mask(img, box, points):
    """SAM(물체 오려내기 AI)에 박스와 '여기가 표지판' 점들을 주고 마스크를 받습니다."""
    import torch
    from transformers import SamModel, SamProcessor
    if not _sam:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _sam.update(proc=SamProcessor.from_pretrained(str(SAM_MODEL), local_files_only=True),
                    model=SamModel.from_pretrained(str(SAM_MODEL), local_files_only=True).to(dev).eval(), dev=dev)
    proc, model = _sam["proc"], _sam["model"]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    inp = proc(rgb, input_boxes=[[list(map(float, box))]], input_points=[[[list(map(float, p)) for p in points]]],
               input_labels=[[[1] * len(points)]], return_tensors="pt").to(_sam["dev"])
    with torch.no_grad():
        out = model(**inp, multimask_output=True)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                    inp["reshaped_input_sizes"].cpu())[0][0].numpy()
    return masks[int(out.iou_scores[0, 0].argmax())].astype(np.uint8)


def sign_mask(img, kpts, box):
    """기준 프레임에서 표지판(기둥 + 팔 + 판) 모양을 따냅니다."""
    base, middle, top, tip = kpts
    L = np.linalg.norm(top - base)
    h, w = img.shape[:2]
    hints = [base * 0.7 + top * 0.3, middle, top * 0.6 + tip * 0.4, top * 0.3 + tip * 0.7]   # 기둥 2점 + 팔 2점
    mask = sam_mask(img, box, hints)
    # 표지판이 있을 수 있는 영역(기둥 띠 + 팔 아래 판 영역) 밖은 버림 → 옆을 지나는 차, 배경 제외
    allowed = np.zeros((h, w), np.uint8)
    cv2.line(allowed, tuple(base.astype(int)), tuple(top.astype(int)), 1, max(6, int(0.10 * L)))
    x1, x2 = sorted([top[0], tip[0]])
    pad = 0.06 * L
    panel = np.array([[x1 - pad, top[1] - 0.12 * L], [x2 + pad, tip[1] - 0.12 * L],
                      [x2 + pad, tip[1] + 0.55 * L], [x1 - pad, top[1] + 0.55 * L]], np.int32)
    cv2.fillPoly(allowed, [panel], 1)
    mask = mask & allowed
    cv2.line(mask, tuple(base.astype(int)), tuple(top.astype(int)), 1, 3)   # 기둥, 팔은 꼭 포함
    cv2.line(mask, tuple(top.astype(int)), tuple(tip.astype(int)), 1, 3)
    # 기둥/팔과 연결된 덩어리만 남김
    n, lab = cv2.connectedComponents(mask)
    keep = {lab[int(p[1]), int(p[0])] for p in (base, middle, top, (top + tip) / 2) if 0 <= p[0] < w and 0 <= p[1] < h}
    keep.discard(0)
    mask = np.isin(lab, list(keep)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def move_mask(mask, from_pts, to_pts):
    """기준 프레임 마스크를 현재 프레임 점 위치(밑동, 꼭대기, 팔 끝)에 맞춰 옮김."""
    M = cv2.getAffineTransform(from_pts[[0, 2, 3]].astype(np.float32), to_pts[[0, 2, 3]].astype(np.float32))
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)


# ---------- 3. 파손 합성 ----------
def tilt_angle(kpts, ratio, lane_ratio):
    """기둥 꼭대기가 차로 폭의 ratio 만큼 도로 쪽으로 넘어가는 회전 각도(도). 도로 쪽 = 팔이 뻗은 방향."""
    base, _, top, tip = kpts
    L = np.linalg.norm(top - base)
    shift = ratio * lane_ratio * L
    deg = math.degrees(math.asin(min(0.95, shift / L)))
    toward_right = tip[0] > top[0]
    return -deg if toward_right else deg      # OpenCV 회전: + 는 반시계(왼쪽으로 기울어짐)


def damage_frame(frame, mask, kpts, angle):
    h, w = frame.shape[:2]
    base = kpts[0]
    hole = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    clean = cv2.inpaint(frame, hole * 255, 7, cv2.INPAINT_TELEA)       # 원래 표지판 지우기
    M = cv2.getRotationMatrix2D((float(base[0]), float(base[1])), angle, 1.0)
    sign = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(m, (3, 3), 0)[..., None]                   # 경계 부드럽게
    out = (clean * (1 - alpha) + sign * alpha).astype(np.uint8)
    new_kpts = np.hstack([kpts, np.ones((4, 1))]) @ M.T
    ys, xs = np.nonzero(m > 0.5)
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else None
    return out, new_kpts, box


def draw_answer(img, box, kpts, text):
    out = img.copy()
    cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
    pts = [tuple(int(v) for v in p) for p in kpts]
    for a, b in ((0, 1), (1, 2), (2, 3)):
        cv2.line(out, pts[a], pts[b], (0, 255, 255), 2)
    for i, p in enumerate(pts):
        cv2.circle(out, p, 7, (0, 0, 255), -1)
    cv2.putText(out, text, (int(box[0]), int(box[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


# ---------- 실행 ----------
def read_frames(wanted):
    """원본 영상에서 필요한 프레임만 순서대로 읽기."""
    s, tmp = _ascii_path(VIDEO)
    cap = cv2.VideoCapture(s)
    fps = cap.get(cv2.CAP_PROP_FPS)
    last, n, frames = max(wanted), 0, {}
    while n <= last:
        if n in wanted:
            ok, img = cap.read()
            frames[n] = img
        else:
            ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return frames, fps


def main():
    out_dir = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    all_frames = {f for a, b, _, _ in SEGMENTS for f in range(a, b + 1)}
    print(f"원본 영상에서 {len(all_frames)}프레임 읽는 중...")
    frames, fps = read_frames(all_frames)
    kpt_ai = KeypointAI(KPT_MODEL)
    h, w = next(iter(frames.values())).shape[:2]
    four = cv2.VideoWriter_fourcc(*"mp4v")
    v_test = cv2.VideoWriter(str(out_dir / "damage_test.mp4"), four, fps, (w, h))
    v_ans = cv2.VideoWriter(str(out_dir / "damage_test_answer.mp4"), four, fps, (w, h))
    answer = {"video": "damage_test.mp4", "fps": fps, "source_video": VIDEO,
              "intrusion_definition": "기둥 꼭대기가 원래 위치에서 도로 쪽으로 차로 폭의 몇 %만큼 넘어갔는지 "
                                      "(차로 폭 = lane_ratio x 기둥 길이, 근사)",
              "segments": [], "frames": []}
    out_index = 0
    for a, b, ratio, lane_ratio in SEGMENTS:
        seg = list(range(a, b + 1))
        labels = load_old_labels(a, b)
        boxes = interpolate_boxes(labels, seg)
        raw = {}
        for f in seg:   # 사람 라벨이 있으면 그대로, 없으면 키포인트 AI
            if f in labels:
                raw[f] = labels[f][1]
            else:
                k, _ = kpt_ai.predict(frames[f], boxes[f].tolist(), min_conf=0.0)
                raw[f] = np.array([[x, y] for x, y, _ in k])
        kpts = smooth(raw, seg)
        key = min(labels, key=lambda f: abs(f - (a + b) / 2))           # 기준 프레임 (사람 라벨, 구간 가운데)
        key_mask = sign_mask(frames[key], labels[key][1], labels[key][0])
        angle = tilt_angle(kpts[key], ratio, lane_ratio)
        cv2.imwrite(str(out_dir / f"mask_check_f{key}.png"),
                    np.hstack([frames[key], cv2.cvtColor(key_mask * 255, cv2.COLOR_GRAY2BGR)]))
        answer["segments"].append({"source_frames": [a, b], "intrusion_percent": int(ratio * 100),
                                   "tilt_deg": round(abs(angle), 1), "lane_ratio": lane_ratio,
                                   "output_frames": [out_index, out_index + len(seg) - 1]})
        print(f"구간 {a}-{b}: 침범 {int(ratio * 100)}%, 기운 각도 {abs(angle):.1f}°, 사람 라벨 {len(labels)}프레임")
        for f in seg:
            mask = move_mask(key_mask, labels[key][1], kpts[f])
            img, new_k, box = damage_frame(frames[f], mask, kpts[f], angle)
            v_test.write(img)
            text = f"damage: lane intrusion {int(ratio * 100)}%"
            v_ans.write(draw_answer(img, box, new_k, text) if box else img)
            answer["frames"].append({"index": out_index, "source_frame": f, "intrusion_percent": int(ratio * 100),
                                     "box": [round(v, 1) for v in box] if box else None,
                                     "kpts": [[round(x, 1), round(y, 1)] for x, y in new_k]})
            out_index += 1
    v_test.release()
    v_ans.release()
    (out_dir / "answer.json").write_text(json.dumps(answer, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"완료: {out_dir}  ({out_index}프레임, {out_index / fps:.1f}초)")


if __name__ == "__main__":
    main()
