"""AI 보조: 박스 AI / 키포인트 AI 를 학습하고(둘 다 또는 하나만), 새 이미지를 자동 라벨링합니다.

학습: 사람이 검수 완료한 이미지만 사용 → 박스 AI, 키포인트 AI (고른 것만) 학습 → models/<날짜>/ 에 저장
자동 라벨링: 박스 AI 가 박스를 찾고 → 키포인트 AI 가 박스마다 점 4개를 찍음
           → 상태 'AI 예측 (확인 필요)' 로 저장, 사람이 확인 후 확정
"""
import json
import random
import shutil
import time
from pathlib import Path

from . import ai_box, ai_keypoint
from . import config as C
from .frames import read_image
from .project import empty_label


def list_models(project):
    """학습된 모델 목록 (최신 순)."""
    out = []
    if project.models_dir.exists():
        for d in sorted(project.models_dir.iterdir(), reverse=True):
            f = d / "model_info.json"
            if f.exists():
                info = json.loads(f.read_text(encoding="utf-8"))
                info["dir"] = str(d)
                out.append(info)
    return out


def has_box(model_info):
    return bool(model_info and model_info.get("box_weights"))


def has_keypoint(model_info):
    return bool(model_info and (Path(model_info["dir"]) / "keypoint" / "config.json").exists())


def model_kind(model_info):
    """모델 종류를 글자로: 박스 + 키포인트 / 박스 전용 / 키포인트 전용."""
    b, k = has_box(model_info), has_keypoint(model_info)
    return "박스 + 키포인트" if b and k else ("박스 전용" if b else "키포인트 전용")


def import_model(project, model_dir):
    """다른 데이터셋에서 학습한 모델을 가져옵니다.

    <다른 데이터셋>/models/<날짜> 폴더 (model_info.json 이 있는 곳)를 통째로 복사합니다.
    원본 폴더는 복사만 하고 수정하지 않습니다.
    """
    src = Path(model_dir)
    if not (src / "model_info.json").exists():
        raise RuntimeError("다른 데이터셋의 models 안 날짜 폴더(model_info.json 이 있는 곳) 선택 필요")
    info = json.loads((src / "model_info.json").read_text(encoding="utf-8"))
    from_project = src.parent.parent.name               # 예: gumi_DJI_0342
    name = f"{from_project}_{src.name}"
    out = project.models_dir / name
    if src.resolve().parent == project.models_dir.resolve():
        raise RuntimeError("이미 이 데이터셋에 있는 모델")
    if out.exists():
        raise RuntimeError("이미 가져온 모델")
    shutil.copytree(src, out)
    info.update(name=name, source=f"다른 데이터셋 모델 ({src})")
    (out / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    info["dir"] = str(out)
    return info


_loaded = {}


def load_ai(model_info, device="auto"):
    """모델을 불러옵니다 (한 번 불러온 모델은 다시 쓰기). 없는 쪽은 None (박스 전용이면 kpt_ai = None)."""
    key = (model_info["dir"], device)
    if key not in _loaded:
        _loaded.clear()                    # GPU 메모리 절약: 한 번에 모델 하나만
        model_dir = Path(model_info["dir"])
        box_ai = ai_box.BoxAI(model_dir / model_info["box_weights"], device) if has_box(model_info) else None
        kpt_ai = ai_keypoint.KeypointAI(model_dir / "keypoint", device) if has_keypoint(model_info) else None
        _loaded[key] = (box_ai, kpt_ai)
    return _loaded[key]


def set_scores(obj, box=None, kpt=None):
    """AI 점수 저장. box_score = 박스 점수, kpt_score = 키포인트 점수 (4개 평균).
    score = 둘 중 낮은 값 (AI 예측 목록을 '자신 없는 것부터' 정렬할 때 사용)."""
    if box is not None:
        obj["box_score"] = round(box, 3)
    if kpt is not None:
        obj["kpt_score"] = round(kpt, 3)
    scores = [v for v in (obj.get("box_score"), obj.get("kpt_score")) if v is not None]
    obj["score"] = min(scores) if scores else None


def fill_points(img, label, kpt_ai, only_empty=True):
    """박스는 있는데 점이 없는 객체에 AI가 점 4개를 찍습니다. 찍은 객체 수를 돌려줍니다."""
    n = 0
    for o in label["objects"]:
        if only_empty and any(k[2] > 0 for k in o["kpts"]):
            continue
        o["kpts"], conf = kpt_ai.predict(img, o["box"])
        set_scores(o, kpt=conf)
        n += 1
    return n


def add_boxes(img, label, box_ai, kpt_ai, threshold=0.4):
    """AI가 찾은 박스 중 기존 박스와 겹치지 않는 것만 추가하고, 키포인트 AI가 있으면 점도 찍습니다."""
    n = 0
    guide = img.shape[0] * C.GUIDE_LINE_RATIO
    for det in box_ai.predict(img, threshold):
        if C.lower_half(det["cls"]):              # 중앙분리대, 펜스: 박스 위쪽 = 기준선, 기준선 위에만 있으면 버림
            if det["box"][3] <= guide + 2:
                continue
            det["box"][1] = max(det["box"][1], guide)
        if any(o["cls"] == det["cls"] and _iou(o["box"], det["box"]) > 0.5 for o in label["objects"]):
            continue
        if kpt_ai is not None:
            kpts, kconf = kpt_ai.predict(img, det["box"])
        else:   # 박스 전용 모델: 키포인트는 비워 둠 (사람이 찍거나 키포인트 AI로)
            kpts, kconf = [[0.0, 0.0, 0] for _ in range(C.NUM_KEYPOINTS)], None
        obj = {"cls": det["cls"], "box": det["box"], "kpts": kpts}
        set_scores(obj, box=det["score"], kpt=kconf)
        label["objects"].append(obj)
        n += 1
    return n


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def training_items(projects, log=print):
    """학습에 쓸 이미지: 고른 데이터셋들에서 사람이 검수 완료한 것. 배경 이미지는 시설물 이미지 수만큼만 사용.
    같은 이름의 이미지(같은 영상의 같은 프레임)가 여러 데이터셋에 있으면 한 번만 씀."""
    if not isinstance(projects, (list, tuple)):
        projects = [projects]
    items, seen = [], set()
    for pr in projects:
        part = [it for it in pr.human_images() if it[0].name not in seen]
        seen.update(it[0].name for it in part)
        items += part
        if len(projects) > 1:
            n_obj = sum(1 for it in part if it[1]["objects"])
            log(f"  - {pr.info['name']}: 사람 검수 {len(part)}장 (시설물 {n_obj}장)")
    with_obj = [it for it in items if it[1]["objects"]]
    background = [it for it in items if not it[1]["objects"]]
    random.Random(0).shuffle(background)
    background = background[:max(len(with_obj), 10)]
    log(f"학습 데이터: 시설물 이미지 {len(with_obj)}장 + 배경 이미지 {len(background)}장")
    return with_obj + background, with_obj


TRAIN_MODES = {"both": "박스 + 키포인트 둘 다 학습", "box": "박스 AI만 학습", "keypoint": "키포인트 AI만 학습"}


def train_all(project, mode="both", box_epochs=50, kpt_epochs=50, batch=2, device="auto", progress=None, log=print,
              should_stop=None, reuse_box=None, reuse_keypoint=None, extra_projects=(), aug_copies=0,
              aug_settings=None):
    """AI 학습.

    mode: "both" = 둘 다 학습 / "box" = 박스만 / "keypoint" = 키포인트만
    학습하지 않는 쪽은 reuse_box / reuse_keypoint 로 고른 기존 모델 것을 복사해서 함께 씁니다 (없으면 비워 둠).
    결과 모델은 models/<날짜>/box (박스 AI), models/<날짜>/keypoint (키포인트 AI) 에 저장됩니다.
    """
    lines = []                       # 학습 로그 전체를 모아 두었다가 train_log.txt 로 저장
    screen_log = log

    def log(text):
        lines.append(f"[{time.strftime('%H:%M:%S')}] {text}")
        screen_log(text)

    projects = [project] + [pr for pr in extra_projects if pr.root != project.root]   # 함께 학습할 데이터셋
    items, with_obj = training_items(projects, log)
    if len(with_obj) < 5:
        raise RuntimeError("시설물이 있는 '사람 검수 완료' 이미지 5장 이상 필요")
    name = time.strftime("%Y%m%d_%H%M%S")
    out = project.models_dir / name
    out.mkdir(parents=True)
    t0 = time.time()
    log(f"설정: {TRAIN_MODES[mode]}, 박스 epoch {box_epochs}, 키포인트 epoch {kpt_epochs}, "
        f"batch {batch} (키포인트 {max(4, batch * 4)}), device {device}")
    box = {"val_mAP50": None, "weights": None}
    kpt = {"val_pck": None}
    try:
        # 박스 AI
        if mode in ("both", "box"):
            log("[박스 AI] 학습 시작")
            box = ai_box.train(items, out / "box", box_epochs, batch, device, progress=progress, log=log,
                               should_stop=should_stop, aug_copies=aug_copies, aug_settings=aug_settings)
            box["weights"] = "box/" + box["weights"]
            box_from = "새로 학습"
        elif reuse_box and reuse_box.get("box_weights"):
            log(f"[박스 AI] 새로 학습하지 않고 '{reuse_box['name']}' 모델 것을 사용")
            (out / "box").mkdir()
            src = Path(reuse_box["dir"]) / reuse_box["box_weights"]
            shutil.copy2(src, out / "box" / src.name)
            box = {"val_mAP50": reuse_box.get("box_val_mAP50"), "weights": "box/" + src.name}
            box_from, box_epochs = reuse_box["name"], 0
        else:
            box_from, box_epochs = "없음", 0
        # 키포인트 AI
        if mode in ("both", "keypoint"):
            log("[키포인트 AI] 학습 시작")
            kpt = ai_keypoint.train(with_obj, out / "keypoint", kpt_epochs, max(4, batch * 4), device,
                                    progress=progress, log=log, should_stop=should_stop, aug_copies=aug_copies,
                                    aug_settings=aug_settings)
            kpt_from = "새로 학습"
        elif reuse_keypoint and has_keypoint(reuse_keypoint):
            log(f"[키포인트 AI] 새로 학습하지 않고 '{reuse_keypoint['name']}' 모델 것을 사용")
            shutil.copytree(Path(reuse_keypoint["dir"]) / "keypoint", out / "keypoint")
            kpt = {"val_pck": reuse_keypoint.get("kpt_val_PCK")}
            kpt_from, kpt_epochs = reuse_keypoint["name"], 0
        else:
            kpt_from, kpt_epochs = "없음", 0
    except Exception as e:
        shutil.rmtree(out, ignore_errors=True)
        # 실패하거나 중지해도 무엇을 했는지 남도록 로그만 따로 저장
        log(f"학습 실패 또는 중지: {e}")
        (project.models_dir / f"{name}_실패_log.txt").write_text("\n".join(lines), encoding="utf-8")
        raise
    info = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t0) / 60, 1),
            "mode": mode, "images": len(items), "images_with_objects": len(with_obj),
            "box_epochs": box_epochs, "kpt_epochs": kpt_epochs, "batch": batch, "device": device,
            "box_val_mAP50": box["val_mAP50"], "kpt_val_PCK": kpt["val_pck"], "box_weights": box["weights"],
            "box_from": box_from, "keypoint_from": kpt_from,
            "datasets": [pr.info["name"] for pr in projects],     # 학습에 쓴 데이터셋
            "aug_copies": aug_copies, "augment": aug_settings,      # 증강 장수, 설정
            # 새로 학습한 쪽의 자세한 기록: 학습/검증 수, 가장 좋았던 epoch, epoch 별 점수
            "box_detail": {k: box[k] for k in ("n_train", "n_val", "batch", "best_epoch", "history") if k in box},
            "kpt_detail": {k: kpt[k] for k in ("n_train", "n_val", "batch", "best_epoch", "history") if k in kpt}}
    log(f"학습 완료: {name} ({info['minutes']}분)  박스 mAP50 {info['box_val_mAP50']}, 키포인트 PCK {info['kpt_val_PCK']}")
    (out / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "train_log.txt").write_text("\n".join(lines), encoding="utf-8")
    info["dir"] = str(out)
    return info


def prelabel(project, model_info, target="todo", box_threshold=0.4, device="auto", progress=None, log=print,
             should_stop=None):
    """AI 자동 라벨링. target: 'todo' = 미작업 이미지만, 'todo_ai' = 미작업 + 기존 AI 예측 다시."""
    box_ai, kpt_ai = load_ai(model_info, device)
    allowed = {C.STATUS_TODO} if target == "todo" else {C.STATUS_TODO, C.STATUS_AI}
    images = [img for img in project.images() if (project.load_label(img) or {}).get("status", C.STATUS_TODO)
              in allowed]
    if box_ai is None:
        log("키포인트 전용 모델: 박스만 있고 키포인트가 빈 객체에만 키포인트 생성")
    elif kpt_ai is None:
        log("박스 전용 모델: 박스만 생성, 키포인트는 비워 둠")
    log(f"자동 라벨링 대상: {len(images)}장")
    found = kept = 0
    for i, img_path in enumerate(images):
        if should_stop and should_stop():
            break
        if progress:
            progress(i + 1, len(images), f"자동 라벨링 {i + 1}/{len(images)}")
        img = read_image(img_path)
        if img is None:                    # 읽을 수 없는 이미지는 건너뜀
            log(f"이미지 읽기 실패, 건너뜀: {img_path.name}")
            continue
        label = project.load_label(img_path) or empty_label(img_path.name, img.shape[1], img.shape[0])
        if label["status"] not in allowed:         # 그 사이 사람이 확정한 이미지는 건드리지 않음
            continue
        if label["status"] == C.STATUS_TODO and label["objects"]:
            kept += 1                      # 미작업이지만 사람이 그린 객체가 있음 (확정 취소, 저장만 한 이미지): 그대로 둠
            continue
        if box_ai is not None:
            label["objects"] = []          # 박스 AI가 있으면 처음부터 새로 예측
            add_boxes(img, label, box_ai, kpt_ai, box_threshold)
        elif fill_points(img, label, kpt_ai) == 0:
            continue                       # 키포인트 전용인데 점 찍을 박스가 없으면 그대로 둠
        scores = [o["score"] for o in label["objects"] if o.get("score") is not None]
        label.update(background=False, status=C.STATUS_AI, ai_score=min(scores) if scores else None, source="")
        latest = project.load_label(img_path)      # 저장 직전에 한 번 더: 그 사이 확정됐으면 덮어쓰지 않음
        if latest and latest["status"] not in allowed:
            continue
        project.save_label(img_path, label)
        found += bool(label["objects"])
    log(f"자동 라벨링 완료: {len(images)}장 중 {found}장에서 시설물 발견. 다음: 라벨링 화면에서 확인 후 확정")
    if kept:
        log(f"직접 그린 객체가 있는 미작업 이미지 {kept}장은 그대로 둠 (확정 취소 또는 저장만 한 이미지)")
