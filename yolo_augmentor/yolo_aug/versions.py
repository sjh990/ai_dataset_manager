"""YOLO 버전별 호환 태스크, 학습 시 온라인 증강 기본값, 추천 프리셋."""
from __future__ import annotations

# 학습기(trainer)가 학습 중 자동으로 적용하는 증강 기본값(hyp).
_ULTRA_HYP = dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5,
                  shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0)
_V7_HYP = dict(_ULTRA_HYP, translate=0.2, scale=0.9, mixup=0.15)
_V9_HYP = dict(_ULTRA_HYP, scale=0.9, mixup=0.15)

VERSIONS: dict[str, dict] = {
    "YOLOv5": dict(
        tasks={"detect", "segment", "classify"}, names_style="list", hyp=_ULTRA_HYP, repo="ultralytics/yolov5",
        note="Detect / Segment(v7.0+) / Classify 지원. data.yaml은 names 리스트 + nc 형식으로 저장합니다."),
    "YOLOv7": dict(
        tasks={"detect", "segment", "pose"}, names_style="list", hyp=_V7_HYP, repo="WongKinYiu/yolov7",
        note="공식 main 브랜치는 Detect 중심(Seg/Pose는 별도 브랜치). 학습 hyp는 scale 0.9, mixup 0.15."),
    "YOLOv8": dict(
        tasks={"detect", "segment", "classify", "pose", "obb"}, names_style="dict", hyp=_ULTRA_HYP, repo="ultralytics",
        note="Ultralytics 패키지. Detect·Segment·Pose·OBB·Classify 지원."),
    "YOLOv9": dict(
        tasks={"detect", "segment"}, names_style="dict", hyp=_V9_HYP, repo="WongKinYiu/yolov9 / ultralytics",
        note="Detect / Segment 지원."),
    "YOLOv10": dict(
        tasks={"detect"}, names_style="dict", hyp=_ULTRA_HYP, repo="THU-MIG/yolov10 / ultralytics",
        note="Detect 전용 (NMS-free)."),
    "YOLO11": dict(
        tasks={"detect", "segment", "classify", "pose", "obb"}, names_style="dict", hyp=_ULTRA_HYP, repo="ultralytics",
        note="Ultralytics 패키지. Detect·Segment·Pose·OBB·Classify 지원 (범용 추천)."),
    "YOLO12": dict(
        tasks={"detect", "segment", "classify", "pose", "obb"}, names_style="dict", hyp=_ULTRA_HYP, repo="ultralytics",
        note="Attention 기반. Ultralytics 패키지로 Detect·Segment·Pose·OBB·Classify 지원."),
    "YOLO26": dict(
        tasks={"detect", "segment", "classify", "pose", "obb"}, names_style="dict", hyp=_ULTRA_HYP, repo="ultralytics",
        note="Ultralytics 최신 계열(엣지 배포 지향). Detect·Segment·Pose·OBB·Classify 지원."),
}
VERSION_NAMES = list(VERSIONS)
DEFAULT_VERSION = "YOLO11"


def recommend_version(task: str, evidence: list[str]) -> str:
    """데이터셋에서 찾은 버전 단서(evidence) 중 태스크와 호환되는 것을 우선 추천."""
    for v in evidence:
        if v in VERSIONS and task in VERSIONS[v]["tasks"]:
            return v
    return DEFAULT_VERSION


def is_compatible(version: str, task: str) -> bool:
    return task in VERSIONS.get(version, {}).get("tasks", set())


def trainer_hyp_preset(version: str) -> dict[str, dict]:
    """학습기 온라인 증강을 오프라인으로 재현하는 프리셋 (op_key -> params)."""
    h = VERSIONS.get(version, VERSIONS[DEFAULT_VERSION])["hyp"]
    preset: dict[str, dict] = {
        "hsv": dict(p=1.0, h_gain=h["hsv_h"], s_gain=h["hsv_s"], v_gain=h["hsv_v"]),
        "translate": dict(p=1.0, fraction=h["translate"]),
        "scale": dict(p=1.0, min_scale=round(1 - h["scale"], 3), max_scale=round(1 + h["scale"], 3)),
    }
    if h["fliplr"] > 0:
        preset["hflip"] = dict(p=h["fliplr"])
    if h["flipud"] > 0:
        preset["vflip"] = dict(p=h["flipud"])
    if h["degrees"] > 0:
        preset["rotate"] = dict(p=1.0, max_deg=h["degrees"])
    if h["mosaic"] > 0:
        preset["mosaic"] = dict(p=h["mosaic"])
    return preset


def offline_recommended_preset(version: str, task: str) -> dict[str, dict]:
    """학습기가 이미 하는 증강(HSV/scale/translate/flip/mosaic)과 겹치지 않는
    '오프라인에서 효과가 큰' 증강 위주 프리셋. 드론·실환경 영상 도메인 기준."""
    preset: dict[str, dict] = {
        "rotate": dict(p=0.5, max_deg=10.0),
        "brightness_contrast": dict(p=0.5, brightness=0.2, contrast=0.2),
        "gamma": dict(p=0.3, min_gamma=0.7, max_gamma=1.5),
        "gaussian_blur": dict(p=0.2, max_kernel=7),
        "motion_blur": dict(p=0.2, max_kernel=11),
        "gaussian_noise": dict(p=0.2, max_sigma=12.0),
        "jpeg": dict(p=0.2, min_quality=40),
        "fog": dict(p=0.1, max_intensity=0.4),
    }
    if task == "obb":  # 항공/위성 영상은 방향 불변성이 큼
        preset["rot90"] = dict(p=0.5)
        preset["vflip"] = dict(p=0.3)
    return preset


def online_overlap_ops(version: str) -> list[str]:
    """해당 버전 학습기가 기본으로 켜 두는 증강 op 키 목록 (중복 경고용)."""
    h = VERSIONS.get(version, VERSIONS[DEFAULT_VERSION])["hyp"]
    ops = ["hsv"]
    if h["translate"] > 0:
        ops.append("translate")
    if h["scale"] > 0:
        ops.append("scale")
    if h["fliplr"] > 0:
        ops.append("hflip")
    if h["flipud"] > 0:
        ops.append("vflip")
    if h["mosaic"] > 0:
        ops.append("mosaic")
    return ops
