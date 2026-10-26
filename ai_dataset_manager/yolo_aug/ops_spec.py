"""전처리/증강 연산 정의 (GUI 폼이 이 정의로부터 자동 생성된다)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Param:
    key: str
    label: str
    kind: str  # float | int | choice | bool
    default: object
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list | None = None
    suffix: str = ""
    tip: str = ""


@dataclass
class OpSpec:
    key: str
    name: str
    group: str
    desc: str
    params: list[Param] = field(default_factory=list)


def P(p=0.5):
    return Param("p", "적용 확률", "float", p, 0.0, 1.0, 0.05, tip="증강본 1장마다 이 연산이 적용될 확률")


# ----------------------------------------------------------------------------- 전처리 (모든 split에 동일 적용)
PREPROCESS_OPS: list[OpSpec] = [
    OpSpec("tile", "타일 분할", "크기",
           "고해상도 이미지를 겹치는 타일로 잘라 작은 객체 검출 성능을 높입니다 (드론·항공 영상 추천). 타일마다 라벨을 잘라 재계산합니다.",
           [Param("tile_w", "타일 너비", "int", 640, 64, 8192, 32, "px"),
            Param("tile_h", "타일 높이", "int", 640, 64, 8192, 32, "px"),
            Param("overlap", "겹침 비율", "float", 0.2, 0.0, 0.9, 0.05, tip="이웃 타일과 겹치는 비율"),
            Param("min_visibility", "경계 객체 최소 잔존율", "float", 0.3, 0.0, 1.0, 0.05,
                  tip="타일 경계에서 잘린 객체가 원래 면적의 이 비율 미만이면 제거")]),
    OpSpec("resize", "리사이즈", "크기",
           "학습 해상도에 맞게 크기를 맞춥니다. letterbox는 비율을 유지하고 여백을 채웁니다.",
           [Param("mode", "방식", "choice", "letterbox", choices=["letterbox", "stretch", "fit_long"],
                  tip="letterbox: 비율 유지+패딩 / stretch: 강제 늘림 / fit_long: 긴 변만 맞춤(패딩 없음)"),
            Param("width", "너비 (fit_long: 긴 변)", "int", 640, 32, 8192, 32, "px"),
            Param("height", "높이", "int", 640, 32, 8192, 32, "px"),
            Param("pad_value", "패딩 색 (회색값)", "int", 114, 0, 255, 1)]),
    OpSpec("auto_contrast", "자동 대비 보정", "화질",
           "히스토그램 양끝을 잘라 대비를 자동으로 늘립니다 (어두운/뿌연 영상 보정).",
           [Param("cutoff", "양끝 절단 비율", "float", 1.0, 0.0, 10.0, 0.5, "%")]),
    OpSpec("clahe", "CLAHE (적응형 히스토그램 평활화)", "화질",
           "국소 대비를 강화합니다. 역광·그림자 영역이 많은 영상에 효과적입니다.",
           [Param("clip_limit", "Clip limit", "float", 2.0, 0.5, 10.0, 0.5),
            Param("grid", "타일 격자 크기", "int", 8, 2, 32, 1)]),
    OpSpec("denoise", "노이즈 제거", "화질",
           "센서 노이즈를 줄입니다. 강도가 높으면 작은 객체 디테일이 손실될 수 있습니다.",
           [Param("method", "방식", "choice", "median", choices=["median", "gaussian", "bilateral", "nlmeans"]),
            Param("strength", "강도 (커널 크기)", "int", 3, 1, 15, 2)]),
    OpSpec("grayscale", "흑백 변환", "화질",
           "모든 이미지를 흑백(3채널 유지)으로 변환합니다. 열화상/IR 혼합 데이터셋에 사용합니다.", []),
    OpSpec("filter_small", "작은 객체 제거", "라벨",
           "최종 해상도 기준 너무 작은 라벨을 삭제합니다 (노이즈 라벨 정리).",
           [Param("min_w", "최소 너비", "int", 4, 1, 512, 1, "px"),
            Param("min_h", "최소 높이", "int", 4, 1, 512, 1, "px")]),
    OpSpec("drop_empty", "빈 이미지(배경) 비율 조정", "라벨",
           "라벨이 없는 이미지(특히 타일 분할 후 생기는 빈 타일)를 지정 비율만 남깁니다.",
           [Param("keep_ratio", "배경 이미지 유지 비율", "float", 0.1, 0.0, 1.0, 0.05,
                  tip="0이면 모두 제거, 1이면 모두 유지. 배경 약 10%는 오탐 억제에 도움")]),
]
PREPROCESS_ORDER = ["tile", "resize", "auto_contrast", "clahe", "denoise", "grayscale", "filter_small", "drop_empty"]

# ----------------------------------------------------------------------------- 증강 (선택한 split에만, 배수만큼 생성)
AUGMENT_OPS: list[OpSpec] = [
    # 기하 변환
    OpSpec("hflip", "좌우 반전", "기하 변환", "좌우 대칭. Pose는 flip_idx로 좌/우 키포인트를 교환합니다.", [P(0.5)]),
    OpSpec("vflip", "상하 반전", "기하 변환", "상하 대칭. 위에서 내려다보는 드론·항공 영상에 적합합니다.", [P(0.3)]),
    OpSpec("rot90", "90° 단위 회전", "기하 변환", "90/180/270° 무손실 회전. 항공 영상/OBB에 적합합니다.", [P(0.3)]),
    OpSpec("rotate", "임의 각도 회전", "기하 변환", "±최대 각도 내 임의 회전. Detect는 회전 후 외접 bbox로 변환됩니다.",
           [P(0.5), Param("max_deg", "최대 각도", "float", 15.0, 0.0, 180.0, 1.0, "°")]),
    OpSpec("scale", "확대/축소", "기하 변환", "중심 기준 배율 변환.",
           [P(0.5), Param("min_scale", "최소 배율", "float", 0.7, 0.1, 1.0, 0.05),
            Param("max_scale", "최대 배율", "float", 1.3, 1.0, 4.0, 0.05)]),
    OpSpec("translate", "이동", "기하 변환", "이미지 크기 대비 비율로 상하좌우 이동.",
           [P(0.5), Param("fraction", "최대 이동 비율", "float", 0.1, 0.0, 0.5, 0.01)]),
    OpSpec("shear", "기울이기 (Shear)", "기하 변환", "평행사변형 형태로 기울입니다.",
           [P(0.3), Param("max_deg", "최대 각도", "float", 5.0, 0.0, 45.0, 0.5, "°")]),
    OpSpec("perspective", "원근 변환", "기하 변환", "카메라 시점 변화 흉내. 값이 작아도 효과가 큽니다.",
           [P(0.3), Param("strength", "강도", "float", 0.0005, 0.0, 0.002, 0.0001)]),
    OpSpec("random_crop", "랜덤 크롭", "기하 변환", "일부 영역을 잘라 원래 크기로 확대합니다 (줌인 효과).",
           [P(0.3), Param("min_area", "최소 잔존 면적", "float", 0.6, 0.1, 1.0, 0.05)]),
    OpSpec("mosaic", "모자이크 (4장 합성)", "합성", "같은 split의 이미지 4장을 한 장으로 합성합니다. 작은 객체·문맥 다양성에 효과적.",
           [P(0.3)]),
    # 색상/화질
    OpSpec("hsv", "HSV 색상 변화", "색상·화질", "YOLO 학습기와 같은 방식의 색상(H)·채도(S)·명도(V) 랜덤 게인.",
           [P(0.8), Param("h_gain", "Hue 게인", "float", 0.015, 0.0, 0.5, 0.005),
            Param("s_gain", "Saturation 게인", "float", 0.7, 0.0, 1.0, 0.05),
            Param("v_gain", "Value 게인", "float", 0.4, 0.0, 1.0, 0.05)]),
    OpSpec("brightness_contrast", "밝기·대비", "색상·화질", "밝기와 대비를 임의로 변경합니다.",
           [P(0.5), Param("brightness", "밝기 범위(±)", "float", 0.2, 0.0, 1.0, 0.05),
            Param("contrast", "대비 범위(±)", "float", 0.2, 0.0, 1.0, 0.05)]),
    OpSpec("gamma", "감마 보정", "색상·화질", "노출 과다/부족 상황을 흉내냅니다.",
           [P(0.3), Param("min_gamma", "최소 감마", "float", 0.7, 0.1, 1.0, 0.05),
            Param("max_gamma", "최대 감마", "float", 1.5, 1.0, 3.0, 0.05)]),
    OpSpec("clahe_aug", "CLAHE", "색상·화질", "국소 대비 강화를 확률적으로 적용합니다.",
           [P(0.2), Param("clip_limit", "Clip limit", "float", 3.0, 0.5, 10.0, 0.5)]),
    OpSpec("to_gray", "흑백화", "색상·화질", "확률적으로 흑백 이미지로 만듭니다 (색 의존도 감소).", [P(0.1)]),
    OpSpec("sharpen", "샤프닝", "색상·화질", "언샤프 마스크로 경계를 강조합니다.",
           [P(0.2), Param("amount", "최대 강도", "float", 1.0, 0.1, 3.0, 0.1)]),
    OpSpec("fog", "안개", "색상·화질", "불균일한 안개/연무 효과를 합성합니다.",
           [P(0.1), Param("max_intensity", "최대 농도", "float", 0.4, 0.05, 0.9, 0.05)]),
    OpSpec("gaussian_blur", "가우시안 블러", "색상·화질", "초점 흐림을 흉내냅니다.",
           [P(0.2), Param("max_kernel", "최대 커널 크기", "int", 7, 3, 31, 2)]),
    OpSpec("motion_blur", "모션 블러", "색상·화질", "기체 진동·이동에 의한 흔들림 블러.",
           [P(0.2), Param("max_kernel", "최대 커널 크기", "int", 11, 3, 51, 2)]),
    OpSpec("gaussian_noise", "가우시안 노이즈", "색상·화질", "센서 노이즈를 추가합니다.",
           [P(0.2), Param("max_sigma", "최대 표준편차", "float", 12.0, 1.0, 60.0, 1.0)]),
    OpSpec("cutout", "Cutout (랜덤 가림)", "색상·화질", "임의 사각형으로 가립니다. 60% 이상 가려진 객체 라벨은 제거됩니다.",
           [P(0.2), Param("max_holes", "최대 개수", "int", 4, 1, 32, 1),
            Param("max_size", "최대 크기(이미지 대비)", "float", 0.2, 0.02, 0.5, 0.01)]),
    OpSpec("jpeg", "JPEG 압축 열화", "색상·화질", "영상 전송/압축 아티팩트를 흉내냅니다.",
           [P(0.2), Param("min_quality", "최소 품질", "int", 40, 5, 100, 5)]),
]
AUGMENT_ORDER = [o.key for o in AUGMENT_OPS]
GEOMETRIC_KEYS = {o.key for o in AUGMENT_OPS if o.group == "기하 변환"}

PREPROCESS_BY_KEY = {o.key: o for o in PREPROCESS_OPS}
AUGMENT_BY_KEY = {o.key: o for o in AUGMENT_OPS}

COMMON_AUG_PARAMS = [
    Param("fill", "빈 영역 채움 값", "int", 114, 0, 255, 1, tip="회전/이동으로 생긴 빈 영역의 회색값 (YOLO 기본 114)"),
    Param("min_visibility", "객체 최소 잔존율", "float", 0.25, 0.0, 1.0, 0.05,
          tip="변환 후 이미지 밖으로 잘려 원래 면적의 이 비율 미만만 남은 객체는 라벨 제거"),
    Param("min_size", "객체 최소 크기", "float", 2.0, 0.0, 64.0, 1.0, "px"),
]


def defaults(spec: OpSpec) -> dict:
    return {p.key: p.default for p in spec.params}


def common_defaults() -> dict:
    return {p.key: p.default for p in COMMON_AUG_PARAMS}
