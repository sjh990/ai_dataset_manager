"""라벨링 기준 (모든 사람이 같은 기준으로 라벨링하도록 여기 한 곳에서 정합니다).

- 클래스 3종, 키포인트는 모두 4점으로 통일합니다.
- YOLO Pose는 데이터셋 전체가 같은 키포인트 개수를 써야 하므로 3종 모두 4점입니다.
- 같은 번호의 점은 3종에서 같은 역할을 합니다.
    0: 고정된 시작점 (아래) / 1: 중간점 1 / 2: 중간점 2 / 3: 끝점 (위)
- 기준선: 화면 위에서 2/3 높이의 가로선 (GUIDE_LINE_RATIO). 이 선 아래(화면 아래 1/3)만 라벨링
- 중앙분리대, 무단횡단방지펜스: 기준선 아래 부분만 박스 1개 (lower_half)
"""
from pathlib import Path

# 이 툴이 설치된 폴더 (facility_labeler)
APP_DIR = Path(__file__).resolve().parent.parent

# 데이터셋 폴더들이 모이는 곳 (데이터셋 1개 = 폴더 1개)
DATASETS_DIR = APP_DIR / "datasets"


def move_old_datasets_folder():
    """예전 이름(projects) 폴더를 datasets 로 바꿈. 툴 시작할 때 한 번 (아직 아무 데이터셋도 열지 않은 상태)."""
    import json
    old = APP_DIR / "projects"
    if old.exists() and not DATASETS_DIR.exists():
        import time
        for _ in range(20):              # 새로고침 직후 이전 창이 아직 닫히는 중이면 잠깐 기다렸다 다시 시도
            try:
                old.rename(DATASETS_DIR)
                break
            except OSError:
                time.sleep(0.5)
    last = APP_DIR / "last_project.json"            # 마지막으로 연 데이터셋 경로도 새 이름으로
    if last.exists():
        try:
            d = json.loads(last.read_text(encoding="utf-8"))
            p = Path(d["path"])
            if not p.exists() and p.parent == old:
                d["path"] = str(DATASETS_DIR / p.name)
                last.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


# AI 모델 가중치 폴더
MODELS_DIR = APP_DIR / "models"
BOX_BASE_WEIGHTS = MODELS_DIR / "rfdetr_nano" / "rf-detr-nano.pth"   # 박스 AI 시작 가중치
KEYPOINT_BASE_MODEL = MODELS_DIR / "vitpose_base"                    # 키포인트 AI 시작 모델

NUM_KEYPOINTS = 4

# 기준선 위치: 화면 위에서부터 이 비율 높이 (2/3 = 화면 아래 1/3만 라벨링)
GUIDE_LINE_RATIO = 2 / 3

# 클래스별 이름과 키포인트 이름 / 라벨링 가이드
CLASSES = [
    {
        "key": "sign",
        "name": "도로표지판",
        "short": "표지판",
        "color": "#e53935",
        "points": ["밑동 (base)", "중간 (middle)", "꼭대기 (top)", "팔 끝 (tip)"],
        "guide": [
            "기준선: 노란 점선(화면 아래 1/3 경계) 아래에 전체가 들어온 표지판만 라벨링",
            "박스: 기둥과 표지판 전체를 감싸기",
            "키포인트 1 밑동: 기둥이 지면에 닿는 곳",
            "키포인트 2 중간: 기둥 가운데 (기둥선 위)",
            "키포인트 3 꼭대기: 기둥 꼭대기, 팔이 시작되는 곳",
            "키포인트 4 팔 끝: 표지판 팔(암)의 끝",
        ],
    },
    {
        "key": "median",
        "name": "중앙분리대",
        "short": "분리대",
        "color": "#1e88e5",
        "points": ["아래 끝", "1/3 지점", "2/3 지점", "위 끝 (기준선)"],
        "guide": [
            "박스: 기준선(노란 점선, 화면 아래 1/3 경계) 아래 부분만 1개. 기준선 위는 라벨링 안 함",
            "박스 위쪽 = 기준선 (그리면 자동으로 기준선에 맞춤)",
            "이어진 시설물 1개 = 박스 1개 (화면 아래 끝 ~ 기준선, 나눠서 잡지 않기)",
            "중간에 실제로 끊겨 있으면 (교차로 등) 이어진 덩어리마다 박스 1개",
            "키포인트: 중앙분리대 윗부분(상단선)을 따라 아래에서 위로 4개. 중앙분리대 아래쪽(지면에 닿는 곳)에 찍지 않기",
            "찍는 법: 1(아래 끝), 4(위 끝)만 클릭 → 2, 3은 일직선 3등분에 자동. 휘어 있으면 2, 3만 끌어서 수정",
            "키포인트 1 아래 끝: 박스 아래쪽에서 시설물이 시작되는 곳 (화면 밖으로 이어지면 화면 맨 아래)",
            "키포인트 4 위 끝: 시설물이 기준선과 만나는 곳 (기준선 전에 끝나면 끝나는 곳)",
            "키포인트 2, 3: 1과 4 사이 3등분 위치, 실제 시설물 선 위 (휘어 있으면 휜 위치 그대로)",
            "파손 판단: 2, 3이 1-4 직선에서 벗어난 정도 = 꺾임/휨, 차로 쪽으로 벗어남 = 차로 침범",
        ],
        "lower_half": True,      # 박스를 기준선 아래로 제한
    },
    {
        "key": "fence",
        "name": "무단횡단방지펜스",
        "short": "펜스",
        "color": "#43a047",
        "points": ["아래 끝", "1/3 지점", "2/3 지점", "위 끝 (기준선)"],
        "guide": [
            "박스: 기준선(노란 점선, 화면 아래 1/3 경계) 아래 부분만 1개. 기준선 위는 라벨링 안 함",
            "박스 위쪽 = 기준선 (그리면 자동으로 기준선에 맞춤)",
            "이어진 시설물 1개 = 박스 1개 (화면 아래 끝 ~ 기준선, 나눠서 잡지 않기)",
            "중간에 실제로 끊겨 있으면 (교차로 등) 이어진 덩어리마다 박스 1개",
            "키포인트: 펜스 윗부분(상단 난간)을 따라 아래에서 위로 4개. 펜스 아래쪽(지면에 닿는 곳)에 찍지 않기",
            "찍는 법: 1(아래 끝), 4(위 끝)만 클릭 → 2, 3은 일직선 3등분에 자동. 휘어 있으면 2, 3만 끌어서 수정",
            "키포인트 1 아래 끝: 박스 아래쪽에서 시설물이 시작되는 곳 (화면 밖으로 이어지면 화면 맨 아래)",
            "키포인트 4 위 끝: 시설물이 기준선과 만나는 곳 (기준선 전에 끝나면 끝나는 곳)",
            "키포인트 2, 3: 1과 4 사이 3등분 위치, 실제 시설물 선 위 (휘어 있으면 휜 위치 그대로)",
            "파손 판단: 2, 3이 1-4 직선에서 벗어난 정도 = 꺾임/휨, 차로 쪽으로 벗어남 = 차로 침범",
        ],
        "lower_half": True,      # 박스를 기준선 아래로 제한
    },
]

CLASS_NAMES = [c["name"] for c in CLASSES]
CLASS_KEYS = [c["key"] for c in CLASSES]


def lower_half(cls):
    """이 클래스가 '기준선 아래만' 라벨링하는 클래스인지 (박스 위쪽을 기준선에 맞춤)."""
    return CLASSES[cls].get("lower_half", False)

# 키포인트 연결선 (0-1, 1-2, 2-3)
SKELETON = [(0, 1), (1, 2), (2, 3)]

# 이미지 상태
STATUS_TODO = "todo"     # 아직 아무도 보지 않음
STATUS_AI = "ai"         # AI가 자동 라벨링함, 사람 확인 필요
STATUS_HUMAN = "human"   # 사람이 검수(확정) 완료

STATUS_TEXT = {
    STATUS_TODO: "미작업",
    STATUS_AI: "AI 예측 (확인 필요)",
    STATUS_HUMAN: "사람 검수 완료",
}
STATUS_COLOR = {STATUS_TODO: "#9e9e9e", STATUS_AI: "#fb8c00", STATUS_HUMAN: "#2e7d32"}
