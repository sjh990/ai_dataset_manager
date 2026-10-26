"""변형 효과 프리셋: 절차적(CPU) 구현 키 + 생성형(SDXL) 프롬프트."""
from __future__ import annotations

from dataclasses import dataclass, field

DOMAIN_PREFIX = {
    "aerial": "aerial drone photograph, top-down bird's-eye view, high altitude, ",
    "ground": "photograph, ",
    "none": "",
}

NEGATIVE = ("cartoon, painting, illustration, drawing, anime, 3d render, text, watermark, logo, "
            "blurry, lowres, deformed, distorted, oversaturated, duplicate, frame, border")


@dataclass
class Effect:
    key: str
    name: str
    group: str
    prompt: str
    clip_text: str               # 품질 필터: 효과 적용 여부 판정용 문장
    mode: str = "global"         # global: 전체 이미지 img2img / region: 대상 클래스 영역만 inpaint
    strength: float = 0.45       # 기본 변형 강도 (img2img denoise)
    control_scale: float = 0.65  # ControlNet(윤곽선) 유지 강도
    procedural: bool = True      # CPU 절차적 구현 제공 여부
    structure_relax: float = 1.0 # 품질 필터 구조 유지 기준 완화 배수 (눈·안개처럼 윤곽을 덮는 효과)
    kind: str = "texture"        # texture: 화소 변형 / structural: 객체 자세(기하) 변형 + 라벨 재계산
    struct_action: str = ""      # structural: tilt | fall
    negative: str = ""           # 효과 전용 네거티브 (비우면 공통 네거티브)
    targets: list = field(default_factory=list)  # 대상 클래스 이름 (비우면 공통 설정)
    remap: str = ""              # 변형 후 클래스 이름 (비우면 공통 설정)
    image_remap: str = ""        # 변형 후 '이미지 분류' 클래스 (비우면 공통 설정/유지)
    custom: bool = False
    desc: str = ""
    params: dict = field(default_factory=dict)


EFFECTS: list[Effect] = [
    # 날씨
    Effect("rain", "비", "날씨",
           "heavy rain, rainy weather, wet ground, wet asphalt, puddles, overcast gray sky, rain streaks, "
           "gloomy diffuse lighting, photorealistic", "a photo taken in heavy rain",
           desc="빗줄기·젖은 노면·흐린 조명"),
    Effect("snow", "눈", "날씨",
           "snowy winter, ground covered with snow, falling snowflakes, snow on roofs and roads, cold white "
           "overcast light, photorealistic", "a photo of a snowy scene in winter",
           structure_relax=0.5, desc="적설·강설·차가운 색감"),
    Effect("fog", "안개", "날씨",
           "dense fog, thick haze, mist, low visibility, washed out colors, soft diffuse light, photorealistic",
           "a photo in dense fog", strength=0.40, structure_relax=0.6, desc="짙은 안개·연무로 가시거리 저하"),
    # 시간대
    Effect("night", "야간", "시간대",
           "at night, dark night scene, street lights, artificial lighting, headlights, low light, high ISO "
           "sensor noise, photorealistic", "a photo taken at night", strength=0.55, structure_relax=0.7,
           desc="저조도·인공조명·센서 노이즈"),
    Effect("dusk", "황혼", "시간대",
           "at sunset, golden hour, dusk, warm orange sunlight, long shadows, dim sky, photorealistic",
           "a photo taken at sunset", desc="석양·긴 그림자·따뜻한 색온도"),
    Effect("backlight", "역광", "시간대",
           "strong backlight, sun glare, lens flare, low sun in front of camera, silhouettes, high contrast, "
           "hazy bright sky, photorealistic", "a backlit photo with strong sun glare",
           strength=0.40, desc="태양 정면·플레어·실루엣"),
    # 객체 파손 (대상 클래스 영역만)
    Effect("damage_building", "건물 붕괴·균열", "객체 파손",
           "collapsed building, earthquake damage, destroyed roof, rubble and debris, cracked concrete walls, "
           "broken structure, photorealistic", "a photo of a collapsed damaged building",
           mode="region", strength=0.75, control_scale=0.35, desc="대상 클래스(건물 등) 영역을 붕괴/균열 상태로"),
    Effect("damage_fire", "화재 피해", "객체 파손",
           "burned building after fire, charred black walls, soot, burnt roof, fire damage, smoke stains, "
           "photorealistic", "a photo of a building damaged by fire",
           mode="region", strength=0.70, control_scale=0.4, desc="그을음·탄화·화재 피해"),
    Effect("damage_road", "도로·교량 파손", "객체 파손",
           "damaged road, large cracks in asphalt, sinkhole, collapsed pavement, broken bridge deck, "
           "debris on road, photorealistic", "a photo of a severely damaged cracked road",
           mode="region", strength=0.70, control_scale=0.4, desc="균열·함몰·파손 노면"),
    Effect("flood_road", "침수", "객체 파손",
           "flooded road, muddy flood water covering the ground, submerged area, water reflections, "
           "photorealistic", "a photo of a flooded area", mode="region", strength=0.70, control_scale=0.45,
           desc="도로·시설물 주변 침수"),
    # 구조물 변형 (신호등·가로등·표지판·중앙분리대 등) — 객체를 잘라 기하 변형, 라벨 재계산
    Effect("struct_tilt", "기울어짐", "구조물 변형",
           "empty road surface, asphalt, sidewalk, background, photorealistic", "a photo of a tilted leaning pole",
           mode="region", kind="structural", struct_action="tilt",
           desc="기둥형은 밑둥 기준 15~30° 기울임, 판넬형(중앙분리대)은 앞뒤로 35~55° 기울임"),
    Effect("struct_fall", "쓰러짐", "구조물 변형",
           "empty road surface, asphalt, sidewalk, background, photorealistic", "a photo of a fallen pole on the road",
           mode="region", kind="structural", struct_action="fall",
           desc="기둥형은 55~75° 쓰러짐(무게중심 방향), 판넬형은 바닥에 눕힘 + 파편·그림자"),
]
EFFECTS_BY_KEY = {e.key: e for e in EFFECTS}


# 사용자 효과의 '적용 방식' → (mode, kind, struct_action)
SCOPES = {
    "global": ("global", "texture", ""),      # 이미지 전체 변형 (img2img + 윤곽 유지)
    "region": ("region", "texture", ""),      # 대상 객체 영역만 변형 (크롭 인페인팅)
    "tilt": ("region", "structural", "tilt"),  # 대상 객체 기울어짐
    "fall": ("region", "structural", "fall"),  # 대상 객체 쓰러짐
}
SCOPE_LABELS = {"global": "이미지 전체 변형", "region": "대상 객체 영역만 변형",
                "tilt": "대상 객체 기울어짐 (+배경 복원)", "fall": "대상 객체 쓰러짐 (+배경 복원)"}


def custom_effect(d: dict) -> Effect:
    """사용자 효과 dict → Effect. d: key,name,scope,prompt,negative,strength,control_scale,targets,remap,clip_text"""
    mode, kind, act = SCOPES.get(d.get("scope", "global"), SCOPES["global"])
    return Effect(key=d["key"], name=d.get("name", d["key"]), group="사용자 효과", prompt=d.get("prompt", ""),
                  clip_text=d.get("clip_text", "") or d.get("prompt", "")[:70], mode=mode,
                  strength=float(d.get("strength", 0.7 if mode == "region" else 0.45)),
                  control_scale=float(d.get("control_scale", 0.4 if mode == "region" else 0.65)),
                  procedural=kind == "structural", kind=kind, struct_action=act,
                  negative=d.get("negative", ""), targets=list(d.get("targets", [])), remap=d.get("remap", ""),
                  image_remap=d.get("image_remap", ""),
                  custom=True, desc=d.get("desc", ""))


def get_effect(key: str, custom: list[dict] | None = None) -> Effect:
    if key in EFFECTS_BY_KEY:
        return EFFECTS_BY_KEY[key]
    for d in custom or []:
        if d.get("key") == key:
            return custom_effect(d)
    raise KeyError(f"알 수 없는 효과: {key}")


def build_prompt(effect: Effect, domain: str = "aerial", custom: str | None = None) -> str:
    body = custom.strip() if custom and custom.strip() else effect.prompt
    return DOMAIN_PREFIX.get(domain, "") + body


# ----------------------------------------------------------------------------- 사용자 효과 라이브러리
def _lib_path():
    from .models import APP_ROOT
    return APP_ROOT / "user_effects.json"


def load_library() -> list[dict]:
    import json
    try:
        return json.loads(_lib_path().read_text(encoding="utf-8"))
    except Exception:
        return []


def save_library(effects: list[dict]) -> None:
    import json
    try:
        _lib_path().write_text(json.dumps(effects, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def new_effect_key(name: str) -> str:
    import re
    import time
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()[:20] or "fx"
    return f"user_{slug}_{int(time.time() * 1000) % 10**8}"
