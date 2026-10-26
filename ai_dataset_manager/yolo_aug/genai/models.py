"""생성형 모델 레지스트리·경로·설치 상태·다운로드·GPU 정보.

라이선스 방침: 상용/과제 납품 가능한 모델만 등록 (OpenRAIL++-M, MIT, Apache-2.0).
  * CreativeML OpenRAIL++-M: 상용 허용. 단 Attachment A 사용 제한(불법·아동 위해·허위정보·
    법집행/범죄예측·의료조언 등) 준수 및 재배포 시 동일 제한 고지 필요.
  * 제외: FLUX.1-dev(비상업), SD3.x(커뮤니티 라이선스 매출 제한), Ultralytics YOLO-World(AGPL),
    Depth-Anything-V2 Base/Large(CC-BY-NC).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = APP_ROOT / "genai_settings.json"


@dataclass
class ModelSpec:
    key: str
    repo: str
    name: str
    purpose: str
    license: str
    size_gb: float
    phase: int = 1
    required: bool = False
    allow_patterns: list[str] = field(default_factory=list)
    fallback_patterns: list[str] = field(default_factory=list)
    url: str = ""
    filename: str = ""
    group: str = "label"


_SDXL_FP16 = ["model_index.json", "*/config.json", "*.txt", "tokenizer*/*", "scheduler/*",
              "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
              "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"]

GENERATIVE_ENABLED = False
_RF = "https://storage.googleapis.com/rfdetr/"
_UL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"

ALL_MODELS: list[ModelSpec] = [
    ModelSpec("owlv2", "google/owlv2-base-patch16-ensemble", "OWLv2",
              "Auto label: open-vocabulary box detection from class prompts / example-based detection",
              "Apache-2.0", 0.6, 1, True, ["*.json", "*.txt", "model.safetensors"]),
    ModelSpec("sam_base", "facebook/sam-vit-base", "SAM ViT-B",
              "Auto label: box -> precise polygon (Segment / OBB)", "Apache-2.0", 0.4, 1, True,
              ["*.json", "*.txt", "model.safetensors"]),
    ModelSpec("vitpose_base", "usyd-community/vitpose-base-simple", "ViTPose Base",
              "Auto label: person 17 keypoints / base for project pose model training", "Apache-2.0", 0.35, 1, True,
              ["*.json", "model.safetensors"]),
    ModelSpec("clip_b32", "openai/clip-vit-base-patch32", "CLIP ViT-B/32",
              "Auto label: zero-shot image class / features for project classifier", "MIT", 0.6, 1, True,
              ["*.json", "*.txt", "model.safetensors"]),
    ModelSpec("grounding_dino", "IDEA-Research/grounding-dino-tiny", "Grounding DINO Tiny",
              "Auto label: alternative open-vocabulary detector", "Apache-2.0", 0.7, 1, False,
              ["*.json", "*.txt", "model.safetensors"]),
    ModelSpec("rfdetr_nano", "", "RF-DETR Nano (detect)", "Project model training: detection base weights",
              "Apache-2.0", 0.12, 1, True, url=_RF + "nano_coco/checkpoint_best_regular.pth",
              filename="rf-detr-nano.pth", group="assist"),
    ModelSpec("rfdetr_small", "", "RF-DETR Small (detect)", "Project model training: detection base weights",
              "Apache-2.0", 0.12, 1, False, url=_RF + "small_coco/checkpoint_best_regular.pth",
              filename="rf-detr-small.pth", group="assist"),
    ModelSpec("rfdetr_medium", "", "RF-DETR Medium (detect)", "Project model training: detection base weights",
              "Apache-2.0", 0.13, 1, False, url=_RF + "medium_coco/checkpoint_best_regular.pth",
              filename="rf-detr-medium.pth", group="assist"),
    ModelSpec("rfdetr_seg_nano", "", "RF-DETR Seg Nano", "Project model training: segmentation base weights",
              "Apache-2.0", 0.13, 1, True, url=_RF + "rf-detr-seg-n-ft.pth", filename="rf-detr-seg-nano.pt",
              group="assist"),
    ModelSpec("rfdetr_seg_small", "", "RF-DETR Seg Small", "Project model training: segmentation base weights",
              "Apache-2.0", 0.13, 1, False, url=_RF + "rf-detr-seg-s-ft.pth", filename="rf-detr-seg-small.pt",
              group="assist"),
    ModelSpec("rfdetr_seg_medium", "", "RF-DETR Seg Medium", "Project model training: segmentation base weights",
              "Apache-2.0", 0.14, 1, False, url=_RF + "rf-detr-seg-m-ft.pth", filename="rf-detr-seg-medium.pt",
              group="assist"),
    ModelSpec("ultra_yolo26n", "", "YOLO26n (Ultralytics)", "Optional engine: detection base weights",
              "AGPL-3.0 (internal use only)", 0.01, 1, False, url=_UL + "yolo26n.pt", filename="yolo26n.pt",
              group="agpl"),
    ModelSpec("ultra_yolo26n_seg", "", "YOLO26n-seg (Ultralytics)", "Optional engine: segmentation base weights",
              "AGPL-3.0 (internal use only)", 0.01, 1, False, url=_UL + "yolo26n-seg.pt", filename="yolo26n-seg.pt",
              group="agpl"),
    ModelSpec("ultra_yolo26n_pose", "", "YOLO26n-pose (Ultralytics)", "Optional engine: pose base weights",
              "AGPL-3.0 (internal use only)", 0.01, 1, False, url=_UL + "yolo26n-pose.pt",
              filename="yolo26n-pose.pt", group="agpl"),
    ModelSpec("ultra_yolo26s", "", "YOLO26s (Ultralytics)", "Optional engine: detection base weights",
              "AGPL-3.0 (internal use only)", 0.02, 1, False, url=_UL + "yolo26s.pt", filename="yolo26s.pt",
              group="agpl"),
    ModelSpec("ultra_yolo26s_seg", "", "YOLO26s-seg (Ultralytics)", "Optional engine: segmentation base weights",
              "AGPL-3.0 (internal use only)", 0.02, 1, False, url=_UL + "yolo26s-seg.pt", filename="yolo26s-seg.pt",
              group="agpl"),
    ModelSpec("ultra_yolo26s_pose", "", "YOLO26s-pose (Ultralytics)", "Optional engine: pose base weights",
              "AGPL-3.0 (internal use only)", 0.02, 1, False, url=_UL + "yolo26s-pose.pt",
              filename="yolo26s-pose.pt", group="agpl"),
    ModelSpec("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0", "SDXL Base 1.0",
              "Generative (disabled)", "CreativeML OpenRAIL++-M", 7.0, 1, False, _SDXL_FP16, group="generative"),
    ModelSpec("sdxl_vae_fp16", "madebyollin/sdxl-vae-fp16-fix", "SDXL VAE fp16-fix",
              "Generative (disabled)", "MIT", 0.3, 1, False, ["config.json", "diffusion_pytorch_model.safetensors"],
              group="generative"),
    ModelSpec("controlnet_canny", "diffusers/controlnet-canny-sdxl-1.0", "ControlNet Canny SDXL",
              "Generative (disabled)", "OpenRAIL++-M", 2.5, 1, False,
              ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
              ["config.json", "diffusion_pytorch_model.safetensors"], group="generative"),
    ModelSpec("sdxl_inpaint", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "SDXL Inpainting 0.1",
              "Generative (disabled)", "OpenRAIL++-M", 7.0, 1, False, _SDXL_FP16, group="generative"),
]
MODELS: list[ModelSpec] = [m for m in ALL_MODELS if GENERATIVE_ENABLED or m.group != "generative"]
MODELS_BY_KEY = {m.key: m for m in ALL_MODELS}


# ----------------------------------------------------------------------------- settings
def load_settings() -> dict:
    d = {"models_dir": str(APP_ROOT / "models"), "vram_mode": "auto", "offline": True}
    try:
        if SETTINGS_PATH.exists():
            d.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    env = os.environ.get("YOLOAUG_MODELS")
    if env:
        d["models_dir"] = env
    return d


def save_settings(d: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def models_dir() -> Path:
    return Path(load_settings()["models_dir"])


def model_path(key: str, root: Path | None = None) -> Path:
    return Path(root or models_dir()) / key


def weight_file(key: str, root: Path | None = None) -> Path | None:
    m = MODELS_BY_KEY.get(key)
    if m is None or not m.filename:
        return None
    f = model_path(key, root) / m.filename
    return f if f.is_file() and f.stat().st_size > 0 else None


def is_installed(key: str, root: Path | None = None) -> bool:
    m = MODELS_BY_KEY.get(key)
    if m is not None and m.url:
        return weight_file(key, root) is not None
    p = model_path(key, root)
    if not p.is_dir():
        return False
    has_cfg = any(p.glob("*.json")) or any(p.glob("*/config.json"))
    has_w = any(p.rglob("*.safetensors")) or any(p.rglob("*.bin"))
    return has_cfg and has_w


def status_table(root: Path | None = None) -> list[dict]:
    return [dict(key=m.key, name=m.name, purpose=m.purpose, license=m.license, size_gb=m.size_gb,
                 phase=m.phase, required=m.required, installed=is_installed(m.key, root)) for m in MODELS]


def download_model(key: str, root: Path | None = None, log=print) -> Path:
    """인터넷 PC에서 HuggingFace Hub로부터 필요한 파일만 받는다."""
    m = MODELS_BY_KEY[key]
    dest = model_path(key, root)
    dest.mkdir(parents=True, exist_ok=True)
    if m.url:
        return _download_url(m, dest, log)
    from huggingface_hub import snapshot_download

    log(f"다운로드: {m.repo} → {dest} (약 {m.size_gb} GB)")
    snapshot_download(m.repo, local_dir=str(dest), allow_patterns=m.allow_patterns or None)
    if not (any(dest.rglob("*.safetensors")) or any(dest.rglob("*.bin"))) and m.fallback_patterns:
        log("fp16 가중치가 없어 기본 가중치로 재시도…")
        snapshot_download(m.repo, local_dir=str(dest), allow_patterns=m.fallback_patterns)
    (dest / "_model_info.json").write_text(json.dumps(
        dict(repo=m.repo, license=m.license, name=m.name), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"완료: {m.name}")
    return dest


def _download_url(m: ModelSpec, dest: Path, log=print) -> Path:
    import urllib.request

    target = dest / m.filename
    tmp = dest / (m.filename + ".part")
    log(f"download: {m.url} -> {target}")
    with urllib.request.urlopen(m.url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total and done - last > (total // 10 or 1):
                last = done
                log(f"  {done * 100 // total}%")
    tmp.replace(target)
    (dest / "_model_info.json").write_text(json.dumps(
        dict(url=m.url, license=m.license, name=m.name), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done: {m.name}")
    return dest


# ----------------------------------------------------------------------------- runtime / gpu
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def diffusers_available() -> bool:
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        return torch_available()
    except Exception:
        return False


def assist_status() -> str:
    parts = []
    for mod, label in (("transformers", "transformers"), ("rfdetr", "RF-DETR"), ("ultralytics", "Ultralytics")):
        try:
            __import__(mod)
            parts.append(f"{label}: OK")
        except Exception:
            parts.append(f"{label}: not installed")
    return " · ".join(parts)


def gpu_info() -> dict:
    info = dict(torch=False, cuda=False, name="", vram_gb=0.0, torch_version="", cuda_version="",
                capability="", note="")
    try:
        import torch
    except Exception:
        info["note"] = "PyTorch 미설치 — 생성형(SDXL) 기능 비활성. windows/install_genai.bat 또는 ubuntu/install_genai.sh 실행"
        return info
    info["torch"] = True
    info["torch_version"] = torch.__version__
    info["cuda_version"] = str(torch.version.cuda or "")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(cuda=True, name=p.name, vram_gb=round(p.total_memory / 1024**3, 1),
                    capability=f"sm_{p.major}{p.minor}")
        arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        if arch and f"sm_{p.major}{p.minor}" not in arch and p.major >= 12:
            info["note"] = "이 PyTorch 빌드는 RTX 50 시리즈(sm_120)를 지원하지 않습니다. CUDA 12.8+ 빌드로 재설치하세요."
    else:
        info["note"] = "CUDA GPU를 찾지 못함 (CPU 전용 PyTorch이거나 드라이버 문제). CPU로도 동작하나 매우 느립니다."
    return info


def resolve_vram_mode(mode: str, vram_gb: float) -> str:
    """auto → 12GB 미만이면 low(모델 CPU 오프로드), 이상이면 high."""
    if mode in ("low", "high", "sequential"):
        return mode
    return "low" if vram_gb < 12 else "high"
