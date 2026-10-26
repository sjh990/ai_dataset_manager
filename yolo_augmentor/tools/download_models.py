"""생성형 모델 다운로드 (인터넷 PC).

  python tools/download_models.py --list
  python tools/download_models.py --required            # 1차 필수 (SDXL Base, VAE fp16-fix, ControlNet Canny)
  python tools/download_models.py --keys sdxl_inpaint,clip_b32
  python tools/download_models.py --required --dir D:/models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_aug.genai.models import MODELS, download_model, is_installed, models_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--required", action="store_true")
    ap.add_argument("--phase1", action="store_true", help="1차 개발 모델 전부 (필수 + Inpainting + CLIP)")
    ap.add_argument("--keys", default="")
    ap.add_argument("--dir", default="")
    a = ap.parse_args()
    root = Path(a.dir) if a.dir else models_dir()
    if a.list or not (a.required or a.phase1 or a.keys):
        for m in MODELS:
            st = "설치됨" if is_installed(m.key, root) else "-"
            print(f"{m.key:18s} {st:6s} {m.size_gb:5.1f}GB  [{m.phase}차{' 필수' if m.required else ''}] {m.name} · {m.license}")
        print(f"\n모델 폴더: {root}")
        return
    keys = [k.strip() for k in a.keys.split(",") if k.strip()]
    if a.required:
        keys += [m.key for m in MODELS if m.required]
    if a.phase1:
        keys += [m.key for m in MODELS if m.phase == 1]
    for k in dict.fromkeys(keys):
        if is_installed(k, root):
            print(f"건너뜀 (이미 설치): {k}")
            continue
        download_model(k, root)


if __name__ == "__main__":
    main()
