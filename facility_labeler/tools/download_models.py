"""AI 시작 가중치 내려받기 (setup.bat 이 자동 실행).

깃허브에는 가중치 파일(100MB 넘음)이 없어서, 원래 받은 곳에서 같은 파일을 받습니다.
라벨링 PC의 파일과 완전히 같은지 sha256 으로 확인합니다. 이미 있는 파일은 건너뜁니다.

사용법:
    .venv\\Scripts\\python.exe tools\\download_models.py          툴에 필요한 모델 (박스 AI, 키포인트 AI, YOLO)
    .venv\\Scripts\\python.exe tools\\download_models.py --all    + SAM (tools/make_damage_video.py 시험 영상용)
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HF = "https://huggingface.co"

# (저장 위치, 받는 주소, sha256, 설명, 기본 설치 여부)
# Hugging Face 는 커밋 번호로 고정해서, 나중에 원본이 바뀌어도 같은 파일을 받음
FILES = [
    ("rfdetr_nano/rf-detr-nano.pth",
     "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
     "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d", "박스 AI 시작 가중치 (RF-DETR Nano)", True),
    ("vitpose_base/model.safetensors",
     f"{HF}/usyd-community/vitpose-base-simple/resolve/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45/model.safetensors",
     "85375373893ddd3641f3912821073e53f5435f9e966e1dca59d004454bfe4fdf", "키포인트 AI 시작 가중치 (ViTPose Base)", True),
    ("yolo11n-pose.pt",
     "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
     "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0", "5번 탭 YOLO 시작 가중치 (YOLO11n-pose)", True),
    ("sam_base/model.safetensors",
     f"{HF}/facebook/sam-vit-base/resolve/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f/model.safetensors",
     "892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579", "SAM (시험 영상 도구 전용)", False),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "facility-labeler-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r    {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="", flush=True)
    print()
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="SAM 까지 받기")
    ap.add_argument("--dest", default=str(MODELS_DIR), help="저장 폴더 (기본: models)")
    args = ap.parse_args()
    root = Path(args.dest)

    failed = []
    for rel, url, digest, desc, default in FILES:
        if not (default or args.all):
            continue
        dest = root / rel
        print(f"[모델] {desc}: {rel}")
        if dest.exists():
            if sha256(dest) == digest:
                print("    이미 있음 (같은 파일)")
            else:
                print("    이미 있음, 라벨링 PC 파일과 다름 → 그대로 둠 (다시 받으려면 파일 삭제 후 실행)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = download(url, dest)
        except Exception as e:  # noqa: BLE001 - 네트워크 오류 등 그대로 보여 줌
            print(f"    내려받기 실패: {type(e).__name__}: {e}")
            failed.append(rel)
            continue
        if sha256(tmp) != digest:
            tmp.unlink()
            print("    파일 확인 실패 (sha256 다름) → 삭제")
            failed.append(rel)
            continue
        tmp.replace(dest)
        print("    완료")

    if failed:
        print(f"\n모델 내려받기 실패: {', '.join(failed)}")
        print("인터넷 확인 후 다시 실행하거나, 라벨링 PC의 models 폴더에서 같은 위치로 복사")
        sys.exit(1)
    print("\n모델 준비 완료")


if __name__ == "__main__":
    main()
