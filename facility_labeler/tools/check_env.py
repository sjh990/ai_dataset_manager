"""설치 확인: 툴에 필요한 패키지가 모두 import 되는지 검사합니다.

사용법: .venv\\Scripts\\python.exe tools\\check_env.py   (setup.bat 이 마지막에 자동 실행)
"""
import importlib
import logging
import sys

# 윈도우에서 torch 가 띄우는 경고 숨김 (labeler/__init__.py 와 같음)
for _name in ("torch.utils.flop_counter", "torch.distributed.elastic.multiprocessing.redirects"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# (import 이름, 쓰는 곳)
PACKAGES = [
    ("PySide6.QtWidgets", "화면"),
    ("cv2", "영상, 이미지"),
    ("numpy", "이미지"),
    ("PIL", "이미지"),
    ("yaml", "data.yaml"),
    ("torch", "AI 공통"),
    ("torchvision", "AI 공통"),
    ("transformers", "키포인트 AI (ViTPose)"),
    ("rfdetr", "박스 AI (RF-DETR)"),
    ("rfdetr.training", "박스 AI 학습"),
    ("pytorch_lightning", "박스 AI 학습"),
    ("ultralytics", "5번 탭 YOLO 학습 / 추론"),
]


def main():
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    failed = []
    for name, use in PACKAGES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            print(f"  OK    {name:<20} {ver:<16} {use}")
        except Exception as e:  # noqa: BLE001 - 어떤 에러든 그대로 보여 줌
            failed.append(name)
            print(f"  실패  {name:<20} {use}: {type(e).__name__}: {e}")

    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print("GPU: 사용 불가 → CPU로 동작 (학습이 매우 느림). NVIDIA 드라이버 최신 버전 설치 확인")

    if failed:
        print(f"\n설치 실패: {', '.join(failed)}")
        sys.exit(1)
    print("\n설치 확인 완료")


if __name__ == "__main__":
    main()
