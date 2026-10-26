#!/usr/bin/env bash
# ==== 생성형 AI(SDXL) 의존성 설치 — 인터넷 연결된 Ubuntu ====
# PyTorch CUDA 12.8 빌드: RTX 30/40/50 공용 (NVIDIA 드라이버 570 이상)
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
CUDA="${1:-cu128}"
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import pip._internal.cli.main_parser" 2>/dev/null; then
  echo "[복구] 가상환경이 손상되어 다시 만듭니다..."; rm -rf .venv
fi
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
echo "PyTorch + torchvision ($CUDA) ..."
.venv/bin/python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
.venv/bin/python -m pip install -r requirements-genai.txt
.venv/bin/python -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else '')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "[안내] nvidia-smi 없음 — NVIDIA 드라이버를 확인하세요."
read -r -p "Install Ultralytics YOLO (optional, AGPL-3.0, internal use only)? [y/N] " UL
if [[ "$UL" =~ ^[Yy]$ ]]; then .venv/bin/python -m pip install -r requirements-ultralytics.txt; fi
read -r -p "Download required models now (about 2.3GB)? [y/N] " DL
if [[ "$DL" =~ ^[Yy]$ ]]; then .venv/bin/python tools/download_models.py --required; fi
echo "완료. 실행: ./ubuntu/run.sh"
