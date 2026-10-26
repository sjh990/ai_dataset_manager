#!/usr/bin/env bash
# ==== YOLO Dataset Augmentor - Ubuntu 실행 스크립트 ====
# 첫 실행 시 .venv 생성 + 패키지 설치 (인터넷 필요). 이후에는 바로 실행됩니다.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."

if ! command -v python3 >/dev/null; then
  echo "[오류] python3가 없습니다: sudo apt install python3 python3-venv"; exit 1
fi
if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
  echo "[오류] python3-venv가 필요합니다: sudo apt install python3-venv"; exit 1
fi

# Qt 6.5+ xcb 플랫폼 플러그인 필수 라이브러리 / 한글 폰트 확인
if ! ldconfig -p 2>/dev/null | grep -q libxcb-cursor; then
  echo "[안내] Qt 실행에 libxcb-cursor0가 필요할 수 있습니다:"
  echo "       sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1"
fi
if command -v fc-list >/dev/null && ! fc-list | grep -qiE "cjk|nanum"; then
  echo "[안내] 한글 폰트가 없으면 글자가 깨집니다: sudo apt install fonts-noto-cjk"
fi

# venv가 깨졌으면(pip import 실패) 다시 만든다
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import pip._internal.cli.main_parser" 2>/dev/null; then
  echo "[복구] 가상환경이 손상되어 다시 만듭니다..."
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  echo "[설치] 가상환경 생성..."
  python3 -m venv .venv
fi
if ! .venv/bin/python -c "import PySide6, cv2, numpy, yaml, PIL" 2>/dev/null; then
  echo "[설치] 기본 패키지 설치 (1~3분)..."
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
fi

exec .venv/bin/python main.py "$@"
