#!/usr/bin/env bash
# ==== Ubuntu 단독 실행 파일 빌드 (PyInstaller) ====
# 결과: dist/YOLOAugmentor/YOLOAugmentor  (폴더째로 다른 Ubuntu PC에 복사해 실행 가능)
set -e
cd "$(dirname "$(readlink -f "$0")")/.."

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt pyinstaller
.venv/bin/python -m PyInstaller --noconfirm --clean YOLOAugmentor.spec

echo
echo "==========================================================="
echo " 빌드 완료: dist/YOLOAugmentor/YOLOAugmentor"
echo " 앱 메뉴 등록: ./ubuntu/install_desktop_entry.sh"
echo "==========================================================="
