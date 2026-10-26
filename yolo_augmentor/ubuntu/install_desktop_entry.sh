#!/usr/bin/env bash
# 우분투 앱 목록(Activities)에 'YOLO Dataset Augmentor' 아이콘을 등록합니다.
# 빌드된 실행 파일이 있으면 그것을, 없으면 run.sh를 실행하도록 등록합니다.
set -e
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
if [ -x "$ROOT/dist/YOLOAugmentor/YOLOAugmentor" ]; then
  EXEC="$ROOT/dist/YOLOAugmentor/YOLOAugmentor"
else
  chmod +x "$ROOT/ubuntu/run.sh"
  EXEC="$ROOT/ubuntu/run.sh"
fi
mkdir -p "$HOME/.local/share/applications"
DESK="$HOME/.local/share/applications/yolo-augmentor.desktop"
cat > "$DESK" <<EOF
[Desktop Entry]
Type=Application
Name=YOLO Dataset Augmentor
Comment=YOLO 데이터셋 분할·전처리·증강
Exec="$EXEC"
Icon=$ROOT/assets/icon.png
Terminal=false
Categories=Development;Graphics;
EOF
chmod +x "$DESK"
echo "등록 완료: $DESK"
