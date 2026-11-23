"""데이터셋 폴더와 라벨 파일을 다룹니다 (화면에서 부르는 "데이터셋" = 코드의 Project).

데이터셋 폴더 구조
    <데이터셋>/project.json        데이터셋 정보 (이름, 만든 날짜, 사용한 영상)
    <데이터셋>/images/*.jpg        추출한 프레임 이미지
    <데이터셋>/labels/*.json       이미지마다 라벨 1개 (같은 이름)
    <데이터셋>/models/<날짜>/      학습한 AI 모델
    <데이터셋>/exports/<날짜>/     내보낸 YOLO 데이터셋

라벨 파일 (labels/<이미지이름>.json) 예시
    {
      "image": "DJI_0342_f002359.jpg", "width": 2688, "height": 1512,
      "status": "human",            # todo / ai / human
      "background": false,          # true = 시설물 없음 (사람이 확인함)
      "source": "",                 # 라벨 출처 메모 (예전에 이전 툴에서 가져온 라벨에만 있음)
      "ai_score": null,             # AI 예측 신뢰도 (낮을수록 먼저 검수)
      "objects": [
        {"cls": 0, "box": [x1, y1, x2, y2], "kpts": [[x, y, v], ...4개], "score": null}
      ]
    }
    좌표는 모두 픽셀 단위입니다. v = 2 보임, 0 없음(가려짐/화면 밖).
"""
import json
import time
from pathlib import Path

from . import config as C

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def empty_label(image_name, width, height):
    return {"image": image_name, "width": width, "height": height, "status": C.STATUS_TODO,
            "background": False, "source": "", "ai_score": None, "objects": []}


class Project:
    def __init__(self, root):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.models_dir = self.root / "models"
        self.exports_dir = self.root / "exports"
        self.info = json.loads((self.root / "project.json").read_text(encoding="utf-8"))

    # ---------- 만들기 / 열기 ----------
    @staticmethod
    def create(root, name=""):
        root = Path(root)
        if (root / "project.json").exists():
            raise RuntimeError("이미 데이터셋이 있는 폴더 (목록에서 열기)")
        for sub in ("images", "labels", "models", "exports"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        info = {"name": name or root.name, "created": time.strftime("%Y-%m-%d %H:%M"), "videos": []}
        (root / "project.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return Project(root)

    def save_info(self):
        (self.root / "project.json").write_text(json.dumps(self.info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 이미지 / 라벨 ----------
    def images(self):
        """데이터셋의 이미지 파일 목록 (이름순)."""
        return sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    def label_path(self, image_path):
        return self.labels_dir / (Path(image_path).stem + ".json")

    def load_label(self, image_path):
        """라벨 읽기. 없거나 깨진 파일(작업 중 꺼짐 등)이면 None (미작업으로 취급)."""
        p = self.label_path(image_path)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                print(f"[경고] 깨진 라벨 파일, 미작업으로 취급: {p}")
        return None

    def save_label(self, image_path, label):
        """임시 파일에 다 쓴 뒤 바꿔치기 (저장 중 꺼져도 라벨 파일이 반쯤 쓰인 채로 남지 않음)."""
        import os
        p = self.label_path(image_path)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    # ---------- 진행 현황 ----------
    def summary(self):
        """전체 진행 현황을 숫자로 돌려줍니다."""
        s = {"total": 0, C.STATUS_TODO: 0, C.STATUS_AI: 0, C.STATUS_HUMAN: 0,
             "human_background": 0, "human_with_objects": 0, "from_previous": 0,
             "objects": {name: 0 for name in C.CLASS_NAMES}}
        for img in self.images():
            s["total"] += 1
            lab = self.load_label(img)
            status = lab["status"] if lab else C.STATUS_TODO
            s[status] += 1
            if status != C.STATUS_HUMAN:
                continue
            if lab.get("source"):
                s["from_previous"] += 1
            if lab["objects"]:
                s["human_with_objects"] += 1
                for o in lab["objects"]:
                    s["objects"][C.CLASS_NAMES[o["cls"]]] += 1
            else:
                s["human_background"] += 1
        return s

    def human_images(self):
        """사람이 검수 완료한 이미지와 라벨 (학습/내보내기에 사용)."""
        out = []
        for img in self.images():
            lab = self.load_label(img)
            if lab and lab["status"] == C.STATUS_HUMAN:
                out.append((img, lab))
        return out
