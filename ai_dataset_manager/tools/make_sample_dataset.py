"""테스트용 합성 YOLO 데이터셋 생성기 (detect / segment / pose / obb).

사용: python tools/make_sample_dataset.py <출력폴더> --task detect --n 40
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_aug.io_utils import imwrite  # noqa: E402

NAMES = ["car", "person", "drone"]


def make(out: Path, task: str, n: int, w: int = 960, h: int = 720, seed: int = 0, layout: str = "roboflow"):
    rng = np.random.default_rng(seed)
    for i in range(n):
        split = "train" if i < n * 0.7 else ("valid" if i < n * 0.9 else "test")
        if layout == "roboflow":
            idir, ldir = out / split / "images", out / split / "labels"
        else:
            idir, ldir = out / "images" / split, out / "labels" / split
        img = np.full((h, w, 3), rng.integers(40, 120, 3), dtype=np.uint8)
        img = cv2.add(img, rng.integers(0, 30, (h, w, 3), dtype=np.uint8))
        lines = []
        for _ in range(int(rng.integers(1, 5))):
            c = int(rng.integers(0, 3))
            col = [(60, 60, 230), (60, 220, 60), (230, 160, 40)][c]
            cx, cy = rng.uniform(100, w - 100), rng.uniform(100, h - 100)
            bw, bh = rng.uniform(40, 180), rng.uniform(40, 180)
            if task == "detect":
                cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), col, -1)
                lines.append(f"{c} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            elif task == "segment":
                k = int(rng.integers(5, 12))
                ang = np.sort(rng.uniform(0, 2 * math.pi, k))
                rad = rng.uniform(0.5, 1.0, k) * min(bw, bh) / 2 + 10
                pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)
                cv2.fillPoly(img, [pts.astype(np.int32)], col)
                lines.append(f"{c} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts))
            elif task == "obb":
                a = rng.uniform(0, 180)
                pts = cv2.boxPoints(((cx, cy), (bw, bh / 2), a))
                cv2.fillPoly(img, [pts.astype(np.int32)], col)
                lines.append(f"{c} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts))
            elif task == "pose":
                c = 1
                # 5 keypoints: head, l-hand, r-hand, l-foot, r-foot (이미지 기준 좌=l)
                kp = np.array([[cx, cy - bh / 2 + 10], [cx - bw / 2 + 8, cy], [cx + bw / 2 - 8, cy],
                               [cx - bw / 4, cy + bh / 2 - 8], [cx + bw / 4, cy + bh / 2 - 8]])
                cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), (80, 80, 80), 2)
                kcols = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 0, 180), (180, 0, 0)]
                for (x, y), kc in zip(kp, kcols):
                    cv2.circle(img, (int(x), int(y)), 6, kc, -1)
                vals = " ".join(f"{x / w:.6f} {y / h:.6f} 2" for x, y in kp)
                lines.append(f"{c} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f} {vals}")
        imwrite(idir / f"img_{i:04d}.jpg", img)
        ldir.mkdir(parents=True, exist_ok=True)
        (ldir / f"img_{i:04d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    yml = "names: [car, person, drone]\nnc: 3\n"
    if task == "pose":
        yml += "kpt_shape: [5, 3]\nflip_idx: [0, 2, 1, 4, 3]\n"
    (out / "data.yaml").write_text(yml, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--task", default="detect", choices=["detect", "segment", "pose", "obb"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--layout", default="roboflow", choices=["roboflow", "ultralytics"])
    a = ap.parse_args()
    make(Path(a.out), a.task, a.n, layout=a.layout)
    print("생성 완료:", a.out)
