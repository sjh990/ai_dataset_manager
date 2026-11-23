"""박스 AI (RF-DETR Nano, Apache-2.0).

사람이 확정한 박스로 학습해서, 새 이미지에 박스를 자동으로 그려 줍니다.
"""
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import config as C
from .ai_keypoint import get_device


def make_dataset(items, out_dir, val_ratio=0.15, aug_copies=0, aug_settings=None, progress=None, log=print,
                 should_stop=None):
    """RF-DETR 학습용 YOLO 형식 폴더를 만듭니다 (images/, labels/, data.yaml).
    aug_copies: train 이미지 1장당 추가로 만들 증강 이미지 수 (val 은 원본만).
    증강 이미지를 만드는 동안 진행률 표시 (이미지가 크면 몇 분 걸림), 중지 버튼으로 멈춤."""
    from . import augment as A
    from .frames import read_image, save_image
    rng_aug = random.Random(1)
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    items = list(items)
    if len(items) < 2:
        raise RuntimeError("박스 AI 학습에는 이미지 2장 이상 필요 (학습, 검증 각 1장 이상)")
    random.Random(0).shuffle(items)
    n_val = max(1, round(len(items) * val_ratio))
    n_aug = (len(items) - n_val) * aug_copies
    log(f"[박스 AI] 학습 데이터 준비: 원본 {len(items)}장"
        + (f" + 증강 이미지 {n_aug}장 만들기 (이미지가 크면 몇 분 걸림)" if n_aug else ""))
    done, t0 = 0, time.time()
    for split, part in (("val", items[:n_val]), ("train", items[n_val:])):
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)
        for img, lab in part:
            if should_stop and should_stop():
                raise RuntimeError("사용자 중지 (학습 데이터 준비 중)")
            dst = out_dir / "images" / split / img.name
            try:
                os.link(img, dst)          # 복사 대신 링크 (디스크 절약)
            except OSError:
                shutil.copy2(img, dst)
            w, h = lab["width"], lab["height"]
            _write_label(out_dir / "labels" / split / (img.stem + ".txt"), lab["objects"], w, h)
            if split == "train" and aug_copies:
                src = read_image(img)
                for k in range(aug_copies):
                    a_img, a_objs = A.augment(src, lab["objects"], aug_settings or A.preset("보통"), rng_aug)
                    stem = f"{img.stem}_aug{k + 1}"
                    save_image(out_dir / "images" / split / f"{stem}.jpg", a_img)
                    _write_label(out_dir / "labels" / split / f"{stem}.txt", a_objs, w, h)
            done += 1
            if progress:
                progress(done, len(items), f"박스 AI 학습 데이터 준비 (증강) {done}/{len(items)}장")
            if n_aug and done % 50 == 0:
                log(f"[박스 AI] 학습 데이터 준비 {done}/{len(items)}장 ({time.time() - t0:.0f}초)")
    data = {"path": str(out_dir).replace("\\", "/"), "train": "images/train", "val": "images/val",
            "nc": len(C.CLASSES), "names": {i: k for i, k in enumerate(C.CLASS_KEYS)}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return (len(items) - n_val) * (1 + aug_copies), n_val


def _write_label(path, objects, w, h):
    lines = []
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
        lines.append(f"{o['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def train(items, out_dir, epochs=50, batch=2, device="auto", val_ratio=0.15, progress=None, log=print,
          should_stop=None, aug_copies=0, aug_settings=None):
    import rfdetr
    import rfdetr.training as T
    from pytorch_lightning import Callback

    # RF-DETR 가 콘솔에 특수문자를 출력하다 윈도우 한글 콘솔(cp949)에서 멈추는 문제 방지
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    out_dir = Path(out_dir)
    n_train, n_val = make_dataset(items, out_dir / "dataset", val_ratio, aug_copies, aug_settings,
                                  progress=progress, log=log, should_stop=should_stop)
    device = get_device(device)
    log(f"[박스 AI] 학습 이미지 {n_train}장 (증강 포함, 원본 1장당 {aug_copies}장 추가) / 검증 이미지 {n_val}장, device {device}")
    weights = C.BOX_BASE_WEIGHTS if C.BOX_BASE_WEIGHTS.exists() else None
    if weights is None:
        log("[박스 AI] 시작 가중치 없음: 처음부터 학습 (데이터가 훨씬 많이 필요)")
    model = rfdetr.RFDETRNano(pretrain_weights=str(weights) if weights else None)
    result = {"val_mAP50": None, "history": []}     # history: epoch 마다 [epoch, val mAP50]

    class Report(Callback):
        """학습 진행률과 점수를 화면에 알려 주는 부분."""

        def on_train_batch_end(self, trainer, *args):
            nb = max(1, int(trainer.num_training_batches or 1))
            if progress:
                progress(trainer.current_epoch * nb + args[-1] + 1, epochs * nb,
                         f"박스 AI epoch {trainer.current_epoch + 1}/{epochs}")
            if should_stop and should_stop():
                trainer.should_stop = True

        def on_train_epoch_end(self, trainer, *args):
            ap = trainer.callback_metrics.get("val/mAP_50")
            ap = round(float(ap), 3) if ap is not None else None
            result["history"].append([trainer.current_epoch + 1, ap])
            log(f"[박스 AI] epoch {trainer.current_epoch + 1}/{epochs}  val mAP50 {ap}")

    original = T.build_trainer

    def with_report(*a, **k):
        trainer = original(*a, **k)
        trainer.callbacks.append(Report())
        return trainer

    T.build_trainer = with_report
    try:
        model.train(dataset_dir=str(out_dir / "dataset"), dataset_file="yolo", epochs=epochs, batch_size=batch,
                    output_dir=str(out_dir), num_workers=0 if sys.platform == "win32" else 2, device=device,
                    tensorboard=False, progress_bar=None, checkpoint_interval=max(1, epochs))
    finally:
        T.build_trainer = original
        shutil.rmtree(out_dir / "dataset", ignore_errors=True)
    if should_stop and should_stop():
        raise RuntimeError("사용자 중지")
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                 "last_ema.pth"):
        if (out_dir / name).exists():
            scored = [h for h in result["history"] if h[1] is not None]
            if scored:   # 가장 점수가 좋았던 epoch (저장되는 가중치도 가장 좋은 것)
                best = max(scored, key=lambda h: h[1])
                result.update(val_mAP50=best[1], best_epoch=best[0])
            result.update(weights=name, n_train=n_train, n_val=n_val, batch=batch, device=device)
            return result
    raise RuntimeError("박스 AI 학습 종료, 가중치 파일 없음")


class BoxAI:
    """학습된 박스 AI로 이미지에서 시설물 박스를 찾습니다."""

    def __init__(self, weights_path, device="auto"):
        import rfdetr
        self.model = rfdetr.from_checkpoint(str(weights_path), device=get_device(device))

    def predict(self, img, threshold=0.4):
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        det = self.model.predict(rgb, threshold=threshold, include_source_image=False)
        names = det.data.get("class_name") if hasattr(det, "data") else None
        out = []
        for i in range(len(det)):
            name = str(names[i]) if names is not None else C.CLASS_KEYS[int(det.class_id[i])]
            if name not in C.CLASS_KEYS:
                continue
            out.append({"cls": C.CLASS_KEYS.index(name), "box": [float(v) for v in det.xyxy[i]],
                        "score": float(det.confidence[i])})
        return out
