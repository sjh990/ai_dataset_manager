"""코어 엔진 테스트: python -m pytest tests  (또는 python tests/test_core.py)"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_sample_dataset import make  # noqa: E402
from yolo_aug.augment import geometric  # noqa: E402
from yolo_aug.dataset import analyze_dataset, scan_dataset  # noqa: E402
from yolo_aug.labels import Annotation, box_corners, clip_polygon, format_labels, parse_labels, polygon_area  # noqa: E402
from yolo_aug.ops_spec import AUGMENT_OPS, PREPROCESS_OPS, common_defaults, defaults  # noqa: E402
from yolo_aug.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from yolo_aug.splitter import split_samples  # noqa: E402


def test_clip_polygon():
    sq = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], float)
    c = clip_polygon(sq, (0, 0, 100, 100))
    assert abs(polygon_area(c) - 100) < 1e-6


def test_hflip_detect_and_pose_swap():
    img = np.zeros((100, 200, 3), np.uint8)
    a = Annotation(0, box_corners(10, 20, 50, 60),
                   np.array([[20, 30, 2], [40, 30, 2]], float))
    out_img, out, _ = geometric(img, [a], {"hflip": {"p": 1.0}}, np.random.default_rng(0), "pose",
                                flip_idx=[1, 0])
    x1, y1, x2, y2 = out[0].bbox()
    assert (x1, y1, x2, y2) == (150, 20, 190, 60)
    # 반전 후 좌/우 교환: 원래 1번(40→160)이 0번 자리로
    assert out[0].kpts[0, 0] == 160 and out[0].kpts[1, 0] == 180


def test_rot90_roundtrip():
    img = np.zeros((100, 200, 3), np.uint8)
    a = Annotation(0, box_corners(10, 20, 50, 60))
    rng = np.random.default_rng(1)
    o_img, out, _ = geometric(img, [a], {"rot90": {"p": 1.0}}, rng, "detect")
    h, w = o_img.shape[:2]
    assert {h, w} == {100, 200}
    x1, y1, x2, y2 = out[0].bbox()
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    assert abs((x2 - x1) * (y2 - y1) - 1600) < 1e-6


def test_parse_format_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.txt"
        p.write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        anns = parse_labels(p, "detect", 640, 480)
        assert format_labels(anns, "detect", 640, 480).strip() == "1 0.5 0.5 0.2 0.4"


def test_split_ratio():
    with tempfile.TemporaryDirectory() as d:
        make(Path(d), "detect", 50)
        info = scan_dataset(d)
        sp = split_samples(info.samples, 70, 20, 10, seed=1)
        assert sum(len(v) for v in sp.values()) == 50
        assert len(sp["train"]) == 35 and len(sp["val"]) == 10 and len(sp["test"]) == 5


def _full_cfg(ds, out, task, info):
    pre = {o.key: defaults(o) for o in PREPROCESS_OPS if o.key not in ("grayscale", "denoise")}
    pre["tile"].update(tile_w=640, tile_h=640)
    aug = {o.key: defaults(o) for o in AUGMENT_OPS}
    return PipelineConfig(dataset_dir=str(ds), output_dir=str(out), task=task, yolo_version="YOLO11",
                          class_names=info.class_names, kpt_shape=list(info.kpt_shape) if info.kpt_shape else None,
                          flip_idx=info.flip_idx, preprocess=pre, augment=aug, aug_common=common_defaults(),
                          multiplier=2, workers=4)


def test_pipeline_all_tasks():
    for task in ("detect", "segment", "pose", "obb"):
        with tempfile.TemporaryDirectory() as d:
            ds, out = Path(d) / "ds", Path(d) / "out"
            make(ds, task, 20)
            info = analyze_dataset(scan_dataset(ds))
            assert info.task == task, (task, info.task)
            summ = run_pipeline(_full_cfg(ds, out, task, info), log=lambda m: None)
            assert summ["n_errors"] == 0, summ["errors"]
            assert (out / "data.yaml").exists()
            # 모든 라벨 좌표가 0~1 이내이고 태스크 포맷을 만족하는지
            for f in (out / "labels").rglob("*.txt"):
                for ln in f.read_text().splitlines():
                    v = [float(x) for x in ln.split()]
                    coords = v[1:5] + [x for i, x in enumerate(v[5:]) if i % 3 != 2] if task == "pose" else v[1:]
                    assert all(0 <= x <= 1 for x in coords), (f, ln)
                    if task == "detect":
                        assert len(v) == 5
                    elif task == "obb":
                        assert len(v) == 9
                    elif task == "pose":
                        assert len(v) == 5 + 5 * 3
                    else:
                        assert len(v) >= 7 and len(v) % 2 == 1
            n_train_imgs = len(list((out / "images" / "train").glob("*")))
            assert n_train_imgs == summ["written"]["train"] > 0


def test_classify_pipeline():
    import cv2
    with tempfile.TemporaryDirectory() as d:
        ds, out = Path(d) / "ds", Path(d) / "out"
        for c in ("damaged", "normal"):
            (ds / c).mkdir(parents=True)
            for i in range(6):
                cv2.imwrite(str(ds / c / f"{i}.jpg"), np.full((40, 60, 3), 60 if c == "normal" else 200, np.uint8))
        info = analyze_dataset(scan_dataset(ds))
        assert info.task == "classify" and info.class_names == ["damaged", "normal"]
        cfg = PipelineConfig(dataset_dir=str(ds), output_dir=str(out), task="classify", class_names=info.class_names,
                             split=dict(train=50, val=50, test=0, seed=1, stratify=True, keep_existing=False),
                             augment={"hflip": dict(p=1.0), "mosaic": dict(p=1.0)}, multiplier=2, workers=1)
        s = run_pipeline(cfg, log=lambda m: None)
        assert s["n_errors"] == 0 and not (out / "data.yaml").exists()
        assert s["written"] == {"train": 18, "val": 6}
        for split in ("train", "val"):  # 층화 분할: 두 클래스 모두 존재, 원본 클래스 유지
            for c in ("damaged", "normal"):
                files = list((out / split / c).glob("*.jpg"))
                assert files
                assert all((cv2.imread(str(f)).mean() > 128) == (c == "damaged") for f in files)
        assert analyze_dataset(scan_dataset(out)).task == "classify"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
