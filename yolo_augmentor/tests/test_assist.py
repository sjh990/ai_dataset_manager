from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))


def _project(d: Path, n: int = 12, labeled: int = 9):
    import cv2
    from make_road_scene import render
    from yolo_aug.genai import annstore
    from yolo_aug.genai.project import KPT_PRESETS, ClassDef, Project

    raw = d / "raw"
    raw.mkdir(parents=True)
    truth = {}
    for i in range(n):
        img, anns = render(seed=i)
        img = cv2.resize(img, (640, 360))
        for a in anns:
            a.pts *= 0.5
        cv2.imwrite(str(raw / f"r{i:03d}.jpg"), img)
        truth[f"r{i:03d}"] = anns
    p = Project.create(d / "proj", "detect", [ClassDef("traffic_light"), ClassDef("median_barrier")])
    p.image_classes = [ClassDef("a"), ClassDef("b")]
    pr = KPT_PRESETS["pole4"]
    p.set_keypoints("pole4", pr["names"], pr["skeleton"], ["traffic_light"])
    p.import_images(raw, log=lambda m: None)
    for k, img in enumerate(p.images()):
        if k >= labeled:
            continue
        objs = [a.copy() for a in truth[img.stem]]
        for a in objs:
            if a.cls == 0:
                x1, y1, x2, y2 = a.bbox()
                cx = (x1 + x2) / 2
                a.kpts = np.array([[cx, y2 - 1, 2], [cx, (y1 + y2) / 2, 2], [cx, y1 + 1, 2], [0, 0, 0]], float)
        p.save_image_ann(img, annstore.ImageAnn(640, 360, objs, "a" if k % 2 else "b"))
    p.save()
    return p


def test_data_selection():
    from yolo_aug.assist import data as D
    from yolo_aug.genai import annstore

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        p = _project(d)
        assert len(D.confirmed_items(p, "detect")) == 9
        img = p.images()[0]
        ia = annstore.load(p.label_path(img))
        ia.objects[0].auto = True
        annstore.save(p.label_path(img), ia)
        assert len(D.confirmed_items(p, "detect")) == 8
        ts = D.build_yolo_set(p, D.confirmed_items(p, "pose"), d / "ds", "pose", 0.2, 0, lambda m: None)
        assert ts.kpt_shape == [4, 3] and ts.names == ["traffic_light"]
        line = next(f for f in (d / "ds" / "labels" / "train").glob("*.txt") if f.read_text().strip()).read_text()
        assert len(line.split("\n")[0].split()) == 5 + 4 * 3
        s = D.summary(p)
        assert s["unlabeled"] == 3 and s["pose"]["objects"] > 0


def test_pose_geometry():
    from yolo_aug.assist import pose as P

    sz = (256, 192)
    box = (100, 50, 180, 300)
    k = np.array([[140, 60, 2], [150, 170, 2], [133, 290, 2]], float)
    c, s = P._box_cs(box, sz)
    M = P._affine(c, s, sz)
    pc = np.hstack([k[:, :2], np.ones((3, 1))]) @ M.T
    hm, wt = P._heatmaps(pc, k[:, 2], 64, 48, sz)
    pts, _ = P._decode(hm, sz)
    back = np.hstack([pts, np.ones((3, 1))]) @ P._affine(c, s, sz, inv=True).T
    assert np.abs(back - k[:, :2]).max() < 3.0 and wt.sum() == 3


def _tiny_models(root: Path):
    from test_genai import _tiny_autolabel_models, _tiny_pose_clip
    _tiny_autolabel_models(root)
    _tiny_pose_clip(root)


def test_assist_engines_tiny():
    try:
        import torch
        import transformers
    except Exception:
        print("  (torch/transformers missing -> skipped)")
        return
    from yolo_aug.assist.core import AssistConfig, list_models, predict_images, target_images, train_model
    from yolo_aug.genai import annstore
    from yolo_aug.labels import Annotation, box_corners

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        md = d / "models"
        _tiny_models(md)
        p = _project(d)
        q = lambda *a: None
        for task, eng in (("classify", "clip"), ("detect", "examples"), ("pose", "vitpose")):
            meta = train_model(p, AssistConfig(task=task, engine=eng, epochs=1, batch=2, models_dir=str(md),
                                               device="cpu"), q, q)
            assert meta["engine"] == eng
        assert len(list_models(p)) == 3
        unl = target_images(p, "unlabeled")
        assert len(unl) == 3
        p.save_image_ann(unl[0], annstore.ImageAnn(640, 360, [Annotation(0, box_corners(100, 50, 140, 200))]))
        for m in list_models(p):
            r = predict_images(p, m, unl, 0.0, True, str(md), "cpu", q, q)
            assert r["images"] == 3
        ia = annstore.load(p.label_path(unl[0]))
        assert ia.image_class_auto and ia.image_class in ("a", "b")
        tl = [a for a in ia.objects if a.cls == 0 and a.kpts is not None]
        assert tl and tl[0].auto
        conf = [a for a in annstore.load(p.label_path(p.images()[0])).objects]
        assert all(not a.auto for a in conf)


def test_rfdetr_tiny():
    from yolo_aug.assist.engines import rfdetr_available
    ok, _ = rfdetr_available()
    if not ok:
        print("  (rfdetr missing -> skipped)")
        return
    from yolo_aug.assist.core import AssistConfig, predict_images, target_images, train_model

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        p = _project(d)
        q = lambda *a: None
        for task in ("detect", "segment"):
            meta = train_model(p, AssistConfig(task=task, engine="rfdetr", epochs=1, batch=2, device="cpu"), q, q)
            assert (Path(meta["dir"]) / meta["weights"]).exists()
            r = predict_images(p, meta, target_images(p, "all")[-2:], 0.0, True, "", "cpu", q, q)
            assert r["images"] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
