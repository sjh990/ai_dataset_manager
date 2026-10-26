from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..genai import annstore
from ..io_utils import imread
from ..labels import Annotation, box_corners
from . import data as D
from .engines import Cancelled

TASKS = ("detect", "segment", "pose", "classify")
ENGINES = ("rfdetr", "ultralytics", "examples", "clip", "vitpose")


@dataclass
class AssistConfig:
    task: str = "detect"
    engine: str = "rfdetr"
    size: str = "nano"
    epochs: int = 50
    batch: int = 4
    imgsz: int = 640
    val_ratio: float = 0.15
    seed: int = 0
    device: str = "auto"
    models_dir: str = ""
    exemplars_per_class: int = 3


def engines_for(task: str) -> list[str]:
    if task in ("detect",):
        return ["rfdetr", "ultralytics", "examples"]
    if task == "segment":
        return ["rfdetr", "ultralytics"]
    if task == "pose":
        return ["vitpose", "ultralytics"]
    return ["clip"]


def assist_root(project) -> Path:
    return Path(project.root) / "assist"


def list_models(project) -> list[dict]:
    root = assist_root(project)
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        mf = d / "assist_model.json"
        if d.is_dir() and mf.exists():
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
                m["dir"] = str(d)
                out.append(m)
            except Exception:
                pass
    return out


def delete_model(model_dir) -> None:
    shutil.rmtree(model_dir, ignore_errors=True)


def train_model(project, cfg: AssistConfig, progress=None, log=print, cancel: threading.Event | None = None) -> dict:
    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    t0 = time.time()
    if cfg.engine not in engines_for(cfg.task):
        raise RuntimeError(f"engine '{cfg.engine}' does not support task '{cfg.task}'")
    items = D.confirmed_items(project, "pose" if cfg.task == "pose" else cfg.task)
    if not items:
        raise RuntimeError("No confirmed labels for this task yet. Label (and confirm) some images first.")
    name = f"{cfg.task}_{cfg.engine}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = assist_root(project) / name
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(name=name, task=cfg.task, engine=cfg.engine, size=cfg.size, created=time.strftime("%Y-%m-%d %H:%M"),
                config=asdict(cfg), n_images=len(items), classes=list(project.class_names),
                kpt_names=list(project.kpt_names), metrics={})
    log(f"training {cfg.task} / {cfg.engine} on {len(items)} confirmed images -> {out}")
    try:
        if cfg.engine == "examples":
            from .fewshot import pick_exemplars, save_exemplars
            ex = pick_exemplars(project, items, cfg.exemplars_per_class)
            if not ex:
                raise RuntimeError("No objects large enough to use as exemplars.")
            save_exemplars(out, ex)
            meta["weights"] = "exemplars.json"
            meta["metrics"] = dict(exemplars=len(ex))
            progress(1, 1, "done")
        elif cfg.engine == "clip":
            from .classify import train_classifier
            m = train_classifier(items, out, cfg.models_dir, cfg.device, progress, log, cancel, cfg.val_ratio, cfg.seed)
            meta["classes"] = m["classes"]
            meta["weights"] = "classifier.npz"
            meta["metrics"] = dict(val_acc=m["val_acc"], n_train=m["n_train"], n_val=m["n_val"])
        elif cfg.engine == "vitpose":
            from .pose import train_pose
            samples = D.pose_samples(project, items)
            log(f"pose objects with keypoints: {len(samples)}")
            m = train_pose(project, samples, out, cfg.epochs, max(2, cfg.batch * 4), cfg.device, cfg.models_dir,
                           progress, log, cancel, max(0.05, cfg.val_ratio), cfg.seed)
            meta["weights"] = "pose_model"
            meta["metrics"] = dict(val_pck=m["val_pck"], n_train=m["n_train"], n_val=m["n_val"])
            meta["pose_classes"] = [project.class_names[i] for i in project.pose_class_ids]
        else:
            ts = D.build_yolo_set(project, items, out / "dataset", cfg.task, cfg.val_ratio, cfg.seed, log)
            meta["train_classes"] = ts.names
            meta["metrics"] = dict(n_train=ts.n_train, n_val=ts.n_val, objects=ts.n_objects)
            if cfg.engine == "rfdetr":
                from .engines import rfdetr_available, train_rfdetr
                ok, why = rfdetr_available()
                if not ok:
                    raise RuntimeError(why)
                w = train_rfdetr(ts.root, out / "run", cfg.task, cfg.size, cfg.epochs, cfg.batch, cfg.device,
                                 cfg.models_dir, progress, log, cancel)
            else:
                from .engines import train_ultralytics, ultralytics_available
                ok, why = ultralytics_available()
                if not ok:
                    raise RuntimeError(why)
                w = train_ultralytics(ts.root, out / "run", cfg.task, cfg.size, cfg.epochs, cfg.batch, cfg.imgsz,
                                      cfg.device, cfg.models_dir, progress, log, cancel)
            meta["weights"] = str(w.relative_to(out)).replace("\\", "/")
            if cfg.task == "pose":
                meta["pose_classes"] = ts.names
    except Cancelled:
        shutil.rmtree(out, ignore_errors=True)
        raise RuntimeError("cancelled")
    except Exception:
        if not (out / "assist_model.json").exists():
            (out / "error.txt").write_text("training failed", encoding="utf-8")
        raise
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "assist_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(out / "dataset", ignore_errors=True)
    log(f"model saved: {name} ({meta['seconds']}s) metrics {meta['metrics']}")
    meta["dir"] = str(out)
    return meta


class Predictor:
    def __init__(self, project, meta: dict, models_dir: str = "", device: str = "auto"):
        self.project = project
        self.meta = meta
        self.task = meta["task"]
        self.engine = meta["engine"]
        d = Path(meta["dir"])
        if self.engine == "examples":
            from .fewshot import ExemplarDetector, load_exemplars
            self.impl = ExemplarDetector(project, load_exemplars(d), models_dir, device)
        elif self.engine == "clip":
            from .classify import ClassifierPredictor
            self.impl = ClassifierPredictor(d, models_dir, device)
        elif self.engine == "vitpose":
            from .pose import PosePredictor
            self.impl = PosePredictor(d, device)
        elif self.engine == "rfdetr":
            from .engines import RFDETRPredictor
            self.impl = RFDETRPredictor(d / meta["weights"], device)
        else:
            from .engines import UltralyticsPredictor
            self.impl = UltralyticsPredictor(d / meta["weights"], device)

    def predict(self, bgr: np.ndarray, existing: annstore.ImageAnn, threshold: float, replace_auto: bool = True,
                kpt_thr: float = 0.3) -> tuple[annstore.ImageAnn, float, int]:
        h, w = bgr.shape[:2]
        p = self.project
        ia = annstore.ImageAnn(w, h, [a.copy() for a in existing.objects], existing.image_class,
                               existing.image_class_auto, existing.image_class_score)
        if self.task == "classify":
            if ia.image_class and not ia.image_class_auto:
                return ia, 0.0, 0
            name, prob = self.impl(bgr)
            ia.image_class, ia.image_class_auto, ia.image_class_score = name, True, prob
            p.add_image_class(name)
            return ia, 1.0 - prob, 1
        if self.engine == "vitpose":
            ids = set(p.pose_class_ids)
            targets = [a for a in ia.objects if a.cls in ids and (a.kpts is None or not (a.kpts[:, 2] > 0).any()
                                                                  or (a.auto and replace_auto))]
            if not targets:
                return ia, 0.0, 0
            res = self.impl(bgr, [a.bbox() for a in targets], kpt_thr)
            for a, (k, s) in zip(targets, res):
                a.kpts = k
                a.auto = True
                a.score = s
            return ia, float(np.mean([1 - s for _, s in res])), len(res)
        if replace_auto:
            own = {p.class_id(n, add=False) for n in self.meta.get("train_classes") or p.class_names}
            keep_pose = self.task != "pose"
            ia.objects = [a for a in ia.objects if not (a.auto and a.cls in own and not
                                                        (keep_pose and a.kpts is not None and (a.kpts[:, 2] > 0).any()))]
        preds = self.impl(bgr, threshold)
        added = 0
        conf = []
        K = len(p.kpt_names)
        for name, pts, score, kpts in preds:
            cid = p.class_id(name, add=False)
            if cid < 0:
                continue
            conf.append(score)
            box = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
            match = None
            for a in ia.objects:
                if a.cls == cid and _iou(a.bbox(), box) > 0.5:
                    match = a
                    break
            kk = None
            if kpts is not None and K:
                kk = np.zeros((K, 3))
                n = min(K, len(kpts))
                kk[:n, :2] = kpts[:n, :2]
                kk[:n, 2] = np.where(kpts[:n, 2] >= kpt_thr, 2, 0)
                kk[kk[:, 2] == 0] = 0
            if match is not None:
                if kk is not None and (match.kpts is None or not (match.kpts[:, 2] > 0).any()):
                    match.kpts = kk
                    match.auto = True
                continue
            pts = np.asarray(pts, np.float64)
            if self.task == "detect" or len(pts) < 3:
                pts = box_corners(*box)
            ia.objects.append(Annotation(cid, pts, kk, True, score))
            added += 1
        unc = float(np.mean([1 - c for c in conf])) if conf else 0.5
        return ia, unc, added


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def target_images(project, scope: str) -> list[Path]:
    imgs = project.images()
    if scope == "all":
        return imgs
    out = []
    for img in imgs:
        ia = annstore.load(project.label_path(img))
        has = bool(ia and (ia.objects or ia.image_class))
        confirmed = bool(ia and ia.objects and not any(a.auto for a in ia.objects))
        if scope == "unlabeled" and not has:
            out.append(img)
        elif scope == "unconfirmed" and not confirmed:
            out.append(img)
    return out


def scores_path(project) -> Path:
    return assist_root(project) / "scores.json"


def load_scores(project) -> dict:
    p = scores_path(project)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def predict_images(project, meta: dict, images: list, threshold: float = 0.3, replace_auto: bool = True,
                   models_dir: str = "", device: str = "auto", progress=None, log=print,
                   cancel: threading.Event | None = None) -> dict:
    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    pred = Predictor(project, meta, models_dir, device)
    scores = load_scores(project)
    items = scores.setdefault("items", {})
    scores["model"] = meta["name"]
    n_img = n_obj = 0
    for i, img_path in enumerate(images):
        if cancel.is_set():
            break
        img = imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        ex = project.load_image_ann(img_path, w, h)
        ia, unc, added = pred.predict(img, ex, threshold, replace_auto)
        project.save_image_ann(img_path, ia)
        items[Path(img_path).name] = round(unc, 4)
        n_img += 1
        n_obj += added
        progress(i + 1, len(images), Path(img_path).name)
    project.save()
    assist_root(project).mkdir(parents=True, exist_ok=True)
    scores_path(project).write_text(json.dumps(scores, indent=2), encoding="utf-8")
    log(f"assist: {n_img} images, {n_obj} new AI labels (model {meta['name']})")
    return dict(images=n_img, added=n_obj, model=meta["name"])
