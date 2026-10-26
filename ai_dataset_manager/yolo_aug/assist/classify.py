from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import cv2
import numpy as np

from ..genai.models import is_installed, model_path
from ..io_utils import imread
from .engines import Cancelled, resolve_device


class ClipFeatures:
    def __init__(self, models_root, device: str = "auto"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        root = Path(models_root) if models_root else None
        if not is_installed("clip_b32", root):
            raise RuntimeError("CLIP model is required (Models / GPU -> clip_b32).")
        p = str(model_path("clip_b32", root))
        self.torch = torch
        self.dev = resolve_device(device)
        self.model = CLIPModel.from_pretrained(p, local_files_only=True).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(p, local_files_only=True)

    def __call__(self, bgr_list: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        pils = [Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in bgr_list]
        with self.torch.no_grad():
            inp = self.proc(images=pils, return_tensors="pt").to(self.dev)
            f = self.model.get_image_features(**inp)
            if not isinstance(f, self.torch.Tensor):
                emb = getattr(f, "image_embeds", None)
                f = emb if emb is not None else f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()


def train_classifier(items: list, out_dir: Path, models_root, device: str, progress, log,
                     cancel: threading.Event, val_ratio: float = 0.15, seed: int = 0, epochs: int = 300) -> dict:
    import torch

    classes = sorted({it.ann.image_class for it in items})
    if len(classes) < 2:
        raise RuntimeError("At least 2 image classes with confirmed labels are needed.")
    feat = ClipFeatures(models_root, device)
    X, Y = [], []
    for i, it in enumerate(items):
        if cancel.is_set():
            raise Cancelled()
        img = imread(it.image)
        if img is None:
            continue
        f = feat([img, img[:, ::-1].copy()])
        X.extend(f)
        Y.extend([classes.index(it.ann.image_class)] * 2)
        progress(i + 1, len(items) + 1, f"features {i + 1}/{len(items)}")
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.int64)
    n = len(X) // 2
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    n_val = max(1, int(round(n * val_ratio))) if n >= 5 else 0
    vi = set(order[:n_val])
    tr = [j for i in range(n) if i not in vi for j in (2 * i, 2 * i + 1)]
    va = [2 * i for i in vi]
    xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])
    W = torch.zeros((X.shape[1], len(classes)), requires_grad=True)
    b = torch.zeros((len(classes),), requires_grad=True)
    counts = np.bincount(Y[tr], minlength=len(classes)).astype(np.float32)
    cw = torch.from_numpy(counts.sum() / np.maximum(counts, 1) / len(classes))
    opt = torch.optim.Adam([W, b], lr=0.01)
    for _ in range(int(epochs)):
        logits = (xt @ W) * 20.0 + b
        loss = torch.nn.functional.cross_entropy(logits, yt, weight=cw) + 1e-4 * (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = None
    if va:
        with torch.no_grad():
            pv = ((torch.from_numpy(X[va]) @ W) * 20.0 + b).argmax(1).numpy()
        acc = float((pv == Y[va]).mean())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "classifier.npz", W=W.detach().numpy(), b=b.detach().numpy())
    meta = dict(classes=classes, val_acc=acc, n_train=n - len(vi), n_val=len(vi))
    (out_dir / "classifier.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    progress(1, 1, "done")
    log(f"classifier: {len(classes)} classes, train {n - len(vi)} / val {len(vi)} images, "
        f"val acc {acc if acc is not None else 'n/a'}")
    return meta


class ClassifierPredictor:
    def __init__(self, model_dir: Path, models_root, device: str = "auto"):
        d = np.load(Path(model_dir) / "classifier.npz")
        self.W, self.b = d["W"], d["b"]
        self.classes = json.loads((Path(model_dir) / "classifier.json").read_text(encoding="utf-8"))["classes"]
        self.feat = ClipFeatures(models_root, device)

    def __call__(self, bgr: np.ndarray) -> tuple[str, float]:
        f = self.feat([bgr])[0]
        z = f @ self.W * 20.0 + self.b
        p = np.exp(z - z.max())
        p /= p.sum()
        i = int(p.argmax())
        return self.classes[i], float(p[i])
