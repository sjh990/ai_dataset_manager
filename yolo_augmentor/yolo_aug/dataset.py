"""데이터셋 스캔·분석: 이미지-라벨 매칭, 폴더 구조, 태스크 판별, YOLO 버전 단서 탐색."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .io_utils import image_size, is_image
from .labels import parse_line_values
from .versions import VERSIONS, recommend_version

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}


@dataclass
class Sample:
    image: Path
    label: Path | None
    orig_split: str | None = None
    cls_name: str | None = None  # classify: 클래스 폴더 이름


@dataclass
class DatasetInfo:
    root: Path
    samples: list[Sample] = field(default_factory=list)
    structure: str = ""
    yaml_path: Path | None = None
    yaml_data: dict = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    n_labeled: int = 0
    n_empty: int = 0
    n_missing: int = 0
    n_instances: int = 0
    col_hist: Counter = field(default_factory=Counter)
    task: str = "detect"
    task_confidence: str = ""
    task_reasons: list[str] = field(default_factory=list)
    kpt_shape: tuple[int, int] | None = None
    flip_idx: list[int] | None = None
    version_evidence: list[str] = field(default_factory=list)
    version_evidence_text: list[str] = field(default_factory=list)
    recommended_version: str = "YOLO11"
    yaml_style: str = ""
    size_stats: dict = field(default_factory=dict)
    split_counts: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.samples)

    @property
    def has_splits(self) -> bool:
        return any(s.orig_split for s in self.samples)

    def dominant_class(self, sample: Sample):
        return sample.cls_name if sample.cls_name is not None else _dominant_class(sample.label)


# ----------------------------------------------------------------------------- scanning
def _find_label(img: Path, root: Path) -> Path | None:
    parts = list(img.parts)
    # ultralytics 규칙: 경로의 마지막 'images' → 'labels'
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
            break
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    flat = root / "labels" / (img.stem + ".txt")
    if flat.exists():
        return flat
    return None


def _split_of(img: Path, root: Path) -> str | None:
    try:
        rel = img.relative_to(root).parts[:-1]
    except ValueError:
        rel = img.parts[:-1]
    for p in rel:
        s = SPLIT_ALIASES.get(p.lower())
        if s:
            return s
    return None


def _detect_structure(root: Path) -> str:
    if (root / "images").is_dir() and (root / "labels").is_dir():
        subs = [p.name.lower() for p in (root / "images").iterdir() if p.is_dir()]
        if any(s in SPLIT_ALIASES for s in subs):
            return "Ultralytics 형식 (images/<split>, labels/<split>)"
        return "평면 형식 (images/, labels/)"
    for d in root.iterdir() if root.is_dir() else []:
        if d.is_dir() and d.name.lower() in SPLIT_ALIASES and (d / "images").is_dir():
            return "Roboflow 형식 (<split>/images, <split>/labels)"
    return "혼합/동일 폴더 형식 (이미지와 .txt가 같은 폴더)"


def _load_yaml(root: Path) -> tuple[Path | None, dict]:
    cands = [root / n for n in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")]
    cands += sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    for p in cands:
        if p.exists():
            try:
                d = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore")) or {}
            except Exception:
                continue
            if isinstance(d, dict) and ("names" in d or "nc" in d or "kpt_shape" in d):
                return p, d
    return None, {}


def _names_from(yaml_data: dict, root: Path) -> list[str]:
    names = yaml_data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    for fn in ("classes.txt", "obj.names", "labels/classes.txt"):
        p = root / fn
        if p.exists():
            return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    return []


VERSION_RE = re.compile(r"\byolo\s*-?\s*v?\s*(5|7|8|9|10|11|12|26)(?!\d)", re.I)


def _norm_version(num: str) -> str:
    n = int(num)
    return f"YOLOv{n}" if n <= 10 else f"YOLO{n}"


def _scan_version_evidence(root: Path) -> tuple[list[str], list[str]]:
    """README(Roboflow 등), args.yaml(학습 기록), *.pt 가중치 파일명 등에서 버전 언급 탐색."""
    found: Counter = Counter()
    texts: list[str] = []
    files = []
    for pat in ("*.txt", "*.md", "*.yaml", "*.yml", "*.pt", "*.onnx", "*.engine"):
        files += list(root.glob(pat))
        files += list(root.glob(f"*/{pat}"))
    for f in files[:200]:
        if f.parent.name.lower() == "labels":
            continue
        cand_text = f.name
        if f.suffix.lower() in (".txt", ".md", ".yaml", ".yml"):
            try:
                if f.stat().st_size < 1_000_000:
                    cand_text += "\n" + f.read_text(encoding="utf-8", errors="ignore")[:20000]
            except OSError:
                pass
        for m in VERSION_RE.finditer(cand_text):
            v = _norm_version(m.group(1))
            found[v] += 1
            if len(texts) < 6:
                line = cand_text[max(0, m.start() - 40): m.end() + 40].replace("\n", " ").strip()
                texts.append(f"{f.name}: …{line}…")
    return [v for v, _ in found.most_common()], texts


def _dominant_class(label: Path | None) -> int:
    if label is None or not label.exists():
        return -1
    c: Counter = Counter()
    for ln in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        vals = parse_line_values(ln)
        if vals:
            c[int(vals[0])] += 1
    return c.most_common(1)[0][0] if c else -1


def scan_dataset(root, exclude: Path | None = None) -> DatasetInfo:
    root = Path(root).resolve()
    info = DatasetInfo(root=root)
    if not root.is_dir():
        info.warnings.append("폴더가 존재하지 않습니다.")
        return info
    excl = Path(exclude).resolve() if exclude else None
    imgs = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or not is_image(p):
            continue
        if excl and (p == excl or excl in p.parents):
            continue
        if any(part.startswith("_rejected") for part in p.relative_to(root).parts[:-1]):
            continue  # 생성형 탭의 품질 필터 탈락본은 제외
        imgs.append(p)
    for img in imgs:
        info.samples.append(Sample(img, _find_label(img, root), _split_of(img, root)))
    info.yaml_path, info.yaml_data = _load_yaml(root)
    if _apply_classify_layout(info, root):
        info.structure = "분류 형식 ([<split>/]<클래스>/이미지)"
    else:
        info.structure = _detect_structure(root)
    return info


_NON_CLASS_DIRS = {"images", "labels", "_rejected"}


def _apply_classify_layout(info: DatasetInfo, root: Path) -> bool:
    """라벨 txt가 하나도 없고 이미지가 클래스 폴더(<split>/<클래스>/ 또는 <클래스>/)에 있으면 classify."""
    if not info.samples or any(s.label for s in info.samples):
        return False
    if info.yaml_data.get("kpt_shape"):
        return False
    names = []
    for s in info.samples:
        rel = s.image.relative_to(root).parts[:-1]
        rel = [p for p in rel if p.lower() not in SPLIT_ALIASES]
        if len(rel) != 1 or rel[0].lower() in _NON_CLASS_DIRS:
            return False
        names.append(rel[0])
    if len(set(names)) < 1:
        return False
    for s, n in zip(info.samples, names):
        s.cls_name = n
    return True


# ----------------------------------------------------------------------------- analysis
def _is_rectangle(pts: np.ndarray, tol: float = 0.03) -> bool:
    """정규화 좌표에서는 가로/세로 스케일이 달라 직사각형이 평행사변형이 되므로
    '대각선이 서로를 이등분하는가(평행사변형)'로 판정한다."""
    size = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-9)
    return bool(np.linalg.norm((pts[0] + pts[2]) - (pts[1] + pts[3])) / size < tol)


def analyze_dataset(info: DatasetInfo, max_size_samples: int = 150) -> DatasetInfo:
    y = info.yaml_data
    if info.samples and all(s.cls_name is not None for s in info.samples):
        return _analyze_classify(info, max_size_samples)
    info.class_names = _names_from(y, info.root)
    if y:
        names = y.get("names")
        info.yaml_style = ("v8+ 스타일 (names: dict)" if isinstance(names, dict)
                           else "v5 스타일 (names: list + nc)" if isinstance(names, list) else "")
    ks = y.get("kpt_shape")
    if isinstance(ks, (list, tuple)) and len(ks) == 2:
        info.kpt_shape = (int(ks[0]), int(ks[1]))
    fi = y.get("flip_idx")
    if isinstance(fi, (list, tuple)):
        info.flip_idx = [int(v) for v in fi]

    lens: Counter = Counter()
    nine_rect, nine_total = 0, 0
    pose_like, pose_total = 0, 0
    for s in info.samples:
        if s.orig_split:
            info.split_counts[s.orig_split] += 1
        if s.label is None:
            info.n_missing += 1
            continue
        lines = [parse_line_values(ln) for ln in s.label.read_text(encoding="utf-8", errors="ignore").splitlines()]
        lines = [v for v in lines if v]
        if not lines:
            info.n_empty += 1
            continue
        info.n_labeled += 1
        for v in lines:
            lens[len(v)] += 1
            info.class_counts[int(v[0])] += 1
            info.n_instances += 1
            if len(v) == 9 and nine_total < 3000:
                nine_total += 1
                if _is_rectangle(np.asarray(v[1:]).reshape(4, 2)):
                    nine_rect += 1
            if len(v) > 9 and pose_total < 3000:
                pose_total += 1
                cx, cy, w, h = v[1:5]
                rest = np.asarray(v[5:])
                d = 3 if len(rest) % 3 == 0 else 2
                kp = rest[: len(rest) // d * d].reshape(-1, d)[:, :2]
                kp = kp[(kp[:, 0] != 0) | (kp[:, 1] != 0)]
                if len(kp) == 0 or np.mean((np.abs(kp[:, 0] - cx) <= w * 0.6) & (np.abs(kp[:, 1] - cy) <= h * 0.6)) > 0.9:
                    pose_like += 1
    info.col_hist = lens
    _decide_task(info, lens, nine_rect, nine_total, pose_like, pose_total)

    if not info.class_names:
        n = (max(info.class_counts) + 1) if info.class_counts else 1
        info.class_names = [f"class_{i}" for i in range(n)]
        info.notes.append("클래스 이름을 찾지 못해 class_N 으로 임시 지정했습니다 (편집 가능).")
    elif info.class_counts and max(info.class_counts) >= len(info.class_names):
        info.warnings.append(
            f"라벨에 클래스 id {max(info.class_counts)}가 있으나 names는 {len(info.class_names)}개입니다.")
        info.class_names += [f"class_{i}" for i in range(len(info.class_names), max(info.class_counts) + 1)]

    _finish(info, max_size_samples)
    if info.n_missing:
        info.notes.append(f"라벨 파일이 없는 이미지 {info.n_missing}장은 배경(빈 라벨)으로 취급합니다.")
    return info


def _analyze_classify(info: DatasetInfo, max_size_samples: int) -> DatasetInfo:
    info.task, info.task_confidence = "classify", "높음"
    info.task_reasons.append("라벨 txt 없이 이미지가 클래스 이름 폴더에 나뉘어 있음 → Classification")
    info.class_names = sorted({s.cls_name for s in info.samples})
    idx = {n: i for i, n in enumerate(info.class_names)}
    for s in info.samples:
        if s.orig_split:
            info.split_counts[s.orig_split] += 1
        info.class_counts[idx[s.cls_name]] += 1
    info.n_labeled = info.n_instances = len(info.samples)
    if len(info.class_names) < 2:
        info.warnings.append("분류 클래스가 1개뿐입니다. 학습에는 2개 이상의 클래스 폴더가 필요합니다.")
    _finish(info, max_size_samples)
    return info


def _finish(info: DatasetInfo, max_size_samples: int) -> None:
    # 버전 단서
    info.version_evidence, info.version_evidence_text = _scan_version_evidence(info.root)
    info.recommended_version = recommend_version(info.task, info.version_evidence)

    # 이미지 해상도 통계
    step = max(1, len(info.samples) // max_size_samples)
    sizes = [image_size(s.image) for s in info.samples[::step][:max_size_samples]]
    sizes = [s for s in sizes if s]
    if sizes:
        ws = np.array([s[0] for s in sizes])
        hs = np.array([s[1] for s in sizes])
        info.size_stats = dict(min_w=int(ws.min()), max_w=int(ws.max()), med_w=int(np.median(ws)),
                               min_h=int(hs.min()), max_h=int(hs.max()), med_h=int(np.median(hs)),
                               sampled=len(sizes))
        if max(ws.max(), hs.max()) >= 1920:
            info.notes.append("고해상도 이미지가 있습니다. 작은 객체(드론 영상 등)는 '타일 분할' 전처리가 유리합니다.")
    if info.n_images == 0:
        info.warnings.append("이미지를 찾지 못했습니다.")


def _decide_task(info, lens, nine_rect, nine_total, pose_like, pose_total):
    r = info.task_reasons
    total = sum(lens.values())
    if info.kpt_shape:
        info.task, info.task_confidence = "pose", "높음"
        r.append(f"data.yaml에 kpt_shape={list(info.kpt_shape)} 존재")
        return
    if total == 0:
        info.task, info.task_confidence = "detect", "낮음"
        r.append("라벨 데이터가 없어 Detect로 가정")
        return
    keys = set(lens)
    if keys == {5}:
        info.task, info.task_confidence = "detect", "높음"
        r.append("모든 라벨 라인이 5개 값 (class cx cy w h)")
        return
    if keys <= {5, 9} and 9 in keys:
        ratio = nine_rect / max(nine_total, 1)
        if ratio >= 0.9:
            info.task, info.task_confidence = "obb", "높음" if ratio > 0.98 else "중간"
            r.append(f"9개 값(4꼭짓점) 라인의 {ratio:.0%}가 회전 사각형 → OBB")
        else:
            info.task, info.task_confidence = "segment", "중간"
            r.append(f"9개 값 라인 중 사각형(평행사변형) 비율 {ratio:.0%} → 4점 폴리곤(Segment)으로 판단")
        return
    long_lens = {k for k in keys if k > 9}
    if len(long_lens) == 1 and keys <= long_lens | {5}:
        L = next(iter(long_lens))
        like = pose_like / max(pose_total, 1)
        if L % 2 == 0 or like > 0.9:
            rest = L - 5
            d = 3 if rest % 3 == 0 else 2
            info.kpt_shape = (rest // d, d)
            info.task, info.task_confidence = "pose", "중간"
            r.append(f"모든 라인 길이가 {L}로 고정, 키포인트가 bbox 내부({like:.0%}) → Pose, kpt_shape={list(info.kpt_shape)} 추정")
            return
    info.task = "segment"
    info.task_confidence = "높음" if all(k % 2 == 1 for k in keys) else "중간"
    r.append(f"라인 길이가 가변({min(keys)}~{max(keys)}개 값) → 폴리곤 Segment")


def describe(info: DatasetInfo) -> str:
    from .labels import TASK_LABELS

    L = [f"<b>경로</b>: {_esc(str(info.root))}",
         f"<b>구조</b>: {_esc(info.structure)}",
         f"<b>이미지</b>: {info.n_images}장 (라벨 있음 {info.n_labeled} · 빈 라벨 {info.n_empty} · 라벨 없음 {info.n_missing})",
         (f"<b>클래스</b>: {len(info.class_names)}개 — " + ", ".join(
             f"{_esc(n)} {info.class_counts.get(i, 0)}" for i, n in enumerate(info.class_names[:12]))
          if info.task == "classify" else
          f"<b>객체 수</b>: {info.n_instances}개 · 클래스 {len(info.class_names)}개")]
    if info.split_counts:
        L.append("<b>기존 분할</b>: " + ", ".join(f"{k} {v}" for k, v in sorted(info.split_counts.items())))
    if info.yaml_path:
        L.append(f"<b>yaml</b>: {info.yaml_path.name} {('· ' + info.yaml_style) if info.yaml_style else ''}")
    if info.size_stats:
        s = info.size_stats
        L.append(f"<b>해상도</b>: {s['min_w']}×{s['min_h']} ~ {s['max_w']}×{s['max_h']} (중앙값 {s['med_w']}×{s['med_h']}, {s['sampled']}장 표본)")
    L.append(f"<b>판별 태스크</b>: {TASK_LABELS[info.task]} (신뢰도 {info.task_confidence})")
    for rr in info.task_reasons:
        L.append(f"&nbsp;&nbsp;└ {rr}")
    if info.version_evidence:
        L.append("<b>버전 단서</b>: " + ", ".join(info.version_evidence))
        for t in info.version_evidence_text[:3]:
            L.append(f"&nbsp;&nbsp;└ <span style='color:#666'>{_esc(t)}</span>")
    else:
        L.append("<b>버전 단서</b>: 없음 — YOLO txt 라벨은 v5~v26이 같은 포맷이라 태스크 기준으로 추천합니다.")
    L.append(f"<b>추천 모델</b>: {info.recommended_version} — {VERSIONS[info.recommended_version]['note']}")
    for n in info.notes:
        L.append(f"<span style='color:#1d6fb8'>ℹ {n}</span>")
    for w in info.warnings:
        L.append(f"<span style='color:#c0392b'>⚠ {w}</span>")
    return "<br>".join(L)


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
