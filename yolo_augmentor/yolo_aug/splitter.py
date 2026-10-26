"""train / val / test 분할."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .dataset import Sample, _dominant_class

SPLITS = ("train", "val", "test")


def split_samples(samples: list[Sample], train: float, val: float, test: float, seed: int = 42,
                  stratify: bool = True, keep_existing: bool = False) -> dict[str, list[Sample]]:
    total = train + val + test
    ratios = np.array([train, val, test], dtype=np.float64) / (total if total > 0 else 1)
    rng = np.random.default_rng(seed)
    out: dict[str, list[Sample]] = {s: [] for s in SPLITS}
    pool = samples
    if keep_existing:
        pool = []
        for s in samples:
            if s.orig_split in out:
                out[s.orig_split].append(s)
            else:
                pool.append(s)
    groups: dict = defaultdict(list)
    if stratify:
        for s in pool:
            groups[s.cls_name if s.cls_name is not None else _dominant_class(s.label)].append(s)
    else:
        groups[0] = list(pool)
    # 작은 그룹의 반올림 오차가 누적되지 않도록 전체 목표치 기준으로 배정
    carry = np.zeros(3)
    for key in sorted(groups, key=str):
        g = groups[key]
        idx = rng.permutation(len(g))
        want = ratios * len(g) + carry
        n = np.floor(want).astype(int)
        rem = len(g) - n.sum()
        for j in np.argsort(-(want - n))[:rem]:
            n[j] += 1
        carry = want - n
        a, b = n[0], n[0] + n[1]
        out["train"] += [g[i] for i in idx[:a]]
        out["val"] += [g[i] for i in idx[a:b]]
        out["test"] += [g[i] for i in idx[b:]]
    return out
