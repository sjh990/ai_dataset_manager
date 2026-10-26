"""생성 결과 검수: gen_report.json 기반으로 통과/거부 목록을 만들고 사람이 뒤집을 수 있게 한다.
거부 = <out>/_rejected/{images,labels} 로 이동 (삭제하지 않음 → 복원 가능)."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReviewItem:
    file: str
    accepted: bool
    record: dict

    def image_path(self, root: Path) -> Path:
        return (root if self.accepted else root / "_rejected") / "images" / self.file

    def label_path(self, root: Path) -> Path:
        base = root if self.accepted else root / "_rejected"
        j = base / "ann" / (Path(self.file).stem + ".json")
        if j.exists() or (root / "ann").is_dir():
            return j
        return base / "labels" / (Path(self.file).stem + ".txt")


def load_review(root) -> tuple[list[ReviewItem], dict]:
    root = Path(root)
    rep_path = root / "gen_report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {"records": []}
    items: list[ReviewItem] = []
    seen = set()
    for r in report.get("records", []):
        f = r["file"]
        acc = (root / "images" / f).exists()
        if not acc and not (root / "_rejected" / "images" / f).exists():
            continue
        items.append(ReviewItem(f, acc, r))
        seen.add(f)
    # 리포트에 없는 이미지도 포함 (수동 추가분)
    for sub, acc in ((root / "images", True), (root / "_rejected" / "images", False)):
        if sub.is_dir():
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.name not in seen:
                    items.append(ReviewItem(p.name, acc, {}))
    return items, report


def set_accepted(root, item: ReviewItem, accepted: bool) -> None:
    root = Path(root)
    if item.accepted == accepted:
        return
    src_i, src_l = item.image_path(root), item.label_path(root)
    item.accepted = accepted
    dst_i, dst_l = item.image_path(root), item.label_path(root)
    dst_i.parent.mkdir(parents=True, exist_ok=True)
    dst_l.parent.mkdir(parents=True, exist_ok=True)
    if src_i.exists():
        shutil.move(str(src_i), str(dst_i))
    if src_l.exists():
        shutil.move(str(src_l), str(dst_l))


def save_decisions(root, items: list[ReviewItem]) -> None:
    root = Path(root)
    p = root / "gen_report.json"
    if not p.exists():
        return
    rep = json.loads(p.read_text(encoding="utf-8"))
    state = {it.file: it.accepted for it in items}
    for r in rep.get("records", []):
        if r["file"] in state:
            r["reviewed_accepted"] = state[r["file"]]
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
