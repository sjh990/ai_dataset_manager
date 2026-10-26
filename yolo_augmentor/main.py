"""YOLO Dataset Augmentor 진입점.

  GUI          : python main.py
  데이터셋 분석 : python main.py --analyze <dataset_dir>
  CLI 실행     : python main.py --cli --config augment_config.json
  생성형 배치  : python main.py --gen-transform gen_report.json  (또는 --gen-copypaste)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cli(config_path: str) -> int:
    from yolo_aug.pipeline import PipelineConfig, run_pipeline

    cfg = PipelineConfig.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))

    def prog(d, t, m):
        bar = int(30 * d / max(t, 1))
        print(f"\r[{'#' * bar}{'.' * (30 - bar)}] {d}/{t}", end="", flush=True)

    res = run_pipeline(cfg, prog, lambda m: print("\n" + m))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if not res["n_errors"] else 1


def _analyze(path: str) -> int:
    import re

    from yolo_aug.dataset import analyze_dataset, describe, scan_dataset

    info = analyze_dataset(scan_dataset(path))
    text = describe(info).replace("<br>", "\n").replace("&nbsp;", " ")
    text = re.sub(r"<[^>]+>", "", text)
    print(text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    return 0


def _gen(kind: str, path: str) -> int:
    """생성형 탭 배치를 GUI 없이 실행 (gen_report.json의 config 그대로 재사용 가능)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d = d.get("config", d)

    def prog(done, total, m):
        print(f"\r{done}/{total} {m[:40]:40s}", end="", flush=True)

    if kind == "transform":
        from yolo_aug.genai.transform import TransformConfig, run_transform
        cfg = TransformConfig(**{k: v for k, v in d.items() if k in TransformConfig.__dataclass_fields__})
        rep = run_transform(cfg, prog, lambda m: print("\n" + m))
    else:
        from yolo_aug.genai.copypaste import CopyPasteConfig, run_copypaste
        cfg = CopyPasteConfig(**{k: v for k, v in d.items() if k in CopyPasteConfig.__dataclass_fields__})
        rep = run_copypaste(cfg, prog, lambda m: print("\n" + m))
    print(json.dumps(rep["stats"], ensure_ascii=False))
    return 0


def _frames(a) -> int:
    from yolo_aug.genai.video import FrameConfig, extract_many, list_videos

    vids = list_videos(a.extract_frames[0])
    if not vids:
        print("no video found")
        return 1
    cfg = FrameConfig(mode="frames" if a.every_frames else "seconds",
                      interval=a.every_frames or a.every_sec, start_sec=a.start, end_sec=a.end,
                      max_frames=a.max_frames, long_side=a.long_side, skip_similar=a.skip_similar,
                      skip_blur=a.skip_blur, image_format=a.format)

    def prog(d, t, m):
        print(f"\r{d * 100 // max(t, 1):3d}% {m[:50]:50s}", end="", flush=True)

    rep = extract_many(vids, a.extract_frames[1], cfg, prog, lambda m: print("\n" + m))
    return 0 if rep["saved"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO Dataset Augmentor")
    ap.add_argument("--cli", action="store_true", help="GUI 없이 설정 파일로 실행")
    ap.add_argument("--config", help="augment_config.json 경로 (--cli와 함께)")
    ap.add_argument("--analyze", metavar="DIR", help="데이터셋 분석 결과만 출력")
    ap.add_argument("--gen-transform", metavar="JSON", help="생성형 이미지 변형 배치 (설정 또는 gen_report.json)")
    ap.add_argument("--gen-copypaste", metavar="JSON", help="객체 합성 배치 (설정 또는 gen_report.json)")
    ap.add_argument("--extract-frames", nargs=2, metavar=("VIDEO", "OUT_DIR"))
    ap.add_argument("--every-sec", type=float, default=1.0)
    ap.add_argument("--every-frames", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--long-side", type=int, default=0)
    ap.add_argument("--skip-similar", type=float, default=0.0)
    ap.add_argument("--skip-blur", type=float, default=0.0)
    ap.add_argument("--format", choices=("jpg", "png"), default="jpg")
    a = ap.parse_args()
    if a.extract_frames:
        return _frames(a)
    if a.gen_transform:
        return _gen("transform", a.gen_transform)
    if a.gen_copypaste:
        return _gen("copypaste", a.gen_copypaste)
    if a.analyze:
        return _analyze(a.analyze)
    if a.cli:
        if not a.config:
            ap.error("--cli 사용 시 --config가 필요합니다.")
        return _cli(a.config)
    from yolo_aug.gui.main_window import run_gui

    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
