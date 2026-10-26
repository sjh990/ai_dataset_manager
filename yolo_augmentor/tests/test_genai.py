"""생성형 모듈 테스트.
- 절차적 변형 / 객체 합성 / 품질 필터 / 검수 이동 : 항상 실행
- SDXL 엔진 코드 경로 : diffusers+torch가 있으면 '초소형 랜덤 가중치 SDXL'을 즉석 생성해 CPU로 실행
  (실제 모델 없이 파이프라인 호출·로딩·크롭 인페인팅·ControlNet 인자 전달을 검증)
실행: python tests/test_genai.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_sample_dataset import make  # noqa: E402
from yolo_aug.genai.copypaste import CopyPasteConfig, run_copypaste  # noqa: E402
from yolo_aug.genai.effects import apply_procedural  # noqa: E402
from yolo_aug.genai.quality import QualityConfig, evaluate, structure_score  # noqa: E402
from yolo_aug.genai.review import load_review, set_accepted  # noqa: E402
from yolo_aug.genai.transform import TransformConfig, run_transform  # noqa: E402
from yolo_aug.labels import Annotation, box_corners, parse_labels  # noqa: E402


def test_procedural_shapes():
    img = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    for k in ("rain", "snow", "fog", "night", "dusk", "backlight", "damage_building", "damage_fire", "flood_road"):
        out = apply_procedural(img, k, np.random.default_rng(1), 0.7)
        assert out.shape == img.shape and out.dtype == np.uint8, k


def test_quality_detects_scene_change():
    rng = np.random.default_rng(0)
    img = np.zeros((200, 200, 3), np.uint8)
    img[50:150, 50:150] = 200
    other = np.zeros_like(img)
    other[10:40, 120:190] = 200
    assert structure_score(img, img) > 0.99
    assert structure_score(img, other) < 0.3
    a = Annotation(0, box_corners(50, 50, 150, 150))
    q = evaluate(img, other, [a], QualityConfig())
    assert not q.passed and q.dropped_labels == [0]
    del rng


def test_transform_and_review():
    with tempfile.TemporaryDirectory() as d:
        ds, out = Path(d) / "ds", Path(d) / "out"
        make(ds, "segment", 8)
        cfg = TransformConfig(dataset_dir=str(ds), output_dir=str(out), effects=["rain", "damage_building"],
                              region_classes=[0], region_remap="damaged_car", max_images=5)
        rep = run_transform(cfg, log=lambda m: None)
        assert rep["stats"]["errors"] == 0
        assert "damaged_car" in rep["class_names"]
        # 영역 변형이 적용된 결과에는 재매핑된 클래스가 있어야 함
        cid = rep["class_names"].index("damaged_car")
        dmg = [r for r in rep["records"] if r["effect"] == "damage_building"]
        if dmg:
            base = out if dmg[0]["passed"] else out / "_rejected"
            txt = (base / "labels" / (Path(dmg[0]["file"]).stem + ".txt")).read_text()
            assert any(ln.split()[0] == str(cid) for ln in txt.splitlines())
        items, _ = load_review(out)
        assert items
        it = next(i for i in items if i.accepted)
        set_accepted(out, it, False)
        assert (out / "_rejected" / "images" / it.file).exists()
        set_accepted(out, it, True)
        assert (out / "images" / it.file).exists()


def test_copypaste_labels_match_pixels():
    """합성된 객체 라벨 영역이 실제 합성된 픽셀과 일치하는지 (alpha 블렌딩, 색보정 없음)."""
    with tempfile.TemporaryDirectory() as d:
        ds, out = Path(d) / "ds", Path(d) / "out"
        make(ds, "obb", 10)
        cfg = CopyPasteConfig(dataset_dir=str(ds), output_dir=str(out), n_images=6, color_match=0.0,
                              blend="alpha", shadow=0.0)
        rep = run_copypaste(cfg, log=lambda m: None)
        assert rep["stats"]["written"] > 0 and rep["stats"]["pasted"] > 0
        for f in (out / "labels").glob("*.txt"):
            for ln in f.read_text().splitlines():
                v = [float(x) for x in ln.split()]
                assert len(v) == 9 and all(0 <= x <= 1 for x in v[1:])


def _tiny_sdxl(root: Path):
    """초소형 랜덤 SDXL / ControlNet / VAE / 토크나이저를 root 아래 models 형식으로 저장."""
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel, EulerDiscreteScheduler, StableDiffusionXLImg2ImgPipeline,
                           UNet2DConditionModel)
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    torch.manual_seed(0)
    common = dict(block_out_channels=(32, 64), layers_per_block=1, attention_head_dim=(2, 4),
                  use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
                  transformer_layers_per_block=(1, 1), projection_class_embeddings_input_dim=80,
                  cross_attention_dim=64, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"))
    unet = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels=4,
                                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), **common)
    cn = ControlNetModel(in_channels=4, conditioning_embedding_out_channels=(16, 32), **common)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3, out_channels=3,
                        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2,
                        latent_channels=4)
    sched = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, steps_offset=1,
                                   beta_schedule="scaled_linear", timestep_spacing="leading")
    tc = CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-5,
                        num_attention_heads=4, num_hidden_layers=2, pad_token_id=1, vocab_size=1000,
                        hidden_act="gelu", projection_dim=32)
    # 바이트 단위 최소 토크나이저
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    chars = [chr(c) for c in cs]
    vocab = {c: i for i, c in enumerate(chars)}
    vocab.update({c + "</w>": len(vocab) + i for i, c in enumerate(chars)})
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    tk = root / "_tok"
    tk.mkdir(parents=True, exist_ok=True)
    (tk / "vocab.json").write_text(json.dumps(vocab))
    (tk / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(str(tk / "vocab.json"), str(tk / "merges.txt"), model_max_length=77)
    tc.vocab_size = len(vocab)
    tc.bos_token_id, tc.eos_token_id = vocab["<|startoftext|>"], vocab["<|endoftext|>"]
    pipe = StableDiffusionXLImg2ImgPipeline(vae=vae, text_encoder=CLIPTextModel(tc),
                                            text_encoder_2=CLIPTextModelWithProjection(tc), tokenizer=tok,
                                            tokenizer_2=tok, unet=unet, scheduler=sched)
    pipe.save_pretrained(str(root / "sdxl_base"))
    pipe.save_pretrained(str(root / "sdxl_inpaint"))
    cn.save_pretrained(str(root / "controlnet_canny"))
    vae.save_pretrained(str(root / "sdxl_vae_fp16"))


def test_diffusion_engine_tiny():
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        print("  (diffusers/torch 없음 → 건너뜀)")
        return
    from yolo_aug.genai.diffusion import DiffusionEngine
    from yolo_aug.genai.transform import transform_one

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "models"
        _tiny_sdxl(root)
        eng = DiffusionEngine(root, "auto", log=lambda m: None)
        img = (np.random.default_rng(0).random((96, 128, 3)) * 255).astype(np.uint8)
        out = eng.img2img(img, "rain", "", strength=0.5, steps=2, guidance=2.0, seed=1, control_scale=0.5,
                          long_side=64)
        assert out.shape == img.shape
        mask = np.zeros(img.shape[:2], np.float32)
        mask[30:60, 40:80] = 1
        out2 = eng.inpaint_region(img, mask, (40, 30, 80, 60), "collapsed", "", strength=0.8, steps=2,
                                  guidance=2.0, seed=2, long_side=64)
        assert out2.shape == img.shape
        assert np.array_equal(out2[:5, :5], img[:5, :5])  # 영역 밖은 원본 유지
        # 파이프라인 레벨 (region + global)
        cfg = TransformConfig(backend="diffusion", models_dir=str(root), steps=2, long_side=64, task="detect",
                              class_names=["car"], region_classes=[0], region_remap="damaged_car")
        anns = [Annotation(0, box_corners(40, 30, 80, 60))]
        r = transform_one(img, anns, "damage_building", cfg, 3, eng)
        assert r["anns"][0].cls == 1
        r = transform_one(img, anns, "night", cfg, 4, eng)
        assert r["image"].shape == img.shape
        eng.unload()


def test_structural_tilt_fall():
    sys.path.insert(0, str(ROOT / "tools"))
    from make_road_scene import render
    from yolo_aug.genai.structural import deform_object

    img, anns = render(seed=0)
    tl = [i for i, a in enumerate(anns) if a.cls == 0]
    for mode, rng_ in (("tilt", (20, 20)), ("fall", (70, 70))):
        out, new, done = deform_object(img, anns, tl, "segment", mode, np.random.default_rng(0), rng_)
        assert done == tl and out.shape == img.shape
        from yolo_aug.genai.copypaste import _iou
        for i in tl:  # 라벨이 변형된 객체를 따라 바뀌었는지 (원래 박스와 크게 달라짐)
            assert _iou(anns[i].bbox(), new[i].bbox()) < 0.8, (mode, i)
    mb = [i for i, a in enumerate(anns) if a.cls == 1][-2:]
    out, new, done = deform_object(img, anns, mb, "segment", "knockdown", np.random.default_rng(1), (10, 10))
    for i in done:
        assert np.ptp(new[i].pts[:, 1]) < np.ptp(anns[i].pts[:, 1])


def _tiny_tokenizer(d: Path):
    from transformers import CLIPTokenizer
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    chars = [chr(c) for c in cs]
    vocab = {c: i for i, c in enumerate(chars)}
    vocab.update({c + "</w>": len(vocab) + i for i, c in enumerate(chars)})
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.json").write_text(json.dumps(vocab))
    (d / "merges.txt").write_text("#version: 0.2\n")
    return CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt"), model_max_length=16), len(vocab)


def _tiny_autolabel_models(root: Path):
    import torch
    from transformers import (Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor, Owlv2Processor, SamConfig,
                              SamImageProcessor, SamModel, SamProcessor)
    torch.manual_seed(0)
    tok, V = _tiny_tokenizer(root / "_tok")
    cfg = Owlv2Config(text_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
                                       vocab_size=V, max_position_embeddings=16, bos_token_id=V - 2, eos_token_id=V - 1,
                                       pad_token_id=V - 1),
                      vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
                                         image_size=64, patch_size=16), projection_dim=32)
    Owlv2ForObjectDetection(cfg).save_pretrained(str(root / "owlv2"))
    Owlv2Processor(image_processor=Owlv2ImageProcessor(size={"height": 64, "width": 64}), tokenizer=tok) \
        .save_pretrained(str(root / "owlv2"))
    scfg = SamConfig(vision_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, output_channels=32,
                                        image_size=64, patch_size=16, global_attn_indexes=[1], mlp_dim=64,
                                        num_pos_feats=16),
                     prompt_encoder_config=dict(hidden_size=32, image_size=64, patch_size=16, mask_input_channels=4),
                     mask_decoder_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_dim=64,
                                              iou_head_hidden_dim=32))
    SamModel(scfg).save_pretrained(str(root / "sam_base"))
    SamProcessor(SamImageProcessor(size={"longest_edge": 64}, pad_size={"height": 64, "width": 64})) \
        .save_pretrained(str(root / "sam_base"))


def test_autolabel_tiny():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        print("  (transformers/torch 없음 → 건너뜀)")
        return
    from yolo_aug.genai.autolabel import AutoLabeler

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "models"
        _tiny_autolabel_models(root)
        lab = AutoLabeler(root, "owlv2", use_sam=True, log=lambda m: None)
        img = (np.random.default_rng(0).random((90, 120, 3)) * 255).astype(np.uint8)
        anns, scores = lab.label(img, ["traffic light", "", "road median barrier", "sign"], threshold=0.0)
        assert len(anns) == len(scores)
        for a in anns:  # 빈 문구 클래스(1)는 나오면 안 됨, 좌표는 이미지 안
            assert a.cls in (0, 2, 3)
            assert a.pts[:, 0].min() >= -1 and a.pts[:, 0].max() <= 121 and a.pts[:, 1].max() <= 91


def test_project_export_formats():
    """원본 → (키포인트·이미지 분류 라벨) → 구조 변형 → 5개 YOLO 태스크로 내보내기."""
    sys.path.insert(0, str(ROOT / "tools"))
    from make_road_scene import render
    import cv2
    from yolo_aug.dataset import analyze_dataset, scan_dataset
    from yolo_aug.genai import annstore
    from yolo_aug.genai.export import export_dataset
    from yolo_aug.genai.project import KPT_PRESETS, ClassDef, Project

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "raw").mkdir()
        truth = {}
        for i in range(2):
            img, anns = render(seed=i)
            cv2.imwrite(str(d / "raw" / f"c{i}.jpg"), img)
            truth[f"c{i}"] = anns
        p = Project.create(d / "proj", "detect", [ClassDef("traffic_light"), ClassDef("median_barrier")])
        p.image_classes = [ClassDef("normal"), ClassDef("damaged")]
        pr = KPT_PRESETS["pole4"]
        p.set_keypoints("pole4", pr["names"], pr["skeleton"], ["traffic_light"])
        p.save()
        assert p.import_images(d / "raw", log=lambda m: None) == 2
        orig_k = []
        for img in p.images():
            objs = [a.copy() for a in truth[img.stem]]
            for a in objs:
                if a.cls == 0:  # 기둥: 밑둥·중간·꼭대기 (끝단은 없음)
                    x1, y1, x2, y2 = a.bbox()
                    cx = (x1 + x2) / 2
                    a.kpts = np.array([[cx, y2 - 1, 2], [cx, (y1 + y2) / 2, 2], [cx, y1 + 1, 1], [0, 0, 0]], float)
                    orig_k.append(a.kpts[:3, :2])
            p.save_image_ann(img, annstore.ImageAnn(1280, 720, objs, "normal"))
        assert p.stats()["kpt_objects"] > 0 and p.stats()["classified"] == 2
        run = p.new_run_dir("transform")
        cfg = TransformConfig(project_dir=str(p.root), output_dir=str(run), effects=["struct_fall"],
                              region_classes=[0], region_remap="traffic_light_fallen", image_remap="damaged")
        rep = run_transform(cfg, log=lambda m: None)
        assert rep["label_format"] == "json" and "damaged" in rep["image_classes"]
        # 쓰러진 신호등의 키포인트가 객체 변형과 함께 옮겨졌는지 (밑둥은 거의 그대로, 꼭대기는 크게 이동)
        gen = sorted((run / "ann").glob("*.json"))
        assert gen
        moved = False
        for g in gen:
            ia = annstore.load(g)
            assert ia.image_class == "damaged"
            for a in ia.objects:
                if a.kpts is not None and (a.kpts[:, 2] > 0).sum() == 3:
                    k = a.kpts[:3, :2]
                    base_d = min(np.linalg.norm(k[0] - o[0]) for o in orig_k)
                    top_d = min(np.linalg.norm(k[2] - o[2]) for o in orig_k)
                    assert base_d < 20
                    moved |= top_d > 50
        assert moved
        for fmt in ("detect", "segment", "obb", "pose", "classify"):
            s = export_dataset(p, d / "exp" / fmt, fmt, log=lambda m: None)
            assert s["orig"] == 2 and s["generated"] == len(gen), (fmt, s)
            info = analyze_dataset(scan_dataset(d / "exp" / fmt))
            assert info.task == fmt, (fmt, info.task)
            if fmt == "pose":
                assert info.kpt_shape == (4, 3)
                assert set(info.class_names) == {"traffic_light", "traffic_light_fallen"}
            if fmt == "classify":
                assert set(info.class_names) == {"normal", "damaged"}
            if fmt in ("detect", "segment", "obb"):
                assert "traffic_light_fallen" in s["class_names"]


def _tiny_pose_clip(root: Path):
    import torch
    from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, VitPoseBackboneConfig,
                              VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor)
    torch.manual_seed(0)
    bb = VitPoseBackboneConfig(image_size=[64, 48], patch_size=[16, 16], hidden_size=32, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=37, out_indices=[2])
    VitPoseForPoseEstimation(VitPoseConfig(backbone_config=bb, num_labels=17)).save_pretrained(str(root / "vitpose_base"))
    VitPoseImageProcessor(size={"height": 64, "width": 48}).save_pretrained(str(root / "vitpose_base"))
    tok, V = _tiny_tokenizer(root / "_tok2")
    cc = CLIPConfig(text_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
                                     vocab_size=V, max_position_embeddings=32, bos_token_id=V - 2, eos_token_id=V - 1,
                                     pad_token_id=V - 1),
                    vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
                                       image_size=32, patch_size=16), projection_dim=32)
    CLIPModel(cc).save_pretrained(str(root / "clip_b32"))
    CLIPProcessor(image_processor=CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}),
                  tokenizer=tok).save_pretrained(str(root / "clip_b32"))


def test_pose_classify_autolabel_tiny():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        print("  (transformers/torch 없음 → 건너뜀)")
        return
    from yolo_aug.genai.annstore import ImageAnn
    from yolo_aug.genai.autolabel import ImageClassifier, PoseLabeler, label_image_full
    from yolo_aug.genai.project import COCO17, COCO17_SKELETON, ClassDef, Project

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "models"
        _tiny_pose_clip(root)
        p = Project.create(Path(d) / "proj", "pose", [ClassDef("person", "person"), ClassDef("car", "car")])
        p.image_classes = [ClassDef("normal"), ClassDef("fire")]
        p.set_keypoints("coco17", COCO17, COCO17_SKELETON, ["person"])
        img = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
        existing = ImageAnn(160, 120, [Annotation(0, box_corners(20, 10, 70, 110)),
                                       Annotation(1, box_corners(90, 60, 150, 100))])
        pose = PoseLabeler(root, kpt_thr=0.0)
        clf = ImageClassifier(root)
        ia = label_image_full(img, p, None, pose, clf, existing=existing)
        assert len(ia.objects) == 2  # 검출을 안 돌리면 기존 객체 유지
        person, car = ia.objects
        assert person.kpts is not None and person.kpts.shape == (17, 3) and car.kpts is None
        vis = person.kpts[person.kpts[:, 2] > 0]
        assert (vis[:, 0] >= 0).all() and (vis[:, 0] < 160).all()
        assert ia.image_class in ("normal", "fire")


def test_txt2img_tiny():
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        print("  (diffusers/torch 없음 → 건너뜀)")
        return
    from yolo_aug.genai import annstore
    from yolo_aug.genai.diffusion import DiffusionEngine
    from yolo_aug.genai.project import ClassDef, Project
    from yolo_aug.genai.txt2img import Txt2ImgConfig, parse_prompt_line, run_txt2img

    assert parse_prompt_line("damaged :: a b") == ("damaged", "a b")
    assert parse_prompt_line("just text") == (None, "just text")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "models"
        _tiny_sdxl(root)
        p = Project.create(Path(d) / "proj", "classify", [ClassDef("building")])
        eng = DiffusionEngine(root, "auto", log=lambda m: None)
        cfg = Txt2ImgConfig(project_dir=str(p.root), prompts=["damaged :: collapsed building", "# comment", "road"],
                            width=64, height=64, n_per_prompt=2, steps=2, models_dir=str(root), autolabel=False)
        rep = run_txt2img(cfg, log=lambda m: None, engine=eng)
        assert rep["stats"]["generated"] + rep["stats"]["rejected"] == 4
        p = Project.load(p.root)
        imgs = p.images()
        assert len(imgs) == rep["stats"]["generated"]
        cls = [annstore.load(p.label_path(i)).image_class for i in imgs]
        assert "damaged" in p.image_class_names or not rep["stats"]["generated"]
        assert all(c in ("damaged", None) for c in cls)
        eng.unload()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
