"""SDXL 기반 생성형 변형 엔진 (8GB VRAM 대응).

- img2img + ControlNet(Canny): 윤곽선을 고정해 객체 위치·형태를 유지 → 원본 라벨 재사용
- inpaint (객체 영역 크롭 방식): 대상 객체 주변만 잘라 1024px로 확대해 변형 후 원래 자리에 합성
  → 고해상도 항공영상에서도 나머지 영역 해상도 손실 없음
메모리: fp16 + VAE fp16-fix + (12GB 미만) model CPU offload + VAE tiling. 파이프라인은 1개만 상주.
"""
from __future__ import annotations

import gc
from pathlib import Path

import cv2
import numpy as np

from .models import gpu_info, is_installed, model_path, resolve_vram_mode


class Cancelled(Exception):
    pass


def _round64(v: float) -> int:
    return max(64, int(round(v / 64.0)) * 64)


def fit_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    r = long_side / max(w, h)
    return _round64(w * r), _round64(h * r)


def canny_image(bgr: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, int(low), int(high))
    return np.repeat(e[..., None], 3, axis=2)


class DiffusionEngine:
    def __init__(self, models_root: Path | None = None, vram_mode: str = "auto", log=print):
        import torch

        self.torch = torch
        self.root = Path(models_root) if models_root else None
        self.log = log
        self.info = gpu_info()
        self.device = "cuda" if self.info["cuda"] else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.mode = resolve_vram_mode(vram_mode, self.info["vram_gb"]) if self.device == "cuda" else "cpu"
        self._pipe = None
        self._pipe_key = None
        self.cancel_flag = None  # threading.Event

    # ------------------------------------------------------------------ loading
    def _path(self, key):
        return model_path(key, self.root)

    def has(self, key) -> bool:
        return is_installed(key, self.root)

    def _pretrained(self, cls, path, **kw):
        errors = []
        variants = ["fp16", None] if self.dtype == self.torch.float16 else [None, "fp16"]
        for variant in variants:
            try:
                return cls.from_pretrained(str(path), torch_dtype=self.dtype, variant=variant,
                                           local_files_only=True, **kw)
            except (OSError, ValueError, EnvironmentError) as e:
                errors.append(f"variant={variant}: {e}")
        raise RuntimeError(f"모델 로드 실패: {path}\n" + "\n".join(errors))

    def _vae(self):
        from diffusers import AutoencoderKL

        if self.has("sdxl_vae_fp16"):
            return AutoencoderKL.from_pretrained(str(self._path("sdxl_vae_fp16")), torch_dtype=self.dtype,
                                                 local_files_only=True)
        if self.dtype == self.torch.float16:
            self.log("⚠ sdxl_vae_fp16 미설치: fp16에서 검은 이미지가 나올 수 있습니다.")
        return None

    def _finalize(self, pipe):
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        if self.device == "cuda":
            if self.mode == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif self.mode == "low":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to("cpu")
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        return pipe

    def _tiling(self, pipe, W, H):
        """VAE 타일 디코딩은 기본 타일(보통 1024px)보다 클 때만 사용 (메모리 절약)."""
        vae = getattr(pipe, "vae", None)
        tile = max(1024, getattr(vae, "tile_sample_min_size", 1024) or 1024)
        try:
            vae.enable_tiling() if max(W, H) > tile else vae.disable_tiling()
        except Exception:
            pass

    def unload(self):
        self._pipe, self._pipe_key = None, None
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def _get(self, key: str):
        if self._pipe_key == key:
            return self._pipe
        self.unload()
        from diffusers import (ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline,
                               StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                               StableDiffusionXLPipeline)

        vae = self._vae()
        extra = {"vae": vae} if vae is not None else {}
        self.log(f"모델 로딩: {key} ({self.device}, {self.mode})…")
        if key == "img2img_cn":
            cn = self._pretrained(ControlNetModel, self._path("controlnet_canny"))
            pipe = self._pretrained(StableDiffusionXLControlNetImg2ImgPipeline, self._path("sdxl_base"),
                                    controlnet=cn, **extra)
        elif key == "img2img":
            pipe = self._pretrained(StableDiffusionXLImg2ImgPipeline, self._path("sdxl_base"), **extra)
        elif key == "txt2img":
            pipe = self._pretrained(StableDiffusionXLPipeline, self._path("sdxl_base"), **extra)
        elif key == "inpaint":
            src = "sdxl_inpaint" if self.has("sdxl_inpaint") else "sdxl_base"
            if src == "sdxl_base":
                self.log("ℹ sdxl_inpaint 미설치 → SDXL Base로 인페인팅 (품질 낮음)")
            pipe = self._pretrained(StableDiffusionXLInpaintPipeline, self._path(src), **extra)
        else:
            raise KeyError(key)
        self._pipe, self._pipe_key = self._finalize(pipe), key
        return self._pipe

    # ------------------------------------------------------------------ helpers
    def _gen(self, seed):
        return self.torch.Generator(device="cpu").manual_seed(int(seed))

    def _cb(self):
        def cb(pipe, i, t, kw):
            if self.cancel_flag is not None and self.cancel_flag.is_set():
                raise Cancelled()
            return kw
        return cb

    @staticmethod
    def _pil(bgr):
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _bgr(pil):
        return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ ops
    def img2img(self, bgr, prompt, negative, strength=0.45, steps=25, guidance=6.0, seed=0,
                control_scale=0.65, canny_low=100, canny_high=200, long_side=1024) -> np.ndarray:
        """전체 이미지 변형. 반환은 원본 크기로 복원된 BGR."""
        h, w = bgr.shape[:2]
        W, H = fit_size(w, h, long_side)
        small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA if W < w else cv2.INTER_CUBIC)
        use_cn = self.has("controlnet_canny") and control_scale > 0
        pipe = self._get("img2img_cn" if use_cn else "img2img")
        self._tiling(pipe, W, H)
        steps = max(int(steps), int(np.ceil(1.0 / max(float(strength), 0.01))))  # strength×steps ≥ 1 보장
        kw = dict(prompt=prompt, negative_prompt=negative, image=self._pil(small), strength=float(strength),
                  num_inference_steps=int(steps), guidance_scale=float(guidance), generator=self._gen(seed),
                  callback_on_step_end=self._cb())
        if use_cn:
            kw.update(control_image=self._pil(canny_image(small, canny_low, canny_high)),
                      controlnet_conditioning_scale=float(control_scale))
        out = self._bgr(pipe(**kw).images[0])
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC if w > W else cv2.INTER_AREA)

    def txt2img(self, prompt, negative, width=1024, height=1024, steps=30, guidance=6.5, seed=0) -> np.ndarray:
        """텍스트만으로 새 장면 생성 (SDXL). width/height는 64의 배수로 맞춤."""
        W, H = _round64(width), _round64(height)
        pipe = self._get("txt2img")
        self._tiling(pipe, W, H)
        res = pipe(prompt=prompt, negative_prompt=negative, width=W, height=H, num_inference_steps=int(steps),
                   guidance_scale=float(guidance), generator=self._gen(seed),
                   callback_on_step_end=self._cb()).images[0]
        return self._bgr(res)

    def inpaint_region(self, bgr, mask, box, prompt, negative, strength=0.75, steps=25, guidance=7.0,
                       seed=0, long_side=1024, margin=0.6) -> np.ndarray:
        """box(x1,y1,x2,y2) 주변을 크롭해 inpaint 후 원위치 합성. mask: 원본 크기 0/1 float."""
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        side = max(bw, bh) * (1 + 2 * margin)
        side = min(max(side, 256), max(w, h))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx0 = int(np.clip(cx - side / 2, 0, max(0, w - side)))
        cy0 = int(np.clip(cy - side / 2, 0, max(0, h - side)))
        cw, ch = int(min(side, w)), int(min(side, h))
        crop = bgr[cy0:cy0 + ch, cx0:cx0 + cw]
        cmask = mask[cy0:cy0 + ch, cx0:cx0 + cw]
        W, H = fit_size(cw, ch, long_side)
        small = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)
        smask = cv2.resize((cmask * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        from PIL import Image

        pipe = self._get("inpaint")
        self._tiling(pipe, W, H)
        steps = max(int(steps), int(np.ceil(1.0 / max(float(strength), 0.01))))
        res = pipe(prompt=prompt, negative_prompt=negative, image=self._pil(small),
                   mask_image=Image.fromarray(smask), strength=float(strength), num_inference_steps=int(steps),
                   guidance_scale=float(guidance), generator=self._gen(seed), height=H, width=W,
                   callback_on_step_end=self._cb()).images[0]
        gen = cv2.resize(self._bgr(res), (cw, ch), interpolation=cv2.INTER_AREA)
        feather = max(2, int(min(cw, ch) * 0.02))
        a = cv2.GaussianBlur(cmask.astype(np.float32), (0, 0), feather)[..., None]
        out = bgr.copy()
        out[cy0:cy0 + ch, cx0:cx0 + cw] = (crop * (1 - a) + gen * a).clip(0, 255).astype(np.uint8)
        return out
