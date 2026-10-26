"""폐쇄망(오프라인) 설치 번들 생성기 — 인터넷 PC에서 실행.

번들 구성:
  <번들>/app/        앱 소스
  <번들>/wheels/     대상 OS·Python용 pip 패키지 (PySide6, OpenCV, torch(CUDA), diffusers …)
  <번들>/models/     HuggingFace 모델 (선택한 것만)
  <번들>/python/     (Windows) Python 설치 파일
  <번들>/install_offline.bat | install_offline.sh
CLI: python -m yolo_aug.genai.offline_bundle --out D:/bundle --os win --python 3.11 --torch cu128 --models required
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import APP_ROOT, MODELS, MODELS_BY_KEY, download_model

PY_INSTALLERS = {"3.10": "3.10.11", "3.11": "3.11.9", "3.12": "3.12.10"}
UBUNTU_GLIBC = {"20.04": 31, "22.04": 35, "24.04": 39}


def platforms(target_os: str, ubuntu: str = "22.04") -> list[str]:
    if target_os == "win":
        return ["win_amd64"]
    g = UBUNTU_GLIBC.get(ubuntu, 35)  # 대상 Ubuntu glibc 이하로 빌드된 wheel만 받음
    return [f"manylinux_2_{m}_x86_64" for m in range(g, 16, -1)] + ["manylinux2014_x86_64", "linux_x86_64"]
COPY_ITEMS = ["main.py", "yolo_aug", "tools", "tests", "windows", "ubuntu", "assets", "requirements.txt",
              "requirements-genai.txt", "requirements-ultralytics.txt", "README.md", "YOLOAugmentor.spec", "LICENSES_MODELS.md"]


@dataclass
class BundleConfig:
    output_dir: str
    target_os: str = "win"          # win | linux
    python: str = "3.11"
    torch_variant: str = "cu128"    # cu128 | cu126 | cpu | none
    models: list = field(default_factory=lambda: [m.key for m in MODELS if m.required])
    include_python_installer: bool = True
    ubuntu: str = "22.04"           # linux 대상 Ubuntu 버전 (glibc 호환)
    include_ultralytics: bool = False


def _run(cmd, log, cancel):
    log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace")
    last = 0.0
    for line in p.stdout:
        if cancel is not None and cancel.is_set():
            p.kill()
            raise RuntimeError("취소됨")
        line = line.rstrip()
        if line and (("Saved" in line or "ERROR" in line or "error" in line) or time.time() - last > 2):
            log("  " + line[:200])
            last = time.time()
    if p.wait() != 0:
        raise RuntimeError(f"명령 실패 (exit {p.returncode}): {' '.join(map(str, cmd[:6]))} …")


def _pip_download(args, wheels: Path, cfg: BundleConfig, log, cancel, extra=()):
    cmd = [sys.executable, "-m", "pip", "download", "-d", str(wheels), "--only-binary=:all:",
           "--python-version", cfg.python, "--implementation", "cp"]
    for p in platforms(cfg.target_os, cfg.ubuntu):
        cmd += ["--platform", p]
    _run(cmd + list(extra) + list(args), log, cancel)


def _copy_app(dst: Path, log):
    dst.mkdir(parents=True, exist_ok=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "models", "dist", "build", "genai_settings.json")
    for name in COPY_ITEMS:
        src = APP_ROOT / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst / name, ignore=ign, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst / name)
    log(f"앱 소스 복사 → {dst}")


def _install_bat(cfg: BundleConfig) -> str:
    ver = PY_INSTALLERS.get(cfg.python, cfg.python)
    short = cfg.python.replace(".", "")
    genai = "" if cfg.torch_variant == "none" else " -r app\\requirements-genai.txt torch torchvision"
    if cfg.include_ultralytics and cfg.torch_variant != "none":
        genai += " -r app\\requirements-ultralytics.txt"
    return f"""@echo off
REM ==== YOLO Dataset Augmentor - OFFLINE installer (no internet needed) ====
setlocal
cd /d "%~dp0"
set BUNDLE=%~dp0
set PY=
where py >nul 2>nul && ( py -{cfg.python} -c "import sys" >nul 2>nul && set PY=py -{cfg.python} )
if not defined PY if exist "%LocalAppData%\\Programs\\Python\\Python{short}\\python.exe" set PY="%LocalAppData%\\Programs\\Python\\Python{short}\\python.exe"
if not defined PY if exist "python\\python-{ver}-amd64.exe" (
    echo [1/4] Installing Python {ver} ...
    "python\\python-{ver}-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
    set PY="%LocalAppData%\\Programs\\Python\\Python{short}\\python.exe"
)
if not defined PY (
    echo [ERROR] Python {cfg.python} not found. Install Python {cfg.python} x64 first.
    pause & exit /b 1
)
REM venv location is shared with app\\windows\\run.bat (outside OneDrive if the bundle is inside OneDrive)
call "app\\windows\\_env.bat"
echo [2/4] Creating virtual environment: %VENV%
if exist "%VPY%" rmdir /s /q "%VENV%"
%PY% -m venv "%VENV%" || ( echo [ERROR] venv failed & pause & exit /b 1 )
echo [3/4] Installing packages from local wheels...
"%VPY%" -m pip install --disable-pip-version-check --no-index --find-links wheels -r app\\requirements.txt{genai} || ( echo [ERROR] pip install failed & pause & exit /b 1 )
echo [4/4] Configuring model folder...
"%VPY%" -c "import json,sys; json.dump({{'models_dir': sys.argv[1], 'vram_mode': 'auto', 'offline': True}}, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=2)" "%BUNDLE%models" "app\\genai_settings.json"
"%VPY%" -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo  Done. Start: app\\windows\\run.bat
pause
endlocal
"""


def _install_sh(cfg: BundleConfig) -> str:
    genai = "" if cfg.torch_variant == "none" else " -r app/requirements-genai.txt torch torchvision"
    if cfg.include_ultralytics and cfg.torch_variant != "none":
        genai += " -r app/requirements-ultralytics.txt"
    return f"""#!/usr/bin/env bash
# ==== YOLO Dataset Augmentor - 오프라인 설치 (인터넷 불필요) ====
set -e
B="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$B"
PY=python{cfg.python}
command -v $PY >/dev/null || PY=python3
if ! $PY -c "import sys; assert sys.version.startswith('{cfg.python}')" 2>/dev/null; then
  echo "[경고] Python {cfg.python}이 아닙니다 ($($PY --version)). wheel이 맞지 않으면 설치가 실패합니다."
fi
if ! $PY -c "import venv, ensurepip" 2>/dev/null; then
  echo "[오류] python{cfg.python}-venv 패키지가 필요합니다. 인터넷 PC에서 'apt-get download python{cfg.python}-venv' 로 받은 .deb를 'sudo dpkg -i'로 설치하세요."
  exit 1
fi
echo "[1/3] 가상환경 생성"
$PY -m venv app/.venv
echo "[2/3] 로컬 wheel로 패키지 설치"
app/.venv/bin/python -m pip install --no-index --find-links wheels --upgrade pip setuptools wheel
app/.venv/bin/python -m pip install --no-index --find-links wheels -r app/requirements.txt{genai}
echo "[3/3] 모델 폴더 설정"
app/.venv/bin/python -c "import json,sys; json.dump({{'models_dir': sys.argv[1], 'vram_mode': 'auto', 'offline': True}}, open(sys.argv[2], 'w'), indent=2)" "$B/models" app/genai_settings.json
app/.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())" || true
chmod +x app/ubuntu/*.sh
echo "완료. 실행: app/ubuntu/run.sh   (Qt 라이브러리: libxcb-cursor0, 한글 폰트: fonts-noto-cjk 필요)"
"""


def build_bundle(cfg: BundleConfig, progress=None, log=print, cancel: threading.Event | None = None) -> str:
    progress = progress or (lambda *a: None)
    cancel = cancel or threading.Event()
    out = Path(cfg.output_dir)
    wheels, models = out / "wheels", out / "models"
    for d in (wheels, models):
        d.mkdir(parents=True, exist_ok=True)
    steps = 5 + len(cfg.models)
    k = 0

    def step(msg):
        nonlocal k
        k += 1
        progress(k, steps, msg)
        log(f"[{k}/{steps}] {msg}")

    step("앱 소스 복사")
    _copy_app(out / "app", log)
    step(f"PyTorch 다운로드 ({cfg.torch_variant})")
    torch_whl = None
    if cfg.torch_variant != "none":
        for old in list(wheels.glob("torch-*.whl")) + list(wheels.glob("torchvision-*.whl")):
            old.unlink()
        _pip_download(["torch>=2.7", "torchvision"], wheels, cfg, log, cancel,
                      ["--index-url", f"https://download.pytorch.org/whl/{cfg.torch_variant}",
                       "--extra-index-url", "https://pypi.org/simple"])
        got = sorted(wheels.glob("torch-*.whl"))
        if not got:
            raise RuntimeError("PyTorch wheel을 받지 못했습니다.")
        torch_whl = got[-1]
        if cfg.torch_variant.startswith("cu") and cfg.target_os == "win" and "+cu" not in torch_whl.name:
            raise RuntimeError(f"CUDA용 PyTorch를 받지 못했습니다 ({torch_whl.name}). download.pytorch.org 접속을 "
                               "확인하세요 (차단 시 PyPI의 CPU 빌드로 대체되는 것을 막기 위해 중단).")
        log(f"PyTorch: {torch_whl.name}")
    step("앱 패키지 다운로드 (PySide6, OpenCV, diffusers …)")
    _pip_download(["pip", "setuptools", "wheel"], wheels, cfg, log, cancel)
    _pip_download(["-r", str(APP_ROOT / "requirements.txt")], wheels, cfg, log, cancel)
    if torch_whl is not None:
        # accelerate 등이 torch를 요구하므로, 방금 받은 CUDA 빌드로 고정해 PyPI의 다른 torch가 섞이지 않게 한다
        ver = torch_whl.name.split("-")[1]
        cons = out / "_constraints.txt"
        lines = [f"torch=={ver}"]
        tv = sorted(wheels.glob("torchvision-*.whl"))
        if tv:
            lines.append(f"torchvision=={tv[-1].name.split('-')[1]}")
        cons.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _pip_download(["-r", str(APP_ROOT / "requirements-genai.txt")], wheels, cfg, log, cancel,
                      ["-c", str(cons), "--find-links", str(wheels)])
        if cfg.include_ultralytics:
            _pip_download(["-r", str(APP_ROOT / "requirements-ultralytics.txt")], wheels, cfg, log, cancel,
                          ["-c", str(cons), "--find-links", str(wheels)])
        cons.unlink()
    else:
        log("PyTorch 제외 → 생성형(SDXL) 패키지도 제외 (절차적 변형·합성·검수만 사용 가능)")
    for key in cfg.models:
        if cancel.is_set():
            raise RuntimeError("취소됨")
        step(f"모델 다운로드: {MODELS_BY_KEY[key].name}")
        download_model(key, models, log)
    step("Python 설치 파일")
    if cfg.target_os == "win" and cfg.include_python_installer:
        ver = PY_INSTALLERS.get(cfg.python, cfg.python)
        url = f"https://www.python.org/ftp/python/{ver}/python-{ver}-amd64.exe"
        dst = out / "python" / f"python-{ver}-amd64.exe"
        dst.parent.mkdir(exist_ok=True)
        if not dst.exists():
            log(f"다운로드: {url}")
            urllib.request.urlretrieve(url, dst)
    step("설치 스크립트 작성")
    if cfg.target_os == "win":
        (out / "install_offline.bat").write_text(_install_bat(cfg).replace("\n", "\r\n"), encoding="ascii",
                                                 newline="")
    else:
        p = out / "install_offline.sh"
        p.write_text(_install_sh(cfg), encoding="utf-8")
        p.chmod(0o755)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024**3
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M"), config=asdict(cfg), size_gb=round(size, 2),
                    wheels=sorted(p.name for p in wheels.glob("*.whl")),
                    models={k: MODELS_BY_KEY[k].repo or MODELS_BY_KEY[k].url for k in cfg.models})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    inst = "install_offline.bat 더블클릭" if cfg.target_os == "win" else "bash install_offline.sh"
    run = "app\\windows\\run.bat" if cfg.target_os == "win" else "app/ubuntu/run.sh"
    (out / "README_OFFLINE.txt").write_text(
        "YOLO Dataset Augmentor 폐쇄망 설치 번들\n\n"
        f"대상: {cfg.target_os} / Python {cfg.python} / PyTorch {cfg.torch_variant}\n"
        "1) 이 폴더 전체를 폐쇄망 PC로 복사 (경로에 한글/공백이 없는 곳 권장, 예: D:\\yolo_bundle)\n"
        f"2) {inst}\n"
        f"3) 실행: {run}\n"
        "NVIDIA 드라이버: CUDA 12.8 빌드는 드라이버 570 이상 필요 (RTX 50 시리즈 포함).\n", encoding="utf-8")
    log(f"번들 완료: {out} ({size:.1f} GB)")
    progress(steps, steps, "완료")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="폐쇄망 오프라인 설치 번들 생성")
    ap.add_argument("--out", required=True)
    ap.add_argument("--os", default="win", choices=["win", "linux"])
    ap.add_argument("--python", default="3.11", choices=list(PY_INSTALLERS))
    ap.add_argument("--torch", default="cu128", choices=["cu128", "cu126", "cpu", "none"])
    ap.add_argument("--models", default="required", help="required | all1 | none | 쉼표구분 키")
    ap.add_argument("--no-python-installer", action="store_true")
    ap.add_argument("--ubuntu", default="22.04", choices=list(UBUNTU_GLIBC))
    ap.add_argument("--ultralytics", action="store_true")
    a = ap.parse_args()
    if a.models == "required":
        keys = [m.key for m in MODELS if m.required]
    elif a.models == "all1":
        keys = [m.key for m in MODELS if m.phase == 1]
    elif a.models == "none":
        keys = []
    else:
        keys = [k.strip() for k in a.models.split(",") if k.strip()]
    build_bundle(BundleConfig(a.out, a.os, a.python, a.torch, keys, not a.no_python_installer, a.ubuntu,
                              a.ultralytics))


if __name__ == "__main__":
    main()
