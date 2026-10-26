# YOLO Dataset Augmentor

> **v1.4 작업 흐름**: ① 데이터셋 생성(원본 이미지·동영상 프레임 가져오기 → 라벨링(박스·폴리곤·키포인트·이미지 분류) → **AI 라벨 보조(프로젝트 전용 모델 학습 → 미리 라벨 → 수정 → 반복)** → 객체 합성 → 검수 → YOLO 태스크 선택 내보내기) → ② 데이터셋 증강(분할·전처리·증강)
>
> v1.4에서는 생성형 기능(텍스트→이미지, SDXL 이미지 변형)을 화면에서 제외했습니다. 코드는 남아 있으며 `yolo_aug/genai/models.py`의 `GENERATIVE_ENABLED`와 단계 목록으로 다시 켤 수 있습니다.
>
> 지원 태스크(YOLO로 통일): **Detect · Instance Segment · OBB · Pose(자세 추정) · Classify**.
> YOLO가 지원하지 않는 태스크(시맨틱 세그멘테이션, 깊이, 광류, 행동 인식, MOT 포맷)는 제외했습니다 — 추적(MOT)은 Detect 데이터셋 + YOLO 트래커(ByteTrack/BoT-SORT)로 대응합니다.

YOLO 형식 데이터셋(이미지 + `.txt` 라벨)을 **분할 → 전처리 → 증강**하여 바로 학습 가능한 새 데이터셋으로 만들어 주는 로컬 GUI 도구입니다.
Detect · Segment · Pose · OBB 라벨 좌표를 모든 변환에 맞춰 정확히 재계산하고, Classify(`<클래스>/이미지` 폴더) 데이터셋도 분할·증강합니다.

## 빠른 실행

| OS | 바로 실행 (Python 필요) | 단독 실행 파일 만들기 |
|---|---|---|
| **Windows 10/11** | `windows\run.bat` 더블클릭 | `windows\build_exe.bat` → `dist\YOLOAugmentor\YOLOAugmentor.exe` |
| **Ubuntu 20.04+** | `./ubuntu/run.sh` | `./ubuntu/build_app.sh` → `dist/YOLOAugmentor/YOLOAugmentor`<br>앱 메뉴 등록: `./ubuntu/install_desktop_entry.sh` |

- 첫 실행 시 가상환경을 만들고 패키지를 설치합니다(인터넷 필요, 1~3분). 이후에는 바로 뜹니다.
  - 가상환경 위치: 기본 `.venv`. **프로젝트가 OneDrive 폴더(바탕화면 포함) 안에 있으면 `%LOCALAPPDATA%\YOLOAugmentor\venv`** 에 만듭니다 — OneDrive 동기화가 venv 파일을 깨뜨리기 때문입니다(`No module named 'pip._vendor.rich.markup'` 등). 직접 지정: `set YOLOAUG_VENV=D:\venvs\yoloaug`
  - 가상환경이 손상되면 실행 스크립트가 자동으로 다시 만듭니다.
- 빌드된 `dist/YOLOAugmentor` **폴더 전체**를 복사하면 Python이 없는 PC에서도 실행됩니다 (같은 OS끼리만).
- Ubuntu에서 Qt가 안 뜨면: `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 fonts-noto-cjk`
- Python 3.9 이상 필요. 직접 실행: `pip install -r requirements.txt && python main.py`

## ② 데이터셋 증강 탭 (GUI 단계)

| 단계 | 내용 |
|---|---|
| 데이터셋 · 모델 | 폴더 선택 → 자동 분석: 폴더 구조, **태스크 자동 판별**, 클래스/객체 수, 해상도, **YOLO 버전 단서** 탐색 및 추천. 태스크·버전·클래스 이름·kpt_shape/flip_idx 수정 가능 |
| 0. 분할 비율 | train/val/test 비율(프리셋 제공), 클래스 층화 분할, 기존 분할 유지, 시드 |
| 1. 전처리 선택 | 타일 분할, 리사이즈(letterbox/stretch/fit_long), 자동 대비, CLAHE, 노이즈 제거, 흑백, 작은 객체 제거, 배경 이미지 비율 조정 |
| 2. 전처리 수치 | 선택 항목의 수치 입력 + 원본/결과 **실시간 미리보기** |
| 3. 증강 선택 | 기하(좌우/상하 반전, 90° 회전, 회전, 스케일, 이동, shear, 원근, 크롭), 모자이크, 색상·화질(HSV, 밝기·대비, 감마, CLAHE, 흑백, 샤픈, 안개, 블러, 모션블러, 노이즈, cutout, JPEG 열화)<br>프리셋: **오프라인 권장(학습기와 중복 최소화)** / **학습기 기본 hyp 재현** |
| 4. 증강 수치 | 항목별 적용 확률·강도 + 공통 설정(채움 값, 최소 잔존율, 최소 크기) + 증강 결과 5장 미리보기 |
| 5. 배수 · 실행 | 원본 1장당 증강 배수, 원본 포함 여부, 적용 split(기본 train만), 저장 형식, 스레드 수 → 실행/취소 |

### 태스크 자동 판별 규칙
- `data.yaml`에 `kpt_shape` → **Pose**
- 모든 라인이 5개 값 → **Detect**
- 9개 값(4점)이고 대부분 회전 사각형 → **OBB**, 아니면 4점 폴리곤 **Segment**
- 길이가 고정된 긴 라인 + 키포인트가 bbox 내부 → **Pose** (kpt_shape 추정)
- 가변 길이 폴리곤 → **Segment**
- 라벨 txt가 없고 이미지가 `[<split>/]<클래스>/` 폴더에 있음 → **Classify** (폴더 이름 = 클래스)
  - Classify에서는 타일 분할·작은 객체 제거·배경 비율·모자이크를 적용하지 않습니다 (라벨이 이미지 단위이므로).

### YOLO 버전은 왜 "추정"인가
YOLO txt 라벨 포맷은 v5~v26이 동일해서 라벨만으로 버전을 확정할 수는 없습니다. 대신
① Roboflow `README*.txt`, 학습 기록 `args.yaml`, `*.pt` 가중치 파일명 등의 버전 언급, ② 태스크 호환성(OBB/Pose → v8 이상)
으로 추천하고, 선택한 버전에 따라 다음이 달라집니다.
- `data.yaml` 형식 (v5/v7: `names` 리스트 + `nc`, v8+: dict)
- 호환되지 않는 태스크 경고 (예: YOLOv10 + Segment)
- **학습기가 이미 온라인으로 하는 증강(⟳ 표시)** 안내 → 오프라인과 중복돼 과증강되지 않도록
- "학습기 기본 hyp 재현" 프리셋 값 (v7/v9는 scale 0.9 등)

## 출력 구조
```
<출력폴더>/
  images/{train,val,test}/   labels/{train,val,test}/
  data.yaml              # 선택한 YOLO 버전 형식, 바로 학습 가능 (yolo train data=.../data.yaml)
  augment_config.json    # 이번 실행 설정 (재현/CLI 재실행용)
  report.json            # split별 이미지·객체 수, 제거된 라벨 통계, 오류
```
파일명: `원본명.jpg`(전처리본), `원본명_aug1.jpg`…(증강본), 타일 분할 시 `원본명_t{행}_{열}...`

Classify 출력: `<출력폴더>/{train,val,test}/<클래스>/이미지` + `classes.txt` (data.yaml 없음 — `yolo classify train data=<출력폴더>`)

## 라벨 처리 방식
- 모든 라벨을 픽셀 좌표 점 집합으로 통일 → 기하 변환을 **3×3 행렬 하나로 합성**해 이미지 보간 1회, 점 변환 정확.
- Detect: 변환된 꼭짓점의 외접 박스 후 클리핑 / Segment: 폴리곤 클리핑(Sutherland–Hodgman) / OBB: 잘리면 최소 외접 회전 사각형 재계산 / Pose: 밖으로 나간 키포인트는 `0 0 0`, **반전 시 flip_idx로 좌우 교환**.
- 원래 면적 대비 `최소 잔존율` 미만만 남은 객체와 `최소 크기` 미만 객체는 라벨에서 제거.
- 전처리는 모든 split에 동일 적용, 증강은 선택한 split(기본 train)만 적용.

## ① 데이터셋 생성 탭 (원본 이미지에서 시작)

| 단계 | 내용 |
|---|---|
| 1 프로젝트 · 원본 | 프로젝트 폴더 생성 → **YOLO 태스크** 선택(나중에 바꿔도 됨) → 객체 클래스 · **키포인트 스키마**(사람 17점 COCO / 사람 5점 / 기둥 4점 / 차량 4점 / 직접 정의) · **이미지 분류 클래스** 정의 → 원본 이미지 가져오기 또는 **동영상 프레임 추출**(N초/N프레임 간격, 구간, 비슷한·흐린 프레임 건너뛰기) |
| 2 라벨링 | 박스(B)·폴리곤(P)·**키포인트(K)**·선택/이동(V), 이미지 분류 콤보. 범용 자동 라벨: OWLv2/DINO+SAM, ViTPose(사람 17점), CLIP. 점선 = 미확정 AI 라벨 → [확정]. 오른쪽 아래 *Project model* 으로 현재 이미지에 프로젝트 모델 적용 |
| 3 AI Label Assist | **이 프로젝트에서 확정한 라벨만으로** 모델을 학습해 나머지 이미지를 미리 라벨 (아래 참고) |
| 4 객체 합성 | 라벨된 객체(폴리곤·키포인트 포함)나 투명 PNG 컷아웃을 배경에 합성 |
| 5 검수 | 합성 실행별 결과 썸네일, 통과/거부 전환 |
| 6 내보내기 → 증강 | 원본 + 통과본을 **Detect / Segment / OBB / Pose / Classify** 중 하나로 변환 → [② 데이터셋 증강 탭으로 보내기] |

### AI Label Assist (프로젝트 전용 라벨 보조)
- 순서: 일부 이미지 라벨링·확정 → **Train** → **Pre-label** (나머지 이미지에 점선 AI 라벨) → 라벨링에서 *AI assist: least confident first* 필터로 불확실한 것부터 수정·확정 → 다시 Train (반복할수록 수정량 감소)
- 학습 데이터: 객체 라벨이 **모두 확정(실선)** 된 이미지만 사용, 점선이 남은 이미지는 제외. 데이터는 프로젝트 밖으로 나가지 않음 (`<프로젝트>/assist/<모델>/`)

| 보조 대상 | 엔진 | 라이선스 | 비고 |
|---|---|---|---|
| 박스 (Detect/OBB) | **RF-DETR** Nano/Small/Medium 파인튜닝 (권장) | Apache-2.0 | 클래스당 확정 20장 이상 권장, 8GB GPU는 Nano + batch 2~4 |
| 박스 | Examples only (OWLv2 예시 기반, 학습 없음) | Apache-2.0 | 라벨 몇 개만 있을 때 |
| 폴리곤 (Segment/OBB) | RF-DETR Seg 파인튜닝 | Apache-2.0 | 마스크 → 폴리곤 |
| 키포인트 (Pose) | **ViTPose 파인튜닝** (프로젝트 스키마 K점, 기둥 4점 등 사용자 정의 가능) | Apache-2.0 | 박스가 있는 대상 클래스 객체에 키포인트 예측 → 새 이미지는 박스 보조 먼저 |
| 이미지 분류 | CLIP 특징 + 선형 분류기 | MIT | CPU로 수 초 |
| 박스·폴리곤·키포인트 | Ultralytics YOLO26 (선택) | **AGPL-3.0** | 사내 라벨링용으로만, 납품물에 포함 금지. `requirements-ultralytics.txt` 별도 설치 |

- 기본 가중치: [모델 · GPU]에서 RF-DETR Nano / Seg Nano(필수, 각 약 0.1GB) 다운로드. 없으면 처음부터 학습(데이터·에폭이 훨씬 많이 필요)
- 기존 확정 라벨은 유지, 같은 클래스·위치(IoU>0.5)의 확정 객체가 있으면 예측 생략. "Replace previous AI labels"는 이전 점선 라벨만 교체
- 이미지 분류 예측은 확정되지 않은 분류에만 들어가고, 라벨링 화면에 *AI suggestion* 으로 표시

- 라벨은 이미지당 JSON 하나(`work/ann/<이름>.json`: 폴리곤 + 키포인트 + 이미지 분류 + 자동/점수)로 보관하고, 내보낼 때 태스크 형식으로 변환합니다.
  - detect: 외접 박스 / segment: 폴리곤 / obb: 최소 외접 회전 사각형 / pose: 박스 + K×(x y v), data.yaml에 `kpt_shape`·`flip_idx` / classify: `<클래스>/이미지` 폴더
  - Pose는 키포인트 대상 클래스(와 변형으로 재매핑됐지만 키포인트가 달린 클래스)만 포함하는 것이 기본입니다.
- 변형으로 생긴 새 클래스(예: `traffic_light_fallen`)는 이름 기준으로 id를 통일합니다.
- v1.2 프로젝트(`work/labels/*.txt`)는 열 때 자동으로 JSON으로 변환되고 원본 txt는 `labels_v12_backup`에 보관됩니다.
- 자동 라벨 모델(OWLv2·Grounding DINO·SAM·ViTPose: Apache-2.0, CLIP: MIT)은 [모델 · GPU] 탭에서 받습니다. 모델 없이도 수동 라벨링은 가능합니다.

**설치**
- 인터넷 PC: `windows\install_genai.bat` / `./ubuntu/install_genai.sh` → PyTorch + torchvision CUDA 12.8(RTX 30/40/50 공용, 드라이버 570+), transformers, **rfdetr[train]** 설치, (선택) Ultralytics, 모델 다운로드(`tools/download_models.py --required`, 약 2.3GB)
- 폐쇄망 PC: 인터넷 PC의 [모델 · GPU → 폐쇄망 오프라인 번들 만들기]로 대상 OS/Python/Ubuntu 버전에 맞는 번들 생성 → USB로 옮겨 `install_offline.bat|sh`
  (CLI: `python -m yolo_aug.genai.offline_bundle --out D:/bundle --os win --python 3.11 --torch cu128 --models required [--ultralytics]`)
- exe 빌드본에는 PyTorch가 포함되지 않습니다(수 GB). 자동 라벨·AI Label Assist는 `run.bat`/`run.sh`(venv)로 실행하세요. 수동 라벨링·합성·검수·내보내기·증강은 exe에서도 동작합니다.
- 모델 라이선스는 `LICENSES_MODELS.md` 참고 (MIT / Apache-2.0, Ultralytics만 AGPL 선택 사항).

**결과 활용**: [6 내보내기]로 만든 폴더(`images/`, `labels/`, `data.yaml` 또는 Classify의 `<클래스>/` 폴더)가 ② 데이터셋 증강 탭의 입력이 됩니다.

## CLI (서버/배치용)
```bash
python main.py --analyze /data/my_dataset          # 분석 결과만 출력
python main.py --cli --config augment_config.json  # GUI에서 저장한 설정으로 헤드리스 실행
python main.py --gen-copypaste out/gen_report.json   # 객체 합성 재실행
python main.py --extract-frames D:/videos D:/frames --every-sec 0.5 --skip-similar 3   # 동영상 프레임 추출
```

## 테스트
```bash
python tools/make_sample_dataset.py sample_ds --task obb --n 40   # 합성 데이터셋 생성 (detect/segment/pose/obb)
python tools/make_road_scene.py road_ds --n 4                      # 신호등·중앙분리대 도로 장면 (segment 라벨)
python tests/test_core.py
python tests/test_genai.py    # 자동 라벨·내보내기 (+ diffusers가 있으면 SDXL 코드 경로)
python tests/test_assist.py   # AI Label Assist (초소형 랜덤 모델 / RF-DETR 1 epoch CPU)
```

## 폴더 구성
```
main.py                 진입점 (GUI / --cli / --analyze)
yolo_aug/
  dataset.py            스캔·분석·태스크 판별·버전 단서
  labels.py             라벨 파싱/저장, 행렬 변환, 클리핑
  preprocess.py         전처리 (타일, 리사이즈, 화질, 라벨 정리)
  augment.py            증강 (모자이크, 기하, 색상·화질)
  splitter.py           층화 분할
  pipeline.py           전체 실행, data.yaml/리포트 생성, 미리보기
  ops_spec.py           전처리/증강 항목·수치 정의 (GUI 폼 자동 생성)
  versions.py           YOLO 버전별 지원 태스크·학습 hyp·프리셋
  genai/                생성: project(원본·스키마 관리), annstore(JSON 라벨), txt2img(텍스트→이미지),
                        autolabel(OWLv2/GDINO+SAM, ViTPose, CLIP), export(5개 태스크 변환),
                        models, prompts(기본·사용자 효과), effects(절차적), structural(기울어짐·쓰러짐),
                        diffusion(SDXL), quality, transform, copypaste, review, offline_bundle
  assist/               AI Label Assist: data(확정 라벨 선택·YOLO 세트), engines(RF-DETR/Ultralytics),
                        pose(ViTPose 파인튜닝), classify(CLIP 선형), fewshot(OWLv2 예시), core(학습·예측)
  gui/                  PySide6 GUI (genai_tab.py·create_pages.py = ① 데이터셋 생성, pages.py = ② 증강)
windows/  run.bat, build_exe.bat, install_genai.bat
ubuntu/   run.sh, build_app.sh, install_desktop_entry.sh, install_genai.sh
tools/    make_sample_dataset.py, download_models.py
YOLOAugmentor.spec      PyInstaller 빌드 설정 (공용)
```
