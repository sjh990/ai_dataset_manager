# 모델 라이선스 정리 (2026-09 기준, 납품 전 원문 재확인 필요)

| 키 | 모델 (HuggingFace) | 라이선스 | 상용 | 비고 |
|---|---|---|---|---|
| sdxl_base | stabilityai/stable-diffusion-xl-base-1.0 | CreativeML OpenRAIL++-M | 가능 | Attachment A 사용 제한 준수 |
| sdxl_vae_fp16 | madebyollin/sdxl-vae-fp16-fix | MIT | 가능 | |
| controlnet_canny | diffusers/controlnet-canny-sdxl-1.0 | OpenRAIL++-M | 가능 | fp16 가중치 2.5GB |
| sdxl_inpaint | diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | OpenRAIL++-M | 가능 | |
| clip_b32 | openai/clip-vit-base-patch32 | MIT | 가능 | 품질 필터 · 이미지 분류 자동 라벨 |
| owlv2 (2차) | google/owlv2-base-patch16-ensemble | Apache-2.0 | 가능 | 자동 라벨 |
| grounding_dino (2차) | IDEA-Research/grounding-dino-tiny | Apache-2.0 | 가능 | 자동 라벨 |
| sam_base (2차) | facebook/sam-vit-base | Apache-2.0 | 가능 | 폴리곤 라벨 |
| vitpose_base (3차) | usyd-community/vitpose-base-simple | Apache-2.0 | 가능 | 사람 17점 자세 자동 라벨 · 프로젝트 키포인트 학습 베이스 |
| rfdetr_nano/small/medium | Roboflow RF-DETR Detection (rfdetr 패키지, storage.googleapis.com/rfdetr) | Apache-2.0 | 가능 | 프로젝트 모델 학습 베이스. XL/2XL·Atto/Femto/Pico는 PML 1.0이라 미사용 |
| rfdetr_seg_nano/small/medium | Roboflow RF-DETR Segmentation | Apache-2.0 | 가능 | 프로젝트 세그 모델 학습 베이스 |
| ultra_yolo26* (선택) | Ultralytics YOLO26 | **AGPL-3.0** | 납품 불가 | 사내 라벨링 보조 전용, 기본 설치에서 제외 |

## OpenRAIL++-M 사용 제한 (Attachment A 요약)
법령 위반 / 미성년자 착취·위해 / 타인을 해칠 목적의 허위정보 생성 / 해를 끼칠 목적의 개인식별정보 생성 /
명예훼손·괴롭힘 / 법적 권리에 불리한 완전자동 의사결정 / 사회적 행동·특성 기반 차별 / 취약계층 착취 /
법적 보호 특성 기반 차별 / 의료 조언·판독 / 사법·법집행·이민 절차 또는 범죄 예측용 정보 생성.
- 군사 용도는 명시적으로 금지되어 있지 않으나, '법집행', '완전자동 의사결정' 조항과의 관계를 과제 성격에 맞게 검토할 것.
- 모델(또는 파생 모델)을 재배포할 때 동일한 사용 제한을 고지하고 라이선스 사본을 포함해야 함.
- 생성된 이미지(출력물)의 권리는 사용자에게 있으나, 사용 제한은 출력물의 사용에도 적용됨.

## 의도적으로 제외한 모델
- FLUX.1-dev (비상업 라이선스), Stable Diffusion 3.x (Stability Community License: 연매출 기준 제한)
- Ultralytics YOLO-World / YOLOv8+ 가중치 (AGPL-3.0 — 폐쇄 납품 시 소스 공개 의무)
- Depth-Anything-V2 Base/Large (CC-BY-NC-4.0)
