# Team 39 — FastMRI Knee Reconstruction

2026 SNU FastMRI Challenge - Team 39

## Final Result

| Metric | Overall | Acceleration 4 | Acceleration 8 |
|---|---:|---:|---:|
| SSIM full | **0.9316** | 0.9491 | 0.9141 |
| SSIM bbox | **0.9301** | 0.9501 | 0.9100 |
| Reconstruction time | **1379.65 s** (623.4 ms/slice) | 721.62 s (633.0 ms/slice) | 658.03 s (613.3 ms/slice) |

## What We Did

- FI-VarNet 구조: 6개의 feature-domain cascade와 6개의 image-domain cascade
- sensitivity map estimation을 포함한 end-to-end MRI reconstruction
- annotation bounding box를 반영한 SSIM 기반 학습
- MRAugment와 train/validation 데이터 통합 학습
- 기본 학습 후 cross-acceleration re-masking을 적용하는 단계적 학습
- optimizer, scheduler, scaler 상태를 포함한 안정적인 resume 지원

## Reproduction

의존성을 설치합니다.

```bash
python -m pip install -r requirements.txt
```

데이터는 다음 구조를 가정합니다.

```text
/root/Data/
├── train/
├── val/
└── leaderboard/
```

epoch 89까지 최종 학습을 재현합니다.

```bash
DATA_ROOT=/root/Data \
RESULT_ROOT=/root/result \
FINAL_STAGE_STOP_EPOCH=89 \
bash scripts/run_final_staged_reproduction.sh
```

최종 체크포인트를 제출 후보로 준비하고 leaderboard 데이터를 재구성합니다.

```bash
RESULT_ROOT=/root/result \
bash scripts/prepare_epoch89_final_candidate.sh

FINAL_CANDIDATE_DIR=/root/result/final_candidates/epoch89 \
LEADERBOARD_PATH=/root/Data/leaderboard \
GPU_NUM=0 \
bash scripts/run_final_eval.sh
```

## Key Files

| Path | Purpose |
|---|---|
| `train.py` | 학습 진입점과 checkpoint resume |
| `experiments/007_fivarnet_mraugment/config.env` | MRAugment를 적용한 기본 학습 구성 |
| `experiments/008_fivarnet_cross_acc/config.env` | cross-acceleration 단계 학습 구성 |
| `scripts/run_final_staged_reproduction.sh` | 전체 최종 학습 파이프라인 |
| `scripts/prepare_epoch89_final_candidate.sh` | epoch 89 제출 후보 생성 |
| `scripts/run_final_eval.sh` | 최종 재구성 및 제출 전 점검 |
| `utils/learning/test_part.py` | leaderboard 추론 |
| `submission/NEXT_STEPS.md` | 최종 제출 실행 체크리스트 |

## Environment

- Python 3.10.12
- PyTorch 2.3.1+cu121
- CUDA 12.1
- NVIDIA GTX 1080
