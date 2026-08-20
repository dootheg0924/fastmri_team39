# Final submission PPT outline

최종 후보가 정해지기 전까지 슬라이드 1–8은 완성할 수 있다. `[FINAL]` 표시만 선택 후 채운다.

## Slide 1 — Team / submission identity

- 팀명과 팀원
- 2026 SNU FastMRI Challenge
- 최종 모델: FI-VarNet 6+6, MRAugment, cross-acceleration
- `[FINAL]` candidate ID

## Slide 2 — 문제와 규정 준수

- 제공 train/val만 사용
- 외부 data/pretrained weight 없음
- leaderboard를 학습·validation·model selection에 사용하지 않음
- inference에서 image/annotation/GRAPPA 미사용
- VESSL GTX 1080 end-to-end 학습

## Slide 3 — 모델 구조

- sensitivity model
- feature-space cascade 6개
- image-space cascade 6개
- batch 1 / gradient accumulation 4
- bbox-aware SSIM objective

그림은 code의 실제 module 이름과 맞춰 단순 block diagram으로 작성한다.

## Slide 4 — Stage 1

- epoch 0–50
- MRAugment
- 100-epoch LR/MRAugment schedule 유지
- deterministic seed/RNG 설정
- epoch-50 checkpoint 저장·SHA-256

## Slide 5 — Stage 2

- epoch-50 model/optimizer/scheduler/RNG 전체 resume
- cross-acceleration re-masking
- epoch 80, 85 snapshot 및 epoch 89 final state
- 마감 때문에 schedule horizon은 그대로 두고 completed epoch 89에서 stop

## Slide 6 — 후보 두 개와 공통 평가 계약

| 후보 | 정의 |
|---|---|
| `epoch89` | epoch 89 원본 checkpoint |
| `avg_80_85_89` | epoch 80/85/89 floating state의 1:1:1 산술평균 |

- 두 후보 모두 `checkpoints/best_model.pt`
- 각각 독립 `candidate_manifest.json`과 SHA-256
- 동일한 변경 없는 `recon_eval.py` 사용
- `[FINAL]` 선택된 후보와 선택 기준

## Slide 7 — Inference 규정 준수

- `prep_volume`: k-space/mask 입력 전처리만 수행
- `recon_slice`: IFFT/sensitivity/model reconstruction을 포함한 timed path
- `test_part.py`가 image field를 열지 않는 code snippet

## Slide 8 — 재현성/환경

- Python 3.10.12, torch 2.3.1+cu121, NumPy 1.24.4
- deterministic algorithms, CUBLAS workspace, purpose-separated RNG
- resolved config, pip freeze, GPU info, git commit/diff
- 초기 trajectory A/B와 stage resume SHA

## Slide 9 — Final score `[FINAL]`

- checkpoint filename / stored epoch / SHA-256
- SSIM_full
- SSIM_bbox
- ms/slice
- official eval log와 leaderboard screenshot

## Slide 10 — Reproduction command

- `scripts/run_final_staged_reproduction.sh`
- `scripts/prepare_final_candidate.py`
- `scripts/run_final_eval.sh`
- 실제 예상 학습 시간과 evidence 위치
