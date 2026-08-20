# Final submission explanation video script

목표 길이: 5–7분. 최종 후보는 epoch 89로 확정됐고 `[FINAL]` 점수·hash만
공식 평가 뒤 교체한다.

## 0:00–0:30 소개

“안녕하세요. 2026 SNU FastMRI Challenge에 참가한 [팀명/팀원]입니다. 저희 최종 제출 모델은 FI-VarNet 6+6 구조에 MRAugment와 cross-acceleration re-masking을 적용한 모델입니다.”

## 0:30–1:15 규정과 데이터

“학습에는 제공된 train과 validation만 사용했고 외부 데이터와 pretrained weight는 사용하지 않았습니다. Leaderboard 데이터는 학습, validation, model selection에 사용하지 않았습니다. 최종 checkpoint는 VESSL GTX 1080에서 처음부터 학습한 결과입니다.”

## 1:15–2:15 모델

- sensitivity estimation
- 6 feature cascades + 6 image cascades
- bbox-aware SSIM
- GTX 1080 8 GB에 맞춘 checkpointing/batch/accumulation

## 2:15–3:15 staged training

“Stage 1은 원래 100-epoch schedule을 유지한 채 epoch 50에서 checkpoint를 저장합니다. Stage 2는 model뿐 아니라 optimizer, scheduler, RNG state를 모두 복원하고 cross-acceleration re-masking을 켜서 epoch 89까지 진행합니다.”

- epoch-50 hash 화면
- stage-2 resume log
- epoch 80/85 snapshot과 epoch 89 저장 화면

## 3:15–4:00 최종 checkpoint 선택 `[FINAL]`

“최종 제출은 완료 epoch 89의 원본 checkpoint를 byte-for-byte 복사한
`epoch89` 후보로 확정했습니다. 80/85/89 weight average는 diagnostic 성능이
하락해 제출 대상에서 제외했습니다. 제출 checkpoint의 SHA-256은
`[FINAL_SHA256]`입니다.”

후보를 추가로 조합하거나 public leaderboard 결과로 다시 선택하지 않았음을
설명한다.

## 4:00–4:45 inference

“`prep_volume()`은 k-space와 mask의 순수 입력 전처리만 수행합니다. IFFT, sensitivity estimation, model forward를 포함한 reconstruction은 `recon_slice()` 안에서 slice별로 실행됩니다. Image label, annotation, GRAPPA image는 inference 입력으로 사용하지 않습니다.”

## 4:45–5:30 공식 평가 `[FINAL]`

- 변경 없는 `recon_eval.py` 실행 화면
- SSIM_full `[FINAL]`
- SSIM_bbox `[FINAL]`
- inference time `[FINAL]` ms/slice
- 평가 전후 checkpoint hash 동일 화면

## 5:30–6:15 재현

“Fresh VESSL directory에서는 requirements 설치 후 `run_final_staged_reproduction.sh`, candidate 생성 script, `run_final_eval.sh` 순서로 실행합니다. 모든 config, train log, pip freeze, GPU 정보, git 상태, 초기 loss trajectory 비교는 evidence 폴더에 포함했습니다.”

## 6:15–종료

“README의 명령과 제출 checkpoint, leaderboard entry는 같은 SHA-256으로 연결되어 있습니다. 감사합니다.”
