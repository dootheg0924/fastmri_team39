# {{TEAM_NAME}} — 2026 SNU FastMRI Final Submission

## 1. 제출 식별 정보

| 항목 | 값 |
|---|---|
| 팀명 | {{TEAM_NAME}} |
| 팀원 | {{TEAM_MEMBERS}} |
| 최종 후보 | `{{FINAL_CANDIDATE_ID}}` |
| 선택 방식 | {{FINAL_MODE_DESCRIPTION}} |
| 원본 epoch | {{FINAL_SOURCE_EPOCHS}} |
| checkpoint stored epoch | {{FINAL_STORED_EPOCH}} |
| checkpoint 파일 | `{{FINAL_CHECKPOINT_FILENAME}}` |
| checkpoint bytes | {{FINAL_CHECKPOINT_BYTES}} |
| checkpoint SHA-256 | `{{FINAL_CHECKPOINT_SHA256}}` |
| Leaderboard SSIM_full | **{{FINAL_SSIM_FULL}}** |
| Leaderboard SSIM_bbox | **{{FINAL_SSIM_BBOX}}** |
| Reconstruction time | **{{FINAL_RECON_SECONDS}} s / {{FINAL_MS_PER_SLICE}} ms/slice** |

위 점수와 시간은 SHA-256이 `{{FINAL_CHECKPOINT_SHA256}}`인 checkpoint를 VESSL GTX 1080에서 변경하지 않은 `recon_eval.py`로 실행한 결과다.

## 2. 환경

- VESSL GPU: NVIDIA GTX 1080
- Python: 3.10.12
- PyTorch: 2.3.1+cu121
- NumPy: 1.24.4
- 전체 직접 의존성: `requirements.txt`
- 실제 VESSL `pip freeze`: `requirements-vessl.lock.txt`와 evidence의 `python_environment.txt`

```bash
python -m pip install -r requirements.txt
python -VV
python -c 'import numpy, torch; print(torch.__version__, numpy.__version__, torch.version.cuda, torch.backends.cudnn.version())'
```

## 3. 데이터와 출력 경로

코드는 제공된 `train`/`val`만 학습에 사용한다. 외부 데이터, pretrained weight, leaderboard 데이터·통계량은 학습이나 model selection에 사용하지 않는다.

```text
${DATA_ROOT}/
├── train/{image,kspace}/
├── val/{image,kspace}/
└── leaderboard/{acc4,acc8}/{image,kspace}/
```

모든 경로는 `DATA_ROOT`와 `RESULT_ROOT`로 전달한다. 코드를 수정해 private data 경로를 맞출 필요가 없다.

## 4. 모델과 학습

- FI-VarNet: feature-space cascade 6개 + image-space cascade 6개
- batch size 1, gradient accumulation 4
- bbox-aware SSIM objective
- MRAugment
- epoch 50부터 cross-acceleration re-masking
- train+val 결합 학습, validation checkpoint selection 미사용
- deterministic PyTorch, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=42`
- augmentation/mask/cross-acceleration은 목적별 hash-derived RNG 사용
- checkpoint에 model/optimizer/scheduler/RNG state를 함께 저장하여 stage 2를 재개

최종 선택 규칙은 public leaderboard와 무관하다. 100-epoch schedule을
유지하면서 마감 전에 완료된 epoch 89를 최종 checkpoint로 고정했다. 최종
후보 ID는 `epoch89`, 방식은 `single`이며 원본 checkpoint를 byte-for-byte
복사한다.

## 5. 처음부터 학습 재현

반드시 fresh `RESULT_ROOT`에서 실행한다. `FINAL_STAGE_STOP_EPOCH=89`는 LR/MRAugment horizon을 89로 바꾸지 않고, 원래 100-epoch schedule의 완료 epoch 89에서만 안전하게 멈춘다.

```bash
export DATA_ROOT=/root/Data
export RESULT_ROOT=/root/result_reproduction
export FINAL_STAGE_STOP_EPOCH=89
bash scripts/run_final_staged_reproduction.sh
```

stage 1은 epoch 50까지 실행한 뒤 checkpoint epoch와 SHA-256을 확인한다. stage 2는 그 checkpoint의 model/optimizer/scheduler/RNG state 전체를 복원하고 epoch 89까지 계속한다.

## 6. 최종 후보 생성

원본 checkpoint와 hash는 다음과 같다.

{{SOURCE_CHECKPOINT_TABLE}}

학습이 끝난 뒤 아래 명령으로 제출 checkpoint를 만든다. 기존 candidate directory는 덮어쓰지 않는다.

```bash
{{FINAL_CANDIDATE_BUILD_COMMAND}}
```

`scripts/finalize_submission.py`는 candidate ID `epoch89`, mode `single`, source
epoch `[89]`, stored epoch `89`가 모두 맞지 않으면 최종 README와 selection
record 생성을 거부한다.

## 7. 공식 reconstruction 및 scoring

`recon_eval.py`는 수정하지 않는다. inference 구현은 `utils/learning/test_part.py`에 있다.

```bash
export FINAL_CANDIDATE_DIR="${RESULT_ROOT}/final_candidates/{{FINAL_CANDIDATE_ID}}"
export LEADERBOARD_PATH="${DATA_ROOT}/leaderboard"
export GPU_NUM=0
bash scripts/run_final_eval.sh
```

wrapper는 다음을 자동 확인·보존한다.

- GTX 1080/CUDA 환경
- checkpoint와 `candidate_manifest.json`
- 평가 전후 checkpoint SHA-256 동일성
- Python/PyTorch/NumPy/CUDA/cuDNN 및 `nvidia-smi`
- `recon_eval.py` 전체 stdout/stderr
- 점수와 `ms/slice`를 담은 `eval_metadata_*.json`

## 8. inference 규정 준수

- `prep_volume()`은 k-space/mask 로딩과 입력 전처리만 수행한다.
- IFFT, sensitivity-map estimation, coil combine, model forward 등 reconstruction은 slice별 `recon_slice()` 내부에서 수행된다.
- `test_part.py`는 inference 입력으로 image label, annotation, GRAPPA image를 읽지 않는다.
- `image/*.h5`는 변경 불가능한 scoring harness가 정답/annotation을 계산할 때만 사용한다.

## 9. 재현성 증빙

validation을 학습에 합쳤으므로 validation loss 대신 train loss를 제출한다.

```text
evidence/
├── final_training/       # 전체 terminal/train log, training_history.csv
├── stage_transition/     # epoch-50 checkpoint와 SHA-256
├── official_eval/        # recon_eval log와 eval_metadata JSON
├── reproducibility/      # 초기 trajectory A/B 비교
├── environment/          # pip freeze, GPU/CUDA, resolved config, git state/diff
└── screenshots/          # VESSL/Codex/clock, leaderboard, 전송 화면
```

실제 장기 학습은 VESSL GTX 1080에서 실행되었으며, 제출 code snapshot과 dirty patch를 포함한 source hash를 evidence에 보존한다.

## 10. 주요 파일

| 파일 | 역할 |
|---|---|
| `scripts/run_final_staged_reproduction.sh` | fresh 007→008 end-to-end 학습 |
| `scripts/prepare_epoch89_final_candidate.sh` | 확정된 epoch 89 단일 후보 고정 |
| `scripts/prepare_final_candidate.py` | immutable candidate builder; final wrapper는 single mode만 사용 |
| `scripts/run_final_eval.sh` | 공식 `recon_eval.py` 실행과 환경/log 보존 |
| `scripts/parse_recon_eval.py` | 공식 출력의 machine-readable 기록 |
| `utils/learning/test_part.py` | checkpoint load, `prep_volume`, `recon_slice` |
| `experiments/007_fivarnet_mraugment/` | stage-1 config |
| `experiments/008_fivarnet_cross_acc/` | stage-2 config |
| `requirements.txt` | 재현 환경 직접 의존성 |

문제가 생기면 코드를 임의 수정하지 말고 log의 첫 `[ERROR]`와 `candidate_manifest.json`, `run_metadata.json`, `resolved_config.env`를 먼저 대조한다.
