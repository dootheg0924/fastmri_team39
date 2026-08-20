# Epoch 89 저장 뒤 남은 명령

## 0. 먼저 현재 VESSL 증빙 보존

최종 준비 commit을 VESSL workspace에 받은 뒤 다음 명령을 실행한다.

```bash
RESULT_ROOT=/root/result bash scripts/capture_submission_evidence.sh
```

생성된 `capture_*` 디렉터리를 보존한다. 지금 한 번, epoch 89 checkpoint와
공식 평가 결과가 생긴 뒤 한 번 더 실행한다.

현재 code-side 준비는 끝났다. 아래에서는 `/root/result`만 실제 VESSL `RESULT_ROOT`와 다를 경우 바꾼다.

## 1. 원본 checkpoint 확인

```bash
export RESULT_ROOT=/root/result
export STAGE2_DIR="${RESULT_ROOT}/final_fivarnet_submission_f6i6_mraugment_cross_acc"

python - <<'PY'
import os
from pathlib import Path
import torch

root = Path(os.environ["STAGE2_DIR"]) / "checkpoints"
for name, expected in [("best_model.pt", 89)]:
    path = root / name
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    observed = int(checkpoint["epoch"])
    if observed != expected:
        raise SystemExit(f"{path}: expected {expected}, found {observed}")
    print(path, observed)
PY

sha256sum "${STAGE2_DIR}/checkpoints/best_model.pt"
```

## 2. 확정된 epoch89 후보 동결

```bash
RESULT_ROOT="${RESULT_ROOT}" bash scripts/prepare_epoch89_final_candidate.sh
```

생성 결과:

```text
${RESULT_ROOT}/final_candidates/epoch89/
```

이미 같은 directory를 생성했다면 덮어쓰지 말고 기존
`candidate_manifest.json`과 checkpoint SHA-256을 검증한다.

## 3. epoch89 공식 평가

```bash
FINAL_CANDIDATE_DIR="${RESULT_ROOT}/final_candidates/epoch89" \
LEADERBOARD_PATH=/root/Data/leaderboard \
GPU_NUM=0 \
bash scripts/run_final_eval.sh
```

이 평가 결과의 checkpoint SHA-256과 candidate manifest SHA-256이 같은지
확인한다. 다른 checkpoint 조합을 추가 비교하지 않는다.

## 4. 확정 선택 주입

최종 ID는 변경하지 않는다.

```bash
export FINAL_ID=epoch89
export FINAL_DIR="${RESULT_ROOT}/final_candidates/${FINAL_ID}"
export EVAL_JSON="$(find "${FINAL_DIR}/evidence/official_eval" -name 'eval_metadata_*.json' -print | sort | tail -n 1)"

python scripts/finalize_submission.py \
  --candidate-dir "${FINAL_DIR}" \
  --eval-metadata "${EVAL_JSON}" \
  --team-name "실제 팀명" \
  --team-members "실제 팀원 이름"

python scripts/verify_submission.py \
  --candidate-dir "${FINAL_DIR}" \
  --eval-metadata "${EVAL_JSON}" \
  --final-readme submission/README.md \
  --require-vessl
```

## 5. PPT/영상과 package

`PRESENTATION_OUTLINE.md`와 `VIDEO_SCRIPT.md`의 `[FINAL]`을 채우고 실제 PPT/영상 파일을 만든다.

```bash
FINAL_CANDIDATE_DIR="${FINAL_DIR}" \
EVAL_METADATA="${EVAL_JSON}" \
TEAM_NAME_SLUG=ascii_team_name \
PRESENTATION_FILE=/absolute/path/to/final.pptx \
VIDEO_FILE=/absolute/path/to/final.mp4 \
EVIDENCE_DIR=/absolute/path/to/evidence \
OUTPUT_ROOT=/absolute/path/to/submission_output \
bash scripts/package_final_submission.sh
```

생성된 `SHA256SUMS`를 다시 검증하고 운영진 이메일 첨부 발표자료의 정확한 파일명/제목 양식을 적용한다.
