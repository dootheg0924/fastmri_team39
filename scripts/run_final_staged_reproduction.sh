#!/usr/bin/env bash
# Reproduce the final 007 -> 008 training pipeline from scratch on VESSL.
# Stage one keeps a 100-epoch schedule but exits after completing epoch 50;
# stage two resumes the complete model/optimizer/scheduler/RNG state. By
# default it runs to epoch 100; FINAL_STAGE_STOP_EPOCH=89 preserves the same
# 100-epoch schedules while finalizing after completed epoch 89.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/root/Data}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/../result}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-final_fivarnet_submission_f6i6_mraugment_all_data}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-final_fivarnet_submission_f6i6_mraugment_cross_acc}"
FINAL_STAGE_STOP_EPOCH="${FINAL_STAGE_STOP_EPOCH:-}"
STAGED_DIR="${RESULT_ROOT}/staged_checkpoints"
STAGE1_DIR="${RESULT_ROOT}/${STAGE1_EXP_NAME}"
STAGE2_DIR="${RESULT_ROOT}/${STAGE2_EXP_NAME}"
STAGE1_CHECKPOINT="${STAGE1_DIR}/checkpoints/model.pt"
STAGED_CHECKPOINT="${STAGED_DIR}/checkpoint_epoch_0050.pt"
STAGE2_CHECKPOINT="${STAGE2_DIR}/checkpoints/model.pt"

if [[ -n "${FINAL_STAGE_STOP_EPOCH}" ]]; then
  if ! [[ "${FINAL_STAGE_STOP_EPOCH}" =~ ^[0-9]+$ ]] \
     || (( FINAL_STAGE_STOP_EPOCH <= 50 || FINAL_STAGE_STOP_EPOCH > 100 )); then
    echo "[ERROR] FINAL_STAGE_STOP_EPOCH must be an integer in [51, 100]." >&2
    exit 2
  fi
fi

for path in "${STAGE1_DIR}" "${STAGE2_DIR}" "${STAGED_CHECKPOINT}"; do
  if [[ -e "${path}" ]]; then
    echo "[ERROR] Reproduction requires fresh outputs; already exists: ${path}" >&2
    exit 2
  fi
done

echo "[STAGE 1/2] Training experiment 007 through completed epoch 50."
DATA_ROOT="${DATA_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}" \
EXP_NAME="${STAGE1_EXP_NAME}" \
STAGE_STOP_EPOCH=50 \
bash scripts/run_fivarnet_mraugment.sh

if [[ ! -s "${STAGE1_CHECKPOINT}" ]]; then
  echo "[ERROR] Stage-one checkpoint is missing or empty: ${STAGE1_CHECKPOINT}" >&2
  exit 3
fi

STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}" python - <<'PY'
import os
import torch

path = os.environ["STAGE1_CHECKPOINT"]
epoch = int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
if epoch != 50:
    raise SystemExit(f"[ERROR] Expected completed epoch 50, found {epoch}: {path}")
print(f"Verified stage-one checkpoint epoch: {epoch}")
PY

mkdir -p "${STAGED_DIR}" "${STAGE2_DIR}/checkpoints"
cp "${STAGE1_CHECKPOINT}" "${STAGED_CHECKPOINT}"
cp "${STAGED_CHECKPOINT}" "${STAGE2_CHECKPOINT}"
sha256sum "${STAGED_CHECKPOINT}" | tee "${STAGED_DIR}/checkpoint_epoch_0050.sha256"

echo "[STAGE 2/2] Resuming experiment 008 from completed epoch 50."
DATA_ROOT="${DATA_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}" \
EXP_NAME="${STAGE2_EXP_NAME}" \
STAGE_STOP_EPOCH="${FINAL_STAGE_STOP_EPOCH}" \
bash scripts/run_fivarnet_cross_acc.sh

EXPECTED_FINAL_EPOCH="${FINAL_STAGE_STOP_EPOCH:-100}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT}" \
EXPECTED_FINAL_EPOCH="${EXPECTED_FINAL_EPOCH}" python - <<'PY'
import os
import torch

path = os.environ["STAGE2_CHECKPOINT"]
expected = int(os.environ["EXPECTED_FINAL_EPOCH"])
epoch = int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
if epoch != expected:
    raise SystemExit(f"[ERROR] Expected completed epoch {expected}, found {epoch}: {path}")
print(f"Verified final stage-two checkpoint epoch: {epoch}")
PY

echo "Final staged reproduction completed: ${STAGE2_DIR}"
