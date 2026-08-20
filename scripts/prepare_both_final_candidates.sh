#!/usr/bin/env bash
# Materialize both pending choices after the stage-two run reaches epoch 89.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/../result}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-final_fivarnet_submission_f6i6_mraugment_cross_acc}"
STAGE2_CHECKPOINT_DIR="${RESULT_ROOT}/${STAGE2_EXP_NAME}/checkpoints"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${RESULT_ROOT}/final_candidates}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EPOCH80="${STAGE2_CHECKPOINT_DIR}/checkpoint_epoch_0080.pt"
EPOCH85="${STAGE2_CHECKPOINT_DIR}/checkpoint_epoch_0085.pt"
EPOCH89="${STAGE2_CHECKPOINT_DIR}/best_model.pt"

for source in "${EPOCH80}" "${EPOCH85}" "${EPOCH89}"; do
  if [[ ! -s "${source}" ]]; then
    echo "[ERROR] Missing or empty source checkpoint: ${source}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" scripts/prepare_final_candidate.py \
  --mode single \
  --checkpoints "${EPOCH89}" \
  --expected-epochs 89 \
  --output-root "${CANDIDATE_ROOT}" \
  --candidate-id epoch89

"${PYTHON_BIN}" scripts/prepare_final_candidate.py \
  --mode average \
  --checkpoints "${EPOCH80}" "${EPOCH85}" "${EPOCH89}" \
  --expected-epochs 80 85 89 \
  --weights 1 1 1 \
  --output-root "${CANDIDATE_ROOT}" \
  --candidate-id avg_80_85_89

echo "Both candidates are immutable and ready under: ${CANDIDATE_ROOT}"
echo "Evaluate each with scripts/run_final_eval.sh; do not overwrite either directory."
