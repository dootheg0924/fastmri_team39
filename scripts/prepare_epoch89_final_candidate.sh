#!/usr/bin/env bash
# Materialize the fixed epoch-89 final-submission candidate.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/../result}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-final_fivarnet_submission_f6i6_mraugment_cross_acc}"
SOURCE_CHECKPOINT="${RESULT_ROOT}/${STAGE2_EXP_NAME}/checkpoints/best_model.pt"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${RESULT_ROOT}/final_candidates}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -s "${SOURCE_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing or empty epoch-89 checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/prepare_final_candidate.py \
  --mode single \
  --checkpoints "${SOURCE_CHECKPOINT}" \
  --expected-epochs 89 \
  --output-root "${CANDIDATE_ROOT}" \
  --candidate-id epoch89

echo "Fixed final candidate: ${CANDIDATE_ROOT}/epoch89"
echo "The candidate checkpoint is a byte-for-byte copy of completed epoch 89."
