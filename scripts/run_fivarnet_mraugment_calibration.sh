#!/usr/bin/env bash
# Measure a near-final MRAugment epoch before the scratch final run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/007_fivarnet_mraugment/calibration.env}"

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

CALIBRATION_DIR="${RESULT_ROOT}/${EXP_NAME}"
CALIBRATION_CHECKPOINT="${CALIBRATION_DIR}/checkpoints/model.pt"
if [[ -e "${CALIBRATION_CHECKPOINT}" ]]; then
  echo "[ERROR] Calibration must start from scratch, but a checkpoint exists:" >&2
  echo "        ${CALIBRATION_CHECKPOINT}" >&2
  echo "Set a fresh CALIBRATION_EXP_NAME and rerun." >&2
  exit 2
fi

start_seconds="$(date +%s)"
"${SCRIPT_DIR}/run_fivarnet_bbox.sh"
end_seconds="$(date +%s)"
wall_seconds="$((end_seconds - start_seconds))"

python "${SCRIPT_DIR}/recommend_final_epochs.py" \
  --history "${CALIBRATION_DIR}/training_history.csv" \
  --calibration-wall-seconds "${wall_seconds}" \
  --budget-hours "${TOTAL_ALLOCATION_HOURS}" \
  --reserve-fraction "${FINAL_RESERVE_FRACTION}" \
  --max-epochs 100 \
  --output-env "${CALIBRATION_DIR}/recommended_final_epochs.env" \
  --output-json "${CALIBRATION_DIR}/recommended_final_epochs.json"

echo "Calibration checkpoint is for timing only; do not warm-start from it."
