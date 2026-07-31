#!/usr/bin/env bash
# Launch the fixed-horizon FI-VarNet + MRAugment final experiment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/007_fivarnet_mraugment/config.env}"

# Resolve the calibration artifact before the generic runner sources the same
# config. An explicit FINAL_NUM_EPOCHS always takes precedence.
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
RECOMMENDATION_FILE="${RESULT_ROOT}/${CALIBRATION_EXP_NAME}/recommended_final_epochs.env"
if [[ -z "${FINAL_NUM_EPOCHS:-}" && -f "${RECOMMENDATION_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${RECOMMENDATION_FILE}"
fi
if [[ -z "${FINAL_NUM_EPOCHS:-}" ]]; then
  echo "[ERROR] A fixed FINAL_NUM_EPOCHS is required." >&2
  echo "Run the two-epoch calibration first:" >&2
  echo "  bash scripts/run_fivarnet_mraugment_calibration.sh" >&2
  echo "Or set FINAL_NUM_EPOCHS explicitly from a completed calibration." >&2
  exit 2
fi
if ! [[ "${FINAL_NUM_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] FINAL_NUM_EPOCHS must be a positive integer." >&2
  exit 2
fi
NUM_EPOCHS="${FINAL_NUM_EPOCHS}"
export FINAL_NUM_EPOCHS NUM_EPOCHS

exec "${SCRIPT_DIR}/run_fivarnet_bbox.sh"
