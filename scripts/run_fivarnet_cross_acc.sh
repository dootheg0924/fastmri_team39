#!/usr/bin/env bash
# Continue the 100-epoch FI-VarNet + MRAugment final run with cross-acceleration
# re-masking. Resumes from EXP_NAME's model.pt when one is present.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/008_fivarnet_cross_acc/config.env}"

exec "${SCRIPT_DIR}/run_fivarnet_bbox.sh"
