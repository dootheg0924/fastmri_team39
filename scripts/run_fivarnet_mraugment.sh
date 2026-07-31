#!/usr/bin/env bash
# Launch the 100-epoch FI-VarNet + MRAugment final experiment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/007_fivarnet_mraugment/config.env}"

exec "${SCRIPT_DIR}/run_fivarnet_bbox.sh"
