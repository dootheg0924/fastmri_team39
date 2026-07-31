#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/006_fivarnet_paper/config.env}"

exec bash "${REPO_ROOT}/scripts/run_fivarnet_bbox.sh"
