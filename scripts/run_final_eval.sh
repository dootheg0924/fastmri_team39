#!/usr/bin/env bash
# Evaluate one immutable candidate with the unchanged official recon_eval.py.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${FINAL_CANDIDATE_DIR:?Set FINAL_CANDIDATE_DIR to a prepared candidate directory}"
LEADERBOARD_PATH="${LEADERBOARD_PATH:-/root/Data/leaderboard}"
GPU_NUM="${GPU_NUM:-0}"
REQUIRE_GTX1080="${REQUIRE_GTX1080:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FINAL_CANDIDATE_DIR="$(realpath "${FINAL_CANDIDATE_DIR}")"
LEADERBOARD_PATH="$(realpath "${LEADERBOARD_PATH}")"
CHECKPOINT="${FINAL_CANDIDATE_DIR}/checkpoints/best_model.pt"
MANIFEST="${FINAL_CANDIDATE_DIR}/candidate_manifest.json"

for required_file in "${CHECKPOINT}" "${MANIFEST}" "${REPO_ROOT}/recon_eval.py"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "[ERROR] Missing or empty required file: ${required_file}" >&2
    exit 2
  fi
done
for acc in acc4 acc8; do
  for kind in image kspace; do
    if [[ ! -d "${LEADERBOARD_PATH}/${acc}/${kind}" ]]; then
      echo "[ERROR] Missing leaderboard directory: ${LEADERBOARD_PATH}/${acc}/${kind}" >&2
      exit 2
    fi
  done
done
if ! [[ "${GPU_NUM}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] GPU_NUM must be a non-negative integer." >&2
  exit 2
fi

REQUIRE_GTX1080="${REQUIRE_GTX1080}" GPU_NUM="${GPU_NUM}" "${PYTHON_BIN}" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is unavailable; official final evaluation must run on VESSL GPU.")
index = int(os.environ["GPU_NUM"])
name = torch.cuda.get_device_name(index)
print(f"Evaluation GPU: {name}")
if os.environ["REQUIRE_GTX1080"] == "1" and "GTX 1080" not in name:
    raise SystemExit(f"[ERROR] Expected a GTX 1080, found: {name}")
PY

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="${FINAL_CANDIDATE_DIR}/evidence/official_eval"
LOG_PATH="${EVIDENCE_DIR}/recon_eval_${TIMESTAMP}.log"
METADATA_PATH="${EVIDENCE_DIR}/eval_metadata_${TIMESTAMP}.json"
mkdir -p "${EVIDENCE_DIR}"

HASH_BEFORE="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
{
  echo "============================================================"
  echo "Official final candidate evaluation"
  echo "Started UTC       : $(date -u --iso-8601=seconds)"
  echo "Repository        : ${REPO_ROOT}"
  echo "Git commit        : $(git rev-parse HEAD)"
  echo "Git status begin"
  git status --short
  echo "Git status end"
  echo "Candidate         : ${FINAL_CANDIDATE_DIR}"
  echo "Checkpoint        : ${CHECKPOINT}"
  echo "Checkpoint SHA256 : ${HASH_BEFORE}"
  echo "Leaderboard       : ${LEADERBOARD_PATH}"
  "${PYTHON_BIN}" -VV
  "${PYTHON_BIN}" -c 'import numpy, torch; print(f"torch={torch.__version__} numpy={numpy.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")'
  nvidia-smi
  echo "============================================================"
  "${PYTHON_BIN}" -u "${REPO_ROOT}/recon_eval.py" \
    -g "${GPU_NUM}" \
    -n "${FINAL_CANDIDATE_DIR}" \
    -p "${LEADERBOARD_PATH}"
  echo "Finished UTC      : $(date -u --iso-8601=seconds)"
} 2>&1 | tee "${LOG_PATH}"

HASH_AFTER="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${HASH_BEFORE}" != "${HASH_AFTER}" ]]; then
  echo "[ERROR] Candidate checkpoint changed during evaluation." >&2
  exit 3
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/parse_recon_eval.py" \
  --candidate-dir "${FINAL_CANDIDATE_DIR}" \
  --log "${LOG_PATH}" \
  --output "${METADATA_PATH}"

echo "Official evaluation log      : ${LOG_PATH}"
echo "Machine-readable score record: ${METADATA_PATH}"
