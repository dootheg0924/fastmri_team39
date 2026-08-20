#!/usr/bin/env bash
# Build the final delivery directory after one candidate has been selected.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Final selection is fixed to epoch89. Runtime package inputs still required:"
  echo "  FINAL_CANDIDATE_DIR=${FINAL_CANDIDATE_DIR:-<pending>}"
  echo "  EVAL_METADATA=${EVAL_METADATA:-<pending>}"
  echo "  TEAM_NAME_SLUG=${TEAM_NAME_SLUG:-<pending>}"
  echo "  PRESENTATION_FILE=${PRESENTATION_FILE:-<pending>}"
  echo "  VIDEO_FILE=${VIDEO_FILE:-<pending>}"
  echo "  EVIDENCE_DIR=${EVIDENCE_DIR:-<pending>}"
  exit 0
fi

: "${FINAL_CANDIDATE_DIR:?Set FINAL_CANDIDATE_DIR}"
: "${EVAL_METADATA:?Set EVAL_METADATA to eval_metadata_*.json}"
: "${TEAM_NAME_SLUG:?Set an ASCII TEAM_NAME_SLUG}"
: "${PRESENTATION_FILE:?Set PRESENTATION_FILE}"
: "${VIDEO_FILE:?Set VIDEO_FILE}"
: "${EVIDENCE_DIR:?Set EVIDENCE_DIR}"

if ! [[ "${TEAM_NAME_SLUG}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "[ERROR] TEAM_NAME_SLUG must contain only ASCII letters, digits, dash, underscore." >&2
  exit 2
fi

FINAL_CANDIDATE_DIR="$(realpath "${FINAL_CANDIDATE_DIR}")"
EVAL_METADATA="$(realpath "${EVAL_METADATA}")"
PRESENTATION_FILE="$(realpath "${PRESENTATION_FILE}")"
VIDEO_FILE="$(realpath "${VIDEO_FILE}")"
EVIDENCE_DIR="$(realpath "${EVIDENCE_DIR}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-${REPO_ROOT}/submission_output}")"
FINAL_README="${REPO_ROOT}/submission/README.md"
FINAL_SELECTION="${REPO_ROOT}/submission/FINAL_SELECTION.json"
CANDIDATE_MANIFEST="${FINAL_CANDIDATE_DIR}/candidate_manifest.json"
CHECKPOINT="${FINAL_CANDIDATE_DIR}/checkpoints/best_model.pt"

for required_file in \
  "${FINAL_README}" "${FINAL_SELECTION}" "${CANDIDATE_MANIFEST}" \
  "${CHECKPOINT}" "${EVAL_METADATA}" "${PRESENTATION_FILE}" "${VIDEO_FILE}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "[ERROR] Missing or empty required file: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -d "${EVIDENCE_DIR}" ]]; then
  echo "[ERROR] Evidence directory not found: ${EVIDENCE_DIR}" >&2
  exit 2
fi
if grep -Eq '\{\{[A-Z0-9_]+\}\}' "${FINAL_README}"; then
  echo "[ERROR] Final README still contains unresolved placeholders." >&2
  exit 2
fi

python scripts/verify_submission.py \
  --candidate-dir "${FINAL_CANDIDATE_DIR}" \
  --eval-metadata "${EVAL_METADATA}" \
  --final-readme "${FINAL_README}" \
  --require-vessl

mkdir -p "${OUTPUT_ROOT}"
BUNDLE_DIR="${OUTPUT_ROOT}/${TEAM_NAME_SLUG}_fastmri_final_submission"
if [[ -e "${BUNDLE_DIR}" ]]; then
  echo "[ERROR] Bundle already exists; refusing to overwrite: ${BUNDLE_DIR}" >&2
  exit 3
fi

TEMP_DIR="$(mktemp -d "${OUTPUT_ROOT}/.${TEAM_NAME_SLUG}.build.XXXXXX")"
cleanup() {
  rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

mkdir -p \
  "${TEMP_DIR}/checkpoint" \
  "${TEMP_DIR}/evidence" \
  "${TEMP_DIR}/presentation" \
  "${TEMP_DIR}/video" \
  "${TEMP_DIR}/code"

cp -- "${CHECKPOINT}" "${TEMP_DIR}/checkpoint/best_model.pt"
cp -- "${CANDIDATE_MANIFEST}" "${TEMP_DIR}/candidate_manifest.json"
cp -- "${FINAL_SELECTION}" "${TEMP_DIR}/FINAL_SELECTION.json"
cp -- "${EVAL_METADATA}" "${TEMP_DIR}/"
cp -- "${PRESENTATION_FILE}" "${TEMP_DIR}/presentation/"
cp -- "${VIDEO_FILE}" "${TEMP_DIR}/video/"
cp -a -- "${EVIDENCE_DIR}/." "${TEMP_DIR}/evidence/"

cp -- \
  "${REPO_ROOT}/train.py" \
  "${REPO_ROOT}/recon_eval.py" \
  "${REPO_ROOT}/pytest.ini" \
  "${REPO_ROOT}/requirements.txt" \
  "${REPO_ROOT}/requirements-vessl.lock.txt" \
  "${TEMP_DIR}/code/"
cp -a -- "${REPO_ROOT}/utils" "${TEMP_DIR}/code/utils"
cp -a -- "${REPO_ROOT}/tests" "${TEMP_DIR}/code/tests"
mkdir -p "${TEMP_DIR}/code/experiments" "${TEMP_DIR}/code/scripts" "${TEMP_DIR}/code/submission"
cp -a -- \
  "${REPO_ROOT}/experiments/007_fivarnet_mraugment" \
  "${REPO_ROOT}/experiments/008_fivarnet_cross_acc" \
  "${TEMP_DIR}/code/experiments/"
for script in \
  run_final_staged_reproduction.sh \
  run_fivarnet_bbox.sh \
  run_fivarnet_mraugment.sh \
  run_fivarnet_cross_acc.sh \
  prepare_final_candidate.py \
  prepare_epoch89_final_candidate.sh \
  run_final_eval.sh \
  capture_submission_evidence.sh \
  parse_recon_eval.py \
  finalize_submission.py \
  verify_submission.py \
  package_final_submission.sh \
  vessl_entrypoint.sh \
  smoke_test_training.py \
  analyze_training.py \
  evaluate_validation.py \
  capture_epoch_checkpoint.py \
  export_experiment_report.py; do
  cp -- "${REPO_ROOT}/scripts/${script}" "${TEMP_DIR}/code/scripts/"
done
cp -- \
  "${REPO_ROOT}/submission/README.template.md" \
  "${REPO_ROOT}/submission/EVIDENCE.md" \
  "${REPO_ROOT}/submission/MANIFEST.md" \
  "${REPO_ROOT}/submission/PRESENTATION_OUTLINE.md" \
  "${REPO_ROOT}/submission/VIDEO_SCRIPT.md" \
  "${REPO_ROOT}/submission/EMAIL.template.md" \
  "${REPO_ROOT}/submission/NEXT_STEPS.md" \
  "${TEMP_DIR}/code/submission/"

# The generated submission README is authoritative inside the code archive.
cp -- "${FINAL_README}" "${TEMP_DIR}/code/README.md"
find "${TEMP_DIR}/code" -type d -name '__pycache__' -prune -exec rm -rf -- {} +
find "${TEMP_DIR}/code" -type f \( -name '*.pyc' -o -name '*.log' \) -delete
tar -C "${TEMP_DIR}/code" -czf "${TEMP_DIR}/code.tar.gz" .
rm -rf -- "${TEMP_DIR}/code"

(
  cd "${TEMP_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

mv -- "${TEMP_DIR}" "${BUNDLE_DIR}"
trap - EXIT
echo "Final submission bundle: ${BUNDLE_DIR}"
echo "Checksums             : ${BUNDLE_DIR}/SHA256SUMS"
