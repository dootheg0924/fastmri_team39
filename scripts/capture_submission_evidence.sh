#!/usr/bin/env bash
# Freeze reproducibility evidence without modifying a running training job.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/../result}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-final_fivarnet_submission_f6i6_mraugment_all_data}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-final_fivarnet_submission_f6i6_mraugment_cross_acc}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${RESULT_ROOT}/final_submission_evidence}"
OUTPUT_DIR="${EVIDENCE_ROOT}/capture_${TIMESTAMP}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] Evidence capture already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p \
  "${OUTPUT_DIR}/repository" \
  "${OUTPUT_DIR}/environment" \
  "${OUTPUT_DIR}/experiments" \
  "${OUTPUT_DIR}/checkpoint_hashes"

{
  echo "captured_utc=$(date -u --iso-8601=seconds)"
  echo "captured_kst=$(TZ=Asia/Seoul date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "repository=${REPO_ROOT}"
  echo "result_root=${RESULT_ROOT}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
} > "${OUTPUT_DIR}/CAPTURE_INFO.txt"

git status --short --branch > "${OUTPUT_DIR}/repository/git_status.txt"
git log -1 --format=fuller > "${OUTPUT_DIR}/repository/git_head.txt"
git remote -v > "${OUTPUT_DIR}/repository/git_remotes.txt"
git diff --binary HEAD -- > "${OUTPUT_DIR}/repository/working_tree.patch"
git diff --cached --binary -- > "${OUTPUT_DIR}/repository/index.patch"
git ls-files --others --exclude-standard \
  > "${OUTPUT_DIR}/repository/untracked_files.txt"
git archive --format=tar.gz \
  --output="${OUTPUT_DIR}/repository/code_at_commit.tar.gz" HEAD

{
  "${PYTHON_BIN}" -VV
  "${PYTHON_BIN}" -c \
    'import numpy, torch; print(f"torch={torch.__version__}"); print(f"numpy={numpy.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"cudnn={torch.backends.cudnn.version()}"); print(f"cuda_available={torch.cuda.is_available()}"); gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"; print(f"gpu={gpu}")'
} > "${OUTPUT_DIR}/environment/runtime.txt" 2>&1 || true
"${PYTHON_BIN}" -m pip freeze \
  > "${OUTPUT_DIR}/environment/pip_freeze.txt" 2>&1 || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUTPUT_DIR}/environment/nvidia_smi.txt" 2>&1 || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader \
    > "${OUTPUT_DIR}/environment/gpu_processes.txt" 2>&1 || true
else
  echo "nvidia-smi unavailable" > "${OUTPUT_DIR}/environment/nvidia_smi.txt"
fi
ps -eo pid,ppid,lstart,etime,%cpu,%mem,args \
  > "${OUTPUT_DIR}/environment/processes.txt" 2>&1 || true

copy_experiment_evidence() {
  local experiment_name="$1"
  local source_dir="${RESULT_ROOT}/${experiment_name}"
  local destination_dir="${OUTPUT_DIR}/experiments/${experiment_name}"
  local relative_path

  mkdir -p "${destination_dir}"
  for relative_path in \
    resolved_config.env \
    git_state.txt \
    python_environment.txt \
    gpu_environment.txt \
    run_metadata.json \
    training_history.csv \
    val_loss_log.npy; do
    if [[ -s "${source_dir}/${relative_path}" ]]; then
      cp --reflink=auto -- \
        "${source_dir}/${relative_path}" "${destination_dir}/${relative_path}"
    fi
  done

  if [[ -d "${source_dir}" ]]; then
    find "${source_dir}" -maxdepth 2 -type f -name '*.log' -print0 \
      | while IFS= read -r -d '' log_path; do
          relative_path="${log_path#"${source_dir}/"}"
          mkdir -p "${destination_dir}/$(dirname -- "${relative_path}")"
          cp --reflink=auto -- \
            "${log_path}" "${destination_dir}/${relative_path}"
        done
  fi
}

copy_experiment_evidence "${STAGE1_EXP_NAME}"
copy_experiment_evidence "${STAGE2_EXP_NAME}"

for extra_log in \
  "${RESULT_ROOT}/stage2_launch.log" \
  "${RESULT_ROOT}/logs/${STAGE1_EXP_NAME}.log" \
  "${RESULT_ROOT}/logs/${STAGE2_EXP_NAME}.log"; do
  if [[ -s "${extra_log}" ]]; then
    cp --reflink=auto -- "${extra_log}" \
      "${OUTPUT_DIR}/experiments/$(basename -- "${extra_log}")"
  fi
done

CHECKPOINT_DIR="${RESULT_ROOT}/${STAGE2_EXP_NAME}/checkpoints"
for checkpoint in \
  "${RESULT_ROOT}/staged_checkpoints/checkpoint_epoch_0050.pt" \
  "${CHECKPOINT_DIR}/checkpoint_epoch_0080.pt" \
  "${CHECKPOINT_DIR}/checkpoint_epoch_0085.pt" \
  "${CHECKPOINT_DIR}/model.pt" \
  "${CHECKPOINT_DIR}/best_model.pt"; do
  if [[ -s "${checkpoint}" ]]; then
    {
      sha256sum "${checkpoint}"
      stat -c 'inode=%i size=%s mtime=%y path=%n' "${checkpoint}"
    } >> "${OUTPUT_DIR}/checkpoint_hashes/checkpoints.txt"
  fi
done

(
  cd "${OUTPUT_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

echo "Evidence capture completed: ${OUTPUT_DIR}"
echo "Checksums               : ${OUTPUT_DIR}/SHA256SUMS"
