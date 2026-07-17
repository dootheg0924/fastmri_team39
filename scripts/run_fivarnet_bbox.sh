#!/usr/bin/env bash
# Experiment 003 runner (FI-VarNet + bbox-aware loss).
# Same telemetry/resume workflow as scripts/run_varnet_c6_long.sh, extended
# with the fivarnet model arguments. Works on VESSL and on a local GTX 1080
# (override DATA_ROOT/RESULT_ROOT via environment variables).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/003_fivarnet_bbox/config.env}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Experiment config not found: ${CONFIG_FILE}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"
cd "${REPO_ROOT}"

TRAIN_DIR="${DATA_ROOT}/train"
VAL_DIR="${DATA_ROOT}/val"
EXP_DIR="${RESULT_ROOT}/${EXP_NAME}"
LOG_DIR="${RESULT_ROOT}/logs"
ANALYSIS_DIR="${EXP_DIR}/analysis"
TRAIN_LOG="${LOG_DIR}/${EXP_NAME}.log"
GPU_LOG="${LOG_DIR}/${EXP_NAME}_gpu.csv"
GPU_MONITOR_TARGET="${GPU_MONITOR_TARGET:-${CUDA_VISIBLE_DEVICES%%,*}}"

mkdir -p "${EXP_DIR}" "${LOG_DIR}" "${ANALYSIS_DIR}"
RESOLVED_CONFIG="${EXP_DIR}/resolved_config.env"
{
  printf 'EXP_NAME=%q\n' "${EXP_NAME}"
  printf 'DATA_ROOT=%q\n' "${DATA_ROOT}"
  printf 'RESULT_ROOT=%q\n' "${RESULT_ROOT}"
  printf 'CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'TORCH_GPU_NUM=%q\n' "${TORCH_GPU_NUM}"
  printf 'BATCH_SIZE=%q\n' "${BATCH_SIZE}"
  printf 'NUM_EPOCHS=%q\n' "${NUM_EPOCHS}"
  printf 'LEARNING_RATE=%q\n' "${LEARNING_RATE}"
  printf 'REPORT_INTERVAL=%q\n' "${REPORT_INTERVAL}"
  printf 'SEED=%q\n' "${SEED}"
  printf 'MODEL_NAME=%q\n' "${MODEL_NAME}"
  printf 'CASCADES=%q\n' "${CASCADES}"
  printf 'IMAGE_CASCADES=%q\n' "${IMAGE_CASCADES}"
  printf 'CHANS=%q\n' "${CHANS}"
  printf 'SENS_CHANS=%q\n' "${SENS_CHANS}"
  printf 'POOLS=%q\n' "${POOLS}"
  printf 'SENS_POOLS=%q\n' "${SENS_POOLS}"
  printf 'ATTENTION_CASCADES=%q\n' "${ATTENTION_CASCADES}"
  printf 'KSPACE_MULT_FACTOR=%q\n' "${KSPACE_MULT_FACTOR}"
  printf 'BBOX_LOSS_WEIGHT=%q\n' "${BBOX_LOSS_WEIGHT}"
  printf 'NUM_WORKERS=%q\n' "${NUM_WORKERS}"
  printf 'PIN_MEMORY=%q\n' "${PIN_MEMORY}"
  printf 'CHECKPOINT_INTERVAL=%q\n' "${CHECKPOINT_INTERVAL}"
  printf 'GPU_SAMPLE_INTERVAL=%q\n' "${GPU_SAMPLE_INTERVAL}"
  printf 'RUN_SMOKE_TEST=%q\n' "${RUN_SMOKE_TEST}"
  printf 'RUN_VALIDATION_ANALYSIS=%q\n' "${RUN_VALIDATION_ANALYSIS}"
} > "${RESOLVED_CONFIG}.tmp"
mv "${RESOLVED_CONFIG}.tmp" "${RESOLVED_CONFIG}"
python -m pip freeze > "${EXP_DIR}/python_environment.txt.tmp"
mv "${EXP_DIR}/python_environment.txt.tmp" "${EXP_DIR}/python_environment.txt"
{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "status_begin"
  git status --short
  echo "status_end"
} > "${EXP_DIR}/git_state.txt.tmp"
mv "${EXP_DIR}/git_state.txt.tmp" "${EXP_DIR}/git_state.txt"
touch "${TRAIN_LOG}"
exec > >(tee -a "${TRAIN_LOG}") 2>&1

GPU_MONITOR_PID=""
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ ${status} -ne 0 ]]; then
    echo "[WARN] Training exited with status ${status}; generating partial analysis."
    python scripts/analyze_training.py \
      --exp-name "${EXP_NAME}" \
      --result-root "${RESULT_ROOT}" \
      --gpu-sample-interval "${GPU_SAMPLE_INTERVAL}" || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "============================================================"
echo "Experiment : ${EXP_NAME}"
echo "Start time : $(date --iso-8601=seconds)"
echo "Host       : $(hostname)"
echo "Git commit : $(git rev-parse HEAD)"
echo "Repo root  : ${REPO_ROOT}"
echo "Data root  : ${DATA_ROOT}"
echo "Result dir : ${EXP_DIR}"
echo "Config     : ${CONFIG_FILE}"
echo "============================================================"

for required_dir in \
  "${TRAIN_DIR}/image" "${TRAIN_DIR}/kspace" \
  "${VAL_DIR}/image" "${VAL_DIR}/kspace"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "[ERROR] Required data directory not found: ${required_dir}" >&2
    exit 3
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi is unavailable. This experiment requires a CUDA GPU." >&2
  exit 4
fi

python - <<'PY'
import torch

print(f"Python/Torch preflight: torch={torch.__version__}, CUDA={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] torch.cuda.is_available() is False")
print(f"Visible GPUs: {torch.cuda.device_count()}")
print(f"Selected GPU: {torch.cuda.get_device_name(0)}")
PY

echo "[GPU before training]"
nvidia-smi
nvidia-smi -q > "${EXP_DIR}/gpu_environment.txt"
echo "[Result filesystem]"
df -h "${RESULT_ROOT}"

# shellcheck disable=SC2086  # ATTENTION_CASCADES is a space-separated int list
if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  echo "[INFO] Running a one-slice forward/backward smoke test (peak VRAM check)."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/smoke_test_training.py \
    --data-root "${DATA_ROOT}" \
    --gpu-num "${TORCH_GPU_NUM}" \
    --model-name "${MODEL_NAME}" \
    --cascade "${CASCADES}" \
    --image-cascades "${IMAGE_CASCADES}" \
    --chans "${CHANS}" \
    --sens-chans "${SENS_CHANS}" \
    --pools "${POOLS}" \
    --sens-pools "${SENS_POOLS}" \
    --attention-cascades ${ATTENTION_CASCADES} \
    --kspace-mult-factor "${KSPACE_MULT_FACTOR}" \
    --bbox-loss-weight "${BBOX_LOSS_WEIGHT}"
fi

if [[ ! -s "${GPU_LOG}" ]]; then
  echo "timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,temperature_c,power_draw_w" > "${GPU_LOG}"
fi
nvidia-smi \
  -i "${GPU_MONITOR_TARGET}" \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l "${GPU_SAMPLE_INTERVAL}" >> "${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
echo "GPU monitor PID: ${GPU_MONITOR_PID}"

# shellcheck disable=SC2086
TRAIN_ARGS=(
  python -u train.py
  -g "${TORCH_GPU_NUM}"
  -b "${BATCH_SIZE}"
  -e "${NUM_EPOCHS}"
  -l "${LEARNING_RATE}"
  -r "${REPORT_INTERVAL}"
  -n "${EXP_NAME}"
  -t "${TRAIN_DIR}"
  -v "${VAL_DIR}"
  --result-root "${RESULT_ROOT}"
  --model-name "${MODEL_NAME}"
  --cascade "${CASCADES}"
  --image-cascades "${IMAGE_CASCADES}"
  --chans "${CHANS}"
  --sens_chans "${SENS_CHANS}"
  --pools "${POOLS}"
  --sens-pools "${SENS_POOLS}"
  --attention-cascades ${ATTENTION_CASCADES}
  --kspace-mult-factor "${KSPACE_MULT_FACTOR}"
  --bbox-loss-weight "${BBOX_LOSS_WEIGHT}"
  --seed "${SEED}"
  --num-workers "${NUM_WORKERS}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --resume
)
if [[ "${PIN_MEMORY}" == "1" ]]; then
  TRAIN_ARGS+=(--pin-memory)
fi

echo "Training command: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${TRAIN_ARGS[*]}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED=1 "${TRAIN_ARGS[@]}"

echo "[INFO] Training completed; generating analysis artifacts."
python scripts/analyze_training.py \
  --exp-name "${EXP_NAME}" \
  --result-root "${RESULT_ROOT}" \
  --gpu-sample-interval "${GPU_SAMPLE_INTERVAL}"

if [[ "${RUN_VALIDATION_ANALYSIS}" == "1" ]]; then
  python scripts/evaluate_validation.py \
    --exp-name "${EXP_NAME}" \
    --data-root "${DATA_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --gpu-num "${TORCH_GPU_NUM}" || \
    echo "[WARN] Validation metric analysis failed; the training checkpoint is unaffected."
fi

date --iso-8601=seconds > "${EXP_DIR}/TRAINING_COMPLETED"
echo "============================================================"
echo "End time       : $(date --iso-8601=seconds)"
echo "Checkpoint dir : ${EXP_DIR}/checkpoints"
echo "Analysis dir   : ${ANALYSIS_DIR}"
echo "Training log   : ${TRAIN_LOG}"
echo "GPU log        : ${GPU_LOG}"
echo "============================================================"
