#!/usr/bin/env bash
# FI-VarNet experiment runner (bbox loss, optional FiLM/attention split).
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

# Keep the smoke test and resolved manifest identical to the atomic paper
# preset that train.py applies. Data paths/devices and the explicit knee-vs-
# brain protocol controls (COMBINE_TRAIN_VAL, CHECKPOINT_METRIC) remain
# runtime-selectable.
if [[ "${TRAINING_PRESET:-legacy}" == "fi-varnet-paper" ]]; then
  BATCH_SIZE=1
  LEARNING_RATE=3e-4
  SEED=42
  MODEL_NAME=fivarnet
  CASCADES=6
  IMAGE_CASCADES=6
  CHANS=32
  SENS_CHANS=8
  POOLS=4
  SENS_POOLS=4
  ATTENTION_CASCADES="0 1 2 3 4 5"
  KSPACE_MULT_FACTOR=1e6
  FEATURE_PROCESSOR=paper-unet2d
  NO_GRAD_CHECKPOINT=0
  GRADIENT_ACCUMULATION_STEPS=4
  DATA_SAMPLER_SEED=0
  CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-paper-final}"
  DETERMINISTIC=0
  FLOAT32_MATMUL_PRECISION=high
  BBOX_LOSS_WEIGHT=0.5
  ACC_FILM=0
  SPLIT_ATTENTION_CASCADES=
  BALANCE_ACCELERATIONS=0
  NUM_WORKERS=4
  PIN_MEMORY=0
fi
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
  printf 'MAX_TRAINING_EPOCHS=%q\n' "${MAX_TRAINING_EPOCHS:-}"
  printf 'LEARNING_RATE=%q\n' "${LEARNING_RATE}"
  printf 'REPORT_INTERVAL=%q\n' "${REPORT_INTERVAL}"
  printf 'SEED=%q\n' "${SEED}"
  printf 'TRAINING_PRESET=%q\n' "${TRAINING_PRESET:-legacy}"
  printf 'MODEL_NAME=%q\n' "${MODEL_NAME}"
  printf 'CASCADES=%q\n' "${CASCADES}"
  printf 'IMAGE_CASCADES=%q\n' "${IMAGE_CASCADES}"
  printf 'CHANS=%q\n' "${CHANS}"
  printf 'SENS_CHANS=%q\n' "${SENS_CHANS}"
  printf 'POOLS=%q\n' "${POOLS}"
  printf 'SENS_POOLS=%q\n' "${SENS_POOLS}"
  printf 'ATTENTION_CASCADES=%q\n' "${ATTENTION_CASCADES}"
  printf 'KSPACE_MULT_FACTOR=%q\n' "${KSPACE_MULT_FACTOR}"
  printf 'FEATURE_PROCESSOR=%q\n' "${FEATURE_PROCESSOR:-norm-unet}"
  printf 'NO_GRAD_CHECKPOINT=%q\n' "${NO_GRAD_CHECKPOINT:-0}"
  printf 'GRADIENT_ACCUMULATION_STEPS=%q\n' "${GRADIENT_ACCUMULATION_STEPS:-1}"
  printf 'DATA_SAMPLER_SEED=%q\n' "${DATA_SAMPLER_SEED:-}"
  printf 'COMBINE_TRAIN_VAL=%q\n' "${COMBINE_TRAIN_VAL:-0}"
  printf 'CHECKPOINT_METRIC=%q\n' "${CHECKPOINT_METRIC:-challenge-final}"
  printf 'DETERMINISTIC=%q\n' "${DETERMINISTIC:-1}"
  printf 'FLOAT32_MATMUL_PRECISION=%q\n' "${FLOAT32_MATMUL_PRECISION:-highest}"
  printf 'MRAUGMENT=%q\n' "${MRAUGMENT:-0}"
  printf 'MRAUGMENT_SCHEDULE=%q\n' "${MRAUGMENT_SCHEDULE:-exp}"
  printf 'MRAUGMENT_STRENGTH=%q\n' "${MRAUGMENT_STRENGTH:-0.55}"
  printf 'MRAUGMENT_EXP_DECAY=%q\n' "${MRAUGMENT_EXP_DECAY:-5.0}"
  printf 'MRAUGMENT_DELAY_EPOCHS=%q\n' "${MRAUGMENT_DELAY_EPOCHS:-0}"
  printf 'MRAUGMENT_SEED=%q\n' "${MRAUGMENT_SEED:-42}"
  printf 'MRAUGMENT_MIN_BBOX_SIZE=%q\n' "${MRAUGMENT_MIN_BBOX_SIZE:-7}"
  printf 'TRAINING_TIME_BUDGET_HOURS=%q\n' "${TRAINING_TIME_BUDGET_HOURS:-}"
  printf 'TRAINING_TIME_RESERVE_FRACTION=%q\n' "${TRAINING_TIME_RESERVE_FRACTION:-0.05}"
  printf 'TRAINING_TIME_PROBE_EPOCHS=%q\n' "${TRAINING_TIME_PROBE_EPOCHS:-2}"
  printf 'BBOX_LOSS_WEIGHT=%q\n' "${BBOX_LOSS_WEIGHT}"
  printf 'ACC_FILM=%q\n' "${ACC_FILM:-0}"
  printf 'SPLIT_ATTENTION_CASCADES=%q\n' "${SPLIT_ATTENTION_CASCADES:-}"
  printf 'WARM_START_CHECKPOINT=%q\n' "${WARM_START_CHECKPOINT:-}"
  printf 'EXPECTED_WARM_START_EPOCH=%q\n' "${EXPECTED_WARM_START_EPOCH:-}"
  printf 'ADDITIONAL_EPOCHS=%q\n' "${ADDITIONAL_EPOCHS:-}"
  printf 'BALANCE_ACCELERATIONS=%q\n' "${BALANCE_ACCELERATIONS:-0}"
  printf 'ACCELERATION_BALANCE_MODE=%q\n' "${ACCELERATION_BALANCE_MODE:-oversample}"
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

if [[ -n "${WARM_START_CHECKPOINT:-}" && ! -f "${WARM_START_CHECKPOINT}" ]]; then
  echo "[ERROR] Warm-start checkpoint not found: ${WARM_START_CHECKPOINT}" >&2
  exit 3
fi

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

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  echo "[INFO] Running a one-slice forward/backward smoke test (peak VRAM check)."
  # shellcheck disable=SC2206
  SMOKE_ATTENTION_CASCADES=(${ATTENTION_CASCADES})
  SMOKE_EXTRA_ARGS=()
  if [[ "${ACC_FILM:-0}" == "1" ]]; then
    SMOKE_EXTRA_ARGS+=(--acc-film)
  fi
  if [[ "${NO_GRAD_CHECKPOINT:-0}" == "1" ]]; then
    SMOKE_EXTRA_ARGS+=(--no-grad-checkpoint)
  fi
  if [[ "${TRAINING_PRESET:-legacy}" == "fi-varnet-paper" ]]; then
    SMOKE_EXTRA_ARGS+=(--paper-training)
  elif [[ "${TRAINING_PRESET:-legacy}" == "fi-varnet-final" ]]; then
    SMOKE_EXTRA_ARGS+=(--final-training)
  fi
  if [[ -n "${SPLIT_ATTENTION_CASCADES:-}" ]]; then
    # shellcheck disable=SC2206
    SMOKE_SPLIT_CASCADES=(${SPLIT_ATTENTION_CASCADES})
    SMOKE_EXTRA_ARGS+=(--split-attention-cascades "${SMOKE_SPLIT_CASCADES[@]}")
  fi
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
    --attention-cascades "${SMOKE_ATTENTION_CASCADES[@]}" \
    --kspace-mult-factor "${KSPACE_MULT_FACTOR}" \
    --feature-processor "${FEATURE_PROCESSOR:-norm-unet}" \
    --bbox-loss-weight "${BBOX_LOSS_WEIGHT}" \
    "${SMOKE_EXTRA_ARGS[@]}"
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

# shellcheck disable=SC2206
TRAIN_ATTENTION_CASCADES=(${ATTENTION_CASCADES})
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
  --training-preset "${TRAINING_PRESET:-legacy}"
  --model-name "${MODEL_NAME}"
  --cascade "${CASCADES}"
  --image-cascades "${IMAGE_CASCADES}"
  --chans "${CHANS}"
  --sens_chans "${SENS_CHANS}"
  --pools "${POOLS}"
  --sens-pools "${SENS_POOLS}"
  --attention-cascades "${TRAIN_ATTENTION_CASCADES[@]}"
  --kspace-mult-factor "${KSPACE_MULT_FACTOR}"
  --feature-processor "${FEATURE_PROCESSOR:-norm-unet}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --checkpoint-metric "${CHECKPOINT_METRIC:-challenge-final}"
  --float32-matmul-precision "${FLOAT32_MATMUL_PRECISION:-highest}"
  --bbox-loss-weight "${BBOX_LOSS_WEIGHT}"
  --seed "${SEED}"
  --num-workers "${NUM_WORKERS}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --resume
)
if [[ -n "${MAX_TRAINING_EPOCHS:-}" ]]; then
  TRAIN_ARGS+=(--max-training-epochs "${MAX_TRAINING_EPOCHS}")
fi
if [[ -n "${TRAINING_TIME_BUDGET_HOURS:-}" ]]; then
  TRAIN_ARGS+=(
    --training-time-budget-hours "${TRAINING_TIME_BUDGET_HOURS}"
    --training-time-reserve-fraction "${TRAINING_TIME_RESERVE_FRACTION:-0.05}"
    --training-time-probe-epochs "${TRAINING_TIME_PROBE_EPOCHS:-2}"
  )
fi
if [[ "${PIN_MEMORY}" == "1" ]]; then
  TRAIN_ARGS+=(--pin-memory)
fi
if [[ "${ACC_FILM:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--acc-film)
fi
if [[ "${NO_GRAD_CHECKPOINT:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-grad-checkpoint)
fi
if [[ -n "${DATA_SAMPLER_SEED:-}" ]]; then
  TRAIN_ARGS+=(--data-sampler-seed "${DATA_SAMPLER_SEED}")
fi
if [[ "${COMBINE_TRAIN_VAL:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--combine-train-val)
else
  TRAIN_ARGS+=(--no-combine-train-val)
fi
if [[ "${DETERMINISTIC:-1}" == "1" ]]; then
  TRAIN_ARGS+=(--deterministic)
else
  TRAIN_ARGS+=(--no-deterministic)
fi
if [[ -n "${SPLIT_ATTENTION_CASCADES:-}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_SPLIT_CASCADES=(${SPLIT_ATTENTION_CASCADES})
  TRAIN_ARGS+=(--split-attention-cascades "${TRAIN_SPLIT_CASCADES[@]}")
fi
if [[ -n "${WARM_START_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=(--warm-start-checkpoint "${WARM_START_CHECKPOINT}")
fi
if [[ -n "${EXPECTED_WARM_START_EPOCH:-}" ]]; then
  TRAIN_ARGS+=(--expected-warm-start-epoch "${EXPECTED_WARM_START_EPOCH}")
fi
if [[ -n "${ADDITIONAL_EPOCHS:-}" ]]; then
  TRAIN_ARGS+=(--additional-epochs "${ADDITIONAL_EPOCHS}")
fi
if [[ "${BALANCE_ACCELERATIONS:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    --balance-accelerations
    --acceleration-balance-mode "${ACCELERATION_BALANCE_MODE:-oversample}"
  )
fi
if [[ "${MRAUGMENT:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    --mraugment
    --mraugment-schedule "${MRAUGMENT_SCHEDULE:-exp}"
    --mraugment-strength "${MRAUGMENT_STRENGTH:-0.55}"
    --mraugment-exp-decay "${MRAUGMENT_EXP_DECAY:-5.0}"
    --mraugment-delay-epochs "${MRAUGMENT_DELAY_EPOCHS:-0}"
    --mraugment-seed "${MRAUGMENT_SEED:-42}"
    --mraugment-min-bbox-size "${MRAUGMENT_MIN_BBOX_SIZE:-7}"
  )
else
  TRAIN_ARGS+=(--no-mraugment)
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
