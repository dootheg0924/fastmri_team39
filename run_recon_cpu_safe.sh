#!/usr/bin/env bash
# Run CPU leaderboard evaluation conservatively beside an active VESSL GPU
# training job. This script never sends a signal to the training process.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_NAME="${EXP_NAME:-final_fivarnet_submission_f6i6_mraugment_cross_acc}"
LEADERBOARD_PATH="${LEADERBOARD_PATH:-/root/Data/leaderboard}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/../result}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_THREADS="${NUM_THREADS:-1}"
START_MIN_AVAILABLE_GIB="${START_MIN_AVAILABLE_GIB:-8}"
ABORT_MIN_AVAILABLE_GIB="${ABORT_MIN_AVAILABLE_GIB:-5}"
MEMORY_CHECK_SECONDS="${MEMORY_CHECK_SECONDS:-5}"
KEEP_SNAPSHOT="${KEEP_SNAPSHOT:-0}"

case "${NUM_THREADS}:${START_MIN_AVAILABLE_GIB}:${ABORT_MIN_AVAILABLE_GIB}:${MEMORY_CHECK_SECONDS}" in
  *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0)
    echo "[ERROR] Thread, memory, and interval settings must be positive integers." >&2
    exit 2
    ;;
esac
if [[ "${KEEP_SNAPSHOT}" != "0" && "${KEEP_SNAPSHOT}" != "1" ]]; then
  echo "[ERROR] KEEP_SNAPSHOT must be 0 or 1." >&2
  exit 2
fi
if (( ABORT_MIN_AVAILABLE_GIB >= START_MIN_AVAILABLE_GIB )); then
  echo "[ERROR] ABORT_MIN_AVAILABLE_GIB must be below START_MIN_AVAILABLE_GIB." >&2
  exit 2
fi

SOURCE_CHECKPOINT="${RESULT_ROOT}/${EXP_NAME}/checkpoints/best_model.pt"
LOG_DIR="${RESULT_ROOT}/${EXP_NAME}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${TIMESTAMP}_$$"
EVAL_LOG="${LOG_DIR}/recon_eval_cpuonly_${TIMESTAMP}.log"
SUPERVISOR_LOG="${LOG_DIR}/recon_eval_cpuonly_${TIMESTAMP}_supervisor.log"
SNAPSHOT_DIR="${LOG_DIR}/eval_cpu_snapshots/${RUN_ID}"
SNAPSHOT_CHECKPOINT="${SNAPSHOT_DIR}/best_model.pt"

available_kib() {
  awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo
}

format_available_gib() {
  awk -v kib="$1" 'BEGIN { printf "%.2f", kib / 1048576 }'
}

if [[ ! -f "${SCRIPT_DIR}/recon_eval_cpuonly.py" ]]; then
  echo "[ERROR] Missing ${SCRIPT_DIR}/recon_eval_cpuonly.py" >&2
  exit 3
fi
if [[ ! -s "${SOURCE_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing or empty checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 3
fi
if [[ ! -d "${LEADERBOARD_PATH}/acc4/image" \
   || ! -d "${LEADERBOARD_PATH}/acc4/kspace" \
   || ! -d "${LEADERBOARD_PATH}/acc8/image" \
   || ! -d "${LEADERBOARD_PATH}/acc8/kspace" ]]; then
  echo "[ERROR] Incomplete leaderboard data under ${LEADERBOARD_PATH}" >&2
  exit 3
fi

mkdir -p "${LOG_DIR}"

START_AVAILABLE_KIB="$(available_kib)"
START_REQUIRED_KIB="$((START_MIN_AVAILABLE_GIB * 1024 * 1024))"
if [[ -z "${START_AVAILABLE_KIB}" ]] || (( START_AVAILABLE_KIB < START_REQUIRED_KIB )); then
  echo "[ERROR] Refusing to start: MemAvailable=$(format_available_gib "${START_AVAILABLE_KIB:-0}") GiB; required=${START_MIN_AVAILABLE_GIB} GiB." >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${NUM_THREADS}"
export MKL_NUM_THREADS="${NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUM_THREADS}"

"${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/recon_eval_cpuonly.py"

# best_model.pt may be atomically replaced whenever validation improves. A
# same-filesystem hard link pins the exact inode selected at this instant and
# costs no extra model-sized disk space. Copying is a portable fallback.
SNAPSHOT_READY=0
cleanup_snapshot() {
  if [[ "${KEEP_SNAPSHOT}" == "0" || "${SNAPSHOT_READY}" == "0" ]]; then
    rm -f -- "${SNAPSHOT_CHECKPOINT}"
    rmdir -- "${SNAPSHOT_DIR}" 2>/dev/null || true
  fi
}
trap cleanup_snapshot EXIT

mkdir -p "${SNAPSHOT_DIR}"
if ln -- "${SOURCE_CHECKPOINT}" "${SNAPSHOT_CHECKPOINT}" 2>/dev/null; then
  SNAPSHOT_METHOD="hard-link"
else
  cp --reflink=auto -- "${SOURCE_CHECKPOINT}" "${SNAPSHOT_CHECKPOINT}"
  SNAPSHOT_METHOD="copy"
fi
if [[ ! -s "${SNAPSHOT_CHECKPOINT}" ]]; then
  echo "[ERROR] Failed to create checkpoint snapshot: ${SNAPSHOT_CHECKPOINT}" >&2
  exit 5
fi
SNAPSHOT_READY=1

RUN_PREFIX=(nice -n 19)
if command -v ionice >/dev/null 2>&1; then
  RUN_PREFIX+=(ionice -c 3)
fi

{
  echo "============================================================"
  echo "Safe CPU reconstruction supervisor"
  echo "Started           : $(date --iso-8601=seconds)"
  echo "Experiment        : ${EXP_NAME}"
  echo "Source checkpoint : ${SOURCE_CHECKPOINT}"
  echo "Snapshot          : ${SNAPSHOT_CHECKPOINT}"
  echo "Snapshot method   : ${SNAPSHOT_METHOD}"
  echo "Snapshot stat     : $(stat -c 'inode=%i size=%s mtime=%y' "${SNAPSHOT_CHECKPOINT}")"
  echo "Leaderboard       : ${LEADERBOARD_PATH}"
  echo "Threads           : ${NUM_THREADS}"
  echo "CPU priority      : nice 19"
  echo "I/O priority      : $([[ ${#RUN_PREFIX[@]} -gt 3 ]] && echo idle || echo default)"
  echo "MemAvailable      : $(format_available_gib "${START_AVAILABLE_KIB}") GiB"
  echo "RAM abort at      : ${ABORT_MIN_AVAILABLE_GIB} GiB"
  echo "Keep snapshot     : ${KEEP_SNAPSHOT}"
  echo "Evaluation log    : ${EVAL_LOG}"
  echo "============================================================"
} | tee -a "${SUPERVISOR_LOG}"

"${RUN_PREFIX[@]}" "${PYTHON_BIN}" -u "${SCRIPT_DIR}/recon_eval_cpuonly.py" \
  --checkpoint "${SNAPSHOT_CHECKPOINT}" \
  --path_data "${LEADERBOARD_PATH}" \
  --num_threads "${NUM_THREADS}" \
  > "${EVAL_LOG}" 2>&1 &
EVAL_PID=$!
echo "Evaluation PID: ${EVAL_PID}" | tee -a "${SUPERVISOR_LOG}"

terminate_eval() {
  if kill -0 "${EVAL_PID}" 2>/dev/null; then
    kill -TERM "${EVAL_PID}" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "${EVAL_PID}" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL "${EVAL_PID}" 2>/dev/null || true
  fi
}

on_interrupt() {
  echo "[WARN] Supervisor interrupted; terminating evaluation PID ${EVAL_PID} only." \
    | tee -a "${SUPERVISOR_LOG}"
  terminate_eval
  exit 130
}
trap on_interrupt INT TERM HUP

ABORT_REQUIRED_KIB="$((ABORT_MIN_AVAILABLE_GIB * 1024 * 1024))"
while kill -0 "${EVAL_PID}" 2>/dev/null; do
  sleep "${MEMORY_CHECK_SECONDS}"
  CURRENT_AVAILABLE_KIB="$(available_kib)"
  if [[ -n "${CURRENT_AVAILABLE_KIB}" ]] \
     && (( CURRENT_AVAILABLE_KIB < ABORT_REQUIRED_KIB )); then
    echo "[SAFETY STOP] MemAvailable=$(format_available_gib "${CURRENT_AVAILABLE_KIB}") GiB fell below ${ABORT_MIN_AVAILABLE_GIB} GiB. Terminating evaluation PID ${EVAL_PID}; training is untouched." \
      | tee -a "${SUPERVISOR_LOG}"
    terminate_eval
    wait "${EVAL_PID}" 2>/dev/null || true
    exit 75
  fi
done

set +e
wait "${EVAL_PID}"
STATUS=$?
set -e
trap - INT TERM HUP

echo "Evaluation exited with status ${STATUS} at $(date --iso-8601=seconds)." \
  | tee -a "${SUPERVISOR_LOG}"
echo "Evaluation log: ${EVAL_LOG}" | tee -a "${SUPERVISOR_LOG}"
if (( STATUS == 0 )); then
  tail -n 12 "${EVAL_LOG}"
else
  echo "[ERROR] Last 40 evaluation-log lines:" >&2
  tail -n 40 "${EVAL_LOG}" >&2
fi
exit "${STATUS}"
