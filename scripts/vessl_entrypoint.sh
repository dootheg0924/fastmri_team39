#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python -m pip install -r requirements.txt
fi

python - <<'PY'
required = ['torch', 'numpy', 'h5py', 'cv2', 'skimage', 'matplotlib']
missing = []
for name in required:
    try:
        __import__(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(
        '[ERROR] Missing Python packages: ' + ', '.join(missing)
        + '. Re-run with INSTALL_DEPS=1 or install requirements.txt in the image.'
    )
PY

: "${FINAL_STAGE_STOP_EPOCH:=89}"
export FINAL_STAGE_STOP_EPOCH

exec bash scripts/run_final_staged_reproduction.sh
