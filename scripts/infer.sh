#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

die() { printf '[infer] error: %s\n' "$*" >&2; exit 1; }

[[ -x "${BREEZE_VENV}/bin/python" ]] || die "venv missing; run ${SCRIPT_DIR}/setup.sh"
[[ -f "${BREEZE_CODE}/infer.py" ]] || die "breeze-tts missing; run ${SCRIPT_DIR}/setup.sh"
[[ -f "${BREEZE_CKPT}/model.safetensors.index.json" ]] || die "checkpoint missing; run ${SCRIPT_DIR}/setup.sh"

# shellcheck disable=SC1091
source "${BREEZE_VENV}/bin/activate"
cd "${BREEZE_CODE}"
mkdir -p "${BREEZE_ROOT}/outputs"

if [[ $# -eq 0 ]]; then
  set -- \
    --text "Hello from Strix Halo." \
    --output "${BREEZE_ROOT}/outputs/smoke.wav"
fi

exec python infer.py "${BREEZE_CKPT}" "$@"
