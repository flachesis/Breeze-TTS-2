#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] error: %s\n' "$*" >&2; exit 1; }

pick_python() {
  local candidate
  for candidate in "${PYTHON:-}" python3.12 python3.13 python3.11 python3; do
    [[ -n "${candidate}" ]] || continue
    command -v "${candidate}" >/dev/null 2>&1 || continue
    if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)" || die "need Python 3.11+ (prefer 3.12). Set PYTHON=..."
log "python: ${PYTHON_BIN} ($("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

if command -v rocminfo >/dev/null 2>&1; then
  log "rocminfo: $(command -v rocminfo)"
elif command -v amd-smi >/dev/null 2>&1; then
  log "amd-smi: $(command -v amd-smi)"
else
  log "warning: rocminfo/amd-smi not found; continuing"
fi

if [[ ! -d "${BREEZE_VENV}" ]]; then
  log "creating venv ${BREEZE_VENV}"
  "${PYTHON_BIN}" -m venv "${BREEZE_VENV}"
fi
# shellcheck disable=SC1091
source "${BREEZE_VENV}/bin/activate"
python -m pip install -U pip setuptools wheel

log "installing ROCm PyTorch from ${AMD_TORCH_INDEX}"
python -m pip install --index-url "${AMD_TORCH_INDEX}" \
  "${AMD_TORCH}" \
  "${AMD_TORCHVISION}" \
  "${AMD_TORCHAUDIO}"

log "installing Breeze deps without replacing ROCm torch"
python -m pip install \
  --upgrade-strategy only-if-needed \
  --extra-index-url "${AMD_TORCH_INDEX}" \
  -c "${BREEZE_ROOT}/constraints-rocm.txt" \
  -r "${BREEZE_ROOT}/requirements-rocm.txt"

python - <<'PY'
import torch
hip = getattr(torch.version, "hip", None)
ok = bool(torch.cuda.is_available() and hip)
print(f"torch={torch.__version__} hip={hip} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device=0 {torch.cuda.get_device_name(0)}")
if not ok:
    raise SystemExit("ROCm torch is missing or GPU not visible")
if "+rocm" not in torch.__version__ and not hip:
    raise SystemExit(f"expected ROCm torch, got {torch.__version__}")
PY

if [[ ! -d "${BREEZE_CODE}/.git" ]]; then
  log "cloning ${BREEZE_REPO} -> ${BREEZE_CODE}"
  git clone --depth 1 "${BREEZE_REPO}" "${BREEZE_CODE}"
else
  log "breeze-tts already present: ${BREEZE_CODE}"
fi

if ! python "${SCRIPT_DIR}/patch_breeze.py" "${BREEZE_CODE}"; then
  log "python patcher failed; applying git patch"
  git -C "${BREEZE_CODE}" apply "${BREEZE_ROOT}/patches/flash-attn-fallback.patch"
fi
grep -q "_flash_attn_2_available" "${BREEZE_CODE}/models/breeze.py" \
  || die "failed to patch flash_attn fallback"

mkdir -p "${BREEZE_CKPT}"
if [[ ! -f "${BREEZE_CKPT}/model.safetensors.index.json" ]]; then
  log "downloading ${BREEZE_MODEL_ID} -> ${BREEZE_CKPT}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${BREEZE_MODEL_ID}", local_dir="${BREEZE_CKPT}")
PY
else
  log "checkpoint already present: ${BREEZE_CKPT}"
fi
[[ -d "${BREEZE_CKPT}/audio_tokenizer" ]] || die "missing ${BREEZE_CKPT}/audio_tokenizer"

mkdir -p "${BREEZE_ROOT}/outputs"
log "done. start API with: ${SCRIPT_DIR}/serve.sh"
log "smoke: ${SCRIPT_DIR}/infer.sh"
