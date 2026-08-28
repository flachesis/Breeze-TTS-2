#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] error: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"

if command -v rocminfo >/dev/null 2>&1; then
  log "rocminfo: $(command -v rocminfo)"
elif command -v amd-smi >/dev/null 2>&1; then
  log "amd-smi: $(command -v amd-smi)"
else
  log "warning: rocminfo/amd-smi not found; continuing"
fi

if [[ ! -x "${BREEZE_VENV}/bin/python" ]]; then
  log "creating venv ${BREEZE_VENV} (python ${UV_PYTHON})"
  uv venv --python "${UV_PYTHON}" "${BREEZE_VENV}"
fi
VENV_PY="${BREEZE_VENV}/bin/python"
log "python: ${VENV_PY} ($("${VENV_PY}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

log "installing ROCm PyTorch from ${AMD_TORCH_INDEX}"
uv pip install --python "${VENV_PY}" --index-url "${AMD_TORCH_INDEX}" \
  "${AMD_TORCH}" \
  "${AMD_TORCHVISION}" \
  "${AMD_TORCHAUDIO}"

log "installing Breeze deps without replacing ROCm torch"
uv pip install --python "${VENV_PY}" \
  --index-url https://pypi.org/simple \
  --extra-index-url "${AMD_TORCH_INDEX}" \
  --index-strategy unsafe-best-match \
  -c "${BREEZE_ROOT}/constraints-rocm.txt" \
  -r "${BREEZE_ROOT}/requirements-rocm.txt"

"${VENV_PY}" - <<'PY'
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

if ! "${VENV_PY}" "${SCRIPT_DIR}/patch_breeze.py" "${BREEZE_CODE}"; then
  log "python patcher failed; applying git patch"
  git -C "${BREEZE_CODE}" apply "${BREEZE_ROOT}/patches/flash-attn-fallback.patch"
fi
grep -q "_flash_attn_2_available" "${BREEZE_CODE}/models/breeze.py" \
  || die "failed to patch flash_attn fallback"

mkdir -p "${BREEZE_CKPT}"
if [[ ! -f "${BREEZE_CKPT}/model.safetensors.index.json" ]]; then
  log "downloading ${BREEZE_MODEL_ID} -> ${BREEZE_CKPT}"
  "${VENV_PY}" - <<PY
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
