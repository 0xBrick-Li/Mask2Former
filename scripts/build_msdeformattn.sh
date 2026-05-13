#!/usr/bin/env bash
set -euo pipefail

# Build CUDA extension for MSDeformAttn.
# Usage:
#   bash scripts/build_msdeformattn.sh .venv-mask2former
# Optional for headless build machine:
#   TORCH_CUDA_ARCH_LIST="8.0" FORCE_CUDA=1 bash scripts/build_msdeformattn.sh .venv-mask2former

VENV_DIR="${1:-.venv-mask2former}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: venv directory not found: ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

pushd mask2former/modeling/pixel_decoder/ops >/dev/null
sh make.sh
popd >/dev/null

echo "MSDeformAttn build completed."
