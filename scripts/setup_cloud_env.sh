#!/usr/bin/env bash
set -euo pipefail

# Stable cloud environment for Mask2Former training.
# Target stack:
#   Python 3.10
#   torch==2.2.2 + cu121
#   torchvision==0.17.2
#   torchaudio==2.2.2
#   detectron2 (from source)

VENV_DIR="${1:-.venv-mask2former}"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "ERROR: python3.10 not found. Please install Python 3.10 first."
  exit 1
fi

python3.10 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel ninja
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Detectron2 source install for best compatibility with current torch.
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Project dependencies
pip install -r requirements/server-cu121.txt

echo
echo "Environment setup completed in ${VENV_DIR}"
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Next step:"
echo "  bash scripts/build_msdeformattn.sh ${VENV_DIR}"
