# Cloud Environment Setup (Stable Stack for Mask2Former)

This guide installs a stable training environment for this repository on NVIDIA CUDA servers.

## Recommended versions

- Python: `3.10`
- PyTorch: `2.2.2` (CUDA `12.1` runtime wheel)
- torchvision: `0.17.2`
- torchaudio: `2.2.2`
- Detectron2: install from source

## 1) Create and install environment

From repo root:

```bash
bash scripts/setup_cloud_env.sh .venv-mask2former
```

This script will:

1. Create a Python 3.10 venv
2. Install torch/vision/audio with cu121 wheels
3. Install detectron2 from source
4. Install project dependencies

## 2) Build Mask2Former CUDA extension

```bash
bash scripts/build_msdeformattn.sh .venv-mask2former
```

For headless build machines (no display GPU but with drivers), you can force CUDA build:

```bash
TORCH_CUDA_ARCH_LIST="<YOUR_GPU_ARCH>" FORCE_CUDA=1 bash scripts/build_msdeformattn.sh .venv-mask2former
```

Example architectures:

- A100: `8.0`
- V100: `7.0`
- RTX 3090: `8.6`

## 3) Quick sanity checks

```bash
source .venv-mask2former/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import detectron2; print('detectron2 ok')"
python -c "import mask2former; print('mask2former ok')"
```

## 4) Training pre-checks for HRC config

Ensure config file:

- `MODEL.WEIGHTS: checkpoints/model_final_f6e0f6.pkl`
- `MODEL.SEM_SEG_HEAD.NUM_CLASSES: 2`
- `MODEL.SWIN.FROZEN_STAGES: 5`

Export dataset root:

```bash
export DETECTRON2_DATASETS=/path/to/datasets_root
```

Then test config/data path with eval-only first:

```bash
python train_net.py \
  --config-file configs/hrc-whu/semantic-segmentation/swin/mask2former_swin_base_hrc_whu.yaml \
  --eval-only
```

## 5) Start training

```bash
python train_net.py \
  --config-file configs/hrc-whu/semantic-segmentation/swin/mask2former_swin_base_hrc_whu.yaml \
  --num-gpus 1
```
