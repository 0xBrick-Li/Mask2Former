#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config


def setup_cfg(config_file, weights, device):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Export semantic masks for HRC_WHU test set.")
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to config file, e.g. configs/hrc-whu/semantic-segmentation/swin/mask2former_swin_base_hrc_whu.yaml",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained weights, e.g. output/hrc_swinb_freeze_v1/model_final.pth",
    )
    parser.add_argument(
        "--dataset-name",
        default="hrc_whu_sem_seg_test",
        help="Registered dataset name.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save predicted masks.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--save-255",
        action="store_true",
        help="If set, save binary mask as 0/255. Otherwise save 0/1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or fix CUDA runtime.")

    cfg = setup_cfg(args.config_file, args.weights, args.device)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(args.dataset_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(dataset_dicts, desc=f"Exporting {args.dataset_name}"):
        image_path = item["file_name"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        outputs = predictor(image)
        pred = outputs["sem_seg"].argmax(dim=0).to("cpu").numpy().astype(np.uint8)
        if args.save_255:
            pred = (pred * 255).astype(np.uint8)

        filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        cv2.imwrite(str(output_dir / filename), pred)

    print(f"Saved predictions to: {output_dir}")


if __name__ == "__main__":
    main()
