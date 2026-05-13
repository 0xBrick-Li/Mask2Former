# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


def _get_hrc_whu_meta():
    return {
        "stuff_classes": ["background", "cloud"],
    }


def register_all_hrc_whu_sem_seg(root):
    root = os.path.join(root, "HRC_WHU")
    meta = _get_hrc_whu_meta()

    for split in ["train", "val"]:
        image_dir = os.path.join(root, "img_dir", split)
        gt_dir = os.path.join(root, "ann_dir", split)
        name = f"hrc_whu_sem_seg_{split}"
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png"),
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hrc_whu_sem_seg(_root)
