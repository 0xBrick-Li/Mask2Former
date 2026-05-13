#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import copy
import datetime
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import default_setup, default_writers, launch
from detectron2.evaluation import DatasetEvaluators, SemSegEvaluator, inference_on_dataset, print_csv_format
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.events import EventStorage
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

# Ensure repo root is importable when running from tools/ directly.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CUR_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mask2former import add_maskformer2_config
from train_net import Trainer as BaseTrainer


def add_distill_config(cfg):
    cfg.DISTILL = CN()
    cfg.DISTILL.ENABLED = True
    cfg.DISTILL.TEACHER_WEIGHTS = ""
    cfg.DISTILL.LOGITS_ROOT = ""
    cfg.DISTILL.MANIFEST_PATH = ""
    cfg.DISTILL.TEMPERATURE = 1.0
    cfg.DISTILL.KD_WEIGHT = 1.0
    cfg.DISTILL.SAVE_DTYPE = "float16"  # float16 or float32
    cfg.DISTILL.DATASET_SPLITS = ["train", "val"]
    cfg.DISTILL.NO_RANDOM_AUG = True


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_distill_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def _stable_id(file_name: str) -> str:
    return hashlib.sha1(file_name.encode("utf-8")).hexdigest()


def _dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported DISTILL.SAVE_DTYPE={name}")


class DeterministicSemanticTrainMapper:
    def __init__(self, cfg):
        self.img_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.INPUT.SIZE_DIVISIBILITY
        self.fixed_short_edge = cfg.INPUT.MIN_SIZE_TRAIN
        if isinstance(self.fixed_short_edge, (list, tuple)):
            self.fixed_short_edge = int(self.fixed_short_edge[0])
        self.ignore_label = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).ignore_label
        self.resize_aug = T.ResizeShortestEdge(
            [self.fixed_short_edge, self.fixed_short_edge],
            cfg.INPUT.MAX_SIZE_TRAIN,
            "choice",
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, _ = T.apply_transform_gens([self.resize_aug], aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            h, w = image.shape[-2:]
            pad_h = (self.size_divisibility - h % self.size_divisibility) % self.size_divisibility
            pad_w = (self.size_divisibility - w % self.size_divisibility) % self.size_divisibility
            padding_size = [0, pad_w, 0, pad_h]
            image = F.pad(image, padding_size, value=128).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])
        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt.long()
        dataset_dict["height"] = int(image_shape[0])
        dataset_dict["width"] = int(image_shape[1])
        dataset_dict["distill_id"] = _stable_id(dataset_dict["file_name"])

        classes = np.unique(sem_seg_gt.numpy())
        classes = classes[classes != self.ignore_label]
        instances = Instances(image_shape)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        masks = []
        for class_id in classes:
            masks.append(sem_seg_gt.numpy() == class_id)
        if len(masks) == 0:
            instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
        dataset_dict["instances"] = instances
        return dataset_dict


class DeterministicSemanticEvalMapper:
    def __init__(self, cfg):
        self.img_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.INPUT.SIZE_DIVISIBILITY
        self.fixed_short_edge = cfg.INPUT.MIN_SIZE_TEST
        if isinstance(self.fixed_short_edge, (list, tuple)):
            self.fixed_short_edge = int(self.fixed_short_edge[0])
        self.resize_aug = T.ResizeShortestEdge(
            [self.fixed_short_edge, self.fixed_short_edge],
            cfg.INPUT.MAX_SIZE_TEST,
            "choice",
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        aug_input = T.AugInput(image)
        aug_input, _ = T.apply_transform_gens([self.resize_aug], aug_input)
        image = aug_input.image
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.size_divisibility > 0:
            h, w = image.shape[-2:]
            pad_h = (self.size_divisibility - h % self.size_divisibility) % self.size_divisibility
            pad_w = (self.size_divisibility - w % self.size_divisibility) % self.size_divisibility
            padding_size = [0, pad_w, 0, pad_h]
            image = F.pad(image, padding_size, value=128).contiguous()

        dataset_dict["image"] = image
        dataset_dict["height"] = int(image.shape[-2])
        dataset_dict["width"] = int(image.shape[-1])
        dataset_dict["distill_id"] = _stable_id(dataset_dict["file_name"])
        return dataset_dict


@dataclass
class ManifestRecord:
    distill_id: str
    file_name: str
    split: str
    logits_relpath: str
    channels: int
    height: int
    width: int
    dtype: str


def _build_semseg_evaluator(cfg, dataset_name):
    out = os.path.join(cfg.OUTPUT_DIR, "inference")
    return DatasetEvaluators([SemSegEvaluator(dataset_name, distributed=False, output_dir=out)])


def do_export_logits(cfg, max_samples: int = 0):
    logger = logging.getLogger("mask2former.offline_kd")
    logits_root = cfg.DISTILL.LOGITS_ROOT
    if not logits_root:
        raise ValueError("DISTILL.LOGITS_ROOT must be set")
    PathManager.mkdirs(logits_root)
    manifest_path = cfg.DISTILL.MANIFEST_PATH or os.path.join(logits_root, "manifest.json")

    model = BaseTrainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.DISTILL.TEACHER_WEIGHTS, resume=False)
    model.eval()

    save_dtype = _dtype_from_name(cfg.DISTILL.SAVE_DTYPE)
    records: List[ManifestRecord] = []
    summary = {"splits": {}, "created_at": datetime.datetime.now().isoformat(), "logits_root": logits_root}

    split_map = {"train": cfg.DATASETS.TRAIN, "val": cfg.DATASETS.TEST}
    for split in cfg.DISTILL.DATASET_SPLITS:
        if split not in split_map:
            raise ValueError(f"Unsupported split '{split}', expected one of {list(split_map.keys())}")
        split_datasets = split_map[split]
        split_start = time.time()
        ok_count, fail_count, bytes_written = 0, 0, 0

        for dataset_name in split_datasets:
            loader = build_detection_test_loader(cfg, dataset_name, mapper=DeterministicSemanticEvalMapper(cfg))
            for idx, batch in enumerate(loader):
                if max_samples > 0 and ok_count >= max_samples:
                    break
                with torch.no_grad():
                    outputs = model(batch)
                for sample, output in zip(batch, outputs):
                    try:
                        sem_seg = output["sem_seg"].detach().to("cpu")
                        sem_seg = sem_seg.to(save_dtype)
                        distill_id = sample["distill_id"]
                        relpath = os.path.join(split, f"{distill_id}.pt")
                        abspath = os.path.join(logits_root, relpath)
                        PathManager.mkdirs(os.path.dirname(abspath))
                        torch.save({"sem_seg": sem_seg, "file_name": sample["file_name"]}, abspath)
                        file_size = os.path.getsize(abspath)
                        bytes_written += file_size
                        ok_count += 1
                        c, h, w = sem_seg.shape
                        records.append(
                            ManifestRecord(
                                distill_id=distill_id,
                                file_name=sample["file_name"],
                                split=split,
                                logits_relpath=relpath,
                                channels=int(c),
                                height=int(h),
                                width=int(w),
                                dtype=str(save_dtype).split(".")[-1],
                            )
                        )
                    except Exception:
                        fail_count += 1
                        logger.exception("Failed exporting logits for %s", sample.get("file_name", "<unknown>"))
                if max_samples > 0 and ok_count >= max_samples:
                    break

        elapsed = time.time() - split_start
        avg_bytes = bytes_written / max(ok_count, 1)
        summary["splits"][split] = {
            "ok": ok_count,
            "fail": fail_count,
            "elapsed_sec": elapsed,
            "total_bytes": bytes_written,
            "avg_bytes_per_sample": avg_bytes,
        }
        logger.info(
            "Exported split=%s ok=%d fail=%d elapsed=%.2fs total=%.2fMB",
            split,
            ok_count,
            fail_count,
            elapsed,
            bytes_written / (1024 * 1024),
        )

    manifest = {
        "meta": {
            "created_at": summary["created_at"],
            "logits_root": logits_root,
            "save_dtype": cfg.DISTILL.SAVE_DTYPE,
        },
        "records": [r.__dict__ for r in records],
        "summary": summary["splits"],
    }
    with PathManager.open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    with PathManager.open(os.path.join(cfg.OUTPUT_DIR, "distill_summary.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s (records=%d)", manifest_path, len(records))


def _prepare_manifest_lookup(cfg) -> Dict[str, Dict]:
    manifest_path = cfg.DISTILL.MANIFEST_PATH or os.path.join(cfg.DISTILL.LOGITS_ROOT, "manifest.json")
    with PathManager.open(manifest_path, "r") as f:
        manifest = json.load(f)
    lookup = {}
    for item in manifest["records"]:
        lookup[item["distill_id"]] = item
    return lookup


def _compute_student_forward(base_model, batched_inputs):
    images = [x["image"].to(base_model.device) for x in batched_inputs]
    images = [(x - base_model.pixel_mean) / base_model.pixel_std for x in images]
    images = ImageList.from_tensors(images, base_model.size_divisibility)

    features = base_model.backbone(images.tensor)
    outputs = base_model.sem_seg_head(features)

    if "instances" in batched_inputs[0]:
        gt_instances = [x["instances"].to(base_model.device) for x in batched_inputs]
        targets = base_model.prepare_targets(gt_instances, images)
    else:
        targets = None

    losses = base_model.criterion(outputs, targets)
    weighted_losses = {}
    for k, v in losses.items():
        if k in base_model.criterion.weight_dict:
            weighted_losses[k] = v * base_model.criterion.weight_dict[k]

    mask_cls_results = outputs["pred_logits"]
    mask_pred_results = outputs["pred_masks"]
    mask_pred_results = F.interpolate(
        mask_pred_results,
        size=(images.tensor.shape[-2], images.tensor.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )

    semsegs = []
    for mask_cls, mask_pred, input_per_image, image_size in zip(
        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
    ):
        r = base_model.semantic_inference(mask_cls, mask_pred)
        r = r[:, : image_size[0], : image_size[1]]
        h = input_per_image.get("height", image_size[0])
        w = input_per_image.get("width", image_size[1])
        if r.shape[-2:] != (h, w):
            r = F.interpolate(r.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        semsegs.append(r)

    return weighted_losses, semsegs


def _build_optimizer(cfg, model):
    # Reuse project optimizer behavior to stay consistent.
    return BaseTrainer.build_optimizer(cfg, model)


def _build_scheduler(cfg, optimizer):
    return build_lr_scheduler(cfg, optimizer)


def _do_eval(cfg, model):
    results = {}
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=DeterministicSemanticEvalMapper(cfg))
        evaluator = _build_semseg_evaluator(cfg, dataset_name)
        res = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = res
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train_kd(cfg, resume=False):
    logger = logging.getLogger("mask2former.offline_kd")
    model = BaseTrainer.build_model(cfg)
    model.train()
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    manifest_lookup = _prepare_manifest_lookup(cfg)
    logits_root = cfg.DISTILL.LOGITS_ROOT
    temperature = float(cfg.DISTILL.TEMPERATURE)
    kd_weight = float(cfg.DISTILL.KD_WEIGHT)

    train_loader = build_detection_train_loader(cfg, mapper=DeterministicSemanticTrainMapper(cfg))
    data_iter = iter(train_loader)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter)

    logger.info(
        "Starting KD train from iter=%d max_iter=%d temperature=%.3f kd_weight=%.3f",
        start_iter,
        max_iter,
        temperature,
        kd_weight,
    )

    with EventStorage(start_iter) as storage:
        for iteration in range(start_iter, max_iter):
            storage.iter = iteration
            iter_start = time.perf_counter()
            data = next(data_iter)
            data_time = time.perf_counter() - iter_start

            base_model = model.module if hasattr(model, "module") else model
            with autocast(enabled=cfg.SOLVER.AMP.ENABLED):
                sup_loss_dict, student_semsegs = _compute_student_forward(base_model, data)
                loss_sup = sum(sup_loss_dict.values())

                loss_kd_list = []
                teacher_mean, teacher_std = [], []
                student_mean, student_std = [], []
                for sample, student_semseg in zip(data, student_semsegs):
                    distill_id = sample["distill_id"]
                    if distill_id not in manifest_lookup:
                        raise KeyError(f"distill_id={distill_id} not found in manifest")
                    rec = manifest_lookup[distill_id]
                    logits_path = os.path.join(logits_root, rec["logits_relpath"])
                    teacher_pack = torch.load(logits_path, map_location=student_semseg.device)
                    teacher_semseg = teacher_pack["sem_seg"].to(student_semseg.device).float()

                    if teacher_semseg.shape != student_semseg.shape:
                        raise ValueError(
                            f"Teacher/student shape mismatch for {sample['file_name']}: "
                            f"{teacher_semseg.shape} vs {student_semseg.shape}"
                        )

                    t_prob = F.softmax(teacher_semseg / temperature, dim=0)
                    s_log_prob = F.log_softmax(student_semseg / temperature, dim=0)
                    kl = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=0).mean()
                    loss_kd_list.append(kl * (temperature * temperature))

                    teacher_mean.append(float(teacher_semseg.mean().detach().cpu()))
                    teacher_std.append(float(teacher_semseg.std().detach().cpu()))
                    student_mean.append(float(student_semseg.mean().detach().cpu()))
                    student_std.append(float(student_semseg.std().detach().cpu()))

                loss_kd = torch.stack(loss_kd_list).mean() if loss_kd_list else torch.tensor(0.0, device=loss_sup.device)
                loss_total = loss_sup + kd_weight * loss_kd

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            iter_time = time.perf_counter() - iter_start
            metrics = {
                "total_loss": float(loss_total.detach().cpu()),
                "loss_sup": float(loss_sup.detach().cpu()),
                "loss_kd": float(loss_kd.detach().cpu()),
                "kd_weight": kd_weight,
                "temperature": temperature,
                "data_time": data_time,
                "iter_time": iter_time,
                "teacher_logit_mean": float(np.mean(teacher_mean)) if teacher_mean else 0.0,
                "teacher_logit_std": float(np.mean(teacher_std)) if teacher_std else 0.0,
                "student_logit_mean": float(np.mean(student_mean)) if student_mean else 0.0,
                "student_logit_std": float(np.mean(student_std)) if student_std else 0.0,
                "kl_raw": float(loss_kd.detach().cpu()),
            }
            for k, v in metrics.items():
                storage.put_scalar(k, v)
            for k, v in sup_loss_dict.items():
                storage.put_scalar(k, float(v.detach().cpu()))
            if torch.cuda.is_available():
                storage.put_scalar("max_mem", torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

            periodic_checkpointer.step(iteration)

            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and (iteration + 1) < max_iter:
                eval_results = _do_eval(cfg, model)
                flat = flatten_results_dict(eval_results)
                for k, v in flat.items():
                    storage.put_scalar(f"eval/{k}", float(v), smoothing_hint=False)
                comm.synchronize()

            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()

    final_results = _do_eval(cfg, model)
    print_csv_format(final_results)
    return final_results


def do_evaluate(cfg):
    model = BaseTrainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()
    results = _do_eval(cfg, model)
    print_csv_format(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Offline semseg KD for Mask2Former")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_export = subparsers.add_parser("export_logits")
    p_export.add_argument("--config-file", required=True)
    p_export.add_argument("--num-gpus", type=int, default=1)
    p_export.add_argument("--num-machines", type=int, default=1)
    p_export.add_argument("--machine-rank", type=int, default=0)
    p_export.add_argument("--dist-url", default="tcp://127.0.0.1:50152")
    p_export.add_argument("--max-samples", type=int, default=0)
    p_export.add_argument("opts", nargs=argparse.REMAINDER)

    p_train = subparsers.add_parser("train_kd")
    p_train.add_argument("--config-file", required=True)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--num-gpus", type=int, default=1)
    p_train.add_argument("--num-machines", type=int, default=1)
    p_train.add_argument("--machine-rank", type=int, default=0)
    p_train.add_argument("--dist-url", default="tcp://127.0.0.1:50152")
    p_train.add_argument("opts", nargs=argparse.REMAINDER)

    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--config-file", required=True)
    p_eval.add_argument("--num-gpus", type=int, default=1)
    p_eval.add_argument("--num-machines", type=int, default=1)
    p_eval.add_argument("--machine-rank", type=int, default=0)
    p_eval.add_argument("--dist-url", default="tcp://127.0.0.1:50152")
    p_eval.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _main_worker(args):
    cfg = setup(args)
    if cfg.DISTILL.NO_RANDOM_AUG:
        logger = logging.getLogger("mask2former.offline_kd")
        logger.info("DISTILL.NO_RANDOM_AUG=True, using deterministic mappers (no crop/flip/color aug).")
    if args.command == "export_logits":
        do_export_logits(cfg, max_samples=args.max_samples)
        return
    if args.command == "train_kd":
        do_train_kd(cfg, resume=args.resume)
        return
    if args.command == "evaluate":
        do_evaluate(cfg)
        return
    raise ValueError(f"Unknown command {args.command}")


def main():
    args = parse_args()
    launch(
        _main_worker,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
