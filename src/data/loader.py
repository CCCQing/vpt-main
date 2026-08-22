#!/usr/bin/env python3

"""XLSA-only data loader helpers."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from ..utils import logging
from ..utils import distributed as du
from ..utils.reproducibility import derive_seed, make_torch_generator, seed_data_worker
from .datasets.xlsa_dataset import CUB200Dataset, AWA2Dataset, SUNAttributeDataset
from .transforms import get_transforms

logger = logging.get_logger("visual_prompt")

_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    "AWA2": AWA2Dataset,
    "SUN": SUNAttributeDataset,
}


def _construct_loader(
    cfg,
    split,
    batch_size,
    shuffle,
    drop_last,
    *,
    transform_split=None,
    eval_class_ids=None,
):
    """Build a DataLoader for one XLSA protocol split."""
    if not bool(cfg.DATA.XLSA.ENABLED):
        raise ValueError("Current data pipeline is XLSA-only. Please set DATA.XLSA.ENABLED=True.")

    dataset_name = cfg.DATA.NAME
    if dataset_name not in _DATASET_CATALOG:
        raise ValueError("Dataset '{}' not supported".format(dataset_name))

    dataset = _DATASET_CATALOG[dataset_name](cfg, split)
    if transform_split is not None:
        dataset.transform = get_transforms(str(transform_split), cfg.DATA.CROPSIZE)
    if eval_class_ids is not None:
        if str(eval_class_ids) == "seen_unseen":
            eval_class_ids = list(dataset.seen_classes) + list(dataset.unseen_classes)
        dataset.eval_local_classes = [int(item) for item in list(eval_class_ids)]
        dataset.eval_global_to_local = dataset._build_global_to_local(
            dataset.num_classes, dataset.eval_local_classes
        )
    world_size = du.get_world_size()
    rank = du.get_rank()
    data_order_seed = derive_seed(cfg.SEED, "data_order")
    sampler = None
    if world_size > 1 and shuffle:
        if data_order_seed is None:
            raise ValueError("Distributed training requires cfg.SEED so DistributedSampler has a stable seed.")
        if len(dataset) % world_size != 0:
            raise ValueError(
                "Training dataset size={} must be divisible by actual world_size={} so DDP ranks "
                "can use non-overlapping, complete sample partitions without padding or omission.".format(
                    len(dataset), world_size
                )
            )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(data_order_seed),
            drop_last=bool(drop_last),
        )
    loader_generator = make_torch_generator(data_order_seed)
    if shuffle:
        logger.info(
            "[reproducibility] data_order split=%s seed=%s workers=%d rank=%d world_size=%d sampler=%s worker_init=%s",
            split,
            str(data_order_seed),
            int(cfg.DATA.NUM_WORKERS),
            rank,
            world_size,
            sampler.__class__.__name__ if sampler is not None else "RandomSampler",
            "seed_data_worker" if int(cfg.DATA.NUM_WORKERS) > 0 else "none",
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
        generator=loader_generator,
        worker_init_fn=seed_data_worker if int(cfg.DATA.NUM_WORKERS) > 0 else None,
    )


def _per_rank_batch_size(cfg):
    """Interpret DATA.BATCH_SIZE as a global batch size in single- and multi-node runs."""
    world_size = du.get_world_size()
    global_batch_size = int(cfg.DATA.BATCH_SIZE)
    if global_batch_size < world_size or global_batch_size % world_size != 0:
        raise ValueError(
            "DATA.BATCH_SIZE={} must be a positive multiple of actual world_size={} "
            "because DATA.BATCH_SIZE is the global batch size.".format(global_batch_size, world_size)
        )
    return global_batch_size // world_size


def construct_train_loader(cfg):
    """Build the `train` loader."""
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=_per_rank_batch_size(cfg),
        shuffle=True,
        drop_last=False,
    )


def construct_trainval_loader(cfg):
    """Build the `trainval` loader."""
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=_per_rank_batch_size(cfg),
        shuffle=True,
        drop_last=False,
    )


def construct_train_eval_loader(cfg):
    if str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() not in {
        "final_gzsl",
        "b3_pseudo_gzsl",
    }:
        raise ValueError("train-eval monitoring is only defined for GZSL protocols")
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=_per_rank_batch_size(cfg),
        shuffle=False,
        drop_last=False,
        transform_split="test_seen",
        eval_class_ids="seen_unseen",
    )


def construct_val_loader(cfg, batch_size=None):
    """Build the `val_unseen` loader."""
    bs = _per_rank_batch_size(cfg) if batch_size is None else batch_size
    return _construct_loader(
        cfg=cfg,
        split="val_unseen",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def construct_test_seen_loader(cfg):
    """Build the `test_seen` loader."""
    return _construct_loader(
        cfg=cfg,
        split="test_seen",
        batch_size=_per_rank_batch_size(cfg),
        shuffle=False,
        drop_last=False,
    )


def construct_test_unseen_loader(cfg):
    """Build the `test_unseen` loader."""
    return _construct_loader(
        cfg=cfg,
        split="test_unseen",
        batch_size=_per_rank_batch_size(cfg),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """Advance the distributed sampler epoch if needed."""
    if not isinstance(loader.sampler, (RandomSampler, DistributedSampler)):
        raise TypeError("Sampler type '{}' not supported".format(type(loader.sampler)))
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(cur_epoch)
