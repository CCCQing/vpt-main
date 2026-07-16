#!/usr/bin/env python3

"""XLSA-only data loader helpers."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from ..utils import logging
from ..utils.reproducibility import derive_seed, make_torch_generator
from .datasets.xlsa_dataset import CUB200Dataset, AWA2Dataset, SUNAttributeDataset

logger = logging.get_logger("visual_prompt")

_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    "AWA2": AWA2Dataset,
    "SUN": SUNAttributeDataset,
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Build a DataLoader for one XLSA protocol split."""
    if not bool(cfg.DATA.XLSA.ENABLED):
        raise ValueError("Current data pipeline is XLSA-only. Please set DATA.XLSA.ENABLED=True.")

    dataset_name = cfg.DATA.NAME
    if dataset_name not in _DATASET_CATALOG:
        raise ValueError("Dataset '{}' not supported".format(dataset_name))

    dataset = _DATASET_CATALOG[dataset_name](cfg, split)
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    data_order_seed = derive_seed(cfg.SEED, "data_order")
    loader_generator = make_torch_generator(data_order_seed) if shuffle else None
    if shuffle:
        logger.info(
            "[reproducibility] data_order split=%s seed=%s workers=%d",
            split,
            str(data_order_seed),
            int(cfg.DATA.NUM_WORKERS),
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
    )


def construct_train_loader(cfg):
    """Build the `train` loader."""
    drop_last = bool(cfg.NUM_GPUS > 1)
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_trainval_loader(cfg):
    """Build the `trainval` loader."""
    drop_last = bool(cfg.NUM_GPUS > 1)
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_val_loader(cfg, batch_size=None):
    """Build the `val_unseen` loader."""
    bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS) if batch_size is None else batch_size
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
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_test_unseen_loader(cfg):
    """Build the `test_unseen` loader."""
    return _construct_loader(
        cfg=cfg,
        split="test_unseen",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """Advance the distributed sampler epoch if needed."""
    if not isinstance(loader.sampler, (RandomSampler, DistributedSampler)):
        raise TypeError("Sampler type '{}' not supported".format(type(loader.sampler)))
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(cur_epoch)
