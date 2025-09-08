#!/usr/bin/env python3

"""Data loader.
数据加载器：负责根据配置构建训练/验证/测试用的 PyTorch DataLoader，
同时在多进程/多卡训练时启用分布式采样器（DistributedSampler）。"""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from ..utils import logging
from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
)
# 日志器（项目统一的 logger），方便在控制台/文件中记录信息
logger = logging.get_logger("visual_prompt")
# 数据集注册表：将配置里的名字映射到实际的数据集类
# 这样可以通过配置切换数据集，而无需改代码
_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset.
        构建一个给定 split（train/val/test/...）的数据加载器（DataLoader）。
    参数:
        cfg         : 配置对象（包含数据集名、线程数、是否 pin_memory、GPU 数等）
        split       : 数据划分标识，常见有 'train' / 'val' / 'test' / 'trainval'
        batch_size  : 本次 DataLoader 使用的 batch 大小（注意：通常为总 batch_size / NUM_GPUS）
        shuffle     : 是否打乱数据（当使用分布式采样器时由采样器接管，DataLoader 的 shuffle 必须为 False）
        drop_last   : 在样本数不能被 batch_size 整除时，是否丢弃最后一个不完整批次
                      - 训练多卡时通常设为 True，以确保每张卡拿到等量样本，避免梯度同步问题
    返回:
        loader      : torch.utils.data.DataLoader 实例"""
    dataset_name = cfg.DATA.NAME

    # Construct the dataset
    # 1) 构建数据集实例
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        # 若是 VTAB 系列数据集，使用 TFRecords（依赖 tensorflow_datasets）
        # 这里延迟导入，避免非 VTAB 场景下强制依赖 tensorflow
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(cfg, split)
    else:
        # 非 VTAB：从注册表中查找对应的数据集类
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        dataset = _DATASET_CATALOG[dataset_name](cfg, split)

    # Create a sampler for multi-process training
    # 2) 采样器（sampler）
    # 在多 GPU（分布式）训练时，使用 DistributedSampler 来：
    # - 按 rank 切分数据，保证每个进程看到不同子集
    # - 在每个 epoch 前提供可重复的 shuffle（由 set_epoch 控制种子）
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    # 3) 构建 DataLoader
    # ⚠️ 注意：当 sampler 不为空时，DataLoader 的 shuffle 必须为 False，否则会冲突
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),    # 分布式用 sampler 接管 shuffle
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,           # 多进程加载加速 IO 与预处理
        pin_memory=cfg.DATA.PIN_MEMORY,             # 将张量固定到内存页，可加速主机到 GPU 的拷贝
        drop_last=drop_last,                        # 是否丢弃最后一个不足 batch 的批次
    )
    return loader


def construct_train_loader(cfg):
    """Train loader wrapper.
    训练集 DataLoader 包装器（split='train'）。
    约定：
      - 多 GPU 训练时，通常 drop_last=True（各卡样本数一致，便于同步）
      - 有些项目将 batch_size 按 GPU 数量平均分配：总 batch / NUM_GPUS
    """
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,   # 训练期需要打乱；分布式时由 sampler 接管
        drop_last=drop_last,
    )


def construct_trainval_loader(cfg):
    """Train loader wrapper.
    训练+验证混合集 DataLoader（split='trainval'）。
    用法与 train_loader 类似，常用于某些协议下的全量训练/选择。"""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(cfg):
    """Test loader wrapper.
    测试集 DataLoader（split='test'）。
    约定：
      - 测试/验证阶段通常不打乱（shuffle=False）
      - 不丢弃最后一个批次（drop_last=False），尽量评完所有样本"""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(cfg, batch_size=None):
    """
    验证集 DataLoader（split='val'）。
    可选地允许单独指定验证 batch_size（例如为了节省显存或提高吞吐）。
    默认依旧按总 batch / NUM_GPUS 计算。
    """
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"
    Shuffles the data.
    在每个 epoch 开始前，为支持的采样器执行“打乱控制”。
    - RandomSampler：DataLoader 自带的 shuffle 已经处理，因此无需手动干预
    - DistributedSampler：需要在每个 epoch 调用 set_epoch(seed)，确保各进程
    基于相同的 epoch 种子进行一致/可复现的随机打乱
    """
    # 仅支持 RandomSampler 或 DistributedSampler 两类采样器
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically # RandomSampler：无需手动 shuffle，DataLoader 的 shuffle 参数已生效
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch # DistributedSampler：通过 epoch 设定随机种子，确保各进程一致地打乱
        loader.sampler.set_epoch(cur_epoch)
