#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")   # 屏蔽第三方库的一些非关键警告，避免日志噪声


def setup(args):
    """
    Create configs and perform basic setups.创建配置并做基础环境设置（包含：合并配置、分布式地址、输出目录等）
    """
    cfg = get_cfg()                             # 拿到默认配置（ConfigNode 的克隆）
    cfg.merge_from_file(args.config_file)       # 从命令行指定的 yaml 文件合并配置
    cfg.merge_from_list(args.opts)              # 从命令行的 KEY VALUE 形式补充/覆盖配置

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])
    # 8.21日修改os.environ访问环境变量在本地/非 SLURM 环境下这个变量不存在
    # -------------------- 分布式初始化地址设置 --------------------
    # 原始实现：在 SLURM 环境下从环境变量取节点名并组装 tcp://<node>:12399
    # 但在本地/非 SLURM 环境下不存在该变量，会触发 KeyError。
    # 因此这里做了健壮化处理：若环境变量存在则覆盖 DIST_INIT_PATH，否则保持配置默认值。
    node = os.environ.get("SLURMD_NODENAME")
    if node:
        # 在 SLURM 集群节点上，显式用 tcp://<节点名>:12399，保证多机能联通
        cfg.DIST_INIT_PATH = f"tcp://{node}:12399"
    # 否则什么也不做，保留配置里的默认值在本地 8.21修改结束

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    # -------------------- 输出目录组织规则 --------------------
    # 期望目录结构：
    # <OUTPUT_DIR>/<DATA.NAME>/<DATA.FEATURE>/lr<lr>_wd<wd>/run<count>
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    # 为了避免多进程/并发提交时互相覆盖，同一组超参最多运行 RUN_N_TIMES 次
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        # 随机睡眠 3~30 秒，降低并发冲突概率（例如多个作业同时创建同名目录）
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)     # 仅当路径不存在时创建
            cfg.OUTPUT_DIR = output_path        # 将最终输出目录写回 cfg，供后续日志/ckpt 使用
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        # 保护：若同配置已经跑满 RUN_N_TIMES 次，则直接报错退出
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()                               # 冻结配置，防止后续代码意外修改
    return cfg


def get_loaders(cfg, logger):
    """
        根据配置构建 DataLoader（train / val / test）
        - VTAB 任务：按照官方约定使用 train+val（trainval）作为最终训练集
        - 其他任务：构建标准的 train / val / test
    """
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        # VTAB 最终训练：使用 800/200 调参后，合并 train+val 作为最终训练集
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    # 注：VTAB 最终运行阶段通常不需要 val，但这里保持统一接口
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        # 某些场景不提供测试集（或只做训练/验证），此时返回 None
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    """
        训练与评估主流程：
        - 清理显存缓存
        - 固定随机种子（若配置提供）
        - 初始化日志与环境
        - 构建数据加载、模型、评估器与训练器
        - 执行训练（或仅评估）
    """
    # clear up residual cache from previous runs    清理上一次运行遗留的 GPU 显存缓存（若可用 GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here
    # ---------- 复现实验：固定随机种子 ----------
    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)    # 注意：这里对 Python 自带的 random 固定为 0，而非 cfg.SEED 若需要完全一致的复现，可改为 random.seed(cfg.SEED)
        random.seed(0)
    # -------------------------------------------

    # ---------- 日志/环境初始化 ----------
    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")
    # -----------------------------------

    # ---------- 构建数据加载器 ----------
    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    # -----------------------------------

    # ---------- 构建模型 ----------------
    logger.info("Constructing models...")   # 打一条 INFO 级别的日志
    model, cur_device = build_model(cfg)    # 根据prompt/linear/adapter 等与骨干网络类型（如 ViT/Swin）来实例化对应的模型与放置设备
    # -----------------------------------

    # ---------- 评估器与训练器 ----------
    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()                 # 组织评估指标与评测逻辑
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)    # Trainer 封装了训练/验证/测试的循环与保存逻辑
    # -----------------------------------

    # ---------- 训练或退出 ----------
    if train_loader:
        # 常规训练流程：在 train 上训练，周期性在 val/test 上评估
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")    # 极少数场景（例如仅做推理/分析）可能没有 train_loader
    # -----------------------------------

    # ---------- 仅评估模式 ----------
    if cfg.SOLVER.TOTAL_EPOCH == 0:         # 若 TOTAL_EPOCH 设为 0，可跳过训练，直接在 test 上评估一次
        trainer.eval_classifier(test_loader, "test", 0)
    # -----------------------------------

def main(args):
    """main function to call from workflow  脚本主入口（供命令行调用）"""

    # set up cfg and args 1) 解析并合并配置
    cfg = setup(args)

    # Perform training. 2) 执行训练/评估
    train(cfg, args)


if __name__ == '__main__':
    # 解析命令行参数（支持 --config-file 与后续的 KEY VALUE 覆盖）
    args = default_argument_parser().parse_args()
    main(args)
