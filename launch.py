#!/usr/bin/env python3
"""
launch helper functions
启动训练/评估前的一些通用帮助函数：
- 收集并打印环境信息（Python/PyTorch/CUDA/第三方库等）
- 构造命令行参数解析器（支持 --config-file 与额外覆盖项）
- 初始化日志系统与输出目录（单/多卡一致）
"""
import argparse
import os
import sys
import pprint
import PIL
from collections import defaultdict
from tabulate import tabulate
from typing import Tuple

import torch
from src.utils.file_io import PathManager
from src.utils import logging
from src.utils.distributed import get_rank, get_world_size


def collect_torch_env() -> str:
    """
    收集 PyTorch 的编译/依赖环境信息。优先使用 torch.__config__.show()（较新版本可用），
    若不可用则回退到较老的 get_pretty_env_info()。返回：格式化后的多行字符串。
    """
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch # 向后兼容：老版本 PyTorch 使用此方式汇总环境信息
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module() -> Tuple[str]:
    """
    读取自定义环境变量 'ENV_MODULE'，该变量常用于 HPC/集群软环境管理（如 Lmod）。
    返回：(变量名, 值/未设置占位符)
    """
    var_name = "ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


# def collect_env_info() -> str:
#     data = []
#     data.append(("Python", sys.version.replace("\n", "")))
#     data.append(get_env_module())
#     data.append(("PyTorch", torch.__version__))
#     data.append(("PyTorch Debug Build", torch.version.debug))
#
#     has_cuda = torch.cuda.is_available()
#     data.append(("CUDA available", has_cuda))
#     if has_cuda:
#         data.append(("CUDA ID", os.environ["CUDA_VISIBLE_DEVICES"]))
#         devices = defaultdict(list)
#         for k in range(torch.cuda.device_count()):
#             devices[torch.cuda.get_device_name(k)].append(str(k))
#         for name, devids in devices.items():
#             data.append(("GPU " + ",".join(devids), name))
#     data.append(("Pillow", PIL.__version__))
#
#     try:
#         import cv2
#
#         data.append(("cv2", cv2.__version__))
#     except ImportError:
#         pass
#     env_str = tabulate(data) + "\n"
#     env_str += collect_torch_env()
#     return env_str
def collect_env_info() -> str:
    """
    汇总并格式化当前运行环境信息，便于在日志里追踪复现实验。
    8.21修改:原代码里强取环境变量，此处把强访问改成安全读取 + 自动回退，这样本地/集群/CPU/GPU 都能跑，且不会因为没设环境变量就崩
    """
    data = []
    # Python 版本（去除换行避免表格错位）
    data.append(("Python", sys.version.replace("\n", "")))
    # 自定义环境模块（若无则显示 <not set>）
    data.append(get_env_module())
    # PyTorch 版本与是否为 Debug 构建
    data.append(("PyTorch", torch.__version__))
    data.append(("PyTorch Debug Build", torch.version.debug))
    # CUDA 可用性与设备信息
    has_cuda = torch.cuda.is_available()
    data.append(("CUDA available", has_cuda))
    if has_cuda:
        # SAFE: 不再抛 KeyError    # 安全读取环境变量，不抛异常
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_env == "":
            if torch.cuda.is_available():   # 未显式设置，但 torch 能检测到 GPU
                visible = ",".join(str(i) for i in range(torch.cuda.device_count()))
                cuda_env = f"(not set) -> detected GPUs: {visible}"
            else:   # 理论上不会进来（因为 has_cuda 为真），但留作稳妥分支
                cuda_env = "(not set) -> CUDA not available (CPU mode)"
        data.append(("CUDA ID", cuda_env))

        from collections import defaultdict
        # 统计每种 GPU 型号对应的设备索引列表，如 "GPU 0,1: NVIDIA GeForce RTX 3090"
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))
    # Pillow 版本
    data.append(("Pillow", PIL.__version__))
    # 可选：OpenCV 版本（如未安装则跳过）
    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    # 使用 tabulate 以“表格”形式美观输出
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()  # 追加更详细的 PyTorch 环境信息
    return env_str



def default_argument_parser():
    """
    create a simple parser to wrap around config file
        构造统一的命令行解析器：
    - --config-file: 指定 YAML 配置文件路径
    - --train-type : 训练类型（可选，自定义用途）
    - opts         : 以 KEY VALUE 形式在命令行临时覆盖配置项（Detectron 风格）
    例如：输入python train.py --config-file cub.yaml SEED 1
    解析后命名空间为：'config_file': 'cub.yaml' 'train_type': '' 'opts': ['SEED', '1']
    """
    parser = argparse.ArgumentParser(description="visual-prompt")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file")    # 指定配置文件路径
    parser.add_argument(
        "--train-type", default="", help="training types")  # 训练类型（可选）
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",    # 以命令行形式覆盖配置项
        default=None,
        nargs=argparse.REMAINDER,   # 余下的参数都归到 opts 列表中
    )

    return parser


def logging_train_setup(args, cfg) -> None:
    """
    初始化训练日志与输出目录，并打印关键信息（进程 rank、环境、命令行参数、配置内容等）。
    参数：
    - args: 命令行解析结果（包含 config_file / opts 等）
    - cfg : 已合并/冻结的配置对象
    """
    # 1) 创建输出目录（多卡/多进程时每个 rank 同样会执行，不存在即创建）
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        PathManager.mkdirs(output_dir)
    # 2) 初始化 logger（内部会根据 rank/world_size 控制主/从进程的日志行为）
    logger = logging.setup_logging(
        cfg.NUM_GPUS,           # 配置文件中声明的 GPU 数量（用于日志标题等）
        get_world_size(),       # 实际分布式世界大小（由启动器/环境决定）
        output_dir,             # 日志/事件文件输出目录
        name="visual_prompt"    # logger 名称前缀
    )

    # Log basic information about environment, cmdline arguments, and config
    # 3) 打印分布式信息与环境信息
    rank = get_rank()
    logger.info(
        f"Rank of current process: {rank}. World size: {get_world_size()}")
    logger.info("Environment info:\n" + collect_env_info())
    # 4) 记录命令行与配置文件内容（方便复现实验）
    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        with PathManager.open(args.config_file, "r", encoding="utf-8") as f:
            cfg_text = f.read()
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                cfg_text  # 直接把 YAML 内容打进日志
            )
        )
    # Show the config
    # 5) 打印最终生效的配置（pprint 友好显示）
    logger.info("Training with config:")
    logger.info(pprint.pformat(cfg))
    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of typical val set.
    # 6) CUDNN benchmark：若只评估（eval_only=True）就不动；训练时按配置项决定是否启用。
    #    注意：cudnn.benchmark 在输入尺寸固定时有加速，但在尺寸变化频繁的验证集上会带来额外开销。
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
