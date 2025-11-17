#!/usr/bin/env python3
"""
Model construction functions.模型构建相关函数：根据 cfg 选择并实例化具体的模型（ResNet/ConvNeXt/ViT/Swin/SSLViT），
完成日志打印、设备迁移（含分布式 DDP 包装）等通用流程。
"""
from tabnanny import verbose
import torch

from .resnet import ResNet
from .convnext import ConvNeXt
from .vit_models import ViT, Swin, SSLViT
from ..utils import logging
# 获取一个名为 "visual_prompt" 的 logger，用于打印训练/构建信息
logger = logging.get_logger("visual_prompt")
# Supported model types
# 支持的模型类型注册表：
#   key  : 配置文件中使用的字符串（cfg.MODEL.TYPE）
#   value: 实际的模型类（构造函数）
_MODEL_TYPES = {
    "resnet": ResNet,
    "convnext": ConvNeXt,
    "vit": ViT,
    "swin": Swin,
    "ssl-vit": SSLViT,
}


def build_model(cfg):
    """
    根据配置 cfg 构建模型，并迁移到合适的设备上。

    build model here步骤：
      1) 校验 cfg.MODEL.TYPE 是否在 _MODEL_TYPES 声明的支持范围内；
      2) 校验 cfg.NUM_GPUS 不超过当前机器可用的 GPU 数量；
      3) 按 cfg.MODEL.TYPE 选出对应的模型类，并传入 cfg 构造实例；
      4) 打印模型的参数规模（总参数量 & 可训练参数量及占比）；
      5) 将模型迁移到当前设备（GPU / CPU），如多卡则用 DistributedDataParallel 包装；
      6) 返回 (model, device) 供外部训练 / 测试使用。
    """
    # 1) 模型类型合法性检查：cfg.MODEL.TYPE 必须在 _MODEL_TYPES 注册表中
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)

    # 2) GPU 数量检查：cfg.NUM_GPUS 不能超过 torch 检测到的物理 GPU 数量
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    # 3) 按配置中的类型名字从字典中取出对应的模型类，并用 cfg 实例化
    train_type = cfg.MODEL.TYPE
    model = _MODEL_TYPES[train_type](cfg)

    # 4) 打印模型结构 & 参数信息（若 cfg.DBG 为 True，会打印完整网络结构）
    log_model_info(model, verbose=cfg.DBG)

    # 5) 将模型迁移到设备（GPU/CPU），如果是多卡训练则用 DDP 包装
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")

    # 6) 返回模型与当前设备
    return model, device


def log_model_info(model, verbose=False):
    """
    Logs model info    打印模型信息：
      - 若 verbose=True，则打印完整的模型结构（str(model)）；
      - 打印模型的总参数量；
      - 打印参与梯度更新的参数量（requires_grad=True）；
      - 打印“可训练参数占总参数比例 tuned percent”，方便核对冻结策略是否生效。
    """
    if verbose:
        # 打印完整模型结构（可能非常长，调试用）
        logger.info(f"Classification Model:\n{model}")

    # 统计模型“总参数量”：所有参数 numel 求和
    model_total_params = sum(p.numel() for p in model.parameters())

    # 统计“可训练参数量”：仅统计 requires_grad=True 的参数
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    # 打印总参数量与可训练参数量
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))

    # 打印“可训练参数占比”：百分数形式（这里直接 *100 再输出）
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))
    # 说明：这里打印的“tuned percent”未附带百分号，仅数值 *100 后输出，保持与原逻辑一致。


def get_current_device():
    """
    获取当前应使用的设备（device）：

    返回值：
      - 若 CUDA 可用：返回当前 CUDA 设备的索引（int），即 torch.cuda.current_device()；
        例如 0 / 1 / 2 ...，用于 .cuda(device=cur_device) / DDP 的 device_ids 等。
      - 若 CUDA 不可用：返回 torch.device('cpu')。

    说明：
      这里返回类型在 GPU 场景是 int，在 CPU 场景是 torch.device，
      虽然类型不统一，但下游的使用方式（model.cuda(device=cur_device) 或 model.to(cur_device)）
      与原作者保持一致，不在此做修改。
    """
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        # 当前进程绑定的 CUDA 设备索引（在分布式/多卡环境由启动器设定）
        cur_device = torch.cuda.current_device()
    else:
        # 若没有 GPU，则统一用 CPU 设备对象
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    """
    将模型迁移到合适的设备，并在多卡场景下用 DistributedDataParallel(DDP) 包装。

    参数：
      - model: 已构建的 nn.Module
      - cfg  : 配置对象（需包含 NUM_GPUS 等字段）

    返回：
      - model     : 迁移到对应设备后的模型；
                    若 cfg.NUM_GPUS > 1 且 CUDA 可用，则为 DDP 包装后的模型；
                    否则为普通的单卡 / CPU 模型。
      - cur_device: 当前使用的设备（GPU 索引或 torch.device('cpu')）。
    """

    # 先获取当前设备（考虑到多进程分布式场景，每个进程负责一个 GPU）
    cur_device = get_current_device()

    if torch.cuda.is_available():
        # Transfer the model to the current GPU device  # 把模型拷贝/绑定到当前 GPU（cur_device 是整型索引）
        model = model.cuda(device=cur_device)

        # Use multi-process data parallel model in the multi-gpu setting  # 多卡分布式：使用 DDP 进行多进程数据并行
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            # DistributedDataParallel 会在每个进程上保存一个模型“副本”，并通过通信同步梯度
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[cur_device],     # 本进程负责的 GPU
                output_device=cur_device,    # 输出也放在本 GPU 上
                find_unused_parameters=True, # 若模型存在条件分支导致部分参数未被用到，设 True 可避免报错
            )
    else:
        model = model.to(cur_device)         # 纯 CPU 场景
    return model, cur_device
