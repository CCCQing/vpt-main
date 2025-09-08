#!/usr/bin/env python3
"""
Model construction functions.模型构建相关函数：根据 cfg 选择并实例化具体的模型（ResNet/ConvNeXt/ViT/Swin/SSLViT），完成日志打印、设备迁移（含分布式 DDP 包装）等通用流程。
"""
from tabnanny import verbose
import torch

from .resnet import ResNet
from .convnext import ConvNeXt
from .vit_models import ViT, Swin, SSLViT
from ..utils import logging
logger = logging.get_logger("visual_prompt")
# Supported model types # 支持的模型类型注册表：键为配置里的字符串，值为对应的模型类
_MODEL_TYPES = {
    "resnet": ResNet,
    "convnext": ConvNeXt,
    "vit": ViT,
    "swin": Swin,
    "ssl-vit": SSLViT,
}


def build_model(cfg):
    """
    build model here步骤：
      1) 校验配置中声明的模型类型是否受支持；
      2) 校验请求的 GPU 数量不超过物理可用数量（避免误配）；
      3) 按类型实例化模型；
      4) 打印模型参数规模与可训练参数占比（便于核对冻结策略是否生效）；
      5) 将模型加载到当前设备（GPU/CPU），如多卡则包一层 DDP；
      6) 返回 (model, device)。
    """
    # 1) 模型类型合法性校验
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)
    # 2) GPU 数量校验（常见于单机多卡场景）
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    # 3) 构造模型实例
    train_type = cfg.MODEL.TYPE
    model = _MODEL_TYPES[train_type](cfg)
    # 4) 打印模型结构（可选）与参数规模信息
    log_model_info(model, verbose=cfg.DBG)
    # 5) 迁移到设备；若多卡则包分布式并行
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")

    return model, device


def log_model_info(model, verbose=False):
    """
    Logs model info    打印模型信息：
      - verbose=True 时打印完整的 nn.Module 文本结构（可能很长）；
      - 总参数量与可训练参数量（requires_grad=True）；
      - 可训练参数占比（tuned percent），用于核对冻结/微调策略。
    """
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    # 模型总参数量
    model_total_params = sum(p.numel() for p in model.parameters())
    # 可参与梯度更新的参数量
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))
    # 说明：这里打印的“tuned percent”未附带百分号，仅数值 *100 后输出，保持与原逻辑一致。


def get_current_device():
    """
    返回当前进程应使用的设备：
      - 若 CUDA 可用：返回当前 CUDA 设备的索引（int），如 0/1/2...
      - 否则：返回 CPU 设备对象 torch.device('cpu')

    注意：该函数在 CUDA 可用时返回的是“int 索引”，在 CPU 时返回的是 torch.device，
    两者类型不同，但均被下游用法所兼容（保持原实现风格，不做改动）。
    """
    if torch.cuda.is_available():
        # Determine the GPU used by the current process  # 当前进程绑定的 CUDA 设备索引（在分布式/多卡环境由启动器设定）
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    """
    将模型迁移到合适的设备，并在多卡场景下用 DistributedDataParallel(DDP) 包装。

    参数：
      - model: 已构建的 nn.Module
      - cfg  : 配置对象（需包含 NUM_GPUS 等字段）

    返回：
      - (model, cur_device)
        若 CUDA 可用且 NUM_GPUS > 1，则返回 DDP 包装后的模型；
        若单卡或无 CUDA，则直接返回迁移到对应设备的普通模型。
    """
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device  # 把模型拷贝/绑定到当前 GPU（cur_device 是整型索引）
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting  # 多卡分布式：使用 DDP 进行多进程数据并行
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device  # 仅在当前设备上创建模型副本，避免不必要的跨卡通信
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[cur_device],     # 本进程负责的 GPU
                output_device=cur_device,    # 输出也放在本 GPU 上
                find_unused_parameters=True, # 若模型存在条件分支导致部分参数未被用到，设 True 可避免报错
            )
    else:
        model = model.to(cur_device)         # 纯 CPU 场景
    return model, cur_device
