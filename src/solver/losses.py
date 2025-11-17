#!/usr/bin/env python3
"""
损失函数
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils import logging
logger = logging.get_logger("visual_prompt")


class SigmoidLoss(nn.Module):
    """
    使用 Sigmoid + BCE（交叉熵）形式的损失封装。

    这里的设计思路是：
    - 先把标签从“类别 id”（如 [0, 3, 5, ...]）转换为 multi-hot 形式（如 [1,0,0,0,...]）；
    - 然后用 `binary_cross_entropy_with_logits` 来计算每个类别的二元交叉熵；
    - 最后乘上 per-class 的权重，并对 batch 进行平均。

    当前实现实际上是“单标签场景用 multi-hot + BCE 来模拟 CE”，
    但写法也兼容将来多标签（multi-label）的情况。
    """
    def __init__(self, cfg=None):
        # 这里 cfg 暂时没用，但保留接口方便以后扩展（比如从 cfg 里读 class_weight 等）
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        """
         返回 True，表示“单一损失函数”（相对多分支、多头组合的复杂 loss）。
         当前项目中这个接口好像没被用到，但可以作为类型判断的标记位。
        """
        return True

    def is_local(self):
        """
        返回 False，表示不是“局部 patch-level 损失”，
        而是对整张图片 / 整个样本的分类损失（global-level）。
        """
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        """
        将类别 id（整型 1D 向量）转换成 multi-hot 向量。

        参数:
            labels: 形状 [B]，每个元素为一个类别 id （0 ~ C-1）
            nb_classes: 总类别数 C

        返回:
            target: 形状 [B, C] 的 0/1 向量，每一行只有一个 1（单标签场景）

        举例:
            labels = [1, 3], nb_classes = 5
            -> target =
               [[0,1,0,0,0],
                [0,0,0,1,0]]
        """
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        """
        计算基于 Sigmoid 的 BCE 损失。

        参数:
            logits: 模型输出的 logit 分数，形状 [B, C]
                    （尚未过 Sigmoid）
            targets: 形状 [B] 的整数标签（单标签场景），也可以扩展为多标签 id
            per_cls_weights: 每个类别的权重（列表或 1D tensor，长度为 C）
            multihot_targets: 目前并未使用，预留给“传入已经是 multi-hot 标签”的场景

        返回:
            标量损失值（对 batch 做平均）
        """
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        # 当前实现中，只支持“传入的是类别 id”这种情况，
        # 所以直接把 targets 转成 multi-hot。
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        # BCE with logits: 内部自动对 logits 做 Sigmoid，再计算二元交叉熵
        # reduction="none"：先保留每个样本、每个类别上的独立损失值
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")

        # per-class 权重：形状 [C] -> [1, C]，便于与 loss 按类别相乘
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")

        # 按类别加权：每个类别的损失乘上对应权重
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        # 对所有样本、所有类别求和，再除以 batch_size，得到平均损失
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        """
        标准 forward 接口（封装 self.loss），方便外部直接调用。

        参数:
            pred_logits: 同 logits
            targets: 同 targets
            per_cls_weights: 同上
            multihot_targets: 目前没有特殊使用，保持接口兼容

        返回:
            标量损失值
        """
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    """
    标准的 Softmax + CrossEntropy 损失封装。
    继承 SigmoidLoss 只是复用一些接口约定（构造函数、is_single 等），
    真正的损失实现覆盖了 loss() 方法。
    """
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        """
        使用 F.cross_entropy 计算多类交叉熵损失。

        参数:
            logits: [B, C]，模型输出的类别分数（未过 softmax）
            targets: [B]，整型类别 id
            per_cls_weights: 长度为 C 的 per-class 权重
            kwargs: 目前没用（保留扩展接口）

        返回:
            标量损失值（对 batch 做平均）
        """
        # 类别权重张量
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        # CrossEntropy 内部：先对 logits 做 log-softmax，再根据 targets 取对应项
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        # 和 SigmoidLoss 一样，对整个 batch 求平均
        return torch.sum(loss) / targets.shape[0]

# 一个字典，用来把字符串名字映射到具体损失类
LOSS = {
    "softmax": SoftmaxLoss,
    # 如果你之后扩展，可以写：
    # "sigmoid": SigmoidLoss,
    # "clip": ClipContrastiveLoss,
    # "softmax_clip": SoftmaxWithClipLoss,
}


def build_loss(cfg):
    """
    根据配置构建损失函数实例。

    cfg.SOLVER.LOSS 必须是上面 LOSS 字典中的一个键，比如 "softmax"。
    使用方法（伪代码）：
        loss_fn = build_loss(cfg)
        loss = loss_fn(logits, targets, per_cls_weights)

    返回:
        nn.Module 子类的实例（如 SoftmaxLoss），或抛出错误（loss 名称不在字典中）。
    """
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)
