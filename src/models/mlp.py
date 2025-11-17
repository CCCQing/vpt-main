#!/usr/bin/env python3
"""
Modified from: fbcode/multimo/models/encoders/mlp.py
多层感知机（MLP）封装：
- 支持任意层数、可配置维度列表；
- 每层可选归一化（BatchNorm / LayerNorm / 也可以 None）；
- 每层可选 Dropout；
- 最后一层可选“特殊偏置初始化”，常用于二分类中将初始预测偏向负类。
"""
import math
import torch

from torch import nn
from typing import List, Type

from ..utils import logging
logger = logging.get_logger("visual_prompt")


class MLP(nn.Module):
    """
    通用 MLP 模块（多层全连接网络）

    参数说明：
        input_dim: 输入特征维度（例如 512）
        mlp_dims:  每层隐藏维度列表，最后一个元素是“输出维度”
                   比如 [1024, 512, 2] 表示：
                     - 隐藏层1:  input_dim → 1024
                     - 隐藏层2: 1024 → 512
                     - 最后一层: 512 → 2（单独用 self.last_layer实现）
        dropout:   每层之后使用的 dropout 概率，0 表示不用
        nonlinearity: 激活函数类（比如 nn.ReLU / nn.GELU），会在每个隐藏层后加入一个实例
        normalization: 归一化层类（比如 nn.BatchNorm1d / nn.LayerNorm / None）
                       若为 None，则不使用归一化
        special_bias: 若为 True，则对最后一层的 bias 做特殊初始化
                      （常见于二分类，把初始预测概率设得较小）
        add_bn_first: 若为 True，则在进入第一层线性层之前先做一次
                      normalization + dropout（如果它们被启用）
    """
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm 也可；或直接设为 None
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()

        # 当前层的输入维度（初始为整体输入维度）
        projection_prev_dim = input_dim
        # 存放“前几层”的模块（不含最后一层线性）
        projection_modulelist = []

        # mlp_dims 的最后一个元素是最终输出维度
        last_dim = mlp_dims[-1]
        # 前面所有元素作为“中间隐藏层维度”
        mlp_dims = mlp_dims[:-1]

        # 如果需要在第一个线性层之前先做一次 BN + Dropout：
        if add_bn_first:
            if normalization is not None:
                # 比如 BatchNorm1d(input_dim) 或 LayerNorm(input_dim)
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        # 逐个构建中间隐藏层（不包含最后一层）
        for idx, mlp_dim in enumerate(mlp_dims):
            # 线性层：projection_prev_dim → mlp_dim
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            # 使用 Kaiming 正态初始化权重（适合 ReLU 系列激活）
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)

            # 激活函数（例如 ReLU）
            projection_modulelist.append(nonlinearity())

            # 可选归一化层（如 BatchNorm1d / LayerNorm）
            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            # 可选 Dropout
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

            # 更新“下一层的输入维度”
            projection_prev_dim = mlp_dim

        # 将所有中间层封装为一个 nn.Sequential
        # 注意：若 mlp_dims 为空列表，则 self.projection 可能是一个空 Sequential（恒等）
        self.projection = nn.Sequential(*projection_modulelist)

        # 最后一层线性层：从最后一个隐藏维度投到最终输出维度 last_dim
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')

        # special_bias=True 时，为最后一层 bias 做“先验概率”风格的初始化
        # 常用场景：二分类（输出维度=1），希望模型一开始更偏向“负类”
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        # 先经过中间层 MLP（可能为空，即恒等）
        x = self.projection(x)

        # 再经过最后一层线性映射，得到最终输出
        x = self.last_layer(x)
        return x
