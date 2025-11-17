#!/usr/bin/env python3
'''
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import Attention
from timm.models.vision_transformer import Block

from ...utils import logging
logger = logging.get_logger("visual_prompt")


class Pfeiffer_Block(Block):
    """
    继承自 timm 的标准 Transformer Block，并在 FFN(MLP) 之后插入一个 Adapter（Pfeiffer 风格）。

    标准 Block 的结构（timm）：
        x = x + DropPath( Attention( LN1(x) ) ) 即 LayerNorm → Self-Attention → DropPath → 与输入做残差相加
        x = x + DropPath(      MLP( LN2(x) ) )  即 LayerNorm → MLP(前馈网络) → DropPath → 与上一结果做残差相加

    本文件在第二条残差里，把 MLP 的输出再过一个“小瓶颈”：
        Adapter = Linear(dim -> dim/r) -> GELU -> Linear(dim/r -> dim)
    然后与 MLP 输出做残差相加（零初始化，初始等价于恒等映射）。
    """

    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        # 先按父类 Block 的方式把“注意力 + FFN + LN + DropPath”等都建好
        super(Pfeiffer_Block, self).__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop=drop, 
            attn_drop=attn_drop,
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer)
        
        self.adapter_config = adapter_config

        # 仅实现了 "Pfeiffer" 风格的 Adapter：FFN 后追加一个瓶颈模块并残差回加
        if adapter_config.STYLE == "Pfeiffer":
            # 降维：dim -> dim // r   （r=REDUCATION_FACTOR，注意代码里拼写就是这样）
            self.adapter_downsample = nn.Linear(
                dim,
                dim // adapter_config.REDUCATION_FACTOR
            )
            # 升维：dim // r -> dim
            self.adapter_upsample = nn.Linear(
                dim // adapter_config.REDUCATION_FACTOR,
                dim
            )
            self.adapter_act_fn = act_layer()

            # 关键：两层线性层全部零初始化，保证“初始 = 恒等映射”，训练初期不扰动预训练表示
            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)

            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)
        else:
            raise ValueError("Other adapter styles are not supported.")

    def forward(self, x):

        if self.adapter_config.STYLE == "Pfeiffer":
            # same as reguluar ViT block
            # ----- 第1条残差：Attention 子层 -----
            h = x                         # 残差分支
            x = self.norm1(x)             # LN1
            x = self.attn(x)              # Multi-Head Self-Attention
            x = self.drop_path(x)         # 随机深度
            x = x + h                     # 残差相加

            # ----- 第2条残差：MLP (+ Adapter) 子层 -----
            h = x                         # 残差分支
            x = self.norm2(x)             # LN2
            x = self.mlp(x)               # 标准 FFN(MLP)

            # start to insert adapter layers...
            # >>> 在这里“插入”Adapter：线性降维 -> 激活 -> 线性升维 -> 残差加回 <<<
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = adpt + x  # 与 MLP 输出做残差（零初始化 => 初始等价恒等）
            # ...end

            x = self.drop_path(x)         # 对整段（MLP+Adapter 后的输出）再做 DropPath
            x = x + h                     # 回到第2条残差的主干

            return x
