#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
models for vits, borrowed from
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling_resnet.py
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
"""
import copy
import logging
import math
# 8.21改动
# 原有 Win 下 os.path.join 会产生反斜杠 "\"，会影响从权重字典中取键（键名一般用 "/"）
# 因此改为从 posixpath 导入 join，确保键名分隔符始终为 "/"
# from os.path import join as pjoin  # 原有Win下os.path.join会产生反斜杠 \
from turtle import forward           # 原有代码就这两行
from posixpath import join as pjoin  # 关键：确保键名里总是用 "/"
from collections import OrderedDict  # 解决 OrderedDict 未导入
import torch.nn.functional as F      # 解决 F 未导入
# 8.21改动结束

import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from ...configs import vit_configs as configs # 结构配置（B/16、L/16、H/14 等）


logger = logging.getLogger(__name__)
# 预设配置名称到具体配置对象的映射（不同尺寸、预训练来源等）
CONFIGS = {
    # "sup_vitb8": configs.get_b16_config(),
    "sup_vitb16_224": configs.get_b16_config(),
    "sup_vitb16": configs.get_b16_config(),
    "sup_vitl16_224": configs.get_l16_config(),
    "sup_vitl16": configs.get_l16_config(),
    "sup_vitb16_imagenet21k": configs.get_b16_config(),
    "sup_vitl16_imagenet21k": configs.get_l16_config(),
    "sup_vitl32_imagenet21k": configs.get_l32_config(),
    'sup_vitb32_imagenet21k': configs.get_b32_config(),
    'sup_vitb8_imagenet21k': configs.get_b8_config(),
    'sup_vith14_imagenet21k': configs.get_h14_config(),
    # 'R50-ViT-B_16': configs.get_r50_b16_config(),
}

# 下列常量为从 TF/Flax 权重转 PyTorch 时对应的键名片段
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """
    Possibly convert HWIO to OIHW.
    将 numpy 权重转换为 torch.Tensor。
    若 conv=True，则从 HWIO 转换为 OIHW（TF/Flax 卷积到 PyTorch 卷积的维度顺序差异）。
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    """Swish 激活函数：x * sigmoid(x)"""
    return x * torch.sigmoid(x)

# 支持的激活函数查表
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    """
    标准多头自注意力模块（MHSA）。
    输入：hidden_states [B, N, D]（含 CLS，N=1+patches）
    输出：attention_output [B, N, D]，可选返回注意力权重 weights（用于可视化）
    """
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis  # 是否返回注意力权重以用于可视化
        self.num_attention_heads = config.transformer["num_heads"]  # 头数 h
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)   # 每头维度 d_k
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # 总维度 D

        # Q/K/V 线性映射：输入/输出维度均为 hidden_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        # 输出线性层 + dropout
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        """将张量从 [B, N, D] 变形为 [B, h, N, d_k] 以便做多头注意力。"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # (..., h, d_k)
        x = x.view(*new_x_shape)    # [B, N, h, d_k]
        return x.permute(0, 2, 1, 3)    # -> [B, h, N, d_k]

# ----------------11.17日更换函数名：构造亲和--------------------------------------------
#    def forward(self, hidden_states):
# ----------------11.17日更换函数名结束--------------------------------------------
    def _project_qkv(self, hidden_states):
        """
        对整段序列一次性做 Q/K/V 线性映射，并且直接变形为多头形式。

        设计目的：
        - 原来的 forward 每次都单独调用 query/key/value 三个 Linear；
        - 现在把这一步封装出来，后续可以在“正常注意力”和“亲和矩阵构造”
          两条分支中复用同一份 q/k/v，避免重复计算线性层。

        输入:
            hidden_states: [B, N, D]，其中 N = 1 + L_p + L_v
                一般约定:
                - 第 0 个 token 是 CLS
                - 后面若干 token 是 prompt（长度为 prompt_length）
                - 剩余 token 是视觉 patch

        输出:
            query_layer: [B, h, N, d_k]
            key_layer:   [B, h, N, d_k]
            value_layer: [B, h, N, d_k]
        """
        # 线性映射到 Q/K/V，维度仍为 D
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 拆分为多头
        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size

        return query_layer, key_layer, value_layer

    def _scaled_attention(self, query_layer, key_layer, value_layer):
        """
        缩放点积注意力的核心计算部分。

        输入:
            query_layer: [B, h, N, d_k]
            key_layer:   [B, h, N, d_k]
            value_layer: [B, h, N, d_k]

        步骤:
            1) 计算未归一化注意力 logits: Q * K^T / sqrt(d_k)
            2) softmax 行归一化 -> attention_probs
            3) Dropout
            4) 加权求和: Attn * V
            5) 合并多头 -> [B, N, D]
            6) 输出线性层 self.out + proj_dropout

        返回:
            attention_output: [B, N, D]
            weights: [B, h, N, N] 或 None（仅在 vis=True 时保留，用于可视化）
        """
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key) # 行归一化
        weights = attention_probs if self.vis else None     # 用于可视化
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B, N, h, d_k]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [B, N, D]
        context_layer = context_layer.view(*new_context_layer_shape)

        # 输出线性映射 + dropout
        attention_output = self.out(context_layer)  # [B, N, D]
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

    def forward(self, hidden_states):
        """
        常规前向：
        - 仅返回注意力后的输出和（可选）注意力权重；
        - 不暴露 q/k，用于绝大多数训练/推理场景。
        """
        query_layer, key_layer, value_layer = self._project_qkv(hidden_states)
        return self._scaled_attention(query_layer, key_layer, value_layer)

    def forward_with_projections(self, hidden_states):
        """
        拓展版前向：
        - 与 forward 完全共享 Q/K/V 投影与注意力计算的逻辑；
        - 额外返回多头形式的 q_proj / k_proj，便于后续构造 QQ / KK 亲和矩阵。

        使用场景：
        - 你想在保持 ViT 正常前向的同时，“窥视”这一层的几何结构（q/k），
          用来自构 App / Avv / Apv 等亲和并做额外的损失。
        """
        query_layer, key_layer, value_layer = self._project_qkv(hidden_states)
        attention_output, weights = self._scaled_attention(query_layer, key_layer, value_layer)
        return attention_output, weights, query_layer, key_layer

    def compute_affinity(self, query_layer, key_layer, prompt_length, mode="qq", *,
                         return_cross=False, normalize=True, detach=True):
        """
        使用（通常是被冻结的）W_q/W_k 投影后的 q/k，按 CLS | prompt | patch 切分，
        构造提示段/视觉段内部及（可选）prompt→patch 的亲和矩阵。

        参数:
            query_layer / key_layer:
                - 形状 [B, h, N, d_k]，来自 forward_with_projections 的输出
                - 一般使用同一层的 q/k，这保证亲和与该层注意力共享几何基底
            prompt_length:
                - prompt token 的数量 L_p（不含 CLS）
                - 序列结构假设为：
                    index 0: CLS
                    index 1..L_p: prompt token
                    index 1+L_p..: patch token
            mode:
                - "qq": 使用 q 计算 self-similarity (Q Q^T)
                - "kk": 使用 k 计算 self-similarity (K K^T)
            return_cross:
                - 若为 True，同时返回 Apv（prompt→patch）亲和矩阵
            normalize:
                - True: 在最后一维上做 softmax，得到“注意力风格”的亲和（行归一）
                - False: 返回未归一化的相似度矩阵（可用于自定义归一/温度）
            detach:
                - True: 对用于构亲和的 q/k 调用 .detach()
                    -> 亲和损失仅作为几何约束，不会对 W_q/W_k 产生梯度
                - False: 允许亲和损失反向到 W_q/W_k（例如最后几层微调）

        返回:
            affinities: dict，键包括:
                - "App": [B, h, L_p, L_p]，prompt 内自亲和
                - "Avv": [B, h, L_v, L_v]，patch 内自亲和
                - "Apv": [B, h, L_p, L_v]，prompt→patch 亲和（可选）
        """
        if mode not in {"qq", "kk"}:
            raise ValueError(f"Unsupported affinity mode: {mode}")

        # 选择使用 q 还是 k 作为特征基底
        base = query_layer if mode == "qq" else key_layer
        if detach:
            base = base.detach()

        # 基本长度检查，防止切片越界
        if base.size(2) < 1 + prompt_length:
            raise ValueError(
                f"Sequence length {base.size(2)} is insufficient for prompt_length={prompt_length} (needs >= {1 + prompt_length})."
            )

        # token 维度切分：
        # [0]            -> CLS
        # [1 : 1+Lp]     -> prompt 段
        # [1+Lp : end]   -> patch 段
        cls_offset = 1
        prompt_slice = slice(cls_offset, cls_offset + prompt_length)
        patch_slice = slice(cls_offset + prompt_length, None)

        # base: [B, h, N, d_k]
        prompt_tokens = base[:, :, prompt_slice, :]
        patch_tokens = base[:, :, patch_slice, :]
        scale = 1.0 / math.sqrt(self.attention_head_size)

        affinities = {}

        # 提示段内部自亲和 App（若存在 prompt）
        if prompt_tokens.numel() > 0:
            # [B, h, L_p, d_k] @ [B, h, d_k, L_p] -> [B, h, L_p, L_p]
            app = torch.matmul(prompt_tokens, prompt_tokens.transpose(-1, -2)) * scale
            affinities["App"] = self.softmax(app) if normalize else app

        # 视觉 patch 段内部自亲和 Avv
        if patch_tokens.numel() > 0:
            # [B, h, L_v, d_k] @ [B, h, d_k, L_v] -> [B, h, L_v, L_v]
            avv = torch.matmul(patch_tokens, patch_tokens.transpose(-1, -2)) * scale
            affinities["Avv"] = self.softmax(avv) if normalize else avv

        # prompt→patch 的交叉亲和 Apv（可选）
        if return_cross and prompt_tokens.numel() > 0 and patch_tokens.numel() > 0:
            # [B, h, L_p, d_k] @ [B, h, d_k, L_v] -> [B, h, L_p, L_v]
            apv = torch.matmul(prompt_tokens, patch_tokens.transpose(-1, -2)) * scale
            affinities["Apv"] = self.softmax(apv) if normalize else apv

        return affinities


class Mlp(nn.Module):
    """
    前馈网络（FFN）：两层 MLP
    输入/输出维度：D → mlp_dim → D（逐 token 的通道内非线性变换）
    """
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 初始化权重，小方差正态初始化偏置。"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # [B, N, D] → [B, N, mlp_dim] → [B, N, D]
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    将图像转换为 Transformer 的输入序列：
    - 纯 ViT：以 patch_size 为 kernel=stride 的卷积划分 patch，再线性映射到隐藏维
    - hybrid：先用 ResNetV2 提取特征（步幅 16），再以小 patch 做映射
    - 加上可学习的 [CLS] 与绝对位置编码
    输出：embeddings [B, 1+N, D]
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)  # 支持单值或二元组
        # 两种 patch 设定：
        # 1) hybrid 模式：使用 ResNetV2 做低层特征，随后以 16x16 的 stride 切块
        # 2) 纯 ViT：直接以 patch_size 卷积进行切块
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            # 由图像 / 16 再除 grid 得到“每个 patch 的核/步幅”，n_patches= (H/16) * (W/16)
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:   # config.patches["grid"] 不存在时vit
            patch_size = _pair(config.patches["size"])  # 例如 (16,16)
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            # ResNetV2 作为视觉前端   # 切块并映射到 hidden_size
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16  # 升维，匹配ResNet 输出的通道数，后面降到hidden_size
        # 将图像/特征图划分为 patch：Conv2d 的 kernel=stride=patch_size # 做“切块 + 线性映射”
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 位置编码，长度为 n_patches + 1（包含 cls）
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        # 可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅计算 patch 的嵌入，不加入 CLS 也不加位置编码。
        输入：x [B, C, H, W]
        输出：patch_tokens [B, N, D]
        """


        if self.hybrid:
            x = self.hybrid_model(x)    # 先经过 ResNetV2
        x = self.patch_embeddings(x)    # 划分 patch 并升维到 D，形状 [B, D, H', W']
        x = x.flatten(2)                # -> [B, D, N]
        patch_tokens = x.transpose(-1, -2)         # -> [B, N, D]
        return patch_tokens

    def add_cls_and_pos(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        在 patch tokens 前追加 CLS，并加上位置编码与 dropout。

        输入：patch_tokens [B, N, D]
        输出：embeddings [B, 1+N, D]
        """
        B = patch_tokens.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, patch_tokens), dim=1)  # 拼接 [CLS] -> [B, 1+N, D]

        embeddings = x + self.position_embeddings   # 加位置编码
        embeddings = self.dropout(embeddings)
        return embeddings

    def forward(self, x):
        """
        输入：x [B, C, H, W]
        输出：embeddings [B, 1+N, D]
        """
        patches = self.forward_patches(x)
        return self.add_cls_and_pos(patches)



class Block(nn.Module):
    """
    标准 Transformer Block：
    段1：LN → MHSA → 残差
    段2：LN → MLP  → 残差
    """
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size   # hidden_size=D：每个 token 的通道维度
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)   # 段1的 LayerNorm
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)         # 段2的 LayerNorm
        self.ffn = Mlp(config)                  # 段2的两层 MLP（D→H→D，通常 H≈4D），完成通道内的非线性变换
        self.attn = Attention(config, vis)      # 段1的多头自注意力

    def forward(self, x):
        # 段1：注意力 + 残差
        h = x  # 残差分支
        x = self.attention_norm(x)  # LN
        x, weights = self.attn(x)   # MHSA（输出同形状）; weights 仅在 vis=True 时非 None
        x = x + h                   # 残差相加

        # 段2：FFN + 残差
        h = x
        x = self.ffn_norm(x)        # LN
        x = self.ffn(x)             # MLP: D→H→D
        x = x + h                   # 残差相加
        return x, weights

    def forward_with_affinity(self, x, affinity_config):
        """
        带亲和矩阵输出的前向：

        主干逻辑：
            - 与标准 forward 完全一致：
                h = x
                x_norm = LN1(x)
                x_attn = Attn(x_norm)
                x = x_attn + h
                h = x
                x = LN2(x)
                x_ffn = MLP(x)
                x = x_ffn + h

        额外逻辑：
            - 在注意力部分，通过 forward_with_projections 一次性拿到 q_proj/k_proj；
            - 使用 compute_affinity 在同一层的 Q/K 上构造 App/Avv(/Apv)；
            - 将该层的亲和 dict 返回给上级 Encoder 统一收集。

        affinity_config: dict，支持字段：
            - "prompt_length": int，prompt token 数量 L_p
            - "mode": "qq" 或 "kk"（决定使用 Q 还是 K）
            - "return_cross": bool，是否额外返回 Apv
            - "normalize": bool，是否 softmax 归一
            - "detach": bool，是否在构亲和前对 q/k detach

        返回:
            x:          [B, N, D]，本层输出
            weights:    注意力权重（仅 vis=True 时非 None）
            affinities: dict，包含 App/Avv(/Apv)
        """
        # --- 注意力分支 + 残差 ---
        h = x
        x_norm = self.attention_norm(x)
        # 同时拿到 MHSA 输出 + 多头形式的 q_proj / k_proj
        x, weights, q_proj, k_proj = self.attn.forward_with_projections(x_norm)
        x = x + h

        # --- FFN 分支 + 残差 ---
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        # --- 基于 Q/K 计算 prompt/patch 亲和 ---
        affinities = self.attn.compute_affinity(
            q_proj,
            k_proj,
            affinity_config.get("prompt_length", 0),
            mode=affinity_config.get("mode", "qq"),
            return_cross=affinity_config.get("return_cross", False),
            normalize=affinity_config.get("normalize", True),
            detach=affinity_config.get("detach", True),
        )

        return x, weights, affinities




    def load_from(self, weights, n_block):
        """从预训练权重字典中加载当前 block 的参数（处理维度与键名）。"""
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # Q/K/V/Out 权重与偏置（需要转置到 PyTorch 线性层格式）
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            # MLP 两层
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            # 两个 LayerNorm
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    """堆叠多个 Transformer Block，并在末尾加一层 LayerNorm。"""
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()    # 保存有序的多层子模块
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6) # 在所有 block 之后再做一次 LayerNorm
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)  # 每层都是同结构的 Transformer Block（内部是 LN→MHSA→残差；LN→MLP→残差）
            self.layer.append(copy.deepcopy(layer))
        # 本实现的 Block 属于 Pre-LN（在每个子层前 LN），额外的末端 LN（有些论文称 final LN）有助于稳定训练并改善表征
    def forward(self, hidden_states):
        """常规前向：返回编码结果与（可选）各层注意力权重。"""
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states) # hidden_states为(B, 1+N, D)D 为 hidden_size
            if self.vis:
                attn_weights.append(weights)    # 把每层的 weights 保存到列表里否则返回空列表
        encoded = self.encoder_norm(hidden_states)  # 对最后一层输出再做一次 LayerNorm，得到 encoded
        return encoded, attn_weights

    def forward_with_affinity(self, hidden_states, affinity_config):
        """
        与标准 Encoder.forward 类似，但在遍历每一层 Block 时，
        额外收集该层的 prompt/patch 亲和矩阵。

        设计目的：
            - 你可以得到:
                - encoder_norm 之后的最终输出 encoded
                - 每层 MHSA 的注意力权重 attn_weights（若 vis=True）
                - 每层基于 W_q/W_k 的几何亲和信息 affinities
            - 方便在外层做：
                - 按层的 QQ/KK 消融（看哪几层几何更有用）
                - 三元一致 / 蒸馏 / 校准 等损失的分层设计

        输入:
            hidden_states: [B, N, D]，embedding 输出（含 CLS + prompt + patch）
            affinity_config: dict，同 Block.forward_with_affinity

        返回:
            encoded:     [B, N, D]，末端 LN 后的输出
            attn_weights: list，长度 = num_layers（视 vis 而定）
            affinities:   list，长度 = num_layers，每个元素是 dict(App/Avv/Apv)
        """
        attn_weights = []
        affinities = []
        for layer_block in self.layer:
            hidden_states, weights, affinity = layer_block.forward_with_affinity(hidden_states, affinity_config)
            if self.vis:
                attn_weights.append(weights)
            affinities.append(affinity)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, affinities


    def forward_cls_layerwise(self, hidden_states):
        """
        返回“逐层 CLS 向量”：
        [输入 embeddings 的 CLS] + [每层输出的 CLS（最后一层前）] + [末端 LN 后的 CLS]
        仅支持 batch_size=1。
        """
        # hidden_states: B, 1+n_patches, dim

        if hidden_states.size(0) != 1:
            raise ValueError('not support batch-wise cls forward yet')
        
        cls_embeds = []
        cls_embeds.append(hidden_states[0][0])  # 输入 embeddings 的 CLS

        for i,layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if i < len(self.layer)-1:
                cls_embeds.append(hidden_states[0][0])  # 每个 block 输出的 CLS（最后一层前）

        encoded = self.encoder_norm(hidden_states)
        cls_embeds.append(hidden_states[0][0])  # 最终 LN 后的 CLS

        cls_embeds = torch.stack(cls_embeds) # 12, dim # [num_layers+1, D]（例如 12 层就是 13×D：输入 CLS + 11 个中间 CLS + 最终 CLS）
        return cls_embeds



class Transformer(nn.Module):
    """
    完整的 Transformer：Embeddings（CLS+patch+pos）+ Encoder（多层 Block）
    """
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        """标准前向：返回编码后的序列与注意力权重。"""
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


    def forward_with_affinity(self, input_ids, affinity_config):
        """
        与标准 forward 类似，但返回“带亲和”的版本：

        步骤:
            1) Embeddings: x -> [CLS | prompt | patch] + pos_embed
            2) Encoder.forward_with_affinity:
                - 得到 encoded（末端 LN）
                - attn_weights: 每层 MHSA 的 attention map（可选）
                - affinities:   每层 App/Avv(/Apv) 亲和字典

        返回:
            encoded:   [B, N, D]
            attn_weights: list
            affinities:   list[dict]
        """
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights, affinities = self.encoder.forward_with_affinity(embedding_output, affinity_config)
        return encoded, attn_weights, affinities

    def forward_cls_layerwise(self, input_ids):
        """逐层返回 CLS 表征。"""
        embedding_output = self.embeddings(input_ids)

        cls_embeds = self.encoder.forward_cls_layerwise(embedding_output)
        return cls_embeds


class VisionTransformer(nn.Module):
    """顶层 ViT 分类模型：Transformer 编码 + 线性分类头。"""
    def __init__(
        self, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        super(VisionTransformer, self).__init__()
        config = CONFIGS[model_type]    # 根据名称取对应配置
        self.num_classes = num_classes
        self.classifier = config.classifier  # 选择"token" or "gap" 等策略（影响位置编码重载方式）
        # "token"：用 CLS token 做分类（标准 ViT）；"gap"：全局平均池化所有 patch token（有些变体这么做）。这里主要用来指导位置编码的拆分/插值
        self.transformer = Transformer(config, img_size, vis)
        # 分类头：若 num_classes<=0 则用恒等映射（仅提取特征）
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        """
        输入：x [B, 3, H, W]
        输出：若 vis=False 返回 logits；否则返回 (logits, attn_weights)
        """
        x, attn_weights = self.transformer(x)   # x: [B, 1+N, D]，attn_weights 典型形状：[num_layers, B, num_heads, 1+N, 1+N]（仅在 vis=True 时非空）
        logits = self.head(x[:, 0])     # 取 CLS token 做分类，x[:, 0] 始终是 CLS 的向量，维度 D=config.hidden_size

        if not vis:
            return logits
        return logits, attn_weights # attn_weights: num_layers, B, num_head, num_patches, num_patches [num_layers, B, h, N, N]

    def forward_with_affinity(self, x, affinity_config, vis=False):
        """
        顶层带亲和输出的前向接口：

        输入:
            x: [B, 3, H, W] 原始图像
            affinity_config: dict，穿透传递给 Transformer / Encoder / Block
            vis: 是否同时返回注意力权重（保持与原 forward 一致的开关语义）

        步骤:
            1) x -> Transformer.forward_with_affinity:
                得到:
                    - 序列输出 x_seq: [B, 1+N, D]
                    - attn_weights:  每层 MHSA 权重（可选）
                    - affinities:    每层 App/Avv(/Apv) 亲和字典
            2) 分类头:
                用 CLS token 的表征 x_seq[:, 0] 做线性分类

        返回:
            若 vis=False:
                logits, affinities
            若 vis=True:
                logits, attn_weights, affinities
        """
        x, attn_weights, affinities = self.transformer.forward_with_affinity(x, affinity_config)
        logits = self.head(x[:, 0])

        if not vis:
            return logits, affinities
        return logits, attn_weights, affinities


    def forward_cls_layerwise(self, x):
        """返回每层 CLS 表征序列（便于可视化/诊断）。依次拿到“输入 embeddings 的 CLS、每层输出的 CLS、末端 LN 后的 CLS”"""
        cls_embeds = self.transformer.forward_cls_layerwise(x)
        return cls_embeds   # 返回形状： [num_layers+1, D]（batch_size=1 的情况）

    def load_from(self, weights):
        """
        从预训练（通常是 JAX/TF）权重字典加载参数，并处理位置编码尺寸不一致。
        网格是什么：图像会按照patch大小切块，每个 patch 相当于一个 token，放回二维排布，就得到一个 patch 网格（grid）
            ViT 的 绝对位置编码是给网格里每个格子（每个 patch）分配一个 D 维向量
            很多实现把网格当作方形（H=W、且p相同），包括在输入之后进行裁剪
            预训练网格（G）= gs×gs故gs = sqrt(G),eg:224×224图,patch=16 → 14×14 网格 →G=196
            输入网格（N）为图像尺寸/patch对应的网格,eg:384×384图,仍 patch=16 → 24×24 网格 → N=576
        插值是什么：预训练的“位置参数”只有 G 份，但你现在需要 N 份。故预训练的位置编码“缩放”到新网格 —— 这就是“插值”
            位置编码是可学习参数，只有在预训练用到的那些格子位置上有“合理”的值
        双线性插值是什么：二维上对每个新坐标的值，在旧网格里找到它落在哪个2×2邻域中（四个最近的旧点），沿x方向做一次线性插值，再沿y方向对前一步结果再做一次线性插值
        hybrid前端是什么：在纯 ViT：直接以 patch_size 做 Conv2d(stride=patch) 切块
                        hybrid：先用 ResNetV2(CNN)把图变为步幅 16 的特征图（例如 224→14×14），再做一个小卷积（常是 1×1）把通道映射到 hidden_size，然后再进 Transformer
                        既是hybrid将图像用CNN进行预处理成特征图，再切成token送进transformer，这样引入 CNN 的归纳偏置，让模型天生具备“局部+共享+层级”的先验，这能在数据不多
                        、训练预算有限时明显提升稳定性和数据效率。
                        CNN 的归纳偏置主要有三条：局部性：近邻像素更相关（小卷积核只看局部）。平移等变/共享权重：同一个卷积核在整个图上共享，相同的图案不管在左上还是右下，都能被“同一组参数”检测到。
                        金字塔/层级特征：多层卷积+下采样，天然形成“边缘→纹理→部件→物体”的层级抽象。
        """
        with torch.no_grad():
            # Patch Embedding（卷积权重需要 HWIO->OIHW）
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            # CLS 与 Encoder 最后 LayerNorm
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            # ----------位置编码插值：若尺寸不同（例如输入分辨率不同），进行双线性插值重排---------------
            # 拆分 CLS 和网格部分
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"]) # [1, 1+G, D]
            posemb_new = self.transformer.embeddings.position_embeddings    # [1, 1+N, D]
            if posemb.size() == posemb_new.size():                          # G = N?
                self.transformer.embeddings.position_embeddings.copy_(posemb)   # copy
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                # 若使用 token 分类器，有一个 cls token，需要拆分
                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]   # 单独保留 CLS
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]  # 无 cls，全部为网格位置编码

                # 把一维的 N 个位置编码还原成 g_s × g_s 的二维网格，再做二维插值
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                # 在二维空间对每个通道 D 做缩放（插值）
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # 双线性插值
                # 摊平成 [1, N, D] 并与 CLS 重新拼回 [1, 1+N, D]
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            # --------------------------------------------------------------------------------
            # 逐层加载 Transformer Block 参数
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
            # 若为 hybrid（带 ResNetV2），加载其权重
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def np2th(weights, conv=False):
    """
    Possibly convert HWIO to OIHW.（重复定义）将 numpy 权重转为 torch.Tensor；conv=True 时执行 HWIO->OIHW。
    注：上方已定义同名函数，这里重复并不影响运行（后者会覆盖前者）。
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    """权重标准化卷积（Weight Standardization），有助于稳定训练。"""
    def forward(self, x):
        w = self.weight
        # 对每个输出通道做标准化：减均值、除以标准差
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    """封装 3x3 StdConv2d。"""
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    """封装 1x1 StdConv2d。"""
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    ResNetV2 的 pre-activation 瓶颈残差块（GN + ReLU 在卷积前）。
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4  # 中间通道数（瓶颈）

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!# 注意：步幅在 conv2
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.下采样分支（投影）：保持残差与主分支形状一致
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch 残差分支（可能包含下采样）
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch 主干分支：GN + ReLU + Conv
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        # 残差相加后再 ReLU
        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        """从预训练字典中加载本残差块的卷积与 GN 参数。"""
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """
    Implementation of Pre-activation (v2) ResNet mode.
    Pre-activation (v2) ResNet 主干的简化实现（root + 3 个 block）。
    - root：7×7 Conv（stride=2）+ GN + ReLU + 3×3 MaxPool（stride=2）
    - body：三个 block（每个 block 含若干 PreActBottleneck；block2/3 的首个 unit stride=2 做下采样）
    作为 ViT 的 hybrid 前端时，输出步幅固定为 16。

    """

    def __init__(self, block_units, width_factor):
        """
        block_units: 每个 block 内的 bottleneck 个数，例如 [3,4,9]
        width_factor: 宽度放大系数（影响通道数）
        """
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        # Stem（root）：7x7 卷积 + GN + ReLU + 3x3 最大池化
        # 说明：padding=0 保持与原实现一致（可能略减小特征图尺寸
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        # 3 个 stage，每个 stage 的第一个 unit 可能带下采样（stride=2）
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        return x

