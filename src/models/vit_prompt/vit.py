#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
带有 Prompt 的 ViT：遵循 VPT（Visual Prompt Tuning）默认设定的干净实现。
核心思想：在不（或少量）更新主干参数的情况下，为输入序列“前置”若干可训练的提示 token，
与图像 patch 的 token 一起送入 Transformer，从而实现参数高效的迁移/微调。
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

# 复用原始 ViT 的组件和配置：
# - CONFIGS: 各个模型类型（如 ViT-B/16）的结构配置字典
# - Transformer: 原始的 Transformer 编码器（包含 embeddings/encoder 等）
# - VisionTransformer: 带分类头的标准 ViT
# - np2th: numpy -> torch 的权重转换工具
from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")

class PromptedTransformer(Transformer):
    """
    在原始 Transformer 基础上，加入“前置（prepend）”的 Prompt token。[CLS] + [PROMPT × P] + [PATCH × N]
    仅支持：
      - LOCATION == "prepend"（提示 token 插在 CLS 之后、patch 之前）
      - INITIATION == "random"（随机初始化）
      - 不启用 Deep-Shared / 指定层数的 deep prompt（NUM_DEEP_LAYERS is None 且 not DEEP_SHARED）
    """

    def __init__(self, prompt_config, config, img_size, vis, prompt_init=None, prompt_init_provider=None):
        # 该实现的假设：只支持“prepend”与“random”初始化；不支持 deep-shared/局部层数选择
        assert prompt_config.LOCATION == "prepend"  # 只支持前置提示（插在 CLS 后、patch 前）
        assert prompt_config.INITIATION == "random" # 提示向量随机初始化（Xavier 风格区间）

        # 不支持你在 12 层里只挑“第 3、7、11 层”插入的那种部分层 deep-prompt
        # 支持的 deep 版本是：第 0 层用“前置 prompt”，从第 1 层到第 L-1 层“每层都用一份 deep-prompt”
        assert prompt_config.NUM_DEEP_LAYERS is None

        # 不支持所有层共用同一组提示向量（共享参数）,支持的是每一层都有自己的提示参数
        assert not prompt_config.DEEP_SHARED

        # 初始化父类（会构建 embeddings、encoder 等）
        super(PromptedTransformer, self).__init__(
            config, img_size, vis)
        self.prompt_config = prompt_config
        self.vit_config = config

        # 规范输入尺寸 & 取出 patch 大小，统一将尺寸转成二元组（H, W）
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        # 提示 token 数量（例如 5/10/...）
        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        # 对提示 token 可选的 dropout（训练期随机丢弃，增强鲁棒性）
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # （可选）把“低维 prompt”线性投影到 ViT 的隐藏维 D
        # if project the prompt embeddings .PROJECT 指定提示向量（prompt token）在参数里存储的维度 d
        # （可选）若 PROJECT = -1则不投影，若 PROJECT >-1，则先将 prompt 的低维嵌入线性 投影到 ViT 的 hidden_size
        # 若有一套预训练的低维 prompt、“同一套低维 prompt”跨不同隐藏维的 ViT 重用等需求时，可设置PROJECT =具体的d
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add # 仅在 prepend/add 场景有意义
            prompt_dim = self.prompt_config.PROJECT # prompt 的“存储维度”d
            self.prompt_proj = nn.Linear(prompt_dim, config.hidden_size)    # # 统一映射到 ViT 的 hidden_size D
            # Kaiming 正态初始化（fan_out 模式），有利于残差/卷积流
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size  # prompt 已经是 D 维，不用投影
            self.prompt_proj = nn.Identity()

        # runtime prompt provider (e.g., pre-ViT prompt distribution)
        self.prompt_init_provider = prompt_init_provider

        # initiate prompt:
        # ====== 初始化提示 token 参数 ====== 让Prompt的数值尺度和“图像 patch 的嵌入”同量级,做一个上下界
        if self.prompt_config.INITIATION == "random":
            # Xavier-uniform 风格的上下界，根据输入维度与 prompt_dim 计算 val = sqrt( 6 / ( fan_in + fan_out ) )
            # 目的：让随机初始化的提示向量和 patch 嵌入的量纲相近，便于两者在同一序列里被注意力网络一起处理
            # fan_in 近似为：每个 patch 的原始输入维度 = 3 * patch_h * patch_w（RGB 三通道 × patch 像素数）
            # fan_out 近似为：prompt_dim（提示向量的维度）
            # eg:patch_size = 16×16 = 256，3 * 256 = 768；设 prompt_dim = 192，则 val = sqrt(6/(768+192)) = sqrt(6/960) ≈ 0.079；
            # 初始化区间 [-0.079, 0.079]。如果 PROJECT < 0 直接用 D=768 做 prompt_dim，则 val = sqrt(6/(768+768)) ≈ 0.0625
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            # 初始化前置 prompt
            # 前置提示：形状 (1, n_prompt, prompt_dim)，前向时会 expand 到 (B, n_prompt, prompt_dim)
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))

            # xavier_uniform initialization or external seed
            if prompt_init is None:
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            else:
                self._seed_prompt(prompt_init, prompt_dim)

            # 初始化 deep prompt
            # Deep Prompt（若开启 Deep Prompt，会为每个中间层准备一组提示向量 token）：每个中间层再有一份 (P, d) 的提示向量，前向里会逐层替换那段 prompt
            if self.prompt_config.DEEP:  # noqa (保持原风格)
                total_d_layer = config.transformer["num_layers"]-1  # 不含第 0 层（与论文设定一致）
                # 第 0 层前，序列里已经有“前置 prompt”；之后从第 1 层开始到第 L-1 层（总计 L-1 个地方），每层再插入（或替换）一段 deep prompt
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        # [CLS] + [ prompt_proj(prompt) × P ] + [ PATCH × N ]   →  (B, 1+P+N, D)
        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        """
        将 prompt token 按“prepend”策略并入：eg:N=14×14=196；若 P=10，则总长 1+10+196=207
        输入：
          x: 原始图像张量 (B, C, H, W)
        流程：
          1) 提取不带 CLS/位置编码的 patch tokens: V_raw [B, N, D]
          2) 由 prompt_init_provider(V_raw) 生成提示，或使用已有 prompt 参数
          3) 重新构造 [CLS|V_raw]+pos，随后在 CLS 后插入提示，得到 (B, 1 + n_prompt + N, D)
        """

        B = x.shape[0]

        # 先拿到纯 patch 嵌入（无 CLS/pos）：V_raw
        patch_tokens = self.embeddings.forward_patches(x)  # (B, n_patches, hidden_dim)

        # 如果提供了运行时的 prompt_init_provider，则在每个 batch 里
        # 基于当前的 V_raw 生成提示 token，用作“初始化后的”提示。
        if self.prompt_init_provider is not None:
            prompt_tokens = self.prompt_init_provider(patch_tokens)
            if prompt_tokens.dim() == 2:
                prompt_tokens = prompt_tokens.unsqueeze(0)
            if prompt_tokens.shape[1] != self.num_tokens:
                raise ValueError(
                    f"prompt_init_provider returned shape {prompt_tokens.shape}, expected num_tokens={self.num_tokens}"
                )
            if prompt_tokens.shape[0] == 1 and B > 1:
                prompt_tokens = prompt_tokens.expand(B, -1, -1)
            elif prompt_tokens.shape[0] != B:
                raise ValueError(
                    f"prompt_init_provider batch {prompt_tokens.shape[0]} incompatible with input batch {B}"
                )
        else:
            prompt_tokens = self.prompt_embeddings

        # 重建带 CLS/位置编码的主序列
        x = self.embeddings.add_cls_and_pos(patch_tokens)  # (B, 1 + n_patches, hidden_dim)

        # 拼接： [CLS] + [PROMPT * n] + [PATCH * N]
        x = torch.cat((
                x[:, :1, :],    # 只取 CLS
                self.prompt_dropout(self.prompt_proj(prompt_tokens).expand(B, -1, -1)),
                x[:, 1:, :]     # 全部 patch token
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim) 最终形状: (B, 1 + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        """
        重写 nn.Module.train：
        训练时“只训练与 prompt 相关的模块”，其余模块（encoder/embeddings）切到 eval，
        以达到“冻结主干、仅更新提示”的效果。
        说明：
          - 这会让 encoder/embeddings 的 BN/Dropout 等处于评估态（禁用随机性）。
          - prompt_proj、prompt_dropout 仍在训练态，以便提示可正常学习。
        """
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training: 训练期：冻结主干
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
            if isinstance(self.prompt_init_provider, torch.nn.Module):
                self.prompt_init_provider.train(mode)
        else:
            # eval: 评估期：统一按 mode 设置
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        """
        深层提示（Deep Prompt）前向：第 0 层：直接用前置 prompt 的序列过一层 encoder
        第 1…L-1 层：在每层输入前再次插入一段该层专属的 deep_prompt_embeddings[i-1]，并用它替换原先那段 prompt，使序列长度保持 1+P+N 不变。
        [CLS] + [DeepPrompt(i) × P] + [PATCH × N]"""
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                # 第 0 层：直接过一层编码器（此时序列已是 [CLS][PROMPT* n][PATCH* N]）
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                # 其余各层：先将上一层输出再插入“深层 prompt”，再过本层
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                    # 维持总长度不变：把上一层的 prompt 段替换成“深层 prompt”
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],                           # 保留 CLS
                        deep_prompt_emb,                                   # 插入新的深层 prompt
                        hidden_states[:, (1 + self.num_tokens):, :]        # 跳过旧的 prompt 段，保留后续 patch
                    ), dim=1)


                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)  # 最后层的 LayerNorm
        return encoded, attn_weights

    def forward(self, x):
        """
        标准前向：
          - 先并入前置 prompt
          - 若开启 deep prompt，则逐层替换；否则直接一次性送入 encoder
          - 返回编码后的序列与可选的注意力权重（由 vis 决定）
        """
        # 1) prepend prompt 并入前置提示
        embedding_output = self.incorporate_prompt(x)
        # 2) deep prompt（可选）
        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights

    def _seed_prompt(self, prompt_init, prompt_dim):
        """Use external prompt_init tensor to seed prompt embeddings."""
        init = prompt_init.detach()
        if init.dim() == 2:
            init = init.unsqueeze(0)
        if init.shape[1:] != (self.num_tokens, prompt_dim):
            raise ValueError(
                f"prompt_init has shape {init.shape}, expected (1, {self.num_tokens}, {prompt_dim})"
            )
        with torch.no_grad():
            self.prompt_embeddings.copy_(init)



class PromptedVisionTransformer(VisionTransformer):
    """
    在标准 VisionTransformer 外壳下，用 PromptedTransformer 替换其内部的 transformer，
    - 复用原有 forward 接口/分类头写法（取 CLS 后线性分类）；
    - 仅改变编码阶段（加入提示 token），其余训练/评测流程不受影响。
    """
    def __init__(self, prompt_cfg, model_type,img_size=224, num_classes=21843, vis=False, prompt_init=None, prompt_init_provider=None):        # 当前实现只支持原生的 CLS 池化方式（original）
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        # 先按普通 ViT 初始化（构造 embeddings、head 等）
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg

        # 取出结构规格（如 hidden_size、层数、patch 大小等）
        vit_cfg = CONFIGS[model_type]
        # 核心替换：把内部的 transformer 用“带 Prompt 的”版本替换
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis,
            prompt_init=prompt_init, prompt_init_provider=prompt_init_provider,
        )

    def forward(self, x, vis=False):
        """
        前向流程：
          1) transformer(x) → 得到编码后的整段序列 encoded（含 CLS + prompt + patch）
          2) 取 CLS 位置的特征 x[:, 0] 作为全局表示
          3) 过分类头 self.head 得到 logits
          4) 若 vis=True，则同时返回注意力权重以便可视化/分析
        """
        x, attn_weights = self.transformer(x)

        # 取 CLS token（位置 0）作为全局表征
        x = x[:, 0]

        # 线性分类头 -> logits（未做 softmax）
        logits = self.head(x)
        if not vis:
            return logits
        return logits, attn_weights
