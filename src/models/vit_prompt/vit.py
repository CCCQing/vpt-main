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
from typing import Optional

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout, LayerNorm, Linear
from scipy import ndimage

# 复用原始 ViT 的组件和配置：
# - CONFIGS: 各个模型类型（如 ViT-B/16）的结构配置字典
# - Transformer: 原始的 Transformer 编码器（包含 embeddings/encoder 等）
# - VisionTransformer: 带分类头的标准 ViT
# - np2th: numpy -> torch 的权重转换工具
from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")

class SharedConceptAligner(nn.Module):
    """
    SharedConceptAligner：共享概念基对齐模块

    作用（对应你写的那部分公式）：
    1）将类级语义 S_raw（属性向量/类原型）投影到与 ViT hidden_size 一致的空间；
    2）引入 K 个“共享概念槽” R ∈ R^{K×D}，对语义和视觉分别做注意力：
        - 语义→R：得到概念化语义 R_S；
        - 视觉→R：得到概念化视觉 R_V；
    3）再用 R_S 作为 Query，R_V 作为 Key/Value 做一次交叉注意力，得到与当前样本相关的
       视觉补充信息 v_hat；
    4）通过 MLP + 残差门控 λ 得到融合后的语义 S^#，后续作为每层 patch→semantic cross-attention 的语义输入。
    """

    def __init__(
        self,
        hidden_size: int,      # ViT 的 hidden_size，对应 R、语义、视觉的统一维度 D
        num_slots: int,        # 共享概念槽个数 K
        num_heads: int,        # 多头注意力 head 数，要求 hidden_size 能整除 num_heads
        dropout: float = 0.0,  # 注意力/MLP 的 dropout
        lambda_init: float = 1.0,  # λ 的初始值（融合残差的缩放因子）
        use_layer_norm: bool = True,  # 是否对输出做 LayerNorm
        proj_norm: bool = True,       # 是否对 semantic_proj 之后的语义做 LayerNorm
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # 每个 head 的维度 d = D / H
        self.head_dim = hidden_size // num_heads
        # 注意力缩放因子 1/sqrt(d)
        self.scale = self.head_dim ** -0.5

        # 共享概念槽 R：形状 [K, D]，在所有样本间共享的一组“概念原型”
        # 初始化时用 N(0, 1/√D) 这样的尺度，避免数值过大
        self.concept_slots = nn.Parameter(
            torch.randn(num_slots, hidden_size) * (hidden_size ** -0.5)
        )

        # 语义投影层：将输入语义 dim_s → hidden_size
        # 使用 lazy 构建的方式，是因为不同数据集语义维度可能不同（AwA2 85, CUB 312 等）
        self.semantic_proj: Optional[nn.Linear] = None
        # 语义投影后的归一化：让不同类/样本的语义分布更稳定
        self.semantic_proj_norm = LayerNorm(hidden_size, eps=1e-6) if proj_norm else nn.Identity()

        # —— 第 1 阶段：语义 / 视觉 → 槽 R 的注意力 —— #
        # 语义侧 Query：Q_s
        self.query_semantic = Linear(hidden_size, hidden_size)
        # 视觉侧 Query：Q_v
        self.query_visual = Linear(hidden_size, hidden_size)
        # 槽 R 的 Key/Value：K_R, V_R（对 R 做线性变换）
        self.key_slots = Linear(hidden_size, hidden_size)
        self.value_slots = Linear(hidden_size, hidden_size)

        # —— 第 2 阶段：R_S ↔ R_V 的交叉注意力 —— #
        # 这里以 R_S 为 Query，R_V 为 Key/Value
        self.cross_query = Linear(hidden_size, hidden_size)
        self.cross_key = Linear(hidden_size, hidden_size)
        self.cross_value = Linear(hidden_size, hidden_size)

        # 残差 MLP：输入是 [r_s || v_hat]（拼接，维度 2D），输出 D 维增量 Δ
        hidden_mlp = hidden_size * 2
        self.delta_mlp = nn.Sequential(
            Linear(hidden_size * 2, hidden_mlp),
            nn.GELU(),
            Dropout(dropout),
            Linear(hidden_mlp, hidden_size),
            Dropout(dropout),
        )
        # U s_1：对基准语义做线性映射，用于 Δ = MLP([r_s||v_hat]) - U r_s
        self.skip_proj = Linear(hidden_size, hidden_size)
        # 通道维度的 λ 门控向量：形状 [D]，逐维缩放 Δ
        self.lambda_gate = nn.Parameter(torch.full((hidden_size,), lambda_init))
        # 输出层归一化：对应 S^# = LN(s_1 + λ ∘ Δ)
        self.out_norm = LayerNorm(hidden_size, eps=1e-6) if use_layer_norm else nn.Identity()

        # 注意力权重的 dropout
        self.attn_dropout = Dropout(dropout)

    def _build_semantic_proj(self, semantic_dim: int, device: torch.device):
        """
        Lazy 构造语义投影层：
        - 第一次 forward 时，根据语义维度 semantic_dim 创建 Linear(semantic_dim, hidden_size)
        - 之后复用同一层，兼容不同数据集的属性维度
        """
        if self.semantic_proj is None:
            self.semantic_proj = Linear(semantic_dim, self.hidden_size)
            self.semantic_proj = self.semantic_proj.to(device)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入 [B, L, D] reshape 成多头注意力格式 [B, H, L, d]：
        - 先 view 成 [B, L, H, d]
        - 再 permute 到 [B, H, L, d]
        """
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        标准多头注意力计算：
        输入：
          query: [B, H, L_q, d]
          key:   [B, H, L_k, d]
          value: [B, H, L_k, d]
        输出：
          context: [B, L_q, D]，即将多头结果拼回 D 维
        """
        # [B, H, L_q, L_k]
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        # softmax + dropout 得到注意力权重
        attn_probs = self.attn_dropout(torch.softmax(attn_scores, dim=-1))
        # [B, H, L_q, d]
        context = torch.matmul(attn_probs, value)
        # 还原回 [B, L_q, D]：先换回 [B, L_q, H, d] 再 view
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.hidden_size,)
        return context.view(*new_shape)

    def encode_semantics_only(self, semantics: torch.Tensor) -> torch.Tensor:
        """
        仅用于“类语义 → R 空间类原型”的编码（给分类头用）。
        不依赖具体图像，只依赖属性向量。
        输入：所有类的属性矩阵 semantics，维度大概是 [C, d_s]，比如 CUB 就是 [200, 312]；
        输出：所有类在 R 空间下的表达，维度 [C, D]，比如 ViT-B/16 是 [200, 768]

        Args:
            semantics: [C, d_s] 或 [C, 1, d_s]，C 为类别数

        Returns:
            [C, D]，每个类别在 R 空间下的类原型向量 s_c
        """
        # 统一成 [C, 1, d_s]
        if semantics.dim() == 1:
            semantics = semantics.unsqueeze(0)
        if semantics.dim() == 2:
            semantics = semantics.unsqueeze(1)

        device = self.concept_slots.device
        semantics = semantics.to(device)
        self._build_semantic_proj(semantics.size(-1), device=device)
        # 属性投影到 D 维并做 LN
        semantic_tokens = self.semantic_proj(semantics)
        semantic_tokens = self.semantic_proj_norm(semantic_tokens)

        num_classes = semantic_tokens.size(0)
        # 槽 R 扩展到 batch 维：[C, K, D]
        slots = self.concept_slots.unsqueeze(0).expand(num_classes, -1, -1)
        # 槽的 K/V：[C, H, K, d]
        slot_keys = self._transpose_for_scores(self.key_slots(slots))
        slot_values = self._transpose_for_scores(self.value_slots(slots))

        # 语义 Query：[C, H, 1, d]
        sem_query = self._transpose_for_scores(self.query_semantic(semantic_tokens))
        # r_s: [C, 1, D] → squeeze 成 [C, D]
        r_s = self._attention(sem_query, slot_keys, slot_values)

        return r_s.squeeze(1)

    def forward(self, patch_tokens: torch.Tensor, semantics: torch.Tensor) -> torch.Tensor:
        """
        输入：
          patch_tokens: [B, N, D]，来自 embeddings.forward_patches 的视觉 patch tokens
          semantics:    [B, d_s] / [B, 1, d_s] / [d_s]，类级属性或语义原型

        输出：
          fused: [B, D]，融合视觉信息后的共享语义 S^#
        """
        # 统一语义的形状：确保为 [B, 1, d_s]
        if semantics.dim() == 1:
            # 单样本 [d_s] → [1, d_s]
            semantics = semantics.unsqueeze(0)
        if semantics.dim() == 2:
            # [B, d_s] → [B, 1, d_s]
            semantics = semantics.unsqueeze(1)  # [B, 1, d_s]

        # 构建语义投影层（第一次调用时）
        self._build_semantic_proj(semantics.size(-1), device=patch_tokens.device)
        # 语义投影到 D 维并归一化：S_raw → S̃_raw
        semantic_tokens = self.semantic_proj(semantics)
        semantic_tokens = self.semantic_proj_norm(semantic_tokens)

        # ====== 阶段 1：语义/视觉 → 槽 R 的注意力，得到 R_S, R_V ======
        # 批大小 B，用于将共享槽 R 扩展到 batch 维度.  此处做的是构造R 的 batch 视图
        B, _, _ = patch_tokens.shape
        # slots: [B, K, D]，所有样本共享参数，但在 batch 维度做了 expand
        slots = self.concept_slots.unsqueeze(0).expand(B, -1, -1)  # [batch数, R槽数num_slots, hidden_size 768 D]

        # 槽的 Key/Value：K_R, V_R 形状 [B, H, K, d]
        slot_keys = self._transpose_for_scores(self.key_slots(slots))
        slot_values = self._transpose_for_scores(self.value_slots(slots))

        # —— 语义→R：R_S —— #
        # Q_s: [B, H, 1, d]     D = 768：ViT hidden size ;H = 8：注意力头数 ;d = D / H = 96：每个 head 的维度
        sem_query = self._transpose_for_scores(self.query_semantic(semantic_tokens))
        # 计算r_s: [B, 1, D]，每个样本的“概念化语义”
        r_s = self._attention(sem_query, slot_keys, slot_values)  # [B, 1, D]

        # —— 视觉→R：R_V —— #
        # Q_v: [B, H, N, d]
        vis_query = self._transpose_for_scores(self.query_visual(patch_tokens))
        # 计算r_v: [B, N, D]，每个 patch 经 R 重新表达后的“概念化视觉 token”
        r_v = self._attention(vis_query, slot_keys, slot_values)  # [B, N, D]

        # ====== 阶段 2：R_S ↔ R_V 交叉注意力，得到视觉补充 v_hat ======
        # 以 r_s 为 Query，r_v 为 Key/Value
        cross_q = self._transpose_for_scores(self.cross_query(r_s))
        cross_k = self._transpose_for_scores(self.cross_key(r_v))
        cross_v = self._transpose_for_scores(self.cross_value(r_v))
        # v_hat: [B, 1, D]，表示“与当前语义相关的那部分视觉信息”
        v_hat = self._attention(cross_q, cross_k, cross_v)  # [B, 1, D]

        # ====== 阶段 3：残差融合，得到 S^# ======
        # 以概念化语义 R_S 作为残差基准，而非原始语义 S_raw
        base_semantics = r_s
        # 拼接 r_s 与 v_hat：[B, 1, 2D] → [B, 1, D]，再减去 U r_s
        delta = self.delta_mlp(torch.cat([base_semantics, v_hat], dim=-1)) - self.skip_proj(base_semantics)
        # 通道门控 λ：逐维缩放 Δ，得到 R_S + λ ∘ Δ（不直接回落到原始语义）
        fused = base_semantics + self.lambda_gate * delta
        # 最终归一化，得到 S^#（去掉长度 1 维度，输出 [B, D]）
        fused = self.out_norm(fused)
        return fused.squeeze(1)

class PromptedTransformer(Transformer):
    """
    在原始 Transformer 编码器基础上加入“前置 Prompt token”的版本。

    序列形式：
        [CLS] + [PROMPT × P] + [PATCH × N]

    限制条件（当前实现只支持最常用的一种设定）：
    - LOCATION == "prepend"：提示 token 只能插入在 CLS 之后、patch tokens 之前；
    - INITIATION == "random"：提示向量采用随机初始化（Xavier 类似的均匀分布）；
    - 不支持：
        * prompt_config.NUM_DEEP_LAYERS 非 None 的“只在部分层插入 deep prompt”；
        * prompt_config.DEEP_SHARED = True 的“所有层共用一份 deep prompt”。

    同时支持：
    - 常规前置 prompt（第 0 层输入前插入）；
    - Deep Prompt：在每个中间层之前都插入一份该层专属的 prompt，并替换掉上一层的 prompt 段。
    """

    def __init__(self, prompt_config, config, img_size, vis, prompt_init=None, prompt_init_provider=None):

        assert prompt_config.LOCATION == "prepend"  # 只支持前置提示（插在 CLS 后、patch 前）
        assert prompt_config.INITIATION == "random" # 提示向量随机初始化（Xavier 风格区间）

        # 不支持你在 12 层里只挑“第 3、7、11 层”插入的那种部分层 deep-prompt
        # 支持的 deep 版本是：第 0 层用“前置 prompt”，从第 1 层到第 L-1 层“每层都用一份 deep-prompt”
        assert prompt_config.NUM_DEEP_LAYERS is None

        # 不支持所有层共用同一组提示向量（共享参数）,支持的是每一层都有自己的提示参数
        assert not prompt_config.DEEP_SHARED

        # 初始化父类（会构建 embeddings、encoder 等）
        # semantic_dim：若你在 encoder 层里加入了“语义 cross-attention”，可以通过这个维度决定语义向量的投影维度
        # 通常等于数据集的属性维度（如 312）。
        semantic_dim = getattr(prompt_config, "SEMANTIC_DIM", None)

        # 读取共享概念基相关配置（SEMANTIC_CONCEPT 子节点）
        concept_cfg = getattr(prompt_config, "SEMANTIC_CONCEPT", None)
        if concept_cfg is not None and getattr(concept_cfg, "ENABLE", False):
            # 若启用共享概念基模块，则强制 semantic_dim = hidden_size
            # 这样 encoder 层里的语义向量就直接用 S^#（已是 D 维），无需额外映射
            semantic_dim = config.hidden_size

        # 调用父类 Transformer 的初始化，并告知 semantic_dim（便于其内部创建语义相关投影）
        super(PromptedTransformer, self).__init__(
            config, img_size, vis, semantic_dim=semantic_dim)

        # 保存 prompt 配置和 vit 配置
        self.prompt_config = prompt_config
        self.vit_config = config

        # 若启用了共享概念基模块，则在此构建 SharedConceptAligner
        self.semantic_concept = None
        if concept_cfg is not None and getattr(concept_cfg, "ENABLE", False):
            self.semantic_concept = SharedConceptAligner(
                hidden_size=config.hidden_size,
                num_slots=concept_cfg.NUM_SLOTS,
                num_heads=concept_cfg.NUM_HEADS,
                dropout=concept_cfg.DROPOUT,
                lambda_init=concept_cfg.LAMBDA_INIT,
                use_layer_norm=concept_cfg.USE_LAYER_NORM,
                proj_norm=concept_cfg.PROJ_NORM,
            )

        # 规范输入尺寸 & 取出 patch 大小，统一将尺寸转成二元组（H, W）
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        # 提示 token 数量（例如 5/10/...）/ patch 尺寸为 (H, W) 形式
        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        # 对提示 token 可选的 dropout（训练期随机丢弃，增强鲁棒性）
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # ====== 提示 token 维度设定 ======
        # 直接在 ViT hidden_size 维度上维护/生成 prompt，同时保留可训练的 prompt 映射层
        prompt_dim = config.hidden_size
        self.prompt_proj = Linear(prompt_dim, config.hidden_size)
        if prompt_dim == config.hidden_size:
            # 维度一致时初始化为接近恒等映射，梯度主要落在可训练的映射层
            with torch.no_grad():
                self.prompt_proj.weight.copy_(torch.eye(config.hidden_size))
                if self.prompt_proj.bias is not None:
                    self.prompt_proj.bias.zero_()

        # 在前向过程中根据视觉 token 生成 prompt 的模块，比如你自己的“提示分布网络”，输入 patch_tokens，输出 prompt_tokens
        self.prompt_init_provider = prompt_init_provider

        # 是否完全依赖“运行时分布”生成提示，而不使用任何可学习的 prompt 参数
        self.runtime_prompt_only = getattr(
            self.prompt_config, "DISTRIBUTION_ONLY", False)
        self.detach_prompt_grad = getattr(
            self.prompt_config, "DETACH_PROMPT_GRAD", False)

        if self.runtime_prompt_only:
            # 既然说“只靠分布/生成器”，那就必须提供一个 prompt_init_provider
            if self.prompt_init_provider is None:
                raise ValueError(
                    "PROMPT.DISTRIBUTION_ONLY=True requires a prompt_init_provider"
                )
            # 并且不允许同时给一个静态初始 prompt（两者矛盾）
            if prompt_init is not None:
                raise ValueError(
                    "prompt_init cannot be provided when DISTRIBUTION_ONLY=True"
                )
        self.use_learned_prompt_params = not self.runtime_prompt_only

        # ====== 初始化提示 token 参数 ======
        if self.prompt_config.INITIATION == "random":
            # Xavier-uniform 风格的上下界，根据输入维度与 prompt_dim 计算 val = sqrt( 6 / ( fan_in + fan_out ) )
            # 目的：让随机初始化的提示向量和 patch 嵌入的量纲相近，便于两者在同一序列里被注意力网络一起处理
            # fan_in 近似为：每个 patch 的原始输入维度 = 3 * patch_h * patch_w（RGB 三通道 × patch 像素数）
            # fan_out 近似为：prompt_dim（提示向量的维度）
            # eg:patch_size = 16×16 = 256，3 * 256 = 768；设 prompt_dim = 192，则 val = sqrt(6/(768+192)) = sqrt(6/960) ≈ 0.079；
            # 初始化区间 [-0.079, 0.079]。如果 PROJECT < 0 直接用 D=768 做 prompt_dim，则 val = sqrt(6/(768+768)) ≈ 0.0625
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            if self.use_learned_prompt_params:
                # -------- 前置 prompt（第 0 层使用） --------
                # 形状：[1, P, prompt_dim]，前向时会扩展到 [B, P, prompt_dim]
                self.prompt_embeddings = nn.Parameter(
                    torch.zeros(1, num_tokens, prompt_dim))

                # 若未提供外部 prompt_init，则采用均匀分布随机初始化
                if prompt_init is None:
                    nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                else:
                    # 若提供了外部 prompt_init 张量，则用其初始化（常见于迁移/微调时）
                    self._seed_prompt(prompt_init, prompt_dim)
            else:
                # 完全不使用学习参数 prompt_embeddings，而是全部由 prompt_init_provider 提供（提示分布）
                self.prompt_embeddings = None

            # -------- Deep Prompt（中间层使用） --------
            # 若开启 Deep Prompt：在每个中间层（除第 0 层）前插入一段该层专属 prompt
            if self.prompt_config.DEEP:  # noqa (保持原风格)
                # 总的中间层数 = 总层数 - 1（第 0 层只用前置 prompt，不算 deep）
                total_d_layer = config.transformer["num_layers"]-1  # 不含第 0 层（与论文设定一致）
                # deep_prompt_embeddings: [L-1, P, prompt_dim]
                # 其中第 i-1 份对应 encoder 第 i 层（i 从 1 到 L-1）
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # 均匀分布初始化 deep prompt
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        # [CLS] + [ PROMPT × P ] + [ PATCH × N ]   →  (B, 1+P+N, D)
        else:
            raise ValueError("Other initiation scheme is not supported")

        # 是否冻结原始 prompt 嵌入参数，使梯度主要落在分布网络等其他支路上
        self.freeze_embeddings = getattr(self.prompt_config, "FREEZE_EMBEDDINGS", True)
        if self.freeze_embeddings:
            if self.prompt_embeddings is not None:
                self.prompt_embeddings.requires_grad = False
            if hasattr(self, "deep_prompt_embeddings"):
                self.deep_prompt_embeddings.requires_grad = False

        # —— Layer-wise prompt evolution：第 1…L-1 层通过线性层更新上一层的 prompt —— #
        num_layers = config.transformer["num_layers"]
        hidden_size = config.hidden_size
        self.prompt_update_layers = nn.ModuleList([
            Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
        ])
        # 近似恒等初始化，确保初始行为稳定
        with torch.no_grad():
            for layer in self.prompt_update_layers:
                eye = torch.eye(hidden_size, device=layer.weight.device, dtype=layer.weight.dtype)
                layer.weight.copy_(eye)
                if layer.bias is not None:
                    layer.bias.zero_()

    def incorporate_prompt(self, x, semantics=None):
        """
        将 prompt token 按“prepend”策略并入：eg:N=14×14=196；若 P=10，则总长 1+10+196=207
        输入：
          x: 原始图像张量 (B, C, H, W)
        主要步骤：
          1) 使用 embeddings.forward_patches 提取纯 patch tokens：V_raw，形状 [B, N, D]
          2) 生成 prompt_tokens：
             - 若提供 prompt_init_provider，则基于 V_raw 动态生成；
             - 否则使用固定的 self.prompt_embeddings。
          3) 调用 embeddings.add_cls_and_pos(V_raw) 重新构造 [CLS|PATCH] + pos 编码；
          4) 在 CLS 之后插入 prompt_tokens（先投影再 dropout），得到：
             [CLS] + [PROMPT × P] + [PATCH × N]，形状 [B, 1+P+N, D]
        """

        B = x.shape[0]

        # 1) 提取纯 patch 嵌入：不包含 CLS / 位置编码：V_raw
        patch_tokens = self.embeddings.forward_patches(x)  # (B, n_patches, hidden_dim)

        # —— 新增：通过共享概念基模块对语义进行“预对齐/增强” —— #
        refined_semantics = semantics
        if semantics is not None and self.semantic_concept is not None:
            # patch_tokens: 视觉侧输入 V_raw
            # semantics:    类级属性 S_raw
            # 输出 refined_semantics: S^#，形状 [B, D]
            refined_semantics = self.semantic_concept(patch_tokens, semantics)

        # 2) 生成 prompt_tokens
        if self.prompt_init_provider is not None:
            # 由“提示分布网络”或其他模块，基于 patch_tokens 生成 prompt
            provider_out = self.prompt_init_provider(patch_tokens)
            # 兼容返回 (prompts, stats) 的形式，仅取 prompts
            if isinstance(provider_out, tuple):
                prompt_tokens = provider_out[0]
            elif isinstance(provider_out, dict) and "prompts" in provider_out:
                prompt_tokens = provider_out["prompts"]
            else:
                prompt_tokens = provider_out

            if not torch.is_tensor(prompt_tokens):
                raise TypeError(
                    "prompt_init_provider must return a Tensor or (Tensor, stats), "
                    f"got {type(prompt_tokens)}"
                )
            # 若输出为 [P, D]，则视为单样本模板，扩展 batch 维
            if prompt_tokens.dim() == 2:
                prompt_tokens = prompt_tokens.unsqueeze(0)

            # 检查 prompt 数量是否与配置一致
            if prompt_tokens.shape[1] != self.num_tokens:
                raise ValueError(
                    f"prompt_init_provider returned shape {prompt_tokens.shape}, expected num_tokens={self.num_tokens}"
                )

            # 若 provider 只返回单个 batch 的 prompt，而真实 batch_size > 1，则复制扩展
            if prompt_tokens.shape[0] == 1 and B > 1:
                prompt_tokens = prompt_tokens.expand(B, -1, -1)
            elif prompt_tokens.shape[0] != B:
                # 若 provider 输出的 batch 数与输入不一致，则直接报错
                raise ValueError(
                    f"prompt_init_provider batch {prompt_tokens.shape[0]} incompatible with input batch {B}"
                )
        else:
            # 若没有 provider，则必须使用学习参数 prompt_embeddings
            if self.prompt_embeddings is None:
                raise RuntimeError(
                    "Prompt embeddings are disabled but no prompt_init_provider "
                    "is available. Set PROMPT.DISTRIBUTION_ONLY=False or "
                    "supply a provider."
                )
            prompt_tokens = self.prompt_embeddings
            if self.detach_prompt_grad or self.freeze_embeddings:
                prompt_tokens = prompt_tokens.detach()# 其余各层
                # 冻结 prompt 参数表，只训练映射层（prompt_proj）及后续模块
                prompt_tokens = prompt_tokens.detach()
        if prompt_tokens.shape[-1] != self.vit_config.hidden_size:
            raise ValueError(
                f"Prompt feature dim {prompt_tokens.shape[-1]} incompatible with hidden_size {self.vit_config.hidden_size}"
            )
        # 3) 重建带 CLS/位置编码的主序列：[CLS|PATCH] + pos
        x = self.embeddings.add_cls_and_pos(patch_tokens)  # (B, 1 + n_patches, hidden_dim)

        # 4) 拼接序列：[CLS] + [PROMPT × P] + [PATCH × N]
        prompt_tokens = self.prompt_proj(prompt_tokens)
        x = torch.cat((
                x[:, :1, :],    # 只取 CLS
                # 直接使用与 hidden_size 对齐的 prompt，并做 dropout
                self.prompt_dropout(prompt_tokens.expand(B, -1, -1)),
                x[:, 1:, :]     # 其余 patch token
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim) 最终形状: (B, 1 + n_prompt + n_patches, hidden_dim)

        return x, refined_semantics

    def train(self, mode=True):
        """
        重写 nn.Module.train，用于控制“只训练 prompt 相关模块，冻结主干”。

        行为：
        - 当 mode=True（训练模式）：
            * encoder / embeddings 置为 eval()（冻结、关闭 Dropout/BN 的随机性）；
            * prompt_dropout 仍处于 train() 状态；
            * 若 prompt_init_provider 是 nn.Module，也会根据 mode 设置。
        - 当 mode=False（评估模式）：
            * 对所有子模块调用 module.train(False)，统一切到 eval。
        """
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training: 训练期：冻结主干
            self.encoder.eval()
            self.embeddings.eval()

            # 只让 prompt 相关层保持 train 状态
            self.prompt_proj.train(mode)
            self.prompt_dropout.train()
            self.prompt_update_layers.train(mode)

            # 若 provider 本身是一个可学习模块，则也遵循 mode 设置
            if isinstance(self.prompt_init_provider, torch.nn.Module):
                self.prompt_init_provider.train(mode)
        else:
            # 评估/推理时：所有子模块统一跟随 mode
            for module in self.children():
                module.train(mode)

            if isinstance(self.prompt_init_provider, torch.nn.Module):
                self.prompt_init_provider.train(mode)

    def forward_deep_prompt(self, embedding_output, semantics=None):
        """
        Deep Prompt 模式下的前向传播。
        - 若未启用共享概念基模块，则为原始语义投影（例如 Linear(att_dim→D) 后的结果）；
        - 若启用了 SharedConceptAligner，则为 S^#（融合视觉后的共享语义），
          会在每层 encoder.layer[i](..., semantics, ...) 内部被用于 patch→semantic cross-attention。

        输入：
          - embedding_output: 经过 incorporate_prompt 的序列 (B, 1+P+N, D)
          - semantics:        语义模态特征（可选），用于你在 encoder 中加入的 cross-attention

        机制：
          - 第 0 层：直接对 [CLS + 前置 PROMPT + PATCH] 做 self-attention 与 MLP；
          - 第 1 … L-1 层：
              * 从上一层输出中截取 prompt 段，通过该层专属 Linear 做“prompt 演化”；
              * 用演化后的 prompt 替换序列中的 prompt 段，再过 encoder.layer[i]。
        """
        attn_weights: list = []           # 按需保存每层的注意力权重（vis=True 时有效）
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                # 第 0 层：使用 provider/表初始化得到的 [CLS|P^0|PATCH] 序列，梯度流向 provider
                hidden_states, weights, semantics = self.encoder.layer[i](hidden_states, semantics, self.num_tokens)
            else:
                # 1) 取出上一层输出中的 prompt 段（长度固定为 self.num_tokens）
                prev_prompt = hidden_states[:, 1:1 + self.num_tokens, :]

                # 2) 通过该层专属的线性层演化 prompt，梯度落在 prompt_update_layers
                evolved_prompt = self.prompt_update_layers[i - 1](prev_prompt)
                evolved_prompt = self.prompt_dropout(evolved_prompt)

                # 3) 重组序列：[CLS | P^i | PATCH]，保持长度 1 + P + N 不变
                hidden_states = torch.cat(
                    (
                        hidden_states[:, :1, :],  # CLS
                        evolved_prompt,  # 更新后的 prompt
                        hidden_states[:, 1 + self.num_tokens:, :],  # PATCH 段
                    ),
                    dim=1,)

                # 4) 经过第 i 层 Transformer block（仍支持语义 cross-attn）
                hidden_states, weights, semantics = self.encoder.layer[i](hidden_states, semantics, self.num_tokens)

            if self.encoder.vis:
                attn_weights.append(weights)

        # 最终输出前做一次 LayerNorm
        encoded = self.encoder.encoder_norm(hidden_states)  # 最后层的 LayerNorm
        return encoded, attn_weights

    def forward_deep_prompt_with_affinity(self, embedding_output, affinity_config, semantics=None):
        """
        带亲和分支的 Deep Prompt 前向：与 forward_deep_prompt 平行。

        对应关系：
          - forward_deep_prompt           ↔ forward_deep_prompt_with_affinity
          - encoder.layer[i].forward      ↔ encoder.layer[i].forward_with_affinity
          - forward/forward_with_affinity 同样共享 incorporate_prompt 生成的 [CLS|P|PATCH]

        返回:
          encoded:     LayerNorm 后的最终序列
          attn_weights: 可视化用注意力权重（vis=True 时）
          affinities:   每层的亲和矩阵列表，来自 compute_affinity（逐层收集，方便上层按需挑选做对齐损失或调试）
        """
        attn_weights: list = []
        affinities: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                # 第 0 层：直接使用 provider/表生成的 prompt 序列，并走带亲和的前向
                hidden_states, weights, affinity, semantics = self.encoder.layer[i].forward_with_affinity(
                    hidden_states, affinity_config, semantics, self.num_tokens
                )
            else:
                prev_prompt = hidden_states[:, 1:1 + self.num_tokens, :]
                evolved_prompt = self.prompt_update_layers[i - 1](prev_prompt)
                evolved_prompt = self.prompt_dropout(evolved_prompt)

                hidden_states = torch.cat(
                    (hidden_states[:, :1, :],evolved_prompt,hidden_states[:, 1 + self.num_tokens:, :],),dim=1,)

                hidden_states, weights, affinity, semantics = self.encoder.layer[i].forward_with_affinity(
                    hidden_states, affinity_config, semantics, self.num_tokens
                )

            if self.encoder.vis:
                attn_weights.append(weights)
            affinities.append(affinity)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights, affinities
    def forward(self, x, semantics=None):
        """
        标准前向：
        - 输入 semantics 为 batch 的类级语义（属性向量），形状一般为 [B, att_dim]；
        - 在 incorporate_prompt 中：
            * 先从 x 提取 patch_tokens；
            * 若启用 SharedConceptAligner，则将 (patch_tokens, semantics) 映射为 refined_semantics=S^#；
        - 后续 encoder / forward_deep_prompt 使用的 semantics 实际上就是 refined_semantics。
        流程：
          1) 调用 incorporate_prompt(x) 将 prompt 合入输入序列；
          2) 若启用 Deep Prompt，则调用 forward_deep_prompt 做多层深度提示；
             否则直接将序列送入 encoder；
          3) 返回编码后的完整 token 序列 encoded 和可选的 attn_weights。
        """
        # 1) prepend prompt（在 CLS 后插入提示）
        embedding_output, semantics = self.incorporate_prompt(x, semantics)

        # 2) deep prompt（可选）
        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output, semantics)
        else:
            # 若不使用 Deep Prompt，则直接将整个序列送入 encoder
            encoded, attn_weights = self.encoder(embedding_output, semantics, self.num_tokens)

        return encoded, attn_weights

    def forward_with_affinity(self, x, affinity_config, semantics=None):
        """
        带亲和矩阵输出的前向，与 forward 平行：

        - incorporate_prompt 提供 [CLS|P|PATCH]
        - 若启用 Deep Prompt，则调用 forward_deep_prompt_with_affinity
        - 否则走 encoder.forward_with_affinity
        """
        embedding_output, semantics = self.incorporate_prompt(x, semantics)

        if self.prompt_config.DEEP:
            encoded, attn_weights, affinities = self.forward_deep_prompt_with_affinity(
                embedding_output, affinity_config, semantics
            )
        else:
            encoded, attn_weights, affinities = self.encoder.forward_with_affinity(
                embedding_output, affinity_config, semantics, self.num_tokens
            )

        return encoded, attn_weights, affinities

    def _seed_prompt(self, prompt_init, prompt_dim):
        """
        使用外部提供的 prompt_init 张量对 prompt_embeddings 进行初始化。

        要求：
        - prompt_init 至少为 2 维（P, d）或 3 维（1, P, d）；
        - 其中 P 必须等于 self.num_tokens，d 必须等于 prompt_dim。
        """
        init = prompt_init.detach()
        if init.dim() == 2:
            init = init.unsqueeze(0)
        if init.shape[1:] != (self.num_tokens, prompt_dim):
            raise ValueError(
                f"prompt_init has shape {init.shape}, expected (1, {self.num_tokens}, {prompt_dim})"
            )

        # 直接拷贝到可学习参数中
        with torch.no_grad():
            self.prompt_embeddings.copy_(init)



class PromptedVisionTransformer(VisionTransformer):
    """
    在标准 VisionTransformer 外壳下，使用 PromptedTransformer 作为内部的 transformer 编码器。

    好处：
    - 复用原有 VisionTransformer 的接口与分类头设计；
    - 在不改变“CLS 池化 + 线性分类”整体逻辑的情况下，将 Prompt 机制无缝注入；
    - 对外的 forward 接口基本保持一致，只是内部编码阶段换成了 PromptedTransformer。
    """
    def __init__(self, prompt_cfg, model_type,img_size=224, num_classes=21843, vis=False, prompt_init=None, prompt_init_provider=None):        # 当前实现只支持原生的 CLS 池化方式（original）
        """
        :param prompt_cfg:   PROMPT 子配置（NUM_TOKENS / PROJECT / DEEP 等）
        :param model_type:   ViT 模型类型（用于从 CONFIGS 中取结构配置）
        :param img_size:     输入图像尺寸（默认 224）
        :param num_classes:  分类类别数（默认 21843，对应 ImageNet-21K）
        :param vis:          是否返回注意力权重
        :param prompt_init:  外部 prompt 初始化张量（可选）
        :param prompt_init_provider: 运行时 prompt 生成器（可选）
        """
        # 当前只支持原始的 CLS 池化方式（不做 GAP 等替代池化）
        assert prompt_cfg.VIT_POOL_TYPE == "original"

        # 先调用父类 VisionTransformer 初始化基础结构
        super(PromptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg

        # 取出结构规格（如 hidden_size、层数、patch 大小等）
        vit_cfg = CONFIGS[model_type]
        # 核心替换：把内部的 transformer 用“带 Prompt 的”版本替换
        self.transformer = PromptedTransformer(prompt_cfg, vit_cfg, img_size, vis, prompt_init=prompt_init, prompt_init_provider=prompt_init_provider,)

    def forward(self, x, vis=False, semantics=None):
        """
        前向流程：

        1) 调用内部的 PromptedTransformer(x, semantics)：
           - 得到编码后的序列 x（含 CLS + PROMPT + PATCH）；
           - 可选返回各层的注意力权重 attn_weights。
        2) 取 CLS 位置的 token（x[:, 0]）作为全局表征；
        3) 通过 self.head（线性层）得到最终 logits；
        4) 若 vis=False，则只返回 logits；
           若 vis=True，则返回 (logits, attn_weights)，便于可视化/分析。
        """
        # transformer 返回的 x 为编码后的 token 序列，attn_weights 为可视化用注意力权重
        x, attn_weights = self.transformer(x, semantics)

        # 取 CLS token（位置 0）作为全局表征
        x = x[:, 0]

        # 线性分类头 -> logits（未做 softmax）
        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights

    def forward_with_affinity(self, x, affinity_config, vis=False, semantics=None):
        """
        带亲和矩阵输出的前向接口（与 VisionTransformer.forward_with_affinity 平行）。

        - 调用内部 PromptedTransformer.forward_with_affinity 返回序列/权重/亲和
        - 取 CLS 做分类
        - vis=False 返回 (logits, affinities)，vis=True 额外返回 attn_weights
        """
        x, attn_weights, affinities = self.transformer.forward_with_affinity(x, affinity_config, semantics)

        logits = self.head(x[:, 0])

        if not vis:
            return logits, affinities
        return logits, attn_weights, affinities