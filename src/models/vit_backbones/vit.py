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
from typing import Any, Dict, Optional
# 8.21改动
# 原有 Win 下 os.path.join 会产生反斜杠 "\"，会影响从权重字典中取键（键名一般用 "/"）
# 因此改为从 posixpath 导入 join，确保键名分隔符始终为 "/"
# from os.path import join as pjoin  # 原有Win下os.path.join会产生反斜杠 \
from turtle import forward           # 原有代码就这两行
from posixpath import join as pjoin  # 关键：确保键名里总是用 "/"
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

ACT2FN = {"gelu": torch.nn.functional.gelu}
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
        self.debug_shapes = False
        self._shape_debug_forward_proj_logged = False

    def transpose_for_scores(self, x):
        """将张量从 [B, N, D] 变形为 [B, h, N, d_k] 以便做多头注意力。"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # (..., h, d_k)
        x = x.view(*new_x_shape)    # [B, N, h, d_k]
        return x.permute(0, 2, 1, 3)    # -> [B, h, N, d_k]

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

    def _merge_heads(self, context_layer: torch.Tensor) -> torch.Tensor:
        """
        将多头上下文从 [B, H, N, Dh] 合并回 [B, N, D]。

        attention mediation 会额外构造一份 delta_context，它和标准 MHSA
        的 context_layer 具有同样的多头形状，因此复用这一段合并逻辑。
        """
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_context_layer_shape)

    def _context_to_attention_output(self, context_layer: torch.Tensor, *, include_bias: bool = True) -> torch.Tensor:
        """
        把 [B, H, N, Dh] 的 attention context 变成标准 attention 输出 [B, N, D]。

        include_bias=True:
            标准 MHSA 主路径，完整经过 self.out 的 weight+bias。
        include_bias=False:
            mediated 分支只表示“修正量 delta”，不能额外叠加一次 out bias，
            否则即使 delta_context 很小也会引入固定偏置，破坏“只修正注意力分布”的语义。
        """
        context_layer = self._merge_heads(context_layer)
        if include_bias:
            attention_output = self.out(context_layer)
        else:
            attention_output = torch.nn.functional.linear(context_layer, self.out.weight, None)
        return self.proj_dropout(attention_output)

    @staticmethod
    def _row_normalize(x: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
        """
        对局部 attention 子块做行归一化。

        这里不使用 softmax，因为 SOURCE="probs" 时输入已经是 full softmax
        后的概率子块；此时只需要在局部区域内转成条件分布。
        """
        return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)

    @staticmethod
    def _route_detach(direct: torch.Tensor, mediated: torch.Tensor, detach_mode: str) -> torch.Tensor:
        """
        根据 teacher/student 设定选择最终写回 visual block 的条件路由。

        mediated / via_prompt:
            使用 mediated route 的前向值，但 detach 掉 mediated 分支梯度。
            适合把 mediated route 当 teacher，只让后续 gamma / token 状态承受影响。
        direct:
            保持 direct route，不进行 mediated 前向替换。
            这一版不制造“前向为 mediated、梯度走 direct”的隐式技巧，避免语义不清。
        none:
            直接使用 mediated route，并允许梯度沿 mediated route 回传。
        """
        if detach_mode in {"mediated", "via_prompt"}:
            return mediated.detach()
        if detach_mode == "direct":
            return direct
        if detach_mode == "none":
            return mediated
        raise ValueError(f"Unsupported attention mediation detach mode='{detach_mode}'.")

    def _conditional_normalize(self, x: torch.Tensor, dim: int, source: str) -> torch.Tensor:
        """
        将局部子块转成条件分布。

        SOURCE="scores":
            子块来自 softmax 前 logits，因此用 softmax 得到局部条件分布。
        SOURCE="probs":
            子块来自 full attention softmax 后的真实概率，因此只做局部行归一化。
        """
        if source == "scores":
            return torch.softmax(x, dim=dim)
        if source == "probs":
            return self._row_normalize(x, dim=dim)
        raise ValueError(f"Unsupported ATTENTION_MEDIATION.SOURCE='{source}'. Expected scores / probs.")

    def _attention_mediation_slices(self, seq_len: int, prompt_length: int, semantic_length: int) -> Dict[str, slice]:
        """
        根据当前主序列布局切分 token 区间。

        当前项目固定布局为:
            [CLS | prompt_tokens | visual_tokens | semantic_tokens]
        mediated attention 只在 prompt/semantic 到 visual 的子块上做修正，
        因此这里必须明确得到 P/V/S 三段 slice。
        """
        prompt_length = int(prompt_length)
        semantic_length = int(semantic_length)
        if prompt_length <= 0:
            raise ValueError("ATTENTION_MEDIATION requires prompt_length > 0.")
        if semantic_length <= 0:
            raise ValueError("ATTENTION_MEDIATION requires semantic_length > 0.")
        if seq_len <= 1 + prompt_length + semantic_length:
            raise ValueError(
                f"ATTENTION_MEDIATION requires visual tokens, got seq_len={seq_len}, "
                f"prompt_length={prompt_length}, semantic_length={semantic_length}."
            )
        prompt_start = 1
        prompt_end = prompt_start + prompt_length
        visual_start = prompt_end
        visual_end = seq_len - semantic_length
        return {
            "prompt": slice(prompt_start, prompt_end),
            "visual": slice(visual_start, visual_end),
            "semantic": slice(visual_end, seq_len),
        }

    def _compose_visual_block_row(
        self,
        direct_full_row: torch.Tensor,
        visual_slice: slice,
        mediated_visual_cond: torch.Tensor,
        detach_mode: str,
        mass_mode: str,
        beta_mass: float,
        mediated_mass_score: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        把 mediated visual 条件分布写回完整 attention row。

        row_preserve:
            每个 query token 分给 visual 区域的总 mass 不变，只改变 visual 内部 patch 分配。
        block_redistribute:
            同组 query token 之间可以重新分配 visual mass；非 visual 区域同步缩放，
            因而完整 row 仍然保持概率和为 1。
        """
        direct_visual_probs = direct_full_row[:, :, :, visual_slice]
        direct_visual_mass = direct_visual_probs.sum(dim=-1, keepdim=True)
        direct_visual_cond = self._row_normalize(direct_visual_probs, dim=-1)
        selected_visual_cond = self._route_detach(direct_visual_cond, mediated_visual_cond, detach_mode)

        if mass_mode == "row_preserve":
            new_visual_mass = direct_visual_mass
        elif mass_mode == "block_redistribute":
            if mediated_mass_score is None:
                raise ValueError("block_redistribute requires mediated_mass_score.")
            direct_mass_per_query = direct_visual_mass.squeeze(-1)
            direct_total_mass = direct_mass_per_query.sum(dim=-1, keepdim=True)
            direct_mass_dist = self._row_normalize(direct_mass_per_query, dim=-1)
            mediated_mass_dist = self._row_normalize(mediated_mass_score.squeeze(-1), dim=-1)
            selected_mass_dist = self._route_detach(direct_mass_dist, mediated_mass_dist, detach_mode)
            mixed_mass_dist = (1.0 - float(beta_mass)) * direct_mass_dist + float(beta_mass) * selected_mass_dist
            new_visual_mass = (direct_total_mass * mixed_mass_dist).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported ATTENTION_MEDIATION.MASS_MODE='{mass_mode}'.")

        modified_row = direct_full_row.clone()
        modified_row[:, :, :, visual_slice] = selected_visual_cond * new_visual_mass

        # visual mass 改变后，非 visual 区域不能保持原值；否则整行概率和会偏离 1。
        direct_nonvisual_mass = 1.0 - direct_visual_mass
        new_nonvisual_mass = 1.0 - new_visual_mass
        nonvisual_scale = new_nonvisual_mass / direct_nonvisual_mass.clamp_min(1e-8)
        if visual_slice.start > 0:
            modified_row[:, :, :, :visual_slice.start] = direct_full_row[:, :, :, :visual_slice.start] * nonvisual_scale
        if visual_slice.stop < direct_full_row.size(-1):
            modified_row[:, :, :, visual_slice.stop:] = direct_full_row[:, :, :, visual_slice.stop:] * nonvisual_scale
        return self._row_normalize(modified_row, dim=-1)

    def _build_prompt_mediated_row(
        self,
        route_base: torch.Tensor,
        attention_probs: torch.Tensor,
        prompt_slice: slice,
        visual_slice: slice,
        semantic_slice: slice,
        source: str,
        route: str,
        detach_mode: str,
        route_scope: str,
        mass_mode: str,
        beta_mass: float,
    ) -> torch.Tensor:
        """
        构造 prompt query 的完整 modified attention row。

        visual_block 只改 P->V 子块；full_row 直接构造 P->[CLS|P|V|S] 完整行。
        """
        a_sp = route_base[:, :, semantic_slice, prompt_slice]
        a_ps = route_base[:, :, prompt_slice, semantic_slice]
        a_sv = route_base[:, :, semantic_slice, visual_slice]
        sv_cond = self._conditional_normalize(a_sv, dim=-1, source=source)
        direct_prompt_row = attention_probs[:, :, prompt_slice, :]

        if route == "S_to_P_and_V":
            sp_cond = self._conditional_normalize(a_sp, dim=-2, source=source)
            if route_scope == "full_row":
                s_full_cond = self._conditional_normalize(route_base[:, :, semantic_slice, :], dim=-1, source=source)
                mediated_full = torch.matmul(sp_cond.transpose(-1, -2), s_full_cond)
            else:
                mediated_visual_base = torch.matmul(sp_cond.transpose(-1, -2), sv_cond)
                semantic_visual_mass = attention_probs[:, :, semantic_slice, visual_slice].sum(dim=-1, keepdim=True)
                mediated_mass_score = torch.matmul(sp_cond.transpose(-1, -2), semantic_visual_mass)
        elif route == "P_to_S_to_V":
            ps_cond = self._conditional_normalize(a_ps, dim=-1, source=source)
            if route_scope == "full_row":
                s_full_cond = self._conditional_normalize(route_base[:, :, semantic_slice, :], dim=-1, source=source)
                mediated_full = torch.matmul(ps_cond, s_full_cond)
            else:
                mediated_visual_base = torch.matmul(ps_cond, sv_cond)
                semantic_visual_mass = attention_probs[:, :, semantic_slice, visual_slice].sum(dim=-1, keepdim=True)
                mediated_mass_score = torch.matmul(ps_cond, semantic_visual_mass)
        else:
            raise ValueError(
                f"Unsupported ATTENTION_MEDIATION.PROMPT_ROUTE='{route}'. "
                "Expected S_to_P_and_V / P_to_S_to_V."
            )

        if route_scope == "full_row":
            mediated_full = self._row_normalize(mediated_full, dim=-1)
            return self._route_detach(direct_prompt_row, mediated_full, detach_mode)
        if route_scope == "visual_block":
            mediated_visual_cond = self._row_normalize(mediated_visual_base, dim=-1)
            return self._compose_visual_block_row(
                direct_prompt_row,
                visual_slice,
                mediated_visual_cond,
                detach_mode,
                mass_mode,
                beta_mass,
                mediated_mass_score,
            )
        raise ValueError(f"Unsupported ATTENTION_MEDIATION.ROUTE_SCOPE='{route_scope}'.")

    def _build_semantic_mediated_row(
        self,
        route_base: torch.Tensor,
        attention_probs: torch.Tensor,
        prompt_slice: slice,
        visual_slice: slice,
        semantic_slice: slice,
        source: str,
        route: str,
        detach_mode: str,
        route_scope: str,
        mass_mode: str,
        beta_mass: float,
    ) -> torch.Tensor:
        """
        构造 semantic query 的完整 modified attention row。

        visual_block 只改 S->V 子块；full_row 直接构造 S->[CLS|P|V|S] 完整行。
        """
        a_sp = route_base[:, :, semantic_slice, prompt_slice]
        a_ps = route_base[:, :, prompt_slice, semantic_slice]
        a_pv = route_base[:, :, prompt_slice, visual_slice]
        pv_cond = self._conditional_normalize(a_pv, dim=-1, source=source)
        direct_semantic_row = attention_probs[:, :, semantic_slice, :]

        if route == "S_to_P_to_V":
            sp_cond = self._conditional_normalize(a_sp, dim=-1, source=source)
            if route_scope == "full_row":
                p_full_cond = self._conditional_normalize(route_base[:, :, prompt_slice, :], dim=-1, source=source)
                mediated_full = torch.matmul(sp_cond, p_full_cond)
            else:
                mediated_visual_base = torch.matmul(sp_cond, pv_cond)
                prompt_visual_mass = attention_probs[:, :, prompt_slice, visual_slice].sum(dim=-1, keepdim=True)
                mediated_mass_score = torch.matmul(sp_cond, prompt_visual_mass)
        elif route == "P_to_S_and_V":
            ps_cond = self._conditional_normalize(a_ps, dim=-2, source=source)
            if route_scope == "full_row":
                p_full_cond = self._conditional_normalize(route_base[:, :, prompt_slice, :], dim=-1, source=source)
                mediated_full = torch.matmul(ps_cond.transpose(-1, -2), p_full_cond)
            else:
                mediated_visual_base = torch.matmul(ps_cond.transpose(-1, -2), pv_cond)
                prompt_visual_mass = attention_probs[:, :, prompt_slice, visual_slice].sum(dim=-1, keepdim=True)
                mediated_mass_score = torch.matmul(ps_cond.transpose(-1, -2), prompt_visual_mass)
        else:
            raise ValueError(
                f"Unsupported ATTENTION_MEDIATION.SEMANTIC_ROUTE='{route}'. "
                "Expected S_to_P_to_V / P_to_S_and_V."
            )

        if route_scope == "full_row":
            mediated_full = self._row_normalize(mediated_full, dim=-1)
            return self._route_detach(direct_semantic_row, mediated_full, detach_mode)
        if route_scope == "visual_block":
            mediated_visual_cond = self._row_normalize(mediated_visual_base, dim=-1)
            return self._compose_visual_block_row(
                direct_semantic_row,
                visual_slice,
                mediated_visual_cond,
                detach_mode,
                mass_mode,
                beta_mass,
                mediated_mass_score,
            )
        raise ValueError(f"Unsupported ATTENTION_MEDIATION.ROUTE_SCOPE='{route_scope}'.")

    def _compute_attention_mediation(
        self,
        attention_scores: torch.Tensor,
        attention_probs: torch.Tensor,
        value_layer: torch.Tensor,
        prompt_length: int,
        semantic_length: int,
        mediation_config: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        在当前层 MHSA 内部构造 mediated attention correction。

        这里先构造目标概率分布 target_probs。若 SOURCE=scores，再用
        log(target_probs)-log(original_probs) 形成 score bias，并重新对完整 row 做 softmax。
        这样 score 分支最终仍由 full softmax 产生合法 attention 概率。
        """
        if not mediation_config or not mediation_config.get("enable", False):
            return None

        source = str(mediation_config.get("source"))
        route_scope = str(mediation_config.get("route_scope"))
        mass_mode = str(mediation_config.get("mass_mode"))
        slices = self._attention_mediation_slices(attention_probs.size(-1), prompt_length, semantic_length)
        prompt_slice = slices["prompt"]
        visual_slice = slices["visual"]
        semantic_slice = slices["semantic"]
        route_base = attention_scores if source == "scores" else attention_probs

        target_probs = attention_probs.clone()
        target_probs[:, :, prompt_slice, :] = self._build_prompt_mediated_row(
            route_base,
            attention_probs,
            prompt_slice,
            visual_slice,
            semantic_slice,
            source,
            str(mediation_config.get("prompt_route")),
            str(mediation_config.get("prompt_detach")),
            route_scope,
            mass_mode,
            float(mediation_config.get("beta_prompt_mass", 0.0)),
        )
        target_probs[:, :, semantic_slice, :] = self._build_semantic_mediated_row(
            route_base,
            attention_probs,
            prompt_slice,
            visual_slice,
            semantic_slice,
            source,
            str(mediation_config.get("semantic_route")),
            str(mediation_config.get("semantic_detach")),
            route_scope,
            mass_mode,
            float(mediation_config.get("beta_semantic_mass", 0.0)),
        )

        if source == "scores":
            # score 空间不能直接比较普通数值和。这里使用 log-ratio bias：
            # score' = score + log(target_prob) - log(original_prob)，再由 full softmax 重新归一化。
            score_bias = torch.log(target_probs.clamp_min(1e-8)) - torch.log(attention_probs.clamp_min(1e-8))
            modified_probs = torch.softmax(attention_scores + score_bias, dim=-1)
        elif source == "probs":
            modified_probs = target_probs
        else:
            raise ValueError(f"Unsupported ATTENTION_MEDIATION.SOURCE='{source}'.")

        delta_probs = modified_probs - attention_probs
        delta_context = torch.matmul(delta_probs, value_layer)
        mediated_context = torch.matmul(modified_probs, value_layer)
        delta_attention_output = self._context_to_attention_output(delta_context, include_bias=False)
        mediated_attention_output = self._context_to_attention_output(mediated_context, include_bias=True)
        return {
            "delta_attention_output": delta_attention_output,
            "mediated_attention_output": mediated_attention_output,
            "prompt_slice": prompt_slice,
            "semantic_slice": semantic_slice,
        }

    def _scaled_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        semantic_length: int = 0,
        block_s_to_cls: bool = False,
        mediation_config: Optional[Dict[str, Any]] = None,
        prompt_length: int = 0,
    ):
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
        semantic_length = int(semantic_length)
        if block_s_to_cls and semantic_length > 0:
            # semantic token 作为 query 时禁止读取 CLS；只改 logits，随后仍由 full softmax 统一归一化。
            if attention_scores.size(-1) <= semantic_length:
                raise ValueError(
                    f"Cannot apply S->CLS attention mask with semantic_length={semantic_length} "
                    f"and sequence length={attention_scores.size(-1)}."
                )
            semantic_slice = slice(attention_scores.size(-2) - semantic_length, attention_scores.size(-2))
            attention_scores[:, :, semantic_slice, 0] = torch.finfo(attention_scores.dtype).min

        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key) # 行归一化
        weights = attention_probs if self.vis else None     # 用于可视化
        # ATTENTION_MEDIATION 的介入点在 softmax 之后、attention dropout 之前。
        # 这样既能拿到 score/prob 两种来源，也能保证 correction 使用未 dropout 的完整注意力分布。
        mediation = self._compute_attention_mediation(
            attention_scores,
            attention_probs,
            value_layer,
            prompt_length,
            semantic_length,
            mediation_config,
        )
        attention_probs = self.attn_dropout(attention_probs)

        # 标准 MHSA 主路径不被替换；mediated correction 作为额外 delta 在 Block.forward 中合并。
        context_layer = torch.matmul(attention_probs, value_layer)
        attention_output = self._context_to_attention_output(context_layer, include_bias=True)
        return attention_output, weights, mediation

    @staticmethod
    def _minmax_normalize_lastdim(x: torch.Tensor) -> torch.Tensor:
        """用于把 raw logits 转成仅用于画图的 vis_norm"""
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        denom = (x_max - x_min).clamp_min(1e-12)
        return (x - x_min) / denom
    def forward(
        self,
        hidden_states,
        semantic_length: int = 0,
        block_s_to_cls: bool = False,
        mediation_config: Optional[Dict[str, Any]] = None,
        prompt_length: int = 0,
    ):
        """
        对整段序列执行标准 self-attention 前向。

        返回 attention 输出及（可选）注意力权重，兼容原有调用路径。
        """
        query_layer, key_layer, value_layer = self._project_qkv(hidden_states)
        return self._scaled_attention(
            query_layer,
            key_layer,
            value_layer,
            semantic_length,
            block_s_to_cls,
            mediation_config,
            prompt_length,
        )

    def forward_with_projections(
        self,
        hidden_states,
        semantic_length: int = 0,
        block_s_to_cls: bool = False,
        mediation_config: Optional[Dict[str, Any]] = None,
        prompt_length: int = 0,
    ):
        """
        在返回 self-attention 输出的同时，额外暴露多头形式的 q/k。

        供亲和矩阵等附加分支复用，避免重复线性映射计算。
        """
        query_layer, key_layer, value_layer = self._project_qkv(hidden_states)
        attention_output, weights, mediation = self._scaled_attention(
            query_layer,
            key_layer,
            value_layer,
            semantic_length,
            block_s_to_cls,
            mediation_config,
            prompt_length,
        )
        if self.debug_shapes and (not self._shape_debug_forward_proj_logged):
            print(
                "[SHAPE-DEBUG] Attention.forward_with_projections q_proj={} k_proj={} v_proj={}".format(
                    tuple(query_layer.shape),
                    tuple(key_layer.shape),
                    tuple(value_layer.shape),
                )
            )
            self._shape_debug_forward_proj_logged = True
        return attention_output, weights, query_layer, key_layer, mediation

    def compute_prompt_visual_monitors(self, query_layer, key_layer, prompt_length, semantic_length=0, *, detach=True):
        """
        统一导出主干 prompt/visual 的原始亲和矩阵。

        这一版直接取代旧的 compute_affinity：
        - 不再接收 normalize
        - 主输出一律保留 raw logits
        - 额外附带仅用于画图的 min-max 归一化版本

        返回：
        - QpQv_raw / QpQv_vis
        - KpKv_raw / KpKv_vis
        - QpKv_raw / QpKv_vis
        """
        q_base = query_layer.detach() if detach else query_layer
        k_base = key_layer.detach() if detach else key_layer

        if q_base.size(2) < 1 + prompt_length + semantic_length or k_base.size(2) < 1 + prompt_length + semantic_length:
            raise ValueError(
                f"Sequence length is insufficient for prompt_length={prompt_length}, semantic_length={semantic_length}: "
                f"q_len={q_base.size(2)}, k_len={k_base.size(2)}"
            )

        cls_offset = 1
        prompt_slice = slice(cls_offset, cls_offset + prompt_length)
        patch_start = cls_offset + prompt_length
        patch_end = q_base.size(2) - int(semantic_length)
        patch_slice = slice(patch_start, patch_end)
        semantic_slice = slice(patch_end, q_base.size(2))

        q_prompt = q_base[:, :, prompt_slice, :]
        q_patch = q_base[:, :, patch_slice, :]
        q_semantic = q_base[:, :, semantic_slice, :]
        k_prompt = k_base[:, :, prompt_slice, :]
        k_patch = k_base[:, :, patch_slice, :]
        k_semantic = k_base[:, :, semantic_slice, :]
        scale = 1.0 / math.sqrt(self.attention_head_size)

        monitors = {}
        if q_prompt.numel() > 0 and q_patch.numel() > 0:
            qpqv_raw = torch.matmul(q_prompt, q_patch.transpose(-1, -2)) * scale
            monitors["QpQv_raw"] = qpqv_raw
            monitors["QpQv_vis"] = self._minmax_normalize_lastdim(qpqv_raw)

        if k_prompt.numel() > 0 and k_patch.numel() > 0:
            kpkv_raw = torch.matmul(k_prompt, k_patch.transpose(-1, -2)) * scale
            monitors["KpKv_raw"] = kpkv_raw
            monitors["KpKv_vis"] = self._minmax_normalize_lastdim(kpkv_raw)

        if q_prompt.numel() > 0 and k_patch.numel() > 0:
            qpkv_raw = torch.matmul(q_prompt, k_patch.transpose(-1, -2)) * scale
            monitors["QpKv_raw"] = qpkv_raw
            monitors["QpKv_vis"] = self._minmax_normalize_lastdim(qpkv_raw)

        if q_semantic.numel() > 0 and k_patch.numel() > 0:
            qskv_raw = torch.matmul(q_semantic, k_patch.transpose(-1, -2)) * scale
            monitors["QsKv_raw"] = qskv_raw
            monitors["QsKv_vis"] = self._minmax_normalize_lastdim(qskv_raw)

        if q_patch.numel() > 0 and k_semantic.numel() > 0:
            qvks_raw = torch.matmul(q_patch, k_semantic.transpose(-1, -2)) * scale
            monitors["QvKs_raw"] = qvks_raw
            monitors["QvKs_vis"] = self._minmax_normalize_lastdim(qvks_raw)

        if q_semantic.numel() > 0 and k_prompt.numel() > 0:
            qskp_raw = torch.matmul(q_semantic, k_prompt.transpose(-1, -2)) * scale
            monitors["QsKp_raw"] = qskp_raw
            monitors["QsKp_vis"] = self._minmax_normalize_lastdim(qskp_raw)

        if q_prompt.numel() > 0 and k_semantic.numel() > 0:
            qpks_raw = torch.matmul(q_prompt, k_semantic.transpose(-1, -2)) * scale
            monitors["QpKs_raw"] = qpks_raw
            monitors["QpKs_vis"] = self._minmax_normalize_lastdim(qpks_raw)

        return monitors

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
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        patch_tokens = x.transpose(-1, -2)
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
        self.debug_shapes = False

    @staticmethod
    def _attention_mediation_gamma(config: Dict[str, Any], name: str, layer_idx: int, ref: torch.Tensor) -> torch.Tensor:
        """
        取当前层的 gamma gate，并整理成可广播到 [B,N,D] 的形状。

        attention mediation 是“当前层内”的修正，因此 gamma 长度等于 ViT block 数；
        这和旧 affinity_evolution 的“层间”更新不同，旧分支通常只有 num_layers-1 个 gamma。
        """
        if name not in config:
            raise ValueError(f"ATTENTION_MEDIATION missing gamma field: {name}.")
        gamma = config[name]
        if torch.is_tensor(gamma):
            if gamma.dim() == 0:
                return gamma.to(device=ref.device, dtype=ref.dtype).view(1, 1, 1)
            return gamma[int(layer_idx)].to(device=ref.device, dtype=ref.dtype).view(1, 1, 1)
        return torch.tensor(float(gamma), device=ref.device, dtype=ref.dtype).view(1, 1, 1)

    def _scale_attention_mediation_delta(
        self,
        delta: torch.Tensor,
        mediation: Optional[Dict[str, Any]],
        config: Optional[Dict[str, Any]],
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        只把 mediated attention correction 写到 prompt / semantic token。

        delta_attention_output 本身是 [B,N,D]，但新分支的目的不是改 CLS 或 visual patch
        主路径，而是让 P/S token 获得 mediated route 修正。因此这里创建全零张量，
        只在 prompt_slice 和 semantic_slice 上乘各自 gamma 后写入。
        """
        if mediation is None:
            return None
        if not config or not config.get("enable", False):
            return None
        prompt_slice = mediation["prompt_slice"]
        semantic_slice = mediation["semantic_slice"]
        gamma_prompt = self._attention_mediation_gamma(config, "prompt_gamma", layer_idx, delta)
        gamma_semantic = self._attention_mediation_gamma(config, "semantic_gamma", layer_idx, delta)
        scaled = torch.zeros_like(delta)
        scaled[:, prompt_slice, :] = gamma_prompt * delta[:, prompt_slice, :]
        scaled[:, semantic_slice, :] = gamma_semantic * delta[:, semantic_slice, :]
        return scaled

    def _mix_attention_mediation_block_output(
        self,
        original: torch.Tensor,
        mediated: torch.Tensor,
        mediation: Optional[Dict[str, Any]],
        config: Optional[Dict[str, Any]],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        方案 B / block_parallel 的输出混合。

        original:
            原始 ViT block 完整输出，CLS/P/V/S 都来自原始路径。
        mediated:
            mediated attention 路径进入同一个 MLP 后得到的完整 block 输出。

        最终只替换/混合 prompt 与 semantic token：
            out[P/S] = original[P/S] + gamma * (mediated[P/S] - original[P/S])
        CLS 和 visual patch 保持原始路径，避免破坏冻结 ViT 的主干视觉表征。
        """
        if mediation is None:
            return original
        if not config or not config.get("enable", False):
            return original
        prompt_slice = mediation["prompt_slice"]
        semantic_slice = mediation["semantic_slice"]
        gamma_prompt = self._attention_mediation_gamma(config, "prompt_gamma", layer_idx, original)
        gamma_semantic = self._attention_mediation_gamma(config, "semantic_gamma", layer_idx, original)
        mixed = original.clone()
        mixed[:, prompt_slice, :] = original[:, prompt_slice, :] + gamma_prompt * (
            mediated[:, prompt_slice, :] - original[:, prompt_slice, :]
        )
        mixed[:, semantic_slice, :] = original[:, semantic_slice, :] + gamma_semantic * (
            mediated[:, semantic_slice, :] - original[:, semantic_slice, :]
        )
        return mixed

    def forward(
        self,
        x,
        semantics: torch.Tensor = None,
        num_prompt_tokens: int = 0,
        semantic_length: int = 0,
        block_s_to_cls: bool = False,
        attention_mediation_config: Optional[Dict[str, Any]] = None,
        layer_idx: int = 0,
    ):
        # 段1：注意力 + 残差
        h = x  # 残差分支
        x = self.attention_norm(x)  # LN
        # Attention 内部会先计算原始 MHSA；如果开启 ATTENTION_MEDIATION，
        # 会额外返回一份基于 mediated route 的 delta_attention_output。
        x, weights, mediation = self.attn(
            x,
            semantic_length,
            block_s_to_cls,
            attention_mediation_config,
            num_prompt_tokens,
        )   # MHSA（输出同形状）; weights 仅在 vis=True 时非 None
        execution_mode = attention_mediation_config.get("execution_mode") if attention_mediation_config else None
        mlp_policy = attention_mediation_config.get("mlp_policy") if attention_mediation_config else None
        if mediation is not None and execution_mode == "block_parallel":
            if mlp_policy != "enter_mlp":
                raise ValueError("ATTENTION_MEDIATION block_parallel requires MLP_POLICY='enter_mlp'.")
            original_after_attn = x + h
            original_out = self.ffn(self.ffn_norm(original_after_attn)) + original_after_attn
            mediated_after_attn = mediation["mediated_attention_output"] + h
            mediated_out = self.ffn(self.ffn_norm(mediated_after_attn)) + mediated_after_attn
            x = self._mix_attention_mediation_block_output(
                original_out,
                mediated_out,
                mediation,
                attention_mediation_config,
                layer_idx,
            )
            return x, weights, semantics

        scaled_delta = self._scale_attention_mediation_delta(
            mediation["delta_attention_output"] if mediation is not None else None,
            mediation,
            attention_mediation_config,
            layer_idx,
        )
        if scaled_delta is not None and mlp_policy == "enter_mlp":
            # enter_mlp：在 attention output 阶段合并 mediated correction，
            # 后续 FFN 会看到修正后的 P/S token，属于更强的 block 内介入。
            x = x + scaled_delta
        x = x + h                   # 残差相加

        # 段2：FFN + 残差
        h = x
        x = self.ffn_norm(x)        # LN
        x = self.ffn(x)             # MLP: D→H→D
        x = x + h                   # 残差相加
        if scaled_delta is not None and mlp_policy == "skip_mlp":
            # skip_mlp：原始 block 完整结束后再写回 P/S residual。
            # 当前层 FFN 不处理 mediated 修改，更接近旧层间 residual controller。
            x = x + scaled_delta

        return x, weights, semantics

    def forward_with_affinity(
        self,
        x,
        affinity_config,
        semantics: torch.Tensor = None,
        num_prompt_tokens: int = 0,
        attention_mediation_config: Optional[Dict[str, Any]] = None,
        layer_idx: int = 0,
    ):
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
            - 使用 compute_prompt_visual_monitors 在同一层的 Q/K 上构造 raw/vis 亲和；
            - 将该层的亲和 dict 返回给上级 Encoder 统一收集。

        affinity_config: dict，支持字段：
            - "prompt_length": int，prompt token 数量 L_p
            - "detach": bool，是否在构亲和前对 q/k detach

        返回:
            x:          [B, N, D]，本层输出
            weights:    注意力权重（仅 vis=True 时非 None）
            affinities: dict，包含 raw/vis 亲和矩阵
        """
        # --- 注意力分支 + 残差 ---
        h = x
        x_norm = self.attention_norm(x)
        # 同时拿到 MHSA 输出 + 多头形式的 q_proj / k_proj
        x, weights, q_proj, k_proj, mediation = self.attn.forward_with_projections(
            x_norm,
            affinity_config.get("semantic_length", 0),
            affinity_config.get("block_s_to_cls", False),
            attention_mediation_config,
            num_prompt_tokens,
        )
        execution_mode = attention_mediation_config.get("execution_mode") if attention_mediation_config else None
        mlp_policy = attention_mediation_config.get("mlp_policy") if attention_mediation_config else None
        if mediation is not None and execution_mode == "block_parallel":
            if mlp_policy != "enter_mlp":
                raise ValueError("ATTENTION_MEDIATION block_parallel requires MLP_POLICY='enter_mlp'.")
            original_after_attn = x + h
            original_out = self.ffn(self.ffn_norm(original_after_attn)) + original_after_attn
            mediated_after_attn = mediation["mediated_attention_output"] + h
            mediated_out = self.ffn(self.ffn_norm(mediated_after_attn)) + mediated_after_attn
            x = self._mix_attention_mediation_block_output(
                original_out,
                mediated_out,
                mediation,
                attention_mediation_config,
                layer_idx,
            )
            attn_aff = self.attn.compute_prompt_visual_monitors(
                q_proj,
                k_proj,
                affinity_config.get("prompt_length", 0),
                affinity_config.get("semantic_length", 0),
                detach=affinity_config.get("detach", True),
            )
            return x, weights, attn_aff, semantics

        scaled_delta = self._scale_attention_mediation_delta(
            mediation["delta_attention_output"] if mediation is not None else None,
            mediation,
            attention_mediation_config,
            layer_idx,
        )
        if scaled_delta is not None and mlp_policy == "enter_mlp":
            # 与标准 forward 保持同一插入点：attention output 合并后再进入残差和 FFN。
            x = x + scaled_delta
        x = x + h

        # --- FFN 分支 + 残差 ---
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        if scaled_delta is not None and mlp_policy == "skip_mlp":
            # affinity monitor 路径也保留同样的 skip_mlp 语义，避免训练/可视化前向不一致。
            x = x + scaled_delta

        # --- 基于 Q/K 计算 prompt/patch 亲和 ---
        attn_aff = self.attn.compute_prompt_visual_monitors(
            q_proj,
            k_proj,
            affinity_config.get("prompt_length", 0),
            affinity_config.get("semantic_length", 0),
            detach=affinity_config.get("detach", True),
        )

        return x, weights, attn_aff, semantics

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
    def forward(
        self,
        hidden_states,
        semantics: torch.Tensor = None,
        num_prompt_tokens: int = 0,
        semantic_length: int = 0,
        block_s_to_cls: bool = False,
        attention_mediation_config: Optional[Dict[str, Any]] = None,
    ):
        """常规前向：返回编码结果与（可选）各层注意力权重。"""
        attn_weights = []
        for layer_idx, layer_block in enumerate(self.layer):
            # attention_mediation_config 在所有层共享，layer_idx 用于取当前层独立的 gamma gate。
            hidden_states, weights, semantics = layer_block(hidden_states, semantics,
                    num_prompt_tokens, semantic_length, block_s_to_cls,
                    attention_mediation_config, layer_idx)  # hidden_states为(B, 1+N, D)D 为 hidden_size
            if self.vis:
                attn_weights.append(weights)    # 把每层的 weights 保存到列表里否则返回空列表
        encoded = self.encoder_norm(hidden_states)  # 对最后一层输出再做一次 LayerNorm，得到 encoded
        return encoded, attn_weights

    def forward_with_affinity(
        self,
        hidden_states,
        affinity_config,
        semantics: torch.Tensor = None,
        num_prompt_tokens: int = 0,
        attention_mediation_config: Optional[Dict[str, Any]] = None,
    ):
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
            affinities:   list，长度 = num_layers，每个元素是一层的 raw/vis 亲和字典
        """
        attn_weights = []
        affinities = []
        for layer_idx, layer_block in enumerate(self.layer):
            # forward_with_affinity 同时服务训练损失/可视化，因此 mediation 的插入点必须和常规 forward 一致。
            hidden_states, weights, affinity, semantics = layer_block.forward_with_affinity(
                hidden_states,
                affinity_config,
                semantics,
                num_prompt_tokens,
                attention_mediation_config,
                layer_idx,
            )
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
            hidden_states, _, _ = layer_block(hidden_states)
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
        self._last_semantic_token_state = None
        self._last_prompt_path_info = {}

    def forward(self, input_ids, semantics: torch.Tensor = None):
        """标准前向：返回编码后的序列与注意力权重。"""
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output, semantics)
        return encoded, attn_weights


    def forward_with_affinity(self, input_ids, affinity_config, semantics: torch.Tensor = None):
        """
        与标准 forward 类似，但返回“带亲和”的版本：

        步骤:
            1) Embeddings: x -> [CLS | prompt | patch] + pos_embed
            2) Encoder.forward_with_affinity:
                - 得到 encoded（末端 LN）
                - attn_weights: 每层 MHSA 的 attention map（可选）
                - affinities:   每层 raw/vis 亲和字典

        返回:
            encoded:   [B, N, D]
            attn_weights: list
            affinities:   list[dict]
        """
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights, affinities = self.encoder.forward_with_affinity(embedding_output, affinity_config, semantics)
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

    def forward(self, x, vis=False, semantics: torch.Tensor = None):
        """
        输入：x [B, 3, H, W]
        输出：若 vis=False 返回 logits；否则返回 (logits, attn_weights)
        """
        x, attn_weights = self.transformer(x, semantics)   # x: [B, 1+N, D]，attn_weights 典型形状：[num_layers, B, num_heads, 1+N, 1+N]（仅在 vis=True 时非空）
        logits = self.head(x[:, 0])     # 取 CLS token 做分类，x[:, 0] 始终是 CLS 的向量，维度 D=config.hidden_size

        if not vis:
            return logits
        return logits, attn_weights # attn_weights: num_layers, B, num_head, num_patches, num_patches [num_layers, B, h, N, N]

    def forward_with_affinity(self, x, affinity_config, vis=False, semantics: torch.Tensor = None):
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
                    - affinities:    每层 raw/vis 亲和字典
            2) 分类头:
                用 CLS token 的表征 x_seq[:, 0] 做线性分类

        返回:
            若 vis=False:
                logits, affinities
            若 vis=True:
                logits, attn_weights, affinities
        """
        x, attn_weights, affinities = self.transformer.forward_with_affinity(x, affinity_config, semantics)
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
        网格是什么：图像会按照patch大小切块，每个 patch 相当于一个 token，放回二维排布，就得到一个 patch 网格（g   rid）
            ViT 的 绝对位置编码是给网格里每个格子（每个 patch）分配一个 D 维向量
            很多实现把网格当作方形（H=W、且p相同），包括在输入之后进行裁剪
            预训练网格（G）= gs×gs故gs = sqrt(G),eg:224×224图,patch=16 → 14×14 网格 →G=196
            输入网格（N）为图像尺寸/patch对应的网格,eg:384×384图,仍 patch=16 → 24×24 网格 → N=576
        插值是什么：预训练的“位置参数”只有 G 份，但你现在需要 N 份。故预训练的位置编码“缩放”到新网格 —— 这就是“插值”
            位置编码是可学习参数，只有在预训练用到的那些格子位置上有“合理”的值
        双线性插值是什么：二维上对每个新坐标的值，在旧网格里找到它落在哪个2×2邻域中（四个最近的旧点），沿x方向做一次线性插值，再沿y方向对前一步结果再做一次线性插值
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
