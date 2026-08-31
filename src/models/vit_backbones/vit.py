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
from typing import Any, Dict, Optional, Tuple
# 8.21改动
# 原有 Win 下 os.path.join 会产生反斜杠 "\"，会影响从权重字典中取键（键名一般用 "/"）
# 因此改为从 posixpath 导入 join，确保键名分隔符始终为 "/"
# from os.path import join as pjoin  # 原有Win下os.path.join会产生反斜杠 \
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


def _offload_diagnostic_tree(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {
            key: _offload_diagnostic_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_offload_diagnostic_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_offload_diagnostic_tree(item) for item in value)
    return value


def _attach_prompt_continuity_to_layer(affinity, previous_output):
    if not isinstance(affinity, dict):
        return None
    current_input = affinity.get("_prompt_layer_input_vector")
    current_output = affinity.get("_prompt_layer_output_vector")
    if torch.is_tensor(previous_output) and torch.is_tensor(current_input):
        affinity["prompt_previous_output_to_current_input_cosine"] = (
            torch.nn.functional.cosine_similarity(
                previous_output, current_input, dim=-1, eps=1e-12
            )
        )
        affinity["prompt_previous_output_to_current_input_gap_norm"] = (
            current_input - previous_output
        ).norm(dim=-1)
    return current_output if torch.is_tensor(current_output) else None

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
        self._last_mediation_stats = None
        self._prompt_path_intervention = None
        self._last_prompt_path_intervention_stats = None

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

    def _apply_shared_attention_dropout(
        self,
        attention_probs: torch.Tensor,
        modified_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [ATTN-MED-DROPOUT-SYNC]
        让原始 attention row 和 mediated attention row 使用同一份 attention dropout 掩码。

        训练时 nn.Dropout 会把被保留的位置除以 keep_prob。这里手动生成同样语义的
        mask，并同时乘到 original/modified 两份概率上，避免主路径使用 dropout 后概率、
        mediated delta 却使用 dropout 前概率。
        """
        dropout_p = float(self.attn_dropout.p)
        if (not self.training) or dropout_p <= 0.0:
            return attention_probs, modified_probs
        keep_prob = 1.0 - dropout_p
        shared_mask = torch.empty_like(attention_probs).bernoulli_(keep_prob).div_(keep_prob)
        return attention_probs * shared_mask, modified_probs * shared_mask

    def _apply_prompt_path_intervention(
        self,
        attention_probs: torch.Tensor,
        prompt_length: int,
        semantic_length: int,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._last_prompt_path_intervention_stats = None
        intervention = self._prompt_path_intervention
        if not intervention:
            return attention_probs
        prompt_length = int(prompt_length)
        semantic_length = int(semantic_length)
        if prompt_length <= 0:
            return attention_probs
        sequence_length = int(attention_probs.shape[-1])
        prompt_slice = slice(1, 1 + prompt_length)
        patch_start = 1 + prompt_length
        patch_end = sequence_length - semantic_length
        if patch_end <= patch_start:
            raise ValueError(
                "Prompt attention intervention requires at least one visual Patch token"
            )
        patch_slice = slice(patch_start, patch_end)
        mode = str(intervention.get("mode", ""))
        target_layer = intervention.get("target_layer")
        if target_layer is not None and int(intervention.get("layer_index", -1)) != int(
            target_layer
        ):
            return attention_probs
        changed = attention_probs.clone()
        if mode == "relevance_edge_delete":
            deletion_mask = intervention.get("deletion_mask")
            if not torch.is_tensor(deletion_mask):
                raise ValueError("relevance_edge_delete requires deletion_mask")
            deletion_mask = deletion_mask.to(
                device=attention_probs.device, dtype=torch.bool
            )
            if deletion_mask.shape != attention_probs.shape:
                raise ValueError(
                    "relevance_edge_delete mask shape does not match attention"
                )
            row_all_deleted = deletion_mask.all(dim=-1)
            if bool(row_all_deleted.any()):
                keep_index = attention_probs.argmax(dim=-1, keepdim=True)
                deletion_mask = deletion_mask.clone()
                deletion_mask.scatter_(
                    -1,
                    keep_index,
                    deletion_mask.gather(-1, keep_index) & ~row_all_deleted.unsqueeze(-1),
                )
            deleted_mass = (
                attention_probs * deletion_mask.to(attention_probs.dtype)
            ).sum(dim=-1)
            original_mass = attention_probs.sum(dim=-1, keepdim=True)
            changed = changed.masked_fill(deletion_mask, 0.0)
            changed = (
                changed
                / changed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                * original_mass
            )
            self._last_prompt_path_intervention_stats = {
                "relevance_delete_applied": deleted_mass.new_ones(
                    int(deleted_mass.shape[0])
                ),
                "relevance_delete_mass": deleted_mass.mean(dim=(1, 2)),
                "relevance_delete_edge_ratio": deletion_mask.float().mean(
                    dim=(1, 2, 3)
                ),
                "relevance_delete_row_mass_abs_error": (
                    changed.sum(dim=-1, keepdim=True) - original_mass
                ).abs().mean(dim=(1, 2, 3)),
            }
            return changed
        if mode in {"prompt_patch_value_globalize", "prompt_value_zero"}:
            return changed
        if mode == "prompt_patch_uniform":
            prompt_patch_mass = changed[:, :, prompt_slice, patch_slice].sum(
                dim=-1, keepdim=True
            )
            changed[:, :, prompt_slice, patch_slice] = (
                prompt_patch_mass / float(patch_end - patch_start)
            )
            prompt_patch_mass_after = changed[
                :, :, prompt_slice, patch_slice
            ].sum(dim=-1)
            batch_size = int(changed.shape[0])
            self._last_prompt_path_intervention_stats = {
                "prompt_patch_uniform_applied": prompt_patch_mass.new_ones(
                    (batch_size,)
                ),
                "prompt_patch_uniform_mass_abs_error": (
                    prompt_patch_mass_after - prompt_patch_mass.squeeze(-1)
                ).abs().mean(dim=(1, 2)),
            }
            return changed
        if mode == "patch_prompt_uniform":
            patch_prompt_mass = changed[:, :, patch_slice, prompt_slice].sum(
                dim=-1, keepdim=True
            )
            changed[:, :, patch_slice, prompt_slice] = (
                patch_prompt_mass / float(prompt_length)
            )
            patch_prompt_mass_after = changed[
                :, :, patch_slice, prompt_slice
            ].sum(dim=-1)
            batch_size = int(changed.shape[0])
            self._last_prompt_path_intervention_stats = {
                "patch_prompt_uniform_applied": patch_prompt_mass.new_ones(
                    (batch_size,)
                ),
                "patch_prompt_uniform_mass_abs_error": (
                    patch_prompt_mass_after - patch_prompt_mass.squeeze(-1)
                ).abs().mean(dim=(1, 2)),
            }
            return changed
        if mode in {
            "attribute_concept_prompt_patch_block",
            "random_prompt_patch_block",
        }:
            if not torch.is_tensor(hidden_states):
                raise ValueError(
                    "Attribute-concept prompt intervention requires layer hidden states"
                )
            attribute_directions = intervention.get("attribute_directions")
            margin_weights = intervention.get("margin_weights")
            if not torch.is_tensor(attribute_directions) or not torch.is_tensor(
                margin_weights
            ):
                raise ValueError(
                    "Attribute-concept prompt intervention requires attribute directions and margin weights"
                )
            patch_hidden = hidden_states[:, patch_slice, :].detach().float()
            directions = attribute_directions.detach().to(
                device=patch_hidden.device, dtype=torch.float32
            )
            weights = margin_weights.detach().to(
                device=patch_hidden.device, dtype=torch.float32
            )
            if (
                directions.dim() != 2
                or weights.dim() != 2
                or directions.shape[0] != weights.shape[1]
                or directions.shape[1] != patch_hidden.shape[-1]
                or weights.shape[0] != patch_hidden.shape[0]
            ):
                raise ValueError(
                    "Attribute-concept prompt intervention tensors have incompatible shapes"
                )
            patch_unit = torch.nn.functional.normalize(
                patch_hidden, dim=-1, eps=1e-12
            )
            direction_unit = torch.nn.functional.normalize(
                directions, dim=-1, eps=1e-12
            )
            attribute_scores = torch.matmul(
                patch_unit, direction_unit.transpose(0, 1)
            )
            weighted_scores = attribute_scores * weights.unsqueeze(1)
            positive_scores = torch.relu(weighted_scores).sum(dim=-1)
            fallback = positive_scores.sum(dim=-1) <= 1e-12
            concept_scores = torch.where(
                fallback.unsqueeze(-1),
                weighted_scores.abs().sum(dim=-1),
                positive_scores,
            )
            patch_count = int(patch_end - patch_start)
            patch_ratio = float(intervention.get("patch_ratio", 0.2))
            if not 0.0 < patch_ratio < 1.0:
                raise ValueError(
                    "Attribute-concept prompt intervention patch_ratio must be between 0 and 1"
                )
            selected_count = min(
                patch_count - 1,
                max(1, int(math.ceil(patch_ratio * patch_count))),
            )
            if mode == "attribute_concept_prompt_patch_block":
                selected_indices = concept_scores.topk(
                    selected_count, dim=-1, largest=True
                ).indices
            else:
                generator = torch.Generator(device=attention_probs.device)
                random_seed = int(intervention.get("random_seed", 0))
                layer_index = int(intervention.get("layer_index", 0))
                generator.manual_seed(
                    int((random_seed + 1000003 * layer_index) % (2 ** 63 - 1))
                )
                random_scores = torch.rand(
                    concept_scores.shape,
                    generator=generator,
                    device=attention_probs.device,
                    dtype=torch.float32,
                )
                selected_indices = random_scores.topk(
                    selected_count, dim=-1, largest=True
                ).indices
            selected_mask = torch.zeros_like(concept_scores, dtype=torch.bool)
            selected_mask.scatter_(1, selected_indices, True)
            selected = selected_mask[:, None, None, :]
            prompt_patch = changed[:, :, prompt_slice, patch_slice]
            patch_mass_before = prompt_patch.sum(dim=-1)
            selected_mass_before = (
                prompt_patch * selected.to(dtype=prompt_patch.dtype)
            ).sum(dim=-1, keepdim=True)
            redistributed = selected_mass_before / float(
                patch_count - selected_count
            )
            prompt_patch = torch.where(
                selected,
                torch.zeros_like(prompt_patch),
                prompt_patch + redistributed,
            )
            changed[:, :, prompt_slice, patch_slice] = prompt_patch
            patch_mass_after = prompt_patch.sum(dim=-1)
            selected_mass_after = (
                prompt_patch * selected.to(dtype=prompt_patch.dtype)
            ).sum(dim=-1)
            selected_score = concept_scores.gather(1, selected_indices).mean(dim=-1)
            unselected_score = (
                concept_scores.masked_fill(selected_mask, 0.0).sum(dim=-1)
                / float(patch_count - selected_count)
            )
            batch_size = int(concept_scores.shape[0])
            self._last_prompt_path_intervention_stats = {
                "concept_intervention_applied": concept_scores.new_ones(
                    (batch_size,)
                ),
                "concept_intervention_selected_patch_ratio": concept_scores.new_full(
                    (batch_size,), float(selected_count / patch_count)
                ),
                "concept_intervention_selected_score_mean": selected_score,
                "concept_intervention_unselected_score_mean": unselected_score,
                "concept_intervention_selection_score_gap": (
                    selected_score - unselected_score
                ),
                "concept_intervention_selected_prompt_attention_mass_before": (
                    selected_mass_before.squeeze(-1).mean(dim=(1, 2))
                ),
                "concept_intervention_selected_prompt_attention_mass_after": (
                    selected_mass_after.mean(dim=(1, 2))
                ),
                "concept_intervention_prompt_patch_mass_abs_error": (
                    patch_mass_after - patch_mass_before
                ).abs().mean(dim=(1, 2)),
                "concept_intervention_fallback_ratio": fallback.float(),
                "concept_intervention_targeted": concept_scores.new_full(
                    (batch_size,),
                    1.0
                    if mode == "attribute_concept_prompt_patch_block"
                    else 0.0,
                ),
            }
            return changed
        if mode in {
            "transport_prompt_patch_block",
            "transport_random_patch_block",
        }:
            selected_indices = intervention.get("selected_patch_indices")
            reference_scores = intervention.get("reference_patch_scores")
            if not torch.is_tensor(selected_indices) or not torch.is_tensor(
                reference_scores
            ):
                raise ValueError(
                    "Transport prompt intervention requires fixed selected indices and reference scores"
                )
            selected_indices = selected_indices.detach().to(
                device=attention_probs.device, dtype=torch.long
            )
            reference_scores = reference_scores.detach().to(
                device=attention_probs.device, dtype=torch.float32
            )
            patch_count = int(patch_end - patch_start)
            if (
                selected_indices.dim() != 2
                or reference_scores.dim() != 2
                or selected_indices.shape[0] != attention_probs.shape[0]
                or reference_scores.shape
                != (attention_probs.shape[0], patch_count)
                or selected_indices.shape[1] <= 0
                or selected_indices.shape[1] >= patch_count
                or int(selected_indices.min().item()) < 0
                or int(selected_indices.max().item()) >= patch_count
            ):
                raise ValueError(
                    "Transport prompt intervention tensors have incompatible shapes or indices"
                )
            selected_count = int(selected_indices.shape[1])
            selected_mask = torch.zeros_like(reference_scores, dtype=torch.bool)
            selected_mask.scatter_(1, selected_indices, True)
            selected = selected_mask[:, None, None, :]
            prompt_patch = changed[:, :, prompt_slice, patch_slice]
            patch_mass_before = prompt_patch.sum(dim=-1)
            selected_mass_before = (
                prompt_patch * selected.to(dtype=prompt_patch.dtype)
            ).sum(dim=-1, keepdim=True)
            redistributed = selected_mass_before / float(
                patch_count - selected_count
            )
            prompt_patch = torch.where(
                selected,
                torch.zeros_like(prompt_patch),
                prompt_patch + redistributed,
            )
            changed[:, :, prompt_slice, patch_slice] = prompt_patch
            patch_mass_after = prompt_patch.sum(dim=-1)
            selected_mass_after = (
                prompt_patch * selected.to(dtype=prompt_patch.dtype)
            ).sum(dim=-1)
            selected_score = reference_scores.gather(
                1, selected_indices
            ).mean(dim=-1)
            unselected_score = (
                reference_scores.masked_fill(selected_mask, 0.0).sum(dim=-1)
                / float(patch_count - selected_count)
            )
            batch_size = int(reference_scores.shape[0])
            self._last_prompt_path_intervention_stats = {
                "transport_intervention_applied": reference_scores.new_ones(
                    (batch_size,)
                ),
                "transport_intervention_selected_patch_ratio": (
                    reference_scores.new_full(
                        (batch_size,), float(selected_count / patch_count)
                    )
                ),
                "transport_intervention_selected_score_mean": selected_score,
                "transport_intervention_unselected_score_mean": unselected_score,
                "transport_intervention_selection_score_gap": (
                    selected_score - unselected_score
                ),
                "transport_intervention_selected_prompt_attention_mass_before": (
                    selected_mass_before.squeeze(-1).mean(dim=(1, 2))
                ),
                "transport_intervention_selected_prompt_attention_mass_after": (
                    selected_mass_after.mean(dim=(1, 2))
                ),
                "transport_intervention_prompt_patch_mass_abs_error": (
                    patch_mass_after - patch_mass_before
                ).abs().mean(dim=(1, 2)),
                "transport_intervention_targeted": reference_scores.new_full(
                    (batch_size,),
                    1.0 if mode == "transport_prompt_patch_block" else 0.0,
                ),
            }
            return changed
        if mode == "prompt_read_block":
            prompt_patch_mass_before = changed[
                :, :, prompt_slice, patch_slice
            ].sum(dim=-1)
            changed[:, :, prompt_slice, patch_slice] = 0.0
            prompt_rows = changed[:, :, prompt_slice, :]
            changed[:, :, prompt_slice, :] = prompt_rows / prompt_rows.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            batch_size = int(changed.shape[0])
            self._last_prompt_path_intervention_stats = {
                "prompt_read_block_applied": prompt_patch_mass_before.new_ones(
                    (batch_size,)
                ),
                "prompt_read_block_patch_mass_before": (
                    prompt_patch_mass_before.mean(dim=(1, 2))
                ),
                "prompt_read_block_patch_mass_after": (
                    prompt_patch_mass_before.new_zeros((batch_size,))
                ),
            }
            return changed
        if mode == "prompt_write_block":
            changed[:, :, :1, prompt_slice] = 0.0
            changed[:, :, patch_start:, prompt_slice] = 0.0
            non_prompt_rows = torch.cat(
                (changed[:, :, :1, :], changed[:, :, patch_start:, :]), dim=-2
            )
            non_prompt_rows = non_prompt_rows / non_prompt_rows.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            changed[:, :, :1, :] = non_prompt_rows[:, :, :1, :]
            changed[:, :, patch_start:, :] = non_prompt_rows[:, :, 1:, :]
            return changed
        raise ValueError(f"Unsupported prompt attention intervention: {mode}")

    def _apply_prompt_value_intervention(
        self,
        context_layer: torch.Tensor,
        attention_probs: torch.Tensor,
        value_layer: torch.Tensor,
        prompt_length: int,
        semantic_length: int,
    ) -> torch.Tensor:
        intervention = self._prompt_path_intervention
        if not intervention:
            return context_layer
        mode = str(intervention.get("mode", ""))
        if mode not in {"prompt_patch_value_globalize", "prompt_value_zero"}:
            return context_layer
        prompt_length = int(prompt_length)
        semantic_length = int(semantic_length)
        if prompt_length <= 0:
            return context_layer
        target_layer = intervention.get("target_layer")
        if target_layer is not None and int(intervention.get("layer_index", -1)) != int(
            target_layer
        ):
            return context_layer

        sequence_length = int(attention_probs.shape[-1])
        prompt_slice = slice(1, 1 + prompt_length)
        if mode == "prompt_value_zero":
            prompt_attention = attention_probs[:, :, :, prompt_slice]
            prompt_values = value_layer[:, :, prompt_slice, :]
            prompt_context = torch.matmul(prompt_attention, prompt_values)
            changed = context_layer - prompt_context
            with torch.no_grad():
                batch_size = int(context_layer.shape[0])
                self._last_prompt_path_intervention_stats = {
                    "prompt_value_zero_applied": prompt_context.new_ones(batch_size),
                    "prompt_value_zero_context_delta_norm": prompt_context.detach().float().norm(
                        dim=-1
                    ).mean(dim=(1, 2)),
                    "prompt_value_zero_attention_mass_abs_error": prompt_context.new_zeros(
                        batch_size
                    ),
                }
            return changed
        patch_start = 1 + prompt_length
        patch_end = sequence_length - semantic_length
        if patch_end <= patch_start:
            raise ValueError(
                "Prompt value globalization requires at least one visual Patch token"
            )
        patch_slice = slice(patch_start, patch_end)
        prompt_patch_attention = attention_probs[:, :, prompt_slice, patch_slice]
        patch_values = value_layer[:, :, patch_slice, :]
        mean_patch_value = patch_values.mean(dim=-2, keepdim=True)
        original_prompt_patch_context = torch.matmul(
            prompt_patch_attention, patch_values
        )
        globalized_prompt_patch_context = (
            prompt_patch_attention.sum(dim=-1, keepdim=True) * mean_patch_value
        )
        changed = context_layer.clone()
        changed[:, :, prompt_slice, :] = (
            changed[:, :, prompt_slice, :]
            - original_prompt_patch_context
            + globalized_prompt_patch_context
        )

        with torch.no_grad():
            batch_size = int(context_layer.shape[0])
            patch_value_dispersion = (
                patch_values.detach().float()
                - mean_patch_value.detach().float()
            ).norm(dim=-1).mean(dim=(1, 2))
            context_delta = (
                globalized_prompt_patch_context.detach().float()
                - original_prompt_patch_context.detach().float()
            )
            self._last_prompt_path_intervention_stats = {
                "prompt_value_globalize_applied": context_delta.new_ones(
                    (batch_size,)
                ),
                "prompt_value_globalize_patch_value_dispersion_before": (
                    patch_value_dispersion
                ),
                "prompt_value_globalize_patch_value_dispersion_after": (
                    patch_value_dispersion.new_zeros((batch_size,))
                ),
                "prompt_value_globalize_prompt_context_delta_norm": (
                    context_delta.norm(dim=-1).mean(dim=(1, 2))
                ),
                "prompt_value_globalize_attention_mass_abs_error": (
                    context_delta.new_zeros((batch_size,))
                ),
            }
        return changed

    @staticmethod
    def _row_normalize(x: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
        """
        对局部 attention 子块做行归一化。

        这里不使用 softmax，因为 SOURCE="probs" 时输入已经是 full softmax
        后的概率子块；此时只需要在局部区域内转成条件分布。
        """
        return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)

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
        mediated attention 需要基于 P/V/S 三段切出局部路径；
        visual_block 模式只改 P->V/S->V，full_row 模式会构造 P/S 的完整 attention row。
        """
        prompt_length = int(prompt_length)
        semantic_length = int(semantic_length)
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

        if mass_mode == "row_preserve":
            new_visual_mass = direct_visual_mass
        elif mass_mode == "block_redistribute":
            # block_redistribute 允许同一组 query token 之间重新分配 visual mass。
            # beta_mass 控制“保留原始 visual mass 分布”和“采用 mediated visual mass 分布”的混合比例：
            #   beta_mass=0 时完全使用原始 attention 的 mass 分布；
            #   beta_mass=1 时完全使用 mediation 计算出的 mass 分布；
            #   中间值则做线性插值，以避免 mass 路由变化过猛。
            if not 0.0 <= float(beta_mass) <= 1.0:
                raise ValueError("ATTENTION_MEDIATION beta mass must be in [0, 1] when MASS_MODE='block_redistribute'.")
            if mediated_mass_score is None:
                raise ValueError("block_redistribute requires mediated_mass_score.")

            # direct_visual_mass 的形状为 [B, H, Q, 1]，表示每个 query row 原本分配给 visual block 的总概率质量。
            # 去掉最后一维后得到 [B, H, Q]，方便在 query维度上做归一化和重分配。
            direct_mass_per_query = direct_visual_mass.squeeze(-1)

            # direct_total_mass 是同一组 query 在 visual block 上的总 mass。
            # block_redistribute 只改变这些 mass 在 query 之间的分布，不凭空增加或减少这一组 query 合计投向 visual block 的概率质量。
            direct_total_mass = direct_mass_per_query.sum(dim=-1, keepdim=True)

            # 将原始 visual mass 归一化成 query 维度上的分布：
            # 每个位置表示该 query 占整组 visual mass 的比例，而不是完整 attention row中的概率值。
            direct_mass_dist = self._row_normalize(direct_mass_per_query, dim=-1)

            # mediated_mass_score 是 mediation 路径给出的 query 级 visual mass 打分。
            # squeeze 后同样归一化为分布，保证后续混合的是两个可比较的 query 分布。
            mediated_mass_dist = self._row_normalize(mediated_mass_score.squeeze(-1), dim=-1)

            # 对原始 query mass 分布和 mediated query mass 分布做线性插值。
            # 这里混合的是“分布形状”，总 mass 仍由 direct_total_mass 保持。
            mixed_mass_dist = (1.0 - float(beta_mass)) * direct_mass_dist + float(beta_mass) * mediated_mass_dist

            # 把混合后的 query 分布重新乘回整组 visual 总 mass，得到每个 query 新的
            # visual mass，并恢复到 [B, H, Q, 1]，以便和 mediated_visual_cond 相乘。
            new_visual_mass = (direct_total_mass * mixed_mass_dist).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported ATTENTION_MEDIATION.MASS_MODE='{mass_mode}'.")

        modified_row = direct_full_row.clone()
        modified_row[:, :, :, visual_slice] = mediated_visual_cond * new_visual_mass

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
        route_scope: str,
        mass_mode: str,
        beta_mass: float,
    ) -> torch.Tensor:
        """
        构造 prompt query 的完整 modified attention row。

        输入的 attention row 按 [CLS | P(prompt) | V(visual) | S(semantic)] 排列。
        本函数只负责 prompt token 作为 query 时的 P->* 行：
        - visual_block 只改 P->V 子块，再把它拼回原始 P->[CLS|P|V|S] 完整行；
        - full_row 直接通过中介路径构造 P->[CLS|P|V|S] 完整行。
        """
        # 取出 mediation 需要的三块原始注意力：
        # a_sp: semantic query 到 prompt key，形状 [B, H, S, P]；
        # a_ps: prompt query 到 semantic key，形状 [B, H, P, S]；
        # a_sv: semantic query 到 visual key，形状 [B, H, S, V]。
        # route_base 可能是 attention_scores，也可能是 attention_probs；
        # _conditional_normalize 会根据 source 选择 softmax 或普通归一化。
        a_sp = route_base[:, :, semantic_slice, prompt_slice]
        a_ps = route_base[:, :, prompt_slice, semantic_slice]
        a_sv = route_base[:, :, semantic_slice, visual_slice]

        # 把 S->V 归一化成“给定某个 semantic query 时，它如何分配 visual key”的条件分布。
        # 后续两种 prompt route 都会把 prompt 与 semantic 的关系投影到这组 S->V 分布上，
        # 从而得到间接的 P->V 分布。
        sv_cond = self._conditional_normalize(a_sv, dim=-1, source=source)

        # prompt query 的原始完整 attention row，形状 [B, H, P, all_tokens]。
        # visual_block 模式会在这个完整行上只替换 visual_slice 对应的 P->V 子块；
        # full_row 模式则不会用它拼接，而是直接返回 mediated_full。
        direct_prompt_row = attention_probs[:, :, prompt_slice, :]

        if route == "S_to_P_and_V":
            # 路径含义：用 semantic token 作为中介，同时参考 S->P 与 S->V。
            # a_sp 原始形状是 [B, H, S, P]，这里沿 S 维归一化，得到“对每个 prompt，
            # 哪些 semantic 更相关”的条件分布。之后转置成 [B, H, P, S]，才能从 prompt
            # query 聚合 semantic 侧的信息。
            sp_cond = self._conditional_normalize(a_sp, dim=-2, source=source)
            if route_scope == "full_row":
                # full_row 模式：不是只构造 P->V，而是先得到每个 semantic query 的完整
                # S->[CLS|P|V|S] 条件分布，再按 P<-S 的权重聚合，得到完整的
                # P->[CLS|P|V|S] mediated row。
                s_full_cond = self._conditional_normalize(route_base[:, :, semantic_slice, :], dim=-1, source=source)
                mediated_full = torch.matmul(sp_cond.transpose(-1, -2), s_full_cond)
            else:
                # visual_block 模式：只用 P<-S 的权重聚合 S->V 条件分布，得到 mediated P->V。
                # mediated_visual_base 形状 [B, H, P, V]，表示 prompt query 经 semantic
                # 中介后应当关注哪些 visual token。
                mediated_visual_base = torch.matmul(sp_cond.transpose(-1, -2), sv_cond)

                # 额外计算 semantic query 原本分配给 visual block 的总 mass，形状 [B, H, S, 1]。
                # block_redistribute 模式会通过 P<-S 权重把这些 mass 投影到 prompt query 上，
                # 用来决定每个 prompt row 的新 visual 总质量；row_preserve 模式下也会传入，
                # 但 _compose_visual_block_row 不会使用它来改变总 mass。
                semantic_visual_mass = attention_probs[:, :, semantic_slice, visual_slice].sum(dim=-1, keepdim=True)
                mediated_mass_score = torch.matmul(sp_cond.transpose(-1, -2), semantic_visual_mass)
        elif route == "P_to_S_to_V":
            # 路径含义：先看 prompt query 直接关注哪些 semantic key，再沿这些 semantic
            # token 的 S->V 分布继续走到 visual token，即 P->S->V。
            # ps_cond 形状 [B, H, P, S]，每个 prompt row 内对 semantic key 归一化。
            ps_cond = self._conditional_normalize(a_ps, dim=-1, source=source)
            if route_scope == "full_row":
                # full_row 模式：用 P->S 权重聚合 semantic query 的完整 attention row，
                # 得到完整 P->[CLS|P|V|S] mediated row。
                s_full_cond = self._conditional_normalize(route_base[:, :, semantic_slice, :], dim=-1, source=source)
                mediated_full = torch.matmul(ps_cond, s_full_cond)
            else:
                # visual_block 模式：用 P->S 权重聚合 S->V 条件分布，得到 mediated P->V。
                mediated_visual_base = torch.matmul(ps_cond, sv_cond)

                # 与上一个分支一样，估计经 P->S 路径传递到每个 prompt query 的 visual mass。
                # 这个值只决定 visual block 的总质量如何分配；P->V 内部具体落在哪些 visual token
                # 由 mediated_visual_base / mediated_visual_cond 决定。
                semantic_visual_mass = attention_probs[:, :, semantic_slice, visual_slice].sum(dim=-1, keepdim=True)
                mediated_mass_score = torch.matmul(ps_cond, semantic_visual_mass)
        else:
            raise ValueError(
                f"Unsupported ATTENTION_MEDIATION.PROMPT_ROUTE='{route}'. "
                "Expected S_to_P_and_V / P_to_S_to_V."
            )

        if route_scope == "full_row":
            # full_row 已经生成完整 P->[CLS|P|V|S] 行。这里再做一次 row normalize，
            # 防止 score/prob source 的数值路径或矩阵乘法带来微小归一化误差。
            mediated_full = self._row_normalize(mediated_full, dim=-1)
            return mediated_full
        if route_scope == "visual_block":
            # visual_block 只替换 P->V 子块：先把 mediated_visual_base 归一化成
            # 每个 prompt query 在 visual tokens 内部的条件分布，再交给
            # _compose_visual_block_row 拼回完整 row。
            mediated_visual_cond = self._row_normalize(mediated_visual_base, dim=-1)

            # _compose_visual_block_row 会根据 mass_mode 决定 P->V 的总质量：
            # - row_preserve: 每个 prompt row 保持原来的 visual mass，只改 visual 内部分布；
            # - block_redistribute: 在 prompt queries 之间重分配 visual mass，强度由 beta_mass 控制。
            # 非 visual 区域会按剩余 mass 等比例缩放，保证最终完整 row 仍然和为 1。
            return self._compose_visual_block_row(
                direct_prompt_row,
                visual_slice,
                mediated_visual_cond,
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
            return mediated_full
        if route_scope == "visual_block":
            mediated_visual_cond = self._row_normalize(mediated_visual_base, dim=-1)
            return self._compose_visual_block_row(
                direct_semantic_row,
                visual_slice,
                mediated_visual_cond,
                mass_mode,
                beta_mass,
                mediated_mass_score,
            )
        raise ValueError(f"Unsupported ATTENTION_MEDIATION.ROUTE_SCOPE='{route_scope}'.")

    def _compute_attention_mediation(
        self,
        attention_scores: torch.Tensor,
        attention_probs: torch.Tensor,
        prompt_length: int,
        semantic_length: int,
        mediation_config: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        在当前层 MHSA 内部构造 mediated attention correction。

        [ATTN-MED-DROPOUT-SYNC]
        这里只构造 dropout 前的目标概率分布 target_probs / modified_probs。若 SOURCE=scores，再用
        log(target_probs)-log(original_probs) 形成 score bias，并重新对完整 row 做 softmax。
        这样 score 分支最终仍由 full softmax 产生合法 attention 概率。

        注意：delta_context 不在这里计算，而是在 _scaled_attention() 中和主路径共享
        attention dropout 之后再计算，保证训练时两条路径的随机性一致。
        """
        if not mediation_config or not mediation_config.get("enable", False):
            return None

        # 读取本层 mediation 的核心路由配置：
        # source 决定用 score logits 还是 softmax 后的 probs 作为构造 mediated row 的基础；
        # route_scope 决定只改 visual block，还是直接构造完整 attention row；
        # mass_mode 决定 visual block 的总 mass 是逐 row 保持，还是在 query 之间重分配。
        source = str(mediation_config.get("source"))
        route_scope = str(mediation_config.get("route_scope"))
        mass_mode = str(mediation_config.get("mass_mode"))

        # 这里尽早做配置合法性检查，避免后续构造 target_probs 时产生难定位的 shape
        # 或概率归一化错误。full_row 已经直接产出完整 row，因此不再支持额外的block_redistribute mass 重分配。
        if source not in {"scores", "probs"}:
            raise ValueError("ATTENTION_MEDIATION.SOURCE must be scores or probs.")
        if mass_mode not in {"row_preserve", "block_redistribute"}:
            raise ValueError("ATTENTION_MEDIATION.MASS_MODE must be row_preserve or block_redistribute.")
        if route_scope == "full_row" and mass_mode == "block_redistribute":
            raise ValueError("ATTENTION_MEDIATION full_row already builds a full probability row; use MASS_MODE='row_preserve'.")

        # 根据当前序列布局 [CLS | prompt_tokens | visual_tokens | semantic_tokens]
        # 切出 P/V/S 三段。mediation 只重写 prompt query 和 semantic query 的 attention row，
        # visual query 与 CLS query 保持原始 attention_probs。
        slices = self._attention_mediation_slices(attention_probs.size(-1), prompt_length, semantic_length)
        prompt_slice = slices["prompt"]
        visual_slice = slices["visual"]
        semantic_slice = slices["semantic"]

        # route_base 是下游构造 mediated row 时读取的“原始注意力基准”。
        # SOURCE=scores 时使用未 softmax 的 logits，适合在 score 空间做路由后再回到 full softmax；
        # SOURCE=probs 时直接使用 softmax 概率，构造出的 target_probs 会作为最终 modified_probs。
        route_base = attention_scores if source == "scores" else attention_probs

        # 先复制完整 attention 概率矩阵作为 target_probs。
        # 后面只覆盖 prompt rows 和 semantic rows；其它 query rows 仍沿用原始 attention_probs，
        # 从而把 mediation 的影响范围限制在 P->* 与 S->*。
        target_probs = attention_probs.clone()

        # 构造 prompt token 作为 query 时的 mediated attention row。
        # prompt_route 控制 prompt 侧如何借助 semantic/visual 路径生成目标分布；
        # beta_prompt_mass 只在 block_redistribute 下影响 prompt query 之间的 visual mass 重分配强度。
        target_probs[:, :, prompt_slice, :] = self._build_prompt_mediated_row(
            route_base,
            attention_probs,
            prompt_slice,
            visual_slice,
            semantic_slice,
            source,
            str(mediation_config.get("prompt_route")),
            route_scope,
            mass_mode,
            float(mediation_config.get("beta_prompt_mass", 0.0)),
        )

        # 构造 semantic token 作为 query 时的 mediated attention row。
        # semantic_route 与 prompt 侧含义对应，但方向变为 semantic 侧读取
        # prompt/visual 信息；beta_semantic_mass 控制 semantic query 之间的 visual mass 重分配强度。
        target_probs[:, :, semantic_slice, :] = self._build_semantic_mediated_row(
            route_base,
            attention_probs,
            prompt_slice,
            visual_slice,
            semantic_slice,
            source,
            str(mediation_config.get("semantic_route")),
            route_scope,
            mass_mode,
            float(mediation_config.get("beta_semantic_mass", 0.0)),
        )

        if source == "scores":
            # score 空间不能直接比较普通数值和。这里使用 log-ratio bias：
            # score' = score + log(target_prob) - log(original_prob)，再由 full softmax 重新归一化。
            # 这样做的好处是仍然让完整 attention row 经过一次统一 softmax，
            # 保证 score 分支输出的 modified_probs 是合法概率分布，并保留 logits 级调制的语义。
            score_bias = torch.log(target_probs.clamp_min(1e-8)) - torch.log(attention_probs.clamp_min(1e-8))
            modified_probs = torch.softmax(attention_scores + score_bias, dim=-1)
        elif source == "probs":
            # probs 分支已经直接构造出完整概率矩阵，因此不再回到 logits 空间。
            # _build_*_mediated_row 内部已经保证被替换的 rows 完成归一化。
            modified_probs = target_probs
        else:
            raise ValueError(f"Unsupported ATTENTION_MEDIATION.SOURCE='{source}'.")

        # [ATTN-MED-DROPOUT-SYNC]
        # 返回 modified_probs，而不是在这里直接计算 delta_context。
        # _scaled_attention() 会先把 original/modified 两份 probs 乘同一份 attention dropout mask，
        # 再计算 delta_attention_output 和 mediated_attention_output。
        return {
            "modified_probs": modified_probs,
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
        intervention_hidden_states: Optional[torch.Tensor] = None,
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
        if self._prompt_path_intervention and mediation_config and mediation_config.get("enable", False):
            raise ValueError(
                "Prompt path intervention cannot be combined with Attention Mediation"
            )
        attention_probs = self._apply_prompt_path_intervention(
            attention_probs,
            prompt_length,
            semantic_length,
            intervention_hidden_states,
        )
        monitor_attention_probs = attention_probs
        weights = monitor_attention_probs if self.vis else None     # 用于可视化
        # ATTENTION_MEDIATION 的介入点在 softmax 之后、attention dropout 之前。
        # 这样既能拿到 score/prob 两种来源，也能保证 correction 使用未 dropout 的完整注意力分布。
        mediation = self._compute_attention_mediation(
            attention_scores,
            attention_probs,
            prompt_length,
            semantic_length,
            mediation_config,
        )

        if mediation is not None:
            # [ATTN-MED-DROPOUT-SYNC]
            # mediation 先构造 dropout 前的 modified_probs；这里让原始主路径和 mediated 分支
            # 共享同一份 attention dropout mask，然后再分别计算主路径 context 与 mediated delta。
            modified_probs = mediation.pop("modified_probs")
            original_probs_monitor = attention_probs.detach().float()
            modified_probs_monitor = modified_probs.detach().float()
            midpoint = 0.5 * (original_probs_monitor + modified_probs_monitor)
            original_kl = (
                original_probs_monitor
                * (original_probs_monitor.clamp_min(1e-12).log() - modified_probs_monitor.clamp_min(1e-12).log())
            ).sum(dim=-1)
            js = 0.5 * (
                original_probs_monitor
                * (original_probs_monitor.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
            ).sum(dim=-1) + 0.5 * (
                modified_probs_monitor
                * (modified_probs_monitor.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
            ).sum(dim=-1)
            attention_probs, modified_probs = self._apply_shared_attention_dropout(attention_probs, modified_probs)

            delta_probs = modified_probs - attention_probs
            delta_context = torch.matmul(delta_probs, value_layer)
            mediated_context = torch.matmul(modified_probs, value_layer)
            mediation["delta_attention_output"] = self._context_to_attention_output(delta_context, include_bias=False)
            mediation["mediated_attention_output"] = self._context_to_attention_output(mediated_context, include_bias=True)
            with torch.no_grad():
                delta_output = mediation["delta_attention_output"].detach().float()
                mediation["monitor_stats"] = {
                    "delta_prob_abs_mean": float(delta_probs.detach().float().abs().mean().item()),
                    "delta_prob_l2": float(delta_probs.detach().float().pow(2).mean().sqrt().item()),
                    "delta_output_abs_mean": float(delta_output.abs().mean().item()),
                    "delta_output_l2": float(delta_output.pow(2).mean().sqrt().item()),
                    "attention_kl": float(original_kl.mean().item()),
                    "attention_js": float(js.mean().item()),
                    "activation_sample_ratio": float(
                        (delta_probs.detach().float().abs().reshape(delta_probs.shape[0], -1).sum(dim=1) > 1e-12)
                        .float().mean().item()
                    ),
                }
                base_context = torch.matmul(attention_probs.detach(), value_layer.detach()).float()
                delta_context_value = delta_context.detach().float()
                mediation["monitor_stats"]["delta_output_to_base_ratio"] = float(
                    delta_context_value.norm(dim=-1).mean().item()
                    / max(float(base_context.norm(dim=-1).mean().item()), 1e-12)
                )
        else:
            attention_probs = self.attn_dropout(attention_probs)

        # 标准 MHSA 主路径不被替换；mediated correction 作为额外 delta 在 Block.forward 中合并。
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self._apply_prompt_value_intervention(
            context_layer,
            attention_probs,
            value_layer,
            prompt_length,
            semantic_length,
        )
        attention_output = self._context_to_attention_output(context_layer, include_bias=True)
        return attention_output, weights, mediation, monitor_attention_probs

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
        attention_output, weights, mediation, _ = self._scaled_attention(
            query_layer,
            key_layer,
            value_layer,
            semantic_length,
            block_s_to_cls,
            mediation_config,
            prompt_length,
            hidden_states,
        )
        return attention_output, weights, mediation

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
        attention_output, weights, mediation, monitor_attention_probs = self._scaled_attention(
            query_layer,
            key_layer,
            value_layer,
            semantic_length,
            block_s_to_cls,
            mediation_config,
            prompt_length,
            hidden_states,
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
        return (
            attention_output,
            weights,
            query_layer,
            key_layer,
            value_layer,
            mediation,
            monitor_attention_probs,
        )

    def compute_prompt_visual_monitors(
        self,
        query_layer,
        key_layer,
        prompt_length,
        semantic_length=0,
        *,
        value_layer=None,
        attention_probs=None,
        detach=True,
        include_visual_normalizations=True,
    ):
        """
        统一导出主干 prompt/visual 的原始亲和矩阵。

        这一版直接取代旧的 compute_affinity：
        - 不再接收 normalize
        - 主输出一律保留 raw logits
        - 额外附带仅用于画图的 min-max 归一化版本

        返回 QpQv、KpKv 辅助关系，以及当前实际 token 类型之间全部有向跨类型
        Q-K relation 的 raw / vis 矩阵。
        """
        q_base = query_layer.detach() if detach else query_layer
        k_base = key_layer.detach() if detach else key_layer
        v_base = value_layer.detach() if detach and torch.is_tensor(value_layer) else value_layer

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
        q_cls = q_base[:, :, :1, :]
        k_cls = k_base[:, :, :1, :]
        k_prompt = k_base[:, :, prompt_slice, :]
        k_patch = k_base[:, :, patch_slice, :]
        k_semantic = k_base[:, :, semantic_slice, :]
        scale = 1.0 / math.sqrt(self.attention_head_size)

        monitors = {}
        if torch.is_tensor(attention_probs):
            attention_probs = attention_probs.detach() if detach else attention_probs
            if attention_probs.shape[-2:] != (q_base.size(2), k_base.size(2)):
                raise ValueError(
                    "Provided Attention probability shape does not match projected Q/K sequence lengths"
                )
        else:
            attention_probs = torch.softmax(
                torch.matmul(q_base, k_base.transpose(-1, -2)) * scale,
                dim=-1,
            )
        monitors["AcKv_attn"] = attention_probs[:, :, :1, patch_slice]
        if prompt_length > 0:
            monitors["AcKp_attn"] = attention_probs[:, :, :1, prompt_slice]
            monitors["ApKv_attn"] = attention_probs[:, :, prompt_slice, patch_slice]
            monitors["AvKp_attn"] = attention_probs[:, :, patch_slice, prompt_slice]
            monitors["ApKc_attn"] = attention_probs[:, :, prompt_slice, :1]
            if torch.is_tensor(v_base):
                cls_probs = attention_probs[:, :, :1, :]
                prompt_context = torch.matmul(cls_probs[:, :, :, prompt_slice], v_base[:, :, prompt_slice, :])
                patch_context = torch.matmul(cls_probs[:, :, :, patch_slice], v_base[:, :, patch_slice, :])
                total_context = torch.matmul(cls_probs, v_base)
                prompt_output = torch.nn.functional.linear(
                    self._merge_heads(prompt_context), self.out.weight, None
                ).squeeze(1)
                patch_output = torch.nn.functional.linear(
                    self._merge_heads(patch_context), self.out.weight, None
                ).squeeze(1)
                total_output = torch.nn.functional.linear(
                    self._merge_heads(total_context), self.out.weight, None
                ).squeeze(1)
                prompt_norm = prompt_output.norm(dim=-1)
                patch_norm = patch_output.norm(dim=-1)
                monitors.update({
                    "cls_prompt_value_contribution_norm": prompt_norm,
                    "cls_patch_value_contribution_norm": patch_norm,
                    "cls_prompt_value_contribution_share": (
                        prompt_norm / (prompt_norm + patch_norm).clamp_min(1e-12)
                    ),
                    "cls_prompt_value_to_total_cosine": torch.nn.functional.cosine_similarity(
                        prompt_output, total_output, dim=-1, eps=1e-12
                    ),
                    "cls_prompt_value_to_patch_cosine": torch.nn.functional.cosine_similarity(
                        prompt_output, patch_output, dim=-1, eps=1e-12
                    ),
                    "_cls_prompt_value_contribution_vector": prompt_output,
                })
                prompt_probs = attention_probs[:, :, prompt_slice, :]
                prompt_patch_context = torch.matmul(
                    prompt_probs[:, :, :, patch_slice],
                    v_base[:, :, patch_slice, :],
                )
                prompt_total_context = torch.matmul(prompt_probs, v_base)
                prompt_other_context = prompt_total_context - prompt_patch_context
                prompt_patch_output = torch.nn.functional.linear(
                    self._merge_heads(prompt_patch_context), self.out.weight, None
                )
                prompt_total_output = torch.nn.functional.linear(
                    self._merge_heads(prompt_total_context), self.out.weight, None
                )
                prompt_other_output = torch.nn.functional.linear(
                    self._merge_heads(prompt_other_context), self.out.weight, None
                )
                prompt_patch_norm = prompt_patch_output.norm(dim=-1)
                prompt_other_norm = prompt_other_output.norm(dim=-1)
                prompt_patch_value_sq_norm = (
                    v_base[:, :, patch_slice, :].float().square().sum(dim=-1)
                )
                prompt_patch_av_magnitude = (
                    prompt_probs[:, :, :, patch_slice].float().square()
                    * prompt_patch_value_sq_norm.unsqueeze(2)
                ).sum(dim=1).clamp_min(0.0).sqrt()
                monitors.update({
                    "prompt_patch_value_contribution_norm": prompt_patch_norm,
                    "prompt_patch_value_contribution_share": (
                        prompt_patch_norm
                        / (prompt_patch_norm + prompt_other_norm).clamp_min(1e-12)
                    ),
                    "prompt_patch_value_to_total_cosine": (
                        torch.nn.functional.cosine_similarity(
                            prompt_patch_output,
                            prompt_total_output,
                            dim=-1,
                            eps=1e-12,
                        )
                    ),
                    "_prompt_patch_value_contribution_vector": prompt_patch_output,
                    "_prompt_patch_pre_output_av_magnitude": (
                        prompt_patch_av_magnitude
                    ),
                    "_prompt_patch_pre_output_content_by_head": (
                        prompt_patch_context
                    ),
                })
        if semantic_length > 0:
            monitors["AcKs_attn"] = attention_probs[:, :, :1, semantic_slice]
            monitors["AsKv_attn"] = attention_probs[:, :, semantic_slice, patch_slice]
            monitors["AvKs_attn"] = attention_probs[:, :, patch_slice, semantic_slice]
            if prompt_length > 0:
                monitors["AsKp_attn"] = attention_probs[:, :, semantic_slice, prompt_slice]
                monitors["ApKs_attn"] = attention_probs[:, :, prompt_slice, semantic_slice]
        if q_prompt.numel() > 0 and q_patch.numel() > 0:
            qpqv_raw = torch.matmul(q_prompt, q_patch.transpose(-1, -2)) * scale
            monitors["QpQv_raw"] = qpqv_raw
            if include_visual_normalizations:
                monitors["QpQv_vis"] = self._minmax_normalize_lastdim(qpqv_raw)

        if k_prompt.numel() > 0 and k_patch.numel() > 0:
            kpkv_raw = torch.matmul(k_prompt, k_patch.transpose(-1, -2)) * scale
            monitors["KpKv_raw"] = kpkv_raw
            if include_visual_normalizations:
                monitors["KpKv_vis"] = self._minmax_normalize_lastdim(kpkv_raw)

        if q_prompt.numel() > 0 and k_patch.numel() > 0:
            qpkv_raw = torch.matmul(q_prompt, k_patch.transpose(-1, -2)) * scale
            monitors["QpKv_raw"] = qpkv_raw
            if include_visual_normalizations:
                monitors["QpKv_vis"] = self._minmax_normalize_lastdim(qpkv_raw)

        if q_patch.numel() > 0 and k_prompt.numel() > 0:
            qvkp_raw = torch.matmul(q_patch, k_prompt.transpose(-1, -2)) * scale
            monitors["QvKp_raw"] = qvkp_raw
            if include_visual_normalizations:
                monitors["QvKp_vis"] = self._minmax_normalize_lastdim(qvkp_raw)

        if q_cls.numel() > 0 and k_patch.numel() > 0:
            qckv_raw = torch.matmul(q_cls, k_patch.transpose(-1, -2)) * scale
            monitors["QcKv_raw"] = qckv_raw
            if include_visual_normalizations:
                monitors["QcKv_vis"] = self._minmax_normalize_lastdim(qckv_raw)

        if q_patch.numel() > 0 and k_cls.numel() > 0:
            qvkc_raw = torch.matmul(q_patch, k_cls.transpose(-1, -2)) * scale
            monitors["QvKc_raw"] = qvkc_raw
            if include_visual_normalizations:
                monitors["QvKc_vis"] = self._minmax_normalize_lastdim(qvkc_raw)

        if q_cls.numel() > 0 and k_prompt.numel() > 0:
            qckp_raw = torch.matmul(q_cls, k_prompt.transpose(-1, -2)) * scale
            monitors["QcKp_raw"] = qckp_raw
            if include_visual_normalizations:
                monitors["QcKp_vis"] = self._minmax_normalize_lastdim(qckp_raw)

        if q_prompt.numel() > 0 and k_cls.numel() > 0:
            qpkc_raw = torch.matmul(q_prompt, k_cls.transpose(-1, -2)) * scale
            monitors["QpKc_raw"] = qpkc_raw
            if include_visual_normalizations:
                monitors["QpKc_vis"] = self._minmax_normalize_lastdim(qpkc_raw)

        if q_semantic.numel() > 0 and k_patch.numel() > 0:
            qskv_raw = torch.matmul(q_semantic, k_patch.transpose(-1, -2)) * scale
            monitors["QsKv_raw"] = qskv_raw
            if include_visual_normalizations:
                monitors["QsKv_vis"] = self._minmax_normalize_lastdim(qskv_raw)

        if q_patch.numel() > 0 and k_semantic.numel() > 0:
            qvks_raw = torch.matmul(q_patch, k_semantic.transpose(-1, -2)) * scale
            monitors["QvKs_raw"] = qvks_raw
            if include_visual_normalizations:
                monitors["QvKs_vis"] = self._minmax_normalize_lastdim(qvks_raw)

        if q_cls.numel() > 0 and k_semantic.numel() > 0:
            qcks_raw = torch.matmul(q_cls, k_semantic.transpose(-1, -2)) * scale
            monitors["QcKs_raw"] = qcks_raw
            if include_visual_normalizations:
                monitors["QcKs_vis"] = self._minmax_normalize_lastdim(qcks_raw)

        if q_semantic.numel() > 0 and k_cls.numel() > 0:
            qskc_raw = torch.matmul(q_semantic, k_cls.transpose(-1, -2)) * scale
            monitors["QsKc_raw"] = qskc_raw
            if include_visual_normalizations:
                monitors["QsKc_vis"] = self._minmax_normalize_lastdim(qskc_raw)

        if q_semantic.numel() > 0 and k_prompt.numel() > 0:
            qskp_raw = torch.matmul(q_semantic, k_prompt.transpose(-1, -2)) * scale
            monitors["QsKp_raw"] = qskp_raw
            if include_visual_normalizations:
                monitors["QsKp_vis"] = self._minmax_normalize_lastdim(qskp_raw)

        if q_prompt.numel() > 0 and k_semantic.numel() > 0:
            qpks_raw = torch.matmul(q_prompt, k_semantic.transpose(-1, -2)) * scale
            monitors["QpKs_raw"] = qpks_raw
            if include_visual_normalizations:
                monitors["QpKs_vis"] = self._minmax_normalize_lastdim(qpks_raw)

        if isinstance(self._last_prompt_path_intervention_stats, dict):
            monitors.update(self._last_prompt_path_intervention_stats)

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
        self._last_attention_mediation_stats = None

    def _cache_attention_mediation_stats(self, mediation, config, layer_idx):
        if mediation is None:
            self._last_attention_mediation_stats = None
            return
        stats = dict(mediation.get("monitor_stats", {}))
        if config:
            ref = mediation["delta_attention_output"]
            prompt_gamma = self._attention_mediation_gamma(config, "prompt_gamma", layer_idx, ref).detach()
            semantic_gamma = self._attention_mediation_gamma(config, "semantic_gamma", layer_idx, ref).detach()
            stats["prompt_gamma"] = float(prompt_gamma.item())
            stats["semantic_gamma"] = float(semantic_gamma.item())
            delta = ref.detach().float()
            prompt_slice = mediation["prompt_slice"]
            semantic_slice = mediation["semantic_slice"]
            prompt_delta = delta[:, prompt_slice, :]
            semantic_delta = delta[:, semantic_slice, :]
            if prompt_delta.numel() > 0:
                stats["prompt_writeback_l2"] = float((prompt_gamma.float() * prompt_delta).pow(2).mean().sqrt().item())
            if semantic_delta.numel() > 0:
                stats["semantic_writeback_l2"] = float((semantic_gamma.float() * semantic_delta).pow(2).mean().sqrt().item())
        stats["layer"] = int(layer_idx)
        self._last_attention_mediation_stats = stats

    @staticmethod
    def _attention_mediation_gamma(config: Dict[str, Any], name: str, layer_idx: int, ref: torch.Tensor) -> torch.Tensor:
        """
        取当前层的 gamma gate，并整理成可广播到 [B,N,D] 的形状。

        attention mediation 是“当前层内”的修正，因此 gamma 长度等于 ViT block 数；
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

    @staticmethod
    def _attach_target_relevance_attention(
        affinity,
        attention_probs,
        affinity_config,
    ):
        """Expose the graph-connected post-softmax Attention only to relevance probes."""
        if bool(affinity_config.get("retain_attention_for_relevance", False)):
            affinity["_target_relevance_attention"] = attention_probs

    @staticmethod
    def _attach_prompt_layer_monitors(
        affinity,
        block_input,
        block_output,
        num_prompt_tokens,
        *,
        detach,
        affinity_config=None,
        layer_idx=0,
    ):
        if int(num_prompt_tokens) <= 0 or not isinstance(affinity, dict):
            return
        prompt_slice = slice(1, 1 + int(num_prompt_tokens))
        prompt_input = block_input[:, prompt_slice, :].mean(dim=1)
        prompt_output = block_output[:, prompt_slice, :].mean(dim=1)
        if detach:
            prompt_input = prompt_input.detach()
            prompt_output = prompt_output.detach()
        affinity.update({
            "prompt_layer_input_norm": prompt_input.norm(dim=-1),
            "prompt_layer_output_norm": prompt_output.norm(dim=-1),
            "prompt_layer_change_norm": (prompt_output - prompt_input).norm(dim=-1),
            "prompt_layer_input_output_cosine": torch.nn.functional.cosine_similarity(
                prompt_input, prompt_output, dim=-1, eps=1e-12
            ),
            "_prompt_layer_input_vector": prompt_input,
            "_prompt_layer_output_vector": prompt_output,
        })
        prompt_contribution = affinity.pop("_cls_prompt_value_contribution_vector", None)
        if torch.is_tensor(prompt_contribution):
            cls_delta = block_output[:, 0, :] - block_input[:, 0, :]
            if detach:
                cls_delta = cls_delta.detach()
            affinity["cls_prompt_value_to_cls_delta_cosine"] = (
                torch.nn.functional.cosine_similarity(
                    prompt_contribution, cls_delta, dim=-1, eps=1e-12
                )
            )
        prompt_patch_contribution = affinity.pop(
            "_prompt_patch_value_contribution_vector", None
        )
        if torch.is_tensor(prompt_patch_contribution):
            prompt_delta = (
                block_output[:, prompt_slice, :] - block_input[:, prompt_slice, :]
            )
            if detach:
                prompt_delta = prompt_delta.detach()
            affinity["prompt_patch_value_to_prompt_delta_cosine"] = (
                torch.nn.functional.cosine_similarity(
                    prompt_patch_contribution,
                    prompt_delta,
                    dim=-1,
                    eps=1e-12,
                )
            )
            affinity["_prompt_patch_content_vector"] = (
                prompt_patch_contribution.detach()
                if detach
                else prompt_patch_contribution
            )

        cls_patch_attention = affinity.get("AcKv_attn")
        if not torch.is_tensor(cls_patch_attention) or cls_patch_attention.numel() == 0:
            return
        patch_count = int(cls_patch_attention.shape[-1])
        patch_start = 1 + int(num_prompt_tokens)
        patch_end = patch_start + patch_count
        if patch_end > int(block_output.shape[1]):
            return
        prompt_tokens = block_output[:, prompt_slice, :]
        patch_tokens = block_output[:, patch_start:patch_end, :]
        if detach:
            prompt_tokens = prompt_tokens.detach()
            patch_tokens = patch_tokens.detach()
            cls_patch_attention = cls_patch_attention.detach()
        prompt_tokens = torch.nn.functional.normalize(prompt_tokens.float(), dim=-1, eps=1e-12)
        patch_tokens = torch.nn.functional.normalize(patch_tokens.float(), dim=-1, eps=1e-12)

        concept_config = affinity_config if isinstance(affinity_config, dict) else {}
        selected_layers = {
            int(item)
            for item in concept_config.get("attribute_concept_selected_layers", [])
        }
        attribute_directions = concept_config.get("attribute_concept_directions")
        true_weights = concept_config.get("attribute_concept_true_weights")
        margin_weights = concept_config.get("attribute_concept_margin_weights")
        concept_enabled = bool(
            concept_config.get("attribute_concept_enable", False)
            and (not selected_layers or int(layer_idx) in selected_layers)
            and torch.is_tensor(attribute_directions)
            and torch.is_tensor(true_weights)
            and torch.is_tensor(margin_weights)
        )
        if concept_enabled:
            directions = attribute_directions.detach().to(
                device=prompt_tokens.device, dtype=torch.float32
            )
            true_attribute_weights = true_weights.detach().to(
                device=prompt_tokens.device, dtype=torch.float32
            )
            margin_attribute_weights = margin_weights.detach().to(
                device=prompt_tokens.device, dtype=torch.float32
            )
            if (
                directions.dim() != 2
                or true_attribute_weights.dim() != 2
                or margin_attribute_weights.shape != true_attribute_weights.shape
                or directions.shape[0] != true_attribute_weights.shape[1]
                or directions.shape[1] != prompt_tokens.shape[-1]
                or true_attribute_weights.shape[0] != prompt_tokens.shape[0]
            ):
                raise ValueError(
                    "Attribute-concept monitor tensors have incompatible shapes"
                )
            direction_unit = torch.nn.functional.normalize(
                directions, dim=-1, eps=1e-12
            )

            def normalize_profile(values):
                return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            def effective_count(values):
                entropy = -(
                    values * values.clamp_min(1e-12).log()
                ).sum(dim=-1)
                observed = values.sum(dim=-1) > 1e-12
                return torch.where(
                    observed,
                    entropy.exp(),
                    torch.zeros_like(entropy),
                )

            def profile_overlap(values):
                if values.shape[1] < 2:
                    return values.new_zeros(values.shape[0])
                unit = torch.nn.functional.normalize(values, dim=-1, eps=1e-12)
                similarity = torch.matmul(unit, unit.transpose(1, 2))
                mask = ~torch.eye(
                    values.shape[1], dtype=torch.bool, device=values.device
                )
                return similarity[:, mask].reshape(values.shape[0], -1).mean(dim=-1)

            local_attribute_indices = [
                int(item)
                for item in concept_config.get("local_attribute_indices", [])
            ]
            global_attribute_indices = [
                int(item)
                for item in concept_config.get("global_attribute_indices", [])
            ]

            def semantic_granularity(values, prefix):
                if not local_attribute_indices or not global_attribute_indices:
                    return {}
                local_index = torch.as_tensor(
                    local_attribute_indices, device=values.device, dtype=torch.long
                )
                global_index = torch.as_tensor(
                    global_attribute_indices, device=values.device, dtype=torch.long
                )
                local_share = values.index_select(-1, local_index).sum(dim=-1)
                global_share = values.index_select(-1, global_index).sum(dim=-1)
                granularity = (global_share - local_share) / (
                    global_share + local_share
                ).clamp_min(1e-12)
                return {
                    f"{prefix}_local_attribute_share": local_share,
                    f"{prefix}_global_attribute_share": global_share,
                    f"{prefix}_semantic_granularity_index": granularity,
                }

            prompt_attribute_similarity = torch.matmul(
                prompt_tokens, direction_unit.transpose(0, 1)
            )
            prompt_true_signal = (
                torch.relu(prompt_attribute_similarity)
                * true_attribute_weights.abs().unsqueeze(1)
            )
            prompt_margin_signal = torch.relu(
                prompt_attribute_similarity
                * margin_attribute_weights.unsqueeze(1)
            )
            prompt_true_profile = normalize_profile(prompt_true_signal)
            prompt_margin_profile = normalize_profile(prompt_margin_signal)
            affinity.update({
                "_prompt_true_attribute_profile": prompt_true_profile,
                "_prompt_margin_attribute_profile": prompt_margin_profile,
                "prompt_true_attribute_effective_count": effective_count(
                    prompt_true_profile
                ),
                "prompt_true_attribute_top1_share": prompt_true_profile.max(
                    dim=-1
                ).values,
                "prompt_margin_attribute_effective_count": effective_count(
                    prompt_margin_profile
                ),
                "prompt_margin_attribute_top1_share": prompt_margin_profile.max(
                    dim=-1
                ).values,
                "prompt_true_attribute_role_overlap": profile_overlap(
                    prompt_true_profile
                ),
                "prompt_margin_attribute_role_overlap": profile_overlap(
                    prompt_margin_profile
                ),
            })
            affinity.update(
                semantic_granularity(prompt_true_profile, "prompt_true")
            )
            affinity.update(
                semantic_granularity(prompt_margin_profile, "prompt_margin")
            )

            if torch.is_tensor(prompt_patch_contribution):
                collection_tokens = torch.nn.functional.normalize(
                    prompt_patch_contribution.detach().float(),
                    dim=-1,
                    eps=1e-12,
                )
                collection_similarity = torch.matmul(
                    collection_tokens, direction_unit.transpose(0, 1)
                )
                collection_true_profile = normalize_profile(
                    torch.relu(collection_similarity)
                    * true_attribute_weights.abs().unsqueeze(1)
                )
                collection_margin_profile = normalize_profile(
                    torch.relu(
                        collection_similarity
                        * margin_attribute_weights.unsqueeze(1)
                    )
                )
                affinity.update({
                    "_collection_true_attribute_profile": collection_true_profile,
                    "_collection_margin_attribute_profile": collection_margin_profile,
                    "collection_true_attribute_effective_count": effective_count(
                        collection_true_profile
                    ),
                    "collection_margin_attribute_effective_count": effective_count(
                        collection_margin_profile
                    ),
                    "collection_prompt_true_attribute_profile_cosine": (
                        torch.nn.functional.cosine_similarity(
                            collection_true_profile,
                            prompt_true_profile,
                            dim=-1,
                            eps=1e-12,
                        )
                    ),
                    "collection_prompt_margin_attribute_profile_cosine": (
                        torch.nn.functional.cosine_similarity(
                            collection_margin_profile,
                            prompt_margin_profile,
                            dim=-1,
                            eps=1e-12,
                        )
                    ),
                })
                affinity.update(
                    semantic_granularity(collection_true_profile, "collection_true")
                )
                affinity.update(
                    semantic_granularity(
                        collection_margin_profile, "collection_margin"
                    )
                )

            patch_attribute_similarity = torch.matmul(
                patch_tokens, direction_unit.transpose(0, 1)
            )
            patch_weighted = (
                patch_attribute_similarity
                * margin_attribute_weights.unsqueeze(1)
            )
            patch_support = torch.relu(patch_weighted).sum(dim=-1)
            patch_fallback = patch_support.sum(dim=-1) <= 1e-12
            patch_support = torch.where(
                patch_fallback.unsqueeze(-1),
                patch_weighted.abs().sum(dim=-1),
                patch_support,
            )
            patch_profile = normalize_profile(patch_support)
            patch_effective = effective_count(patch_profile)
            affinity.update({
                "_attribute_concept_patch_support": patch_support,
                "attribute_concept_patch_effective_count": patch_effective,
                "attribute_concept_patch_effective_ratio": (
                    patch_effective / float(patch_count)
                ),
                "attribute_concept_patch_top1_share": patch_profile.max(
                    dim=-1
                ).values,
                "attribute_concept_patch_fallback_ratio": patch_fallback.float(),
            })
            prompt_patch_attention = affinity.get("ApKv_attn")
            if torch.is_tensor(prompt_patch_attention) and prompt_patch_attention.numel():
                attention_profile = prompt_patch_attention.detach().float().mean(dim=1)
                attention_profile = normalize_profile(attention_profile)
                patch_ratio = float(
                    concept_config.get("attribute_concept_patch_ratio", 0.2)
                )
                if not 0.0 < patch_ratio < 1.0:
                    raise ValueError(
                        "attribute_concept_patch_ratio must be between 0 and 1"
                    )
                selected_count = min(
                    patch_count - 1,
                    max(1, int(math.ceil(patch_ratio * patch_count))),
                )
                concept_indices = patch_support.topk(
                    selected_count, dim=-1, largest=True
                ).indices
                concept_mass = attention_profile.gather(
                    2,
                    concept_indices.unsqueeze(1).expand(
                        -1, attention_profile.shape[1], -1
                    ),
                ).sum(dim=-1)
                attention_indices = attention_profile.topk(
                    selected_count, dim=-1, largest=True
                ).indices
                concept_mask = torch.zeros_like(patch_support, dtype=torch.bool)
                concept_mask.scatter_(1, concept_indices, True)
                overlap = concept_mask.unsqueeze(1).expand(
                    -1, attention_profile.shape[1], -1
                ).gather(2, attention_indices).float().mean(dim=-1)
                affinity.update({
                    "prompt_attention_to_attribute_concept_patch_mass": concept_mass,
                    "prompt_attention_to_attribute_concept_patch_lift": (
                        concept_mass / float(selected_count / patch_count)
                    ),
                    "prompt_attention_attribute_concept_topk_overlap": overlap,
                })

        prompt_patch_similarity = torch.matmul(
            prompt_tokens, patch_tokens.transpose(-1, -2)
        ).mean(dim=1)
        cls_patch_mass = cls_patch_attention.float().mean(dim=1).squeeze(1)
        centered_mass = cls_patch_mass - cls_patch_mass.mean(dim=-1, keepdim=True)
        centered_similarity = (
            prompt_patch_similarity
            - prompt_patch_similarity.mean(dim=-1, keepdim=True)
        )
        correlation = (centered_mass * centered_similarity).sum(dim=-1) / (
            centered_mass.square().sum(dim=-1).sqrt()
            * centered_similarity.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-12)
        selected_count = max(1, int(math.ceil(0.2 * patch_count)))
        high_indices = cls_patch_mass.topk(selected_count, dim=-1, largest=True).indices
        low_indices = cls_patch_mass.topk(selected_count, dim=-1, largest=False).indices
        high_similarity = prompt_patch_similarity.gather(1, high_indices).mean(dim=-1)
        low_similarity = prompt_patch_similarity.gather(1, low_indices).mean(dim=-1)
        affinity.update({
            "prompt_patch_similarity_mean": prompt_patch_similarity.mean(dim=-1),
            "prompt_patch_similarity_within_sample_std": prompt_patch_similarity.std(
                dim=-1, unbiased=False
            ),
            "cls_attention_prompt_patch_similarity_correlation": correlation,
            "cls_high_attention_prompt_patch_similarity": high_similarity,
            "cls_low_attention_prompt_patch_similarity": low_similarity,
            "cls_high_minus_low_prompt_patch_similarity": high_similarity - low_similarity,
        })

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
        if mediation is not None and execution_mode not in {"attention_parallel", "block_parallel"}:
            raise ValueError("ATTENTION_MEDIATION.EXECUTION_MODE must be attention_parallel or block_parallel.")
        if mediation is not None and mlp_policy not in {"enter_mlp", "skip_mlp"}:
            raise ValueError("ATTENTION_MEDIATION.MLP_POLICY must be enter_mlp or skip_mlp.")
        self._cache_attention_mediation_stats(mediation, attention_mediation_config, layer_idx)
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
        selected_layers = affinity_config.get("selected_layers")
        collect_full_diagnostics = (
            selected_layers is None
            or int(layer_idx) in {int(item) for item in selected_layers}
        )

        def build_affinity(block_output):
            if collect_full_diagnostics:
                affinity = self.attn.compute_prompt_visual_monitors(
                    q_proj,
                    k_proj,
                    affinity_config.get("prompt_length", 0),
                    affinity_config.get("semantic_length", 0),
                    value_layer=v_proj,
                    attention_probs=monitor_attention_probs,
                    detach=affinity_config.get("detach", True),
                    include_visual_normalizations=affinity_config.get(
                        "include_visual_normalizations", True
                    ),
                )
                self._attach_target_relevance_attention(
                    affinity,
                    monitor_attention_probs,
                    affinity_config,
                )
                if bool(affinity_config.get("collect_prompt_slot_states", False)):
                    prompt_length = int(
                        affinity_config.get("prompt_length", num_prompt_tokens)
                    )
                    prompt_slice = slice(1, 1 + prompt_length)
                    detach = bool(affinity_config.get("detach", True))

                    def compact(value):
                        value = value[:, prompt_slice, :]
                        return value.detach() if detach else value

                    def compact_projection(value):
                        value = value[:, :, prompt_slice, :]
                        value = self.attn._merge_heads(value)
                        return value.detach() if detach else value

                    affinity.update(
                        {
                            "_prompt_slot_raw_input": compact(block_input),
                            "_prompt_slot_ln_input": compact(x_norm),
                            "_prompt_slot_key": compact_projection(k_proj),
                            "_prompt_slot_value": compact_projection(v_proj),
                        }
                    )
            else:
                # Continuity only needs compact Prompt input/output vectors.
                # Avoid constructing full Q/K and value diagnostics for layers
                # that the fixed Probe will discard.
                affinity = {}
            self._attach_prompt_layer_monitors(
                affinity,
                block_input,
                block_output,
                num_prompt_tokens,
                detach=affinity_config.get("detach", True),
                affinity_config=affinity_config,
                layer_idx=layer_idx,
            )
            return affinity

        # --- 注意力分支 + 残差 ---
        block_input = x
        h = x
        x_norm = self.attention_norm(x)
        # 同时拿到 MHSA 输出 + 多头形式的 q_proj / k_proj
        (
            x,
            weights,
            q_proj,
            k_proj,
            v_proj,
            mediation,
            monitor_attention_probs,
        ) = self.attn.forward_with_projections(
            x_norm,
            affinity_config.get("semantic_length", 0),
            affinity_config.get("block_s_to_cls", False),
            attention_mediation_config,
            num_prompt_tokens,
        )
        execution_mode = attention_mediation_config.get("execution_mode") if attention_mediation_config else None
        mlp_policy = attention_mediation_config.get("mlp_policy") if attention_mediation_config else None
        if mediation is not None and execution_mode not in {"attention_parallel", "block_parallel"}:
            raise ValueError("ATTENTION_MEDIATION.EXECUTION_MODE must be attention_parallel or block_parallel.")
        if mediation is not None and mlp_policy not in {"enter_mlp", "skip_mlp"}:
            raise ValueError("ATTENTION_MEDIATION.MLP_POLICY must be enter_mlp or skip_mlp.")
        self._cache_attention_mediation_stats(mediation, attention_mediation_config, layer_idx)
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
            attn_aff = build_affinity(x)
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
        attn_aff = build_affinity(x)

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
        self._last_attention_mediation_stats = []
        self._prompt_state_intervention = None
        self._last_prompt_state_intervention_stats = []
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)  # 每层都是同结构的 Transformer Block（内部是 LN→MHSA→残差；LN→MLP→残差）
            self.layer.append(copy.deepcopy(layer))
        # 本实现的 Block 属于 Pre-LN（在每个子层前 LN），额外的末端 LN（有些论文称 final LN）有助于稳定训练并改善表征
    def _apply_prompt_state_intervention(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_idx: int,
        num_prompt_tokens: int,
    ):
        intervention = self._prompt_state_intervention
        if not intervention or int(num_prompt_tokens) <= 0:
            return hidden_states, None
        if int(intervention.get("target_layer", -1)) != int(layer_idx):
            return hidden_states, None
        mode = str(intervention.get("mode", ""))
        if mode != "prompt_context_swap":
            raise ValueError(f"Unsupported prompt state intervention: {mode}")
        permutation = intervention.get("permutation")
        if not torch.is_tensor(permutation):
            raise ValueError("Prompt context swap requires a batch permutation")
        permutation = permutation.detach().to(
            device=hidden_states.device, dtype=torch.long
        ).view(-1)
        batch_size = int(hidden_states.shape[0])
        if permutation.numel() != batch_size:
            raise ValueError(
                "Prompt context swap permutation length does not match batch size"
            )
        if not torch.equal(
            permutation.sort().values,
            torch.arange(batch_size, device=permutation.device),
        ):
            raise ValueError("Prompt context swap requires a complete batch permutation")
        fixed_point = permutation == torch.arange(
            batch_size, device=permutation.device
        )
        if batch_size > 1 and bool(fixed_point.any().item()):
            raise ValueError("Prompt context swap permutation must not contain self-pairs")
        prompt_slice = slice(1, 1 + int(num_prompt_tokens))
        prompt_state = hidden_states[:, prompt_slice, :]
        swapped_prompt = prompt_state.index_select(0, permutation)
        changed = hidden_states.clone()
        changed[:, prompt_slice, :] = swapped_prompt
        with torch.no_grad():
            cosine = torch.nn.functional.cosine_similarity(
                prompt_state.detach().float().reshape(batch_size, -1),
                swapped_prompt.detach().float().reshape(batch_size, -1),
                dim=-1,
            )
            stats = {
                "prompt_context_swap_applied": cosine.new_full(
                    (batch_size,), 1.0 if batch_size > 1 else 0.0
                ),
                "prompt_context_swap_fixed_point_ratio": fixed_point.float(),
                "prompt_context_swap_before_after_cosine": cosine,
                "prompt_context_swap_delta_norm": (
                    swapped_prompt.detach().float() - prompt_state.detach().float()
                ).norm(dim=-1).mean(dim=-1),
            }
        return changed, stats

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
        self._last_attention_mediation_stats = []
        self._last_prompt_state_intervention_stats = []
        for layer_idx, layer_block in enumerate(self.layer):
            hidden_states, prompt_state_stats = self._apply_prompt_state_intervention(
                hidden_states,
                layer_idx=layer_idx,
                num_prompt_tokens=num_prompt_tokens,
            )
            if prompt_state_stats is not None:
                self._last_prompt_state_intervention_stats.append({
                    "layer_index": int(layer_idx),
                    **prompt_state_stats,
                })
            # attention_mediation_config 在所有层共享，layer_idx 用于取当前层独立的 gamma gate。
            hidden_states, weights, semantics = layer_block(hidden_states, semantics,
                    num_prompt_tokens, semantic_length, block_s_to_cls,
                    attention_mediation_config, layer_idx)  # hidden_states为(B, 1+N, D)D 为 hidden_size
            if self.vis:
                attn_weights.append(weights)    # 把每层的 weights 保存到列表里否则返回空列表
            if layer_block._last_attention_mediation_stats is not None:
                self._last_attention_mediation_stats.append(dict(layer_block._last_attention_mediation_stats))
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
        offload_diagnostics = bool(
            affinity_config.get("offload_diagnostics_to_cpu", False)
        )
        if offload_diagnostics and torch.is_grad_enabled():
            raise RuntimeError(
                "Affinity diagnostic CPU offload is only valid in a no-grad forward"
            )
        attn_weights = []
        affinities = []
        previous_prompt_output = None
        self._last_attention_mediation_stats = []
        self._last_prompt_state_intervention_stats = []
        for layer_idx, layer_block in enumerate(self.layer):
            hidden_states, prompt_state_stats = self._apply_prompt_state_intervention(
                hidden_states,
                layer_idx=layer_idx,
                num_prompt_tokens=num_prompt_tokens,
            )
            # forward_with_affinity 同时服务训练损失/可视化，因此 mediation 的插入点必须和常规 forward 一致。
            hidden_states, weights, affinity, semantics = layer_block.forward_with_affinity(
                hidden_states,
                affinity_config,
                semantics,
                num_prompt_tokens,
                attention_mediation_config,
                layer_idx,
            )
            if prompt_state_stats is not None:
                affinity.update(prompt_state_stats)
                prompt_state_record = {
                    "layer_index": int(layer_idx),
                    **prompt_state_stats,
                }
                self._last_prompt_state_intervention_stats.append(
                    _offload_diagnostic_tree(prompt_state_record)
                    if offload_diagnostics
                    else prompt_state_record
                )
            if offload_diagnostics:
                previous_prompt_output = _attach_prompt_continuity_to_layer(
                    affinity, previous_prompt_output
                )
            if self.vis:
                attn_weights.append(
                    _offload_diagnostic_tree(weights)
                    if offload_diagnostics
                    else weights
                )
            affinities.append(
                _offload_diagnostic_tree(affinity)
                if offload_diagnostics
                else affinity
            )
            if layer_block._last_attention_mediation_stats is not None:
                mediation_stats = dict(
                    layer_block._last_attention_mediation_stats
                )
                self._last_attention_mediation_stats.append(
                    _offload_diagnostic_tree(mediation_stats)
                    if offload_diagnostics
                    else mediation_stats
                )
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
        self._last_token_sequence = None

    def forward(self, input_ids, semantics: torch.Tensor = None):
        """标准前向：返回编码后的序列与注意力权重。"""
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output, semantics)
        self._last_token_sequence = encoded
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
        self._last_token_sequence = encoded
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
