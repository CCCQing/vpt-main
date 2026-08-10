#!/usr/bin/env python3
"""
vit_prompt 主干实现。

当前文件同时承载两条 prompt 主线和一条轻量语义支线：

1. Prompt 主线
   - BACKEND="dynamic"：
     输入 prompt 可来自 learned prompt 或 distributor(mean)，
     只注入输入层 prompt，不再支持层间线性演化。
   - BACKEND="vpt_deep"：
     输入 prompt 同样可来自 learned prompt 或 distributor(mean)，
     但后续层 prompt 直接使用各层独立的 deep_prompt_embeddings。

2. 语义侧支线
   - 当前已简化成最薄形式：
       semantics -> hidden 对齐 -> 与 prompt/visual 交互 -> 输出
   - 旧版 anchor/free token、复杂 readout 等语义建模结构已不再使用。
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from typing import Optional, Dict, Any, Tuple, List

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout, LayerNorm
from scipy import ndimage


from ..vit_backbones.vit import (
    CONFIGS,
    Transformer,
    VisionTransformer,
    _attach_prompt_continuity_to_layer,
    _offload_diagnostic_tree,
    np2th,
)
from ...utils import logging
from ...utils.reproducibility import make_torch_generator

logger = logging.get_logger("visual_prompt")


class _UnusedCrossAttention(nn.Module):
    """最小跨注意力单元。

    它只负责一件事：让 query 序列从 source 序列中读取上下文。
    文件里语义、prompt、visual 三者的交互都复用它，因此它不绑定任何业务语义，
    只关心输入输出的张量形状。
    """
class SemanticTokenProjector(nn.Module):
    """轻量语义交互分支。

    当前版本不再把语义拆成多 token 并单独做 readout，而是维护一个单语义状态：

        semantics [B,S]
          -> semantic_input_proj
          -> semantic_input [B,D]
          -> sem_state [B,D]
          -> 与 prompt_tokens / visual_tokens 逐层交互
          -> semantic_output [B,D]

    这里保留的核心能力只有一条语义读取路由：
    - Sem <- Prompt+Visual

    因此它更像一个“语义状态机”，而不是完整的语义编码器。
    """
    def __init__(self, hidden_size: int, semantic_tokens_cfg) -> None:
        super().__init__()
        if semantic_tokens_cfg is None:
            raise ValueError("semantic_tokens_cfg is required for SemanticTokenProjector")
        self.hidden_size = int(hidden_size)

        self.num_tokens = int(semantic_tokens_cfg.NUM_TOKENS)
        if self.num_tokens != 1:
            raise ValueError("The current semantic-token main-sequence design requires NUM_TOKENS=1.")
        self.input_dim = int(semantic_tokens_cfg.INPUT_DIM)
        self.semantic_input_proj = nn.Linear(self.input_dim, self.hidden_size)

        # 语义状态及 joint cross-attention 输入的归一化，主要用于稳定更新过程
        self.semantic_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.semantic_type_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))
        # BEGIN SEMANTIC_ABLATION_EXPERIMENT
        self.learned_semantic_token = nn.Parameter(torch.empty(1, self.num_tokens, hidden_size))
        nn.init.normal_(
            self.learned_semantic_token,
            mean=0.0,
            std=float(semantic_tokens_cfg.LEARNED_INIT_STD),
        )
        # END SEMANTIC_ABLATION_EXPERIMENT

        # 唯一真实语义交互路由：Qs-K(p+v)
        self.debug_shapes = False
        self._shape_debug_init_logged = False

    def _ensure_semantic_input_proj(self, semantic_dim: int, device: torch.device):
        """按需创建语义输入投影层。

        原始语义维度由数据集属性维决定，不一定等于 ViT hidden size。
        这里第一次看到真实语义维度时，再创建 semantic_dim -> hidden_size
        的线性层，避免把语义维度硬编码在配置里。
        """
        if int(semantic_dim) != self.input_dim:
            raise ValueError(f"Semantic input dim mismatch: expected {self.input_dim}, got {int(semantic_dim)}")

    def _gamma(self, layer_idx: int, num_layers: int) -> float:
        """返回当前层的语义更新强度。

        设计意图是：
        - 早层尽量少动语义状态
        - 到允许交互的层区间后，再逐步增大语义更新幅度
        """
        if layer_idx < self.start_layer:
            return 0.0
        end_layer = self.end_layer if self.end_layer >= 0 else (num_layers - 1)
        if layer_idx > end_layer:
            return 0.0
        span = max(1, end_layer - self.start_layer)
        t = float(layer_idx - self.start_layer) / float(span)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t

    def init_state(self, semantics: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化语义状态。

        当前极简版本不再生成多个 semantic tokens，而是只保留一个单语义向量：

            semantics [B,S]
              -> semantic_input_proj
              -> semantic_input [B,D]
              -> delta_norm
              -> sem_state [B,D]

        返回：
        - sem_state：后续逐层被更新的语义状态
        - semantic_input：初始语义输入，便于后续监控或做残差比较
        """

        # BEGIN SEMANTIC_ABLATION_EXPERIMENT
        if semantics.dim() == 2 and semantics.shape[-1] == 0:
            batch_size = int(semantics.shape[0])
            semantic_token = self.learned_semantic_token.to(device).expand(batch_size, -1, -1)
            semantic_projected = semantic_token[:, 0, :]
            state = {
                "semantic_projected": semantic_projected,
                "semantic_normalized": semantic_projected,
                "semantic_token": semantic_token,
            }
            return semantic_token, state
        # END SEMANTIC_ABLATION_EXPERIMENT

        if semantics.dim() == 3 and semantics.shape[1] == 1:
            semantics = semantics[:, 0, :]
        if semantics.dim() != 2:
            raise ValueError(f"SemanticTokenProjector expects [B, S] or [B,1,S], got {tuple(semantics.shape)}")

        self._ensure_semantic_input_proj(semantics.shape[-1], device=device)
        semantic_projected = self.semantic_input_proj(semantics.to(device))
        semantic_normalized = self.semantic_norm(semantic_projected)
        semantic_token = semantic_normalized.unsqueeze(1) + self.semantic_type_embed

        if self.debug_shapes and (not self._shape_debug_init_logged):
            print(
                "[SHAPE-DEBUG] SemanticTokenProjector.init_state semantics={} projected={} token={}".format(
                    tuple(semantics.shape),
                    tuple(semantic_projected.shape),
                    tuple(semantic_token.shape),
                )
            )
            self._shape_debug_init_logged = True
        state = {
            "semantic_projected": semantic_projected,
            "semantic_normalized": semantic_normalized,
            "semantic_token": semantic_token,
        }
        return semantic_token, state

    def _disabled_step(self, sem_state: torch.Tensor, prompt_tokens: torch.Tensor, visual_tokens: torch.Tensor,
        layer_idx: int, num_layers: int,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        """执行单层语义交互更新：sem_state 作为 query，从 prompt+visual token 中读取上下文。"""
        raise RuntimeError("Semantic tokens are updated only by ViT self-attention in the main token sequence.")

    def export_token_state(self, sem_state: torch.Tensor) -> torch.Tensor:
        """最简输出接口。

        旧版会再过 readout head，把语义 token 汇聚成最终语义向量。
        当前语义状态已经是单个 [B,D] 向量，所以直接返回即可。
        """
        return sem_state

    def finalize_state(self, encoded: torch.Tensor, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # linear 旧路径只缓存输入/输出语义 token，便于和 orthogonal 路径统一读取。
        final_tokens = encoded[:, -self.num_tokens:, :]
        input_tokens = state["semantic_token"]
        state["input_tokens"] = input_tokens
        state["final_tokens"] = final_tokens
        state["semantic_input"] = input_tokens[:, 0, :]
        state["semantic_output"] = final_tokens[:, 0, :]
        state["semantic_delta"] = final_tokens[:, 0, :] - input_tokens[:, 0, :]
        return state


class OrthogonalSemanticTokenizer(nn.Module):
    """CUB 8 组正交语义 tokenizer。

    每组属性生成一个 semantic token；数值属性走 W_i 行空间，文本 residual 和 slot 身份只走 W_i 正交补。
    """

    MANUAL_CUB8_GROUPS = (
        ("wing_color", "primary_color", "wing_shape", "wing_pattern"),
        ("upper_tail_color", "under_tail_color", "tail_shape", "tail_pattern"),
        ("bill_color", "eye_color", "bill_shape", "bill_length"),
        ("crown_color", "forehead_color", "head_pattern"),
        ("throat_color", "breast_color", "size", "breast_pattern"),
        ("belly_color", "underparts_color", "belly_pattern"),
        ("nape_color", "back_color", "back_pattern"),
        ("upperparts_color", "leg_color", "shape"),
    )

    def __init__(self, hidden_size: int, semantic_tokens_cfg) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(semantic_tokens_cfg.NUM_TOKENS)
        self.input_dim = int(semantic_tokens_cfg.INPUT_DIM)
        self.ortho_cfg = semantic_tokens_cfg.ORTHO
        self.group_mode = str(self.ortho_cfg.GROUP_MODE)
        self.text_mode = str(self.ortho_cfg.TEXT_MODE)
        self.debug_shapes = False
        self._shape_debug_init_logged = False

        if self.input_dim != 312:
            raise ValueError("Orthogonal semantic tokenizer requires MODEL.SEMANTIC_TOKENS.INPUT_DIM=312.")
        if self.group_mode not in {"manual_cub8", "equal", "prefix"}:
            raise ValueError("ORTHO.GROUP_MODE must be one of manual_cub8 / equal / prefix.")
        if self.num_tokens <= 0:
            raise ValueError("MODEL.SEMANTIC_TOKENS.NUM_TOKENS must be positive.")
        if self.num_tokens > self.input_dim:
            raise ValueError("MODEL.SEMANTIC_TOKENS.NUM_TOKENS must not exceed MODEL.SEMANTIC_TOKENS.INPUT_DIM.")
        if self.group_mode == "manual_cub8" and self.num_tokens != len(self.MANUAL_CUB8_GROUPS):
            raise ValueError("ORTHO.GROUP_MODE='manual_cub8' requires MODEL.SEMANTIC_TOKENS.NUM_TOKENS=8.")
        if self.text_mode not in {"none", "null_residual", "text_init_codebook"}:
            raise ValueError("ORTHO.TEXT_MODE must be one of none / null_residual / text_init_codebook.")
        if bool(self.ortho_cfg.CODEBOOK_TRAINABLE):
            raise ValueError("Orthogonal semantic tokenizer first version requires ORTHO.CODEBOOK_TRAINABLE=False.")

        raw_names, prefixes = self._load_attributes(str(self.ortho_cfg.ATTRIBUTES_PATH))
        groups = self._build_groups(prefixes)
        self.group_lengths = [int(idx.numel()) for idx in groups]
        self.max_group_len = max(self.group_lengths)
        self.register_buffer("segment_lengths", torch.as_tensor(self.group_lengths, dtype=torch.long))

        text_codebook = None
        if self.text_mode == "text_init_codebook":
            # text_init_codebook: 文本只用于初始化全局正交码表 W，不作为 forward residual 输入。
            text_embeddings = self._load_text_embeddings(str(self.ortho_cfg.TEXT_EMBED_PATH), raw_names)
            text_codebook = self._make_text_init_codebook(text_embeddings)

        group_indices = torch.full((self.num_tokens, self.max_group_len), -1, dtype=torch.long)
        codebooks = []
        for group_id, indices in enumerate(groups):
            group_indices[group_id, :int(indices.numel())] = indices
            if self.text_mode == "text_init_codebook":
                codebook = text_codebook.index_select(0, indices)
            else:
                codebook = self._make_row_orthogonal_codebook(
                    rows=int(indices.numel()),
                    cols=self.hidden_size,
                    seed=int(self.ortho_cfg.CODEBOOK_SEED) + group_id,
                )
            padded_codebook = codebook.new_zeros((self.max_group_len, self.hidden_size))
            padded_codebook[:int(indices.numel()), :] = codebook
            codebooks.append(padded_codebook)
        self.register_buffer("group_indices", group_indices)
        self.register_buffer("codebooks", torch.stack(codebooks, dim=0))

        if self.text_mode == "null_residual":
            # null_residual 使用每个 semantic slot 一个标量 gate 控制文本补空间残差强度。
            # TEXT_GATE_INIT 通常设为 0，使初始 token 严格退化为纯数值主路径 a_i @ W_i。
            self.text_gate = nn.Parameter(torch.full((self.num_tokens,), float(self.ortho_cfg.TEXT_GATE_INIT)))
            # 文本 embedding 是离线生成的属性名向量，顺序必须和 attributes.txt 的 312 个属性完全一致。
            # 训练时不调用文本 encoder，只加载缓存矩阵，避免把文本模型引入训练图。
            text_embeddings = self._load_text_embeddings(str(self.ortho_cfg.TEXT_EMBED_PATH), raw_names)
            self.register_buffer("text_embeddings", text_embeddings)
            # 预先为每个语义组缓存每个属性各自的 null-space 文本向量。
            self._register_text_null_buffers(groups)
        else:
            # none / text_init_codebook 都不在 forward 中动态注入文本 residual。
            self.register_buffer("text_gate", torch.zeros(self.num_tokens))
            self.register_buffer("text_nulls", torch.zeros(self.num_tokens, self.hidden_size))

        logger.info(
            "[semantic-tokenizer] tokenizer=orthogonal group_mode=%s text_mode=%s group_lengths=%s codebook_trainable=%s",
            self.group_mode,
            self.text_mode,
            self.group_lengths,
            False,
        )

    @staticmethod
    def _attribute_prefix(raw_name: str) -> str:
        if raw_name.startswith("has_"):
            raw_name = raw_name[len("has_"):]
        if "::" not in raw_name:
            raise ValueError(f"Attribute name does not contain '::': {raw_name}")
        return raw_name.split("::", 1)[0]

    def _load_attributes(self, path: str) -> Tuple[List[str], List[str]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ORTHO.ATTRIBUTES_PATH not found: {path}")
        raw_names: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    raise ValueError(f"Empty attribute line at {path}:{line_no}")
                parts = stripped.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Expected '<index> <attribute_name>' at {path}:{line_no}, got: {stripped}")
                expected_index = len(raw_names) + 1
                if int(parts[0]) != expected_index:
                    raise ValueError(
                        f"Attribute index mismatch at {path}:{line_no}: expected {expected_index}, got {parts[0]}"
                    )
                raw_names.append(parts[1])
        if len(raw_names) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} attributes, got {len(raw_names)} from {path}")
        return raw_names, [self._attribute_prefix(name) for name in raw_names]

    def _build_groups(self, prefixes: List[str]) -> List[torch.Tensor]:
        if self.group_mode == "manual_cub8":
            return self._build_manual_groups(prefixes)
        if self.group_mode == "equal":
            return self._build_equal_groups()
        if self.group_mode == "prefix":
            return self._build_prefix_groups(prefixes)
        raise ValueError(f"Unsupported ORTHO.GROUP_MODE='{self.group_mode}'")

    def _build_manual_groups(self, prefixes: List[str]) -> List[torch.Tensor]:
        groups: List[torch.Tensor] = []
        used = set()
        for group_prefixes in self.MANUAL_CUB8_GROUPS:
            group_ids = [idx for idx, prefix in enumerate(prefixes) if prefix in group_prefixes]
            if not group_ids:
                raise ValueError(f"Manual CUB8 group has no attributes: {group_prefixes}")
            overlap = used.intersection(group_ids)
            if overlap:
                raise ValueError(f"Manual CUB8 groups overlap at indices: {sorted(overlap)}")
            used.update(group_ids)
            groups.append(torch.as_tensor(group_ids, dtype=torch.long))
        expected = set(range(self.input_dim))
        if used != expected:
            missing = sorted(expected - used)
            extra = sorted(used - expected)
            raise ValueError(f"Manual CUB8 groups must cover all attributes exactly once; missing={missing}, extra={extra}")
        return groups

    def _build_equal_groups(self) -> List[torch.Tensor]:
        groups: List[torch.Tensor] = []
        base = self.input_dim // self.num_tokens
        remainder = self.input_dim % self.num_tokens
        start = 0
        for group_id in range(self.num_tokens):
            length = base + (1 if group_id < remainder else 0)
            end = start + length
            groups.append(torch.arange(start, end, dtype=torch.long))
            start = end
        if start != self.input_dim:
            raise ValueError(f"Equal groups must cover {self.input_dim} attributes, got {start}.")
        return groups

    def _build_prefix_groups(self, prefixes: List[str]) -> List[torch.Tensor]:
        prefix_groups: List[List[int]] = []
        prefix_to_group: Dict[str, List[int]] = {}
        for idx, prefix in enumerate(prefixes):
            if prefix not in prefix_to_group:
                prefix_to_group[prefix] = []
                prefix_groups.append(prefix_to_group[prefix])
            prefix_to_group[prefix].append(idx)

        units = [list(group) for group in prefix_groups]
        while len(units) < self.num_tokens:
            # 当 k 大于前缀数时，只拆分最大的前缀组；这样尽量保留 prefix 语义，又能服从 NUM_TOKENS。
            split_idx = max(range(len(units)), key=lambda idx: len(units[idx]))
            unit = units[split_idx]
            if len(unit) <= 1:
                raise ValueError("Prefix grouping cannot create more non-empty groups from singleton attributes.")
            mid = len(unit) // 2
            units[split_idx:split_idx + 1] = [unit[:mid], unit[mid:]]

        # prefix 模式保留属性顺序，同时用动态规划把 unit 切成尽量均衡的 k 组。
        prefix_sizes = [len(group) for group in units]
        prefix_sums = [0]
        for size in prefix_sizes:
            prefix_sums.append(prefix_sums[-1] + size)

        target = float(self.input_dim) / float(self.num_tokens)
        num_prefixes = len(units)
        dp = [[math.inf for _ in range(num_prefixes + 1)] for _ in range(self.num_tokens + 1)]
        prev = [[-1 for _ in range(num_prefixes + 1)] for _ in range(self.num_tokens + 1)]
        dp[0][0] = 0.0

        for group_id in range(1, self.num_tokens + 1):
            min_end = group_id
            max_end = num_prefixes - (self.num_tokens - group_id)
            for end in range(min_end, max_end + 1):
                for start in range(group_id - 1, end):
                    if not math.isfinite(dp[group_id - 1][start]):
                        continue
                    group_size = prefix_sums[end] - prefix_sums[start]
                    cost = dp[group_id - 1][start] + (float(group_size) - target) ** 2
                    if cost < dp[group_id][end]:
                        dp[group_id][end] = cost
                        prev[group_id][end] = start

        if not math.isfinite(dp[self.num_tokens][num_prefixes]):
            raise ValueError("Prefix grouping failed to produce a valid deterministic partition.")

        boundaries: List[Tuple[int, int]] = []
        end = num_prefixes
        for group_id in range(self.num_tokens, 0, -1):
            start = prev[group_id][end]
            if start < 0:
                raise ValueError("Prefix grouping reconstruction failed.")
            boundaries.append((start, end))
            end = start
        boundaries.reverse()

        groups: List[torch.Tensor] = []
        for start, end in boundaries:
            group_ids: List[int] = []
            for prefix_group in units[start:end]:
                group_ids.extend(prefix_group)
            groups.append(torch.as_tensor(group_ids, dtype=torch.long))

        if len(groups) != self.num_tokens:
            raise ValueError(f"Prefix grouping produced {len(groups)} groups; expected {self.num_tokens}.")
        covered = torch.cat(groups).tolist()
        if covered != list(range(self.input_dim)):
            raise ValueError("Prefix groups must preserve and cover the original 312 attribute order exactly.")
        return groups

    @staticmethod
    def _make_row_orthogonal_codebook(rows: int, cols: int, seed: int) -> torch.Tensor:
        if rows > cols:
            raise ValueError(f"Cannot create row-orthogonal codebook with rows={rows} > cols={cols}")
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        raw = torch.randn(cols, rows, generator=generator)
        q, _ = torch.qr(raw)
        return q.t().contiguous()

    def _make_text_init_codebook(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        # 用 312 个属性名文本向量初始化全局 row-orthogonal 语义码表 W。
        # forward 阶段只使用切分后的 W_i 做 a_i @ W_i，不再读取原始文本向量。
        if tuple(text_embeddings.shape) != (self.input_dim, self.hidden_size):
            raise ValueError(
                f"text_init_codebook expects text embeddings shape {(self.input_dim, self.hidden_size)}, "
                f"got {tuple(text_embeddings.shape)}"
            )
        q, _ = torch.qr(text_embeddings.float().t())
        codebook = q.t().contiguous()
        if tuple(codebook.shape) != (self.input_dim, self.hidden_size):
            raise ValueError(
                f"text_init_codebook produced codebook shape {tuple(codebook.shape)}, "
                f"expected {(self.input_dim, self.hidden_size)}"
            )
        return codebook

    def _load_text_embeddings(self, path: str, raw_names: List[str]) -> torch.Tensor:
        # 读取离线文本缓存；这里故意严格检查顺序，避免属性名文本和 312 维数值属性错位。
        # 若顺序错位，null_residual 会把错误属性的文本残差加到当前属性组上，实验解释会失效。
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ORTHO.TEXT_EMBED_PATH not found: {path}")
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Text embedding cache must be a dict, got {type(payload)}")
        if "raw_attribute_names" not in payload or "embeddings" not in payload:
            raise KeyError("Text embedding cache must contain raw_attribute_names and embeddings.")
        if list(payload["raw_attribute_names"]) != list(raw_names):
            raise ValueError("Text embedding cache raw_attribute_names do not match ORTHO.ATTRIBUTES_PATH order.")
        embeddings = payload["embeddings"]
        if not torch.is_tensor(embeddings):
            raise TypeError("Text embedding cache embeddings must be a torch.Tensor.")
        expected_shape = (self.input_dim, self.hidden_size)
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(f"Expected text embeddings shape {expected_shape}, got {tuple(embeddings.shape)}")
        return embeddings.float().contiguous()

    @staticmethod
    def _null_project(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # W_i 行空间承载数值属性；文本投到正交补，避免直接改变属性回投。
        return x - torch.matmul(torch.matmul(x, codebook.t()), codebook)

    def _register_text_null_buffers(self, groups: List[torch.Tensor]) -> None:
        text_nulls = []
        for group_id, indices in enumerate(groups):
            length = int(indices.numel())
            codebook = self.codebooks[group_id, :length, :]
            # text_group: 当前 semantic slot 覆盖的属性名文本向量，形状 [group_attr_count, hidden]。
            text_group = self.text_embeddings.index_select(0, indices)
            # 正交投影是线性算子：
            # mean(null_project(text_group)) == null_project(mean(text_group))。
            # 因此这里先求组内文本均值再投到 W_i 正交补，避免每个 trial 启动时重复做更大的矩阵乘法。
            text_mean = text_group.mean(dim=0, keepdim=True)
            # null_residual 使用该组文本整体的 null-space 残差；不与属性置信度相乘。
            text_null = self._null_project(text_mean, codebook)
            text_nulls.append(text_null.squeeze(0).contiguous())
        self.register_buffer("text_nulls", torch.stack(text_nulls, dim=0))

    def init_state(self, semantics: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if semantics.dim() == 3 and semantics.shape[1] == 1:
            semantics = semantics[:, 0, :]
        if semantics.dim() != 2:
            raise ValueError(f"OrthogonalSemanticTokenizer expects [B,312] or [B,1,312], got {tuple(semantics.shape)}")
        if int(semantics.shape[-1]) != self.input_dim:
            raise ValueError(f"Semantic input dim mismatch: expected {self.input_dim}, got {int(semantics.shape[-1])}")

        semantics = semantics.to(device=device, dtype=torch.float32)
        batch_size = int(semantics.shape[0])
        tokens: List[torch.Tensor] = []
        # a_i @ W_i
        for group_id in range(self.num_tokens):
            length = int(self.segment_lengths[group_id].item())
            indices = self.group_indices[group_id, :length].to(device=device)
            codebook = self.codebooks[group_id, :length, :].to(device=device)
            attrs = semantics.index_select(dim=1, index=indices)
            numeric_token = torch.matmul(attrs, codebook)
            token = numeric_token

            if self.text_mode == "null_residual":
                # null_residual:
                # text_null_i 是该组属性文本在 W_i 正交补中的整体残差。
                # 它不与属性置信度 attrs 相乘，因此文本路径只提供组级文本补充信息。
                # 置信度只作用于主路径 a_i @ W_i。
                text_null_i = self.text_nulls[group_id].to(device=device).view(1, -1)
                # text_gate[group_id] 控制该语义组文本残差的注入强度；
                # 若初始化为 0，训练初始等价于 numeric_token。
                token = token + self.text_gate[group_id] * text_null_i

            tokens.append(token)
        semantic_tokens = torch.stack(tokens, dim=1)

        if self.debug_shapes and (not self._shape_debug_init_logged):
            print(
                "[SHAPE-DEBUG] OrthogonalSemanticTokenizer.init_state semantics={} tokens={} group_lengths={}".format(
                    tuple(semantics.shape),
                    tuple(semantic_tokens.shape),
                    self.group_lengths,
                )
            )
            self._shape_debug_init_logged = True

        state = {
            "tokenizer": "orthogonal",
            "semantic_token": semantic_tokens,
            "input_tokens": semantic_tokens,
            "input_attributes": semantics,
            "segment_lengths": self.segment_lengths.to(device=device),
        }
        return semantic_tokens, state

    def decode_attributes(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        decoded = semantic_tokens.new_zeros((semantic_tokens.shape[0], self.input_dim))
        for group_id in range(self.num_tokens):
            length = int(self.segment_lengths[group_id].item())
            indices = self.group_indices[group_id, :length].to(device=semantic_tokens.device)
            codebook = self.codebooks[group_id, :length, :].to(device=semantic_tokens.device)
            values = torch.matmul(semantic_tokens[:, group_id, :], codebook.t())
            decoded.index_copy_(dim=1, index=indices, source=values)
        return decoded

    def _split_delta(self, token_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        semantic_parts: List[torch.Tensor] = []
        null_parts: List[torch.Tensor] = []
        for group_id in range(self.num_tokens):
            length = int(self.segment_lengths[group_id].item())
            codebook = self.codebooks[group_id, :length, :].to(device=token_delta.device)
            delta_i = token_delta[:, group_id, :]
            semantic_delta = torch.matmul(torch.matmul(delta_i, codebook.t()), codebook)
            semantic_parts.append(semantic_delta)
            null_parts.append(delta_i - semantic_delta)
        return torch.stack(semantic_parts, dim=1), torch.stack(null_parts, dim=1)

    def finalize_state(self, encoded: torch.Tensor, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        final_tokens = encoded[:, -self.num_tokens:, :]
        input_tokens = state["input_tokens"]
        token_delta = final_tokens - input_tokens
        semantic_delta_part, null_delta_part = self._split_delta(token_delta)
        state["final_tokens"] = final_tokens
        state["decoded_attributes"] = self.decode_attributes(final_tokens)
        state["token_delta"] = token_delta
        state["semantic_delta_part"] = semantic_delta_part
        state["null_delta_part"] = null_delta_part
        state["semantic_input"] = input_tokens
        state["semantic_output"] = final_tokens
        state["semantic_delta"] = token_delta
        return state

class PromptedTransformer(Transformer):
    """PromptedTransformer 主入口。

    统一管理：
    1. Prompt 是否启用及其后端（dynamic / vpt_deep）
    2. 输入 prompt 来源（learned / distributor_mean）
    3. 轻量语义交互分支在 backbone 中的逐层接入
    """

    def __init__(
        self,
        prompt_config,
        config,
        img_size,
        vis,
        prompt_init=None,
        prompt_init_provider=None,
        prompt_init_seed=None,
    ):

        self.semantic_tokens_cfg = prompt_config.SEMANTIC_TOKENS
        self.semantic_tokens_enable = bool(self.semantic_tokens_cfg.ENABLE)
        self.semantic_tokenizer = str(self.semantic_tokens_cfg.TOKENIZER).lower()
        if self.semantic_tokenizer not in {"linear", "orthogonal"}:
            raise ValueError(f"Unsupported MODEL.SEMANTIC_TOKENS.TOKENIZER='{self.semantic_tokens_cfg.TOKENIZER}'")
        self.block_s_to_cls = bool(self.semantic_tokens_cfg.BLOCK_S_TO_CLS)
        self.attention_mediation_cfg = prompt_config.ATTENTION_MEDIATION
        self.attention_mediation_enable = bool(self.attention_mediation_cfg.ENABLE)
        self.prompt_enable = bool(prompt_config.ENABLE)
        self.prompt_backend = prompt_config.BACKEND.lower()
        if self.prompt_backend not in {"dynamic", "vpt_deep"}:
            raise ValueError(f"Unsupported MODEL.PROMPT.BACKEND='{prompt_config.BACKEND}'")
        self.prompt_init_source = prompt_config.INIT_SOURCE.lower()
        if self.prompt_init_source not in {"learned", "distributor_mean"}:
            raise ValueError(f"Unsupported MODEL.PROMPT.INIT_SOURCE='{prompt_config.INIT_SOURCE}'")

        super().__init__(config, img_size, vis)

        self.prompt_config = prompt_config
        self.vit_config = config
        # 运行时缓存：最近一次前向得到的语义侧最终状态。
        self._last_semantic_token_state = None

        if self.semantic_tokens_enable:
            if self.semantic_tokenizer == "linear":
                self.semantic_token_projector = SemanticTokenProjector(
                    hidden_size=int(config.hidden_size),
                    semantic_tokens_cfg=self.semantic_tokens_cfg,
                )
            elif self.semantic_tokenizer == "orthogonal":
                if str(self.semantic_tokens_cfg.TRAIN_SOURCE).lower() != "class_mean":
                    raise ValueError("TOKENIZER='orthogonal' requires MODEL.SEMANTIC_TOKENS.TRAIN_SOURCE='class_mean'.")
                if str(self.semantic_tokens_cfg.EVAL_SOURCE).lower() != "class_mean":
                    raise ValueError("TOKENIZER='orthogonal' requires MODEL.SEMANTIC_TOKENS.EVAL_SOURCE='class_mean'.")
                self.semantic_token_projector = OrthogonalSemanticTokenizer(
                    hidden_size=int(config.hidden_size),
                    semantic_tokens_cfg=self.semantic_tokens_cfg,
                )
            else:
                raise ValueError(f"Unsupported MODEL.SEMANTIC_TOKENS.TOKENIZER='{self.semantic_tokens_cfg.TOKENIZER}'")
        else:
            self.semantic_token_projector = None

        # 统一 image / patch 大小表示，避免后续 mixed tuple/int 判断
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        # prompt token 数只由 prompt 主线控制；语义侧已固定为单语义状态
        num_tokens = self.prompt_config.NUM_TOKENS if self.prompt_enable else 0
        self.num_tokens = num_tokens  # number of prompted tokens
        # prompt token 上的轻量 dropout
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        self.prompt_init_provider = prompt_init_provider
        self.prompt_init_seed = None if prompt_init_seed is None else int(prompt_init_seed)
        prompt_init_generator = make_torch_generator(self.prompt_init_seed)
        self.prompt_proj = nn.Identity()
        self.prompt_embeddings = None
        self.deep_prompt_embeddings = None
        # 新增的 ATTENTION_MEDIATION 分支不在这里构造 route。
        # PromptedTransformer 只保存每层的可学习 gamma gate，并把配置传给 ViT block 内部执行。
        self.attention_mediation_prompt_gamma = None
        self.attention_mediation_semantic_gamma = None

        # 运行期 debug 配置
        self.debug_shapes = bool(self.prompt_config.DEBUG_SHAPES)
        self._shape_debug_incorporate_logged = False
        self._last_prompt_path_info = {}
        self._last_prompt_distribution_stats = None
        self._last_injected_prompt_tokens = None
        self._last_attention_mediation_stats = []
        self._runtime_prompt_distribution_override = None

        if prompt_init is not None:
            raise ValueError("Static prompt initialization has been removed; prompt_init must be None.")
        if self.prompt_enable and self.prompt_backend == "dynamic" and bool(self.prompt_config.DEEP):
            raise ValueError(
                "MODEL.PROMPT.BACKEND='dynamic' no longer supports MODEL.PROMPT.DEEP=True. "
                "Use MODEL.PROMPT.DEEP=False for dynamic input prompt, or use BACKEND='vpt_deep' for deep prompts."
            )

        if self.prompt_enable and self.prompt_backend == "dynamic" and self.prompt_init_source == "distributor_mean":
            if self.prompt_init_provider is None:
                raise ValueError(
                    "Dynamic prompt with INIT_SOURCE='distributor_mean' requires "
                    "MODEL.PROMPT.DISTRIBUTOR.ENABLE=True and a prompt_init_provider."
                )
        if self.prompt_enable and self.prompt_backend == "vpt_deep" and self.prompt_init_source == "distributor_mean":
            if self.prompt_init_provider is None:
                raise ValueError(
                    "VPT deep prompt with INIT_SOURCE='distributor_mean' requires "
                    "MODEL.PROMPT.DISTRIBUTOR.ENABLE=True and a prompt_init_provider."
                )

        # 把 shape debug 开关同步到语义分支与编码器各层
        if self.semantic_token_projector is not None:
            # orthogonal tokenizer 允许单独打开语义 token 形状调试。
            self.semantic_token_projector.debug_shapes = bool(self.debug_shapes) or bool(self.semantic_tokens_cfg.ORTHO.DEBUG)
        for layer_block in self.encoder.layer:
            setattr(layer_block, "debug_shapes", bool(self.debug_shapes))
            if hasattr(layer_block, "attn"):
                setattr(layer_block.attn, "debug_shapes", bool(self.debug_shapes))

        num_layers = config.transformer["num_layers"]
        if self.prompt_enable and self.prompt_backend == "dynamic":
            if self.prompt_init_source == "learned":
                prompt_dim = config.hidden_size
                val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
                self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
                self.prompt_embeddings.data.uniform_(-val, val, generator=prompt_init_generator)
        elif self.prompt_enable and self.prompt_backend == "vpt_deep":
            prompt_dim = config.hidden_size
            val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
            if self.prompt_init_source == "learned":
                self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
                self.prompt_embeddings.data.uniform_(-val, val, generator=prompt_init_generator)
            if self.prompt_config.DEEP:
                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                self.deep_prompt_embeddings.data.uniform_(-val, val, generator=prompt_init_generator)

        if self.attention_mediation_enable:
            # attention mediation 是“当前 block 内”的注意力修正，因此每一层 ViT block 都有独立 gamma。
            # gamma 初始化为 0 时，mediated correction 初始不影响前向，训练中再逐步学习是否打开。
            self.attention_mediation_prompt_gamma = nn.Parameter(
                torch.full((num_layers,), float(self.attention_mediation_cfg.PROMPT_GAMMA_INIT))
            )
            self.attention_mediation_semantic_gamma = nn.Parameter(
                torch.full((num_layers,), float(self.attention_mediation_cfg.SEMANTIC_GAMMA_INIT))
            )
            logger.info(
                "[attention-mediation] enable=True source=%s execution=%s mlp_policy=%s route_scope=%s "
                "prompt_route=%s semantic_route=%s mass_mode=%s prompt_gamma_init=%.6g semantic_gamma_init=%.6g",
                str(self.attention_mediation_cfg.SOURCE),
                str(self.attention_mediation_cfg.EXECUTION_MODE),
                str(self.attention_mediation_cfg.MLP_POLICY),
                str(self.attention_mediation_cfg.ROUTE_SCOPE),
                str(self.attention_mediation_cfg.PROMPT_ROUTE),
                str(self.attention_mediation_cfg.SEMANTIC_ROUTE),
                str(self.attention_mediation_cfg.MASS_MODE),
                float(self.attention_mediation_cfg.PROMPT_GAMMA_INIT),
                float(self.attention_mediation_cfg.SEMANTIC_GAMMA_INIT),
            )

    def _make_attention_mediation_config(self):
        """
        将配置节点转成传给 Encoder/Block/Attention 的运行时 dict。“配置 + gamma 参数”的运行时包

        这里同时携带 prompt_gamma 和 semantic_gamma 两个 Parameter；
        Block 会按照 layer_idx 取当前层 gamma，只把 delta 写回 P/S token。
        """
        if not self.attention_mediation_enable:
            return None
        return {
            "enable": True,
            "source": str(self.attention_mediation_cfg.SOURCE),
            "execution_mode": str(self.attention_mediation_cfg.EXECUTION_MODE),
            "mlp_policy": str(self.attention_mediation_cfg.MLP_POLICY),
            "route_scope": str(self.attention_mediation_cfg.ROUTE_SCOPE),
            "prompt_route": str(self.attention_mediation_cfg.PROMPT_ROUTE),
            "semantic_route": str(self.attention_mediation_cfg.SEMANTIC_ROUTE),
            "mass_mode": str(self.attention_mediation_cfg.MASS_MODE),
            "beta_prompt_mass": float(self.attention_mediation_cfg.BETA_PROMPT_MASS),
            "beta_semantic_mass": float(self.attention_mediation_cfg.BETA_SEMANTIC_MASS),
            "prompt_gamma": self.attention_mediation_prompt_gamma,
            "semantic_gamma": self.attention_mediation_semantic_gamma,
        }

    def _replace_prompt_tokens(self, hidden_states: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """把主序列中的 prompt 区段替换成新 prompt。

        当前主序列统一采用：
            [CLS | PROMPT | PATCH]
        这里不改 CLS 和 PATCH，只替换中间 prompt 段。
        """
        return torch.cat(
            (
                hidden_states[:, :1, :],
                prompt_tokens,
                hidden_states[:, 1 + self.num_tokens:, :],
            ),
            dim=1,
        )

    @staticmethod
    def _attach_prompt_layer_continuity(affinities):
        previous_output = None
        for affinity in affinities or []:
            previous_output = _attach_prompt_continuity_to_layer(
                affinity, previous_output
            )

    def _active_semantic_length(self, semantics) -> int:
        if self.semantic_tokens_enable and torch.is_tensor(semantics):
            return int(self.semantic_tokens_cfg.NUM_TOKENS)
        return 0

    def _build_vit_cls_prepass(self, x_base: torch.Tensor) -> torch.Tensor:
        """
        为 SOURCE="vit_cls_prepass" 额外跑一次无 prompt / 无 semantic 的 ViT。

        得到的 CLS 只作为 prompt distributor 的视觉统计输入；
        全程 no_grad，不让这次 prepass 参与主训练反传。
        """
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            encoded, _ = self.encoder(
                x_base,
                None,
                num_prompt_tokens=0,
                semantic_length=0,
                block_s_to_cls=False,
            )
        if was_training:
            self.encoder.train(True)
        return encoded[:, 0, :].detach()

    def _call_prompt_init_provider(self, raw_image: torch.Tensor, patch_tokens: torch.Tensor, x_base: torch.Tensor):
        """
        通过统一接口调用 prompt distributor。

        这里同时传入 raw image、未加位置的 patch token、加位置后的 image token，
        以及可选 vit_cls。具体使用哪个由 DISTRIBUTOR.SOURCE 决定。
        """
        if self.prompt_init_provider is None:
            raise ValueError("prompt_init_provider is required for INIT_SOURCE='distributor_mean'.")
        if self._runtime_prompt_distribution_override is not None:
            override = self._runtime_prompt_distribution_override
            self._runtime_prompt_distribution_override = None
            mu = override["mu"].to(device=x_base.device, dtype=x_base.dtype)
            logvar = override["logvar"].to(device=x_base.device, dtype=x_base.dtype)
            eps = override.get("eps")
            if torch.is_tensor(eps):
                eps = eps.to(device=x_base.device, dtype=x_base.dtype)
            if mu.shape[0] != raw_image.shape[0]:
                raise ValueError(
                    f"External prompt batch {mu.shape[0]} does not match image batch {raw_image.shape[0]}."
                )
            return self.prompt_init_provider.prompt_from_distribution(mu, logvar, eps=eps)
        vit_image_tokens = x_base[:, 1:, :]
        vit_cls = None
        if str(self.prompt_init_provider.source) == "vit_cls_prepass":
            vit_cls = self._build_vit_cls_prepass(x_base)
        provider_out = self.prompt_init_provider(
            raw_image=raw_image,
            vit_patch_tokens=patch_tokens,
            vit_image_tokens=vit_image_tokens,
            vit_cls=vit_cls,
        )
        if (not isinstance(provider_out, tuple)) or len(provider_out) != 2:
            raise TypeError("prompt_init_provider must return exactly (prompt_tokens, provider_stats).")
        return provider_out

    def set_runtime_prompt_distribution_override(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> None:
        if self.prompt_init_source != "distributor_mean":
            raise ValueError("External prompt distribution requires INIT_SOURCE='distributor_mean'.")
        self._runtime_prompt_distribution_override = {"mu": mu, "logvar": logvar, "eps": eps}

    def clear_runtime_prompt_distribution_override(self) -> None:
        self._runtime_prompt_distribution_override = None

    def incorporate_prompt(self, x, semantics=None):
        """构造输入层主序列，并在需要时注入输入 prompt。

        真实流程：
        1. 先提取 patch token
        2. 再拼上 CLS 和位置编码，得到不含 prompt 的基础序列 x_base
        3. 若 prompt_enable：
           - learned：输入 prompt 来自 prompt_embeddings
           - distributor_mean：输入 prompt 来自 prompt_init_provider 输出的均值 prompt
        4. 输出统一的 [CLS | PROMPT | PATCH] 序列

        注意：
        - 这里只决定“输入 prompt 从哪里来”
        - 后续层 prompt 如何承接，由 dynamic/vpt_deep 两条后端各自负责。
        """

        # 当前主序列构造顺序为 [CLS | prompt_tokens | visual_tokens | semantic_tokens]。
        # 本函数只负责输入层拼接和 distributor stats 缓存；dynamic backend 不再做层间 prompt 演化。
        B = x.shape[0]
        self._last_semantic_token_state = None
        self._last_prompt_distribution_stats = None
        self._last_injected_prompt_tokens = None

        # 提取 patch token，但此时还没有 CLS / pos / prompt
        # 先提取原始 ViT patch token；此时还没有 CLS、position embedding 和 prompt。
        patch_tokens = self.embeddings.forward_patches(x)  # (B, n_patches, hidden_dim)

        # 先形成不含 prompt 的基础主序列
        # x_base 是无 prompt 的基础序列：[CLS | PATCH+POS]。
        x_base = self.embeddings.add_cls_and_pos(patch_tokens)  # (B, 1 + n_patches, hidden_dim)
        if self.prompt_enable:
            if self.prompt_backend == "dynamic":
                if self.prompt_init_source == "learned":
                    if self.prompt_embeddings is None:
                        raise ValueError("Dynamic prompt with INIT_SOURCE='learned' requires prompt_embeddings.")
                    prompt_tokens = self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
                elif self.prompt_init_source == "distributor_mean":
                    # distributor_mean：输入 prompt 由实例条件均值 mu 生成
                    # distributor_mean：prompt 来自图像条件分布采样，并缓存 mu/logvar 供 KL/语义图 loss 使用。
                    prompt_tokens, provider_stats = self._call_prompt_init_provider(x, patch_tokens, x_base)
                    self._last_prompt_distribution_stats = provider_stats
                    expected_shape = (B, self.num_tokens, self.vit_config.hidden_size)
                    got_shape = tuple(prompt_tokens.shape) if torch.is_tensor(prompt_tokens) else None
                    if (not torch.is_tensor(prompt_tokens)) or tuple(prompt_tokens.shape) != expected_shape:
                        raise ValueError(f"prompt_init_provider must return prompt_tokens with shape {expected_shape}, got {type(prompt_tokens)} {got_shape}")
                else:
                    raise ValueError(f"Unsupported MODEL.PROMPT.INIT_SOURCE='{self.prompt_config.INIT_SOURCE}'")
            elif self.prompt_backend == "vpt_deep":
                if self.prompt_init_source == "learned":
                    if self.prompt_embeddings is None:
                        raise ValueError("VPT deep prompt with INIT_SOURCE='learned' requires prompt_embeddings.")
                    prompt_tokens = self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
                elif self.prompt_init_source == "distributor_mean":
                    # vpt_deep 下也允许只替换输入 prompt 来源，而不改后续 deep prompt 承接方式
                    # vpt_deep 下也只替换输入 prompt 来源，不改变后续 deep prompt 承接方式。
                    prompt_tokens, provider_stats = self._call_prompt_init_provider(x, patch_tokens, x_base)
                    self._last_prompt_distribution_stats = provider_stats
                    expected_shape = (B, self.num_tokens, self.vit_config.hidden_size)
                    got_shape = tuple(prompt_tokens.shape) if torch.is_tensor(prompt_tokens) else None
                    if (not torch.is_tensor(prompt_tokens)) or tuple(prompt_tokens.shape) != expected_shape:
                        raise ValueError(
                            f"prompt_init_provider must return prompt_tokens with shape {expected_shape}, "
                            f"got {type(prompt_tokens)} {got_shape}"
                        )
                else:
                    raise ValueError(f"Unsupported MODEL.PROMPT.INIT_SOURCE='{self.prompt_config.INIT_SOURCE}'")
            else:
                raise ValueError(f"Unsupported MODEL.PROMPT.BACKEND='{self.prompt_backend}'")
            x = torch.cat((
                    x_base[:, :1, :],
                    self.prompt_dropout(prompt_tokens),
                    x_base[:, 1:, :]
                ), dim=1)
        else:
            prompt_tokens = x_base[:, :0, :]
            x = x_base

        self._last_injected_prompt_tokens = (
            prompt_tokens.detach() if torch.is_tensor(prompt_tokens) else None
        )

        # semantic tokens 始终拼在序列最后，保持 affinity monitor 的切片协议不变。
        semantic_tokens = x[:, :0, :]
        if self.semantic_tokens_enable and torch.is_tensor(semantics):
            semantic_tokens, semantic_state = self._init_semantic_tokens(semantics)
            x = torch.cat((x, semantic_tokens), dim=1)
            self._last_semantic_token_state = semantic_state

        if self.debug_shapes and (not self._shape_debug_incorporate_logged):
            print(
                "[SHAPE-DEBUG] PromptedTransformer.incorporate_prompt patch_tokens={} prompt_tokens={} semantic_tokens={} x_base={} x={} semantics={}".format(
                    tuple(patch_tokens.shape),
                    tuple(prompt_tokens.shape) if torch.is_tensor(prompt_tokens) else None,
                    tuple(semantic_tokens.shape),
                    tuple(x_base.shape),
                    tuple(x.shape),
                    tuple(semantics.shape) if torch.is_tensor(semantics) else None,
                )
            )
            self._shape_debug_incorporate_logged = True

        # 记录本次 prompt/semantic 路径，便于 trainer trace 和实验日志确认配置是否真正生效。
        self._last_prompt_path_info = {
            "prompt_enable": bool(self.prompt_enable),
            "prompt_backend": self.prompt_backend,
            "prompt_init_source": self.prompt_init_source,
            "semantic_tokens_enable": bool(self.semantic_tokens_enable),
            "semantic_token_shape": tuple(semantic_tokens.shape),
            "actual_token_shape_entering_backbone": tuple(x.shape),
            "visual_feature_norm": float(patch_tokens.float().norm(dim=-1).mean().item()),
            "prompt_distribution_source": (
                self._last_prompt_distribution_stats.get("visual_source")
                if isinstance(self._last_prompt_distribution_stats, dict)
                else None
            ),
        }

        return x, semantics

    def train(self, mode=True):
        """
        控制训练态。

        这套 prompt tuning 的基本原则是：
        - 冻结 backbone（encoder / embeddings）
        - 只训练 prompt 相关模块，以及需要时的 prompt_init_provider
        """
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_dropout.train()
            if self.semantic_token_projector is not None:
                self.semantic_token_projector.train(mode)
            if self.prompt_backend == "dynamic":
                self.prompt_proj.train(mode)
                if self.prompt_init_source == "distributor_mean" and isinstance(self.prompt_init_provider, torch.nn.Module):
                    self.prompt_init_provider.train(mode)
            elif self.prompt_backend == "vpt_deep":
                self.prompt_proj.train(mode)
                if self.prompt_init_source == "distributor_mean" and isinstance(self.prompt_init_provider, torch.nn.Module):
                    self.prompt_init_provider.train(mode)
        else:
            for module in self.children():
                module.train(mode)

            if isinstance(self.prompt_init_provider, torch.nn.Module):
                self.prompt_init_provider.train(mode)

    def _init_semantic_tokens(self, semantics: Optional[torch.Tensor]):
        """初始化语义侧运行时状态。

        若语义分支关闭，或当前 batch 没有语义输入，则直接返回 None。
        否则创建：
        - sem_state：后续逐层更新的语义状态
        - semantic_input：语义初始输入，便于监控与残差比较
        """
        self._last_semantic_token_state = None
        if (not self.semantic_tokens_enable) or (self.semantic_token_projector is None) or (not torch.is_tensor(semantics)):
            return None, None
        sem_state, semantic_input = self.semantic_token_projector.init_state(
            semantics=semantics,
            device=semantics.device,
        )
        return sem_state, semantic_input

    def _finalize_semantic_token_state(self, encoded: torch.Tensor) -> None:
        # ViT 编码结束后，用最后 k 个 semantic tokens 解码并缓存观测量。
        if self._last_semantic_token_state is None:
            return
        if self.semantic_token_projector is None:
            return
        self._last_semantic_token_state = self.semantic_token_projector.finalize_state(
            encoded=encoded,
            state=self._last_semantic_token_state,
        )

    def _update_semantic_tokens(self, sem_state, hidden_states, layer_idx: int, num_layers: int):
        """
        从主序列中切出当前层 prompt / visual token，
        并让语义分支执行一步交互更新。
        """
        raise RuntimeError("Semantic tokens are updated only by ViT self-attention in the main token sequence.")

    def forward_deep_prompt(self, embedding_output, semantics=None):
        """
        Deep prompt 主前向。

        层内动线：
        - 第 0 层：直接使用 incorporate_prompt 生成的 [CLS|P|PATCH]
        - 第 1~L-1 层：
          - vpt_deep：直接取 deep_prompt_embeddings[i-1]
        - 每一层 transformer block 执行后，再让语义分支读取当前 prompt / visual
          并更新自己的 sem_state
        """
        attn_weights: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_state, semantic_input = None, None
        semantic_length = self._active_semantic_length(semantics)
        if self.attention_mediation_enable and semantic_length <= 0:
            raise ValueError("Attention mediation requires semantic tensor input for every forward pass.")
        attention_mediation_config = self._make_attention_mediation_config()
        # 深层 prompt 前向逐层调用 encoder.layer[i]，因此 mediation 配置在这里逐层传入。
        # 每层 Block 会用 layer index 选择自己的 gamma，只在当前层 attention 内修正 P/S。
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights, _ = self.encoder.layer[i](
                    hidden_states,
                    None,
                    self.num_tokens,
                    semantic_length,
                    self.block_s_to_cls,
                    attention_mediation_config,
                    i,
                )
                if torch.is_tensor(hidden_states):
                    row_hidden_ok = torch.isfinite(hidden_states).flatten(1).all(dim=1)
                    if not bool(row_hidden_ok.all().item()):
                        bad_hidden = (~row_hidden_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                        print(f"[nan-locate] layer={i} bad_hidden_rows={bad_hidden}")
            else:
                prev_prompt = hidden_states[:, 1:1 + self.num_tokens, :]
                if torch.is_tensor(prev_prompt):
                    row_prev_ok = torch.isfinite(prev_prompt).flatten(1).all(dim=1)
                    if not bool(row_prev_ok.all().item()):
                        bad_prev = (~row_prev_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                        print(f"[nan-locate] layer={i} bad_prev_prompt_rows={bad_prev}")

                if self.prompt_backend == "vpt_deep":
                    if self.deep_prompt_embeddings is None:
                        raise ValueError("VPT deep prompt backend requires deep_prompt_embeddings when PROMPT.DEEP=True")
                    next_prompt = self.prompt_dropout(
                        self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(hidden_states.shape[0], -1, -1)
                    )
                else:
                    raise ValueError(f"Unsupported MODEL.PROMPT.BACKEND='{self.prompt_backend}'")

                if torch.is_tensor(next_prompt):
                    row_next_ok = torch.isfinite(next_prompt).flatten(1).all(dim=1)
                    if not bool(row_next_ok.all().item()):
                        bad_next = (~row_next_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                        print(f"[nan-locate] layer={i} bad_next_prompt_rows={bad_next}")

                hidden_states = self._replace_prompt_tokens(hidden_states, next_prompt)

                hidden_states, weights, _ = self.encoder.layer[i](
                    hidden_states,
                    None,
                    self.num_tokens,
                    semantic_length,
                    self.block_s_to_cls,
                    attention_mediation_config,
                    i,
                )
                if torch.is_tensor(hidden_states):
                    row_hidden_ok = torch.isfinite(hidden_states).flatten(1).all(dim=1)
                    if not bool(row_hidden_ok.all().item()):
                        bad_hidden = (~row_hidden_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                        print(f"[nan-locate] layer={i} bad_hidden_rows={bad_hidden}")
            # 每层 block 之后，让语义侧读取当前 prompt/visual 状态再更新一次
            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_deep_prompt_with_affinity(self, embedding_output, affinity_config, semantics=None):
        """带 affinity 输出的 deep prompt 前向。

        主线和 forward_deep_prompt 一致，只是会额外把每层 affinity
        以及语义侧交互产生的注意力统计一并带出。
        """
        attn_weights: list = []
        affinities: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_state, semantic_input = None, None
        semantic_length = self._active_semantic_length(semantics)
        if self.attention_mediation_enable and semantic_length <= 0:
            raise ValueError("Attention mediation requires semantic tensor input for every forward pass.")
        attention_mediation_config = self._make_attention_mediation_config()
        # 带 affinity 输出的 deep prompt 路径也必须传入同一 mediation 配置，
        # 否则训练前向和可视化/监测前向会走不同的 attention 计算图。
        effective_affinity_config = affinity_config
        offload_diagnostics = bool(
            effective_affinity_config.get(
                "offload_diagnostics_to_cpu", False
            )
        )
        if offload_diagnostics and torch.is_grad_enabled():
            raise RuntimeError(
                "Affinity diagnostic CPU offload is only valid in a no-grad forward"
            )
        previous_prompt_output = None
        for i in range(num_layers):
            if i == 0:
                # 第 0 层直接跑 ViT block 并导出本层 affinity，供监测/可视化使用。
                hidden_states, weights, affinity, _ = self.encoder.layer[i].forward_with_affinity(
                    hidden_states,
                    effective_affinity_config,
                    None,
                    self.num_tokens,
                    attention_mediation_config,
                    i,
                )
            else:
                if self.prompt_backend == "vpt_deep":
                    if self.deep_prompt_embeddings is None:
                        raise ValueError("VPT deep prompt backend requires deep_prompt_embeddings when PROMPT.DEEP=True")
                    next_prompt = self.prompt_dropout(
                        self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(hidden_states.shape[0], -1, -1)
                    )
                else:
                    raise ValueError(f"Unsupported MODEL.PROMPT.BACKEND='{self.prompt_backend}'")

                hidden_states = self._replace_prompt_tokens(hidden_states, next_prompt)

                hidden_states, weights, affinity, _ = self.encoder.layer[i].forward_with_affinity(
                    hidden_states,
                    effective_affinity_config,
                    None,
                    self.num_tokens,
                    attention_mediation_config,
                    i,
                )
            if offload_diagnostics:
                previous_prompt_output = _attach_prompt_continuity_to_layer(
                    affinity, previous_prompt_output
                )
            if self.encoder.vis:
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

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights, affinities
    def forward(self, x, semantics=None):
        """
        标准前向入口。

        - 若启用 deep prompt，则走 forward_deep_prompt
        - 否则直接走 encoder，并在末层后补一次语义侧更新
        """
        embedding_output, semantics = self.incorporate_prompt(x, semantics)

        effective_prompt_tokens = self.num_tokens
        semantic_length = self._active_semantic_length(semantics)
        if self.attention_mediation_enable and semantic_length <= 0:
            raise ValueError("Attention mediation requires semantic tensor input for every forward pass.")
        attention_mediation_config = self._make_attention_mediation_config()
        # 非 deep 路径直接进入 Encoder；配置会在 Encoder 内逐层下发给每个 Block。
        if self.prompt_enable and self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output, semantics)
        else:
            encoded, attn_weights = self.encoder(
                embedding_output,
                None,
                effective_prompt_tokens,
                semantic_length,
                self.block_s_to_cls,
                attention_mediation_config,
            )

        self._last_attention_mediation_stats = [
            dict(block._last_attention_mediation_stats)
            for block in self.encoder.layer
            if block._last_attention_mediation_stats is not None
        ]
        self._finalize_semantic_token_state(encoded)
        self._last_token_sequence = encoded
        return encoded, attn_weights

    def forward_with_affinity(self, x, affinity_config, semantics=None):
        """
        带 affinity 输出的标准前向入口。

        - 先通过 incorporate_prompt 得到 [CLS|P|PATCH]
        - 若开启 deep prompt，走 forward_deep_prompt_with_affinity
        - 否则直接走 encoder.forward_with_affinity
        - 若未走 deep prompt，则在最后再补一次语义侧更新
        """
        embedding_output, semantics = self.incorporate_prompt(x, semantics)

        effective_prompt_tokens = self.num_tokens
        effective_affinity_config = affinity_config
        semantic_length = self._active_semantic_length(semantics)
        if self.attention_mediation_enable and semantic_length <= 0:
            raise ValueError("Attention mediation requires semantic tensor input for every forward pass.")
        attention_mediation_config = self._make_attention_mediation_config()
        # monitor/affinity 前向复用同一 mediation 配置，保证导出的亲和矩阵对应当前真实前向。

        if self.prompt_enable and self.prompt_config.DEEP:
            encoded, attn_weights, affinities = self.forward_deep_prompt_with_affinity(
                embedding_output, effective_affinity_config, semantics
            )
        else:
            encoded, attn_weights, affinities = self.encoder.forward_with_affinity(
                embedding_output,
                effective_affinity_config,
                None,
                effective_prompt_tokens,
                attention_mediation_config,
            )

        offload_diagnostics = bool(
            effective_affinity_config.get(
                "offload_diagnostics_to_cpu", False
            )
        )
        if not offload_diagnostics:
            self._attach_prompt_layer_continuity(affinities)

        attention_mediation_stats = [
            dict(block._last_attention_mediation_stats)
            for block in self.encoder.layer
            if block._last_attention_mediation_stats is not None
        ]
        self._last_attention_mediation_stats = (
            _offload_diagnostic_tree(attention_mediation_stats)
            if offload_diagnostics
            else attention_mediation_stats
        )
        self._finalize_semantic_token_state(encoded)
        self._last_token_sequence = encoded
        return encoded, attn_weights, affinities

class PromptedVisionTransformer(VisionTransformer):
    """
    Replace the original VisionTransformer's internal transformer backbone with PromptedTransformer
    """
    def __init__(
        self,
        prompt_cfg,
        model_type,
        img_size=224,
        num_classes=21843,
        vis=False,
        prompt_init=None,
        prompt_init_provider=None,
        prompt_init_seed=None,
    ):
        super(PromptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg

        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg,
            vit_cfg,
            img_size,
            vis,
            prompt_init=prompt_init,
            prompt_init_provider=prompt_init_provider,
            prompt_init_seed=prompt_init_seed,
        )

    def forward(self, x, vis=False, semantics=None):
        x, attn_weights = self.transformer(x, semantics)

        x = x[:, 0] # CLS

        logits = self.head(x)   # Category Header

        if not vis:
            return logits
        return logits, attn_weights

    def forward_with_affinity(self, x, affinity_config, vis=False, semantics=None):

        x, attn_weights, affinities = self.transformer.forward_with_affinity(x, affinity_config, semantics)

        logits = self.head(x[:, 0])

        if not vis:
            return logits, affinities
        return logits, attn_weights, affinities
