#!/usr/bin/env python3
"""
vit_prompt 主干实现。

当前文件同时承载两条 prompt 主线和一条轻量语义支线：

1. Prompt 主线
   - BACKEND="dynamic"：
     输入 prompt 可来自 learned prompt 或 distributor(mean)，
     后续层 prompt 通过 prompt_update_layers 逐层演化。
   - BACKEND="vpt_deep"：
     输入 prompt 同样可来自 learned prompt 或 distributor(mean)，
     但后续层 prompt 直接使用各层独立的 deep_prompt_embeddings。

2. 语义侧支线
   - 当前已简化成最薄形式：
       semantics -> hidden 对齐 -> 与 prompt/visual 交互 -> 输出
   - 旧版 anchor/free token、复杂 readout 等语义建模结构已不再使用。
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from typing import Optional, Dict, Any, Tuple, List

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout, LayerNorm, Linear
from scipy import ndimage


from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

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

class PromptedTransformer(Transformer):
    """PromptedTransformer 主入口。

    统一管理：
    1. Prompt 是否启用及其后端（dynamic / vpt_deep）
    2. 输入 prompt 来源（learned / distributor_mean）
    3. 轻量语义交互分支在 backbone 中的逐层接入
    """

    def __init__(self, prompt_config, config, img_size, vis, prompt_init=None, prompt_init_provider=None):

        self.semantic_tokens_cfg = prompt_config.SEMANTIC_TOKENS
        self.semantic_tokens_enable = bool(self.semantic_tokens_cfg.ENABLE)
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
            self.semantic_token_projector = SemanticTokenProjector(
                hidden_size=int(config.hidden_size),
                semantic_tokens_cfg=self.semantic_tokens_cfg,
            )
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
        self.prompt_proj = nn.Identity()
        self.prompt_embeddings = None
        self.deep_prompt_embeddings = None
        self.prompt_update_layers = nn.ModuleList()

        # 运行期 debug 配置
        self.debug_shapes = bool(self.prompt_config.DEBUG_SHAPES)
        self._shape_debug_incorporate_logged = False
        self._last_prompt_path_info = {}

        if prompt_init is not None:
            raise ValueError("Static prompt initialization has been removed; prompt_init must be None.")

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
            self.semantic_token_projector.debug_shapes = bool(self.debug_shapes)
        for layer_block in self.encoder.layer:
            setattr(layer_block, "debug_shapes", bool(self.debug_shapes))
            if hasattr(layer_block, "attn"):
                setattr(layer_block.attn, "debug_shapes", bool(self.debug_shapes))

        # dynamic backend：
        # - 输入 prompt 可来自 learned/distributor
        # - 后续层 prompt 一律由上一层 prompt 通过 prompt_update_layers 演化得到
        num_layers = config.transformer["num_layers"]
        hidden_size = config.hidden_size
        if self.prompt_enable and self.prompt_backend == "dynamic":
            self.prompt_update_layers = nn.ModuleList([Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])

            evolve_mode = self.prompt_config.EVOLVE_INIT_MODE.lower()
            if evolve_mode != "identity":
                raise ValueError(f"Unsupported PROMPT.EVOLVE_INIT_MODE: {self.prompt_config.EVOLVE_INIT_MODE}")
            self.evolve_init_mode = evolve_mode

            with torch.no_grad():
                for layer in self.prompt_update_layers:
                    eye = torch.eye(hidden_size, device=layer.weight.device, dtype=layer.weight.dtype)
                    layer.weight.copy_(eye)
                    if layer.bias is not None:
                        layer.bias.zero_()
            if self.prompt_init_source == "learned":
                prompt_dim = config.hidden_size
                val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
                self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        elif self.prompt_enable and self.prompt_backend == "vpt_deep":
            prompt_dim = config.hidden_size
            val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
            if self.prompt_init_source == "learned":
                self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.prompt_config.DEEP:
                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
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
        - 后续层 prompt 如何承接，由 dynamic/vpt_deep 两条后端各自负责
        """

        B = x.shape[0]
        self._last_semantic_token_state = None

        # 提取 patch token，但此时还没有 CLS / pos / prompt
        patch_tokens = self.embeddings.forward_patches(x)  # (B, n_patches, hidden_dim)

        # 先形成不含 prompt 的基础主序列
        x_base = self.embeddings.add_cls_and_pos(patch_tokens)  # (B, 1 + n_patches, hidden_dim)
        if self.prompt_enable:
            if self.prompt_backend == "dynamic":
                if self.prompt_init_source == "learned":
                    if self.prompt_embeddings is None:
                        raise ValueError("Dynamic prompt with INIT_SOURCE='learned' requires prompt_embeddings.")
                    prompt_tokens = self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
                elif self.prompt_init_source == "distributor_mean":
                    # distributor_mean：输入 prompt 由实例条件均值 mu 生成
                    provider_out = self.prompt_init_provider(patch_tokens)
                    if (not isinstance(provider_out, tuple)) or len(provider_out) != 2:
                        raise TypeError("prompt_init_provider must return exactly (prompt_tokens, provider_stats).")
                    prompt_tokens, provider_stats = provider_out
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
                    provider_out = self.prompt_init_provider(patch_tokens)
                    if (not isinstance(provider_out, tuple)) or len(provider_out) != 2:
                        raise TypeError("prompt_init_provider must return exactly (prompt_tokens, provider_stats).")
                    prompt_tokens, provider_stats = provider_out
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

        self._last_prompt_path_info = {
            "prompt_enable": bool(self.prompt_enable),
            "prompt_backend": self.prompt_backend,
            "prompt_init_source": self.prompt_init_source,
            "semantic_tokens_enable": bool(self.semantic_tokens_enable),
            "semantic_token_shape": tuple(semantic_tokens.shape),
            "actual_token_shape_entering_backbone": tuple(x.shape),
            "visual_feature_norm": float(patch_tokens.float().norm(dim=-1).mean().item()),
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
                self.prompt_update_layers.train(mode)
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
          - dynamic：上一层 prompt 经 prompt_update_layers 演化得到新 prompt
          - vpt_deep：直接取 deep_prompt_embeddings[i-1]
        - 每一层 transformer block 执行后，再让语义分支读取当前 prompt / visual
          并更新自己的 sem_state
        """
        attn_weights: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_state, semantic_input = None, None
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights, _ = self.encoder.layer[i](hidden_states, None, self.num_tokens)
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

                if self.prompt_backend == "dynamic":
                    next_prompt = self.prompt_update_layers[i - 1](prev_prompt)
                    next_prompt = self.prompt_dropout(next_prompt)
                elif self.prompt_backend == "vpt_deep":
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

                hidden_states, weights, _ = self.encoder.layer[i](hidden_states, None, self.num_tokens)
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
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights, affinity, _ = self.encoder.layer[i].forward_with_affinity(
                    hidden_states, affinity_config, None, self.num_tokens
                )
            else:
                prev_prompt = hidden_states[:, 1:1 + self.num_tokens, :]
                if self.prompt_backend == "dynamic":
                    next_prompt = self.prompt_update_layers[i - 1](prev_prompt)
                    next_prompt = self.prompt_dropout(next_prompt)
                elif self.prompt_backend == "vpt_deep":
                    if self.deep_prompt_embeddings is None:
                        raise ValueError("VPT deep prompt backend requires deep_prompt_embeddings when PROMPT.DEEP=True")
                    next_prompt = self.prompt_dropout(
                        self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(hidden_states.shape[0], -1, -1)
                    )
                else:
                    raise ValueError(f"Unsupported MODEL.PROMPT.BACKEND='{self.prompt_backend}'")

                hidden_states = self._replace_prompt_tokens(hidden_states, next_prompt)

                hidden_states, weights, affinity, _ = self.encoder.layer[i].forward_with_affinity(
                    hidden_states, affinity_config, None, self.num_tokens
                )
            if self.encoder.vis:
                attn_weights.append(weights)
            affinities.append(affinity)

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
        if self.prompt_enable and self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output, semantics)
        else:
            encoded, attn_weights = self.encoder(embedding_output, None, effective_prompt_tokens)

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

        if self.prompt_enable and self.prompt_config.DEEP:
            encoded, attn_weights, affinities = self.forward_deep_prompt_with_affinity(
                embedding_output, effective_affinity_config, semantics
            )
        else:
            encoded, attn_weights, affinities = self.encoder.forward_with_affinity(
                embedding_output, effective_affinity_config, None, effective_prompt_tokens
            )

        return encoded, attn_weights, affinities

class PromptedVisionTransformer(VisionTransformer):
    """
    Replace the original VisionTransformer's internal transformer backbone with PromptedTransformer
    """
    def __init__(self, prompt_cfg, model_type,img_size=224, num_classes=21843, vis=False, prompt_init=None, prompt_init_provider=None):
        super(PromptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg

        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(prompt_cfg, vit_cfg, img_size, vis, prompt_init=prompt_init, prompt_init_provider=prompt_init_provider,)

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
