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
        self.block_s_to_cls = bool(self.semantic_tokens_cfg.BLOCK_S_TO_CLS)
        self.affinity_evolution_cfg = prompt_config.AFFINITY_EVOLUTION
        self.affinity_evolution_enable = bool(self.affinity_evolution_cfg.ENABLE)
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
        self.affinity_evolution_prompt_norm = nn.Identity()
        self.affinity_evolution_semantic_norm = nn.Identity()
        self.affinity_evolution_prompt_gamma = None
        self.affinity_evolution_semantic_gamma = None
        self._affinity_evolution_scale_logged = False

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
        self._validate_affinity_evolution_config()

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
            if self.affinity_evolution_enable:
                self.affinity_evolution_prompt_norm = LayerNorm(hidden_size, eps=1e-6)
                self.affinity_evolution_semantic_norm = LayerNorm(hidden_size, eps=1e-6)
                self.affinity_evolution_prompt_gamma = nn.Parameter(
                    torch.full((num_layers - 1,), float(self.affinity_evolution_cfg.PROMPT_GAMMA_INIT))
                )
                self.affinity_evolution_semantic_gamma = nn.Parameter(
                    torch.full((num_layers - 1,), float(self.affinity_evolution_cfg.SEMANTIC_GAMMA_INIT))
                )
                logger.info(
                    "[affinity-evolution] enable=%s prompt=%s semantic=%s prompt_target=%s semantic_target=%s "
                    "prompt_lambda=%.6g semantic_lambda=%.6g prompt_gamma_init=%.6g semantic_gamma_init=%.6g "
                    "prompt_detach=%s semantic_detach=%s semantic_compose=%s teacher_student_route=True",
                    True,
                    bool(self.affinity_evolution_cfg.PROMPT_ENABLE),
                    bool(self.affinity_evolution_cfg.SEMANTIC_ENABLE),
                    str(self.affinity_evolution_cfg.PROMPT_TARGET),
                    str(self.affinity_evolution_cfg.SEMANTIC_TARGET),
                    float(self.affinity_evolution_cfg.PROMPT_LAMBDA),
                    float(self.affinity_evolution_cfg.SEMANTIC_LAMBDA),
                    float(self.affinity_evolution_cfg.PROMPT_GAMMA_INIT),
                    float(self.affinity_evolution_cfg.SEMANTIC_GAMMA_INIT),
                    str(self.affinity_evolution_cfg.PROMPT_DETACH),
                    str(self.affinity_evolution_cfg.SEMANTIC_DETACH),
                    str(self.affinity_evolution_cfg.SEMANTIC_COMPOSE),
                )
            else:
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

    def _validate_affinity_evolution_config(self):
        if not self.affinity_evolution_enable:
            return
        if not self.prompt_enable:
            raise ValueError("MODEL.AFFINITY_EVOLUTION.ENABLE requires MODEL.PROMPT.ENABLE=True.")
        if self.prompt_backend != "dynamic":
            raise ValueError("MODEL.AFFINITY_EVOLUTION.ENABLE currently supports only MODEL.PROMPT.BACKEND='dynamic'.")
        if not bool(self.prompt_config.DEEP):
            raise ValueError("MODEL.AFFINITY_EVOLUTION.ENABLE requires MODEL.PROMPT.DEEP=True.")
        if not self.semantic_tokens_enable:
            raise ValueError("MODEL.AFFINITY_EVOLUTION.ENABLE requires MODEL.SEMANTIC_TOKENS.ENABLE=True.")
        if int(self.semantic_tokens_cfg.NUM_TOKENS) <= 0:
            raise ValueError("MODEL.AFFINITY_EVOLUTION.ENABLE requires semantic token length > 0.")
        if int(self.prompt_config.NUM_TOKENS) <= 0:
            raise ValueError("MODEL.AFFINITY_EVOLUTION.ENABLE requires prompt token length > 0.")

        valid_targets = {"QpKv", "QpQv", "KpKv"}
        if str(self.affinity_evolution_cfg.PROMPT_TARGET) not in valid_targets:
            raise ValueError(f"Unsupported AFFINITY_EVOLUTION.PROMPT_TARGET='{self.affinity_evolution_cfg.PROMPT_TARGET}'")
        if str(self.affinity_evolution_cfg.SEMANTIC_TARGET) not in valid_targets:
            raise ValueError(f"Unsupported AFFINITY_EVOLUTION.SEMANTIC_TARGET='{self.affinity_evolution_cfg.SEMANTIC_TARGET}'")
        if str(self.affinity_evolution_cfg.PROMPT_DETACH) not in {"mediated", "direct", "none"}:
            raise ValueError(f"Unsupported AFFINITY_EVOLUTION.PROMPT_DETACH='{self.affinity_evolution_cfg.PROMPT_DETACH}'")
        if str(self.affinity_evolution_cfg.SEMANTIC_DETACH) not in {"via_prompt", "direct", "none"}:
            raise ValueError(f"Unsupported AFFINITY_EVOLUTION.SEMANTIC_DETACH='{self.affinity_evolution_cfg.SEMANTIC_DETACH}'")
        if str(self.affinity_evolution_cfg.SEMANTIC_COMPOSE) not in {"prob", "raw_then_norm"}:
            raise ValueError(f"Unsupported AFFINITY_EVOLUTION.SEMANTIC_COMPOSE='{self.affinity_evolution_cfg.SEMANTIC_COMPOSE}'")
        if not 0.0 <= float(self.affinity_evolution_cfg.PROMPT_LAMBDA) <= 1.0:
            raise ValueError("AFFINITY_EVOLUTION.PROMPT_LAMBDA must be in [0, 1].")
        if not 0.0 <= float(self.affinity_evolution_cfg.SEMANTIC_LAMBDA) <= 1.0:
            raise ValueError("AFFINITY_EVOLUTION.SEMANTIC_LAMBDA must be in [0, 1].")

    def _make_affinity_evolution_config(self, affinity_config=None):
        if affinity_config is None:
            return {
                "prompt_length": int(self.num_tokens),
                "semantic_length": int(self.semantic_tokens_cfg.NUM_TOKENS),
                "detach": False,
                "block_s_to_cls": bool(self.block_s_to_cls),
            }
        cfg = dict(affinity_config)
        cfg["prompt_length"] = int(self.num_tokens)
        cfg["semantic_length"] = int(self.semantic_tokens_cfg.NUM_TOKENS)
        cfg["detach"] = False
        cfg["block_s_to_cls"] = bool(self.block_s_to_cls)
        return cfg

    @staticmethod
    def _mean_head_affinity(affinity: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        raw = affinity[key]
        if raw.dim() != 4:
            raise ValueError(f"Affinity '{key}' must have shape [B,H,*,*], got {tuple(raw.shape)}.")
        return raw.mean(dim=1)

    @staticmethod
    def _affinity_evolution_tensor_stats(name: str, tensor: torch.Tensor) -> Dict[str, float]:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Affinity evolution tensor '{name}' contains NaN or Inf.")
        t = tensor.detach().float()
        return {
            f"{name}_mean": float(t.mean().item()),
            f"{name}_std": float(t.std(unbiased=False).item()),
            f"{name}_min": float(t.min().item()),
            f"{name}_max": float(t.max().item()),
        }

    def _check_affinity_evolution_scales(
        self,
        layer_idx: int,
        apv_prompt: torch.Tensor,
        sem_pv: torch.Tensor,
        qskp: torch.Tensor,
        qskv: torch.Tensor,
        apv_semantic: torch.Tensor,
    ) -> None:
        stats = {}
        stats.update(self._affinity_evolution_tensor_stats("apv_prompt", apv_prompt))
        stats.update(self._affinity_evolution_tensor_stats("sem_pv", sem_pv))
        stats.update(self._affinity_evolution_tensor_stats("qskp", qskp))
        stats.update(self._affinity_evolution_tensor_stats("qskv", qskv))
        stats.update(self._affinity_evolution_tensor_stats("apv_semantic", apv_semantic))

        sem_std = stats["sem_pv_std"]
        apv_std = stats["apv_prompt_std"]
        if sem_std > apv_std * 100.0 or apv_std > sem_std * 100.0:
            logger.warning(
                "[affinity-evolution-scale] layer=%d large scale gap: apv_prompt_std=%.6g sem_pv_std=%.6g",
                int(layer_idx),
                apv_std,
                sem_std,
            )

        if not self._affinity_evolution_scale_logged:
            logger.info(
                "[affinity-evolution-scale] layer=%d "
                "apv_prompt(mean=%.6g,std=%.6g,min=%.6g,max=%.6g) "
                "sem_pv(mean=%.6g,std=%.6g,min=%.6g,max=%.6g) "
                "qskp(mean=%.6g,std=%.6g,min=%.6g,max=%.6g) "
                "qskv(mean=%.6g,std=%.6g,min=%.6g,max=%.6g) "
                "apv_semantic(mean=%.6g,std=%.6g,min=%.6g,max=%.6g)",
                int(layer_idx),
                stats["apv_prompt_mean"], stats["apv_prompt_std"], stats["apv_prompt_min"], stats["apv_prompt_max"],
                stats["sem_pv_mean"], stats["sem_pv_std"], stats["sem_pv_min"], stats["sem_pv_max"],
                stats["qskp_mean"], stats["qskp_std"], stats["qskp_min"], stats["qskp_max"],
                stats["qskv_mean"], stats["qskv_std"], stats["qskv_min"], stats["qskv_max"],
                stats["apv_semantic_mean"], stats["apv_semantic_std"], stats["apv_semantic_min"], stats["apv_semantic_max"],
            )
            self._affinity_evolution_scale_logged = True

    @staticmethod
    def _teacher_student_route(
        student: torch.Tensor,
        teacher: torch.Tensor,
        correction_lambda: float,
    ) -> torch.Tensor:
        # teacher 只提供前向修正方向；梯度仍从 student 路径回传。
        return student + correction_lambda * (teacher.detach() - student).detach()

    def _build_prompt_evolution_route(
        self,
        direct: torch.Tensor,
        mediated: torch.Tensor,
    ) -> torch.Tensor:
        prompt_detach = str(self.affinity_evolution_cfg.PROMPT_DETACH)
        prompt_lambda = float(self.affinity_evolution_cfg.PROMPT_LAMBDA)

        if prompt_detach == "mediated":
            # direct 学生被 semantic-mediated teacher 拉向语义中介路径。
            return self._teacher_student_route(direct, mediated, prompt_lambda)
        if prompt_detach == "direct":
            # semantic-mediated 学生被 direct teacher 拉向主干直接亲和路径。
            return self._teacher_student_route(mediated, direct, prompt_lambda)
        if prompt_detach == "none":
            # 无 teacher 固定方向时，两条概率路径共同参与前向和反向。
            return (1.0 - prompt_lambda) * direct + prompt_lambda * mediated
        raise ValueError(f"Unsupported AFFINITY_EVOLUTION.PROMPT_DETACH='{prompt_detach}'")

    def _build_semantic_evolution_route(
        self,
        direct: torch.Tensor,
        via_prompt: torch.Tensor,
    ) -> torch.Tensor:
        semantic_detach = str(self.affinity_evolution_cfg.SEMANTIC_DETACH)
        semantic_lambda = float(self.affinity_evolution_cfg.SEMANTIC_LAMBDA)

        if semantic_detach == "via_prompt":
            # direct 学生被 semantic->prompt->visual teacher 修正。
            return self._teacher_student_route(direct, via_prompt, semantic_lambda)
        if semantic_detach == "direct":
            # via_prompt 学生被 semantic->visual direct teacher 修正。
            return self._teacher_student_route(via_prompt, direct, semantic_lambda)
        if semantic_detach == "none":
            # 无 teacher 固定方向时，direct 与 via_prompt 做概率插值。
            return (1.0 - semantic_lambda) * direct + semantic_lambda * via_prompt
        raise ValueError(f"Unsupported AFFINITY_EVOLUTION.SEMANTIC_DETACH='{semantic_detach}'")

    def _replace_prompt_and_semantic_tokens(
        self,
        hidden_states: torch.Tensor,
        prompt_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
    ) -> torch.Tensor:
        semantic_length = int(semantic_tokens.shape[1])
        return torch.cat(
            (
                hidden_states[:, :1, :],
                prompt_tokens,
                hidden_states[:, 1 + self.num_tokens:-semantic_length, :],
                semantic_tokens,
            ),
            dim=1,
        )

    def _apply_affinity_evolution(
        self,
        hidden_states: torch.Tensor,
        prev_affinity: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        semantic_length = int(self.semantic_tokens_cfg.NUM_TOKENS)
        if semantic_length <= 0:
            raise ValueError("Affinity evolution requires semantic_length > 0.")

        prev_prompt = hidden_states[:, 1:1 + self.num_tokens, :]
        prev_visual = hidden_states[:, 1 + self.num_tokens:-semantic_length, :]
        prev_semantic = hidden_states[:, -semantic_length:, :]

        prompt_target_key = f"{self.affinity_evolution_cfg.PROMPT_TARGET}_raw"
        semantic_target_key = f"{self.affinity_evolution_cfg.SEMANTIC_TARGET}_raw"
        apv_prompt = self._mean_head_affinity(prev_affinity, prompt_target_key)
        apv_semantic = self._mean_head_affinity(prev_affinity, semantic_target_key)
        qskp = self._mean_head_affinity(prev_affinity, "QsKp_raw")
        qskv = self._mean_head_affinity(prev_affinity, "QsKv_raw")

        next_prompt = prev_prompt
        sem_pv = torch.bmm(qskp.transpose(1, 2), qskv)
        self._check_affinity_evolution_scales(layer_idx, apv_prompt, sem_pv, qskp, qskv, apv_semantic)
        if bool(self.affinity_evolution_cfg.PROMPT_ENABLE):
            # 先把两条 prompt->visual 路径变成概率路由，再按 detach 字段选择 teacher/student。
            direct_prompt = torch.softmax(apv_prompt, dim=-1)
            mediated_prompt = torch.softmax(sem_pv, dim=-1)
            prompt_route = self._build_prompt_evolution_route(direct_prompt, mediated_prompt)
            delta_prompt = torch.bmm(prompt_route, prev_visual)
            gamma_prompt = self.affinity_evolution_prompt_gamma[layer_idx - 1].view(1, 1, 1)
            next_prompt = prev_prompt + gamma_prompt * self.affinity_evolution_prompt_norm(delta_prompt - prev_prompt)
            next_prompt = self.prompt_dropout(next_prompt)

        next_semantic = prev_semantic
        if bool(self.affinity_evolution_cfg.SEMANTIC_ENABLE):
            compose = str(self.affinity_evolution_cfg.SEMANTIC_COMPOSE)
            if compose == "prob":
                # prob: 每一段先转成概率，再组合 semantic->prompt->visual 路由。
                via_prompt = torch.bmm(torch.softmax(qskp, dim=-1), torch.softmax(apv_semantic, dim=-1))
                direct = torch.softmax(qskv, dim=-1)
            elif compose == "raw_then_norm":
                # raw_then_norm: 先做原始三元路径乘法，再整体 softmax 成 semantic->visual 路由。
                via_prompt = torch.bmm(qskp, apv_semantic)
                via_prompt = torch.softmax(via_prompt, dim=-1)
                direct = torch.softmax(qskv, dim=-1)
            else:
                raise ValueError(f"Unsupported AFFINITY_EVOLUTION.SEMANTIC_COMPOSE='{compose}'")
            semantic_route = self._build_semantic_evolution_route(direct, via_prompt)
            delta_semantic = torch.bmm(semantic_route, prev_visual)
            gamma_semantic = self.affinity_evolution_semantic_gamma[layer_idx - 1].view(1, 1, 1)
            next_semantic = prev_semantic + gamma_semantic * self.affinity_evolution_semantic_norm(delta_semantic - prev_semantic)

        return self._replace_prompt_and_semantic_tokens(hidden_states, next_prompt, next_semantic)

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

    def _active_semantic_length(self, semantics) -> int:
        if self.semantic_tokens_enable and torch.is_tensor(semantics):
            return int(self.semantic_tokens_cfg.NUM_TOKENS)
        return 0

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
            "affinity_evolution_enable": bool(self.affinity_evolution_enable),
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
        if self.affinity_evolution_enable:
            if self._active_semantic_length(semantics) <= 0:
                raise ValueError("Affinity evolution requires semantic tensor input for every forward pass.")
            encoded, attn_weights, _ = self.forward_deep_prompt_with_affinity(
                embedding_output,
                self._make_affinity_evolution_config(),
                semantics,
            )
            return encoded, attn_weights

        attn_weights: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_state, semantic_input = None, None
        semantic_length = self._active_semantic_length(semantics)
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights, _ = self.encoder.layer[i](
                    hidden_states,
                    None,
                    self.num_tokens,
                    semantic_length,
                    self.block_s_to_cls,
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

                hidden_states, weights, _ = self.encoder.layer[i](
                    hidden_states,
                    None,
                    self.num_tokens,
                    semantic_length,
                    self.block_s_to_cls,
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
        if self.affinity_evolution_enable and self._active_semantic_length(semantics) <= 0:
            raise ValueError("Affinity evolution requires semantic tensor input for every forward pass.")
        effective_affinity_config = (
            self._make_affinity_evolution_config(affinity_config)
            if self.affinity_evolution_enable
            else affinity_config
        )
        prev_affinity = None
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights, affinity, _ = self.encoder.layer[i].forward_with_affinity(
                    hidden_states, effective_affinity_config, None, self.num_tokens
                )
            else:
                if self.prompt_backend == "dynamic" and self.affinity_evolution_enable:
                    if prev_affinity is None:
                        raise RuntimeError("Affinity evolution requires previous-layer affinity.")
                    hidden_states = self._apply_affinity_evolution(hidden_states, prev_affinity, i)
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
                    hidden_states, effective_affinity_config, None, self.num_tokens
                )
            if self.encoder.vis:
                attn_weights.append(weights)
            affinities.append(affinity)
            prev_affinity = affinity

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
            encoded, attn_weights = self.encoder(
                embedding_output,
                None,
                effective_prompt_tokens,
                self._active_semantic_length(semantics),
                self.block_s_to_cls,
            )

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
