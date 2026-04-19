#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
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


class SemanticCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(float(dropout))
        self.proj_dropout = nn.Dropout(float(dropout))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        x = x.view(bsz, seq, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query: torch.Tensor, source: torch.Tensor):
        # query: [B,Q,D], source: [B,K,D]
        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(source))
        v = self._reshape_heads(self.v_proj(source))

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,Q,K]
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, v)  # [B,H,Q,d]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(query.shape[0], query.shape[1], self.hidden_size)

        out = self.out_proj(ctx)
        out = self.proj_dropout(out)
        return out, attn, logits


class LateSemanticSideBranch(nn.Module):
    """
    Lightweight semantic side branch:
      - multi semantic tokens initialized from anchor h_y
      - per-layer interactions with prompt/visual tokens
      - progressively larger update strength from early to late layers
    """

    def __init__(self, hidden_size: int, semantic_branch_cfg, affinity_cfg) -> None:
        super().__init__()
        if semantic_branch_cfg is None:
            raise ValueError("semantic_branch_cfg is required for LateSemanticSideBranch")
        if affinity_cfg is None:
            raise ValueError("affinity_cfg is required for LateSemanticSideBranch")
        self.hidden_size = int(hidden_size)

        sb = semantic_branch_cfg
        af = affinity_cfg

        self.cross_attn_enable = bool(sb.CROSS_ATTN_ENABLE)
        self.cross_attn_num_heads = int(sb.CROSS_ATTN_HEADS)
        self.cross_attn_dropout = float(sb.CROSS_ATTN_DROPOUT)
        self.cross_attn_pre_norm = bool(sb.CROSS_ATTN_PRE_NORM)
        self.cross_attn_use_ffn = bool(sb.CROSS_ATTN_USE_FFN)

        self.delta_gate_sp = float(sb.DELTA_GATE_SP)
        self.delta_gate_sv = float(sb.DELTA_GATE_SV)
        self.delta_gate_ps = float(sb.DELTA_GATE_PS)
        self.delta_gate_vs = float(sb.DELTA_GATE_VS)
        self.cross_attn_compute_all_routes = bool(sb.CROSS_ATTN_COMPUTE_ALL_ROUTES)
        self.delta_gate_open_routes = [str(x).lower() for x in list(sb.DELTA_GATE_OPEN_ROUTES)]
        self.delta_gate_threshold = float(sb.DELTA_GATE_THRESHOLD)
        self.delta_gate_normalize = bool(sb.DELTA_GATE_NORMALIZE)

        invalid_routes = [r for r in self.delta_gate_open_routes if r not in {"sp-att", "sv-att", "ps-att", "vs-att"}]
        if len(invalid_routes) > 0:
            raise ValueError(f"Invalid DELTA_GATE_OPEN_ROUTES entries: {invalid_routes}")
        if self.delta_gate_threshold < 0.0:
            raise ValueError("DELTA_GATE_THRESHOLD must be >= 0")

        self.use_anchor_free = bool(sb.USE_ANCHOR_FREE)
        self.anchor_tokens = int(max(1, sb.ANCHOR_TOKENS))
        self.free_tokens = int(max(0, sb.FREE_TOKENS))
        self.num_tokens = (int(max(1, self.anchor_tokens + self.free_tokens))
            if self.use_anchor_free else int(max(1, sb.NUM_TOKENS)))
        # Competition intensity coefficient of free token against anchor token
        self.free_compete_lambda = float(sb.FREE_COMPETE_LAMBDA)
        # Actual update intensity = gamma * gamma_anchor_scale
        self.gamma_anchor_scale = float(sb.GAMMA_ANCHOR_SCALE)
        # Actual update intensity = gamma * gamma_free_scale
        self.gamma_free_scale = float(sb.GAMMA_FREE_SCALE)
        self.gamma_min = float(sb.GAMMA_MIN)
        self.gamma_max = float(sb.GAMMA_MAX)
        self.start_layer = int(sb.START_LAYER)
        self.end_layer = int(sb.END_LAYER)

        self.anchor_proj: Optional[nn.Linear] = None

        self.token_init = nn.Linear(hidden_size, self.num_tokens * hidden_size)
        self.anchor_token_init = nn.Linear(hidden_size, self.anchor_tokens * hidden_size) if self.use_anchor_free else None
        self.free_token_init = nn.Linear(hidden_size, max(1, self.free_tokens) * hidden_size) if (self.use_anchor_free and self.free_tokens > 0) else None

        # Identity bias
        self.anchor_slot_embed = nn.Parameter(torch.zeros(1, self.anchor_tokens, hidden_size)) if self.use_anchor_free else None
        self.free_slot_embed = nn.Parameter(torch.zeros(1, self.free_tokens, hidden_size)) if (self.use_anchor_free and self.free_tokens > 0) else None

        # Semantic update in normal mode
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Normalization
        self.delta_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.sem_prompt_q_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.sem_prompt_kv_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.sem_visual_q_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.sem_visual_kv_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # cross-attention
        self.sem_from_prompt_attn = SemanticCrossAttention(
            hidden_size=hidden_size,
            num_heads=self.cross_attn_num_heads,
            dropout=self.cross_attn_dropout,
        )
        self.prompt_from_sem_attn = SemanticCrossAttention(
            hidden_size=hidden_size,
            num_heads=self.cross_attn_num_heads,
            dropout=self.cross_attn_dropout,
        )
        self.sem_from_visual_attn = SemanticCrossAttention(
            hidden_size=hidden_size,
            num_heads=self.cross_attn_num_heads,
            dropout=self.cross_attn_dropout,
        )
        self.visual_from_sem_attn = SemanticCrossAttention(
            hidden_size=hidden_size,
            num_heads=self.cross_attn_num_heads,
            dropout=self.cross_attn_dropout,
        )

        # semantic token's own FFN (optional)
        self.sem_ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.sem_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(self.cross_attn_dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(self.cross_attn_dropout),
        )

        self.readout_token_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.readout_gate = nn.Linear(hidden_size, 1)
        self.readout_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self._last_readout_alpha: Optional[torch.Tensor] = None

        self.debug_shapes = False
        self._shape_debug_step_logged = False
        self._shape_debug_readout_logged = False
        self._shape_debug_init_logged = False

        if self.anchor_slot_embed is not None:
            nn.init.normal_(self.anchor_slot_embed, mean=0.0, std=0.02)
        if self.free_slot_embed is not None:
            nn.init.normal_(self.free_slot_embed, mean=0.0, std=0.02)

    def _ensure_anchor_proj(self, semantic_dim: int, device: torch.device):
        if self.anchor_proj is None:
            self.anchor_proj = nn.Linear(int(semantic_dim), self.hidden_size).to(device)

    def _gamma(self, layer_idx: int, num_layers: int) -> float:
        """Computes the layer-dependent update strength for semantic token evolution,
        increasing from early to late layers within the configured  range"""
        if layer_idx < self.start_layer:
            return 0.0
        end_layer = self.end_layer if self.end_layer >= 0 else (num_layers - 1)
        if layer_idx > end_layer:
            return 0.0
        span = max(1, end_layer - self.start_layer)
        t = float(layer_idx - self.start_layer) / float(span)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t

    def init_state( self, semantics: torch.Tensor, device: torch.device, visual_stats: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:

        if semantics.dim() == 3 and semantics.shape[1] == 1:
            semantics = semantics[:, 0, :]
        if semantics.dim() != 2:
            raise ValueError(f"LateSemanticSideBranch expects [B, S] or [B,1,S], got {tuple(semantics.shape)}")

        self._ensure_anchor_proj(semantics.shape[-1], device=device)
        h_y = self.anchor_proj(semantics.to(device))

        if self.use_anchor_free:
            anchor_tokens = self.anchor_token_init(h_y).view(h_y.shape[0], self.anchor_tokens, self.hidden_size)
            if self.anchor_slot_embed is not None:
                anchor_tokens = anchor_tokens + self.anchor_slot_embed.expand(h_y.shape[0], -1, -1)

            if self.free_tokens > 0:
                if torch.is_tensor(visual_stats):
                    v_ctx = visual_stats.to(device)
                    if v_ctx.dim() == 3:
                        v_ctx = v_ctx.mean(dim=1)
                    if v_ctx.dim() != 2:
                        raise ValueError(f"visual_stats should be [B,D] or [B,L,D], got {tuple(v_ctx.shape)}")
                    if v_ctx.shape[0] != h_y.shape[0]:
                        raise ValueError(
                            f"visual_stats batch {v_ctx.shape[0]} incompatible with semantics batch {h_y.shape[0]}"
                        )
                    if v_ctx.shape[-1] != self.hidden_size:
                        raise ValueError(
                            f"visual_stats dim {v_ctx.shape[-1]} incompatible with hidden_size {self.hidden_size}"
                        )
                else:
                    raise ValueError(
                        "visual_stats is required when FREE_TOKENS > 0. "
                        "Expected Tensor [B,D] or [B,L,D], got None/non-tensor."
                    )

                free_tokens = self.free_token_init(v_ctx).view(h_y.shape[0], self.free_tokens, self.hidden_size)

                if self.free_slot_embed is not None:
                    free_tokens = free_tokens + self.free_slot_embed.expand(h_y.shape[0], -1, -1)

                sem_tokens = torch.cat([anchor_tokens, free_tokens], dim=1)
            else:
                sem_tokens = anchor_tokens
        else:
            sem_tokens = self.token_init(h_y).view(h_y.shape[0], self.num_tokens, self.hidden_size)

        sem_tokens = self.delta_norm(sem_tokens)

        if self.debug_shapes and (not self._shape_debug_init_logged):
            print(
                "[SHAPE-DEBUG] LateSemanticSideBranch.init_state semantics={} h_y={} sem_tokens={}".format(
                    tuple(semantics.shape),
                    tuple(h_y.shape),
                    tuple(sem_tokens.shape),
                )
            )
            self._shape_debug_init_logged = True
        return sem_tokens, h_y

    def step(self, sem_tokens: torch.Tensor, prompt_tokens: torch.Tensor, visual_tokens: torch.Tensor,
        layer_idx: int, num_layers: int,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        """Update semantic tokens at each layer"""
        gamma = float(self._gamma(layer_idx, num_layers))

        if self.cross_attn_enable:
            # Scores before attention softmax
            asp_logits = None
            asv_logits = None
            aps_logits = None
            avs_logits = None
            # Multi-head raw attention weights, usually shaped [B,H,Q,K]
            asp_raw = None
            asv_raw = None
            aps_raw = None
            avs_raw = None
            # Initialize to all zeros
            ctx_p = sem_tokens.new_zeros(sem_tokens.shape)
            ctx_v = sem_tokens.new_zeros(sem_tokens.shape)

            open_routes = set(self.delta_gate_open_routes)
            compute_all = bool(self.cross_attn_compute_all_routes)
            need_sp = compute_all or ("sp-att" in open_routes)
            need_ps = compute_all or ("ps-att" in open_routes)
            need_sv = compute_all or ("sv-att" in open_routes)
            need_vs = compute_all or ("vs-att" in open_routes)
            need_prompt = need_sp or need_ps
            need_visual = need_sv or need_vs

            if need_prompt:
                if (not torch.is_tensor(prompt_tokens)) or prompt_tokens.numel() == 0:
                    raise ValueError(
                        "prompt_tokens is required for requested prompt/semantic routes "
                        f"(open_routes={self.delta_gate_open_routes}), but got None/empty prompt_tokens."
                    )
                # Sem <- Prompt
                # ctx_p:   [B, M, D]
                # asp_raw: [B, H, M, P]
                if need_sp:
                    sem_q = self.sem_prompt_q_norm(sem_tokens) if self.cross_attn_pre_norm else sem_tokens
                    p_kv = self.sem_prompt_kv_norm(prompt_tokens) if self.cross_attn_pre_norm else prompt_tokens
                    ctx_p, asp_raw, asp_logits = self.sem_from_prompt_attn(sem_q, p_kv)
                # Prompt <- Sem
                if need_ps:
                    p_q = self.sem_prompt_kv_norm(prompt_tokens) if self.cross_attn_pre_norm else prompt_tokens
                    s_kv = self.sem_prompt_q_norm(sem_tokens) if self.cross_attn_pre_norm else sem_tokens
                    _, aps_raw, aps_logits = self.prompt_from_sem_attn(p_q, s_kv)

            if need_visual:
                if (not torch.is_tensor(visual_tokens)) or visual_tokens.numel() == 0:
                    raise ValueError(
                        "visual_tokens is required for requested visual/semantic routes "
                        f"(open_routes={self.delta_gate_open_routes}), but got None/empty visual_tokens."
                    )
                if need_sv:
                    sem_q = self.sem_visual_q_norm(sem_tokens) if self.cross_attn_pre_norm else sem_tokens
                    v_kv = self.sem_visual_kv_norm(visual_tokens) if self.cross_attn_pre_norm else visual_tokens
                    # ctx_v: [B, M, D], asv_raw: [B, H, M, N]
                    ctx_v, asv_raw, asv_logits = self.sem_from_visual_attn(sem_q, v_kv)
                if need_vs:
                    v_q = self.sem_visual_kv_norm(visual_tokens) if self.cross_attn_pre_norm else visual_tokens
                    s_kv = self.sem_visual_q_norm(sem_tokens) if self.cross_attn_pre_norm else sem_tokens
                    _, avs_raw, avs_logits = self.visual_from_sem_attn(v_q, s_kv)

            # Forward directions (query is semantic)
            ctx_sp = ctx_p  # Sem <- Prompt
            ctx_sv = ctx_v  # Sem <- Visual

            # Reverse directions projected back to semantic-token space
            aps_mean = aps_raw.mean(dim=1) if torch.is_tensor(aps_raw) and aps_raw.dim() == 4 else aps_raw  # [B,P,M]
            avs_mean = avs_raw.mean(dim=1) if torch.is_tensor(avs_raw) and avs_raw.dim() == 4 else avs_raw  # [B,N,M]

            if torch.is_tensor(prompt_tokens) and prompt_tokens.numel() > 0 and torch.is_tensor(aps_mean) and aps_mean.numel() > 0:
                w_ps = aps_mean.transpose(1, 2).contiguous()  # [B,M,P]
                w_ps = w_ps / w_ps.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                ctx_ps = torch.matmul(w_ps, prompt_tokens)     # [B,M,D]
            else:
                ctx_ps = sem_tokens.new_zeros(sem_tokens.shape)
            if torch.is_tensor(visual_tokens) and visual_tokens.numel() > 0 and torch.is_tensor(avs_mean) and avs_mean.numel() > 0:
                w_vs = avs_mean.transpose(1, 2).contiguous()   # [B,M,N]
                w_vs = w_vs / w_vs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                ctx_vs = torch.matmul(w_vs, visual_tokens)      # [B,M,D]
            else:
                ctx_vs = sem_tokens.new_zeros(sem_tokens.shape)

            # Four-channel gated fusion
            gate_raw = sem_tokens.new_tensor( [self.delta_gate_sp, self.delta_gate_sv, self.delta_gate_ps, self.delta_gate_vs] )
            route_mask = sem_tokens.new_zeros((4,))
            route_to_idx = {"sp-att": 0, "sv-att": 1, "ps-att": 2, "vs-att": 3}
            for r in self.delta_gate_open_routes:
                route_mask[route_to_idx[r]] = 1.0
            thr_mask = (gate_raw >= float(self.delta_gate_threshold)).float()
            gate = gate_raw * route_mask * thr_mask
            if float(gate.sum().item()) <= 0.0:
                raise RuntimeError(
                    "No active delta routes after DELTA_GATE_OPEN_ROUTES + DELTA_GATE_THRESHOLD filtering. "
                    f"gate_raw={gate_raw.tolist()}, open_routes={self.delta_gate_open_routes}, "
                    f"threshold={self.delta_gate_threshold}"
                )
            if self.delta_gate_normalize:
                gate = gate / gate.sum().clamp_min(1e-12)
            delta = (
                gate[0] * ctx_sp
                + gate[1] * ctx_sv
                + gate[2] * ctx_ps
                + gate[3] * ctx_vs
            )

            # anchor / free token update
            if self.use_anchor_free and self.anchor_tokens > 0:
                a = sem_tokens[:, :self.anchor_tokens, :]
                d_a = delta[:, :self.anchor_tokens, :]
                a_next = a + (gamma * self.gamma_anchor_scale) * d_a
                if self.free_tokens > 0:
                    f = sem_tokens[:, self.anchor_tokens:, :]
                    d_f = delta[:, self.anchor_tokens:, :]
                    f_next = f + (gamma * self.gamma_free_scale) * d_f
                    sem_next = torch.cat([a_next, f_next], dim=1)
                else:
                    sem_next = a_next
            else:
                sem_next = sem_tokens + gamma * delta

            if self.cross_attn_use_ffn:
                ffn_in = self.sem_ffn_norm(sem_next) if self.cross_attn_pre_norm else sem_next
                sem_next = sem_next + gamma * self.sem_ffn(ffn_in)
            sem_next = self.delta_norm(sem_next)

            # output attention map for monitoring
            asp = asp_raw.mean(dim=1, keepdim=True) if torch.is_tensor(asp_raw) and asp_raw.dim() == 4 else asp_raw
            asv = asv_raw.mean(dim=1, keepdim=True) if torch.is_tensor(asv_raw) and asv_raw.dim() == 4 else asv_raw
            aps = aps_raw.mean(dim=1, keepdim=True) if torch.is_tensor(aps_raw) and aps_raw.dim() == 4 else aps_raw
            avs = avs_raw.mean(dim=1, keepdim=True) if torch.is_tensor(avs_raw) and avs_raw.dim() == 4 else avs_raw
            out_aff = {
                # New canonical direction (Q=Sem): [B,1,M,P], [B,1,M,N]
                "Asp": asp,
                "Asv": asv,
                "Asp_raw": asp_raw,
                "Asv_raw": asv_raw,
                # True reverse direction (Q=Prompt/Visual)
                "Aps": aps,
                "Avs": avs,
                "Aps_raw": aps_raw,
                "Avs_raw": avs_raw,
            }
            diag = {
                "gamma": gamma,
                "asp_energy": float(asp.float().abs().mean().item()) if torch.is_tensor(asp) and asp.numel() > 0 else 0.0,
                "asv_energy": float(asv.float().abs().mean().item()) if torch.is_tensor(asv) and asv.numel() > 0 else 0.0,
                "aps_energy": float(aps.float().abs().mean().item()) if torch.is_tensor(aps) and aps.numel() > 0 else 0.0,
                "avs_energy": float(avs.float().abs().mean().item()) if torch.is_tensor(avs) and avs.numel() > 0 else 0.0,
                "delta_gate_sp": float(gate[0].detach().item()),
                "delta_gate_sv": float(gate[1].detach().item()),
                "delta_gate_ps": float(gate[2].detach().item()),
                "delta_gate_vs": float(gate[3].detach().item()),
                "delta_gate_threshold": float(self.delta_gate_threshold),
                "delta_gate_normalize": bool(self.delta_gate_normalize),
                "delta_gate_open_routes": list(self.delta_gate_open_routes),
                "cross_attn_compute_all_routes": bool(self.cross_attn_compute_all_routes),
                "cross_attn_enable": True,
            }
            if self.debug_shapes and (not self._shape_debug_step_logged):
                print(
                    "[SHAPE-DEBUG] LateSemanticSideBranch.step sem_tokens={} prompt_tokens={} visual_tokens={} "
                    "asp_logits={} asv_logits={} aps_logits={} avs_logits={} "
                    "Asp_raw={} Asv_raw={} Aps_raw={} Avs_raw={} "
                    "ctx_sp={} ctx_sv={} ctx_ps={} ctx_vs={} gate={} sem_next={}".format(
                        tuple(sem_tokens.shape),
                        tuple(prompt_tokens.shape) if torch.is_tensor(prompt_tokens) else None,
                        tuple(visual_tokens.shape) if torch.is_tensor(visual_tokens) else None,
                        tuple(asp_logits.shape) if torch.is_tensor(asp_logits) else None,
                        tuple(asv_logits.shape) if torch.is_tensor(asv_logits) else None,
                        tuple(aps_logits.shape) if torch.is_tensor(aps_logits) else None,
                        tuple(avs_logits.shape) if torch.is_tensor(avs_logits) else None,
                        tuple(asp_raw.shape),
                        tuple(asv_raw.shape),
                        tuple(aps_raw.shape),
                        tuple(avs_raw.shape),
                        tuple(ctx_sp.shape),
                        tuple(ctx_sv.shape),
                        tuple(ctx_ps.shape),
                        tuple(ctx_vs.shape),
                        tuple(gate.shape),
                        tuple(sem_next.shape),
                    )
                )
                self._shape_debug_step_logged = True
            return sem_next, out_aff, diag
        raise RuntimeError(
            "LateSemanticSideBranch requires MODEL.SEMANTIC_BRANCH.CROSS_ATTN_ENABLE=True; "
            "the legacy normalize/einsum semantic update path was removed."
        )

    def readout(self, sem_tokens: torch.Tensor) -> torch.Tensor:
        """Convert the token into a vector"""
        # sem_tokens: [B, M, D]
        read_tokens = sem_tokens[:, :self.anchor_tokens, :] if self.use_anchor_free else sem_tokens
        tokens_n = self.readout_token_norm(read_tokens)
        # score: [B, M, 1] -> [B, M]
        score = self.readout_gate(tokens_n).squeeze(-1)
        alpha = torch.softmax(score, dim=1)

        self._last_readout_alpha = alpha.detach()
        # weighted sum over token dimension
        mu = torch.sum(alpha.unsqueeze(-1) * read_tokens, dim=1)
        out = self.readout_norm(mu)
        if self.debug_shapes and (not self._shape_debug_readout_logged):
            print(
                "[SHAPE-DEBUG] LateSemanticSideBranch.readout sem_tokens={} score={} alpha={} mu={} out={}".format(
                    tuple(sem_tokens.shape),
                    tuple(score.shape),
                    tuple(alpha.shape),
                    tuple(mu.shape),
                    tuple(out.shape),
                )
            )
            self._shape_debug_readout_logged = True
        return out

class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis, prompt_init=None, prompt_init_provider=None):

        self.semantic_branch_cfg = prompt_config.SEMANTIC_BRANCH
        self.semantic_branch_enable = bool(self.semantic_branch_cfg.ENABLE)
        self.prompt_enable = bool(prompt_config.ENABLE)

        super().__init__(config, img_size, vis)

        # 淇濆瓨 prompt 閰嶇疆鍜?vit 閰嶇疆
        self.prompt_config = prompt_config
        self.vit_config = config
        self._monitor_last_raw_semantics = None
        self._monitor_last_refined_semantics = None
        self._last_visual_stats = None
        self._last_semantic_side_state = None
        self._last_prompt_role_stats = None

        if self.semantic_branch_enable:
            affinity_cfg = prompt_config.AFFINITY
            self.semantic_side_branch = LateSemanticSideBranch(
                hidden_size=int(config.hidden_size),
                semantic_branch_cfg=self.semantic_branch_cfg,
                affinity_cfg=affinity_cfg,
            )
        else:
            self.semantic_side_branch = None

        # Unify image and patch size formats
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        # prompt token number
        num_tokens = self.prompt_config.NUM_TOKENS if self.prompt_enable else 0
        self.num_tokens = num_tokens  # number of prompted tokens
        # prompt dropout
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        self.prompt_init_provider = prompt_init_provider

        # Runtime Configuration
        self.debug_shapes = bool(self.prompt_config.DEBUG_SHAPES)
        self._shape_debug_incorporate_logged = False
        self._last_prompt_path_info = {}

        if self.prompt_enable and self.prompt_init_provider is None:
            raise ValueError("Prompt static embeddings have been removed. "
                "Please enable and provide MODEL.PROMPT.DISTRIBUTOR/prompt_init_provider.")
        if prompt_init is not None:
            raise ValueError("Static prompt initialization has been removed; prompt_init must be None.")

        # Synchronize the debug switch to each layer of the semantic branch and encoder
        if self.semantic_side_branch is not None:
            self.semantic_side_branch.debug_shapes = bool(self.debug_shapes)
        for layer_block in self.encoder.layer:
            setattr(layer_block, "debug_shapes", bool(self.debug_shapes))
            if hasattr(layer_block, "attn"):
                setattr(layer_block.attn, "debug_shapes", bool(self.debug_shapes))

        # Layer-wise prompt evolution
        num_layers = config.transformer["num_layers"]
        hidden_size = config.hidden_size
        self.prompt_update_layers = nn.ModuleList([Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])

        evolve_mode = str(self.prompt_config.EVOLVE_INIT_MODE).lower()
        if evolve_mode != "identity":
            raise ValueError(f"Unsupported PROMPT.EVOLVE_INIT_MODE: {self.prompt_config.EVOLVE_INIT_MODE}")
        self.evolve_init_mode = evolve_mode

        # Layer-wise prompt evolution init: identity.
        with torch.no_grad():
            for layer in self.prompt_update_layers:
                eye = torch.eye(hidden_size, device=layer.weight.device, dtype=layer.weight.dtype)
                layer.weight.copy_(eye)
                if layer.bias is not None:
                    layer.bias.zero_()
    def incorporate_prompt(self, x, semantics=None):

        B = x.shape[0]
        self._last_semantic_side_state = None
        self._last_visual_stats = None

        # extract vision patch
        patch_tokens = self.embeddings.forward_patches(x)  # (B, n_patches, hidden_dim)

        # New mainline: semantic update is done by lightweight side-branch per layer.
        refined_semantics = semantics
        self._monitor_last_raw_semantics = semantics.detach() if torch.is_tensor(semantics) else None
        self._monitor_last_refined_semantics = (
            refined_semantics.detach() if torch.is_tensor(refined_semantics) else None
        )

        x_base = self.embeddings.add_cls_and_pos(patch_tokens)  # (B, 1 + n_patches, hidden_dim)
        if self.prompt_enable:
            provider_out = self.prompt_init_provider(patch_tokens)
            if (not isinstance(provider_out, tuple)) or len(provider_out) != 2:
                raise TypeError("prompt_init_provider must return exactly (prompt_tokens, provider_stats).")
            prompt_tokens, provider_stats = provider_out
            expected_shape = (B, self.num_tokens, self.vit_config.hidden_size)
            got_shape = tuple(prompt_tokens.shape) if torch.is_tensor(prompt_tokens) else None
            if (not torch.is_tensor(prompt_tokens)) or tuple(prompt_tokens.shape) != expected_shape:
                raise ValueError(f"prompt_init_provider must return prompt_tokens with shape {expected_shape}, got {type(prompt_tokens)} {got_shape}")
            h_v = provider_stats["h_v"]
            self._last_visual_stats = h_v
            x = torch.cat((
                    x_base[:, :1, :],
                    self.prompt_dropout(prompt_tokens),
                    x_base[:, 1:, :]
                ), dim=1)
        else:
            prompt_tokens = x_base[:, :0, :]
            self._last_visual_stats = patch_tokens.mean(dim=1)
            x = x_base

        if self.debug_shapes and (not self._shape_debug_incorporate_logged):
            print(
                "[SHAPE-DEBUG] PromptedTransformer.incorporate_prompt patch_tokens={} prompt_tokens={} x_base={} x={} semantics={}".format(
                    tuple(patch_tokens.shape),
                    tuple(prompt_tokens.shape) if torch.is_tensor(prompt_tokens) else None,
                    tuple(x_base.shape),
                    tuple(x.shape),
                    tuple(semantics.shape) if torch.is_tensor(semantics) else None,
                )
            )
            self._shape_debug_incorporate_logged = True

        self._last_prompt_path_info = {
            "prompt_enable": bool(self.prompt_enable),
            "semantic_branch_enable": bool(self.semantic_branch_enable),
            "actual_token_shape_entering_backbone": tuple(x.shape),
            "visual_feature_norm": float(patch_tokens.float().norm(dim=-1).mean().item()),
        }

        return x, refined_semantics

    def train(self, mode=True):
        """
        set train status for this class: disable all but the prompt-related modules
        """
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_dropout.train()
            self.prompt_update_layers.train(mode)

            if isinstance(self.prompt_init_provider, torch.nn.Module):
                self.prompt_init_provider.train(mode)
        else:
            for module in self.children():
                module.train(mode)

            if isinstance(self.prompt_init_provider, torch.nn.Module):
                self.prompt_init_provider.train(mode)

    def _init_semantic_side_state(self, semantics: Optional[torch.Tensor], num_layers: int):
        self._last_semantic_side_state = None
        self._last_prompt_role_stats = None
        if (not self.semantic_branch_enable) or (self.semantic_side_branch is None) or (not torch.is_tensor(semantics)):
            return None, None
        sem_tokens, h_y = self.semantic_side_branch.init_state(
            semantics=semantics,
            device=semantics.device,
            visual_stats=self._last_visual_stats,
        )
        self._last_prompt_role_stats = {
            "semantic_branch_enable": True,
            "semantic_branch_num_tokens": int(self.semantic_side_branch.num_tokens),
            "semantic_branch_use_anchor_free": bool(self.semantic_side_branch.use_anchor_free),
            "semantic_branch_anchor_tokens": int(self.semantic_side_branch.anchor_tokens),
            "semantic_branch_free_tokens": int(self.semantic_side_branch.free_tokens),
            "semantic_branch_start_layer": int(self.semantic_side_branch.start_layer),
            "semantic_branch_end_layer": int(self.semantic_side_branch.end_layer),
            "semantic_branch_num_layers": int(num_layers),
            "semantic_branch_layer_stats": {},
        }
        return sem_tokens, h_y

    def _update_semantic_side_branch(self, sem_tokens, hidden_states, layer_idx: int, num_layers: int):
        """
        Cut prompt/visual tokens from the main sequence and feed them to the semantic branch for one step of update
        """
        if sem_tokens is None:
            return None, {}, None
        p_start = 1
        p_end = 1 + int(self.num_tokens)
        prompt_tokens = hidden_states[:, p_start:p_end, :] if self.num_tokens > 0 else hidden_states[:, :0, :]
        visual_tokens = hidden_states[:, p_end:, :]
        sem_tokens, sem_aff, diag = self.semantic_side_branch.step(
            sem_tokens=sem_tokens,
            prompt_tokens=prompt_tokens,
            visual_tokens=visual_tokens,
            layer_idx=layer_idx,
            num_layers=num_layers,
        )
        if isinstance(self._last_prompt_role_stats, dict):
            self._last_prompt_role_stats["semantic_branch_layer_stats"][int(layer_idx)] = {
                "gamma": float(diag.get("gamma", 0.0)),
                "asp_energy": float(diag.get("asp_energy", 0.0)),
                "asv_energy": float(diag.get("asv_energy", 0.0)),
            }
        return sem_tokens, sem_aff, diag

    def _split_anchor_free_tokens(self, sem_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sb = self.semantic_side_branch
        if sb is None:
            raise ValueError("semantic_side_branch is None")
        if sb.use_anchor_free:
            return sem_tokens[:, :sb.anchor_tokens, :], sem_tokens[:, sb.anchor_tokens:, :]
        return sem_tokens, sem_tokens[:, :0, :]

    def forward_deep_prompt(self, embedding_output, semantics=None):
        """
        - Layer 0: Directly use the [CLS|P|PATCH] generated by incorporate_prompt
        - Layers 1 to L-1: First perform layer-wise evolution on the prompt, then replace it back into the main sequence
        - After each layer, let the semantic branch read the current prompt / visual token and update
        """
        attn_weights: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_tokens, h_y = self._init_semantic_side_state(semantics, num_layers)
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
                evolved_prompt = self.prompt_update_layers[i - 1](prev_prompt)
                evolved_prompt = self.prompt_dropout(evolved_prompt)
                if torch.is_tensor(evolved_prompt):
                    row_evolved_ok = torch.isfinite(evolved_prompt).flatten(1).all(dim=1)
                    if not bool(row_evolved_ok.all().item()):
                        bad_evolved = (~row_evolved_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                        print(f"[nan-locate] layer={i} bad_evolved_prompt_rows={bad_evolved}")

                hidden_states = torch.cat(
                    (
                        hidden_states[:, :1, :],  # CLS
                        evolved_prompt,  # prompt
                        hidden_states[:, 1 + self.num_tokens:, :],  # PATCH
                    ),
                    dim=1,)

                hidden_states, weights, _ = self.encoder.layer[i](hidden_states, None, self.num_tokens)
                if torch.is_tensor(hidden_states):
                    row_hidden_ok = torch.isfinite(hidden_states).flatten(1).all(dim=1)
                    if not bool(row_hidden_ok.all().item()):
                        bad_hidden = (~row_hidden_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                        print(f"[nan-locate] layer={i} bad_hidden_rows={bad_hidden}")
            # update sem side
            sem_tokens, sem_aff, _ = self._update_semantic_side_branch(
                sem_tokens=sem_tokens,
                hidden_states=hidden_states,
                layer_idx=i,
                num_layers=num_layers,
            )
            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        if sem_tokens is not None and h_y is not None:
            mu_s_final = self.semantic_side_branch.readout(sem_tokens)
            delta_sem = mu_s_final - h_y
            anchor_tokens, free_tokens = self._split_anchor_free_tokens(sem_tokens)
            self._last_semantic_side_state = {
                "h_y": h_y,
                "sem_tokens": sem_tokens,
                "mu_s_final": mu_s_final,
                "delta_sem": delta_sem,
                "anchor_tokens": anchor_tokens,
                "free_tokens": free_tokens,
            }
            self._monitor_last_refined_semantics = mu_s_final.detach()
        return encoded, attn_weights

    def forward_deep_prompt_with_affinity(self, embedding_output, affinity_config, semantics=None):
        attn_weights: list = []
        affinities: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_tokens, h_y = self._init_semantic_side_state(semantics, num_layers)
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights, affinity, _ = self.encoder.layer[i].forward_with_affinity(
                    hidden_states, affinity_config, None, self.num_tokens
                )
            else:
                prev_prompt = hidden_states[:, 1:1 + self.num_tokens, :]
                evolved_prompt = self.prompt_update_layers[i - 1](prev_prompt)
                evolved_prompt = self.prompt_dropout(evolved_prompt)

                hidden_states = torch.cat(
                    (hidden_states[:, :1, :],evolved_prompt,hidden_states[:, 1 + self.num_tokens:, :],),dim=1,)

                hidden_states, weights, affinity, _ = self.encoder.layer[i].forward_with_affinity(
                    hidden_states, affinity_config, None, self.num_tokens
                )
            sem_tokens, sem_aff, _ = self._update_semantic_side_branch(
                sem_tokens=sem_tokens,
                hidden_states=hidden_states,
                layer_idx=i,
                num_layers=num_layers,
            )
            if isinstance(affinity, dict) and isinstance(sem_aff, dict):
                affinity.update(sem_aff)

            if self.encoder.vis:
                attn_weights.append(weights)
            affinities.append(affinity)

        encoded = self.encoder.encoder_norm(hidden_states)
        if sem_tokens is not None and h_y is not None:
            mu_s_final = self.semantic_side_branch.readout(sem_tokens)
            delta_sem = mu_s_final - h_y
            anchor_tokens, free_tokens = self._split_anchor_free_tokens(sem_tokens)
            self._last_semantic_side_state = {
                "h_y": h_y,
                "sem_tokens": sem_tokens,
                "mu_s_final": mu_s_final,
                "delta_sem": delta_sem,
                "anchor_tokens": anchor_tokens,
                "free_tokens": free_tokens,
            }
            self._monitor_last_refined_semantics = mu_s_final.detach()
        return encoded, attn_weights, affinities
    def forward(self, x, semantics=None):
        """
        standard forward
        """
        embedding_output, semantics = self.incorporate_prompt(x, semantics)

        effective_prompt_tokens = self.num_tokens
        if self.prompt_enable and self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output, semantics)
        else:
            encoded, attn_weights = self.encoder(embedding_output, None, effective_prompt_tokens)
            num_layers = self.vit_config.transformer["num_layers"]
            sem_tokens, h_y = self._init_semantic_side_state(semantics, num_layers)
            if sem_tokens is not None and h_y is not None:
                p_start = 1
                p_end = 1 + int(self.num_tokens)
                prompt_tokens = encoded[:, p_start:p_end, :] if self.num_tokens > 0 else encoded[:, :0, :]
                visual_tokens = encoded[:, p_end:, :]
                sem_tokens, _, _ = self.semantic_side_branch.step(
                    sem_tokens=sem_tokens,
                    prompt_tokens=prompt_tokens,
                    visual_tokens=visual_tokens,
                    layer_idx=num_layers - 1,
                    num_layers=num_layers,
                )
                mu_s_final = self.semantic_side_branch.readout(sem_tokens)
                anchor_tokens, free_tokens = self._split_anchor_free_tokens(sem_tokens)
                self._last_semantic_side_state = {
                    "h_y": h_y,
                    "sem_tokens": sem_tokens,
                    "mu_s_final": mu_s_final,
                    "delta_sem": mu_s_final - h_y,
                    "anchor_tokens": anchor_tokens,
                    "free_tokens": free_tokens,
                }
                self._monitor_last_refined_semantics = mu_s_final.detach()

        return encoded, attn_weights

    def forward_with_affinity(self, x, affinity_config, semantics=None):
        """
        甯︿翰鍜岀煩闃佃緭鍑虹殑鍓嶅悜锛屼笌 forward 骞宠锛?

        - incorporate_prompt 鎻愪緵 [CLS|P|PATCH]
        - 鑻ュ惎鐢?Deep Prompt锛屽垯璋冪敤 forward_deep_prompt_with_affinity
        - 鍚﹀垯璧?encoder.forward_with_affinity
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
            num_layers = self.vit_config.transformer["num_layers"]
            sem_tokens, h_y = self._init_semantic_side_state(semantics, num_layers)
            if sem_tokens is not None and h_y is not None:
                p_start = 1
                p_end = 1 + int(self.num_tokens)
                prompt_tokens = encoded[:, p_start:p_end, :] if self.num_tokens > 0 else encoded[:, :0, :]
                visual_tokens = encoded[:, p_end:, :]
                sem_tokens, sem_aff, _ = self.semantic_side_branch.step(
                    sem_tokens=sem_tokens,
                    prompt_tokens=prompt_tokens,
                    visual_tokens=visual_tokens,
                    layer_idx=num_layers - 1,
                    num_layers=num_layers,
                )
                if isinstance(affinities, list) and len(affinities) > 0 and isinstance(affinities[-1], dict):
                    affinities[-1].update(sem_aff)
                mu_s_final = self.semantic_side_branch.readout(sem_tokens)
                anchor_tokens, free_tokens = self._split_anchor_free_tokens(sem_tokens)
                self._last_semantic_side_state = {
                    "h_y": h_y,
                    "sem_tokens": sem_tokens,
                    "mu_s_final": mu_s_final,
                    "delta_sem": mu_s_final - h_y,
                    "anchor_tokens": anchor_tokens,
                    "free_tokens": free_tokens,
                }
                self._monitor_last_refined_semantics = mu_s_final.detach()

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
