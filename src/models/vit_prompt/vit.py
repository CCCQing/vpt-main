#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
甯︽湁 Prompt 鐨?ViT锛氶伒寰?VPT锛圴isual Prompt Tuning锛夐粯璁よ瀹氱殑骞插噣瀹炵幇銆?
鏍稿績鎬濇兂锛氬湪涓嶏紙鎴栧皯閲忥級鏇存柊涓诲共鍙傛暟鐨勬儏鍐典笅锛屼负杈撳叆搴忓垪鈥滃墠缃€濊嫢骞插彲璁粌鐨勬彁绀?token锛?
涓庡浘鍍?patch 鐨?token 涓€璧烽€佸叆 Transformer锛屼粠鑰屽疄鐜板弬鏁伴珮鏁堢殑杩佺Щ/寰皟銆?
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

# 澶嶇敤鍘熷 ViT 鐨勭粍浠跺拰閰嶇疆锛?
# - CONFIGS: 鍚勪釜妯″瀷绫诲瀷锛堝 ViT-B/16锛夌殑缁撴瀯閰嶇疆瀛楀吀
# - Transformer: 鍘熷鐨?Transformer 缂栫爜鍣紙鍖呭惈 embeddings/encoder 绛夛級
# - VisionTransformer: 甯﹀垎绫诲ご鐨勬爣鍑?ViT
# - np2th: numpy -> torch 鐨勬潈閲嶈浆鎹㈠伐鍏?
from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")

class SharedConceptAligner(nn.Module):
    """
    SharedConceptAligner锛氬叡浜蹇靛熀瀵归綈妯″潡

    浣滅敤锛堝搴斾綘鍐欑殑閭ｉ儴鍒嗗叕寮忥級锛?
    1锛夊皢绫荤骇璇箟 S_raw锛堝睘鎬у悜閲?绫诲師鍨嬶級鎶曞奖鍒颁笌 ViT hidden_size 涓€鑷寸殑绌洪棿锛?
    2锛夊紩鍏?K 涓€滃叡浜蹇垫Ы鈥?R 鈭?R^{K脳D}锛屽璇箟鍜岃瑙夊垎鍒仛娉ㄦ剰鍔涳細
        - 璇箟鈫扲锛氬緱鍒版蹇靛寲璇箟 R_S锛?
        - 瑙嗚鈫扲锛氬緱鍒版蹇靛寲瑙嗚 R_V锛?
    3锛夊啀鐢?R_S 浣滀负 Query锛孯_V 浣滀负 Key/Value 鍋氫竴娆′氦鍙夋敞鎰忓姏锛屽緱鍒颁笌褰撳墠鏍锋湰鐩稿叧鐨?
       瑙嗚琛ュ厖淇℃伅 v_hat锛?
    4锛夐€氳繃 MLP + 娈嬪樊闂ㄦ帶 位 寰楀埌铻嶅悎鍚庣殑璇箟 S^#锛屽悗缁綔涓烘瘡灞?patch鈫抯emantic cross-attention 鐨勮涔夎緭鍏ャ€?
    """

    def __init__(
        self,
        hidden_size: int,      # ViT 鐨?hidden_size锛屽搴?R銆佽涔夈€佽瑙夌殑缁熶竴缁村害 D
        num_slots: int,        # 鍏变韩姒傚康妲戒釜鏁?K
        num_heads: int,        # 澶氬ご娉ㄦ剰鍔?head 鏁帮紝瑕佹眰 hidden_size 鑳芥暣闄?num_heads
        dropout: float = 0.0,  # 娉ㄦ剰鍔?MLP 鐨?dropout
        lambda_init: float = 1.0,  # 位 鐨勫垵濮嬪€硷紙铻嶅悎娈嬪樊鐨勭缉鏀惧洜瀛愶級
        use_layer_norm: bool = True,  # 鏄惁瀵硅緭鍑哄仛 LayerNorm
        proj_norm: bool = True,       # 鏄惁瀵?semantic_proj 涔嬪悗鐨勮涔夊仛 LayerNorm
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # 姣忎釜 head 鐨勭淮搴?d = D / H
        self.head_dim = hidden_size // num_heads
        # 娉ㄦ剰鍔涚缉鏀惧洜瀛?1/sqrt(d)
        self.scale = self.head_dim ** -0.5

        # 鍏变韩姒傚康妲?R锛氬舰鐘?[K, D]锛屽湪鎵€鏈夋牱鏈棿鍏变韩鐨勪竴缁勨€滄蹇靛師鍨嬧€?
        # 鍒濆鍖栨椂鐢?N(0, 1/鈭欴) 杩欐牱鐨勫昂搴︼紝閬垮厤鏁板€艰繃澶?
        self.concept_slots = nn.Parameter(
            torch.randn(num_slots, hidden_size) * (hidden_size ** -0.5)
        )

        # 璇箟鎶曞奖灞傦細灏嗚緭鍏ヨ涔?dim_s 鈫?hidden_size
        # 浣跨敤 lazy 鏋勫缓鐨勬柟寮忥紝鏄洜涓轰笉鍚屾暟鎹泦璇箟缁村害鍙兘涓嶅悓锛圓wA2 85, CUB 312 绛夛級
        self.semantic_proj: Optional[nn.Linear] = None
        # 璇箟鎶曞奖鍚庣殑褰掍竴鍖栵細璁╀笉鍚岀被/鏍锋湰鐨勮涔夊垎甯冩洿绋冲畾
        self.semantic_proj_norm = LayerNorm(hidden_size, eps=1e-6) if proj_norm else nn.Identity()

        # 鈥斺€?绗?1 闃舵锛氳涔?/ 瑙嗚 鈫?妲?R 鐨勬敞鎰忓姏 鈥斺€?#
        # 璇箟渚?Query锛歈_s
        self.query_semantic = Linear(hidden_size, hidden_size)
        # 瑙嗚渚?Query锛歈_v
        self.query_visual = Linear(hidden_size, hidden_size)
        # 妲?R 鐨?Key/Value锛欿_R, V_R锛堝 R 鍋氱嚎鎬у彉鎹級
        self.key_slots = Linear(hidden_size, hidden_size)
        self.value_slots = Linear(hidden_size, hidden_size)

        # 鈥斺€?绗?2 闃舵锛歊_S 鈫?R_V 鐨勪氦鍙夋敞鎰忓姏 鈥斺€?#
        # 杩欓噷浠?R_S 涓?Query锛孯_V 涓?Key/Value
        self.cross_query = Linear(hidden_size, hidden_size)
        self.cross_key = Linear(hidden_size, hidden_size)
        self.cross_value = Linear(hidden_size, hidden_size)

        # 娈嬪樊 MLP锛氳緭鍏ユ槸 [r_s || v_hat]锛堟嫾鎺ワ紝缁村害 2D锛夛紝杈撳嚭 D 缁村閲?螖
        hidden_mlp = hidden_size * 2
        self.delta_mlp = nn.Sequential(
            Linear(hidden_size * 2, hidden_mlp),
            nn.GELU(),
            Dropout(dropout),
            Linear(hidden_mlp, hidden_size),
            Dropout(dropout),
        )
        # U s_1锛氬鍩哄噯璇箟鍋氱嚎鎬ф槧灏勶紝鐢ㄤ簬 螖 = MLP([r_s||v_hat]) - U r_s
        self.skip_proj = Linear(hidden_size, hidden_size)
        # 閫氶亾缁村害鐨?位 闂ㄦ帶鍚戦噺锛氬舰鐘?[D]锛岄€愮淮缂╂斁 螖
        self.lambda_gate = nn.Parameter(torch.full((hidden_size,), lambda_init))
        # 杈撳嚭灞傚綊涓€鍖栵細瀵瑰簲 S^# = LN(s_1 + 位 鈭?螖)
        self.out_norm = LayerNorm(hidden_size, eps=1e-6) if use_layer_norm else nn.Identity()

        # 娉ㄦ剰鍔涙潈閲嶇殑 dropout
        self.attn_dropout = Dropout(dropout)

    def _build_semantic_proj(self, semantic_dim: int, device: torch.device):
        """
        Lazy 鏋勯€犺涔夋姇褰卞眰锛?
        - 绗竴娆?forward 鏃讹紝鏍规嵁璇箟缁村害 semantic_dim 鍒涘缓 Linear(semantic_dim, hidden_size)
        - 涔嬪悗澶嶇敤鍚屼竴灞傦紝鍏煎涓嶅悓鏁版嵁闆嗙殑灞炴€х淮搴?
        """
        if self.semantic_proj is None:
            self.semantic_proj = Linear(semantic_dim, self.hidden_size)
            self.semantic_proj = self.semantic_proj.to(device)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        灏嗚緭鍏?[B, L, D] reshape 鎴愬澶存敞鎰忓姏鏍煎紡 [B, H, L, d]锛?
        - 鍏?view 鎴?[B, L, H, d]
        - 鍐?permute 鍒?[B, H, L, d]
        """
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        鏍囧噯澶氬ご娉ㄦ剰鍔涜绠楋細
        杈撳叆锛?
          query: [B, H, L_q, d]
          key:   [B, H, L_k, d]
          value: [B, H, L_k, d]
        杈撳嚭锛?
          context: [B, L_q, D]锛屽嵆灏嗗澶寸粨鏋滄嫾鍥?D 缁?
        """
        # [B, H, L_q, L_k]
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        # softmax + dropout 寰楀埌娉ㄦ剰鍔涙潈閲?
        attn_probs = self.attn_dropout(torch.softmax(attn_scores, dim=-1))
        # [B, H, L_q, d]
        context = torch.matmul(attn_probs, value)
        # 杩樺師鍥?[B, L_q, D]锛氬厛鎹㈠洖 [B, L_q, H, d] 鍐?view
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.hidden_size,)
        return context.view(*new_shape)

    def encode_semantics_only(self, semantics: torch.Tensor) -> torch.Tensor:
        """
        浠呯敤浜庘€滅被璇箟 鈫?R 绌洪棿绫诲師鍨嬧€濈殑缂栫爜锛堢粰鍒嗙被澶寸敤锛夈€?
        涓嶄緷璧栧叿浣撳浘鍍忥紝鍙緷璧栧睘鎬у悜閲忋€?
        杈撳叆锛氭墍鏈夌被鐨勫睘鎬х煩闃?semantics锛岀淮搴﹀ぇ姒傛槸 [C, d_s]锛屾瘮濡?CUB 灏辨槸 [200, 312]锛?
        杈撳嚭锛氭墍鏈夌被鍦?R 绌洪棿涓嬬殑琛ㄨ揪锛岀淮搴?[C, D]锛屾瘮濡?ViT-B/16 鏄?[200, 768]

        Args:
            semantics: [C, d_s] 鎴?[C, 1, d_s]锛孋 涓虹被鍒暟

        Returns:
            [C, D]锛屾瘡涓被鍒湪 R 绌洪棿涓嬬殑绫诲師鍨嬪悜閲?s_c
        """
        # 缁熶竴鎴?[C, 1, d_s]
        if semantics.dim() == 1:
            semantics = semantics.unsqueeze(0)
        if semantics.dim() == 2:
            semantics = semantics.unsqueeze(1)

        device = self.concept_slots.device
        semantics = semantics.to(device)
        self._build_semantic_proj(semantics.size(-1), device=device)
        # 灞炴€ф姇褰卞埌 D 缁村苟鍋?LN
        semantic_tokens = self.semantic_proj(semantics)
        semantic_tokens = self.semantic_proj_norm(semantic_tokens)

        num_classes = semantic_tokens.size(0)
        # 妲?R 鎵╁睍鍒?batch 缁达細[C, K, D]
        slots = self.concept_slots.unsqueeze(0).expand(num_classes, -1, -1)
        # 妲界殑 K/V锛歔C, H, K, d]
        slot_keys = self._transpose_for_scores(self.key_slots(slots))
        slot_values = self._transpose_for_scores(self.value_slots(slots))

        # 璇箟 Query锛歔C, H, 1, d]
        sem_query = self._transpose_for_scores(self.query_semantic(semantic_tokens))
        # r_s: [C, 1, D] 鈫?squeeze 鎴?[C, D]
        r_s = self._attention(sem_query, slot_keys, slot_values)

        return r_s.squeeze(1)

    def forward(self, patch_tokens: torch.Tensor, semantics: torch.Tensor) -> torch.Tensor:
        """
        杈撳叆锛?
          patch_tokens: [B, N, D]锛屾潵鑷?embeddings.forward_patches 鐨勮瑙?patch tokens
          semantics:    [B, d_s] / [B, 1, d_s] / [d_s]锛岀被绾у睘鎬ф垨璇箟鍘熷瀷

        杈撳嚭锛?
          fused: [B, D]锛岃瀺鍚堣瑙変俊鎭悗鐨勫叡浜涔?S^#
        """
        # 缁熶竴璇箟鐨勫舰鐘讹細纭繚涓?[B, 1, d_s]
        if semantics.dim() == 1:
            # 鍗曟牱鏈?[d_s] 鈫?[1, d_s]
            semantics = semantics.unsqueeze(0)
        if semantics.dim() == 2:
            # [B, d_s] 鈫?[B, 1, d_s]
            semantics = semantics.unsqueeze(1)  # [B, 1, d_s]

        # 鏋勫缓璇箟鎶曞奖灞傦紙绗竴娆¤皟鐢ㄦ椂锛?
        self._build_semantic_proj(semantics.size(-1), device=patch_tokens.device)
        # 璇箟鎶曞奖鍒?D 缁村苟褰掍竴鍖栵細S_raw 鈫?S虄_raw
        semantic_tokens = self.semantic_proj(semantics)
        semantic_tokens = self.semantic_proj_norm(semantic_tokens)

        # ====== 闃舵 1锛氳涔?瑙嗚 鈫?妲?R 鐨勬敞鎰忓姏锛屽緱鍒?R_S, R_V ======
        # 鎵瑰ぇ灏?B锛岀敤浜庡皢鍏变韩妲?R 鎵╁睍鍒?batch 缁村害.  姝ゅ鍋氱殑鏄瀯閫燫 鐨?batch 瑙嗗浘
        B, _, _ = patch_tokens.shape
        # slots: [B, K, D]锛屾墍鏈夋牱鏈叡浜弬鏁帮紝浣嗗湪 batch 缁村害鍋氫簡 expand
        slots = self.concept_slots.unsqueeze(0).expand(B, -1, -1)  # [batch鏁? R妲芥暟num_slots, hidden_size 768 D]

        # 妲界殑 Key/Value锛欿_R, V_R 褰㈢姸 [B, H, K, d]
        slot_keys = self._transpose_for_scores(self.key_slots(slots))
        slot_values = self._transpose_for_scores(self.value_slots(slots))

        # 鈥斺€?璇箟鈫扲锛歊_S 鈥斺€?#
        # Q_s: [B, H, 1, d]     D = 768锛歏iT hidden size ;H = 8锛氭敞鎰忓姏澶存暟 ;d = D / H = 96锛氭瘡涓?head 鐨勭淮搴?
        sem_query = self._transpose_for_scores(self.query_semantic(semantic_tokens))
        # 璁＄畻r_s: [B, 1, D]锛屾瘡涓牱鏈殑鈥滄蹇靛寲璇箟鈥?
        r_s = self._attention(sem_query, slot_keys, slot_values)  # [B, 1, D]

        # 鈥斺€?瑙嗚鈫扲锛歊_V 鈥斺€?#
        # Q_v: [B, H, N, d]
        vis_query = self._transpose_for_scores(self.query_visual(patch_tokens))
        # 璁＄畻r_v: [B, N, D]锛屾瘡涓?patch 缁?R 閲嶆柊琛ㄨ揪鍚庣殑鈥滄蹇靛寲瑙嗚 token鈥?
        r_v = self._attention(vis_query, slot_keys, slot_values)  # [B, N, D]

        # ====== 闃舵 2锛歊_S 鈫?R_V 浜ゅ弶娉ㄦ剰鍔涳紝寰楀埌瑙嗚琛ュ厖 v_hat ======
        # 浠?r_s 涓?Query锛宺_v 涓?Key/Value
        cross_q = self._transpose_for_scores(self.cross_query(r_s))
        cross_k = self._transpose_for_scores(self.cross_key(r_v))
        cross_v = self._transpose_for_scores(self.cross_value(r_v))
        # v_hat: [B, 1, D]锛岃〃绀衡€滀笌褰撳墠璇箟鐩稿叧鐨勯偅閮ㄥ垎瑙嗚淇℃伅鈥?
        v_hat = self._attention(cross_q, cross_k, cross_v)  # [B, 1, D]

        # ====== 闃舵 3锛氭畫宸瀺鍚堬紝寰楀埌 S^# ======
        # 浠ユ蹇靛寲璇箟 R_S 浣滀负娈嬪樊鍩哄噯锛岃€岄潪鍘熷璇箟 S_raw
        base_semantics = r_s
        # 鎷兼帴 r_s 涓?v_hat锛歔B, 1, 2D] 鈫?[B, 1, D]锛屽啀鍑忓幓 U r_s
        delta = self.delta_mlp(torch.cat([base_semantics, v_hat], dim=-1)) - self.skip_proj(base_semantics)
        # 閫氶亾闂ㄦ帶 位锛氶€愮淮缂╂斁 螖锛屽緱鍒?R_S + 位 鈭?螖锛堜笉鐩存帴鍥炶惤鍒板師濮嬭涔夛級
        fused = base_semantics + self.lambda_gate * delta
        # 鏈€缁堝綊涓€鍖栵紝寰楀埌 S^#锛堝幓鎺夐暱搴?1 缁村害锛岃緭鍑?[B, D]锛?
        fused = self.out_norm(fused)
        return fused.squeeze(1)


class LateSemanticSideBranch(nn.Module):
    """
    Lightweight semantic side branch:
      - multi semantic tokens initialized from anchor h_y
      - per-layer interactions with prompt/visual tokens
      - progressively larger update strength from early to late layers
    """

    def __init__(
        self,
        hidden_size: int,
        num_tokens: int = 4,
        use_anchor_free: bool = False,
        anchor_tokens: int = 8,
        free_tokens: int = 2,
        free_compete_lambda: float = 0.5,
        gamma_anchor_scale: float = 1.0,
        gamma_free_scale: float = 1.0,
        gamma_min: float = 0.05,
        gamma_max: float = 1.0,
        start_layer: int = 0,
        end_layer: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_anchor_free = bool(use_anchor_free)
        self.anchor_tokens = int(max(1, anchor_tokens))
        self.free_tokens = int(max(0, free_tokens))
        self.num_tokens = int(max(1, self.anchor_tokens + self.free_tokens)) if self.use_anchor_free else int(max(1, num_tokens))
        self.free_compete_lambda = float(free_compete_lambda)
        self.gamma_anchor_scale = float(gamma_anchor_scale)
        self.gamma_free_scale = float(gamma_free_scale)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)

        self.anchor_proj: Optional[nn.Linear] = None
        self.token_init = nn.Linear(hidden_size, self.num_tokens * hidden_size)
        self.anchor_token_init = nn.Linear(hidden_size, self.anchor_tokens * hidden_size) if self.use_anchor_free else None
        self.free_token_init = nn.Linear(hidden_size, max(1, self.free_tokens) * hidden_size) if (self.use_anchor_free and self.free_tokens > 0) else None
        self.anchor_slot_embed = nn.Parameter(torch.zeros(1, self.anchor_tokens, hidden_size)) if self.use_anchor_free else None
        self.free_slot_embed = nn.Parameter(torch.zeros(1, self.free_tokens, hidden_size)) if (self.use_anchor_free and self.free_tokens > 0) else None
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.delta_norm = nn.LayerNorm(hidden_size, eps=1e-6)
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
        if layer_idx < self.start_layer:
            return 0.0
        end_layer = self.end_layer if self.end_layer >= 0 else (num_layers - 1)
        if layer_idx > end_layer:
            return 0.0
        span = max(1, end_layer - self.start_layer)
        t = float(layer_idx - self.start_layer) / float(span)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t

    def init_state(self, semantics: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
                free_tokens = self.free_token_init(h_y).view(h_y.shape[0], self.free_tokens, self.hidden_size)
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

    def step(
        self,
        sem_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        gamma = float(self._gamma(layer_idx, num_layers))
        sem_n = torch.nn.functional.normalize(sem_tokens.float(), dim=-1)

        aps = None
        aps_logits = None
        if torch.is_tensor(prompt_tokens) and prompt_tokens.numel() > 0:
            p_n = torch.nn.functional.normalize(prompt_tokens.float(), dim=-1)
            aps_logits = torch.einsum("bpd,bmd->bpm", p_n, sem_n)
            aps = torch.softmax(aps_logits, dim=-1)
            ctx_p = torch.einsum("bpm,bpd->bmd", aps, prompt_tokens)
        else:
            ctx_p = sem_tokens.new_zeros(sem_tokens.shape)

        avs_logits = None
        if torch.is_tensor(visual_tokens) and visual_tokens.numel() > 0:
            v_n = torch.nn.functional.normalize(visual_tokens.float(), dim=-1)
            avs_logits = torch.einsum("bnd,bmd->bnm", v_n, sem_n)
            avs = torch.softmax(avs_logits, dim=-1)
            ctx_v = torch.einsum("bnm,bnd->bmd", avs, visual_tokens)
        else:
            avs = sem_tokens.new_zeros((sem_tokens.shape[0], 0, sem_tokens.shape[1]))
            ctx_v = sem_tokens.new_zeros(sem_tokens.shape)

        free_patch_score = None
        anchor_logits_adj = None
        anchor_attn = None
        free_logits = None
        free_attn = None
        anchor_delta = None
        free_delta = None
        if self.use_anchor_free and torch.is_tensor(visual_tokens) and visual_tokens.numel() > 0 and self.anchor_tokens > 0:
            a = sem_tokens[:, :self.anchor_tokens, :]
            f = sem_tokens[:, self.anchor_tokens:, :] if self.free_tokens > 0 else sem_tokens[:, :0, :]
            a_n = torch.nn.functional.normalize(a.float(), dim=-1)
            v_n = torch.nn.functional.normalize(visual_tokens.float(), dim=-1)
            anchor_logits = torch.einsum("bad,bnd->ban", a_n, v_n)
            if self.free_tokens > 0 and f.numel() > 0:
                f_n = torch.nn.functional.normalize(f.float(), dim=-1)
                free_logits = torch.einsum("bfd,bnd->bfn", f_n, v_n)
                free_patch_score = free_logits.max(dim=1, keepdim=True).values
                anchor_logits_adj = anchor_logits - self.free_compete_lambda * free_patch_score.detach()
                free_attn = torch.softmax(free_logits, dim=-1)
            else:
                anchor_logits_adj = anchor_logits
            anchor_attn = torch.softmax(anchor_logits_adj, dim=-1)
            anchor_delta = torch.matmul(anchor_attn, visual_tokens)
            anchor_next = a + (gamma * self.gamma_anchor_scale) * anchor_delta
            if self.free_tokens > 0 and f.numel() > 0 and free_attn is not None:
                free_delta = torch.matmul(free_attn, visual_tokens)
                free_next = f + (gamma * self.gamma_free_scale) * free_delta
                sem_next = torch.cat([anchor_next, free_next], dim=1)
            else:
                sem_next = anchor_next
            sem_next = self.delta_norm(sem_next)
        else:
            delta = self.delta_mlp(torch.cat([sem_tokens, ctx_p, ctx_v], dim=-1))
            sem_next = self.delta_norm(sem_tokens + gamma * delta)
            anchor_delta = delta

        out_aff = {
            "Aps": aps.unsqueeze(1) if aps is not None else sem_tokens.new_zeros((sem_tokens.shape[0], 1, 0, sem_tokens.shape[1])),
            "Avs": avs.unsqueeze(1),
        }
        diag = {
            "gamma": gamma,
            "aps_energy": float(out_aff["Aps"].float().abs().mean().item()) if out_aff["Aps"].numel() > 0 else 0.0,
            "avs_energy": float(out_aff["Avs"].float().abs().mean().item()) if out_aff["Avs"].numel() > 0 else 0.0,
        }
        if self.debug_shapes and (not self._shape_debug_step_logged):
            print(
                "[SHAPE-DEBUG] LateSemanticSideBranch.step sem_tokens={} prompt_tokens={} visual_tokens={} "
                "aps_logits={} avs_logits={} aps={} avs={} anchor_logits_adj={} free_patch_score={} "
                "anchor_attn={} free_attn={} ctx_p={} ctx_v={} delta={} sem_next={}".format(
                    tuple(sem_tokens.shape),
                    tuple(prompt_tokens.shape) if torch.is_tensor(prompt_tokens) else None,
                    tuple(visual_tokens.shape) if torch.is_tensor(visual_tokens) else None,
                    tuple(aps_logits.shape) if torch.is_tensor(aps_logits) else None,
                    tuple(avs_logits.shape) if torch.is_tensor(avs_logits) else None,
                    tuple(aps.shape) if torch.is_tensor(aps) else None,
                    tuple(avs.shape) if torch.is_tensor(avs) else None,
                    tuple(anchor_logits_adj.shape) if torch.is_tensor(anchor_logits_adj) else None,
                    tuple(free_patch_score.shape) if torch.is_tensor(free_patch_score) else None,
                    tuple(anchor_attn.shape) if torch.is_tensor(anchor_attn) else None,
                    tuple(free_attn.shape) if torch.is_tensor(free_attn) else None,
                    tuple(ctx_p.shape),
                    tuple(ctx_v.shape),
                    tuple(anchor_delta.shape) if torch.is_tensor(anchor_delta) else None,
                    tuple(sem_next.shape),
                )
            )
            self._shape_debug_step_logged = True
        return sem_next, out_aff, diag

    def readout(self, sem_tokens: torch.Tensor) -> torch.Tensor:
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
    """
    鍦ㄥ師濮?Transformer 缂栫爜鍣ㄥ熀纭€涓婂姞鍏モ€滃墠缃?Prompt token鈥濈殑鐗堟湰銆?

    搴忓垪褰㈠紡锛?
        [CLS] + [PROMPT 脳 P] + [PATCH 脳 N]

    闄愬埗鏉′欢锛堝綋鍓嶅疄鐜板彧鏀寔鏈€甯哥敤鐨勪竴绉嶈瀹氾級锛?
    - LOCATION == "prepend"锛氭彁绀?token 鍙兘鎻掑叆鍦?CLS 涔嬪悗銆乸atch tokens 涔嬪墠锛?
    - INITIATION == "random"锛氭彁绀哄悜閲忛噰鐢ㄩ殢鏈哄垵濮嬪寲锛圶avier 绫讳技鐨勫潎鍖€鍒嗗竷锛夛紱
    - 涓嶆敮鎸侊細
        * prompt_config.NUM_DEEP_LAYERS 闈?None 鐨勨€滃彧鍦ㄩ儴鍒嗗眰鎻掑叆 deep prompt鈥濓紱
        * prompt_config.DEEP_SHARED = True 鐨勨€滄墍鏈夊眰鍏辩敤涓€浠?deep prompt鈥濄€?

    鍚屾椂鏀寔锛?
    - 甯歌鍓嶇疆 prompt锛堢 0 灞傝緭鍏ュ墠鎻掑叆锛夛紱
    - Deep Prompt锛氬湪姣忎釜涓棿灞備箣鍓嶉兘鎻掑叆涓€浠借灞備笓灞炵殑 prompt锛屽苟鏇挎崲鎺変笂涓€灞傜殑 prompt 娈点€?
    """

    def __init__(self, prompt_config, config, img_size, vis, prompt_init=None, prompt_init_provider=None):

        assert prompt_config.LOCATION == "prepend"  # 鍙敮鎸佸墠缃彁绀猴紙鎻掑湪 CLS 鍚庛€乸atch 鍓嶏級
        assert prompt_config.INITIATION == "random" # 鎻愮ず鍚戦噺闅忔満鍒濆鍖栵紙Xavier 椋庢牸鍖洪棿锛?

        # 涓嶆敮鎸佷綘鍦?12 灞傞噷鍙寫鈥滅 3銆?銆?1 灞傗€濇彃鍏ョ殑閭ｇ閮ㄥ垎灞?deep-prompt
        # 鏀寔鐨?deep 鐗堟湰鏄細绗?0 灞傜敤鈥滃墠缃?prompt鈥濓紝浠庣 1 灞傚埌绗?L-1 灞傗€滄瘡灞傞兘鐢ㄤ竴浠?deep-prompt鈥?
        assert prompt_config.NUM_DEEP_LAYERS is None

        # 涓嶆敮鎸佹墍鏈夊眰鍏辩敤鍚屼竴缁勬彁绀哄悜閲忥紙鍏变韩鍙傛暟锛?鏀寔鐨勬槸姣忎竴灞傞兘鏈夎嚜宸辩殑鎻愮ず鍙傛暟
        assert not prompt_config.DEEP_SHARED

        # 鍒濆鍖栫埗绫伙紙浼氭瀯寤?embeddings銆乪ncoder 绛夛級
        # semantic_dim锛氳嫢浣犲湪 encoder 灞傞噷鍔犲叆浜嗏€滆涔?cross-attention鈥濓紝鍙互閫氳繃杩欎釜缁村害鍐冲畾璇箟鍚戦噺鐨勬姇褰辩淮搴?
        # 閫氬父绛変簬鏁版嵁闆嗙殑灞炴€х淮搴︼紙濡?312锛夈€?
        semantic_dim = getattr(prompt_config, "SEMANTIC_DIM", None)

        # 璇诲彇鍏变韩姒傚康鍩虹浉鍏抽厤缃紙SEMANTIC_CONCEPT 瀛愯妭鐐癸級
        concept_cfg = getattr(prompt_config, "SEMANTIC_CONCEPT", None)
        self.semantic_branch_cfg = getattr(prompt_config, "SEMANTIC_BRANCH", None)
        self.semantic_branch_enable = bool(
            self.semantic_branch_cfg is not None and getattr(self.semantic_branch_cfg, "ENABLE", True)
        )
        self.semantic_cross_attn_enable = bool(getattr(prompt_config, "SEMANTIC_CROSS_ATTN_ENABLE", False)) and (not self.semantic_branch_enable)
        self.shared_concept_enable = bool(getattr(prompt_config, "SHARED_CONCEPT_ENABLE", False))
        self.shared_aligner_enable = bool(getattr(prompt_config, "SHARED_ALIGNER_ENABLE", False))
        concept_enabled = bool(concept_cfg is not None and getattr(concept_cfg, "ENABLE", False))
        use_shared_semantic_module = bool(
            (not self.semantic_branch_enable)
            and concept_enabled
            and self.shared_concept_enable
            and self.shared_aligner_enable
        )
        if use_shared_semantic_module:
            # 鑻ュ惎鐢ㄥ叡浜蹇靛熀妯″潡锛屽垯寮哄埗 semantic_dim = hidden_size
            # 杩欐牱 encoder 灞傞噷鐨勮涔夊悜閲忓氨鐩存帴鐢?S^#锛堝凡鏄?D 缁达級锛屾棤闇€棰濆鏄犲皠
            semantic_dim = config.hidden_size

        # 璋冪敤鐖剁被 Transformer 鐨勫垵濮嬪寲锛屽苟鍛婄煡 semantic_dim锛堜究浜庡叾鍐呴儴鍒涘缓璇箟鐩稿叧鎶曞奖锛?
        super(PromptedTransformer, self).__init__(
            config,
            img_size,
            vis,
            semantic_dim=semantic_dim if self.semantic_cross_attn_enable else None,
            semantic_cross_attn_enable=self.semantic_cross_attn_enable,
        )

        # 淇濆瓨 prompt 閰嶇疆鍜?vit 閰嶇疆
        self.prompt_config = prompt_config
        self.vit_config = config
        self._monitor_last_raw_semantics = None
        self._monitor_last_refined_semantics = None
        self._last_semantic_side_state = None
        self._last_prompt_role_stats = None

        if self.semantic_branch_enable:
            num_layers = int(config.transformer["num_layers"])
            start_layer = int(getattr(self.semantic_branch_cfg, "START_LAYER", 0))
            end_layer_cfg = int(getattr(self.semantic_branch_cfg, "END_LAYER", -1))
            end_layer = (num_layers - 1) if end_layer_cfg < 0 else end_layer_cfg
            self.semantic_side_branch = LateSemanticSideBranch(
                hidden_size=int(config.hidden_size),
                num_tokens=int(getattr(self.semantic_branch_cfg, "NUM_TOKENS", 4)),
                use_anchor_free=bool(getattr(self.semantic_branch_cfg, "USE_ANCHOR_FREE", False)),
                anchor_tokens=int(getattr(self.semantic_branch_cfg, "ANCHOR_TOKENS", 8)),
                free_tokens=int(getattr(self.semantic_branch_cfg, "FREE_TOKENS", 2)),
                free_compete_lambda=float(getattr(self.semantic_branch_cfg, "FREE_COMPETE_LAMBDA", 0.5)),
                gamma_anchor_scale=float(getattr(self.semantic_branch_cfg, "GAMMA_ANCHOR_SCALE", 1.0)),
                gamma_free_scale=float(getattr(self.semantic_branch_cfg, "GAMMA_FREE_SCALE", 1.0)),
                gamma_min=float(getattr(self.semantic_branch_cfg, "GAMMA_MIN", 0.05)),
                gamma_max=float(getattr(self.semantic_branch_cfg, "GAMMA_MAX", 1.0)),
                start_layer=start_layer,
                end_layer=end_layer,
            )
        else:
            self.semantic_side_branch = None

        # 鑻ュ惎鐢ㄤ簡鍏变韩姒傚康鍩烘ā鍧楋紝鍒欏湪姝ゆ瀯寤?SharedConceptAligner
        self.semantic_concept = None
        if use_shared_semantic_module:
            self.semantic_concept = SharedConceptAligner(
                hidden_size=config.hidden_size,
                num_slots=concept_cfg.NUM_SLOTS,
                num_heads=concept_cfg.NUM_HEADS,
                dropout=concept_cfg.DROPOUT,
                lambda_init=concept_cfg.LAMBDA_INIT,
                use_layer_norm=concept_cfg.USE_LAYER_NORM,
                proj_norm=concept_cfg.PROJ_NORM,
            )

        # 瑙勮寖杈撳叆灏哄 & 鍙栧嚭 patch 澶у皬锛岀粺涓€灏嗗昂瀵歌浆鎴愪簩鍏冪粍锛圚, W锛?
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        # 鎻愮ず token 鏁伴噺锛堜緥濡?5/10/...锛? patch 灏哄涓?(H, W) 褰㈠紡
        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        # 瀵规彁绀?token 鍙€夌殑 dropout锛堣缁冩湡闅忔満涓㈠純锛屽寮洪瞾妫掓€э級
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # ====== 鎻愮ず token 缁村害璁惧畾 ======
        # 鐩存帴鍦?ViT hidden_size 缁村害涓婄淮鎶?鐢熸垚 prompt锛屽悓鏃朵繚鐣欏彲璁粌鐨?prompt 鏄犲皠灞?
        prompt_dim = config.hidden_size
        self.prompt_proj = Linear(prompt_dim, config.hidden_size)
        if prompt_dim == config.hidden_size:
            # 缁村害涓€鑷存椂鍒濆鍖栦负鎺ヨ繎鎭掔瓑鏄犲皠锛屾搴︿富瑕佽惤鍦ㄥ彲璁粌鐨勬槧灏勫眰
            with torch.no_grad():
                self.prompt_proj.weight.copy_(torch.eye(config.hidden_size))
                if self.prompt_proj.bias is not None:
                    self.prompt_proj.bias.zero_()

        # 鍦ㄥ墠鍚戣繃绋嬩腑鏍规嵁瑙嗚 token 鐢熸垚 prompt 鐨勬ā鍧楋紝姣斿浣犺嚜宸辩殑鈥滄彁绀哄垎甯冪綉缁溾€濓紝杈撳叆 patch_tokens锛岃緭鍑?prompt_tokens
        self.prompt_init_provider = prompt_init_provider

        # 鏄惁瀹屽叏渚濊禆鈥滆繍琛屾椂鍒嗗竷鈥濈敓鎴愭彁绀猴紝鑰屼笉浣跨敤浠讳綍鍙涔犵殑 prompt 鍙傛暟
        self.runtime_prompt_only = getattr(
            self.prompt_config, "DISTRIBUTION_ONLY", False)
        self.detach_prompt_grad = getattr(
            self.prompt_config, "DETACH_PROMPT_GRAD", False)
        self.debug_prompt_flow = getattr(self.prompt_config, "DEBUG_FLOW", False)
        self.debug_shapes = bool(getattr(self.prompt_config, "DEBUG_SHAPES", False))
        self.noop_keep_params = bool(getattr(self.prompt_config, "NOOP_KEEP_PARAMS", False))
        self._debug_prompt_flow_logged = False
        self._shape_debug_incorporate_logged = False
        self._last_prompt_noop_info = {}

        if self.runtime_prompt_only:
            # 鏃㈢劧璇粹€滃彧闈犲垎甯?鐢熸垚鍣ㄢ€濓紝閭ｅ氨蹇呴』鎻愪緵涓€涓?prompt_init_provider
            if self.prompt_init_provider is None:
                raise ValueError(
                    "PROMPT.DISTRIBUTION_ONLY=True requires a prompt_init_provider"
                )
            # 骞朵笖涓嶅厑璁稿悓鏃剁粰涓€涓潤鎬佸垵濮?prompt锛堜袱鑰呯煕鐩撅級
            if prompt_init is not None:
                raise ValueError(
                    "prompt_init cannot be provided when DISTRIBUTION_ONLY=True"
                )
        self.use_learned_prompt_params = not self.runtime_prompt_only

        # ====== 鍒濆鍖栨彁绀?token 鍙傛暟 ======
        if self.prompt_config.INITIATION == "random":
            # Xavier-uniform 椋庢牸鐨勪笂涓嬬晫锛屾牴鎹緭鍏ョ淮搴︿笌 prompt_dim 璁＄畻 val = sqrt( 6 / ( fan_in + fan_out ) )
            # 鐩殑锛氳闅忔満鍒濆鍖栫殑鎻愮ず鍚戦噺鍜?patch 宓屽叆鐨勯噺绾茬浉杩戯紝渚夸簬涓よ€呭湪鍚屼竴搴忓垪閲岃娉ㄦ剰鍔涚綉缁滀竴璧峰鐞?
            # fan_in 杩戜技涓猴細姣忎釜 patch 鐨勫師濮嬭緭鍏ョ淮搴?= 3 * patch_h * patch_w锛圧GB 涓夐€氶亾 脳 patch 鍍忕礌鏁帮級
            # fan_out 杩戜技涓猴細prompt_dim锛堟彁绀哄悜閲忕殑缁村害锛?
            # eg:patch_size = 16脳16 = 256锛? * 256 = 768锛涜 prompt_dim = 192锛屽垯 val = sqrt(6/(768+192)) = sqrt(6/960) 鈮?0.079锛?
            # 鍒濆鍖栧尯闂?[-0.079, 0.079]銆傚鏋?PROJECT < 0 鐩存帴鐢?D=768 鍋?prompt_dim锛屽垯 val = sqrt(6/(768+768)) 鈮?0.0625
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            if self.use_learned_prompt_params:
                # -------- 鍓嶇疆 prompt锛堢 0 灞備娇鐢級 --------
                # 褰㈢姸锛歔1, P, prompt_dim]锛屽墠鍚戞椂浼氭墿灞曞埌 [B, P, prompt_dim]
                self.prompt_embeddings = nn.Parameter(
                    torch.zeros(1, num_tokens, prompt_dim))

                # 鑻ユ湭鎻愪緵澶栭儴 prompt_init锛屽垯閲囩敤鍧囧寑鍒嗗竷闅忔満鍒濆鍖?
                if prompt_init is None:
                    nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                else:
                    # 鑻ユ彁渚涗簡澶栭儴 prompt_init 寮犻噺锛屽垯鐢ㄥ叾鍒濆鍖栵紙甯歌浜庤縼绉?寰皟鏃讹級
                    self._seed_prompt(prompt_init, prompt_dim)
            else:
                # 瀹屽叏涓嶄娇鐢ㄥ涔犲弬鏁?prompt_embeddings锛岃€屾槸鍏ㄩ儴鐢?prompt_init_provider 鎻愪緵锛堟彁绀哄垎甯冿級
                self.prompt_embeddings = None

            # -------- Deep Prompt锛堜腑闂村眰浣跨敤锛?--------
            # 鑻ュ紑鍚?Deep Prompt锛氬湪姣忎釜涓棿灞傦紙闄ょ 0 灞傦級鍓嶆彃鍏ヤ竴娈佃灞備笓灞?prompt
            if self.prompt_config.DEEP:  # noqa (淇濇寔鍘熼鏍?
                # 鎬荤殑涓棿灞傛暟 = 鎬诲眰鏁?- 1锛堢 0 灞傚彧鐢ㄥ墠缃?prompt锛屼笉绠?deep锛?
                total_d_layer = config.transformer["num_layers"]-1  # 涓嶅惈绗?0 灞傦紙涓庤鏂囪瀹氫竴鑷达級
                # deep_prompt_embeddings: [L-1, P, prompt_dim]
                # 鍏朵腑绗?i-1 浠藉搴?encoder 绗?i 灞傦紙i 浠?1 鍒?L-1锛?
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # 鍧囧寑鍒嗗竷鍒濆鍖?deep prompt
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        # [CLS] + [ PROMPT 脳 P ] + [ PATCH 脳 N ]   鈫? (B, 1+P+N, D)
        else:
            raise ValueError("Other initiation scheme is not supported")

        if self.semantic_side_branch is not None:
            self.semantic_side_branch.debug_shapes = bool(self.debug_shapes)
        for layer_block in getattr(self.encoder, "layer", []):
            setattr(layer_block, "debug_shapes", bool(self.debug_shapes))
            if hasattr(layer_block, "attn"):
                setattr(layer_block.attn, "debug_shapes", bool(self.debug_shapes))
            if hasattr(layer_block, "semantic_attn") and layer_block.semantic_attn is not None:
                setattr(layer_block.semantic_attn, "debug_shapes", bool(self.debug_shapes))

        # 鏄惁鍐荤粨鍘熷 prompt 宓屽叆鍙傛暟锛屼娇姊害涓昏钀藉湪鍒嗗竷缃戠粶绛夊叾浠栨敮璺笂
        self.freeze_embeddings = getattr(self.prompt_config, "FREEZE_EMBEDDINGS", True)
        if self.freeze_embeddings:
            if self.prompt_embeddings is not None:
                self.prompt_embeddings.requires_grad = False
            if hasattr(self, "deep_prompt_embeddings"):
                self.deep_prompt_embeddings.requires_grad = False

        # 鈥斺€?Layer-wise prompt evolution锛氱 1鈥-1 灞傞€氳繃绾挎€у眰鏇存柊涓婁竴灞傜殑 prompt 鈥斺€?#
        num_layers = config.transformer["num_layers"]
        hidden_size = config.hidden_size
        self.prompt_update_layers = nn.ModuleList([
            Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
        ])
        # 杩戜技鎭掔瓑鍒濆鍖栵紝纭繚鍒濆琛屼负绋冲畾
        with torch.no_grad():
            for layer in self.prompt_update_layers:
                eye = torch.eye(hidden_size, device=layer.weight.device, dtype=layer.weight.dtype)
                layer.weight.copy_(eye)
                if layer.bias is not None:
                    layer.bias.zero_()

    def incorporate_prompt(self, x, semantics=None):
        """
        灏?prompt token 鎸夆€減repend鈥濈瓥鐣ュ苟鍏ワ細eg:N=14脳14=196锛涜嫢 P=10锛屽垯鎬婚暱 1+10+196=207
        杈撳叆锛?
          x: 鍘熷鍥惧儚寮犻噺 (B, C, H, W)
        涓昏姝ラ锛?
          1) 浣跨敤 embeddings.forward_patches 鎻愬彇绾?patch tokens锛歏_raw锛屽舰鐘?[B, N, D]
          2) 鐢熸垚 prompt_tokens锛?
             - 鑻ユ彁渚?prompt_init_provider锛屽垯鍩轰簬 V_raw 鍔ㄦ€佺敓鎴愶紱
             - 鍚﹀垯浣跨敤鍥哄畾鐨?self.prompt_embeddings銆?
          3) 璋冪敤 embeddings.add_cls_and_pos(V_raw) 閲嶆柊鏋勯€?[CLS|PATCH] + pos 缂栫爜锛?
          4) 鍦?CLS 涔嬪悗鎻掑叆 prompt_tokens锛堝厛鎶曞奖鍐?dropout锛夛紝寰楀埌锛?
             [CLS] + [PROMPT 脳 P] + [PATCH 脳 N]锛屽舰鐘?[B, 1+P+N, D]
        """

        B = x.shape[0]
        self._last_semantic_side_state = None

        # 1) 鎻愬彇绾?patch 宓屽叆锛氫笉鍖呭惈 CLS / 浣嶇疆缂栫爜锛歏_raw
        patch_tokens = self.embeddings.forward_patches(x)  # (B, n_patches, hidden_dim)

        # New mainline: semantic update is done by lightweight side-branch per layer.
        refined_semantics = semantics
        self._monitor_last_raw_semantics = semantics.detach() if torch.is_tensor(semantics) else None
        self._monitor_last_refined_semantics = (
            refined_semantics.detach() if torch.is_tensor(refined_semantics) else None
        )

        # 2) 鐢熸垚 prompt_tokens
        provider_used = bool(self.prompt_init_provider is not None)
        prompt_tokens = None
        if self.prompt_init_provider is not None:
            # 鐢扁€滄彁绀哄垎甯冪綉缁溾€濇垨鍏朵粬妯″潡锛屽熀浜?patch_tokens 鐢熸垚 prompt
            provider_out = self.prompt_init_provider(patch_tokens)
            # 鍏煎杩斿洖 (prompts, stats) 鐨勫舰寮忥紝浠呭彇 prompts
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
            # 鑻ヨ緭鍑轰负 [P, D]锛屽垯瑙嗕负鍗曟牱鏈ā鏉匡紝鎵╁睍 batch 缁?
            if prompt_tokens.dim() == 2:
                prompt_tokens = prompt_tokens.unsqueeze(0)

            # 妫€鏌?prompt 鏁伴噺鏄惁涓庨厤缃竴鑷?
            if prompt_tokens.shape[1] != self.num_tokens:
                raise ValueError(
                    f"prompt_init_provider returned shape {prompt_tokens.shape}, expected num_tokens={self.num_tokens}"
                )

            # 鑻?provider 鍙繑鍥炲崟涓?batch 鐨?prompt锛岃€岀湡瀹?batch_size > 1锛屽垯澶嶅埗鎵╁睍
            if prompt_tokens.shape[0] == 1 and B > 1:
                prompt_tokens = prompt_tokens.expand(B, -1, -1)
            elif prompt_tokens.shape[0] != B:
                # 鑻?provider 杈撳嚭鐨?batch 鏁颁笌杈撳叆涓嶄竴鑷达紝鍒欑洿鎺ユ姤閿?
                raise ValueError(
                    f"prompt_init_provider batch {prompt_tokens.shape[0]} incompatible with input batch {B}"
                )
        else:
            # 鑻ユ病鏈?provider锛屽垯蹇呴』浣跨敤瀛︿範鍙傛暟 prompt_embeddings
            if self.prompt_embeddings is None:
                raise RuntimeError(
                    "Prompt embeddings are disabled but no prompt_init_provider "
                    "is available. Set PROMPT.DISTRIBUTION_ONLY=False or "
                    "supply a provider."
                )
            prompt_tokens = self.prompt_embeddings
            if self.detach_prompt_grad or self.freeze_embeddings:
                prompt_tokens = prompt_tokens.detach()# 鍏朵綑鍚勫眰
                # 鍐荤粨 prompt 鍙傛暟琛紝鍙缁冩槧灏勫眰锛坧rompt_proj锛夊強鍚庣画妯″潡
                prompt_tokens = prompt_tokens.detach()
        if prompt_tokens.shape[-1] != self.vit_config.hidden_size:
            raise ValueError(
                f"Prompt feature dim {prompt_tokens.shape[-1]} incompatible with hidden_size {self.vit_config.hidden_size}"
            )
        # 3) 閲嶅缓甯?CLS/浣嶇疆缂栫爜鐨勪富搴忓垪锛歔CLS|PATCH] + pos
        x_base = self.embeddings.add_cls_and_pos(patch_tokens)  # (B, 1 + n_patches, hidden_dim)

        prompt_generated = torch.is_tensor(prompt_tokens)
        prompt_injected = False
        prompt_norm = None
        if prompt_generated:
            with torch.no_grad():
                prompt_norm = float(prompt_tokens.float().norm(dim=-1).mean().item())

        # 4) 浠呭湪姝ｅ父妯″紡涓嬫敞鍏?prompt锛汵OOP 妯″紡涓嬩繚鎸佸弬鏁?妯″潡瀛樺湪浣嗕笉鏀瑰啓涓?token 搴忓垪
        if self.noop_keep_params:
            x = x_base
        else:
            prompt_tokens = self.prompt_proj(prompt_tokens)
            x = torch.cat((
                    x_base[:, :1, :],    # 鍙彇 CLS
                    self.prompt_dropout(prompt_tokens.expand(B, -1, -1)),
                    x_base[:, 1:, :]     # 鍏朵綑 patch token
                ), dim=1)
            prompt_injected = True

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

        if self.debug_prompt_flow and not self._debug_prompt_flow_logged:
            trace_id = getattr(self, "_debug_trace_id", "trace=NA")
            logger.info(
                "[trace] %s node=B.incorporate_prompt prompt_noop_keep_params=%s provider_used=%s "
                "whether_prompt_generated=%s whether_prompt_injected_into_tokens=%s detach_prompt_grad=%s freeze_embeddings=%s "
                "prompt_requires_grad=%s token_len_before=%d token_len_after=%d prompt_shape=%s "
                "actual_token_shape_entering_backbone=%s prompt_norm=%s visual_feature_norm=%s refined_semantics=%s",
                trace_id,
                bool(self.noop_keep_params),
                provider_used,
                bool(prompt_generated),
                bool(prompt_injected),
                bool(self.detach_prompt_grad),
                bool(self.freeze_embeddings),
                bool(getattr(prompt_tokens, "requires_grad", False)) if prompt_generated else False,
                int(1 + patch_tokens.shape[1]),
                int(x.shape[1]),
                tuple(prompt_tokens.shape) if prompt_generated else None,
                tuple(x.shape),
                prompt_norm,
                float(patch_tokens.float().norm(dim=-1).mean().item()),
                tuple(refined_semantics.shape) if torch.is_tensor(refined_semantics) else None,
            )
            self._debug_prompt_flow_logged = True
        self._last_prompt_noop_info = {
            "prompt_noop_keep_params": bool(self.noop_keep_params),
            "semantic_branch_enable": bool(self.semantic_branch_enable),
            "semantic_cross_attn_enable": bool(self.semantic_cross_attn_enable),
            "shared_concept_enable": bool(self.shared_concept_enable),
            "shared_aligner_enable": bool(self.shared_aligner_enable),
            "whether_prompt_generated": bool(prompt_generated),
            "whether_prompt_injected_into_tokens": bool(prompt_injected),
            "actual_token_shape_entering_backbone": tuple(x.shape),
            "prompt_norm": prompt_norm,
            "visual_feature_norm": float(patch_tokens.float().norm(dim=-1).mean().item()),
        }

        return x, refined_semantics

    def train(self, mode=True):
        """
        閲嶅啓 nn.Module.train锛岀敤浜庢帶鍒垛€滃彧璁粌 prompt 鐩稿叧妯″潡锛屽喕缁撲富骞测€濄€?

        琛屼负锛?
        - 褰?mode=True锛堣缁冩ā寮忥級锛?
            * encoder / embeddings 缃负 eval()锛堝喕缁撱€佸叧闂?Dropout/BN 鐨勯殢鏈烘€э級锛?
            * prompt_dropout 浠嶅浜?train() 鐘舵€侊紱
            * 鑻?prompt_init_provider 鏄?nn.Module锛屼篃浼氭牴鎹?mode 璁剧疆銆?
        - 褰?mode=False锛堣瘎浼版ā寮忥級锛?
            * 瀵规墍鏈夊瓙妯″潡璋冪敤 module.train(False)锛岀粺涓€鍒囧埌 eval銆?
        """
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training: 璁粌鏈燂細鍐荤粨涓诲共
            self.encoder.eval()
            self.embeddings.eval()

            # 鍙 prompt 鐩稿叧灞備繚鎸?train 鐘舵€?
            self.prompt_proj.train(mode)
            self.prompt_dropout.train()
            self.prompt_update_layers.train(mode)

            # 鑻?provider 鏈韩鏄竴涓彲瀛︿範妯″潡锛屽垯涔熼伒寰?mode 璁剧疆
            if isinstance(self.prompt_init_provider, torch.nn.Module):
                self.prompt_init_provider.train(mode)
        else:
            # 璇勪及/鎺ㄧ悊鏃讹細鎵€鏈夊瓙妯″潡缁熶竴璺熼殢 mode
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
        )
        self._last_prompt_role_stats = {
            "semantic_branch_enable": True,
            "semantic_branch_num_tokens": int(self.semantic_side_branch.num_tokens),
            "semantic_branch_use_anchor_free": bool(getattr(self.semantic_side_branch, "use_anchor_free", False)),
            "semantic_branch_anchor_tokens": int(getattr(self.semantic_side_branch, "anchor_tokens", self.semantic_side_branch.num_tokens)),
            "semantic_branch_free_tokens": int(getattr(self.semantic_side_branch, "free_tokens", 0)),
            "semantic_branch_start_layer": int(self.semantic_side_branch.start_layer),
            "semantic_branch_end_layer": int(self.semantic_side_branch.end_layer),
            "semantic_branch_num_layers": int(num_layers),
            "semantic_branch_layer_stats": {},
        }
        return sem_tokens, h_y

    def _update_semantic_side_branch(self, sem_tokens, hidden_states, layer_idx: int, num_layers: int):
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
                "aps_energy": float(diag.get("aps_energy", 0.0)),
                "avs_energy": float(diag.get("avs_energy", 0.0)),
            }
        return sem_tokens, sem_aff, diag

    def forward_deep_prompt(self, embedding_output, semantics=None):
        """
        Deep Prompt 妯″紡涓嬬殑鍓嶅悜浼犳挱銆?
        - 鑻ユ湭鍚敤鍏变韩姒傚康鍩烘ā鍧楋紝鍒欎负鍘熷璇箟鎶曞奖锛堜緥濡?Linear(att_dim鈫扗) 鍚庣殑缁撴灉锛夛紱
        - 鑻ュ惎鐢ㄤ簡 SharedConceptAligner锛屽垯涓?S^#锛堣瀺鍚堣瑙夊悗鐨勫叡浜涔夛級锛?
          浼氬湪姣忓眰 encoder.layer[i](..., semantics, ...) 鍐呴儴琚敤浜?patch鈫抯emantic cross-attention銆?

        杈撳叆锛?
          - embedding_output: 缁忚繃 incorporate_prompt 鐨勫簭鍒?(B, 1+P+N, D)
          - semantics:        璇箟妯℃€佺壒寰侊紙鍙€夛級锛岀敤浜庝綘鍦?encoder 涓姞鍏ョ殑 cross-attention

        鏈哄埗锛?
          - 绗?0 灞傦細鐩存帴瀵?[CLS + 鍓嶇疆 PROMPT + PATCH] 鍋?self-attention 涓?MLP锛?
          - 绗?1 鈥?L-1 灞傦細
              * 浠庝笂涓€灞傝緭鍑轰腑鎴彇 prompt 娈碉紝閫氳繃璇ュ眰涓撳睘 Linear 鍋氣€減rompt 婕斿寲鈥濓紱
              * 鐢ㄦ紨鍖栧悗鐨?prompt 鏇挎崲搴忓垪涓殑 prompt 娈碉紝鍐嶈繃 encoder.layer[i]銆?
        """
        attn_weights: list = []           # 鎸夐渶淇濆瓨姣忓眰鐨勬敞鎰忓姏鏉冮噸锛坴is=True 鏃舵湁鏁堬級
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_tokens, h_y = self._init_semantic_side_state(semantics, num_layers)

        for i in range(num_layers):
            if i == 0:
                # 绗?0 灞傦細浣跨敤 provider/琛ㄥ垵濮嬪寲寰楀埌鐨?[CLS|P^0|PATCH] 搴忓垪锛屾搴︽祦鍚?provider
                hidden_states, weights, _ = self.encoder.layer[i](hidden_states, None, self.num_tokens)
            else:
                # 1) 鍙栧嚭涓婁竴灞傝緭鍑轰腑鐨?prompt 娈碉紙闀垮害鍥哄畾涓?self.num_tokens锛?
                prev_prompt = hidden_states[:, 1:1 + self.num_tokens, :]

                # 2) 閫氳繃璇ュ眰涓撳睘鐨勭嚎鎬у眰婕斿寲 prompt锛屾搴﹁惤鍦?prompt_update_layers
                evolved_prompt = self.prompt_update_layers[i - 1](prev_prompt)
                evolved_prompt = self.prompt_dropout(evolved_prompt)

                # 3) 閲嶇粍搴忓垪锛歔CLS | P^i | PATCH]锛屼繚鎸侀暱搴?1 + P + N 涓嶅彉
                hidden_states = torch.cat(
                    (
                        hidden_states[:, :1, :],  # CLS
                        evolved_prompt,  # 鏇存柊鍚庣殑 prompt
                        hidden_states[:, 1 + self.num_tokens:, :],  # PATCH 娈?
                    ),
                    dim=1,)

                # 4) 缁忚繃绗?i 灞?Transformer block锛堜粛鏀寔璇箟 cross-attn锛?
                hidden_states, weights, _ = self.encoder.layer[i](hidden_states, None, self.num_tokens)

            sem_tokens, _, _ = self._update_semantic_side_branch(
                sem_tokens=sem_tokens,
                hidden_states=hidden_states,
                layer_idx=i,
                num_layers=num_layers,
            )

            if self.encoder.vis:
                attn_weights.append(weights)

        # 鏈€缁堣緭鍑哄墠鍋氫竴娆?LayerNorm
        encoded = self.encoder.encoder_norm(hidden_states)  # 鏈€鍚庡眰鐨?LayerNorm
        if sem_tokens is not None and h_y is not None:
            mu_s_final = self.semantic_side_branch.readout(sem_tokens)
            delta_sem = mu_s_final - h_y
            self._last_semantic_side_state = {
                "h_y": h_y,
                "sem_tokens": sem_tokens,
                "mu_s_final": mu_s_final,
                "delta_sem": delta_sem,
                "anchor_tokens": sem_tokens[:, :self.semantic_side_branch.anchor_tokens, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens,
                "free_tokens": sem_tokens[:, self.semantic_side_branch.anchor_tokens:, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens[:, :0, :],
            }
            self._monitor_last_refined_semantics = mu_s_final.detach()
        return encoded, attn_weights

    def forward_deep_prompt_with_affinity(self, embedding_output, affinity_config, semantics=None):
        """
        甯︿翰鍜屽垎鏀殑 Deep Prompt 鍓嶅悜锛氫笌 forward_deep_prompt 骞宠銆?

        瀵瑰簲鍏崇郴锛?
          - forward_deep_prompt           鈫?forward_deep_prompt_with_affinity
          - encoder.layer[i].forward      鈫?encoder.layer[i].forward_with_affinity
          - forward/forward_with_affinity 鍚屾牱鍏变韩 incorporate_prompt 鐢熸垚鐨?[CLS|P|PATCH]

        杩斿洖:
          encoded:     LayerNorm 鍚庣殑鏈€缁堝簭鍒?
          attn_weights: 鍙鍖栫敤娉ㄦ剰鍔涙潈閲嶏紙vis=True 鏃讹級
          affinities:   姣忓眰鐨勪翰鍜岀煩闃靛垪琛紝鏉ヨ嚜 compute_affinity锛堥€愬眰鏀堕泦锛屾柟渚夸笂灞傛寜闇€鎸戦€夊仛瀵归綈鎹熷け鎴栬皟璇曪級
        """
        attn_weights: list = []
        affinities: list = []
        hidden_states = embedding_output
        weights = None
        num_layers = self.vit_config.transformer["num_layers"]
        sem_tokens, h_y = self._init_semantic_side_state(semantics, num_layers)

        for i in range(num_layers):
            if i == 0:
                # 绗?0 灞傦細鐩存帴浣跨敤 provider/琛ㄧ敓鎴愮殑 prompt 搴忓垪锛屽苟璧板甫浜插拰鐨勫墠鍚?
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
            self._last_semantic_side_state = {
                "h_y": h_y,
                "sem_tokens": sem_tokens,
                "mu_s_final": mu_s_final,
                "delta_sem": delta_sem,
                "anchor_tokens": sem_tokens[:, :self.semantic_side_branch.anchor_tokens, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens,
                "free_tokens": sem_tokens[:, self.semantic_side_branch.anchor_tokens:, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens[:, :0, :],
            }
            self._monitor_last_refined_semantics = mu_s_final.detach()
        return encoded, attn_weights, affinities
    def forward(self, x, semantics=None):
        """
        鏍囧噯鍓嶅悜锛?
        - 杈撳叆 semantics 涓?batch 鐨勭被绾ц涔夛紙灞炴€у悜閲忥級锛屽舰鐘朵竴鑸负 [B, att_dim]锛?
        - 鍦?incorporate_prompt 涓細
            * 鍏堜粠 x 鎻愬彇 patch_tokens锛?
            * 鑻ュ惎鐢?SharedConceptAligner锛屽垯灏?(patch_tokens, semantics) 鏄犲皠涓?refined_semantics=S^#锛?
        - 鍚庣画 encoder / forward_deep_prompt 浣跨敤鐨?semantics 瀹為檯涓婂氨鏄?refined_semantics銆?
        娴佺▼锛?
          1) 璋冪敤 incorporate_prompt(x) 灏?prompt 鍚堝叆杈撳叆搴忓垪锛?
          2) 鑻ュ惎鐢?Deep Prompt锛屽垯璋冪敤 forward_deep_prompt 鍋氬灞傛繁搴︽彁绀猴紱
             鍚﹀垯鐩存帴灏嗗簭鍒楅€佸叆 encoder锛?
          3) 杩斿洖缂栫爜鍚庣殑瀹屾暣 token 搴忓垪 encoded 鍜屽彲閫夌殑 attn_weights銆?
        """
        # 1) prepend prompt锛堝湪 CLS 鍚庢彃鍏ユ彁绀猴級
        embedding_output, semantics = self.incorporate_prompt(x, semantics)

        # 2) deep prompt锛堝彲閫夛級
        # NOOP 妯″紡涓嬩笉鍏佽 prompt 褰卞搷涓?token 璺緞锛屽洜姝や笉杩涘叆 deep prompt 婕斿寲鍒嗘敮銆?
        effective_prompt_tokens = 0 if self.noop_keep_params else self.num_tokens
        if self.prompt_config.DEEP and (not self.noop_keep_params):
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output, semantics)
        else:
            # 鑻ヤ笉浣跨敤 Deep Prompt锛屽垯鐩存帴灏嗘暣涓簭鍒楅€佸叆 encoder
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
                self._last_semantic_side_state = {
                    "h_y": h_y,
                    "sem_tokens": sem_tokens,
                    "mu_s_final": mu_s_final,
                    "delta_sem": mu_s_final - h_y,
                    "anchor_tokens": sem_tokens[:, :self.semantic_side_branch.anchor_tokens, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens,
                    "free_tokens": sem_tokens[:, self.semantic_side_branch.anchor_tokens:, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens[:, :0, :],
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

        effective_prompt_tokens = 0 if self.noop_keep_params else self.num_tokens
        effective_affinity_config = affinity_config
        if self.noop_keep_params and isinstance(affinity_config, dict):
            # In NOOP mode there is no prompt segment in the sequence.
            # Force affinity prompt length to 0 to avoid treating first patches as prompt tokens.
            effective_affinity_config = dict(affinity_config)
            effective_affinity_config["prompt_length"] = 0

        if self.prompt_config.DEEP and (not self.noop_keep_params):
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
                self._last_semantic_side_state = {
                    "h_y": h_y,
                    "sem_tokens": sem_tokens,
                    "mu_s_final": mu_s_final,
                    "delta_sem": mu_s_final - h_y,
                    "anchor_tokens": sem_tokens[:, :self.semantic_side_branch.anchor_tokens, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens,
                    "free_tokens": sem_tokens[:, self.semantic_side_branch.anchor_tokens:, :] if getattr(self.semantic_side_branch, "use_anchor_free", False) else sem_tokens[:, :0, :],
                }
                self._monitor_last_refined_semantics = mu_s_final.detach()

        return encoded, attn_weights, affinities

    def _seed_prompt(self, prompt_init, prompt_dim):
        """
        浣跨敤澶栭儴鎻愪緵鐨?prompt_init 寮犻噺瀵?prompt_embeddings 杩涜鍒濆鍖栥€?

        瑕佹眰锛?
        - prompt_init 鑷冲皯涓?2 缁达紙P, d锛夋垨 3 缁达紙1, P, d锛夛紱
        - 鍏朵腑 P 蹇呴』绛変簬 self.num_tokens锛宒 蹇呴』绛変簬 prompt_dim銆?
        """
        init = prompt_init.detach()
        if init.dim() == 2:
            init = init.unsqueeze(0)
        if init.shape[1:] != (self.num_tokens, prompt_dim):
            raise ValueError(
                f"prompt_init has shape {init.shape}, expected (1, {self.num_tokens}, {prompt_dim})"
            )

        # 鐩存帴鎷疯礉鍒板彲瀛︿範鍙傛暟涓?
        with torch.no_grad():
            self.prompt_embeddings.copy_(init)



class PromptedVisionTransformer(VisionTransformer):
    """
    鍦ㄦ爣鍑?VisionTransformer 澶栧３涓嬶紝浣跨敤 PromptedTransformer 浣滀负鍐呴儴鐨?transformer 缂栫爜鍣ㄣ€?

    濂藉锛?
    - 澶嶇敤鍘熸湁 VisionTransformer 鐨勬帴鍙ｄ笌鍒嗙被澶磋璁★紱
    - 鍦ㄤ笉鏀瑰彉鈥淐LS 姹犲寲 + 绾挎€у垎绫烩€濇暣浣撻€昏緫鐨勬儏鍐典笅锛屽皢 Prompt 鏈哄埗鏃犵紳娉ㄥ叆锛?
    - 瀵瑰鐨?forward 鎺ュ彛鍩烘湰淇濇寔涓€鑷达紝鍙槸鍐呴儴缂栫爜闃舵鎹㈡垚浜?PromptedTransformer銆?
    """
    def __init__(self, prompt_cfg, model_type,img_size=224, num_classes=21843, vis=False, prompt_init=None, prompt_init_provider=None):        # 褰撳墠瀹炵幇鍙敮鎸佸師鐢熺殑 CLS 姹犲寲鏂瑰紡锛坥riginal锛?
        """
        :param prompt_cfg:   PROMPT 瀛愰厤缃紙NUM_TOKENS / PROJECT / DEEP 绛夛級
        :param model_type:   ViT 妯″瀷绫诲瀷锛堢敤浜庝粠 CONFIGS 涓彇缁撴瀯閰嶇疆锛?
        :param img_size:     杈撳叆鍥惧儚灏哄锛堥粯璁?224锛?
        :param num_classes:  鍒嗙被绫诲埆鏁帮紙榛樿 21843锛屽搴?ImageNet-21K锛?
        :param vis:          鏄惁杩斿洖娉ㄦ剰鍔涙潈閲?
        :param prompt_init:  澶栭儴 prompt 鍒濆鍖栧紶閲忥紙鍙€夛級
        :param prompt_init_provider: 杩愯鏃?prompt 鐢熸垚鍣紙鍙€夛級
        """
        # 褰撳墠鍙敮鎸佸師濮嬬殑 CLS 姹犲寲鏂瑰紡锛堜笉鍋?GAP 绛夋浛浠ｆ睜鍖栵級
        assert prompt_cfg.VIT_POOL_TYPE == "original"

        # 鍏堣皟鐢ㄧ埗绫?VisionTransformer 鍒濆鍖栧熀纭€缁撴瀯
        super(PromptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg

        # 鍙栧嚭缁撴瀯瑙勬牸锛堝 hidden_size銆佸眰鏁般€乸atch 澶у皬绛夛級
        vit_cfg = CONFIGS[model_type]
        # 鏍稿績鏇挎崲锛氭妸鍐呴儴鐨?transformer 鐢ㄢ€滃甫 Prompt 鐨勨€濈増鏈浛鎹?
        self.transformer = PromptedTransformer(prompt_cfg, vit_cfg, img_size, vis, prompt_init=prompt_init, prompt_init_provider=prompt_init_provider,)

    def forward(self, x, vis=False, semantics=None):
        """
        鍓嶅悜娴佺▼锛?

        1) 璋冪敤鍐呴儴鐨?PromptedTransformer(x, semantics)锛?
           - 寰楀埌缂栫爜鍚庣殑搴忓垪 x锛堝惈 CLS + PROMPT + PATCH锛夛紱
           - 鍙€夎繑鍥炲悇灞傜殑娉ㄦ剰鍔涙潈閲?attn_weights銆?
        2) 鍙?CLS 浣嶇疆鐨?token锛坸[:, 0]锛変綔涓哄叏灞€琛ㄥ緛锛?
        3) 閫氳繃 self.head锛堢嚎鎬у眰锛夊緱鍒版渶缁?logits锛?
        4) 鑻?vis=False锛屽垯鍙繑鍥?logits锛?
           鑻?vis=True锛屽垯杩斿洖 (logits, attn_weights)锛屼究浜庡彲瑙嗗寲/鍒嗘瀽銆?
        """
        # transformer 杩斿洖鐨?x 涓虹紪鐮佸悗鐨?token 搴忓垪锛宎ttn_weights 涓哄彲瑙嗗寲鐢ㄦ敞鎰忓姏鏉冮噸
        x, attn_weights = self.transformer(x, semantics)

        # 鍙?CLS token锛堜綅缃?0锛変綔涓哄叏灞€琛ㄥ緛
        x = x[:, 0]

        # 绾挎€у垎绫诲ご -> logits锛堟湭鍋?softmax锛?
        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights

    def forward_with_affinity(self, x, affinity_config, vis=False, semantics=None):
        """
        甯︿翰鍜岀煩闃佃緭鍑虹殑鍓嶅悜鎺ュ彛锛堜笌 VisionTransformer.forward_with_affinity 骞宠锛夈€?

        - 璋冪敤鍐呴儴 PromptedTransformer.forward_with_affinity 杩斿洖搴忓垪/鏉冮噸/浜插拰
        - 鍙?CLS 鍋氬垎绫?
        - vis=False 杩斿洖 (logits, affinities)锛寁is=True 棰濆杩斿洖 attn_weights
        """
        x, attn_weights, affinities = self.transformer.forward_with_affinity(x, affinity_config, semantics)

        logits = self.head(x[:, 0])

        if not vis:
            return logits, affinities
        return logits, attn_weights, affinities
