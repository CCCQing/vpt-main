#!/usr/bin/env python3
"""Prompt distribution modules used before prompted ViT.

本文件负责在进入 PromptedTransformer 主序列前，生成 ViaPT-style prompt：
1. 先从某种视觉来源 E0 中抽取实例级统计量；
2. 用轻量 stats head 输出 mu / logvar；
3. 从 N(mu, std^2) 采样 instance prompt；
4. 再拼接 dataset/task-level 的 learnable domain prompt。

类名仍保留为 ``PreViTPromptDistributor``，这样 build_vit_backbone.py 的构建入口
不用大范围改动。
"""

from __future__ import annotations

import os
from urllib.parse import urlparse
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


_ALLOWED_SOURCES = {
    "vit_cls_prepass",
    "vit_cls_prepass_constant",
    "cnn_torchvision",
    "clip_frozen",
    "dinov2_small",
    "token_mlp",
}


class MeanConditionedDeepPromptResidual(nn.Module):
    """Build a deterministic Distributor-mean correction for static Deep Prompt.

    The historical ``shared`` mode repeats ``mu(x)`` over Prompt slots and is
    parameter-compatible with the original B1 implementation.  Optional
    ``slot_low_rank`` content and sample-adaptive gates are constructed only by
    explicit follow-up configs.  A zero layer gate remains exactly equivalent
    to the static Deep Prompt baseline and consumes no sampling RNG.
    """

    _INTERVENTION_MODES = {
        "delta_zero",
        "mean_swap",
        "mean_replace",
        "layer_scales",
    }
    _CONTENT_MODES = {"shared", "slot_low_rank"}
    _SAMPLE_GATE_MODES = {"none", "shared", "grouped", "layerwise"}
    _SAMPLE_GATE_INPUTS = {"residual_source", "constant"}

    def __init__(
        self,
        *,
        dim: int,
        prompt_len: int,
        num_layers: int,
        gate_init: float = 0.0,
        content_mode: str = "shared",
        slot_rank: int = 8,
        sample_gate_mode: str = "none",
        sample_gate_input: str = "residual_source",
        sample_gate_hidden_dim: int = 32,
        sample_gate_init: float = 0.95,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.prompt_len = int(prompt_len)
        self.num_layers = int(num_layers)
        if self.dim <= 0 or self.prompt_len <= 0 or self.num_layers <= 0:
            raise ValueError("Deep Prompt residual dimensions must be positive")
        self.content_mode = str(content_mode).strip().lower()
        if self.content_mode not in self._CONTENT_MODES:
            raise ValueError(
                "Unsupported Deep Prompt residual CONTENT_MODE={!r}".format(
                    content_mode
                )
            )
        self.slot_rank = int(slot_rank)
        if self.slot_rank <= 0:
            raise ValueError("Deep Prompt residual SLOT_RANK must be positive")
        self.sample_gate_mode = str(sample_gate_mode).strip().lower()
        if self.sample_gate_mode not in self._SAMPLE_GATE_MODES:
            raise ValueError(
                "Unsupported Deep Prompt residual SAMPLE_GATE_MODE={!r}".format(
                    sample_gate_mode
                )
            )
        self.sample_gate_input = str(sample_gate_input).strip().lower()
        if self.sample_gate_input not in self._SAMPLE_GATE_INPUTS:
            raise ValueError(
                "Unsupported Deep Prompt residual SAMPLE_GATE_INPUT={!r}".format(
                    sample_gate_input
                )
            )
        self.sample_gate_hidden_dim = int(sample_gate_hidden_dim)
        if self.sample_gate_hidden_dim <= 0:
            raise ValueError(
                "Deep Prompt residual SAMPLE_GATE_HIDDEN_DIM must be positive"
            )
        self.sample_gate_init = float(sample_gate_init)
        if not 0.0 < self.sample_gate_init < 1.0:
            raise ValueError(
                "Deep Prompt residual SAMPLE_GATE_INIT must be inside (0, 1)"
            )
        self.layer_gate = nn.Parameter(
            torch.full((self.num_layers,), float(gate_init))
        )
        self.slot_coefficients = None
        self.slot_basis = None
        if self.content_mode == "slot_low_rank":
            self.slot_coefficients = nn.ModuleList(
                nn.Linear(
                    self.dim,
                    self.prompt_len * self.slot_rank,
                    bias=True,
                )
                for _ in range(self.num_layers)
            )
            self.slot_basis = nn.Parameter(
                torch.empty(self.num_layers, self.slot_rank, self.dim)
            )
            nn.init.xavier_uniform_(self.slot_basis)

        self.sample_gate_head = None
        if self.sample_gate_mode != "none":
            output_dim = {
                "shared": 1,
                "grouped": 3,
                "layerwise": self.num_layers,
            }[self.sample_gate_mode]
            self.sample_gate_head = nn.Sequential(
                nn.Linear(self.dim, self.sample_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.sample_gate_hidden_dim, output_dim),
            )
            final = self.sample_gate_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(
                final.bias,
                math.log(self.sample_gate_init / (1.0 - self.sample_gate_init)),
            )
        self._runtime_intervention = None

    def _effective_mean(self, mean: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if mean.dim() != 2 or int(mean.shape[-1]) != self.dim:
            raise ValueError(
                f"Deep Prompt residual expects mean [B,{self.dim}], got {tuple(mean.shape)}"
            )
        source = mean
        stats: Dict[str, torch.Tensor] = {}
        intervention = self._runtime_intervention
        if isinstance(intervention, dict):
            mode = str(intervention.get("mode", ""))
            if mode not in self._INTERVENTION_MODES:
                raise ValueError(f"Unsupported Deep Prompt residual intervention: {mode}")
            if mode == "mean_swap":
                permutation = intervention.get("permutation")
                if not torch.is_tensor(permutation):
                    raise ValueError("mean_swap requires a permutation tensor")
                permutation = permutation.to(device=source.device, dtype=torch.long)
                if tuple(permutation.shape) != (int(source.shape[0]),):
                    raise ValueError("mean_swap permutation has incompatible shape")
                if sorted(permutation.detach().cpu().tolist()) != list(range(int(source.shape[0]))):
                    raise ValueError("mean_swap permutation must be bijective")
                source = source.index_select(0, permutation)
                stats["mean_swap_fixed_point_ratio"] = (
                    permutation
                    == torch.arange(int(source.shape[0]), device=permutation.device)
                ).to(source.dtype)
            elif mode == "mean_replace":
                replacement = intervention.get("replacement")
                if not torch.is_tensor(replacement):
                    raise ValueError("mean_replace requires a replacement tensor")
                replacement = replacement.to(device=source.device, dtype=source.dtype)
                if tuple(replacement.shape) != tuple(source.shape):
                    raise ValueError(
                        "mean_replace tensor has incompatible shape: {} != {}".format(
                            tuple(replacement.shape), tuple(source.shape)
                        )
                    )
                stats["mean_replace_distance"] = (
                    replacement - source
                ).float().norm(dim=-1)
                source = replacement
        return source, stats

    def _raw_delta(self, source: torch.Tensor, layer_id: int) -> torch.Tensor:
        if self.content_mode == "shared":
            return source[:, None, :].expand(-1, self.prompt_len, -1)
        coefficients = self.slot_coefficients[layer_id](source).reshape(
            int(source.shape[0]), self.prompt_len, self.slot_rank
        )
        basis = self.slot_basis[layer_id].to(
            device=source.device, dtype=source.dtype
        )
        return torch.matmul(coefficients, basis)

    def _sample_gate(self, source: torch.Tensor, layer_id: int) -> torch.Tensor:
        if self.sample_gate_head is None:
            return source.new_ones(int(source.shape[0]))
        gate_input = (
            source
            if self.sample_gate_input == "residual_source"
            else torch.ones_like(source)
        )
        values = torch.sigmoid(self.sample_gate_head(gate_input))
        if self.sample_gate_mode == "shared":
            index = 0
        elif self.sample_gate_mode == "grouped":
            index = min(2, (int(layer_id) * 3) // self.num_layers)
        else:
            index = int(layer_id)
        return values[:, index]

    def _runtime_layer_scale(self, layer_id: int, reference: torch.Tensor) -> torch.Tensor:
        intervention = self._runtime_intervention
        if not isinstance(intervention, dict):
            return reference.new_tensor(1.0)
        mode = str(intervention.get("mode", ""))
        if mode == "delta_zero":
            return reference.new_tensor(0.0)
        if mode != "layer_scales":
            return reference.new_tensor(1.0)
        scales: Sequence[float] = intervention.get("scales", ())
        if len(scales) != self.num_layers:
            raise ValueError(
                "layer_scales requires exactly {} values".format(self.num_layers)
            )
        value = float(scales[int(layer_id)])
        if not math.isfinite(value):
            raise ValueError("layer_scales values must be finite")
        return reference.new_tensor(value)

    def forward_layer(
        self,
        mean: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"Deep Prompt residual layer_id={layer_id} outside [0,{self.num_layers - 1}]"
            )
        source, intervention_stats = self._effective_mean(mean)
        raw_delta = self._raw_delta(source, layer_id)
        layer_gate = self.layer_gate[layer_id].to(
            device=raw_delta.device, dtype=raw_delta.dtype
        )
        sample_gate = self._sample_gate(source, layer_id).to(
            device=raw_delta.device, dtype=raw_delta.dtype
        )
        runtime_scale = self._runtime_layer_scale(layer_id, raw_delta)
        effective_gate = layer_gate * sample_gate * runtime_scale
        applied_delta = effective_gate[:, None, None] * raw_delta
        stats = {
            "layer_id": layer_id,
            "source_mu": source,
            "raw_delta": raw_delta,
            "applied_delta": applied_delta,
            "gate": effective_gate,
            "layer_gate": layer_gate.expand(int(raw_delta.shape[0])),
            "sample_gate": sample_gate,
            "runtime_scale": runtime_scale.expand(int(raw_delta.shape[0])),
            "content_mode": self.content_mode,
            "sample_gate_mode": self.sample_gate_mode,
            "sample_gate_input": self.sample_gate_input,
        }
        stats.update(intervention_stats)
        return applied_delta, stats


def prompt_kl_loss(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    计算 instance prompt 分布的 KL 正则。

    数学形式:
        KL(N(mu, sigma^2) || N(0, I))
        = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)

    注意:
    - 这里只约束 instance prompt 的高斯分布中心和方差；
    - domain prompt 是独立可学习参数，不参与这个 KL。
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    if reduction == "none":
        return kl
    raise ValueError(f"Unsupported prompt KL reduction: {reduction}")


class _VectorStatsHead(nn.Module):
    """
    向量视觉输入的统计头。

    输入可以来自 vit_cls_prepass / CNN / CLIP / DINO 等，形状为 [B, D_in]。
    不先强行投到 768，而是直接降到 STATS_HIDDEN_DIM，再输出 [mu, logvar]。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(out_dim) * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Vector stats head expects [B,D], got {tuple(x.shape)}")
        return self.net(x)


class _TokenStatsHead(nn.Module):
    """
    token 输入的统计头，对应 SOURCE="token_mlp"。

    输入是 ViT image tokens [B, N, 768]，先逐 token 降维到 H，
    再对 token 维求均值，最后输出 [B, 2*768]。
    这是当前最轻量、无外部模型依赖的 ViaPT-style 分布生成器。
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.token_down = nn.Sequential(
            nn.Linear(int(dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.stats_out = nn.Linear(int(hidden_dim), int(dim) * 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Token stats head expects [B,N,D], got {tuple(tokens.shape)}")
        pooled = self.token_down(tokens).mean(dim=1)
        return self.stats_out(pooled)


class FrozenTorchvisionCNN(nn.Module):
    """
    冻结 torchvision CNN 的图像编码器。

    设计边界:
    - 只使用 features，不使用 classifier；
    - forward 全程 no_grad；
    - train(mode=True) 后也强制保持 eval，避免 BN/Dropout 改变行为；
    - EXTERNAL_ALLOW_DOWNLOAD=False 时只读取本地 torchvision cache，不自动联网下载。
    """

    _SUPPORTED = {"efficientnet_b0", "mobilenet_v3_small"}
    _OUT_DIMS = {"efficientnet_b0": 1280, "mobilenet_v3_small": 576}

    def __init__(self, name: str, allow_download: bool) -> None:
        super().__init__()
        if name not in self._SUPPORTED:
            raise ValueError(f"Unsupported CNN_NAME='{name}'. Expected one of {sorted(self._SUPPORTED)}")
        self.name = str(name)
        self.out_dim = int(self._OUT_DIMS[name])

        import torchvision.models as tv_models

        weights = tv_models.get_model_weights(name).DEFAULT
        if allow_download:
            self.model = tv_models.get_model(name, weights=weights)
        else:
            self.model = tv_models.get_model(name, weights=None)
            weight_path = self._torchvision_cache_path(weights.url)
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    "cnn_torchvision requires cached torchvision weights when EXTERNAL_ALLOW_DOWNLOAD=False: "
                    f"{weight_path}"
                )
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        self.features = self.model.features
        self._freeze()

    @staticmethod
    def _torchvision_cache_path(url: str) -> str:
        file_name = os.path.basename(urlparse(url).path)
        return os.path.join(torch.hub.get_dir(), "checkpoints", file_name)

    def _freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(False)
        self._freeze()
        return self

    def forward(self, raw_image: torch.Tensor) -> torch.Tensor:
        if raw_image.dim() != 4:
            raise ValueError(f"cnn_torchvision expects raw_image [B,3,H,W], got {tuple(raw_image.shape)}")
        with torch.no_grad():
            feat = self.features(raw_image)
            vec = feat.mean(dim=(-2, -1))
        return vec


class _LocalTorchModuleImageEncoder(nn.Module):
    """
    本地冻结图像编码器加载器，用于 clip_frozen / dinov2_small。

    当前只接受本地 .pt / TorchScript 模块，目的是避免训练时隐式下载外部权重。
    输出会被统一解析成 [B, D] 向量，再送入 _VectorStatsHead。
    """

    def __init__(self, local_dir: str, file_name: str, source_name: str, normalize_output: bool) -> None:
        super().__init__()
        if not local_dir:
            raise ValueError(f"{source_name} requires a non-empty local directory.")
        path = os.path.join(local_dir, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{source_name} local weight file not found: {path}")
        self.encoder = self._load_module(path)
        self.source_name = str(source_name)
        self.normalize_output = bool(normalize_output)
        self._freeze()

    @staticmethod
    def _load_module(path: str) -> nn.Module:
        try:
            module = torch.jit.load(path, map_location="cpu")
        except RuntimeError:
            module = torch.load(path, map_location="cpu")
        if not isinstance(module, nn.Module):
            raise TypeError(f"Local encoder file must load as nn.Module, got {type(module)} from {path}")
        return module

    def _freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(False)
        self._freeze()
        return self

    @staticmethod
    def _extract_vector(out) -> torch.Tensor:
        if torch.is_tensor(out):
            if out.dim() == 2:
                return out
            if out.dim() == 3:
                return out[:, 0, :]
            raise ValueError(f"Frozen local encoder returned unsupported tensor shape {tuple(out.shape)}")
        if isinstance(out, dict):
            for key in ("image_embeds", "pooler_output", "last_hidden_state", "x_norm_clstoken"):
                if key in out:
                    return _LocalTorchModuleImageEncoder._extract_vector(out[key])
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise ValueError("Frozen local encoder returned an empty tuple/list.")
            return _LocalTorchModuleImageEncoder._extract_vector(out[0])
        raise TypeError(f"Frozen local encoder returned unsupported output type {type(out)}")

    def forward(self, raw_image: torch.Tensor) -> torch.Tensor:
        if raw_image.dim() != 4:
            raise ValueError(f"{self.source_name} expects raw_image [B,3,H,W], got {tuple(raw_image.shape)}")
        with torch.no_grad():
            vec = self._extract_vector(self.encoder(raw_image))
            if self.normalize_output:
                vec = torch.nn.functional.normalize(vec, dim=-1)
        return vec


class PreViTPromptDistributor(nn.Module):
    """
    根据视觉输入生成 prompt distribution。

    输出:
    - prompt_tokens: [B, prompt_len, 768]
    - stats["mu"]: [B, 768]
    - stats["logvar"]: [B, 768]
    - stats["std"]: [B, 768]

    prompt 组成:
    - 前 INSTANCE_TOKENS 个是 image-conditioned instance prompt；
    - 后 DOMAIN_TOKENS 个是 dataset/task-level learnable domain prompt。
    """

    _CLIP_OUT_DIMS = {
        "mobileclip_s0": 512,
        "tinyclip_vit8m": 512,
    }

    def __init__(
        self,
        dim: int,
        prompt_len: int,
        hidden_dim: int,
        source: str,
        instance_tokens: int,
        domain_tokens: int,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
        eval_sample_mode: str = "mean",
        use_slot_embed: bool = False,
        cnn_name: str = "efficientnet_b0",
        clip_name: str = "mobileclip_s0",
        clip_local_dir: str = "",
        dino_local_dir: str = "",
        external_allow_download: bool = False,
        output_param: str = "logvar",
        fixed_eps_seed: int = 0,
        debug_distributor_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.prompt_len = int(prompt_len)
        self.hidden_dim = int(hidden_dim)
        self.source = str(source)
        self.instance_tokens = int(instance_tokens)
        self.domain_tokens = int(domain_tokens)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.eval_sample_mode = str(eval_sample_mode)
        self.use_slot_embed = bool(use_slot_embed)
        self.output_param = str(output_param)
        self.fixed_eps_seed = int(fixed_eps_seed)
        self.debug_distributor_shapes = bool(debug_distributor_shapes)
        self._debug_shapes_logged = False
        self._prompt_output_intervention = None

        # 当前 ViT-B/16 主线 hidden size 固定为 768；其它 hidden size 需要同步检查
        # stats head、domain prompt、slot embedding 和可视化接口。
        if self.dim != 768:
            raise ValueError(f"PreViTPromptDistributor currently requires dim=768, got {self.dim}")
        if self.instance_tokens + self.domain_tokens != self.prompt_len:
            raise ValueError(
                "DISTRIBUTOR.INSTANCE_TOKENS + DISTRIBUTOR.DOMAIN_TOKENS must equal MODEL.PROMPT.NUM_TOKENS."
            )
        if self.source not in _ALLOWED_SOURCES:
            raise ValueError(f"Unsupported DISTRIBUTOR.SOURCE='{self.source}'. Expected {sorted(_ALLOWED_SOURCES)}")
        if self.eval_sample_mode not in {"mean", "fixed_eps"}:
            raise ValueError("DISTRIBUTOR.EVAL_SAMPLE_MODE must be 'mean' or 'fixed_eps'.")
        if self.output_param != "logvar":
            raise ValueError("DISTRIBUTOR.OUTPUT_PARAM currently supports only 'logvar'.")
        if self.logvar_min > self.logvar_max:
            raise ValueError("DISTRIBUTOR.LOGVAR_MIN must be <= LOGVAR_MAX.")
        # 根据 SOURCE 选择视觉统计量来源。所有来源最终只负责输出 stats_out=[B,1536]，
        # 后续 mu/logvar 切分、采样、拼接 domain prompt 都走同一条逻辑。
        self.frozen_encoder = None
        self.stats_head = None
        if self.source in {"vit_cls_prepass", "vit_cls_prepass_constant"}:
            self.stats_head = _VectorStatsHead(768, self.hidden_dim, self.dim)
        elif self.source == "cnn_torchvision":
            self.frozen_encoder = FrozenTorchvisionCNN(cnn_name, bool(external_allow_download))
            self.stats_head = _VectorStatsHead(self.frozen_encoder.out_dim, self.hidden_dim, self.dim)
        elif self.source == "clip_frozen":
            if clip_name not in self._CLIP_OUT_DIMS:
                raise ValueError(f"Unsupported CLIP_NAME='{clip_name}'. Expected one of {sorted(self._CLIP_OUT_DIMS)}")
            file_name = f"{clip_name}.pt"
            self.frozen_encoder = _LocalTorchModuleImageEncoder(
                clip_local_dir,
                file_name=file_name,
                source_name="clip_frozen",
                normalize_output=True,
            )
            self.stats_head = _VectorStatsHead(self._CLIP_OUT_DIMS[clip_name], self.hidden_dim, self.dim)
        elif self.source == "dinov2_small":
            self.frozen_encoder = _LocalTorchModuleImageEncoder(
                dino_local_dir,
                file_name="dinov2_small.pt",
                source_name="dinov2_small",
                normalize_output=False,
            )
            self.stats_head = _VectorStatsHead(384, self.hidden_dim, self.dim)
        elif self.source == "token_mlp":
            self.stats_head = _TokenStatsHead(self.dim, self.hidden_dim)

        # domain prompt 不依赖单张图像，是任务/数据集级可学习提示。
        self.domain_prompt = nn.Parameter(torch.zeros(1, self.domain_tokens, self.dim))
        if self.domain_tokens > 0:
            nn.init.normal_(self.domain_prompt, mean=0.0, std=0.02)
        # slot_embed 是可选的 instance prompt 槽位编码，只区分第几个 instance prompt。
        if self.use_slot_embed:
            self.slot_embed = nn.Parameter(torch.zeros(1, self.instance_tokens, self.dim))
            nn.init.normal_(self.slot_embed, mean=0.0, std=0.02)
        # fixed_eps 只在 eval 且 EVAL_SAMPLE_MODE="fixed_eps" 时使用。
        # 使用独立 Generator，避免初始化该 buffer 消耗或扰动全局 torch 随机序列。
        fixed_eps_generator = torch.Generator()
        fixed_eps_generator.manual_seed(self.fixed_eps_seed)
        self.register_buffer(
            "fixed_eps",
            torch.randn(1, self.instance_tokens, self.dim, generator=fixed_eps_generator),
        )
        self._freeze_external_encoder()

    def _freeze_external_encoder(self) -> None:
        """确保外部视觉编码器不训练、不进 optimizer、不受 train() 切换影响。"""
        if self.frozen_encoder is None:
            return
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        # stats head / domain prompt / slot embedding 正常切 train/eval；
        # 但外部 CNN/CLIP/DINO 必须重新锁回 eval。
        super().train(mode)
        self._freeze_external_encoder()
        return self

    def _encode_visual(
        self,
        raw_image: Optional[torch.Tensor],
        vit_patch_tokens: Optional[torch.Tensor],
        vit_image_tokens: Optional[torch.Tensor],
        vit_cls: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将不同 SOURCE 的视觉输入统一转换为:
        - visual_input: 原始统计输入，供 debug/后续诊断保存；  patch_tokens不含 CLS，不含 position embedding image_tokens含 position embedding
        - stats_out: [B, 2*768]，前半是 mu，后半是 logvar。
        """
        if self.source in {"vit_cls_prepass", "vit_cls_prepass_constant"}:
            if vit_cls is None:
                raise ValueError(
                    f"SOURCE='{self.source}' requires vit_cls from PromptedTransformer prepass."
                )
            visual_input = (
                vit_cls
                if self.source == "vit_cls_prepass"
                else torch.ones_like(vit_cls)
            )
            return visual_input, self.stats_head(visual_input)

        if self.source == "cnn_torchvision":
            if raw_image is None:
                raise ValueError("SOURCE='cnn_torchvision' requires raw_image.")
            visual_input = self.frozen_encoder(raw_image)
            return visual_input, self.stats_head(visual_input)

        if self.source == "clip_frozen":
            if raw_image is None:
                raise ValueError("SOURCE='clip_frozen' requires raw_image.")
            visual_input = self.frozen_encoder(raw_image)
            return visual_input, self.stats_head(visual_input)

        if self.source == "dinov2_small":
            if raw_image is None:
                raise ValueError("SOURCE='dinov2_small' requires raw_image.")
            visual_input = self.frozen_encoder(raw_image)
            return visual_input, self.stats_head(visual_input)

        if self.source == "token_mlp":
            if vit_image_tokens is None:
                raise ValueError("SOURCE='token_mlp' requires vit_image_tokens.")
            visual_input = vit_image_tokens
            return visual_input, self.stats_head(visual_input)

        raise ValueError(f"Unsupported DISTRIBUTOR.SOURCE='{self.source}'")

    def _sample_instance_prompt(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        从 N(mu, std^2) 生成 instance prompt。

        训练阶段使用随机 eps；评测阶段默认 eps=0，即直接使用均值 prompt，
        这样评测指标不会受采样随机性影响。
        """
        if self.training:
            eps = torch.randn(
                mu.shape[0],
                self.instance_tokens,
                self.dim,
                device=mu.device,
                dtype=mu.dtype,
            )
        elif self.eval_sample_mode == "mean":
            eps = torch.zeros(
                mu.shape[0],
                self.instance_tokens,
                self.dim,
                device=mu.device,
                dtype=mu.dtype,
            )
        elif self.eval_sample_mode == "fixed_eps":
            eps = self.fixed_eps.to(device=mu.device, dtype=mu.dtype).expand(mu.shape[0], -1, -1)
        else:
            raise ValueError(f"Unsupported EVAL_SAMPLE_MODE='{self.eval_sample_mode}'")
        instance_prompt = mu[:, None, :] + std[:, None, :] * eps
        if self.use_slot_embed:
            instance_prompt = instance_prompt + self.slot_embed.to(device=mu.device, dtype=mu.dtype)
        return instance_prompt

    def _apply_prompt_output_intervention(
        self,
        instance_prompt: torch.Tensor,
        domain_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        intervention = self._prompt_output_intervention
        if not isinstance(intervention, dict):
            return instance_prompt, domain_prompt, {}
        mode = str(intervention.get("mode", ""))
        batch_size = int(instance_prompt.shape[0])
        stats: Dict[str, torch.Tensor] = {
            "prompt_output_intervention_applied": instance_prompt.new_ones(batch_size),
        }
        if mode == "instance_prompt_zero":
            stats["instance_prompt_norm_before"] = instance_prompt.norm(dim=-1).mean(dim=1)
            instance_prompt = torch.zeros_like(instance_prompt)
        elif mode == "domain_prompt_zero":
            stats["domain_prompt_norm_before"] = domain_prompt.norm(dim=-1).mean(dim=1)
            domain_prompt = torch.zeros_like(domain_prompt)
        elif mode == "both_prompt_zero":
            stats["instance_prompt_norm_before"] = instance_prompt.norm(dim=-1).mean(dim=1)
            stats["domain_prompt_norm_before"] = domain_prompt.norm(dim=-1).mean(dim=1)
            instance_prompt = torch.zeros_like(instance_prompt)
            domain_prompt = torch.zeros_like(domain_prompt)
        elif mode == "instance_prompt_swap":
            permutation = intervention.get("permutation")
            if not torch.is_tensor(permutation):
                raise ValueError("instance_prompt_swap requires a permutation tensor")
            permutation = permutation.to(device=instance_prompt.device, dtype=torch.long)
            if permutation.shape != (batch_size,):
                raise ValueError("instance_prompt_swap permutation has incompatible shape")
            if sorted(permutation.detach().cpu().tolist()) != list(range(batch_size)):
                raise ValueError("instance_prompt_swap permutation must be bijective")
            instance_prompt = instance_prompt.index_select(0, permutation)
            stats["instance_prompt_swap_fixed_point_ratio"] = (
                permutation == torch.arange(batch_size, device=permutation.device)
            ).to(instance_prompt.dtype)
        else:
            raise ValueError(f"Unsupported Prompt Distributor intervention: {mode}")
        stats["instance_prompt_norm_after"] = instance_prompt.norm(dim=-1).mean(dim=1)
        stats["domain_prompt_norm_after"] = domain_prompt.norm(dim=-1).mean(dim=1)
        return instance_prompt, domain_prompt, stats

    def prompt_from_distribution(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if mu.dim() != 2 or tuple(logvar.shape) != tuple(mu.shape) or mu.shape[1] != self.dim:
            raise ValueError(
                f"External mu/logvar must be aligned [B,{self.dim}], got {tuple(mu.shape)} and {tuple(logvar.shape)}."
            )
        logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = mu.new_zeros((mu.shape[0], self.instance_tokens, self.dim))
        if tuple(eps.shape) != (mu.shape[0], self.instance_tokens, self.dim):
            raise ValueError(
                f"External eps must be [B,{self.instance_tokens},{self.dim}], got {tuple(eps.shape)}."
            )
        instance_prompt = mu[:, None, :] + std[:, None, :] * eps
        if self.use_slot_embed:
            instance_prompt = instance_prompt + self.slot_embed.to(device=mu.device, dtype=mu.dtype)
        domain_prompt = self.domain_prompt.to(device=mu.device, dtype=mu.dtype).expand(mu.shape[0], -1, -1)
        instance_prompt, domain_prompt, intervention_stats = (
            self._apply_prompt_output_intervention(instance_prompt, domain_prompt)
        )
        prompt_tokens = torch.cat((instance_prompt, domain_prompt), dim=1)
        if tuple(prompt_tokens.shape) != (mu.shape[0], self.prompt_len, self.dim):
            raise ValueError(
                f"External prompt_tokens must be [B,{self.prompt_len},{self.dim}], got {tuple(prompt_tokens.shape)}."
            )
        stats = {
            "visual_source": "external_distribution",
            "visual_input": mu,
            "visual_input_shape": tuple(mu.shape),
            "stats_out": torch.cat((mu, logvar), dim=-1),
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "instance_prompt": instance_prompt,
            "domain_prompt": domain_prompt,
            "prompt_tokens": prompt_tokens,
        }
        stats.update(intervention_stats)
        return prompt_tokens, stats

    def _debug_shapes(
        self,
        raw_image: Optional[torch.Tensor],
        vit_patch_tokens: Optional[torch.Tensor],
        vit_image_tokens: Optional[torch.Tensor],
        vit_cls: Optional[torch.Tensor],
        visual_input: torch.Tensor,
        stats_out: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        std: torch.Tensor,
        instance_prompt: torch.Tensor,
        domain_prompt: torch.Tensor,
        prompt_tokens: torch.Tensor,
    ) -> None:
        """按需打印一次 distributor 的关键张量形状。"""
        if not self.debug_distributor_shapes or self._debug_shapes_logged:
            return
        print(
            "[PROMPT-DIST-SHAPE] source={} raw_image={} vit_patch_tokens={} vit_image_tokens={} vit_cls={} "
            "visual_input={} stats_out={} mu={} logvar={} std={} instance_prompt={} domain_prompt={} prompt_tokens={}".format(
                self.source,
                tuple(raw_image.shape) if torch.is_tensor(raw_image) else None,
                tuple(vit_patch_tokens.shape) if torch.is_tensor(vit_patch_tokens) else None,
                tuple(vit_image_tokens.shape) if torch.is_tensor(vit_image_tokens) else None,
                tuple(vit_cls.shape) if torch.is_tensor(vit_cls) else None,
                tuple(visual_input.shape),
                tuple(stats_out.shape),
                tuple(mu.shape),
                tuple(logvar.shape),
                tuple(std.shape),
                tuple(instance_prompt.shape),
                tuple(domain_prompt.shape),
                tuple(prompt_tokens.shape),
            )
        )
        self._debug_shapes_logged = True

    def distribution_parameters(
        self,
        raw_image: Optional[torch.Tensor] = None,
        vit_patch_tokens: Optional[torch.Tensor] = None,
        vit_image_tokens: Optional[torch.Tensor] = None,
        vit_cls: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return deterministic distribution parameters without sampling Prompt.

        The deterministic Deep Prompt residual uses this interface so a zero
        residual gate neither samples Prompt nor consumes RNG state relative to
        the static A2 baseline.
        """
        visual_input, stats_out = self._encode_visual(
            raw_image, vit_patch_tokens, vit_image_tokens, vit_cls
        )
        if tuple(stats_out.shape) != (visual_input.shape[0], self.dim * 2):
            raise ValueError(
                f"stats_out must be [B,{self.dim * 2}], got {tuple(stats_out.shape)}"
            )
        mu, logvar = stats_out.chunk(2, dim=-1)
        logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        std = torch.exp(0.5 * logvar)
        return {
            "visual_source": self.source,
            "visual_input": visual_input,
            "visual_input_shape": tuple(visual_input.shape),
            "stats_out": stats_out,
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "sampling_performed": False,
        }

    def forward(
        self,
        raw_image: Optional[torch.Tensor] = None,
        vit_patch_tokens: Optional[torch.Tensor] = None,
        vit_image_tokens: Optional[torch.Tensor] = None,
        vit_cls: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        统一 forward 入口。

        不接收 label，不计算 loss，只负责生成 prompt 和缓存 stats。
        KL / GraphProbPrior 等约束在 trainer/loss 侧读取 stats 后计算。
        """
        # 1. 先根据 DISTRIBUTOR.SOURCE 选择视觉统计输入，并通过 stats_head 得到分布参数。
        #   - vit_cls_prepass: 使用 ViT prepass 得到的 CLS 表征；
        #   - token_mlp: 使用 ViT image tokens；
        #   - cnn_torchvision / clip_frozen / dinov2_small: 使用 raw_image 经过冻结外部编码器。
        # _encode_visual 会把这些来源统一成：
        #   visual_input: stats_head 的真实输入，形状由 source 决定；
        #   stats_out:    [B, 2*768]，前 768 维是 mu，后 768 维是 logvar。
        parameter_stats = self.distribution_parameters(
            raw_image=raw_image,
            vit_patch_tokens=vit_patch_tokens,
            vit_image_tokens=vit_image_tokens,
            vit_cls=vit_cls,
        )
        visual_input = parameter_stats["visual_input"]
        stats_out = parameter_stats["stats_out"]
        mu = parameter_stats["mu"]
        logvar = parameter_stats["logvar"]
        std = parameter_stats["std"]

        # 3. 使用完整 768 维 q(z|x) 生成图像条件 instance prompt。
        instance_prompt = self._sample_instance_prompt(mu, std)

        # 4. 构造 domain prompt。
        # domain_prompt 是任务/数据集级可学习参数，不依赖单张图像。
        # 参数本体形状是 [1, domain_tokens, 768]；这里 expand 到 batch 维，得到 [B, domain_tokens, 768]。
        domain_prompt = self.domain_prompt.to(device=mu.device, dtype=mu.dtype).expand(mu.shape[0], -1, -1)

        instance_prompt, domain_prompt, intervention_stats = (
            self._apply_prompt_output_intervention(instance_prompt, domain_prompt)
        )

        # token 顺序固定为 [instance prompt | domain prompt]，总长度必须等于 MODEL.PROMPT.NUM_TOKENS。
        # instance prompt 放前面，domain prompt 放后面；这个顺序会影响后续 ViT 中 prompt token 的位置。
        prompt_tokens = torch.cat((instance_prompt, domain_prompt), dim=1)

        # 最终 prompt_tokens 是真正送回 PromptedTransformer、插入 ViT 的提示 token。
        # 形状必须是 [B, NUM_TOKENS, 768]。
        if tuple(prompt_tokens.shape) != (mu.shape[0], self.prompt_len, self.dim):
            raise ValueError(f"prompt_tokens must be [B,{self.prompt_len},{self.dim}], got {tuple(prompt_tokens.shape)}")

        # 5. 如果开启 DEBUG_DISTRIBUTOR_SHAPES，只打印一次关键 shape。
        # 这不参与训练，只用于排查 source、stats_head、prompt 拼接是否对齐。
        self._debug_shapes(
            raw_image,
            vit_patch_tokens,
            vit_image_tokens,
            vit_cls,
            visual_input,
            stats_out,
            mu,
            logvar,
            std,
            instance_prompt,
            domain_prompt,
            prompt_tokens,
        )

        # stats 会被 PromptedTransformer 缓存，再由 trainer/loss 读取。
        #
        # 这里返回的 stats 有两个用途：
        #   1. PromptKLAuxLoss / GraphProbPriorAuxLoss 读取 mu/logvar；
        #   2. debug、可视化或监控读取 visual_input、prompt_tokens 和 posterior stats。
        #
        # 注意：forward 本身不接收 targets，也不在这里计算任何 loss。
        # label 相关信息由 trainer 在 loss_kwargs 中单独传给 loss。
        stats = {
            "visual_source": self.source,
            "visual_input": visual_input,
            "visual_input_shape": tuple(visual_input.shape),
            "stats_out": stats_out,
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "instance_prompt": instance_prompt,
            "domain_prompt": domain_prompt,
            "prompt_tokens": prompt_tokens,
            "sampling_performed": True,
        }
        stats.update(intervention_stats)

        # 返回值一：
        #   prompt_tokens: 立即送回 ViT 主干使用。
        # 返回值二：
        #   stats: 缓存在 transformer/model 上，供 trainer/loss 在本次 forward 后读取。
        return prompt_tokens, stats


def generate_prompt_init(
    distributor: PreViTPromptDistributor,
    V_raw: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    兼容旧 one-shot prompt 初始化接口。

    新训练主线不再依赖旧 PromptGenerator 大解码器；这个 helper 只用于少量旧脚本
    需要从 distributor 生成一个静态初始化 prompt 的情况。
    """
    with torch.no_grad():
        prompts, _ = distributor(vit_image_tokens=V_raw)
        if reduce == "first":
            prompts = prompts[:1]
        elif reduce == "mean":
            prompts = prompts.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
        return prompts.detach()


__all__ = [
    "PreViTPromptDistributor",
    "MeanConditionedDeepPromptResidual",
    "prompt_kl_loss",
    "generate_prompt_init",
]
