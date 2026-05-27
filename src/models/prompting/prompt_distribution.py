#!/usr/bin/env python3
"""Prompt distribution modules used before prompted ViT.

The public class name is intentionally kept as ``PreViTPromptDistributor`` so
the existing backbone builder can keep the same construction entry.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple

import torch
from torch import nn


_ALLOWED_SOURCES = {
    "vit_cls_prepass",
    "cnn_torchvision",
    "clip_frozen",
    "dinov2_small",
    "token_mlp",
}


def prompt_kl_loss(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) for diagonal Gaussian prompt statistics."""
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    if reduction == "none":
        return kl
    raise ValueError(f"Unsupported prompt KL reduction: {reduction}")


class _VectorStatsHead(nn.Module):
    """Map one visual vector [B,D_in] to Gaussian stats [B,2*D]."""

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
    """ViaPT-style lightweight token MLP: [B,N,768] -> [B,2*768]."""

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
    """Frozen torchvision feature extractor with explicit local-cache semantics."""

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
    """Strict local image encoder loader for CLIP-like or DINO-like frozen modules."""

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
    """Generate ViaPT-style instance/domain prompt tokens from a visual source."""

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
        debug_preprocess_shapes: bool = False,
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
        self.debug_preprocess_shapes = bool(debug_preprocess_shapes)
        self._debug_shapes_logged = False

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
        if self.logvar_min > self.logvar_max:
            raise ValueError("DISTRIBUTOR.LOGVAR_MIN must be <= LOGVAR_MAX.")

        self.frozen_encoder = None
        self.stats_head = None
        if self.source == "vit_cls_prepass":
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

        self.domain_prompt = nn.Parameter(torch.zeros(1, self.domain_tokens, self.dim))
        nn.init.normal_(self.domain_prompt, mean=0.0, std=0.02)
        if self.use_slot_embed:
            self.slot_embed = nn.Parameter(torch.zeros(1, self.instance_tokens, self.dim))
            nn.init.normal_(self.slot_embed, mean=0.0, std=0.02)
        self.register_buffer("fixed_eps", torch.randn(1, self.instance_tokens, self.dim))
        self._freeze_external_encoder()

    def _freeze_external_encoder(self) -> None:
        if self.frozen_encoder is None:
            return
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
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
        if self.source == "vit_cls_prepass":
            if vit_cls is None:
                raise ValueError("SOURCE='vit_cls_prepass' requires vit_cls from PromptedTransformer prepass.")
            visual_input = vit_cls
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
        if not self.debug_preprocess_shapes or self._debug_shapes_logged:
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

    def forward(
        self,
        raw_image: Optional[torch.Tensor] = None,
        vit_patch_tokens: Optional[torch.Tensor] = None,
        vit_image_tokens: Optional[torch.Tensor] = None,
        vit_cls: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        visual_input, stats_out = self._encode_visual(raw_image, vit_patch_tokens, vit_image_tokens, vit_cls)
        if tuple(stats_out.shape) != (visual_input.shape[0], self.dim * 2):
            raise ValueError(f"stats_out must be [B,{self.dim * 2}], got {tuple(stats_out.shape)}")

        mu, logvar = stats_out.chunk(2, dim=-1)
        logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        std = torch.exp(0.5 * logvar)
        instance_prompt = self._sample_instance_prompt(mu, std)
        domain_prompt = self.domain_prompt.to(device=mu.device, dtype=mu.dtype).expand(mu.shape[0], -1, -1)
        prompt_tokens = torch.cat((instance_prompt, domain_prompt), dim=1)
        if tuple(prompt_tokens.shape) != (mu.shape[0], self.prompt_len, self.dim):
            raise ValueError(f"prompt_tokens must be [B,{self.prompt_len},{self.dim}], got {tuple(prompt_tokens.shape)}")

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
        }
        return prompt_tokens, stats


def generate_prompt_init(
    distributor: PreViTPromptDistributor,
    V_raw: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Compatibility helper for one-shot prompt initialization from ViT tokens."""
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
    "prompt_kl_loss",
    "generate_prompt_init",
]
