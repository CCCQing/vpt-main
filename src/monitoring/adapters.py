"""Convert existing runtime caches into detached scalar monitor dictionaries."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


def _finite_number(value: Any) -> Optional[float]:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def scalar_metrics(values: Mapping[str, Any], *, prefix: str = "") -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name, value in values.items():
        scalar = _finite_number(value)
        if scalar is not None:
            result[f"{prefix}{name}"] = scalar
    return result


def _tensor_distribution(prefix: str, tensor: Any) -> Dict[str, float]:
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return {}
    with torch.no_grad():
        value = tensor.detach().float()
        metrics = {
            f"{prefix}_mean": float(value.mean().item()),
            f"{prefix}_std": float(value.std(unbiased=False).item()),
            f"{prefix}_abs_mean": float(value.abs().mean().item()),
        }
        if value.dim() >= 1:
            metrics[f"{prefix}_norm_mean"] = float(value.reshape(value.shape[0], -1).norm(dim=1).mean().item())
    return metrics


def _effective_rank(prefix: str, tensor: Any) -> Dict[str, float]:
    if not torch.is_tensor(tensor) or tensor.numel() == 0 or tensor.dim() < 2:
        return {}
    with torch.no_grad():
        value = tensor.detach().float().reshape(-1, tensor.shape[-1])
        if value.shape[0] < 2:
            return {f"{prefix}_effective_rank": 1.0}
        value = value - value.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(value)
        total = singular.sum()
        if float(total.item()) <= 1e-12:
            rank = 0.0
        else:
            prob = singular / total
            rank = float(torch.exp(-(prob * prob.clamp_min(1e-12).log()).sum()).item())
        return {f"{prefix}_effective_rank": rank}


def prompt_distribution_metrics(stats: Any) -> Dict[str, float]:
    if not isinstance(stats, Mapping):
        return {}
    result: Dict[str, float] = {}
    tensor_fields = (
        "visual_input", "mu", "logvar", "std", "instance_prompt", "domain_prompt",
        "prompt_tokens", "semantic_component", "variation_component",
    )
    for field in tensor_fields:
        result.update(_tensor_distribution(field, stats.get(field)))
    mu = stats.get("mu")
    logvar = stats.get("logvar")
    prompt_tokens = stats.get("prompt_tokens")
    if torch.is_tensor(mu) and torch.is_tensor(logvar) and mu.shape == logvar.shape and mu.numel() > 0:
        with torch.no_grad():
            mu_value = mu.detach().float()
            logvar_value = logvar.detach().float()
            kl = 0.5 * (logvar_value.exp() + mu_value.pow(2) - 1.0 - logvar_value)
            if mu_value.dim() >= 2:
                instance_variance = mu_value.reshape(mu_value.shape[0], -1).var(dim=0, unbiased=False)
                active = instance_variance > 1e-4
                result["mu_between_instance_variance"] = float(instance_variance.mean().item())
                result["active_latent_ratio"] = float(active.float().mean().item())
            result["kl_to_prior"] = float(kl.reshape(kl.shape[0], -1).sum(dim=1).mean().item()) if kl.dim() >= 2 else float(kl.mean().item())
            result["kl_per_latent_dim"] = float(kl.mean().item())
            noise_variance = float(logvar_value.exp().mean().item())
            instance_variance_mean = float(result.get("mu_between_instance_variance", 0.0))
            result["instance_to_noise_variance_ratio"] = instance_variance_mean / max(noise_variance, 1e-12)
    result.update(_effective_rank("generated_prompt", prompt_tokens))
    return result


def semantic_token_metrics(state: Any) -> Dict[str, float]:
    if not isinstance(state, Mapping):
        return {}
    result: Dict[str, float] = {}
    for name, value in state.items():
        if torch.is_tensor(value):
            result.update(_tensor_distribution(str(name), value))
            if value.dim() >= 2:
                result.update(_effective_rank(str(name), value))
    semantic_input = state.get("semantic_input")
    semantic_output = state.get("semantic_output")
    if torch.is_tensor(semantic_input) and torch.is_tensor(semantic_output) and semantic_input.shape == semantic_output.shape:
        with torch.no_grad():
            left = semantic_input.detach().float().reshape(-1, semantic_input.shape[-1])
            right = semantic_output.detach().float().reshape(-1, semantic_output.shape[-1])
            cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
            result["input_output_identity_cosine"] = float(cosine.mean().item())
            delta = right - left
            result["semantic_token_delta_norm"] = float(delta.norm(dim=-1).mean().item())
    return result


def auxiliary_loss_metrics(loss_stats: Any) -> Dict[str, float]:
    if not isinstance(loss_stats, Mapping):
        return {}
    result: Dict[str, float] = {}
    main_loss = None
    for candidate in ("main_loss", "ce_loss", "classification_loss", "primary_ce_loss"):
        scalar = _finite_number(loss_stats.get(candidate))
        if scalar is not None:
            main_loss = scalar
            break
    for name, value in loss_stats.items():
        key = str(name)
        lowered = key.lower()
        if lowered.startswith("graph_prob_prior_") or "loss" not in lowered:
            continue
        scalar = _finite_number(value)
        if scalar is None:
            continue
        result[key] = scalar
        if main_loss is not None and key not in {"main_loss", "ce_loss", "classification_loss", "primary_ce_loss"}:
            result[f"{key}.to_primary_ce_ratio"] = scalar / max(abs(main_loss), 1e-12)
        result[f"{key}.is_zero"] = float(abs(scalar) <= 1e-12)
    if result:
        result["finite_ratio"] = 1.0
    return result


def train_debug_metrics(debug: Any) -> Dict[str, float]:
    if not isinstance(debug, Mapping):
        return {}
    result: Dict[str, float] = {}
    for name, value in debug.items():
        if str(name).startswith("graph_prob_prior_"):
            continue
        if isinstance(value, Mapping):
            result.update(scalar_metrics(value, prefix=f"{name}."))
        else:
            result.update(scalar_metrics({str(name): value}))
    return result


def graph_prob_prior_metrics(loss_stats: Any) -> Dict[str, float]:
    if not isinstance(loss_stats, Mapping):
        return {}
    prefix = "graph_prob_prior_"
    return {
        str(name)[len(prefix):]: scalar
        for name, value in loss_stats.items()
        if str(name).startswith(prefix)
        for scalar in [_finite_number(value)]
        if scalar is not None
    }


def attention_mediation_metrics(layer_stats: Any) -> Dict[str, float]:
    if not isinstance(layer_stats, Iterable):
        return {}
    buckets: Dict[str, list] = {}
    for item in layer_stats:
        if not isinstance(item, Mapping):
            continue
        for name, value in item.items():
            scalar = _finite_number(value)
            if scalar is not None:
                buckets.setdefault(str(name), []).append(scalar)
    result = {
        f"{name}_mean": float(sum(values) / len(values))
        for name, values in buckets.items()
        if values
    }
    for gamma_name in ("prompt_gamma", "semantic_gamma"):
        values = buckets.get(gamma_name, [])
        if values:
            tensor = torch.as_tensor(values, dtype=torch.float32)
            result[f"{gamma_name}_std"] = float(tensor.std(unbiased=False).item())
            result[f"{gamma_name}_near_zero_ratio"] = float((tensor.abs() <= 1e-6).float().mean().item())
            result[f"{gamma_name}_saturation_ratio"] = float((tensor.abs() >= 1.0).float().mean().item())
    return result


def affinity_metrics(affinities: Any) -> Dict[str, float]:
    if not isinstance(affinities, Iterable):
        return {}
    buckets: Dict[str, list] = {}
    for layer in affinities:
        if not isinstance(layer, Mapping):
            continue
        for name, value in layer.items():
            if not str(name).endswith("_raw") or not torch.is_tensor(value) or value.numel() == 0:
                continue
            with torch.no_grad():
                data = value.detach().float()
                buckets.setdefault(f"{name}.mean", []).append(float(data.mean().item()))
                buckets.setdefault(f"{name}.std", []).append(float(data.std(unbiased=False).item()))
                buckets.setdefault(f"{name}.abs_mean", []).append(float(data.abs().mean().item()))
    return {
        f"{name}_layers_mean": float(sum(values) / len(values))
        for name, values in buckets.items()
        if values
    }
