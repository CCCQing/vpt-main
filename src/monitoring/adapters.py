"""Convert existing runtime caches into detached scalar monitor dictionaries."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
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
    if isinstance(stats.get("sampling_performed"), bool):
        result["sampling_performed"] = float(stats["sampling_performed"])
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


def deep_prompt_residual_metrics(trace: Any) -> Dict[str, float]:
    """Summarize the per-layer deterministic residual without storing samples."""
    if not isinstance(trace, (list, tuple)):
        return {}
    result: Dict[str, float] = {}
    ratios = []
    gates = []
    sample_gates = []
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        layer_id = int(item.get("layer_id", -1))
        if layer_id < 0:
            continue
        prefix = f"layer_{layer_id}"
        base = item.get("base_prompt")
        raw = item.get("raw_delta")
        applied = item.get("applied_delta")
        gate = item.get("gate")
        sample_gate = item.get("sample_gate")
        layer_gate = item.get("layer_gate")
        runtime_scale = item.get("runtime_scale")
        applied_ratio_tensor = item.get("applied_ratio")
        budget_exceed = item.get("budget_exceed")
        active_layer = item.get("active_layer")
        if torch.is_tensor(base):
            base_norm = float(base.detach().float().norm(dim=-1).mean().item())
            result[f"{prefix}.base_prompt_norm"] = base_norm
        else:
            base_norm = 0.0
        if torch.is_tensor(raw):
            result[f"{prefix}.raw_delta_norm"] = float(
                raw.detach().float().norm(dim=-1).mean().item()
            )
            if raw.dim() == 3 and raw.shape[1] > 0:
                result[f"{prefix}.raw_delta_slot_variance"] = float(
                    raw.detach().float().var(dim=1, unbiased=False).mean().item()
                )
                raw_value = raw.detach().float()
                if str(item.get("content_mode", "shared")) == "shared":
                    effective_rank = (
                        raw_value.norm(dim=(-2, -1)) > 1.0e-12
                    ).float()
                else:
                    gram = torch.matmul(raw_value, raw_value.transpose(-1, -2))
                    eigenvalues = (
                        torch.linalg.eigvalsh(gram)
                        if hasattr(torch.linalg, "eigvalsh")
                        else torch.symeig(gram, eigenvectors=False).eigenvalues
                    )
                    singular = eigenvalues.clamp_min(0.0).sqrt()
                    singular = torch.where(
                        singular
                        > singular.max(dim=-1, keepdim=True).values.clamp_min(1.0e-12)
                        * 1.0e-3,
                        singular,
                        torch.zeros_like(singular),
                    )
                    singular_total = singular.sum(dim=-1, keepdim=True)
                    probability = singular / singular_total.clamp_min(1.0e-12)
                    effective_rank = torch.exp(
                        -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
                    )
                    effective_rank = torch.where(
                        singular_total.squeeze(-1) > 1.0e-12,
                        effective_rank,
                        torch.zeros_like(effective_rank),
                    )
                result[f"{prefix}.raw_delta_slot_effective_rank"] = float(
                    effective_rank.mean().item()
                )
            if raw.shape[0] > 1:
                result[f"{prefix}.raw_delta_between_instance_variance"] = float(
                    raw.detach().float().var(dim=0, unbiased=False).mean().item()
                )
        if torch.is_tensor(applied):
            applied_norm = float(
                applied.detach().float().norm(dim=-1).mean().item()
            )
            result[f"{prefix}.applied_delta_norm"] = applied_norm
            ratio = applied_norm / max(base_norm, 1.0e-12)
            result[f"{prefix}.applied_delta_to_base_ratio"] = ratio
            ratios.append(ratio)
        if torch.is_tensor(applied_ratio_tensor):
            ratio_value = applied_ratio_tensor.detach().float().reshape(-1)
            if ratio_value.numel() > 0:
                result[f"{prefix}.applied_ratio_mean"] = float(
                    ratio_value.mean().item()
                )
                result[f"{prefix}.applied_ratio_p90"] = float(
                    torch.quantile(ratio_value, 0.9).item()
                )
                result[f"{prefix}.applied_ratio_max"] = float(
                    ratio_value.max().item()
                )
        if (
            str(item.get("amplitude_mode", "legacy_gate")) == "bounded_ratio"
            and torch.is_tensor(budget_exceed)
        ):
            result[f"{prefix}.budget_exceed_rate"] = float(
                budget_exceed.detach().float().mean().item()
            )
        if torch.is_tensor(active_layer):
            result[f"{prefix}.active_layer"] = float(
                active_layer.detach().float().mean().item()
            )
        if torch.is_tensor(gate):
            gate_value = float(gate.detach().float().mean().item())
            result[f"{prefix}.gate"] = gate_value
            gates.append(gate_value)
        if torch.is_tensor(layer_gate):
            result[f"{prefix}.layer_gate"] = float(
                layer_gate.detach().float().mean().item()
            )
        if torch.is_tensor(sample_gate):
            sample_value = sample_gate.detach().float()
            result[f"{prefix}.sample_gate_mean"] = float(sample_value.mean().item())
            result[f"{prefix}.sample_gate_std"] = float(
                sample_value.std(unbiased=False).item()
            )
            result[f"{prefix}.sample_gate_low_saturation_rate"] = float(
                (sample_value <= 0.05).float().mean().item()
            )
            result[f"{prefix}.sample_gate_high_saturation_rate"] = float(
                (sample_value >= 0.95).float().mean().item()
            )
            sample_gates.extend(sample_value.cpu().tolist())
        if torch.is_tensor(runtime_scale):
            result[f"{prefix}.runtime_scale"] = float(
                runtime_scale.detach().float().mean().item()
            )
    if ratios:
        result["layers_mean.applied_delta_to_base_ratio"] = float(np.mean(ratios))
        result["layers_max.applied_delta_to_base_ratio"] = float(np.max(ratios))
    if gates:
        result["layers_mean.gate"] = float(np.mean(gates))
        result["layers_max_abs.gate"] = float(np.max(np.abs(gates)))
    if sample_gates:
        result["layers_samples.sample_gate_mean"] = float(np.mean(sample_gates))
        result["layers_samples.sample_gate_std"] = float(np.std(sample_gates))
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


def loss_component_metrics(loss_stats: Any) -> Dict[str, float]:
    if not isinstance(loss_stats, Mapping):
        return {}
    result: Dict[str, float] = {}
    for name, value in loss_stats.items():
        key = str(name)
        if key != "total_loss" and not key.endswith(
            (".raw", ".weight", ".weighted", ".weighted_share")
        ):
            continue
        scalar = _finite_number(value)
        if scalar is not None:
            result[key] = scalar
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
