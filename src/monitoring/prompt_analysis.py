from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _finite_float_tensor(value: Any) -> Optional[torch.Tensor]:
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    data = value.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(data).all()):
        return None
    return data


class _VectorDecompositionState:
    def __init__(self) -> None:
        self.count = 0
        self.feature_sum: Optional[torch.Tensor] = None
        self.square_sum = 0.0
        self.class_sum: Dict[int, torch.Tensor] = {}
        self.class_count: Dict[int, int] = defaultdict(int)

    def update(self, values: Any, targets: Any) -> None:
        data = _finite_float_tensor(values)
        target = torch.as_tensor(targets, dtype=torch.long).reshape(-1).cpu()
        if data is None or data.shape[0] != target.numel():
            return
        data = data.reshape(data.shape[0], -1).to(torch.float64)
        if self.feature_sum is None:
            self.feature_sum = torch.zeros(data.shape[1], dtype=torch.float64)
        if self.feature_sum.numel() != data.shape[1]:
            raise ValueError("Prompt source decomposition feature shape changed")
        self.feature_sum += data.sum(dim=0)
        self.square_sum += float(data.square().sum().item())
        self.count += int(data.shape[0])
        for class_id in torch.unique(target).tolist():
            mask = target == int(class_id)
            class_values = data[mask]
            if int(class_id) not in self.class_sum:
                self.class_sum[int(class_id)] = torch.zeros(
                    data.shape[1], dtype=torch.float64
                )
            self.class_sum[int(class_id)] += class_values.sum(dim=0)
            self.class_count[int(class_id)] += int(class_values.shape[0])

    def finalize(self, prefix: str) -> Dict[str, float]:
        if self.count <= 0 or self.feature_sum is None:
            return {}
        mean = self.feature_sum / self.count
        total_ss = max(
            0.0,
            self.square_sum - self.count * float(mean.square().sum().item()),
        )
        between_ss = 0.0
        for class_id, class_sum in self.class_sum.items():
            count = self.class_count[class_id]
            class_mean = class_sum / max(1, count)
            between_ss += count * float((class_mean - mean).square().sum().item())
        between_ss = min(total_ss, max(0.0, between_ss))
        within_ss = max(0.0, total_ss - between_ss)
        feature_count = int(mean.numel())
        denominator = max(1, self.count * feature_count)
        return {
            f"{prefix}_sample_count": float(self.count),
            f"{prefix}_observed_class_count": float(len(self.class_count)),
            f"{prefix}_global_common_norm": float(mean.norm().item()),
            f"{prefix}_total_variance": float(total_ss / denominator),
            f"{prefix}_class_component_variance": float(between_ss / denominator),
            f"{prefix}_instance_residual_variance": float(within_ss / denominator),
            f"{prefix}_class_component_variance_share": float(
                between_ss / max(total_ss, 1e-12)
            ),
            f"{prefix}_instance_residual_variance_share": float(
                within_ss / max(total_ss, 1e-12)
            ),
        }


class _StreamingCorrelationState:
    def __init__(self) -> None:
        self.count = 0
        self.x_sum = 0.0
        self.y_sum = 0.0
        self.xx_sum = 0.0
        self.yy_sum = 0.0
        self.xy_sum = 0.0

    def update(self, left: Any, right: Any) -> None:
        x = _finite_float_tensor(left)
        y = _finite_float_tensor(right)
        if x is None or y is None:
            return
        x = x.reshape(-1).to(torch.float64)
        y = y.reshape(-1).to(torch.float64)
        if x.numel() != y.numel():
            return
        self.count += int(x.numel())
        self.x_sum += float(x.sum().item())
        self.y_sum += float(y.sum().item())
        self.xx_sum += float(x.square().sum().item())
        self.yy_sum += float(y.square().sum().item())
        self.xy_sum += float((x * y).sum().item())

    def finalize(self) -> Optional[float]:
        if self.count < 2:
            return None
        numerator = self.xy_sum - self.x_sum * self.y_sum / self.count
        x_ss = self.xx_sum - self.x_sum * self.x_sum / self.count
        y_ss = self.yy_sum - self.y_sum * self.y_sum / self.count
        denominator = math.sqrt(max(0.0, x_ss) * max(0.0, y_ss))
        if denominator <= 1e-12:
            return None
        return float(numerator / denominator)


class PromptSourceDecompositionAccumulator:
    def __init__(
        self,
        *,
        instance_tokens: int,
        domain_tokens: int,
        contextualized_domain_applicable: bool = True,
    ) -> None:
        self.instance_tokens = int(instance_tokens)
        self.domain_tokens = int(domain_tokens)
        self.contextualized_domain_applicable = bool(
            contextualized_domain_applicable
        )
        self.instance = _VectorDecompositionState()
        self.contextualized_domain = _VectorDecompositionState()
        self.domain_norm_sum = 0.0
        self.domain_sample_count = 0
        self.domain_replication_max_abs_error = 0.0
        self.domain_reference: Optional[torch.Tensor] = None
        self.raw_observed = False
        self.contextualized_observed = False
        self.instance_cls_correlation = _StreamingCorrelationState()
        self.instance_margin_correlation = _StreamingCorrelationState()
        self.contextualized_domain_cls_correlation = (
            _StreamingCorrelationState()
        )
        self.contextualized_domain_margin_correlation = (
            _StreamingCorrelationState()
        )

    def update_raw(
        self,
        stats: Any,
        targets: Any,
        *,
        cls_repr: Any = None,
        true_margin: Any = None,
    ) -> None:
        if not isinstance(stats, Mapping):
            return
        instance = _finite_float_tensor(stats.get("instance_prompt"))
        domain = _finite_float_tensor(stats.get("domain_prompt"))
        if instance is not None and self.instance_tokens > 0:
            self.instance.update(instance, targets)
            instance_norm = instance.reshape(instance.shape[0], -1).norm(dim=1)
            cls_values = _finite_float_tensor(cls_repr)
            if cls_values is not None and cls_values.shape[0] == instance.shape[0]:
                self.instance_cls_correlation.update(
                    instance_norm,
                    cls_values.reshape(cls_values.shape[0], -1).norm(dim=1),
                )
            self.instance_margin_correlation.update(
                instance_norm, true_margin
            )
            self.raw_observed = True
        if domain is None or self.domain_tokens <= 0 or domain.dim() != 3:
            return
        flattened = domain.reshape(domain.shape[0], -1)
        reference = flattened[:1]
        self.domain_replication_max_abs_error = max(
            self.domain_replication_max_abs_error,
            float((flattened - reference).abs().max().item()),
        )
        if self.domain_reference is None:
            self.domain_reference = reference.squeeze(0).clone()
        else:
            self.domain_replication_max_abs_error = max(
                self.domain_replication_max_abs_error,
                float((reference.squeeze(0) - self.domain_reference).abs().max().item()),
            )
        norms = flattened.norm(dim=1)
        self.domain_norm_sum += float(norms.sum().item())
        self.domain_sample_count += int(norms.numel())
        self.raw_observed = True

    def update_contextualized(
        self,
        token_sequence: Any,
        targets: Any,
        *,
        prompt_length: int,
        semantic_length: int,
        cls_repr: Any = None,
        true_margin: Any = None,
    ) -> None:
        tokens = _finite_float_tensor(token_sequence)
        if (
            tokens is None
            or tokens.dim() != 3
            or self.domain_tokens <= 0
            or not self.contextualized_domain_applicable
            or int(prompt_length) < self.instance_tokens + self.domain_tokens
        ):
            return
        prompt_start = 1
        domain_start = prompt_start + self.instance_tokens
        domain_end = domain_start + self.domain_tokens
        patch_end = int(tokens.shape[1]) - int(semantic_length)
        if domain_end > patch_end:
            return
        domain_tokens = tokens[:, domain_start:domain_end, :]
        self.contextualized_domain.update(domain_tokens, targets)
        domain_norm = domain_tokens.reshape(domain_tokens.shape[0], -1).norm(dim=1)
        cls_values = _finite_float_tensor(cls_repr)
        if cls_values is not None and cls_values.shape[0] == domain_tokens.shape[0]:
            self.contextualized_domain_cls_correlation.update(
                domain_norm,
                cls_values.reshape(cls_values.shape[0], -1).norm(dim=1),
            )
        self.contextualized_domain_margin_correlation.update(
            domain_norm, true_margin
        )
        self.contextualized_observed = True

    def finalize(self) -> Dict[str, Any]:
        metrics: Dict[str, float] = {}
        metrics.update(self.instance.finalize("raw_instance_prompt"))
        metrics.update(
            self.contextualized_domain.finalize("contextualized_domain_prompt")
        )
        for name, state in (
            (
                "raw_instance_prompt_norm_vs_cls_norm_pearson",
                self.instance_cls_correlation,
            ),
            (
                "raw_instance_prompt_norm_vs_true_margin_pearson",
                self.instance_margin_correlation,
            ),
            (
                "contextualized_domain_prompt_norm_vs_cls_norm_pearson",
                self.contextualized_domain_cls_correlation,
            ),
            (
                "contextualized_domain_prompt_norm_vs_true_margin_pearson",
                self.contextualized_domain_margin_correlation,
            ),
        ):
            value = state.finalize()
            if value is not None:
                metrics[name] = value
        if self.domain_sample_count > 0:
            metrics.update({
                "raw_domain_prompt_parameter_norm": float(
                    self.domain_norm_sum / self.domain_sample_count
                ),
                "raw_domain_prompt_replication_max_abs_error": float(
                    self.domain_replication_max_abs_error
                ),
                "raw_domain_prompt_replication_pass": float(
                    self.domain_replication_max_abs_error <= 1e-7
                ),
                "raw_domain_prompt_sample_variance_by_construction": 0.0,
                "raw_domain_prompt_class_variance_by_construction": 0.0,
            })
        return {
            "format": "prompt_source_decomposition_v1",
            "applicability": (
                "applicable" if self.raw_observed else "not_applicable_no_distributor_state"
            ),
            "raw_observed": bool(self.raw_observed),
            "contextualized_domain_observed": bool(self.contextualized_observed),
            "contextualized_domain_applicability": (
                "applicable"
                if self.contextualized_domain_applicable
                else "not_applicable_deep_prompt_replacement_breaks_slot_identity"
            ),
            "instance_tokens": self.instance_tokens,
            "domain_tokens": self.domain_tokens,
            "metrics": metrics,
        }


class PromptContentSlotAccumulator:
    def __init__(
        self,
        *,
        prompt_length: int,
        selected_layers: Sequence[int],
        instance_tokens: int = 0,
        domain_tokens: int = 0,
        redundancy_cosine: float = 0.9,
        opposition_cosine: float = -0.5,
        cancellation_ratio: float = 0.25,
        low_usage_fraction: float = 0.25,
        low_function_fraction: float = 0.25,
        low_role_coverage: float = 0.01,
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.selected_layers = {int(item) for item in selected_layers}
        self.instance_tokens = int(instance_tokens)
        self.domain_tokens = int(domain_tokens)
        self.redundancy_cosine = float(redundancy_cosine)
        self.opposition_cosine = float(opposition_cosine)
        self.cancellation_ratio = float(cancellation_ratio)
        self.low_usage_fraction = float(low_usage_fraction)
        self.low_function_fraction = float(low_function_fraction)
        self.low_role_coverage = float(low_role_coverage)
        self.states: Dict[int, Dict[str, Any]] = {}
        self.head_states: Dict[tuple[int, int], Dict[str, Any]] = {}

    @staticmethod
    def _new_pair_state() -> Dict[str, Any]:
        return {
            "pair_count": 0,
            "cosine_sum": 0.0,
            "sample_max_cosine_sum": 0.0,
            "sample_max_cosine_count": 0,
            "redundancy_count": 0,
            "opposition_count": 0,
            "cancellation_count": 0,
        }

    def _state(self, layer_index: int) -> Dict[str, Any]:
        return self.states.setdefault(
            int(layer_index),
            {
                "usage_sum": torch.zeros(self.prompt_length, dtype=torch.float64),
                "content_sum": torch.zeros(self.prompt_length, dtype=torch.float64),
                "top1_count": torch.zeros(self.prompt_length, dtype=torch.float64),
                "usage_count": 0,
                "content_count": 0,
                **self._new_pair_state(),
            },
        )

    def _head_state(self, layer_index: int, head_index: int) -> Dict[str, Any]:
        return self.head_states.setdefault(
            (int(layer_index), int(head_index)), self._new_pair_state()
        )

    def _update_content_pairs(
        self, state: Dict[str, Any], content: torch.Tensor
    ) -> None:
        if (
            content.dim() != 3
            or content.shape[1] != self.prompt_length
            or self.prompt_length < 2
        ):
            return
        norms = content.norm(dim=-1)
        unit = torch.nn.functional.normalize(content, dim=-1, eps=1e-12)
        cosine = torch.matmul(unit, unit.transpose(1, 2))
        upper = torch.triu_indices(
            self.prompt_length,
            self.prompt_length,
            offset=1,
            device=cosine.device,
        )
        pair_cosine = cosine[:, upper[0], upper[1]]
        pair_norm_sum = norms[:, upper[0]] + norms[:, upper[1]]
        pair_sum_norm = (
            content[:, upper[0], :] + content[:, upper[1], :]
        ).norm(dim=-1)
        cancellation = pair_sum_norm / pair_norm_sum.clamp_min(1e-12)
        valid = pair_norm_sum > 1e-12
        if not bool(valid.any()):
            return
        values = pair_cosine[valid]
        state["pair_count"] += int(values.numel())
        state["cosine_sum"] += float(values.sum().item())
        valid_by_sample = valid.any(dim=1)
        if bool(valid_by_sample.any()):
            sample_max = pair_cosine.masked_fill(
                ~valid, float("-inf")
            ).max(dim=1).values
            state["sample_max_cosine_sum"] += float(
                sample_max[valid_by_sample].sum().item()
            )
            state["sample_max_cosine_count"] += int(
                valid_by_sample.sum().item()
            )
        state["redundancy_count"] += int(
            ((pair_cosine >= self.redundancy_cosine) & valid).sum().item()
        )
        state["opposition_count"] += int(
            ((pair_cosine <= self.opposition_cosine) & valid).sum().item()
        )
        state["cancellation_count"] += int(
            ((cancellation <= self.cancellation_ratio) & valid).sum().item()
        )

    @staticmethod
    def _finalize_pair_state(state: Mapping[str, Any]) -> Dict[str, float]:
        if int(state["pair_count"]) <= 0:
            return {}
        pair_count = int(state["pair_count"])
        metrics = {
            "prompt_content_pairwise_cosine_mean": float(
                state["cosine_sum"] / pair_count
            ),
            "prompt_content_redundancy_candidate_ratio": float(
                state["redundancy_count"] / pair_count
            ),
            "prompt_content_opposition_candidate_ratio": float(
                state["opposition_count"] / pair_count
            ),
            "prompt_content_cancellation_candidate_ratio": float(
                state["cancellation_count"] / pair_count
            ),
        }
        if int(state["sample_max_cosine_count"]) > 0:
            metrics["prompt_content_pairwise_cosine_max_mean"] = float(
                state["sample_max_cosine_sum"]
                / state["sample_max_cosine_count"]
            )
        return metrics

    def update(
        self,
        attention_layers: Sequence[Any],
        affinity_layers: Sequence[Any],
    ) -> None:
        layer_count = max(len(attention_layers or []), len(affinity_layers or []))
        prompt_slice = slice(1, 1 + self.prompt_length)
        for layer_index in range(layer_count):
            if self.selected_layers and layer_index not in self.selected_layers:
                continue
            attention = (
                attention_layers[layer_index]
                if layer_index < len(attention_layers or [])
                and torch.is_tensor(attention_layers[layer_index])
                else None
            )
            affinity = (
                affinity_layers[layer_index]
                if layer_index < len(affinity_layers or [])
                and isinstance(affinity_layers[layer_index], Mapping)
                else {}
            )
            usage = None
            if attention is not None and attention.dim() == 4:
                usage = attention[:, :, 0, prompt_slice].detach().float().mean(dim=1)
            else:
                cls_prompt = affinity.get("AcKp_attn")
                if torch.is_tensor(cls_prompt) and cls_prompt.dim() == 4:
                    usage = cls_prompt.detach().float().mean(dim=1).squeeze(1)
            content = affinity.get("_prompt_patch_content_vector")
            content = content.detach().float() if torch.is_tensor(content) else None
            content_by_head = affinity.get(
                "_prompt_patch_pre_output_content_by_head"
            )
            content_by_head = (
                content_by_head.detach().float()
                if torch.is_tensor(content_by_head)
                else None
            )
            if usage is None and content is None:
                continue
            state = self._state(layer_index)
            if usage is not None and usage.shape[-1] == self.prompt_length:
                usage = usage.clamp_min(0.0)
                usage = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-12)
                state["usage_sum"] += usage.sum(dim=0).cpu().to(torch.float64)
                state["top1_count"] += torch.bincount(
                    usage.argmax(dim=1).cpu(), minlength=self.prompt_length
                ).to(torch.float64)
                state["usage_count"] += int(usage.shape[0])
            if (
                content is None
                or content.dim() != 3
                or content.shape[1] != self.prompt_length
            ):
                continue
            norms = content.norm(dim=-1)
            content_share = norms / norms.sum(dim=1, keepdim=True).clamp_min(1e-12)
            state["content_sum"] += content_share.sum(dim=0).cpu().to(torch.float64)
            state["content_count"] += int(content.shape[0])
            self._update_content_pairs(state, content)
            if (
                content_by_head is not None
                and content_by_head.dim() == 4
                and content_by_head.shape[2] == self.prompt_length
            ):
                for head_index in range(content_by_head.shape[1]):
                    self._update_content_pairs(
                        self._head_state(layer_index, head_index),
                        content_by_head[:, head_index, :, :],
                    )

    def _prompt_type(self, prompt_index: int) -> str:
        if self.instance_tokens > 0 and prompt_index < self.instance_tokens:
            return "instance"
        if (
            self.domain_tokens > 0
            and self.instance_tokens <= prompt_index < self.instance_tokens + self.domain_tokens
        ):
            return "domain"
        return "visual"

    def finalize(self) -> Dict[str, Any]:
        by_layer: Dict[int, Dict[str, float]] = {}
        by_layer_prompt: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for layer_index, state in sorted(self.states.items()):
            metrics: Dict[str, float] = {}
            prompt_metrics: Dict[int, Dict[str, Any]] = {}
            usage = None
            content = None
            coverage = None
            if state["usage_count"] > 0:
                usage = state["usage_sum"] / state["usage_count"]
                usage = usage / usage.sum().clamp_min(1e-12)
                entropy = -float(
                    (usage * usage.clamp_min(1e-12).log()).sum().item()
                )
                metrics["prompt_slot_usage_entropy"] = entropy
                metrics["prompt_slot_effective_count"] = float(math.exp(entropy))
                coverage = state["top1_count"] / state["usage_count"]
            if state["content_count"] > 0:
                content = state["content_sum"] / state["content_count"]
                content = content / content.sum().clamp_min(1e-12)
            low_flags = torch.zeros(self.prompt_length, dtype=torch.bool)
            if usage is not None and content is not None and coverage is not None:
                low_flags = (
                    (usage < self.low_usage_fraction / max(1, self.prompt_length))
                    & (content < self.low_function_fraction / max(1, self.prompt_length))
                    & (coverage < self.low_role_coverage)
                )
                metrics["low_use_low_function_slot_ratio"] = float(
                    low_flags.float().mean().item()
                )
            metrics.update(self._finalize_pair_state(state))
            for prompt_index in range(self.prompt_length):
                item: Dict[str, Any] = {
                    "prompt_type": self._prompt_type(prompt_index),
                    "low_use_low_function": bool(low_flags[prompt_index].item()),
                }
                if usage is not None:
                    item["usage_share"] = float(usage[prompt_index].item())
                if content is not None:
                    item["content_share"] = float(content[prompt_index].item())
                if coverage is not None:
                    item["top1_role_coverage"] = float(coverage[prompt_index].item())
                prompt_metrics[prompt_index] = item
            if metrics:
                by_layer[layer_index] = metrics
            if prompt_metrics:
                by_layer_prompt[layer_index] = prompt_metrics
        aggregate: Dict[str, float] = {}
        names = sorted({name for metrics in by_layer.values() for name in metrics})
        for name in names:
            values = [metrics[name] for metrics in by_layer.values() if name in metrics]
            if values:
                aggregate[name] = float(np.mean(values))
        by_layer_head: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(dict)
        for (layer_index, head_index), state in sorted(self.head_states.items()):
            metrics = self._finalize_pair_state(state)
            if metrics:
                by_layer_head[layer_index][head_index] = metrics
        return {
            "metrics": aggregate,
            "by_layer": by_layer,
            "by_layer_and_head": dict(by_layer_head),
            "by_layer_and_prompt": by_layer_prompt,
        }


def _inverse_horizontal_patch_map(values: torch.Tensor) -> Optional[torch.Tensor]:
    patch_count = int(values.shape[-1])
    side = int(math.sqrt(patch_count))
    if side * side != patch_count:
        return None
    return values.reshape(*values.shape[:-1], side, side).flip(-1).reshape_as(values)


class PairedFlipAccumulator:
    def __init__(
        self,
        *,
        prompt_length: int,
        semantic_length: int,
        topk: int,
        selected_layers: Sequence[int] = (),
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.semantic_length = int(semantic_length)
        self.topk = max(1, int(topk))
        self.selected_layers = {int(item) for item in selected_layers}
        self.by_layer: Dict[int, Dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.prediction_matches = 0
        self.prediction_count = 0
        self.non_square_layers = set()
        self.observed_layers = set()

    @staticmethod
    def _normalized(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp_min(0.0)
        return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _update_map(
        self,
        layer_index: int,
        name: str,
        normal: torch.Tensor,
        flipped: torch.Tensor,
    ) -> None:
        flipped = _inverse_horizontal_patch_map(flipped)
        if flipped is None:
            self.non_square_layers.add(int(layer_index))
            return
        normal = self._normalized(normal)
        flipped = self._normalized(flipped)
        cosine = torch.nn.functional.cosine_similarity(
            normal, flipped, dim=-1, eps=1e-12
        )
        k = min(self.topk, int(normal.shape[-1]))
        normal_top = normal.topk(k, dim=-1).indices
        flipped_top = flipped.topk(k, dim=-1).indices
        overlap = (
            normal_top.unsqueeze(-1) == flipped_top.unsqueeze(-2)
        ).any(dim=-1).float().mean(dim=-1)
        patch_count = int(normal.shape[-1])
        side = int(math.sqrt(patch_count))
        axis = torch.linspace(0.0, 1.0, side, device=normal.device)
        yy, xx = torch.meshgrid(axis, axis)
        positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        normal_centroid = torch.matmul(normal, positions)
        flipped_centroid = torch.matmul(flipped, positions)
        centroid_error = (normal_centroid - flipped_centroid).norm(dim=-1)
        state = self.by_layer[int(layer_index)]
        state[f"{name}_map_cosine"].extend(cosine.reshape(-1).cpu().tolist())
        state[f"{name}_topk_overlap"].extend(overlap.reshape(-1).cpu().tolist())
        state[f"{name}_centroid_error"].extend(
            centroid_error.reshape(-1).cpu().tolist()
        )
        self.observed_layers.add(int(layer_index))

    def update(
        self,
        normal_attention_layers: Sequence[Any],
        normal_affinity_layers: Sequence[Any],
        flipped_attention_layers: Sequence[Any],
        flipped_affinity_layers: Sequence[Any],
        *,
        normal_predictions: Any,
        flipped_predictions: Any,
    ) -> None:
        normal_pred = torch.as_tensor(normal_predictions).reshape(-1).cpu()
        flipped_pred = torch.as_tensor(flipped_predictions).reshape(-1).cpu()
        if normal_pred.shape == flipped_pred.shape:
            self.prediction_matches += int((normal_pred == flipped_pred).sum().item())
            self.prediction_count += int(normal_pred.numel())
        layer_count = max(
            len(normal_attention_layers or []),
            len(normal_affinity_layers or []),
            len(flipped_attention_layers or []),
            len(flipped_affinity_layers or []),
        )
        prompt_slice = slice(1, 1 + self.prompt_length)
        for layer_index in range(layer_count):
            if self.selected_layers and layer_index not in self.selected_layers:
                continue
            normal_attention = (
                normal_attention_layers[layer_index]
                if layer_index < len(normal_attention_layers or [])
                and torch.is_tensor(normal_attention_layers[layer_index])
                else None
            )
            flipped_attention = (
                flipped_attention_layers[layer_index]
                if layer_index < len(flipped_attention_layers or [])
                and torch.is_tensor(flipped_attention_layers[layer_index])
                else None
            )
            normal_affinity = (
                normal_affinity_layers[layer_index]
                if layer_index < len(normal_affinity_layers or [])
                and isinstance(normal_affinity_layers[layer_index], Mapping)
                else {}
            )
            flipped_affinity = (
                flipped_affinity_layers[layer_index]
                if layer_index < len(flipped_affinity_layers or [])
                and isinstance(flipped_affinity_layers[layer_index], Mapping)
                else {}
            )
            prompt_patch_normal = None
            prompt_patch_flipped = None
            cls_prompt_normal = None
            cls_prompt_flipped = None
            if (
                normal_attention is not None
                and flipped_attention is not None
                and normal_attention.shape == flipped_attention.shape
                and normal_attention.dim() == 4
            ):
                patch_start = 1 + self.prompt_length
                patch_end = int(normal_attention.shape[-1]) - self.semantic_length
                prompt_patch_normal = normal_attention[
                    :, :, prompt_slice, patch_start:patch_end
                ].detach().float().mean(dim=1)
                prompt_patch_flipped = flipped_attention[
                    :, :, prompt_slice, patch_start:patch_end
                ].detach().float().mean(dim=1)
                cls_prompt_normal = normal_attention[
                    :, :, 0, prompt_slice
                ].detach().float().mean(dim=1)
                cls_prompt_flipped = flipped_attention[
                    :, :, 0, prompt_slice
                ].detach().float().mean(dim=1)
            if prompt_patch_normal is not None and prompt_patch_normal.numel():
                self._update_map(
                    layer_index,
                    "prompt_patch",
                    prompt_patch_normal,
                    prompt_patch_flipped,
                )
            normal_av = normal_affinity.get("_prompt_patch_pre_output_av_magnitude")
            flipped_av = flipped_affinity.get("_prompt_patch_pre_output_av_magnitude")
            if (
                torch.is_tensor(normal_av)
                and torch.is_tensor(flipped_av)
                and normal_av.shape == flipped_av.shape
            ):
                self._update_map(
                    layer_index,
                    "prompt_av",
                    normal_av.detach().float(),
                    flipped_av.detach().float(),
                )
            if cls_prompt_normal is not None and cls_prompt_normal.numel():
                left = self._normalized(cls_prompt_normal)
                right = self._normalized(cls_prompt_flipped)
                midpoint = 0.5 * (left + right)
                js = 0.5 * (
                    (left * (left.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum(dim=-1)
                    + (right * (right.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum(dim=-1)
                )
                top1 = (left.argmax(dim=-1) == right.argmax(dim=-1)).float()
                left_effective = (-(left * left.clamp_min(1e-12).log()).sum(dim=-1)).exp()
                right_effective = (-(right * right.clamp_min(1e-12).log()).sum(dim=-1)).exp()
                state = self.by_layer[int(layer_index)]
                state["assignment_js_divergence"].extend(js.cpu().tolist())
                state["assignment_top1_consistency"].extend(top1.cpu().tolist())
                state["assignment_effective_count_abs_delta"].extend(
                    (right_effective - left_effective).abs().cpu().tolist()
                )
                self.observed_layers.add(int(layer_index))

    def finalize(self) -> Dict[str, Any]:
        by_layer = {
            layer: {
                name: float(np.mean(values))
                for name, values in sorted(metrics.items())
                if values
            }
            for layer, metrics in sorted(self.by_layer.items())
        }
        aggregate: Dict[str, float] = {}
        names = sorted({name for metrics in by_layer.values() for name in metrics})
        for name in names:
            values = [metrics[name] for metrics in by_layer.values() if name in metrics]
            if values:
                aggregate[name] = float(np.mean(values))
        if self.prediction_count > 0:
            aggregate["prediction_top1_consistency"] = float(
                self.prediction_matches / self.prediction_count
            )
        valid = bool(self.observed_layers)
        return {
            "format": "paired_prompt_horizontal_flip_v1",
            "valid": valid,
            "applicability": (
                "applicable"
                if valid
                else "not_applicable_non_square_patch_grid"
                if self.non_square_layers
                else "not_applicable_no_prompt_attention"
            ),
            "observed_layers": sorted(self.observed_layers),
            "non_square_layers": sorted(self.non_square_layers),
            "metrics": aggregate,
            "by_layer": by_layer,
            "metric_semantics": "descriptive_not_hard_equivariance_gate",
        }


def _eligible_prompt_edges(
    sequence_length: int,
    prompt_length: int,
    semantic_length: int,
    paths: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(sequence_length, sequence_length, dtype=torch.bool, device=device)
    prompt_slice = slice(1, 1 + int(prompt_length))
    patch_start = 1 + int(prompt_length)
    patch_end = sequence_length - int(semantic_length)
    requested = {str(item) for item in paths}
    if "cls_to_prompt" in requested:
        mask[:1, prompt_slice] = True
    if "prompt_to_cls" in requested:
        mask[prompt_slice, :1] = True
    if "prompt_to_patch" in requested and patch_end > patch_start:
        mask[prompt_slice, patch_start:patch_end] = True
    if "patch_to_prompt" in requested and patch_end > patch_start:
        mask[patch_start:patch_end, prompt_slice] = True
    return mask


def build_relevance_deletion_masks(
    relevance: torch.Tensor,
    *,
    prompt_length: int,
    semantic_length: int,
    paths: Sequence[str],
    conditions: Sequence[str],
    fraction: float,
    random_seed: int,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    if relevance.dim() != 4:
        raise ValueError("relevance must have shape [batch, heads, query, key]")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("deletion fraction must be between zero and one")
    supported = {"positive", "negative", "absolute", "low", "random"}
    conditions = [str(item).lower() for item in conditions]
    unknown = sorted(set(conditions).difference(supported))
    if unknown:
        raise ValueError(f"unsupported relevance deletion conditions: {unknown}")
    batch, heads, query_count, key_count = relevance.shape
    if query_count != key_count:
        raise ValueError("relevance attention matrix must be square")
    base = _eligible_prompt_edges(
        key_count,
        prompt_length,
        semantic_length,
        paths,
        device=relevance.device,
    )
    eligible = base.unsqueeze(0).expand(heads, -1, -1).reshape(-1)
    eligible_count = int(eligible.sum().item())
    if eligible_count <= 0:
        return {}, {}
    masks = {
        condition: torch.zeros_like(relevance, dtype=torch.bool)
        for condition in conditions
    }
    selected_counts: Dict[str, int] = defaultdict(int)
    generator = torch.Generator(device=relevance.device)
    generator.manual_seed(int(random_seed) % (2 ** 63 - 1))
    for sample_index in range(batch):
        values = relevance[sample_index].reshape(-1)
        base_budget = max(1, int(round(float(fraction) * eligible_count)))
        positive_count = int(((values > 0) & eligible).sum().item())
        negative_count = int(((values < 0) & eligible).sum().item())
        signed_budget = base_budget
        if "positive" in conditions and "negative" in conditions:
            signed_budget = min(base_budget, positive_count, negative_count)
        random_budget = signed_budget if (
            "positive" in conditions and "negative" in conditions
        ) else base_budget
        for condition in conditions:
            if condition == "positive":
                candidate = eligible & (values > 0)
                budget = min(signed_budget, int(candidate.sum().item()))
                scores = values
                largest = True
            elif condition == "negative":
                candidate = eligible & (values < 0)
                budget = min(signed_budget, int(candidate.sum().item()))
                scores = values
                largest = False
            elif condition == "absolute":
                candidate = eligible
                budget = min(base_budget, eligible_count)
                scores = values.abs()
                largest = True
            elif condition == "low":
                candidate = eligible
                budget = min(base_budget, eligible_count)
                scores = values.abs()
                largest = False
            else:
                candidate = eligible
                budget = min(random_budget, eligible_count)
                scores = torch.rand(
                    values.shape,
                    generator=generator,
                    device=values.device,
                    dtype=torch.float32,
                )
                largest = True
            if budget <= 0:
                continue
            fill = float("-inf") if largest else float("inf")
            ranked = scores.masked_fill(~candidate, fill)
            selected = ranked.topk(budget, largest=largest).indices
            flat_mask = masks[condition][sample_index].reshape(-1)
            flat_mask[selected] = True
            selected_counts[condition] += int(budget)
    metadata = {
        condition: {
            "selected_edge_count": float(selected_counts[condition]),
            "eligible_edge_count": float(batch * eligible_count),
            "actual_deletion_ratio": float(
                selected_counts[condition] / max(1, batch * eligible_count)
            ),
        }
        for condition in conditions
    }
    return masks, metadata


def summarize_deletion_curves(
    effects: Mapping[tuple[int, str, float], Mapping[str, Any]],
) -> Dict[str, Any]:
    by_layer_condition: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    grouped: Dict[tuple[int, str], list[tuple[float, Mapping[str, Any]]]] = defaultdict(list)
    for (layer, condition, fraction), effect in effects.items():
        grouped[(int(layer), str(condition))].append((float(fraction), effect))
    for (layer, condition), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item[0])
        fractions = [0.0] + [item[0] for item in rows]
        margin_drop = [0.0]
        accuracy_drop = [0.0]
        raw = []
        for fraction, effect in rows:
            summary = dict(effect.get("summary", {}))
            margin = -float(summary.get("delta_true_margin", 0.0))
            accuracy = -float(summary.get("net_beneficial_flip", 0.0))
            margin_drop.append(margin)
            accuracy_drop.append(accuracy)
            raw.append({"fraction": fraction, **summary})
        by_layer_condition[layer][condition] = {
            "fractions": fractions,
            "margin_drop_auc": float(np.trapz(margin_drop, fractions)),
            "accuracy_drop_auc": float(np.trapz(accuracy_drop, fractions)),
            "points": raw,
        }
    comparisons: Dict[int, Dict[str, float]] = {}
    for layer, conditions in by_layer_condition.items():
        random_curve = conditions.get("random")
        if random_curve is None:
            continue
        item: Dict[str, float] = {}
        for condition in ("positive", "negative", "absolute", "low"):
            curve = conditions.get(condition)
            if curve is None:
                continue
            item[f"{condition}_minus_random_margin_drop_auc"] = float(
                curve["margin_drop_auc"] - random_curve["margin_drop_auc"]
            )
            item[f"{condition}_minus_random_accuracy_drop_auc"] = float(
                curve["accuracy_drop_auc"] - random_curve["accuracy_drop_auc"]
            )
        if item:
            comparisons[layer] = item
    return {
        "format": "prompt_explanation_validity_v1",
        "by_layer_condition": dict(by_layer_condition),
        "comparisons_by_layer": comparisons,
    }


def match_prompt_role_vectors(
    reference_vectors: Any,
    candidate_vectors: Any,
    *,
    reference_types: Optional[Sequence[str]] = None,
    candidate_types: Optional[Sequence[str]] = None,
    max_cost: float = 0.5,
) -> Dict[str, Any]:
    left = np.asarray(reference_vectors, dtype=np.float64)
    right = np.asarray(candidate_vectors, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("role vectors must be two matrices with the same feature dimension")
    left_count, right_count = left.shape[0], right.shape[0]
    reference_types = list(reference_types or ["visual"] * left_count)
    candidate_types = list(candidate_types or ["visual"] * right_count)
    if len(reference_types) != left_count or len(candidate_types) != right_count:
        raise ValueError("role type count does not match role vector count")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    cosine = (left @ right.T) / np.maximum(
        left_norm[:, None] * right_norm[None, :], 1e-12
    )
    cost = 1.0 - cosine
    valid_pair = (
        (left_norm[:, None] > 1e-12)
        & (right_norm[None, :] > 1e-12)
        & np.asarray([
            [left_type == right_type for right_type in candidate_types]
            for left_type in reference_types
        ], dtype=bool)
    )
    big = 1e6
    real_cost = np.where(valid_pair, cost, big)
    size = left_count + right_count
    matrix = np.full((size, size), big, dtype=np.float64)
    matrix[:left_count, :right_count] = real_cost
    for index in range(left_count):
        matrix[index, right_count + index] = float(max_cost)
    for index in range(right_count):
        matrix[left_count + index, index] = float(max_cost)
    matrix[left_count:, right_count:] = 0.0
    row_ind, col_ind = linear_sum_assignment(matrix)
    pairs = []
    matched_left = set()
    matched_right = set()
    ambiguity_gaps = []
    for row, column in zip(row_ind.tolist(), col_ind.tolist()):
        if row >= left_count or column >= right_count:
            continue
        current_cost = float(real_cost[row, column])
        if not math.isfinite(current_cost) or current_cost > float(max_cost):
            continue
        alternatives = np.sort(real_cost[row][real_cost[row] < big])
        gap = (
            float(alternatives[1] - alternatives[0])
            if alternatives.size >= 2
            else None
        )
        if gap is not None:
            ambiguity_gaps.append(gap)
        dominant_agreement = bool(left[row].argmax() == right[column].argmax())
        pairs.append({
            "reference_index": int(row),
            "candidate_index": int(column),
            "prompt_type": str(reference_types[row]),
            "cost": current_cost,
            "cosine": float(1.0 - current_cost),
            "dominant_class_agreement": dominant_agreement,
            "best_second_cost_gap": gap,
        })
        matched_left.add(row)
        matched_right.add(column)
    denominator = max(left_count, right_count, 1)
    return {
        "matched_role_count": len(pairs),
        "matched_role_acceptance_ratio": float(len(pairs) / denominator),
        "matched_role_cosine_mean": float(
            np.mean([item["cosine"] for item in pairs]) if pairs else 0.0
        ),
        "matched_dominant_class_agreement": float(
            np.mean([item["dominant_class_agreement"] for item in pairs])
            if pairs
            else 0.0
        ),
        "unmatched_or_dead_prompt_ratio": float(
            ((left_count - len(matched_left)) + (right_count - len(matched_right)))
            / max(1, left_count + right_count)
        ),
        "best_second_cost_gap_mean": float(
            np.mean(ambiguity_gaps) if ambiguity_gaps else 0.0
        ),
        "unmatched_reference_indices": sorted(set(range(left_count)) - matched_left),
        "unmatched_candidate_indices": sorted(set(range(right_count)) - matched_right),
        "pairs": pairs,
    }


__all__ = [
    "PairedFlipAccumulator",
    "PromptContentSlotAccumulator",
    "PromptSourceDecompositionAccumulator",
    "build_relevance_deletion_masks",
    "match_prompt_role_vectors",
    "summarize_deletion_curves",
]
