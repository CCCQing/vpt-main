from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ParameterIntervention(AbstractContextManager):
    def __init__(self, model: torch.nn.Module, predicate: Callable[[str, torch.nn.Parameter], bool], value: float = 0.0) -> None:
        self.model = model
        self.predicate = predicate
        self.value = float(value)
        self.saved: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if self.predicate(name, parameter):
                    self.saved[name] = parameter.detach().clone()
                    parameter.fill_(self.value)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with torch.no_grad():
            named = dict(self.model.named_parameters())
            for name, value in self.saved.items():
                named[name].copy_(value)
        self.saved.clear()
        return False


class PromptAttentionPathIntervention(AbstractContextManager):
    VALID_MODES = {
        "prompt_read_block",
        "prompt_write_block",
        "prompt_patch_uniform",
        "patch_prompt_uniform",
        "prompt_patch_value_globalize",
        "prompt_value_zero",
        "relevance_edge_delete",
        "attribute_concept_prompt_patch_block",
        "random_prompt_patch_block",
        "transport_prompt_patch_block",
        "transport_random_patch_block",
    }

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str,
        **payload: Any,
    ) -> None:
        if str(mode) not in self.VALID_MODES:
            raise ValueError(f"Unsupported prompt attention intervention: {mode}")
        self.model = model
        self.mode = str(mode)
        self.payload = dict(payload)
        self.saved: Dict[torch.nn.Module, Any] = {}

    def __enter__(self):
        layer_index = 0
        for module in self.model.modules():
            if not hasattr(module, "_prompt_path_intervention"):
                continue
            self.saved[module] = getattr(module, "_prompt_path_intervention")
            if hasattr(module, "_last_prompt_path_intervention_stats"):
                setattr(module, "_last_prompt_path_intervention_stats", None)
            intervention = dict(self.payload)
            intervention.update({"mode": self.mode, "layer_index": layer_index})
            setattr(module, "_prompt_path_intervention", intervention)
            layer_index += 1
        if not self.saved:
            raise RuntimeError("No compatible Attention module was found for prompt path intervention")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, value in self.saved.items():
            setattr(module, "_prompt_path_intervention", value)
        self.saved.clear()
        return False


class PromptStateIntervention(AbstractContextManager):
    VALID_MODES = {"prompt_context_swap"}

    def __init__(self, model: torch.nn.Module, mode: str, **payload: Any) -> None:
        if str(mode) not in self.VALID_MODES:
            raise ValueError(f"Unsupported prompt state intervention: {mode}")
        self.model = model
        self.mode = str(mode)
        self.payload = dict(payload)
        self.saved: Dict[torch.nn.Module, Any] = {}

    def __enter__(self):
        for module in self.model.modules():
            if not hasattr(module, "_prompt_state_intervention"):
                continue
            self.saved[module] = getattr(module, "_prompt_state_intervention")
            setattr(
                module,
                "_prompt_state_intervention",
                {"mode": self.mode, **self.payload},
            )
        if not self.saved:
            raise RuntimeError(
                "No compatible Encoder module was found for prompt state intervention"
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, value in self.saved.items():
            setattr(module, "_prompt_state_intervention", value)
        self.saved.clear()
        return False


class PromptDistributionIntervention(AbstractContextManager):
    VALID_MODES = {
        "instance_prompt_zero",
        "domain_prompt_zero",
        "both_prompt_zero",
        "instance_prompt_swap",
    }

    def __init__(self, model: torch.nn.Module, mode: str, **payload: Any) -> None:
        if str(mode) not in self.VALID_MODES:
            raise ValueError(f"Unsupported Prompt Distributor intervention: {mode}")
        self.model = model
        self.mode = str(mode)
        self.payload = dict(payload)
        self.saved: Dict[torch.nn.Module, Any] = {}

    def __enter__(self):
        for module in self.model.modules():
            if not hasattr(module, "_prompt_output_intervention"):
                continue
            self.saved[module] = getattr(module, "_prompt_output_intervention")
            setattr(
                module,
                "_prompt_output_intervention",
                {"mode": self.mode, **self.payload},
            )
        if not self.saved:
            raise RuntimeError("No compatible Prompt Distributor was found")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, value in self.saved.items():
            setattr(module, "_prompt_output_intervention", value)
        self.saved.clear()
        return False


class DeepPromptResidualIntervention(AbstractContextManager):
    VALID_MODES = {
        "delta_zero",
        "mean_swap",
        "mean_replace",
        "layer_scales",
    }

    def __init__(self, model: torch.nn.Module, mode: str, **payload: Any) -> None:
        if str(mode) not in self.VALID_MODES:
            raise ValueError(f"Unsupported Deep Prompt residual intervention: {mode}")
        self.model = model
        self.mode = str(mode)
        self.payload = dict(payload)
        self.saved: Dict[torch.nn.Module, Any] = {}

    def __enter__(self):
        for module in self.model.modules():
            if module.__class__.__name__ != "MeanConditionedDeepPromptResidual":
                continue
            self.saved[module] = getattr(module, "_runtime_intervention", None)
            setattr(
                module,
                "_runtime_intervention",
                {"mode": self.mode, **self.payload},
            )
        if not self.saved:
            raise RuntimeError("No compatible Deep Prompt residual module was found")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, value in self.saved.items():
            setattr(module, "_runtime_intervention", value)
        self.saved.clear()
        return False


def prompt_zero_intervention(model: torch.nn.Module) -> ParameterIntervention:
    return ParameterIntervention(
        model,
        lambda name, parameter: (
            parameter.requires_grad
            and "prompt" in name.lower()
            and "prompt_init_provider" not in name.lower()
            and "attention_mediation" not in name.lower()
        ),
        value=0.0,
    )


def attention_mediation_gamma_zero_intervention(model: torch.nn.Module) -> ParameterIntervention:
    return ParameterIntervention(
        model,
        lambda name, parameter: "attention_mediation" in name.lower() and "gamma" in name.lower(),
        value=0.0,
    )


def prompt_read_block_intervention(model: torch.nn.Module) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(model, "prompt_read_block")


def layer_prompt_read_block_intervention(
    model: torch.nn.Module,
    *,
    target_layer: int,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(
        model,
        "prompt_read_block",
        target_layer=int(target_layer),
    )


def prompt_write_block_intervention(model: torch.nn.Module) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(model, "prompt_write_block")


def prompt_patch_uniform_intervention(model: torch.nn.Module) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(model, "prompt_patch_uniform")


def patch_prompt_uniform_intervention(model: torch.nn.Module) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(model, "patch_prompt_uniform")


def prompt_patch_value_globalize_intervention(
    model: torch.nn.Module,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(
        model,
        "prompt_patch_value_globalize",
    )


def prompt_value_zero_intervention(
    model: torch.nn.Module,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(model, "prompt_value_zero")


def relevance_edge_delete_intervention(
    model: torch.nn.Module,
    *,
    target_layer: int,
    deletion_mask: torch.Tensor,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(
        model,
        "relevance_edge_delete",
        target_layer=int(target_layer),
        deletion_mask=deletion_mask,
    )


def instance_prompt_zero_intervention(
    model: torch.nn.Module,
) -> PromptDistributionIntervention:
    return PromptDistributionIntervention(model, "instance_prompt_zero")


def domain_prompt_zero_intervention(
    model: torch.nn.Module,
) -> PromptDistributionIntervention:
    return PromptDistributionIntervention(model, "domain_prompt_zero")


def both_prompt_zero_intervention(
    model: torch.nn.Module,
) -> PromptDistributionIntervention:
    return PromptDistributionIntervention(model, "both_prompt_zero")


def instance_prompt_swap_intervention(
    model: torch.nn.Module,
    *,
    permutation: torch.Tensor,
) -> PromptDistributionIntervention:
    return PromptDistributionIntervention(
        model,
        "instance_prompt_swap",
        permutation=permutation,
    )


def deep_prompt_residual_zero_intervention(
    model: torch.nn.Module,
) -> DeepPromptResidualIntervention:
    return DeepPromptResidualIntervention(model, "delta_zero")


def deep_prompt_residual_swap_intervention(
    model: torch.nn.Module,
    *,
    permutation: torch.Tensor,
) -> DeepPromptResidualIntervention:
    return DeepPromptResidualIntervention(
        model,
        "mean_swap",
        permutation=permutation,
    )


def deep_prompt_residual_replace_intervention(
    model: torch.nn.Module,
    *,
    replacement: torch.Tensor,
) -> DeepPromptResidualIntervention:
    return DeepPromptResidualIntervention(
        model,
        "mean_replace",
        replacement=replacement,
    )


def deep_prompt_residual_layer_scales_intervention(
    model: torch.nn.Module,
    *,
    scales: Sequence[float],
) -> DeepPromptResidualIntervention:
    values = tuple(float(value) for value in scales)
    if not values:
        raise ValueError("Deep Prompt residual layer scales must not be empty")
    return DeepPromptResidualIntervention(
        model,
        "layer_scales",
        scales=values,
    )


def prompt_context_swap_intervention(
    model: torch.nn.Module,
    *,
    target_layer: int,
    permutation: torch.Tensor,
) -> PromptStateIntervention:
    return PromptStateIntervention(
        model,
        "prompt_context_swap",
        target_layer=int(target_layer),
        permutation=permutation,
    )


def attribute_concept_prompt_patch_block_intervention(
    model: torch.nn.Module,
    *,
    attribute_directions: torch.Tensor,
    margin_weights: torch.Tensor,
    patch_ratio: float,
    random_seed: int,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(
        model,
        "attribute_concept_prompt_patch_block",
        attribute_directions=attribute_directions,
        margin_weights=margin_weights,
        patch_ratio=float(patch_ratio),
        random_seed=int(random_seed),
    )


def random_prompt_patch_block_intervention(
    model: torch.nn.Module,
    *,
    attribute_directions: torch.Tensor,
    margin_weights: torch.Tensor,
    patch_ratio: float,
    random_seed: int,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(
        model,
        "random_prompt_patch_block",
        attribute_directions=attribute_directions,
        margin_weights=margin_weights,
        patch_ratio=float(patch_ratio),
        random_seed=int(random_seed),
    )


def transport_prompt_patch_block_intervention(
    model: torch.nn.Module,
    *,
    selected_patch_indices: torch.Tensor,
    reference_patch_scores: torch.Tensor,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(
        model,
        "transport_prompt_patch_block",
        selected_patch_indices=selected_patch_indices,
        reference_patch_scores=reference_patch_scores,
    )


def transport_random_patch_block_intervention(
    model: torch.nn.Module,
    *,
    selected_patch_indices: torch.Tensor,
    reference_patch_scores: torch.Tensor,
) -> PromptAttentionPathIntervention:
    return PromptAttentionPathIntervention(
        model,
        "transport_random_patch_block",
        selected_patch_indices=selected_patch_indices,
        reference_patch_scores=reference_patch_scores,
    )


def paired_module_effect_metrics(
    normal_logits: Any,
    intervention_logits: Any,
    targets: Any,
    candidate_global_ids: Sequence[int],
    seen_global_ids: Sequence[int],
    *,
    normal_features: Optional[Any] = None,
    intervention_features: Optional[Any] = None,
) -> Dict[str, Any]:
    normal = np.asarray(normal_logits, dtype=np.float32)
    changed = np.asarray(intervention_logits, dtype=np.float32)
    target = np.asarray(targets, dtype=np.int64).reshape(-1)
    if normal.ndim != 2 or normal.shape != changed.shape or normal.shape[0] != target.size:
        raise ValueError("paired module-effect arrays have incompatible shapes")
    candidate = np.asarray(candidate_global_ids, dtype=np.int64).reshape(-1)
    if candidate.size != normal.shape[1]:
        raise ValueError("candidate_global_ids length does not match logits")
    seen_set = {int(item) for item in seen_global_ids}
    seen_columns = np.asarray([int(item) in seen_set for item in candidate], dtype=bool)
    unseen_columns = ~seen_columns

    def softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    def true_margin(values: np.ndarray) -> np.ndarray:
        true = values[np.arange(target.size), target]
        other = values.copy()
        other[np.arange(target.size), target] = -np.inf
        return true - other.max(axis=1)

    def bias_margin(values: np.ndarray) -> np.ndarray:
        if not seen_columns.any() or not unseen_columns.any():
            return np.zeros(values.shape[0], dtype=np.float32)
        return values[:, seen_columns].max(axis=1) - values[:, unseen_columns].max(axis=1)

    normal_pred = normal.argmax(axis=1)
    changed_pred = changed.argmax(axis=1)
    normal_correct = normal_pred == target
    changed_correct = changed_pred == target
    normal_prob = softmax(normal)
    changed_prob = softmax(changed)
    normal_entropy = -np.sum(normal_prob * np.log(np.maximum(normal_prob, 1e-12)), axis=1)
    changed_entropy = -np.sum(changed_prob * np.log(np.maximum(changed_prob, 1e-12)), axis=1)
    delta_logits = changed - normal
    delta_true_margin = true_margin(changed) - true_margin(normal)
    delta_bias = bias_margin(changed) - bias_margin(normal)
    beneficial = (~normal_correct) & changed_correct
    harmful = normal_correct & (~changed_correct)
    summary = {
        "delta_logits_norm": float(np.linalg.norm(delta_logits, axis=1).mean()),
        "delta_true_margin": float(delta_true_margin.mean()),
        "delta_seen_bias_margin": float(delta_bias.mean()),
        "prediction_flip_rate": float((normal_pred != changed_pred).mean()),
        "beneficial_flip_rate": float(beneficial.mean()),
        "harmful_flip_rate": float(harmful.mean()),
        "net_beneficial_flip": float(beneficial.mean() - harmful.mean()),
        "delta_entropy": float((changed_entropy - normal_entropy).mean()),
    }
    if normal_features is not None and intervention_features is not None:
        left = np.asarray(normal_features, dtype=np.float32)
        right = np.asarray(intervention_features, dtype=np.float32)
        if left.shape == right.shape and left.ndim == 2 and left.shape[0] == target.size:
            denom = np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12)
            summary["feature_cosine_before_after"] = float((np.sum(left * right, axis=1) / denom).mean())

    class_delta = []
    for class_id in np.unique(target):
        mask = target == class_id
        class_delta.append({
            "local_class_id": int(class_id),
            "global_class_id": int(candidate[int(class_id)]),
            "normal_accuracy": float(normal_correct[mask].mean()),
            "intervention_accuracy": float(changed_correct[mask].mean()),
            "delta_accuracy": float(changed_correct[mask].mean() - normal_correct[mask].mean()),
            "support": int(mask.sum()),
        })
    return {"summary": summary, "per_class": class_delta}


class PairedModuleEffectAccumulator:
    def __init__(self, candidate_global_ids: Sequence[int], seen_global_ids: Sequence[int]) -> None:
        self.candidate = np.asarray(candidate_global_ids, dtype=np.int64).reshape(-1)
        seen_set = {int(item) for item in seen_global_ids}
        self.seen_columns = np.asarray([int(item) in seen_set for item in self.candidate], dtype=bool)
        self.unseen_columns = ~self.seen_columns
        self.sample_count = 0
        self.sums: Dict[str, float] = {
            "delta_logits_norm": 0.0,
            "delta_true_margin": 0.0,
            "delta_seen_bias_margin": 0.0,
            "prediction_flip_rate": 0.0,
            "beneficial_flip_rate": 0.0,
            "harmful_flip_rate": 0.0,
            "delta_entropy": 0.0,
            "feature_cosine_before_after": 0.0,
        }
        self.feature_count = 0
        class_count = int(self.candidate.size)
        self.class_support = np.zeros(class_count, dtype=np.int64)
        self.class_normal_correct = np.zeros(class_count, dtype=np.int64)
        self.class_changed_correct = np.zeros(class_count, dtype=np.int64)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _as_numpy(values: Any, dtype: Any) -> np.ndarray:
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def _true_margin(values: np.ndarray, target: np.ndarray) -> np.ndarray:
        true = values[np.arange(target.size), target]
        other = values.copy()
        other[np.arange(target.size), target] = -np.inf
        return true - other.max(axis=1)

    def _bias_margin(self, values: np.ndarray) -> np.ndarray:
        if not self.seen_columns.any() or not self.unseen_columns.any():
            return np.zeros(values.shape[0], dtype=np.float32)
        return values[:, self.seen_columns].max(axis=1) - values[:, self.unseen_columns].max(axis=1)

    def update(
        self,
        normal_logits: Any,
        intervention_logits: Any,
        targets: Any,
        *,
        normal_features: Optional[Any] = None,
        intervention_features: Optional[Any] = None,
    ) -> None:
        normal = self._as_numpy(normal_logits, np.float32)
        changed = self._as_numpy(intervention_logits, np.float32)
        target = self._as_numpy(targets, np.int64).reshape(-1)
        if normal.ndim != 2 or normal.shape != changed.shape or normal.shape[0] != target.size:
            raise ValueError("paired module-effect batches have incompatible shapes")
        if normal.shape[1] != self.candidate.size:
            raise ValueError("candidate_global_ids length does not match logits")
        count = int(target.size)
        if count == 0:
            return
        normal_pred = normal.argmax(axis=1)
        changed_pred = changed.argmax(axis=1)
        normal_correct = normal_pred == target
        changed_correct = changed_pred == target
        normal_prob = self._softmax(normal)
        changed_prob = self._softmax(changed)
        normal_entropy = -np.sum(normal_prob * np.log(np.maximum(normal_prob, 1e-12)), axis=1)
        changed_entropy = -np.sum(changed_prob * np.log(np.maximum(changed_prob, 1e-12)), axis=1)
        values = {
            "delta_logits_norm": np.linalg.norm(changed - normal, axis=1),
            "delta_true_margin": self._true_margin(changed, target) - self._true_margin(normal, target),
            "delta_seen_bias_margin": self._bias_margin(changed) - self._bias_margin(normal),
            "prediction_flip_rate": normal_pred != changed_pred,
            "beneficial_flip_rate": (~normal_correct) & changed_correct,
            "harmful_flip_rate": normal_correct & (~changed_correct),
            "delta_entropy": changed_entropy - normal_entropy,
        }
        for name, data in values.items():
            self.sums[name] += float(np.asarray(data, dtype=np.float32).sum())
        if normal_features is not None and intervention_features is not None:
            left = self._as_numpy(normal_features, np.float32)
            right = self._as_numpy(intervention_features, np.float32)
            if left.shape == right.shape and left.ndim == 2 and left.shape[0] == count:
                denom = np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12)
                cosine = np.sum(left * right, axis=1) / denom
                self.sums["feature_cosine_before_after"] += float(cosine.sum())
                self.feature_count += count
        np.add.at(self.class_support, target, 1)
        np.add.at(self.class_normal_correct, target, normal_correct.astype(np.int64))
        np.add.at(self.class_changed_correct, target, changed_correct.astype(np.int64))
        self.sample_count += count

    def finalize(self) -> Dict[str, Any]:
        if self.sample_count <= 0:
            return {"summary": {}, "per_class": []}
        summary = {
            name: float(value / self.sample_count)
            for name, value in self.sums.items()
            if name != "feature_cosine_before_after"
        }
        if self.feature_count > 0:
            summary["feature_cosine_before_after"] = float(
                self.sums["feature_cosine_before_after"] / self.feature_count
            )
        summary["net_beneficial_flip"] = float(
            summary["beneficial_flip_rate"] - summary["harmful_flip_rate"]
        )
        per_class = []
        deltas = []
        for class_id in np.flatnonzero(self.class_support > 0):
            support = int(self.class_support[class_id])
            normal_accuracy = float(self.class_normal_correct[class_id] / support)
            changed_accuracy = float(self.class_changed_correct[class_id] / support)
            delta = changed_accuracy - normal_accuracy
            deltas.append(delta)
            per_class.append({
                "local_class_id": int(class_id),
                "global_class_id": int(self.candidate[class_id]),
                "normal_accuracy": normal_accuracy,
                "intervention_accuracy": changed_accuracy,
                "delta_accuracy": float(delta),
                "support": support,
            })
        delta_array = np.asarray(deltas, dtype=np.float32)
        if delta_array.size:
            summary.update({
                "per_class_delta_mean": float(delta_array.mean()),
                "per_class_delta_median": float(np.quantile(delta_array, 0.5)),
                "per_class_delta_std": float(delta_array.std()),
                "per_class_delta_q25": float(np.quantile(delta_array, 0.25)),
                "per_class_delta_q75": float(np.quantile(delta_array, 0.75)),
                "per_class_decline_ratio": float((delta_array < 0.0).mean()),
            })
            seen_mask = np.asarray([int(row["global_class_id"]) in set(self.candidate[self.seen_columns].tolist()) for row in per_class], dtype=bool)
            if seen_mask.any():
                summary["seen_per_class_delta_mean"] = float(delta_array[seen_mask].mean())
            if (~seen_mask).any():
                summary["unseen_per_class_delta_mean"] = float(delta_array[~seen_mask].mean())
        return {"summary": summary, "per_class": per_class}
