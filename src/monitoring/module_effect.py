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
    arrays = {
        "targets": target,
        "normal_logits": normal,
        "intervention_logits": changed,
        "normal_predictions": normal_pred,
        "intervention_predictions": changed_pred,
        "delta_true_margin": delta_true_margin,
        "delta_seen_bias_margin": delta_bias,
        "delta_entropy": changed_entropy - normal_entropy,
        "beneficial_flip": beneficial.astype(np.int8),
        "harmful_flip": harmful.astype(np.int8),
    }
    return {"summary": summary, "per_class": class_delta, "arrays": arrays}
