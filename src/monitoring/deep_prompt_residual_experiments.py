from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


DEPTH_GROUPS = {
    "shallow": (0, 1, 2, 3),
    "middle": (4, 5, 6, 7),
    "deep": (8, 9, 10, 11),
}


def layer_scales(
    num_layers: int,
    *,
    selected_layers: Optional[Sequence[int]] = None,
    selected_scale: float = 1.0,
    other_scale: float = 1.0,
) -> tuple[float, ...]:
    count = int(num_layers)
    if count <= 0:
        raise ValueError("num_layers must be positive")
    selected = (
        set(range(count))
        if selected_layers is None
        else {int(value) for value in selected_layers}
    )
    if any(value < 0 or value >= count for value in selected):
        raise ValueError("selected residual layer is outside the model")
    values = tuple(
        float(selected_scale) if layer_id in selected else float(other_scale)
        for layer_id in range(count)
    )
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("residual layer scales must be finite")
    return values


def _slot_effective_rank(values: torch.Tensor, eps: float) -> torch.Tensor:
    if values.dim() != 3:
        raise ValueError("slot effective rank expects [B,P,D]")

    # This is a detached monitoring statistic over a small P x P matrix.  CUDA
    # MAGMA can fail to converge for the nearly rank-one Gram matrices produced
    # by shared residuals, especially in the server's older PyTorch build.
    # Compute the eigenspectrum on CPU in float64 so the diagnostic cannot abort
    # an otherwise valid replay; no model tensor or prediction is modified.
    matrix = values.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("slot effective rank received non-finite residual values")
    gram = np.matmul(matrix, np.swapaxes(matrix, -1, -2))
    gram = 0.5 * (gram + np.swapaxes(gram, -1, -2))
    try:
        eigenvalues = np.linalg.eigvalsh(gram)
        singular = np.sqrt(np.clip(eigenvalues, 0.0, None))
    except np.linalg.LinAlgError:
        # Direct SVD avoids squaring the condition number if LAPACK still
        # rejects an exceptionally ill-conditioned Gram matrix.
        singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)

    maximum = np.maximum(singular.max(axis=-1, keepdims=True), float(eps))
    singular = np.where(singular > maximum * 1.0e-3, singular, 0.0)
    total = singular.sum(axis=-1, keepdims=True)
    probability = singular / np.maximum(total, float(eps))
    entropy_terms = np.where(
        probability > 0.0,
        probability * np.log(np.maximum(probability, float(eps))),
        0.0,
    )
    rank = np.exp(-entropy_terms.sum(axis=-1))
    rank = np.where(total[:, 0] > float(eps), rank, 0.0)
    return torch.from_numpy(rank.astype(np.float32, copy=False))


def residual_static_geometry(
    trace: Sequence[Mapping[str, Any]],
    *,
    eps: float = 1.0e-8,
) -> Dict[str, np.ndarray]:
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    if not trace:
        raise ValueError("Deep Prompt residual trace is empty")
    rows: Dict[str, list[np.ndarray]] = {
        "residual_static_cosine": [],
        "signed_parallel_projection": [],
        "orthogonal_component_ratio": [],
        "residual_static_norm_ratio": [],
        "slot_shift_effective_rank": [],
    }
    expected_batch = None
    layer_ids = []
    for item in sorted(trace, key=lambda value: int(value.get("layer_id", -1))):
        layer_id = int(item.get("layer_id", -1))
        base = item.get("base_prompt")
        residual = item.get("applied_delta")
        if layer_id < 0 or not torch.is_tensor(base) or not torch.is_tensor(residual):
            raise ValueError("residual geometry requires layer_id, base_prompt and applied_delta")
        if base.dim() != 3 or tuple(base.shape) != tuple(residual.shape):
            raise ValueError("base Prompt and residual must share [B,P,D] shape")
        if expected_batch is None:
            expected_batch = int(base.shape[0])
        elif expected_batch != int(base.shape[0]):
            raise ValueError("residual trace layers use different batch sizes")
        base_value = base.detach().float()
        residual_value = residual.detach().float()
        base_norm = base_value.norm(dim=-1)
        residual_norm = residual_value.norm(dim=-1)
        dot = (base_value * residual_value).sum(dim=-1)
        cosine = dot / (base_norm * residual_norm).clamp_min(float(eps))
        base_unit = base_value / base_norm.unsqueeze(-1).clamp_min(float(eps))
        signed_projection = (residual_value * base_unit).sum(dim=-1)
        parallel = signed_projection.unsqueeze(-1) * base_unit
        orthogonal = residual_value - parallel
        orthogonal_ratio = orthogonal.norm(dim=-1) / residual_norm.clamp_min(float(eps))
        norm_ratio = residual_norm / base_norm.clamp_min(float(eps))
        rows["residual_static_cosine"].append(cosine.mean(dim=1).cpu().numpy())
        rows["signed_parallel_projection"].append(
            signed_projection.mean(dim=1).cpu().numpy()
        )
        rows["orthogonal_component_ratio"].append(
            orthogonal_ratio.mean(dim=1).cpu().numpy()
        )
        rows["residual_static_norm_ratio"].append(norm_ratio.mean(dim=1).cpu().numpy())
        rows["slot_shift_effective_rank"].append(
            _slot_effective_rank(residual_value, float(eps)).cpu().numpy()
        )
        layer_ids.append(layer_id)
    result = {
        name: np.stack(values, axis=1).astype(np.float32, copy=False)
        for name, values in rows.items()
    }
    result["layer_ids"] = np.asarray(layer_ids, dtype=np.int64)
    return result


def summarize_geometry(
    geometry: Mapping[str, np.ndarray],
    *,
    groups: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    layer_ids = np.asarray(geometry["layer_ids"], dtype=np.int64)
    metric_names = [name for name in geometry if name != "layer_ids"]
    sample_count = int(np.asarray(geometry[metric_names[0]]).shape[0])
    masks = {"all": np.ones(sample_count, dtype=bool)}
    for name, value in (groups or {}).items():
        mask = np.asarray(value, dtype=bool).reshape(-1)
        if mask.size != sample_count:
            raise ValueError("geometry group mask has incompatible length")
        masks[str(name)] = mask
    result: Dict[str, Any] = {
        "sample_count": sample_count,
        "layer_ids": layer_ids.tolist(),
        "groups": {},
    }
    for group_name, mask in masks.items():
        group = {"sample_count": int(mask.sum()), "layers": {}}
        for column, layer_id in enumerate(layer_ids.tolist()):
            layer_metrics = {}
            for metric_name in metric_names:
                values = np.asarray(geometry[metric_name], dtype=np.float64)[mask, column]
                layer_metrics[metric_name] = {
                    "mean": float(values.mean()) if values.size else None,
                    "std": float(values.std()) if values.size else None,
                    "min": float(values.min()) if values.size else None,
                    "max": float(values.max()) if values.size else None,
                }
            group["layers"][str(layer_id)] = layer_metrics
        result["groups"][group_name] = group
    return result


def deterministic_donor_indices(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    *,
    relation: str,
    seed: int,
) -> np.ndarray:
    identifiers = [str(value) for value in sample_ids]
    target_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(identifiers) != target_labels.size or not identifiers:
        raise ValueError("donor inputs must have the same non-zero length")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_ids must be unique")
    relation = str(relation).strip().lower()
    if relation not in {"same_class", "different_class"}:
        raise ValueError("relation must be same_class or different_class")
    by_class: Dict[int, list[int]] = {}
    for index, class_id in enumerate(target_labels.tolist()):
        by_class.setdefault(int(class_id), []).append(index)
    all_indices = np.arange(target_labels.size, dtype=np.int64)
    donors = np.empty(target_labels.size, dtype=np.int64)
    for target_index, (sample_id, class_id) in enumerate(
        zip(identifiers, target_labels.tolist())
    ):
        if relation == "same_class":
            candidates = [
                index for index in by_class[int(class_id)] if index != target_index
            ]
        else:
            candidates = all_indices[target_labels != int(class_id)].tolist()
        if not candidates:
            raise ValueError(
                "no {} donor exists for sample {}".format(relation, sample_id)
            )
        digest = hashlib.sha256(
            "{}|{}|{}".format(int(seed), relation, sample_id).encode("utf-8")
        ).digest()
        donors[target_index] = candidates[
            int.from_bytes(digest[:8], byteorder="little") % len(candidates)
        ]
    if relation == "same_class":
        if np.any(donors == all_indices) or np.any(target_labels[donors] != target_labels):
            raise RuntimeError("same-class donor contract failed")
    elif np.any(target_labels[donors] == target_labels):
        raise RuntimeError("different-class donor contract failed")
    return donors
