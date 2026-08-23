from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torchvision as tv

from .eval_metrics import spearman_correlation
from .prompt_analysis import PromptContentSlotAccumulator


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    import json

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_probe_manifest(
    dataset: Any,
    *,
    split: str,
    per_class: int,
    max_samples: int,
    selection_seed: int,
    candidate_class_ids: Sequence[int],
) -> Dict[str, Any]:
    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    candidate_class_set = {int(class_id) for class_id in candidate_class_ids}
    for dataset_index, record in enumerate(dataset._imdb):
        class_id = int(record["class"])
        if class_id not in candidate_class_set:
            continue
        sample_id = str(record.get("sample_id", f"{split}:{dataset_index}"))
        digest = hashlib.sha256(f"{int(selection_seed)}|{sample_id}".encode("utf-8")).hexdigest()
        by_class[class_id].append({
            "dataset_index": int(dataset_index),
            "sample_id": sample_id,
            "image_path": str(record["im_path"]),
            "global_label": class_id,
            "selection_hash": digest,
        })
    available_class_ids = sorted(by_class)
    pre_cap_selected = []
    for class_id in sorted(by_class):
        rows = sorted(by_class[class_id], key=lambda row: (row["selection_hash"], row["sample_id"]))
        pre_cap_selected.extend(rows[: max(1, int(per_class))])
    selected = sorted(pre_cap_selected, key=lambda row: (row["selection_hash"], row["sample_id"]))[: max(1, int(max_samples))]
    selected = sorted(selected, key=lambda row: (row["global_label"], row["sample_id"]))
    selected_support = {class_id: 0 for class_id in available_class_ids}
    for row in selected:
        selected_support[int(row["global_label"])] += 1
    selected_class_ids = [class_id for class_id in available_class_ids if selected_support[class_id] > 0]
    missing_available_class_ids = [class_id for class_id in available_class_ids if selected_support[class_id] == 0]
    per_class_quota = max(1, int(per_class))
    quota_shortfall_class_ids = [
        class_id for class_id in available_class_ids
        if selected_support[class_id] < per_class_quota
    ]
    seen_set = {int(item) for item in getattr(dataset, "seen_classes", [])}
    rows = []
    for order, row in enumerate(selected):
        rows.append({
            **row,
            "sample_order": int(order),
            "seen_unseen_identity": "seen" if int(row["global_label"]) in seen_set else "unseen",
        })
    manifest = {
        "format": "baseline_fixed_probe_v1",
        "probe_id": f"{split}-seed{int(selection_seed)}-n{len(rows)}",
        "split": str(split),
        "selection_seed": int(selection_seed),
        "per_class": int(per_class),
        "max_samples": int(max_samples),
        "candidate_class_ids": [int(item) for item in candidate_class_ids],
        "candidate_class_count": int(len(candidate_class_set)),
        "candidate_class_ids_absent_from_split": sorted(candidate_class_set.difference(available_class_ids)),
        "available_probe_class_ids": available_class_ids,
        "available_probe_class_count": int(len(available_class_ids)),
        "selected_class_ids": selected_class_ids,
        "selected_class_count": int(len(selected_class_ids)),
        "class_coverage_ratio_of_available": (
            float(len(selected_class_ids) / len(available_class_ids)) if available_class_ids else 1.0
        ),
        "missing_available_class_ids": missing_available_class_ids,
        "per_class_available": {str(class_id): int(len(by_class[class_id])) for class_id in available_class_ids},
        "per_class_support": {str(class_id): int(selected_support[class_id]) for class_id in available_class_ids},
        "per_class_quota_satisfied": bool(not quota_shortfall_class_ids),
        "per_class_quota_shortfall_class_ids": quota_shortfall_class_ids,
        "pre_cap_sample_count": int(len(pre_cap_selected)),
        "selected_sample_count": int(len(rows)),
        "max_samples_truncated": bool(len(selected) < len(pre_cap_selected)),
        "input_transform": "Resize->CenterCrop->ToTensor->Normalize",
        "samples": rows,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def validate_probe_manifest(
    manifest: Mapping[str, Any],
    *,
    require_full_class_coverage: bool = True,
    require_per_class_quota: bool = True,
    allow_max_samples_truncation: bool = False,
) -> Dict[str, Any]:
    checks = {
        "selected_samples_present": int(manifest.get("selected_sample_count", 0)) > 0,
        "full_available_class_coverage": (
            not require_full_class_coverage
            or (
                int(manifest.get("selected_class_count", 0))
                == int(manifest.get("available_probe_class_count", 0))
                and float(manifest.get("class_coverage_ratio_of_available", 0.0)) == 1.0
                and not manifest.get("missing_available_class_ids", [])
            )
        ),
        "per_class_quota_satisfied": (
            not require_per_class_quota or bool(manifest.get("per_class_quota_satisfied", False))
        ),
        "max_samples_policy_satisfied": (
            allow_max_samples_truncation or not bool(manifest.get("max_samples_truncated", False))
        ),
    }
    failure_messages = {
        "selected_samples_present": "probe selection produced no samples",
        "full_available_class_coverage": "probe does not cover every class available in this split",
        "per_class_quota_satisfied": "one or more available classes did not reach the per-class quota",
        "max_samples_policy_satisfied": "MAX_SAMPLES truncated the class-balanced probe selection",
    }
    failure_reasons = [failure_messages[name] for name, passed in checks.items() if not passed]
    return {
        "format": "baseline_fixed_probe_validity_v1",
        "probe_id": manifest.get("probe_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "selection_seed": manifest.get("selection_seed"),
        "split": manifest.get("split"),
        "policy": {
            "require_full_class_coverage": bool(require_full_class_coverage),
            "require_per_class_quota": bool(require_per_class_quota),
            "allow_max_samples_truncation": bool(allow_max_samples_truncation),
        },
        "checks": checks,
        "valid": bool(all(checks.values())),
        "failure_reasons": failure_reasons,
    }


class FixedProbeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_dataset: Any,
        manifest: Mapping[str, Any],
        transform: Any,
        *,
        cache_transformed_images: bool = False,
    ) -> None:
        self.source_dataset = source_dataset
        self.manifest = dict(manifest)
        self.rows = list(manifest.get("samples", []))
        self.transform = transform
        self.cache_transformed_images = bool(cache_transformed_images)
        self._transformed_image_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        row = self.rows[index]
        image = self._transformed_image_cache.get(index)
        if image is None:
            image = tv.datasets.folder.default_loader(str(row["image_path"]))
            image = self.transform(image)
            if self.cache_transformed_images:
                # Fixed-Probe transforms are deterministic.  The cached CPU
                # tensor is read-only from the dataset's perspective and is
                # reused only within this process/checkpoint execution.
                self._transformed_image_cache[index] = image
        label = int(row["global_label"])
        return {
            "image": image,
            "label": label,
            "attribute": self.source_dataset.class_attributes[label],
            "sample_id": str(row["sample_id"]),
            "sample_index": int(row["dataset_index"]),
            "image_path": str(row["image_path"]),
        }


def _entropy_and_mass(scores: torch.Tensor, temperature: float) -> Dict[str, float]:
    probabilities = torch.softmax(scores.float() / max(float(temperature), 1e-8), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / max(math.log(max(2, int(probabilities.shape[-1]))), 1e-12)
    sorted_prob = probabilities.sort(dim=-1, descending=True).values
    return {
        "normalized_row_entropy": float(entropy.mean().item()),
        "top1_mass": float(sorted_prob[..., :1].sum(dim=-1).mean().item()),
        "top5_mass": float(sorted_prob[..., : min(5, sorted_prob.shape[-1])].sum(dim=-1).mean().item()),
    }


def _matrix_effective_rank(values: torch.Tensor) -> float:
    matrix = values.detach().float().reshape(-1, values.shape[-2], values.shape[-1])
    ranks = []
    for item in matrix[: min(128, matrix.shape[0])]:
        singular = (
            torch.linalg.svdvals(item)
            if hasattr(torch.linalg, "svdvals")
            else torch.svd(item, some=False).S
        )
        total = singular.sum()
        if float(total.item()) <= 1e-12:
            continue
        prob = singular / total
        ranks.append(torch.exp(-(prob * prob.clamp_min(1e-12).log()).sum()))
    return float(torch.stack(ranks).mean().item()) if ranks else 0.0


def _head_diversity(values: torch.Tensor) -> float:
    if values.dim() < 4 or values.shape[1] < 2:
        return 0.0
    flattened = values.detach().float().reshape(values.shape[0], values.shape[1], -1)
    flattened = torch.nn.functional.normalize(flattened, dim=-1)
    similarity = torch.matmul(flattened, flattened.transpose(1, 2))
    mask = ~torch.eye(similarity.shape[1], dtype=torch.bool, device=similarity.device)
    offdiag = similarity[:, mask]
    return float((1.0 - offdiag).mean().item()) if offdiag.numel() else 0.0


class _RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update_values(self, values: torch.Tensor) -> None:
        data = values.detach().reshape(-1)
        if data.numel() == 0:
            return
        self.total += float(data.sum().item())
        self.count += int(data.numel())

    def update_scalar(self, value: float, weight: int = 1) -> None:
        if int(weight) <= 0:
            return
        self.total += float(value) * int(weight)
        self.count += int(weight)

    def value(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return float(self.total / self.count)


class _RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: torch.Tensor) -> None:
        data = values.detach().reshape(-1)
        batch_count = int(data.numel())
        if batch_count <= 0:
            return
        batch_mean = float(data.mean().item())
        batch_m2 = float(((data - batch_mean) ** 2).sum().item())
        if self.count <= 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total_count
        self.count = total_count

    def std(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return float(math.sqrt(max(0.0, self.m2 / self.count)))


class _RunningVectorVariance:
    def __init__(self) -> None:
        self.count = 0
        self.vector_sum: Optional[torch.Tensor] = None
        self.squared_norm_sum = 0.0

    def update(self, values: torch.Tensor) -> None:
        data = values.detach().to(device="cpu", dtype=torch.float32)
        if data.dim() != 2 or data.numel() == 0:
            return
        vector_sum = data.sum(dim=0)
        if self.vector_sum is None:
            self.vector_sum = vector_sum
        else:
            if self.vector_sum.numel() != vector_sum.numel():
                raise ValueError("Prompt layer vector size changed across probe batches.")
            self.vector_sum += vector_sum
        self.squared_norm_sum += float(data.pow(2).sum().item())
        self.count += int(data.shape[0])

    def value(self) -> Optional[float]:
        if self.count <= 0 or self.vector_sum is None:
            return None
        mean_squared_norm = self.squared_norm_sum / self.count
        squared_mean_norm = float((self.vector_sum / self.count).pow(2).sum().item())
        return float(max(0.0, mean_squared_norm - squared_mean_norm))


class _CappedEffectiveRank:
    def __init__(self, limit: int = 128) -> None:
        self.limit = max(1, int(limit))
        self.examined = 0
        self.total = 0.0
        self.count = 0

    def update(self, values: torch.Tensor) -> None:
        if values.dim() < 2 or self.examined >= self.limit:
            return
        matrices = values.detach().float().reshape(-1, values.shape[-2], values.shape[-1])
        take = min(int(matrices.shape[0]), self.limit - self.examined)
        for item in matrices[:take]:
            self.examined += 1
            singular = (
                torch.linalg.svdvals(item)
                if hasattr(torch.linalg, "svdvals")
                else torch.svd(item, some=False).S
            )
            total = singular.sum()
            if float(total.item()) <= 1e-12:
                continue
            probability = singular / total
            rank = torch.exp(
                -(probability * probability.clamp_min(1e-12).log()).sum()
            )
            self.total += float(rank.item())
            self.count += 1

    def value(self) -> float:
        return float(self.total / self.count) if self.count > 0 else 0.0


def _entropy_and_mass_values(scores: torch.Tensor, temperature: float) -> Dict[str, torch.Tensor]:
    probabilities = torch.softmax(scores.float() / max(float(temperature), 1e-8), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / max(math.log(max(2, int(probabilities.shape[-1]))), 1e-12)
    sorted_probability = probabilities.sort(dim=-1, descending=True).values
    return {
        "normalized_row_entropy": entropy,
        "top1_mass": sorted_probability[..., :1].sum(dim=-1),
        "top5_mass": sorted_probability[..., : min(5, sorted_probability.shape[-1])].sum(dim=-1),
    }


def _conditional_attention_values(
    values: torch.Tensor,
    *,
    topk: int,
) -> Dict[str, torch.Tensor]:
    data = values.detach().float()
    support_size = int(data.shape[-1])
    mass = data.sum(dim=-1, keepdim=True)
    valid = mass > 1e-12
    normalized = data / mass.clamp_min(1e-12)
    entropy_nats = -(
        normalized * normalized.clamp_min(1e-12).log()
    ).sum(dim=-1)
    normalized_entropy = entropy_nats / max(
        math.log(max(2, support_size)), 1e-12
    )
    effective_count = entropy_nats.exp()
    sorted_mass = normalized.sort(dim=-1, descending=True).values
    selected_mass = sorted_mass[..., : min(max(1, int(topk)), support_size)].sum(
        dim=-1
    )
    valid_rows = valid.squeeze(-1)
    zeros = torch.zeros_like(entropy_nats)
    return {
        "conditional_entropy": torch.where(valid_rows, normalized_entropy, zeros),
        "conditional_topk_mass": torch.where(valid_rows, selected_mass, zeros),
        "effective_count": torch.where(valid_rows, effective_count, zeros),
        "effective_ratio": torch.where(
            valid_rows,
            effective_count / max(1, support_size),
            zeros,
        ),
    }


def _patch_prompt_coordination_values(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    data = values.detach().float()
    prompt_count = int(data.shape[-1])
    patch_count = int(data.shape[-2])
    row_mass = data.sum(dim=-1, keepdim=True)
    valid_rows = row_mass.squeeze(-1) > 1e-12
    conditional = data / row_mass.clamp_min(1e-12)
    conditional = conditional * valid_rows.unsqueeze(-1).to(dtype=conditional.dtype)
    valid_count = valid_rows.sum(dim=-1)
    valid_count_float = valid_count.to(dtype=conditional.dtype)

    mean_usage = conditional.sum(dim=-2) / valid_count_float.clamp_min(1.0).unsqueeze(-1)
    mean_usage = mean_usage / mean_usage.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy_denominator = max(math.log(max(2, prompt_count)), 1e-12)
    global_usage_entropy = -(
        mean_usage * mean_usage.clamp_min(1e-12).log()
    ).sum(dim=-1) / entropy_denominator

    row_entropy = -(
        conditional * conditional.clamp_min(1e-12).log()
    ).sum(dim=-1) / entropy_denominator
    mean_row_entropy = (
        row_entropy * valid_rows.to(dtype=row_entropy.dtype)
    ).sum(dim=-1) / valid_count_float.clamp_min(1.0)
    coordination_information = (global_usage_entropy - mean_row_entropy).clamp(
        min=0.0, max=1.0
    )

    distribution_direction = conditional / conditional.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    direction_sum = distribution_direction.sum(dim=-2)
    pairwise_numerator = (
        direction_sum.square().sum(dim=-1)
        - distribution_direction.square().sum(dim=(-2, -1))
    )
    pairwise_denominator = valid_count_float * (valid_count_float - 1.0)
    selection_overlap = (
        pairwise_numerator / pairwise_denominator.clamp_min(1.0)
    ).clamp(min=0.0, max=1.0)

    top1_prompt = conditional.argmax(dim=-1)
    top1_one_hot = torch.nn.functional.one_hot(
        top1_prompt,
        num_classes=prompt_count,
    ).to(dtype=conditional.dtype)
    top1_one_hot = top1_one_hot * valid_rows.unsqueeze(-1).to(
        dtype=conditional.dtype
    )
    used_prompt = top1_one_hot.max(dim=-2).values
    unique_count = used_prompt.sum(dim=-1)
    maximum_unique = valid_count_float.clamp(max=float(min(patch_count, prompt_count)))
    repeat_rate = 1.0 - unique_count / maximum_unique.clamp_min(1.0)

    has_valid_rows = valid_count > 0
    zeros = torch.zeros_like(global_usage_entropy)
    return {
        "global_usage_entropy": torch.where(
            has_valid_rows, global_usage_entropy, zeros
        ),
        "coordination_information": torch.where(
            has_valid_rows, coordination_information, zeros
        ),
        "selection_overlap": torch.where(
            valid_count > 1, selection_overlap, zeros
        ),
        "top1_unique_count": torch.where(has_valid_rows, unique_count, zeros),
        "top1_coverage_ratio": torch.where(
            has_valid_rows,
            unique_count / max(1, prompt_count),
            zeros,
        ),
        "top1_repeat_rate": torch.where(has_valid_rows, repeat_rate, zeros),
    }


def _prompt_patch_retrieval_values(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    data = values.detach().float()
    coordination = _patch_prompt_coordination_values(data)
    result = {
        "global_patch_usage_entropy": coordination["global_usage_entropy"],
        "coordination_information": coordination["coordination_information"],
        "selection_overlap": coordination["selection_overlap"],
        "top1_unique_patch_count": coordination["top1_unique_count"],
        "top1_patch_coverage_ratio": coordination["top1_coverage_ratio"],
        "top1_repeat_rate": coordination["top1_repeat_rate"],
    }

    prompt_count = int(data.shape[-2])
    patch_count = int(data.shape[-1])
    grid_size = int(math.sqrt(max(0, patch_count)))
    if prompt_count <= 0 or patch_count <= 0 or grid_size * grid_size != patch_count:
        return result

    axis = torch.arange(grid_size, device=data.device, dtype=data.dtype)
    axis = axis / float(max(1, grid_size - 1))
    grid_y, grid_x = torch.meshgrid(axis, axis)
    positions = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

    row_mass = data.sum(dim=-1, keepdim=True)
    valid_rows = row_mass.squeeze(-1) > 1e-12
    conditional = data / row_mass.clamp_min(1e-12)
    conditional = conditional * valid_rows.unsqueeze(-1).to(conditional.dtype)
    centroids = torch.matmul(conditional, positions)

    pairwise_distance = (
        centroids.unsqueeze(-2) - centroids.unsqueeze(-3)
    ).norm(dim=-1) / math.sqrt(2.0)
    pair_mask = valid_rows.unsqueeze(-1) & valid_rows.unsqueeze(-2)
    diagonal = torch.eye(
        prompt_count, device=data.device, dtype=torch.bool
    ).view(*([1] * (pair_mask.dim() - 2)), prompt_count, prompt_count)
    pair_mask = pair_mask & (~diagonal)
    pair_count = pair_mask.sum(dim=(-2, -1))
    centroid_dispersion = (
        pairwise_distance * pair_mask.to(pairwise_distance.dtype)
    ).sum(dim=(-2, -1)) / pair_count.clamp_min(1).to(pairwise_distance.dtype)

    patch_distance = (
        positions.view(*([1] * (centroids.dim() - 1)), patch_count, 2)
        - centroids.unsqueeze(-2)
    ).square().sum(dim=-1)
    retrieval_radius = (
        (conditional * patch_distance).sum(dim=-1).clamp_min(0.0).sqrt()
        / math.sqrt(2.0)
    )
    valid_count = valid_rows.sum(dim=-1)
    mean_radius = (
        retrieval_radius * valid_rows.to(retrieval_radius.dtype)
    ).sum(dim=-1) / valid_count.clamp_min(1).to(retrieval_radius.dtype)
    zeros = torch.zeros_like(mean_radius)
    result.update({
        "spatial_centroid_dispersion": torch.where(
            pair_count > 0, centroid_dispersion, zeros
        ),
        "spatial_radius": torch.where(valid_count > 0, mean_radius, zeros),
    })
    return result


def _head_diversity_values(values: torch.Tensor) -> Optional[torch.Tensor]:
    if values.dim() < 4 or values.shape[1] < 2:
        return None
    flattened = values.detach().float().reshape(values.shape[0], values.shape[1], -1)
    flattened = torch.nn.functional.normalize(flattened, dim=-1)
    similarity = torch.matmul(flattened, flattened.transpose(1, 2))
    mask = ~torch.eye(similarity.shape[1], dtype=torch.bool, device=similarity.device)
    offdiag = similarity[:, mask]
    if not offdiag.numel():
        return None
    return (1.0 - offdiag).mean(dim=1)


def _normalized_patch_attention_distance(
    attention: torch.Tensor,
    patch_slice: slice,
) -> Optional[torch.Tensor]:
    patch_attention = attention[:, :, patch_slice, patch_slice]
    patch_count = int(patch_attention.shape[-1])
    side = int(round(math.sqrt(patch_count)))
    if patch_count <= 1 or side * side != patch_count:
        return None
    positions = torch.arange(patch_count, device=attention.device)
    coordinates = torch.stack((
        positions // side,
        positions.remainder(side),
    ), dim=-1).to(dtype=torch.float32)
    distance = torch.cdist(coordinates, coordinates, p=2)
    distance = distance / max(math.sqrt(2.0) * float(side - 1), 1e-12)
    normalized = patch_attention.float()
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (normalized * distance).sum(dim=-1)


def _attention_path_slices(
    sequence_length: int,
    prompt_length: int,
    semantic_length: int,
) -> Dict[str, tuple]:
    prompt_length = max(0, int(prompt_length))
    semantic_length = max(0, int(semantic_length))
    patch_start = 1 + prompt_length
    patch_end = int(sequence_length) - semantic_length
    if patch_end < patch_start:
        return {}
    cls_slice = slice(0, 1)
    prompt_slice = slice(1, patch_start)
    patch_slice = slice(patch_start, patch_end)
    semantic_slice = slice(patch_end, int(sequence_length))
    paths = {
        "cls_to_patch": (cls_slice, patch_slice),
        "patch_to_cls": (patch_slice, cls_slice),
        "patch_to_patch": (patch_slice, patch_slice),
    }
    if prompt_length > 0:
        paths.update({
            "cls_to_prompt": (cls_slice, prompt_slice),
            "prompt_to_cls": (prompt_slice, cls_slice),
            "prompt_to_patch": (prompt_slice, patch_slice),
            "patch_to_prompt": (patch_slice, prompt_slice),
            "prompt_to_prompt": (prompt_slice, prompt_slice),
        })
    if semantic_length > 0:
        paths.update({
            "cls_to_semantic": (cls_slice, semantic_slice),
            "semantic_to_cls": (semantic_slice, cls_slice),
            "semantic_to_patch": (semantic_slice, patch_slice),
            "patch_to_semantic": (patch_slice, semantic_slice),
            "semantic_to_semantic": (semantic_slice, semantic_slice),
        })
        if prompt_length > 0:
            paths.update({
                "prompt_to_semantic": (prompt_slice, semantic_slice),
                "semantic_to_prompt": (semantic_slice, prompt_slice),
            })
    return paths


class _AttentionLayerAccumulator:
    CONTENT_METRICS = (
        "cls_prompt_value_contribution_norm",
        "cls_patch_value_contribution_norm",
        "cls_prompt_value_contribution_share",
        "cls_prompt_value_to_total_cosine",
        "cls_prompt_value_to_patch_cosine",
        "cls_prompt_value_to_cls_delta_cosine",
        "prompt_patch_value_contribution_norm",
        "prompt_patch_value_contribution_share",
        "prompt_patch_value_to_total_cosine",
        "prompt_patch_value_to_prompt_delta_cosine",
    )
    INTERVENTION_METRICS = (
        "prompt_patch_uniform_applied",
        "prompt_patch_uniform_mass_abs_error",
        "patch_prompt_uniform_applied",
        "patch_prompt_uniform_mass_abs_error",
        "prompt_value_globalize_applied",
        "prompt_value_globalize_patch_value_dispersion_before",
        "prompt_value_globalize_patch_value_dispersion_after",
        "prompt_value_globalize_prompt_context_delta_norm",
        "prompt_value_globalize_attention_mass_abs_error",
        "prompt_read_block_applied",
        "prompt_read_block_patch_mass_before",
        "prompt_read_block_patch_mass_after",
        "prompt_value_zero_applied",
        "prompt_value_zero_context_delta_norm",
        "prompt_value_zero_attention_mass_abs_error",
        "relevance_delete_applied",
        "relevance_delete_mass",
        "relevance_delete_edge_ratio",
        "relevance_delete_row_mass_abs_error",
    )

    def __init__(self) -> None:
        self.means: Dict[str, _RunningMean] = defaultdict(_RunningMean)
        self.effective_rank = _CappedEffectiveRank()
        self.observed = False

    def _update_shared(self, data: torch.Tensor) -> None:
        diversity = _head_diversity_values(data)
        if diversity is None:
            self.means["head_diversity"].update_scalar(0.0)
        else:
            self.means["head_diversity"].update_values(diversity)
        self.effective_rank.update(data)
        self.observed = True

    def update_attention(
        self,
        data: torch.Tensor,
        *,
        prompt_length: int,
        semantic_length: int,
    ) -> Optional[torch.Tensor]:
        if data.dim() != 4 or data.numel() == 0:
            return None
        sequence_length = int(data.shape[-1])
        prompt_slice = slice(1, 1 + int(prompt_length))
        patch_start = 1 + int(prompt_length)
        patch_end = sequence_length - int(semantic_length)
        patch_slice = slice(patch_start, patch_end)
        cls_patch = data[:, :, 0, patch_slice]
        sample_entropy = None
        patch_distance = _normalized_patch_attention_distance(data, patch_slice)
        if patch_distance is not None:
            self.means["patch_to_patch_attention_distance"].update_values(patch_distance)
        if cls_patch.numel():
            self.means["cls_to_patch_mass"].update_values(cls_patch.sum(dim=-1))
            normalized = cls_patch / cls_patch.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy / max(math.log(max(2, normalized.shape[-1])), 1e-12)
            sorted_mass = normalized.sort(dim=-1, descending=True).values
            self.means["cls_to_patch_entropy"].update_values(entropy)
            self.means["cls_to_patch_topk_mass"].update_values(
                sorted_mass[..., : min(5, sorted_mass.shape[-1])].sum(dim=-1)
            )
            sample_entropy = entropy.mean(dim=1)
        if int(prompt_length) > 0:
            cls_to_prompt = data[:, :, 0, prompt_slice]
            prompt_to_patch = data[:, :, prompt_slice, patch_slice]
            patch_to_prompt = data[:, :, patch_slice, prompt_slice]
            prompt_to_cls = data[:, :, prompt_slice, 0]
            self.means["cls_to_prompt_mass"].update_values(cls_to_prompt.sum(dim=-1))
            self.means["prompt_to_patch_mass"].update_values(prompt_to_patch.sum(dim=-1))
            self.means["patch_to_prompt_mass"].update_values(patch_to_prompt.sum(dim=-1))
            self.means["prompt_to_cls_mass"].update_values(prompt_to_cls)
            prompt_patch_selection = _conditional_attention_values(
                prompt_to_patch,
                topk=5,
            )
            self.means["prompt_to_patch_conditional_entropy"].update_values(
                prompt_patch_selection["conditional_entropy"]
            )
            self.means["prompt_to_patch_conditional_top5_mass"].update_values(
                prompt_patch_selection["conditional_topk_mass"]
            )
            self.means["prompt_to_patch_effective_patch_count"].update_values(
                prompt_patch_selection["effective_count"]
            )
            self.means["prompt_to_patch_effective_patch_ratio"].update_values(
                prompt_patch_selection["effective_ratio"]
            )
            prompt_patch_retrieval = _prompt_patch_retrieval_values(
                prompt_to_patch
            )
            for metric_name, metric_values in prompt_patch_retrieval.items():
                self.means[
                    f"prompt_patch_retrieval_{metric_name}"
                ].update_values(metric_values)
            patch_prompt_selection = _conditional_attention_values(
                patch_to_prompt,
                topk=1,
            )
            self.means["patch_to_prompt_conditional_entropy"].update_values(
                patch_prompt_selection["conditional_entropy"]
            )
            self.means["patch_to_prompt_top1_share"].update_values(
                patch_prompt_selection["conditional_topk_mass"]
            )
            self.means["patch_to_prompt_effective_prompt_count"].update_values(
                patch_prompt_selection["effective_count"]
            )
            self.means["patch_to_prompt_effective_prompt_ratio"].update_values(
                patch_prompt_selection["effective_ratio"]
            )
            patch_prompt_coordination = _patch_prompt_coordination_values(
                patch_to_prompt
            )
            for metric_name, metric_values in patch_prompt_coordination.items():
                self.means[f"patch_prompt_{metric_name}"].update_values(
                    metric_values
                )
            prompt_usage = _conditional_attention_values(
                cls_to_prompt,
                topk=1,
            )
            self.means["configured_prompt_count"].update_scalar(float(prompt_length))
            self.means["cls_to_prompt_conditional_entropy"].update_values(
                prompt_usage["conditional_entropy"]
            )
            self.means["cls_to_prompt_top1_share"].update_values(
                prompt_usage["conditional_topk_mass"]
            )
            self.means["cls_to_prompt_effective_prompt_count"].update_values(
                prompt_usage["effective_count"]
            )
            self.means["cls_to_prompt_effective_prompt_ratio"].update_values(
                prompt_usage["effective_ratio"]
            )
        self._update_shared(data)
        return sample_entropy

    def update_affinity_fallback(self, layer: Mapping[str, Any]) -> Optional[torch.Tensor]:
        cls_patch = layer.get("AcKv_attn")
        sample_entropy = None
        if torch.is_tensor(cls_patch) and cls_patch.numel():
            data = cls_patch.detach().float()
            normalized = data / data.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy / max(math.log(max(2, normalized.shape[-1])), 1e-12)
            sorted_mass = normalized.sort(dim=-1, descending=True).values
            self.means["cls_to_patch_entropy"].update_values(entropy)
            self.means["cls_to_patch_topk_mass"].update_values(
                sorted_mass[..., : min(5, sorted_mass.shape[-1])].sum(dim=-1)
            )
            self.means["cls_to_patch_mass"].update_values(data.sum(dim=-1))
            self._update_shared(data)
            sample_entropy = entropy.mean(dim=1).squeeze(-1)
        relation_metrics = {
            "AcKp_attn": "cls_to_prompt_mass",
            "ApKv_attn": "prompt_to_patch_mass",
            "AvKp_attn": "patch_to_prompt_mass",
            "ApKc_attn": "prompt_to_cls_mass",
            "AsKv_attn": "semantic_to_patch_mass",
            "AvKs_attn": "patch_to_semantic_mass",
            "AsKp_attn": "semantic_to_prompt_mass",
            "ApKs_attn": "prompt_to_semantic_mass",
        }
        for key, metric_name in relation_metrics.items():
            value = layer.get(key)
            if torch.is_tensor(value) and value.numel():
                data = value.detach().float()
                self.means[metric_name].update_values(data.sum(dim=-1))
                if key == "ApKv_attn":
                    selection = _conditional_attention_values(data, topk=5)
                    self.means["prompt_to_patch_conditional_entropy"].update_values(
                        selection["conditional_entropy"]
                    )
                    self.means["prompt_to_patch_conditional_top5_mass"].update_values(
                        selection["conditional_topk_mass"]
                    )
                    self.means["prompt_to_patch_effective_patch_count"].update_values(
                        selection["effective_count"]
                    )
                    self.means["prompt_to_patch_effective_patch_ratio"].update_values(
                        selection["effective_ratio"]
                    )
                    retrieval = _prompt_patch_retrieval_values(data)
                    for metric_name, metric_values in retrieval.items():
                        self.means[
                            f"prompt_patch_retrieval_{metric_name}"
                        ].update_values(metric_values)
                elif key == "AvKp_attn":
                    selection = _conditional_attention_values(data, topk=1)
                    self.means["patch_to_prompt_conditional_entropy"].update_values(
                        selection["conditional_entropy"]
                    )
                    self.means["patch_to_prompt_top1_share"].update_values(
                        selection["conditional_topk_mass"]
                    )
                    self.means["patch_to_prompt_effective_prompt_count"].update_values(
                        selection["effective_count"]
                    )
                    self.means["patch_to_prompt_effective_prompt_ratio"].update_values(
                        selection["effective_ratio"]
                    )
                    coordination = _patch_prompt_coordination_values(data)
                    for metric_name, metric_values in coordination.items():
                        self.means[f"patch_prompt_{metric_name}"].update_values(
                            metric_values
                        )
                elif key == "AcKp_attn":
                    usage = _conditional_attention_values(data, topk=1)
                    self.means["configured_prompt_count"].update_scalar(
                        float(data.shape[-1])
                    )
                    self.means["cls_to_prompt_conditional_entropy"].update_values(
                        usage["conditional_entropy"]
                    )
                    self.means["cls_to_prompt_top1_share"].update_values(
                        usage["conditional_topk_mass"]
                    )
                    self.means["cls_to_prompt_effective_prompt_count"].update_values(
                        usage["effective_count"]
                    )
                    self.means["cls_to_prompt_effective_prompt_ratio"].update_values(
                        usage["effective_ratio"]
                    )
                self.observed = True
        return sample_entropy

    def update_content_contribution(self, layer: Mapping[str, Any]) -> None:
        for metric_name in self.CONTENT_METRICS:
            value = layer.get(metric_name)
            if torch.is_tensor(value) and value.numel():
                self.means[metric_name].update_values(value.detach().float())

    def update_intervention_stats(self, layer: Mapping[str, Any]) -> None:
        for metric_name in self.INTERVENTION_METRICS:
            value = layer.get(metric_name)
            if torch.is_tensor(value) and value.numel():
                self.means[metric_name].update_values(value.detach().float())

    def finalize(self) -> Dict[str, float]:
        result = {}
        for name, state in self.means.items():
            value = state.value()
            if value is not None:
                result[name] = float(value)
        if self.observed:
            result["attention_effective_rank"] = self.effective_rank.value()
        return result


class TargetRelevanceAccumulator:
    METRIC_NAMES = (
        "positive_sum",
        "negative_abs_sum",
        "net_sum",
        "absolute_sum",
        "positive_edge_fraction",
        "layer_absolute_share",
        "layer_normalized_net",
    )

    def __init__(
        self,
        *,
        prompt_length: int,
        semantic_length: int,
        selected_layers: Sequence[int],
        margin_metric_name: str = "true_margin_mean",
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.semantic_length = int(semantic_length)
        self.selected_layers = {int(item) for item in selected_layers}
        self.margin_metric_name = str(margin_metric_name)
        self.states: Dict[int, Dict[str, Dict[str, Dict[str, _RunningMean]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(_RunningMean)))
        )
        self.group_counts: Dict[str, int] = defaultdict(int)
        self.margin_means: Dict[str, _RunningMean] = defaultdict(_RunningMean)
        self.observed_layers = set()

    @staticmethod
    def _group_masks(correct: torch.Tensor) -> Dict[str, torch.Tensor]:
        correct = correct.detach().to(dtype=torch.bool).reshape(-1)
        return {
            "all": torch.ones_like(correct, dtype=torch.bool),
            "correct": correct,
            "wrong": ~correct,
        }

    def update_target(self, margins: torch.Tensor, correct: torch.Tensor) -> None:
        margins = margins.detach().float().reshape(-1)
        for group, mask in self._group_masks(correct).items():
            if bool(mask.any()):
                self.margin_means[group].update_values(margins[mask])
                self.group_counts[group] += int(mask.sum().item())

    def update_layer(
        self,
        layer_index: int,
        attention: torch.Tensor,
        gradient: torch.Tensor,
        correct: torch.Tensor,
    ) -> bool:
        layer_index = int(layer_index)
        if self.selected_layers and layer_index not in self.selected_layers:
            return False
        if not torch.is_tensor(attention) or not torch.is_tensor(gradient):
            return False
        if attention.shape != gradient.shape or attention.dim() != 4 or attention.numel() == 0:
            return False
        relevance = attention.detach().float() * gradient.detach().float()
        layer_absolute = relevance.reshape(relevance.shape[0], -1).abs().sum(dim=-1).clamp_min(1e-12)
        paths = _attention_path_slices(
            int(relevance.shape[-1]),
            self.prompt_length,
            self.semantic_length,
        )
        group_masks = self._group_masks(correct)
        observed = False

        def update_path(path_name: str, values: torch.Tensor) -> None:
            nonlocal observed
            values = values.reshape(relevance.shape[0], -1)
            if values.numel() == 0:
                return
            positive = values.clamp_min(0.0).sum(dim=-1)
            negative_abs = (-values.clamp_max(0.0)).sum(dim=-1)
            net = values.sum(dim=-1)
            absolute = values.abs().sum(dim=-1)
            metrics = {
                "positive_sum": positive,
                "negative_abs_sum": negative_abs,
                "net_sum": net,
                "absolute_sum": absolute,
                "positive_edge_fraction": (values > 0).float().mean(dim=-1),
                "layer_absolute_share": absolute / layer_absolute,
                "layer_normalized_net": net / layer_absolute,
            }
            for group, mask in group_masks.items():
                if not bool(mask.any()):
                    continue
                for metric_name, metric_values in metrics.items():
                    self.states[layer_index][path_name][group][metric_name].update_values(
                        metric_values[mask]
                    )
            observed = True

        for path_name, (query_slice, key_slice) in paths.items():
            update_path(path_name, relevance[:, :, query_slice, key_slice])

        sequence_length = int(relevance.shape[-1])
        prompt_end = min(1 + self.prompt_length, sequence_length)
        patch_start = prompt_end
        patch_end = max(patch_start, sequence_length - self.semantic_length)
        for prompt_index, token_index in enumerate(range(1, prompt_end)):
            prompt_slice = slice(token_index, token_index + 1)
            prompt_name = f"prompt_{prompt_index}"
            update_path(
                f"cls_to_prompt/{prompt_name}",
                relevance[:, :, :1, prompt_slice],
            )
            update_path(
                f"prompt_to_cls/{prompt_name}",
                relevance[:, :, prompt_slice, :1],
            )
            if patch_end > patch_start:
                patch_slice = slice(patch_start, patch_end)
                update_path(
                    f"prompt_to_patch/{prompt_name}",
                    relevance[:, :, prompt_slice, patch_slice],
                )
                update_path(
                    f"patch_to_prompt/{prompt_name}",
                    relevance[:, :, patch_slice, prompt_slice],
                )
        if observed:
            self.observed_layers.add(layer_index)
        return observed

    def finalize_target(self) -> Dict[str, float]:
        result = {}
        all_count = int(self.group_counts.get("all", 0))
        for group in ("all", "correct", "wrong"):
            count = int(self.group_counts.get(group, 0))
            if count <= 0:
                continue
            result[f"{group}.sample_count"] = float(count)
            margin = self.margin_means[group].value()
            if margin is not None:
                result[f"{group}.{self.margin_metric_name}"] = float(margin)
        if all_count > 0:
            result["correct_rate"] = float(self.group_counts.get("correct", 0) / all_count)
        return result

    def finalize_by_layer(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        result = {}
        for layer_index in sorted(self.states):
            layer_result = {}
            for path_name in sorted(self.states[layer_index]):
                path_result = {}
                for group in ("all", "correct", "wrong"):
                    group_states = self.states[layer_index][path_name].get(group, {})
                    for metric_name in self.METRIC_NAMES:
                        state = group_states.get(metric_name)
                        value = state.value() if state is not None else None
                        if value is not None:
                            path_result[f"{group}.{metric_name}"] = float(value)
                if path_result:
                    layer_result[path_name] = path_result
            if layer_result:
                result[int(layer_index)] = layer_result
        return result


class _AttentionFlowAccumulator:
    def __init__(
        self,
        *,
        prompt_length: int,
        semantic_length: int,
        selected_layers: Sequence[int],
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.semantic_length = int(semantic_length)
        self.selected_layers = {int(item) for item in selected_layers}
        self.primary_layers: Dict[int, _AttentionLayerAccumulator] = {}
        self.fallback_layers: Dict[int, _AttentionLayerAccumulator] = {}
        self.content_layers: Dict[int, _AttentionLayerAccumulator] = {}
        self.primary_correct = _RunningMean()
        self.primary_wrong = _RunningMean()
        self.fallback_correct = _RunningMean()
        self.fallback_wrong = _RunningMean()
        self.primary_entropy_observed = False

    def _selected(self, layer_index: int) -> bool:
        return not self.selected_layers or int(layer_index) in self.selected_layers

    @staticmethod
    def _update_correct_wrong(
        sample_entropies: Sequence[torch.Tensor],
        predictions: Sequence[int],
        targets: Sequence[int],
        correct_state: _RunningMean,
        wrong_state: _RunningMean,
    ) -> bool:
        valid = [item.reshape(-1) for item in sample_entropies if torch.is_tensor(item)]
        if not valid:
            return False
        sample_count = min(int(item.numel()) for item in valid)
        if sample_count <= 0:
            return False
        entropy = torch.stack([item[:sample_count] for item in valid], dim=0).mean(dim=0)
        pred = torch.as_tensor(predictions, dtype=torch.long).reshape(-1)[:sample_count]
        target = torch.as_tensor(targets, dtype=torch.long).reshape(-1)[:sample_count]
        if pred.numel() != sample_count or target.numel() != sample_count:
            return False
        correct = pred == target
        if bool(correct.any()):
            correct_state.update_values(entropy[correct])
        if bool((~correct).any()):
            wrong_state.update_values(entropy[~correct])
        return True

    def update(
        self,
        attention_layers: Sequence[Any],
        affinity_layers: Sequence[Any],
        *,
        predictions: Sequence[int],
        targets: Sequence[int],
    ) -> None:
        primary_entropy = []
        for layer_index, attention in enumerate(attention_layers or []):
            if not self._selected(layer_index) or not torch.is_tensor(attention):
                continue
            data = attention.detach().to(device="cpu", dtype=torch.float32)
            state = self.primary_layers.setdefault(layer_index, _AttentionLayerAccumulator())
            entropy = state.update_attention(
                data,
                prompt_length=self.prompt_length,
                semantic_length=self.semantic_length,
            )
            if entropy is not None:
                primary_entropy.append(entropy)
            del data
        if self._update_correct_wrong(
            primary_entropy,
            predictions,
            targets,
            self.primary_correct,
            self.primary_wrong,
        ):
            self.primary_entropy_observed = True

        fallback_entropy = []
        for layer_index, affinity in enumerate(affinity_layers or []):
            if not self._selected(layer_index) or not isinstance(affinity, Mapping):
                continue
            cpu_layer = {
                key: value.detach().to(device="cpu", dtype=torch.float32)
                for key, value in affinity.items()
                if torch.is_tensor(value)
            }
            state = self.fallback_layers.setdefault(layer_index, _AttentionLayerAccumulator())
            entropy = state.update_affinity_fallback(cpu_layer)
            if entropy is not None:
                fallback_entropy.append(entropy)
            del cpu_layer
        self._update_correct_wrong(
            fallback_entropy,
            predictions,
            targets,
            self.fallback_correct,
            self.fallback_wrong,
        )

    def update_content_contribution(self, affinity_layers: Sequence[Any]) -> None:
        for layer_index, affinity in enumerate(affinity_layers or []):
            if not self._selected(layer_index) or not isinstance(affinity, Mapping):
                continue
            state = self.content_layers.setdefault(layer_index, _AttentionLayerAccumulator())
            state.update_content_contribution(affinity)

    def update_intervention_stats(self, affinity_layers: Sequence[Any]) -> None:
        for layer_index, affinity in enumerate(affinity_layers or []):
            if not self._selected(layer_index) or not isinstance(affinity, Mapping):
                continue
            state = self.content_layers.setdefault(
                layer_index, _AttentionLayerAccumulator()
            )
            state.update_intervention_stats(affinity)

    @staticmethod
    def _finalize_layers(states: Iterable[_AttentionLayerAccumulator]) -> Dict[str, float]:
        buckets: Dict[str, List[float]] = defaultdict(list)
        for state in states:
            for name, value in state.finalize().items():
                buckets[name].append(float(value))
        return {
            name: float(sum(values) / len(values))
            for name, values in buckets.items()
            if values
        }

    def finalize_by_layer(self) -> Dict[int, Dict[str, float]]:
        states = self.primary_layers if self.primary_entropy_observed else self.fallback_layers
        result = {}
        for layer_index in sorted(set(states).union(self.content_layers)):
            metrics = states[layer_index].finalize() if layer_index in states else {}
            if layer_index in self.content_layers:
                metrics.update(self.content_layers[layer_index].finalize())
            result[int(layer_index)] = metrics
        return result

    def finalize(self) -> Dict[str, float]:
        if self.primary_entropy_observed:
            states = [self.primary_layers[index] for index in sorted(self.primary_layers)]
            correct_state = self.primary_correct
            wrong_state = self.primary_wrong
        else:
            states = [self.primary_layers[index] for index in sorted(self.primary_layers)]
            states.extend(self.fallback_layers[index] for index in sorted(self.fallback_layers))
            correct_state = self.fallback_correct
            wrong_state = self.fallback_wrong
        result = self._finalize_layers(states)
        result.update(
            self._finalize_layers(
                self.content_layers[index] for index in sorted(self.content_layers)
            )
        )
        correct_mean = correct_state.value()
        wrong_mean = wrong_state.value()
        if correct_mean is not None and wrong_mean is not None:
            result["attention_correct_wrong_gap"] = float(correct_mean - wrong_mean)
        return result


class _AffinityRelationLayerAccumulator:
    def __init__(self, *, temperature: float, saturation_threshold: float) -> None:
        self.temperature = float(temperature)
        self.saturation_threshold = float(saturation_threshold)
        self.moments = _RunningMoments()
        self.absolute = _RunningMean()
        self.entropy = _RunningMean()
        self.top1 = _RunningMean()
        self.top5 = _RunningMean()
        self.head_diversity = _RunningMean()
        self.effective_rank = _CappedEffectiveRank()
        self.positive_count = 0
        self.negative_count = 0
        self.saturated_count = 0
        self.element_count = 0
        self.vector_sum: Optional[torch.Tensor] = None
        self.vector_count = 0

    def update(self, values: torch.Tensor) -> None:
        data = values.detach().to(device="cpu", dtype=torch.float32)
        if data.numel() == 0:
            return
        self.moments.update(data)
        self.absolute.update_values(data.abs())
        self.positive_count += int((data > 0).sum().item())
        self.negative_count += int((data < 0).sum().item())
        self.saturated_count += int((data.abs() >= self.saturation_threshold).sum().item())
        self.element_count += int(data.numel())
        entropy_mass = _entropy_and_mass_values(data, self.temperature)
        self.entropy.update_values(entropy_mass["normalized_row_entropy"])
        self.top1.update_values(entropy_mass["top1_mass"])
        self.top5.update_values(entropy_mass["top5_mass"])
        self.effective_rank.update(data)
        diversity = _head_diversity_values(data)
        if diversity is None:
            self.head_diversity.update_scalar(0.0)
        else:
            self.head_diversity.update_values(diversity)
        if data.dim() >= 2:
            vector = data.sum(dim=(0, 1)).reshape(-1)
            if self.vector_sum is None:
                self.vector_sum = vector
            else:
                if self.vector_sum.numel() != vector.numel():
                    raise ValueError("Affinity relation shape changed across probe batches.")
                self.vector_sum += vector
            self.vector_count += int(data.shape[0]) * int(data.shape[1])

    def mean_vector(self) -> Optional[torch.Tensor]:
        if self.vector_sum is None or self.vector_count <= 0:
            return None
        return self.vector_sum / self.vector_count

    def finalize(self) -> Dict[str, float]:
        if self.element_count <= 0:
            return {}
        positive = self.positive_count / self.element_count
        negative = self.negative_count / self.element_count
        result = {
            "raw_mean": float(self.moments.mean),
            "raw_std": float(self.moments.std() or 0.0),
            "raw_abs_mean": float(self.absolute.value() or 0.0),
            "positive_negative_ratio": float(positive / max(negative, 1e-12)),
            "saturation_ratio": float(self.saturated_count / self.element_count),
            "normalized_row_entropy": float(self.entropy.value() or 0.0),
            "top1_mass": float(self.top1.value() or 0.0),
            "top5_mass": float(self.top5.value() or 0.0),
            "affinity_effective_rank": self.effective_rank.value(),
            "head_diversity": float(self.head_diversity.value() or 0.0),
        }
        return result


class _AffinityHealthAccumulator:
    RELATION_KEYS = (
        "QcKp_raw", "QpKc_raw",
        "QcKv_raw", "QvKc_raw",
        "QcKs_raw", "QsKc_raw",
        "QpKv_raw", "QvKp_raw",
        "QpKs_raw", "QsKp_raw",
        "QvKs_raw", "QsKv_raw",
    )

    def __init__(
        self,
        *,
        temperature: float,
        saturation_threshold: float,
        selected_layers: Sequence[int],
    ) -> None:
        self.temperature = float(temperature)
        self.saturation_threshold = float(saturation_threshold)
        self.selected_layers = {int(item) for item in selected_layers}
        self.states: Dict[str, Dict[int, _AffinityRelationLayerAccumulator]] = defaultdict(dict)

    def _selected(self, layer_index: int) -> bool:
        return not self.selected_layers or int(layer_index) in self.selected_layers

    def update(self, affinity_layers: Sequence[Any]) -> None:
        for layer_index, layer in enumerate(affinity_layers or []):
            if not self._selected(layer_index) or not isinstance(layer, Mapping):
                continue
            for key in self.RELATION_KEYS:
                value = layer.get(key)
                if not torch.is_tensor(value) or value.numel() == 0:
                    continue
                relation = key[:-4]
                state = self.states[relation].setdefault(
                    layer_index,
                    _AffinityRelationLayerAccumulator(
                        temperature=self.temperature,
                        saturation_threshold=self.saturation_threshold,
                    ),
                )
                state.update(value)

    def finalize(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for relation, layer_states in self.states.items():
            buckets: Dict[str, List[float]] = defaultdict(list)
            vectors = []
            for layer_index in sorted(layer_states):
                state = layer_states[layer_index]
                for name, value in state.finalize().items():
                    buckets[name].append(float(value))
                vector = state.mean_vector()
                if vector is not None:
                    vectors.append(vector)
            for name, values in buckets.items():
                if values:
                    result[f"{relation}.{name}"] = float(sum(values) / len(values))
            if len(vectors) < 2:
                result[f"{relation}.layer_diversity"] = 0.0
            else:
                similarities = []
                for left, right in zip(vectors[:-1], vectors[1:]):
                    size = min(left.numel(), right.numel())
                    if size <= 0:
                        continue
                    similarities.append(float(
                        torch.nn.functional.cosine_similarity(left[:size], right[:size], dim=0).item()
                    ))
                result[f"{relation}.layer_diversity"] = (
                    float(1.0 - np.mean(similarities)) if similarities else 0.0
                )
        result["softmax_temperature"] = self.temperature
        return result

    def finalize_by_layer(self) -> Dict[int, Dict[str, float]]:
        result: Dict[int, Dict[str, float]] = defaultdict(dict)
        for relation, layer_states in self.states.items():
            for layer_index, state in layer_states.items():
                for name, value in state.finalize().items():
                    result[int(layer_index)][f"{relation}.{name}"] = float(value)
        return {
            layer_index: result[layer_index]
            for layer_index in sorted(result)
        }


class _PromptLayerMechanismAccumulator:
    METRIC_KEYS = (
        "prompt_layer_input_norm",
        "prompt_layer_output_norm",
        "prompt_layer_change_norm",
        "prompt_layer_input_output_cosine",
        "prompt_previous_output_to_current_input_cosine",
        "prompt_previous_output_to_current_input_gap_norm",
        "prompt_context_swap_applied",
        "prompt_context_swap_fixed_point_ratio",
        "prompt_context_swap_before_after_cosine",
        "prompt_context_swap_delta_norm",
    )

    def __init__(self, selected_layers: Sequence[int]) -> None:
        self.selected_layers = {int(item) for item in selected_layers}
        self.means: Dict[int, Dict[str, _RunningMean]] = defaultdict(
            lambda: defaultdict(_RunningMean)
        )
        self.input_variance: Dict[int, _RunningVectorVariance] = defaultdict(
            _RunningVectorVariance
        )
        self.output_variance: Dict[int, _RunningVectorVariance] = defaultdict(
            _RunningVectorVariance
        )

    def _selected(self, layer_index: int) -> bool:
        return not self.selected_layers or int(layer_index) in self.selected_layers

    def update(self, affinity_layers: Sequence[Any]) -> None:
        for layer_index, affinity in enumerate(affinity_layers or []):
            if not self._selected(layer_index) or not isinstance(affinity, Mapping):
                continue
            for metric_name in self.METRIC_KEYS:
                value = affinity.get(metric_name)
                if torch.is_tensor(value) and value.numel():
                    self.means[layer_index][metric_name].update_values(value.detach().float())
            input_vector = affinity.get("_prompt_layer_input_vector")
            output_vector = affinity.get("_prompt_layer_output_vector")
            if torch.is_tensor(input_vector):
                self.input_variance[layer_index].update(input_vector)
            if torch.is_tensor(output_vector):
                self.output_variance[layer_index].update(output_vector)

    def finalize_by_layer(self) -> Dict[int, Dict[str, float]]:
        layer_indices = set(self.means).union(self.input_variance).union(self.output_variance)
        result = {}
        for layer_index in sorted(layer_indices):
            metrics = {
                name: float(state.value())
                for name, state in self.means[layer_index].items()
                if state.value() is not None
            }
            input_variance = self.input_variance[layer_index].value()
            output_variance = self.output_variance[layer_index].value()
            if input_variance is not None:
                metrics["prompt_layer_input_instance_variance"] = float(input_variance)
            if output_variance is not None:
                metrics["prompt_layer_output_instance_variance"] = float(output_variance)
            if metrics:
                result[int(layer_index)] = metrics
        return result


class _PromptPatchBridgeAccumulator:
    METRIC_KEYS = (
        "prompt_patch_similarity_mean",
        "prompt_patch_similarity_within_sample_std",
        "cls_attention_prompt_patch_similarity_correlation",
        "cls_high_attention_prompt_patch_similarity",
        "cls_low_attention_prompt_patch_similarity",
        "cls_high_minus_low_prompt_patch_similarity",
    )

    def __init__(self, selected_layers: Sequence[int]) -> None:
        self.selected_layers = {int(item) for item in selected_layers}
        self.overall: Dict[str, _RunningMean] = defaultdict(_RunningMean)
        self.by_layer: Dict[int, Dict[str, _RunningMean]] = defaultdict(
            lambda: defaultdict(_RunningMean)
        )

    def update(self, affinity_layers: Sequence[Any]) -> None:
        for layer_index, affinity in enumerate(affinity_layers or []):
            if self.selected_layers and layer_index not in self.selected_layers:
                continue
            if not isinstance(affinity, Mapping):
                continue
            for metric_name in self.METRIC_KEYS:
                value = affinity.get(metric_name)
                if not torch.is_tensor(value) or not value.numel():
                    continue
                values = value.detach().float()
                self.overall[metric_name].update_values(values)
                self.by_layer[layer_index][metric_name].update_values(values)

    def finalize(self) -> Dict[str, float]:
        return {
            metric_name: float(state.value())
            for metric_name, state in self.overall.items()
            if state.value() is not None
        }

    def finalize_by_layer(self) -> Dict[int, Dict[str, float]]:
        result = {}
        for layer_index in sorted(self.by_layer):
            metrics = {
                metric_name: float(state.value())
                for metric_name, state in self.by_layer[layer_index].items()
                if state.value() is not None
            }
            if metrics:
                result[int(layer_index)] = metrics
        return result


class _ClassPromptProfileState:
    def __init__(self, class_count: int, prompt_length: int) -> None:
        self.class_count = int(class_count)
        self.prompt_length = int(prompt_length)
        self.profile_sum = torch.zeros(
            self.class_count, self.prompt_length, dtype=torch.float64
        )
        self.top1_counts = torch.zeros(
            self.class_count, self.prompt_length, dtype=torch.long
        )
        self.class_counts = torch.zeros(self.class_count, dtype=torch.long)

    def update(self, values: Any, targets: Any) -> None:
        data = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float32)
        target = torch.as_tensor(targets, dtype=torch.long).view(-1).cpu()
        if data.dim() != 2 or data.shape != (target.numel(), self.prompt_length):
            return
        data = data.clamp_min(0.0)
        row_sum = data.sum(dim=1)
        valid = (
            torch.isfinite(data).all(dim=1)
            & (row_sum > 1e-12)
            & (target >= 0)
            & (target < self.class_count)
        )
        if not bool(valid.any()):
            return
        data = data[valid]
        target = target[valid]
        probabilities = data / data.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.profile_sum.index_add_(0, target, probabilities.to(torch.float64))
        self.class_counts.index_add_(
            0, target, torch.ones_like(target, dtype=torch.long)
        )
        top1 = probabilities.argmax(dim=1)
        self.top1_counts.index_put_(
            (target, top1),
            torch.ones_like(target, dtype=torch.long),
            accumulate=True,
        )

    def observed_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        observed = torch.nonzero(self.class_counts > 0, as_tuple=False).view(-1)
        if not observed.numel():
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, self.prompt_length), dtype=np.float32),
            )
        profiles = self.profile_sum.index_select(0, observed) / self.class_counts.index_select(
            0, observed
        ).to(torch.float64).unsqueeze(1)
        return observed.numpy(), profiles.to(torch.float32).numpy()

    def summary(self) -> Dict[str, float]:
        observed, profiles = self.observed_profiles()
        if not observed.size:
            return {}
        probabilities = profiles / np.maximum(profiles.sum(axis=1, keepdims=True), 1e-12)
        entropy = -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
        )
        effective_prompt_count = np.exp(entropy)
        counts = self.class_counts.index_select(
            0, torch.as_tensor(observed, dtype=torch.long)
        ).to(torch.float64)
        top_counts = self.top1_counts.index_select(
            0, torch.as_tensor(observed, dtype=torch.long)
        ).to(torch.float64)
        consistency = (top_counts.max(dim=1).values / counts.clamp_min(1.0)).numpy()
        class_modes = profiles.argmax(axis=1)
        column_mass = profiles.sum(axis=0)
        active = column_mass > 1e-12
        effective_class_count = np.asarray([], dtype=np.float32)
        overlap = 0.0
        if bool(active.any()):
            columns = profiles[:, active].T
            class_distribution = columns / np.maximum(
                columns.sum(axis=1, keepdims=True), 1e-12
            )
            class_entropy = -np.sum(
                class_distribution
                * np.log(np.maximum(class_distribution, 1e-12)),
                axis=1,
            )
            effective_class_count = np.exp(class_entropy)
            if columns.shape[0] > 1:
                normalized_columns = columns / np.maximum(
                    np.linalg.norm(columns, axis=1, keepdims=True), 1e-12
                )
                similarity = normalized_columns @ normalized_columns.T
                upper = np.triu_indices(columns.shape[0], k=1)
                overlap = float(similarity[upper].mean())
        return {
            "class_effective_prompt_count_mean": float(effective_prompt_count.mean()),
            "class_effective_prompt_ratio_mean": float(
                effective_prompt_count.mean() / max(1, self.prompt_length)
            ),
            "class_top1_prompt_share_mean": float(probabilities.max(axis=1).mean()),
            "class_prompt_assignment_consistency_mean": float(consistency.mean()),
            "class_prompt_coverage_ratio": float(
                np.unique(class_modes).size / max(1, self.prompt_length)
            ),
            "prompt_effective_class_count_mean": float(
                effective_class_count.mean() if effective_class_count.size else 0.0
            ),
            "prompt_effective_class_ratio_mean": float(
                effective_class_count.mean() / max(1, observed.size)
                if effective_class_count.size
                else 0.0
            ),
            "prompt_role_overlap_mean": float(overlap),
            "observed_class_count": float(observed.size),
        }

    def prompt_summaries(self) -> Dict[int, Dict[str, float]]:
        observed, profiles = self.observed_profiles()
        if not observed.size:
            return {}
        result: Dict[int, Dict[str, float]] = {}
        total_profile_mass = float(profiles.sum())
        total_samples = float(self.class_counts.sum().item())
        for prompt_index in range(self.prompt_length):
            class_mass = profiles[:, prompt_index]
            mass_sum = float(class_mass.sum())
            if mass_sum <= 1e-12:
                continue
            distribution = class_mass / mass_sum
            entropy = -float(
                np.sum(
                    distribution * np.log(np.maximum(distribution, 1e-12))
                )
            )
            prompt_top1_count = float(
                self.top1_counts[:, prompt_index].sum().item()
            )
            result[prompt_index] = {
                "profile_mass_share": float(
                    mass_sum / max(total_profile_mass, 1e-12)
                ),
                "effective_class_count": float(np.exp(entropy)),
                "effective_class_ratio": float(
                    np.exp(entropy) / max(1, observed.size)
                ),
                "top1_assignment_rate": float(
                    prompt_top1_count / max(total_samples, 1.0)
                ),
            }
            top_indices = np.argsort(-distribution)[:min(3, observed.size)]
            for rank, class_index in enumerate(top_indices, start=1):
                result[prompt_index][f"top{rank}_class_local_id"] = float(
                    observed[int(class_index)]
                )
                result[prompt_index][f"top{rank}_class_share"] = float(
                    distribution[int(class_index)]
                )
        return result


def _class_relation_alignment(
    class_ids: np.ndarray,
    class_profiles: np.ndarray,
    semantic_reference: Optional[np.ndarray],
    *,
    neighbor_k: int = 5,
) -> Dict[str, float]:
    if (
        semantic_reference is None
        or class_ids.size < 2
        or semantic_reference.ndim != 2
        or int(class_ids.max()) >= semantic_reference.shape[0]
    ):
        return {}
    profiles = _normalize_numpy_rows(class_profiles)
    semantics = _normalize_numpy_rows(semantic_reference[class_ids])
    profile_relation = profiles @ profiles.T
    semantic_relation = semantics @ semantics.T
    upper = np.triu_indices(class_ids.size, k=1)
    k = min(max(1, int(neighbor_k)), class_ids.size - 1)
    profile_neighbors = profile_relation.copy()
    semantic_neighbors = semantic_relation.copy()
    np.fill_diagonal(profile_neighbors, -np.inf)
    np.fill_diagonal(semantic_neighbors, -np.inf)
    profile_topk = np.argsort(-profile_neighbors, axis=1)[:, :k]
    semantic_topk = np.argsort(-semantic_neighbors, axis=1)[:, :k]
    preservation = np.mean([
        len(set(profile_topk[index]).intersection(semantic_topk[index])) / k
        for index in range(class_ids.size)
    ])
    return {
        "graph_spearman": float(
            spearman_correlation(
                profile_relation[upper], semantic_relation[upper]
            )
        ),
        "neighbor_preservation_at_k": float(preservation),
    }


class _PromptClassRoleAccumulator:
    def __init__(
        self,
        *,
        class_count: int,
        prompt_length: int,
        selected_layers: Sequence[int],
        raw_semantic_reference: Optional[Any] = None,
    ) -> None:
        self.class_count = int(class_count)
        self.prompt_length = int(prompt_length)
        self.selected_layers = {int(item) for item in selected_layers}
        self.raw_semantic_reference = self._semantic_numpy(raw_semantic_reference)
        self.projected_semantic_reference: Optional[np.ndarray] = None
        self.consumption_by_layer: Dict[int, _ClassPromptProfileState] = {}
        self.collection_by_layer: Dict[int, _ClassPromptProfileState] = {}

    def _semantic_numpy(self, values: Optional[Any]) -> Optional[np.ndarray]:
        if values is None:
            return None
        matrix = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float32).numpy()
        if matrix.ndim != 2 or matrix.shape[0] != self.class_count:
            return None
        return matrix

    def _state(
        self, states: Dict[int, _ClassPromptProfileState], layer_index: int
    ) -> _ClassPromptProfileState:
        return states.setdefault(
            int(layer_index),
            _ClassPromptProfileState(self.class_count, self.prompt_length),
        )

    def update(
        self,
        attention_layers: Sequence[Any],
        affinity_layers: Sequence[Any],
        *,
        targets: Sequence[int],
        projected_semantic_reference: Optional[Any] = None,
    ) -> None:
        if self.prompt_length <= 0 or self.class_count <= 0:
            return
        projected = self._semantic_numpy(projected_semantic_reference)
        if projected is not None:
            self.projected_semantic_reference = projected
        layer_count = max(len(attention_layers or []), len(affinity_layers or []))
        prompt_slice = slice(1, 1 + self.prompt_length)
        for layer_index in range(layer_count):
            if self.selected_layers and layer_index not in self.selected_layers:
                continue
            affinity = (
                affinity_layers[layer_index]
                if layer_index < len(affinity_layers or [])
                and isinstance(affinity_layers[layer_index], Mapping)
                else {}
            )
            cls_prompt = affinity.get("AcKp_attn")
            if not torch.is_tensor(cls_prompt) and layer_index < len(attention_layers or []):
                attention = attention_layers[layer_index]
                if torch.is_tensor(attention) and attention.dim() == 4:
                    cls_prompt = attention[:, :, :1, prompt_slice]
            if torch.is_tensor(cls_prompt) and cls_prompt.numel():
                consumption = cls_prompt.detach().float().mean(dim=1).squeeze(1)
                self._state(self.consumption_by_layer, layer_index).update(
                    consumption, targets
                )
            collection = affinity.get("prompt_patch_value_contribution_norm")
            if torch.is_tensor(collection) and collection.numel():
                self._state(self.collection_by_layer, layer_index).update(
                    collection, targets
                )

    def _profile_metrics(
        self, state: _ClassPromptProfileState, prefix: str
    ) -> Dict[str, float]:
        result = {
            f"{prefix}_{name}": value
            for name, value in state.summary().items()
        }
        class_ids, profiles = state.observed_profiles()
        references = (
            ("original_attribute", self.raw_semantic_reference),
            ("projected_semantic", self.projected_semantic_reference),
        )
        for reference_name, reference in references:
            alignment = _class_relation_alignment(class_ids, profiles, reference)
            result.update({
                f"{prefix}_{reference_name}_{name}": value
                for name, value in alignment.items()
            })
        return result

    def finalize_by_layer(self) -> Dict[int, Dict[str, float]]:
        result: Dict[int, Dict[str, float]] = {}
        layer_ids = sorted(
            set(self.consumption_by_layer).union(self.collection_by_layer)
        )
        for layer_index in layer_ids:
            metrics: Dict[str, float] = {}
            consumption = self.consumption_by_layer.get(layer_index)
            collection = self.collection_by_layer.get(layer_index)
            if consumption is not None:
                metrics.update(self._profile_metrics(consumption, "cls_consumption"))
            if collection is not None:
                metrics.update(self._profile_metrics(collection, "patch_collection"))
            if consumption is not None and collection is not None:
                consumption_ids, consumption_profiles = consumption.observed_profiles()
                collection_ids, collection_profiles = collection.observed_profiles()
                common = np.intersect1d(consumption_ids, collection_ids)
                if common.size:
                    consumption_index = {
                        int(class_id): index
                        for index, class_id in enumerate(consumption_ids)
                    }
                    collection_index = {
                        int(class_id): index
                        for index, class_id in enumerate(collection_ids)
                    }
                    left = np.stack([
                        collection_profiles[collection_index[int(class_id)]]
                        for class_id in common
                    ])
                    right = np.stack([
                        consumption_profiles[consumption_index[int(class_id)]]
                        for class_id in common
                    ])
                    left_norm = _normalize_numpy_rows(left)
                    right_norm = _normalize_numpy_rows(right)
                    metrics["collection_consumption_profile_cosine_mean"] = float(
                        np.sum(left_norm * right_norm, axis=1).mean()
                    )
                    metrics["collection_consumption_profile_spearman"] = float(
                        spearman_correlation(left.reshape(-1), right.reshape(-1))
                    )
            if metrics:
                result[int(layer_index)] = metrics
        return result

    def finalize_by_layer_and_prompt(
        self,
    ) -> Dict[int, Dict[int, Dict[str, float]]]:
        result: Dict[int, Dict[int, Dict[str, float]]] = {}
        layer_ids = sorted(
            set(self.consumption_by_layer).union(self.collection_by_layer)
        )
        for layer_index in layer_ids:
            prompt_metrics: Dict[int, Dict[str, float]] = defaultdict(dict)
            consumption = self.consumption_by_layer.get(layer_index)
            collection = self.collection_by_layer.get(layer_index)
            if consumption is not None:
                for prompt_index, metrics in consumption.prompt_summaries().items():
                    prompt_metrics[prompt_index].update({
                        f"cls_consumption_{name}": value
                        for name, value in metrics.items()
                    })
            if collection is not None:
                for prompt_index, metrics in collection.prompt_summaries().items():
                    prompt_metrics[prompt_index].update({
                        f"patch_collection_{name}": value
                        for name, value in metrics.items()
                    })
            if prompt_metrics:
                result[int(layer_index)] = {
                    int(prompt_index): dict(metrics)
                    for prompt_index, metrics in sorted(prompt_metrics.items())
                }
        return result

    def export_role_profiles(self) -> Dict[int, Dict[str, Any]]:
        result: Dict[int, Dict[str, Any]] = {}
        layer_ids = sorted(
            set(self.consumption_by_layer).union(self.collection_by_layer)
        )
        for layer_index in layer_ids:
            layer: Dict[str, Any] = {}
            for name, state in (
                ("cls_consumption", self.consumption_by_layer.get(layer_index)),
                ("patch_collection", self.collection_by_layer.get(layer_index)),
            ):
                if state is None:
                    continue
                class_ids, profiles = state.observed_profiles()
                if not class_ids.size:
                    continue
                layer[name] = {
                    "class_local_ids": class_ids.astype(np.int64).tolist(),
                    "prompt_vectors": profiles.T.astype(np.float32).tolist(),
                }
            if layer:
                result[int(layer_index)] = layer
        return result

    def finalize(self) -> Dict[str, float]:
        by_layer = self.finalize_by_layer()
        metric_values: Dict[str, List[float]] = defaultdict(list)
        for metrics in by_layer.values():
            for name, value in metrics.items():
                metric_values[name].append(float(value))
        return {
            name: float(np.mean(values))
            for name, values in sorted(metric_values.items())
            if values
        }


class _AttributeConceptGroundingAccumulator:
    METRIC_KEYS = {
        "prompt_true_attribute_effective_count",
        "prompt_true_attribute_top1_share",
        "prompt_margin_attribute_effective_count",
        "prompt_margin_attribute_top1_share",
        "prompt_true_attribute_role_overlap",
        "prompt_margin_attribute_role_overlap",
        "collection_true_attribute_effective_count",
        "collection_margin_attribute_effective_count",
        "collection_prompt_true_attribute_profile_cosine",
        "collection_prompt_margin_attribute_profile_cosine",
        "prompt_true_local_attribute_share",
        "prompt_true_global_attribute_share",
        "prompt_true_semantic_granularity_index",
        "prompt_margin_local_attribute_share",
        "prompt_margin_global_attribute_share",
        "prompt_margin_semantic_granularity_index",
        "collection_true_local_attribute_share",
        "collection_true_global_attribute_share",
        "collection_true_semantic_granularity_index",
        "collection_margin_local_attribute_share",
        "collection_margin_global_attribute_share",
        "collection_margin_semantic_granularity_index",
        "attribute_concept_patch_effective_count",
        "attribute_concept_patch_effective_ratio",
        "attribute_concept_patch_top1_share",
        "attribute_concept_patch_fallback_ratio",
        "prompt_attention_to_attribute_concept_patch_mass",
        "prompt_attention_to_attribute_concept_patch_lift",
        "prompt_attention_attribute_concept_topk_overlap",
        "concept_intervention_applied",
        "concept_intervention_selected_patch_ratio",
        "concept_intervention_selected_score_mean",
        "concept_intervention_unselected_score_mean",
        "concept_intervention_selection_score_gap",
        "concept_intervention_selected_prompt_attention_mass_before",
        "concept_intervention_selected_prompt_attention_mass_after",
        "concept_intervention_prompt_patch_mass_abs_error",
        "concept_intervention_fallback_ratio",
        "concept_intervention_targeted",
    }
    PROFILE_KEYS = {
        "prompt_true": "_prompt_true_attribute_profile",
        "prompt_margin": "_prompt_margin_attribute_profile",
        "collection_true": "_collection_true_attribute_profile",
        "collection_margin": "_collection_margin_attribute_profile",
    }

    def __init__(
        self,
        *,
        selected_layers: Sequence[int],
        prompt_length: int,
        attribute_count: int,
        prompt_mode: str,
        score_mode: str,
        topk_attributes: int = 3,
    ) -> None:
        self.selected_layers = {int(item) for item in selected_layers}
        self.prompt_length = int(prompt_length)
        self.attribute_count = int(attribute_count)
        self.prompt_mode = str(prompt_mode)
        self.score_mode = str(score_mode).lower()
        self.topk_attributes = max(
            1, min(int(topk_attributes), max(1, self.attribute_count))
        )
        self.layer_means: Dict[int, Dict[str, _RunningMean]] = defaultdict(
            lambda: defaultdict(_RunningMean)
        )
        self.profile_sums: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.profile_counts: Dict[int, Dict[str, int]] = defaultdict(dict)
        self.continuity_means: Dict[
            tuple, Dict[str, _RunningMean]
        ] = defaultdict(lambda: defaultdict(_RunningMean))

    @staticmethod
    def _topk_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> torch.Tensor:
        left_ids = left.topk(k, dim=-1, largest=True).indices
        right_ids = right.topk(k, dim=-1, largest=True).indices
        matches = left_ids.unsqueeze(-1) == right_ids.unsqueeze(-2)
        return matches.any(dim=-1).float().mean(dim=-1)

    @staticmethod
    def _set_best_match(
        left: torch.Tensor,
        right: torch.Tensor,
        k: int,
    ) -> Dict[str, torch.Tensor]:
        left_unit = torch.nn.functional.normalize(left, dim=-1, eps=1e-12)
        right_unit = torch.nn.functional.normalize(right, dim=-1, eps=1e-12)
        similarity = torch.matmul(left_unit, right_unit.transpose(1, 2))
        row_best, row_index = similarity.max(dim=-1)
        col_best, col_index = similarity.max(dim=-2)
        left_ids = left.topk(k, dim=-1, largest=True).indices
        right_ids = right.topk(k, dim=-1, largest=True).indices
        matched_right_ids = right_ids.gather(
            1, row_index.unsqueeze(-1).expand(-1, -1, k)
        )
        matched_left_ids = left_ids.gather(
            1, col_index.unsqueeze(-1).expand(-1, -1, k)
        )
        row_overlap = (
            left_ids.unsqueeze(-1) == matched_right_ids.unsqueeze(-2)
        ).any(dim=-1).float().mean(dim=-1)
        col_overlap = (
            right_ids.unsqueeze(-1) == matched_left_ids.unsqueeze(-2)
        ).any(dim=-1).float().mean(dim=-1)
        return {
            "cosine": 0.5 * (row_best.mean(dim=-1) + col_best.mean(dim=-1)),
            "topk_overlap": 0.5 * (
                row_overlap.mean(dim=-1) + col_overlap.mean(dim=-1)
            ),
        }

    def _update_grouped_mean(
        self,
        means: Dict[str, _RunningMean],
        name: str,
        values: torch.Tensor,
        correct: torch.Tensor,
    ) -> None:
        data = values.detach().to(device="cpu", dtype=torch.float32)
        if data.dim() == 0:
            data = data.reshape(1)
        if data.shape[0] != correct.numel():
            return
        sample_values = data.reshape(data.shape[0], -1).mean(dim=-1)
        means[f"all.{name}"].update_values(sample_values)
        if bool(correct.any().item()):
            means[f"correct.{name}"].update_values(sample_values[correct])
        wrong = ~correct
        if bool(wrong.any().item()):
            means[f"wrong.{name}"].update_values(sample_values[wrong])

    def _update_profile_sum(
        self,
        layer_index: int,
        name: str,
        values: torch.Tensor,
    ) -> None:
        data = values.detach().to(device="cpu", dtype=torch.float32)
        if (
            data.dim() != 3
            or data.shape[1] != self.prompt_length
            or data.shape[2] != self.attribute_count
        ):
            return
        batch_sum = data.sum(dim=0)
        if name not in self.profile_sums[layer_index]:
            self.profile_sums[layer_index][name] = batch_sum
        else:
            self.profile_sums[layer_index][name] += batch_sum
        self.profile_counts[layer_index][name] = (
            self.profile_counts[layer_index].get(name, 0) + int(data.shape[0])
        )

    def update(
        self,
        affinity_layers: Sequence[Any],
        *,
        predictions: Sequence[int],
        targets: Sequence[int],
    ) -> None:
        pred = torch.as_tensor(predictions, dtype=torch.long).reshape(-1).cpu()
        target = torch.as_tensor(targets, dtype=torch.long).reshape(-1).cpu()
        if pred.numel() != target.numel():
            return
        correct = pred == target
        batch_profiles: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_index, layer in enumerate(affinity_layers or []):
            if self.selected_layers and layer_index not in self.selected_layers:
                continue
            if not isinstance(layer, Mapping):
                continue
            for name in self.METRIC_KEYS:
                values = layer.get(name)
                if torch.is_tensor(values) and values.numel():
                    self._update_grouped_mean(
                        self.layer_means[layer_index],
                        name,
                        values,
                        correct,
                    )
            profiles: Dict[str, torch.Tensor] = {}
            for profile_name, key in self.PROFILE_KEYS.items():
                values = layer.get(key)
                if not torch.is_tensor(values) or values.numel() == 0:
                    continue
                data = values.detach().to(device="cpu", dtype=torch.float32)
                if (
                    data.dim() != 3
                    or data.shape[0] != pred.numel()
                    or data.shape[1] != self.prompt_length
                    or data.shape[2] != self.attribute_count
                ):
                    continue
                profiles[profile_name] = data
                self._update_profile_sum(layer_index, profile_name, data)
            if profiles:
                batch_profiles[layer_index] = profiles

        observed_layers = sorted(batch_profiles)
        for left_layer, right_layer in zip(observed_layers[:-1], observed_layers[1:]):
            common_profiles = sorted(
                set(batch_profiles[left_layer]).intersection(
                    batch_profiles[right_layer]
                )
            )
            for profile_name in common_profiles:
                left = batch_profiles[left_layer][profile_name]
                right = batch_profiles[right_layer][profile_name]
                if self.prompt_mode == "persistent_contextualized":
                    cosine = torch.nn.functional.cosine_similarity(
                        left, right, dim=-1, eps=1e-12
                    ).mean(dim=-1)
                    overlap = self._topk_overlap(
                        left, right, self.topk_attributes
                    ).mean(dim=-1)
                    prefix = "same_slot"
                else:
                    matched = self._set_best_match(
                        left, right, self.topk_attributes
                    )
                    cosine = matched["cosine"]
                    overlap = matched["topk_overlap"]
                    prefix = "set_best_match"
                pair_means = self.continuity_means[(left_layer, right_layer)]
                self._update_grouped_mean(
                    pair_means,
                    f"{prefix}_{profile_name}_cosine",
                    cosine,
                    correct,
                )
                self._update_grouped_mean(
                    pair_means,
                    f"{prefix}_{profile_name}_topk_overlap",
                    overlap,
                    correct,
                )

    @staticmethod
    def _mean_dict(means: Mapping[str, _RunningMean]) -> Dict[str, float]:
        result = {}
        for name, state in sorted(means.items()):
            value = state.value()
            if value is not None:
                result[name] = value
        return result

    def finalize_by_layer(self) -> Dict[int, Dict[str, float]]:
        exact = 1.0 if self.score_mode == "dot" else 0.0
        result = {}
        for layer_index, means in sorted(self.layer_means.items()):
            metrics = self._mean_dict(means)
            metrics["attribute_margin_additive_reference_exact"] = exact
            result[int(layer_index)] = metrics
        return result

    def finalize_by_layer_and_prompt(
        self,
    ) -> Dict[int, Dict[int, Dict[str, float]]]:
        result: Dict[int, Dict[int, Dict[str, float]]] = {}
        for layer_index in sorted(self.profile_sums):
            prompt_metrics: Dict[int, Dict[str, float]] = defaultdict(dict)
            for profile_name, profile_sum in sorted(
                self.profile_sums[layer_index].items()
            ):
                count = max(1, self.profile_counts[layer_index].get(profile_name, 0))
                profile = profile_sum / float(count)
                profile = profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                entropy = -(
                    profile * profile.clamp_min(1e-12).log()
                ).sum(dim=-1)
                top_values, top_ids = profile.topk(
                    self.topk_attributes, dim=-1, largest=True
                )
                for prompt_index in range(int(profile.shape[0])):
                    current = prompt_metrics[prompt_index]
                    current[f"{profile_name}_effective_attribute_count"] = float(
                        entropy[prompt_index].exp().item()
                    )
                    current[f"{profile_name}_top1_share"] = float(
                        top_values[prompt_index, 0].item()
                    )
                    for rank in range(self.topk_attributes):
                        current[
                            f"{profile_name}_top{rank + 1}_attribute_id"
                        ] = int(top_ids[prompt_index, rank].item())
                        current[
                            f"{profile_name}_top{rank + 1}_attribute_share"
                        ] = float(top_values[prompt_index, rank].item())
            if prompt_metrics:
                result[int(layer_index)] = {
                    int(prompt_index): dict(metrics)
                    for prompt_index, metrics in sorted(prompt_metrics.items())
                }
        return result

    def finalize_continuity(self) -> Dict[str, Any]:
        return {
            f"layer_{left}_to_layer_{right}": self._mean_dict(means)
            for (left, right), means in sorted(self.continuity_means.items())
        }

    def finalize(self) -> Dict[str, float]:
        by_layer = self.finalize_by_layer()
        buckets: Dict[str, List[float]] = defaultdict(list)
        for metrics in by_layer.values():
            for name, value in metrics.items():
                buckets[name].append(float(value))
        return {
            name: float(np.mean(values))
            for name, values in sorted(buckets.items())
            if values
        }

    def finalize_semantic_granularity(self) -> Dict[str, float]:
        by_layer = self.finalize_by_layer()
        result: Dict[str, float] = {}
        names = sorted({
            name
            for metrics in by_layer.values()
            for name in metrics
            if name.endswith("semantic_granularity_index")
        })
        for name in names:
            points = [
                (int(layer), float(metrics[name]))
                for layer, metrics in sorted(by_layer.items())
                if name in metrics
            ]
            if len(points) < 2:
                continue
            layers = np.asarray([item[0] for item in points], dtype=np.float64)
            values = np.asarray([item[1] for item in points], dtype=np.float64)
            result[f"{name}_depth_spearman"] = float(
                spearman_correlation(layers, values)
            )
            result[f"{name}_late_minus_early"] = float(values[-1] - values[0])
        return result


class _PatchSemanticTransportAccumulator:
    def __init__(
        self,
        *,
        class_count: int,
        prompt_length: int,
        semantic_length: int,
        raw_semantic_reference: Optional[Any],
        cost_name: str,
        temperature: float,
        patch_ratio: float,
        topk: int = 3,
    ) -> None:
        self.class_count = int(class_count)
        self.prompt_length = int(prompt_length)
        self.semantic_length = int(semantic_length)
        self.cost_name = str(cost_name).lower()
        if self.cost_name != "cosine":
            raise ValueError(
                "PATCH_SEMANTIC_TRANSPORT.COST currently supports only 'cosine'"
            )
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError(
                "PATCH_SEMANTIC_TRANSPORT.TEMPERATURE must be positive"
            )
        self.patch_ratio = float(patch_ratio)
        if not 0.0 < self.patch_ratio < 1.0:
            raise ValueError(
                "PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO must be between 0 and 1"
            )
        self.topk = max(1, min(int(topk), max(1, self.class_count)))
        self.raw_semantic_reference = None
        if raw_semantic_reference is not None:
            raw = torch.as_tensor(raw_semantic_reference).detach().to(
                device="cpu", dtype=torch.float32
            )
            if raw.dim() == 2 and raw.shape[0] == self.class_count:
                self.raw_semantic_reference = raw
        self.means: Dict[str, _RunningMean] = defaultdict(_RunningMean)
        self.class_profile_sums: Dict[str, torch.Tensor] = {}
        self.attribute_profile_sums: Dict[str, torch.Tensor] = {}
        self.profile_sample_count = 0
        self.final_layer_index: Optional[int] = None

    @staticmethod
    def _normalize_profile(values: torch.Tensor) -> torch.Tensor:
        data = values.clamp_min(0.0)
        return data / data.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _effective_count(values: torch.Tensor) -> torch.Tensor:
        entropy = -(
            values * values.clamp_min(1e-12).log()
        ).sum(dim=-1)
        observed = values.sum(dim=-1) > 1e-12
        return torch.where(observed, entropy.exp(), torch.zeros_like(entropy))

    @staticmethod
    def _profile_overlap(values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] < 2:
            return values.new_zeros(values.shape[0])
        unit = torch.nn.functional.normalize(values, dim=-1, eps=1e-12)
        similarity = torch.matmul(unit, unit.transpose(1, 2))
        mask = ~torch.eye(
            values.shape[1], dtype=torch.bool, device=values.device
        )
        return similarity[:, mask].reshape(values.shape[0], -1).mean(dim=-1)

    def _update_grouped_mean(
        self,
        name: str,
        values: torch.Tensor,
        correct: torch.Tensor,
    ) -> None:
        data = values.detach().to(device="cpu", dtype=torch.float32)
        if data.dim() == 0:
            data = data.reshape(1)
        if data.shape[0] != correct.numel():
            return
        sample_values = data.reshape(data.shape[0], -1).mean(dim=-1)
        self.means[f"all.{name}"].update_values(sample_values)
        if bool(correct.any().item()):
            self.means[f"correct.{name}"].update_values(sample_values[correct])
        wrong = ~correct
        if bool(wrong.any().item()):
            self.means[f"wrong.{name}"].update_values(sample_values[wrong])

    def _store_profile(self, name: str, values: torch.Tensor) -> None:
        batch_sum = values.detach().to(
            device="cpu", dtype=torch.float64
        ).sum(dim=0)
        if name not in self.class_profile_sums:
            self.class_profile_sums[name] = batch_sum
        else:
            self.class_profile_sums[name] += batch_sum

    def _store_attribute_profile(self, name: str, values: torch.Tensor) -> None:
        batch_sum = values.detach().to(
            device="cpu", dtype=torch.float64
        ).sum(dim=0)
        if name not in self.attribute_profile_sums:
            self.attribute_profile_sums[name] = batch_sum
        else:
            self.attribute_profile_sums[name] += batch_sum

    def update(
        self,
        token_sequence: Any,
        affinity_layers: Sequence[Any],
        *,
        logits: Any,
        targets: Sequence[int],
        projected_semantic_reference: Any,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not torch.is_tensor(token_sequence) or not affinity_layers:
            return None
        final_layer_index = len(affinity_layers) - 1
        final_affinity = affinity_layers[final_layer_index]
        if not isinstance(final_affinity, Mapping):
            return None
        tokens = token_sequence.detach().float()
        scores = torch.as_tensor(logits).detach().to(
            device=tokens.device, dtype=torch.float32
        )
        semantics = torch.as_tensor(projected_semantic_reference).detach().to(
            device=tokens.device, dtype=torch.float32
        )
        target = torch.as_tensor(targets, device=tokens.device, dtype=torch.long).view(-1)
        patch_start = 1 + self.prompt_length
        patch_end = int(tokens.shape[1]) - self.semantic_length
        if (
            tokens.dim() != 3
            or scores.dim() != 2
            or semantics.dim() != 2
            or patch_end <= patch_start
            or scores.shape != (tokens.shape[0], self.class_count)
            or semantics.shape != (self.class_count, tokens.shape[-1])
            or target.numel() != tokens.shape[0]
        ):
            return None
        patch_tokens = tokens[:, patch_start:patch_end, :]
        patch_unit = torch.nn.functional.normalize(
            patch_tokens, dim=-1, eps=1e-12
        )
        semantic_unit = torch.nn.functional.normalize(
            semantics, dim=-1, eps=1e-12
        )
        similarity = torch.matmul(patch_unit, semantic_unit.transpose(0, 1))
        cost = 1.0 - similarity
        class_weights = torch.softmax(scores, dim=-1)
        patch_to_class_plan = torch.softmax(
            similarity / self.temperature
            + class_weights.clamp_min(1e-12).log().unsqueeze(1),
            dim=-1,
        )
        patch_to_semantic_cost = (
            patch_to_class_plan * cost
        ).sum(dim=-1)
        semantic_to_patch_plan = torch.softmax(
            similarity / self.temperature, dim=1
        )
        semantic_to_patch_class_cost = (
            semantic_to_patch_plan * cost
        ).sum(dim=1)
        semantic_to_patch_cost = (
            class_weights * semantic_to_patch_class_cost
        ).sum(dim=-1)
        patch_to_semantic_sample = patch_to_semantic_cost.mean(dim=-1)
        bidirectional_cost = 0.5 * (
            patch_to_semantic_sample + semantic_to_patch_cost
        )

        hard_scores = scores.clone()
        hard_scores.scatter_(
            1,
            target.unsqueeze(1),
            torch.finfo(hard_scores.dtype).min,
        )
        hard_negative = hard_scores.argmax(dim=-1)
        batch_index = torch.arange(tokens.shape[0], device=tokens.device)
        true_patch_cost = cost[batch_index, :, target].mean(dim=-1)
        hard_patch_cost = cost[batch_index, :, hard_negative].mean(dim=-1)
        true_reverse_cost = semantic_to_patch_class_cost[batch_index, target]
        hard_reverse_cost = semantic_to_patch_class_cost[
            batch_index, hard_negative
        ]

        transport_patch_support = 1.0 - 0.5 * patch_to_semantic_cost
        transport_patch_support = transport_patch_support.clamp(min=0.0, max=1.0)
        transport_patch_profile = self._normalize_profile(transport_patch_support)
        patch_count = int(transport_patch_support.shape[1])
        selected_count = min(
            patch_count - 1,
            max(1, int(math.ceil(self.patch_ratio * patch_count))),
        )
        selected_indices = transport_patch_support.topk(
            selected_count, dim=-1, largest=True
        ).indices
        selected_mask = torch.zeros_like(transport_patch_support, dtype=torch.bool)
        selected_mask.scatter_(1, selected_indices, True)
        selected_score = transport_patch_support.gather(
            1, selected_indices
        ).mean(dim=-1)
        unselected_score = (
            transport_patch_support.masked_fill(selected_mask, 0.0).sum(dim=-1)
            / float(patch_count - selected_count)
        )
        correct = scores.argmax(dim=-1).to(device="cpu") == target.to(device="cpu")
        sample_metrics = {
            "patch_to_semantic_expected_cost": patch_to_semantic_sample,
            "semantic_to_patch_expected_cost": semantic_to_patch_cost,
            "bidirectional_expected_cost": bidirectional_cost,
            "true_class_patch_to_semantic_cost": true_patch_cost,
            "hard_negative_patch_to_semantic_cost": hard_patch_cost,
            "true_vs_hard_patch_cost_gap": hard_patch_cost - true_patch_cost,
            "true_class_semantic_to_patch_cost": true_reverse_cost,
            "hard_negative_semantic_to_patch_cost": hard_reverse_cost,
            "true_vs_hard_reverse_cost_gap": hard_reverse_cost - true_reverse_cost,
            "transport_patch_effective_count": self._effective_count(
                transport_patch_profile
            ),
            "transport_patch_effective_ratio": self._effective_count(
                transport_patch_profile
            ) / float(patch_count),
            "transport_patch_top1_share": transport_patch_profile.max(dim=-1).values,
            "transport_selected_score_mean": selected_score,
            "transport_unselected_score_mean": unselected_score,
            "transport_selection_score_gap": selected_score - unselected_score,
        }
        for name in (
            "transport_intervention_applied",
            "transport_intervention_selected_patch_ratio",
            "transport_intervention_selected_score_mean",
            "transport_intervention_unselected_score_mean",
            "transport_intervention_selection_score_gap",
            "transport_intervention_selected_prompt_attention_mass_before",
            "transport_intervention_selected_prompt_attention_mass_after",
            "transport_intervention_prompt_patch_mass_abs_error",
            "transport_intervention_targeted",
        ):
            values = final_affinity.get(name)
            if torch.is_tensor(values) and values.numel():
                sample_metrics[name] = values

        attribute_support = final_affinity.get("_attribute_concept_patch_support")
        if torch.is_tensor(attribute_support):
            attribute_support = attribute_support.detach().to(
                device=tokens.device, dtype=torch.float32
            )
            if attribute_support.shape == transport_patch_support.shape:
                attribute_indices = attribute_support.topk(
                    selected_count, dim=-1, largest=True
                ).indices
                attribute_mask = torch.zeros_like(selected_mask)
                attribute_mask.scatter_(1, attribute_indices, True)
                intersection = (selected_mask & attribute_mask).sum(dim=-1).float()
                union = (selected_mask | attribute_mask).sum(dim=-1).float()
                sample_metrics.update({
                    "transport_attribute_topk_overlap": (
                        intersection / float(selected_count)
                    ),
                    "transport_attribute_topk_jaccard": (
                        intersection / union.clamp_min(1.0)
                    ),
                    "transport_only_patch_ratio": (
                        (selected_mask & ~attribute_mask).sum(dim=-1).float()
                        / float(selected_count)
                    ),
                    "attribute_only_patch_ratio": (
                        (attribute_mask & ~selected_mask).sum(dim=-1).float()
                        / float(selected_count)
                    ),
                })
                rank_correlations = []
                transport_cpu = transport_patch_support.detach().cpu().numpy()
                attribute_cpu = attribute_support.detach().cpu().numpy()
                for left, right in zip(transport_cpu, attribute_cpu):
                    rank_correlations.append(
                        spearman_correlation(left, right)
                    )
                sample_metrics["transport_attribute_score_spearman"] = (
                    transport_patch_support.new_tensor(rank_correlations)
                )

        prompt_attention = final_affinity.get("ApKv_attn")
        prompt_av = final_affinity.get("_prompt_patch_pre_output_av_magnitude")
        if (
            self.prompt_length > 0
            and torch.is_tensor(prompt_attention)
            and torch.is_tensor(prompt_av)
        ):
            attention_weight = prompt_attention.detach().to(
                device=tokens.device, dtype=torch.float32
            ).mean(dim=1)
            av_weight = prompt_av.detach().to(
                device=tokens.device, dtype=torch.float32
            )
            if (
                attention_weight.shape
                == (tokens.shape[0], self.prompt_length, patch_count)
                and av_weight.shape == attention_weight.shape
            ):
                attention_weight = self._normalize_profile(attention_weight)
                av_weight = self._normalize_profile(av_weight)
                attention_class_profile = self._normalize_profile(
                    torch.matmul(attention_weight, patch_to_class_plan)
                )
                av_class_profile = self._normalize_profile(
                    torch.matmul(av_weight, patch_to_class_plan)
                )
                sample_metrics.update({
                    "prompt_attention_transport_effective_class_count": (
                        self._effective_count(attention_class_profile)
                    ),
                    "prompt_attention_transport_top1_class_share": (
                        attention_class_profile.max(dim=-1).values
                    ),
                    "prompt_attention_transport_role_overlap": (
                        self._profile_overlap(attention_class_profile)
                    ),
                    "prompt_av_transport_effective_class_count": (
                        self._effective_count(av_class_profile)
                    ),
                    "prompt_av_transport_top1_class_share": (
                        av_class_profile.max(dim=-1).values
                    ),
                    "prompt_av_transport_role_overlap": (
                        self._profile_overlap(av_class_profile)
                    ),
                    "prompt_attention_av_transport_profile_cosine": (
                        torch.nn.functional.cosine_similarity(
                            attention_class_profile,
                            av_class_profile,
                            dim=-1,
                            eps=1e-12,
                        )
                    ),
                })
                self._store_profile("attention", attention_class_profile)
                self._store_profile("av", av_class_profile)
                if self.raw_semantic_reference is not None:
                    raw = self.raw_semantic_reference.to(
                        device=tokens.device, dtype=torch.float32
                    ).abs()
                    attention_attribute_profile = self._normalize_profile(
                        torch.matmul(attention_class_profile, raw)
                    )
                    av_attribute_profile = self._normalize_profile(
                        torch.matmul(av_class_profile, raw)
                    )
                    sample_metrics.update({
                        "prompt_attention_transport_effective_attribute_count": (
                            self._effective_count(attention_attribute_profile)
                        ),
                        "prompt_attention_transport_top1_attribute_share": (
                            attention_attribute_profile.max(dim=-1).values
                        ),
                        "prompt_av_transport_effective_attribute_count": (
                            self._effective_count(av_attribute_profile)
                        ),
                        "prompt_av_transport_top1_attribute_share": (
                            av_attribute_profile.max(dim=-1).values
                        ),
                        "prompt_attention_av_transport_attribute_cosine": (
                            torch.nn.functional.cosine_similarity(
                                attention_attribute_profile,
                                av_attribute_profile,
                                dim=-1,
                                eps=1e-12,
                            )
                        ),
                    })
                    self._store_attribute_profile(
                        "attention", attention_attribute_profile
                    )
                    self._store_attribute_profile("av", av_attribute_profile)
                self.profile_sample_count += int(tokens.shape[0])

        for name, values in sample_metrics.items():
            self._update_grouped_mean(name, values, correct)
        self.final_layer_index = int(final_layer_index)
        return {
            "selected_patch_indices": selected_indices.detach(),
            "reference_patch_scores": transport_patch_support.detach(),
        }

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        return values / np.maximum(
            np.linalg.norm(values, axis=1, keepdims=True), 1e-12
        )

    def _mean_profile(
        self, profiles: Mapping[str, torch.Tensor], name: str
    ) -> Optional[np.ndarray]:
        values = profiles.get(name)
        if values is None or self.profile_sample_count <= 0:
            return None
        mean = values.to(torch.float32) / float(self.profile_sample_count)
        mean = mean / mean.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return mean.numpy()

    def _role_profile(
        self,
        prompt_class_role: Optional[Any],
        source: str,
    ) -> Optional[np.ndarray]:
        if prompt_class_role is None or self.final_layer_index is None:
            return None
        states = (
            prompt_class_role.collection_by_layer
            if source == "patch_collection"
            else prompt_class_role.consumption_by_layer
        )
        state = states.get(self.final_layer_index)
        if state is None:
            return None
        observed, profiles = state.observed_profiles()
        if not observed.size:
            return None
        result = np.zeros(
            (self.prompt_length, self.class_count), dtype=np.float32
        )
        result[:, observed] = profiles.T
        return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)

    def _attribute_reference_profile(
        self,
        attribute_concept: Optional[Any],
        name: str,
    ) -> Optional[np.ndarray]:
        if attribute_concept is None or self.final_layer_index is None:
            return None
        profile = attribute_concept.profile_sums.get(
            self.final_layer_index, {}
        ).get(name)
        count = attribute_concept.profile_counts.get(
            self.final_layer_index, {}
        ).get(name, 0)
        if profile is None or count <= 0:
            return None
        values = profile.to(torch.float32) / float(count)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return values.numpy()

    def _alignment(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        if left is None or right is None or left.shape != right.shape:
            return None
        left_unit = self._normalize_rows(left)
        right_unit = self._normalize_rows(right)
        per_prompt = np.sum(left_unit * right_unit, axis=1)
        return {
            "cosine_mean": float(per_prompt.mean()),
            "spearman": float(
                spearman_correlation(left.reshape(-1), right.reshape(-1))
            ),
            "per_prompt_cosine": per_prompt,
        }

    def connection_metrics(
        self,
        *,
        prompt_class_role: Optional[Any],
        attribute_concept: Optional[Any],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        attention = self._mean_profile(self.class_profile_sums, "attention")
        av = self._mean_profile(self.class_profile_sums, "av")
        for transport_name, transport_profile in (
            ("attention", attention),
            ("av", av),
        ):
            for role_name in ("patch_collection", "cls_consumption"):
                aligned = self._alignment(
                    transport_profile,
                    self._role_profile(prompt_class_role, role_name),
                )
                if aligned is not None:
                    prefix = f"{transport_name}_transport_to_{role_name}_role"
                    result[f"{prefix}_cosine_mean"] = aligned["cosine_mean"]
                    result[f"{prefix}_spearman"] = aligned["spearman"]
                    result[f"_{prefix}_per_prompt_cosine"] = aligned[
                        "per_prompt_cosine"
                    ]
        attention_attribute = self._mean_profile(
            self.attribute_profile_sums, "attention"
        )
        av_attribute = self._mean_profile(self.attribute_profile_sums, "av")
        for transport_name, transport_profile, reference_name in (
            ("attention", attention_attribute, "prompt_true"),
            ("av", av_attribute, "collection_true"),
        ):
            aligned = self._alignment(
                transport_profile,
                self._attribute_reference_profile(
                    attribute_concept, reference_name
                ),
            )
            if aligned is not None:
                prefix = (
                    f"{transport_name}_transport_to_{reference_name}_attribute"
                )
                result[f"{prefix}_cosine_mean"] = aligned["cosine_mean"]
                result[f"{prefix}_spearman"] = aligned["spearman"]
                result[f"_{prefix}_per_prompt_cosine"] = aligned[
                    "per_prompt_cosine"
                ]
        return result

    def finalize(
        self,
        *,
        prompt_class_role: Optional[Any],
        attribute_concept: Optional[Any],
    ) -> Dict[str, float]:
        result = {
            name: float(state.value())
            for name, state in sorted(self.means.items())
            if state.value() is not None
        }
        for name, value in self.connection_metrics(
            prompt_class_role=prompt_class_role,
            attribute_concept=attribute_concept,
        ).items():
            if not name.startswith("_"):
                result[name] = float(value)
        return result

    def finalize_by_prompt(
        self,
        *,
        prompt_class_role: Optional[Any],
        attribute_concept: Optional[Any],
    ) -> Dict[int, Dict[str, float]]:
        if self.prompt_length <= 0 or self.profile_sample_count <= 0:
            return {}
        result: Dict[int, Dict[str, float]] = defaultdict(dict)
        for profile_name in ("attention", "av"):
            profile = self._mean_profile(self.class_profile_sums, profile_name)
            if profile is None:
                continue
            topk = min(self.topk, profile.shape[1])
            top_ids = np.argsort(-profile, axis=1)[:, :topk]
            for prompt_index in range(self.prompt_length):
                distribution = profile[prompt_index]
                entropy = -float(
                    np.sum(
                        distribution
                        * np.log(np.maximum(distribution, 1e-12))
                    )
                )
                current = result[prompt_index]
                current[f"{profile_name}_effective_class_count"] = float(
                    np.exp(entropy)
                )
                current[f"{profile_name}_top1_class_share"] = float(
                    distribution[top_ids[prompt_index, 0]]
                )
                for rank in range(topk):
                    class_id = int(top_ids[prompt_index, rank])
                    current[
                        f"{profile_name}_top{rank + 1}_class_local_id"
                    ] = float(class_id)
                    current[
                        f"{profile_name}_top{rank + 1}_class_share"
                    ] = float(distribution[class_id])
            attribute_profile = self._mean_profile(
                self.attribute_profile_sums, profile_name
            )
            if attribute_profile is not None:
                attribute_topk = min(self.topk, attribute_profile.shape[1])
                attribute_ids = np.argsort(
                    -attribute_profile, axis=1
                )[:, :attribute_topk]
                for prompt_index in range(self.prompt_length):
                    distribution = attribute_profile[prompt_index]
                    entropy = -float(
                        np.sum(
                            distribution
                            * np.log(np.maximum(distribution, 1e-12))
                        )
                    )
                    current = result[prompt_index]
                    current[
                        f"{profile_name}_effective_attribute_count"
                    ] = float(np.exp(entropy))
                    current[
                        f"{profile_name}_top1_attribute_share"
                    ] = float(distribution[attribute_ids[prompt_index, 0]])
                    for rank in range(attribute_topk):
                        attribute_id = int(attribute_ids[prompt_index, rank])
                        current[
                            f"{profile_name}_top{rank + 1}_attribute_id"
                        ] = float(attribute_id)
                        current[
                            f"{profile_name}_top{rank + 1}_attribute_share"
                        ] = float(distribution[attribute_id])
        connections = self.connection_metrics(
            prompt_class_role=prompt_class_role,
            attribute_concept=attribute_concept,
        )
        for name, values in connections.items():
            if not name.startswith("_") or not name.endswith(
                "_per_prompt_cosine"
            ):
                continue
            metric_name = name[1:-len("_per_prompt_cosine")] + "_cosine"
            for prompt_index, value in enumerate(values):
                result[prompt_index][metric_name] = float(value)
        return {
            int(prompt_index): dict(metrics)
            for prompt_index, metrics in sorted(result.items())
        }


class ProbeAttentionAffinityAccumulator:
    def __init__(
        self,
        *,
        prompt_length: int,
        semantic_length: int,
        selected_layers: Sequence[int],
        class_count: int = 0,
        raw_semantic_reference: Optional[Any] = None,
        attribute_count: int = 0,
        attribute_concept_enable: bool = False,
        patch_semantic_transport_enable: bool = False,
        patch_semantic_transport_cost: str = "cosine",
        patch_semantic_transport_temperature: float = 1.0,
        patch_semantic_transport_patch_ratio: float = 0.2,
        prompt_mode: str = "persistent_contextualized",
        score_mode: str = "dot",
        temperature: float = 1.0,
        saturation_threshold: float = 10.0,
        prompt_content_enable: bool = False,
        instance_prompt_length: int = 0,
        domain_prompt_length: int = 0,
        content_redundancy_cosine: float = 0.9,
        content_opposition_cosine: float = -0.5,
        content_cancellation_ratio: float = 0.25,
        low_usage_fraction: float = 0.25,
        low_function_fraction: float = 0.25,
        low_role_coverage: float = 0.01,
        role_profile_export_enable: bool = False,
    ) -> None:
        self.selected_layers = {int(item) for item in selected_layers}
        self.attention = _AttentionFlowAccumulator(
            prompt_length=prompt_length,
            semantic_length=semantic_length,
            selected_layers=selected_layers,
        )
        self.affinity = _AffinityHealthAccumulator(
            temperature=temperature,
            saturation_threshold=saturation_threshold,
            selected_layers=selected_layers,
        )
        self.prompt_mechanism = _PromptLayerMechanismAccumulator(selected_layers)
        self.prompt_patch_bridge = _PromptPatchBridgeAccumulator(selected_layers)
        self.prompt_content = (
            PromptContentSlotAccumulator(
                prompt_length=prompt_length,
                selected_layers=selected_layers,
                instance_tokens=instance_prompt_length,
                domain_tokens=domain_prompt_length,
                redundancy_cosine=content_redundancy_cosine,
                opposition_cosine=content_opposition_cosine,
                cancellation_ratio=content_cancellation_ratio,
                low_usage_fraction=low_usage_fraction,
                low_function_fraction=low_function_fraction,
                low_role_coverage=low_role_coverage,
            )
            if bool(prompt_content_enable) and int(prompt_length) > 0
            else None
        )
        self.role_profile_export_enable = bool(role_profile_export_enable)
        self.prompt_class_role = (
            _PromptClassRoleAccumulator(
                class_count=class_count,
                prompt_length=prompt_length,
                selected_layers=selected_layers,
                raw_semantic_reference=raw_semantic_reference,
            )
            if int(class_count) > 0 and int(prompt_length) > 0
            else None
        )
        self.attribute_concept = (
            _AttributeConceptGroundingAccumulator(
                selected_layers=selected_layers,
                prompt_length=prompt_length,
                attribute_count=attribute_count,
                prompt_mode=prompt_mode,
                score_mode=score_mode,
            )
            if bool(attribute_concept_enable)
            and int(attribute_count) > 0
            and int(prompt_length) > 0
            else None
        )
        self.patch_semantic_transport = (
            _PatchSemanticTransportAccumulator(
                class_count=class_count,
                prompt_length=prompt_length,
                semantic_length=semantic_length,
                raw_semantic_reference=raw_semantic_reference,
                cost_name=patch_semantic_transport_cost,
                temperature=patch_semantic_transport_temperature,
                patch_ratio=patch_semantic_transport_patch_ratio,
            )
            if bool(patch_semantic_transport_enable) and int(class_count) > 0
            else None
        )
        self.fallback_affinity_keys = {
            "AcKv_attn", "AcKp_attn", "ApKv_attn", "AvKp_attn", "ApKc_attn",
            "AsKv_attn", "AvKs_attn", "AsKp_attn", "ApKs_attn",
        }
        self.content_contribution_keys = set(_AttentionLayerAccumulator.CONTENT_METRICS)
        self.prompt_intervention_keys = set(
            _AttentionLayerAccumulator.INTERVENTION_METRICS
        )
        self.prompt_mechanism_keys = set(_PromptLayerMechanismAccumulator.METRIC_KEYS).union({
            "_prompt_layer_input_vector",
            "_prompt_layer_output_vector",
        })
        self.prompt_patch_bridge_keys = set(_PromptPatchBridgeAccumulator.METRIC_KEYS)
        self.attribute_concept_keys = set(
            _AttributeConceptGroundingAccumulator.METRIC_KEYS
        ).union(_AttributeConceptGroundingAccumulator.PROFILE_KEYS.values())

    def update(
        self,
        attention_layers: Sequence[Any],
        affinity_layers: Sequence[Any],
        *,
        predictions: Sequence[int],
        targets: Sequence[int],
        projected_semantic_reference: Optional[Any] = None,
        transport_semantic_reference: Optional[Any] = None,
        token_sequence: Optional[Any] = None,
        logits: Optional[Any] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        for layer_index, value in enumerate(attention_layers or []):
            if self.selected_layers and layer_index not in self.selected_layers:
                continue
            if not torch.is_tensor(value):
                continue
            cpu_value = value.detach().to(device="cpu", dtype=torch.float32)
            indexed_attention = [None] * layer_index + [cpu_value]
            self.attention.update(
                indexed_attention,
                [],
                predictions=predictions,
                targets=targets,
            )
            del indexed_attention, cpu_value
        for layer_index, layer in enumerate(affinity_layers or []):
            if self.selected_layers and layer_index not in self.selected_layers:
                continue
            if not isinstance(layer, Mapping):
                continue
            required_keys = set(_AffinityHealthAccumulator.RELATION_KEYS)
            required_keys.update(self.content_contribution_keys)
            required_keys.update(self.prompt_intervention_keys)
            required_keys.update(self.prompt_mechanism_keys)
            required_keys.update(self.prompt_patch_bridge_keys)
            if self.attribute_concept is not None:
                required_keys.update(self.attribute_concept_keys)
            if not self.attention.primary_entropy_observed:
                required_keys.update(self.fallback_affinity_keys)
            cpu_layer = {
                key: value.detach().to(device="cpu", dtype=torch.float32)
                for key, value in layer.items()
                if key in required_keys and torch.is_tensor(value)
            }
            indexed_affinity = [{} for _ in range(layer_index)] + [cpu_layer]
            self.attention.update_content_contribution(indexed_affinity)
            self.attention.update_intervention_stats(indexed_affinity)
            self.prompt_mechanism.update(indexed_affinity)
            self.prompt_patch_bridge.update(indexed_affinity)
            if not self.attention.primary_entropy_observed:
                self.attention.update(
                    [],
                    indexed_affinity,
                    predictions=predictions,
                    targets=targets,
                )
            self.affinity.update(indexed_affinity)
            del indexed_affinity, cpu_layer
        if self.prompt_class_role is not None:
            self.prompt_class_role.update(
                attention_layers,
                affinity_layers,
                targets=targets,
                projected_semantic_reference=projected_semantic_reference,
            )
        if self.prompt_content is not None:
            self.prompt_content.update(attention_layers, affinity_layers)
        if self.attribute_concept is not None:
            self.attribute_concept.update(
                affinity_layers,
                predictions=predictions,
                targets=targets,
            )
        if (
            self.patch_semantic_transport is not None
            and token_sequence is not None
            and logits is not None
            and transport_semantic_reference is not None
        ):
            return self.patch_semantic_transport.update(
                token_sequence,
                affinity_layers,
                logits=logits,
                targets=targets,
                projected_semantic_reference=transport_semantic_reference,
            )
        return None

    def finalize(self) -> Dict[str, Any]:
        prompt_semantic_role = (
            self.prompt_class_role.finalize()
            if self.prompt_class_role is not None
            else {}
        )
        prompt_semantic_role_by_layer = (
            self.prompt_class_role.finalize_by_layer()
            if self.prompt_class_role is not None
            else {}
        )
        prompt_semantic_role_by_layer_and_prompt = (
            self.prompt_class_role.finalize_by_layer_and_prompt()
            if self.prompt_class_role is not None
            else {}
        )
        attribute_concept_grounding = (
            self.attribute_concept.finalize()
            if self.attribute_concept is not None
            else {}
        )
        if self.attribute_concept is not None:
            attribute_concept_grounding.update(
                self.attribute_concept.finalize_semantic_granularity()
            )
        prompt_content = (
            self.prompt_content.finalize()
            if self.prompt_content is not None
            else {
                "metrics": {},
                "by_layer": {},
                "by_layer_and_head": {},
                "by_layer_and_prompt": {},
            }
        )
        prompt_role_profiles = (
            self.prompt_class_role.export_role_profiles()
            if self.prompt_class_role is not None
            and self.role_profile_export_enable
            else {}
        )
        attribute_concept_grounding_by_layer = (
            self.attribute_concept.finalize_by_layer()
            if self.attribute_concept is not None
            else {}
        )
        attribute_concept_grounding_by_layer_and_prompt = (
            self.attribute_concept.finalize_by_layer_and_prompt()
            if self.attribute_concept is not None
            else {}
        )
        cross_layer_concept_continuity_by_pair = (
            self.attribute_concept.finalize_continuity()
            if self.attribute_concept is not None
            else {}
        )
        patch_semantic_transport = (
            self.patch_semantic_transport.finalize(
                prompt_class_role=self.prompt_class_role,
                attribute_concept=self.attribute_concept,
            )
            if self.patch_semantic_transport is not None
            else {}
        )
        patch_semantic_transport_by_prompt = (
            self.patch_semantic_transport.finalize_by_prompt(
                prompt_class_role=self.prompt_class_role,
                attribute_concept=self.attribute_concept,
            )
            if self.patch_semantic_transport is not None
            else {}
        )
        return {
            "attention_flow": self.attention.finalize(),
            "attention_flow_by_layer": self.attention.finalize_by_layer(),
            "affinity_health": self.affinity.finalize(),
            "affinity_health_by_layer": self.affinity.finalize_by_layer(),
            "prompt_layer_mechanism_by_layer": self.prompt_mechanism.finalize_by_layer(),
            "prompt_patch_bridge": self.prompt_patch_bridge.finalize(),
            "prompt_patch_bridge_by_layer": self.prompt_patch_bridge.finalize_by_layer(),
            "prompt_content_metrics": prompt_content["metrics"],
            "prompt_content_by_layer": prompt_content["by_layer"],
            "prompt_content_by_layer_and_head": prompt_content[
                "by_layer_and_head"
            ],
            "prompt_content_by_layer_and_prompt": prompt_content[
                "by_layer_and_prompt"
            ],
            "prompt_semantic_role": prompt_semantic_role,
            "prompt_semantic_role_by_layer": prompt_semantic_role_by_layer,
            "prompt_semantic_role_by_layer_and_prompt": (
                prompt_semantic_role_by_layer_and_prompt
            ),
            "prompt_role_profiles": prompt_role_profiles,
            "attribute_concept_grounding": attribute_concept_grounding,
            "attribute_concept_grounding_by_layer": (
                attribute_concept_grounding_by_layer
            ),
            "attribute_concept_grounding_by_layer_and_prompt": (
                attribute_concept_grounding_by_layer_and_prompt
            ),
            "cross_layer_concept_continuity_by_pair": (
                cross_layer_concept_continuity_by_pair
            ),
            "patch_semantic_transport": patch_semantic_transport,
            "patch_semantic_transport_by_prompt": (
                patch_semantic_transport_by_prompt
            ),
        }


def affinity_health_metrics(
    affinities: Iterable[Mapping[str, Any]],
    *,
    temperature: float = 1.0,
    saturation_threshold: float = 10.0,
) -> Dict[str, float]:
    relation_keys = (
        "QcKp_raw", "QpKc_raw",
        "QcKv_raw", "QvKc_raw",
        "QcKs_raw", "QsKc_raw",
        "QpKv_raw", "QvKp_raw",
        "QpKs_raw", "QsKp_raw",
        "QvKs_raw", "QsKv_raw",
    )
    buckets: Dict[str, List[float]] = defaultdict(list)
    layer_vectors: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for layer in affinities or []:
        if not isinstance(layer, Mapping):
            continue
        for key in relation_keys:
            value = layer.get(key)
            if not torch.is_tensor(value) or value.numel() == 0:
                continue
            data = value.detach().float()
            relation = key[:-4]
            buckets[f"{relation}.raw_mean"].append(float(data.mean().item()))
            buckets[f"{relation}.raw_std"].append(float(data.std(unbiased=False).item()))
            buckets[f"{relation}.raw_abs_mean"].append(float(data.abs().mean().item()))
            positive = float((data > 0).float().mean().item())
            negative = float((data < 0).float().mean().item())
            buckets[f"{relation}.positive_negative_ratio"].append(positive / max(negative, 1e-12))
            buckets[f"{relation}.saturation_ratio"].append(
                float((data.abs() >= float(saturation_threshold)).float().mean().item())
            )
            for name, metric in _entropy_and_mass(data, temperature).items():
                buckets[f"{relation}.{name}"].append(metric)
            buckets[f"{relation}.affinity_effective_rank"].append(_matrix_effective_rank(data))
            buckets[f"{relation}.head_diversity"].append(_head_diversity(data))
            layer_vectors[relation].append(data.mean(dim=(0, 1)).reshape(-1).cpu())
    result = {
        name: float(sum(values) / len(values))
        for name, values in buckets.items()
        if values
    }
    for relation, vectors in layer_vectors.items():
        if len(vectors) < 2:
            result[f"{relation}.layer_diversity"] = 0.0
            continue
        similarities = []
        for left, right in zip(vectors[:-1], vectors[1:]):
            size = min(left.numel(), right.numel())
            if size == 0:
                continue
            similarities.append(float(torch.nn.functional.cosine_similarity(left[:size], right[:size], dim=0).item()))
        result[f"{relation}.layer_diversity"] = float(1.0 - np.mean(similarities)) if similarities else 0.0
    result["softmax_temperature"] = float(temperature)
    return result


def attention_flow_metrics(
    attention_layers: Iterable[Any],
    *,
    prompt_length: int,
    semantic_length: int,
    affinity_layers: Optional[Iterable[Any]] = None,
    predictions: Optional[Sequence[int]] = None,
    targets: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    sample_cls_entropy: List[np.ndarray] = []
    for attention in attention_layers or []:
        if not torch.is_tensor(attention) or attention.dim() != 4 or attention.numel() == 0:
            continue
        data = attention.detach().float()
        sequence_length = int(data.shape[-1])
        prompt_slice = slice(1, 1 + int(prompt_length))
        patch_start = 1 + int(prompt_length)
        patch_end = sequence_length - int(semantic_length)
        patch_slice = slice(patch_start, patch_end)
        cls_patch = data[:, :, 0, patch_slice]
        patch_distance = _normalized_patch_attention_distance(data, patch_slice)
        if patch_distance is not None:
            buckets["patch_to_patch_attention_distance"].append(
                float(patch_distance.mean().item())
            )
        if cls_patch.numel():
            buckets["cls_to_patch_mass"].append(float(cls_patch.sum(dim=-1).mean().item()))
            normalized = cls_patch / cls_patch.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy / max(math.log(max(2, normalized.shape[-1])), 1e-12)
            sorted_mass = normalized.sort(dim=-1, descending=True).values
            buckets["cls_to_patch_entropy"].append(float(entropy.mean().item()))
            buckets["cls_to_patch_topk_mass"].append(
                float(sorted_mass[..., : min(5, sorted_mass.shape[-1])].sum(dim=-1).mean().item())
            )
            sample_cls_entropy.append(entropy.mean(dim=1).cpu().numpy())
        if int(prompt_length) > 0:
            cls_to_prompt = data[:, :, 0, prompt_slice]
            prompt_to_patch = data[:, :, prompt_slice, patch_slice]
            patch_to_prompt = data[:, :, patch_slice, prompt_slice]
            prompt_to_cls = data[:, :, prompt_slice, 0]
            buckets["cls_to_prompt_mass"].append(
                float(cls_to_prompt.sum(dim=-1).mean().item())
            )
            buckets["prompt_to_patch_mass"].append(float(prompt_to_patch.sum(dim=-1).mean().item()))
            buckets["patch_to_prompt_mass"].append(float(patch_to_prompt.sum(dim=-1).mean().item()))
            buckets["prompt_to_cls_mass"].append(float(prompt_to_cls.mean().item()))
            prompt_patch_selection = _conditional_attention_values(
                prompt_to_patch,
                topk=5,
            )
            buckets["prompt_to_patch_conditional_entropy"].append(
                float(prompt_patch_selection["conditional_entropy"].mean().item())
            )
            buckets["prompt_to_patch_conditional_top5_mass"].append(
                float(prompt_patch_selection["conditional_topk_mass"].mean().item())
            )
            buckets["prompt_to_patch_effective_patch_count"].append(
                float(prompt_patch_selection["effective_count"].mean().item())
            )
            buckets["prompt_to_patch_effective_patch_ratio"].append(
                float(prompt_patch_selection["effective_ratio"].mean().item())
            )
            prompt_patch_retrieval = _prompt_patch_retrieval_values(
                prompt_to_patch
            )
            for metric_name, metric_values in prompt_patch_retrieval.items():
                buckets[f"prompt_patch_retrieval_{metric_name}"].append(
                    float(metric_values.mean().item())
                )
            patch_prompt_selection = _conditional_attention_values(
                patch_to_prompt,
                topk=1,
            )
            buckets["patch_to_prompt_conditional_entropy"].append(
                float(patch_prompt_selection["conditional_entropy"].mean().item())
            )
            buckets["patch_to_prompt_top1_share"].append(
                float(patch_prompt_selection["conditional_topk_mass"].mean().item())
            )
            buckets["patch_to_prompt_effective_prompt_count"].append(
                float(patch_prompt_selection["effective_count"].mean().item())
            )
            buckets["patch_to_prompt_effective_prompt_ratio"].append(
                float(patch_prompt_selection["effective_ratio"].mean().item())
            )
            patch_prompt_coordination = _patch_prompt_coordination_values(
                patch_to_prompt
            )
            for metric_name, metric_values in patch_prompt_coordination.items():
                buckets[f"patch_prompt_{metric_name}"].append(
                    float(metric_values.mean().item())
                )
            prompt_usage = _conditional_attention_values(cls_to_prompt, topk=1)
            buckets["configured_prompt_count"].append(float(prompt_length))
            buckets["cls_to_prompt_conditional_entropy"].append(
                float(prompt_usage["conditional_entropy"].mean().item())
            )
            buckets["cls_to_prompt_top1_share"].append(
                float(prompt_usage["conditional_topk_mass"].mean().item())
            )
            buckets["cls_to_prompt_effective_prompt_count"].append(
                float(prompt_usage["effective_count"].mean().item())
            )
            buckets["cls_to_prompt_effective_prompt_ratio"].append(
                float(prompt_usage["effective_ratio"].mean().item())
            )
        buckets["head_diversity"].append(_head_diversity(data))
        buckets["attention_effective_rank"].append(_matrix_effective_rank(data))
    if not sample_cls_entropy:
        for affinity in affinity_layers or []:
            if not isinstance(affinity, dict):
                continue
            cls_patch = affinity.get("AcKv_attn")
            if torch.is_tensor(cls_patch) and cls_patch.numel():
                data = cls_patch.detach().float()
                normalized = data / data.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=-1)
                entropy = entropy / max(math.log(max(2, normalized.shape[-1])), 1e-12)
                sorted_mass = normalized.sort(dim=-1, descending=True).values
                buckets["cls_to_patch_entropy"].append(float(entropy.mean().item()))
                buckets["cls_to_patch_topk_mass"].append(
                    float(sorted_mass[..., : min(5, sorted_mass.shape[-1])].sum(dim=-1).mean().item())
                )
                buckets["cls_to_patch_mass"].append(float(data.sum(dim=-1).mean().item()))
                buckets["head_diversity"].append(_head_diversity(data))
                buckets["attention_effective_rank"].append(_matrix_effective_rank(data))
                sample_cls_entropy.append(entropy.mean(dim=1).squeeze(-1).cpu().numpy())
            relation_metrics = {
                "AcKp_attn": "cls_to_prompt_mass",
                "ApKv_attn": "prompt_to_patch_mass",
                "AvKp_attn": "patch_to_prompt_mass",
                "ApKc_attn": "prompt_to_cls_mass",
                "AsKv_attn": "semantic_to_patch_mass",
                "AvKs_attn": "patch_to_semantic_mass",
                "AsKp_attn": "semantic_to_prompt_mass",
                "ApKs_attn": "prompt_to_semantic_mass",
            }
            for key, metric_name in relation_metrics.items():
                value = affinity.get(key)
                if torch.is_tensor(value) and value.numel():
                    data = value.detach().float()
                    buckets[metric_name].append(float(data.sum(dim=-1).mean().item()))
                    if key == "ApKv_attn":
                        selection = _conditional_attention_values(data, topk=5)
                        buckets["prompt_to_patch_conditional_entropy"].append(
                            float(selection["conditional_entropy"].mean().item())
                        )
                        buckets["prompt_to_patch_conditional_top5_mass"].append(
                            float(selection["conditional_topk_mass"].mean().item())
                        )
                        buckets["prompt_to_patch_effective_patch_count"].append(
                            float(selection["effective_count"].mean().item())
                        )
                        buckets["prompt_to_patch_effective_patch_ratio"].append(
                            float(selection["effective_ratio"].mean().item())
                        )
                        retrieval = _prompt_patch_retrieval_values(data)
                        for retrieval_name, metric_values in retrieval.items():
                            buckets[
                                f"prompt_patch_retrieval_{retrieval_name}"
                            ].append(float(metric_values.mean().item()))
                    elif key == "AvKp_attn":
                        selection = _conditional_attention_values(data, topk=1)
                        buckets["patch_to_prompt_conditional_entropy"].append(
                            float(selection["conditional_entropy"].mean().item())
                        )
                        buckets["patch_to_prompt_top1_share"].append(
                            float(selection["conditional_topk_mass"].mean().item())
                        )
                        buckets["patch_to_prompt_effective_prompt_count"].append(
                            float(selection["effective_count"].mean().item())
                        )
                        buckets["patch_to_prompt_effective_prompt_ratio"].append(
                            float(selection["effective_ratio"].mean().item())
                        )
                        coordination = _patch_prompt_coordination_values(data)
                        for metric_name, metric_values in coordination.items():
                            buckets[f"patch_prompt_{metric_name}"].append(
                                float(metric_values.mean().item())
                            )
                    elif key == "AcKp_attn":
                        usage = _conditional_attention_values(data, topk=1)
                        buckets["configured_prompt_count"].append(
                            float(data.shape[-1])
                        )
                        buckets["cls_to_prompt_conditional_entropy"].append(
                            float(usage["conditional_entropy"].mean().item())
                        )
                        buckets["cls_to_prompt_top1_share"].append(
                            float(usage["conditional_topk_mass"].mean().item())
                        )
                        buckets["cls_to_prompt_effective_prompt_count"].append(
                            float(usage["effective_count"].mean().item())
                        )
                        buckets["cls_to_prompt_effective_prompt_ratio"].append(
                            float(usage["effective_ratio"].mean().item())
                        )
    result = {
        name: float(sum(values) / len(values))
        for name, values in buckets.items()
        if values
    }
    if sample_cls_entropy and predictions is not None and targets is not None:
        entropy = np.mean(np.stack(sample_cls_entropy, axis=0), axis=0)
        pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
        target = np.asarray(targets, dtype=np.int64).reshape(-1)
        if entropy.shape[0] == pred.shape[0] == target.shape[0]:
            correct = pred == target
            if correct.any() and (~correct).any():
                correct_mean = float(entropy[correct].mean())
                wrong_mean = float(entropy[~correct].mean())
                result["attention_correct_wrong_gap"] = correct_mean - wrong_mean
    return result


def _distribution_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(data.mean()),
        f"{prefix}_std": float(data.std()),
        f"{prefix}_q25": float(np.quantile(data, 0.25)),
        f"{prefix}_median": float(np.quantile(data, 0.5)),
        f"{prefix}_q75": float(np.quantile(data, 0.75)),
    }


def _normalize_numpy_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class StreamingRepresentationAccumulator:
    def __init__(self, class_count: int, *, track_covariance: bool = True) -> None:
        self.class_count = int(class_count)
        self.track_covariance = bool(track_covariance)
        self.count = 0
        self.feature_dim = 0
        self.feature_sum: Optional[torch.Tensor] = None
        self.feature_cross: Optional[torch.Tensor] = None
        self.normalized_sum: Optional[torch.Tensor] = None
        self.normalized_cross: Optional[torch.Tensor] = None
        self.norm_sum = 0.0
        self.norm_square_sum = 0.0
        self.class_support = torch.zeros(self.class_count, dtype=torch.long)
        self.class_sum: Optional[torch.Tensor] = None
        self.class_square_norm_sum = torch.zeros(self.class_count, dtype=torch.float32)

    def update(self, features: Any, labels: Any) -> None:
        values = torch.as_tensor(features).detach().to(device="cpu", dtype=torch.float32)
        target = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
        if values.dim() != 2 or values.shape[0] != target.numel():
            raise ValueError("streaming representation batch has incompatible shapes")
        finite = torch.isfinite(values).all(dim=1)
        values = values[finite]
        target = target[finite]
        if values.numel() == 0:
            return
        if self.feature_sum is None:
            self.feature_dim = int(values.shape[1])
            self.feature_sum = torch.zeros(self.feature_dim, dtype=torch.float32)
            self.class_sum = torch.zeros(self.class_count, self.feature_dim, dtype=torch.float32)
            if self.track_covariance:
                self.feature_cross = torch.zeros(self.feature_dim, self.feature_dim, dtype=torch.float32)
                self.normalized_sum = torch.zeros(self.feature_dim, dtype=torch.float32)
                self.normalized_cross = torch.zeros(self.feature_dim, self.feature_dim, dtype=torch.float32)
        if int(values.shape[1]) != self.feature_dim:
            raise ValueError("representation feature dimension changed across probe batches")
        norms = values.norm(dim=1)
        self.feature_sum += values.sum(dim=0)
        self.norm_sum += float(norms.sum().item())
        self.norm_square_sum += float(norms.square().sum().item())
        if self.track_covariance:
            normalized = values / norms.clamp_min(1e-12).unsqueeze(1)
            self.feature_cross += values.t().matmul(values)
            self.normalized_sum += normalized.sum(dim=0)
            self.normalized_cross += normalized.t().matmul(normalized)
        self.class_support.index_add_(0, target, torch.ones_like(target))
        self.class_sum.index_add_(0, target, values)
        self.class_square_norm_sum.index_add_(0, target, values.square().sum(dim=1))
        self.count += int(values.shape[0])

    def class_centers(self) -> tuple[np.ndarray, np.ndarray]:
        if self.class_sum is None:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        observed = torch.nonzero(self.class_support > 0, as_tuple=False).view(-1)
        centers = self.class_sum.index_select(0, observed) / self.class_support.index_select(0, observed).to(torch.float32).unsqueeze(1)
        return observed.numpy(), centers.numpy()

    def release_covariance(self) -> None:
        self.feature_cross = None
        self.normalized_sum = None
        self.normalized_cross = None

    def finalize(self) -> Dict[str, float]:
        if self.count <= 0 or self.feature_sum is None:
            return {}
        count = float(self.count)
        norm_mean = self.norm_sum / count
        norm_var = max(0.0, self.norm_square_sum / count - norm_mean * norm_mean)
        result = {
            "feature_norm_mean": float(norm_mean),
            "feature_norm_std": float(math.sqrt(norm_var)),
        }
        if self.track_covariance and self.feature_cross is not None:
            pair_count = self.count * (self.count - 1)
            if pair_count > 0:
                pair_sum = float(self.normalized_sum.dot(self.normalized_sum).item() - self.count)
                pair_square_sum = float(self.normalized_cross.square().sum().item() - self.count)
                pair_mean = pair_sum / pair_count
                pair_var = max(0.0, pair_square_sum / pair_count - pair_mean * pair_mean)
                result["pairwise_cosine_mean"] = float(pair_mean)
                result["pairwise_cosine_std"] = float(math.sqrt(pair_var))
            else:
                result["pairwise_cosine_mean"] = 0.0
                result["pairwise_cosine_std"] = 0.0
            centered_cross = self.feature_cross - torch.ger(self.feature_sum, self.feature_sum) / count
            symmetric = (0.5 * (centered_cross + centered_cross.t())).numpy()
            eigenvalues = np.maximum(np.linalg.eigvalsh(symmetric), 0.0)
            singular = np.sqrt(eigenvalues)[::-1].copy()
            total = float(singular.sum())
            if total > 1e-12:
                probability = singular / total
                entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12))))
                result["effective_rank"] = float(np.exp(entropy))
                result["top_singular_value_ratio"] = float(singular[0] / total)
            else:
                result["effective_rank"] = 0.0
                result["top_singular_value_ratio"] = 0.0
        observed, centers = self.class_centers()
        if observed.size:
            within_by_class = []
            for class_id in observed:
                support = int(self.class_support[int(class_id)].item())
                class_sum = self.class_sum[int(class_id)]
                class_sse = float(
                    self.class_square_norm_sum[int(class_id)].item()
                    - class_sum.square().sum().item() / max(1, support)
                )
                within_by_class.append(max(0.0, class_sse) / max(1, support))
            macro_center = centers.mean(axis=0)
            within = float(np.mean(within_by_class))
            between = float(np.square(centers - macro_center[None, :]).sum(axis=1).mean())
            result.update({
                "within_class_scatter_trace": within,
                "between_class_scatter_trace": between,
                "fisher_trace_ratio": float(between / max(within, 1e-12)),
            })
        return result


class StreamingFixedProbeAccumulator:
    def __init__(self, candidate_class_ids: Sequence[int], *, track_geometry: bool = True, recall_k: int = 5) -> None:
        self.candidate = np.asarray(candidate_class_ids, dtype=np.int64).reshape(-1)
        self.class_count = int(self.candidate.size)
        self.recall_k = min(max(1, int(recall_k)), max(1, self.class_count))
        self.representation = StreamingRepresentationAccumulator(
            self.class_count, track_covariance=track_geometry
        )
        self.semantic: Optional[np.ndarray] = None
        self.sample_count = 0
        self.correct_count = 0
        self.top5_count = 0
        self.nll_sum = 0.0
        self.class_support = np.zeros(self.class_count, dtype=np.int64)
        self.class_correct = np.zeros(self.class_count, dtype=np.int64)
        self.confusion = np.zeros((self.class_count, self.class_count), dtype=np.int64)
        self.true_similarity_sum = 0.0
        self.hard_negative_sum = 0.0
        self.hard_negative_square_sum = 0.0
        self.semantic_margin_sum = 0.0
        self.rank_sum = 0.0
        self.recall_count = 0
        self.ambiguity_count = 0

    def update(self, logits: Any, targets: Any, visual_features: Any, semantic_prototypes: Any) -> None:
        score = torch.as_tensor(logits).detach().to(device="cpu", dtype=torch.float32)
        target = torch.as_tensor(targets, dtype=torch.long).view(-1).cpu()
        visual = torch.as_tensor(visual_features).detach().to(device="cpu", dtype=torch.float32)
        semantic = torch.as_tensor(semantic_prototypes).detach().to(device="cpu", dtype=torch.float32)
        if score.dim() != 2 or score.shape[0] != target.numel() or score.shape[1] != self.class_count:
            raise ValueError("streaming fixed-probe logits have incompatible shapes")
        if visual.dim() != 2 or visual.shape[0] != target.numel() or semantic.shape != (self.class_count, visual.shape[1]):
            raise ValueError("streaming fixed-probe representations have incompatible shapes")
        if self.semantic is None:
            self.semantic = semantic.numpy().copy()
        elif not np.allclose(self.semantic, semantic.numpy(), rtol=0.0, atol=1e-6):
            raise ValueError("semantic prototypes changed across fixed-probe batches")
        prediction = score.argmax(dim=1)
        correct = prediction.eq(target)
        k = min(5, self.class_count)
        topk = score.topk(k=k, dim=1).indices
        probability = torch.softmax(score, dim=1)
        self.correct_count += int(correct.sum().item())
        self.top5_count += int(topk.eq(target.unsqueeze(1)).any(dim=1).sum().item())
        self.nll_sum += float(-probability[torch.arange(target.numel()), target].clamp_min(1e-12).log().sum().item())
        target_np = target.numpy()
        prediction_np = prediction.numpy()
        np.add.at(self.class_support, target_np, 1)
        np.add.at(self.class_correct, target_np, correct.numpy().astype(np.int64))
        np.add.at(self.confusion, (target_np, prediction_np), 1)

        visual_norm = visual / visual.norm(dim=1, keepdim=True).clamp_min(1e-12)
        semantic_norm = semantic / semantic.norm(dim=1, keepdim=True).clamp_min(1e-12)
        similarity = visual_norm.matmul(semantic_norm.t())
        true_similarity = similarity[torch.arange(target.numel()), target]
        wrong = similarity.clone()
        wrong[torch.arange(target.numel()), target] = float("-inf")
        hard_negative = wrong.max(dim=1).values
        margin = true_similarity - hard_negative
        rank = 1 + similarity.gt(true_similarity.unsqueeze(1)).sum(dim=1)
        self.true_similarity_sum += float(true_similarity.sum().item())
        self.hard_negative_sum += float(hard_negative.sum().item())
        self.hard_negative_square_sum += float(hard_negative.square().sum().item())
        self.semantic_margin_sum += float(margin.sum().item())
        self.rank_sum += float(rank.to(torch.float32).sum().item())
        self.recall_count += int(rank.le(self.recall_k).sum().item())
        self.ambiguity_count += int(margin.lt(0.05).sum().item())
        self.representation.update(visual, target)
        self.sample_count += int(target.numel())

    def merge_from(self, other: "StreamingFixedProbeAccumulator") -> None:
        if not np.array_equal(self.candidate, other.candidate):
            raise ValueError("cannot merge fixed-probe accumulators with different candidate orders")
        if other.sample_count <= 0:
            return
        if self.semantic is None:
            self.semantic = other.semantic.copy() if other.semantic is not None else None
        elif other.semantic is not None and not np.allclose(self.semantic, other.semantic, rtol=0.0, atol=1e-6):
            raise ValueError("cannot merge fixed-probe accumulators with different prototypes")
        for name in (
            "sample_count", "correct_count", "top5_count", "recall_count", "ambiguity_count"
        ):
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))
        for name in (
            "nll_sum", "true_similarity_sum", "hard_negative_sum",
            "hard_negative_square_sum", "semantic_margin_sum", "rank_sum",
        ):
            setattr(self, name, float(getattr(self, name)) + float(getattr(other, name)))
        self.class_support += other.class_support
        self.class_correct += other.class_correct
        self.confusion += other.confusion
        left = self.representation
        right = other.representation
        if right.count <= 0:
            return
        if left.feature_sum is None:
            left.count = right.count
            left.feature_dim = right.feature_dim
            left.feature_sum = right.feature_sum.clone()
            left.feature_cross = right.feature_cross.clone() if right.feature_cross is not None else None
            left.normalized_sum = right.normalized_sum.clone() if right.normalized_sum is not None else None
            left.normalized_cross = right.normalized_cross.clone() if right.normalized_cross is not None else None
            left.norm_sum = right.norm_sum
            left.norm_square_sum = right.norm_square_sum
            left.class_support = right.class_support.clone()
            left.class_sum = right.class_sum.clone() if right.class_sum is not None else None
            left.class_square_norm_sum = right.class_square_norm_sum.clone()
            return
        left.count += right.count
        left.feature_sum += right.feature_sum
        if left.feature_cross is not None and right.feature_cross is not None:
            left.feature_cross += right.feature_cross
            left.normalized_sum += right.normalized_sum
            left.normalized_cross += right.normalized_cross
        left.norm_sum += right.norm_sum
        left.norm_square_sum += right.norm_square_sum
        left.class_support += right.class_support
        left.class_sum += right.class_sum
        left.class_square_norm_sum += right.class_square_norm_sum

    def _relation_metrics(self) -> tuple[Dict[str, float], Dict[str, float]]:
        if self.semantic is None:
            return {}, {}
        observed, centers = self.representation.class_centers()
        if observed.size == 0:
            return {}, {}
        semantic = self.semantic[observed]
        normalized_centers = _normalize_numpy_rows(centers)
        normalized_semantic = _normalize_numpy_rows(semantic)
        center_cosine = np.sum(normalized_centers * normalized_semantic, axis=1)
        if observed.size > 1:
            visual_relation = normalized_centers @ normalized_centers.T
            semantic_relation = normalized_semantic @ normalized_semantic.T
            upper = np.triu_indices(observed.size, k=1)
            visual_edges = visual_relation[upper]
            semantic_edges = semantic_relation[upper]
            structure = spearman_correlation(visual_edges, semantic_edges)
            neighbor_k = min(self.recall_k, observed.size - 1)
            overlap = []
            ranking = []
            for row in range(observed.size):
                mask = np.arange(observed.size) != row
                ranking.append(spearman_correlation(visual_relation[row, mask], semantic_relation[row, mask]))
                visual_order = [int(item) for item in np.argsort(-visual_relation[row], kind="mergesort") if int(item) != row][:neighbor_k]
                semantic_order = [int(item) for item in np.argsort(-semantic_relation[row], kind="mergesort") if int(item) != row][:neighbor_k]
                overlap.append(len(set(visual_order).intersection(semantic_order)) / max(1, neighbor_k))
            semantic_without_diagonal = semantic_relation.copy()
            np.fill_diagonal(semantic_without_diagonal, -np.inf)
            semantic_hard_ids = semantic_without_diagonal.argmax(axis=1)
            semantic_hard_values = semantic_without_diagonal[
                np.arange(observed.size), semantic_hard_ids
            ]
            visual_at_semantic_hard = visual_relation[
                np.arange(observed.size), semantic_hard_ids
            ]
            neighbor_preservation = float(np.mean(overlap))
            neighbor_ranking = float(np.mean(ranking))
            visual_sq = np.square(centers).sum(axis=1)
            semantic_sq = np.square(semantic).sum(axis=1)
            visual_distance = np.sqrt(np.maximum(visual_sq[:, None] + visual_sq[None, :] - 2.0 * centers @ centers.T, 0.0))[upper]
            semantic_distance = np.sqrt(np.maximum(semantic_sq[:, None] + semantic_sq[None, :] - 2.0 * semantic @ semantic.T, 0.0))[upper]
            distance_spearman = spearman_correlation(visual_distance, semantic_distance)
            high_mask = semantic_edges >= 0.8
            false_high = high_mask & (visual_edges <= 0.2)
            graph = {
                "semantic_visual_graph_spearman": structure,
                "false_high_semantic_edge_rate": float(false_high.sum() / max(1, high_mask.sum())),
                "neighbor_ranking_consistency": neighbor_ranking,
                **_distribution_summary(visual_edges, "visual_relation"),
                **_distribution_summary(semantic_edges, "semantic_relation"),
                **_distribution_summary(np.abs(visual_edges - semantic_edges), "relation_abs_error"),
                **_distribution_summary(semantic_hard_values, "semantic_hard_negative_relation"),
                **_distribution_summary(visual_at_semantic_hard, "visual_at_semantic_hard_negative_relation"),
            }
        else:
            visual_distance = np.asarray([], dtype=np.float32)
            semantic_distance = np.asarray([], dtype=np.float32)
            structure = 0.0
            neighbor_preservation = 0.0
            distance_spearman = 0.0
            graph = {}
        hard_mean = self.hard_negative_sum / max(1, self.sample_count)
        hard_var = max(0.0, self.hard_negative_square_sum / max(1, self.sample_count) - hard_mean * hard_mean)
        alignment = {
            "true_prototype_similarity": float(self.true_similarity_sum / max(1, self.sample_count)),
            "hard_negative_similarity": float(hard_mean),
            "hard_negative_similarity_std": float(math.sqrt(hard_var)),
            "semantic_margin": float(self.semantic_margin_sum / max(1, self.sample_count)),
            "true_prototype_rank": float(self.rank_sum / max(1, self.sample_count)),
            "prototype_recall_at_k": float(self.recall_count / max(1, self.sample_count)),
            "class_center_prototype_cosine": float(center_cosine.mean()),
            "visual_semantic_structure_spearman": float(structure),
            "neighbor_preservation_at_k": float(neighbor_preservation),
            "visual_interclass_distance_mean": float(visual_distance.mean()) if visual_distance.size else 0.0,
            "visual_interclass_distance_std": float(visual_distance.std()) if visual_distance.size else 0.0,
            "semantic_interclass_distance_mean": float(semantic_distance.mean()) if semantic_distance.size else 0.0,
            "semantic_interclass_distance_std": float(semantic_distance.std()) if semantic_distance.size else 0.0,
            "visual_semantic_distance_spearman": float(distance_spearman),
            "semantic_ambiguity_rate": float(self.ambiguity_count / max(1, self.sample_count)),
        }
        if observed.size > 1:
            full_semantic = _normalize_numpy_rows(self.semantic) @ _normalize_numpy_rows(self.semantic).T
            np.fill_diagonal(full_semantic, -np.inf)
            k = min(self.recall_k, self.class_count - 1)
            neighbors = np.argsort(-full_semantic, axis=1)[:, :k]
            wrong_count = int(self.confusion.sum() - np.trace(self.confusion))
            covered = 0
            for target_id in range(self.class_count):
                for prediction_id in range(self.class_count):
                    if target_id == prediction_id:
                        continue
                    count = int(self.confusion[target_id, prediction_id])
                    if prediction_id in neighbors[target_id]:
                        covered += count
            graph["hard_negative_coverage"] = float(covered / wrong_count) if wrong_count else 1.0
            graph["confusion_edge_precision"] = graph["hard_negative_coverage"]
        return alignment, graph

    def class_aggregates(self) -> List[Dict[str, Any]]:
        rows = []
        for index, class_id in enumerate(self.candidate.tolist()):
            support = int(self.class_support[index])
            if support <= 0:
                continue
            correct_count = int(self.class_correct[index])
            rows.append({
                "class_id": int(class_id),
                "support": support,
                "correct_count": correct_count,
                "accuracy": float(correct_count / support),
            })
        return rows

    def finalize(self) -> Dict[str, Any]:
        if self.sample_count <= 0:
            return {
                "classification": {},
                "representation_geometry": {},
                "visual_semantic_alignment": {},
                "semantic_graph_reference": {},
            }
        per_class_values = [
            self.class_correct[index] / self.class_support[index]
            for index in range(self.class_count)
            if self.class_support[index] > 0
        ]
        classification = {
            "top1": float(self.correct_count / self.sample_count),
            "top5": float(self.top5_count / self.sample_count),
            "nll": float(self.nll_sum / self.sample_count),
            "per_class": float(np.mean(per_class_values)) if per_class_values else 0.0,
        }
        alignment, graph = self._relation_metrics()
        return {
            "classification": classification,
            "representation_geometry": self.representation.finalize(),
            "visual_semantic_alignment": alignment,
            "semantic_graph_reference": graph,
        }


class StreamingPatchDiversityAccumulator:
    def __init__(
        self,
        class_count: int,
        prompt_length: int,
        semantic_length: int,
    ) -> None:
        self.class_count = int(class_count)
        self.prompt_length = int(prompt_length)
        self.semantic_length = int(semantic_length)
        self.count = 0
        self.pairwise_cosine_sum = 0.0
        self.pairwise_cosine_square_sum = 0.0
        self.class_sum = np.zeros(self.class_count, dtype=np.float64)
        self.class_count_values = np.zeros(self.class_count, dtype=np.int64)

    def update(self, token_sequence: Any, labels: Any) -> None:
        tokens = torch.as_tensor(token_sequence).detach().to(
            device="cpu", dtype=torch.float32
        )
        target = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
        if tokens.dim() != 3 or tokens.shape[0] != target.numel():
            raise ValueError("token sequence batch has incompatible shapes")
        patch_start = 1 + self.prompt_length
        patch_end = int(tokens.shape[1]) - self.semantic_length
        patch_tokens = tokens[:, patch_start:patch_end, :]
        patch_count = int(patch_tokens.shape[1])
        if patch_count <= 1:
            return
        patch_direction = patch_tokens / patch_tokens.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        direction_sum = patch_direction.sum(dim=1)
        pairwise_cosine = (
            direction_sum.square().sum(dim=-1)
            - patch_direction.square().sum(dim=(1, 2))
        ) / float(patch_count * (patch_count - 1))
        pairwise_cosine = pairwise_cosine.clamp(min=-1.0, max=1.0)
        self.count += int(pairwise_cosine.numel())
        self.pairwise_cosine_sum += float(pairwise_cosine.sum().item())
        self.pairwise_cosine_square_sum += float(
            pairwise_cosine.square().sum().item()
        )
        pairwise_numpy = pairwise_cosine.numpy().astype(np.float64, copy=False)
        target_numpy = target.numpy()
        for class_id in np.unique(target_numpy):
            if int(class_id) < 0 or int(class_id) >= self.class_count:
                continue
            class_mask = target_numpy == class_id
            self.class_sum[int(class_id)] += float(
                pairwise_numpy[class_mask].sum()
            )
            self.class_count_values[int(class_id)] += int(class_mask.sum())

    def finalize(self) -> Dict[str, float]:
        if self.count <= 0:
            return {}
        cosine_mean = self.pairwise_cosine_sum / self.count
        cosine_variance = max(
            0.0,
            self.pairwise_cosine_square_sum / self.count
            - cosine_mean * cosine_mean,
        )
        observed_classes = self.class_count_values > 0
        class_cosine = (
            self.class_sum[observed_classes]
            / self.class_count_values[observed_classes]
        )
        return {
            "pairwise_cosine_mean": float(cosine_mean),
            "pairwise_cosine_std": float(math.sqrt(cosine_variance)),
            "dispersion_mean": float(1.0 - cosine_mean),
            "dispersion_std": float(math.sqrt(cosine_variance)),
            **_distribution_summary(
                class_cosine,
                "per_class_pairwise_cosine",
            ),
            **_distribution_summary(
                1.0 - class_cosine,
                "per_class_dispersion",
            ),
        }


class _PromptSemanticAlignmentAccumulator:
    def __init__(self, class_count: int, prompt_length: int) -> None:
        self.class_count = int(class_count)
        self.prompt_length = int(prompt_length)
        self.metrics: Dict[str, _RunningMean] = defaultdict(_RunningMean)
        self.class_profiles = _ClassPromptProfileState(
            self.class_count, self.prompt_length
        )

    def update(
        self,
        token_sequence: Any,
        labels: Any,
        semantic_prototypes: Optional[Any],
    ) -> None:
        if (
            semantic_prototypes is None
            or self.prompt_length <= 0
            or self.class_count < 2
        ):
            return
        tokens = torch.as_tensor(token_sequence).detach().to(
            device="cpu", dtype=torch.float32
        )
        targets = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
        semantics = torch.as_tensor(semantic_prototypes).detach().to(
            device="cpu", dtype=torch.float32
        )
        if (
            tokens.dim() != 3
            or tokens.shape[0] != targets.numel()
            or semantics.dim() != 2
            or semantics.shape[0] != self.class_count
            or tokens.shape[-1] != semantics.shape[-1]
            or tokens.shape[1] < 1 + self.prompt_length
        ):
            return
        valid = (targets >= 0) & (targets < self.class_count)
        if not bool(valid.any()):
            return
        prompts = tokens[valid, 1:1 + self.prompt_length, :]
        targets = targets[valid]
        prompts = torch.nn.functional.normalize(prompts, dim=-1)
        semantics = torch.nn.functional.normalize(semantics, dim=-1)
        similarities = torch.einsum("bpd,cd->bpc", prompts, semantics)
        batch_indices = torch.arange(targets.numel(), dtype=torch.long)
        true_scores = similarities[
            batch_indices[:, None],
            torch.arange(self.prompt_length, dtype=torch.long)[None, :],
            targets[:, None],
        ]
        non_target = similarities.clone()
        non_target[
            batch_indices[:, None],
            torch.arange(self.prompt_length, dtype=torch.long)[None, :],
            targets[:, None],
        ] = float("-inf")
        hard_negative = non_target.max(dim=-1).values
        margins = true_scores - hard_negative
        ranks = 1 + (similarities > true_scores.unsqueeze(-1)).sum(dim=-1)
        probabilities = torch.softmax(true_scores, dim=1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=1)
        effective_count = entropy.exp()
        sample_metrics = {
            "mean_prompt_true_prototype_similarity": true_scores.mean(dim=1),
            "best_prompt_true_prototype_similarity": true_scores.max(dim=1).values,
            "mean_prompt_hard_negative_similarity": hard_negative.mean(dim=1),
            "mean_prompt_semantic_margin": margins.mean(dim=1),
            "best_prompt_semantic_margin": margins.max(dim=1).values,
            "best_prompt_true_prototype_rank": ranks.min(dim=1).values.float(),
            "true_semantic_prompt_effective_count": effective_count,
            "true_semantic_prompt_effective_ratio": effective_count
            / max(1, self.prompt_length),
            "true_semantic_prompt_top1_share": probabilities.max(dim=1).values,
        }
        for name, values in sample_metrics.items():
            self.metrics[name].update_values(values)
        self.class_profiles.update(probabilities, targets)

    def finalize(self) -> Dict[str, float]:
        result = {
            name: state.value()
            for name, state in sorted(self.metrics.items())
            if state.count > 0
        }
        result.update({
            f"semantic_{name}": value
            for name, value in self.class_profiles.summary().items()
        })
        return result


class StreamingTokenViewAccumulator:
    def __init__(self, class_count: int, prompt_length: int, semantic_length: int) -> None:
        self.class_count = int(class_count)
        self.prompt_length = int(prompt_length)
        self.semantic_length = int(semantic_length)
        self.views: Dict[str, StreamingRepresentationAccumulator] = {}
        self.prompt_count = 0
        self.prompt_norm_sum = 0.0
        self.prompt_norm_square_sum = 0.0
        self.prompt_cls_cosine_sum = 0.0
        self.prompt_patch_cosine_sum = 0.0
        self.patch_diversity = StreamingPatchDiversityAccumulator(
            self.class_count,
            self.prompt_length,
            self.semantic_length,
        )
        self.prompt_semantic = (
            _PromptSemanticAlignmentAccumulator(
                self.class_count, self.prompt_length
            )
            if self.prompt_length > 0
            else None
        )

    def update(
        self,
        token_sequence: Any,
        labels: Any,
        semantic_prototypes: Optional[Any] = None,
    ) -> None:
        tokens = torch.as_tensor(token_sequence).detach().to(device="cpu", dtype=torch.float32)
        target = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
        if tokens.dim() != 3 or tokens.shape[0] != target.numel():
            raise ValueError("token sequence batch has incompatible shapes")
        patch_start = 1 + self.prompt_length
        patch_end = int(tokens.shape[1]) - self.semantic_length
        patch_tokens = tokens[:, patch_start:patch_end, :]
        view_values = {
            "cls": tokens[:, 0, :],
            "pooled_patch": patch_tokens.mean(dim=1),
        }
        if self.prompt_length > 0:
            view_values["contextualized_prompt"] = tokens[:, 1:patch_start, :].mean(dim=1)
        if self.semantic_length > 0:
            view_values["semantic_token"] = tokens[:, patch_end:, :].mean(dim=1)
        for name, values in view_values.items():
            state = self.views.setdefault(name, StreamingRepresentationAccumulator(self.class_count))
            state.update(values, target)
        self.patch_diversity.update(tokens, target)
        if self.prompt_semantic is not None:
            self.prompt_semantic.update(tokens, target, semantic_prototypes)
        if self.prompt_length > 0:
            prompt = view_values["contextualized_prompt"]
            cls = view_values["cls"]
            patch = view_values["pooled_patch"]
            prompt_norm = prompt.norm(dim=1)
            prompt_normalized = prompt / prompt_norm.clamp_min(1e-12).unsqueeze(1)
            cls_normalized = cls / cls.norm(dim=1).clamp_min(1e-12).unsqueeze(1)
            patch_normalized = patch / patch.norm(dim=1).clamp_min(1e-12).unsqueeze(1)
            self.prompt_norm_sum += float(prompt_norm.sum().item())
            self.prompt_norm_square_sum += float(prompt_norm.square().sum().item())
            self.prompt_cls_cosine_sum += float((prompt_normalized * cls_normalized).sum(dim=1).sum().item())
            self.prompt_patch_cosine_sum += float((prompt_normalized * patch_normalized).sum(dim=1).sum().item())
            self.prompt_count += int(prompt.shape[0])

    def finalize(self) -> Dict[str, Any]:
        geometry = {name: state.finalize() for name, state in self.views.items()}
        patch_diversity = self.patch_diversity.finalize()
        if patch_diversity:
            geometry["within_image_patch_diversity"] = patch_diversity
        prompt_response: Dict[str, float] = {}
        relation: Dict[str, float] = {}
        if self.prompt_count > 0:
            mean_norm = self.prompt_norm_sum / self.prompt_count
            prompt_response = {
                "contextualized_prompt_norm": float(mean_norm),
                "contextualized_prompt_norm_std": float(math.sqrt(max(0.0, self.prompt_norm_square_sum / self.prompt_count - mean_norm * mean_norm))),
                "prompt_cls_cosine": float(self.prompt_cls_cosine_sum / self.prompt_count),
                "prompt_patch_cosine": float(self.prompt_patch_cosine_sum / self.prompt_count),
            }
            prompt_state = self.views.get("contextualized_prompt")
            cls_state = self.views.get("cls")
            patch_state = self.views.get("pooled_patch")
            if prompt_state is not None and cls_state is not None and patch_state is not None:
                if prompt_state.feature_cross is not None and prompt_state.feature_sum is not None:
                    prompt_mean = prompt_state.feature_sum / max(1, prompt_state.count)
                    prompt_variance = (
                        torch.diag(prompt_state.feature_cross) / max(1, prompt_state.count)
                        - prompt_mean.square()
                    ).clamp_min(0.0)
                    prompt_response["contextualized_prompt_instance_variance"] = float(
                        prompt_variance.mean().item()
                    )
                prompt_ids, prompt_centers = prompt_state.class_centers()
                cls_ids, cls_centers = cls_state.class_centers()
                patch_ids, patch_centers = patch_state.class_centers()
                if np.array_equal(prompt_ids, cls_ids) and np.array_equal(prompt_ids, patch_ids) and prompt_ids.size > 1:
                    prompt_relation = _normalize_numpy_rows(prompt_centers) @ _normalize_numpy_rows(prompt_centers).T
                    cls_relation = _normalize_numpy_rows(cls_centers) @ _normalize_numpy_rows(cls_centers).T
                    patch_relation = _normalize_numpy_rows(patch_centers) @ _normalize_numpy_rows(patch_centers).T
                    upper = np.triu_indices(prompt_ids.size, k=1)
                    relation = {
                        "prompt_cls_gram_alignment": spearman_correlation(prompt_relation[upper], cls_relation[upper]),
                        "prompt_patch_gram_alignment": spearman_correlation(prompt_relation[upper], patch_relation[upper]),
                        **_distribution_summary(np.abs(prompt_relation[upper] - cls_relation[upper]), "prompt_cls_relation_abs_error"),
                        **_distribution_summary(np.abs(prompt_relation[upper] - patch_relation[upper]), "prompt_patch_relation_abs_error"),
                    }
        return {
            "representation_geometry": geometry,
            "prompt_parameter_health": prompt_response,
            "relation_stability": relation,
            "prompt_semantic_role_reference": (
                self.prompt_semantic.finalize()
                if self.prompt_semantic is not None
                else {}
            ),
        }
