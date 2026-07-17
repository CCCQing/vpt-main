from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torchvision as tv


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
        if selected_support[class_id] < min(per_class_quota, len(by_class[class_id]))
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


class FixedProbeDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset: Any, manifest: Mapping[str, Any], transform: Any) -> None:
        self.source_dataset = source_dataset
        self.manifest = dict(manifest)
        self.rows = list(manifest.get("samples", []))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[int(index)]
        image = tv.datasets.folder.default_loader(str(row["image_path"]))
        image = self.transform(image)
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


def affinity_health_metrics(
    affinities: Iterable[Mapping[str, Any]],
    *,
    temperature: float = 1.0,
    saturation_threshold: float = 10.0,
) -> Dict[str, float]:
    relation_keys = (
        "QpKv_raw", "QvKp_raw", "QcKp_raw",
        "QsKv_raw", "QvKs_raw", "QcKs_raw",
        "QsKp_raw", "QpKs_raw",
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
        if cls_patch.numel():
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
            prompt_to_patch = data[:, :, prompt_slice, patch_slice]
            patch_to_prompt = data[:, :, patch_slice, prompt_slice]
            prompt_to_cls = data[:, :, prompt_slice, 0]
            buckets["prompt_to_patch_mass"].append(float(prompt_to_patch.sum(dim=-1).mean().item()))
            buckets["patch_to_prompt_mass"].append(float(patch_to_prompt.sum(dim=-1).mean().item()))
            buckets["prompt_to_cls_mass"].append(float(prompt_to_cls.mean().item()))
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
                    buckets[metric_name].append(float(value.detach().float().sum(dim=-1).mean().item()))
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
