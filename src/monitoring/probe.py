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


class _AttentionLayerAccumulator:
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
        if cls_patch.numel():
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
            prompt_to_patch = data[:, :, prompt_slice, patch_slice]
            patch_to_prompt = data[:, :, patch_slice, prompt_slice]
            prompt_to_cls = data[:, :, prompt_slice, 0]
            self.means["prompt_to_patch_mass"].update_values(prompt_to_patch.sum(dim=-1))
            self.means["patch_to_prompt_mass"].update_values(patch_to_prompt.sum(dim=-1))
            self.means["prompt_to_cls_mass"].update_values(prompt_to_cls)
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
                self.means[metric_name].update_values(value.detach().float().sum(dim=-1))
                self.observed = True
        return sample_entropy

    def finalize(self) -> Dict[str, float]:
        result = {}
        for name, state in self.means.items():
            value = state.value()
            if value is not None:
                result[name] = float(value)
        if self.observed:
            result["attention_effective_rank"] = self.effective_rank.value()
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
        "QpKv_raw", "QvKp_raw", "QcKp_raw",
        "QsKv_raw", "QvKs_raw", "QcKs_raw",
        "QsKp_raw", "QpKs_raw",
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


class ProbeAttentionAffinityAccumulator:
    def __init__(
        self,
        *,
        prompt_length: int,
        semantic_length: int,
        selected_layers: Sequence[int],
        temperature: float = 1.0,
        saturation_threshold: float = 10.0,
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
        self.fallback_affinity_keys = {
            "AcKv_attn", "ApKv_attn", "AvKp_attn", "ApKc_attn",
            "AsKv_attn", "AvKs_attn", "AsKp_attn", "ApKs_attn",
        }

    def update(
        self,
        attention_layers: Sequence[Any],
        affinity_layers: Sequence[Any],
        *,
        predictions: Sequence[int],
        targets: Sequence[int],
    ) -> None:
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
            if not self.attention.primary_entropy_observed:
                required_keys.update(self.fallback_affinity_keys)
            cpu_layer = {
                key: value.detach().to(device="cpu", dtype=torch.float32)
                for key, value in layer.items()
                if key in required_keys and torch.is_tensor(value)
            }
            indexed_affinity = [{} for _ in range(layer_index)] + [cpu_layer]
            if not self.attention.primary_entropy_observed:
                self.attention.update(
                    [],
                    indexed_affinity,
                    predictions=predictions,
                    targets=targets,
                )
            self.affinity.update(indexed_affinity)
            del indexed_affinity, cpu_layer

    def finalize(self) -> Dict[str, Dict[str, float]]:
        return {
            "attention_flow": self.attention.finalize(),
            "affinity_health": self.affinity.finalize(),
        }


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
