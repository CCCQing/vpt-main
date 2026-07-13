#!/usr/bin/env python3
"""Evaluate Stage-2 Graph-GP held-out class-center transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


METRIC_KEYS = (
    "raw_cosine",
    "centered_cosine",
    "nrmse",
    "geometry_spearman",
    "pseudo_seen",
    "pseudo_unseen",
    "pseudo_h",
    "pseudo_unseen_zsl",
    "uncertainty_error_spearman",
)
ALIGNMENT_METRIC_KEYS = (
    "graph_prompt_spearman",
    "graph_prompt_cka",
    "graph_prompt_topk_overlap",
    "graph_final_logits_spearman",
    "graph_final_logits_cka",
    "graph_final_logits_topk_overlap",
    "graph_final_visual_spearman",
    "graph_final_visual_cka",
    "graph_final_visual_topk_overlap",
    "graph_final_semantic_spearman",
    "graph_final_semantic_cka",
    "graph_final_semantic_topk_overlap",
    "prompt_final_logits_spearman",
    "prompt_final_logits_cka",
    "prompt_final_logits_topk_overlap",
    "prompt_final_visual_spearman",
    "prompt_final_visual_cka",
    "prompt_final_visual_topk_overlap",
    "prompt_final_semantic_spearman",
    "prompt_final_semantic_cka",
    "prompt_final_semantic_topk_overlap",
    "final_logits_visual_spearman",
    "final_logits_visual_cka",
    "final_logits_visual_topk_overlap",
    "final_visual_semantic_spearman",
    "final_visual_semantic_cka",
    "final_visual_semantic_topk_overlap",
    "final_visual_semantic_paired_cosine_mean",
    "final_visual_semantic_paired_centered_cosine_mean",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_cache(path: Path) -> Dict[str, Any]:
    with np.load(str(path), allow_pickle=False) as payload:
        required = {
            "sample_ids",
            "global_labels",
            "posterior_mu",
            "posterior_logvar",
            "image_paths",
            "class_attributes",
            "seen_class_ids",
            "classifier_logits",
            "classifier_visual_repr",
            "classifier_semantic_repr",
            "classifier_class_ids",
            "metadata_json",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"Posterior cache is missing keys {missing}: {path}")
        cache = {key: payload[key] for key in required}
    metadata = json.loads(str(cache.pop("metadata_json").item()))
    if metadata.get("format") != "graph_gp_posterior_cache_v2":
        raise ValueError(f"Unsupported posterior cache format in {path}.")
    sample_ids = np.asarray(cache["sample_ids"], dtype=np.int64)
    labels = np.asarray(cache["global_labels"], dtype=np.int64)
    mu = np.asarray(cache["posterior_mu"], dtype=np.float64)
    logvar = np.asarray(cache["posterior_logvar"], dtype=np.float64)
    classifier_logits = np.asarray(cache["classifier_logits"], dtype=np.float64)
    classifier_visual_repr = np.asarray(cache["classifier_visual_repr"], dtype=np.float64)
    classifier_semantic_repr = np.asarray(cache["classifier_semantic_repr"], dtype=np.float64)
    classifier_class_ids = np.asarray(cache["classifier_class_ids"], dtype=np.int64)
    if sample_ids.ndim != 1 or labels.shape != sample_ids.shape:
        raise ValueError("sample_ids/global_labels must be aligned rank-1 arrays.")
    if mu.ndim != 2 or logvar.shape != mu.shape or mu.shape[0] != sample_ids.shape[0]:
        raise ValueError("posterior_mu/posterior_logvar must be aligned [N,D] arrays.")
    if classifier_logits.ndim != 2 or classifier_logits.shape[0] != sample_ids.shape[0]:
        raise ValueError("classifier_logits must be an aligned [N,C] array.")
    if classifier_visual_repr.ndim != 2 or classifier_visual_repr.shape[0] != sample_ids.shape[0]:
        raise ValueError("classifier_visual_repr must be an aligned [N,D] array.")
    if classifier_semantic_repr.ndim != 2 or classifier_semantic_repr.shape[0] != classifier_logits.shape[1]:
        raise ValueError("classifier_semantic_repr must align with classifier logit columns.")
    if classifier_semantic_repr.shape[1] != classifier_visual_repr.shape[1]:
        raise ValueError("classifier visual/semantic representations must share the classifier space dimension.")
    if classifier_class_ids.shape != (classifier_logits.shape[1],):
        raise ValueError("classifier_class_ids must align with classifier logit columns.")
    if not np.array_equal(sample_ids, np.arange(sample_ids.size, dtype=np.int64)):
        raise ValueError("Stage-2 cache requires deterministic contiguous sample_ids 0..N-1.")
    finite_arrays = (mu, logvar, classifier_logits, classifier_visual_repr, classifier_semantic_repr)
    if not all(np.isfinite(array).all() for array in finite_arrays):
        raise FloatingPointError("Posterior/classifier cache contains NaN or Inf.")
    cache.update(
        {
            "sample_ids": sample_ids,
            "global_labels": labels,
            "posterior_mu": mu,
            "posterior_logvar": logvar,
            "classifier_logits": classifier_logits,
            "classifier_visual_repr": classifier_visual_repr,
            "classifier_semantic_repr": classifier_semantic_repr,
            "classifier_class_ids": classifier_class_ids,
            "class_attributes": np.asarray(cache["class_attributes"], dtype=np.float64),
            "seen_class_ids": np.asarray(cache["seen_class_ids"], dtype=np.int64),
            "image_paths": np.asarray(cache["image_paths"], dtype=np.str_),
            "metadata": metadata,
        }
    )
    seen = set(int(x) for x in cache["seen_class_ids"])
    labels_present = set(int(x) for x in np.unique(labels))
    if labels_present != seen:
        raise ValueError(
            f"Cache labels do not exactly cover seen classes: labels={len(labels_present)} seen={len(seen)}."
        )
    if not np.array_equal(cache["classifier_class_ids"], cache["seen_class_ids"]):
        raise ValueError("classifier_class_ids must exactly match seen_class_ids and their order.")
    for class_id in sorted(seen):
        if int((labels == class_id).sum()) < 2:
            raise ValueError(f"Class {class_id} has fewer than two cached samples.")
    return cache


def _cache_checksum(cache: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(cache["sample_ids"], dtype=np.int64).tobytes())
    digest.update(np.asarray(cache["global_labels"], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _create_manifest(
    path: Path,
    cache: Mapping[str, Any],
    fold_seed: int,
    num_folds: int,
    center_ratio: float,
    shuffle_count: int,
) -> Dict[str, Any]:
    seen_ids = np.asarray(cache["seen_class_ids"], dtype=np.int64)
    labels = np.asarray(cache["global_labels"], dtype=np.int64)
    rng = np.random.RandomState(int(fold_seed))
    class_perm = rng.permutation(seen_ids)
    query_folds = [sorted(int(x) for x in fold.tolist()) for fold in np.array_split(class_perm, num_folds)]
    image_splits: Dict[str, Dict[str, List[int]]] = {}
    for class_id in sorted(int(x) for x in seen_ids):
        ids = np.flatnonzero(labels == class_id).astype(np.int64)
        ids = rng.permutation(ids)
        center_count = int(math.floor(float(center_ratio) * float(ids.size)))
        center_count = max(1, min(center_count, int(ids.size) - 1))
        image_splits[str(class_id)] = {
            "center_ids": sorted(int(x) for x in ids[:center_count]),
            "eval_ids": sorted(int(x) for x in ids[center_count:]),
        }
    shuffle_permutations = [rng.permutation(len(seen_ids)).astype(int).tolist() for _ in range(shuffle_count)]
    manifest = {
        "format": "graph_gp_stage2_manifest_v1",
        "dataset": str(cache["metadata"].get("dataset", "")),
        "fold_seed": int(fold_seed),
        "num_folds": int(num_folds),
        "center_ratio": float(center_ratio),
        "shuffle_count": int(shuffle_count),
        "seen_class_ids": [int(x) for x in seen_ids],
        "query_folds": query_folds,
        "image_splits": image_splits,
        "shuffle_permutations": shuffle_permutations,
        "cache_sample_checksum": _cache_checksum(cache),
    }
    _write_json(path, manifest)
    return manifest


def _load_or_create_manifest(args: argparse.Namespace, cache: Mapping[str, Any]) -> Dict[str, Any]:
    if args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest = _create_manifest(
            args.manifest,
            cache,
            args.fold_seed,
            args.num_folds,
            args.center_ratio,
            args.shuffle_count,
        )
        print(f"wrote {args.manifest}", flush=True)
    expected = {
        "format": "graph_gp_stage2_manifest_v1",
        "fold_seed": int(args.fold_seed),
        "num_folds": int(args.num_folds),
        "shuffle_count": int(args.shuffle_count),
        "seen_class_ids": [int(x) for x in cache["seen_class_ids"]],
        "cache_sample_checksum": _cache_checksum(cache),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest mismatch for {key}: expected={value} actual={manifest.get(key)}")
    if not math.isclose(float(manifest.get("center_ratio", -1.0)), float(args.center_ratio), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Manifest center_ratio does not match the requested value.")
    _validate_manifest_contents(manifest, cache)
    return manifest


def _validate_manifest_contents(manifest: Mapping[str, Any], cache: Mapping[str, Any]) -> None:
    seen_ids = [int(x) for x in cache["seen_class_ids"]]
    seen_set = set(seen_ids)
    query_folds = manifest.get("query_folds")
    if not isinstance(query_folds, list) or len(query_folds) != int(manifest["num_folds"]):
        raise ValueError("Manifest query_folds does not match num_folds.")
    flattened = [int(class_id) for fold in query_folds for class_id in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != seen_set:
        raise ValueError("Manifest query folds must partition every seen class exactly once.")

    labels = np.asarray(cache["global_labels"], dtype=np.int64)
    image_splits = manifest.get("image_splits")
    if not isinstance(image_splits, dict) or set(image_splits) != {str(x) for x in seen_ids}:
        raise ValueError("Manifest image_splits must contain every seen class exactly once.")
    for class_id in seen_ids:
        split = image_splits[str(class_id)]
        center_ids = [int(x) for x in split.get("center_ids", [])]
        eval_ids = [int(x) for x in split.get("eval_ids", [])]
        expected_ids = set(int(x) for x in np.flatnonzero(labels == class_id))
        if not center_ids or not eval_ids:
            raise ValueError(f"Manifest class {class_id} requires non-empty center/eval image splits.")
        if set(center_ids).intersection(eval_ids) or set(center_ids).union(eval_ids) != expected_ids:
            raise ValueError(f"Manifest image split is not an exact partition for class {class_id}.")

    permutations = manifest.get("shuffle_permutations")
    if not isinstance(permutations, list) or len(permutations) != int(manifest["shuffle_count"]):
        raise ValueError("Manifest shuffle_permutations does not match shuffle_count.")
    expected_permutation = list(range(len(seen_ids)))
    for index, permutation in enumerate(permutations):
        if sorted(int(x) for x in permutation) != expected_permutation:
            raise ValueError(f"Manifest shuffled permutation {index} is invalid.")


def _clean_kernel(kernel: np.ndarray, eps: float) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Kernel must be square, got {kernel.shape}.")
    kernel = 0.5 * (kernel + kernel.T)
    kernel = np.maximum(kernel, 0.0)
    diag = np.sqrt(np.maximum(np.diag(kernel), eps))
    kernel = kernel / np.maximum(diag[:, None] * diag[None, :], eps)
    if not np.isfinite(kernel).all():
        raise FloatingPointError("Kernel contains NaN or Inf after cleaning.")
    return kernel


def _nearest_psd(kernel: np.ndarray, eps: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (kernel + kernel.T))
    projected = (vectors * np.maximum(values, eps)[None, :]) @ vectors.T
    return _clean_kernel(projected, eps)


def _load_kernels(path: Path, seen_ids: np.ndarray, class_attributes: np.ndarray, eps: float) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as payload:
        required = {"method1_diff", "method2_diff", "method3_diff"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"Graph bundle is missing {missing}: {path}")
        full = {name: _clean_kernel(payload[name], eps) for name in sorted(required)}
    full["method1_psd"] = _nearest_psd(full["method1_diff"], eps)
    max_seen = int(seen_ids.max())
    for name, kernel in full.items():
        if kernel.shape[0] <= max_seen:
            raise ValueError(f"Kernel {name} shape {kernel.shape} cannot index class {max_seen}.")
    kernels = {name: matrix[np.ix_(seen_ids, seen_ids)] for name, matrix in full.items()}
    attrs = np.asarray(class_attributes, dtype=np.float64)[seen_ids]
    attr_norm = attrs / np.maximum(np.linalg.norm(attrs, axis=1, keepdims=True), eps)
    kernels["attribute_cosine"] = _clean_kernel(attr_norm @ attr_norm.T, eps)
    return kernels


def _class_center(mu: np.ndarray, sample_ids: Sequence[int]) -> Tuple[np.ndarray, float, float]:
    values = mu[np.asarray(sample_ids, dtype=np.int64)]
    center = values.mean(axis=0)
    residual = values - center[None, :]
    within_var = float(np.maximum(values.var(axis=0), 0.0).mean())
    within_squared_l2 = float(np.sum(residual * residual, axis=1).mean())
    return center, within_var, within_squared_l2


def graph_gp_condition(
    kernel: np.ndarray,
    support_indices: np.ndarray,
    query_indices: np.ndarray,
    support_centers: np.ndarray,
    observation_noise: np.ndarray,
    ridge: float,
    mean_function: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    k_ss = kernel[np.ix_(support_indices, support_indices)]
    k_qs = kernel[np.ix_(query_indices, support_indices)]
    system = k_ss + np.diag(np.asarray(observation_noise, dtype=np.float64) + float(ridge))
    mean = support_centers.mean(axis=0, keepdims=True) if mean_function else np.zeros((1, support_centers.shape[1]))
    rhs = support_centers - mean
    solved = np.linalg.solve(system, rhs)
    prediction = mean + k_qs @ solved
    solved_k = np.linalg.solve(system, k_qs.T)
    uncertainty = np.maximum(np.diag(kernel)[query_indices] - np.sum(k_qs * solved_k.T, axis=1), 0.0)
    residual = np.linalg.norm(system @ solved - rhs) / max(np.linalg.norm(rhs), 1e-12)
    diagnostics = {
        "system_condition": float(np.linalg.cond(system)),
        "solve_relative_residual": float(residual),
        "prediction_center_norm": float(np.linalg.norm(prediction, axis=1).mean()),
    }
    return prediction, uncertainty, diagnostics


def _global_mean_predict(support_centers: np.ndarray, query_count: int) -> np.ndarray:
    return np.repeat(support_centers.mean(axis=0, keepdims=True), int(query_count), axis=0)


def _graph_knn_predict(
    kernel: np.ndarray,
    support_indices: np.ndarray,
    query_indices: np.ndarray,
    support_centers: np.ndarray,
    topk: int,
    tau: float,
) -> np.ndarray:
    similarity = kernel[np.ix_(query_indices, support_indices)]
    k = min(int(topk), int(support_indices.size))
    top = np.argpartition(similarity, -k, axis=1)[:, -k:]
    top_similarity = np.take_along_axis(similarity, top, axis=1)
    shifted = top_similarity / float(tau)
    shifted -= shifted.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    centers = support_centers[top]
    return np.sum(weights[:, :, None] * centers, axis=1)


def _attribute_ridge_predict(
    class_attributes: np.ndarray,
    support_global_ids: np.ndarray,
    query_global_ids: np.ndarray,
    support_centers: np.ndarray,
    ridge: float,
) -> np.ndarray:
    a_s = np.asarray(class_attributes[support_global_ids], dtype=np.float64)
    a_q = np.asarray(class_attributes[query_global_ids], dtype=np.float64)
    system = a_s @ a_s.T + float(ridge) * np.eye(a_s.shape[0], dtype=np.float64)
    dual = np.linalg.solve(system, support_centers)
    return (a_q @ a_s.T) @ dual


def _cosine_rows(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)
    return np.sum(a * b, axis=1) / denom


def _pairwise_distance_values(x: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    return distance[np.triu_indices(x.shape[0], k=1)]


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    result = spearmanr(a, b)
    return float(result[0] if isinstance(result, tuple) else result.correlation)


def _centered_cosine_kernel(features: np.ndarray, eps: float) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), eps)
    relation = normalized @ normalized.T
    return np.clip(0.5 * (relation + relation.T), -1.0, 1.0)


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=np.float64)
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def _kernel_alignment(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    centered_a = _center_kernel(a)
    centered_b = _center_kernel(b)
    denom = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)
    return float(np.sum(centered_a * centered_b) / max(float(denom), eps))


def _topk_neighbor_overlap(a: np.ndarray, b: np.ndarray, topk: int) -> float:
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Neighbor-overlap relations must be aligned square matrices.")
    k = min(int(topk), int(a.shape[0]) - 1)
    if k <= 0:
        return float("nan")
    a_rank = np.asarray(a, dtype=np.float64).copy()
    b_rank = np.asarray(b, dtype=np.float64).copy()
    np.fill_diagonal(a_rank, -np.inf)
    np.fill_diagonal(b_rank, -np.inf)
    a_top = np.argpartition(a_rank, -k, axis=1)[:, -k:]
    b_top = np.argpartition(b_rank, -k, axis=1)[:, -k:]
    overlaps = [len(set(left).intersection(int(x) for x in right)) / float(k) for left, right in zip(a_top, b_top)]
    return float(np.mean(overlaps))


def _relation_alignment(a: np.ndarray, b: np.ndarray, topk: int, eps: float) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Relation shape mismatch: {a.shape} vs {b.shape}.")
    upper = np.triu_indices(a.shape[0], k=1)
    return {
        "spearman": _safe_spearman(a[upper], b[upper]),
        "cka": _kernel_alignment(a, b, eps),
        "topk_overlap": _topk_neighbor_overlap(a, b, topk),
    }


def _prefixed_alignment(prefix: str, values: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in values.items()}


def _macro_accuracy(pred: np.ndarray, target: np.ndarray, class_ids: np.ndarray) -> float:
    values = []
    for class_id in class_ids:
        mask = target == int(class_id)
        if not mask.any():
            raise ValueError(f"No evaluation samples for class {int(class_id)}.")
        values.append(float((pred[mask] == target[mask]).mean()))
    return float(np.mean(values))


def _nearest_prototype(samples: np.ndarray, prototypes: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    samples32 = np.asarray(samples, dtype=np.float32)
    prototypes32 = np.asarray(prototypes, dtype=np.float32)
    distance = (
        np.sum(samples32 * samples32, axis=1, keepdims=True)
        + np.sum(prototypes32 * prototypes32, axis=1)[None, :]
        - 2.0 * (samples32 @ prototypes32.T)
    )
    return class_ids[np.argmin(distance, axis=1)]


def _center_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    support_mean: np.ndarray,
    uncertainty: Optional[np.ndarray],
    eps: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    raw_cos = _cosine_rows(prediction, truth, eps)
    centered_pred = prediction - support_mean[None, :]
    centered_truth = truth - support_mean[None, :]
    centered_cos = _cosine_rows(centered_pred, centered_truth, eps)
    errors = np.linalg.norm(prediction - truth, axis=1)
    denom = np.maximum(np.linalg.norm(centered_truth, axis=1), eps)
    nrmse = errors / denom
    metrics = {
        "raw_cosine": float(raw_cos.mean()),
        "centered_cosine": float(centered_cos.mean()),
        "nrmse": float(nrmse.mean()),
        "geometry_spearman": _safe_spearman(
            _pairwise_distance_values(prediction),
            _pairwise_distance_values(truth),
        ),
        "uncertainty_error_spearman": (
            _safe_spearman(np.asarray(uncertainty), errors) if uncertainty is not None else float("nan")
        ),
    }
    rows = []
    for index in range(prediction.shape[0]):
        rows.append(
            {
                "raw_cosine": float(raw_cos[index]),
                "centered_cosine": float(centered_cos[index]),
                "nrmse": float(nrmse[index]),
                "center_error": float(errors[index]),
                "predictive_uncertainty": (
                    float(uncertainty[index]) if uncertainty is not None else float("nan")
                ),
            }
        )
    return metrics, rows


def _pseudo_gzsl_metrics(
    mu: np.ndarray,
    labels: np.ndarray,
    seen_ids: np.ndarray,
    support_global_ids: np.ndarray,
    query_global_ids: np.ndarray,
    support_centers: np.ndarray,
    query_prediction: np.ndarray,
    support_eval_ids: Sequence[int],
    query_sample_ids: Sequence[int],
) -> Dict[str, float]:
    prototype_by_class = {
        int(class_id): support_centers[index]
        for index, class_id in enumerate(support_global_ids)
    }
    prototype_by_class.update(
        {int(class_id): query_prediction[index] for index, class_id in enumerate(query_global_ids)}
    )
    candidate_ids = np.asarray(seen_ids, dtype=np.int64)
    prototypes = np.stack([prototype_by_class[int(class_id)] for class_id in candidate_ids], axis=0)
    eval_ids = np.asarray(list(support_eval_ids) + list(query_sample_ids), dtype=np.int64)
    eval_labels = labels[eval_ids]
    prediction = _nearest_prototype(mu[eval_ids], prototypes, candidate_ids)
    support_mask = np.isin(eval_labels, support_global_ids)
    query_mask = np.isin(eval_labels, query_global_ids)
    pseudo_seen = _macro_accuracy(prediction[support_mask], eval_labels[support_mask], support_global_ids)
    pseudo_unseen = _macro_accuracy(prediction[query_mask], eval_labels[query_mask], query_global_ids)
    pseudo_h = (
        0.0
        if pseudo_seen + pseudo_unseen <= 0.0
        else 2.0 * pseudo_seen * pseudo_unseen / (pseudo_seen + pseudo_unseen)
    )
    query_prediction_only = _nearest_prototype(mu[np.asarray(query_sample_ids)], query_prediction, query_global_ids)
    pseudo_unseen_zsl = _macro_accuracy(
        query_prediction_only,
        labels[np.asarray(query_sample_ids)],
        query_global_ids,
    )
    return {
        "pseudo_seen": float(pseudo_seen),
        "pseudo_unseen": float(pseudo_unseen),
        "pseudo_h": float(pseudo_h),
        "pseudo_unseen_zsl": float(pseudo_unseen_zsl),
    }


def _kernel_diagnostics(kernel: Optional[np.ndarray]) -> Dict[str, float]:
    if kernel is None:
        return {}
    eigenvalues = np.linalg.eigvalsh(0.5 * (kernel + kernel.T))
    positive = np.maximum(eigenvalues, 0.0)
    total = positive.sum()
    if total > 0.0:
        prob = positive / total
        effective_rank = float(np.exp(-(prob[prob > 0.0] * np.log(prob[prob > 0.0])).sum()))
    else:
        effective_rank = 0.0
    return {
        "kernel_min_eigenvalue": float(eigenvalues.min()),
        "kernel_negative_eigenvalue_count": int((eigenvalues < -1e-10).sum()),
        "kernel_effective_rank": effective_rank,
    }


def _method_predictions(
    args: argparse.Namespace,
    kernels: Mapping[str, np.ndarray],
    class_attributes: np.ndarray,
    seen_ids: np.ndarray,
    support_indices: np.ndarray,
    query_indices: np.ndarray,
    support_centers: np.ndarray,
    support_noise: np.ndarray,
    manifest: Mapping[str, Any],
) -> Iterable[Tuple[str, str, int, np.ndarray, Optional[np.ndarray], Dict[str, float], Optional[np.ndarray]]]:
    for method in ("method1_diff", "method2_diff", "method3_diff"):
        prediction, uncertainty, diagnostics = graph_gp_condition(
            kernels[method], support_indices, query_indices, support_centers, support_noise, args.gp_ridge
        )
        yield method, method, -1, prediction, uncertainty, diagnostics, kernels[method]

    prediction, uncertainty, diagnostics = graph_gp_condition(
        kernels["method1_diff"],
        support_indices,
        query_indices,
        support_centers,
        support_noise,
        args.gp_ridge,
        mean_function=True,
    )
    yield "method1_mean_function", "method1_mean_function", -1, prediction, uncertainty, diagnostics, kernels["method1_diff"]

    prediction, uncertainty, diagnostics = graph_gp_condition(
        kernels["method1_psd"], support_indices, query_indices, support_centers, support_noise, args.gp_ridge
    )
    yield "method1_nearest_psd", "method1_nearest_psd", -1, prediction, uncertainty, diagnostics, kernels["method1_psd"]

    prediction, uncertainty, diagnostics = graph_gp_condition(
        kernels["attribute_cosine"], support_indices, query_indices, support_centers, support_noise, args.gp_ridge
    )
    yield "attribute_cosine_gp", "attribute_cosine_gp", -1, prediction, uncertainty, diagnostics, kernels["attribute_cosine"]

    for shuffle_index, permutation in enumerate(manifest["shuffle_permutations"]):
        permutation = np.asarray(permutation, dtype=np.int64)
        shuffled = kernels["method1_diff"][np.ix_(permutation, permutation)]
        prediction, uncertainty, diagnostics = graph_gp_condition(
            shuffled, support_indices, query_indices, support_centers, support_noise, args.gp_ridge
        )
        yield (
            f"method1_shuffled_{shuffle_index:02d}",
            "method1_shuffled",
            int(shuffle_index),
            prediction,
            uncertainty,
            diagnostics,
            shuffled,
        )

    prediction = _global_mean_predict(support_centers, query_indices.size)
    yield "global_mean", "global_mean", -1, prediction, None, {}, None

    prediction = np.zeros((query_indices.size, support_centers.shape[1]), dtype=np.float64)
    yield "identity_zero_mean", "identity_zero_mean", -1, prediction, None, {}, None

    prediction = _graph_knn_predict(
        kernels["method1_diff"],
        support_indices,
        query_indices,
        support_centers,
        args.knn_k,
        args.knn_tau,
    )
    yield "method1_graph_knn", "method1_graph_knn", -1, prediction, None, {}, kernels["method1_diff"]

    prediction = _attribute_ridge_predict(
        class_attributes,
        seen_ids[support_indices],
        seen_ids[query_indices],
        support_centers,
        args.attribute_ridge,
    )
    yield "attribute_ridge", "attribute_ridge", -1, prediction, None, {}, None


def evaluate(args: argparse.Namespace) -> None:
    cache = _load_cache(args.cache)
    manifest = _load_or_create_manifest(args, cache)
    seen_ids = np.asarray(cache["seen_class_ids"], dtype=np.int64)
    class_to_local = {int(class_id): index for index, class_id in enumerate(seen_ids)}
    mu = np.asarray(cache["posterior_mu"], dtype=np.float64)
    labels = np.asarray(cache["global_labels"], dtype=np.int64)
    classifier_logits = np.asarray(cache["classifier_logits"], dtype=np.float64)
    classifier_visual_repr = np.asarray(cache["classifier_visual_repr"], dtype=np.float64)
    classifier_semantic_repr = np.asarray(cache["classifier_semantic_repr"], dtype=np.float64)
    class_attributes = np.asarray(cache["class_attributes"], dtype=np.float64)
    kernels = _load_kernels(args.graph_bundle, seen_ids, class_attributes, args.eps)
    model_seed = int(cache["metadata"].get("seed"))

    fold_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    class_center_rows: List[Dict[str, Any]] = []
    support_class_rows: List[Dict[str, Any]] = []
    alignment_rows: List[Dict[str, Any]] = []
    full_class_centers: Dict[int, np.ndarray] = {}
    for class_id in seen_ids:
        ids = np.flatnonzero(labels == int(class_id)).astype(np.int64)
        center, within_var, within_squared_l2 = _class_center(mu, ids)
        full_class_centers[int(class_id)] = center
        class_center_rows.append(
            {
                "model_seed": model_seed,
                "cell_id": str(cache["metadata"].get("cell_id", "")),
                "class_id": int(class_id),
                "sample_count": int(ids.size),
                "within_center_squared_l2": within_squared_l2,
                "within_dimension_variance": within_var,
                "center_norm": float(np.linalg.norm(center)),
            }
        )
    all_seen_set = set(int(x) for x in seen_ids)
    for fold_index, query_list in enumerate(manifest["query_folds"]):
        query_global_ids = np.asarray(sorted(int(x) for x in query_list), dtype=np.int64)
        support_global_ids = np.asarray(sorted(all_seen_set.difference(int(x) for x in query_global_ids)), dtype=np.int64)
        support_indices = np.asarray([class_to_local[int(x)] for x in support_global_ids], dtype=np.int64)
        query_indices = np.asarray([class_to_local[int(x)] for x in query_global_ids], dtype=np.int64)

        support_centers = []
        support_logit_centers = []
        support_visual_centers = []
        support_noise = []
        support_eval_ids: List[int] = []
        for class_id in support_global_ids:
            split = manifest["image_splits"][str(int(class_id))]
            center, within_var, within_squared_l2 = _class_center(mu, split["center_ids"])
            observation_noise = min(
                max(within_var / float(len(split["center_ids"])), args.obs_noise_min),
                args.obs_noise_max,
            )
            support_centers.append(center)
            center_ids = np.asarray(split["center_ids"], dtype=np.int64)
            support_logit_centers.append(classifier_logits[center_ids].mean(axis=0))
            support_visual_centers.append(classifier_visual_repr[center_ids].mean(axis=0))
            support_noise.append(observation_noise)
            support_class_rows.append(
                {
                    "model_seed": model_seed,
                    "cell_id": str(cache["metadata"].get("cell_id", "")),
                    "fold": int(fold_index),
                    "support_class_id": int(class_id),
                    "center_sample_count": int(len(split["center_ids"])),
                    "eval_sample_count": int(len(split["eval_ids"])),
                    "within_center_squared_l2": within_squared_l2,
                    "within_dimension_variance": within_var,
                    "observation_noise": float(observation_noise),
                    "center_norm": float(np.linalg.norm(center)),
                }
            )
            support_eval_ids.extend(int(x) for x in split["eval_ids"])
        support_centers_array = np.stack(support_centers, axis=0)
        support_logit_centers_array = np.stack(support_logit_centers, axis=0)
        support_visual_centers_array = np.stack(support_visual_centers, axis=0)
        support_noise_array = np.asarray(support_noise, dtype=np.float64)

        prompt_relation = _centered_cosine_kernel(support_centers_array, args.eps)
        final_logits_relation = _centered_cosine_kernel(support_logit_centers_array, args.eps)
        final_visual_relation = _centered_cosine_kernel(support_visual_centers_array, args.eps)
        support_semantic_repr = classifier_semantic_repr[support_indices]
        final_semantic_relation = _centered_cosine_kernel(support_semantic_repr, args.eps)
        prompt_final_logits = _relation_alignment(
            prompt_relation, final_logits_relation, args.alignment_topk, args.eps
        )
        prompt_final_visual = _relation_alignment(
            prompt_relation, final_visual_relation, args.alignment_topk, args.eps
        )
        final_logits_visual = _relation_alignment(
            final_logits_relation, final_visual_relation, args.alignment_topk, args.eps
        )
        prompt_final_semantic = _relation_alignment(
            prompt_relation, final_semantic_relation, args.alignment_topk, args.eps
        )
        final_visual_semantic = _relation_alignment(
            final_visual_relation, final_semantic_relation, args.alignment_topk, args.eps
        )
        paired_visual_semantic = float(
            _cosine_rows(support_visual_centers_array, support_semantic_repr, args.eps).mean()
        )
        paired_visual_semantic_centered = float(
            _cosine_rows(
                support_visual_centers_array - support_visual_centers_array.mean(axis=0, keepdims=True),
                support_semantic_repr - support_semantic_repr.mean(axis=0, keepdims=True),
                args.eps,
            ).mean()
        )
        graph_relations = [
            ("method1_diff", "method1_diff", -1, kernels["method1_diff"]),
            ("method2_diff", "method2_diff", -1, kernels["method2_diff"]),
            ("method3_diff", "method3_diff", -1, kernels["method3_diff"]),
            ("method1_nearest_psd", "method1_nearest_psd", -1, kernels["method1_psd"]),
            ("attribute_cosine_gp", "attribute_cosine_gp", -1, kernels["attribute_cosine"]),
        ]
        for shuffle_index, permutation in enumerate(manifest["shuffle_permutations"]):
            permutation_array = np.asarray(permutation, dtype=np.int64)
            graph_relations.append(
                (
                    f"method1_shuffled_{shuffle_index:02d}",
                    "method1_shuffled",
                    int(shuffle_index),
                    kernels["method1_diff"][np.ix_(permutation_array, permutation_array)],
                )
            )
        for method, method_group, replicate, full_relation in graph_relations:
            graph_relation = full_relation[np.ix_(support_indices, support_indices)]
            alignment_row: Dict[str, Any] = {
                "model_seed": model_seed,
                "cell_id": str(cache["metadata"].get("cell_id", "")),
                "fold": int(fold_index),
                "method": method,
                "method_group": method_group,
                "replicate": int(replicate),
                "support_class_count": int(support_global_ids.size),
                "alignment_topk": int(min(args.alignment_topk, support_global_ids.size - 1)),
            }
            alignment_row.update(
                _prefixed_alignment(
                    "graph_prompt",
                    _relation_alignment(graph_relation, prompt_relation, args.alignment_topk, args.eps),
                )
            )
            alignment_row.update(
                _prefixed_alignment(
                    "graph_final_logits",
                    _relation_alignment(graph_relation, final_logits_relation, args.alignment_topk, args.eps),
                )
            )
            alignment_row.update(
                _prefixed_alignment(
                    "graph_final_visual",
                    _relation_alignment(graph_relation, final_visual_relation, args.alignment_topk, args.eps),
                )
            )
            alignment_row.update(
                _prefixed_alignment(
                    "graph_final_semantic",
                    _relation_alignment(graph_relation, final_semantic_relation, args.alignment_topk, args.eps),
                )
            )
            alignment_row.update(_prefixed_alignment("prompt_final_logits", prompt_final_logits))
            alignment_row.update(_prefixed_alignment("prompt_final_visual", prompt_final_visual))
            alignment_row.update(_prefixed_alignment("prompt_final_semantic", prompt_final_semantic))
            alignment_row.update(_prefixed_alignment("final_logits_visual", final_logits_visual))
            alignment_row.update(_prefixed_alignment("final_visual_semantic", final_visual_semantic))
            alignment_row["final_visual_semantic_paired_cosine_mean"] = paired_visual_semantic
            alignment_row[
                "final_visual_semantic_paired_centered_cosine_mean"
            ] = paired_visual_semantic_centered
            alignment_rows.append(alignment_row)

        query_truth = []
        query_sample_ids: List[int] = []
        for class_id in query_global_ids:
            ids = np.flatnonzero(labels == int(class_id)).astype(np.int64)
            query_truth.append(full_class_centers[int(class_id)])
            query_sample_ids.extend(int(x) for x in ids)
        query_truth_array = np.stack(query_truth, axis=0)
        support_mean = support_centers_array.mean(axis=0)

        for method, method_group, replicate, prediction, uncertainty, diagnostics, kernel in _method_predictions(
            args,
            kernels,
            class_attributes,
            seen_ids,
            support_indices,
            query_indices,
            support_centers_array,
            support_noise_array,
            manifest,
        ):
            center_metrics, per_class = _center_metrics(
                prediction, query_truth_array, support_mean, uncertainty, args.eps
            )
            pseudo_metrics = _pseudo_gzsl_metrics(
                mu,
                labels,
                seen_ids,
                support_global_ids,
                query_global_ids,
                support_centers_array,
                prediction,
                support_eval_ids,
                query_sample_ids,
            )
            row: Dict[str, Any] = {
                "model_seed": model_seed,
                "cell_id": str(cache["metadata"].get("cell_id", "")),
                "fold": int(fold_index),
                "method": method,
                "method_group": method_group,
                "replicate": int(replicate),
                "support_class_count": int(support_global_ids.size),
                "query_class_count": int(query_global_ids.size),
                "support_eval_sample_count": int(len(support_eval_ids)),
                "query_sample_count": int(len(query_sample_ids)),
            }
            row.update(center_metrics)
            row.update(pseudo_metrics)
            row.update(diagnostics)
            row.update(_kernel_diagnostics(kernel))
            fold_rows.append(row)
            for class_index, class_id in enumerate(query_global_ids):
                class_row = dict(per_class[class_index])
                class_row.update(
                    {
                        "model_seed": model_seed,
                        "cell_id": str(cache["metadata"].get("cell_id", "")),
                        "fold": int(fold_index),
                        "method": method,
                        "method_group": method_group,
                        "replicate": int(replicate),
                        "query_class_id": int(class_id),
                    }
                )
                class_rows.append(class_row)
        print(f"[stage2] completed fold {fold_index + 1}/{len(manifest['query_folds'])}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "fold_results.csv", fold_rows)
    _write_csv(args.output_dir / "query_class_results.csv", class_rows)
    _write_csv(args.output_dir / "class_center_stats.csv", class_center_rows)
    _write_csv(args.output_dir / "support_class_stats.csv", support_class_rows)
    _write_csv(args.output_dir / "cross_space_alignment.csv", alignment_rows)
    metadata = {
        "format": "graph_gp_stage2_results_v1",
        "cache": str(args.cache),
        "manifest": str(args.manifest),
        "graph_bundle": str(args.graph_bundle),
        "model_seed": model_seed,
        "cell_id": str(cache["metadata"].get("cell_id", "")),
        "fold_count": int(len(manifest["query_folds"])),
        "fold_result_count": int(len(fold_rows)),
        "query_class_result_count": int(len(class_rows)),
        "class_center_stat_count": int(len(class_center_rows)),
        "support_class_stat_count": int(len(support_class_rows)),
        "cross_space_alignment_count": int(len(alignment_rows)),
        "metrics": list(METRIC_KEYS),
        "alignment_metrics": list(ALIGNMENT_METRIC_KEYS),
        "parameters": {
            "gp_ridge": float(args.gp_ridge),
            "obs_noise_min": float(args.obs_noise_min),
            "obs_noise_max": float(args.obs_noise_max),
            "knn_k": int(args.knn_k),
            "knn_tau": float(args.knn_tau),
            "attribute_ridge": float(args.attribute_ridge),
            "alignment_topk": int(args.alignment_topk),
        },
    }
    _write_json(
        args.output_dir / "results.json",
        {"metadata": metadata, "fold_rows": fold_rows, "alignment_rows": alignment_rows},
    )
    print(f"wrote {args.output_dir / 'fold_results.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'query_class_results.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'class_center_stats.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'support_class_stats.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'cross_space_alignment.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'results.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 Graph-GP held-out center transfer.")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-seed", type=int, default=2027)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--center-ratio", type=float, default=0.8)
    parser.add_argument("--shuffle-count", type=int, default=10)
    parser.add_argument("--gp-ridge", type=float, default=1e-4)
    parser.add_argument("--obs-noise-min", type=float, default=1e-4)
    parser.add_argument("--obs-noise-max", type=float, default=1.0)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-tau", type=float, default=0.1)
    parser.add_argument("--attribute-ridge", type=float, default=1e-3)
    parser.add_argument("--alignment-topk", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.graph_bundle = args.graph_bundle.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.cache.is_file():
        parser.error(f"Posterior cache does not exist: {args.cache}")
    if not args.graph_bundle.is_file():
        parser.error(f"Graph bundle does not exist: {args.graph_bundle}")
    if args.fold_seed < 0 or args.num_folds < 2 or args.shuffle_count <= 0:
        parser.error("fold-seed must be non-negative, num-folds >= 2, and shuffle-count positive.")
    if not 0.0 < args.center_ratio < 1.0:
        parser.error("center-ratio must be in (0,1).")
    if min(args.gp_ridge, args.obs_noise_min, args.knn_tau, args.attribute_ridge, args.eps) <= 0.0:
        parser.error("ridge/noise/tau/eps values must be positive.")
    if args.obs_noise_max < args.obs_noise_min or args.knn_k <= 0 or args.alignment_topk <= 0:
        parser.error("obs-noise-max must be >= min and knn-k/alignment-topk must be positive.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
