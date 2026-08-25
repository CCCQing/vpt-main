#!/usr/bin/env python3
"""Checkpoint-only geometry tests for prepass CLS -> Distributor mean.

The helpers operate on CPU NumPy arrays and avoid materialising an N x N
distance matrix.  They are intentionally independent from the model so the
same fixed sample/class pairs can be reused for trained and random controls.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from .logit_geometry import analyze_vector_geometry


ABSOLUTE_GEOMETRY_METRICS = (
    "within_class_scatter_trace",
    "between_class_scatter_trace",
    "fisher_trace_ratio",
    "same_minus_interclass_cosine_gap",
    "leave_one_out_center_accuracy",
    "nearest_class_center_cosine_margin_mean",
    "effective_rank",
)


def _as_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty [N,D] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _standardize(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    return centered / np.maximum(scale, eps)


def _l2_normalize(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), eps
    )


def _prepare(values: np.ndarray) -> np.ndarray:
    return _l2_normalize(_standardize(values))


def absolute_class_geometry(
    values: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    matrix = _prepare(_as_matrix(values, "geometry_values"))
    label_ids = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_ids.size != matrix.shape[0]:
        raise ValueError("geometry label count does not match samples")
    classes = np.unique(label_ids)
    metrics, arrays, validity = analyze_vector_geometry(
        matrix,
        label_ids,
        expected_classes=classes,
    )
    metrics = dict(metrics)
    class_ids = np.asarray(arrays["class_ids"], dtype=np.int64)
    class_support = np.asarray(arrays["class_support"], dtype=np.int64)
    class_loo_accuracy = np.asarray(
        arrays["leave_one_out_center_accuracy"], dtype=np.float64
    )
    class_center_margin = np.asarray(
        arrays["nearest_class_center_cosine_margin_mean"], dtype=np.float64
    )
    metrics["leave_one_out_center_accuracy"] = float(
        np.nanmean(class_loo_accuracy)
    )
    metrics["nearest_class_center_cosine_margin_mean"] = float(
        np.nanmean(class_center_margin)
    )
    validity = dict(validity)
    validity["macro_class_metrics_finite"] = bool(
        np.isfinite(metrics["leave_one_out_center_accuracy"])
        and np.isfinite(metrics["nearest_class_center_cosine_margin_mean"])
    )
    validity["valid"] = bool(all(validity.values()))
    return {
        "metrics": metrics,
        "validity": validity,
        "preprocessing": "featurewise_standardize_then_row_l2_normalize",
        "class_weighting": "macro_equal_observed_classes",
        "scatter_definition": "full_squared_l2_trace_on_preprocessed_vectors",
        "loo_definition": "macro_equal_class_leave_one_out_nearest_center",
        "class_support": {
            "class_ids": [int(value) for value in class_ids.tolist()],
            "counts": [int(value) for value in class_support.tolist()],
            "min_count": int(class_support.min()) if class_support.size else 0,
            "max_count": int(class_support.max()) if class_support.size else 0,
            "all_classes_have_leave_one_out_support": bool(
                class_support.size > 0 and np.all(class_support >= 2)
            ),
        },
    }


def summarize_geometry_controls(
    controls: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    if not controls:
        raise ValueError("at least one geometry control is required")
    summary: Dict[str, Dict[str, float]] = {}
    for name in ABSOLUTE_GEOMETRY_METRICS:
        values = np.asarray(
            [float(item["metrics"][name]) for item in controls],
            dtype=np.float64,
        )
        summary[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.size),
        }
    return summary


def geometry_deltas(
    target: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, float]:
    return {
        name: float(target["metrics"][name]) - float(reference["metrics"][name])
        for name in ABSOLUTE_GEOMETRY_METRICS
    }


def geometry_deltas_from_summary(
    target: Mapping[str, Any],
    reference_summary: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    return {
        name: float(target["metrics"][name])
        - float(reference_summary[name]["mean"])
        for name in ABSOLUTE_GEOMETRY_METRICS
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    left_std = left_rank.std()
    right_std = right_rank.std()
    if left_std <= 1.0e-12 or right_std <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def fixed_pair_indices(
    sample_count: int,
    *,
    seed: int,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sample_count = int(sample_count)
    if sample_count < 2 or int(max_pairs) < 1:
        raise ValueError("fixed pairs require at least two samples and one pair")
    total = sample_count * (sample_count - 1) // 2
    wanted = min(int(max_pairs), total)
    if wanted == total and total <= 250_000:
        return np.triu_indices(sample_count, k=1)
    rng = np.random.RandomState(int(seed))
    pairs = set()
    while len(pairs) < wanted:
        left = rng.randint(0, sample_count, size=min(wanted * 2, 200_000))
        right = rng.randint(0, sample_count, size=left.size)
        for first, second in zip(left.tolist(), right.tolist()):
            if first == second:
                continue
            pairs.add((min(first, second), max(first, second)))
            if len(pairs) >= wanted:
                break
    ordered = np.asarray(sorted(pairs), dtype=np.int64)
    return ordered[:, 0], ordered[:, 1]


def pair_distance_spearman(
    source: np.ndarray,
    target: np.ndarray,
    pair_indices: Tuple[np.ndarray, np.ndarray],
) -> float:
    source = _prepare(_as_matrix(source, "source"))
    target = _prepare(_as_matrix(target, "target"))
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target sample counts differ")
    left, right = pair_indices
    source_distance = np.linalg.norm(source[left] - source[right], axis=1)
    target_distance = np.linalg.norm(target[left] - target[right], axis=1)
    return _spearman(source_distance, target_distance)


def class_center_distance_spearman(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, int]:
    source = _prepare(_as_matrix(source, "source"))
    target = _prepare(_as_matrix(target, "target"))
    labels = np.asarray(labels).reshape(-1)
    if source.shape[0] != target.shape[0] or labels.size != source.shape[0]:
        raise ValueError("source, target and label sample counts differ")
    classes = np.unique(labels)
    source_centers = np.stack([source[labels == cls].mean(axis=0) for cls in classes])
    target_centers = np.stack([target[labels == cls].mean(axis=0) for cls in classes])
    indices = np.triu_indices(classes.size, k=1)
    if indices[0].size < 2:
        return 0.0, int(classes.size)
    source_distance = np.linalg.norm(
        source_centers[indices[0]] - source_centers[indices[1]], axis=1
    )
    target_distance = np.linalg.norm(
        target_centers[indices[0]] - target_centers[indices[1]], axis=1
    )
    return _spearman(source_distance, target_distance), int(classes.size)


def linear_cka(source: np.ndarray, target: np.ndarray) -> float:
    source = _standardize(_as_matrix(source, "source"))
    target = _standardize(_as_matrix(target, "target"))
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target sample counts differ")
    cross = source.T @ target
    source_cov = source.T @ source
    target_cov = target.T @ target
    numerator = float(np.square(cross).sum())
    denominator = float(
        np.sqrt(np.square(source_cov).sum() * np.square(target_cov).sum())
    )
    return 0.0 if denominator <= 1.0e-12 else numerator / denominator


def source_retention_metrics(
    prepass_cls: np.ndarray,
    transformed: np.ndarray,
    labels: np.ndarray,
    *,
    pair_seed: int,
    max_pairs: int,
    pair_indices: Tuple[np.ndarray, np.ndarray] | None = None,
) -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
    prepass_cls = _as_matrix(prepass_cls, "prepass_cls")
    transformed = _as_matrix(transformed, "transformed")
    if pair_indices is None:
        pair_indices = fixed_pair_indices(
            prepass_cls.shape[0], seed=pair_seed, max_pairs=max_pairs
        )
    class_spearman, class_count = class_center_distance_spearman(
        prepass_cls, transformed, labels
    )
    metrics = {
        "sample_pair_distance_spearman": pair_distance_spearman(
            prepass_cls, transformed, pair_indices
        ),
        "class_center_distance_spearman": class_spearman,
        "linear_cka": linear_cka(prepass_cls, transformed),
        "sample_count": int(prepass_cls.shape[0]),
        "class_count": int(class_count),
        "pair_count": int(pair_indices[0].size),
        "preprocessing": "featurewise_standardize_then_row_l2_for_distance; centered_standardized_linear_CKA",
    }
    return metrics, pair_indices


def compare_to_random_controls(
    trained: Mapping[str, float],
    controls: Sequence[Mapping[str, float]],
) -> Dict[str, Any]:
    if not controls:
        raise ValueError("at least one random control is required")
    result: Dict[str, Any] = {}
    for name in (
        "sample_pair_distance_spearman",
        "class_center_distance_spearman",
        "linear_cka",
    ):
        values = np.asarray([float(item[name]) for item in controls], dtype=np.float64)
        trained_value = float(trained[name])
        result[name] = {
            "trained": trained_value,
            "random_mean": float(values.mean()),
            "random_min": float(values.min()),
            "random_max": float(values.max()),
            "trained_minus_random_mean": float(trained_value - values.mean()),
            "trained_above_all_random_controls": bool(trained_value > values.max()),
        }
    return result
