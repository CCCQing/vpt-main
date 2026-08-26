"""Deterministic checkpoint-only helpers for B-series P0 diagnostics."""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .eval_metrics import spearman_correlation


def normalize_rows(values, eps: float = 1.0e-8) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("expected a non-empty [rows, dim] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    return matrix / np.maximum(norms[:, None], float(eps)), norms


def array_sha256(values) -> str:
    matrix = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(tuple(matrix.shape)).encode("ascii"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def class_centers(
    features,
    labels,
    class_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("features and labels are not aligned")
    centers = []
    support = []
    for class_id in classes:
        mask = target == int(class_id)
        count = int(mask.sum())
        if count == 0:
            raise ValueError("class {} has no visual support".format(int(class_id)))
        centers.append(matrix[mask].mean(axis=0))
        support.append(count)
    result = np.asarray(centers, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("class centers contain non-finite values")
    return result, np.asarray(support, dtype=np.int64)


def cosine_relation(values) -> np.ndarray:
    normalized, _ = normalize_rows(values)
    return normalized @ normalized.T


def relation_validity(values) -> Dict[str, object]:
    relation = cosine_relation(values)
    symmetric = 0.5 * (relation + relation.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    tolerance = 1.0e-7 * max(1.0, float(np.abs(eigenvalues).max()))
    return {
        "shape": list(relation.shape),
        "finite": bool(np.isfinite(relation).all()),
        "symmetry_max_abs_error": float(np.abs(relation - relation.T).max()),
        "diagonal_max_abs_error": float(
            np.abs(np.diag(relation) - 1.0).max()
        ),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "negative_eigenvalue_count": int(np.sum(eigenvalues < -tolerance)),
        "psd_within_tolerance": bool(np.all(eigenvalues >= -tolerance)),
        "interpretation": "cosine relation used as a deterministic smoother, not asserted to be a GP covariance",
        "valid": bool(np.isfinite(relation).all()),
    }


def semantic_support_weights(
    semantic_values,
    seen_class_ids: Sequence[int],
    unseen_class_ids: Sequence[int],
    *,
    method: str,
    permutation: Optional[Sequence[int]] = None,
) -> np.ndarray:
    semantic = np.asarray(semantic_values, dtype=np.float64)
    seen = np.asarray(seen_class_ids, dtype=np.int64)
    unseen = np.asarray(unseen_class_ids, dtype=np.int64)
    if semantic.ndim != 2 or semantic.shape[0] <= int(max(seen.max(), unseen.max())):
        raise ValueError("semantic matrix does not cover the requested class ids")
    seen_semantic = semantic[seen]
    if permutation is not None:
        order = np.asarray(permutation, dtype=np.int64).reshape(-1)
        if sorted(order.tolist()) != list(range(seen.size)):
            raise ValueError("seen semantic permutation must be bijective")
        seen_semantic = seen_semantic[order]
    unseen_normalized, _ = normalize_rows(semantic[unseen])
    seen_normalized, _ = normalize_rows(seen_semantic)
    relation = unseen_normalized @ seen_normalized.T
    method_name = str(method).strip().lower()
    if method_name == "positive_cosine":
        weights = np.maximum(relation, 0.0)
        row_sum = weights.sum(axis=1, keepdims=True)
        zero = row_sum[:, 0] <= 1.0e-12
        if np.any(zero):
            weights[zero] = 1.0
            row_sum = weights.sum(axis=1, keepdims=True)
        return weights / row_sum
    if method_name == "nearest_neighbor":
        weights = np.zeros_like(relation)
        weights[np.arange(unseen.size), np.argmax(relation, axis=1)] = 1.0
        return weights
    if method_name == "global_mean":
        return np.full_like(relation, 1.0 / float(seen.size))
    raise ValueError("unsupported semantic smoother method: {}".format(method))


def predict_unseen_centers(weights, seen_centers) -> np.ndarray:
    coefficient = np.asarray(weights, dtype=np.float64)
    centers = np.asarray(seen_centers, dtype=np.float64)
    if coefficient.ndim != 2 or centers.ndim != 2 or coefficient.shape[1] != centers.shape[0]:
        raise ValueError("support weights and seen centers are not aligned")
    predicted = coefficient @ centers
    if not np.isfinite(predicted).all():
        raise ValueError("predicted centers contain non-finite values")
    return predicted


def _neighbor_overlap(left_relation: np.ndarray, right_relation: np.ndarray, k: int) -> float:
    count = int(left_relation.shape[0])
    if count < 2:
        return 0.0
    use_k = min(max(1, int(k)), count - 1)
    rows = []
    for index in range(count):
        left_order = [
            int(value)
            for value in np.argsort(-left_relation[index], kind="mergesort")
            if int(value) != index
        ][:use_k]
        right_order = [
            int(value)
            for value in np.argsort(-right_relation[index], kind="mergesort")
            if int(value) != index
        ][:use_k]
        rows.append(len(set(left_order).intersection(right_order)) / float(use_k))
    return float(np.mean(rows))


def center_prediction_metrics(
    predicted_centers,
    true_centers,
    test_features,
    test_labels,
    unseen_class_ids: Sequence[int],
    *,
    neighbor_k: int = 5,
    high_edge_threshold: float = 0.8,
    low_edge_threshold: float = 0.2,
) -> Dict[str, object]:
    predicted = np.asarray(predicted_centers, dtype=np.float64)
    truth = np.asarray(true_centers, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("predicted and true centers must have the same shape")
    predicted_norm, predicted_lengths = normalize_rows(predicted)
    truth_norm, truth_lengths = normalize_rows(truth)
    center_cosine = np.sum(predicted_norm * truth_norm, axis=1)
    normalized_error = np.linalg.norm(predicted - truth, axis=1) / np.maximum(
        truth_lengths, 1.0e-8
    )
    predicted_relation = predicted_norm @ predicted_norm.T
    true_relation = truth_norm @ truth_norm.T
    upper = np.triu_indices(predicted.shape[0], k=1)
    predicted_edges = predicted_relation[upper]
    true_edges = true_relation[upper]
    high = predicted_edges >= float(high_edge_threshold)
    false_high = high & (true_edges <= float(low_edge_threshold))

    features, _ = normalize_rows(test_features)
    logits = features @ predicted_norm.T
    class_to_local = {
        int(class_id): index for index, class_id in enumerate(unseen_class_ids)
    }
    local_targets = np.asarray(
        [class_to_local[int(value)] for value in np.asarray(test_labels).reshape(-1)],
        dtype=np.int64,
    )
    predictions = logits.argmax(axis=1)
    true_score = logits[np.arange(local_targets.size), local_targets]
    wrong = logits.copy()
    wrong[np.arange(local_targets.size), local_targets] = -np.inf
    margin = true_score - wrong.max(axis=1)
    stable = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(stable).sum(axis=1)) + logits.max(axis=1)
    nll = -(true_score - logsumexp)
    class_accuracy = []
    for class_id in range(len(unseen_class_ids)):
        mask = local_targets == class_id
        class_accuracy.append(float(np.mean(predictions[mask] == class_id)))
    return {
        "predicted_to_true_center_cosine_mean": float(center_cosine.mean()),
        "predicted_to_true_center_cosine_min": float(center_cosine.min()),
        "normalized_center_error_mean": float(normalized_error.mean()),
        "relation_spearman": float(
            spearman_correlation(predicted_edges, true_edges)
        ),
        "visual_neighbor_recovery_at_k": _neighbor_overlap(
            predicted_relation, true_relation, neighbor_k
        ),
        "false_high_predicted_edge_rate": float(
            false_high.sum() / max(1, high.sum())
        ),
        "unseen_only_top1": float(np.mean(predictions == local_targets)),
        "unseen_only_per_class_top1": float(np.mean(class_accuracy)),
        "unseen_only_margin_mean": float(margin.mean()),
        "unseen_only_nll": float(nll.mean()),
        "predicted_center_coverage": float(
            np.mean(np.isfinite(predicted_lengths) & (predicted_lengths > 1.0e-8))
        ),
        "sample_count": int(local_targets.size),
        "class_count": int(len(unseen_class_ids)),
        "valid": bool(
            np.isfinite(center_cosine).all()
            and np.isfinite(normalized_error).all()
            and np.isfinite(logits).all()
        ),
    }


def visual_centroid_oracle_logits(
    features,
    labels,
    candidate_class_ids: Sequence[int],
    fixed_centers: Mapping[int, np.ndarray],
    *,
    loo_classes: Sequence[int] = (),
) -> Tuple[np.ndarray, Dict[str, object]]:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    candidates = [int(value) for value in candidate_class_ids]
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("features and labels are not aligned")
    centers = np.stack(
        [np.asarray(fixed_centers[class_id], dtype=np.float64) for class_id in candidates],
        axis=0,
    )
    normalized_features, _ = normalize_rows(matrix)
    normalized_centers, _ = normalize_rows(centers)
    logits = normalized_features @ normalized_centers.T
    class_to_local = {class_id: index for index, class_id in enumerate(candidates)}
    loo_set = {int(value) for value in loo_classes}
    loo_applied = 0
    support_min = None
    for class_id in sorted(loo_set):
        sample_indices = np.flatnonzero(target == class_id)
        count = int(sample_indices.size)
        if count < 2:
            raise ValueError(
                "LOO visual centroid requires at least two samples for class {}".format(
                    class_id
                )
            )
        support_min = count if support_min is None else min(support_min, count)
        full_center = matrix[sample_indices].mean(axis=0)
        loo_center = (full_center[None, :] * count - matrix[sample_indices]) / float(
            count - 1
        )
        loo_center, _ = normalize_rows(loo_center)
        logits[sample_indices, class_to_local[class_id]] = np.sum(
            normalized_features[sample_indices] * loo_center,
            axis=1,
        )
        loo_applied += count
    return logits, {
        "oracle": True,
        "deployable": False,
        "loo_class_count": int(len(loo_set)),
        "loo_sample_count": int(loo_applied),
        "minimum_loo_support": int(support_min or 0),
        "finite": bool(np.isfinite(logits).all()),
        "valid": bool(np.isfinite(logits).all()),
    }
