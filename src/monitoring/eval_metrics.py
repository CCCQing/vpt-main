from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def _as_scores(scores: Any) -> np.ndarray:
    value = np.asarray(scores, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("scores must be a non-empty [samples, classes] matrix")
    if not np.isfinite(value).all():
        raise ValueError("scores contain non-finite values")
    return value


def _as_targets(targets: Any, samples: int, classes: int) -> np.ndarray:
    value = np.asarray(targets, dtype=np.int64).reshape(-1)
    if value.shape[0] != int(samples):
        raise ValueError("targets length does not match scores")
    if value.size and (int(value.min()) < 0 or int(value.max()) >= int(classes)):
        raise ValueError("targets contain indices outside the candidate class space")
    return value


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _pairwise_euclidean_upper(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.asarray([], dtype=np.float32)
    squared_norm = np.square(matrix).sum(axis=1)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (matrix @ matrix.T)
    upper = np.triu_indices(matrix.shape[0], k=1)
    return np.sqrt(np.maximum(squared_distance[upper], 0.0))


def _per_class_accuracy(predictions: np.ndarray, targets: np.ndarray, class_ids: Iterable[int]) -> float:
    values = []
    for class_id in class_ids:
        mask = targets == int(class_id)
        if mask.any():
            values.append(float((predictions[mask] == targets[mask]).mean()))
    return float(np.mean(values)) if values else 0.0


def _gini_nonnegative(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0 or float(values.sum()) <= 0.0:
        return 0.0
    values = np.sort(np.maximum(values, 0.0))
    count = values.size
    index = np.arange(1, count + 1, dtype=np.float32)
    return float((2.0 * np.sum(index * values) / (count * values.sum())) - (count + 1.0) / count)


def _effective_rank(values: np.ndarray) -> Tuple[float, float]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        return 0.0, 0.0
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    total = float(singular.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    prob = singular / total
    entropy = -float(np.sum(prob * np.log(np.maximum(prob, 1e-12))))
    return float(np.exp(entropy)), float(singular[0] / total)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float32)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return 0.0
    left_rank = _rankdata(left[valid])
    right_rank = _rankdata(right[valid])
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denom = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return float(np.dot(left_rank, right_rank) / denom) if denom > 1e-12 else 0.0


def classification_metrics(scores: Any, targets: Any) -> Dict[str, float]:
    score_matrix = _as_scores(scores)
    target_ids = _as_targets(targets, score_matrix.shape[0], score_matrix.shape[1])
    predictions = score_matrix.argmax(axis=1)
    top1 = float((predictions == target_ids).mean())
    k = min(5, score_matrix.shape[1])
    topk = np.argpartition(score_matrix, kth=score_matrix.shape[1] - k, axis=1)[:, -k:]
    top5 = float(np.any(topk == target_ids[:, None], axis=1).mean())
    probabilities = _softmax(score_matrix)
    nll = float(-np.log(np.maximum(probabilities[np.arange(target_ids.size), target_ids], 1e-12)).mean())
    per_class = _per_class_accuracy(predictions, target_ids, np.unique(target_ids))
    return {"top1": top1, "top5": top5, "nll": nll, "per_class": per_class}


def prediction_health_metrics(
    scores: Any,
    targets: Any,
    candidate_global_ids: Sequence[int],
    seen_global_ids: Sequence[int],
) -> Dict[str, float]:
    score_matrix = _as_scores(scores)
    target_ids = _as_targets(targets, score_matrix.shape[0], score_matrix.shape[1])
    candidate = np.asarray(candidate_global_ids, dtype=np.int64).reshape(-1)
    if candidate.size != score_matrix.shape[1]:
        raise ValueError("candidate_global_ids length does not match score columns")
    seen_set = {int(item) for item in seen_global_ids}
    seen_columns = np.asarray([int(item) in seen_set for item in candidate], dtype=bool)
    unseen_columns = ~seen_columns
    if not seen_columns.any() or not unseen_columns.any():
        return {}

    probabilities = _softmax(score_matrix)
    predictions = score_matrix.argmax(axis=1)
    correct = predictions == target_ids
    true_scores = score_matrix[np.arange(target_ids.size), target_ids]
    competitor = score_matrix.copy()
    competitor[np.arange(target_ids.size), target_ids] = -np.inf
    true_margin = true_scores - competitor.max(axis=1)
    true_rank = 1 + np.sum(score_matrix > true_scores[:, None], axis=1)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    confidence = probabilities.max(axis=1)
    predicted_seen = seen_columns[predictions]
    target_seen = seen_columns[target_ids]
    seen_max = score_matrix[:, seen_columns].max(axis=1)
    unseen_max = score_matrix[:, unseen_columns].max(axis=1)

    def mean_or_zero(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else 0.0

    return {
        "seen_unseen_logit_margin_mean": float((seen_max - unseen_max).mean()),
        "seen_probability_mass_mean": float(probabilities[:, seen_columns].sum(axis=1).mean()),
        "wrong_domain_prediction_rate": float((predicted_seen != target_seen).mean()),
        "true_class_margin_mean": float(true_margin.mean()),
        "true_class_rank_mean": float(true_rank.mean()),
        "entropy_mean": float(entropy.mean()),
        "confidence_incorrect": mean_or_zero(confidence[~correct]),
    }


def class_error_metrics(
    scores: Any,
    targets: Any,
    candidate_global_ids: Sequence[int],
    *,
    class_names: Optional[Sequence[str]] = None,
    class_attributes: Optional[Any] = None,
    top_confusions: int = 10,
) -> Dict[str, Any]:
    score_matrix = _as_scores(scores)
    target_ids = _as_targets(targets, score_matrix.shape[0], score_matrix.shape[1])
    candidate = np.asarray(candidate_global_ids, dtype=np.int64).reshape(-1)
    if candidate.size != score_matrix.shape[1]:
        raise ValueError("candidate_global_ids length does not match score columns")
    predictions = score_matrix.argmax(axis=1)
    true_scores = score_matrix[np.arange(target_ids.size), target_ids]
    competitor = score_matrix.copy()
    competitor[np.arange(target_ids.size), target_ids] = -np.inf
    margins = true_scores - competitor.max(axis=1)

    class_count = score_matrix.shape[1]
    support = np.bincount(target_ids, minlength=class_count).astype(np.int64)
    correct_count = np.bincount(target_ids[predictions == target_ids], minlength=class_count).astype(np.int64)
    predicted_frequency = np.bincount(predictions, minlength=class_count).astype(np.int64)
    accuracy = np.divide(
        correct_count,
        support,
        out=np.full(class_count, np.nan, dtype=np.float32),
        where=support > 0,
    )
    margin = np.full(class_count, np.nan, dtype=np.float32)
    for class_id in range(class_count):
        mask = target_ids == class_id
        if mask.any():
            margin[class_id] = float(margins[mask].mean())

    confusion_counts: Dict[Tuple[int, int], int] = {}
    wrong = predictions != target_ids
    for true_id, pred_id in zip(target_ids[wrong].tolist(), predictions[wrong].tolist()):
        key = (int(true_id), int(pred_id))
        confusion_counts[key] = confusion_counts.get(key, 0) + 1
    attr_norm = None
    if class_attributes is not None:
        attributes = np.asarray(class_attributes, dtype=np.float32)
        candidate_max = int(candidate.max()) if candidate.size else -1
        if attributes.ndim == 2 and candidate_max < attributes.shape[0]:
            attr_norm = _normalize_rows(attributes[candidate])
    names = list(class_names) if class_names is not None else None
    confusions = []
    for (true_id, pred_id), count in sorted(confusion_counts.items(), key=lambda item: (-item[1], item[0]))[: max(0, int(top_confusions))]:
        row = {
            "true_global_id": int(candidate[true_id]),
            "pred_global_id": int(candidate[pred_id]),
            "count": int(count),
        }
        if names is not None and len(names) > max(int(candidate[true_id]), int(candidate[pred_id])):
            row["true_class_name"] = str(names[int(candidate[true_id])])
            row["pred_class_name"] = str(names[int(candidate[pred_id])])
        if attr_norm is not None:
            row["semantic_similarity"] = float(np.dot(attr_norm[true_id], attr_norm[pred_id]))
        confusions.append(row)

    valid_accuracy = accuracy[np.isfinite(accuracy)]
    bottom_count = max(1, int(math.ceil(max(1, valid_accuracy.size) * 0.1)))
    sortable_accuracy = accuracy.copy()
    sortable_accuracy[~np.isfinite(sortable_accuracy)] = np.inf
    worst_local = np.argsort(sortable_accuracy)[:bottom_count]
    summary = {
        "max_prediction_share": float((predicted_frequency.max() if predicted_frequency.size else 0) / max(1, predicted_frequency.sum())),
        "bottom_k_class_mean": float(np.nanmean(accuracy[worst_local])) if worst_local.size else 0.0,
    }
    arrays = {
        "candidate_global_ids": candidate,
        "per_class_accuracy": accuracy,
        "class_support": support,
        "class_true_margin": margin,
        "predicted_class_frequency": predicted_frequency,
    }
    return {"summary": summary, "arrays": arrays, "top_confusion_pairs": confusions}


def calibration_profile_metrics(
    seen_scores: Any,
    seen_targets: Any,
    unseen_scores: Any,
    unseen_targets: Any,
    candidate_global_ids: Sequence[int],
    seen_global_ids: Sequence[int],
    gamma_grid: Sequence[float],
) -> Dict[str, Any]:
    seen_matrix = _as_scores(seen_scores)
    unseen_matrix = _as_scores(unseen_scores)
    if seen_matrix.shape[1] != unseen_matrix.shape[1]:
        raise ValueError("seen and unseen scores use different candidate spaces")
    seen_target_ids = _as_targets(seen_targets, seen_matrix.shape[0], seen_matrix.shape[1])
    unseen_target_ids = _as_targets(unseen_targets, unseen_matrix.shape[0], unseen_matrix.shape[1])
    candidate = np.asarray(candidate_global_ids, dtype=np.int64).reshape(-1)
    if candidate.size != seen_matrix.shape[1]:
        raise ValueError("candidate_global_ids length does not match score columns")
    seen_set = {int(item) for item in seen_global_ids}
    seen_columns = np.asarray([int(item) in seen_set for item in candidate], dtype=bool)
    seen_class_local = np.flatnonzero(seen_columns)
    unseen_class_local = np.flatnonzero(~seen_columns)
    gamma = np.asarray(list(gamma_grid), dtype=np.float32).reshape(-1)
    if gamma.size == 0 or not np.isfinite(gamma).all():
        raise ValueError("gamma_grid must contain finite values")

    seen_curve = []
    unseen_curve = []
    h_curve = []
    for value in gamma:
        calibrated_seen = seen_matrix.copy()
        calibrated_unseen = unseen_matrix.copy()
        calibrated_seen[:, seen_columns] -= float(value)
        calibrated_unseen[:, seen_columns] -= float(value)
        s = _per_class_accuracy(calibrated_seen.argmax(axis=1), seen_target_ids, seen_class_local)
        u = _per_class_accuracy(calibrated_unseen.argmax(axis=1), unseen_target_ids, unseen_class_local)
        h = 0.0 if (s + u) <= 0.0 else float(2.0 * s * u / (s + u + 1e-12))
        seen_curve.append(s)
        unseen_curve.append(u)
        h_curve.append(h)

    seen_values = np.asarray(seen_curve, dtype=np.float32)
    unseen_values = np.asarray(unseen_curve, dtype=np.float32)
    h_values = np.asarray(h_curve, dtype=np.float32)
    peak_index = int(np.argmax(h_values))
    raw_index = int(np.argmin(np.abs(gamma)))
    order = np.argsort(seen_values)
    ausuc = float(abs(np.trapz(unseen_values[order], seen_values[order]))) if gamma.size > 1 else 0.0
    summary = {
        "oracle_peak_gamma": float(gamma[peak_index]),
        "ausuc": ausuc,
        "raw_to_oracle_gain": float(h_values[peak_index] - h_values[raw_index]),
    }
    return {
        "summary": summary,
        "gamma_grid": gamma,
        "seen_at_gamma": seen_values,
        "unseen_at_gamma": unseen_values,
    }


def class_geometry_trace_metrics(features: Any, labels: Any) -> Dict[str, float]:
    """Class-balanced trace scatter metrics in the input feature space.

    Each observed class contributes equally.  Both within- and between-class
    scatter use the full squared L2 distance (the covariance trace), so their
    ratio is dimensionally consistent.
    """
    matrix = np.asarray(features, dtype=np.float64)
    label_ids = np.asarray(labels, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return {}
    if label_ids.size != matrix.shape[0]:
        raise ValueError("labels length does not match features")
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    label_ids = label_ids[finite]
    if matrix.shape[0] == 0:
        return {}
    centers = []
    within_by_class = []
    for class_id in np.unique(label_ids):
        values = matrix[label_ids == class_id]
        center = values.mean(axis=0)
        centers.append(center)
        within_by_class.append(float(np.square(values - center).sum(axis=1).mean()))
    centers_matrix = np.asarray(centers, dtype=np.float64)
    macro_center = centers_matrix.mean(axis=0)
    within = float(np.mean(within_by_class))
    between = float(np.square(centers_matrix - macro_center).sum(axis=1).mean())
    return {
        "within_class_scatter_trace": within,
        "between_class_scatter_trace": between,
        "fisher_trace_ratio": float(between / max(within, 1e-12)),
    }


def representation_geometry_metrics(features: Any, labels: Optional[Any] = None) -> Dict[str, float]:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return {}
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    if matrix.shape[0] == 0:
        return {}
    norms = np.linalg.norm(matrix, axis=1)
    normalized = _normalize_rows(matrix)
    sample = normalized[: min(512, normalized.shape[0])]
    pairwise = sample @ sample.T
    offdiag = pairwise[~np.eye(pairwise.shape[0], dtype=bool)] if pairwise.shape[0] > 1 else np.asarray([], dtype=np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    effective_rank, top_ratio = _effective_rank(centered)
    result = {
        "feature_norm_mean": float(norms.mean()),
        "feature_norm_std": float(norms.std()),
        "pairwise_cosine_mean": float(offdiag.mean()) if offdiag.size else 0.0,
        "pairwise_cosine_std": float(offdiag.std()) if offdiag.size else 0.0,
        "effective_rank": effective_rank,
        "top_singular_value_ratio": top_ratio,
    }
    if labels is None:
        return result
    label_ids = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_ids.size != finite.size:
        raise ValueError("labels length does not match features")
    result.update(class_geometry_trace_metrics(matrix, label_ids[finite]))
    return result


def visual_semantic_alignment_metrics(
    visual_features: Any,
    semantic_prototypes: Any,
    targets: Any,
    *,
    recall_k: int = 5,
    ambiguity_margin: float = 0.05,
) -> Dict[str, float]:
    visual = np.asarray(visual_features, dtype=np.float32)
    semantic = np.asarray(semantic_prototypes, dtype=np.float32)
    if visual.ndim != 2 or semantic.ndim != 2 or visual.shape[1] != semantic.shape[1]:
        return {}
    target_ids = _as_targets(targets, visual.shape[0], semantic.shape[0])
    similarity = _normalize_rows(visual) @ _normalize_rows(semantic).T
    true_similarity = similarity[np.arange(target_ids.size), target_ids]
    wrong = similarity.copy()
    wrong[np.arange(target_ids.size), target_ids] = -np.inf
    hard_negative = wrong.max(axis=1)
    margin = true_similarity - hard_negative
    rank = 1 + np.sum(similarity > true_similarity[:, None], axis=1)
    k = min(max(1, int(recall_k)), semantic.shape[0])

    class_ids = np.unique(target_ids)
    visual_centers = []
    aligned_semantics = []
    for class_id in class_ids:
        visual_centers.append(visual[target_ids == class_id].mean(axis=0))
        aligned_semantics.append(semantic[int(class_id)])
    visual_centers = np.asarray(visual_centers, dtype=np.float32)
    aligned_semantics = np.asarray(aligned_semantics, dtype=np.float32)
    center_cosine = np.sum(_normalize_rows(visual_centers) * _normalize_rows(aligned_semantics), axis=1)
    visual_interclass_distance = _pairwise_euclidean_upper(visual_centers)
    semantic_interclass_distance = _pairwise_euclidean_upper(aligned_semantics)
    distance_structure = spearman_correlation(visual_interclass_distance, semantic_interclass_distance)
    if class_ids.size > 1:
        visual_relation = _normalize_rows(visual_centers) @ _normalize_rows(visual_centers).T
        semantic_relation = _normalize_rows(aligned_semantics) @ _normalize_rows(aligned_semantics).T
        upper = np.triu_indices(class_ids.size, k=1)
        structure = spearman_correlation(visual_relation[upper], semantic_relation[upper])
        neighbor_k = min(k, class_ids.size - 1)
        overlap = []
        for row in range(class_ids.size):
            v_order = np.argsort(-visual_relation[row], kind="mergesort")
            s_order = np.argsort(-semantic_relation[row], kind="mergesort")
            v_neigh = [int(item) for item in v_order if int(item) != row][:neighbor_k]
            s_neigh = [int(item) for item in s_order if int(item) != row][:neighbor_k]
            overlap.append(len(set(v_neigh).intersection(s_neigh)) / max(1, neighbor_k))
        neighbor_preservation = float(np.mean(overlap))
    else:
        structure = 0.0
        neighbor_preservation = 0.0
    return {
        "true_prototype_similarity": float(true_similarity.mean()),
        "hard_negative_similarity": float(hard_negative.mean()),
        "semantic_margin": float(margin.mean()),
        "true_prototype_rank": float(rank.mean()),
        "prototype_recall_at_k": float((rank <= k).mean()),
        "class_center_prototype_cosine": float(center_cosine.mean()) if center_cosine.size else 0.0,
        "visual_semantic_structure_spearman": structure,
        "neighbor_preservation_at_k": neighbor_preservation,
        "visual_interclass_distance_mean": float(visual_interclass_distance.mean()) if visual_interclass_distance.size else 0.0,
        "visual_interclass_distance_std": float(visual_interclass_distance.std()) if visual_interclass_distance.size else 0.0,
        "semantic_interclass_distance_mean": float(semantic_interclass_distance.mean()) if semantic_interclass_distance.size else 0.0,
        "semantic_interclass_distance_std": float(semantic_interclass_distance.std()) if semantic_interclass_distance.size else 0.0,
        "visual_semantic_distance_spearman": distance_structure,
        "semantic_ambiguity_rate": float((margin < float(ambiguity_margin)).mean()),
    }


def semantic_graph_reference_metrics(
    class_attributes: Any,
    seen_global_ids: Sequence[int],
    unseen_global_ids: Sequence[int],
    *,
    edge_threshold: float = 0.5,
    neighbor_k: int = 5,
    temperature: float = 1.0,
) -> Dict[str, float]:
    attributes = np.asarray(class_attributes, dtype=np.float32)
    if attributes.ndim != 2 or attributes.shape[0] < 2:
        return {}
    relation = _normalize_rows(attributes) @ _normalize_rows(attributes).T
    count = relation.shape[0]
    np.fill_diagonal(relation, 0.0)
    adjacency = relation >= float(edge_threshold)
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)
    degree = adjacency.sum(axis=1).astype(np.float32)

    visited = np.zeros(count, dtype=bool)
    components = 0
    for start in range(count):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

    logits = relation / max(float(temperature), 1e-8)
    logits[np.eye(count, dtype=bool)] = -np.inf
    row_max = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits - row_max)
    weights[np.eye(count, dtype=bool)] = 0.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    entropy /= max(math.log(max(2, count - 1)), 1e-12)
    sorted_mass = np.sort(weights, axis=1)[:, ::-1]
    k = min(max(1, int(neighbor_k)), count - 1)
    neighbor_relation = relation.copy()
    np.fill_diagonal(neighbor_relation, -np.inf)
    top_neighbors = np.argsort(-neighbor_relation, axis=1)[:, :k]
    mutual = []
    for node in range(count):
        for neighbor in top_neighbors[node]:
            mutual.append(float(node in top_neighbors[int(neighbor)]))

    weighted_adjacency = np.maximum(relation, 0.0)
    np.fill_diagonal(weighted_adjacency, 0.0)
    laplacian = np.diag(weighted_adjacency.sum(axis=1)) - weighted_adjacency
    eigenvalues = np.linalg.eigvalsh(0.5 * (laplacian + laplacian.T))
    effective_rank, _ = _effective_rank(weighted_adjacency)
    seen = np.asarray(sorted({int(item) for item in seen_global_ids}), dtype=np.int64)
    unseen = np.asarray(sorted({int(item) for item in unseen_global_ids}), dtype=np.int64)
    seen = seen[(seen >= 0) & (seen < count)]
    unseen = unseen[(unseen >= 0) & (unseen < count)]
    cross_mass = float(weights[np.ix_(unseen, seen)].sum() / max(1, unseen.size)) if seen.size and unseen.size else 0.0
    unseen_ratios = []
    if seen.size and unseen.size:
        for class_id in unseen:
            nearest_seen_values = relation[int(class_id), seen]
            nearest_seen = float(nearest_seen_values.max()) if nearest_seen_values.size else 0.0
            other_unseen = unseen[unseen != class_id]
            nearest_unseen = float(relation[int(class_id), other_unseen].max()) if other_unseen.size else 0.0
            unseen_ratios.append(nearest_seen / max(abs(nearest_unseen), 1e-12))
    possible_edges = count * (count - 1)
    return {
        "graph_density": float(adjacency.sum() / max(1, possible_edges)),
        "connected_component_count": float(components),
        "degree_mean": float(degree.mean()),
        "degree_std": float(degree.std()),
        "degree_gini": _gini_nonnegative(degree),
        "neighbor_entropy": float(entropy.mean()),
        "top1_neighbor_mass": float(sorted_mass[:, :1].sum(axis=1).mean()),
        "top5_neighbor_mass": float(sorted_mass[:, : min(5, sorted_mass.shape[1])].sum(axis=1).mean()),
        "mutual_knn_ratio": float(np.mean(mutual)) if mutual else 0.0,
        "laplacian_spectral_gap": float(eigenvalues[1]) if eigenvalues.size > 1 else 0.0,
        "graph_effective_rank": effective_rank,
        "seen_unseen_edge_mass": cross_mass,
        "unseen_nearest_seen_ratio": float(np.mean(unseen_ratios)) if unseen_ratios else 0.0,
    }


def semantic_visual_graph_metrics(
    visual_features: Any,
    semantic_prototypes: Any,
    targets: Any,
    *,
    logits: Optional[Any] = None,
    neighbor_k: int = 5,
    high_semantic_threshold: float = 0.8,
    low_visual_threshold: float = 0.2,
) -> Dict[str, float]:
    visual = np.asarray(visual_features, dtype=np.float32)
    semantic = np.asarray(semantic_prototypes, dtype=np.float32)
    if visual.ndim != 2 or semantic.ndim != 2 or visual.shape[1] != semantic.shape[1]:
        return {}
    target_ids = _as_targets(targets, visual.shape[0], semantic.shape[0])
    observed = np.unique(target_ids)
    if observed.size < 2:
        return {}
    centers = np.asarray([visual[target_ids == class_id].mean(axis=0) for class_id in observed], dtype=np.float32)
    semantic_observed = semantic[observed]
    visual_relation = _normalize_rows(centers) @ _normalize_rows(centers).T
    semantic_relation = _normalize_rows(semantic_observed) @ _normalize_rows(semantic_observed).T
    upper = np.triu_indices(observed.size, k=1)
    semantic_edges = semantic_relation[upper]
    visual_edges = visual_relation[upper]
    high_mask = semantic_edges >= float(high_semantic_threshold)
    false_high = high_mask & (visual_edges <= float(low_visual_threshold))
    neighbor_ranking = []
    for row in range(observed.size):
        mask = np.arange(observed.size) != row
        neighbor_ranking.append(
            spearman_correlation(
                visual_relation[row, mask], semantic_relation[row, mask]
            )
        )
    result = {
        "semantic_visual_graph_spearman": spearman_correlation(semantic_edges, visual_edges),
        "false_high_semantic_edge_rate": float(false_high.sum() / max(1, high_mask.sum())),
        "neighbor_ranking_consistency": float(np.mean(neighbor_ranking))
        if neighbor_ranking
        else 0.0,
    }
    if logits is None:
        return result
    score_matrix = _as_scores(logits)
    if score_matrix.shape[0] != target_ids.size or score_matrix.shape[1] != semantic.shape[0]:
        return result
    predictions = score_matrix.argmax(axis=1)
    wrong = predictions != target_ids
    k = min(max(1, int(neighbor_k)), semantic.shape[0] - 1)
    full_semantic_relation = _normalize_rows(semantic) @ _normalize_rows(semantic).T
    np.fill_diagonal(full_semantic_relation, -np.inf)
    neighbors = np.argsort(-full_semantic_relation, axis=1)[:, :k]
    if wrong.any():
        covered = [int(prediction) in set(neighbors[int(target)]) for target, prediction in zip(target_ids[wrong], predictions[wrong])]
        result["hard_negative_coverage"] = float(np.mean(covered)) if covered else 0.0
    else:
        result["hard_negative_coverage"] = 1.0
    high_neighbor_pairs = {
        (int(class_id), int(neighbor))
        for class_id in range(semantic.shape[0])
        for neighbor in neighbors[class_id]
    }
    confused_pairs = [(int(target), int(prediction)) for target, prediction in zip(target_ids[wrong], predictions[wrong])]
    result["confusion_edge_precision"] = float(
        np.mean([pair in high_neighbor_pairs for pair in confused_pairs])
    ) if confused_pairs else 1.0
    return result
