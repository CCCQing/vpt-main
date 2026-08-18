"""Full-test geometry diagnostics for final-checkpoint GZSL logits."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .decision_gain_decomposition import _domain_indices, factorize_logits


SPLITS = ("test_seen", "test_unseen")
VIEWS = ("centered_logits", "direction_normalized_logits", "class_pattern")


def _as_logits(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("logits must be a non-empty [samples, classes] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("logits contain non-finite values")
    return matrix


def _as_targets(value, sample_count: int, class_count: int) -> np.ndarray:
    targets = np.asarray(value, dtype=np.int64).reshape(-1)
    if targets.size != int(sample_count):
        raise ValueError("target count does not match logits")
    if targets.size and (targets.min() < 0 or targets.max() >= int(class_count)):
        raise ValueError("targets are outside the candidate class space")
    return targets


def _normalize_rows(values: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    return values / np.maximum(norms[:, None], float(eps)), norms


def _spectral_metrics(values: np.ndarray, eps: float) -> Dict[str, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues = np.maximum(np.linalg.eigvalsh(0.5 * (covariance + covariance.T)), 0.0)
    singular = np.sqrt(eigenvalues)[::-1]
    total = float(singular.sum())
    if total <= float(eps):
        return {"effective_rank": 0.0, "top_singular_value_ratio": 0.0}
    probability = singular / total
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, float(eps)))))
    return {
        "effective_rank": float(np.exp(entropy)),
        "top_singular_value_ratio": float(singular[0] / total),
    }


def _class_geometry(
    values: np.ndarray,
    targets: np.ndarray,
    expected_classes: Sequence[int],
    eps: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, object]]:
    class_ids = np.asarray(expected_classes, dtype=np.int64).reshape(-1)
    if class_ids.size < 2:
        raise ValueError("logit geometry requires at least two expected classes")
    centers = []
    support = []
    within_trace = []
    within_cosine = []
    normalized, norms = _normalize_rows(values, eps)
    for class_id in class_ids:
        mask = targets == int(class_id)
        class_values = values[mask]
        class_normalized = normalized[mask]
        count = int(mask.sum())
        support.append(count)
        if count == 0:
            raise ValueError("full-test logit geometry requires every expected class to have support")
        center = class_values.mean(axis=0)
        centers.append(center)
        within_trace.append(float(np.square(class_values - center).sum(axis=1).mean()))
        if count >= 2:
            pair_sum = float(np.square(class_normalized.sum(axis=0)).sum() - count)
            within_cosine.append(pair_sum / float(count * (count - 1)))
        else:
            within_cosine.append(np.nan)
    support_array = np.asarray(support, dtype=np.int64)
    centers_matrix = np.asarray(centers, dtype=np.float64)
    macro_center = centers_matrix.mean(axis=0)
    within = float(np.mean(within_trace))
    between_by_class = np.square(centers_matrix - macro_center).sum(axis=1)
    between = float(between_by_class.mean())

    normalized_centers, center_norms = _normalize_rows(centers_matrix, eps)
    center_cosine = normalized_centers @ normalized_centers.T
    upper = center_cosine[np.triu_indices(class_ids.size, k=1)]
    interclass_center_cosine = float(upper.mean()) if upper.size else 0.0
    valid_within = np.asarray(within_cosine, dtype=np.float64)
    valid_within = valid_within[np.isfinite(valid_within)]
    within_class_cosine = float(valid_within.mean()) if valid_within.size else 0.0

    class_position = {int(class_id): index for index, class_id in enumerate(class_ids)}
    target_position = np.asarray([class_position[int(item)] for item in targets], dtype=np.int64)
    center_similarity = normalized @ normalized_centers.T
    wrong_similarity = center_similarity.copy()
    wrong_similarity[np.arange(targets.size), target_position] = -np.inf
    wrong_max = wrong_similarity.max(axis=1)
    true_loo_similarity = np.full(targets.size, np.nan, dtype=np.float64)
    for position, class_id in enumerate(class_ids):
        sample_indices = np.flatnonzero(targets == int(class_id))
        count = int(sample_indices.size)
        if count < 2:
            continue
        leave_one_out = (centers_matrix[position] * count - values[sample_indices]) / float(count - 1)
        leave_one_out, _ = _normalize_rows(leave_one_out, eps)
        true_loo_similarity[sample_indices] = np.sum(
            normalized[sample_indices] * leave_one_out, axis=1
        )
    valid_loo = np.isfinite(true_loo_similarity)
    loo_margin = true_loo_similarity[valid_loo] - wrong_max[valid_loo]
    loo_correct = true_loo_similarity[valid_loo] > wrong_max[valid_loo]
    class_loo_margin = []
    class_loo_accuracy = []
    for class_id in class_ids:
        mask = (targets == int(class_id)) & valid_loo
        class_loo_margin.append(float(np.mean(true_loo_similarity[mask] - wrong_max[mask])) if mask.any() else np.nan)
        class_loo_accuracy.append(float(np.mean(true_loo_similarity[mask] > wrong_max[mask])) if mask.any() else np.nan)

    metrics = {
        "sample_count": int(values.shape[0]),
        "observed_class_count": int(class_ids.size),
        "vector_dim": int(values.shape[1]),
        "vector_norm_mean": float(norms.mean()),
        "vector_norm_std": float(norms.std()),
        "within_class_scatter_trace": within,
        "between_class_scatter_trace": between,
        "fisher_trace_ratio": float(between / max(within, float(eps))),
        "within_class_pairwise_cosine_mean": within_class_cosine,
        "interclass_center_cosine_mean": interclass_center_cosine,
        "same_minus_interclass_cosine_gap": float(
            within_class_cosine - interclass_center_cosine
        ),
        "nearest_class_center_cosine_margin_mean": float(loo_margin.mean()) if loo_margin.size else 0.0,
        "leave_one_out_center_accuracy": float(loo_correct.mean()) if loo_correct.size else 0.0,
        "nearest_center_valid_sample_ratio": float(valid_loo.mean()),
        "class_center_norm_mean": float(center_norms.mean()),
        "class_center_norm_std": float(center_norms.std()),
        **_spectral_metrics(values, eps),
    }
    arrays = {
        "class_ids": class_ids,
        "class_support": support_array,
        "within_class_scatter_trace": np.asarray(within_trace, dtype=np.float64),
        "between_class_scatter_trace": np.asarray(between_by_class, dtype=np.float64),
        "within_class_pairwise_cosine_mean": np.asarray(within_cosine, dtype=np.float64),
        "nearest_class_center_cosine_margin_mean": np.asarray(class_loo_margin, dtype=np.float64),
        "leave_one_out_center_accuracy": np.asarray(class_loo_accuracy, dtype=np.float64),
    }
    validity = {
        "all_expected_classes_observed": bool(np.all(support_array > 0)),
        "all_classes_have_leave_one_out_support": bool(np.all(support_array >= 2)),
        "finite_metrics": bool(all(np.isfinite(float(value)) for value in metrics.values())),
    }
    validity["valid"] = bool(all(validity.values()))
    return metrics, arrays, validity


def _factor_context(factorized: Mapping[str, np.ndarray]) -> Dict[str, float]:
    bias = np.asarray(factorized["domain_bias"], dtype=np.float64)
    scale = np.asarray(factorized["domain_scale"], dtype=np.float64)
    gap = bias[:, 1] - bias[:, 0]
    return {
        "unseen_minus_seen_domain_bias_mean": float(gap.mean()),
        "unseen_minus_seen_domain_bias_std": float(gap.std()),
        "seen_within_domain_scale_mean": float(scale[:, 0].mean()),
        "seen_within_domain_scale_std": float(scale[:, 0].std()),
        "unseen_within_domain_scale_mean": float(scale[:, 1].mean()),
        "unseen_within_domain_scale_std": float(scale[:, 1].std()),
        "unseen_to_seen_scale_ratio_of_means": float(
            scale[:, 1].mean() / max(scale[:, 0].mean(), 1.0e-30)
        ),
    }


def analyze_logit_geometry(
    outputs_by_split: Mapping[str, Mapping[str, object]],
    candidate_class_ids: Sequence[int],
    seen_class_ids: Sequence[int],
    unseen_class_ids: Sequence[int],
    *,
    eps: float = 1.0e-8,
    reconstruction_atol: float = 1.0e-5,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Analyze decision-space class geometry without retaining sample logits."""
    if float(eps) <= 0.0 or float(reconstruction_atol) < 0.0:
        raise ValueError("eps must be positive and reconstruction_atol non-negative")
    seen_indices, unseen_indices = _domain_indices(
        candidate_class_ids, seen_class_ids, unseen_class_ids
    )
    expected_by_split = {
        "test_seen": seen_indices,
        "test_unseen": unseen_indices,
    }
    split_inputs = {}
    for split in SPLITS:
        if split not in outputs_by_split:
            raise ValueError("missing logit geometry split: {}".format(split))
        logits = _as_logits(outputs_by_split[split]["logits"])
        targets = _as_targets(
            outputs_by_split[split]["targets_local"], logits.shape[0], logits.shape[1]
        )
        if logits.shape[1] != len(candidate_class_ids):
            raise ValueError("candidate class count does not match logits")
        factorized = factorize_logits(logits, seen_indices, unseen_indices, eps=eps)
        direction, direction_norms = _normalize_rows(factorized["centered"], eps)
        split_inputs[split] = {
            "logits": logits,
            "targets": targets,
            "factorized": factorized,
            "views": {
                "centered_logits": factorized["centered"],
                "direction_normalized_logits": direction,
                "class_pattern": factorized["class_pattern"],
            },
            "zero_direction_count": int(np.sum(direction_norms <= float(eps))),
        }

    summary: Dict[str, object] = {
        "format": "logit_geometry_reference_summary_v1",
        "analysis_unit": "single_method_final_checkpoint_full_test",
        "class_weighting": "macro_equal_observed_classes",
        "scatter_definition": "full_squared_l2_trace",
        "views": {
            "centered_logits": "per-sample class-mean removed; argmax-equivalent to raw logits",
            "direction_normalized_logits": "centered logits divided by per-sample L2 norm; removes global positive scale",
            "class_pattern": "5.4 factorization with seen/unseen domain bias and within-domain scale removed",
        },
        "splits": {},
    }
    arrays: Dict[str, np.ndarray] = {}
    validity_checks = []
    reconstruction_errors = []
    endpoint_equivalence = {}
    for split in SPLITS:
        item = split_inputs[split]
        raw_prediction = item["logits"].argmax(axis=1)
        centered_prediction = item["views"]["centered_logits"].argmax(axis=1)
        direction_prediction = item["views"]["direction_normalized_logits"].argmax(axis=1)
        endpoint_equivalence[split] = {
            "raw_vs_centered_prediction_flip_rate": float(np.mean(raw_prediction != centered_prediction)),
            "raw_vs_direction_normalized_prediction_flip_rate": float(np.mean(raw_prediction != direction_prediction)),
            "zero_direction_count": int(item["zero_direction_count"]),
        }
        reconstruction_errors.append(float(item["factorized"]["reconstruction_max_abs_error"]))
        split_summary = {
            "sample_count": int(item["targets"].size),
            "expected_class_count": int(expected_by_split[split].size),
            "factor_context": _factor_context(item["factorized"]),
            "views": {},
        }
        for view_name in VIEWS:
            metrics, view_arrays, view_validity = _class_geometry(
                item["views"][view_name],
                item["targets"],
                expected_by_split[split],
                eps,
            )
            split_summary["views"][view_name] = {
                **metrics,
                "validity": view_validity,
            }
            validity_checks.append(bool(view_validity["valid"]))
            for name, value in view_arrays.items():
                arrays["{}__{}__{}".format(split, view_name, name)] = value
        summary["splits"][split] = split_summary

    joint_targets = np.concatenate(
        [split_inputs[split]["targets"] for split in SPLITS], axis=0
    )
    joint_summary = {"views": {}}
    all_class_indices = np.arange(len(candidate_class_ids), dtype=np.int64)
    for view_name in VIEWS:
        joint_values = np.concatenate(
            [split_inputs[split]["views"][view_name] for split in SPLITS], axis=0
        )
        metrics, view_arrays, view_validity = _class_geometry(
            joint_values, joint_targets, all_class_indices, eps
        )
        joint_summary["views"][view_name] = {**metrics, "validity": view_validity}
        validity_checks.append(bool(view_validity["valid"]))
        for name, value in view_arrays.items():
            arrays["joint__{}__{}".format(view_name, name)] = value
    summary["joint"] = joint_summary

    max_reconstruction_error = max(reconstruction_errors)
    equivalence_pass = all(
        item["raw_vs_centered_prediction_flip_rate"] == 0.0
        and item["raw_vs_direction_normalized_prediction_flip_rate"] == 0.0
        and item["zero_direction_count"] == 0
        for item in endpoint_equivalence.values()
    )
    validity = {
        "all_view_geometry_valid": bool(all(validity_checks)),
        "factorization_reconstruction_max_abs_error": float(max_reconstruction_error),
        "factorization_reconstruction_pass": bool(
            max_reconstruction_error <= float(reconstruction_atol)
        ),
        "endpoint_equivalence": endpoint_equivalence,
        "centered_and_direction_endpoint_equivalence_pass": bool(equivalence_pass),
    }
    validity["valid"] = bool(
        validity["all_view_geometry_valid"]
        and validity["factorization_reconstruction_pass"]
        and validity["centered_and_direction_endpoint_equivalence_pass"]
    )
    summary["validity"] = validity
    arrays["candidate_class_ids"] = np.asarray(candidate_class_ids, dtype=np.int64)
    arrays["seen_class_indices"] = seen_indices
    arrays["unseen_class_indices"] = unseen_indices
    return summary, arrays
