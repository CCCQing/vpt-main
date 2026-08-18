import itertools
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


COMPONENTS = ("domain_bias", "within_domain_scale", "class_pattern")
SPLITS = ("test_seen", "test_unseen")


def harmonic_mean(seen: float, unseen: float) -> float:
    denom = float(seen) + float(unseen)
    return 0.0 if denom <= 0.0 else float(2.0 * seen * unseen / denom)


def _as_logits(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError("logits must be a non-empty [samples, classes] matrix")
    if not np.isfinite(array).all():
        raise ValueError("logits contain non-finite values")
    return array


def _as_targets(value, sample_count: int, class_count: int) -> np.ndarray:
    targets = np.asarray(value, dtype=np.int64).reshape(-1)
    if targets.size != int(sample_count):
        raise ValueError("target count does not match logits")
    if targets.size and (targets.min() < 0 or targets.max() >= int(class_count)):
        raise ValueError("targets are outside the candidate class space")
    return targets


def _domain_indices(
    candidate_class_ids: Sequence[int],
    seen_class_ids: Sequence[int],
    unseen_class_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = [int(item) for item in candidate_class_ids]
    seen = {int(item) for item in seen_class_ids}
    unseen = {int(item) for item in unseen_class_ids}
    if seen.intersection(unseen):
        raise ValueError("seen and unseen class identities overlap")
    seen_indices = np.asarray(
        [index for index, class_id in enumerate(candidates) if class_id in seen],
        dtype=np.int64,
    )
    unseen_indices = np.asarray(
        [index for index, class_id in enumerate(candidates) if class_id in unseen],
        dtype=np.int64,
    )
    if seen_indices.size == 0 or unseen_indices.size == 0:
        raise ValueError("both seen and unseen candidates are required")
    if seen_indices.size + unseen_indices.size != len(candidates):
        raise ValueError("candidate class space is not exactly partitioned into seen/unseen")
    return seen_indices, unseen_indices


def factorize_logits(
    logits,
    seen_indices: Sequence[int],
    unseen_indices: Sequence[int],
    eps: float = 1.0e-8,
) -> Dict[str, np.ndarray]:
    logits = _as_logits(logits)
    if float(eps) <= 0.0:
        raise ValueError("factorization eps must be positive")
    centered = logits - logits.mean(axis=1, keepdims=True)
    bias_by_class = np.zeros_like(centered)
    scale_by_class = np.zeros_like(centered)
    class_pattern = np.zeros_like(centered)
    domain_bias = np.zeros((centered.shape[0], 2), dtype=np.float64)
    domain_scale = np.zeros((centered.shape[0], 2), dtype=np.float64)
    for domain_index, indices in enumerate(
        (np.asarray(seen_indices, dtype=np.int64), np.asarray(unseen_indices, dtype=np.int64))
    ):
        if indices.size == 0:
            raise ValueError("each logit domain must contain at least one class")
        values = centered[:, indices]
        bias = values.mean(axis=1, keepdims=True)
        residual = values - bias
        raw_scale = np.sqrt(np.mean(np.square(residual), axis=1, keepdims=True))
        safe_scale = np.maximum(raw_scale, float(eps))
        pattern = residual / safe_scale
        bias_by_class[:, indices] = bias
        scale_by_class[:, indices] = safe_scale
        class_pattern[:, indices] = pattern
        domain_bias[:, domain_index] = bias[:, 0]
        domain_scale[:, domain_index] = safe_scale[:, 0]
    reconstructed = bias_by_class + scale_by_class * class_pattern
    return {
        "centered": centered,
        "bias_by_class": bias_by_class,
        "scale_by_class": scale_by_class,
        "class_pattern": class_pattern,
        "domain_bias": domain_bias,
        "domain_scale": domain_scale,
        "reconstructed": reconstructed,
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - centered))),
    }


def hybrid_logits(
    reference: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    target_components: Sequence[str],
) -> np.ndarray:
    selected = set(str(item) for item in target_components)
    unknown = selected.difference(COMPONENTS)
    if unknown:
        raise ValueError("unknown decomposition components: {}".format(sorted(unknown)))
    bias = target["bias_by_class"] if "domain_bias" in selected else reference["bias_by_class"]
    scale = target["scale_by_class"] if "within_domain_scale" in selected else reference["scale_by_class"]
    pattern = target["class_pattern"] if "class_pattern" in selected else reference["class_pattern"]
    return np.asarray(bias) + np.asarray(scale) * np.asarray(pattern)


def coalition_name(components: Sequence[str]) -> str:
    selected = set(str(item) for item in components)
    if not selected:
        return "reference"
    if selected == set(COMPONENTS):
        return "target"
    return "+".join(item for item in COMPONENTS if item in selected)


def _per_class_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    expected_classes: Sequence[int],
) -> Tuple[float, np.ndarray, np.ndarray]:
    values = []
    support = []
    for class_index in np.asarray(expected_classes, dtype=np.int64):
        mask = targets == int(class_index)
        count = int(mask.sum())
        support.append(count)
        values.append(float(np.mean(predictions[mask] == targets[mask])) if count else np.nan)
    per_class = np.asarray(values, dtype=np.float64)
    if not np.isfinite(per_class).all():
        raise ValueError("full-test decomposition requires every expected class to have support")
    return float(per_class.mean()), per_class, np.asarray(support, dtype=np.int64)


def _per_class_mean(
    values: np.ndarray,
    targets: np.ndarray,
    expected_classes: Sequence[int],
) -> np.ndarray:
    result = []
    for class_index in np.asarray(expected_classes, dtype=np.int64):
        mask = targets == int(class_index)
        result.append(float(np.mean(values[mask])) if np.any(mask) else np.nan)
    array = np.asarray(result, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("full-test decomposition requires every expected class to have support")
    return array


def _split_metrics(
    logits,
    targets,
    expected_classes: Sequence[int],
    seen_indices: Sequence[int],
) -> Dict[str, object]:
    logits = _as_logits(logits)
    targets = _as_targets(targets, logits.shape[0], logits.shape[1])
    predictions = logits.argmax(axis=1).astype(np.int64)
    per_class, class_accuracy, class_support = _per_class_accuracy(
        predictions, targets, expected_classes
    )
    true_logits = logits[np.arange(targets.size), targets]
    other = logits.copy()
    other[np.arange(targets.size), targets] = -np.inf
    true_margin = true_logits - other.max(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    seen_probability_mass = probabilities[:, np.asarray(seen_indices, dtype=np.int64)].sum(axis=1)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-30)), axis=1)
    class_true_margin = _per_class_mean(true_margin, targets, expected_classes)
    prediction_frequency = np.bincount(
        predictions, minlength=logits.shape[1]
    ).astype(np.float64)
    prediction_frequency /= float(targets.size)
    return {
        "sample_count": int(targets.size),
        "sample_accuracy": float(np.mean(predictions == targets)),
        "per_class_accuracy": per_class,
        "true_margin_mean": float(true_margin.mean()),
        "seen_probability_mass_mean": float(seen_probability_mass.mean()),
        "entropy_mean": float(entropy.mean()),
        "predictions": predictions,
        "class_accuracy": class_accuracy,
        "class_support": class_support,
        "class_true_margin": class_true_margin,
        "prediction_frequency": prediction_frequency,
    }


def evaluate_gzsl(
    logits_by_split: Mapping[str, np.ndarray],
    targets_by_split: Mapping[str, np.ndarray],
    seen_indices: Sequence[int],
    unseen_indices: Sequence[int],
) -> Dict[str, object]:
    seen_metrics = _split_metrics(
        logits_by_split["test_seen"],
        targets_by_split["test_seen"],
        seen_indices,
        seen_indices,
    )
    unseen_metrics = _split_metrics(
        logits_by_split["test_unseen"],
        targets_by_split["test_unseen"],
        unseen_indices,
        seen_indices,
    )
    seen = float(seen_metrics["per_class_accuracy"])
    unseen = float(unseen_metrics["per_class_accuracy"])
    return {
        "seen_accuracy": seen,
        "unseen_accuracy": unseen,
        "h": harmonic_mean(seen, unseen),
        "splits": {"test_seen": seen_metrics, "test_unseen": unseen_metrics},
    }


def _public_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        "seen_accuracy": float(metrics["seen_accuracy"]),
        "unseen_accuracy": float(metrics["unseen_accuracy"]),
        "h": float(metrics["h"]),
        "splits": {
            split: {
                key: float(value) if isinstance(value, (float, np.floating)) else int(value)
                for key, value in metrics["splits"][split].items()
                if key not in {
                    "predictions",
                    "class_accuracy",
                    "class_support",
                    "class_true_margin",
                    "prediction_frequency",
                    "targets",
                }
            }
            for split in SPLITS
        },
    }


def _seen_unseen_h_shapley(reference: Mapping[str, object], target: Mapping[str, object]):
    s0, u0 = float(reference["seen_accuracy"]), float(reference["unseen_accuracy"])
    s1, u1 = float(target["seen_accuracy"]), float(target["unseen_accuracy"])
    seen_contribution = 0.5 * (
        harmonic_mean(s1, u0) - harmonic_mean(s0, u0)
        + harmonic_mean(s1, u1) - harmonic_mean(s0, u1)
    )
    unseen_contribution = 0.5 * (
        harmonic_mean(s0, u1) - harmonic_mean(s0, u0)
        + harmonic_mean(s1, u1) - harmonic_mean(s1, u0)
    )
    delta_h = harmonic_mean(s1, u1) - harmonic_mean(s0, u0)
    return {
        "seen_accuracy_contribution_to_h": float(seen_contribution),
        "unseen_accuracy_contribution_to_h": float(unseen_contribution),
        "delta_h": float(delta_h),
        "additivity_error": float(seen_contribution + unseen_contribution - delta_h),
    }


def _shapley_values(coalitions: Mapping[frozenset, Mapping[str, object]], metric: str):
    count = len(COMPONENTS)
    result = {}
    all_components = set(COMPONENTS)
    for component in COMPONENTS:
        contribution = 0.0
        others = [item for item in COMPONENTS if item != component]
        for size in range(len(others) + 1):
            for subset_tuple in itertools.combinations(others, size):
                subset = frozenset(subset_tuple)
                weight = (
                    math.factorial(size)
                    * math.factorial(count - size - 1)
                    / math.factorial(count)
                )
                with_component = frozenset(set(subset).union({component}))
                contribution += weight * (
                    float(coalitions[with_component][metric])
                    - float(coalitions[subset][metric])
                )
        result[component] = float(contribution)
    total = float(coalitions[frozenset(all_components)][metric]) - float(
        coalitions[frozenset()][metric]
    )
    return {
        "values": result,
        "total_delta": total,
        "additivity_error": float(sum(result.values()) - total),
    }


def _transition_metrics(
    reference_metrics: Mapping[str, object],
    target_metrics: Mapping[str, object],
    seen_indices: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    seen_set = set(int(item) for item in seen_indices)
    result = {}
    for split in SPLITS:
        reference_predictions = np.asarray(
            reference_metrics["splits"][split]["predictions"], dtype=np.int64
        )
        target_predictions = np.asarray(
            target_metrics["splits"][split]["predictions"], dtype=np.int64
        )
        targets = np.asarray(target_metrics["splits"][split]["targets"], dtype=np.int64)
        reference_correct = reference_predictions == targets
        target_correct = target_predictions == targets
        changed = reference_predictions != target_predictions
        reference_seen = np.asarray([int(item) in seen_set for item in reference_predictions])
        target_seen = np.asarray([int(item) in seen_set for item in target_predictions])
        domain_flip = reference_seen != target_seen
        result[split] = {
            "accuracy_delta": float(target_correct.mean() - reference_correct.mean()),
            "correction_rate": float(np.mean(~reference_correct & target_correct)),
            "regression_rate": float(np.mean(reference_correct & ~target_correct)),
            "agreement_rate": float(np.mean(~changed)),
            "prediction_flip_rate": float(np.mean(changed)),
            "domain_flip_rate": float(np.mean(domain_flip)),
            "within_domain_class_flip_rate": float(np.mean(changed & ~domain_flip)),
        }
    return result


def analyze_decision_gain(
    reference_splits: Mapping[str, Mapping[str, object]],
    target_splits: Mapping[str, Mapping[str, object]],
    candidate_class_ids: Sequence[int],
    seen_class_ids: Sequence[int],
    unseen_class_ids: Sequence[int],
    *,
    eps: float = 1.0e-8,
    reconstruction_atol: float = 1.0e-5,
    global_scale_factors: Sequence[float] = (0.5, 2.0, 4.0),
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    seen_indices, unseen_indices = _domain_indices(
        candidate_class_ids, seen_class_ids, unseen_class_ids
    )
    targets_by_split = {}
    factors = {"reference": {}, "target": {}}
    reconstruction_checks = {}
    for split in SPLITS:
        reference_logits = _as_logits(reference_splits[split]["logits"])
        target_logits = _as_logits(target_splits[split]["logits"])
        if reference_logits.shape != target_logits.shape:
            raise ValueError("paired logits shapes differ for {}".format(split))
        reference_targets = _as_targets(
            reference_splits[split]["targets_local"],
            reference_logits.shape[0],
            reference_logits.shape[1],
        )
        target_targets = _as_targets(
            target_splits[split]["targets_local"],
            target_logits.shape[0],
            target_logits.shape[1],
        )
        if not np.array_equal(reference_targets, target_targets):
            raise ValueError("paired targets differ for {}".format(split))
        if tuple(reference_splits[split].get("sample_ids", ())) != tuple(
            target_splits[split].get("sample_ids", ())
        ):
            raise ValueError("paired sample order differs for {}".format(split))
        targets_by_split[split] = reference_targets
        factors["reference"][split] = factorize_logits(
            reference_logits, seen_indices, unseen_indices, eps=eps
        )
        factors["target"][split] = factorize_logits(
            target_logits, seen_indices, unseen_indices, eps=eps
        )
        reconstruction_checks[split] = {
            method: float(factors[method][split]["reconstruction_max_abs_error"])
            for method in ("reference", "target")
        }

    coalition_internal = {}
    coalition_public = {}
    for size in range(len(COMPONENTS) + 1):
        for subset_tuple in itertools.combinations(COMPONENTS, size):
            subset = frozenset(subset_tuple)
            logits_by_split = {
                split: hybrid_logits(
                    factors["reference"][split], factors["target"][split], subset
                )
                for split in SPLITS
            }
            metrics = evaluate_gzsl(
                logits_by_split, targets_by_split, seen_indices, unseen_indices
            )
            for split in SPLITS:
                metrics["splits"][split]["targets"] = targets_by_split[split]
            coalition_internal[subset] = metrics
            coalition_public[coalition_name(subset)] = _public_metrics(metrics)

    reference_metrics = coalition_internal[frozenset()]
    target_metrics = coalition_internal[frozenset(COMPONENTS)]
    original_reference_metrics = evaluate_gzsl(
        {split: reference_splits[split]["logits"] for split in SPLITS},
        targets_by_split,
        seen_indices,
        unseen_indices,
    )
    original_target_metrics = evaluate_gzsl(
        {split: target_splits[split]["logits"] for split in SPLITS},
        targets_by_split,
        seen_indices,
        unseen_indices,
    )
    for metrics in (original_reference_metrics, original_target_metrics):
        for split in SPLITS:
            metrics["splits"][split]["targets"] = targets_by_split[split]
    endpoint_delta = {
        metric: float(target_metrics[metric]) - float(reference_metrics[metric])
        for metric in ("seen_accuracy", "unseen_accuracy", "h")
    }
    logit_shapley = {
        metric: _shapley_values(coalition_internal, metric)
        for metric in ("seen_accuracy", "unseen_accuracy", "h")
    }
    seen_unseen_h = _seen_unseen_h_shapley(reference_metrics, target_metrics)
    transitions = _transition_metrics(
        original_reference_metrics, original_target_metrics, seen_indices
    )

    scale_checks = {}
    target_centered = {
        split: factors["target"][split]["centered"] for split in SPLITS
    }
    target_public = _public_metrics(original_target_metrics)
    for factor in global_scale_factors:
        factor = float(factor)
        if factor <= 0.0:
            raise ValueError("global scale factors must be positive")
        scaled = evaluate_gzsl(
            {split: target_centered[split] * factor for split in SPLITS},
            targets_by_split,
            seen_indices,
            unseen_indices,
        )
        scale_checks[str(factor)] = {
            "seen_accuracy_delta": float(scaled["seen_accuracy"] - target_metrics["seen_accuracy"]),
            "unseen_accuracy_delta": float(scaled["unseen_accuracy"] - target_metrics["unseen_accuracy"]),
            "h_delta": float(scaled["h"] - target_metrics["h"]),
            "prediction_flip_count": int(
                sum(
                    np.count_nonzero(
                        scaled["splits"][split]["predictions"]
                        != original_target_metrics["splits"][split]["predictions"]
                    )
                    for split in SPLITS
                )
            ),
        }

    reconstruction_pass = all(
        error <= float(reconstruction_atol)
        for split in reconstruction_checks.values()
        for error in split.values()
    )
    endpoint_equivalence = {}
    for name, reconstructed, original in (
        ("reference", reference_metrics, original_reference_metrics),
        ("target", target_metrics, original_target_metrics),
    ):
        endpoint_equivalence[name] = {
            "prediction_flip_count": int(
                sum(
                    np.count_nonzero(
                        reconstructed["splits"][split]["predictions"]
                        != original["splits"][split]["predictions"]
                    )
                    for split in SPLITS
                )
            ),
            "h_abs_diff": float(abs(reconstructed["h"] - original["h"])),
        }
    endpoint_equivalence_pass = all(
        item["prediction_flip_count"] == 0 and item["h_abs_diff"] <= 1.0e-12
        for item in endpoint_equivalence.values()
    )
    scale_pass = all(
        item["prediction_flip_count"] == 0 and abs(item["h_delta"]) <= 1.0e-12
        for item in scale_checks.values()
    )
    shapley_pass = all(
        abs(payload["additivity_error"]) <= 1.0e-10
        for payload in logit_shapley.values()
    ) and abs(seen_unseen_h["additivity_error"]) <= 1.0e-10
    validity = {
        "reconstruction_pass": bool(reconstruction_pass),
        "endpoint_factorization_equivalence_pass": bool(endpoint_equivalence_pass),
        "global_positive_scale_invariance_pass": bool(scale_pass),
        "shapley_additivity_pass": bool(shapley_pass),
        "full_test_class_support_pass": True,
        "valid": bool(
            reconstruction_pass
            and endpoint_equivalence_pass
            and scale_pass
            and shapley_pass
        ),
        "failure_reasons": [],
    }
    if not reconstruction_pass:
        validity["failure_reasons"].append("factorization_reconstruction_failed")
    if not endpoint_equivalence_pass:
        validity["failure_reasons"].append("factorization_changed_endpoint_predictions")
    if not scale_pass:
        validity["failure_reasons"].append("global_positive_scale_changed_prediction_or_h")
    if not shapley_pass:
        validity["failure_reasons"].append("shapley_additivity_failed")

    class_count = len(candidate_class_ids)
    arrays = {
        "candidate_class_ids": np.asarray(candidate_class_ids, dtype=np.int64),
        "seen_class_mask": np.asarray(
            [int(class_id) in set(int(item) for item in seen_class_ids) for class_id in candidate_class_ids],
            dtype=np.bool_,
        ),
    }
    for split, expected_indices in (
        ("test_seen", seen_indices), ("test_unseen", unseen_indices)
    ):
        reference_accuracy = np.full(class_count, np.nan, dtype=np.float64)
        target_accuracy = np.full(class_count, np.nan, dtype=np.float64)
        support = np.zeros(class_count, dtype=np.int64)
        reference_margin = np.full(class_count, np.nan, dtype=np.float64)
        target_margin = np.full(class_count, np.nan, dtype=np.float64)
        reference_accuracy[expected_indices] = original_reference_metrics["splits"][split]["class_accuracy"]
        target_accuracy[expected_indices] = original_target_metrics["splits"][split]["class_accuracy"]
        reference_margin[expected_indices] = original_reference_metrics["splits"][split]["class_true_margin"]
        target_margin[expected_indices] = original_target_metrics["splits"][split]["class_true_margin"]
        support[expected_indices] = original_target_metrics["splits"][split]["class_support"]
        arrays[f"{split}_reference_class_accuracy"] = reference_accuracy
        arrays[f"{split}_target_class_accuracy"] = target_accuracy
        arrays[f"{split}_class_accuracy_delta"] = target_accuracy - reference_accuracy
        arrays[f"{split}_reference_class_true_margin"] = reference_margin
        arrays[f"{split}_target_class_true_margin"] = target_margin
        arrays[f"{split}_class_true_margin_delta"] = target_margin - reference_margin
        arrays[f"{split}_reference_prediction_frequency"] = original_reference_metrics["splits"][split]["prediction_frequency"]
        arrays[f"{split}_target_prediction_frequency"] = original_target_metrics["splits"][split]["prediction_frequency"]
        arrays[f"{split}_prediction_frequency_delta"] = (
            original_target_metrics["splits"][split]["prediction_frequency"]
            - original_reference_metrics["splits"][split]["prediction_frequency"]
        )
        arrays[f"{split}_class_support"] = support

    summary = {
        "format": "decision_gain_decomposition_summary_v1",
        "components": list(COMPONENTS),
        "factorization": {
            "formula": "center(logits)=domain_bias+within_domain_scale*class_pattern",
            "eps": float(eps),
            "reconstruction_atol": float(reconstruction_atol),
            "reconstruction_max_abs_error": reconstruction_checks,
            "endpoint_equivalence": endpoint_equivalence,
            "interpretation": "decision-space counterfactual decomposition, not training-mechanism causal share",
        },
        "endpoints": {
            "reference": _public_metrics(original_reference_metrics),
            "target": target_public,
            "target_minus_reference": endpoint_delta,
        },
        "seen_unseen_h_decomposition": seen_unseen_h,
        "coalitions": coalition_public,
        "logit_component_shapley": logit_shapley,
        "endpoint_prediction_transitions": transitions,
        "global_positive_scale_negative_control": scale_checks,
        "validity": validity,
    }
    return summary, arrays
