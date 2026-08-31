from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .eval_metrics import (
    classification_metrics,
    prediction_health_metrics,
    semantic_projection_health_metrics,
    semantic_visual_graph_metrics,
    visual_semantic_alignment_metrics,
)
from .prediction_transition import PredictionTransitionAccumulator
from .prompt_source_retention import (
    absolute_class_geometry,
    geometry_deltas,
    source_retention_metrics,
)


REPRESENTATION_NAMES = (
    "frozen_prepass_cls",
    "normal_final_cls",
    "cls_delta",
)
TRANSITION_GROUP_NAMES = (
    "stable_correct",
    "corrected",
    "regressed",
    "stable_wrong",
)


def _matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty [samples, dimensions] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _targets(values: Any, sample_count: int, class_count: int) -> np.ndarray:
    targets = np.asarray(values, dtype=np.int64).reshape(-1)
    if targets.size != int(sample_count):
        raise ValueError("target count does not match samples")
    if targets.size and (
        int(targets.min()) < 0 or int(targets.max()) >= int(class_count)
    ):
        raise ValueError("targets fall outside the candidate class order")
    return targets


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12
    )


def _summary(values: Any) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("distribution summary requires finite non-empty values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def _sample_transport(prepass_cls: np.ndarray, final_cls: np.ndarray) -> Dict[str, np.ndarray]:
    prepass = _matrix(prepass_cls, "prepass_cls")
    final = _matrix(final_cls, "final_cls")
    if prepass.shape != final.shape:
        raise ValueError("prepass CLS and final CLS must have identical shapes")
    delta = final - prepass
    cosine = np.sum(_normalize_rows(prepass) * _normalize_rows(final), axis=1)
    distance = np.linalg.norm(delta, axis=1)
    norm_ratio = distance / np.maximum(np.linalg.norm(prepass, axis=1), 1.0e-12)
    return {
        "prepass_final_cls_cosine": cosine,
        "prepass_final_cls_l2_distance": distance,
        "delta_to_prepass_norm_ratio": norm_ratio,
    }


def representation_transport_metrics(
    prepass_cls: Any,
    final_cls: Any,
    labels: Any,
    *,
    pair_seed: int,
    max_pairs: int,
) -> Dict[str, Any]:
    prepass = _matrix(prepass_cls, "prepass_cls")
    final = _matrix(final_cls, "final_cls")
    if prepass.shape != final.shape:
        raise ValueError("prepass CLS and final CLS must have identical shapes")
    label_ids = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_ids.size != prepass.shape[0]:
        raise ValueError("label count does not match representation samples")
    delta = final - prepass
    absolute = {
        "frozen_prepass_cls": absolute_class_geometry(prepass, label_ids),
        "normal_final_cls": absolute_class_geometry(final, label_ids),
        "cls_delta": absolute_class_geometry(delta, label_ids),
    }
    relationship, _ = source_retention_metrics(
        prepass,
        final,
        label_ids,
        pair_seed=int(pair_seed),
        max_pairs=int(max_pairs),
    )
    sample_metrics = {
        name: _summary(values)
        for name, values in _sample_transport(prepass, final).items()
    }
    validity = {
        "same_shape": True,
        "sample_count_matches_labels": True,
        "all_absolute_geometries_valid": bool(
            all(item["validity"]["valid"] for item in absolute.values())
        ),
        "paired_metrics_finite": bool(
            all(
                np.isfinite(float(value))
                for summary in sample_metrics.values()
                for name, value in summary.items()
                if name != "count"
            )
        ),
        "relationship_metrics_finite": bool(
            all(
                np.isfinite(float(relationship[name]))
                for name in (
                    "sample_pair_distance_spearman",
                    "class_center_distance_spearman",
                    "linear_cka",
                )
            )
        ),
    }
    validity["valid"] = bool(all(validity.values()))
    return {
        "entity_type": "paired_transport",
        "entity_id": "frozen_prepass_cls_to_final_cls",
        "definition": "final_cls_minus_frozen_prepass_cls_on_identical_samples",
        "absolute_geometry": absolute,
        "final_minus_prepass_geometry": geometry_deltas(
            absolute["normal_final_cls"], absolute["frozen_prepass_cls"]
        ),
        "sample_transport": sample_metrics,
        "relationship_metrics": relationship,
        "validity": validity,
    }


def classifier_logits_from_features(
    features: Any,
    semantic_prototypes: Any,
    *,
    score_mode: str,
    logit_scale: float = 1.0,
) -> np.ndarray:
    visual = _matrix(features, "classifier_features")
    semantic = _matrix(semantic_prototypes, "semantic_prototypes")
    if visual.shape[1] != semantic.shape[1]:
        raise ValueError("classifier features and semantic prototypes differ in dimension")
    mode = str(score_mode).lower()
    if mode == "dot":
        logits = visual @ semantic.T
    elif mode == "cosine":
        logits = _normalize_rows(visual) @ _normalize_rows(semantic).T
        logits = logits * float(logit_scale)
    else:
        raise ValueError("score_mode must be dot or cosine")
    if not np.isfinite(logits).all():
        raise ValueError("reconstructed classifier logits contain non-finite values")
    return logits


def _visual_centers(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.unique(targets)
    centers = np.stack(
        [features[targets == class_id].mean(axis=0) for class_id in observed]
    )
    return centers, observed


def _semantic_condition(
    features: np.ndarray,
    semantic_prototypes: np.ndarray,
    targets: np.ndarray,
    logits: Optional[np.ndarray],
) -> Dict[str, Any]:
    alignment = visual_semantic_alignment_metrics(
        features, semantic_prototypes, targets, recall_k=5
    )
    graph = semantic_visual_graph_metrics(
        features,
        semantic_prototypes,
        targets,
        logits=logits,
        neighbor_k=5,
    )
    validity = {
        "alignment_nonempty": bool(alignment),
        "semantic_graph_nonempty": bool(graph),
        "finite": bool(
            alignment
            and graph
            and all(
                np.isfinite(float(value))
                for value in (*alignment.values(), *graph.values())
            )
        ),
    }
    validity["valid"] = bool(all(validity.values()))
    return {
        "semantic_space": "projected_768d_semantic_prototypes",
        "alignment": alignment,
        "semantic_graph": graph,
        "validity": validity,
    }


def _numeric_deltas(
    target: Mapping[str, float], reference: Mapping[str, float]
) -> Dict[str, float]:
    shared = sorted(set(target).intersection(reference))
    return {
        name: float(target[name]) - float(reference[name])
        for name in shared
    }


def semantic_transport_metrics(
    prepass_cls: Any,
    final_cls: Any,
    projected_semantic_prototypes: Any,
    targets: Any,
    *,
    prepass_logits: Optional[Any] = None,
    final_logits: Optional[Any] = None,
    raw_semantic_prototypes: Optional[Any] = None,
) -> Dict[str, Any]:
    prepass = _matrix(prepass_cls, "prepass_cls")
    final = _matrix(final_cls, "final_cls")
    semantic = _matrix(projected_semantic_prototypes, "projected_semantics")
    if prepass.shape != final.shape or prepass.shape[1] != semantic.shape[1]:
        raise ValueError("prepass, final and projected semantic dimensions do not align")
    target_ids = _targets(targets, prepass.shape[0], semantic.shape[0])
    pre_logits = None if prepass_logits is None else _matrix(prepass_logits, "prepass_logits")
    fin_logits = None if final_logits is None else _matrix(final_logits, "final_logits")
    prepass_condition = _semantic_condition(prepass, semantic, target_ids, pre_logits)
    final_condition = _semantic_condition(final, semantic, target_ids, fin_logits)
    alignment_delta = _numeric_deltas(
        final_condition["alignment"], prepass_condition["alignment"]
    )
    graph_delta = _numeric_deltas(
        final_condition["semantic_graph"], prepass_condition["semantic_graph"]
    )
    paired = {
        "final_minus_prepass": {
            "alignment": alignment_delta,
            "semantic_graph": graph_delta,
            "true_prototype_rank_improvement": float(
                prepass_condition["alignment"]["true_prototype_rank"]
                - final_condition["alignment"]["true_prototype_rank"]
            ),
        }
    }
    raw_relation: Dict[str, Any] = {
        "status": "not_requested",
        "reason": "raw_semantic_prototypes_not_provided",
    }
    if raw_semantic_prototypes is not None:
        raw = _matrix(raw_semantic_prototypes, "raw_semantic_prototypes")
        if raw.shape[0] != semantic.shape[0]:
            raise ValueError("raw and projected semantics differ in class count")
        raw_relation = {
            "status": "available",
            "semantic_projection_identity": semantic_projection_health_metrics(
                raw, semantic
            ),
            "frozen_prepass_cls": {},
            "normal_final_cls": {},
        }
        for name, values in (
            ("frozen_prepass_cls", prepass),
            ("normal_final_cls", final),
        ):
            centers, observed = _visual_centers(values, target_ids)
            raw_relation[name] = semantic_projection_health_metrics(
                raw,
                semantic,
                visual_class_centers=centers,
                observed_class_ids=observed,
            )
        visual_relation_fields = (
            "semantic_visual_relation_spearman_312",
            "semantic_visual_relation_spearman_768",
            "semantic_visual_relation_spearman_delta_768_minus_312",
            "semantic_visual_neighbor_overlap_at_5_312",
            "semantic_visual_neighbor_overlap_at_5_768",
            "semantic_visual_neighbor_overlap_at_5_delta_768_minus_312",
            "false_high_semantic_edge_rate_312",
            "false_high_semantic_edge_rate_768",
            "false_high_semantic_edge_rate_delta_768_minus_312",
        )
        raw_relation["final_minus_prepass_visual_relation"] = {
            name: float(raw_relation["normal_final_cls"][name])
            - float(raw_relation["frozen_prepass_cls"][name])
            for name in visual_relation_fields
        }
    validity = {
        "prepass_alignment_valid": bool(prepass_condition["validity"]["valid"]),
        "final_alignment_valid": bool(final_condition["validity"]["valid"]),
        "paired_deltas_finite": bool(
            all(
                np.isfinite(float(value))
                for block in paired["final_minus_prepass"].values()
                for value in (
                    block.values() if isinstance(block, Mapping) else (block,)
                )
            )
        ),
        "raw_relation_available": bool(raw_semantic_prototypes is not None),
    }
    validity["valid"] = bool(all(validity.values()))
    return {
        "entity_type": "paired_transport",
        "entity_id": "frozen_prepass_cls_to_final_cls_semantic_alignment",
        "direct_768d_alignment": {
            "frozen_prepass_cls": prepass_condition,
            "normal_final_cls": final_condition,
        },
        "paired_deltas": paired,
        "raw_312d_vs_projected_768d_relation": raw_relation,
        "validity": validity,
    }


def _true_margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    true_values = logits[np.arange(targets.size), targets]
    wrong = logits.copy()
    wrong[np.arange(targets.size), targets] = -np.inf
    return true_values - wrong.max(axis=1)


def _sample_semantic_values(
    features: np.ndarray,
    semantic: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, np.ndarray]:
    similarity = _normalize_rows(features) @ _normalize_rows(semantic).T
    true_similarity = similarity[np.arange(targets.size), targets]
    wrong = similarity.copy()
    wrong[np.arange(targets.size), targets] = -np.inf
    hard_negative = wrong.max(axis=1)
    return {
        "true_prototype_similarity": true_similarity,
        "hard_negative_similarity": hard_negative,
        "semantic_margin": true_similarity - hard_negative,
        "true_prototype_rank": 1.0
        + np.sum(similarity > true_similarity[:, None], axis=1),
    }


def prediction_transition_group_metrics(
    prepass_cls: Any,
    final_cls: Any,
    reference_logits: Any,
    target_logits: Any,
    targets: Any,
    candidate_global_ids: Sequence[int],
    seen_global_ids: Sequence[int],
    semantic_prototypes: Any,
    *,
    reference_name: str,
    target_name: str,
) -> Dict[str, Any]:
    prepass = _matrix(prepass_cls, "prepass_cls")
    final = _matrix(final_cls, "final_cls")
    reference = _matrix(reference_logits, "reference_logits")
    target = _matrix(target_logits, "target_logits")
    semantic = _matrix(semantic_prototypes, "semantic_prototypes")
    if prepass.shape != final.shape:
        raise ValueError("prepass and final representations differ in shape")
    if reference.shape != target.shape or reference.shape[0] != prepass.shape[0]:
        raise ValueError("paired logits do not match representation samples")
    if semantic.shape != (reference.shape[1], prepass.shape[1]):
        raise ValueError("semantic prototypes do not match logits and CLS dimensions")
    target_ids = _targets(targets, prepass.shape[0], reference.shape[1])
    candidate = np.asarray(candidate_global_ids, dtype=np.int64).reshape(-1)
    if candidate.size != reference.shape[1]:
        raise ValueError("candidate class order does not match logits")
    reference_prediction = reference.argmax(axis=1)
    target_prediction = target.argmax(axis=1)
    reference_correct = reference_prediction == target_ids
    target_correct = target_prediction == target_ids
    masks = {
        "stable_correct": reference_correct & target_correct,
        "corrected": ~reference_correct & target_correct,
        "regressed": reference_correct & ~target_correct,
        "stable_wrong": ~reference_correct & ~target_correct,
    }
    transport = _sample_transport(prepass, final)
    pre_semantic = _sample_semantic_values(prepass, semantic, target_ids)
    final_semantic = _sample_semantic_values(final, semantic, target_ids)
    reference_margin = _true_margin(reference, target_ids)
    target_margin = _true_margin(target, target_ids)
    true_global = candidate[target_ids]
    seen_set = {int(value) for value in seen_global_ids}
    is_seen = np.asarray([int(value) in seen_set for value in true_global])
    groups: Dict[str, Any] = {}
    for name in TRANSITION_GROUP_NAMES:
        mask = masks[name]
        count = int(mask.sum())
        if count == 0:
            groups[name] = {
                "status": "no_samples",
                "sample_count": 0,
                "metrics": None,
                "validity": {"valid": True, "reason": "empty_observed_group"},
            }
            continue
        group_metrics: Dict[str, Any] = {
            metric_name: _summary(values[mask])
            for metric_name, values in transport.items()
        }
        for metric_name in (
            "true_prototype_similarity",
            "hard_negative_similarity",
            "semantic_margin",
            "true_prototype_rank",
        ):
            group_metrics[f"prepass_{metric_name}"] = _summary(
                pre_semantic[metric_name][mask]
            )
            group_metrics[f"final_{metric_name}"] = _summary(
                final_semantic[metric_name][mask]
            )
            direction = (
                pre_semantic[metric_name] - final_semantic[metric_name]
                if metric_name == "true_prototype_rank"
                else final_semantic[metric_name] - pre_semantic[metric_name]
            )
            delta_name = (
                "true_prototype_rank_improvement"
                if metric_name == "true_prototype_rank"
                else f"{metric_name}_delta_final_minus_prepass"
            )
            group_metrics[delta_name] = _summary(direction[mask])
        group_metrics["reference_true_logit_margin"] = _summary(
            reference_margin[mask]
        )
        group_metrics["target_true_logit_margin"] = _summary(target_margin[mask])
        group_metrics["true_logit_margin_delta_target_minus_reference"] = _summary(
            (target_margin - reference_margin)[mask]
        )
        groups[name] = {
            "status": "observed",
            "sample_count": count,
            "class_count": int(np.unique(true_global[mask]).size),
            "seen_sample_count": int(is_seen[mask].sum()),
            "unseen_sample_count": int((~is_seen[mask]).sum()),
            "seen_sample_ratio": float(is_seen[mask].mean()),
            "reference_accuracy": float(reference_correct[mask].mean()),
            "target_accuracy": float(target_correct[mask].mean()),
            "prediction_flip_rate": float(
                (reference_prediction[mask] != target_prediction[mask]).mean()
            ),
            "metrics": group_metrics,
            "validity": {"valid": True},
        }
    transition = PredictionTransitionAccumulator(reference_name, target_name)
    transition.update(reference_prediction, target_prediction, target_ids)
    overall = transition.finalize()
    validity = {
        "four_groups_partition_samples": bool(
            sum(int(mask.sum()) for mask in masks.values()) == prepass.shape[0]
        ),
        "overall_transition_valid": bool(overall["valid"]),
        "all_observed_metrics_finite": bool(
            all(
                np.isfinite(float(value))
                for group in groups.values()
                if group["metrics"] is not None
                for summary in group["metrics"].values()
                for field, value in summary.items()
                if field != "count"
            )
        ),
    }
    validity["valid"] = bool(all(validity.values()))
    return {
        "entity_type": "posthoc_prediction_transition_audit",
        "reference_name": str(reference_name),
        "target_name": str(target_name),
        "group_definition": {
            "stable_correct": "reference_correct_and_target_correct",
            "corrected": "reference_wrong_and_target_correct",
            "regressed": "reference_correct_and_target_wrong",
            "stable_wrong": "reference_wrong_and_target_wrong",
        },
        "interpretation_boundary": "posthoc_descriptive_evidence_not_a_training_target_or_causal_proof",
        "overall_transition": overall,
        "groups": groups,
        "validity": validity,
    }


def _cell_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    candidate_global_ids: Sequence[int],
    seen_global_ids: Sequence[int],
) -> Dict[str, Any]:
    return {
        "classification": classification_metrics(logits, targets),
        "prediction_health": prediction_health_metrics(
            logits, targets, candidate_global_ids, seen_global_ids
        ),
    }


def _metric_difference(
    target: Mapping[str, Any], reference: Mapping[str, Any]
) -> Dict[str, Any]:
    classification = _numeric_deltas(
        target["classification"], reference["classification"]
    )
    classification["nll_improvement"] = float(
        reference["classification"]["nll"] - target["classification"]["nll"]
    )
    return {
        "classification_target_minus_reference": classification,
        "prediction_health_target_minus_reference": _numeric_deltas(
            target["prediction_health"], reference["prediction_health"]
        ),
    }


def head_representation_factorial_metrics(
    cell_logits: Mapping[str, Mapping[str, Any]],
    targets: Any,
    candidate_global_ids: Sequence[int],
    seen_global_ids: Sequence[int],
    *,
    reference_representation: str = "frozen_prepass_cls",
    target_representation: str = "normal_final_cls",
    reference_head: str = "current_head",
    target_head: str = "candidate_head",
) -> Dict[str, Any]:
    candidate = np.asarray(candidate_global_ids, dtype=np.int64).reshape(-1)
    available: Dict[str, Dict[str, Any]] = {}
    sample_count = None
    for representation_name, heads in cell_logits.items():
        for head_name, values in heads.items():
            if values is None:
                continue
            logits = _matrix(values, f"{representation_name}/{head_name}")
            if logits.shape[1] != candidate.size:
                raise ValueError("factorial cell class count differs from candidate order")
            if sample_count is None:
                sample_count = int(logits.shape[0])
            elif sample_count != int(logits.shape[0]):
                raise ValueError("factorial cells differ in sample count")
            available.setdefault(str(representation_name), {})[str(head_name)] = {
                "logits": logits,
            }
    if sample_count is None:
        raise ValueError("at least one factorial cell is required")
    target_ids = _targets(targets, sample_count, candidate.size)
    cells: Dict[str, Dict[str, Any]] = {}
    for representation_name, heads in available.items():
        cells[representation_name] = {}
        for head_name, payload in heads.items():
            cells[representation_name][head_name] = _cell_metrics(
                payload["logits"], target_ids, candidate, seen_global_ids
            )
    representation_gain: Dict[str, Any] = {}
    for head_name in sorted(
        set(cells.get(reference_representation, {})).intersection(
            cells.get(target_representation, {})
        )
    ):
        representation_gain[head_name] = _metric_difference(
            cells[target_representation][head_name],
            cells[reference_representation][head_name],
        )
    head_gain: Dict[str, Any] = {}
    for representation_name in (reference_representation, target_representation):
        heads = cells.get(representation_name, {})
        if reference_head in heads and target_head in heads:
            head_gain[representation_name] = _metric_difference(
                heads[target_head], heads[reference_head]
            )
    complete = bool(
        all(
            head_name in cells.get(representation_name, {})
            for representation_name in (
                reference_representation,
                target_representation,
            )
            for head_name in (reference_head, target_head)
        )
    )
    interaction: Optional[Dict[str, Any]] = None
    if complete:
        target_rep_gain = representation_gain[target_head]
        reference_rep_gain = representation_gain[reference_head]
        interaction = {
            "classification": _numeric_deltas(
                target_rep_gain["classification_target_minus_reference"],
                reference_rep_gain["classification_target_minus_reference"],
            ),
            "prediction_health": _numeric_deltas(
                target_rep_gain["prediction_health_target_minus_reference"],
                reference_rep_gain["prediction_health_target_minus_reference"],
            ),
            "formula": "candidate_head_representation_gain_minus_current_head_representation_gain",
        }
    required_cells = [
        f"{reference_representation}/{reference_head}",
        f"{target_representation}/{reference_head}",
        f"{reference_representation}/{target_head}",
        f"{target_representation}/{target_head}",
    ]
    observed_cells = [
        f"{representation_name}/{head_name}"
        for representation_name, heads in cells.items()
        for head_name in heads
    ]
    current_cells_valid = bool(
        reference_head in cells.get(reference_representation, {})
        and reference_head in cells.get(target_representation, {})
    )
    status = (
        "complete"
        if complete
        else "partial_current_head_only"
        if current_cells_valid
        and all(
            head_name == reference_head
            for heads in cells.values()
            for head_name in heads
        )
        else "partial_incomplete_factorial"
    )
    return {
        "status": status,
        "required_cells": required_cells,
        "observed_cells": sorted(observed_cells),
        "missing_cells": sorted(set(required_cells).difference(observed_cells)),
        "cells": cells,
        "representation_gain_by_head": representation_gain,
        "head_gain_by_representation": head_gain,
        "interaction": interaction,
        "metric_orientation": {
            "top1_top5_per_class": "higher_is_better",
            "nll": "lower_is_better",
            "nll_improvement": "positive_is_better",
            "prediction_health_deltas": "descriptive_no_single_global_direction",
        },
        "validity": {
            "current_available_cells_valid": current_cells_valid,
            "candidate_head_available": bool(complete),
            "complete": bool(complete),
        },
    }
