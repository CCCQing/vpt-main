#!/usr/bin/env python3
"""Isolated checkpoint-only diagnostics registered after B3-D1.

Experiments:
  D2: prepass CLS -> trained mu retention versus random MLP controls.
  D2G: absolute class geometry of prepass CLS, trained mu and random MLP mu.
  D3: residual-induced Prompt slot allocation and raw/LN/K/V propagation.
  D4: label-using class-center oracle boundaries (never a deployable method).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.eval_metrics import (
    representation_geometry_metrics,
    semantic_visual_graph_metrics,
    visual_semantic_alignment_metrics,
)
from src.monitoring.logit_geometry import VIEWS, analyze_logit_geometry
from src.monitoring.module_effect import (
    checkpoint_sha256,
    deep_prompt_residual_zero_intervention,
)
from src.monitoring.prompt_source_retention import (
    ABSOLUTE_GEOMETRY_METRICS,
    absolute_class_geometry,
    compare_to_random_controls,
    geometry_deltas,
    geometry_deltas_from_summary,
    source_retention_metrics,
    summarize_geometry_controls,
)
from src.monitoring.representation_transport import (
    head_representation_factorial_metrics,
    prediction_transition_group_metrics,
    representation_transport_metrics,
    semantic_transport_metrics,
)
from src.tools.search_plans.a_series.replay_decision_gain_decomposition import (
    _load_model_and_loaders,
    _source_identity,
)
from src.tools.search_plans.b_series.replay_b_series_experiments import (
    _atomic_json,
    _build_loaders,
    _candidate_class_ids,
    _condition_summary,
    _json_safe,
    _load_cfg,
    _model_num_layers,
    _predict,
    _source_dataset,
)


DEFAULT_RANDOM_MLP_SEEDS = (23001, 23002, 23003, 23004, 23005)
DEFAULT_WRONG_CLASS_SEEDS = (24001, 24002, 24003)
LOGIT_GEOMETRY_METRICS = (
    "within_class_scatter_trace",
    "between_class_scatter_trace",
    "fisher_trace_ratio",
    "same_minus_interclass_cosine_gap",
    "nearest_class_center_cosine_margin_mean",
    "leave_one_out_center_accuracy",
    "effective_rank",
)


def _parse_ints(raw: str, *, minimum_count: int) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if (
        len(values) < int(minimum_count)
        or len(values) != len(set(values))
        or any(value < 0 for value in values)
    ):
        raise ValueError(
            "seed lists must contain at least {} unique non-negative integers".format(
                minimum_count
            )
        )
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiments", default="D2,D3,D4")
    parser.add_argument("--scope", choices=("full", "probe"), default="full")
    parser.add_argument("--selection-seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pair-seed", type=int, default=22001)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument(
        "--random-mlp-seeds",
        default=",".join(str(value) for value in DEFAULT_RANDOM_MLP_SEEDS),
    )
    parser.add_argument(
        "--wrong-class-seeds",
        default=",".join(str(value) for value in DEFAULT_WRONG_CLASS_SEEDS),
    )
    parser.add_argument("--selected-layers", default="8,9,10,11")
    parser.add_argument("--role-permutation-seed", type=int, default=25001)
    return parser.parse_args()


def _experiments(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip().upper() for item in str(raw).split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(
        value not in {"D2", "D2G", "D3", "D4"} for value in values
    ):
        raise ValueError("--experiments must be a unique subset of D2,D2G,D3,D4")
    return values


def _selected_layers(raw: str, num_layers: int) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(
        value < 0 or value >= int(num_layers) for value in values
    ):
        raise ValueError("--selected-layers contains invalid layer ids")
    return values


def _model_module(model):
    return model.module if hasattr(model, "module") else model


def _provider(model):
    provider = _model_module(model).enc.transformer.prompt_init_provider
    if provider is None or not hasattr(provider, "stats_head"):
        raise RuntimeError("Prompt Distributor stats_head is unavailable")
    return provider


@torch.no_grad()
def _random_mlp_means(
    model,
    visual_input: np.ndarray,
    seeds: Sequence[int],
) -> Dict[int, np.ndarray]:
    provider = _provider(model)
    visual = torch.as_tensor(visual_input, dtype=torch.float32)
    outputs: Dict[int, np.ndarray] = {}
    for seed in seeds:
        head = copy.deepcopy(provider.stats_head).cpu()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for module in head.modules():
                if not tuple(module.children()) and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        head.eval()
        chunks = []
        for start in range(0, int(visual.shape[0]), 512):
            stats = head(visual[start : start + 512])
            chunks.append(stats.chunk(2, dim=-1)[0].cpu().numpy())
        outputs[int(seed)] = np.concatenate(chunks, axis=0)
    return outputs


def _run_d2(model, device, loaders, pair_seed, max_pairs, random_seeds):
    provider = _provider(model)
    if str(provider.source).lower() != "vit_cls_prepass":
        raise ValueError("D2 requires SOURCE=vit_cls_prepass, not a constant control")
    report: Dict[str, Any] = {
        "format": "b3_d2_prepass_to_mu_retention_v1",
        "object_pair": ["frozen_prepass_cls", "trained_distributor_mu"],
        "random_control_count": len(random_seeds),
        "random_mlp_seeds": list(random_seeds),
        "splits": {},
    }
    for split, loader in loaders.items():
        output = _predict(
            model,
            device,
            loader,
            collect_source_input=True,
        )
        trained, pairs = source_retention_metrics(
            output["source_input"],
            output["source_mu"],
            output["targets_global"],
            pair_seed=int(pair_seed),
            max_pairs=int(max_pairs),
        )
        random_means = _random_mlp_means(
            model, output["source_input"], random_seeds
        )
        random_metrics = []
        for seed in random_seeds:
            metrics, _ = source_retention_metrics(
                output["source_input"],
                random_means[int(seed)],
                output["targets_global"],
                pair_seed=int(pair_seed),
                max_pairs=int(max_pairs),
                pair_indices=pairs,
            )
            random_metrics.append({"seed": int(seed), **metrics})
        report["splits"][split] = {
            "trained": trained,
            "random_controls": random_metrics,
            "trained_vs_random": compare_to_random_controls(
                trained, random_metrics
            ),
            "same_fixed_pair_indices_for_all_controls": True,
        }
    return report


def _array_sha256(values: np.ndarray, *, decimals: Optional[int] = None) -> str:
    matrix = np.asarray(values, dtype=np.float32)
    if decimals is not None:
        matrix = np.round(matrix.astype(np.float64), decimals=int(decimals)).astype(
            np.float32
        )
    contiguous = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _sample_manifest(output: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [
        [str(sample_id), int(label)]
        for sample_id, label in zip(
            output["sample_ids"], output["targets_global"].tolist()
        )
    ]
    return {
        "sha256": _manifest_hash(rows),
        "sample_count": len(rows),
        "class_count": int(np.unique(output["targets_global"]).size),
    }


def _semantic_reference(
    semantic_prototypes: np.ndarray,
    candidate_class_ids: Sequence[int],
) -> Dict[str, Any]:
    prototypes = np.asarray(semantic_prototypes, dtype=np.float32)
    candidate = np.asarray(candidate_class_ids, dtype=np.int64).reshape(-1)
    if prototypes.ndim != 2 or prototypes.shape[0] != candidate.size:
        raise ValueError("semantic prototypes do not match candidate class order")
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    normalized = prototypes / np.maximum(norms, 1.0e-12)
    relation = normalized @ normalized.T
    diagonal = np.diag(relation)
    geometry = representation_geometry_metrics(prototypes)
    validity = {
        "finite": bool(np.isfinite(prototypes).all() and np.isfinite(relation).all()),
        "candidate_order_matches": bool(prototypes.shape[0] == candidate.size),
        "no_zero_norm_prototype": bool(np.all(norms.reshape(-1) > 1.0e-12)),
        "relation_symmetric": bool(
            np.max(np.abs(relation - relation.T)) <= 1.0e-6
        ),
        "relation_diagonal_is_one": bool(
            np.max(np.abs(diagonal - 1.0)) <= 1.0e-5
        ),
        "effective_rank_present": bool(
            "effective_rank" in geometry
            and np.isfinite(float(geometry["effective_rank"]))
        ),
    }
    validity["valid"] = bool(all(validity.values()))
    return {
        "prototype_count": int(prototypes.shape[0]),
        "prototype_dim": int(prototypes.shape[1]),
        "candidate_class_ids": candidate.tolist(),
        "candidate_order_sha256": _manifest_hash(candidate.tolist()),
        "prototype_float32_sha256": _array_sha256(prototypes),
        "geometry": geometry,
        "cosine_relation_matrix": relation,
        "cosine_relation_matrix_sha256": _array_sha256(relation),
        "validity": validity,
    }


def _visual_semantic_condition(output: Mapping[str, Any]) -> Dict[str, Any]:
    alignment = visual_semantic_alignment_metrics(
        output["features"],
        output["semantic_prototypes"],
        output["targets_local"],
        recall_k=5,
    )
    graph = semantic_visual_graph_metrics(
        output["features"],
        output["semantic_prototypes"],
        output["targets_local"],
        logits=output["logits"],
        neighbor_k=5,
    )
    required_alignment = {
        "class_center_prototype_cosine",
        "visual_semantic_structure_spearman",
        "visual_semantic_distance_spearman",
        "true_prototype_rank",
        "semantic_margin",
        "neighbor_preservation_at_k",
        "prototype_recall_at_k",
        "semantic_ambiguity_rate",
    }
    required_graph = {
        "neighbor_ranking_consistency",
        "confusion_edge_precision",
        "false_high_semantic_edge_rate",
    }
    values = [*alignment.values(), *graph.values()]
    validity = {
        "required_alignment_metrics_present": required_alignment.issubset(alignment),
        "required_graph_metrics_present": required_graph.issubset(graph),
        "finite": bool(all(np.isfinite(float(value)) for value in values)),
    }
    validity["valid"] = bool(all(validity.values()))
    return {
        "visual_space": "final_cls_visual_input",
        "semantic_space": "projected_semantic_repr",
        "recall_k": 5,
        "alignment": alignment,
        "semantic_graph": graph,
        "validity": validity,
    }


def _numeric_deltas(
    normal: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, float]:
    if set(normal) != set(reference):
        raise ValueError("paired metric fields differ")
    return {
        name: float(normal[name]) - float(reference[name])
        for name in normal
    }


def _visual_semantic_deltas(
    normal: Mapping[str, Any],
    residual_zero: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "normal_minus_residual_zero": {
            "alignment": _numeric_deltas(
                normal["alignment"], residual_zero["alignment"]
            ),
            "semantic_graph": _numeric_deltas(
                normal["semantic_graph"], residual_zero["semantic_graph"]
            ),
        }
    }


def _logit_geometry_deltas(
    normal: Mapping[str, Any],
    residual_zero: Mapping[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split in ("train_seen", "test_seen", "test_unseen", "joint"):
        normal_split = normal["joint"] if split == "joint" else normal["splits"][split]
        zero_split = (
            residual_zero["joint"]
            if split == "joint"
            else residual_zero["splits"][split]
        )
        result[split] = {}
        for view in VIEWS:
            result[split][view] = {
                metric: float(normal_split["views"][view][metric])
                - float(zero_split["views"][view][metric])
                for metric in LOGIT_GEOMETRY_METRICS
            }
    return {"normal_minus_residual_zero": result}


def _task_chain_summary(normal, residual_zero, cfg) -> Dict[str, Any]:
    normal_summary = _condition_summary(normal, normal, cfg, is_normal=True)
    zero_summary = _condition_summary(
        residual_zero, normal, cfg, is_normal=False
    )
    normal_summary.pop("logit_geometry", None)
    zero_summary.pop("logit_geometry", None)
    task_metrics = (
        "seen_per_class_accuracy",
        "unseen_per_class_accuracy",
        "harmonic_mean",
        "ausuc",
    )
    delta = {
        metric: float(normal_summary["gzsl"][metric])
        - float(zero_summary["gzsl"][metric])
        for metric in task_metrics
    }
    return {
        "normal": normal_summary,
        "residual_zero": zero_summary,
        "normal_minus_residual_zero": delta,
        "valid": bool(normal_summary["valid"] and zero_summary["valid"]),
    }


def _current_head_representation_task_summary(
    normal_outputs: Mapping[str, Mapping[str, Any]], cfg
) -> Dict[str, Any]:
    prepass_outputs = {
        split: {
            **output,
            "logits": output["prepass_current_head_logits"],
            "features": output["source_input"],
        }
        for split, output in normal_outputs.items()
    }
    prepass = _condition_summary(
        prepass_outputs, prepass_outputs, cfg, is_normal=True
    )
    final = _condition_summary(
        normal_outputs, normal_outputs, cfg, is_normal=True
    )
    prepass.pop("logit_geometry", None)
    final.pop("logit_geometry", None)
    task_metrics = (
        "seen_per_class_accuracy",
        "unseen_per_class_accuracy",
        "harmonic_mean",
        "ausuc",
        "raw_to_oracle_gain",
        "oracle_peak_gamma",
    )
    delta = {
        metric: float(final["gzsl"][metric]) - float(prepass["gzsl"][metric])
        for metric in task_metrics
    }
    return {
        "status": "partial_current_head_only",
        "cells": {
            "frozen_prepass_cls/current_head": prepass,
            "normal_final_cls/current_head": final,
            "frozen_prepass_cls/candidate_head": None,
            "normal_final_cls/candidate_head": None,
        },
        "representation_gain_current_head_final_minus_prepass": delta,
        "candidate_head_status": "not_available_not_trained",
        "interaction": None,
        "valid": bool(prepass["valid"] and final["valid"]),
    }


def _run_d2g(
    model,
    device,
    loaders,
    cfg,
    pair_seed,
    max_pairs,
    random_seeds,
    source_checkpoint_sha256,
):
    provider = _provider(model)
    if str(provider.source).lower() != "vit_cls_prepass":
        raise ValueError("D2G requires SOURCE=vit_cls_prepass")
    report: Dict[str, Any] = {
        "format": "b3_d2g_source_to_decision_chain_v3",
        "objects": [
            "frozen_prepass_cls",
            "trained_distributor_mu",
            "random_untrained_distributor_mu",
            "normal_final_cls",
            "cls_delta",
            "residual_zero_final_cls",
            "projected_semantic_prototypes",
            "final_200d_logits",
            "current_head_on_frozen_prepass_cls",
        ],
        "absolute_geometry_metrics": list(ABSOLUTE_GEOMETRY_METRICS),
        "logit_views": list(VIEWS),
        "logit_geometry_metrics": list(LOGIT_GEOMETRY_METRICS),
        "random_mlp_seeds": list(random_seeds),
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "analysis_contract": {
            "sample_identity": "same sample ids, labels and candidate class order inside each split",
            "absolute_geometry_preprocessing": "featurewise_standardize_then_row_l2_normalize",
            "class_weighting": "macro_equal_observed_classes_where_applicable",
            "random_control_nesting": "five random MLP seeds nested inside one checkpoint",
            "probe_nesting": "three selection seeds nested inside one training checkpoint",
            "raw_scatter_cross_space_subtraction_allowed": False,
            "missing_metric_fill_value": None,
            "prepass_execution": "one cached extraction per batch reused by normal and Residual-zero",
            "prepass_final_transport": "paired on identical samples; cls_delta is a deterministic transform, not an independent Bayesian object",
            "prediction_group_boundary": "posthoc descriptive audit; never a training target or causal proof",
        },
        "atomic_evidence_group_schema": [
            "representation_identity",
            "prepass_final_transport",
            "absolute_class_geometry",
            "semantic_transport",
            "head_representation_factorial",
            "prediction_transition_groups",
            "official_final_task",
        ],
        "p13a_extension_contract": {
            "status": "candidate_head_not_available",
            "current_replay_scope": "current_head_on_prepass_and_final_cls",
            "required_future_cells": [
                "frozen_prepass_cls/current_head",
                "normal_final_cls/current_head",
                "frozen_prepass_cls/candidate_head",
                "normal_final_cls/candidate_head",
            ],
            "future_training_contract": "capacity-matched heads; all official Seen classes; predeclared fixed epochs; three independent training seeds; official final-GZSL evaluation",
            "missing_cells_are_not_filled_with_zero": True,
        },
        "splits": {},
        "cross_method_comparability": {
            "status": "not_requested_in_single_checkpoint_replay",
            "allowed_only_after": [
                "sample_manifest_match",
                "candidate_order_match",
                "checkpoint_provenance_match",
                "metric_contract_match",
            ],
        },
        "valid": True,
        "failure_reasons": [],
    }
    normal_outputs: Dict[str, Dict[str, Any]] = {}
    residual_zero_outputs: Dict[str, Dict[str, Any]] = {}
    semantic_reference = None
    module = _model_module(model)
    zero_scales = [0.0] * _model_num_layers(model)
    for split, loader in loaders.items():
        cache_namespace = "d2g|{}|{}".format(
            str(source_checkpoint_sha256)[:16], split
        )
        try:
            output = _predict(
                model,
                device,
                loader,
                collect_source_input=True,
                collect_classifier_readout=True,
                prepass_cache_mode="populate",
                prepass_cache_namespace=cache_namespace,
            )
            residual_zero = _predict(
                model,
                device,
                loader,
                scales=zero_scales,
                collect_source_input=True,
                prepass_cache_mode="reuse",
                prepass_cache_namespace=cache_namespace,
            )
        finally:
            module.end_runtime_vit_cls_prepass_cache()
        normal_outputs[split] = output
        residual_zero_outputs[split] = residual_zero
        labels = output["targets_global"]
        representation_transport = representation_transport_metrics(
            output["source_input"],
            output["features"],
            labels,
            pair_seed=int(pair_seed),
            max_pairs=int(max_pairs),
        )
        prepass = representation_transport["absolute_geometry"][
            "frozen_prepass_cls"
        ]
        trained = absolute_class_geometry(output["source_mu"], labels)
        normal_final_cls = representation_transport["absolute_geometry"][
            "normal_final_cls"
        ]
        residual_zero_final_cls = absolute_class_geometry(
            residual_zero["features"], labels
        )
        source_dataset = _source_dataset(loader)
        raw_attribute_value = source_dataset.class_attributes
        raw_attributes = (
            raw_attribute_value.detach().cpu().float().numpy()
            if torch.is_tensor(raw_attribute_value)
            else np.asarray(raw_attribute_value, dtype=np.float32)
        )
        candidate_indices = np.asarray(
            output["candidate_class_ids"], dtype=np.int64
        )
        if (
            raw_attributes.ndim != 2
            or candidate_indices.size < 2
            or int(candidate_indices.min()) < 0
            or int(candidate_indices.max()) >= raw_attributes.shape[0]
        ):
            raise ValueError(
                "raw class attributes do not cover the candidate class order"
            )
        raw_candidate_semantics = raw_attributes[candidate_indices]
        prepass_logits = output["prepass_current_head_logits"]
        reconstructed_final_logits = output[
            "classifier_reconstructed_logits"
        ]
        classifier_reconstruction = {
            "contract": output["classifier_contract"],
            "max_abs_logit_error": float(
                np.max(np.abs(reconstructed_final_logits - output["logits"]))
            ),
            "mean_abs_logit_error": float(
                np.mean(np.abs(reconstructed_final_logits - output["logits"]))
            ),
            "prediction_equivalence": float(
                np.mean(
                    reconstructed_final_logits.argmax(axis=1)
                    == output["logits"].argmax(axis=1)
                )
            ),
            "exact_float32_equal": bool(
                np.array_equal(reconstructed_final_logits, output["logits"])
            ),
        }
        classifier_reconstruction["valid"] = bool(
            classifier_reconstruction["prediction_equivalence"] == 1.0
            and classifier_reconstruction["max_abs_logit_error"] <= 1.0e-5
            and output["classifier_contract"][
                "logit_scale_constant_across_batches"
            ]
        )
        semantic_transport = semantic_transport_metrics(
            output["source_input"],
            output["features"],
            output["semantic_prototypes"],
            output["targets_local"],
            prepass_logits=prepass_logits,
            final_logits=output["logits"],
            raw_semantic_prototypes=raw_candidate_semantics,
        )
        head_factorial = head_representation_factorial_metrics(
            {
                "frozen_prepass_cls": {"current_head": prepass_logits},
                "normal_final_cls": {"current_head": output["logits"]},
            },
            output["targets_local"],
            output["candidate_class_ids"],
            output["seen_class_ids"],
        )
        transition_groups = prediction_transition_group_metrics(
            output["source_input"],
            output["features"],
            prepass_logits,
            output["logits"],
            output["targets_local"],
            output["candidate_class_ids"],
            output["seen_class_ids"],
            output["semantic_prototypes"],
            reference_name="current_head_on_frozen_prepass_cls",
            target_name="current_head_on_normal_final_cls",
        )
        normal_visual_semantic = _visual_semantic_condition(output)
        residual_zero_visual_semantic = _visual_semantic_condition(residual_zero)
        relationship, pairs = source_retention_metrics(
            output["source_input"],
            output["source_mu"],
            labels,
            pair_seed=int(pair_seed),
            max_pairs=int(max_pairs),
        )
        random_means = _random_mlp_means(
            model, output["source_input"], random_seeds
        )
        random_geometry = []
        random_relationships = []
        for seed in random_seeds:
            geometry = absolute_class_geometry(random_means[int(seed)], labels)
            random_geometry.append({"seed": int(seed), **geometry})
            relation, _ = source_retention_metrics(
                output["source_input"],
                random_means[int(seed)],
                labels,
                pair_seed=int(pair_seed),
                max_pairs=int(max_pairs),
                pair_indices=pairs,
            )
            random_relationships.append({"seed": int(seed), **relation})
        random_summary = summarize_geometry_controls(random_geometry)
        random_deltas = [
            {
                "seed": int(item["seed"]),
                **geometry_deltas(trained, item),
            }
            for item in random_geometry
        ]
        normal_manifest = _sample_manifest(output)
        residual_zero_manifest = _sample_manifest(residual_zero)
        current_semantic_reference = _semantic_reference(
            output["semantic_prototypes"], output["candidate_class_ids"]
        )
        if semantic_reference is None:
            semantic_reference = current_semantic_reference
        semantic_reference_matches = bool(
            semantic_reference["prototype_float32_sha256"]
            == current_semantic_reference["prototype_float32_sha256"]
            and semantic_reference["candidate_order_sha256"]
            == current_semantic_reference["candidate_order_sha256"]
        )
        prepass_hash = _array_sha256(output["source_input"])
        zero_prepass_hash = _array_sha256(residual_zero["source_input"])
        trained_mu_hash = _array_sha256(output["source_mu"])
        zero_mu_hash = _array_sha256(residual_zero["source_mu"])
        semantic_hash = _array_sha256(output["semantic_prototypes"])
        zero_semantic_hash = _array_sha256(residual_zero["semantic_prototypes"])
        validity = {
            "prepass_geometry_valid": bool(prepass["validity"]["valid"]),
            "trained_geometry_valid": bool(trained["validity"]["valid"]),
            "all_random_geometry_valid": bool(
                all(item["validity"]["valid"] for item in random_geometry)
            ),
            "normal_final_cls_geometry_valid": bool(
                normal_final_cls["validity"]["valid"]
            ),
            "residual_zero_final_cls_geometry_valid": bool(
                residual_zero_final_cls["validity"]["valid"]
            ),
            "normal_visual_semantic_valid": bool(
                normal_visual_semantic["validity"]["valid"]
            ),
            "residual_zero_visual_semantic_valid": bool(
                residual_zero_visual_semantic["validity"]["valid"]
            ),
            "same_fixed_pair_indices_for_all_relationships": True,
            "normal_residual_zero_sample_manifest_equal": bool(
                normal_manifest["sha256"] == residual_zero_manifest["sha256"]
            ),
            "normal_residual_zero_candidate_order_equal": bool(
                output["candidate_class_ids"]
                == residual_zero["candidate_class_ids"]
            ),
            "normal_residual_zero_prepass_exact_equal": bool(
                prepass_hash == zero_prepass_hash
            ),
            "normal_residual_zero_trained_mu_exact_equal": bool(
                trained_mu_hash == zero_mu_hash
            ),
            "normal_residual_zero_semantic_prototype_exact_equal": bool(
                semantic_hash == zero_semantic_hash
            ),
            "semantic_reference_matches_all_splits": semantic_reference_matches,
            "normal_intervention_contract_pass": bool(
                output["intervention_contract"]["pass"]
            ),
            "residual_zero_intervention_contract_pass": bool(
                residual_zero["intervention_contract"]["pass"]
            ),
            "prepass_cache_batch_counts_equal": bool(
                output["prepass_cache_contract"]["batch_key_count"] > 0
                and output["prepass_cache_contract"]["batch_key_count"]
                == residual_zero["prepass_cache_contract"]["batch_key_count"]
            ),
            "prepass_final_transport_valid": bool(
                representation_transport["validity"]["valid"]
            ),
            "prepass_final_semantic_transport_valid": bool(
                semantic_transport["validity"]["valid"]
            ),
            "current_head_reconstruction_valid": bool(
                classifier_reconstruction["valid"]
            ),
            "current_head_factorial_cells_valid": bool(
                head_factorial["validity"]["current_available_cells_valid"]
            ),
            "prediction_transition_groups_valid": bool(
                transition_groups["validity"]["valid"]
            ),
            "finite_deltas": bool(
                all(
                    np.isfinite(float(value))
                    for value in (
                        list(geometry_deltas(trained, prepass).values())
                        + list(
                            geometry_deltas_from_summary(
                                trained, random_summary
                            ).values()
                        )
                        + list(
                            geometry_deltas(
                                normal_final_cls, residual_zero_final_cls
                            ).values()
                        )
                        + [
                            delta
                            for item in random_deltas
                            for name, delta in item.items()
                            if name != "seed"
                        ]
                    )
                )
            ),
        }
        validity["valid"] = bool(all(validity.values()))
        if not validity["valid"]:
            report["valid"] = False
            report["failure_reasons"].append(
                "{}: absolute geometry validity failed".format(split)
            )
        report["splits"][split] = {
            "sample_manifest": normal_manifest,
            "residual_zero_sample_manifest": residual_zero_manifest,
            "prepass_equivalence": {
                "float32_sha256": prepass_hash,
                "rounded_1e-5_sha256": _array_sha256(
                    output["source_input"], decimals=5
                ),
                "normal_vs_residual_zero_exact_equal": bool(
                    prepass_hash == zero_prepass_hash
                ),
                "shape": list(output["source_input"].shape),
            },
            "candidate_class_ids": list(output["candidate_class_ids"]),
            "candidate_order_sha256": _manifest_hash(
                output["candidate_class_ids"]
            ),
            "prepass_cls_geometry": prepass,
            "trained_mu_geometry": trained,
            "random_mu_geometry": random_geometry,
            "random_mu_geometry_summary": random_summary,
            "normal_final_cls_geometry": normal_final_cls,
            "cls_delta_geometry": representation_transport[
                "absolute_geometry"
            ]["cls_delta"],
            "residual_zero_final_cls_geometry": residual_zero_final_cls,
            "prepass_final_cls_transport": representation_transport,
            "prepass_final_semantic_transport": semantic_transport,
            "current_head_representation_factorial": head_factorial,
            "prediction_transition_groups": transition_groups,
            "classifier_reconstruction_contract": classifier_reconstruction,
            "visual_semantic_alignment": {
                "normal": normal_visual_semantic,
                "residual_zero": residual_zero_visual_semantic,
                "paired_deltas": _visual_semantic_deltas(
                    normal_visual_semantic, residual_zero_visual_semantic
                ),
            },
            "relationship_metrics": {
                "trained": relationship,
                "random_controls": random_relationships,
                "trained_vs_random": compare_to_random_controls(
                    relationship, random_relationships
                ),
            },
            "paired_deltas": {
                "trained_mu_minus_prepass_cls": geometry_deltas(
                    trained, prepass
                ),
                "trained_mu_minus_random_mean": geometry_deltas_from_summary(
                    trained, random_summary
                ),
                "trained_mu_minus_each_random": random_deltas,
                "normal_final_cls_minus_residual_zero_final_cls": geometry_deltas(
                    normal_final_cls, residual_zero_final_cls
                ),
            },
            "identity_hashes": {
                "normal_trained_mu_float32_sha256": trained_mu_hash,
                "residual_zero_trained_mu_float32_sha256": zero_mu_hash,
                "normal_semantic_prototype_float32_sha256": semantic_hash,
                "residual_zero_semantic_prototype_float32_sha256": zero_semantic_hash,
            },
            "prepass_cache_contract": {
                "normal": output["prepass_cache_contract"],
                "residual_zero": residual_zero["prepass_cache_contract"],
                "same_cached_prepass_verified_by_exact_hash": bool(
                    prepass_hash == zero_prepass_hash
                ),
                "cache_released_after_split": True,
            },
            "validity": validity,
        }
        del random_means
    if semantic_reference is None:
        raise RuntimeError("D2G did not observe projected semantic prototypes")
    normal_logit_geometry, _ = analyze_logit_geometry(
        {
            split: {
                "logits": output["logits"],
                "targets_local": output["targets_local"],
            }
            for split, output in normal_outputs.items()
        },
        normal_outputs["test_seen"]["candidate_class_ids"],
        normal_outputs["test_seen"]["seen_class_ids"],
        normal_outputs["test_seen"]["unseen_class_ids"],
    )
    residual_zero_logit_geometry, _ = analyze_logit_geometry(
        {
            split: {
                "logits": output["logits"],
                "targets_local": output["targets_local"],
            }
            for split, output in residual_zero_outputs.items()
        },
        residual_zero_outputs["test_seen"]["candidate_class_ids"],
        residual_zero_outputs["test_seen"]["seen_class_ids"],
        residual_zero_outputs["test_seen"]["unseen_class_ids"],
    )
    task_results = _task_chain_summary(
        normal_outputs, residual_zero_outputs, cfg
    )
    head_representation_task = _current_head_representation_task_summary(
        normal_outputs, cfg
    )
    downstream_validity = {
        "semantic_reference_valid": bool(semantic_reference["validity"]["valid"]),
        "normal_logit_geometry_valid": bool(
            normal_logit_geometry["validity"]["valid"]
        ),
        "residual_zero_logit_geometry_valid": bool(
            residual_zero_logit_geometry["validity"]["valid"]
        ),
        "task_results_valid": bool(task_results["valid"]),
        "current_head_representation_task_valid": bool(
            head_representation_task["valid"]
        ),
    }
    downstream_validity["valid"] = bool(all(downstream_validity.values()))
    if not downstream_validity["valid"]:
        report["valid"] = False
        report["failure_reasons"].append(
            "semantic, logit or task-chain validity failed"
        )
    report["semantic_reference"] = semantic_reference
    report["decision_space_dim"] = int(
        semantic_reference["prototype_count"]
    )
    report["logit_geometry"] = {
        "normal": normal_logit_geometry,
        "residual_zero": residual_zero_logit_geometry,
        "paired_deltas": _logit_geometry_deltas(
            normal_logit_geometry, residual_zero_logit_geometry
        ),
    }
    report["task_results"] = task_results
    report["head_representation_task_factorial"] = head_representation_task
    report["atomic_evidence_status"] = {
        "representation_identity": "available" if report["valid"] else "invalid",
        "prepass_final_transport": "available" if report["valid"] else "invalid",
        "absolute_class_geometry": "available" if report["valid"] else "invalid",
        "semantic_transport": "available" if report["valid"] else "invalid",
        "head_representation_factorial": "partial_current_head_only",
        "prediction_transition_groups": "available" if report["valid"] else "invalid",
        "official_final_task": "partial_candidate_head_not_trained",
    }
    report["downstream_validity"] = downstream_validity
    report["scientific_status"] = (
        "implementation_valid_pending_formal_checkpoint_replay"
        if report["valid"]
        else "invalid"
    )
    return report


def _entropy_effective_ratio(distribution: torch.Tensor) -> torch.Tensor:
    distribution = distribution.float().clamp_min(1.0e-12)
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return torch.exp(-(distribution * distribution.log()).sum(dim=-1)) / int(
        distribution.shape[-1]
    )


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float().clamp_min(1.0e-12)
    right = right.float().clamp_min(1.0e-12)
    left = left / left.sum(dim=-1, keepdim=True)
    right = right / right.sum(dim=-1, keepdim=True)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def _row_spearman(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    left_rank = rankdata(left.detach().cpu().numpy(), axis=1, method="average")
    right_rank = rankdata(right.detach().cpu().numpy(), axis=1, method="average")
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_rank, axis=1) * np.linalg.norm(
        right_rank, axis=1
    )
    return np.divide(
        (left_rank * right_rank).sum(axis=1),
        denominator,
        out=np.zeros(left_rank.shape[0], dtype=np.float64),
        where=denominator > 1.0e-12,
    )


def _slot_relation_spearman(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    left = torch.nn.functional.normalize(left.float(), dim=-1, eps=1.0e-12)
    right = torch.nn.functional.normalize(right.float(), dim=-1, eps=1.0e-12)
    left_relation = left @ left.transpose(-1, -2)
    right_relation = right @ right.transpose(-1, -2)
    indices = torch.triu_indices(
        int(left.shape[1]), int(left.shape[1]), offset=1
    )
    return _row_spearman(
        left_relation[:, indices[0], indices[1]],
        right_relation[:, indices[0], indices[1]],
    )


def _effective_rank(values: torch.Tensor) -> torch.Tensor:
    singular = torch.svd(values.float()).S
    probability = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return torch.exp(
        -(probability.clamp_min(1.0e-12) * probability.clamp_min(1.0e-12).log()).sum(
            dim=-1
        )
    )


def _offdiag_cosine(values: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(values.float(), dim=-1, eps=1.0e-12)
    matrix = normalized @ normalized.transpose(-1, -2)
    size = int(matrix.shape[-1])
    return (matrix.sum(dim=(-2, -1)) - size) / max(1, size * (size - 1))


def _attention_slot_distribution(affinity, name: str):
    values = affinity[name].float()
    distribution = values.mean(dim=1).mean(dim=1)
    mass = distribution.sum(dim=-1)
    return mass, distribution / mass[:, None].clamp_min(1.0e-12)


def _summary(values: Sequence[np.ndarray]) -> Dict[str, float]:
    merged = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    return {
        "mean": float(merged.mean()),
        "min": float(merged.min()),
        "max": float(merged.max()),
        "sample_count": int(merged.size),
    }


def _paired_affinity_forward(model, inputs, candidate, affinity_config, zero):
    intervention = deep_prompt_residual_zero_intervention(model) if zero else nullcontext()
    with intervention:
        output = model.forward_with_affinity(
            inputs,
            affinity_config,
            semantics=None,
            class_ids=candidate,
            runtime_targets=None,
        )
    logits, affinities = output[0], output[-1]
    module = _model_module(model)
    trace = [dict(item) for item in module.get_runtime_deep_prompt_residual_trace()]
    return logits.detach().cpu(), affinities, trace


@torch.no_grad()
def _run_d3_split(model, device, loader, selected_layers):
    dataset = _source_dataset(loader)
    candidate = _candidate_class_ids(dataset)
    prompt_len = int(_provider(model).prompt_len)
    affinity_config = {
        "prompt_length": prompt_len,
        "semantic_length": 0,
        "detach": True,
        "include_visual_normalizations": False,
        "selected_layers": list(selected_layers),
        "collect_prompt_slot_states": True,
        "offload_diagnostics_to_cpu": True,
    }
    buckets: Dict[str, list[np.ndarray]] = {}

    def add(name, value):
        buckets.setdefault(name, []).append(np.asarray(value))

    model.eval()
    for batch in loader:
        inputs = batch["image"].float().to(device, non_blocking=True)
        normal_logits, normal_aff, normal_trace = _paired_affinity_forward(
            model, inputs, candidate, affinity_config, False
        )
        zero_logits, zero_aff, _ = _paired_affinity_forward(
            model, inputs, candidate, affinity_config, True
        )
        add(
            "delta_logits_norm",
            (normal_logits - zero_logits).float().norm(dim=-1).numpy(),
        )
        normal_by_layer = {int(item["layer_id"]): item for item in normal_trace}
        for layer_id in selected_layers:
            normal_layer = normal_aff[layer_id]
            zero_layer = zero_aff[layer_id]
            prefix = "layer_{}".format(layer_id)
            for route_name, affinity_key in (
                ("cls_to_prompt", "AcKp_attn"),
                ("patch_to_prompt", "AvKp_attn"),
            ):
                normal_mass, normal_distribution = _attention_slot_distribution(
                    normal_layer, affinity_key
                )
                zero_mass, zero_distribution = _attention_slot_distribution(
                    zero_layer, affinity_key
                )
                add(prefix + "__" + route_name + "__normal_mass", normal_mass.numpy())
                add(prefix + "__" + route_name + "__zero_mass", zero_mass.numpy())
                add(
                    prefix + "__" + route_name + "__mass_delta",
                    (normal_mass - zero_mass).numpy(),
                )
                add(
                    prefix + "__" + route_name + "__slot_js",
                    _js_divergence(normal_distribution, zero_distribution).numpy(),
                )
                add(
                    prefix + "__" + route_name + "__slot_spearman",
                    _row_spearman(normal_distribution, zero_distribution),
                )
                add(
                    prefix + "__" + route_name + "__normal_effective_ratio",
                    _entropy_effective_ratio(normal_distribution).numpy(),
                )
                add(
                    prefix + "__" + route_name + "__zero_effective_ratio",
                    _entropy_effective_ratio(zero_distribution).numpy(),
                )
                add(
                    prefix + "__" + route_name + "__normal_effective_count",
                    (
                        prompt_len
                        * _entropy_effective_ratio(normal_distribution)
                    ).numpy(),
                )
                add(
                    prefix + "__" + route_name + "__zero_effective_count",
                    (
                        prompt_len
                        * _entropy_effective_ratio(zero_distribution)
                    ).numpy(),
                )
                normal_head = normal_layer[affinity_key].float().mean(dim=2)
                zero_head = zero_layer[affinity_key].float().mean(dim=2)
                normal_head_mass = normal_head.sum(dim=-1)
                zero_head_mass = zero_head.sum(dim=-1)
                normal_head_distribution = normal_head / normal_head_mass[
                    :, :, None
                ].clamp_min(1.0e-12)
                zero_head_distribution = zero_head / zero_head_mass[
                    :, :, None
                ].clamp_min(1.0e-12)
                add(
                    prefix + "__" + route_name + "__head_mass_delta",
                    (normal_head_mass - zero_head_mass).numpy(),
                )
                add(
                    prefix + "__" + route_name + "__head_slot_js",
                    _js_divergence(
                        normal_head_distribution.reshape(-1, prompt_len),
                        zero_head_distribution.reshape(-1, prompt_len),
                    ).numpy(),
                )
                add(
                    prefix + "__" + route_name + "__head_slot_spearman",
                    _row_spearman(
                        normal_head_distribution.reshape(-1, prompt_len),
                        zero_head_distribution.reshape(-1, prompt_len),
                    ),
                )
            for state_name, key in (
                ("raw", "_prompt_slot_raw_input"),
                ("layernorm", "_prompt_slot_ln_input"),
                ("key", "_prompt_slot_key"),
                ("value", "_prompt_slot_value"),
            ):
                delta = normal_layer[key] - zero_layer[key]
                add(
                    prefix + "__delta_" + state_name + "__effective_rank",
                    _effective_rank(delta).numpy(),
                )
                add(
                    prefix + "__delta_" + state_name + "__offdiag_cosine",
                    _offdiag_cosine(delta).numpy(),
                )
                add(
                    prefix + "__delta_" + state_name + "__rms",
                    delta.float().square().mean(dim=(-2, -1)).sqrt().numpy(),
                )
            trace = normal_by_layer[layer_id]
            base = trace["base_prompt"].detach().cpu().float()
            residual = trace["applied_delta"].detach().cpu().float()
            slot_cosine = torch.nn.functional.cosine_similarity(
                residual, base, dim=-1, eps=1.0e-12
            )
            slot_ratio = residual.norm(dim=-1) / base.norm(dim=-1).clamp_min(1.0e-12)
            parallel_coefficient = (residual * base).sum(dim=-1) / base.float().square().sum(
                dim=-1
            ).clamp_min(1.0e-12)
            orthogonal = residual - parallel_coefficient[:, :, None] * base
            orthogonal_ratio = orthogonal.norm(dim=-1) / residual.norm(
                dim=-1
            ).clamp_min(1.0e-12)
            add(prefix + "__residual_static_slot_cosine", slot_cosine.numpy())
            add(prefix + "__residual_static_slot_norm_ratio", slot_ratio.numpy())
            add(
                prefix + "__residual_static_signed_parallel_projection",
                parallel_coefficient.numpy(),
            )
            add(
                prefix + "__residual_static_orthogonal_ratio",
                orthogonal_ratio.numpy(),
            )
            add(
                prefix + "__residual_static_cosine_across_slot_std",
                slot_cosine.std(dim=1, unbiased=False).numpy(),
            )
            add(
                prefix + "__residual_static_parallel_across_slot_std",
                parallel_coefficient.std(dim=1, unbiased=False).numpy(),
            )
            add(
                prefix + "__residual_static_orthogonal_across_slot_std",
                orthogonal_ratio.std(dim=1, unbiased=False).numpy(),
            )
            add(
                prefix + "__slot_relation_spearman_before_after_injection",
                _slot_relation_spearman(base, base + residual),
            )
            add(
                prefix + "__applied_group_rms_ratio",
                trace["applied_ratio"].detach().cpu().float().numpy(),
            )
            add(
                prefix + "__common_component_rms",
                trace["common_component_rms"].detach().cpu().float().numpy(),
            )
            add(
                prefix + "__role_component_rms",
                trace["role_component_rms"].detach().cpu().float().numpy(),
            )
            delta_value = normal_layer["_prompt_slot_value"] - zero_layer[
                "_prompt_slot_value"
            ]
            delta_value_norm = delta_value.float().norm(dim=-1)
            for consumer, affinity_key in (
                ("cls", "AcKp_attn"),
                ("patch", "AvKp_attn"),
            ):
                _, consumer_distribution = _attention_slot_distribution(
                    normal_layer, affinity_key
                )
                value_score = consumer_distribution * delta_value_norm
                value_score_sum = value_score.sum(dim=-1)
                value_distribution = value_score / value_score_sum[
                    :, None
                ].clamp_min(1.0e-12)
                add(
                    prefix
                    + "__{}_attention_weighted_delta_value".format(consumer),
                    value_score_sum.numpy(),
                )
                add(
                    prefix
                    + "__{}_value_contribution_effective_count".format(
                        consumer
                    ),
                    (
                        prompt_len
                        * _entropy_effective_ratio(value_distribution)
                    ).numpy(),
                )
                add(
                    prefix
                    + "__{}_value_contribution_top4_share".format(consumer),
                    torch.topk(
                        value_distribution,
                        k=min(4, prompt_len),
                        dim=-1,
                    ).values.sum(dim=-1).numpy(),
                )
                for slot_id in range(prompt_len):
                    add(
                        prefix
                        + "__{}_attention_weighted_delta_value_slot_{}".format(
                            consumer, slot_id
                        ),
                        value_score[:, slot_id].numpy(),
                    )
        if hasattr(_model_module(model), "clear_runtime_state"):
            _model_module(model).clear_runtime_state()
        del inputs, normal_aff, zero_aff, normal_trace
    return {name: _summary(values) for name, values in sorted(buckets.items())}


def _run_d3(model, device, loaders, selected_layers):
    return {
        "format": "b3_d3_prompt_slot_propagation_v1",
        "normal_vs": "all_deep_residual_zero",
        "selected_layers": list(selected_layers),
        "metrics": {
            split: _run_d3_split(model, device, loader, selected_layers)
            for split, loader in loaders.items()
        },
        "interpretation_boundary": (
            "slot allocation changes localize propagation; they do not rank a single slot's causal importance"
        ),
    }


def _run_d3_content_conditions(model, device, loaders, cfg, seed):
    normal = {
        split: _predict(model, device, loader)
        for split, loader in loaders.items()
    }
    prompt_len = int(_provider(model).prompt_len)
    rng = np.random.RandomState(int(seed))
    permutation = rng.permutation(prompt_len)
    if np.array_equal(permutation, np.arange(prompt_len)):
        permutation = np.roll(permutation, 1)
    conditions = {}
    for name, mode in (
        ("common_only", "common_only"),
        ("role_only", "role_only"),
        ("role_permuted", "role_permuted"),
    ):
        outputs = {
            split: _predict(
                model,
                device,
                loader,
                residual_content_mode=mode,
                slot_permutation=(
                    permutation.tolist() if mode == "role_permuted" else None
                ),
            )
            for split, loader in loaders.items()
        }
        conditions[name] = _condition_summary(
            outputs, normal, cfg, is_normal=False
        )
    return {
        "conditions": conditions,
        "role_permutation_seed": int(seed),
        "role_permutation": permutation.tolist(),
        "common_definition": "slot mean repeated over all Prompt slots",
        "role_definition": "per-slot residual minus the repeated slot mean",
    }


def _loo_center_map(output, *, classwise: bool):
    means = np.asarray(output["source_mu"], dtype=np.float64)
    labels = np.asarray(output["targets_global"])
    identifiers = list(output["sample_ids"])
    result = {}
    manifest = []
    for index, sample_id in enumerate(identifiers):
        mask = np.ones(means.shape[0], dtype=bool)
        mask[index] = False
        if classwise:
            mask &= labels == labels[index]
        donor_indices = np.flatnonzero(mask)
        if donor_indices.size < 1:
            raise ValueError("LOO center has no remaining donor")
        center = means[donor_indices].mean(axis=0).astype(np.float32)
        result[sample_id] = center
        manifest.append(
            {
                "target_id": sample_id,
                "target_class": int(labels[index]),
                "donor_count": int(donor_indices.size),
                "self_excluded": True,
                "label_used": bool(classwise),
            }
        )
    return result, manifest


def _wrong_class_center_map(output, seed: int):
    means = np.asarray(output["source_mu"], dtype=np.float64)
    labels = np.asarray(output["targets_global"])
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("wrong-class center requires at least two classes")
    rng = np.random.RandomState(int(seed))
    shuffled = classes.copy()
    rng.shuffle(shuffled)
    shift = int(rng.randint(1, classes.size))
    mapping = {
        int(shuffled[index]): int(shuffled[(index + shift) % classes.size])
        for index in range(classes.size)
    }
    centers = {int(cls): means[labels == cls].mean(axis=0) for cls in classes}
    result = {
        sample_id: centers[mapping[int(label)]].astype(np.float32)
        for sample_id, label in zip(output["sample_ids"], labels)
    }
    manifest = {
        "seed": int(seed),
        "mapping": {str(key): int(value) for key, value in mapping.items()},
        "fixed_point_count": int(sum(key == value for key, value in mapping.items())),
        "true_label_used": True,
    }
    return result, manifest


def _norm_match_map(donor_map, output):
    target_by_id = {
        sample_id: value
        for sample_id, value in zip(output["sample_ids"], output["source_mu"])
    }
    matched = {}
    for sample_id, donor in donor_map.items():
        donor = np.asarray(donor, dtype=np.float64)
        target = np.asarray(target_by_id[sample_id], dtype=np.float64)
        donor_norm = np.linalg.norm(donor)
        target_norm = np.linalg.norm(target)
        matched[sample_id] = (
            donor * (target_norm / max(donor_norm, 1.0e-12))
        ).astype(np.float32)
    return matched


def _manifest_hash(payload) -> str:
    encoded = json.dumps(
        _json_safe(payload), ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_replacement(model, device, loaders, maps):
    return {
        split: _predict(
            model,
            device,
            loader,
            donor_mean_by_id=maps[split],
        )
        for split, loader in loaders.items()
    }


def _run_d4(model, device, loaders, normal, cfg, wrong_class_seeds):
    report: Dict[str, Any] = {
        "format": "b3_d4_label_oracle_center_test_v1",
        "training_performed": False,
        "deployable_method": False,
        "true_test_label_used": True,
        "conditions": {},
        "manifests": {},
    }
    natural_maps: Dict[str, Dict[str, Mapping[str, np.ndarray]]] = {
        "same_class_loo_center": {},
        "global_loo_center": {},
    }
    for split, output in normal.items():
        same_map, same_manifest = _loo_center_map(output, classwise=True)
        global_map, global_manifest = _loo_center_map(output, classwise=False)
        natural_maps["same_class_loo_center"][split] = same_map
        natural_maps["global_loo_center"][split] = global_map
        report["manifests"].setdefault(split, {}).update(
            {
                "same_class_loo_center": {
                    "sha256": _manifest_hash(same_manifest),
                    "rows": same_manifest,
                },
                "global_loo_center": {
                    "sha256": _manifest_hash(global_manifest),
                    "rows": global_manifest,
                },
            }
        )
    for condition_name, maps in natural_maps.items():
        outputs = _run_replacement(model, device, loaders, maps)
        report["conditions"][condition_name] = _condition_summary(
            outputs, normal, cfg, is_normal=False
        )
        matched_maps = {
            split: _norm_match_map(maps[split], normal[split])
            for split in loaders
        }
        matched_name = condition_name + "_source_norm_matched"
        matched_outputs = _run_replacement(model, device, loaders, matched_maps)
        report["conditions"][matched_name] = _condition_summary(
            matched_outputs, normal, cfg, is_normal=False
        )
        report["conditions"][matched_name]["bounded_ratio_note"] = (
            "source norm is normalized by the current bounded-ratio residual; equality with the natural condition is therefore expected and audited"
        )
    for seed in wrong_class_seeds:
        maps = {}
        for split, output in normal.items():
            maps[split], manifest = _wrong_class_center_map(output, int(seed))
            report["manifests"].setdefault(split, {})[
                "wrong_class_seed_{}".format(seed)
            ] = manifest
        name = "wrong_class_center_seed_{}".format(seed)
        outputs = _run_replacement(model, device, loaders, maps)
        report["conditions"][name] = _condition_summary(
            outputs, normal, cfg, is_normal=False
        )
    report["valid"] = all(
        bool(item.get("valid", False)) for item in report["conditions"].values()
    )
    report["interpretation_boundary"] = {
        "same_class_helpful": "class-common information exists in mu but is not a label-free deployment mechanism",
        "global_similar": "aggregation mainly denoises rather than selects class-common structure",
        "wrong_class_similar": "center replacement is not class specific",
    }
    return report


def main() -> None:
    args = _parse_args()
    experiments = _experiments(args.experiments)
    random_seeds = _parse_ints(args.random_mlp_seeds, minimum_count=5)
    wrong_class_seeds = _parse_ints(args.wrong_class_seeds, minimum_count=3)
    if (
        args.selection_seed < 0
        or args.num_workers < 0
        or args.max_pairs < 1
        or args.role_permutation_seed < 0
    ):
        raise SystemExit("seed/worker/pair values are invalid")
    source_run = Path(args.source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(str(source_run))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to mix diagnostics into a non-empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(source_run, args.batch_size, args.num_workers)
    torch.manual_seed(int(cfg.SEED))
    np.random.seed(int(cfg.SEED))
    random.seed(int(cfg.SEED))
    identity, checkpoint = _source_identity(source_run, cfg)
    model, device, full_test_loaders = _load_model_and_loaders(
        source_run, cfg, checkpoint
    )
    layers = _selected_layers(args.selected_layers, _model_num_layers(model))
    loaders, probe_manifests = _build_loaders(
        cfg,
        full_test_loaders,
        args.scope,
        args.selection_seed,
        include_train_seen=bool({"D2", "D2G"}.intersection(experiments)),
    )
    summary: Dict[str, Any] = {
        "format": "b3_followup_diagnostics_v1",
        "experiments": list(experiments),
        "scope": args.scope,
        "selection_seed": args.selection_seed if args.scope == "probe" else None,
        "training_performed": False,
        "optimizer_created": False,
        "source": identity,
        "source_checkpoint_sha256": checkpoint_sha256(identity["checkpoint_path"]),
        "probe_manifests": probe_manifests,
        "valid": False,
    }
    _atomic_json(output_dir / "b3_followup_running.json", _json_safe(summary))
    try:
        if "D2" in experiments:
            d2 = _run_d2(
                model,
                device,
                loaders,
                args.pair_seed,
                args.max_pairs,
                random_seeds,
            )
            _atomic_json(output_dir / "B3-D2-prepass-to-mu.json", _json_safe(d2))
            summary["D2"] = {"path": "B3-D2-prepass-to-mu.json", "valid": True}
        if "D2G" in experiments:
            d2g = _run_d2g(
                model,
                device,
                loaders,
                cfg,
                args.pair_seed,
                args.max_pairs,
                random_seeds,
                summary["source_checkpoint_sha256"],
            )
            _atomic_json(
                output_dir / "B3-D2G-source-to-decision-chain.json",
                _json_safe(d2g),
            )
            summary["D2G"] = {
                "path": "B3-D2G-source-to-decision-chain.json",
                "valid": bool(d2g["valid"]),
            }
        if "D3" in experiments:
            d3_loaders = {
                key: value
                for key, value in loaders.items()
                if key in {"test_seen", "test_unseen"}
            }
            d3 = _run_d3(model, device, d3_loaders, layers)
            d3["common_role_counterfactuals"] = _run_d3_content_conditions(
                model,
                device,
                d3_loaders,
                cfg,
                args.role_permutation_seed,
            )
            _atomic_json(output_dir / "B3-D3-slot-propagation.json", _json_safe(d3))
            summary["D3"] = {"path": "B3-D3-slot-propagation.json", "valid": True}
        if "D4" in experiments:
            d4_loaders = {
                key: value
                for key, value in loaders.items()
                if key in {"test_seen", "test_unseen"}
            }
            normal = {
                split: _predict(model, device, loader)
                for split, loader in d4_loaders.items()
            }
            d4 = _run_d4(
                model,
                device,
                d4_loaders,
                normal,
                cfg,
                wrong_class_seeds,
            )
            _atomic_json(output_dir / "B3-D4-class-center-oracle.json", _json_safe(d4))
            summary["D4"] = {
                "path": "B3-D4-class-center-oracle.json",
                "valid": bool(d4["valid"]),
            }
        summary["valid"] = all(bool(summary[name]["valid"]) for name in experiments)
        summary["status"] = "completed" if summary["valid"] else "invalid"
        _atomic_json(output_dir / "b3_followup_summary.json", _json_safe(summary))
        if not summary["valid"]:
            raise RuntimeError("B3 follow-up diagnostics completed but failed validation")
    except Exception:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        _atomic_json(output_dir / "b3_followup_failure.json", _json_safe(summary))
        raise


if __name__ == "__main__":
    main()
