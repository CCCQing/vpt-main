#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import statistics
from collections import ChainMap, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from scipy.stats import t as student_t

try:
    from .artifact_io import (
        artifact_exists,
        compact_to_gzip,
        open_text_artifact,
        resolve_text_artifact,
    )
    from .probe_evidence import (
        MECHANISM_EVIDENCE,
        PROBE_CONTEXT_ONLY,
        VALIDITY_OR_IDENTITY,
        probe_record_evidence_role,
    )
except ImportError:  # Direct script execution.
    from artifact_io import (
        artifact_exists,
        compact_to_gzip,
        open_text_artifact,
        resolve_text_artifact,
    )
    from probe_evidence import (
        MECHANISM_EVIDENCE,
        PROBE_CONTEXT_ONLY,
        VALIDITY_OR_IDENTITY,
        probe_record_evidence_role,
    )


TASK_METRICS = (
    "gzsl_seen",
    "gzsl_unseen",
    "gzsl_h",
    "ausuc",
    "raw_to_oracle_gain",
    "diagnostic_peak_epoch",
    "oracle_peak_gamma",
)

TRAJECTORY_REQUIRED_FIELDS = (
    "train_loss",
    "test_seen_nll",
    "test_unseen_nll",
    "gzsl_seen",
    "gzsl_unseen",
    "gzsl_h",
)
TRAJECTORY_FIELD_MAP = {
    ("train_epoch", "train", "loss"): "train_loss",
    ("train_epoch", "train", "lr"): "train_lr",
    ("classification", "test_seen", "nll"): "test_seen_nll",
    ("classification", "test_unseen", "nll"): "test_unseen_nll",
    ("classification", "test_seen", "top1"): "test_seen_top1",
    ("classification", "test_unseen", "top1"): "test_unseen_top1",
    ("classification", "train_eval_seen", "nll"): "train_eval_seen_nll",
    ("classification", "train_eval_seen", "top1"): "train_eval_seen_top1",
    ("classification", "test_gzsl", "gzsl_seen"): "gzsl_seen",
    ("classification", "test_gzsl", "gzsl_unseen"): "gzsl_unseen",
    ("classification", "test_gzsl", "gzsl_h"): "gzsl_h",
    ("prediction_health", "test_seen", "true_class_rank_mean"): "test_seen_true_class_rank_mean",
    ("prediction_health", "test_unseen", "true_class_rank_mean"): "test_unseen_true_class_rank_mean",
    ("prediction_health", "train_eval_seen", "true_class_rank_mean"): "train_eval_seen_true_class_rank_mean",
    ("prediction_health", "test_seen", "wrong_domain_prediction_rate"): "test_seen_wrong_domain_prediction_rate",
    ("prediction_health", "test_unseen", "wrong_domain_prediction_rate"): "test_unseen_wrong_domain_prediction_rate",
    ("prediction_health", "test_seen", "true_class_margin_mean"): "test_seen_true_class_margin_mean",
    ("prediction_health", "test_unseen", "true_class_margin_mean"): "test_unseen_true_class_margin_mean",
    ("prediction_health", "test_seen", "seen_probability_mass_mean"): "test_seen_seen_probability_mass_mean",
    ("prediction_health", "test_unseen", "seen_probability_mass_mean"): "test_unseen_seen_probability_mass_mean",
    ("prediction_health", "test_seen", "seen_unseen_logit_margin_mean"): "test_seen_seen_unseen_logit_margin_mean",
    ("prediction_health", "test_unseen", "seen_unseen_logit_margin_mean"): "test_unseen_seen_unseen_logit_margin_mean",
    ("prediction_health", "test_seen", "entropy_mean"): "test_seen_entropy_mean",
    ("prediction_health", "test_unseen", "entropy_mean"): "test_unseen_entropy_mean",
    ("prediction_health", "test_seen", "confidence_incorrect"): "test_seen_confidence_incorrect",
    ("prediction_health", "test_unseen", "confidence_incorrect"): "test_unseen_confidence_incorrect",
    ("class_error", "test_seen", "bottom_k_class_mean"): "test_seen_bottom_k_class_mean",
    ("class_error", "test_unseen", "bottom_k_class_mean"): "test_unseen_bottom_k_class_mean",
    ("class_error", "test_seen", "max_prediction_share"): "test_seen_max_prediction_share",
    ("class_error", "test_unseen", "max_prediction_share"): "test_unseen_max_prediction_share",
    ("calibration_profile", "test_gzsl", "ausuc"): "test_gzsl_ausuc",
    ("calibration_profile", "test_gzsl", "raw_to_oracle_gain"): "test_gzsl_raw_to_oracle_gain",
    ("calibration_profile", "test_gzsl", "oracle_peak_gamma"): "test_gzsl_oracle_peak_gamma",
    ("prompt_parameter_health", "train", "prompt_param_norm"): "prompt_param_norm",
    ("prompt_parameter_health", "train", "prompt_relative_update"): "prompt_relative_update",
    ("prompt_parameter_health", "train", "distance_from_initialization"): "prompt_distance_from_initialization",
    ("prompt_parameter_health", "train", "prompt_effective_rank"): "prompt_effective_rank",
    ("prompt_parameter_health", "train", "token_pair_cosine_mean"): "prompt_token_pair_cosine_mean",
    ("prompt_parameter_health", "train", "token_pair_cosine_max"): "prompt_token_pair_cosine_max",
    ("prompt_parameter_health", "train", "prompt_grad_norm"): "prompt_grad_norm_last_batch",
    ("prompt_parameter_health", "train", "epoch_parameter_step_norm"): "prompt_epoch_parameter_step_norm",
    ("prompt_parameter_health", "train", "epoch_update_to_weight_ratio"): "prompt_epoch_update_to_weight_ratio",
    ("prompt_parameter_health", "train", "cross_epoch_prompt_cosine"): "prompt_cross_epoch_cosine",
}
TRAJECTORY_DERIVED_OPTIONAL_FIELDS = (
    "test_seen_minus_train_eval_nll_gap",
    "train_eval_minus_test_seen_top1_gap",
    "test_seen_minus_train_eval_true_class_rank_gap",
)
TRAJECTORY_OPTIONAL_FIELDS = tuple(dict.fromkeys(
    (*TRAJECTORY_FIELD_MAP.values(), *TRAJECTORY_DERIVED_OPTIONAL_FIELDS)
))
TRAJECTORY_ANCHOR_FIELDS = (
    "gzsl_seen",
    "gzsl_unseen",
    "gzsl_h",
    "test_seen_true_class_rank_mean",
    "test_unseen_true_class_rank_mean",
    "test_unseen_wrong_domain_prediction_rate",
    "test_gzsl_ausuc",
    "test_seen_bottom_k_class_mean",
    "test_unseen_bottom_k_class_mean",
    "prompt_param_norm",
    "prompt_relative_update",
    "prompt_distance_from_initialization",
    "prompt_effective_rank",
    "prompt_token_pair_cosine_mean",
    "prompt_epoch_parameter_step_norm",
    "prompt_epoch_update_to_weight_ratio",
    "prompt_cross_epoch_cosine",
    *TRAJECTORY_DERIVED_OPTIONAL_FIELDS,
)
TRAJECTORY_WINDOW_FRACTION = 0.2
TRAJECTORY_MIN_WINDOW_EPOCHS = 2
TRAJECTORY_MIN_COMPLETE_EPOCHS = 4
TRAJECTORY_ACCURACY_TOLERANCE = 0.005
TRAJECTORY_LOSS_ABS_TOLERANCE = 0.01
TRAJECTORY_LOSS_REL_TOLERANCE = 0.01
TRAJECTORY_FORMATION_SUFFIX_RATIO = 0.8
TRAJECTORY_FORMATION_MIN_WINDOWS = 3
TRAJECTORY_LATE_LR_FRACTION = 0.25


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="METHOD=OUTPUT_DIR; repeat for every seed/run")
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", default="0,1,2")
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="Recompute trajectory/cross-experiment inputs without parsing or rewriting large fixed-probe mechanism summaries.",
    )
    parser.add_argument(
        "--compact-probe-metrics",
        action="store_true",
        help="After every compact summary is committed, gzip completed-run probe_metrics.csv files and remove the verified originals.",
    )
    parser.add_argument(
        "--compact-module-effect-json",
        action="store_true",
        help="After summary generation, gzip completed-run module_effect JSON details and remove the verified originals.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _classification_by_epoch(path: Path) -> Dict[int, Dict[str, float]]:
    epochs: Dict[int, Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("namespace") != "classification" or row.get("split") != "test_gzsl":
                continue
            epoch = int(float(row.get("epoch") or 0))
            value = _float(row.get("value"))
            if value is not None:
                epochs[epoch][str(row.get("metric"))] = value
    return dict(epochs)


def _generalization_trajectory_by_epoch(path: Path) -> List[Dict[str, Any]]:
    epochs: Dict[int, Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("namespace") or ""),
                str(row.get("split") or ""),
                str(row.get("metric") or ""),
            )
            output_field = TRAJECTORY_FIELD_MAP.get(key)
            if output_field is None:
                continue
            value = _float(row.get("value"))
            if value is None:
                continue
            epoch = int(float(row.get("epoch") or 0))
            epochs[epoch][output_field] = float(value)
    rows = []
    for epoch in sorted(epochs):
        values = epochs[epoch]
        if "train_eval_seen_nll" in values and "test_seen_nll" in values:
            values["test_seen_minus_train_eval_nll_gap"] = float(
                values["test_seen_nll"] - values["train_eval_seen_nll"]
            )
        if "train_eval_seen_top1" in values and "test_seen_top1" in values:
            values["train_eval_minus_test_seen_top1_gap"] = float(
                values["train_eval_seen_top1"] - values["test_seen_top1"]
            )
        if (
            "train_eval_seen_true_class_rank_mean" in values
            and "test_seen_true_class_rank_mean" in values
        ):
            values["test_seen_minus_train_eval_true_class_rank_gap"] = float(
                values["test_seen_true_class_rank_mean"]
                - values["train_eval_seen_true_class_rank_mean"]
            )
        missing = [field for field in TRAJECTORY_REQUIRED_FIELDS if field not in values]
        rows.append({
            "epoch": int(epoch),
            **values,
            "complete": not missing,
            "missing_fields": ";".join(missing),
        })
    return rows


def _prompt_layer_trajectory_by_epoch(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(r"layer_(\d+)\.(.+)")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("namespace") or "") != "prompt_parameter_health":
                continue
            match = pattern.fullmatch(str(row.get("metric") or ""))
            value = _float(row.get("value"))
            if match is None or value is None:
                continue
            rows.append({
                "epoch": int(float(row.get("epoch") or 0)),
                "layer": int(match.group(1)),
                "metric": str(match.group(2)),
                "value": float(value),
                "split": str(row.get("split") or "train"),
            })
    return sorted(rows, key=lambda row: (int(row["epoch"]), int(row["layer"]), str(row["metric"])))


def _read_training_context(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    data = payload.get("DATA", {}) or {}
    xlsa = data.get("XLSA", {}) or {}
    model = payload.get("MODEL", {}) or {}
    prompt = model.get("PROMPT", {}) or {}
    solver = payload.get("SOLVER", {}) or {}
    rsim = solver.get("RSIM", {}) or {}
    condition_fields = {
        "data.name": data.get("NAME"),
        "data.feature": data.get("FEATURE"),
        "data.protocol_mode": xlsa.get("PROTOCOL_MODE"),
        "model.type": model.get("TYPE"),
        "model.transfer_type": model.get("TRANSFER_TYPE"),
        "prompt.backend": prompt.get("BACKEND"),
        "prompt.deep": bool(prompt.get("DEEP", False)),
        "prompt.num_tokens": int(prompt.get("NUM_TOKENS", 0) or 0),
        "solver.optimizer": solver.get("OPTIMIZER"),
        "solver.base_lr": _float(solver.get("BASE_LR")),
        "solver.weight_decay": _float(solver.get("WEIGHT_DECAY")),
        "solver.scheduler": solver.get("SCHEDULER"),
        "solver.total_epoch": int(solver.get("TOTAL_EPOCH", 0) or 0),
        "solver.warmup_epoch": int(solver.get("WARMUP_EPOCH", 0) or 0),
        "loss.rsim_align_mode": rsim.get("ALIGN_MODE"),
        "loss.rsim_align_weight": _float(rsim.get("ALIGN_WEIGHT")),
        "loss.attr_weight": _float(solver.get("LOSS_ATTR_WEIGHT")),
        "loss.prompt_kl_weight": _float(solver.get("LOSS_PROMPT_KL_WEIGHT")),
        "loss.semantic_mediation_weight": _float(solver.get("LOSS_SEM_MED_WEIGHT")),
        "loss.semantic_prompt_value_weight": _float(solver.get("LOSS_SPV_WEIGHT")),
    }
    canonical = json.dumps(
        condition_fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        "warmup_epoch": int(solver.get("WARMUP_EPOCH", 0) or 0),
        "configured_total_epoch": int(solver.get("TOTAL_EPOCH", 0) or 0),
        "scheduler": str(solver.get("SCHEDULER", "unknown")),
        "base_lr": _float(solver.get("BASE_LR")),
        "condition_fields": condition_fields,
        "condition_fingerprint": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


def _trajectory_window_size(epoch_count: int) -> int:
    proposed = max(
        TRAJECTORY_MIN_WINDOW_EPOCHS,
        int(math.ceil(float(epoch_count) * TRAJECTORY_WINDOW_FRACTION)),
    )
    return min(proposed, max(1, int(epoch_count) // 2))


def _trajectory_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(statistics.mean(float(row[field]) for row in rows))


def _trajectory_loss_tolerance(reference: float) -> float:
    return max(
        TRAJECTORY_LOSS_ABS_TOLERANCE,
        TRAJECTORY_LOSS_REL_TOLERANCE * abs(float(reference)),
    )


def _summarize_generalization_trajectory(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_completed: bool,
    training_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    observed = sorted(rows, key=lambda row: int(row["epoch"]))
    complete = [row for row in observed if bool(row.get("complete"))]
    base = {
        "valid": False,
        "status": "no_trajectory_data" if not observed else "insufficient_complete_epochs",
        "observed_epoch_count": int(len(observed)),
        "complete_epoch_count": int(len(complete)),
        "analysis_role": "diagnostic_only",
        "checkpoint_selection_allowed": False,
    }
    if len(complete) < TRAJECTORY_MIN_COMPLETE_EPOCHS:
        return base
    complete_epochs = [int(row["epoch"]) for row in complete]
    expected_epochs = list(range(complete_epochs[0], complete_epochs[-1] + 1))
    if complete_epochs != expected_epochs or len(complete) != len(observed):
        base["status"] = "incomplete_or_noncontiguous_epoch_fields"
        return base

    window_size = _trajectory_window_size(len(complete))
    early_rows = complete[:window_size]
    late_rows = complete[-window_size:]
    rolling = [complete[index:index + window_size] for index in range(len(complete) - window_size + 1)]
    best_h_rows = max(rolling, key=lambda window: _trajectory_mean(window, "gzsl_h"))

    summary: Dict[str, Any] = {
        **base,
        "valid": bool(run_completed),
        "status": "completed" if run_completed else "run_not_completed",
        "first_epoch": complete_epochs[0],
        "final_epoch": complete_epochs[-1],
        "window_size": int(window_size),
        "early_window_start": int(early_rows[0]["epoch"]),
        "early_window_end": int(early_rows[-1]["epoch"]),
        "late_window_start": int(late_rows[0]["epoch"]),
        "late_window_end": int(late_rows[-1]["epoch"]),
        "diagnostic_best_h_window_start": int(best_h_rows[0]["epoch"]),
        "diagnostic_best_h_window_end": int(best_h_rows[-1]["epoch"]),
    }
    context = dict(training_context or {})
    warmup_end = max(0, int(context.get("warmup_epoch", 0) or 0))
    summary.update({
        "warmup_end_epoch": warmup_end if warmup_end > 0 else None,
        "configured_total_epoch": int(context.get("configured_total_epoch", 0) or 0) or None,
        "scheduler": str(context.get("scheduler", "unknown")),
        "base_lr": _float(context.get("base_lr")),
    })
    lr_rows = [row for row in complete if _float(row.get("train_lr")) is not None]
    if lr_rows:
        peak_lr = max(float(row["train_lr"]) for row in lr_rows)
        lr_tolerance = max(abs(peak_lr) * 1e-9, 1e-15)
        peak_epochs = [
            int(row["epoch"])
            for row in lr_rows
            if abs(float(row["train_lr"]) - peak_lr) <= lr_tolerance
        ]
        peak_start = min(peak_epochs)
        peak_end = max(peak_epochs)
        late_threshold = peak_lr * TRAJECTORY_LATE_LR_FRACTION
        late_start = None
        for index, row in enumerate(lr_rows):
            epoch = int(row["epoch"])
            if epoch <= peak_end or float(row["train_lr"]) > late_threshold + lr_tolerance:
                continue
            suffix = lr_rows[index:]
            if all(float(item["train_lr"]) <= late_threshold + lr_tolerance for item in suffix):
                late_start = epoch
                break
        summary.update({
            "lr_phase_status": "observed",
            "peak_lr": float(peak_lr),
            "peak_lr_epoch_start": int(peak_start),
            "peak_lr_epoch_end": int(peak_end),
            "late_low_lr_threshold": float(late_threshold),
            "late_low_lr_start_epoch": late_start,
            "warmup_lr_observation_available": bool(
                warmup_end > 0 and any(int(row["epoch"]) == warmup_end for row in lr_rows)
            ),
        })
        for row in complete:
            epoch = int(row["epoch"])
            if warmup_end > 0 and epoch <= warmup_end:
                row["lr_phase"] = "warmup"
            elif late_start is not None and epoch >= int(late_start):
                row["lr_phase"] = "late_low_lr"
            elif epoch <= peak_end:
                row["lr_phase"] = "post_warmup_peak_lr"
            else:
                row["lr_phase"] = "regular_decay"
    else:
        summary.update({
            "lr_phase_status": "missing_train_lr",
            "peak_lr": None,
            "peak_lr_epoch_start": None,
            "peak_lr_epoch_end": None,
            "late_low_lr_threshold": None,
            "late_low_lr_start_epoch": None,
            "warmup_lr_observation_available": False,
        })

    rolling_h = [_trajectory_mean(window, "gzsl_h") for window in rolling]
    diagnostic_best_h = max(rolling_h)
    formation_threshold = diagnostic_best_h - TRAJECTORY_ACCURACY_TOLERANCE
    formation_index = None
    formation_ratio = None
    for index, value in enumerate(rolling_h):
        suffix = rolling_h[index:]
        if len(suffix) < TRAJECTORY_FORMATION_MIN_WINDOWS or value < formation_threshold:
            continue
        ratio = float(sum(item >= formation_threshold for item in suffix) / len(suffix))
        if ratio >= TRAJECTORY_FORMATION_SUFFIX_RATIO:
            formation_index = index
            formation_ratio = ratio
            break
    if formation_index is None:
        summary.update({
            "formation_status": "not_formed_within_observed_epochs",
            "self_relative_formation_epoch": None,
            "persistence_ratio": None,
            "formation_window_count": 0,
        })
    else:
        formation_window = rolling[formation_index]
        summary.update({
            "formation_status": "formed",
            "self_relative_formation_epoch": int(formation_window[-1]["epoch"]),
            "persistence_ratio": float(formation_ratio),
            "formation_window_count": int(len(rolling_h) - formation_index),
        })
    summary.update({
        "formation_h_threshold": float(formation_threshold),
        "formation_suffix_ratio_required": float(TRAJECTORY_FORMATION_SUFFIX_RATIO),
        "formation_min_windows_required": int(TRAJECTORY_FORMATION_MIN_WINDOWS),
        "formation_epoch_semantics": "rolling_window_end_epoch_relative_to_own_diagnostic_best_h",
    })

    rows_by_epoch = {int(row["epoch"]): row for row in complete}
    anchor_epochs = {
        "warmup_end": warmup_end if warmup_end in rows_by_epoch else None,
        "formation": summary.get("self_relative_formation_epoch"),
        "final": complete_epochs[-1],
    }
    anchors = {}
    for name, epoch in anchor_epochs.items():
        if epoch is None or int(epoch) not in rows_by_epoch:
            anchors[name] = {"available": False, "epoch": epoch, "metrics": {}}
            continue
        source = rows_by_epoch[int(epoch)]
        anchors[name] = {
            "available": True,
            "epoch": int(epoch),
            "metrics": {
                field: float(source[field])
                for field in TRAJECTORY_ANCHOR_FIELDS
                if _float(source.get(field)) is not None
            },
        }
    summary["decision_and_prompt_anchors"] = anchors
    for field in TRAJECTORY_REQUIRED_FIELDS:
        early_mean = _trajectory_mean(early_rows, field)
        late_mean = _trajectory_mean(late_rows, field)
        best_h_window_mean = _trajectory_mean(best_h_rows, field)
        summary[f"early_{field}_mean"] = early_mean
        summary[f"late_{field}_mean"] = late_mean
        summary[f"delta_{field}"] = late_mean - early_mean
        summary[f"diagnostic_best_h_window_{field}_mean"] = best_h_window_mean
        summary[f"best_h_window_to_late_{field}_delta"] = late_mean - best_h_window_mean

    train_decreased = summary["delta_train_loss"] <= -_trajectory_loss_tolerance(
        summary["early_train_loss_mean"]
    )
    seen_improved = summary["delta_gzsl_seen"] >= TRAJECTORY_ACCURACY_TOLERANCE
    seen_degraded = summary["delta_gzsl_seen"] <= -TRAJECTORY_ACCURACY_TOLERANCE
    unseen_improved = summary["delta_gzsl_unseen"] >= TRAJECTORY_ACCURACY_TOLERANCE
    unseen_degraded = summary["delta_gzsl_unseen"] <= -TRAJECTORY_ACCURACY_TOLERANCE
    h_improved = summary["delta_gzsl_h"] >= TRAJECTORY_ACCURACY_TOLERANCE
    best_to_late_h_drop = -summary["best_h_window_to_late_gzsl_h_delta"]
    train_after_best_decreased = summary[
        "best_h_window_to_late_train_loss_delta"
    ] <= -_trajectory_loss_tolerance(
        summary["diagnostic_best_h_window_train_loss_mean"]
    )
    seen_nll_after_best_worsened = summary[
        "best_h_window_to_late_test_seen_nll_delta"
    ] >= _trajectory_loss_tolerance(
        summary["diagnostic_best_h_window_test_seen_nll_mean"]
    )
    unseen_nll_after_best_worsened = summary[
        "best_h_window_to_late_test_unseen_nll_delta"
    ] >= _trajectory_loss_tolerance(
        summary["diagnostic_best_h_window_test_unseen_nll_mean"]
    )
    seen_after_best_degraded = summary[
        "best_h_window_to_late_gzsl_seen_delta"
    ] <= -TRAJECTORY_ACCURACY_TOLERANCE
    unseen_after_best_degraded = summary[
        "best_h_window_to_late_gzsl_unseen_delta"
    ] <= -TRAJECTORY_ACCURACY_TOLERANCE

    flags = {
        "joint_generalization_gain": bool(
            train_decreased and seen_improved and unseen_improved and h_improved
        ),
        "seen_specialization_tradeoff": bool(
            train_decreased and seen_improved and unseen_degraded
        ),
        "late_generalization_degradation_candidate": bool(
            train_after_best_decreased
            and best_to_late_h_drop >= TRAJECTORY_ACCURACY_TOLERANCE
            and (
                seen_nll_after_best_worsened
                or unseen_nll_after_best_worsened
                or seen_after_best_degraded
                or unseen_after_best_degraded
            )
        ),
        "optimization_without_h_gain": bool(train_decreased and not h_improved),
        "seen_degraded": bool(seen_degraded),
        "unseen_degraded": bool(unseen_degraded),
    }
    if flags["late_generalization_degradation_candidate"]:
        primary_pattern = "late_generalization_degradation_candidate"
    elif flags["seen_specialization_tradeoff"]:
        primary_pattern = "seen_specialization_tradeoff"
    elif flags["joint_generalization_gain"]:
        primary_pattern = "joint_generalization_gain"
    elif flags["optimization_without_h_gain"]:
        primary_pattern = "optimization_without_h_gain"
    else:
        primary_pattern = "mixed_or_flat"
    summary.update(flags)
    summary["primary_pattern"] = primary_pattern
    summary["diagnostic_best_h_window_mean"] = summary[
        "diagnostic_best_h_window_gzsl_h_mean"
    ]
    summary["diagnostic_best_h_window_to_late_drop"] = float(best_to_late_h_drop)
    return summary


def _latest_calibration(run_dir: Path) -> Dict[str, float]:
    paths = sorted((run_dir / "diagnostics" / "calibration_profile").glob("epoch_*.json"))
    if not paths:
        return {}
    summary = _read_json(paths[-1]).get("summary", {})
    return {str(name): float(value) for name, value in summary.items() if _float(value) is not None}


def _mechanism_key(row: Mapping[str, Any]) -> str:
    return "|".join((
        f"checkpoint={row.get('checkpoint_id') or 'unknown'}",
        f"probe_id={row.get('probe_id') or 'unknown'}",
        f"probe_selection_seed={row.get('selection_seed') if row.get('selection_seed') not in (None, '') else 'unknown'}",
        f"probe_manifest_sha256={row.get('probe_manifest_sha256') or 'unknown'}",
        f"split={row.get('split') or 'unknown'}",
        f"condition={row.get('condition') or 'normal'}",
        f"domain={row.get('domain') or 'unknown'}",
        f"entity_type={row.get('entity_type') or 'unknown'}",
        f"entity_id={row.get('entity_id') or 'unknown'}",
        f"metric={row.get('metric') or 'unknown'}",
    ))


def _latest_probe_records(run_dir: Path) -> List[Dict[str, Any]]:
    path = resolve_text_artifact(
        run_dir / "diagnostics" / "probe_metrics.csv"
    )
    if not artifact_exists(path):
        return []
    records = []
    with open_text_artifact(
        path, "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            value = _float(row.get("value"))
            if value is not None:
                record = dict(row)
                record["value"] = float(value)
                selection_seed = _float(row.get("selection_seed"))
                if selection_seed is None:
                    match = re.search(r"(?:^|-)seed(\d+)(?:-|$)", str(row.get("probe_id") or ""))
                    selection_seed = int(match.group(1)) if match else None
                record["selection_seed"] = None if selection_seed is None else int(selection_seed)
                records.append(record)
    return records


def _latest_probe_value_groups(run_dir: Path) -> Dict[str, Dict[str, float]]:
    path = resolve_text_artifact(
        run_dir / "diagnostics" / "probe_metrics.csv"
    )
    groups = {
        MECHANISM_EVIDENCE: {},
        PROBE_CONTEXT_ONLY: {},
        VALIDITY_OR_IDENTITY: {},
    }
    if not artifact_exists(path):
        return groups
    with open_text_artifact(
        path, "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            value = _float(row.get("value"))
            if value is None:
                continue
            record = dict(row)
            selection_seed = _float(row.get("selection_seed"))
            if selection_seed is None:
                match = re.search(
                    r"(?:^|-)seed(\d+)(?:-|$)",
                    str(row.get("probe_id") or ""),
                )
                selection_seed = int(match.group(1)) if match else None
            record["selection_seed"] = (
                None if selection_seed is None else int(selection_seed)
            )
            role = probe_record_evidence_role(record)
            if role in groups:
                groups[role][_mechanism_key(record)] = float(value)
    return groups


def _latest_probe_values(
    records: Sequence[Mapping[str, Any]],
    evidence_role: str,
) -> Dict[str, float]:
    return {
        _mechanism_key(row): float(row["value"])
        for row in records
        if _float(row.get("value")) is not None
        and probe_record_evidence_role(row) == evidence_role
    }


def _latest_probe_mechanisms(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return _latest_probe_values(records, MECHANISM_EVIDENCE)


def _latest_epoch_mechanisms(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    selected = {
        "prompt_parameter_health": {
            "prompt_relative_update", "prompt_effective_rank", "prompt_param_norm",
            "prompt_grad_norm", "distance_from_initialization",
        },
        "prediction_health": {
            "seen_unseen_logit_margin_mean", "seen_probability_mass_mean",
            "wrong_domain_prediction_rate", "true_class_margin_mean",
            "true_class_rank_mean", "entropy_mean", "confidence_incorrect",
        },
        "class_error": {"bottom_k_class_mean", "max_prediction_share"},
    }
    latest: Dict[Tuple[str, str, str], Tuple[int, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            namespace = str(row.get("namespace") or "")
            metric = str(row.get("metric") or "")
            layer_match = (
                re.fullmatch(r"layer_(\d+)\.(.+)", metric)
                if namespace == "prompt_parameter_health"
                else None
            )
            if namespace not in selected or (
                metric not in selected[namespace] and layer_match is None
            ):
                continue
            value = _float(row.get("value"))
            if value is None:
                continue
            epoch = int(float(row.get("epoch") or 0))
            identity = (str(row.get("split") or "train"), namespace, metric)
            if identity not in latest or epoch >= latest[identity][0]:
                latest[identity] = (epoch, value)
    result = {}
    for (split, namespace, metric), (epoch, value) in latest.items():
        layer_match = (
            re.fullmatch(r"layer_(\d+)\.(.+)", metric)
            if namespace == "prompt_parameter_health"
            else None
        )
        entity_type = "layer" if layer_match else (
            "module" if namespace == "prompt_parameter_health" else "split"
        )
        entity_id = f"layer_{layer_match.group(1)}" if layer_match else (
            "prompt" if namespace == "prompt_parameter_health" else "all"
        )
        output_metric = layer_match.group(2) if layer_match else metric
        result["|".join((
            f"checkpoint=epoch_{epoch:04d}",
            f"split={split}",
            "condition=normal",
            f"domain={namespace}",
            f"entity_type={entity_type}",
            f"entity_id={entity_id}",
            f"metric={output_metric}",
        ))] = value
    return result


def _comparability(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "diagnostics" / "comparability.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return {
        "run_identity": payload.get("run_identity", {}).get("sha256"),
        "shared_condition_fingerprint": payload.get("shared_condition_fingerprint", {}).get("sha256"),
    }


def _stage(method: str) -> Optional[str]:
    match = re.search(r"(?:^|[^A-Z0-9])A([012])(?:[^0-9]|$)", str(method).upper())
    return f"A{match.group(1)}" if match else None


def _resolve_run_dir(run_dir: Path) -> Path:
    if (run_dir / "monitor_runtime_summary.json").exists():
        return run_dir
    candidates = []
    for runtime_path in run_dir.rglob("monitor_runtime_summary.json"):
        try:
            runtime = _read_json(runtime_path)
        except (OSError, ValueError):
            continue
        if str(runtime.get("status")) == "completed":
            candidates.append(runtime_path.parent)
    return candidates[0] if len(candidates) == 1 else run_dir


def load_run(
    method: str,
    run_dir: Path,
    *,
    include_probe_records: bool = True,
) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_dir)
    runtime_path = run_dir / "monitor_runtime_summary.json"
    epoch_path = run_dir / "metrics_epoch.csv"
    row: Dict[str, Any] = {
        "method": method,
        "stage": _stage(method),
        "run_dir": str(run_dir),
        "seed": None,
        "status": "missing",
        "failed": True,
        "mechanisms": {},
        "probe_records": [],
        "generalization_trajectory_epochs": [],
        "prompt_layer_trajectory": [],
        "generalization_trajectory_summary": {
            "valid": False,
            "status": "missing_run_artifacts",
            "analysis_role": "diagnostic_only",
            "checkpoint_selection_allowed": False,
        },
    }
    if not runtime_path.exists() or not epoch_path.exists():
        return row
    runtime = _read_json(runtime_path)
    row["seed"] = runtime.get("seed")
    row["run_id"] = runtime.get("run_id")
    row["session_id"] = runtime.get("session_id")
    row["status"] = str(runtime.get("status", "unknown"))
    row["failed"] = row["status"] != "completed"
    trajectory_epochs = _generalization_trajectory_by_epoch(epoch_path)
    row["generalization_trajectory_epochs"] = trajectory_epochs
    row["prompt_layer_trajectory"] = _prompt_layer_trajectory_by_epoch(epoch_path)
    training_context = _read_training_context(run_dir)
    row["training_context"] = training_context
    row["generalization_trajectory_summary"] = _summarize_generalization_trajectory(
        trajectory_epochs,
        run_completed=row["status"] == "completed",
        training_context=training_context,
    )
    epochs = _classification_by_epoch(epoch_path)
    complete = {
        epoch: values for epoch, values in epochs.items()
        if {"gzsl_seen", "gzsl_unseen", "gzsl_h"}.issubset(values)
    }
    if not complete:
        row["failed"] = True
        return row
    final_epoch = max(complete)
    final = complete[final_epoch]
    row.update({
        "final_epoch": int(final_epoch),
        "gzsl_seen": float(final["gzsl_seen"]),
        "gzsl_unseen": float(final["gzsl_unseen"]),
        "gzsl_h": float(final["gzsl_h"]),
    })
    peak_epoch, peak_values = max(complete.items(), key=lambda item: item[1]["gzsl_h"])
    row["diagnostic_peak_epoch"] = int(peak_epoch)
    row["diagnostic_peak_h"] = float(peak_values["gzsl_h"])
    calibration = _latest_calibration(run_dir)
    for name in ("ausuc", "raw_to_oracle_gain", "oracle_peak_gamma"):
        if name in calibration:
            row[name] = calibration[name]
    if include_probe_records:
        probe_groups = _latest_probe_value_groups(run_dir)
        mechanisms = dict(probe_groups[MECHANISM_EVIDENCE])
        mechanisms.update(_latest_epoch_mechanisms(epoch_path))
        row["mechanisms"] = mechanisms
        row["probe_context"] = probe_groups[PROBE_CONTEXT_ONLY]
        row["probe_validity"] = probe_groups[VALIDITY_OR_IDENTITY]
    row.update(_comparability(run_dir))
    row["prompt_mechanisms_status"] = "not_applicable" if row.get("stage") == "A0" else "applicable"
    return row


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    items = [float(value) for value in values if _float(value) is not None]
    if not items:
        return {
            "count": 0, "values": [], "mean": None, "std": None, "median": None,
            "min": None, "max": None, "ci95": None, "ci95_low": None,
            "ci95_high": None, "ci95_method": "student_t", "positive_count": 0,
            "negative_count": 0, "zero_count": 0, "direction": "empty",
            "direction_consistency": None,
        }
    std = statistics.stdev(items) if len(items) > 1 else 0.0
    mean = statistics.mean(items)
    ci95 = (
        float(student_t.ppf(0.975, df=len(items) - 1) * std / math.sqrt(len(items)))
        if len(items) > 1 else None
    )
    positive_count = sum(value > 0.0 for value in items)
    negative_count = sum(value < 0.0 for value in items)
    zero_count = len(items) - positive_count - negative_count
    dominant_count = max(positive_count, negative_count, zero_count)
    if positive_count == len(items):
        direction = "positive"
    elif negative_count == len(items):
        direction = "negative"
    elif zero_count == len(items):
        direction = "zero"
    else:
        direction = "mixed"
    return {
        "count": len(items),
        "values": items,
        "mean": mean,
        "std": std,
        "median": statistics.median(items),
        "min": min(items),
        "max": max(items),
        "ci95": ci95,
        "ci95_low": None if ci95 is None else float(mean - ci95),
        "ci95_high": None if ci95 is None else float(mean + ci95),
        "ci95_method": "student_t",
        "positive_count": positive_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "direction": direction,
        "direction_consistency": float(dominant_count / len(items)),
    }


def _flatten_run(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {
            "mechanisms",
            "probe_context",
            "probe_validity",
            "probe_records",
            "generalization_trajectory_epochs",
            "generalization_trajectory_summary",
            "prompt_layer_trajectory",
            "training_context",
        }
    }


@contextmanager
def _atomic_csv_writer(path: Path, fieldnames: Sequence[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    handle = None
    try:
        if path.suffix == ".gz":
            raw_handle = temp_path.open("wb")
            gzip_handle = gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_handle,
                mtime=0,
            )
            handle = io.TextIOWrapper(
                gzip_handle, encoding="utf-8", newline=""
            )
        else:
            raw_handle = None
            handle = temp_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        yield writer
        handle.flush()
        handle.close()
        handle = None
        if raw_handle is not None and not raw_handle.closed:
            raw_handle.close()
        os.replace(temp_path, path)
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        if temp_path.exists():
            temp_path.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    fields = list(fieldnames) if fieldnames is not None else sorted({str(key) for row in rows for key in row})
    with _atomic_csv_writer(path, fields) as writer:
        for row in rows:
            writer.writerow({name: row.get(name) for name in fields})


NAMED_SUMMARY_FIELDS = (
    "evidence_role",
    "method",
    "metric_key",
    "count",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "ci95",
    "ci95_low",
    "ci95_high",
    "ci95_method",
    "positive_count",
    "negative_count",
    "zero_count",
    "direction",
    "direction_consistency",
    "seed_values_json",
)

PAIRED_MECHANISM_DELTA_FIELDS = (
    "pair",
    "method",
    "reference_method",
    "seed",
    "metric_key",
    "delta",
)

PAIRED_MECHANISM_SUMMARY_FIELDS = (
    "pair",
    "method",
    "reference_method",
    "metric_key",
    "count",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "ci95",
    "ci95_low",
    "ci95_high",
    "ci95_method",
    "positive_count",
    "negative_count",
    "zero_count",
    "direction",
    "direction_consistency",
    "positive_delta_ratio",
    "effect_size",
    "seed_values_json",
)


def _method_summaries(
    by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    named_summary_writer=None,
    retain_named_payload: bool = True,
):
    csv_rows = []
    payload = {}
    for method, rows in sorted(by_method.items()):
        tasks = {}
        for name in TASK_METRICS:
            values_by_seed = {
                str(int(row["seed"])): float(row[name])
                for row in rows
                if row.get("seed") is not None and _float(row.get(name)) is not None
            }
            tasks[name] = {
                **_summary(values_by_seed.values()),
                "seed_values": values_by_seed,
            }
        named_payloads = {}
        evidence_inventory = {}
        for evidence_role, field in (
            ("mechanism_evidence", "mechanisms"),
            ("probe_context_only", "probe_context"),
            ("validity_or_identity", "probe_validity"),
        ):
            names = sorted({name for row in rows for name in row.get(field, {})})
            retained = {}
            for name in names:
                values_by_seed = {
                    str(int(row["seed"])): float(row[field][name])
                    for row in rows
                    if row.get("seed") is not None
                    and _float(row.get(field, {}).get(name)) is not None
                }
                stats = {
                    **_summary(values_by_seed.values()),
                    "seed_values": values_by_seed,
                }
                if named_summary_writer is not None:
                    named_summary_writer.writerow({
                        "evidence_role": evidence_role,
                        "method": method,
                        "metric_key": name,
                        **{
                            key: value
                            for key, value in stats.items()
                            if key not in {"values", "seed_values"}
                        },
                        "seed_values_json": json.dumps(
                            values_by_seed, sort_keys=True, separators=(",", ":")
                        ),
                    })
                if retain_named_payload:
                    retained[name] = stats
            named_payloads[field] = retained
            evidence_inventory[evidence_role] = int(len(names))
        failed_count = int(sum(bool(row.get("failed")) for row in rows))
        payload[method] = {
            "task_metrics": tasks,
            "mechanisms": named_payloads["mechanisms"],
            "probe_context": named_payloads["probe_context"],
            "probe_validity": named_payloads["probe_validity"],
            "evidence_inventory": evidence_inventory,
            "failed_run_count": failed_count,
            "prompt_mechanisms_status": "not_applicable" if any(row.get("stage") == "A0" for row in rows) else "applicable",
        }
        flat = {
            "method": method,
            "failed_run_count": failed_count,
            "prompt_mechanisms_status": payload[method]["prompt_mechanisms_status"],
        }
        for metric, stats in tasks.items():
            for field, value in stats.items():
                flat[f"{metric}_{field}"] = value
        csv_rows.append(flat)
    return csv_rows, payload


def _summarize_named_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> Dict[str, Any]:
    names = sorted({name for row in rows for name in row.get(field, {})})
    payload = {}
    for name in names:
        values_by_seed = {
            str(int(row["seed"])): float(row[field][name])
            for row in rows
            if row.get("seed") is not None
            and _float(row.get(field, {}).get(name)) is not None
        }
        payload[name] = {
            **_summary(values_by_seed.values()),
            "seed_values": values_by_seed,
        }
    return payload


def _pair_definitions(by_method: Mapping[str, Sequence[Mapping[str, Any]]], baseline_method: str):
    by_stage = {}
    for method, rows in by_method.items():
        stages = {row.get("stage") for row in rows if row.get("stage")}
        if len(stages) == 1:
            by_stage[next(iter(stages))] = method
    pairs = []
    for high, low in (("A1", "A0"), ("A2", "A0"), ("A2", "A1")):
        if high in by_stage and low in by_stage:
            pairs.append((by_stage[high], by_stage[low], f"{high}-{low}"))
    if pairs:
        return pairs
    return [
        (method, baseline_method, f"{method}-{baseline_method}")
        for method in sorted(by_method) if method != baseline_method
    ]


def _is_gate_representation_metric(metric_key: str) -> bool:
    return "condition=normal" in metric_key and any(
        name in metric_key
        for name in (
            "metric=semantic_margin",
            # Backward compatibility for summaries generated before the
            # dimension-standardized Fisher metric was renamed.
            "metric=fisher_trace_ratio",
            "metric=fisher_ratio",
        )
    )


def _paired_summaries(
    by_method,
    baseline_method,
    *,
    mechanism_delta_writer=None,
    mechanism_summary_writer=None,
    retain_mechanisms=True,
):
    def paired_metric_values(row):
        # The evidence-role split is a reporting distinction. Historical pair
        # comparisons covered both causal/mechanism evidence and descriptive
        # Probe context, so preserve that comparison universe without putting
        # either group back into the compact JSON payload.
        return ChainMap(
            row.get("mechanisms", {}),
            row.get("probe_context", {}),
        )

    paired_rows = []
    payload = {}
    for method, reference_method, pair_name in _pair_definitions(by_method, baseline_method):
        reference_by_seed = {
            int(row["seed"]): row for row in by_method.get(reference_method, [])
            if row.get("seed") is not None and not row.get("failed")
        }
        task_deltas: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        incompatible = 0
        compatible_pairs = []
        for row in by_method.get(method, []):
            seed = row.get("seed")
            if seed is None or int(seed) not in reference_by_seed or row.get("failed"):
                continue
            reference = reference_by_seed[int(seed)]
            fingerprint = row.get("shared_condition_fingerprint")
            reference_fingerprint = reference.get("shared_condition_fingerprint")
            pair = {
                "pair": pair_name,
                "method": method,
                "reference_method": reference_method,
                "seed": int(seed),
                "shared_condition_fingerprint": fingerprint,
            }
            if not fingerprint or fingerprint != reference_fingerprint:
                pair["pair_status"] = "incompatible_fingerprint"
                incompatible += 1
                paired_rows.append(pair)
                continue
            pair["pair_status"] = "paired"
            for metric in TASK_METRICS:
                if _float(row.get(metric)) is None or _float(reference.get(metric)) is None:
                    continue
                delta = float(row[metric]) - float(reference[metric])
                pair[f"delta_{metric}"] = delta
                task_deltas[metric].append((int(seed), delta))
            compatible_pairs.append((int(seed), row, reference, pair))
            paired_rows.append(pair)
        task_payload = {}
        for metric, seed_values in task_deltas.items():
            values_by_seed = {str(seed): value for seed, value in seed_values}
            values = list(values_by_seed.values())
            stats = _summary(values)
            task_payload[metric] = {
                **stats,
                "seed_values": values_by_seed,
                "positive_delta_ratio": float(sum(value > 0.0 for value in values) / max(1, len(values))),
                "effect_size": float(stats["mean"] / stats["std"]) if stats["std"] and stats["std"] > 0 else 0.0,
            }
        mechanism_payload = {}
        compatible_metric_pairs = [
            (
                seed,
                paired_metric_values(row),
                paired_metric_values(reference),
                pair,
            )
            for seed, row, reference, pair in compatible_pairs
        ]
        mechanism_names = sorted({
            name
            for _, row_values, reference_values, _ in compatible_metric_pairs
            for name in set(row_values).intersection(reference_values)
        })
        for metric in mechanism_names:
            values_by_seed = {}
            for seed, row_values, reference_values, pair in compatible_metric_pairs:
                left = _float(row_values.get(metric))
                right = _float(reference_values.get(metric))
                if left is None or right is None:
                    continue
                delta = float(left - right)
                values_by_seed[str(seed)] = delta
                if retain_mechanisms:
                    pair[f"mechanism_delta::{metric}"] = delta
                if mechanism_delta_writer is not None:
                    mechanism_delta_writer.writerow({
                        "pair": pair_name,
                        "method": method,
                        "reference_method": reference_method,
                        "seed": int(seed),
                        "metric_key": metric,
                        "delta": delta,
                    })
            if not values_by_seed:
                continue
            values = list(values_by_seed.values())
            stats = _summary(values)
            metric_payload = {
                **stats,
                "seed_values": values_by_seed,
                "positive_delta_ratio": float(sum(value > 0.0 for value in values) / max(1, len(values))),
                "effect_size": float(stats["mean"] / stats["std"]) if stats["std"] and stats["std"] > 0 else 0.0,
            }
            if mechanism_summary_writer is not None:
                mechanism_summary_writer.writerow({
                    "pair": pair_name,
                    "method": method,
                    "reference_method": reference_method,
                    "metric_key": metric,
                    **{
                        key: value
                        for key, value in metric_payload.items()
                        if key not in {"values", "seed_values"}
                    },
                    "seed_values_json": json.dumps(
                        values_by_seed, sort_keys=True, separators=(",", ":")
                    ),
                })
            gate_relevant = _is_gate_representation_metric(metric)
            if retain_mechanisms or gate_relevant:
                mechanism_payload[metric] = metric_payload
        payload[pair_name] = {
            "method": method,
            "reference_method": reference_method,
            "task_metrics": task_payload,
            "mechanisms": mechanism_payload,
            "mechanism_metric_count": int(len(mechanism_names)),
            "incompatible_fingerprint_count": incompatible,
        }
    return paired_rows, payload


def _summarize_paired_trajectory(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["epoch"]))
    if len(ordered) < TRAJECTORY_MIN_COMPLETE_EPOCHS:
        return {
            "valid": False,
            "status": "insufficient_shared_epochs",
            "shared_epoch_count": int(len(ordered)),
        }
    epochs = [int(row["epoch"]) for row in ordered]
    if epochs != list(range(epochs[0], epochs[-1] + 1)):
        return {
            "valid": False,
            "status": "noncontiguous_shared_epochs",
            "shared_epoch_count": int(len(ordered)),
        }
    window_size = _trajectory_window_size(len(ordered))
    early_rows = ordered[:window_size]
    late_rows = ordered[-window_size:]
    rolling = [ordered[index:index + window_size] for index in range(len(ordered) - window_size + 1)]
    best_h_rows = max(
        rolling,
        key=lambda window: _trajectory_mean(window, "delta_gzsl_h"),
    )
    summary: Dict[str, Any] = {
        "valid": True,
        "status": "paired",
        "shared_epoch_count": int(len(ordered)),
        "first_epoch": epochs[0],
        "final_epoch": epochs[-1],
        "window_size": int(window_size),
        "early_window_start": int(early_rows[0]["epoch"]),
        "early_window_end": int(early_rows[-1]["epoch"]),
        "late_window_start": int(late_rows[0]["epoch"]),
        "late_window_end": int(late_rows[-1]["epoch"]),
        "diagnostic_best_delta_h_window_start": int(best_h_rows[0]["epoch"]),
        "diagnostic_best_delta_h_window_end": int(best_h_rows[-1]["epoch"]),
        "analysis_role": "diagnostic_only",
        "checkpoint_selection_allowed": False,
    }
    fields = (
        "delta_train_loss",
        "delta_test_seen_nll",
        "delta_test_unseen_nll",
        "delta_gzsl_seen",
        "delta_gzsl_unseen",
        "delta_gzsl_h",
    )
    for field in fields:
        summary[f"early_{field}_mean"] = _trajectory_mean(early_rows, field)
        summary[f"late_{field}_mean"] = _trajectory_mean(late_rows, field)
        summary[f"early_to_late_{field}_change"] = (
            summary[f"late_{field}_mean"] - summary[f"early_{field}_mean"]
        )
        summary[f"diagnostic_best_delta_h_window_{field}_mean"] = _trajectory_mean(
            best_h_rows, field
        )
    h_values = [float(row["delta_gzsl_h"]) for row in ordered]
    summary["full_epoch_mean_delta_gzsl_h"] = float(statistics.mean(h_values))
    summary["h_positive_epoch_ratio"] = float(
        sum(value >= TRAJECTORY_ACCURACY_TOLERANCE for value in h_values)
        / len(h_values)
    )
    summary["h_negative_epoch_ratio"] = float(
        sum(value <= -TRAJECTORY_ACCURACY_TOLERANCE for value in h_values)
        / len(h_values)
    )
    best_delta_h = summary["diagnostic_best_delta_h_window_delta_gzsl_h_mean"]
    late_delta_h = summary["late_delta_gzsl_h_mean"]
    early_delta_h = summary["early_delta_gzsl_h_mean"]
    summary["temporary_h_advantage_lost"] = bool(
        best_delta_h >= TRAJECTORY_ACCURACY_TOLERANCE
        and late_delta_h < TRAJECTORY_ACCURACY_TOLERANCE
    )
    summary["late_h_advantage_emerged"] = bool(
        early_delta_h < TRAJECTORY_ACCURACY_TOLERANCE
        and late_delta_h >= TRAJECTORY_ACCURACY_TOLERANCE
    )
    return summary


def _trajectory_rule_contract() -> Dict[str, Any]:
    return {
        "format": "generalization_trajectory_rule_v2",
        "required_fields": list(TRAJECTORY_REQUIRED_FIELDS),
        "window_fraction": TRAJECTORY_WINDOW_FRACTION,
        "minimum_window_epochs": TRAJECTORY_MIN_WINDOW_EPOCHS,
        "minimum_complete_epochs": TRAJECTORY_MIN_COMPLETE_EPOCHS,
        "accuracy_change_tolerance": TRAJECTORY_ACCURACY_TOLERANCE,
        "loss_absolute_tolerance": TRAJECTORY_LOSS_ABS_TOLERANCE,
        "loss_relative_tolerance": TRAJECTORY_LOSS_REL_TOLERANCE,
        "formation_suffix_ratio": TRAJECTORY_FORMATION_SUFFIX_RATIO,
        "formation_min_windows": TRAJECTORY_FORMATION_MIN_WINDOWS,
        "formation_epoch_semantics": "rolling_window_end_epoch_relative_to_own_diagnostic_best_h",
        "late_low_lr_fraction_of_peak": TRAJECTORY_LATE_LR_FRACTION,
        "paired_positive_epoch_semantics": "delta_gzsl_h_greater_than_or_equal_to_accuracy_tolerance",
        "analysis_role": "diagnostic_only",
        "checkpoint_selection_allowed": False,
        "formal_checkpoint_rule": "predeclared_final_epoch_unchanged",
        "nll_boundary": "compare_within_split_trends_not_train_test_absolute_gap",
    }


def _trajectory_csv_fields() -> Dict[str, List[str]]:
    run_identity = [
        "method", "stage", "seed", "run_id", "session_id", "run_dir", "run_status",
    ]
    epoch_fields = [
        *run_identity,
        "trajectory_valid", "trajectory_status", "epoch",
        *TRAJECTORY_OPTIONAL_FIELDS,
        "lr_phase", "complete", "missing_fields",
    ]
    run_fields = [
        *run_identity,
        "valid", "status", "observed_epoch_count", "complete_epoch_count",
        "analysis_role", "checkpoint_selection_allowed", "first_epoch", "final_epoch",
        "window_size", "early_window_start", "early_window_end", "late_window_start",
        "late_window_end", "diagnostic_best_h_window_start", "diagnostic_best_h_window_end",
        "warmup_end_epoch", "configured_total_epoch", "scheduler", "base_lr",
        "lr_phase_status", "peak_lr", "peak_lr_epoch_start", "peak_lr_epoch_end",
        "late_low_lr_threshold", "late_low_lr_start_epoch", "warmup_lr_observation_available",
        "formation_status", "self_relative_formation_epoch", "persistence_ratio",
        "formation_window_count", "formation_h_threshold", "formation_suffix_ratio_required",
        "formation_min_windows_required", "formation_epoch_semantics",
    ]
    for field in TRAJECTORY_REQUIRED_FIELDS:
        run_fields.extend([
            f"early_{field}_mean",
            f"late_{field}_mean",
            f"delta_{field}",
            f"diagnostic_best_h_window_{field}_mean",
            f"best_h_window_to_late_{field}_delta",
        ])
    run_fields.extend([
        "joint_generalization_gain", "seen_specialization_tradeoff",
        "late_generalization_degradation_candidate", "optimization_without_h_gain",
        "seen_degraded", "unseen_degraded", "primary_pattern",
        "diagnostic_best_h_window_mean", "diagnostic_best_h_window_to_late_drop",
    ])
    pair_identity = [
        "pair", "method", "reference_method", "seed", "shared_condition_fingerprint",
        "analysis_role", "checkpoint_selection_allowed",
    ]
    pair_epoch_fields = [
        *pair_identity,
        "epoch",
        *(f"delta_{field}" for field in TRAJECTORY_REQUIRED_FIELDS),
    ]
    pair_fields = [
        *pair_identity,
        "valid", "status", "shared_epoch_count", "first_epoch", "final_epoch", "window_size",
        "early_window_start", "early_window_end", "late_window_start", "late_window_end",
        "diagnostic_best_delta_h_window_start", "diagnostic_best_delta_h_window_end",
    ]
    paired_metrics = (
        "delta_train_loss", "delta_test_seen_nll", "delta_test_unseen_nll",
        "delta_gzsl_seen", "delta_gzsl_unseen", "delta_gzsl_h",
    )
    for field in paired_metrics:
        pair_fields.extend([
            f"early_{field}_mean",
            f"late_{field}_mean",
            f"early_to_late_{field}_change",
            f"diagnostic_best_delta_h_window_{field}_mean",
        ])
    pair_fields.extend([
        "full_epoch_mean_delta_gzsl_h",
        "h_positive_epoch_ratio", "h_negative_epoch_ratio",
        "temporary_h_advantage_lost", "late_h_advantage_emerged",
    ])
    return {
        "epochs": epoch_fields,
        "runs": run_fields,
        "pair_epochs": pair_epoch_fields,
        "pairs": pair_fields,
    }


def _generalization_trajectory_outputs(
    runs: Sequence[Mapping[str, Any]],
    by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_method: str,
):
    epoch_rows = []
    run_rows = []
    for run in runs:
        identity = {
            "method": run.get("method"),
            "stage": run.get("stage"),
            "seed": run.get("seed"),
            "run_id": run.get("run_id"),
            "session_id": run.get("session_id"),
            "run_dir": run.get("run_dir"),
            "run_status": run.get("status"),
        }
        summary = dict(run.get("generalization_trajectory_summary", {}))
        run_rows.append({**identity, **summary})
        for epoch_row in run.get("generalization_trajectory_epochs", []):
            epoch_rows.append({
                **identity,
                "trajectory_valid": bool(summary.get("valid")),
                "trajectory_status": summary.get("status"),
                **epoch_row,
            })

    method_payload = {}
    summary_metrics = (
        "delta_train_loss",
        "delta_test_seen_nll",
        "delta_test_unseen_nll",
        "delta_gzsl_seen",
        "delta_gzsl_unseen",
        "delta_gzsl_h",
        "diagnostic_best_h_window_to_late_drop",
        "self_relative_formation_epoch",
        "persistence_ratio",
    )
    flag_fields = (
        "joint_generalization_gain",
        "seen_specialization_tradeoff",
        "late_generalization_degradation_candidate",
        "optimization_without_h_gain",
    )
    for method, method_runs in sorted(by_method.items()):
        valid = [
            run
            for run in method_runs
            if bool(run.get("generalization_trajectory_summary", {}).get("valid"))
        ]
        patterns: Dict[str, int] = defaultdict(int)
        for run in valid:
            patterns[str(run["generalization_trajectory_summary"].get("primary_pattern"))] += 1
        metrics = {}
        for field in summary_metrics:
            seed_values = {
                str(int(run["seed"])): float(run["generalization_trajectory_summary"][field])
                for run in valid
                if run.get("seed") is not None
                and _float(run.get("generalization_trajectory_summary", {}).get(field)) is not None
            }
            metrics[field] = {
                **_summary(seed_values.values()),
                "seed_values": seed_values,
            }
        flag_rates = {
            field: (
                float(sum(bool(run["generalization_trajectory_summary"].get(field)) for run in valid) / len(valid))
                if valid else None
            )
            for field in flag_fields
        }
        method_payload[method] = {
            "valid_run_count": int(len(valid)),
            "invalid_run_count": int(len(method_runs) - len(valid)),
            "primary_pattern_counts": dict(sorted(patterns.items())),
            "flag_rates": flag_rates,
            "metrics": metrics,
        }

    pair_rows = []
    pair_epoch_rows = []
    pair_payload = {}
    for method, reference_method, pair_name in _pair_definitions(by_method, baseline_method):
        pair_row_start = len(pair_rows)
        reference_by_seed = {
            int(run["seed"]): run
            for run in by_method.get(reference_method, [])
            if run.get("seed") is not None
        }
        pair_summaries = []
        incompatible_fingerprint_count = 0
        for run in by_method.get(method, []):
            seed = run.get("seed")
            base = {
                "pair": pair_name,
                "method": method,
                "reference_method": reference_method,
                "seed": seed,
                "shared_condition_fingerprint": run.get("shared_condition_fingerprint"),
                "analysis_role": "diagnostic_only",
                "checkpoint_selection_allowed": False,
            }
            if seed is None or int(seed) not in reference_by_seed:
                pair_rows.append({**base, "valid": False, "status": "missing_reference_seed"})
                continue
            reference = reference_by_seed[int(seed)]
            fingerprint = run.get("shared_condition_fingerprint")
            if not fingerprint or fingerprint != reference.get("shared_condition_fingerprint"):
                incompatible_fingerprint_count += 1
                pair_rows.append({**base, "valid": False, "status": "incompatible_fingerprint"})
                continue
            if run.get("failed") or reference.get("failed"):
                pair_rows.append({**base, "valid": False, "status": "run_not_completed"})
                continue
            if not run.get("generalization_trajectory_summary", {}).get("valid") or not reference.get(
                "generalization_trajectory_summary", {}
            ).get("valid"):
                pair_rows.append({**base, "valid": False, "status": "invalid_trajectory"})
                continue
            method_epochs = {
                int(row["epoch"]): row
                for row in run.get("generalization_trajectory_epochs", [])
                if row.get("complete")
            }
            reference_epochs = {
                int(row["epoch"]): row
                for row in reference.get("generalization_trajectory_epochs", [])
                if row.get("complete")
            }
            if set(method_epochs) != set(reference_epochs):
                pair_rows.append({**base, "valid": False, "status": "epoch_identity_mismatch"})
                continue
            deltas = []
            for epoch in sorted(method_epochs):
                delta_row = {
                    **base,
                    "epoch": int(epoch),
                }
                for field in TRAJECTORY_REQUIRED_FIELDS:
                    delta_row[f"delta_{field}"] = float(
                        method_epochs[epoch][field] - reference_epochs[epoch][field]
                    )
                deltas.append(delta_row)
                pair_epoch_rows.append(delta_row)
            pair_summary = {**base, **_summarize_paired_trajectory(deltas)}
            pair_rows.append(pair_summary)
            if pair_summary.get("valid"):
                pair_summaries.append(pair_summary)

        pair_metric_fields = (
            "full_epoch_mean_delta_gzsl_h",
            "early_delta_gzsl_h_mean",
            "late_delta_gzsl_h_mean",
            "early_to_late_delta_gzsl_h_change",
            "h_positive_epoch_ratio",
            "h_negative_epoch_ratio",
        )
        metrics = {}
        for field in pair_metric_fields:
            seed_values = {
                str(int(row["seed"])): float(row[field])
                for row in pair_summaries
                if row.get("seed") is not None and _float(row.get(field)) is not None
            }
            metrics[field] = {
                **_summary(seed_values.values()),
                "seed_values": seed_values,
            }
        pair_payload[pair_name] = {
            "method": method,
            "reference_method": reference_method,
            "valid_pair_count": int(len(pair_summaries)),
            "invalid_pair_count": int(
                len(pair_rows) - pair_row_start - len(pair_summaries)
            ),
            "incompatible_fingerprint_count": int(incompatible_fingerprint_count),
            "temporary_h_advantage_lost_count": int(
                sum(bool(row.get("temporary_h_advantage_lost")) for row in pair_summaries)
            ),
            "late_h_advantage_emerged_count": int(
                sum(bool(row.get("late_h_advantage_emerged")) for row in pair_summaries)
            ),
            "metrics": metrics,
        }
    return epoch_rows, run_rows, pair_epoch_rows, pair_rows, {
        "format": "baseline_generalization_trajectory_summary_v2",
        "rule_contract": _trajectory_rule_contract(),
        "methods": method_payload,
        "pairs": pair_payload,
    }


def _probe_record_identity(record: Mapping[str, Any]) -> str:
    return "|".join((
        f"checkpoint={record.get('checkpoint_id') or 'unknown'}",
        f"split={record.get('split') or 'unknown'}",
        f"condition={record.get('condition') or 'normal'}",
        f"domain={record.get('domain') or 'unknown'}",
        f"entity_type={record.get('entity_type') or 'unknown'}",
        f"entity_id={record.get('entity_id') or 'unknown'}",
        f"metric={record.get('metric') or 'unknown'}",
    ))


def _mechanism_key_fields(key: str) -> Dict[str, str]:
    fields = {}
    for item in str(key).split("|"):
        name, separator, value = item.partition("=")
        if separator:
            fields[name] = value
    return fields


def _descriptive_summary(values: Sequence[float]) -> Dict[str, Any]:
    payload = _summary(values)
    payload.update({
        "ci95": None,
        "ci95_low": None,
        "ci95_high": None,
        "ci95_method": "not_reported_for_correlated_probe_selections",
    })
    return payload


def _probe_robustness_summaries(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[int, Dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for run in runs:
        if run.get("failed") or run.get("seed") is None:
            continue
        method = str(run.get("method"))
        training_seed = int(run["seed"])
        for mechanism_key, raw_value in run.get("mechanisms", {}).items():
            record = _mechanism_key_fields(mechanism_key)
            probe_seed = _float(record.get("probe_selection_seed"))
            value = _float(raw_value)
            if probe_seed is None or value is None:
                continue
            grouped[method][_probe_record_identity(record)][training_seed][
                int(probe_seed)
            ] = float(value)

    payload = {}
    for method, metric_groups in sorted(grouped.items()):
        method_payload = {}
        for identity, matrix in sorted(metric_groups.items()):
            probe_seeds = sorted({seed for by_probe in matrix.values() for seed in by_probe})
            if len(probe_seeds) < 2:
                continue
            by_probe_seed = {}
            for probe_seed in probe_seeds:
                training_values = {
                    str(training_seed): by_probe[probe_seed]
                    for training_seed, by_probe in sorted(matrix.items())
                    if probe_seed in by_probe
                }
                by_probe_seed[str(probe_seed)] = {
                    **_summary(training_values.values()),
                    "training_seed_values": training_values,
                    "inference_unit": "independent_training_seed",
                }
            by_training_seed = {}
            training_seed_means = {}
            for training_seed, by_probe in sorted(matrix.items()):
                probe_values = {str(seed): value for seed, value in sorted(by_probe.items())}
                descriptive = _descriptive_summary(probe_values.values())
                by_training_seed[str(training_seed)] = {
                    **descriptive,
                    "probe_seed_values": probe_values,
                    "inference_unit": "correlated_probe_selection_on_one_checkpoint",
                }
                training_seed_means[str(training_seed)] = float(statistics.mean(probe_values.values()))
            probe_seed_means = {
                str(probe_seed): float(statistics.mean(
                    by_probe[probe_seed] for by_probe in matrix.values() if probe_seed in by_probe
                ))
                for probe_seed in probe_seeds
            }
            method_payload[identity] = {
                "training_seed_count": len(matrix),
                "probe_selection_seed_count": len(probe_seeds),
                "raw_matrix": {
                    str(training_seed): {str(seed): value for seed, value in sorted(by_probe.items())}
                    for training_seed, by_probe in sorted(matrix.items())
                },
                "by_probe_selection_seed": by_probe_seed,
                "within_training_seed_probe_variability": by_training_seed,
                "training_seed_mean_summary": {
                    **_summary(training_seed_means.values()),
                    "training_seed_values": training_seed_means,
                    "inference_unit": "independent_training_seed",
                },
                "probe_selection_mean_summary": {
                    **_descriptive_summary(probe_seed_means.values()),
                    "probe_seed_values": probe_seed_means,
                    "inference_unit": "correlated_probe_selection_summary",
                },
            }
        if method_payload:
            payload[method] = method_payload
    return payload


def _find_mechanism_values(runs, *needles):
    values = []
    for row in runs:
        if row.get("failed"):
            continue
        for field in ("mechanisms", "probe_context", "probe_validity"):
            for key, value in row.get(field, {}).items():
                if all(needle in key for needle in needles) and _float(value) is not None:
                    values.append(float(value))
    return values


def _gate_report(runs, paired_payload, expected_seeds):
    sync = _find_mechanism_values(
        runs, "split=probe_test_unseen", "metric=synchronized_equivalence_pass"
    )
    mismatch = _find_mechanism_values(
        runs, "split=probe_test_unseen", "metric=mismatched_semantic_effect_pass"
    )
    relation = _find_mechanism_values(
        runs, "split=probe_test_unseen", "domain=relation_stability"
    )
    relation.extend(_find_mechanism_values(
        runs, "split=probe_test_unseen", "metric=semantic_visual_graph_spearman"
    ))
    prompt_runs = [row for row in runs if row.get("stage") in {"A1", "A2"} and not row.get("failed")]
    prompt_update = _find_mechanism_values(prompt_runs, "domain=prompt_parameter_health", "metric=prompt_relative_update")
    prompt_zero = _find_mechanism_values(prompt_runs, "condition=prompt_zeroed", "metric=delta_true_margin")
    prompt_zero_attention_equivalence = _find_mechanism_values(
        prompt_runs,
        "split=probe_test_unseen",
        "condition=prompt_zeroed_affinity_forward",
        "metric=affinity_forward_equivalence_pass",
    )
    prompt_zero_cls_prompt_delta = _find_mechanism_values(
        prompt_runs,
        "split=probe_test_unseen",
        "condition=prompt_zeroed",
        "domain=module_effect",
        "entity_type=layer",
        "metric=delta_cls_to_prompt_mass",
    )
    prompt_zero_alignment_delta = _find_mechanism_values(
        prompt_runs,
        "split=probe_test_unseen",
        "condition=prompt_zeroed",
        "domain=module_effect",
        "entity_id=prompt_zeroed",
        "metric=delta_semantic_margin",
    )
    completed_runs = [row for row in runs if not row.get("failed")]
    calibration_gains = [
        float(row["raw_to_oracle_gain"]) for row in completed_runs
        if _float(row.get("raw_to_oracle_gain")) is not None
    ]
    calibration_complete = len(calibration_gains) == len(completed_runs) and bool(completed_runs)
    completed_seeds_by_method = {
        method: sorted({int(row["seed"]) for row in runs if row.get("method") == method and row.get("seed") is not None and not row.get("failed")})
        for method in sorted({str(row.get("method")) for row in runs})
    }
    seed_protocol_pass = all(
        set(values) == set(expected_seeds) for values in completed_seeds_by_method.values()
    )
    raw_pairs = []
    representation_pairs = []
    for name, payload in paired_payload.items():
        if name not in {"A1-A0", "A2-A0"}:
            continue
        h = payload["task_metrics"].get("gzsl_h")
        u = payload["task_metrics"].get("gzsl_unseen")
        if h:
            raw_pairs.append(h.get("positive_delta_ratio", 0.0))
        if u:
            raw_pairs.append(u.get("positive_delta_ratio", 0.0))
        for key, stats in payload["mechanisms"].items():
            if _is_gate_representation_metric(key):
                representation_pairs.append(stats.get("positive_delta_ratio", 0.0))
    gate1_pass = bool(seed_protocol_pass and sync and mismatch and all(value >= 1.0 for value in sync + mismatch))
    gate2_pass = bool(seed_protocol_pass and sync and all(value >= 1.0 for value in sync) and relation)
    gate3_pass = bool(seed_protocol_pass and prompt_runs and prompt_update and any(abs(value) > 0.0 for value in prompt_update) and prompt_zero)
    gate4_pass = bool(
        seed_protocol_pass and raw_pairs and max(raw_pairs) >= 2.0 / 3.0
        and representation_pairs and all(value >= 2.0 / 3.0 for value in representation_pairs)
        and prompt_zero and all(value < 0.0 for value in prompt_zero)
        and calibration_complete
    )

    def entry(passed, available, evidence, failure_reason):
        if not seed_protocol_pass:
            return {
                "status": "insufficient_evidence",
                "passed": False,
                "evidence": evidence,
                "failure_reasons": [
                    "completed training seeds do not match the predeclared expected seed set"
                ],
            }
        if not available:
            return {"status": "insufficient_evidence", "passed": False, "evidence": evidence, "failure_reasons": [failure_reason]}
        return {
            "status": "passed" if passed else "failed",
            "passed": bool(passed),
            "evidence": evidence,
            "failure_reasons": [] if passed else [failure_reason],
        }

    return {
        "gate_1_identifiable_evidence": entry(
            gate1_pass, bool(sync and mismatch),
            {"synchronized_pass_values": sync, "mismatched_pass_values": mismatch, "completed_seeds_by_method": completed_seeds_by_method, "expected_seeds": expected_seeds},
            "test-unseen semantic intervention did not pass in every available run",
        ),
        "gate_2_accessible_stable_representation": entry(
            gate2_pass, bool(sync and relation),
            {"synchronized_pass_values": sync, "relation_metric_count": len(relation), "completed_seeds_by_method": completed_seeds_by_method, "expected_seeds": expected_seeds},
            "relabeling equivalence or relationship evidence is incomplete",
        ),
        "gate_3_architecture_matches_computation": entry(
            gate3_pass, bool(prompt_runs),
            {
                "prompt_update_values": prompt_update,
                "prompt_zero_delta_true_margin": prompt_zero,
                "prompt_zero_attention_equivalence": prompt_zero_attention_equivalence,
                "prompt_zero_cls_prompt_delta_by_layer": prompt_zero_cls_prompt_delta,
                "prompt_zero_alignment_delta": prompt_zero_alignment_delta,
                "completed_seeds_by_method": completed_seeds_by_method,
                "expected_seeds": expected_seeds,
            },
            "Prompt update and Prompt-zero effects were not both observed",
        ),
        "gate_4_optimization_and_calibration": entry(
            gate4_pass, bool(raw_pairs and prompt_zero),
            {
                "raw_positive_delta_ratios": raw_pairs,
                "representation_positive_delta_ratios": representation_pairs,
                "prompt_zero_delta_true_margin": prompt_zero,
                "prompt_zero_attention_equivalence": prompt_zero_attention_equivalence,
                "prompt_zero_cls_prompt_delta_by_layer": prompt_zero_cls_prompt_delta,
                "prompt_zero_alignment_delta": prompt_zero_alignment_delta,
                "raw_to_oracle_gain_values": calibration_gains,
                "calibration_profile_complete": calibration_complete,
                "completed_seeds_by_method": completed_seeds_by_method,
                "expected_seeds": expected_seeds,
            },
            "raw task gain, representation gain, and Prompt-zero degradation were not jointly supported",
        ),
    }


def _cross_experiment_input(
    runs: Sequence[Mapping[str, Any]],
    method_payload: Mapping[str, Any],
    trajectory_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a compact, mechanism-free input for independent robustness studies."""
    methods = {}
    for method in sorted({str(run.get("method")) for run in runs}):
        method_runs = [run for run in runs if str(run.get("method")) == method]
        condition_by_seed = {}
        stages = set()
        for run in method_runs:
            seed = run.get("seed")
            if seed is None:
                continue
            if run.get("stage"):
                stages.add(str(run["stage"]))
            context = run.get("training_context", {}) or {}
            condition_by_seed[str(int(seed))] = {
                "condition_fingerprint": context.get("condition_fingerprint"),
                "condition_fields": context.get("condition_fields", {}),
                "run_status": run.get("status"),
            }
        fingerprints = sorted({
            str(item.get("condition_fingerprint"))
            for item in condition_by_seed.values()
            if item.get("condition_fingerprint")
        })
        methods[method] = {
            "stage": next(iter(stages)) if len(stages) == 1 else None,
            "condition_status": (
                "consistent_across_seeds" if len(fingerprints) == 1
                else "missing_condition" if not fingerprints
                else "mixed_conditions_across_seeds"
            ),
            "condition_fingerprints": fingerprints,
            "condition_by_seed": condition_by_seed,
            "task_metrics": dict(
                method_payload.get(method, {}).get("task_metrics", {})
            ),
            "trajectory_metrics": dict(
                trajectory_payload.get("methods", {})
                .get(method, {})
                .get("metrics", {})
            ),
        }
    return {
        "format": "baseline_cross_experiment_input_v1",
        "analysis_role": "diagnostic_only",
        "checkpoint_selection_allowed": False,
        "selection_protocol_required": "independent_dev_or_pseudo_unseen_validation",
        "methods": methods,
        "within_experiment_pairs": dict(trajectory_payload.get("pairs", {})),
    }


def _compact_source_artifacts(
    runs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    compact_probe_metrics: bool,
    compact_module_effect_json: bool,
) -> Dict[str, Any]:
    records = []
    seen_run_dirs = set()
    for run in runs:
        if run.get("failed") or str(run.get("status")) != "completed":
            continue
        run_dir = Path(str(run.get("run_dir"))).resolve()
        if run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir)
        if compact_probe_metrics:
            source = run_dir / "diagnostics" / "probe_metrics.csv"
            if artifact_exists(source):
                records.append({
                    "artifact_kind": "probe_metrics",
                    "method": run.get("method"),
                    "seed": run.get("seed"),
                    **compact_to_gzip(source, remove_source=True),
                })
        if compact_module_effect_json:
            module_root = run_dir / "diagnostics" / "module_effect"
            if module_root.is_dir():
                module_sources = {
                    str(path): path for path in module_root.rglob("*.json")
                }
                for compressed in module_root.rglob("*.json.gz"):
                    source = compressed.with_suffix("")
                    module_sources.setdefault(str(source), source)
                for source in sorted(module_sources.values(), key=str):
                    records.append({
                        "artifact_kind": "module_effect_json",
                        "method": run.get("method"),
                        "seed": run.get("seed"),
                        **compact_to_gzip(source, remove_source=True),
                    })
    payload = {
        "format": "baseline_artifact_compaction_v1",
        "lossless_verified": True,
        "records": records,
        "uncompressed_bytes": int(sum(
            int(item.get("uncompressed_size", 0)) for item in records
        )),
        "compressed_bytes": int(sum(
            int(item.get("compressed_size", 0)) for item in records
        )),
    }
    payload["space_saved_bytes"] = int(
        payload["uncompressed_bytes"] - payload["compressed_bytes"]
    )
    _write_json_atomic(output_dir / "artifact_compaction_manifest.json", payload)
    return payload


def main():
    args = parse_args()
    runs = []
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must use METHOD=OUTPUT_DIR")
        method, raw_path = item.split("=", 1)
        runs.append(load_run(
            method.strip(),
            Path(raw_path).expanduser().resolve(),
            include_probe_records=not bool(args.trajectory_only),
        ))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.trajectory_only:
        _write_csv(output_dir / "cross_seed_runs.csv", [_flatten_run(row) for row in runs])

    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_method[str(row["method"])].append(row)
    if args.trajectory_only:
        method_rows, method_payload = _method_summaries(by_method)
        paired_rows, paired_payload = _paired_summaries(
            by_method, args.baseline_method
        )
    else:
        with _atomic_csv_writer(
            output_dir / "method_named_metric_summary.csv.gz",
            NAMED_SUMMARY_FIELDS,
        ) as named_summary_writer, _atomic_csv_writer(
            output_dir / "paired_mechanism_deltas.csv.gz",
            PAIRED_MECHANISM_DELTA_FIELDS,
        ) as mechanism_delta_writer, _atomic_csv_writer(
            output_dir / "paired_mechanism_summary.csv.gz",
            PAIRED_MECHANISM_SUMMARY_FIELDS,
        ) as mechanism_summary_writer:
            method_rows, method_payload = _method_summaries(
                by_method,
                named_summary_writer=named_summary_writer,
                retain_named_payload=False,
            )
            paired_rows, paired_payload = _paired_summaries(
                by_method,
                args.baseline_method,
                mechanism_delta_writer=mechanism_delta_writer,
                mechanism_summary_writer=mechanism_summary_writer,
                retain_mechanisms=False,
            )
        _write_csv(output_dir / "method_summary.csv", method_rows)
        _write_csv(output_dir / "paired_deltas.csv", paired_rows)
    expected_seeds = [int(item.strip()) for item in str(args.expected_seeds).split(",") if item.strip()]
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("--expected-seeds must contain unique comma-separated integers")
    (
        trajectory_epoch_rows,
        trajectory_run_rows,
        trajectory_pair_epoch_rows,
        trajectory_pair_rows,
        trajectory_payload,
    ) = _generalization_trajectory_outputs(runs, by_method, args.baseline_method)
    trajectory_payload["expected_training_seeds"] = expected_seeds
    trajectory_fields = _trajectory_csv_fields()
    _write_csv(
        output_dir / "generalization_trajectory_epochs.csv",
        trajectory_epoch_rows,
        fieldnames=trajectory_fields["epochs"],
    )
    _write_csv(
        output_dir / "generalization_trajectory_runs.csv",
        trajectory_run_rows,
        fieldnames=trajectory_fields["runs"],
    )
    _write_csv(
        output_dir / "generalization_trajectory_pair_epochs.csv",
        trajectory_pair_epoch_rows,
        fieldnames=trajectory_fields["pair_epochs"],
    )
    _write_csv(
        output_dir / "generalization_trajectory_pairs.csv",
        trajectory_pair_rows,
        fieldnames=trajectory_fields["pairs"],
    )
    prompt_layer_rows = []
    for run in runs:
        identity = {
            "method": run.get("method"),
            "stage": run.get("stage"),
            "seed": run.get("seed"),
            "run_id": run.get("run_id"),
            "session_id": run.get("session_id"),
            "run_dir": run.get("run_dir"),
        }
        for metric_row in run.get("prompt_layer_trajectory", []):
            prompt_layer_rows.append({**identity, **metric_row})
    _write_csv(
        output_dir / "generalization_prompt_layer_trajectory.csv",
        prompt_layer_rows,
        fieldnames=[
            "method", "stage", "seed", "run_id", "session_id", "run_dir",
            "epoch", "split", "layer", "metric", "value",
        ],
    )
    _write_json_atomic(
        output_dir / "generalization_trajectory_summary.json",
        trajectory_payload,
    )
    _write_json_atomic(
        output_dir / "cross_experiment_input.json",
        _cross_experiment_input(runs, method_payload, trajectory_payload),
    )
    if args.trajectory_only:
        print(
            "wrote trajectory-only summaries for {} runs to {} without parsing fixed-probe mechanisms".format(
                len(runs), output_dir
            )
        )
        return
    gates = _gate_report(runs, paired_payload, expected_seeds)
    _write_json_atomic(
        output_dir / "four_gate_report.json",
        {"format": "a_series_four_gate_report_v1", "gates": gates},
    )
    probe_robustness = _probe_robustness_summaries(runs)
    _write_json_atomic(
        output_dir / "probe_robustness_summary.json",
        {
            "format": "baseline_probe_robustness_summary_v1",
            "expected_training_seeds": expected_seeds,
            "methods": probe_robustness,
        },
    )
    compact_methods = {
        method: {
            "task_metrics": payload["task_metrics"],
            "evidence_inventory": payload.get("evidence_inventory", {}),
            "failed_run_count": payload["failed_run_count"],
            "prompt_mechanisms_status": payload[
                "prompt_mechanisms_status"
            ],
        }
        for method, payload in method_payload.items()
    }
    summary = {
        "format": "baseline_cross_seed_summary_v4",
        "baseline_method": args.baseline_method,
        "expected_training_seeds": expected_seeds,
        "methods": compact_methods,
        "paired_deltas": {
            pair: payload["task_metrics"] for pair, payload in paired_payload.items()
        },
        "pairing_audit": {
            pair: {
                "method": payload["method"],
                "reference_method": payload["reference_method"],
                "incompatible_fingerprint_count": payload["incompatible_fingerprint_count"],
            }
            for pair, payload in paired_payload.items()
        },
        "probe_robustness": probe_robustness,
        "gates": gates,
        "artifact_index": {
            "run_table": "cross_seed_runs.csv",
            "method_task_summary": "method_summary.csv",
            "method_named_metric_summary": "method_named_metric_summary.csv.gz",
            "paired_task_deltas": "paired_deltas.csv",
            "paired_mechanism_deltas": "paired_mechanism_deltas.csv.gz",
            "paired_mechanism_summary": "paired_mechanism_summary.csv.gz",
            "source_probe_metrics": "each run diagnostics/probe_metrics.csv or probe_metrics.csv.gz",
        },
        "storage_contract": {
            "wide_mechanism_columns_emitted": False,
            "full_named_metrics_embedded_in_json": False,
            "lossless_named_metric_tables": True,
            "compressed_long_form": True,
        },
    }
    _write_json_atomic(
        output_dir / "cross_seed_summary.json",
        summary,
    )
    if args.compact_probe_metrics or args.compact_module_effect_json:
        _compact_source_artifacts(
            runs,
            output_dir,
            compact_probe_metrics=bool(args.compact_probe_metrics),
            compact_module_effect_json=bool(args.compact_module_effect_json),
        )


if __name__ == "__main__":
    main()
