#!/usr/bin/env python3
"""Summarize the active B3 normal Seen/Unseen development protocol.

The independent scientific unit is the training seed.  Probe selection seeds
are nested within each checkpoint and are never counted as extra experiments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
TRAINING_SEEDS = (0, 1, 2)
METHOD_DIRS = {
    "R1I-0.25": "B3-R1I-R025",
    "R1I-0.50": "B3-R1I-R050",
    "R1N-0.25": "B3-R1N-R025",
    "R1N-0.50": "B3-R1N-R050",
}
REPORT_METHODS = {directory: method for method, directory in METHOD_DIRS.items()}
PAIR_SPECS = {
    "R1I-0.25_minus_P0-A2": ("R1I-0.25", "P0-A2"),
    "R1I-0.50_minus_P0-A2": ("R1I-0.50", "P0-A2"),
    "R1I-0.25_minus_R1N-0.25": ("R1I-0.25", "R1N-0.25"),
    "R1I-0.50_minus_R1N-0.50": ("R1I-0.50", "R1N-0.50"),
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _describe(values: Iterable[float]) -> Dict[str, object]:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _read_metrics(path: Path) -> Sequence[Mapping[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(csv.DictReader(handle))


def _one_metric(
    rows: Sequence[Mapping[str, str]],
    *,
    epoch: int,
    split: str,
    namespace: str,
    metric: str,
) -> float:
    values = [
        float(row["value"])
        for row in rows
        if int(row["epoch"]) == int(epoch)
        and row["split"] == split
        and row["namespace"] == namespace
        and row["metric"] == metric
    ]
    if len(values) != 1 or not _finite(values[0]):
        raise RuntimeError(
            "expected one finite metric epoch={} split={} namespace={} metric={}; got {}"
            .format(epoch, split, namespace, metric, len(values))
        )
    return values[0]


def _epoch_metric(
    rows: Sequence[Mapping[str, str]], split: str, namespace: str, metric: str
) -> Dict[int, float]:
    result = {}
    for row in rows:
        if row["split"] != split or row["namespace"] != namespace or row["metric"] != metric:
            continue
        epoch = int(row["epoch"])
        value = float(row["value"])
        if epoch in result:
            raise RuntimeError("duplicate trajectory metric at epoch {}".format(epoch))
        if not _finite(value):
            raise RuntimeError("non-finite trajectory metric")
        result[epoch] = value
    return result


def _formal_metrics(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    return {
        "gzsl_seen": _one_metric(
            rows, epoch=15, split="test_gzsl", namespace="classification", metric="gzsl_seen"
        ),
        "gzsl_unseen": _one_metric(
            rows, epoch=15, split="test_gzsl", namespace="classification", metric="gzsl_unseen"
        ),
        "gzsl_h": _one_metric(
            rows, epoch=15, split="test_gzsl", namespace="classification", metric="gzsl_h"
        ),
        "ausuc": _one_metric(
            rows, epoch=15, split="test_gzsl", namespace="calibration_profile", metric="ausuc"
        ),
        "raw_to_oracle_gain": _one_metric(
            rows,
            epoch=15,
            split="test_gzsl",
            namespace="calibration_profile",
            metric="raw_to_oracle_gain",
        ),
        "oracle_peak_gamma": _one_metric(
            rows,
            epoch=15,
            split="test_gzsl",
            namespace="calibration_profile",
            metric="oracle_peak_gamma",
        ),
        "seen_nll": _one_metric(
            rows, epoch=15, split="test_seen", namespace="classification", metric="nll"
        ),
        "unseen_nll": _one_metric(
            rows, epoch=15, split="test_unseen", namespace="classification", metric="nll"
        ),
        "unseen_bottom_k_class_mean": _one_metric(
            rows,
            epoch=15,
            split="test_unseen",
            namespace="class_error",
            metric="bottom_k_class_mean",
        ),
        "unseen_max_prediction_share": _one_metric(
            rows,
            epoch=15,
            split="test_unseen",
            namespace="class_error",
            metric="max_prediction_share",
        ),
        "unseen_wrong_domain_prediction_rate": _one_metric(
            rows,
            epoch=15,
            split="test_unseen",
            namespace="prediction_health",
            metric="wrong_domain_prediction_rate",
        ),
        "unseen_confidence_incorrect": _one_metric(
            rows,
            epoch=15,
            split="test_unseen",
            namespace="prediction_health",
            metric="confidence_incorrect",
        ),
        "unseen_true_class_margin_mean": _one_metric(
            rows,
            epoch=15,
            split="test_unseen",
            namespace="prediction_health",
            metric="true_class_margin_mean",
        ),
        "unseen_true_class_rank_mean": _one_metric(
            rows,
            epoch=15,
            split="test_unseen",
            namespace="prediction_health",
            metric="true_class_rank_mean",
        ),
        "unseen_entropy_mean": _one_metric(
            rows,
            epoch=15,
            split="test_unseen",
            namespace="prediction_health",
            metric="entropy_mean",
        ),
    }


def _trajectory_metrics(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    sources = {
        "train_loss": ("train", "train_epoch", "loss"),
        "seen_nll": ("test_seen", "classification", "nll"),
        "unseen_nll": ("test_unseen", "classification", "nll"),
        "gzsl_h": ("test_gzsl", "classification", "gzsl_h"),
    }
    result = {}
    for name, (split, namespace, metric) in sources.items():
        values = _epoch_metric(rows, split, namespace, metric)
        if set(values) != set(range(1, 16)):
            raise RuntimeError("{} trajectory must contain epochs 1..15".format(name))
        early = np.asarray([values[index] for index in (1, 2, 3)], dtype=np.float64)
        late = np.asarray([values[index] for index in (13, 14, 15)], dtype=np.float64)
        sequence = np.asarray([values[index] for index in range(1, 16)], dtype=np.float64)
        best = float(sequence.min() if name.endswith("nll") or name == "train_loss" else sequence.max())
        result["{}.early_mean".format(name)] = float(early.mean())
        result["{}.late_mean".format(name)] = float(late.mean())
        result["{}.late_minus_early".format(name)] = float(late.mean() - early.mean())
        result["{}.final".format(name)] = float(sequence[-1])
        result["{}.best".format(name)] = best
        result["{}.final_minus_best".format(name)] = float(sequence[-1] - best)
    return result


def _last_residual_metrics(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "metrics_step.jsonl"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("namespace") == "deep_prompt_residual":
                last = payload
    if last is None:
        raise RuntimeError("missing deep_prompt_residual step record")
    metrics = dict(last.get("metrics") or {})
    active_layers = [
        layer for layer in range(12)
        if float(metrics.get("layer_{}.active_layer".format(layer), 0.0)) > 0.5
    ]
    if active_layers != [8, 9, 10, 11]:
        raise RuntimeError("B3 R1 active layers must be 8,9,10,11")

    def values(suffix: str):
        result = [float(metrics["layer_{}.{}".format(layer, suffix)]) for layer in active_layers]
        if not all(_finite(value) for value in result):
            raise RuntimeError("non-finite residual metric {}".format(suffix))
        return result

    between = values("raw_delta_between_instance_variance")
    contract = _read_json(run_dir / "residual_freeze_contract_final.json")
    if contract.get("pass") is not True:
        raise RuntimeError("residual freeze contract failed")
    return {
        "monitor_epoch": float(last["epoch"]),
        "monitor_global_step": float(last["global_step"]),
        "mu_norm_mean": float(np.mean(values("raw_delta_norm"))),
        "mu_between_instance_variance": float(np.mean(between)),
        "active_latent_ratio": float(np.mean(np.asarray(between) > 1e-12)),
        "active_ratio_mean": float(np.mean(values("applied_ratio_mean"))),
        "active_ratio_p90_max": float(np.max(values("applied_ratio_p90"))),
        "active_ratio_max": float(np.max(values("applied_ratio_max"))),
        "active_budget_exceed_rate_max": float(np.max(values("budget_exceed_rate"))),
        "active_gate_mean": float(np.mean(values("gate"))),
        "active_raw_delta_norm_mean": float(np.mean(values("raw_delta_norm"))),
        "freeze_contract_pass": 1.0,
    }


def _a2_run(a2_root: Path, seed: int) -> Path:
    candidates = (
        a2_root / "A2" / "seed{}".format(seed) / RUN_SUFFIX,
        a2_root / "seed{}".format(seed) / RUN_SUFFIX,
    )
    found = tuple(path for path in candidates if path.is_dir())
    if len(found) != 1:
        raise FileNotFoundError("expected one A2 run for seed {}; checked {}".format(seed, candidates))
    return found[0]


def _normal_run(training_root: Path, directory: str, seed: int) -> Path:
    path = training_root / "final_gzsl" / directory / "seed{}".format(seed) / RUN_SUFFIX
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    return path


def _metric_key(**fields) -> str:
    return "|".join("{}={}".format(key, value) for key, value in fields.items())


def _summary_row(method: str, metric_key: str, values: Iterable[float], role="formal_result"):
    stats = _describe(values)
    return {
        "evidence_role": role,
        "method": method,
        "metric_key": metric_key,
        "count": stats["count"],
        "mean": stats["mean"],
        "min": stats["min"],
        "max": stats["max"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys()) if rows else ("empty",)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys()) if rows else ("empty",)
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_replay_rows(path: Path) -> Sequence[Mapping[str, str]]:
    if not path.is_file():
        return ()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(csv.DictReader(handle))


def _load_fixed_probe_rows(path: Path) -> Sequence[Mapping[str, str]]:
    if not path.is_file():
        return ()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(csv.DictReader(handle))


def _replay_metric_key(metric_path: str, scope: str) -> str:
    parts = metric_path.split(".")
    condition = "normal"
    split = "all"
    domain = "b3_source_mu_geometry"
    metric = metric_path
    if parts[:1] == ["conditions"] and len(parts) >= 4:
        condition = parts[1]
        domain = "b3_module_effect"
        if parts[2] == "splits" and len(parts) >= 6:
            split = parts[3]
            metric = ".".join(parts[4:])
        elif parts[2] in {"gzsl", "logit_geometry"}:
            split = "test_gzsl"
            metric = ".".join(parts[2:])
    elif parts[:2] == ["B3EVIDENCE", "source_mu_geometry"] and len(parts) >= 4:
        split = parts[2]
        metric = ".".join(parts[3:])
    return _metric_key(
        checkpoint="epoch_0015",
        split=split,
        condition=condition,
        domain=domain,
        entity_type="training_seed_summary",
        entity_id="all",
        probe_selection_seed="nested_three" if scope == "probe" else "full_dataset",
        metric=metric,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--a2-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evidence-full-summary", type=Path)
    parser.add_argument("--evidence-probe-summary", type=Path)
    parser.add_argument("--fixed-probe-summary", type=Path)
    args = parser.parse_args()

    training_root = args.training_root.resolve()
    a2_root = args.a2_root.resolve()
    output_dir = args.output_dir.resolve()
    formal_cells = []
    trajectory_cells = []
    residual_cells = []
    by_method_seed = {}
    run_identity = []
    method_runs = {"P0-A2": {seed: _a2_run(a2_root, seed) for seed in TRAINING_SEEDS}}
    for method, directory in METHOD_DIRS.items():
        method_runs[method] = {
            seed: _normal_run(training_root, directory, seed) for seed in TRAINING_SEEDS
        }
    for method, runs in method_runs.items():
        for seed, run_dir in runs.items():
            rows = _read_metrics(run_dir / "metrics_epoch.csv")
            formal = _formal_metrics(rows)
            trajectory = _trajectory_metrics(rows)
            by_method_seed[(method, seed)] = formal
            formal_cells.append({"method": method, "training_seed": seed, **formal})
            trajectory_cells.append({"method": method, "training_seed": seed, **trajectory})
            if method != "P0-A2":
                residual_cells.append(
                    {"method": method, "training_seed": seed, **_last_residual_metrics(run_dir)}
                )
            marker = run_dir / "training_checkpoint_ready.json"
            run_identity.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "run_dir": str(run_dir),
                    "normal_unseen_development_evidence": True,
                    "training_checkpoint_ready": marker.is_file() if method != "P0-A2" else None,
                }
            )

    pair_cells = []
    for pair_name, (target, reference) in PAIR_SPECS.items():
        for seed in TRAINING_SEEDS:
            target_metrics = by_method_seed[(target, seed)]
            reference_metrics = by_method_seed[(reference, seed)]
            pair_cells.append(
                {
                    "pair": pair_name,
                    "target": target,
                    "reference": reference,
                    "training_seed": seed,
                    **{
                        metric: target_metrics[metric] - reference_metrics[metric]
                        for metric in sorted(target_metrics)
                    },
                }
            )

    metric_rows = []
    formal_names = tuple(key for key in formal_cells[0] if key not in {"method", "training_seed"})
    for method in method_runs:
        cells = [cell for cell in formal_cells if cell["method"] == method]
        for metric in formal_names:
            split = "test_unseen" if metric.startswith("unseen_") else "test_gzsl"
            metric_rows.append(
                _summary_row(
                    method,
                    _metric_key(
                        checkpoint="epoch_0015",
                        split=split,
                        condition="normal",
                        domain="formal_task",
                        entity_type="training_seed_summary",
                        entity_id="all",
                        probe_selection_seed="all",
                        metric=metric,
                    ),
                    (cell[metric] for cell in cells),
                )
            )
    pair_names = tuple(
        key for key in pair_cells[0]
        if key not in {"pair", "target", "reference", "training_seed"}
    )
    for pair_name in PAIR_SPECS:
        cells = [cell for cell in pair_cells if cell["pair"] == pair_name]
        for metric in pair_names:
            metric_rows.append(
                _summary_row(
                    pair_name,
                    _metric_key(
                        checkpoint="epoch_0015",
                        split="test_unseen" if metric.startswith("unseen_") else "test_gzsl",
                        condition="target_minus_reference",
                        domain="formal_pair_delta",
                        entity_type="paired_training_seed_summary",
                        entity_id="all",
                        probe_selection_seed="all",
                        metric=metric,
                    ),
                    (cell[metric] for cell in cells),
                )
            )
    for method in method_runs:
        cells = [cell for cell in trajectory_cells if cell["method"] == method]
        trajectory_names = tuple(
            key for key in cells[0] if key not in {"method", "training_seed"}
        )
        for metric in trajectory_names:
            metric_rows.append(
                _summary_row(
                    method,
                    _metric_key(
                        checkpoint="epochs_0001_0015",
                        split="trajectory",
                        condition="normal",
                        domain="trajectory",
                        entity_type="training_seed_summary",
                        entity_id="all",
                        probe_selection_seed="all",
                        metric=metric,
                    ),
                    (cell[metric] for cell in cells),
                )
            )
    for method in METHOD_DIRS:
        cells = [cell for cell in residual_cells if cell["method"] == method]
        residual_names = tuple(
            key for key in cells[0] if key not in {"method", "training_seed"}
        )
        for metric in residual_names:
            metric_rows.append(
                _summary_row(
                    method,
                    _metric_key(
                        checkpoint="epoch_0015",
                        split="train_monitor_batch",
                        condition="normal",
                        domain="residual_contract",
                        entity_type="training_seed_summary",
                        entity_id="active_layers_8_11",
                        probe_selection_seed="all",
                        metric=metric,
                    ),
                    (cell[metric] for cell in cells),
                    role="validity_or_identity" if metric.endswith("contract_pass") else "mechanism_evidence",
                )
            )

    evidence_inputs = (
        ("full", args.evidence_full_summary),
        ("probe", args.evidence_probe_summary),
    )
    evidence_status = {}
    for scope, path in evidence_inputs:
        if path is None:
            evidence_status[scope] = "not_requested"
            continue
        rows = _load_replay_rows(path.resolve())
        if not rows:
            raise FileNotFoundError(str(path))
        evidence_status[scope] = "included"
        for row in rows:
            metric_rows.append(
                {
                    "evidence_role": "mechanism_evidence",
                    "method": REPORT_METHODS.get(row["method"], row["method"]),
                    "metric_key": _replay_metric_key(row["metric"], scope),
                    "count": int(row["count"]),
                    "mean": float(row["mean"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
            )

    if args.fixed_probe_summary is None:
        evidence_status["strict_three_probe"] = "not_requested"
    else:
        fixed_rows = _load_fixed_probe_rows(args.fixed_probe_summary.resolve())
        if not fixed_rows:
            raise FileNotFoundError(str(args.fixed_probe_summary))
        evidence_status["strict_three_probe"] = "included"
        for row in fixed_rows:
            method = REPORT_METHODS.get(row["method"], row["method"])
            metric_rows.append(
                {
                    "evidence_role": "mechanism_evidence",
                    "method": method,
                    "metric_key": _metric_key(
                        checkpoint="epoch_0015",
                        split=row["split"],
                        condition=row["condition"],
                        domain=row["domain"],
                        objective=row.get("objective", ""),
                        entity_type=row["entity_type"],
                        entity_id=row["entity_id"],
                        probe_selection_seed="nested_three",
                        metric=row["metric"],
                    ),
                    "count": int(row["training_seed_count"]),
                    "mean": float(row["training_seed_mean"]),
                    "min": float(row["training_seed_min"]),
                    "max": float(row["training_seed_max"]),
                }
            )

    formal_summary = defaultdict(dict)
    for method in method_runs:
        cells = [cell for cell in formal_cells if cell["method"] == method]
        for metric in formal_names:
            formal_summary[method][metric] = _describe(cell[metric] for cell in cells)
    pair_summary = defaultdict(dict)
    for pair_name in PAIR_SPECS:
        cells = [cell for cell in pair_cells if cell["pair"] == pair_name]
        for metric in pair_names:
            values = [cell[metric] for cell in cells]
            pair_summary[pair_name][metric] = {
                **_describe(values),
                "positive_seed_count": int(sum(value > 0 for value in values)),
                "negative_seed_count": int(sum(value < 0 for value in values)),
                "zero_seed_count": int(sum(value == 0 for value in values)),
            }
    payload = {
        "format": "b3_final_unseen_analysis_v1",
        "protocol": {
            "mode": "final_gzsl",
            "evaluation_classes": "normal_seen_and_normal_unseen",
            "normal_unseen_used_for_development": True,
            "untouched_final_confirmation_claim_allowed": False,
            "independent_unit": "training_seed",
            "training_seeds": list(TRAINING_SEEDS),
            "probe_selection_seeds": [424242, 424243, 424244],
            "probe_seed_role": "nested_within_checkpoint_robustness_axis",
        },
        "completeness": {
            "expected_method_count": len(method_runs),
            "expected_training_cell_count": len(method_runs) * len(TRAINING_SEEDS),
            "observed_training_cell_count": len(formal_cells),
            "evidence_summary_status": evidence_status,
            "complete": len(formal_cells) == len(method_runs) * len(TRAINING_SEEDS)
            and all(value == "included" for value in evidence_status.values()),
        },
        "run_identity": run_identity,
        "formal_summary": formal_summary,
        "paired_delta_summary": pair_summary,
    }
    _write_csv(output_dir / "formal_cells.csv", formal_cells)
    _write_csv(output_dir / "formal_pair_deltas.csv", pair_cells)
    _write_csv(output_dir / "trajectory_cells.csv", trajectory_cells)
    _write_csv(output_dir / "residual_contract_cells.csv", residual_cells)
    _write_gzip_csv(output_dir / "b3_final_metric_summary.csv.gz", metric_rows)
    _atomic_json(output_dir / "b3_final_analysis_summary.json", payload)
    _atomic_json(output_dir / "b3_final_completeness.json", payload["completeness"])
    print(
        "B3 normal-Unseen summary complete={} training_cells={} metric_rows={}".format(
            payload["completeness"]["complete"], len(formal_cells), len(metric_rows)
        )
    )
    if not payload["completeness"]["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
