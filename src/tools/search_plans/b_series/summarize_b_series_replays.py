"""Validate and summarize E1--E4 replay cells without inflating Probe n."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


def _comma_strings(raw: str) -> Tuple[str, ...]:
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("comma-separated values must be non-empty and unique")
    return values


def _comma_ints(raw: str) -> Tuple[int, ...]:
    values = tuple(int(value) for value in _comma_strings(raw))
    if min(values) < 0:
        raise ValueError("seed values must be non-negative")
    return values


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _finite_number(value) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _flatten_numbers(value, prefix: str = "") -> Dict[str, float]:
    result: Dict[str, float] = {}
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_prefix = "{}.{}".format(prefix, name) if prefix else str(name)
            result.update(_flatten_numbers(child, child_prefix))
    elif _finite_number(value):
        result[prefix] = float(value)
    return result


def _scientific_metrics(payload: Mapping[str, object], cell_dir: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for condition_name, condition in (payload.get("conditions") or {}).items():
        if not isinstance(condition, Mapping):
            continue
        for split, split_payload in (condition.get("splits") or {}).items():
            if not isinstance(split_payload, Mapping):
                continue
            for block_name in ("classification", "prediction_health", "paired_vs_normal"):
                block = split_payload.get(block_name)
                if isinstance(block, Mapping):
                    metrics.update(
                        _flatten_numbers(
                            block,
                            "conditions.{}.splits.{}.{}".format(
                                condition_name, split, block_name
                            ),
                        )
                    )
        for block_name in ("gzsl", "logit_geometry"):
            block = condition.get(block_name)
            if isinstance(block, Mapping):
                metrics.update(
                    _flatten_numbers(
                        block, "conditions.{}.{}".format(condition_name, block_name)
                    )
                )
    geometry_path = cell_dir / "E2-residual-static-geometry.json"
    if geometry_path.is_file():
        geometry = _read_json(geometry_path)
        metrics.update(_flatten_numbers(geometry.get("splits") or {}, "E2.geometry"))
    return metrics


def _describe(values: Iterable[float]) -> Dict[str, object]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def nested_scientific_summary(
    cells: Sequence[Mapping[str, object]], *, scope: str
) -> Tuple[Dict[str, object], Sequence[Mapping[str, object]]]:
    by_checkpoint = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        identity = (str(cell["method"]), int(cell["training_seed"]))
        for metric, value in cell["metrics"].items():
            by_checkpoint[identity][str(metric)].append(float(value))
    within = {}
    method_metric_training = defaultdict(lambda: defaultdict(list))
    method_metric_probe_range = defaultdict(lambda: defaultdict(list))
    for (method, training_seed), metric_values in sorted(by_checkpoint.items()):
        key = "{}|seed{}".format(method, training_seed)
        within[key] = {metric: _describe(values) for metric, values in metric_values.items()}
        for metric, values in metric_values.items():
            described = _describe(values)
            method_metric_training[method][metric].append(float(described["mean"]))
            method_metric_probe_range[method][metric].append(
                float(described["max"]) - float(described["min"])
            )
    rows = []
    across = {}
    for method, metric_values in sorted(method_metric_training.items()):
        across[method] = {}
        for metric, values in sorted(metric_values.items()):
            described = _describe(values)
            probe_ranges = _describe(method_metric_probe_range[method][metric])
            item = dict(described)
            item.update(
                {
                    "independent_unit": "training_seed",
                    "within_checkpoint_unit": (
                        "probe_selection_seed" if scope == "probe" else "full_dataset"
                    ),
                    "within_checkpoint_range_mean": probe_ranges["mean"],
                    "within_checkpoint_range_max": probe_ranges["max"],
                }
            )
            across[method][metric] = item
            rows.append({"method": method, "metric": metric, **item})
    return {"within_checkpoint": within, "across_training_seed": across}, rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "method", "metric", "count", "mean", "std", "min", "max",
        "independent_unit", "within_checkpoint_unit",
        "within_checkpoint_range_mean", "within_checkpoint_range_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--scope", choices=("probe", "full"), default="probe")
    parser.add_argument("--methods", default="B1,B2")
    parser.add_argument("--training-seeds", default="0,1,2")
    parser.add_argument("--selection-seeds", default="424242,424243,424244")
    parser.add_argument("--experiments", default="E1,E2,E3,E4")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    methods = _comma_strings(args.methods)
    training_seeds = _comma_ints(args.training_seeds)
    selection_seeds = _comma_ints(args.selection_seeds) if args.scope == "probe" else (0,)
    expected_experiments = set(_comma_strings(args.experiments))
    cells = []
    cell_status = []
    for selection_seed in selection_seeds:
        scope_dir = "probe_seed_{}".format(selection_seed) if args.scope == "probe" else "full"
        for method in methods:
            for training_seed in training_seeds:
                cell_dir = root / scope_dir / method / "seed{}".format(training_seed)
                summary_path = cell_dir / "b_series_replay_summary.json"
                failures = []
                payload = None
                if not summary_path.is_file():
                    failures.append("missing summary")
                else:
                    try:
                        payload = _read_json(summary_path)
                    except (OSError, ValueError) as error:
                        failures.append("unreadable summary: {}".format(error))
                if payload is not None:
                    if payload.get("valid") is not True or payload.get("status") != "completed":
                        failures.append("summary is not completed and valid")
                    if payload.get("scope") != args.scope:
                        failures.append("scope mismatch")
                    if args.scope == "probe" and int(payload.get("selection_seed", -1)) != selection_seed:
                        failures.append("selection seed mismatch")
                    if payload.get("training_performed") is not False or payload.get("optimizer_created") is not False:
                        failures.append("checkpoint-only identity failed")
                    if set(payload.get("experiments") or ()) != expected_experiments:
                        failures.append("experiment set mismatch")
                    if payload.get("method_name") != method:
                        failures.append("method identity mismatch")
                    source = payload.get("source") or {}
                    if int(source.get("seed", -1)) != training_seed:
                        failures.append("source training seed mismatch")
                status = {
                    "method": method,
                    "training_seed": training_seed,
                    "selection_seed": selection_seed if args.scope == "probe" else None,
                    "path": str(summary_path),
                    "valid": not failures,
                    "failure_reasons": failures,
                    "source_checkpoint_sha256": (
                        payload.get("source_checkpoint_sha256")
                        if payload is not None
                        else None
                    ),
                }
                cell_status.append(status)
                if not failures:
                    cells.append({**status, "metrics": _scientific_metrics(payload, cell_dir)})
    checkpoint_shas = defaultdict(set)
    for item in cell_status:
        if item["source_checkpoint_sha256"]:
            checkpoint_shas[(item["method"], item["training_seed"])].add(
                item["source_checkpoint_sha256"]
            )
    for identity, values in checkpoint_shas.items():
        if len(values) != 1:
            for item in cell_status:
                if (item["method"], item["training_seed"]) == identity:
                    item["valid"] = False
                    item["failure_reasons"].append(
                        "Probe cells do not share one source checkpoint SHA-256"
                    )
    expected_count = len(methods) * len(training_seeds) * len(selection_seeds)
    valid = all(item["valid"] for item in cell_status) and len(cells) == expected_count
    scientific, rows = nested_scientific_summary(cells, scope=args.scope)
    output_dir = (args.output_dir or (root / "b_series_summary")).resolve()
    payload = {
        "format": "b_series_experiments_replay_aggregate_v1",
        "suite_name": "B-series",
        "scope": args.scope,
        "methods": list(methods),
        "training_seeds": list(training_seeds),
        "selection_seeds": list(selection_seeds) if args.scope == "probe" else None,
        "experiments": sorted(expected_experiments),
        "expected_cell_count": expected_count,
        "valid_cell_count": len(cells),
        "independent_n_definition": "training_seed; Probe selection seeds are nested within checkpoint",
        "cells": cell_status,
        "scientific_summary": scientific,
        "valid": valid,
        "status": "completed" if valid else "incomplete_or_invalid",
    }
    _write_json(output_dir / "b_series_replay_aggregate.json", payload)
    _write_csv(output_dir / "b_series_replay_scientific_summary.csv", rows)
    print(
        "B-series aggregate status={} valid_cells={}/{} metrics={}".format(
            payload["status"], len(cells), expected_count, len(rows)
        )
    )
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
