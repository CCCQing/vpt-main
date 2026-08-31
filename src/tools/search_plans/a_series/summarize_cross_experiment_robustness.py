#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


HARD_COMPATIBILITY_FIELDS = {
    "data.name",
    "data.feature",
    "data.protocol_mode",
    "model.type",
    "model.transfer_type",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare independently produced A-series summaries without using final-test "
            "results for checkpoint or hyperparameter selection."
        )
    )
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="NAME=SUMMARY_DIR_OR_JSON",
    )
    parser.add_argument("--reference-experiment", required=True)
    parser.add_argument(
        "--comparison-axis",
        action="append",
        default=[],
        help="Predeclared condition field allowed to differ, for example solver.base_lr.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _float(value: Any):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_experiment(spec: str) -> Tuple[str, Dict[str, Any], Path]:
    if "=" not in spec:
        raise ValueError("--experiment must use NAME=SUMMARY_DIR_OR_JSON")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("experiment name must not be empty")
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        path = path / "cross_experiment_input.json"
    if not path.is_file():
        raise FileNotFoundError(
            "cross-experiment input is missing: {}. Rerun "
            "summarize_baseline_monitoring.py for that experiment first.".format(path)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "baseline_cross_experiment_input_v1":
        raise ValueError("unsupported cross-experiment input format in {}".format(path))
    return name, payload, path


def _representative_condition(method_payload: Mapping[str, Any]):
    if method_payload.get("condition_status") != "consistent_across_seeds":
        return None
    conditions = [
        item.get("condition_fields", {})
        for item in method_payload.get("condition_by_seed", {}).values()
        if item.get("condition_fields")
    ]
    if not conditions:
        return None
    first = conditions[0]
    if any(condition != first for condition in conditions[1:]):
        return None
    return dict(first)


def _condition_comparison(
    reference: Mapping[str, Any],
    target: Mapping[str, Any],
    declared_axes: Sequence[str],
) -> Dict[str, Any]:
    reference_condition = _representative_condition(reference)
    target_condition = _representative_condition(target)
    base = {
        "declared_comparison_axes": list(declared_axes),
        "reference_condition_status": reference.get("condition_status"),
        "target_condition_status": target.get("condition_status"),
    }
    if reference_condition is None or target_condition is None:
        return {
            **base,
            "valid": False,
            "status": "missing_or_mixed_condition_identity",
            "changed_fields": [],
            "undeclared_changed_fields": [],
            "hard_incompatible_fields": [],
        }
    all_fields = sorted(set(reference_condition).union(target_condition))
    changed = [
        field
        for field in all_fields
        if reference_condition.get(field) != target_condition.get(field)
    ]
    declared = set(str(item) for item in declared_axes)
    undeclared = sorted(set(changed) - declared)
    hard = sorted(set(changed).intersection(HARD_COMPATIBILITY_FIELDS))
    if hard:
        status = "incompatible_data_protocol_or_model_identity"
        valid = False
    elif undeclared:
        status = "exploratory_uncontrolled_condition_differences"
        valid = False
    elif changed:
        status = "compatible_predeclared_axes"
        valid = True
    else:
        status = "identical_condition"
        valid = True
    return {
        **base,
        "valid": valid,
        "status": status,
        "changed_fields": changed,
        "undeclared_changed_fields": undeclared,
        "hard_incompatible_fields": hard,
        "reference_condition": reference_condition,
        "target_condition": target_condition,
    }


def _metric_records(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for method, method_payload in sorted(payload.get("methods", {}).items()):
        for family, source_name in (
            ("task", "task_metrics"),
            ("trajectory", "trajectory_metrics"),
        ):
            for metric, summary in sorted(method_payload.get(source_name, {}).items()):
                yield {
                    "entity_type": "method",
                    "entity": method,
                    "metric_family": family,
                    "metric": metric,
                    "summary": summary,
                }
    for pair, pair_payload in sorted(payload.get("within_experiment_pairs", {}).items()):
        for metric, summary in sorted(pair_payload.get("metrics", {}).items()):
            yield {
                "entity_type": "within_experiment_pair",
                "entity": pair,
                "metric_family": "paired_trajectory",
                "metric": metric,
                "summary": summary,
            }


def _paired_seed_delta(
    reference_summary: Mapping[str, Any], target_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    reference_values = reference_summary.get("seed_values", {}) or {}
    target_values = target_summary.get("seed_values", {}) or {}
    shared = sorted(set(reference_values).intersection(target_values), key=lambda item: int(item))
    deltas = {
        seed: float(target_values[seed]) - float(reference_values[seed])
        for seed in shared
        if _float(target_values.get(seed)) is not None
        and _float(reference_values.get(seed)) is not None
    }
    values = list(deltas.values())
    return {
        "paired_seed_count": len(values),
        "paired_seed_values": deltas,
        "paired_seed_delta_mean": statistics.mean(values) if values else None,
        "paired_seed_delta_min": min(values) if values else None,
        "paired_seed_delta_max": max(values) if values else None,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "reference_experiment",
        "target_experiment",
        "entity_type",
        "entity",
        "metric_family",
        "metric",
        "reference_mean",
        "target_mean",
        "target_minus_reference_mean",
        "paired_seed_count",
        "paired_seed_delta_mean",
        "paired_seed_delta_min",
        "paired_seed_delta_max",
        "comparability_valid",
        "comparability_status",
        "changed_fields",
        "undeclared_changed_fields",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _pair_methods(pair_payload: Mapping[str, Any]) -> List[str]:
    return [
        str(item)
        for item in (pair_payload.get("method"), pair_payload.get("reference_method"))
        if item
    ]


def main():
    args = parse_args()
    experiments = {}
    input_paths = {}
    for spec in args.experiment:
        name, payload, path = _read_experiment(spec)
        if name in experiments:
            raise ValueError("duplicate experiment name: {}".format(name))
        experiments[name] = payload
        input_paths[name] = str(path)
    if args.reference_experiment not in experiments:
        raise ValueError("--reference-experiment must name one supplied experiment")
    declared_axes = [str(item) for item in args.comparison_axis]
    if len(set(declared_axes)) != len(declared_axes):
        raise ValueError("--comparison-axis values must be unique")

    reference_name = args.reference_experiment
    reference = experiments[reference_name]
    rows = []
    comparability = {}
    reference_records = {
        (item["entity_type"], item["entity"], item["metric_family"], item["metric"]): item
        for item in _metric_records(reference)
    }
    for target_name, target in sorted(experiments.items()):
        if target_name == reference_name:
            continue
        method_comparability = {}
        shared_methods = sorted(set(reference.get("methods", {})).intersection(target.get("methods", {})))
        for method in shared_methods:
            method_comparability[method] = _condition_comparison(
                reference["methods"][method], target["methods"][method], declared_axes
            )
        comparability[target_name] = method_comparability

        for record in _metric_records(target):
            key = (
                record["entity_type"], record["entity"],
                record["metric_family"], record["metric"],
            )
            reference_record = reference_records.get(key)
            if reference_record is None:
                continue
            if record["entity_type"] == "method":
                comparison = method_comparability.get(record["entity"], {
                    "valid": False,
                    "status": "method_missing_from_condition_audit",
                    "changed_fields": [],
                    "undeclared_changed_fields": [],
                })
            else:
                pair_payload = target.get("within_experiment_pairs", {}).get(record["entity"], {})
                pair_methods = _pair_methods(pair_payload)
                method_checks = [method_comparability.get(method) for method in pair_methods]
                valid_checks = [item for item in method_checks if item is not None]
                comparison = {
                    "valid": bool(valid_checks) and all(bool(item.get("valid")) for item in valid_checks),
                    "status": (
                        "compatible_pair_conditions" if valid_checks and all(bool(item.get("valid")) for item in valid_checks)
                        else "pair_condition_not_comparable"
                    ),
                    "changed_fields": sorted({field for item in valid_checks for field in item.get("changed_fields", [])}),
                    "undeclared_changed_fields": sorted({field for item in valid_checks for field in item.get("undeclared_changed_fields", [])}),
                }
            reference_mean = _float(reference_record["summary"].get("mean"))
            target_mean = _float(record["summary"].get("mean"))
            paired = _paired_seed_delta(reference_record["summary"], record["summary"])
            rows.append({
                "reference_experiment": reference_name,
                "target_experiment": target_name,
                "entity_type": record["entity_type"],
                "entity": record["entity"],
                "metric_family": record["metric_family"],
                "metric": record["metric"],
                "reference_mean": reference_mean,
                "target_mean": target_mean,
                "target_minus_reference_mean": (
                    target_mean - reference_mean
                    if target_mean is not None and reference_mean is not None else None
                ),
                **paired,
                "comparability_valid": comparison.get("valid"),
                "comparability_status": comparison.get("status"),
                "changed_fields": "|".join(comparison.get("changed_fields", [])),
                "undeclared_changed_fields": "|".join(
                    comparison.get("undeclared_changed_fields", [])
                ),
            })

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "cross_experiment_robustness_rows.csv", rows)
    audit_payload = {
        "format": "cross_experiment_comparability_v1",
        "reference_experiment": reference_name,
        "declared_comparison_axes": declared_axes,
        "hard_compatibility_fields": sorted(HARD_COMPATIBILITY_FIELDS),
        "input_paths": input_paths,
        "comparisons": comparability,
    }
    (output_dir / "cross_experiment_comparability.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_payload = {
        "format": "cross_experiment_robustness_summary_v1",
        "analysis_role": "diagnostic_only",
        "checkpoint_selection_allowed": False,
        "selection_protocol_required": "predeclared_fixed_training_and_official_final_gzsl_evaluation",
        "reference_experiment": reference_name,
        "comparison_row_count": len(rows),
        "rows": rows,
    }
    (output_dir / "cross_experiment_robustness_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("wrote {} cross-experiment comparison rows to {}".format(len(rows), output_dir))


if __name__ == "__main__":
    main()
