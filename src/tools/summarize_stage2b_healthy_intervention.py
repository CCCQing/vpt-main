#!/usr/bin/env python3
"""Summarize Stage-2B healthy-distribution interventions across model seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


GROUPS = (
    "D0_image_posterior",
    "D0_empirical_replace_oracle",
    "D0_empirical_fusion_oracle",
    "D1_moment_replace_oracle",
    "D1_moment_fusion_oracle",
    "D2_task_replace_oracle",
    "D2_task_fusion_oracle",
    "D2_graph_gp_deployable",
    "D2_shuffled_graph_deployable",
    "D3_ce_only_oracle_reference",
)
COMPARISONS = {
    "empirical_replace_vs_image": ("D0_empirical_replace_oracle", "D0_image_posterior"),
    "empirical_fusion_vs_image": ("D0_empirical_fusion_oracle", "D0_image_posterior"),
    "moment_vs_empirical_replace": ("D1_moment_replace_oracle", "D0_empirical_replace_oracle"),
    "moment_vs_empirical_fusion": ("D1_moment_fusion_oracle", "D0_empirical_fusion_oracle"),
    "task_vs_moment_replace": ("D2_task_replace_oracle", "D1_moment_replace_oracle"),
    "task_vs_moment_fusion": ("D2_task_fusion_oracle", "D1_moment_fusion_oracle"),
    "task_fusion_vs_replace": ("D2_task_fusion_oracle", "D2_task_replace_oracle"),
    "moment_replace_vs_image": ("D1_moment_replace_oracle", "D0_image_posterior"),
    "moment_fusion_vs_image": ("D1_moment_fusion_oracle", "D0_image_posterior"),
    "task_replace_vs_image": ("D2_task_replace_oracle", "D0_image_posterior"),
    "task_fusion_vs_image": ("D2_task_fusion_oracle", "D0_image_posterior"),
    "deployable_vs_image": ("D2_graph_gp_deployable", "D0_image_posterior"),
    "shuffled_deployable_vs_image": ("D2_shuffled_graph_deployable", "D0_image_posterior"),
    "real_graph_vs_shuffled": ("D2_graph_gp_deployable", "D2_shuffled_graph_deployable"),
    "ce_oracle_vs_image": ("D3_ce_only_oracle_reference", "D0_image_posterior"),
    "ce_oracle_vs_task_oracle": ("D3_ce_only_oracle_reference", "D2_task_replace_oracle"),
}
PSEUDO_VARIANTS = ("D0_empirical", "D1_moment", "D2_task")
PSEUDO_METHODS = ("graph_gp_real", "graph_gp_shuffled", "attribute_ridge", "seen_mean")
PSEUDO_COMPARISONS = {
    "real_vs_shuffled": ("graph_gp_real", "graph_gp_shuffled"),
    "real_vs_attribute_ridge": ("graph_gp_real", "attribute_ridge"),
    "real_vs_seen_mean": ("graph_gp_real", "seen_mean"),
}


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(statistics.mean(finite)), float(statistics.stdev(finite))


def summarize(args: argparse.Namespace) -> None:
    rows = [row for path in args.inputs for row in _read(path)]
    seeds = [int(item.strip()) for item in args.expected_seeds.split(",") if item.strip()]
    expected = {(seed, group) for seed in seeds for group in GROUPS}
    actual = {(int(row["model_seed"]), str(row["group"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"Stage-2B healthy coverage mismatch: missing={sorted(expected.difference(actual))} "
            f"extra={sorted(actual.difference(expected))}"
        )
    metric_keys = sorted({key for row in rows for key in row if key not in {"model_seed", "group"}})
    by_seed_group = {(int(row["model_seed"]), str(row["group"])): row for row in rows}
    group_rows: List[Dict[str, Any]] = []
    for group in GROUPS:
        current = [by_seed_group[(seed, group)] for seed in seeds]
        summary: Dict[str, Any] = {"group": group, "seed_count": len(seeds)}
        for metric in metric_keys:
            mean, std = _mean_std([_number(row, metric) for row in current])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        group_rows.append(summary)
    pair_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for name, (lhs, rhs) in COMPARISONS.items():
            row: Dict[str, Any] = {
                "comparison": name,
                "model_seed": seed,
                "lhs": lhs,
                "rhs": rhs,
            }
            for metric in metric_keys:
                row[f"{metric}_delta"] = _number(by_seed_group[(seed, lhs)], metric) - _number(
                    by_seed_group[(seed, rhs)], metric
                )
            pair_rows.append(row)
    pair_summary: List[Dict[str, Any]] = []
    delta_keys = [key for key in pair_rows[0] if key.endswith("_delta")]
    for name in COMPARISONS:
        current = [row for row in pair_rows if row["comparison"] == name]
        summary = {"comparison": name, "seed_count": len(current)}
        for metric in delta_keys:
            mean, std = _mean_std([_number(row, metric) for row in current])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        pair_summary.append(summary)
    pseudo_rows = [row for path in args.pseudo_inputs for row in _read(path)]
    expected_pseudo = {
        (seed, fold, variant, method)
        for seed in seeds
        for fold in range(int(args.expected_folds))
        for variant in PSEUDO_VARIANTS
        for method in PSEUDO_METHODS
    }
    actual_pseudo = {
        (int(row["model_seed"]), int(row["fold"]), str(row["variant"]), str(row["method"]))
        for row in pseudo_rows
    }
    if actual_pseudo != expected_pseudo or len(pseudo_rows) != len(expected_pseudo):
        raise ValueError(
            f"Stage-2B pseudo-unseen coverage mismatch: "
            f"missing={sorted(expected_pseudo.difference(actual_pseudo))} "
            f"extra={sorted(actual_pseudo.difference(expected_pseudo))}"
        )
    pseudo_metric_keys = sorted(
        {
            key
            for row in pseudo_rows
            for key in row
            if key not in {"model_seed", "fold", "variant", "method", "support_classes", "query_classes"}
        }
    )
    pseudo_lookup = {
        (int(row["model_seed"]), int(row["fold"]), str(row["variant"]), str(row["method"])): row
        for row in pseudo_rows
    }
    pseudo_summary: List[Dict[str, Any]] = []
    for variant in PSEUDO_VARIANTS:
        for method in PSEUDO_METHODS:
            current = [
                pseudo_lookup[(seed, fold, variant, method)]
                for seed in seeds
                for fold in range(int(args.expected_folds))
            ]
            summary = {
                "variant": variant,
                "method": method,
                "seed_fold_count": len(current),
            }
            for metric in pseudo_metric_keys:
                mean, std = _mean_std([_number(row, metric) for row in current])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
            pseudo_summary.append(summary)
    pseudo_pair_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for fold in range(int(args.expected_folds)):
            for variant in PSEUDO_VARIANTS:
                for name, (lhs, rhs) in PSEUDO_COMPARISONS.items():
                    row = {
                        "comparison": name,
                        "model_seed": seed,
                        "fold": fold,
                        "variant": variant,
                        "lhs": lhs,
                        "rhs": rhs,
                    }
                    for metric in pseudo_metric_keys:
                        row[f"{metric}_delta"] = _number(
                            pseudo_lookup[(seed, fold, variant, lhs)], metric
                        ) - _number(pseudo_lookup[(seed, fold, variant, rhs)], metric)
                    pseudo_pair_rows.append(row)
    pseudo_delta_keys = [key for key in pseudo_pair_rows[0] if key.endswith("_delta")]
    pseudo_pair_summary: List[Dict[str, Any]] = []
    for variant in PSEUDO_VARIANTS:
        for name in PSEUDO_COMPARISONS:
            current = [
                row
                for row in pseudo_pair_rows
                if row["variant"] == variant and row["comparison"] == name
            ]
            summary = {
                "variant": variant,
                "comparison": name,
                "seed_fold_count": len(current),
            }
            for metric in pseudo_delta_keys:
                mean, std = _mean_std([_number(row, metric) for row in current])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
            pseudo_pair_summary.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "group_seed_results.csv", rows)
    _write(args.output_dir / "group_summary.csv", group_rows)
    _write(args.output_dir / "paired_effects_by_seed.csv", pair_rows)
    _write(args.output_dir / "paired_effect_summary.csv", pair_summary)
    _write(args.output_dir / "pseudo_unseen_rows.csv", pseudo_rows)
    _write(args.output_dir / "pseudo_unseen_summary.csv", pseudo_summary)
    _write(args.output_dir / "pseudo_unseen_paired_effects.csv", pseudo_pair_rows)
    _write(args.output_dir / "pseudo_unseen_paired_summary.csv", pseudo_pair_summary)
    metadata = {
        "format": "stage2b_healthy_intervention_summary_v1",
        "groups": list(GROUPS),
        "seeds": seeds,
        "comparisons": COMPARISONS,
        "pseudo_variants": list(PSEUDO_VARIANTS),
        "pseudo_methods": list(PSEUDO_METHODS),
        "pseudo_comparisons": PSEUDO_COMPARISONS,
        "expected_folds": int(args.expected_folds),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'group_summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-2B healthy interventions.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--pseudo-inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-seeds", default="17,29,43")
    parser.add_argument("--expected-folds", type=int, default=5)
    args = parser.parse_args()
    args.inputs = [path.resolve() for path in args.inputs]
    args.pseudo_inputs = [path.resolve() for path in args.pseudo_inputs]
    args.output_dir = args.output_dir.resolve()
    for path in args.inputs + args.pseudo_inputs:
        if not path.is_file():
            parser.error(f"Input does not exist: {path}")
    if args.expected_folds <= 1:
        parser.error("--expected-folds must be greater than one.")
    return args


if __name__ == "__main__":
    summarize(parse_args())
