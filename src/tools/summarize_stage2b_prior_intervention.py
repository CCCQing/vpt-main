#!/usr/bin/env python3
"""Summarize Stage-2B B0-B4 results across model seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.writer import stage2_metadata, write_stage2_json, write_stage2_table  # noqa: E402


GROUPS = (
    "B0_image_only",
    "B1_attribute_ridge",
    "B2_graph_gp",
    "B3_shuffled_graph_gp",
    "B4_oracle_bank",
)
COMPARISONS = {
    "good_non_graph_vs_image_only": ("B1_attribute_ridge", "B0_image_only"),
    "graph_vs_good_non_graph": ("B2_graph_gp", "B1_attribute_ridge"),
    "graph_vs_shuffled_graph": ("B2_graph_gp", "B3_shuffled_graph_gp"),
    "oracle_vs_image_only": ("B4_oracle_bank", "B0_image_only"),
    "oracle_gap_over_graph": ("B4_oracle_bank", "B2_graph_gp"),
}


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_stage2_table(path, rows, stage2_metadata("stage2B_prior_intervention_summary"))


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
            f"Stage-2B coverage mismatch: missing={sorted(expected.difference(actual))} extra={sorted(actual.difference(expected))}"
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "group_seed_results.csv", rows)
    _write(args.output_dir / "group_summary.csv", group_rows)
    _write(args.output_dir / "paired_effects_by_seed.csv", pair_rows)
    _write(args.output_dir / "paired_effect_summary.csv", pair_summary)
    metadata = {
        "format": "stage2b_prior_intervention_summary_v1",
        "groups": list(GROUPS),
        "seeds": seeds,
        "comparisons": COMPARISONS,
    }
    write_stage2_json(
        args.output_dir / "summary.json",
        metadata,
        stage2_metadata("stage2B_prior_intervention_summary"),
    )
    print(f"wrote {args.output_dir / 'group_summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-2B B0-B4 across three seeds.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-seeds", default="17,29,43")
    args = parser.parse_args()
    args.inputs = [path.resolve() for path in args.inputs]
    args.output_dir = args.output_dir.resolve()
    for path in args.inputs:
        if not path.is_file():
            parser.error(f"Input does not exist: {path}")
    return args


if __name__ == "__main__":
    summarize(parse_args())
