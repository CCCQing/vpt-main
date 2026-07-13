#!/usr/bin/env python3
"""Summarize Stage-2 Graph-GP results across model seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


METRICS = (
    "raw_cosine",
    "centered_cosine",
    "nrmse",
    "geometry_spearman",
    "pseudo_seen",
    "pseudo_unseen",
    "pseudo_h",
    "pseudo_unseen_zsl",
    "uncertainty_error_spearman",
)
HIGHER_IS_BETTER = {metric: metric != "nrmse" for metric in METRICS}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _number(row: Mapping[str, Any], key: str) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return float("nan")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(finite)) if finite else float("nan")


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(statistics.mean(finite)), float(statistics.stdev(finite))


def _parse_expected_seeds(raw: str) -> List[int]:
    seeds = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("--expected-seeds must contain unique non-negative integers.")
    return seeds


def _load_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        current = _read_csv(path)
        if not current:
            raise ValueError(f"Input CSV is empty: {path}")
        for row in current:
            row["source_path"] = str(path)
            row["model_seed"] = int(row["model_seed"])
            row["fold"] = int(row["fold"])
            row["replicate"] = int(row.get("replicate", -1))
            rows.append(row)
    return rows


def _validate_coverage(rows: Sequence[Mapping[str, Any]], expected_seeds: Sequence[int], expected_folds: int) -> None:
    actual_seeds = sorted({int(row["model_seed"]) for row in rows})
    if actual_seeds != sorted(int(seed) for seed in expected_seeds):
        raise ValueError(f"Model seed coverage mismatch: expected={expected_seeds} actual={actual_seeds}")
    groups: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("cell_id", "")), int(row["model_seed"]), str(row["method_group"]))
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        folds = sorted({int(row["fold"]) for row in group})
        if folds != list(range(expected_folds)):
            raise ValueError(f"Fold coverage mismatch for {key}: {folds}")
        method_group = key[2]
        if method_group == "method1_shuffled":
            counts = {
                fold: len([row for row in group if int(row["fold"]) == fold])
                for fold in folds
            }
            if len(set(counts.values())) != 1 or min(counts.values()) <= 0:
                raise ValueError(f"Shuffled replicate coverage mismatch for {key}: {counts}")
        elif len(group) != expected_folds:
            raise ValueError(f"Expected one row per fold for {key}, got {len(group)} rows.")


def _seed_method_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("cell_id", "")), int(row["model_seed"]), str(row["method_group"]))
        grouped.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    for (cell_id, seed, method_group), group in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "cell_id": cell_id,
            "model_seed": int(seed),
            "method_group": method_group,
            "fold_count": len({int(row["fold"]) for row in group}),
            "row_count": len(group),
            "replicate_count": len({int(row["replicate"]) for row in group if int(row["replicate"]) >= 0}),
        }
        for metric in METRICS:
            summary[metric] = _mean([_number(row, metric) for row in group])
        summaries.append(summary)
    return summaries


def _method_summary_rows(seed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["cell_id"]), str(row["method_group"])), []).append(row)
    summaries: List[Dict[str, Any]] = []
    for (cell_id, method_group), group in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "cell_id": cell_id,
            "method_group": method_group,
            "seed_count": len(group),
            "seeds": ",".join(str(row["model_seed"]) for row in sorted(group, key=lambda item: int(item["model_seed"]))),
        }
        for metric in METRICS:
            mean_value, std_value = _mean_std([_number(row, metric) for row in group])
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_std"] = std_value
        summaries.append(summary)
    ranked: List[Dict[str, Any]] = []
    for cell_id in sorted({str(row["cell_id"]) for row in summaries}):
        cell_rows = [row for row in summaries if str(row["cell_id"]) == cell_id]
        cell_rows.sort(
            key=lambda row: (
                not math.isfinite(float(row.get("pseudo_h_mean", float("nan")))),
                -float(row.get("pseudo_h_mean", float("-inf")))
                if math.isfinite(float(row.get("pseudo_h_mean", float("nan"))))
                else 0.0,
                str(row["method_group"]),
            )
        )
        for rank, row in enumerate(cell_rows, start=1):
            row["pseudo_h_rank"] = rank
        ranked.extend(cell_rows)
    return ranked


def _paired_graph_shuffled_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("cell_id", "")),
            int(row["model_seed"]),
            int(row["fold"]),
            str(row["method_group"]),
        )
        grouped.setdefault(key, []).append(row)

    paired_fold_rows: List[Dict[str, Any]] = []
    base_keys = sorted({(key[0], key[1], key[2]) for key in grouped})
    for cell_id, seed, fold in base_keys:
        real = grouped.get((cell_id, seed, fold, "method1_diff"), [])
        shuffled = grouped.get((cell_id, seed, fold, "method1_shuffled"), [])
        if len(real) != 1 or not shuffled:
            raise ValueError(
                f"Cannot pair method1_diff with shuffled for cell={cell_id} seed={seed} fold={fold}."
            )
        row: Dict[str, Any] = {
            "cell_id": cell_id,
            "model_seed": seed,
            "fold": fold,
            "shuffle_replicates": len(shuffled),
        }
        for metric in METRICS:
            real_value = _number(real[0], metric)
            shuffled_value = _mean([_number(item, metric) for item in shuffled])
            delta = real_value - shuffled_value
            row[f"{metric}_real"] = real_value
            row[f"{metric}_shuffled_mean"] = shuffled_value
            row[f"{metric}_delta"] = delta
            row[f"{metric}_improvement"] = delta if HIGHER_IS_BETTER[metric] else -delta
        paired_fold_rows.append(row)

    grouped_seed: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in paired_fold_rows:
        grouped_seed.setdefault((str(row["cell_id"]), int(row["model_seed"])), []).append(row)
    seed_rows: List[Dict[str, Any]] = []
    for (cell_id, seed), group in sorted(grouped_seed.items()):
        summary: Dict[str, Any] = {
            "cell_id": cell_id,
            "model_seed": seed,
            "fold_count": len(group),
        }
        for metric in METRICS:
            summary[f"{metric}_delta"] = _mean([_number(row, f"{metric}_delta") for row in group])
            summary[f"{metric}_improvement"] = _mean(
                [_number(row, f"{metric}_improvement") for row in group]
            )
        seed_rows.append(summary)
    return paired_fold_rows, seed_rows


def _paired_summary_rows(seed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["cell_id"]), []).append(row)
    summaries: List[Dict[str, Any]] = []
    for cell_id, group in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "cell_id": cell_id,
            "comparison": "method1_diff-minus-method1_shuffled",
            "seed_count": len(group),
            "seeds": ",".join(str(row["model_seed"]) for row in sorted(group, key=lambda item: int(item["model_seed"]))),
        }
        for metric in METRICS:
            for suffix in ("delta", "improvement"):
                mean_value, std_value = _mean_std(
                    [_number(row, f"{metric}_{suffix}") for row in group]
                )
                summary[f"{metric}_{suffix}_mean"] = mean_value
                summary[f"{metric}_{suffix}_std"] = std_value
        summaries.append(summary)
    return summaries


def summarize(args: argparse.Namespace) -> None:
    rows = _load_rows(args.inputs)
    expected_seeds = _parse_expected_seeds(args.expected_seeds)
    _validate_coverage(rows, expected_seeds, int(args.expected_folds))
    seed_methods = _seed_method_rows(rows)
    methods = _method_summary_rows(seed_methods)
    paired_folds, paired_seeds = _paired_graph_shuffled_rows(rows)
    paired_summary = _paired_summary_rows(paired_seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "seed_method_summary.csv", seed_methods)
    _write_csv(args.output_dir / "method_summary.csv", methods)
    _write_csv(args.output_dir / "graph_vs_shuffled_fold_pairs.csv", paired_folds)
    _write_csv(args.output_dir / "graph_vs_shuffled_seed_pairs.csv", paired_seeds)
    _write_csv(args.output_dir / "graph_vs_shuffled_summary.csv", paired_summary)
    _write_json(
        args.output_dir / "summary.json",
        {
            "format": "graph_gp_stage2_summary_v1",
            "inputs": [str(path) for path in args.inputs],
            "expected_seeds": expected_seeds,
            "expected_folds": int(args.expected_folds),
            "method_rows": methods,
            "graph_vs_shuffled_rows": paired_summary,
        },
    )
    for name in (
        "seed_method_summary.csv",
        "method_summary.csv",
        "graph_vs_shuffled_fold_pairs.csv",
        "graph_vs_shuffled_seed_pairs.csv",
        "graph_vs_shuffled_summary.csv",
        "summary.json",
    ):
        print(f"wrote {args.output_dir / name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-2 Graph-GP results across model seeds.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-seeds", default="17,29,43")
    parser.add_argument("--expected-folds", type=int, default=5)
    args = parser.parse_args()
    args.inputs = [path.resolve() for path in args.inputs]
    args.output_dir = args.output_dir.resolve()
    for path in args.inputs:
        if not path.is_file():
            parser.error(f"Input fold_results.csv does not exist: {path}")
    if args.expected_folds < 2:
        parser.error("expected-folds must be >= 2.")
    return args


if __name__ == "__main__":
    summarize(parse_args())
