#!/usr/bin/env python3
"""Summarize valid Graph-GP factorial effects across Stage-1 and Stage-2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CELLS = ("A00", "A10", "C00", "C10", "C01", "C11")
STAGE1_METRICS = (
    "dev_unseen_last",
    "zsl_unseen_last",
    "gzsl_seen_last",
    "gzsl_unseen_last",
    "gzsl_h_last",
    "gzsl_seen_best",
    "gzsl_unseen_best",
    "gzsl_h_best",
)
STAGE2_METRICS = (
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
ALIGNMENT_METRICS = (
    "graph_prompt_spearman",
    "graph_prompt_cka",
    "graph_prompt_topk_overlap",
    "graph_final_logits_spearman",
    "graph_final_logits_cka",
    "graph_final_logits_topk_overlap",
    "graph_final_visual_spearman",
    "graph_final_visual_cka",
    "graph_final_visual_topk_overlap",
    "graph_final_semantic_spearman",
    "graph_final_semantic_cka",
    "graph_final_semantic_topk_overlap",
    "prompt_final_logits_spearman",
    "prompt_final_logits_cka",
    "prompt_final_logits_topk_overlap",
    "prompt_final_visual_spearman",
    "prompt_final_visual_cka",
    "prompt_final_visual_topk_overlap",
    "prompt_final_semantic_spearman",
    "prompt_final_semantic_cka",
    "prompt_final_semantic_topk_overlap",
    "final_logits_visual_spearman",
    "final_logits_visual_cka",
    "final_logits_visual_topk_overlap",
    "final_visual_semantic_spearman",
    "final_visual_semantic_cka",
    "final_visual_semantic_topk_overlap",
    "final_visual_semantic_paired_cosine_mean",
    "final_visual_semantic_paired_centered_cosine_mean",
)
PAIR_EFFECTS = {
    "graph_gp_no_semantic": ("A10", "A00"),
    "graph_gp_semantic_no_am": ("C10", "C00"),
    "graph_gp_semantic_with_am": ("C11", "C01"),
    "am_no_graph_gp": ("C01", "C00"),
    "am_with_graph_gp": ("C11", "C10"),
    "semantic_no_graph_gp": ("C00", "A00"),
    "semantic_with_graph_gp": ("C10", "A10"),
}
INTERACTIONS = {
    "graph_gp_am_interaction_semantic_on": (("C11", "C01"), ("C10", "C00")),
    "graph_gp_semantic_interaction_no_am": (("C10", "C00"), ("A10", "A00")),
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(finite)) if finite else float("nan")


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(statistics.mean(finite)), float(statistics.stdev(finite))


def _load_many(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        current = _read_csv(path)
        if not current:
            raise ValueError(f"Input CSV is empty: {path}")
        rows.extend(current)
    return rows


def _group(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def _validate_cell_seed_coverage(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], source: str) -> None:
    actual = {(str(row["cell_id"]), int(row["model_seed"])) for row in rows}
    expected = {(cell, int(seed)) for cell in CELLS for seed in seeds}
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(f"{source} cell/seed coverage mismatch: missing={missing} extra={extra}")


def _seed_cell_rows(
    stage1_rows: Sequence[Mapping[str, Any]],
    fold_rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    expected_folds: int,
) -> List[Dict[str, Any]]:
    stage1 = {
        (str(row["cell_id"]), int(row["seed"])): row
        for row in stage1_rows
        if int(row.get("returncode", 0)) == 0
    }
    expected = {(cell, int(seed)) for cell in CELLS for seed in seeds}
    if set(stage1) != expected:
        raise ValueError(f"Stage-1 successful coverage mismatch: missing={sorted(expected.difference(stage1))}")

    fold_groups = _group(fold_rows, ("cell_id", "model_seed", "method_group"))
    alignment_groups = _group(alignment_rows, ("cell_id", "model_seed", "method_group"))
    output: List[Dict[str, Any]] = []
    for cell in CELLS:
        for seed in seeds:
            key = (cell, str(int(seed)))
            real = fold_groups.get(key + ("method1_diff",), [])
            shuffled = fold_groups.get(key + ("method1_shuffled",), [])
            align_real = alignment_groups.get(key + ("method1_diff",), [])
            align_shuffled = alignment_groups.get(key + ("method1_shuffled",), [])
            if len(real) != expected_folds or len(align_real) != expected_folds:
                raise ValueError(f"Expected {expected_folds} real folds for {cell}/seed{seed}.")
            if len(shuffled) % expected_folds or len(align_shuffled) % expected_folds:
                raise ValueError(f"Shuffled fold coverage is incomplete for {cell}/seed{seed}.")
            row: Dict[str, Any] = {"cell_id": cell, "model_seed": int(seed)}
            for metric in STAGE1_METRICS:
                row[f"stage1_{metric}"] = _number(stage1[(cell, int(seed))], metric)
            for metric in STAGE2_METRICS:
                real_value = _mean(_number(item, metric) for item in real)
                shuffled_value = _mean(_number(item, metric) for item in shuffled)
                row[f"stage2_{metric}"] = real_value
                delta = real_value - shuffled_value
                row[f"stage2_graph_vs_shuffle_{metric}_improvement"] = -delta if metric == "nrmse" else delta
            for metric in ALIGNMENT_METRICS:
                real_value = _mean(_number(item, metric) for item in align_real)
                row[f"alignment_{metric}"] = real_value
                if metric.startswith("graph_"):
                    shuffled_value = _mean(_number(item, metric) for item in align_shuffled)
                    row[f"alignment_graph_vs_shuffle_{metric}_improvement"] = real_value - shuffled_value
            output.append(row)
    return output


def _cell_summary(seed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metric_keys = [key for key in seed_rows[0] if key not in {"cell_id", "model_seed"}]
    rows: List[Dict[str, Any]] = []
    for cell in CELLS:
        group = [row for row in seed_rows if str(row["cell_id"]) == cell]
        summary: Dict[str, Any] = {"cell_id": cell, "seed_count": len(group)}
        for metric in metric_keys:
            mean, std = _mean_std(_number(row, metric) for row in group)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        rows.append(summary)
    return rows


def _factorial_effects(seed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metric_keys = [key for key in seed_rows[0] if key not in {"cell_id", "model_seed"}]
    by_seed_cell = {(int(row["model_seed"]), str(row["cell_id"])): row for row in seed_rows}
    effects: List[Dict[str, Any]] = []
    for seed in sorted({int(row["model_seed"]) for row in seed_rows}):
        for name, (lhs, rhs) in PAIR_EFFECTS.items():
            row: Dict[str, Any] = {
                "effect": name,
                "model_seed": seed,
                "lhs": lhs,
                "rhs": rhs,
            }
            for metric in metric_keys:
                row[f"{metric}_delta"] = _number(by_seed_cell[(seed, lhs)], metric) - _number(
                    by_seed_cell[(seed, rhs)], metric
                )
            effects.append(row)
        for name, ((lhs_a, lhs_b), (rhs_a, rhs_b)) in INTERACTIONS.items():
            row = {
                "effect": name,
                "model_seed": seed,
                "lhs": f"({lhs_a}-{lhs_b})",
                "rhs": f"({rhs_a}-{rhs_b})",
            }
            for metric in metric_keys:
                row[f"{metric}_delta"] = (
                    _number(by_seed_cell[(seed, lhs_a)], metric)
                    - _number(by_seed_cell[(seed, lhs_b)], metric)
                    - _number(by_seed_cell[(seed, rhs_a)], metric)
                    + _number(by_seed_cell[(seed, rhs_b)], metric)
                )
            effects.append(row)
    return effects


def _effect_summary(effect_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metric_keys = [key for key in effect_rows[0] if key.endswith("_delta")]
    rows: List[Dict[str, Any]] = []
    for effect, group in sorted(_group(effect_rows, ("effect",)).items()):
        summary: Dict[str, Any] = {"effect": effect[0], "seed_count": len(group)}
        for metric in metric_keys:
            mean, std = _mean_std(_number(row, metric) for row in group)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        rows.append(summary)
    return rows


def summarize(args: argparse.Namespace) -> None:
    seeds = [int(item) for item in args.expected_seeds.split(",") if item.strip()]
    stage1_rows = _read_csv(args.stage1_summary)
    fold_rows = _load_many(args.fold_inputs)
    alignment_rows = _load_many(args.alignment_inputs)
    _validate_cell_seed_coverage(fold_rows, seeds, "Stage-2 folds")
    _validate_cell_seed_coverage(alignment_rows, seeds, "Stage-2 alignments")
    seed_rows = _seed_cell_rows(stage1_rows, fold_rows, alignment_rows, seeds, args.expected_folds)
    cell_rows = _cell_summary(seed_rows)
    effect_rows = _factorial_effects(seed_rows)
    effect_summary = _effect_summary(effect_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "seed_cell_joint_metrics.csv", seed_rows)
    _write_csv(args.output_dir / "cell_joint_summary.csv", cell_rows)
    _write_csv(args.output_dir / "factorial_effects_by_seed.csv", effect_rows)
    _write_csv(args.output_dir / "factorial_effect_summary.csv", effect_summary)
    payload = {
        "format": "graph_gp_factorial_summary_v1",
        "cells": list(CELLS),
        "seeds": seeds,
        "identifiable_interactions": list(INTERACTIONS),
        "undefined_interactions": ["semantic_x_am", "graph_gp_x_semantic_x_am"],
    }
    (args.output_dir / "factorial_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in (
        "seed_cell_joint_metrics.csv",
        "cell_joint_summary.csv",
        "factorial_effects_by_seed.csv",
        "factorial_effect_summary.csv",
        "factorial_summary.json",
    ):
        print(f"wrote {args.output_dir / name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the six valid Graph-GP factorial cells.")
    parser.add_argument("--stage1-summary", required=True, type=Path)
    parser.add_argument("--fold-inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--alignment-inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-seeds", default="17,29,43")
    parser.add_argument("--expected-folds", type=int, default=5)
    args = parser.parse_args()
    args.stage1_summary = args.stage1_summary.resolve()
    args.fold_inputs = [path.resolve() for path in args.fold_inputs]
    args.alignment_inputs = [path.resolve() for path in args.alignment_inputs]
    args.output_dir = args.output_dir.resolve()
    for path in [args.stage1_summary] + args.fold_inputs + args.alignment_inputs:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if args.expected_folds < 2:
        parser.error("expected-folds must be >= 2.")
    return args


if __name__ == "__main__":
    summarize(parse_args())
