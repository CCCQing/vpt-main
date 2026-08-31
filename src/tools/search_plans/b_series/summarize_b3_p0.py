#!/usr/bin/env python3
"""Validate and aggregate the fixed B-series P0 evidence matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


METHODS = ("B3-R1I-R025", "B3-R1I-R050")
TRAINING_SEEDS = (0, 1, 2)
SELECTION_SEEDS = (424242, 424243, 424244)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: Iterable[float]):
    rows = [float(value) for value in values]
    return {
        "mean": float(mean(rows)),
        "min": float(min(rows)),
        "max": float(max(rows)),
        "count": len(rows),
    }


def _result(path: Path):
    summary = _load(path / "p0_summary.json")
    if summary.get("status") != "completed" or summary.get("valid") is not True:
        raise ValueError("invalid P0 summary: {}".format(path))
    result = _load(path / summary["result_path"])
    if result.get("valid") is not True:
        raise ValueError("invalid P0 result: {}".format(path))
    return result


def _scope_path(root: Path, experiment: str, method: str, seed: int, selection):
    base = root / experiment / method / "seed{}".format(seed)
    return base / "full" if selection is None else base / "probe" / "selection_seed_{}".format(selection)


def _p03_paired_metrics(result: Mapping[str, Any], condition: str):
    p03 = result["P0-3"]
    stored = p03.get("paired_312_vs_768", {}).get(condition)
    if stored is not None:
        return stored
    raw = p03["spaces"]["attribute_312"]["conditions"][condition]["metrics"]
    projected = p03["spaces"]["projected_768"]["conditions"][condition]["metrics"]
    sources = {
        "predicted_unseen_center_cosine": "predicted_to_true_center_cosine_mean",
        "predicted_unseen_relation_spearman": "relation_spearman",
        "predicted_unseen_neighbor_recovery_at_5": "visual_neighbor_recovery_at_k",
        "unseen_center_top1": "unseen_only_top1",
        "unseen_center_nll": "unseen_only_nll",
    }
    paired = {}
    for public_name, source_name in sources.items():
        raw_value = float(raw[source_name])
        projected_value = float(projected[source_name])
        paired[public_name + "_312"] = raw_value
        paired[public_name + "_768"] = projected_value
        if public_name == "unseen_center_nll":
            paired["unseen_center_nll_improvement_768_over_312"] = float(
                raw_value - projected_value
            )
        else:
            paired[public_name + "_delta_768_minus_312"] = float(
                projected_value - raw_value
            )
    return paired


def _aggregate_p02(root: Path):
    report = {}
    for method in METHODS:
        full = [_result(_scope_path(root, "P0-2", method, seed, None)) for seed in TRAINING_SEEDS]
        conditions = {}
        for condition in full[0]["conditions"]:
            conditions[condition] = {
                metric: _stats(
                    item["conditions"][condition]["gzsl"][metric] for item in full
                )
                for metric in (
                    "seen_per_class_accuracy",
                    "unseen_per_class_accuracy",
                    "harmonic_mean",
                    "ausuc",
                )
            }
        probe = {}
        for selection in SELECTION_SEEDS:
            rows = [
                _result(_scope_path(root, "P0-2", method, seed, selection))
                for seed in TRAINING_SEEDS
            ]
            probe[str(selection)] = {
                condition: {
                    "test_unseen_delta_true_margin": _stats(
                        item["conditions"][condition]["paired_vs_residual_zero"]["test_unseen"]["summary"]["delta_true_margin"]
                        for item in rows
                    ),
                    "test_unseen_prediction_flip": _stats(
                        item["conditions"][condition]["paired_vs_residual_zero"]["test_unseen"]["summary"]["prediction_flip_rate"]
                        for item in rows
                    ),
                }
                for condition in rows[0]["conditions"]
            }
        report[method] = {"full": conditions, "strict_three_probe": probe}
    return report


def _aggregate_p034(root: Path):
    full = [_result(_scope_path(root, "P0-34", "A2", seed, None)) for seed in TRAINING_SEEDS]
    p03 = {}
    for space in ("attribute_312", "projected_768"):
        p03[space] = {}
        for condition in ("positive_cosine", "semantic_1nn", "global_seen_mean"):
            metric_names = full[0]["P0-3"]["spaces"][space]["conditions"][condition]["metrics"]
            p03[space][condition] = {
                metric: _stats(
                    item["P0-3"]["spaces"][space]["conditions"][condition]["metrics"][metric]
                    for item in full
                )
                for metric, value in metric_names.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        shuffled_names = full[0]["P0-3"]["spaces"][space]["class_shuffled_summary"]
        p03[space]["class_shuffled"] = {
            metric: _stats(
                item["P0-3"]["spaces"][space]["class_shuffled_summary"][metric]["mean"]
                for item in full
            )
            for metric in shuffled_names
        }
    paired = {}
    for condition in ("positive_cosine", "semantic_1nn", "global_seen_mean"):
        metric_names = _p03_paired_metrics(full[0], condition)
        paired[condition] = {
            metric: _stats(
                _p03_paired_metrics(item, condition)[metric]
                for item in full
            )
            for metric in metric_names
        }
    p03["paired_312_vs_768"] = paired
    p04 = {}
    for condition in (
        "current_semantic_dot",
        "semantic_cosine",
        "loo_visual_centroid_cosine",
    ):
        p04[condition] = {
            metric: _stats(item["P0-4"]["conditions"][condition]["gzsl"][metric] for item in full)
            for metric in (
                "seen_per_class_accuracy",
                "unseen_per_class_accuracy",
                "harmonic_mean",
                "ausuc",
            )
        }
    probes = {}
    for selection in SELECTION_SEEDS:
        rows = [
            _result(_scope_path(root, "P0-34", "A2", seed, selection))
            for seed in TRAINING_SEEDS
        ]
        probes[str(selection)] = {
            condition: {
                metric: _stats(item["P0-4"]["conditions"][condition]["gzsl"][metric] for item in rows)
                for metric in ("unseen_per_class_accuracy", "harmonic_mean", "ausuc")
            }
            for condition in (
                "current_semantic_dot",
                "semantic_cosine",
                "loo_visual_centroid_cosine",
            )
        }
    return {"P0-3": p03, "P0-4": p04, "strict_three_probe": probes}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# B-series P0 checkpoint-only aggregate",
        "",
        "- status: complete only after all 36 atomic jobs pass their validators",
        "- training_performed: false",
        "- optimizer_created: false",
        "- backward_performed: false",
        "- P0-3 uses official Seen to Unseen evaluation; true Unseen centers are oracle-only evaluation targets.",
        "",
        "## P0-2 full-test H",
        "",
        "| method | condition | mean | min | max |",
        "|---|---|---:|---:|---:|",
    ]
    for method, payload in report["P0-2"].items():
        for condition, metrics in payload["full"].items():
            row = metrics["harmonic_mean"]
            lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(method, condition, row["mean"], row["min"], row["max"]))
    lines.extend([
        "",
        "## P0-3 unseen-only top1",
        "",
        "| semantic space | condition | mean | min | max |",
        "|---|---|---:|---:|---:|",
    ])
    for space in ("attribute_312", "projected_768"):
        payload = report["P0-34"]["P0-3"][space]
        for condition in ("positive_cosine", "semantic_1nn", "global_seen_mean"):
            row = payload[condition]["unseen_only_top1"]
            lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(space, condition, row["mean"], row["min"], row["max"]))
    lines.extend([
        "",
        "## P0-4 full-test H",
        "",
        "| condition | mean | min | max |",
        "|---|---:|---:|---:|",
    ])
    for condition, metrics in report["P0-34"]["P0-4"].items():
        row = metrics["harmonic_mean"]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(condition, row["mean"], row["min"], row["max"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "b3_p0_aggregate_v1",
        "expected_atomic_job_count": 36,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "P0-2": _aggregate_p02(root),
        "P0-34": _aggregate_p034(root),
        "valid": True,
    }
    (output / "b3_p0_aggregate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "b3_p0_aggregate.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"valid": True, "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
