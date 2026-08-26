#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _range(values):
    rows = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(rows)),
        "min": float(min(rows)),
        "max": float(max(rows)),
        "values": rows,
    }


def summarize(root: Path) -> Dict[str, Any]:
    result_paths = sorted(root.glob("seed*/p13a_result.json"))
    if len(result_paths) != 3:
        raise ValueError("P1-3a aggregate requires exactly three seed results")
    results = [_read(path) for path in result_paths]
    seeds = [int(item["head_seed"]) for item in results]
    if sorted(seeds) != [0, 1, 2] or len(set(seeds)) != 3:
        raise ValueError("P1-3a aggregate requires head seeds 0/1/2")
    conditions = sorted(set.intersection(*(set(item["conditions"]) for item in results)))
    summary = {}
    for condition in conditions:
        rows = [item["conditions"][condition] for item in results]
        summary[condition] = {
            "normal_gzsl": {
                key: _range(
                    row["task_metrics"]["normal_gzsl"][key] for row in rows
                )
                for key in ("seen", "unseen", "harmonic_mean", "ausuc")
            },
            "test_unseen": {
                "nll": _range(
                    row["task_metrics"]["splits"]["test_unseen"][
                        "classification"
                    ]["nll"]
                    for row in rows
                ),
                "true_hard_negative_margin": _range(
                    row["task_metrics"]["splits"]["test_unseen"][
                        "semantic_hard_negative"
                    ]["true_vs_semantic_hard_negative_margin_mean"]
                    for row in rows
                ),
                "wrong_domain_prediction_rate": _range(
                    row["task_metrics"]["splits"]["test_unseen"][
                        "prediction_health"
                    ]["wrong_domain_prediction_rate"]
                    for row in rows
                ),
            },
            "selection": (
                {
                    "heldout_macro_accuracy": _range(
                        row["selection"]["best_heldout_macro_accuracy"]
                        for row in rows
                    ),
                    "selected_epoch": _range(
                        row["selection"]["selected_epoch"] for row in rows
                    ),
                }
                if all("selection" in row for row in rows)
                else None
            ),
        }
    main = summary["candidate_raw312"]["normal_gzsl"]
    comparisons = {}
    for reference in (
        "current_semantic_dot",
        "semantic_cosine",
        "candidate_projected768",
        "semantic_permuted_raw312",
        "image_constant_raw312",
        "image_only_temperature",
    ):
        comparisons["candidate_raw312_minus_{}".format(reference)] = {
            metric: _range(
                results[index]["conditions"]["candidate_raw312"]["task_metrics"][
                    "normal_gzsl"
                ][metric]
                - results[index]["conditions"][reference]["task_metrics"][
                    "normal_gzsl"
                ][metric]
                for index in range(3)
            )
            for metric in ("seen", "unseen", "harmonic_mean", "ausuc")
        }
    valid = bool(
        len(conditions) == 8
        and all(item.get("valid") is True for item in results)
        and all(
            item["execution_contract"]["normal_unseen_used_for_model_selection"]
            is False
            for item in results
        )
    )
    return {
        "format": "p13a_candidate_conditioned_head_only_aggregate_v1",
        "status": "completed" if valid else "invalid",
        "valid": valid,
        "training_seeds": seeds,
        "training_seed_count": len(seeds),
        "probe_selection_seeds": [424242, 424243, 424244],
        "conditions": summary,
        "paired_comparisons": comparisons,
        "primary_endpoint": main,
        "conclusion_status": "requires_analysis",
    }


def main() -> None:
    args = _parse()
    root = args.root.expanduser().resolve()
    payload = summarize(root)
    path = root / "p13a_aggregate.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"valid": payload["valid"], "output": str(path)}, ensure_ascii=False))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
