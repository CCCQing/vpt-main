#!/usr/bin/env python3
"""Validate and aggregate the checkpoint-only B3 D2/D3/D4 follow-up matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


METHODS = ("B3-R1I-R025", "B3-R1I-R050")
TRAINING_SEEDS = (0, 1, 2)
SELECTION_SEEDS = (424242, 424243, 424244)
SPLITS = ("train_seen", "test_seen", "test_unseen")
D2_METRICS = (
    "sample_pair_distance_spearman",
    "class_center_distance_spearman",
    "linear_cka",
)
D3_LAYERS = (8, 9, 10, 11)
D3_METRICS = (
    "applied_group_rms_ratio",
    "common_component_rms",
    "role_component_rms",
    "delta_raw__effective_rank",
    "delta_raw__offdiag_cosine",
    "delta_layernorm__effective_rank",
    "delta_layernorm__offdiag_cosine",
    "delta_key__effective_rank",
    "delta_key__offdiag_cosine",
    "delta_value__effective_rank",
    "delta_value__offdiag_cosine",
    "residual_static_slot_cosine",
    "residual_static_cosine_across_slot_std",
    "residual_static_signed_parallel_projection",
    "residual_static_orthogonal_ratio",
    "slot_relation_spearman_before_after_injection",
    "cls_to_prompt__mass_delta",
    "cls_to_prompt__slot_js",
    "cls_to_prompt__slot_spearman",
    "patch_to_prompt__mass_delta",
    "patch_to_prompt__slot_js",
    "patch_to_prompt__slot_spearman",
    "cls_value_contribution_effective_count",
    "cls_value_contribution_top4_share",
    "patch_value_contribution_effective_count",
    "patch_value_contribution_top4_share",
)
D4_CONDITIONS = (
    "normal",
    "same_class_loo_center",
    "same_class_loo_center_source_norm_matched",
    "global_loo_center",
    "global_loo_center_source_norm_matched",
    "wrong_class_center_mean",
)
GZSL_METRICS = (
    "seen_per_class_accuracy",
    "unseen_per_class_accuracy",
    "harmonic_mean",
    "ausuc",
)
PAIRED_METRICS = (
    "delta_logits_norm",
    "delta_true_margin",
    "prediction_flip_rate",
    "beneficial_flip_rate",
    "harmful_flip_rate",
    "net_beneficial_flip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot summarize an empty value list")
    return {
        "mean": mean(data),
        "min": min(data),
        "max": max(data),
        "count": len(data),
    }


def summary_mean(value: Any) -> float:
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def run_dir(root: Path, method: str, training_seed: int, selection_seed: int | None) -> Path:
    base = root / method / f"seed{training_seed}"
    if selection_seed is None:
        return base / "full"
    return base / "probe" / f"selection_seed_{selection_seed}"


def validate(root: Path) -> dict[str, Any]:
    cells = []
    failures = []
    for method in METHODS:
        for training_seed in TRAINING_SEEDS:
            for selection_seed in (None, *SELECTION_SEEDS):
                directory = run_dir(root, method, training_seed, selection_seed)
                expected = [
                    directory / "b3_followup_summary.json",
                    directory / "B3-D2-prepass-to-mu.json",
                    directory / "B3-D3-slot-propagation.json",
                    directory / "B3-D4-class-center-oracle.json",
                ]
                missing = [str(path) for path in expected if not path.is_file()]
                status = None
                valid = False
                if not missing:
                    run_summary = load_json(expected[0])
                    status = run_summary.get("status")
                    valid = bool(run_summary.get("valid")) and status == "completed"
                    valid = valid and all(bool(run_summary[name]["valid"]) for name in ("D2", "D3", "D4"))
                cell = {
                    "method": method,
                    "training_seed": training_seed,
                    "scope": "full" if selection_seed is None else "probe",
                    "selection_seed": selection_seed,
                    "valid": valid,
                    "status": status,
                    "missing": missing,
                }
                cells.append(cell)
                if not valid:
                    failures.append(cell)
    return {
        "expected_cells": 24,
        "observed_cells": len(cells),
        "valid_cells": sum(int(cell["valid"]) for cell in cells),
        "full_valid": sum(int(cell["valid"] and cell["scope"] == "full") for cell in cells),
        "probe_valid": sum(int(cell["valid"] and cell["scope"] == "probe") for cell in cells),
        "failure_count": len(failures),
        "complete": len(cells) == 24 and not failures,
        "failures": failures,
    }


def d2_full(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        method_result: dict[str, Any] = {}
        docs = [
            load_json(run_dir(root, method, seed, None) / "B3-D2-prepass-to-mu.json")
            for seed in TRAINING_SEEDS
        ]
        for split in SPLITS:
            split_result: dict[str, Any] = {}
            for metric in D2_METRICS:
                trained = [doc["splits"][split]["trained"][metric] for doc in docs]
                random_mean = [
                    doc["splits"][split]["trained_vs_random"][metric]["random_mean"]
                    for doc in docs
                ]
                delta = [left - right for left, right in zip(trained, random_mean)]
                above_all = [
                    bool(doc["splits"][split]["trained_vs_random"][metric]["trained_above_all_random_controls"])
                    for doc in docs
                ]
                split_result[metric] = {
                    "trained": stats(trained),
                    "random_mean": stats(random_mean),
                    "trained_minus_random_mean": stats(delta),
                    "trained_above_all_random_controls_count": sum(above_all),
                }
            method_result[split] = split_result
        result[method] = method_result
    return result


def d3_full(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        docs = [
            load_json(run_dir(root, method, seed, None) / "B3-D3-slot-propagation.json")
            for seed in TRAINING_SEEDS
        ]
        method_result: dict[str, Any] = {"layers": {}, "counterfactuals": {}}
        for split in ("test_seen", "test_unseen"):
            split_result: dict[str, Any] = {}
            for layer in D3_LAYERS:
                layer_result: dict[str, Any] = {}
                for metric in D3_METRICS:
                    key = f"layer_{layer}__{metric}"
                    layer_result[metric] = stats(
                        summary_mean(doc["metrics"][split][key]) for doc in docs
                    )
                split_result[str(layer)] = layer_result
            method_result["layers"][split] = split_result
        for condition in ("common_only", "role_only", "role_permuted"):
            condition_result: dict[str, Any] = {"gzsl": {}}
            for metric in GZSL_METRICS:
                condition_result["gzsl"][metric] = stats(
                    doc["common_role_counterfactuals"]["conditions"][condition]["gzsl"][metric]
                    for doc in docs
                )
            condition_result["paired"] = {}
            for split in ("test_seen", "test_unseen"):
                condition_result["paired"][split] = {
                    metric: stats(
                        doc["common_role_counterfactuals"]["conditions"][condition]["splits"][split]["paired_vs_normal"]["summary"][metric]
                        for doc in docs
                    )
                    for metric in PAIRED_METRICS
                }
            method_result["counterfactuals"][condition] = condition_result
        result[method] = method_result
    return result


def wrong_condition(doc: dict[str, Any]) -> dict[str, Any]:
    conditions = [
        value
        for name, value in doc["conditions"].items()
        if name.startswith("wrong_class_center_seed_")
    ]
    return {
        "gzsl": {
            metric: mean(float(condition["gzsl"][metric]) for condition in conditions)
            for metric in GZSL_METRICS
        },
        "paired": {
            split: {
                metric: mean(
                    float(condition["splits"][split]["paired_vs_normal"]["summary"][metric])
                    for condition in conditions
                )
                for metric in PAIRED_METRICS
            }
            for split in ("test_seen", "test_unseen")
        },
    }


def d4_condition(doc_d3: dict[str, Any], doc_d4: dict[str, Any], name: str) -> dict[str, Any]:
    if name == "normal":
        condition = doc_d3["common_role_counterfactuals"]["conditions"]["common_only"]
        return {
            "gzsl": condition["gzsl"],
            "paired": {
                split: {metric: 0.0 for metric in PAIRED_METRICS}
                for split in ("test_seen", "test_unseen")
            },
        }
    if name == "wrong_class_center_mean":
        return wrong_condition(doc_d4)
    condition = doc_d4["conditions"][name]
    return {
        "gzsl": condition["gzsl"],
        "paired": {
            split: condition["splits"][split]["paired_vs_normal"]["summary"]
            for split in ("test_seen", "test_unseen")
        },
    }


def d4_full(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        pairs = [
            (
                load_json(run_dir(root, method, seed, None) / "B3-D3-slot-propagation.json"),
                load_json(run_dir(root, method, seed, None) / "B3-D4-class-center-oracle.json"),
            )
            for seed in TRAINING_SEEDS
        ]
        method_result: dict[str, Any] = {}
        for name in D4_CONDITIONS:
            values = [d4_condition(d3, d4, name) for d3, d4 in pairs]
            condition_result = {
                "gzsl": {
                    metric: stats(value["gzsl"][metric] for value in values)
                    for metric in GZSL_METRICS
                },
                "paired": {
                    split: {
                        metric: stats(value["paired"][split][metric] for value in values)
                        for metric in PAIRED_METRICS
                    }
                    for split in ("test_seen", "test_unseen")
                },
            }
            method_result[name] = condition_result
        normal = method_result["normal"]["gzsl"]
        same = method_result["same_class_loo_center"]["gzsl"]
        global_center = method_result["global_loo_center"]["gzsl"]
        wrong = method_result["wrong_class_center_mean"]["gzsl"]
        method_result["contrasts"] = {}
        for metric in GZSL_METRICS:
            # Contrast summaries must preserve pairing, not subtract aggregate ranges.
            per_seed = []
            per_seed_global = []
            per_seed_wrong = []
            for d3_doc, d4_doc in pairs:
                normal_value = d4_condition(d3_doc, d4_doc, "normal")["gzsl"][metric]
                same_value = d4_condition(d3_doc, d4_doc, "same_class_loo_center")["gzsl"][metric]
                global_value = d4_condition(d3_doc, d4_doc, "global_loo_center")["gzsl"][metric]
                wrong_value = d4_condition(d3_doc, d4_doc, "wrong_class_center_mean")["gzsl"][metric]
                per_seed.append(same_value - normal_value)
                per_seed_global.append(same_value - global_value)
                per_seed_wrong.append(same_value - wrong_value)
            method_result["contrasts"][metric] = {
                "same_minus_normal": stats(per_seed),
                "same_minus_global": stats(per_seed_global),
                "same_minus_wrong_mean": stats(per_seed_wrong),
            }
        # These reads make an accidental missing condition obvious in the JSON.
        assert normal and same and global_center and wrong
        result[method] = method_result
    return result


def hierarchical_probe(values_by_checkpoint: dict[int, list[float]]) -> dict[str, Any]:
    checkpoint_means = {str(seed): mean(values) for seed, values in values_by_checkpoint.items()}
    all_values = [value for values in values_by_checkpoint.values() for value in values]
    return {
        "checkpoint_probe_means": checkpoint_means,
        "across_training_seed_checkpoint_means": stats(checkpoint_means.values()),
        "all_selection_values_range": {"min": min(all_values), "max": max(all_values), "count": len(all_values)},
    }


def probe_robustness(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        method_result: dict[str, Any] = {"D2": {}, "D3": {}, "D4": {}}
        for split in SPLITS:
            method_result["D2"][split] = {}
            for metric in D2_METRICS:
                values: dict[int, list[float]] = {}
                for seed in TRAINING_SEEDS:
                    values[seed] = []
                    for selection_seed in SELECTION_SEEDS:
                        doc = load_json(run_dir(root, method, seed, selection_seed) / "B3-D2-prepass-to-mu.json")
                        entry = doc["splits"][split]["trained_vs_random"][metric]
                        values[seed].append(float(entry["trained_minus_random_mean"]))
                method_result["D2"][split][metric] = hierarchical_probe(values)

        for metric in (
            "delta_raw__effective_rank",
            "delta_layernorm__effective_rank",
            "delta_value__effective_rank",
            "role_component_rms",
            "cls_to_prompt__slot_js",
            "patch_to_prompt__slot_js",
            "slot_relation_spearman_before_after_injection",
        ):
            values = {}
            for seed in TRAINING_SEEDS:
                values[seed] = []
                for selection_seed in SELECTION_SEEDS:
                    doc = load_json(run_dir(root, method, seed, selection_seed) / "B3-D3-slot-propagation.json")
                    layer_means = [
                        summary_mean(doc["metrics"]["test_unseen"][f"layer_{layer}__{metric}"])
                        for layer in D3_LAYERS
                    ]
                    values[seed].append(mean(layer_means))
            method_result["D3"][metric] = hierarchical_probe(values)

        for contrast_name in ("same_minus_normal", "same_minus_global", "same_minus_wrong_mean"):
            method_result["D4"][contrast_name] = {}
            for metric in ("unseen_per_class_accuracy", "harmonic_mean", "ausuc"):
                values = {}
                for seed in TRAINING_SEEDS:
                    values[seed] = []
                    for selection_seed in SELECTION_SEEDS:
                        directory = run_dir(root, method, seed, selection_seed)
                        d3_doc = load_json(directory / "B3-D3-slot-propagation.json")
                        d4_doc = load_json(directory / "B3-D4-class-center-oracle.json")
                        normal_value = d4_condition(d3_doc, d4_doc, "normal")["gzsl"][metric]
                        if contrast_name == "same_minus_normal":
                            reference = normal_value
                        elif contrast_name == "same_minus_global":
                            reference = d4_condition(d3_doc, d4_doc, "global_loo_center")["gzsl"][metric]
                        else:
                            reference = d4_condition(d3_doc, d4_doc, "wrong_class_center_mean")["gzsl"][metric]
                        same_value = d4_condition(d3_doc, d4_doc, "same_class_loo_center")["gzsl"][metric]
                        values[seed].append(same_value - reference)
                method_result["D4"][contrast_name][metric] = hierarchical_probe(values)
        result[method] = method_result
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# B3 D2-D4 checkpoint-only aggregate",
        "",
        "Values are full-test means over three training seeds unless noted. Probe values use a hierarchical aggregation: first average the three selection seeds inside each checkpoint, then summarize the three checkpoint means.",
        "",
        "## Completeness",
        "",
        "| expected | valid | full | probe | failures | complete |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    completeness = summary["completeness"]
    lines.append(
        f"| {completeness['expected_cells']} | {completeness['valid_cells']} | {completeness['full_valid']} | {completeness['probe_valid']} | {completeness['failure_count']} | {str(completeness['complete']).lower()} |"
    )
    lines += ["", "## D2 full", "", "| method | split | metric | trained | random mean | trained-random | above all random seeds |", "|---|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        for split in SPLITS:
            for metric in D2_METRICS:
                entry = summary["full"]["D2"][method][split][metric]
                lines.append(
                    f"| {method} | {split} | {metric} | {entry['trained']['mean']:.4f} | {entry['random_mean']['mean']:.4f} | {entry['trained_minus_random_mean']['mean']:+.4f} | {entry['trained_above_all_random_controls_count']}/3 |"
                )
    lines += ["", "## D3 full test-unseen", "", "| method | layer | raw rank | LN rank | K rank | V rank | role RMS | CLS slot JS | Patch slot JS | slot relation rho |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for layer in D3_LAYERS:
            entry = summary["full"]["D3"][method]["layers"]["test_unseen"][str(layer)]
            lines.append(
                f"| {method} | {layer} | {entry['delta_raw__effective_rank']['mean']:.4f} | {entry['delta_layernorm__effective_rank']['mean']:.4f} | {entry['delta_key__effective_rank']['mean']:.4f} | {entry['delta_value__effective_rank']['mean']:.4f} | {entry['role_component_rms']['mean']:.6f} | {entry['cls_to_prompt__slot_js']['mean']:.6f} | {entry['patch_to_prompt__slot_js']['mean']:.6f} | {entry['slot_relation_spearman_before_after_injection']['mean']:.4f} |"
            )
    lines += ["", "## D4 full", "", "| method | condition | Seen | Unseen | H | AUSUC | test-unseen net flip |", "|---|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for condition in ("normal", "same_class_loo_center", "global_loo_center", "wrong_class_center_mean"):
            entry = summary["full"]["D4"][method][condition]
            lines.append(
                f"| {method} | {condition} | {entry['gzsl']['seen_per_class_accuracy']['mean']:.4f} | {entry['gzsl']['unseen_per_class_accuracy']['mean']:.4f} | {entry['gzsl']['harmonic_mean']['mean']:.4f} | {entry['gzsl']['ausuc']['mean']:.4f} | {entry['paired']['test_unseen']['net_beneficial_flip']['mean']:+.4f} |"
            )
    lines += ["", "## D4 paired contrasts", "", "| method | metric | same-normal | same-global | same-wrong mean |", "|---|---|---:|---:|---:|"]
    for method in METHODS:
        for metric in ("unseen_per_class_accuracy", "harmonic_mean", "ausuc"):
            entry = summary["full"]["D4"][method]["contrasts"][metric]
            lines.append(
                f"| {method} | {metric} | {entry['same_minus_normal']['mean']:+.4f} | {entry['same_minus_global']['mean']:+.4f} | {entry['same_minus_wrong_mean']['mean']:+.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = args.input_root.resolve()
    output_dir = (args.output_dir or (root / "analysis_summary")).resolve()
    completeness = validate(root)
    if not completeness["complete"]:
        raise RuntimeError(f"B3 follow-up matrix incomplete: {completeness}")
    summary = {
        "format": "b3_followup_aggregate_v1",
        "input_root": str(root),
        "statistical_identity": {
            "full": "three independent training seeds",
            "probe": "three selection seeds nested inside each checkpoint, then three training seeds",
            "wrong_class": "three wrong-center seeds averaged inside each checkpoint",
        },
        "completeness": completeness,
        "full": {
            "D2": d2_full(root),
            "D3": d3_full(root),
            "D4": d4_full(root),
        },
        "probe_robustness": probe_robustness(root),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "b3_followup_aggregate.json"
    md_path = output_dir / "b3_followup_aggregate.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"complete": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
