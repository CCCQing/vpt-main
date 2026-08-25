#!/usr/bin/env python3

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
METRICS = (
    "within_class_scatter_trace",
    "between_class_scatter_trace",
    "fisher_trace_ratio",
    "same_minus_interclass_cosine_gap",
    "leave_one_out_center_accuracy",
    "nearest_class_center_cosine_margin_mean",
    "effective_rank",
)
SPACES = (
    "prepass_cls_geometry",
    "trained_mu_geometry",
    "random_mu_geometry_mean",
    "normal_final_cls_geometry",
    "residual_zero_final_cls_geometry",
)
DELTA_NAMES = (
    "trained_mu_minus_prepass_cls",
    "trained_mu_minus_random_mean",
    "normal_final_cls_minus_residual_zero_final_cls",
)
ALIGNMENT_METRICS = (
    "class_center_prototype_cosine",
    "visual_semantic_structure_spearman",
    "visual_semantic_distance_spearman",
    "true_prototype_rank",
    "semantic_margin",
    "neighbor_preservation_at_k",
    "prototype_recall_at_k",
    "semantic_ambiguity_rate",
)
SEMANTIC_GRAPH_METRICS = (
    "neighbor_ranking_consistency",
    "confusion_edge_precision",
    "false_high_semantic_edge_rate",
)
LOGIT_VIEWS = ("centered_logits", "direction_normalized_logits", "class_pattern")
TASK_METRICS = (
    "seen_per_class_accuracy",
    "unseen_per_class_accuracy",
    "harmonic_mean",
    "ausuc",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("empty statistic")
    return {
        "mean": mean(data),
        "min": min(data),
        "max": max(data),
        "count": len(data),
        "positive_count": sum(value > 0.0 for value in data),
        "negative_count": sum(value < 0.0 for value in data),
        "zero_count": sum(value == 0.0 for value in data),
    }


def _run_dir(
    root: Path,
    method: str,
    training_seed: int,
    selection_seed: int | None,
) -> Path:
    base = root / method / f"seed{training_seed}"
    if selection_seed is None:
        return base / "full"
    return base / "probe" / f"selection_seed_{selection_seed}"


def _document(
    root: Path,
    method: str,
    training_seed: int,
    selection_seed: int | None,
) -> dict[str, Any]:
    return _load(
        _run_dir(root, method, training_seed, selection_seed)
        / "B3-D2G-source-to-decision-chain.json"
    )


def _validate(root: Path) -> dict[str, Any]:
    cells = []
    failures = []
    equivalence_groups: dict[str, list[dict[str, str]]] = {}
    for method in METHODS:
        for training_seed in TRAINING_SEEDS:
            for selection_seed in (None, *SELECTION_SEEDS):
                directory = _run_dir(root, method, training_seed, selection_seed)
                summary_path = directory / "b3_followup_summary.json"
                result_path = directory / "B3-D2G-source-to-decision-chain.json"
                missing = [
                    str(path)
                    for path in (summary_path, result_path)
                    if not path.is_file()
                ]
                valid = False
                if not missing:
                    summary = _load(summary_path)
                    result = _load(result_path)
                    valid = bool(
                        summary.get("status") == "completed"
                        and summary.get("valid")
                        and summary.get("experiments") == ["D2G"]
                        and summary.get("D2G", {}).get("valid")
                        and result.get("valid")
                        and result.get("format")
                        == "b3_d2g_source_to_decision_chain_v2"
                        and result.get("decision_space_dim") == 200
                        and result.get("downstream_validity", {}).get("valid")
                        and result.get("semantic_reference", {})
                        .get("validity", {})
                        .get("valid")
                        and all(
                            split in result.get("splits", {})
                            and result["splits"][split]["validity"]["valid"]
                            for split in SPLITS
                        )
                    )
                    if valid:
                        for split in SPLITS:
                            key = "{}:{}:{}".format(
                                "full" if selection_seed is None else "probe",
                                selection_seed if selection_seed is not None else "all",
                                split,
                            )
                            entry = result["splits"][split]
                            equivalence_groups.setdefault(key, []).append(
                                {
                                    "sample_manifest": entry["sample_manifest"]["sha256"],
                                    "prepass_exact": entry["prepass_equivalence"]["float32_sha256"],
                                    "prepass_rounded": entry["prepass_equivalence"]["rounded_1e-5_sha256"],
                                }
                            )
                cell = {
                    "method": method,
                    "training_seed": training_seed,
                    "scope": "full" if selection_seed is None else "probe",
                    "selection_seed": selection_seed,
                    "valid": valid,
                    "missing": missing,
                }
                cells.append(cell)
                if not valid:
                    failures.append(cell)
    equivalence = {}
    for key, values in equivalence_groups.items():
        sample_hashes = {item["sample_manifest"] for item in values}
        exact_hashes = {item["prepass_exact"] for item in values}
        rounded_hashes = {item["prepass_rounded"] for item in values}
        equivalence[key] = {
            "checkpoint_count": len(values),
            "sample_manifest_equal": len(sample_hashes) == 1,
            "prepass_exact_equal": len(exact_hashes) == 1,
            "prepass_rounded_1e5_equal": len(rounded_hashes) == 1,
            "valid": bool(
                len(values) == len(METHODS) * len(TRAINING_SEEDS)
                and len(sample_hashes) == 1
                and (len(exact_hashes) == 1 or len(rounded_hashes) == 1)
            ),
        }
    equivalence_complete = bool(
        len(equivalence) == (1 + len(SELECTION_SEEDS)) * len(SPLITS)
        and all(item["valid"] for item in equivalence.values())
    )
    complete = bool(len(cells) == 24 and not failures and equivalence_complete)
    return {
        "expected_cells": 24,
        "valid_cells": sum(int(cell["valid"]) for cell in cells),
        "full_valid": sum(
            int(cell["valid"] and cell["scope"] == "full") for cell in cells
        ),
        "probe_valid": sum(
            int(cell["valid"] and cell["scope"] == "probe") for cell in cells
        ),
        "failure_count": len(failures),
        "prepass_equivalence_complete": equivalence_complete,
        "prepass_equivalence": equivalence,
        "complete": complete,
        "failures": failures,
    }


def _random_mean_metrics(split: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(split["random_mu_geometry_summary"][metric]["mean"])
        for metric in METRICS
    }


def _space_metrics(split: dict[str, Any], space: str) -> dict[str, float]:
    if space == "random_mu_geometry_mean":
        return _random_mean_metrics(split)
    return {
        metric: float(split[space]["metrics"][metric]) for metric in METRICS
    }


def _full(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        documents = [
            _document(root, method, training_seed, None)
            for training_seed in TRAINING_SEEDS
        ]
        method_result: dict[str, Any] = {}
        for split_name in SPLITS:
            split_result: dict[str, Any] = {
                "spaces": {},
                "deltas": {},
                "relationships": {},
                "visual_semantic": {
                    "normal": {"alignment": {}, "semantic_graph": {}},
                    "residual_zero": {"alignment": {}, "semantic_graph": {}},
                    "normal_minus_residual_zero": {
                        "alignment": {},
                        "semantic_graph": {},
                    },
                },
                "logit_geometry": {
                    "normal": {},
                    "residual_zero": {},
                    "normal_minus_residual_zero": {},
                },
            }
            for space in SPACES:
                split_result["spaces"][space] = {
                    metric: _stats(
                        _space_metrics(document["splits"][split_name], space)[metric]
                        for document in documents
                    )
                    for metric in METRICS
                }
            for delta_name in DELTA_NAMES:
                split_result["deltas"][delta_name] = {
                    metric: _stats(
                        document["splits"][split_name]["paired_deltas"][delta_name][metric]
                        for document in documents
                    )
                    for metric in METRICS
                }
            for metric in (
                "sample_pair_distance_spearman",
                "class_center_distance_spearman",
                "linear_cka",
            ):
                split_result["relationships"][metric] = _stats(
                    document["splits"][split_name]["relationship_metrics"]["trained"][metric]
                    for document in documents
                )
            for condition in ("normal", "residual_zero"):
                for metric in ALIGNMENT_METRICS:
                    split_result["visual_semantic"][condition]["alignment"][
                        metric
                    ] = _stats(
                        document["splits"][split_name][
                            "visual_semantic_alignment"
                        ][condition]["alignment"][metric]
                        for document in documents
                    )
                for metric in SEMANTIC_GRAPH_METRICS:
                    split_result["visual_semantic"][condition]["semantic_graph"][
                        metric
                    ] = _stats(
                        document["splits"][split_name][
                            "visual_semantic_alignment"
                        ][condition]["semantic_graph"][metric]
                        for document in documents
                    )
            semantic_delta = split_result["visual_semantic"][
                "normal_minus_residual_zero"
            ]
            for metric in ALIGNMENT_METRICS:
                semantic_delta["alignment"][metric] = _stats(
                    document["splits"][split_name]["visual_semantic_alignment"][
                        "paired_deltas"
                    ]["normal_minus_residual_zero"]["alignment"][metric]
                    for document in documents
                )
            for metric in SEMANTIC_GRAPH_METRICS:
                semantic_delta["semantic_graph"][metric] = _stats(
                    document["splits"][split_name]["visual_semantic_alignment"][
                        "paired_deltas"
                    ]["normal_minus_residual_zero"]["semantic_graph"][metric]
                    for document in documents
                )
            for condition in ("normal", "residual_zero"):
                for view in LOGIT_VIEWS:
                    split_result["logit_geometry"][condition][view] = {
                        metric: _stats(
                            document["logit_geometry"][condition]["splits"][
                                split_name
                            ]["views"][view][metric]
                            for document in documents
                        )
                        for metric in METRICS
                    }
            for view in LOGIT_VIEWS:
                split_result["logit_geometry"]["normal_minus_residual_zero"][
                    view
                ] = {
                    metric: _stats(
                        document["logit_geometry"]["paired_deltas"][
                            "normal_minus_residual_zero"
                        ][split_name][view][metric]
                        for document in documents
                    )
                    for metric in METRICS
                }
            method_result[split_name] = split_result
        method_result["semantic_reference"] = {
            "effective_rank": _stats(
                document["semantic_reference"]["geometry"]["effective_rank"]
                for document in documents
            ),
            "prototype_count": sorted(
                {
                    int(document["semantic_reference"]["prototype_count"])
                    for document in documents
                }
            ),
        }
        method_result["task_results"] = {
            condition: {
                metric: _stats(
                    document["task_results"][condition]["gzsl"][metric]
                    for document in documents
                )
                for metric in TASK_METRICS
            }
            for condition in ("normal", "residual_zero")
        }
        method_result["task_results"]["normal_minus_residual_zero"] = {
            metric: _stats(
                document["task_results"]["normal_minus_residual_zero"][metric]
                for document in documents
            )
            for metric in TASK_METRICS
        }
        result[method] = method_result
    return result


def _hierarchical_probe(values: dict[int, list[float]]) -> dict[str, Any]:
    checkpoint_means = {
        str(training_seed): mean(items)
        for training_seed, items in values.items()
    }
    flattened = [value for items in values.values() for value in items]
    return {
        "checkpoint_probe_means": checkpoint_means,
        "across_training_seed_checkpoint_means": _stats(
            checkpoint_means.values()
        ),
        "all_selection_values": _stats(flattened),
    }


def _probe(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        method_result: dict[str, Any] = {}
        for split_name in SPLITS:
            split_result: dict[str, Any] = {
                "absolute_geometry_deltas": {},
                "visual_semantic_deltas": {
                    "alignment": {},
                    "semantic_graph": {},
                },
                "logit_geometry_deltas": {},
            }
            for delta_name in DELTA_NAMES:
                split_result["absolute_geometry_deltas"][delta_name] = {}
                for metric in METRICS:
                    values: dict[int, list[float]] = {}
                    for training_seed in TRAINING_SEEDS:
                        values[training_seed] = [
                            float(
                                _document(
                                    root,
                                    method,
                                    training_seed,
                                    selection_seed,
                                )["splits"][split_name]["paired_deltas"][delta_name][metric]
                            )
                            for selection_seed in SELECTION_SEEDS
                        ]
                    split_result["absolute_geometry_deltas"][delta_name][
                        metric
                    ] = _hierarchical_probe(values)
            for metric in ALIGNMENT_METRICS:
                values = {
                    training_seed: [
                        float(
                            _document(root, method, training_seed, selection_seed)[
                                "splits"
                            ][split_name]["visual_semantic_alignment"][
                                "paired_deltas"
                            ]["normal_minus_residual_zero"]["alignment"][metric]
                        )
                        for selection_seed in SELECTION_SEEDS
                    ]
                    for training_seed in TRAINING_SEEDS
                }
                split_result["visual_semantic_deltas"]["alignment"][
                    metric
                ] = _hierarchical_probe(values)
            for metric in SEMANTIC_GRAPH_METRICS:
                values = {
                    training_seed: [
                        float(
                            _document(root, method, training_seed, selection_seed)[
                                "splits"
                            ][split_name]["visual_semantic_alignment"][
                                "paired_deltas"
                            ]["normal_minus_residual_zero"]["semantic_graph"][metric]
                        )
                        for selection_seed in SELECTION_SEEDS
                    ]
                    for training_seed in TRAINING_SEEDS
                }
                split_result["visual_semantic_deltas"]["semantic_graph"][
                    metric
                ] = _hierarchical_probe(values)
            for view in LOGIT_VIEWS:
                split_result["logit_geometry_deltas"][view] = {}
                for metric in METRICS:
                    values = {
                        training_seed: [
                            float(
                                _document(
                                    root, method, training_seed, selection_seed
                                )["logit_geometry"]["paired_deltas"][
                                    "normal_minus_residual_zero"
                                ][split_name][view][metric]
                            )
                            for selection_seed in SELECTION_SEEDS
                        ]
                        for training_seed in TRAINING_SEEDS
                    }
                    split_result["logit_geometry_deltas"][view][
                        metric
                    ] = _hierarchical_probe(values)
            method_result[split_name] = split_result
        method_result["task_result_deltas"] = {}
        for metric in TASK_METRICS:
            values = {
                training_seed: [
                    float(
                        _document(root, method, training_seed, selection_seed)[
                            "task_results"
                        ]["normal_minus_residual_zero"][metric]
                    )
                    for selection_seed in SELECTION_SEEDS
                ]
                for training_seed in TRAINING_SEEDS
            }
            method_result["task_result_deltas"][metric] = _hierarchical_probe(values)
        result[method] = method_result
    return result


def _format(entry: dict[str, Any]) -> str:
    return "{:.4f} ({:.4f}~{:.4f})".format(
        entry["mean"], entry["min"], entry["max"]
    )


def _render(summary: dict[str, Any]) -> str:
    completeness = summary["completeness"]
    lines = [
        "# B3-D2-G aggregate",
        "",
        "Full values summarize three independent training seeds. Probe values are nested inside each checkpoint and are not independent training runs.",
        "",
        "## Completeness",
        "",
        "| valid/expected | full | probe | prepass equivalence | complete |",
        "|---:|---:|---:|---|---|",
        "| {}/{} | {} | {} | {} | {} |".format(
            completeness["valid_cells"],
            completeness["expected_cells"],
            completeness["full_valid"],
            completeness["probe_valid"],
            completeness["prepass_equivalence_complete"],
            completeness["complete"],
        ),
        "",
        "## Full absolute geometry",
        "",
        "| method | split | space | Fisher | cosine gap | LOO accuracy | center margin | effective rank |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for split_name in SPLITS:
            for space in SPACES:
                entry = summary["full"][method][split_name]["spaces"][space]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        method,
                        split_name,
                        space,
                        _format(entry["fisher_trace_ratio"]),
                        _format(entry["same_minus_interclass_cosine_gap"]),
                        _format(entry["leave_one_out_center_accuracy"]),
                        _format(entry["nearest_class_center_cosine_margin_mean"]),
                        _format(entry["effective_rank"]),
                    )
                )
    lines += [
        "",
        "## Full paired gains",
        "",
        "| method | split | contrast | dFisher | dGap | dLOO | dMargin | dRank |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for split_name in SPLITS:
            for delta_name in DELTA_NAMES:
                entry = summary["full"][method][split_name]["deltas"][delta_name]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        method,
                        split_name,
                        delta_name,
                        _format(entry["fisher_trace_ratio"]),
                        _format(entry["same_minus_interclass_cosine_gap"]),
                        _format(entry["leave_one_out_center_accuracy"]),
                        _format(entry["nearest_class_center_cosine_margin_mean"]),
                        _format(entry["effective_rank"]),
                    )
                )
    lines += [
        "",
        "## Full source-to-decision paired chain",
        "",
        "All values are normal minus Residual-zero inside the same checkpoint and sample manifest.",
        "",
        "| method | split | dCLS Fisher | dSemantic margin | dLogit class-pattern Fisher |",
        "|---|---|---:|---:|---:|",
    ]
    for method in METHODS:
        for split_name in SPLITS:
            split = summary["full"][method][split_name]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    method,
                    split_name,
                    _format(
                        split["deltas"][
                            "normal_final_cls_minus_residual_zero_final_cls"
                        ]["fisher_trace_ratio"]
                    ),
                    _format(
                        split["visual_semantic"]["normal_minus_residual_zero"][
                            "alignment"
                        ]["semantic_margin"]
                    ),
                    _format(
                        split["logit_geometry"]["normal_minus_residual_zero"][
                            "class_pattern"
                        ]["fisher_trace_ratio"]
                    ),
                )
            )
    lines += [
        "",
        "## Full task endpoints",
        "",
        "| method | condition | Seen | Unseen | H | AUSUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        task = summary["full"][method]["task_results"]
        for condition in ("normal", "residual_zero", "normal_minus_residual_zero"):
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    method,
                    condition,
                    _format(task[condition]["seen_per_class_accuracy"]),
                    _format(task[condition]["unseen_per_class_accuracy"]),
                    _format(task[condition]["harmonic_mean"]),
                    _format(task[condition]["ausuc"]),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    root = args.input_root.expanduser().resolve()
    output_dir = (args.output_dir or root / "analysis_summary").resolve()
    completeness = _validate(root)
    if not completeness["complete"]:
        raise RuntimeError("D2-G evidence matrix is incomplete: {}".format(completeness))
    summary = {
        "format": "b3_d2g_aggregate_v1",
        "statistical_identity": {
            "full": "three independent training seeds",
            "probe": "three selection seeds nested inside each checkpoint",
            "random_mlp": "five random seeds nested inside each checkpoint",
        },
        "completeness": completeness,
        "full": _full(root),
        "probe_robustness": _probe(root),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "b3_d2g_aggregate.json"
    markdown_path = output_dir / "b3_d2g_aggregate.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": True,
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
