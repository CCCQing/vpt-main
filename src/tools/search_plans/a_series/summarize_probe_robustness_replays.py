#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from .artifact_io import open_text_artifact
except ImportError:
    from artifact_io import open_text_artifact

try:
    from .probe_evidence import is_probe_mechanism_evidence
except ImportError:  # Direct script execution.
    from probe_evidence import is_probe_mechanism_evidence


DEFAULT_METHODS = ("A0", "A1", "A2")
SPLITS = ("probe_train_seen", "probe_test_seen", "probe_test_unseen")
RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
TARGET_RELEVANCE_DOMAIN = "target_relevance_reference"
TARGET_RELEVANCE_AGGREGATE_PATHS = {
    "cls_to_patch",
    "cls_to_prompt",
    "patch_to_cls",
    "patch_to_patch",
    "patch_to_prompt",
    "prompt_to_cls",
    "prompt_to_patch",
    "prompt_to_prompt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit additional fixed-probe selection replays and summarize "
            "within-checkpoint probe-selection variability without inflating training n."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--primary-gap-root",
        help=(
            "Optional checkpoint-only gap-replay root to merge into the primary "
            "selection seed. Omit when the source runs already contain the complete "
            "primary fixed-probe evidence."
        ),
    )
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-selection-seed", type=int, default=424242)
    parser.add_argument(
        "--robustness-selection-seeds", default="424243,424244"
    )
    parser.add_argument("--training-seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    return parser.parse_args()


def _parse_int_list(value):
    items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not items or any(item < 0 for item in items) or len(set(items)) != len(items):
        raise ValueError(f"invalid unique non-negative integer list: {value}")
    return items


def _parse_name_list(value):
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items or len(set(items)) != len(items):
        raise ValueError(f"invalid unique method list: {value}")
    if any("/" in item or "\\" in item or item in {".", ".."} for item in items):
        raise ValueError(f"method names must be path components: {value}")
    return items


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_dir(root, method, training_seed):
    return Path(root) / method / f"seed{training_seed}" / RUN_SUFFIX


def _replay_run_dir(root, selection_seed, method, training_seed):
    return (
        Path(root)
        / f"selection_seed_{selection_seed}"
        / method
        / f"seed{training_seed}"
        / RUN_SUFFIX
    )


def _manifest_record(run_dir):
    runtime = _read_json(run_dir / "diagnostics" / "probe_runtime_summary.json")
    manifest = _read_json(run_dir / "diagnostics" / "probe_manifest.json")
    probe_loader = dict(runtime.get("probe_loader") or {})
    if "batch_size" not in probe_loader:
        config = yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        probe_loader["batch_size"] = int(
            config["MONITOR"]["PROBE"]["BATCH_SIZE"]
        )
    execution_profile = str(runtime.get("execution_profile") or "final_full")
    required_splits = list(runtime.get("required_splits") or SPLITS)
    return {
        "runtime_status": runtime.get("status"),
        "execution_profile": execution_profile,
        "required_splits": required_splits,
        "selection_seed": int(runtime.get("selection_seed", -1)),
        "metric_row_count": int(runtime.get("metric_row_count", 0)),
        "probe_loader": probe_loader,
        "checkpoint": dict(runtime.get("checkpoint") or {}),
        "manifest_sha256_by_split": {
            split: manifest.get("probes", {}).get(split, {}).get("manifest_sha256")
            for split in required_splits
        },
        "sample_count_by_split": {
            split: manifest.get("probes", {})
            .get(split, {})
            .get("selected_sample_count")
            for split in required_splits
        },
        "valid_by_split": {
            split: bool(manifest.get("validity", {}).get(split, {}).get("valid", False))
            for split in required_splits
        },
    }


def _identity(row):
    objective = str(row.get("objective", ""))
    if not objective and str(row.get("domain", "")) == TARGET_RELEVANCE_DOMAIN:
        # probe_metrics.csv exports the backward-compatible primary objective.
        objective = "true_class_margin"
    return "|".join(
        [
            str(row.get("split", "")),
            str(row.get("condition", "")),
            str(row.get("domain", "")),
            objective,
            str(row.get("entity_type", "")),
            str(row.get("entity_id", "")),
            str(row.get("metric", "")),
        ]
    )


def _include_scientific_row(row):
    if str(row.get("split")) not in SPLITS:
        return False
    if not is_probe_mechanism_evidence(row):
        return False
    domain = str(row.get("domain"))
    if domain in {
        "target_relevance_reference",
        "target_relevance_forward_equivalence",
    }:
        return True
    if domain == "module_effect":
        entity_type = str(row.get("entity_type"))
        entity_id = str(row.get("entity_id"))
        condition = str(row.get("condition"))
        if entity_type == "intervention" and entity_id == condition:
            return True
        return (
            entity_type == "intervention_control"
            and str(row.get("metric", "")).startswith("targeted_minus_random_")
        )
    return (
        str(row.get("entity_type")) in {"split", "overall"}
        and str(row.get("entity_id")) == "all"
    )


def _load_selected_metrics(path, expected_selection_seed):
    selected = {}
    has_target_relevance = False
    with open_text_artifact(
        Path(path), "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if not _include_scientific_row(row):
                continue
            has_target_relevance = has_target_relevance or (
                str(row.get("domain")) == TARGET_RELEVANCE_DOMAIN
            )
            if int(row.get("selection_seed", -1)) != int(expected_selection_seed):
                raise ValueError(
                    f"unexpected selection_seed in {path}: {row.get('selection_seed')}"
                )
            try:
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            selected[_identity(row)] = value
    if has_target_relevance:
        selected.update(
            _load_predicted_target_relevance_metrics(
                Path(path).parent,
                expected_selection_seed,
            )
        )
    return selected


def _load_predicted_target_relevance_metrics(
    diagnostics_dir, expected_selection_seed
):
    """Load the second target-relevance objective without merging objectives.

    The backward-compatible probe CSV contains the primary true-class objective.
    The predicted-class objective is stored in each target-relevance summary JSON.
    Only aggregate token-type paths are promoted into the cross-Probe matrix; the
    much wider per-Prompt entities remain in the source artifact for traceability.
    """
    diagnostics_dir = Path(diagnostics_dir)
    runtime = _read_json(diagnostics_dir / "probe_runtime_summary.json")
    checkpoint_id = str(
        (runtime.get("checkpoint") or {}).get("checkpoint_id") or ""
    )
    if not checkpoint_id:
        raise ValueError(
            f"missing checkpoint id for target relevance in {diagnostics_dir}"
        )

    selected = {}
    for split in SPLITS:
        summary_path = (
            diagnostics_dir
            / "target_relevance"
            / checkpoint_id
            / f"{split}_summary.json"
        )
        if not summary_path.is_file():
            raise ValueError(
                f"missing target relevance summary for {split}: {summary_path}"
            )
        summary = _read_json(summary_path)
        if int(summary.get("selection_seed", -1)) != int(expected_selection_seed):
            raise ValueError(
                "unexpected target relevance selection seed in "
                f"{summary_path}: {summary.get('selection_seed')}"
            )
        result = dict(
            (summary.get("objective_results") or {}).get(
                "predicted_class_margin"
            )
            or {}
        )
        if result.get("applicability") != "applicable" or not bool(
            result.get("valid", False)
        ):
            raise ValueError(
                "predicted-class target relevance is not valid/applicable in "
                f"{summary_path}"
            )

        for layer, paths in (result.get("by_layer") or {}).items():
            for path_name, metrics in (paths or {}).items():
                if path_name not in TARGET_RELEVANCE_AGGREGATE_PATHS:
                    continue
                for metric, raw_value in (metrics or {}).items():
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value):
                        continue
                    row = {
                        "split": split,
                        "condition": "normal",
                        "domain": TARGET_RELEVANCE_DOMAIN,
                        "objective": "predicted_class_margin",
                        "entity_type": "layer_path",
                        "entity_id": f"layer_{layer}/{path_name}",
                        "metric": metric,
                    }
                    selected[_identity(row)] = value
    return selected


def _summary(values):
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(items),
        "mean": float(statistics.mean(items)),
        "std": float(statistics.stdev(items)) if len(items) > 1 else 0.0,
        "min": float(min(items)),
        "max": float(max(items)),
        "range": float(max(items) - min(items)),
    }


def _scientific_summary(metric_matrix, selection_seeds, training_seeds):
    payload = {}
    for method, identities in sorted(metric_matrix.items()):
        method_payload = {}
        for identity, matrix in sorted(identities.items()):
            complete = all(
                probe_seed in matrix.get(training_seed, {})
                for training_seed in training_seeds
                for probe_seed in selection_seeds
            )
            within_training = {}
            training_seed_means = {}
            for training_seed in training_seeds:
                probe_values = {
                    str(probe_seed): matrix.get(training_seed, {}).get(probe_seed)
                    for probe_seed in selection_seeds
                    if probe_seed in matrix.get(training_seed, {})
                }
                values = list(probe_values.values())
                current = _summary(values)
                signs = {
                    1 if value > 0 else -1 if value < 0 else 0 for value in values
                }
                current.update(
                    {
                        "probe_seed_values": probe_values,
                        "sign_consistent": len(signs) <= 1,
                        "inference_unit": "correlated_probe_selection_on_one_checkpoint",
                    }
                )
                within_training[str(training_seed)] = current
                if values:
                    training_seed_means[str(training_seed)] = float(
                        statistics.mean(values)
                    )
            by_probe = {}
            for probe_seed in selection_seeds:
                training_values = {
                    str(training_seed): matrix.get(training_seed, {}).get(probe_seed)
                    for training_seed in training_seeds
                    if probe_seed in matrix.get(training_seed, {})
                }
                by_probe[str(probe_seed)] = {
                    **_summary(training_values.values()),
                    "training_seed_values": training_values,
                    "inference_unit": "independent_training_seed",
                }
            max_within_range = max(
                (item.get("range", 0.0) or 0.0) for item in within_training.values()
            ) if within_training else None
            method_payload[identity] = {
                "complete_matrix": complete,
                "training_seed_count": len(training_seed_means),
                "probe_selection_seed_count": len(selection_seeds),
                "within_training_seed_probe_variability": within_training,
                "by_probe_selection_seed": by_probe,
                "training_seed_mean_summary": {
                    **_summary(training_seed_means.values()),
                    "training_seed_values": training_seed_means,
                    "inference_unit": "independent_training_seed",
                },
                "max_within_training_seed_probe_range": max_within_range,
            }
        payload[method] = method_payload
    return payload


def _write_scientific_csv(path, scientific):
    rows = []
    for method, identities in sorted(scientific.items()):
        for identity, payload in sorted(identities.items()):
            (
                split,
                condition,
                domain,
                objective,
                entity_type,
                entity_id,
                metric,
            ) = identity.split("|", 6)
            summary = payload["training_seed_mean_summary"]
            row = {
                "method": method,
                "split": split,
                "condition": condition,
                "domain": domain,
                "objective": objective,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": metric,
                "complete_matrix": payload["complete_matrix"],
                "training_seed_count": payload["training_seed_count"],
                "probe_selection_seed_count": payload["probe_selection_seed_count"],
                "training_seed_mean": summary.get("mean"),
                "training_seed_std": summary.get("std"),
                "training_seed_min": summary.get("min"),
                "training_seed_max": summary.get("max"),
                "training_seed_range": summary.get("range"),
                "max_within_training_seed_probe_range": payload.get(
                    "max_within_training_seed_probe_range"
                ),
            }
            training_values = list(
                (summary.get("training_seed_values") or {}).values()
            )
            training_signs = {
                1 if value > 0 else -1 if value < 0 else 0
                for value in training_values
            }
            row["training_seed_sign_consistent"] = len(training_signs) <= 1
            for training_seed, probe_payload in payload[
                "within_training_seed_probe_variability"
            ].items():
                prefix = f"training_seed_{training_seed}_probe"
                row[f"{prefix}_mean"] = probe_payload.get("mean")
                row[f"{prefix}_min"] = probe_payload.get("min")
                row[f"{prefix}_max"] = probe_payload.get("max")
                row[f"{prefix}_range"] = probe_payload.get("range")
                row[f"{prefix}_sign_consistent"] = probe_payload.get(
                    "sign_consistent"
                )
            for probe_seed, probe_payload in payload["by_probe_selection_seed"].items():
                row[f"probe_seed_{probe_seed}_mean"] = probe_payload.get("mean")
                row[f"probe_seed_{probe_seed}_std_across_training"] = probe_payload.get("std")
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    primary_gap_root = (
        Path(args.primary_gap_root).resolve() if args.primary_gap_root else None
    )
    replay_root = Path(args.replay_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    primary_seed = int(args.primary_selection_seed)
    robustness_seeds = _parse_int_list(args.robustness_selection_seeds)
    training_seeds = _parse_int_list(args.training_seeds)
    methods = _parse_name_list(args.methods)
    if primary_seed in robustness_seeds:
        raise ValueError("primary selection seed must not appear in robustness seeds")
    selection_seeds = [primary_seed, *robustness_seeds]
    if len(selection_seeds) != 3:
        raise ValueError(
            "strict fixed-Probe evidence requires exactly three selection seeds"
        )

    technical_rows = []
    metric_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    manifest_hashes = defaultdict(lambda: defaultdict(set))
    checkpoint_consistency = []

    for method in methods:
        for training_seed in training_seeds:
            source_run = _run_dir(source_root, method, training_seed)
            source_record = _manifest_record(source_run)
            gap_run = (
                _run_dir(primary_gap_root, method, training_seed)
                if primary_gap_root is not None
                else None
            )
            gap_valid = True
            if gap_run is not None:
                gap_summary_path = gap_run / "monitor_gap_replay_summary.json"
                gap_valid = gap_summary_path.is_file() and bool(
                    _read_json(gap_summary_path).get("valid", False)
                )
            primary_valid = (
                source_record["runtime_status"] == "completed"
                and all(source_record["valid_by_split"].values())
                and gap_valid
            )
            technical_rows.append(
                {
                    "method": method,
                    "training_seed": training_seed,
                    "selection_seed": primary_seed,
                    "source_kind": (
                        "primary_unified_source_plus_gap_replay"
                        if gap_run is not None
                        else "primary_unified_source"
                    ),
                    "run_dir": str(source_run),
                    "valid": primary_valid,
                    **source_record,
                }
            )
            for split, digest in source_record["manifest_sha256_by_split"].items():
                manifest_hashes[primary_seed][split].add(digest)

            source_metrics = _load_selected_metrics(
                source_run / "diagnostics" / "probe_metrics.csv", primary_seed
            )
            if gap_run is not None:
                gap_metrics = _load_selected_metrics(
                    gap_run / "diagnostics" / "probe_metrics.csv", primary_seed
                )
                source_metrics.update(gap_metrics)
            for identity, value in source_metrics.items():
                metric_matrix[method][identity][training_seed][primary_seed] = value

            source_checkpoint_sha = source_record["checkpoint"].get("checkpoint_sha256")
            for selection_seed in robustness_seeds:
                replay_run = _replay_run_dir(
                    replay_root, selection_seed, method, training_seed
                )
                replay_summary_path = (
                    replay_run / "probe_robustness_replay_summary.json"
                )
                replay_summary = _read_json(replay_summary_path)
                replay_record = _manifest_record(replay_run)
                replay_valid = bool(replay_summary.get("valid", False))
                technical_rows.append(
                    {
                        "method": method,
                        "training_seed": training_seed,
                        "selection_seed": selection_seed,
                        "source_kind": "checkpoint_only_probe_robustness_replay",
                        "run_dir": str(replay_run),
                        "valid": replay_valid,
                        **replay_record,
                    }
                )
                for split, digest in replay_record["manifest_sha256_by_split"].items():
                    manifest_hashes[selection_seed][split].add(digest)
                replay_checkpoint_sha = replay_record["checkpoint"].get(
                    "checkpoint_sha256"
                )
                checkpoint_consistency.append(
                    {
                        "method": method,
                        "training_seed": training_seed,
                        "selection_seed": selection_seed,
                        "source_checkpoint_sha256": source_checkpoint_sha,
                        "replay_checkpoint_sha256": replay_checkpoint_sha,
                        "match": source_checkpoint_sha == replay_checkpoint_sha,
                        "source_probe_batch_size": int(
                            (source_record.get("probe_loader") or {}).get(
                                "batch_size", -1
                            )
                        ),
                        "replay_probe_batch_size": int(
                            (replay_record.get("probe_loader") or {}).get(
                                "batch_size", -2
                            )
                        ),
                        "probe_batch_size_match": int(
                            (source_record.get("probe_loader") or {}).get(
                                "batch_size", -1
                            )
                        )
                        == int(
                            (replay_record.get("probe_loader") or {}).get(
                                "batch_size", -2
                            )
                        ),
                    }
                )
                replay_metrics = _load_selected_metrics(
                    replay_run / "diagnostics" / "probe_metrics.csv", selection_seed
                )
                for identity, value in replay_metrics.items():
                    metric_matrix[method][identity][training_seed][selection_seed] = value

    manifest_checks = {
        str(selection_seed): {
            split: {
                "unique_sha256": sorted(digest for digest in digests if digest),
                "unique_count": len({digest for digest in digests if digest}),
                "pass": len({digest for digest in digests if digest}) == 1,
            }
            for split, digests in sorted(split_payload.items())
        }
        for selection_seed, split_payload in sorted(manifest_hashes.items())
    }
    technical_valid = (
        len(technical_rows) == len(methods) * len(training_seeds) * len(selection_seeds)
        and all(bool(row["valid"]) for row in technical_rows)
        and all(
            str(row.get("execution_profile")) == "final_full"
            and list(row.get("required_splits") or []) == list(SPLITS)
            for row in technical_rows
        )
        and all(
            item["match"] and item["probe_batch_size_match"]
            for item in checkpoint_consistency
        )
        and all(
            item["pass"]
            for split_payload in manifest_checks.values()
            for item in split_payload.values()
        )
    )
    scientific = _scientific_summary(
        metric_matrix, selection_seeds, training_seeds
    )
    complete_metric_count = sum(
        bool(payload["complete_matrix"])
        for method_payload in scientific.values()
        for payload in method_payload.values()
    )
    total_metric_count = sum(len(item) for item in scientific.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_a_series_mode = (
        tuple(methods) == DEFAULT_METHODS and primary_gap_root is not None
    )
    aggregate = {
        "format": (
            "a_series_probe_robustness_aggregate_v2"
            if legacy_a_series_mode
            else "probe_robustness_aggregate_v3"
        ),
        "status": "valid" if technical_valid else "invalid",
        "source_root": str(source_root),
        "primary_gap_root": str(primary_gap_root) if primary_gap_root else None,
        "replay_root": str(replay_root),
        "methods": methods,
        "training_seeds": training_seeds,
        "probe_selection_seeds": selection_seeds,
        "independent_training_seed_count_per_method": len(training_seeds),
        "probe_selection_seed_count_per_checkpoint": len(selection_seeds),
        "strict_three_probe_contract": {
            "required_selection_seed_count": 3,
            "required_execution_profile": "final_full",
            "required_splits": list(SPLITS),
            "pass": technical_valid,
        },
        "independent_sample_size_inflated": False,
        "expected_run_probe_cells": len(methods)
        * len(training_seeds)
        * len(selection_seeds),
        "observed_run_probe_cells": len(technical_rows),
        "technical_rows": technical_rows,
        "checkpoint_consistency": checkpoint_consistency,
        "manifest_identity_checks": manifest_checks,
        "scientific_identity_fields": [
            "split",
            "condition",
            "domain",
            "objective",
            "entity_type",
            "entity_id",
            "metric",
        ],
        "target_relevance_objective_aggregation": {
            "true_class_margin": "probe_metrics_primary_objective",
            "predicted_class_margin": (
                "objective_results_aggregate_token_type_paths"
            ),
            "per_prompt_predicted_objective": "source_artifact_only",
        },
        "scientific_metric_identity_count": total_metric_count,
        "complete_scientific_metric_identity_count": complete_metric_count,
        "technical_valid": technical_valid,
    }
    _write_json(output_dir / "probe_robustness_aggregate.json", aggregate)
    _write_json(output_dir / "probe_robustness_scientific_summary.json", scientific)
    _write_scientific_csv(
        output_dir / "probe_robustness_scientific_summary.csv", scientific
    )
    if not technical_valid:
        raise RuntimeError("probe robustness aggregate failed technical identity gates")


if __name__ == "__main__":
    main()
