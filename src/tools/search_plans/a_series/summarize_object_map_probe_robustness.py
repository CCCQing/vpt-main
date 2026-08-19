#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
SPLITS = ("probe_train_seen", "probe_test_seen", "probe_test_unseen")
LAYERS = (0, 3, 6, 9, 11)
METRICS = {
    "source_to_injected": (
        "distance_correspondence",
        "source_to_injected_distance_spearman",
        "spearman",
    ),
    "source_to_contextualized": (
        "distance_correspondence",
        "source_to_contextualized_distance_spearman",
        "spearman",
    ),
    "source_to_cls_effect": (
        "distance_correspondence",
        "source_to_cls_effect_distance_spearman",
        "spearman",
    ),
    "source_to_logit_effect": (
        "distance_correspondence",
        "source_to_logit_effect_distance_spearman",
        "spearman",
    ),
    "contextualized_to_cls_effect": (
        "distance_correspondence",
        "contextualized_to_cls_effect_distance_spearman",
        "spearman",
    ),
    "cls_effect_to_logit_effect": (
        "distance_correspondence",
        "cls_effect_to_logit_effect_distance_spearman",
        "spearman",
    ),
    "functional_null_direction_ratio": (
        "perturbation_functional_null_direction_ratio",
        "ratio",
    ),
    "prediction_flip_ratio": (
        "prediction_effect",
        "perturbation_prediction_flip_or_disagreement",
        "mean",
    ),
    "delta_true_margin": ("prediction_effect", "delta_true_margin", "mean"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize the A2 Prompt functional object map across "
            "three fixed-Probe selection seeds without treating Probe seeds as "
            "independent training repetitions."
        )
    )
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-seeds", default="0,1,2")
    parser.add_argument("--primary-selection-seed", type=int, default=424242)
    parser.add_argument(
        "--robustness-selection-seeds", default="424243,424244"
    )
    return parser.parse_args()


def _parse_ints(text):
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"expected unique integer list, got {text!r}")
    return values


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _primary_run(root, training_seed):
    return Path(root) / f"A2_seed{training_seed}_final"


def _replay_run(root, selection_seed, training_seed):
    return (
        Path(root)
        / f"selection_seed_{selection_seed}"
        / "A2"
        / f"seed{training_seed}"
        / RUN_SUFFIX
    )


def _nested_float(payload, keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError("/".join(keys))
        current = current[key]
    value = float(current)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value for {'/'.join(keys)}")
    return value


def _run_record(run_dir, expected_selection_seed, training_seed):
    run_dir = Path(run_dir)
    replay_summary = _read_json(
        run_dir / "bayesian_object_map_replay_summary.json"
    )
    runtime = _read_json(run_dir / "diagnostics" / "probe_runtime_summary.json")
    manifest = _read_json(run_dir / "diagnostics" / "probe_manifest.json")
    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    selection_seed = int(runtime.get("selection_seed", -1))
    required_splits = list(runtime.get("required_splits") or SPLITS)
    checkpoint = dict(runtime.get("checkpoint") or {})
    failures = []
    if replay_summary.get("valid") is not True:
        failures.append("replay_summary_invalid")
    if runtime.get("status") != "completed":
        failures.append("runtime_not_completed")
    if selection_seed != int(expected_selection_seed):
        failures.append("selection_seed_mismatch")
    if required_splits != list(SPLITS):
        failures.append("required_splits_mismatch")
    if replay_summary.get("training_performed") is not False:
        failures.append("training_performed_not_false")
    if replay_summary.get("optimizer_step_performed") is not False:
        failures.append("optimizer_step_performed_not_false")

    manifest_sha_by_split = {}
    sample_count_by_split = {}
    hierarchy_by_split = {}
    checkpoint_id = str(
        checkpoint.get(
            "checkpoint_id",
            f"final_epoch_{int(resolved['SOLVER']['TOTAL_EPOCH']):04d}",
        )
    )
    hierarchy_root = (
        run_dir
        / "diagnostics"
        / "bayesian_object_selection"
        / checkpoint_id
    )
    for split in SPLITS:
        probe = manifest.get("probes", {}).get(split, {})
        validity = manifest.get("validity", {}).get(split, {})
        if int(probe.get("selection_seed", -1)) != int(expected_selection_seed):
            failures.append(f"{split}:selection_seed_mismatch")
        if validity.get("valid") is not True:
            failures.append(f"{split}:manifest_invalid")
        manifest_sha_by_split[split] = probe.get("manifest_sha256")
        sample_count_by_split[split] = int(probe.get("selected_sample_count", 0))
        hierarchy_path = hierarchy_root / f"{split}_hierarchy_trace.json"
        if not hierarchy_path.is_file():
            failures.append(f"{split}:hierarchy_missing")
            continue
        hierarchy = _read_json(hierarchy_path)
        if hierarchy.get("valid") is not True:
            failures.append(f"{split}:hierarchy_invalid")
        layer_map = hierarchy.get("layer_interface_map", {})
        for layer in LAYERS:
            if layer_map.get(f"layer_{layer}", {}).get("valid") is not True:
                failures.append(f"{split}:layer_{layer}_invalid")
        hierarchy_by_split[split] = hierarchy

    probe_loader = dict(runtime.get("probe_loader") or {})
    probe_batch_size = int(
        probe_loader.get(
            "batch_size", resolved["MONITOR"]["PROBE"]["BATCH_SIZE"]
        )
    )
    return {
        "training_seed": int(training_seed),
        "selection_seed": selection_seed,
        "run_dir": str(run_dir),
        "valid": not failures,
        "failure_reasons": failures,
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "checkpoint_global_step": checkpoint.get("checkpoint_global_step"),
        "source_run_id": checkpoint.get("source_run_id"),
        "source_session_id": checkpoint.get("source_session_id"),
        "probe_batch_size": probe_batch_size,
        "manifest_sha256_by_split": manifest_sha_by_split,
        "sample_count_by_split": sample_count_by_split,
        "hierarchy_by_split": hierarchy_by_split,
    }


def _sign_pattern(values, tolerance=1e-12):
    signs = set()
    for value in values:
        if value > tolerance:
            signs.add("positive")
        elif value < -tolerance:
            signs.add("negative")
        else:
            signs.add("zero")
    if len(signs) == 1:
        return next(iter(signs))
    return "mixed"


def main():
    args = parse_args()
    training_seeds = _parse_ints(args.training_seeds)
    selection_seeds = [int(args.primary_selection_seed)] + _parse_ints(
        args.robustness_selection_seeds
    )
    if len(selection_seeds) != 3 or len(set(selection_seeds)) != 3:
        raise ValueError("formal object-map summary requires exactly three Probe seeds")

    records = []
    for training_seed in training_seeds:
        for selection_seed in selection_seeds:
            run_dir = (
                _primary_run(args.primary_root, training_seed)
                if selection_seed == int(args.primary_selection_seed)
                else _replay_run(args.replay_root, selection_seed, training_seed)
            )
            records.append(
                _run_record(run_dir, selection_seed, training_seed)
            )

    failures = [
        {
            "training_seed": item["training_seed"],
            "selection_seed": item["selection_seed"],
            "failure_reasons": item["failure_reasons"],
        }
        for item in records
        if not item["valid"]
    ]
    identity_failures = []
    for training_seed in training_seeds:
        group = [item for item in records if item["training_seed"] == training_seed]
        checkpoint_hashes = {item["checkpoint_sha256"] for item in group}
        batches = {item["probe_batch_size"] for item in group}
        if len(checkpoint_hashes) != 1 or None in checkpoint_hashes:
            identity_failures.append(
                f"training_seed_{training_seed}:checkpoint_sha256_mismatch"
            )
        if len(batches) != 1:
            identity_failures.append(
                f"training_seed_{training_seed}:probe_batch_size_mismatch"
            )
    for selection_seed in selection_seeds:
        group = [item for item in records if item["selection_seed"] == selection_seed]
        for split in SPLITS:
            hashes = {
                item["manifest_sha256_by_split"].get(split) for item in group
            }
            if len(hashes) != 1 or None in hashes:
                identity_failures.append(
                    f"selection_seed_{selection_seed}:{split}:manifest_mismatch"
                )

    detailed_rows = []
    for record in records:
        for split, hierarchy in record["hierarchy_by_split"].items():
            stages = [("all_layers", hierarchy)] + [
                (f"layer_{layer}", hierarchy["layer_interface_map"][f"layer_{layer}"])
                for layer in LAYERS
            ]
            for layer_name, payload in stages:
                for metric, keys in METRICS.items():
                    detailed_rows.append(
                        {
                            "training_seed": record["training_seed"],
                            "selection_seed": record["selection_seed"],
                            "split": split,
                            "layer": layer_name,
                            "metric": metric,
                            "value": _nested_float(payload, keys),
                        }
                    )

    grouped = defaultdict(list)
    for row in detailed_rows:
        grouped[(row["split"], row["layer"], row["metric"])].append(row)
    summary_rows = []
    for (split, layer, metric), rows in sorted(grouped.items()):
        by_training = {}
        for training_seed in training_seeds:
            values = [
                row["value"]
                for row in rows
                if row["training_seed"] == training_seed
            ]
            if len(values) != 3:
                raise RuntimeError(
                    f"incomplete Probe triple for {split}/{layer}/{metric}/seed{training_seed}"
                )
            by_training[training_seed] = {
                "probe_mean": sum(values) / len(values),
                "probe_min": min(values),
                "probe_max": max(values),
                "probe_range": max(values) - min(values),
                "probe_sign_pattern": _sign_pattern(values),
            }
        training_means = [
            by_training[training_seed]["probe_mean"]
            for training_seed in training_seeds
        ]
        payload = {
            "split": split,
            "layer": layer,
            "metric": metric,
            "training_seed_mean": sum(training_means) / len(training_means),
            "training_seed_min": min(training_means),
            "training_seed_max": max(training_means),
            "training_seed_sign_pattern": _sign_pattern(training_means),
            "max_probe_range_within_training_seed": max(
                item["probe_range"] for item in by_training.values()
            ),
        }
        for training_seed in training_seeds:
            for key, value in by_training[training_seed].items():
                payload[f"training_seed_{training_seed}_{key}"] = value
        for selection_seed in selection_seeds:
            values = [
                row["value"]
                for row in rows
                if row["selection_seed"] == selection_seed
            ]
            payload[f"probe_seed_{selection_seed}_mean"] = sum(values) / len(values)
        summary_rows.append(payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "object_map_three_probe_detailed.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed_rows[0]))
        writer.writeheader()
        writer.writerows(detailed_rows)
    summary_path = output_dir / "object_map_three_probe_scientific_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    valid = not failures and not identity_failures
    manifest = {
        "format": "a_series_object_map_three_probe_summary_v1",
        "status": "valid" if valid else "invalid",
        "valid": valid,
        "training_seeds": training_seeds,
        "selection_seeds": selection_seeds,
        "run_probe_cell_count_expected": len(training_seeds) * 3,
        "run_probe_cell_count_observed": len(records),
        "independent_statistical_unit": "training_seed",
        "probe_seed_role": "correlated_selection_repeat_within_checkpoint",
        "aggregation_order": "Probe seeds within training checkpoint, then training seeds",
        "record_failures": failures,
        "identity_failures": identity_failures,
        "records": [
            {key: value for key, value in item.items() if key != "hierarchy_by_split"}
            for item in records
        ],
        "artifacts": {
            "detailed_csv": detailed_path.name,
            "scientific_summary_csv": summary_path.name,
        },
    }
    _write_json(output_dir / "object_map_three_probe_summary.json", manifest)
    if not valid:
        raise RuntimeError(
            "object-map three-Probe summary failed validity or identity checks"
        )
    print(output_dir / "object_map_three_probe_summary.json")


if __name__ == "__main__":
    main()
