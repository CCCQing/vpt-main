#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
METHODS = ("A0", "A1", "A2")
SEEDS = (0, 1, 2)
SPLITS = ("probe_train_seen", "probe_test_seen", "probe_test_unseen")
TARGET_PATHS = (
    "cls_to_patch",
    "cls_to_prompt",
    "prompt_to_cls",
    "prompt_to_patch",
    "patch_to_prompt",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate validity gates from A-series monitoring-gap replays."
    )
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _flatten_interventions(payload):
    rows = []
    for condition, splits in payload.get("selection_intervention_checks", {}).items():
        for split, item in splits.items():
            mass = item.get("mass_preservation", {})
            effect_summary = {}
            effect_path = Path(str(item.get("path", "")))
            if effect_path.is_file():
                effect_summary = _read_json(effect_path).get("summary", {})
            rows.append({
                "condition": condition,
                "split": split,
                "pass": bool(item.get("pass", False)),
                "valid": bool(item.get("valid", False)),
                "local_mass_abs_error": mass.get("local_mass_abs_error"),
                "applied": mass.get("applied"),
                "applied_pass": bool(mass.get("applied_pass", False)),
                "delta_true_margin": effect_summary.get("delta_true_margin"),
                "prediction_flip_rate": effect_summary.get(
                    "prediction_flip_rate"
                ),
                "beneficial_flip_rate": effect_summary.get(
                    "beneficial_flip_rate"
                ),
                "harmful_flip_rate": effect_summary.get("harmful_flip_rate"),
                "feature_cosine_before_after": effect_summary.get(
                    "feature_cosine_before_after"
                ),
            })
    return rows


def _target_relevance_details(run_dir):
    root = run_dir / "diagnostics" / "target_relevance" / "final_epoch_0015"
    objective_count = 0
    valid_objective_count = 0
    max_logit_abs_diff = 0.0
    max_margin_abs_diff = 0.0
    max_prediction_flip_rate = 0.0
    path_rows = []
    for split in SPLITS:
        payload = _read_json(root / f"{split}_summary.json")
        objectives = payload.get("objective_results", {})
        objective_count += len(objectives)
        valid_objective_count += sum(
            bool(item.get("valid", False)) for item in objectives.values()
        )
        equivalence = payload.get("equivalence", {})
        max_logit_abs_diff = max(
            max_logit_abs_diff,
            abs(float(equivalence.get("logit_max_abs_diff", float("inf")))),
        )
        max_margin_abs_diff = max(
            max_margin_abs_diff,
            abs(float(equivalence.get("true_margin_abs_diff", float("inf")))),
        )
        max_prediction_flip_rate = max(
            max_prediction_flip_rate,
            abs(float(equivalence.get("prediction_flip_rate", float("inf")))),
        )
        for objective_name, objective in objectives.items():
            for layer, paths in objective.get("by_layer", {}).items():
                for path_name in TARGET_PATHS:
                    metrics = paths.get(path_name)
                    if not isinstance(metrics, dict):
                        continue
                    path_rows.append({
                        "split": split,
                        "objective": objective_name,
                        "layer": int(layer),
                        "path": path_name,
                        "positive_sum": metrics.get("all.positive_sum"),
                        "negative_abs_sum": metrics.get(
                            "all.negative_abs_sum"
                        ),
                        "net_sum": metrics.get("all.net_sum"),
                        "absolute_sum": metrics.get("all.absolute_sum"),
                        "positive_edge_fraction": metrics.get(
                            "all.positive_edge_fraction"
                        ),
                        "layer_absolute_share": metrics.get(
                            "all.layer_absolute_share"
                        ),
                    })
    return {
        "target_objective_count": objective_count,
        "target_valid_objective_count": valid_objective_count,
        "target_max_logit_abs_diff": max_logit_abs_diff,
        "target_max_margin_abs_diff": max_margin_abs_diff,
        "target_max_prediction_flip_rate": max_prediction_flip_rate,
        "path_rows": path_rows,
    }


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    rows = []
    missing = []
    intervention_rows = []
    target_path_rows = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = root / method / f"seed{seed}" / RUN_SUFFIX
            path = run_dir / "monitor_gap_replay_summary.json"
            if not path.is_file():
                missing.append(f"{method}:seed{seed}")
                rows.append({
                    "method": method,
                    "seed": seed,
                    "status": "missing",
                    "valid": False,
                    "checkpoint_identity_pass": False,
                    "probe_manifest_pass": False,
                    "target_relevance_pass": False,
                    "selection_intervention_pass": False,
                    "target_split_count": 0,
                    "target_objective_count": 0,
                    "target_valid_objective_count": 0,
                    "target_max_logit_abs_diff": None,
                    "target_max_margin_abs_diff": None,
                    "target_max_prediction_flip_rate": None,
                    "selection_condition_split_count": 0,
                    "max_local_mass_abs_error": None,
                    "run_dir": str(run_dir),
                })
                continue
            payload = _read_json(path)
            checkpoint_pass = all(payload.get("checkpoint_checks", {}).values())
            manifest_pass = all(
                bool(item.get("match", False))
                for item in payload.get("probe_manifest_checks", {}).values()
            )
            target_checks = payload.get("target_relevance_checks", {})
            target_pass = len(target_checks) == 3 and all(
                bool(item.get("pass", False)) for item in target_checks.values()
            )
            target_details = _target_relevance_details(run_dir)
            for item in target_details.pop("path_rows"):
                target_path_rows.append({
                    "method": method,
                    "seed": seed,
                    **item,
                })
            target_pass = bool(
                target_pass
                and target_details["target_objective_count"] == 6
                and target_details["target_valid_objective_count"] == 6
            )
            current_interventions = _flatten_interventions(payload)
            intervention_pass = (
                all(item["pass"] for item in current_interventions)
                if current_interventions else method == "A0"
            )
            mass_errors = [
                abs(float(item["local_mass_abs_error"]))
                for item in current_interventions
                if item["local_mass_abs_error"] is not None
            ]
            for item in current_interventions:
                intervention_rows.append({
                    "method": method,
                    "seed": seed,
                    **item,
                })
            rows.append({
                "method": method,
                "seed": seed,
                "status": payload.get("status"),
                "valid": bool(payload.get("valid", False)),
                "checkpoint_identity_pass": checkpoint_pass,
                "probe_manifest_pass": manifest_pass,
                "target_relevance_pass": target_pass,
                "selection_intervention_pass": intervention_pass,
                "target_split_count": len(target_checks),
                **target_details,
                "selection_condition_split_count": len(current_interventions),
                "max_local_mass_abs_error": max(mass_errors) if mass_errors else None,
                "run_dir": str(run_dir),
            })

    complete = not missing
    valid = complete and all(
        row["valid"]
        and row["checkpoint_identity_pass"]
        and row["probe_manifest_pass"]
        and row["target_relevance_pass"]
        and row["selection_intervention_pass"]
        for row in rows
    )
    aggregate = {
        "format": "a_series_monitor_gap_replay_aggregate_v1",
        "root": str(root),
        "expected_run_count": 9,
        "observed_run_count": 9 - len(missing),
        "missing_runs": missing,
        "complete": complete,
        "valid": valid,
        "target_split_count": sum(row["target_split_count"] for row in rows),
        "target_objective_count": sum(
            row["target_objective_count"] for row in rows
        ),
        "target_valid_objective_count": sum(
            row["target_valid_objective_count"] for row in rows
        ),
        "runs": rows,
        "target_relevance_paths": target_path_rows,
        "selection_interventions": intervention_rows,
    }
    _write_json(root / "monitor_gap_replay_aggregate.json", aggregate)
    with (root / "monitor_gap_replay_aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"complete={complete} valid={valid} observed={aggregate['observed_run_count']}/9"
    )
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
