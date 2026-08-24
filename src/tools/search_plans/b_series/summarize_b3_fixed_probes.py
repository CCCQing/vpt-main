#!/usr/bin/env python3
"""Aggregate B3 deferred strict-three-Probe collections.

All three selections are correlated views of one checkpoint.  The scientific
summary first averages selections within each checkpoint and only then compares
the three independent training seeds.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from src.tools.search_plans.a_series.summarize_probe_robustness_replays import (
    _load_selected_metrics,
    _scientific_summary,
    _write_scientific_csv,
)


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
STRICT_TRAINING_SEEDS = (0, 1, 2)
STRICT_PROBE_SEEDS = (424242, 424243, 424244)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _comma_names(raw: str):
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("method names must be non-empty and unique")
    if any("/" in value or "\\" in value or value in {".", ".."} for value in values):
        raise ValueError("method names must be safe path components")
    return values


def _run_dir(root: Path, method: str, seed: int) -> Path:
    return root / method / "seed{}".format(seed) / RUN_SUFFIX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--methods",
        default="B3-R1I-R025,B3-R1I-R050,B3-R1N-R025,B3-R1N-R050",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    methods = _comma_names(args.methods)
    metric_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    technical_cells = []
    manifest_hashes = defaultdict(lambda: defaultdict(set))

    for method in methods:
        for training_seed in STRICT_TRAINING_SEEDS:
            run_dir = _run_dir(source_root, method, training_seed)
            collection_path = run_dir / "b3_fixed_probe_collection.json"
            if not collection_path.is_file():
                raise FileNotFoundError(str(collection_path))
            collection = _read_json(collection_path)
            executions = tuple(collection.get("probe_executions") or ())
            collection_valid = (
                collection.get("format") == "b3_deferred_fixed_probe_collection_v1"
                and collection.get("valid") is True
                and collection.get("status") == "valid"
                and tuple(int(value) for value in collection.get("selection_seeds") or ())
                == STRICT_PROBE_SEEDS
                and len(executions) == len(STRICT_PROBE_SEEDS)
            )
            if not collection_valid:
                raise RuntimeError("invalid fixed-Probe collection {}".format(collection_path))
            expected_checkpoint = str(collection.get("training_checkpoint_sha256") or "")
            observed_seeds = []
            for execution in executions:
                selection_seed = int(execution.get("selection_seed", -1))
                observed_seeds.append(selection_seed)
                replay_run = (run_dir / str(execution.get("replay_root", ""))).resolve()
                replay_summary_path = replay_run / "probe_robustness_replay_summary.json"
                replay_summary = _read_json(replay_summary_path)
                manifest_checks = dict(execution.get("probe_manifest_checks") or {})
                cell_valid = (
                    execution.get("valid") is True
                    and execution.get("execution_profile") == "final_full"
                    and str(execution.get("checkpoint_sha256") or "") == expected_checkpoint
                    and replay_summary.get("valid") is True
                    and replay_summary.get("execution_profile") == "final_full"
                    and all(bool(item.get("pass", False)) for item in manifest_checks.values())
                )
                technical_cells.append(
                    {
                        "method": method,
                        "training_seed": training_seed,
                        "selection_seed": selection_seed,
                        "run_dir": str(replay_run),
                        "checkpoint_sha256": expected_checkpoint,
                        "valid": cell_valid,
                    }
                )
                if not cell_valid:
                    raise RuntimeError("invalid deferred Probe replay {}".format(replay_run))
                for split, item in manifest_checks.items():
                    manifest_hashes[selection_seed][split].add(item.get("manifest_sha256"))
                metrics_path = replay_run / "diagnostics" / "probe_metrics.csv"
                selected = _load_selected_metrics(metrics_path, selection_seed)
                if not selected:
                    raise RuntimeError("no scientific Probe metrics in {}".format(metrics_path))
                for identity, value in selected.items():
                    metric_matrix[method][identity][training_seed][selection_seed] = value
            if tuple(observed_seeds) != STRICT_PROBE_SEEDS:
                raise RuntimeError("fixed-Probe execution order or identity mismatch")

    scientific = _scientific_summary(
        metric_matrix, STRICT_PROBE_SEEDS, STRICT_TRAINING_SEEDS
    )
    incomplete = [
        "{}|{}".format(method, identity)
        for method, identities in scientific.items()
        for identity, payload in identities.items()
        if not bool(payload.get("complete_matrix", False))
    ]
    expected_cells = len(methods) * len(STRICT_TRAINING_SEEDS) * len(STRICT_PROBE_SEEDS)
    technical_valid = (
        len(technical_cells) == expected_cells
        and all(bool(item["valid"]) for item in technical_cells)
        and not incomplete
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "b3_deferred_fixed_probe_aggregate_v1",
        "status": "valid" if technical_valid else "invalid",
        "methods": list(methods),
        "training_seeds": list(STRICT_TRAINING_SEEDS),
        "probe_selection_seeds": list(STRICT_PROBE_SEEDS),
        "independent_unit": "training_seed",
        "probe_seed_role": "nested_correlated_selection_within_checkpoint",
        "independent_sample_size_inflated": False,
        "expected_probe_cell_count": expected_cells,
        "observed_probe_cell_count": len(technical_cells),
        "technical_cells": technical_cells,
        "manifest_identity": {
            str(seed): {
                split: sorted(value for value in values if value)
                for split, values in splits.items()
            }
            for seed, splits in manifest_hashes.items()
        },
        "scientific_metric_identity_count": sum(
            len(identities) for identities in scientific.values()
        ),
        "incomplete_metric_identities": incomplete,
        "valid": technical_valid,
    }
    _atomic_json(output_dir / "b3_fixed_probe_aggregate.json", payload)
    _atomic_json(output_dir / "b3_fixed_probe_scientific_summary.json", scientific)
    _write_scientific_csv(
        output_dir / "b3_fixed_probe_scientific_summary.csv", scientific
    )
    print(
        "B3 fixed-Probe aggregate status={} cells={}/{} metrics={}".format(
            payload["status"],
            len(technical_cells),
            expected_cells,
            payload["scientific_metric_identity_count"],
        )
    )
    if not technical_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
