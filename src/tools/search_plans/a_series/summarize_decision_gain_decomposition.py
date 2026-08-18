#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate valid per-seed decision gain decomposition artifacts."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", default="0,1,2")
    return parser.parse_args()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _summary(values):
    values = [float(value) for value in values]
    count = len(values)
    mean = sum(values) / count
    std = 0.0 if count < 2 else math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1))
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "positive_count": sum(value > 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "zero_count": sum(value == 0.0 for value in values),
    }


def _flatten(summary):
    values = {}
    endpoint = summary["endpoints"]["target_minus_reference"]
    for metric, value in endpoint.items():
        values["endpoint_delta.{}".format(metric)] = value
    for name, value in summary["seen_unseen_h_decomposition"].items():
        if name != "additivity_error":
            values["seen_unseen_h.{}".format(name)] = value
    for metric, payload in summary["logit_component_shapley"].items():
        for component, value in payload["values"].items():
            values["logit_shapley.{}.{}".format(metric, component)] = value
    for split, payload in summary["endpoint_prediction_transitions"].items():
        for metric, value in payload.items():
            values["transition.{}.{}".format(split, metric)] = value
    return values


def _per_class_cross_seed(per_seed_arrays, candidate_class_ids, tolerance=1.0e-12):
    result = {}
    for split in ("test_seen", "test_unseen"):
        rows = []
        finite_mask = None
        for seed in sorted(per_seed_arrays):
            delta = np.asarray(
                per_seed_arrays[seed][f"{split}_class_accuracy_delta"],
                dtype=np.float64,
            )
            current_mask = np.isfinite(delta)
            if finite_mask is None:
                finite_mask = current_mask
            elif not np.array_equal(finite_mask, current_mask):
                raise RuntimeError(
                    f"per-class support mask differs across seeds for {split}"
                )
            rows.append(delta[current_mask])
        matrix = np.stack(rows, axis=0)
        class_ids = np.asarray(candidate_class_ids, dtype=np.int64)[finite_mask]
        positive = matrix > float(tolerance)
        negative = matrix < -float(tolerance)
        zero = np.abs(matrix) <= float(tolerance)
        positive_all = np.all(positive, axis=0)
        negative_all = np.all(negative, axis=0)
        mixed = np.any(positive, axis=0) & np.any(negative, axis=0)
        zero_all = np.all(zero, axis=0)
        nonnegative = np.all(matrix >= -float(tolerance), axis=0) & np.any(
            positive, axis=0
        )
        nonpositive = np.all(matrix <= float(tolerance), axis=0) & np.any(
            negative, axis=0
        )

        def ids(mask):
            return [int(item) for item in class_ids[np.asarray(mask, dtype=bool)]]

        result[split] = {
            "class_count": int(matrix.shape[1]),
            "strictly_positive_all_seeds_count": int(positive_all.sum()),
            "strictly_positive_all_seeds_class_ids": ids(positive_all),
            "strictly_negative_all_seeds_count": int(negative_all.sum()),
            "strictly_negative_all_seeds_class_ids": ids(negative_all),
            "mixed_positive_negative_count": int(mixed.sum()),
            "mixed_positive_negative_class_ids": ids(mixed),
            "zero_all_seeds_count": int(zero_all.sum()),
            "zero_all_seeds_class_ids": ids(zero_all),
            "nonnegative_all_seeds_positive_once_count": int(nonnegative.sum()),
            "nonnegative_all_seeds_positive_once_class_ids": ids(nonnegative),
            "nonpositive_all_seeds_negative_once_count": int(nonpositive.sum()),
            "nonpositive_all_seeds_negative_once_class_ids": ids(nonpositive),
            "tolerance": float(tolerance),
        }
    return result


def main():
    args = parse_args()
    expected = {int(item.strip()) for item in str(args.expected_seeds).split(",") if item.strip()}
    per_seed = {}
    values = defaultdict(list)
    pair_names = set()
    per_seed_arrays = {}
    canonical_candidate_class_ids = None
    for input_dir in args.input:
        root = Path(input_dir).resolve()
        summary_path = root / "decision_gain_decomposition_summary.json"
        manifest_path = root / "decision_gain_decomposition_manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("missing decision decomposition artifact in {}".format(root))
        summary = _read_json(summary_path)
        manifest = _read_json(manifest_path)
        if not bool(summary.get("validity", {}).get("valid", False)):
            raise RuntimeError("invalid decomposition: {}".format(root))
        seed = int(summary["pair"]["training_seed"])
        if seed in per_seed:
            raise ValueError("duplicate training seed {}".format(seed))
        array_path = root / "decision_gain_decomposition_per_class.npz"
        if not array_path.is_file():
            raise FileNotFoundError(str(array_path))
        with np.load(array_path) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
        candidate_class_ids = np.asarray(
            arrays["candidate_class_ids"], dtype=np.int64
        )
        if canonical_candidate_class_ids is None:
            canonical_candidate_class_ids = candidate_class_ids
        elif not np.array_equal(canonical_candidate_class_ids, candidate_class_ids):
            raise RuntimeError("candidate class ids differ across training seeds")
        per_seed_arrays[seed] = arrays
        pair_names.add((summary["pair"]["reference_name"], summary["pair"]["target_name"]))
        flattened = _flatten(summary)
        for name, value in flattened.items():
            values[name].append(float(value))
        per_seed[str(seed)] = {
            "artifact_dir": str(root),
            "reference_checkpoint_sha256": manifest["reference"]["checkpoint_sha256"],
            "target_checkpoint_sha256": manifest["target"]["checkpoint_sha256"],
            "metrics": flattened,
        }
    observed = {int(seed) for seed in per_seed}
    if observed != expected:
        raise RuntimeError("training seed coverage mismatch: expected={} observed={}".format(sorted(expected), sorted(observed)))
    if len(pair_names) != 1:
        raise RuntimeError("mixed comparison identities: {}".format(sorted(pair_names)))
    aggregate = {name: _summary(metric_values) for name, metric_values in sorted(values.items())}
    reference_name, target_name = next(iter(pair_names))
    per_class_cross_seed = _per_class_cross_seed(
        per_seed_arrays, canonical_candidate_class_ids
    )
    payload = {
        "format": "decision_gain_decomposition_cross_seed_v1",
        "status": "completed",
        "reference_name": reference_name,
        "target_name": target_name,
        "independent_unit": "training_seed",
        "expected_seeds": sorted(expected),
        "observed_seeds": sorted(observed),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "per_class_cross_seed": per_class_cross_seed,
        "interpretation_boundary": "Shapley values explain the declared decision-space factorization and are not training-mechanism causal percentages.",
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision_gain_decomposition_cross_seed.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "decision_gain_decomposition_cross_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "count", "mean", "std", "min", "max", "positive_count", "negative_count", "zero_count"],
        )
        writer.writeheader()
        for metric, item in aggregate.items():
            writer.writerow({"metric": metric, **item})
    print("Decision gain cross-seed summary: seeds={} output={}".format(len(observed), output_dir))


if __name__ == "__main__":
    main()
