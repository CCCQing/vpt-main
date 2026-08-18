"""Cross-seed and paired summaries for method-level logit geometry replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize valid logit geometry replays across methods and training seeds."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="METHOD=LOGIT_GEOMETRY_OUTPUT_DIR; repeat once per method and seed",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", default="")
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_scalar(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _flatten_metrics(payload: Mapping[str, object]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for split in ("test_seen", "test_unseen"):
        split_payload = payload["splits"][split]
        for name, value in split_payload.get("factor_context", {}).items():
            if _finite_scalar(value):
                result["{}.factor_context.{}".format(split, name)] = float(value)
        for view, metrics in split_payload.get("views", {}).items():
            for name, value in metrics.items():
                if _finite_scalar(value):
                    result["{}.{}.{}".format(split, view, name)] = float(value)
    for view, metrics in payload.get("joint", {}).get("views", {}).items():
        for name, value in metrics.items():
            if _finite_scalar(value):
                result["joint.{}.{}".format(view, name)] = float(value)
    return result


def _summary(values: Iterable[float]) -> Dict[str, object]:
    items = [float(value) for value in values]
    return {
        "n": len(items),
        "mean": float(statistics.mean(items)),
        "sample_std": float(statistics.stdev(items)) if len(items) >= 2 else None,
        "min": float(min(items)),
        "max": float(max(items)),
        "positive_count": int(sum(value > 0.0 for value in items)),
        "negative_count": int(sum(value < 0.0 for value in items)),
        "zero_count": int(sum(value == 0.0 for value in items)),
    }


def _parse_runs(raw_runs: Sequence[str], expected_seeds: Sequence[int]):
    rows = []
    for item in raw_runs:
        if "=" not in item:
            raise ValueError("--run must use METHOD=OUTPUT_DIR")
        method, raw_path = item.split("=", 1)
        method = method.strip()
        run_dir = Path(raw_path).expanduser().resolve()
        manifest_path = run_dir / "logit_geometry_reference_manifest.json"
        summary_path = run_dir / "logit_geometry_reference_summary.json"
        validity_path = run_dir / "logit_geometry_reference_validity.json"
        if not manifest_path.is_file() or not summary_path.is_file() or not validity_path.is_file():
            raise FileNotFoundError("incomplete logit geometry output: {}".format(run_dir))
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
        validity = _read_json(validity_path)
        if manifest.get("status") != "completed" or not validity.get("valid"):
            raise ValueError("invalid logit geometry output: {}".format(run_dir))
        seed = int(manifest["source"]["seed"])
        if str(manifest.get("method_name")) != method:
            raise ValueError("method name does not match manifest: {}".format(run_dir))
        rows.append(
            {
                "method": method,
                "seed": seed,
                "run_dir": str(run_dir),
                "checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
                "dataset": manifest["source"]["dataset"],
                "candidate_class_ids": manifest["candidate_class_ids"],
                "sample_order_sha256_by_split": manifest["sample_order_sha256_by_split"],
                "metrics": _flatten_metrics(summary),
            }
        )
    identities = {(row["method"], row["seed"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate method and seed in --run inputs")
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        seeds = sorted(row["seed"] for row in rows if row["method"] == method)
        if expected_seeds and seeds != sorted(expected_seeds):
            raise ValueError(
                "{} seeds {} do not match expected {}".format(method, seeds, sorted(expected_seeds))
            )
    return rows


def _method_summaries(rows):
    result = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        metric_names = sorted(set.intersection(*(set(row["metrics"]) for row in method_rows)))
        result[method] = {
            "training_seeds": sorted(row["seed"] for row in method_rows),
            "metrics": {
                name: {
                    **_summary(row["metrics"][name] for row in method_rows),
                    "seed_values": {
                        str(row["seed"]): float(row["metrics"][name]) for row in method_rows
                    },
                }
                for name in metric_names
            },
        }
    return result


def _paired_summaries(rows):
    by_identity = {(row["method"], row["seed"]): row for row in rows}
    methods = sorted({row["method"] for row in rows})
    result = {}
    for reference_index, reference_name in enumerate(methods):
        for target_name in methods[reference_index + 1 :]:
            reference_seeds = {row["seed"] for row in rows if row["method"] == reference_name}
            target_seeds = {row["seed"] for row in rows if row["method"] == target_name}
            common_seeds = sorted(reference_seeds.intersection(target_seeds))
            if not common_seeds:
                continue
            deltas = defaultdict(dict)
            for seed in common_seeds:
                reference = by_identity[(reference_name, seed)]
                target = by_identity[(target_name, seed)]
                if (
                    reference["dataset"] != target["dataset"]
                    or reference["candidate_class_ids"] != target["candidate_class_ids"]
                    or reference["sample_order_sha256_by_split"]
                    != target["sample_order_sha256_by_split"]
                ):
                    raise ValueError(
                        "paired logit geometry identity mismatch: {} -> {} seed {}".format(
                            reference_name, target_name, seed
                        )
                    )
                metric_names = set(reference["metrics"]).intersection(target["metrics"])
                for name in metric_names:
                    deltas[name][str(seed)] = float(
                        target["metrics"][name] - reference["metrics"][name]
                    )
            result["{}-{}".format(target_name, reference_name)] = {
                "direction": "target_minus_reference",
                "reference": reference_name,
                "target": target_name,
                "training_seeds": common_seeds,
                "metrics": {
                    name: {**_summary(seed_values.values()), "seed_values": seed_values}
                    for name, seed_values in sorted(deltas.items())
                },
            }
    return result


def _write_csv(path: Path, method_summaries, paired_summaries) -> None:
    rows = []
    for method, payload in method_summaries.items():
        for metric, stats in payload["metrics"].items():
            rows.append({"record_type": "method", "identity": method, "metric": metric, **stats})
    for pair, payload in paired_summaries.items():
        for metric, stats in payload["metrics"].items():
            rows.append({"record_type": "paired_delta", "identity": pair, "metric": metric, **stats})
    fields = [
        "record_type",
        "identity",
        "metric",
        "n",
        "mean",
        "sample_std",
        "min",
        "max",
        "positive_count",
        "negative_count",
        "zero_count",
        "seed_values",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["seed_values"] = json.dumps(row["seed_values"], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    expected_seeds = [
        int(item.strip()) for item in str(args.expected_seeds).split(",") if item.strip()
    ]
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("--expected-seeds must be unique")
    rows = _parse_runs(args.run, expected_seeds)
    method_summaries = _method_summaries(rows)
    paired_summaries = _paired_summaries(rows)
    payload = {
        "format": "logit_geometry_cross_seed_summary_v1",
        "independent_unit": "training_seed",
        "paired_delta_direction": "target_minus_reference",
        "runs": [
            {key: value for key, value in row.items() if key != "metrics"} for row in rows
        ],
        "methods": method_summaries,
        "paired_deltas": paired_summaries,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logit_geometry_cross_seed.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "logit_geometry_cross_seed.csv", method_summaries, paired_summaries)
    print("wrote logit geometry summary for {} runs to {}".format(len(rows), output_dir))


if __name__ == "__main__":
    main()
