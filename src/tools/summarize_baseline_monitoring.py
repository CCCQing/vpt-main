#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="METHOD=OUTPUT_DIR; repeat for every seed/run")
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _classification_by_epoch(path: Path) -> Dict[int, Dict[str, float]]:
    epochs: Dict[int, Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("namespace") != "classification" or row.get("split") != "test_gzsl":
                continue
            epoch = int(float(row.get("epoch") or 0))
            value = _float(row.get("value"))
            if value is not None:
                epochs[epoch][str(row.get("metric"))] = value
    return dict(epochs)


def _latest_calibration(run_dir: Path) -> Dict[str, float]:
    paths = sorted((run_dir / "diagnostics" / "calibration_profile").glob("epoch_*.json"))
    if not paths:
        return {}
    summary = _read_json(paths[-1]).get("summary", {})
    return {str(name): float(value) for name, value in summary.items() if _float(value) is not None}


def _latest_probe_mechanisms(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "diagnostics" / "probe_metrics.csv"
    if not path.exists():
        return {}
    values: Dict[str, List[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = _float(row.get("value"))
            if value is None:
                continue
            key = "{}.{}.{}".format(row.get("domain"), row.get("entity_id"), row.get("metric"))
            values[key].append(value)
    return {name: float(statistics.mean(items)) for name, items in values.items() if items}


def load_run(method: str, run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "monitor_manifest.json"
    runtime_path = run_dir / "monitor_runtime_summary.json"
    epoch_path = run_dir / "metrics_epoch.csv"
    row: Dict[str, Any] = {
        "method": method,
        "run_dir": str(run_dir),
        "seed": None,
        "status": "missing",
        "failed": True,
    }
    if not manifest_path.exists() or not runtime_path.exists() or not epoch_path.exists():
        return row
    manifest = _read_json(manifest_path)
    runtime = _read_json(runtime_path)
    row["seed"] = manifest.get("seed")
    row["run_id"] = manifest.get("run_id")
    row["session_id"] = manifest.get("session", {}).get("id")
    row["status"] = str(runtime.get("status", "unknown"))
    row["failed"] = row["status"] != "completed"
    epochs = _classification_by_epoch(epoch_path)
    complete = {
        epoch: values
        for epoch, values in epochs.items()
        if {"gzsl_seen", "gzsl_unseen", "gzsl_h"}.issubset(values)
    }
    if not complete:
        row["failed"] = True
        return row
    final_epoch = max(complete)
    final = complete[final_epoch]
    row.update({
        "final_epoch": int(final_epoch),
        "gzsl_seen": float(final["gzsl_seen"]),
        "gzsl_unseen": float(final["gzsl_unseen"]),
        "gzsl_h": float(final["gzsl_h"]),
    })
    peak_epoch, peak_values = max(complete.items(), key=lambda item: item[1]["gzsl_h"])
    row["diagnostic_peak_epoch"] = int(peak_epoch)
    row["diagnostic_peak_h"] = float(peak_values["gzsl_h"])
    calibration = _latest_calibration(run_dir)
    for name in ("ausuc", "oracle_peak_gamma", "oracle_peak_h", "raw_to_oracle_gain"):
        if name in calibration:
            row[name] = calibration[name]
    row["mechanisms"] = _latest_probe_mechanisms(run_dir)
    return row


def _summary(values: Sequence[float]) -> Dict[str, float]:
    items = [float(value) for value in values if math.isfinite(float(value))]
    if not items:
        return {"count": 0, "mean": None, "std": None, "ci95": None}
    std = statistics.stdev(items) if len(items) > 1 else 0.0
    return {
        "count": len(items),
        "mean": statistics.mean(items),
        "std": std,
        "ci95": 1.96 * std / math.sqrt(len(items)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row if key != "mechanisms"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fields})


def main():
    args = parse_args()
    runs = []
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must use METHOD=OUTPUT_DIR")
        method, raw_path = item.split("=", 1)
        runs.append(load_run(method.strip(), Path(raw_path).expanduser().resolve()))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "cross_seed_runs.csv", runs)

    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_method[str(row["method"])].append(row)
    method_rows = []
    method_payload = {}
    metric_names = ("gzsl_seen", "gzsl_unseen", "gzsl_h", "ausuc", "diagnostic_peak_epoch", "oracle_peak_gamma")
    for method, rows in sorted(by_method.items()):
        summary = {
            name: _summary([row[name] for row in rows if _float(row.get(name)) is not None])
            for name in metric_names
        }
        summary["failed_run_count"] = int(sum(bool(row.get("failed")) for row in rows))
        method_payload[method] = summary
        flat = {"method": method, "failed_run_count": summary["failed_run_count"]}
        for metric, values in summary.items():
            if isinstance(values, Mapping):
                for field, value in values.items():
                    flat[f"{metric}_{field}"] = value
        method_rows.append(flat)
    _write_csv(output_dir / "method_summary.csv", method_rows)

    baseline_rows = {
        int(row["seed"]): row
        for row in by_method.get(args.baseline_method, [])
        if row.get("seed") is not None and not row.get("failed")
    }
    paired_rows = []
    paired_payload = {}
    for method, rows in sorted(by_method.items()):
        if method == args.baseline_method:
            continue
        deltas: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            seed = row.get("seed")
            if seed is None or int(seed) not in baseline_rows or row.get("failed"):
                continue
            baseline = baseline_rows[int(seed)]
            pair = {"method": method, "baseline_method": args.baseline_method, "seed": int(seed)}
            for metric in ("gzsl_seen", "gzsl_unseen", "gzsl_h", "ausuc"):
                if _float(row.get(metric)) is None or _float(baseline.get(metric)) is None:
                    continue
                delta = float(row[metric]) - float(baseline[metric])
                pair[f"delta_{metric}"] = delta
                deltas[metric].append(delta)
            paired_rows.append(pair)
        paired_payload[method] = {}
        for metric, values in deltas.items():
            stats = _summary(values)
            positive = float(sum(value > 0.0 for value in values) / max(1, len(values)))
            effect_size = float(stats["mean"] / stats["std"]) if stats["std"] > 0 else 0.0
            paired_payload[method][metric] = {
                **stats,
                "positive_delta_ratio": positive,
                "effect_size": effect_size,
            }
    _write_csv(output_dir / "paired_deltas.csv", paired_rows)
    (output_dir / "cross_seed_summary.json").write_text(
        json.dumps(
            {
                "format": "baseline_cross_seed_summary_v1",
                "baseline_method": args.baseline_method,
                "methods": method_payload,
                "paired_deltas": paired_payload,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
