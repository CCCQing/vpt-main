#!/usr/bin/env python3
"""Validate that a checkpoint-only fixed Probe reproduces integrated evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


IGNORED_KEYS = {
    "run_id",
    "session_id",
    "diagnostic_execution_run_id",
    "diagnostic_execution_session_id",
    "diagnostic_replay",
    "checkpoint_path",
    "resolved_config",
    "reproducibility_manifest",
    "dataset_manifest",
    "probe_total_time_sec",
    "probe_data_time_sec",
    "probe_compute_time_sec",
    "probe_data_time_ratio",
}


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def _resolve(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_file():
        return path
    compressed = path.with_name(path.name + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(str(path))


def _read_json(root: Path, relative: str):
    return json.loads(_read_text(_resolve(root, relative)))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _numeric_tier(path: str) -> str:
    lowered = path.lower()
    if "neighbor_preservation_at_k" in lowered:
        return "discontinuous_neighbor_rank"
    if "spearman" in lowered:
        return "continuous_rank"
    if path.startswith(
        (
            "fixed_probe_class_aggregates.json",
            "probe_equivalence/",
            "target_relevance/",
        )
    ):
        return "strict_scientific"
    return "continuous_scientific"


def _compare(
    reference,
    replay,
    path,
    failures,
    *,
    tolerances,
    tier_audit,
):
    if isinstance(reference, dict) and isinstance(replay, dict):
        ref_keys = set(reference).difference(IGNORED_KEYS)
        replay_keys = set(replay).difference(IGNORED_KEYS)
        if ref_keys != replay_keys:
            failures.append(
                {
                    "path": path,
                    "reason": "key_mismatch",
                    "reference_only": sorted(ref_keys.difference(replay_keys)),
                    "replay_only": sorted(replay_keys.difference(ref_keys)),
                }
            )
            return
        for key in sorted(ref_keys):
            _compare(
                reference[key],
                replay[key],
                "{}.{}".format(path, key),
                failures,
                tolerances=tolerances,
                tier_audit=tier_audit,
            )
        return
    if isinstance(reference, list) and isinstance(replay, list):
        if len(reference) != len(replay):
            failures.append(
                {
                    "path": path,
                    "reason": "length_mismatch",
                    "reference": len(reference),
                    "replay": len(replay),
                }
            )
            return
        for index, (left, right) in enumerate(zip(reference, replay)):
            _compare(
                left,
                right,
                "{}[{}]".format(path, index),
                failures,
                tolerances=tolerances,
                tier_audit=tier_audit,
            )
        return
    if (
        isinstance(reference, int)
        and not isinstance(reference, bool)
        and isinstance(replay, int)
        and not isinstance(replay, bool)
    ):
        tier = "exact_integer"
        audit = tier_audit[tier]
        audit["compared_count"] += 1
        if reference != replay:
            audit["changed_count"] += 1
            failures.append(
                {
                    "path": path,
                    "reason": "integer_mismatch",
                    "tier": tier,
                    "reference": reference,
                    "replay": replay,
                }
            )
        return
    if (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(replay, (int, float))
        and not isinstance(replay, bool)
    ):
        tier = _numeric_tier(path)
        atol, rtol = tolerances[tier]
        absolute_error = abs(float(reference) - float(replay))
        audit = tier_audit[tier]
        audit["compared_count"] += 1
        if absolute_error > 0.0:
            audit["changed_count"] += 1
            audit["max_absolute_error"] = max(
                float(audit["max_absolute_error"]), absolute_error
            )
        if not math.isclose(float(reference), float(replay), abs_tol=atol, rel_tol=rtol):
            failures.append(
                {
                    "path": path,
                    "reason": "numeric_mismatch",
                    "tier": tier,
                    "atol": atol,
                    "rtol": rtol,
                    "reference": reference,
                    "replay": replay,
                    "absolute_error": absolute_error,
                }
            )
        return
    if reference != replay:
        failures.append(
            {
                "path": path,
                "reason": "value_mismatch",
                "reference": reference,
                "replay": replay,
            }
        )


def _logical_json_files(root: Path):
    result = {}
    for path in root.rglob("*.json"):
        result[path.relative_to(root).as_posix()] = path
    for path in root.rglob("*.json.gz"):
        relative = path.relative_to(root).as_posix()
        result.setdefault(relative[:-3], path)
    return result


def _selected_files(reference_root: Path, replay_root: Path):
    reference = _logical_json_files(reference_root / "diagnostics")
    replay = _logical_json_files(replay_root / "diagnostics")
    prefixes = (
        "probe_manifest.json",
        "probe_validity.json",
        "fixed_probe_class_aggregates.json",
        "probe_equivalence/",
        "module_effect/",
        "target_relevance/",
        "prompt_analysis/",
        "semantic_intervention/",
    )
    selected = sorted(
        name for name in reference if any(name == prefix or name.startswith(prefix) for prefix in prefixes)
    )
    missing = sorted(set(selected).difference(replay))
    extra = sorted(
        name
        for name in replay
        if any(name == prefix or name.startswith(prefix) for prefix in prefixes)
        and name not in reference
    )
    return selected, missing, extra


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--replay-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-5,
        help="absolute tolerance for ordinary continuous scientific fields",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--strict-atol", type=float, default=1.0e-6)
    parser.add_argument("--strict-rtol", type=float, default=1.0e-6)
    parser.add_argument("--rank-atol", type=float, default=2.0e-4)
    parser.add_argument(
        "--neighbor-rank-atol",
        type=float,
        default=7.0e-3,
        help=(
            "bounded tolerance for discontinuous top-k neighbor-preservation "
            "rates; the observed error remains recorded"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reference_run = args.reference_run.expanduser().resolve()
    replay_run = args.replay_run.expanduser().resolve()
    replay_summary = _read_json(replay_run, "probe_robustness_replay_summary.json")
    selected, missing, extra = _selected_files(reference_run, replay_run)
    failures = []
    tolerances = {
        "strict_scientific": (float(args.strict_atol), float(args.strict_rtol)),
        "continuous_scientific": (float(args.atol), float(args.rtol)),
        "continuous_rank": (float(args.rank_atol), float(args.strict_rtol)),
        "discontinuous_neighbor_rank": (
            float(args.neighbor_rank_atol),
            0.0,
        ),
    }
    tier_audit = {
        "exact_integer": {
            "comparison": "exact",
            "compared_count": 0,
            "changed_count": 0,
        }
    }
    for name, (atol, rtol) in tolerances.items():
        tier_audit[name] = {
            "comparison": "math.isclose",
            "atol": atol,
            "rtol": rtol,
            "compared_count": 0,
            "changed_count": 0,
            "max_absolute_error": 0.0,
        }
    for relative in selected:
        reference = _read_json(reference_run / "diagnostics", relative)
        replay = _read_json(replay_run / "diagnostics", relative)
        _compare(
            reference,
            replay,
            relative,
            failures,
            tolerances=tolerances,
            tier_audit=tier_audit,
        )
    valid = bool(replay_summary.get("valid", False)) and not missing and not extra and not failures
    payload = {
        "format": "fixed_probe_replay_equivalence_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_run": str(reference_run),
        "replay_run": str(replay_run),
        "reference_selection_seed": _read_json(
            reference_run / "diagnostics", "probe_manifest.json"
        ).get("selection_seed"),
        "replay_selection_seed": _read_json(
            replay_run / "diagnostics", "probe_manifest.json"
        ).get("selection_seed"),
        "replay_validator_pass": bool(replay_summary.get("valid", False)),
        "compared_json_file_count": len(selected),
        "missing_files": missing,
        "extra_files": extra,
        "failure_count": len(failures),
        "failures": failures[:200],
        "failure_payload_truncated": len(failures) > 200,
        "numeric_equivalence_tiers": tier_audit,
        "equivalence_rule": (
            "integers and nonnumeric identities are exact; fixed class aggregates, "
            "forward equivalence and target relevance use strict tolerance; ordinary "
            "continuous fields use floating-accumulation tolerance; Spearman and "
            "top-k neighbor preservation use separately declared bounded tolerances "
            "because they are rank-derived"
        ),
        "valid": valid,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    if not valid:
        raise SystemExit("fixed-Probe replay equivalence failed")
    print("Fixed-Probe replay equivalence passed: {} files".format(len(selected)))


if __name__ == "__main__":
    main()
