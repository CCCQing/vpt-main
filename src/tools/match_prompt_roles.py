#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitoring.prompt_analysis import match_prompt_role_vectors


def _read_profile(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "prompt_role_profile_v1":
        raise ValueError(f"unsupported role profile format: {path}")
    if not isinstance(payload.get("layers"), dict):
        raise ValueError(f"role profile has no layers: {path}")
    return payload


def _aligned_vectors(reference: dict, candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    reference_ids = [int(item) for item in reference["class_local_ids"]]
    candidate_ids = [int(item) for item in candidate["class_local_ids"]]
    if set(reference_ids) != set(candidate_ids):
        raise ValueError("role profiles do not contain the same class-local IDs")
    reference_vectors = np.asarray(reference["prompt_vectors"], dtype=np.float64)
    candidate_vectors = np.asarray(candidate["prompt_vectors"], dtype=np.float64)
    candidate_position = {
        class_id: index for index, class_id in enumerate(candidate_ids)
    }
    order = [candidate_position[class_id] for class_id in reference_ids]
    return reference_vectors, candidate_vectors[:, order]


def match_profiles(reference: dict, candidate: dict, *, max_cost: float) -> dict:
    if reference.get("split") != candidate.get("split"):
        raise ValueError("role profiles must use the same split")
    for field in ("probe_id", "probe_manifest_sha256", "selection_seed"):
        left = reference.get(field)
        right = candidate.get(field)
        if left is None or right is None or left != right:
            raise ValueError(
                f"role profiles must have the same non-empty {field}"
            )
    reference_checkpoint = reference.get("checkpoint", {})
    candidate_checkpoint = candidate.get("checkpoint", {})
    if (
        reference_checkpoint.get("checkpoint_id") is None
        or reference_checkpoint.get("checkpoint_id")
        != candidate_checkpoint.get("checkpoint_id")
    ):
        raise ValueError("role profiles must use the same checkpoint identity")
    reference_types = list(reference.get("prompt_types", []))
    candidate_types = list(candidate.get("prompt_types", []))
    if not reference_types or not candidate_types:
        raise ValueError("role profiles must declare prompt_types")
    reference_layers = reference["layers"]
    candidate_layers = candidate["layers"]
    common_layers = sorted(
        set(reference_layers).intersection(candidate_layers), key=int
    )
    if not common_layers:
        raise ValueError("role profiles have no common transformer layer")
    by_layer_role = {}
    aggregate_values = []
    for layer_key in common_layers:
        left_layer = reference_layers[layer_key]
        right_layer = candidate_layers[layer_key]
        for role_name in ("cls_consumption", "patch_collection"):
            if role_name not in left_layer or role_name not in right_layer:
                continue
            left_vectors, right_vectors = _aligned_vectors(
                left_layer[role_name], right_layer[role_name]
            )
            result = match_prompt_role_vectors(
                left_vectors,
                right_vectors,
                reference_types=reference_types,
                candidate_types=candidate_types,
                max_cost=max_cost,
            )
            by_layer_role[f"layer_{int(layer_key)}/{role_name}"] = result
            aggregate_values.append(result)
    if not aggregate_values:
        raise ValueError("role profiles have no common role vector")
    summary_names = (
        "matched_role_acceptance_ratio",
        "matched_role_cosine_mean",
        "matched_dominant_class_agreement",
        "unmatched_or_dead_prompt_ratio",
        "best_second_cost_gap_mean",
    )
    summary = {
        name: float(np.mean([item[name] for item in aggregate_values]))
        for name in summary_names
    }
    return {
        "format": "prompt_role_stability_v1",
        "split": reference["split"],
        "max_cost": float(max_cost),
        "cost": "one_minus_cosine",
        "matching": "hungarian_with_dummy_nodes_and_prompt_type_constraint",
        "layer_scope": "same_layer_only",
        "same_source_run": bool(
            reference_checkpoint.get("source_run_id")
            == candidate_checkpoint.get("source_run_id")
        ),
        "reference": {
            "run_id": reference.get("checkpoint", {}).get("source_run_id"),
            "checkpoint_id": reference.get("checkpoint", {}).get(
                "checkpoint_id"
            ),
            "probe_id": reference.get("probe_id"),
            "probe_manifest_sha256": reference.get(
                "probe_manifest_sha256"
            ),
        },
        "candidate": {
            "run_id": candidate.get("checkpoint", {}).get("source_run_id"),
            "checkpoint_id": candidate.get("checkpoint", {}).get(
                "checkpoint_id"
            ),
            "probe_id": candidate.get("probe_id"),
            "probe_manifest_sha256": candidate.get(
                "probe_manifest_sha256"
            ),
        },
        "summary": summary,
        "by_layer_role": by_layer_role,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match Prompt functional role profiles across two runs."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cost", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.max_cost <= 2.0:
        raise ValueError("--max-cost must be in [0, 2]")
    result = match_profiles(
        _read_profile(args.reference),
        _read_profile(args.candidate),
        max_cost=args.max_cost,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
