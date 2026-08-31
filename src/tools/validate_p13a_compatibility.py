#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.models.candidate_compatibility import (
    CandidateConditionedCompatibility,
    GlobalTemperatureCompatibility,
    ImageOnlyTemperatureCompatibility,
)


TRAINED = {
    "candidate_raw312",
    "candidate_projected768",
    "semantic_permuted_raw312",
    "image_constant_raw312",
    "image_only_temperature",
    "temperature_only",
}
REFERENCES = {"current_semantic_dot", "semantic_cosine"}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def _synthetic() -> dict:
    torch.manual_seed(19)
    visual = torch.randn(8, 12, requires_grad=True)
    semantic = torch.randn(5, 7)
    candidate = CandidateConditionedCompatibility(12, 7, hidden_dim=6, dropout=0.0)
    logits = candidate(visual, semantic, candidate_chunk_size=2)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(8) % 5)
    loss.backward()
    image_only_semantic = torch.randn(5, 12)
    image_only = ImageOnlyTemperatureCompatibility(12, hidden_dim=4)
    image_only_logits = image_only(visual.detach(), image_only_semantic)
    global_temperature = GlobalTemperatureCompatibility()
    global_logits = global_temperature(visual.detach(), image_only_semantic)
    checks = {
        "candidate_shape": tuple(logits.shape) == (8, 5),
        "candidate_finite": bool(torch.isfinite(logits).all().item()),
        "candidate_gradients": all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in candidate.parameters()
        ),
        "image_only_shape": tuple(image_only_logits.shape) == (8, 5),
        "global_temperature_shape": tuple(global_logits.shape) == (8, 5),
        "chunk_equivalence": bool(
            torch.allclose(
                candidate(visual.detach(), semantic, candidate_chunk_size=2),
                candidate(visual.detach(), semantic, candidate_chunk_size=5),
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        ),
    }
    return {"mode": "synthetic", "checks": checks, "valid": all(checks.values())}


def _validate_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions") or {}
    contract = payload.get("execution_contract") or {}
    cache = payload.get("feature_cache_manifest") or {}
    checks = {
        "format": payload.get("format") == "p13a_candidate_conditioned_head_only_result_v1",
        "completed": payload.get("status") == "completed",
        "condition_coverage": set(conditions) == TRAINED | REFERENCES,
        "head_only": bool(contract.get("a2_backbone_frozen"))
        and bool(contract.get("a2_prompt_frozen"))
        and contract.get("optimizer_scope") == "candidate_compatibility_head_only",
        "fixed_training_epoch_rule": contract.get("training_epoch_rule")
        == "predeclared_fixed_epoch"
        and int(contract.get("train_epochs", 0)) > 0,
        "official_seen_training": contract.get(
            "all_official_seen_classes_used_for_training"
        )
        is True,
        "official_final_evaluation": contract.get("evaluation_protocol")
        == "official_final_gzsl",
        "single_feature_extraction": cache.get("feature_extraction_count_per_split") == 1,
        "feature_cache_valid": cache.get("valid") is True,
        "strict_three_probe": all(
            set((item.get("strict_three_probe") or {}).keys())
            == {"424242", "424243", "424244"}
            for item in conditions.values()
        ),
        "finite_task_metrics": True,
        "trained_checkpoint_identity": True,
    }
    for name, item in conditions.items():
        metrics = ((item.get("task_metrics") or {}).get("normal_gzsl") or {})
        for key in ("seen", "unseen", "harmonic_mean", "ausuc"):
            value = metrics.get(key)
            if value is None or not math.isfinite(float(value)):
                checks["finite_task_metrics"] = False
        if name in TRAINED:
            final = item.get("final_training") or {}
            checkpoint = path.parent / str(final.get("checkpoint_path", ""))
            if (
                item.get("identity", {}).get("kind") != "trained_head_only"
                or not checkpoint.is_file()
                or not str(final.get("checkpoint_sha256", ""))
                or int(final.get("train_epochs", 0)) <= 0
            ):
                checks["trained_checkpoint_identity"] = False
    return {
        "mode": "artifact",
        "result": str(path),
        "checks": checks,
        "valid": all(checks.values()),
    }


def main() -> None:
    args = _parse()
    report = _validate_result(args.result.resolve()) if args.result else _synthetic()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
