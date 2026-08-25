"""Validate one completed B3-D2-G checkpoint-only result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train_seen", "test_seen", "test_unseen")
RANDOM_MLP_SEEDS = [23001, 23002, 23003, 23004, 23005]
GEOMETRY_SPACES = (
    "prepass_cls_geometry",
    "trained_mu_geometry",
    "normal_final_cls_geometry",
    "residual_zero_final_cls_geometry",
)
LOGIT_VIEWS = ("centered_logits", "direction_normalized_logits", "class_pattern")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.expanduser().resolve()
    summary = _load(root / "b3_followup_summary.json")
    result = _load(root / "B3-D2G-source-to-decision-chain.json")

    assert summary["status"] == "completed"
    assert summary["valid"] is True
    assert summary["experiments"] == ["D2G"]
    assert summary["D2G"]["valid"] is True
    assert result["format"] == "b3_d2g_source_to_decision_chain_v2"
    assert result["valid"] is True
    assert result["training_performed"] is False
    assert result["optimizer_created"] is False
    assert result["backward_performed"] is False
    assert summary["source_checkpoint_sha256"] == result["source_checkpoint_sha256"]
    assert result["random_mlp_seeds"] == RANDOM_MLP_SEEDS
    assert result["downstream_validity"]["valid"] is True
    semantic = result["semantic_reference"]
    assert semantic["validity"]["valid"] is True
    assert semantic["prototype_count"] == len(semantic["candidate_class_ids"])
    assert result["decision_space_dim"] == semantic["prototype_count"]
    assert result["decision_space_dim"] == 200
    relation = semantic["cosine_relation_matrix"]
    assert len(relation) == semantic["prototype_count"]
    assert all(len(row) == semantic["prototype_count"] for row in relation)

    compact = {}
    for split in SPLITS:
        item = result["splits"][split]
        assert item["validity"]["valid"] is True
        assert item["sample_manifest"]["sha256"] == item[
            "residual_zero_sample_manifest"
        ]["sha256"]
        assert item["prepass_equivalence"][
            "normal_vs_residual_zero_exact_equal"
        ] is True
        for space in GEOMETRY_SPACES:
            assert item[space]["validity"]["valid"] is True
            assert item[space]["class_support"][
                "all_classes_have_leave_one_out_support"
            ] is True
        assert len(item["random_mu_geometry"]) == len(result["random_mlp_seeds"])
        assert all(control["validity"]["valid"] for control in item["random_mu_geometry"])
        assert item["random_mu_geometry_summary"]["fisher_trace_ratio"]["count"] == len(
            result["random_mlp_seeds"]
        )
        assert item["visual_semantic_alignment"]["normal"]["validity"]["valid"]
        assert item["visual_semantic_alignment"]["residual_zero"]["validity"][
            "valid"
        ]
        assert item["prepass_cache_contract"]["normal"]["mode"] == "populate"
        assert item["prepass_cache_contract"]["residual_zero"]["mode"] == "reuse"
        assert item["prepass_cache_contract"][
            "same_cached_prepass_verified_by_exact_hash"
        ] is True
        assert item["prepass_cache_contract"]["cache_released_after_split"] is True
        assert set(item["paired_deltas"]) == {
            "trained_mu_minus_prepass_cls",
            "trained_mu_minus_random_mean",
            "trained_mu_minus_each_random",
            "normal_final_cls_minus_residual_zero_final_cls",
        }
        compact[split] = {
            "valid": True,
            "sample_count": item["sample_manifest"]["sample_count"],
            "class_count": item["sample_manifest"]["class_count"],
            "minimum_class_support": item["trained_mu_geometry"]["class_support"][
                "min_count"
            ],
            "prepass_fisher": item["prepass_cls_geometry"]["metrics"][
                "fisher_trace_ratio"
            ],
            "trained_mu_fisher": item["trained_mu_geometry"]["metrics"][
                "fisher_trace_ratio"
            ],
            "random_mu_fisher_mean": item["random_mu_geometry_summary"][
                "fisher_trace_ratio"
            ]["mean"],
            "normal_final_cls_fisher": item["normal_final_cls_geometry"][
                "metrics"
            ]["fisher_trace_ratio"],
            "residual_zero_final_cls_fisher": item[
                "residual_zero_final_cls_geometry"
            ]["metrics"]["fisher_trace_ratio"],
            "normal_minus_residual_zero_semantic_margin": item[
                "visual_semantic_alignment"
            ]["paired_deltas"]["normal_minus_residual_zero"]["alignment"][
                "semantic_margin"
            ],
        }
    logit = result["logit_geometry"]
    for condition in ("normal", "residual_zero"):
        payload = logit[condition]
        assert payload["validity"]["valid"] is True
        assert payload["validity"]["factorization_reconstruction_pass"] is True
        assert payload["validity"][
            "centered_and_direction_endpoint_equivalence_pass"
        ] is True
        for split in SPLITS:
            assert split in payload["splits"]
            for view in LOGIT_VIEWS:
                view_payload = payload["splits"][split]["views"][view]
                assert view_payload["validity"]["valid"]
                assert view_payload["vector_dim"] == result["decision_space_dim"]
        for split, equivalence in payload["validity"]["endpoint_equivalence"].items():
            assert split in SPLITS
            assert equivalence["raw_vs_centered_prediction_flip_rate"] == 0.0
            assert equivalence["raw_vs_direction_normalized_prediction_flip_rate"] == 0.0
            assert equivalence["zero_direction_count"] == 0
    assert set(logit["paired_deltas"]["normal_minus_residual_zero"]) == {
        *SPLITS,
        "joint",
    }
    tasks = result["task_results"]
    assert tasks["valid"] is True
    for condition in ("normal", "residual_zero"):
        assert tasks[condition]["valid"] is True
        for metric in (
            "seen_per_class_accuracy",
            "unseen_per_class_accuracy",
            "harmonic_mean",
            "ausuc",
        ):
            assert metric in tasks[condition]["gzsl"]
            assert metric in tasks["normal_minus_residual_zero"]
    compact["task_delta"] = tasks["normal_minus_residual_zero"]
    print(json.dumps({"valid": True, "splits": compact}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
