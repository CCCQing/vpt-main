"""Validate one completed B-series P0 result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


P02_CONDITIONS = {
    "full",
    "residual_zero",
    "span_natural",
    "span_norm_matched",
    "orthogonal_natural",
    "orthogonal_norm_matched",
    "random_span_natural",
    "random_span_norm_matched",
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_p02(result):
    assert result["format"] == "b3_p0_2_prompt_subspace_consumer_v1"
    assert result["static_prompt_contract"]["valid"] is True
    assert result["static_prompt_contract"]["exact_static_prompt_match"] is True
    assert result["a2_prompt_basis"]["valid"] is True
    assert set(result["conditions"]) == P02_CONDITIONS
    reference_manifests = result["conditions"]["full"]["manifests"]
    for name, condition in result["conditions"].items():
        assert condition["valid"] is True
        assert condition["manifests"] == reference_manifests
        for split in condition["splits"].values():
            assert split["intervention_contract"]["pass"] is True
        if name not in {"full", "residual_zero"}:
            assert condition["projection_contract_pass"] is True
            for split in condition["residual_subspace"].values():
                assert split["layers"]
                for layer in split["layers"].values():
                    assert layer["subspace_projection_reconstruction_error"]["max"] <= 1.0e-5
                    assert layer["subspace_projection_basis_orthonormal_error"]["max"] <= 1.0e-4
        if result["scope"] == "probe":
            evidence = condition["prompt_state_vs_residual_zero"]
            assert evidence
            for split in evidence.values():
                assert len(split) == result["a2_prompt_basis"]["layer_count"]
                for layer in split.values():
                    assert set(layer) == {
                        "raw_input",
                        "layernorm_input",
                        "key",
                        "value",
                    }
    return {
        "condition_count": len(result["conditions"]),
        "static_prompt_tensor_count": result["static_prompt_contract"]["tensor_count"],
        "basis_layer_count": result["a2_prompt_basis"]["layer_count"],
    }


def _validate_p034(result):
    assert result["format"] == "b3_p0_34_shared_a2_replay_v1"
    assert result["identity_contract"]["valid"] is True
    p03 = result["P0-3"]
    p04 = result["P0-4"]
    assert p03["valid"] is True
    assert p03["official_seen_to_unseen"] is True
    assert p03["graph_prob_prior_used"] is False
    assert p03["unseen_used_for_fitting_or_tuning"] is False
    assert set(p03["spaces"]) == {"attribute_312", "projected_768"}
    for space in p03["spaces"].values():
        assert space["relation_validity"]["valid"] is True
        assert set(space["conditions"]) == {
            "positive_cosine",
            "semantic_1nn",
            "global_seen_mean",
        }
        assert len(space["class_shuffled_controls"]) == 5
        assert all(item["metrics"]["valid"] for item in space["conditions"].values())
    assert p04["valid"] is True
    assert p04["oracle"] is True
    assert p04["deployable"] is False
    assert p04["formal_score"] is False
    assert set(p04["conditions"]) == {
        "current_semantic_dot",
        "semantic_cosine",
        "loo_visual_centroid_cosine",
    }
    assert len(p04["centroid_shuffled_controls"]) == 5
    assert all(
        item["prediction_equivalence"] == 1.0
        and item["relative_max_abs_error"] <= 1.0e-4
        for item in p04["current_semantic_dot_reconstruction"].values()
    )
    return {
        "semantic_spaces": sorted(p03["spaces"]),
        "p03_unseen_class_count": len(p03["unseen_class_ids"]),
        "p04_unseen_minimum_support": p04["unseen_center_minimum_support"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.expanduser().resolve()
    summary = _load(root / "p0_summary.json")
    assert summary["status"] == "completed"
    assert summary["valid"] is True
    assert summary["training_performed"] is False
    assert summary["optimizer_created"] is False
    assert summary["backward_performed"] is False
    result = _load(root / summary["result_path"])
    assert result["valid"] is True
    assert result["training_performed"] is False
    assert result["optimizer_created"] is False
    assert result["backward_performed"] is False
    assert len(result["implementation_sha256"]) == 5
    assert all(
        len(value) == 64 for value in result["implementation_sha256"].values()
    )
    compact = _validate_p02(result) if summary["experiment"] == "P0-2" else _validate_p034(result)
    print(json.dumps({"valid": True, "experiment": summary["experiment"], **compact}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
