#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.representation_transport import (
    classifier_logits_from_features,
    head_representation_factorial_metrics,
    prediction_transition_group_metrics,
    representation_transport_metrics,
    semantic_transport_metrics,
)


def main() -> int:
    rng = np.random.RandomState(20260826)
    class_count = 6
    sample_per_class = 5
    visual_dim = 12
    raw_dim = 7
    targets = np.repeat(np.arange(class_count), sample_per_class)
    semantic = rng.normal(size=(class_count, visual_dim))
    raw_semantic = rng.normal(size=(class_count, raw_dim))
    prepass = semantic[targets] + rng.normal(scale=1.2, size=(targets.size, visual_dim))
    final = semantic[targets] + rng.normal(scale=0.35, size=(targets.size, visual_dim))
    transport = representation_transport_metrics(
        prepass, final, targets, pair_seed=41, max_pairs=200
    )
    assert transport["validity"]["valid"]
    assert set(transport["absolute_geometry"]) == {
        "frozen_prepass_cls",
        "normal_final_cls",
        "cls_delta",
    }
    assert set(transport["final_minus_prepass_geometry"])

    prepass_logits = classifier_logits_from_features(
        prepass, semantic, score_mode="dot"
    )
    final_logits = classifier_logits_from_features(
        final, semantic, score_mode="dot"
    )
    cosine_logits = classifier_logits_from_features(
        final, semantic, score_mode="cosine", logit_scale=3.5
    )
    assert prepass_logits.shape == final_logits.shape == (targets.size, class_count)
    assert np.isfinite(cosine_logits).all()

    semantic_transport = semantic_transport_metrics(
        prepass,
        final,
        semantic,
        targets,
        prepass_logits=prepass_logits,
        final_logits=final_logits,
        raw_semantic_prototypes=raw_semantic,
    )
    assert semantic_transport["validity"]["valid"]
    assert semantic_transport["raw_312d_vs_projected_768d_relation"][
        "status"
    ] == "available"
    assert "true_prototype_rank_improvement" in semantic_transport[
        "paired_deltas"
    ]["final_minus_prepass"]

    manual_reference = np.full((targets.size, class_count), -2.0)
    manual_target = np.full((targets.size, class_count), -2.0)
    for index, true_class in enumerate(targets):
        wrong_class = int((true_class + 1) % class_count)
        pattern = index % 4
        reference_prediction = true_class if pattern in {0, 2} else wrong_class
        target_prediction = true_class if pattern in {0, 1} else wrong_class
        manual_reference[index, reference_prediction] = 2.0
        manual_target[index, target_prediction] = 2.0
    transitions = prediction_transition_group_metrics(
        prepass,
        final,
        manual_reference,
        manual_target,
        targets,
        list(range(class_count)),
        [0, 1, 2],
        semantic,
        reference_name="reference",
        target_name="target",
    )
    assert transitions["validity"]["valid"]
    assert all(
        transitions["groups"][name]["sample_count"] > 0
        for name in ("stable_correct", "corrected", "regressed", "stable_wrong")
    )

    partial = head_representation_factorial_metrics(
        {
            "frozen_prepass_cls": {"current_head": prepass_logits},
            "normal_final_cls": {"current_head": final_logits},
        },
        targets,
        list(range(class_count)),
        [0, 1, 2],
    )
    assert partial["status"] == "partial_current_head_only"
    assert len(partial["missing_cells"]) == 2
    assert partial["interaction"] is None

    candidate_prepass = prepass_logits + 0.05 * rng.normal(size=prepass_logits.shape)
    candidate_final = final_logits + 0.05 * rng.normal(size=final_logits.shape)
    complete = head_representation_factorial_metrics(
        {
            "frozen_prepass_cls": {
                "current_head": prepass_logits,
                "candidate_head": candidate_prepass,
            },
            "normal_final_cls": {
                "current_head": final_logits,
                "candidate_head": candidate_final,
            },
        },
        targets,
        list(range(class_count)),
        [0, 1, 2],
    )
    assert complete["status"] == "complete"
    assert complete["interaction"] is not None
    print(
        json.dumps(
            {
                "valid": True,
                "transport_valid": transport["validity"]["valid"],
                "semantic_transport_valid": semantic_transport["validity"]["valid"],
                "transition_groups": {
                    name: transitions["groups"][name]["sample_count"]
                    for name in transitions["groups"]
                },
                "partial_factorial_status": partial["status"],
                "complete_factorial_status": complete["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
