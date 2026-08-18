#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src" / "monitoring" / "decision_gain_decomposition.py"
SPEC = importlib.util.spec_from_file_location("decision_gain_decomposition", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
AGGREGATOR_PATH = (
    ROOT
    / "src"
    / "tools"
    / "search_plans"
    / "a_series"
    / "summarize_decision_gain_decomposition.py"
)
AGGREGATOR_SPEC = importlib.util.spec_from_file_location(
    "summarize_decision_gain_decomposition", AGGREGATOR_PATH
)
AGGREGATOR = importlib.util.module_from_spec(AGGREGATOR_SPEC)
AGGREGATOR_SPEC.loader.exec_module(AGGREGATOR)
COMPONENTS = MODULE.COMPONENTS
analyze_decision_gain = MODULE.analyze_decision_gain
factorize_logits = MODULE.factorize_logits
hybrid_logits = MODULE.hybrid_logits


def _payload(seen_logits, unseen_logits):
    return {
        "test_seen": {
            "logits": np.asarray(seen_logits, dtype=np.float64),
            "targets_local": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "sample_ids": ["s0", "s1", "s2", "s3"],
        },
        "test_unseen": {
            "logits": np.asarray(unseen_logits, dtype=np.float64),
            "targets_local": np.asarray([2, 2, 3, 3], dtype=np.int64),
            "sample_ids": ["u0", "u1", "u2", "u3"],
        },
    }


def main():
    reference = _payload(
        [[3, 1, 2, 0], [2, 1, 3, 0], [1, 3, 2, 0], [2, 3, 1, 0]],
        [[3, 1, 2, 0], [2, 1, 3, 0], [1, 2, 0, 3], [1, 3, 0, 2]],
    )
    target = _payload(
        [[4, 1, 2, 0], [3, 1, 2, 0], [1, 4, 2, 0], [2, 4, 1, 0]],
        [[1, 0, 4, 2], [1, 0, 3, 2], [1, 0, 2, 4], [1, 0, 2, 3]],
    )
    summary, arrays = analyze_decision_gain(
        reference,
        target,
        [0, 1, 2, 3],
        [0, 1],
        [2, 3],
    )
    assert summary["validity"]["valid"]
    assert summary["validity"]["endpoint_factorization_equivalence_pass"]
    assert arrays["candidate_class_ids"].tolist() == [0, 1, 2, 3]
    assert abs(summary["seen_unseen_h_decomposition"]["additivity_error"]) < 1.0e-12
    for metric in ("seen_accuracy", "unseen_accuracy", "h"):
        assert abs(summary["logit_component_shapley"][metric]["additivity_error"]) < 1.0e-12
    seen_indices = [0, 1]
    unseen_indices = [2, 3]
    factored = factorize_logits(reference["test_seen"]["logits"], seen_indices, unseen_indices)
    assert np.max(np.abs(factored["reconstructed"] - factored["centered"])) < 1.0e-12
    assert np.allclose(
        hybrid_logits(factored, factored, COMPONENTS), factored["centered"]
    )
    for split in ("test_seen", "test_unseen"):
        assert f"{split}_class_true_margin_delta" in arrays
        assert f"{split}_prediction_frequency_delta" in arrays

    per_seed_arrays = {}
    rows = (
        np.asarray([0.1, -0.1, 0.0, 0.2]),
        np.asarray([0.2, -0.2, 0.0, -0.1]),
        np.asarray([0.3, -0.3, 0.0, 0.0]),
    )
    for seed, row in enumerate(rows):
        per_seed_arrays[seed] = {
            "test_seen_class_accuracy_delta": row,
            "test_unseen_class_accuracy_delta": row,
        }
    per_class = AGGREGATOR._per_class_cross_seed(
        per_seed_arrays, np.asarray([10, 11, 12, 13])
    )["test_unseen"]
    assert per_class["strictly_positive_all_seeds_count"] == 1
    assert per_class["strictly_negative_all_seeds_count"] == 1
    assert per_class["mixed_positive_negative_count"] == 1
    assert per_class["zero_all_seeds_count"] == 1
    print("decision gain decomposition validation passed")


if __name__ == "__main__":
    main()
