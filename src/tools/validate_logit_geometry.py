"""Synthetic validation for standardized Fisher and logit geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.eval_metrics import class_geometry_trace_metrics
from src.monitoring.logit_geometry import analyze_logit_geometry, analyze_vector_geometry
from src.monitoring.probe import StreamingRepresentationAccumulator


NEW_FISHER_FIELDS = {
    "within_class_scatter_trace",
    "between_class_scatter_trace",
    "fisher_trace_ratio",
}
OLD_FISHER_FIELDS = {
    "within_class_scatter",
    "between_class_scatter",
    "fisher_ratio",
}


def _validate_standardized_fisher() -> None:
    values = np.asarray([[0.0], [2.0], [5.0], [5.0], [5.0], [5.0]], dtype=np.float32)
    targets = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    metrics = class_geometry_trace_metrics(values, targets)
    assert set(metrics) == NEW_FISHER_FIELDS
    assert OLD_FISHER_FIELDS.isdisjoint(metrics)
    # Class 0 has mean squared distance 1; class 1 has 0. Macro class weighting gives W=0.5.
    assert np.isclose(metrics["within_class_scatter_trace"], 0.5)
    # Class centers are 1 and 5, macro center is 3, so B=(4+4)/2=4 and F=8.
    assert np.isclose(metrics["between_class_scatter_trace"], 4.0)
    assert np.isclose(metrics["fisher_trace_ratio"], 8.0)

    accumulator = StreamingRepresentationAccumulator(2, track_covariance=False)
    accumulator.update(values[:3], targets[:3])
    accumulator.update(values[3:], targets[3:])
    streamed = accumulator.finalize()
    for name, expected in metrics.items():
        assert np.isclose(streamed[name], expected, rtol=1.0e-6, atol=1.0e-6), name
    assert OLD_FISHER_FIELDS.isdisjoint(streamed)


def _synthetic_outputs():
    targets_seen = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    targets_unseen = np.asarray([2, 2, 2, 3, 3, 3], dtype=np.int64)
    logits_seen = np.asarray(
        [
            [5.0, 1.0, 0.1, -0.2],
            [4.7, 1.2, 0.0, -0.1],
            [5.2, 0.8, -0.1, 0.0],
            [1.0, 5.1, 0.2, -0.2],
            [1.3, 4.8, 0.0, -0.1],
            [0.8, 5.3, -0.1, 0.1],
        ],
        dtype=np.float64,
    )
    logits_unseen = np.asarray(
        [
            [0.3, 0.1, 4.8, 1.1],
            [0.1, 0.2, 5.2, 0.8],
            [0.2, 0.0, 4.9, 1.0],
            [0.1, 0.3, 0.9, 5.0],
            [0.0, 0.2, 1.2, 4.7],
            [0.2, 0.1, 0.8, 5.3],
        ],
        dtype=np.float64,
    )
    return {
        "test_seen": {"logits": logits_seen, "targets_local": targets_seen},
        "test_unseen": {"logits": logits_unseen, "targets_local": targets_unseen},
    }


def _validate_logit_geometry() -> None:
    outputs = _synthetic_outputs()
    summary, arrays = analyze_logit_geometry(
        outputs,
        candidate_class_ids=[10, 11, 20, 21],
        seen_class_ids=[10, 11],
        unseen_class_ids=[20, 21],
    )
    assert summary["validity"]["valid"]
    assert set(summary["views"]) == {
        "centered_logits",
        "direction_normalized_logits",
        "class_pattern",
    }
    for split in ("test_seen", "test_unseen"):
        for view in summary["views"]:
            metrics = summary["splits"][split]["views"][view]
            assert NEW_FISHER_FIELDS.issubset(metrics)
            assert OLD_FISHER_FIELDS.isdisjoint(metrics)
            assert metrics["nearest_center_valid_sample_ratio"] == 1.0
    assert not any("logits" == key or key.endswith("sample_logits") for key in arrays)

    transformed = {}
    for split, payload in outputs.items():
        logits = payload["logits"]
        offsets = np.arange(logits.shape[0], dtype=np.float64)[:, None]
        transformed[split] = {
            **payload,
            "logits": 3.0 * logits + offsets,
        }
    shifted, _ = analyze_logit_geometry(
        transformed,
        candidate_class_ids=[10, 11, 20, 21],
        seen_class_ids=[10, 11],
        unseen_class_ids=[20, 21],
    )
    for split in ("test_seen", "test_unseen"):
        for view in ("direction_normalized_logits", "class_pattern"):
            original_metrics = summary["splits"][split]["views"][view]
            shifted_metrics = shifted["splits"][split]["views"][view]
            for name in (
                "fisher_trace_ratio",
                "within_class_pairwise_cosine_mean",
                "interclass_center_cosine_mean",
                "nearest_class_center_cosine_margin_mean",
                "leave_one_out_center_accuracy",
            ):
                assert np.isclose(original_metrics[name], shifted_metrics[name], atol=1.0e-9), (
                    split,
                    view,
                    name,
                )


def _validate_generic_vector_geometry() -> None:
    values = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float64,
    )
    targets = np.asarray([10, 10, 20, 20], dtype=np.int64)
    metrics, arrays, validity = analyze_vector_geometry(
        values, targets, expected_classes=[10, 20]
    )
    assert validity["valid"]
    assert metrics["leave_one_out_center_accuracy"] == 1.0
    assert arrays["class_ids"].tolist() == [10, 20]


def main() -> None:
    _validate_standardized_fisher()
    _validate_logit_geometry()
    _validate_generic_vector_geometry()
    print("logit geometry validation passed")


if __name__ == "__main__":
    main()
