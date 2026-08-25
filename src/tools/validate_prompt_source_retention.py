"""Validate the lightweight Prompt-source geometry helpers used by B3-D2-G."""

from __future__ import annotations

import numpy as np

from src.monitoring.prompt_source_retention import (
    ABSOLUTE_GEOMETRY_METRICS,
    absolute_class_geometry,
    geometry_deltas,
    summarize_geometry_controls,
)


def main() -> None:
    rng = np.random.RandomState(17)
    labels = np.repeat(np.arange(4, dtype=np.int64), 6)
    centers = np.eye(4, 12, dtype=np.float64) * 3.0
    values = centers[labels] + rng.normal(scale=0.15, size=(labels.size, 12))

    geometry = absolute_class_geometry(values, labels)
    assert geometry["validity"]["valid"]
    assert geometry["class_support"]["all_classes_have_leave_one_out_support"]
    assert set(ABSOLUTE_GEOMETRY_METRICS).issubset(geometry["metrics"])
    assert geometry["metrics"]["leave_one_out_center_accuracy"] > 0.95

    shifted_scaled = values * 7.0 + 13.0
    invariant_geometry = absolute_class_geometry(shifted_scaled, labels)
    invariant_delta = geometry_deltas(invariant_geometry, geometry)
    assert max(abs(value) for value in invariant_delta.values()) < 1.0e-8

    control_summary = summarize_geometry_controls(
        [
            {"seed": 11, **geometry},
            {"seed": 12, **invariant_geometry},
        ]
    )
    for metric_name in ABSOLUTE_GEOMETRY_METRICS:
        assert np.isfinite(control_summary[metric_name]["mean"])
        assert control_summary[metric_name]["count"] == 2

    print("prompt_source_retention validation passed")


if __name__ == "__main__":
    main()
