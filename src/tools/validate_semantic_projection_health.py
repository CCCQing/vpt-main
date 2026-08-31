"""Synthetic validation for 312-to-768 semantic projection monitoring."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is None:
    module_path = ROOT / "src" / "monitoring" / "eval_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "semantic_projection_eval_metrics", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load eval_metrics for dependency-light validation")
    eval_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_metrics)
    semantic_projection_health_metrics = (
        eval_metrics.semantic_projection_health_metrics
    )
    StreamingFixedProbeAccumulator = None
else:
    from src.monitoring.eval_metrics import semantic_projection_health_metrics
    from src.monitoring.probe import StreamingFixedProbeAccumulator


def _semantic_inputs():
    raw = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.8, 0.3],
            [0.0, 0.0, 1.0],
            [0.4, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    projected = np.concatenate(
        [raw, np.zeros((raw.shape[0], 5), dtype=np.float32)], axis=1
    )
    visual_centers = np.concatenate(
        [raw, np.zeros((raw.shape[0], 5), dtype=np.float32)], axis=1
    )
    return raw, projected, visual_centers


def _validate_identical_relations() -> None:
    raw, projected, visual_centers = _semantic_inputs()
    metrics = semantic_projection_health_metrics(
        raw,
        projected,
        visual_class_centers=visual_centers,
        observed_class_ids=np.arange(raw.shape[0]),
    )
    assert np.isclose(metrics["projection_relation_spearman"], 1.0)
    assert np.isclose(metrics["projection_neighbor_preservation_at_1"], 1.0)
    assert np.isclose(metrics["projection_neighbor_preservation_at_5"], 1.0)
    assert np.isclose(metrics["projection_new_false_high_edge_rate"], 0.0)
    assert np.isclose(metrics["projection_dropped_high_edge_rate"], 0.0)
    assert np.isclose(metrics["semantic_visual_relation_spearman_312"], 1.0)
    assert np.isclose(metrics["semantic_visual_relation_spearman_768"], 1.0)
    assert np.isclose(
        metrics["semantic_visual_relation_spearman_delta_768_minus_312"], 0.0
    )
    assert 0.0 < metrics["semantic_relation_normalized_effective_rank_312"] <= 1.0
    assert np.isclose(
        metrics["semantic_relation_normalized_effective_rank_312"],
        metrics["semantic_relation_normalized_effective_rank_768"],
    )


def _validate_distorted_projection() -> None:
    raw, projected, visual_centers = _semantic_inputs()
    distorted = projected[[0, 2, 4, 1, 3, 5]]
    metrics = semantic_projection_health_metrics(
        raw,
        distorted,
        visual_class_centers=visual_centers,
        observed_class_ids=np.arange(raw.shape[0]),
    )
    assert metrics["projection_relation_spearman"] < 0.95
    assert metrics["projection_neighbor_preservation_at_1"] < 1.0
    assert metrics["semantic_visual_relation_spearman_768"] < metrics[
        "semantic_visual_relation_spearman_312"
    ]


def _validate_streaming_integration() -> None:
    if StreamingFixedProbeAccumulator is None:
        return
    raw, projected, _ = _semantic_inputs()
    targets = np.repeat(np.arange(raw.shape[0]), 2)
    rng = np.random.RandomState(37)
    visual = projected[targets] + rng.normal(
        scale=0.01, size=(targets.size, projected.shape[1])
    ).astype(np.float32)
    logits = visual @ projected.T

    left = StreamingFixedProbeAccumulator(np.arange(raw.shape[0]))
    right = StreamingFixedProbeAccumulator(np.arange(raw.shape[0]))
    left.set_raw_semantic_prototypes(raw)
    right.set_raw_semantic_prototypes(raw)
    midpoint = targets.size // 2
    left.update(
        logits[:midpoint], targets[:midpoint], visual[:midpoint], projected
    )
    right.update(
        logits[midpoint:], targets[midpoint:], visual[midpoint:], projected
    )
    left.merge_from(right)
    result = left.finalize()
    health = result["semantic_projection_health"]
    assert "projection_relation_spearman" in health
    assert "semantic_visual_relation_spearman_312" in health
    assert "semantic_visual_relation_spearman_768" in health
    assert 0.0 <= result["classification"]["top1"] <= 1.0
    assert left.sample_count == targets.size


def _validate_contract_failures() -> None:
    raw, projected, _ = _semantic_inputs()
    try:
        semantic_projection_health_metrics(raw[:-1], projected)
    except ValueError:
        pass
    else:
        raise AssertionError("class-order mismatch must fail the validity contract")
    invalid = raw.copy()
    invalid[0] = 0.0
    try:
        semantic_projection_health_metrics(invalid, projected)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-direction semantic rows must fail")


def main() -> None:
    _validate_identical_relations()
    _validate_distorted_projection()
    _validate_streaming_integration()
    _validate_contract_failures()
    print("semantic projection health validation passed")
    if StreamingFixedProbeAccumulator is None:
        print("streaming integration skipped: torch is not installed")


if __name__ == "__main__":
    main()
