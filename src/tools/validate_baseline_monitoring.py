#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.configs.config import get_cfg
from src.monitoring import (
    DiagnosticManager,
    MonitorManager,
    NumericalGuard,
    OptimizerSanity,
    PromptParameterTracker,
)
from src.monitoring.eval_metrics import (
    calibration_profile_metrics,
    class_error_metrics,
    classification_metrics,
    prediction_health_metrics,
    representation_geometry_metrics,
    semantic_graph_reference_metrics,
    semantic_visual_graph_metrics,
    visual_semantic_alignment_metrics,
)
from src.monitoring.module_effect import paired_module_effect_metrics


class SyntheticDataset:
    protocol_mode = "final_gzsl"
    eval_local_classes = [0, 1, 2, 3]
    seen_classes = [0, 1]
    unseen_classes = [2, 3]
    all_classnames = ["c0", "c1", "c2", "c3"]
    class_attributes = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.2],
            [0.8, 0.2, 0.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 0.8, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )


def main():
    rng = np.random.RandomState(7)
    seen_targets = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    unseen_targets = np.asarray([2, 2, 3, 3, 2, 3], dtype=np.int64)
    seen_scores = rng.normal(size=(seen_targets.size, 4))
    unseen_scores = rng.normal(size=(unseen_targets.size, 4))
    seen_scores[np.arange(seen_targets.size), seen_targets] += 2.0
    unseen_scores[np.arange(unseen_targets.size), unseen_targets] += 1.5
    visual_seen = rng.normal(size=(seen_targets.size, 5))
    visual_unseen = rng.normal(size=(unseen_targets.size, 5))
    semantic = SyntheticDataset.class_attributes.numpy()

    assert set(classification_metrics(seen_scores, seen_targets)) == {"top1", "top5", "nll", "per_class"}
    assert prediction_health_metrics(seen_scores, seen_targets, [0, 1, 2, 3], [0, 1])
    assert class_error_metrics(seen_scores, seen_targets, [0, 1, 2, 3])["arrays"]["per_class_accuracy"].shape == (4,)
    assert calibration_profile_metrics(
        seen_scores, seen_targets, unseen_scores, unseen_targets, [0, 1, 2, 3], [0, 1], [-1.0, 0.0, 1.0]
    )["summary"]["raw_h"] >= 0.0
    assert representation_geometry_metrics(visual_seen, seen_targets)
    assert visual_semantic_alignment_metrics(visual_seen, semantic, seen_targets)
    assert semantic_graph_reference_metrics(semantic, [0, 1], [2, 3])
    assert semantic_visual_graph_metrics(visual_seen, semantic, seen_targets, logits=seen_scores)
    effect = paired_module_effect_metrics(
        seen_scores,
        seen_scores + rng.normal(scale=0.01, size=seen_scores.shape),
        seen_targets,
        [0, 1, 2, 3],
        [0, 1],
    )
    assert "prediction_flip_rate" in effect["summary"]

    model = torch.nn.Linear(5, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    modules = (("model", model),)
    numerical = NumericalGuard(modules)
    sanity = OptimizerSanity(modules, optimizer)
    assert sanity.initialization_report()["passed"]
    batch = torch.randn(8, 5)
    target = torch.randint(0, 4, (8,))
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, target)
    assert numerical.check_forward(loss, logits) is None
    loss.backward()
    assert numerical.check_gradients() is None
    optimizer.step()
    assert numerical.check_parameters() is None
    assert sanity.first_step_report()["passed"]
    prompt_model = torch.nn.Module()
    prompt_model.register_parameter("prompt_embeddings", torch.nn.Parameter(torch.randn(1, 5, 8)))
    prompt_metrics = PromptParameterTracker(prompt_model).metrics()
    assert "prompt_effective_rank" in prompt_metrics

    with tempfile.TemporaryDirectory(prefix="baseline_monitor_validation_") as temp_dir:
        cfg = get_cfg()
        cfg.OUTPUT_DIR = temp_dir
        cfg.DATA.XLSA.PROTOCOL_MODE = "final_gzsl"
        cfg.MONITOR.ENABLE = True
        cfg.MONITOR.OUTPUT_POLICY = "error_if_exists"
        cfg.MONITOR.PROBE.ENABLE = False
        cfg.MONITOR.MODULE_EFFECT.ENABLE = False
        manager = MonitorManager(cfg, is_writer=True)
        diagnostics = DiagnosticManager(cfg, manager, is_writer=True)
        dataset = SyntheticDataset()
        diagnostics.record_static_semantic_graph(dataset)
        manager.set_context(stage="eval", epoch=1, global_step=10)
        diagnostics.record_eval(
            epoch=1,
            split="test_seen",
            scores=seen_scores,
            targets_local=seen_targets,
            targets_global=seen_targets,
            sample_ids=[f"seen:{index}" for index in range(seen_targets.size)],
            dataset=dataset,
            visual_features=visual_seen,
        )
        diagnostics.record_eval(
            epoch=1,
            split="test_unseen",
            scores=unseen_scores,
            targets_local=unseen_targets,
            targets_global=unseen_targets,
            sample_ids=[f"unseen:{index}" for index in range(unseen_targets.size)],
            dataset=dataset,
            visual_features=visual_unseen,
        )
        diagnostics.record_calibration(1)
        cadence_failed = False
        try:
            manager.record_epoch("train", "train", {"loss": 1.0})
        except ValueError:
            cadence_failed = True
        assert cadence_failed
        diagnostics.finalize(status="completed")
        manager.finalize(status="completed")
        required = (
            "monitor_manifest.json",
            "metrics_epoch.csv",
            "metrics_events.jsonl",
            "monitor_runtime_summary.json",
            "diagnostics/diagnostic_manifest.json",
            "diagnostics/diagnostic_runtime_summary.json",
            "diagnostics/calibration_profile/epoch_0001.json",
            "diagnostics/class_error/epoch_0001/test_seen.npz",
        )
        for relative in required:
            assert (Path(temp_dir) / relative).exists(), relative

        original_session_id = manager.session_id
        collision_failed = False
        try:
            MonitorManager(cfg, is_writer=True)
        except FileExistsError:
            collision_failed = True
        assert collision_failed

        cfg.MONITOR.OUTPUT_POLICY = "resume"
        resumed = MonitorManager(cfg, is_writer=True)
        assert resumed.resumed
        assert resumed.session_id == original_session_id
        resumed_diagnostics = DiagnosticManager(cfg, resumed, is_writer=True)
        resumed_diagnostics.finalize(status="completed")
        resumed.finalize(status="completed")

        cfg.MONITOR.OUTPUT_POLICY = "overwrite"
        overwritten = MonitorManager(cfg, is_writer=True)
        assert not overwritten.resumed
        assert overwritten.session_id != original_session_id
        overwritten_diagnostics = DiagnosticManager(cfg, overwritten, is_writer=True)
        overwritten_diagnostics.finalize(status="completed")
        overwritten.finalize(status="completed")

        interrupted_dir = Path(temp_dir) / "interrupted"
        cfg.OUTPUT_DIR = str(interrupted_dir)
        cfg.MONITOR.OUTPUT_POLICY = "error_if_exists"
        interrupted = MonitorManager(cfg, is_writer=True)
        interrupted_diagnostics = DiagnosticManager(cfg, interrupted, is_writer=True)
        interrupted_diagnostics.finalize(status="interrupted")
        interrupted.finalize(status="interrupted")
        interrupted_summary = json.loads(
            (interrupted_dir / "monitor_runtime_summary.json").read_text(encoding="utf-8")
        )
        assert interrupted_summary["status"] == "interrupted"
    print("PASS: baseline monitoring synthetic validation")


if __name__ == "__main__":
    main()
