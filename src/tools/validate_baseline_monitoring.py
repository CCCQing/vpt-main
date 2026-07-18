#!/usr/bin/env python3

from __future__ import annotations

import csv
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
from src.monitoring.probe import build_probe_manifest


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


class SyntheticProbeDataset:
    seen_classes = [0, 1]
    _imdb = [
        {"class": 0, "sample_id": "c0-a", "im_path": "c0-a.jpg"},
        {"class": 0, "sample_id": "c0-b", "im_path": "c0-b.jpg"},
        {"class": 1, "sample_id": "c1-a", "im_path": "c1-a.jpg"},
        {"class": 1, "sample_id": "c1-b", "im_path": "c1-b.jpg"},
        {"class": 2, "sample_id": "c2-a", "im_path": "c2-a.jpg"},
        {"class": 2, "sample_id": "c2-b", "im_path": "c2-b.jpg"},
    ]


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
    prediction_fields = {
        "seen_unseen_logit_margin_mean",
        "seen_probability_mass_mean",
        "wrong_domain_prediction_rate",
        "true_class_margin_mean",
        "true_class_rank_mean",
        "entropy_mean",
        "confidence_incorrect",
    }
    assert set(prediction_health_metrics(seen_scores, seen_targets, [0, 1, 2, 3], [0, 1])) == prediction_fields
    assert set(prediction_health_metrics(unseen_scores, unseen_targets, [0, 1, 2, 3], [0, 1])) == prediction_fields
    class_error = class_error_metrics(
        seen_scores,
        seen_targets,
        [0, 1, 2, 3],
        class_names=SyntheticDataset.all_classnames,
        class_attributes=semantic,
    )
    assert set(class_error["summary"]) == {"bottom_k_class_mean", "max_prediction_share"}
    assert set(class_error["arrays"]) == {
        "candidate_global_ids",
        "per_class_accuracy",
        "class_support",
        "class_true_margin",
        "predicted_class_frequency",
    }
    assert class_error["arrays"]["per_class_accuracy"].shape == (4,)
    assert len(class_error["top_confusion_pairs"]) <= 10
    for pair in class_error["top_confusion_pairs"]:
        assert "true_local_id" not in pair and "pred_local_id" not in pair
    calibration_profile = calibration_profile_metrics(
        seen_scores, seen_targets, unseen_scores, unseen_targets, [0, 1, 2, 3], [0, 1], [-1.0, 0.0, 1.0]
    )
    assert set(calibration_profile["summary"]) == {
        "ausuc",
        "raw_to_oracle_gain",
        "oracle_peak_gamma",
    }
    assert set(calibration_profile) == {
        "summary",
        "gamma_grid",
        "seen_at_gamma",
        "unseen_at_gamma",
    }
    assert calibration_profile["summary"]["raw_to_oracle_gain"] >= 0.0
    assert representation_geometry_metrics(visual_seen, seen_targets)
    alignment_fields = {
        "true_prototype_similarity",
        "hard_negative_similarity",
        "semantic_margin",
        "true_prototype_rank",
        "prototype_recall_at_k",
        "class_center_prototype_cosine",
        "visual_semantic_structure_spearman",
        "neighbor_preservation_at_k",
        "visual_interclass_distance_mean",
        "visual_interclass_distance_std",
        "semantic_interclass_distance_mean",
        "semantic_interclass_distance_std",
        "visual_semantic_distance_spearman",
        "semantic_ambiguity_rate",
    }
    assert set(visual_semantic_alignment_metrics(visual_seen, semantic, seen_targets)) == alignment_fields
    neighbor_visual = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    neighbor_semantic = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    neighbor_alignment = visual_semantic_alignment_metrics(
        neighbor_visual,
        neighbor_semantic,
        np.arange(4, dtype=np.int64),
        recall_k=1,
    )
    assert neighbor_alignment["neighbor_preservation_at_k"] < 1.0
    expected_visual_distances = np.linalg.norm(
        neighbor_visual[:, None, :] - neighbor_visual[None, :, :], axis=-1
    )[np.triu_indices(4, k=1)]
    assert np.isclose(
        neighbor_alignment["visual_interclass_distance_mean"],
        expected_visual_distances.mean(),
    )
    assert np.isfinite(neighbor_alignment["visual_semantic_distance_spearman"])
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
    full_probe_manifest = build_probe_manifest(
        SyntheticProbeDataset(),
        split="synthetic",
        per_class=1,
        max_samples=3,
        selection_seed=17,
        candidate_class_ids=[0, 1, 2, 3],
    )
    assert full_probe_manifest["candidate_class_ids_absent_from_split"] == [3]
    assert full_probe_manifest["available_probe_class_count"] == 3
    assert full_probe_manifest["selected_class_count"] == 3
    assert full_probe_manifest["class_coverage_ratio_of_available"] == 1.0
    assert full_probe_manifest["per_class_quota_satisfied"]
    capped_probe_manifest = build_probe_manifest(
        SyntheticProbeDataset(),
        split="synthetic",
        per_class=2,
        max_samples=4,
        selection_seed=17,
        candidate_class_ids=[0, 1, 2, 3],
    )
    assert capped_probe_manifest["pre_cap_sample_count"] == 6
    assert capped_probe_manifest["selected_sample_count"] == 4
    assert capped_probe_manifest["max_samples_truncated"]
    assert not capped_probe_manifest["per_class_quota_satisfied"]

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
        running_summary = json.loads(
            (Path(temp_dir) / "monitor_runtime_summary.json").read_text(encoding="utf-8")
        )
        assert running_summary["status"] == "running"
        assert running_summary["finalized_at"] is None
        assert running_summary["seed"] == cfg.SEED
        assert "sampling" in running_summary
        assert not (Path(temp_dir) / "monitor_manifest.json").exists()
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
        manager.record_epoch(
            "train",
            "train_epoch",
            {
                "loss": 1.0,
                "lr": 0.01,
                "batch_time_sec": 0.2,
                "data_time_sec": 0.05,
            },
            reducer={
                "loss": "sample_mean",
                "lr": "last",
                "batch_time_sec": "mean",
                "data_time_sec": "mean",
            },
            n=4,
        )
        diagnostics.finalize(status="completed")
        manager.finalize(status="completed")
        required = (
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
        with (Path(temp_dir) / "metrics_epoch.csv").open("r", encoding="utf-8", newline="") as handle:
            epoch_rows = list(csv.DictReader(handle))
        train_epoch_reducers = {
            row["metric"]: row["reducer"]
            for row in epoch_rows
            if row["namespace"] == "train_epoch"
        }
        assert train_epoch_reducers == {
            "loss": "sample_mean",
            "lr": "last",
            "batch_time_sec": "mean",
            "data_time_sec": "mean",
        }

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
