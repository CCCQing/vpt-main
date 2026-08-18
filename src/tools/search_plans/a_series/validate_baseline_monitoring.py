#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.configs.config import get_cfg
from src.models.classifiers import RSimilarityClassifier
from src.models.vit_backbones.vit import Attention, Block, Encoder
from src.models.vit_prompt.vit import PromptedTransformer
from src.engine.trainer import Trainer
from src.monitoring.comparability import build_comparability_identity
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
from src.monitoring.adapters import loss_component_metrics
from src.monitoring.epoch_transition import EpochPredictionTransitionTracker
from src.monitoring.module_effect import (
    PairedModuleEffectAccumulator,
    attribute_concept_prompt_patch_block_intervention,
    patch_prompt_uniform_intervention,
    paired_module_effect_metrics,
    prompt_patch_uniform_intervention,
    prompt_read_block_intervention,
    prompt_write_block_intervention,
    random_prompt_patch_block_intervention,
    transport_prompt_patch_block_intervention,
    transport_random_patch_block_intervention,
)
from src.monitoring.prediction_transition import (
    PredictionTransitionAccumulator,
    correctness_pattern_counts,
)
from src.monitoring.probe import (
    ProbeAttentionAffinityAccumulator,
    StreamingFixedProbeAccumulator,
    StreamingTokenViewAccumulator,
    TargetRelevanceAccumulator,
    _prompt_patch_retrieval_values,
    affinity_health_metrics,
    attention_flow_metrics,
    build_probe_manifest,
    validate_probe_manifest,
)
from src.tools.search_plans.a_series.summarize_baseline_monitoring import (
    _generalization_trajectory_outputs,
    _gate_report,
    _is_gate_representation_metric,
    _method_summaries,
    _paired_summaries,
    _probe_robustness_summaries,
    _summarize_generalization_trajectory,
    _summary,
    _stage,
    load_run,
)
from src.tools.search_plans.a_series.summarize_cross_experiment_robustness import (
    _condition_comparison,
    _paired_seed_delta,
)
from src.tools.search_plans.a_series.probe_evidence import (
    MECHANISM_EVIDENCE,
    PROBE_CONTEXT_ONLY,
    VALIDITY_OR_IDENTITY,
    probe_record_evidence_role,
)
from src.tools.search_plans.a_series.artifact_io import (
    compact_to_gzip,
    open_text_artifact,
    read_json_artifact,
)


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


class SyntheticProbeModel(torch.nn.Module):
    def __init__(self, semantic, cfg):
        super().__init__()
        self.prompt_embeddings = torch.nn.Parameter(torch.full((semantic.shape[1],), 0.1))
        attention_config = SimpleNamespace(
            hidden_size=int(semantic.shape[1]),
            transformer={
                "num_heads": 2,
                "attention_dropout_rate": 0.0,
            },
        )
        self.route_attentions = torch.nn.ModuleList(
            [Attention(attention_config, vis=True) for _ in range(4)]
        )
        self._prompt_state_intervention = None
        self._runtime_token_sequence = None
        self.r_similarity_head = RSimilarityClassifier(semantic, hidden_size=semantic.shape[1], cfg=cfg)
        with torch.no_grad():
            self.r_similarity_head.prototype_proj.weight.copy_(torch.eye(semantic.shape[1]))
            self.r_similarity_head.prototype_proj.bias.zero_()

    def forward(self, inputs, semantics=None, class_ids=None, prototype_class_ids=None, runtime_targets=None):
        intervention = next(
            (
                module._prompt_path_intervention
                for module in self.route_attentions
                if module._prompt_path_intervention
            ),
            {},
        )
        intervention_shift = {
            "prompt_read_block": -0.02,
            "prompt_write_block": -0.03,
            "prompt_patch_uniform": -0.01,
            "patch_prompt_uniform": -0.015,
            "prompt_patch_value_globalize": -0.025,
        }.get(str(intervention.get("mode", "")), 0.0)
        state_shift = 0.0
        state_intervention = self._prompt_state_intervention or {}
        if str(state_intervention.get("mode", "")) == "prompt_context_swap":
            permutation = torch.as_tensor(
                state_intervention.get("permutation"),
                dtype=torch.long,
                device=inputs.device,
            ).view(-1)
            if permutation.numel() != inputs.shape[0]:
                raise ValueError("Synthetic Prompt swap permutation has wrong batch size")
            state_shift = 0.05 * (inputs.index_select(0, permutation) - inputs)
        return self.r_similarity_head(
            inputs
            + self.prompt_embeddings.unsqueeze(0)
            + intervention_shift
            + state_shift,
            class_ids=class_ids,
            prototype_class_ids=prototype_class_ids,
            runtime_targets=runtime_targets,
        )

    def get_runtime_classifier_stats(self):
        head = self.r_similarity_head
        return {
            "visual_input": head._loss_last_visual_input,
            "visual_repr": head._loss_last_visual_repr,
            "semantic_input": head._loss_last_semantic_input,
            "semantic_repr": head._loss_last_semantic_repr,
        }

    def forward_with_affinity(
        self,
        inputs,
        affinity_cfg,
        semantics=None,
        vis=False,
        class_ids=None,
        runtime_targets=None,
    ):
        logits = self.forward(
            inputs,
            semantics=semantics,
            class_ids=class_ids,
            runtime_targets=runtime_targets,
        )
        prompt_length = int(affinity_cfg.get("prompt_length", 0))
        patch_length = 4
        sequence_length = 1 + prompt_length + patch_length
        prompt_strength = self.prompt_embeddings.mean()
        attention_layers = []
        affinity_layers = []
        prompt_state = (
            self.prompt_embeddings.view(1, 1, -1).expand(
                inputs.shape[0], prompt_length, -1
            )
            + 0.05 * inputs.unsqueeze(1)
        )
        previous_prompt_output = None
        for layer_index in range(4):
            route_attention = self.route_attentions[layer_index]
            scores = torch.zeros(
                inputs.shape[0], 2, sequence_length, sequence_length,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if prompt_length > 0:
                scores[:, :, 0, 1:1 + prompt_length] = prompt_strength * (layer_index + 1)
            prompt_slice = slice(1, 1 + prompt_length)
            patch_slice = slice(1 + prompt_length, sequence_length)
            if prompt_length > 0:
                patch_scores = torch.arange(
                    patch_length,
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
                scores[:, :, prompt_slice, patch_slice] = (
                    patch_scores * prompt_strength * (layer_index + 1)
                )
                patch_ids = torch.arange(
                    patch_length,
                    device=inputs.device,
                ).view(patch_length, 1)
                prompt_ids = torch.arange(
                    prompt_length,
                    device=inputs.device,
                ).view(1, prompt_length)
                patch_prompt_scores = (
                    (patch_ids.remainder(prompt_length) == prompt_ids)
                    .to(dtype=inputs.dtype)
                    * (0.4 * float(layer_index + 1))
                )
                scores[:, :, patch_slice, prompt_slice] = patch_prompt_scores
            attention = torch.softmax(scores, dim=-1)
            attention = route_attention._apply_prompt_path_intervention(
                attention,
                prompt_length,
                0,
            )
            value_layer = torch.arange(
                inputs.shape[0] * 2 * sequence_length * 2,
                dtype=inputs.dtype,
                device=inputs.device,
            ).view(inputs.shape[0], 2, sequence_length, 2)
            value_layer = value_layer / float(max(1, value_layer.numel()))
            context_layer = torch.matmul(attention, value_layer)
            route_attention._apply_prompt_value_intervention(
                context_layer,
                attention,
                value_layer,
                prompt_length,
                0,
            )
            attention_layers.append(attention)
            layer_affinity = {
                "QcKv_raw": scores[:, :, :1, patch_slice],
                "QvKc_raw": scores[:, :, patch_slice, :1],
            }
            if prompt_length > 0:
                prompt_patch_attention = attention[
                    :, :, prompt_slice, patch_slice
                ]
                prompt_patch_context = torch.matmul(
                    prompt_patch_attention,
                    value_layer[:, :, patch_slice, :],
                )
                prompt_patch_content = prompt_patch_context.permute(
                    0, 2, 1, 3
                ).reshape(inputs.shape[0], prompt_length, -1)
                patch_content = (
                    prompt_patch_attention.sum(dim=-1).mean(dim=1)
                    * (prompt_strength.abs() + 0.1)
                )
                layer_affinity.update({
                    "QcKp_raw": scores[:, :, :1, prompt_slice],
                    "QpKc_raw": scores[:, :, prompt_slice, :1],
                    "QpKv_raw": scores[:, :, prompt_slice, patch_slice],
                    "QvKp_raw": scores[:, :, patch_slice, prompt_slice],
                    "AcKp_attn": attention[:, :, :1, prompt_slice],
                    "ApKv_attn": prompt_patch_attention,
                    "_prompt_patch_pre_output_av_magnitude": (
                        prompt_patch_attention.mean(dim=1).abs() + 0.01
                    ),
                    "_prompt_patch_content_vector": prompt_patch_content,
                    "_prompt_patch_pre_output_content_by_head": (
                        prompt_patch_context
                    ),
                    "prompt_patch_value_contribution_norm": patch_content,
                    "prompt_patch_value_contribution_share": (
                        patch_content / (patch_content + 0.5)
                    ),
                    "prompt_patch_value_to_total_cosine": torch.ones_like(
                        patch_content
                    ) * 0.75,
                    "prompt_patch_value_to_prompt_delta_cosine": torch.ones_like(
                        patch_content
                    ) * 0.5,
                    "prompt_patch_similarity_mean": torch.ones(
                        inputs.shape[0], device=inputs.device
                    ) * prompt_strength * (layer_index + 1),
                    "prompt_patch_similarity_within_sample_std": torch.zeros(
                        inputs.shape[0], device=inputs.device
                    ),
                    "cls_attention_prompt_patch_similarity_correlation": torch.ones(
                        inputs.shape[0], device=inputs.device
                    ) * prompt_strength,
                    "cls_high_attention_prompt_patch_similarity": torch.ones(
                        inputs.shape[0], device=inputs.device
                    ) * prompt_strength,
                    "cls_low_attention_prompt_patch_similarity": torch.zeros(
                        inputs.shape[0], device=inputs.device
                    ),
                    "cls_high_minus_low_prompt_patch_similarity": torch.ones(
                        inputs.shape[0], device=inputs.device
                    ) * prompt_strength,
                })
            if isinstance(route_attention._last_prompt_path_intervention_stats, dict):
                layer_affinity.update(
                    route_attention._last_prompt_path_intervention_stats
                )
            prompt_input = prompt_state
            state_intervention = self._prompt_state_intervention or {}
            if (
                str(state_intervention.get("mode", "")) == "prompt_context_swap"
                and int(state_intervention.get("target_layer", -1)) == layer_index
            ):
                permutation = torch.as_tensor(
                    state_intervention.get("permutation"),
                    dtype=torch.long,
                    device=inputs.device,
                ).view(-1)
                before_swap = prompt_input
                prompt_input = prompt_input.index_select(0, permutation)
                fixed_point = permutation.eq(
                    torch.arange(inputs.shape[0], device=inputs.device)
                ).to(dtype=inputs.dtype)
                layer_affinity.update({
                    "prompt_context_swap_applied": torch.ones_like(fixed_point),
                    "prompt_context_swap_fixed_point_ratio": fixed_point,
                    "prompt_context_swap_before_after_cosine": (
                        torch.nn.functional.cosine_similarity(
                            before_swap.mean(dim=1),
                            prompt_input.mean(dim=1),
                            dim=-1,
                            eps=1e-12,
                        )
                    ),
                    "prompt_context_swap_delta_norm": (
                        prompt_input - before_swap
                    ).mean(dim=1).norm(dim=-1),
                })
            read_blocked_here = bool(
                str((route_attention._prompt_path_intervention or {}).get("mode", ""))
                == "prompt_read_block"
                and int(
                    (route_attention._prompt_path_intervention or {}).get(
                        "target_layer", -1
                    )
                )
                == layer_index
            )
            prompt_output = prompt_input + (
                0.0 if read_blocked_here else 0.03 * float(layer_index + 1)
            ) * inputs.unsqueeze(1)
            prompt_input_mean = prompt_input.mean(dim=1)
            prompt_output_mean = prompt_output.mean(dim=1)
            layer_affinity.update({
                "prompt_layer_input_norm": prompt_input_mean.norm(dim=-1),
                "prompt_layer_output_norm": prompt_output_mean.norm(dim=-1),
                "prompt_layer_change_norm": (
                    prompt_output_mean - prompt_input_mean
                ).norm(dim=-1),
                "prompt_layer_input_output_cosine": (
                    torch.nn.functional.cosine_similarity(
                        prompt_input_mean,
                        prompt_output_mean,
                        dim=-1,
                        eps=1e-12,
                    )
                ),
                "_prompt_layer_input_vector": prompt_input_mean,
                "_prompt_layer_output_vector": prompt_output_mean,
            })
            if previous_prompt_output is not None:
                layer_affinity.update({
                    "prompt_previous_output_to_current_input_cosine": (
                        torch.nn.functional.cosine_similarity(
                            previous_prompt_output,
                            prompt_input_mean,
                            dim=-1,
                            eps=1e-12,
                        )
                    ),
                    "prompt_previous_output_to_current_input_gap_norm": (
                        prompt_input_mean - previous_prompt_output
                    ).norm(dim=-1),
                })
            previous_prompt_output = prompt_output_mean
            prompt_state = prompt_output
            affinity_layers.append(layer_affinity)
        patch_scale = (
            0.05
            if str(
                (self.route_attentions[0]._prompt_path_intervention or {}).get(
                    "mode", ""
                )
            ) == "patch_prompt_uniform"
            else 0.25
        )
        patch_tokens = inputs.unsqueeze(1) + patch_scale * torch.eye(
            patch_length,
            dtype=inputs.dtype,
            device=inputs.device,
        ).unsqueeze(0)
        prompt_tokens = prompt_state
        self._runtime_token_sequence = torch.cat(
            (inputs.unsqueeze(1), prompt_tokens, patch_tokens),
            dim=1,
        )
        if attention_layers:
            if prompt_length > 0:
                relevance_signal = sum(
                    attention[:, :, 0, 1:1 + prompt_length].mean(dim=(1, 2))
                    for attention in attention_layers
                )
            else:
                relevance_signal = sum(
                    attention[:, :, 0, 1:].mean(dim=(1, 2))
                    for attention in attention_layers
                )
            class_weight = torch.arange(
                logits.shape[1], dtype=logits.dtype, device=logits.device
            )
            relevance_adjustment = relevance_signal[:, None] * class_weight[None, :]
            logits = logits + relevance_adjustment - relevance_adjustment.detach()
        if vis:
            return logits, attention_layers, affinity_layers
        return logits, affinity_layers

    def get_runtime_token_sequence(self):
        return self._runtime_token_sequence

    def clear_runtime_state(self):
        self._runtime_token_sequence = None


def _merge_probe_batches(attention_batches, affinity_batches, selected_layers):
    attention_layers = []
    affinity_layers = []
    for layer_index in selected_layers:
        attention_layers.append(torch.cat(
            [batch[layer_index] for batch in attention_batches],
            dim=0,
        ))
        keys = sorted(set().union(*[
            set(batch[layer_index]) for batch in affinity_batches
        ]))
        affinity_layers.append({
            key: torch.cat(
                [batch[layer_index][key] for batch in affinity_batches],
                dim=0,
            )
            for key in keys
        })
    return attention_layers, affinity_layers


def _assert_metric_maps_close(expected, actual, label):
    assert set(actual) == set(expected), (
        label,
        sorted(set(expected) - set(actual)),
        sorted(set(actual) - set(expected)),
    )
    for name, expected_value in expected.items():
        assert np.isclose(
            float(actual[name]),
            float(expected_value),
            rtol=2e-5,
            atol=2e-6,
        ), (label, name, expected_value, actual[name])


def _validate_prompt_patch_retrieval_diversity():
    specialized = torch.tensor(
        [[[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    repeated = torch.tensor(
        [[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    specialized_metrics = _prompt_patch_retrieval_values(specialized)
    repeated_metrics = _prompt_patch_retrieval_values(repeated)
    assert specialized_metrics["top1_unique_patch_count"].item() == 2.0
    assert repeated_metrics["top1_unique_patch_count"].item() == 1.0
    assert specialized_metrics["top1_patch_coverage_ratio"].item() == 0.5
    assert repeated_metrics["top1_patch_coverage_ratio"].item() == 0.25
    assert specialized_metrics["selection_overlap"].item() == 0.0
    assert repeated_metrics["selection_overlap"].item() == 1.0
    assert specialized_metrics["coordination_information"].item() > (
        repeated_metrics["coordination_information"].item()
    )
    assert specialized_metrics["spatial_centroid_dispersion"].item() > 0.0


def _validate_streaming_probe_metrics():
    torch.manual_seed(17)
    batch_sizes = [3, 2]
    layer_count = 4
    selected_layers = [0, 2, 3]
    prompt_length = 2
    semantic_length = 1
    temperature = 0.7
    saturation_threshold = 1.25
    attention_batches = []
    affinity_batches = []
    prediction_batches = [[0, 1, 1], [2, 0]]
    target_batches = [[0, 0, 1], [3, 0]]
    accumulator = ProbeAttentionAffinityAccumulator(
        prompt_length=prompt_length,
        semantic_length=semantic_length,
        selected_layers=selected_layers,
        temperature=temperature,
        saturation_threshold=saturation_threshold,
    )
    for batch_size, predictions, targets in zip(
        batch_sizes,
        prediction_batches,
        target_batches,
    ):
        attention_layers = [
            torch.softmax(torch.randn(batch_size, 3, 8, 8), dim=-1)
            for _ in range(layer_count)
        ]
        affinity_layers = []
        for _ in range(layer_count):
            affinity_layers.append({
                "QcKp_raw": torch.randn(batch_size, 3, 1, 2),
                "QpKc_raw": torch.randn(batch_size, 3, 2, 1),
                "QcKv_raw": torch.randn(batch_size, 3, 1, 4),
                "QvKc_raw": torch.randn(batch_size, 3, 4, 1),
                "QcKs_raw": torch.randn(batch_size, 3, 1, 1),
                "QsKc_raw": torch.randn(batch_size, 3, 1, 1),
                "QpKv_raw": torch.randn(batch_size, 3, 2, 4),
                "QvKp_raw": torch.randn(batch_size, 3, 4, 2),
                "QpKs_raw": torch.randn(batch_size, 3, 2, 1),
                "QsKp_raw": torch.randn(batch_size, 3, 1, 2),
                "QvKs_raw": torch.randn(batch_size, 3, 4, 1),
                "QsKv_raw": torch.randn(batch_size, 3, 1, 4),
                "AcKv_attn": torch.softmax(torch.randn(batch_size, 3, 1, 4), dim=-1),
                "AcKp_attn": torch.softmax(torch.randn(batch_size, 3, 1, 2), dim=-1),
                "ApKv_attn": torch.softmax(torch.randn(batch_size, 3, 2, 4), dim=-1),
                "AvKp_attn": torch.softmax(torch.randn(batch_size, 3, 4, 2), dim=-1),
                "ApKc_attn": torch.softmax(torch.randn(batch_size, 3, 2, 1), dim=-1),
            })
        attention_batches.append(attention_layers)
        affinity_batches.append(affinity_layers)
        accumulator.update(
            attention_layers,
            affinity_layers,
            predictions=predictions,
            targets=targets,
        )

    attention_layers, affinity_layers = _merge_probe_batches(
        attention_batches,
        affinity_batches,
        selected_layers,
    )
    predictions = np.concatenate([
        np.asarray(batch, dtype=np.int64) for batch in prediction_batches
    ])
    targets = np.concatenate([
        np.asarray(batch, dtype=np.int64) for batch in target_batches
    ])
    expected_attention = attention_flow_metrics(
        attention_layers,
        prompt_length=prompt_length,
        semantic_length=semantic_length,
        affinity_layers=affinity_layers,
        predictions=predictions,
        targets=targets,
    )
    expected_affinity = affinity_health_metrics(
        affinity_layers,
        temperature=temperature,
        saturation_threshold=saturation_threshold,
    )
    streamed = accumulator.finalize()
    expected_cross_type_relations = {
        "QcKp", "QpKc", "QcKv", "QvKc", "QcKs", "QsKc",
        "QpKv", "QvKp", "QpKs", "QsKp", "QvKs", "QsKv",
    }
    assert all(
        f"{relation}.raw_mean" in streamed["affinity_health"]
        for relation in expected_cross_type_relations
    )
    assert set(streamed["affinity_health_by_layer"]) == set(selected_layers)
    assert all(
        f"{relation}.raw_mean" in streamed["affinity_health_by_layer"][layer_index]
        for layer_index in selected_layers
        for relation in expected_cross_type_relations
    )
    _assert_metric_maps_close(
        expected_attention,
        streamed["attention_flow"],
        "attention",
    )
    _assert_metric_maps_close(
        expected_affinity,
        streamed["affinity_health"],
        "affinity",
    )
    assert set(streamed["attention_flow_by_layer"]) == set(selected_layers)
    assert 0.0 <= streamed["attention_flow"]["patch_to_patch_attention_distance"] <= 1.0
    for layer_index in selected_layers:
        expected_layer = attention_flow_metrics(
            [torch.cat([batch[layer_index] for batch in attention_batches], dim=0)],
            prompt_length=prompt_length,
            semantic_length=semantic_length,
        )
        _assert_metric_maps_close(
            expected_layer,
            streamed["attention_flow_by_layer"][layer_index],
            f"attention_layer_{layer_index}",
        )
        merged_affinity_layer = {
            key: torch.cat(
                [batch[layer_index][key] for batch in affinity_batches],
                dim=0,
            )
            for key in affinity_batches[0][layer_index]
        }
        expected_affinity_layer = {
            name: value
            for name, value in affinity_health_metrics(
                [merged_affinity_layer],
                temperature=temperature,
                saturation_threshold=saturation_threshold,
            ).items()
            if name != "softmax_temperature" and not name.endswith(".layer_diversity")
        }
        _assert_metric_maps_close(
            expected_affinity_layer,
            streamed["affinity_health_by_layer"][layer_index],
            f"affinity_layer_{layer_index}",
        )

    fallback_accumulator = ProbeAttentionAffinityAccumulator(
        prompt_length=prompt_length,
        semantic_length=semantic_length,
        selected_layers=selected_layers,
        temperature=temperature,
        saturation_threshold=saturation_threshold,
    )
    for affinity_batch, predictions_batch, targets_batch in zip(
        affinity_batches,
        prediction_batches,
        target_batches,
    ):
        fallback_accumulator.update(
            [],
            affinity_batch,
            predictions=predictions_batch,
            targets=targets_batch,
        )
    expected_fallback = attention_flow_metrics(
        [],
        prompt_length=prompt_length,
        semantic_length=semantic_length,
        affinity_layers=affinity_layers,
        predictions=predictions,
        targets=targets,
    )
    _assert_metric_maps_close(
        expected_fallback,
        fallback_accumulator.finalize()["attention_flow"],
        "attention_fallback",
    )
    assert set(fallback_accumulator.finalize()["attention_flow_by_layer"]) == set(selected_layers)


def _validate_target_relevance_accumulator():
    torch.manual_seed(31)
    scores = torch.randn(3, 2, 7, 7, requires_grad=True)
    attention = torch.softmax(scores, dim=-1)
    objective = (
        2.0 * attention[:, :, 0, 1:3].sum()
        - attention[:, :, 0, 3:].sum()
    )
    gradient = torch.autograd.grad(objective, attention)[0]
    accumulator = TargetRelevanceAccumulator(
        prompt_length=2,
        semantic_length=0,
        selected_layers=[0],
    )
    correct = torch.tensor([True, False, True])
    accumulator.update_target(torch.tensor([1.0, -0.5, 0.25]), correct)
    assert accumulator.update_layer(0, attention, gradient, correct)
    result = accumulator.finalize_by_layer()[0]
    assert result["cls_to_prompt"]["all.positive_sum"] > 0.0
    assert result["cls_to_patch"]["all.negative_abs_sum"] > 0.0
    assert result["cls_to_prompt"]["correct.positive_sum"] > 0.0
    assert result["cls_to_prompt/prompt_0"]["all.positive_sum"] > 0.0
    assert result["cls_to_prompt/prompt_1"]["all.positive_sum"] > 0.0
    assert "prompt_to_patch/prompt_0" in result
    assert "patch_to_prompt/prompt_1" in result
    target = accumulator.finalize_target()
    assert target["all.sample_count"] == 3.0
    assert np.isclose(target["correct_rate"], 2.0 / 3.0)
    predicted = TargetRelevanceAccumulator(
        prompt_length=2,
        semantic_length=0,
        selected_layers=[0],
        margin_metric_name="predicted_margin_mean",
    )
    predicted.update_target(torch.tensor([0.5]), torch.tensor([False]))
    assert predicted.finalize_target()["all.predicted_margin_mean"] == 0.5


def _validate_prediction_transitions():
    accumulator = PredictionTransitionAccumulator("A0", "A1")
    accumulator.update(
        reference_predictions=[0, 1, 0, 1],
        target_predictions=[0, 0, 1, 1],
        targets=[0, 0, 0, 1],
    )
    result = accumulator.finalize()
    assert result["both_correct_count"] == 2
    assert result["corrected_count"] == 1
    assert result["regressed_count"] == 1
    assert result["net_correction_count"] == 0
    patterns = correctness_pattern_counts(
        {"A0": [0, 1, 0, 1], "A1": [0, 0, 1, 1]},
        [0, 0, 0, 1],
    )
    assert patterns["sample_count"] == 4
    assert patterns["counts"]["A0=0|A1=1"] == 1


def _validate_prompt_content_and_layer_mechanism():
    tiny_config = SimpleNamespace(
        hidden_size=4,
        transformer={
            "num_heads": 2,
            "attention_dropout_rate": 0.0,
            "mlp_dim": 8,
            "dropout_rate": 0.0,
        },
    )
    attention = Attention(tiny_config, vis=False)
    hidden = torch.randn(2, 4, 4)
    query, key, value = attention._project_qkv(hidden)
    monitors = attention.compute_prompt_visual_monitors(
        query,
        key,
        prompt_length=1,
        semantic_length=0,
        value_layer=value,
        detach=True,
    )
    assert monitors["cls_prompt_value_contribution_norm"].shape == (2,)
    assert monitors["_cls_prompt_value_contribution_vector"].shape == (2, 4)
    assert bool((monitors["cls_prompt_value_contribution_share"] >= 0.0).all())
    assert bool((monitors["cls_prompt_value_contribution_share"] <= 1.0).all())
    assert monitors["prompt_patch_value_contribution_norm"].shape == (2, 1)
    assert monitors["_prompt_patch_value_contribution_vector"].shape == (2, 1, 4)
    assert bool((monitors["prompt_patch_value_contribution_share"] >= 0.0).all())
    assert bool((monitors["prompt_patch_value_contribution_share"] <= 1.0).all())

    semantic_monitors = attention.compute_prompt_visual_monitors(
        query,
        key,
        prompt_length=1,
        semantic_length=1,
        value_layer=value,
        detach=True,
    )
    expected_cross_type_relations = {
        "QcKp_raw", "QpKc_raw",
        "QcKv_raw", "QvKc_raw",
        "QcKs_raw", "QsKc_raw",
        "QpKv_raw", "QvKp_raw",
        "QpKs_raw", "QsKp_raw",
        "QvKs_raw", "QsKv_raw",
    }
    assert expected_cross_type_relations.issubset(semantic_monitors)
    assert semantic_monitors["QcKv_raw"].shape == (2, 2, 1, 1)
    assert semantic_monitors["QvKc_raw"].shape == (2, 2, 1, 1)
    assert semantic_monitors["QpKc_raw"].shape == (2, 2, 1, 1)
    assert semantic_monitors["QsKc_raw"].shape == (2, 2, 1, 1)

    block = Block(tiny_config, vis=False).eval()
    block_output, _, block_affinity, _ = block.forward_with_affinity(
        hidden,
        {
            "prompt_length": 1,
            "semantic_length": 0,
            "detach": True,
            "block_s_to_cls": False,
        },
        num_prompt_tokens=1,
    )
    assert block_output.shape == hidden.shape
    assert "cls_prompt_value_to_cls_delta_cosine" in block_affinity
    assert "prompt_patch_value_to_prompt_delta_cosine" in block_affinity
    assert "prompt_patch_similarity_mean" in block_affinity
    assert "cls_attention_prompt_patch_similarity_correlation" in block_affinity
    assert "_target_relevance_attention" not in block_affinity

    relevance_output, _, relevance_affinity, _ = block.forward_with_affinity(
        hidden,
        {
            "prompt_length": 1,
            "semantic_length": 0,
            "detach": True,
            "block_s_to_cls": False,
            "retain_attention_for_relevance": True,
        },
        num_prompt_tokens=1,
    )
    retained_attention = relevance_affinity.pop(
        "_target_relevance_attention"
    )
    assert retained_attention.requires_grad
    relevance_gradient = torch.autograd.grad(
        relevance_output.square().sum(),
        retained_attention,
        allow_unused=True,
    )[0]
    assert relevance_gradient is not None
    assert bool(torch.isfinite(relevance_gradient).all())
    continuity_layers = [
        block_affinity,
        {
            "_prompt_layer_input_vector": block_affinity["_prompt_layer_output_vector"].clone(),
            "_prompt_layer_output_vector": block_affinity["_prompt_layer_output_vector"].clone(),
        },
    ]
    PromptedTransformer._attach_prompt_layer_continuity(continuity_layers)
    assert torch.allclose(
        continuity_layers[1]["prompt_previous_output_to_current_input_gap_norm"],
        torch.zeros(hidden.shape[0]),
    )

    accumulator = ProbeAttentionAffinityAccumulator(
        prompt_length=1,
        semantic_length=0,
        selected_layers=[0, 1],
        class_count=2,
        raw_semantic_reference=torch.eye(2),
    )
    attention_layers = [
        torch.softmax(torch.randn(2, 2, 4, 4), dim=-1),
        torch.softmax(torch.randn(2, 2, 4, 4), dim=-1),
    ]
    affinity_layers = [
        {
            "cls_prompt_value_contribution_norm": torch.tensor([1.0, 3.0]),
            "cls_patch_value_contribution_norm": torch.tensor([3.0, 5.0]),
            "cls_prompt_value_contribution_share": torch.tensor([0.25, 0.375]),
            "cls_prompt_value_to_total_cosine": torch.tensor([0.8, 0.6]),
            "cls_prompt_value_to_patch_cosine": torch.tensor([0.4, -0.2]),
            "cls_prompt_value_to_cls_delta_cosine": torch.tensor([0.5, 0.7]),
            "AcKp_attn": torch.tensor([[[[0.4]]], [[[0.6]]]]),
            "prompt_patch_value_contribution_norm": torch.tensor([[1.0], [2.0]]),
            "prompt_patch_value_contribution_share": torch.tensor([[0.4], [0.5]]),
            "prompt_patch_value_to_total_cosine": torch.tensor([[0.7], [0.8]]),
            "prompt_patch_value_to_prompt_delta_cosine": torch.tensor([[0.5], [0.6]]),
            "prompt_layer_input_norm": torch.tensor([1.0, 3.0]),
            "prompt_layer_output_norm": torch.tensor([2.0, 4.0]),
            "prompt_layer_change_norm": torch.tensor([1.0, 1.0]),
            "prompt_layer_input_output_cosine": torch.tensor([1.0, 1.0]),
            "_prompt_layer_input_vector": torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
            "_prompt_layer_output_vector": torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
        },
        {
            "cls_prompt_value_contribution_norm": torch.tensor([2.0, 4.0]),
            "cls_patch_value_contribution_norm": torch.tensor([4.0, 6.0]),
            "cls_prompt_value_contribution_share": torch.tensor([1.0 / 3.0, 0.4]),
            "cls_prompt_value_to_total_cosine": torch.tensor([0.7, 0.5]),
            "cls_prompt_value_to_patch_cosine": torch.tensor([0.2, -0.4]),
            "cls_prompt_value_to_cls_delta_cosine": torch.tensor([0.4, 0.6]),
            "AcKp_attn": torch.tensor([[[[0.5]]], [[[0.7]]]]),
            "prompt_patch_value_contribution_norm": torch.tensor([[1.5], [2.5]]),
            "prompt_patch_value_contribution_share": torch.tensor([[0.45], [0.55]]),
            "prompt_patch_value_to_total_cosine": torch.tensor([[0.6], [0.7]]),
            "prompt_patch_value_to_prompt_delta_cosine": torch.tensor([[0.4], [0.5]]),
            "prompt_layer_input_norm": torch.tensor([2.0, 4.0]),
            "prompt_layer_output_norm": torch.tensor([3.0, 5.0]),
            "prompt_layer_change_norm": torch.tensor([1.0, 1.0]),
            "prompt_layer_input_output_cosine": torch.tensor([1.0, 1.0]),
            "prompt_previous_output_to_current_input_cosine": torch.tensor([1.0, 1.0]),
            "prompt_previous_output_to_current_input_gap_norm": torch.tensor([0.0, 0.0]),
            "_prompt_layer_input_vector": torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
            "_prompt_layer_output_vector": torch.tensor([[3.0, 0.0], [5.0, 0.0]]),
        },
    ]
    accumulator.update(
        attention_layers,
        affinity_layers,
        predictions=[0, 1],
        targets=[0, 1],
        projected_semantic_reference=torch.eye(2),
    )
    result = accumulator.finalize()
    assert np.isclose(result["attention_flow"]["cls_prompt_value_contribution_norm"], 2.5)
    assert result["attention_flow"]["configured_prompt_count"] == 1.0
    assert 0.0 <= result["attention_flow"]["prompt_to_patch_conditional_entropy"] <= 1.0
    assert 0.0 <= result["attention_flow"]["prompt_to_patch_effective_patch_ratio"] <= 1.0
    assert 0.0 <= result["attention_flow"]["patch_to_prompt_conditional_entropy"] <= 1.0
    assert np.isclose(
        result["attention_flow"]["patch_to_prompt_effective_prompt_count"],
        1.0,
    )
    assert np.isclose(
        result["attention_flow"]["patch_prompt_top1_coverage_ratio"],
        1.0,
    )
    assert np.isclose(
        result["attention_flow"]["cls_to_prompt_effective_prompt_count"],
        1.0,
    )
    assert np.isclose(
        result["attention_flow"]["cls_to_prompt_effective_prompt_ratio"],
        1.0,
    )
    assert np.isclose(
        result["attention_flow_by_layer"][0]["cls_prompt_value_to_cls_delta_cosine"],
        0.6,
    )
    mechanism = result["prompt_layer_mechanism_by_layer"]
    assert set(mechanism) == {0, 1}
    assert np.isclose(mechanism[0]["prompt_layer_input_instance_variance"], 1.0)
    assert np.isclose(
        mechanism[1]["prompt_previous_output_to_current_input_gap_norm"], 0.0
    )
    assert not any(name.startswith("_") for values in mechanism.values() for name in values)
    assert set(result["prompt_semantic_role_by_layer"]) == {0, 1}
    assert set(result["prompt_semantic_role_by_layer_and_prompt"][0]) == {0}
    assert (
        result["prompt_semantic_role_by_layer_and_prompt"][0][0][
            "patch_collection_effective_class_count"
        ]
        >= 1.0
    )
    assert "patch_collection_projected_semantic_graph_spearman" in (
        result["prompt_semantic_role_by_layer"][0]
    )


def _validate_affinity_diagnostic_cpu_offload():
    tiny_config = SimpleNamespace(
        hidden_size=4,
        transformer={
            "num_layers": 2,
            "num_heads": 2,
            "attention_dropout_rate": 0.0,
            "mlp_dim": 8,
            "dropout_rate": 0.0,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(tiny_config, vis=True).to(device).eval()
    hidden = torch.randn(2, 4, 4, device=device)
    base_cfg = {
        "prompt_length": 1,
        "semantic_length": 0,
        "detach": True,
        "block_s_to_cls": False,
    }
    relevance_encoder = Encoder(tiny_config, vis=False).to(device).eval()
    relevance_hidden = hidden.clone().requires_grad_(True)
    relevance_encoded, relevance_visual_attention, relevance_affinity = (
        relevance_encoder.forward_with_affinity(
            relevance_hidden,
            {**base_cfg, "retain_attention_for_relevance": True},
            num_prompt_tokens=1,
        )
    )
    assert relevance_visual_attention == []
    retained_attention = [
        layer.pop("_target_relevance_attention")
        for layer in relevance_affinity
    ]
    assert len(retained_attention) == tiny_config.transformer["num_layers"]
    relevance_gradients = torch.autograd.grad(
        relevance_encoded[..., 0].sum(),
        retained_attention,
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in relevance_gradients)
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in relevance_gradients
    )
    with torch.no_grad():
        expected_encoded, expected_attention, expected_affinity = (
            encoder.forward_with_affinity(
                hidden.clone(),
                dict(base_cfg),
                num_prompt_tokens=1,
            )
        )
        PromptedTransformer._attach_prompt_layer_continuity(
            expected_affinity
        )
        actual_encoded, actual_attention, actual_affinity = (
            encoder.forward_with_affinity(
                hidden.clone(),
                {
                    **base_cfg,
                    "offload_diagnostics_to_cpu": True,
                },
                num_prompt_tokens=1,
            )
        )

    assert torch.allclose(expected_encoded, actual_encoded, atol=0.0, rtol=0.0)

    def assert_tree_equal(expected, actual):
        if torch.is_tensor(expected):
            assert torch.is_tensor(actual)
            assert actual.device.type == "cpu"
            assert torch.allclose(
                expected.detach().cpu(), actual, atol=0.0, rtol=0.0
            )
            return
        if isinstance(expected, dict):
            assert isinstance(actual, dict)
            assert set(expected) == set(actual)
            for key in expected:
                assert_tree_equal(expected[key], actual[key])
            return
        if isinstance(expected, (list, tuple)):
            assert isinstance(actual, type(expected))
            assert len(expected) == len(actual)
            for left, right in zip(expected, actual):
                assert_tree_equal(left, right)
            return
        assert expected == actual

    assert_tree_equal(expected_attention, actual_attention)
    assert_tree_equal(expected_affinity, actual_affinity)

    with torch.no_grad():
        selected_encoded, _, selected_affinity = encoder.forward_with_affinity(
            hidden.clone(),
            {
                **base_cfg,
                "offload_diagnostics_to_cpu": True,
                "selected_layers": [1],
                "include_visual_normalizations": False,
            },
            num_prompt_tokens=1,
        )
    assert torch.allclose(expected_encoded, selected_encoded, atol=0.0, rtol=0.0)
    assert not any(key.endswith("_raw") for key in selected_affinity[0])
    assert any(key.endswith("_raw") for key in selected_affinity[1])
    assert not any(key.endswith("_vis") for layer in selected_affinity for key in layer)
    assert "prompt_previous_output_to_current_input_cosine" in selected_affinity[1]

    deep_transformer = PromptedTransformer.__new__(PromptedTransformer)
    torch.nn.Module.__init__(deep_transformer)
    deep_transformer.encoder = Encoder(tiny_config, vis=True)
    deep_transformer.vit_config = tiny_config
    deep_transformer.num_tokens = 1
    deep_transformer.prompt_backend = "vpt_deep"
    deep_transformer.deep_prompt_embeddings = torch.nn.Parameter(
        torch.randn(1, 1, 4)
    )
    deep_transformer.prompt_proj = torch.nn.Identity()
    deep_transformer.prompt_dropout = torch.nn.Identity()
    deep_transformer.prompt_init_provider = None
    deep_transformer.semantic_tokens_enable = False
    deep_transformer.attention_mediation_enable = False
    deep_transformer.to(device).eval()
    with torch.no_grad():
        expected_deep = deep_transformer.forward_deep_prompt_with_affinity(
            hidden.clone(), dict(base_cfg)
        )
        PromptedTransformer._attach_prompt_layer_continuity(
            expected_deep[2]
        )
        actual_deep = deep_transformer.forward_deep_prompt_with_affinity(
            hidden.clone(),
            {**base_cfg, "offload_diagnostics_to_cpu": True},
        )
    assert torch.allclose(
        expected_deep[0], actual_deep[0], atol=0.0, rtol=0.0
    )
    assert_tree_equal(expected_deep[1], actual_deep[1])
    assert_tree_equal(expected_deep[2], actual_deep[2])

    try:
        encoder.forward_with_affinity(
            hidden.clone().requires_grad_(True),
            {**base_cfg, "offload_diagnostics_to_cpu": True},
            num_prompt_tokens=1,
        )
    except RuntimeError as exc:
        assert "only valid in a no-grad forward" in str(exc)
    else:
        raise AssertionError(
            "gradient-enabled affinity forward accepted CPU offload"
        )


def _validate_lossless_artifact_compaction():
    with tempfile.TemporaryDirectory(prefix="artifact_compaction_") as temp_dir:
        root = Path(temp_dir)
        csv_path = root / "probe_metrics.csv"
        csv_payload = "metric,value\nmargin,0.5\n" * 32
        csv_path.write_text(csv_payload, encoding="utf-8")
        manifest = compact_to_gzip(csv_path)
        assert manifest["status"] == "compacted"
        assert not csv_path.exists()
        assert Path(manifest["compressed_path"]).is_file()
        with open_text_artifact(csv_path, "r") as handle:
            assert handle.read() == csv_payload

        json_path = root / "module_effect.json"
        json_payload = {"status": "valid", "values": list(range(64))}
        json_path.write_text(json.dumps(json_payload), encoding="utf-8")
        compact_to_gzip(json_path)
        assert read_json_artifact(json_path) == json_payload


def _validate_prompt_role_metric_rows():
    trainer = Trainer.__new__(Trainer)
    trainer.monitor_manager = SimpleNamespace(
        run_id="role-run",
        session_id="role-session",
    )
    rows = trainer._prompt_role_metric_rows(
        {
            "patch_collection_effective_class_count": 2.0,
            "patch_collection_top1_class_local_id": 3.0,
            "patch_collection_top1_class_share": 0.7,
            "cls_consumption_top1_class_local_id": 1.0,
            "cls_consumption_top1_class_share": 0.6,
        },
        checkpoint_id="checkpoint",
        probe_id="probe",
        split="probe_test_unseen",
        layer_index=2,
        prompt_index=1,
        selection_seed=17,
        probe_manifest_sha256="manifest",
    )
    assert not any(row["metric"].endswith("class_local_id") for row in rows)
    class_rows = [
        row for row in rows if row["entity_type"] == "layer_prompt_class"
    ]
    assert {
        row["entity_id"] for row in class_rows
    } == {
        "layer_2/prompt_1/patch_collection/top1/class_local_3",
        "layer_2/prompt_1/cls_consumption/top1/class_local_1",
    }
    assert all(row["metric"] == "class_share" for row in class_rows)
    transport_rows = trainer._patch_semantic_transport_metric_rows(
        {
            "attention_effective_class_count": 2.0,
            "attention_top1_class_local_id": 2.0,
            "attention_top1_class_share": 0.65,
            "av_top1_attribute_id": 7.0,
            "av_top1_attribute_share": 0.55,
        },
        checkpoint_id="checkpoint",
        probe_id="probe",
        split="probe_test_unseen",
        prompt_index=1,
        selection_seed=17,
        probe_manifest_sha256="manifest",
    )
    assert not any(
        row["metric"].endswith("class_local_id")
        or row["metric"].endswith("attribute_id")
        for row in transport_rows
    )
    assert any(
        row["entity_id"]
        == "final_layer/prompt_1/attention/top1/class_local_2"
        and row["metric"] == "class_share"
        for row in transport_rows
    )
    assert any(
        row["entity_id"] == "final_layer/prompt_1/av/top1/attribute_7"
        and row["metric"] == "attribute_share"
        for row in transport_rows
    )


def _validate_prompt_attention_path_interventions():
    tiny_config = SimpleNamespace(
        hidden_size=8,
        transformer={
            "num_heads": 2,
            "attention_dropout_rate": 0.0,
        },
    )
    holder = torch.nn.Module()
    holder.attention = Attention(tiny_config, vis=True).eval()
    hidden = torch.randn(3, 7, 8)
    prompt_slice = slice(1, 3)
    patch_slice = slice(3, 7)

    _, normal_attention, _ = holder.attention(
        hidden,
        prompt_length=2,
    )
    with prompt_read_block_intervention(holder):
        _, read_blocked, _ = holder.attention(hidden, prompt_length=2)
    assert holder.attention._prompt_path_intervention is None
    assert torch.allclose(
        read_blocked[:, :, prompt_slice, patch_slice],
        torch.zeros_like(read_blocked[:, :, prompt_slice, patch_slice]),
    )
    assert torch.allclose(
        read_blocked.sum(dim=-1),
        torch.ones_like(read_blocked.sum(dim=-1)),
        atol=1e-6,
    )

    with prompt_write_block_intervention(holder):
        _, write_blocked, _ = holder.attention(hidden, prompt_length=2)
    non_prompt_rows = torch.cat(
        (
            write_blocked[:, :, :1, prompt_slice],
            write_blocked[:, :, 3:, prompt_slice],
        ),
        dim=-2,
    )
    assert torch.allclose(non_prompt_rows, torch.zeros_like(non_prompt_rows))
    assert torch.allclose(
        write_blocked.sum(dim=-1),
        torch.ones_like(write_blocked.sum(dim=-1)),
        atol=1e-6,
    )

    with prompt_patch_uniform_intervention(holder):
        _, uniform_attention, _ = holder.attention(hidden, prompt_length=2)
    prompt_uniform_stats = holder.attention._last_prompt_path_intervention_stats
    normal_prompt_patch = normal_attention[:, :, prompt_slice, patch_slice]
    uniform_prompt_patch = uniform_attention[:, :, prompt_slice, patch_slice]
    assert torch.allclose(
        normal_prompt_patch.sum(dim=-1),
        uniform_prompt_patch.sum(dim=-1),
        atol=1e-6,
    )
    assert bool((prompt_uniform_stats["prompt_patch_uniform_applied"] == 1.0).all())
    assert float(
        prompt_uniform_stats["prompt_patch_uniform_mass_abs_error"].max().item()
    ) <= 1e-6

    with patch_prompt_uniform_intervention(holder):
        _, patch_uniform_attention, _ = holder.attention(hidden, prompt_length=2)
    patch_uniform_stats = holder.attention._last_prompt_path_intervention_stats
    assert holder.attention._prompt_path_intervention is None
    normal_patch_prompt = normal_attention[:, :, patch_slice, prompt_slice]
    uniform_patch_prompt = patch_uniform_attention[:, :, patch_slice, prompt_slice]
    assert torch.allclose(
        normal_patch_prompt.sum(dim=-1),
        uniform_patch_prompt.sum(dim=-1),
        atol=1e-6,
    )
    assert bool((patch_uniform_stats["patch_prompt_uniform_applied"] == 1.0).all())
    assert float(
        patch_uniform_stats["patch_prompt_uniform_mass_abs_error"].max().item()
    ) <= 1e-6
    assert torch.allclose(
        uniform_patch_prompt,
        uniform_patch_prompt.mean(dim=-1, keepdim=True).expand_as(
            uniform_patch_prompt
        ),
        atol=1e-6,
    )
    expected_patch_uniform = normal_attention.clone()
    expected_patch_uniform[:, :, patch_slice, prompt_slice] = (
        uniform_patch_prompt
    )
    assert torch.allclose(
        expected_patch_uniform,
        patch_uniform_attention,
        atol=1e-6,
    )
    assert torch.allclose(
        uniform_prompt_patch,
        uniform_prompt_patch.mean(dim=-1, keepdim=True).expand_as(
            uniform_prompt_patch
        ),
        atol=1e-6,
    )
    assert torch.allclose(
        normal_attention[:, :, prompt_slice, :3],
        uniform_attention[:, :, prompt_slice, :3],
        atol=1e-6,
    )


def _validate_attribute_concept_grounding_and_intervention():
    tiny_config = SimpleNamespace(
        hidden_size=4,
        transformer={
            "num_heads": 2,
            "attention_dropout_rate": 0.0,
        },
    )
    holder = torch.nn.Module()
    holder.attention = Attention(tiny_config, vis=True).eval()
    hidden = torch.tensor(
        [
            [[0.2, 0.1, 0.0, 0.0], [0.1, 0.2, 0.0, 0.0],
             [0.0, 0.1, 0.2, 0.0], [1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            [[0.1, 0.2, 0.0, 0.0], [0.2, 0.1, 0.0, 0.0],
             [0.1, 0.0, 0.2, 0.0], [1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    directions = torch.eye(4, dtype=torch.float32)
    margin_weights = torch.tensor(
        [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
        dtype=torch.float32,
    )
    _, normal_attention, _ = holder.attention(hidden, prompt_length=2)
    with attribute_concept_prompt_patch_block_intervention(
        holder,
        attribute_directions=directions,
        margin_weights=margin_weights,
        patch_ratio=0.25,
        random_seed=17,
    ):
        _, targeted_attention, _ = holder.attention(hidden, prompt_length=2)
    targeted_stats = holder.attention._last_prompt_path_intervention_stats
    prompt_patch = (slice(1, 3), slice(3, 7))
    assert torch.allclose(
        normal_attention[:, :, prompt_patch[0], prompt_patch[1]].sum(dim=-1),
        targeted_attention[:, :, prompt_patch[0], prompt_patch[1]].sum(dim=-1),
        atol=1e-6,
    )
    assert float(
        targeted_stats[
            "concept_intervention_selected_prompt_attention_mass_after"
        ].abs().max().item()
    ) <= 1e-7
    assert float(
        targeted_stats[
            "concept_intervention_prompt_patch_mass_abs_error"
        ].max().item()
    ) <= 1e-6
    assert float(
        targeted_stats["concept_intervention_selection_score_gap"].min().item()
    ) > 0.0
    with random_prompt_patch_block_intervention(
        holder,
        attribute_directions=directions,
        margin_weights=margin_weights,
        patch_ratio=0.25,
        random_seed=17,
    ):
        _, random_attention, _ = holder.attention(hidden, prompt_length=2)
    random_stats = holder.attention._last_prompt_path_intervention_stats
    assert torch.allclose(
        normal_attention[:, :, prompt_patch[0], prompt_patch[1]].sum(dim=-1),
        random_attention[:, :, prompt_patch[0], prompt_patch[1]].sum(dim=-1),
        atol=1e-6,
    )
    assert float(
        random_stats[
            "concept_intervention_selected_prompt_attention_mass_after"
        ].abs().max().item()
    ) <= 1e-7
    assert float(
        random_stats[
            "concept_intervention_prompt_patch_mass_abs_error"
        ].max().item()
    ) <= 1e-6
    assert float(
        random_stats["concept_intervention_targeted"].abs().max().item()
    ) == 0.0

    batch_size = 2
    prompt_length = 2
    attribute_count = 4
    affinity_layers = []
    for layer_index in range(3):
        prompt_profile = torch.tensor(
            [
                [[0.7, 0.2, 0.1, 0.0], [0.1, 0.7, 0.2, 0.0]],
                [[0.6, 0.3, 0.1, 0.0], [0.2, 0.6, 0.2, 0.0]],
            ],
            dtype=torch.float32,
        ).roll(layer_index, dims=-1)
        affinity_layers.append({
            "prompt_true_attribute_effective_count": torch.full(
                (batch_size, prompt_length), 2.0 + layer_index
            ),
            "prompt_attention_to_attribute_concept_patch_mass": torch.full(
                (batch_size, prompt_length), 0.3 + 0.1 * layer_index
            ),
            "_prompt_true_attribute_profile": prompt_profile,
            "_prompt_margin_attribute_profile": prompt_profile,
            "_collection_true_attribute_profile": prompt_profile,
            "_collection_margin_attribute_profile": prompt_profile,
        })
    shallow = ProbeAttentionAffinityAccumulator(
        prompt_length=prompt_length,
        semantic_length=0,
        selected_layers=[0, 1, 2],
        attribute_count=attribute_count,
        attribute_concept_enable=True,
        prompt_mode="persistent_contextualized",
        score_mode="dot",
    )
    shallow.update(
        [], affinity_layers, predictions=[0, 1], targets=[0, 0]
    )
    shallow_result = shallow.finalize()
    assert set(shallow_result["attribute_concept_grounding_by_layer"]) == {
        0, 1, 2
    }
    assert set(
        shallow_result["attribute_concept_grounding_by_layer_and_prompt"][0]
    ) == {0, 1}
    assert all(
        "all.same_slot_prompt_true_cosine" in metrics
        for metrics in shallow_result[
            "cross_layer_concept_continuity_by_pair"
        ].values()
    )
    assert all(
        metrics["attribute_margin_additive_reference_exact"] == 1.0
        for metrics in shallow_result[
            "attribute_concept_grounding_by_layer"
        ].values()
    )
    deep = ProbeAttentionAffinityAccumulator(
        prompt_length=prompt_length,
        semantic_length=0,
        selected_layers=[0, 1, 2],
        attribute_count=attribute_count,
        attribute_concept_enable=True,
        prompt_mode="layerwise_replaced",
        score_mode="cosine",
    )
    deep.update([], affinity_layers, predictions=[0, 1], targets=[0, 0])
    deep_result = deep.finalize()
    assert all(
        "all.set_best_match_prompt_true_cosine" in metrics
        for metrics in deep_result[
            "cross_layer_concept_continuity_by_pair"
        ].values()
    )
    assert all(
        metrics["attribute_margin_additive_reference_exact"] == 0.0
        for metrics in deep_result[
            "attribute_concept_grounding_by_layer"
        ].values()
    )


def _validate_patch_semantic_transport_and_intervention():
    batch_size = 2
    prompt_length = 2
    patch_count = 4
    class_count = 3
    hidden_size = 4
    token_sequence = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0],
             [0.1, 0.8, 0.1, 0.0], [1.0, 0.0, 0.0, 0.0],
             [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0],
             [0.7, 0.2, 0.1, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.2, 0.8, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    semantic = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    raw_semantic = torch.tensor(
        [[1.0, 0.2, 0.0, 0.0], [0.1, 1.0, 0.2, 0.0],
         [0.0, 0.1, 1.0, 0.2]],
        dtype=torch.float32,
    )
    logits = torch.tensor([[4.0, 1.0, 0.0], [0.5, 4.0, 1.0]])
    prompt_attention = torch.tensor(
        [
            [[[0.6, 0.3, 0.05, 0.05], [0.1, 0.2, 0.3, 0.4]],
             [[0.5, 0.35, 0.1, 0.05], [0.15, 0.15, 0.3, 0.4]]],
            [[[0.1, 0.7, 0.1, 0.1], [0.4, 0.2, 0.2, 0.2]],
             [[0.1, 0.6, 0.2, 0.1], [0.35, 0.25, 0.2, 0.2]]],
        ],
        dtype=torch.float32,
    )
    av_magnitude = torch.tensor(
        [
            [[0.8, 0.4, 0.1, 0.1], [0.1, 0.2, 0.5, 0.6]],
            [[0.1, 0.9, 0.2, 0.1], [0.5, 0.3, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )
    attribute_profile = torch.tensor(
        [
            [[0.7, 0.2, 0.1, 0.0], [0.1, 0.2, 0.6, 0.1]],
            [[0.1, 0.7, 0.2, 0.0], [0.5, 0.2, 0.2, 0.1]],
        ],
        dtype=torch.float32,
    )
    final_affinity = {
        "ApKv_attn": prompt_attention,
        "AcKp_attn": torch.tensor(
            [[[[0.7, 0.3]], [[0.6, 0.4]]],
             [[[0.2, 0.8]], [[0.3, 0.7]]]],
            dtype=torch.float32,
        ),
        "prompt_patch_value_contribution_norm": av_magnitude.sum(dim=-1),
        "_prompt_patch_pre_output_av_magnitude": av_magnitude,
        "_attribute_concept_patch_support": torch.tensor(
            [[0.9, 0.7, 0.1, 0.2], [0.2, 0.9, 0.4, 0.1]],
            dtype=torch.float32,
        ),
        "_prompt_true_attribute_profile": attribute_profile,
        "_prompt_margin_attribute_profile": attribute_profile,
        "_collection_true_attribute_profile": attribute_profile,
        "_collection_margin_attribute_profile": attribute_profile,
    }
    accumulator = ProbeAttentionAffinityAccumulator(
        prompt_length=prompt_length,
        semantic_length=0,
        selected_layers=[0],
        class_count=class_count,
        raw_semantic_reference=raw_semantic,
        attribute_count=4,
        attribute_concept_enable=True,
        patch_semantic_transport_enable=True,
        patch_semantic_transport_cost="cosine",
        patch_semantic_transport_temperature=0.5,
        patch_semantic_transport_patch_ratio=0.25,
        prompt_mode="persistent_contextualized",
        score_mode="dot",
    )
    reference = accumulator.update(
        [],
        [final_affinity],
        predictions=[0, 1],
        targets=[0, 1],
        projected_semantic_reference=semantic,
        transport_semantic_reference=semantic,
        token_sequence=token_sequence,
        logits=logits,
    )
    assert reference["selected_patch_indices"].shape == (batch_size, 1)
    assert reference["reference_patch_scores"].shape == (
        batch_size, patch_count
    )
    result = accumulator.finalize()
    transport = result["patch_semantic_transport"]
    assert "all.patch_to_semantic_expected_cost" in transport
    assert "all.semantic_to_patch_expected_cost" in transport
    assert "all.bidirectional_expected_cost" in transport
    assert "all.transport_attribute_topk_jaccard" in transport
    assert "attention_transport_to_patch_collection_role_cosine_mean" in transport
    assert "av_transport_to_collection_true_attribute_cosine_mean" in transport
    assert set(result["patch_semantic_transport_by_prompt"]) == {0, 1}

    tiny_config = SimpleNamespace(
        hidden_size=hidden_size,
        transformer={"num_heads": 2, "attention_dropout_rate": 0.0},
    )
    holder = torch.nn.Module()
    holder.attention = Attention(tiny_config, vis=True).eval()
    _, normal_attention, _ = holder.attention(
        token_sequence, prompt_length=prompt_length
    )
    targeted_indices = torch.tensor([[0], [1]], dtype=torch.long)
    reference_scores = torch.tensor(
        [[0.9, 0.4, 0.2, 0.1], [0.2, 0.8, 0.3, 0.1]],
        dtype=torch.float32,
    )
    with transport_prompt_patch_block_intervention(
        holder,
        selected_patch_indices=targeted_indices,
        reference_patch_scores=reference_scores,
    ):
        _, targeted_attention, _ = holder.attention(
            token_sequence, prompt_length=prompt_length
        )
    targeted_stats = holder.attention._last_prompt_path_intervention_stats
    prompt_slice = slice(1, 1 + prompt_length)
    patch_slice = slice(1 + prompt_length, 1 + prompt_length + patch_count)
    assert torch.allclose(
        normal_attention[:, :, prompt_slice, patch_slice].sum(dim=-1),
        targeted_attention[:, :, prompt_slice, patch_slice].sum(dim=-1),
        atol=1e-6,
    )
    assert float(
        targeted_stats[
            "transport_intervention_selected_prompt_attention_mass_after"
        ].abs().max().item()
    ) <= 1e-7
    assert float(
        targeted_stats[
            "transport_intervention_prompt_patch_mass_abs_error"
        ].max().item()
    ) <= 1e-6
    random_indices = torch.tensor([[2], [3]], dtype=torch.long)
    with transport_random_patch_block_intervention(
        holder,
        selected_patch_indices=random_indices,
        reference_patch_scores=reference_scores,
    ):
        _, random_attention, _ = holder.attention(
            token_sequence, prompt_length=prompt_length
        )
    random_stats = holder.attention._last_prompt_path_intervention_stats
    assert torch.allclose(
        normal_attention[:, :, prompt_slice, patch_slice].sum(dim=-1),
        random_attention[:, :, prompt_slice, patch_slice].sum(dim=-1),
        atol=1e-6,
    )
    assert float(
        random_stats["transport_intervention_targeted"].abs().max().item()
    ) == 0.0


def _validate_deep_prompt_parameter_health():
    shallow_model = torch.nn.Module()
    shallow_model.register_parameter(
        "prompt_embeddings", torch.nn.Parameter(torch.randn(1, 3, 4))
    )
    assert PromptParameterTracker(shallow_model).layer_metrics() == {}

    deep_model = torch.nn.Module()
    deep_model.register_parameter(
        "prompt_embeddings", torch.nn.Parameter(torch.randn(1, 3, 4))
    )
    deep_model.register_parameter(
        "deep_prompt_embeddings", torch.nn.Parameter(torch.randn(2, 3, 4))
    )
    tracker = PromptParameterTracker(deep_model)
    loss = sum(parameter.pow(2).sum() for parameter in deep_model.parameters())
    loss.backward()
    tracker.observe_gradients()
    with torch.no_grad():
        deep_model.deep_prompt_embeddings.add_(0.1)
    metrics = tracker.layer_metrics()
    assert set(metrics) == {0, 1, 2}
    assert all(values["prompt_grad_observation_count"] == 1.0 for values in metrics.values())
    assert metrics[1]["prompt_relative_update"] > 0.0
    assert metrics[1]["epoch_parameter_step_norm"] > 0.0
    assert "previous_layer_prompt_cosine" in metrics[1]
    assert "prompt_effective_rank" in metrics[2]
    overall = tracker.metrics()
    assert overall["epoch_parameter_step_norm"] > 0.0
    tracker.commit_epoch_snapshot()
    assert np.isclose(tracker.metrics()["epoch_parameter_step_norm"], 0.0)
    assert np.isclose(tracker.metrics()["cross_epoch_prompt_cosine"], 1.0)


def _validate_streaming_fixed_probe_metrics():
    rng = np.random.RandomState(29)
    targets = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    semantic = rng.normal(size=(3, 5)).astype(np.float32)
    visual = semantic[targets] + rng.normal(scale=0.1, size=(targets.size, 5)).astype(np.float32)
    logits = visual @ semantic.T
    accumulator = StreamingFixedProbeAccumulator([0, 1, 2])
    accumulator.update(logits[:3], targets[:3], visual[:3], semantic)
    accumulator.update(logits[3:], targets[3:], visual[3:], semantic)
    streamed = accumulator.finalize()
    class_aggregates = accumulator.class_aggregates()
    assert sum(row["support"] for row in class_aggregates) == targets.size
    assert {row["class_id"] for row in class_aggregates} == {0, 1, 2}
    assert all(set(row) == {"class_id", "support", "correct_count", "accuracy"} for row in class_aggregates)
    _assert_metric_maps_close(
        classification_metrics(logits, targets), streamed["classification"], "fixed_classification"
    )
    expected_geometry = representation_geometry_metrics(visual, targets)
    _assert_metric_maps_close(expected_geometry, streamed["representation_geometry"], "fixed_geometry")
    assert {
        "within_class_scatter_trace",
        "between_class_scatter_trace",
        "fisher_trace_ratio",
    }.issubset(expected_geometry)
    assert {
        "within_class_scatter",
        "between_class_scatter",
        "fisher_ratio",
    }.isdisjoint(expected_geometry)
    expected_alignment = visual_semantic_alignment_metrics(visual, semantic, targets)
    for name, value in expected_alignment.items():
        assert np.isclose(streamed["visual_semantic_alignment"][name], value, rtol=2e-5, atol=2e-6), name
    expected_graph = semantic_visual_graph_metrics(visual, semantic, targets, logits=logits)
    for name, value in expected_graph.items():
        assert np.isclose(streamed["semantic_graph_reference"][name], value, rtol=2e-5, atol=2e-6), name

    changed = logits.copy()
    changed[:, [0, 1, 2]] = changed[:, [1, 2, 0]]
    expected_effect = paired_module_effect_metrics(logits, changed, targets, [0, 1, 2], [0, 1])
    effect_accumulator = PairedModuleEffectAccumulator([0, 1, 2], [0, 1])
    effect_accumulator.update(logits[:3], changed[:3], targets[:3])
    effect_accumulator.update(logits[3:], changed[3:], targets[3:])
    streamed_effect = effect_accumulator.finalize()
    for name, value in expected_effect["summary"].items():
        assert np.isclose(streamed_effect["summary"][name], value, rtol=2e-5, atol=2e-6), name
    assert streamed_effect["per_class"] == expected_effect["per_class"]

    tensor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_effect_accumulator = PairedModuleEffectAccumulator([0, 1, 2], [0, 1])
    tensor_effect_accumulator.update(
        torch.as_tensor(logits, device=tensor_device).requires_grad_(True),
        torch.as_tensor(changed, device=tensor_device).requires_grad_(True),
        torch.as_tensor(targets, device=tensor_device),
    )
    tensor_effect = tensor_effect_accumulator.finalize()
    for name, value in expected_effect["summary"].items():
        assert np.isclose(tensor_effect["summary"][name], value, rtol=2e-5, atol=2e-6), name
    assert tensor_effect["per_class"] == expected_effect["per_class"]

    cfg = get_cfg()
    head = RSimilarityClassifier(torch.from_numpy(semantic), hidden_size=5, cfg=cfg)
    with torch.no_grad():
        head.prototype_proj.weight.copy_(torch.eye(5))
        head.prototype_proj.bias.zero_()
    visual_tensor = torch.from_numpy(visual)
    normal_logits = head(visual_tensor, class_ids=[0, 1, 2])
    permutation = [2, 0, 1]
    inverse = torch.tensor([1, 2, 0], dtype=torch.long)
    synchronized_logits = head(
        visual_tensor, class_ids=permutation, prototype_class_ids=permutation
    )
    assert torch.allclose(normal_logits, synchronized_logits.index_select(1, inverse))
    mismatched_logits = head(
        visual_tensor, class_ids=[0, 1, 2], prototype_class_ids=permutation
    )
    assert not torch.allclose(normal_logits, mismatched_logits)

    token_accumulator = StreamingTokenViewAccumulator(3, prompt_length=2, semantic_length=0)
    tokens = torch.randn(6, 7, 5)
    token_accumulator.update(tokens[:3], targets[:3], semantic)
    token_accumulator.update(tokens[3:], targets[3:], semantic)
    token_metrics = token_accumulator.finalize()
    assert set(token_metrics["representation_geometry"]) == {
        "cls",
        "pooled_patch",
        "contextualized_prompt",
        "within_image_patch_diversity",
    }
    patch_diversity = token_metrics["representation_geometry"][
        "within_image_patch_diversity"
    ]
    assert -1.0 <= patch_diversity["pairwise_cosine_mean"] <= 1.0
    assert 0.0 <= patch_diversity["dispersion_mean"] <= 2.0
    assert "per_class_dispersion_median" in patch_diversity
    assert "contextualized_prompt_instance_variance" in token_metrics["prompt_parameter_health"]
    assert "prompt_cls_gram_alignment" in token_metrics["relation_stability"]
    semantic_role = token_metrics["prompt_semantic_role_reference"]
    assert "best_prompt_true_prototype_similarity" in semantic_role
    assert 1.0 <= semantic_role["true_semantic_prompt_effective_count"] <= 2.0
    assert 0.0 <= semantic_role["true_semantic_prompt_top1_share"] <= 1.0


def _validate_comparability_identity():
    probe_manifest = {
        "probes": {
            "probe_test_unseen": {
                "probe_id": "probe-test-unseen",
                "manifest_sha256": "probe-sha",
                "selection_seed": 424242,
                "candidate_class_ids": [0, 1, 2],
            }
        }
    }
    checkpoint = {"checkpoint_selection_rule": "predeclared_final_epoch", "checkpoint_sha256": "checkpoint"}
    with tempfile.TemporaryDirectory(prefix="comparability_validation_") as temp_dir:
        a0 = get_cfg()
        a0.OUTPUT_DIR = str(Path(temp_dir) / "a0")
        a0.SEED = 0
        a0.MODEL.PROMPT.ENABLE = False
        a0.MODEL.PROMPT.BACKEND = "dynamic"
        a0.MODEL.PROMPT.DEEP = False
        a1 = a0.clone()
        a1.OUTPUT_DIR = str(Path(temp_dir) / "a1")
        a1.SEED = 1
        a1.MODEL.PROMPT.ENABLE = True
        a1.MODEL.PROMPT.BACKEND = "vpt_deep"
        a1.MODEL.PROMPT.DEEP = True
        identity_a0 = build_comparability_identity(
            a0, run_id="run-a0", session_id="session-a0",
            checkpoint_manifest=checkpoint, probe_manifest=probe_manifest,
        )
        identity_a1 = build_comparability_identity(
            a1, run_id="run-a1", session_id="session-a1",
            checkpoint_manifest=checkpoint, probe_manifest=probe_manifest,
        )
        assert identity_a0["run_identity"]["sha256"] != identity_a1["run_identity"]["sha256"]
        assert identity_a0["shared_condition_fingerprint"]["sha256"] == identity_a1["shared_condition_fingerprint"]["sha256"]
        a1.MODEL.PROMPT.NUM_TOKENS += 1
        changed = build_comparability_identity(
            a1, run_id="run-a1", session_id="session-a1",
            checkpoint_manifest=checkpoint, probe_manifest=probe_manifest,
        )
        assert identity_a0["shared_condition_fingerprint"]["sha256"] != changed["shared_condition_fingerprint"]["sha256"]


def _validate_fixed_probe_seed_dispatch():
    cfg = get_cfg()
    cfg.MONITOR.PROBE.ENABLE = True
    cfg.MONITOR.PROBE.SELECTION_SEED = 17
    cfg.MONITOR.PROBE.ROBUSTNESS_SELECTION_SEEDS = [18, 19]
    calls = []
    artifacts = []

    class RecordingDiagnostics:
        enabled = True

        @staticmethod
        def record_probe_artifact(path, payload):
            artifacts.append((path, payload))

    class DispatchHarness:
        diagnostic_manager = RecordingDiagnostics()
        _fixed_probe_execution_profile = staticmethod(
            Trainer._fixed_probe_execution_profile
        )

        def __init__(self):
            self.cfg = cfg

        def _run_fixed_probes(
            self,
            train_loader,
            test_seen_loader,
            test_unseen_loader,
            *,
            checkpoint_epoch,
            selection_seed=None,
            artifact_prefix="",
            execution_profile=None,
        ):
            if selection_seed is None:
                return Trainer._run_fixed_probes(
                    self,
                    train_loader,
                    test_seen_loader,
                    test_unseen_loader,
                    checkpoint_epoch=checkpoint_epoch,
                    execution_profile=execution_profile,
                )
            calls.append(
                (
                    int(selection_seed),
                    str(artifact_prefix),
                    str(execution_profile),
                )
            )
            return {
                "selection_seed": int(selection_seed),
                "artifact_prefix": str(artifact_prefix),
                "execution_profile": str(execution_profile),
                "valid": True,
            }

    DispatchHarness()._run_fixed_probes(None, None, None, checkpoint_epoch=1)
    assert calls == [
        (17, "", "final_full"),
        (18, "fixed_probe_robustness/selection_seed_18", "robustness_core"),
        (19, "fixed_probe_robustness/selection_seed_19", "robustness_core"),
    ]
    assert artifacts[0][0] == "probe_robustness_manifest.json"
    assert artifacts[0][1]["selection_seeds"] == [17, 18, 19]


def _validate_probe_runtime_timing():
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = get_cfg()
    trainer.device = torch.device("cpu")
    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    loader = trainer._build_fixed_probe_loader(dataset, batch_size=4)
    assert loader.num_workers == 0
    assert not loader.pin_memory

    def consume(timed_loader):
        return sum(int(batch[0].shape[0]) for batch in timed_loader)

    sample_count, timing = trainer._execute_timed_probe_stage(loader, consume)
    assert sample_count == 10
    assert timing["batch_count"] == 3
    assert timing["probe_total_time_sec"] >= timing["probe_data_time_sec"] >= 0.0
    assert timing["probe_compute_time_sec"] >= 0.0
    assert 0.0 <= timing["probe_data_time_ratio"] <= 1.0
    aggregate = trainer._aggregate_probe_stage_timings(
        {"probe_test_unseen": {"fixed_probe_bundle": timing}}
    )
    assert aggregate["stage_count"] == 1
    assert aggregate["batch_count"] == 3
    assert aggregate["probe_total_time_sec"] == timing["probe_total_time_sec"]

    trainer.cfg.defrost()
    trainer.cfg.MONITOR.PROBE.NUM_WORKERS = -1
    failed = False
    try:
        trainer._fixed_probe_loader_settings()
    except ValueError:
        failed = True
    assert failed


def _validate_probe_evidence_roles():
    assert probe_record_evidence_role({
        "condition": "normal", "domain": "classification"
    }) == PROBE_CONTEXT_ONLY
    assert probe_record_evidence_role({
        "condition": "normal", "domain": "probe_context"
    }) == PROBE_CONTEXT_ONLY
    assert probe_record_evidence_role({
        "condition": "prompt_zeroed_affinity_forward",
        "domain": "affinity_forward_equivalence",
    }) == VALIDITY_OR_IDENTITY
    assert probe_record_evidence_role({
        "condition": "prompt_zeroed", "domain": "module_effect"
    }) == MECHANISM_EVIDENCE
    assert probe_record_evidence_role({
        "condition": "normal", "domain": "target_relevance_reference"
    }) == MECHANISM_EVIDENCE


def _validate_fixed_probe_semantic_bundle():
    cfg = get_cfg()
    cfg.MODEL.PROMPT.ENABLE = True
    cfg.MODEL.PROMPT.DEEP = False
    cfg.MODEL.PROMPT.NUM_TOKENS = 2
    cfg.MODEL.SEMANTIC_TOKENS.ENABLE = False
    cfg.MONITOR.PROBE.AFFINITY_ENABLE = True
    cfg.MONITOR.PROBE.ATTENTION_ENABLE = True
    cfg.MONITOR.PROBE.LAYERS = [0, 2, 3]
    cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE = True
    cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.ENABLE = True
    cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO = 0.25
    cfg.MONITOR.PROBE.PROMPT_ANALYSIS.ENABLE = True
    cfg.MONITOR.PROBE.PROMPT_ANALYSIS.CONTENT_ENABLE = True
    cfg.MONITOR.MODULE_EFFECT.ENABLE = True
    cfg.MONITOR.MODULE_EFFECT.PROMPT_ZERO = True
    cfg.MONITOR.MODULE_EFFECT.PROMPT_READ_BLOCK = True
    cfg.MONITOR.MODULE_EFFECT.PROMPT_WRITE_BLOCK = True
    cfg.MONITOR.MODULE_EFFECT.PROMPT_SELECTION_UNIFORM = True
    cfg.MONITOR.MODULE_EFFECT.PATCH_PROMPT_SELECTION_UNIFORM = True
    cfg.MONITOR.MODULE_EFFECT.PROMPT_VALUE_GLOBALIZE = True
    cfg.MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_BLOCK = True
    cfg.MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_LAYERS = [0, 2]
    cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP = True
    cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_LAYER = 2
    cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_SEED = 31
    cfg.MONITOR.MODULE_EFFECT.TRANSPORT_PATCH_BLOCK = True
    semantic = torch.eye(4, dtype=torch.float32)
    inputs = torch.cat([semantic, semantic], dim=0)
    targets = torch.tensor([1, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    attributes = torch.zeros(inputs.shape[0], 1)
    probe_samples = [
        {
            "image": inputs[index],
            "label": targets[index],
            "attribute": attributes[index],
            "sample_id": f"synthetic-{index}",
        }
        for index in range(inputs.shape[0])
    ]
    loader = torch.utils.data.DataLoader(
        probe_samples, batch_size=4, shuffle=False
    )
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.model = SyntheticProbeModel(semantic, cfg)
    trainer.prompt_parameter_tracker = PromptParameterTracker(trainer.model)
    trainer.get_input = lambda input_data: (
        input_data["image"],
        input_data["label"],
        input_data["attribute"],
    )
    trainer._prepare_semantics_for_stage = lambda attributes, dataset, batch_size, is_train: None
    dataset = type(
        "Dataset",
        (),
        {"seen_classes": [0, 1], "class_attributes": semantic},
    )()
    bundle = trainer._execute_fixed_probe_bundle(loader, dataset, [0, 1, 2, 3], split="probe_test_unseen")
    assert set(bundle["conditions"]) == {
        "normal", "synchronized_class_permutation", "mismatched_semantic_permutation"
    }
    assert bundle["semantic_intervention"]["synchronized_equivalence_pass"]
    assert bundle["semantic_intervention"]["synchronized"]["logit_max_abs_diff"] == 0.0
    assert bundle["semantic_intervention"]["mismatched"]["prediction_flip_rate"] > 0.0
    assert set(bundle["affinity"]["affinity_health_by_layer"]) == {0, 2, 3}
    assert set(bundle["affinity"]["prompt_patch_bridge_by_layer"]) == {0, 2, 3}
    assert set(bundle["affinity"]["prompt_semantic_role_by_layer"]) == {0, 2, 3}
    assert set(
        bundle["affinity"]["prompt_semantic_role_by_layer_and_prompt"][0]
    ) == {0, 1}
    assert set(bundle["affinity"]["prompt_content_by_layer"]) == {0, 2, 3}
    assert set(bundle["affinity"]["prompt_content_by_layer_and_head"][0]) == {
        0, 1
    }
    assert set(
        bundle["affinity"]["prompt_content_by_layer_and_prompt"][0]
    ) == {0, 1}
    assert bundle["affinity"]["token_metrics"][
        "prompt_semantic_role_reference"
    ]
    assert bundle["affinity"]["patch_semantic_transport_metrics"]
    assert set(bundle["affinity"]["patch_semantic_transport_by_prompt"]) == {
        0, 1
    }
    assert all(
        "QcKv.raw_mean" in bundle["affinity"]["affinity_health_by_layer"][layer_index]
        for layer_index in (0, 2, 3)
    )
    retrieval_metrics = bundle["affinity"]["attention_flow_metrics"]
    assert {
        "prompt_patch_retrieval_global_patch_usage_entropy",
        "prompt_patch_retrieval_coordination_information",
        "prompt_patch_retrieval_selection_overlap",
        "prompt_patch_retrieval_top1_unique_patch_count",
        "prompt_patch_retrieval_top1_patch_coverage_ratio",
        "prompt_patch_retrieval_top1_repeat_rate",
        "prompt_patch_retrieval_spatial_centroid_dispersion",
        "prompt_patch_retrieval_spatial_radius",
    }.issubset(retrieval_metrics)
    assert "prompt_zeroed" in bundle["module_effects"]
    assert {
        "prompt_zeroed",
        "prompt_read_blocked",
        "prompt_write_blocked",
        "prompt_patch_selection_uniform",
        "patch_prompt_selection_uniform",
        "prompt_patch_value_globalized",
        "prompt_read_blocked_layer_0",
        "prompt_read_blocked_layer_2",
        "prompt_context_swapped",
        "transport_targeted_prompt_patch_blocked",
        "transport_random_patch_blocked",
    }.issubset(bundle["module_effects"])
    value_globalized = bundle["module_effects"]["prompt_patch_value_globalized"]
    assert value_globalized["valid"], value_globalized
    value_contract = bundle["intervention_diagnostics"][
        "prompt_patch_value_globalized"
    ]["value_globalization_contract"]
    assert value_contract["applied_pass"]
    assert value_contract["attention_preserved_pass"]
    assert value_contract["value_collapsed_pass"]
    for source_layer in (0, 2):
        intervention_name = f"prompt_read_blocked_layer_{source_layer}"
        read_effect = bundle["module_effects"][intervention_name]
        assert read_effect["valid"], read_effect
        read_contract = bundle["intervention_diagnostics"][intervention_name][
            "read_save_consume_contract"
        ]
        assert read_contract["applied_pass"]
        assert read_contract["source_removed_pass"]
        assert read_contract["chain_summary"]["downstream_selected_layers"]
        assert read_contract["chain_summary"][
            "source_delta_prompt_layer_change_norm"
        ] is not None
        assert read_contract["chain_summary"][
            "downstream_delta_prompt_layer_output_norm_mean"
        ] is not None
        assert read_contract["chain_summary"]["final_delta_logits_norm"] is not None
    prompt_swap = bundle["module_effects"]["prompt_context_swapped"]
    assert prompt_swap["valid"], prompt_swap
    assert prompt_swap["runtime_contract"]["pairing_storage"] == "hash_only"
    assert prompt_swap["runtime_contract"]["sample_count"] == inputs.shape[0]
    swap_contract = bundle["intervention_diagnostics"][
        "prompt_context_swapped"
    ]["prompt_context_swap_contract"]
    assert swap_contract["applied_pass"]
    assert swap_contract["no_self_pair_pass"]
    assert swap_contract["full_coverage_pass"]
    transport_targeted = bundle["module_effects"][
        "transport_targeted_prompt_patch_blocked"
    ]
    assert transport_targeted["summary"][
        "transport_targeted_prompt_patch_mass_preservation_pass"
    ] == 1.0
    assert transport_targeted["summary"][
        "transport_intervention_selected_path_removed_pass"
    ] == 1.0
    assert bundle["transport_intervention_comparison"][
        "transport_targeted_vs_random_selection_pass"
    ] == 1.0
    assert bundle["module_effects"]["prompt_zeroed"]["summary"]["delta_logits_norm"] > 0.0
    assert "delta_semantic_margin" in bundle["module_effects"]["prompt_zeroed"]["summary"]
    assert bundle["module_effects"]["prompt_zeroed"]["intervention_semantics"] == {
        "zeroed_object": "trainable_prompt_parameters",
        "prompt_slots_removed": False,
        "attention_route_retained": True,
    }
    prompt_zero = bundle["prompt_zero_diagnostics"]
    assert prompt_zero["alignment_metrics"]
    assert prompt_zero["affinity"]["paired_attention_valid"]
    assert set(prompt_zero["affinity"]["attention_flow_by_layer"]) == {0, 2, 3}
    assert set(prompt_zero["affinity"]["attention_delta_by_layer"]) == {0, 2, 3}
    assert set(prompt_zero["affinity"]["prompt_patch_bridge_delta_by_layer"]) == {
        0, 2, 3
    }
    assert set(prompt_zero["affinity"]["prompt_semantic_role_delta_by_layer"]) == {
        0, 2, 3
    }
    assert any(
        abs(metrics["delta_cls_to_prompt_mass"]) > 0.0
        for metrics in prompt_zero["affinity"]["attention_delta_by_layer"].values()
    )
    selection_uniform = bundle["intervention_diagnostics"][
        "prompt_patch_selection_uniform"
    ]
    assert selection_uniform["affinity"]["paired_attention_valid"]
    assert selection_uniform["selection_mass_preservation"]["pass"], (
        selection_uniform["selection_mass_preservation"],
    )
    assert selection_uniform["selection_mass_preservation"][
        "contract_basis"
    ] == "local_intervention_mass_error"
    assert selection_uniform["selection_mass_preservation"]["applied_pass"]
    assert selection_uniform["selection_mass_preservation"][
        "local_mass_abs_error"
    ] <= cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
    assert abs(
        selection_uniform["selection_mass_preservation"][
            "delta_prompt_to_patch_mass"
        ]
    ) <= cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
    assert all(
        metrics["delta_prompt_to_patch_conditional_entropy"] > 0.0
        for metrics in selection_uniform["affinity"][
            "attention_delta_by_layer"
        ].values()
    )
    patch_selection_uniform = bundle["intervention_diagnostics"][
        "patch_prompt_selection_uniform"
    ]
    assert patch_selection_uniform["affinity"]["paired_attention_valid"]
    assert patch_selection_uniform["selection_mass_preservation"]["pass"], (
        patch_selection_uniform["selection_mass_preservation"],
    )
    assert patch_selection_uniform["selection_mass_preservation"][
        "contract_basis"
    ] == "local_intervention_mass_error"
    assert patch_selection_uniform["selection_mass_preservation"]["applied_pass"]
    assert patch_selection_uniform["selection_mass_preservation"][
        "local_mass_abs_error"
    ] <= cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
    assert abs(
        patch_selection_uniform["selection_mass_preservation"][
            "delta_patch_to_prompt_mass"
        ]
    ) <= cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
    assert all(
        metrics["delta_patch_to_prompt_conditional_entropy"] > 0.0
        for metrics in patch_selection_uniform["affinity"][
            "attention_delta_by_layer"
        ].values()
    )
    assert all(
        metrics["delta_patch_prompt_coordination_information"] < 0.0
        for metrics in patch_selection_uniform["affinity"][
            "attention_delta_by_layer"
        ].values()
    )
    assert all(
        metrics["delta_patch_prompt_selection_overlap"] > 0.0
        for metrics in patch_selection_uniform["affinity"][
            "attention_delta_by_layer"
        ].values()
    )
    patch_diversity_delta = patch_selection_uniform[
        "representation_geometry"
    ]["within_image_patch_diversity_delta"]
    assert "delta_dispersion_mean" in patch_diversity_delta
    assert "delta_within_image_patch_dispersion_mean" in bundle[
        "module_effects"
    ]["patch_prompt_selection_uniform"]["summary"]
    assert all(
        metrics["delta_patch_prompt_top1_coverage_ratio"] < 0.0
        and metrics["delta_patch_prompt_top1_repeat_rate"] > 0.0
        for metrics in patch_selection_uniform["affinity"][
            "attention_delta_by_layer"
        ].values()
    )
    target_relevance = trainer._execute_target_relevance_probe(
        loader, dataset, [0, 1, 2, 3], split="probe_test_unseen"
    )
    assert target_relevance["valid"], target_relevance
    assert target_relevance["observed_layers"] == [0, 2, 3]
    assert target_relevance["equivalence"]["target_relevance_forward_equivalence_pass"] == 1.0
    assert target_relevance["by_layer"][0]["cls_to_prompt"]["all.absolute_sum"] > 0.0
    assert target_relevance["by_layer"][0]["cls_to_prompt/prompt_0"][
        "all.absolute_sum"
    ] > 0.0
    assert "prompt_to_patch/prompt_1" in target_relevance["by_layer"][0]
    predicted_objective = target_relevance["objective_results"][
        "predicted_class_margin"
    ]
    assert predicted_objective["applicability"] == "applicable"
    assert predicted_objective["valid"]
    assert predicted_objective["by_layer"][0]["cls_to_prompt"]["all.absolute_sum"] > 0.0
    cfg.MODEL.PROMPT.DEEP = True
    deep_specs = {
        spec["name"]: spec
        for spec in trainer._module_effect_intervention_specs(
            prompt_length=cfg.MODEL.PROMPT.NUM_TOKENS,
            attribute_concept_available=True,
            patch_semantic_transport_available=True,
        )
    }
    assert not deep_specs["prompt_context_swapped"]["applicable"]
    assert deep_specs["prompt_context_swapped"]["not_applicable_reason"] == (
        "deep_prompt_layerwise_replacement"
    )
    assert deep_specs["prompt_read_blocked_layer_2"]["applicable"]
    assert deep_specs["prompt_read_blocked_layer_2"]["intervention_semantics"][
        "a2_interpretation"
    ] == "layer_output_replacement_negative_control"


def _validate_generalization_trajectory_rules():
    tradeoff_rows = []
    for epoch in range(1, 7):
        tradeoff_rows.append({
            "epoch": epoch,
            "complete": True,
            "train_loss": 2.0 - 0.15 * epoch,
            "test_seen_nll": 2.5 - 0.1 * epoch,
            "test_unseen_nll": 2.5 + 0.1 * epoch,
            "gzsl_seen": 0.40 + 0.02 * epoch,
            "gzsl_unseen": 0.50 - 0.02 * epoch,
            "gzsl_h": 0.45,
        })
    tradeoff = _summarize_generalization_trajectory(
        tradeoff_rows,
        run_completed=True,
    )
    assert tradeoff["valid"]
    assert tradeoff["seen_specialization_tradeoff"]
    assert tradeoff["primary_pattern"] == "seen_specialization_tradeoff"
    assert not tradeoff["checkpoint_selection_allowed"]
    assert tradeoff["self_relative_formation_epoch"] == 2
    assert np.isclose(tradeoff["persistence_ratio"], 1.0)

    h_values = [0.30, 0.42, 0.55, 0.58, 0.46, 0.40]
    seen_nll_values = [2.8, 2.4, 2.0, 1.9, 2.2, 2.4]
    unseen_nll_values = [3.2, 2.8, 2.3, 2.1, 2.5, 2.8]
    degradation_rows = []
    for index, epoch in enumerate(range(1, 7)):
        degradation_rows.append({
            "epoch": epoch,
            "complete": True,
            "train_loss": 2.0 - 0.18 * epoch,
            "test_seen_nll": seen_nll_values[index],
            "test_unseen_nll": unseen_nll_values[index],
            "gzsl_seen": h_values[index] + 0.05,
            "gzsl_unseen": h_values[index] - 0.05,
            "gzsl_h": h_values[index],
        })
    degradation = _summarize_generalization_trajectory(
        degradation_rows,
        run_completed=True,
    )
    assert degradation["valid"]
    assert degradation["late_generalization_degradation_candidate"]
    assert degradation["primary_pattern"] == (
        "late_generalization_degradation_candidate"
    )


def _validate_low_cost_trajectory_extensions():
    tracker = EpochPredictionTransitionTracker()
    first = tracker.update(
        epoch=1,
        split="test_unseen",
        sample_ids=["sample-b", "sample-a", "sample-c"],
        predictions=[1, 0, 2],
        targets_local=[0, 0, 2],
        candidate_global_ids=[10, 11, 12],
    )
    assert not first["valid"]
    assert first["status"] == "previous_epoch_state_unavailable"
    second = tracker.update(
        epoch=2,
        split="test_unseen",
        sample_ids=["sample-c", "sample-a", "sample-b"],
        predictions=[1, 0, 0],
        targets_local=[2, 0, 0],
        candidate_global_ids=[10, 11, 12],
    )
    assert second["valid"]
    assert np.isclose(second["summary"]["prediction_flip_rate"], 2.0 / 3.0)
    assert np.isclose(second["summary"]["correction_rate"], 1.0 / 3.0)
    assert np.isclose(second["summary"]["regression_rate"], 1.0 / 3.0)
    assert int(second["arrays"]["support"].sum()) == 3

    loss_metrics = loss_component_metrics({
        "ce_loss": 1.25,
        "ce_loss.raw": 1.25,
        "ce_loss.weight": 1.0,
        "ce_loss.weighted": 1.25,
        "ce_loss.weighted_share": 1.0,
        "total_loss": 1.25,
        "unrelated_debug": 7.0,
    })
    assert set(loss_metrics) == {
        "ce_loss.raw",
        "ce_loss.weight",
        "ce_loss.weighted",
        "ce_loss.weighted_share",
        "total_loss",
    }

    reference_condition = {
        "condition_status": "consistent_across_seeds",
        "condition_by_seed": {
            "0": {"condition_fields": {"data.name": "CUB", "solver.base_lr": 0.0006}}
        },
    }
    target_condition = {
        "condition_status": "consistent_across_seeds",
        "condition_by_seed": {
            "0": {"condition_fields": {"data.name": "CUB", "solver.base_lr": 0.0003}}
        },
    }
    controlled = _condition_comparison(
        reference_condition, target_condition, ["solver.base_lr"]
    )
    assert controlled["valid"]
    assert controlled["status"] == "compatible_predeclared_axes"
    uncontrolled = _condition_comparison(reference_condition, target_condition, [])
    assert not uncontrolled["valid"]
    paired = _paired_seed_delta(
        {"seed_values": {"0": 0.4, "1": 0.5}},
        {"seed_values": {"0": 0.5, "1": 0.45}},
    )
    assert paired["paired_seed_count"] == 2
    assert np.isclose(paired["paired_seed_delta_mean"], 0.025)


def _validate_cross_seed_mechanism_summary():
    with tempfile.TemporaryDirectory(prefix="cross_seed_validation_") as temp_dir:
        root = Path(temp_dir)
        runs = []
        stage_value = {"A0": 0.4, "A1": 0.5, "A2": 0.6}
        for stage, base_value in stage_value.items():
            for seed in (0, 1, 2):
                run_dir = root / stage / f"seed{seed}"
                diagnostics = run_dir / "diagnostics"
                diagnostics.mkdir(parents=True)
                calibration_dir = diagnostics / "calibration_profile"
                calibration_dir.mkdir()
                (calibration_dir / "epoch_0030.json").write_text(
                    json.dumps({"summary": {
                        "ausuc": base_value,
                        "raw_to_oracle_gain": 0.02,
                        "oracle_peak_gamma": 0.0,
                    }}),
                    encoding="utf-8",
                )
                (run_dir / "monitor_runtime_summary.json").write_text(
                    json.dumps({
                        "seed": seed,
                        "run_id": f"{stage}-{seed}",
                        "session_id": f"session-{stage}-{seed}",
                        "status": "completed",
                    }),
                    encoding="utf-8",
                )
                with (run_dir / "metrics_epoch.csv").open("w", encoding="utf-8", newline="") as handle:
                    fields = ["epoch", "split", "namespace", "metric", "value"]
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    epoch_rows = []
                    for epoch in range(1, 7):
                        progress = float(epoch - 1) / 5.0
                        epoch_rows.extend([
                            {"epoch": epoch, "split": "train", "namespace": "train_epoch", "metric": "loss", "value": 2.0 - 0.15 * epoch},
                            {"epoch": epoch, "split": "test_seen", "namespace": "classification", "metric": "nll", "value": 2.5 - 0.3 * progress},
                            {"epoch": epoch, "split": "test_unseen", "namespace": "classification", "metric": "nll", "value": 3.0 - 0.2 * progress},
                            {"epoch": epoch, "split": "test_gzsl", "namespace": "classification", "metric": "gzsl_seen", "value": base_value + 0.07 + 0.03 * progress},
                            {"epoch": epoch, "split": "test_gzsl", "namespace": "classification", "metric": "gzsl_unseen", "value": base_value - 0.03 + 0.03 * progress},
                            {"epoch": epoch, "split": "test_gzsl", "namespace": "classification", "metric": "gzsl_h", "value": base_value + 0.02 + 0.03 * progress},
                        ])
                    epoch_rows.append(
                        {"epoch": 6, "split": "train", "namespace": "prompt_parameter_health", "metric": "prompt_relative_update", "value": 0.0 if stage == "A0" else 0.1}
                    )
                    if stage == "A2":
                        epoch_rows.append({
                            "epoch": 6,
                            "split": "train",
                            "namespace": "prompt_parameter_health",
                            "metric": "layer_1.prompt_grad_norm_epoch_mean",
                            "value": 0.25,
                        })
                    writer.writerows(epoch_rows)
                probe_rows = [
                    ("normal", "classification", "split", "all", "top1", base_value),
                    ("synchronized_class_permutation", "semantic_prototype_intervention", "intervention", "inverse_recovery", "synchronized_equivalence_pass", 1.0),
                    ("mismatched_semantic_permutation", "semantic_prototype_intervention", "intervention", "semantic_mismatch", "mismatched_semantic_effect_pass", 1.0),
                    ("normal", "relation_stability", "relationship", "prompt_cls_patch", "prompt_cls_gram_alignment", 0.7 + base_value),
                    ("normal", "visual_semantic_alignment", "split", "all", "semantic_margin", base_value),
                    ("normal", "target_relevance_reference", "layer_path", "layer_3/cls_to_patch", "all.layer_normalized_net", 0.1 + base_value),
                ]
                if stage != "A0":
                    probe_rows.extend([
                        ("prompt_zeroed", "module_effect", "intervention", "prompt_zeroed", "delta_true_margin", -0.2),
                        ("prompt_zeroed_affinity_forward", "affinity_forward_equivalence", "split", "all", "affinity_forward_equivalence_pass", 1.0),
                        ("prompt_zeroed", "module_effect", "layer", "layer_3", "delta_cls_to_prompt_mass", -0.1),
                        ("prompt_zeroed", "module_effect", "intervention", "prompt_zeroed", "delta_semantic_margin", -0.05),
                        ("prompt_read_blocked", "module_effect", "intervention", "prompt_read_blocked", "delta_true_margin", -0.08),
                        ("prompt_write_blocked", "module_effect", "intervention", "prompt_write_blocked", "delta_semantic_margin", -0.04),
                        ("prompt_patch_selection_uniform", "module_effect", "intervention", "prompt_patch_selection_uniform", "prompt_patch_mass_preservation_pass", 1.0),
                        ("patch_prompt_selection_uniform", "module_effect", "intervention", "patch_prompt_selection_uniform", "patch_prompt_mass_preservation_pass", 1.0),
                        ("patch_prompt_selection_uniform", "module_effect", "intervention", "patch_prompt_selection_uniform", "delta_within_image_patch_dispersion_mean", -0.03),
                    ])
                with (diagnostics / "probe_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                    fields = [
                        "run_id", "session_id", "checkpoint_id", "probe_id", "selection_seed",
                        "probe_manifest_sha256", "split", "condition",
                        "domain", "entity_type", "entity_id", "metric", "value",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for probe_seed in (424242, 424243):
                        for condition, domain, entity_type, entity_id, metric, value in probe_rows:
                            writer.writerow({
                                "run_id": f"{stage}-{seed}",
                                "session_id": f"session-{stage}-{seed}",
                                "checkpoint_id": "final_epoch_0030",
                                "probe_id": f"probe-test-unseen-seed{probe_seed}",
                                "selection_seed": probe_seed,
                                "probe_manifest_sha256": f"probe-sha-{probe_seed}",
                                "split": "probe_test_unseen",
                                "condition": condition,
                                "domain": domain,
                                "entity_type": entity_type,
                                "entity_id": entity_id,
                                "metric": metric,
                                "value": value,
                            })
                (diagnostics / "comparability.json").write_text(
                    json.dumps({
                        "run_identity": {"sha256": f"run-{stage}-{seed}"},
                        "shared_condition_fingerprint": {"sha256": "shared"},
                    }),
                    encoding="utf-8",
                )
                runs.append(load_run(stage, run_dir))
        by_method = {stage: [row for row in runs if row["method"] == stage] for stage in stage_value}
        method_rows, method_payload = _method_summaries(by_method)
        assert method_rows and method_payload["A0"]["prompt_mechanisms_status"] == "not_applicable"
        assert any(
            "domain=classification" in key and "metric=top1" in key
            for key in by_method["A0"][0]["probe_context"]
        )
        assert not any(
            "domain=classification" in key and "metric=top1" in key
            for key in by_method["A0"][0]["mechanisms"]
        )
        assert any(
            "entity_type=layer" in key
            and "entity_id=layer_1" in key
            and "metric=prompt_grad_norm_epoch_mean" in key
            for key in by_method["A2"][0]["mechanisms"]
        )
        assert any(
            "domain=target_relevance_reference" in key
            and "entity_id=layer_3/cls_to_patch" in key
            for key in by_method["A1"][0]["mechanisms"]
        )
        assert any(
            "condition=prompt_patch_selection_uniform" in key
            and "metric=prompt_patch_mass_preservation_pass" in key
            for key in by_method["A1"][0]["mechanisms"]
        )
        assert any(
            "condition=patch_prompt_selection_uniform" in key
            and "metric=patch_prompt_mass_preservation_pass" in key
            for key in by_method["A1"][0]["mechanisms"]
        )
        assert any(
            "condition=patch_prompt_selection_uniform" in key
            and "metric=delta_within_image_patch_dispersion_mean" in key
            for key in by_method["A1"][0]["mechanisms"]
        )
        paired_rows, paired_payload = _paired_summaries(by_method, "A0")
        assert set(paired_payload) == {"A1-A0", "A2-A0", "A2-A1"}
        assert any(any(key.startswith("mechanism_delta::") for key in row) for row in paired_rows)
        gates = _gate_report(runs, paired_payload, [0, 1, 2])
        assert all(entry["status"] == "passed" for entry in gates.values()), gates
        gate3_evidence = gates["gate_3_architecture_matches_computation"]["evidence"]
        assert gate3_evidence["prompt_zero_attention_equivalence"]
        assert gate3_evidence["prompt_zero_cls_prompt_delta_by_layer"]
        assert gate3_evidence["prompt_zero_alignment_delta"]
        incomplete_gates = _gate_report(runs, paired_payload, [0, 1, 2, 3])
        assert all(entry["status"] == "insufficient_evidence" for entry in incomplete_gates.values())
        robustness = _probe_robustness_summaries(runs)
        assert set(robustness) == {"A0", "A1", "A2"}
        first_metric = next(iter(robustness["A0"].values()))
        assert first_metric["probe_selection_seed_count"] == 2
        assert first_metric["training_seed_count"] == 3
        three_seed_summary = _summary([1.0, 2.0, 3.0])
        assert three_seed_summary["ci95_method"] == "student_t"
        assert np.isclose(three_seed_summary["ci95"], 4.302652729911275 / np.sqrt(3.0))
        (
            trajectory_epoch_rows,
            trajectory_run_rows,
            trajectory_pair_epoch_rows,
            trajectory_pair_rows,
            trajectory_payload,
        ) = _generalization_trajectory_outputs(runs, by_method, "A0")
        assert len(trajectory_epoch_rows) == 54
        assert len(trajectory_run_rows) == 9
        assert all(row["valid"] for row in trajectory_run_rows)
        assert trajectory_pair_epoch_rows
        assert all(row["valid"] for row in trajectory_pair_rows)
        assert all(
            row.get("full_epoch_mean_delta_gzsl_h") is not None
            for row in trajectory_pair_rows
        )
        assert set(trajectory_payload["pairs"]) == {"A1-A0", "A2-A0", "A2-A1"}
        assert trajectory_payload["rule_contract"]["analysis_role"] == "diagnostic_only"
        assert not trajectory_payload["rule_contract"]["checkpoint_selection_allowed"]


def main():
    _validate_lossless_artifact_compaction()
    assert _stage("B1") == "B1"
    assert _stage("B2-direct-mean-nonconditional-control") == "B2"
    assert _is_gate_representation_metric(
        "condition=normal|domain=representation_geometry|metric=fisher_ratio"
    )
    assert _is_gate_representation_metric(
        "condition=normal|domain=representation_geometry|metric=fisher_trace_ratio"
    )
    _validate_prompt_patch_retrieval_diversity()
    _validate_streaming_probe_metrics()
    _validate_target_relevance_accumulator()
    _validate_prediction_transitions()
    _validate_prompt_content_and_layer_mechanism()
    _validate_affinity_diagnostic_cpu_offload()
    _validate_prompt_role_metric_rows()
    _validate_prompt_attention_path_interventions()
    _validate_attribute_concept_grounding_and_intervention()
    _validate_patch_semantic_transport_and_intervention()
    _validate_deep_prompt_parameter_health()
    _validate_streaming_fixed_probe_metrics()
    _validate_comparability_identity()
    _validate_fixed_probe_seed_dispatch()
    _validate_probe_runtime_timing()
    _validate_probe_evidence_roles()
    _validate_fixed_probe_semantic_bundle()
    _validate_generalization_trajectory_rules()
    _validate_low_cost_trajectory_extensions()
    _validate_cross_seed_mechanism_summary()
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
    assert class_error["arrays"]["per_class_accuracy"].dtype == np.float32
    assert class_error["arrays"]["class_true_margin"].dtype == np.float32
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
    assert calibration_profile["gamma_grid"].dtype == np.float32
    assert calibration_profile["seen_at_gamma"].dtype == np.float32
    assert calibration_profile["unseen_at_gamma"].dtype == np.float32
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
    assert "arrays" not in effect
    assert effect["per_class"]
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
    repeated_probe_manifest = build_probe_manifest(
        SyntheticProbeDataset(),
        split="synthetic",
        per_class=1,
        max_samples=3,
        selection_seed=17,
        candidate_class_ids=[0, 1, 2, 3],
    )
    assert repeated_probe_manifest["manifest_sha256"] == full_probe_manifest["manifest_sha256"]
    full_probe_validity = validate_probe_manifest(full_probe_manifest)
    assert full_probe_validity["valid"]
    assert all(full_probe_validity["checks"].values())
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
    capped_probe_validity = validate_probe_manifest(capped_probe_manifest)
    assert not capped_probe_validity["valid"]
    assert not capped_probe_validity["checks"]["per_class_quota_satisfied"]
    assert not capped_probe_validity["checks"]["max_samples_policy_satisfied"]
    sparse_dataset = SyntheticProbeDataset()
    sparse_dataset._imdb = [row for row in SyntheticProbeDataset._imdb if row["sample_id"].endswith("-a")]
    sparse_probe_manifest = build_probe_manifest(
        sparse_dataset,
        split="synthetic",
        per_class=2,
        max_samples=6,
        selection_seed=17,
        candidate_class_ids=[0, 1, 2],
    )
    assert sparse_probe_manifest["per_class_available"] == {"0": 1, "1": 1, "2": 1}
    assert not sparse_probe_manifest["per_class_quota_satisfied"]
    assert not validate_probe_manifest(sparse_probe_manifest)["valid"]

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
        cfg.SOLVER.TOTAL_EPOCH = 2
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
            dataset=dataset,
        )
        diagnostics.record_eval(
            epoch=1,
            split="test_unseen",
            scores=unseen_scores,
            targets_local=unseen_targets,
            dataset=dataset,
        )
        diagnostics.record_calibration(1)
        assert not diagnostics.calibration_pending
        assert not (Path(temp_dir) / "diagnostics/eval_cache").exists()
        diagnostics.append_probe_metrics([{
            "run_id": manager.run_id,
            "session_id": manager.session_id,
            "checkpoint_id": "final_epoch_0002",
            "probe_id": "synthetic",
            "selection_seed": 17,
            "probe_manifest_sha256": "synthetic-sha",
            "split": "probe_test_unseen",
            "condition": "mismatched_semantic_permutation",
            "domain": "semantic_prototype_intervention",
            "entity_type": "intervention",
            "entity_id": "semantic_mismatch",
            "metric": "mismatched_semantic_effect_pass",
            "value": 1.0,
        }])
        with (Path(temp_dir) / "diagnostics/probe_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            probe_rows = list(csv.DictReader(handle))
        assert probe_rows[0]["condition"] == "mismatched_semantic_permutation"
        assert probe_rows[0]["selection_seed"] == "17"
        assert probe_rows[0]["probe_manifest_sha256"] == "synthetic-sha"
        diagnostics.record_eval(
            epoch=2,
            split="test_seen",
            scores=seen_scores,
            targets_local=seen_targets,
            dataset=dataset,
        )
        assert not (Path(temp_dir) / "diagnostics/eval_cache").exists()
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
