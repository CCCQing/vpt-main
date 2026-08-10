#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.prompting.prompt_distribution import PreViTPromptDistributor
from src.models.vit_backbones.vit import Attention
from src.monitoring.module_effect import (
    both_prompt_zero_intervention,
    domain_prompt_zero_intervention,
    instance_prompt_swap_intervention,
    instance_prompt_zero_intervention,
    prompt_value_zero_intervention,
    relevance_edge_delete_intervention,
)
from src.monitoring.prompt_analysis import (
    PairedFlipAccumulator,
    PromptContentSlotAccumulator,
    PromptSourceDecompositionAccumulator,
    build_relevance_deletion_masks,
    match_prompt_role_vectors,
)
from src.tools.match_prompt_roles import match_profiles


def _attention_module() -> Attention:
    config = SimpleNamespace(
        hidden_size=8,
        transformer={"num_heads": 2, "attention_dropout_rate": 0.0},
    )
    return Attention(config, vis=True)


def test_relevance_deletion() -> None:
    relevance = torch.arange(2 * 2 * 6 * 6, dtype=torch.float32).reshape(
        2, 2, 6, 6
    )
    relevance = torch.where(
        torch.arange(relevance.numel()).reshape_as(relevance) % 2 == 0,
        relevance + 1.0,
        -(relevance + 1.0),
    )
    masks, metadata = build_relevance_deletion_masks(
        relevance,
        prompt_length=2,
        semantic_length=0,
        paths=(
            "cls_to_prompt",
            "prompt_to_cls",
            "prompt_to_patch",
            "patch_to_prompt",
        ),
        conditions=("positive", "negative", "random"),
        fraction=0.3,
        random_seed=7,
    )
    counts = [int(masks[name].sum().item()) for name in masks]
    assert counts[0] == counts[1] == counts[2] > 0
    assert all(metadata[name]["selected_edge_count"] == counts[0] for name in masks)

    module = _attention_module()
    probabilities = torch.rand(1, 2, 5, 5)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    module._prompt_path_intervention = {
        "mode": "relevance_edge_delete",
        "layer_index": 0,
        "target_layer": 0,
        "deletion_mask": torch.ones_like(probabilities, dtype=torch.bool),
    }
    changed = module._apply_prompt_path_intervention(
        probabilities,
        prompt_length=1,
        semantic_length=0,
        hidden_states=torch.zeros(1, 5, 8),
    )
    assert torch.allclose(
        changed.sum(dim=-1), probabilities.sum(dim=-1), atol=1e-6
    )
    assert bool((changed > 0).any(dim=-1).all())
    stats = module._last_prompt_path_intervention_stats
    assert float(stats["relevance_delete_row_mass_abs_error"].max()) < 1e-6

    holder = torch.nn.Module()
    holder.attention = module
    deletion_mask = torch.zeros(1, 2, 5, 5, dtype=torch.bool)
    deletion_mask[:, :, 0, 1] = True
    with relevance_edge_delete_intervention(
        holder,
        target_layer=0,
        deletion_mask=deletion_mask,
    ):
        assert module._last_prompt_path_intervention_stats is None
        module(torch.randn(1, 5, 8), prompt_length=1)
        assert module._last_prompt_path_intervention_stats is not None


def test_prompt_value_zero() -> None:
    module = _attention_module()
    probabilities = torch.rand(2, 2, 5, 5)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    values = torch.randn(2, 2, 5, 4)
    context = torch.matmul(probabilities, values)
    reference_probabilities = probabilities.clone()
    module._prompt_path_intervention = {
        "mode": "prompt_value_zero",
        "layer_index": 0,
    }
    changed = module._apply_prompt_value_intervention(
        context,
        probabilities,
        values,
        prompt_length=2,
        semantic_length=0,
    )
    expected = context - torch.matmul(
        probabilities[:, :, :, 1:3], values[:, :, 1:3, :]
    )
    assert torch.allclose(changed, expected, atol=1e-6)
    assert torch.equal(probabilities, reference_probabilities)

    holder = torch.nn.Module()
    holder.attention = module
    module._prompt_path_intervention = None
    hidden = torch.randn(2, 5, 8)
    module.eval()
    baseline_output, baseline_weights, _ = module(
        hidden, prompt_length=2
    )
    with prompt_value_zero_intervention(holder):
        changed_output, changed_weights, _ = module(
            hidden, prompt_length=2
        )
    assert torch.allclose(changed_weights, baseline_weights, atol=1e-7)
    assert not torch.allclose(changed_output, baseline_output)


def test_prompt_distribution_interventions() -> None:
    distributor = PreViTPromptDistributor(
        dim=768,
        prompt_len=4,
        hidden_dim=4,
        source="token_mlp",
        instance_tokens=2,
        domain_tokens=2,
        eval_sample_mode="mean",
    )
    holder = torch.nn.Module()
    holder.distributor = distributor
    with torch.no_grad():
        distributor.domain_prompt.fill_(2.0)
    mu = torch.stack((torch.ones(768), torch.full((768,), 3.0)))
    logvar = torch.zeros_like(mu)
    baseline, _ = distributor.prompt_from_distribution(mu, logvar)
    with instance_prompt_zero_intervention(holder):
        changed, _ = distributor.prompt_from_distribution(mu, logvar)
        assert float(changed[:, :2].detach().abs().sum().item()) == 0.0
        assert torch.equal(changed[:, 2:], baseline[:, 2:])
    with domain_prompt_zero_intervention(holder):
        changed, _ = distributor.prompt_from_distribution(mu, logvar)
        assert torch.equal(changed[:, :2], baseline[:, :2])
        assert float(changed[:, 2:].detach().abs().sum().item()) == 0.0
    with both_prompt_zero_intervention(holder):
        changed, _ = distributor.prompt_from_distribution(mu, logvar)
        assert float(changed.detach().abs().sum().item()) == 0.0
    with instance_prompt_swap_intervention(
        holder, permutation=torch.tensor([1, 0])
    ):
        changed, _ = distributor.prompt_from_distribution(mu, logvar)
        assert torch.equal(changed[0, :2], baseline[1, :2])
        assert torch.equal(changed[1, :2], baseline[0, :2])
        assert torch.equal(changed[:, 2:], baseline[:, 2:])
    restored, _ = distributor.prompt_from_distribution(mu, logvar)
    assert torch.equal(restored, baseline)


def test_source_decomposition() -> None:
    accumulator = PromptSourceDecompositionAccumulator(
        instance_tokens=1, domain_tokens=1
    )
    targets = torch.tensor([0, 0, 0, 1, 1, 1])
    instance = torch.tensor(
        [[[-2.1, 1.0]], [[-2.0, 1.1]], [[-1.9, 0.9]],
         [[1.9, 1.0]], [[2.0, 0.9]], [[2.1, 1.1]]]
    )
    domain = torch.full((6, 1, 2), 0.5)
    cls_repr = instance.squeeze(1) * 1.5
    true_margin = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    accumulator.update_raw(
        {"instance_prompt": instance, "domain_prompt": domain},
        targets,
        cls_repr=cls_repr,
        true_margin=true_margin,
    )
    contextualized = torch.cat(
        (
            torch.zeros(6, 1, 2),
            instance,
            domain + instance * 0.1,
            torch.zeros(6, 4, 2),
        ),
        dim=1,
    )
    accumulator.update_contextualized(
        contextualized,
        targets,
        prompt_length=2,
        semantic_length=0,
        cls_repr=cls_repr,
        true_margin=true_margin,
    )
    result = accumulator.finalize()
    metrics = result["metrics"]
    assert metrics["raw_instance_prompt_class_component_variance_share"] > 0.95
    assert metrics["raw_instance_prompt_instance_residual_variance_share"] < 0.05
    assert metrics["raw_domain_prompt_replication_pass"] == 1.0
    assert "raw_instance_prompt_norm_vs_cls_norm_pearson" in metrics
    assert "contextualized_domain_prompt_norm_vs_true_margin_pearson" in metrics
    assert result["contextualized_domain_observed"]


def test_content_and_slot_health() -> None:
    accumulator = PromptContentSlotAccumulator(
        prompt_length=4,
        selected_layers=(0,),
        redundancy_cosine=0.9,
        opposition_cosine=-0.9,
        cancellation_ratio=0.1,
    )
    attention = torch.zeros(1, 1, 9, 9)
    attention[:, :, 0, 1:5] = torch.tensor([0.6, 0.2, 0.15, 0.05])
    content = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]]
    )
    accumulator.update(
        [attention],
        [{
            "_prompt_patch_content_vector": content,
            "_prompt_patch_pre_output_content_by_head": content.unsqueeze(1),
        }],
    )
    result = accumulator.finalize()
    metrics = result["by_layer"][0]
    assert metrics["prompt_content_redundancy_candidate_ratio"] > 0.0
    assert metrics["prompt_content_opposition_candidate_ratio"] > 0.0
    assert metrics["prompt_content_cancellation_candidate_ratio"] > 0.0
    assert result["by_layer_and_head"][0][0][
        "prompt_content_redundancy_candidate_ratio"
    ] > 0.0
    assert len(result["by_layer_and_prompt"][0]) == 4


def test_paired_flip() -> None:
    accumulator = PairedFlipAccumulator(
        prompt_length=2,
        semantic_length=0,
        topk=2,
        selected_layers=(0,),
    )
    normal_attention = torch.zeros(1, 1, 7, 7)
    normal_attention[:, :, 0, 1:3] = torch.tensor([0.8, 0.2])
    normal_map = torch.tensor(
        [[[[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]]]
    ).squeeze(2)
    normal_attention[:, :, 1:3, 3:7] = normal_map
    flipped_attention = normal_attention.clone()
    flipped_attention[:, :, 1:3, 3:7] = normal_map.reshape(
        1, 1, 2, 2, 2
    ).flip(-1).reshape(1, 1, 2, 4)
    normal_av = normal_map.mean(dim=1)
    flipped_av = normal_av.reshape(1, 2, 2, 2).flip(-1).reshape(1, 2, 4)
    accumulator.update(
        [normal_attention],
        [{"_prompt_patch_pre_output_av_magnitude": normal_av}],
        [flipped_attention],
        [{"_prompt_patch_pre_output_av_magnitude": flipped_av}],
        normal_predictions=torch.tensor([1]),
        flipped_predictions=torch.tensor([1]),
    )
    result = accumulator.finalize()
    metrics = result["by_layer"][0]
    assert abs(metrics["prompt_patch_map_cosine"] - 1.0) < 1e-6
    assert abs(metrics["prompt_patch_topk_overlap"] - 1.0) < 1e-6
    assert abs(metrics["prompt_av_map_cosine"] - 1.0) < 1e-6
    assert abs(metrics["assignment_top1_consistency"] - 1.0) < 1e-6


def test_role_matching() -> None:
    reference = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    candidate = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    result = match_prompt_role_vectors(
        reference,
        candidate,
        reference_types=("instance", "instance", "domain"),
        candidate_types=("instance", "instance", "domain"),
        max_cost=0.2,
    )
    pairs = {
        item["reference_index"]: item["candidate_index"]
        for item in result["pairs"]
    }
    assert pairs == {0: 1, 1: 0}
    assert result["unmatched_or_dead_prompt_ratio"] > 0.0
    assert result["matched_role_cosine_mean"] == 1.0


def test_profile_matching_layer_constraint() -> None:
    role = {
        "class_local_ids": [0, 1],
        "prompt_vectors": [[1.0, 0.0], [0.0, 1.0]],
    }
    swapped = {
        "class_local_ids": [0, 1],
        "prompt_vectors": [[0.0, 1.0], [1.0, 0.0]],
    }
    reference = {
        "format": "prompt_role_profile_v1",
        "split": "probe_test_unseen",
        "probe_id": "probe-7",
        "probe_manifest_sha256": "abc",
        "selection_seed": 7,
        "prompt_types": ["visual", "visual"],
        "checkpoint": {"source_run_id": "left", "checkpoint_id": "final"},
        "layers": {
            "0": {"cls_consumption": role},
            "1": {"cls_consumption": role},
        },
    }
    candidate = {
        "format": "prompt_role_profile_v1",
        "split": "probe_test_unseen",
        "probe_id": "probe-7",
        "probe_manifest_sha256": "abc",
        "selection_seed": 7,
        "prompt_types": ["visual", "visual"],
        "checkpoint": {"source_run_id": "right", "checkpoint_id": "final"},
        "layers": {
            "0": {"cls_consumption": swapped},
            "2": {"cls_consumption": swapped},
        },
    }
    result = match_profiles(reference, candidate, max_cost=0.2)
    assert set(result["by_layer_role"]) == {"layer_0/cls_consumption"}
    assert result["summary"]["matched_role_cosine_mean"] == 1.0


def main() -> None:
    tests = (
        test_relevance_deletion,
        test_prompt_value_zero,
        test_prompt_distribution_interventions,
        test_source_decomposition,
        test_content_and_slot_health,
        test_paired_flip,
        test_role_matching,
        test_profile_matching_layer_constraint,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
