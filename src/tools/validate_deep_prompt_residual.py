#!/usr/bin/env python3
"""Focused contracts for deterministic direct-mean Deep Prompt residual."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.prompting.prompt_distribution import (
    MeanConditionedDeepPromptResidual,
    PreViTPromptDistributor,
)
from src.monitoring.adapters import deep_prompt_residual_metrics
from src.monitoring.deep_prompt_residual_experiments import (
    deterministic_donor_groups,
    deterministic_donor_indices,
    layer_scales,
    residual_static_geometry,
    summarize_geometry,
)
from src.monitoring.module_effect import (
    deep_prompt_residual_common_only_intervention,
    deep_prompt_residual_layer_scales_intervention,
    deep_prompt_residual_replace_intervention,
    deep_prompt_residual_role_only_intervention,
    deep_prompt_residual_role_permuted_intervention,
    deep_prompt_residual_swap_intervention,
    deep_prompt_residual_zero_intervention,
)
from src.configs.config import get_cfg
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.vit_models import ViT
from src.solver.losses import build_loss
from src.solver.optimizer import make_optimizer
from src.tools.search_plans.b_series.summarize_b_series_replays import (
    nested_scientific_summary,
)
from src.tools.search_plans.b_series.run_b_series_training import (
    _validate_e7_preconditions,
)


class _ResidualHolder(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.deep_prompt_residual = module


def validate_zero_gate_and_gradient() -> None:
    torch.manual_seed(11)
    module = MeanConditionedDeepPromptResidual(
        dim=8,
        prompt_len=3,
        num_layers=2,
        gate_init=0.0,
    )
    latent = torch.randn(5, 8)
    delta, trace = module.forward_layer(latent, 1)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(
        trace["raw_delta"], latent[:, None, :].expand(-1, 3, -1)
    )
    assert float(trace["raw_delta"].abs().sum().item()) > 0.0
    base = torch.randn_like(delta)
    assert torch.equal(base + delta, base)
    (base + delta).square().mean().backward()
    assert module.layer_gate.grad is not None
    assert float(module.layer_gate.grad[1].abs().item()) > 0.0


def validate_controls_and_metrics() -> None:
    torch.manual_seed(17)
    module = MeanConditionedDeepPromptResidual(
        dim=8,
        prompt_len=2,
        num_layers=1,
        gate_init=1.0,
    )
    holder = _ResidualHolder(module)
    latent = torch.randn(4, 8, requires_grad=True)
    normal, normal_trace = module.forward_layer(latent, 0)
    with deep_prompt_residual_zero_intervention(holder):
        zeroed, _ = module.forward_layer(latent, 0)
    assert torch.equal(zeroed, torch.zeros_like(zeroed))
    permutation = torch.tensor([1, 2, 3, 0])
    with deep_prompt_residual_swap_intervention(
        holder, permutation=permutation
    ):
        swapped, swapped_trace = module.forward_layer(latent, 0)
    assert not torch.allclose(normal, swapped)
    assert float(swapped_trace["mean_swap_fixed_point_ratio"].sum().item()) == 0.0
    metrics = deep_prompt_residual_metrics(
        [
            {
                **normal_trace,
                "base_prompt": torch.ones_like(normal),
                "injected_prompt": torch.ones_like(normal) + normal,
            }
        ]
    )
    assert metrics["layer_0.applied_delta_norm"] > 0.0
    assert metrics["layer_0.applied_delta_to_base_ratio"] > 0.0
    assert metrics["layer_0.raw_delta_slot_variance"] == 0.0
    normal.square().mean().backward()
    assert latent.grad is not None
    assert float(latent.grad.abs().sum().item()) > 0.0


def validate_experiment_interventions_and_geometry() -> None:
    torch.manual_seed(19)
    module = MeanConditionedDeepPromptResidual(
        dim=8,
        prompt_len=3,
        num_layers=4,
        gate_init=1.0,
    )
    holder = _ResidualHolder(module)
    latent = torch.randn(5, 8)
    base = torch.randn(5, 3, 8)
    normal, normal_trace = module.forward_layer(latent, 2)
    with deep_prompt_residual_layer_scales_intervention(
        holder, scales=layer_scales(4, selected_scale=0.5)
    ):
        half, half_trace = module.forward_layer(latent, 2)
    assert torch.allclose(half, normal * 0.5)
    assert torch.allclose(half_trace["runtime_scale"], torch.full((5,), 0.5))
    replacement = torch.flip(latent, dims=(0,))
    with deep_prompt_residual_replace_intervention(
        holder, replacement=replacement
    ):
        replaced, replace_trace = module.forward_layer(latent, 2)
    assert torch.equal(replaced[:, 0], replacement)
    assert float(replace_trace["mean_replace_distance"].sum().item()) > 0.0
    geometry = residual_static_geometry(
        [
            {
                **normal_trace,
                "base_prompt": base,
            }
        ]
    )
    assert tuple(geometry["residual_static_cosine"].shape) == (5, 1)
    assert np.allclose(geometry["slot_shift_effective_rank"], 1.0)
    rank_one = torch.randn(32, 1, 768).repeat(1, 16, 1) * 1.0e4
    stress_geometry = residual_static_geometry(
        [
            {
                "layer_id": 0,
                "base_prompt": torch.randn_like(rank_one),
                "applied_delta": rank_one,
            }
        ]
    )
    assert np.isfinite(stress_geometry["slot_shift_effective_rank"]).all()
    assert np.allclose(stress_geometry["slot_shift_effective_rank"], 1.0)
    grouped = summarize_geometry(
        geometry,
        groups={"first_two": np.asarray([1, 1, 0, 0, 0], dtype=bool)},
    )
    assert grouped["groups"]["first_two"]["sample_count"] == 2


def validate_slot_and_sample_gate_modes() -> None:
    torch.manual_seed(21)
    latent = torch.randn(6, 10)
    shared = MeanConditionedDeepPromptResidual(
        dim=10,
        prompt_len=4,
        num_layers=3,
        gate_init=1.0,
    )
    assert set(dict(shared.named_parameters())) == {"layer_gate"}
    slot = MeanConditionedDeepPromptResidual(
        dim=10,
        prompt_len=4,
        num_layers=3,
        gate_init=1.0,
        content_mode="slot_low_rank",
        slot_rank=3,
    )
    slot_delta, slot_trace = slot.forward_layer(latent, 1)
    assert tuple(slot_delta.shape) == (6, 4, 10)
    assert float(slot_delta.var(dim=1, unbiased=False).sum().item()) > 0.0
    slot_metrics = deep_prompt_residual_metrics(
        [{**slot_trace, "base_prompt": torch.ones_like(slot_delta)}]
    )
    assert slot_metrics["layer_1.raw_delta_slot_effective_rank"] > 1.0

    slot_scalar = MeanConditionedDeepPromptResidual(
        dim=10,
        prompt_len=4,
        num_layers=3,
        gate_init=1.0,
        content_mode="slot_scalar",
    )
    scalar_delta, scalar_trace = slot_scalar.forward_layer(latent, 1)
    expected_shared = latent[:, None, :].expand_as(scalar_delta)
    assert torch.allclose(scalar_delta, expected_shared)
    assert torch.allclose(
        scalar_trace["slot_scalar_coefficients"],
        torch.ones(6, 4),
    )
    scalar_metrics = deep_prompt_residual_metrics(
        [{**scalar_trace, "base_prompt": torch.ones_like(scalar_delta)}]
    )
    assert scalar_metrics["layer_1.slot_scalar_coefficient_mean"] == 1.0
    assert scalar_metrics["layer_1.slot_scalar_coefficient_std"] == 0.0
    assert abs(scalar_metrics["layer_1.slot_scalar_effective_count"] - 4.0) < 1.0e-6
    assert abs(scalar_metrics["layer_1.slot_scalar_effective_ratio"] - 1.0) < 1.0e-6
    assert scalar_metrics["layer_1.slot_scalar_between_instance_variance"] == 0.0
    scalar_trace["slot_scalar_coefficients"].sum().backward()
    assert slot_scalar.slot_coefficients[1].weight.grad is not None

    conditional = MeanConditionedDeepPromptResidual(
        dim=10,
        prompt_len=4,
        num_layers=3,
        gate_init=1.0,
        sample_gate_mode="shared",
        sample_gate_input="residual_source",
        sample_gate_init=0.8,
    )
    fixed = MeanConditionedDeepPromptResidual(
        dim=10,
        prompt_len=4,
        num_layers=3,
        gate_init=1.0,
        sample_gate_mode="shared",
        sample_gate_input="constant",
        sample_gate_init=0.8,
    )
    assert {
        name: tuple(value.shape) for name, value in conditional.named_parameters()
    } == {name: tuple(value.shape) for name, value in fixed.named_parameters()}
    _, conditional_trace = conditional.forward_layer(latent, 0)
    _, fixed_trace = fixed.forward_layer(latent, 0)
    assert torch.allclose(
        conditional_trace["sample_gate"], torch.full((6,), 0.8), atol=1e-6
    )
    assert torch.allclose(
        fixed_trace["sample_gate"], torch.full((6,), 0.8), atol=1e-6
    )


def validate_common_role_interventions() -> None:
    torch.manual_seed(22)
    module = MeanConditionedDeepPromptResidual(
        dim=9,
        prompt_len=4,
        num_layers=2,
        gate_init=1.0,
        content_mode="slot_low_rank",
        slot_rank=3,
    )
    holder = _ResidualHolder(module)
    latent = torch.randn(5, 9)
    normal, trace = module.forward_layer(latent, 1)
    common = trace["common_component"]
    role = trace["role_component"]
    assert torch.allclose(common + role, trace["raw_delta"])
    assert torch.allclose(role.mean(dim=1), torch.zeros_like(role.mean(dim=1)), atol=1.0e-6)
    with deep_prompt_residual_common_only_intervention(holder):
        common_only, common_trace = module.forward_layer(latent, 1)
    with deep_prompt_residual_role_only_intervention(holder):
        role_only, role_trace = module.forward_layer(latent, 1)
    permutation = torch.tensor([2, 0, 3, 1])
    with deep_prompt_residual_role_permuted_intervention(
        holder, permutation=permutation
    ):
        permuted, permuted_trace = module.forward_layer(latent, 1)
    assert torch.allclose(common_only, common)
    assert torch.allclose(role_only, role)
    assert not torch.allclose(permuted, normal)
    assert torch.allclose(
        permuted_trace["raw_delta"].square().sum(dim=(-2, -1)),
        trace["raw_delta"].square().sum(dim=(-2, -1)),
        atol=1.0e-5,
    )
    assert torch.equal(
        common_trace["content_intervention_applied"], torch.ones(5)
    )
    assert torch.equal(role_trace["content_intervention_applied"], torch.ones(5))


def validate_donor_contracts() -> None:
    sample_ids = ["a0", "a1", "b0", "b1", "c0", "c1"]
    labels = [0, 0, 1, 1, 2, 2]
    same = deterministic_donor_indices(
        sample_ids, labels, relation="same_class", seed=31
    )
    different = deterministic_donor_indices(
        sample_ids, labels, relation="different_class", seed=31
    )
    label_array = np.asarray(labels)
    assert np.all(same != np.arange(len(sample_ids)))
    assert np.all(label_array[same] == label_array)
    assert np.all(label_array[different] != label_array)

    dense_ids = ["a{}".format(index) for index in range(6)] + [
        "b{}".format(index) for index in range(6)
    ]
    dense_labels = [0] * 6 + [1] * 6
    k4 = deterministic_donor_groups(
        dense_ids, dense_labels, relation="same_class", seed=31, k=4
    )
    loo = deterministic_donor_groups(
        dense_ids, dense_labels, relation="same_class", seed=31, k=None
    )
    assert all(group.size == 4 and np.unique(group).size == 4 for group in k4)
    assert all(group.size == 5 and np.unique(group).size == 5 for group in loo)


def validate_bounded_deep_residual_contract() -> None:
    torch.manual_seed(37)
    module = MeanConditionedDeepPromptResidual(
        dim=8,
        prompt_len=3,
        num_layers=12,
        amplitude_mode="bounded_ratio",
        active_layers=(8, 9, 10, 11),
        bounded_max_ratio=0.25,
        bounded_init_ratio=0.125,
    )
    latent = torch.randn(5, 8, requires_grad=True)
    base = torch.randn(5, 3, 8)
    inactive, inactive_trace = module.forward_layer(latent, 7, base_prompt=base)
    active, active_trace = module.forward_layer(latent, 8, base_prompt=base)
    assert torch.equal(inactive, torch.zeros_like(inactive))
    assert torch.equal(
        inactive_trace["applied_ratio"],
        torch.zeros_like(inactive_trace["applied_ratio"]),
    )
    assert torch.allclose(
        active_trace["applied_ratio"],
        torch.full((5,), 0.125),
        atol=2.0e-6,
    )
    assert float(active_trace["budget_exceed"].sum().item()) == 0.0
    active.square().mean().backward()
    assert module.layer_gate.grad is not None
    assert float(module.layer_gate.grad[8].abs().item()) > 0.0


class _B3ConsistencyHolder(torch.nn.Module):
    def __init__(self, class_count: int, dim: int):
        super().__init__()
        self.feat_dim = int(dim)
        self.b3_class_consistency_enabled = True
        self.register_buffer("b3_class_centers", torch.zeros(class_count, dim))
        self.register_buffer("b3_class_center_counts", torch.zeros(class_count))
        self._runtime_prompt_distribution_stats = None

    compute_b3_class_consistency = ViT.compute_b3_class_consistency

    def get_runtime_prompt_distribution_stats(self):
        return self._runtime_prompt_distribution_stats


def validate_b3_class_consistency_state() -> None:
    torch.manual_seed(41)
    holder = _B3ConsistencyHolder(class_count=4, dim=6)
    mu = torch.randn(8, 6, requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    result = holder.compute_b3_class_consistency(
        mu,
        groups,
        momentum=0.9,
        margin=0.2,
        update=True,
    )
    loss = result["intra_loss"] + result["inter_loss"]
    loss.backward()
    assert mu.grad is not None and float(mu.grad.abs().sum().item()) > 0.0
    assert torch.equal(holder.b3_class_center_counts, torch.full((4,), 2.0))
    assert torch.isfinite(holder.b3_class_centers).all()

    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/b_series_experiments/B3-R2I-class-consistent.yaml"
    )
    criterion = build_loss(cfg)
    second_mu = torch.randn(8, 6, requires_grad=True)
    holder._runtime_prompt_distribution_stats = {"mu": second_mu}
    logits = torch.randn(8, 4, requires_grad=True)
    total = criterion(
        logits,
        groups,
        [1.0] * 4,
        kwargs={
            "model": holder,
            "targets_global": groups,
            "sample_ids": ["sample{}".format(index) for index in range(8)],
            "is_train": True,
        },
    )
    total.backward()
    assert second_mu.grad is not None
    assert "b3_intra_loss.weighted" in criterion._last_loss_stats


class _OptimizerMultiplierHolder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = torch.nn.Parameter(torch.ones(2, 3))
        self.deep_prompt_embeddings = torch.nn.Parameter(torch.ones(2, 2, 3))
        self.residual_projection = torch.nn.Linear(3, 3, bias=False)
        self.r_similarity_head = torch.nn.Linear(3, 2, bias=False)


def validate_b3_optimizer_multipliers() -> None:
    cfg = get_cfg()
    cfg.defrost()
    cfg.SOLVER.OPTIMIZER = "adamw"
    cfg.SOLVER.BASE_LR = 6.0e-4
    cfg.SOLVER.STATIC_PROMPT_LR_MULTIPLIER = 0.1
    cfg.SOLVER.CLASSIFIER_LR_MULTIPLIER = 1.0
    cfg.freeze()
    model = _OptimizerMultiplierHolder()
    optimizer = make_optimizer([model], cfg.SOLVER)
    lr_by_parameter = {
        id(parameter): float(group["lr"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert abs(lr_by_parameter[id(model.prompt_embeddings)] - 6.0e-5) < 1.0e-12
    assert abs(lr_by_parameter[id(model.deep_prompt_embeddings)] - 6.0e-5) < 1.0e-12
    assert abs(lr_by_parameter[id(model.residual_projection.weight)] - 6.0e-4) < 1.0e-12
    assert abs(lr_by_parameter[id(model.r_similarity_head.weight)] - 6.0e-4) < 1.0e-12


def validate_parameter_only_provider_is_rng_neutral() -> None:
    torch.manual_seed(23)
    provider = PreViTPromptDistributor(
        dim=768,
        prompt_len=2,
        hidden_dim=8,
        source="token_mlp",
        instance_tokens=1,
        domain_tokens=1,
        eval_sample_mode="mean",
        use_slot_embed=False,
    )
    tokens = torch.randn(3, 4, 768)
    before = torch.random.get_rng_state().clone()
    stats = provider.distribution_parameters(vit_image_tokens=tokens)
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after)
    assert tuple(stats["mu"].shape) == (3, 768)
    assert stats["sampling_performed"] is False


def validate_nonconditional_control() -> None:
    torch.manual_seed(29)
    kwargs = dict(
        dim=768,
        prompt_len=2,
        hidden_dim=8,
        instance_tokens=2,
        domain_tokens=0,
        eval_sample_mode="mean",
        use_slot_embed=False,
    )
    provider = PreViTPromptDistributor(
        source="vit_cls_prepass_constant",
        **kwargs,
    )
    conditional_provider = PreViTPromptDistributor(
        source="vit_cls_prepass",
        **kwargs,
    )
    assert {
        name: tuple(parameter.shape)
        for name, parameter in provider.named_parameters()
    } == {
        name: tuple(parameter.shape)
        for name, parameter in conditional_provider.named_parameters()
    }
    first = provider.distribution_parameters(vit_cls=torch.randn(3, 768))
    second = provider.distribution_parameters(vit_cls=torch.randn(3, 768) * 100.0)
    assert torch.equal(first["visual_input"], torch.ones_like(first["visual_input"]))
    assert torch.equal(first["mu"], second["mu"])
    assert torch.equal(first["mu"][0], first["mu"][1])
    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/baseline_rebuild/B-02-direct-mean-nonconditional-control.yaml"
    )
    assert cfg.MODEL.PROMPT.DISTRIBUTOR.SOURCE == "vit_cls_prepass_constant"
    assert (
        cfg.MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.ARCHITECTURE_ID
        == "B2-direct-mean-nonconditional-control"
    )


def validate_direct_prepass_sources() -> None:
    kwargs = dict(
        dim=768,
        prompt_len=2,
        hidden_dim=8,
        instance_tokens=2,
        domain_tokens=0,
        eval_sample_mode="mean",
        use_slot_embed=False,
    )
    direct = PreViTPromptDistributor(
        source="vit_cls_prepass_direct",
        **kwargs,
    )
    constant = PreViTPromptDistributor(
        source="vit_cls_prepass_direct_constant",
        **kwargs,
    )
    fixed_direction = PreViTPromptDistributor(
        source="vit_cls_prepass_direct_fixed_direction",
        **kwargs,
    )
    assert direct.stats_head is None
    assert constant.stats_head is None
    assert fixed_direction.stats_head is None
    cls = torch.randn(3, 768)
    direct_stats = direct.distribution_parameters(vit_cls=cls)
    expected = torch.nn.functional.normalize(cls, p=2.0, dim=-1)
    assert torch.allclose(direct_stats["mu"], expected)
    assert torch.equal(direct_stats["logvar"], torch.zeros_like(expected))
    assert not torch.equal(direct_stats["mu"][0], direct_stats["mu"][1])
    first = constant.distribution_parameters(vit_cls=cls)
    second = constant.distribution_parameters(vit_cls=cls * 17.0)
    assert torch.equal(first["mu"], second["mu"])
    assert torch.equal(first["mu"][0], first["mu"][1])
    fixed_first = fixed_direction.distribution_parameters(vit_cls=cls)
    fixed_second = fixed_direction.distribution_parameters(vit_cls=cls * 17.0)
    assert torch.equal(fixed_first["mu"], fixed_second["mu"])
    assert torch.equal(fixed_first["mu"][0], fixed_first["mu"][1])
    assert torch.allclose(
        fixed_first["mu"].float().mean(dim=-1),
        torch.zeros(int(cls.shape[0])),
        atol=1.0e-7,
    )
    assert bool((fixed_first["mu"] > 0).any())
    assert bool((fixed_first["mu"] < 0).any())
    assert not torch.equal(fixed_first["mu"], first["mu"])


def validate_full_vit_config_and_trace() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/baseline_rebuild/B-01-direct-mean-residual.yaml"
    )
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(31)
    model = ViT(cfg, load_pretrain=False).to(device).eval()
    named = dict(model.named_parameters())
    residual_parameter_names = {
        name for name in named if "deep_prompt_residual" in name
    }
    assert residual_parameter_names == {
        "enc.transformer.deep_prompt_residual.layer_gate"
    }
    assert any(
        parameter.requires_grad
        for name, parameter in named.items()
        if "prompt_init_provider.stats_head" in name
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in named.items()
        if "prompt_init_provider.domain_prompt" in name
    )
    image = torch.randn(1, 3, int(cfg.DATA.CROPSIZE), int(cfg.DATA.CROPSIZE), device=device)
    with torch.no_grad():
        reference, _ = model(image, return_feature=True)
        trace = model.enc.transformer._last_layer_prompt_trace
        assert len(trace) == 12
        assert [int(item["layer_id"]) for item in trace] == list(range(12))
        assert all(torch.is_tensor(item.get("contextualized_prompt")) for item in trace)
        assert all(
            torch.equal(
                item["raw_delta"],
                item["source_mu"][:, None, :].expand_as(item["raw_delta"]),
            )
            for item in trace
        )
        # Repeated controlled forwards must replace, rather than accumulate,
        # the per-layer GPU trace.
        model(image, return_feature=True)
        repeated_trace = model.enc.transformer._last_layer_prompt_trace
        assert len(repeated_trace) == 12
        assert [int(item["layer_id"]) for item in repeated_trace] == list(range(12))
        with deep_prompt_residual_zero_intervention(model):
            zeroed, _ = model(image, return_feature=True)
        assert torch.equal(reference, zeroed)
        model.enc.transformer.deep_prompt_residual.layer_gate.fill_(0.25)
        changed, _ = model(image, return_feature=True)
        with deep_prompt_residual_zero_intervention(model):
            changed_zeroed, _ = model(image, return_feature=True)
        assert not torch.allclose(changed, changed_zeroed, atol=1.0e-8, rtol=1.0e-7)
    registry = model.get_bayesian_candidate_registry()
    object_ids = {item["object_id"] for item in registry}
    assert "shared_mu" in object_ids
    assert "delta_prompt/layer_11" in object_ids
    assert "contextualized_prompt/layer_11" in object_ids


def validate_b3_full_vit_isolation() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/b_series_experiments/B3-R1I-bounded-deep.yaml"
    )
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(43)
    model = ViT(cfg, load_pretrain=False).to(device).eval()
    model.attach_r_similarity_head(torch.zeros(200, 312, device=device))
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert not any(name.endswith("prompt_embeddings") for name in trainable)
    assert not any(name.startswith("r_similarity_head.") for name in trainable)
    assert any("prompt_init_provider.stats_head" in name for name in trainable)
    assert "enc.transformer.deep_prompt_residual.layer_gate" in trainable
    image = torch.randn(
        1,
        3,
        int(cfg.DATA.CROPSIZE),
        int(cfg.DATA.CROPSIZE),
        device=device,
    )
    transformer = model.enc.transformer
    encoder_forward = transformer.encoder.forward
    encoder_call_count = {"value": 0}

    def counted_encoder_forward(*args, **kwargs):
        encoder_call_count["value"] += 1
        return encoder_forward(*args, **kwargs)

    transformer.encoder.forward = counted_encoder_forward
    with torch.no_grad():
        model.begin_runtime_vit_cls_prepass_cache("same-fixed-probe-batch")
        first_feature, _ = model(image, return_feature=True)
        second_feature, _ = model(image, return_feature=True)
        model.end_runtime_vit_cls_prepass_cache()
    transformer.encoder.forward = encoder_forward
    assert torch.equal(first_feature, second_feature)
    # Prompted Deep-VPT executes blocks directly; only the frozen no-Prompt
    # prepass enters encoder.forward().  Two prompted forwards with one cache
    # key therefore require exactly one encoder.forward() prepass call.
    assert encoder_call_count["value"] == 1
    with torch.no_grad():
        model(image, return_feature=True)
    trace = model.enc.transformer._last_deep_prompt_residual_trace
    assert len(trace) == 12
    for item in trace:
        layer_id = int(item["layer_id"])
        if layer_id < 8:
            assert torch.equal(
                item["applied_delta"], torch.zeros_like(item["applied_delta"])
            )
        else:
            assert float(item["budget_exceed"].sum().item()) == 0.0
            assert torch.allclose(
                item["applied_ratio"],
                torch.full_like(item["applied_ratio"], 0.125),
                atol=2.0e-6,
            )
    with torch.no_grad():
        _, affinities = model.forward_with_affinity(
            image,
            {
                "prompt_length": 16,
                "semantic_length": 0,
                "detach": True,
                "include_visual_normalizations": False,
                "selected_layers": [8],
                "collect_prompt_slot_states": True,
                "offload_diagnostics_to_cpu": True,
            },
        )
    for key in (
        "_prompt_slot_raw_input",
        "_prompt_slot_ln_input",
        "_prompt_slot_key",
        "_prompt_slot_value",
    ):
        assert tuple(affinities[8][key].shape) == (1, 16, 768)
        assert affinities[8][key].device.type == "cpu"


def validate_s0_full_vit_isolation() -> None:
    cases = (
        (
            "configs/b_series_experiments/B3-S0I-direct-deep.yaml",
            {8, 9, 10, 11},
            True,
        ),
        (
            "configs/b_series_experiments/B3-S0N-direct-shallow-control.yaml",
            {0, 1, 2, 3},
            False,
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for config_path, active_layers, conditional in cases:
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.freeze()
        torch.manual_seed(47)
        model = ViT(cfg, load_pretrain=False).to(device).eval()
        model.attach_r_similarity_head(torch.zeros(200, 312, device=device))
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        assert trainable == {"enc.transformer.deep_prompt_residual.layer_gate"}
        image = torch.randn(
            2,
            3,
            int(cfg.DATA.CROPSIZE),
            int(cfg.DATA.CROPSIZE),
            device=device,
        )
        with torch.no_grad():
            model(image, return_feature=True)
        trace = model.enc.transformer._last_deep_prompt_residual_trace
        assert len(trace) == 12
        for item in trace:
            layer_id = int(item["layer_id"])
            if layer_id in active_layers:
                assert torch.allclose(
                    item["applied_ratio"],
                    torch.full_like(item["applied_ratio"], 0.125),
                    atol=2.0e-6,
                )
            else:
                assert torch.equal(
                    item["applied_delta"], torch.zeros_like(item["applied_delta"])
                )
        source_mu = trace[min(active_layers)]["source_mu"]
        assert torch.allclose(
            source_mu.float().norm(dim=-1),
            torch.ones(int(source_mu.shape[0]), device=source_mu.device),
            atol=2.0e-5,
        )
        if conditional:
            assert not torch.equal(source_mu[0], source_mu[1])
        else:
            assert torch.equal(source_mu[0], source_mu[1])


def validate_t1_full_vit_isolation() -> None:
    cases = (
        (
            "configs/b_series_experiments/B3-T1I-slot-scalar.yaml",
            True,
        ),
        (
            "configs/b_series_experiments/B3-T1N-slot-scalar-control.yaml",
            False,
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for config_path, conditional in cases:
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.freeze()
        torch.manual_seed(53)
        model = ViT(cfg, load_pretrain=False).to(device).eval()
        model.attach_r_similarity_head(torch.zeros(200, 312, device=device))
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        assert "enc.transformer.deep_prompt_residual.layer_gate" in trainable
        slot_parameters = {
            name for name in trainable if ".slot_coefficients." in name
        }
        assert len(slot_parameters) == 24
        assert trainable == slot_parameters | {
            "enc.transformer.deep_prompt_residual.layer_gate"
        }
        image = torch.randn(
            2,
            3,
            int(cfg.DATA.CROPSIZE),
            int(cfg.DATA.CROPSIZE),
            device=device,
        )
        with torch.no_grad():
            model(image, return_feature=True)
        trace = model.enc.transformer._last_deep_prompt_residual_trace
        assert len(trace) == 12
        for item in trace:
            layer_id = int(item["layer_id"])
            coefficients = item["slot_scalar_coefficients"]
            assert torch.allclose(coefficients, torch.ones_like(coefficients))
            if layer_id < 8:
                assert torch.equal(
                    item["applied_delta"], torch.zeros_like(item["applied_delta"])
                )
            else:
                assert torch.allclose(
                    item["applied_ratio"],
                    torch.full_like(item["applied_ratio"], 0.125),
                    atol=2.0e-6,
                )
        source_mu = trace[8]["source_mu"]
        assert torch.allclose(
            source_mu.float().norm(dim=-1),
            torch.ones(int(source_mu.shape[0]), device=source_mu.device),
            atol=2.0e-5,
        )
        if conditional:
            assert not torch.equal(source_mu[0], source_mu[1])
        else:
            assert torch.equal(source_mu[0], source_mu[1])
            assert torch.allclose(
                source_mu.float().mean(dim=-1),
                torch.zeros(int(source_mu.shape[0]), device=source_mu.device),
                atol=1.0e-7,
            )
        del model


def validate_a2_initialization_and_freeze_contract() -> None:
    with tempfile.TemporaryDirectory(dir=".") as temporary:
        root = Path(temporary).resolve()
        attributes = torch.zeros(200, 312)
        source_cfg = get_cfg()
        source_cfg.merge_from_file(
            "configs/baseline_rebuild/A-04-A2-vpt-deep-ce.yaml"
        )
        source_cfg.defrost()
        source_cfg.SEED = 0
        source_cfg.freeze()
        source = ViT(source_cfg, load_pretrain=False).eval()
        source.attach_r_similarity_head(attributes)
        trainable_names = [
            name for name, parameter in source.named_parameters() if parameter.requires_grad
        ]
        state = source.state_dict()
        checkpoint_path = root / "a2_trainable.pth"
        torch.save(
            {
                "format": "vpt_trainable_v1",
                "model_state": {name: state[name] for name in trainable_names},
                "trainable_parameter_names": trainable_names,
                "seed": 0,
                "protocol_mode": str(source_cfg.DATA.XLSA.PROTOCOL_MODE),
                "total_epoch": int(source_cfg.SOLVER.TOTAL_EPOCH),
            },
            str(checkpoint_path),
        )

        target_cfg = get_cfg()
        target_cfg.merge_from_file("configs/b_series_experiments/E5-B1-freeze.yaml")
        target_cfg.defrost()
        target_cfg.SEED = 0
        target_cfg.NUM_GPUS = 1
        target_cfg.OUTPUT_DIR = str(root / "freeze_run")
        target_cfg.SOLVER.INIT_TRAINABLE_CHECKPOINT = str(checkpoint_path)
        target_cfg.freeze()
        target = ViT(target_cfg, load_pretrain=False).eval()
        target.attach_r_similarity_head(attributes)
        trainer = Trainer(
            target_cfg,
            target,
            Evaluator(task_type="gzsl"),
            torch.device("cpu"),
        )
        manifest = trainer._initialization_checkpoint_manifest
        assert manifest["checkpoint_seed"] == 0
        assert manifest["new_trainable_parameter_names"]
        freeze = trainer._residual_freeze_contract_payload(final=False)
        assert freeze["pass"] is True
        assert all(row["in_optimizer"] is False for row in freeze["parameters"])
        _, auxiliary_names, checkpoint_state = trainer._trainable_model_state()
        assert any(name.endswith("prompt_embeddings") for name in auxiliary_names)
        assert all(name in checkpoint_state for name in auxiliary_names)
        trainer.diagnostic_manager.finalize(status="completed")
        trainer.monitor_manager.finalize(status="completed")


def validate_nested_probe_summary() -> None:
    cells = []
    for training_seed, values in ((0, (1.0, 2.0, 3.0)), (1, (3.0, 4.0, 5.0))):
        for value in values:
            cells.append(
                {
                    "method": "B1",
                    "training_seed": training_seed,
                    "metrics": {"condition.metric": value},
                }
            )
    payload, rows = nested_scientific_summary(cells, scope="probe")
    within_seed0 = payload["within_checkpoint"]["B1|seed0"]["condition.metric"]
    assert within_seed0["count"] == 3
    assert within_seed0["mean"] == 2.0
    across = payload["across_training_seed"]["B1"]["condition.metric"]
    assert across["count"] == 2
    assert across["mean"] == 3.0
    assert across["within_checkpoint_range_mean"] == 2.0
    assert len(rows) == 1


def validate_e7_precondition_evidence_gate() -> None:
    with tempfile.TemporaryDirectory(dir=".") as temporary:
        root = Path(temporary).resolve()
        evidence = root / "evidence.json"
        evidence.write_text("{}\n", encoding="utf-8")
        manifest = root / "preconditions.json"
        manifest.write_text(
            json.dumps(
                {
                    "E7": {
                        "eligible": True,
                        "max_gate_stage": "G1",
                        "checks": {
                            "bidirectional_nontrivial_response": True,
                            "predictable_without_test_leakage": True,
                            "pseudo_unseen_better_than_constant": True,
                        },
                        "evidence_paths": [evidence.name],
                    }
                }
            ),
            encoding="utf-8",
        )
        result = _validate_e7_preconditions(manifest, ("E7-G1",))
        assert result["pass"] is True
        evidence.unlink()
        try:
            _validate_e7_preconditions(manifest, ("E7-G1",))
        except ValueError:
            pass
        else:
            raise AssertionError("missing E7 evidence file was accepted")


def main() -> None:
    checks = (
        validate_zero_gate_and_gradient,
        validate_controls_and_metrics,
        validate_experiment_interventions_and_geometry,
        validate_slot_and_sample_gate_modes,
        validate_common_role_interventions,
        validate_donor_contracts,
        validate_bounded_deep_residual_contract,
        validate_b3_class_consistency_state,
        validate_b3_optimizer_multipliers,
        validate_parameter_only_provider_is_rng_neutral,
        validate_nonconditional_control,
        validate_direct_prepass_sources,
        validate_full_vit_config_and_trace,
        validate_b3_full_vit_isolation,
        validate_s0_full_vit_isolation,
        validate_t1_full_vit_isolation,
        validate_a2_initialization_and_freeze_contract,
        validate_nested_probe_summary,
        validate_e7_precondition_evidence_gate,
    )
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")


if __name__ == "__main__":
    main()
