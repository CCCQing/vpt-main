#!/usr/bin/env python3
"""Focused contracts for deterministic direct-mean Deep Prompt residual."""

from __future__ import annotations

import torch

from src.models.prompting.prompt_distribution import (
    MeanConditionedDeepPromptResidual,
    PreViTPromptDistributor,
)
from src.monitoring.adapters import deep_prompt_residual_metrics
from src.monitoring.module_effect import (
    deep_prompt_residual_swap_intervention,
    deep_prompt_residual_zero_intervention,
)
from src.configs.config import get_cfg
from src.models.vit_models import ViT


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


def main() -> None:
    checks = (
        validate_zero_gate_and_gradient,
        validate_controls_and_metrics,
        validate_parameter_only_provider_is_rng_neutral,
        validate_nonconditional_control,
        validate_full_vit_config_and_trace,
    )
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")


if __name__ == "__main__":
    main()
