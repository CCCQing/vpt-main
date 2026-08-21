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
    deterministic_donor_indices,
    layer_scales,
    residual_static_geometry,
    summarize_geometry,
)
from src.monitoring.module_effect import (
    deep_prompt_residual_layer_scales_intervention,
    deep_prompt_residual_replace_intervention,
    deep_prompt_residual_swap_intervention,
    deep_prompt_residual_zero_intervention,
)
from src.configs.config import get_cfg
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.vit_models import ViT
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
        validate_donor_contracts,
        validate_parameter_only_provider_is_rng_neutral,
        validate_nonconditional_control,
        validate_full_vit_config_and_trace,
        validate_a2_initialization_and_freeze_contract,
        validate_nested_probe_summary,
        validate_e7_precondition_evidence_gate,
    )
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")


if __name__ == "__main__":
    main()
