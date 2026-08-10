#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.bayesian_object_selection import (
    BayesianHierarchyTraceAccumulator,
    StaticPromptPerturbation,
    build_candidate_registry,
    build_object_selection_report,
)
from src.configs.config import get_cfg
from src.engine.trainer import Trainer


class _StaticPromptModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = torch.nn.Parameter(torch.arange(12).float().reshape(1, 3, 4))
        self.deep_prompt_embeddings = torch.nn.Parameter(torch.arange(24).float().reshape(2, 3, 4))
        self.other = torch.nn.Parameter(torch.ones(2))


class _ExecutorSmokeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = torch.nn.Parameter(
            torch.tensor([[[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]])
        )
        self.register_buffer(
            "class_weights",
            torch.tensor(
                [[1.0, 0.0, 0.2], [0.1, 1.0, 0.0], [0.0, 0.2, 1.0]]
            ),
        )
        self.clear_runtime_state()

    def forward(
        self,
        inputs,
        semantics=None,
        class_ids=None,
        runtime_targets=None,
    ):
        batch_size = int(inputs.shape[0])
        image_scalar = inputs.float().reshape(batch_size, -1).mean(dim=1, keepdim=True)
        injected = self.prompt_embeddings.expand(batch_size, -1, -1)
        contextualized = injected + image_scalar.unsqueeze(-1)
        cls_tensor = contextualized.mean(dim=1) + image_scalar
        logits = cls_tensor @ self.class_weights.transpose(0, 1)
        self._runtime_injected = injected.detach()
        self._runtime_tokens = torch.cat(
            (cls_tensor.unsqueeze(1), contextualized), dim=1
        ).detach()
        self._runtime_classifier = {
            "visual_input": cls_tensor.detach(),
            "visual_repr": cls_tensor.detach(),
            "semantic_input": self.class_weights.detach(),
            "semantic_repr": self.class_weights.detach(),
        }
        return logits

    def get_runtime_classifier_stats(self):
        return self._runtime_classifier

    def get_runtime_injected_prompt_tokens(self):
        return self._runtime_injected

    def get_runtime_token_sequence(self):
        return self._runtime_tokens

    def get_runtime_prompt_distribution_stats(self):
        return None

    def clear_runtime_prompt_distribution_override(self):
        return None

    def clear_runtime_state(self):
        self._runtime_injected = None
        self._runtime_tokens = None
        self._runtime_classifier = None


class _ExecutorDistributorSmokeModel(_ExecutorSmokeModel):
    def __init__(self):
        super().__init__()
        del self.prompt_embeddings
        self.register_buffer("domain_prompt", torch.tensor([[[0.1, 0.2, 0.3]]]))
        self._distribution_override = None
        self._runtime_distribution = None

    def set_runtime_prompt_distribution_override(self, mu, logvar, eps=None):
        self._distribution_override = (
            mu.detach().clone(),
            logvar.detach().clone(),
            None if eps is None else eps.detach().clone(),
        )

    def clear_runtime_prompt_distribution_override(self):
        self._distribution_override = None

    def forward(
        self,
        inputs,
        semantics=None,
        class_ids=None,
        runtime_targets=None,
    ):
        batch_size = int(inputs.shape[0])
        image_scalar = inputs.float().reshape(batch_size, -1).mean(dim=1, keepdim=True)
        if self._distribution_override is None:
            mu = torch.cat((image_scalar, image_scalar * 2.0, image_scalar * 3.0), dim=1)
            logvar = torch.zeros_like(mu)
        else:
            mu, logvar, _ = self._distribution_override
            self._distribution_override = None
        instance_prompt = mu.unsqueeze(1)
        domain_prompt = self.domain_prompt.expand(batch_size, -1, -1)
        injected = torch.cat((instance_prompt, domain_prompt), dim=1)
        contextualized = injected + image_scalar.unsqueeze(-1)
        cls_tensor = contextualized.mean(dim=1) + image_scalar
        logits = cls_tensor @ self.class_weights.transpose(0, 1)
        self._runtime_injected = injected.detach()
        self._runtime_tokens = torch.cat(
            (cls_tensor.unsqueeze(1), contextualized), dim=1
        ).detach()
        self._runtime_classifier = {
            "visual_input": cls_tensor.detach(),
            "visual_repr": cls_tensor.detach(),
            "semantic_input": self.class_weights.detach(),
            "semantic_repr": self.class_weights.detach(),
        }
        self._runtime_distribution = {
            "mu": mu.detach(),
            "logvar": logvar.detach(),
            "prompt_tokens": injected.detach(),
        }
        return logits

    def get_runtime_prompt_distribution_stats(self):
        return self._runtime_distribution


def _variant(variant_id, sample_ids, source, reference_logits, multiplier):
    effect = source * float(multiplier)
    logits_effect = effect[:, :3]
    return {
        "variant_id": variant_id,
        "sample_ids": sample_ids,
        "source_object": source,
        "injected_prompt": source * 2.0,
        "contextualized_prompt": source * 3.0,
        "semantic_aligned_contextualized_prompt": source[:, :3] * 3.0,
        "cls_effect": effect,
        "logit_effect": logits_effect,
        "logits": reference_logits + logits_effect,
    }


def validate_static_intervention():
    model = _StaticPromptModel()
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    with StaticPromptPerturbation(
        model,
        direction_id=1,
        scale=0.1,
        seed=7,
        selected_layers=[0, 2],
    ) as intervention:
        assert intervention.delta_norm > 0.0
        assert not torch.equal(model.prompt_embeddings, original["prompt_embeddings"])
        assert torch.equal(model.deep_prompt_embeddings[0], original["deep_prompt_embeddings"][0])
        assert not torch.equal(model.deep_prompt_embeddings[1], original["deep_prompt_embeddings"][1])
        assert torch.equal(model.other, original["other"])
    for name, value in model.named_parameters():
        assert torch.equal(value, original[name])


def validate_trace_and_registry():
    sample_ids = ["s0", "s1", "s2", "s3"]
    targets = [0, 1, 2, 0]
    base = torch.tensor(
        [
            [0.2, 0.4, 0.8],
            [0.4, 0.7, 1.1],
            [0.8, 0.2, 1.4],
            [1.2, 0.9, 0.3],
        ]
    )
    reference_logits = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0],
        ]
    )
    zero = torch.zeros_like(base)
    variants = [
        _variant("reference", sample_ids, zero, reference_logits, 0.0),
        _variant("direction_000_scale_0.1", sample_ids, base, reference_logits, 4.0),
        _variant("direction_001_scale_0.2", sample_ids, base * 2.5, reference_logits, 4.0),
    ]
    accumulator = BayesianHierarchyTraceAccumulator(
        [0, 1, 2],
        bootstrap_samples=20,
        random_seed=9,
        collapse_relative_threshold=0.05,
        distance_eps=1.0e-8,
    )
    accumulator.update(
        sample_ids=sample_ids,
        targets_local=targets,
        variants=variants,
    )
    trace = accumulator.finalize()
    assert trace["valid"]
    assert trace["sample_count"] == 4
    assert trace["distance_correspondence"]["source_to_logit_effect_distance_spearman"]["spearman"] > 0.999
    assert trace["propagation"]["perturbation_cls_effect_to_logit_effect_propagation_ratio"]["ratio"] > 0.0
    assert not trace["posterior_interpretation_allowed"]

    registry = build_candidate_registry(
        requested_candidates=[
            "raw_latent",
            "injected_prompt",
            "contextualized_prompt",
            "cls_effect",
            "logit_effect",
        ],
        requested_auxiliary_views=[
            "prompt_to_cls_contribution",
            "semantic_aligned_contextualized_prompt",
            "decision_margin_effect",
        ],
        prompt_enabled=True,
        distributor_active=False,
        prompt_deep=True,
    )
    assert not registry["candidate_spaces"]["raw_latent"]["applicable"]
    assert registry["candidate_spaces"]["injected_prompt"]["applicable"]
    assert not registry["auxiliary_views"]["prompt_to_cls_contribution"]["applicable"]
    report = build_object_selection_report(registry, trace)
    assert report["recommended_candidate"] is None
    assert report["posterior_metrics_deferred"]

    bad = [dict(item) for item in variants]
    bad[1] = {**bad[1], "sample_ids": list(reversed(sample_ids))}
    rejected = BayesianHierarchyTraceAccumulator(
        [0, 1, 2],
        bootstrap_samples=0,
        random_seed=9,
        collapse_relative_threshold=0.05,
        distance_eps=1.0e-8,
    )
    try:
        rejected.update(
            sample_ids=sample_ids,
            targets_local=targets,
            variants=bad,
        )
    except ValueError as error:
        assert "sample_id order mismatch" in str(error)
    else:
        raise AssertionError("variant/sample identity mismatch was not rejected")

    incompatible = BayesianHierarchyTraceAccumulator(
        [0, 1],
        bootstrap_samples=0,
        random_seed=9,
        collapse_relative_threshold=0.05,
        distance_eps=1.0e-8,
    )
    try:
        incompatible.update(
            sample_ids=sample_ids,
            targets_local=targets,
            variants=variants,
        )
    except ValueError as error:
        assert "candidate_class_ids" in str(error)
    else:
        raise AssertionError("candidate class identity mismatch was not rejected")

    collapsed_variants = []
    for index, item in enumerate(variants):
        collapsed = dict(item)
        if index > 0:
            collapsed["contextualized_prompt"] = torch.zeros_like(
                item["contextualized_prompt"]
            )
            collapsed["cls_effect"] = torch.zeros_like(item["cls_effect"])
            collapsed["logit_effect"] = torch.zeros_like(item["logit_effect"])
            collapsed["logits"] = reference_logits.clone()
        collapsed_variants.append(collapsed)
    collapse_accumulator = BayesianHierarchyTraceAccumulator(
        [0, 1, 2],
        bootstrap_samples=0,
        random_seed=9,
        collapse_relative_threshold=0.05,
        distance_eps=1.0e-8,
    )
    collapse_accumulator.update(
        sample_ids=sample_ids,
        targets_local=targets,
        variants=collapsed_variants,
    )
    collapse_trace = collapse_accumulator.finalize()
    assert collapse_trace["collapse"][
        "perturbation_functional_collapse_ratio/injected_to_contextualized"
    ]["ratio"] == 1.0
    assert collapse_trace["perturbation_functional_null_direction_ratio"][
        "ratio"
    ] == 1.0
    return trace


def validate_executor_smoke():
    cfg = get_cfg()
    cfg.defrost()
    cfg.MODEL.PROMPT.ENABLE = True
    cfg.MODEL.PROMPT.NUM_TOKENS = 2
    cfg.MODEL.PROMPT.DEEP = False
    cfg.MODEL.PROMPT.INIT_SOURCE = "learned"
    cfg.MODEL.PROMPT.DISTRIBUTOR.ENABLE = False
    cfg.MODEL.SEMANTIC_TOKENS.ENABLE = False
    object_cfg = cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION
    object_cfg.ENABLE = True
    object_cfg.PERTURBATION_SCALES = [0.01, 0.03]
    object_cfg.DIRECTION_COUNT = 2
    object_cfg.HIERARCHY_VARIANT_COUNT = 4
    object_cfg.BOOTSTRAP_SAMPLES = 10
    object_cfg.AUXILIARY_VIEWS = [
        "semantic_aligned_contextualized_prompt",
        "decision_margin_effect",
    ]
    cfg.freeze()
    trainer = object.__new__(Trainer)
    trainer.cfg = cfg
    trainer.model = _ExecutorSmokeModel()
    trainer.device = torch.device("cpu")
    probe_loader = [
        {
            "image": torch.tensor([[0.2, 0.4], [0.7, 1.1]]),
            "label": torch.tensor([0, 1]),
            "sample_id": ["executor_s0", "executor_s1"],
        }
    ]
    source_dataset = type("SourceDataset", (), {})()
    result = trainer._execute_bayesian_object_selection_probe(
        probe_loader,
        source_dataset,
        [0, 1, 2],
        split="probe_smoke",
    )
    assert result["valid"]
    assert result["hierarchy_trace"]["sample_count"] == 2
    assert result["hierarchy_trace"]["variant_count_including_reference"] == 5
    assert result["execution_contract"]["variant_storage_device"] == "cpu"
    assert result["execution_contract"]["gpu_live_variant_policy"] == (
        "reference_plus_current_variant"
    )
    assert result["registry"]["candidate_spaces"]["raw_latent"]["failure_reason"] == "prompt_distributor_not_active"
    assert result["object_selection_report"]["recommended_candidate"] is None

    distributor_cfg = cfg.clone()
    distributor_cfg.defrost()
    distributor_cfg.MODEL.PROMPT.INIT_SOURCE = "distributor_mean"
    distributor_cfg.MODEL.PROMPT.DISTRIBUTOR.ENABLE = True
    distributor_cfg.MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS = 1
    distributor_cfg.MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS = 1
    distributor_cfg.freeze()
    distributor_trainer = object.__new__(Trainer)
    distributor_trainer.cfg = distributor_cfg
    distributor_trainer.model = _ExecutorDistributorSmokeModel()
    distributor_trainer.device = torch.device("cpu")
    distributor_result = distributor_trainer._execute_bayesian_object_selection_probe(
        probe_loader,
        source_dataset,
        [0, 1, 2],
        split="probe_distributor_smoke",
    )
    raw_state = distributor_result["registry"]["candidate_spaces"]["raw_latent"]
    assert distributor_result["valid"]
    assert raw_state["applicable"] and raw_state["observed"] and raw_state["valid"]
    assert distributor_result["hierarchy_trace"]["source_object_name"] == "raw_latent"
    assert "latent_to_logit_delta_distance_spearman" in distributor_result[
        "functional_geometry"
    ]["distance_correspondence"]


def main():
    validate_static_intervention()
    trace = validate_trace_and_registry()
    validate_executor_smoke()
    print(json.dumps({
        "status": "passed",
        "checks": [
            "static_prompt_perturbation_restore",
            "deep_prompt_layer_identity",
            "one_to_one_functional_geometry",
            "dimension_normalized_propagation",
            "candidate_applicability",
            "no_automatic_candidate_selection",
            "variant_sample_identity_rejection",
            "candidate_class_identity_rejection",
            "functional_collapse_localization",
            "functional_null_direction_detection",
            "trainer_executor_static_vpt_smoke",
            "trainer_executor_distributor_smoke",
            "cpu_variant_storage_contract",
        ],
        "observed_stages": trace["observed_stages"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
