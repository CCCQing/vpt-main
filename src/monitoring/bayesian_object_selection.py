from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from contextlib import AbstractContextManager
from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


MAIN_CANDIDATES = (
    "raw_latent",
    "injected_prompt",
    "contextualized_prompt",
    "cls_effect",
    "logit_effect",
)

AUXILIARY_VIEWS = (
    "prompt_to_cls_contribution",
    "semantic_aligned_contextualized_prompt",
    "decision_margin_effect",
)


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _spearman_summary(
    x: Sequence[float],
    y: Sequence[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    coefficient = _spearman(x, y)
    interval = [None, None]
    if x.size >= 4 and int(bootstrap_samples) > 0 and math.isfinite(coefficient):
        generator = np.random.RandomState(int(seed))
        draws = []
        for _ in range(int(bootstrap_samples)):
            indices = generator.randint(0, x.size, size=x.size)
            value = _spearman(x[indices], y[indices])
            if math.isfinite(value):
                draws.append(value)
        if draws:
            interval = [
                float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5)),
            ]
    return {
        "spearman": coefficient if math.isfinite(coefficient) else None,
        "pair_count": int(x.size),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_interval_95": interval,
    }


def build_candidate_registry(
    *,
    requested_candidates: Sequence[str],
    requested_auxiliary_views: Sequence[str],
    prompt_enabled: bool,
    distributor_active: bool,
    prompt_deep: bool,
) -> Dict[str, Any]:
    unknown_candidates = sorted(set(requested_candidates).difference(MAIN_CANDIDATES))
    unknown_views = sorted(set(requested_auxiliary_views).difference(AUXILIARY_VIEWS))
    if unknown_candidates:
        raise ValueError(
            "Unknown Bayesian object candidate spaces: " + ", ".join(unknown_candidates)
        )
    if unknown_views:
        raise ValueError(
            "Unknown Bayesian object auxiliary views: " + ", ".join(unknown_views)
        )
    requested_candidates = list(dict.fromkeys(str(item) for item in requested_candidates))
    requested_auxiliary_views = list(
        dict.fromkeys(str(item) for item in requested_auxiliary_views)
    )
    candidate_registry = {}
    for name in MAIN_CANDIDATES:
        requested = name in requested_candidates
        applicable = bool(prompt_enabled and (name != "raw_latent" or distributor_active))
        reason = None
        if requested and not prompt_enabled:
            reason = "prompt_not_enabled"
        elif requested and name == "raw_latent" and not distributor_active:
            reason = "prompt_distributor_not_active"
        candidate_registry[name] = {
            "role": "main_candidate",
            "requested": requested,
            "applicable": applicable if requested else False,
            "observed": False,
            "valid": None,
            "failure_reason": reason if requested else "not_requested",
        }
    auxiliary_registry = {}
    for name in AUXILIARY_VIEWS:
        requested = name in requested_auxiliary_views
        applicable = bool(prompt_enabled)
        reason = None
        if requested and not prompt_enabled:
            reason = "prompt_not_enabled"
        elif requested and name == "prompt_to_cls_contribution":
            applicable = False
            reason = "requires_affinity_value_contribution_variant_trace"
        auxiliary_registry[name] = {
            "role": "auxiliary_view",
            "requested": requested,
            "applicable": applicable if requested else False,
            "observed": False,
            "valid": None,
            "failure_reason": reason if requested else "not_requested",
        }
    return {
        "format": "bayesian_object_candidate_registry_v1",
        "prompt_identity": (
            "layer_x_prompt_type_x_slot" if prompt_deep else "prompt_type_x_slot"
        ),
        "candidate_spaces": candidate_registry,
        "auxiliary_views": auxiliary_registry,
        "excluded_objects": {
            "classifier_weight": {
                "reason": "fixed_classifier_weights_do_not_vary_with_prompt_sample",
                "replacement_view": "decision_margin_effect",
            }
        },
    }


def _selected_prompt_parameter_slices(
    model: torch.nn.Module,
    selected_layers: Sequence[int],
):
    selected = {int(item) for item in selected_layers}
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if lowered.endswith("deep_prompt_embeddings") and parameter.dim() == 3:
            for index in range(int(parameter.shape[0])):
                layer = index + 1
                if not selected or layer in selected:
                    yield name, parameter, index, layer
        elif (
            lowered.endswith("prompt_embeddings")
            and not lowered.endswith("deep_prompt_embeddings")
        ):
            if not selected or 0 in selected:
                yield name, parameter, None, 0


def static_prompt_vector(
    model: torch.nn.Module,
    selected_layers: Sequence[int] = (),
) -> torch.Tensor:
    chunks = []
    for _, parameter, index, _ in _selected_prompt_parameter_slices(
        model, selected_layers
    ):
        value = parameter if index is None else parameter[index]
        chunks.append(value.detach().float().reshape(-1).cpu())
    if not chunks:
        raise RuntimeError("No static Prompt parameter matched the requested layers")
    return torch.cat(chunks, dim=0)


class StaticPromptPerturbation(AbstractContextManager):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        direction_id: int,
        scale: float,
        seed: int,
        selected_layers: Sequence[int] = (),
    ) -> None:
        if float(scale) <= 0.0:
            raise ValueError("Prompt perturbation scale must be positive")
        self.model = model
        self.direction_id = int(direction_id)
        self.scale = float(scale)
        self.seed = int(seed)
        self.selected_layers = [int(item) for item in selected_layers]
        self.saved: Dict[str, torch.Tensor] = {}
        self.delta_norm = 0.0
        self.parameter_norm = 0.0

    def __enter__(self):
        slices = list(
            _selected_prompt_parameter_slices(self.model, self.selected_layers)
        )
        if not slices:
            raise RuntimeError("No static Prompt parameter matched the requested layers")
        with torch.no_grad():
            for name, parameter, index, layer in slices:
                if name not in self.saved:
                    self.saved[name] = parameter.detach().clone()
                target = parameter if index is None else parameter[index]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    _stable_seed(self.seed, self.direction_id, name, layer)
                )
                direction = torch.randn(
                    tuple(target.shape), generator=generator, dtype=torch.float32
                ).to(device=target.device, dtype=target.dtype)
                direction_norm = direction.float().norm().clamp_min(1.0e-12)
                parameter_norm = target.detach().float().norm()
                reference_norm = parameter_norm.clamp_min(
                    math.sqrt(max(1, target.numel())) * 1.0e-6
                )
                delta = direction / direction_norm.to(direction.dtype)
                delta = delta * (self.scale * reference_norm).to(delta.dtype)
                target.add_(delta)
                self.delta_norm += float(delta.float().norm().item()) ** 2
                self.parameter_norm += float(parameter_norm.item()) ** 2
        self.delta_norm = math.sqrt(self.delta_norm)
        self.parameter_norm = math.sqrt(self.parameter_norm)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with torch.no_grad():
            named = dict(self.model.named_parameters())
            for name, value in self.saved.items():
                named[name].copy_(value)
        self.saved.clear()
        return False


def normalized_latent_direction(
    reference: torch.Tensor,
    *,
    direction_id: int,
    scale: float,
    seed: int,
) -> torch.Tensor:
    if reference.dim() != 2:
        raise ValueError("raw latent reference must be [batch, dim]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, direction_id, "raw_latent"))
    direction = torch.randn(
        int(reference.shape[1]), generator=generator, dtype=torch.float32
    ).to(device=reference.device, dtype=reference.dtype)
    direction = direction / direction.float().norm().clamp_min(1.0e-12).to(
        direction.dtype
    )
    reference_norm = reference.detach().float().norm(dim=-1, keepdim=True)
    reference_norm = reference_norm.clamp_min(
        math.sqrt(max(1, int(reference.shape[1]))) * 1.0e-6
    )
    return reference + direction.unsqueeze(0) * (
        float(scale) * reference_norm
    ).to(direction.dtype)


class BayesianHierarchyTraceAccumulator:
    STAGES = (
        "source_object",
        "injected_prompt",
        "contextualized_prompt",
        "semantic_aligned_contextualized_prompt",
        "cls_effect",
        "logit_effect",
    )

    RELATIONS = (
        ("source_object", "injected_prompt", "source_to_injected_distance_spearman"),
        ("source_object", "contextualized_prompt", "source_to_contextualized_distance_spearman"),
        ("source_object", "cls_effect", "source_to_cls_effect_distance_spearman"),
        ("source_object", "logit_effect", "source_to_logit_effect_distance_spearman"),
        ("injected_prompt", "contextualized_prompt", "injected_to_contextualized_distance_spearman"),
        ("injected_prompt", "cls_effect", "injected_prompt_to_cls_delta_distance_spearman"),
        ("contextualized_prompt", "cls_effect", "contextualized_to_cls_effect_distance_spearman"),
        ("contextualized_prompt", "logit_effect", "contextualized_prompt_to_logit_delta_distance_spearman"),
        ("cls_effect", "logit_effect", "cls_effect_to_logit_effect_distance_spearman"),
        (
            "semantic_aligned_contextualized_prompt",
            "cls_effect",
            "semantic_distance_to_cls_effect_distance_spearman",
        ),
    )

    TRANSITIONS = (
        ("injected_prompt", "contextualized_prompt", "injected_to_contextualized"),
        ("contextualized_prompt", "cls_effect", "contextualized_to_cls_effect"),
        ("cls_effect", "logit_effect", "cls_effect_to_logit_effect"),
    )

    def __init__(
        self,
        candidate_class_ids: Sequence[int],
        *,
        bootstrap_samples: int,
        random_seed: int,
        collapse_relative_threshold: float,
        distance_eps: float,
    ) -> None:
        self.candidate_class_ids = tuple(int(item) for item in candidate_class_ids)
        self.bootstrap_samples = int(bootstrap_samples)
        self.random_seed = int(random_seed)
        self.collapse_relative_threshold = float(collapse_relative_threshold)
        self.distance_eps = float(distance_eps)
        self.sample_ids = set()
        self.variant_ids = None
        self.observed_stages = set()
        self.relation_values = {
            name: [[], []] for _, _, name in self.RELATIONS
        }
        self.stage_variance_sum = defaultdict(float)
        self.stage_variance_count = defaultdict(int)
        self.transition_values = {
            name: [[], []] for _, _, name in self.TRANSITIONS
        }
        self.reference_distance_values = defaultdict(list)
        self.prediction_flip = []
        self.probability_l2 = []
        self.true_margin_delta = []
        self.hard_negative_margin_delta = []
        self.reference_features = defaultdict(list)
        self.reference_targets = []

    @staticmethod
    def _flatten(value: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not torch.is_tensor(value) or int(value.shape[0]) != int(batch_size):
            raise ValueError("Bayesian hierarchy stage tensor has incompatible batch identity")
        value = value.detach().float().reshape(batch_size, -1).cpu()
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("Bayesian hierarchy stage tensor contains non-finite values")
        return value

    @staticmethod
    def _rms_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).pow(2).mean(dim=-1).sqrt()

    @staticmethod
    def _true_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        other = logits.clone()
        other.scatter_(1, targets.unsqueeze(1), torch.finfo(other.dtype).min)
        return logits.gather(1, targets.unsqueeze(1)).squeeze(1) - other.max(dim=1).values

    def update(
        self,
        *,
        sample_ids: Sequence[str],
        targets_local: Sequence[int],
        variants: Sequence[Mapping[str, Any]],
    ) -> None:
        sample_ids = [str(item) for item in sample_ids]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Bayesian hierarchy batch contains duplicate sample_id")
        if self.sample_ids.intersection(sample_ids):
            raise ValueError("Bayesian hierarchy sample_id was observed more than once")
        if len(variants) < 2 or str(variants[0].get("variant_id")) != "reference":
            raise ValueError("Bayesian hierarchy requires reference followed by at least one variant")
        variant_ids = tuple(str(item.get("variant_id")) for item in variants)
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Bayesian hierarchy variant_id values must be unique")
        if self.variant_ids is None:
            self.variant_ids = variant_ids
        elif self.variant_ids != variant_ids:
            raise ValueError("Bayesian hierarchy variant_id order changed across batches")
        batch_size = len(sample_ids)
        targets = torch.as_tensor(targets_local, dtype=torch.long)
        if tuple(targets.shape) != (batch_size,):
            raise ValueError("Bayesian hierarchy target identity is incompatible with sample_id")
        stages: Dict[str, torch.Tensor] = {}
        for stage in self.STAGES:
            values = []
            for variant in variants:
                if list(variant.get("sample_ids", [])) != sample_ids:
                    raise ValueError("Bayesian hierarchy variant sample_id order mismatch")
                value = variant.get(stage)
                if not torch.is_tensor(value):
                    values = []
                    break
                values.append(self._flatten(value, batch_size))
            if values:
                stacked = torch.stack(values, dim=0)
                stages[stage] = stacked
                self.observed_stages.add(stage)
                variance = stacked.var(dim=0, unbiased=False).mean(dim=-1)
                self.stage_variance_sum[stage] += float(variance.sum().item())
                self.stage_variance_count[stage] += int(variance.numel())
                for index in range(1, len(variants)):
                    self.reference_distance_values[stage].extend(
                        self._rms_distance(stacked[0], stacked[index]).tolist()
                    )

        pair_distances = {}
        for stage, stacked in stages.items():
            distances = []
            for left, right in combinations(range(len(variants)), 2):
                distances.append(self._rms_distance(stacked[left], stacked[right]))
            pair_distances[stage] = torch.cat(distances, dim=0).numpy()
        for left, right, name in self.RELATIONS:
            if left in pair_distances and right in pair_distances:
                self.relation_values[name][0].extend(pair_distances[left].tolist())
                self.relation_values[name][1].extend(pair_distances[right].tolist())

        for upstream, downstream, name in self.TRANSITIONS:
            if upstream not in stages or downstream not in stages:
                continue
            for index in range(1, len(variants)):
                upstream_distance = self._rms_distance(
                    stages[upstream][0], stages[upstream][index]
                )
                downstream_distance = self._rms_distance(
                    stages[downstream][0], stages[downstream][index]
                )
                self.transition_values[name][0].extend(upstream_distance.tolist())
                self.transition_values[name][1].extend(downstream_distance.tolist())

        reference_logits = self._flatten(variants[0].get("logits"), batch_size)
        if int(reference_logits.shape[1]) != len(self.candidate_class_ids):
            raise ValueError(
                "Bayesian hierarchy logits do not match candidate_class_ids"
            )
        reference_prediction = reference_logits.argmax(dim=1)
        reference_probability = torch.softmax(reference_logits, dim=1)
        reference_margin = self._true_margin(reference_logits, targets)
        hard_negative_scores = reference_logits.clone()
        hard_negative_scores.scatter_(
            1, targets.unsqueeze(1), torch.finfo(reference_logits.dtype).min
        )
        hard_negative = hard_negative_scores.argmax(dim=1)
        reference_hard_margin = (
            reference_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            - reference_logits.gather(1, hard_negative.unsqueeze(1)).squeeze(1)
        )
        for variant in variants[1:]:
            logits = self._flatten(variant.get("logits"), batch_size)
            if tuple(logits.shape) != tuple(reference_logits.shape):
                raise ValueError(
                    "Bayesian hierarchy variant logits changed candidate class identity"
                )
            probability = torch.softmax(logits, dim=1)
            self.prediction_flip.extend(
                (logits.argmax(dim=1) != reference_prediction).float().tolist()
            )
            self.probability_l2.extend(
                self._rms_distance(probability, reference_probability).tolist()
            )
            self.true_margin_delta.extend(
                (self._true_margin(logits, targets) - reference_margin).tolist()
            )
            hard_margin = (
                logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                - logits.gather(1, hard_negative.unsqueeze(1)).squeeze(1)
            )
            self.hard_negative_margin_delta.extend(
                (hard_margin - reference_hard_margin).tolist()
            )

        for stage in ("source_object", "injected_prompt", "contextualized_prompt"):
            if stage in stages:
                self.reference_features[stage].append(stages[stage][0])
        self.reference_targets.extend(targets.tolist())
        self.sample_ids.update(sample_ids)

    def _class_structure(self, stage: str) -> Dict[str, Any]:
        chunks = self.reference_features.get(stage, [])
        if not chunks:
            return {"status": "not_observed"}
        features = torch.cat(chunks, dim=0)
        targets = torch.as_tensor(self.reference_targets, dtype=torch.long)
        within = []
        between = []
        for left, right in combinations(range(int(features.shape[0])), 2):
            distance = float(self._rms_distance(features[left], features[right]).item())
            if int(targets[left]) == int(targets[right]):
                within.append(distance)
            else:
                between.append(distance)
        if not within or not between:
            return {
                "status": "insufficient_evidence",
                "within_pair_count": int(len(within)),
                "between_pair_count": int(len(between)),
            }
        within_mean = float(np.mean(within))
        between_mean = float(np.mean(between))
        return {
            "status": "observed",
            "between_class_distance_mean": between_mean,
            "within_class_distance_mean": within_mean,
            "between_class_within_class_ratio": (
                between_mean / max(within_mean, self.distance_eps)
            ),
            "within_pair_count": int(len(within)),
            "between_pair_count": int(len(between)),
            "class_count": int(torch.unique(targets).numel()),
            "sample_count": int(features.shape[0]),
        }

    @staticmethod
    def _summary(values: Sequence[float]) -> Dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"mean": None, "std": None, "count": 0}
        return {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "count": int(finite.size),
        }

    def finalize(self) -> Dict[str, Any]:
        correspondence = {}
        for index, (_, _, name) in enumerate(self.RELATIONS):
            x, y = self.relation_values[name]
            if not x:
                continue
            correspondence[name] = _spearman_summary(
                x,
                y,
                bootstrap_samples=self.bootstrap_samples,
                seed=_stable_seed(self.random_seed, "bootstrap", index),
            )
        trace_variance = {
            stage: self.stage_variance_sum[stage]
            / max(1, self.stage_variance_count[stage])
            for stage in sorted(self.stage_variance_sum)
        }
        propagation = {}
        collapse = {}
        for upstream, downstream, name in self.TRANSITIONS:
            if upstream in trace_variance and downstream in trace_variance:
                denominator = float(trace_variance[upstream])
                clamped = denominator <= self.distance_eps
                propagation[f"perturbation_{name}_propagation_ratio"] = {
                    "ratio": float(trace_variance[downstream])
                    / max(denominator, self.distance_eps),
                    "numerator_normalized_trace_variance": float(
                        trace_variance[downstream]
                    ),
                    "denominator_normalized_trace_variance": denominator,
                    "denominator_clamp_hit": bool(clamped),
                }
            upstream_distance, downstream_distance = self.transition_values[name]
            upstream_distance = np.asarray(upstream_distance, dtype=np.float64)
            downstream_distance = np.asarray(downstream_distance, dtype=np.float64)
            valid = np.isfinite(upstream_distance) & np.isfinite(downstream_distance)
            valid &= upstream_distance > self.distance_eps
            effective_downstream = downstream_distance[valid]
            if effective_downstream.size:
                positive = effective_downstream[effective_downstream > self.distance_eps]
                reference_floor = (
                    float(np.median(positive)) if positive.size else self.distance_eps
                )
                threshold = max(
                    self.distance_eps,
                    self.collapse_relative_threshold * reference_floor,
                )
                collapsed = effective_downstream <= threshold
                collapse[f"perturbation_functional_collapse_ratio/{name}"] = {
                    "ratio": float(collapsed.mean()),
                    "valid_transition_count": int(collapsed.size),
                    "downstream_threshold": float(threshold),
                    "relative_threshold": self.collapse_relative_threshold,
                }

        margin_values = np.asarray(self.hard_negative_margin_delta, dtype=np.float64)
        finite_margin = margin_values[np.isfinite(margin_values)]
        sign_consistency = None
        if finite_margin.size:
            positive = float(np.mean(finite_margin > 0.0))
            negative = float(np.mean(finite_margin < 0.0))
            sign_consistency = max(positive, negative)
        prediction_effect = {
            "perturbation_prediction_flip_or_disagreement": self._summary(
                self.prediction_flip
            ),
            "probability_vector_rms_distance": self._summary(self.probability_l2),
            "delta_true_margin": self._summary(self.true_margin_delta),
            "true_vs_hard_negative_margin_effect": self._summary(
                self.hard_negative_margin_delta
            ),
            "boundary_crossing_rate": self._summary(self.prediction_flip),
            "margin_effect_sign_consistency": sign_consistency,
        }
        reference_distance = {
            stage: self._summary(values)
            for stage, values in sorted(self.reference_distance_values.items())
        }
        functional_null = {
            "ratio": None,
            "valid_pair_count": 0,
            "cls_effect_threshold": None,
            "logit_effect_threshold": None,
        }
        if all(
            stage in self.reference_distance_values
            for stage in ("source_object", "cls_effect", "logit_effect")
        ):
            source_distance = np.asarray(
                self.reference_distance_values["source_object"], dtype=np.float64
            )
            cls_distance = np.asarray(
                self.reference_distance_values["cls_effect"], dtype=np.float64
            )
            logit_distance = np.asarray(
                self.reference_distance_values["logit_effect"], dtype=np.float64
            )
            valid = np.isfinite(source_distance)
            valid &= np.isfinite(cls_distance) & np.isfinite(logit_distance)
            valid &= source_distance > self.distance_eps
            if bool(valid.any()):
                positive_cls = cls_distance[valid & (cls_distance > self.distance_eps)]
                positive_logit = logit_distance[
                    valid & (logit_distance > self.distance_eps)
                ]
                cls_reference = (
                    float(np.median(positive_cls))
                    if positive_cls.size
                    else self.distance_eps
                )
                logit_reference = (
                    float(np.median(positive_logit))
                    if positive_logit.size
                    else self.distance_eps
                )
                cls_threshold = max(
                    self.distance_eps,
                    self.collapse_relative_threshold * cls_reference,
                )
                logit_threshold = max(
                    self.distance_eps,
                    self.collapse_relative_threshold * logit_reference,
                )
                null = (cls_distance[valid] <= cls_threshold) & (
                    logit_distance[valid] <= logit_threshold
                )
                functional_null = {
                    "ratio": float(null.mean()),
                    "valid_pair_count": int(null.size),
                    "cls_effect_threshold": float(cls_threshold),
                    "logit_effect_threshold": float(logit_threshold),
                }
        return {
            "format": "bayesian_hierarchy_trace_v1",
            "valid": bool(self.sample_ids and self.variant_ids and correspondence),
            "sample_count": int(len(self.sample_ids)),
            "variant_ids": list(self.variant_ids or ()),
            "variant_count_including_reference": int(len(self.variant_ids or ())),
            "candidate_class_ids": list(self.candidate_class_ids),
            "candidate_class_ids_sha256": hashlib.sha256(
                ",".join(str(item) for item in self.candidate_class_ids).encode("utf-8")
            ).hexdigest(),
            "observed_stages": sorted(self.observed_stages),
            "missing_stages": sorted(set(self.STAGES).difference(self.observed_stages)),
            "distance_correspondence": correspondence,
            "normalized_trace_variance": trace_variance,
            "propagation": propagation,
            "collapse": collapse,
            "perturbation_functional_null_direction_ratio": functional_null,
            "reference_distance_by_stage": reference_distance,
            "prediction_effect": prediction_effect,
            "class_structure": {
                stage: self._class_structure(stage)
                for stage in (
                    "source_object",
                    "injected_prompt",
                    "contextualized_prompt",
                )
            },
            "storage_mode": "aggregate_only",
            "posterior_interpretation_allowed": False,
        }


def build_object_selection_report(
    registry: Mapping[str, Any],
    hierarchy_trace: Mapping[str, Any],
) -> Dict[str, Any]:
    correspondence = hierarchy_trace.get("distance_correspondence", {})
    trace_variance = hierarchy_trace.get("normalized_trace_variance", {})
    class_structure = hierarchy_trace.get("class_structure", {})
    mapping = {
        "raw_latent": "source_to_logit_effect_distance_spearman",
        "injected_prompt": "injected_prompt_to_cls_delta_distance_spearman",
        "contextualized_prompt": "contextualized_prompt_to_logit_delta_distance_spearman",
        "cls_effect": "cls_effect_to_logit_effect_distance_spearman",
        "logit_effect": None,
    }
    cards = {}
    for name, state in registry.get("candidate_spaces", {}).items():
        if not state.get("requested", False):
            cards[name] = {"status": "not_applicable", "reason": "not_requested"}
            continue
        if not state.get("applicable", False):
            cards[name] = {
                "status": "not_applicable",
                "reason": state.get("failure_reason"),
            }
            continue
        relation_name = mapping.get(name)
        relation = correspondence.get(relation_name, {}) if relation_name else {}
        relation_value = relation.get("spearman")
        observed = name in hierarchy_trace.get("observed_stages", []) or (
            name == "raw_latent" and "source_object" in hierarchy_trace.get("observed_stages", [])
        )
        variance_stage = "source_object" if name == "raw_latent" else name
        candidate_variance = trace_variance.get(variance_stage)
        cls_variance = trace_variance.get("cls_effect")
        logit_variance = trace_variance.get("logit_effect")
        if candidate_variance is None:
            causal_status = "insufficient_evidence"
        elif candidate_variance <= 0.0:
            causal_status = "fail"
        elif name == "logit_effect":
            causal_status = "pass"
        elif max(cls_variance or 0.0, logit_variance or 0.0) > 0.0:
            causal_status = "pass"
        else:
            causal_status = "fail"
        cards[name] = {
            "status": "insufficient_evidence",
            "causal_access": {
                "status": causal_status if observed else "insufficient_evidence",
                "normalized_trace_variance": candidate_variance,
                "cls_effect_normalized_trace_variance": cls_variance,
                "logit_effect_normalized_trace_variance": logit_variance,
            },
            "functional_correspondence": {
                "status": (
                    "pass"
                    if relation_value is not None and relation_value > 0.0
                    else "fail"
                    if relation_value is not None
                    else "insufficient_evidence"
                ),
                "metric": relation_name,
                "value": relation_value,
            },
            "class_structure": class_structure.get(
                "source_object" if name == "raw_latent" else name,
                {"status": "not_observed"},
            ),
            "effective_dimension": {"status": "not_observed_jacobian_deferred"},
            "heldout_transfer": {"status": "not_observed"},
            "cross_seed_stability": {"status": "not_observed"},
            "coordinate_robustness": {"status": "not_observed"},
            "operational_cost": {
                "status": "observed",
                "execution": "checkpoint_fixed_probe_replay",
            },
        }
    insufficient = [
        name
        for name, card in cards.items()
        if card.get("status") == "insufficient_evidence"
    ]
    rejected = [
        name for name, card in cards.items() if card.get("status") == "fail"
    ]
    return {
        "format": "bayesian_object_selection_report_v1",
        "recommended_candidate": None,
        "rejected_candidates": rejected,
        "insufficient_evidence_candidates": insufficient,
        "candidate_cards": cards,
        "evidence_summary": (
            "controlled perturbation establishes local functional propagation only; "
            "held-out and cross-training-seed evidence is still required before selection"
        ),
        "automatic_composite_score_used": False,
        "posterior_metrics_deferred": True,
    }


def numeric_leaf_metrics(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    output: Dict[str, float] = {}
    for name, value in payload.items():
        key = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(value, Mapping):
            output.update(numeric_leaf_metrics(value, key))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if math.isfinite(value):
                output[key] = value
    return output
