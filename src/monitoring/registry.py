"""Declarative registry for monitor groups and their activation rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple


@dataclass(frozen=True)
class MonitorSpec:
    namespace: str
    cadence: str
    artifact: str
    sampling: str
    description: str
    source_requirement: str
    is_requested: Callable[[Any], bool]
    is_source_active: Callable[[Any], bool]


def _always(_: Any) -> bool:
    return True


def _prompt_distribution_source_active(cfg: Any) -> bool:
    return bool(cfg.MODEL.PROMPT.ENABLE) and str(cfg.MODEL.PROMPT.INIT_SOURCE).lower() == "distributor_mean"


def _attention_mediation_source_active(cfg: Any) -> bool:
    return (
        bool(cfg.MODEL.PROMPT.ENABLE)
        and bool(cfg.MODEL.SEMANTIC_TOKENS.ENABLE)
        and int(cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS) > 0
        and bool(cfg.MODEL.ATTENTION_MEDIATION.ENABLE)
    )


def _affinity_source_active(cfg: Any) -> bool:
    return bool(cfg.MODEL.AFFINITY.ENABLE) or (
        bool(cfg.MONITOR.ENABLE) and bool(cfg.MONITOR.AFFINITY.ENABLE)
    )


def _auxiliary_loss_source_active(cfg: Any) -> bool:
    main_alignment_weight = (
        float(cfg.SOLVER.RSIM.ALIGN_WEIGHT)
        if str(cfg.SOLVER.RSIM.ALIGN_MODE).lower() != "none"
        else 0.0
    )
    scalar_weights = (
        main_alignment_weight,
        cfg.SOLVER.LOSS_SEM_MED_WEIGHT,
        cfg.SOLVER.LOSS_SPV_WEIGHT,
        cfg.SOLVER.LOSS_ATTR_WEIGHT,
        cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT,
        cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT,
    )
    return any(float(value) > 0.0 for value in scalar_weights)


MONITOR_SPECS: Tuple[MonitorSpec, ...] = (
    MonitorSpec(
        namespace="train",
        cadence="step",
        artifact="step_jsonl",
        sampling="global_step",
        description="training loss and learning rate",
        source_requirement="every completed training batch",
        is_requested=_always,
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="train_epoch",
        cadence="epoch",
        artifact="epoch_csv",
        sampling="every_epoch",
        description="epoch-level training loss, learning rate, and timing",
        source_requirement="completed training epoch",
        is_requested=_always,
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="train_debug",
        cadence="step",
        artifact="step_jsonl",
        sampling="global_step",
        description="compact CE logit health and one-time CE/raw path check",
        source_requirement="trainer debug cache",
        is_requested=_always,
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="numerical_guard",
        cadence="event",
        artifact="events_jsonl",
        sampling="on_event",
        description="first-step numerical pass evidence and non-finite failures",
        source_requirement="completed forward/backward/optimizer step",
        is_requested=lambda cfg: bool(cfg.MONITOR.NUMERICAL_GUARD.ENABLE),
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="optimizer_sanity",
        cadence="event",
        artifact="events_jsonl",
        sampling="on_event",
        description="compact optimizer membership, gradients, and first update evidence",
        source_requirement="Trainer optimizer and first completed update",
        is_requested=lambda cfg: bool(cfg.MONITOR.OPTIMIZER_SANITY.ENABLE),
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="graph_prob_prior",
        cadence="step",
        artifact="step_jsonl",
        sampling="graph_prob_prior_forward",
        description="GraphProbPrior loss and geometry diagnostics",
        source_requirement="MODEL.GRAPH_PROB_PRIOR.ENABLE and LOSS_WEIGHT>0",
        is_requested=lambda cfg: bool(cfg.MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE),
        is_source_active=lambda cfg: (
            bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE)
            and float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT) > 0.0
        ),
    ),
    MonitorSpec(
        namespace="prompt_distribution",
        cadence="step",
        artifact="step_jsonl",
        sampling="global_step",
        description="instance-conditioned prompt distribution summaries",
        source_requirement="MODEL.PROMPT.ENABLE and INIT_SOURCE=distributor_mean",
        is_requested=lambda cfg: bool(cfg.MONITOR.PROMPT.ENABLE),
        is_source_active=_prompt_distribution_source_active,
    ),
    MonitorSpec(
        namespace="prompt_parameter_health",
        cadence="epoch",
        artifact="epoch_csv",
        sampling="every_epoch",
        description="static prompt parameter norm, gradient, update, and collapse summaries",
        source_requirement="MODEL.PROMPT.ENABLE with trainable prompt parameters",
        is_requested=lambda cfg: bool(cfg.MONITOR.PROMPT_PARAMETER.ENABLE),
        is_source_active=lambda cfg: bool(cfg.MODEL.PROMPT.ENABLE),
    ),
    MonitorSpec(
        namespace="semantic_token_health",
        cadence="step",
        artifact="step_jsonl",
        sampling="global_step",
        description="semantic token scale, diversity, and input/output identity summaries",
        source_requirement="MODEL.SEMANTIC_TOKENS.ENABLE",
        is_requested=lambda cfg: bool(cfg.MONITOR.SEMANTIC_TOKEN.ENABLE),
        is_source_active=lambda cfg: bool(cfg.MODEL.SEMANTIC_TOKENS.ENABLE),
    ),
    MonitorSpec(
        namespace="attention_mediation",
        cadence="step",
        artifact="step_jsonl",
        sampling="global_step",
        description="per-layer attention mediation summaries",
        source_requirement="MODEL.PROMPT.ENABLE and MODEL.ATTENTION_MEDIATION.ENABLE",
        is_requested=lambda cfg: bool(cfg.MONITOR.ATTENTION_MEDIATION.ENABLE),
        is_source_active=_attention_mediation_source_active,
    ),
    MonitorSpec(
        namespace="affinity_summary",
        cadence="step",
        artifact="step_jsonl",
        sampling="global_step",
        description="layer-aggregated prompt and visual affinity summaries",
        source_requirement="MODEL.AFFINITY.ENABLE or requested affinity monitoring",
        is_requested=lambda cfg: bool(cfg.MONITOR.AFFINITY.ENABLE),
        is_source_active=_affinity_source_active,
    ),
    MonitorSpec(
        namespace="auxiliary_loss_health",
        cadence="step",
        artifact="step_jsonl",
        sampling="global_step",
        description="finite scalar auxiliary-loss balance and activity summaries",
        source_requirement="at least one non-zero auxiliary loss weight",
        is_requested=lambda cfg: bool(cfg.MONITOR.AUXILIARY_LOSS.ENABLE),
        is_source_active=_auxiliary_loss_source_active,
    ),
    MonitorSpec(
        namespace="classification",
        cadence="epoch",
        artifact="epoch_csv",
        sampling="every_epoch",
        description="dataset-level classification evaluation metrics",
        source_requirement="completed evaluator pass",
        is_requested=_always,
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="prediction_health",
        cadence="epoch",
        artifact="epoch_csv",
        sampling="every_epoch",
        description="compact joint-space bias, decision quality, and error confidence",
        source_requirement="completed joint-logit evaluator pass",
        is_requested=lambda cfg: bool(cfg.MONITOR.PREDICTION_HEALTH.ENABLE),
        is_source_active=lambda cfg: str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() == "final_gzsl",
    ),
    MonitorSpec(
        namespace="class_error",
        cadence="epoch",
        artifact="epoch_csv",
        sampling="every_epoch",
        description="compact worst-class and prediction-hub summary with vector artifacts",
        source_requirement="completed evaluator pass",
        is_requested=lambda cfg: bool(cfg.MONITOR.CLASS_ERROR.ENABLE),
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="calibration_profile",
        cadence="epoch",
        artifact="epoch_csv",
        sampling="every_epoch",
        description="compact fixed-grid final_gzsl bias-gap and curve-quality summary",
        source_requirement="same-epoch test_seen and test_unseen joint logits",
        is_requested=lambda cfg: bool(cfg.MONITOR.CALIBRATION.ENABLE),
        is_source_active=lambda cfg: str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() == "final_gzsl",
    ),
    MonitorSpec(
        namespace="checkpoint_selection_debug",
        cadence="epoch",
        artifact="epoch_csv",
        sampling="every_epoch",
        description="historical independent-best upper bound, never a formal result",
        source_requirement="same run has seen and unseen evaluation history",
        is_requested=_always,
        is_source_active=lambda cfg: str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() == "final_gzsl",
    ),
    MonitorSpec(
        namespace="monitor_initialized",
        cadence="event",
        artifact="events_jsonl",
        sampling="on_event",
        description="monitoring session initialization event",
        source_requirement="Trainer created MonitorManager",
        is_requested=_always,
        is_source_active=_always,
    ),
    MonitorSpec(
        namespace="monitor_finalized",
        cadence="event",
        artifact="events_jsonl",
        sampling="on_event",
        description="monitoring session finalization event",
        source_requirement="Trainer training lifecycle exit",
        is_requested=_always,
        is_source_active=_always,
    ),
)

MONITOR_SPEC_BY_NAMESPACE = {spec.namespace: spec for spec in MONITOR_SPECS}


def get_monitor_spec(namespace: str) -> MonitorSpec:
    try:
        return MONITOR_SPEC_BY_NAMESPACE[str(namespace)]
    except KeyError as exc:
        raise KeyError(f"Unknown monitor namespace: {namespace}") from exc


def resolve_monitor_groups(cfg: Any) -> Dict[str, Dict[str, Any]]:
    monitor_enabled = bool(cfg.MONITOR.ENABLE)
    return {
        spec.namespace: {
            "requested": bool(spec.is_requested(cfg)),
            "source_active": bool(spec.is_source_active(cfg)),
            "effective": bool(monitor_enabled and spec.is_requested(cfg) and spec.is_source_active(cfg)),
            "cadence": spec.cadence,
            "artifact": spec.artifact,
            "sampling": spec.sampling,
            "source_requirement": spec.source_requirement,
            "description": spec.description,
        }
        for spec in MONITOR_SPECS
    }
