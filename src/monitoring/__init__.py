"""Structured, low-overhead monitoring interfaces shared by training and Stage-2."""

from .collector import MonitorManager
from .diagnostics import DiagnosticManager
from .fields import GPP_MONITOR_ALIASES, GPP_MONITOR_ALIAS_ITEMS, MetricFieldSpec
from .guards import NumericalGuard, OptimizerSanity, PromptParameterTracker
from .multi_loss_gradient import MultiLossGradientAuditor, ParameterBlockRegistry
from .representation_transport import (
    classifier_logits_from_features,
    head_representation_factorial_metrics,
    prediction_transition_group_metrics,
    representation_transport_metrics,
    semantic_transport_metrics,
)
from .registry import MONITOR_SPECS, get_monitor_spec
from .writer import stage2_metadata, write_stage2_json, write_stage2_table

__all__ = [
    "MonitorManager",
    "DiagnosticManager",
    "NumericalGuard",
    "OptimizerSanity",
    "PromptParameterTracker",
    "MultiLossGradientAuditor",
    "ParameterBlockRegistry",
    "representation_transport_metrics",
    "semantic_transport_metrics",
    "classifier_logits_from_features",
    "head_representation_factorial_metrics",
    "prediction_transition_group_metrics",
    "MetricFieldSpec",
    "MONITOR_SPECS",
    "GPP_MONITOR_ALIASES",
    "GPP_MONITOR_ALIAS_ITEMS",
    "get_monitor_spec",
    "stage2_metadata",
    "write_stage2_json",
    "write_stage2_table",
]
