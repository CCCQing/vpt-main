"""Shared evidence-role rules for fixed-Probe analysis outputs.

The rule changes analysis routing only.  It does not alter Probe sampling,
model forward passes, metric formulas, or the long-table schema.
"""

from __future__ import annotations

from typing import Any, Mapping


PROBE_CONTEXT_ONLY = "probe_context_only"
VALIDITY_OR_IDENTITY = "validity_or_identity"
MECHANISM_EVIDENCE = "mechanism_evidence"

_CONTEXT_DOMAINS = {
    "classification",
    "prediction_health",
    "class_error",
    "probe_context",
}


def probe_record_evidence_role(row: Mapping[str, Any]) -> str:
    """Classify one existing ``probe_metrics.csv`` row by analysis role."""

    domain = str(row.get("domain") or "").strip().lower()
    condition = str(row.get("condition") or "normal").strip().lower()
    entity_type = str(row.get("entity_type") or "").strip().lower()

    if domain.endswith("forward_equivalence") or entity_type in {
        "equivalence",
        "identity_check",
        "validity_check",
    }:
        return VALIDITY_OR_IDENTITY
    if condition == "normal" and domain in _CONTEXT_DOMAINS:
        return PROBE_CONTEXT_ONLY
    return MECHANISM_EVIDENCE


def is_probe_mechanism_evidence(row: Mapping[str, Any]) -> bool:
    return probe_record_evidence_role(row) == MECHANISM_EVIDENCE

