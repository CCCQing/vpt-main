"""Stable schemas for experiment monitoring outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MonitorContext:
    run_id: str
    session_id: Optional[str]
    cell_id: Optional[str]
    seed: Optional[int]
    stage: str
    epoch: Optional[int]
    global_step: Optional[int]
    graph_prob_prior_forward: Optional[int] = None


EPOCH_CSV_FIELDS = (
    "schema_version",
    "run_id",
    "session_id",
    "cell_id",
    "seed",
    "stage",
    "epoch",
    "global_step",
    "graph_prob_prior_forward",
    "split",
    "namespace",
    "metric",
    "value",
    "reducer",
    "n",
)
