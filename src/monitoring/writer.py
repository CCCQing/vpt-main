"""Reusable writers for structured monitors and Stage-2 tables/metadata."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from .schema import SCHEMA_VERSION


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def stage2_metadata(stage: str, *, seed: Optional[int] = None, cell_id: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "stage": str(stage)}
    if seed is not None:
        result["model_seed"] = int(seed)
    if cell_id:
        result["cell_id"] = str(cell_id)
    return result


def write_stage2_table(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    decorated = [{**dict(metadata), **dict(row)} for row in rows]
    write_table(path, decorated)


def write_stage2_json(path: Path, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    write_json(path, {**dict(metadata), **dict(payload)})
