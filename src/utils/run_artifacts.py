#!/usr/bin/env python3

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from .distributed import get_rank


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary_path, path)


def _atomic_json_dump(path: str, payload: Dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_resolved_config(cfg) -> Optional[str]:
    if get_rank() != 0:
        return None
    config_text = cfg.dump()
    path = os.path.join(str(cfg.OUTPUT_DIR), "resolved_config.yaml")
    _atomic_write_text(path, config_text)
    return path


def write_trainable_parameter_manifest(cfg, model) -> Optional[str]:
    if get_rank() != 0:
        return None

    model_ref = model.module if hasattr(model, "module") else model
    tensors = [
        {
            "name": str(name),
            "shape": [int(dim) for dim in parameter.shape],
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model_ref.named_parameters()
        if parameter.requires_grad
    ]
    total_parameters = int(sum(parameter.numel() for parameter in model_ref.parameters()))
    trainable_parameters = int(sum(item["numel"] for item in tensors))
    resolved_config = cfg.dump()
    payload = {
        "schema_version": "trainable_parameter_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": "{}.{}".format(model_ref.__class__.__module__, model_ref.__class__.__name__),
        "classifier": str(cfg.MODEL.CLASSIFIER),
        "resolved_config_sha256": _sha256_text(resolved_config),
        "total_parameters": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "trainable_tensor_count": len(tensors),
        "trainable_tensors": tensors,
    }
    path = os.path.join(str(cfg.OUTPUT_DIR), "trainable_parameters.json")
    _atomic_json_dump(path, payload)
    return path
