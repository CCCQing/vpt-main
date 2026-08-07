from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_if_available(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _drop_path(payload: Dict[str, Any], path: Sequence[str]) -> None:
    current: Any = payload
    for name in path[:-1]:
        if not isinstance(current, dict) or name not in current:
            return
        current = current[name]
    if isinstance(current, dict):
        current.pop(path[-1], None)


def build_comparability_identity(
    cfg,
    *,
    run_id: str,
    session_id: str,
    checkpoint_manifest: Mapping[str, Any],
    probe_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    output_root = Path(str(cfg.OUTPUT_DIR))
    resolved_config = _plain(cfg)
    artifact_hashes = {
        name: _file_sha256(output_root / name)
        for name in (
            "resolved_config.yaml",
            "dataset_manifest.json",
            "reproducibility_manifest.json",
            "trainable_parameters.json",
        )
    }
    runtime_summary = _read_json_if_available(output_root / "monitor_runtime_summary.json")
    run_payload = {
        "resolved_config": resolved_config,
        "seed": int(cfg.SEED),
        "run_id": str(run_id),
        "session_id": str(session_id),
        "artifact_sha256": artifact_hashes,
        "git": runtime_summary.get("git"),
        "checkpoint": dict(checkpoint_manifest),
        "probe_manifest_sha256_by_split": {
            str(split): manifest.get("manifest_sha256")
            for split, manifest in probe_manifest.get("probes", {}).items()
        },
    }
    shared_config = deepcopy(resolved_config)
    for excluded in (
        ("OUTPUT_DIR",),
        ("SEED",),
        ("MODEL", "PROMPT", "ENABLE"),
        ("MODEL", "PROMPT", "BACKEND"),
        ("MODEL", "PROMPT", "DEEP"),
    ):
        _drop_path(shared_config, excluded)
    shared_payload = {
        "resolved_config_with_declared_exclusions": shared_config,
        "declared_exclusions": [
            "OUTPUT_DIR",
            "SEED",
            "MODEL.PROMPT.ENABLE",
            "MODEL.PROMPT.BACKEND",
            "MODEL.PROMPT.DEEP",
        ],
        "dataset_manifest_sha256": artifact_hashes["dataset_manifest.json"],
        "protocol_mode": str(cfg.DATA.XLSA.PROTOCOL_MODE),
        "backbone_identity": {
            "feature": str(cfg.DATA.FEATURE),
            "crop_size": int(cfg.DATA.CROPSIZE),
            "model_root": str(cfg.MODEL.MODEL_ROOT),
        },
        "checkpoint_selection_rule": checkpoint_manifest.get("checkpoint_selection_rule"),
        "probe_selection": {
            str(split): {
                "probe_id": manifest.get("probe_id"),
                "manifest_sha256": manifest.get("manifest_sha256"),
                "selection_seed": manifest.get("selection_seed"),
                "candidate_class_ids": manifest.get("candidate_class_ids"),
            }
            for split, manifest in probe_manifest.get("probes", {}).items()
        },
    }
    return {
        "format": "a_series_comparability_v1",
        "run_identity": {
            "sha256": _canonical_hash(run_payload),
            "payload": run_payload,
        },
        "shared_condition_fingerprint": {
            "sha256": _canonical_hash(shared_payload),
            "payload": shared_payload,
        },
    }
