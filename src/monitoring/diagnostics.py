from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .eval_metrics import (
    calibration_profile_metrics,
    class_error_metrics,
    prediction_health_metrics,
    semantic_graph_reference_metrics,
)
from .writer import json_safe, write_json


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiagnosticManager:
    def __init__(self, cfg: Any, monitor_manager: Any, *, is_writer: bool = True) -> None:
        self.cfg = cfg
        self.monitor_manager = monitor_manager
        self.enabled = bool(cfg.MONITOR.ENABLE) and bool(cfg.MONITOR.DIAGNOSTICS.ENABLE) and bool(is_writer)
        self.root = Path(str(cfg.OUTPUT_DIR)) / "diagnostics"
        self.output_policy = str(cfg.MONITOR.OUTPUT_POLICY).lower()
        self.run_id = str(monitor_manager.run_id)
        self.session_id = str(monitor_manager.session_id)
        self.started_at = self._utc_now()
        self.finalized = False
        self.eval_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.runtime = {
            "static_semantic_graph": self._new_state("one_time_evidence"),
            "prediction_health": self._new_state("eval_diagnostics"),
            "class_error": self._new_state("eval_diagnostics"),
            "calibration_profile": self._new_state("eval_diagnostics"),
            "fixed_probe": self._new_state("fixed_probe"),
            "module_effect": self._new_state("paired_intervention"),
        }
        if self.enabled:
            self._prepare_root()
            self._write_manifest()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_state(execution_kind: str) -> Dict[str, Any]:
        return {
            "execution_kind": str(execution_kind),
            "observed": False,
            "write_count": 0,
            "artifacts": [],
            "first_observed": None,
            "last_observed": None,
        }

    def _prepare_root(self) -> None:
        manifest = self.root / "diagnostic_manifest.json"
        if not manifest.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            return
        if self.output_policy == "error_if_exists":
            raise FileExistsError(
                f"Diagnostic artifacts already exist in {self.root}. Use a new OUTPUT_DIR or resume/overwrite."
            )
        if self.output_policy == "overwrite":
            resolved_root = self.root.resolve()
            resolved_output = Path(str(self.cfg.OUTPUT_DIR)).resolve()
            if resolved_root.parent != resolved_output:
                raise RuntimeError("refusing to overwrite a diagnostic directory outside OUTPUT_DIR")
            shutil.rmtree(resolved_root)
            self.root.mkdir(parents=True, exist_ok=True)
            return
        summary_path = self.root / "diagnostic_runtime_summary.json"
        if summary_path.exists():
            try:
                old = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                old = {}
            if str(old.get("session_id", "")) == self.session_id:
                for name, state in old.get("domains", {}).items():
                    if name in self.runtime and isinstance(state, Mapping):
                        self.runtime[name].update({
                            "observed": bool(state.get("observed", False)),
                            "write_count": int(state.get("write_count", 0)),
                            "artifacts": list(state.get("artifacts", [])),
                            "first_observed": state.get("first_observed"),
                            "last_observed": state.get("last_observed"),
                        })

    def _write_manifest(self) -> None:
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "protocol_mode": str(self.cfg.DATA.XLSA.PROTOCOL_MODE),
            "execution_contracts": {
                "static_semantic_graph": {"kind": "one_time_evidence", "intrusive": False},
                "prediction_health": {"kind": "eval_diagnostics", "intrusive": False, "extra_forward": False},
                "class_error": {"kind": "eval_diagnostics", "intrusive": False, "extra_forward": False},
                "calibration_profile": {"kind": "eval_diagnostics", "intrusive": False, "extra_forward": False},
                "fixed_probe": {"kind": "fixed_probe", "intrusive": False, "requires_probe_manifest": True},
                "module_effect": {
                    "kind": "paired_intervention",
                    "intrusive": True,
                    "requires_probe_manifest": True,
                    "same_checkpoint": True,
                    "same_probe_manifest": True,
                },
            },
            "resolved_switches": {
                "save_eval_cache": bool(self.cfg.MONITOR.DIAGNOSTICS.SAVE_EVAL_CACHE),
                "prediction_health": bool(self.cfg.MONITOR.PREDICTION_HEALTH.ENABLE),
                "class_error": bool(self.cfg.MONITOR.CLASS_ERROR.ENABLE),
                "calibration_profile": bool(self.cfg.MONITOR.CALIBRATION.ENABLE),
                "fixed_probe": bool(self.cfg.MONITOR.PROBE.ENABLE),
                "module_effect": bool(self.cfg.MONITOR.MODULE_EFFECT.ENABLE),
            },
        }
        write_json(self.root / "diagnostic_manifest.json", payload)

    def _observe(self, domain: str, path: Path, position: Mapping[str, Any]) -> None:
        state = self.runtime[domain]
        relative = str(path.relative_to(self.root)).replace("\\", "/")
        if not state["observed"]:
            state["first_observed"] = dict(position)
        state["observed"] = True
        state["write_count"] += 1
        if relative not in state["artifacts"]:
            state["artifacts"].append(relative)
        state["last_observed"] = dict(position)

    def _write_json_artifact(self, domain: str, relative_path: str, payload: Mapping[str, Any], position: Mapping[str, Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"run_id": self.run_id, "session_id": self.session_id, **dict(payload)})
        self._observe(domain, path, position)
        return path

    def _write_npz_artifact(self, domain: str, relative_path: str, arrays: Mapping[str, Any], position: Mapping[str, Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{str(name): _numpy(value) for name, value in arrays.items()})
        self._observe(domain, path, position)
        return path

    def record_static_semantic_graph(self, dataset: Any) -> Dict[str, float]:
        if not self.enabled or not bool(self.cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.ENABLE):
            return {}
        attributes = getattr(dataset, "class_attributes", None)
        if attributes is None:
            return {}
        metrics = semantic_graph_reference_metrics(
            _numpy(attributes),
            getattr(dataset, "seen_classes", []),
            getattr(dataset, "unseen_classes", []),
            edge_threshold=float(self.cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.EDGE_THRESHOLD),
            neighbor_k=int(self.cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.NEIGHBOR_K),
            temperature=float(self.cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.TEMPERATURE),
        )
        self._write_json_artifact(
            "static_semantic_graph",
            "semantic_graph_reference/static.json",
            {
                "protocol_mode": str(getattr(dataset, "protocol_mode", "unknown")),
                "class_count": int(_numpy(attributes).shape[0]),
                "metrics": metrics,
            },
            {"stage": "initialization"},
        )
        return metrics

    def record_eval(
        self,
        *,
        epoch: int,
        split: str,
        scores: Any,
        targets_local: Any,
        targets_global: Any,
        sample_ids: Sequence[str],
        dataset: Any,
        visual_features: Optional[Any] = None,
    ) -> Dict[str, Dict[str, float]]:
        if not self.enabled:
            return {"prediction_health": {}, "class_error": {}}
        score_matrix = _numpy(scores).astype(np.float64, copy=False)
        local_targets = _numpy(targets_local).astype(np.int64, copy=False)
        global_targets = _numpy(targets_global).astype(np.int64, copy=False)
        candidate = np.asarray(list(dataset.eval_local_classes), dtype=np.int64)
        position = {"epoch": int(epoch), "split": str(split)}
        cache = {
            "scores": score_matrix,
            "targets_local": local_targets,
            "targets_global": global_targets,
            "sample_ids": np.asarray([str(item) for item in sample_ids]),
            "candidate_global_ids": candidate,
            "seen_global_ids": np.asarray(list(getattr(dataset, "seen_classes", [])), dtype=np.int64),
            "unseen_global_ids": np.asarray(list(getattr(dataset, "unseen_classes", [])), dtype=np.int64),
            "visual_features": None if visual_features is None else _numpy(visual_features),
        }
        self.eval_cache[(int(epoch), str(split))] = cache
        if bool(self.cfg.MONITOR.DIAGNOSTICS.SAVE_EVAL_CACHE):
            arrays = {name: value for name, value in cache.items() if value is not None}
            self._write_npz_artifact(
                "prediction_health",
                f"eval_cache/epoch_{int(epoch):04d}/{split}.npz",
                arrays,
                position,
            )

        prediction = {}
        if bool(self.cfg.MONITOR.PREDICTION_HEALTH.ENABLE):
            prediction = prediction_health_metrics(
                score_matrix,
                local_targets,
                candidate,
                getattr(dataset, "seen_classes", []),
            )
            path = self._write_json_artifact(
                "prediction_health",
                f"prediction_health/epoch_{int(epoch):04d}/{split}.json",
                {"split": str(split), "epoch": int(epoch), "metrics": prediction},
                position,
            )
            self.monitor_manager.record_epoch(
                str(split),
                "prediction_health",
                prediction,
                reducer="dataset",
                n=int(score_matrix.shape[0]),
            )

        class_summary = {}
        if bool(self.cfg.MONITOR.CLASS_ERROR.ENABLE):
            details = class_error_metrics(
                score_matrix,
                local_targets,
                candidate,
                class_names=getattr(dataset, "all_classnames", None),
                class_attributes=_numpy(dataset.class_attributes) if getattr(dataset, "class_attributes", None) is not None else None,
                top_confusions=int(self.cfg.MONITOR.CLASS_ERROR.TOP_CONFUSIONS),
            )
            class_summary = details["summary"]
            vector_path = self._write_npz_artifact(
                "class_error",
                f"class_error/epoch_{int(epoch):04d}/{split}.npz",
                details["arrays"],
                position,
            )
            self._write_json_artifact(
                "class_error",
                f"class_error/epoch_{int(epoch):04d}/{split}.json",
                {
                    "split": str(split),
                    "epoch": int(epoch),
                    "summary": class_summary,
                    "top_confusion_pairs": details["top_confusion_pairs"],
                    "vectors_path": str(vector_path.relative_to(self.root)).replace("\\", "/"),
                    "vectors_sha256": _sha256(vector_path),
                },
                position,
            )
            self.monitor_manager.record_epoch(
                str(split),
                "class_error",
                class_summary,
                reducer="dataset",
                n=int(score_matrix.shape[0]),
            )
        return {"prediction_health": prediction, "class_error": class_summary}

    def record_calibration(self, epoch: int) -> Dict[str, float]:
        if not self.enabled or not bool(self.cfg.MONITOR.CALIBRATION.ENABLE):
            return {}
        seen = self.eval_cache.get((int(epoch), "test_seen"))
        unseen = self.eval_cache.get((int(epoch), "test_unseen"))
        if seen is None or unseen is None:
            return {}
        if not np.array_equal(seen["candidate_global_ids"], unseen["candidate_global_ids"]):
            raise ValueError("test_seen and test_unseen calibration caches use different candidate class order")
        profile = calibration_profile_metrics(
            seen["scores"],
            seen["targets_local"],
            unseen["scores"],
            unseen["targets_local"],
            seen["candidate_global_ids"],
            self.cfg.MONITOR.CALIBRATION.SEEN_CLASS_IDS or seen["seen_global_ids"].tolist(),
            self.cfg.MONITOR.CALIBRATION.GAMMA_GRID,
        )
        position = {"epoch": int(epoch), "split": "test_gzsl"}
        curve_path = self._write_npz_artifact(
            "calibration_profile",
            f"calibration_profile/epoch_{int(epoch):04d}.npz",
            {
                "gamma_grid": profile["gamma_grid"],
                "seen_at_gamma": profile["seen_at_gamma"],
                "unseen_at_gamma": profile["unseen_at_gamma"],
                "h_at_gamma": profile["h_at_gamma"],
            },
            position,
        )
        self._write_json_artifact(
            "calibration_profile",
            f"calibration_profile/epoch_{int(epoch):04d}.json",
            {
                "epoch": int(epoch),
                "protocol_mode": str(self.cfg.DATA.XLSA.PROTOCOL_MODE),
                "summary": profile["summary"],
                "curve_path": str(curve_path.relative_to(self.root)).replace("\\", "/"),
                "curve_sha256": _sha256(curve_path),
            },
            position,
        )
        self.monitor_manager.record_epoch(
            "test_gzsl",
            "calibration_profile",
            profile["summary"],
            reducer="dataset",
            n=int(seen["scores"].shape[0] + unseen["scores"].shape[0]),
        )
        return dict(profile["summary"])

    def append_probe_metrics(self, rows: Sequence[Mapping[str, Any]]) -> Optional[Path]:
        if not self.enabled or not rows:
            return None
        path = self.root / "probe_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id", "session_id", "checkpoint_id", "probe_id", "split",
            "domain", "entity_type", "entity_id", "metric", "value",
        ]
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({name: json_safe(row.get(name)) for name in fieldnames})
        self._observe("fixed_probe", path, {"stage": "fixed_probe"})
        return path

    def record_probe_artifact(self, relative_path: str, payload: Mapping[str, Any], *, is_array: bool = False) -> Path:
        position = {"stage": "fixed_probe"}
        if is_array:
            return self._write_npz_artifact("fixed_probe", relative_path, payload, position)
        return self._write_json_artifact("fixed_probe", relative_path, payload, position)

    def record_module_effect_artifact(self, relative_path: str, payload: Mapping[str, Any], *, is_array: bool = False) -> Path:
        position = {"stage": "module_effect"}
        if is_array:
            return self._write_npz_artifact("module_effect", relative_path, payload, position)
        return self._write_json_artifact("module_effect", relative_path, payload, position)

    def record_module_effect_jsonl(self, relative_path: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = {"run_id": self.run_id, "session_id": self.session_id, **dict(row)}
                handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")
        self._observe("module_effect", path, {"stage": "module_effect"})
        return path

    def finalize(self, *, status: str) -> None:
        if not self.enabled or self.finalized:
            return
        write_json(
            self.root / "diagnostic_runtime_summary.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "started_at": self.started_at,
                "finalized_at": self._utc_now(),
                "status": str(status),
                "domains": self.runtime,
            },
        )
        self.finalized = True
