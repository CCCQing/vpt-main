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
from .epoch_transition import EpochPredictionTransitionTracker
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
        if (
            self.enabled
            and bool(cfg.MONITOR.PREDICTION_TRANSITION_TRAJECTORY.ENABLE)
            and (int(cfg.NUM_GPUS) != 1 or int(cfg.NUM_SHARDS) != 1)
        ):
            raise ValueError(
                "epoch prediction transition currently requires the single-GPU, single-shard evaluator contract"
            )
        self.root = Path(str(cfg.OUTPUT_DIR)) / "diagnostics"
        self.output_policy = str(cfg.MONITOR.OUTPUT_POLICY).lower()
        self.run_id = str(monitor_manager.run_id)
        self.session_id = str(monitor_manager.session_id)
        self.started_at = self._utc_now()
        self.finalized = False
        self.calibration_pending: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.epoch_transition_tracker = EpochPredictionTransitionTracker()
        self.runtime = {
            "static_semantic_graph": self._new_state("one_time_evidence"),
            "prediction_health": self._new_state("eval_diagnostics"),
            "class_error": self._new_state("eval_diagnostics"),
            "epoch_prediction_transition": self._new_state("cross_epoch_eval_diagnostics"),
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
            "schema_version": 2,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "protocol_mode": str(self.cfg.DATA.XLSA.PROTOCOL_MODE),
            "execution_contracts": {
                "static_semantic_graph": {"kind": "one_time_evidence", "intrusive": False},
                "prediction_health": {"kind": "eval_diagnostics", "intrusive": False, "extra_forward": False},
                "class_error": {"kind": "eval_diagnostics", "intrusive": False, "extra_forward": False},
                "epoch_prediction_transition": {
                    "kind": "cross_epoch_eval_diagnostics",
                    "intrusive": False,
                    "extra_forward": False,
                    "sample_state_persisted": False,
                    "aggregate_artifacts_only": True,
                    "single_gpu_single_shard_required": True,
                    "resume_without_previous_state": "invalid_until_next_consecutive_pair",
                },
                "calibration_profile": {"kind": "eval_diagnostics", "intrusive": False, "extra_forward": False},
                "fixed_probe": {"kind": "fixed_probe", "intrusive": False, "requires_probe_manifest": True},
                "bayesian_object_selection": {
                    "kind": "checkpoint_fixed_probe",
                    "intrusive": True,
                    "requires_probe_manifest": True,
                    "same_checkpoint": True,
                    "same_sample_and_candidate_identity": True,
                    "default_enabled": False,
                    "posterior_interpretation_allowed": False,
                },
                "module_effect": {
                    "kind": "paired_intervention",
                    "intrusive": True,
                    "requires_probe_manifest": True,
                    "same_checkpoint": True,
                    "same_probe_manifest": True,
                },
            },
            "resolved_switches": {
                "prediction_health": bool(self.cfg.MONITOR.PREDICTION_HEALTH.ENABLE),
                "class_error": bool(self.cfg.MONITOR.CLASS_ERROR.ENABLE),
                "epoch_prediction_transition": bool(
                    self.cfg.MONITOR.PREDICTION_TRANSITION_TRAJECTORY.ENABLE
                ),
                "calibration_profile": bool(self.cfg.MONITOR.CALIBRATION.ENABLE),
                "fixed_probe": bool(self.cfg.MONITOR.PROBE.ENABLE),
                "bayesian_object_selection": bool(
                    self.cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION.ENABLE
                ),
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
        dataset: Any,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        if not self.enabled:
            return {"prediction_health": {}, "class_error": {}}
        score_matrix = _numpy(scores).astype(np.float32, copy=False)
        local_targets = _numpy(targets_local).astype(np.int64, copy=False)
        candidate = np.asarray(list(dataset.eval_local_classes), dtype=np.int64)
        position = {"epoch": int(epoch), "split": str(split)}
        if bool(self.cfg.MONITOR.CALIBRATION.ENABLE) and str(split) in {"test_seen", "test_unseen"}:
            self.calibration_pending[(int(epoch), str(split))] = {
                "scores": score_matrix,
                "targets_local": local_targets,
                "candidate_global_ids": candidate,
                "seen_global_ids": np.asarray(list(getattr(dataset, "seen_classes", [])), dtype=np.int64),
            }

        prediction = {}
        if bool(self.cfg.MONITOR.PREDICTION_HEALTH.ENABLE):
            prediction = prediction_health_metrics(
                score_matrix,
                local_targets,
                candidate,
                getattr(dataset, "seen_classes", []),
            )
            self._write_json_artifact(
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
        transition = self._record_epoch_prediction_transition(
            epoch=int(epoch),
            split=str(split),
            scores=score_matrix,
            targets_local=local_targets,
            candidate_global_ids=candidate,
            sample_ids=sample_ids,
        )
        return {
            "prediction_health": prediction,
            "class_error": class_summary,
            "epoch_prediction_transition": transition,
        }

    def _record_epoch_prediction_transition(
        self,
        *,
        epoch: int,
        split: str,
        scores: np.ndarray,
        targets_local: np.ndarray,
        candidate_global_ids: np.ndarray,
        sample_ids: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        cfg = self.cfg.MONITOR.PREDICTION_TRANSITION_TRAJECTORY
        requested_splits = {str(item) for item in list(cfg.SPLITS)}
        if not bool(cfg.ENABLE) or str(split) not in requested_splits:
            return {}
        if sample_ids is None:
            raise ValueError(
                "epoch prediction transition requires stable sample_ids from the evaluator"
            )
        payload = self.epoch_transition_tracker.update(
            epoch=int(epoch),
            split=str(split),
            sample_ids=sample_ids,
            predictions=np.asarray(scores).argmax(axis=1),
            targets_local=targets_local,
            candidate_global_ids=candidate_global_ids,
        )
        position = {"epoch": int(epoch), "split": str(split)}
        arrays = payload.pop("arrays")
        vectors_path = None
        vectors_sha256 = None
        if arrays:
            vectors_path = self._write_npz_artifact(
                "epoch_prediction_transition",
                f"epoch_prediction_transition/epoch_{int(epoch):04d}/{split}.npz",
                arrays,
                position,
            )
            vectors_sha256 = _sha256(vectors_path)
        candidate_ids = np.asarray(payload.pop("candidate_global_ids"), dtype=np.int64)
        json_payload = {
            **payload,
            "candidate_global_ids": candidate_ids.tolist(),
            "storage_mode": "aggregate_only",
            "sample_ids_persisted": False,
            "previous_epoch_sample_state_in_memory_only": True,
            "vectors_path": (
                str(vectors_path.relative_to(self.root)).replace("\\", "/")
                if vectors_path is not None else None
            ),
            "vectors_sha256": vectors_sha256,
        }
        self._write_json_artifact(
            "epoch_prediction_transition",
            f"epoch_prediction_transition/epoch_{int(epoch):04d}/{split}.json",
            json_payload,
            position,
        )
        self.monitor_manager.record_epoch(
            str(split),
            "epoch_prediction_transition",
            payload["summary"],
            reducer="dataset",
            n=int(scores.shape[0]),
        )
        return json_payload

    def record_calibration(self, epoch: int) -> Dict[str, float]:
        if not self.enabled or not bool(self.cfg.MONITOR.CALIBRATION.ENABLE):
            return {}
        seen_key = (int(epoch), "test_seen")
        unseen_key = (int(epoch), "test_unseen")
        seen = self.calibration_pending.get(seen_key)
        unseen = self.calibration_pending.get(unseen_key)
        if seen is None or unseen is None:
            return {}
        seen = self.calibration_pending.pop(seen_key)
        unseen = self.calibration_pending.pop(unseen_key)
        if not np.array_equal(seen["candidate_global_ids"], unseen["candidate_global_ids"]):
            raise ValueError("test_seen and test_unseen calibration inputs use different candidate class order")
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
        compressed_path = path.with_name(path.name + ".gz")
        if compressed_path.exists() and not path.exists():
            raise RuntimeError(
                "Cannot append fixed-probe metrics after probe_metrics.csv was compacted; use a new diagnostic output directory."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id", "session_id", "checkpoint_id", "probe_id", "selection_seed",
            "probe_manifest_sha256", "split",
            "condition", "domain", "entity_type", "entity_id", "metric", "value",
        ]
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                existing_fields = list(reader.fieldnames or [])
                condition_only_fields = [
                    name for name in fieldnames
                    if name not in {"selection_seed", "probe_manifest_sha256"}
                ]
                legacy_fields = [
                    name for name in condition_only_fields if name != "condition"
                ]
                legacy_schema = existing_fields in (
                    legacy_fields, condition_only_fields
                )
                existing_rows = list(reader) if legacy_schema else None
            if legacy_schema:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in existing_rows or []:
                        writer.writerow({
                            name: (
                                "normal" if name == "condition" and "condition" not in row
                                else json_safe(row.get(name))
                            )
                            for name in fieldnames
                        })
            elif existing_fields != fieldnames:
                raise ValueError(
                    "probe_metrics.csv schema does not match the current probe-identity format"
                )
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({name: json_safe(row.get(name)) for name in fieldnames})
        self._observe("fixed_probe", path, {"stage": "fixed_probe"})
        return path

    def record_probe_artifact(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        position = {"stage": "fixed_probe"}
        return self._write_json_artifact("fixed_probe", relative_path, payload, position)

    def record_module_effect_artifact(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        position = {"stage": "module_effect"}
        return self._write_json_artifact("module_effect", relative_path, payload, position)

    def finalize(self, *, status: str) -> None:
        if not self.enabled or self.finalized:
            return
        self.calibration_pending.clear()
        self.epoch_transition_tracker.clear()
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
