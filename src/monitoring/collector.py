"""Trainer-owned collector for structured monitor artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import uuid4

from .registry import MONITOR_SPECS, get_monitor_spec, resolve_monitor_groups
from .schema import EPOCH_CSV_FIELDS, MonitorContext, SCHEMA_VERSION
from .writer import json_safe, write_json


class MonitorManager:
    _ARTIFACT_FILENAMES = (
        "metrics_epoch.csv",
        "metrics_step.jsonl",
        "metrics_events.jsonl",
        "monitor_runtime_summary.json",
        "optimizer_sanity.json",
        "multi_loss_gradient_audit.jsonl",
        "multi_loss_gradient_audit_summary.json",
        "multi_loss_parameter_group_manifest.json",
        "multi_loss_gradient_audit_validation.json",
    )

    def __init__(self, cfg: Any, *, is_writer: bool = True) -> None:
        monitor_cfg = cfg.MONITOR
        self.enabled = bool(monitor_cfg.ENABLE) and bool(is_writer)
        self.write_epoch_csv = bool(monitor_cfg.WRITE_EPOCH_CSV)
        self.write_step_jsonl = bool(monitor_cfg.WRITE_STEP_JSONL)
        self.write_events_jsonl = bool(monitor_cfg.WRITE_EVENTS_JSONL)
        self.output_policy = str(monitor_cfg.OUTPUT_POLICY).lower()
        if self.output_policy not in {"error_if_exists", "resume", "overwrite"}:
            raise ValueError("MONITOR.OUTPUT_POLICY must be error_if_exists, resume, or overwrite.")
        self.step_every_n = max(1, int(monitor_cfg.STEP_EVERY_N))
        self.graph_prob_prior_every_n = max(1, int(cfg.MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N))
        self.output_dir = Path(str(cfg.OUTPUT_DIR))
        self.run_id = self.output_dir.name
        self.session_id = uuid4().hex
        self.started_at = self._utc_now()
        self.resumed_at: Optional[str] = None
        self.resumed = False
        self.finalized = False
        self.monitor_groups = resolve_monitor_groups(cfg)
        self.runtime_groups = self._new_runtime_groups()
        self.cell_id = str(cfg.SOLVER.STAGE2_CHECKPOINT_CELL_ID) or None
        self.seed = int(cfg.SEED) if cfg.SEED is not None else None
        self.git_identity = self._collect_git_identity() if self.enabled else None
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._prepare_output_artifacts()
            self._write_runtime_summary("running")
        self.context = MonitorContext(
            run_id=self.run_id,
            session_id=self.session_id,
            cell_id=self.cell_id,
            seed=self.seed,
            stage="init",
            epoch=None,
            global_step=None,
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _artifact_paths(self) -> Dict[str, Path]:
        return {name: self.output_dir / name for name in self._ARTIFACT_FILENAMES}

    def _new_runtime_groups(self) -> Dict[str, Dict[str, Any]]:
        return {
            spec.namespace: {
                "observed": False,
                "write_count": 0,
                "metric_count": 0,
                "observed_fields": set(),
                "first_observed": None,
                "last_observed": None,
            }
            for spec in MONITOR_SPECS
        }

    def _prepare_output_artifacts(self) -> None:
        paths = self._artifact_paths()
        existing = [path for path in paths.values() if path.is_file()]
        if not existing:
            return
        if self.output_policy == "error_if_exists":
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                "Monitoring artifacts already exist in {}: {}. "
                "Use a new OUTPUT_DIR or set MONITOR.OUTPUT_POLICY=resume/overwrite explicitly."
                .format(self.output_dir, names)
            )
        if self.output_policy == "overwrite":
            for path in existing:
                path.unlink()
            return
        self._resume_existing_session(paths)

    def _resume_existing_session(self, paths: Mapping[str, Path]) -> None:
        summary_path = paths["monitor_runtime_summary.json"]
        if not summary_path.is_file():
            raise FileNotFoundError(
                "MONITOR.OUTPUT_POLICY=resume requires an existing monitor_runtime_summary.json."
            )
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot resume monitoring session from {summary_path}.") from exc
        if int(summary.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError(
                "Cannot resume a monitoring artifact with a different schema version. "
                "Use a new OUTPUT_DIR or MONITOR.OUTPUT_POLICY=overwrite."
            )
        if str(summary.get("run_id", "")) != self.run_id:
            raise RuntimeError("Monitoring runtime summary run_id does not match the current OUTPUT_DIR.")
        session_id = str(summary.get("session_id", "")).strip()
        if not session_id:
            raise RuntimeError("Monitoring runtime summary has no session id and cannot be resumed safely.")
        self.session_id = session_id
        self.started_at = str(summary.get("started_at", self.started_at))
        self.resumed_at = self._utc_now()
        self.resumed = True
        self._restore_runtime_groups(summary_path)

    def _restore_runtime_groups(self, path: Path) -> None:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if str(summary.get("session_id", "")) != self.session_id:
            return
        previous = summary.get("groups", {})
        if not isinstance(previous, Mapping):
            return
        for namespace, state in self.runtime_groups.items():
            old = previous.get(namespace)
            if not isinstance(old, Mapping):
                continue
            state["observed"] = bool(old.get("observed", False))
            state["write_count"] = max(0, int(old.get("write_count", 0)))
            state["metric_count"] = max(0, int(old.get("metric_count", 0)))
            state["observed_fields"] = set(str(item) for item in old.get("observed_fields", []))
            state["first_observed"] = old.get("first_observed")
            state["last_observed"] = old.get("last_observed")

    @staticmethod
    def _collect_git_identity() -> Dict[str, Any]:
        repository_root = Path(__file__).resolve().parents[2]
        git_identity = {"commit": None, "dirty": None, "root": str(repository_root)}
        try:
            commit = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(repository_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            git_identity.update({"commit": commit or None, "dirty": bool(status.strip())})
        except (OSError, subprocess.SubprocessError):
            pass
        return git_identity

    def write_evidence(self, filename: str, payload: Mapping[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        name = str(filename).strip()
        if not name or Path(name).name != name or not name.endswith(".json"):
            raise ValueError("monitor evidence filename must be a plain .json file name")
        path = self.output_dir / name
        write_json(path, {**self._base(), **dict(payload)})
        return path

    def append_evidence_jsonl(
        self, filename: str, payload: Mapping[str, Any]
    ) -> Optional[Path]:
        """Append a named high-cost evidence record outside shared step logs."""
        if not self.enabled:
            return None
        name = str(filename).strip()
        if not name or Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError("monitor evidence filename must be a plain .jsonl file name")
        self._append_jsonl(name, {**self._base(), **dict(payload)})
        return self.output_dir / name

    def set_context(
        self,
        *,
        stage: str,
        epoch: Optional[int],
        global_step: Optional[int],
        graph_prob_prior_forward: Optional[int] = None,
    ) -> None:
        self.context = MonitorContext(
            run_id=self.context.run_id,
            session_id=self.context.session_id,
            cell_id=self.context.cell_id,
            seed=self.context.seed,
            stage=str(stage),
            epoch=None if epoch is None else int(epoch),
            global_step=None if global_step is None else int(global_step),
            graph_prob_prior_forward=(
                None if graph_prob_prior_forward is None else int(graph_prob_prior_forward)
            ),
        )

    def update_runtime_source_state(
        self, namespace: str, *, source_active: bool
    ) -> None:
        """Refine a config-only source decision using a constructed runtime object."""
        spec = get_monitor_spec(namespace)
        state = self.monitor_groups[spec.namespace]
        state["source_active"] = bool(source_active)
        state["effective"] = bool(
            self.enabled
            and state["requested"]
            and state["source_active"]
        )
        if self.enabled:
            self._write_runtime_summary("running")

    def _is_recordable(self, namespace: str, cadence: str) -> bool:
        spec = get_monitor_spec(namespace)
        if spec.cadence != cadence:
            raise ValueError(
                f"Monitor namespace {namespace} has cadence={spec.cadence}, expected {cadence}."
            )
        return self.enabled and bool(self.monitor_groups[spec.namespace]["effective"])

    def should_sample_step(self, namespace: str = "train") -> bool:
        spec = get_monitor_spec(namespace)
        if not self._is_recordable(namespace, "step"):
            return False
        if spec.sampling == "global_step":
            return self.context.global_step is not None and self.context.global_step % self.step_every_n == 0
        if spec.sampling == "graph_prob_prior_forward":
            forward = self.context.graph_prob_prior_forward
            return forward is not None and forward > 0 and forward % self.graph_prob_prior_every_n == 0
        raise ValueError(f"Unsupported step monitor sampling policy: {spec.sampling}")

    def _base(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.context.run_id,
            "session_id": self.context.session_id,
            "cell_id": self.context.cell_id,
            "seed": self.context.seed,
            "stage": self.context.stage,
            "epoch": self.context.epoch,
            "global_step": self.context.global_step,
            "graph_prob_prior_forward": self.context.graph_prob_prior_forward,
        }

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")

    def _observe(self, namespace: str, fields: Iterable[str]) -> None:
        field_names = sorted({str(field) for field in fields})
        if not field_names:
            return
        state = self.runtime_groups[namespace]
        position = {
            key: value
            for key, value in self._base().items()
            if key in {"stage", "epoch", "global_step", "graph_prob_prior_forward"} and value is not None
        }
        if not state["observed"]:
            state["first_observed"] = position
        state["observed"] = True
        state["write_count"] += 1
        state["metric_count"] += len(field_names)
        state["observed_fields"].update(field_names)
        state["last_observed"] = position

    def record_step(self, namespace: str, metrics: Mapping[str, Any], *, force: bool = False) -> None:
        if (not self.write_step_jsonl) or (not self._is_recordable(namespace, "step")):
            return
        if (not force) and (not self.should_sample_step(namespace)):
            return
        clean = {name: value for name, value in json_safe(metrics).items() if value is not None}
        if clean:
            self._append_jsonl("metrics_step.jsonl", {**self._base(), "namespace": namespace, "metrics": clean})
            self._observe(namespace, clean)

    def record_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if (not self.write_events_jsonl) or (not self._is_recordable(event, "event")):
            return
        self._append_jsonl(
            "metrics_events.jsonl",
            {**self._base(), "namespace": event, "event": event, "payload": json_safe(payload)},
        )
        self._observe(event, ("event",))

    def record_epoch(
        self,
        split: str,
        namespace: str,
        metrics: Mapping[str, Any],
        *,
        reducer: Any = "last",
        n: int = 1,
    ) -> None:
        if (not self.write_epoch_csv) or (not self._is_recordable(namespace, "epoch")):
            return
        rows = []
        clean = {metric: value for metric, value in json_safe(metrics).items() if value is not None}
        for metric, value in clean.items():
            metric_reducer = reducer.get(metric, "last") if isinstance(reducer, Mapping) else reducer
            rows.append({
                **self._base(),
                "split": str(split),
                "namespace": str(namespace),
                "metric": str(metric),
                "value": value,
                "reducer": str(metric_reducer),
                "n": int(n),
            })
        if not rows:
            return
        path = self.output_dir / "metrics_epoch.csv"
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPOCH_CSV_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        self._observe(namespace, clean)

    def _runtime_summary_payload(self, status: str) -> Dict[str, Any]:
        groups = {}
        for spec in MONITOR_SPECS:
            runtime = self.runtime_groups[spec.namespace]
            groups[spec.namespace] = {
                **self.monitor_groups[spec.namespace],
                "observed": bool(runtime["observed"]),
                "write_count": int(runtime["write_count"]),
                "metric_count": int(runtime["metric_count"]),
                "observed_fields": sorted(runtime["observed_fields"]),
                "first_observed": runtime["first_observed"],
                "last_observed": runtime["last_observed"],
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "cell_id": self.cell_id,
            "seed": self.seed,
            "started_at": self.started_at,
            "resumed": self.resumed,
            "resumed_at": self.resumed_at,
            "finalized_at": None if str(status) == "running" else self._utc_now(),
            "status": str(status),
            "output_policy": self.output_policy,
            "git": self.git_identity,
            "sampling": {
                "global_step_every_n": self.step_every_n,
                "graph_prob_prior_forward_every_n": self.graph_prob_prior_every_n,
            },
            "groups": groups,
        }

    def _write_runtime_summary(self, status: str) -> None:
        write_json(
            self.output_dir / "monitor_runtime_summary.json",
            self._runtime_summary_payload(status),
        )

    def finalize(self, *, status: str) -> None:
        if (not self.enabled) or self.finalized:
            return
        self.record_event("monitor_finalized", {"status": str(status)})
        self._write_runtime_summary(status)
        self.finalized = True
