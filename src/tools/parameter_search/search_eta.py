from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_time(value: Any) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    last_error = None
    for attempt in range(3):
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(str(temp_path), str(path))
            return
        except OSError as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.02 * (attempt + 1))
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass
    if last_error is not None:
        raise last_error


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _gpu_count(gpu_group: str) -> int:
    values = [item.strip() for item in str(gpu_group).split(",") if item.strip()]
    return max(1, len(values))


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "unknown"
    total = max(0, int(round(float(seconds))))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class SearchEtaTracker:
    ACTIVE_PROGRESS_WEIGHT = 0.7
    HISTORICAL_WEIGHT = 0.3

    def __init__(
        self,
        out_root: Path,
        trials: Sequence[Mapping[str, Any]],
        worker_gpus: Sequence[str],
        update_interval_seconds: float = 30.0,
    ) -> None:
        self.out_root = Path(out_root)
        self.trials = list(trials)
        self.worker_gpus = list(worker_gpus)
        self.update_interval_seconds = max(1.0, float(update_interval_seconds))
        self.started_perf = time.perf_counter()
        self.session_started_at = _utc_now()
        self.search_started_at = self.session_started_at
        self.last_emit_perf = 0.0
        self.lock = threading.Lock()
        self.write_warning_emitted = False
        self.active: Dict[int, Dict[str, Any]] = {}
        self.samples: List[Dict[str, Any]] = []
        self.completed_trials: Dict[str, Dict[str, Any]] = {}
        self.state_path = self.out_root / "search_state.json"
        self.search_fingerprint = self._search_fingerprint()
        restored_keys = self._load_search_state()
        self._load_history(restored_keys)

    def _signature(self, trial: Mapping[str, Any], nproc: int) -> Dict[str, Any]:
        overrides = trial.get("overrides") or {}
        signature = {
            "runner": str(trial.get("runner", "train")),
            "stage": str(trial.get("stage", "")),
            "nproc": int(nproc),
            "total_epochs": int(overrides.get("SOLVER.TOTAL_EPOCH", 0) or 0),
            "batch_size": int(overrides.get("DATA.BATCH_SIZE", 0) or 0),
        }
        signature.update(dict(trial.get("eta_fields") or {}))
        return signature

    @staticmethod
    def _signature_key(signature: Mapping[str, Any]) -> str:
        return json.dumps(dict(signature), sort_keys=True, ensure_ascii=True)

    @staticmethod
    def _trial_identity(trial: Mapping[str, Any]) -> Dict[str, Any]:
        identity = {
            "trial_index": int(trial.get("trial_index", 0)),
            "trial_name": str(trial.get("trial_name", "")),
            "runner": str(trial.get("runner", "train")),
            "stage": str(trial.get("stage", "")),
            "combo": dict(trial.get("combo") or {}),
            "overrides": dict(trial.get("overrides") or {}),
            "output_dir": str(Path(str(trial.get("output_dir", ""))).resolve()),
        }
        identity.update(dict(trial.get("identity_fields") or {}))
        return identity

    def _trial_key(self, trial: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            self._trial_identity(trial),
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def _search_fingerprint(self) -> str:
        identities = [
            self._trial_identity(trial)
            for trial in sorted(self.trials, key=lambda item: int(item.get("trial_index", 0)))
        ]
        encoded = json.dumps(
            identities,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_search_state(self) -> set[str]:
        state = _read_json(self.state_path)
        if not state or str(state.get("search_fingerprint", "")) != self.search_fingerprint:
            return set()
        self.search_started_at = str(state.get("search_started_at") or self.session_started_at)
        trial_by_key = {self._trial_key(trial): trial for trial in self.trials}
        restored_keys = set()
        for item in state.get("completed_trials", []):
            if not isinstance(item, dict):
                continue
            trial_key = str(item.get("trial_key", ""))
            trial = trial_by_key.get(trial_key)
            if trial is None:
                continue
            record = dict(item)
            self.completed_trials[trial_key] = record
            if int(record.get("returncode", 1)) != 0:
                continue
            duration = _positive_float(record.get("duration_seconds"))
            if duration is None:
                continue
            self._add_sample(
                trial,
                max(1, int(record.get("nproc", 1))),
                duration,
                str(record.get("gpu_group", "")),
                "search_state",
            )
            restored_keys.add(trial_key)
        return restored_keys

    def _add_sample(
        self,
        trial: Mapping[str, Any],
        nproc: int,
        duration_seconds: float,
        gpu_group: str,
        source: str,
    ) -> None:
        duration = _positive_float(duration_seconds)
        if duration is None:
            return
        signature = self._signature(trial, nproc)
        self.samples.append(
            {
                "signature": signature,
                "signature_key": self._signature_key(signature),
                "gpu_group": str(gpu_group),
                "duration_seconds": float(duration),
                "source": str(source),
            }
        )

    def _history_duration(self, trial: Mapping[str, Any]) -> Optional[Tuple[float, int, str, str]]:
        output_dir = Path(str(trial["output_dir"]))
        runtime_path = output_dir / "trial_runtime.json"
        runtime = _read_json(runtime_path)
        if runtime and int(runtime.get("returncode", 1)) == 0:
            duration = _positive_float(runtime.get("duration_seconds"))
            if duration is not None:
                return (
                    duration,
                    max(1, int(runtime.get("nproc", 1))),
                    str(runtime.get("gpu_group", "")),
                    str(runtime_path),
                )
        progress_paths: List[Path] = []
        if output_dir.is_dir():
            try:
                progress_paths = sorted(
                    output_dir.rglob("progress.json"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                progress_paths = []
        for path in progress_paths:
            progress = _read_json(path)
            if not progress or str(progress.get("status", "")) != "completed":
                continue
            duration = _positive_float(progress.get("elapsed_seconds"))
            if duration is not None:
                return (
                    duration,
                    max(1, int(progress.get("world_size", 1))),
                    "",
                    str(path),
                )
        summaries: List[Path] = []
        if output_dir.is_dir():
            try:
                summaries = sorted(output_dir.rglob("monitor_runtime_summary.json"))
            except OSError:
                summaries = []
        durations = []
        for path in summaries:
            summary = _read_json(path)
            if not summary or str(summary.get("status", "")) != "completed":
                continue
            started = _parse_time(summary.get("started_at"))
            finished = _parse_time(summary.get("finalized_at"))
            if started is not None and finished is not None:
                duration = (finished - started).total_seconds()
                if duration > 0.0:
                    durations.append(duration)
        if durations:
            return float(sum(durations)), 1, "", "monitor_runtime_summary.json"
        return None

    def _completed_record(
        self,
        trial: Mapping[str, Any],
        nproc: int,
        gpu_group: str,
        duration_seconds: Optional[float],
        started_at: str = "",
        finished_at: str = "",
        skipped_existing: bool = False,
        source: str = "current_session",
    ) -> Dict[str, Any]:
        return {
            "trial_key": self._trial_key(trial),
            "trial_name": str(trial.get("trial_name", "")),
            "trial_index": int(trial.get("trial_index", 0)),
            "output_dir": str(trial.get("output_dir", "")),
            "gpu_group": str(gpu_group),
            "nproc": int(nproc),
            "started_at": str(started_at),
            "finished_at": str(finished_at),
            "duration_seconds": duration_seconds,
            "returncode": 0,
            "status": "completed_existing" if skipped_existing else "completed",
            "skipped_existing": bool(skipped_existing),
            "signature": self._signature(trial, int(nproc)),
            "source": str(source),
        }

    def _load_history(self, restored_keys: set[str]) -> None:
        for trial in self.trials:
            trial_key = self._trial_key(trial)
            if trial_key in restored_keys:
                continue
            history = self._history_duration(trial)
            if history is None:
                continue
            duration, nproc, gpu_group, source = history
            self._add_sample(trial, nproc, duration, gpu_group, source)
            self.completed_trials[trial_key] = self._completed_record(
                trial,
                nproc,
                gpu_group,
                duration,
                skipped_existing=True,
                source=source,
            )

    def register_start(
        self,
        worker_index: int,
        trial_index: int,
        trial: Mapping[str, Any],
        gpu_group: str,
        nproc: int,
    ) -> None:
        with self.lock:
            self.active[int(worker_index)] = {
                "worker_index": int(worker_index),
                "trial_index": int(trial_index),
                "trial": trial,
                "gpu_group": str(gpu_group),
                "nproc": int(nproc),
                "started_at": _utc_now(),
                "started_perf": time.perf_counter(),
            }

    def register_finish(self, worker_index: int, result: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            record = self.active.pop(int(worker_index), None)
        if record is None:
            return result
        finished_perf = time.perf_counter()
        duration = max(0.0, finished_perf - float(record["started_perf"]))
        result["started_at"] = str(record["started_at"])
        result["finished_at"] = _utc_now()
        result["duration_seconds"] = float(duration)
        result["eta_signature"] = self._signature_key(
            self._signature(record["trial"], int(record["nproc"]))
        )
        trial = record["trial"]
        trial_key = self._trial_key(trial)
        if bool(result.get("skipped_existing", False)):
            with self.lock:
                self.completed_trials.setdefault(
                    trial_key,
                    self._completed_record(
                        trial,
                        int(record["nproc"]),
                        str(record["gpu_group"]),
                        None,
                        started_at=str(record["started_at"]),
                        finished_at=str(result["finished_at"]),
                        skipped_existing=True,
                        source="completion_check",
                    ),
                )
            return result
        if int(result.get("returncode", -1)) == 0:
            with self.lock:
                self._add_sample(
                    trial,
                    int(record["nproc"]),
                    duration,
                    str(record["gpu_group"]),
                    "current_session",
                )
                self.completed_trials[trial_key] = self._completed_record(
                    trial,
                    int(record["nproc"]),
                    str(record["gpu_group"]),
                    float(duration),
                    started_at=str(record["started_at"]),
                    finished_at=str(result["finished_at"]),
                )
        return result

    def register_error(self, worker_index: int) -> None:
        with self.lock:
            self.active.pop(int(worker_index), None)

    def _warn_write(self, path: Path, error: OSError) -> None:
        with self.lock:
            if self.write_warning_emitted:
                return
            self.write_warning_emitted = True
        print(f"[eta] unable to update {path}: {error}", flush=True)

    def _duration_prediction(
        self,
        trial: Mapping[str, Any],
        gpu_group: str,
        nproc: int,
        fallback: Optional[float] = None,
    ) -> Dict[str, Any]:
        signature = self._signature(trial, nproc)
        key = self._signature_key(signature)
        exact_gpu = [
            float(sample["duration_seconds"])
            for sample in self.samples
            if sample["signature_key"] == key and sample["gpu_group"] == str(gpu_group)
        ]
        exact = [
            float(sample["duration_seconds"])
            for sample in self.samples
            if sample["signature_key"] == key
        ]
        compatibility_keys = [
            str(item)
            for item in trial.get("eta_compatibility_keys", ["runner"])
        ]
        compatible = [
            float(sample["duration_seconds"])
            for sample in self.samples
            if int(sample["signature"].get("nproc", 1)) == int(nproc)
            and all(
                sample["signature"].get(field) == signature.get(field)
                for field in compatibility_keys
            )
        ]
        global_values = [float(sample["duration_seconds"]) for sample in self.samples]
        values = exact_gpu if len(exact_gpu) >= 2 else exact or compatible or global_values
        if not values and fallback is not None and fallback > 0.0:
            values = [float(fallback)]
        if not values:
            return {"estimate": None, "low": None, "high": None, "sample_count": 0}
        estimate = float(statistics.median(values))
        if len(values) == 1:
            low = estimate * 0.75
            high = estimate * 1.35
        else:
            low = min(estimate, _quantile(values, 0.25))
            high = max(estimate, _quantile(values, 0.75))
            if high <= low:
                low = estimate * 0.9
                high = estimate * 1.1
        return {
            "estimate": estimate,
            "low": max(0.0, low),
            "high": max(estimate, high),
            "sample_count": len(values),
        }

    @staticmethod
    def _latest_progress(trial: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        output_dir = Path(str(trial["output_dir"]))
        if not output_dir.is_dir():
            return None
        direct_path = output_dir / "progress.json"
        direct = _read_json(direct_path)
        if direct:
            direct["path"] = str(direct_path)
            return direct
        try:
            paths = sorted(output_dir.rglob("progress.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            return None
        for path in paths:
            payload = _read_json(path)
            if payload:
                payload["path"] = str(path)
                return payload
        return None

    def _active_prediction(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        trial = record["trial"]
        elapsed = max(0.0, time.perf_counter() - float(record["started_perf"]))
        historical = self._duration_prediction(
            trial,
            str(record["gpu_group"]),
            int(record["nproc"]),
        )
        progress = self._latest_progress(trial)
        if progress:
            progress_updated = _parse_time(progress.get("updated_at"))
            trial_started = _parse_time(record.get("started_at"))
            if progress_updated is not None and trial_started is not None and progress_updated < trial_started:
                progress = None
        progress_remaining = None
        progress_low = None
        progress_high = None
        epoch_sample_count = 0
        progress_completed = False
        if progress:
            total_epochs = max(0, int(progress.get("total_epochs", 0) or 0))
            epoch = max(0, int(progress.get("epoch", 0) or 0))
            completed_epochs = max(0, int(progress.get("completed_epochs", 0) or 0))
            epoch_durations = [
                value
                for value in (_positive_float(item) for item in progress.get("epoch_durations_seconds", []))
                if value is not None
            ]
            train_durations = [
                value
                for value in (_positive_float(item) for item in progress.get("train_durations_seconds", []))
                if value is not None
            ]
            epoch_sample_count = len(epoch_durations)
            epoch_estimate = float(statistics.median(epoch_durations)) if epoch_durations else None
            if epoch_estimate is None and historical.get("estimate") and total_epochs > 0:
                epoch_estimate = float(historical["estimate"]) / float(total_epochs)
            if epoch_estimate is None:
                phase_batch = _positive_float(progress.get("phase_batch_time_ewma_seconds"))
                total_batches = max(0, int(progress.get("total_batches", 0) or 0))
                if phase_batch is not None and total_batches > 0:
                    epoch_estimate = phase_batch * total_batches
            if epoch_estimate is not None and total_epochs > 0:
                phase = str(progress.get("phase", ""))
                epoch_elapsed = max(0.0, float(progress.get("epoch_elapsed_seconds", 0.0) or 0.0))
                total_batches = max(0, int(progress.get("total_batches", 0) or 0))
                batch = max(0, int(progress.get("batch", 0) or 0))
                phase_batch = _positive_float(progress.get("phase_batch_time_ewma_seconds"))
                eval_overhead = 0.0
                if epoch_durations and train_durations:
                    eval_values = [
                        max(0.0, full - train)
                        for full, train in zip(epoch_durations[-len(train_durations):], train_durations)
                    ]
                    if eval_values:
                        eval_overhead = float(statistics.median(eval_values))
                if phase == "train":
                    batch_remaining = (
                        max(0, total_batches - batch) * phase_batch
                        if phase_batch is not None
                        else max(0.0, epoch_estimate - epoch_elapsed)
                    )
                    current_remaining = max(0.0, batch_remaining + eval_overhead)
                elif phase == "epoch_complete":
                    current_remaining = 0.0
                else:
                    phase_remaining = (
                        max(0, total_batches - batch) * phase_batch
                        if phase_batch is not None
                        else 0.0
                    )
                    current_remaining = max(phase_remaining, epoch_estimate - epoch_elapsed, 0.0)
                current_epoch = max(epoch, completed_epochs)
                future_epochs = max(0, total_epochs - current_epoch)
                progress_remaining = current_remaining + future_epochs * epoch_estimate
                progress_low = progress_remaining * (0.85 if epoch_sample_count >= 2 else 0.7)
                progress_high = progress_remaining * (1.2 if epoch_sample_count >= 2 else 1.5)
            if str(progress.get("status", "")) == "completed":
                progress_remaining = 0.0
                progress_low = 0.0
                progress_high = 0.0
                progress_completed = True
        history_remaining = (
            max(0.0, float(historical["estimate"]) - elapsed)
            if historical.get("estimate") is not None
            else None
        )
        if progress_completed:
            estimate = low = high = 0.0
        elif progress_remaining is not None and history_remaining is not None:
            estimate = (
                self.ACTIVE_PROGRESS_WEIGHT * progress_remaining
                + self.HISTORICAL_WEIGHT * history_remaining
            )
            low = (
                self.ACTIVE_PROGRESS_WEIGHT * float(progress_low)
                + self.HISTORICAL_WEIGHT * max(0.0, float(historical["low"]) - elapsed)
            )
            high = (
                self.ACTIVE_PROGRESS_WEIGHT * float(progress_high)
                + self.HISTORICAL_WEIGHT * max(0.0, float(historical["high"]) - elapsed)
            )
        elif progress_remaining is not None:
            estimate, low, high = progress_remaining, float(progress_low), float(progress_high)
        elif history_remaining is not None:
            estimate = history_remaining
            low = max(0.0, float(historical["low"]) - elapsed)
            high = max(estimate, float(historical["high"]) - elapsed)
        else:
            estimate = low = high = None
        return {
            "estimate": estimate,
            "low": low,
            "high": high,
            "elapsed_seconds": elapsed,
            "history_sample_count": int(historical.get("sample_count", 0)),
            "epoch_sample_count": int(epoch_sample_count),
            "progress": progress or {},
        }

    def _simulate(
        self,
        active_predictions: Mapping[int, Mapping[str, Any]],
        pending_trials: Sequence[Mapping[str, Any]],
        field: str,
        fallback_duration: Optional[float],
    ) -> Optional[float]:
        if not self.worker_gpus:
            return None
        available = []
        for worker_index, gpu_group in enumerate(self.worker_gpus):
            active = active_predictions.get(worker_index)
            value = active.get(field) if active else 0.0
            if value is None:
                return None
            available.append(float(value))
        for trial in pending_trials:
            worker_index = min(range(len(available)), key=lambda index: available[index])
            gpu_group = self.worker_gpus[worker_index]
            prediction = self._duration_prediction(
                trial,
                gpu_group,
                _gpu_count(gpu_group),
                fallback=fallback_duration,
            )
            value = prediction.get(field)
            if value is None:
                return None
            available[worker_index] += float(value)
        return max(available) if available else 0.0

    def snapshot(
        self,
        pending_trials: Sequence[Mapping[str, Any]],
        completed_count: int,
        total_count: int,
    ) -> Dict[str, Any]:
        with self.lock:
            active_records = {index: dict(record) for index, record in self.active.items()}
            history_count = len(self.samples)
            completed_trials = sorted(
                (dict(record) for record in self.completed_trials.values()),
                key=lambda record: int(record.get("trial_index", 0)),
            )
        active_predictions = {
            worker_index: self._active_prediction(record)
            for worker_index, record in active_records.items()
        }
        active_total_predictions = [
            float(prediction["estimate"]) + float(prediction["elapsed_seconds"])
            for prediction in active_predictions.values()
            if prediction.get("estimate") is not None
        ]
        fallback = float(statistics.median(active_total_predictions)) if active_total_predictions else None
        eta_seconds = self._simulate(active_predictions, pending_trials, "estimate", fallback)
        eta_low = self._simulate(active_predictions, pending_trials, "low", fallback * 0.75 if fallback else None)
        eta_high = self._simulate(active_predictions, pending_trials, "high", fallback * 1.35 if fallback else None)
        if eta_seconds is not None and eta_low is not None:
            eta_low = min(float(eta_low), float(eta_seconds))
        if eta_seconds is not None and eta_high is not None:
            eta_high = max(float(eta_high), float(eta_seconds))
        epoch_samples = sum(int(item.get("epoch_sample_count", 0)) for item in active_predictions.values())
        if eta_seconds is None:
            confidence = "unavailable"
        elif history_count >= 5 and epoch_samples >= 2:
            confidence = "high"
        elif history_count >= 2 or epoch_samples >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        finish_at = None
        if eta_seconds is not None:
            finish_at = (
                datetime.datetime.now().astimezone() + datetime.timedelta(seconds=float(eta_seconds))
            ).isoformat()
        active_rows = []
        for worker_index, record in sorted(active_records.items()):
            prediction = active_predictions[worker_index]
            progress = prediction.get("progress") or {}
            active_rows.append(
                {
                    "worker_index": int(worker_index),
                    "gpu_group": str(record["gpu_group"]),
                    "nproc": int(record["nproc"]),
                    "trial_index": int(record["trial_index"]),
                    "trial_name": str(record["trial"].get("trial_name", "")),
                    "phase": str(progress.get("phase", "starting")),
                    "epoch": int(progress.get("epoch", 0) or 0),
                    "completed_epochs": int(progress.get("completed_epochs", 0) or 0),
                    "total_epochs": int(progress.get("total_epochs", 0) or 0),
                    "batch": int(progress.get("batch", 0) or 0),
                    "total_batches": int(progress.get("total_batches", 0) or 0),
                    "elapsed_seconds": float(prediction["elapsed_seconds"]),
                    "remaining_seconds": prediction.get("estimate"),
                    "remaining_low_seconds": prediction.get("low"),
                    "remaining_high_seconds": prediction.get("high"),
                    "history_sample_count": int(prediction.get("history_sample_count", 0)),
                    "epoch_sample_count": int(prediction.get("epoch_sample_count", 0)),
                    "progress_path": str(progress.get("path", "")),
                }
            )
        return {
            "schema_version": 2,
            "search_fingerprint": self.search_fingerprint,
            "search_started_at": self.search_started_at,
            "session_started_at": self.session_started_at,
            "updated_at": _utc_now(),
            "session_elapsed_seconds": float(time.perf_counter() - self.started_perf),
            "completed_count": int(completed_count),
            "recorded_completed_count": len(completed_trials),
            "active_count": len(active_rows),
            "queued_count": len(pending_trials),
            "total_count": int(total_count),
            "history_sample_count": int(history_count),
            "eta_seconds": eta_seconds,
            "eta_low_seconds": eta_low,
            "eta_high_seconds": eta_high,
            "expected_finish_at": finish_at,
            "confidence": confidence,
            "estimator": {
                "active_progress_weight": self.ACTIVE_PROGRESS_WEIGHT,
                "historical_weight": self.HISTORICAL_WEIGHT,
                "history_statistic": "median_with_interquartile_range",
                "queue_model": "earliest_available_worker_simulation",
            },
            "active_trials": active_rows,
            "completed_trials": completed_trials,
        }

    def emit(
        self,
        pending_trials: Sequence[Mapping[str, Any]],
        completed_count: int,
        total_count: int,
        force: bool = False,
        print_output: bool = True,
    ) -> Dict[str, Any]:
        now_perf = time.perf_counter()
        if not force and now_perf - self.last_emit_perf < self.update_interval_seconds:
            return {}
        payload = self.snapshot(pending_trials, completed_count, total_count)
        payload["state_persisted"] = True
        try:
            _atomic_json(self.state_path, payload)
        except OSError as error:
            payload["state_persisted"] = False
            self._warn_write(self.state_path, error)
        if print_output:
            eta = _format_duration(payload.get("eta_seconds"))
            low = _format_duration(payload.get("eta_low_seconds"))
            high = _format_duration(payload.get("eta_high_seconds"))
            print(
                "[eta] done={}/{} active={} queued={} elapsed={} eta={} range={}..{} confidence={} finish={}".format(
                    payload["completed_count"],
                    payload["total_count"],
                    payload["active_count"],
                    payload["queued_count"],
                    _format_duration(payload["session_elapsed_seconds"]),
                    eta,
                    low,
                    high,
                    payload["confidence"],
                    payload.get("expected_finish_at") or "unknown",
                ),
                flush=True,
            )
        self.last_emit_perf = now_perf
        return payload

    def cleanup_progress(self, result: Mapping[str, Any]) -> bool:
        path = Path(str(result.get("output_dir", ""))) / "progress.json"
        try:
            if path.exists():
                path.unlink()
            return True
        except OSError as error:
            self._warn_write(path, error)
            return False
