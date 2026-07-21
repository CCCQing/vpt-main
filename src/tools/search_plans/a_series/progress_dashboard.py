from __future__ import annotations

import math
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple

from src.tools.parameter_search.search_eta import SearchEtaTracker


RunItem = Callable[[Any, str], Dict[str, Any]]
ItemName = Callable[[Any], str]
ResultCallback = Callable[[Sequence[Dict[str, Any]]], None]


def _duration(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value):
        return "unknown"
    total = max(0, int(round(value)))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours:02d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:02d}m{secs:02d}s"
    return f"{secs:02d}s"


def _bar(fraction: float, width: int) -> str:
    bounded = max(0.0, min(1.0, float(fraction)))
    filled = min(width, int(round(bounded * width)))
    return "#" * filled + "-" * (width - filled)


def _trial_fraction(row: Mapping[str, Any]) -> float:
    total_epochs = max(0, int(row.get("total_epochs", 0) or 0))
    if total_epochs <= 0:
        return 0.0
    completed_epochs = max(0, int(row.get("completed_epochs", 0) or 0))
    epoch = max(0, int(row.get("epoch", 0) or 0))
    batch = max(0, int(row.get("batch", 0) or 0))
    total_batches = max(0, int(row.get("total_batches", 0) or 0))
    phase = str(row.get("phase", ""))
    current_epoch = max(completed_epochs, epoch - 1)
    if phase == "epoch_complete":
        current_epoch = max(current_epoch, completed_epochs)
        within_epoch = 0.0
    elif phase == "train" and total_batches > 0:
        within_epoch = min(0.9, 0.9 * batch / total_batches)
    elif epoch > completed_epochs:
        within_epoch = 0.95
    else:
        within_epoch = 0.0
    return max(0.0, min(1.0, (current_epoch + within_epoch) / total_epochs))


def format_dashboard_lines(snapshot: Mapping[str, Any], width: int = 120) -> List[str]:
    width = max(80, int(width))
    completed = max(0, int(snapshot.get("completed_count", 0) or 0))
    total = max(0, int(snapshot.get("total_count", 0) or 0))
    active = list(snapshot.get("active_trials") or [])
    queued = max(0, int(snapshot.get("queued_count", 0) or 0))
    active_fractions = [_trial_fraction(row) for row in active]
    overall = (
        min(1.0, (completed + sum(active_fractions)) / total)
        if total > 0
        else 1.0
    )
    total_line = (
        f"TOTAL {completed:>3}/{total:<3} {100.0 * overall:5.1f}% "
        f"|{_bar(overall, 32)}| active={len(active)} queued={queued} "
        f"ETA={_duration(snapshot.get('eta_seconds'))} confidence={snapshot.get('confidence', 'unknown')}"
    )
    lines = [total_line[:width].ljust(width)]
    for row in active:
        fraction = _trial_fraction(row)
        trial_name = str(row.get("trial_name", ""))[:22]
        gpu = str(row.get("gpu_group", ""))[:7]
        phase = str(row.get("phase", "starting"))[:15]
        epoch = max(0, int(row.get("epoch", 0) or 0))
        total_epochs = max(0, int(row.get("total_epochs", 0) or 0))
        batch = max(0, int(row.get("batch", 0) or 0))
        total_batches = max(0, int(row.get("total_batches", 0) or 0))
        line = (
            f"GPU {gpu:<7} {trial_name:<22} E{epoch:02d}/{total_epochs:02d} "
            f"B{batch:04d}/{total_batches:04d} {phase:<15} "
            f"|{_bar(fraction, 20)}| {100.0 * fraction:5.1f}% "
            f"ETA={_duration(row.get('remaining_seconds'))}"
        )
        lines.append(line[:width].ljust(width))
    return lines


class ASeriesProgressDashboard:
    def __init__(
        self,
        out_root: Path,
        trials: Sequence[Mapping[str, Any]],
        worker_gpus: Sequence[str],
        interval_seconds: float,
        width: int,
        enabled: bool,
        stream: TextIO = sys.stdout,
    ) -> None:
        if not math.isfinite(float(interval_seconds)) or float(interval_seconds) <= 0.0:
            raise ValueError("Progress interval must be a finite positive number.")
        if int(width) < 80:
            raise ValueError("Progress width must be at least 80 columns.")
        self.tracker = SearchEtaTracker(
            out_root=Path(out_root),
            trials=trials,
            worker_gpus=worker_gpus,
            update_interval_seconds=float(interval_seconds),
        )
        self.interval_seconds = float(interval_seconds)
        self.width = int(width)
        self.enabled = bool(enabled)
        self.stream = stream
        self.interactive = bool(getattr(stream, "isatty", lambda: False)())
        self.output_interval_seconds = (
            self.interval_seconds if self.interactive else max(30.0, self.interval_seconds)
        )
        self.last_render_perf = 0.0
        self.rendered_lines = 0
        self.lock = threading.Lock()

    def register_start(
        self,
        worker_index: int,
        trial_index: int,
        trial: Mapping[str, Any],
        gpu: str,
    ) -> None:
        self.tracker.register_start(worker_index, trial_index, trial, gpu, 1)

    def register_finish(self, worker_index: int, result: Dict[str, Any]) -> Dict[str, Any]:
        tracking_result = dict(result)
        if not str(result.get("status", "")).startswith("completed"):
            tracking_result["returncode"] = 1
        self.tracker.register_finish(worker_index, tracking_result)
        return result

    def register_error(self, worker_index: int) -> None:
        self.tracker.register_error(worker_index)

    def _clear_locked(self) -> None:
        if not self.interactive or self.rendered_lines <= 0:
            self.rendered_lines = 0
            return
        count = self.rendered_lines
        self.stream.write(f"\x1b[{count}A")
        for index in range(count):
            self.stream.write("\r\x1b[2K")
            if index < count - 1:
                self.stream.write("\x1b[1B")
        if count > 1:
            self.stream.write(f"\x1b[{count - 1}A")
        self.stream.write("\r")
        self.rendered_lines = 0

    def write_event(self, message: str) -> None:
        with self.lock:
            self._clear_locked()
            self.stream.write(str(message).rstrip() + "\n")
            self.stream.flush()

    def render(
        self,
        pending_trials: Sequence[Mapping[str, Any]],
        completed_count: int,
        total_count: int,
        force: bool = False,
    ) -> Dict[str, Any]:
        snapshot = self.tracker.snapshot(pending_trials, completed_count, total_count)
        if not self.enabled:
            return snapshot
        now = time.perf_counter()
        if not force and now - self.last_render_perf < self.output_interval_seconds:
            return snapshot
        lines = format_dashboard_lines(snapshot, self.width)
        with self.lock:
            self._clear_locked()
            for line in lines:
                self.stream.write(line + "\n")
            self.stream.flush()
            self.rendered_lines = len(lines) if self.interactive else 0
            self.last_render_perf = now
        return snapshot


def run_parallel_trials(
    *,
    all_trials: Sequence[Mapping[str, Any]],
    pending_items: Sequence[Any],
    pending_trials: Sequence[Mapping[str, Any]],
    out_root: Path,
    gpu_groups: Sequence[str],
    max_workers: int,
    initial_completed: int,
    progress_interval: float,
    progress_width: int,
    progress_enabled: bool,
    run_item: RunItem,
    item_name: ItemName,
    on_result: Optional[ResultCallback] = None,
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[Any, Mapping[str, Any]]] = list(zip(pending_items, pending_trials))
    if len(pairs) != len(pending_items) or len(pairs) != len(pending_trials):
        raise ValueError("Pending items and progress trials must have identical lengths.")
    worker_count = min(int(max_workers), len(gpu_groups), len(pairs))
    if worker_count <= 0:
        return []
    worker_gpus = list(gpu_groups[:worker_count])
    dashboard = ASeriesProgressDashboard(
        out_root=out_root,
        trials=all_trials,
        worker_gpus=worker_gpus,
        interval_seconds=progress_interval,
        width=progress_width,
        enabled=progress_enabled,
    )
    results: List[Dict[str, Any]] = []
    next_pair = 0
    completed_count = int(initial_completed)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_worker: Dict[Any, Tuple[int, Any]] = {}

        def submit_next(worker_index: int) -> bool:
            nonlocal next_pair
            if next_pair >= len(pairs):
                return False
            pair_index = next_pair
            next_pair += 1
            item, trial = pairs[pair_index]
            gpu = worker_gpus[worker_index]
            trial_index = int(trial.get("trial_index", pair_index))
            dashboard.register_start(worker_index, trial_index, trial, gpu)
            try:
                future = executor.submit(run_item, item, gpu)
            except BaseException:
                dashboard.register_error(worker_index)
                raise
            future_to_worker[future] = (worker_index, item)
            return True

        for worker_index in range(worker_count):
            submit_next(worker_index)
        dashboard.render(
            [trial for _, trial in pairs[next_pair:]],
            completed_count,
            len(all_trials),
            force=True,
        )

        while future_to_worker:
            done, _ = wait(
                list(future_to_worker),
                timeout=float(progress_interval),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                dashboard.render(
                    [trial for _, trial in pairs[next_pair:]],
                    completed_count,
                    len(all_trials),
                    force=False,
                )
                continue
            for future in done:
                worker_index, item = future_to_worker.pop(future)
                try:
                    result = dashboard.register_finish(worker_index, future.result())
                except BaseException:
                    dashboard.register_error(worker_index)
                    dashboard.write_event(f"[failed] worker={worker_index + 1}")
                    raise
                results.append(result)
                completed_count += 1
                if on_result is not None:
                    on_result(results)
                status = str(result.get("status", "finished"))
                dashboard.write_event(
                    f"[{completed_count}/{len(all_trials)}] {item_name(item)} status={status}"
                )
                submit_next(worker_index)
            dashboard.render(
                [trial for _, trial in pairs[next_pair:]],
                completed_count,
                len(all_trials),
                force=True,
            )
    return results
