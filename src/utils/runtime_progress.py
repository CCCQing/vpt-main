from __future__ import annotations

import datetime
import json
import os
import sys
import time
from typing import Any

from tqdm import tqdm

from . import distributed as du


class TrainingProgressController:
    def __init__(self, cfg: Any, logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger
        self.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.started_perf = time.perf_counter()
        self.last_write_perf = 0.0
        self.write_warned = False
        self.epoch_index = None
        self.epoch_started_perf = None
        self.current_train_duration = None
        self.epoch_durations = []
        self.train_durations = []
        self.phase = "initializing"
        self.phase_batch_ewma = None
        self.state = {
            "schema_version": 1,
            "status": "running",
            "phase": "initializing",
            "epoch": 0,
            "completed_epochs": 0,
            "total_epochs": int(cfg.SOLVER.TOTAL_EPOCH),
            "batch": 0,
            "total_batches": 0,
            "epoch_elapsed_seconds": 0.0,
            "phase_batch_time_ewma_seconds": None,
            "epoch_durations_seconds": [],
            "train_durations_seconds": [],
        }

    def build_train_progress(self, epoch: int, total_epochs: int, total_batches: int, train_loader: Any):
        progress_cfg = self.cfg.SOLVER.PROGRESS
        if not bool(progress_cfg.ENABLE) or du.get_rank() != 0:
            return None
        if not bool(progress_cfg.FORCE) and not sys.stdout.isatty():
            return None
        world_size = max(1, int(du.get_world_size()))
        device_label = "1GPU" if world_size == 1 else f"{world_size}GPU-DDP"
        description = f"Epoch {epoch + 1}/{total_epochs} {device_label}"
        return tqdm(
            train_loader,
            total=int(total_batches),
            desc=description,
            ascii=True,
            ncols=int(progress_cfg.NCOLS),
            dynamic_ncols=False,
            leave=bool(progress_cfg.LEAVE),
            mininterval=float(progress_cfg.MININTERVAL),
            maxinterval=float(progress_cfg.MAXINTERVAL),
            smoothing=0.1,
            file=sys.stdout,
            position=0,
            unit="batch",
            bar_format=(
                "{desc:<24} {percentage:3.0f}%|{bar:30}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

    def write_state(self, force: bool = False, **updates: Any) -> None:
        progress_cfg = self.cfg.SOLVER.PROGRESS
        if not bool(progress_cfg.WRITE_STATE) or du.get_rank() != 0:
            return
        now_perf = time.perf_counter()
        if not force and now_perf - self.last_write_perf < float(progress_cfg.STATE_MININTERVAL):
            return
        self.state.update(updates)
        self.state.update(
            {
                "started_at": self.started_at,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "elapsed_seconds": float(now_perf - self.started_perf),
                "world_size": int(du.get_world_size()),
                "rank": int(du.get_rank()),
                "pid": int(os.getpid()),
                "seed": int(self.cfg.SEED) if self.cfg.SEED is not None else None,
                "output_dir": str(self.cfg.OUTPUT_DIR),
            }
        )
        configured_path = os.environ.get("VPT_SEARCH_PROGRESS_PATH", "").strip()
        if configured_path:
            path = os.path.abspath(configured_path)
            output_dir = os.path.dirname(path)
        else:
            output_dir = str(self.cfg.OUTPUT_DIR)
            path = os.path.join(output_dir, "progress.json")
        os.makedirs(output_dir, exist_ok=True)
        temp_path = f"{path}.{os.getpid()}.tmp"
        last_error = None
        for attempt in range(3):
            try:
                with open(temp_path, "w", encoding="utf-8") as handle:
                    json.dump(self.state, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                os.replace(temp_path, path)
                last_error = None
                break
            except OSError as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.02 * (attempt + 1))
        if last_error is not None:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            if not self.write_warned:
                self.logger.warning("Unable to update progress state %s: %s", path, last_error)
                self.write_warned = True
            return
        self.last_write_perf = now_perf

    def begin_epoch(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        if self.epoch_index != int(epoch):
            self.epoch_index = int(epoch)
            self.epoch_started_perf = time.perf_counter()
            self.current_train_duration = None
        self.phase = "train"
        self.phase_batch_ewma = None
        self.write_state(
            force=True,
            status="running",
            phase="train",
            epoch=int(epoch + 1),
            completed_epochs=int(epoch),
            total_epochs=int(total_epochs),
            batch=0,
            total_batches=int(total_batches),
            epoch_elapsed_seconds=0.0,
            phase_batch_time_ewma_seconds=None,
        )

    def record_batch(
        self,
        phase: str,
        epoch: int,
        total_epochs: int,
        batch: int,
        total_batches: int,
        batch_time_seconds: float,
    ) -> None:
        phase = str(phase)
        if phase != self.phase:
            self.phase = phase
            self.phase_batch_ewma = None
        value = max(0.0, float(batch_time_seconds))
        if self.phase_batch_ewma is None:
            self.phase_batch_ewma = value
        else:
            self.phase_batch_ewma = 0.2 * value + 0.8 * self.phase_batch_ewma
        epoch_elapsed = (
            time.perf_counter() - self.epoch_started_perf
            if self.epoch_started_perf is not None
            else 0.0
        )
        every_n = max(1, int(self.cfg.SOLVER.PROGRESS.STATE_EVERY_N))
        force = int(batch) >= int(total_batches) or int(batch) % every_n == 0
        self.write_state(
            force=force,
            status="running",
            phase=phase,
            epoch=max(0, int(epoch + 1)),
            completed_epochs=max(0, int(epoch)),
            total_epochs=int(total_epochs),
            batch=int(batch),
            total_batches=int(total_batches),
            epoch_elapsed_seconds=float(epoch_elapsed),
            phase_batch_time_ewma_seconds=float(self.phase_batch_ewma),
        )

    def finish_train_phase(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        if self.epoch_started_perf is not None:
            self.current_train_duration = time.perf_counter() - self.epoch_started_perf
        self.write_state(
            force=True,
            status="running",
            phase="train_complete",
            epoch=int(epoch + 1),
            completed_epochs=int(epoch),
            total_epochs=int(total_epochs),
            batch=int(total_batches),
            total_batches=int(total_batches),
            epoch_elapsed_seconds=float(self.current_train_duration or 0.0),
            current_train_duration_seconds=float(self.current_train_duration or 0.0),
        )

    def finish_epoch(self, epoch: int, total_epochs: int) -> None:
        if self.epoch_index != int(epoch) or self.epoch_started_perf is None:
            return
        duration = time.perf_counter() - self.epoch_started_perf
        self.epoch_durations.append(float(duration))
        if self.current_train_duration is not None:
            self.train_durations.append(float(self.current_train_duration))
        self.epoch_durations = self.epoch_durations[-10:]
        self.train_durations = self.train_durations[-10:]
        self.write_state(
            force=True,
            status="running",
            phase="epoch_complete",
            epoch=int(epoch + 1),
            completed_epochs=int(epoch + 1),
            total_epochs=int(total_epochs),
            batch=0,
            total_batches=0,
            epoch_elapsed_seconds=float(duration),
            epoch_durations_seconds=list(self.epoch_durations),
            train_durations_seconds=list(self.train_durations),
        )
        self.epoch_index = None
        self.epoch_started_perf = None

    def current_epoch_elapsed(self) -> float:
        if self.epoch_started_perf is None:
            return 0.0
        return float(time.perf_counter() - self.epoch_started_perf)

    def finalize(self, status: str) -> None:
        self.write_state(
            force=True,
            status=str(status),
            phase="completed" if str(status) == "completed" else "interrupted",
            finalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            epoch_durations_seconds=list(self.epoch_durations),
            train_durations_seconds=list(self.train_durations),
        )
