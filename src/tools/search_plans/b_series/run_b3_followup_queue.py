#!/usr/bin/env python3
"""Durable multi-GPU queue for B3 checkpoint-only diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
STRICT_TRAINING_SEEDS = (0, 1, 2)
STRICT_SELECTION_SEEDS = (424242, 424243, 424244)
DEFAULT_METHODS = ("B3-R1I-R025", "B3-R1I-R050")


@dataclass(frozen=True)
class Job:
    method: str
    training_seed: int
    scope: str
    selection_seed: Optional[int]
    source_run: Path
    output_dir: Path
    log_path: Path

    @property
    def name(self) -> str:
        suffix = (
            "full"
            if self.scope == "full"
            else "probe{}".format(self.selection_seed)
        )
        return "{}_seed{}_{}".format(
            self.method, self.training_seed, suffix
        )


def _comma_values(raw: str) -> List[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("comma-separated values must be non-empty and unique")
    return values


def _integer_values(raw: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(
        value < 0 for value in values
    ):
        raise ValueError("integer values must be unique and non-negative")
    return values


def _gpu_groups(raw: str) -> List[str]:
    values = [item.strip() for item in str(raw).split(";") if item.strip()]
    if not values or any("," in value for value in values):
        raise ValueError("GPU worker slots must be single cards separated by semicolons")
    return values


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _completed(output_dir: Path, experiments: Sequence[str]) -> bool:
    summary = output_dir / "b3_followup_summary.json"
    if not summary.is_file():
        return False
    try:
        payload = _read_json(summary)
    except (OSError, ValueError):
        return False
    return (
        str(payload.get("status", "")).lower() == "completed"
        and bool(payload.get("valid", False))
        and list(payload.get("experiments") or []) == list(experiments)
    )


def _has_contents(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _build_jobs(
    source_root: Path,
    output_root: Path,
    methods: Sequence[str],
    training_seeds: Sequence[int],
    selection_seeds: Sequence[int],
) -> List[Job]:
    jobs = []
    for method in methods:
        for training_seed in training_seeds:
            source_run = (
                source_root
                / method
                / "seed{}".format(training_seed)
                / RUN_SUFFIX
            )
            required = (
                source_run / "resolved_config.yaml",
                source_run / "model_final_trainable.pth",
            )
            if any(not path.is_file() for path in required):
                raise FileNotFoundError(
                    "source run is incomplete: {}".format(source_run)
                )
            scopes = [("full", None)] + [
                ("probe", int(seed)) for seed in selection_seeds
            ]
            for scope, selection_seed in scopes:
                suffix = (
                    "full"
                    if scope == "full"
                    else "probe/selection_seed_{}".format(selection_seed)
                )
                output_dir = (
                    output_root
                    / method
                    / "seed{}".format(training_seed)
                    / suffix
                )
                jobs.append(
                    Job(
                        method=method,
                        training_seed=int(training_seed),
                        scope=scope,
                        selection_seed=selection_seed,
                        source_run=source_run,
                        output_dir=output_dir,
                        log_path=(
                            output_root / "launcher_logs" / (Job.__name__)
                        ),
                    )
                )
    return [
        Job(
            method=job.method,
            training_seed=job.training_seed,
            scope=job.scope,
            selection_seed=job.selection_seed,
            source_run=job.source_run,
            output_dir=job.output_dir,
            log_path=output_root / "launcher_logs" / (job.name + ".log"),
        )
        for job in jobs
    ]


def _command(job: Job, args) -> List[str]:
    command = [
        str(args.python_bin),
        "-m",
        "src.tools.search_plans.b_series.replay_b3_followup_diagnostics",
        "--source-run",
        str(job.source_run),
        "--output-dir",
        str(job.output_dir),
        "--experiments",
        args.experiments,
        "--scope",
        job.scope,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--selected-layers",
        args.selected_layers,
    ]
    if job.selection_seed is not None:
        command.extend(["--selection-seed", str(job.selection_seed)])
    return command


def _format(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _run_job(job: Job, args, gpu: str) -> Dict[str, object]:
    command = _command(job, args)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
        }
    )
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with job.log_path.open("w", encoding="utf-8") as handle:
        handle.write("CUDA_VISIBLE_DEVICES={}\n{}\n\n".format(gpu, _format(command)))
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    completed = process.returncode == 0 and _completed(
        job.output_dir, args.experiment_names
    )
    return {
        "name": job.name,
        "method": job.method,
        "training_seed": job.training_seed,
        "scope": job.scope,
        "selection_seed": job.selection_seed,
        "gpu": str(gpu),
        "status": "completed" if completed else "failed",
        "returncode": int(process.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "source_run": str(job.source_run),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--training-seeds", default="0,1,2")
    parser.add_argument("--selection-seeds", default="424242,424243,424244")
    parser.add_argument("--gpu-groups", default="0;2;3;4;5;7")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--selected-layers", default="8,9,10,11")
    parser.add_argument("--experiments", default="D2,D3,D4")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    methods = _comma_values(args.methods)
    experiments = [value.upper() for value in _comma_values(args.experiments)]
    if any(value not in {"D2", "D2G", "D3", "D4"} for value in experiments):
        raise SystemExit("--experiments contains an unsupported diagnostic")
    args.experiments = ",".join(experiments)
    args.experiment_names = experiments
    training_seeds = _integer_values(args.training_seeds)
    selection_seeds = _integer_values(args.selection_seeds)
    gpus = _gpu_groups(args.gpu_groups)
    if args.max_workers < 1 or args.max_workers > len(gpus):
        raise SystemExit("--max-workers must be inside the GPU slot count")
    if min(args.batch_size, args.cpu_threads) < 1 or args.num_workers < 0:
        raise SystemExit("batch/thread/worker values are invalid")
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    jobs = _build_jobs(
        source_root,
        output_root,
        methods,
        training_seeds,
        selection_seeds,
    )
    pending = []
    skipped = []
    for job in jobs:
        if _completed(job.output_dir, experiments):
            skipped.append(job)
        elif _has_contents(job.output_dir):
            raise RuntimeError(
                "refusing to overwrite incomplete output {}".format(job.output_dir)
            )
        else:
            pending.append(job)
    if args.dry_run:
        for index, job in enumerate(pending):
            print(
                "[dry-run] gpu={} {}".format(
                    gpus[index % args.max_workers], _format(_command(job, args))
                )
            )
        return
    trials = [
        {
            "trial_index": index,
            "trial_name": job.name,
            "runner": "b3_followup",
            "stage": "_".join(experiments),
            "combo": {
                "method": job.method,
                "training_seed": job.training_seed,
                "scope": job.scope,
                "selection_seed": job.selection_seed,
            },
            "overrides": {},
            "output_dir": str(job.output_dir),
            "identity_fields": {"source_run": str(job.source_run)},
            "eta_fields": {"scope": job.scope},
            "eta_compatibility_keys": ["runner", "stage", "scope"],
        }
        for index, job in enumerate(jobs)
    ]
    trial_by_job = {job: trial for job, trial in zip(jobs, trials)}
    results = run_parallel_trials(
        all_trials=trials,
        pending_items=pending,
        pending_trials=[trial_by_job[job] for job in pending],
        out_root=output_root,
        gpu_groups=gpus,
        max_workers=int(args.max_workers),
        initial_completed=len(skipped),
        progress_interval=5.0,
        progress_width=120,
        progress_enabled=not args.no_progress,
        run_item=lambda job, gpu: _run_job(job, args, gpu),
        item_name=lambda job: job.name,
    )
    status = (
        "completed"
        if all(result.get("status") == "completed" for result in results)
        else "failed"
    )
    summary = {
        "format": "b3_followup_queue_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "methods": methods,
        "training_seeds": training_seeds,
        "selection_seeds": selection_seeds,
        "experiments": experiments,
        "gpu_groups": gpus,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "cpu_threads": args.cpu_threads,
        "results": results,
        "skipped_existing": [job.name for job in skipped],
        "status": status,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "b3_followup_queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if status != "completed":
        raise RuntimeError("one or more B3 follow-up jobs failed")


if __name__ == "__main__":
    main()
