#!/usr/bin/env python3
"""Durable multi-GPU queue for B-series P0-2/P0-3/P0-4 replay."""

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
from src.tools.search_plans.common import (
    directory_has_contents as _has_contents,
    read_json as _read_json,
)


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
METHODS = ("B3-R1I-R025", "B3-R1I-R050")
TRAINING_SEEDS = (0, 1, 2)
SELECTION_SEEDS = (424242, 424243, 424244)


@dataclass(frozen=True)
class Job:
    experiment: str
    method: str
    training_seed: int
    scope: str
    selection_seed: Optional[int]
    source_run: Path
    a2_run: Path
    output_dir: Path
    log_path: Path

    @property
    def name(self) -> str:
        selection = "full" if self.scope == "full" else "probe{}".format(self.selection_seed)
        return "{}_{}_seed{}_{}".format(
            self.experiment.replace("-", ""), self.method, self.training_seed, selection
        )


def _integers(raw: str) -> List[int]:
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError("integer list must be unique and non-negative")
    return values


def _gpus(raw: str) -> List[str]:
    values = [value.strip() for value in str(raw).split(";") if value.strip()]
    if not values or any("," in value for value in values):
        raise ValueError("GPU workers must be single cards separated by semicolons")
    return values


def _completed(job: Job) -> bool:
    path = job.output_dir / "p0_summary.json"
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, ValueError):
        return False
    return bool(
        payload.get("status") == "completed"
        and payload.get("valid") is True
        and payload.get("experiment") == job.experiment
    )


def _source_run(root: Path, method: str, seed: int) -> Path:
    return root / method / "seed{}".format(seed) / RUN_SUFFIX


def _a2_run(root: Path, seed: int) -> Path:
    return root / "A2" / "seed{}".format(seed) / RUN_SUFFIX


def _build_jobs(b3_root: Path, a2_root: Path, output_root: Path) -> List[Job]:
    jobs = []
    scopes = [("full", None)] + [("probe", seed) for seed in SELECTION_SEEDS]
    for method in METHODS:
        for seed in TRAINING_SEEDS:
            b3_run = _source_run(b3_root, method, seed)
            paired_a2 = _a2_run(a2_root, seed)
            for required in (
                b3_run / "resolved_config.yaml",
                b3_run / "model_final_trainable.pth",
                paired_a2 / "resolved_config.yaml",
                paired_a2 / "model_final_trainable.pth",
            ):
                if not required.is_file():
                    raise FileNotFoundError(str(required))
            for scope, selection in scopes:
                suffix = "full" if scope == "full" else "probe/selection_seed_{}".format(selection)
                out = output_root / "P0-2" / method / "seed{}".format(seed) / suffix
                jobs.append(
                    Job(
                        experiment="P0-2",
                        method=method,
                        training_seed=seed,
                        scope=scope,
                        selection_seed=selection,
                        source_run=b3_run,
                        a2_run=paired_a2,
                        output_dir=out,
                        log_path=output_root / "launcher_logs" / ("P02_{}_seed{}_{}.log".format(method, seed, suffix.replace("/", "_"))),
                    )
                )
    for seed in TRAINING_SEEDS:
        paired_a2 = _a2_run(a2_root, seed)
        for required in (
            paired_a2 / "resolved_config.yaml",
            paired_a2 / "model_final_trainable.pth",
        ):
            if not required.is_file():
                raise FileNotFoundError(str(required))
        for scope, selection in scopes:
            suffix = "full" if scope == "full" else "probe/selection_seed_{}".format(selection)
            out = output_root / "P0-34" / "A2" / "seed{}".format(seed) / suffix
            jobs.append(
                Job(
                    experiment="P0-34",
                    method="A2",
                    training_seed=seed,
                    scope=scope,
                    selection_seed=selection,
                    source_run=paired_a2,
                    a2_run=paired_a2,
                    output_dir=out,
                    log_path=output_root / "launcher_logs" / ("P034_A2_seed{}_{}.log".format(seed, suffix.replace("/", "_"))),
                )
            )
    return jobs


def _command(job: Job, args) -> List[str]:
    command = [
        str(args.python_bin),
        "-m",
        "src.tools.search_plans.b_series.replay_b3_p0_diagnostics",
        "--experiment",
        job.experiment,
        "--source-run",
        str(job.source_run),
        "--output-dir",
        str(job.output_dir),
        "--scope",
        job.scope,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if job.experiment == "P0-2":
        command.extend(["--a2-run", str(job.a2_run)])
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
    completed = process.returncode == 0 and _completed(job)
    return {
        "name": job.name,
        "experiment": job.experiment,
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


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3-root", type=Path, required=True)
    parser.add_argument("--a2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-groups", default="0;2;3;4;5;7")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    gpus = _gpus(args.gpu_groups)
    if args.max_workers < 1 or args.max_workers > len(gpus):
        raise SystemExit("--max-workers must be inside the GPU slot count")
    if args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1:
        raise SystemExit("invalid batch/worker/thread values")
    b3_root = args.b3_root.expanduser().resolve()
    a2_root = args.a2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    jobs = _build_jobs(b3_root, a2_root, output_root)
    pending = []
    skipped = []
    for job in jobs:
        if _completed(job):
            skipped.append(job)
        elif _has_contents(job.output_dir):
            raise RuntimeError("refusing to overwrite incomplete output {}".format(job.output_dir))
        else:
            pending.append(job)
    if args.dry_run:
        for index, job in enumerate(pending):
            print("[dry-run] gpu={} {}".format(gpus[index % args.max_workers], _format(_command(job, args))))
        return
    trials = [
        {
            "trial_index": index,
            "trial_name": job.name,
            "runner": "b3_p0",
            "stage": job.experiment,
            "combo": {
                "method": job.method,
                "training_seed": job.training_seed,
                "scope": job.scope,
                "selection_seed": job.selection_seed,
            },
            "overrides": {},
            "output_dir": str(job.output_dir),
            "identity_fields": {
                "source_run": str(job.source_run),
                "a2_run": str(job.a2_run),
            },
            "eta_fields": {"scope": job.scope, "experiment": job.experiment},
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
    status = "completed" if all(item.get("status") == "completed" for item in results) else "failed"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "b3_p0_queue_v1",
        "b3_root": str(b3_root),
        "a2_root": str(a2_root),
        "output_root": str(output_root),
        "gpu_groups": gpus,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "cpu_threads": args.cpu_threads,
        "job_count": len(jobs),
        "results": results,
        "skipped_existing": [job.name for job in skipped],
        "status": status,
    }
    (output_root / "b3_p0_queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if status != "completed":
        raise RuntimeError("one or more B3 P0 jobs failed")


if __name__ == "__main__":
    main()
