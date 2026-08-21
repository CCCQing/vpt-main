#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")


@dataclass(frozen=True)
class Job:
    method: str
    training_seed: int
    scope: str
    selection_seed: int
    source_run: Path
    output_dir: Path
    log_path: Path


def _parse_run(value: str):
    method, separator, seed_text = str(value).partition(":")
    method = method.strip().upper()
    if not separator or method not in {"B1", "B2"}:
        raise ValueError("--run must use B1:SEED or B2:SEED")
    seed = int(seed_text)
    if seed < 0:
        raise ValueError("training seed must be non-negative")
    return method, seed


def _selection_seeds(raw: str, scope: str) -> List[int]:
    if scope == "full":
        return [0]
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError("selection seeds must be unique non-negative integers")
    return values


def _gpu_groups(raw: str) -> List[str]:
    values = [value.strip() for value in str(raw).split(";") if value.strip()]
    if not values or len(set(values)) != len(values) or any("," in value for value in values):
        raise ValueError("GPU groups must be unique single cards separated by semicolons")
    return values


def _valid_output(path: Path) -> bool:
    summary = path / "b_series_replay_summary.json"
    if not summary.is_file():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("status") == "completed" and payload.get("valid") is True


def _has_contents(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _build_jobs(args, seeds: List[int]) -> List[Job]:
    jobs = []
    for raw in args.run:
        method, training_seed = _parse_run(raw)
        source_run = (
            args.source_root.resolve()
            / method
            / "seed{}".format(training_seed)
            / RUN_SUFFIX
        )
        if not source_run.is_dir():
            raise FileNotFoundError(str(source_run))
        for selection_seed in seeds:
            scope_identity = (
                "full"
                if args.scope == "full"
                else "probe_seed_{}".format(selection_seed)
            )
            output = (
                args.output_root.resolve()
                / scope_identity
                / method
                / "seed{}".format(training_seed)
            )
            jobs.append(
                Job(
                    method=method,
                    training_seed=training_seed,
                    scope=args.scope,
                    selection_seed=int(selection_seed),
                    source_run=source_run,
                    output_dir=output,
                    log_path=args.output_root.resolve()
                    / "launcher_logs"
                    / "{}_seed{}_{}.log".format(
                        method, training_seed, scope_identity
                    ),
                )
            )
    if len(set(jobs)) != len(jobs):
        raise ValueError("duplicate B1 replay jobs were requested")
    return jobs


def _command(job: Job, args) -> List[str]:
    command = [
        args.python_bin,
        str(Path(__file__).with_name("replay_b_series_experiments.py")),
        "--source-run",
        str(job.source_run),
        "--output-dir",
        str(job.output_dir),
        "--method-name",
        str(job.method),
        "--experiments",
        str(args.experiments),
        "--scope",
        str(job.scope),
        "--selection-seed",
        str(job.selection_seed),
        "--donor-seed",
        str(args.donor_seed),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.shortlist_layers:
        command.extend(["--shortlist-layers", str(args.shortlist_layers)])
    return command


def _run(job: Job, args, gpu: str) -> Dict[str, object]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with job.log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            _command(job, args),
            cwd=str(ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    valid = process.returncode == 0 and _valid_output(job.output_dir)
    return {
        "method": job.method,
        "training_seed": job.training_seed,
        "scope": job.scope,
        "selection_seed": job.selection_seed if job.scope == "probe" else None,
        "gpu": gpu,
        "status": "completed" if valid else "failed",
        "returncode": int(process.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "source_run": str(job.source_run),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Run E1..E4 replay jobs in parallel.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--scope", choices=("full", "probe"), default="probe")
    parser.add_argument("--selection-seeds", default="424242,424243,424244")
    parser.add_argument("--experiments", default="E1,E2,E3,E4")
    parser.add_argument("--shortlist-layers", default="")
    parser.add_argument("--donor-seed", type=int, default=130363)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-groups", default="0;1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = _selection_seeds(args.selection_seeds, args.scope)
    gpus = _gpu_groups(args.gpu_groups)
    if args.max_workers <= 0 or args.max_workers > len(gpus):
        raise SystemExit("--max-workers must be positive and no larger than GPU count")
    jobs = _build_jobs(args, seeds)
    pending = []
    skipped = []
    for job in jobs:
        if _valid_output(job.output_dir):
            skipped.append(job)
        elif _has_contents(job.output_dir):
            raise RuntimeError("refusing to overwrite {}".format(job.output_dir))
        else:
            pending.append(job)
    if args.dry_run:
        for index, job in enumerate(pending):
            print(
                "[dry-run] gpu={} {}".format(
                    gpus[index % min(args.max_workers, len(gpus))],
                    subprocess.list2cmdline(_command(job, args)),
                )
            )
        return
    trials = [
        {
            "trial_index": index,
            "trial_name": "{}_seed{}_{}_{}".format(
                job.method, job.training_seed, job.scope, job.selection_seed
            ),
            "runner": "b_series_experiments_replay",
            "stage": job.scope,
            "combo": {
                "method": job.method,
                "training_seed": job.training_seed,
                "selection_seed": job.selection_seed,
            },
            "overrides": {},
            "output_dir": str(job.output_dir),
            "identity_fields": {"source_run": str(job.source_run)},
            "eta_fields": {"scope": job.scope},
            "eta_compatibility_keys": ["runner", "stage"],
        }
        for index, job in enumerate(jobs)
    ]
    trial_by_job = {job: trial for job, trial in zip(jobs, trials)}
    results = (
        run_parallel_trials(
            all_trials=trials,
            pending_items=pending,
            pending_trials=[trial_by_job[job] for job in pending],
            out_root=args.output_root.resolve(),
            gpu_groups=gpus,
            max_workers=args.max_workers,
            initial_completed=len(skipped),
            progress_interval=2.0,
            progress_width=120,
            progress_enabled=not args.no_progress,
            run_item=lambda job, gpu: _run(job, args, gpu),
            item_name=lambda job: "{} seed={} selection={}".format(
                job.method, job.training_seed, job.selection_seed
            ),
        )
        if pending
        else []
    )
    failed = [result for result in results if result["status"] != "completed"]
    summary = {
        "format": "b_series_experiments_replay_launcher_v1",
        "suite_name": "B-series",
        "scope": args.scope,
        "selection_seeds": seeds if args.scope == "probe" else None,
        "experiments": args.experiments,
        "results": results,
        "skipped_existing_count": len(skipped),
        "status": "failed" if failed else "completed",
    }
    path = args.output_root.resolve() / "b_series_replay_launcher_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
