#!/usr/bin/env python3

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
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials


METHOD_CONFIGS = {
    "B1": ROOT / "configs" / "baseline_rebuild" / "B-01-direct-mean-residual.yaml",
    "B2": ROOT / "configs" / "baseline_rebuild" / "B-02-direct-mean-nonconditional-control.yaml",
}


@dataclass(frozen=True)
class Job:
    method: str
    seed: int
    config_file: Path
    output_root: Path
    log_path: Path


def _parse_methods(raw: str) -> List[str]:
    methods = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain B1 and/or B2")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods must not contain duplicates")
    unknown = [method for method in methods if method not in METHOD_CONFIGS]
    if unknown:
        raise ValueError("unknown B-series method(s): {}".format(",".join(unknown)))
    return methods


def _parse_seeds(raw: str) -> List[int]:
    try:
        seeds = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated non-negative integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("--seeds must contain unique non-negative integers")
    return seeds


def _parse_gpu_groups(raw: str) -> List[str]:
    groups = [item.strip() for item in str(raw).split(";") if item.strip()]
    if not groups:
        raise ValueError("--gpu-groups must contain at least one GPU, for example '0;1'")
    if any("," in group for group in groups):
        raise ValueError(
            "each B-series task is single-GPU; separate cards with semicolons"
        )
    if len(set(groups)) != len(groups):
        raise ValueError("--gpu-groups must not contain duplicate GPU entries")
    return groups


def _runtime_summaries(output_root: Path) -> List[Path]:
    if not output_root.is_dir():
        return []
    return sorted(output_root.rglob("monitor_runtime_summary.json"))


def _completed_summary(output_root: Path) -> Optional[Path]:
    completed = []
    for path in _runtime_summaries(output_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(payload.get("status", "")).lower() == "completed":
            completed.append(path)
    if len(completed) > 1:
        raise RuntimeError(
            "expected at most one completed run under {}, found {}".format(
                output_root, len(completed)
            )
        )
    return completed[0] if completed else None


def _has_contents(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _command(python_bin: str, job: Job) -> List[str]:
    return [
        python_bin,
        str(ROOT / "train.py"),
        "--config-file",
        str(job.config_file),
        "SEED",
        str(job.seed),
        "OUTPUT_DIR",
        str(job.output_root),
        "RUN_N_TIMES",
        "1",
        "NUM_GPUS",
        "1",
        "NUM_SHARDS",
        "1",
    ]


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def _build_jobs(out_root: Path, methods: Sequence[str], seeds: Sequence[int]) -> List[Job]:
    log_root = out_root / "launcher_logs"
    return [
        Job(
            method=method,
            seed=int(seed),
            config_file=METHOD_CONFIGS[method],
            output_root=out_root / method / "seed{}".format(seed),
            log_path=log_root / "{}_seed{}.log".format(method, seed),
        )
        for method in methods
        for seed in seeds
    ]


def _select_pending(jobs: Sequence[Job], no_resume: bool) -> Tuple[List[Job], List[Job]]:
    pending = []
    skipped = []
    for job in jobs:
        completed = _completed_summary(job.output_root)
        if completed is not None and not no_resume:
            skipped.append(job)
            continue
        if _has_contents(job.output_root):
            raise RuntimeError(
                "refusing to overwrite {} for {} seed {}; use a new --out-root".format(
                    job.output_root, job.method, job.seed
                )
            )
        pending.append(job)
    return pending, skipped


def _run_job(job: Job, python_bin: str, gpu: str) -> Dict[str, object]:
    command = _command(python_bin, job)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["VPT_SEARCH_PROGRESS_PATH"] = str((job.output_root / "progress.json").resolve())
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with job.log_path.open("w", encoding="utf-8") as handle:
        handle.write("CUDA_VISIBLE_DEVICES={}\n".format(gpu))
        handle.write(_format_command(command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    completed = _completed_summary(job.output_root)
    status = (
        "completed"
        if process.returncode == 0 and completed is not None
        else "failed_missing_completed_summary"
        if process.returncode == 0
        else "failed"
    )
    return {
        "method": job.method,
        "seed": job.seed,
        "gpu": gpu,
        "status": status,
        "returncode": process.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "output_root": str(job.output_root),
        "log_path": str(job.log_path),
    }


def _progress_trial(job: Job, trial_index: int) -> Dict[str, object]:
    return {
        "trial_index": int(trial_index),
        "trial_name": "{}_seed{}".format(job.method, job.seed),
        "runner": "train",
        "stage": str(job.method),
        "combo": {"method": str(job.method), "seed": int(job.seed)},
        "overrides": {"SOLVER.TOTAL_EPOCH": 15},
        "output_dir": str(job.output_root),
        "identity_fields": {"config_file": str(job.config_file)},
        "eta_fields": {"method": str(job.method)},
        "eta_compatibility_keys": ["runner", "stage", "nproc", "total_epochs"],
    }


def _write_summary(out_root: Path, payload: Dict[str, object]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "b_series_launcher_summary.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B1/B2 deterministic Prompt Distributor experiments"
    )
    parser.add_argument("--methods", default="B1,B2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpu-groups", default="0;1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "output" / "prompt_distribution_b_series_round1",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=2.0)
    parser.add_argument("--progress-width", type=int, default=120)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        methods = _parse_methods(args.methods)
        seeds = _parse_seeds(args.seeds)
        gpu_groups = _parse_gpu_groups(args.gpu_groups)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.max_workers <= 0 or args.max_workers > len(gpu_groups):
        raise SystemExit("--max-workers must be positive and no larger than GPU count")
    if args.progress_interval <= 0.0 or args.progress_width < 80:
        raise SystemExit("invalid progress interval or width")
    missing = [
        str(METHOD_CONFIGS[method])
        for method in methods
        if not METHOD_CONFIGS[method].is_file()
    ]
    if missing:
        raise SystemExit("missing B-series config(s): {}".format(", ".join(missing)))

    out_root = args.out_root.expanduser().resolve()
    jobs = _build_jobs(out_root, methods, seeds)
    try:
        pending, skipped = _select_pending(jobs, args.no_resume)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    print(
        "B-series plan: methods={} seeds={} total={} pending={} skipped={} gpus={} workers={}".format(
            ",".join(methods),
            ",".join(str(seed) for seed in seeds),
            len(jobs),
            len(pending),
            len(skipped),
            ";".join(gpu_groups),
            args.max_workers,
        ),
        flush=True,
    )
    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpu_groups[index % min(args.max_workers, len(gpu_groups))]
            print("[dry-run] gpu={} {}".format(gpu, _format_command(_command(args.python_bin, job))))
        return

    all_trials = [_progress_trial(job, index) for index, job in enumerate(jobs)]
    trial_by_job = {job: trial for job, trial in zip(jobs, all_trials)}
    results = run_parallel_trials(
        all_trials=all_trials,
        pending_items=pending,
        pending_trials=[trial_by_job[job] for job in pending],
        out_root=out_root,
        gpu_groups=gpu_groups,
        max_workers=args.max_workers,
        initial_completed=len(skipped),
        progress_interval=args.progress_interval,
        progress_width=args.progress_width,
        progress_enabled=not args.no_progress,
        run_item=lambda job, gpu: _run_job(job, args.python_bin, gpu),
        item_name=lambda job: "{} seed={} log={}".format(
            job.method, job.seed, job.log_path.name
        ),
    ) if pending else []
    failed = [result for result in results if result["status"] != "completed"]
    _write_summary(
        out_root,
        {
            "format": "prompt_distribution_b_series_launcher_v1",
            "methods": methods,
            "seeds": seeds,
            "gpu_groups": gpu_groups,
            "max_workers": int(args.max_workers),
            "completed_before_launch": [
                {"method": job.method, "seed": job.seed, "output_root": str(job.output_root)}
                for job in skipped
            ],
            "results": results,
            "status": "failed" if failed else "completed",
        },
    )
    if failed:
        raise SystemExit(1)
    print(
        "B-series finished: completed={} skipped={} failed=0".format(
            len(results), len(skipped)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
