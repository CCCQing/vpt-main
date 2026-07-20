from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIGS = {
    "A0": ROOT / "configs" / "baseline_rebuild" / "A-02-A0-frozen-vit-ce.yaml",
    "A1": ROOT / "configs" / "baseline_rebuild" / "A-03-A1-vpt-shallow-ce.yaml",
    "A2": ROOT / "configs" / "baseline_rebuild" / "A-04-A2-vpt-deep-ce.yaml",
}


@dataclass(frozen=True)
class Job:
    model: str
    seed: int
    config_file: Path
    output_root: Path
    log_path: Path


def _parse_models(raw: str) -> List[str]:
    models = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not models:
        raise ValueError("--models must contain at least one of A0,A1,A2.")
    if len(set(models)) != len(models):
        raise ValueError("--models must not contain duplicates.")
    unknown = [model for model in models if model not in MODEL_CONFIGS]
    if unknown:
        raise ValueError("Unknown A-series model(s): {}".format(",".join(unknown)))
    return models


def _parse_seeds(raw: str) -> List[int]:
    try:
        seeds = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    except ValueError:
        raise ValueError("--seeds must be comma-separated non-negative integers.")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("--seeds must contain unique non-negative integers.")
    return seeds


def _parse_gpu_groups(raw: str) -> List[str]:
    groups = [item.strip() for item in str(raw).split(";") if item.strip()]
    if not groups:
        raise ValueError("--gpu-groups must contain at least one GPU, for example '0;1'.")
    if any("," in group for group in groups):
        raise ValueError(
            "Each A-series task is single-GPU. Use semicolons between cards, for example '0;1'."
        )
    if len(set(groups)) != len(groups):
        raise ValueError("--gpu-groups must not contain duplicate GPU entries.")
    return groups


def _runtime_summaries(output_root: Path) -> List[Path]:
    if not output_root.is_dir():
        return []
    return sorted(output_root.rglob("monitor_runtime_summary.json"))


def _completed_summary(output_root: Path) -> Optional[Path]:
    completed = []
    for path in _runtime_summaries(output_root):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if str(payload.get("status", "")).lower() == "completed":
            completed.append(path)
    if len(completed) > 1:
        raise RuntimeError(
            "Expected at most one completed run under {}, found {}.".format(
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


def _build_jobs(out_root: Path, models: Sequence[str], seeds: Sequence[int]) -> List[Job]:
    log_root = out_root / "launcher_logs"
    return [
        Job(
            model=model,
            seed=int(seed),
            config_file=MODEL_CONFIGS[model],
            output_root=out_root / model / "seed{}".format(seed),
            log_path=log_root / "{}_seed{}.log".format(model, seed),
        )
        for model in models
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
            if completed is not None:
                reason = "--no-resume requested but a completed output already exists"
            else:
                reason = "an incomplete or unrecognized output already exists"
            raise RuntimeError(
                "Refusing to overwrite {} for {} seed {}: {}. Move that task directory or use "
                "a new --out-root.".format(job.output_root, job.model, job.seed, reason)
            )
        pending.append(job)
    return pending, skipped


def _run_job(job: Job, python_bin: str, gpu: str) -> Dict[str, object]:
    command = _command(python_bin, job)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        "[start] {} seed={} gpu={} log={}".format(job.model, job.seed, gpu, job.log_path),
        flush=True,
    )
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
    duration = time.time() - started
    completed = _completed_summary(job.output_root)
    if process.returncode == 0 and completed is not None:
        status = "completed"
    elif process.returncode == 0:
        status = "failed_missing_completed_summary"
    else:
        status = "failed"
    print(
        "[{}] {} seed={} gpu={} returncode={} duration_min={:.1f}".format(
            status, job.model, job.seed, gpu, process.returncode, duration / 60.0
        ),
        flush=True,
    )
    return {
        "model": job.model,
        "seed": job.seed,
        "gpu": gpu,
        "status": status,
        "returncode": process.returncode,
        "duration_seconds": round(duration, 3),
        "output_root": str(job.output_root),
        "log_path": str(job.log_path),
    }


def _run_workers(
    jobs: Sequence[Job], python_bin: str, gpu_groups: Sequence[str], max_workers: int
) -> List[Dict[str, object]]:
    worker_count = min(max_workers, len(gpu_groups), len(jobs))
    assignments = [[] for _ in range(worker_count)]
    for index, job in enumerate(jobs):
        assignments[index % worker_count].append(job)

    def run_worker(worker_index: int) -> List[Dict[str, object]]:
        gpu = gpu_groups[worker_index]
        return [_run_job(job, python_bin, gpu) for job in assignments[worker_index]]

    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_worker, index) for index in range(worker_count)]
        for future in as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda item: (str(item["model"]), int(item["seed"])))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the A0/A1/A2 baseline series across fixed single-GPU workers."
    )
    parser.add_argument("--models", default="A0,A1,A2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpu-groups", default="0;1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "output" / "baseline_rebuild"
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        models = _parse_models(args.models)
        seeds = _parse_seeds(args.seeds)
        gpu_groups = _parse_gpu_groups(args.gpu_groups)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive.")
    if args.max_workers > len(gpu_groups):
        raise SystemExit("--max-workers must not exceed the number of --gpu-groups entries.")
    missing = [str(MODEL_CONFIGS[model]) for model in models if not MODEL_CONFIGS[model].is_file()]
    if missing:
        raise SystemExit("Missing A-series config(s): {}".format(", ".join(missing)))

    out_root = args.out_root.expanduser().resolve()
    jobs = _build_jobs(out_root, models, seeds)
    try:
        pending, skipped = _select_pending(jobs, args.no_resume)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    print(
        "A-series plan: models={} seeds={} total={} pending={} skipped={} gpus={} workers={}".format(
            ",".join(models),
            ",".join(str(seed) for seed in seeds),
            len(jobs),
            len(pending),
            len(skipped),
            ";".join(gpu_groups),
            args.max_workers,
        ),
        flush=True,
    )
    for job in skipped:
        print("[skip completed] {} seed={} output={}".format(job.model, job.seed, job.output_root))
    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpu_groups[index % min(args.max_workers, len(gpu_groups), max(len(pending), 1))]
            print(
                "[dry-run] gpu={} {}".format(gpu, _format_command(_command(args.python_bin, job)))
            )
        return
    if not pending:
        print("All requested A-series tasks are already completed.", flush=True)
        return

    results = _run_workers(pending, args.python_bin, gpu_groups, args.max_workers)
    failed = [result for result in results if result["status"] != "completed"]
    print(
        "A-series finished: completed={} failed={} skipped={}.".format(
            len(results) - len(failed), len(failed), len(skipped)
        ),
        flush=True,
    )
    if failed:
        for result in failed:
            print(
                "[failure] {model} seed={seed} status={status} log={log_path}".format(**result),
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
