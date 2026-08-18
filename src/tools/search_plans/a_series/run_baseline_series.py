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


def _summary_command(
    python_bin: str,
    jobs: Sequence[Job],
    out_root: Path,
    seeds: Sequence[int],
    *,
    compact_probe_metrics: bool = True,
    compact_module_effect_json: bool = True,
) -> List[str]:
    models = []
    for job in jobs:
        if job.model not in models:
            models.append(job.model)
    command = [
        python_bin,
        str(ROOT / "src" / "tools" / "search_plans" / "a_series" / "summarize_baseline_monitoring.py"),
        "--baseline-method",
        "A0" if "A0" in models else models[0],
        "--output-dir",
        str(out_root / "a_series_summary"),
        "--expected-seeds",
        ",".join(str(seed) for seed in seeds),
    ]
    for job in jobs:
        command.extend(["--run", "{}={}".format(job.model, job.output_root)])
    if compact_probe_metrics:
        command.append("--compact-probe-metrics")
    if compact_module_effect_json:
        command.append("--compact-module-effect-json")
    return command


def _run_summary(
    python_bin: str,
    jobs: Sequence[Job],
    out_root: Path,
    seeds: Sequence[int],
    *,
    compact_probe_metrics: bool = True,
    compact_module_effect_json: bool = True,
) -> None:
    command = _summary_command(
        python_bin,
        jobs,
        out_root,
        seeds,
        compact_probe_metrics=compact_probe_metrics,
        compact_module_effect_json=compact_module_effect_json,
    )
    process = subprocess.run(command, cwd=str(ROOT), check=False)
    if process.returncode != 0:
        raise SystemExit("A-series runs completed, but monitoring summary generation failed.")
    print("A-series monitoring summary: {}".format(out_root / "a_series_summary"), flush=True)


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
    duration = time.time() - started
    completed = _completed_summary(job.output_root)
    if process.returncode == 0 and completed is not None:
        status = "completed"
    elif process.returncode == 0:
        status = "failed_missing_completed_summary"
    else:
        status = "failed"
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


def _progress_trial(job: Job, trial_index: int) -> Dict[str, object]:
    return {
        "trial_index": int(trial_index),
        "trial_name": "{}_seed{}".format(job.model, job.seed),
        "runner": "train",
        "stage": str(job.model),
        "combo": {"model": str(job.model), "seed": int(job.seed)},
        "overrides": {"SOLVER.TOTAL_EPOCH": 15},
        "output_dir": str(job.output_root),
        "identity_fields": {"config_file": str(job.config_file)},
        "eta_fields": {"model": str(job.model)},
        "eta_compatibility_keys": ["runner", "stage", "nproc", "total_epochs"],
    }


def _run_workers(
    all_jobs: Sequence[Job],
    pending_jobs: Sequence[Job],
    python_bin: str,
    gpu_groups: Sequence[str],
    max_workers: int,
    initial_completed: int,
    progress_interval: float,
    progress_width: int,
    progress_enabled: bool,
) -> List[Dict[str, object]]:
    all_trials = [_progress_trial(job, index) for index, job in enumerate(all_jobs)]
    trial_by_job = {job: trial for job, trial in zip(all_jobs, all_trials)}
    results = run_parallel_trials(
        all_trials=all_trials,
        pending_items=pending_jobs,
        pending_trials=[trial_by_job[job] for job in pending_jobs],
        out_root=Path(all_jobs[0].output_root).parents[1],
        gpu_groups=gpu_groups,
        max_workers=max_workers,
        initial_completed=initial_completed,
        progress_interval=progress_interval,
        progress_width=progress_width,
        progress_enabled=progress_enabled,
        run_item=lambda job, gpu: _run_job(job, python_bin, gpu),
        item_name=lambda job: "{} seed={} gpu-log={}".format(job.model, job.seed, job.log_path.name),
    )
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
    parser.add_argument("--progress-interval", type=float, default=2.0)
    parser.add_argument("--progress-width", type=int, default=120)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--keep-uncompressed-probe-metrics", action="store_true")
    parser.add_argument("--keep-uncompressed-module-effect-json", action="store_true")
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
    if args.progress_interval <= 0.0:
        raise SystemExit("--progress-interval must be positive.")
    if args.progress_width < 80:
        raise SystemExit("--progress-width must be at least 80.")
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
        if not args.no_summary:
            print("[dry-run summary] {}".format(
                _format_command(_summary_command(
                    args.python_bin,
                    jobs,
                    out_root,
                    seeds,
                    compact_probe_metrics=not args.keep_uncompressed_probe_metrics,
                    compact_module_effect_json=not args.keep_uncompressed_module_effect_json,
                ))
            ))
        return
    if not pending:
        print("All requested A-series tasks are already completed.", flush=True)
        if not args.no_summary:
            _run_summary(
                args.python_bin,
                jobs,
                out_root,
                seeds,
                compact_probe_metrics=not args.keep_uncompressed_probe_metrics,
                compact_module_effect_json=not args.keep_uncompressed_module_effect_json,
            )
        return

    results = _run_workers(
        jobs,
        pending,
        args.python_bin,
        gpu_groups,
        args.max_workers,
        len(skipped),
        args.progress_interval,
        args.progress_width,
        not args.no_progress,
    )
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
    if not args.no_summary:
        _run_summary(
            args.python_bin,
            jobs,
            out_root,
            seeds,
            compact_probe_metrics=not args.keep_uncompressed_probe_metrics,
            compact_module_effect_json=not args.keep_uncompressed_module_effect_json,
        )


if __name__ == "__main__":
    main()
