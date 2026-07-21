from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials


DEFAULT_CONFIG = ROOT / "configs" / "baseline_rebuild" / "A-04-A2-vpt-deep-ce.yaml"
FIXED_WEIGHT_DECAY = 1.0e-5


@dataclass(frozen=True)
class Job:
    index: int
    num_tokens: int
    base_lr: float
    total_epoch: int
    seed: int
    output_root: Path
    log_path: Path

    @property
    def trial_name(self) -> str:
        return self.output_root.name


def _parse_int_grid(raw: str, name: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    except ValueError:
        raise ValueError("{} must be a comma-separated integer list.".format(name))
    if not values or len(set(values)) != len(values) or min(values) <= 0:
        raise ValueError("{} must contain unique positive integers.".format(name))
    return values


def _parse_float_grid(raw: str, name: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    except ValueError:
        raise ValueError("{} must be a comma-separated numeric list.".format(name))
    if not values or len(set(values)) != len(values) or min(values) <= 0:
        raise ValueError("{} must contain unique positive values.".format(name))
    return values


def _parse_gpu_groups(raw: str) -> List[str]:
    groups = [item.strip() for item in str(raw).split(";") if item.strip()]
    if not groups:
        raise ValueError("--gpu-groups must contain at least one GPU, for example '0;1'.")
    if any("," in group for group in groups):
        raise ValueError("Each trial is single-GPU; separate cards with semicolons, for example '0;1'.")
    if len(set(groups)) != len(groups):
        raise ValueError("--gpu-groups must not contain duplicate GPU entries.")
    return groups


def _float_tag(value: float) -> str:
    mantissa, exponent = "{:.8e}".format(float(value)).split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return "{}e{}".format(mantissa, int(exponent))


def _trial_name(index: int, num_tokens: int, base_lr: float, total_epoch: int) -> str:
    return "exp{index:03d}_tok{tokens}_lr{lr}_ep{epochs}".format(
        index=index,
        tokens=num_tokens,
        lr=_float_tag(base_lr),
        epochs=total_epoch,
    )


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def _runtime_summaries(output_root: Path) -> List[Path]:
    return sorted(output_root.rglob("monitor_runtime_summary.json")) if output_root.is_dir() else []


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
        raise RuntimeError("Multiple completed runs found under {}.".format(output_root))
    return completed[0] if completed else None


def _has_contents(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _find_log(output_root: Path) -> Optional[Path]:
    logs = sorted(output_root.rglob("logs.txt")) if output_root.is_dir() else []
    if len(logs) > 1:
        raise RuntimeError("Multiple logs.txt files found under {}.".format(output_root))
    return logs[0] if logs else None


def _parse_gzsl_records(log_path: Path) -> Dict[str, object]:
    pattern = re.compile(
        r"\[gzsl-record\]\s+epoch=(\d+)\s+gzsl_seen=([0-9eE+.-]+)\s+"
        r"gzsl_unseen=([0-9eE+.-]+)\s+gzsl_h=([0-9eE+.-]+)"
    )
    records: List[Tuple[int, float, float, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                records.append(
                    (
                        int(match.group(1)),
                        100.0 * float(match.group(2)),
                        100.0 * float(match.group(3)),
                        100.0 * float(match.group(4)),
                    )
                )
    if not records:
        return {}
    best = max(records, key=lambda item: (item[3], item[0]))
    final = records[-1]
    return {
        "best_epoch": best[0],
        "gzsl_seen_at_best_h": round(best[1], 4),
        "gzsl_unseen_at_best_h": round(best[2], 4),
        "gzsl_h_best": round(best[3], 4),
        "final_epoch": final[0],
        "gzsl_seen_final": round(final[1], 4),
        "gzsl_unseen_final": round(final[2], 4),
        "gzsl_h_final": round(final[3], 4),
    }


def _command(python_bin: str, config_file: Path, job: Job) -> List[str]:
    return [
        python_bin,
        str(ROOT / "train.py"),
        "--config-file",
        str(config_file),
        "OUTPUT_DIR",
        str(job.output_root),
        "SEED",
        str(job.seed),
        "RUN_N_TIMES",
        "1",
        "NUM_GPUS",
        "1",
        "NUM_SHARDS",
        "1",
        "DATA.XLSA.PROTOCOL_MODE",
        "final_gzsl",
        "MODEL.CLASSIFIER",
        "vspcn_baseline",
        "MODEL.PROMPT.ENABLE",
        "True",
        "MODEL.PROMPT.BACKEND",
        "vpt_deep",
        "MODEL.PROMPT.INIT_SOURCE",
        "learned",
        "MODEL.PROMPT.DEEP",
        "True",
        "MODEL.PROMPT.NUM_TOKENS",
        str(job.num_tokens),
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE",
        "False",
        "SOLVER.MAIN_LOSS",
        "vspcn",
        "SOLVER.LOSS_VSPCN_AR_WEIGHT",
        "0.0",
        "SOLVER.BASE_LR",
        str(job.base_lr),
        "SOLVER.TOTAL_EPOCH",
        str(job.total_epoch),
        "SOLVER.WEIGHT_DECAY",
        str(FIXED_WEIGHT_DECAY),
    ]


def _base_row(job: Job) -> Dict[str, object]:
    return {
        "trial_name": job.trial_name,
        "num_tokens": job.num_tokens,
        "base_lr": job.base_lr,
        "total_epoch": job.total_epoch,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "ar_weight": 0.0,
        "seed": job.seed,
        "status": "pending",
        "returncode": "",
        "duration_seconds": "",
        "best_epoch": "",
        "gzsl_seen_at_best_h": "",
        "gzsl_unseen_at_best_h": "",
        "gzsl_h_best": "",
        "final_epoch": "",
        "gzsl_seen_final": "",
        "gzsl_unseen_final": "",
        "gzsl_h_final": "",
        "output_root": str(job.output_root),
        "run_dir": "",
        "launcher_log": str(job.log_path),
    }


def _read_existing(job: Job) -> Dict[str, object]:
    row = _base_row(job)
    log_path = _find_log(job.output_root)
    if log_path is None:
        row["status"] = "failed_missing_logs"
        return row
    row.update(_parse_gzsl_records(log_path))
    row["run_dir"] = str(log_path.parent)
    row["status"] = "completed_existing" if row.get("gzsl_h_final", "") != "" else "failed_missing_metrics"
    row["returncode"] = 0
    return row


def _run_job(job: Job, python_bin: str, config_file: Path, gpu: str) -> Dict[str, object]:
    command = _command(python_bin, config_file, job)
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
    row = _base_row(job)
    row["returncode"] = process.returncode
    row["duration_seconds"] = round(time.time() - started, 3)
    log_path = _find_log(job.output_root)
    if log_path is not None:
        row.update(_parse_gzsl_records(log_path))
        row["run_dir"] = str(log_path.parent)
    if process.returncode != 0:
        row["status"] = "failed"
    elif _completed_summary(job.output_root) is None:
        row["status"] = "failed_missing_completed_summary"
    elif row.get("gzsl_h_final", "") == "":
        row["status"] = "failed_missing_metrics"
    else:
        row["status"] = "completed"
    return row


def _sort_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    def score(row: Dict[str, object]) -> Tuple[float, float, str]:
        final_h = float(row["gzsl_h_final"]) if row.get("gzsl_h_final", "") != "" else float("-inf")
        best_h = float(row["gzsl_h_best"]) if row.get("gzsl_h_best", "") != "" else float("-inf")
        return (-final_h, -best_h, str(row["trial_name"]))

    return sorted(rows, key=score)


def _write_summaries(out_root: Path, rows: Sequence[Dict[str, object]]) -> None:
    ordered = _sort_rows(rows)
    fieldnames = list(_base_row(
        Job(0, 0, 0.0, 0, 0, Path("placeholder"), Path("placeholder"))
    ).keys())
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    with (out_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)


def _build_jobs(
    out_root: Path,
    token_grid: Sequence[int],
    lr_grid: Sequence[float],
    epoch_grid: Sequence[int],
    seed: int,
) -> List[Job]:
    jobs = []
    for index, (num_tokens, base_lr, total_epoch) in enumerate(
        product(token_grid, lr_grid, epoch_grid), start=1
    ):
        trial_name = _trial_name(index, num_tokens, base_lr, total_epoch)
        jobs.append(
            Job(
                index=index,
                num_tokens=int(num_tokens),
                base_lr=float(base_lr),
                total_epoch=int(total_epoch),
                seed=int(seed),
                output_root=out_root / trial_name,
                log_path=out_root / "launcher_logs" / "{}.log".format(trial_name),
            )
        )
    return jobs


def _progress_trial(job: Job, trial_index: int) -> Dict[str, object]:
    return {
        "trial_index": int(trial_index),
        "trial_name": str(job.trial_name),
        "runner": "train",
        "stage": "A2_grid",
        "combo": {
            "num_tokens": int(job.num_tokens),
            "base_lr": float(job.base_lr),
            "total_epoch": int(job.total_epoch),
            "seed": int(job.seed),
        },
        "overrides": {
            "SOLVER.TOTAL_EPOCH": int(job.total_epoch),
            "MODEL.PROMPT.NUM_TOKENS": int(job.num_tokens),
        },
        "output_dir": str(job.output_root),
        "identity_fields": {"base_lr": float(job.base_lr), "seed": int(job.seed)},
        "eta_fields": {"num_tokens": int(job.num_tokens)},
        "eta_compatibility_keys": ["runner", "stage", "nproc", "total_epochs"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small final_gzsl grid around the historical VPT-Deep baseline."
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-tokens-grid", default="8,16,32")
    parser.add_argument("--lr-grid", default="6e-4,1e-3,1.4e-3")
    parser.add_argument("--epoch-grid", default="15,20,25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-groups", default="0;1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "output" / "vpt_deep_final_gzsl_small_grid",
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
        token_grid = _parse_int_grid(args.num_tokens_grid, "--num-tokens-grid")
        lr_grid = _parse_float_grid(args.lr_grid, "--lr-grid")
        epoch_grid = _parse_int_grid(args.epoch_grid, "--epoch-grid")
        gpu_groups = _parse_gpu_groups(args.gpu_groups)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative.")
    if args.max_workers <= 0 or args.max_workers > len(gpu_groups):
        raise SystemExit("--max-workers must be positive and no larger than the GPU-group count.")
    if args.progress_interval <= 0.0:
        raise SystemExit("--progress-interval must be positive.")
    if args.progress_width < 80:
        raise SystemExit("--progress-width must be at least 80.")
    config_file = args.config_file.expanduser().resolve()
    if not config_file.is_file():
        raise SystemExit("Missing config file: {}".format(config_file))
    out_root = args.out_root.expanduser().resolve()
    jobs = _build_jobs(out_root, token_grid, lr_grid, epoch_grid, args.seed)

    pending = []
    existing_rows = []
    for job in jobs:
        completed = _completed_summary(job.output_root)
        if completed is not None and not args.no_resume:
            existing_rows.append(_read_existing(job))
        elif _has_contents(job.output_root):
            raise SystemExit(
                "Refusing to overwrite existing trial output: {}. Use a new --out-root.".format(
                    job.output_root
                )
            )
        else:
            pending.append(job)

    plan = {
        "config_file": str(config_file),
        "protocol_mode": "final_gzsl",
        "prompt_backend": "vpt_deep",
        "prompt_init_source": "learned",
        "prompt_deep": True,
        "num_tokens_grid": token_grid,
        "lr_grid": lr_grid,
        "epoch_grid": epoch_grid,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "ar_weight": 0.0,
        "seed": args.seed,
        "total_trials": len(jobs),
        "selection_metric": "gzsl_h_final",
        "gpu_groups": gpu_groups,
        "max_workers": args.max_workers,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        worker_count = min(args.max_workers, len(gpu_groups), max(len(pending), 1))
        for index, job in enumerate(pending):
            gpu = gpu_groups[index % worker_count]
            print("[dry-run] gpu={} {}".format(gpu, _format_command(
                _command(args.python_bin, config_file, job)
            )))
        return

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "search_space.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    if not pending:
        _write_summaries(out_root, existing_rows)
        print("All {} grid trials are already completed.".format(len(jobs)), flush=True)
        return

    all_trials = [_progress_trial(job, index) for index, job in enumerate(jobs)]
    trial_by_job = {job: trial for job, trial in zip(jobs, all_trials)}
    new_rows = run_parallel_trials(
        all_trials=all_trials,
        pending_items=pending,
        pending_trials=[trial_by_job[job] for job in pending],
        out_root=out_root,
        gpu_groups=gpu_groups,
        max_workers=args.max_workers,
        initial_completed=len(existing_rows),
        progress_interval=args.progress_interval,
        progress_width=args.progress_width,
        progress_enabled=not args.no_progress,
        run_item=lambda job, gpu: _run_job(job, args.python_bin, config_file, gpu),
        item_name=lambda job: str(job.trial_name),
        on_result=lambda rows: _write_summaries(out_root, existing_rows + list(rows)),
    )

    rows = existing_rows + new_rows
    _write_summaries(out_root, rows)
    failed = [row for row in rows if not str(row["status"]).startswith("completed")]
    top = _sort_rows(rows)[:5]
    print(json.dumps({"top5": top, "failed": len(failed)}, ensure_ascii=False, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
