from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials
from src.tools.search_plans.common import directory_has_contents as _has_contents


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "baseline_rebuild"
    / "A-05-A2-vpt-deep-cosine-search.yaml"
)
DEFAULT_OUTPUT = ROOT / "output" / "a2_cosine_baseline_search"
BASE_NUM_TOKENS = 16
BASE_LR = 6.0e-4
BASE_EPOCHS = 15
WEIGHT_DECAY = 1.0e-5


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    learnable: bool
    fixed_scale: float
    init_scale: float


SCALE_SPECS = {
    "fixed1": ScaleSpec("fixed1", False, 1.0, 10.0),
    "fixed10": ScaleSpec("fixed10", False, 10.0, 10.0),
    "learnable10": ScaleSpec("learnable10", True, 0.0, 10.0),
}


@dataclass(frozen=True)
class Job:
    stage: str
    num_tokens: int
    base_lr: float
    total_epoch: int
    seed: int
    scale: ScaleSpec
    output_root: Path
    log_path: Path

    @property
    def config_key(self) -> Tuple[int, float, int, str]:
        return (
            int(self.num_tokens),
            float(self.base_lr),
            int(self.total_epoch),
            str(self.scale.name),
        )

    @property
    def trial_name(self) -> str:
        return self.output_root.name


def _parse_ints(raw: str, name: str, *, allow_zero: bool = False) -> List[int]:
    try:
        values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("{} must be a comma-separated integer list.".format(name)) from exc
    minimum = 0 if allow_zero else 1
    if not values or len(set(values)) != len(values) or min(values) < minimum:
        raise ValueError("{} contains invalid or duplicate values.".format(name))
    return values


def _parse_floats(raw: str, name: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("{} must be a comma-separated numeric list.".format(name)) from exc
    if not values or len(set(values)) != len(values) or min(values) <= 0.0:
        raise ValueError("{} contains invalid or duplicate values.".format(name))
    return values


def _parse_gpu_groups(raw: str) -> List[str]:
    groups = [item.strip() for item in str(raw).split(";") if item.strip()]
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("--gpu-groups must contain unique GPU ids separated by semicolons.")
    if any("," in group for group in groups):
        raise ValueError("Each training job is single-GPU; use semicolons between GPU ids.")
    return groups


def _float_tag(value: float) -> str:
    mantissa, exponent = "{:.8e}".format(float(value)).split("e")
    return "{}e{}".format(mantissa.rstrip("0").rstrip(".").replace(".", "p"), int(exponent))


def _name(num_tokens: int, base_lr: float, total_epoch: int, scale: ScaleSpec, seed: int) -> str:
    return "tok{}_lr{}_ep{}_{}_seed{}".format(
        int(num_tokens), _float_tag(base_lr), int(total_epoch), scale.name, int(seed)
    )


def _job(
    out_root: Path,
    stage: str,
    num_tokens: int,
    base_lr: float,
    total_epoch: int,
    seed: int,
    scale: ScaleSpec,
) -> Job:
    name = _name(num_tokens, base_lr, total_epoch, scale, seed)
    return Job(
        stage=str(stage),
        num_tokens=int(num_tokens),
        base_lr=float(base_lr),
        total_epoch=int(total_epoch),
        seed=int(seed),
        scale=scale,
        output_root=out_root / stage / name,
        log_path=out_root / "launcher_logs" / "{}_{}.log".format(stage, name),
    )


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


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
        "MODEL.R_SIMILARITY.SCORE_MODE",
        "cosine",
        "MODEL.R_SIMILARITY.LEARNABLE_SCALE",
        str(bool(job.scale.learnable)),
        "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE",
        str(float(job.scale.fixed_scale)),
        "MODEL.R_SIMILARITY.LOGIT_SCALE_INIT",
        str(float(job.scale.init_scale)),
        "MODEL.PROMPT.NUM_TOKENS",
        str(job.num_tokens),
        "SOLVER.BASE_LR",
        str(job.base_lr),
        "SOLVER.TOTAL_EPOCH",
        str(job.total_epoch),
        "SOLVER.WEIGHT_DECAY",
        str(WEIGHT_DECAY),
        "MONITOR.PROBE.ENABLE",
        "False",
        "MONITOR.MODULE_EFFECT.ENABLE",
        "False",
        "MONITOR.MILESTONE_PROBE.ENABLE",
        "False",
    ]


def _runtime_summary(output_root: Path) -> Optional[Path]:
    completed = []
    if output_root.is_dir():
        for path in output_root.rglob("monitor_runtime_summary.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(payload.get("status", "")).lower() == "completed":
                completed.append(path)
    if len(completed) > 1:
        raise RuntimeError("Multiple completed runtime summaries found under {}.".format(output_root))
    return completed[0] if completed else None


def _metric_rows(metrics_path: Path) -> List[Dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_result(job: Job, *, existing: bool) -> Dict[str, object]:
    row: Dict[str, object] = {
        "stage": job.stage,
        "trial_name": job.trial_name,
        "num_tokens": job.num_tokens,
        "base_lr": job.base_lr,
        "total_epoch": job.total_epoch,
        "seed": job.seed,
        "score_mode": "cosine",
        "scale_policy": job.scale.name,
        "learnable_scale": job.scale.learnable,
        "fixed_logit_scale": job.scale.fixed_scale,
        "logit_scale_init": job.scale.init_scale,
        "weight_decay": WEIGHT_DECAY,
        "gzsl_seen": None,
        "gzsl_unseen": None,
        "gzsl_h": None,
        "ausuc": None,
        "final_effective_logit_scale": None,
        "ausuc_guard_pass": None,
        "status": "missing",
        "returncode": None,
        "duration_seconds": None,
        "output_root": str(job.output_root),
        "run_dir": None,
        "launcher_log": str(job.log_path),
    }
    runtime_path = _runtime_summary(job.output_root)
    if runtime_path is None:
        row["status"] = "failed_missing_completed_summary"
        return row
    metrics_path = runtime_path.parent / "metrics_epoch.csv"
    if not metrics_path.is_file():
        row["status"] = "failed_missing_metrics_epoch"
        return row
    by_epoch: Dict[int, Dict[str, float]] = {}
    for metric_row in _metric_rows(metrics_path):
        try:
            epoch = int(float(metric_row.get("epoch", "")))
        except (TypeError, ValueError):
            continue
        split = str(metric_row.get("split", ""))
        namespace = str(metric_row.get("namespace", ""))
        metric = str(metric_row.get("metric", ""))
        value = _finite_float(metric_row.get("value"))
        if value is None:
            continue
        if split == "test_gzsl" and namespace == "classification" and metric in {
            "gzsl_seen",
            "gzsl_unseen",
            "gzsl_h",
        }:
            by_epoch.setdefault(epoch, {})[metric] = value
        elif split == "test_gzsl" and namespace == "calibration_profile" and metric == "ausuc":
            by_epoch.setdefault(epoch, {})["ausuc"] = value
        elif namespace == "train_debug" and metric == "effective_logit_scale":
            by_epoch.setdefault(epoch, {})["effective_logit_scale"] = value
    complete = {
        epoch: values
        for epoch, values in by_epoch.items()
        if {"gzsl_seen", "gzsl_unseen", "gzsl_h", "ausuc"}.issubset(values)
    }
    if not complete:
        row["status"] = "failed_missing_final_task_metrics"
        return row
    final_epoch = max(complete)
    values = complete[final_epoch]
    row.update(
        {
            "final_epoch": final_epoch,
            "gzsl_seen": values["gzsl_seen"],
            "gzsl_unseen": values["gzsl_unseen"],
            "gzsl_h": values["gzsl_h"],
            "ausuc": values["ausuc"],
            "final_effective_logit_scale": values.get("effective_logit_scale"),
            "run_dir": str(runtime_path.parent),
            "status": "completed_existing" if existing else "completed",
            "returncode": 0,
        }
    )
    if final_epoch != job.total_epoch:
        row["status"] = "failed_final_epoch_mismatch"
    return row


def _run_job(job: Job, python_bin: str, config_file: Path, gpu: str) -> Dict[str, object]:
    command = _command(python_bin, config_file, job)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
    row = _read_result(job, existing=False)
    row["returncode"] = process.returncode
    row["duration_seconds"] = round(time.time() - started, 3)
    if process.returncode != 0:
        row["status"] = "failed"
    return row


def _progress_trial(job: Job, index: int) -> Dict[str, object]:
    return {
        "trial_index": int(index),
        "trial_name": job.trial_name,
        "runner": "train",
        "stage": job.stage,
        "combo": {
            "num_tokens": job.num_tokens,
            "base_lr": job.base_lr,
            "total_epoch": job.total_epoch,
            "seed": job.seed,
            "scale_policy": job.scale.name,
        },
        "overrides": {
            "SOLVER.TOTAL_EPOCH": job.total_epoch,
            "MODEL.PROMPT.NUM_TOKENS": job.num_tokens,
            "MODEL.R_SIMILARITY.SCORE_MODE": "cosine",
            "MODEL.R_SIMILARITY.LEARNABLE_SCALE": job.scale.learnable,
            "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE": job.scale.fixed_scale,
        },
        "output_dir": str(job.output_root),
        "identity_fields": {"seed": job.seed, "scale_policy": job.scale.name},
        "eta_fields": {"num_tokens": job.num_tokens},
        "eta_compatibility_keys": ["runner", "stage", "nproc", "total_epochs"],
    }


def _row_ok(row: Dict[str, object]) -> bool:
    return str(row.get("status", "")).startswith("completed") and all(
        _finite_float(row.get(key)) is not None
        for key in ("gzsl_seen", "gzsl_unseen", "gzsl_h", "ausuc")
    )


def _rank_rows(
    rows: Sequence[Dict[str, object]],
    *,
    anchor_ausuc: Optional[float],
    ausuc_tolerance: float,
) -> List[Dict[str, object]]:
    candidates = [dict(row) for row in rows if _row_ok(row)]
    for row in candidates:
        ausuc = float(row["ausuc"])
        row["ausuc_guard_pass"] = (
            anchor_ausuc is None or ausuc >= float(anchor_ausuc) - float(ausuc_tolerance)
        )
    guarded = [row for row in candidates if bool(row["ausuc_guard_pass"])]
    pool = guarded if guarded else candidates
    return sorted(
        pool,
        key=lambda row: (
            -float(row["gzsl_h"]),
            -float(row["ausuc"]),
            -float(row["gzsl_unseen"]),
            str(row["trial_name"]),
        ),
    )


def _write_rows(out_root: Path, rows: Sequence[Dict[str, object]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("stage", "")),
            int(row.get("seed", -1)),
            str(row.get("trial_name", "")),
        ),
    )
    fields: List[str] = []
    for row in ordered:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    (out_root / "summary.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _run_stage(
    jobs: Sequence[Job],
    *,
    python_bin: str,
    config_file: Path,
    out_root: Path,
    gpu_groups: Sequence[str],
    max_workers: int,
    progress_interval: float,
    all_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    pending: List[Job] = []
    rows: List[Dict[str, object]] = []
    for job in jobs:
        if _runtime_summary(job.output_root) is not None:
            rows.append(_read_result(job, existing=True))
        elif _has_contents(job.output_root):
            raise RuntimeError(
                "Refusing to overwrite incomplete output {}. Move it or use a new --out-root.".format(
                    job.output_root
                )
            )
        else:
            pending.append(job)
    if pending:
        trials = [_progress_trial(job, index) for index, job in enumerate(jobs)]
        trial_by_job = {job: trial for job, trial in zip(jobs, trials)}
        new_rows = run_parallel_trials(
            all_trials=trials,
            pending_items=pending,
            pending_trials=[trial_by_job[job] for job in pending],
            out_root=out_root,
            gpu_groups=gpu_groups,
            max_workers=min(max_workers, len(gpu_groups), len(pending)),
            initial_completed=len(rows),
            progress_interval=progress_interval,
            progress_width=120,
            progress_enabled=True,
            run_item=lambda job, gpu: _run_job(job, python_bin, config_file, gpu),
            item_name=lambda job: "{} {}".format(job.stage, job.trial_name),
            on_result=lambda partial: _write_rows(
                out_root, all_rows + rows + list(partial)
            ),
        )
        rows.extend(new_rows)
    if any(not _row_ok(row) for row in rows):
        _write_rows(out_root, all_rows + rows)
        raise RuntimeError("Stage {} contains failed trials.".format(jobs[0].stage if jobs else "unknown"))
    all_rows.extend(rows)
    _write_rows(out_root, all_rows)
    return rows


def _unique_configs(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    result = []
    seen = set()
    for row in rows:
        key = (
            int(row["num_tokens"]),
            float(row["base_lr"]),
            int(row["total_epoch"]),
            str(row["scale_policy"]),
        )
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _scale_from_row(row: Dict[str, object]) -> ScaleSpec:
    return SCALE_SPECS[str(row["scale_policy"])]


def _aggregate_shortlist(
    rows: Sequence[Dict[str, object]], shortlist: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    summaries = []
    for candidate in shortlist:
        key = (
            int(candidate["num_tokens"]),
            float(candidate["base_lr"]),
            int(candidate["total_epoch"]),
            str(candidate["scale_policy"]),
        )
        matched = [
            row
            for row in rows
            if _row_ok(row)
            and (
                int(row["num_tokens"]),
                float(row["base_lr"]),
                int(row["total_epoch"]),
                str(row["scale_policy"]),
            )
            == key
        ]
        by_seed = {int(row["seed"]): row for row in matched}
        metrics = {}
        for metric in ("gzsl_seen", "gzsl_unseen", "gzsl_h", "ausuc"):
            values = [float(by_seed[seed][metric]) for seed in sorted(by_seed)]
            metrics[metric] = {
                "mean": mean(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "values": values,
            }
        summaries.append(
            {
                "num_tokens": key[0],
                "base_lr": key[1],
                "total_epoch": key[2],
                "scale_policy": key[3],
                "training_seeds": sorted(by_seed),
                "three_seed_complete": sorted(by_seed) == [0, 1, 2],
                "metrics": metrics,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -(item["metrics"]["gzsl_h"]["mean"] or float("-inf")),
            -(item["metrics"]["ausuc"]["mean"] or float("-inf")),
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged A2-Cosine baseline adaptation without heavy Probe monitoring."
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu-groups", default="0;1")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent single-GPU jobs assigned to each listed card.",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--token-grid", default="8,16,32")
    parser.add_argument("--lr-grid", default="3e-4,6e-4,1e-3")
    parser.add_argument("--epoch-grid", default="10,15,20")
    parser.add_argument("--anchor-seeds", default="0,1,2")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--ausuc-tolerance", type=float, default=0.01)
    parser.add_argument("--progress-interval", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        token_grid = _parse_ints(args.token_grid, "--token-grid")
        lr_grid = _parse_floats(args.lr_grid, "--lr-grid")
        epoch_grid = _parse_ints(args.epoch_grid, "--epoch-grid")
        anchor_seeds = _parse_ints(args.anchor_seeds, "--anchor-seeds", allow_zero=True)
        physical_gpu_groups = _parse_gpu_groups(args.gpu_groups)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if sorted(anchor_seeds) != [0, 1, 2]:
        raise SystemExit("--anchor-seeds must be exactly 0,1,2 for the formal anchor contract.")
    if args.search_seed != 0:
        raise SystemExit("--search-seed must remain 0 so shortlist supplementation is unambiguous.")
    if args.shortlist_size <= 0 or args.shortlist_size > 3:
        raise SystemExit("--shortlist-size must be between 1 and 3.")
    if args.workers_per_gpu <= 0:
        raise SystemExit("--workers-per-gpu must be positive.")
    gpu_groups = [
        gpu
        for gpu in physical_gpu_groups
        for _ in range(int(args.workers_per_gpu))
    ]
    if args.max_workers <= 0 or args.max_workers > len(gpu_groups):
        raise SystemExit(
            "--max-workers must be positive and no larger than GPU count times workers-per-gpu."
        )
    if args.ausuc_tolerance < 0.0 or args.progress_interval <= 0.0:
        raise SystemExit("Tolerance must be non-negative and progress interval must be positive.")
    if BASE_NUM_TOKENS not in token_grid or BASE_LR not in lr_grid or BASE_EPOCHS not in epoch_grid:
        raise SystemExit("The search grids must retain the historical 16-token, lr=6e-4, 15-epoch anchor.")
    config_file = args.config_file.expanduser().resolve()
    if not config_file.is_file():
        raise SystemExit("Missing config file: {}".format(config_file))
    out_root = args.out_root.expanduser().resolve()

    plan = {
        "evidence_role": "development_hyperparameter_search",
        "formal_test_reused_for_selection": True,
        "heavy_monitoring_enabled": False,
        "fixed_probe_enabled": False,
        "anchor": {
            "num_tokens": BASE_NUM_TOKENS,
            "base_lr": BASE_LR,
            "total_epoch": BASE_EPOCHS,
            "scale_policy": "fixed1",
            "training_seeds": anchor_seeds,
        },
        "scale_screen": list(SCALE_SPECS),
        "token_grid": token_grid,
        "lr_grid": lr_grid,
        "epoch_grid": epoch_grid,
        "shortlist_size": args.shortlist_size,
        "shortlist_training_seeds": [0, 1, 2],
        "physical_gpu_groups": physical_gpu_groups,
        "workers_per_gpu": args.workers_per_gpu,
        "worker_gpu_slots": gpu_groups,
        "selection_order": "final_gzsl_h_desc_then_ausuc_desc_then_unseen_desc",
        "ausuc_guard": "candidate_ausuc >= fixed1_seed0_ausuc - tolerance",
        "ausuc_tolerance": args.ausuc_tolerance,
        "full_monitoring_deferred_until_winner_selected": True,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Planned maximum training jobs: 25 (3 anchor + 2 scale + 8 token/lr + 6 epoch + 6 shortlist).")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "search_contract.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_rows: List[Dict[str, object]] = []

    anchor_jobs = [
        _job(
            out_root,
            "01_anchor",
            BASE_NUM_TOKENS,
            BASE_LR,
            BASE_EPOCHS,
            seed,
            SCALE_SPECS["fixed1"],
        )
        for seed in anchor_seeds
    ]
    anchor_rows = _run_stage(
        anchor_jobs,
        python_bin=args.python_bin,
        config_file=config_file,
        out_root=out_root,
        gpu_groups=gpu_groups,
        max_workers=args.max_workers,
        progress_interval=args.progress_interval,
        all_rows=all_rows,
    )
    anchor_seed0 = next(row for row in anchor_rows if int(row["seed"]) == args.search_seed)
    anchor_ausuc = float(anchor_seed0["ausuc"])

    scale_jobs = [
        _job(
            out_root,
            "02_scale_screen",
            BASE_NUM_TOKENS,
            BASE_LR,
            BASE_EPOCHS,
            args.search_seed,
            SCALE_SPECS[name],
        )
        for name in ("fixed10", "learnable10")
    ]
    scale_rows = _run_stage(
        scale_jobs,
        python_bin=args.python_bin,
        config_file=config_file,
        out_root=out_root,
        gpu_groups=gpu_groups,
        max_workers=args.max_workers,
        progress_interval=args.progress_interval,
        all_rows=all_rows,
    )
    scale_rank = _rank_rows(
        [anchor_seed0] + scale_rows,
        anchor_ausuc=anchor_ausuc,
        ausuc_tolerance=args.ausuc_tolerance,
    )
    if not scale_rank:
        raise SystemExit("No valid scale policy completed.")
    scale_winner = scale_rank[0]
    selected_scale = _scale_from_row(scale_winner)

    coarse_jobs = []
    for num_tokens in token_grid:
        for base_lr in lr_grid:
            if num_tokens == BASE_NUM_TOKENS and math.isclose(base_lr, BASE_LR):
                continue
            coarse_jobs.append(
                _job(
                    out_root,
                    "03_token_lr",
                    num_tokens,
                    base_lr,
                    BASE_EPOCHS,
                    args.search_seed,
                    selected_scale,
                )
            )
    coarse_rows = _run_stage(
        coarse_jobs,
        python_bin=args.python_bin,
        config_file=config_file,
        out_root=out_root,
        gpu_groups=gpu_groups,
        max_workers=args.max_workers,
        progress_interval=args.progress_interval,
        all_rows=all_rows,
    )
    coarse_rank = _rank_rows(
        _unique_configs([scale_winner] + coarse_rows),
        anchor_ausuc=anchor_ausuc,
        ausuc_tolerance=args.ausuc_tolerance,
    )
    top_pairs = coarse_rank[: min(3, len(coarse_rank))]

    refine_jobs = []
    for row in top_pairs:
        for total_epoch in epoch_grid:
            if total_epoch == BASE_EPOCHS:
                continue
            refine_jobs.append(
                _job(
                    out_root,
                    "04_epoch_refine",
                    int(row["num_tokens"]),
                    float(row["base_lr"]),
                    total_epoch,
                    args.search_seed,
                    selected_scale,
                )
            )
    refine_rows = _run_stage(
        refine_jobs,
        python_bin=args.python_bin,
        config_file=config_file,
        out_root=out_root,
        gpu_groups=gpu_groups,
        max_workers=args.max_workers,
        progress_interval=args.progress_interval,
        all_rows=all_rows,
    )
    candidate_rank = _rank_rows(
        _unique_configs([scale_winner] + coarse_rows + refine_rows),
        anchor_ausuc=anchor_ausuc,
        ausuc_tolerance=args.ausuc_tolerance,
    )
    shortlist = candidate_rank[: args.shortlist_size]
    if not shortlist:
        raise SystemExit("No valid configuration remained for three-seed confirmation.")

    existing_by_config_seed = {
        (
            int(row["num_tokens"]),
            float(row["base_lr"]),
            int(row["total_epoch"]),
            str(row["scale_policy"]),
            int(row["seed"]),
        )
        for row in all_rows
        if _row_ok(row)
    }
    shortlist_jobs = []
    for row in shortlist:
        scale = _scale_from_row(row)
        for seed in (1, 2):
            key = (
                int(row["num_tokens"]),
                float(row["base_lr"]),
                int(row["total_epoch"]),
                scale.name,
                seed,
            )
            if key in existing_by_config_seed:
                continue
            shortlist_jobs.append(
                _job(
                    out_root,
                    "05_shortlist",
                    key[0],
                    key[1],
                    key[2],
                    seed,
                    scale,
                )
            )
    if shortlist_jobs:
        _run_stage(
            shortlist_jobs,
            python_bin=args.python_bin,
            config_file=config_file,
            out_root=out_root,
            gpu_groups=gpu_groups,
            max_workers=args.max_workers,
            progress_interval=args.progress_interval,
            all_rows=all_rows,
        )

    aggregate = _aggregate_shortlist(all_rows, shortlist)
    final_payload = {
        "selection_is_development_evidence": True,
        "normal_test_environment_used_for_selection": True,
        "winner_requires_full_monitoring_retrain": True,
        "anchor_three_seed": _aggregate_shortlist(all_rows, [anchor_seed0])[0],
        "selected_scale_policy": selected_scale.name,
        "shortlist": aggregate,
    }
    (out_root / "final_shortlist.json").write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_payload, ensure_ascii=False, indent=2), flush=True)
    if not aggregate or not all(item["three_seed_complete"] for item in aggregate):
        raise SystemExit("Shortlist aggregation is incomplete; do not declare a winner.")


if __name__ == "__main__":
    main()
