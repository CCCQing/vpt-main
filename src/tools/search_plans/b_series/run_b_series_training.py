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
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials


CONFIG_ROOT = ROOT / "configs" / "b_series_experiments"
RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
GROUPS = {
    "E5": (
        ("E5-A2-resume-control", "E5-A2-resume-control.yaml"),
        ("E5-B1-freeze", "E5-B1-freeze.yaml"),
        ("E5-B2-freeze", "E5-B2-freeze.yaml"),
        ("E5-B1-joint-matched", "E5-B1-joint-matched.yaml"),
        ("E5-B2-joint-matched", "E5-B2-joint-matched.yaml"),
    ),
    "E6": (
        ("E6-B1-shared-control", "E5-B1-joint-matched.yaml"),
        ("E6-B2-shared-control", "E5-B2-joint-matched.yaml"),
        ("E6-B1-slot", "E6-B1-slot.yaml"),
        ("E6-B2-slot-control", "E6-B2-slot-control.yaml"),
    ),
    "E7-G1": (
        ("E7-G0-no-sample-gate", "E5-B1-joint-matched.yaml"),
        ("E7-G1-sample-shared", "E7-G1-sample-shared.yaml"),
        ("E7-G1-fixed-input-control", "E7-G1-fixed-input-control.yaml"),
        ("E7-G1-B2-gate-control", "E7-G1-B2-gate-control.yaml"),
    ),
    "E7-G2": (
        ("E7-G0-no-sample-gate", "E5-B1-joint-matched.yaml"),
        ("E7-G2-sample-grouped", "E7-G2-sample-grouped.yaml"),
        ("E7-G2-fixed-input-control", "E7-G2-fixed-input-control.yaml"),
        ("E7-G2-B2-gate-control", "E7-G2-B2-gate-control.yaml"),
    ),
    "E7-G3": (
        ("E7-G0-no-sample-gate", "E5-B1-joint-matched.yaml"),
        ("E7-G3-sample-layerwise", "E7-G3-sample-layerwise.yaml"),
        ("E7-G3-fixed-input-control", "E7-G3-fixed-input-control.yaml"),
        ("E7-G3-B2-gate-control", "E7-G3-B2-gate-control.yaml"),
    ),
}


@dataclass(frozen=True)
class Job:
    method: str
    group: str
    seed: int
    config_file: Path
    checkpoint: Path
    output_root: Path
    log_path: Path
    extra_overrides: tuple[str, ...]


def _comma_values(raw: str) -> List[str]:
    values = [value.strip().upper() for value in str(raw).split(",") if value.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError("comma-separated values must be non-empty and unique")
    return values


def _seeds(raw: str) -> List[int]:
    values = [int(value) for value in _comma_values(raw)]
    if min(values) < 0:
        raise ValueError("training seeds must be non-negative")
    return values


def _gpu_groups(raw: str) -> List[str]:
    values = [value.strip() for value in str(raw).split(";") if value.strip()]
    if not values or len(set(values)) != len(values) or any("," in value for value in values):
        raise ValueError("GPU groups must be unique single cards separated by semicolons")
    return values


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_e7_preconditions(path: Path, requested_groups: Sequence[str]) -> Dict[str, object]:
    e7_requested = any(group.startswith("E7-") for group in requested_groups)
    if not e7_requested:
        return {"required": False, "pass": True}
    if not path.is_file():
        raise FileNotFoundError(
            "E7 requires --precondition-manifest with the three declared eligibility checks"
        )
    payload = _read_json(path)
    block = payload.get("E7", payload)
    checks = dict(block.get("checks") or {})
    required = {
        "bidirectional_nontrivial_response",
        "predictable_without_test_leakage",
        "pseudo_unseen_better_than_constant",
    }
    missing = sorted(required.difference(checks))
    passed = bool(block.get("eligible", False)) and not missing and all(
        checks[name] is True for name in required
    )
    raw_evidence_paths = list(block.get("evidence_paths") or ())
    evidence_paths = [
        (path.parent / str(value)).resolve()
        if not Path(str(value)).is_absolute()
        else Path(str(value)).resolve()
        for value in raw_evidence_paths
    ]
    missing_evidence = [str(value) for value in evidence_paths if not value.is_file()]
    passed = passed and bool(evidence_paths) and not missing_evidence
    if not passed:
        raise ValueError(
            "E7 preconditions failed or are incomplete: missing_checks={} "
            "evidence_count={} missing_evidence={}".format(
                missing, len(evidence_paths), missing_evidence
            )
        )
    if "E7-G2" in requested_groups and block.get("max_gate_stage") not in {
        "G2",
        "G3",
    }:
        raise ValueError("E7-G2 requires precondition max_gate_stage G2 or G3")
    if "E7-G3" in requested_groups and block.get("max_gate_stage") != "G3":
        raise ValueError("E7-G3 requires precondition max_gate_stage G3")
    return {
        "required": True,
        "pass": True,
        "path": str(path),
        "evidence_paths": [str(value) for value in evidence_paths],
        "payload": payload,
    }


def _checkpoint(a2_root: Path, seed: int) -> Path:
    candidates = [
        a2_root / "A2" / "seed{}".format(seed) / RUN_SUFFIX / "model_final_trainable.pth",
        a2_root / "seed{}".format(seed) / RUN_SUFFIX / "model_final_trainable.pth",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            "expected one A2 seed {} checkpoint; checked {}".format(
                seed, ", ".join(str(path) for path in candidates)
            )
        )
    return existing[0]


def _completed(output_root: Path) -> bool:
    paths = list(output_root.rglob("monitor_runtime_summary.json")) if output_root.is_dir() else []
    completed = []
    for path in paths:
        try:
            if str(_read_json(path).get("status", "")).lower() == "completed":
                completed.append(path)
        except (OSError, ValueError):
            continue
    if len(completed) > 1:
        raise RuntimeError("multiple completed runs found under {}".format(output_root))
    return bool(completed)


def _has_contents(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _build_jobs(
    groups: Sequence[str],
    seeds: Sequence[int],
    a2_root: Path,
    out_root: Path,
    e7_base_mode: str,
) -> List[Job]:
    jobs = []
    seen = set()
    for group in groups:
        for method, config_name in GROUPS[group]:
            for seed in seeds:
                identity = (method, int(seed))
                if identity in seen:
                    continue
                seen.add(identity)
                overrides: List[str] = []
                if group.startswith("E7-") and str(e7_base_mode) == "slot":
                    overrides.extend(
                        [
                            "MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.CONTENT_MODE",
                            "slot_low_rank",
                            "MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.SLOT_RANK",
                            "8",
                        ]
                    )
                jobs.append(
                    Job(
                        method=method,
                        group=group,
                        seed=int(seed),
                        config_file=CONFIG_ROOT / config_name,
                        checkpoint=_checkpoint(a2_root, int(seed)),
                        output_root=out_root / method / "seed{}".format(seed),
                        log_path=out_root / "launcher_logs" / "{}_seed{}.log".format(
                            method, seed
                        ),
                        extra_overrides=tuple(overrides),
                    )
                )
    return jobs


def _command(job: Job, python_bin: str) -> List[str]:
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
        "SOLVER.INIT_TRAINABLE_CHECKPOINT",
        str(job.checkpoint),
        "MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.ARCHITECTURE_ID",
        str(job.method),
        *job.extra_overrides,
    ]


def _format(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _run_job(job: Job, python_bin: str, gpu: str) -> Dict[str, object]:
    command = _command(job, python_bin)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["VPT_SEARCH_PROGRESS_PATH"] = str(
        (job.output_root / "progress.json").resolve()
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
    completed = process.returncode == 0 and _completed(job.output_root)
    return {
        "method": job.method,
        "group": job.group,
        "seed": job.seed,
        "gpu": gpu,
        "status": "completed" if completed else "failed",
        "returncode": int(process.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "output_root": str(job.output_root),
        "log_path": str(job.log_path),
        "initialization_checkpoint": str(job.checkpoint),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E5/E6/E7 training suite.")
    parser.add_argument("--groups", default="E5,E6")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--a2-root", required=True, type=Path)
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "output" / "b_series_experiments" / "training"
    )
    parser.add_argument("--gpu-groups", default="0;1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--precondition-manifest", type=Path, default=Path(""))
    parser.add_argument("--e7-base-mode", choices=("shared", "slot"), default="shared")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    groups = _comma_values(args.groups)
    unknown = [group for group in groups if group not in GROUPS]
    if unknown:
        raise SystemExit("unknown B1 training group(s): {}".format(",".join(unknown)))
    seeds = _seeds(args.seeds)
    gpus = _gpu_groups(args.gpu_groups)
    if args.max_workers <= 0 or args.max_workers > len(gpus):
        raise SystemExit("--max-workers must be positive and no larger than GPU count")
    preconditions = _validate_e7_preconditions(
        args.precondition_manifest.resolve(), groups
    )
    a2_root = args.a2_root.resolve()
    out_root = args.out_root.resolve()
    jobs = _build_jobs(groups, seeds, a2_root, out_root, args.e7_base_mode)
    missing_configs = sorted(
        {str(job.config_file) for job in jobs if not job.config_file.is_file()}
    )
    if missing_configs:
        raise FileNotFoundError(", ".join(missing_configs))
    pending = []
    skipped = []
    for job in jobs:
        if _completed(job.output_root) and not args.no_resume:
            skipped.append(job)
        elif _has_contents(job.output_root):
            raise RuntimeError(
                "refusing to overwrite incomplete output {}".format(job.output_root)
            )
        else:
            pending.append(job)
    print(
        "B1 training plan groups={} seeds={} total={} pending={} skipped={}".format(
            ",".join(groups),
            ",".join(str(seed) for seed in seeds),
            len(jobs),
            len(pending),
            len(skipped),
        ),
        flush=True,
    )
    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % min(args.max_workers, len(gpus))]
            print("[dry-run] gpu={} {}".format(gpu, _format(_command(job, args.python_bin))))
        return
    trials = [
        {
            "trial_index": index,
            "trial_name": "{}_seed{}".format(job.method, job.seed),
            "runner": "train",
            "stage": job.group,
            "combo": {"method": job.method, "seed": job.seed},
            "overrides": {"initialization_checkpoint": str(job.checkpoint)},
            "output_dir": str(job.output_root),
            "identity_fields": {"config_file": str(job.config_file)},
            "eta_fields": {"method": job.method},
            "eta_compatibility_keys": ["runner", "stage", "nproc", "total_epochs"],
        }
        for index, job in enumerate(jobs)
    ]
    by_job = {job: trial for job, trial in zip(jobs, trials)}
    results = (
        run_parallel_trials(
            all_trials=trials,
            pending_items=pending,
            pending_trials=[by_job[job] for job in pending],
            out_root=out_root,
            gpu_groups=gpus,
            max_workers=int(args.max_workers),
            initial_completed=len(skipped),
            progress_interval=2.0,
            progress_width=120,
            progress_enabled=not args.no_progress,
            run_item=lambda job, gpu: _run_job(job, args.python_bin, gpu),
            item_name=lambda job: "{} seed={}".format(job.method, job.seed),
        )
        if pending
        else []
    )
    failed = [result for result in results if result["status"] != "completed"]
    summary = {
        "format": "b_series_experiments_training_launcher_v1",
        "suite_name": "B-series",
        "groups": groups,
        "seeds": seeds,
        "e7_base_mode": args.e7_base_mode,
        "preconditions": preconditions,
        "a2_root": str(a2_root),
        "results": results,
        "skipped_existing": [
            {"method": job.method, "seed": job.seed} for job in skipped
        ],
        "status": "failed" if failed else "completed",
    }
    _path = out_root / "b_series_training_launcher_summary.json"
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
