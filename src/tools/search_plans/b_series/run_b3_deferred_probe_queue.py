#!/usr/bin/env python3
"""Resume B3 training with checkpoint-only, independently scheduled Probes.

The scientific unit is complete only after one training checkpoint and all
three declared final_full Probe selections pass their own replay validators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import yaml


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials
from src.tools.search_plans.b_series.run_b3_series_training import (
    RUN_SUFFIX,
    STAGE_ORDER,
    _build_jobs,
    _command,
    _comma_values,
    _gpu_groups,
    _load_splits,
    _pilot_weights,
    _ratios,
)


STRICT_PROBE_SEEDS = (424242, 424243, 424244)


@dataclass(frozen=True)
class ProbeJob:
    source_run: Path
    output_run: Path
    split: str
    method: str
    training_seed: int
    selection_seed: int
    log_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _actual_run(output_root: Path) -> Path:
    return output_root / RUN_SUFFIX


def _integrated_complete(run_dir: Path) -> bool:
    summary = run_dir / "monitor_runtime_summary.json"
    robustness = run_dir / "diagnostics" / "probe_robustness_manifest.json"
    if not summary.is_file() or not robustness.is_file():
        return False
    try:
        if str(_read_json(summary).get("status", "")).lower() != "completed":
            return False
        payload = _read_json(robustness)
        seeds = tuple(int(item) for item in payload.get("selection_seeds", []))
        executions = list(payload.get("executions") or [])
        return (
            seeds == STRICT_PROBE_SEEDS
            and len(executions) == len(STRICT_PROBE_SEEDS)
            and all(
                isinstance(item, dict)
                and str(item.get("execution_profile")) == "final_full"
                for item in executions
            )
        )
    except (OSError, ValueError, TypeError):
        return False


def _collection_complete(run_dir: Path) -> bool:
    path = run_dir / "b3_fixed_probe_collection.json"
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        return bool(payload.get("valid", False)) and tuple(
            int(item) for item in payload.get("selection_seeds", [])
        ) == STRICT_PROBE_SEEDS
    except (OSError, ValueError, TypeError):
        return False


def _marker_valid(run_dir: Path) -> bool:
    marker_path = run_dir / "training_checkpoint_ready.json"
    if not marker_path.is_file():
        return False
    try:
        marker = _read_json(marker_path)
        checkpoint = dict(marker.get("checkpoint") or {})
        checkpoint_path = run_dir / str(checkpoint.get("checkpoint_path", ""))
        return (
            marker.get("format") == "deferred_fixed_probe_checkpoint_ready_v1"
            and marker.get("status") == "checkpoint_ready"
            and checkpoint_path.is_file()
            and _sha256(checkpoint_path) == checkpoint.get("checkpoint_sha256")
            and tuple(
                int(item)
                for item in marker.get("fixed_probe_contract", {}).get(
                    "selection_seeds", []
                )
            )
            == STRICT_PROBE_SEEDS
        )
    except (OSError, ValueError, TypeError):
        return False


def _file_identity(run_dir: Path, name: str):
    path = run_dir / name
    return {
        "path": name,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _recover_training_marker(run_dir: Path) -> bool:
    """Recover only when the final checkpoint proves training already finished."""
    if _marker_valid(run_dir):
        return True
    checkpoint_path = run_dir / "model_final_trainable.pth"
    config_path = run_dir / "resolved_config.yaml"
    if not checkpoint_path.is_file() or not config_path.is_file():
        return False
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    total_epoch = int(config["SOLVER"]["TOTAL_EPOCH"])
    if (
        checkpoint.get("format") != "vpt_trainable_v1"
        or int(checkpoint.get("total_epoch", -1)) != total_epoch
        or str(checkpoint.get("protocol_mode", "")).lower()
        != "b3_pseudo_gzsl"
    ):
        raise RuntimeError(
            "refusing to recover marker from an incompatible checkpoint: {}".format(
                checkpoint_path
            )
        )
    monitor_summary = (
        _read_json(run_dir / "monitor_runtime_summary.json")
        if (run_dir / "monitor_runtime_summary.json").is_file()
        else {}
    )
    diagnostic_manifest = (
        _read_json(run_dir / "diagnostics" / "diagnostic_manifest.json")
        if (run_dir / "diagnostics" / "diagnostic_manifest.json").is_file()
        else {}
    )
    primary_runtime_path = run_dir / "diagnostics" / "probe_runtime_summary.json"
    primary_runtime = (
        _read_json(primary_runtime_path) if primary_runtime_path.is_file() else {}
    )
    recorded_checkpoint = dict(primary_runtime.get("checkpoint") or {})
    probe_cfg = config["MONITOR"]["PROBE"]
    selection_seeds = [int(probe_cfg["SELECTION_SEED"])]
    selection_seeds.extend(
        int(item) for item in probe_cfg.get("ROBUSTNESS_SELECTION_SEEDS", [])
    )
    selection_seeds = list(dict.fromkeys(selection_seeds))
    if tuple(selection_seeds) != STRICT_PROBE_SEEDS:
        raise RuntimeError("recovered B3 run does not declare strict three Probes")
    run_id = str(
        monitor_summary.get("run_id")
        or diagnostic_manifest.get("run_id")
        or run_dir.name
    )
    session_id = str(
        monitor_summary.get("session_id")
        or diagnostic_manifest.get("session_id")
        or "recovered-{}".format(_sha256(checkpoint_path)[:16])
    )
    marker = {
        "format": "deferred_fixed_probe_checkpoint_ready_v1",
        "status": "checkpoint_ready",
        "training_performed": True,
        "fixed_probe_performed": False,
        "created_at": _utc_now(),
        "recovered_after_interrupted_integrated_probe": True,
        "run_id": run_id,
        "session_id": session_id,
        "training_seed": int(checkpoint.get("seed", config.get("SEED", 0))),
        "protocol_mode": "b3_pseudo_gzsl",
        "b3_pseudo_manifest": checkpoint.get("b3_pseudo_manifest"),
        "checkpoint": {
            "checkpoint_id": "final_epoch_{:04d}".format(total_epoch),
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": total_epoch,
            "checkpoint_global_step": int(
                recorded_checkpoint.get("checkpoint_global_step", 0)
            ),
            "checkpoint_selection_rule": "predeclared_final_epoch",
            "source_run_id": run_id,
            "source_session_id": session_id,
        },
        "fixed_probe_contract": {
            "execution_profile": "final_full",
            "selection_seeds": selection_seeds,
            "probe_loader": {
                "batch_size": int(probe_cfg["BATCH_SIZE"]),
                "num_workers": 0,
                "pin_memory": False,
                "cache_transformed_images": True,
                "cache_vit_cls_prepass": True,
            },
            "cache_contract": {
                "transformed_images": True,
                "vit_cls_prepass": True,
                "prompt_conditioned_features_cached": False,
                "attention_or_logits_cached": False,
            },
        },
        "identity_files": {
            "resolved_config": _file_identity(run_dir, "resolved_config.yaml"),
            "dataset_manifest": _file_identity(run_dir, "dataset_manifest.json"),
            "reproducibility_manifest": _file_identity(
                run_dir, "reproducibility_manifest.json"
            ),
        },
    }
    _atomic_json(run_dir / "training_checkpoint_ready.json", marker)
    if (run_dir / "diagnostics").is_dir() and not _integrated_complete(run_dir):
        _atomic_json(
            run_dir / "partial_integrated_probe_exclusion.json",
            {
                "format": "partial_integrated_probe_exclusion_v1",
                "status": "excluded_from_scientific_completion",
                "created_at": _utc_now(),
                "reason": "integrated strict-three-Probe execution was interrupted",
                "preserved_path": "diagnostics",
                "analysis_rule": (
                    "use only validated replay roots referenced by "
                    "b3_fixed_probe_collection.json"
                ),
                "checkpoint_sha256": marker["checkpoint"]["checkpoint_sha256"],
            },
        )
    return True


def _format(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def _run_training(job, python_bin: str, protocol: str, gpu: str):
    command = _command(job, python_bin, protocol)
    command.extend(
        [
            "MONITOR.PROBE.FINAL_EXECUTION_MODE",
            "deferred",
            "MONITOR.PROBE.CACHE_TRANSFORMED_IMAGES",
            "true",
            "MONITOR.PROBE.CACHE_VIT_CLS_PREPASS",
            "true",
        ]
    )
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
    ready = process.returncode == 0 and _marker_valid(_actual_run(job.output_root))
    return {
        "kind": "training",
        "split": job.split.name,
        "method": job.method,
        "training_seed": job.seed,
        "gpu": str(gpu),
        "status": "completed" if ready else "failed",
        "returncode": int(process.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "source_run": str(_actual_run(job.output_root)),
        "log_path": str(job.log_path),
    }


def _replay_valid(output_run: Path) -> bool:
    path = output_run / "probe_robustness_replay_summary.json"
    try:
        return path.is_file() and bool(_read_json(path).get("valid", False))
    except (OSError, ValueError, TypeError):
        return False


def _run_probe(job: ProbeJob, python_bin: str, gpu: str, cpu_threads: int):
    replay = ROOT / "src" / "tools" / "search_plans" / "a_series" / "replay_probe_robustness.py"
    command = [
        python_bin,
        "-u",
        str(replay),
        "--source-run",
        str(job.source_run),
        "--output-run",
        str(job.output_run),
        "--selection-seed",
        str(job.selection_seed),
        "--source-kind",
        "training_checkpoint",
        "--execution-profile",
        "final_full",
        "--cpu-threads",
        str(cpu_threads),
        "--cache-transformed-images",
        "--cache-vit-cls-prepass",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = str(cpu_threads)
    environment["MKL_NUM_THREADS"] = str(cpu_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    environment["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    environment["VPT_SEARCH_PROGRESS_PATH"] = str(
        (job.output_run / "progress.json").resolve()
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
    valid = process.returncode == 0 and _replay_valid(job.output_run)
    return {
        "kind": "fixed_probe",
        "split": job.split,
        "method": job.method,
        "training_seed": job.training_seed,
        "selection_seed": job.selection_seed,
        "gpu": str(gpu),
        "status": "completed" if valid else "failed",
        "returncode": int(process.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "source_run": str(job.source_run),
        "output_run": str(job.output_run),
        "log_path": str(job.log_path),
    }


def _probe_output_root(out_root: Path, job, selection_seed: int) -> Path:
    return (
        out_root
        / "_deferred_fixed_probes"
        / job.split.name
        / job.method
        / "seed{}".format(job.seed)
        / "selection_seed_{}".format(selection_seed)
        / RUN_SUFFIX
    )


def _quarantine_incomplete_replay(output_run: Path) -> Path:
    resolved = output_run.resolve()
    deferred_roots = [
        parent for parent in resolved.parents if parent.name == "_deferred_fixed_probes"
    ]
    if len(deferred_roots) != 1:
        raise RuntimeError(
            "refusing to quarantine replay outside _deferred_fixed_probes: {}".format(
                resolved
            )
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output_run.with_name(
        "{}.failed_{}_{}".format(output_run.name, timestamp, os.getpid())
    )
    output_run.rename(target)
    return target


def _build_probe_jobs(
    out_root: Path, jobs, *, quarantine_incomplete: bool = True
) -> List[ProbeJob]:
    result = []
    log_root = out_root / "launcher_logs" / "deferred_fixed_probes"
    for job in jobs:
        source_run = _actual_run(job.output_root)
        if _integrated_complete(source_run) or _collection_complete(source_run):
            continue
        if not _marker_valid(source_run):
            raise RuntimeError("training checkpoint is not ready: {}".format(source_run))
        for selection_seed in STRICT_PROBE_SEEDS:
            output_run = _probe_output_root(out_root, job, selection_seed)
            if _replay_valid(output_run):
                continue
            if output_run.is_dir() and next(output_run.iterdir(), None) is not None:
                if not quarantine_incomplete:
                    raise RuntimeError(
                        "incomplete replay requires quarantine before resume: {}".format(
                            output_run
                        )
                    )
                quarantined = _quarantine_incomplete_replay(output_run)
                print(
                    "Quarantined incomplete replay {} -> {}".format(
                        output_run, quarantined
                    ),
                    flush=True,
                )
            result.append(
                ProbeJob(
                    source_run=source_run,
                    output_run=output_run,
                    split=job.split.name,
                    method=job.method,
                    training_seed=job.seed,
                    selection_seed=selection_seed,
                    log_path=log_root
                    / job.split.name
                    / "{}_seed{}_probe{}.log".format(
                        job.method, job.seed, selection_seed
                    ),
                )
            )
    return result


def _collect_one(out_root: Path, job) -> bool:
    run_dir = _actual_run(job.output_root)
    if _integrated_complete(run_dir):
        return True
    marker = _read_json(run_dir / "training_checkpoint_ready.json")
    checkpoint_sha = marker["checkpoint"]["checkpoint_sha256"]
    records = []
    for selection_seed in STRICT_PROBE_SEEDS:
        output_run = _probe_output_root(out_root, job, selection_seed)
        summary_path = output_run / "probe_robustness_replay_summary.json"
        if not _replay_valid(output_run):
            return False
        summary = _read_json(summary_path)
        observed_sha = summary.get("source", {}).get("checkpoint", {}).get(
            "checkpoint_sha256"
        )
        if observed_sha != checkpoint_sha:
            raise RuntimeError("Probe collection checkpoint identity mismatch")
        records.append(
            {
                "selection_seed": selection_seed,
                "execution_profile": summary.get("execution_profile"),
                "replay_root": os.path.relpath(str(output_run), str(run_dir)),
                "replay_summary": os.path.relpath(str(summary_path), str(run_dir)),
                "valid": True,
                "checkpoint_sha256": observed_sha,
                "probe_manifest_checks": summary.get("probe_manifest_checks"),
            }
        )
    payload = {
        "format": "b3_deferred_fixed_probe_collection_v1",
        "status": "valid",
        "valid": True,
        "created_at": _utc_now(),
        "training_run": str(run_dir),
        "training_checkpoint_sha256": checkpoint_sha,
        "selection_seeds": list(STRICT_PROBE_SEEDS),
        "execution_profile": "final_full",
        "training_performed_once": True,
        "probe_executions": records,
        "partial_integrated_probe_artifacts_excluded": (
            run_dir / "partial_integrated_probe_exclusion.json"
        ).is_file(),
        "scientific_completion_rule": (
            "one fixed training checkpoint plus three independently validated "
            "full Probe selections"
        ),
    }
    _atomic_json(run_dir / "b3_fixed_probe_collection.json", payload)
    return True


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run resumable B3 training and strict three-Probe checkpoint replays."
    )
    parser.add_argument("--stages", default="R1")
    parser.add_argument(
        "--protocol", choices=("b3_pseudo_gzsl", "final_gzsl"), default="b3_pseudo_gzsl"
    )
    parser.add_argument("--manifest-suite", type=Path, default=Path(""))
    parser.add_argument("--max-ratios", default="0.25,0.50")
    parser.add_argument("--r2-pilot-weights", default="")
    parser.add_argument("--formal-intra-weight", type=float)
    parser.add_argument("--formal-inter-weight", type=float)
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "output" / "b3_series" / "training"
    )
    parser.add_argument("--gpu-groups", default="0;1;2")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--probe-cpu-threads", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--equivalence-summary", type=Path, default=Path(""))
    parser.add_argument("--skip-equivalence-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages = _comma_values(args.stages)
    unknown = sorted(set(stages).difference(STAGE_ORDER))
    if unknown:
        raise SystemExit("unknown B3 stages: {}".format(",".join(unknown)))
    stages = [stage for stage in STAGE_ORDER if stage in stages]
    gpus = _gpu_groups(args.gpu_groups)
    if args.max_workers <= 0 or args.max_workers > len(gpus):
        raise SystemExit("--max-workers must be positive and no larger than GPU count")
    if int(args.probe_cpu_threads) < 1:
        raise SystemExit("--probe-cpu-threads must be positive")
    if not args.dry_run and not args.skip_equivalence_gate:
        equivalence_path = args.equivalence_summary.expanduser().resolve()
        if not equivalence_path.is_file() or not bool(
            _read_json(equivalence_path).get("valid", False)
        ):
            raise RuntimeError(
                "formal B3 resume requires a passing replay-equivalence summary"
            )
    suite_path = args.manifest_suite.expanduser().resolve()
    splits = _load_splits(args.protocol, suite_path)
    ratios = _ratios(args.max_ratios)
    pilot_weights = _pilot_weights(args.r2_pilot_weights)
    formal_weights = None
    if (args.formal_intra_weight is None) != (args.formal_inter_weight is None):
        raise SystemExit("formal intra/inter weights must be provided together")
    if args.formal_intra_weight is not None:
        formal_weights = (
            float(args.formal_intra_weight),
            float(args.formal_inter_weight),
        )
    out_root = args.out_root.expanduser().resolve()
    all_jobs = []
    for stage in stages:
        all_jobs.extend(
            _build_jobs(
                stage,
                splits,
                ratios,
                out_root,
                require_checkpoints=not args.dry_run,
                pilot_weights=pilot_weights,
                formal_weights=formal_weights,
            )
        )

    scientific_complete = []
    training_ready = []
    training_pending = []
    for job in all_jobs:
        run_dir = _actual_run(job.output_root)
        if _integrated_complete(run_dir) or _collection_complete(run_dir):
            scientific_complete.append(job)
        elif _marker_valid(run_dir):
            training_ready.append(job)
        elif args.dry_run and (run_dir / "model_final_trainable.pth").is_file():
            training_ready.append(job)
        elif not args.dry_run and _recover_training_marker(run_dir):
            training_ready.append(job)
        elif job.output_root.is_dir() and next(job.output_root.iterdir(), None) is not None:
            raise RuntimeError(
                "incomplete B3 training has no valid final checkpoint: {}".format(
                    job.output_root
                )
            )
        else:
            training_pending.append(job)

    print(
        "B3 decoupled total={} scientific_complete={} training_ready={} training_pending={}".format(
            len(all_jobs),
            len(scientific_complete),
            len(training_ready),
            len(training_pending),
        ),
        flush=True,
    )
    if args.dry_run:
        for index, job in enumerate(training_pending):
            command = _command(job, args.python_bin, args.protocol)
            command.extend(["MONITOR.PROBE.FINAL_EXECUTION_MODE", "deferred"])
            print("[train] gpu={} {}".format(gpus[index % len(gpus)], _format(command)))
        return

    summary_path = out_root / "b3_deferred_probe_queue_summary.json"
    summary = {
        "format": "b3_deferred_probe_queue_v1",
        "status": "running",
        "started_at": _utc_now(),
        "stages": stages,
        "gpu_groups": gpus,
        "max_workers": int(args.max_workers),
        "probe_cpu_threads": int(args.probe_cpu_threads),
        "equivalence_summary": str(args.equivalence_summary),
        "results": [],
    }
    _atomic_json(summary_path, summary)
    try:
        if training_pending:
            trials = [
                {
                    "trial_index": index,
                    "trial_name": "train_{}_{}_seed{}".format(
                        job.split.name, job.method, job.seed
                    ),
                    "runner": "b3_deferred_train",
                    "stage": job.stage,
                    "combo": {
                        "method": job.method,
                        "seed": job.seed,
                        "split": job.split.name,
                    },
                    "overrides": {"fixed_probe": "deferred"},
                    "output_dir": str(job.output_root),
                    "identity_fields": {
                        "config_file": str(job.config_file),
                        "manifest_sha256": job.split.sha256,
                    },
                    "eta_fields": {"method": job.method},
                    "eta_compatibility_keys": ["runner", "stage", "method"],
                }
                for index, job in enumerate(training_pending)
            ]
            by_job = {job: trial for job, trial in zip(training_pending, trials)}
            results = run_parallel_trials(
                all_trials=trials,
                pending_items=training_pending,
                pending_trials=[by_job[job] for job in training_pending],
                out_root=out_root,
                gpu_groups=gpus,
                max_workers=int(args.max_workers),
                initial_completed=len(scientific_complete) + len(training_ready),
                progress_interval=2.0,
                progress_width=120,
                progress_enabled=not args.no_progress,
                run_item=lambda job, gpu: _run_training(
                    job, args.python_bin, args.protocol, gpu
                ),
                item_name=lambda job: "train {} {} seed={}".format(
                    job.split.name, job.method, job.seed
                ),
            )
            summary["results"].extend(results)
            _atomic_json(summary_path, summary)
            if any(item["status"] != "completed" for item in results):
                raise RuntimeError("one or more deferred B3 training jobs failed")

        probe_jobs = _build_probe_jobs(out_root, all_jobs)
        print("B3 pending independent fixed Probes={}".format(len(probe_jobs)), flush=True)
        if probe_jobs:
            trials = [
                {
                    "trial_index": index,
                    "trial_name": "probe_{}_{}_seed{}_selection{}".format(
                        job.split,
                        job.method,
                        job.training_seed,
                        job.selection_seed,
                    ),
                    "runner": "b3_checkpoint_probe",
                    "stage": "fixed_probe",
                    "combo": {
                        "method": job.method,
                        "training_seed": job.training_seed,
                        "selection_seed": job.selection_seed,
                    },
                    "overrides": {"cpu_threads": int(args.probe_cpu_threads)},
                    "output_dir": str(job.output_run),
                    "identity_fields": {"source_run": str(job.source_run)},
                    "eta_fields": {"method": job.method},
                    "eta_compatibility_keys": ["runner", "stage", "method"],
                }
                for index, job in enumerate(probe_jobs)
            ]
            by_job = {job: trial for job, trial in zip(probe_jobs, trials)}
            results = run_parallel_trials(
                all_trials=trials,
                pending_items=probe_jobs,
                pending_trials=[by_job[job] for job in probe_jobs],
                out_root=out_root / "_deferred_fixed_probes",
                gpu_groups=gpus,
                max_workers=int(args.max_workers),
                initial_completed=0,
                progress_interval=5.0,
                progress_width=120,
                progress_enabled=not args.no_progress,
                run_item=lambda job, gpu: _run_probe(
                    job, args.python_bin, gpu, int(args.probe_cpu_threads)
                ),
                item_name=lambda job: "probe {} {} seed={} selection={}".format(
                    job.split,
                    job.method,
                    job.training_seed,
                    job.selection_seed,
                ),
            )
            summary["results"].extend(results)
            _atomic_json(summary_path, summary)
            if any(item["status"] != "completed" for item in results):
                raise RuntimeError("one or more independent B3 fixed Probes failed")

        incomplete = []
        for job in all_jobs:
            if not _collect_one(out_root, job):
                incomplete.append(
                    "{}:{}:seed{}".format(job.split.name, job.method, job.seed)
                )
        if incomplete:
            raise RuntimeError(
                "B3 collection incomplete: {}".format(",".join(incomplete))
            )
        summary["status"] = "completed"
        summary["scientific_complete_count"] = len(all_jobs)
    except Exception as error:
        summary["status"] = "failed"
        summary["failure"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        summary["finished_at"] = _utc_now()
        _atomic_json(summary_path, summary)


if __name__ == "__main__":
    main()
