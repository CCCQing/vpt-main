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
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials
from src.tools.search_plans.common import (
    directory_has_contents as _has_contents,
    parse_gpu_worker_slots as _gpu_groups,
    read_json as _read_json,
)


CONFIG_ROOT = ROOT / "configs" / "b_series_experiments"
RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
STRICT_TRAINING_SEEDS = (0, 1, 2)
STAGE_ORDER = ("P0", "R1", "S0", "R2", "R3", "T1", "T2")
STAGE_SPECS = {
    "P0": (
        ("B3-P0-A2-final", "B3-P0-A2-final.yaml", None),
    ),
    "R1": (
        ("B3-R1I", "B3-R1I-bounded-deep.yaml", "B3-P0-A2-final"),
        ("B3-R1N", "B3-R1N-bounded-control.yaml", "B3-P0-A2-final"),
    ),
    "S0": (
        ("B3-S0I-deep", "B3-S0I-direct-deep.yaml", "B3-P0-A2-final"),
        ("B3-S0N-deep-control", "B3-S0N-direct-deep-control.yaml", "B3-P0-A2-final"),
        ("B3-S0I-shallow", "B3-S0I-direct-shallow.yaml", "B3-P0-A2-final"),
        ("B3-S0N-shallow-control", "B3-S0N-direct-shallow-control.yaml", "B3-P0-A2-final"),
    ),
    "R2": (
        ("B3-R2I", "B3-R2I-class-consistent.yaml", "B3-R1I"),
        ("B3-R2N", "B3-R2N-class-consistent-control.yaml", "B3-R1N"),
        ("B3-R2S", "B3-R2S-shuffled-relation.yaml", "B3-R1I"),
    ),
    "R3": (
        ("B3-R3I", "B3-R3I-partial-unfreeze.yaml", "B3-R2I"),
        ("B3-R3A", "B3-R3A-A2-continuation.yaml", "B3-P0-A2-final"),
    ),
    "T1": (
        ("B3-T1I-direct-slot-scalar", "B3-T1I-slot-scalar.yaml", "B3-P0-A2-final"),
        (
            "B3-T1N-direct-slot-scalar-control",
            "B3-T1N-slot-scalar-control.yaml",
            "B3-P0-A2-final",
        ),
    ),
    "T2": (
        ("B3-T2I-rank2", "B3-T2I-slot-low-rank2.yaml", "B3-P0-A2-final"),
        ("B3-T2N-rank2-control", "B3-T2N-slot-low-rank2-control.yaml", "B3-P0-A2-final"),
        ("B3-T2I-rank4", "B3-T2I-slot-low-rank4.yaml", "B3-P0-A2-final"),
        ("B3-T2N-rank4-control", "B3-T2N-slot-low-rank4-control.yaml", "B3-P0-A2-final"),
    ),
}


@dataclass(frozen=True)
class SplitIdentity:
    name: str
    manifest: Optional[Path]
    sha256: Optional[str]


@dataclass(frozen=True)
class Job:
    stage: str
    method: str
    seed: int
    split: SplitIdentity
    config_file: Path
    checkpoint: Optional[Path]
    output_root: Path
    log_path: Path
    ratio: Optional[float]
    intra_weight: Optional[float]
    inter_weight: Optional[float]


def _comma_values(raw: str) -> List[str]:
    values = [value.strip().upper() for value in str(raw).split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("comma-separated values must be non-empty and unique")
    return values


def _ratios(raw: str) -> List[float]:
    values = [float(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--max-ratios must be non-empty and unique")
    if any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("B3 residual ratios must lie inside (0, 1]")
    return values


def _pilot_weights(raw: str):
    if not str(raw).strip():
        return []
    pairs = []
    for item in str(raw).split(","):
        left, separator, right = item.strip().partition(":")
        if not separator:
            raise ValueError("pilot weights must use INTRA:INTER pairs")
        pair = (float(left), float(right))
        if pair[0] < 0.0 or pair[1] < 0.0 or sum(pair) <= 0.0:
            raise ValueError("pilot loss weights must be non-negative and non-zero")
        pairs.append(pair)
    if len(pairs) != len(set(pairs)):
        raise ValueError("pilot loss-weight pairs must be unique")
    return pairs


def _load_splits(protocol: str) -> List[SplitIdentity]:
    if protocol != "final_gzsl":
        raise ValueError("the active B3 protocol requires normal final-GZSL classes")
    return [SplitIdentity(name="final_gzsl", manifest=None, sha256=None)]


def _ratio_tag(ratio: float) -> str:
    return "R{:03d}".format(int(round(float(ratio) * 100.0)))


def _resolved_method(base_method: str, ratio: Optional[float]) -> str:
    return base_method if ratio is None else "{}-{}".format(base_method, _ratio_tag(ratio))


def _weight_tag(intra_weight: float, inter_weight: float) -> str:
    def encode(value):
        return ("{:.6g}".format(float(value))).replace(".", "p")

    return "wi{}-we{}".format(encode(intra_weight), encode(inter_weight))


def _checkpoint_path(
    out_root: Path,
    split: SplitIdentity,
    base_method: str,
    seed: int,
    ratio: Optional[float],
) -> Path:
    method = _resolved_method(base_method, ratio if base_method != "B3-P0-A2-final" else None)
    return (
        out_root
        / split.name
        / method
        / "seed{}".format(seed)
        / RUN_SUFFIX
        / "model_final_trainable.pth"
    )


def _external_a2_checkpoint(a2_root: Path, seed: int) -> Path:
    candidates = [
        a2_root / "A2" / "seed{}".format(seed) / RUN_SUFFIX
        / "model_final_trainable.pth",
        a2_root / "seed{}".format(seed) / RUN_SUFFIX
        / "model_final_trainable.pth",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            "expected one final-GZSL A2 seed {} checkpoint; checked {}".format(
                seed, ", ".join(str(path) for path in candidates)
            )
        )
    return existing[0]


def _completed(output_root: Path, allow_checkpoint_ready: bool = False) -> bool:
    summaries = list(output_root.rglob("monitor_runtime_summary.json")) if output_root.is_dir() else []
    completed = []
    for path in summaries:
        try:
            if str(_read_json(path).get("status", "")).lower() != "completed":
                continue
            run_dir = path.parent
            checkpoint_ready = run_dir / "training_checkpoint_ready.json"
            if allow_checkpoint_ready and checkpoint_ready.is_file():
                marker = _read_json(checkpoint_ready)
                checkpoint = marker.get("checkpoint") or {}
                checkpoint_path = run_dir / str(checkpoint.get("checkpoint_path", ""))
                if (
                    str(marker.get("status", "")).lower() == "checkpoint_ready"
                    and bool(marker.get("training_performed", False))
                    and not bool(marker.get("fixed_probe_performed", True))
                    and checkpoint_path.is_file()
                ):
                    completed.append(path)
                    continue
            deferred_collection = run_dir / "b3_fixed_probe_collection.json"
            if deferred_collection.is_file() and bool(
                _read_json(deferred_collection).get("valid", False)
            ):
                completed.append(path)
                continue
            robustness = run_dir / "diagnostics" / "probe_robustness_manifest.json"
            if not robustness.is_file():
                continue
            payload = _read_json(robustness)
            seeds = [int(item) for item in payload.get("selection_seeds", [])]
            executions = list(payload.get("executions") or [])
            if (
                seeds == [424242, 424243, 424244]
                and len(executions) == 3
                and all(
                    isinstance(item, dict)
                    and str(item.get("execution_profile")) == "final_full"
                    for item in executions
                )
            ):
                completed.append(path)
        except (OSError, ValueError):
            continue
    if len(completed) > 1:
        raise RuntimeError("multiple completed runs found under {}".format(output_root))
    return bool(completed)


def _build_jobs(
    stage: str,
    splits: Sequence[SplitIdentity],
    ratios: Sequence[float],
    out_root: Path,
    require_checkpoints: bool,
    pilot_weights=(),
    a2_root: Optional[Path] = None,
    formal_weights=None,
    screening_seed: Optional[int] = None,
) -> List[Job]:
    jobs = []
    pilot_mode = bool(pilot_weights)
    active_splits = list(splits[:1]) if pilot_mode else list(splits)
    specs = STAGE_SPECS[stage]
    if pilot_mode:
        if stage != "R2":
            raise ValueError("class-consistency pilot is only defined for R2")
        specs = (STAGE_SPECS[stage][0],)
    for split in active_splits:
        for base_method, config_name, parent_method in specs:
            stage_ratios: Sequence[Optional[float]] = (None,) if stage == "P0" else ratios
            for ratio in stage_ratios:
                weight_pairs = pilot_weights if pilot_mode else ((None, None),)
                for intra_weight, inter_weight in weight_pairs:
                    method = _resolved_method(base_method, ratio)
                    if pilot_mode:
                        method = "B3-R2P-{}-{}".format(
                            _ratio_tag(ratio),
                            _weight_tag(intra_weight, inter_weight),
                        )
                    training_seeds = (
                        (0,)
                        if pilot_mode
                        else (int(screening_seed),)
                        if screening_seed is not None
                        else STRICT_TRAINING_SEEDS
                    )
                    for seed in training_seeds:
                        job_intra_weight = intra_weight
                        job_inter_weight = inter_weight
                        if (
                            not pilot_mode
                            and formal_weights is not None
                            and base_method in {"B3-R2I", "B3-R2N", "B3-R2S", "B3-R3I"}
                        ):
                            job_intra_weight, job_inter_weight = formal_weights
                        checkpoint = None
                        if parent_method is not None:
                            checkpoint = (
                                _external_a2_checkpoint(a2_root, seed)
                                if parent_method == "B3-P0-A2-final"
                                and a2_root is not None
                                else _checkpoint_path(
                                    out_root, split, parent_method, seed, ratio
                                )
                            )
                            if require_checkpoints and not checkpoint.is_file():
                                raise FileNotFoundError(
                                    "{} requires completed parent checkpoint {}".format(
                                        method, checkpoint
                                    )
                                )
                        output_root = out_root / split.name / method / "seed{}".format(seed)
                        jobs.append(
                            Job(
                                stage=stage,
                                method=method,
                                seed=seed,
                                split=split,
                                config_file=CONFIG_ROOT / config_name,
                                checkpoint=checkpoint,
                                output_root=output_root,
                                log_path=out_root / "launcher_logs" / split.name
                                / "{}_seed{}.log".format(method, seed),
                                ratio=ratio,
                                intra_weight=job_intra_weight,
                                inter_weight=job_inter_weight,
                            )
                        )
    return jobs


def _command(job: Job, python_bin: str, protocol: str) -> List[str]:
    command = [
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
        "DATA.XLSA.PROTOCOL_MODE",
        protocol,
        "MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.ARCHITECTURE_ID",
        job.method,
    ]
    if job.checkpoint is not None:
        command.extend(
            ["SOLVER.INIT_TRAINABLE_CHECKPOINT", str(job.checkpoint)]
        )
    if job.ratio is not None:
        command.extend(
            [
                "MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.BOUNDED_MAX_RATIO",
                str(job.ratio),
                "MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.BOUNDED_INIT_RATIO",
                str(float(job.ratio) / 2.0),
            ]
        )
    if job.intra_weight is not None and job.inter_weight is not None:
        command.extend(
            [
                "SOLVER.B3_CLASS_CONSISTENCY.INTRA_WEIGHT",
                str(job.intra_weight),
                "SOLVER.B3_CLASS_CONSISTENCY.INTER_WEIGHT",
                str(job.inter_weight),
            ]
        )
    return command


def _format(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _run_job(
    job: Job,
    python_bin: str,
    protocol: str,
    gpu: str,
    allow_checkpoint_ready: bool = False,
) -> Dict[str, object]:
    command = _command(job, python_bin, protocol)
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
    completed = process.returncode == 0 and _completed(
        job.output_root, allow_checkpoint_ready=allow_checkpoint_ready
    )
    return {
        "stage": job.stage,
        "method": job.method,
        "seed": job.seed,
        "evaluation_split": job.split.name,
        "split_manifest_sha256": job.split.sha256,
        "gpu": gpu,
        "status": "completed" if completed else "failed",
        "returncode": int(process.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "output_root": str(job.output_root),
        "log_path": str(job.log_path),
        "initialization_checkpoint": str(job.checkpoint) if job.checkpoint else None,
    }


def _run_stage(args, stage, splits, ratios, gpus, out_root):
    jobs = _build_jobs(
        stage,
        splits,
        ratios,
        out_root,
        require_checkpoints=not args.dry_run,
        pilot_weights=args._pilot_weights,
        a2_root=args._a2_root,
        formal_weights=args._formal_weights,
        screening_seed=args.screening_seed,
    )
    missing_configs = sorted(
        {str(job.config_file) for job in jobs if not job.config_file.is_file()}
    )
    if missing_configs:
        raise FileNotFoundError(", ".join(missing_configs))
    pending = []
    skipped = []
    allow_checkpoint_ready = args.screening_seed is not None
    for job in jobs:
        if _completed(
            job.output_root, allow_checkpoint_ready=allow_checkpoint_ready
        ) and not args.no_resume:
            skipped.append(job)
        elif _has_contents(job.output_root):
            raise RuntimeError("refusing to overwrite incomplete output {}".format(job.output_root))
        else:
            pending.append(job)
    print(
        "B3 stage={} total={} pending={} skipped={}".format(
            stage, len(jobs), len(pending), len(skipped)
        ),
        flush=True,
    )
    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % min(args.max_workers, len(gpus))]
            print("[dry-run] gpu={} {}".format(gpu, _format(_command(job, args.python_bin, args.protocol))))
        return [], skipped
    trials = [
        {
            "trial_index": index,
            "trial_name": "{}_{}_seed{}".format(job.split.name, job.method, job.seed),
            "runner": "train",
            "stage": stage,
            "combo": {"method": job.method, "seed": job.seed, "split": job.split.name},
            "overrides": {"initialization_checkpoint": str(job.checkpoint)},
            "output_dir": str(job.output_root),
            "identity_fields": {
                "config_file": str(job.config_file),
                "manifest_sha256": job.split.sha256,
            },
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
            run_item=lambda job, gpu: _run_job(
                job,
                args.python_bin,
                args.protocol,
                gpu,
                allow_checkpoint_ready=allow_checkpoint_ready,
            ),
            item_name=lambda job: "{} {} seed={}".format(
                job.split.name, job.method, job.seed
            ),
        )
        if pending
        else []
    )
    if any(result["status"] != "completed" for result in results):
        raise RuntimeError("B3 stage {} has failed jobs; later stages were not started".format(stage))
    return results, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the active B3 normal Seen/Unseen GZSL training stages."
    )
    parser.add_argument("--stages", default="P0")
    parser.add_argument(
        "--protocol", choices=("final_gzsl",), default="final_gzsl"
    )
    parser.add_argument(
        "--a2-root",
        type=Path,
        default=Path(""),
        help="Optional existing full-GZSL A2 root used as the matched P0 source.",
    )
    parser.add_argument("--max-ratios", default="0.25,0.50")
    parser.add_argument(
        "--r2-pilot-weights",
        default="",
        help="Optional comma-separated INTRA:INTER grid; runs only split 1 / seed 0 and is never formal evidence.",
    )
    parser.add_argument("--formal-intra-weight", type=float)
    parser.add_argument("--formal-inter-weight", type=float)
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "output" / "b3_series" / "training"
    )
    parser.add_argument("--gpu-groups", default="0;1;2")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--screening-seed",
        type=int,
        help="Run one declared training seed as non-formal screening evidence.",
    )
    parser.add_argument("--no-resume", action="store_true")
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
    splits = _load_splits(args.protocol)
    ratios = _ratios(args.max_ratios)
    args._a2_root = (
        args.a2_root.expanduser().resolve()
        if str(args.a2_root).strip() not in {"", "."}
        else None
    )
    args._pilot_weights = _pilot_weights(args.r2_pilot_weights)
    if (args.formal_intra_weight is None) != (args.formal_inter_weight is None):
        raise SystemExit("formal intra/inter weights must be provided together")
    args._formal_weights = (
        (float(args.formal_intra_weight), float(args.formal_inter_weight))
        if args.formal_intra_weight is not None
        else None
    )
    if args._formal_weights is not None and (
        min(args._formal_weights) < 0.0 or sum(args._formal_weights) <= 0.0
    ):
        raise SystemExit("formal B3 class-consistency weights are invalid")
    if args._pilot_weights and args._formal_weights is not None:
        raise SystemExit("pilot and formal class-consistency weights cannot be combined")
    if args.screening_seed is not None and args.screening_seed < 0:
        raise SystemExit("--screening-seed must be non-negative")
    if args.screening_seed is not None and args._pilot_weights:
        raise SystemExit("--screening-seed cannot be combined with the R2 pilot")
    if args._pilot_weights:
        if stages != ["R2"] or len(ratios) != 1:
            raise SystemExit(
                "R2 pilot requires --stages R2 and exactly one residual ratio"
            )
    out_root = args.out_root.expanduser().resolve()
    all_results = []
    all_skipped = []
    for stage in stages:
        results, skipped = _run_stage(
            args, stage, splits, ratios, gpus, out_root
        )
        all_results.extend(results)
        all_skipped.extend(skipped)
    if args.dry_run:
        return
    summary = {
        "format": "b3_series_training_launcher_v1",
        "suite_name": "B3-series",
        "protocol": args.protocol,
        "external_a2_root": str(args._a2_root) if args._a2_root else None,
        "stages": stages,
        "training_seeds": (
            [0]
            if args._pilot_weights
            else [int(args.screening_seed)]
            if args.screening_seed is not None
            else list(STRICT_TRAINING_SEEDS)
        ),
        "formal_evidence": not bool(args._pilot_weights) and args.screening_seed is None,
        "screening_seed": args.screening_seed,
        "r2_pilot_weights": [list(pair) for pair in args._pilot_weights],
        "formal_class_consistency_weights": (
            list(args._formal_weights) if args._formal_weights else None
        ),
        "evaluation_splits": [
            {
                "name": split.name,
                "manifest": str(split.manifest) if split.manifest else None,
                "sha256": split.sha256,
            }
            for split in splits
        ],
        "max_ratios": ratios,
        "reused_controls": {
            "B3-R1A": "B3-P0-A2-final paired by training seed",
            "B3-R2B": "B3-R1I checkpoint at the same residual ratio",
            "B3-R3F": "B3-R2I checkpoint before partial unfreezing",
        },
        "results": all_results,
        "skipped_existing": [
            {"split": job.split.name, "method": job.method, "seed": job.seed}
            for job in all_skipped
        ],
        "status": "completed",
    }
    path = out_root / "b3_series_training_launcher_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
