#!/usr/bin/env python3

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
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_plans.a_series.progress_dashboard import run_parallel_trials


CONFIG_ROOT = ROOT / "configs" / "b_series_experiments"
RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
STRICT_TRAINING_SEEDS = (0, 1, 2)
STAGE_ORDER = ("P0", "R1", "R2", "R3")
STAGE_SPECS = {
    "P0": (
        ("B3-P0-A2-pseudo", "B3-P0-A2-pseudo.yaml", None),
    ),
    "R1": (
        ("B3-R1I", "B3-R1I-bounded-deep.yaml", "B3-P0-A2-pseudo"),
        ("B3-R1N", "B3-R1N-bounded-control.yaml", "B3-P0-A2-pseudo"),
    ),
    "R2": (
        ("B3-R2I", "B3-R2I-class-consistent.yaml", "B3-R1I"),
        ("B3-R2N", "B3-R2N-class-consistent-control.yaml", "B3-R1N"),
        ("B3-R2S", "B3-R2S-shuffled-relation.yaml", "B3-R1I"),
    ),
    "R3": (
        ("B3-R3I", "B3-R3I-partial-unfreeze.yaml", "B3-R2I"),
        ("B3-R3A", "B3-R3A-A2-continuation.yaml", "B3-P0-A2-pseudo"),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def _gpu_groups(raw: str) -> List[str]:
    values = [value.strip() for value in str(raw).split(";") if value.strip()]
    if not values or len(values) != len(set(values)) or any("," in value for value in values):
        raise ValueError("GPU groups must be unique single cards separated by semicolons")
    return values


def _load_splits(protocol: str, suite_path: Path) -> List[SplitIdentity]:
    if protocol == "final_gzsl":
        return [SplitIdentity(name="final_gzsl", manifest=None, sha256=None)]
    if not suite_path.is_file():
        raise FileNotFoundError("pseudo protocol requires --manifest-suite")
    payload = _read_json(suite_path)
    if payload.get("format") != "b3_class_disjoint_suite_v1":
        raise ValueError("unsupported B3 manifest suite format")
    records = list(payload.get("records") or ())
    if len(records) != 3:
        raise ValueError("strict B3 protocol requires exactly three pseudo splits")
    splits = []
    seen_names = set()
    for record in records:
        manifest = Path(str(record.get("path", ""))).expanduser()
        if not manifest.is_absolute():
            manifest = (suite_path.parent / manifest).resolve()
        else:
            manifest = manifest.resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        expected = str(record.get("sha256", ""))
        actual = _sha256(manifest)
        if expected != actual:
            raise ValueError("B3 manifest suite hash mismatch: {}".format(manifest))
        manifest_payload = _read_json(manifest)
        if manifest_payload.get("format") != "b3_class_disjoint_manifest_v1":
            raise ValueError("invalid B3 manifest: {}".format(manifest))
        name = "pseudo_seed{}".format(int(manifest_payload["class_seed"]))
        if name in seen_names:
            raise ValueError("duplicate B3 pseudo split {}".format(name))
        seen_names.add(name)
        splits.append(SplitIdentity(name=name, manifest=manifest, sha256=actual))
    return splits


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
    method = _resolved_method(base_method, ratio if base_method != "B3-P0-A2-pseudo" else None)
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


def _completed(output_root: Path) -> bool:
    summaries = list(output_root.rglob("monitor_runtime_summary.json")) if output_root.is_dir() else []
    completed = []
    for path in summaries:
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
    stage: str,
    splits: Sequence[SplitIdentity],
    ratios: Sequence[float],
    out_root: Path,
    require_checkpoints: bool,
    pilot_weights=(),
    a2_root: Optional[Path] = None,
    formal_weights=None,
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
                    training_seeds = (0,) if pilot_mode else STRICT_TRAINING_SEEDS
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
                                if parent_method == "B3-P0-A2-pseudo"
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
        "DATA.XLSA.B3_PSEUDO_MANIFEST",
        str(job.split.manifest) if job.split.manifest is not None else "",
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


def _run_job(job: Job, python_bin: str, protocol: str, gpu: str) -> Dict[str, object]:
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
    completed = process.returncode == 0 and _completed(job.output_root)
    return {
        "stage": job.stage,
        "method": job.method,
        "seed": job.seed,
        "pseudo_split": job.split.name,
        "pseudo_manifest_sha256": job.split.sha256,
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
    )
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
                job, args.python_bin, args.protocol, gpu
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
        description="Run isolated B3 pseudo-GZSL or locked-final training stages."
    )
    parser.add_argument("--stages", default="P0")
    parser.add_argument(
        "--protocol", choices=("b3_pseudo_gzsl", "final_gzsl"), default="b3_pseudo_gzsl"
    )
    parser.add_argument("--manifest-suite", type=Path, default=Path(""))
    parser.add_argument(
        "--a2-root",
        type=Path,
        default=Path(""),
        help="Optional existing full-GZSL A2 root used only by locked final validation.",
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
    suite_path = args.manifest_suite.expanduser().resolve()
    splits = _load_splits(args.protocol, suite_path)
    ratios = _ratios(args.max_ratios)
    args._a2_root = (
        args.a2_root.expanduser().resolve()
        if str(args.a2_root).strip() not in {"", "."}
        else None
    )
    if args._a2_root is not None and args.protocol != "final_gzsl":
        raise SystemExit("--a2-root is only allowed for locked final_gzsl validation")
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
    if args._pilot_weights:
        if stages != ["R2"] or args.protocol != "b3_pseudo_gzsl" or len(ratios) != 1:
            raise SystemExit(
                "R2 pilot requires --stages R2, pseudo protocol, and exactly one residual ratio"
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
            [0] if args._pilot_weights else list(STRICT_TRAINING_SEEDS)
        ),
        "formal_evidence": not bool(args._pilot_weights),
        "r2_pilot_weights": [list(pair) for pair in args._pilot_weights],
        "formal_class_consistency_weights": (
            list(args._formal_weights) if args._formal_weights else None
        ),
        "pseudo_splits": [
            {
                "name": split.name,
                "manifest": str(split.manifest) if split.manifest else None,
                "sha256": split.sha256,
            }
            for split in splits
        ],
        "max_ratios": ratios,
        "reused_controls": {
            "B3-R1A": "B3-P0-A2-pseudo paired by split and training seed",
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
