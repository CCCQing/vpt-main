#!/usr/bin/env python3
"""Run the six-cell Graph-GP training and Stage-2 analysis pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.run_arch_ablation_graph_gp_energy import _cell_specs  # noqa: E402


CELLS = ("A00", "A10", "C00", "C10", "C01", "C11")
STAGE2B_GROUPS = (
    "D0_image_posterior",
    "D0_empirical_replace_oracle",
    "D0_empirical_fusion_oracle",
    "D1_moment_replace_oracle",
    "D1_moment_fusion_oracle",
    "D2_task_replace_oracle",
    "D2_task_fusion_oracle",
    "D2_graph_gp_deployable",
    "D2_shuffled_graph_deployable",
    "D3_ce_only_oracle_reference",
)
DEFAULT_SEEDS = (17, 29, 43)
DEFAULT_GRAPH_BUNDLE = (
    ROOT
    / "cub_attribute_localization"
    / "05_hparam_searches"
    / "diff_only_graphs_v1"
    / "diff_only_method_matrices_v1.npz"
)


def _cell_names() -> Dict[str, str]:
    return {str(spec["cell_id"]): str(spec["cell_name"]) for spec in _cell_specs()}


def _parse_seeds(raw: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError("--seeds must contain unique non-negative integers.")
    return values


def _parse_gpu_groups(raw: str) -> List[str]:
    groups = [item.strip() for item in str(raw).split(";") if item.strip()]
    return groups or [""]


def _run_command(
    command: Sequence[str], log_path: Path, env: Optional[Mapping[str, str]] = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=str(ROOT),
            env=dict(env) if env is not None else None,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with returncode={process.returncode}; see {log_path}")


def _checkpoint_path(stage1_root: Path, cell: str, seed: int) -> Path:
    cell_dir = stage1_root / _cell_names()[cell] / f"seed_{seed}"
    matches = sorted(cell_dir.rglob("model_final_trainable.pth")) if cell_dir.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one final checkpoint for {cell}/seed{seed} under {cell_dir}, found {matches}"
        )
    return matches[0]


def _cache_path(out_root: Path, cell: str, seed: int) -> Path:
    return out_root / "caches" / cell.lower() / f"seed_{seed}.npz"


def _stage1_result_root(out_root: Path) -> Path:
    return out_root / "stage1" / "final_gzsl"


def _result_dir(out_root: Path, cell: str, seed: int) -> Path:
    return out_root / "stage2" / cell.lower() / f"seed_{seed}"


def _stage2b_oracle_path(out_root: Path, seed: int) -> Path:
    return out_root / "stage2b" / "oracles" / f"seed_{seed}.npz"


def _stage2b_healthy_path(out_root: Path, seed: int) -> Path:
    return out_root / "stage2b" / "healthy_banks" / f"seed_{seed}.npz"


def _stage2b_result_dir(out_root: Path, seed: int) -> Path:
    return out_root / "stage2b" / "healthy_results" / f"seed_{seed}"


def _job_specs(out_root: Path, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    return [
        {
            "cell": cell,
            "seed": int(seed),
            "cache": _cache_path(out_root, cell, int(seed)),
            "result_dir": _result_dir(out_root, cell, int(seed)),
        }
        for cell in CELLS
        for seed in seeds
    ]


def _train(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    command = [
        args.python_bin,
        str(ROOT / "src" / "tools" / "run_arch_ablation_graph_gp_energy.py"),
        "--config-file",
        str(args.config_file),
        "--out-root",
        str(args.out_root / "stage1"),
        "--protocol-mode",
        "final_gzsl",
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--cells",
        ",".join(CELLS),
        "--gpu-groups",
        str(args.gpu_groups),
        "--nproc-per-trial",
        str(args.nproc_per_trial),
        "--max-workers",
        str(args.max_workers),
    ]
    if args.no_resume:
        command.append("--no-resume")
    _run_command(command, args.out_root / "logs" / "train_pipeline.log")


def _export_command(args: argparse.Namespace, job: Mapping[str, Any], checkpoint: Path) -> List[str]:
    return [
        args.python_bin,
        str(ROOT / "src" / "tools" / "export_prompt_posterior_cache.py"),
        "--config-file",
        str(args.config_file),
        "--cell",
        str(job["cell"]),
        "--seed",
        str(job["seed"]),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(job["cache"]),
        "--batch-size",
        str(args.export_batch_size),
        "--num-workers",
        str(args.export_num_workers),
    ]


def _export(args: argparse.Namespace, jobs: Sequence[Mapping[str, Any]], gpu_groups: Sequence[str]) -> None:
    pending = []
    for job in jobs:
        cache = Path(job["cache"])
        if cache.is_file() and cache.with_suffix(".json").is_file() and not args.no_resume:
            continue
        checkpoint = _checkpoint_path(_stage1_result_root(args.out_root), str(job["cell"]), int(job["seed"]))
        pending.append((job, checkpoint))
    worker_count = min(args.max_workers, len(gpu_groups), max(len(pending), 1))

    assignments: List[List[Any]] = [[] for _ in range(worker_count)]
    for index, item in enumerate(pending):
        assignments[index % worker_count].append(item)

    def run_worker(worker_index: int) -> None:
        gpu_group = gpu_groups[worker_index]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_group
        for job, checkpoint in assignments[worker_index]:
            _run_command(
                _export_command(args, job, checkpoint),
                args.out_root / "logs" / "export" / f"{str(job['cell']).lower()}_seed_{job['seed']}.log",
                env=env,
            )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_worker, worker_index) for worker_index in range(worker_count)]
        for future in as_completed(futures):
            future.result()
    missing = [str(job["cache"]) for job in jobs if not Path(job["cache"]).is_file()]
    if missing:
        raise RuntimeError(f"Posterior cache coverage is incomplete: {missing}")


def _evaluate_command(args: argparse.Namespace, job: Mapping[str, Any]) -> List[str]:
    return [
        args.python_bin,
        str(ROOT / "src" / "tools" / "evaluate_graph_gp_center_transfer.py"),
        "--cache",
        str(job["cache"]),
        "--graph-bundle",
        str(args.graph_bundle),
        "--manifest",
        str(args.out_root / "manifests" / "CUB_graph_gp_class_folds_seed2027.json"),
        "--output-dir",
        str(job["result_dir"]),
        "--num-folds",
        str(args.num_folds),
        "--shuffle-count",
        str(args.shuffle_count),
        "--alignment-topk",
        str(args.alignment_topk),
    ]


def _evaluate(args: argparse.Namespace, jobs: Sequence[Mapping[str, Any]]) -> None:
    manifest = args.out_root / "manifests" / "CUB_graph_gp_class_folds_seed2027.json"
    pending = [
        job
        for job in jobs
        if args.no_resume
        or not all(
            (Path(job["result_dir"]) / name).is_file()
            for name in ("fold_results.csv", "cross_space_alignment.csv", "results.json")
        )
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    if pending and not manifest.is_file():
        first = pending.pop(0)
        _run_command(
            _evaluate_command(args, first),
            args.out_root / "logs" / "evaluate" / f"{str(first['cell']).lower()}_seed_{first['seed']}.log",
            env=env,
        )

    def run_one(job: Mapping[str, Any]) -> None:
        _run_command(
            _evaluate_command(args, job),
            args.out_root / "logs" / "evaluate" / f"{str(job['cell']).lower()}_seed_{job['seed']}.log",
            env=env,
        )

    with ThreadPoolExecutor(max_workers=min(args.cpu_workers, max(len(pending), 1))) as executor:
        futures = [executor.submit(run_one, job) for job in pending]
        for future in as_completed(futures):
            future.result()
    missing = [
        str(Path(job["result_dir"]))
        for job in jobs
        if not all(
            (Path(job["result_dir"]) / name).is_file()
            for name in ("fold_results.csv", "cross_space_alignment.csv", "results.json")
        )
    ]
    if missing:
        raise RuntimeError(f"Stage-2 result coverage is incomplete: {missing}")


def _summarize(args: argparse.Namespace, jobs: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> None:
    fold_inputs = [str(Path(job["result_dir"]) / "fold_results.csv") for job in jobs]
    alignment_inputs = [str(Path(job["result_dir"]) / "cross_space_alignment.csv") for job in jobs]
    stage2_dir = args.out_root / "summaries" / "stage2"
    factorial_dir = args.out_root / "summaries" / "factorial"
    stage2_command = [
        args.python_bin,
        str(ROOT / "src" / "tools" / "summarize_graph_gp_offline.py"),
        "--inputs",
        *fold_inputs,
        "--output-dir",
        str(stage2_dir),
        "--expected-seeds",
        ",".join(str(seed) for seed in seeds),
        "--expected-folds",
        str(args.num_folds),
    ]
    _run_command(stage2_command, args.out_root / "logs" / "summarize_stage2.log")
    factorial_command = [
        args.python_bin,
        str(ROOT / "src" / "tools" / "summarize_graph_gp_factorial.py"),
        "--stage1-summary",
        str(_stage1_result_root(args.out_root) / "summary.csv"),
        "--fold-inputs",
        *fold_inputs,
        "--alignment-inputs",
        *alignment_inputs,
        "--output-dir",
        str(factorial_dir),
        "--expected-seeds",
        ",".join(str(seed) for seed in seeds),
        "--expected-folds",
        str(args.num_folds),
    ]
    _run_command(factorial_command, args.out_root / "logs" / "summarize_factorial.log")


def _stage2b_oracle_command(args: argparse.Namespace, seed: int, checkpoint: Path) -> List[str]:
    return [
        args.python_bin,
        str(ROOT / "src" / "tools" / "build_stage2b_oracle_prompt_distributions.py"),
        "--config-file",
        str(args.config_file),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(_stage2b_oracle_path(args.out_root, seed)),
        "--batch-size",
        str(args.stage2b_batch_size),
        "--num-workers",
        str(args.export_num_workers),
        "--build-ratio",
        str(args.stage2b_build_ratio),
        "--epochs",
        str(args.stage2b_oracle_epochs),
        "--lr",
        str(args.stage2b_oracle_lr),
        "--align-weight",
        str(args.stage2b_align_weight),
        "--anchor-weight",
        str(args.stage2b_anchor_weight),
    ]


def _run_gpu_seed_jobs(
    args: argparse.Namespace,
    seeds: Sequence[int],
    gpu_groups: Sequence[str],
    callback,
) -> None:
    worker_count = min(args.max_workers, len(gpu_groups), max(len(seeds), 1))
    assignments: List[List[int]] = [[] for _ in range(worker_count)]
    for index, seed in enumerate(seeds):
        assignments[index % worker_count].append(int(seed))

    def run_worker(worker_index: int) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_groups[worker_index]
        for seed in assignments[worker_index]:
            callback(seed, env)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_worker, index) for index in range(worker_count)]
        for future in as_completed(futures):
            future.result()


def _stage2b_oracle(args: argparse.Namespace, seeds: Sequence[int], gpu_groups: Sequence[str]) -> None:
    pending = [
        int(seed)
        for seed in seeds
        if args.no_resume
        or not (
            _stage2b_oracle_path(args.out_root, int(seed)).is_file()
            and _stage2b_oracle_path(args.out_root, int(seed)).with_suffix(".json").is_file()
        )
    ]

    def run_one(seed: int, env: Mapping[str, str]) -> None:
        checkpoint = _checkpoint_path(_stage1_result_root(args.out_root), "A00", seed)
        _run_command(
            _stage2b_oracle_command(args, seed, checkpoint),
            args.out_root / "logs" / "stage2b_oracle" / f"seed_{seed}.log",
            env=env,
        )

    _run_gpu_seed_jobs(args, pending, gpu_groups, run_one)
    missing = [str(_stage2b_oracle_path(args.out_root, seed)) for seed in seeds if not _stage2b_oracle_path(args.out_root, seed).is_file()]
    if missing:
        raise RuntimeError(f"Stage-2B oracle coverage is incomplete: {missing}")


def _stage2b_healthy_command(args: argparse.Namespace, seed: int, checkpoint: Path) -> List[str]:
    return [
        args.python_bin,
        str(ROOT / "src" / "tools" / "build_stage2b_healthy_prompt_distributions.py"),
        "--config-file",
        str(args.config_file),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--posterior-cache",
        str(_cache_path(args.out_root, "A00", seed)),
        "--graph-bundle",
        str(args.graph_bundle),
        "--graph-key",
        str(args.stage2b_graph_key),
        "--output",
        str(_stage2b_healthy_path(args.out_root, seed)),
        "--num-workers",
        str(args.export_num_workers),
        "--epochs",
        str(args.stage2b_healthy_epochs),
        "--lr",
        str(args.stage2b_healthy_lr),
        "--generator-hidden-dim",
        str(args.stage2b_generator_hidden_dim),
        "--classes-per-batch",
        str(args.stage2b_classes_per_batch),
        "--samples-per-class",
        str(args.stage2b_samples_per_class),
        "--moment-samples",
        str(args.stage2b_moment_samples),
        "--ce-weight",
        str(args.stage2b_ce_weight),
        "--mean-weight",
        str(args.stage2b_mean_weight),
        "--variance-weight",
        str(args.stage2b_variance_weight),
        "--semantic-weight",
        str(args.stage2b_semantic_weight),
        "--prompt-visual-weight",
        str(args.stage2b_prompt_visual_weight),
        "--prompt-semantic-weight",
        str(args.stage2b_prompt_semantic_weight),
        "--anchor-weight",
        str(args.stage2b_anchor_weight),
        "--graph-weight",
        str(args.stage2b_graph_weight),
        "--mu-delta-scale",
        str(args.stage2b_mu_delta_scale),
        "--logvar-delta-scale",
        str(args.stage2b_logvar_delta_scale),
        "--prior-ridge",
        str(args.stage2b_prior_ridge),
        "--gp-var-weight",
        str(args.stage2b_gp_var_weight),
        "--support-var-weight",
        str(args.stage2b_support_var_weight),
        "--pseudo-folds",
        str(args.num_folds),
    ]


def _stage2b_healthy(args: argparse.Namespace, seeds: Sequence[int], gpu_groups: Sequence[str]) -> None:
    missing_caches = [
        str(_cache_path(args.out_root, "A00", int(seed)))
        for seed in seeds
        if not _cache_path(args.out_root, "A00", int(seed)).is_file()
    ]
    if missing_caches:
        raise RuntimeError(f"Stage-2B healthy-bank A00 caches are incomplete: {missing_caches}")
    pending = [
        int(seed)
        for seed in seeds
        if args.no_resume
        or not all(
            path.is_file()
            for path in (
                _stage2b_healthy_path(args.out_root, int(seed)),
                _stage2b_healthy_path(args.out_root, int(seed)).with_suffix(".json"),
                _stage2b_healthy_path(args.out_root, int(seed)).with_suffix(".pth"),
                _stage2b_healthy_path(args.out_root, int(seed)).with_name(
                    _stage2b_healthy_path(args.out_root, int(seed)).stem + "_pseudo_unseen.csv"
                ),
                _stage2b_healthy_path(args.out_root, int(seed)).with_name(
                    _stage2b_healthy_path(args.out_root, int(seed)).stem + "_history.csv"
                ),
            )
        )
    ]

    def run_one(seed: int, env: Mapping[str, str]) -> None:
        checkpoint = _checkpoint_path(_stage1_result_root(args.out_root), "A00", seed)
        _run_command(
            _stage2b_healthy_command(args, seed, checkpoint),
            args.out_root / "logs" / "stage2b_healthy" / f"seed_{seed}.log",
            env=env,
        )

    _run_gpu_seed_jobs(args, pending, gpu_groups, run_one)
    missing = [
        str(_stage2b_healthy_path(args.out_root, seed))
        for seed in seeds
        if not _stage2b_healthy_path(args.out_root, seed).is_file()
    ]
    if missing:
        raise RuntimeError(f"Stage-2B healthy-bank coverage is incomplete: {missing}")


def _stage2b_evaluate_command(args: argparse.Namespace, seed: int, checkpoint: Path) -> List[str]:
    return [
        args.python_bin,
        str(ROOT / "src" / "tools" / "evaluate_stage2b_healthy_intervention.py"),
        "--config-file",
        str(args.config_file),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--healthy-bank",
        str(_stage2b_healthy_path(args.out_root, seed)),
        "--oracle",
        str(_stage2b_oracle_path(args.out_root, seed)),
        "--output-dir",
        str(_stage2b_result_dir(args.out_root, seed)),
        "--batch-size",
        str(args.stage2b_batch_size),
        "--num-workers",
        str(args.export_num_workers),
        "--fusion-alpha",
        str(args.stage2b_fusion_alpha),
        "--energy-beta",
        str(args.stage2b_energy_beta),
        "--candidate-temperature",
        str(args.stage2b_candidate_temperature),
        "--candidate-topk",
        str(args.stage2b_candidate_topk),
    ]


def _stage2b_evaluate(args: argparse.Namespace, seeds: Sequence[int], gpu_groups: Sequence[str]) -> None:
    missing_inputs = [
        str(path)
        for seed in seeds
        for path in (
            _stage2b_healthy_path(args.out_root, int(seed)),
            _stage2b_oracle_path(args.out_root, int(seed)),
        )
        if not path.is_file()
    ]
    if missing_inputs:
        raise RuntimeError(f"Stage-2B evaluation inputs are incomplete: {missing_inputs}")
    pending = [
        int(seed)
        for seed in seeds
        if args.no_resume
        or not all(
            (_stage2b_result_dir(args.out_root, int(seed)) / name).is_file()
            for name in ("group_results.csv", "prior_banks.npz", "results.json")
        )
    ]

    def run_one(seed: int, env: Mapping[str, str]) -> None:
        checkpoint = _checkpoint_path(_stage1_result_root(args.out_root), "A00", seed)
        _run_command(
            _stage2b_evaluate_command(args, seed, checkpoint),
            args.out_root / "logs" / "stage2b_evaluate" / f"seed_{seed}.log",
            env=env,
        )

    _run_gpu_seed_jobs(args, pending, gpu_groups, run_one)
    missing = [
        str(_stage2b_result_dir(args.out_root, seed))
        for seed in seeds
        if not (_stage2b_result_dir(args.out_root, seed) / "group_results.csv").is_file()
    ]
    if missing:
        raise RuntimeError(f"Stage-2B result coverage is incomplete: {missing}")


def _stage2b_summarize(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    inputs = [str(_stage2b_result_dir(args.out_root, seed) / "group_results.csv") for seed in seeds]
    pseudo_inputs = [
        str(
            _stage2b_healthy_path(args.out_root, seed).with_name(
                _stage2b_healthy_path(args.out_root, seed).stem + "_pseudo_unseen.csv"
            )
        )
        for seed in seeds
    ]
    command = [
        args.python_bin,
        str(ROOT / "src" / "tools" / "summarize_stage2b_healthy_intervention.py"),
        "--inputs",
        *inputs,
        "--pseudo-inputs",
        *pseudo_inputs,
        "--output-dir",
        str(args.out_root / "summaries" / "stage2b_healthy"),
        "--expected-seeds",
        ",".join(str(seed) for seed in seeds),
        "--expected-folds",
        str(args.num_folds),
    ]
    _run_command(command, args.out_root / "logs" / "summarize_stage2b.log")


def _write_plan(args: argparse.Namespace, jobs: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> None:
    payload = {
        "format": "graph_gp_full_factorial_plan_v3",
        "stage": args.stage,
        "cells": list(CELLS),
        "seeds": list(seeds),
        "training_jobs": len(CELLS) * len(seeds),
        "export_jobs": len(jobs),
        "stage2_fold_method_rows": len(jobs) * args.num_folds * (10 + args.shuffle_count),
        "stage2b_oracle_jobs": len(seeds),
        "stage2b_healthy_jobs": len(seeds),
        "stage2b_groups": list(STAGE2B_GROUPS),
        "stage2b_group_seed_conditions": len(STAGE2B_GROUPS) * len(seeds),
        "stage2b_jobs": [
            {
                "seed": int(seed),
                "checkpoint_search_root": str(
                    _stage1_result_root(args.out_root) / _cell_names()["A00"] / f"seed_{seed}"
                ),
                "oracle": str(_stage2b_oracle_path(args.out_root, int(seed))),
                "posterior_cache": str(_cache_path(args.out_root, "A00", int(seed))),
                "healthy_bank": str(_stage2b_healthy_path(args.out_root, int(seed))),
                "result_dir": str(_stage2b_result_dir(args.out_root, int(seed))),
            }
            for seed in seeds
        ],
        "jobs": [
            {
                "cell": job["cell"],
                "seed": job["seed"],
                "checkpoint_search_root": str(
                    _stage1_result_root(args.out_root)
                    / _cell_names()[str(job["cell"])]
                    / f"seed_{job['seed']}"
                ),
                "cache": str(job["cache"]),
                "result_dir": str(job["result_dir"]),
            }
            for job in jobs
        ],
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root / "pipeline_plan.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-2A diagnostics and Stage-2B healthy-distribution interventions.")
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "train",
            "export",
            "evaluate",
            "summarize",
            "stage2b_oracle",
            "stage2b_healthy",
            "stage2b_evaluate",
            "stage2b_summarize",
        ),
        default="all",
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "prompt" / "cub.yaml")
    parser.add_argument("--graph-bundle", type=Path, default=DEFAULT_GRAPH_BUNDLE)
    parser.add_argument("--out-root", type=Path, default=ROOT / "output" / "graph_gp_full_factorial")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--gpu-groups", default="0")
    parser.add_argument("--nproc-per-trial", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--export-batch-size", type=int, default=64)
    parser.add_argument("--export-num-workers", type=int, default=4)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--shuffle-count", type=int, default=10)
    parser.add_argument("--alignment-topk", type=int, default=10)
    parser.add_argument("--stage2b-batch-size", type=int, default=32)
    parser.add_argument("--stage2b-build-ratio", type=float, default=0.8)
    parser.add_argument("--stage2b-oracle-epochs", type=int, default=10)
    parser.add_argument("--stage2b-oracle-lr", type=float, default=1e-2)
    parser.add_argument("--stage2b-align-weight", type=float, default=1.0)
    parser.add_argument("--stage2b-anchor-weight", type=float, default=0.1)
    parser.add_argument("--stage2b-healthy-epochs", type=int, default=10)
    parser.add_argument("--stage2b-healthy-lr", type=float, default=1e-3)
    parser.add_argument("--stage2b-generator-hidden-dim", type=int, default=256)
    parser.add_argument("--stage2b-classes-per-batch", type=int, default=8)
    parser.add_argument("--stage2b-samples-per-class", type=int, default=4)
    parser.add_argument("--stage2b-moment-samples", type=int, default=1)
    parser.add_argument("--stage2b-ce-weight", type=float, default=1.0)
    parser.add_argument("--stage2b-mean-weight", type=float, default=1.0)
    parser.add_argument("--stage2b-variance-weight", type=float, default=0.1)
    parser.add_argument("--stage2b-semantic-weight", type=float, default=1.0)
    parser.add_argument("--stage2b-prompt-visual-weight", type=float, default=0.1)
    parser.add_argument("--stage2b-prompt-semantic-weight", type=float, default=0.1)
    parser.add_argument("--stage2b-graph-weight", type=float, default=0.1)
    parser.add_argument("--stage2b-mu-delta-scale", type=float, default=1.0)
    parser.add_argument("--stage2b-logvar-delta-scale", type=float, default=1.0)
    parser.add_argument("--stage2b-gp-var-weight", type=float, default=1.0)
    parser.add_argument("--stage2b-support-var-weight", type=float, default=1.0)
    parser.add_argument("--stage2b-graph-key", default="method1_diff")
    parser.add_argument("--stage2b-prior-ridge", type=float, default=1e-3)
    parser.add_argument("--stage2b-fusion-alpha", type=float, default=1.0)
    parser.add_argument("--stage2b-energy-beta", type=float, default=1.0)
    parser.add_argument("--stage2b-candidate-temperature", type=float, default=1.0)
    parser.add_argument("--stage2b-candidate-topk", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.config_file = args.config_file.resolve()
    args.graph_bundle = args.graph_bundle.resolve()
    args.out_root = args.out_root.resolve()
    if Path(args.repo_root).resolve() != ROOT:
        parser.error(f"--repo-root must resolve to {ROOT}")
    if not args.config_file.is_file():
        parser.error(f"Config file does not exist: {args.config_file}")
    if args.stage in {"all", "evaluate", "stage2b_healthy"} and not args.graph_bundle.is_file():
        parser.error(f"Graph bundle does not exist: {args.graph_bundle}")
    if min(
        args.max_workers,
        args.cpu_workers,
        args.export_batch_size,
        args.num_folds,
        args.shuffle_count,
        args.alignment_topk,
        args.stage2b_batch_size,
        args.stage2b_oracle_epochs,
        args.stage2b_oracle_lr,
        args.stage2b_healthy_epochs,
        args.stage2b_healthy_lr,
        args.stage2b_generator_hidden_dim,
        args.stage2b_classes_per_batch,
        args.stage2b_samples_per_class,
        args.stage2b_moment_samples,
        args.stage2b_prior_ridge,
        args.stage2b_fusion_alpha,
        args.stage2b_candidate_temperature,
    ) <= 0:
        parser.error("Worker/batch/fold/shuffle/top-k values must be positive.")
    if not 0.0 < args.stage2b_build_ratio < 1.0:
        parser.error("--stage2b-build-ratio must be in (0,1).")
    if args.stage2b_candidate_topk < 0:
        parser.error("--stage2b-candidate-topk must be non-negative.")
    if min(
        args.stage2b_align_weight,
        args.stage2b_anchor_weight,
        args.stage2b_energy_beta,
        args.stage2b_ce_weight,
        args.stage2b_mean_weight,
        args.stage2b_variance_weight,
        args.stage2b_semantic_weight,
        args.stage2b_prompt_visual_weight,
        args.stage2b_prompt_semantic_weight,
        args.stage2b_graph_weight,
        args.stage2b_mu_delta_scale,
        args.stage2b_logvar_delta_scale,
        args.stage2b_gp_var_weight,
        args.stage2b_support_var_weight,
    ) < 0.0:
        parser.error("Stage-2B weights must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    gpu_groups = _parse_gpu_groups(args.gpu_groups)
    if args.max_workers > len(gpu_groups):
        raise ValueError("--max-workers must not exceed the number of --gpu-groups entries.")
    jobs = _job_specs(args.out_root, seeds)
    _write_plan(args, jobs, seeds)
    if args.dry_run:
        return
    stages = (
        (
            "train",
            "export",
            "evaluate",
            "summarize",
            "stage2b_oracle",
            "stage2b_healthy",
            "stage2b_evaluate",
            "stage2b_summarize",
        )
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(f"[pipeline] stage={stage}", flush=True)
        if stage == "train":
            _train(args, seeds)
        elif stage == "export":
            _export(args, jobs, gpu_groups)
        elif stage == "evaluate":
            _evaluate(args, jobs)
        elif stage == "summarize":
            _summarize(args, jobs, seeds)
        elif stage == "stage2b_oracle":
            _stage2b_oracle(args, seeds, gpu_groups)
        elif stage == "stage2b_healthy":
            _stage2b_healthy(args, seeds, gpu_groups)
        elif stage == "stage2b_evaluate":
            _stage2b_evaluate(args, seeds, gpu_groups)
        else:
            _stage2b_summarize(args, seeds)
    print(f"[pipeline] complete out_root={args.out_root}", flush=True)


if __name__ == "__main__":
    main()
