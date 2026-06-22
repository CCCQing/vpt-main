#!/usr/bin/env python3
"""Dispatch V5 semantic-relation GraphProbPrior temperature diagnostics.

This script does not compute model losses by itself. It expands a YAML search
space into many calls to diagnose_graph_prob_prior_temperatures.py, then merges
the generated per-trial JSON summaries into one table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID_CONFIG = "configs/graph_prob_prior/cub_v5_temperature_grid.yaml"
GRAPH_PROB_PRIOR_MODES = [
    "true_class_kl",
    "graph_conditioned_semantic_prior",
    "class_aggregate_moment",
    "class_aggregate_mmd",
    "factorized_latent",
    "dual_metric_semantic_distribution",
]
GRAPH_METHOD_ALIASES = {
    "method1_diff": "m1d",
    "method2_diff": "m2d",
    "method3_diff": "m3d",
    "llm_gate": "llmg",
    "method1_diff_llm_gate": "m1dlg",
    "method2_diff_llm_gate": "m2dlg",
    "method3_diff_llm_gate": "m3dlg",
}
MODE_ALIASES = {
    "true_class_kl": "tckl",
    "graph_conditioned_semantic_prior": "gcsp",
    "class_aggregate_moment": "cam",
    "class_aggregate_mmd": "mmd",
    "factorized_latent": "fact",
    "dual_metric_semantic_distribution": "dual",
}
SEARCH_STAGES = {"temperature", "other", "all"}


def _default_dist_backend() -> str:
    return "gloo" if os.name == "nt" else "nccl"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_gpu_groups(raw_groups: str, raw_gpus: str) -> List[str]:
    if str(raw_groups).strip():
        return [item.strip() for item in str(raw_groups).split(";") if item.strip()]
    if str(raw_gpus).strip():
        return _parse_csv(raw_gpus)
    return [""]


def _group_nproc(gpu_group: str, fallback_nproc: int) -> int:
    if fallback_nproc > 0:
        return int(fallback_nproc)
    if not gpu_group:
        return 1
    return len([item for item in str(gpu_group).split(",") if item.strip()])


def _patch_ddp_forward_missing_attrs() -> None:
    import torch

    ddp_cls = torch.nn.parallel.DistributedDataParallel
    if getattr(ddp_cls, "_gpp_forward_missing_attrs", False):
        return
    original_getattr = ddp_cls.__getattr__

    def forwarded_getattr(self, name):
        try:
            return original_getattr(self, name)
        except AttributeError as exc:
            module = original_getattr(self, "module")
            if hasattr(module, name):
                return getattr(module, name)
            raise exc

    ddp_cls.__getattr__ = forwarded_getattr
    ddp_cls._gpp_forward_missing_attrs = True


def _diagnose_with_ddp_attr_forward(argv: Sequence[str], _unused: Any = None) -> None:
    _patch_ddp_forward_missing_attrs()
    from src.tools import diagnose_graph_prob_prior_temperatures as diagnose

    old_argv = sys.argv
    try:
        sys.argv = [str(Path(diagnose.__file__).resolve())] + list(argv)
        diagnose.main()
    finally:
        sys.argv = old_argv


def _ddp_diagnose_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser("grid_search_graph_prob_prior_v5_temperatures ddp diagnose")
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--dist-backend", default=_default_dist_backend(), choices=["nccl", "gloo"])
    parser.add_argument("--dist-url", default="")
    known, diagnose_argv = parser.parse_known_args(list(argv))
    if known.nproc_per_node <= 1:
        raise ValueError("--nproc-per-node must be greater than 1 in DDP diagnose mode.")
    if not diagnose_argv:
        raise ValueError("Missing diagnose_graph_prob_prior_temperatures.py arguments after DDP launcher options.")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch.multiprocessing as mp

    from src.utils import distributed as du

    init_method = known.dist_url or "tcp://127.0.0.1:{}".format(_pick_free_port())
    mp.spawn(
        du.run,
        nprocs=int(known.nproc_per_node),
        args=(
            int(known.nproc_per_node),
            _diagnose_with_ddp_attr_forward,
            init_method,
            0,
            1,
            str(known.dist_backend),
            list(diagnose_argv),
            None,
        ),
        join=True,
    )


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Grid config must be a YAML mapping: {path}")
    return data


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _value_to_opt(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return str(value)


def _mapping_to_opts(mapping: Mapping[str, Any]) -> List[str]:
    opts: List[str] = []
    for key, value in mapping.items():
        opts.extend([str(key), _value_to_opt(value)])
    return opts


def _validate_extra_opts(opts: Sequence[str]) -> List[str]:
    opts = list(opts)
    if opts and opts[0] == "--":
        opts = opts[1:]
    if len(opts) % 2 != 0:
        raise ValueError("Extra opts must use KEY VALUE pairs.")
    return opts


def _sanitize(value: Any) -> str:
    text = str(value)
    for old, new in [
        ("MODEL.", ""),
        ("SEMANTIC_GRAPH.", "sg_"),
        ("GRAPH_PROB_PRIOR.", "gpp_"),
        ("PROMPT.DISTRIBUTOR.", "dist_"),
        (".", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("-", "m"),
    ]:
        text = text.replace(old, new)
    return text.replace(" ", "_")


def _cartesian_grid(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [_as_list(grid[key]) for key in keys]
    for key, values in zip(keys, value_lists):
        if not values:
            raise ValueError(f"Grid key has no values: {key}")
    combos = []
    for values in itertools.product(*value_lists):
        combos.append({key: value for key, value in zip(keys, values)})
    return combos


def _stage_specs(spec: Mapping[str, Any], selected_stage: str) -> List[Dict[str, Any]]:
    base_fixed = dict(spec.get("FIXED_OPTS", {}) or {})
    stages: List[Dict[str, Any]] = []
    if selected_stage in {"temperature", "all"}:
        grid = spec.get("TEMPERATURE_GRID", spec.get("GRID", {}) or {}) or {}
        fixed = dict(base_fixed)
        fixed.update(dict(spec.get("TEMPERATURE_FIXED_OPTS", {}) or {}))
        if grid or selected_stage == "temperature":
            stages.append({"stage": "temperature", "fixed": fixed, "grid": grid})
    if selected_stage in {"other", "all"}:
        grid = spec.get("OTHER_GRID", {}) or {}
        fixed = dict(base_fixed)
        fixed.update(dict(spec.get("OTHER_FIXED_OPTS", {}) or {}))
        if grid or selected_stage == "other":
            stages.append({"stage": "other", "fixed": fixed, "grid": grid})
    return stages


def _load_graph_methods(grid_cfg: Mapping[str, Any], selected: str) -> List[str]:
    external = grid_cfg.get("EXTERNAL_GRAPH", {})
    methods = [str(item) for item in _as_list(external.get("METHODS"))]
    if not methods:
        raise ValueError("Grid config EXTERNAL_GRAPH.METHODS must list at least one graph key.")
    if selected and selected.lower() != "all":
        wanted = _parse_csv(selected)
        missing = [item for item in wanted if item not in methods]
        if missing:
            raise ValueError(f"Unknown --graph-methods values {missing}; expected subset of {methods}.")
        return wanted
    return methods


def _load_modes(grid_cfg: Mapping[str, Any], selected: str) -> List[str]:
    mode_space = grid_cfg.get("MODE_SEARCH_SPACE", {})
    if not isinstance(mode_space, dict) or not mode_space:
        raise ValueError("Grid config MODE_SEARCH_SPACE must define at least one mode.")
    modes = [str(mode) for mode in mode_space.keys()]
    bad = [mode for mode in modes if mode not in GRAPH_PROB_PRIOR_MODES]
    if bad:
        raise ValueError(f"Unsupported modes in config: {bad}")
    if selected and selected.lower() != "all":
        wanted = _parse_csv(selected)
        missing = [item for item in wanted if item not in modes]
        if missing:
            raise ValueError(f"Unknown --modes values {missing}; expected subset of {modes}.")
        return wanted
    return modes


def _validate_external_graph_file(repo_root: Path, grid_cfg: Mapping[str, Any], graph_methods: Sequence[str]) -> Path:
    external = grid_cfg.get("EXTERNAL_GRAPH", {})
    raw_path = str(external.get("PATH", "")).strip()
    if not raw_path:
        raise ValueError("Grid config EXTERNAL_GRAPH.PATH must be set.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise FileNotFoundError(f"External graph npz not found: {path}")
    payload = np.load(str(path), allow_pickle=False)
    missing = [method for method in graph_methods if method not in payload.files]
    if missing:
        raise KeyError(f"External graph keys missing from {path}: {missing}; available={payload.files}.")
    for method in graph_methods:
        matrix = payload[method]
        if matrix.shape != (200, 200):
            raise RuntimeError(f"External graph key '{method}' must be [200,200], got {matrix.shape}.")
    return path


def _trial_name(index: int, stage: str, graph_method: str, mode: str, combo: Mapping[str, Any]) -> str:
    combo_json = json.dumps(dict(combo), sort_keys=True, ensure_ascii=True)
    combo_hash = hashlib.md5(combo_json.encode("utf-8")).hexdigest()[:8]
    graph_alias = GRAPH_METHOD_ALIASES.get(str(graph_method), _sanitize(graph_method)[:12])
    mode_alias = MODE_ALIASES.get(str(mode), _sanitize(mode)[:12])
    return f"t{index:04d}_{stage[:4]}_{graph_alias}_{mode_alias}_{combo_hash}"


def _build_trials(
    repo_root: Path,
    grid_cfg: Mapping[str, Any],
    graph_methods: Sequence[str],
    modes: Sequence[str],
    config_file: str,
    diagnose_script: str,
    python_bin: str,
    out_root: Path,
    max_batches: int,
    no_train_step: bool,
    extra_opts: Sequence[str],
    stage: str,
) -> List[Dict[str, Any]]:
    fixed_opts = dict(grid_cfg.get("FIXED_OPTS", {}) or {})
    mode_space = grid_cfg["MODE_SEARCH_SPACE"]
    graph_path = str(grid_cfg["EXTERNAL_GRAPH"]["PATH"])
    trials: List[Dict[str, Any]] = []
    index = 0
    for graph_method in graph_methods:
        for mode in modes:
            spec = mode_space[mode] or {}
            for stage_spec in _stage_specs(spec, stage):
                stage_name = str(stage_spec["stage"])
                mode_fixed = dict(stage_spec["fixed"])
                combos = _cartesian_grid(stage_spec.get("grid", {}) or {})
                for combo_index, combo in enumerate(combos):
                    overrides: Dict[str, Any] = {}
                    overrides.update(fixed_opts)
                    overrides.update(mode_fixed)
                    overrides.update(
                        {
                            "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_PATH": graph_path,
                            "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_KEY": graph_method,
                            "MODEL.GRAPH_PROB_PRIOR.MODE": mode,
                        }
                    )
                    overrides.update(combo)
                    trial_name = _trial_name(index, stage_name, graph_method, mode, combo)
                    output_dir = out_root / stage_name / graph_method / mode / trial_name
                    cmd = [
                        python_bin,
                        diagnose_script,
                        "--config-file",
                        config_file,
                        "--max-batches",
                        str(max_batches),
                        "--output-dir",
                        str(output_dir),
                    ]
                    if no_train_step:
                        cmd.append("--no-train-step")
                    cmd.extend(_mapping_to_opts(overrides))
                    cmd.extend(extra_opts)
                    trials.append(
                        {
                            "trial_index": index,
                            "stage": stage_name,
                            "combo_index": combo_index,
                            "trial_name": trial_name,
                            "graph_method": graph_method,
                            "mode": mode,
                            "combo": dict(combo),
                            "overrides": overrides,
                            "output_dir": str(output_dir),
                            "stdout_path": str(output_dir / "launcher_stdout.txt"),
                            "cmd": cmd,
                            "repo_root": str(repo_root),
                        }
                    )
                    index += 1
    return trials


def _summary_payload(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "graph_prob_prior_temperature_diagnosis.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_result(
    trial: Mapping[str, Any],
    returncode: int,
    gpu_id: str = "",
    command: Optional[Sequence[str]] = None,
    num_gpus: int = 1,
) -> Dict[str, Any]:
    command = list(command) if command is not None else list(trial["cmd"])
    row: Dict[str, Any] = {
        "trial_index": trial["trial_index"],
        "stage": trial.get("stage", ""),
        "trial_name": trial["trial_name"],
        "graph_method": trial["graph_method"],
        "mode": trial["mode"],
        "combo_index": trial["combo_index"],
        "returncode": returncode,
        "gpu": gpu_id,
        "num_gpus": int(num_gpus),
        "output_dir": trial["output_dir"],
        "stdout_path": trial["stdout_path"],
        "command": subprocess.list2cmdline(command),
    }
    for key, value in trial["combo"].items():
        row[key] = value
    payload = _summary_payload(Path(str(trial["output_dir"])))
    if payload:
        row["num_batches"] = payload.get("num_batches", "")
        for key, value in (payload.get("temperatures") or {}).items():
            row[f"temperature_{key}"] = value
        for key, value in (payload.get("summary") or {}).items():
            if isinstance(value, (int, float)):
                row[key] = value
    return row


def _row_float(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _target_penalty(row: Mapping[str, Any], key: str, target: float, tolerance: float, weight: float) -> float:
    value = _row_float(row, key)
    if value is None:
        return 0.0
    return float(weight) * abs(value - float(target)) / max(float(tolerance), 1e-12)


def _upper_penalty(row: Mapping[str, Any], key: str, limit: float, weight: float) -> float:
    value = _row_float(row, key)
    if value is None:
        return 0.0
    return float(weight) * max(0.0, value - float(limit)) / max(abs(float(limit)), 1e-12)


def _lower_is_better(row: Mapping[str, Any], key: str, scale: float, weight: float) -> float:
    value = _row_float(row, key)
    if value is None:
        return 0.0
    return float(weight) * max(0.0, value) / max(float(scale), 1e-12)


def _higher_is_better(row: Mapping[str, Any], key: str, target: float, weight: float) -> float:
    value = _row_float(row, key)
    if value is None:
        return 0.0
    return float(weight) * max(0.0, float(target) - value) / max(abs(float(target)), 1e-12)


def _with_monitor_scores(row: MutableMapping[str, Any]) -> Dict[str, Any]:
    scored = dict(row)
    if int(scored.get("returncode", 0)) != 0:
        scored["monitor_score"] = 1.0e9
        scored["selection_score"] = 1.0e9
        return scored

    score = 0.0
    # TAU_GRAPH: old results show 0.05 was too sharp and 0.20 too flat; target a middle-entropy neighbor distribution.
    score += _target_penalty(scored, "graph_prob_prior_monitor_tau_graph_neighbor_entropy_norm_mean", 0.55, 0.30, 1.2)
    score += _target_penalty(scored, "graph_prob_prior_monitor_neighbor_entropy_norm_mean", 0.55, 0.30, 1.2)
    score += _target_penalty(scored, "graph_prob_prior_monitor_tau_graph_neighbor_top1_mean", 0.35, 0.30, 0.8)
    score += _target_penalty(scored, "graph_prob_prior_monitor_neighbor_top1_mean", 0.35, 0.30, 0.8)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_neighbor_hubness_gini", 1.0, 0.5)

    # TAU_LATENT: avoid all-class latent probabilities being almost uniform, but also avoid one-hot collapse.
    score += _target_penalty(scored, "graph_prob_prior_monitor_tau_latent_entropy_norm_mean", 0.75, 0.25, 1.0)
    score += _target_penalty(scored, "graph_prob_prior_monitor_latent_prob_entropy_norm_mean", 0.75, 0.25, 1.0)
    score += _higher_is_better(scored, "graph_prob_prior_monitor_posterior_true_rank_top1", 0.20, 1.0)
    score += _higher_is_better(scored, "graph_prob_prior_monitor_posterior_kl_margin_positive_ratio", 0.50, 0.8)

    # MMD_SIGMA: old MMD_SIGMA=16 underflowed; 64 was too close to all-ones; target an informative mid kernel.
    score += _target_penalty(scored, "graph_prob_prior_monitor_mmd_kernel_mean", 0.50, 0.35, 1.2)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_mmd_kernel_saturation_low_ratio", 1.0, 1.0)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_mmd_kernel_saturation_high_ratio", 1.0, 1.0)

    # Prior shape and graph-specific pathology monitors from graph_prob_prior_monitors.py.
    score += _lower_is_better(scored, "graph_prob_prior_monitor_prior_overlap_risk_rate", 1.0, 1.0)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_false_high_prior_relation_still_gt_0_9_count", 20.0, 0.8)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_prior_gzsl_unseen_to_seen_bias_risk_mean", 1.0, 0.5)

    # Dual mode: prefer P+/P- separation and fewer hard-negative violations.
    score += _lower_is_better(scored, "graph_prob_prior_monitor_dual_sample_beta_hardneg_violation_rate", 1.0, 1.0)
    score += _higher_is_better(scored, "graph_prob_prior_monitor_dual_pos_neg_js_divergence_mean", 0.20, 0.6)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_dual_alpha_beta_conflict_rate", 1.0, 0.8)

    # Keep auxiliary loss from overpowering classification when the monitor is available.
    score += _upper_penalty(scored, "graph_prob_prior_monitor_loss_weighted_gpp_to_main_loss_ratio", 0.10, 1.0)

    loss = _row_float(scored, "graph_prob_prior_match_loss")
    loss_term = 0.0 if loss is None else 0.01 * np.log1p(max(loss, 0.0))
    scored["monitor_score"] = float(score)
    scored["selection_score"] = float(score + loss_term)
    return scored


def _rank_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    scored = [_with_monitor_scores(dict(row)) for row in rows]
    return sorted(
        scored,
        key=lambda row: (
            int(row.get("returncode", 0)) != 0,
            float(row.get("selection_score", 1.0e9)),
            str(row.get("stage", "")),
            str(row.get("graph_method", "")),
            str(row.get("mode", "")),
            int(row.get("trial_index", 0)),
        ),
    )


def _best_rows(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for row in _rank_rows(rows):
        if int(row.get("returncode", 0)) != 0:
            continue
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        if key not in best:
            best[key] = dict(row)
    return sorted(best.values(), key=lambda row: tuple(str(row.get(k, "")) for k in group_keys))


def _build_run_command(
    trial: Mapping[str, Any],
    python_bin: str,
    gpu_group: str,
    nproc_per_trial: int,
    dist_backend: str,
) -> tuple[List[str], int]:
    nproc = _group_nproc(gpu_group, nproc_per_trial)
    if nproc <= 1:
        return list(trial["cmd"]), 1

    diagnose_cmd = list(trial["cmd"])
    diagnose_args = diagnose_cmd[2:]
    return (
        [
            python_bin,
            str(Path(__file__).resolve()),
            "--ddp-diagnose",
            "--nproc-per-node",
            str(nproc),
            "--dist-backend",
            str(dist_backend),
        ]
        + diagnose_args
        + [
            "NUM_GPUS",
            str(nproc),
        ],
        nproc,
    )


def _run_trial(
    trial: Mapping[str, Any],
    gpu_id: str = "",
    python_bin: str = sys.executable,
    nproc_per_trial: int = 0,
    dist_backend: str = "",
) -> Dict[str, Any]:
    output_dir = Path(str(trial["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_cmd, nproc = _build_run_command(
        trial,
        python_bin=python_bin,
        gpu_group=gpu_id,
        nproc_per_trial=int(nproc_per_trial),
        dist_backend=dist_backend or _default_dist_backend(),
    )
    with Path(str(trial["stdout_path"])).open("w", encoding="utf-8", errors="replace") as stdout:
        proc = subprocess.run(
            run_cmd,
            cwd=str(trial["repo_root"]),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
    return _flatten_result(trial, int(proc.returncode), gpu_id=gpu_id, command=run_cmd, num_gpus=nproc)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-dispatch GraphProbPrior V5 temperature diagnostics.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--grid-config", default=DEFAULT_GRID_CONFIG)
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--config-file", default="")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--graph-methods", default="all")
    parser.add_argument("--modes", default="all")
    parser.add_argument("--stage", default="", choices=["", "temperature", "other", "all"])
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids for parallel trials.")
    parser.add_argument(
        "--gpu-groups",
        default="",
        help="Semicolon-separated GPU groups. Example: '0,5' runs one DDP trial on GPUs 0 and 5; '0,1;2,3' runs two DDP trials.",
    )
    parser.add_argument(
        "--nproc-per-trial",
        type=int,
        default=0,
        help="DDP process count for each trial. Default: GPU count in each --gpu-groups item; 0 keeps single-GPU behavior for --gpus.",
    )
    parser.add_argument("--dist-backend", default=_default_dist_backend(), choices=["nccl", "gloo"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N expanded trials; 0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-train-step", action="store_true", help="Override YAML NO_TRAIN_STEP to true.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra KEY VALUE config overrides appended to every child run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    grid_config = Path(args.grid_config)
    if not grid_config.is_absolute():
        grid_config = repo_root / grid_config
    grid_cfg = _read_yaml(grid_config)

    config_file = args.config_file or str(grid_cfg.get("BASE_CONFIG_FILE", "configs/prompt/cub.yaml"))
    diagnose_script = str(grid_cfg.get("DIAGNOSE_SCRIPT", "src/tools/diagnose_graph_prob_prior_temperatures.py"))
    python_bin = str(args.python_bin or grid_cfg.get("PYTHON_BIN") or sys.executable)
    out_root = Path(args.out_root or str(grid_cfg.get("OUTPUT_DIR", "output/gpp_v5_temperature_grid")))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    max_batches = int(args.max_batches if args.max_batches is not None else int(grid_cfg.get("MAX_BATCHES", 111)))
    no_train_step = bool(grid_cfg.get("NO_TRAIN_STEP", False)) or bool(args.no_train_step)
    extra_opts = _validate_extra_opts(args.opts)
    stage = str(args.stage or grid_cfg.get("STAGE", "temperature")).lower()
    if stage not in SEARCH_STAGES:
        raise ValueError(f"--stage must be one of {sorted(SEARCH_STAGES)}, got {stage}.")

    graph_methods = _load_graph_methods(grid_cfg, args.graph_methods)
    modes = _load_modes(grid_cfg, args.modes)
    graph_npz = _validate_external_graph_file(repo_root, grid_cfg, graph_methods)
    gpu_groups = _parse_gpu_groups(args.gpu_groups, args.gpus)
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    if args.max_workers > len(gpu_groups):
        raise ValueError("--max-workers must not exceed the number of GPU groups.")
    for gpu_group in gpu_groups:
        nproc = _group_nproc(gpu_group, int(args.nproc_per_trial))
        visible_gpu_count = len([item for item in str(gpu_group).split(",") if item.strip()])
        if gpu_group and nproc > 1 and nproc != visible_gpu_count:
            raise ValueError("--nproc-per-trial must match each --gpu-groups item size when CUDA_VISIBLE_DEVICES is set.")

    trials = _build_trials(
        repo_root=repo_root,
        grid_cfg=grid_cfg,
        graph_methods=graph_methods,
        modes=modes,
        config_file=config_file,
        diagnose_script=diagnose_script,
        python_bin=python_bin,
        out_root=out_root,
        max_batches=max_batches,
        no_train_step=no_train_step,
        extra_opts=extra_opts,
        stage=stage,
    )
    if args.limit > 0:
        trials = trials[: int(args.limit)]

    out_root.mkdir(parents=True, exist_ok=True)
    stage_suffix = f"_{stage}"
    commands_path = out_root / f"commands{stage_suffix}.txt"
    with commands_path.open("w", encoding="utf-8") as handle:
        for trial in trials:
            run_cmd, _ = _build_run_command(
                trial,
                python_bin=python_bin,
                gpu_group=gpu_groups[0] if gpu_groups else "",
                nproc_per_trial=int(args.nproc_per_trial),
                dist_backend=str(args.dist_backend),
            )
            handle.write(subprocess.list2cmdline(run_cmd) + "\n")

    search_space = {
        "grid_config": str(grid_config),
        "base_config_file": config_file,
        "diagnose_script": diagnose_script,
        "python_bin": python_bin,
        "external_graph_npz": str(graph_npz),
        "graph_methods": graph_methods,
        "modes": modes,
        "max_batches": max_batches,
        "no_train_step": no_train_step,
        "stage": stage,
        "total_trials": len(trials),
        "dry_run": bool(args.dry_run),
        "gpu_groups": gpu_groups,
        "nproc_per_trial": int(args.nproc_per_trial),
        "dist_backend": str(args.dist_backend),
        "max_workers": int(args.max_workers),
        "extra_opts": extra_opts,
        "mode_search_space": grid_cfg.get("MODE_SEARCH_SPACE", {}),
        "fixed_opts": grid_cfg.get("FIXED_OPTS", {}),
        "commands_path": str(commands_path),
        "ranking": {
            "selection_score": "monitor_score + 0.01*log1p(graph_prob_prior_match_loss)",
            "notes": [
                "lower selection_score is better",
                "score prefers balanced graph-neighbor entropy/top1, informative latent probabilities, non-saturated MMD kernel, low prior overlap, low false-high residue, low dual beta violation, and reasonable weighted loss scale",
            ],
        },
    }
    _write_json(out_root / "search_space.json", search_space)

    if args.dry_run:
        rows = []
        for idx, trial in enumerate(trials):
            gpu = gpu_groups[idx % len(gpu_groups)] if gpu_groups else ""
            run_cmd, nproc = _build_run_command(
                trial,
                python_bin=python_bin,
                gpu_group=gpu,
                nproc_per_trial=int(args.nproc_per_trial),
                dist_backend=str(args.dist_backend),
            )
            rows.append(_flatten_result(trial, returncode=-1, gpu_id=gpu, command=run_cmd, num_gpus=nproc))
    elif args.max_workers == 1:
        rows = []
        for idx, trial in enumerate(trials):
            gpu = gpu_groups[idx % len(gpu_groups)] if gpu_groups else ""
            nproc = _group_nproc(gpu, int(args.nproc_per_trial))
            print(f"[{idx + 1}/{len(trials)}] {trial['trial_name']} gpu={gpu} nproc={nproc}", flush=True)
            rows.append(
                _run_trial(
                    trial,
                    gpu_id=gpu,
                    python_bin=python_bin,
                    nproc_per_trial=int(args.nproc_per_trial),
                    dist_backend=str(args.dist_backend),
                )
            )
    else:
        rows = []
        worker_count = int(args.max_workers)
        worker_gpus = gpu_groups[:worker_count] if gpu_groups else [""] * worker_count
        worker_trials = [[] for _ in range(worker_count)]
        for idx, trial in enumerate(trials):
            worker_trials[idx % worker_count].append(trial)

        def _run_worker(worker_idx: int, gpu: str, assigned_trials: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
            worker_rows = []
            for local_idx, trial in enumerate(assigned_trials, start=1):
                print(
                    f"[worker {worker_idx + 1}/{worker_count} gpu={gpu}] "
                    f"started {local_idx}/{len(assigned_trials)} {trial['trial_name']}",
                    flush=True,
                )
                result = _run_trial(
                    trial,
                    gpu_id=gpu,
                    python_bin=python_bin,
                    nproc_per_trial=int(args.nproc_per_trial),
                    dist_backend=str(args.dist_backend),
                )
                print(
                    f"[worker {worker_idx + 1}/{worker_count} gpu={gpu}] "
                    f"finished {local_idx}/{len(assigned_trials)} {trial['trial_name']} "
                    f"returncode={result['returncode']}",
                    flush=True,
                )
                worker_rows.append(result)
            return worker_rows

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_run_worker, worker_idx, worker_gpus[worker_idx], assigned_trials)
                for worker_idx, assigned_trials in enumerate(worker_trials)
            ]
            finished = 0
            for future in as_completed(futures):
                worker_rows = future.result()
                rows.extend(worker_rows)
                finished += len(worker_rows)
                print(f"[{finished}/{len(trials)}] collected worker results", flush=True)
        rows.sort(key=lambda row: int(row["trial_index"]))

    ranked_rows = _rank_rows(rows)
    best_by_stage_graph_mode = _best_rows(ranked_rows, ["stage", "graph_method", "mode"])
    best_by_stage_mode = _best_rows(ranked_rows, ["stage", "mode"])
    _write_csv(out_root / "summary.csv", ranked_rows)
    _write_json(out_root / "summary.json", {"search_space": search_space, "rows": ranked_rows})
    _write_csv(out_root / "ranked_summary.csv", ranked_rows)
    _write_json(out_root / "ranked_summary.json", {"search_space": search_space, "rows": ranked_rows})
    _write_csv(out_root / "best_by_stage_graph_mode.csv", best_by_stage_graph_mode)
    _write_json(out_root / "best_by_stage_graph_mode.json", {"rows": best_by_stage_graph_mode})
    _write_csv(out_root / "best_by_stage_mode.csv", best_by_stage_mode)
    _write_json(out_root / "best_by_stage_mode.json", {"rows": best_by_stage_mode})
    _write_csv(out_root / f"summary{stage_suffix}.csv", ranked_rows)
    _write_json(out_root / f"summary{stage_suffix}.json", {"search_space": search_space, "rows": ranked_rows})
    _write_csv(out_root / f"ranked_summary{stage_suffix}.csv", ranked_rows)
    _write_json(out_root / f"ranked_summary{stage_suffix}.json", {"search_space": search_space, "rows": ranked_rows})
    _write_csv(out_root / f"best_by_stage_graph_mode{stage_suffix}.csv", best_by_stage_graph_mode)
    _write_json(out_root / f"best_by_stage_graph_mode{stage_suffix}.json", {"rows": best_by_stage_graph_mode})
    _write_csv(out_root / f"best_by_stage_mode{stage_suffix}.csv", best_by_stage_mode)
    _write_json(out_root / f"best_by_stage_mode{stage_suffix}.json", {"rows": best_by_stage_mode})
    failures = [row for row in rows if int(row.get("returncode", 0)) not in {0, -1}]
    if failures:
        _write_json(out_root / "failures.json", {"failures": failures})
        raise SystemExit(f"{len(failures)} trials failed; see {out_root / 'failures.json'}")
    print(f"wrote {out_root / 'summary.csv'}")
    print(f"wrote {out_root / 'summary.json'}")
    print(f"wrote {out_root / 'ranked_summary.csv'}")
    print(f"wrote {out_root / 'best_by_stage_graph_mode.csv'}")
    print(f"wrote {out_root / 'best_by_stage_mode.csv'}")
    print(f"wrote {out_root / f'ranked_summary{stage_suffix}.csv'}")
    print(f"wrote {out_root / f'best_by_stage_graph_mode{stage_suffix}.csv'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ddp-diagnose":
        _ddp_diagnose_main(sys.argv[2:])
    else:
        main()
