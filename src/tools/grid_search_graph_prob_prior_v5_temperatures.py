#!/usr/bin/env python3
"""Dispatch V5 semantic-relation GraphProbPrior temperature diagnostics.

This script does not compute model losses by itself. It expands a YAML search
space into many calls to diagnose_graph_prob_prior_temperatures.py, then merges
the generated per-trial JSON summaries into one table.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

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
]


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


def _trial_name(index: int, graph_method: str, mode: str, combo: Mapping[str, Any]) -> str:
    parts = [f"t{index:04d}", _sanitize(graph_method), _sanitize(mode)]
    for key, value in combo.items():
        short_key = str(key).split(".")[-1].lower()
        parts.append(f"{_sanitize(short_key)}{_sanitize(value)}")
    return "_".join(parts)


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
) -> List[Dict[str, Any]]:
    fixed_opts = dict(grid_cfg.get("FIXED_OPTS", {}) or {})
    mode_space = grid_cfg["MODE_SEARCH_SPACE"]
    graph_path = str(grid_cfg["EXTERNAL_GRAPH"]["PATH"])
    trials: List[Dict[str, Any]] = []
    index = 0
    for graph_method in graph_methods:
        for mode in modes:
            spec = mode_space[mode] or {}
            mode_fixed = dict(spec.get("FIXED_OPTS", {}) or {})
            combos = _cartesian_grid(spec.get("GRID", {}) or {})
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
                trial_name = _trial_name(index, graph_method, mode, combo)
                output_dir = out_root / graph_method / mode / trial_name
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


def _flatten_result(trial: Mapping[str, Any], returncode: int, gpu_id: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial_index": trial["trial_index"],
        "trial_name": trial["trial_name"],
        "graph_method": trial["graph_method"],
        "mode": trial["mode"],
        "combo_index": trial["combo_index"],
        "returncode": returncode,
        "gpu": gpu_id,
        "output_dir": trial["output_dir"],
        "stdout_path": trial["stdout_path"],
        "command": subprocess.list2cmdline(list(trial["cmd"])),
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


def _run_trial(trial: Mapping[str, Any], gpu_id: str = "") -> Dict[str, Any]:
    output_dir = Path(str(trial["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with Path(str(trial["stdout_path"])).open("w", encoding="utf-8", errors="replace") as stdout:
        proc = subprocess.run(
            list(trial["cmd"]),
            cwd=str(trial["repo_root"]),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
    return _flatten_result(trial, int(proc.returncode), gpu_id=gpu_id)


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
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids for parallel trials.")
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

    graph_methods = _load_graph_methods(grid_cfg, args.graph_methods)
    modes = _load_modes(grid_cfg, args.modes)
    graph_npz = _validate_external_graph_file(repo_root, grid_cfg, graph_methods)
    gpus = _parse_csv(args.gpus)
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    if gpus and args.max_workers > len(gpus):
        raise ValueError("--max-workers must not exceed the number of --gpus.")

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
    )
    if args.limit > 0:
        trials = trials[: int(args.limit)]

    out_root.mkdir(parents=True, exist_ok=True)
    commands_path = out_root / "commands.txt"
    with commands_path.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(subprocess.list2cmdline(list(trial["cmd"])) + "\n")

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
        "total_trials": len(trials),
        "dry_run": bool(args.dry_run),
        "gpus": gpus,
        "max_workers": int(args.max_workers),
        "extra_opts": extra_opts,
        "mode_search_space": grid_cfg.get("MODE_SEARCH_SPACE", {}),
        "fixed_opts": grid_cfg.get("FIXED_OPTS", {}),
        "commands_path": str(commands_path),
    }
    _write_json(out_root / "search_space.json", search_space)

    if args.dry_run:
        rows = [_flatten_result(trial, returncode=-1, gpu_id="") for trial in trials]
    elif args.max_workers == 1:
        rows = []
        for idx, trial in enumerate(trials):
            gpu = gpus[idx % len(gpus)] if gpus else ""
            print(f"[{idx + 1}/{len(trials)}] {trial['trial_name']} gpu={gpu}", flush=True)
            rows.append(_run_trial(trial, gpu_id=gpu))
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            future_to_trial = {}
            for idx, trial in enumerate(trials):
                gpu = gpus[idx % len(gpus)] if gpus else ""
                future = executor.submit(_run_trial, trial, gpu)
                future_to_trial[future] = trial
            for done_idx, future in enumerate(as_completed(future_to_trial), start=1):
                trial = future_to_trial[future]
                print(f"[{done_idx}/{len(trials)}] finished {trial['trial_name']}", flush=True)
                rows.append(future.result())
        rows.sort(key=lambda row: int(row["trial_index"]))

    _write_csv(out_root / "summary.csv", rows)
    _write_json(out_root / "summary.json", {"search_space": search_space, "rows": rows})
    failures = [row for row in rows if int(row.get("returncode", 0)) not in {0, -1}]
    if failures:
        _write_json(out_root / "failures.json", {"failures": failures})
        raise SystemExit(f"{len(failures)} trials failed; see {out_root / 'failures.json'}")
    print(f"wrote {out_root / 'summary.csv'}")
    print(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
