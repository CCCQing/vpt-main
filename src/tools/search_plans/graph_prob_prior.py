#!/usr/bin/env python3
"""Define and run the GraphProbPrior parameter-search plan."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.fields import GPP_MONITOR_ALIASES
from src.configs.config import get_cfg
from src.tools.parameter_search.search_scheduler import (
    SearchScheduler,
    build_run_command,
    default_dist_backend,
    mapping_to_opts,
    parse_csv,
    parse_gpu_groups,
    validate_extra_opts,
    validate_gpu_groups,
    write_csv,
    write_json,
)

GRAPH_METHOD_ALIASES = {
    "method1_diff": "m1d",
    "method2_diff": "m2d",
    "method3_diff": "m3d",
    "llm_gate": "llmg",
    "method1_diff_llm_gate": "m1dlg",
    "method2_diff_llm_gate": "m2dlg",
    "method3_diff_llm_gate": "m3dlg",
}


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


def _sanitize(value: Any) -> str:
    text = str(value)
    for old, new in [
        ("MODEL.", ""),
        ("GRAPH_INPUT.", "gi_"),
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


def _configured_stage_names(grid_cfg: Mapping[str, Any]) -> List[str]:
    raw_order = grid_cfg.get("STAGE_ORDER", grid_cfg.get("SEARCH_STAGES", None))
    if raw_order is not None:
        order = [str(item).lower() for item in _as_list(raw_order)]
        if not order:
            raise ValueError("STAGE_ORDER / SEARCH_STAGES is set but empty.")
        return order

    search_space = grid_cfg.get("GRAPH_GP_SEARCH_SPACE", {})
    stage_map = search_space.get("STAGES", {}) if isinstance(search_space, Mapping) else {}
    if not isinstance(stage_map, Mapping) or not stage_map:
        raise ValueError("GRAPH_GP_SEARCH_SPACE.STAGES must define at least one stage.")
    return [str(stage).lower() for stage in stage_map.keys()]


def _configured_stage_groups(grid_cfg: Mapping[str, Any], stage_order: Sequence[str]) -> List[List[str]]:
    raw_groups = grid_cfg.get("STAGE_GROUP_ORDER", None)
    if raw_groups is None:
        return [[str(stage).lower()] for stage in stage_order]

    valid = {str(stage).lower() for stage in stage_order}
    groups: List[List[str]] = []
    seen = set()
    for raw_group in _as_list(raw_groups):
        group_items = raw_group if isinstance(raw_group, (list, tuple)) else [raw_group]
        group: List[str] = []
        for raw_stage in group_items:
            stage = str(raw_stage).lower()
            if stage not in valid:
                raise ValueError(f"Unknown stage in STAGE_GROUP_ORDER: {stage}; expected subset of {list(stage_order)}.")
            if stage in seen:
                raise ValueError(f"Duplicate stage in STAGE_GROUP_ORDER: {stage}.")
            seen.add(stage)
            group.append(stage)
        if group:
            groups.append(group)

    for stage in [str(item).lower() for item in stage_order]:
        if stage not in seen:
            groups.append([stage])
    return groups


def _selected_stage_groups(stage_groups: Sequence[Sequence[str]], selected_stages: Sequence[str]) -> List[List[str]]:
    selected = {str(stage).lower() for stage in selected_stages}
    groups: List[List[str]] = []
    for group in stage_groups:
        filtered = [str(stage).lower() for stage in group if str(stage).lower() in selected]
        if filtered:
            groups.append(filtered)
    return groups


def _parse_stage_selection(raw_stage: str, stage_order: Sequence[str]) -> List[str]:
    raw_stage = str(raw_stage).strip()
    if not raw_stage:
        raise ValueError("Stage selection cannot be empty.")
    if raw_stage.lower() == "all":
        return [str(stage).lower() for stage in stage_order]

    wanted = [str(item).lower() for item in parse_csv(raw_stage)]
    if not wanted:
        raise ValueError("Stage selection cannot be empty.")
    valid = set(str(stage).lower() for stage in stage_order)
    missing = [stage for stage in wanted if stage not in valid]
    if missing:
        raise ValueError(f"Unknown stage values {missing}; expected subset of {list(stage_order)} or all.")

    selected: List[str] = []
    seen = set()
    for stage in wanted:
        if stage not in seen:
            seen.add(stage)
            selected.append(stage)
    return selected


def _stage_selection_label(selected_stages: Sequence[str], stage_order: Sequence[str]) -> str:
    selected = [str(stage).lower() for stage in selected_stages]
    ordered = [str(stage).lower() for stage in stage_order]
    if selected == ordered:
        return "all"
    if len(selected) == 1:
        return selected[0]
    return "multi_" + "_".join(_sanitize(stage)[:10] for stage in selected)


def _stage_specs(
    spec: Mapping[str, Any],
    selected_stages: Sequence[str],
    stage_order: Sequence[str],
    graph_method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    base_fixed = dict(spec.get("FIXED_OPTS", {}) or {})
    stages: List[Dict[str, Any]] = []
    selected_set = {str(stage).lower() for stage in selected_stages}
    explicit_stages = spec.get("STAGES", None)
    if isinstance(explicit_stages, Mapping):
        selected = [str(item).lower() for item in stage_order if str(item).lower() in selected_set]
        for stage_name in selected:
            stage_cfg = explicit_stages.get(stage_name, explicit_stages.get(str(stage_name), None))
            if stage_cfg is None:
                continue
            if not isinstance(stage_cfg, Mapping):
                raise ValueError(f"Stage spec must be a mapping: {stage_name}")
            fixed = dict(base_fixed)
            fixed.update(dict(stage_cfg.get("FIXED_OPTS", {}) or {}))
            grid = dict(stage_cfg.get("GRID", stage_cfg.get("PARAM_GRID", {}) or {}) or {})
            graph_stage_cfgs = stage_cfg.get("GRAPH_METHODS", None)
            if isinstance(graph_stage_cfgs, Mapping):
                current_graph_method = str(graph_method or "")
                graph_stage_cfg = graph_stage_cfgs.get(current_graph_method, None)
                if graph_stage_cfg is None:
                    continue
                if not isinstance(graph_stage_cfg, Mapping):
                    raise ValueError(
                        f"Graph-method stage spec must be a mapping: "
                        f"stage={stage_name} graph_method={current_graph_method}"
                    )
                fixed.update(dict(graph_stage_cfg.get("FIXED_OPTS", {}) or {}))
                grid.update(dict(graph_stage_cfg.get("GRID", graph_stage_cfg.get("PARAM_GRID", {}) or {}) or {}))
            if grid or stage_name in selected_set:
                stages.append({"stage": stage_name, "fixed": fixed, "grid": grid})
        return stages
    raise ValueError("GRAPH_GP_SEARCH_SPACE.STAGES must be a mapping.")


def _load_graph_methods(grid_cfg: Mapping[str, Any], selected: str) -> List[str]:
    external = grid_cfg.get("EXTERNAL_GRAPH", {})
    methods = [str(item) for item in _as_list(external.get("METHODS"))]
    if not methods:
        raise ValueError("Grid config EXTERNAL_GRAPH.METHODS must list at least one graph key.")
    if selected and selected.lower() != "all":
        wanted = parse_csv(selected)
        missing = [item for item in wanted if item not in methods]
        if missing:
            raise ValueError(f"Unknown --graph-methods values {missing}; expected subset of {methods}.")
        return wanted
    return methods


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


def _trial_name(index: int, stage: str, graph_method: str, combo: Mapping[str, Any]) -> str:
    combo_json = json.dumps(dict(combo), sort_keys=True, ensure_ascii=True)
    combo_hash = hashlib.md5(combo_json.encode("utf-8")).hexdigest()[:8]
    graph_alias = GRAPH_METHOD_ALIASES.get(str(graph_method), _sanitize(graph_method)[:12])
    stage_alias = _sanitize(stage)[:12]
    return f"t{index:04d}_{stage_alias}_{graph_alias}_{combo_hash}"


def _build_trials(
    repo_root: Path,
    grid_cfg: Mapping[str, Any],
    graph_methods: Sequence[str],
    config_file: str,
    train_script: str,
    python_bin: str,
    out_root: Path,
    extra_opts: Sequence[str],
    selected_stages: Sequence[str],
    stage_order: Sequence[str],
    stage_groups: Sequence[Sequence[str]],
    trial_order: str,
    base_total_epochs: int,
) -> List[Dict[str, Any]]:
    fixed_opts = dict(grid_cfg.get("FIXED_OPTS", {}) or {})
    search_space = grid_cfg.get("GRAPH_GP_SEARCH_SPACE", {})
    if not isinstance(search_space, Mapping):
        raise ValueError("GRAPH_GP_SEARCH_SPACE must be a mapping.")
    graph_path = str(grid_cfg["EXTERNAL_GRAPH"]["PATH"])
    trials: List[Dict[str, Any]] = []
    index = 0
    trial_order = str(trial_order or "graph_stage").lower()
    extra_override_map = dict(zip(extra_opts[0::2], extra_opts[1::2]))

    def append_trials(
        graph_method: str,
        active_selected_stages: Sequence[str],
        active_stage_order: Sequence[str],
    ) -> None:
        nonlocal index
        for stage_spec in _stage_specs(
            search_space,
            active_selected_stages,
            active_stage_order,
            graph_method=graph_method,
        ):
            stage_name = str(stage_spec["stage"])
            stage_fixed = dict(stage_spec["fixed"])
            combos = _cartesian_grid(stage_spec.get("grid", {}) or {})
            for combo_index, combo in enumerate(combos):
                overrides: Dict[str, Any] = {}
                overrides.update(fixed_opts)
                overrides.update(stage_fixed)
                overrides.update(
                    {
                        "MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_PATH": graph_path,
                        "MODEL.GRAPH_INPUT.GRAPH_SOURCE": graph_method,
                    }
                )
                overrides.update(combo)
                trial_name = _trial_name(index, stage_name, graph_method, combo)
                output_dir = out_root / stage_name / graph_method / trial_name
                run_overrides = dict(overrides)
                run_overrides["OUTPUT_DIR"] = str(output_dir)
                expected_epochs = int(
                    extra_override_map.get(
                        "SOLVER.TOTAL_EPOCH",
                        run_overrides.get("SOLVER.TOTAL_EPOCH", base_total_epochs),
                    )
                )
                if expected_epochs <= 0:
                    raise ValueError("SOLVER.TOTAL_EPOCH must be positive for complete-trial detection.")
                cmd = [python_bin, train_script, "--config-file", config_file]
                cmd.extend(mapping_to_opts(run_overrides))
                cmd.extend(extra_opts)
                trials.append(
                    {
                        "trial_index": index,
                        "stage": stage_name,
                        "combo_index": combo_index,
                        "trial_name": trial_name,
                        "graph_method": graph_method,
                        "combo": dict(combo),
                        "overrides": dict(run_overrides),
                        "output_dir": str(output_dir),
                        "stdout_path": str(output_dir / "launcher_stdout.txt"),
                        "cmd": cmd,
                        "repo_root": str(repo_root),
                        "runner": "train",
                        "expected_epochs": expected_epochs,
                        "identity_fields": {
                            "graph_method": graph_method,
                            "stage": stage_name,
                        },
                        "eta_fields": {
                            "graph_method": graph_method,
                            "stage": stage_name,
                        },
                        "eta_compatibility_keys": ["runner", "stage"],
                    }
                )
                index += 1

    if trial_order == "graph_stage":
        for graph_method in graph_methods:
            append_trials(graph_method, selected_stages, stage_order)
    elif trial_order == "stage_group":
        for stage_group in _selected_stage_groups(stage_groups, selected_stages):
            for graph_method in graph_methods:
                append_trials(graph_method, stage_group, stage_group)
    else:
        raise ValueError("TRIAL_ORDER must be 'graph_stage' or 'stage_group'.")
    return trials


_GZSL_RECORD_RE = re.compile(
    r"\[gzsl-record\]\s+epoch=(?P<epoch>\d+)\s+"
    r"gzsl_seen=(?P<seen>[-+0-9.eE]+)\s+"
    r"gzsl_unseen=(?P<unseen>[-+0-9.eE]+)\s+"
    r"gzsl_h=(?P<h>[-+0-9.eE]+)\s+"
    r"best_seen_recorded=(?P<best_seen>[-+0-9.eE]+)\s+"
    r"best_unseen_recorded=(?P<best_unseen>[-+0-9.eE]+)\s+"
    r"gzsl_h_recorded=(?P<best_h_recorded>[-+0-9.eE]+)"
)


_GPP_MONITOR_LINE_RE = re.compile(r"\[graph-prob-prior-monitor\]\s+(?P<body>.+)")
_GPP_MONITOR_KV_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9]*)=(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _train_log_paths(output_dir: Path) -> List[Path]:
    if not output_dir.is_dir():
        return []
    logs = sorted(output_dir.rglob("logs.txt"), key=lambda path: (path.stat().st_mtime, str(path)))
    if logs:
        return logs
    stdout_path = output_dir / "launcher_stdout.txt"
    return [stdout_path] if stdout_path.is_file() else []


def _parse_train_gzsl_records(path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return records
    for match in _GZSL_RECORD_RE.finditer(text):
        record = {
            "epoch": float(match.group("epoch")),
            "gzsl_seen": float(match.group("seen")) * 100.0,
            "gzsl_unseen": float(match.group("unseen")) * 100.0,
            "gzsl_h": float(match.group("h")) * 100.0,
            "best_seen_recorded": float(match.group("best_seen")) * 100.0,
            "best_unseen_recorded": float(match.group("best_unseen")) * 100.0,
            "gzsl_h_recorded": float(match.group("best_h_recorded")) * 100.0,
        }
        records.append(record)
    return records


def _parse_train_monitor_records(path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return records
    for match in _GPP_MONITOR_LINE_RE.finditer(text):
        record: Dict[str, float] = {}
        body = match.group("body")
        for kv in _GPP_MONITOR_KV_RE.finditer(body):
            raw_key = kv.group("key")
            key = GPP_MONITOR_ALIASES.get(raw_key)
            if not key:
                continue
            try:
                value = float(kv.group("value"))
            except ValueError:
                continue
            if np.isfinite(value):
                record[key] = value
        if record:
            records.append(record)
    return records


def _summarize_train_monitor_records(records: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not records:
        return summary
    summary["graph_prob_prior_monitor_log_count"] = float(len(records))
    keys = sorted({key for record in records for key in record.keys()})
    for key in keys:
        values = [float(record[key]) for record in records if key in record and np.isfinite(float(record[key]))]
        if not values:
            continue
        last = float(values[-1])
        summary[key] = last
        summary[f"{key}_log_last"] = last
        summary[f"{key}_log_mean"] = float(np.mean(values))
    return summary


def _training_summary_payload(output_dir: Path) -> Dict[str, Any]:
    records: List[Dict[str, float]] = []
    monitor_records: List[Dict[str, float]] = []
    log_paths = _train_log_paths(output_dir)
    for path in log_paths:
        records.extend(_parse_train_gzsl_records(path))
        monitor_records.extend(_parse_train_monitor_records(path))
    if not records:
        return {}

    records.sort(key=lambda row: (int(row.get("epoch", 0)), row.get("gzsl_h", 0.0)))
    last = records[-1]
    best = max(records, key=lambda row: float(row.get("gzsl_h", float("-inf"))))
    performance = {
        "eval_ran": 1.0,
        "gzsl_seen_last": float(last["gzsl_seen"]),
        "gzsl_unseen_last": float(last["gzsl_unseen"]),
        "gzsl_h_last": float(last["gzsl_h"]),
        "gzsl_seen_best": float(best["gzsl_seen"]),
        "gzsl_unseen_best": float(best["gzsl_unseen"]),
        "gzsl_h_best": float(best["gzsl_h"]),
        "gzsl_best_epoch": float(best["epoch"]),
        "gzsl_seen_recorded_best": float(last["best_seen_recorded"]),
        "gzsl_unseen_recorded_best": float(last["best_unseen_recorded"]),
        "gzsl_h_recorded_best": float(last["gzsl_h_recorded"]),
    }
    return {
        "runner": "train",
        "log_paths": [str(path) for path in log_paths],
        "num_epochs": int(max(int(row["epoch"]) for row in records)),
        "num_batches": "",
        "performance": performance,
        "summary": _summarize_train_monitor_records(monitor_records),
        "records": records,
    }


def _trial_payload(trial: Mapping[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(trial["output_dir"]))
    return _training_summary_payload(output_dir)


def _trial_has_complete_output(trial: Mapping[str, Any]) -> bool:
    payload = _trial_payload(trial)
    if not payload:
        return False
    performance = payload.get("performance")
    if not isinstance(performance, dict):
        return False
    try:
        expected_epochs = int(trial.get("expected_epochs", 0))
        num_epochs = int(payload.get("num_epochs", 0))
    except (TypeError, ValueError):
        return False
    return expected_epochs > 0 and num_epochs >= expected_epochs


def _resume_status(trial: Mapping[str, Any]) -> str:
    output_dir = Path(str(trial["output_dir"]))
    payload = _training_summary_payload(output_dir)
    if not payload:
        return f"missing train gzsl-record under {output_dir}"
    expected_epochs = int(trial.get("expected_epochs", 0))
    completed_epochs = int(payload.get("num_epochs", 0))
    if expected_epochs <= 0 or completed_epochs < expected_epochs:
        return f"incomplete train epochs {completed_epochs}/{expected_epochs} under {output_dir}"
    return f"complete train logs {payload.get('log_paths', [])}"


def _flatten_result(
    trial: Mapping[str, Any],
    returncode: int,
    gpu_id: str = "",
    command: Optional[Sequence[str]] = None,
    num_gpus: int = 1,
    skipped_existing: bool = False,
) -> Dict[str, Any]:
    command = list(command) if command is not None else list(trial["cmd"])
    row: Dict[str, Any] = {
        "trial_index": trial["trial_index"],
        "stage": trial.get("stage", ""),
        "trial_name": trial["trial_name"],
        "graph_method": trial["graph_method"],
        "combo_index": trial["combo_index"],
        "returncode": returncode,
        "gpu": gpu_id,
        "num_gpus": int(num_gpus),
        "skipped_existing": bool(skipped_existing),
        "output_dir": trial["output_dir"],
        "stdout_path": trial["stdout_path"],
        "command": subprocess.list2cmdline(command),
    }
    for key, value in trial["combo"].items():
        row[key] = value
    payload = _trial_payload(trial)
    if payload:
        if payload.get("runner"):
            row["runner"] = payload.get("runner")
        if payload.get("num_epochs", "") != "":
            row["num_epochs"] = payload.get("num_epochs", "")
        row["num_batches"] = payload.get("num_batches", "")
        if payload.get("log_paths"):
            row["log_paths"] = ";".join(str(path) for path in payload.get("log_paths", []))
        for key, value in (payload.get("performance") or {}).items():
            if isinstance(value, (int, float, str, bool)):
                row[key] = value
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
        scored["primary_score"] = float("-inf")
        scored["primary_score_key"] = "failed"
        return scored

    score = 0.0
    score += _target_penalty(scored, "graph_prob_prior_monitor_neighbor_entropy_norm_mean", 0.55, 0.30, 1.2)
    score += _target_penalty(scored, "graph_prob_prior_monitor_neighbor_top1_mean", 0.35, 0.30, 0.8)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_neighbor_hubness_gini", 1.0, 0.5)

    score += _lower_is_better(scored, "graph_prob_prior_monitor_prior_overlap_risk_rate", 1.0, 1.0)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_false_high_prior_relation_still_gt_0_9_count", 20.0, 0.8)
    score += _lower_is_better(scored, "graph_prob_prior_monitor_prior_gzsl_unseen_to_seen_bias_risk_mean", 1.0, 0.5)
    score += _higher_is_better(scored, "graph_prob_prior_graph_gp_energy_margin_positive_ratio", 0.50, 1.0)
    score += _higher_is_better(scored, "graph_prob_prior_graph_gp_energy_pseudo_unseen_acc", 0.20, 1.0)
    score += _upper_penalty(scored, "graph_prob_prior_monitor_loss_weighted_gpp_to_main_loss_ratio", 0.10, 1.0)

    loss = _row_float(scored, "graph_prob_prior_match_loss")
    loss_term = 0.0 if loss is None else 0.01 * np.log1p(max(loss, 0.0))
    scored["monitor_score"] = float(score)
    scored["selection_score"] = float(score + loss_term)
    primary = _row_float(scored, "gzsl_h_last")
    if primary is not None:
        scored["primary_score"] = float(primary)
        scored["primary_score_key"] = "gzsl_h_last"
    else:
        primary = _row_float(scored, "zsl_unseen_last")
        if primary is not None:
            scored["primary_score"] = float(primary)
            scored["primary_score_key"] = "zsl_unseen_last"
        else:
            primary = _row_float(scored, "dev_unseen_last")
            scored["primary_score"] = float(primary) if primary is not None else float("-inf")
            scored["primary_score_key"] = "dev_unseen_last" if primary is not None else "monitor_only"
    return scored


def _rank_metric(row: Mapping[str, Any], key: str) -> tuple[bool, float]:
    value = _row_float(row, key)
    if value is None:
        return True, 0.0
    return False, -float(value)


def _rank_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    scored = [_with_monitor_scores(dict(row)) for row in rows]
    return sorted(
        scored,
        key=lambda row: (
            int(row.get("returncode", 0)) != 0,
            *_rank_metric(row, "gzsl_h_last"),
            *_rank_metric(row, "gzsl_unseen_last"),
            *_rank_metric(row, "gzsl_seen_last"),
            *_rank_metric(row, "zsl_unseen_last"),
            *_rank_metric(row, "dev_unseen_last"),
            float(row.get("selection_score", 1.0e9)),
            str(row.get("stage", "")),
            str(row.get("graph_method", "")),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GraphProbPrior parameter-search plan.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--grid-config", required=True)
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--config-file", default="")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--graph-methods", default="all")
    parser.add_argument("--stage", default="", help="Stage name from the YAML, or all.")
    parser.add_argument(
        "--stages",
        default="",
        help="Comma-separated stage names from the YAML. Overrides --stage when set.",
    )
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
    parser.add_argument("--dist-backend", default=default_dist_backend(), choices=["nccl", "gloo"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--eta-interval",
        type=float,
        default=30.0,
        help="Seconds between task-level ETA updates; 0 disables periodic ETA output.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N expanded trials; 0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Rerun trials even when a complete training result already exists.")
    parser.add_argument("--resume-debug", action="store_true", help="Print why an existing trial was or was not skipped.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra KEY VALUE config overrides appended to every child run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    grid_config = Path(args.grid_config)
    if not grid_config.is_absolute():
        grid_config = repo_root / grid_config
    grid_cfg = _read_yaml(grid_config)

    config_file = args.config_file or str(
        grid_cfg.get("BASE_CONFIG_FILE", "configs/graph_prob_prior/cub_graph_gp_energy_vpt_deep.yaml")
    )
    base_config_path = Path(config_file)
    if not base_config_path.is_absolute():
        base_config_path = repo_root / base_config_path
    base_cfg = get_cfg()
    base_cfg.merge_from_file(str(base_config_path))
    base_total_epochs = int(base_cfg.SOLVER.TOTAL_EPOCH)
    train_script = str(grid_cfg.get("TRAIN_SCRIPT", "train.py"))
    runner = str(grid_cfg.get("RUNNER", "train")).lower()
    if runner != "train":
        raise ValueError(f"Graph-GP search RUNNER must be 'train', got {runner}.")
    python_bin = str(args.python_bin or grid_cfg.get("PYTHON_BIN") or sys.executable)
    out_root = Path(args.out_root or str(grid_cfg.get("OUTPUT_DIR", "output/graph_prob_prior_search")))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    extra_opts = validate_extra_opts(args.opts)
    stage_order = _configured_stage_names(grid_cfg)
    stage_groups = _configured_stage_groups(grid_cfg, stage_order)
    trial_order = str(grid_cfg.get("TRIAL_ORDER", "graph_stage")).lower()
    raw_stage_selection = str(args.stages or args.stage or grid_cfg.get("STAGE", stage_order[0])).lower()
    selected_stages = _parse_stage_selection(raw_stage_selection, stage_order)
    stage_label = _stage_selection_label(selected_stages, stage_order)

    graph_methods = _load_graph_methods(grid_cfg, args.graph_methods)
    graph_npz = _validate_external_graph_file(repo_root, grid_cfg, graph_methods)
    gpu_groups = parse_gpu_groups(args.gpu_groups, args.gpus)
    validate_gpu_groups(gpu_groups, int(args.max_workers), int(args.nproc_per_trial))
    if not np.isfinite(float(args.eta_interval)) or args.eta_interval < 0:
        raise ValueError("--eta-interval must be finite and non-negative.")

    trials = _build_trials(
        repo_root=repo_root,
        grid_cfg=grid_cfg,
        graph_methods=graph_methods,
        config_file=config_file,
        train_script=train_script,
        python_bin=python_bin,
        out_root=out_root,
        extra_opts=extra_opts,
        selected_stages=selected_stages,
        stage_order=stage_order,
        stage_groups=stage_groups,
        trial_order=trial_order,
        base_total_epochs=base_total_epochs,
    )
    if args.limit > 0:
        trials = trials[: int(args.limit)]

    out_root.mkdir(parents=True, exist_ok=True)
    stage_suffix = f"_{stage_label}"
    commands_path = out_root / f"commands{stage_suffix}.txt"

    def command_builder(trial: Mapping[str, Any], gpu_group: str):
        return build_run_command(
            trial,
            python_bin=python_bin,
            gpu_group=gpu_group,
            nproc_per_trial=int(args.nproc_per_trial),
            dist_backend=str(args.dist_backend),
        )

    def complete_check(trial: Mapping[str, Any]) -> bool:
        return _trial_has_complete_output(trial)

    scheduler = SearchScheduler(
        trials=trials,
        out_root=out_root,
        gpu_groups=gpu_groups,
        max_workers=int(args.max_workers),
        nproc_per_trial=int(args.nproc_per_trial),
        eta_interval=float(args.eta_interval),
        command_builder=command_builder,
        complete_check=complete_check,
        result_flattener=_flatten_result,
        resume_status=_resume_status,
        resume=not bool(args.no_resume),
        resume_debug=bool(args.resume_debug),
    )
    scheduler.write_commands(commands_path)

    search_space = {
        "grid_config": str(grid_config),
        "base_config_file": config_file,
        "base_total_epochs": base_total_epochs,
        "train_script": train_script,
        "runner": runner,
        "python_bin": python_bin,
        "external_graph_npz": str(graph_npz),
        "graph_methods": graph_methods,
        "stage": stage_label,
        "selected_stages": selected_stages,
        "stage_order": stage_order,
        "stage_groups": stage_groups,
        "trial_order": trial_order,
        "total_trials": len(trials),
        "dry_run": bool(args.dry_run),
        "resume": not bool(args.no_resume),
        "gpu_groups": gpu_groups,
        "nproc_per_trial": int(args.nproc_per_trial),
        "dist_backend": str(args.dist_backend),
        "max_workers": int(args.max_workers),
        "eta_interval_seconds": float(args.eta_interval),
        "eta_estimator": {
            "active_progress_weight": SearchScheduler.ACTIVE_PROGRESS_WEIGHT,
            "historical_weight": SearchScheduler.HISTORICAL_WEIGHT,
            "history_statistic": "median_with_interquartile_range",
            "queue_model": "earliest_available_worker_simulation",
        },
        "runtime_state": {
            "search_state": str(out_root / "search_state.json"),
            "active_trial_progress": "<trial_output_dir>/progress.json",
            "completed_progress_cleanup": True,
        },
        "extra_opts": extra_opts,
        "graph_gp_search_space": grid_cfg.get("GRAPH_GP_SEARCH_SPACE", {}),
        "fixed_opts": grid_cfg.get("FIXED_OPTS", {}),
        "commands_path": str(commands_path),
        "ranking": {
            "primary_order": "gzsl_h_last desc, gzsl_unseen_last desc, gzsl_seen_last desc, then monitor selection_score asc",
            "selection_score": "monitor_score + 0.01*log1p(graph_prob_prior_match_loss)",
            "notes": [
                "ranked_summary is performance-first when training eval metrics are present",
                "lower selection_score is only a tie-breaker after final seen/unseen/H metrics",
                "monitor score prefers balanced graph neighborhoods, low prior overlap, low false-high residue, healthy Graph-GP energy classification, and reasonable weighted loss scale",
            ],
        },
    }
    write_json(out_root / "search_space.json", search_space)

    rows = scheduler.dry_run() if args.dry_run else scheduler.run()

    ranked_rows = _rank_rows(rows)
    best_by_stage_graph = _best_rows(ranked_rows, ["stage", "graph_method"])
    best_by_stage = _best_rows(ranked_rows, ["stage"])
    write_csv(out_root / "summary.csv", ranked_rows)
    write_json(out_root / "summary.json", {"search_space": search_space, "rows": ranked_rows})
    write_csv(out_root / "ranked_summary.csv", ranked_rows)
    write_json(out_root / "ranked_summary.json", {"search_space": search_space, "rows": ranked_rows})
    write_csv(out_root / "best_by_stage_graph.csv", best_by_stage_graph)
    write_json(out_root / "best_by_stage_graph.json", {"rows": best_by_stage_graph})
    write_csv(out_root / "best_by_stage.csv", best_by_stage)
    write_json(out_root / "best_by_stage.json", {"rows": best_by_stage})
    write_csv(out_root / f"summary{stage_suffix}.csv", ranked_rows)
    write_json(out_root / f"summary{stage_suffix}.json", {"search_space": search_space, "rows": ranked_rows})
    write_csv(out_root / f"ranked_summary{stage_suffix}.csv", ranked_rows)
    write_json(out_root / f"ranked_summary{stage_suffix}.json", {"search_space": search_space, "rows": ranked_rows})
    write_csv(out_root / f"best_by_stage_graph{stage_suffix}.csv", best_by_stage_graph)
    write_json(out_root / f"best_by_stage_graph{stage_suffix}.json", {"rows": best_by_stage_graph})
    write_csv(out_root / f"best_by_stage{stage_suffix}.csv", best_by_stage)
    write_json(out_root / f"best_by_stage{stage_suffix}.json", {"rows": best_by_stage})
    failures = [row for row in rows if int(row.get("returncode", 0)) not in {0, -1}]
    if failures:
        write_json(out_root / "failures.json", {"failures": failures})
        raise SystemExit(f"{len(failures)} trials failed; see {out_root / 'failures.json'}")
    print(f"wrote {out_root / 'summary.csv'}")
    print(f"wrote {out_root / 'summary.json'}")
    print(f"wrote {out_root / 'ranked_summary.csv'}")
    print(f"wrote {out_root / 'best_by_stage_graph.csv'}")
    print(f"wrote {out_root / 'best_by_stage.csv'}")
    print(f"wrote {out_root / f'ranked_summary{stage_suffix}.csv'}")
    print(f"wrote {out_root / f'best_by_stage_graph{stage_suffix}.csv'}")


def cli_main() -> None:
    main()


if __name__ == "__main__":
    cli_main()
