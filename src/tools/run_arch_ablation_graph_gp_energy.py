#!/usr/bin/env python3
"""Run the valid Graph-GP x semantic-token x Attention Mediation experiment.

The default design is C00/C10/C01/C11/A00/A10 across seeds 17/29/43, for 18
training runs. Attention Mediation is nested under semantic tokens, so the two
semantic-off/AM-on cells are intentionally undefined.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.grid_search_graph_prob_prior_v5_temperatures import (  # noqa: E402
    _build_run_command,
    _default_dist_backend,
    _existing_trial_result,
    _flatten_result,
    _group_nproc,
    _mapping_to_opts,
    _parse_gpu_groups,
    _rank_rows,
    _resume_status,
    _run_trial,
    _validate_extra_opts,
    _write_csv,
    _write_json,
)


DEFAULT_CONFIG_FILE = "configs/prompt/cub.yaml"
DEFAULT_OUT_ROOT = "output/stage1_clean_graph_gp_am"
TRAIN_SCRIPT = "train.py"
DEFAULT_SEEDS = (17, 29, 43)
SUMMARY_METRICS = (
    "dev_unseen_last",
    "zsl_unseen_last",
    "gzsl_seen_last",
    "gzsl_unseen_last",
    "gzsl_h_last",
    "gzsl_seen_best",
    "gzsl_unseen_best",
    "gzsl_h_best",
)
PAIRED_EFFECTS = {
    "graph_gp_no_semantic": ("A10", "A00"),
    "graph_gp_no_am": ("C10", "C00"),
    "graph_gp_with_am": ("C11", "C01"),
    "am_no_graph_gp": ("C01", "C00"),
    "am_with_graph_gp": ("C11", "C10"),
    "semantic_token_only": ("C00", "A00"),
    "semantic_with_graph_gp": ("C10", "A10"),
    "semantic_am_package": ("C01", "A00"),
}


def _base_overrides(protocol_mode: str) -> Dict[str, Any]:
    return {
        "RUN_N_TIMES": 1,
        "DATA.NAME": "CUB",
        "DATA.NUMBER_CLASSES": 200,
        "DATA.XLSA.PROTOCOL_MODE": str(protocol_mode),
        "DATA.BATCH_SIZE": 32,
        "MODEL.TYPE": "vit",
        "MODEL.CLASSIFIER": "vspcn_baseline",
        "SOLVER.MAIN_LOSS": "vspcn",
        "SOLVER.LOSS_VSPCN_AR_WEIGHT": 0.0005,
        "SOLVER.LOSS_CM_WEIGHT": 0.0,
        "SOLVER.BASE_LR": 0.0006,
        "SOLVER.WEIGHT_DECAY": 0.00001,
        "SOLVER.WARMUP_EPOCH": 3,
        "SOLVER.TOTAL_EPOCH": 30,
        "SOLVER.OPTIMIZER": "adamw",
        "SOLVER.SCHEDULER": "cosine",
        "MODEL.PROMPT.ENABLE": True,
        "MODEL.PROMPT.NUM_TOKENS": 32,
        "MODEL.PROMPT.DROPOUT": 0.0,
        "MODEL.AFFINITY.ENABLE": False,
        "MODEL.AFFINITY.DETACH": False,
        "MODEL.AFFINITY.VIS": False,
        "SOLVER.LOSS_PROMPT_KL_WEIGHT": 0.0,
        "SOLVER.LOSS_ATTR_WEIGHT": 0.0,
        "SOLVER.LOSS_SEM_MED_WEIGHT": 0.0,
        "SOLVER.LOSS_SPV_WEIGHT": 0.0,
        "SOLVER.VIS.ENABLE": False,
        "SOLVER.VIS.SAVE_RAW": False,
        "SOLVER.VIS.SAVE_IMAGES": False,
        "SOLVER.VIS.ROLLOUT": False,
    }


def _instance_prompt_distributor_overrides() -> Dict[str, Any]:
    return {
        "MODEL.PROMPT.BACKEND": "dynamic",
        "MODEL.PROMPT.INIT_SOURCE": "distributor_mean",
        "MODEL.PROMPT.DEEP": False,
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE": True,
        "MODEL.PROMPT.DISTRIBUTOR.SOURCE": "token_mlp",
        "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM": 64,
        "MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS": 16,
        "MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS": 16,
        "MODEL.PROMPT.DISTRIBUTOR.OUTPUT_PARAM": "logvar",
        "MODEL.PROMPT.DISTRIBUTOR.LOGVAR_MIN": -10.0,
        "MODEL.PROMPT.DISTRIBUTOR.LOGVAR_MAX": 5.0,
        "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": "mean",
        "MODEL.PROMPT.DISTRIBUTOR.FIXED_EPS_SEED": 0,
        "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED": True,
        "MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_ENABLE": False,
    }


def _semantic_token_overrides(enable: bool) -> Dict[str, Any]:
    if not enable:
        return {"MODEL.SEMANTIC_TOKENS.ENABLE": False}
    return {
        "MODEL.SEMANTIC_TOKENS.ENABLE": True,
        "MODEL.SEMANTIC_TOKENS.TOKENIZER": "orthogonal",
        "MODEL.SEMANTIC_TOKENS.NUM_TOKENS": 8,
        "MODEL.SEMANTIC_TOKENS.INPUT_DIM": 312,
        "MODEL.SEMANTIC_TOKENS.TRAIN_SOURCE": "class_mean",
        "MODEL.SEMANTIC_TOKENS.EVAL_SOURCE": "class_mean",
        "MODEL.SEMANTIC_TOKENS.BLOCK_S_TO_CLS": False,
        "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE": "equal",
        "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE": "none",
        "MODEL.SEMANTIC_TOKENS.ORTHO.CODEBOOK_TRAINABLE": False,
        "MODEL.SEMANTIC_TOKENS.ORTHO.CODEBOOK_SEED": 0,
        "MODEL.SEMANTIC_TOKENS.ORTHO.DEBUG": False,
    }


def _attention_mediation_overrides(enable: bool) -> Dict[str, Any]:
    if not enable:
        return {
            "MODEL.ATTENTION_MEDIATION.ENABLE": False,
        }
    return {
        "MODEL.ATTENTION_MEDIATION.ENABLE": True,
        "MODEL.ATTENTION_MEDIATION.SOURCE": "scores",
        "MODEL.ATTENTION_MEDIATION.EXECUTION_MODE": "block_parallel",
        "MODEL.ATTENTION_MEDIATION.MLP_POLICY": "enter_mlp",
        "MODEL.ATTENTION_MEDIATION.ROUTE_SCOPE": "visual_block",
        "MODEL.ATTENTION_MEDIATION.MASS_MODE": "block_redistribute",
        "MODEL.ATTENTION_MEDIATION.PROMPT_ROUTE": "S_to_P_and_V",
        "MODEL.ATTENTION_MEDIATION.SEMANTIC_ROUTE": "P_to_S_and_V",
        "MODEL.ATTENTION_MEDIATION.BETA_PROMPT_MASS": 1.0,
        "MODEL.ATTENTION_MEDIATION.BETA_SEMANTIC_MASS": 1.0,
        "MODEL.ATTENTION_MEDIATION.PROMPT_GAMMA_INIT": 0.0,
        "MODEL.ATTENTION_MEDIATION.SEMANTIC_GAMMA_INIT": 0.0,
    }


def _graph_prob_prior_overrides(enable: bool) -> Dict[str, Any]:
    if not enable:
        return {
            "MODEL.GRAPH_PROB_PRIOR.ENABLE": False,
            "MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT": 0.0,
            "MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE": False,
        }
    return {
        "MODEL.GRAPH_INPUT.GRAPH_SOURCE": "method1_diff",
        "MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_PATH": (
            "cub_attribute_localization/05_hparam_searches/"
            "diff_only_graphs_v1/diff_only_method_matrices_v1.npz"
        ),
        "MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_SYMMETRIZE": True,
        "MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_CLAMP": True,
        "MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_DIAG_VALUE": 1.0,
        "MODEL.GRAPH_INPUT.NUM_CLASSES": 200,
        "MODEL.GRAPH_INPUT.ATTR_DIM": 312,
        "MODEL.GRAPH_INPUT.TEXT_DIM": 768,
        "MODEL.GRAPH_INPUT.EPS": 1e-8,
        "MODEL.GRAPH_PROB_PRIOR.ENABLE": True,
        "MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT": 0.001,
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_MEAN_MODE": "graph_gp_conditioned",
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_VAR_MODE": "unit",
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SUPPORT_RATIO": 0.8,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_USE_PSEUDO_UNSEEN": True,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SPLIT_EVERY_EPOCH": 1,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SPLIT_SEED": 2027,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_CENTER_SOURCE": "posterior_mu",
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_DETACH_CENTERS": True,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MODE": "class_var_over_count",
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_CONST": 0.05,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MIN": 1e-4,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MAX": 1.0,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_RIDGE": 1e-4,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_SYMMETRIZE": True,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_CLAMP": True,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_NORMALIZE": "diag",
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_SOURCE": "unit",
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_MATCH_DETACH_PRIOR": True,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBJECTIVE": "energy_classification",
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_ENERGY_TAU": 1.0,
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_ENERGY_CLASS_SPACE": "seen",
        "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PSEUDO_WEIGHT": 1.0,
        "MODEL.GRAPH_PROB_PRIOR.TAU_GRAPH": 0.10,
        "MODEL.GRAPH_PROB_PRIOR.REL_WEIGHT": 0.0,
        "MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_ENABLE": False,
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE": True,
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_INACTIVE": False,
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_TOPK": 5,
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N": 37,
    }


def _merge_overrides(*parts: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for part in parts:
        merged.update(dict(part))
    return merged


def _cell_specs() -> List[Dict[str, Any]]:
    return [
        {
            "cell_id": "C00",
            "cell_name": "c00_gp0_am0_sem1",
            "graph_gp": False,
            "attention_mediation": False,
            "semantic_tokens": True,
            "core_2x2": True,
            "purpose": "Distributor + semantic tokens; Graph-GP off; AM off.",
        },
        {
            "cell_id": "C10",
            "cell_name": "c10_gp1_am0_sem1",
            "graph_gp": True,
            "attention_mediation": False,
            "semantic_tokens": True,
            "core_2x2": True,
            "purpose": "Distributor + semantic tokens + Graph-GP; AM off.",
        },
        {
            "cell_id": "C01",
            "cell_name": "c01_gp0_am1_sem1",
            "graph_gp": False,
            "attention_mediation": True,
            "semantic_tokens": True,
            "core_2x2": True,
            "purpose": "Distributor + semantic tokens + AM; Graph-GP off.",
        },
        {
            "cell_id": "C11",
            "cell_name": "c11_gp1_am1_sem1",
            "graph_gp": True,
            "attention_mediation": True,
            "semantic_tokens": True,
            "core_2x2": True,
            "purpose": "Distributor + semantic tokens + Graph-GP + AM.",
        },
        {
            "cell_id": "A00",
            "cell_name": "a00_gp0_am0_sem0",
            "graph_gp": False,
            "attention_mediation": False,
            "semantic_tokens": False,
            "core_2x2": False,
            "purpose": "Distributor anchor without semantic tokens, Graph-GP, or AM.",
        },
        {
            "cell_id": "A10",
            "cell_name": "a10_gp1_am0_sem0",
            "graph_gp": True,
            "attention_mediation": False,
            "semantic_tokens": False,
            "core_2x2": False,
            "purpose": "Distributor + Graph-GP without semantic tokens or AM.",
        },
    ]


def _select_cell_specs(raw_cells: Sequence[str]) -> List[Dict[str, Any]]:
    specs = _cell_specs()
    raw_values = [raw_cells] if isinstance(raw_cells, str) else list(raw_cells or [])
    requested = [
        item.strip()
        for value in raw_values
        for item in str(value).split(",")
        if item.strip()
    ]
    if not requested:
        return specs

    selected: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for index, spec in enumerate(specs, start=1):
        by_key[str(index)] = spec
        by_key[str(spec["cell_id"]).lower()] = spec
        by_key[str(spec["cell_name"]).lower()] = spec
    seen = set()
    for item in requested:
        spec = by_key.get(item.lower())
        if spec is None:
            raise ValueError(
                f"Unknown cell '{item}'. Valid values: C00,C10,C01,C11,A00,A10; 1-6; or cell names."
            )
        cell_id = str(spec["cell_id"])
        if cell_id not in seen:
            selected.append(spec)
            seen.add(cell_id)
    return selected


def _parse_seeds(raw_seeds: Sequence[str]) -> List[int]:
    raw_values = [raw_seeds] if isinstance(raw_seeds, str) else list(raw_seeds or [])
    items = [
        item.strip()
        for value in raw_values
        for item in str(value).split(",")
        if item.strip()
    ]
    if not items:
        return list(DEFAULT_SEEDS)
    seeds: List[int] = []
    for item in items:
        seed = int(item)
        if seed < 0:
            raise ValueError("Experiment seeds must be non-negative integers.")
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def _build_trials(
    repo_root: Path,
    python_bin: str,
    config_file: str,
    out_root: Path,
    protocol_mode: str,
    seeds: Sequence[int],
    extra_opts: Sequence[str],
    selected_specs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    train_script = repo_root / TRAIN_SCRIPT
    trials: List[Dict[str, Any]] = []
    for combo_index, spec in enumerate(selected_specs):
        graph_gp = bool(spec["graph_gp"])
        attention_mediation = bool(spec["attention_mediation"])
        semantic_tokens = bool(spec["semantic_tokens"])
        for seed in seeds:
            trial_index = len(trials)
            cell_id = str(spec["cell_id"])
            cell_name = str(spec["cell_name"])
            trial_name = f"{cell_id.lower()}_seed_{int(seed)}"
            output_dir = out_root / cell_name / f"seed_{int(seed)}"
            overrides = _merge_overrides(
                _base_overrides(protocol_mode),
                _instance_prompt_distributor_overrides(),
                _semantic_token_overrides(semantic_tokens),
                _graph_prob_prior_overrides(graph_gp),
                _attention_mediation_overrides(attention_mediation),
                {
                    "SEED": int(seed),
                    "SOLVER.SAVE_TRAINABLE_FINAL_CHECKPOINT": True,
                    "SOLVER.STAGE2_CHECKPOINT_CELL_ID": cell_id,
                    "OUTPUT_DIR": str(output_dir),
                },
            )
            cmd = [
                python_bin,
                str(train_script),
                "--config-file",
                str(config_file),
            ] + _mapping_to_opts(overrides) + list(extra_opts)
            trials.append(
                {
                    "trial_index": trial_index,
                    "stage": "stage1_clean_graph_gp_am",
                    "trial_name": trial_name,
                    "graph_method": "method1_diff" if graph_gp else "none",
                    "mode": "graph_gp_energy" if graph_gp else "no_graph_gp",
                    "combo_index": int(combo_index),
                    "combo": {
                        "cell_id": cell_id,
                        "cell_name": cell_name,
                        "seed": int(seed),
                        "protocol_mode": str(protocol_mode),
                        "graph_gp": graph_gp,
                        "attention_mediation": attention_mediation,
                        "semantic_tokens": semantic_tokens,
                        "core_2x2": bool(spec["core_2x2"]),
                        "purpose": str(spec["purpose"]),
                    },
                    "overrides": overrides,
                    "output_dir": str(output_dir),
                    "stdout_path": str(output_dir / "launcher_stdout.txt"),
                    "cmd": cmd,
                    "repo_root": str(repo_root),
                    "runner": "train",
                }
            )
    return trials


def _metric_value(row: Mapping[str, Any], key: str):
    value = row.get(key, "")
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _cell_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        cell_id = str(row.get("cell_id", ""))
        if cell_id:
            grouped.setdefault(cell_id, []).append(row)

    cell_order = {str(spec["cell_id"]): index for index, spec in enumerate(_cell_specs())}
    summaries: List[Dict[str, Any]] = []
    for cell_id in sorted(grouped, key=lambda item: cell_order.get(item, 999)):
        cell_rows = grouped[cell_id]
        successful = [row for row in cell_rows if int(row.get("returncode", 0)) == 0]
        first = cell_rows[0]
        summary: Dict[str, Any] = {
            "cell_id": cell_id,
            "cell_name": first.get("cell_name", ""),
            "graph_gp": first.get("graph_gp", ""),
            "attention_mediation": first.get("attention_mediation", ""),
            "semantic_tokens": first.get("semantic_tokens", ""),
            "requested_runs": len(cell_rows),
            "successful_runs": len(successful),
            "seeds": ",".join(str(row.get("seed", "")) for row in cell_rows),
        }
        for metric in SUMMARY_METRICS:
            values = [
                value
                for row in successful
                for value in [_metric_value(row, metric)]
                if value is not None
            ]
            if values:
                mean_value, std_value = _mean_std(values)
                summary[f"{metric}_mean"] = mean_value
                summary[f"{metric}_std"] = std_value
        summaries.append(summary)
    return summaries


def _paired_effect_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    successful = [row for row in rows if int(row.get("returncode", 0)) == 0]
    by_seed_cell = {
        (int(row["seed"]), str(row["cell_id"])): row
        for row in successful
        if row.get("seed", "") != "" and row.get("cell_id", "") != ""
    }
    seeds = sorted({seed for seed, _ in by_seed_cell})
    effects: List[Dict[str, Any]] = []
    for seed in seeds:
        for effect_name, (lhs_cell, rhs_cell) in PAIRED_EFFECTS.items():
            lhs = by_seed_cell.get((seed, lhs_cell))
            rhs = by_seed_cell.get((seed, rhs_cell))
            if lhs is None or rhs is None:
                continue
            effect: Dict[str, Any] = {
                "effect": effect_name,
                "seed": int(seed),
                "lhs_cell": lhs_cell,
                "rhs_cell": rhs_cell,
            }
            for metric in SUMMARY_METRICS:
                lhs_value = _metric_value(lhs, metric)
                rhs_value = _metric_value(rhs, metric)
                if lhs_value is not None and rhs_value is not None:
                    effect[f"{metric}_delta"] = lhs_value - rhs_value
            effects.append(effect)

        required = {
            cell_id: by_seed_cell.get((seed, cell_id))
            for cell_id in ("C00", "C10", "C01", "C11")
        }
        if all(row is not None for row in required.values()):
            interaction: Dict[str, Any] = {
                "effect": "graph_gp_am_interaction",
                "seed": int(seed),
                "lhs_cell": "(C11-C01)",
                "rhs_cell": "(C10-C00)",
            }
            for metric in SUMMARY_METRICS:
                values = {
                    cell_id: _metric_value(row, metric)
                    for cell_id, row in required.items()
                }
                if all(value is not None for value in values.values()):
                    interaction[f"{metric}_delta"] = (
                        values["C11"] - values["C01"]
                    ) - (
                        values["C10"] - values["C00"]
                    )
            effects.append(interaction)

        graph_semantic_required = {
            cell_id: by_seed_cell.get((seed, cell_id))
            for cell_id in ("A00", "A10", "C00", "C10")
        }
        if all(row is not None for row in graph_semantic_required.values()):
            interaction = {
                "effect": "graph_gp_semantic_interaction_no_am",
                "seed": int(seed),
                "lhs_cell": "(C10-C00)",
                "rhs_cell": "(A10-A00)",
            }
            for metric in SUMMARY_METRICS:
                values = {
                    cell_id: _metric_value(row, metric)
                    for cell_id, row in graph_semantic_required.items()
                }
                if all(value is not None for value in values.values()):
                    interaction[f"{metric}_delta"] = (
                        values["C10"] - values["C00"]
                    ) - (
                        values["A10"] - values["A00"]
                    )
            effects.append(interaction)
    return effects


def _paired_effect_summary_rows(effect_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in effect_rows:
        grouped.setdefault(str(row["effect"]), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for effect_name in list(PAIRED_EFFECTS) + ["graph_gp_am_interaction"]:
        rows_for_effect = grouped.get(effect_name, [])
        if not rows_for_effect:
            continue
        summary: Dict[str, Any] = {
            "effect": effect_name,
            "seed_count": len(rows_for_effect),
            "seeds": ",".join(str(row["seed"]) for row in rows_for_effect),
        }
        for metric in SUMMARY_METRICS:
            key = f"{metric}_delta"
            values = [
                value
                for row in rows_for_effect
                for value in [_metric_value(row, key)]
                if value is not None
            ]
            if values:
                mean_value, std_value = _mean_std(values)
                summary[f"{key}_mean"] = mean_value
                summary[f"{key}_std"] = std_value
        summaries.append(summary)
    return summaries


def _validate_gpu_args(gpu_groups: Sequence[str], max_workers: int, nproc_per_trial: int) -> None:
    if int(max_workers) <= 0:
        raise ValueError("--max-workers must be positive.")
    if int(max_workers) > len(gpu_groups):
        raise ValueError("--max-workers must not exceed the number of GPU groups.")
    for gpu_group in gpu_groups:
        nproc = _group_nproc(gpu_group, int(nproc_per_trial))
        visible_gpu_count = len([item for item in str(gpu_group).split(",") if item.strip()])
        if gpu_group and nproc > 1 and nproc != visible_gpu_count:
            raise ValueError("--nproc-per-trial must match each --gpu-groups item size when CUDA_VISIBLE_DEVICES is set.")


def _write_commands(
    commands_path: Path,
    trials: Sequence[Mapping[str, Any]],
    python_bin: str,
    gpu_groups: Sequence[str],
    nproc_per_trial: int,
    dist_backend: str,
) -> None:
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    with commands_path.open("w", encoding="utf-8") as handle:
        for idx, trial in enumerate(trials):
            gpu = gpu_groups[idx % len(gpu_groups)] if gpu_groups else ""
            run_cmd, _ = _build_run_command(
                trial,
                python_bin=python_bin,
                gpu_group=gpu,
                nproc_per_trial=int(nproc_per_trial),
                dist_backend=str(dist_backend),
            )
            handle.write(subprocess.list2cmdline(run_cmd) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run valid C00/C10/C01/C11/A00/A10 cells across three seeds (18 runs by default)."
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--cells",
        "--trials",
        dest="cells",
        nargs="+",
        default=[],
        help="Cell ids/names separated by commas or spaces. Default: C00,C10,C01,C11,A00,A10.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[",".join(str(seed) for seed in DEFAULT_SEEDS)],
        help="Experiment seeds separated by commas or spaces. Default: 17,29,43.",
    )
    parser.add_argument(
        "--protocol-mode",
        default="final_gzsl",
        choices=["dev", "final_gzsl"],
        help="XLSA protocol for all runs. Default: final_gzsl; pass dev explicitly for dev-unseen evaluation.",
    )
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids for independent trials.")
    parser.add_argument(
        "--gpu-groups",
        default="",
        help="Semicolon-separated GPU groups. Example: '0,1;2,3' runs two DDP trials.",
    )
    parser.add_argument(
        "--nproc-per-trial",
        type=int,
        default=0,
        help="DDP process count per trial. Default: GPU count in each --gpu-groups item; 0 keeps --gpus single-process.",
    )
    parser.add_argument("--dist-backend", default=_default_dist_backend(), choices=["nccl", "gloo"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N expanded cell/seed trials; 0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Rerun trials even when complete train logs already exist.")
    parser.add_argument("--resume-debug", action="store_true", help="Print why an existing trial was or was not skipped.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra KEY VALUE config overrides appended to every child run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.protocol_mode) == "dev":
        print(
            "[stage1] protocol=dev uses the repository's current dev-unseen evaluation; "
            "dev-seen/dev-H require the separate dev-GZSL infrastructure described in the experiment plan.",
            flush=True,
        )
    else:
        print(
            "[stage1] protocol=final_gzsl was explicitly requested; the current trainer evaluates "
            "test seen/unseen every epoch, so do not use best-test epoch selection.",
            flush=True,
        )
    repo_root = Path(args.repo_root).resolve()
    python_bin = str(args.python_bin or sys.executable)
    config_file = str(args.config_file)
    if not Path(config_file).is_absolute():
        config_file = str(repo_root / config_file)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_root = out_root / str(args.protocol_mode)

    extra_opts = _validate_extra_opts(args.opts)
    selected_specs = _select_cell_specs(args.cells)
    seeds = _parse_seeds(args.seeds)
    gpu_groups = _parse_gpu_groups(args.gpu_groups, args.gpus)
    _validate_gpu_args(gpu_groups, int(args.max_workers), int(args.nproc_per_trial))

    trials = _build_trials(
        repo_root=repo_root,
        python_bin=python_bin,
        config_file=config_file,
        out_root=out_root,
        protocol_mode=str(args.protocol_mode),
        seeds=seeds,
        extra_opts=extra_opts,
        selected_specs=selected_specs,
    )
    if int(args.limit) > 0:
        trials = trials[: int(args.limit)]
    out_root.mkdir(parents=True, exist_ok=True)
    commands_path = out_root / "commands.txt"
    _write_commands(
        commands_path=commands_path,
        trials=trials,
        python_bin=python_bin,
        gpu_groups=gpu_groups,
        nproc_per_trial=int(args.nproc_per_trial),
        dist_backend=str(args.dist_backend),
    )

    search_space = {
        "source": "stage1_clean_graph_gp_am_plan",
        "base_config_file": config_file,
        "train_script": TRAIN_SCRIPT,
        "runner": "train",
        "python_bin": python_bin,
        "output_root": str(out_root),
        "total_trials": len(trials),
        "default_design_trials": len(_cell_specs()) * len(DEFAULT_SEEDS),
        "protocol_mode": str(args.protocol_mode),
        "seeds": seeds,
        "dry_run": bool(args.dry_run),
        "resume": not bool(args.no_resume),
        "gpu_groups": gpu_groups,
        "nproc_per_trial": int(args.nproc_per_trial),
        "dist_backend": str(args.dist_backend),
        "max_workers": int(args.max_workers),
        "extra_opts": extra_opts,
        "commands_path": str(commands_path),
        "cells": [
            {
                "cell_id": str(spec["cell_id"]),
                "cell_name": str(spec["cell_name"]),
                "graph_gp": bool(spec["graph_gp"]),
                "attention_mediation": bool(spec["attention_mediation"]),
                "semantic_tokens": bool(spec["semantic_tokens"]),
                "core_2x2": bool(spec["core_2x2"]),
                "purpose": str(spec["purpose"]),
            }
            for spec in selected_specs
        ],
        "trials": [
            {
                "trial_name": str(trial["trial_name"]),
                "cell_id": str(trial["combo"]["cell_id"]),
                "seed": int(trial["combo"]["seed"]),
                "output_dir": str(trial["output_dir"]),
            }
            for trial in trials
        ],
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
    elif int(args.max_workers) == 1:
        rows = []
        for idx, trial in enumerate(trials):
            gpu = gpu_groups[idx % len(gpu_groups)] if gpu_groups else ""
            nproc = _group_nproc(gpu, int(args.nproc_per_trial))
            run_cmd, run_nproc = _build_run_command(
                trial,
                python_bin=python_bin,
                gpu_group=gpu,
                nproc_per_trial=int(args.nproc_per_trial),
                dist_backend=str(args.dist_backend),
            )
            existing = None if bool(args.no_resume) else _existing_trial_result(
                trial,
                gpu,
                run_cmd,
                run_nproc,
                expected_batches=0,
            )
            if existing is not None:
                print(f"[{idx + 1}/{len(trials)}] skip existing {trial['trial_name']} gpu={gpu} nproc={run_nproc}", flush=True)
                rows.append(existing)
                continue
            if bool(args.resume_debug) and not bool(args.no_resume):
                print(f"[{idx + 1}/{len(trials)}] resume miss {trial['trial_name']}: {_resume_status(trial)}", flush=True)
            print(f"[{idx + 1}/{len(trials)}] {trial['trial_name']} gpu={gpu} nproc={nproc}", flush=True)
            rows.append(
                _run_trial(
                    trial,
                    gpu_id=gpu,
                    python_bin=python_bin,
                    nproc_per_trial=int(args.nproc_per_trial),
                    dist_backend=str(args.dist_backend),
                    expected_batches=0,
                    resume=not bool(args.no_resume),
                )
            )
    else:
        rows = []
        worker_count = int(args.max_workers)
        worker_gpus = gpu_groups[:worker_count] if gpu_groups else [""] * worker_count

        def _run_one_on_worker(
            worker_idx: int,
            gpu: str,
            trial_idx: int,
            trial: Mapping[str, Any],
        ) -> Dict[str, Any]:
            run_cmd, run_nproc = _build_run_command(
                trial,
                python_bin=python_bin,
                gpu_group=gpu,
                nproc_per_trial=int(args.nproc_per_trial),
                dist_backend=str(args.dist_backend),
            )
            existing = None if bool(args.no_resume) else _existing_trial_result(
                trial,
                gpu,
                run_cmd,
                run_nproc,
                expected_batches=0,
            )
            if existing is not None:
                print(
                    f"[worker {worker_idx + 1}/{worker_count} gpu={gpu}] "
                    f"skip existing {trial_idx + 1}/{len(trials)} {trial['trial_name']} nproc={run_nproc}",
                    flush=True,
                )
                return existing
            if bool(args.resume_debug) and not bool(args.no_resume):
                print(
                    f"[worker {worker_idx + 1}/{worker_count} gpu={gpu}] "
                    f"resume miss {trial_idx + 1}/{len(trials)} {trial['trial_name']}: {_resume_status(trial)}",
                    flush=True,
                )
            print(
                f"[worker {worker_idx + 1}/{worker_count} gpu={gpu}] "
                f"started {trial_idx + 1}/{len(trials)} {trial['trial_name']}",
                flush=True,
            )
            result = _run_trial(
                trial,
                gpu_id=gpu,
                python_bin=python_bin,
                nproc_per_trial=int(args.nproc_per_trial),
                dist_backend=str(args.dist_backend),
                expected_batches=0,
                resume=not bool(args.no_resume),
            )
            print(
                f"[worker {worker_idx + 1}/{worker_count} gpu={gpu}] "
                f"finished {trial_idx + 1}/{len(trials)} {trial['trial_name']} "
                f"returncode={result['returncode']}",
                flush=True,
            )
            return result

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending_trial_idx = 0
            future_to_worker: Dict[Any, int] = {}

            def _submit_next(worker_idx: int) -> bool:
                nonlocal pending_trial_idx
                if pending_trial_idx >= len(trials):
                    return False
                trial_idx = pending_trial_idx
                pending_trial_idx += 1
                gpu = worker_gpus[worker_idx]
                future = executor.submit(
                    _run_one_on_worker,
                    worker_idx,
                    gpu,
                    trial_idx,
                    trials[trial_idx],
                )
                future_to_worker[future] = worker_idx
                return True

            for worker_idx in range(worker_count):
                _submit_next(worker_idx)

            finished = 0
            while future_to_worker:
                for future in as_completed(list(future_to_worker.keys())):
                    worker_idx = future_to_worker.pop(future)
                    rows.append(future.result())
                    finished += 1
                    print(f"[{finished}/{len(trials)}] collected worker result", flush=True)
                    _submit_next(worker_idx)
                    break
        rows.sort(key=lambda row: int(row["trial_index"]))

    ranked_rows = _rank_rows(rows)
    cell_summaries = _cell_summary_rows(rows)
    paired_effects = _paired_effect_rows(rows)
    paired_effect_summaries = _paired_effect_summary_rows(paired_effects)
    _write_csv(out_root / "summary.csv", ranked_rows)
    _write_json(out_root / "summary.json", {"search_space": search_space, "rows": ranked_rows})
    _write_csv(out_root / "ranked_summary.csv", ranked_rows)
    _write_json(out_root / "ranked_summary.json", {"search_space": search_space, "rows": ranked_rows})
    _write_csv(out_root / "cell_summary.csv", cell_summaries)
    _write_json(out_root / "cell_summary.json", {"rows": cell_summaries})
    _write_csv(out_root / "paired_effects.csv", paired_effects)
    _write_json(out_root / "paired_effects.json", {"rows": paired_effects})
    _write_csv(out_root / "paired_effect_summary.csv", paired_effect_summaries)
    _write_json(out_root / "paired_effect_summary.json", {"rows": paired_effect_summaries})

    failures = [row for row in rows if int(row.get("returncode", 0)) not in {0, -1}]
    if failures:
        _write_json(out_root / "failures.json", {"failures": failures})
        raise SystemExit(f"{len(failures)} trials failed; see {out_root / 'failures.json'}")
    print(f"wrote {out_root / 'summary.csv'}")
    print(f"wrote {out_root / 'summary.json'}")
    print(f"wrote {out_root / 'cell_summary.csv'}")
    print(f"wrote {out_root / 'paired_effect_summary.csv'}")
    print(f"wrote {commands_path}")


if __name__ == "__main__":
    main()
