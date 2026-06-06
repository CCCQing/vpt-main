#!/usr/bin/env python3

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from baseline_grid_search import (
    _find_run_dir,
    _format_duration,
    _has_valid_score,
    _parse_bool,
    _parse_float_list,
    _parse_metrics,
    _parse_str_list,
    _read_tail,
    _run_cmd,
    _score_key,
    _trial_name,
    _trial_tag,
    _validate_extra_opts,
    _write_summary_csv,
)


def _default_dist_backend() -> str:
    return "gloo" if os.name == "nt" else "nccl"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_train_main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser("baseline_grid_search_multigpu ddp train")
    ap.add_argument("--nproc-per-node", type=int, required=True)
    ap.add_argument("--dist-backend", default=_default_dist_backend())
    ap.add_argument("--dist-url", default="")
    known, train_argv = ap.parse_known_args(argv)
    if known.nproc_per_node <= 1:
        raise ValueError("--nproc-per-node must be greater than 1 in DDP train mode.")
    if not train_argv:
        raise ValueError("Missing train.py arguments after DDP launcher options.")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import torch.multiprocessing as mp

    from launch import default_argument_parser
    from src.utils import distributed as du
    from train import setup

    train_args = default_argument_parser().parse_args(train_argv)
    cfg = setup(train_args)
    if int(cfg.NUM_GPUS) != int(known.nproc_per_node):
        raise ValueError(
            "NUM_GPUS must match --nproc-per-node, got NUM_GPUS={} and nproc={}.".format(
                cfg.NUM_GPUS,
                known.nproc_per_node,
            )
        )

    init_method = known.dist_url or "tcp://127.0.0.1:{}".format(_pick_free_port())
    mp.spawn(
        du.run,
        nprocs=known.nproc_per_node,
        args=(
            known.nproc_per_node,
            _train_with_ddp_attr_forward,
            init_method,
            0,
            1,
            known.dist_backend,
            cfg,
            train_args,
        ),
        join=True,
    )


def _train_with_ddp_attr_forward(cfg, args) -> None:
    import torch

    ddp_cls = torch.nn.parallel.DistributedDataParallel
    if not getattr(ddp_cls, "_vpt_forward_missing_attrs", False):
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
        ddp_cls._vpt_forward_missing_attrs = True

    from train import train

    train(cfg, args)


def _parse_gpu_groups(raw_groups: str, raw_gpus: str) -> List[str]:
    if raw_groups.strip():
        groups = [item.strip() for item in raw_groups.split(";") if item.strip()]
    elif raw_gpus.strip():
        groups = [raw_gpus.strip()]
    else:
        groups = [""]
    return groups


def _group_nproc(group: str, fallback_nproc: int) -> int:
    if fallback_nproc > 0:
        return fallback_nproc
    if not group:
        return 1
    return len([item for item in group.split(",") if item.strip()])


def _make_base_opts(
    tau_acc: float,
    tau_sem: float,
    tau_prompt: float,
    prompt_scale_learnable: bool,
    extra_opts: List[str],
) -> List[str]:
    base_opts = [
        "MODEL.PROMPT.BACKEND", "dynamic",
        "MODEL.PROMPT.INIT_SOURCE", "distributor_mean",
        "MODEL.PROMPT.NUM_TOKENS", "32",
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE", "True",
        "MODEL.PROMPT.DISTRIBUTOR.SOURCE", "vit_cls_prepass",
        "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM", "64",
        "MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS", "16",
        "MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS", "16",
        "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE", "fixed_eps",
        "MODEL.PROMPT.DISTRIBUTOR.FIXED_EPS_SEED", "0",
        "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED", "True",
        "MODEL.SEMANTIC_GRAPH.ENABLE", "True",
        "MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE", "fuse",
        "MODEL.SEMANTIC_GRAPH.RHO", "0.0",
        "MODEL.SEMANTIC_GRAPH.TOPK", "16",
        "MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA", "0.1",
        "MODEL.SEMANTIC_GRAPH.LOSS_TYPE", "fgw",
        "MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT", "0.01",
        "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE", "mu",
        "SOLVER.LOSS_PROMPT_KL_WEIGHT", "0.0",
        "MODEL.SEMANTIC_GRAPH.OT_EPS", "0.05",
        "MODEL.SEMANTIC_GRAPH.OT_ITERS", "20",
        "MODEL.SEMANTIC_GRAPH.OT_ALPHA", "0.5",
        "MODEL.SEMANTIC_GRAPH.OT_DELTA", "1e-8",
        "MODEL.SEMANTIC_GRAPH.OT_PRIOR_ETA", "1.0",
        "MODEL.SEMANTIC_GRAPH.OT_DETACH_PLAN", "True",
        "MODEL.SEMANTIC_GRAPH.TAU_ACC", str(tau_acc),
        "MODEL.SEMANTIC_GRAPH.TAU_SEM", str(tau_sem),
        "MODEL.SEMANTIC_GRAPH.TAU_PROMPT", str(tau_prompt),
        "MODEL.SEMANTIC_GRAPH.PROMPT_SCALE_LEARNABLE", str(prompt_scale_learnable),
        "MODEL.SEMANTIC_TOKENS.ENABLE", "True",
        "MODEL.SEMANTIC_TOKENS.TRAIN_SOURCE", "class_mean",
        "MODEL.SEMANTIC_TOKENS.EVAL_SOURCE", "class_mean",
        "MODEL.SEMANTIC_TOKENS.TOKENIZER", "orthogonal",
        "MODEL.SEMANTIC_TOKENS.NUM_TOKENS", "8",
        "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE", "manual_cub8",
        "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE", "null_residual",
        "MODEL.AFFINITY_EVOLUTION.ENABLE", "False",
        "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA", "0.0",
        "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA", "0.0",
        "MODEL.ATTENTION_MEDIATION.ENABLE", "True",
        "MODEL.ATTENTION_MEDIATION.BETA_PROMPT_MASS", "1.0",
        "MODEL.ATTENTION_MEDIATION.BETA_SEMANTIC_MASS", "1.0",
        "DATA.BATCH_SIZE", "64",
        "SOLVER.BASE_LR", "0.00125",
        "SOLVER.TOTAL_EPOCH", "40",
        "SOLVER.WARMUP_EPOCH", "3",
        "SOLVER.LOSS_ATTR_WEIGHT", "0.001",
        "SOLVER.ATTR.METRIC", "mse",
        "SOLVER.LOSS_SEM_MED_WEIGHT", "0.0",
        "SOLVER.LOSS_SPV_WEIGHT", "0.0",
        "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT", "0.0",
        "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT", "0.0",
        "SOLVER.VIS.ENABLE", "False",
        "SOLVER.VIS.SAVE_RAW", "False",
        "SOLVER.VIS.SAVE_IMAGES", "False",
        "SOLVER.VIS.ROLLOUT", "False",
    ]
    if extra_opts:
        _validate_extra_opts(extra_opts)
        base_opts.extend(extra_opts)
    return base_opts


def _build_trials(
    tau_acc: float,
    tau_sem: float,
    tau_prompt: float,
    prompt_scale_learnable: bool,
) -> List[Dict[str, object]]:
    trials: List[Dict[str, object]] = []
    graph_source = "fuse"
    graph_loss_weight = 0.01
    target_mix_alpha = 0.1
    topk = 16

    def add_trial(
        *,
        source: str,
        execution_mode: str,
        mlp_policy: str,
        route_scope: str,
        mass_mode: str,
        prompt_route: str,
        semantic_route: str,
    ) -> None:
        params = {
            "src": source,
            "exec": execution_mode,
            "mlp": mlp_policy,
            "scope": route_scope,
            "mass": mass_mode,
            "pr": prompt_route,
            "sr": semantic_route,
        }
        opts = [
            "MODEL.ATTENTION_MEDIATION.SOURCE", source,
            "MODEL.ATTENTION_MEDIATION.EXECUTION_MODE", execution_mode,
            "MODEL.ATTENTION_MEDIATION.MLP_POLICY", mlp_policy,
            "MODEL.ATTENTION_MEDIATION.ROUTE_SCOPE", route_scope,
            "MODEL.ATTENTION_MEDIATION.MASS_MODE", mass_mode,
            "MODEL.ATTENTION_MEDIATION.PROMPT_ROUTE", prompt_route,
            "MODEL.ATTENTION_MEDIATION.SEMANTIC_ROUTE", semantic_route,
        ]
        trials.append(
            {
                "group": "attention_mediation_72_multigpu",
                "tag": _trial_tag("am", params),
                "prompt_mode": "attention_mediation",
                "prompt_backend": "dynamic",
                "prompt_init_source": "distributor_mean",
                "prompt_distributor_enable": True,
                "distributor_source": "vit_cls_prepass",
                "stats_hidden_dim": 64,
                "instance_tokens": 16,
                "domain_tokens": 16,
                "eval_sample_mode": "fixed_eps",
                "use_slot_embed": True,
                "prompt_stat_source": "mu",
                "semantic_graph_enable": True,
                "graph_source": graph_source,
                "graph_rho": 0.0,
                "graph_topk": topk,
                "graph_loss_type": "fgw",
                "graph_loss_weight": graph_loss_weight,
                "target_mix_alpha": target_mix_alpha,
                "tau_acc": tau_acc,
                "tau_sem": tau_sem,
                "tau_prompt": tau_prompt,
                "prompt_scale_learnable": prompt_scale_learnable,
                "prompt_kl_weight": 0.0,
                "affinity_evolution_enable": False,
                "prompt_lambda": 0.0,
                "semantic_lambda": 0.0,
                "route_ts_prompt_weight": 0.0,
                "route_ts_semantic_weight": 0.0,
                "attention_mediation_enable": True,
                "attention_source": source,
                "execution_mode": execution_mode,
                "mlp_policy": mlp_policy,
                "route_scope": route_scope,
                "mass_mode": mass_mode,
                "prompt_route": prompt_route,
                "semantic_route": semantic_route,
                "beta_prompt_mass": 1.0,
                "beta_semantic_mass": 1.0,
                "opts": opts,
            }
        )

    execution_mlp_cases = [
        ("attention_parallel", "enter_mlp"),
        ("attention_parallel", "skip_mlp"),
        ("block_parallel", "enter_mlp"),
    ]
    mass_cases = [
        ("visual_block", "row_preserve"),
        ("visual_block", "block_redistribute"),
        ("full_row", "row_preserve"),
    ]
    for source in ["probs", "scores"]:
        for execution_mode, mlp_policy in execution_mlp_cases:
            for route_scope, mass_mode in mass_cases:
                for prompt_route in ["S_to_P_and_V", "P_to_S_to_V"]:
                    for semantic_route in ["S_to_P_to_V", "P_to_S_and_V"]:
                        add_trial(
                            source=source,
                            execution_mode=execution_mode,
                            mlp_policy=mlp_policy,
                            route_scope=route_scope,
                            mass_mode=mass_mode,
                            prompt_route=prompt_route,
                            semantic_route=semantic_route,
                        )
    return trials


def _write_search_space(
    path: str,
    args: argparse.Namespace,
    out_root: str,
    trials: List[Dict[str, object]],
    gpu_groups: List[str],
) -> None:
    source_best_trial = "exp047_pdgraph_graph_True_src_vit_cls_prepass_h_64_kl_0_eval_fixed_eps_slot_True_stat_mu_rho_0_loss_fgw"
    search_space = {
        "config_file": args.config_file,
        "python_bin": args.python_bin,
        "out_root": out_root,
        "total_trials": len(trials),
        "source_best_trial": source_best_trial,
        "launcher": "baseline_grid_search_multigpu",
        "ddp": {
            "gpu_groups": gpu_groups,
            "nproc_per_trial": args.nproc_per_trial,
            "dist_backend": args.dist_backend,
            "max_workers": args.max_workers,
            "batch_size_is_global": True,
        },
        "sweep": [
            "MODEL.ATTENTION_MEDIATION.SOURCE",
            "MODEL.ATTENTION_MEDIATION.EXECUTION_MODE",
            "MODEL.ATTENTION_MEDIATION.MLP_POLICY",
            "MODEL.ATTENTION_MEDIATION.ROUTE_SCOPE",
            "MODEL.ATTENTION_MEDIATION.MASS_MODE",
            "MODEL.ATTENTION_MEDIATION.PROMPT_ROUTE",
            "MODEL.ATTENTION_MEDIATION.SEMANTIC_ROUTE",
        ],
        "groups": {
            "attention_mediation_72_multigpu": {
                "count": 72,
                "MODEL.ATTENTION_MEDIATION.SOURCE": ["probs", "scores"],
                "execution_mlp_cases": [
                    {"MODEL.ATTENTION_MEDIATION.EXECUTION_MODE": "attention_parallel", "MODEL.ATTENTION_MEDIATION.MLP_POLICY": "enter_mlp"},
                    {"MODEL.ATTENTION_MEDIATION.EXECUTION_MODE": "attention_parallel", "MODEL.ATTENTION_MEDIATION.MLP_POLICY": "skip_mlp"},
                    {"MODEL.ATTENTION_MEDIATION.EXECUTION_MODE": "block_parallel", "MODEL.ATTENTION_MEDIATION.MLP_POLICY": "enter_mlp"},
                ],
                "mass_cases": [
                    {"MODEL.ATTENTION_MEDIATION.ROUTE_SCOPE": "visual_block", "MODEL.ATTENTION_MEDIATION.MASS_MODE": "row_preserve"},
                    {"MODEL.ATTENTION_MEDIATION.ROUTE_SCOPE": "visual_block", "MODEL.ATTENTION_MEDIATION.MASS_MODE": "block_redistribute"},
                    {"MODEL.ATTENTION_MEDIATION.ROUTE_SCOPE": "full_row", "MODEL.ATTENTION_MEDIATION.MASS_MODE": "row_preserve"},
                ],
                "MODEL.ATTENTION_MEDIATION.PROMPT_ROUTE": ["S_to_P_and_V", "P_to_S_to_V"],
                "MODEL.ATTENTION_MEDIATION.SEMANTIC_ROUTE": ["S_to_P_to_V", "P_to_S_and_V"],
            },
        },
        "fixed_setting": {
            "MODEL.PROMPT.BACKEND": "dynamic",
            "MODEL.PROMPT.INIT_SOURCE": "distributor_mean",
            "MODEL.PROMPT.NUM_TOKENS": 32,
            "MODEL.PROMPT.DISTRIBUTOR.ENABLE": True,
            "MODEL.PROMPT.DISTRIBUTOR.SOURCE": "vit_cls_prepass",
            "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM": 64,
            "MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS": 16,
            "MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS": 16,
            "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": "fixed_eps",
            "MODEL.PROMPT.DISTRIBUTOR.FIXED_EPS_SEED": 0,
            "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED": True,
            "MODEL.SEMANTIC_GRAPH.ENABLE": True,
            "MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE": "fuse",
            "MODEL.SEMANTIC_GRAPH.RHO": 0.0,
            "MODEL.SEMANTIC_GRAPH.TOPK": 16,
            "MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA": 0.1,
            "MODEL.SEMANTIC_GRAPH.LOSS_TYPE": "fgw",
            "MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT": 0.01,
            "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE": "mu",
            "MODEL.SEMANTIC_GRAPH.OT_EPS": 0.05,
            "MODEL.SEMANTIC_GRAPH.OT_ITERS": 20,
            "MODEL.SEMANTIC_GRAPH.OT_ALPHA": 0.5,
            "MODEL.SEMANTIC_GRAPH.OT_DELTA": 1e-8,
            "MODEL.SEMANTIC_GRAPH.OT_PRIOR_ETA": 1.0,
            "MODEL.SEMANTIC_GRAPH.OT_DETACH_PLAN": True,
            "MODEL.SEMANTIC_TOKENS.ENABLE": True,
            "MODEL.SEMANTIC_TOKENS.TRAIN_SOURCE": "class_mean",
            "MODEL.SEMANTIC_TOKENS.EVAL_SOURCE": "class_mean",
            "MODEL.SEMANTIC_TOKENS.TOKENIZER": "orthogonal",
            "MODEL.SEMANTIC_TOKENS.NUM_TOKENS": 8,
            "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE": "manual_cub8",
            "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE": "null_residual",
            "MODEL.AFFINITY_EVOLUTION.ENABLE": False,
            "MODEL.ATTENTION_MEDIATION.ENABLE": True,
            "MODEL.ATTENTION_MEDIATION.BETA_PROMPT_MASS": 1.0,
            "MODEL.ATTENTION_MEDIATION.BETA_SEMANTIC_MASS": 1.0,
            "DATA.BATCH_SIZE": 64,
            "SOLVER.BASE_LR": 0.00125,
            "SOLVER.TOTAL_EPOCH": 40,
            "SOLVER.WARMUP_EPOCH": 3,
            "SOLVER.LOSS_ATTR_WEIGHT": 0.001,
            "SOLVER.ATTR.METRIC": "mse",
            "SOLVER.LOSS_PROMPT_KL_WEIGHT": 0.0,
            "SOLVER.LOSS_SEM_MED_WEIGHT": 0.0,
            "SOLVER.LOSS_SPV_WEIGHT": 0.0,
            "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT": 0.0,
            "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT": 0.0,
            "SOLVER.VIS.ENABLE": False,
            "SOLVER.VIS.SAVE_RAW": False,
            "SOLVER.VIS.SAVE_IMAGES": False,
            "SOLVER.VIS.ROLLOUT": False,
        },
        "extra_opts": args.opts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser("attention_mediation_multigpu_grid_search")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--config-file", default="configs/prompt/cub.yaml")
    ap.add_argument("--out-root", default="output/grid_attention_mediation_multigpu")
    ap.add_argument("--gpus", default="", help="GPU ids for one DDP trial, e.g. 0,1.")
    ap.add_argument("--gpu-groups", default="", help="Semicolon-separated DDP groups, e.g. 0,1;2,3.")
    ap.add_argument("--nproc-per-trial", type=int, default=0, help="Override process count per trial. Default: size of GPU group.")
    ap.add_argument("--dist-backend", default=_default_dist_backend(), choices=["nccl", "gloo"])
    ap.add_argument("--max-workers", type=int, default=1, help="Concurrent DDP trials. Needs separate --gpu-groups for values > 1.")
    ap.add_argument("--tau-acc", default="1.0")
    ap.add_argument("--tau-sem", default="1.0")
    ap.add_argument("--tau-prompt", default="1.0")
    ap.add_argument("--prompt-scale-learnable", default="true", choices=["true", "false"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")

    repo_root = os.path.abspath(args.repo_root)
    out_root = os.path.abspath(os.path.join(repo_root, args.out_root))
    os.makedirs(out_root, exist_ok=True)

    tau_acc = float(args.tau_acc)
    tau_sem = float(args.tau_sem)
    tau_prompt = float(args.tau_prompt)
    prompt_scale_learnable = _parse_bool(args.prompt_scale_learnable)
    gpu_groups = _parse_gpu_groups(args.gpu_groups, args.gpus)
    if args.max_workers > len(gpu_groups):
        raise ValueError("--max-workers must not exceed the number of GPU groups.")

    for group in gpu_groups:
        nproc = _group_nproc(group, int(args.nproc_per_trial))
        if nproc <= 1:
            raise ValueError("Multi-GPU grid requires at least 2 processes per trial. Pass --gpus 0,1 or --gpu-groups 0,1.")
        if group and nproc != len([item for item in group.split(",") if item.strip()]):
            raise ValueError("--nproc-per-trial must match each GPU group size when CUDA_VISIBLE_DEVICES is set.")

    base_opts = _make_base_opts(tau_acc, tau_sem, tau_prompt, prompt_scale_learnable, args.opts)
    trials = _build_trials(tau_acc, tau_sem, tau_prompt, prompt_scale_learnable)
    _write_search_space(os.path.join(out_root, "search_space.json"), args, out_root, trials, gpu_groups)

    rows: List[Dict[str, object]] = []
    grid_start = time.time()
    total_trials = len(trials)
    script_path = os.path.abspath(__file__)

    def build_trial(idx: int, trial: Dict[str, object], gpu_group: str) -> Tuple[str, str, str, List[str], Dict[str, object], int]:
        nproc = _group_nproc(gpu_group, int(args.nproc_per_trial))
        trial_name = _trial_name(idx, str(trial["tag"]))
        trial_root = os.path.join(out_root, trial_name)
        os.makedirs(trial_root, exist_ok=True)
        stdout_path = os.path.join(trial_root, "launcher_stdout.txt")
        cmd = [
            args.python_bin,
            script_path,
            "--ddp-train",
            "--nproc-per-node",
            str(nproc),
            "--dist-backend",
            args.dist_backend,
            "--config-file",
            args.config_file,
            "OUTPUT_DIR",
            trial_root,
            "NUM_GPUS",
            str(nproc),
            "RUN_N_TIMES",
            "1",
        ] + base_opts + list(trial["opts"])
        row: Dict[str, object] = dict(trial)
        row.pop("opts", None)
        row.update(
            {
                "trial_name": trial_name,
                "gpu": gpu_group,
                "num_gpus": nproc,
                "exit_code": -1,
                "run_dir": "",
            }
        )
        return trial_name, trial_root, stdout_path, cmd, row, nproc

    def finish_trial(
        trial_name: str,
        trial_root: str,
        stdout_path: str,
        row: Dict[str, object],
        code: int,
        trial_start: float,
        completed: int,
    ) -> None:
        row["exit_code"] = code
        if code != 0:
            print(f"[grid-ddp] failure stdout tail for {trial_name}:", flush=True)
            print(_read_tail(stdout_path), flush=True)

        run_dir = _find_run_dir(trial_root)
        if run_dir is not None:
            metrics = _parse_metrics(os.path.join(run_dir, "logs.txt"))
            score_key, score = _score_key(metrics)
            row.update(metrics)
            row["score_key"] = score_key
            row["score"] = score
            row["run_dir"] = run_dir
        else:
            row["score_key"] = "missing_logs"
            row["score"] = ""

        rows.append(row)
        elapsed = time.time() - trial_start
        total_elapsed = time.time() - grid_start
        avg_elapsed = total_elapsed / float(max(1, completed))
        eta = avg_elapsed * float(total_trials - completed)
        print(
            f"[grid-ddp] done {completed}/{total_trials}: {trial_name} exit_code={code} "
            f"trial_time={_format_duration(elapsed)} total_time={_format_duration(total_elapsed)} eta={_format_duration(eta)}",
            flush=True,
        )
        _write_summary_csv(os.path.join(out_root, "summary.csv"), rows)
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    pending = list(enumerate(trials, start=1))
    active: List[Dict[str, object]] = []
    completed = 0

    while pending or active:
        while pending and len(active) < args.max_workers:
            idx, trial = pending.pop(0)
            used_groups = {str(item["gpu"]) for item in active}
            available_groups = [group for group in gpu_groups if group not in used_groups]
            if not available_groups:
                pending.insert(0, (idx, trial))
                break
            gpu_group = available_groups[0]
            trial_name, trial_root, stdout_path, cmd, row, nproc = build_trial(idx, trial, gpu_group)
            trial_start = time.time()

            run_dir = _find_run_dir(trial_root)
            if run_dir is not None:
                metrics = _parse_metrics(os.path.join(run_dir, "logs.txt"))
                score_key, score = _score_key(metrics)
                if _has_valid_score(score):
                    row.update(metrics)
                    row["score_key"] = score_key
                    row["score"] = score
                    row["run_dir"] = run_dir
                    row["exit_code"] = 0
                    rows.append(row)
                    completed += 1
                    print(f"[grid-ddp] skip existing {completed}/{total_trials}: {trial_name}", flush=True)
                    continue
                print(f"[grid-ddp] rerun incomplete existing trial {idx}/{total_trials}: {trial_name}", flush=True)

            env = os.environ.copy()
            if gpu_group:
                env["CUDA_VISIBLE_DEVICES"] = gpu_group

            if args.dry_run:
                row["score_key"] = "dry_run"
                row["score"] = ""
                rows.append(row)
                completed += 1
                prefix = f"CUDA_VISIBLE_DEVICES={gpu_group} " if gpu_group else ""
                print(prefix + " ".join(cmd), flush=True)
                continue

            stdout_file = open(stdout_path, "w", encoding="utf-8", errors="ignore")
            process = subprocess.Popen(
                cmd,
                cwd=repo_root,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            active.append(
                {
                    "trial_name": trial_name,
                    "trial_root": trial_root,
                    "stdout_path": stdout_path,
                    "row": row,
                    "process": process,
                    "stdout_file": stdout_file,
                    "trial_start": trial_start,
                    "gpu": gpu_group,
                }
            )
            print(
                f"[grid-ddp] start {idx}/{total_trials}: {trial_name} gpu={gpu_group} nproc={nproc}",
                flush=True,
            )

        if args.dry_run:
            continue

        time.sleep(5)
        still_active: List[Dict[str, object]] = []
        for item in active:
            process = item["process"]
            code = process.poll()
            if code is None:
                still_active.append(item)
                continue
            item["stdout_file"].close()
            completed += 1
            finish_trial(
                str(item["trial_name"]),
                str(item["trial_root"]),
                str(item["stdout_path"]),
                item["row"],
                int(code),
                float(item["trial_start"]),
                completed,
            )
        active = still_active

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            -float(x["score"]) if str(x.get("score", "")) not in ["", "None"] else float("-inf"),
            int(-x["exit_code"]),
        ),
    )
    _write_summary_csv(os.path.join(out_root, "summary.csv"), rows_sorted)
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "out_root": out_root,
                "total_trials": len(trials),
                "total_time": _format_duration(time.time() - grid_start),
                "top5": rows_sorted[:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ddp-train":
        _ddp_train_main(sys.argv[2:])
    else:
        main()
