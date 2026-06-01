#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple


def _parse_float_list(raw: str) -> List[float]:
    vals = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("Empty grid value list.")
    return vals


def _parse_int_list(raw: str) -> List[int]:
    vals = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("Empty integer grid value list.")
    return vals


def _format_float_tag(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e-2 and ax < 1e3:
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    mantissa, exp = f"{x:.0e}".split("e")
    exp_i = int(exp)
    return f"{mantissa}e{exp_i}"


def _parse_str_list(raw: str) -> List[str]:
    vals = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        vals.append(s)
    if not vals:
        raise ValueError("Empty string grid value list.")
    return vals


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected boolean string true / false, got '{raw}'.")


def _validate_choices(name: str, values: List[str], allowed: List[str]) -> None:
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported {name}: {bad}. Expected values from {allowed}.")


def _validate_no_output_dir_override(opts: List[str]) -> None:
    bad = [item for item in opts if item.strip().upper() == "OUTPUT_DIR"]
    if bad:
        raise ValueError("Do not pass OUTPUT_DIR through extra opts; the grid launcher owns per-trial OUTPUT_DIR.")


def _validate_extra_opts(opts: List[str]) -> None:
    _validate_no_output_dir_override(opts)
    if len(opts) % 2 != 0:
        if len(opts) == 1 and ("/" in opts[0] or "\\" in opts[0]):
            raise ValueError(
                "Output directory must be passed with --out-root, e.g. "
                f"--out-root {opts[0]}. Positional arguments are reserved for KEY VALUE config overrides."
            )
        raise ValueError("Extra config overrides must be KEY VALUE pairs.")


def _trial_name(idx: int, tag: str) -> str:
    return "exp{idx:03d}_{tag}".format(idx=idx, tag=tag)


def _trial_tag(prefix: str, params: Dict[str, object]) -> str:
    parts = [prefix]
    for name, value in params.items():
        if isinstance(value, float):
            value_str = _format_float_tag(value)
        else:
            value_str = str(value).replace(".", "p")
        parts.append(f"{name}_{value_str}")
    return "_".join(parts)


def _run_cmd(cmd: List[str], cwd: str, stdout_path: str, env: Optional[Dict[str, str]] = None) -> int:
    with open(stdout_path, "w", encoding="utf-8", errors="ignore") as f:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
        )
    return p.returncode


def _read_tail(path: str, max_lines: int = 60) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _find_run_dir(base_out: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(base_out, "**", "logs.txt"), recursive=True)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(latest)


def _parse_metrics(log_path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    train_losses: List[float] = []
    dev_unseen: List[float] = []
    zsl_unseen: List[float] = []
    gzsl_seen: List[float] = []
    gzsl_unseen: List[float] = []
    gzsl_h: List[float] = []
    best_epoch = None
    protocol_mode = None
    eval_mode = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"Epoch\s+\d+/\d+\s+train:\s+loss=([0-9eE+\-.]+)", line)
            if m:
                train_losses.append(float(m.group(1)))

            m = re.search(r"XLSA protocol mode=([a-z_]+)\s+eval_mode=([a-z_]+)", line)
            if m:
                protocol_mode = m.group(1)
                eval_mode = m.group(2)

            m = re.search(r"dev_unseen=([0-9.]+)", line)
            if m:
                dev_unseen.append(float(m.group(1)))

            m = re.search(r"\bzsl_unseen=([0-9.]+)", line)
            if m:
                zsl_unseen.append(float(m.group(1)))

            m = re.search(r"Eval\s+test_seen_[^:]+:.*gzsl_seen=([0-9.]+)", line)
            if m:
                gzsl_seen.append(float(m.group(1)))

            m = re.search(r"Eval\s+test_unseen_[^:]+:.*gzsl_unseen=([0-9.]+)", line)
            if m:
                gzsl_unseen.append(float(m.group(1)))

            m = re.search(r"\[gzsl-record\].*gzsl_h=([0-9.]+)", line)
            if m:
                gzsl_h.append(float(m.group(1)) * 100.0)

            m = re.search(r"Best epoch\s+(\d+)", line)
            if m:
                best_epoch = int(m.group(1))

    if protocol_mode is not None:
        out["protocol_mode"] = protocol_mode
    if eval_mode is not None:
        out["eval_mode"] = eval_mode
    if train_losses:
        out["train_loss_first"] = train_losses[0]
        out["train_loss_last"] = train_losses[-1]
    if dev_unseen:
        out["dev_unseen_best"] = max(dev_unseen)
        out["dev_unseen_last"] = dev_unseen[-1]
    if zsl_unseen:
        out["zsl_unseen_best"] = max(zsl_unseen)
        out["zsl_unseen_last"] = zsl_unseen[-1]
    if gzsl_seen:
        out["gzsl_seen_best"] = max(gzsl_seen)
        out["gzsl_seen_last"] = gzsl_seen[-1]
    if gzsl_unseen:
        out["gzsl_unseen_best"] = max(gzsl_unseen)
        out["gzsl_unseen_last"] = gzsl_unseen[-1]
    if gzsl_h:
        out["gzsl_h_best"] = max(gzsl_h)
        out["gzsl_h_last"] = gzsl_h[-1]
    if best_epoch is not None:
        out["best_epoch"] = float(best_epoch)
    return out


def _score_key(metrics: Dict[str, float]) -> Tuple[str, float]:
    for key in ["gzsl_h_best", "dev_unseen_best", "zsl_unseen_best", "gzsl_unseen_best"]:
        if key in metrics:
            return key, metrics[key]
    return "score", float("-inf")


def _has_valid_score(score: object) -> bool:
    try:
        return float(score) != float("-inf")
    except (TypeError, ValueError):
        return False


def _write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    keys = [
        "trial_name",
        "group",
        "prompt_backend",
        "prompt_init_source",
        "prompt_distributor_enable",
        "distributor_source",
        "stats_hidden_dim",
        "instance_tokens",
        "domain_tokens",
        "semantic_graph_enable",
        "graph_source",
        "graph_rho",
        "graph_topk",
        "graph_loss_type",
        "graph_loss_weight",
        "target_mix_alpha",
        "tau_acc",
        "tau_sem",
        "tau_prompt",
        "prompt_scale_learnable",
        "prompt_kl_weight",
        "gpu",
        "exit_code",
        "score_key",
        "score",
        "best_epoch",
        "dev_unseen_best",
        "dev_unseen_last",
        "zsl_unseen_best",
        "zsl_unseen_last",
        "gzsl_seen_best",
        "gzsl_seen_last",
        "gzsl_unseen_best",
        "gzsl_unseen_last",
        "gzsl_h_best",
        "gzsl_h_last",
        "train_loss_first",
        "train_loss_last",
        "protocol_mode",
        "eval_mode",
        "run_dir",
    ]
    extra_keys = []
    seen = set(keys)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                extra_keys.append(key)
    keys.extend(extra_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser("prompt_distribution_semantic_graph_grid_search")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--config-file", default="configs/prompt/cub.yaml")
    ap.add_argument("--out-root", default="output/grid_prompt_distribution_semantic_graph")
    ap.add_argument("--sources", default="token_mlp,vit_cls_prepass")
    ap.add_argument("--graph-sources", default="fuse")
    ap.add_argument("--topks", default="16")
    ap.add_argument("--loss-types", default="acc_hidden,rel_kl,rel_all,ot,fgw")
    ap.add_argument("--loss-weights", default="1e-2")
    ap.add_argument("--rhos", default="0,1,0.5")
    ap.add_argument("--stats-hidden-dims", default="4")
    ap.add_argument("--target-mix-alphas", default="0.1,0.5")
    ap.add_argument("--prompt-kl-weight", default="0.1")
    ap.add_argument("--tau-acc", default="1.0")
    ap.add_argument("--tau-sem", default="1.0")
    ap.add_argument("--tau-prompt", default="1.0")
    ap.add_argument("--prompt-scale-learnable", default="true", choices=["true", "false"])
    ap.add_argument("--gpus", default="", help="Comma-separated GPU ids for parallel server runs, e.g. 0,1.")
    ap.add_argument("--max-workers", type=int, default=1, help="Number of concurrent trials. Use 2 with --gpus 0,1 for dual-card runs.")
    ap.add_argument("--vis-save-raw", default="false", choices=["true", "false"])
    ap.add_argument("--vis-save-images", default="false", choices=["true", "false"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_root = os.path.abspath(os.path.join(repo_root, args.out_root))
    os.makedirs(out_root, exist_ok=True)

    sources = _parse_str_list(args.sources)
    graph_sources = _parse_str_list(args.graph_sources)
    topks = _parse_int_list(args.topks)
    loss_types = _parse_str_list(args.loss_types)
    loss_weights = _parse_float_list(args.loss_weights)
    rhos = _parse_float_list(args.rhos)
    stats_hidden_dims = _parse_int_list(args.stats_hidden_dims)
    target_mix_alphas = _parse_float_list(args.target_mix_alphas)
    prompt_kl_weight = float(args.prompt_kl_weight)
    tau_acc = float(args.tau_acc)
    tau_sem = float(args.tau_sem)
    tau_prompt = float(args.tau_prompt)
    prompt_scale_learnable = _parse_bool(args.prompt_scale_learnable)
    gpu_ids = _parse_str_list(args.gpus) if args.gpus.strip() else []
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    if gpu_ids and args.max_workers > len(gpu_ids):
        raise ValueError("--max-workers must not exceed the number of --gpus.")
    vis_save_raw = _parse_bool(args.vis_save_raw)
    vis_save_images = _parse_bool(args.vis_save_images)
    _validate_choices("MODEL.PROMPT.DISTRIBUTOR.SOURCE", sources, ["token_mlp", "vit_cls_prepass"])
    _validate_choices("MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE", graph_sources, ["acc", "acssc", "fuse"])
    _validate_choices("MODEL.SEMANTIC_GRAPH.LOSS_TYPE", loss_types, ["acc_hidden", "rel_kl", "rel_all", "ot", "fgw"])

    base_opts = [
        "MODEL.PROMPT.BACKEND", "dynamic",
        "MODEL.PROMPT.INIT_SOURCE", "distributor_mean",
        "MODEL.PROMPT.NUM_TOKENS", "32",
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE", "True",
        "MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS", "16",
        "MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS", "16",
        "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE", "mean",
        "MODEL.SEMANTIC_GRAPH.ENABLE", "True",
        "SOLVER.LOSS_PROMPT_KL_WEIGHT", str(prompt_kl_weight),
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
        "MODEL.AFFINITY_EVOLUTION.ENABLE", "True",
        "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA", "0.0",
        "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA", "0.0",
        "SOLVER.LOSS_ATTR_WEIGHT", "0.001",
        "SOLVER.ATTR.METRIC", "mse",
        "SOLVER.LOSS_SEM_MED_WEIGHT", "0.0",
        "SOLVER.LOSS_SPV_WEIGHT", "0.0",
        "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT", "0.0",
        "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT", "0.0",
        "SOLVER.VIS.SAVE_RAW", str(vis_save_raw),
        "SOLVER.VIS.SAVE_IMAGES", str(vis_save_images),
    ]
    if args.opts:
        _validate_extra_opts(args.opts)
        base_opts.extend(args.opts)

    graph_loss_weight = loss_weights[0]
    target_mix_alpha = 0.1
    graph_source = "fuse"
    topk = 16

    trials: List[Dict[str, object]] = []

    def _add_trial(
        *,
        group: str,
        source: str,
        stats_hidden_dim: int,
        prompt_kl: float,
        eval_sample_mode: str,
        use_slot_embed: bool,
        prompt_stat_source: str,
        graph_enable: bool,
        rho: float,
        loss_type: str,
        loss_weight: float,
    ) -> None:
        params = {
            "graph": graph_enable,
            "src": source,
            "h": stats_hidden_dim,
            "kl": prompt_kl,
            "eval": eval_sample_mode,
            "slot": use_slot_embed,
            "stat": prompt_stat_source,
            "rho": rho,
            "loss": loss_type,
        }
        opts = [
            "MODEL.PROMPT.DISTRIBUTOR.SOURCE", source,
            "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM", str(stats_hidden_dim),
            "SOLVER.LOSS_PROMPT_KL_WEIGHT", str(prompt_kl),
            "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE", eval_sample_mode,
            "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED", str(use_slot_embed),
            "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE", prompt_stat_source,
            "MODEL.SEMANTIC_GRAPH.ENABLE", str(graph_enable),
            "MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE", graph_source,
            "MODEL.SEMANTIC_GRAPH.RHO", str(rho),
            "MODEL.SEMANTIC_GRAPH.TOPK", str(topk),
            "MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA", str(target_mix_alpha),
            "MODEL.SEMANTIC_GRAPH.LOSS_TYPE", loss_type,
            "MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT", str(loss_weight),
        ]
        trials.append(
            {
                "group": group,
                "tag": _trial_tag("pdgraph", params),
                "prompt_backend": "dynamic",
                "prompt_init_source": "distributor_mean",
                "prompt_distributor_enable": True,
                "distributor_source": source,
                "stats_hidden_dim": stats_hidden_dim,
                "instance_tokens": 16,
                "domain_tokens": 16,
                "eval_sample_mode": eval_sample_mode,
                "use_slot_embed": use_slot_embed,
                "prompt_stat_source": prompt_stat_source,
                "semantic_graph_enable": graph_enable,
                "graph_source": graph_source,
                "graph_rho": rho,
                "graph_topk": topk,
                "graph_loss_type": loss_type,
                "graph_loss_weight": loss_weight,
                "target_mix_alpha": target_mix_alpha,
                "tau_acc": tau_acc,
                "tau_sem": tau_sem,
                "tau_prompt": tau_prompt,
                "prompt_scale_learnable": prompt_scale_learnable,
                "prompt_kl_weight": prompt_kl,
                "opts": opts,
            }
        )

    for stats_hidden_dim in [32, 64, 128]:
        for prompt_kl in [0.0, 0.001]:
            for prompt_stat_source in ["mu", "instance_mean"]:
                for use_slot_embed in [True, False]:
                    _add_trial(
                        group="graph_off_token_mlp",
                        source="token_mlp",
                        stats_hidden_dim=stats_hidden_dim,
                        prompt_kl=prompt_kl,
                        eval_sample_mode="fixed_eps",
                        use_slot_embed=use_slot_embed,
                        prompt_stat_source=prompt_stat_source,
                        graph_enable=False,
                        rho=0.0,
                        loss_type="none",
                        loss_weight=0.0,
                    )

    for stats_hidden_dim in [32, 64, 128]:
        _add_trial(
            group="graph_off_vit_cls_prepass",
            source="vit_cls_prepass",
            stats_hidden_dim=stats_hidden_dim,
            prompt_kl=0.0,
            eval_sample_mode="fixed_eps",
            use_slot_embed=True,
            prompt_stat_source="mu",
            graph_enable=False,
            rho=0.0,
            loss_type="none",
            loss_weight=0.0,
        )

    for rho in [1.0, 0.0]:
        for loss_type in ["acc_hidden", "rel_kl", "rel_all", "ot", "fgw"]:
            _add_trial(
                group="graph_on_token_mlp",
                source="token_mlp",
                stats_hidden_dim=64,
                prompt_kl=0.0,
                eval_sample_mode="fixed_eps",
                use_slot_embed=True,
                prompt_stat_source="mu",
                graph_enable=True,
                rho=rho,
                loss_type=loss_type,
                loss_weight=graph_loss_weight,
            )

    for rho in [1.0, 0.0]:
        for loss_type in ["acc_hidden", "rel_kl", "rel_all", "ot", "fgw"]:
            _add_trial(
                group="graph_on_vit_cls_prepass",
                source="vit_cls_prepass",
                stats_hidden_dim=64,
                prompt_kl=0.0,
                eval_sample_mode="fixed_eps",
                use_slot_embed=True,
                prompt_stat_source="mu",
                graph_enable=True,
                rho=rho,
                loss_type=loss_type,
                loss_weight=graph_loss_weight,
            )

    search_space = {
        "config_file": args.config_file,
        "python_bin": args.python_bin,
        "out_root": out_root,
        "total_trials": len(trials),
        "sweep": [
            "MODEL.PROMPT.DISTRIBUTOR.SOURCE",
            "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM",
            "SOLVER.LOSS_PROMPT_KL_WEIGHT",
            "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE",
            "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED",
            "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE",
            "MODEL.SEMANTIC_GRAPH.ENABLE",
            "MODEL.SEMANTIC_GRAPH.RHO",
            "MODEL.SEMANTIC_GRAPH.LOSS_TYPE",
        ],
        "groups": {
            "graph_off_token_mlp": {
                "count": 24,
                "MODEL.SEMANTIC_GRAPH.ENABLE": False,
                "MODEL.PROMPT.DISTRIBUTOR.SOURCE": ["token_mlp"],
                "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM": [32, 64, 128],
                "SOLVER.LOSS_PROMPT_KL_WEIGHT": [0.0, 0.001],
                "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": ["fixed_eps"],
                "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED": [True, False],
                "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE": ["mu", "instance_mean"],
            },
            "graph_off_vit_cls_prepass": {
                "count": 3,
                "MODEL.SEMANTIC_GRAPH.ENABLE": False,
                "MODEL.PROMPT.DISTRIBUTOR.SOURCE": ["vit_cls_prepass"],
                "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM": [32, 64, 128],
                "SOLVER.LOSS_PROMPT_KL_WEIGHT": [0.0],
                "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": ["fixed_eps"],
                "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED": [True],
                "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE": ["mu"],
            },
            "graph_on_token_mlp": {
                "count": 10,
                "MODEL.SEMANTIC_GRAPH.ENABLE": True,
                "MODEL.PROMPT.DISTRIBUTOR.SOURCE": ["token_mlp"],
                "MODEL.SEMANTIC_GRAPH.RHO": [1.0, 0.0],
                "MODEL.SEMANTIC_GRAPH.LOSS_TYPE": ["acc_hidden", "rel_kl", "rel_all", "ot", "fgw"],
                "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM": [64],
                "SOLVER.LOSS_PROMPT_KL_WEIGHT": [0.0],
                "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": ["fixed_eps"],
                "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED": [True],
                "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE": ["mu"],
            },
            "graph_on_vit_cls_prepass": {
                "count": 10,
                "MODEL.SEMANTIC_GRAPH.ENABLE": True,
                "MODEL.PROMPT.DISTRIBUTOR.SOURCE": ["vit_cls_prepass"],
                "MODEL.SEMANTIC_GRAPH.RHO": [1.0, 0.0],
                "MODEL.SEMANTIC_GRAPH.LOSS_TYPE": ["acc_hidden", "rel_kl", "rel_all", "ot", "fgw"],
                "MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM": [64],
                "SOLVER.LOSS_PROMPT_KL_WEIGHT": [0.0],
                "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": ["fixed_eps"],
                "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED": [True],
                "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE": ["mu"],
            },
        },
        "fixed_graph_setting": {
            "MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE": graph_source,
            "MODEL.SEMANTIC_GRAPH.TOPK": topk,
            "MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA": target_mix_alpha,
            "MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT": graph_loss_weight,
        },
        "temperature_setting": {
            "MODEL.SEMANTIC_GRAPH.TAU_ACC": tau_acc,
            "MODEL.SEMANTIC_GRAPH.TAU_SEM": tau_sem,
            "MODEL.SEMANTIC_GRAPH.TAU_PROMPT": tau_prompt,
            "MODEL.SEMANTIC_GRAPH.PROMPT_SCALE_LEARNABLE": prompt_scale_learnable,
        },
        "server_parallel": {
            "gpus": gpu_ids,
            "max_workers": args.max_workers,
        },
        "fixed_setting": {
            "MODEL.PROMPT.BACKEND": "dynamic",
            "MODEL.PROMPT.INIT_SOURCE": "distributor_mean",
            "MODEL.PROMPT.NUM_TOKENS": 32,
            "MODEL.PROMPT.DISTRIBUTOR.ENABLE": True,
            "MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS": 16,
            "MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS": 16,
            "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": "trial_specific",
            "MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED": "trial_specific",
            "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE": "trial_specific",
            "MODEL.SEMANTIC_GRAPH.ENABLE": "trial_specific",
            "SOLVER.LOSS_PROMPT_KL_WEIGHT": "trial_specific",
            "MODEL.SEMANTIC_GRAPH.OT_EPS": 0.05,
            "MODEL.SEMANTIC_GRAPH.OT_ITERS": 20,
            "MODEL.SEMANTIC_GRAPH.OT_ALPHA": 0.5,
            "MODEL.SEMANTIC_GRAPH.OT_DELTA": 1e-8,
            "MODEL.SEMANTIC_GRAPH.OT_PRIOR_ETA": 1.0,
            "MODEL.SEMANTIC_GRAPH.OT_DETACH_PLAN": True,
            "MODEL.SEMANTIC_GRAPH.TAU_ACC": tau_acc,
            "MODEL.SEMANTIC_GRAPH.TAU_SEM": tau_sem,
            "MODEL.SEMANTIC_GRAPH.TAU_PROMPT": tau_prompt,
            "MODEL.SEMANTIC_GRAPH.PROMPT_SCALE_LEARNABLE": prompt_scale_learnable,
            "MODEL.SEMANTIC_TOKENS.ENABLE": True,
            "MODEL.SEMANTIC_TOKENS.TRAIN_SOURCE": "class_mean",
            "MODEL.SEMANTIC_TOKENS.EVAL_SOURCE": "class_mean",
            "MODEL.SEMANTIC_TOKENS.TOKENIZER": "orthogonal",
            "MODEL.SEMANTIC_TOKENS.NUM_TOKENS": 8,
            "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE": "manual_cub8",
            "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE": "null_residual",
            "MODEL.AFFINITY_EVOLUTION.ENABLE": True,
            "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA": 0.0,
            "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA": 0.0,
            "SOLVER.LOSS_ATTR_WEIGHT": 0.001,
            "SOLVER.ATTR.METRIC": "mse",
            "SOLVER.LOSS_SEM_MED_WEIGHT": 0.0,
            "SOLVER.LOSS_SPV_WEIGHT": 0.0,
            "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT": 0.0,
            "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT": 0.0,
        },
        "visualization_setting": {
            "SOLVER.VIS.SAVE_RAW": vis_save_raw,
            "SOLVER.VIS.SAVE_IMAGES": vis_save_images,
        },
        "note": "All other settings come from the yaml config unless passed through extra opts.",
        "extra_opts": args.opts,
    }
    with open(os.path.join(out_root, "search_space.json"), "w", encoding="utf-8") as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)

    rows: List[Dict[str, object]] = []
    grid_start = time.time()
    total_trials = len(trials)

    def _build_trial(idx: int, trial: Dict[str, object], gpu_id: str = ""):
        trial_name = _trial_name(idx, str(trial["tag"]))
        trial_root = os.path.join(out_root, trial_name)
        os.makedirs(trial_root, exist_ok=True)
        stdout_path = os.path.join(trial_root, "launcher_stdout.txt")
        cmd = [
            args.python_bin,
            "train.py",
            "--config-file",
            args.config_file,
            "OUTPUT_DIR",
            trial_root,
        ] + base_opts + list(trial["opts"])
        row: Dict[str, object] = {
            "trial_name": trial_name,
            "group": trial["group"],
            "prompt_backend": trial["prompt_backend"],
            "prompt_init_source": trial["prompt_init_source"],
            "prompt_distributor_enable": trial["prompt_distributor_enable"],
            "distributor_source": trial["distributor_source"],
            "stats_hidden_dim": trial["stats_hidden_dim"],
            "instance_tokens": trial["instance_tokens"],
            "domain_tokens": trial["domain_tokens"],
            "eval_sample_mode": trial["eval_sample_mode"],
            "use_slot_embed": trial["use_slot_embed"],
            "prompt_stat_source": trial["prompt_stat_source"],
            "semantic_graph_enable": trial["semantic_graph_enable"],
            "graph_source": trial["graph_source"],
            "graph_rho": trial["graph_rho"],
            "graph_topk": trial["graph_topk"],
            "graph_loss_type": trial["graph_loss_type"],
            "graph_loss_weight": trial["graph_loss_weight"],
            "target_mix_alpha": trial["target_mix_alpha"],
            "tau_acc": trial["tau_acc"],
            "tau_sem": trial["tau_sem"],
            "tau_prompt": trial["tau_prompt"],
            "prompt_scale_learnable": trial["prompt_scale_learnable"],
            "prompt_kl_weight": trial["prompt_kl_weight"],
            "gpu": gpu_id,
            "exit_code": -1,
            "run_dir": "",
        }
        return trial_name, trial_root, stdout_path, cmd, row

    def _finish_trial(
        idx: int,
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
            print(f"[grid] failure stdout tail for {trial_name}:", flush=True)
            print(_read_tail(stdout_path), flush=True)

        run_dir = _find_run_dir(trial_root)
        if run_dir is not None:
            log_path = os.path.join(run_dir, "logs.txt")
            metrics = _parse_metrics(log_path)
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
            f"[grid] done {completed}/{total_trials}: {trial_name} exit_code={code} "
            f"trial_time={_format_duration(elapsed)} total_time={_format_duration(total_elapsed)} eta={_format_duration(eta)}",
            flush=True,
        )
        _write_summary_csv(os.path.join(out_root, "summary.csv"), rows)
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    if args.max_workers == 1:
        for idx, trial in enumerate(trials, start=1):
            gpu_id = gpu_ids[(idx - 1) % len(gpu_ids)] if gpu_ids else ""
            trial_name, trial_root, stdout_path, cmd, row = _build_trial(idx, trial, gpu_id)
            trial_start = time.time()
            print(f"[grid] start {idx}/{total_trials}: {trial_name} gpu={gpu_id or 'default'}", flush=True)

            env = os.environ.copy()
            if gpu_id:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

            run_dir = _find_run_dir(trial_root)
            if run_dir is not None:
                log_path = os.path.join(run_dir, "logs.txt")
                metrics = _parse_metrics(log_path)
                score_key, score = _score_key(metrics)
                if _has_valid_score(score):
                    row.update(metrics)
                    row["score_key"] = score_key
                    row["score"] = score
                    row["run_dir"] = run_dir
                    row["exit_code"] = 0
                    rows.append(row)
                    elapsed = time.time() - trial_start
                    total_elapsed = time.time() - grid_start
                    avg_elapsed = total_elapsed / float(idx)
                    eta = avg_elapsed * float(total_trials - idx)
                    print(
                        f"[grid] skip existing {idx}/{total_trials}: {trial_name} "
                        f"trial_time={_format_duration(elapsed)} total_time={_format_duration(total_elapsed)} eta={_format_duration(eta)}",
                        flush=True,
                    )
                    continue
                print(f"[grid] rerun incomplete existing trial {idx}/{total_trials}: {trial_name}", flush=True)

            if args.dry_run:
                row["score_key"] = "dry_run"
                row["score"] = ""
                rows.append(row)
                prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id else ""
                print(prefix + " ".join(cmd))
                elapsed = time.time() - trial_start
                total_elapsed = time.time() - grid_start
                avg_elapsed = total_elapsed / float(idx)
                eta = avg_elapsed * float(total_trials - idx)
                print(
                    f"[grid] dry-run done {idx}/{total_trials}: {trial_name} "
                    f"trial_time={_format_duration(elapsed)} total_time={_format_duration(total_elapsed)} eta={_format_duration(eta)}",
                    flush=True,
                )
                continue

            code = _run_cmd(cmd, cwd=repo_root, stdout_path=stdout_path, env=env)
            _finish_trial(idx, trial_name, trial_root, stdout_path, row, code, trial_start, idx)
    else:
        pending = list(enumerate(trials, start=1))
        active: List[Dict[str, object]] = []
        completed = 0
        while pending or active:
            while pending and len(active) < args.max_workers:
                idx, trial = pending.pop(0)
                gpu_id = ""
                if gpu_ids:
                    if args.dry_run:
                        gpu_id = gpu_ids[(idx - 1) % len(gpu_ids)]
                    else:
                        used_gpus = {str(item["gpu"]) for item in active}
                        free_gpus = [gpu for gpu in gpu_ids if gpu not in used_gpus]
                        if not free_gpus:
                            break
                        gpu_id = free_gpus[0]
                trial_name, trial_root, stdout_path, cmd, row = _build_trial(idx, trial, gpu_id)
                trial_start = time.time()

                run_dir = _find_run_dir(trial_root)
                if run_dir is not None:
                    log_path = os.path.join(run_dir, "logs.txt")
                    metrics = _parse_metrics(log_path)
                    score_key, score = _score_key(metrics)
                    if _has_valid_score(score):
                        row.update(metrics)
                        row["score_key"] = score_key
                        row["score"] = score
                        row["run_dir"] = run_dir
                        row["exit_code"] = 0
                        rows.append(row)
                        completed += 1
                        print(f"[grid] skip existing {completed}/{total_trials}: {trial_name}", flush=True)
                        continue
                    print(f"[grid] rerun incomplete existing trial {idx}/{total_trials}: {trial_name}", flush=True)

                if args.dry_run:
                    row["score_key"] = "dry_run"
                    row["score"] = ""
                    rows.append(row)
                    completed += 1
                    prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id else ""
                    print(prefix + " ".join(cmd))
                    continue

                env = os.environ.copy()
                if gpu_id:
                    env["CUDA_VISIBLE_DEVICES"] = gpu_id
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
                        "idx": idx,
                        "trial_name": trial_name,
                        "trial_root": trial_root,
                        "stdout_path": stdout_path,
                        "row": row,
                        "process": process,
                        "stdout_file": stdout_file,
                        "trial_start": trial_start,
                        "gpu": gpu_id,
                    }
                )
                print(
                    f"[grid] start {idx}/{total_trials}: {trial_name} gpu={gpu_id or 'default'}",
                    flush=True,
                )

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
                _finish_trial(
                    int(item["idx"]),
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

    topk = rows_sorted[:5]
    print(
        json.dumps(
            {
                "out_root": out_root,
                "total_trials": len(trials),
                "total_time": _format_duration(time.time() - grid_start),
                "top5": topk,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
