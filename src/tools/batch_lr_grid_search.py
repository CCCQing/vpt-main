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
        if s:
            vals.append(float(s))
    if not vals:
        raise ValueError("Empty float grid value list.")
    return vals


def _parse_int_list(raw: str) -> List[int]:
    vals = []
    for item in raw.split(","):
        s = item.strip()
        if s:
            vals.append(int(s))
    if not vals:
        raise ValueError("Empty integer grid value list.")
    return vals


def _parse_str_list(raw: str) -> List[str]:
    vals = []
    for item in raw.split(","):
        s = item.strip()
        if s:
            vals.append(s)
    if not vals:
        raise ValueError("Empty string list.")
    return vals


def _parse_pair_list(raw: str) -> List[Dict[str, object]]:
    vals: List[Dict[str, object]] = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        parts = [part.strip() for part in s.split(":")]
        if len(parts) != 4:
            raise ValueError(
                "Pair values must use batch_size:lr:total_epoch:warmup_epoch format, "
                "e.g. 64:0.0012:40:3."
            )
        batch_raw, lr_raw, total_raw, warmup_raw = parts
        vals.append(
            {
                "batch_size": int(batch_raw),
                "base_lr": float(lr_raw),
                "total_epoch": int(total_raw),
                "warmup_epoch": int(warmup_raw),
            }
        )
    if not vals:
        raise ValueError("Empty batch/lr/epoch/warmup pair list.")
    return vals


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected true / false, got '{raw}'.")


def _format_float_tag(x: float) -> str:
    if x == 0:
        return "0"
    s = f"{x:.8g}".replace("+", "")
    return s.replace("-", "m").replace(".", "p")


def _trial_name(idx: int, batch_size: int, lr: float, total_epoch: int, warmup_epoch: int) -> str:
    return (
        f"exp{idx:03d}_best47_bs_{batch_size}_lr_{_format_float_tag(lr)}"
        f"_ep_{total_epoch}_warm_{warmup_epoch}"
    )


def _validate_extra_opts(opts: List[str]) -> None:
    if len(opts) % 2 != 0:
        raise ValueError("Extra config overrides must be KEY VALUE pairs.")
    owned = {"OUTPUT_DIR", "DATA.BATCH_SIZE", "SOLVER.BASE_LR", "SOLVER.TOTAL_EPOCH", "SOLVER.WARMUP_EPOCH"}
    bad = [item for item in opts if item.strip().upper() in owned]
    if bad:
        raise ValueError(
            "Do not pass OUTPUT_DIR, DATA.BATCH_SIZE, SOLVER.BASE_LR, SOLVER.TOTAL_EPOCH, "
            "or SOLVER.WARMUP_EPOCH through extra opts; use the dedicated grid arguments."
        )


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
        "batch_size",
        "base_lr",
        "total_epoch",
        "warmup_epoch",
        "source_best_trial",
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


def _best47_base_opts(vis_save_raw: bool, vis_save_images: bool) -> List[str]:
    return [
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
        "SOLVER.LOSS_PROMPT_KL_WEIGHT", "0.0",
        "MODEL.SEMANTIC_GRAPH.ENABLE", "True",
        "MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE", "fuse",
        "MODEL.SEMANTIC_GRAPH.RHO", "0.0",
        "MODEL.SEMANTIC_GRAPH.TOPK", "16",
        "MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA", "0.1",
        "MODEL.SEMANTIC_GRAPH.LOSS_TYPE", "fgw",
        "MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT", "0.01",
        "MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE", "mu",
        "MODEL.SEMANTIC_GRAPH.OT_EPS", "0.05",
        "MODEL.SEMANTIC_GRAPH.OT_ITERS", "20",
        "MODEL.SEMANTIC_GRAPH.OT_ALPHA", "0.5",
        "MODEL.SEMANTIC_GRAPH.OT_DELTA", "1e-8",
        "MODEL.SEMANTIC_GRAPH.OT_PRIOR_ETA", "1.0",
        "MODEL.SEMANTIC_GRAPH.OT_DETACH_PLAN", "True",
        "MODEL.SEMANTIC_GRAPH.TAU_ACC", "1.0",
        "MODEL.SEMANTIC_GRAPH.TAU_SEM", "1.0",
        "MODEL.SEMANTIC_GRAPH.TAU_PROMPT", "1.0",
        "MODEL.SEMANTIC_GRAPH.PROMPT_SCALE_LEARNABLE", "True",
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


def main() -> None:
    ap = argparse.ArgumentParser("batch_lr_grid_search_best47")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--config-file", default="configs/prompt/cub.yaml")
    ap.add_argument("--out-root", default="output/grid_prompt_distribution_47_batch_lr")
    ap.add_argument(
        "--pairs",
        default="",
        help="Comma-separated paired search points in batch_size:lr:total_epoch:warmup_epoch format.",
    )
    ap.add_argument("--batch-sizes", default="64", help="Fallback Cartesian batch sizes when --pairs is empty.")
    ap.add_argument("--lrs", default="0.0012,0.00125,0.00115", help="Fallback Cartesian learning rates when --pairs is empty.")
    ap.add_argument("--total-epochs", default="40,35,45", help="Fallback Cartesian total epochs when --pairs is empty.")
    ap.add_argument("--warmup-epochs", default="3,5,6", help="Fallback Cartesian warmup epochs when --pairs is empty.")
    ap.add_argument("--gpus", default="", help="Comma-separated GPU ids for parallel server runs, e.g. 0,1.")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--vis-save-raw", default="false", choices=["true", "false"])
    ap.add_argument("--vis-save-images", default="false", choices=["true", "false"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_root = os.path.abspath(os.path.join(repo_root, args.out_root))
    os.makedirs(out_root, exist_ok=True)

    if args.pairs.strip():
        trials = _parse_pair_list(args.pairs)
        batch_sizes = list(dict.fromkeys(int(item["batch_size"]) for item in trials))
        lrs = list(dict.fromkeys(float(item["base_lr"]) for item in trials))
        total_epochs = list(dict.fromkeys(int(item["total_epoch"]) for item in trials))
        warmup_epochs = list(dict.fromkeys(int(item["warmup_epoch"]) for item in trials))
    else:
        batch_sizes = _parse_int_list(args.batch_sizes)
        lrs = _parse_float_list(args.lrs)
        total_epochs = _parse_int_list(args.total_epochs)
        warmup_epochs = _parse_int_list(args.warmup_epochs)
        trials = []
        for batch_size in batch_sizes:
            for lr in lrs:
                for total_epoch in total_epochs:
                    for warmup_epoch in warmup_epochs:
                        trials.append(
                            {
                                "batch_size": batch_size,
                                "base_lr": lr,
                                "total_epoch": total_epoch,
                                "warmup_epoch": warmup_epoch,
                            }
                        )
    gpu_ids = _parse_str_list(args.gpus) if args.gpus.strip() else []
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    if gpu_ids and args.max_workers > len(gpu_ids):
        raise ValueError("--max-workers must not exceed the number of --gpus.")
    _validate_extra_opts(args.opts)

    base_opts = _best47_base_opts(
        vis_save_raw=_parse_bool(args.vis_save_raw),
        vis_save_images=_parse_bool(args.vis_save_images),
    ) + list(args.opts)

    search_space = {
        "config_file": args.config_file,
        "python_bin": args.python_bin,
        "out_root": out_root,
        "source_grid": "output/grid_prompt_distribution_47",
        "source_best_trial": "exp047_pdgraph_graph_True_src_vit_cls_prepass_h_64_kl_0_eval_fixed_eps_slot_True_stat_mu_rho_0_loss_fgw",
        "source_best_score": {"score_key": "gzsl_h_best", "score": 48.29},
        "total_trials": len(trials),
        "sweep": ["DATA.BATCH_SIZE", "SOLVER.BASE_LR", "SOLVER.TOTAL_EPOCH", "SOLVER.WARMUP_EPOCH"],
        "pairs": trials,
        "batch_sizes": batch_sizes,
        "lrs": lrs,
        "total_epochs": total_epochs,
        "warmup_epochs": warmup_epochs,
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
            "SOLVER.LOSS_PROMPT_KL_WEIGHT": 0.0,
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
            "SOLVER.LOSS_SEM_MED_WEIGHT": 0.0,
            "SOLVER.LOSS_SPV_WEIGHT": 0.0,
            "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT": 0.0,
            "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT": 0.0,
        },
        "server_parallel": {"gpus": gpu_ids, "max_workers": args.max_workers},
        "extra_opts": args.opts,
    }
    with open(os.path.join(out_root, "search_space.json"), "w", encoding="utf-8") as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)

    rows: List[Dict[str, object]] = []
    grid_start = time.time()
    total_trials = len(trials)

    def _build_trial(idx: int, trial: Dict[str, object], gpu_id: str = ""):
        trial_name = _trial_name(
            idx,
            int(trial["batch_size"]),
            float(trial["base_lr"]),
            int(trial["total_epoch"]),
            int(trial["warmup_epoch"]),
        )
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
        ] + base_opts + [
            "DATA.BATCH_SIZE",
            str(trial["batch_size"]),
            "SOLVER.BASE_LR",
            str(trial["base_lr"]),
            "SOLVER.TOTAL_EPOCH",
            str(trial["total_epoch"]),
            "SOLVER.WARMUP_EPOCH",
            str(trial["warmup_epoch"]),
        ]
        row: Dict[str, object] = {
            "trial_name": trial_name,
            "batch_size": trial["batch_size"],
            "base_lr": trial["base_lr"],
            "total_epoch": trial["total_epoch"],
            "warmup_epoch": trial["warmup_epoch"],
            "source_best_trial": search_space["source_best_trial"],
            "gpu": gpu_id,
            "exit_code": -1,
            "run_dir": "",
        }
        return trial_name, trial_root, stdout_path, cmd, row

    def _finish_trial(
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
                    print(f"[grid] skip existing {idx}/{total_trials}: {trial_name}", flush=True)
                    continue
                print(f"[grid] rerun incomplete existing trial {idx}/{total_trials}: {trial_name}", flush=True)

            env = os.environ.copy()
            if gpu_id:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

            if args.dry_run:
                row["score_key"] = "dry_run"
                row["score"] = ""
                rows.append(row)
                prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id else ""
                print(prefix + " ".join(cmd))
                continue

            code = _run_cmd(cmd, cwd=repo_root, stdout_path=stdout_path, env=env)
            _finish_trial(trial_name, trial_root, stdout_path, row, code, trial_start, idx)
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
                print(f"[grid] start {idx}/{total_trials}: {trial_name} gpu={gpu_id or 'default'}", flush=True)

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
    main()
