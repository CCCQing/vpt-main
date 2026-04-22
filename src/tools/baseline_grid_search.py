#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
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


def _trial_name(idx: int, aw: float, lr: float, scale: float, wd: float) -> str:
    return "exp{idx:03d}_aw{aw}_lr{lr}_s{scale}_wd{wd}".format(
        idx=idx,
        aw=_format_float_tag(aw),
        lr=_format_float_tag(lr),
        scale=_format_float_tag(scale),
        wd=_format_float_tag(wd),
    )


def _run_cmd(cmd: List[str], cwd: str, stdout_path: str) -> int:
    with open(stdout_path, "w", encoding="utf-8", errors="ignore") as f:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    return p.returncode


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


def _write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    keys = [
        "trial_name",
        "align_weight",
        "base_lr",
        "fixed_logit_scale",
        "weight_decay",
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
    ap = argparse.ArgumentParser("baseline_grid_search")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config-file", default="configs/prompt/cub.yaml")
    ap.add_argument("--out-root", default="output/grid_rsim_v2_cosine_cm_stage1")
    ap.add_argument("--align-grid", default="1e-5,5e-5,1e-4,5e-4")
    ap.add_argument("--lr-grid", default="1e-4,3e-4,7e-4")
    ap.add_argument("--scale-grid", default="1.0,5.0")
    ap.add_argument("--wd-grid", default="0")
    ap.add_argument("--total-epoch", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_root = os.path.abspath(os.path.join(repo_root, args.out_root))
    os.makedirs(out_root, exist_ok=True)

    align_grid = _parse_float_list(args.align_grid)
    lr_grid = _parse_float_list(args.lr_grid)
    scale_grid = _parse_float_list(args.scale_grid)
    wd_grid = _parse_float_list(args.wd_grid)
    total_trials = len(align_grid) * len(lr_grid) * len(scale_grid) * len(wd_grid)

    search_space = {
        "config_file": args.config_file,
        "out_root": out_root,
        "total_trials": total_trials,
        "align_grid": align_grid,
        "lr_grid": lr_grid,
        "scale_grid": scale_grid,
        "wd_grid": wd_grid,
        "total_epoch": args.total_epoch,
        "extra_opts": args.opts,
    }
    with open(os.path.join(out_root, "search_space.json"), "w", encoding="utf-8") as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)

    base_opts = [
        "RUN_N_TIMES", "1",
        "MODEL.CLASSIFIER", "r_similarity_v2",
        "MODEL.PROMPT.ENABLE", "False",
        "MODEL.SEMANTIC_BRANCH.ENABLE", "False",
        "SOLVER.LOSS", "r_similarity_v2",
        "MODEL.R_SIMILARITY_V2.SCORE_MODE", "cosine",
        "MODEL.R_SIMILARITY_V2.LEARNABLE_SCALE", "False",
        "SOLVER.RSIM_V2.ALIGN_MODE", "cm",
    ]
    if args.total_epoch is not None:
        base_opts.extend(["SOLVER.TOTAL_EPOCH", str(args.total_epoch)])
    if args.opts:
        base_opts.extend(args.opts)

    rows: List[Dict[str, object]] = []
    idx = 1
    for align_weight in align_grid:
        for lr in lr_grid:
            for fixed_scale in scale_grid:
                for wd in wd_grid:
                    trial_name = _trial_name(idx, align_weight, lr, fixed_scale, wd)
                    trial_root = os.path.join(out_root, trial_name)
                    os.makedirs(trial_root, exist_ok=True)
                    stdout_path = os.path.join(trial_root, "launcher_stdout.txt")

                    cmd = [
                        sys.executable,
                        "train.py",
                        "--config-file",
                        args.config_file,
                        "OUTPUT_DIR",
                        trial_root,
                        "SOLVER.RSIM_V2.ALIGN_WEIGHT",
                        str(align_weight),
                        "SOLVER.BASE_LR",
                        str(lr),
                        "MODEL.R_SIMILARITY_V2.FIXED_LOGIT_SCALE",
                        str(fixed_scale),
                        "SOLVER.WEIGHT_DECAY",
                        str(wd),
                    ] + base_opts

                    row: Dict[str, object] = {
                        "trial_name": trial_name,
                        "align_weight": align_weight,
                        "base_lr": lr,
                        "fixed_logit_scale": fixed_scale,
                        "weight_decay": wd,
                        "exit_code": -1,
                        "run_dir": "",
                    }

                    run_dir = _find_run_dir(trial_root)
                    if run_dir is not None:
                        log_path = os.path.join(run_dir, "logs.txt")
                        metrics = _parse_metrics(log_path)
                        score_key, score = _score_key(metrics)
                        row.update(metrics)
                        row["score_key"] = score_key
                        row["score"] = score
                        row["run_dir"] = run_dir
                        row["exit_code"] = 0
                        rows.append(row)
                        idx += 1
                        continue

                    if args.dry_run:
                        row["score_key"] = "dry_run"
                        row["score"] = ""
                        rows.append(row)
                        print(" ".join(cmd))
                        idx += 1
                        continue

                    code = _run_cmd(cmd, cwd=repo_root, stdout_path=stdout_path)
                    row["exit_code"] = code

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

                    _write_summary_csv(os.path.join(out_root, "summary.csv"), rows)
                    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
                        json.dump(rows, f, ensure_ascii=False, indent=2)

                    idx += 1

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
    print(json.dumps({"out_root": out_root, "total_trials": total_trials, "top5": topk}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
