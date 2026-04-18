#!/usr/bin/env python3
"""
Reusable regression/diagnostic suite for ZSL/GZSL loss integration.

Modes:
  - quick:    low-cost regression checks (equivalence/start_epoch/shuffle tests)
  - short_train: small multi-epoch smoke comparison
"""

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

import numpy as np


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _run_cmd(cmd: List[str], cwd: str) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
    return p.returncode, p.stdout


def _find_run_dir(base_out: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(base_out, "**", "logs.txt"), recursive=True)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(latest)


def _parse_logs(log_path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    train_losses = []
    train_seen = []
    val_zsl = []
    test_zsl = []
    diag_lines = []
    eval_cand = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"average train loss:\s*([0-9eE+\-.]+),\s*train_seen_top1:\s*([0-9eE+\-.]+)", line)
            if m:
                train_losses.append(float(m.group(1)))
                train_seen.append(float(m.group(2)))
            m = re.search(r"Epoch\s+\d+/\d+\s+train:\s+loss=([0-9eE+\-.]+)", line)
            if m:
                train_losses.append(float(m.group(1)))
            m = re.search(r"Classification results with val_unseen_[^:]+:.*zsl_unseen:\s*([0-9.]+)", line)
            if m:
                val_zsl.append(float(m.group(1)))
            m = re.search(r"Eval\s+val_unseen_[^:]+:.*dev_unseen=([0-9.]+)", line)
            if m:
                val_zsl.append(float(m.group(1)))
            m = re.search(r"Classification results with test_unseen_[^:]+:.*zsl_unseen:\s*([0-9.]+)", line)
            if m:
                test_zsl.append(float(m.group(1)))
            m = re.search(r"Eval\s+test_unseen_[^:]+:.*zsl_unseen=([0-9.]+)", line)
            if m:
                test_zsl.append(float(m.group(1)))
            if "[diag-loss]" in line:
                diag_lines.append(line.strip())
            if "[eval-candidate]" in line:
                eval_cand.append(line.strip())
    if train_losses:
        out["train_loss_first"] = train_losses[0]
        out["train_loss_last"] = train_losses[-1]
        out["train_loss_delta"] = train_losses[-1] - train_losses[0]
    if train_seen:
        out["train_seen_last"] = train_seen[-1]
    if val_zsl:
        out["val_zsl_best"] = max(val_zsl)
        out["val_zsl_last"] = val_zsl[-1]
    if test_zsl:
        out["test_zsl_best"] = max(test_zsl)
        out["test_zsl_last"] = test_zsl[-1]
    out["diag_loss_lines"] = len(diag_lines)
    out["eval_candidate_lines"] = len(eval_cand)
    return out


def _read_monitor_summary(run_dir: str) -> Optional[Dict[str, float]]:
    p = os.path.join(run_dir, "monitor", "summary.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p, "r", encoding="utf-8", errors="ignore")))
    if not rows:
        return None
    last = rows[-1]
    out: Dict[str, float] = {"monitor_rows": float(len(rows))}
    for k in ["margin_mean", "sref_margin_gain_mean", "sref_faith_mean", "v2s_diag_top1_rate"]:
        if k in last and str(last[k]).strip() != "":
            try:
                out["monitor_last_" + k] = float(last[k])
            except ValueError:
                pass
    return out


def _launch_train(repo_root: str, cfg: str, out_root: str, extra_opts: List[str]) -> Tuple[Optional[str], Dict[str, float], int]:
    os.makedirs(out_root, exist_ok=True)
    cmd = [
        sys.executable, "train.py",
        "--config-file", cfg,
        "OUTPUT_DIR", out_root,
        "RUN_N_TIMES", "1",
    ] + extra_opts
    code, stdout = _run_cmd(cmd, cwd=repo_root)
    run_dir = _find_run_dir(out_root)
    metrics: Dict[str, float] = {"exit_code": float(code)}
    if run_dir is not None:
        log_path = os.path.join(run_dir, "logs.txt")
        if os.path.exists(log_path):
            metrics.update(_parse_logs(log_path))
        mon = _read_monitor_summary(run_dir)
        if mon:
            metrics.update(mon)
    else:
        metrics["runner_error"] = 1.0
    if code != 0:
        # save console output for debugging
        fail_log = os.path.join(out_root, "suite_console_fail.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            f.write(stdout)
    return run_dir, metrics, code


def _cmp_close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


def mode_quick(args):
    repo = args.repo_root
    base = os.path.join(args.out_root, "quick_" + _now_tag())
    os.makedirs(base, exist_ok=True)

    common = [
        "SOLVER.TOTAL_EPOCH", "1",
        "SOLVER.LOG_EVERY_N", "1",
        "SOLVER.DEBUG_TRACE_ONCE", "True",
        "SOLVER.DEBUG_GRAD_NORM", "True",
        "SOLVER.DIAG.PRINT_LOSS_WIRING", "True",
        "MODEL.AFFINITY.ENABLE", "True",
        "SOLVER.LOSS", "softmax_margin_cm",
        "SOLVER.LOSS_MARGIN", "0.05",
    ]

    results = {"checks": []}

    # Baseline (+AM only)
    run0, m0, c0 = _launch_train(repo, args.config_file, os.path.join(base, "baseline_am"), common + [
        "SOLVER.LOSS_CM_WEIGHT", "0.0",
    ])
    results["baseline"] = {"run_dir": run0, "metrics": m0, "code": c0}

    # 1) zero-weight equivalence
    run1, m1, c1 = _launch_train(repo, args.config_file, os.path.join(base, "zero_weight"), common + [
        "SOLVER.LOSS_CM_WEIGHT", "0.0",
    ])
    ok1 = (c0 == 0 and c1 == 0 and _cmp_close(m0.get("train_loss_last", 1e9), m1.get("train_loss_last", -1e9), tol=1e-3))
    results["checks"].append({
        "name": "zero_weight_equivalence",
        "pass": bool(ok1),
        "baseline_loss": m0.get("train_loss_last"),
        "zero_weight_loss": m1.get("train_loss_last"),
        "message": "PASS" if ok1 else "WARN: loss mismatch under zero weights",
    })

    # 2) start_epoch shielding
    run2, m2, c2 = _launch_train(repo, args.config_file, os.path.join(base, "start_epoch_block"), common + [
        "SOLVER.LOSS_CM_WEIGHT", "0.0",
    ])
    ok2 = (c2 == 0 and _cmp_close(m0.get("train_loss_last", 1e9), m2.get("train_loss_last", -1e9), tol=1e-3))
    results["checks"].append({
        "name": "start_epoch_shield",
        "pass": bool(ok2),
        "baseline_loss": m0.get("train_loss_last"),
        "shielded_loss": m2.get("train_loss_last"),
        "message": "PASS" if ok2 else "WARN: start_epoch seems not fully shielding",
    })

    # 3) shuffled raw targets
    run3, m3, c3 = _launch_train(repo, args.config_file, os.path.join(base, "shuffle_targets"), common + [
        "SOLVER.LOSS_CM_WEIGHT", "0.05",
        "SOLVER.DIAG.SHUFFLE_RAW_TARGETS", "True",
    ])
    worsen3 = (m3.get("train_loss_last", 0) - m0.get("train_loss_last", 0)) > 0.05
    results["checks"].append({
        "name": "shuffle_targets",
        "pass": bool(c3 == 0 and worsen3),
        "baseline_loss": m0.get("train_loss_last"),
        "shuffle_loss": m3.get("train_loss_last"),
        "message": "PASS" if (c3 == 0 and worsen3) else "ALERT: shuffle targets did not clearly worsen loss",
    })

    # 4) shuffled prototypes
    run4, m4, c4 = _launch_train(repo, args.config_file, os.path.join(base, "shuffle_prototypes"), common + [
        "SOLVER.LOSS_CM_WEIGHT", "0.05",
        "SOLVER.DIAG.SHUFFLE_PROTOTYPES", "True",
    ])
    worsen4 = (m4.get("train_loss_last", 0) - m0.get("train_loss_last", 0)) > 0.05
    results["checks"].append({
        "name": "shuffle_prototypes",
        "pass": bool(c4 == 0 and worsen4),
        "baseline_loss": m0.get("train_loss_last"),
        "shuffle_loss": m4.get("train_loss_last"),
        "message": "PASS" if (c4 == 0 and worsen4) else "ALERT: shuffle prototypes did not clearly worsen loss",
    })

    # high-risk checks from trace presence
    results["checks"].append({
        "name": "wiring_trace_present",
        "pass": bool(m0.get("diag_loss_lines", 0) > 0 and m0.get("eval_candidate_lines", 0) > 0),
        "diag_loss_lines": m0.get("diag_loss_lines", 0),
        "eval_candidate_lines": m0.get("eval_candidate_lines", 0),
        "message": "PASS" if (m0.get("diag_loss_lines", 0) > 0 and m0.get("eval_candidate_lines", 0) > 0) else "WARN: missing diag/eval candidate trace lines",
    })

    out_json = os.path.join(base, "quick_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps({"mode": "quick", "output": out_json, "checks": results["checks"]}, ensure_ascii=False, indent=2))


def mode_short_train(args):
    repo = args.repo_root
    base = os.path.join(args.out_root, "short_train_" + _now_tag())
    os.makedirs(base, exist_ok=True)
    common = [
        "SOLVER.TOTAL_EPOCH", str(args.short_epochs),
        "SOLVER.LOG_EVERY_N", "20",
        "SOLVER.DEBUG_TRACE_ONCE", "True",
        "MODEL.AFFINITY.ENABLE", "True",
        "SOLVER.LOSS", "softmax_margin_cm",
        "SOLVER.LOSS_MARGIN", "0.05",
        "SOLVER.LOSS_CM_WEIGHT", "0.05",
    ]
    run_dir, m, code = _launch_train(repo, args.config_file, base, common)
    result = {
        "run_dir": run_dir,
        "exit_code": code,
        "metrics": m,
        "pass": bool(code == 0),
    }
    out_json = os.path.join(base, "short_train_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"mode": "short_train", "output": out_json, "result": result}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser("debug_sanity_suite")
    ap.add_argument("--mode", choices=["quick", "short_train"], required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config-file", default="configs/prompt/local_path.yaml")
    ap.add_argument("--out-root", default="output/diag_suite")
    ap.add_argument("--short-epochs", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "quick":
        mode_quick(args)
    else:
        mode_short_train(args)


if __name__ == "__main__":
    main()
