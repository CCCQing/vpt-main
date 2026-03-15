#!/usr/bin/env python3
"""
Reusable regression/diagnostic suite for ZSL/GZSL loss integration.

Modes:
  - quick:    low-cost regression checks (equivalence/start_epoch/shuffle tests)
  - overfit:  one-batch overfit checks across loss variants
  - short_train: small multi-epoch smoke comparison
  - analyze_confusion: hardest-negative/confusion report from saved logits
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


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
            m = re.search(r"Classification results with val_[^:]+:.*zsl_unseen:\s*([0-9.]+)", line)
            if m:
                val_zsl.append(float(m.group(1)))
            m = re.search(r"Classification results with test_[^:]+:.*zsl_unseen:\s*([0-9.]+)", line)
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
        "SOLVER.OVERFIT_ONE_BATCH_STEPS", "3",
        "SOLVER.LOG_EVERY_N", "1",
        "SOLVER.DEBUG_TRACE_ONCE", "True",
        "SOLVER.DEBUG_GRAD_NORM", "True",
        "SOLVER.DIAG.PRINT_LOSS_WIRING", "True",
        "MODEL.PROMPT.DEBUG_FLOW", "True",
        "MODEL.AFFINITY.ENABLE", "True",
        "SOLVER.LOSS", "softmax_margin_cm",
        "SOLVER.LOSS_MARGIN", "0.05",
    ]

    results = {"checks": []}

    # Baseline (+AM only)
    run0, m0, c0 = _launch_train(repo, args.config_file, os.path.join(base, "baseline_am"), common + [
        "SOLVER.LOSS_CM_WEIGHT", "0.0",
        "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.0",
    ])
    results["baseline"] = {"run_dir": run0, "metrics": m0, "code": c0}

    # 1) zero-weight equivalence
    run1, m1, c1 = _launch_train(repo, args.config_file, os.path.join(base, "zero_weight"), common + [
        "SOLVER.LOSS_CM_WEIGHT", "0.0",
        "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.0",
        "SOLVER.LOSS_SEM_ROUTE_START_EPOCH", "0",
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
        "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.01",
        "SOLVER.LOSS_SEM_ROUTE_START_EPOCH", "999",
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


def mode_overfit(args):
    repo = args.repo_root
    base = os.path.join(args.out_root, "overfit_" + _now_tag())
    os.makedirs(base, exist_ok=True)
    common = [
        "SOLVER.TOTAL_EPOCH", "1",
        "SOLVER.OVERFIT_ONE_BATCH_STEPS", str(args.overfit_steps),
        "SOLVER.LOG_EVERY_N", "10",
        "SOLVER.DEBUG_TRACE_ONCE", "True",
        "MODEL.PROMPT.DEBUG_FLOW", "True",
        "MODEL.AFFINITY.ENABLE", "True",
        "SOLVER.LOSS_MARGIN", "0.05",
    ]
    variants = [
        ("baseline", ["SOLVER.LOSS", "softmax", "MODEL.AFFINITY.ENABLE", "False"]),
        ("am", ["SOLVER.LOSS", "softmax_margin_cm", "SOLVER.LOSS_CM_WEIGHT", "0.0", "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.0"]),
        ("am_cm", ["SOLVER.LOSS", "softmax_margin_cm", "SOLVER.LOSS_CM_WEIGHT", "0.05", "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.0"]),
        ("am_cm_semdist_like", [
            "SOLVER.LOSS", "softmax_margin_cm",
            "SOLVER.LOSS_CM_WEIGHT", "0.05",
            "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.01",
            "SOLVER.LOSS_SEM_ROUTE_MASK_TYPE", "all_ones",
            "SOLVER.LOSS_SEM_ROUTE_GAMMA_IND", "1.0",
            "SOLVER.LOSS_SEM_ROUTE_GAMMA_DIR", "1.0",
            "SOLVER.LOSS_SEM_ROUTE_START_EPOCH", "0",
        ]),
        ("am_cm_soft_ea", [
            "SOLVER.LOSS", "softmax_margin_cm",
            "SOLVER.LOSS_CM_WEIGHT", "0.05",
            "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.01",
            "SOLVER.LOSS_SEM_ROUTE_MASK_TYPE", "hard_topk",
            "SOLVER.LOSS_SEM_ROUTE_TOPK", "8",
            "SOLVER.LOSS_SEM_ROUTE_GAMMA_IND", "0.0",
            "SOLVER.LOSS_SEM_ROUTE_GAMMA_DIR", "1.0",
            "SOLVER.LOSS_SEM_ROUTE_START_EPOCH", "0",
        ]),
    ]
    out = []
    for name, extra in variants:
        run_dir, m, code = _launch_train(repo, args.config_file, os.path.join(base, name), common + extra)
        fit_ok = (m.get("train_loss_last", 1e9) < m.get("train_loss_first", 1e9))
        out.append({
            "name": name,
            "run_dir": run_dir,
            "exit_code": code,
            "train_loss_first": m.get("train_loss_first"),
            "train_loss_last": m.get("train_loss_last"),
            "fit_ok": bool(code == 0 and fit_ok),
        })
    out_json = os.path.join(base, "overfit_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"variants": out}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"mode": "overfit", "output": out_json, "variants": out}, ensure_ascii=False, indent=2))


def mode_short_train(args):
    repo = args.repo_root
    base = os.path.join(args.out_root, "short_train_" + _now_tag())
    os.makedirs(base, exist_ok=True)
    common = [
        "SOLVER.TOTAL_EPOCH", str(args.short_epochs),
        "SOLVER.OVERFIT_ONE_BATCH_STEPS", "0",
        "SOLVER.LOG_EVERY_N", "20",
        "SOLVER.DEBUG_TRACE_ONCE", "True",
        "MODEL.PROMPT.DEBUG_FLOW", "True",
        "MODEL.AFFINITY.ENABLE", "True",
        "SOLVER.LOSS", "softmax_margin_cm",
        "SOLVER.LOSS_MARGIN", "0.05",
        "SOLVER.LOSS_CM_WEIGHT", "0.05",
        "SOLVER.LOSS_SEM_ROUTE_WEIGHT", "0.01",
        "SOLVER.LOSS_SEM_ROUTE_MASK_TYPE", "hard_topk",
        "SOLVER.LOSS_SEM_ROUTE_TOPK", "8",
        "SOLVER.LOSS_SEM_ROUTE_GAMMA_IND", "0.0",
        "SOLVER.LOSS_SEM_ROUTE_GAMMA_DIR", "1.0",
        "SOLVER.LOSS_SEM_ROUTE_START_EPOCH", "2",
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


def _find_logits_file(run_dir: str) -> Optional[str]:
    cands = glob.glob(os.path.join(run_dir, "*_logits.pth"))
    if not cands:
        cands = glob.glob(os.path.join(run_dir, "**", "*_logits.pth"), recursive=True)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _find_checkpoint_file(run_dir: str) -> Optional[str]:
    cands = glob.glob(os.path.join(run_dir, "*.pth"))
    cands = [p for p in cands if not p.endswith("_logits.pth")]
    if not cands:
        cands = glob.glob(os.path.join(run_dir, "**", "*.pth"), recursive=True)
        cands = [p for p in cands if not p.endswith("_logits.pth")]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _safe_entropy_from_counts(counts: np.ndarray) -> float:
    if counts.size == 0:
        return float("nan")
    tot = float(counts.sum())
    if tot <= 0:
        return float("nan")
    p = counts.astype(np.float64) / tot
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _to_jsonable(x: Any):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _extract_optional_array(d: Dict[str, Any], names: List[str], length: int) -> Optional[np.ndarray]:
    for n in names:
        v = d.get(n, None)
        if v is None:
            continue
        arr = np.asarray(v)
        if arr.ndim == 1 and arr.shape[0] == length:
            return arr.astype(np.float64)
    return None


def _extract_optional_matrix(d: Dict[str, Any], names: List[str]) -> Optional[np.ndarray]:
    for n in names:
        v = d.get(n, None)
        if v is None:
            continue
        arr = np.asarray(v)
        if arr.ndim == 2:
            return arr.astype(np.float64)
    return None


def _name_for_class(class_id: int, class_names: Optional[List[str]], id2name: Optional[Dict[int, str]]) -> str:
    if class_names is not None and 0 <= int(class_id) < len(class_names):
        return str(class_names[int(class_id)])
    if id2name is not None and int(class_id) in id2name:
        return str(id2name[int(class_id)])
    return ""


def _compute_concentration(
    targets: np.ndarray,
    hn: np.ndarray,
    max_pairs_preview: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    n = int(len(targets))
    pair_count: Dict[Tuple[int, int], int] = {}
    true_count: Dict[int, int] = {}
    per_true_hn: Dict[int, Dict[int, int]] = {}
    for y, h in zip(targets.tolist(), hn.tolist()):
        y = int(y)
        h = int(h)
        pair_count[(y, h)] = pair_count.get((y, h), 0) + 1
        true_count[y] = true_count.get(y, 0) + 1
        if y not in per_true_hn:
            per_true_hn[y] = {}
        per_true_hn[y][h] = per_true_hn[y].get(h, 0) + 1

    sorted_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)
    top_pairs_csv = []
    freqs = []
    for (y, h), c in sorted_pairs:
        f_global = (c / n) if n > 0 else float("nan")
        f_within = (c / true_count.get(y, 1))
        top_pairs_csv.append({
            "y_true": y,
            "hn_class": h,
            "count": c,
            "freq_global": f_global,
            "freq_within_true_class": f_within,
        })
        freqs.append(f_global if n > 0 else 0.0)

    class_rows = []
    top1_ratio_list = []
    for y, ctot in sorted(true_count.items()):
        dist = per_true_hn.get(y, {})
        sorted_h = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top1_h, top1_c = sorted_h[0] if sorted_h else (-1, 0)
        top2_h, top2_c = sorted_h[1] if len(sorted_h) > 1 else (-1, 0)
        top1_r = (top1_c / ctot) if ctot > 0 else float("nan")
        top2_r = (top2_c / ctot) if ctot > 0 else float("nan")
        ent = _safe_entropy_from_counts(np.asarray([v for _, v in sorted_h], dtype=np.float64))
        class_rows.append({
            "class_id": y,
            "num_samples": ctot,
            "top_hn_class": top1_h,
            "top_hn_ratio": top1_r,
            "second_hn_class": top2_h,
            "second_hn_ratio": top2_r,
            "hn_entropy": ent,
        })
        if not np.isnan(top1_r):
            top1_ratio_list.append(top1_r)

    freqs_np = np.asarray(freqs, dtype=np.float64) if len(freqs) > 0 else np.asarray([], dtype=np.float64)
    if freqs_np.size > 0:
        top10_pair_mass = float(freqs_np[:10].sum())
        top20_pair_mass = float(freqs_np[:20].sum())
        top50_pair_mass = float(freqs_np[:50].sum())
        herfindahl = float((freqs_np ** 2).sum())
    else:
        top10_pair_mass = top20_pair_mass = top50_pair_mass = herfindahl = float("nan")

    summary = {
        "num_pairs_total": int(len(sorted_pairs)),
        "top10_pair_mass": top10_pair_mass,
        "top20_pair_mass": top20_pair_mass,
        "top50_pair_mass": top50_pair_mass,
        "mean_top1_hn_ratio_per_class": float(np.mean(top1_ratio_list)) if len(top1_ratio_list) > 0 else float("nan"),
        "herfindahl_pair": herfindahl,
        "top_confused_pairs_preview": top_pairs_csv[:max_pairs_preview],
    }
    return top_pairs_csv, class_rows, summary


def _compute_margin_distribution(margin: np.ndarray) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if margin.size == 0:
        dist = {
            "mean": float("nan"), "std": float("nan"), "median": float("nan"),
            "q10": float("nan"), "q25": float("nan"), "q50": float("nan"), "q75": float("nan"), "q90": float("nan"),
            "p_margin_lt_0": float("nan"), "p_margin_lt_neg1": float("nan"), "p_margin_lt_neg2": float("nan"),
            "p_margin_ge_0": float("nan"), "p_margin_ge_1": float("nan"),
        }
        return dist, []

    dist = {
        "mean": float(np.mean(margin)),
        "std": float(np.std(margin)),
        "median": float(np.median(margin)),
        "q10": _safe_quantile(margin, 0.10),
        "q25": _safe_quantile(margin, 0.25),
        "q50": _safe_quantile(margin, 0.50),
        "q75": _safe_quantile(margin, 0.75),
        "q90": _safe_quantile(margin, 0.90),
        "p_margin_lt_0": float(np.mean(margin < 0)),
        "p_margin_lt_neg1": float(np.mean(margin < -1)),
        "p_margin_lt_neg2": float(np.mean(margin < -2)),
        "p_margin_ge_0": float(np.mean(margin >= 0)),
        "p_margin_ge_1": float(np.mean(margin >= 1)),
    }

    buckets = [
        ("catastrophic", margin < -2),
        ("strong_negative", (margin >= -2) & (margin < -1)),
        ("borderline_negative", (margin >= -1) & (margin < 0)),
        ("borderline_positive", (margin >= 0) & (margin < 1)),
        ("safe_positive", margin >= 1),
    ]
    rows = []
    n = float(max(len(margin), 1))
    for name, mk in buckets:
        vals = margin[mk]
        rows.append({
            "bucket_name": name,
            "count": int(vals.size),
            "ratio": float(vals.size / n),
            "mean_margin": float(np.mean(vals)) if vals.size > 0 else float("nan"),
            "median_margin": float(np.median(vals)) if vals.size > 0 else float("nan"),
        })
    return dist, rows


def mode_analyze_confusion(args):
    run_dir = args.run_dir
    if run_dir is None:
        raise ValueError("--run-dir is required for analyze_confusion mode")
    logits_file = args.logits_file or _find_logits_file(run_dir)
    out_dir = os.path.join(run_dir, "diagnose")
    os.makedirs(out_dir, exist_ok=True)
    if logits_file is None:
        ckpt = _find_checkpoint_file(run_dir)
        # graceful fallback: do not crash; emit unavailable report
        report = {
            "available": False,
            "unavailable_reason": "No *_logits.pth found.",
            "checkpoint_found": bool(ckpt is not None),
            "checkpoint_path": ckpt,
            "hint": "Run with MODEL.SAVE_CKPT=True or provide --logits-file.",
        }
        json_path = os.path.join(out_dir, "confusion_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps({
            "mode": "analyze_confusion",
            "csv": None,
            "json": json_path,
            "extra": {
                "top_confused_pairs_csv": None,
                "class_confusion_summary_csv": None,
                "margin_distribution_json": None,
                "margin_bucket_report_csv": None,
                "refinement_gain_samples_csv": None,
                "refinement_gain_summary_json": None,
            },
            "summary": {
                "top1_acc": None,
                "margin_mean": None,
                "margin_median": None,
                "p_margin_lt_0": None,
                "top20_pair_mass": None,
                "num_samples": 0,
            }
        }, ensure_ascii=False, indent=2))
        return

    data = torch.load(logits_file, map_location="cpu")
    logits = np.asarray(data["joint_logits"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.int64)
    n = int(len(targets))
    if logits.ndim != 2 or n == 0:
        raise ValueError("Invalid logits/targets payload in {}".format(logits_file))

    pred = logits.argmax(axis=1)
    pos = logits[np.arange(len(targets)), targets]
    tmp = logits.copy()
    tmp[np.arange(len(targets)), targets] = -1e18
    hn = tmp.argmax(axis=1)
    hn_score = logits[np.arange(len(targets)), hn]
    margin = pos - hn_score

    # class name metadata (optional)
    class_names = None
    id2name = None
    if "class_names" in data:
        try:
            class_names = [str(x) for x in list(data["class_names"])]
        except Exception:
            class_names = None
    if "class_id_to_name" in data and isinstance(data["class_id_to_name"], dict):
        id2name = {int(k): str(v) for k, v in data["class_id_to_name"].items()}

    # top-k negatives
    topk = max(1, int(args.topk_neg))
    neg_order = np.argsort(logits, axis=1)[:, ::-1]
    topk_neg_cls = []
    topk_neg_scores = []
    for i in range(n):
        negs = [int(c) for c in neg_order[i].tolist() if int(c) != int(targets[i])]
        negs = negs[:topk]
        topk_neg_cls.append(negs)
        topk_neg_scores.append([float(logits[i, c]) for c in negs])

    # hardest-negative table (enhanced, backward-compatible fields kept)
    rows = []
    for i in range(n):
        y = int(targets[i])
        h = int(hn[i])
        rows.append({
            "sample_idx": int(i),
            "idx": int(i),  # backward compatibility
            "y_true": y,
            "y_pred": int(pred[i]),
            "top1_pred": int(pred[i]),  # backward compatibility
            "hn_class": h,
            "hardest_negative_class": h,  # backward compatibility
            "pos_score": float(pos[i]),
            "hn_score": float(hn_score[i]),
            "sim_pos": float(pos[i]),  # backward compatibility
            "sim_hn": float(hn_score[i]),  # backward compatibility
            "margin": float(margin[i]),
            "topk_neg_classes": "|".join(str(x) for x in topk_neg_cls[i]),
            "topk_neg_scores": "|".join("{:.8f}".format(x) for x in topk_neg_scores[i]),
            "correct": int(int(pred[i]) == y),
            "image_path": str(data.get("image_paths", [""] * n)[i]) if ("image_paths" in data and len(data.get("image_paths")) == n) else "",
            "class_name_true": _name_for_class(y, class_names, id2name),
            "class_name_hn": _name_for_class(h, class_names, id2name),
        })
    csv_path = os.path.join(out_dir, "hardest_negative_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # A) confusion concentration
    top_pairs_csv, class_rows, conc_summary = _compute_concentration(
        targets=targets, hn=hn, max_pairs_preview=max(1, int(args.max_pairs_preview))
    )
    top_pairs_csv_path = os.path.join(out_dir, "top_confused_pairs.csv")
    with open(top_pairs_csv_path, "w", newline="", encoding="utf-8") as f:
        if len(top_pairs_csv) > 0:
            w = csv.DictWriter(f, fieldnames=list(top_pairs_csv[0].keys()))
            w.writeheader()
            w.writerows(top_pairs_csv)
        else:
            f.write("y_true,hn_class,count,freq_global,freq_within_true_class\n")

    class_csv_rows = []
    for r in class_rows:
        rr = dict(r)
        rr["class_name"] = _name_for_class(int(r["class_id"]), class_names, id2name)
        class_csv_rows.append(rr)
    class_summary_csv_path = os.path.join(out_dir, "class_confusion_summary.csv")
    with open(class_summary_csv_path, "w", newline="", encoding="utf-8") as f:
        if len(class_csv_rows) > 0:
            w = csv.DictWriter(f, fieldnames=list(class_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(class_csv_rows)
        else:
            f.write("class_id,class_name,num_samples,top_hn_class,top_hn_ratio,second_hn_class,second_hn_ratio,hn_entropy\n")

    # B) refinement gain analysis (graceful fallback)
    gain_samples_csv_path = os.path.join(out_dir, "refinement_gain_samples.csv")
    gain_summary_json_path = os.path.join(out_dir, "refinement_gain_summary.json")
    sim_true_raw = _extract_optional_array(data, ["sim_true_raw"], n)
    sim_true_ref = _extract_optional_array(data, ["sim_true_ref"], n)
    sim_hn_raw = _extract_optional_array(data, ["sim_hn_raw"], n)
    sim_hn_ref = _extract_optional_array(data, ["sim_hn_ref"], n)
    sim_raw_all = _extract_optional_matrix(data, ["sim_raw_all"])
    sim_ref_all = _extract_optional_matrix(data, ["sim_ref_all"])

    gain_available = all(x is not None for x in [sim_true_raw, sim_true_ref, sim_hn_raw, sim_hn_ref])
    gain_summary: Dict[str, Any]
    gain_rows: List[Dict[str, Any]] = []
    if gain_available:
        gain_true = sim_true_ref - sim_true_raw
        gain_hn = sim_hn_ref - sim_hn_raw
        delta_gain = gain_true - gain_hn
        margin_raw = sim_true_raw - sim_hn_raw
        margin_ref = sim_true_ref - sim_hn_ref
        margin_delta = margin_ref - margin_raw
        for i in range(n):
            gain_rows.append({
                "sample_idx": int(i),
                "y_true": int(targets[i]),
                "hn_class": int(hn[i]),
                "sim_true_raw": float(sim_true_raw[i]),
                "sim_true_ref": float(sim_true_ref[i]),
                "sim_hn_raw": float(sim_hn_raw[i]),
                "sim_hn_ref": float(sim_hn_ref[i]),
                "gain_true": float(gain_true[i]),
                "gain_hn": float(gain_hn[i]),
                "delta_gain": float(delta_gain[i]),
                "margin_raw": float(margin_raw[i]),
                "margin_ref": float(margin_ref[i]),
                "margin_delta": float(margin_delta[i]),
            })
        with open(gain_samples_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(gain_rows[0].keys()))
            w.writeheader()
            w.writerows(gain_rows)
        gain_summary = {
            "available": True,
            "num_samples_with_gain": int(n),
            "mean_gain_true": float(np.mean(gain_true)),
            "mean_gain_hn": float(np.mean(gain_hn)),
            "mean_delta_gain": float(np.mean(delta_gain)),
            "mean_margin_delta": float(np.mean(margin_delta)),
            "frac_gain_true_gt_0": float(np.mean(gain_true > 0)),
            "frac_gain_hn_gt_0": float(np.mean(gain_hn > 0)),
            "frac_delta_gain_gt_0": float(np.mean(delta_gain > 0)),
        }
    else:
        gain_summary = {
            "available": False,
            "unavailable_reason": "sim_true_raw/sim_true_ref/sim_hn_raw/sim_hn_ref are not present in logits cache.",
        }
        # create placeholder csv for compatibility
        with open(gain_samples_csv_path, "w", encoding="utf-8") as f:
            f.write("sample_idx,y_true,hn_class,sim_true_raw,sim_true_ref,sim_hn_raw,sim_hn_ref,gain_true,gain_hn,delta_gain,margin_raw,margin_ref,margin_delta\n")
    with open(gain_summary_json_path, "w", encoding="utf-8") as f:
        json.dump(gain_summary, f, ensure_ascii=False, indent=2)

    # B2) coarse-stage / candidate-delta diagnostics (pre-union GT, candidate_source=raw_topk)
    coarse_candidate_summary: Dict[str, Any]
    if (
        sim_raw_all is not None
        and sim_ref_all is not None
        and sim_raw_all.ndim == 2
        and sim_ref_all.ndim == 2
        and sim_raw_all.shape[0] == n
        and sim_ref_all.shape[0] == n
    ):
        num_classes = int(sim_raw_all.shape[1])
        k_cfg = int(data.get("semantic_score_topk", 0) or 0)
        k = int(min(max(1, k_cfg if k_cfg > 0 else 5), num_classes))
        topk_idx = np.argsort(sim_raw_all, axis=1)[:, ::-1][:, :k]  # pre-union topk from raw
        hit = (topk_idx == targets.reshape(-1, 1)).any(axis=1)
        coarse_recall_at_k = float(np.mean(hit))
        def _recall_at(kk: int) -> float:
            kk = int(min(max(1, kk), num_classes))
            idx = np.argsort(sim_raw_all, axis=1)[:, ::-1][:, :kk]
            return float(np.mean((idx == targets.reshape(-1, 1)).any(axis=1)))
        coarse_recall_at_1 = _recall_at(1)
        coarse_recall_at_3 = _recall_at(3)
        coarse_recall_at_5 = _recall_at(5)
        coarse_recall_at_10 = _recall_at(10)

        pos_raw_all = sim_raw_all[np.arange(n), targets]
        neg_raw_all = sim_raw_all.copy()
        neg_raw_all[np.arange(n), targets] = -1e18
        hn_raw_idx = neg_raw_all.argmax(axis=1)
        hn_raw_score = sim_raw_all[np.arange(n), hn_raw_idx]
        coarse_gap = pos_raw_all - hn_raw_score

        # candidate delta on raw topk (pre-union)
        delta = sim_ref_all - sim_raw_all
        mask = np.zeros_like(sim_raw_all, dtype=bool)
        rows_ar = np.arange(n).reshape(-1, 1)
        mask[rows_ar, topk_idx] = True
        candidate_delta_mean = float(np.mean(delta[mask])) if np.any(mask) else float("nan")
        gt_in = mask[np.arange(n), targets]
        hn_in = mask[np.arange(n), hn_raw_idx]
        candidate_delta_gt_mean = float(np.mean(delta[np.arange(n), targets][gt_in])) if np.any(gt_in) else float("nan")
        candidate_delta_hn_mean = float(np.mean(delta[np.arange(n), hn_raw_idx][hn_in])) if np.any(hn_in) else float("nan")
        both = gt_in & hn_in
        candidate_delta_gap = float(np.mean(delta[np.arange(n), targets][both] - delta[np.arange(n), hn_raw_idx][both])) if np.any(both) else float("nan")
        coarse_candidate_summary = {
            "available": True,
            "candidate_source": str(data.get("candidate_source", "raw_topk")),
            "semantic_score_mode": str(data.get("semantic_score_mode", "unknown")),
            "semantic_score_topk": int(k),
            "semantic_score_alpha": float(data.get("semantic_score_alpha", float("nan"))),
            "semantic_score_train_include_gt": bool(data.get("semantic_score_train_include_gt", False)),
            "coarse_recall_at_k": coarse_recall_at_k,
            "coarse_recall_at_1": coarse_recall_at_1,
            "coarse_recall_at_3": coarse_recall_at_3,
            "coarse_recall_at_5": coarse_recall_at_5,
            "coarse_recall_at_10": coarse_recall_at_10,
            "coarse_topk_contains_gt_rate": coarse_recall_at_k,
            "coarse_gap_mean": float(np.mean(coarse_gap)),
            "coarse_gap_median": float(np.median(coarse_gap)),
            "candidate_delta_mean": candidate_delta_mean,
            "candidate_delta_gt_mean": candidate_delta_gt_mean,
            "candidate_delta_hn_mean": candidate_delta_hn_mean,
            "candidate_delta_gap": candidate_delta_gap,
            "candidate_delta_gap_available_ratio": float(np.mean(both)),
        }
    else:
        coarse_candidate_summary = {
            "available": False,
            "unavailable_reason": "sim_raw_all/sim_ref_all missing in logits cache.",
        }

    # C) margin distribution
    margin_dist, margin_bucket_rows = _compute_margin_distribution(margin)
    if gain_available and len(gain_rows) == n:
        gain_true = np.asarray([r["gain_true"] for r in gain_rows], dtype=np.float64)
        gain_hn = np.asarray([r["gain_hn"] for r in gain_rows], dtype=np.float64)
        delta_gain = np.asarray([r["delta_gain"] for r in gain_rows], dtype=np.float64)
        margin_delta = np.asarray([r["margin_delta"] for r in gain_rows], dtype=np.float64)
        # append gain stats per bucket
        for r in margin_bucket_rows:
            bn = r["bucket_name"]
            if bn == "catastrophic":
                mk = margin < -2
            elif bn == "strong_negative":
                mk = (margin >= -2) & (margin < -1)
            elif bn == "borderline_negative":
                mk = (margin >= -1) & (margin < 0)
            elif bn == "borderline_positive":
                mk = (margin >= 0) & (margin < 1)
            else:
                mk = margin >= 1
            if np.any(mk):
                r["mean_gain_true"] = float(np.mean(gain_true[mk]))
                r["mean_gain_hn"] = float(np.mean(gain_hn[mk]))
                r["mean_delta_gain"] = float(np.mean(delta_gain[mk]))
                r["mean_margin_delta"] = float(np.mean(margin_delta[mk]))
            else:
                r["mean_gain_true"] = float("nan")
                r["mean_gain_hn"] = float("nan")
                r["mean_delta_gain"] = float("nan")
                r["mean_margin_delta"] = float("nan")

    margin_dist_json_path = os.path.join(out_dir, "margin_distribution.json")
    with open(margin_dist_json_path, "w", encoding="utf-8") as f:
        json.dump({k: _to_jsonable(v) for k, v in margin_dist.items()}, f, ensure_ascii=False, indent=2)

    margin_bucket_csv_path = os.path.join(out_dir, "margin_bucket_report.csv")
    with open(margin_bucket_csv_path, "w", newline="", encoding="utf-8") as f:
        if len(margin_bucket_rows) > 0:
            w = csv.DictWriter(f, fieldnames=list(margin_bucket_rows[0].keys()))
            w.writeheader()
            w.writerows(margin_bucket_rows)
        else:
            f.write("bucket_name,count,ratio,mean_margin,median_margin\n")

    # final report (backward compatible + richer fields)
    top_hn = {}
    for c in hn.tolist():
        c = int(c)
        top_hn[c] = top_hn.get(c, 0) + 1
    top_hn_list = [{"class": int(k), "count": int(v)} for k, v in sorted(top_hn.items(), key=lambda x: x[1], reverse=True)[:50]]
    report = {
        "available": True,
        "logits_file": logits_file,
        "num_samples": int(n),
        "top1_acc": float((pred == targets).mean()),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
        "margin_distribution": margin_dist,
        "concentration": conc_summary,
        "gain_summary": gain_summary,
        "coarse_candidate_summary": coarse_candidate_summary,
        "top_hard_negative_classes": top_hn_list,
        "top_confused_pairs": conc_summary.get("top_confused_pairs_preview", []),
        "notes": [
            "hardest-negative table upgraded with top-k negatives and optional names/paths.",
            "gain analysis gracefully degrades when raw/refined similarity caches are unavailable.",
        ],
    }
    report["summary"] = {
        "top1_acc": report["top1_acc"],
        "margin_mean": report["margin_mean"],
        "margin_median": margin_dist.get("median", float("nan")),
        "p_margin_lt_0": margin_dist.get("p_margin_lt_0", float("nan")),
        "p_margin_lt_neg1": margin_dist.get("p_margin_lt_neg1", float("nan")),
        "p_margin_lt_neg2": margin_dist.get("p_margin_lt_neg2", float("nan")),
        "top20_pair_mass": conc_summary.get("top20_pair_mass", float("nan")),
        "gain_analysis_available": bool(gain_summary.get("available", False)),
        "coarse_candidate_available": bool(coarse_candidate_summary.get("available", False)),
        "coarse_recall_at_k": coarse_candidate_summary.get("coarse_recall_at_k", float("nan")),
        "coarse_gap_mean": coarse_candidate_summary.get("coarse_gap_mean", float("nan")),
        "coarse_gap_median": coarse_candidate_summary.get("coarse_gap_median", float("nan")),
        "candidate_delta_gap": coarse_candidate_summary.get("candidate_delta_gap", float("nan")),
        "candidate_delta_gap_available_ratio": coarse_candidate_summary.get("candidate_delta_gap_available_ratio", float("nan")),
    }
    json_path = os.path.join(out_dir, "confusion_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "mode": "analyze_confusion",
        "csv": csv_path,
        "json": json_path,
        "extra": {
            "top_confused_pairs_csv": top_pairs_csv_path,
            "class_confusion_summary_csv": class_summary_csv_path,
            "margin_distribution_json": margin_dist_json_path,
            "margin_bucket_report_csv": margin_bucket_csv_path,
            "refinement_gain_samples_csv": gain_samples_csv_path if gain_summary.get("available", False) else None,
            "refinement_gain_summary_json": gain_summary_json_path,
        },
        "summary": {
            "top1_acc": report["top1_acc"],
            "margin_mean": report["margin_mean"],
            "margin_median": margin_dist.get("median", float("nan")),
            "p_margin_lt_0": margin_dist.get("p_margin_lt_0", float("nan")),
            "p_margin_lt_neg1": margin_dist.get("p_margin_lt_neg1", float("nan")),
            "p_margin_lt_neg2": margin_dist.get("p_margin_lt_neg2", float("nan")),
            "top20_pair_mass": conc_summary.get("top20_pair_mass", float("nan")),
            "num_samples": report["num_samples"],
            "gain_analysis_available": bool(gain_summary.get("available", False)),
            "gain_analysis_skipped_reason": None if gain_summary.get("available", False) else gain_summary.get("unavailable_reason"),
            "coarse_candidate_available": bool(coarse_candidate_summary.get("available", False)),
            "coarse_candidate_skipped_reason": None if coarse_candidate_summary.get("available", False) else coarse_candidate_summary.get("unavailable_reason"),
            "coarse_recall_at_k": coarse_candidate_summary.get("coarse_recall_at_k", float("nan")),
            "coarse_gap_mean": coarse_candidate_summary.get("coarse_gap_mean", float("nan")),
            "candidate_delta_gap": coarse_candidate_summary.get("candidate_delta_gap", float("nan")),
        }
    }, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser("debug_sanity_suite")
    ap.add_argument("--mode", choices=["quick", "overfit", "short_train", "analyze_confusion"], required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config-file", default="configs/prompt/cub.yaml")
    ap.add_argument("--out-root", default="output/diag_suite")
    ap.add_argument("--overfit-steps", type=int, default=200)
    ap.add_argument("--short-epochs", type=int, default=5)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--logits-file", default=None)
    ap.add_argument("--topk-neg", type=int, default=5)
    ap.add_argument("--max-pairs-preview", type=int, default=20)
    args = ap.parse_args()

    if args.mode == "quick":
        mode_quick(args)
    elif args.mode == "overfit":
        mode_overfit(args)
    elif args.mode == "short_train":
        mode_short_train(args)
    else:
        mode_analyze_confusion(args)


if __name__ == "__main__":
    main()
