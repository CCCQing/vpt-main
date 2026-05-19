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


def _write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    keys = [
        "trial_name",
        "group",
        "prompt_lambda",
        "semantic_lambda",
        "route_ts_prompt_weight",
        "route_ts_semantic_weight",
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
    ap = argparse.ArgumentParser("lambda_route_ts_dynamic_learned_grid_search")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config-file", default="configs/prompt/cub.yaml")
    ap.add_argument("--out-root", default="output/grid_lambda_route_ts_dynamic_learned")
    ap.add_argument("--prompt-backend", default="dynamic", choices=["dynamic", "vpt_deep"])
    ap.add_argument("--prompt-init-source", default="learned", choices=["learned", "distributor_mean"])
    ap.add_argument("--distributor-enable", default="false", choices=["true", "false"])
    ap.add_argument("--affinity-evolution-enable", default="true", choices=["true", "false"])
    ap.add_argument("--vis-save-raw", default="true", choices=["true", "false"])
    ap.add_argument("--vis-save-images", default="false", choices=["true", "false"])
    ap.add_argument("--prompt-lambda-grid", default="0,0.1,0.2")
    ap.add_argument("--semantic-lambda-grid", default="0,0.1,0.2")
    ap.add_argument("--prompt-weight-grid", default="0,0.00001,0.001")
    ap.add_argument("--semantic-weight-grid", default="0,0.0001,0.001")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_root = os.path.abspath(os.path.join(repo_root, args.out_root))
    os.makedirs(out_root, exist_ok=True)

    prompt_lambda_grid = _parse_float_list(args.prompt_lambda_grid)
    semantic_lambda_grid = _parse_float_list(args.semantic_lambda_grid)
    prompt_weight_grid = _parse_float_list(args.prompt_weight_grid)
    semantic_weight_grid = _parse_float_list(args.semantic_weight_grid)
    distributor_enable = _parse_bool(args.distributor_enable)
    affinity_evolution_enable = _parse_bool(args.affinity_evolution_enable)
    vis_save_raw = _parse_bool(args.vis_save_raw)
    vis_save_images = _parse_bool(args.vis_save_images)
    # Fixed learned P0 + affinity evolution: no distributor and no old prompt_update_layers.
    base_opts = [
        "MODEL.PROMPT.BACKEND", args.prompt_backend,
        "MODEL.PROMPT.INIT_SOURCE", args.prompt_init_source,
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE", str(distributor_enable),
        "MODEL.AFFINITY_EVOLUTION.ENABLE", str(affinity_evolution_enable),
        "SOLVER.VIS.SAVE_RAW", str(vis_save_raw),
        "SOLVER.VIS.SAVE_IMAGES", str(vis_save_images),
    ]
    if args.opts:
        _validate_extra_opts(args.opts)
        base_opts.extend(args.opts)

    trials: List[Dict[str, object]] = []
    for prompt_lambda in prompt_lambda_grid:
        for semantic_lambda in semantic_lambda_grid:
            for prompt_weight in prompt_weight_grid:
                for semantic_weight in semantic_weight_grid:
                    params = {
                        "PROMPT_LAMBDA": prompt_lambda,
                        "SEMANTIC_LAMBDA": semantic_lambda,
                        "LOSS_ROUTE_TS_PROMPT_WEIGHT": prompt_weight,
                        "LOSS_ROUTE_TS_SEMANTIC_WEIGHT": semantic_weight,
                    }
                    trials.append(
                        {
                            "group": "lambda_route_ts_dynamic_learned",
                            "tag": _trial_tag("lambda_route_ts", params),
                            "prompt_lambda": prompt_lambda,
                            "semantic_lambda": semantic_lambda,
                            "route_ts_prompt_weight": prompt_weight,
                            "route_ts_semantic_weight": semantic_weight,
                            "opts": [
                                "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA", str(prompt_lambda),
                                "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA", str(semantic_lambda),
                                "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT", str(prompt_weight),
                                "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT", str(semantic_weight),
                            ],
                        }
                    )

    search_space = {
        "config_file": args.config_file,
        "out_root": out_root,
        "total_trials": len(trials),
        "sweep": [
            "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA",
            "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA",
            "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT",
            "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT",
        ],
        "prompt_lambda_grid": prompt_lambda_grid,
        "semantic_lambda_grid": semantic_lambda_grid,
        "lambda_policy": "PROMPT_LAMBDA and SEMANTIC_LAMBDA are swept independently.",
        "prompt_weight_grid": prompt_weight_grid,
        "semantic_weight_grid": semantic_weight_grid,
        "fixed_prompt_setting": {
            "MODEL.PROMPT.BACKEND": args.prompt_backend,
            "MODEL.PROMPT.INIT_SOURCE": args.prompt_init_source,
            "MODEL.PROMPT.DISTRIBUTOR.ENABLE": distributor_enable,
            "MODEL.AFFINITY_EVOLUTION.ENABLE": affinity_evolution_enable,
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
    for idx, trial in enumerate(trials, start=1):
        trial_name = _trial_name(idx, str(trial["tag"]))
        trial_root = os.path.join(out_root, trial_name)
        os.makedirs(trial_root, exist_ok=True)
        stdout_path = os.path.join(trial_root, "launcher_stdout.txt")
        trial_start = time.time()
        print(f"[grid] start {idx}/{total_trials}: {trial_name}", flush=True)

        cmd = [
            sys.executable,
            "train.py",
            "--config-file",
            args.config_file,
            "OUTPUT_DIR",
            trial_root,
        ] + base_opts + list(trial["opts"])

        row: Dict[str, object] = {
            "trial_name": trial_name,
            "group": trial["group"],
            "prompt_lambda": trial["prompt_lambda"],
            "semantic_lambda": trial["semantic_lambda"],
            "route_ts_prompt_weight": trial["route_ts_prompt_weight"],
            "route_ts_semantic_weight": trial["route_ts_semantic_weight"],
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

        if args.dry_run:
            row["score_key"] = "dry_run"
            row["score"] = ""
            rows.append(row)
            print(" ".join(cmd))
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
        elapsed = time.time() - trial_start
        total_elapsed = time.time() - grid_start
        avg_elapsed = total_elapsed / float(idx)
        eta = avg_elapsed * float(total_trials - idx)
        print(
            f"[grid] done {idx}/{total_trials}: {trial_name} exit_code={code} "
            f"trial_time={_format_duration(elapsed)} total_time={_format_duration(total_elapsed)} eta={_format_duration(eta)}",
            flush=True,
        )

        _write_summary_csv(os.path.join(out_root, "summary.csv"), rows)
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

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
