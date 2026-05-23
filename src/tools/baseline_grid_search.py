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


def _write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    keys = [
        "trial_name",
        "group",
        "semantic_mode",
        "semantic_tokenizer",
        "semantic_group_mode",
        "semantic_text_mode",
        "semantic_num_tokens",
        "prompt_backend",
        "prompt_init_source",
        "prompt_distributor_enable",
        "route_setting",
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
    ap = argparse.ArgumentParser("semantic_tokenizer_text_mode_grid_search")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--config-file", default="configs/prompt/cub.yaml")
    ap.add_argument("--out-root", default="output/grid_semantic_tokenizer_equal_manual_route_ts")
    ap.add_argument("--affinity-evolution-enable", default="true", choices=["true", "false"])
    ap.add_argument("--vis-save-raw", default="true", choices=["true", "false"])
    ap.add_argument("--vis-save-images", default="false", choices=["true", "false"])
    ap.add_argument(
        "--semantic-modes",
        default="equal_none,equal_null_residual,equal_text_init_codebook,manual_cub8_null_residual,manual_cub8_text_init_codebook",
    )
    ap.add_argument("--distributor-grid", default="false,true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_root = os.path.abspath(os.path.join(repo_root, args.out_root))
    os.makedirs(out_root, exist_ok=True)

    semantic_modes = _parse_str_list(args.semantic_modes)
    allowed_semantic_modes = [
        "equal_none",
        "equal_null_residual",
        "equal_text_init_codebook",
        "manual_cub8_null_residual",
        "manual_cub8_text_init_codebook",
    ]
    _validate_choices("semantic mode", semantic_modes, allowed_semantic_modes)
    distributor_grid = [_parse_bool(x) for x in _parse_str_list(args.distributor_grid)]
    affinity_evolution_enable = _parse_bool(args.affinity_evolution_enable)
    vis_save_raw = _parse_bool(args.vis_save_raw)
    vis_save_images = _parse_bool(args.vis_save_images)
    base_opts = [
        "MODEL.PROMPT.BACKEND", "dynamic",
        "MODEL.AFFINITY_EVOLUTION.ENABLE", str(affinity_evolution_enable),
        "SOLVER.VIS.SAVE_RAW", str(vis_save_raw),
        "SOLVER.VIS.SAVE_IMAGES", str(vis_save_images),
        "MODEL.SEMANTIC_TOKENS.ENABLE", "True",
        "MODEL.SEMANTIC_TOKENS.TRAIN_SOURCE", "class_mean",
        "MODEL.SEMANTIC_TOKENS.EVAL_SOURCE", "class_mean",
    ]
    if args.opts:
        _validate_extra_opts(args.opts)
        base_opts.extend(args.opts)

    semantic_mode_specs = {
        "equal_none": {
            "semantic_tokenizer": "orthogonal",
            "semantic_group_mode": "equal",
            "semantic_text_mode": "none",
            "semantic_num_tokens": 8,
            "opts": [
                "MODEL.SEMANTIC_TOKENS.TOKENIZER", "orthogonal",
                "MODEL.SEMANTIC_TOKENS.NUM_TOKENS", "8",
                "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE", "equal",
                "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE", "none",
            ],
        },
        "equal_null_residual": {
            "semantic_tokenizer": "orthogonal",
            "semantic_group_mode": "equal",
            "semantic_text_mode": "null_residual",
            "semantic_num_tokens": 8,
            "opts": [
                "MODEL.SEMANTIC_TOKENS.TOKENIZER", "orthogonal",
                "MODEL.SEMANTIC_TOKENS.NUM_TOKENS", "8",
                "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE", "equal",
                "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE", "null_residual",
            ],
        },
        "equal_text_init_codebook": {
            "semantic_tokenizer": "orthogonal",
            "semantic_group_mode": "equal",
            "semantic_text_mode": "text_init_codebook",
            "semantic_num_tokens": 8,
            "opts": [
                "MODEL.SEMANTIC_TOKENS.TOKENIZER", "orthogonal",
                "MODEL.SEMANTIC_TOKENS.NUM_TOKENS", "8",
                "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE", "equal",
                "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE", "text_init_codebook",
            ],
        },
        "manual_cub8_null_residual": {
            "semantic_tokenizer": "orthogonal",
            "semantic_group_mode": "manual_cub8",
            "semantic_text_mode": "null_residual",
            "semantic_num_tokens": 8,
            "opts": [
                "MODEL.SEMANTIC_TOKENS.TOKENIZER", "orthogonal",
                "MODEL.SEMANTIC_TOKENS.NUM_TOKENS", "8",
                "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE", "manual_cub8",
                "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE", "null_residual",
            ],
        },
        "manual_cub8_text_init_codebook": {
            "semantic_tokenizer": "orthogonal",
            "semantic_group_mode": "manual_cub8",
            "semantic_text_mode": "text_init_codebook",
            "semantic_num_tokens": 8,
            "opts": [
                "MODEL.SEMANTIC_TOKENS.TOKENIZER", "orthogonal",
                "MODEL.SEMANTIC_TOKENS.NUM_TOKENS", "8",
                "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE", "manual_cub8",
                "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE", "text_init_codebook",
            ],
        },
    }
    route_setting_specs = {
        "route_off": {
            "prompt_lambda": 0.0,
            "semantic_lambda": 0.0,
            "route_ts_prompt_weight": 0.0,
            "route_ts_semantic_weight": 0.0,
            "opts": [
                "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA", "0",
                "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA", "0",
                "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT", "0",
                "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT", "0",
            ],
        },
        "route_best": {
            "prompt_lambda": 0.1,
            "semantic_lambda": 0.2,
            "route_ts_prompt_weight": 1e-3,
            "route_ts_semantic_weight": 0.0,
            "opts": [
                "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA", "0.1",
                "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA", "0.2",
                "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT", "0.001",
                "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT", "0",
            ],
        },
    }

    trials: List[Dict[str, object]] = []
    for route_setting, route_spec in route_setting_specs.items():
        for distributor_enable in distributor_grid:
            prompt_init_source = "distributor_mean" if distributor_enable else "learned"
            prompt_setting = "dynamic_distributor_mean" if distributor_enable else "dynamic_learned"
            for semantic_mode in semantic_modes:
                spec = semantic_mode_specs[semantic_mode]
                params = {
                    "route": route_setting,
                    "prompt": prompt_setting,
                    "semantic": semantic_mode,
                }
                trials.append(
                    {
                        "group": "semantic_tokenizer_text_modes_route_ts",
                        "tag": _trial_tag("semantic_mode", params),
                        "semantic_mode": semantic_mode,
                        "semantic_tokenizer": spec["semantic_tokenizer"],
                        "semantic_group_mode": spec["semantic_group_mode"],
                        "semantic_text_mode": spec["semantic_text_mode"],
                        "semantic_num_tokens": spec["semantic_num_tokens"],
                        "prompt_backend": "dynamic",
                        "prompt_init_source": prompt_init_source,
                        "prompt_distributor_enable": distributor_enable,
                        "route_setting": route_setting,
                        "prompt_lambda": route_spec["prompt_lambda"],
                        "semantic_lambda": route_spec["semantic_lambda"],
                        "route_ts_prompt_weight": route_spec["route_ts_prompt_weight"],
                        "route_ts_semantic_weight": route_spec["route_ts_semantic_weight"],
                        "opts": [
                            "MODEL.PROMPT.INIT_SOURCE", prompt_init_source,
                            "MODEL.PROMPT.DISTRIBUTOR.ENABLE", str(distributor_enable),
                        ] + list(spec["opts"]) + list(route_spec["opts"]),
                    }
                )

    search_space = {
        "config_file": args.config_file,
        "python_bin": args.python_bin,
        "out_root": out_root,
        "total_trials": len(trials),
        "sweep": [
            "MODEL.PROMPT.INIT_SOURCE",
            "MODEL.PROMPT.DISTRIBUTOR.ENABLE",
            "MODEL.SEMANTIC_TOKENS.TOKENIZER",
            "MODEL.SEMANTIC_TOKENS.NUM_TOKENS",
            "MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE",
            "MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE",
            "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA",
            "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA",
            "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT",
            "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT",
        ],
        "semantic_modes": semantic_modes,
        "distributor_grid": distributor_grid,
        "route_settings": {
            name: {
                "MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA": spec["prompt_lambda"],
                "MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA": spec["semantic_lambda"],
                "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT": spec["route_ts_prompt_weight"],
                "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT": spec["route_ts_semantic_weight"],
            }
            for name, spec in route_setting_specs.items()
        },
        "fixed_prompt_setting": {
            "MODEL.PROMPT.BACKEND": "dynamic",
            "MODEL.AFFINITY_EVOLUTION.ENABLE": affinity_evolution_enable,
            "distributor_false": {
                "MODEL.PROMPT.INIT_SOURCE": "learned",
                "MODEL.PROMPT.DISTRIBUTOR.ENABLE": False,
            },
            "distributor_true": {
                "MODEL.PROMPT.INIT_SOURCE": "distributor_mean",
                "MODEL.PROMPT.DISTRIBUTOR.ENABLE": True,
            },
        },
        "semantic_mode_specs": {
            name: {
                "semantic_tokenizer": spec["semantic_tokenizer"],
                "semantic_group_mode": spec["semantic_group_mode"],
                "semantic_text_mode": spec["semantic_text_mode"],
                "semantic_num_tokens": spec["semantic_num_tokens"],
            }
            for name, spec in semantic_mode_specs.items()
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
            "semantic_mode": trial["semantic_mode"],
            "semantic_tokenizer": trial["semantic_tokenizer"],
            "semantic_group_mode": trial["semantic_group_mode"],
            "semantic_text_mode": trial["semantic_text_mode"],
            "semantic_num_tokens": trial["semantic_num_tokens"],
            "prompt_backend": trial["prompt_backend"],
            "prompt_init_source": trial["prompt_init_source"],
            "prompt_distributor_enable": trial["prompt_distributor_enable"],
            "route_setting": trial["route_setting"],
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
