#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a fail-fast sequential queue of A-series monitoring replays."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--queue-name", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="METHOD:SEED, for example A1:2",
    )
    parser.add_argument(
        "--scope", choices=("gaps", "object_map", "full"), default="gaps"
    )
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_run(spec):
    method, separator, seed_text = str(spec).partition(":")
    method = method.strip().upper()
    if not separator or method not in {"A0", "A1", "A2"}:
        raise ValueError(f"invalid --run value: {spec}")
    seed = int(seed_text)
    if seed < 0:
        raise ValueError(f"seed must be non-negative: {spec}")
    return method, seed


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    queue_dir = output_root / "_launcher_logs"
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_summary_path = queue_dir / f"{args.queue_name}_summary.json"
    if queue_summary_path.exists():
        raise FileExistsError(str(queue_summary_path))

    replay_script = Path(__file__).with_name("replay_monitoring_gaps.py")
    summary = {
        "format": "a_series_monitor_gap_replay_queue_v1",
        "queue_name": str(args.queue_name),
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "scope": str(args.scope),
        "selection_seed": (
            int(args.selection_seed) if args.selection_seed is not None else None
        ),
        "batch_size": int(args.batch_size),
        "runs": [],
    }
    _write_json(queue_summary_path, summary)

    try:
        for spec in args.run:
            method, seed = _parse_run(spec)
            source_run = source_root / method / f"seed{seed}" / RUN_SUFFIX
            scoped_root = (
                output_root / f"selection_seed_{int(args.selection_seed)}"
                if args.selection_seed is not None
                else output_root
            )
            output_run = scoped_root / method / f"seed{seed}" / RUN_SUFFIX
            replay_summary_path = output_run / (
                "bayesian_object_map_replay_summary.json"
                if args.scope == "object_map"
                else "monitor_gap_replay_summary.json"
            )
            item = {
                "method": method,
                "seed": seed,
                "source_run": str(source_run),
                "output_run": str(output_run),
                "started_at": _utc_now(),
                "finished_at": None,
                "status": "running",
                "return_code": None,
            }
            summary["runs"].append(item)
            _write_json(queue_summary_path, summary)

            if replay_summary_path.is_file() and bool(
                _read_json(replay_summary_path).get("valid", False)
            ):
                item.update({
                    "finished_at": _utc_now(),
                    "status": "skipped_existing_valid",
                    "return_code": 0,
                })
                _write_json(queue_summary_path, summary)
                continue

            selection_prefix = (
                f"selection_seed_{int(args.selection_seed)}_"
                if args.selection_seed is not None
                else ""
            )
            log_path = queue_dir / (
                f"{selection_prefix}{method}_seed{seed}.log"
            )
            command = [
                sys.executable,
                "-u",
                str(replay_script),
                "--source-run",
                str(source_run),
                "--output-run",
                str(output_run),
                "--scope",
                str(args.scope),
                "--batch-size",
                str(args.batch_size),
            ]
            if args.selection_seed is not None:
                command.extend(["--selection-seed", str(int(args.selection_seed))])
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(replay_script.parents[4]),
                    check=False,
                )
                handle.write(f"EXIT_CODE={completed.returncode}\n")
            valid = replay_summary_path.is_file() and bool(
                _read_json(replay_summary_path).get("valid", False)
            )
            item.update({
                "finished_at": _utc_now(),
                "status": "completed" if completed.returncode == 0 and valid else "failed",
                "return_code": int(completed.returncode),
                "valid": valid,
                "log_path": str(log_path),
            })
            _write_json(queue_summary_path, summary)
            if item["status"] != "completed":
                raise RuntimeError(
                    f"monitoring replay failed for {method}:seed{seed}; see {log_path}"
                )
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["finished_at"] = _utc_now()
        _write_json(queue_summary_path, summary)


if __name__ == "__main__":
    main()
