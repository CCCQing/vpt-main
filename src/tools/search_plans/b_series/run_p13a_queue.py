#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


ROOT = Path(__file__).resolve().parents[4]
RUN_SUFFIX = Path("CUB/sup_vitb16_224/lr0.0006_wd1e-05/run1")
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Job:
    seed: int
    gpu: str
    source_run: Path
    output_dir: Path
    log_path: Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-groups", default="0;3;5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs" / "b_series_experiments" / "P1-3a-head-only.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _gpus(raw: str) -> List[str]:
    values = [value.strip() for value in str(raw).split(";") if value.strip()]
    if len(values) < len(SEEDS) or any("," in value for value in values):
        raise ValueError("P1-3a needs at least three single-GPU slots separated by semicolons")
    return values


def _a2_run(root: Path, seed: int) -> Path:
    candidates = (
        root / "A2" / "seed{}".format(seed) / RUN_SUFFIX,
        root / "seed{}".format(seed) / RUN_SUFFIX,
    )
    matches = [
        path
        for path in candidates
        if (path / "model_final_trainable.pth").is_file()
        and (path / "resolved_config.yaml").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one A2 seed{} run under {}; checked {}".format(
                seed, root, ", ".join(str(path) for path in candidates)
            )
        )
    return matches[0]


def _command(job: Job, args) -> List[str]:
    return [
        str(args.python_bin),
        "-m",
        "src.tools.search_plans.b_series.run_p13a_compatibility",
        "--source-run",
        str(job.source_run),
        "--output-dir",
        str(job.output_dir),
        "--experiment-config",
        str(args.experiment_config.expanduser().resolve()),
        "--head-seed",
        str(job.seed),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]


def _format(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _completed(path: Path) -> bool:
    state = path / "execution_state.json"
    result = path / "p13a_result.json"
    if not state.is_file() or not result.is_file():
        return False
    try:
        state_payload = json.loads(state.read_text(encoding="utf-8"))
        result_payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        state_payload.get("status") == "completed"
        and result_payload.get("status") == "completed"
        and result_payload.get("valid") is True
    )


def _run(job: Job, args) -> dict:
    command = _command(job, args)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": job.gpu,
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
        }
    )
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with job.log_path.open("w", encoding="utf-8") as handle:
        handle.write("CUDA_VISIBLE_DEVICES={}\n{}\n\n".format(job.gpu, _format(command)))
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    valid = process.returncode == 0 and _completed(job.output_dir)
    return {
        "seed": job.seed,
        "gpu": job.gpu,
        "status": "completed" if valid else "failed",
        "returncode": int(process.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "source_run": str(job.source_run),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
    }


def main() -> None:
    args = _parse()
    gpus = _gpus(args.gpu_groups)
    a2_root = args.a2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    jobs = [
        Job(
            seed=seed,
            gpu=gpus[index],
            source_run=_a2_run(a2_root, seed),
            output_dir=output_root / "seed{}".format(seed),
            log_path=output_root / "launcher_logs" / "seed{}.log".format(seed),
        )
        for index, seed in enumerate(SEEDS)
    ]
    pending = []
    for job in jobs:
        if _completed(job.output_dir):
            continue
        if job.output_dir.is_dir() and next(job.output_dir.iterdir(), None) is not None:
            raise FileExistsError("refusing to overwrite incomplete {}".format(job.output_dir))
        pending.append(job)
    if args.dry_run:
        for job in pending:
            print("CUDA_VISIBLE_DEVICES={} {}".format(job.gpu, _format(_command(job, args))))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=len(pending) or 1) as executor:
        futures = {executor.submit(_run, job, args): job for job in pending}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    for job in jobs:
        if not any(record["seed"] == job.seed for record in records):
            records.append(
                {
                    "seed": job.seed,
                    "gpu": job.gpu,
                    "status": "skipped_completed",
                    "source_run": str(job.source_run),
                    "output_dir": str(job.output_dir),
                    "log_path": str(job.log_path),
                }
            )
    queue = {
        "format": "p13a_queue_summary_v1",
        "status": (
            "completed"
            if all(_completed(job.output_dir) for job in jobs)
            else "failed"
        ),
        "jobs": sorted(records, key=lambda item: int(item["seed"])),
    }
    (output_root / "queue_summary.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if queue["status"] != "completed":
        raise SystemExit(1)
    for job in jobs:
        subprocess.run(
            [
                str(args.python_bin),
                "-m",
                "src.tools.validate_p13a_compatibility",
                "--result",
                str(job.output_dir / "p13a_result.json"),
            ],
            cwd=str(ROOT),
            check=True,
        )
    subprocess.run(
        [
            str(args.python_bin),
            "-m",
            "src.tools.search_plans.b_series.summarize_p13a_compatibility",
            "--root",
            str(output_root),
        ],
        cwd=str(ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
