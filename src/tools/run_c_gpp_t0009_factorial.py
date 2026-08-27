#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DDP_ENTRY = ROOT / "src" / "tools" / "grid_search_graph_prob_prior_v5_temperatures.py"
DEFAULT_CONFIG = ROOT / "configs" / "c_series" / "gpp_t0009_frozen.yaml"

GROUPS = {
    "c1_full_historical": {"gpp": True, "am": True},
    "c2_gpp_only": {"gpp": True, "am": False},
    "c3_am_only": {"gpp": False, "am": True},
    "c4_neither": {"gpp": False, "am": False},
}
EXPECTED_GRAPH_SHA256 = "50B6AAC7F1D4DEAB1680510500D2DB308DEF213AD323C8E503163DE5C5CEA7BE"


def _bool_text(value):
    return "True" if bool(value) else "False"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _run_command(python_bin, config, graph_path, run_root, group_name, seed, epochs, gpu_group):
    group = GROUPS[group_name]
    nproc = len([item for item in gpu_group.split(",") if item.strip()])
    if nproc != 2:
        raise ValueError(f"Historical t0009 contract requires two GPUs per run, got '{gpu_group}'.")
    return [
        str(python_bin),
        str(DDP_ENTRY),
        "--ddp-train",
        "--nproc-per-node",
        str(nproc),
        "--dist-backend",
        "nccl",
        "--config-file",
        str(config),
        "SEED",
        str(seed),
        "SOLVER.TOTAL_EPOCH",
        str(epochs),
        "MODEL.GRAPH_PROB_PRIOR.ENABLE",
        _bool_text(group["gpp"]),
        "MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT",
        "0.001" if group["gpp"] else "0.0",
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE",
        _bool_text(group["gpp"]),
        "MODEL.ATTENTION_MEDIATION.ENABLE",
        _bool_text(group["am"]),
        "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_PATH",
        str(graph_path),
        "OUTPUT_DIR",
        str(run_root),
        "NUM_GPUS",
        str(nproc),
    ]


def _validate(run_root, group_name, seed, epochs):
    sys.path.insert(0, str(ROOT))
    from src.tools.validate_c_gpp_t0009_run import validate_run

    group = GROUPS[group_name]
    result = validate_run(run_root, seed, epochs, group["gpp"], group["am"])
    (Path(run_root) / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _run_one(task, python_bin, config, graph_path, gpu_group, resume, dry_run):
    group_name = task["group"]
    seed = int(task["seed"])
    epochs = int(task["epochs"])
    run_root = Path(task["run_root"]).absolute()
    run_root.mkdir(parents=True, exist_ok=True)
    validation_path = run_root / "validation.json"
    if resume and validation_path.is_file():
        try:
            previous = json.loads(validation_path.read_text(encoding="utf-8"))
            if bool(previous.get("valid")):
                return {"status": "skipped_valid", "gpu_group": gpu_group, **previous}
        except Exception:
            pass

    command = _run_command(
        python_bin, config, graph_path, run_root, group_name, seed, epochs, gpu_group
    )
    command_record = {
        "group": group_name,
        "seed": seed,
        "epochs": epochs,
        "gpu_group": gpu_group,
        "command": command,
        "command_shell": " ".join(shlex.quote(part) for part in command),
    }
    (run_root / "runner_command.json").write_text(
        json.dumps(command_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if dry_run:
        return {"status": "dry_run", **command_record}

    print(
        "[start] group={} seed={} epochs={} gpus={} output={}".format(
            group_name, seed, epochs, gpu_group, run_root
        ),
        flush=True,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_group
    launcher_log = run_root / "factorial_launcher.log"
    with launcher_log.open("a", encoding="utf-8") as handle:
        handle.write(command_record["command_shell"] + "\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if proc.returncode != 0:
        result = {
            "valid": False,
            "errors": [f"launcher return code {proc.returncode}"],
            "run_root": str(run_root),
        }
    else:
        result = _validate(run_root, group_name, seed, epochs)
    result.update(
        {
            "status": "complete" if result.get("valid") else "failed",
            "group": group_name,
            "seed": seed,
            "epochs": epochs,
            "gpu_group": gpu_group,
            "launcher_returncode": int(proc.returncode),
        }
    )
    (run_root / "run_status.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "[done] group={} seed={} gpus={} status={} h={}".format(
            group_name,
            seed,
            gpu_group,
            result["status"],
            result.get("final_metrics", {}).get("h", ""),
        ),
        flush=True,
    )
    return result


def _run_task_batch(tasks, args, gpu_groups):
    assignments = [[] for _ in gpu_groups]
    for index, task in enumerate(tasks):
        assignments[index % len(gpu_groups)].append(task)

    results = []
    with ThreadPoolExecutor(max_workers=len(gpu_groups)) as pool:
        futures = []
        for gpu_group, assigned in zip(gpu_groups, assignments):
            if not assigned:
                continue

            def worker(queue=assigned, group=gpu_group):
                worker_results = []
                for item in queue:
                    worker_results.append(
                        _run_one(
                            item,
                            args.python,
                            args.config,
                            args.graph_path,
                            group,
                            args.resume,
                            args.dry_run,
                        )
                    )
                    if not worker_results[-1].get("valid", args.dry_run):
                        break
                return worker_results

            futures.append(pool.submit(worker))
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def _tasks_for_matrix(output_root, groups, seeds, epochs):
    tasks = []
    for seed in seeds:
        for group_name in groups:
            tasks.append(
                {
                    "group": group_name,
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "run_root": str(Path(output_root) / "matrix" / group_name / f"seed_{seed}"),
                }
            )
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run the C-series t0009 GPP x AM factorial audit.")
    parser.add_argument("--stage", choices=["smoke", "seed0", "formal", "all"], default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "c_gpp_t0009_factorial")
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=ROOT / "cub_attribute_localization" / "05_hparam_searches" / "diff_only_graphs_v1" / "diff_only_method_matrices_v1.npz",
    )
    parser.add_argument("--gpu-groups", default="0,1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.graph_path = args.graph_path.resolve()
    args.output_root = args.output_root.absolute()
    if not args.dry_run:
        if not args.graph_path.is_file():
            raise FileNotFoundError(f"Missing historical graph file: {args.graph_path}")
        graph_sha256 = _sha256_file(args.graph_path)
        if graph_sha256 != EXPECTED_GRAPH_SHA256:
            raise ValueError(
                f"Historical graph SHA-256 mismatch: expected {EXPECTED_GRAPH_SHA256}, got {graph_sha256}"
            )
    gpu_groups = [item.strip() for item in args.gpu_groups.split(";") if item.strip()]
    if not gpu_groups:
        raise ValueError("--gpu-groups must contain at least one two-GPU group.")
    for group in gpu_groups:
        if len([item for item in group.split(",") if item.strip()]) != 2:
            raise ValueError(f"Each GPU group must contain exactly two GPUs: '{group}'.")

    all_results = []
    if args.stage in {"smoke", "all"}:
        smoke_task = {
            "group": "c1_full_historical",
            "seed": 0,
            "epochs": 1,
            "run_root": str(args.output_root / "smoke" / "c1_full_historical" / "seed_0"),
        }
        smoke_results = _run_task_batch([smoke_task], args, [gpu_groups[0]])
        all_results.extend(smoke_results)
        if not args.dry_run and not all(item.get("valid") for item in smoke_results):
            raise SystemExit("C0 smoke failed; formal matrix was not started.")

    if args.stage in {"seed0", "all"}:
        seed0_tasks = _tasks_for_matrix(args.output_root, args.groups, [0], 10)
        seed0_results = _run_task_batch(seed0_tasks, args, gpu_groups)
        all_results.extend(seed0_results)
        if not args.dry_run and not all(item.get("valid") for item in seed0_results):
            raise SystemExit("Seed0 factorial screening failed; seed1/2 were not started.")

    if args.stage in {"formal", "all"}:
        formal_seeds = args.seeds
        if args.stage == "all":
            formal_seeds = [seed for seed in args.seeds if int(seed) != 0]
        formal_tasks = _tasks_for_matrix(args.output_root, args.groups, formal_seeds, 10)
        formal_results = _run_task_batch(formal_tasks, args, gpu_groups)
        all_results.extend(formal_results)
        if not args.dry_run and not all(item.get("valid") for item in formal_results):
            raise SystemExit("One or more formal factorial runs failed validation.")

    summary = {
        "schema_version": 1,
        "stage": args.stage,
        "output_root": str(args.output_root),
        "graph_path": str(args.graph_path),
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "gpu_groups": gpu_groups,
        "results": all_results,
        "all_valid": bool(all_results) and all(item.get("valid", args.dry_run) for item in all_results),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "factorial_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
