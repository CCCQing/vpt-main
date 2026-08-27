#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = "6e70b9e4c8fda553be7b193eace8249dd2cf5d43"
EXPECTED_GRAPH_SHA256 = "50B6AAC7F1D4DEAB1680510500D2DB308DEF213AD323C8E503163DE5C5CEA7BE"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_output(root, *args):
    proc = subprocess.run(
        ["git", "-C", str(root)] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def _prepare_local_path_config(legacy_root, local_path_config):
    source = Path(local_path_config).resolve()
    if not source.is_file():
        raise FileNotFoundError("Missing historical local path config: {}".format(source))
    target = Path(legacy_root) / "src" / "configs" / "local_path.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(target))
    return target


def _legacy_identity(legacy_root, graph_path, local_path_config):
    installed_local_path = _prepare_local_path_config(legacy_root, local_path_config)
    commit = _git_output(legacy_root, "rev-parse", "HEAD")
    dirty_text = _git_output(legacy_root, "status", "--porcelain")
    if commit != EXPECTED_COMMIT:
        raise ValueError(
            "Legacy worktree commit mismatch: expected {}, got {}".format(
                EXPECTED_COMMIT, commit
            )
        )
    if dirty_text:
        raise ValueError("Legacy worktree must be clean before exact replay.")
    config = Path(legacy_root) / "configs" / "prompt" / "cub.yaml"
    entry = (
        Path(legacy_root)
        / "src"
        / "tools"
        / "grid_search_graph_prob_prior_v5_temperatures.py"
    )
    if not config.is_file():
        raise FileNotFoundError("Missing original config: {}".format(config))
    if not entry.is_file():
        raise FileNotFoundError("Missing original training entry: {}".format(entry))
    graph_sha256 = _sha256_file(graph_path)
    if graph_sha256 != EXPECTED_GRAPH_SHA256:
        raise ValueError(
            "Historical graph SHA-256 mismatch: expected {}, got {}".format(
                EXPECTED_GRAPH_SHA256, graph_sha256
            )
        )
    return {
        "legacy_root": str(Path(legacy_root).resolve()),
        "legacy_commit": commit,
        "legacy_dirty": False,
        "source_config": str(config.resolve()),
        "source_config_sha256": _sha256_file(config),
        "training_entry": str(entry.resolve()),
        "training_entry_sha256": _sha256_file(entry),
        "local_path_config": str(installed_local_path.resolve()),
        "local_path_config_sha256": _sha256_file(installed_local_path),
        "graph_path": str(Path(graph_path).resolve()),
        "graph_sha256": graph_sha256,
    }


def _historical_overrides(graph_path, run_root, seed_mode, seed):
    overrides = [
        "RUN_N_TIMES", "1",
        "DATA.BATCH_SIZE", "64",
        "SOLVER.BASE_LR", "0.00125",
        "SOLVER.TOTAL_EPOCH", "10",
        "SOLVER.WARMUP_EPOCH", "1",
        "SOLVER.LOSS_PROMPT_KL_WEIGHT", "0.0",
        "SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT", "0.0",
        "SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT", "0.0",
        "SOLVER.VIS.ENABLE", "False",
        "SOLVER.VIS.SAVE_RAW", "False",
        "SOLVER.VIS.SAVE_IMAGES", "False",
        "SOLVER.VIS.ROLLOUT", "False",
        "MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_ENABLE", "False",
        "MODEL.SEMANTIC_GRAPH.ENABLE", "False",
        "MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE", "external",
        "MODEL.SEMANTIC_GRAPH.LOSS_TYPE", "none",
        "MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT", "0.0",
        "MODEL.SEMANTIC_GRAPH.TOPK", "16",
        "MODEL.SEMANTIC_GRAPH.TAU_ACC", "0.1",
        "MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA", "0.1",
        "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_SYMMETRIZE", "True",
        "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_CLAMP", "True",
        "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_DIAG_VALUE", "1.0",
        "MODEL.GRAPH_PROB_PRIOR.ENABLE", "True",
        "MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT", "0.001",
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_MEAN_MODE", "residual_anchor",
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_VAR_MODE", "unit",
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_LOGVAR_CONST", "0.0",
        "MODEL.GRAPH_PROB_PRIOR.RESIDUAL_SIGMA_MIN", "0.05",
        "MODEL.GRAPH_PROB_PRIOR.RESIDUAL_CLIP", "3.0",
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_CONTEXT_TOPK", "16",
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_MU_SCALE", "2.0",
        "MODEL.GRAPH_PROB_PRIOR.PRIOR_DELTA_SCALE", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.TAU_GRAPH", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.TAU_LATENT", "0.05",
        "MODEL.GRAPH_PROB_PRIOR.REL_WEIGHT", "0.0",
        "MODEL.GRAPH_PROB_PRIOR.TAU_PRIOR", "0.07",
        "MODEL.GRAPH_PROB_PRIOR.MMD_SAMPLES", "1",
        "MODEL.GRAPH_PROB_PRIOR.MMD_SIGMA", "32.0",
        "MODEL.GRAPH_PROB_PRIOR.FACTORIZED_VARIATION_WEIGHT", "0.0",
        "MODEL.GRAPH_PROB_PRIOR.FACTORIZED_DECOUPLE_WEIGHT", "0.0",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_ENABLE", "False",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_TYPE", "soft_distribution_matching",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_WEIGHT", "0.0001",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_TOPK", "5",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_SIGMA_PRIOR", "0.2",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_UNI", "2.0",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_GRAPH_TOP", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_CON", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_GRAPH_POS", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_GRAPH_DIST", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_DIST", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.GEOM_BOUND_WEIGHT", "0.1",
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE", "True",
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_INACTIVE", "False",
        "MODEL.GRAPH_PROB_PRIOR.MONITOR_TOPK", "5",
        "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_PATH", str(Path(graph_path).resolve()),
        "MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_KEY", "method1_diff",
        "MODEL.GRAPH_PROB_PRIOR.MODE", "class_aggregate_mmd",
        "OUTPUT_DIR", str(Path(run_root).resolve()),
        "NUM_GPUS", "2",
    ]
    if seed_mode == "fixed":
        overrides[0:0] = ["SEED", str(int(seed))]
    return overrides


def _command(python_bin, identity, graph_path, run_root, seed_mode, seed):
    return [
        str(python_bin),
        identity["training_entry"],
        "--ddp-train",
        "--nproc-per-node", "2",
        "--dist-backend", "nccl",
        "--config-file", identity["source_config"],
    ] + _historical_overrides(graph_path, run_root, seed_mode, seed)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _validate(run_root, seed_mode, seed):
    sys.path.insert(0, str(ROOT))
    from src.tools.validate_c_legacy_rng_replay import validate_run

    result = validate_run(run_root, seed_mode, seed, 10)
    _write_json(Path(run_root) / "validation.json", result)
    return result


def _run_one(task, args, identity, gpu_group):
    run_root = Path(task["run_root"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    validation_path = run_root / "validation.json"
    if args.resume and validation_path.is_file():
        try:
            previous = json.loads(validation_path.read_text(encoding="utf-8"))
            if bool(previous.get("valid")):
                return {"status": "skipped_valid", "gpu_group": gpu_group, **previous}
        except Exception:
            pass

    seed_mode = task["seed_mode"]
    seed = task.get("seed")
    command = _command(
        args.python, identity, args.graph_path, run_root, seed_mode, seed
    )
    command_shell = " ".join(shlex.quote(part) for part in command)
    command_record = {
        "seed_mode": seed_mode,
        "seed": seed,
        "gpu_group": gpu_group,
        "command": command,
        "command_shell": command_shell,
        "cwd": identity["legacy_root"],
    }
    _write_json(run_root / "runner_command.json", command_record)
    run_identity = dict(identity)
    run_identity.update(
        {
            "schema_version": 1,
            "protocol": "exact_historical_cli",
            "seed_mode": seed_mode,
            "seed": seed,
            "gpu_group": gpu_group,
            "training_performed": False,
            "optimizer_created": False,
            "checkpoint_expected": False,
            "unseeded_not_in_fixed_seed_mean": seed_mode == "unseeded",
        }
    )
    _write_json(run_root / "legacy_identity.json", run_identity)
    shutil.copyfile(identity["source_config"], run_root / "source_config_snapshot.yaml")
    shutil.copyfile(identity["local_path_config"], run_root / "local_path_config_snapshot.yaml")
    if args.dry_run:
        return {"status": "dry_run", "valid": True, **command_record}

    print(
        "[start] seed_mode={} seed={} gpus={} output={}".format(
            seed_mode, seed, gpu_group, run_root
        ),
        flush=True,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_group
    launcher_log = run_root / "legacy_launcher.log"
    with launcher_log.open("a", encoding="utf-8") as handle:
        handle.write(command_shell + "\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=identity["legacy_root"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode == 0:
        run_identity["training_performed"] = True
        run_identity["optimizer_created"] = True
        _write_json(run_root / "legacy_identity.json", run_identity)
        result = _validate(run_root, seed_mode, seed)
    else:
        result = {
            "valid": False,
            "errors": ["launcher return code {}".format(proc.returncode)],
            "run_root": str(run_root),
        }
    result.update(
        {
            "status": "complete" if result.get("valid") else "failed",
            "seed_mode": seed_mode,
            "seed": seed,
            "gpu_group": gpu_group,
            "launcher_returncode": int(proc.returncode),
        }
    )
    _write_json(run_root / "run_status.json", result)
    print(
        "[done] seed_mode={} seed={} gpus={} status={} h={}".format(
            seed_mode,
            seed,
            gpu_group,
            result["status"],
            result.get("final_metrics", {}).get("h", ""),
        ),
        flush=True,
    )
    return result


def _tasks(output_root, stage, seeds):
    tasks = []
    if stage in {"unseeded", "all"}:
        tasks.append(
            {
                "seed_mode": "unseeded",
                "seed": None,
                "run_root": str(Path(output_root) / "legacy_unseeded" / "replay_0"),
            }
        )
    if stage in {"fixed", "all"}:
        for seed in seeds:
            tasks.append(
                {
                    "seed_mode": "fixed",
                    "seed": int(seed),
                    "run_root": str(Path(output_root) / "legacy_fixed" / "seed_{}".format(seed)),
                }
            )
    return tasks


def _run_task_batch(tasks, args, identity, gpu_groups):
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
                    current = _run_one(item, args, identity, group)
                    worker_results.append(current)
                    if not current.get("valid", False):
                        break
                return worker_results

            futures.append(pool.submit(worker))
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Replay historical t0009 on the exact legacy commit with unseeded and fixed-seed protocols."
    )
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output" / "c_gpp_t0009_legacy_replay",
    )
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--local-path-config", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage", choices=["unseeded", "fixed", "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpu-groups", default="0,1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.legacy_root = args.legacy_root.resolve()
    args.output_root = args.output_root.resolve()
    args.graph_path = args.graph_path.resolve()
    args.local_path_config = args.local_path_config.resolve()
    if not args.graph_path.is_file():
        raise FileNotFoundError("Missing historical graph file: {}".format(args.graph_path))
    identity = _legacy_identity(
        args.legacy_root, args.graph_path, args.local_path_config
    )
    gpu_groups = [item.strip() for item in args.gpu_groups.split(";") if item.strip()]
    if not gpu_groups:
        raise ValueError("--gpu-groups must contain at least one two-GPU group.")
    for group in gpu_groups:
        if len([item for item in group.split(",") if item.strip()]) != 2:
            raise ValueError("Each GPU group must contain exactly two GPUs: '{}'".format(group))
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")

    task_list = _tasks(args.output_root, args.stage, args.seeds)
    results = _run_task_batch(task_list, args, identity, gpu_groups)
    fixed_results = sorted(
        [item for item in results if item.get("seed_mode") == "fixed"],
        key=lambda item: int(item.get("seed")),
    )
    unseeded_results = [item for item in results if item.get("seed_mode") == "unseeded"]
    summary = {
        "schema_version": 1,
        "protocol": "exact_historical_cli_rng_replay",
        "stage": args.stage,
        "output_root": str(args.output_root),
        "legacy_identity": identity,
        "gpu_groups": gpu_groups,
        "fixed_seed_results": fixed_results,
        "unseeded_exploratory_results": unseeded_results,
        "aggregation_contract": {
            "fixed_seed_mean_members": [item.get("seed") for item in fixed_results],
            "unseeded_excluded_from_fixed_seed_mean": True,
        },
        "all_valid": bool(results) and all(item.get("valid", False) for item in results),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "legacy_replay_status.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["all_valid"] else 1)


if __name__ == "__main__":
    main()
