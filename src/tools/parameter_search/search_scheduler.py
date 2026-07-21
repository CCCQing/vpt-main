from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.parameter_search.search_eta import SearchEtaTracker


CommandBuilder = Callable[[Mapping[str, Any], str], Tuple[List[str], int]]
CompleteCheck = Callable[[Mapping[str, Any]], bool]
ResultFlattener = Callable[..., Dict[str, Any]]
ResumeStatus = Callable[[Mapping[str, Any]], str]


def default_dist_backend() -> str:
    return "gloo" if os.name == "nt" else "nccl"


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_gpu_groups(raw_groups: str, raw_gpus: str) -> List[str]:
    if str(raw_groups).strip():
        return [item.strip() for item in str(raw_groups).split(";") if item.strip()]
    if str(raw_gpus).strip():
        return parse_csv(raw_gpus)
    return [""]


def group_nproc(gpu_group: str, fallback_nproc: int) -> int:
    if fallback_nproc > 0:
        return int(fallback_nproc)
    if not gpu_group:
        return 1
    return len([item for item in str(gpu_group).split(",") if item.strip()])


def value_to_opt(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return str(value)


def mapping_to_opts(mapping: Mapping[str, Any]) -> List[str]:
    opts: List[str] = []
    for key, value in mapping.items():
        opts.extend([str(key), value_to_opt(value)])
    return opts


def validate_extra_opts(opts: Sequence[str]) -> List[str]:
    values = list(opts)
    if values and values[0] == "--":
        values = values[1:]
    if len(values) % 2 != 0:
        raise ValueError("Extra opts must use KEY VALUE pairs.")
    return values


def validate_gpu_groups(gpu_groups: Sequence[str], max_workers: int, nproc_per_trial: int) -> None:
    if int(max_workers) <= 0:
        raise ValueError("--max-workers must be positive.")
    if int(max_workers) > len(gpu_groups):
        raise ValueError("--max-workers must not exceed the number of GPU groups.")
    for gpu_group in gpu_groups:
        nproc = group_nproc(gpu_group, int(nproc_per_trial))
        visible_gpu_count = len([item for item in str(gpu_group).split(",") if item.strip()])
        if gpu_group and nproc > 1 and nproc != visible_gpu_count:
            raise ValueError(
                "--nproc-per-trial must match each --gpu-groups item size "
                "when CUDA_VISIBLE_DEVICES is set."
            )


def patch_ddp_forward_missing_attrs() -> None:
    import torch

    ddp_cls = torch.nn.parallel.DistributedDataParallel
    if getattr(ddp_cls, "_search_forward_missing_attrs", False):
        return
    original_getattr = ddp_cls.__getattr__

    def forwarded_getattr(self, name):
        try:
            return original_getattr(self, name)
        except AttributeError as exc:
            module = original_getattr(self, "module")
            if hasattr(module, name):
                return getattr(module, name)
            raise exc

    ddp_cls.__getattr__ = forwarded_getattr
    ddp_cls._search_forward_missing_attrs = True


def _set_cli_opt(opts: Sequence[str], key: str, value: Any) -> List[str]:
    updated = list(opts)
    for idx, item in enumerate(updated):
        if item == key:
            if idx + 1 >= len(updated):
                raise ValueError(f"Malformed CLI opts: key {key} has no value.")
            updated[idx + 1] = str(value)
            return updated
    updated.extend([key, str(value)])
    return updated


def _rebuild_train_argv(train_args: Any, opts: Sequence[str]) -> List[str]:
    argv: List[str] = []
    if getattr(train_args, "config_file", ""):
        argv.extend(["--config-file", str(train_args.config_file)])
    if getattr(train_args, "train_type", ""):
        argv.extend(["--train-type", str(train_args.train_type)])
    argv.extend(list(opts))
    return argv


def _select_fixed_train_output(train_argv: Sequence[str], nproc: int) -> Dict[str, Any]:
    import train as train_entry
    from launch import default_argument_parser

    train_args = default_argument_parser().parse_args(list(train_argv))
    cfg = train_entry.get_cfg()
    cfg.merge_from_file(train_args.config_file)
    train_entry._merge_local_path_cfg_if_exists(cfg)
    cfg.merge_from_list(train_args.opts)
    train_entry._sync_xlsa_protocol(cfg)

    base_output_dir = str(cfg.OUTPUT_DIR)
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")
    count = 1
    while True:
        fixed_output_dir = os.path.join(base_output_dir, output_folder, f"run{count}")
        if not train_entry.PathManager.exists(fixed_output_dir):
            train_entry.PathManager.mkdirs(fixed_output_dir)
            break
        count += 1
        if count > 1000:
            raise RuntimeError(f"Too many existing run directories under {base_output_dir}.")

    opts = _set_cli_opt(train_args.opts, "OUTPUT_DIR", fixed_output_dir)
    opts = _set_cli_opt(opts, "NUM_GPUS", int(nproc))
    return {
        "argv": _rebuild_train_argv(train_args, opts),
        "output_dir": fixed_output_dir,
        "nproc": int(nproc),
    }


def _train_with_ddp_attr_forward(payload: Any, _unused: Any = None) -> None:
    patch_ddp_forward_missing_attrs()
    import train as train_entry
    from launch import default_argument_parser
    from src.utils.distributed import get_rank

    old_argv = sys.argv
    try:
        if isinstance(payload, Mapping):
            train_argv = list(payload["argv"])
            fixed_output_dir = str(payload["output_dir"])
            nproc = int(payload.get("nproc", 1))
        else:
            train_argv = list(payload)
            fixed_output_dir = ""
            nproc = 1

        sys.argv = [str(ROOT / "train.py")] + list(train_argv)
        train_args = default_argument_parser().parse_args(list(train_argv))
        if not fixed_output_dir:
            train_entry.main(train_args)
            return

        cfg = train_entry.get_cfg()
        cfg.merge_from_file(train_args.config_file)
        train_entry._merge_local_path_cfg_if_exists(cfg)
        cfg.merge_from_list(train_args.opts)
        train_entry._sync_xlsa_protocol(cfg)

        node = os.environ.get("SLURMD_NODENAME")
        if node:
            cfg.DIST_INIT_PATH = f"tcp://{node}:12399"
        cfg.OUTPUT_DIR = fixed_output_dir
        cfg.NUM_GPUS = int(nproc)
        cfg.DIST_RANK = int(get_rank())
        cfg.freeze()
        train_entry.train(cfg, train_args)
    finally:
        sys.argv = old_argv


def ddp_train_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser("parameter search DDP train")
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--dist-backend", default=default_dist_backend(), choices=["nccl", "gloo"])
    parser.add_argument("--dist-url", default="")
    known, train_argv = parser.parse_known_args(list(argv))
    if known.nproc_per_node <= 1:
        raise ValueError("--nproc-per-node must be greater than 1 in DDP train mode.")
    if not train_argv:
        raise ValueError("Missing train.py arguments after DDP launcher options.")

    import torch.multiprocessing as mp

    from src.utils import distributed as du

    train_payload = _select_fixed_train_output(train_argv, int(known.nproc_per_node))
    init_method = known.dist_url or "tcp://127.0.0.1:{}".format(pick_free_port())
    mp.spawn(
        du.run,
        nprocs=int(known.nproc_per_node),
        args=(
            int(known.nproc_per_node),
            _train_with_ddp_attr_forward,
            init_method,
            0,
            1,
            str(known.dist_backend),
            train_payload,
            None,
        ),
        join=True,
    )


def build_run_command(
    trial: Mapping[str, Any],
    python_bin: str,
    gpu_group: str,
    nproc_per_trial: int,
    dist_backend: str,
    ddp_launcher_path: str = "",
    ddp_mode: str = "",
) -> Tuple[List[str], int]:
    nproc = group_nproc(gpu_group, nproc_per_trial)
    if nproc <= 1:
        return list(trial["cmd"]), 1

    runner = str(trial.get("runner", "train")).lower()
    if runner == "train":
        launcher_path = str(Path(__file__).resolve())
        launcher_mode = "--ddp-train"
    else:
        launcher_path = str(ddp_launcher_path)
        launcher_mode = str(ddp_mode)
        if not launcher_path or not launcher_mode:
            raise ValueError("A non-training DDP trial must provide a launcher path and mode.")

    child_cmd = list(trial["cmd"])
    child_args_start = int(trial.get("ddp_child_args_start", 2))
    child_args = child_cmd[child_args_start:]
    command = [
        python_bin,
        launcher_path,
        launcher_mode,
        "--nproc-per-node",
        str(nproc),
        "--dist-backend",
        str(dist_backend),
    ]
    command.extend(child_args)
    num_gpus_opt = str(trial.get("num_gpus_opt", "NUM_GPUS"))
    if num_gpus_opt:
        command.extend([num_gpus_opt, str(nproc)])
    return command, nproc


def completed_result(
    trial: Mapping[str, Any],
    gpu_id: str,
    command: Sequence[str],
    num_gpus: int,
    complete_check: CompleteCheck,
    result_flattener: ResultFlattener,
) -> Optional[Dict[str, Any]]:
    if not complete_check(trial):
        return None
    return result_flattener(
        trial,
        0,
        gpu_id=gpu_id,
        command=command,
        num_gpus=num_gpus,
        skipped_existing=True,
    )


def execute_trial(
    trial: Mapping[str, Any],
    gpu_id: str,
    command_builder: CommandBuilder,
    complete_check: CompleteCheck,
    result_flattener: ResultFlattener,
    resume: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(str(trial["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["VPT_SEARCH_PROGRESS_PATH"] = str((output_dir / "progress.json").resolve())
    if gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_cmd, nproc = command_builder(trial, gpu_id)
    if resume:
        existing = completed_result(
            trial,
            gpu_id,
            run_cmd,
            nproc,
            complete_check,
            result_flattener,
        )
        if existing is not None:
            return existing
    with Path(str(trial["stdout_path"])).open("w", encoding="utf-8", errors="replace") as stdout:
        proc = subprocess.run(
            run_cmd,
            cwd=str(trial["repo_root"]),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
    return result_flattener(
        trial,
        int(proc.returncode),
        gpu_id=gpu_id,
        command=run_cmd,
        num_gpus=nproc,
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        _replace_file(temp_path, path)
    except BaseException:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _replace_file(temp_path, path)
    except BaseException:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise


def _replace_file(temp_path: Path, path: Path) -> None:
    last_error = None
    for attempt in range(3):
        try:
            os.replace(str(temp_path), str(path))
            return
        except OSError as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.02 * (attempt + 1))
    if last_error is not None:
        raise last_error


class SearchScheduler:
    ACTIVE_PROGRESS_WEIGHT = SearchEtaTracker.ACTIVE_PROGRESS_WEIGHT
    HISTORICAL_WEIGHT = SearchEtaTracker.HISTORICAL_WEIGHT

    def __init__(
        self,
        trials: Sequence[Mapping[str, Any]],
        out_root: Path,
        gpu_groups: Sequence[str],
        max_workers: int,
        nproc_per_trial: int,
        eta_interval: float,
        command_builder: CommandBuilder,
        complete_check: CompleteCheck,
        result_flattener: ResultFlattener,
        resume_status: ResumeStatus,
        resume: bool = True,
        resume_debug: bool = False,
    ) -> None:
        self.trials = list(trials)
        self.out_root = Path(out_root)
        self.gpu_groups = list(gpu_groups) or [""]
        self.max_workers = int(max_workers)
        self.nproc_per_trial = int(nproc_per_trial)
        self.eta_interval = float(eta_interval)
        self.command_builder = command_builder
        self.complete_check = complete_check
        self.result_flattener = result_flattener
        self.resume_status = resume_status
        self.resume = bool(resume)
        self.resume_debug = bool(resume_debug)
        validate_gpu_groups(self.gpu_groups, self.max_workers, self.nproc_per_trial)
        if not math.isfinite(self.eta_interval) or self.eta_interval < 0.0:
            raise ValueError("--eta-interval must be finite and non-negative.")

    def write_commands(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for index, trial in enumerate(self.trials):
                gpu = self.gpu_groups[index % len(self.gpu_groups)]
                run_cmd, _ = self.command_builder(trial, gpu)
                handle.write(subprocess.list2cmdline(run_cmd) + "\n")

    def dry_run(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, trial in enumerate(self.trials):
            gpu = self.gpu_groups[index % len(self.gpu_groups)]
            run_cmd, nproc = self.command_builder(trial, gpu)
            rows.append(
                self.result_flattener(
                    trial,
                    returncode=-1,
                    gpu_id=gpu,
                    command=run_cmd,
                    num_gpus=nproc,
                )
            )
        return rows

    def run(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        worker_gpus = self.gpu_groups[: self.max_workers]
        eta_tracker = SearchEtaTracker(
            out_root=self.out_root,
            trials=self.trials,
            worker_gpus=worker_gpus,
            update_interval_seconds=max(1.0, float(self.eta_interval or 30.0)),
        )

        def run_one(worker_idx: int, gpu: str, trial_idx: int, trial: Mapping[str, Any]) -> Dict[str, Any]:
            try:
                run_cmd, run_nproc = self.command_builder(trial, gpu)
                existing = None
                if self.resume:
                    existing = completed_result(
                        trial,
                        gpu,
                        run_cmd,
                        run_nproc,
                        self.complete_check,
                        self.result_flattener,
                    )
                if existing is not None:
                    print(
                        f"[worker {worker_idx + 1}/{self.max_workers} gpu={gpu}] "
                        f"skip existing {trial_idx + 1}/{len(self.trials)} "
                        f"{trial['trial_name']} nproc={run_nproc}",
                        flush=True,
                    )
                    return eta_tracker.register_finish(worker_idx, existing)
                if self.resume and self.resume_debug:
                    print(
                        f"[worker {worker_idx + 1}/{self.max_workers} gpu={gpu}] "
                        f"resume miss {trial_idx + 1}/{len(self.trials)} "
                        f"{trial['trial_name']}: {self.resume_status(trial)}",
                        flush=True,
                    )
                print(
                    f"[worker {worker_idx + 1}/{self.max_workers} gpu={gpu}] "
                    f"started {trial_idx + 1}/{len(self.trials)} {trial['trial_name']}",
                    flush=True,
                )
                result = execute_trial(
                    trial,
                    gpu,
                    self.command_builder,
                    self.complete_check,
                    self.result_flattener,
                    resume=False,
                )
                result = eta_tracker.register_finish(worker_idx, result)
            except BaseException:
                eta_tracker.register_error(worker_idx)
                raise
            print(
                f"[worker {worker_idx + 1}/{self.max_workers} gpu={gpu}] "
                f"finished {trial_idx + 1}/{len(self.trials)} {trial['trial_name']} "
                f"returncode={result['returncode']}",
                flush=True,
            )
            return result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pending_trial_idx = 0
            future_to_worker: Dict[Any, int] = {}

            def submit_next(worker_idx: int) -> bool:
                nonlocal pending_trial_idx
                if pending_trial_idx >= len(self.trials):
                    return False
                trial_idx = pending_trial_idx
                pending_trial_idx += 1
                if self.max_workers == 1:
                    gpu = self.gpu_groups[trial_idx % len(self.gpu_groups)]
                else:
                    gpu = worker_gpus[worker_idx]
                trial = self.trials[trial_idx]
                _, nproc = self.command_builder(trial, gpu)
                eta_tracker.register_start(worker_idx, trial_idx, trial, gpu, nproc)
                try:
                    future = executor.submit(run_one, worker_idx, gpu, trial_idx, trial)
                except BaseException:
                    eta_tracker.register_error(worker_idx)
                    raise
                future_to_worker[future] = worker_idx
                return True

            for worker_idx in range(self.max_workers):
                submit_next(worker_idx)

            finished = 0
            show_eta = self.eta_interval > 0.0
            eta_tracker.emit(
                self.trials[pending_trial_idx:],
                finished,
                len(self.trials),
                force=True,
                print_output=show_eta,
            )
            while future_to_worker:
                done, _ = wait(
                    list(future_to_worker.keys()),
                    timeout=float(self.eta_interval or 30.0),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if show_eta:
                        eta_tracker.emit(
                            self.trials[pending_trial_idx:],
                            finished,
                            len(self.trials),
                            force=True,
                        )
                    continue
                completed_results = []
                for future in done:
                    worker_idx = future_to_worker.pop(future)
                    result = future.result()
                    rows.append(result)
                    completed_results.append(result)
                    finished += 1
                    print(f"[{finished}/{len(self.trials)}] collected worker result", flush=True)
                    submit_next(worker_idx)
                state = eta_tracker.emit(
                    self.trials[pending_trial_idx:],
                    finished,
                    len(self.trials),
                    force=True,
                    print_output=show_eta,
                )
                if bool(state.get("state_persisted", False)):
                    for result in completed_results:
                        if int(result.get("returncode", -1)) == 0:
                            eta_tracker.cleanup_progress(result)
        rows.sort(key=lambda row: int(row["trial_index"]))
        return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ddp-train":
        ddp_train_main(sys.argv[2:])
    else:
        raise SystemExit("search_scheduler.py is an internal launcher; run a search plan entry instead.")
