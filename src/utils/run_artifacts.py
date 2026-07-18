#!/usr/bin/env python3

import json
import os
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional

import torch
import torch.distributed as dist
from torch.utils.data import SequentialSampler

from .distributed import get_rank, get_world_size
from .reproducibility import seed_streams


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary_path, path)


def _atomic_json_dump(path: str, payload: Dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_resolved_config(cfg) -> Optional[str]:
    if get_rank() != 0:
        return None
    config_text = cfg.dump()
    path = os.path.join(str(cfg.OUTPUT_DIR), "resolved_config.yaml")
    _atomic_write_text(path, config_text)
    return path


def write_trainable_parameter_manifest(cfg, model) -> Optional[str]:
    if get_rank() != 0:
        return None

    model_ref = model.module if hasattr(model, "module") else model
    tensors = [
        {
            "name": str(name),
            "shape": [int(dim) for dim in parameter.shape],
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model_ref.named_parameters()
        if parameter.requires_grad
    ]
    total_parameters = int(sum(parameter.numel() for parameter in model_ref.parameters()))
    trainable_parameters = int(sum(item["numel"] for item in tensors))
    payload = {
        "schema_version": "trainable_parameter_manifest_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": "{}.{}".format(model_ref.__class__.__module__, model_ref.__class__.__name__),
        "classifier": str(cfg.MODEL.CLASSIFIER),
        "total_parameters": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "trainable_tensors": tensors,
    }
    path = os.path.join(str(cfg.OUTPUT_DIR), "trainable_parameters.json")
    _atomic_json_dump(path, payload)
    return path


def _loader_record(loader, include_batch_size: bool = False) -> Optional[Dict[str, object]]:
    if loader is None:
        return None
    sampler = getattr(loader, "sampler", None)
    record = {
        "sampler": sampler.__class__.__name__ if sampler is not None else None,
        "drop_last": bool(loader.drop_last),
    }
    if include_batch_size:
        record["batch_size"] = int(loader.batch_size) if loader.batch_size is not None else None
    return record


def _module_fingerprint(modules: Mapping[str, torch.nn.Module]) -> Dict[str, float]:
    """Build a low-cost, rank-comparable fingerprint without serializing tensors."""
    parameter_count = 0
    value_sum = 0.0
    square_sum = 0.0
    abs_max = 0.0
    for module in modules.values():
        model_ref = module.module if hasattr(module, "module") else module
        for parameter in model_ref.parameters():
            value = parameter.detach()
            parameter_count += int(value.numel())
            if value.numel() == 0:
                continue
            value_float = value.float()
            value_sum += float(value_float.sum().item())
            square_sum += float(value_float.square().sum().item())
            abs_max = max(abs_max, float(value_float.abs().max().item()))
    return {
        "parameter_count": float(parameter_count),
        "sum": value_sum,
        "square_sum": square_sum,
        "abs_max": abs_max,
    }


def _all_gather_tensor(value: torch.Tensor) -> list:
    if get_world_size() <= 1:
        return [value.detach().cpu()]
    gathered = [torch.empty_like(value) for _ in range(get_world_size())]
    dist.all_gather(gathered, value)
    return [item.detach().cpu() for item in gathered]


def collect_distributed_runtime_checks(
    model,
    cls_criterion,
    train_loader,
    rank_runtime_seed: Optional[int],
    evaluation_loaders: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Collect small DDP correctness checks collectively on every rank."""
    world_size = get_world_size()
    if world_size <= 1:
        return {}

    fingerprint = _module_fingerprint({"model": model, "cls_criterion": cls_criterion})
    device = next((parameter.device for parameter in model.parameters()), torch.device("cpu"))
    fingerprint_tensor = torch.tensor(
        [
            fingerprint["parameter_count"],
            fingerprint["sum"],
            fingerprint["square_sum"],
            fingerprint["abs_max"],
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered_fingerprints = _all_gather_tensor(fingerprint_tensor)
    first_fingerprint = gathered_fingerprints[0]
    fingerprints_match = all(
        torch.allclose(item, first_fingerprint, rtol=1e-10, atol=1e-10)
        for item in gathered_fingerprints[1:]
    )

    if rank_runtime_seed is None:
        raise ValueError("Distributed execution requires a configured SEED and rank runtime seed.")
    seed_tensor = torch.tensor([int(rank_runtime_seed)], dtype=torch.long, device=device)
    gathered_seeds = _all_gather_tensor(seed_tensor)
    runtime_seeds = [int(item.item()) for item in gathered_seeds]

    sampler = getattr(train_loader, "sampler", None)
    sampler_tensor = torch.tensor(
        [
            int(getattr(sampler, "rank", -1)),
            int(getattr(sampler, "num_replicas", -1)),
            int(getattr(sampler, "seed", -1)),
        ],
        dtype=torch.long,
        device=device,
    )
    gathered_samplers = _all_gather_tensor(sampler_tensor)
    sampler_rows = [tuple(int(value) for value in item.tolist()) for item in gathered_samplers]
    sampler_ranks = sorted(row[0] for row in sampler_rows)
    sampler_replicas = [row[1] for row in sampler_rows]
    sampler_seeds = [row[2] for row in sampler_rows]
    complete_partition = bool(
        sampler is not None
        and not bool(getattr(sampler, "drop_last", True))
        and int(len(train_loader.dataset)) % world_size == 0
        and not bool(train_loader.drop_last)
    )

    active_evaluation_loaders = [
        loader
        for loader in (evaluation_loaders or {}).values()
        if loader is not None
    ]
    local_evaluation_loaders_full_dataset = bool(active_evaluation_loaders) and all(
        isinstance(getattr(loader, "sampler", None), SequentialSampler)
        and not bool(loader.drop_last)
        and int(len(loader.sampler)) == int(len(loader.dataset))
        for loader in active_evaluation_loaders
    )
    evaluation_check_tensor = torch.tensor(
        [int(local_evaluation_loaders_full_dataset)],
        dtype=torch.long,
        device=device,
    )
    gathered_evaluation_checks = _all_gather_tensor(evaluation_check_tensor)

    return {
        "initial_parameters_equal": bool(fingerprints_match),
        "rank_runtime_seeds_unique": len(set(runtime_seeds)) == world_size,
        "train_sampler_partition_valid": bool(
            sampler_ranks == list(range(world_size))
            and all(value == world_size for value in sampler_replicas)
            and len(set(sampler_seeds)) == 1
            and complete_partition
        ),
        "evaluation_loaders_full_dataset": all(
            bool(int(item.item())) for item in gathered_evaluation_checks
        ),
    }


def write_reproducibility_manifest(
    cfg,
    loaders: Mapping[str, object],
    distributed_checks: Optional[Mapping[str, object]] = None,
) -> Optional[str]:
    """Write the one per-run random-state and loader evidence artifact."""
    if get_rank() != 0:
        return None

    world_size = get_world_size()
    distributed_initialized = bool(dist.is_available() and dist.is_initialized())
    distributed_record = {"world_size": int(world_size)}
    if world_size > 1:
        distributed_record["backend"] = str(dist.get_backend()) if distributed_initialized else None

    active_loaders = [loader for loader in loaders.values() if loader is not None]
    loader_record = {
        "num_workers": int(active_loaders[0].num_workers) if active_loaders else None,
    }
    loader_record.update(
        {
            str(name): _loader_record(loader, include_batch_size=str(name) == "train")
            for name, loader in loaders.items()
        }
    )
    payload = {
        "schema_version": "reproducibility_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "ddp" if world_size > 1 else "single_gpu",
        "master_seed": int(cfg.SEED) if cfg.SEED is not None else None,
        "shared_streams": seed_streams(cfg.SEED),
        "distributed": distributed_record,
        "loaders": loader_record,
    }
    if distributed_checks:
        payload["distributed_checks"] = dict(distributed_checks)

    path = os.path.join(str(cfg.OUTPUT_DIR), "reproducibility_manifest.json")
    _atomic_json_dump(path, payload)
    return path
