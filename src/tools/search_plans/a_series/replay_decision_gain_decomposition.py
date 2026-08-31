#!/usr/bin/env python3

import argparse
import hashlib
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fvcore.common.checkpoint import Checkpointer

from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.models.build_model import build_model
from src.monitoring.decision_gain_decomposition import analyze_decision_gain
from src.monitoring.module_effect import checkpoint_sha256
from src.tools.search_plans.common import (
    read_json as _read_json,
    write_json as _write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Checkpoint-only full-test decision gain decomposition for one paired training seed."
    )
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-name", default="A0")
    parser.add_argument("--target-name", default="A2")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _sha256_lines(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_output(output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to mix decision decomposition into non-empty output: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_cfg(
    run_dir,
    batch_size,
    num_workers,
    monitor_config_name="DECISION_GAIN_DECOMPOSITION",
):
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    cfg = get_cfg()
    cfg.merge_from_file(str(path))
    cfg.defrost()
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.DATA.NUM_WORKERS = max(0, int(num_workers))
    cfg.DATA.PIN_MEMORY = bool(torch.cuda.is_available())
    monitor_cfg = getattr(cfg.MONITOR, str(monitor_config_name))
    requested_batch = (
        int(batch_size)
        if batch_size is not None
        else int(monitor_cfg.BATCH_SIZE)
    )
    cfg.DATA.BATCH_SIZE = max(1, requested_batch)
    cfg.freeze()
    if str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() != "final_gzsl":
        raise ValueError("decision gain decomposition requires final_gzsl")
    if bool(cfg.MODEL.SEMANTIC_TOKENS.ENABLE):
        raise ValueError("current decision decomposition replay does not support semantic-token inputs")
    return cfg


def _dataset_identity(run_dir):
    path = run_dir / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    payload = _read_json(path)
    result = {"schema_version": payload.get("schema_version"), "splits": {}}
    for split in ("test_seen", "test_unseen"):
        item = payload.get("datasets", {}).get(split)
        if not item:
            raise ValueError("dataset manifest is missing {}".format(split))
        result["splits"][split] = {
            "dataset_name": item.get("dataset_name"),
            "protocol_mode": item.get("protocol_mode"),
            "image_count": int(item.get("image_count", 0)),
            "image_records_sha256": item.get("image_records_sha256"),
            "eval_local_class_ids": [int(value) for value in item.get("eval_local_class_ids", [])],
            "seen_class_ids": [int(value) for value in item.get("seen_class_ids", [])],
            "unseen_class_ids": [int(value) for value in item.get("unseen_class_ids", [])],
            "class_attributes_sha256": (item.get("class_attributes") or {}).get("sha256"),
        }
    return result


def _source_identity(run_dir, cfg):
    checkpoint_path = run_dir / str(cfg.SOLVER.TRAINABLE_FINAL_CHECKPOINT_NAME)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    runtime_path = run_dir / "monitor_runtime_summary.json"
    runtime = _read_json(runtime_path) if runtime_path.is_file() else {}
    if runtime and runtime.get("status") != "completed":
        raise ValueError("source run monitor status is not completed: {}".format(runtime_path))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("format") != "vpt_trainable_v1":
        raise ValueError("unsupported trainable checkpoint format: {}".format(checkpoint_path))
    checkpoint_seed = int(checkpoint.get("seed", cfg.SEED))
    if checkpoint_seed != int(cfg.SEED):
        raise ValueError(
            "checkpoint seed does not match resolved config: {} != {}".format(
                checkpoint_seed, int(cfg.SEED)
            )
        )
    checkpoint_protocol = str(
        checkpoint.get("protocol_mode", cfg.DATA.XLSA.PROTOCOL_MODE)
    )
    if checkpoint_protocol != str(cfg.DATA.XLSA.PROTOCOL_MODE):
        raise ValueError(
            "checkpoint protocol does not match resolved config: {} != {}".format(
                checkpoint_protocol, cfg.DATA.XLSA.PROTOCOL_MODE
            )
        )
    checkpoint_total_epoch = int(
        checkpoint.get("total_epoch", cfg.SOLVER.TOTAL_EPOCH)
    )
    if checkpoint_total_epoch != int(cfg.SOLVER.TOTAL_EPOCH):
        raise ValueError(
            "checkpoint total_epoch does not match resolved config: {} != {}".format(
                checkpoint_total_epoch, int(cfg.SOLVER.TOTAL_EPOCH)
            )
        )
    return {
        "run_dir": str(run_dir),
        "seed": int(cfg.SEED),
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(str(checkpoint_path)),
        "checkpoint_format": checkpoint.get("format"),
        "checkpoint_total_epoch": checkpoint_total_epoch,
        "checkpoint_protocol_mode": checkpoint_protocol,
        "dataset": _dataset_identity(run_dir),
    }, checkpoint


def _load_model_and_loaders(run_dir, cfg, checkpoint):
    model, device = build_model(cfg)
    if str(cfg.MODEL.WEIGHT_PATH):
        checkpointer = Checkpointer(model)
        checkpointables = [
            key
            for key in checkpointer.checkpointables
            if key not in ["head.last_layer.bias", "head.last_layer.weight"]
        ]
        checkpointer.load(str(cfg.MODEL.WEIGHT_PATH), checkpointables)
    train_loader = data_loader.construct_trainval_loader(cfg)
    if hasattr(model, "attach_r_similarity_head"):
        model.attach_r_similarity_head(train_loader.dataset.class_attributes)
    incompatible = model.load_state_dict(checkpoint.get("model_state", {}), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("unexpected trainable checkpoint keys: {}".format(",".join(incompatible.unexpected_keys[:20])))
    model.eval()
    loaders = {
        "test_seen": data_loader.construct_test_seen_loader(cfg),
        "test_unseen": data_loader.construct_test_unseen_loader(cfg),
    }
    return model, device, loaders


@torch.no_grad()
def _predict_split(model, device, loader):
    dataset = loader.dataset
    candidate_class_ids = [int(item) for item in dataset.eval_local_classes]
    eval_map = torch.as_tensor(dataset.eval_global_to_local, dtype=torch.long)
    logits_batches = []
    targets_local = []
    targets_global = []
    sample_ids = []
    for batch in loader:
        inputs = batch["image"].float().to(device, non_blocking=True)
        logits = model(
            inputs,
            semantics=None,
            class_ids=candidate_class_ids,
            runtime_targets=None,
        )
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits["logits"]
        labels = torch.as_tensor(batch["label"], dtype=torch.long)
        local = eval_map.index_select(0, labels)
        if (local < 0).any():
            raise ValueError("full-test target is outside eval class space")
        logits_batches.append(logits.detach().to(device="cpu", dtype=torch.float32).numpy())
        targets_local.extend(int(item) for item in local.tolist())
        targets_global.extend(int(item) for item in labels.tolist())
        sample_ids.extend(str(item) for item in batch["sample_id"])
        if hasattr(model, "clear_runtime_state"):
            model.clear_runtime_state()
    return {
        "logits": np.concatenate(logits_batches, axis=0),
        "targets_local": np.asarray(targets_local, dtype=np.int64),
        "targets_global": np.asarray(targets_global, dtype=np.int64),
        "sample_ids": sample_ids,
        "sample_order_sha256": _sha256_lines(sample_ids),
        "candidate_class_ids": candidate_class_ids,
        "seen_class_ids": [int(item) for item in dataset.seen_classes],
        "unseen_class_ids": [int(item) for item in dataset.unseen_classes],
    }


def _run_source(
    run_dir,
    batch_size,
    num_workers,
    monitor_config_name="DECISION_GAIN_DECOMPOSITION",
):
    cfg = _load_cfg(run_dir, batch_size, num_workers, monitor_config_name)
    identity, checkpoint = _source_identity(run_dir, cfg)
    model, device, loaders = _load_model_and_loaders(run_dir, cfg, checkpoint)
    outputs = {split: _predict_split(model, device, loader) for split, loader in loaders.items()}
    del model, loaders, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cfg, identity, outputs


def _pair_checks(reference_cfg, target_cfg, reference_identity, target_identity, reference, target):
    checks = {
        "training_seed_match": int(reference_cfg.SEED) == int(target_cfg.SEED),
        "checkpoint_seed_match": reference_identity["checkpoint_seed"] == target_identity["checkpoint_seed"],
        "dataset_manifest_match": reference_identity["dataset"] == target_identity["dataset"],
        "protocol_match": str(reference_cfg.DATA.XLSA.PROTOCOL_MODE) == str(target_cfg.DATA.XLSA.PROTOCOL_MODE),
        "classifier_score_mode_match": str(reference_cfg.MODEL.R_SIMILARITY.SCORE_MODE) == str(target_cfg.MODEL.R_SIMILARITY.SCORE_MODE),
        "candidate_space_match": all(reference[split]["candidate_class_ids"] == target[split]["candidate_class_ids"] for split in reference),
        "sample_order_match": all(reference[split]["sample_ids"] == target[split]["sample_ids"] for split in reference),
        "target_identity_match": all(np.array_equal(reference[split]["targets_global"], target[split]["targets_global"]) for split in reference),
        "full_test_count_match": all(len(reference[split]["sample_ids"]) == reference_identity["dataset"]["splits"][split]["image_count"] for split in reference),
    }
    return {**checks, "pass": all(checks.values())}


def main():
    args = parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    reference_run = Path(args.reference_run).resolve()
    target_run = Path(args.target_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not reference_run.is_dir() or not target_run.is_dir():
        raise FileNotFoundError("reference and target run directories must exist")
    _require_output(output_dir)
    try:
        reference_cfg, reference_identity, reference = _run_source(
            reference_run, args.batch_size, args.num_workers
        )
        target_cfg, target_identity, target = _run_source(
            target_run, args.batch_size, args.num_workers
        )
        pair_checks = _pair_checks(
            reference_cfg, target_cfg, reference_identity, target_identity, reference, target
        )
        if not pair_checks["pass"]:
            raise RuntimeError("paired decision decomposition identity checks failed: {}".format(pair_checks))
        candidate_class_ids = reference["test_seen"]["candidate_class_ids"]
        seen_class_ids = reference["test_seen"]["seen_class_ids"]
        unseen_class_ids = reference["test_seen"]["unseen_class_ids"]
        monitor_cfg = reference_cfg.MONITOR.DECISION_GAIN_DECOMPOSITION
        summary, arrays = analyze_decision_gain(
            reference,
            target,
            candidate_class_ids,
            seen_class_ids,
            unseen_class_ids,
            eps=float(monitor_cfg.FACTORIZATION_EPS),
            reconstruction_atol=float(monitor_cfg.RECONSTRUCTION_ATOL),
            global_scale_factors=list(monitor_cfg.GLOBAL_SCALE_FACTORS),
        )
        manifest = {
            "format": "decision_gain_decomposition_manifest_v1",
            "status": "completed" if summary["validity"]["valid"] else "invalid",
            "channel": "cross_run_checkpoint_only_full_test_replay",
            "reference_name": str(args.reference_name),
            "target_name": str(args.target_name),
            "reference": reference_identity,
            "target": target_identity,
            "pair_checks": pair_checks,
            "candidate_class_ids": candidate_class_ids,
            "seen_class_ids": seen_class_ids,
            "unseen_class_ids": unseen_class_ids,
            "sample_order_sha256_by_split": {
                split: reference[split]["sample_order_sha256"] for split in reference
            },
            "runtime": {
                "batch_size": int(reference_cfg.DATA.BATCH_SIZE),
                "num_workers": int(reference_cfg.DATA.NUM_WORKERS),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "optimizer_created": False,
                "parameter_update": False,
                "storage_mode": "aggregate_only",
                "sample_logits_persisted": False,
                "sample_logits_retained_on_cpu_until_pair_aggregation": True,
            },
        }
        summary["pair"] = {
            "reference_name": str(args.reference_name),
            "target_name": str(args.target_name),
            "training_seed": int(reference_cfg.SEED),
        }
        summary["pair_checks"] = pair_checks
        _write_json(output_dir / "decision_gain_decomposition_manifest.json", manifest)
        _write_json(output_dir / "decision_gain_decomposition_summary.json", summary)
        _write_json(output_dir / "decision_gain_decomposition_validity.json", summary["validity"])
        np.savez_compressed(output_dir / "decision_gain_decomposition_per_class.npz", **arrays)
        if not summary["validity"]["valid"]:
            raise RuntimeError("decision gain decomposition failed validity gates")
        print("Decision gain decomposition completed: seed={} output={}".format(reference_cfg.SEED, output_dir))
    except Exception:
        _write_json(
            output_dir / "decision_gain_decomposition_failure.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
