#!/usr/bin/env python3

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.tools.search_plans.a_series.artifact_io import (
    artifact_exists,
    resolve_text_artifact,
)
from src.tools.search_plans.a_series.replay_monitoring_gaps import (
    SPLITS,
    _construct_loaders,
    _load_trainable_checkpoint,
    _read_json,
    _require_new_output,
    _source_identity,
    _write_json,
)
from src.utils.dataset_manifest import write_xlsa_dataset_manifest
from src.utils.run_artifacts import write_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay one additional fixed-probe selection seed from an existing "
            "A-series trainable checkpoint without entering the training loop."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--selection-seed", required=True, type=int)
    parser.add_argument(
        "--execution-profile",
        choices=("final_full", "robustness_core"),
        default="final_full",
        help=(
            "Fixed-Probe profile for the replay. final_full is the strict "
            "three-Probe default; robustness_core is only for explicitly "
            "labelled exploratory or smoke runs."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional fixed-probe batch-size override; defaults to the source resolved config.",
    )
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help=(
            "Re-run identity and validity checks for an existing replay without "
            "executing model inference. This is intended for validator-only "
            "repairs when the scientific artifacts already exist."
        ),
    )
    return parser.parse_args()


def _load_cfg(
    source_run,
    output_run,
    selection_seed,
    batch_size,
    execution_profile="final_full",
):
    config_path = source_run / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    if int(selection_seed) < 0:
        raise ValueError("--selection-seed must be non-negative")

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.OUTPUT_DIR = str(output_run)
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.MONITOR.OUTPUT_POLICY = "error_if_exists"
    cfg.MONITOR.PROBE.SELECTION_SEED = int(selection_seed)
    cfg.MONITOR.PROBE.ROBUSTNESS_SELECTION_SEEDS = []
    cfg.MONITOR.PROBE.EXECUTION_PROFILE = str(execution_profile)
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError("--batch-size must be positive")
        cfg.MONITOR.PROBE.BATCH_SIZE = int(batch_size)

    # The replay keeps the source run's full fixed-probe/domain configuration,
    # but disables train/epoch-only monitors because no optimizer step occurs.
    cfg.MONITOR.NUMERICAL_GUARD.ENABLE = False
    cfg.MONITOR.OPTIMIZER_SANITY.ENABLE = False
    cfg.MONITOR.PREDICTION_HEALTH.ENABLE = False
    cfg.MONITOR.CLASS_ERROR.ENABLE = False
    cfg.MONITOR.CALIBRATION.ENABLE = False
    cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.ENABLE = False
    cfg.MONITOR.PROMPT_PARAMETER.ENABLE = False
    cfg.MONITOR.SEMANTIC_TOKEN.ENABLE = False
    cfg.MONITOR.AUXILIARY_LOSS.ENABLE = False
    cfg.freeze()
    return cfg


def _normal_equivalence_checks(output_run, checkpoint_id, required_splits):
    checks = {}
    root = output_run / "diagnostics" / "probe_equivalence"
    for split in required_splits:
        path = root / f"{checkpoint_id}_{split}.json"
        if not path.is_file():
            checks[split] = {
                "path": str(path),
                "valid": False,
                "pass": False,
                "failure_reasons": ["artifact_missing"],
            }
            continue
        payload = _read_json(path)
        metric_pass = float(
            payload.get("metrics", {}).get("affinity_forward_equivalence_pass", 0.0)
        ) == 1.0
        passed = bool(payload.get("valid", False)) and metric_pass
        checks[split] = {
            "path": str(path),
            "valid": bool(payload.get("valid", False)),
            "metric_pass": metric_pass,
            "pass": passed,
            "failure_reasons": [] if passed else ["forward_equivalence_failed"],
        }
    return checks


def _target_relevance_checks(runtime, cfg, required_splits):
    if not bool(cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE):
        return {"requested": False, "pass": True, "splits": {}}
    expected_layers = sorted(int(item) for item in cfg.MONITOR.PROBE.LAYERS)
    split_checks = {}
    for split in required_splits:
        status = runtime.get("target_relevance_by_split", {}).get(split, {})
        observed = sorted(int(item) for item in status.get("observed_layers", []))
        missing = sorted(int(item) for item in status.get("missing_layers", []))
        passed = (
            bool(status.get("valid", False))
            and observed == expected_layers
            and not missing
        )
        split_checks[split] = {
            "valid": bool(status.get("valid", False)),
            "expected_layers": expected_layers,
            "observed_layers": observed,
            "missing_layers": missing,
            "pass": passed,
            "failure_reasons": list(status.get("failure_reasons", [])),
        }
    return {
        "requested": True,
        "pass": all(item["pass"] for item in split_checks.values()),
        "splits": split_checks,
    }


def _module_effect_checks(output_run, checkpoint_id, cfg, required_splits):
    if not bool(cfg.MONITOR.MODULE_EFFECT.ENABLE) or not bool(cfg.MODEL.PROMPT.ENABLE):
        return {
            "requested": bool(cfg.MONITOR.MODULE_EFFECT.ENABLE),
            "applicable": False,
            "pass": True,
            "conditions": {},
        }

    manifest_path = output_run / "diagnostics" / "module_effect_manifest.json"
    if not manifest_path.is_file():
        return {
            "requested": True,
            "applicable": True,
            "pass": False,
            "failure_reasons": ["module_effect_manifest_missing"],
            "conditions": {},
        }
    manifest = _read_json(manifest_path)
    conditions = [
        str(item) for item in manifest.get("conditions", []) if str(item) != "normal"
    ]
    required_conditions = set()
    if bool(cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_ZERO):
        required_conditions.add("deep_prompt_residual_zeroed")
    if bool(cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_SWAP):
        required_conditions.add("deep_prompt_residual_swapped")
    missing_required_conditions = sorted(required_conditions.difference(conditions))
    root = output_run / "diagnostics" / "module_effect" / checkpoint_id
    condition_checks = {}
    for condition in conditions:
        condition_checks[condition] = {}
        for split in required_splits:
            path = root / f"{split}_{condition}_summary.json"
            resolved_path = resolve_text_artifact(path)
            if not artifact_exists(resolved_path):
                condition_checks[condition][split] = {
                    "path": str(path),
                    "valid": False,
                    "pass": False,
                    "failure_reasons": ["artifact_missing"],
                }
                continue
            payload = _read_json(resolved_path)
            passed = bool(payload.get("valid", False))
            condition_checks[condition][split] = {
                "path": str(resolved_path),
                "valid": passed,
                "pass": passed,
                "failure_reasons": list(payload.get("failure_reasons", [])),
            }
    return {
        "requested": True,
        "applicable": True,
        "manifest_path": str(manifest_path),
        "requested_conditions": list(manifest.get("requested_conditions", [])),
        "required_conditions": sorted(required_conditions),
        "missing_required_conditions": missing_required_conditions,
        "not_applicable_conditions": dict(
            manifest.get("not_applicable_conditions", {})
        ),
        "conditions": condition_checks,
        "pass": bool(conditions)
        and not missing_required_conditions
        and all(
            item["pass"]
            for split_checks in condition_checks.values()
            for item in split_checks.values()
        ),
    }


def _validate_replay(output_run, source, cfg, selection_seed):
    diagnostics = output_run / "diagnostics"
    runtime = _read_json(diagnostics / "probe_runtime_summary.json")
    manifest = _read_json(diagnostics / "probe_manifest.json")
    validity = _read_json(diagnostics / "probe_validity.json")
    checkpoint = dict(runtime.get("checkpoint") or {})
    execution_profile = str(
        runtime.get("execution_profile") or "final_full"
    )
    required_splits = list(runtime.get("required_splits") or SPLITS)
    checkpoint_id = str(
        checkpoint.get("checkpoint_id", f"final_epoch_{int(cfg.SOLVER.TOTAL_EPOCH):04d}")
    )
    runtime_probe_loader = dict(runtime.get("probe_loader") or {})
    source_probe_loader = dict(source.get("probe_loader") or {})
    replay_config_batch_size = int(cfg.MONITOR.PROBE.BATCH_SIZE)
    replay_config_path = output_run / "resolved_config.yaml"
    if replay_config_path.is_file():
        replay_config = yaml.safe_load(
            replay_config_path.read_text(encoding="utf-8")
        )
        replay_config_batch_size = int(
            replay_config["MONITOR"]["PROBE"]["BATCH_SIZE"]
        )
    replay_batch_size = int(
        runtime_probe_loader.get(
            "batch_size", replay_config_batch_size
        )
    )
    source_batch_size = int(
        source_probe_loader.get(
            "batch_size", int(cfg.MONITOR.PROBE.BATCH_SIZE)
        )
    )
    checkpoint_checks = {
        "diagnostic_replay": checkpoint.get("diagnostic_replay") is True,
        "checkpoint_sha256_match": checkpoint.get("checkpoint_sha256")
        == source["checkpoint"]["checkpoint_sha256"],
        "source_run_id_match": checkpoint.get("source_run_id")
        == source["checkpoint"]["source_run_id"],
        "source_session_id_match": checkpoint.get("source_session_id")
        == source["checkpoint"]["source_session_id"],
        "probe_batch_size_match": replay_batch_size == source_batch_size,
    }
    seed_checks = {
        "runtime": int(runtime.get("selection_seed", -1)) == int(selection_seed),
        "manifest": int(manifest.get("selection_seed", -1)) == int(selection_seed),
        "validity": int(validity.get("selection_seed", -1)) == int(selection_seed),
    }
    manifest_checks = {}
    for split in required_splits:
        probe = manifest.get("probes", {}).get(split, {})
        split_validity = manifest.get("validity", {}).get(split, {})
        passed = (
            int(probe.get("selection_seed", -1)) == int(selection_seed)
            and bool(probe.get("manifest_sha256"))
            and bool(split_validity.get("valid", False))
        )
        manifest_checks[split] = {
            "selection_seed": probe.get("selection_seed"),
            "manifest_sha256": probe.get("manifest_sha256"),
            "sample_count": probe.get("selected_sample_count"),
            "valid": bool(split_validity.get("valid", False)),
            "pass": passed,
            "failure_reasons": list(split_validity.get("failure_reasons", [])),
        }

    normal_equivalence = _normal_equivalence_checks(
        output_run, checkpoint_id, required_splits
    )
    target_relevance = _target_relevance_checks(runtime, cfg, required_splits)
    module_effect = _module_effect_checks(
        output_run, checkpoint_id, cfg, required_splits
    )
    semantic_pass = (
        execution_profile != "final_full"
        or not bool(cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE)
        or runtime.get("semantic_intervention_pass") is True
    )
    expected_splits = (
        list(SPLITS)
        if execution_profile == "final_full"
        else ["probe_test_unseen"]
    )
    overall_valid = (
        runtime.get("status") == "completed"
        and execution_profile == str(cfg.MONITOR.PROBE.EXECUTION_PROFILE)
        and required_splits == expected_splits
        and int(runtime.get("probe_count", 0)) == len(required_splits)
        and int(runtime.get("metric_row_count", 0)) > 0
        and bool(validity.get("valid", False))
        and all(checkpoint_checks.values())
        and all(seed_checks.values())
        and all(item["pass"] for item in manifest_checks.values())
        and all(item["pass"] for item in normal_equivalence.values())
        and bool(target_relevance["pass"])
        and bool(module_effect["pass"])
        and semantic_pass
    )
    return {
        "format": "a_series_probe_robustness_replay_v1",
        "status": "valid" if overall_valid else "invalid",
        "selection_seed": int(selection_seed),
        "execution_profile": execution_profile,
        "required_splits": required_splits,
        "source": source,
        "replay_output": str(output_run),
        "runtime_status": runtime.get("status"),
        "metric_row_count": int(runtime.get("metric_row_count", 0)),
        "checkpoint_checks": checkpoint_checks,
        "probe_loader_identity": {
            "source_batch_size": source_batch_size,
            "replay_batch_size": replay_batch_size,
            "match": replay_batch_size == source_batch_size,
        },
        "selection_seed_checks": seed_checks,
        "probe_manifest_checks": manifest_checks,
        "normal_forward_equivalence_checks": normal_equivalence,
        "target_relevance_checks": target_relevance,
        "module_effect_checks": module_effect,
        "semantic_intervention_pass": semantic_pass,
        "valid": overall_valid,
        "training_performed": False,
        "optimizer_created": False,
    }


def main():
    args = parse_args()
    if args.cpu_threads is not None:
        if int(args.cpu_threads) < 1:
            raise ValueError("--cpu-threads must be positive")
        torch.set_num_threads(int(args.cpu_threads))
        torch.set_num_interop_threads(max(1, min(2, int(args.cpu_threads))))
    source_run = Path(args.source_run)
    output_run = Path(args.output_run)
    if not source_run.is_dir():
        raise FileNotFoundError(str(source_run))
    cfg = _load_cfg(
        source_run,
        output_run,
        selection_seed=int(args.selection_seed),
        batch_size=args.batch_size,
        execution_profile=args.execution_profile,
    )
    source = _source_identity(source_run, cfg)
    if int(args.selection_seed) == int(source["selection_seed"]):
        raise ValueError(
            "probe robustness replay requires a selection seed different from the source primary seed"
        )

    replay_summary_path = output_run / "probe_robustness_replay_summary.json"
    if args.validate_existing:
        if not output_run.is_dir():
            raise FileNotFoundError(str(output_run))
        existing = (
            _read_json(replay_summary_path)
            if replay_summary_path.is_file()
            else {}
        )
        replay_summary = _validate_replay(
            output_run, source, cfg, int(args.selection_seed)
        )
        for key in ("cpu_threads", "fixed_probe_result"):
            if key in existing:
                replay_summary[key] = existing[key]
        _write_json(replay_summary_path, replay_summary)
        if not replay_summary["valid"]:
            raise RuntimeError(
                "existing probe robustness replay failed identity or validity gates"
            )
        print(f"Validated existing probe robustness replay: {output_run}")
        return

    _require_new_output(source_run, output_run)

    output_run.mkdir(parents=True, exist_ok=True)
    logging.setup_logging(
        int(cfg.NUM_GPUS), int(cfg.NUM_SHARDS), str(output_run), color=False
    )
    logger = logging.get_logger("visual_prompt")
    torch.manual_seed(int(cfg.SEED))
    np.random.seed(int(cfg.SEED))
    random.seed(int(cfg.SEED))
    write_resolved_config(cfg)

    trainer = None
    status = "failed"
    try:
        train_loader, test_seen_loader, test_unseen_loader = _construct_loaders(
            cfg, logger
        )
        write_xlsa_dataset_manifest(
            cfg,
            {
                "train": train_loader.dataset,
                "test_seen": test_seen_loader.dataset,
                "test_unseen": test_unseen_loader.dataset,
            },
        )
        model, device = build_model(cfg)
        if hasattr(model, "attach_r_similarity_head"):
            model.attach_r_similarity_head(train_loader.dataset.class_attributes)
        trainer = Trainer(cfg, model, Evaluator(task_type="gzsl"), device)
        checkpoint_path = Path(source["checkpoint"]["checkpoint_path"])
        checkpoint_payload = _load_trainable_checkpoint(trainer, checkpoint_path)
        checkpoint_epoch = int(source["checkpoint"]["checkpoint_epoch"])
        if int(checkpoint_payload.get("total_epoch", checkpoint_epoch)) != checkpoint_epoch:
            raise RuntimeError("source checkpoint epoch disagrees with fixed-probe manifest")
        trainer._trace_global_step = int(source["checkpoint"]["checkpoint_global_step"])
        trainer._final_trainable_checkpoint_path = str(checkpoint_path)
        trainer._fixed_probe_checkpoint_source = dict(source["checkpoint"])
        result = trainer._run_fixed_probes(
            train_loader,
            test_seen_loader,
            test_unseen_loader,
            checkpoint_epoch=checkpoint_epoch,
            selection_seed=int(args.selection_seed),
        )
        replay_summary = _validate_replay(
            output_run, source, cfg, int(args.selection_seed)
        )
        replay_summary["cpu_threads"] = {
            "intra_op": int(torch.get_num_threads()),
            "inter_op": int(torch.get_num_interop_threads()),
        }
        replay_summary["fixed_probe_result"] = result
        _write_json(
            output_run / "probe_robustness_replay_summary.json", replay_summary
        )
        if not replay_summary["valid"]:
            raise RuntimeError(
                "probe robustness replay completed but failed identity or validity gates"
            )
        status = "completed"
        logger.info("Probe robustness replay completed and validated: %s", output_run)
    except Exception:
        _write_json(
            output_run / "probe_robustness_replay_failure.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise
    finally:
        if trainer is not None:
            trainer.diagnostic_manager.finalize(status=status)
            trainer.monitor_manager.finalize(status=status)


if __name__ == "__main__":
    main()
