#!/usr/bin/env python3

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.monitoring.module_effect import checkpoint_sha256
from src.tools.search_plans.a_series.artifact_io import (
    artifact_exists,
    read_json_artifact,
    resolve_text_artifact,
)
from src.utils.dataset_manifest import write_xlsa_dataset_manifest
from src.utils.run_artifacts import write_resolved_config


SPLITS = ("probe_train_seen", "probe_test_seen", "probe_test_unseen")
SELECTION_CONDITIONS = (
    "prompt_patch_selection_uniform",
    "patch_prompt_selection_uniform",
    "attribute_concept_prompt_patch_blocked",
    "attribute_concept_random_patch_blocked",
    "transport_targeted_prompt_patch_blocked",
    "transport_random_patch_blocked",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay selected A-series fixed-probe diagnostics or the Prompt object map from a saved checkpoint."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument(
        "--scope", choices=("gaps", "object_map", "full"), default="gaps"
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        help=(
            "Optional Probe selection seed. The source seed is reused when "
            "omitted; object-map strict three-Probe replays pass it explicitly."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Optional fixed-Probe batch-size override; defaults to the source resolved config."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Limit PyTorch CPU threads for replay-only diagnostics.",
    )
    return parser.parse_args()


def _read_json(path):
    return read_json_artifact(Path(path))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _require_new_output(source_run, output_run):
    source_run = source_run.resolve()
    output_run = output_run.resolve()
    if source_run == output_run:
        raise ValueError("--output-run must differ from --source-run")
    if output_run.exists() and any(output_run.iterdir()):
        raise FileExistsError(f"refusing to mix replay artifacts into non-empty output: {output_run}")


def _configure_gap_scope(cfg):
    cfg.MONITOR.PROBE.ROBUSTNESS_SELECTION_SEEDS = []
    cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE = True
    cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE = False
    cfg.MONITOR.PROBE.PROMPT_ANALYSIS.ENABLE = False
    cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION.ENABLE = False
    cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE = False

    module_cfg = cfg.MONITOR.MODULE_EFFECT
    module_cfg.ENABLE = bool(cfg.MODEL.PROMPT.ENABLE)
    module_cfg.PROMPT_ZERO = False
    module_cfg.PROMPT_READ_BLOCK = False
    module_cfg.PROMPT_WRITE_BLOCK = False
    module_cfg.PROMPT_SELECTION_UNIFORM = bool(cfg.MODEL.PROMPT.ENABLE)
    module_cfg.PATCH_PROMPT_SELECTION_UNIFORM = bool(cfg.MODEL.PROMPT.ENABLE)
    module_cfg.PROMPT_VALUE_GLOBALIZE = False
    module_cfg.LAYERWISE_PROMPT_READ_BLOCK = False
    module_cfg.PROMPT_CONTEXT_SWAP = False
    module_cfg.INSTANCE_PROMPT_ZERO = False
    module_cfg.DOMAIN_PROMPT_ZERO = False
    module_cfg.BOTH_PROMPT_ZERO = False
    module_cfg.INSTANCE_PROMPT_SWAP = False
    module_cfg.PROMPT_VALUE_ZERO = False
    module_cfg.ATTRIBUTE_CONCEPT_PATCH_BLOCK = bool(cfg.MODEL.PROMPT.ENABLE)
    module_cfg.TRANSPORT_PATCH_BLOCK = bool(cfg.MODEL.PROMPT.ENABLE)
    module_cfg.ATTENTION_MEDIATION_GAMMA_ZERO = False


def _configure_object_map_scope(cfg):
    cfg.MONITOR.PROBE.EXECUTION_PROFILE = "final_full"
    cfg.MONITOR.PROBE.ROBUSTNESS_SELECTION_SEEDS = []
    cfg.MONITOR.PROBE.ATTENTION_ENABLE = False
    cfg.MONITOR.PROBE.AFFINITY_ENABLE = False
    cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_ENABLE = False
    cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.ENABLE = False
    cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE = False
    cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE = False
    cfg.MONITOR.PROBE.PROMPT_ANALYSIS.ENABLE = False
    cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE = False
    object_cfg = cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION
    object_cfg.ENABLE = bool(cfg.MODEL.PROMPT.ENABLE)
    object_cfg.CANDIDATE_SPACES = [
        "injected_prompt",
        "contextualized_prompt",
        "cls_effect",
        "logit_effect",
    ]
    object_cfg.AUXILIARY_VIEWS = ["decision_margin_effect"]
    if not list(object_cfg.LAYERS):
        object_cfg.LAYERS = list(cfg.MONITOR.PROBE.LAYERS)
    cfg.MONITOR.MODULE_EFFECT.ENABLE = False


def _load_cfg(
    source_run,
    output_run,
    scope,
    batch_size,
    selection_seed=None,
):
    config_path = source_run / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.OUTPUT_DIR = str(output_run)
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.MONITOR.OUTPUT_POLICY = "error_if_exists"
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError("--batch-size must be positive")
        cfg.MONITOR.PROBE.BATCH_SIZE = int(batch_size)
    cfg.MONITOR.PROBE.ROBUSTNESS_SELECTION_SEEDS = []
    if selection_seed is not None:
        if int(selection_seed) < 0:
            raise ValueError("--selection-seed must be non-negative")
        cfg.MONITOR.PROBE.SELECTION_SEED = int(selection_seed)
    cfg.MONITOR.NUMERICAL_GUARD.ENABLE = False
    cfg.MONITOR.OPTIMIZER_SANITY.ENABLE = False
    cfg.MONITOR.PREDICTION_HEALTH.ENABLE = False
    cfg.MONITOR.CLASS_ERROR.ENABLE = False
    cfg.MONITOR.CALIBRATION.ENABLE = False
    cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.ENABLE = False
    cfg.MONITOR.PROMPT_PARAMETER.ENABLE = False
    cfg.MONITOR.SEMANTIC_TOKEN.ENABLE = False
    cfg.MONITOR.AUXILIARY_LOSS.ENABLE = False
    if scope == "gaps":
        _configure_gap_scope(cfg)
    elif scope == "object_map":
        _configure_object_map_scope(cfg)
    cfg.freeze()
    return cfg


def _construct_loaders(cfg, logger):
    protocol_mode = str(cfg.DATA.XLSA.PROTOCOL_MODE).lower()
    if protocol_mode not in {"final_gzsl", "b3_pseudo_gzsl"}:
        raise ValueError(
            "fixed-Probe replay requires final_gzsl or b3_pseudo_gzsl"
        )
    logger.info(
        "Loading train/test datasets for fixed-probe replay protocol=%s",
        protocol_mode,
    )
    train_loader = data_loader.construct_trainval_loader(cfg)
    test_seen_loader = data_loader.construct_test_seen_loader(cfg)
    test_unseen_loader = data_loader.construct_test_unseen_loader(cfg)
    return train_loader, test_seen_loader, test_unseen_loader


def _source_identity(source_run, cfg):
    runtime_path = source_run / "diagnostics" / "probe_runtime_summary.json"
    manifest_path = source_run / "diagnostics" / "probe_manifest.json"
    if not runtime_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("source run must contain completed fixed-probe runtime and manifest JSON")
    runtime = _read_json(runtime_path)
    manifest = _read_json(manifest_path)
    if runtime.get("status") != "completed":
        raise ValueError(f"source fixed probe is not completed: {runtime_path}")
    checkpoint = dict(runtime.get("checkpoint") or {})
    checkpoint_path = source_run / str(cfg.SOLVER.TRAINABLE_FINAL_CHECKPOINT_NAME)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    actual_sha256 = checkpoint_sha256(str(checkpoint_path))
    expected_sha256 = checkpoint.get("checkpoint_sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"source checkpoint hash mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    source_hashes = {
        split: payload["manifest_sha256"]
        for split, payload in manifest.get("probes", {}).items()
    }
    missing = sorted(set(SPLITS).difference(source_hashes))
    if missing:
        raise ValueError(f"source probe manifest is missing splits: {missing}")
    probe_loader = dict(runtime.get("probe_loader") or {})
    probe_loader.setdefault("batch_size", int(cfg.MONITOR.PROBE.BATCH_SIZE))
    return {
        "runtime_path": str(runtime_path),
        "manifest_path": str(manifest_path),
        "selection_seed": int(runtime["selection_seed"]),
        "probe_loader": probe_loader,
        "probe_manifest_sha256_by_split": source_hashes,
        "checkpoint": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": actual_sha256,
            "checkpoint_global_step": int(checkpoint.get("checkpoint_global_step", 0)),
            "checkpoint_selection_rule": checkpoint.get(
                "checkpoint_selection_rule", "predeclared_final_epoch"
            ),
            "source_run_id": checkpoint.get("source_run_id", runtime.get("run_id")),
            "source_session_id": checkpoint.get(
                "source_session_id", runtime.get("session_id")
            ),
            "checkpoint_epoch": int(
                checkpoint.get("checkpoint_epoch", cfg.SOLVER.TOTAL_EPOCH)
            ),
        },
    }


def _load_trainable_checkpoint(trainer, checkpoint_path):
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if payload.get("format") != "vpt_trainable_v1":
        raise ValueError(f"unsupported trainable checkpoint format: {checkpoint_path}")
    state = payload.get("model_state", {})
    incompatible = trainer._model_ref(trainer.model).load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "unexpected trainable checkpoint keys: "
            + ",".join(incompatible.unexpected_keys[:20])
        )
    return payload


def _validate_replay(output_run, source, cfg):
    runtime = _read_json(output_run / "diagnostics" / "probe_runtime_summary.json")
    manifest = _read_json(output_run / "diagnostics" / "probe_manifest.json")
    actual_hashes = {
        split: payload["manifest_sha256"]
        for split, payload in manifest.get("probes", {}).items()
    }
    checkpoint = runtime.get("checkpoint", {})
    checkpoint_checks = {
        "diagnostic_replay": checkpoint.get("diagnostic_replay") is True,
        "checkpoint_sha256_match": checkpoint.get("checkpoint_sha256")
        == source["checkpoint"]["checkpoint_sha256"],
        "source_run_id_match": checkpoint.get("source_run_id")
        == source["checkpoint"]["source_run_id"],
        "source_session_id_match": checkpoint.get("source_session_id")
        == source["checkpoint"]["source_session_id"],
    }
    manifest_checks = {
        split: {
            "source_sha256": source["probe_manifest_sha256_by_split"][split],
            "replay_sha256": actual_hashes.get(split),
            "match": actual_hashes.get(split)
            == source["probe_manifest_sha256_by_split"][split],
        }
        for split in SPLITS
    }
    expected_layers = sorted(int(item) for item in cfg.MONITOR.PROBE.LAYERS)
    target_checks = {}
    for split in SPLITS:
        status = runtime.get("target_relevance_by_split", {}).get(split, {})
        observed = sorted(int(item) for item in status.get("observed_layers", []))
        missing = sorted(int(item) for item in status.get("missing_layers", []))
        target_checks[split] = {
            "valid": bool(status.get("valid", False)),
            "expected_layers": expected_layers,
            "observed_layers": observed,
            "missing_layers": missing,
            "pass": bool(status.get("valid", False))
            and observed == expected_layers
            and not missing,
            "failure_reasons": list(status.get("failure_reasons", [])),
        }

    module_checks = {}
    if bool(cfg.MODEL.PROMPT.ENABLE):
        checkpoint_id = checkpoint.get(
            "checkpoint_id", f"final_epoch_{int(cfg.SOLVER.TOTAL_EPOCH):04d}"
        )
        module_root = output_run / "diagnostics" / "module_effect" / checkpoint_id
        for condition in SELECTION_CONDITIONS:
            module_checks[condition] = {}
            for split in SPLITS:
                path = module_root / f"{split}_{condition}_summary.json"
                resolved_path = resolve_text_artifact(path)
                if not artifact_exists(resolved_path):
                    module_checks[condition][split] = {
                        "path": str(path),
                        "pass": False,
                        "failure_reasons": ["artifact_missing"],
                    }
                    continue
                payload = _read_json(resolved_path)
                chain = payload.get("diagnostic_chain", {})
                mass = chain.get("selection_mass_preservation", {})
                contract = (
                    chain.get("concept_intervention_contract")
                    or chain.get("transport_intervention_contract")
                    or {}
                )
                contract_pass = all(
                    bool(contract.get(key, False))
                    for key in (
                        "algorithm_mass_pass",
                        "selected_path_removed_pass",
                        "targeted_selection_pass",
                    )
                ) if contract else True
                passed = bool(payload.get("valid", False)) and bool(
                    mass.get("pass", False)
                ) and bool(mass.get("applied_pass", False)) and contract_pass
                module_checks[condition][split] = {
                    "path": str(resolved_path),
                    "valid": bool(payload.get("valid", False)),
                    "mass_preservation": mass,
                    "intervention_contract": contract,
                    "pass": passed,
                    "failure_reasons": list(payload.get("failure_reasons", [])),
                }

    overall_valid = (
        runtime.get("status") == "completed"
        and all(checkpoint_checks.values())
        and all(item["match"] for item in manifest_checks.values())
        and all(item["pass"] for item in target_checks.values())
        and all(
            item["pass"]
            for condition in module_checks.values()
            for item in condition.values()
        )
    )
    return {
        "format": "a_series_monitor_gap_replay_v1",
        "status": "valid" if overall_valid else "invalid",
        "scope": "target_relevance_and_selection_intervention_contracts",
        "source": source,
        "replay_output": str(output_run),
        "runtime_status": runtime.get("status"),
        "checkpoint_checks": checkpoint_checks,
        "probe_manifest_checks": manifest_checks,
        "target_relevance_checks": target_checks,
        "selection_intervention_checks": module_checks,
        "valid": overall_valid,
    }


def _validate_object_map_replay(
    output_run,
    source,
    cfg,
    selection_seed,
):
    runtime = _read_json(output_run / "diagnostics" / "probe_runtime_summary.json")
    manifest = _read_json(output_run / "diagnostics" / "probe_manifest.json")
    actual_hashes = {
        split: payload["manifest_sha256"]
        for split, payload in manifest.get("probes", {}).items()
    }
    checkpoint = runtime.get("checkpoint", {})
    checkpoint_checks = {
        "diagnostic_replay": checkpoint.get("diagnostic_replay") is True,
        "checkpoint_sha256_match": checkpoint.get("checkpoint_sha256")
        == source["checkpoint"]["checkpoint_sha256"],
        "source_run_id_match": checkpoint.get("source_run_id")
        == source["checkpoint"]["source_run_id"],
        "source_session_id_match": checkpoint.get("source_session_id")
        == source["checkpoint"]["source_session_id"],
    }
    manifest_checks = {}
    for split in SPLITS:
        probe = manifest.get("probes", {}).get(split, {})
        split_validity = manifest.get("validity", {}).get(split, {})
        passed = bool(
            int(probe.get("selection_seed", -1)) == int(selection_seed)
            and actual_hashes.get(split)
            and split_validity.get("valid", False)
        )
        manifest_checks[split] = {
            "selection_seed": probe.get("selection_seed"),
            "manifest_sha256": actual_hashes.get(split),
            "valid": bool(split_validity.get("valid", False)),
            "pass": passed,
        }
    selected_layers = sorted(
        int(item) for item in cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION.LAYERS
    )
    checkpoint_id = checkpoint.get(
        "checkpoint_id", f"final_epoch_{int(cfg.SOLVER.TOTAL_EPOCH):04d}"
    )
    split_checks = {}
    runtime_statuses = runtime.get("bayesian_object_selection_by_split", {})
    for split in SPLITS:
        status = runtime_statuses.get(split, {})
        root = (
            output_run
            / "diagnostics"
            / "bayesian_object_selection"
            / str(checkpoint_id)
        )
        hierarchy_path = root / f"{split}_hierarchy_trace.json"
        report_path = root / f"{split}_candidate_report.json"
        failure_reasons = []
        hierarchy = report = None
        if hierarchy_path.is_file():
            hierarchy = _read_json(hierarchy_path)
        else:
            failure_reasons.append("hierarchy_trace_missing")
        if report_path.is_file():
            report = _read_json(report_path)
        else:
            failure_reasons.append("candidate_report_missing")
        layer_map = (
            hierarchy.get("layer_interface_map", {})
            if isinstance(hierarchy, dict)
            else {}
        )
        if bool(cfg.MODEL.PROMPT.DEEP):
            missing_layers = [
                layer_id
                for layer_id in selected_layers
                if not bool(layer_map.get(f"layer_{layer_id}", {}).get("valid", False))
            ]
        else:
            missing_layers = []
        if missing_layers:
            failure_reasons.append(
                "invalid_or_missing_layer_interface_map:"
                + ",".join(str(item) for item in missing_layers)
            )
        role_output_pass = bool(
            isinstance(report, dict)
            and "recommended_stochastic_root" in report
            and "recommended_transfer_space" in report
            and report.get("predictive_validation_space") == "logit_effect"
        )
        if not role_output_pass:
            failure_reasons.append("role_separated_object_report_invalid")
        split_pass = bool(
            status.get("observed", False)
            and status.get("valid") is True
            and not failure_reasons
        )
        split_checks[split] = {
            "runtime_status": status,
            "selected_layers": selected_layers,
            "observed_layer_keys": sorted(layer_map),
            "role_output_pass": role_output_pass,
            "pass": split_pass,
            "failure_reasons": failure_reasons,
        }
    overall_valid = bool(
        runtime.get("status") == "completed"
        and int(runtime.get("selection_seed", -1)) == int(selection_seed)
        and all(checkpoint_checks.values())
        and all(item["pass"] for item in manifest_checks.values())
        and all(item["pass"] for item in split_checks.values())
    )
    return {
        "format": "a_series_bayesian_object_map_replay_v1",
        "status": "valid" if overall_valid else "invalid",
        "selection_seed": int(selection_seed),
        "scope": "layerwise_prompt_functional_interface_map",
        "training_performed": False,
        "optimizer_step_performed": False,
        "source": source,
        "replay_output": str(output_run),
        "runtime_status": runtime.get("status"),
        "checkpoint_checks": checkpoint_checks,
        "probe_manifest_checks": manifest_checks,
        "object_map_checks": split_checks,
        "valid": overall_valid,
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
    _require_new_output(source_run, output_run)
    cfg = _load_cfg(
        source_run,
        output_run,
        args.scope,
        args.batch_size,
        selection_seed=args.selection_seed,
    )
    source = _source_identity(source_run, cfg)
    selection_seed = (
        int(args.selection_seed)
        if args.selection_seed is not None
        else int(source["selection_seed"])
    )
    if args.scope != "object_map" and selection_seed != int(source["selection_seed"]):
        raise ValueError(
            "a non-source --selection-seed is currently supported only for object_map; "
            "use replay_probe_robustness.py for a complete fixed-Probe replay"
        )

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
        trainer._trace_global_step = int(
            source["checkpoint"]["checkpoint_global_step"]
        )
        trainer._final_trainable_checkpoint_path = str(checkpoint_path)
        trainer._fixed_probe_checkpoint_source = dict(source["checkpoint"])
        result = trainer._run_fixed_probes(
            train_loader,
            test_seen_loader,
            test_unseen_loader,
            checkpoint_epoch=checkpoint_epoch,
            selection_seed=selection_seed,
        )
        replay_summary = (
            _validate_object_map_replay(
                output_run,
                source,
                cfg,
                selection_seed,
            )
            if args.scope == "object_map"
            else _validate_replay(output_run, source, cfg)
        )
        replay_summary["fixed_probe_result"] = result
        replay_summary["cpu_threads"] = {
            "intra_op": int(torch.get_num_threads()),
            "inter_op": int(torch.get_num_interop_threads()),
        }
        summary_name = (
            "bayesian_object_map_replay_summary.json"
            if args.scope == "object_map"
            else "monitor_gap_replay_summary.json"
        )
        _write_json(output_run / summary_name, replay_summary)
        if not replay_summary["valid"]:
            raise RuntimeError(
                "monitoring replay completed but failed its identity or validity gates"
            )
        status = "completed"
        logger.info("Monitoring gap replay completed and validated: %s", output_run)
    except Exception:
        _write_json(
            output_run / "monitor_gap_replay_failure.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise
    finally:
        if trainer is not None:
            trainer.diagnostic_manager.finalize(status=status)
            trainer.monitor_manager.finalize(status=status)


if __name__ == "__main__":
    main()
