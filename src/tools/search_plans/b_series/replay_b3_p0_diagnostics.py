#!/usr/bin/env python3
"""Run isolated checkpoint-only P0-2 or shared P0-3/P0-4 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import loader as data_loader
from src.monitoring.eval_metrics import (
    calibration_profile_metrics,
    classification_metrics,
    prediction_health_metrics,
    visual_semantic_alignment_metrics,
)
from src.monitoring.module_effect import checkpoint_sha256, paired_module_effect_metrics
from src.monitoring.p0_diagnostics import (
    array_sha256,
    center_prediction_metrics,
    class_centers,
    predict_unseen_centers,
    relation_validity,
    semantic_support_weights,
    visual_centroid_oracle_logits,
)
from src.tools.search_plans.a_series.replay_decision_gain_decomposition import (
    _dataset_identity,
    _load_cfg as _load_a_cfg,
    _load_model_and_loaders,
    _source_identity,
)
from src.tools.search_plans.b_series.replay_b_series_experiments import (
    TEST_SPLITS,
    _atomic_json,
    _build_loaders,
    _candidate_class_ids,
    _condition_summary,
    _json_safe,
    _load_cfg as _load_b_cfg,
    _model_num_layers,
    _predict,
    _run_condition,
    _source_dataset,
)


SHUFFLE_SEEDS = (26001, 26002, 26003, 26004, 26005)
CENTROID_SHUFFLE_SEEDS = (27001, 27002, 27003, 27004, 27005)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_identity() -> Dict[str, str]:
    relative_paths = (
        "src/models/prompting/prompt_distribution.py",
        "src/monitoring/module_effect.py",
        "src/monitoring/p0_diagnostics.py",
        "src/tools/search_plans/b_series/replay_b_series_experiments.py",
        "src/tools/search_plans/b_series/replay_b3_p0_diagnostics.py",
    )
    return {
        relative: _file_sha256(ROOT / relative) for relative in relative_paths
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("P0-2", "P0-34"), required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--a2-run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("full", "probe"), default="full")
    parser.add_argument("--selection-seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--random-subspace-seed", type=int, default=25001)
    return parser.parse_args()


def _manifest(output: Mapping[str, Any]) -> Dict[str, Any]:
    digest = hashlib.sha256()
    for sample_id, label in zip(output["sample_ids"], output["targets_global"]):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "sample_count": len(output["sample_ids"]),
        "class_count": int(np.unique(output["targets_global"]).size),
    }


def _require_empty(path: Path) -> None:
    if path.exists() and next(path.iterdir(), None) is not None:
        raise FileExistsError("refusing to mix P0 evidence into non-empty output: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)


def _state_static_prompt_contract(
    b3_checkpoint: Mapping[str, Any],
    a2_checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    b3_state = b3_checkpoint.get("model_state", {})
    a2_state = a2_checkpoint.get("model_state", {})
    suffixes = ("prompt_embeddings", "deep_prompt_embeddings")
    b3_keys = sorted(
        key
        for key, value in b3_state.items()
        if torch.is_tensor(value)
        and (key.endswith(suffixes) or ".prompt_proj." in key)
    )
    if not b3_keys:
        raise RuntimeError("no static Prompt tensors were found in the B3 checkpoint")
    missing = [key for key in b3_keys if key not in a2_state]
    mismatched = []
    tensor_rows = []
    for key in b3_keys:
        if key in missing:
            continue
        left = b3_state[key].detach().cpu()
        right = a2_state[key].detach().cpu()
        equal = tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
        if not equal:
            mismatched.append(key)
        tensor_rows.append(
            {
                "name": key,
                "shape": list(left.shape),
                "exact_equal": bool(equal),
                "b3_sha256": array_sha256(left.numpy()),
                "a2_sha256": array_sha256(right.numpy()),
            }
        )
    valid = not missing and not mismatched
    return {
        "tensor_count": len(b3_keys),
        "missing_in_a2": missing,
        "mismatched": mismatched,
        "tensors": tensor_rows,
        "exact_static_prompt_match": bool(valid),
        "valid": bool(valid),
    }


def _a2_source_identity(run_dir: Path, cfg):
    """Accept the historical A2 endpoint only when training completion is proven.

    The original A-series runtime summary is marked ``interrupted`` because its
    post-training Probe failed, although all 15 epochs and the final trainable
    checkpoint were written.  P0 is checkpoint-only, so this local compatibility
    path verifies that exact boundary without weakening the shared loader.
    """
    try:
        identity, checkpoint = _source_identity(run_dir, cfg)
        identity["historical_runtime_status_compatibility_used"] = False
        return identity, checkpoint
    except ValueError as error:
        if "source run monitor status is not completed" not in str(error):
            raise
    runtime_path = run_dir / "monitor_runtime_summary.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    train_epoch = (runtime.get("groups") or {}).get("train_epoch") or {}
    last_epoch = (train_epoch.get("last_observed") or {}).get("epoch")
    checkpoint_path = run_dir / str(cfg.SOLVER.TRAINABLE_FINAL_CHECKPOINT_NAME)
    if (
        runtime.get("status") != "interrupted"
        or int(train_epoch.get("write_count", 0)) != int(cfg.SOLVER.TOTAL_EPOCH)
        or int(last_epoch or -1) != int(cfg.SOLVER.TOTAL_EPOCH)
        or not checkpoint_path.is_file()
    ):
        raise ValueError(
            "historical A2 endpoint lacks complete training evidence: {}".format(
                run_dir
            )
        )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("format") != "vpt_trainable_v1":
        raise ValueError("unsupported A2 trainable checkpoint format")
    checkpoint_seed = int(checkpoint.get("seed", cfg.SEED))
    checkpoint_epoch = int(checkpoint.get("total_epoch", cfg.SOLVER.TOTAL_EPOCH))
    checkpoint_protocol = str(
        checkpoint.get("protocol_mode", cfg.DATA.XLSA.PROTOCOL_MODE)
    )
    if (
        checkpoint_seed != int(cfg.SEED)
        or checkpoint_epoch != int(cfg.SOLVER.TOTAL_EPOCH)
        or checkpoint_protocol != str(cfg.DATA.XLSA.PROTOCOL_MODE)
    ):
        raise ValueError("historical A2 checkpoint identity mismatch")
    return {
        "run_dir": str(run_dir),
        "seed": int(cfg.SEED),
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(str(checkpoint_path)),
        "checkpoint_format": checkpoint.get("format"),
        "checkpoint_total_epoch": checkpoint_epoch,
        "checkpoint_protocol_mode": checkpoint_protocol,
        "dataset": _dataset_identity(run_dir),
        "historical_runtime_status_compatibility_used": True,
        "historical_runtime_status": runtime.get("status"),
        "completed_train_epoch_count": int(train_epoch.get("write_count", 0)),
        "last_completed_train_epoch": int(last_epoch),
        "compatibility_boundary": "training complete; historical post-training Probe status is not reused as P0 evidence",
    }, checkpoint


@torch.no_grad()
def _live_prompt_bases(model) -> tuple[tuple[torch.Tensor, ...], Dict[str, Any]]:
    module = model.module if hasattr(model, "module") else model
    transformer = module.enc.transformer
    residual = transformer.deep_prompt_residual
    num_layers = int(residual.num_layers)
    rows = []
    first = transformer.prompt_proj(transformer.prompt_embeddings).squeeze(0)
    rows.append(first.detach().cpu().float())
    for layer_id in range(1, num_layers):
        rows.append(
            transformer.prompt_proj(transformer.deep_prompt_embeddings[layer_id - 1])
            .detach()
            .cpu()
            .float()
        )
    bases = []
    metadata = []
    for layer_id, prompt in enumerate(rows):
        _, singular_np, vh_np = np.linalg.svd(
            prompt.numpy().astype(np.float64), full_matrices=False
        )
        tolerance = max(prompt.shape) * np.finfo(np.float64).eps * float(
            singular_np.max()
        )
        rank = int(np.sum(singular_np > tolerance))
        if rank <= 0:
            raise RuntimeError("A2 Prompt layer {} has zero rank".format(layer_id))
        basis = torch.from_numpy(vh_np[:rank].astype(np.float32)).contiguous()
        gram_error = float(
            (
                basis @ basis.t()
                - torch.eye(rank, dtype=basis.dtype)
            )
            .abs()
            .max()
            .item()
        )
        bases.append(basis)
        metadata.append(
            {
                "layer_id": layer_id,
                "rank": rank,
                "prompt_shape": list(prompt.shape),
                "basis_sha256": array_sha256(basis.numpy()),
                "orthonormal_error": gram_error,
                "smallest_retained_singular": float(singular_np[rank - 1]),
            }
        )
    return tuple(bases), {
        "layer_count": num_layers,
        "layers": metadata,
        "valid": bool(all(row["orthonormal_error"] <= 1.0e-5 for row in metadata)),
    }


def _random_bases(
    reference: Sequence[torch.Tensor], seed: int
) -> tuple[torch.Tensor, ...]:
    generator = np.random.default_rng(int(seed))
    result = []
    for basis in reference:
        rank, dim = basis.shape
        matrix = generator.standard_normal((dim, rank))
        q, _ = np.linalg.qr(matrix, mode="reduced")
        result.append(torch.from_numpy(q.T.astype(np.float32)))
    return tuple(result)


def _paired_to_zero(
    condition: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> Dict[str, Any]:
    result = {}
    for split in condition:
        left = condition[split]
        right = zero[split]
        result[split] = paired_module_effect_metrics(
            right["logits"],
            left["logits"],
            left["targets_local"],
            left["candidate_class_ids"],
            left["seen_class_ids"],
            normal_features=right["features"],
            intervention_features=left["features"],
        )
    return result


def _projection_contract(outputs: Mapping[str, Mapping[str, Any]]) -> bool:
    for output in outputs.values():
        if not bool(output["intervention_contract"]["pass"]):
            return False
        for layer in output["residual_subspace_summary"]["layers"].values():
            if layer["subspace_projection_reconstruction_error"]["max"] > 1.0e-5:
                return False
            if layer["subspace_projection_basis_orthonormal_error"]["max"] > 1.0e-4:
                return False
    return True


def _prompt_state_delta_summary(
    condition: Mapping[str, Mapping[str, Any]],
    residual_zero: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    report = {}
    for split in condition:
        left = condition[split].get("prompt_slot_states") or {}
        right = residual_zero[split].get("prompt_slot_states") or {}
        if set(left) != set(right):
            raise RuntimeError("Prompt slot state layer identity mismatch")
        split_report = {}
        for layer_id in sorted(left, key=int):
            if set(left[layer_id]) != set(right[layer_id]):
                raise RuntimeError("Prompt slot state field identity mismatch")
            layer_report = {}
            for state_name in sorted(left[layer_id]):
                changed = np.asarray(left[layer_id][state_name], dtype=np.float32)
                zero = np.asarray(right[layer_id][state_name], dtype=np.float32)
                if changed.shape != zero.shape:
                    raise RuntimeError("Prompt slot state tensor shape mismatch")
                delta = changed - zero
                per_sample_rms = np.sqrt(
                    np.mean(np.square(delta.astype(np.float64)), axis=tuple(range(1, delta.ndim)))
                )
                layer_report[state_name] = {
                    "delta_rms_mean": float(per_sample_rms.mean()),
                    "delta_rms_min": float(per_sample_rms.min()),
                    "delta_rms_max": float(per_sample_rms.max()),
                    "exact_zero_sample_ratio": float(np.mean(per_sample_rms == 0.0)),
                    "sample_count": int(per_sample_rms.size),
                    "tensor_shape": list(changed.shape),
                }
            split_report[str(layer_id)] = layer_report
        report[split] = split_report
    return report


def _drop_prompt_states(outputs: Mapping[str, Dict[str, Any]]) -> None:
    for output in outputs.values():
        output.pop("prompt_slot_states", None)


def _run_p02(args, output_dir: Path) -> Dict[str, Any]:
    if not args.a2_run:
        raise ValueError("P0-2 requires --a2-run")
    source_run = Path(args.source_run).expanduser().resolve()
    a2_run = Path(args.a2_run).expanduser().resolve()
    cfg = _load_b_cfg(source_run, args.batch_size, args.num_workers)
    identity, checkpoint = _source_identity(source_run, cfg)
    a2_cfg = _load_a_cfg(a2_run, args.batch_size, args.num_workers)
    a2_identity, a2_checkpoint = _a2_source_identity(a2_run, a2_cfg)
    if int(identity["seed"]) != int(a2_identity["seed"]):
        raise ValueError("B3 and A2 training seeds do not match")
    static_contract = _state_static_prompt_contract(checkpoint, a2_checkpoint)
    if not static_contract["valid"]:
        raise RuntimeError("B3 static Prompt is not exactly the paired A2 Prompt")
    model, device, full_loaders = _load_model_and_loaders(source_run, cfg, checkpoint)
    loaders, probe_manifests = _build_loaders(
        cfg,
        full_loaders,
        args.scope,
        args.selection_seed,
        include_train_seen=True,
    )
    bases, basis_contract = _live_prompt_bases(model)
    if not basis_contract["valid"]:
        raise RuntimeError("A2 Prompt basis contract failed")
    random_bases = _random_bases(bases, args.random_subspace_seed)
    num_layers = _model_num_layers(model)
    collect_states = args.scope == "probe"
    state_kwargs = {
        "collect_prompt_slot_states": collect_states,
        "prompt_state_layers": tuple(range(num_layers)),
    }
    normal = _run_condition(model, device, loaders, **state_kwargs)
    zero = _run_condition(
        model, device, loaders, scales=[0.0] * num_layers, **state_kwargs
    )
    conditions: Dict[str, Any] = {}
    prompt_state_evidence = {
        "full": _prompt_state_delta_summary(normal, zero) if collect_states else {},
        "residual_zero": _prompt_state_delta_summary(zero, zero) if collect_states else {},
    }
    if collect_states:
        _drop_prompt_states(normal)
    condition_specs = (
        ("span_natural", bases, "span", False),
        ("span_norm_matched", bases, "span", True),
        ("orthogonal_natural", bases, "orthogonal", False),
        ("orthogonal_norm_matched", bases, "orthogonal", True),
        ("random_span_natural", random_bases, "span", False),
        ("random_span_norm_matched", random_bases, "span", True),
    )
    raw_outputs = {"full": normal, "residual_zero": zero}
    for name, condition_bases, component, norm_match in condition_specs:
        raw_outputs[name] = _run_condition(
            model,
            device,
            loaders,
            residual_subspace={
                "basis_by_layer": condition_bases,
                "component": component,
                "norm_match": norm_match,
            },
            **state_kwargs,
        )
        prompt_state_evidence[name] = (
            _prompt_state_delta_summary(raw_outputs[name], zero)
            if collect_states
            else {}
        )
        if collect_states:
            _drop_prompt_states(raw_outputs[name])
    if collect_states:
        _drop_prompt_states(zero)
    for name, outputs in raw_outputs.items():
        summary = _condition_summary(
            outputs,
            normal,
            cfg,
            is_normal=name == "full",
        )
        summary["paired_vs_residual_zero"] = _paired_to_zero(outputs, zero)
        summary["manifests"] = {
            split: _manifest(value) for split, value in outputs.items()
        }
        summary["residual_subspace"] = {
            split: value["residual_subspace_summary"]
            for split, value in outputs.items()
        }
        summary["prompt_state_vs_residual_zero"] = prompt_state_evidence[name]
        if name not in {"full", "residual_zero"}:
            summary["projection_contract_pass"] = _projection_contract(outputs)
        conditions[name] = summary
    del model, checkpoint, a2_checkpoint, raw_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    valid = bool(
        static_contract["valid"]
        and basis_contract["valid"]
        and all(bool(item["valid"]) for item in conditions.values())
        and all(
            bool(item.get("projection_contract_pass", True))
            for item in conditions.values()
        )
    )
    return {
        "format": "b3_p0_2_prompt_subspace_consumer_v1",
        "experiment": "P0-2",
        "scope": args.scope,
        "selection_seed": args.selection_seed if args.scope == "probe" else None,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "implementation_sha256": _implementation_identity(),
        "source_identity": identity,
        "a2_identity": a2_identity,
        "static_prompt_contract": static_contract,
        "a2_prompt_basis": basis_contract,
        "random_subspace_seed": int(args.random_subspace_seed),
        "probe_manifests": probe_manifests,
        "conditions": conditions,
        "valid": valid,
    }


@torch.no_grad()
def _predict_a2(model, device, loader) -> Dict[str, Any]:
    dataset = _source_dataset(loader)
    candidate = _candidate_class_ids(dataset)
    eval_map = torch.as_tensor(dataset.eval_global_to_local, dtype=torch.long)
    logits_batches = []
    feature_batches = []
    local_batches = []
    global_batches = []
    sample_ids = []
    semantic = None
    module = model.module if hasattr(model, "module") else model
    model.eval()
    for batch in loader:
        inputs = batch["image"].float().to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.long)
        local = eval_map.index_select(0, labels)
        logits = model(
            inputs,
            semantics=None,
            class_ids=candidate,
            runtime_targets=None,
        )
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits["logits"]
        state = module.get_runtime_classifier_stats()
        features = state.get("visual_input") if isinstance(state, dict) else None
        prototypes = state.get("semantic_repr") if isinstance(state, dict) else None
        if not torch.is_tensor(features) or not torch.is_tensor(prototypes):
            raise RuntimeError("A2 classifier runtime trace is unavailable")
        current_semantic = prototypes.detach().cpu().float().numpy()
        if semantic is None:
            semantic = current_semantic
        elif not np.array_equal(semantic, current_semantic):
            raise RuntimeError("A2 projected semantic prototypes changed between batches")
        logits_batches.append(logits.detach().cpu().float().numpy())
        feature_batches.append(features.detach().cpu().float().numpy())
        local_batches.append(local.numpy())
        global_batches.append(labels.numpy())
        sample_ids.extend(str(value) for value in batch["sample_id"])
        if hasattr(module, "clear_runtime_state"):
            module.clear_runtime_state()
    return {
        "logits": np.concatenate(logits_batches, axis=0),
        "features": np.concatenate(feature_batches, axis=0),
        "targets_local": np.concatenate(local_batches, axis=0),
        "targets_global": np.concatenate(global_batches, axis=0),
        "sample_ids": sample_ids,
        "candidate_class_ids": candidate,
        "seen_class_ids": [int(value) for value in dataset.seen_classes],
        "unseen_class_ids": [int(value) for value in dataset.unseen_classes],
        "semantic_prototypes": semantic,
    }


def _task_summary(logits_by_split, outputs, cfg) -> Dict[str, Any]:
    result = {"splits": {}}
    for split in TEST_SPLITS:
        output = outputs[split]
        logits = np.asarray(logits_by_split[split], dtype=np.float64)
        result["splits"][split] = {
            "classification": classification_metrics(logits, output["targets_local"]),
            "prediction_health": prediction_health_metrics(
                logits,
                output["targets_local"],
                output["candidate_class_ids"],
                output["seen_class_ids"],
            ),
        }
    seen_score = result["splits"]["test_seen"]["classification"]["per_class"]
    unseen_score = result["splits"]["test_unseen"]["classification"]["per_class"]
    harmonic = 0.0 if seen_score + unseen_score <= 0.0 else 2.0 * seen_score * unseen_score / (seen_score + unseen_score)
    seen = outputs["test_seen"]
    unseen = outputs["test_unseen"]
    calibration = calibration_profile_metrics(
        logits_by_split["test_seen"],
        seen["targets_local"],
        logits_by_split["test_unseen"],
        unseen["targets_local"],
        seen["candidate_class_ids"],
        seen["seen_class_ids"],
        list(cfg.MONITOR.CALIBRATION.GAMMA_GRID),
    )
    result["gzsl"] = {
        "seen_per_class_accuracy": float(seen_score),
        "unseen_per_class_accuracy": float(unseen_score),
        "harmonic_mean": float(harmonic),
        "ausuc": float(calibration["summary"]["ausuc"]),
        "raw_to_oracle_gain": float(calibration["summary"]["raw_to_oracle_gain"]),
    }
    result["valid"] = True
    return result


def _paired_conditions(reference_logits, changed_logits, outputs) -> Dict[str, Any]:
    return {
        split: paired_module_effect_metrics(
            reference_logits[split],
            changed_logits[split],
            outputs[split]["targets_local"],
            outputs[split]["candidate_class_ids"],
            outputs[split]["seen_class_ids"],
        )
        for split in TEST_SPLITS
    }


def _global_semantic_matrix(output: Mapping[str, Any], total_classes: int) -> np.ndarray:
    result = np.zeros((int(total_classes), output["semantic_prototypes"].shape[1]), dtype=np.float64)
    for local, global_id in enumerate(output["candidate_class_ids"]):
        result[int(global_id)] = output["semantic_prototypes"][local]
    return result


def _run_p03(outputs, raw_attributes, projected_semantic, shuffle_seeds) -> Dict[str, Any]:
    train = outputs["train_seen"]
    unseen = outputs["test_unseen"]
    seen_ids = train["seen_class_ids"]
    unseen_ids = train["unseen_class_ids"]
    seen_centers, seen_support = class_centers(
        train["features"], train["targets_global"], seen_ids
    )
    true_unseen_centers, unseen_support = class_centers(
        unseen["features"], unseen["targets_global"], unseen_ids
    )
    spaces = {
        "attribute_312": np.asarray(raw_attributes, dtype=np.float64),
        "projected_768": np.asarray(projected_semantic, dtype=np.float64),
    }
    reports = {}
    for space_name, semantic in spaces.items():
        conditions = {}
        for condition_name, method in (
            ("positive_cosine", "positive_cosine"),
            ("semantic_1nn", "nearest_neighbor"),
            ("global_seen_mean", "global_mean"),
        ):
            weights = semantic_support_weights(
                semantic, seen_ids, unseen_ids, method=method
            )
            predicted = predict_unseen_centers(weights, seen_centers)
            conditions[condition_name] = {
                "metrics": center_prediction_metrics(
                    predicted,
                    true_unseen_centers,
                    unseen["features"],
                    unseen["targets_global"],
                    unseen_ids,
                ),
                "support_weight_sha256": array_sha256(weights),
                "predicted_center_sha256": array_sha256(predicted),
                "support_weight_entropy_mean": float(
                    np.mean(-np.sum(weights * np.log(np.maximum(weights, 1.0e-12)), axis=1))
                ),
            }
        shuffled = []
        for seed in shuffle_seeds:
            permutation = np.random.default_rng(int(seed)).permutation(len(seen_ids))
            weights = semantic_support_weights(
                semantic,
                seen_ids,
                unseen_ids,
                method="positive_cosine",
                permutation=permutation,
            )
            predicted = predict_unseen_centers(weights, seen_centers)
            shuffled.append(
                {
                    "seed": int(seed),
                    "permutation_sha256": array_sha256(permutation),
                    "metrics": center_prediction_metrics(
                        predicted,
                        true_unseen_centers,
                        unseen["features"],
                        unseen["targets_global"],
                        unseen_ids,
                    ),
                }
            )
        metric_names = tuple(shuffled[0]["metrics"])
        shuffled_summary = {}
        for metric in metric_names:
            values = [item["metrics"][metric] for item in shuffled]
            if all(isinstance(value, (int, float, np.number)) and not isinstance(value, bool) for value in values):
                shuffled_summary[metric] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
        reports[space_name] = {
            "semantic_dim": int(semantic.shape[1]),
            "semantic_sha256": array_sha256(semantic),
            "relation_validity": relation_validity(semantic),
            "conditions": conditions,
            "class_shuffled_controls": shuffled,
            "class_shuffled_summary": shuffled_summary,
        }
    return {
        "format": "b3_p0_3_seen_to_unseen_semantic_prediction_v1",
        "official_seen_to_unseen": True,
        "pseudo_unseen_used": False,
        "graph_prob_prior_used": False,
        "unseen_used_for_fitting_or_tuning": False,
        "oracle_evaluation_uses_true_unseen_centers": True,
        "seen_class_ids": seen_ids,
        "unseen_class_ids": unseen_ids,
        "seen_center_support": seen_support.tolist(),
        "unseen_center_support": unseen_support.tolist(),
        "spaces": reports,
        "valid": bool(
            all(
                item["relation_validity"]["valid"]
                and all(condition["metrics"]["valid"] for condition in item["conditions"].values())
                for item in reports.values()
            )
        ),
    }


def _run_p04(outputs, cfg, projected_semantic, shuffle_seeds) -> Dict[str, Any]:
    train = outputs["train_seen"]
    seen_ids = train["seen_class_ids"]
    unseen_ids = train["unseen_class_ids"]
    candidates = train["candidate_class_ids"]
    candidate_semantic = np.asarray(projected_semantic, dtype=np.float64)[
        np.asarray(candidates, dtype=np.int64)
    ]
    seen_centers, _ = class_centers(train["features"], train["targets_global"], seen_ids)
    unseen_centers, unseen_support = class_centers(
        outputs["test_unseen"]["features"],
        outputs["test_unseen"]["targets_global"],
        unseen_ids,
    )
    center_map = {
        int(class_id): center for class_id, center in zip(seen_ids, seen_centers)
    }
    center_map.update(
        {int(class_id): center for class_id, center in zip(unseen_ids, unseen_centers)}
    )
    semantic_dot = {split: outputs[split]["logits"] for split in TEST_SPLITS}
    semantic_normalized = candidate_semantic / np.maximum(
        np.linalg.norm(candidate_semantic, axis=1, keepdims=True), 1.0e-8
    )
    semantic_cosine = {}
    recomputed_dot = {}
    for split in TEST_SPLITS:
        features = outputs[split]["features"].astype(np.float64)
        recomputed_dot[split] = features @ candidate_semantic.T
        feature_normalized = features / np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True), 1.0e-8
        )
        semantic_cosine[split] = feature_normalized @ semantic_normalized.T
    oracle_logits = {}
    oracle_contract = {}
    for split in TEST_SPLITS:
        oracle_logits[split], oracle_contract[split] = visual_centroid_oracle_logits(
            outputs[split]["features"],
            outputs[split]["targets_global"],
            candidates,
            center_map,
            loo_classes=unseen_ids if split == "test_unseen" else (),
        )
    shuffled_controls = []
    centers = np.stack([center_map[int(class_id)] for class_id in candidates], axis=0)
    for seed in shuffle_seeds:
        permutation = np.random.default_rng(int(seed)).permutation(len(candidates))
        shuffled_map = {
            int(class_id): centers[int(permutation[index])]
            for index, class_id in enumerate(candidates)
        }
        logits_by_split = {}
        for split in TEST_SPLITS:
            logits_by_split[split], _ = visual_centroid_oracle_logits(
                outputs[split]["features"],
                outputs[split]["targets_global"],
                candidates,
                shuffled_map,
                loo_classes=(),
            )
        shuffled_controls.append(
            {
                "seed": int(seed),
                "permutation_sha256": array_sha256(permutation),
                "task": _task_summary(logits_by_split, outputs, cfg),
            }
        )
    conditions = {
        "current_semantic_dot": _task_summary(semantic_dot, outputs, cfg),
        "semantic_cosine": _task_summary(semantic_cosine, outputs, cfg),
        "loo_visual_centroid_cosine": _task_summary(oracle_logits, outputs, cfg),
    }
    conditions["semantic_cosine"]["paired_vs_current_semantic_dot"] = _paired_conditions(
        semantic_dot, semantic_cosine, outputs
    )
    conditions["loo_visual_centroid_cosine"]["paired_vs_current_semantic_dot"] = _paired_conditions(
        semantic_dot, oracle_logits, outputs
    )
    alignment = {}
    for split in TEST_SPLITS:
        alignment[split] = visual_semantic_alignment_metrics(
            outputs[split]["features"],
            candidate_semantic,
            outputs[split]["targets_local"],
        )
    dot_errors = {
        split: {
            "max_abs_error": float(np.max(np.abs(recomputed_dot[split] - semantic_dot[split]))),
            "relative_max_abs_error": float(
                np.max(np.abs(recomputed_dot[split] - semantic_dot[split]))
                / max(1.0, float(np.max(np.abs(semantic_dot[split]))))
            ),
            "prediction_equivalence": float(
                np.mean(
                    np.argmax(recomputed_dot[split], axis=1)
                    == np.argmax(semantic_dot[split], axis=1)
                )
            ),
        }
        for split in TEST_SPLITS
    }
    valid = bool(
        all(item["valid"] for item in conditions.values())
        and all(item["valid"] for item in oracle_contract.values())
        and all(item["prediction_equivalence"] == 1.0 for item in dot_errors.values())
        and all(item["relative_max_abs_error"] <= 1.0e-4 for item in dot_errors.values())
        and int(unseen_support.min()) >= 2
    )
    return {
        "format": "b3_p0_4_backbone_classifier_ceiling_v1",
        "oracle": True,
        "deployable": False,
        "formal_score": False,
        "conditions": conditions,
        "current_semantic_dot_reconstruction": dot_errors,
        "oracle_contract": oracle_contract,
        "centroid_shuffled_controls": shuffled_controls,
        "visual_semantic_alignment": alignment,
        "unseen_center_minimum_support": int(unseen_support.min()),
        "valid": valid,
    }


def _run_p034(args, output_dir: Path) -> Dict[str, Any]:
    source_run = Path(args.source_run).expanduser().resolve()
    cfg = _load_a_cfg(source_run, args.batch_size, args.num_workers)
    identity, checkpoint = _a2_source_identity(source_run, cfg)
    model, device, full_loaders = _load_model_and_loaders(source_run, cfg, checkpoint)
    loaders, probe_manifests = _build_loaders(
        cfg,
        full_loaders,
        args.scope,
        args.selection_seed,
        include_train_seen=True,
    )
    outputs = {
        split: _predict_a2(model, device, loader) for split, loader in loaders.items()
    }
    source_dataset = _source_dataset(loaders["train_seen"])
    raw_attribute_value = source_dataset.class_attributes
    raw_attributes = (
        raw_attribute_value.detach().cpu().float().numpy()
        if torch.is_tensor(raw_attribute_value)
        else np.asarray(raw_attribute_value, dtype=np.float32)
    )
    total_classes = int(raw_attributes.shape[0])
    projected_semantic = _global_semantic_matrix(outputs["test_seen"], total_classes)
    identity_contract = {
        "manifests": {split: _manifest(output) for split, output in outputs.items()},
        "candidate_order_equal": bool(
            all(
                output["candidate_class_ids"] == outputs["test_seen"]["candidate_class_ids"]
                for output in outputs.values()
            )
        ),
        "semantic_prototype_equal": bool(
            all(
                np.array_equal(
                    output["semantic_prototypes"],
                    outputs["test_seen"]["semantic_prototypes"],
                )
                for output in outputs.values()
            )
        ),
        "raw_attribute_sha256": array_sha256(raw_attributes),
        "projected_semantic_sha256": array_sha256(projected_semantic),
    }
    identity_contract["valid"] = bool(
        identity_contract["candidate_order_equal"]
        and identity_contract["semantic_prototype_equal"]
    )
    p03 = _run_p03(outputs, raw_attributes, projected_semantic, SHUFFLE_SEEDS)
    p04 = _run_p04(outputs, cfg, projected_semantic, CENTROID_SHUFFLE_SEEDS)
    del model, checkpoint, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "format": "b3_p0_34_shared_a2_replay_v1",
        "experiment": "P0-34",
        "scope": args.scope,
        "selection_seed": args.selection_seed if args.scope == "probe" else None,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "implementation_sha256": _implementation_identity(),
        "source_identity": identity,
        "identity_contract": identity_contract,
        "probe_manifests": probe_manifests,
        "P0-3": p03,
        "P0-4": p04,
        "valid": bool(identity_contract["valid"] and p03["valid"] and p04["valid"]),
    }


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _require_empty(output_dir)
    running = {
        "status": "running",
        "experiment": args.experiment,
        "scope": args.scope,
        "selection_seed": args.selection_seed if args.scope == "probe" else None,
    }
    _atomic_json(output_dir / "p0_running.json", running)
    try:
        result = (
            _run_p02(args, output_dir)
            if args.experiment == "P0-2"
            else _run_p034(args, output_dir)
        )
        result_path = output_dir / (
            "P0-2-prompt-subspace.json"
            if args.experiment == "P0-2"
            else "P0-34-semantic-ceiling.json"
        )
        _atomic_json(result_path, _json_safe(result))
        summary = {
            "status": "completed" if result["valid"] else "invalid",
            "valid": bool(result["valid"]),
            "experiment": args.experiment,
            "result_path": result_path.name,
            "training_performed": False,
            "optimizer_created": False,
            "backward_performed": False,
        }
        _atomic_json(output_dir / "p0_summary.json", summary)
        if not result["valid"]:
            raise RuntimeError("P0 result failed its validity contract")
        return 0
    except Exception:
        _atomic_json(
            output_dir / "p0_failure.json",
            {**running, "status": "failed", "error": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
