#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.data.transforms import get_transforms
from src.monitoring.deep_prompt_residual_experiments import (
    DEPTH_GROUPS,
    deterministic_donor_indices,
    layer_scales,
    residual_static_geometry,
    summarize_geometry,
)
from src.monitoring.eval_metrics import (
    calibration_profile_metrics,
    classification_metrics,
    prediction_health_metrics,
)
from src.monitoring.logit_geometry import analyze_logit_geometry
from src.monitoring.module_effect import (
    checkpoint_sha256,
    deep_prompt_residual_layer_scales_intervention,
    deep_prompt_residual_replace_intervention,
    paired_module_effect_metrics,
)
from src.monitoring.probe import (
    FixedProbeDataset,
    build_probe_manifest,
    validate_probe_manifest,
)
from src.tools.search_plans.a_series.replay_decision_gain_decomposition import (
    _load_model_and_loaders,
    _source_identity,
)


SPLITS = ("train_seen", "test_seen", "test_unseen")
TEST_SPLITS = ("test_seen", "test_unseen")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E1 through E4 checkpoint-only residual diagnostics."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", choices=("B1", "B2"), default="B1")
    parser.add_argument(
        "--experiments",
        default="E1,E2,E3,E4",
        help="Comma-separated subset of E1,E2,E3,E4.",
    )
    parser.add_argument("--scope", choices=("full", "probe"), default="probe")
    parser.add_argument("--selection-seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--donor-seed", type=int, default=130363)
    parser.add_argument("--shortlist-layers", default="")
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_cfg(run_dir: Path, batch_size: Optional[int], num_workers: int):
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.DATA.NUM_WORKERS = max(0, int(num_workers))
    cfg.DATA.PIN_MEMORY = bool(torch.cuda.is_available())
    if batch_size is not None:
        cfg.DATA.BATCH_SIZE = max(1, int(batch_size))
    cfg.freeze()
    if str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() != "final_gzsl":
        raise ValueError("B1 follow-up replay requires final_gzsl")
    if bool(cfg.MODEL.SEMANTIC_TOKENS.ENABLE):
        raise ValueError("B-series replay expects the semantic-token-free protocol")
    if not bool(cfg.MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.ENABLE):
        raise ValueError("source run does not contain an active Deep Prompt residual")
    return cfg


def _parse_experiments(raw: str) -> tuple[str, ...]:
    values = tuple(
        value.strip().upper() for value in str(raw).split(",") if value.strip()
    )
    allowed = {"E1", "E2", "E3", "E4"}
    if not values or len(set(values)) != len(values) or any(
        value not in allowed for value in values
    ):
        raise ValueError("--experiments must be a unique subset of E1,E2,E3,E4")
    return values


def _parse_shortlist(raw: str, num_layers: int) -> tuple[int, ...]:
    if not str(raw).strip():
        return ()
    values = tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())
    if len(set(values)) != len(values) or any(
        value < 0 or value >= int(num_layers) for value in values
    ):
        raise ValueError("--shortlist-layers contains duplicates or invalid layer ids")
    return values


def _source_dataset(loader):
    dataset = loader.dataset
    return dataset.source_dataset if isinstance(dataset, FixedProbeDataset) else dataset


def _candidate_class_ids(dataset) -> list[int]:
    return [int(value) for value in dataset.seen_classes] + [
        int(value) for value in dataset.unseen_classes
    ]


def _build_loaders(cfg, full_test_loaders, scope: str, selection_seed: int):
    if scope == "full":
        return {name: full_test_loaders[name] for name in TEST_SPLITS}, {}
    train_loader = data_loader.construct_trainval_loader(cfg)
    source_loaders = {
        "train_seen": train_loader,
        "test_seen": full_test_loaders["test_seen"],
        "test_unseen": full_test_loaders["test_unseen"],
    }
    transform = get_transforms("test_seen", cfg.DATA.CROPSIZE)
    loaders = {}
    manifests = {}
    for split, source_loader in source_loaders.items():
        dataset = source_loader.dataset
        manifest = build_probe_manifest(
            dataset,
            split="probe_{}".format(split),
            per_class=int(cfg.MONITOR.PROBE.PER_CLASS),
            max_samples=int(cfg.MONITOR.PROBE.MAX_SAMPLES),
            selection_seed=int(selection_seed),
            candidate_class_ids=_candidate_class_ids(dataset),
        )
        validity = validate_probe_manifest(
            manifest,
            require_full_class_coverage=bool(
                cfg.MONITOR.PROBE.REQUIRE_FULL_CLASS_COVERAGE
            ),
            require_per_class_quota=bool(
                cfg.MONITOR.PROBE.REQUIRE_PER_CLASS_QUOTA
            ),
            allow_max_samples_truncation=bool(
                cfg.MONITOR.PROBE.ALLOW_MAX_SAMPLES_TRUNCATION
            ),
        )
        if not bool(validity["valid"]):
            raise RuntimeError(
                "invalid {} Probe manifest: {}".format(
                    split, "; ".join(validity["failure_reasons"])
                )
            )
        probe_dataset = FixedProbeDataset(dataset, manifest, transform)
        loaders[split] = torch.utils.data.DataLoader(
            probe_dataset,
            batch_size=max(1, int(cfg.MONITOR.PROBE.BATCH_SIZE)),
            shuffle=False,
            num_workers=max(0, int(cfg.MONITOR.PROBE.NUM_WORKERS)),
            pin_memory=bool(cfg.MONITOR.PROBE.PIN_MEMORY),
            drop_last=False,
        )
        manifests[split] = {"manifest": manifest, "validity": validity}
    return loaders, manifests


def _model_num_layers(model) -> int:
    module = model.module if hasattr(model, "module") else model
    residual = module.enc.transformer.deep_prompt_residual
    if residual is None:
        raise RuntimeError("Deep Prompt residual module was not constructed")
    return int(residual.num_layers)


@torch.no_grad()
def _predict(
    model,
    device,
    loader,
    *,
    scales: Optional[Sequence[float]] = None,
    donor_mean_by_id: Optional[Mapping[str, np.ndarray]] = None,
    collect_geometry: bool = False,
) -> Dict[str, Any]:
    dataset = _source_dataset(loader)
    candidate = _candidate_class_ids(dataset)
    eval_map = torch.as_tensor(dataset.eval_global_to_local, dtype=torch.long)
    logits_batches = []
    feature_batches = []
    mean_batches = []
    target_local_batches = []
    target_global_batches = []
    sample_ids = []
    geometry_batches: Dict[str, list[np.ndarray]] = {}
    layer_ids = None
    applied_layers = set()
    intervention_contract_pass = True
    intervention_failure_reasons = []
    replacement_distances = []
    requested_mode = (
        "mean_replace"
        if donor_mean_by_id is not None
        else "layer_scales"
        if scales is not None
        else "none"
    )
    expected_scales = (
        np.asarray(tuple(float(value) for value in scales), dtype=np.float32)
        if scales is not None
        else None
    )
    model.eval()
    for batch in loader:
        inputs = batch["image"].float().to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.long)
        local = eval_map.index_select(0, labels)
        if (local < 0).any():
            raise ValueError("target is outside the GZSL candidate space")
        identifiers = [str(value) for value in batch["sample_id"]]
        if donor_mean_by_id is not None:
            replacement = torch.as_tensor(
                np.stack([donor_mean_by_id[value] for value in identifiers], axis=0),
                device=device,
                dtype=inputs.dtype,
            )
            intervention = deep_prompt_residual_replace_intervention(
                model, replacement=replacement
            )
        elif scales is not None:
            intervention = deep_prompt_residual_layer_scales_intervention(
                model, scales=scales
            )
        else:
            intervention = nullcontext()
        with intervention:
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
        module = model.module if hasattr(model, "module") else model
        classifier_state = module.get_runtime_classifier_stats()
        distribution_state = module.get_runtime_prompt_distribution_stats()
        trace = module.get_runtime_deep_prompt_residual_trace()
        if not isinstance(classifier_state, dict) or not torch.is_tensor(
            classifier_state.get("visual_input")
        ):
            raise RuntimeError("classifier feature trace is unavailable")
        if not isinstance(distribution_state, dict) or not torch.is_tensor(
            distribution_state.get("mu")
        ):
            raise RuntimeError("Prompt Distributor mean trace is unavailable")
        if len(trace) != _model_num_layers(model):
            raise RuntimeError("Deep Prompt residual trace does not cover all layers")
        for item in trace:
            layer_id = int(item["layer_id"])
            runtime_scale = item.get("runtime_scale")
            if not torch.is_tensor(runtime_scale):
                intervention_contract_pass = False
                intervention_failure_reasons.append(
                    "layer {} has no runtime_scale trace".format(layer_id)
                )
            else:
                runtime_scale_float = runtime_scale.detach().float()
                if not torch.allclose(
                    runtime_scale_float, torch.ones_like(runtime_scale_float)
                ):
                    applied_layers.add(layer_id)
                expected_scale = (
                    float(expected_scales[layer_id])
                    if expected_scales is not None
                    else 1.0
                )
                if not torch.allclose(
                    runtime_scale_float,
                    torch.full_like(runtime_scale_float, expected_scale),
                ):
                    intervention_contract_pass = False
                    intervention_failure_reasons.append(
                        "layer {} runtime_scale does not match {}".format(
                            layer_id, expected_scale
                        )
                    )
            if donor_mean_by_id is not None:
                distance = item.get("mean_replace_distance")
                if not torch.is_tensor(distance):
                    intervention_contract_pass = False
                    intervention_failure_reasons.append(
                        "layer {} has no mean_replace_distance trace".format(layer_id)
                    )
                elif layer_id == 0:
                    replacement_distances.append(
                        distance.detach().cpu().float().numpy()
                    )
        if collect_geometry:
            current = residual_static_geometry(trace)
            if layer_ids is None:
                layer_ids = current.pop("layer_ids")
            else:
                if not np.array_equal(layer_ids, current.pop("layer_ids")):
                    raise RuntimeError("residual geometry layer identity changed")
            for name, values in current.items():
                geometry_batches.setdefault(name, []).append(values)
        logits_batches.append(logits.detach().cpu().float().numpy())
        feature_batches.append(
            classifier_state["visual_input"].detach().cpu().float().numpy()
        )
        mean_batches.append(distribution_state["mu"].detach().cpu().float().numpy())
        target_local_batches.append(local.numpy())
        target_global_batches.append(labels.numpy())
        sample_ids.extend(identifiers)
        if hasattr(module, "clear_runtime_state"):
            module.clear_runtime_state()
        del inputs, logits, classifier_state, distribution_state, trace
    result = {
        "logits": np.concatenate(logits_batches, axis=0),
        "features": np.concatenate(feature_batches, axis=0),
        "source_mu": np.concatenate(mean_batches, axis=0),
        "targets_local": np.concatenate(target_local_batches, axis=0),
        "targets_global": np.concatenate(target_global_batches, axis=0),
        "sample_ids": sample_ids,
        "candidate_class_ids": candidate,
        "seen_class_ids": [int(value) for value in dataset.seen_classes],
        "unseen_class_ids": [int(value) for value in dataset.unseen_classes],
        "applied_layer_ids": sorted(applied_layers),
        "intervention_contract": {
            "requested_mode": requested_mode,
            "pass": bool(intervention_contract_pass),
            "failure_reasons": sorted(set(intervention_failure_reasons)),
            "expected_changed_layer_ids": (
                np.flatnonzero(np.abs(expected_scales - 1.0) > 1.0e-7).tolist()
                if expected_scales is not None
                else []
            ),
            "observed_changed_layer_ids": sorted(applied_layers),
            "mean_replace_executed": bool(
                donor_mean_by_id is not None and replacement_distances
            ),
            "mean_replace_distance_mean": (
                float(np.concatenate(replacement_distances).mean())
                if replacement_distances
                else None
            ),
            "mean_replace_distance_max": (
                float(np.concatenate(replacement_distances).max())
                if replacement_distances
                else None
            ),
        },
    }
    if collect_geometry:
        result["geometry"] = {
            name: np.concatenate(values, axis=0)
            for name, values in geometry_batches.items()
        }
        result["geometry"]["layer_ids"] = layer_ids
    return result


def _same_identity(reference: Mapping[str, Any], changed: Mapping[str, Any]) -> bool:
    return (
        reference["sample_ids"] == changed["sample_ids"]
        and reference["candidate_class_ids"] == changed["candidate_class_ids"]
        and np.array_equal(reference["targets_local"], changed["targets_local"])
    )


def _condition_summary(
    outputs: Mapping[str, Mapping[str, Any]],
    normal: Mapping[str, Mapping[str, Any]],
    cfg,
    *,
    is_normal: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"splits": {}, "valid": True, "failure_reasons": []}
    for split, output in outputs.items():
        identity_ok = _same_identity(normal[split], output)
        if not identity_ok:
            summary["valid"] = False
            summary["failure_reasons"].append("{}: sample identity mismatch".format(split))
        item = {
            "sample_count": len(output["sample_ids"]),
            "classification": classification_metrics(
                output["logits"], output["targets_local"]
            ),
            "prediction_health": prediction_health_metrics(
                output["logits"],
                output["targets_local"],
                output["candidate_class_ids"],
                output["seen_class_ids"],
            ),
            "applied_layer_ids": list(output["applied_layer_ids"]),
            "identity_match": identity_ok,
            "intervention_contract": output["intervention_contract"],
        }
        if not bool(output["intervention_contract"]["pass"]):
            summary["valid"] = False
            summary["failure_reasons"].append(
                "{}: intervention contract failed".format(split)
            )
        if not is_normal:
            item["paired_vs_normal"] = paired_module_effect_metrics(
                normal[split]["logits"],
                output["logits"],
                output["targets_local"],
                output["candidate_class_ids"],
                output["seen_class_ids"],
                normal_features=normal[split]["features"],
                intervention_features=output["features"],
            )
        summary["splits"][split] = item
    if all(split in outputs for split in TEST_SPLITS):
        seen = outputs["test_seen"]
        unseen = outputs["test_unseen"]
        seen_score = summary["splits"]["test_seen"]["classification"]["per_class"]
        unseen_score = summary["splits"]["test_unseen"]["classification"]["per_class"]
        harmonic = (
            0.0
            if seen_score + unseen_score <= 0.0
            else 2.0 * seen_score * unseen_score / (seen_score + unseen_score)
        )
        calibration = calibration_profile_metrics(
            seen["logits"],
            seen["targets_local"],
            unseen["logits"],
            unseen["targets_local"],
            seen["candidate_class_ids"],
            seen["seen_class_ids"],
            list(cfg.MONITOR.CALIBRATION.GAMMA_GRID),
        )
        geometry_summary, _ = analyze_logit_geometry(
            {
                split: {
                    "logits": outputs[split]["logits"],
                    "targets_local": outputs[split]["targets_local"],
                }
                for split in TEST_SPLITS
            },
            seen["candidate_class_ids"],
            seen["seen_class_ids"],
            seen["unseen_class_ids"],
        )
        summary["gzsl"] = {
            "seen_per_class_accuracy": float(seen_score),
            "unseen_per_class_accuracy": float(unseen_score),
            "harmonic_mean": float(harmonic),
            "ausuc": float(calibration["summary"]["ausuc"]),
            "raw_to_oracle_gain": float(
                calibration["summary"]["raw_to_oracle_gain"]
            ),
            "oracle_peak_gamma": float(
                calibration["summary"]["oracle_peak_gamma"]
            ),
        }
        summary["logit_geometry"] = geometry_summary
    return _json_safe(summary)


def _run_condition(model, device, loaders, **kwargs):
    return {
        split: _predict(model, device, loader, **kwargs)
        for split, loader in loaders.items()
    }


def _donor_payload(normal_split: Mapping[str, Any], relation: str, seed: int):
    indices = deterministic_donor_indices(
        normal_split["sample_ids"],
        normal_split["targets_global"],
        relation=relation,
        seed=seed,
    )
    means = normal_split["source_mu"]
    donor_by_id = {
        sample_id: means[int(donor_index)]
        for sample_id, donor_index in zip(normal_split["sample_ids"], indices.tolist())
    }
    manifest = []
    for target_index, donor_index in enumerate(indices.tolist()):
        manifest.append(
            {
                "target_id": normal_split["sample_ids"][target_index],
                "target_class": int(normal_split["targets_global"][target_index]),
                "donor_id": normal_split["sample_ids"][donor_index],
                "donor_class": int(normal_split["targets_global"][donor_index]),
                "self_pair": bool(target_index == donor_index),
                "donor_residual_distance": float(
                    np.linalg.norm(means[donor_index] - means[target_index])
                ),
            }
        )
    return donor_by_id, manifest


def _geometry_report(normal, zero):
    result = {"format": "b_series_e2_residual_static_geometry_v1", "splits": {}}
    arrays = {}
    for split, output in normal.items():
        normal_pred = output["logits"].argmax(axis=1)
        zero_pred = zero[split]["logits"].argmax(axis=1)
        target = output["targets_local"]
        normal_correct = normal_pred == target
        zero_correct = zero_pred == target
        groups = {
            "zero_beneficial_normal_wrong_to_zero_correct": (~normal_correct)
            & zero_correct,
            "zero_harmful_normal_correct_to_zero_wrong": normal_correct
            & (~zero_correct),
            "correctness_unchanged": normal_correct == zero_correct,
        }
        result["splits"][split] = summarize_geometry(
            output["geometry"], groups=groups
        )
        for name, values in output["geometry"].items():
            arrays["{}__{}".format(split, name)] = values
        arrays["{}__targets_local".format(split)] = target
        arrays["{}__normal_correct".format(split)] = normal_correct.astype(np.int8)
        arrays["{}__zero_correct".format(split)] = zero_correct.astype(np.int8)
    return result, arrays


def main() -> None:
    args = _parse_args()
    experiments = _parse_experiments(args.experiments)
    if args.num_workers < 0 or args.selection_seed < 0 or args.donor_seed < 0:
        raise SystemExit("worker and seed values must be non-negative")
    source_run = Path(args.source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(str(source_run))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "refusing to mix B1 follow-up replay into non-empty output: {}".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(source_run, args.batch_size, args.num_workers)
    torch.manual_seed(int(cfg.SEED))
    np.random.seed(int(cfg.SEED))
    random.seed(int(cfg.SEED))
    identity, checkpoint = _source_identity(source_run, cfg)
    model, device, full_test_loaders = _load_model_and_loaders(
        source_run, cfg, checkpoint
    )
    num_layers = _model_num_layers(model)
    shortlist = _parse_shortlist(args.shortlist_layers, num_layers)
    loaders, probe_manifests = _build_loaders(
        cfg, full_test_loaders, args.scope, args.selection_seed
    )
    summary: Dict[str, Any] = {
        "format": "b_series_experiments_checkpoint_replay_v1",
        "suite_name": "B-series",
        "method_name": str(args.method_name),
        "experiments": list(experiments),
        "scope": str(args.scope),
        "selection_seed": int(args.selection_seed) if args.scope == "probe" else None,
        "training_performed": False,
        "optimizer_created": False,
        "source": identity,
        "source_checkpoint_sha256": checkpoint_sha256(
            identity["checkpoint_path"]
        ),
        "probe_manifests": probe_manifests,
        "conditions": {},
        "valid": False,
    }
    _atomic_json(output_dir / "b_series_replay_running.json", _json_safe(summary))
    try:
        normal = _run_condition(
            model, device, loaders, collect_geometry="E2" in experiments
        )
        summary["conditions"]["normal"] = _condition_summary(
            normal, normal, cfg, is_normal=True
        )
        zero_scales = layer_scales(num_layers, selected_scale=0.0)
        zero = None
        if set(experiments).intersection({"E1", "E2", "E3"}):
            zero = _run_condition(model, device, loaders, scales=zero_scales)
            summary["conditions"]["all_residual_zero"] = _condition_summary(
                zero, normal, cfg, is_normal=False
            )

        if "E1" in experiments:
            for alpha in (0.25, 0.5, 0.75):
                name = "E1_alpha_{:.2f}".format(alpha)
                outputs = _run_condition(
                    model,
                    device,
                    loaders,
                    scales=layer_scales(num_layers, selected_scale=alpha),
                )
                summary["conditions"][name] = _condition_summary(
                    outputs, normal, cfg, is_normal=False
                )
            summary["E1"] = {
                "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                "endpoint_identity": {
                    "alpha_0_condition": "all_residual_zero",
                    "alpha_1_condition": "normal",
                },
            }

        if "E2" in experiments:
            if zero is None:
                raise RuntimeError("E2 requires the all-residual-zero endpoint")
            geometry_report, geometry_arrays = _geometry_report(normal, zero)
            _atomic_json(
                output_dir / "E2-residual-static-geometry.json",
                _json_safe(geometry_report),
            )
            np.savez_compressed(
                output_dir / "E2-residual-static-geometry.npz",
                **geometry_arrays,
            )
            summary["E2"] = {
                "summary_path": "E2-residual-static-geometry.json",
                "arrays_path": "E2-residual-static-geometry.npz",
                "group_identity": "correctness transition under Residual-zero versus normal",
            }

        if "E3" in experiments:
            e3_conditions = []
            for group_name, group_layers in DEPTH_GROUPS.items():
                zero_name = "E3_{}_zero".format(group_name)
                only_name = "E3_{}_only".format(group_name)
                group_zero = _run_condition(
                    model,
                    device,
                    loaders,
                    scales=layer_scales(
                        num_layers,
                        selected_layers=group_layers,
                        selected_scale=0.0,
                        other_scale=1.0,
                    ),
                )
                group_only = _run_condition(
                    model,
                    device,
                    loaders,
                    scales=layer_scales(
                        num_layers,
                        selected_layers=group_layers,
                        selected_scale=1.0,
                        other_scale=0.0,
                    ),
                )
                summary["conditions"][zero_name] = _condition_summary(
                    group_zero, normal, cfg, is_normal=False
                )
                summary["conditions"][only_name] = _condition_summary(
                    group_only, normal, cfg, is_normal=False
                )
                e3_conditions.extend([zero_name, only_name])
            for layer_id in shortlist:
                for mode, selected_scale, other_scale in (
                    ("zero", 0.0, 1.0),
                    ("only", 1.0, 0.0),
                ):
                    name = "E3_layer_{}_{}".format(layer_id, mode)
                    outputs = _run_condition(
                        model,
                        device,
                        loaders,
                        scales=layer_scales(
                            num_layers,
                            selected_layers=(layer_id,),
                            selected_scale=selected_scale,
                            other_scale=other_scale,
                        ),
                    )
                    summary["conditions"][name] = _condition_summary(
                        outputs, normal, cfg, is_normal=False
                    )
                    e3_conditions.append(name)
                for alpha in (0.25, 0.5, 0.75):
                    name = "E3_layer_{}_alpha_{:.2f}".format(layer_id, alpha)
                    outputs = _run_condition(
                        model,
                        device,
                        loaders,
                        scales=layer_scales(
                            num_layers,
                            selected_layers=(layer_id,),
                            selected_scale=alpha,
                            other_scale=1.0,
                        ),
                    )
                    summary["conditions"][name] = _condition_summary(
                        outputs, normal, cfg, is_normal=False
                    )
                    e3_conditions.append(name)
            summary["E3"] = {
                "depth_groups": {
                    name: list(values) for name, values in DEPTH_GROUPS.items()
                },
                "shortlist_layers": list(shortlist),
                "condition_names": e3_conditions,
            }

        if "E4" in experiments:
            donor_manifests = {}
            donor_maps_by_relation = {"same_class": {}, "different_class": {}}
            for split, output in normal.items():
                for relation in donor_maps_by_relation:
                    donor_map, manifest = _donor_payload(
                        output, relation, int(args.donor_seed)
                    )
                    donor_maps_by_relation[relation][split] = donor_map
                    donor_manifests.setdefault(split, {})[relation] = manifest
            for relation in ("same_class", "different_class"):
                outputs = {
                    split: _predict(
                        model,
                        device,
                        loader,
                        donor_mean_by_id=donor_maps_by_relation[relation][split],
                    )
                    for split, loader in loaders.items()
                }
                name = "E4_{}_swap".format(relation)
                summary["conditions"][name] = _condition_summary(
                    outputs, normal, cfg, is_normal=False
                )
            _atomic_json(
                output_dir / "E4-donor-manifest.json",
                {
                    "format": "b_series_e4_donor_manifest_v1",
                    "donor_seed": int(args.donor_seed),
                    "splits": donor_manifests,
                },
            )
            summary["E4"] = {
                "donor_manifest_path": "E4-donor-manifest.json",
                "self_pair_allowed": False,
                "same_class_contract": "same label and different sample id",
                "different_class_contract": "different label",
            }

        if args.save_logits:
            arrays = {}
            for split, output in normal.items():
                arrays["normal__{}__logits".format(split)] = output["logits"]
                arrays["normal__{}__targets_local".format(split)] = output[
                    "targets_local"
                ]
            np.savez_compressed(output_dir / "B1-normal-logits.npz", **arrays)
            summary["normal_logits_path"] = "B1-normal-logits.npz"
        summary["valid"] = all(
            bool(value.get("valid", False))
            for value in summary["conditions"].values()
        )
        summary["status"] = "completed" if summary["valid"] else "invalid"
        _atomic_json(output_dir / "b_series_replay_summary.json", _json_safe(summary))
        if not summary["valid"]:
            raise RuntimeError("B1 follow-up replay completed with invalid conditions")
    except Exception:
        _atomic_json(
            output_dir / "b_series_replay_failure.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise
    finally:
        running = output_dir / "b_series_replay_running.json"
        if running.is_file():
            running.unlink()


if __name__ == "__main__":
    main()
