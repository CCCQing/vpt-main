#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import loader as data_loader
from src.models.candidate_compatibility import (
    CandidateConditionedCompatibility,
    GlobalTemperatureCompatibility,
    ImageOnlyTemperatureCompatibility,
    trainable_parameter_count,
)
from src.monitoring.eval_metrics import (
    calibration_profile_metrics,
    classification_metrics,
    prediction_health_metrics,
)
from src.monitoring.logit_geometry import analyze_logit_geometry
from src.monitoring.module_effect import checkpoint_sha256
from src.monitoring.probe import build_probe_manifest, validate_probe_manifest
from src.tools.search_plans.a_series.replay_decision_gain_decomposition import (
    _load_cfg,
    _load_model_and_loaders,
)
from src.tools.search_plans.b_series.replay_b3_p0_diagnostics import (
    _a2_source_identity,
)


SPLITS = ("train_seen", "test_seen", "test_unseen")
TRAINED_VARIANTS = (
    "candidate_raw312",
    "candidate_projected768",
    "semantic_permuted_raw312",
    "image_constant_raw312",
    "image_only_temperature",
    "temperature_only",
)
REFERENCE_VARIANTS = ("current_semantic_dot", "semantic_cosine")
PROBE_SELECTION_SEEDS = (424242, 424243, 424244)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs" / "b_series_experiments" / "P1-3a-head-only.yaml",
    )
    parser.add_argument("--head-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_rows(sample_ids: Sequence[str], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id, label in zip(sample_ids, labels.tolist()):
        digest.update("{}|{}\n".format(sample_id, int(label)).encode("utf-8"))
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _require_empty(path: Path) -> None:
    if path.exists() and next(path.iterdir(), None) is not None:
        raise FileExistsError("refusing to mix P1-3a evidence into {}".format(path))
    path.mkdir(parents=True, exist_ok=True)


def _load_experiment_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("EXPERIMENT")) != "P1-3a":
        raise ValueError("invalid P1-3a experiment config")
    return payload


@torch.no_grad()
def _extract_split(model, device, loader) -> Dict[str, Any]:
    feature_batches = []
    target_global = []
    sample_ids = []
    model.eval()
    for batch in loader:
        inputs = batch["image"].float().to(device, non_blocking=True)
        features, _ = model(inputs, return_feature=True, semantics=None)
        if features.dim() != 2 or not bool(torch.isfinite(features).all().item()):
            raise RuntimeError("A2 final CLS extraction produced invalid features")
        feature_batches.append(features.detach().cpu().float().numpy())
        target_global.extend(int(value) for value in batch["label"])
        sample_ids.extend(str(value) for value in batch["sample_id"])
        if hasattr(model, "clear_runtime_state"):
            model.clear_runtime_state()
        del inputs, features
    features = np.concatenate(feature_batches, axis=0)
    labels = np.asarray(target_global, dtype=np.int64)
    return {
        "features": features,
        "targets_global": labels,
        "sample_ids": sample_ids,
        "manifest_sha256": _sha256_rows(sample_ids, labels),
        "feature_sha256": _sha256_array(features),
        "sample_count": int(features.shape[0]),
    }


def _feature_cache(
    source_run: Path,
    experiment: Mapping[str, Any],
    *,
    batch_size: Optional[int],
    num_workers: Optional[int],
) -> tuple[Dict[str, Any], Dict[str, Any], Any]:
    extraction = experiment["FEATURE_EXTRACTION"]
    cfg = _load_cfg(
        source_run,
        batch_size or int(extraction["BATCH_SIZE"]),
        int(extraction["NUM_WORKERS"] if num_workers is None else num_workers),
    )
    source_identity, checkpoint = _a2_source_identity(source_run, cfg)
    model, device, full_test_loaders = _load_model_and_loaders(
        source_run, cfg, checkpoint
    )
    train_loader = data_loader.construct_train_eval_loader(cfg)
    loaders = {
        "train_seen": train_loader,
        "test_seen": full_test_loaders["test_seen"],
        "test_unseen": full_test_loaders["test_unseen"],
    }
    module = model.module if hasattr(model, "module") else model
    if str(module.r_similarity_head.score_mode).lower() != "dot":
        raise ValueError("P1-3a current reference requires the A2 dot-product head")
    projected = (
        module.r_similarity_head._project_class_prototypes()
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    raw_value = train_loader.dataset.class_attributes
    raw = (
        raw_value.detach().cpu().float().numpy()
        if torch.is_tensor(raw_value)
        else np.asarray(raw_value, dtype=np.float32)
    )
    seen = [int(value) for value in train_loader.dataset.seen_classes]
    unseen = [int(value) for value in train_loader.dataset.unseen_classes]
    candidate = seen + unseen
    outputs = {split: _extract_split(model, device, loader) for split, loader in loaders.items()}
    probe_manifests = {}
    per_class = int(experiment["EVALUATION"]["PROBE_PER_CLASS"])
    max_samples = max(
        int(cfg.MONITOR.PROBE.MAX_SAMPLES),
        per_class * len(candidate),
    )
    for selection_seed in PROBE_SELECTION_SEEDS:
        seed_payload = {}
        for split in ("test_seen", "test_unseen"):
            dataset = loaders[split].dataset
            manifest = build_probe_manifest(
                dataset,
                split="probe_{}".format(split),
                per_class=per_class,
                max_samples=max_samples,
                selection_seed=int(selection_seed),
                candidate_class_ids=candidate,
            )
            validity = validate_probe_manifest(
                manifest,
                require_full_class_coverage=True,
                require_per_class_quota=True,
                allow_max_samples_truncation=False,
            )
            if not bool(validity["valid"]):
                raise RuntimeError("invalid P1-3a Probe manifest for {}".format(split))
            id_to_index = {
                sample_id: index
                for index, sample_id in enumerate(outputs[split]["sample_ids"])
            }
            indices = np.asarray(
                [id_to_index[str(row["sample_id"])] for row in manifest["samples"]],
                dtype=np.int64,
            )
            seed_payload[split] = {
                "manifest": manifest,
                "validity": validity,
                "indices": indices,
            }
        probe_manifests[str(selection_seed)] = seed_payload
    del model, checkpoint, full_test_loaders, train_loader, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cache_manifest = {
        "format": "p13a_frozen_a2_feature_cache_manifest_v1",
        "source": source_identity,
        "candidate_class_ids": candidate,
        "seen_class_ids": seen,
        "unseen_class_ids": unseen,
        "raw_semantic_shape": list(raw.shape),
        "projected_semantic_shape": list(projected.shape),
        "raw_semantic_sha256": _sha256_array(raw),
        "projected_semantic_sha256": _sha256_array(projected),
        "splits": {
            split: {
                key: value
                for key, value in output.items()
                if key not in {"features", "targets_global", "sample_ids"}
            }
            for split, output in outputs.items()
        },
        "feature_extraction_count_per_split": 1,
        "features_persisted_to_disk": False,
        "valid": True,
    }
    cache = {
        "outputs": outputs,
        "raw_semantics": raw[candidate],
        "projected_semantics": projected[candidate],
        "candidate_class_ids": candidate,
        "seen_class_ids": seen,
        "unseen_class_ids": unseen,
        "probe_manifests": probe_manifests,
    }
    return cache, cache_manifest, cfg


def _global_to_local(candidate: Sequence[int]) -> Dict[int, int]:
    return {int(class_id): index for index, class_id in enumerate(candidate)}


def _targets_local(labels: np.ndarray, candidate: Sequence[int]) -> np.ndarray:
    mapping = _global_to_local(candidate)
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def _heldout_classes(
    seen_classes: Sequence[int],
    split_seed: int,
    fraction: float,
) -> tuple[list[int], list[int]]:
    values = np.asarray(sorted(int(value) for value in seen_classes), dtype=np.int64)
    generator = np.random.default_rng(int(split_seed))
    shuffled = generator.permutation(values)
    holdout_count = max(2, int(round(values.size * float(fraction))))
    heldout = sorted(int(value) for value in shuffled[:holdout_count])
    train = sorted(int(value) for value in shuffled[holdout_count:])
    if len(train) < 2:
        raise ValueError("class-disjoint selection left fewer than two training classes")
    return train, heldout


def _model_for_variant(
    variant: str,
    visual_dim: int,
    semantic_dim: int,
    training: Mapping[str, Any],
) -> torch.nn.Module:
    if variant in {
        "candidate_raw312",
        "candidate_projected768",
        "semantic_permuted_raw312",
        "image_constant_raw312",
    }:
        return CandidateConditionedCompatibility(
            visual_dim=visual_dim,
            semantic_dim=semantic_dim,
            hidden_dim=int(training["HIDDEN_DIM"]),
            dropout=float(training["DROPOUT"]),
        )
    if variant == "image_only_temperature":
        return ImageOnlyTemperatureCompatibility(
            visual_dim=visual_dim,
            hidden_dim=int(training["IMAGE_ONLY_HIDDEN_DIM"]),
        )
    if variant == "temperature_only":
        return GlobalTemperatureCompatibility()
    raise ValueError("unknown P1-3a variant {}".format(variant))


def _variant_semantics(
    variant: str,
    cache: Mapping[str, Any],
    permutation: np.ndarray,
) -> np.ndarray:
    if variant == "candidate_projected768":
        return np.asarray(cache["projected_semantics"], dtype=np.float32)
    if variant in {"image_only_temperature", "temperature_only"}:
        return np.asarray(cache["projected_semantics"], dtype=np.float32)
    raw = np.asarray(cache["raw_semantics"], dtype=np.float32)
    return raw[permutation] if variant == "semantic_permuted_raw312" else raw


def _variant_features(
    variant: str,
    features: np.ndarray,
    constant: Optional[np.ndarray],
) -> np.ndarray:
    if variant != "image_constant_raw312":
        return features
    if constant is None:
        raise ValueError("image-constant control requires a fixed feature")
    return np.broadcast_to(constant.reshape(1, -1), features.shape).copy()


def _macro_accuracy(scores: np.ndarray, targets: np.ndarray) -> float:
    predictions = scores.argmax(axis=1)
    values = []
    for class_id in np.unique(targets):
        mask = targets == int(class_id)
        values.append(float(np.mean(predictions[mask] == targets[mask])))
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def _score(
    model: torch.nn.Module,
    features: np.ndarray,
    semantics: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    candidate_chunk_size: int,
) -> np.ndarray:
    model.eval()
    semantic = torch.as_tensor(semantics, dtype=torch.float32, device=device)
    rows = []
    for start in range(0, features.shape[0], int(batch_size)):
        visual = torch.as_tensor(
            features[start : start + int(batch_size)],
            dtype=torch.float32,
            device=device,
        )
        logits = model(
            visual,
            semantic,
            candidate_chunk_size=int(candidate_chunk_size),
        )
        rows.append(logits.detach().cpu().float().numpy())
    return np.concatenate(rows, axis=0)


def _train_epochs(
    model: torch.nn.Module,
    features: np.ndarray,
    labels_global: np.ndarray,
    candidate_classes: Sequence[int],
    all_candidate_classes: Sequence[int],
    semantics: np.ndarray,
    device: torch.device,
    training: Mapping[str, Any],
    *,
    seed: int,
    epochs: int,
    constant: Optional[np.ndarray],
    variant: str,
) -> list[Dict[str, float]]:
    candidate_positions = np.asarray(
        [_global_to_local(all_candidate_classes)[int(value)] for value in candidate_classes],
        dtype=np.int64,
    )
    candidate_set = set(int(value) for value in candidate_classes)
    selected = np.asarray(
        [int(value) in candidate_set for value in labels_global], dtype=bool
    )
    x = _variant_features(variant, features[selected], constant)
    labels = labels_global[selected]
    local_map = _global_to_local(candidate_classes)
    y = np.asarray([local_map[int(value)] for value in labels], dtype=np.int64)
    semantic = torch.as_tensor(
        semantics[candidate_positions], dtype=torch.float32, device=device
    )
    counts = np.bincount(y, minlength=len(candidate_classes)).astype(np.float64)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["LR"]),
        weight_decay=float(training["WEIGHT_DECAY"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order_source = torch.arange(x.shape[0], dtype=torch.long)
    history = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        permutation = order_source[torch.randperm(x.shape[0], generator=generator)]
        loss_sum = 0.0
        correct = 0
        count = 0
        for start in range(0, x.shape[0], int(training["BATCH_SIZE"])):
            indices = permutation[start : start + int(training["BATCH_SIZE"])].numpy()
            visual = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
            targets = torch.as_tensor(y[indices], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                visual,
                semantic,
                candidate_chunk_size=int(training["CANDIDATE_CHUNK_SIZE"]),
            )
            loss = F.cross_entropy(logits, targets, weight=weight_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["GRAD_CLIP_NORM"]))
            optimizer.step()
            batch_count = int(targets.numel())
            loss_sum += float(loss.detach().item()) * batch_count
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            count += batch_count
        history.append(
            {
                "epoch": int(epoch),
                "train_ce": float(loss_sum / max(1, count)),
                "train_sample_accuracy": float(correct / max(1, count)),
            }
        )
    return history


def _selection_phase(
    variant: str,
    cache: Mapping[str, Any],
    semantics: np.ndarray,
    train_classes: Sequence[int],
    heldout_classes: Sequence[int],
    device: torch.device,
    training: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    output = cache["outputs"]["train_seen"]
    features = np.asarray(output["features"], dtype=np.float32)
    labels = np.asarray(output["targets_global"], dtype=np.int64)
    train_mask = np.isin(labels, np.asarray(train_classes, dtype=np.int64))
    constant = features[train_mask].mean(axis=0) if variant == "image_constant_raw312" else None
    _seed_everything(seed)
    model = _model_for_variant(variant, features.shape[1], semantics.shape[1], training).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["LR"]),
        weight_decay=float(training["WEIGHT_DECAY"]),
    )
    train_positions = np.asarray(
        [_global_to_local(cache["candidate_class_ids"])[int(value)] for value in train_classes],
        dtype=np.int64,
    )
    heldout_positions = np.asarray(
        [_global_to_local(cache["candidate_class_ids"])[int(value)] for value in heldout_classes],
        dtype=np.int64,
    )
    train_semantic = torch.as_tensor(
        semantics[train_positions], dtype=torch.float32, device=device
    )
    heldout_semantic = semantics[heldout_positions]
    x_train = _variant_features(variant, features[train_mask], constant)
    local_map = _global_to_local(train_classes)
    y_train = np.asarray([local_map[int(value)] for value in labels[train_mask]], dtype=np.int64)
    heldout_mask = np.isin(labels, np.asarray(heldout_classes, dtype=np.int64))
    x_val = _variant_features(variant, features[heldout_mask], constant)
    val_map = _global_to_local(heldout_classes)
    y_val = np.asarray([val_map[int(value)] for value in labels[heldout_mask]], dtype=np.int64)
    counts = np.bincount(y_train, minlength=len(train_classes)).astype(np.float64)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order_source = torch.arange(x_train.shape[0], dtype=torch.long)
    best = None
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, int(training["MAX_SELECTION_EPOCHS"]) + 1):
        model.train()
        permutation = order_source[torch.randperm(x_train.shape[0], generator=generator)]
        loss_sum = 0.0
        count = 0
        for start in range(0, x_train.shape[0], int(training["BATCH_SIZE"])):
            indices = permutation[start : start + int(training["BATCH_SIZE"])].numpy()
            visual = torch.as_tensor(x_train[indices], dtype=torch.float32, device=device)
            targets = torch.as_tensor(y_train[indices], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                visual,
                train_semantic,
                candidate_chunk_size=int(training["CANDIDATE_CHUNK_SIZE"]),
            )
            loss = F.cross_entropy(logits, targets, weight=weight_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["GRAD_CLIP_NORM"]))
            optimizer.step()
            loss_sum += float(loss.detach().item()) * int(targets.numel())
            count += int(targets.numel())
        val_scores = _score(
            model,
            x_val,
            heldout_semantic,
            device,
            batch_size=int(training["EVAL_BATCH_SIZE"]),
            candidate_chunk_size=int(training["CANDIDATE_CHUNK_SIZE"]),
        )
        macro = _macro_accuracy(val_scores, y_val)
        nll = float(classification_metrics(val_scores, y_val)["nll"])
        row = {
            "epoch": int(epoch),
            "train_ce": float(loss_sum / max(1, count)),
            "heldout_macro_accuracy": macro,
            "heldout_nll": nll,
        }
        history.append(row)
        key = (macro, -nll, -epoch)
        if best is None or key > best:
            best = key
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if (
            epoch >= int(training["MIN_SELECTION_EPOCHS"])
            and stale >= int(training["PATIENCE"])
        ):
            break
    best_row = max(
        history,
        key=lambda row: (
            float(row["heldout_macro_accuracy"]),
            -float(row["heldout_nll"]),
            -int(row["epoch"]),
        ),
    )
    model.load_state_dict(best_state)
    return {
        "selected_epoch": int(best_row["epoch"]),
        "best_heldout_macro_accuracy": float(best_row["heldout_macro_accuracy"]),
        "best_heldout_nll": float(best_row["heldout_nll"]),
        "epochs_executed": int(len(history)),
        "history": history,
        "parameter_count": trainable_parameter_count(model),
        "optimizer_created": True,
        "optimizer_parameter_scope": "candidate_compatibility_head_only",
    }


def _nearest_semantic_negative(raw_semantics: np.ndarray) -> np.ndarray:
    values = raw_semantics / np.maximum(
        np.linalg.norm(raw_semantics, axis=1, keepdims=True), 1.0e-12
    )
    similarity = values @ values.T
    np.fill_diagonal(similarity, -np.inf)
    return similarity.argmax(axis=1).astype(np.int64)


def _hard_negative_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    hard_negative_by_class: np.ndarray,
) -> Dict[str, float]:
    negative = hard_negative_by_class[targets]
    margin = scores[np.arange(targets.size), targets] - scores[
        np.arange(targets.size), negative
    ]
    return {
        "true_vs_semantic_hard_negative_margin_mean": float(margin.mean()),
        "true_above_semantic_hard_negative_rate": float(np.mean(margin > 0.0)),
    }


def _condition_task_metrics(
    scores_by_split: Mapping[str, np.ndarray],
    cache: Mapping[str, Any],
    gamma_grid: Sequence[float],
) -> Dict[str, Any]:
    candidate = cache["candidate_class_ids"]
    seen = cache["seen_class_ids"]
    unseen = cache["unseen_class_ids"]
    hard_negative = _nearest_semantic_negative(cache["raw_semantics"])
    split_metrics = {}
    geometry_inputs = {}
    for split in SPLITS:
        output = cache["outputs"][split]
        targets = _targets_local(output["targets_global"], candidate)
        scores = np.asarray(scores_by_split[split], dtype=np.float32)
        split_metrics[split] = {
            "classification": classification_metrics(scores, targets),
            "prediction_health": prediction_health_metrics(
                scores, targets, candidate, seen
            ),
            "semantic_hard_negative": _hard_negative_metrics(
                scores, targets, hard_negative
            ),
        }
        geometry_inputs[split] = {"logits": scores, "targets_local": targets}
    seen_acc = float(split_metrics["test_seen"]["classification"]["top1"])
    unseen_acc = float(split_metrics["test_unseen"]["classification"]["top1"])
    harmonic = (
        0.0
        if seen_acc + unseen_acc <= 0.0
        else float(2.0 * seen_acc * unseen_acc / (seen_acc + unseen_acc))
    )
    seen_targets = _targets_local(
        cache["outputs"]["test_seen"]["targets_global"], candidate
    )
    unseen_targets = _targets_local(
        cache["outputs"]["test_unseen"]["targets_global"], candidate
    )
    calibration = calibration_profile_metrics(
        scores_by_split["test_seen"],
        seen_targets,
        scores_by_split["test_unseen"],
        unseen_targets,
        candidate,
        seen,
        gamma_grid,
    )
    logit_geometry, _ = analyze_logit_geometry(
        geometry_inputs,
        candidate,
        seen,
        unseen,
    )
    return {
        "splits": split_metrics,
        "normal_gzsl": {
            "seen": seen_acc,
            "unseen": unseen_acc,
            "harmonic_mean": harmonic,
            "ausuc": float(calibration["summary"]["ausuc"]),
            "oracle_peak_gamma": float(
                calibration["summary"]["oracle_peak_gamma"]
            ),
            "raw_to_oracle_h_gain": float(
                calibration["summary"]["raw_to_oracle_gain"]
            ),
        },
        "logit_geometry": logit_geometry,
    }


def _probe_metrics(
    scores_by_split: Mapping[str, np.ndarray],
    cache: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate = cache["candidate_class_ids"]
    seen = cache["seen_class_ids"]
    result = {}
    for selection_seed, split_payload in cache["probe_manifests"].items():
        row = {"splits": {}}
        accuracies = {}
        for split in ("test_seen", "test_unseen"):
            indices = split_payload[split]["indices"]
            targets = _targets_local(
                cache["outputs"][split]["targets_global"][indices], candidate
            )
            scores = scores_by_split[split][indices]
            metrics = classification_metrics(scores, targets)
            row["splits"][split] = {
                "manifest_sha256": split_payload[split]["manifest"][
                    "manifest_sha256"
                ],
                "sample_count": int(indices.size),
                "classification": metrics,
                "prediction_health": prediction_health_metrics(
                    scores, targets, candidate, seen
                ),
            }
            accuracies[split] = float(metrics["top1"])
        denominator = accuracies["test_seen"] + accuracies["test_unseen"]
        row["harmonic_mean"] = (
            0.0
            if denominator <= 0.0
            else float(
                2.0
                * accuracies["test_seen"]
                * accuracies["test_unseen"]
                / denominator
            )
        )
        result[str(selection_seed)] = row
    return result


def _reference_scores(cache: Mapping[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    semantics = np.asarray(cache["projected_semantics"], dtype=np.float32)
    semantic_norm = semantics / np.maximum(
        np.linalg.norm(semantics, axis=1, keepdims=True), 1.0e-12
    )
    dot = {}
    cosine = {}
    for split, output in cache["outputs"].items():
        features = np.asarray(output["features"], dtype=np.float32)
        dot[split] = features @ semantics.T
        feature_norm = features / np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True), 1.0e-12
        )
        cosine[split] = feature_norm @ semantic_norm.T
    return {"current_semantic_dot": dot, "semantic_cosine": cosine}


def _train_final_variant(
    variant: str,
    cache: Mapping[str, Any],
    semantics: np.ndarray,
    selected_epoch: int,
    device: torch.device,
    training: Mapping[str, Any],
    seed: int,
    output_dir: Path,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    train_output = cache["outputs"]["train_seen"]
    features = np.asarray(train_output["features"], dtype=np.float32)
    labels = np.asarray(train_output["targets_global"], dtype=np.int64)
    seen_mask = np.isin(labels, np.asarray(cache["seen_class_ids"], dtype=np.int64))
    constant = features[seen_mask].mean(axis=0) if variant == "image_constant_raw312" else None
    _seed_everything(seed)
    model = _model_for_variant(
        variant, features.shape[1], semantics.shape[1], training
    ).to(device)
    history = _train_epochs(
        model,
        features,
        labels,
        cache["seen_class_ids"],
        cache["candidate_class_ids"],
        semantics,
        device,
        training,
        seed=seed,
        epochs=selected_epoch,
        constant=constant,
        variant=variant,
    )
    scores = {}
    for split, output in cache["outputs"].items():
        values = _variant_features(
            variant,
            np.asarray(output["features"], dtype=np.float32),
            constant,
        )
        scores[split] = _score(
            model,
            values,
            semantics,
            device,
            batch_size=int(training["EVAL_BATCH_SIZE"]),
            candidate_chunk_size=int(training["CANDIDATE_CHUNK_SIZE"]),
        )
    checkpoint_path = output_dir / "heads" / (variant + ".pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "p13a_head_only_v1",
            "variant": variant,
            "head_seed": int(seed),
            "selected_epoch": int(selected_epoch),
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "training_performed": True,
            "backbone_frozen": True,
            "prompt_frozen": True,
            "optimizer_scope": "head_only",
        },
        checkpoint_path,
    )
    metadata = {
        "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
        "checkpoint_sha256": checkpoint_sha256(str(checkpoint_path)),
        "parameter_count": trainable_parameter_count(model),
        "selected_epoch": int(selected_epoch),
        "final_training_history": history,
        "constant_feature_used": variant == "image_constant_raw312",
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, metadata


def main() -> None:
    args = _parse_args()
    started = time.time()
    source_run = args.source_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    experiment_config_path = args.experiment_config.expanduser().resolve()
    _require_empty(output_dir)
    experiment = _load_experiment_config(experiment_config_path)
    _atomic_json(
        output_dir / "execution_state.json",
        {
            "format": "p13a_execution_state_v1",
            "status": "running",
            "head_seed": int(args.head_seed),
            "source_run": str(source_run),
            "training_performed": False,
        },
    )
    cache, cache_manifest, cfg = _feature_cache(
        source_run,
        experiment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _atomic_json(output_dir / "feature_cache_manifest.json", cache_manifest)
    _atomic_json(
        output_dir / "probe_manifests.json",
        {
            "format": "p13a_probe_manifest_collection_v1",
            "selection_seeds": list(PROBE_SELECTION_SEEDS),
            "manifests": {
                seed: {
                    split: {
                        "manifest": item["manifest"],
                        "validity": item["validity"],
                    }
                    for split, item in split_payload.items()
                }
                for seed, split_payload in cache["probe_manifests"].items()
            },
        },
    )
    training = experiment["TRAINING"]
    train_classes, heldout_classes = _heldout_classes(
        cache["seen_class_ids"],
        int(training["HELDOUT_CLASS_SPLIT_SEED"]),
        float(training["HELDOUT_CLASS_FRACTION"]),
    )
    class_split = {
        "format": "p13a_class_disjoint_selection_manifest_v1",
        "split_seed": int(training["HELDOUT_CLASS_SPLIT_SEED"]),
        "train_class_ids": train_classes,
        "heldout_class_ids": heldout_classes,
        "train_class_count": len(train_classes),
        "heldout_class_count": len(heldout_classes),
        "normal_unseen_class_ids_used_for_selection": False,
    }
    _atomic_json(output_dir / "class_disjoint_selection_manifest.json", class_split)
    requested_device = str(args.device).lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P1-3a requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)
    permutation_generator = np.random.default_rng(
        int(training["SEMANTIC_PERMUTATION_SEED"])
    )
    semantic_permutation = permutation_generator.permutation(
        len(cache["candidate_class_ids"])
    )
    references = _reference_scores(cache)
    conditions = {}
    for name, scores in references.items():
        conditions[name] = {
            "identity": {
                "kind": "frozen_reference",
                "training_performed": False,
                "optimizer_created": False,
            },
            "task_metrics": _condition_task_metrics(
                scores, cache, cfg.MONITOR.CALIBRATION.GAMMA_GRID
            ),
            "strict_three_probe": _probe_metrics(scores, cache),
        }
    for variant_index, variant in enumerate(TRAINED_VARIANTS):
        variant_seed = int(args.head_seed) * 100 + 51001 + variant_index
        semantics = _variant_semantics(variant, cache, semantic_permutation)
        selection = _selection_phase(
            variant,
            cache,
            semantics,
            train_classes,
            heldout_classes,
            device,
            training,
            variant_seed,
        )
        scores, final_metadata = _train_final_variant(
            variant,
            cache,
            semantics,
            int(selection["selected_epoch"]),
            device,
            training,
            variant_seed,
            output_dir,
        )
        conditions[variant] = {
            "identity": {
                "kind": "trained_head_only",
                "training_performed": True,
                "optimizer_created": True,
                "a2_backbone_frozen": True,
                "a2_prompt_frozen": True,
                "head_seed": variant_seed,
                "semantic_input": (
                    "projected_768"
                    if variant
                    in {
                        "candidate_projected768",
                        "image_only_temperature",
                        "temperature_only",
                    }
                    else "raw_attribute_312"
                ),
                "semantic_permuted": variant == "semantic_permuted_raw312",
                "image_constant": variant == "image_constant_raw312",
                "candidate_specific_interaction": variant
                in {
                    "candidate_raw312",
                    "candidate_projected768",
                    "semantic_permuted_raw312",
                    "image_constant_raw312",
                },
            },
            "selection": selection,
            "final_training": final_metadata,
            "task_metrics": _condition_task_metrics(
                scores, cache, cfg.MONITOR.CALIBRATION.GAMMA_GRID
            ),
            "strict_three_probe": _probe_metrics(scores, cache),
        }
    result = {
        "format": "p13a_candidate_conditioned_head_only_result_v1",
        "status": "completed",
        "valid": True,
        "experiment": "P1-3a",
        "head_seed": int(args.head_seed),
        "a2_training_seed": int(cache_manifest["source"]["checkpoint_seed"]),
        "source": cache_manifest["source"],
        "implementation": {
            "experiment_config_path": str(experiment_config_path),
            "experiment_config_sha256": _sha256_file(experiment_config_path),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "head_module_sha256": _sha256_file(
                ROOT / "src" / "models" / "candidate_compatibility.py"
            ),
        },
        "execution_contract": {
            "training_performed": True,
            "optimizer_created": True,
            "optimizer_scope": "candidate_compatibility_head_only",
            "a2_backbone_frozen": True,
            "a2_prompt_frozen": True,
            "a2_classifier_frozen_and_used_only_as_reference": True,
            "normal_unseen_used_for_model_selection": False,
            "class_disjoint_heldout_seen_used_for_selection": True,
            "strict_training_seed_role": "independent_head_initialization_and_batch_order",
            "strict_probe_seed_role": "correlated_posthoc_sample_selection_robustness",
        },
        "class_disjoint_selection": class_split,
        "feature_cache_manifest": cache_manifest,
        "conditions": conditions,
        "semantic_permutation": {
            "seed": int(training["SEMANTIC_PERMUTATION_SEED"]),
            "permutation": semantic_permutation,
        },
        "duration_seconds": round(time.time() - started, 3),
    }
    _atomic_json(output_dir / "p13a_result.json", result)
    _atomic_json(
        output_dir / "execution_state.json",
        {
            "format": "p13a_execution_state_v1",
            "status": "completed",
            "valid": True,
            "head_seed": int(args.head_seed),
            "a2_training_seed": int(cache_manifest["source"]["checkpoint_seed"]),
            "training_performed": True,
            "duration_seconds": result["duration_seconds"],
        },
    )
    print(json.dumps({
        "status": "completed",
        "output": str(output_dir / "p13a_result.json"),
        "duration_seconds": result["duration_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
