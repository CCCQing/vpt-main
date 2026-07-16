#!/usr/bin/env python3
"""Evaluate frozen healthy Stage-2B prompt banks on full CUB GZSL/ZSL splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets.xlsa_dataset import CUB200Dataset  # noqa: E402
from src.data.transforms import get_transforms  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.tools.evaluate_graph_gp_center_transfer import (  # noqa: E402
    _centered_cosine_kernel,
    _kernel_alignment,
    _safe_spearman,
)
from src.tools.export_prompt_posterior_cache import _load_trainable_checkpoint, _setup_cfg  # noqa: E402
from src.utils import logging  # noqa: E402
from src.monitoring.writer import stage2_metadata, write_stage2_json, write_stage2_table  # noqa: E402


GROUP_SPECS = (
    ("D0_image_posterior", "image", None),
    ("D0_empirical_replace_oracle", "replace", "d0_empirical_real"),
    ("D0_empirical_fusion_oracle", "oracle_fusion", "d0_empirical_real"),
    ("D1_moment_replace_oracle", "replace", "d1_moment_real"),
    ("D1_moment_fusion_oracle", "oracle_fusion", "d1_moment_real"),
    ("D2_task_replace_oracle", "replace", "d2_task_real"),
    ("D2_task_fusion_oracle", "oracle_fusion", "d2_task_real"),
    ("D2_graph_gp_deployable", "deployable", "d2_task_real"),
    ("D2_shuffled_graph_deployable", "deployable", "d2_task_shuffled"),
    ("D3_ce_only_oracle_reference", "ce_oracle", "ce_oracle"),
)
GROUPS = tuple(spec[0] for spec in GROUP_SPECS)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] = None) -> None:
    write_stage2_table(path, rows, metadata or stage2_metadata("stage2B_eval"))


def _load_healthy(path: Path, seed: int) -> Dict[str, Any]:
    required = {
        "seen_class_ids",
        "unseen_class_ids",
        "semantic_repr",
        "graph",
        "metadata_json",
    }
    for _, _, bank_key in GROUP_SPECS:
        if bank_key and bank_key not in {"ce_oracle"}:
            required.add(f"{bank_key}_mu")
            required.add(f"{bank_key}_logvar")
            required.add(f"{bank_key}_gp_uncertainty")
    with np.load(str(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"Healthy bundle is missing {missing}: {path}")
        result = {key: payload[key] for key in required}
    metadata = json.loads(str(result.pop("metadata_json").item()))
    if metadata.get("format") != "stage2b_healthy_prompt_distributions_v1":
        raise ValueError(f"Unsupported healthy bundle format: {path}")
    if int(metadata.get("seed")) != int(seed):
        raise ValueError(f"Healthy bundle seed {metadata.get('seed')} does not match {seed}.")
    result["metadata"] = metadata
    result["seen_class_ids"] = np.asarray(result["seen_class_ids"], dtype=np.int64)
    result["unseen_class_ids"] = np.asarray(result["unseen_class_ids"], dtype=np.int64)
    for key, value in list(result.items()):
        if key not in {"metadata", "seen_class_ids", "unseen_class_ids"}:
            result[key] = np.asarray(value, dtype=np.float64)
            if not np.isfinite(result[key]).all():
                raise FloatingPointError(f"Healthy bundle field {key} contains NaN or Inf.")
    return result


def _load_oracle(path: Optional[Path], seed: int) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with np.load(str(path), allow_pickle=False) as payload:
        required = {"oracle_mu", "oracle_logvar", "metadata_json"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"CE Oracle bundle is missing {missing}: {path}")
        result = {key: payload[key] for key in required}
    metadata = json.loads(str(result.pop("metadata_json").item()))
    if metadata.get("format") != "stage2b_oracle_prompt_distributions_v1":
        raise ValueError(f"Unsupported CE Oracle format: {path}")
    if int(metadata.get("seed")) != int(seed):
        raise ValueError(f"CE Oracle seed {metadata.get('seed')} does not match {seed}.")
    result["metadata"] = metadata
    result["oracle_mu"] = np.asarray(result["oracle_mu"], dtype=np.float64)
    result["oracle_logvar"] = np.asarray(result["oracle_logvar"], dtype=np.float64)
    return result


def _dataset_loader(cfg, split: str, batch_size: int, num_workers: int):
    dataset = CUB200Dataset(cfg, split)
    dataset.transform = get_transforms("stage2_extract", int(cfg.DATA.CROPSIZE))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def _empty_stats(image_mu: torch.Tensor, image_logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
    image_variance_mean = image_logvar.exp().mean(dim=-1)
    return {
        "candidate_entropy": image_mu.new_full((image_mu.shape[0],), float("nan")),
        "candidate_top1_mass": image_mu.new_full((image_mu.shape[0],), float("nan")),
        "candidate_top1_correct": image_mu.new_full((image_mu.shape[0],), float("nan")),
        "candidate_true_mass": image_mu.new_full((image_mu.shape[0],), float("nan")),
        "posterior_shift": image_mu.new_zeros(image_mu.shape[0]),
        "variance_ratio": image_mu.new_ones(image_mu.shape[0]),
        "image_variance_mean": image_variance_mean,
        "intervention_variance_mean": image_variance_mean,
    }


def _pair_fusion(
    image_mu: torch.Tensor,
    image_logvar: torch.Tensor,
    class_mu: torch.Tensor,
    class_logvar: torch.Tensor,
    alpha: float,
    variance_min: float,
    variance_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    image_variance = image_logvar.exp().clamp(min=variance_min, max=variance_max)
    class_variance = class_logvar.exp().clamp(min=variance_min, max=variance_max)
    precision = image_variance.reciprocal() + float(alpha) * class_variance.reciprocal()
    fused_variance = precision.reciprocal().clamp(min=variance_min, max=variance_max)
    fused_mu = fused_variance * (
        image_mu / image_variance + float(alpha) * class_mu / class_variance
    )
    return fused_mu, fused_variance.log(), {
        "candidate_entropy": image_mu.new_zeros(image_mu.shape[0]),
        "candidate_top1_mass": image_mu.new_ones(image_mu.shape[0]),
        "candidate_top1_correct": image_mu.new_full((image_mu.shape[0],), float("nan")),
        "candidate_true_mass": image_mu.new_full((image_mu.shape[0],), float("nan")),
        "posterior_shift": torch.norm(fused_mu - image_mu, dim=-1),
        "variance_ratio": fused_variance.mean(dim=-1) / image_variance.mean(dim=-1),
        "image_variance_mean": image_variance.mean(dim=-1),
        "intervention_variance_mean": fused_variance.mean(dim=-1),
    }


def _posterior_fusion(
    image_mu: torch.Tensor,
    image_logvar: torch.Tensor,
    first_logits: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    alpha: float,
    beta: float,
    temperature: float,
    candidate_topk: int,
    variance_min: float,
    variance_max: float,
    candidate_ids: torch.Tensor,
    target_labels: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    image_variance = image_logvar.exp().clamp(min=variance_min, max=variance_max)
    prior_variance = prior_logvar.exp().clamp(min=variance_min, max=variance_max)
    difference = image_mu[:, None, :] - prior_mu[None, :, :]
    energy = -0.5 * (
        (difference * difference) / prior_variance[None, :, :] + prior_logvar[None, :, :]
    ).mean(dim=-1)
    scores = (first_logits + float(beta) * energy) / float(temperature)
    if 0 < int(candidate_topk) < scores.shape[1]:
        top_values, top_indices = scores.topk(int(candidate_topk), dim=1)
        masked = scores.new_full(scores.shape, float("-inf"))
        masked.scatter_(1, top_indices, top_values)
        scores = masked
    weights = torch.softmax(scores, dim=-1)
    selected_class = candidate_ids.index_select(0, weights.argmax(dim=-1))
    if target_labels is None:
        candidate_top1_correct = image_mu.new_full((image_mu.shape[0],), float("nan"))
        candidate_true_mass = image_mu.new_full((image_mu.shape[0],), float("nan"))
    else:
        target_mask = candidate_ids[None, :] == target_labels[:, None]
        candidate_top1_correct = (selected_class == target_labels).float()
        candidate_true_mass = (weights * target_mask.float()).sum(dim=-1)
        candidate_true_mass = torch.where(
            target_mask.any(dim=-1),
            candidate_true_mass,
            candidate_true_mass.new_full(candidate_true_mass.shape, float("nan")),
        )
    mixture_mu = weights @ prior_mu
    mixture_second = weights @ (prior_variance + prior_mu * prior_mu)
    mixture_variance = (mixture_second - mixture_mu * mixture_mu).clamp(
        min=variance_min, max=variance_max
    )
    precision = image_variance.reciprocal() + float(alpha) * mixture_variance.reciprocal()
    fused_variance = precision.reciprocal().clamp(min=variance_min, max=variance_max)
    fused_mu = fused_variance * (
        image_mu / image_variance + float(alpha) * mixture_mu / mixture_variance
    )
    return fused_mu, fused_variance.log(), {
        "candidate_entropy": -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1),
        "candidate_top1_mass": weights.max(dim=-1).values,
        "candidate_top1_correct": candidate_top1_correct,
        "candidate_true_mass": candidate_true_mass,
        "posterior_shift": torch.norm(fused_mu - image_mu, dim=-1),
        "variance_ratio": fused_variance.mean(dim=-1) / image_variance.mean(dim=-1),
        "image_variance_mean": image_variance.mean(dim=-1),
        "intervention_variance_mean": fused_variance.mean(dim=-1),
    }


def _forward_override(
    model,
    images: torch.Tensor,
    class_ids: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.set_runtime_prompt_distribution_override(mu, logvar)
    try:
        with torch.no_grad():
            logits = model(images, semantics=None, class_ids=class_ids, runtime_targets=None)
    finally:
        model.clear_runtime_prompt_distribution_override()
    classifier = model.get_runtime_classifier_stats()
    return logits, classifier["visual_repr"], classifier["semantic_repr"]


def _classification_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    wrong = logits.clone()
    wrong.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - wrong.max(dim=1).values


def _unseen_margin(
    unseen_logits: torch.Tensor,
    labels: torch.Tensor,
    unseen_ids: torch.Tensor,
) -> torch.Tensor:
    local_by_global = labels.new_full((int(unseen_ids.max().item()) + 1,), -1)
    local_by_global[unseen_ids] = torch.arange(unseen_ids.numel(), device=labels.device)
    local_labels = local_by_global[labels]
    return _classification_margin(unseen_logits, local_labels)


def _macro_accuracy(prediction: np.ndarray, target: np.ndarray, class_ids: np.ndarray) -> float:
    values = []
    for class_id in class_ids:
        mask = target == int(class_id)
        if not mask.any():
            raise ValueError(f"No samples for class {int(class_id)}.")
        values.append(float((prediction[mask] == target[mask]).mean()))
    return float(np.mean(values))


def _summary_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}_{name}": float("nan") for name in (
            "mean", "median", "q05", "q25", "q75", "q95", "positive_ratio"
        )}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_positive_ratio": float((values > 0.0).mean()),
    }


def _finite_tensor_mean(values: Sequence[torch.Tensor]) -> float:
    merged = torch.cat(list(values))
    finite = torch.isfinite(merged)
    return float(merged[finite].mean().item()) if bool(finite.any().item()) else float("nan")


def _relation_metrics(
    prompt_centers: np.ndarray,
    visual_centers: np.ndarray,
    logit_profiles: np.ndarray,
    semantic_repr: np.ndarray,
    graph: np.ndarray,
    eps: float,
) -> Dict[str, float]:
    relations = {
        "prompt": _centered_cosine_kernel(prompt_centers, eps),
        "visual": _centered_cosine_kernel(visual_centers, eps),
        "logits": _centered_cosine_kernel(logit_profiles, eps),
        "semantic": _centered_cosine_kernel(semantic_repr, eps),
    }
    upper = np.triu_indices(graph.shape[0], k=1)
    pairs = {
        "graph_prompt": (graph, relations["prompt"]),
        "graph_visual": (graph, relations["visual"]),
        "graph_logits": (graph, relations["logits"]),
        "graph_semantic": (graph, relations["semantic"]),
        "prompt_visual": (relations["prompt"], relations["visual"]),
        "prompt_logits": (relations["prompt"], relations["logits"]),
        "prompt_semantic": (relations["prompt"], relations["semantic"]),
        "visual_logits": (relations["visual"], relations["logits"]),
        "visual_semantic": (relations["visual"], relations["semantic"]),
        "logits_semantic": (relations["logits"], relations["semantic"]),
    }
    result: Dict[str, float] = {}
    for name, (left, right) in pairs.items():
        result[f"{name}_cka"] = _kernel_alignment(left, right, eps)
        result[f"{name}_spearman"] = _safe_spearman(left[upper], right[upper])
    return result


def evaluate(args: argparse.Namespace) -> None:
    setup_args = argparse.Namespace(
        config_file=args.config_file,
        cell="A00",
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output=args.output_dir / "cache_placeholder.npz",
        opts=[],
    )
    cfg, _ = _setup_cfg(setup_args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.setup_logging(1, 1, output=str(args.output_dir), name="visual_prompt")
    healthy = _load_healthy(args.healthy_bank, args.seed)
    oracle = _load_oracle(args.oracle, args.seed)
    active_specs = [spec for spec in GROUP_SPECS if spec[1] != "ce_oracle" or oracle is not None]
    active_groups = [spec[0] for spec in active_specs]
    seen_ids = healthy["seen_class_ids"]
    unseen_ids = healthy["unseen_class_ids"]
    seen_dataset, seen_loader = _dataset_loader(cfg, "test_seen", args.batch_size, args.num_workers)
    _, unseen_loader = _dataset_loader(cfg, "test_unseen", args.batch_size, args.num_workers)
    model, device = build_model(cfg)
    runtime_device = torch.device(f"cuda:{int(device)}") if isinstance(device, int) else torch.device(device)
    model.attach_r_similarity_head(seen_dataset.class_attributes)
    _load_trainable_checkpoint(model, args.checkpoint, args.seed, "A00")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    class_ids = torch.arange(int(cfg.DATA.NUMBER_CLASSES), device=runtime_device, dtype=torch.long)
    unseen_tensor = torch.from_numpy(unseen_ids).to(runtime_device)
    seen_tensor = torch.from_numpy(seen_ids).to(runtime_device)
    bank_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    bank_uncertainty: Dict[str, np.ndarray] = {}
    for _, _, bank_key in active_specs:
        if not bank_key or bank_key in bank_arrays:
            continue
        if bank_key == "ce_oracle":
            bank_arrays[bank_key] = (oracle["oracle_mu"], oracle["oracle_logvar"])
            bank_uncertainty[bank_key] = np.full(class_ids.numel(), np.nan, dtype=np.float64)
        else:
            bank_arrays[bank_key] = (healthy[f"{bank_key}_mu"], healthy[f"{bank_key}_logvar"])
            bank_uncertainty[bank_key] = healthy[f"{bank_key}_gp_uncertainty"]
    bank_tensors = {
        key: (
            torch.from_numpy(mu.astype(np.float32)).to(runtime_device),
            torch.from_numpy(logvar.astype(np.float32)).to(runtime_device),
        )
        for key, (mu, logvar) in bank_arrays.items()
    }
    accumulators: Dict[str, Dict[str, Any]] = {}
    for group in active_groups:
        accumulators[group] = {
            "labels": [],
            "predictions": [],
            "zsl_predictions": [],
            "split": [],
            "paired_visual_semantic": [],
            "paired_prompt_semantic": [],
            "paired_prompt_visual": [],
            "candidate_entropy": [],
            "candidate_top1_mass": [],
            "candidate_top1_correct": [],
            "candidate_true_mass": [],
            "posterior_shift": [],
            "variance_ratio": [],
            "image_variance_mean": [],
            "intervention_variance_mean": [],
            "gzsl_margin": [],
            "zsl_margin": [],
            "unseen_vs_seen_margin": [],
            "top_wrong_is_seen": [],
            "count": torch.zeros(class_ids.numel(), device=runtime_device, dtype=torch.float64),
            "prompt_sum": torch.zeros(class_ids.numel(), int(cfg.MODEL.GRAPH_INPUT.TEXT_DIM), device=runtime_device, dtype=torch.float64),
            "visual_sum": torch.zeros(class_ids.numel(), int(cfg.MODEL.GRAPH_INPUT.TEXT_DIM), device=runtime_device, dtype=torch.float64),
            "logit_sum": torch.zeros(class_ids.numel(), class_ids.numel(), device=runtime_device, dtype=torch.float64),
            "semantic_repr": None,
        }
    for split_name, loader in (("seen", seen_loader), ("unseen", unseen_loader)):
        for batch in loader:
            images = batch["image"].to(device=runtime_device, non_blocking=True)
            labels = batch["label"].to(device=runtime_device, dtype=torch.long)
            with torch.no_grad():
                first_logits = model(images, semantics=None, class_ids=class_ids, runtime_targets=None)
                image_stats = model.get_runtime_prompt_distribution_stats()
                image_mu = image_stats["mu"]
                image_logvar = image_stats["logvar"]
            for group, mode, bank_key in active_specs:
                if mode == "image":
                    fused_mu, fused_logvar = image_mu, image_logvar
                    fusion_stats = _empty_stats(image_mu, image_logvar)
                else:
                    prior_mu, prior_logvar = bank_tensors[bank_key]
                    if mode in {"replace", "ce_oracle"}:
                        fused_mu = prior_mu.index_select(0, labels)
                        fused_logvar = prior_logvar.index_select(0, labels)
                        fusion_stats = _empty_stats(image_mu, image_logvar)
                        fusion_stats["posterior_shift"] = torch.norm(fused_mu - image_mu, dim=-1)
                        fusion_stats["variance_ratio"] = (
                            fused_logvar.exp().mean(dim=-1) / image_logvar.exp().mean(dim=-1)
                        )
                        fusion_stats["intervention_variance_mean"] = fused_logvar.exp().mean(dim=-1)
                    elif mode == "oracle_fusion":
                        fused_mu, fused_logvar, fusion_stats = _pair_fusion(
                            image_mu,
                            image_logvar,
                            prior_mu.index_select(0, labels),
                            prior_logvar.index_select(0, labels),
                            args.fusion_alpha,
                            args.variance_min,
                            args.variance_max,
                        )
                    else:
                        fused_mu, fused_logvar, fusion_stats = _posterior_fusion(
                            image_mu,
                            image_logvar,
                            first_logits,
                            prior_mu,
                            prior_logvar,
                            args.fusion_alpha,
                            args.energy_beta,
                            args.candidate_temperature,
                            args.candidate_topk,
                            args.variance_min,
                            args.variance_max,
                            class_ids,
                            labels,
                        )
                logits, visual, semantic_bank = _forward_override(
                    model, images, class_ids, fused_mu, fused_logvar
                )
                target_semantic = semantic_bank.index_select(0, labels)
                prediction = class_ids.index_select(0, logits.argmax(dim=1))
                unseen_logits = logits.index_select(1, unseen_tensor)
                if split_name == "unseen" and mode == "deployable":
                    prior_mu, prior_logvar = bank_tensors[bank_key]
                    zsl_mu, zsl_logvar, _ = _posterior_fusion(
                        image_mu,
                        image_logvar,
                        first_logits.index_select(1, unseen_tensor),
                        prior_mu.index_select(0, unseen_tensor),
                        prior_logvar.index_select(0, unseen_tensor),
                        args.fusion_alpha,
                        args.energy_beta,
                        args.candidate_temperature,
                        args.candidate_topk,
                        args.variance_min,
                        args.variance_max,
                        unseen_tensor,
                        labels,
                    )
                    zsl_full_logits, _, _ = _forward_override(
                        model, images, class_ids, zsl_mu, zsl_logvar
                    )
                    unseen_logits = zsl_full_logits.index_select(1, unseen_tensor)
                zsl_prediction = unseen_tensor.index_select(0, unseen_logits.argmax(dim=1))
                gzsl_margin = _classification_margin(logits, labels)
                acc = accumulators[group]
                acc["labels"].append(labels.cpu().numpy())
                acc["predictions"].append(prediction.cpu().numpy())
                acc["zsl_predictions"].append(zsl_prediction.cpu().numpy())
                acc["split"].extend([split_name] * labels.shape[0])
                acc["paired_visual_semantic"].append(F.cosine_similarity(visual, target_semantic, dim=-1).cpu())
                acc["paired_prompt_semantic"].append(F.cosine_similarity(fused_mu, target_semantic, dim=-1).cpu())
                acc["paired_prompt_visual"].append(F.cosine_similarity(fused_mu, visual, dim=-1).cpu())
                acc["gzsl_margin"].append(gzsl_margin.cpu())
                if split_name == "unseen":
                    acc["zsl_margin"].append(_unseen_margin(unseen_logits, labels, unseen_tensor).cpu())
                    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
                    seen_scores = logits.index_select(1, seen_tensor)
                    acc["unseen_vs_seen_margin"].append((true_logits - seen_scores.max(dim=1).values).cpu())
                    wrong = logits.clone()
                    wrong.scatter_(1, labels[:, None], float("-inf"))
                    top_wrong = class_ids.index_select(0, wrong.argmax(dim=1))
                    acc["top_wrong_is_seen"].append(
                        (top_wrong[:, None] == seen_tensor[None, :]).any(dim=1).float().cpu()
                    )
                for key in (
                    "candidate_entropy",
                    "candidate_top1_mass",
                    "candidate_top1_correct",
                    "candidate_true_mass",
                    "posterior_shift",
                    "variance_ratio",
                    "image_variance_mean",
                    "intervention_variance_mean",
                ):
                    acc[key].append(fusion_stats[key].detach().cpu())
                ones = torch.ones(labels.shape[0], device=runtime_device, dtype=torch.float64)
                acc["count"].index_add_(0, labels, ones)
                acc["prompt_sum"].index_add_(0, labels, fused_mu.double())
                acc["visual_sum"].index_add_(0, labels, visual.double())
                acc["logit_sum"].index_add_(0, labels, logits.double())
                acc["semantic_repr"] = semantic_bank.detach().cpu().numpy().astype(np.float64)
    rows: List[Dict[str, Any]] = []
    for group in active_groups:
        acc = accumulators[group]
        labels = np.concatenate(acc["labels"])
        predictions = np.concatenate(acc["predictions"])
        zsl_predictions = np.concatenate(acc["zsl_predictions"])
        split = np.asarray(acc["split"])
        seen_mask = split == "seen"
        unseen_mask = split == "unseen"
        seen_accuracy = _macro_accuracy(predictions[seen_mask], labels[seen_mask], seen_ids)
        unseen_accuracy = _macro_accuracy(predictions[unseen_mask], labels[unseen_mask], unseen_ids)
        harmonic = 0.0 if seen_accuracy + unseen_accuracy <= 0.0 else (
            2.0 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy)
        )
        zsl_unseen = _macro_accuracy(zsl_predictions[unseen_mask], labels[unseen_mask], unseen_ids)
        count = acc["count"]
        if bool((count <= 0).any().item()):
            raise RuntimeError(f"Stage-2B evaluation did not cover all classes for {group}.")
        prompt_centers = (acc["prompt_sum"] / count[:, None]).cpu().numpy()
        visual_centers = (acc["visual_sum"] / count[:, None]).cpu().numpy()
        logit_profiles = (acc["logit_sum"] / count[:, None]).cpu().numpy()
        gzsl_margins = torch.cat(acc["gzsl_margin"]).numpy()
        paired_visual_semantic = torch.cat(acc["paired_visual_semantic"]).numpy()
        paired_prompt_semantic = torch.cat(acc["paired_prompt_semantic"]).numpy()
        paired_prompt_visual = torch.cat(acc["paired_prompt_visual"]).numpy()
        _, _, bank_key = next(spec for spec in active_specs if spec[0] == group)
        if bank_key is None:
            prior_variance_seen = float("nan")
            prior_variance_unseen = float("nan")
            gp_uncertainty_unseen = float("nan")
        else:
            prior_variance = np.exp(
                np.clip(bank_arrays[bank_key][1], np.log(args.variance_min), np.log(args.variance_max))
            )
            prior_variance_seen = float(prior_variance[seen_ids].mean())
            prior_variance_unseen = float(prior_variance[unseen_ids].mean())
            uncertainty = np.asarray(bank_uncertainty[bank_key], dtype=np.float64)
            finite_uncertainty = uncertainty[unseen_ids][np.isfinite(uncertainty[unseen_ids])]
            gp_uncertainty_unseen = (
                float(finite_uncertainty.mean()) if finite_uncertainty.size else float("nan")
            )
        row: Dict[str, Any] = {
            "model_seed": int(args.seed),
            "group": group,
            "heldout_gzsl_seen": seen_accuracy,
            "heldout_gzsl_unseen": unseen_accuracy,
            "heldout_gzsl_h": harmonic,
            "heldout_zsl_unseen": zsl_unseen,
            "paired_visual_semantic_cosine": float(paired_visual_semantic.mean()),
            "paired_visual_semantic_cosine_seen": float(paired_visual_semantic[seen_mask].mean()),
            "paired_visual_semantic_cosine_unseen": float(paired_visual_semantic[unseen_mask].mean()),
            "paired_prompt_semantic_cosine": float(paired_prompt_semantic.mean()),
            "paired_prompt_semantic_cosine_seen": float(paired_prompt_semantic[seen_mask].mean()),
            "paired_prompt_semantic_cosine_unseen": float(paired_prompt_semantic[unseen_mask].mean()),
            "paired_prompt_visual_cosine": float(paired_prompt_visual.mean()),
            "paired_prompt_visual_cosine_seen": float(paired_prompt_visual[seen_mask].mean()),
            "paired_prompt_visual_cosine_unseen": float(paired_prompt_visual[unseen_mask].mean()),
            "candidate_entropy": _finite_tensor_mean(acc["candidate_entropy"]),
            "candidate_top1_mass": _finite_tensor_mean(acc["candidate_top1_mass"]),
            "candidate_top1_accuracy": _finite_tensor_mean(acc["candidate_top1_correct"]),
            "candidate_true_mass": _finite_tensor_mean(acc["candidate_true_mass"]),
            "posterior_shift": float(torch.cat(acc["posterior_shift"]).mean().item()),
            "variance_ratio": float(torch.cat(acc["variance_ratio"]).mean().item()),
            "image_variance_mean": float(torch.cat(acc["image_variance_mean"]).mean().item()),
            "intervention_variance_mean": float(
                torch.cat(acc["intervention_variance_mean"]).mean().item()
            ),
            "prior_variance_seen_mean": prior_variance_seen,
            "prior_variance_unseen_mean": prior_variance_unseen,
            "gp_uncertainty_unseen_mean": gp_uncertainty_unseen,
            "top_wrong_is_seen_ratio": float(torch.cat(acc["top_wrong_is_seen"]).mean().item()),
        }
        row.update(_summary_stats(gzsl_margins[seen_mask], "gzsl_seen_margin"))
        row.update(_summary_stats(gzsl_margins[unseen_mask], "gzsl_unseen_margin"))
        row.update(_summary_stats(torch.cat(acc["zsl_margin"]).numpy(), "zsl_unseen_margin"))
        row.update(_summary_stats(
            torch.cat(acc["unseen_vs_seen_margin"]).numpy(), "unseen_vs_seen_margin"
        ))
        row.update(_relation_metrics(
            prompt_centers,
            visual_centers,
            logit_profiles,
            np.asarray(acc["semantic_repr"]),
            healthy["graph"],
            args.eps,
        ))
        rows.append(row)
    output_metadata = stage2_metadata("stage2B_eval", seed=args.seed)
    _write_csv(args.output_dir / "group_results.csv", rows, output_metadata)
    np.savez_compressed(
        str(args.output_dir / "prior_banks.npz"),
        **{f"{key}_mu": values[0].astype(np.float32) for key, values in bank_arrays.items()},
        **{f"{key}_logvar": values[1].astype(np.float32) for key, values in bank_arrays.items()},
    )
    metadata = {
        "format": "stage2b_healthy_intervention_results_v1",
        "seed": int(args.seed),
        "groups": active_groups,
        "group_specs": active_specs,
        "healthy_bank": str(args.healthy_bank),
        "ce_oracle": str(args.oracle) if args.oracle is not None else None,
        "checkpoint": str(args.checkpoint),
        "evaluation_protocol": "full test_seen plus full test_unseen",
        "fusion_alpha": float(args.fusion_alpha),
        "energy_beta": float(args.energy_beta),
        "candidate_temperature": float(args.candidate_temperature),
        "candidate_topk": int(args.candidate_topk),
        "leakage_contract": {
            "D0_D1_D2": "healthy banks use seen trainval only; true-label groups are diagnostic only",
            "deployable": "no true label is used before prediction; ZSL selection is unseen-restricted",
            "D3": "legacy all-class CE Oracle; formal unseen leakage upper bound only",
        },
        "group_contracts": {
            group: {
                "bank_source": bank_key,
                "whether_ce_trained": mode == "ce_oracle" or str(bank_key).startswith("d2_task"),
                "uses_unseen_images": mode == "ce_oracle",
                "uses_true_label_for_selection": mode in {"replace", "oracle_fusion", "ce_oracle"},
                "deployable": mode in {"image", "deployable"},
            }
            for group, mode, bank_key in active_specs
        },
    }
    write_stage2_json(args.output_dir / "results.json", metadata, output_metadata)
    print(f"wrote {args.output_dir / 'group_results.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2B healthy prompt distribution interventions.")
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "prompt" / "cub.yaml")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--healthy-bank", required=True, type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fusion-alpha", type=float, default=1.0)
    parser.add_argument("--energy-beta", type=float, default=1.0)
    parser.add_argument("--candidate-temperature", type=float, default=1.0)
    parser.add_argument("--candidate-topk", type=int, default=0)
    parser.add_argument("--variance-min", type=float, default=1e-4)
    parser.add_argument("--variance-max", type=float, default=100.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    args.config_file = args.config_file.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.healthy_bank = args.healthy_bank.resolve()
    args.oracle = args.oracle.resolve() if args.oracle is not None else None
    args.output_dir = args.output_dir.resolve()
    required = [args.config_file, args.checkpoint, args.healthy_bank]
    if args.oracle is not None:
        required.append(args.oracle)
    for path in required:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    positive = (
        args.batch_size,
        args.fusion_alpha,
        args.candidate_temperature,
        args.variance_min,
        args.variance_max,
        args.eps,
    )
    if (
        args.seed < 0
        or args.num_workers < 0
        or args.candidate_topk < 0
        or args.energy_beta < 0
        or min(positive) <= 0
        or args.variance_max < args.variance_min
    ):
        parser.error("Stage-2B healthy evaluation arguments are invalid.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
