#!/usr/bin/env python3
"""Evaluate B0-B4 prompt-prior interventions with three A00 model seeds."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets.xlsa_dataset import CUB200Dataset  # noqa: E402
from src.data.transforms import get_transforms  # noqa: E402
from src.monitoring.writer import stage2_metadata, write_stage2_json, write_stage2_table  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.tools.evaluate_graph_gp_center_transfer import (  # noqa: E402
    _centered_cosine_kernel,
    _kernel_alignment,
    _safe_spearman,
)
from src.tools.export_prompt_posterior_cache import _load_trainable_checkpoint, _setup_cfg  # noqa: E402
from src.utils import logging  # noqa: E402


GROUPS = (
    "B0_image_only",
    "B1_attribute_ridge",
    "B2_graph_gp",
    "B3_shuffled_graph_gp",
    "B4_oracle_bank",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    write_stage2_table(path, rows, metadata)


def _load_oracle(path: Path, seed: int) -> Dict[str, Any]:
    with np.load(str(path), allow_pickle=False) as payload:
        required = {
            "oracle_mu",
            "oracle_logvar",
            "class_attributes",
            "seen_class_ids",
            "unseen_class_ids",
            "metadata_json",
            "split_manifest_json",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"Oracle bundle is missing {missing}: {path}")
        result = {key: payload[key] for key in required}
    metadata = json.loads(str(result.pop("metadata_json").item()))
    split_manifest = json.loads(str(result.pop("split_manifest_json").item()))
    if metadata.get("format") != "stage2b_oracle_prompt_distributions_v1":
        raise ValueError(f"Unsupported oracle bundle format: {path}")
    if int(metadata.get("seed")) != int(seed):
        raise ValueError(f"Oracle seed {metadata.get('seed')} does not match requested seed {seed}.")
    result["metadata"] = metadata
    result["split_manifest"] = split_manifest
    for key in ("oracle_mu", "oracle_logvar", "class_attributes"):
        result[key] = np.asarray(result[key], dtype=np.float64)
        if not np.isfinite(result[key]).all():
            raise FloatingPointError(f"Oracle bundle field {key} contains NaN or Inf.")
    result["seen_class_ids"] = np.asarray(result["seen_class_ids"], dtype=np.int64)
    result["unseen_class_ids"] = np.asarray(result["unseen_class_ids"], dtype=np.int64)
    return result


def _clean_kernel(matrix: np.ndarray, eps: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.maximum(0.5 * (matrix + matrix.T), 0.0)
    diagonal = np.maximum(np.diag(matrix), eps)
    matrix = matrix / np.sqrt(diagonal[:, None] * diagonal[None, :])
    matrix = np.clip(0.5 * (matrix + matrix.T), 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _load_graph(path: Path, eps: float) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as payload:
        if "method1_diff" not in payload.files:
            raise KeyError(f"Graph bundle has no method1_diff: {path}")
        return _clean_kernel(payload["method1_diff"], eps)


def _attribute_ridge(
    attributes: np.ndarray,
    seen_ids: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> np.ndarray:
    design = np.concatenate(
        (attributes, np.ones((attributes.shape[0], 1), dtype=np.float64)), axis=1
    )
    support = design[seen_ids]
    system = support @ support.T + float(ridge) * np.eye(seen_ids.size, dtype=np.float64)
    dual = np.linalg.solve(system, target[seen_ids])
    return (design @ support.T) @ dual


def _graph_predict(
    kernel: np.ndarray,
    seen_ids: np.ndarray,
    unseen_ids: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> np.ndarray:
    system = kernel[np.ix_(seen_ids, seen_ids)] + float(ridge) * np.eye(seen_ids.size, dtype=np.float64)
    mean = target[seen_ids].mean(axis=0, keepdims=True)
    solved = np.linalg.solve(system, target[seen_ids] - mean)
    return mean + kernel[np.ix_(unseen_ids, seen_ids)] @ solved


def _prior_banks(
    oracle: Mapping[str, Any],
    graph: np.ndarray,
    seed: int,
    ridge: float,
    logvar_min: float,
    logvar_max: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    oracle_mu = np.asarray(oracle["oracle_mu"], dtype=np.float64)
    oracle_logvar = np.asarray(oracle["oracle_logvar"], dtype=np.float64)
    attributes = np.asarray(oracle["class_attributes"], dtype=np.float64)
    seen = np.asarray(oracle["seen_class_ids"], dtype=np.int64)
    unseen = np.asarray(oracle["unseen_class_ids"], dtype=np.int64)

    ridge_mu = _attribute_ridge(attributes, seen, oracle_mu, ridge)
    ridge_logvar = _attribute_ridge(attributes, seen, oracle_logvar, ridge)
    ridge_mu[seen] = oracle_mu[seen]
    ridge_logvar[seen] = oracle_logvar[seen]

    graph_mu = np.empty_like(oracle_mu)
    graph_logvar = np.empty_like(oracle_logvar)
    graph_mu[seen] = oracle_mu[seen]
    graph_logvar[seen] = oracle_logvar[seen]
    graph_mu[unseen] = _graph_predict(graph, seen, unseen, oracle_mu, ridge)
    graph_logvar[unseen] = _graph_predict(graph, seen, unseen, oracle_logvar, ridge)

    rng = np.random.RandomState(int(seed) + 200003)
    permutation = rng.permutation(graph.shape[0])
    shuffled = graph[np.ix_(permutation, permutation)]
    shuffled_mu = np.empty_like(oracle_mu)
    shuffled_logvar = np.empty_like(oracle_logvar)
    shuffled_mu[seen] = oracle_mu[seen]
    shuffled_logvar[seen] = oracle_logvar[seen]
    shuffled_mu[unseen] = _graph_predict(shuffled, seen, unseen, oracle_mu, ridge)
    shuffled_logvar[unseen] = _graph_predict(shuffled, seen, unseen, oracle_logvar, ridge)

    banks = {
        "B1_attribute_ridge": (ridge_mu, ridge_logvar),
        "B2_graph_gp": (graph_mu, graph_logvar),
        "B3_shuffled_graph_gp": (shuffled_mu, shuffled_logvar),
        "B4_oracle_bank": (oracle_mu.copy(), oracle_logvar.copy()),
    }
    return {
        name: (mu.astype(np.float32), np.clip(logvar, logvar_min, logvar_max).astype(np.float32))
        for name, (mu, logvar) in banks.items()
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
    variance_min: float,
    variance_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    image_variance = image_logvar.exp().clamp(min=variance_min, max=variance_max)
    prior_variance = prior_logvar.exp().clamp(min=variance_min, max=variance_max)
    difference = image_mu[:, None, :] - prior_mu[None, :, :]
    energy = -0.5 * ((difference * difference) / prior_variance[None, :, :] + prior_logvar[None, :, :]).mean(dim=-1)
    weights = torch.softmax((first_logits + float(beta) * energy) / float(temperature), dim=-1)
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
        "posterior_shift": torch.norm(fused_mu - image_mu, dim=-1),
        "variance_ratio": fused_variance.mean(dim=-1) / image_variance.mean(dim=-1),
    }


def _macro_accuracy(prediction: np.ndarray, target: np.ndarray, class_ids: np.ndarray) -> float:
    values = []
    for class_id in class_ids:
        mask = target == int(class_id)
        if not mask.any():
            raise ValueError(f"No samples for class {int(class_id)}.")
        values.append(float((prediction[mask] == target[mask]).mean()))
    return float(np.mean(values))


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
        "prompt_visual": (relations["prompt"], relations["visual"]),
        "prompt_logits": (relations["prompt"], relations["logits"]),
        "visual_semantic": (relations["visual"], relations["semantic"]),
    }
    result: Dict[str, float] = {}
    for name, (left, right) in pairs.items():
        result[f"{name}_cka"] = _kernel_alignment(left, right, eps)
        result[f"{name}_spearman"] = _safe_spearman(left[upper], right[upper])
    return result


def _prior_quality(
    prior_mu: np.ndarray,
    prior_logvar: np.ndarray,
    oracle_mu: np.ndarray,
    oracle_logvar: np.ndarray,
    unseen_ids: np.ndarray,
    eps: float,
) -> Dict[str, float]:
    prediction = prior_mu[unseen_ids]
    truth = oracle_mu[unseen_ids]
    raw_cosine = np.sum(prediction * truth, axis=1) / np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1), eps
    )
    support_mean = oracle_mu.mean(axis=0, keepdims=True)
    centered_prediction = prediction - support_mean
    centered_truth = truth - support_mean
    centered_cosine = np.sum(centered_prediction * centered_truth, axis=1) / np.maximum(
        np.linalg.norm(centered_prediction, axis=1) * np.linalg.norm(centered_truth, axis=1), eps
    )
    error = np.linalg.norm(prediction - truth, axis=1)
    denom = np.maximum(np.linalg.norm(centered_truth, axis=1), eps)
    return {
        "prior_unseen_raw_cosine": float(raw_cosine.mean()),
        "prior_unseen_centered_cosine": float(centered_cosine.mean()),
        "prior_unseen_nrmse": float((error / denom).mean()),
        "prior_unseen_logvar_rmse": float(
            np.sqrt(np.mean((prior_logvar[unseen_ids] - oracle_logvar[unseen_ids]) ** 2))
        ),
    }


def _dataset_loader(
    cfg,
    split: str,
    batch_size: int,
    num_workers: int,
    allowed_paths: Sequence[str],
):
    dataset = CUB200Dataset(cfg, split)
    dataset.transform = get_transforms("stage2_extract", int(cfg.DATA.CROPSIZE))
    allowed = set(str(path) for path in allowed_paths)
    indices = [index for index, record in enumerate(dataset._imdb) if str(record["im_path"]) in allowed]
    actual = {str(dataset._imdb[index]["im_path"]) for index in indices}
    if actual != allowed:
        raise ValueError(f"Oracle held-out paths do not match split={split}: missing={sorted(allowed.difference(actual))[:5]}")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


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

    oracle = _load_oracle(args.oracle, args.seed)
    graph = _load_graph(args.graph_bundle, args.eps)
    banks = _prior_banks(
        oracle,
        graph,
        args.seed,
        args.prior_ridge,
        args.logvar_min,
        args.logvar_max,
    )
    seen_ids = np.asarray(oracle["seen_class_ids"], dtype=np.int64)
    unseen_ids = np.asarray(oracle["unseen_class_ids"], dtype=np.int64)
    oracle_mu_np = np.asarray(oracle["oracle_mu"], dtype=np.float64)
    oracle_logvar_np = np.asarray(oracle["oracle_logvar"], dtype=np.float64)

    split_classes = oracle["split_manifest"]["classes"]
    seen_eval_paths = [
        path
        for class_id in seen_ids
        for path in split_classes[str(int(class_id))]["eval_paths"]
    ]
    unseen_eval_paths = [
        path
        for class_id in unseen_ids
        for path in split_classes[str(int(class_id))]["eval_paths"]
    ]
    seen_dataset, seen_loader = _dataset_loader(
        cfg, "test_seen", args.batch_size, args.num_workers, seen_eval_paths
    )
    unseen_dataset, unseen_loader = _dataset_loader(
        cfg, "test_unseen", args.batch_size, args.num_workers, unseen_eval_paths
    )
    model, device = build_model(cfg)
    runtime_device = torch.device(f"cuda:{int(device)}") if isinstance(device, int) else torch.device(device)
    model.attach_r_similarity_head(seen_dataset.class_attributes)
    _load_trainable_checkpoint(model, args.checkpoint, args.seed, "A00")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    class_ids = torch.arange(int(cfg.DATA.NUMBER_CLASSES), device=runtime_device, dtype=torch.long)
    bank_tensors = {
        name: (
            torch.from_numpy(mu).to(runtime_device),
            torch.from_numpy(logvar).to(runtime_device),
        )
        for name, (mu, logvar) in banks.items()
    }
    accumulators: Dict[str, Dict[str, Any]] = {}
    for group in GROUPS:
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
            "posterior_shift": [],
            "variance_ratio": [],
            "count": torch.zeros(class_ids.numel(), device=runtime_device, dtype=torch.float64),
            "prompt_sum": torch.zeros(class_ids.numel(), oracle_mu_np.shape[1], device=runtime_device, dtype=torch.float64),
            "visual_sum": torch.zeros(class_ids.numel(), oracle_mu_np.shape[1], device=runtime_device, dtype=torch.float64),
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
            for group in GROUPS:
                if group == "B0_image_only":
                    fused_mu = image_mu
                    fused_logvar = image_logvar
                    fusion_stats = {
                        "candidate_entropy": image_mu.new_full((image_mu.shape[0],), float("nan")),
                        "candidate_top1_mass": image_mu.new_full((image_mu.shape[0],), float("nan")),
                        "posterior_shift": image_mu.new_zeros(image_mu.shape[0]),
                        "variance_ratio": image_mu.new_ones(image_mu.shape[0]),
                    }
                else:
                    prior_mu, prior_logvar = bank_tensors[group]
                    fused_mu, fused_logvar, fusion_stats = _posterior_fusion(
                        image_mu,
                        image_logvar,
                        first_logits,
                        prior_mu,
                        prior_logvar,
                        args.fusion_alpha,
                        args.energy_beta,
                        args.candidate_temperature,
                        args.variance_min,
                        args.variance_max,
                    )
                model.set_runtime_prompt_distribution_override(fused_mu, fused_logvar)
                try:
                    with torch.no_grad():
                        logits = model(images, semantics=None, class_ids=class_ids, runtime_targets=None)
                finally:
                    model.clear_runtime_prompt_distribution_override()
                classifier = model.get_runtime_classifier_stats()
                visual = classifier["visual_repr"]
                semantic_bank = classifier["semantic_repr"]
                target_semantic = semantic_bank.index_select(0, labels)
                prediction = class_ids.index_select(0, logits.argmax(dim=1))
                unseen_logits = logits.index_select(1, torch.from_numpy(unseen_ids).to(runtime_device))
                zsl_prediction = torch.from_numpy(unseen_ids).to(runtime_device).index_select(
                    0, unseen_logits.argmax(dim=1)
                )
                acc = accumulators[group]
                acc["labels"].append(labels.cpu().numpy())
                acc["predictions"].append(prediction.cpu().numpy())
                acc["zsl_predictions"].append(zsl_prediction.cpu().numpy())
                acc["split"].extend([split_name] * labels.shape[0])
                acc["paired_visual_semantic"].append(F.cosine_similarity(visual, target_semantic, dim=-1).cpu())
                acc["paired_prompt_semantic"].append(F.cosine_similarity(fused_mu, target_semantic, dim=-1).cpu())
                acc["paired_prompt_visual"].append(F.cosine_similarity(fused_mu, visual, dim=-1).cpu())
                for key in ("candidate_entropy", "candidate_top1_mass", "posterior_shift", "variance_ratio"):
                    acc[key].append(fusion_stats[key].detach().cpu())
                ones = torch.ones(labels.shape[0], device=runtime_device, dtype=torch.float64)
                acc["count"].index_add_(0, labels, ones)
                acc["prompt_sum"].index_add_(0, labels, fused_mu.double())
                acc["visual_sum"].index_add_(0, labels, visual.double())
                acc["logit_sum"].index_add_(0, labels, logits.double())
                acc["semantic_repr"] = semantic_bank.detach().cpu().numpy().astype(np.float64)

    rows: List[Dict[str, Any]] = []
    for group in GROUPS:
        acc = accumulators[group]
        labels = np.concatenate(acc["labels"])
        predictions = np.concatenate(acc["predictions"])
        zsl_predictions = np.concatenate(acc["zsl_predictions"])
        split = np.asarray(acc["split"])
        seen_mask = split == "seen"
        unseen_mask = split == "unseen"
        seen_accuracy = _macro_accuracy(predictions[seen_mask], labels[seen_mask], seen_ids)
        unseen_accuracy = _macro_accuracy(predictions[unseen_mask], labels[unseen_mask], unseen_ids)
        harmonic = 0.0 if seen_accuracy + unseen_accuracy <= 0 else 2.0 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy)
        zsl_unseen = _macro_accuracy(zsl_predictions[unseen_mask], labels[unseen_mask], unseen_ids)
        count = acc["count"]
        if bool((count <= 0).any().item()):
            raise RuntimeError(f"Stage-2B evaluation did not cover all classes for {group}.")
        prompt_centers = (acc["prompt_sum"] / count[:, None]).cpu().numpy()
        visual_centers = (acc["visual_sum"] / count[:, None]).cpu().numpy()
        logit_profiles = (acc["logit_sum"] / count[:, None]).cpu().numpy()
        row: Dict[str, Any] = {
            "model_seed": int(args.seed),
            "group": group,
            "heldout_gzsl_seen": seen_accuracy,
            "heldout_gzsl_unseen": unseen_accuracy,
            "heldout_gzsl_h": harmonic,
            "heldout_zsl_unseen": zsl_unseen,
            "paired_visual_semantic_cosine": float(torch.cat(acc["paired_visual_semantic"]).mean().item()),
            "paired_prompt_semantic_cosine": float(torch.cat(acc["paired_prompt_semantic"]).mean().item()),
            "paired_prompt_visual_cosine": float(torch.cat(acc["paired_prompt_visual"]).mean().item()),
            "candidate_entropy": float(torch.cat(acc["candidate_entropy"]).mean().item()) if group != "B0_image_only" else float("nan"),
            "candidate_top1_mass": float(torch.cat(acc["candidate_top1_mass"]).mean().item()) if group != "B0_image_only" else float("nan"),
            "posterior_shift": float(torch.cat(acc["posterior_shift"]).mean().item()),
            "variance_ratio": float(torch.cat(acc["variance_ratio"]).mean().item()),
        }
        row.update(
            _relation_metrics(
                prompt_centers,
                visual_centers,
                logit_profiles,
                np.asarray(acc["semantic_repr"]),
                graph,
                args.eps,
            )
        )
        if group in banks:
            row.update(
                _prior_quality(
                    banks[group][0].astype(np.float64),
                    banks[group][1].astype(np.float64),
                    oracle_mu_np,
                    oracle_logvar_np,
                    unseen_ids,
                    args.eps,
                )
            )
        rows.append(row)

    output_metadata = stage2_metadata("stage2B_prior_intervention", seed=int(args.seed))
    _write_csv(args.output_dir / "group_results.csv", rows, output_metadata)
    np.savez_compressed(
        str(args.output_dir / "prior_banks.npz"),
        **{f"{group}_mu": values[0] for group, values in banks.items()},
        **{f"{group}_logvar": values[1] for group, values in banks.items()},
    )
    metadata = {
        "format": "stage2b_prior_intervention_results_v1",
        "seed": int(args.seed),
        "groups": list(GROUPS),
        "oracle": str(args.oracle),
        "checkpoint": str(args.checkpoint),
        "fusion_alpha": float(args.fusion_alpha),
        "energy_beta": float(args.energy_beta),
        "candidate_temperature": float(args.candidate_temperature),
        "evaluation_protocol": "test_seen plus the held-out 20 percent of test_unseen oracle images",
        "leakage_contract": {
            "B1_B2_B3": "unseen oracle rows are not read while constructing predicted prior banks",
            "B4": "all-class oracle bank; oracle upper bound only",
        },
    }
    write_stage2_json(
        args.output_dir / "results.json",
        metadata,
        output_metadata,
    )
    print(f"wrote {args.output_dir / 'group_results.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2B B0-B4 prior interventions.")
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "prompt" / "cub.yaml")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prior-ridge", type=float, default=1e-3)
    parser.add_argument("--fusion-alpha", type=float, default=1.0)
    parser.add_argument("--energy-beta", type=float, default=1.0)
    parser.add_argument("--candidate-temperature", type=float, default=1.0)
    parser.add_argument("--variance-min", type=float, default=1e-4)
    parser.add_argument("--variance-max", type=float, default=100.0)
    parser.add_argument("--logvar-min", type=float, default=-10.0)
    parser.add_argument("--logvar-max", type=float, default=5.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    args.config_file = args.config_file.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.oracle = args.oracle.resolve()
    args.graph_bundle = args.graph_bundle.resolve()
    args.output_dir = args.output_dir.resolve()
    for path in (args.config_file, args.checkpoint, args.oracle, args.graph_bundle):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    positive = (
        args.batch_size,
        args.prior_ridge,
        args.fusion_alpha,
        args.candidate_temperature,
        args.variance_min,
        args.variance_max,
        args.eps,
    )
    if args.seed < 0 or args.num_workers < 0 or min(positive) <= 0 or args.energy_beta < 0:
        parser.error("Seed/worker/fusion parameters are invalid.")
    if args.variance_max < args.variance_min or args.logvar_max < args.logvar_min:
        parser.error("Variance/logvar bounds are invalid.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
