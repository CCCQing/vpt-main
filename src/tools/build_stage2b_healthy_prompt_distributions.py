#!/usr/bin/env python3
"""Build seen-only Stage-2B prompt banks with visual-moment constraints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets.xlsa_dataset import CUB200Dataset  # noqa: E402
from src.data.transforms import get_transforms  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.tools.evaluate_graph_gp_center_transfer import (  # noqa: E402
    _attribute_ridge_predict,
    _pairwise_distance_values,
    _safe_spearman,
)
from src.tools.export_prompt_posterior_cache import (  # noqa: E402
    _load_trainable_checkpoint,
    _setup_cfg,
)
from src.utils import logging  # noqa: E402


VARIANTS = ("D0_empirical", "D1_moment", "D2_task")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _clean_kernel(matrix: np.ndarray, eps: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.maximum(0.5 * (matrix + matrix.T), 0.0)
    diagonal = np.maximum(np.diag(matrix), eps)
    matrix = matrix / np.sqrt(diagonal[:, None] * diagonal[None, :])
    matrix = np.clip(0.5 * (matrix + matrix.T), 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _load_graph(path: Path, key: str, eps: float) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"Graph bundle has no '{key}': {path}")
        return _clean_kernel(payload[key], eps)


def _graph_predict(
    kernel: np.ndarray,
    support_ids: np.ndarray,
    query_ids: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> np.ndarray:
    system = kernel[np.ix_(support_ids, support_ids)] + float(ridge) * np.eye(
        support_ids.size, dtype=np.float64
    )
    support_target = np.asarray(target, dtype=np.float64)[support_ids]
    mean = support_target.mean(axis=0, keepdims=True)
    solved = np.linalg.solve(system, support_target - mean)
    return mean + kernel[np.ix_(query_ids, support_ids)] @ solved


def _graph_predictive_variance(
    kernel: np.ndarray,
    support_ids: np.ndarray,
    query_ids: np.ndarray,
    ridge: float,
) -> np.ndarray:
    system = kernel[np.ix_(support_ids, support_ids)] + float(ridge) * np.eye(
        support_ids.size, dtype=np.float64
    )
    query_support = kernel[np.ix_(query_ids, support_ids)]
    solved = np.linalg.solve(system, query_support.T)
    reduction = np.sum(query_support * solved.T, axis=1)
    return np.maximum(np.diag(kernel[np.ix_(query_ids, query_ids)]) - reduction, 0.0)


def _propagated_variance(
    kernel: np.ndarray,
    support_ids: np.ndarray,
    query_ids: np.ndarray,
    support_variance: np.ndarray,
    eps: float,
) -> np.ndarray:
    weights = np.maximum(kernel[np.ix_(query_ids, support_ids)], 0.0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), eps)
    return weights @ np.asarray(support_variance, dtype=np.float64)[support_ids]


def _predict_variance(
    kernel: np.ndarray,
    support_ids: np.ndarray,
    query_ids: np.ndarray,
    support_logvar: np.ndarray,
    ridge: float,
    gp_var_weight: float,
    support_var_weight: float,
    logvar_min: float,
    logvar_max: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    support_variance = _propagated_variance(
        kernel,
        support_ids,
        query_ids,
        np.exp(np.clip(support_logvar, logvar_min, logvar_max)),
        eps,
    )
    gp_uncertainty = _graph_predictive_variance(
        kernel, support_ids, query_ids, ridge
    )[:, None]
    variance = (
        float(support_var_weight) * support_variance
        + float(gp_var_weight) * gp_uncertainty
    )
    return np.maximum(variance, eps), gp_uncertainty[:, 0]


def _complete_bank(
    seen_mu: np.ndarray,
    seen_logvar: np.ndarray,
    seen_ids: np.ndarray,
    unseen_ids: np.ndarray,
    kernel: np.ndarray,
    ridge: float,
    gp_var_weight: float,
    support_var_weight: float,
    logvar_min: float,
    logvar_max: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_classes, latent_dim = seen_mu.shape
    mu = np.zeros((num_classes, latent_dim), dtype=np.float64)
    logvar = np.zeros_like(mu)
    mu[seen_ids] = seen_mu[seen_ids]
    logvar[seen_ids] = seen_logvar[seen_ids]
    mu[unseen_ids] = _graph_predict(kernel, seen_ids, unseen_ids, seen_mu, ridge)
    unseen_variance, gp_uncertainty = _predict_variance(
        kernel,
        seen_ids,
        unseen_ids,
        seen_logvar,
        ridge,
        gp_var_weight,
        support_var_weight,
        logvar_min,
        logvar_max,
        eps,
    )
    logvar[unseen_ids] = np.log(unseen_variance)
    logvar = np.clip(logvar, logvar_min, logvar_max)
    uncertainty = np.zeros(num_classes, dtype=np.float64)
    uncertainty[unseen_ids] = gp_uncertainty
    return mu.astype(np.float32), logvar.astype(np.float32), uncertainty.astype(np.float32)


class ClassBalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        labels: Sequence[int],
        class_ids: Sequence[int],
        classes_per_batch: int,
        samples_per_class: int,
        seed: int,
    ) -> None:
        self.class_ids = np.asarray(list(class_ids), dtype=np.int64)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        labels_array = np.asarray(list(labels), dtype=np.int64)
        self.indices = {
            int(class_id): np.flatnonzero(labels_array == int(class_id)).astype(np.int64)
            for class_id in self.class_ids
        }
        if any(values.size == 0 for values in self.indices.values()):
            raise ValueError("Class-balanced sampler received an empty class.")
        batch_size = self.classes_per_batch * self.samples_per_class
        self.batch_count = max(1, int(math.ceil(labels_array.size / float(batch_size))))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + 1009 * self.epoch)
        for _ in range(self.batch_count):
            chosen_classes = rng.choice(
                self.class_ids,
                size=self.classes_per_batch,
                replace=self.class_ids.size < self.classes_per_batch,
            )
            batch: List[int] = []
            for class_id in chosen_classes:
                candidates = self.indices[int(class_id)]
                selected = rng.choice(
                    candidates,
                    size=self.samples_per_class,
                    replace=candidates.size < self.samples_per_class,
                )
                batch.extend(int(index) for index in selected)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batch_count


class SharedPromptDistribution(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        latent_dim: int,
        mu_delta_scale: float,
        logvar_delta_scale: float,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2 * int(latent_dim)),
        )
        self.mu_delta_scale = float(mu_delta_scale)
        self.logvar_delta_scale = float(logvar_delta_scale)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(
        self,
        features: torch.Tensor,
        reference_mu: torch.Tensor,
        reference_logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_mu, delta_logvar = self.network(features).chunk(2, dim=-1)
        mu = reference_mu + self.mu_delta_scale * torch.tanh(delta_mu)
        logvar = reference_logvar + self.logvar_delta_scale * torch.tanh(delta_logvar)
        return mu, logvar.clamp(min=self.logvar_min, max=self.logvar_max)


def _seen_dataset(cfg) -> CUB200Dataset:
    dataset = CUB200Dataset(cfg, "trainval")
    dataset.transform = get_transforms("stage2_extract", int(cfg.DATA.CROPSIZE))
    labels = {int(record["class"]) for record in dataset._imdb}
    expected = set(int(class_id) for class_id in dataset.seen_classes)
    if labels != expected:
        raise ValueError(f"Seen trainval classes mismatch: missing={sorted(expected.difference(labels))}")
    return dataset


def _load_statistics_cache(
    path: Path,
    seed: int,
    seen_ids: np.ndarray,
    device: torch.device,
    num_classes: int,
    latent_dim: int,
    variance_floor: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], np.ndarray]:
    required = {
        "global_labels",
        "posterior_mu",
        "posterior_logvar",
        "classifier_visual_repr",
        "classifier_semantic_repr",
        "classifier_class_ids",
        "class_attributes",
        "seen_class_ids",
        "unseen_class_ids",
        "metadata_json",
    }
    with np.load(str(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"A00 posterior cache is missing {missing}: {path}")
        arrays = {key: payload[key] for key in required}
    metadata = json.loads(str(arrays.pop("metadata_json").item()))
    if (
        metadata.get("format") != "graph_gp_posterior_cache_v2"
        or metadata.get("split") != "trainval"
        or metadata.get("cell_id") != "A00"
        or str(metadata.get("eval_sample_mode", "")).lower() != "mean"
        or int(metadata.get("seed", -1)) != int(seed)
    ):
        raise ValueError(f"Posterior cache is not the required A00 trainval seed {seed}: {path}")
    labels = np.asarray(arrays["global_labels"], dtype=np.int64)
    prompt_mu = np.asarray(arrays["posterior_mu"], dtype=np.float64)
    prompt_logvar = np.asarray(arrays["posterior_logvar"], dtype=np.float64)
    visual = np.asarray(arrays["classifier_visual_repr"], dtype=np.float64)
    semantic = np.asarray(arrays["classifier_semantic_repr"], dtype=np.float64)
    cached_class_ids = np.asarray(arrays["classifier_class_ids"], dtype=np.int64)
    cached_seen = np.asarray(arrays["seen_class_ids"], dtype=np.int64)
    cached_unseen = np.asarray(arrays["unseen_class_ids"], dtype=np.int64)
    if not np.array_equal(np.sort(cached_seen), np.sort(seen_ids)):
        raise ValueError("A00 cache seen class ids do not match the current dataset.")
    if not np.array_equal(np.sort(cached_class_ids), np.sort(seen_ids)):
        raise ValueError("A00 trainval cache classifier class ids do not exactly cover seen classes.")
    if (
        np.intersect1d(cached_seen, cached_unseen).size
        or not np.array_equal(
            np.sort(np.concatenate((cached_seen, cached_unseen))),
            np.arange(num_classes, dtype=np.int64),
        )
    ):
        raise ValueError("A00 cache seen/unseen ids are not a disjoint global class partition.")
    if set(np.unique(labels).tolist()) != set(seen_ids.tolist()):
        raise ValueError("A00 trainval cache contains an unexpected class set.")
    if (
        prompt_mu.shape != prompt_logvar.shape
        or prompt_mu.shape != visual.shape
        or prompt_mu.ndim != 2
        or prompt_mu.shape[1] != latent_dim
        or semantic.shape != (cached_class_ids.size, latent_dim)
    ):
        raise ValueError(
            f"A00 cache dimensions are invalid: mu={prompt_mu.shape} logvar={prompt_logvar.shape} "
            f"visual={visual.shape} semantic={semantic.shape}."
        )
    count = np.bincount(labels, minlength=num_classes).astype(np.float64)
    prompt_sum = np.zeros((num_classes, latent_dim), dtype=np.float64)
    prompt_second = np.zeros_like(prompt_sum)
    visual_sum = np.zeros_like(prompt_sum)
    visual_second = np.zeros_like(prompt_sum)
    np.add.at(prompt_sum, labels, prompt_mu)
    np.add.at(prompt_second, labels, np.exp(prompt_logvar) + prompt_mu * prompt_mu)
    np.add.at(visual_sum, labels, visual)
    np.add.at(visual_second, labels, visual * visual)
    if np.any(count[seen_ids] <= 0):
        raise RuntimeError("Seen statistics contain an empty class.")
    divisor = np.maximum(count, 1.0)[:, None]
    prompt_mean = prompt_sum / divisor
    prompt_variance = np.maximum(
        prompt_second / divisor - prompt_mean * prompt_mean, variance_floor
    )
    visual_mean = visual_sum / divisor
    visual_variance = np.maximum(
        visual_second / divisor - visual_mean * visual_mean, variance_floor
    )
    semantic_global = np.zeros((num_classes, latent_dim), dtype=np.float64)
    semantic_global[cached_class_ids] = semantic
    stats = {
        "count": torch.from_numpy(count.astype(np.float32)).to(device),
        "prompt_mean": torch.from_numpy(prompt_mean.astype(np.float32)).to(device),
        "prompt_logvar": torch.from_numpy(np.log(prompt_variance).astype(np.float32)).to(device),
        "visual_mean": torch.from_numpy(visual_mean.astype(np.float32)).to(device),
        "visual_logvar": torch.from_numpy(np.log(visual_variance).astype(np.float32)).to(device),
        "semantic_repr": torch.from_numpy(semantic_global.astype(np.float32)).to(device),
    }
    return stats, metadata, np.asarray(arrays["class_attributes"], dtype=np.float32)


def _conditioning_features(
    stats: Mapping[str, torch.Tensor],
    graph: torch.Tensor,
    seen_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.cat(
        (
            F.normalize(stats["visual_mean"], dim=-1),
            stats["visual_logvar"],
            F.normalize(stats["semantic_repr"], dim=-1),
            F.normalize(stats["prompt_mean"], dim=-1),
            stats["prompt_logvar"],
            graph,
        ),
        dim=-1,
    )
    seen_raw = raw.index_select(0, seen_ids)
    feature_mean = seen_raw.mean(dim=0, keepdim=True)
    feature_std = seen_raw.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    return (raw - feature_mean) / feature_std, feature_mean, feature_std


def _class_moment_losses(
    visual: torch.Tensor,
    labels: torch.Tensor,
    target_mean: torch.Tensor,
    target_logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_terms: List[torch.Tensor] = []
    variance_terms: List[torch.Tensor] = []
    for class_id in labels.unique(sorted=True):
        current = visual[labels == class_id]
        predicted_mean = current.mean(dim=0)
        predicted_variance = current.var(dim=0, unbiased=False).clamp_min(1e-6)
        mean_terms.append(1.0 - F.cosine_similarity(
            predicted_mean[None], target_mean[class_id][None], dim=-1
        ).mean())
        variance_terms.append(F.smooth_l1_loss(
            predicted_variance.log(), target_logvar[class_id]
        ))
    return torch.stack(mean_terms).mean(), torch.stack(variance_terms).mean()


def _train_variant(
    name: str,
    model,
    dataset,
    class_ids: torch.Tensor,
    seen_ids: torch.Tensor,
    features: torch.Tensor,
    stats: Mapping[str, torch.Tensor],
    graph: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    ce_weight: float,
    seed_offset: int,
) -> Tuple[SharedPromptDistribution, List[Dict[str, float]]]:
    torch.manual_seed(int(args.seed) + int(seed_offset))
    generator = SharedPromptDistribution(
        features.shape[1],
        args.generator_hidden_dim,
        stats["prompt_mean"].shape[1],
        args.mu_delta_scale,
        args.logvar_delta_scale,
        args.logvar_min,
        args.logvar_max,
    ).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(args.lr))
    labels = [int(record["class"]) for record in dataset._imdb]
    sampler = ClassBalancedBatchSampler(
        labels,
        seen_ids.detach().cpu().tolist(),
        args.classes_per_batch,
        args.samples_per_class,
        args.seed + seed_offset,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    history: List[Dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        sampler.set_epoch(epoch)
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "mean": 0.0,
            "variance": 0.0,
            "semantic": 0.0,
            "prompt_visual": 0.0,
            "prompt_semantic": 0.0,
            "anchor": 0.0,
            "graph": 0.0,
            "batches": 0.0,
        }
        for batch in loader:
            images = batch["image"].to(device=device, non_blocking=True)
            batch_labels = batch["label"].to(device=device, dtype=torch.long)
            mu, logvar = generator(
                features.index_select(0, batch_labels),
                stats["prompt_mean"].index_select(0, batch_labels),
                stats["prompt_logvar"].index_select(0, batch_labels),
            )
            logits_samples: List[torch.Tensor] = []
            visual_samples: List[torch.Tensor] = []
            for _ in range(int(args.moment_samples)):
                eps = torch.randn(
                    batch_labels.shape[0],
                    int(args.instance_tokens),
                    mu.shape[1],
                    device=device,
                    dtype=mu.dtype,
                )
                model.set_runtime_prompt_distribution_override(mu, logvar, eps=eps)
                try:
                    logits_samples.append(
                        model(images, semantics=None, class_ids=class_ids, runtime_targets=None)
                    )
                finally:
                    model.clear_runtime_prompt_distribution_override()
                visual_samples.append(model.get_runtime_classifier_stats()["visual_repr"])
            logits = torch.stack(logits_samples, dim=0)
            visual = torch.stack(visual_samples, dim=0)
            expanded_labels = batch_labels.repeat(int(args.moment_samples))
            flat_visual = visual.reshape(-1, visual.shape[-1])
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), expanded_labels)
            mean_loss, variance_loss = _class_moment_losses(
                flat_visual,
                expanded_labels,
                stats["visual_mean"],
                stats["visual_logvar"],
            )
            target_semantic = stats["semantic_repr"].index_select(0, expanded_labels)
            semantic_loss = (1.0 - F.cosine_similarity(flat_visual, target_semantic, dim=-1)).mean()
            unique_labels = batch_labels.unique(sorted=True)
            unique_mu, unique_logvar = generator(
                features.index_select(0, unique_labels),
                stats["prompt_mean"].index_select(0, unique_labels),
                stats["prompt_logvar"].index_select(0, unique_labels),
            )
            prompt_visual = (
                1.0
                - F.cosine_similarity(
                    unique_mu,
                    stats["visual_mean"].index_select(0, unique_labels),
                    dim=-1,
                )
            ).mean()
            prompt_semantic = (
                1.0
                - F.cosine_similarity(
                    unique_mu,
                    stats["semantic_repr"].index_select(0, unique_labels),
                    dim=-1,
                )
            ).mean()
            anchor = F.mse_loss(
                unique_mu, stats["prompt_mean"].index_select(0, unique_labels)
            ) + F.mse_loss(
                unique_logvar, stats["prompt_logvar"].index_select(0, unique_labels)
            )
            prompt_relation = F.normalize(unique_mu, dim=-1) @ F.normalize(unique_mu, dim=-1).t()
            graph_target = graph.index_select(0, unique_labels).index_select(1, unique_labels)
            graph_loss = F.mse_loss(prompt_relation, graph_target)
            loss = (
                float(ce_weight) * ce
                + float(args.mean_weight) * mean_loss
                + float(args.variance_weight) * variance_loss
                + float(args.semantic_weight) * semantic_loss
                + float(args.prompt_visual_weight) * prompt_visual
                + float(args.prompt_semantic_weight) * prompt_semantic
                + float(args.anchor_weight) * anchor
                + float(args.graph_weight) * graph_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            values = {
                "loss": loss,
                "ce": ce,
                "mean": mean_loss,
                "variance": variance_loss,
                "semantic": semantic_loss,
                "prompt_visual": prompt_visual,
                "prompt_semantic": prompt_semantic,
                "anchor": anchor,
                "graph": graph_loss,
            }
            for key, value in values.items():
                totals[key] += float(value.detach().item())
            totals["batches"] += 1.0
        row = {"variant": name, "epoch": float(epoch + 1)}
        row.update({key: value / totals["batches"] for key, value in totals.items() if key != "batches"})
        history.append(row)
        print(
            "[stage2b-healthy] variant={} epoch={}/{} loss={:.6f} ce={:.6f} mean={:.6f} "
            "variance={:.6f} semantic={:.6f} prompt_visual={:.6f} prompt_semantic={:.6f} "
            "anchor={:.6f} graph={:.6f}".format(
                name,
                epoch + 1,
                args.epochs,
                row["loss"],
                row["ce"],
                row["mean"],
                row["variance"],
                row["semantic"],
                row["prompt_visual"],
                row["prompt_semantic"],
                row["anchor"],
                row["graph"],
            ),
            flush=True,
        )
    return generator, history


def _cosine_rows(left: np.ndarray, right: np.ndarray, eps: float) -> np.ndarray:
    return np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), eps
    )


def _pseudo_unseen_rows(
    banks: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    seen_ids: np.ndarray,
    graph: np.ndarray,
    class_attributes: np.ndarray,
    seed: int,
    fold_count: int,
    ridge: float,
    gp_var_weight: float,
    support_var_weight: float,
    logvar_min: float,
    logvar_max: float,
    eps: float,
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(int(seed) + 2027)
    folds = np.array_split(rng.permutation(seen_ids), int(fold_count))
    shuffled_rng = np.random.RandomState(int(seed) + 200003)
    permutation = shuffled_rng.permutation(graph.shape[0])
    shuffled = graph[np.ix_(permutation, permutation)]
    rows: List[Dict[str, Any]] = []
    for fold_index, query_ids in enumerate(folds):
        support_ids = np.asarray(sorted(set(seen_ids.tolist()).difference(query_ids.tolist())), dtype=np.int64)
        for variant, (truth_mu, truth_logvar) in banks.items():
            support_mean = truth_mu[support_ids].mean(axis=0, keepdims=True)
            support_variance_mean = np.exp(
                np.clip(truth_logvar[support_ids], logvar_min, logvar_max)
            ).mean(axis=0, keepdims=True)
            methods: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
            for method_name, current_graph in (("graph_gp_real", graph), ("graph_gp_shuffled", shuffled)):
                predicted_mu = _graph_predict(
                    current_graph, support_ids, query_ids, truth_mu, ridge
                )
                predicted_variance, gp_uncertainty = _predict_variance(
                    current_graph,
                    support_ids,
                    query_ids,
                    truth_logvar,
                    ridge,
                    gp_var_weight,
                    support_var_weight,
                    logvar_min,
                    logvar_max,
                    eps,
                )
                methods.append(
                    (method_name, predicted_mu, predicted_variance, float(gp_uncertainty.mean()))
                )
            attribute_mu = _attribute_ridge_predict(
                class_attributes,
                support_ids,
                query_ids,
                truth_mu[support_ids],
                ridge,
            )
            attribute_logvar = _attribute_ridge_predict(
                class_attributes,
                support_ids,
                query_ids,
                truth_logvar[support_ids],
                ridge,
            )
            methods.append(
                (
                    "attribute_ridge",
                    attribute_mu,
                    np.exp(np.clip(attribute_logvar, logvar_min, logvar_max)),
                    float("nan"),
                )
            )
            methods.append(
                (
                    "seen_mean",
                    np.repeat(support_mean, query_ids.size, axis=0),
                    np.repeat(support_variance_mean, query_ids.size, axis=0),
                    float("nan"),
                )
            )
            for method_name, predicted_mu, predicted_variance, gp_uncertainty_mean in methods:
                truth = truth_mu[query_ids]
                truth_variance = np.exp(
                    np.clip(truth_logvar[query_ids], logvar_min, logvar_max)
                )
                centered_prediction = predicted_mu - support_mean
                centered_truth = truth - support_mean
                error = np.linalg.norm(predicted_mu - truth, axis=1)
                denom = np.maximum(np.linalg.norm(centered_truth, axis=1), eps)
                predicted_logvar = np.log(np.maximum(predicted_variance, eps))
                truth_logvar_query = np.log(np.maximum(truth_variance, eps))
                predicted_nearest = np.linalg.norm(
                    predicted_mu[:, None, :] - truth_mu[support_ids][None, :, :], axis=-1
                ).argmin(axis=1)
                truth_nearest = np.linalg.norm(
                    truth[:, None, :] - truth_mu[support_ids][None, :, :], axis=-1
                ).argmin(axis=1)
                rows.append(
                    {
                        "model_seed": int(seed),
                        "fold": int(fold_index),
                        "variant": variant,
                        "method": method_name,
                        "support_classes": int(support_ids.size),
                        "query_classes": int(query_ids.size),
                        "raw_cosine": float(_cosine_rows(predicted_mu, truth, eps).mean()),
                        "centered_cosine": float(
                            _cosine_rows(centered_prediction, centered_truth, eps).mean()
                        ),
                        "nrmse": float((error / denom).mean()),
                        "distance_spearman": _safe_spearman(
                            _pairwise_distance_values(predicted_mu),
                            _pairwise_distance_values(truth),
                        ),
                        "nearest_support_retention": float(
                            (predicted_nearest == truth_nearest).mean()
                        ),
                        "logvar_rmse": float(
                            np.sqrt(np.mean((predicted_logvar - truth_logvar_query) ** 2))
                        ),
                        "variance_nrmse": float(
                            np.linalg.norm(predicted_variance - truth_variance)
                            / max(np.linalg.norm(truth_variance), eps)
                        ),
                        "variance_rank_spearman": _safe_spearman(
                            predicted_variance.mean(axis=1), truth_variance.mean(axis=1)
                        ),
                        "prediction_interval_coverage": float(
                            (
                                np.abs(truth - predicted_mu)
                                <= 1.96 * np.sqrt(np.maximum(predicted_variance, eps))
                            ).mean()
                        ),
                        "gp_uncertainty_mean": gp_uncertainty_mean,
                    }
                )
    return rows


def build(args: argparse.Namespace) -> None:
    setup_args = argparse.Namespace(
        config_file=args.config_file,
        cell="A00",
        seed=args.seed,
        batch_size=1,
        num_workers=args.num_workers,
        output=args.output,
        opts=[],
    )
    cfg, _ = _setup_cfg(setup_args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.setup_logging(1, 1, output=str(args.output.parent), name="visual_prompt")
    dataset = _seen_dataset(cfg)
    class_attributes = dataset.class_attributes
    if torch.is_tensor(class_attributes):
        class_attributes_np = class_attributes.detach().cpu().numpy().astype(np.float32)
    else:
        class_attributes_np = np.asarray(class_attributes, dtype=np.float32)
    model, device = build_model(cfg)
    runtime_device = torch.device(f"cuda:{int(device)}") if isinstance(device, int) else torch.device(device)
    model.attach_r_similarity_head(class_attributes)
    checkpoint = _load_trainable_checkpoint(model, args.checkpoint, args.seed, "A00")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    num_classes = int(cfg.DATA.NUMBER_CLASSES)
    latent_dim = int(cfg.MODEL.GRAPH_INPUT.TEXT_DIM)
    class_ids = torch.arange(num_classes, device=runtime_device, dtype=torch.long)
    seen_np = np.asarray(dataset.seen_classes, dtype=np.int64)
    unseen_np = np.asarray(dataset.unseen_classes, dtype=np.int64)
    seen = torch.from_numpy(seen_np).to(runtime_device)
    stats, cache_metadata, cached_class_attributes = _load_statistics_cache(
        args.posterior_cache,
        args.seed,
        seen_np,
        runtime_device,
        num_classes,
        latent_dim,
        args.variance_floor,
    )
    if not np.allclose(class_attributes_np, cached_class_attributes, rtol=0.0, atol=1e-6):
        raise ValueError("A00 posterior cache class attributes do not match the current dataset.")
    args.instance_tokens = int(cfg.MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS)
    graph_np = _load_graph(args.graph_bundle, args.graph_key, args.eps)
    graph = torch.from_numpy(graph_np.astype(np.float32)).to(runtime_device)
    features, feature_mean, feature_std = _conditioning_features(stats, graph, seen)
    d1_generator, d1_history = _train_variant(
        "D1_moment",
        model,
        dataset,
        class_ids,
        seen,
        features,
        stats,
        graph,
        runtime_device,
        args,
        ce_weight=0.0,
        seed_offset=11000,
    )
    d2_generator, d2_history = _train_variant(
        "D2_task",
        model,
        dataset,
        class_ids,
        seen,
        features,
        stats,
        graph,
        runtime_device,
        args,
        ce_weight=args.ce_weight,
        seed_offset=11000,
    )
    with torch.no_grad():
        d1_mu, d1_logvar = d1_generator(features, stats["prompt_mean"], stats["prompt_logvar"])
        d2_mu, d2_logvar = d2_generator(features, stats["prompt_mean"], stats["prompt_logvar"])
    seen_banks = {
        "D0_empirical": (
            stats["prompt_mean"].cpu().numpy().astype(np.float64),
            stats["prompt_logvar"].cpu().numpy().astype(np.float64),
        ),
        "D1_moment": (
            d1_mu.cpu().numpy().astype(np.float64),
            d1_logvar.cpu().numpy().astype(np.float64),
        ),
        "D2_task": (
            d2_mu.cpu().numpy().astype(np.float64),
            d2_logvar.cpu().numpy().astype(np.float64),
        ),
    }
    rng = np.random.RandomState(int(args.seed) + 200003)
    permutation = rng.permutation(num_classes)
    shuffled_graph = graph_np[np.ix_(permutation, permutation)]
    output_arrays: Dict[str, np.ndarray] = {}
    bank_metadata: Dict[str, Any] = {}
    for variant, (seen_mu, seen_logvar) in seen_banks.items():
        real_mu, real_logvar, real_uncertainty = _complete_bank(
            seen_mu,
            seen_logvar,
            seen_np,
            unseen_np,
            graph_np,
            args.prior_ridge,
            args.gp_var_weight,
            args.support_var_weight,
            args.logvar_min,
            args.logvar_max,
            args.eps,
        )
        shuffled_mu, shuffled_logvar, shuffled_uncertainty = _complete_bank(
            seen_mu,
            seen_logvar,
            seen_np,
            unseen_np,
            shuffled_graph,
            args.prior_ridge,
            args.gp_var_weight,
            args.support_var_weight,
            args.logvar_min,
            args.logvar_max,
            args.eps,
        )
        key = variant.lower()
        output_arrays[f"{key}_real_mu"] = real_mu
        output_arrays[f"{key}_real_logvar"] = real_logvar
        output_arrays[f"{key}_real_gp_uncertainty"] = real_uncertainty
        output_arrays[f"{key}_shuffled_mu"] = shuffled_mu
        output_arrays[f"{key}_shuffled_logvar"] = shuffled_logvar
        output_arrays[f"{key}_shuffled_gp_uncertainty"] = shuffled_uncertainty
        bank_metadata[variant] = {
            "ce_weight": 0.0 if variant != "D2_task" else float(args.ce_weight),
            "seen_source": "CUB trainval only",
            "unseen_source": "Graph-GP from seen bank and semantic graph only",
        }
    pseudo_rows = _pseudo_unseen_rows(
        seen_banks,
        seen_np,
        graph_np,
        class_attributes_np,
        args.seed,
        args.pseudo_folds,
        args.prior_ridge,
        args.gp_var_weight,
        args.support_var_weight,
        args.logvar_min,
        args.logvar_max,
        args.eps,
    )
    _write_csv(args.output.with_name(args.output.stem + "_pseudo_unseen.csv"), pseudo_rows)
    _write_csv(args.output.with_name(args.output.stem + "_history.csv"), d1_history + d2_history)
    torch.save(
        {
            "format": "stage2b_healthy_generator_state_v1",
            "seed": int(args.seed),
            "feature_mean": feature_mean.detach().cpu(),
            "feature_std": feature_std.detach().cpu(),
            "D1_moment": d1_generator.state_dict(),
            "D2_task": d2_generator.state_dict(),
        },
        str(args.output.with_suffix(".pth")),
    )
    metadata = {
        "format": "stage2b_healthy_prompt_distributions_v1",
        "seed": int(args.seed),
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_epoch": int(checkpoint.get("total_epoch", 0)),
        "posterior_cache": str(args.posterior_cache),
        "posterior_cache_sample_count": int(cache_metadata.get("sample_count", 0)),
        "graph_bundle": str(args.graph_bundle),
        "graph_key": str(args.graph_key),
        "variants": list(VARIANTS),
        "bank_metadata": bank_metadata,
        "training_protocol": "seen trainval only; frozen A00 ViT/classifier; no formal unseen images",
        "loss_weights": {
            "ce": float(args.ce_weight),
            "mean": float(args.mean_weight),
            "variance": float(args.variance_weight),
            "semantic": float(args.semantic_weight),
            "prompt_visual": float(args.prompt_visual_weight),
            "prompt_semantic": float(args.prompt_semantic_weight),
            "anchor": float(args.anchor_weight),
            "graph": float(args.graph_weight),
        },
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "moment_samples": int(args.moment_samples),
        "pseudo_folds": int(args.pseudo_folds),
        "gp_var_weight": float(args.gp_var_weight),
        "support_var_weight": float(args.support_var_weight),
    }
    np.savez_compressed(
        str(args.output),
        **output_arrays,
        seen_prompt_empirical_mu=stats["prompt_mean"].cpu().numpy().astype(np.float32),
        seen_prompt_empirical_logvar=stats["prompt_logvar"].cpu().numpy().astype(np.float32),
        seen_visual_mean=stats["visual_mean"].cpu().numpy().astype(np.float32),
        seen_visual_logvar=stats["visual_logvar"].cpu().numpy().astype(np.float32),
        semantic_repr=stats["semantic_repr"].cpu().numpy().astype(np.float32),
        class_attributes=class_attributes_np,
        graph=graph_np.astype(np.float32),
        seen_class_ids=seen_np,
        unseen_class_ids=unseen_np,
        seen_class_counts=stats["count"].cpu().numpy().astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-2B healthy prompt distributions from seen classes only.")
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "prompt" / "cub.yaml")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--posterior-cache", required=True, type=Path)
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--graph-key", default="method1_diff")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--generator-hidden-dim", type=int, default=256)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--moment-samples", type=int, default=1)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--prompt-visual-weight", type=float, default=0.1)
    parser.add_argument("--prompt-semantic-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--graph-weight", type=float, default=0.1)
    parser.add_argument("--mu-delta-scale", type=float, default=1.0)
    parser.add_argument("--logvar-delta-scale", type=float, default=1.0)
    parser.add_argument("--prior-ridge", type=float, default=1e-3)
    parser.add_argument("--gp-var-weight", type=float, default=1.0)
    parser.add_argument("--support-var-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-folds", type=int, default=5)
    parser.add_argument("--variance-floor", type=float, default=1e-6)
    parser.add_argument("--logvar-min", type=float, default=-10.0)
    parser.add_argument("--logvar-max", type=float, default=5.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    args.config_file = args.config_file.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.posterior_cache = args.posterior_cache.resolve()
    args.graph_bundle = args.graph_bundle.resolve()
    args.output = args.output.resolve()
    for path in (args.config_file, args.checkpoint, args.posterior_cache, args.graph_bundle):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    positive = (
        args.epochs,
        args.lr,
        args.generator_hidden_dim,
        args.classes_per_batch,
        args.samples_per_class,
        args.moment_samples,
        args.prior_ridge,
        args.pseudo_folds,
        args.variance_floor,
        args.eps,
    )
    weights = (
        args.ce_weight,
        args.mean_weight,
        args.variance_weight,
        args.semantic_weight,
        args.prompt_visual_weight,
        args.prompt_semantic_weight,
        args.anchor_weight,
        args.graph_weight,
        args.gp_var_weight,
        args.support_var_weight,
    )
    if args.seed < 0 or args.num_workers < 0 or min(positive) <= 0 or min(weights) < 0:
        parser.error("Stage-2B healthy distribution arguments are invalid.")
    if args.logvar_max < args.logvar_min:
        parser.error("--logvar-max must be >= --logvar-min.")
    return args


if __name__ == "__main__":
    build(parse_args())
