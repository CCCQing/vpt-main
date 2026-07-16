#!/usr/bin/env python3
"""Build per-class oracle-aligned prompt distributions with a frozen A00 model."""

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
from src.models.build_model import build_model  # noqa: E402
from src.monitoring.writer import stage2_metadata, write_stage2_json, write_stage2_table  # noqa: E402
from src.tools.export_prompt_posterior_cache import (  # noqa: E402
    _load_trainable_checkpoint,
    _setup_cfg,
)
from src.utils import logging  # noqa: E402


class FullCUBDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: Sequence[CUB200Dataset]):
        records: List[Mapping[str, Any]] = []
        seen_paths = set()
        for dataset in datasets:
            for record in dataset._imdb:
                path = str(record["im_path"])
                if path in seen_paths:
                    raise ValueError(f"Duplicate image across full-CUB splits: {path}")
                seen_paths.add(path)
                records.append(
                    {
                        "im_path": path,
                        "class": int(record["class"]),
                        "source_split": str(dataset.split_name),
                    }
                )
        self.records = records
        self.transform = datasets[0].transform
        self.class_attributes = datasets[0].class_attributes
        self.seen_classes = list(datasets[0].seen_classes)
        self.unseen_classes = list(datasets[0].unseen_classes)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        from torchvision.datasets.folder import default_loader

        record = self.records[index]
        return {
            "image": self.transform(default_loader(record["im_path"])),
            "label": int(record["class"]),
            "sample_id": int(index),
            "image_path": str(record["im_path"]),
        }


def _full_dataset(cfg) -> FullCUBDataset:
    transform = get_transforms("stage2_extract", int(cfg.DATA.CROPSIZE))
    datasets = [CUB200Dataset(cfg, split) for split in ("trainval", "test_seen", "test_unseen")]
    for dataset in datasets:
        dataset.transform = transform
    full = FullCUBDataset(datasets)
    labels = {int(record["class"]) for record in full.records}
    expected = set(range(int(cfg.DATA.NUMBER_CLASSES)))
    if labels != expected:
        raise ValueError(f"Full-CUB oracle pool must cover all classes: missing={sorted(expected.difference(labels))}")
    return full


def _split_indices(dataset: FullCUBDataset, seed: int, build_ratio: float) -> Tuple[List[int], List[int], Dict[str, Any]]:
    labels = np.asarray([int(record["class"]) for record in dataset.records], dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    build_ids: List[int] = []
    eval_ids: List[int] = []
    classes: Dict[str, Any] = {}
    seen_set = set(int(x) for x in dataset.seen_classes)
    for class_id in range(int(dataset.class_attributes.shape[0])):
        ids = np.flatnonzero(labels == class_id).astype(np.int64)
        if class_id in seen_set:
            current_build = sorted(
                int(x) for x in ids if dataset.records[int(x)]["source_split"] == "trainval"
            )
            current_eval = sorted(
                int(x) for x in ids if dataset.records[int(x)]["source_split"] == "test_seen"
            )
        else:
            if any(dataset.records[int(x)]["source_split"] != "test_unseen" for x in ids):
                raise ValueError(f"Unseen class {class_id} contains a non-test_unseen image.")
            ids = rng.permutation(ids)
            build_count = min(max(int(math.floor(float(build_ratio) * ids.size)), 1), ids.size - 1)
            current_build = sorted(int(x) for x in ids[:build_count])
            current_eval = sorted(int(x) for x in ids[build_count:])
        if not current_build or not current_eval:
            raise ValueError(f"Oracle class {class_id} has an empty build/eval split.")
        build_ids.extend(current_build)
        eval_ids.extend(current_eval)
        classes[str(class_id)] = {
            "build_ids": current_build,
            "eval_ids": current_eval,
            "build_paths": [str(dataset.records[index]["im_path"]) for index in current_build],
            "eval_paths": [str(dataset.records[index]["im_path"]) for index in current_eval],
        }
    manifest = {
        "format": "stage2b_oracle_image_split_v1",
        "seed": int(seed),
        "build_ratio": float(build_ratio),
        "sample_count": len(dataset),
        "seen_build_source": "trainval",
        "seen_eval_source": "test_seen",
        "unseen_source": "test_unseen split by build_ratio",
        "classes": classes,
    }
    return sorted(build_ids), sorted(eval_ids), manifest


def _loader(dataset, indices: Sequence[int], batch_size: int, num_workers: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, list(indices)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def _class_moments(
    model,
    loader,
    class_ids: torch.Tensor,
    device: torch.device,
    num_classes: int,
    latent_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = torch.zeros(num_classes, device=device, dtype=torch.float64)
    sum_mu = torch.zeros(num_classes, latent_dim, device=device, dtype=torch.float64)
    sum_second = torch.zeros_like(sum_mu)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, non_blocking=True)
            labels = batch["label"].to(device=device, dtype=torch.long)
            model(images, semantics=None, class_ids=class_ids, runtime_targets=None)
            stats = model.get_runtime_prompt_distribution_stats()
            mu = stats["mu"].double()
            variance = stats["logvar"].double().exp()
            ones = torch.ones(labels.shape[0], device=device, dtype=torch.float64)
            count.index_add_(0, labels, ones)
            sum_mu.index_add_(0, labels, mu)
            sum_second.index_add_(0, labels, variance + mu * mu)
    if bool((count <= 0).any().item()):
        raise RuntimeError("Oracle build split has an empty class.")
    mean = sum_mu / count[:, None]
    variance = torch.clamp(sum_second / count[:, None] - mean * mean, min=1e-6)
    return mean.float(), variance.log().float(), count.float()


def _macro_accuracy(predictions: np.ndarray, targets: np.ndarray, class_ids: Sequence[int]) -> float:
    values = []
    for class_id in class_ids:
        mask = targets == int(class_id)
        values.append(float((predictions[mask] == targets[mask]).mean()))
    return float(np.mean(values))


def _heldout_classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_ids: Sequence[int],
    seen_class_ids: Sequence[int],
    unseen_class_ids: Sequence[int],
) -> Dict[str, float]:
    class_array = np.asarray(list(class_ids), dtype=np.int64)
    seen_array = np.asarray(list(seen_class_ids), dtype=np.int64)
    unseen_array = np.asarray(list(unseen_class_ids), dtype=np.int64)
    if set(seen_array.tolist()).intersection(unseen_array.tolist()):
        raise ValueError("Seen and unseen class ids must be disjoint.")
    if set(seen_array.tolist()).union(unseen_array.tolist()) != set(class_array.tolist()):
        raise ValueError("Seen and unseen class ids must partition all classifier classes.")
    column_by_class = {int(class_id): index for index, class_id in enumerate(class_array)}
    unseen_columns = np.asarray(
        [column_by_class[int(class_id)] for class_id in unseen_array], dtype=np.int64
    )
    predictions = class_array[np.asarray(logits).argmax(axis=1)]
    seen_mask = np.isin(targets, seen_array)
    unseen_mask = np.isin(targets, unseen_array)
    if not seen_mask.any() or not unseen_mask.any():
        raise ValueError("Held-out evaluation must contain both seen and unseen samples.")
    seen_accuracy = _macro_accuracy(predictions[seen_mask], targets[seen_mask], seen_array)
    unseen_accuracy = _macro_accuracy(predictions[unseen_mask], targets[unseen_mask], unseen_array)
    harmonic = (
        0.0
        if seen_accuracy + unseen_accuracy <= 0.0
        else 2.0 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy)
    )
    unseen_logits = np.asarray(logits)[unseen_mask][:, unseen_columns]
    zsl_predictions = unseen_array[unseen_logits.argmax(axis=1)]
    zsl_unseen = _macro_accuracy(zsl_predictions, targets[unseen_mask], unseen_array)
    return {
        "heldout_gzsl_seen": float(seen_accuracy),
        "heldout_gzsl_unseen": float(unseen_accuracy),
        "heldout_gzsl_h": float(harmonic),
        "heldout_zsl_unseen": float(zsl_unseen),
    }


def _evaluate(
    model,
    loader,
    class_ids: torch.Tensor,
    seen_class_ids: Sequence[int],
    unseen_class_ids: Sequence[int],
    oracle_mu: torch.Tensor,
    oracle_logvar: torch.Tensor,
    device: torch.device,
    use_oracle: bool,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    labels_all: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    visual_sum = torch.zeros(
        class_ids.numel(), oracle_mu.shape[1], device=device, dtype=torch.float64
    )
    logit_sum = torch.zeros(
        class_ids.numel(), class_ids.numel(), device=device, dtype=torch.float64
    )
    count = torch.zeros(class_ids.numel(), device=device, dtype=torch.float64)
    paired_cosine: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, non_blocking=True)
            labels = batch["label"].to(device=device, dtype=torch.long)
            if use_oracle:
                model.set_runtime_prompt_distribution_override(
                    oracle_mu.index_select(0, labels), oracle_logvar.index_select(0, labels)
                )
            try:
                logits = model(images, semantics=None, class_ids=class_ids, runtime_targets=None)
            finally:
                model.clear_runtime_prompt_distribution_override()
            classifier = model.get_runtime_classifier_stats()
            visual = classifier["visual_repr"]
            semantic = classifier["semantic_repr"].index_select(0, labels)
            paired_cosine.append(F.cosine_similarity(visual, semantic, dim=-1).detach().cpu())
            labels_all.append(labels.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            ones = torch.ones(labels.shape[0], device=device, dtype=torch.float64)
            count.index_add_(0, labels, ones)
            visual_sum.index_add_(0, labels, visual.double())
            logit_sum.index_add_(0, labels, logits.double())
    labels_np = np.concatenate(labels_all)
    logits_np = np.concatenate(logits_all)
    class_list = [int(x) for x in class_ids.detach().cpu().tolist()]
    metrics = _heldout_classification_metrics(
        logits_np,
        labels_np,
        class_list,
        seen_class_ids,
        unseen_class_ids,
    )
    metrics["paired_visual_semantic_cosine"] = float(torch.cat(paired_cosine).mean().item())
    visual_centers = (visual_sum / count[:, None]).cpu().numpy().astype(np.float32)
    logit_profiles = (logit_sum / count[:, None]).cpu().numpy().astype(np.float32)
    return metrics, visual_centers, logit_profiles


def build_oracle(args: argparse.Namespace) -> None:
    setup_args = argparse.Namespace(
        config_file=args.config_file,
        cell="A00",
        seed=args.seed,
        batch_size=args.batch_size,
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
    dataset = _full_dataset(cfg)
    build_ids, eval_ids, split_manifest = _split_indices(dataset, args.seed, args.build_ratio)
    build_loader = _loader(dataset, build_ids, args.batch_size, args.num_workers, True, args.seed)
    moment_loader = _loader(dataset, build_ids, args.batch_size, args.num_workers, False, args.seed)
    eval_loader = _loader(dataset, eval_ids, args.batch_size, args.num_workers, False, args.seed)

    model, device = build_model(cfg)
    runtime_device = torch.device(f"cuda:{int(device)}") if isinstance(device, int) else torch.device(device)
    model.attach_r_similarity_head(dataset.class_attributes)
    checkpoint = _load_trainable_checkpoint(model, args.checkpoint, args.seed, "A00")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    class_ids = torch.arange(int(cfg.DATA.NUMBER_CLASSES), device=runtime_device, dtype=torch.long)
    latent_dim = int(cfg.MODEL.GRAPH_INPUT.TEXT_DIM)
    initial_mu, initial_logvar, build_count = _class_moments(
        model,
        moment_loader,
        class_ids,
        runtime_device,
        int(cfg.DATA.NUMBER_CLASSES),
        latent_dim,
    )
    initial_mu = initial_mu.to(runtime_device)
    initial_logvar = initial_logvar.to(runtime_device)
    oracle_mu = torch.nn.Parameter(initial_mu.clone())
    oracle_logvar = torch.nn.Parameter(initial_logvar.clone())
    optimizer = torch.optim.Adam([oracle_mu, oracle_logvar], lr=float(args.lr))
    history: List[Dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        totals = {"loss": 0.0, "ce": 0.0, "align": 0.0, "anchor": 0.0, "samples": 0.0}
        for batch in build_loader:
            images = batch["image"].to(device=runtime_device, non_blocking=True)
            labels = batch["label"].to(device=runtime_device, dtype=torch.long)
            mu = oracle_mu.index_select(0, labels)
            logvar = oracle_logvar.index_select(0, labels)
            eps = torch.randn(
                labels.shape[0],
                int(cfg.MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS),
                latent_dim,
                device=runtime_device,
                dtype=mu.dtype,
            )
            model.set_runtime_prompt_distribution_override(mu, logvar, eps=eps)
            try:
                logits = model(images, semantics=None, class_ids=class_ids, runtime_targets=None)
            finally:
                model.clear_runtime_prompt_distribution_override()
            classifier = model.get_runtime_classifier_stats()
            visual = classifier["visual_repr"]
            semantic = classifier["semantic_repr"].index_select(0, labels)
            ce = F.cross_entropy(logits, labels)
            align = (1.0 - F.cosine_similarity(visual, semantic, dim=-1)).mean()
            anchor = F.mse_loss(mu, initial_mu.index_select(0, labels)) + F.mse_loss(
                logvar, initial_logvar.index_select(0, labels)
            )
            loss = ce + float(args.align_weight) * align + float(args.anchor_weight) * anchor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                oracle_logvar.clamp_(min=float(args.logvar_min), max=float(args.logvar_max))
            batch_size = float(labels.shape[0])
            totals["loss"] += float(loss.detach().item()) * batch_size
            totals["ce"] += float(ce.detach().item()) * batch_size
            totals["align"] += float(align.detach().item()) * batch_size
            totals["anchor"] += float(anchor.detach().item()) * batch_size
            totals["samples"] += batch_size
        row = {"epoch": float(epoch + 1)}
        row.update({key: value / totals["samples"] for key, value in totals.items() if key != "samples"})
        history.append(row)
        print(
            "[stage2b-oracle] epoch={}/{} loss={:.6f} ce={:.6f} align={:.6f} anchor={:.6f}".format(
                epoch + 1, args.epochs, row["loss"], row["ce"], row["align"], row["anchor"]
            ),
            flush=True,
        )

    original_metrics, _, _ = _evaluate(
        model,
        eval_loader,
        class_ids,
        dataset.seen_classes,
        dataset.unseen_classes,
        oracle_mu,
        oracle_logvar,
        runtime_device,
        use_oracle=False,
    )
    oracle_metrics, visual_centers, logit_profiles = _evaluate(
        model,
        eval_loader,
        class_ids,
        dataset.seen_classes,
        dataset.unseen_classes,
        oracle_mu,
        oracle_logvar,
        runtime_device,
        use_oracle=True,
    )
    metadata = {
        **stage2_metadata("stage2B_oracle_prompt_distribution", seed=int(args.seed)),
        "format": "stage2b_oracle_prompt_distributions_v1",
        "seed": int(args.seed),
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_epoch": int(checkpoint.get("total_epoch", 0)),
        "build_ratio": float(args.build_ratio),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "align_weight": float(args.align_weight),
        "anchor_weight": float(args.anchor_weight),
        "sample_count": len(dataset),
        "build_sample_count": len(build_ids),
        "eval_sample_count": len(eval_ids),
        "original_eval": original_metrics,
        "oracle_eval": oracle_metrics,
        "oracle_usage": "unseen rows are hidden evaluation targets and B4 upper bound only",
    }
    class_attributes = dataset.class_attributes.detach().cpu().numpy()
    np.savez_compressed(
        str(args.output),
        oracle_mu=oracle_mu.detach().cpu().numpy().astype(np.float32),
        oracle_logvar=oracle_logvar.detach().cpu().numpy().astype(np.float32),
        initial_mu=initial_mu.detach().cpu().numpy().astype(np.float32),
        initial_logvar=initial_logvar.detach().cpu().numpy().astype(np.float32),
        oracle_final_visual_centers=visual_centers,
        oracle_final_logit_profiles=logit_profiles,
        class_attributes=class_attributes.astype(np.float32),
        seen_class_ids=np.asarray(dataset.seen_classes, dtype=np.int64),
        unseen_class_ids=np.asarray(dataset.unseen_classes, dtype=np.int64),
        build_class_counts=build_count.detach().cpu().numpy().astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        split_manifest_json=np.asarray(json.dumps(split_manifest, ensure_ascii=False)),
    )
    write_stage2_json(args.output.with_suffix(".json"), metadata, {})
    write_stage2_table(
        args.output.with_name(args.output.stem + "_history.csv"),
        history,
        stage2_metadata("stage2B_oracle_prompt_distribution", seed=int(args.seed)),
    )
    print(f"wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-2B oracle prompt distributions for all 200 CUB classes.")
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "prompt" / "cub.yaml")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--build-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--logvar-min", type=float, default=-10.0)
    parser.add_argument("--logvar-max", type=float, default=5.0)
    args = parser.parse_args()
    args.config_file = args.config_file.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.config_file.is_file() or not args.checkpoint.is_file():
        parser.error("config-file and checkpoint must exist.")
    if args.seed < 0 or args.batch_size <= 0 or args.num_workers < 0 or args.epochs <= 0:
        parser.error("seed/workers/batch/epochs are invalid.")
    if not 0.0 < args.build_ratio < 1.0 or args.lr <= 0.0:
        parser.error("build-ratio must be in (0,1) and lr positive.")
    if min(args.align_weight, args.anchor_weight) < 0.0 or args.logvar_min > args.logvar_max:
        parser.error("weights/logvar bounds are invalid.")
    return args


if __name__ == "__main__":
    build_oracle(parse_args())
