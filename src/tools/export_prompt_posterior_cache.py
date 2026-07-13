#!/usr/bin/env python3
"""Export deterministic prompt posterior caches for Stage-2 Graph-GP evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.configs.config import get_cfg  # noqa: E402
from src.data.datasets.xlsa_dataset import CUB200Dataset  # noqa: E402
from src.data.transforms import get_transforms  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.tools.grid_search_graph_prob_prior_v5_temperatures import _mapping_to_opts  # noqa: E402
from src.tools.run_arch_ablation_graph_gp_energy import (  # noqa: E402
    _attention_mediation_overrides,
    _base_overrides,
    _cell_specs,
    _graph_prob_prior_overrides,
    _instance_prompt_distributor_overrides,
    _merge_overrides,
    _semantic_token_overrides,
)
from src.utils import logging  # noqa: E402


SUPPORTED_CELLS = {"A00", "A10", "C00", "C10", "C01", "C11"}


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["sample_id"] = int(index)
        item["image_path"] = str(self.dataset._imdb[index]["im_path"])
        return item


def _merge_local_path_cfg_if_exists(cfg) -> None:
    local_cfg = ROOT / "src" / "configs" / "local_path.yaml"
    if local_cfg.is_file():
        cfg.merge_from_file(str(local_cfg))
        print(f"[config] merged local path overrides: {local_cfg}", flush=True)


def _cell_spec(cell_id: str) -> Mapping[str, Any]:
    normalized = str(cell_id).upper()
    for spec in _cell_specs():
        if str(spec["cell_id"]).upper() == normalized:
            if normalized not in SUPPORTED_CELLS:
                break
            return spec
    raise ValueError(f"Stage-2 posterior export supports cells {sorted(SUPPORTED_CELLS)}, got '{cell_id}'.")


def _setup_cfg(args: argparse.Namespace):
    spec = _cell_spec(args.cell)
    cfg = get_cfg()
    cfg.merge_from_file(str(args.config_file))
    _merge_local_path_cfg_if_exists(cfg)
    overrides = _merge_overrides(
        _base_overrides("final_gzsl"),
        _instance_prompt_distributor_overrides(),
        _semantic_token_overrides(bool(spec["semantic_tokens"])),
        _graph_prob_prior_overrides(bool(spec["graph_gp"])),
        _attention_mediation_overrides(bool(spec["attention_mediation"])),
        {
            "SEED": int(args.seed),
            "NUM_GPUS": 1,
            "DATA.BATCH_SIZE": int(args.batch_size),
            "DATA.NUM_WORKERS": int(args.num_workers),
            "MODEL.WEIGHT_PATH": "",
            "MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE": "mean",
            "OUTPUT_DIR": str(args.output.parent),
        },
    )
    cfg.merge_from_list(_mapping_to_opts(overrides))
    cfg.merge_from_list(list(args.opts))
    if str(cfg.DATA.NAME).upper() != "CUB":
        raise ValueError("Stage-2 posterior export currently requires DATA.NAME=CUB.")
    if str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() != "final_gzsl":
        raise ValueError("Stage-2 posterior export requires DATA.XLSA.PROTOCOL_MODE=final_gzsl.")
    if str(cfg.MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE).lower() != "mean":
        raise ValueError("Stage-2 posterior export requires EVAL_SAMPLE_MODE=mean.")
    return cfg, spec


def _load_trainable_checkpoint(model, checkpoint_path: Path, seed: int, cell_id: str) -> Dict[str, Any]:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "vpt_trainable_v1":
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}; expected vpt_trainable_v1.")
    if payload.get("seed") is not None and int(payload["seed"]) != int(seed):
        raise ValueError(
            f"Checkpoint seed {payload['seed']} does not match requested seed {seed}: {checkpoint_path}"
        )
    if str(payload.get("cell_id", "")).upper() != str(cell_id).upper():
        raise ValueError(
            f"Checkpoint cell {payload.get('cell_id')!r} does not match requested cell {cell_id}: {checkpoint_path}"
        )
    saved_state = payload.get("model_state")
    if not isinstance(saved_state, dict) or not saved_state:
        raise ValueError(f"Checkpoint has no model_state: {checkpoint_path}")

    current_state = model.state_dict()
    current_trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    saved_names = set(saved_state)
    if saved_names != current_trainable:
        missing = sorted(current_trainable.difference(saved_names))
        unexpected = sorted(saved_names.difference(current_trainable))
        raise ValueError(
            "Checkpoint/config trainable parameter mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    for name, tensor in saved_state.items():
        if name not in current_state:
            raise KeyError(f"Checkpoint parameter '{name}' is absent from the constructed model.")
        if tuple(tensor.shape) != tuple(current_state[name].shape):
            raise ValueError(
                f"Checkpoint shape mismatch for {name}: saved={tuple(tensor.shape)} "
                f"current={tuple(current_state[name].shape)}"
            )
    incompatible = model.load_state_dict(saved_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    return payload


def _class_mean_semantics(dataset, batch_size: int, device: torch.device) -> torch.Tensor:
    class_attributes = dataset.class_attributes
    if not torch.is_tensor(class_attributes):
        class_attributes = torch.from_numpy(np.asarray(class_attributes))
    mean_attr = class_attributes.to(device=device, dtype=torch.float32).mean(dim=0)
    return mean_attr.unsqueeze(0).expand(int(batch_size), -1)


def _metadata(cfg, spec, args, checkpoint_payload, dataset) -> Dict[str, Any]:
    return {
        "format": "graph_gp_posterior_cache_v2",
        "dataset": str(cfg.DATA.NAME),
        "split": "trainval",
        "cell_id": str(spec["cell_id"]),
        "seed": int(args.seed),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_total_epoch": int(checkpoint_payload.get("total_epoch", 0)),
        "config_file": str(Path(args.config_file).resolve()),
        "sample_count": int(len(dataset)),
        "seen_class_count": int(len(dataset.seen_classes)),
        "crop_size": int(cfg.DATA.CROPSIZE),
        "batch_size": int(args.batch_size),
        "eval_sample_mode": str(cfg.MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE),
        "semantic_tokens": bool(spec["semantic_tokens"]),
        "graph_gp": bool(spec["graph_gp"]),
        "attention_mediation": bool(spec["attention_mediation"]),
        "transform": "Resize+CenterCrop+ToTensor+Normalize",
    }


def export_cache(args: argparse.Namespace) -> None:
    cfg, spec = _setup_cfg(args)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.setup_logging(1, 1, output=str(args.output.parent), name="visual_prompt")
    dataset = CUB200Dataset(cfg, "trainval")
    dataset.transform = get_transforms("stage2_extract", int(cfg.DATA.CROPSIZE))
    loader = torch.utils.data.DataLoader(
        IndexedDataset(dataset),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(cfg.DATA.PIN_MEMORY),
        drop_last=False,
    )

    model, device = build_model(cfg)
    runtime_device = torch.device(f"cuda:{int(device)}") if isinstance(device, int) else torch.device(device)
    if not hasattr(model, "attach_r_similarity_head"):
        raise RuntimeError("Constructed model does not expose attach_r_similarity_head().")
    model.attach_r_similarity_head(dataset.class_attributes)
    checkpoint_payload = _load_trainable_checkpoint(
        model, args.checkpoint, int(args.seed), str(spec["cell_id"])
    )
    model.eval()

    class_ids = torch.as_tensor(dataset.local_classes, device=runtime_device, dtype=torch.long)
    sample_ids: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    image_paths: List[str] = []
    posterior_mu: List[np.ndarray] = []
    posterior_logvar: List[np.ndarray] = []
    classifier_logits: List[np.ndarray] = []
    classifier_visual_repr: List[np.ndarray] = []
    classifier_semantic_repr = None

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device=runtime_device, non_blocking=True)
            semantics = None
            if bool(spec["semantic_tokens"]):
                semantics = _class_mean_semantics(dataset, int(images.shape[0]), runtime_device)
            logits = model(images, semantics=semantics, class_ids=class_ids, runtime_targets=None)
            stats = model.get_runtime_prompt_distribution_stats()
            if not isinstance(stats, dict) or "mu" not in stats or "logvar" not in stats:
                raise RuntimeError("Model forward did not expose prompt posterior mu/logvar.")
            mu = stats["mu"].detach().cpu().float().numpy()
            logvar = stats["logvar"].detach().cpu().float().numpy()
            if mu.ndim != 2 or logvar.shape != mu.shape:
                raise ValueError(f"Expected posterior [B,D], got mu={mu.shape} logvar={logvar.shape}.")
            if not np.isfinite(mu).all() or not np.isfinite(logvar).all():
                raise FloatingPointError(f"Non-finite posterior at extraction batch {batch_index}.")
            classifier_stats = model.get_runtime_classifier_stats()
            if not isinstance(classifier_stats, dict):
                raise RuntimeError("Model forward did not expose classifier visual/semantic representations.")
            visual_repr = classifier_stats["visual_repr"].detach().cpu().float().numpy()
            semantic_repr = classifier_stats["semantic_repr"].detach().cpu().float().numpy()
            logits_array = logits.detach().cpu().float().numpy()
            if visual_repr.ndim != 2 or visual_repr.shape[0] != mu.shape[0]:
                raise ValueError(f"Expected classifier visual_repr [B,D], got {visual_repr.shape}.")
            if semantic_repr.ndim != 2 or semantic_repr.shape[0] != class_ids.numel():
                raise ValueError(f"Expected classifier semantic_repr [C,D], got {semantic_repr.shape}.")
            if logits_array.shape != (mu.shape[0], class_ids.numel()):
                raise ValueError(f"Expected classifier logits [B,C], got {logits_array.shape}.")
            if not (
                np.isfinite(visual_repr).all()
                and np.isfinite(semantic_repr).all()
                and np.isfinite(logits_array).all()
            ):
                raise FloatingPointError(f"Non-finite classifier representation at extraction batch {batch_index}.")
            if classifier_semantic_repr is None:
                classifier_semantic_repr = semantic_repr
            elif not np.array_equal(classifier_semantic_repr, semantic_repr):
                raise RuntimeError("Classifier semantic representation changed across deterministic extraction batches.")
            sample_ids.append(batch["sample_id"].cpu().numpy().astype(np.int64, copy=False))
            labels.append(batch["label"].cpu().numpy().astype(np.int64, copy=False))
            image_paths.extend(str(path) for path in batch["image_path"])
            posterior_mu.append(mu)
            posterior_logvar.append(logvar)
            classifier_logits.append(logits_array)
            classifier_visual_repr.append(visual_repr)
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader):
                print(f"[posterior-export] {batch_index + 1}/{len(loader)} batches", flush=True)

    sample_ids_array = np.concatenate(sample_ids, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    mu_array = np.concatenate(posterior_mu, axis=0)
    logvar_array = np.concatenate(posterior_logvar, axis=0)
    logits_array = np.concatenate(classifier_logits, axis=0)
    visual_repr_array = np.concatenate(classifier_visual_repr, axis=0)
    paths_array = np.asarray(image_paths, dtype=np.str_)
    expected_ids = np.arange(len(dataset), dtype=np.int64)
    if not np.array_equal(sample_ids_array, expected_ids):
        raise RuntimeError("Extraction sample order is not the expected deterministic 0..N-1 order.")
    if not (
        labels_array.shape[0]
        == mu_array.shape[0]
        == logvar_array.shape[0]
        == logits_array.shape[0]
        == visual_repr_array.shape[0]
        == paths_array.shape[0]
        == len(dataset)
    ):
        raise RuntimeError("Posterior cache arrays have inconsistent sample counts.")

    metadata = _metadata(cfg, spec, args, checkpoint_payload, dataset)
    class_attributes = dataset.class_attributes
    if torch.is_tensor(class_attributes):
        class_attributes = class_attributes.detach().cpu().numpy()
    np.savez_compressed(
        str(args.output),
        sample_ids=sample_ids_array,
        global_labels=labels_array,
        posterior_mu=mu_array,
        posterior_logvar=logvar_array,
        classifier_logits=logits_array,
        classifier_visual_repr=visual_repr_array,
        classifier_semantic_repr=np.asarray(classifier_semantic_repr, dtype=np.float32),
        classifier_class_ids=class_ids.detach().cpu().numpy().astype(np.int64, copy=False),
        image_paths=paths_array,
        class_attributes=np.asarray(class_attributes, dtype=np.float32),
        seen_class_ids=np.asarray(dataset.seen_classes, dtype=np.int64),
        unseen_class_ids=np.asarray(dataset.unseen_classes, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {metadata_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic Stage-2 prompt posterior cache.")
    parser.add_argument("--config-file", default=str(ROOT / "configs" / "prompt" / "cub.yaml"))
    parser.add_argument("--cell", required=True, choices=sorted(SUPPORTED_CELLS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.config_file = Path(args.config_file)
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.config_file.is_file():
        parser.error(f"Config file does not exist: {args.config_file}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.seed < 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("seed must be non-negative, batch-size positive, and num-workers non-negative.")
    return args


if __name__ == "__main__":
    export_cache(parse_args())
