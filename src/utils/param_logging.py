#!/usr/bin/env python3
"""Utilities for inspecting and logging trainable parameters.

This helper categorizes parameters according to the project's conceptual
decomposition (prompt path, semantic side branch, affinity alignment,
classification head, and backbone), then reports how gradients are
distributed across these groups.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch.nn as nn


def _init_stats(groups: List[str]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for g in groups:
        stats[g] = {
            "total_numel": 0,
            "train_numel": 0,
            "train_count": 0,
        }
    stats["prompt_tokens"]["subtypes"] = {
        "prompt_dropout": {"total": 0, "train": 0},
        "prompt_other": {"total": 0, "train": 0},
    }
    return stats


def _classify_prompt_subtype(name: str) -> str:
    lower = name.lower()
    if "dropout" in lower and "prompt" in lower:
        return "prompt_dropout"
    return "prompt_other"


def _classify_group(name: str) -> str:
    lower = name.lower()

    prompt_dist_keys = [
        "prompt_dist",
        "prompt_generator",
        "promptgen",
        "prompt_distributor",
        "previtpromptdistributor",
        "posterior",
        "latent",
        "z_head",
        "logvar",
        "mu",
        "vae",
        "prompt_vae",
    ]
    if any(k in lower for k in prompt_dist_keys):
        return "prompt_distribution"

    if "semantic_token_projector" in lower:
        return "semantic_token_projector"

    if any(k in lower for k in ["affinity", "a_vs", "a_pv", "a_ps", "triple", "consistency"]):
        return "affinity_branch"

    if any(
        k in lower
        for k in [
            "cls_head",
            "classifier",
            "logit_scale",
            "sim_head",
            "r_similarity_head",
            "prototype_proj",
            "vs_projection",
        ]
    ) or lower.startswith("head"):
        return "cls_head"

    if "prompt" in lower:
        return "prompt_tokens"

    backbone_keys = [
        "backbone",
        "transformer",
        "patch_embed",
        "pos_embed",
        "blocks",
        "encoder",
        "embeddings",
        "cls_token",
        "ln_f",
        "norm",
    ]
    if any(k in lower for k in backbone_keys):
        return "backbone_pretrained"

    return "other"


def log_trainable_parameters(
    model: nn.Module, logger, max_examples_per_group: int = 10
) -> None:
    groups = [
        "backbone_pretrained",
        "prompt_tokens",
        "prompt_distribution",
        "semantic_token_projector",
        "affinity_branch",
        "cls_head",
        "other",
    ]
    stats = _init_stats(groups)

    for name, param in model.named_parameters():
        group = _classify_group(name)
        group_stats = stats[group]
        group_stats["total_numel"] += param.numel()
        if param.requires_grad:
            group_stats["train_numel"] += param.numel()
            group_stats["train_count"] += 1

        if group == "prompt_tokens":
            subtype = _classify_prompt_subtype(name)
            substats = group_stats["subtypes"][subtype]
            substats["total"] += param.numel()
            if param.requires_grad:
                substats["train"] += param.numel()

    total_params = sum(v["total_numel"] for v in stats.values())
    trainable_params = sum(v["train_numel"] for v in stats.values())
    trainable_tensors = [
        "{} shape={} numel={}".format(name, list(param.shape), int(param.numel()))
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    sorted_groups = sorted(stats.items(), key=lambda item: item[1]["train_numel"], reverse=True)
    nonzero_groups = [
        (group_name, group_stats)
        for group_name, group_stats in sorted_groups
        if group_stats["train_numel"] > 0
    ]
    summary_parts = []
    for group_name, group_stats in nonzero_groups[:3]:
        total = group_stats["total_numel"]
        trainable = group_stats["train_numel"]
        pct_all = (total / total_params * 100) if total_params else 0.0
        summary_parts.append(f"{group_name}={trainable} ({pct_all:.2f}% all)")
    if summary_parts:
        logger.info("[visual_prompt]: trainable modules: %s", ", ".join(summary_parts))
    logger.info(
        "[visual_prompt]: effective trainable tensors=%d parameters=%d: %s",
        len(trainable_tensors),
        trainable_params,
        "; ".join(trainable_tensors) if trainable_tensors else "<none>",
    )

    if stats["backbone_pretrained"]["train_numel"] > 0:
        logger.warning(
            "[visual_prompt]: WARNING: backbone_pretrained has "
            f"{stats['backbone_pretrained']['train_numel']} trainable params (unexpected in pure prompt-tuning mode)."
        )

    if stats["affinity_branch"]["train_numel"] > 0:
        logger.warning(
            "[visual_prompt]: WARNING: affinity_branch has "
            f"{stats['affinity_branch']['train_numel']} trainable params, but affinity is intended to be a loss-only alignment path."
        )


def debug_list_trainable_params(model: nn.Module, logger, substring: Optional[str] = None) -> None:
    """
    If substring is None, list all trainable parameters (name, shape).
    If substring is given, only list parameters whose names contain that substring.
    """

    logger.info("[visual_prompt]: Trainable parameter list (debug):")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if substring and substring.lower() not in name.lower():
            continue
        logger.info(f"[visual_prompt]:   {name}: shape={list(param.shape)}")
