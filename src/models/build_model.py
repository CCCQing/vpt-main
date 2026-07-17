#!/usr/bin/env python3
"""Model construction helpers for the current ViT-only codepath."""

import torch

from .vit_models import ViT
from ..utils import logging
from ..utils import distributed as du

logger = logging.get_logger("visual_prompt")

_MODEL_TYPES = {
    "vit": ViT,
}


def build_model(cfg):
    assert cfg.MODEL.TYPE in _MODEL_TYPES, "Model type '{}' not supported".format(cfg.MODEL.TYPE)
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), "Cannot use more GPU devices than available"

    vis = bool(cfg.MODEL.AFFINITY.VIS or cfg.SOLVER.VIS.ENABLE)
    model = _MODEL_TYPES[cfg.MODEL.TYPE](cfg, vis=vis)

    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")
    return model, device



def log_model_info(model, verbose=False, label="Model"):
    if verbose:
        logger.info(f"Classification Model:\n{model}")

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "%s params: total=%d trainable=%d",
        label,
        model_total_params,
        model_grad_params,
    )



def get_current_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return torch.device("cpu")



def load_model_to_device(model, cfg):
    cur_device = get_current_device()

    if torch.cuda.is_available():
        model = model.cuda(device=cur_device)
    else:
        model = model.to(cur_device)
    return model, cur_device


def wrap_distributed_model(model, cfg):
    """Wrap a fully assembled model in DDP when a process group is active."""
    world_size = du.get_world_size()
    if world_size <= 1:
        return model
    if not torch.cuda.is_available():
        raise RuntimeError("DistributedDataParallel training requires CUDA in the current codepath.")
    if hasattr(model, "module"):
        raise ValueError("Model is already wrapped in DistributedDataParallel.")

    cur_device = torch.cuda.current_device()
    logger.info(
        "Wrapping complete model in DDP: rank=%d world_size=%d device=%d",
        du.get_rank(),
        world_size,
        cur_device,
    )
    return torch.nn.parallel.DistributedDataParallel(
        module=model,
        device_ids=[cur_device],
        output_device=cur_device,
        find_unused_parameters=True,
    )
