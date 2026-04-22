#!/usr/bin/env python3
"""Build supervised ViT backbones and optionally wrap them with prompt/adapter modules."""

import os

import numpy as np

from .prompting.prompt_distribution import PreViTPromptDistributor
from .vit_adapter.vit import ADPT_VisionTransformer
from .vit_backbones.vit import VisionTransformer
from .vit_prompt.vit import PromptedVisionTransformer


MODEL_ZOO = {
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
}


def build_vit_sup_models(
    model_type,
    crop_size,
    prompt_cfg=None,
    model_root=None,
    adapter_cfg=None,
    load_pretrain=True,
    vis=False,
    prompt_init=None,
    prompt_init_provider=None,
):
    """Construct a supervised ViT backbone and load the matching `.npz` checkpoint."""

    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }

    prompt_provider = prompt_init_provider
    prompt_backend = str(prompt_cfg.BACKEND).lower() if prompt_cfg is not None else "dynamic"
    if prompt_cfg is not None:
        dist_cfg = getattr(prompt_cfg, "DISTRIBUTOR", None)
        if (
            prompt_backend == "dynamic"
            and dist_cfg is not None
            and getattr(dist_cfg, "ENABLE", False)
            and prompt_provider is None
        ):
            prompt_provider = PreViTPromptDistributor(
                dim=m2featdim[model_type],
                prompt_len=prompt_cfg.NUM_TOKENS,
                latent_dim=dist_cfg.LATENT_DIM,
                hidden_dim=dist_cfg.HIDDEN_DIM,
                pool=dist_cfg.POOL,
            )

    if prompt_cfg is not None:
        model = PromptedVisionTransformer(
            prompt_cfg,
            model_type,
            crop_size,
            num_classes=-1,
            vis=vis,
            prompt_init=prompt_init,
            prompt_init_provider=prompt_provider,
        )
    elif adapter_cfg is not None:
        model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, adapter_cfg=adapter_cfg)
    else:
        model = VisionTransformer(model_type, crop_size, num_classes=-1, vis=vis)

    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]
