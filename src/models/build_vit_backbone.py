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
    prompt_backend = prompt_cfg.BACKEND.lower() if prompt_cfg is not None else "dynamic"
    if prompt_cfg is not None:
        dist_cfg = prompt_cfg.DISTRIBUTOR
        use_distributor = False
        if prompt_backend == "dynamic":
            use_distributor = (
                prompt_cfg.INIT_SOURCE.lower() == "distributor_mean"
                and bool(dist_cfg.ENABLE)
            )
        elif prompt_backend == "vpt_deep":
            use_distributor = (
                prompt_cfg.INIT_SOURCE.lower() == "distributor_mean"
                and bool(dist_cfg.ENABLE)
            )
        if (
            use_distributor
            and prompt_provider is None
        ):
            prompt_provider = PreViTPromptDistributor(
                dim=m2featdim[model_type],
                prompt_len=prompt_cfg.NUM_TOKENS,
                hidden_dim=dist_cfg.STATS_HIDDEN_DIM,
                source=dist_cfg.SOURCE,
                instance_tokens=dist_cfg.INSTANCE_TOKENS,
                domain_tokens=dist_cfg.DOMAIN_TOKENS,
                logvar_min=dist_cfg.LOGVAR_MIN,
                logvar_max=dist_cfg.LOGVAR_MAX,
                eval_sample_mode=dist_cfg.EVAL_SAMPLE_MODE,
                use_slot_embed=dist_cfg.USE_SLOT_EMBED,
                cnn_name=dist_cfg.CNN_NAME,
                clip_name=dist_cfg.CLIP_NAME,
                clip_local_dir=dist_cfg.CLIP_LOCAL_DIR,
                dino_local_dir=dist_cfg.DINO_LOCAL_DIR,
                external_allow_download=dist_cfg.EXTERNAL_ALLOW_DOWNLOAD,
                debug_preprocess_shapes=dist_cfg.DEBUG_PREPROCESS_SHAPES,
            )
        if prompt_provider is not None:
            if bool(dist_cfg.DISABLE_SAMPLING):
                raise ValueError("DISTRIBUTOR.DISABLE_SAMPLING is deprecated by EVAL_SAMPLE_MODE and cannot be enabled.")

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
