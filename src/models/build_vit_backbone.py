#!/usr/bin/env python3
"""
ViT/Swin 相关 backbone 的构建与权重加载工具函数。
支持：
- 监督预训练的 ViT（.npz 权重）
- 自监督预训练的 MAE/MoCo v3（.pth/.tar 权重）
- Prompt/Adapter 变体（只在骨干网络中插入相应模块，可与部分权重加载共存）
返回：
- 已构建且加载了预训练权重的模型（去掉原分类头 head）
- 对应的特征维度 out_dim（用于上层分类头 MLP 的输入维度）
"""
import numpy as np
import torch
import os
# Swin / ViT 主干

from .vit_backbones.vit import VisionTransformer

# Prompt 版本骨干
from .vit_prompt.vit import PromptedVisionTransformer
from .prompting.prompt_distribution import PreViTPromptDistributor

# Adapter 版本骨干
from .vit_adapter.vit import ADPT_VisionTransformer
# 预训练权重文件名表（相对 model_root 的路径）
# 键是 model_type（在 cfg.DATA.FEATURE 等处使用），值是实际权重文件名
MODEL_ZOO = {
    # Swin 系列（监督 / 22k / 不同输入尺寸）
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224":  "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    # ViT 监督（.npz，典型来自 Google 官方权重）
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
    # MAE 自监督（.pth）
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
}


def build_mae_model(model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None):
    """
    构建 MAE 变体的 ViT，并加载预训练权重。
    - 三种形态互斥：Prompt > Adapter > 原生 MAE
    - 去掉原分类头（置为 Identity），返回特征维度 embed_dim

    参数：
        model_type : 字符串键，决定选用哪份权重（MODEL_ZOO 中的 MAE 键）
        crop_size  : 输入图像裁剪尺寸（本函数中不直接影响权重加载，仅影响某些模型构建）
        prompt_cfg : 若不为 None，则构建 Prompt 版本骨干
        model_root : 预训练权重的根目录
        adapter_cfg: 若不为 None，则构建 Adapter 版本骨干
    返回：
        model      : 已加载权重且去掉 head 的骨干模型
        out_dim    : 特征维度（embed_dim）
    """


    # 加载预训练检查点（map_location='cpu' 方便在无 GPU 环境下加载后再 .to(device)）
    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']    # MAE 预训练权重常见为 'model' 键

# 10.27 加入fusion_cfg=None
# def build_vit_sup_models(model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None, load_pretrain=True, vis=False):
def build_vit_sup_models(model_type, crop_size, prompt_cfg=None, fusion_cfg=None, model_root=None, adapter_cfg=None,
                             load_pretrain=True, vis=False, prompt_init=None, prompt_init_provider=None):
    """
    构建“监督预训练”的 ViT 及其 Prompt/Adapter 变体，并按需加载 .npz 权重。

    参数：
        model_type   : 监督权重标识（见 MODEL_ZOO 中 sup_* 键）
        crop_size    : 输入裁剪尺寸（224/384/…）
        prompt_cfg   : Prompt 设置（不为 None 则构建 PromptedVisionTransformer）
        fusion_cfg   : 动态提示/多层融合设置（若启用则构建 DynamicPromptedVisionTransformer）
        model_root   : 预训练权重根目录（.npz 放置于此）
        adapter_cfg  : Adapter 设置（不为 None 则构建 ADPT_VisionTransformer）
        load_pretrain: 是否从 .npz 加载监督预训练权重
        vis          : 可视化/调试标志（向下透传给骨干）

        prompt_init  : （可选）外部提供的 prompt 初始化张量，形状应为
                        (1, prompt_len, hidden_size)
        prompt_init_provider: （可选）可调用对象，返回 prompt_init，用于
                        在 build 阶段一次性从分布生成初始化（例如使用
                        PreViTPromptDistributor）。仅当 prompt_init 为 None 时生效。
    返回：
        model        : 已构建（并可能加载了权重）的骨干（head 已经是 identity）
        feat_dim     : backbone 的输出特征维度（供上层 MLP 使用）
    """
    # 各监督模型类型到输出维度的映射（来自官方配置/论文）
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,              # sup = supervised
        "sup_vitb16": 768,                  # vitb  ViT-Base 768/ vitl  ViT-Large 1024/ vith ViT-Huge 1280
        "sup_vitl16_224": 1024,             # vitb8 → 8×8 patch /vitb16 → 16×16 patch
        "sup_vitl16": 1024,                 # _224以 224×224 分辨率训练 / 没 _224 的，有时候代表“默认高分辨率版本”
        "sup_vitb8_imagenet21k": 768,       # _imagenet21k预训练时用了 ImageNet-21k（21000 类的超大版本），而不是普通的 ImageNet-1k（1000 类）
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }
    # 选择具体骨干：Prompt > Adapter > 原生 ViT
    # if prompt_cfg is not None:
    distribution_only = False   # 仅运行时调用 provider 生成 prompt，而不在构建阶段预生成/持有 prompt 参数（用于“分布专用”模式）
    prompt_provider = prompt_init_provider
    if prompt_cfg is not None:
        distribution_only = getattr(prompt_cfg, "DISTRIBUTION_ONLY", False)

        # 如配置启用提示分布模块，则在此构建 PreViTPromptDistributor 作为 provider
        dist_cfg = getattr(prompt_cfg, "DISTRIBUTOR", None)
        if dist_cfg is not None and getattr(dist_cfg, "ENABLE", False) and prompt_provider is None:
            semantic_dim = dist_cfg.SEMANTIC_DIM if dist_cfg.SEMANTIC_DIM > 0 else None
            semantic_proj_dim = dist_cfg.SEMANTIC_PROJ_DIM if dist_cfg.SEMANTIC_PROJ_DIM > 0 else None
            prompt_provider = PreViTPromptDistributor(
                dim=m2featdim[model_type],
                prompt_len=prompt_cfg.NUM_TOKENS,
                latent_dim=dist_cfg.LATENT_DIM,
                hidden_dim=dist_cfg.HIDDEN_DIM,
                pool=dist_cfg.POOL,
                semantic_dim=semantic_dim,
                semantic_proj_dim=semantic_proj_dim,
            )


    if (not distribution_only and prompt_init is None and
            prompt_provider is not None):
        prompt_init = prompt_provider()

    if prompt_cfg is not None:
        model = PromptedVisionTransformer(
            prompt_cfg, model_type,
            crop_size, num_classes=-1, vis=vis,
            prompt_init=prompt_init, prompt_init_provider=prompt_provider,
        )
    elif adapter_cfg is not None:
        model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, adapter_cfg=adapter_cfg)

    else:
        model = VisionTransformer(
            model_type, crop_size, num_classes=-1, vis=vis)

    # 监督预训练权重（.npz）加载：Google 风格，内部自己完成 key 映射
    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]

