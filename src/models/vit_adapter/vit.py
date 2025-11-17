#!/usr/bin/env python3
"""
vit with adapter
这里用到的是“项目自带”的 ViT 实现与工具（不是 timm 的）：
包含 Embeddings、Attention、Mlp、CONFIGS、np2th、常量字符串等
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from scipy import ndimage

from torch.nn import Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class ADPT_Block(nn.Module):
    """
    一个带 Adapter 的 ViT 编码器 block。
    结构与标准 ViT block 相同，只是在第二条残差（FFN 之后）插入 Adapter。
    """
    def __init__(self, config, vis, adapter_config):
        """
        Args:
            config: ViT 结构配置（hidden_size、heads、mlp_ratio 等）
            vis: 是否在前向时返回注意力权重（可视化用）
            adapter_config: Adapter 的超参（STYLE、REDUCATION_FACTOR 等）
        """
        super(ADPT_Block, self).__init__()
        self.hidden_size = config.hidden_size

        # 两个 LayerNorm：分别对应 Self-Attention 和 FFN 之前
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # 前馈网络（FFN/MLP）与自注意力模块（来自 vit_backbones.vit）
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.adapter_config = adapter_config

        # 仅实现 Pfeiffer 风格的 Adapter：FFN 之后再过一个小瓶颈（降→激活→升），并做短残差
        if adapter_config.STYLE == "Pfeiffer":
            # 线性降维：H -> H/r
            self.adapter_downsample = nn.Linear(
                config.hidden_size,
                config.hidden_size // adapter_config.REDUCATION_FACTOR
            )
            # 线性升维：H/r -> H
            self.adapter_upsample = nn.Linear(
                config.hidden_size // adapter_config.REDUCATION_FACTOR,
                config.hidden_size
            )
            # 激活函数与 MLP 保持一致（这里用 GELU）
            self.adapter_act_fn = ACT2FN["gelu"]

            # 零初始化：保证一开始 Adapter(x)≈0，不破坏预训练表示
            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)

            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)
        else:
            raise ValueError("Other adapter styles are not supported.")

    def forward(self, x):
        """
        前向：标准两段残差 + 在 MLP 后插入 Adapter，小残差加回
        Args:
            x: [B, 1+N, H]（CLS + patches）
        Returns:
            x: 更新后的隐藏状态 [B, 1+N, H]
            weights: 注意力权重（当 vis=True 时由 Attention 返回）
        """
        if self.adapter_config.STYLE == "Pfeiffer":
            # same as reguluar ViT block
            # 段1：LN → Self-Attn → 残差
            h = x                                 # 残差分支
            x = self.attention_norm(x)            # LN
            x, weights = self.attn(x)             # 自注意力；若 vis=True，会返回注意力图
            x = x + h                             # 残差相加

            # 段2：LN → MLP
            h = x
            x = self.ffn_norm(x)                  # LN
            x = self.ffn(x)                       # FFN(MLP)，形状保持 [B, 1+N, H]

            # start to insert adapter layers...
            # 在此处插入 Adapter（降→激活→升），并与 MLP 输出做“短残差”
            adpt = self.adapter_downsample(x)     # 降维 [B, 1+N, H/r]
            adpt = self.adapter_act_fn(adpt)      # GELU
            adpt = self.adapter_upsample(adpt)    # 升维回 [B, 1+N, H]
            x = adpt + x                          # 短残差（由于零初始化，初始等价恒等）
            # -- Adapter 结束 --
            # ...end

            x = x + h 
            return x, weights

    def load_from(self, weights, n_block):
        """
        从 JAX/npz 风格的官方 ViT 权重里，取出第 n_block 层的参数并拷入本地 PyTorch 模块。
        - 包含 Q/K/V/Out 四个注意力的线性层、两层 FFN 的权重与偏置、两处 LayerNorm。
        - 注意这里的 np2th 会做必要的转置/reshape。
        """
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # --- 注意力 Q/K/V/OUT 的权重与偏置（需要转置/reshape） ---
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            # --- MLP 两层全连接的权重与偏置 ---
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            # --- 两处 LayerNorm（scale/bias） ---
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class ADPT_Encoder(nn.Module):
    """多层堆叠的 Encoder：由若干个 ADPT_Block 组成，最后再做一次 LayerNorm"""
    def __init__(self, config, vis, adapter_cfg):
        super(ADPT_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = ADPT_Block(config, vis, adapter_cfg)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, 1+N, H]（来自 embeddings 的序列）
        Returns:
            encoded: 最终编码（LayerNorm 之后）[B, 1+N, H]
            attn_weights: 若 vis=True，收集每层的注意力权重列表；否则为空列表
        """
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class ADPT_Transformer(nn.Module):
    """Transformer = Embeddings + Encoder（都是自家实现的版本）"""
    def __init__(self, config, img_size, vis, adapter_cfg):
        super(ADPT_Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = ADPT_Encoder(config, vis, adapter_cfg)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class ADPT_VisionTransformer(nn.Module):
    """
    顶层封装：包含 ADPT_Transformer 与分类头（head）
    取 CLS（x[:,0]）过线性层得到 logits；
    vis=True 时返回注意力可视化用的权重
    """
    def __init__(
        self, model_type,
        img_size=224, num_classes=21843, vis=False, adapter_cfg=None
    ):
        super(ADPT_VisionTransformer, self).__init__()
        # 根据 model_type（如 ViT-B/16）取结构配置
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = ADPT_Transformer(config, img_size, vis, adapter_cfg)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        """
        Args:
            x: [B, C, H, W]
            vis: 若 True，返回注意力权重（用于可视化）
        """
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights

    def load_from(self, weights):
        """
        从 JAX/npz 官方权重加载到本模型。
        处理内容：
          - patch embedding 的卷积权重与偏置；
          - CLS token；
          - 编码器末端的 LayerNorm；
          - 位置编码（若分辨率不同，做网格插值到新大小）；
          - 逐层把 encoder block 参数拷入（调用上面的 ADPT_Block.load_from）
          - 若启用了 hybrid stem（如 CNN 前端），也一并拷入。
        """
        with torch.no_grad():
            # ----- patch embedding -----
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            # ----- CLS token -----
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))

            # ----- encoder 末端 LayerNorm -----
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # 位置编码：若分辨率不同，使用 scipy.ndimage.zoom 做双线性插值（保 CLS 位）
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                # 尺寸一致，直接复制
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                # 网格大小不一致，需要插值
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                # 若使用的是 "token" 分类方式，位置编码中含有 CLS，需要单独处理
                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # [1,1,H] 与 [G,H]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]      # 无 CLS 情况

                # 计算旧/新网格边长（假设正方形网格）
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))

                # reshape 成 [gs_old, gs_old, H] 做双线性插值，再 reshape 回 [1, gs_new*gs_new, H]
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)

                # 重新拼回（带 CLS 的情况把 token 放回前面）
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # ----- 逐层把 encoder block 的权重拷入（包括注意力、FFN、Norm）-----
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # 若采用“混合嵌入”（hybrid，卷积干预的那种），还要加载最前面的 conv + GN
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
