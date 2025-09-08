#!/usr/bin/env python3

"""
ViT-related models ViT 系列模型定义与封装。
Note: models return logits instead of prob 注意：各模型的 forward 返回的是“logits”（未过 softmax），方便配合交叉熵等损失。
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models, build_swin_model,
    build_mocov3_model, build_mae_model
)
from .mlp import MLP
from ..utils import logging
logger = logging.get_logger("visual_prompt")


class ViT(nn.Module):
    """ViT-related model.封装了“通用的 ViT 微调/迁移”逻辑：根据 TRANSFER_TYPE 决定冻结哪些参数，
    选择是否启用 prompt、adapter、side 分支等，并统一在顶层接一个分类头（MLP）。
    """
    def __init__(self, cfg, load_pretrain=True, vis=False):
        """
        参数:
          cfg            : 全局配置对象
          load_pretrain  : 是否从预训练权重初始化 backbone
          vis            : 可视化/调试相关开关（由构建函数透传）
        """
        super(ViT, self).__init__()
        # 如果 TRANSFER_TYPE 涉及 prompt，则从配置中取出 prompt 子配置；否则置空
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None
        # 是否“冻结编码器主体”标记。
        # - 非 end2end 且不含 prompt 的场景（如 linear/cls/tiny-tl/partial/adapter）默认冻结主体；
        # - end2end 或含 prompt（prompt、cls+prompt）则允许主体被训练（后续还会更细致地控制）。
        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False
        # adapter 微调场景需要读取 ADAPTER 子配置，其余为 None
        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None
        # 构建骨干网络（ViT/DeiT 等）并按照 TRANSFER_TYPE 设置参数可训练性
        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        # 可选的 side 分支（AlexNet 特征 + 线性投影 + 可学习融合）
        self.setup_side()
        # 分类头（MLP）：输入 feat_dim，输出类别数
        self.setup_head(cfg)

    def setup_side(self):
        """
        设置“side”分支（旁路特征）：只在 TRANSFER_TYPE == "side" 时启用。
        实现：加载一个预训练 AlexNet 的特征部分，接 avgpool，再用线性层投影到与主干相同的维度；
        最终在 forward 中通过一个可学习的 α（sigmoid 约束在 0~1）和主干特征做凸组合。
        """
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            # 可学习的融合系数（标量），实际使用时会过 sigmoid 变到 (0,1)
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            # 旁路分支采用 torchvision 的 AlexNet 预训练特征
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),   # 卷积特征
                ("avgpool", m.avgpool),     # 平均池化到固定空间
            ]))
            # AlexNet 的展平特征为 9216 维（典型 6*6*256），投影到 self.feat_dim
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        """
        构建 ViT 主干，并依据 TRANSFER_TYPE 设置各层参数的 requires_grad。
        核心思路：
          - 通过 build_vit_sup_models 构建编码器 self.enc，并得到特征维度 self.feat_dim；
          - 根据不同 TRANSFER_TYPE（linear/prompt/cls/partial/adapter/end2end 等），
            选择性地解冻/冻结特定参数，实现低成本微调。
        """
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        # enc: 编码器（ViT/DeiT）
        # feat_dim: 编码器输出特征维度（分类前的 embedding 维）
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE,         # 预训练模型标识（如 vit-base-patch16-224）
            cfg.DATA.CROPSIZE,        # 输入裁剪尺寸（如 224）
            prompt_cfg,               # prompt 子配置（可能为 None）
            cfg.MODEL.MODEL_ROOT,     # 预训练权重根目录
            adapter_cfg,              # adapter 子配置（可能为 None）
            load_pretrain,            # 是否加载预训练权重
            vis                       # 可视化/调试开关
        )
        # ====== 下方分支控制“参数可训练性” ======
        # 约定：
        # - partial-k：仅微调最后 k 层 encoder block（以及 layernorm）
        # - linear/side：完全冻结编码器，仅训练顶层线性/融合头
        # - tinytl-bias：只训练 bias 参数
        # - prompt：只训练 prompt 相关参数（可选：below 时也训练 patch embed）
        # - prompt+bias：训练 prompt 与全部 bias
        # - prompt-noupdate：prompt 也不训练（完全冻结）
        # - cls：只训练 cls_token
        # - cls-reinit：重置 cls_token 后，只训练 cls_token
        # - cls+prompt / cls-reinit+prompt：同时训练 prompt 与 cls_token
        # - adapter：只训练 adapter 模块
        # - end2end：全量训练
        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer)
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            # 仅允许“最后 1 层 block + encoder_norm”更新，其余冻结
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.layer.{}".format(total_layer - 3) not in k and "transformer.encoder.layer.{}".format(total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            # 纯线性微调或 side 融合，完全冻结 backbone
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            # TinyTL：仅训练 bias，显著降低可训练参数量与显存占用
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            # 仅训练 prompt；若 LOCATION=below，再放开 patch embedding（权重与偏置）
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k  and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            # 仅训练 prompt 相关参数（如前置/中间插入的虚拟 token）
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            # 训练 prompt 与所有 bias
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt-noupdate":
            # prompt 也不更新（全冻结，常用于 ablation）
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "cls":
            # 仅训练 cls_token（其他全冻结）
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls-reinit":
            # 先重置 cls_token，再仅训练 cls_token
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )

            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls+prompt":
            # 同时训练 cls_token 与 prompt
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls-reinit+prompt":
            # 重置 cls_token，并同时训练 cls_token 与 prompt
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False
        
        # adapter
        elif transfer_type == "adapter":
            # 仅训练注入到各层的 adapter 模块，其余冻结
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            # 端到端全量更新（不做冻结）
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

    def setup_head(self, cfg):
        """
        构建分类头（MLP）。维度规则：
          - 输入维度：self.feat_dim（来自 backbone）
          - 中间层：重复 self.cfg.MODEL.MLP_NUM 次的 feat_dim 全连接层
          - 输出层：类别数 cfg.DATA.NUMBER_CLASSES
          - special_bias=True：MLP 实现中可能使用特殊的偏置初始化/形态（与项目实现相关）
        """
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):
        """
        主前向：
          1) 可选 side 分支提取 + 线性投影；
          2) 如果在训练阶段且标记 froze_enc=True，则将 enc 设为 eval()（冻结 BN/Dropout 行为）；
          3) 通过 enc 得到全局特征（batch_size, feat_dim）；
          4) 若存在 side 分支，则用 α 做凸组合：x = σ(α)*x + (1-σ(α))*side；
          5) 若请求 return_feature=True，直接返回特征（用于下游或可视化）；
          6) 否则通过 MLP 分类头，输出 logits。
        """
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)
        # 冻结策略：若 enc 被标记冻结且处于训练模式，则切到 eval 防止 BN/Dropout 发生随机行为
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        # 主干编码：输出 (B, feat_dim)
        x = self.enc(x)  # batch_size x self.feat_dim
        # 若启用 side 分支，则与主干特征做可学习融合
        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha) # 标量 ∈ (0,1)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output
        # 需要返回特征（而非 logits）时，直接返回；第二个返回与第一个相同（兼容既有接口）
        if return_feature:
            return x, x
        # 走分类头得到 logits
        x = self.head(x)

        return x
    
    def forward_cls_layerwise(self, x):
        """
        返回“逐层的 CLS 表征”（具体格式由 backbone 的 forward_cls_layerwise 定义），
        常用于分析可视化或层级特征蒸馏等。
        """
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature
        仅提取 (batch_size, feat_dim) 的全局特征，不过分类头。"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


class Swin(ViT):
    """Swin-related model.
    Swin Transformer 版本，复用 ViT 的整体框架，仅重写 build_backbone 与冻结逻辑。"""

    def __init__(self, cfg):
        super(Swin, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        """
        构建 Swin 主干，并根据 TRANSFER_TYPE 选择性冻结。
        Swin 的模块命名与 ViT 不同，这里针对其层级结构（layers/blocks）做了适配。
        """
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT
        )

        # linear, prompt, cls, cls+prompt, partial_1
        # ====== Swin 的冻结策略分支 ======
        if transfer_type == "partial-1":
            # 仅训练最后一层的最后一个 block，以及最终的 norm
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-1].blocks)
            for k, p in self.enc.named_parameters():
                if "layers.{}.blocks.{}".format(total_layer - 1, total_blocks - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            # 仅训练最后一层（整个 stage）和最终 norm
            total_layer = len(self.enc.layers)
            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            # 训练最后一层 + 倒数第二层的后若干模块（包括 downsample）+ 最终 norm
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-2].blocks)

            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 2) not in k and "layers.{}.downsample".format(total_layer - 2) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION in ["below"]:
            # Swin 下，below 场景放开 patch_embed；其余层仅训练 prompt
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


class SSLViT(ViT):
    """moco-v3 and mae model.
     自监督预训练（MoCo v3 / MAE）版本的 ViT 封装，构建函数会选用相应的 build_fn。"""

    def __init__(self, cfg):
        super(SSLViT, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        """
        根据 cfg.DATA.FEATURE 选择构建 MoCo v3 或 MAE 的 ViT 主干，
        随后按照 TRANSFER_TYPE 冻结/解冻参数。
        """
        if "moco" in cfg.DATA.FEATURE:
            build_fn = build_mocov3_model
        elif "mae" in cfg.DATA.FEATURE:
            build_fn = build_mae_model

        self.enc, self.feat_dim = build_fn(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg=adapter_cfg
        )

        transfer_type = cfg.MODEL.TRANSFER_TYPE
        # linear, prompt, cls, cls+prompt, partial_1
        # ====== 自监督 ViT 的冻结策略分支 ======
        if transfer_type == "partial-1":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "sidetune":
            # 这里的 "sidetune" 对应其他项目里的一类旁路微调策略；本实现中与 linear 等价为冻结主干
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            # below：同前，放开 patch_embed 的 conv（proj）权重与偏置
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed.proj.weight" not in k  and "patch_embed.proj.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")
        
        # adapter
        # adapter：仅训练 adapter 模块
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))
