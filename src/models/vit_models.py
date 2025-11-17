#!/usr/bin/env python3

"""
ViT-related models ViT 系列模型定义与封装。
Note: models return logits instead of prob
注意：各模型的 forward 返回的是“logits”（未过 softmax），方便配合交叉熵等损失。
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
    """
    ViT-related model.

    这个类是“统一的 ViT 外壳”：
    - 负责根据 cfg 构建不同类型的 ViT 主干（监督 / 自监督 / 带 prompt / 带 adapter 等）；
    - 根据 TRANSFER_TYPE 控制“哪些参数需要训练、哪些被冻结”；
    - 可选 side 分支（旁路 AlexNet 特征）；
    - 顶层统一接一个 MLP 头做分类。

    真正的 ViT 编码器本体在 self.enc 里，由 build_vit_sup_models 构建。
    """
    def __init__(self, cfg, load_pretrain=True, vis=False):
        """
        参数:
          cfg            : 全局配置对象
          load_pretrain  : 是否从预训练权重初始化 backbone
          vis            : 可视化/调试相关开关（由构建函数透传）
        """
        super(ViT, self).__init__()

        # 如果 TRANSFER_TYPE 里包含 "prompt"，则需要传入 prompt 配置；否则不使用 prompt
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        # self.froze_enc 标记“是否把编码器整体看作是冻结的”
        # - 当 transfer_type 不是 end2end 且不含 prompt 时（例如 linear、cls、adapter、partial 等），
        #   默认认为是“冻主干、只训头部/局部”，这里置 True；
        # - 当是 end2end 或包含 prompt 的模式下（prompt / cls+prompt），置 False，
        #   后续会按更细粒度控制哪些参数 requires_grad。
        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False

        # adapter 模式时，才需要读取 ADAPTER 配置，其余情况 adapter_cfg 置 None
        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        # ===== 核心：构建 ViT 主干，并根据 transfer_type 设置参数可训练性 =====
        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.cfg = cfg

        # 可选：构建 side 分支（用 AlexNet 做旁路特征）
        self.setup_side()

        # 最终分类头：MLP(输入维度 = feat_dim, 输出维度 = 类别数)
        self.setup_head(cfg)

    # --------------------------------------------------------------------- #
    #  side 分支：旁路特征（AlexNet），用于 "side" 迁移类型
    # --------------------------------------------------------------------- #
    def setup_side(self):
        """
        side 分支（旁路特征）只在 TRANSFER_TYPE == "side" 时启用。

        做法：
        - 使用 torchvision 的 AlexNet 预训练特征部分（features + avgpool）；
        - 将输出 (B, 9216) 线性投影到和 ViT 主干相同的 feat_dim；
        - 在 forward 里，使用一个可训练标量 alpha，通过 sigmoid(alpha) 把主干和 side 的特征做凸组合：
              x = σ(alpha) * vit_feat + (1 - σ(alpha)) * side_feat
        """
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            # 可学习的融合系数，初始化为 0，后续通过 sigmoid 压到 (0,1)
            self.side_alpha = nn.Parameter(torch.tensor(0.0))

            # AlexNet 作为 side 分支
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),   # 卷积特征
                ("avgpool", m.avgpool),     # 平均池化到固定空间
            ]))

            # AlexNet 展平特征维度为 9216（典型 6*6*256），投影到与 ViT 一致的 self.feat_dim
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    # --------------------------------------------------------------------- #
    #  构建 ViT 主干并设置冻结策略
    # --------------------------------------------------------------------- #
    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        """
        构建 ViT 主干，并依据 TRANSFER_TYPE 设置各层参数的 requires_grad。

        步骤：
          1) 使用 build_vit_sup_models 构建编码器 self.enc 以及其特征维度 self.feat_dim；
          2) 根据 transfer_type 的不同，在 self.enc.named_parameters() 上修改 requires_grad；
             通过字符串匹配参数名的方式，精确地解冻/冻结对应的层或模块。
        """
        transfer_type = cfg.MODEL.TRANSFER_TYPE

        # enc: 具体的 ViT 编码器（VisionTransformer 或 PromptedVisionTransformer 等）
        # feat_dim: 编码器最后输出的特征维度（一般是 hidden_size，如 768）
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE,         # 预训练名称，如 "imagenet21k_sup_vitb16"
            cfg.DATA.CROPSIZE,        # 输入裁剪尺寸（如 224）
            prompt_cfg,               # prompt 子配置（可能为 None）
            cfg.MODEL.MODEL_ROOT,     # 存放预训练权重的路径
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

        # ---------- partial-k：只微调最后若干层 block + encoder_norm ----------
        if transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer) # 总层数 L
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            # 仅允许“最后 1 层 block + encoder_norm”更新，其余冻结
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                # 只保留倒数第1、第2层 block + encoder_norm
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                # 只保留倒数 4 层 block + encoder_norm
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.layer.{}".format(total_layer - 3) not in k and "transformer.encoder.layer.{}".format(total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        # ---------- linear / side：完全冻结主干，只训练线性头或 side ----------
        elif transfer_type == "linear" or transfer_type == "side":
            # 纯线性微调或 side 融合，完全冻结 backbone
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        # ---------- tinytl-bias：只训练所有 bias ----------
        elif transfer_type == "tinytl-bias":
            # TinyTL：仅训练 bias，显著降低可训练参数量与显存占用
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        # ---------- prompt（below）：只训练 prompt + patch embedding ----------
        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            # ViT “below 模式”：在输入通道维或 patch embedding 上引入 prompt
            # 这里允许 prompt 和 patch_embeddings（weight, bias）更新，其余冻结
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k  and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        # ---------- prompt：只训练 prompt 模块 ----------
        elif transfer_type == "prompt":
            # 仅训练 prompt 相关参数（如前置/中间插入的虚拟 token）
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        # ---------- prompt+bias：训练 prompt + 所有 bias ----------
        elif transfer_type == "prompt+bias":
            # 训练 prompt 与所有 bias
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        # ---------- prompt-noupdate：prompt 也不更新（全冻结，用于 ablation） ----------
        elif transfer_type == "prompt-noupdate":
            # prompt 也不更新（全冻结，常用于 ablation）
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        # ---------- cls：只训练 cls_token ----------
        elif transfer_type == "cls":
            # 仅训练 cls_token（其他全冻结）
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        # ---------- cls-reinit：重置 cls_token 再只训练 cls_token ----------
        elif transfer_type == "cls-reinit":
            # 先重置 cls_token，再仅训练 cls_token
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            # 再只训练 cls_token
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        # ---------- cls+prompt：同时训练 cls_token 和 prompt ----------
        elif transfer_type == "cls+prompt":
            # 同时训练 cls_token 与 prompt
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        # ---------- cls-reinit+prompt：重置 cls_token + 训练 cls_token 和 prompt ----------
        elif transfer_type == "cls-reinit+prompt":
            # 重置 cls_token，并同时训练 cls_token 与 prompt
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False
        
        # ---------- adapter：只训练 adapter 模块 ----------
        elif transfer_type == "adapter":
            # 仅训练注入到各层的 adapter 模块，其余冻结
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        # ---------- end2end：所有参数都可训练 ----------
        elif transfer_type == "end2end":
            # 端到端全量更新（不做冻结）
            logger.info("Enable all parameters update during training")

        # ---------- 其他未支持的类型 ----------
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

    # --------------------------------------------------------------------- #
    #  分类头：统一用一个 MLP
    # --------------------------------------------------------------------- #
    def forward(self, x, return_feature=False):
        """
        主前向：
          1) 若配置了 side 分支，则先通过 side 提取 AlexNet 特征，并线性投影；
          2) 如果在训练阶段且标记 froze_enc=True，则将 enc 设为 eval()（冻结 BN/Dropout 行为）；
          3) 通过 enc 得到全局特征（batch_size, feat_dim）；
          4) 若存在 side 分支，则用 α 做凸组合：x = σ(α)*x + (1-σ(α))*side；
          5) 若请求 return_feature=True，直接返回特征（用于下游或可视化）；
          6) 否则通过 MLP 分类头，输出 logits。
        """
        # 1) side 分支（可选）
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        # 2) 若 enc 被标记为“整体冻结”，且当前确实在训练阶段，则把 enc 切到 eval()
        #    —— 这样其中的 Dropout/LayerNorm/BatchNorm 行为固定下来，更接近“固定特征”的语义。
        if self.froze_enc and self.enc.training:
            self.enc.eval()

        # 3) 主干编码器获取全局特征（通常是 CLS 或 GAP 后的 embedding）
        x = self.enc(x)  # batch_size x self.feat_dim

        # 4) 若有 side 分支，用一个标量 alpha（经 sigmoid 后）融合主干与 side 特征
        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha) # 标量 ∈ (0,1)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        # 5) 若只需要特征，不需要分类 logits，则直接返回特征（第二个返回值为同样的 x，兼容旧接口）
        if return_feature:
            return x, x

        # 6) 通过 MLP 头输出 logits
        x = self.head(x)

        return x
    
    def forward_cls_layerwise(self, x):
        """
        获取“逐层的 CLS 表征”：
        - 调用 self.enc.forward_cls_layerwise(x)；
        - 输出格式视 backbone 实现而定，一般是 (num_layers, B, D) 或类似结构。
        常用于分析可视化或层级特征蒸馏等。
        """
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """
        get a (batch_size, self.feat_dim) feature
        仅提取 (batch_size, feat_dim) 的全局特征，不过分类头。
        """
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


# ===================================================================== #
#  Swin Transformer：继承 ViT 外壳，只重写 build_backbone（构建 Swin + 冻结规则）
# ===================================================================== #
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


# ===================================================================== #
#  自监督 ViT：MoCo v3 / MAE 封装（同样继承 ViT）
# ===================================================================== #
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
