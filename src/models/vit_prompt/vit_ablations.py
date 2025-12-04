#!/usr/bin/env python3
"""
vit with prompt: also included different VPT ablations
带 Prompt 的 ViT：包含多种 VPT（Visual Prompt Tuning）消融变体。
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer(Transformer):
    """
    在标准 Transformer 基础上，加入“多种提示插入方式”的变体。
    支持的 prompt 位置（LOCATION）包括：
      - "prepend"：序列维，CLS 后、patch 前插入 P 个 token（常见 VPT 设置）
      - "add" / "add-1"：在序列维“元素相加”注入提示（分别为加到所有 token、加到长度=1 的广播）
      - "prepend-pixel" / "prepend-pixel-gaussian" / "prepend-pixel-imgnettransform"：
            先在像素块空间构造若干“虚拟 patch”，经 patch Embedding 后拼到序列前端
      - "pad"：在像素平面四周拼接可训练边框，再正常做 patch Embedding（序列长度不变，但图像尺寸变大）
      - "below"：在通道维上拼接可训练通道（由 3→3+P），再经 patch Embedding
    同时支持多种初始化（random / cls-based）与多种 deep-prompt 前向策略。
    """
    def __init__(self, prompt_config, config, img_size, vis):
        # 若使用像素“pad”提示：先把输入图像尺寸在高/宽两边各扩 NUM_TOKENS 像素
        # 目的是让 pad 出来的可训练边框参与后续的 patch 切分
        if prompt_config.LOCATION == "pad":
            img_size += 2 * prompt_config.NUM_TOKENS

        # 构建标准 ViT（embeddings + encoder）
        semantic_dim = getattr(prompt_config, "SEMANTIC_DIM", None)

        super(PromptedTransformer, self).__init__(
            config, img_size, vis, semantic_dim=semantic_dim)

        self.prompt_config = prompt_config  # 所有 prompt 的配置超参
        self.vit_config = config  # ViT 结构规格（层数、hidden_size、patch 等）

        img_size = _pair(img_size)  # 转成 (H, W)
        patch_size = _pair(config.patches["size"])  # 转成 (ph, pw)

        # 根据不同 LOCATION 确定“提示 token 的个数 self.num_tokens”
        # - add：提示将与“每个输入 token（CLS+patch）”相加，因此 num_tokens = 序列长度
        # - add-1：只准备 1 份，前向时广播到全部 token 上
        # - 其他：就是配置中指定的 P
        if self.prompt_config.LOCATION == "add":
            num_tokens = self.embeddings.position_embeddings.shape[1]
        elif self.prompt_config.LOCATION == "add-1":
            num_tokens = 1
        else:
            num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        # 仅作用于提示段的 dropout（训练期正则）
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        # 如果提示以低维 d 存储（PROJECT > -1），需要线性投影到 ViT 的 hidden_size（即 D）
        # 适用场景：跨不同模型复用同一份低维 prompt；或希望把参数规模集中在一层线性投影上
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            # 仅对 "prepend"/"add" 系列有意义（像素空间类位置不使用该投影）
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            # Kaiming 正态初始化（fan_out），有利于残差/深层稳定
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        # ======= 提示参数的初始化（INITIATION） =======
        if self.prompt_config.INITIATION == "random":
            # Xavier-uniform 的上下界：
            #   val = sqrt( 6 / ( 3 * ph * pw + prompt_dim ) )
            # 令提示与 patch 嵌入处于相近量级，便于融合
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            if self.prompt_config.LOCATION == "below":
                # 在通道维拼接提示（输入从 3 通道变成 3+P 通道），再做 patch Embedding
                self.embeddings.patch_embeddings = Conv2d(
                    in_channels=num_tokens+3,
                    out_channels=config.hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size
                )
                # add xavier_uniform initialization
                # 卷积权重/偏置初始化
                nn.init.uniform_(
                    self.embeddings.patch_embeddings.weight, -val, val)
                nn.init.zeros_(
                    self.embeddings.patch_embeddings.bias)

                # 可训练“通道平面”：形状 [1, P, H, W]，前向时 expand 到 batch
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, img_size[0], img_size[1]))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            elif self.prompt_config.LOCATION == "pad":
                # 在上下/左右方向各准备可训练“像素边框”
                # tb: [1, 3, 2P, H]；lr: [1, 3, H-2P, 2P]（拼接后得到四周一圈）
                self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * num_tokens, img_size[0]
                ))
                self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, img_size[0] - 2 * num_tokens, 2 * num_tokens
                ))

                # 用 [0,1] 均匀初始化（随后会用 ImageNet 归一化）
                nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
                nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

                self.prompt_norm = tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )

            elif self.prompt_config.LOCATION == "prepend-pixel":
                # 先构造“像素级”提示块（大小为 1×(P patch)），经 patch embedding 后当作序列前端的提示
                p_size = config.patches["size"][0]  # 16 or 14
                # [1, 3, ph, ph*P]：沿宽方向拼成 P 个 patch
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, 3, p_size, p_size * num_tokens))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                self.prompt_norm = nn.Identity()

            elif self.prompt_config.LOCATION == "prepend-pixel-gaussian":
                # 同上，但用高斯初始化像素提示
                p_size = config.patches["size"][0]  # 16 or 14
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, 3, p_size, p_size * num_tokens))

                nn.init.normal_(self.prompt_embeddings.data)
                self.prompt_norm = nn.Identity()

            elif self.prompt_config.LOCATION == "prepend-pixel-imgnettransform":
                # 同上，但用 [0,1] 均匀初始化 + ImageNet 归一化
                p_size = config.patches["size"][0]  # 16 or 14
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, 3, p_size, p_size * num_tokens))

                self.prompt_norm = tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
                nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)

            # including ['prepend', 'add', 'add-1']
            # 其余情况（"prepend" / "add" / "add-1"）：直接在“序列维”存储提示向量
            else:
                # [1, P, d]，前向时 expand → [B, P, d]，再过投影到 D
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

                # 若启用了 DEEP，在“序列维”路径下还可以配置 deep-prompt
                if self.prompt_config.LOCATION in ["prepend", "add", "add-1"] and self.prompt_config.DEEP:  # noqa

                    if self.prompt_config.NUM_DEEP_LAYERS is None:
                        # 标准 deep：从第 1 层到第 L-1 层共 L-1 份
                        total_d_layer = config.transformer["num_layers"]-1

                    else:
                        # 若只在“若干层”插入 deep：
                        if self.prompt_config.REVERSE_DEEP:
                            # 反向插：只在最后 NUM_DEEP_LAYERS 层插入
                            total_d_layer = self.prompt_config.NUM_DEEP_LAYERS
                            # 且删除浅层的 prompt（只用 deep）
                            del self.prompt_embeddings

                        else:
                            # 正向插：从第 1 层起插 NUM_DEEP_LAYERS-1 份
                            total_d_layer = self.prompt_config.NUM_DEEP_LAYERS - 1

                    # deep-prompt 参数：[L', P, d]
                    self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        total_d_layer, num_tokens, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            # val = 0.5
        elif self.prompt_config.INITIATION == "final-cls":
            # use final cls-token embds for all prompts
            # 使用“最终层 CLS token 的轨迹”作为提示（从 npy 载入）
            cls_embds = np.load(self.prompt_config.CLSEMB_PATH)
            cls_embds = torch.from_numpy(cls_embds).to(torch.float32)
            cls_embds = cls_embds[:, -1, :].unsqueeze(0)   # 只取“最后一层 CLS”
            assert num_tokens == cls_embds.shape[1]        # 形状对齐：[1, P, D]
            # (1,  num_tokens, 768)

            if self.prompt_config.LOCATION == "prepend":
                self.prompt_embeddings = nn.Parameter(cls_embds)

                if self.prompt_config.DEEP:  # noqa # 标准 deep：L-1 份
                    assert self.prompt_config.NUM_DEEP_LAYERS is None
                    total_d_layer = config.transformer["num_layers"] - 1
                    # (total_d_layer, num_tokens, prompt_dim) # 用相同的 CLS 复制到各层
                    self.deep_prompt_embeddings = nn.Parameter(
                        cls_embds.expand(total_d_layer, -1, -1))

        elif self.prompt_config.INITIATION == "cls-nolastl":
            # use the corresponding cls-token embds for all prompts, excluding the last output
            # 使用“每层 CLS（去掉最后一层）”作为提示：
            # 形状先变成 [L, P, D]，再对齐到 shallow & deep
            cls_embds = np.load(self.prompt_config.CLSEMB_PATH)
            cls_embds = torch.from_numpy(cls_embds).to(torch.float32)
            cls_embds = cls_embds[:, :-1, :].transpose(1, 0)
            assert num_tokens == cls_embds.shape[1]
            # (12,  num_tokens, 768)

            if self.prompt_config.LOCATION == "prepend":
                self.prompt_embeddings = nn.Parameter(cls_embds[:1, :, :])  # 第 0 层用第 0 份

                if self.prompt_config.DEEP:  # noqa
                    assert self.prompt_config.NUM_DEEP_LAYERS is None
                    # (total_d_layer, num_tokens, prompt_dim)
                    self.deep_prompt_embeddings = nn.Parameter(
                        cls_embds[1:, :, :])    # 其余层依次使用

        elif self.prompt_config.INITIATION == "cls-nofirstl":
            # use the corresponding cls-token embds for all prompts, excluding the first input
            cls_embds = np.load(self.prompt_config.CLSEMB_PATH)
            cls_embds = torch.from_numpy(cls_embds).to(torch.float32)
            cls_embds = cls_embds[:, 1:, :].transpose(1, 0)
            assert num_tokens == cls_embds.shape[1]
            # (12,  num_tokens, 768)

            if self.prompt_config.LOCATION == "prepend":
                self.prompt_embeddings = nn.Parameter(cls_embds[:1, :, :])

                if self.prompt_config.DEEP:  # noqa
                    assert self.prompt_config.NUM_DEEP_LAYERS is None
                    # (total_d_layer, num_tokens, prompt_dim)
                    self.deep_prompt_embeddings = nn.Parameter(
                        cls_embds[1:, :, :])
        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        """
        将 prompt 融入到“将要送入 Encoder 的输入”中。
        不同 LOCATION 的行为差异较大，注意每条路径的形状变化。
        """
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            # 标准 VPT：在 CLS 后插 P 个提示，再接所有 patch
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION in ["prepend-pixel", "prepend-pixel-gaussian", "prepend-pixel-imgnettransform"]:
            # 先把“像素提示块”过 patch Embedding → (1, D, 1, P) → (1, P, D)
            prompt_embeds = self.embeddings.patch_embeddings(
                self.prompt_norm(self.prompt_embeddings))  # (1, hidden_dim, 1, n_prompt)
            prompt_embeds = prompt_embeds.flatten(2).transpose(-1, -2)  # (1, n_prompt, hidden_dim)  # noqa
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    prompt_embeds.expand(B, -1, -1),
                    x[:, 1:, :]
                ), dim=1)

        elif self.prompt_config.LOCATION == "add-1":
            # 准备一份 (B,1,D) 的提示，广播加到整段序列（CLS+patch）
            # add to the input patches + CLS
            # assert self.prompt_config.NUM_TOKENS == x.shape[1]
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            L = x.shape[1]
            prompt_emb = self.prompt_dropout(self.prompt_proj(
                self.prompt_embeddings).expand(B, -1, -1))
            x = x + prompt_emb.expand(-1, L, -1)
            # (batch_size, cls_token + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "add":
            # 准备 (B, 1+N, D) 的提示，逐 token 相加
            # add to the input patches + CLS
            # assert self.prompt_config.NUM_TOKENS == x.shape[1]
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = x + self.prompt_dropout(
                self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
            # (batch_size, cls_token + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "pad":
            prompt_emb_lr = self.prompt_norm(
                self.prompt_embeddings_lr).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self.prompt_embeddings_tb).expand(B, -1, -1, -1)

            # 先左右拼接，再上下拼接（注意维度 -1/-2）
            x = torch.cat((
                prompt_emb_lr[:, :, :, :self.num_tokens],
                x, prompt_emb_lr[:, :, :, self.num_tokens:]
                ), dim=-1)
            x = torch.cat((
                prompt_emb_tb[:, :, :self.num_tokens, :],
                x, prompt_emb_tb[:, :, self.num_tokens:, :]
            ), dim=-2)
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "below":
            # (batch, 3, height, width)
            # 在通道维上拼接： (B, 3, H, W) → (B, 3+P, H, W)
            x = torch.cat((
                    x,
                    self.prompt_embeddings.expand(B, -1, -1, -1),
                ), dim=1)
            x = self.embeddings(x)
            # (batch_size, cls_token + n_patches, hidden_dim)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def train(self, mode=True):
        """
        训练/评估模式切换：
        - 训练时：冻结 ViT 主干（encoder/embeddings 置 eval），只训练 prompt 相关模块
         （prompt_proj、prompt_dropout、以及 prompt 参数本身）。
        - 评估时：恢复统一切换。
        """
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output, semantics=None):
        """
        “标准” deep-prompt 前向（逐层替换）：
        - 第0层：使用“前置 prompt 后”的序列直接进入；
        - 第1..L-1层：进入该层前，将序列中 [1:1+P] 的提示段替换为该层专属的 deep-prompt；
        - 支持 DEEP_SHARED=True：所有层共享“浅层 prompt”作为 deep（少见的变体）。
        返回：encoded（末端 LN 后）、attn_weights（若 vis=True 收集每层注意力图）
        """
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                # 第0层：直接吃“前置 prompt 后”的序列
                hidden_states, weights, semantics = self.encoder.layer[i](hidden_states, semantics, self.num_tokens)
            else:
                # 1..L-1 层：可选择替换“提示段”再进层
                if i <= self.deep_prompt_embeddings.shape[0]:
                    if self.prompt_config.DEEP_SHARED:
                        # use the same shallow prompt embd for all layers
                        # 所有层共用同一份浅层 prompt（极少使用）
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.prompt_embeddings).expand(B, -1, -1))
                    else:
                        # 正常：每层有自己的一份 deep prompt
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    # 对应不同 LOCATION 的替换/相加规则
                    if self.prompt_config.LOCATION == "prepend":
                        # 替换 [1:1+P] 段，保持序列长度不变
                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1+self.num_tokens):, :]
                        ), dim=1)

                    elif self.prompt_config.LOCATION == "add":
                        # 直接相加（形状需一致）
                        hidden_states = hidden_states + deep_prompt_emb

                    elif self.prompt_config.LOCATION == "add-1":
                        # 广播相加到每个 token
                        L = hidden_states.shape[1]
                        hidden_states = hidden_states + deep_prompt_emb.expand(
                            -1, L, -1)
                    else:
                        raise ValueError("prompt location {} is not supported".format(self.prompt_config.LOCATION))

                # 进入该层计算
                hidden_states, weights = self.encoder.layer[i](hidden_states, semantics, self.num_tokens)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_reverse_deep_prompt(self, x, semantics=None):
        """
        “反向” deep-prompt 前向：
        - 先不带任何 prompt 跑前面若干层；
        - 只在“最后 NUM_DEEP_LAYERS 层”插入 deep-prompt，再继续前向。
        适用于：希望仅在高层语义处注入任务提示的场景。
        """
        hidden_states = self.embeddings(x)

        attn_weights = []
        weights = None
        B = x.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]
        num_deep_layers = self.deep_prompt_embeddings.shape[0]
        assert num_deep_layers == self.prompt_config.NUM_DEEP_LAYERS

        # no prompt
        # （一）前半段：不插 prompt
        for i in range(num_layers - num_deep_layers):
            hidden_states, weights, semantics = self.encoder.layer[i](hidden_states, semantics, 0)
            if self.encoder.vis:
                attn_weights.append(weights)

        # insert prompt
        # （二）后半段：逐层插入 deep prompt
        for deep_idx in range(num_deep_layers):
            i = num_layers - num_deep_layers + deep_idx
            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                self.deep_prompt_embeddings[deep_idx]).expand(B, -1, -1))

            if self.prompt_config.LOCATION == "prepend":
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    deep_prompt_emb,
                    hidden_states[:, (1+self.num_tokens):, :]
                ), dim=1)
            elif self.prompt_config.LOCATION == "add":
                hidden_states = hidden_states + deep_prompt_emb

            elif self.prompt_config.LOCATION == "add-1":
                L = hidden_states.shape[1]
                hidden_states = hidden_states + deep_prompt_emb.expand(
                    -1, L, -1)
            else:
                raise ValueError("prompt location {} is not supported".format(self.prompt_config.LOCATION))

            hidden_states, weights, semantics = self.encoder.layer[i](hidden_states, semantics)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_noexpand_deep_prompt(self, embedding_output, semantics=None):
        """
        “不扩展长度”的 deep-prompt 前向（只支持 prepend）：
        - 在前若干层临时插入（替换）deep-prompt；
        - 到第 K+1 层入口时，将提示段“移除”，序列恢复成原始长度（1+N）；
        - 后续层以原长度继续前向。
        适用于：希望只在底/中层注入提示，但又不想一直带着“更长的序列”走完整个网络。
        """
        # insert deep prompts up to some layers, and reduce the input sequence back to the original
        if self.prompt_config.LOCATION != "prepend":
            raise ValueError("prompt location {} is not supported".format(self.prompt_config.LOCATION))
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]
        prompt_len = self.num_tokens


        for i in range(num_layers):
            if i == 0:
                hidden_states, weights, semantics = self.encoder.layer[i](embedding_output, semantics, prompt_len)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    # 在第 1..K 层入口用 deep prompt 替换
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)

                elif i == self.deep_prompt_embeddings.shape[0] + 1:
                    # 在第 K+1 层入口“移除提示段”，恢复长度
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)
                    prompt_len = 0


                hidden_states, weights, semantics = self.encoder.layer[i](hidden_states, semantics, prompt_len)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x, semantics=None):
        """
        总前向入口：根据配置选择三条 deep-prompt 路径之一；
        若未启用 deep，则走标准“prepend/add/... + encoder”的路径。
        """
        if self.prompt_config.REVERSE_DEEP:
            # 只在最后若干层插入 deep
            encoded, attn_weights = self.forward_reverse_deep_prompt(x, semantics)

        elif self.prompt_config.FORWARD_DEEP_NOEXPAND:
            # 先并入“前置 prompt”，再走“不扩展长度”的 deep 前向
            embedding_output = self.incorporate_prompt(x)
            encoded, attn_weights = self.forward_noexpand_deep_prompt(
                embedding_output, semantics)

        else:
            # this is the default version: # 默认：标准路径
            embedding_output = self.incorporate_prompt(x)

            if self.prompt_config.DEEP:
                encoded, attn_weights = self.forward_deep_prompt(
                    embedding_output, semantics)
            else:
                encoded, attn_weights = self.encoder(embedding_output, semantics)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    """
    顶层封装：复用 VisionTransformer 的分类头/接口，
    仅替换内部 transformer 为带多消融能力的 PromptedTransformer。
    另外提供多种“池化”策略（如何从序列中取最终表示）：
      - original：取 CLS
      - imgprompt_pool：对“提示+patch”的整段做平均（排除 CLS）
      - img_pool：只对 patch 段做平均（排除 CLS+提示）
      - prompt_pool：只对提示段做平均
    """
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False, semantics=None):
        # 得到编码后的整段序列（可能含 CLS + Prompt + Patch）
        x, attn_weights = self.transformer(x, semantics)

        # 依据池化策略选择输出向量
        if self.prompt_cfg.VIT_POOL_TYPE == "original":
            x = x[:, 0] # 取 CLS
        elif self.prompt_cfg.VIT_POOL_TYPE == "imgprompt_pool":
            # 要求为 prepend：对 CLS 以外的全部 token（提示+patch）平均
            assert self.prompt_cfg.LOCATION == "prepend"
            x = x[:, 1:, :].mean(dim=1)
        elif self.prompt_cfg.VIT_POOL_TYPE == "img_pool":
            # 只对 patch 段平均（排除 CLS + 提示）
            assert self.prompt_cfg.LOCATION == "prepend"
            x = x[:, self.transformer.num_tokens+1:, :].mean(dim=1)
        elif self.prompt_cfg.VIT_POOL_TYPE == "prompt_pool":
            # 只对提示段平均
            assert self.prompt_cfg.LOCATION == "prepend"
            x = x[:, 1:self.transformer.num_tokens+1, :].mean(dim=1)
        else:
            raise ValueError("pooling type for output is not supported")

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights

    def load_from(self, weights):
        """
        从 JAX/TF 风格权重加载到当前模型（包含：
        patch embedding / CLS / 末端 LN / 位置编码（如需插值）/ 各层 Block）。
        当 LOCATION == "below" 时，patch Embedding 的输入通道变为 3+P，
        因此只拷贝前 3 个通道的预训练权重到对应位置。
        """
        with torch.no_grad():
            if self.transformer.prompt_config.LOCATION != "below":
                # 标准：完整拷贝 HWIO → OIHW
                self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            else:
                # [768, 4, 16, 16]
                # below: 仅把预训练的 RGB 三通道拷到前 3 个通道
                # 形状示意：[768, 4, 16, 16]（若 P=1），前 3 个为 RGB，后 1 个为可训练通道
                self.transformer.embeddings.patch_embeddings.weight[:, :3, :, :].copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            # CLS / 编码器末端 LayerNorm
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # 位置编码（若输入分辨率不同，需要插值到新网格）
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                # 旧/新网格边长（方阵）
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                # 双线性插值到新网格
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)

                # 拼回（若带 CLS）
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # 逐层加载 Encoder Block（Q/K/V/Out、FFN、两处 LN）
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # 若使用 hybrid 前端（ResNetV2），同时加载其权重
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

