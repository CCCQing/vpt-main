"""
Pre-ViT prompt distribution modules for ZSL/GZSL.

在 ViT 编码器 *之前* 构造提示分布的模块。

整体思路：
- 对早期的视觉 token `V_raw` 做池化，得到一个低维的视觉统计向量 `h_v`；
- 用一个“后验头”估计高斯后验 q(z|x) 的均值 mu 与 log 方差 logvar，并使用重参数化技巧采样 z；
- 将隐变量 z（可以与语义属性拼接）解码为一组 prompt tokens，
  这些 prompt tokens 会在输入序列维度上与 [CLS]、patch tokens 进行拼接送入 ViT。

设计目标：
- 能够直接插入现有 VPT 的 prompt 注入流程（[CLS] + prompt + patch），
  ViT 只需要知道 prompt_len，而不需要知道 prompt 是如何由分布生成的。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    使用重参数化技巧从高斯后验中采样：
        z = mu + sigma * eps,
        其中 sigma = exp(0.5 * logvar), eps ~ N(0, I)

    这样采样 z 的过程对 mu、logvar 是可导的，方便做 KL 约束。
    """
    std = torch.exp(0.5 * logvar)        # 标准差 sigma，保持非负
    eps = torch.randn_like(std)          # 与 std 同形状的标准正态噪声
    return mu + eps * std               # 采样得到 z


class VisualStatsEncoder(nn.Module):
    """
    将原始视觉 token 序列 V_raw 编码为一个全局统计向量 h_v 的模块。

    支持的池化模式：
    - "gap"   : global average pooling，全局平均池化；
    - "gem"   : generalized mean pooling，带可学习指数的广义均值池化；
    - "attnpool": 单查询的注意力池化（类似 CLIP 的 AttentionPool2d 思路）；
    - "gated" : gated sum，给每个 token 学一个 gate，再加权求和归一化。
    """

    def __init__(self, dim: int, pool: str = "gap"):
        """
        参数：
            dim  : 每个视觉 token 的维度 D；
            pool : 池化类型字符串，见上。
        """
        super().__init__()
        self.pool = pool
        if pool == "attnpool":
            # 注意力池化中使用的 query 向量，维度与 token 相同
            self.query = nn.Parameter(torch.randn(dim))
        elif pool == "gem":
            # GeM 的指数 p，可学习，初始化为 3.0（常见的经验值）
            self.p = nn.Parameter(torch.ones(1) * 3.0)
        elif pool == "gated":
            # Gated pooling 使用的 gate 线性层：对每个 token 输出一个标量 gate
            self.gate = nn.Linear(dim, 1)
        elif pool != "gap":
            # 非法的池化类型给出报错
            raise ValueError(f"Unsupported pooling mode: {pool}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        输入：
            tokens: [B, L, D]，B 为 batch 大小，L 为 token 数，D 为通道维度。

        输出：
            h_v: [B, D]，每个样本对应一个全局视觉统计向量。
        """
        if self.pool == "gap":
            # 简单的 Token 维平均
            return tokens.mean(dim=1)

        if self.pool == "gem":
            # GeM: (1/L * sum(x^p))^(1/p)
            # 这里使用一个共享的 p 参数，并对输入做 clamp 防止数值问题
            p = torch.clamp(self.p, min=1e-3)
            # 先对 token 维求平均，再做 1/p 次幂
            return torch.pow(tokens.clamp(min=1e-6).mean(dim=1), 1.0 / p)

        if self.pool == "attnpool":
            # 单查询注意力池化：
            # 对每个 token 计算与 query 的点积，做 softmax 得权重，再按权重加权求和
            q = self.query.to(tokens.dtype)                 # 保证与 tokens 的 dtype 一致（兼容 AMP）
            # [B, L, D] @ [D] -> [B, L]
            attn = torch.matmul(tokens, q) / math.sqrt(tokens.size(-1))
            weights = attn.softmax(dim=1)                   # 在 token 维度做 softmax
            # 按权重对 tokens 加权求和：sum_l w_l * token_l
            return torch.einsum("bl, bld -> bd", weights, tokens)

        if self.pool == "gated":
            # Gated pooling:
            # 用一层线性层产生 gate，再经过 sigmoid 映射到 (0,1)
            gates = torch.sigmoid(self.gate(tokens))        # [B, L, 1]
            # 对每个 token 乘上 gate 系数
            gated_tokens = tokens * gates                   # [B, L, D]
            # 归一化因子：所有 gate 的和，防止全 0 用 clamp
            denom = gates.sum(dim=1).clamp(min=1e-6)        # [B, 1]
            # 加权和除以因子 -> 类似“加权平均”
            return gated_tokens.sum(dim=1) / denom          # [B, D]

        # 理论上不应该到这里，因为非法模式在 __init__ 中已经抛异常
        raise ValueError(f"Unsupported pooling mode: {self.pool}")


class PosteriorHead(nn.Module):
    """
    后验推断头（amortized posterior head）：

    输入一个统计向量 h（例如 h_v），输出两路：
        - mu     : 高斯后验的均值向量
        - logvar : 高斯后验对角协方差的 log 方差

    内部结构为两个独立的 MLP（mu_head 和 logvar_head），共享输入 h。
    """

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        """
        参数：
            in_dim    : 输入统计向量 h 的维度；
            hidden_dim: 中间隐藏层维度；
            latent_dim: 潜变量 z 的维度。
        """
        super().__init__()
        # 均值分支
        self.mu_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # log 方差分支
        self.logvar_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入：
            h: [B, in_dim] 视觉统计向量

        输出：
            mu:     [B, latent_dim]
            logvar: [B, latent_dim]
        """
        return self.mu_head(h), self.logvar_head(h)


class PromptGenerator(nn.Module):
    """
    将潜变量 z（以及可选的语义属性向量）解码为一组 prompt tokens 的模块。

    Args:
        latent_dim:  隐变量 z 的维度；
        prompt_dim:  输出的 prompt token 维度，应与 ViT 的 hidden_size 一致；
        prompt_len:  需要生成的 prompt token 个数；
        hidden_dim:  解码器内部的隐藏层维度；
        semantic_dim: 若不为 None，则表示将语义属性拼接到 z 上进行条件生成，
                      语义向量的维度。
    """

    def __init__(
        self,
        latent_dim: int,
        prompt_dim: int,
        prompt_len: int,
        hidden_dim: int,
        semantic_dim: Optional[int] = None,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.semantic_dim = semantic_dim or 0

        # 若引入语义条件，则先在特征维度上与 z 拼接
        fusion_dim = latent_dim + self.semantic_dim
        # 先做一次线性变换，把 (z, s) 融合到一个 hidden 向量
        self.fusion = nn.Linear(fusion_dim, hidden_dim)
        # 后续 MLP：hidden -> hidden -> (prompt_len * prompt_dim)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_len * prompt_dim),
        )
        self.prompt_dim = prompt_dim

    def forward(self, z: torch.Tensor, semantics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入：
            z:         [B, latent_dim] 潜变量样本；
            semantics: [B, semantic_dim] 或 None，语义条件向量。

        输出：
            prompts: [B, prompt_len, prompt_dim] 生成的提示 tokens。
        """
        if self.semantic_dim > 0:
            if semantics is None:
                # 若启用了语义条件但未提供语义，使用零向量占位，保持维度一致
                semantics = torch.zeros(z.size(0), self.semantic_dim, device=z.device, dtype=z.dtype)
            elif semantics.size(-1) != self.semantic_dim:
                raise ValueError(
                    f"Expected semantic_dim={self.semantic_dim}, got {semantics.size(-1)}"
                )
            # 条件生成：将 z 与语义向量在特征维度上拼接
            fused = torch.cat([z, semantics], dim=-1)
        else:
            fused = z
        # 先映射到 hidden 维度
        h = self.fusion(fused)                       # [B, hidden_dim]
        # MLP 解码到 (prompt_len * prompt_dim)
        prompts = self.mlp(h)                        # [B, prompt_len * prompt_dim]
        # 再 reshape 成 [B, L_p, D]
        prompts = prompts.view(-1, self.prompt_len, self.prompt_dim)
        return prompts


class PreViTPromptDistributor(nn.Module):
    """
    将早期视觉 tokens（以及可选的语义属性）映射为 prompt tokens 的端到端封装模块。

    典型调用方式：
        prompts, stats = module(V_raw, S_raw)

    其中：
        - V_raw: [B, L, D]，为 ViT patch+pos 编码后的视觉 tokens（可只取第一层或若干层输出）；
        - S_raw: [B, d_s] 或 [B, T, d_s]，为原始语义属性（可选）；
        - prompts: [B, prompt_len, D]，可直接拼到 ViT 的输入序列里；
        - stats: dict，包含 mu/logvar/z/h_v，用于做 KL 正则或分析。
    """

    def __init__(
        self,
        dim: int,
        prompt_len: int,
        latent_dim: int,
        hidden_dim: int,
        pool: str = "gap",
        semantic_dim: Optional[int] = None,
        semantic_proj_dim: Optional[int] = None,
    ):
        """
        参数：
            dim               : 视觉 token 的通道维度 D（即 ViT hidden_size）；
            prompt_len        : 生成的 prompt token 个数；
            latent_dim        : 潜变量 z 维度；
            hidden_dim        : 后验头与生成器中的隐藏维度；
            pool              : 视觉池化方式，传给 VisualStatsEncoder；
            semantic_dim      : 原始语义属性维度（若有）；
            semantic_proj_dim : 若不为 None，则先将语义从 semantic_dim 映射到该维度，
                                再与 z 拼接进入 PromptGenerator。
        """
        super().__init__()
        # 在做统计前先对 V_raw 做 LayerNorm，相当于 “LN(V_raw)” 的步骤
        self.norm = nn.LayerNorm(dim)
        # 视觉统计编码器：LN 后的 V_raw -> h_v
        self.visual_encoder = VisualStatsEncoder(dim, pool=pool)
        # 后验头：h_v -> (mu, logvar)
        self.posterior = PosteriorHead(dim, hidden_dim, latent_dim)
        # 提示生成器：z (+ 语义) -> prompt tokens
        self.prompt_generator = PromptGenerator(
            latent_dim=latent_dim,
            prompt_dim=dim,
            prompt_len=prompt_len,
            hidden_dim=hidden_dim,
            semantic_dim=semantic_proj_dim or semantic_dim,
        )
        # 语义投影层（可选）：将原始语义 S_raw 投影到 semantic_proj_dim
        if semantic_dim is not None and semantic_proj_dim is not None:
            self.semantic_proj = nn.Linear(semantic_dim, semantic_proj_dim)
        else:
            self.semantic_proj = None

    def forward(
        self, V_raw: torch.Tensor, S_raw: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        输入：
            V_raw: [B, L, D]，视觉 tokens（通常是 ViT 的 patch+pos 编码输出）；
            S_raw: [B, d_s] 或 [B, T, d_s]（可选），原始语义属性。

        输出：
            prompts: [B, prompt_len, D]，生成的 prompt tokens；
            stats:   dict，包含若干中间量：
                     - "mu"   : [B, latent_dim]，高斯后验均值；
                     - "logvar": [B, latent_dim]，log 方差；
                     - "z"    : [B, latent_dim]，采样得到的潜变量；
                     - "h_v"  : [B, D]，视觉统计向量。
        """
        # 1) 对视觉 tokens 做 LN
        V_norm = self.norm(V_raw)                  # [B, L, D]
        # 2) 视觉池化，得到 h_v
        h_v = self.visual_encoder(V_norm)          # [B, D]
        # 3) 后验头输出 mu 与 logvar
        mu, logvar = self.posterior(h_v)           # 各为 [B, latent_dim]
        # 4) 重参数化采样 z
        z = _reparameterize(mu, logvar)            # [B, latent_dim]

        # 5) 语义条件处理（若提供）
        if S_raw is not None:
            if self.semantic_proj is not None:
                # 若指定 semantic_proj_dim，则先把 S_raw 投影到该维度
                S_cond = self.semantic_proj(S_raw)
            else:
                S_cond = S_raw
            # 若语义是三维 [B, T, d_s]，例如多属性向量，
            # 这里简单对 T 维做平均，得到一个全局语义表示
            if S_cond.dim() == 3:
                S_cond = S_cond.mean(dim=1)        # [B, d_s]
        else:
            S_cond = None

        # 6) 由 z（以及可选语义）生成 prompt tokens
        prompts = self.prompt_generator(z, semantics=S_cond)  # [B, prompt_len, D]

        # 7) 打包中间统计量，便于在外部构造 KL loss 或可视化
        stats = {"mu": mu, "logvar": logvar, "z": z, "h_v": h_v}
        return prompts, stats


__all__ = [
    "PreViTPromptDistributor",
    "VisualStatsEncoder",
    "PosteriorHead",
    "PromptGenerator",
    "generate_prompt_init",
]

def generate_prompt_init(
    distributor: PreViTPromptDistributor,
    V_raw: torch.Tensor,
    S_raw: Optional[torch.Tensor] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """利用预 ViT 提示分布模块生成一次性的 prompt 初始化张量。

    Args:
        distributor: 预先构建好的 ``PreViTPromptDistributor`` 实例。
        V_raw: 形状 (B, L, D) 的早期视觉 token（含位置编码），通常可
            取训练集的一个 batch 进行初始化。
        S_raw: （可选）形状 (B, M, d_sem) 的语义属性或类原型。
        reduce: 将 batch 维合并为单个初始化向量的方式，目前支持 "mean"
            和 "first"，分别表示对 batch 平均或取第一条样本。

    Returns:
        prompt_init: 形状 (1, prompt_len, D) 的张量，可直接传给
            ``PromptedVisionTransformer(prompt_init=...)`` 用于一次性初始化。
    """
    with torch.no_grad():
        prompts, _ = distributor(V_raw, S_raw)
        if reduce == "first":
            prompts = prompts[:1]
        elif reduce == "mean":
            prompts = prompts.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
        return prompts.detach()