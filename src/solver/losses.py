#!/usr/bin/env python3
"""
损失函数定义文件：
- 基础分类损失：SoftmaxLoss（标准多类交叉熵）
- 分类 + 提示对齐组合损失：SoftmaxWithPromptAlignLoss（CE + L_avg）
- RSimilarityClassifier：基于共享概念基 R 的相似度分类头（不是损失，而是一个 head）
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.vit_prompt.vit import SharedConceptAligner
from typing import Any, Dict, Optional


# ===========================
# 一些小工具函数
# ===========================
def norm1(u: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    类似 L1 的归一化：
        Norm1(u) = u / (sum_j u_j + eps)
    用在注意力分布上，把某一维度上的值约成“和为 1”的概率分布。
    """
    denom = u.sum(dim=dim, keepdim=True) + eps
    return u / denom

def l_avg(
    attn_pv: torch.Tensor,
    attn_vs: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算单层的 L_avg^(l) 损失（提示–视觉–语义三者中的“提示–视觉”对齐项）。

    约定：
        attn_pv: [B, T, N]，提示 → patch 的注意力 A_PV^(l)
                 B：batch_size；T：prompt 数；N：patch 数
        attn_vs: [B, N, M]，patch → semantic 的注意力 A_VS^(l)
                 M：语义 token / 属性 slot 数

    公式对应关系（和你设想里的那一段）：
    1）image-level 语义分布 π(x)：对每张图，把所有 patch→语义注意力按 patch 求和再做归一化：
            π(x) = Norm1( sum_n A_VS[n, :] )
       实现：pi = norm1(attn_vs.sum(dim=1))        # [B, M]

    2）patch 责任度 r(x)：一个 patch 在整张图语义分布下的重要性，
       按 A_VS 与 π(x) 做加权和，再做一次归一化：
            r = Norm1( A_VS * π(x) )
       实现：matmul + squeeze，得到 [B, N]

    3）平均提示注意力 a_bar(x)：对提示维度求平均：
            a_bar = mean_t A_PV[t, n]
       实现：attn_pv.mean(dim=1) → [B, N]

    4）L_avg^(l)：对比“平均提示关注的 patch 分布 a_bar(x)”和“语义责任度 r(x)”，
       在 patch 维度上做 L2 距离：
            L_avg(x) = sum_n (a_bar[n] - r[n])^2

    reduction:
        - "none": 返回 [B]，每张图一个 loss
        - "mean": batch 上平均 → 标量
        - "sum":  batch 上求和 → 标量
    """

    # 1) π(x)：对 patch 维度求和再做 Norm1，形状 [B, M]
    pi = norm1(attn_vs.sum(dim=1), dim=-1, eps=eps)  # sum over patches

    # 2) r(x)：A_VS * π(x)，先 [B, N, M] @ [B, M, 1] → [B, N, 1]，再 squeeze 成 [B, N]
    r = torch.matmul(attn_vs, pi.unsqueeze(-1)).squeeze(-1)
    r = norm1(r, dim=-1, eps=eps)

    # 3) a_bar(x)：对所有 prompt 取平均，得到 [B, N]
    a_bar = attn_pv.mean(dim=1)

    # 4) L2 差的和：每个样本一个标量 [B]
    per_sample = ((a_bar - r) ** 2).sum(dim=-1)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    # default: mean
    return per_sample.mean()


def l_avg_multi(
    attn_pv_dict: Dict[int, torch.Tensor],
    attn_vs_dict: Dict[int, torch.Tensor],
    layers: Any,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    把多层的 L_avg^(l) 聚合起来。

    参数：
        attn_pv_dict: {层号 l → [B, T, N] A_PV^(l)}
        attn_vs_dict: {层号 l → [B, N, M] A_VS^(l)}
        layers:       要参与计算的层列表，如 [0, 1, 2, ...]
    做法：
        - 对每一层单独算 l_avg(..., reduction="none") 得到 [B]
        - 在“层维度”上取平均，得到每个样本一个数
        - 再按 reduction 决定是 batch-mean 还是 batch-sum
    """
    losses = []
    for l in layers:
        losses.append(l_avg(attn_pv_dict[l], attn_vs_dict[l], eps=eps, reduction="none"))
    # losses: [num_layers, B] → 在层上平均 → [B]
    stacked = torch.stack(losses, dim=0).mean(dim=0)
    if reduction == "none":
        return stacked
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


# ===========================
# 1. 基础 Softmax 分类损失
# ===========================
class SoftmaxLoss(nn.Module):
    """
    标准的 Softmax + CrossEntropy 分类损失。
    """
    def __init__(self, cfg=None):
        # 这里 cfg 暂时没用，但保留接口方便以后扩展（比如从 cfg 里读 class_weight 等）
        super().__init__()

    def is_single(self):
        """返回 True，表示这是“单一分支”的损失函数。"""
        return True

    def is_local(self):
        """
        返回 False，表示这不是“局部 patch-level 损失”，
        而是对整张图片 / 整个样本的全局分类损失。
        Trainer 里会用这个标志来区分：是否需要把 model / inputs 也传给 loss。
        """
        return False

    def loss(self, logits, targets, per_cls_weights, kwargs=None):
        """
        使用 F.cross_entropy 计算多类交叉熵损失。

        参数:
            logits: [B, C]，模型输出的类别分数（未过 softmax）
            targets: [B]，整型类别 id（0~C-1）
            per_cls_weights: 长度为 C 的类别权重列表或张量
                             一般来自 dataset.get_class_weights(...)，
                             为全 1 则表示不做重加权。
            kwargs: 当前没用，预留扩展。

        返回:
            标量损失值（对 batch 做平均）。
        """
        # 把 python list 转成放在同一设备上的 tensor
        weight = torch.tensor(per_cls_weights, device=logits.device)

        # reduction="none"：先得到 [B]，方便以后做自定义聚合
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        # 这里选择对 batch 做简单平均
        return torch.sum(loss) / targets.shape[0]

    def forward(self, pred_logits, targets, per_cls_weights, kwargs=None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


# ===========================
# 2. Softmax + 提示对齐组合损失
# ===========================
class SoftmaxWithPromptAlignLoss(nn.Module):
    """
    基础分类交叉熵 + 提示-视觉注意力对齐正则（L_avg）。

    - 主干仍是标准 softmax 交叉熵（SoftmaxLoss）。
    - 若提供 attn_pv / attn_vs（单层或多层）且权重 alpha>0，
      则额外叠加 L_avg 对齐项：
          total = CE + alpha * L_avg
    - 兼容三种 logits 表达形式：
        * 直接 tensor
        * (logits, aux) 元组
        * {"logits": tensor, ...} 字典
      aux 中存放从前向传回来的亲和 / 注意力等辅助信息。
    """

    def __init__(self, cfg=None):
        super().__init__()
        # 来自配置的对齐损失权重 alpha（默认为 0 表示只用 CE）
        self.alpha = getattr(cfg.SOLVER, "LOSS_ALPHA", 0.0) if cfg is not None else 0.0
        # 复用已有的 SoftmaxLoss 实现，保证接口一致
        self.cls_loss = SoftmaxLoss(cfg)

    def is_single(self):
        return True

    def is_local(self):
        return False

    def _extract_logits_and_aux(self, pred_logits: Any, kwargs: Optional[Dict[str, Any]]):
        """
        兼容多种 logits/aux 表达形式，统一抽取出：
            - logits: [B, C]
            - aux: dict 或 None（里面可能有 attn_pv / attn_vs）
        """

        aux = None
        logits = pred_logits

        # 情况一：传进来的是 (logits, aux) 或 [logits, aux]
        if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
            logits = pred_logits[0]
            # 若第二个元素是 dict，则认为它是 aux 信息
            if len(pred_logits) > 1 and isinstance(pred_logits[1], Dict):
                aux = pred_logits[1]

        # 情况二：传进来的是 {"logits": tensor, ...}
        if isinstance(pred_logits, Dict) and "logits" in pred_logits:
            logits = pred_logits["logits"]
            aux = pred_logits   # 整个字典都当成 aux

        # 兜底：如果 aux 还是 None，可以从 kwargs["aux"] 里取
        if aux is None and kwargs is not None and isinstance(kwargs, Dict):
            aux = kwargs.get("aux")

        return logits, aux

    def _compute_align_loss(self, aux: Optional[Dict[str, Any]]):
        """
        从 aux 中解析注意力张量，计算 L_avg 对齐项。

        期望 aux 至少包含：
            aux["attn_pv"]: [B, T, N] 或 {层 → [B, T, N]}
            aux["attn_vs"]: [B, N, M] 或 {层 → [B, N, M]}
        若找不到或 alpha <= 0，则返回 None，表示不加对齐正则。
        """

        if aux is None or self.alpha <= 0:
            return None

        attn_pv = aux.get("attn_pv") if isinstance(aux, Dict) else None
        attn_vs = aux.get("attn_vs") if isinstance(aux, Dict) else None

        if attn_pv is None or attn_vs is None:
            return None

        # 多层形式：用 l_avg_multi 聚合
        if isinstance(attn_pv, Dict) and isinstance(attn_vs, Dict):
            layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()))
            if not layers:
                return None
            return l_avg_multi(attn_pv, attn_vs, layers=layers, reduction="mean")

        # 单层 tensor 形式
        if torch.is_tensor(attn_pv) and torch.is_tensor(attn_vs):
            return l_avg(attn_pv, attn_vs, reduction="mean")

        return None

    def loss(self, logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        组合损失：
          1. 先抽出 logits / aux；
          2. 用 SoftmaxLoss 算基础 CE；
          3. 视情况叠加 alpha * L_avg。
        """
        logits, aux = self._extract_logits_and_aux(logits, kwargs)

        # 基础 CE 分类损失
        base = self.cls_loss.loss(logits, targets, per_cls_weights, kwargs)

        # 对齐正则项
        align = self._compute_align_loss(aux)

        if align is None:
            return base
        return base + self.alpha * align

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


# ===========================
# 3. 损失名到类的映射
# ===========================
LOSS = {
    "softmax": SoftmaxLoss,
    "softmax_prompt_align": SoftmaxWithPromptAlignLoss,

}

# ===========================
# 4. 基于共享 R 的相似度分类头（不是损失）
# ===========================
class RSimilarityClassifier(nn.Module):
    """
    使用共享概念基 R 生成的语义原型与 CLS 对齐的分类头。

    思路：
      - 视觉侧：取 CLS 特征 cls_feat（[B, D]），经过可选 Linear 投影到 d 维。
      - 语义侧：给定每个类别的属性向量 class_attr（[C, A]），
                通过 SharedConceptAligner.encode_semantics_only 映射到 R 空间得到 [C, D]，
                再经过 Linear 投影到同一个 d 维。
      - 分类：计算 visual 与 semantic 原型的点积或余弦相似度，
              再乘一个可学习的温度 logit_scale，得到 [B, C] 的 logits。

    这个模块是“分类头”，通常挂在 ViT/PromptedTransformer 的外面，
    用来替代单纯的 Linear MLP head。
    """

    def __init__(
        self,
        semantic_concept: SharedConceptAligner,
        class_attr: torch.Tensor,
        hidden_size: int,
        proj_dim: int = None,
        use_cosine: bool = True,
        logit_scale_init: float = 10.0,
        visual_proj_enable: bool = False,
        fixed_logit_scale: float = 0.0,
    ) -> None:
        super().__init__()
        # 若未指定投影维度，则默认等于 hidden_size
        if proj_dim is None or proj_dim <= 0:
            proj_dim = hidden_size

        # 共享概念基模块（用于把属性映射到 R 空间）
        self.semantic_concept = semantic_concept
        # 类别属性缓存在 buffer 中（不参与梯度）
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]

        # 是否对视觉 CLS 再做一次 Linear 投影
        self.visual_proj_enabled = visual_proj_enable
        out_dim = proj_dim if self.visual_proj_enabled else hidden_size
        self.visual_proj = (
            nn.Linear(hidden_size, out_dim) if self.visual_proj_enabled else None
        )
        # 语义原型的投影层
        self.semantic_proj = nn.Linear(hidden_size, out_dim)

        self.use_cosine = use_cosine
        self.fixed_logit_scale = float(fixed_logit_scale)
        self._debug_last_raw_sim = None
        self._debug_last_scaled_logits = None
        if use_cosine:
            # 可学习温度：实际使用时是 exp(logit_scale)
            self.logit_scale = nn.Parameter(
                torch.log(torch.tensor(logit_scale_init, dtype=torch.float32))
            )
        else:
            self.logit_scale = None

    def _class_prototypes(self) -> torch.Tensor:
        """
        通过共享概念基获取 R 空间中的语义原型：
            class_attr [C, A] → SharedConceptAligner → [C, D]
        """

        attr = self.class_attr.to(self.semantic_concept.concept_slots.device)
        return self.semantic_concept.encode_semantics_only(attr)

    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        """
        输入：
            cls_feat: [B, D]，来自 ViT 的 CLS 特征

        输出：
            logits: [B, C]，每个类别的得分
        """

        prototypes = self._class_prototypes()  # [C, D]
        # 视觉侧：可选投影
        visual = self.visual_proj(cls_feat) if self.visual_proj else cls_feat  # [B, d]
        # 语义侧：线性投影到同一 d 维
        semantic = self.semantic_proj(prototypes)  # [C, d]

        if self.use_cosine:
            # 余弦相似度 + 温度缩放
            visual = F.normalize(visual, dim=-1)
            semantic = F.normalize(semantic, dim=-1)
            raw_sim = visual @ semantic.t()
            if self.fixed_logit_scale > 0:
                scale = raw_sim.new_tensor(self.fixed_logit_scale)
            else:
                scale = self.logit_scale.exp()
            logits = raw_sim * scale
            self._debug_last_raw_sim = raw_sim.detach()
            self._debug_last_scaled_logits = logits.detach()
        else:
            # 直接点积相似度
            logits = visual @ semantic.t()
            self._debug_last_raw_sim = logits.detach()
            self._debug_last_scaled_logits = logits.detach()

        return logits



def build_loss(cfg):
    """
    根据配置构建损失函数实例。

    使用方式（伪代码）：
        loss_fn = build_loss(cfg)
        loss = loss_fn(logits, targets, per_cls_weights)

    其中：
        cfg.SOLVER.LOSS 必须是 LOSS 字典中的一个键，比如：
            - "softmax"               → 纯 SoftmaxLoss
            - "softmax_prompt_align"  → Softmax + L_avg 对齐正则

    返回:
        nn.Module 子类实例（如 SoftmaxLoss / SoftmaxWithPromptAlignLoss），
        若名字不在 LOSS 中则抛出断言错误。
    """
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)
