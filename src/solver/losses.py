#!/usr/bin/env python3
"""
损失函数
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.vit_prompt.vit import SharedConceptAligner
from typing import Any, Dict, Optional


def norm1(u: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    L1-like normalization along `dim`:
        Norm1(u) = u / (sum_j u_j + eps)
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
    Compute L_avg^(l) for one layer.

    Args:
        attn_pv: [B, T, N] prompt->patch attention A_PV^(l)
        attn_vs: [B, N, M] patch->semantic attention A_VS^(l)
        eps: small constant for numerical stability in Norm1
        reduction: "none" | "mean" | "sum"
            - "none": return per-sample loss [B]
            - "mean": return scalar average over batch
            - "sum":  return scalar sum over batch

    Returns:
        If reduction == "none": tensor of shape [B]
        Else: scalar tensor
    """
    # 1) image-level semantic distribution pi(x): [B, M]
    pi = norm1(attn_vs.sum(dim=1), dim=-1, eps=eps)  # sum over patches

    # 2) patch responsibility r(x): [B, N]
    # matmul: [B, N, M] @ [B, M, 1] -> [B, N, 1]
    r = torch.matmul(attn_vs, pi.unsqueeze(-1)).squeeze(-1)
    r = norm1(r, dim=-1, eps=eps)

    # 3) average prompt->patch attention a_bar(x): [B, N]
    a_bar = attn_pv.mean(dim=1)

    # 4) squared L2 norm over patches, per sample: [B]
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
    Aggregate L_avg over multiple layers.
        `attn_pv_dict[layer]` and `attn_vs_dict[layer]` should give the
    [B, T, N] and [B, N, M] tensors for that layer.
    """
    losses = []
    for l in layers:
        losses.append(l_avg(attn_pv_dict[l], attn_vs_dict[l], eps=eps, reduction="none"))
    # stack: [num_layers, B] -> [B]
    stacked = torch.stack(losses, dim=0).mean(dim=0)
    if reduction == "none":
        return stacked
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


class SoftmaxLoss(nn.Module):
    """
    标准的 Softmax + CrossEntropy 分类损失。
    """
    def __init__(self, cfg=None):
        # 这里 cfg 暂时没用，但保留接口方便以后扩展（比如从 cfg 里读 class_weight 等）
        super().__init__()

    def is_single(self):
        """返回 True，表示单一分支损失。"""
        return True

    def is_local(self):
        """
        返回 False，表示不是“局部 patch-level 损失”，
        而是对整张图片 / 整个样本的分类损失（global-level）。
        """
        return False

    def loss(self, logits, targets, per_cls_weights, kwargs=None):
        """
        使用 F.cross_entropy 计算多类交叉熵损失。

        参数:
            logits: [B, C]，模型输出的类别分数（未过 softmax）
            targets: [B]，整型类别 id
            per_cls_weights: 长度为 C 的 per-class 权重
            kwargs: 目前没用（保留扩展接口）

        返回:
            标量损失值（对 batch 做平均）
        """
        weight = torch.tensor(per_cls_weights, device=logits.device)
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        return torch.sum(loss) / targets.shape[0]

    def forward(self, pred_logits, targets, per_cls_weights, kwargs=None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)



class SoftmaxWithPromptAlignLoss(nn.Module):
    """
    基础分类交叉熵 + 提示-视觉注意力对齐正则（可选）。

    - 主干仍是标准 softmax 交叉熵。
    - 若提供 attn_pv / attn_vs（单层或多层）且权重 alpha>0，则额外叠加 L_avg 对齐项。
    - 兼容三种 logits 形式：Tensor、(logits, aux) 元组或包含 "logits"/注意力的字典。
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.alpha = getattr(cfg.SOLVER, "LOSS_ALPHA", 0.0) if cfg is not None else 0.0
        # 复用已有的 SoftmaxLoss 实现，保证接口一致
        self.cls_loss = SoftmaxLoss(cfg)

    def is_single(self):
        return True

    def is_local(self):
        return False

    def _extract_logits_and_aux(self, pred_logits: Any, kwargs: Optional[Dict[str, Any]]):
        """兼容多种 logits/aux 表达形式，返回 (logits, aux_dict_or_None)。"""

        aux = None
        logits = pred_logits

        # (logits, aux) 或 (logits, extra)
        if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
            logits = pred_logits[0]
            if len(pred_logits) > 1 and isinstance(pred_logits[1], Dict):
                aux = pred_logits[1]

        # {"logits": tensor, ...}
        if isinstance(pred_logits, Dict) and "logits" in pred_logits:
            logits = pred_logits["logits"]
            aux = pred_logits

        # 备用：从 kwargs 里取 aux
        if aux is None and kwargs is not None and isinstance(kwargs, Dict):
            aux = kwargs.get("aux")

        return logits, aux

    def _compute_align_loss(self, aux: Optional[Dict[str, Any]]):
        """从 aux 中解析注意力张量，计算 L_avg 对齐项。"""

        if aux is None or self.alpha <= 0:
            return None

        attn_pv = aux.get("attn_pv") if isinstance(aux, Dict) else None
        attn_vs = aux.get("attn_vs") if isinstance(aux, Dict) else None

        if attn_pv is None or attn_vs is None:
            return None

        if isinstance(attn_pv, Dict) and isinstance(attn_vs, Dict):
            layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()))
            if not layers:
                return None
            return l_avg_multi(attn_pv, attn_vs, layers=layers, reduction="mean")

        if torch.is_tensor(attn_pv) and torch.is_tensor(attn_vs):
            return l_avg(attn_pv, attn_vs, reduction="mean")

        return None

    def loss(self, logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        logits, aux = self._extract_logits_and_aux(logits, kwargs)

        base = self.cls_loss.loss(logits, targets, per_cls_weights, kwargs)
        align = self._compute_align_loss(aux)

        if align is None:
            return base
        return base + self.alpha * align

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


# 一个字典，用来把字符串名字映射到具体损失类
LOSS = {
    "softmax": SoftmaxLoss,
    "softmax_prompt_align": SoftmaxWithPromptAlignLoss,

}


class RSimilarityClassifier(nn.Module):
    """使用共享概念基 R 生成的语义原型与 CLS 对齐的分类头。

    视觉侧：CLS -> M_v；语义侧：类属性 -> SharedConceptAligner.encode_semantics_only -> R 空间原型 -> M_s。
    logits 基于余弦相似度（可选）计算，并带可学习温度。
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
    ) -> None:
        super().__init__()
        if proj_dim is None or proj_dim <= 0:
            proj_dim = hidden_size

        self.semantic_concept = semantic_concept
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]

        self.visual_proj_enabled = visual_proj_enable
        out_dim = proj_dim if self.visual_proj_enabled else hidden_size
        self.visual_proj = (
            nn.Linear(hidden_size, out_dim) if self.visual_proj_enabled else None
        )
        self.semantic_proj = nn.Linear(hidden_size, out_dim)

        self.use_cosine = use_cosine
        if use_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(torch.tensor(logit_scale_init, dtype=torch.float32))
            )
        else:
            self.logit_scale = None

    def _class_prototypes(self) -> torch.Tensor:
        """通过共享概念基获取 R 空间语义原型。"""

        attr = self.class_attr.to(self.semantic_concept.concept_slots.device)
        return self.semantic_concept.encode_semantics_only(attr)

    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        prototypes = self._class_prototypes()  # [C, D]

        visual = self.visual_proj(cls_feat) if self.visual_proj else cls_feat  # [B, d]
        semantic = self.semantic_proj(prototypes)  # [C, d]

        if self.use_cosine:
            visual = F.normalize(visual, dim=-1)
            semantic = F.normalize(semantic, dim=-1)
            logits = visual @ semantic.t()
            logits = logits * self.logit_scale.exp()
        else:
            logits = visual @ semantic.t()

        return logits



def build_loss(cfg):
    """
    根据配置构建损失函数实例。

    cfg.SOLVER.LOSS 必须是上面 LOSS 字典中的一个键，比如 "softmax"
    或包含提示对齐正则的 "softmax_prompt_align"。
    使用方法（伪代码）：
        loss_fn = build_loss(cfg)
        loss = loss_fn(logits, targets, per_cls_weights)

    返回:
        nn.Module 子类的实例（如 SoftmaxLoss），或抛出错误（loss 名称不在字典中）。
    """
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)
