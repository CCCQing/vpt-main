#!/usr/bin/env python3
"""Semantic graph losses for prompt distribution centers.

本模块只依赖 prompt distributor 缓存的 mu 和全局类别标签，不要求
distributor.forward() 接收 label。

核心目标:
1. 用类别属性置信度构造 Acc，即类别-类别属性相似图；
2. 用属性名文本 embedding 构造 Acssc，即属性语义增强后的类别图；
3. 从 Acc / Acssc / fuse 图中取当前样本对应行，形成全类语义 target；
4. 约束 prompt distribution center mu 与这个语义图关系一致。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_normalize(x: torch.Tensor) -> torch.Tensor:
    """行向量 L2 归一化，用于余弦相似度图构造。"""
    return F.normalize(x, p=2, dim=-1)


def _normalize_prob(x: torch.Tensor, eps: float) -> torch.Tensor:
    """把非负张量规范成概率分布，并避免除 0。"""
    x = x.clamp_min(float(eps))
    return x / x.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def _kl_target_pred(target: torch.Tensor, pred: torch.Tensor, eps: float) -> torch.Tensor:
    """KL(target || pred)，输入会先 clamp 并按行归一化。"""
    target = _normalize_prob(target, eps)
    pred = _normalize_prob(pred, eps)
    return (target * (target.log() - pred.log())).sum(dim=-1).mean()


def _load_tensor_file(path: str, expected_name: str) -> torch.Tensor:
    """
    严格读取语义图所需张量。

    支持:
    - .npy
    - .pt/.pth tensor
    - .pt/.pth dict，其中常见键包括 embeddings / class_attributes / attributes。
    """
    if not path:
        raise ValueError(f"MODEL.SEMANTIC_GRAPH.{expected_name} must be set.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MODEL.SEMANTIC_GRAPH.{expected_name} not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        obj = np.load(path)
    else:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("class_attributes", "attributes", "att", "embeddings", "tensor"):
            if key in obj:
                obj = obj[key]
                break
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if not torch.is_tensor(obj):
        raise TypeError(f"{expected_name} must load as tensor/ndarray/dict tensor, got {type(obj)} from {path}")
    return obj.float()


class SemanticGraphLossComputer(nn.Module):
    """
    构造语义图并计算指定 graph loss。

    该类作为 nn.Module 是因为 SEMANTIC_BANK_SOURCE="learned_proj" 时会产生
    一个可训练的 312->768 投影；trainer 已把 cls_criterion 加入 optimizer。
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        graph_cfg = cfg.MODEL.SEMANTIC_GRAPH
        self.num_classes = int(graph_cfg.NUM_CLASSES)
        self.attr_dim = int(graph_cfg.ATTR_DIM)
        self.text_dim = int(graph_cfg.TEXT_DIM)
        self.loss_type = str(graph_cfg.LOSS_TYPE).lower()
        self.bank_source = str(graph_cfg.SEMANTIC_BANK_SOURCE).lower()
        if self.bank_source not in {"asem", "learned_proj"}:
            raise ValueError("MODEL.SEMANTIC_GRAPH.SEMANTIC_BANK_SOURCE must be asem or learned_proj.")
        if self.bank_source == "learned_proj":
            # learned_proj 只用于构造全类 semantic bank S=f_s(A_conf)，
            # 不改变数据集属性本身，也不参与 distributor.forward。
            self.semantic_proj = nn.Linear(self.attr_dim, self.text_dim)
        self._debug_logged = False

    def _load_or_prepare_class_attributes(
        self,
        class_attributes: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        读取类别属性矩阵 A_conf。

        优先使用 trainer 从 dataset.class_attributes 传入的 [C,A]；
        若数据集没有提供，再读取 MODEL.SEMANTIC_GRAPH.CLASS_ATTR_PATH。
        """
        if class_attributes is None:
            class_attributes = _load_tensor_file(
                str(self.cfg.MODEL.SEMANTIC_GRAPH.CLASS_ATTR_PATH),
                "CLASS_ATTR_PATH",
            )
        if not torch.is_tensor(class_attributes):
            class_attributes = torch.as_tensor(class_attributes)
        class_attributes = class_attributes.to(device=device, dtype=dtype)
        if tuple(class_attributes.shape) != (self.num_classes, self.attr_dim):
            raise RuntimeError(
                "Semantic graph class attributes must be [{},{}], got {}.".format(
                    self.num_classes,
                    self.attr_dim,
                    tuple(class_attributes.shape),
                )
            )
        return class_attributes

    def _load_attr_embeddings(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """读取属性名文本 embedding E_attr，期望形状为 [ATTR_DIM, TEXT_DIM]。"""
        embeddings = _load_tensor_file(
            str(self.cfg.MODEL.SEMANTIC_GRAPH.ATTR_NAME_EMBED_PATH),
            "ATTR_NAME_EMBED_PATH",
        )
        embeddings = embeddings.to(device=device, dtype=dtype)
        if tuple(embeddings.shape) != (self.attr_dim, self.text_dim):
            raise RuntimeError(
                "Semantic graph attr name embeddings must be [{},{}], got {}.".format(
                    self.attr_dim,
                    self.text_dim,
                    tuple(embeddings.shape),
                )
            )
        return embeddings

    def _build_graphs(
        self,
        class_attributes: torch.Tensor,
        attr_embeddings: torch.Tensor,
    ):
        """
        构造三类类别图:
        - Acc   = norm(A_conf) @ norm(A_conf)^T
        - Acssc = norm(A_conf @ E_attr) @ norm(A_conf @ E_attr)^T
        - G     = acc / acssc / rho 加权融合

        G 默认 detach，因为它是语义先验，不是要被训练的参数。
        """
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        acc = _row_normalize(class_attributes).matmul(_row_normalize(class_attributes).t())
        asem = class_attributes.matmul(attr_embeddings)
        acssc = _row_normalize(asem).matmul(_row_normalize(asem).t())

        graph_source = str(graph_cfg.GRAPH_SOURCE).lower()
        if graph_source == "acc":
            graph = acc
        elif graph_source == "acssc":
            graph = acssc
        elif graph_source == "fuse":
            rho = float(graph_cfg.RHO)
            graph = rho * acc + (1.0 - rho) * acssc
        else:
            raise ValueError("MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE must be acc / acssc / fuse.")
        return acc.detach(), acssc.detach(), graph.detach(), asem.detach()

    def _validate_targets(self, targets_global: torch.Tensor, device: torch.device) -> torch.Tensor:
        """检查语义图使用的是全局类别 id，而不是 local-output remap 后的局部 id。"""
        if not torch.is_tensor(targets_global):
            raise RuntimeError("Semantic graph loss requires tensor targets_global.")
        targets_global = targets_global.to(device=device, dtype=torch.long)
        if targets_global.dim() != 1:
            raise RuntimeError(f"targets_global must be [B], got {tuple(targets_global.shape)}.")
        if targets_global.numel() == 0:
            raise RuntimeError("targets_global is empty.")
        if int(targets_global.min().item()) < 0 or int(targets_global.max().item()) >= self.num_classes:
            raise RuntimeError(
                "Semantic graph loss requires global targets in [0,{}], got min={} max={}.".format(
                    self.num_classes - 1,
                    int(targets_global.min().item()),
                    int(targets_global.max().item()),
                )
            )
        return targets_global

    def _build_target(self, graph: torch.Tensor, targets_global: torch.Tensor) -> torch.Tensor:
        """
        从语义图中取每个样本对应的目标分布 T_y。

        流程:
        1. 取 G[y] 得到该类到所有类的相似度；
        2. 只保留 TOPK 个候选类，其余置为 -inf；
        3. 用 TAU_ACC 做 softmax；
        4. 与 one-hot(y) 按 TARGET_MIX_ALPHA 混合。
        """
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        eps = float(graph_cfg.OT_DELTA)
        rows = graph.index_select(0, targets_global)
        topk = int(graph_cfg.TOPK)
        if topk <= 0 or topk > self.num_classes:
            raise ValueError("MODEL.SEMANTIC_GRAPH.TOPK must be in [1, NUM_CLASSES].")
        values, indices = torch.topk(rows, k=topk, dim=-1)
        masked = torch.full_like(rows, float("-inf"))
        masked.scatter_(1, indices, values)
        target_sem = F.softmax(masked / float(graph_cfg.TAU_ACC), dim=-1)

        alpha = float(graph_cfg.TARGET_MIX_ALPHA)
        onehot = F.one_hot(targets_global, num_classes=self.num_classes).to(dtype=target_sem.dtype)
        target = (1.0 - alpha) * onehot + alpha * target_sem
        return _normalize_prob(target, eps).detach()

    def _semantic_bank(self, class_attributes: torch.Tensor, asem: torch.Tensor) -> torch.Tensor:
        """返回 acc_hidden / OT 节点项使用的全类语义 bank S。"""
        if self.bank_source == "asem":
            return asem
        return self.semantic_proj(class_attributes)

    def _sinkhorn(self, cost: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Entropic Sinkhorn，用于把 batch prompt 节点软匹配到 200 个语义类节点。

        当前第一版只实现 batch_semantic_mean:
        - 源边界 a 是 batch 内均匀分布；
        - 目标边界 b 是当前 batch 的语义 target 均值。
        """
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        if str(graph_cfg.OT_BALANCED_MODE).lower() != "batch_semantic_mean":
            raise ValueError("MODEL.SEMANTIC_GRAPH.OT_BALANCED_MODE currently supports only batch_semantic_mean.")

        eps = float(graph_cfg.OT_EPS)
        delta = float(graph_cfg.OT_DELTA)
        batch_size = int(cost.shape[0])
        a = torch.full((batch_size,), 1.0 / float(batch_size), device=cost.device, dtype=cost.dtype)
        b = _normalize_prob(target.mean(dim=0), delta)
        kernel = torch.exp(-cost / eps).clamp_min(delta)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(int(graph_cfg.OT_ITERS)):
            u = a / kernel.matmul(v).clamp_min(delta)
            v = b / kernel.t().matmul(u).clamp_min(delta)
        plan = u[:, None] * kernel * v[None, :]
        return plan / plan.sum().clamp_min(delta)

    def _node_cost(self, mu: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """节点 cost: 1 - cosine(mu_i, semantic_bank_c)。"""
        mu_n = _row_normalize(mu)
        bank_n = _row_normalize(bank)
        return 1.0 - mu_n.matmul(bank_n.t())

    def _gw_term(self, mu: torch.Tensor, graph: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """
        Gromov-Wasserstein 结构项。

        比较的是:
        - prompt batch 内关系 R_p = cosine(mu_i, mu_j)
        - 全类语义图关系 G[c,d]

        使用矩阵化公式，避免四重 Python 循环。
        """
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        delta = float(graph_cfg.OT_DELTA)
        rp = _row_normalize(mu).matmul(_row_normalize(mu).t())
        a = _normalize_prob(plan.sum(dim=1), delta)
        b = _normalize_prob(plan.sum(dim=0), delta)
        term_prompt = (rp.pow(2) * torch.outer(a, a)).sum()
        term_graph = (graph.pow(2) * torch.outer(b, b)).sum()
        term_cross = 2.0 * (rp * plan.matmul(graph).matmul(plan.t())).sum()
        return term_prompt + term_graph - term_cross

    def _debug_once(
        self,
        class_attributes: torch.Tensor,
        attr_embeddings: torch.Tensor,
        asem: torch.Tensor,
        acc: torch.Tensor,
        acssc: torch.Tensor,
        graph: torch.Tensor,
        targets_global: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor,
        extra: str,
    ) -> None:
        """MODEL.SEMANTIC_GRAPH.DEBUG=True 时打印一次关键张量形状。"""
        if not bool(self.cfg.MODEL.SEMANTIC_GRAPH.DEBUG) or self._debug_logged:
            return
        print(
            "[SEM-GRAPH-DEBUG] A_conf={} E_attr={} A_sem={} Acc={} Acssc={} G={} loss_type={} "
            "targets_minmax=({}, {}) T={} loss={} {}".format(
                tuple(class_attributes.shape),
                tuple(attr_embeddings.shape),
                tuple(asem.shape),
                tuple(acc.shape),
                tuple(acssc.shape),
                tuple(graph.shape),
                self.loss_type,
                int(targets_global.min().item()),
                int(targets_global.max().item()),
                tuple(target.shape),
                float(loss.detach().item()),
                extra,
            )
        )
        self._debug_logged = True

    def forward(
        self,
        mu: torch.Tensor,
        targets_global: torch.Tensor,
        class_attributes: Optional[torch.Tensor] = None,
        attr_name_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算当前配置选择的 semantic graph loss。

        输入:
        - mu: prompt distribution center，[B,768]
        - targets_global: 全局类别 id，[B]
        - class_attributes: 可选 [200,312]，优先来自 dataset.class_attributes
        - attr_name_embeddings: 可选 [312,768]，默认从配置路径读取
        """
        if self.loss_type == "none":
            return mu.sum() * 0.0
        if mu.dim() != 2 or mu.shape[1] != self.text_dim:
            raise RuntimeError(f"Semantic graph loss expects mu [B,{self.text_dim}], got {tuple(mu.shape)}.")

        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        eps = float(graph_cfg.OT_DELTA)
        targets_global = self._validate_targets(targets_global, mu.device)
        class_attributes = self._load_or_prepare_class_attributes(class_attributes, mu.device, mu.dtype)
        if attr_name_embeddings is None:
            attr_name_embeddings = self._load_attr_embeddings(mu.device, mu.dtype)
        else:
            attr_name_embeddings = attr_name_embeddings.to(device=mu.device, dtype=mu.dtype)
        acc, acssc, graph, asem = self._build_graphs(class_attributes, attr_name_embeddings)
        target = self._build_target(graph, targets_global)

        bank = self._semantic_bank(class_attributes, asem)
        extra = ""
        if self.loss_type == "acc_hidden":
            # 方案 A：把 mu 直接和 200 个语义原型比较，监督其全类分布。
            logits = _row_normalize(mu).matmul(_row_normalize(bank).t()) / float(graph_cfg.TAU_PROMPT)
            pred = F.softmax(logits, dim=-1)
            loss = _kl_target_pred(target, pred, eps)
        elif self.loss_type == "rel_kl":
            # 方案 B：只约束 batch 内 prompt 关系图与语义关系图一致。
            rp = _row_normalize(mu).matmul(_row_normalize(mu).t())
            rs = graph.index_select(0, targets_global).index_select(1, targets_global)
            pred = F.softmax(rp / float(graph_cfg.TAU_PROMPT), dim=-1)
            sem = F.softmax(rs / float(graph_cfg.TAU_SEM), dim=-1)
            loss = _kl_target_pred(sem.detach(), pred, eps)
        elif self.loss_type == "rel_all":
            # 方案 C：先经 batch 内 prompt 关系传播，再对齐到全类语义 target。
            rp = _row_normalize(mu).matmul(_row_normalize(mu).t())
            pred_batch = F.softmax(rp / float(graph_cfg.TAU_PROMPT), dim=-1)
            pred_all = _normalize_prob(pred_batch.matmul(target), eps)
            loss = _kl_target_pred(target, pred_all, eps)
        elif self.loss_type == "ot":
            # 方案 D：用 OT 学习 batch prompt 到全类语义节点的软匹配。
            cost = self._node_cost(mu, bank)
            plan = self._sinkhorn(cost, target)
            if bool(graph_cfg.OT_DETACH_PLAN):
                plan = plan.detach()
            loss = (plan * cost).sum()
            extra = "M={} Pi={}".format(tuple(cost.shape), tuple(plan.shape))
        elif self.loss_type == "gw":
            # 方案 E：在 OT plan 下对齐 prompt 图结构与语义类图结构。
            cost = self._node_cost(mu, bank)
            plan = self._sinkhorn(cost, target)
            if bool(graph_cfg.OT_DETACH_PLAN):
                plan = plan.detach()
            loss = self._gw_term(mu, graph, plan)
            extra = "M={} Pi={} Rp={}".format(tuple(cost.shape), tuple(plan.shape), (mu.shape[0], mu.shape[0]))
        elif self.loss_type == "fgw":
            # 方案 F：节点 cost + 语义先验 cost + GW 结构项的融合版本。
            cost = self._node_cost(mu, bank)
            prior_cost = cost - float(graph_cfg.OT_PRIOR_ETA) * torch.log(target + eps)
            plan = self._sinkhorn(prior_cost, target)
            if bool(graph_cfg.OT_DETACH_PLAN):
                plan = plan.detach()
            node_loss = (plan * prior_cost).sum()
            gw_loss = self._gw_term(mu, graph, plan)
            alpha = float(graph_cfg.OT_ALPHA)
            loss = (1.0 - alpha) * node_loss + alpha * gw_loss
            extra = "M={} Pi={} node={} gw={}".format(
                tuple(prior_cost.shape),
                tuple(plan.shape),
                float(node_loss.detach().item()),
                float(gw_loss.detach().item()),
            )
        else:
            raise ValueError(
                "MODEL.SEMANTIC_GRAPH.LOSS_TYPE must be none / acc_hidden / rel_kl / rel_all / ot / gw / fgw."
            )

        self._debug_once(class_attributes, attr_name_embeddings, asem, acc, acssc, graph, targets_global, target, loss, extra)
        return loss
