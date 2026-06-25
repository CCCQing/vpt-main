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

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .graph_prob_prior_monitors import (
    aggregate_moment_monitor,
    dual_metric_distribution_monitor,
    dual_sample_beta_monitor,
    factorized_health_monitor,
    false_high_pair_monitor,
    graph_prior_geometry_monitor,
    gzsl_prior_risk_monitor,
    graph_health_monitor,
    graph_neighbor_monitor,
    latent_matching_monitor,
    mmd_monitor,
    posterior_prior_alignment_monitor,
    prior_health_monitor,
    relation_monitor,
    residual_anchor_prior_monitor,
    semantic_target_monitor,
    seen_unseen_prior_monitor,
    true_class_kl_monitor,
)

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


class SemanticGraphBuilder:
    """
    语义图共享构造器。

    这个类只负责把 dataloader/trainer 传入的语义资源转成三个核心对象：
    - bank:  A_conf @ E_attr，形状 [C, D]，每个类别的属性文本语义原型；
    - graph: Acc / Acssc / fuse 得到的全类关系图，形状 [C, C]；
    - target: 从 graph[y] 取 top-k 并混合 one-hot 后得到的监督分布，形状 [B, C]。

    SemanticGraphLossComputer 和 GraphProbPriorLossComputer 都复用这里，
    避免同一套 A/G/T 构造逻辑在两个 auxiliary loss 中各写一遍。
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        graph_cfg = cfg.MODEL.SEMANTIC_GRAPH
        # C=全局类别数，attr_dim=属性维度，text_dim=属性名 embedding / prompt latent 维度。
        self.num_classes = int(graph_cfg.NUM_CLASSES)
        self.attr_dim = int(graph_cfg.ATTR_DIM)
        self.text_dim = int(graph_cfg.TEXT_DIM)
        self._external_graph_cache = None
        self._external_graph_cache_id = None

    def prepare_class_attributes(
        self,
        class_attributes: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """校验并搬运全类属性矩阵 A_conf；该矩阵必须由 trainer/dataset 显式传入。"""
        if class_attributes is None:
            raise RuntimeError("Semantic graph losses require dataset.class_attributes from dataloader.")
        if not torch.is_tensor(class_attributes):
            class_attributes = torch.as_tensor(class_attributes)
        # class_attributes 通常来自 XLSA/CUB 的 att_splits.mat::att，形状应为 [C, attr_dim]。
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

    def prepare_attr_name_embeddings(
        self,
        attr_name_embeddings: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """校验并搬运属性名文本 embedding E_attr；loss 内部不读路径，不做隐式兜底。"""
        if attr_name_embeddings is None:
            raise RuntimeError("Semantic graph losses require attr_name_embeddings from trainer or dataset.")
        attr_name_embeddings = attr_name_embeddings.to(device=device, dtype=dtype)
        if tuple(attr_name_embeddings.shape) != (self.attr_dim, self.text_dim):
            raise RuntimeError(
                "Semantic graph attr name embeddings must be [{},{}], got {}.".format(
                    self.attr_dim,
                    self.text_dim,
                    tuple(attr_name_embeddings.shape),
                )
            )
        return attr_name_embeddings

    def _load_external_graph(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        raw_path = str(graph_cfg.EXTERNAL_GRAPH_PATH).strip()
        key = str(graph_cfg.EXTERNAL_GRAPH_KEY).strip()
        if not raw_path:
            raise ValueError("MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_PATH must be set when GRAPH_SOURCE=external.")
        if not key:
            raise ValueError("MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_KEY must be set when GRAPH_SOURCE=external.")

        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"External semantic graph file not found: {path}")

        cache_id = (str(path.resolve()), key)
        if self._external_graph_cache is None or self._external_graph_cache_id != cache_id:
            suffix = path.suffix.lower()
            if suffix == ".npz":
                payload = np.load(str(path), allow_pickle=False)
                if key not in payload.files:
                    raise KeyError(f"External semantic graph key '{key}' not found in {path}; available={payload.files}.")
                array = payload[key]
            elif suffix == ".npy":
                array = np.load(str(path), allow_pickle=False)
            elif suffix in {".pt", ".pth"}:
                payload = torch.load(str(path), map_location="cpu")
                if isinstance(payload, dict):
                    if key not in payload:
                        raise KeyError(f"External semantic graph key '{key}' not found in {path}; available={list(payload.keys())}.")
                    payload = payload[key]
                array = payload.detach().cpu().numpy() if torch.is_tensor(payload) else np.asarray(payload)
            else:
                raise ValueError("External semantic graph file must be .npz, .npy, .pt, or .pth.")

            if getattr(array, "ndim", None) != 2:
                raise RuntimeError(f"External semantic graph must be a 2-D matrix, got shape={getattr(array, 'shape', None)}.")
            if tuple(array.shape) != (self.num_classes, self.num_classes):
                raise RuntimeError(
                    "External semantic graph must be [{},{}], got {} from key '{}'.".format(
                        self.num_classes,
                        self.num_classes,
                        tuple(array.shape),
                        key,
                    )
                )

            graph = torch.as_tensor(array, dtype=torch.float32, device="cpu")
            if not bool(torch.isfinite(graph).all().item()):
                raise RuntimeError(f"External semantic graph contains NaN or Inf: {path} key={key}")
            if bool(graph_cfg.EXTERNAL_GRAPH_SYMMETRIZE):
                graph = 0.5 * (graph + graph.t())
            if bool(graph_cfg.EXTERNAL_GRAPH_CLAMP):
                graph = graph.clamp(0.0, 1.0)
            diag_value = float(graph_cfg.EXTERNAL_GRAPH_DIAG_VALUE)
            if diag_value >= 0.0:
                graph.fill_diagonal_(diag_value)
            self._external_graph_cache = graph.contiguous()
            self._external_graph_cache_id = cache_id

        return self._external_graph_cache.to(device=device, dtype=dtype)

    def build_graphs(
        self,
        class_attributes: torch.Tensor,
        attr_embeddings: torch.Tensor,
    ):
        """
        构造语义 bank 和全类语义图。

        - Acc   = norm(A_conf) @ norm(A_conf)^T，只看属性置信度相似；
        - A_sem = A_conf @ E_attr，用属性置信度加权属性名文本 embedding；
        - Acssc = norm(A_sem) @ norm(A_sem)^T，看文本语义增强后的类别相似；
        - graph = acc / acssc / rho 融合图。
        """
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        # Acc 是纯属性置信度图：先行归一化，再做类别间余弦式相似度。
        acc = _row_normalize(class_attributes).matmul(_row_normalize(class_attributes).t())

        # bank / A_sem 是每个类别的文本语义原型：
        # 用该类的属性置信度加权所有属性名 embedding，得到 [C, text_dim]。
        bank = class_attributes.matmul(attr_embeddings)

        # Acssc 是 bank 上的类别关系图，仍然使用行归一化后的相似度。
        acssc = _row_normalize(bank).matmul(_row_normalize(bank).t())

        # GRAPH_SOURCE 决定最终使用哪张图：
        #   acc   -> 只用属性置信度图；
        #   acssc -> 只用属性文本语义图；
        #   fuse  -> 按 rho 融合二者。
        graph_source = str(graph_cfg.GRAPH_SOURCE).lower()
        if graph_source == "acc":
            graph = acc
        elif graph_source == "acssc":
            graph = acssc
        elif graph_source == "fuse":
            rho = float(graph_cfg.RHO)
            if rho < 0.0 or rho > 1.0:
                raise ValueError("MODEL.SEMANTIC_GRAPH.RHO must be in [0, 1].")
            graph = rho * acc + (1.0 - rho) * acssc
        elif graph_source == "external":
            graph = self._load_external_graph(class_attributes.device, class_attributes.dtype)
        else:
            raise ValueError("MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE must be acc / acssc / fuse / external.")
        return acc.detach(), acssc.detach(), graph.detach(), bank.detach()

    def validate_targets(self, targets_global: torch.Tensor, device: torch.device) -> torch.Tensor:
        """校验 semantic graph 使用的是全局类别 id，而不是 local-output remap 后的局部 id。"""
        if not torch.is_tensor(targets_global):
            raise RuntimeError("Semantic graph losses require tensor targets_global.")
        targets_global = targets_global.to(device=device, dtype=torch.long)
        if targets_global.dim() != 1:
            raise RuntimeError(f"targets_global must be [B], got {tuple(targets_global.shape)}.")
        if targets_global.numel() == 0:
            raise RuntimeError("targets_global is empty.")
        if int(targets_global.min().item()) < 0 or int(targets_global.max().item()) >= self.num_classes:
            raise RuntimeError(
                "Semantic graph losses require global targets in [0,{}], got min={} max={}.".format(
                    self.num_classes - 1,
                    int(targets_global.min().item()),
                    int(targets_global.max().item()),
                )
            )
        return targets_global

    def build_target(self, graph: torch.Tensor, targets_global: torch.Tensor) -> torch.Tensor:
        """
        从语义图中为 batch 构造全类 target T。

        每个样本取 graph[y]，仅保留 top-k 类别做 softmax，再与 one-hot(y) 混合。
        该 target 既服务旧 semantic graph loss，也服务新的 GraphProbPrior。
        """
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        eps = float(graph_cfg.OT_DELTA)

        # rows 是当前 batch 每个真实类别 y 在全类语义图里的那一行，形状 [B, C]。
        rows = graph.index_select(0, targets_global)
        topk = int(graph_cfg.TOPK)
        if topk <= 0 or topk > self.num_classes:
            raise ValueError("MODEL.SEMANTIC_GRAPH.TOPK must be in [1, NUM_CLASSES].")

        # 只让每个样本的 top-k 语义邻居进入 soft target，其余类别置为 -inf。
        values, indices = torch.topk(rows, k=topk, dim=-1)
        masked = torch.full_like(rows, float("-inf"))
        masked.scatter_(1, indices, values)

        # TAU_ACC 控制语义邻居分布的尖锐程度；这里只构造监督 target，不是可学习预测。
        target_sem = F.softmax(masked / float(graph_cfg.TAU_ACC), dim=-1)

        alpha = float(graph_cfg.TARGET_MIX_ALPHA)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA must be in [0, 1].")

        # 最终 target = one-hot 真类 与 semantic top-k soft target 的混合。
        # alpha=0 时退化为纯 one-hot；alpha=1 时完全使用语义邻居分布。
        onehot = F.one_hot(targets_global, num_classes=self.num_classes).to(dtype=target_sem.dtype)
        target = (1.0 - alpha) * onehot + alpha * target_sem
        return _normalize_prob(target, eps).detach()

    def prepare(
        self,
        targets_global: torch.Tensor,
        class_attributes: Optional[torch.Tensor],
        attr_name_embeddings: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """一次性准备 GraphProbPrior / SemanticGraphLoss 所需的所有语义图张量。"""
        # prepare 是 loss 侧唯一需要调用的入口：
        # 先校验 batch targets 和语义资源，再统一返回 A/G/T 相关张量。
        targets_global = self.validate_targets(targets_global, device)
        class_attributes = self.prepare_class_attributes(class_attributes, device, dtype)
        attr_name_embeddings = self.prepare_attr_name_embeddings(attr_name_embeddings, device, dtype)
        acc, acssc, graph, bank = self.build_graphs(class_attributes, attr_name_embeddings)
        target = self.build_target(graph, targets_global)
        return {
            "class_attributes": class_attributes,
            "attr_name_embeddings": attr_name_embeddings,
            "acc": acc,
            "acssc": acssc,
            "graph": graph,
            "bank": bank,
            "target": target,
            "targets_global": targets_global,
        }


class SemanticGraphLossComputer(torch.nn.Module):
    """
    构造语义图并计算指定 graph loss。

    该类不再包含可训练语义投影。语义 bank 固定为:
        A_sem = A_conf @ E_attr
    即用类别属性置信度对属性名文本 embedding 做加权求和。
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        graph_cfg = cfg.MODEL.SEMANTIC_GRAPH
        self.num_classes = int(graph_cfg.NUM_CLASSES)
        self.attr_dim = int(graph_cfg.ATTR_DIM)
        self.text_dim = int(graph_cfg.TEXT_DIM)
        self.graph_builder = SemanticGraphBuilder(cfg)
        self.loss_type = str(graph_cfg.LOSS_TYPE).lower()
        prompt_tau = float(graph_cfg.TAU_PROMPT)
        if prompt_tau <= 0.0:
            raise ValueError("MODEL.SEMANTIC_GRAPH.TAU_PROMPT must be positive.")
        prompt_logit_scale = torch.tensor(math.log(1.0 / prompt_tau))
        if bool(graph_cfg.PROMPT_SCALE_LEARNABLE):
            self.prompt_logit_scale = torch.nn.Parameter(prompt_logit_scale)
        else:
            self.register_buffer("prompt_logit_scale", prompt_logit_scale)
        self._debug_logged = False

    def _sinkhorn(self, cost: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Entropic Sinkhorn，用于把 batch prompt 节点软匹配到 200 个语义类节点。

        当前第一版只实现 batch_semantic_mean:
        - 源边界 a 是 batch 内均匀分布；
        - 目标边界 b 是当前 batch 的语义 target 均值。
        """
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        balanced_mode = str(graph_cfg.OT_BALANCED_MODE).lower()
        if balanced_mode != "batch_semantic_mean":
            raise ValueError("MODEL.SEMANTIC_GRAPH.OT_BALANCED_MODE currently supports only batch_semantic_mean.")
        eps = float(graph_cfg.OT_EPS)
        delta = float(graph_cfg.OT_DELTA)
        if eps <= 0.0:
            raise ValueError("MODEL.SEMANTIC_GRAPH.OT_EPS must be positive.")
        iters = int(graph_cfg.OT_ITERS)
        if iters <= 0:
            raise ValueError("MODEL.SEMANTIC_GRAPH.OT_ITERS must be positive.")
        batch_size = int(cost.shape[0])
        a = torch.full((batch_size,), 1.0 / float(batch_size), device=cost.device, dtype=cost.dtype)
        b = _normalize_prob(target.mean(dim=0), delta)
        kernel = torch.exp(-cost / eps).clamp_min(delta)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(iters):
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
        delta = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)
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
        - targets_global: 全局类别 id，[B],当前 batch 里每个样本属于哪个类
        - class_attributes: 可选 [200,312]，优先来自 dataset.class_attributes
        - attr_name_embeddings: [312,768]，由 trainer 或 dataset 显式传入
        """
        if self.loss_type == "none":
            return mu.sum() * 0.0
        if mu.dim() != 2 or mu.shape[1] != self.text_dim:
            raise RuntimeError(f"Semantic graph loss expects mu [B,{self.text_dim}], got {tuple(mu.shape)}.")

        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        eps = float(graph_cfg.OT_DELTA)
        # 输入校验与标准化
        graph_inputs = self.graph_builder.prepare(
            targets_global=targets_global,
            class_attributes=class_attributes,
            attr_name_embeddings=attr_name_embeddings,
            device=mu.device,
            dtype=mu.dtype,
        )
        targets_global = graph_inputs["targets_global"]
        class_attributes = graph_inputs["class_attributes"]
        attr_name_embeddings = graph_inputs["attr_name_embeddings"]
        acc = graph_inputs["acc"]
        acssc = graph_inputs["acssc"]
        graph = graph_inputs["graph"]
        asem = graph_inputs["bank"]
        target = graph_inputs["target"]

        # 固定语义 bank：每个类别的属性置信度加权属性名文本 embedding。
        # shape: [200,312] @ [312,768] -> [200,768]
        bank = asem
        prompt_scale = self.prompt_logit_scale.exp()
        extra = ""
        if self.loss_type == "acc_hidden":
            # 方案 A：把 mu 直接和 200 个语义原型比较，监督其全类分布。
            logits = _row_normalize(mu).matmul(_row_normalize(bank).t()) * prompt_scale
            pred = F.softmax(logits, dim=-1)
            loss = _kl_target_pred(target, pred, eps)
        elif self.loss_type == "rel_kl":
            # 方案 B：只约束 batch 内 prompt 关系图与语义关系图一致。
            rp = _row_normalize(mu).matmul(_row_normalize(mu).t())
            rs = graph.index_select(0, targets_global).index_select(1, targets_global)
            pred = F.softmax(rp * prompt_scale, dim=-1)
            sem = F.softmax(rs / float(graph_cfg.TAU_SEM), dim=-1)
            loss = _kl_target_pred(sem.detach(), pred, eps)
        elif self.loss_type == "rel_all":
            # 方案 C：先经 batch 内 prompt 关系传播，再对齐到全类语义 target。
            rp = _row_normalize(mu).matmul(_row_normalize(mu).t())
            pred_batch = F.softmax(rp * prompt_scale, dim=-1)
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
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError("MODEL.SEMANTIC_GRAPH.OT_ALPHA must be in [0, 1].")
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


class GraphProbPriorLossComputer(torch.nn.Module):
    """
    GraphProbPrior: graph-conditioned semantic prior auxiliary loss。

    与标准 PromptKLAuxLoss 的区别：
    - 标准 KL 把 q(z|x)=N(mu, var) 拉向无条件 N(0,I)；
    - GraphProbPrior 先为每个类别 c 构造 p(z|c)=N(m_c, t_c)，
      再用 q(z|x_i) 到所有类别 prior 的 Gaussian KL 生成全类匹配分布。

    直观上，它是“概率版 acc_hidden”：
    acc_hidden 用 cosine(prompt_stat, semantic_bank_c) 产生类别分布；
    GraphProbPrior 用 -KL(q(z|x), p(z|c)) 产生类别分布。

    函数摘要：
    - __init__: 读取配置、校验温度/权重，并按 mode 创建 learned prior 或 residual-anchor prior head。
    - _standardized_residual_attributes: 把 [C,312] 属性置信度转成相对“平均鸟”的标准化残差。
    - _residual_anchor_class_priors: 用 residual_attr -> anchor -> graph top-k context -> delta 构造 prior_mu。
    - _class_priors: 按 PRIOR_MEAN_MODE 分派 learned prior 或 residual-anchor prior。
    - _gaussian_kl_all_classes: 计算每个样本 posterior 到所有 class prior 的 KL 距离。

    - _relation_regularization: class-aggregate 模式专用，用全类 prior Gaussian symKL 对齐语义图关系。
    - compute_prior_distribution_distance: 计算 prior 类别分布之间的标准化几何距离 D_cd。
    - compute_graph_prior_geometry_loss: 可选几何正则入口，支持 uniform_topk / graph_supcon / soft_distribution_matching。
    - _true_class_kl_loss: 逐样本只对齐真类 prior 的最简 KL matching。
    - _samplewise_latent_matching_loss: 逐样本 all-class KL matching，服务 graph_conditioned_semantic_prior 和 factorized semantic loss。

    - _aggregate_class_moments: 把 batch 内同类 posterior 聚合成类别级均值/方差。
    - _class_aggregate_moment_loss: 用聚合均值/方差对齐 class prior。
    - _rbf_kernel: class_aggregate_mmd 使用的 RBF kernel。
    - _class_aggregate_mmd_loss: 对同类 posterior mixture 和 class prior 做 MMD 分布匹配。
    - _factorized_variation_aggregate_loss: factorized 模式下可选的 variation 弱聚合约束。
    - _factorized_decouple_loss: factorized 模式下可选的 semantic/variation 去相关约束。
    - _factorized_latent_loss: factorized 模式总入口，只让 semantic factor 接语义 prior。
    - forward: 准备语义图、生成 class prior、按 MODE 分派并记录诊断 stats。
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # dual_metric_semantic_distribution 函数摘要：
        # - _masked_topk_distribution: 从图的一行里取非自身 top-k，并转成 P+ / P- 概率分布。
        # - _dual_class_priors: 用 312 维 residual anchor 与 P+ / P- 残差上下文生成 alpha/beta 两套 prior center。
        # - _dual_metric_semantic_distribution_loss: alpha 做语义上下文分布匹配，beta 做 hard-negative 间隔分离。
        # - forward: dual 模式不调用普通 _class_priors，而是直接进入双分布 loss 分支。
        # 复用 SemanticGraphBuilder，确保 GraphProbPrior 和旧 semantic graph loss 使用同一套 A/G/T。
        self.graph_builder = SemanticGraphBuilder(cfg)
        graph_cfg = cfg.MODEL.SEMANTIC_GRAPH
        prior_cfg = cfg.MODEL.GRAPH_PROB_PRIOR
        dist_cfg = cfg.MODEL.PROMPT.DISTRIBUTOR

        # 基础维度和 posterior/prior logvar 裁剪范围都和 prompt distributor 对齐。
        self.num_classes = int(graph_cfg.NUM_CLASSES)
        self.attr_dim = int(graph_cfg.ATTR_DIM)
        self.text_dim = int(graph_cfg.TEXT_DIM)
        self.hidden_dim = int(dist_cfg.STATS_HIDDEN_DIM)
        self.logvar_min = float(dist_cfg.LOGVAR_MIN)
        self.logvar_max = float(dist_cfg.LOGVAR_MAX)
        self.factorized_semantic_dim = int(dist_cfg.FACTORIZED_SEMANTIC_DIM)
        self.factorized_variation_dim = int(dist_cfg.FACTORIZED_VARIATION_DIM)
        self.mode = str(prior_cfg.MODE).lower()
        self.prior_mean_mode = str(prior_cfg.PRIOR_MEAN_MODE).lower()
        self.prior_var_mode = str(prior_cfg.PRIOR_VAR_MODE).lower()
        self.supported_modes = {
            "true_class_kl",
            "graph_conditioned_semantic_prior",
            "class_aggregate_moment",
            "class_aggregate_mmd",
            "factorized_latent",
            "dual_metric_semantic_distribution",
        }
        if self.mode not in self.supported_modes:
            raise ValueError(
                "MODEL.GRAPH_PROB_PRIOR.MODE must be true_class_kl / "
                "graph_conditioned_semantic_prior / class_aggregate_moment / "
                "class_aggregate_mmd / factorized_latent / dual_metric_semantic_distribution."
            )
        if self.prior_mean_mode not in {"learned", "residual_anchor"}:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_MEAN_MODE must be learned / residual_anchor.")
        if self.prior_var_mode not in {"learned", "unit", "constant"}:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_VAR_MODE must be learned / unit / constant.")

        tau_graph = float(prior_cfg.TAU_GRAPH)
        tau_latent = float(prior_cfg.TAU_LATENT)
        tau_prior = float(prior_cfg.TAU_PRIOR)
        # 温度必须为正；这些当前都是固定超参，不在这里创建 learnable parameter。
        if tau_graph <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.TAU_GRAPH must be positive.")
        if tau_latent <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.TAU_LATENT must be positive.")
        if tau_prior <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.TAU_PRIOR must be positive.")
        if float(prior_cfg.REL_WEIGHT) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.REL_WEIGHT must be non-negative.")
        if float(prior_cfg.MOMENT_VAR_WEIGHT) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MOMENT_VAR_WEIGHT must be non-negative.")
        if int(prior_cfg.MMD_SAMPLES) <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MMD_SAMPLES must be positive.")
        if float(prior_cfg.MMD_SIGMA) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MMD_SIGMA must be positive.")
        if float(prior_cfg.RESIDUAL_SIGMA_MIN) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.RESIDUAL_SIGMA_MIN must be positive.")
        if float(prior_cfg.RESIDUAL_CLIP) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.RESIDUAL_CLIP must be positive.")
        if int(prior_cfg.PRIOR_CONTEXT_TOPK) <= 0 or int(prior_cfg.PRIOR_CONTEXT_TOPK) > self.num_classes - 1:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_CONTEXT_TOPK must be in [1, NUM_CLASSES-1].")
        if float(prior_cfg.PRIOR_DELTA_SCALE) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_DELTA_SCALE must be non-negative.")
        if float(prior_cfg.PRIOR_MU_SCALE) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_MU_SCALE must be positive.")
        if float(prior_cfg.GEOM_LOSS_WEIGHT) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_WEIGHT must be non-negative.")
        if str(prior_cfg.GEOM_LOSS_TYPE).lower() not in {
            "uniform_topk",
            "graph_supcon",
            "soft_distribution_matching",
        }:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_TYPE must be uniform_topk / graph_supcon / soft_distribution_matching.")
        if int(prior_cfg.GEOM_TOPK) <= 0 or int(prior_cfg.GEOM_TOPK) > self.num_classes - 1:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_TOPK must be in [1, NUM_CLASSES-1].")
        for key in (
            "GEOM_TAU_UNI",
            "GEOM_TAU_GRAPH_TOP",
            "GEOM_MARGIN_GAMMA",
            "GEOM_SIGMA_PRIOR",
            "GEOM_TAU_CON",
            "GEOM_TAU_GRAPH_POS",
            "GEOM_TAU_BARRIER",
            "GEOM_TAU_GRAPH_DIST",
            "GEOM_TAU_DIST",
        ):
            if float(getattr(prior_cfg, key)) <= 0.0:
                raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{key} must be positive.")
        for key in (
            "GEOM_UNI_WEIGHT",
            "GEOM_TOP_WEIGHT",
            "GEOM_MARGIN_MIN",
            "GEOM_MARGIN_SCALE",
            "GEOM_CLEAR_WEIGHT",
            "GEOM_BOUND_WEIGHT",
        ):
            if float(getattr(prior_cfg, key)) < 0.0:
                raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{key} must be non-negative.")
        if self.factorized_semantic_dim <= 0:
            raise ValueError("DISTRIBUTOR.FACTORIZED_SEMANTIC_DIM must be positive.")
        if self.factorized_variation_dim <= 0:
            raise ValueError("DISTRIBUTOR.FACTORIZED_VARIATION_DIM must be positive.")
        if self.factorized_semantic_dim + self.factorized_variation_dim != self.text_dim:
            raise ValueError(
                "DISTRIBUTOR.FACTORIZED_SEMANTIC_DIM + FACTORIZED_VARIATION_DIM must equal TEXT_DIM."
            )
        if float(prior_cfg.FACTORIZED_VARIATION_WEIGHT) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.FACTORIZED_VARIATION_WEIGHT must be non-negative.")
        if float(prior_cfg.FACTORIZED_DECOUPLE_WEIGHT) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.FACTORIZED_DECOUPLE_WEIGHT must be non-negative.")
        if self.mode == "dual_metric_semantic_distribution":
            # dual_metric_semantic_distribution 使用 factorized posterior 的两段：
            # semantic_mu/logvar 是 alpha/context posterior，variation_mu/logvar 是 beta/separation posterior。
            # 这里不做旧 posterior 兜底，因此一旦进入该模式，就把所有双分布专用超参先校验清楚。
            if int(graph_cfg.TOPK) <= 0 or int(graph_cfg.TOPK) > self.num_classes - 1:
                raise ValueError(
                    "MODEL.SEMANTIC_GRAPH.TOPK must be in [1, NUM_CLASSES-1] for dual_metric_semantic_distribution."
                )
            if int(prior_cfg.DUAL_NEG_TOPK) <= 0 or int(prior_cfg.DUAL_NEG_TOPK) > self.num_classes - 1:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_NEG_TOPK must be in [1, NUM_CLASSES-1].")
            if float(prior_cfg.DUAL_TAU_NEG) <= 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_TAU_NEG must be positive.")
            if float(prior_cfg.DUAL_ALPHA_WEIGHT) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_ALPHA_WEIGHT must be non-negative.")
            if float(prior_cfg.DUAL_BETA_LOWER_WEIGHT) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_BETA_LOWER_WEIGHT must be non-negative.")
            if float(prior_cfg.DUAL_BETA_UPPER_WEIGHT) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_BETA_UPPER_WEIGHT must be non-negative.")
            if float(prior_cfg.DUAL_MARGIN_BASE) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_MARGIN_BASE must be non-negative.")
            if float(prior_cfg.DUAL_MARGIN_RISK_WEIGHT) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_MARGIN_RISK_WEIGHT must be non-negative.")
            if float(prior_cfg.DUAL_UPPER_BASE) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_UPPER_BASE must be non-negative.")
            if float(prior_cfg.DUAL_UPPER_CONTEXT_WEIGHT) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_UPPER_CONTEXT_WEIGHT must be non-negative.")
            if str(prior_cfg.DUAL_PRIOR_VAR_MODE).lower() != "unit":
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_PRIOR_VAR_MODE currently supports only 'unit'.")
            if float(prior_cfg.DUAL_ALPHA_DELTA_SCALE) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_ALPHA_DELTA_SCALE must be non-negative.")
            if float(prior_cfg.DUAL_BETA_DELTA_SCALE) < 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_BETA_DELTA_SCALE must be non-negative.")
        self.monitor_enable = bool(prior_cfg.MONITOR_ENABLE)
        self.monitor_inactive = bool(prior_cfg.MONITOR_INACTIVE)
        self.monitor_topk = int(prior_cfg.MONITOR_TOPK)
        self.monitor_every_n = int(prior_cfg.MONITOR_EVERY_N)
        if self.monitor_topk <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MONITOR_TOPK must be positive.")
        if self.monitor_every_n <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N must be positive.")

        def make_mlp(input_dim: int, output_dim: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_dim, output_dim),
            )

        # learned 是旧路径：cat(bank, dense_neighbor_bank) 直接输出 [prior_mu, prior_logvar]。
        # residual_anchor 是新路径：312 维标准化属性残差先生成 anchor，再用 graph top-k context 做小修正。
        full_prior_mode = self.mode not in {"factorized_latent", "dual_metric_semantic_distribution"}
        if full_prior_mode and self.prior_mean_mode == "learned":
            self.prior_head = make_mlp(self.text_dim * 2, self.text_dim * 2)
        if self.mode == "factorized_latent" and self.prior_mean_mode == "learned":
            self.factorized_semantic_prior_head = make_mlp(self.text_dim * 2, self.factorized_semantic_dim * 2)

        if full_prior_mode and self.prior_mean_mode == "residual_anchor":
            # 普通 GraphProbPrior 的 anchor_head: 312 维 residual_attr -> 768 维 latent anchor。
            # delta_head 只接收 [anchor, graph_topk_context, anchor-context]，
            # 因而它只能做小修正，不能像旧 prior_head 那样自由生成整个 prior_mu。
            self.residual_anchor_head = make_mlp(self.attr_dim, self.text_dim)
            self.residual_delta_head = make_mlp(self.text_dim * 3, self.text_dim)
            if self.prior_var_mode == "learned":
                self.residual_logvar_head = make_mlp(self.text_dim * 3, self.text_dim)
        if self.mode == "factorized_latent" and self.prior_mean_mode == "residual_anchor":
            # factorized_latent 只让 semantic factor 接语义 prior，所以这里输出 semantic_dim。
            self.factorized_residual_anchor_head = make_mlp(self.attr_dim, self.factorized_semantic_dim)
            self.factorized_residual_delta_head = make_mlp(self.factorized_semantic_dim * 3, self.factorized_semantic_dim)
            if self.prior_var_mode == "learned":
                self.factorized_residual_logvar_head = make_mlp(self.factorized_semantic_dim * 3, self.factorized_semantic_dim)

        if self.mode == "dual_metric_semantic_distribution":
            # dual 第一版按提示词改成 residual-anchor：
            # alpha/beta 都先从 312 维属性残差生成自己的 anchor，再分别用正邻域/难负邻域 residual 做小修正。
            self.dual_alpha_anchor_head = make_mlp(self.attr_dim, self.factorized_semantic_dim)
            self.dual_alpha_delta_head = make_mlp(self.factorized_semantic_dim * 3, self.factorized_semantic_dim)
            self.dual_beta_anchor_head = make_mlp(self.attr_dim, self.factorized_variation_dim)
            self.dual_beta_delta_head = make_mlp(self.factorized_variation_dim * 3, self.factorized_variation_dim)
        self._last_loss_stats: Dict[str, float] = {}
        self._debug_logged = False
        self._monitor_step = 0
        self._last_prior_debug: Dict[str, torch.Tensor] = {}

    def _standardized_residual_attributes(self, class_attributes: torch.Tensor) -> torch.Tensor:
        """
        构造提示词中的 312 维标准化属性残差。

        原始 A_conf[c, i] 表示类别 c 对属性 i 的置信度。直接使用它会包含大量“所有鸟共有”的属性。
        这里先减去全类平均鸟，再除以跨类标准差，得到“这个类相对平均鸟偏离多少个标准差”。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        mean = class_attributes.mean(dim=0, keepdim=True)
        std = class_attributes.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(prior_cfg.RESIDUAL_SIGMA_MIN))
        residual = (class_attributes - mean) / std
        return residual.clamp(min=-float(prior_cfg.RESIDUAL_CLIP), max=float(prior_cfg.RESIDUAL_CLIP))

    def _fixed_or_learned_logvar(
        self,
        reference: torch.Tensor,
        logvar_input: Optional[torch.Tensor] = None,
        factorized: bool = False,
    ) -> torch.Tensor:
        """
        根据 PRIOR_VAR_MODE 生成 prior_logvar。

        learned 只在当前 residual-anchor 分支显式创建 logvar_head 时使用；
        unit/constant 则完全固定方差，不让语义图无依据地预测类别不确定性。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        if self.prior_var_mode == "unit":
            return torch.zeros_like(reference)
        if self.prior_var_mode == "constant":
            return torch.full_like(reference, float(prior_cfg.PRIOR_LOGVAR_CONST))
        if logvar_input is None:
            raise RuntimeError("PRIOR_VAR_MODE=learned requires residual logvar input.")
        head = self.factorized_residual_logvar_head if bool(factorized) else self.residual_logvar_head
        return head(logvar_input).clamp(min=self.logvar_min, max=self.logvar_max)

    def _residual_anchor_class_priors(
        self,
        class_attributes: torch.Tensor,
        graph: torch.Tensor,
        factorized: bool = False,
    ):
        """
        新的 residual-anchor class prior。

        数据流：
        1. class_attributes [C,312] -> 标准化 residual_attr；
        2. residual_attr -> anchor_head -> 类别自身语义锚点 anchor；
        3. graph top-k -> P_plus，只聚合少数正邻域，不做 dense smoothing；
        4. context = P_plus @ anchor；
        5. delta_head(cat(anchor, context, anchor-context)) 给出小修正；
        6. prior_mu = PRIOR_MU_SCALE * normalize(anchor + scale*tanh(delta))。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)
        residual_attr = self._standardized_residual_attributes(class_attributes)

        if bool(factorized):
            anchor = self.factorized_residual_anchor_head(residual_attr)
            delta_head = self.factorized_residual_delta_head
        else:
            anchor = self.residual_anchor_head(residual_attr)
            delta_head = self.residual_delta_head

        positive_weight = self._masked_topk_distribution(
            graph,
            topk=int(prior_cfg.PRIOR_CONTEXT_TOPK),
            tau=float(prior_cfg.TAU_GRAPH),
            exclude_self=True,
        )
        context = positive_weight.matmul(anchor)
        delta_input = torch.cat((anchor, context, anchor - context), dim=-1)
        delta = delta_head(delta_input)

        # tanh(delta) 把修正方向限制在有界范围内；PRIOR_DELTA_SCALE 控制修正只能是 anchor 的小偏移。
        raw_mu = anchor + float(prior_cfg.PRIOR_DELTA_SCALE) * torch.tanh(delta)
        prior_mu = float(prior_cfg.PRIOR_MU_SCALE) * F.normalize(raw_mu, p=2, dim=-1, eps=eps)
        prior_logvar = self._fixed_or_learned_logvar(prior_mu, logvar_input=delta_input, factorized=factorized)

        # 仅缓存监测需要的中间张量。loss 仍从 prior_mu/prior_logvar 正常回传。
        self._last_prior_debug = {
            "residual_attr": residual_attr,
            "anchor": anchor,
            "context": context,
            "delta": delta,
            "positive_weight": positive_weight,
        }
        return prior_mu, prior_logvar

    def _learned_class_priors(self, bank: torch.Tensor, graph: torch.Tensor, factorized: bool = False):
        """
        旧 learned prior 路径。

        该路径保留给复现实验：dense softmax(graph/tau) 聚合 neighbor_bank，
        再由 prior_head 直接输出 prior_mu/prior_logvar。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        neighbor_weight = F.softmax(graph / float(prior_cfg.TAU_GRAPH), dim=-1)
        neighbor_bank = neighbor_weight.matmul(bank)
        prior_input = torch.cat((bank, neighbor_bank), dim=-1)
        prior_head = self.factorized_semantic_prior_head if bool(factorized) else self.prior_head
        prior_stats = prior_head(prior_input)
        prior_mu, prior_logvar = prior_stats.chunk(2, dim=-1)
        self._last_prior_debug = {}
        if self.prior_var_mode == "learned":
            prior_logvar = prior_logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        else:
            prior_logvar = self._fixed_or_learned_logvar(prior_mu)
        return prior_mu, prior_logvar

    def _class_priors(
        self,
        bank: torch.Tensor,
        graph: torch.Tensor,
        class_attributes: torch.Tensor,
        factorized: bool = False,
    ):
        """
        生成普通五个 mode 使用的全类 Gaussian prior。

        PRIOR_MEAN_MODE=learned 时走旧 prior_head；
        PRIOR_MEAN_MODE=residual_anchor 时走提示词要求的 312 维属性残差锚点。
        """
        if self.prior_mean_mode == "learned":
            return self._learned_class_priors(bank, graph, factorized=factorized)
        return self._residual_anchor_class_priors(class_attributes, graph, factorized=factorized)

    def _masked_topk_distribution(
        self,
        graph: torch.Tensor,
        topk: int,
        tau: float,
        exclude_self: bool = True,
    ) -> torch.Tensor:
        """
        从全类关系图的一整行里构造“只保留 top-k 类”的概率分布。

        输入:
        - graph: [C, C]，graph[c, d] 表示类别 c 到类别 d 的语义关系强度。
        - topk: 每一行保留多少个候选类。
        - tau: softmax 温度；越小，概率越集中到最高关系值的类别。
        - exclude_self: True 时把对角线 graph[c, c] 排除掉。

        输出:
        - prob: [C, C]，每行和为 1，非 top-k 位置为 0。

        dual_metric_semantic_distribution 里会调用两次:
        - P+ = _masked_topk_distribution(graph, SEMANTIC_GRAPH.TOPK, TAU_GRAPH)
          表示“正语义上下文”借信息比例。
        - P- = _masked_topk_distribution(graph, DUAL_NEG_TOPK, DUAL_TAU_NEG)
          表示“高风险难负类”加权比例。
        """
        class_count = int(graph.shape[0])
        topk = int(topk)
        tau = float(tau)
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)
        if graph.dim() != 2 or int(graph.shape[1]) != class_count:
            raise RuntimeError(f"GraphProbPrior expects square graph [C,C], got {tuple(graph.shape)}.")
        if topk <= 0:
            raise ValueError("topk must be positive.")
        max_topk = class_count - 1 if exclude_self else class_count
        if topk > max_topk:
            raise ValueError(f"topk must be <= {max_topk}, got {topk}.")
        if tau <= 0.0:
            raise ValueError("tau must be positive.")

        # candidate_logits 是可被选择的原始关系值。
        # exclude_self=True 时，对角线被置为 -inf，因此 top-k 不会选到类别自身。
        candidate_logits = graph
        if exclude_self:
            candidate_logits = graph.clone()
            diag = torch.arange(class_count, device=graph.device)
            candidate_logits[diag, diag] = float("-inf")

        # 只把每行 top-k 的关系值保留下来，其余位置保持 -inf。
        # 这样 softmax 后非 top-k 位置概率严格为 0，不会像 clamp 归一化那样给所有类漏一点概率。
        values, indices = torch.topk(candidate_logits, k=topk, dim=-1)
        masked_logits = torch.full_like(graph, float("-inf"))
        masked_logits.scatter_(1, indices, values)
        prob = F.softmax(masked_logits / tau, dim=-1)

        # 数值安全归一化：理论上 softmax 后每行已经为 1；这里仅防止极端 dtype/输入导致行和轻微漂移。
        prob = prob.masked_fill(~torch.isfinite(masked_logits), 0.0)
        return prob / prob.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _dual_class_priors(self, class_attributes: torch.Tensor, graph: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        为 dual_metric_semantic_distribution 生成两套 residual-anchor class prior center。

        alpha/beta 不再从 cat(bank, context) 直接自由生成，而是：
        1. 从 312 维属性残差分别生成 alpha_anchor / beta_anchor；
        2. alpha 用正邻域 context 做小修正；
        3. beta 显式看到 b-C+ 与 b-C-，同时知道正邻域和难负类方向；
        4. 两套 prior 方差固定为 I。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        if str(prior_cfg.DUAL_PRIOR_VAR_MODE).lower() != "unit":
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.DUAL_PRIOR_VAR_MODE currently supports only 'unit'.")

        residual_attr = self._standardized_residual_attributes(class_attributes)
        alpha_anchor = self.dual_alpha_anchor_head(residual_attr)
        beta_anchor = self.dual_beta_anchor_head(residual_attr)

        positive_weight = self._masked_topk_distribution(
            graph,
            topk=int(prior_cfg.PRIOR_CONTEXT_TOPK),
            tau=float(prior_cfg.TAU_GRAPH),
            exclude_self=True,
        )
        negative_weight = self._masked_topk_distribution(
            graph,
            topk=int(prior_cfg.DUAL_NEG_TOPK),
            tau=float(prior_cfg.DUAL_TAU_NEG),
            exclude_self=True,
        )

        alpha_context = positive_weight.matmul(alpha_anchor)
        beta_positive_context = positive_weight.matmul(beta_anchor)
        beta_negative_context = negative_weight.matmul(beta_anchor)

        alpha_input = torch.cat((alpha_anchor, alpha_context, alpha_anchor - alpha_context), dim=-1)
        beta_input = torch.cat(
            (
                beta_anchor,
                beta_anchor - beta_positive_context,
                beta_anchor - beta_negative_context,
            ),
            dim=-1,
        )
        alpha_delta = self.dual_alpha_delta_head(alpha_input)
        beta_delta = self.dual_beta_delta_head(beta_input)

        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)
        alpha_mu = float(prior_cfg.PRIOR_MU_SCALE) * F.normalize(
            alpha_anchor + float(prior_cfg.DUAL_ALPHA_DELTA_SCALE) * torch.tanh(alpha_delta),
            p=2,
            dim=-1,
            eps=eps,
        )
        beta_mu = float(prior_cfg.PRIOR_MU_SCALE) * F.normalize(
            beta_anchor + float(prior_cfg.DUAL_BETA_DELTA_SCALE) * torch.tanh(beta_delta),
            p=2,
            dim=-1,
            eps=eps,
        )

        alpha_logvar = torch.zeros_like(alpha_mu)
        beta_logvar = torch.zeros_like(beta_mu)
        return {
            "positive_weight": positive_weight,
            "negative_weight": negative_weight,
            "residual_attr": residual_attr,
            "alpha_anchor": alpha_anchor,
            "alpha_context": alpha_context,
            "alpha_delta": alpha_delta,
            "beta_anchor": beta_anchor,
            "beta_positive_context": beta_positive_context,
            "beta_negative_context": beta_negative_context,
            "beta_delta": beta_delta,
            "alpha_input": alpha_input,
            "beta_input": beta_input,
            "alpha_mu": alpha_mu,
            "alpha_logvar": alpha_logvar,
            "beta_mu": beta_mu,
            "beta_logvar": beta_logvar,
        }

    @staticmethod
    def _gaussian_kl_all_classes(
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        monitor: bool = False,
    ):
        """
        计算 D_ic = KL(q_i || p_c)，输出 [B, C]。

        q_i = N(mu_i, diag(var_i)) 当前 batch 每个样本的 posterior
        p_c = N(m_c, diag(t_c)) 所有类别的 class prior
        KL(q||p)=0.5*sum(log t - log var + (var + (mu-m)^2)/t - 1)
        """
        # 目标是一次性算出每个样本 i 到每个类别 c 的 KL：distance[i, c] = KL(q_i || p_c)
        # 所以通过插入维度触发广播，得到 [B, C, D] 的逐维 KL 项。
        q_mu = posterior_mu[:, None, :]
        q_logvar = posterior_logvar[:, None, :]
        p_mu = prior_mu[None, :, :]
        p_logvar = prior_logvar[None, :, :]
        q_var = q_logvar.exp()
        p_var = p_logvar.exp()

        kl = p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var - 1.0

        # 最后在 latent 维度 D 上求和，得到 [B, C]。这里不 detach，梯度会回传到 posterior 参数和 prior head。
        return 0.5 * kl.sum(dim=-1)

    def _relation_regularization(
        self,
        graph: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        monitor: bool = False,
    ):
        """
        class-aggregate InfoVAE/WAE-style matching 专用的全类 prior 关系正则。

        class_aggregate_moment / class_aggregate_mmd 只直接对当前 batch 出现类别做
        posterior aggregate -> class prior matching。该正则额外对所有类别的 prior
        Gaussian 建立关系矩阵，让 seen/unseen 全类 prior 都按语义图 G 组织起来。

        关系定义：
            P_c = N(mu_c, diag(var_c))
            D_cd = 0.5 * [KL(P_c || P_d) + KL(P_d || P_c)]
            pred_rel[c] = softmax(-D_c / TAU_PRIOR)
            target_rel[c] = softmax(G_c / TAU_GRAPH)

        这样 prior_mu 和 prior_logvar 都会受到全类结构约束。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)

        # 语义图 target 是固定监督信号，不让梯度回传到 graph。
        target_rel = F.softmax(graph / float(prior_cfg.TAU_GRAPH), dim=-1).detach()

        # 以下用矩阵公式计算所有类别 prior 两两之间的 KL，避免构造 [C,C,D] 大张量。
        # KL(P_c || P_d) = 0.5 * sum[logvar_d - logvar_c + var_c / var_d + (mu_c - mu_d)^2 / var_d - 1]
        class_count = int(prior_mu.shape[0])
        latent_dim = int(prior_mu.shape[1])
        prior_var = prior_logvar.exp()
        prior_inv_var = torch.exp(-prior_logvar)
        logvar_sum = prior_logvar.sum(dim=-1)

        log_term = logvar_sum[None, :] - logvar_sum[:, None]
        var_term = prior_var.matmul(prior_inv_var.t())

        mu_sq = prior_mu.pow(2)
        mu_sq_over_var = mu_sq.matmul(prior_inv_var.t())
        cross = prior_mu.matmul((prior_mu * prior_inv_var).t())
        prior_self_quad = (mu_sq * prior_inv_var).sum(dim=-1)
        mean_term = mu_sq_over_var - 2.0 * cross + prior_self_quad[None, :]

        kl_cd = 0.5 * (log_term + var_term + mean_term - float(latent_dim))
        sym_kl = 0.5 * (kl_cd + kl_cd.t())
        sym_kl = sym_kl.clamp_min(0.0)

        if tuple(sym_kl.shape) != (class_count, class_count):
            raise RuntimeError(
                f"GraphProbPrior relation matrix must be [{class_count},{class_count}], got {tuple(sym_kl.shape)}."
            )

        # 距离越小表示两个 class prior 越接近，因此用 -sym_kl 转成关系 logits。
        pred_rel = F.softmax(-sym_kl / float(prior_cfg.TAU_PRIOR), dim=-1)
        monitor_stats = relation_monitor(sym_kl, pred_rel, topk=self.monitor_topk) if monitor else {}
        return _kl_target_pred(target_rel, pred_rel, eps), monitor_stats

    def _geometry_topk(self, graph: torch.Tensor, topk: int, tau: float):
        """
        geometry loss 专用 top-k 读取。

        它和 _masked_topk_distribution 的核心一致，但这里还需要返回 top-k 下标和值，
        因为三种 geometry loss 都要在 D_cd 或 similarity 矩阵里 gather 对应类别对。
        """
        class_count = int(graph.shape[0])
        logits = graph.clone()
        diag = torch.arange(class_count, device=graph.device)
        logits[diag, diag] = float("-inf")
        values, indices = torch.topk(logits, k=int(topk), dim=-1)
        weights = F.softmax(values / float(tau), dim=-1)
        return values, indices, weights

    def compute_prior_distribution_distance(
        self,
        prior_mu: torch.Tensor,
        prior_logvar: Optional[torch.Tensor] = None,
    ):
        """
        计算提示词中的类别分布标准化距离 D_cd。

        dist_mu(c,d) = ||mu_c - mu_d||_2 / sqrt(D)
        D_cd = dist_mu(c,d) / (r_c + r_d + eps)

        如果 prior 方差是 unit/constant，就使用 GEOM_SIGMA_PRIOR 作为半径；
        如果 prior 方差是 learned，就从 exp(prior_logvar) 估计半径，并按配置决定是否 detach。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)
        latent_dim = int(prior_mu.shape[1])
        dist_mu = torch.cdist(prior_mu, prior_mu, p=2) / math.sqrt(float(latent_dim))

        if self.prior_var_mode == "learned" and prior_logvar is not None:
            radius = prior_logvar.exp().mean(dim=-1).clamp_min(eps).sqrt()
        else:
            radius = torch.full(
                (int(prior_mu.shape[0]),),
                float(prior_cfg.GEOM_SIGMA_PRIOR),
                device=prior_mu.device,
                dtype=prior_mu.dtype,
            )
        if bool(prior_cfg.GEOM_DETACH_RADIUS):
            radius = radius.detach()
        d_norm = dist_mu / (radius[:, None] + radius[None, :] + eps)
        eye = torch.eye(int(prior_mu.shape[0]), dtype=torch.bool, device=prior_mu.device)
        d_norm = d_norm.masked_fill(eye, 0.0)
        return d_norm, dist_mu, radius

    def _graph_prior_geometry_stats(
        self,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        graph: torch.Tensor,
    ) -> Dict[str, float]:
        """只计算 geometry 监测量，不产生额外 loss。"""
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        d_norm, dist_mu, radius = self.compute_prior_distribution_distance(prior_mu, prior_logvar)
        _, top_indices, _ = self._geometry_topk(
            graph,
            topk=int(prior_cfg.GEOM_TOPK),
            tau=float(prior_cfg.GEOM_TAU_GRAPH_DIST),
        )
        return graph_prior_geometry_monitor(
            d_norm,
            dist_mu,
            radius,
            prior_mu,
            graph=graph,
            top_indices=top_indices,
            margin_min=float(prior_cfg.GEOM_MARGIN_MIN),
        )

    def compute_graph_prior_geometry_loss(
        self,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        graph: torch.Tensor,
    ):
        """
        prior_mu 几何校准正则统一入口。

        它不生成 prior_mu，只在 residual-anchor 已经得到 prior_mu 后追加轻量约束：
        - uniform_topk: 全局 soft repulsion + graph top-k 距离校准；
        - graph_supcon: graph top-k 作为 soft positives，其他类进入 contrastive denominator；
        - soft_distribution_matching: 让 graph top-k 分布与 prior-distance 诱导分布一致。
        """
        if self.mode == "dual_metric_semantic_distribution":
            raise RuntimeError("GEOM_LOSS_ENABLE=True is not connected to dual_metric_semantic_distribution in this version.")

        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)
        method = str(prior_cfg.GEOM_LOSS_TYPE).lower()
        class_count = int(prior_mu.shape[0])
        row = torch.arange(class_count, device=prior_mu.device)[:, None]
        eye = torch.eye(class_count, dtype=torch.bool, device=prior_mu.device)
        d_norm, dist_mu, radius = self.compute_prior_distribution_distance(prior_mu, prior_logvar)

        stats = graph_prior_geometry_monitor(
            d_norm,
            dist_mu,
            radius,
            prior_mu,
            graph=graph,
            margin_min=float(prior_cfg.GEOM_MARGIN_MIN),
        )

        if method == "uniform_topk":
            off_d = d_norm[~eye]
            uni_logits = -float(prior_cfg.GEOM_TAU_UNI) * off_d.pow(2)
            uniform_loss = torch.logsumexp(uni_logits, dim=0) - math.log(float(max(int(off_d.numel()), 1)))

            top_values, top_indices, top_weight = self._geometry_topk(
                graph,
                topk=int(prior_cfg.GEOM_TOPK),
                tau=float(prior_cfg.GEOM_TAU_GRAPH_TOP),
            )
            top_d = d_norm[row, top_indices]
            denom = (top_values[:, :1] - top_values[:, -1:]).clamp_min(eps)
            local_rank_score = (top_values - top_values[:, -1:]) / denom
            target_d = 1.0 + float(prior_cfg.GEOM_MARGIN_MIN) + float(prior_cfg.GEOM_MARGIN_SCALE) * (
                1.0 - local_rank_score
            ).pow(float(prior_cfg.GEOM_MARGIN_GAMMA))
            top_loss = top_weight.mul(F.smooth_l1_loss(top_d, target_d.detach(), reduction="none")).sum(dim=-1).mean()
            loss = float(prior_cfg.GEOM_UNI_WEIGHT) * uniform_loss + float(prior_cfg.GEOM_TOP_WEIGHT) * top_loss
            stats.update(
                {
                    "graph_prob_prior_geom_uniform_loss": float(uniform_loss.detach().item()),
                    "graph_prob_prior_geom_top_loss": float(top_loss.detach().item()),
                    "graph_prob_prior_geom_loss": float(loss.detach().item()),
                }
            )
            stats.update(
                graph_prior_geometry_monitor(
                    d_norm,
                    dist_mu,
                    radius,
                    prior_mu,
                    graph=graph,
                    top_indices=top_indices,
                    margin_min=float(prior_cfg.GEOM_MARGIN_MIN),
                )
            )
            return loss, stats

        if method == "graph_supcon":
            top_values, top_indices, positive_weight = self._geometry_topk(
                graph,
                topk=int(prior_cfg.GEOM_TOPK),
                tau=float(prior_cfg.GEOM_TAU_GRAPH_POS),
            )
            if bool(prior_cfg.GEOM_USE_DISTANCE_SIM):
                similarity = -d_norm
            else:
                similarity = F.normalize(prior_mu, p=2, dim=-1, eps=eps).matmul(
                    F.normalize(prior_mu, p=2, dim=-1, eps=eps).t()
                )
            logits = similarity / float(prior_cfg.GEOM_TAU_CON)
            logits = logits.masked_fill(eye, float("-inf"))
            log_den = torch.logsumexp(logits, dim=-1)
            pos_logits = logits[row, top_indices]
            gcon_loss = -positive_weight.mul(pos_logits - log_den[:, None]).sum(dim=-1).mean()

            top_d = d_norm[row, top_indices]
            clear_loss = F.softplus(
                (1.0 + float(prior_cfg.GEOM_MARGIN_MIN) - top_d) / float(prior_cfg.GEOM_TAU_BARRIER)
            ).mul(positive_weight).sum(dim=-1).mean()
            loss = gcon_loss + float(prior_cfg.GEOM_CLEAR_WEIGHT) * clear_loss
            stats.update(
                {
                    "graph_prob_prior_geom_gcon_loss": float(gcon_loss.detach().item()),
                    "graph_prob_prior_geom_gcon_clear_loss": float(clear_loss.detach().item()),
                    "graph_prob_prior_geom_loss": float(loss.detach().item()),
                    "graph_prob_prior_geom_gcon_den_logsumexp_mean": float(log_den.detach().mean().item()),
                    "graph_prob_prior_geom_gcon_pos_sim_mean": float(similarity[row, top_indices].detach().mean().item()),
                }
            )
            stats.update(
                graph_prior_geometry_monitor(
                    d_norm,
                    dist_mu,
                    radius,
                    prior_mu,
                    graph=graph,
                    top_indices=top_indices,
                    margin_min=float(prior_cfg.GEOM_MARGIN_MIN),
                )
            )
            return loss, stats

        if method == "soft_distribution_matching":
            top_values, top_indices, target_weight = self._geometry_topk(
                graph,
                topk=int(prior_cfg.GEOM_TOPK),
                tau=float(prior_cfg.GEOM_TAU_GRAPH_DIST),
            )
            logits = (-d_norm / float(prior_cfg.GEOM_TAU_DIST)).masked_fill(eye, float("-inf"))
            log_q = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            log_q_top = log_q[row, top_indices]
            dist_match_kl = target_weight.mul(target_weight.clamp_min(eps).log() - log_q_top).sum(dim=-1).mean()

            top_d = d_norm[row, top_indices]
            bound_loss = F.softplus(
                (1.0 + float(prior_cfg.GEOM_MARGIN_MIN) - top_d) / float(prior_cfg.GEOM_TAU_BARRIER)
            ).mul(target_weight).sum(dim=-1).mean()
            loss = dist_match_kl + float(prior_cfg.GEOM_BOUND_WEIGHT) * bound_loss
            q_top = log_q_top.exp()
            q_entropy = -(log_q.exp() * log_q).masked_fill(eye, 0.0).sum(dim=-1).mean()
            stats.update(
                {
                    "graph_prob_prior_geom_dist_match_kl": float(dist_match_kl.detach().item()),
                    "graph_prob_prior_geom_dist_match_bound_loss": float(bound_loss.detach().item()),
                    "graph_prob_prior_geom_loss": float(loss.detach().item()),
                    "graph_prob_prior_geom_dist_match_top_prob_mean": float(target_weight.detach().mean().item()),
                    "graph_prob_prior_geom_dist_match_q_top_prob_mean": float(q_top.detach().mean().item()),
                    "graph_prob_prior_geom_dist_match_q_entropy_mean": float(q_entropy.detach().item()),
                }
            )
            stats.update(
                graph_prior_geometry_monitor(
                    d_norm,
                    dist_mu,
                    radius,
                    prior_mu,
                    graph=graph,
                    top_indices=top_indices,
                    margin_min=float(prior_cfg.GEOM_MARGIN_MIN),
                )
            )
            return loss, stats

        raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_TYPE must be uniform_topk / graph_supcon / soft_distribution_matching.")

    def _samplewise_latent_matching_loss(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        target: torch.Tensor,
        targets_global: Optional[torch.Tensor] = None,
        monitor: bool = False,
    ):
        """
        第一种 GraphProbPrior 模式：逐样本 latent matching。

        每个样本 q(z|x_i) 与所有类别 prior p(z|c) 计算 Gaussian KL，
        再用 softmax(-KL/tau) 得到 all-class latent matching 分布，
        最后对齐语义图 target T_i。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)

        # distance: [B, C]
        distance = self._gaussian_kl_all_classes(posterior_mu, posterior_logvar, prior_mu, prior_logvar)

        # 把“距离”变成“类别概率”：距离越小 -> -distance 越大 -> softmax 后概率越高。
        # TAU_LATENT 是 prediction 侧温度：tau 小 -> latent_prob 更尖，更偏向 KL 最小的类别； tau 大 -> latent_prob 更平，多个类别会分到概率。
        latent_prob = F.softmax(-distance / float(prior_cfg.TAU_LATENT), dim=-1)

        # target 是 SemanticGraphBuilder 根据 graph[y] 构造的语义监督分布 [B, C]。
        # 这里训练 latent_prob 去贴近 target：
        #   KL(T_i || latent_prob_i)
        # 注意 target 在 builder 中已经 detach，梯度只回到 posterior / prior。
        match_loss = _kl_target_pred(target, latent_prob, eps)

        # entropy 只是诊断项：
        #   熵低 -> latent_prob 很尖；
        #   熵高 -> latent_prob 很平。
        # 它不单独加进 loss，只用于日志判断 TAU_LATENT / distance 尺度是否合理。
        entropy = -(latent_prob * latent_prob.clamp_min(eps).log()).sum(dim=-1).mean()

        # stats 会经 GraphProbPriorAuxLoss -> CompositeLoss 合并到 trainer 日志。
        stats = {
            "graph_prob_prior_match_loss": float(match_loss.detach().item()),
            "graph_prob_prior_latent_entropy": float(entropy.detach().item()),
            "graph_prob_prior_distance_mean": float(distance.detach().mean().item()),
        }
        if monitor:
            stats.update(latent_matching_monitor(distance, latent_prob, targets_global, topk=self.monitor_topk))
            stats.update(posterior_prior_alignment_monitor(distance, latent_prob, targets_global, topk=self.monitor_topk))

        # debug 只记录 shape，配合 GRAPH_PROB_PRIOR.DEBUG=True 时打印一次。
        debug = {
            "distance_shape": tuple(distance.shape),
            "target_shape": tuple(target.shape),
            "latent_prob_shape": tuple(latent_prob.shape),
        }
        return match_loss, stats, debug

    def _true_class_kl_loss(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        monitor: bool = False,
    ):
        """
        最简 GraphProbPrior 替换项：每个样本只和自己的真类 class prior 做 KL。

        prior 仍然由 _class_priors() 生成；当 PRIOR_MEAN_MODE=residual_anchor 时，
        它来自 312 维属性残差锚点和 graph top-k context。区别是 loss 不再构造 all-class target，
        也不使用 softmax(-KL/TAU_LATENT)，只优化 KL(q_i || p_{y_i})。
        """
        selected_prior_mu = prior_mu.index_select(0, targets_global)
        selected_prior_logvar = prior_logvar.index_select(0, targets_global)

        posterior_var = posterior_logvar.exp()
        selected_prior_var = selected_prior_logvar.exp()
        kl = (
            selected_prior_logvar
            - posterior_logvar
            + (posterior_var + (posterior_mu - selected_prior_mu).pow(2)) / selected_prior_var
            - 1.0
        )
        true_kl_per_sample = 0.5 * kl.sum(dim=-1)
        match_loss = true_kl_per_sample.mean()

        stats = {
            "graph_prob_prior_match_loss": float(match_loss.detach().item()),
            "graph_prob_prior_true_kl_mean": float(true_kl_per_sample.detach().mean().item()),
            "graph_prob_prior_prior_var_mean": float(selected_prior_var.detach().mean().item()),
        }
        if monitor:
            stats.update(true_class_kl_monitor(true_kl_per_sample))
        debug = {
            "selected_prior_mu_shape": tuple(selected_prior_mu.shape),
            "selected_prior_logvar_shape": tuple(selected_prior_logvar.shape),
            "true_kl_shape": tuple(true_kl_per_sample.shape),
        }
        return match_loss, stats, debug

    @staticmethod
    def _aggregate_class_moments(
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
    ):
        """
        构造 batch 内出现类别的聚合 posterior moment。

        对每个当前 batch 出现的类别 c：
        - 聚合均值 mean_c = mean_i mu_i；
        - 聚合方差 var_c = mean_i [var_i + (mu_i - mean_c)^2]。

        当某个类别只有一个样本时，第二项自动为 0，
        因而 var_c 就是该样本自己的 posterior variance，不会产生 NaN。
        """
        # 只聚合当前 batch 真实出现过的类别，避免对没有样本的类别构造空统计。
        class_ids = torch.unique(targets_global, sorted=True)
        posterior_var = posterior_logvar.exp()
        class_means = []
        class_vars = []
        for class_id in class_ids:
            mask = targets_global == class_id
            mu_c = posterior_mu[mask]
            var_c = posterior_var[mask]

            # aggregate_var 使用全方差公式：
            # E[var_i] + Var(mu_i)，既包含每个样本自己的 posterior 不确定性，
            # 也包含同类样本 posterior center 的离散程度。
            mean_c = mu_c.mean(dim=0)
            aggregate_var_c = (var_c + (mu_c - mean_c).pow(2)).mean(dim=0)
            class_means.append(mean_c)
            class_vars.append(aggregate_var_c)
        return class_ids, torch.stack(class_means, dim=0), torch.stack(class_vars, dim=0)

    def _class_aggregate_moment_loss(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        monitor: bool = False,
    ):
        """
        第二种 GraphProbPrior 模式：class-aggregate moment matching。

        它不再逐样本强迫 q(z|x_i) 靠近 p(z|y_i)，而是先把同一类别
        在当前 batch 内的 posterior 聚合成 Q_c，再让 Q_c 的均值/方差
        对齐 graph-conditioned class prior P_c。这样可以保留类内实例差异。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)

        # 1. 先把当前 batch 中同一全局类别的 posterior 聚合起来。
        # 输入：posterior_mu/logvar: [B, D] targets_global:  [B]
        # 输出：class_ids:      当前 batch 出现过的全局类别 id，形状 [K]
        #   aggregate_*:   每个出现类别的聚合 posterior 均值和方差，形状 [K, D]
        # 这里的 K 是当前 batch 中出现的类别数，不一定等于全类数 C。
        class_ids, aggregate_mu, aggregate_var = self._aggregate_class_moments(
            posterior_mu,
            posterior_logvar,
            targets_global,
        )

        # 2. 从全类 prior 中取出当前 batch 出现类别对应的 prior。
        # prior_mu/logvar 是 _class_priors() 生成的全类 prior，形状 [C, D]。
        # residual_anchor 模式下，prior_mu 主体来自 312 维属性残差 anchor，prior_logvar 可固定为 unit/constant。
        # 取出selected_prior_mu/logvar: [K, D]和 aggregate_mu/var 按类别一一对应。
        # 只取当前 batch 出现类别对应的 prior，与聚合 posterior Q_c 一一对齐。
        selected_prior_mu = prior_mu.index_select(0, class_ids)
        selected_prior_logvar = prior_logvar.index_select(0, class_ids)
        selected_prior_var = selected_prior_logvar.exp()

        # 3. 均值匹配项。
        # 目标：让同类 posterior 聚合中心 aggregate_mu[c]靠近该类 graph-conditioned prior 的中心 selected_prior_mu[c]。
        # 这里使用 prior 方差做 precision weighting：
        #   prior_var 小 -> prior 对该维度更有把握 -> 均值偏移惩罚更强；
        #   prior_var 大 -> prior 对该维度更不确定 -> 均值偏移惩罚更弱。
        # mean_loss_per_class 形状 [K]，每个类别一个均值匹配损失。
        # prior 很确定的维度，不能乱偏，prior 不确定的维度，可以多偏一点。
        mean_loss_per_class = (aggregate_mu - selected_prior_mu).pow(2).div(selected_prior_var).sum(dim=-1)

        # 4. 方差匹配项。
        # aggregate_var 是聚合 posterior 方差，先转成 log 方差，再和 selected_prior_logvar 对齐。
        # clamp_min(eps) 是为了避免 log(0)。
        log_aggregate_var = aggregate_var.clamp_min(eps).log()
        var_loss_per_class = (log_aggregate_var - selected_prior_logvar).pow(2).sum(dim=-1)

        # 5. 对当前 batch 出现类别取平均。
        # 这里按类别平均，而不是按样本平均，所以每个出现类别在该项中权重相同。
        mean_loss = mean_loss_per_class.mean()
        var_loss = var_loss_per_class.mean()

        # MOMENT_VAR_WEIGHT 控制方差匹配项强度；均值匹配始终保留。MOMENT_VAR_WEIGHT=0 时：只对齐类别聚合均值；MOMENT_VAR_WEIGHT>0 时：同时约束聚合方差与 prior 方差一致。
        match_loss = mean_loss + float(prior_cfg.MOMENT_VAR_WEIGHT) * var_loss

        # 6. 记录日志指标。
        # 注意这里的 graph_prob_prior_match_loss 是未乘外层 LOSS_WEIGHT 的原始 aux loss。
        stats = {
            "graph_prob_prior_match_loss": float(match_loss.detach().item()),
            "graph_prob_prior_agg_mean_loss": float(mean_loss.detach().item()),
            "graph_prob_prior_agg_var_loss": float(var_loss.detach().item()),
            "graph_prob_prior_agg_class_count": float(class_ids.numel()),
            "graph_prob_prior_agg_var_mean": float(aggregate_var.detach().mean().item()),
        }
        if monitor:
            stats.update(
                aggregate_moment_monitor(
                    posterior_mu,
                    posterior_logvar,
                    targets_global,
                    aggregate_mu,
                    aggregate_var,
                    class_ids,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                )
            )

        # debug 只记录 shape，配合 GRAPH_PROB_PRIOR.DEBUG=True 打印一次。
        debug = {
            "class_ids_shape": tuple(class_ids.shape),
            "aggregate_mu_shape": tuple(aggregate_mu.shape),
            "aggregate_var_shape": tuple(aggregate_var.shape),
        }
        return match_loss, stats, debug

    @staticmethod
    def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """class_aggregate_mmd 使用的 RBF kernel。"""
        # x/y 是采样出的 latent 点集；kernel 越接近 1 表示两点越近。
        dist = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-dist / (2.0 * float(sigma) * float(sigma)))

    def _class_aggregate_mmd_loss(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        monitor: bool = False,
    ):
        """
        可选的 class-aggregate MMD matching。

        对 batch 内每个出现类别 c，从该类 posterior mixture 和对应 prior
        中采样 latent，然后用 RBF-MMD 对齐两个分布。该模式不使用 memory bank，
        只依赖当前 batch 内样本。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR

        # 1. 只处理当前 batch 中真实出现过的类别。
        # class_ids 形状 [K]，K 是当前 batch 出现类别数。
        class_ids = torch.unique(targets_global, sorted=True)

        # 每个 posterior 高斯采样多少次；RBF kernel 的宽度 sigma。 sample_count 越大，MMD 估计更稳定，但计算量也更大。
        sample_count = int(prior_cfg.MMD_SAMPLES)
        sigma = float(prior_cfg.MMD_SIGMA)
        class_losses = []
        monitor_pair_dists = []
        monitor_kernels = []

        # 将 logvar 转成 std，后面用重参数化采样 latent 点。
        posterior_std = torch.exp(0.5 * posterior_logvar)
        prior_std = torch.exp(0.5 * prior_logvar)

        # 2. 对每个出现类别分别做 posterior mixture vs class prior 的两样本 MMD。
        for class_id in class_ids:
            # mask 选出当前类别 c 在 batch 中的所有样本。
            # mu_c/std_c 形状 [N_c, D]，N_c 是该类在当前 batch 中的样本数。
            mask = targets_global == class_id
            mu_c = posterior_mu[mask]
            std_c = posterior_std[mask]

            # 从当前类别的 posterior mixture 中采样：
            # 每个样本 posterior 采 MMD_SAMPLES 次，再展平成一个点集。
            # posterior_eps:     [N_c, sample_count, D]
            # posterior_samples: [N_c * sample_count, D]
            # 这相当于从该类别的 batch posterior mixture:Q_c = average_i q_i(z|x_i)中抽样。
            posterior_eps = torch.randn(
                mu_c.shape[0],
                sample_count,
                mu_c.shape[1],
                device=mu_c.device,
                dtype=mu_c.dtype,
            )
            posterior_samples = mu_c[:, None, :] + std_c[:, None, :] * posterior_eps
            posterior_samples = posterior_samples.reshape(-1, mu_c.shape[1])

            # 从该类别对应的 class prior 中采同样数量的点，便于做两样本 MMD。
            # prior_mu_c/std_c 是单个类别 prior 的参数，形状 [D]。
            # prior_samples 数量对齐 posterior_samples，形状 [N_c * sample_count, D]。
            prior_mu_c = prior_mu.index_select(0, class_id.view(1)).squeeze(0)
            prior_std_c = prior_std.index_select(0, class_id.view(1)).squeeze(0)
            prior_eps = torch.randn(
                posterior_samples.shape[0],
                prior_mu_c.shape[0],
                device=prior_mu_c.device,
                dtype=prior_mu_c.dtype,
            )
            prior_samples = prior_mu_c[None, :] + prior_std_c[None, :] * prior_eps

            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]。
            # k_xx: posterior samples 内部相似度，描述 Q_c 自身分布；
            # k_yy: prior samples 内部相似度，描述 P_c 自身分布；
            # k_xy: posterior 与 prior 之间相似度。
            # 当两个分布接近时，k_xx + k_yy - 2*k_xy 会变小。
            k_xx = self._rbf_kernel(posterior_samples, posterior_samples, sigma).mean()
            k_yy = self._rbf_kernel(prior_samples, prior_samples, sigma).mean()
            if monitor:
                pair_dist = torch.cdist(posterior_samples, prior_samples, p=2).pow(2)
                kernel_xy = torch.exp(-pair_dist / (2.0 * sigma * sigma))
                k_xy = kernel_xy.mean()
                monitor_pair_dists.append(pair_dist.detach().reshape(-1))
                monitor_kernels.append(kernel_xy.detach().reshape(-1))
            else:
                k_xy = self._rbf_kernel(posterior_samples, prior_samples, sigma).mean()
            class_losses.append(k_xx + k_yy - 2.0 * k_xy)

        # 3. 对当前 batch 出现类别取平均，得到 class_aggregate_mmd 的 match_loss。
        # 这里同样是按类别平均，而不是按样本数加权平均。
        match_loss = torch.stack(class_losses).mean()

        # 记录未乘外层 LOSS_WEIGHT 的原始 MMD loss，以及本次 batch 的类别数/采样数。
        stats = {
            "graph_prob_prior_match_loss": float(match_loss.detach().item()),
            "graph_prob_prior_mmd_loss": float(match_loss.detach().item()),
            "graph_prob_prior_mmd_class_count": float(class_ids.numel()),
            "graph_prob_prior_mmd_samples": float(sample_count),
        }
        if monitor and monitor_pair_dists and monitor_kernels:
            stats.update(mmd_monitor(torch.cat(monitor_pair_dists), torch.cat(monitor_kernels), sigma=sigma))

        # debug 只记录 shape，配合 GRAPH_PROB_PRIOR.DEBUG=True 打印一次。
        debug = {
            "class_ids_shape": tuple(class_ids.shape),
        }
        return match_loss, stats, debug

    def _dual_metric_semantic_distribution_loss(
        self,
        alpha_mu: torch.Tensor,
        alpha_logvar: torch.Tensor,
        beta_mu: torch.Tensor,
        beta_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        class_attributes: torch.Tensor,
        graph: torch.Tensor,
        monitor: bool = False,
    ):
        """
        第六种 GraphProbPrior 模式：Context-Separation Dual-Metric Semantic Distribution。

        该模式把 factorized posterior 当成两套不同用途的后验:
        - alpha 后验 q_x^alpha: 来自 semantic_mu/logvar，负责“语义上下文/可迁移性”。
        - beta 后验 q_x^beta: 来自 variation_mu/logvar，负责“难负类分离/类别边界”。

        训练目标由三部分组成:
        1. alpha_loss:
           让 q_x^alpha 到所有 alpha class prior 的 KL-softmax 分布，
           对齐 “one-hot 真类 + P+ 正上下文” 的 target。
        2. beta_lower_loss:
           对 hard negatives 施加间隔约束，让样本到负类 beta prior 的距离
           至少大于“到真类 beta prior 的距离 + margin”。
        3. beta_upper_loss:
           可选弱上界，让 beta 空间里正上下文类不要离得过远；
           默认权重为 0，因此第一版默认不启用。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
        eps = float(graph_cfg.OT_DELTA)

        dual_priors = self._dual_class_priors(class_attributes, graph)
        positive_weight = dual_priors["positive_weight"]
        negative_weight = dual_priors["negative_weight"]
        alpha_prior_mu = dual_priors["alpha_mu"]
        alpha_prior_logvar = dual_priors["alpha_logvar"]
        beta_prior_mu = dual_priors["beta_mu"]
        beta_prior_logvar = dual_priors["beta_logvar"]

        # alpha target: 真类 one-hot 负责保留判别主锚点，P+ 负责注入正语义上下文。
        # P+ 已经排除了自身类，因此 TARGET_MIX_ALPHA 不会把对角线 1.0 再重复混入语义邻居。
        positive_for_batch = positive_weight.index_select(0, targets_global)
        onehot = F.one_hot(targets_global, num_classes=self.num_classes).to(dtype=alpha_mu.dtype)
        mix_alpha = float(graph_cfg.TARGET_MIX_ALPHA)
        if mix_alpha < 0.0 or mix_alpha > 1.0:
            raise ValueError("MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA must be in [0, 1].")
        alpha_target = _normalize_prob((1.0 - mix_alpha) * onehot + mix_alpha * positive_for_batch, eps).detach()

        # alpha 分布匹配：计算 KL(q_x^alpha || p_c^alpha)，再用 softmax(-KL/TAU_LATENT)
        # 得到模型预测的全类语义上下文分布。
        alpha_distance = self._gaussian_kl_all_classes(alpha_mu, alpha_logvar, alpha_prior_mu, alpha_prior_logvar)
        alpha_prob = F.softmax(-alpha_distance / float(prior_cfg.TAU_LATENT), dim=-1)
        alpha_loss = _kl_target_pred(alpha_target, alpha_prob, eps)

        # beta 距离矩阵：distance_beta[i, c] = KL(q_i^beta || p_c^beta)。
        # 对每个样本 i，希望真类 y_i 的距离比 hard negative 类更小。
        beta_distance = self._gaussian_kl_all_classes(beta_mu, beta_logvar, beta_prior_mu, beta_prior_logvar)
        true_beta_distance = beta_distance.gather(1, targets_global[:, None])
        negative_for_batch = negative_weight.index_select(0, targets_global)
        risk_for_batch = graph.index_select(0, targets_global)

        # hard-negative 下界:
        #   D_beta(i, neg) >= D_beta(i, y_i) + m_ij
        #   m_ij = DUAL_MARGIN_BASE + DUAL_MARGIN_RISK_WEIGHT * graph[y_i, j]
        # 若负类距离不够大，就产生平方 hinge 惩罚。
        margin = float(prior_cfg.DUAL_MARGIN_BASE) + float(prior_cfg.DUAL_MARGIN_RISK_WEIGHT) * risk_for_batch
        lower_violation = F.relu(true_beta_distance + margin - beta_distance)
        beta_lower_per_sample = negative_for_batch.mul(lower_violation.pow(2)).sum(dim=-1)
        beta_lower_loss = beta_lower_per_sample.mean()

        # context 弱上界:
        #   D_beta(i, pos) <= u_ij
        #   u_ij = DUAL_UPPER_BASE + DUAL_UPPER_CONTEXT_WEIGHT * (1 - graph[y_i, j])
        # 直觉是：越相似的正上下文类，上界越低；越不相似，上界越宽。
        # 该项默认权重为 0；仍保留完整计算分支，方便后续显式打开。
        beta_upper_weight = float(prior_cfg.DUAL_BETA_UPPER_WEIGHT)
        if beta_upper_weight > 0.0:
            upper_bound = float(prior_cfg.DUAL_UPPER_BASE) + float(prior_cfg.DUAL_UPPER_CONTEXT_WEIGHT) * (1.0 - risk_for_batch)
            upper_violation = F.relu(beta_distance - upper_bound)
            beta_upper_per_sample = positive_for_batch.mul(upper_violation.pow(2)).sum(dim=-1)
            beta_upper_loss = beta_upper_per_sample.mean()
        else:
            upper_violation = beta_distance.new_zeros(beta_distance.shape)
            beta_upper_loss = beta_distance.new_tensor(0.0)

        # 总 loss 只由显式权重控制，不引入隐式兜底项。
        alpha_weight = float(prior_cfg.DUAL_ALPHA_WEIGHT)
        beta_lower_weight = float(prior_cfg.DUAL_BETA_LOWER_WEIGHT)
        match_loss = (
            alpha_weight * alpha_loss
            + beta_lower_weight * beta_lower_loss
            + beta_upper_weight * beta_upper_loss
        )

        # 诊断量：这些都是未乘外层 LOSS_WEIGHT 的原始尺度，方便看 alpha/beta 哪一项主导训练。
        positive_entropy = -(positive_for_batch * positive_for_batch.clamp_min(eps).log()).sum(dim=-1).mean()
        negative_entropy = -(negative_for_batch * negative_for_batch.clamp_min(eps).log()).sum(dim=-1).mean()
        alpha_entropy = -(alpha_prob * alpha_prob.clamp_min(eps).log()).sum(dim=-1).mean()
        alpha_true_prob = alpha_prob.gather(1, targets_global[:, None]).mean()
        hard_negative_distance = beta_distance.mul(negative_for_batch).sum(dim=-1)
        negative_mask = negative_for_batch > 0.0
        positive_mask = positive_for_batch > 0.0
        beta_lower_violation_rate = (lower_violation.detach()[negative_mask] > 0.0).float().mean() if bool(negative_mask.any().item()) else beta_distance.new_tensor(0.0)
        beta_upper_violation_rate = (upper_violation.detach()[positive_mask] > 0.0).float().mean() if bool(positive_mask.any().item()) else beta_distance.new_tensor(0.0)

        stats = {
            "graph_prob_prior_match_loss": float(match_loss.detach().item()),
            "graph_prob_prior_dual_alpha_loss": float(alpha_loss.detach().item()),
            "graph_prob_prior_dual_beta_lower_loss": float(beta_lower_loss.detach().item()),
            "graph_prob_prior_dual_beta_upper_loss": float(beta_upper_loss.detach().item()),
            "graph_prob_prior_dual_alpha_entropy": float(alpha_entropy.detach().item()),
            "graph_prob_prior_dual_alpha_true_prob": float(alpha_true_prob.detach().item()),
            "graph_prob_prior_dual_alpha_distance_mean": float(alpha_distance.detach().mean().item()),
            "graph_prob_prior_dual_beta_distance_mean": float(beta_distance.detach().mean().item()),
            "graph_prob_prior_dual_beta_true_distance_mean": float(true_beta_distance.detach().mean().item()),
            "graph_prob_prior_dual_beta_hard_negative_distance_mean": float(hard_negative_distance.detach().mean().item()),
            "graph_prob_prior_dual_beta_margin_mean": float(margin.detach()[negative_mask].mean().item()) if bool(negative_mask.any().item()) else 0.0,
            "graph_prob_prior_dual_beta_lower_violation_rate": float(beta_lower_violation_rate.detach().item()),
            "graph_prob_prior_dual_beta_upper_violation_rate": float(beta_upper_violation_rate.detach().item()),
            "graph_prob_prior_dual_positive_entropy": float(positive_entropy.detach().item()),
            "graph_prob_prior_dual_negative_entropy": float(negative_entropy.detach().item()),
            "graph_prob_prior_dual_positive_top_mass": float(positive_for_batch.detach().max(dim=-1).values.mean().item()),
            "graph_prob_prior_dual_negative_top_mass": float(negative_for_batch.detach().max(dim=-1).values.mean().item()),
            "graph_prob_prior_dual_alpha_prior_mu_norm": float(alpha_prior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_dual_beta_prior_mu_norm": float(beta_prior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_dual_alpha_prior_var_mean": float(alpha_prior_logvar.detach().exp().mean().item()),
            "graph_prob_prior_dual_beta_prior_var_mean": float(beta_prior_logvar.detach().exp().mean().item()),
            "graph_prob_prior_dual_alpha_posterior_var_mean": float(alpha_logvar.detach().exp().mean().item()),
            "graph_prob_prior_dual_beta_posterior_var_mean": float(beta_logvar.detach().exp().mean().item()),
        }
        if monitor:
            stats.update({
                "graph_prob_prior_dual_alpha_target_entropy": float(
                    (-(alpha_target * alpha_target.clamp_min(eps).log()).sum(dim=-1).mean()).detach().item()
                ),
            })
            stats.update(latent_matching_monitor(alpha_distance, alpha_prob, targets_global, topk=self.monitor_topk))
            stats.update(
                posterior_prior_alignment_monitor(
                    alpha_distance,
                    alpha_prob,
                    targets_global,
                    topk=self.monitor_topk,
                )
            )
            stats.update(
                dual_sample_beta_monitor(
                    beta_distance,
                    targets_global,
                    p_neg=negative_for_batch,
                    margin=margin,
                    topk=self.monitor_topk,
                )
            )
            stats.update(
                dual_metric_distribution_monitor(
                    alpha_prior_mu,
                    alpha_prior_logvar,
                    beta_prior_mu,
                    beta_prior_logvar,
                    t_alpha=positive_weight,
                    p_pos=positive_weight,
                    p_neg=negative_weight,
                    bank=dual_priors["alpha_anchor"],
                    topk=self.monitor_topk,
                    margin=float(prior_cfg.DUAL_MARGIN_BASE),
                )
            )
            stats.update(
                residual_anchor_prior_monitor(
                    dual_priors["residual_attr"],
                    dual_priors["alpha_anchor"],
                    alpha_prior_mu,
                    delta=dual_priors["alpha_delta"],
                    context=dual_priors["alpha_context"],
                    graph=graph,
                    positive_weight=positive_weight,
                )
            )

        debug = {
            "alpha_mu_shape": tuple(alpha_mu.shape),
            "beta_mu_shape": tuple(beta_mu.shape),
            "alpha_prior_mu_shape": tuple(alpha_prior_mu.shape),
            "beta_prior_mu_shape": tuple(beta_prior_mu.shape),
            "positive_weight_shape": tuple(positive_weight.shape),
            "negative_weight_shape": tuple(negative_weight.shape),
            "alpha_target_shape": tuple(alpha_target.shape),
            "alpha_distance_shape": tuple(alpha_distance.shape),
            "beta_distance_shape": tuple(beta_distance.shape),
        }
        return match_loss, stats, debug, alpha_prior_mu, alpha_prior_logvar

    def _factorized_variation_aggregate_loss(
        self,
        variation_mu: torch.Tensor,
        variation_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        variation factor 的弱 aggregate matching。

        第一版默认权重为 0，不主动约束 variation。
        若后续打开该项，它只约束 batch 聚合分布接近弱参考 N(0,I)，
        不会像标准 KL 那样逐样本压缩类内变化。
        """
        eps = float(self.cfg.MODEL.SEMANTIC_GRAPH.OT_DELTA)
        variation_var = variation_logvar.exp()

        # 这里约束的是整个 batch 的 variation 聚合分布，而不是逐样本 variation posterior。
        # 因此它比标准 KL 更弱，保留 variation 表达类内差异的空间。
        aggregate_mu = variation_mu.mean(dim=0)
        aggregate_var = (variation_var + (variation_mu - aggregate_mu).pow(2)).mean(dim=0)

        # 弱参考 N(0,I)：聚合均值接近 0，聚合 logvar 接近 0。
        mean_loss = aggregate_mu.pow(2).mean()
        var_loss = aggregate_var.clamp_min(eps).log().pow(2).mean()
        return mean_loss + var_loss

    @staticmethod
    def _factorized_decouple_loss(
        semantic_mu: torch.Tensor,
        variation_mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        semantic / variation posterior center 的 batch 去相关约束。

        先对 batch 内两个 factor 的中心向量去均值，再计算 cross-covariance。
        Frobenius norm 越小，说明 semantic factor 与 variation factor 线性相关越弱。
        """
        # 只使用 posterior center 做去相关，不处理 logvar 或采样后的 prompt。
        semantic_centered = semantic_mu - semantic_mu.mean(dim=0, keepdim=True)
        variation_centered = variation_mu - variation_mu.mean(dim=0, keepdim=True)

        # cross-covariance 形状 [semantic_dim, variation_dim]。
        covariance = semantic_centered.t().matmul(variation_centered) / float(semantic_mu.shape[0])
        return covariance.pow(2).mean()

    def _factorized_latent_loss(
        self,
        semantic_mu: torch.Tensor,
        semantic_logvar: torch.Tensor,
        variation_mu: torch.Tensor,
        variation_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        target: torch.Tensor,
        targets_global: Optional[torch.Tensor] = None,
        monitor: bool = False,
    ):
        """
        第三种 GraphProbPrior 模式：factorized latent。

        该模式只把 semantic graph prior 作用到 semantic factor：
            q_s(z_s|x) -> p_s(z_s|c)
        variation factor 默认不做逐样本 KL，只保留两个默认关闭的接口：
        - FACTORIZED_VARIATION_WEIGHT: batch 聚合 variation 到弱 N(0,I)；
        - FACTORIZED_DECOUPLE_WEIGHT: semantic/variation posterior center 去相关。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR

        # semantic factor 仍复用逐样本 latent matching：
        # q_s(z_s|x) 与所有 semantic class prior p_s(z_s|c) 计算 KL，再对齐语义 target。
        semantic_loss, semantic_stats, debug = self._samplewise_latent_matching_loss(
            semantic_mu,
            semantic_logvar,
            prior_mu,
            prior_logvar,
            target,
            targets_global=targets_global,
            monitor=monitor,
        )
        variation_weight = float(prior_cfg.FACTORIZED_VARIATION_WEIGHT)
        decouple_weight = float(prior_cfg.FACTORIZED_DECOUPLE_WEIGHT)

        # variation / decouple 默认权重为 0，仅在用户显式打开时参与总 loss。
        if variation_weight > 0.0:
            variation_loss = self._factorized_variation_aggregate_loss(variation_mu, variation_logvar)
        else:
            variation_loss = semantic_mu.new_tensor(0.0)
        if decouple_weight > 0.0:
            decouple_loss = self._factorized_decouple_loss(semantic_mu, variation_mu)
        else:
            decouple_loss = semantic_mu.new_tensor(0.0)

        # factorized 总损失 = semantic 主损失 + 两个可选弱正则。
        match_loss = semantic_loss + variation_weight * variation_loss + decouple_weight * decouple_loss
        stats = {
            **semantic_stats,
            "graph_prob_prior_factorized_semantic_loss": float(semantic_loss.detach().item()),
            "graph_prob_prior_factorized_variation_loss": float(variation_loss.detach().item()),
            "graph_prob_prior_factorized_decouple_loss": float(decouple_loss.detach().item()),
            "graph_prob_prior_factorized_variation_mu_norm": float(variation_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_factorized_variation_var_mean": float(variation_logvar.detach().exp().mean().item()),
        }
        if monitor:
            stats.update(
                factorized_health_monitor(
                    semantic_mu,
                    semantic_logvar,
                    variation_mu,
                    variation_logvar,
                    targets_global=targets_global,
                )
            )
        debug.update(
            {
                "semantic_mu_shape": tuple(semantic_mu.shape),
                "variation_mu_shape": tuple(variation_mu.shape),
            }
        )
        return match_loss, stats, debug

    def forward(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        class_attributes: Optional[torch.Tensor] = None,
        attr_name_embeddings: Optional[torch.Tensor] = None,
        variation_mu: Optional[torch.Tensor] = None,
        variation_logvar: Optional[torch.Tensor] = None,
        seen_class_ids=None,
        unseen_class_ids=None,
    ) -> torch.Tensor:
        """
        计算 GraphProbPrior loss。

        输入 posterior_mu/logvar 来自 prompt distributor runtime stats；
        语义图资源仍由 trainer 通过 loss kwargs 显式传入。
        """
        # factorized_latent 下 posterior_mu/logvar 应该是 semantic factor 维度；
        # 其他 mode 下则使用完整 text_dim=768。
        # dual_metric_semantic_distribution 的 posterior_mu/logvar 对应 alpha/context factor，
        # 因此和 factorized_latent 一样检查 semantic_dim，而不是完整 768 维。
        expected_dim = self.factorized_semantic_dim if self.mode in {"factorized_latent", "dual_metric_semantic_distribution"} else self.text_dim
        if posterior_mu.dim() != 2 or posterior_mu.shape[1] != expected_dim:
            raise RuntimeError(f"GraphProbPrior expects posterior mu [B,{expected_dim}], got {tuple(posterior_mu.shape)}.")
        if tuple(posterior_logvar.shape) != tuple(posterior_mu.shape):
            raise RuntimeError(
                "GraphProbPrior expects posterior logvar shape {}, got {}.".format(
                    tuple(posterior_mu.shape),
                    tuple(posterior_logvar.shape),
                )
            )
        if self.mode in {"factorized_latent", "dual_metric_semantic_distribution"}:
            # factorized_latent 还需要 variation factor，供可选 variation/decouple 正则使用。
            if variation_mu is None or variation_logvar is None:
                raise RuntimeError(f"GraphProbPrior {self.mode} requires variation_mu and variation_logvar.")
            if variation_mu.dim() != 2 or variation_mu.shape[1] != self.factorized_variation_dim:
                raise RuntimeError(
                    "GraphProbPrior expects variation_mu [B,{}], got {}.".format(
                        self.factorized_variation_dim,
                        tuple(variation_mu.shape),
                    )
                )
            if tuple(variation_logvar.shape) != tuple(variation_mu.shape):
                raise RuntimeError(
                    "GraphProbPrior expects variation_logvar shape {}, got {}.".format(
                        tuple(variation_mu.shape),
                        tuple(variation_logvar.shape),
                    )
                )

        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        self._monitor_step += 1
        monitor_active = self.monitor_enable and (self._monitor_step % self.monitor_every_n == 0)

        # 统一构造语义图输入：
        #   bank:   [C,768] 类别语义原型；
        #   graph:  [C,C] 全类关系图；
        #   target: [B,C] 当前 batch 的语义监督分布。
        graph_inputs = self.graph_builder.prepare(
            targets_global=targets_global,
            class_attributes=class_attributes,
            attr_name_embeddings=attr_name_embeddings,
            device=posterior_mu.device,
            dtype=posterior_mu.dtype,
        )
        targets_global = graph_inputs["targets_global"]
        class_attributes = graph_inputs["class_attributes"]
        acc = graph_inputs["acc"]
        graph = graph_inputs["graph"]
        bank = graph_inputs["bank"]
        target = graph_inputs["target"]
        monitor_stats: Dict[str, float] = {}

        # 根据当前 mode 生成完整 768 维 prior 或 factorized semantic 维 prior。
        # 普通五个 mode 使用 _class_priors()，内部再按 PRIOR_MEAN_MODE 走 learned 或 residual_anchor。
        # dual_metric_semantic_distribution 不调用普通 _class_priors；它在自己的 loss 中生成 alpha/beta prior。
        if self.mode == "dual_metric_semantic_distribution":
            prior_mu = posterior_mu.new_zeros((self.num_classes, self.factorized_semantic_dim))
            prior_logvar = posterior_mu.new_zeros((self.num_classes, self.factorized_semantic_dim))
        else:
            prior_mu, prior_logvar = self._class_priors(
                bank,
                graph,
                class_attributes,
                factorized=(self.mode == "factorized_latent"),
            )
        if monitor_active:
            graph_cfg = self.cfg.MODEL.SEMANTIC_GRAPH
            monitor_stats.update(graph_neighbor_monitor(graph, float(prior_cfg.TAU_GRAPH), topk=self.monitor_topk))
            monitor_stats.update(graph_health_monitor(graph, float(prior_cfg.TAU_GRAPH), topk=self.monitor_topk))
            monitor_stats.update(
                false_high_pair_monitor(
                    acc,
                    graph,
                    threshold=0.9,
                    prefix="graph_prob_prior_monitor_false_high_graph_relation",
                )
            )
            if self.mode in {"graph_conditioned_semantic_prior", "factorized_latent"} or (
                self.monitor_inactive and self.mode != "dual_metric_semantic_distribution"
            ):
                monitor_stats.update(
                    semantic_target_monitor(
                        graph,
                        targets_global,
                        tau_acc=float(graph_cfg.TAU_ACC),
                        graph_topk=int(graph_cfg.TOPK),
                        target_mix_alpha=float(graph_cfg.TARGET_MIX_ALPHA),
                        num_classes=int(graph_cfg.NUM_CLASSES),
                        eps=float(graph_cfg.OT_DELTA),
                        topk=self.monitor_topk,
                    )
                )
            if self.prior_mean_mode == "residual_anchor" and self.mode != "dual_metric_semantic_distribution":
                prior_debug = self._last_prior_debug
                monitor_stats.update(
                    residual_anchor_prior_monitor(
                        prior_debug["residual_attr"],
                        prior_debug["anchor"],
                        prior_mu,
                        delta=prior_debug["delta"],
                        context=prior_debug["context"],
                        graph=graph,
                        positive_weight=prior_debug["positive_weight"],
                    )
                )

        # 按 MODE 分派到具体 matching 目标。
        if self.mode == "true_class_kl":
            match_loss, match_stats, debug_info = self._true_class_kl_loss(
                posterior_mu,
                posterior_logvar,
                targets_global,
                prior_mu,
                prior_logvar,
                monitor=monitor_active,
            )
        elif self.mode == "graph_conditioned_semantic_prior":
            match_loss, match_stats, debug_info = self._samplewise_latent_matching_loss(
                posterior_mu,
                posterior_logvar,
                prior_mu,
                prior_logvar,
                target,
                targets_global=targets_global,
                monitor=monitor_active,
            )
        elif self.mode == "class_aggregate_moment":
            match_loss, match_stats, debug_info = self._class_aggregate_moment_loss(
                posterior_mu,
                posterior_logvar,
                targets_global,
                prior_mu,
                prior_logvar,
                monitor=monitor_active,
            )
        elif self.mode == "class_aggregate_mmd":
            match_loss, match_stats, debug_info = self._class_aggregate_mmd_loss(
                posterior_mu,
                posterior_logvar,
                targets_global,
                prior_mu,
                prior_logvar,
                monitor=monitor_active,
            )
        elif self.mode == "factorized_latent":
            match_loss, match_stats, debug_info = self._factorized_latent_loss(
                posterior_mu,
                posterior_logvar,
                variation_mu,
                variation_logvar,
                prior_mu,
                prior_logvar,
                target,
                targets_global=targets_global,
                monitor=monitor_active,
            )
        elif self.mode == "dual_metric_semantic_distribution":
            # dual_metric 把 posterior_mu/logvar 当作 alpha/context posterior，
            # 把 variation_mu/logvar 当作 beta/separation posterior。
            # 该分支内部生成 alpha/beta 两套 residual-anchor prior，不使用上面的普通 _class_priors() 输出。
            match_loss, match_stats, debug_info, prior_mu, prior_logvar = self._dual_metric_semantic_distribution_loss(
                posterior_mu,
                posterior_logvar,
                variation_mu,
                variation_logvar,
                targets_global,
                class_attributes,
                graph,
                monitor=monitor_active,
            )
        else:
            raise ValueError(
                "MODEL.GRAPH_PROB_PRIOR.MODE must be true_class_kl / "
                "graph_conditioned_semantic_prior / class_aggregate_moment / "
                "class_aggregate_mmd / factorized_latent / dual_metric_semantic_distribution."
            )

        rel_weight = float(prior_cfg.REL_WEIGHT)
        class_aggregate_mode = self.mode in {"class_aggregate_moment", "class_aggregate_mmd"}
        if rel_weight > 0.0 and class_aggregate_mode:
            rel_loss, rel_monitor_stats = self._relation_regularization(
                graph,
                prior_mu,
                prior_logvar,
                monitor=monitor_active,
            )
            monitor_stats.update(rel_monitor_stats)
            loss = match_loss + rel_weight * rel_loss
        else:
            rel_loss = posterior_mu.new_tensor(0.0)
            loss = match_loss
            if monitor_active and self.monitor_inactive and self.mode != "dual_metric_semantic_distribution":
                _, rel_monitor_stats = self._relation_regularization(
                    graph,
                    prior_mu,
                    prior_logvar,
                    monitor=True,
                )
                monitor_stats.update(rel_monitor_stats)

        if bool(prior_cfg.GEOM_LOSS_ENABLE):
            geometry_loss, geometry_stats = self.compute_graph_prior_geometry_loss(prior_mu, prior_logvar, graph)
            monitor_stats.update(geometry_stats)
            loss = loss + float(prior_cfg.GEOM_LOSS_WEIGHT) * geometry_loss
        else:
            geometry_loss = posterior_mu.new_tensor(0.0)
            if monitor_active and self.mode != "dual_metric_semantic_distribution":
                monitor_stats.update(self._graph_prior_geometry_stats(prior_mu, prior_logvar, graph))

        if monitor_active and self.monitor_inactive and self.mode not in {
            "graph_conditioned_semantic_prior",
            "factorized_latent",
            "dual_metric_semantic_distribution",
        }:
            inactive_distance = self._gaussian_kl_all_classes(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
            inactive_latent_prob = F.softmax(-inactive_distance / float(prior_cfg.TAU_LATENT), dim=-1)
            monitor_stats.update(
                latent_matching_monitor(
                    inactive_distance,
                    inactive_latent_prob,
                    targets_global,
                    topk=self.monitor_topk,
                )
            )

        posterior_var = posterior_logvar.exp()
        prior_var = prior_logvar.exp()
        if monitor_active:
            monitor_stats.update(
                prior_health_monitor(
                    prior_mu,
                    prior_logvar,
                    logvar_min=self.logvar_min,
                    logvar_max=self.logvar_max,
                    topk=self.monitor_topk,
                )
            )
            monitor_stats.update(
                seen_unseen_prior_monitor(
                    prior_mu,
                    prior_logvar,
                    seen_class_ids,
                    unseen_class_ids,
                )
            )
            prior_relation = F.normalize(prior_mu.detach(), p=2, dim=-1, eps=1e-12).matmul(
                F.normalize(prior_mu.detach(), p=2, dim=-1, eps=1e-12).t()
            )
            monitor_stats.update(
                gzsl_prior_risk_monitor(
                    prior_relation,
                    seen_class_ids,
                    unseen_class_ids,
                    topk=self.monitor_topk,
                    prefix="graph_prob_prior_monitor_prior_gzsl",
                )
            )
            monitor_stats.update(
                gzsl_prior_risk_monitor(
                    graph,
                    seen_class_ids,
                    unseen_class_ids,
                    topk=self.monitor_topk,
                    prefix="graph_prob_prior_monitor_graph_gzsl",
                )
            )
            monitor_stats.update(
                false_high_pair_monitor(
                    acc,
                    prior_relation,
                    threshold=0.9,
                    prefix="graph_prob_prior_monitor_false_high_prior_relation",
                )
            )

        # 记录诊断项，供 GraphProbPriorAuxLoss / CompositeLoss 合并到 trainer 日志。
        self._last_loss_stats = {
            **match_stats,
            **monitor_stats,
            "graph_prob_prior_rel_loss": float(rel_loss.detach().item()),
            "graph_prob_prior_rel_enabled": float(rel_weight > 0.0 and class_aggregate_mode),
            "graph_prob_prior_geom_loss": float(geometry_loss.detach().item()),
            "graph_prob_prior_geom_enabled": float(bool(prior_cfg.GEOM_LOSS_ENABLE)),
            "graph_prob_prior_posterior_mu_norm": float(posterior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_prior_mu_norm": float(prior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_posterior_var_mean": float(posterior_var.detach().mean().item()),
            "graph_prob_prior_prior_var_mean": float(prior_var.detach().mean().item()),
        }

        # DEBUG=True 时只打印一次关键 shape 和 loss，方便核对 mode 分支与张量维度。
        if bool(prior_cfg.DEBUG) and not self._debug_logged:
            print(
                "[GRAPH-PROB-PRIOR-DEBUG] mode={} posterior_mu={} posterior_logvar={} bank={} graph={} "
                "prior_mu={} prior_logvar={} target={} debug={} loss={:.6f}".format(
                    self.mode,
                    tuple(posterior_mu.shape),
                    tuple(posterior_logvar.shape),
                    tuple(bank.shape),
                    tuple(graph.shape),
                    tuple(prior_mu.shape),
                    tuple(prior_logvar.shape),
                    tuple(target.shape),
                    debug_info,
                    float(loss.detach().item()),
                )
            )
            self._debug_logged = True
        return loss
