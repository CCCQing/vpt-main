#!/usr/bin/env python3
"""GraphProbPrior 的图输入构造与先验匹配损失。

本模块保留当前 GraphProbPrior 主线需要的共享图输入：
1. 用类别属性置信度构造 Acc，即类别-类别属性相似图；
2. 用属性名文本 embedding 构造 Acssc，即属性语义增强后的类别图；
3. 支持外部矩阵作为 graph 输入；
4. 根据 graph 构造 target，并进一步生成类别 Gaussian prior。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .graph_prob_prior_monitors import (
    factorized_health_monitor,
    false_high_pair_monitor,
    graph_gp_prototype_monitor,
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


class GraphPriorInputBuilder:
    """
    语义图共享构造器。

    这个类负责把 dataloader/trainer 传入的语义资源转成 GraphProbPrior 输入：
    - bank:  A_conf @ E_attr，形状 [C, D]，graph_gp_conditioned 下不构造；
    - graph: Acc / Acssc / fuse / external 得到的全类关系图，形状 [C, C]；
    - target: 从 graph[y] 取 top-k 并混合 one-hot 后得到的监督分布，形状 [B, C]。

    GraphProbPriorLossComputer 复用这里生成 graph / target 和非 Graph-GP 分支的 bank，
    避免图输入构造逻辑散落在多个分支中。
    """

    GRAPH_GP_EXTERNAL_GRAPH_PATH = (
        "cub_attribute_localization/05_hparam_searches/diff_only_graphs_v1/diff_only_method_matrices_v1.npz"
    )
    GRAPH_GP_GRAPH_SOURCES = ("method1_diff", "method2_diff", "method3_diff")
    GRAPH_GP_DEFAULT_GRAPH_SOURCE = "method1_diff"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        graph_cfg = cfg.MODEL.GRAPH_INPUT
        # C=全局类别数，attr_dim=属性维度，text_dim=属性名 embedding / prompt latent 维度。
        self.num_classes = int(graph_cfg.NUM_CLASSES)
        self.attr_dim = int(graph_cfg.ATTR_DIM)
        self.text_dim = int(graph_cfg.TEXT_DIM)
        self._external_graph_cache = None
        self._external_graph_cache_id = None

    def _is_graph_gp_conditioned(self) -> bool:
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        return str(prior_cfg.PRIOR_MEAN_MODE).lower() == "graph_gp_conditioned"

    def prepare_class_attributes(
        self,
        class_attributes: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """校验并搬运全类属性矩阵 A_conf；该矩阵必须由 trainer/dataset 显式传入。"""
        if class_attributes is None:
            raise RuntimeError("Graph prior input requires dataset.class_attributes from dataloader.")
        if not torch.is_tensor(class_attributes):
            class_attributes = torch.as_tensor(class_attributes)
        # class_attributes 通常来自 XLSA/CUB 的 att_splits.mat::att，形状应为 [C, attr_dim]。
        class_attributes = class_attributes.to(device=device, dtype=dtype)
        if tuple(class_attributes.shape) != (self.num_classes, self.attr_dim):
            raise RuntimeError(
                "Graph prior class attributes must be [{},{}], got {}.".format(
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
            raise RuntimeError("Graph prior input requires attr_name_embeddings from trainer or dataset.")
        attr_name_embeddings = attr_name_embeddings.to(device=device, dtype=dtype)
        if tuple(attr_name_embeddings.shape) != (self.attr_dim, self.text_dim):
            raise RuntimeError(
                "Graph prior attr name embeddings must be [{},{}], got {}.".format(
                    self.attr_dim,
                    self.text_dim,
                    tuple(attr_name_embeddings.shape),
                )
            )
        return attr_name_embeddings

    def _load_external_graph(
        self,
        device: torch.device,
        dtype: torch.dtype,
        key: str,
        default_path: Optional[str] = None,
        default_key: Optional[str] = None,
        allowed_keys: Optional[tuple] = None,
    ) -> torch.Tensor:
        graph_cfg = self.cfg.MODEL.GRAPH_INPUT
        raw_path = str(graph_cfg.EXTERNAL_GRAPH_PATH).strip()
        key = str(key).strip()
        if default_path is not None and not raw_path:
            raw_path = str(default_path)
        if default_key is not None and (not key or key == "graph"):
            key = str(default_key)
        if key == "external":
            key = "graph"
        if not raw_path:
            raise ValueError("MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_PATH must be set when GRAPH_SOURCE uses an external graph.")
        if not key:
            raise ValueError("External semantic graph key is empty.")
        if allowed_keys is not None and key not in allowed_keys:
            raise ValueError(
                "Graph-GP external graph key must be one of {}, got '{}'.".format(
                    allowed_keys,
                    key,
                )
            )

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
        attr_embeddings: Optional[torch.Tensor],
        graph_gp_conditioned: bool = False,
    ):
        """
        构造语义 bank 和全类语义图；graph_gp_conditioned 只构造 acc 和 external graph。

        - Acc   = norm(A_conf) @ norm(A_conf)^T，只看属性置信度相似；
        - A_sem = A_conf @ E_attr，用属性置信度加权属性名文本 embedding；Graph-GP 下跳过；
        - Acssc = norm(A_sem) @ norm(A_sem)^T，看文本语义增强后的类别相似；Graph-GP 下跳过；
        - graph = external / acc / acssc / rho 融合图。
        """
        graph_cfg = self.cfg.MODEL.GRAPH_INPUT
        graph_source = str(graph_cfg.GRAPH_SOURCE).lower()
        # Acc 是纯属性置信度图：先行归一化，再做类别间余弦式相似度。
        acc = _row_normalize(class_attributes).matmul(_row_normalize(class_attributes).t())

        if graph_gp_conditioned:
            graph_key = graph_source if graph_source in self.GRAPH_GP_GRAPH_SOURCES else self.GRAPH_GP_DEFAULT_GRAPH_SOURCE
            graph = self._load_external_graph(
                class_attributes.device,
                class_attributes.dtype,
                key=graph_key,
                default_path=self.GRAPH_GP_EXTERNAL_GRAPH_PATH,
                allowed_keys=self.GRAPH_GP_GRAPH_SOURCES,
            )
            return acc.detach(), None, graph.detach(), None

        if attr_embeddings is None:
            raise RuntimeError("Graph prior input requires attr_name_embeddings outside graph_gp_conditioned.")

        # bank / A_sem 是每个类别的文本语义原型：
        # 用该类的属性置信度加权所有属性名 embedding，得到 [C, text_dim]。
        bank = class_attributes.matmul(attr_embeddings)

        # Acssc 是 bank 上的类别关系图，仍然使用行归一化后的相似度。
        acssc = _row_normalize(bank).matmul(_row_normalize(bank).t())

        # GRAPH_SOURCE 决定最终使用哪张图：
        #   acc   -> 只用属性置信度图；
        #   acssc -> 只用属性文本语义图；
        #   fuse  -> 按 rho 融合二者。
        if graph_source == "acc":
            graph = acc
        elif graph_source == "acssc":
            graph = acssc
        elif graph_source == "fuse":
            rho = float(graph_cfg.RHO)
            if rho < 0.0 or rho > 1.0:
                raise ValueError("MODEL.GRAPH_INPUT.RHO must be in [0, 1].")
            graph = rho * acc + (1.0 - rho) * acssc
        elif graph_source == "external":
            graph = self._load_external_graph(class_attributes.device, class_attributes.dtype, key=graph_source)
        elif graph_source in self.GRAPH_GP_GRAPH_SOURCES:
            graph = self._load_external_graph(
                class_attributes.device,
                class_attributes.dtype,
                key=graph_source,
                default_path=self.GRAPH_GP_EXTERNAL_GRAPH_PATH,
            )
        else:
            if str(graph_cfg.EXTERNAL_GRAPH_PATH).strip():
                graph = self._load_external_graph(class_attributes.device, class_attributes.dtype, key=graph_source)
            else:
                raise ValueError(
                    "MODEL.GRAPH_INPUT.GRAPH_SOURCE must be acc / acssc / fuse / external / "
                    "method1_diff / method2_diff / method3_diff, or an external matrix key with "
                    "MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_PATH set."
                )
        return acc.detach(), acssc.detach(), graph.detach(), bank.detach()

    def validate_targets(self, targets_global: torch.Tensor, device: torch.device) -> torch.Tensor:
        """校验 GraphProbPrior 使用的是全局类别 id，而不是 local-output remap 后的局部 id。"""
        if not torch.is_tensor(targets_global):
            raise RuntimeError("GraphProbPrior requires tensor targets_global.")
        targets_global = targets_global.to(device=device, dtype=torch.long)
        if targets_global.dim() != 1:
            raise RuntimeError(f"targets_global must be [B], got {tuple(targets_global.shape)}.")
        if targets_global.numel() == 0:
            raise RuntimeError("targets_global is empty.")
        if int(targets_global.min().item()) < 0 or int(targets_global.max().item()) >= self.num_classes:
            raise RuntimeError(
                "GraphProbPrior requires global targets in [0,{}], got min={} max={}.".format(
                    self.num_classes - 1,
                    int(targets_global.min().item()),
                    int(targets_global.max().item()),
                )
            )
        return targets_global

    def build_target(self, graph: torch.Tensor, targets_global: torch.Tensor) -> torch.Tensor:
        """
        从图输入中为 batch 构造全类 target T。

        每个样本取 graph[y]，排除自身类后仅保留 top-k 类别做 softmax，再与 one-hot(y) 混合。
        该 target 服务 GraphProbPrior 中的 samplewise latent matching / true-class KL 相关分支。
        """
        graph_cfg = self.cfg.MODEL.GRAPH_INPUT
        eps = float(graph_cfg.EPS)

        # rows 是当前 batch 每个真实类别 y 在全类语义图里的那一行，形状 [B, C]。
        rows = graph.index_select(0, targets_global)
        topk = int(graph_cfg.TOPK)
        if topk <= 0 or topk > self.num_classes - 1:
            raise ValueError("MODEL.GRAPH_INPUT.TOPK must be in [1, NUM_CLASSES-1].")

        # 只让每个样本的非自身 top-k 语义邻居进入 soft target，其余类别置为 -inf。
        candidate_rows = rows.clone()
        row_idx = torch.arange(candidate_rows.shape[0], device=candidate_rows.device)
        candidate_rows[row_idx, targets_global] = float("-inf")
        values, indices = torch.topk(candidate_rows, k=topk, dim=-1)
        masked = torch.full_like(rows, float("-inf"))
        masked.scatter_(1, indices, values)

        # TAU_ACC 控制语义邻居分布的尖锐程度；这里只构造监督 target，不是可学习预测。
        target_sem = F.softmax(masked / float(graph_cfg.TAU_ACC), dim=-1)

        alpha = float(graph_cfg.TARGET_MIX_ALPHA)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("MODEL.GRAPH_INPUT.TARGET_MIX_ALPHA must be in [0, 1].")

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
    ) -> Dict[str, Optional[torch.Tensor]]:
        """一次性准备 GraphProbPrior 所需的所有图输入张量。"""
        # prepare 是 loss 侧唯一需要调用的入口：
        # 先校验 batch targets 和语义资源，再统一返回 A/G/T 相关张量。
        targets_global = self.validate_targets(targets_global, device)
        graph_gp_conditioned = self._is_graph_gp_conditioned()
        class_attributes = self.prepare_class_attributes(class_attributes, device, dtype)
        if graph_gp_conditioned:
            attr_name_embeddings = None
        else:
            attr_name_embeddings = self.prepare_attr_name_embeddings(attr_name_embeddings, device, dtype)
        acc, acssc, graph, bank = self.build_graphs(
            class_attributes,
            attr_name_embeddings,
            graph_gp_conditioned=graph_gp_conditioned,
        )
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
    - compute_graph_prior_geometry_loss: 可选几何正则入口，支持 soft_distribution_matching / graph_ordinal_ranking。
    - _samplewise_latent_matching_loss: 逐样本 all-class KL matching，服务 graph_conditioned_semantic_prior 和 factorized semantic loss。

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
        # 复用 GraphPriorInputBuilder，确保 GraphProbPrior 的 bank / graph / target 来自同一套图输入。
        self.graph_builder = GraphPriorInputBuilder(cfg)
        graph_cfg = cfg.MODEL.GRAPH_INPUT
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
            "graph_conditioned_semantic_prior",
            "class_aggregate_mmd",
            "factorized_latent",
        }
        if self.mode not in self.supported_modes:
            raise ValueError(
                "MODEL.GRAPH_PROB_PRIOR.MODE must be graph_conditioned_semantic_prior / "
                "class_aggregate_mmd / factorized_latent."
            )
        if self.prior_mean_mode not in {"learned", "residual_anchor", "graph_gp_conditioned"}:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_MEAN_MODE must be learned / residual_anchor / graph_gp_conditioned.")
        if self.prior_var_mode not in {"learned", "unit", "constant"}:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_VAR_MODE must be learned / unit / constant.")
        if self.prior_mean_mode == "graph_gp_conditioned":
            if self.mode == "factorized_latent":
                raise ValueError(
                    "PRIOR_MEAN_MODE=graph_gp_conditioned first version supports only full 768-d posterior modes, "
                    "not factorized_latent."
                )
            if self.prior_var_mode == "learned":
                raise ValueError("PRIOR_MEAN_MODE=graph_gp_conditioned does not support PRIOR_VAR_MODE=learned in first version.")

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
        if int(prior_cfg.MMD_SAMPLES) <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MMD_SAMPLES must be positive.")
        if float(prior_cfg.MMD_SIGMA) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MMD_SIGMA must be positive.")
        if float(prior_cfg.RESIDUAL_SIGMA_MIN) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.RESIDUAL_SIGMA_MIN must be positive.")
        if float(prior_cfg.RESIDUAL_CLIP) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.RESIDUAL_CLIP must be positive.")
        if int(graph_cfg.TOPK) <= 0 or int(graph_cfg.TOPK) > self.num_classes - 1:
            raise ValueError("MODEL.GRAPH_INPUT.TOPK must be in [1, NUM_CLASSES-1].")
        if float(prior_cfg.PRIOR_DELTA_SCALE) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_DELTA_SCALE must be non-negative.")
        if float(prior_cfg.PRIOR_MU_SCALE) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_MU_SCALE must be positive.")
        self.learn_prior_mu_scale = bool(prior_cfg.LEARN_PRIOR_MU_SCALE)
        self.learn_prior_delta_scale = bool(prior_cfg.LEARN_PRIOR_DELTA_SCALE)
        self.learn_tau_graph = bool(prior_cfg.LEARN_TAU_GRAPH)
        self.learn_tau_latent = bool(prior_cfg.LEARN_TAU_LATENT)
        self.learn_geom_tau_dist = bool(prior_cfg.LEARN_GEOM_TAU_DIST)
        self.learn_geom_bound_weight = bool(prior_cfg.LEARN_GEOM_BOUND_WEIGHT)
        self.learn_geom_ord_margin_scale = bool(prior_cfg.LEARN_GEOM_ORD_MARGIN_SCALE)
        self.learn_geom_ord_non_overlap_weight = bool(prior_cfg.LEARN_GEOM_ORD_NON_OVERLAP_WEIGHT)
        self._validate_bounded_scalar(
            "PRIOR_MU_SCALE",
            value=float(prior_cfg.PRIOR_MU_SCALE),
            min_value=float(prior_cfg.PRIOR_MU_SCALE_MIN),
            max_value=float(prior_cfg.PRIOR_MU_SCALE_MAX),
            learnable=self.learn_prior_mu_scale,
        )
        self._validate_bounded_scalar(
            "PRIOR_DELTA_SCALE",
            value=float(prior_cfg.PRIOR_DELTA_SCALE),
            min_value=float(prior_cfg.PRIOR_DELTA_SCALE_MIN),
            max_value=float(prior_cfg.PRIOR_DELTA_SCALE_MAX),
            learnable=self.learn_prior_delta_scale,
        )
        self._validate_bounded_scalar(
            "TAU_GRAPH",
            value=float(prior_cfg.TAU_GRAPH),
            min_value=float(prior_cfg.TAU_GRAPH_MIN),
            max_value=float(prior_cfg.TAU_GRAPH_MAX),
            learnable=self.learn_tau_graph,
        )
        self._validate_bounded_scalar(
            "TAU_LATENT",
            value=float(prior_cfg.TAU_LATENT),
            min_value=float(prior_cfg.TAU_LATENT_MIN),
            max_value=float(prior_cfg.TAU_LATENT_MAX),
            learnable=self.learn_tau_latent,
        )
        self._validate_bounded_scalar(
            "GEOM_TAU_DIST",
            value=float(prior_cfg.GEOM_TAU_DIST),
            min_value=float(prior_cfg.GEOM_TAU_DIST_MIN),
            max_value=float(prior_cfg.GEOM_TAU_DIST_MAX),
            learnable=self.learn_geom_tau_dist,
        )
        self._validate_bounded_scalar(
            "GEOM_BOUND_WEIGHT",
            value=float(prior_cfg.GEOM_BOUND_WEIGHT),
            min_value=float(prior_cfg.GEOM_BOUND_WEIGHT_MIN),
            max_value=float(prior_cfg.GEOM_BOUND_WEIGHT_MAX),
            learnable=self.learn_geom_bound_weight,
        )
        self._validate_bounded_scalar(
            "GEOM_ORD_MARGIN_SCALE",
            value=float(prior_cfg.GEOM_ORD_MARGIN_SCALE),
            min_value=float(prior_cfg.GEOM_ORD_MARGIN_SCALE_MIN),
            max_value=float(prior_cfg.GEOM_ORD_MARGIN_SCALE_MAX),
            learnable=self.learn_geom_ord_margin_scale,
        )
        self._validate_bounded_scalar(
            "GEOM_ORD_NON_OVERLAP_WEIGHT",
            value=float(prior_cfg.GEOM_ORD_NON_OVERLAP_WEIGHT),
            min_value=float(prior_cfg.GEOM_ORD_NON_OVERLAP_WEIGHT_MIN),
            max_value=float(prior_cfg.GEOM_ORD_NON_OVERLAP_WEIGHT_MAX),
            learnable=self.learn_geom_ord_non_overlap_weight,
        )
        if (self.learn_prior_mu_scale or self.learn_prior_delta_scale) and self.prior_mean_mode != "residual_anchor":
            raise ValueError(
                "MODEL.GRAPH_PROB_PRIOR.LEARN_PRIOR_*_SCALE currently requires PRIOR_MEAN_MODE=residual_anchor."
            )
        if str(prior_cfg.PRIOR_RADIUS_MODE).lower() not in {"fixed", "residual_norm"}:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.PRIOR_RADIUS_MODE must be fixed / residual_norm.")
        if self.prior_mean_mode == "graph_gp_conditioned":
            if not (0.0 < float(prior_cfg.GRAPH_GP_SUPPORT_RATIO) < 1.0):
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SUPPORT_RATIO must be in (0, 1).")
            if int(prior_cfg.GRAPH_GP_SPLIT_EVERY_EPOCH) <= 0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SPLIT_EVERY_EPOCH must be positive.")
            if str(prior_cfg.GRAPH_GP_CENTER_SOURCE).lower() != "posterior_mu":
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_CENTER_SOURCE first version supports only posterior_mu.")
            if not bool(prior_cfg.GRAPH_GP_DETACH_CENTERS):
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_DETACH_CENTERS must be True in first version.")
            if str(prior_cfg.GRAPH_GP_OBS_NOISE_MODE).lower() not in {"constant", "class_var_over_count"}:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MODE must be constant / class_var_over_count.")
            if float(prior_cfg.GRAPH_GP_OBS_NOISE_CONST) <= 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_CONST must be positive.")
            if float(prior_cfg.GRAPH_GP_OBS_NOISE_MIN) <= 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MIN must be positive.")
            if float(prior_cfg.GRAPH_GP_OBS_NOISE_MAX) < float(prior_cfg.GRAPH_GP_OBS_NOISE_MIN):
                raise ValueError("GRAPH_GP_OBS_NOISE_MAX must be >= GRAPH_GP_OBS_NOISE_MIN.")
            if float(prior_cfg.GRAPH_GP_RIDGE) <= 0.0:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_RIDGE must be positive.")
            if str(prior_cfg.GRAPH_GP_KERNEL_NORMALIZE).lower() not in {"diag", "none"}:
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_NORMALIZE must be diag / none.")
            if str(prior_cfg.GRAPH_GP_PRIOR_VAR_SOURCE).lower() not in {
                "unit",
                "constant",
                "current_prior_var_mode",
                "dynamic_uncertainty",
            }:
                raise ValueError(
                    "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_SOURCE must be "
                    "unit / constant / current_prior_var_mode / dynamic_uncertainty."
                )
            for key in (
                "GRAPH_GP_PRIOR_VAR_FLOOR",
                "GRAPH_GP_PRIOR_VAR_MIN",
                "GRAPH_GP_PRIOR_VAR_MAX",
            ):
                if float(getattr(prior_cfg, key)) <= 0.0:
                    raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{key} must be positive.")
            if float(prior_cfg.GRAPH_GP_PRIOR_VAR_MAX) < float(prior_cfg.GRAPH_GP_PRIOR_VAR_MIN):
                raise ValueError("GRAPH_GP_PRIOR_VAR_MAX must be >= GRAPH_GP_PRIOR_VAR_MIN.")
            for key in (
                "GRAPH_GP_PRIOR_VAR_PROTO_WEIGHT",
                "GRAPH_GP_PRIOR_VAR_VISUAL_WEIGHT",
            ):
                if float(getattr(prior_cfg, key)) < 0.0:
                    raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{key} must be non-negative.")
            if not bool(prior_cfg.GRAPH_GP_MATCH_DETACH_PRIOR):
                raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_MATCH_DETACH_PRIOR must be True in first version.")
        if float(prior_cfg.GEOM_LOSS_WEIGHT) < 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_WEIGHT must be non-negative.")
        if str(prior_cfg.GEOM_LOSS_TYPE).lower() not in {
            "soft_distribution_matching",
            "graph_ordinal_ranking",
        }:
            raise ValueError(
                "MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_TYPE must be soft_distribution_matching / graph_ordinal_ranking."
            )
        if int(prior_cfg.GEOM_TOPK) <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_TOPK must be positive.")
        for key in (
            "GEOM_SIGMA_PRIOR",
            "GEOM_TAU_BARRIER",
            "GEOM_TAU_GRAPH_DIST",
            "GEOM_TAU_DIST",
            "GEOM_ORD_GRAPH_GAP_EPS",
            "GEOM_ORD_EPS",
            "GEOM_ORD_NON_OVERLAP_MIN_DIST",
        ):
            if float(getattr(prior_cfg, key)) <= 0.0:
                raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{key} must be positive.")
        if str(prior_cfg.GEOM_ORD_DISTANCE_TYPE).lower() not in {"clearance", "cosine", "euclidean"}:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_DISTANCE_TYPE must be clearance / cosine / euclidean.")
        if str(prior_cfg.GEOM_ORD_NON_OVERLAP_SCOPE).lower() not in {"topk", "all"}:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_NON_OVERLAP_SCOPE must be topk / all.")
        for key in (
            "GEOM_MARGIN_MIN",
            "GEOM_BOUND_WEIGHT",
            "GEOM_ORD_MARGIN_BASE",
            "GEOM_ORD_MARGIN_SCALE",
            "GEOM_ORD_NON_OVERLAP_WEIGHT",
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
        self.monitor_enable = bool(prior_cfg.MONITOR_ENABLE)
        self.monitor_inactive = bool(prior_cfg.MONITOR_INACTIVE)
        self.monitor_topk = int(prior_cfg.MONITOR_TOPK)
        self.monitor_effective_rank = bool(getattr(prior_cfg, "MONITOR_EFFECTIVE_RANK", False))
        self.monitor_every_n = int(prior_cfg.MONITOR_EVERY_N)
        if self.monitor_topk <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MONITOR_TOPK must be positive.")
        if self.monitor_every_n <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N must be positive.")

        if self.learn_prior_mu_scale:
            self.learnable_prior_mu_scale_raw = torch.nn.Parameter(
                torch.tensor(
                    self._bounded_scalar_init_raw(
                        float(prior_cfg.PRIOR_MU_SCALE),
                        float(prior_cfg.PRIOR_MU_SCALE_MIN),
                        float(prior_cfg.PRIOR_MU_SCALE_MAX),
                    ),
                    dtype=torch.float32,
                )
            )
        if self.learn_prior_delta_scale:
            self.learnable_prior_delta_scale_raw = torch.nn.Parameter(
                torch.tensor(
                    self._bounded_scalar_init_raw(
                        float(prior_cfg.PRIOR_DELTA_SCALE),
                        float(prior_cfg.PRIOR_DELTA_SCALE_MIN),
                        float(prior_cfg.PRIOR_DELTA_SCALE_MAX),
                    ),
                    dtype=torch.float32,
                )
            )
        bounded_specs = (
            (
                self.learn_tau_graph,
                "learnable_tau_graph_raw",
                "TAU_GRAPH",
                "TAU_GRAPH_MIN",
                "TAU_GRAPH_MAX",
            ),
            (
                self.learn_tau_latent,
                "learnable_tau_latent_raw",
                "TAU_LATENT",
                "TAU_LATENT_MIN",
                "TAU_LATENT_MAX",
            ),
            (
                self.learn_geom_tau_dist,
                "learnable_geom_tau_dist_raw",
                "GEOM_TAU_DIST",
                "GEOM_TAU_DIST_MIN",
                "GEOM_TAU_DIST_MAX",
            ),
            (
                self.learn_geom_bound_weight,
                "learnable_geom_bound_weight_raw",
                "GEOM_BOUND_WEIGHT",
                "GEOM_BOUND_WEIGHT_MIN",
                "GEOM_BOUND_WEIGHT_MAX",
            ),
            (
                self.learn_geom_ord_margin_scale,
                "learnable_geom_ord_margin_scale_raw",
                "GEOM_ORD_MARGIN_SCALE",
                "GEOM_ORD_MARGIN_SCALE_MIN",
                "GEOM_ORD_MARGIN_SCALE_MAX",
            ),
            (
                self.learn_geom_ord_non_overlap_weight,
                "learnable_geom_ord_non_overlap_weight_raw",
                "GEOM_ORD_NON_OVERLAP_WEIGHT",
                "GEOM_ORD_NON_OVERLAP_WEIGHT_MIN",
                "GEOM_ORD_NON_OVERLAP_WEIGHT_MAX",
            ),
        )
        for learnable, raw_attr, value_key, min_key, max_key in bounded_specs:
            if bool(learnable):
                setattr(
                    self,
                    raw_attr,
                    torch.nn.Parameter(
                        torch.tensor(
                            self._bounded_scalar_init_raw(
                                float(getattr(prior_cfg, value_key)),
                                float(getattr(prior_cfg, min_key)),
                                float(getattr(prior_cfg, max_key)),
                            ),
                            dtype=torch.float32,
                        )
                    ),
                )

        def make_mlp(input_dim: int, output_dim: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_dim, output_dim),
            )

        # learned 是旧路径：cat(bank, dense_neighbor_bank) 直接输出 [prior_mu, prior_logvar]。
        # residual_anchor 是新路径：312 维标准化属性残差先生成 anchor，再用 graph top-k context 做小修正。
        full_prior_mode = self.mode != "factorized_latent"
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

        self._last_loss_stats: Dict[str, float] = {}
        self._debug_logged = False
        self._monitor_step = 0
        self._last_prior_debug: Dict[str, torch.Tensor] = {}
        # Graph-GP 第一版不建立 M0，而是从 support-seen posterior center 条件推断 M_star。
        # 这些 buffer 只保存本进程见过的 support 样本统计；真正计算 prior 前会对克隆值做 DDP all_reduce，
        # 不把 all_reduce 结果写回本地 buffer，避免下一次同步时把已经同步过的全局值重复累加。
        self.register_buffer("_graph_gp_support_sum", torch.zeros(self.num_classes, self.text_dim), persistent=False)
        self.register_buffer("_graph_gp_support_sq_sum", torch.zeros(self.num_classes, self.text_dim), persistent=False)
        self.register_buffer("_graph_gp_support_var_sum", torch.zeros(self.num_classes, self.text_dim), persistent=False)
        self.register_buffer("_graph_gp_support_count", torch.zeros(self.num_classes), persistent=False)
        self._graph_gp_split_id: Optional[int] = None
        self._graph_gp_support_ids: Optional[torch.Tensor] = None
        self._graph_gp_pseudo_unseen_ids: Optional[torch.Tensor] = None
        self._last_graph_gp_debug: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _validate_bounded_scalar(
        name: str,
        value: float,
        min_value: float,
        max_value: float,
        learnable: bool,
    ) -> None:
        if min_value >= max_value:
            raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{name}_MIN must be smaller than {name}_MAX.")
        if bool(learnable) and not (min_value < value < max_value):
            raise ValueError(
                f"MODEL.GRAPH_PROB_PRIOR.{name} must be inside ({name}_MIN, {name}_MAX) when learnable."
            )

    @staticmethod
    def _bounded_scalar_init_raw(value: float, min_value: float, max_value: float) -> float:
        eps = 1e-6
        ratio = (float(value) - float(min_value)) / max(float(max_value) - float(min_value), eps)
        ratio = min(max(ratio, eps), 1.0 - eps)
        return math.log(ratio / (1.0 - ratio))

    @staticmethod
    def _bounded_scalar_value(
        raw_value: torch.Tensor,
        min_value: float,
        max_value: float,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        raw_value = raw_value.to(device=reference.device, dtype=reference.dtype)
        min_tensor = reference.new_tensor(float(min_value))
        span_tensor = reference.new_tensor(float(max_value) - float(min_value))
        return min_tensor + span_tensor * torch.sigmoid(raw_value)

    def _prior_mu_scale_value(self, reference: torch.Tensor) -> torch.Tensor:
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        if self.learn_prior_mu_scale:
            return self._bounded_scalar_value(
                self.learnable_prior_mu_scale_raw,
                float(prior_cfg.PRIOR_MU_SCALE_MIN),
                float(prior_cfg.PRIOR_MU_SCALE_MAX),
                reference,
            )
        return reference.new_tensor(float(prior_cfg.PRIOR_MU_SCALE))

    def _prior_delta_scale_value(self, reference: torch.Tensor) -> torch.Tensor:
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        if self.learn_prior_delta_scale:
            return self._bounded_scalar_value(
                self.learnable_prior_delta_scale_raw,
                float(prior_cfg.PRIOR_DELTA_SCALE_MIN),
                float(prior_cfg.PRIOR_DELTA_SCALE_MAX),
                reference,
            )
        return reference.new_tensor(float(prior_cfg.PRIOR_DELTA_SCALE))

    def _cfg_bounded_value(
        self,
        reference: torch.Tensor,
        learnable: bool,
        raw_attr: str,
        value_key: str,
        min_key: str,
        max_key: str,
    ) -> torch.Tensor:
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        if bool(learnable):
            return self._bounded_scalar_value(
                getattr(self, raw_attr),
                float(getattr(prior_cfg, min_key)),
                float(getattr(prior_cfg, max_key)),
                reference,
            )
        return reference.new_tensor(float(getattr(prior_cfg, value_key)))

    def _tau_graph_value(self, reference: torch.Tensor) -> torch.Tensor:
        return self._cfg_bounded_value(
            reference,
            self.learn_tau_graph,
            "learnable_tau_graph_raw",
            "TAU_GRAPH",
            "TAU_GRAPH_MIN",
            "TAU_GRAPH_MAX",
        )

    def _tau_latent_value(self, reference: torch.Tensor) -> torch.Tensor:
        return self._cfg_bounded_value(
            reference,
            self.learn_tau_latent,
            "learnable_tau_latent_raw",
            "TAU_LATENT",
            "TAU_LATENT_MIN",
            "TAU_LATENT_MAX",
        )

    def _geom_tau_dist_value(self, reference: torch.Tensor) -> torch.Tensor:
        return self._cfg_bounded_value(
            reference,
            self.learn_geom_tau_dist,
            "learnable_geom_tau_dist_raw",
            "GEOM_TAU_DIST",
            "GEOM_TAU_DIST_MIN",
            "GEOM_TAU_DIST_MAX",
        )

    def _geom_bound_weight_value(self, reference: torch.Tensor) -> torch.Tensor:
        return self._cfg_bounded_value(
            reference,
            self.learn_geom_bound_weight,
            "learnable_geom_bound_weight_raw",
            "GEOM_BOUND_WEIGHT",
            "GEOM_BOUND_WEIGHT_MIN",
            "GEOM_BOUND_WEIGHT_MAX",
        )

    def _geom_ord_margin_scale_value(self, reference: torch.Tensor) -> torch.Tensor:
        return self._cfg_bounded_value(
            reference,
            self.learn_geom_ord_margin_scale,
            "learnable_geom_ord_margin_scale_raw",
            "GEOM_ORD_MARGIN_SCALE",
            "GEOM_ORD_MARGIN_SCALE_MIN",
            "GEOM_ORD_MARGIN_SCALE_MAX",
        )

    def _geom_ord_non_overlap_weight_value(self, reference: torch.Tensor) -> torch.Tensor:
        return self._cfg_bounded_value(
            reference,
            self.learn_geom_ord_non_overlap_weight,
            "learnable_geom_ord_non_overlap_weight_raw",
            "GEOM_ORD_NON_OVERLAP_WEIGHT",
            "GEOM_ORD_NON_OVERLAP_WEIGHT_MIN",
            "GEOM_ORD_NON_OVERLAP_WEIGHT_MAX",
        )

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

    def _residual_radius_scale(self, residual_attr: torch.Tensor) -> torch.Tensor:
        """
        根据类别属性残差强度生成 prior_mu 半径因子。

        fixed: 旧行为，所有类别使用同一个 PRIOR_MU_SCALE 半径。
        residual_norm: 用 ||residual_attr_c|| / mean_c ||residual_attr_c|| 表示类别 c
        相对“平均鸟”的偏离强度；偏离越大，prior center 半径越大。这里不再做
        min/max 截断，让属性残差本身决定类别间半径差异。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        mode = str(prior_cfg.PRIOR_RADIUS_MODE).lower()
        if mode == "fixed":
            return residual_attr.new_ones((int(residual_attr.shape[0]), 1))

        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        strength = residual_attr.norm(p=2, dim=-1, keepdim=True)
        mean_strength = strength.mean().clamp_min(eps)
        return strength / mean_strength

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
        6. radius_scale 由 residual_attr 的 norm 产生，表示该类别偏离平均鸟的强弱；
        7. prior_mu = PRIOR_MU_SCALE * radius_scale * normalize(anchor + scale*tanh(delta))。
        """
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        residual_attr = self._standardized_residual_attributes(class_attributes)

        if bool(factorized):
            anchor = self.factorized_residual_anchor_head(residual_attr)
            delta_head = self.factorized_residual_delta_head
        else:
            anchor = self.residual_anchor_head(residual_attr)
            delta_head = self.residual_delta_head

        positive_weight = self._masked_topk_distribution(
            graph,
            topk=int(self.cfg.MODEL.GRAPH_INPUT.TOPK),
            tau=self._tau_graph_value(graph),
            exclude_self=True,
        )
        context = positive_weight.matmul(anchor)
        delta_input = torch.cat((anchor, context, anchor - context), dim=-1)
        delta = delta_head(delta_input)

        # PRIOR_DELTA_SCALE 作用在归一化之前：控制 graph-context delta 能把类别 prior
        # 从自身属性 anchor 方向上修正多远；设为 0 时 prior_mu 只由 anchor 决定方向。
        prior_delta_scale = self._prior_delta_scale_value(anchor)
        raw_mu = anchor + prior_delta_scale * torch.tanh(delta)
        # radius_scale 来自 residual_attr 的 L2 norm：越偏离平均鸟的类别，prior center
        # 半径越大；这里不做 min/max 截断，先完整保留属性残差带来的半径差异。
        radius_scale = self._residual_radius_scale(residual_attr)
        # PRIOR_MU_SCALE 作用在归一化之后：控制 prior_mu 的基础半径；radius_scale
        # 在此基础上做类别级半径调制，不改变 normalize(raw_mu) 得到的方向。
        prior_mu_scale = self._prior_mu_scale_value(anchor)
        prior_mu = prior_mu_scale * radius_scale * F.normalize(raw_mu, p=2, dim=-1, eps=eps)
        prior_logvar = self._fixed_or_learned_logvar(prior_mu, logvar_input=delta_input, factorized=factorized)

        # 仅缓存监测需要的中间张量。loss 仍从 prior_mu/prior_logvar 正常回传。
        self._last_prior_debug = {
            "residual_attr": residual_attr,
            "anchor": anchor,
            "context": context,
            "delta": delta,
            "radius_scale": radius_scale,
            "positive_weight": positive_weight,
            "prior_mu_scale_value": prior_mu_scale.detach(),
            "prior_delta_scale_value": prior_delta_scale.detach(),
        }
        return prior_mu, prior_logvar

    def _learned_class_priors(self, bank: torch.Tensor, graph: torch.Tensor, factorized: bool = False):
        """
        旧 learned prior 路径。

        该路径保留给复现实验：dense softmax(graph/tau) 聚合 neighbor_bank，
        再由 prior_head 直接输出 prior_mu/prior_logvar。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        neighbor_weight = F.softmax(graph / self._tau_graph_value(graph).clamp_min(eps), dim=-1)
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
        if self.prior_mean_mode == "graph_gp_conditioned":
            raise RuntimeError("graph_gp_conditioned prior must be built in forward because it depends on posterior_mu and epoch.")
        return self._residual_anchor_class_priors(class_attributes, graph, factorized=factorized)

    def _class_ids_tensor(self, class_ids, device: torch.device) -> torch.Tensor:
        """
        把 dataset 传入的 seen/unseen 类 id 转成全局类别 id tensor。

        Graph-GP 的 split 必须按全局类别 id 做，不能使用 local-output remap 后的类别编号。
        这里严格校验范围；如果没有 seen_class_ids，就直接报错，不做“全类都当 seen”的保底。
        """
        if class_ids is None:
            raise RuntimeError("PRIOR_MEAN_MODE=graph_gp_conditioned requires dataset seen_class_ids.")
        if torch.is_tensor(class_ids):
            ids = class_ids.detach().to(device=device, dtype=torch.long).view(-1)
        else:
            ids = torch.as_tensor(list(class_ids), device=device, dtype=torch.long).view(-1)
        if ids.numel() == 0:
            raise RuntimeError("PRIOR_MEAN_MODE=graph_gp_conditioned received empty seen_class_ids.")
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_classes:
            raise RuntimeError(
                "Graph-GP seen_class_ids must be global ids in [0,{}], got min={} max={}.".format(
                    self.num_classes - 1,
                    int(ids.min().item()),
                    int(ids.max().item()),
                )
            )
        return torch.unique(ids, sorted=True)

    def _get_graph_gp_episode_split(self, epoch: Optional[int], seen_class_ids, device: torch.device):
        """
        根据 epoch 生成 support-seen / pseudo-unseen 类别划分。

        划分按类别进行，不按 batch 样本进行。split_id 变化时清空本地累计的 support center buffer，
        这样每个 episode 都重新用当前 posterior 统计 V_support。
        """
        if epoch is None:
            raise RuntimeError("PRIOR_MEAN_MODE=graph_gp_conditioned requires epoch in loss kwargs.")
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        seen_ids = self._class_ids_tensor(seen_class_ids, device=torch.device("cpu"))
        split_period = int(prior_cfg.GRAPH_GP_SPLIT_EVERY_EPOCH)
        split_id = max(int(epoch) - 1, 0) // split_period
        use_pseudo = bool(prior_cfg.GRAPH_GP_USE_PSEUDO_UNSEEN)
        if use_pseudo and seen_ids.numel() <= 1:
            raise RuntimeError("Graph-GP pseudo-unseen split requires at least two seen classes.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(prior_cfg.GRAPH_GP_SPLIT_SEED) + int(split_id))
        perm = torch.randperm(int(seen_ids.numel()), generator=generator)
        if use_pseudo:
            support_count = int(round(float(prior_cfg.GRAPH_GP_SUPPORT_RATIO) * float(seen_ids.numel())))
            support_count = max(1, min(support_count, int(seen_ids.numel()) - 1))
        else:
            support_count = int(seen_ids.numel())
        support_ids_cpu = torch.sort(seen_ids.index_select(0, perm[:support_count])).values
        pseudo_ids_cpu = torch.sort(seen_ids.index_select(0, perm[support_count:])).values

        if self._graph_gp_split_id != split_id:
            # split 变化时必须重置本进程 buffer；否则旧 support 划分的视觉中心会混入新 episode。
            self._graph_gp_support_sum.zero_()
            self._graph_gp_support_sq_sum.zero_()
            self._graph_gp_support_var_sum.zero_()
            self._graph_gp_support_count.zero_()
            self._graph_gp_split_id = split_id
            self._graph_gp_support_ids = support_ids_cpu
            self._graph_gp_pseudo_unseen_ids = pseudo_ids_cpu
        return support_ids_cpu.to(device=device), pseudo_ids_cpu.to(device=device), int(split_id)

    @staticmethod
    def _class_membership_mask(values: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        """
        旧 PyTorch 兼容版 membership mask。

        torch.isin 在部分旧环境不可用；这里用广播比较判断 batch targets 是否属于 support_ids。
        """
        if members.numel() == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        return (values[:, None] == members.to(device=values.device, dtype=values.dtype)[None, :]).any(dim=1)

    def _update_graph_gp_support_buffers(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        support_ids: torch.Tensor,
    ) -> None:
        """
        用当前 batch 中属于 support-seen 的样本更新本进程类别统计。

        统计对象是 posterior_mu 而不是 ViT CLS，因为 posterior-prior KL 就发生在同一个 latent 空间。
        第一版强制 detach center，避免 M_star 这个训练目标被当前 batch 的 posterior 梯度反向拖动。
        """
        support_mask = self._class_membership_mask(targets_global, support_ids)
        if not bool(support_mask.any().item()):
            return
        cls = targets_global[support_mask].to(dtype=torch.long)
        values = posterior_mu[support_mask].detach()
        variances = posterior_logvar[support_mask].detach().exp()
        self._graph_gp_support_sum.index_add_(0, cls, values)
        self._graph_gp_support_sq_sum.index_add_(0, cls, values.pow(2))
        self._graph_gp_support_var_sum.index_add_(0, cls, variances)
        self._graph_gp_support_count.index_add_(0, cls, torch.ones_like(cls, dtype=self._graph_gp_support_count.dtype))

    @staticmethod
    def _distributed_sum_clone(x: torch.Tensor) -> torch.Tensor:
        """
        DDP 下同步一个 clone，而不是同步本地累计 buffer 本身。

        如果把 all_reduce 结果写回累计 buffer，下一次 forward 再 all_reduce 会把已经同步过的全局统计重复相加。
        因此这里始终 clone -> all_reduce -> 返回同步副本。
        """
        y = x.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
        return y

    @staticmethod
    def _solve_graph_gp_system(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """
        解 Graph-GP 条件均值里的线性方程 system * X = rhs。

        这是同一个数学操作的 PyTorch API 兼容层：新版本用 torch.linalg.solve，
        旧版本用 torch.solve(...).solution。它不改变模型逻辑，也不提供替代中心或保底 prior。
        """
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
            return torch.linalg.solve(system, rhs)
        return torch.solve(rhs, system).solution

    def _graph_gp_kernel(self, graph: torch.Tensor) -> torch.Tensor:
        """
        把当前 graph 处理成 Graph-GP 条件推断使用的 kernel。

        这里不重新构图，只对已经选定的 external/method graph 做必要的协方差式清洗：
        对称化保证 K_cd 和 K_dc 一致；非负裁剪避免负边进入协方差；diag 归一化让对角尺度接近 1。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        kernel = graph.detach()
        if bool(prior_cfg.GRAPH_GP_KERNEL_SYMMETRIZE):
            kernel = 0.5 * (kernel + kernel.t())
        if bool(prior_cfg.GRAPH_GP_KERNEL_CLAMP):
            kernel = kernel.clamp_min(0.0)
        normalize = str(prior_cfg.GRAPH_GP_KERNEL_NORMALIZE).lower()
        if normalize == "diag":
            diag = kernel.diag().clamp_min(eps).sqrt()
            kernel = kernel / (diag[:, None] * diag[None, :]).clamp_min(eps)
        elif normalize != "none":
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_NORMALIZE must be diag / none.")
        if not bool(torch.isfinite(kernel).all().item()):
            raise RuntimeError("Graph-GP kernel contains NaN or Inf after preprocessing.")
        return kernel

    def _graph_gp_dynamic_prior_var(
        self,
        reference: torch.Tensor,
        prototype_uncertainty: torch.Tensor,
        visual_within_var: torch.Tensor,
    ):
        """
        根据 Graph-GP 外推不确定性和类内视觉不确定性构造动态 prior variance。

        prototype_uncertainty 是 [C]，表示类别 prototype 均值由 support 类外推时的不确定性；
        visual_within_var 是 [C, D]，表示 support 类内视觉方差传播到所有类别后的结果。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        if not torch.is_tensor(prototype_uncertainty) or not torch.is_tensor(visual_within_var):
            raise RuntimeError(
                "GRAPH_GP_PRIOR_VAR_SOURCE=dynamic_uncertainty requires prototype_uncertainty and visual_within_var."
            )
        if prototype_uncertainty.dim() != 1 or int(prototype_uncertainty.shape[0]) != int(reference.shape[0]):
            raise RuntimeError("Graph-GP prototype_uncertainty must have shape [num_classes].")
        if tuple(visual_within_var.shape) != tuple(reference.shape):
            raise RuntimeError("Graph-GP visual_within_var must have the same shape as prior_mu/reference.")

        proto = prototype_uncertainty.to(device=reference.device, dtype=reference.dtype).clamp_min(0.0)[:, None]
        proto = proto.expand_as(reference)
        visual = visual_within_var.to(device=reference.device, dtype=reference.dtype).clamp_min(0.0)
        proto_term = float(prior_cfg.GRAPH_GP_PRIOR_VAR_PROTO_WEIGHT) * proto
        visual_term = float(prior_cfg.GRAPH_GP_PRIOR_VAR_VISUAL_WEIGHT) * visual
        floor = reference.new_tensor(float(prior_cfg.GRAPH_GP_PRIOR_VAR_FLOOR))
        prior_var_raw = floor + proto_term + visual_term
        prior_var = prior_var_raw.clamp(
            min=float(prior_cfg.GRAPH_GP_PRIOR_VAR_MIN),
            max=float(prior_cfg.GRAPH_GP_PRIOR_VAR_MAX),
        )
        return prior_var, {
            "prototype_uncertainty_expanded": proto.detach(),
            "visual_within_var": visual.detach(),
            "proto_var_term": proto_term.detach(),
            "visual_var_term": visual_term.detach(),
            "prior_var_raw": prior_var_raw.detach(),
            "prior_var": prior_var.detach(),
        }

    def _graph_gp_prior_logvar(
        self,
        reference: torch.Tensor,
        prototype_uncertainty: Optional[torch.Tensor] = None,
        visual_within_var: Optional[torch.Tensor] = None,
    ):
        """
        Graph-GP 第一版只让条件推断负责 prior_mu，不把 Sigma_{u|s} 直接塞进 prior_logvar。

        因此这里仅支持 unit / constant / current_prior_var_mode 中的固定方差形式。
        如果 current_prior_var_mode 实际是 learned，则直接报错，不做隐式降级。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        source = str(prior_cfg.GRAPH_GP_PRIOR_VAR_SOURCE).lower()
        if source == "unit":
            return torch.zeros_like(reference), {}
        if source == "constant":
            return torch.full_like(reference, float(prior_cfg.PRIOR_LOGVAR_CONST)), {}
        if source == "current_prior_var_mode":
            if self.prior_var_mode == "learned":
                raise RuntimeError("GRAPH_GP_PRIOR_VAR_SOURCE=current_prior_var_mode does not support PRIOR_VAR_MODE=learned.")
            return self._fixed_or_learned_logvar(reference), {}
        if source == "dynamic_uncertainty":
            prior_var, debug = self._graph_gp_dynamic_prior_var(
                reference,
                prototype_uncertainty=prototype_uncertainty,
                visual_within_var=visual_within_var,
            )
            prior_logvar = prior_var.log().clamp(min=self.logvar_min, max=self.logvar_max)
            return prior_logvar, debug
        raise ValueError(
            "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_SOURCE must be "
            "unit / constant / current_prior_var_mode / dynamic_uncertainty."
        )

    def _graph_gp_conditioned_class_priors(
        self,
        graph: torch.Tensor,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        seen_class_ids,
        epoch: Optional[int],
        is_train: bool,
    ):
        """
        Graph-GP zero-mean hyperprior + seen visual conditioning。

        数学形式：
            M ~ MN(0, K_G, I_D)
            V_s = M_s + eps, eps ~ N(0, R_s)
            E[M_all | V_s] = K_all,s (K_ss + R_s)^(-1) V_s

        代码中 V_s 来自 support-seen 类 posterior_mu 的累计均值；pseudo-unseen 类不参与 V_s，
        它们的位置只能通过 graph kernel 和 support-seen 视觉中心被推断出来。
        """
        if not bool(is_train):
            raise RuntimeError("graph_gp_conditioned first version is a training-time prior generator; eval prototype scoring is not implemented.")
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        support_ids, pseudo_unseen_ids, split_id = self._get_graph_gp_episode_split(epoch, seen_class_ids, posterior_mu.device)
        self._update_graph_gp_support_buffers(posterior_mu, posterior_logvar, targets_global, support_ids)

        synced_sum = self._distributed_sum_clone(self._graph_gp_support_sum.to(device=posterior_mu.device, dtype=posterior_mu.dtype))
        synced_sq_sum = self._distributed_sum_clone(self._graph_gp_support_sq_sum.to(device=posterior_mu.device, dtype=posterior_mu.dtype))
        synced_var_sum = self._distributed_sum_clone(self._graph_gp_support_var_sum.to(device=posterior_mu.device, dtype=posterior_mu.dtype))
        synced_count = self._distributed_sum_clone(self._graph_gp_support_count.to(device=posterior_mu.device, dtype=posterior_mu.dtype))

        support_count_all = synced_count.index_select(0, support_ids)
        observed_mask = support_count_all > 0.0
        if not bool(observed_mask.any().item()):
            raise RuntimeError(
                "Graph-GP has no observed support-seen class in the accumulated buffer; "
                "this implementation does not create fallback pseudo centers."
            )
        observed_support_ids = support_ids[observed_mask]
        observed_count = support_count_all[observed_mask].clamp_min(1.0)
        center_sum = synced_sum.index_select(0, observed_support_ids)
        center_sq_sum = synced_sq_sum.index_select(0, observed_support_ids)
        center_var_sum = synced_var_sum.index_select(0, observed_support_ids)
        v_support = center_sum / observed_count[:, None]

        center_var_dim = (center_sq_sum / observed_count[:, None] - v_support.pow(2)).clamp_min(0.0)
        support_post_var_dim = (center_var_sum / observed_count[:, None]).clamp_min(0.0)
        support_visual_var_dim = center_var_dim + support_post_var_dim
        center_var = center_var_dim.mean(dim=-1)
        obs_mode = str(prior_cfg.GRAPH_GP_OBS_NOISE_MODE).lower()
        if obs_mode == "constant":
            obs_noise = torch.full_like(observed_count, float(prior_cfg.GRAPH_GP_OBS_NOISE_CONST))
        elif obs_mode == "class_var_over_count":
            obs_noise = center_var / observed_count.clamp_min(1.0)
        else:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MODE must be constant / class_var_over_count.")
        obs_noise = obs_noise.clamp(
            min=float(prior_cfg.GRAPH_GP_OBS_NOISE_MIN),
            max=float(prior_cfg.GRAPH_GP_OBS_NOISE_MAX),
        )

        kernel = self._graph_gp_kernel(graph).to(device=posterior_mu.device, dtype=posterior_mu.dtype)
        k_all_s = kernel.index_select(1, observed_support_ids)
        k_ss = k_all_s.index_select(0, observed_support_ids)
        system = k_ss + torch.diag(obs_noise + float(prior_cfg.GRAPH_GP_RIDGE))

        # 不显式求 inverse：solve(system, V_support) 更稳定，也避免构造完整逆矩阵。
        solved_v = self._solve_graph_gp_system(system, v_support)
        prior_mu = k_all_s.matmul(solved_v)

        # predictive uncertainty 先只做监测，不进入 prior_logvar。
        solved_k = self._solve_graph_gp_system(system, k_all_s.t())
        smoothing_coeff = solved_k.t()
        uncertainty_diag = (kernel.diag() - (k_all_s * solved_k.t()).sum(dim=1)).clamp_min(0.0)
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        var_weight = smoothing_coeff.clamp_min(0.0)
        fallback_weight = k_all_s.clamp_min(0.0)
        var_weight_sum = var_weight.sum(dim=1, keepdim=True)
        fallback_sum = fallback_weight.sum(dim=1, keepdim=True)
        var_weight = torch.where(
            var_weight_sum > eps,
            var_weight / var_weight_sum.clamp_min(eps),
            fallback_weight / fallback_sum.clamp_min(eps),
        )
        visual_within_var = var_weight.matmul(support_visual_var_dim).clamp_min(0.0)
        solve_residual = (system.matmul(solved_v) - v_support).norm() / v_support.norm().clamp_min(1e-12)
        system_diag = system.diag().abs().clamp_min(1e-12)
        system_diag_ratio = system_diag.max() / system_diag.min()

        if bool(prior_cfg.GRAPH_GP_MATCH_DETACH_PRIOR):
            prior_mu = prior_mu.detach()
        prior_logvar, prior_var_debug = self._graph_gp_prior_logvar(
            prior_mu,
            prototype_uncertainty=uncertainty_diag,
            visual_within_var=visual_within_var,
        )
        self._last_prior_debug = {}
        self._last_graph_gp_debug = {
            "support_ids": support_ids.detach(),
            "pseudo_unseen_ids": pseudo_unseen_ids.detach(),
            "observed_support_ids": observed_support_ids.detach(),
            "support_count": synced_count.detach(),
            "center_var": center_var.detach(),
            "support_post_var": support_post_var_dim.mean(dim=-1).detach(),
            "support_visual_var": support_visual_var_dim.mean(dim=-1).detach(),
            "obs_noise": obs_noise.detach(),
            "solve_residual": solve_residual.detach(),
            "uncertainty_diag": uncertainty_diag.detach(),
            "visual_within_var": visual_within_var.detach(),
            "system_diag_ratio": system_diag_ratio.detach(),
            "kernel": kernel.detach(),
            "system": system.detach(),
            "k_all_s": k_all_s.detach(),
            "smoothing_coeff": smoothing_coeff.detach(),
            "var_weight": var_weight.detach(),
            "split_id": posterior_mu.new_tensor(float(split_id)),
            "observed_support_ratio": posterior_mu.new_tensor(
                float(observed_support_ids.numel()) / float(max(int(support_ids.numel()), 1))
            ),
        }
        self._last_graph_gp_debug.update(prior_var_debug)
        return prior_mu, prior_logvar

    def _masked_topk_distribution(
        self,
        graph: torch.Tensor,
        topk: int,
        tau: torch.Tensor,
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

        ?? residual_anchor prior ????? P_plus?
        - P_plus = _masked_topk_distribution(graph, GRAPH_INPUT.TOPK, TAU_GRAPH)
          ??????? graph top-k ??????????
        """
        class_count = int(graph.shape[0])
        topk = int(topk)
        tau = torch.as_tensor(tau, device=graph.device, dtype=graph.dtype)
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        if graph.dim() != 2 or int(graph.shape[1]) != class_count:
            raise RuntimeError(f"GraphProbPrior expects square graph [C,C], got {tuple(graph.shape)}.")
        if topk <= 0:
            raise ValueError("topk must be positive.")
        max_topk = class_count - 1 if exclude_self else class_count
        if topk > max_topk:
            raise ValueError(f"topk must be <= {max_topk}, got {topk}.")
        if bool((tau.detach() <= 0.0).any().item()):
            raise ValueError("tau must be positive.")

        # candidate_logits 是可被选择的原始关系值。
        # exclude_self=True 时，对角线被置为 -inf，因此 top-k 不会选到类别自身。
        candidate_logits = graph
        if exclude_self:
            candidate_logits = graph.clone()
            diag = torch.arange(class_count, device=graph.device)
            candidate_logits[diag, diag] = float("-inf")

        # 只在有限的 top-k values 上做 softmax，再 scatter 回完整矩阵。
        # 不要对含 -inf 的 masked_logits 除以 learnable tau；否则 tau 反传会遇到 -inf，
        # 容易出现 0 * inf 形式的 NaN 梯度，导致 LEARN_TAU_GRAPH 首批训练崩溃。
        values, indices = torch.topk(candidate_logits, k=topk, dim=-1)
        topk_prob = F.softmax(values / tau.clamp_min(eps), dim=-1)

        # 非 top-k 位置保持严格为 0，top-k 行和理论上为 1。
        prob = torch.zeros_like(graph)
        prob.scatter_(1, indices, topk_prob)
        return prob / prob.sum(dim=-1, keepdim=True).clamp_min(eps)

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

        class_aggregate_mmd 只直接对当前 batch 出现类别做
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
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)

        # 语义图 target 是固定监督信号，不让梯度回传到 graph。
        target_rel = F.softmax(graph / self._tau_graph_value(graph).clamp_min(eps), dim=-1).detach()

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

    def _geometry_topk(self, graph: torch.Tensor, topk: int, tau: torch.Tensor):
        """
        geometry loss 专用 top-k 读取。

        它和 _masked_topk_distribution 的核心一致，但这里还需要返回 top-k 下标和值，
        因为三种 geometry loss 都要在 D_cd 或 similarity 矩阵里 gather 对应类别对。
        """
        class_count = int(graph.shape[0])
        topk = min(int(topk), class_count - 1)
        if topk <= 0:
            raise ValueError("geometry topk must be positive after excluding self.")
        tau = torch.as_tensor(tau, device=graph.device, dtype=graph.dtype)
        if bool((tau.detach() <= 0.0).any().item()):
            raise ValueError("geometry tau must be positive.")
        logits = graph.clone()
        diag = torch.arange(class_count, device=graph.device)
        logits[diag, diag] = float("-inf")
        values, indices = torch.topk(logits, k=topk, dim=-1)
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        weights = F.softmax(values / tau.clamp_min(eps), dim=-1)
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
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
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

    def graph_ordinal_ranking_loss(
        self,
        prior_mu: torch.Tensor,
        graph: torch.Tensor,
        d_norm: torch.Tensor,
        dist_mu: torch.Tensor,
        eye: torch.Tensor,
    ):
        """
        局部 graph ordinal ranking loss。

        这个 loss 只回答一个问题：在 semantic graph 的局部 top-k 邻域内，
        graph 认为更相似的类别，是否在 class prior 空间里也更近。

        重要设计：
        - 半径/方差相关距离不在这里重新实现，而是复用
          compute_prior_distribution_distance() 传入的 d_norm/dist_mu。
        - GEOM_ORD_DISTANCE_TYPE=clearance 时直接使用 d_norm，
          即 ||mu_i-mu_j|| / (radius_i+radius_j)，天然考虑 prior 分布宽度。
        - non-overlap 弱边界也固定使用 d_norm，因为它的语义就是“分布间隔”
          而不是单纯的均值方向或均值欧氏距离。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(prior_cfg.GEOM_ORD_EPS)
        class_count = int(prior_mu.shape[0])
        topk = min(int(prior_cfg.GEOM_TOPK), class_count - 1)
        row = torch.arange(class_count, device=prior_mu.device)[:, None]

        # graph 只提供排序目标，不应该从 geometry loss 反向更新 graph 构造过程。
        # clone 后再屏蔽对角线，避免对传入的原始 graph 做原地修改。
        graph_detached = graph.detach()
        graph_for_topk = graph_detached.clone()
        diag = torch.arange(class_count, device=prior_mu.device)
        graph_for_topk[diag, diag] = float("-inf")
        top_values, top_indices = torch.topk(graph_for_topk, k=topk, dim=-1)

        # 选择 ranking 使用的 prior 距离矩阵。
        # clearance 直接复用 compute_prior_distribution_distance() 的 d_norm；
        # euclidean 复用同一个函数返回的 dist_mu；
        # cosine 只作为备选方向距离，不参与半径计算。
        distance_type = str(prior_cfg.GEOM_ORD_DISTANCE_TYPE).lower()
        if distance_type == "clearance":
            distance = d_norm
        elif distance_type == "euclidean":
            distance = dist_mu
        elif distance_type == "cosine":
            prior_dir = F.normalize(prior_mu, p=2, dim=-1, eps=eps)
            distance = (1.0 - prior_dir.matmul(prior_dir.t())).masked_fill(eye, 0.0)
        else:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_DISTANCE_TYPE must be clearance / cosine / euclidean.")

        # d[c, t] 是类别 c 到第 t 个 graph top-k 邻居的 prior 距离。
        d = distance[row, top_indices]

        # 构造 top-k 内所有有序 pair：
        # 如果 graph gap 满足 G[c,i] > G[c,j] + eps，
        # 就希望 prior distance 满足 D[c,i] + margin < D[c,j]。
        g_i = top_values.unsqueeze(2)
        g_j = top_values.unsqueeze(1)
        d_i = d.unsqueeze(2)
        d_j = d.unsqueeze(1)
        eye_k = torch.eye(topk, dtype=torch.bool, device=prior_mu.device).view(1, topk, topk)
        valid = (g_i > g_j + float(prior_cfg.GEOM_ORD_GRAPH_GAP_EPS)) & (~eye_k)
        valid_float = valid.to(dtype=prior_mu.dtype)

        # 每一行 graph 的 top-k 相似度范围不同，先做行内归一化；
        # 这样 gap_norm 只表达“这一行内部谁明显更像”，不会被不同类别的图尺度影响。
        row_range = (top_values.max(dim=-1).values - top_values.min(dim=-1).values).clamp_min(eps)
        row_range = row_range.view(class_count, 1, 1)
        gap_norm = ((g_i - g_j).clamp_min(0.0) / row_range).clamp(0.0, 1.0)

        # graph gap 越大，排序间隔越大；如果关闭 gap weighting，
        # 所有有效 pair 权重相同，但 margin 仍然可以随 gap 增大。
        margin_base = prior_mu.new_tensor(float(prior_cfg.GEOM_ORD_MARGIN_BASE))
        margin_scale = self._geom_ord_margin_scale_value(prior_mu)
        margin = margin_base + margin_scale * gap_norm
        if bool(prior_cfg.GEOM_ORD_WEIGHT_BY_GAP):
            pair_weight = gap_norm
        else:
            pair_weight = torch.ones_like(gap_norm)

        # ReLU hinge ranking：只有排序错误或间隔不足时才产生惩罚。
        # violation <= 0 表示 D(c,i)+margin <= D(c,j)，该 pair 已满足 graph 排序要求；
        # violation > 0 表示更相似的 i 没有比 j 足够近，按违反幅度线性惩罚。
        violation = d_i - d_j + margin
        pair_loss = F.relu(violation)
        weighted_valid = pair_weight * valid_float
        valid_weight_sum = weighted_valid.sum()
        ordinal_loss = (pair_loss * weighted_valid).sum() / valid_weight_sum.clamp_min(eps)

        # non-overlap 是可选弱边界项。它不改变 ordinal ranking 的主体逻辑，
        # 只在明确打开权重时，防止 top-k 或 all 非自身类别的分布间隔 d_norm 过小。
        non_overlap_scope = str(prior_cfg.GEOM_ORD_NON_OVERLAP_SCOPE).lower()
        if non_overlap_scope == "topk":
            overlap_dist = d_norm[row, top_indices]
        elif non_overlap_scope == "all":
            overlap_dist = d_norm[~eye]
        else:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_NON_OVERLAP_SCOPE must be topk / all.")

        non_overlap_weight = self._geom_ord_non_overlap_weight_value(prior_mu)
        non_overlap_enabled = float(non_overlap_weight.detach().item()) > 0.0
        if bool(non_overlap_enabled):
            # ReLU hinge non-overlap：只有分布间隔 d_norm 小于最小安全距离时才惩罚。
            non_overlap_loss = F.relu(float(prior_cfg.GEOM_ORD_NON_OVERLAP_MIN_DIST) - overlap_dist).mean()
        else:
            # 保持 device/dtype 和反向图兼容；默认关闭时该项不产生梯度。
            non_overlap_loss = prior_mu.sum() * 0.0
        total_loss = ordinal_loss + non_overlap_weight * non_overlap_loss

        # 下面的 stats 全部 detach，只用于日志诊断，不参与反向传播。
        valid_count = valid_float.sum()
        valid_denom = valid_count.clamp_min(1.0)
        valid_mask = valid_float > 0.0
        violation_rate = ((violation > 0.0).to(dtype=prior_mu.dtype) * valid_float).sum() / valid_denom
        rank_acc = ((d_i < d_j).to(dtype=prior_mu.dtype) * valid_float).sum() / valid_denom
        gap_mean = (gap_norm * valid_float).sum() / valid_denom
        margin_mean = (margin * valid_float).sum() / valid_denom
        if bool(valid_mask.any().item()):
            valid_violation_mean = violation[valid_mask].mean()
        else:
            valid_violation_mean = prior_mu.sum() * 0.0
        non_overlap_violation_rate = (
            (overlap_dist < float(prior_cfg.GEOM_ORD_NON_OVERLAP_MIN_DIST)).to(dtype=prior_mu.dtype).mean()
            if overlap_dist.numel() > 0
            else prior_mu.sum() * 0.0
        )
        non_overlap_dist_mean = overlap_dist.mean() if overlap_dist.numel() > 0 else prior_mu.sum() * 0.0

        stats = {
            "graph_prob_prior_geom_ord_loss": float(ordinal_loss.detach().item()),
            "graph_prob_prior_geom_ord_total_loss": float(total_loss.detach().item()),
            "graph_prob_prior_geom_ord_valid_pair_count": float(valid_count.detach().item()),
            "graph_prob_prior_geom_ord_violation_rate": float(violation_rate.detach().item()),
            "graph_prob_prior_geom_ord_rank_acc": float(rank_acc.detach().item()),
            "graph_prob_prior_geom_ord_gap_mean": float(gap_mean.detach().item()),
            "graph_prob_prior_geom_ord_margin_mean": float(margin_mean.detach().item()),
            "graph_prob_prior_geom_ord_violation_mean": float(valid_violation_mean.detach().item()),
            "graph_prob_prior_geom_ord_top1_dist_mean": float(d[:, 0].detach().mean().item()),
            "graph_prob_prior_geom_ord_topk_last_dist_mean": float(d[:, -1].detach().mean().item()),
            "graph_prob_prior_geom_ord_topk_dist_mean": float(d.detach().mean().item()),
            "graph_prob_prior_geom_ord_non_overlap_loss": float(non_overlap_loss.detach().item()),
            "graph_prob_prior_geom_ord_non_overlap_enabled": 1.0 if bool(non_overlap_enabled) else 0.0,
            "graph_prob_prior_geom_ord_non_overlap_violation_rate": float(non_overlap_violation_rate.detach().item()),
            "graph_prob_prior_geom_ord_non_overlap_dist_mean": float(non_overlap_dist_mean.detach().item()),
        }
        return total_loss, stats, top_indices

    def compute_graph_prior_geometry_loss(
        self,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        graph: torch.Tensor,
    ):
        """
        prior_mu 几何校准正则统一入口。

        它不生成 prior_mu，只在 residual-anchor 已经得到 prior_mu 后追加轻量约束：
        - soft_distribution_matching: 让 graph top-k 分布与 prior-distance 诱导分布一致。
        - graph_ordinal_ranking: 只约束 graph top-k 内的 prior 距离相对排序。
        """
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
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

        if method == "soft_distribution_matching":
            top_values, top_indices, target_weight = self._geometry_topk(
                graph,
                topk=int(prior_cfg.GEOM_TOPK),
                tau=float(prior_cfg.GEOM_TAU_GRAPH_DIST),
            )
            geom_tau_dist = self._geom_tau_dist_value(d_norm).clamp_min(eps)
            logits = (-d_norm / geom_tau_dist).masked_fill(eye, float("-inf"))
            log_q = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            log_q_top = log_q[row, top_indices]
            dist_match_kl = target_weight.mul(target_weight.clamp_min(eps).log() - log_q_top).sum(dim=-1).mean()

            top_d = d_norm[row, top_indices]
            bound_loss = F.softplus(
                (1.0 + float(prior_cfg.GEOM_MARGIN_MIN) - top_d) / float(prior_cfg.GEOM_TAU_BARRIER)
            ).mul(target_weight).sum(dim=-1).mean()
            geom_bound_weight = self._geom_bound_weight_value(d_norm)
            loss = dist_match_kl + geom_bound_weight * bound_loss
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

        if method == "graph_ordinal_ranking":
            ordinal_loss, ordinal_stats, top_indices = self.graph_ordinal_ranking_loss(
                prior_mu=prior_mu,
                graph=graph,
                d_norm=d_norm,
                dist_mu=dist_mu,
                eye=eye,
            )
            loss = ordinal_loss
            stats.update(ordinal_stats)
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
            stats["graph_prob_prior_geom_loss"] = float(loss.detach().item())
            return loss, stats

        raise ValueError(
            "MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_TYPE must be soft_distribution_matching / graph_ordinal_ranking."
        )

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
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)

        # distance: [B, C]
        distance = self._gaussian_kl_all_classes(posterior_mu, posterior_logvar, prior_mu, prior_logvar)

        # 把“距离”变成“类别概率”：距离越小 -> -distance 越大 -> softmax 后概率越高。
        # TAU_LATENT 是 prediction 侧温度：tau 小 -> latent_prob 更尖，更偏向 KL 最小的类别； tau 大 -> latent_prob 更平，多个类别会分到概率。
        latent_prob = F.softmax(-distance / self._tau_latent_value(distance).clamp_min(eps), dim=-1)

        # target 是 GraphPriorInputBuilder 根据 graph[y] 构造的语义监督分布 [B, C]。
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
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
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
                    compute_effective_rank=self.monitor_effective_rank,
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
        epoch: Optional[int] = None,
        is_train: bool = True,
    ) -> torch.Tensor:
        """
        计算 GraphProbPrior loss。

        输入 posterior_mu/logvar 来自 prompt distributor runtime stats；
        语义图资源仍由 trainer 通过 loss kwargs 显式传入。
        """
        # factorized_latent 下 posterior_mu/logvar 应该是 semantic factor 维度；
        # 其他 mode 下则使用完整 text_dim=768。
        # 因此和 factorized_latent 一样检查 semantic_dim，而不是完整 768 维。
        expected_dim = self.factorized_semantic_dim if self.mode == "factorized_latent" else self.text_dim
        if posterior_mu.dim() != 2 or posterior_mu.shape[1] != expected_dim:
            raise RuntimeError(f"GraphProbPrior expects posterior mu [B,{expected_dim}], got {tuple(posterior_mu.shape)}.")
        if tuple(posterior_logvar.shape) != tuple(posterior_mu.shape):
            raise RuntimeError(
                "GraphProbPrior expects posterior logvar shape {}, got {}.".format(
                    tuple(posterior_mu.shape),
                    tuple(posterior_logvar.shape),
                )
            )
        if self.mode == "factorized_latent":
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
        #   bank:   [C,768] 类别语义原型；graph_gp_conditioned 下为 None；
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
        if self.prior_mean_mode == "graph_gp_conditioned":
            prior_mu, prior_logvar = self._graph_gp_conditioned_class_priors(
                graph=graph,
                posterior_mu=posterior_mu,
                posterior_logvar=posterior_logvar,
                targets_global=targets_global,
                seen_class_ids=seen_class_ids,
                epoch=epoch,
                is_train=bool(is_train),
            )
        else:
            prior_mu, prior_logvar = self._class_priors(
                bank,
                graph,
                class_attributes,
                factorized=(self.mode == "factorized_latent"),
            )
        if monitor_active:
            graph_cfg = self.cfg.MODEL.GRAPH_INPUT
            tau_graph_value = float(self._tau_graph_value(graph).detach().item())
            monitor_stats.update(graph_neighbor_monitor(graph, tau_graph_value, topk=self.monitor_topk))
            monitor_stats.update(graph_health_monitor(graph, tau_graph_value, topk=self.monitor_topk))
            monitor_stats["graph_prob_prior_monitor_learnable_tau_graph_value"] = tau_graph_value
            monitor_stats["graph_prob_prior_monitor_learnable_tau_latent_value"] = float(
                self._tau_latent_value(posterior_mu).detach().item()
            )
            monitor_stats["graph_prob_prior_monitor_learnable_geom_tau_dist_value"] = float(
                self._geom_tau_dist_value(posterior_mu).detach().item()
            )
            monitor_stats["graph_prob_prior_monitor_learnable_geom_bound_weight_value"] = float(
                self._geom_bound_weight_value(posterior_mu).detach().item()
            )
            monitor_stats["graph_prob_prior_monitor_learnable_geom_ord_margin_scale_value"] = float(
                self._geom_ord_margin_scale_value(posterior_mu).detach().item()
            )
            monitor_stats["graph_prob_prior_monitor_learnable_geom_ord_non_overlap_weight_value"] = float(
                self._geom_ord_non_overlap_weight_value(posterior_mu).detach().item()
            )
            monitor_stats.update(
                false_high_pair_monitor(
                    acc,
                    graph,
                    threshold=0.9,
                    prefix="graph_prob_prior_monitor_false_high_graph_relation",
                )
            )
            if self.mode in {"graph_conditioned_semantic_prior", "factorized_latent"} or self.monitor_inactive:
                monitor_stats.update(
                    semantic_target_monitor(
                        graph,
                        targets_global,
                        tau_acc=float(graph_cfg.TAU_ACC),
                        graph_topk=int(graph_cfg.TOPK),
                        target_mix_alpha=float(graph_cfg.TARGET_MIX_ALPHA),
                        num_classes=int(graph_cfg.NUM_CLASSES),
                        eps=float(graph_cfg.EPS),
                        topk=self.monitor_topk,
                    )
                )
            if self.prior_mean_mode == "residual_anchor":
                prior_debug = self._last_prior_debug
                if "prior_mu_scale_value" in prior_debug:
                    monitor_stats["graph_prob_prior_monitor_learnable_prior_mu_scale_value"] = float(
                        prior_debug["prior_mu_scale_value"].detach().item()
                    )
                if "prior_delta_scale_value" in prior_debug:
                    monitor_stats["graph_prob_prior_monitor_learnable_prior_delta_scale_value"] = float(
                        prior_debug["prior_delta_scale_value"].detach().item()
                    )
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
        if self.mode == "graph_conditioned_semantic_prior":
            match_loss, match_stats, debug_info = self._samplewise_latent_matching_loss(
                posterior_mu,
                posterior_logvar,
                prior_mu,
                prior_logvar,
                target,
                targets_global=targets_global,
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
        else:
            raise ValueError(
                "MODEL.GRAPH_PROB_PRIOR.MODE must be graph_conditioned_semantic_prior / "
                "class_aggregate_mmd / factorized_latent."
            )
        if monitor_active and self.prior_mean_mode == "graph_gp_conditioned":
            # Graph-GP 专属监测需要当前 batch 到 M_star 的距离，用于区分 support-seen 和 pseudo-unseen 表现。
            # 这里仅在 monitor step 额外计算一次，不改变训练 loss。
            graph_gp_distance = self._gaussian_kl_all_classes(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
            eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
            graph_gp_latent_prob = F.softmax(
                -graph_gp_distance / self._tau_latent_value(graph_gp_distance).clamp_min(eps),
                dim=-1,
            )
            graph_gp_sample_match_loss = (
                _normalize_prob(target, eps)
                * (_normalize_prob(target, eps).log() - _normalize_prob(graph_gp_latent_prob, eps).log())
            ).sum(dim=-1)
            graph_gp_debug = self._last_graph_gp_debug
            monitor_stats.update(
                graph_gp_prototype_monitor(
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    support_ids=graph_gp_debug["support_ids"],
                    pseudo_unseen_ids=graph_gp_debug["pseudo_unseen_ids"],
                    observed_support_ids=graph_gp_debug["observed_support_ids"],
                    support_count=graph_gp_debug["support_count"],
                    center_var=graph_gp_debug["center_var"],
                    obs_noise=graph_gp_debug["obs_noise"],
                    solve_residual=graph_gp_debug["solve_residual"],
                    uncertainty_diag=graph_gp_debug["uncertainty_diag"],
                    system_diag_ratio=graph_gp_debug["system_diag_ratio"],
                    support_post_var=graph_gp_debug.get("support_post_var"),
                    support_visual_var=graph_gp_debug.get("support_visual_var"),
                    visual_within_var=graph_gp_debug.get("visual_within_var"),
                    proto_var_term=graph_gp_debug.get("proto_var_term"),
                    visual_var_term=graph_gp_debug.get("visual_var_term"),
                    prior_var_raw=graph_gp_debug.get("prior_var_raw"),
                    dynamic_prior_var=graph_gp_debug.get("prior_var"),
                    kernel=graph_gp_debug["kernel"],
                    system=graph_gp_debug["system"],
                    k_all_s=graph_gp_debug["k_all_s"],
                    smoothing_coeff=graph_gp_debug["smoothing_coeff"],
                    graph=graph_gp_debug["kernel"],
                    distance=graph_gp_distance,
                    sample_match_loss=graph_gp_sample_match_loss,
                    posterior_mu=posterior_mu,
                    targets_global=targets_global,
                    seen_class_ids=seen_class_ids,
                    unseen_class_ids=unseen_class_ids,
                    topk=self.monitor_topk,
                )
            )
            debug_info.update(
                {
                    "graph_gp_support_count": int(graph_gp_debug["support_ids"].numel()),
                    "graph_gp_observed_support_count": int(graph_gp_debug["observed_support_ids"].numel()),
                }
            )

        rel_weight = float(prior_cfg.REL_WEIGHT)
        rel_enabled = rel_weight > 0.0 and self.mode == "class_aggregate_mmd"
        if rel_enabled:
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
            if monitor_active and self.monitor_inactive:
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
            if monitor_active:
                monitor_stats.update(self._graph_prior_geometry_stats(prior_mu, prior_logvar, graph))

        if monitor_active and self.monitor_inactive and self.mode not in {
            "graph_conditioned_semantic_prior",
            "factorized_latent",
        }:
            inactive_distance = self._gaussian_kl_all_classes(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
            eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
            inactive_latent_prob = F.softmax(
                -inactive_distance / self._tau_latent_value(inactive_distance).clamp_min(eps),
                dim=-1,
            )
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
                    compute_effective_rank=self.monitor_effective_rank,
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
            "graph_prob_prior_rel_enabled": float(rel_enabled),
            "graph_prob_prior_geom_loss": float(geometry_loss.detach().item()),
            "graph_prob_prior_geom_enabled": float(bool(prior_cfg.GEOM_LOSS_ENABLE)),
            "graph_prob_prior_posterior_mu_norm": float(posterior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_prior_mu_norm": float(prior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_posterior_var_mean": float(posterior_var.detach().mean().item()),
            "graph_prob_prior_prior_var_mean": float(prior_var.detach().mean().item()),
        }

        # DEBUG=True 时只打印一次关键 shape 和 loss，方便核对 mode 分支与张量维度。
        if bool(prior_cfg.DEBUG) and not self._debug_logged:
            bank_shape = None if bank is None else tuple(bank.shape)
            print(
                "[GRAPH-PROB-PRIOR-DEBUG] mode={} posterior_mu={} posterior_logvar={} bank={} graph={} "
                "prior_mu={} prior_logvar={} target={} debug={} loss={:.6f}".format(
                    self.mode,
                    tuple(posterior_mu.shape),
                    tuple(posterior_logvar.shape),
                    bank_shape,
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
