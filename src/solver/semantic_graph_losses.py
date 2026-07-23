#!/usr/bin/env python3
"""Graph-GP 图输入、条件 prototype 与 energy classification。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .graph_prob_prior_monitors import (
    false_high_pair_monitor,
    graph_gp_prototype_monitor,
    gzsl_prior_risk_monitor,
    graph_health_monitor,
    prior_health_monitor,
    seen_unseen_prior_monitor,
)

def _row_normalize(x: torch.Tensor) -> torch.Tensor:
    """行向量 L2 归一化，用于余弦相似度图构造。"""
    return F.normalize(x, p=2, dim=-1)


class GraphPriorInputBuilder:
    """
    Graph-GP 输入构造器：校验类别属性与全局标签，加载 external graph。

    类别属性仅用于构造 Acc 诊断 false-high 关系；Graph-GP prototype 本身只使用
    external method graph 和 support-seen posterior center。
    """

    GRAPH_GP_EXTERNAL_GRAPH_PATH = (
        "cub_attribute_localization/05_hparam_searches/diff_only_graphs_v1/diff_only_method_matrices_v1.npz"
    )
    GRAPH_GP_GRAPH_SOURCES = ("method1_diff", "method2_diff", "method3_diff")
    GRAPH_GP_DEFAULT_GRAPH_SOURCE = "method1_diff"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        graph_cfg = cfg.MODEL.GRAPH_INPUT
        self.num_classes = int(graph_cfg.NUM_CLASSES)
        self.attr_dim = int(graph_cfg.ATTR_DIM)
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
    ):
        """构造 false-high 诊断所需 Acc，并加载 Graph-GP external graph。"""
        graph_cfg = self.cfg.MODEL.GRAPH_INPUT
        graph_source = str(graph_cfg.GRAPH_SOURCE).lower()
        acc = _row_normalize(class_attributes).matmul(_row_normalize(class_attributes).t())
        if graph_source not in self.GRAPH_GP_GRAPH_SOURCES:
            raise ValueError(
                "MODEL.GRAPH_INPUT.GRAPH_SOURCE must be method1_diff / method2_diff / method3_diff."
            )
        graph = self._load_external_graph(
            class_attributes.device,
            class_attributes.dtype,
            key=graph_source,
            default_path=self.GRAPH_GP_EXTERNAL_GRAPH_PATH,
            allowed_keys=self.GRAPH_GP_GRAPH_SOURCES,
        )
        return acc.detach(), graph.detach()

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

    def prepare(
        self,
        targets_global: torch.Tensor,
        class_attributes: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """一次性准备 Graph-GP 所需的图输入张量。"""
        targets_global = self.validate_targets(targets_global, device)
        class_attributes = self.prepare_class_attributes(class_attributes, device, dtype)
        acc, graph = self.build_graphs(class_attributes)
        return {
            "class_attributes": class_attributes,
            "acc": acc,
            "graph": graph,
            "targets_global": targets_global,
        }

class GraphProbPriorLossComputer(torch.nn.Module):
    """Graph-GP conditionally inferred class prototypes and energy classification."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.graph_builder = GraphPriorInputBuilder(cfg)
        graph_cfg = cfg.MODEL.GRAPH_INPUT
        prior_cfg = cfg.MODEL.GRAPH_PROB_PRIOR
        dist_cfg = cfg.MODEL.PROMPT.DISTRIBUTOR

        self.num_classes = int(graph_cfg.NUM_CLASSES)
        self.text_dim = int(graph_cfg.TEXT_DIM)
        self.logvar_min = float(dist_cfg.LOGVAR_MIN)
        self.logvar_max = float(dist_cfg.LOGVAR_MAX)
        self.graph_gp_objective = str(prior_cfg.GRAPH_GP_OBJECTIVE).lower()
        self.graph_gp_energy_class_space = str(prior_cfg.GRAPH_GP_ENERGY_CLASS_SPACE).lower()

        if self.graph_gp_objective != "energy_classification":
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBJECTIVE must be energy_classification.")
        if float(prior_cfg.GRAPH_GP_ENERGY_TAU) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_ENERGY_TAU must be positive.")
        if self.graph_gp_energy_class_space != "seen":
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_ENERGY_CLASS_SPACE must be seen.")
        if float(prior_cfg.GRAPH_GP_PSEUDO_WEIGHT) <= 0.0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PSEUDO_WEIGHT must be positive.")
        if not (0.0 < float(prior_cfg.GRAPH_GP_SUPPORT_RATIO) < 1.0):
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SUPPORT_RATIO must be in (0, 1).")
        if int(prior_cfg.GRAPH_GP_SPLIT_EVERY_EPOCH) <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SPLIT_EVERY_EPOCH must be positive.")
        if str(prior_cfg.GRAPH_GP_CENTER_SOURCE).lower() != "posterior_mu":
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_CENTER_SOURCE supports only posterior_mu.")
        if not bool(prior_cfg.GRAPH_GP_DETACH_CENTERS):
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_DETACH_CENTERS must be True.")
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
        if str(prior_cfg.GRAPH_GP_PRIOR_VAR_SOURCE).lower() not in {"unit", "constant", "dynamic_uncertainty"}:
            raise ValueError(
                "MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_SOURCE must be "
                "unit / constant / dynamic_uncertainty."
            )
        for key in ("GRAPH_GP_PRIOR_VAR_FLOOR", "GRAPH_GP_PRIOR_VAR_MIN", "GRAPH_GP_PRIOR_VAR_MAX"):
            if float(getattr(prior_cfg, key)) <= 0.0:
                raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{key} must be positive.")
        if float(prior_cfg.GRAPH_GP_PRIOR_VAR_MAX) < float(prior_cfg.GRAPH_GP_PRIOR_VAR_MIN):
            raise ValueError("GRAPH_GP_PRIOR_VAR_MAX must be >= GRAPH_GP_PRIOR_VAR_MIN.")
        for key in ("GRAPH_GP_PRIOR_VAR_PROTO_WEIGHT", "GRAPH_GP_PRIOR_VAR_VISUAL_WEIGHT"):
            if float(getattr(prior_cfg, key)) < 0.0:
                raise ValueError(f"MODEL.GRAPH_PROB_PRIOR.{key} must be non-negative.")
        if not bool(prior_cfg.GRAPH_GP_MATCH_DETACH_PRIOR):
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_MATCH_DETACH_PRIOR must be True.")

        self.monitor_enable = bool(prior_cfg.MONITOR_ENABLE)
        self.monitor_topk = int(prior_cfg.MONITOR_TOPK)
        self.monitor_effective_rank = bool(getattr(prior_cfg, "MONITOR_EFFECTIVE_RANK", False))
        self.monitor_every_n = int(prior_cfg.MONITOR_EVERY_N)
        if self.monitor_topk <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MONITOR_TOPK must be positive.")
        if self.monitor_every_n <= 0:
            raise ValueError("MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N must be positive.")

        self._last_loss_stats: Dict[str, float] = {}
        self._debug_logged = False
        self._monitor_step = 0
        self.register_buffer("_graph_gp_support_sum", torch.zeros(self.num_classes, self.text_dim), persistent=False)
        self.register_buffer("_graph_gp_support_sq_sum", torch.zeros(self.num_classes, self.text_dim), persistent=False)
        self.register_buffer("_graph_gp_support_var_sum", torch.zeros(self.num_classes, self.text_dim), persistent=False)
        self.register_buffer("_graph_gp_support_count", torch.zeros(self.num_classes), persistent=False)
        self._graph_gp_split_id: Optional[int] = None
        self._graph_gp_support_ids: Optional[torch.Tensor] = None
        self._graph_gp_pseudo_unseen_ids: Optional[torch.Tensor] = None
        self._last_graph_gp_debug: Dict[str, torch.Tensor] = {}

    def _class_ids_tensor(self, class_ids, device: torch.device) -> torch.Tensor:
        """
        把 dataset 传入的 seen/unseen 类 id 转成全局类别 id tensor。

        Graph-GP 的 split 必须按全局类别 id 做，不能使用 local-output remap 后的类别编号。
        这里严格校验范围；如果没有 seen_class_ids，就直接报错，不做“全类都当 seen”的保底。
        """
        if class_ids is None:
            raise RuntimeError("Graph-GP requires dataset seen_class_ids.")
        if torch.is_tensor(class_ids):
            ids = class_ids.detach().to(device=device, dtype=torch.long).view(-1)
        else:
            ids = torch.as_tensor(list(class_ids), device=device, dtype=torch.long).view(-1)
        if ids.numel() == 0:
            raise RuntimeError("Graph-GP received empty seen_class_ids.")
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
            raise RuntimeError("Graph-GP requires epoch in loss kwargs.")
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
        在 CPU FP64 上解 Graph-GP 线性方程，再以 FP64 返回原 device。
        """
        if system.dim() != 2 or int(system.shape[0]) != int(system.shape[1]):
            raise RuntimeError(f"Graph-GP system must be square [S,S], got {tuple(system.shape)}.")
        if rhs.dim() != 2 or int(rhs.shape[0]) != int(system.shape[0]):
            raise RuntimeError(
                f"Graph-GP rhs must have shape [S,K] matching system, got {tuple(rhs.shape)}."
            )
        if system.device != rhs.device:
            raise RuntimeError("Graph-GP system and rhs must be on the same device before CPU solve.")
        output_device = rhs.device
        system_cpu = system.detach().to(device="cpu", dtype=torch.float64)
        rhs_cpu = rhs.detach().to(device="cpu", dtype=torch.float64)
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
            solution_cpu = torch.linalg.solve(system_cpu, rhs_cpu)
        elif hasattr(torch, "solve"):
            solution_cpu = torch.solve(rhs_cpu, system_cpu).solution
        else:
            raise RuntimeError("Graph-GP CPU solve requires torch.linalg.solve or torch.solve.")
        if not bool(torch.isfinite(solution_cpu).all().item()):
            raise RuntimeError("Graph-GP CPU solve produced NaN or Inf.")

        residual = (system_cpu.matmul(solution_cpu) - rhs_cpu).norm()
        relative_residual = residual / rhs_cpu.norm().clamp_min(1e-12)
        if float(relative_residual.item()) > 1e-8:
            raise RuntimeError(
                f"Graph-GP CPU solve relative residual is too large: {float(relative_residual.item()):.6e}."
            )
        return solution_cpu.to(device=output_device)

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
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        source = str(prior_cfg.GRAPH_GP_PRIOR_VAR_SOURCE).lower()
        if source == "unit":
            return torch.zeros_like(reference), {}
        if source == "constant":
            return torch.full_like(reference, float(prior_cfg.PRIOR_LOGVAR_CONST)), {}
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
            "unit / constant / dynamic_uncertainty."
        )

    def _graph_gp_class_priors(
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
            raise RuntimeError("Graph-GP is a training-time prior generator; eval prototype scoring is not implemented.")
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

        latent_dim = int(v_support.shape[1])
        joint_rhs = torch.cat((v_support, k_all_s.t()), dim=1)
        joint_solution = self._solve_graph_gp_system(system, joint_rhs)
        solved_v = joint_solution[:, :latent_dim]
        solved_k = joint_solution[:, latent_dim:]
        solve_dtype = joint_solution.dtype
        k_all_s_solve = k_all_s.to(dtype=solve_dtype)
        kernel_diag_solve = kernel.diag().to(dtype=solve_dtype)
        support_visual_var_solve = support_visual_var_dim.to(dtype=solve_dtype)
        prior_mu_solve = k_all_s_solve.matmul(solved_v)

        # predictive uncertainty 始终记录；dynamic_uncertainty 配置下也用于 prior_logvar。
        smoothing_coeff = solved_k.t()
        uncertainty_diag_solve = (
            kernel_diag_solve - (k_all_s_solve * smoothing_coeff).sum(dim=1)
        ).clamp_min(0.0)
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)
        var_weight = smoothing_coeff.clamp_min(0.0)
        fallback_weight = k_all_s_solve.clamp_min(0.0)
        var_weight_sum = var_weight.sum(dim=1, keepdim=True)
        fallback_sum = fallback_weight.sum(dim=1, keepdim=True)
        var_weight = torch.where(
            var_weight_sum > eps,
            var_weight / var_weight_sum.clamp_min(eps),
            fallback_weight / fallback_sum.clamp_min(eps),
        )
        visual_within_var_solve = var_weight.matmul(support_visual_var_solve).clamp_min(0.0)
        system_solve = system.to(dtype=solve_dtype)
        v_support_solve = v_support.to(dtype=solve_dtype)
        solve_residual = (
            (system_solve.matmul(solved_v) - v_support_solve).norm()
            / v_support_solve.norm().clamp_min(1e-12)
        )
        system_diag = system_solve.diag().abs().clamp_min(1e-12)
        system_diag_ratio = system_diag.max() / system_diag.min()

        prior_mu = prior_mu_solve.to(dtype=posterior_mu.dtype)
        uncertainty_diag = uncertainty_diag_solve.to(dtype=posterior_mu.dtype)
        visual_within_var = visual_within_var_solve.to(dtype=posterior_mu.dtype)
        smoothing_coeff = smoothing_coeff.to(dtype=posterior_mu.dtype)
        var_weight = var_weight.to(dtype=posterior_mu.dtype)

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

    def _graph_gp_energy_classification_loss(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        seen_class_ids,
        support_ids: torch.Tensor,
        pseudo_unseen_ids: torch.Tensor,
    ):
        prior_cfg = self.cfg.MODEL.GRAPH_PROB_PRIOR
        eps = float(self.cfg.MODEL.GRAPH_INPUT.EPS)

        class_ids = self._class_ids_tensor(seen_class_ids, device=posterior_mu.device)
        if class_ids.numel() < 2:
            raise RuntimeError("Graph-GP energy classification requires at least two seen classes.")
        distance = self._gaussian_kl_all_classes(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
        energy_distance = distance.index_select(1, class_ids)
        in_class_space = targets_global[:, None].eq(class_ids[None, :])
        if not bool(in_class_space.any(dim=1).all().item()):
            missing = targets_global[~in_class_space.any(dim=1)].detach().unique(sorted=True)
            raise RuntimeError(
                "Graph-GP energy classification requires every batch target to be in seen_class_ids; "
                f"missing global ids={missing.detach().cpu().tolist()}."
            )

        target_local = in_class_space.to(dtype=torch.long).argmax(dim=1)
        tau = posterior_mu.new_tensor(float(prior_cfg.GRAPH_GP_ENERGY_TAU)).clamp_min(eps)
        logits = -energy_distance / tau
        per_sample_loss = F.cross_entropy(logits, target_local, reduction="none")
        weights = torch.ones_like(per_sample_loss)
        pseudo_weight = float(prior_cfg.GRAPH_GP_PSEUDO_WEIGHT)
        if pseudo_weight != 1.0:
            pseudo_mask = self._class_membership_mask(targets_global, pseudo_unseen_ids)
            weights = torch.where(pseudo_mask, weights.new_full(weights.shape, pseudo_weight), weights)
        loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(eps)

        prob = F.softmax(logits, dim=-1)
        pred_local = prob.argmax(dim=-1)
        pred_global = class_ids.index_select(0, pred_local)
        hit = pred_global.eq(targets_global).to(dtype=posterior_mu.dtype)
        row = torch.arange(targets_global.numel(), device=targets_global.device)
        true_distance = energy_distance[row, target_local]
        wrong_distance = energy_distance.clone()
        wrong_distance[row, target_local] = float("inf")
        nearest_wrong = wrong_distance.min(dim=-1).values
        margin = nearest_wrong - true_distance
        entropy = -(prob * prob.clamp_min(eps).log()).sum(dim=-1).mean()

        stats = {
            "graph_prob_prior_graph_gp_energy_ce": float(loss.detach().item()),
            "graph_prob_prior_graph_gp_energy_tau": float(tau.detach().item()),
            "graph_prob_prior_graph_gp_energy_class_count": float(class_ids.numel()),
            "graph_prob_prior_graph_gp_energy_acc": float(hit.detach().mean().item()),
            "graph_prob_prior_graph_gp_energy_entropy": float(entropy.detach().item()),
            "graph_prob_prior_graph_gp_energy_true_kl_mean": float(true_distance.detach().mean().item()),
            "graph_prob_prior_graph_gp_energy_nearest_wrong_kl_mean": float(nearest_wrong.detach().mean().item()),
            "graph_prob_prior_graph_gp_energy_margin_mean": float(margin.detach().mean().item()),
            "graph_prob_prior_graph_gp_energy_margin_positive_ratio": float((margin.detach() > 0.0).float().mean().item()),
            "graph_prob_prior_graph_gp_energy_sample_weight_mean": float(weights.detach().mean().item()),
        }
        support_mask = self._class_membership_mask(targets_global, support_ids)
        pseudo_mask = self._class_membership_mask(targets_global, pseudo_unseen_ids)
        if bool(support_mask.any().item()):
            stats["graph_prob_prior_graph_gp_energy_support_seen_acc"] = float(hit[support_mask].detach().mean().item())
            stats["graph_prob_prior_graph_gp_energy_support_seen_ce"] = float(
                per_sample_loss[support_mask].detach().mean().item()
            )
            stats["graph_prob_prior_graph_gp_energy_support_seen_margin_mean"] = float(
                margin[support_mask].detach().mean().item()
            )
        if bool(pseudo_mask.any().item()):
            stats["graph_prob_prior_graph_gp_energy_pseudo_unseen_acc"] = float(hit[pseudo_mask].detach().mean().item())
            stats["graph_prob_prior_graph_gp_energy_pseudo_unseen_ce"] = float(
                per_sample_loss[pseudo_mask].detach().mean().item()
            )
            stats["graph_prob_prior_graph_gp_energy_pseudo_unseen_margin_mean"] = float(
                margin[pseudo_mask].detach().mean().item()
            )
        if bool(support_mask.any().item()) and bool(pseudo_mask.any().item()):
            stats["graph_prob_prior_graph_gp_energy_support_pseudo_acc_gap"] = (
                stats["graph_prob_prior_graph_gp_energy_support_seen_acc"]
                - stats["graph_prob_prior_graph_gp_energy_pseudo_unseen_acc"]
            )
            stats["graph_prob_prior_graph_gp_energy_support_pseudo_ce_gap"] = (
                stats["graph_prob_prior_graph_gp_energy_pseudo_unseen_ce"]
                - stats["graph_prob_prior_graph_gp_energy_support_seen_ce"]
            )

        debug = {
            "distance_shape": tuple(distance.shape),
            "energy_logits_shape": tuple(logits.shape),
            "energy_class_count": int(class_ids.numel()),
        }
        return loss, stats, debug, distance, per_sample_loss

    def forward(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        targets_global: torch.Tensor,
        class_attributes: Optional[torch.Tensor] = None,
        seen_class_ids=None,
        unseen_class_ids=None,
        epoch: Optional[int] = None,
        is_train: bool = True,
    ) -> torch.Tensor:
        if posterior_mu.dim() != 2 or posterior_mu.shape[1] != self.text_dim:
            raise RuntimeError(
                f"Graph-GP expects posterior mu [B,{self.text_dim}], got {tuple(posterior_mu.shape)}."
            )
        if tuple(posterior_logvar.shape) != tuple(posterior_mu.shape):
            raise RuntimeError(
                "Graph-GP expects posterior logvar shape {}, got {}.".format(
                    tuple(posterior_mu.shape),
                    tuple(posterior_logvar.shape),
                )
            )

        monitor_active = False
        if bool(is_train):
            self._monitor_step += 1
            monitor_active = self.monitor_enable and (self._monitor_step % self.monitor_every_n == 0)

        graph_inputs = self.graph_builder.prepare(
            targets_global=targets_global,
            class_attributes=class_attributes,
            device=posterior_mu.device,
            dtype=posterior_mu.dtype,
        )
        targets_global = graph_inputs["targets_global"]
        acc = graph_inputs["acc"]
        graph = graph_inputs["graph"]

        prior_mu, prior_logvar = self._graph_gp_class_priors(
            graph=graph,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            targets_global=targets_global,
            seen_class_ids=seen_class_ids,
            epoch=epoch,
            is_train=bool(is_train),
        )
        graph_gp_debug = self._last_graph_gp_debug
        loss, match_stats, debug_info, distance, sample_loss = self._graph_gp_energy_classification_loss(
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            targets_global=targets_global,
            seen_class_ids=seen_class_ids,
            support_ids=graph_gp_debug["support_ids"],
            pseudo_unseen_ids=graph_gp_debug["pseudo_unseen_ids"],
        )

        monitor_stats: Dict[str, float] = {}
        if monitor_active:
            monitor_stats.update(graph_health_monitor(graph, topk=self.monitor_topk))
            monitor_stats.update(
                false_high_pair_monitor(
                    acc,
                    graph,
                    threshold=0.9,
                    prefix="graph_prob_prior_monitor_false_high_graph_relation",
                )
            )
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
                    distance=distance,
                    sample_energy_loss=sample_loss,
                    posterior_mu=posterior_mu,
                    targets_global=targets_global,
                    seen_class_ids=seen_class_ids,
                    unseen_class_ids=unseen_class_ids,
                    topk=self.monitor_topk,
                )
            )
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
            debug_info.update(
                {
                    "graph_gp_support_count": int(graph_gp_debug["support_ids"].numel()),
                    "graph_gp_observed_support_count": int(graph_gp_debug["observed_support_ids"].numel()),
                }
            )

        posterior_var = posterior_logvar.exp()
        prior_var = prior_logvar.exp()
        self._last_loss_stats = {
            **match_stats,
            **monitor_stats,
            "graph_prob_prior_posterior_mu_norm": float(posterior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_prior_mu_norm": float(prior_mu.detach().norm(dim=-1).mean().item()),
            "graph_prob_prior_posterior_var_mean": float(posterior_var.detach().mean().item()),
            "graph_prob_prior_prior_var_mean": float(prior_var.detach().mean().item()),
        }

        if bool(self.cfg.MODEL.GRAPH_PROB_PRIOR.DEBUG) and not self._debug_logged:
            print(
                "[GRAPH-GP-DEBUG] posterior_mu={} posterior_logvar={} graph={} "
                "prior_mu={} prior_logvar={} debug={} loss={:.6f}".format(
                    tuple(posterior_mu.shape),
                    tuple(posterior_logvar.shape),
                    tuple(graph.shape),
                    tuple(prior_mu.shape),
                    tuple(prior_logvar.shape),
                    debug_info,
                    float(loss.detach().item()),
                )
            )
            self._debug_logged = True
        return loss
