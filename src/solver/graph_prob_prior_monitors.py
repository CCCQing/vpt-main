#!/usr/bin/env python3
"""
GraphProbPrior 温度/距离尺度监测工具。

这个文件只负责“把中间张量转成日志标量”，不参与 loss 计算，也不改变梯度。
核心用途有两个：
1. 短跑诊断脚本复用这里，统计一小段训练中各温度处理的真实数值范围；
2. 日常训练开启 MONITOR_ENABLE 后，也复用这里把关键分布写进 loss stats。

字段命名规则：
- *_finite_ratio: 有限数值比例；小于 1 说明出现 inf/nan。
- *_min/max/mean/std: 原始张量的基础统计。
- *_q05/q50/q95: 原始张量 5%/50%/95% 分位数，比 min/max 更能反映典型范围。
- *_entropy_mean: 概率分布每行熵的平均值；越小越尖，越大越平。
- *_entropy_norm_mean: 熵除以 log(C) 后的归一化熵；接近 0 表示近似 one-hot，接近 1 表示近似均匀。
- *_top1_mean: 每行最大概率的平均值；越大说明分布越集中在第一名。
- *_top{k}_mass_mean: 每行 top-k 概率和的平均值；越大说明概率集中在少数类别。
- *_true_mean: 每行真实类别位置的概率平均值；只在传入 true_index 时记录。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


def _as_float(value: torch.Tensor) -> float:
    """把 0 维 tensor 安全转成 Python float，便于写入 json/csv/log。"""
    return float(value.detach().float().cpu().item())


def _finite_flatten(x: torch.Tensor) -> torch.Tensor:
    """取出 tensor 中所有有限值并展平成一维；当前保留给后续扩展使用。"""
    if not torch.is_tensor(x):
        return torch.empty(0)
    values = x.detach().float().reshape(-1)
    return values[torch.isfinite(values)]


def tensor_stats(prefix: str, x: torch.Tensor) -> Dict[str, float]:
    """
    统计任意实值张量的数值范围。

    这个函数用于监测“温度处理之前”的原始量，例如：
    - graph: 类别语义相似度矩阵；
    - top_values: graph[y] top-k 后进入 TAU_ACC softmax 的 logits；
    - distance: KL(q_i || p_c) 距离；
    - sym_kl: prior-prior Gaussian 对称 KL；
    - pair_dist: MMD 中 posterior/prior 样本两两平方距离；
    - kernel: MMD 中 RBF kernel 值。

    输出字段含义：
    - {prefix}_finite_ratio: 有限值比例；如果小于 1，说明这个张量已经有数值异常。
    - {prefix}_min / max: 极值；用于发现爆炸或异常离群点。
    - {prefix}_mean / std: 均值和标准差；用于判断整体尺度。
    - {prefix}_q05 / q50 / q95: 分位数；用于判断大多数值落在哪个范围。

    读法：
    - 如果 q95 - q05 很大，说明该位置 logits/距离跨度很大，softmax 温度可能需要更大；
    - 如果 std 很小，说明该位置本身区分度弱，温度再小也可能只是在放大噪声。
    """
    if not torch.is_tensor(x):
        return {}
    values = x.detach().float().reshape(-1)
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    total = max(int(values.numel()), 1)
    stats = {f"{prefix}_finite_ratio": float(finite.numel()) / float(total)}
    if finite.numel() == 0:
        return stats

    stats.update(
        {
            f"{prefix}_min": _as_float(finite.min()),
            f"{prefix}_max": _as_float(finite.max()),
            f"{prefix}_mean": _as_float(finite.mean()),
            f"{prefix}_std": _as_float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0,
        }
    )
    if finite.numel() >= 3:
        qs = torch.quantile(
            finite,
            torch.tensor([0.05, 0.50, 0.95], device=finite.device, dtype=finite.dtype),
        )
        stats.update(
            {
                f"{prefix}_q05": _as_float(qs[0]),
                f"{prefix}_q50": _as_float(qs[1]),
                f"{prefix}_q95": _as_float(qs[2]),
            }
        )
    else:
        stats.update(
            {
                f"{prefix}_q05": _as_float(finite.min()),
                f"{prefix}_q50": _as_float(finite.mean()),
                f"{prefix}_q95": _as_float(finite.max()),
            }
        )
    return stats


def probability_stats(
    prefix: str,
    prob: torch.Tensor,
    true_index: Optional[torch.Tensor] = None,
    topk: int = 5,
) -> Dict[str, float]:
    """
    统计概率分布的尖锐程度和真实类别质量。

    输入 prob 通常是某个 softmax 之后的矩阵，形状类似 [N, C]：
    - N 可以是类别数 C，也可以是 batch size B；
    - 最后一维 C 表示一行概率分布。

    输出字段含义：
    - {prefix}_prob_*: 对概率值本身做 tensor_stats，例如概率是否出现 nan/inf。
    - {prefix}_entropy_mean: 行熵平均值。熵越低，分布越尖；熵越高，分布越平。
    - {prefix}_entropy_norm_mean: 归一化熵，范围大致在 [0, 1]。
      接近 0: 几乎 one-hot；接近 1: 接近均匀分布。
    - {prefix}_top1_mean: 每行最大概率平均值。
      例如接近 0.9 表示大部分质量集中在第一名。
    - {prefix}_top{k}_mass_mean: 每行 top-k 总概率。
      例如 top5_mass 接近 1 表示前 5 个类别几乎吃掉全部概率。
    - {prefix}_true_mean: 真实类别位置的概率均值。
      只有 true_index 形状与行数一致时才会记录。

    读法：
    - top1 高、entropy_norm 低：温度偏小或原始 logits/距离跨度太大；
    - top1 低、entropy_norm 接近 1：温度偏大或原始信号区分度弱；
    - true_mean 低：真实类别在该分布里没有得到足够概率质量。
    """
    if not torch.is_tensor(prob) or prob.dim() < 2:
        return {}
    p = prob.detach().float()
    stats = tensor_stats(f"{prefix}_prob", p)
    class_count = int(p.shape[-1])
    if class_count <= 0:
        return stats

    # 监测函数不假设输入严格归一化。这里先截断负值，再按行重新归一化，
    # 避免上游微小数值误差影响熵和 top-k mass 的可读性。
    p = p.clamp_min(0.0)
    row_sum = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = p / row_sum
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
    max_entropy = math.log(max(class_count, 2))
    topk = max(1, min(int(topk), class_count))
    top_values = torch.topk(p, k=topk, dim=-1).values

    stats.update(
        {
            f"{prefix}_entropy_mean": _as_float(entropy.mean()),
            f"{prefix}_entropy_norm_mean": _as_float((entropy / max_entropy).mean()),
            f"{prefix}_top1_mean": _as_float(top_values[..., 0].mean()),
            f"{prefix}_top{topk}_mass_mean": _as_float(top_values.sum(dim=-1).mean()),
        }
    )
    if true_index is not None and torch.is_tensor(true_index):
        idx = true_index.detach().to(device=p.device, dtype=torch.long).view(-1)
        if idx.numel() == p.shape[0] and bool((idx >= 0).all().item()) and bool((idx < class_count).all().item()):
            true_prob = p.gather(dim=-1, index=idx[:, None]).squeeze(-1)
            stats[f"{prefix}_true_mean"] = _as_float(true_prob.mean())
    return stats


def graph_neighbor_monitor(
    graph: torch.Tensor,
    tau_graph: float,
    topk: int = 5,
) -> Dict[str, float]:
    """
    监测 TAU_GRAPH 对 graph 邻居聚合权重的影响。

    对应 GraphProbPrior 里的：
        neighbor_weight = softmax(graph / TAU_GRAPH)
        neighbor_bank = neighbor_weight @ bank

    记录两类量：
    1. graph_prob_prior_monitor_tau_graph_raw_*
       graph 原始相似度矩阵的数值范围。
    2. graph_prob_prior_monitor_tau_graph_neighbor_*
       softmax 后的邻居权重分布。

    读法：
    - neighbor_entropy_norm 低、neighbor_top1 高：
      每个类别主要从极少数邻居借信息，TAU_GRAPH 偏小或 graph 差距过大。
    - neighbor_entropy_norm 高、neighbor_top1 低：
      每个类别几乎平均借很多类的信息，TAU_GRAPH 偏大或 graph 差距太小。
    """
    stats = tensor_stats("graph_prob_prior_monitor_tau_graph_raw", graph)
    neighbor_weight = F.softmax(graph / float(tau_graph), dim=-1)
    stats.update(
        probability_stats(
            "graph_prob_prior_monitor_tau_graph_neighbor",
            neighbor_weight,
            topk=topk,
        )
    )
    return stats


def semantic_target_monitor(
    graph: torch.Tensor,
    targets_global: torch.Tensor,
    tau_acc: float,
    graph_topk: int,
    target_mix_alpha: float,
    num_classes: int,
    eps: float,
    topk: int = 5,
) -> Dict[str, float]:
    """
    监测 TAU_ACC 和 TARGET_MIX_ALPHA 构造监督 target 的效果。

    对应 SemanticGraphBuilder.build_target() 的逻辑：
        rows = graph[y]
        masked = 只保留 TOPK 个语义近邻，其余置为 -inf
        target_sem = softmax(masked / TAU_ACC)
        target = (1 - TARGET_MIX_ALPHA) * onehot + TARGET_MIX_ALPHA * target_sem

    记录三类量：
    1. graph_prob_prior_monitor_tau_acc_topk_logits_*
       进入 TAU_ACC softmax 的 top-k graph logits 的原始范围。
    2. graph_prob_prior_monitor_tau_acc_target_sem_*
       纯语义近邻 soft target 的分布形状。
    3. graph_prob_prior_monitor_tau_acc_target_*
       与 one-hot 混合后的最终监督 target 分布。

    读法：
    - target_sem_entropy_norm 低：
      语义近邻 target 很尖，TAU_ACC 可能偏小。
    - target_entropy_norm 明显低于 target_sem_entropy_norm：
      TARGET_MIX_ALPHA 较小，one-hot 真类成分占主导。
    - target_true_mean 越高：
      最终监督 target 给真类的概率质量越大。
    """
    rows = graph.index_select(0, targets_global)
    k = max(1, min(int(graph_topk), int(num_classes)))
    top_values, top_idx = torch.topk(rows, k=k, dim=-1)
    masked = rows.new_full(rows.shape, float("-inf"))
    masked.scatter_(dim=-1, index=top_idx, src=top_values)
    target_sem = F.softmax(masked / float(tau_acc), dim=-1)
    onehot = F.one_hot(targets_global, num_classes=int(num_classes)).to(dtype=target_sem.dtype)
    target = (1.0 - float(target_mix_alpha)) * onehot + float(target_mix_alpha) * target_sem
    target = target.clamp_min(float(eps))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(float(eps))

    stats = tensor_stats("graph_prob_prior_monitor_tau_acc_topk_logits", top_values)
    stats.update(
        probability_stats(
            "graph_prob_prior_monitor_tau_acc_target_sem",
            target_sem,
            true_index=targets_global,
            topk=topk,
        )
    )
    stats.update(
        probability_stats(
            "graph_prob_prior_monitor_tau_acc_target",
            target,
            true_index=targets_global,
            topk=topk,
        )
    )
    return stats


def latent_matching_monitor(
    distance: torch.Tensor,
    latent_prob: torch.Tensor,
    targets_global: Optional[torch.Tensor],
    topk: int = 5,
) -> Dict[str, float]:
    """
    监测 TAU_LATENT 对 all-class latent matching 的影响。

    对应 _samplewise_latent_matching_loss()：
        distance[i, c] = KL(q_i || p_c)
        latent_prob[i] = softmax(-distance[i] / TAU_LATENT)

    记录两类量：
    1. graph_prob_prior_monitor_tau_latent_distance_*
       样本 posterior 到所有类别 prior 的 KL 距离范围。
    2. graph_prob_prior_monitor_tau_latent_*
       softmax(-KL / TAU_LATENT) 后的类别分布。

    读法：
    - distance_q95 很大且 latent_entropy_norm 很低：
      KL 距离跨度很大，TAU_LATENT 可能太小，分布接近 one-hot。
    - distance_std 很小且 latent_entropy_norm 很高：
      所有类距离差不多，latent matching 区分度弱。
    - latent_true_mean 低：
      样本 posterior 并没有把真类 prior 排到较高概率。
    """
    stats = tensor_stats("graph_prob_prior_monitor_tau_latent_distance", distance)
    stats.update(
        probability_stats(
            "graph_prob_prior_monitor_tau_latent",
            latent_prob,
            true_index=targets_global,
            topk=topk,
        )
    )
    return stats


def true_class_kl_monitor(true_kl_per_sample: torch.Tensor) -> Dict[str, float]:
    """
    监测 true_class_kl 模式下的逐样本真类 KL。

    对应：
        KL(q_i || p_{y_i})

    输出字段：
    - graph_prob_prior_monitor_true_class_kl_mean/std/q05/q50/q95 等。

    读法：
    - true_class_kl_mean 很大：
      posterior 和真类 prior 距离较远，可能是 LOSS_WEIGHT 太小、prior 不合理或 posterior 尺度异常。
    - true_class_kl_q95 远大于 q50：
      少数样本 KL 很大，需要检查是否有类别/样本异常。
    """
    return tensor_stats("graph_prob_prior_monitor_true_class_kl", true_kl_per_sample)


def relation_monitor(sym_kl: torch.Tensor, pred_rel: torch.Tensor, topk: int = 5) -> Dict[str, float]:
    """
    监测 TAU_PRIOR 对 prior-prior 关系分布的影响。

    对应 _relation_regularization()：
        sym_kl[c, d] = 0.5 * (KL(P_c || P_d) + KL(P_d || P_c))
        pred_rel[c] = softmax(-sym_kl[c] / TAU_PRIOR)

    记录两类量：
    1. graph_prob_prior_monitor_tau_prior_symkl_*
       类别 prior Gaussian 两两对称 KL 的原始距离范围。
    2. graph_prob_prior_monitor_tau_prior_*
       用 TAU_PRIOR 转成的 prior 关系概率分布。

    读法：
    - symkl_q95 很大、tau_prior_entropy_norm 很低：
      prior 关系分布过尖，TAU_PRIOR 可能偏小。
    - symkl_std 很小、tau_prior_entropy_norm 高：
      prior 之间距离差别小，relation regularization 难以提供结构区分。
    """
    stats = tensor_stats("graph_prob_prior_monitor_tau_prior_symkl", sym_kl)
    stats.update(
        probability_stats(
            "graph_prob_prior_monitor_tau_prior",
            pred_rel,
            topk=topk,
        )
    )
    return stats


def mmd_monitor(pair_dist: torch.Tensor, kernel: torch.Tensor, sigma: Optional[float] = None) -> Dict[str, float]:
    """
    监测 MMD_SIGMA 对 RBF-MMD kernel 的影响。

    对应 class_aggregate_mmd：
        pair_dist = ||posterior_sample - prior_sample||^2
        kernel = exp(-pair_dist / (2 * MMD_SIGMA^2))

    记录两类量：
    1. graph_prob_prior_monitor_mmd_pair_dist_*
       posterior/prior latent 样本两两平方距离范围。
    2. graph_prob_prior_monitor_mmd_kernel_*
       RBF kernel 值范围。

    读法：
    - kernel_mean 接近 0：
      sigma 相对距离太小，posterior/prior 样本几乎都被认为“不相似”，MMD 梯度可能弱或不稳定。
    - kernel_mean 接近 1：
      sigma 相对距离太大，样本几乎都被认为“很相似”，MMD 区分度弱。
    - 理想情况通常是 kernel 有一定分散度，而不是全部挤在 0 或 1。
    """
    stats = tensor_stats("graph_prob_prior_monitor_mmd_pair_dist", pair_dist)
    stats.update(tensor_stats("graph_prob_prior_monitor_mmd_kernel", kernel))
    if torch.is_tensor(kernel):
        kernel_flat = kernel.detach().float().reshape(-1)
        finite = kernel_flat[torch.isfinite(kernel_flat)]
        if finite.numel() > 0:
            stats["graph_prob_prior_monitor_mmd_kernel_saturation_low_ratio"] = _as_float((finite < 1e-4).float().mean())
            stats["graph_prob_prior_monitor_mmd_kernel_saturation_high_ratio"] = _as_float((finite > 0.99).float().mean())
    if sigma is not None:
        stats["graph_prob_prior_monitor_mmd_sigma"] = float(sigma)
    return stats


def _offdiag_values(matrix: torch.Tensor) -> torch.Tensor:
    """取方阵非对角线元素，用于统计类间关系而不让自相似对角线污染结果。"""
    if not torch.is_tensor(matrix) or matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        return torch.empty(0)
    n = int(matrix.shape[0])
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix.detach().float()[mask]


def _row_entropy_stats(prefix: str, prob: torch.Tensor) -> Dict[str, float]:
    """统计概率矩阵每行熵的分位数，判断分布是大锅饭还是过尖。"""
    if not torch.is_tensor(prob) or prob.dim() != 2:
        return {}
    p = prob.detach().float().clamp_min(0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
    norm_entropy = entropy / math.log(max(int(p.shape[-1]), 2))
    stats = tensor_stats(f"{prefix}_entropy_norm", norm_entropy)
    stats[f"{prefix}_entropy_norm_mean"] = _as_float(norm_entropy.mean())
    return stats


def _gini(x: torch.Tensor) -> float:
    """Gini 系数：越大表示 hubness 越集中在少数类别。"""
    values = x.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 0.0
    finite = finite.clamp_min(0.0).sort().values
    total = finite.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    n = finite.numel()
    index = torch.arange(1, n + 1, device=finite.device, dtype=finite.dtype)
    return _as_float((2.0 * (index * finite).sum() / (float(n) * total)) - (float(n) + 1.0) / float(n))


def _effective_rank(x: torch.Tensor, center: bool = True) -> float:
    """
    effective rank：看一组向量是否塌缩到低维子空间。
    center=True 时先去掉整体均值，避免共同偏移影响 rank 判断。
    """
    if not torch.is_tensor(x) or x.dim() != 2:
        return 0.0
    values = x.detach().float()
    if center:
        values = values - values.mean(dim=0, keepdim=True)
    try:
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "svdvals"):
            singular = torch.linalg.svdvals(values)
        else:
            singular = torch.svd(values).S
    except RuntimeError:
        return 0.0
    total = singular.sum()
    if float(total.item()) <= 1e-12:
        return 0.0
    p = singular / total.clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return _as_float(entropy.exp())


def _pairwise_symkl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """计算所有类别 diagonal Gaussian prior 的两两对称 KL，返回 [C,C]。"""
    mu = mu.detach().float()
    logvar = logvar.detach().float()
    var = logvar.exp()
    inv_var = torch.exp(-logvar)
    dim = int(mu.shape[1])
    logvar_sum = logvar.sum(dim=-1)
    log_term = logvar_sum[None, :] - logvar_sum[:, None]
    var_term = var.matmul(inv_var.t())
    mu_sq = mu.pow(2)
    mu_sq_over_var = mu_sq.matmul(inv_var.t())
    cross = mu.matmul((mu * inv_var).t())
    self_quad = (mu_sq * inv_var).sum(dim=-1)
    mean_term = mu_sq_over_var - 2.0 * cross + self_quad[None, :]
    kl = 0.5 * (log_term + var_term + mean_term - float(dim))
    return 0.5 * (kl + kl.t()).clamp_min(0.0)


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """无 scipy 版本 Spearman：先转 rank，再算 Pearson。"""
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        return 0.0
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 3:
        return 0.0
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return _as_float(rx.dot(ry) / denom)


def graph_health_monitor(
    graph: torch.Tensor,
    tau_graph: float,
    topk: int = 5,
    exclude_diag: bool = True,
) -> Dict[str, float]:
    """
    Graph health：看语义图是不是大锅饭、是不是 hub 很严重、是不是所有类都互相很像。
    这些指标直接作用于图本身和 softmax(graph/tau) 后的邻居分布。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(graph) or graph.dim() != 2 or graph.shape[0] != graph.shape[1]:
        return stats
    graph = graph.detach().float()
    class_count = int(graph.shape[0])
    offdiag = _offdiag_values(graph) if exclude_diag else graph.reshape(-1)
    stats.update(tensor_stats("graph_prob_prior_monitor_graph_offdiag", offdiag))
    if offdiag.numel() > 0:
        stats["graph_prob_prior_monitor_graph_max_offdiag"] = _as_float(offdiag.max())
        directed_09 = (offdiag > 0.9).float().sum()
        directed_08 = (offdiag > 0.8).float().sum()
        stats["graph_prob_prior_monitor_graph_pairs_gt_0_9_directed"] = _as_float(directed_09)
        stats["graph_prob_prior_monitor_graph_pairs_gt_0_8_directed"] = _as_float(directed_08)
        stats["graph_prob_prior_monitor_graph_pairs_gt_0_9_undirected"] = float(int(directed_09.item()) // 2)
        stats["graph_prob_prior_monitor_graph_pairs_gt_0_8_undirected"] = float(int(directed_08.item()) // 2)

    neighbor = F.softmax(graph / float(tau_graph), dim=-1)
    diag = torch.diagonal(neighbor)
    stats.update(tensor_stats("graph_prob_prior_monitor_neighbor_self_mass", diag))
    stats.update(_row_entropy_stats("graph_prob_prior_monitor_neighbor", neighbor))
    stats.update(probability_stats("graph_prob_prior_monitor_neighbor", neighbor, topk=topk))

    k = max(1, min(int(topk), class_count - 1))
    logits = graph.clone()
    idx = torch.arange(class_count, device=graph.device)
    logits[idx, idx] = float("-inf")
    top_idx = torch.topk(logits, k=k, dim=-1).indices
    top_mask = torch.zeros_like(graph, dtype=torch.bool)
    top_mask.scatter_(1, top_idx, True)
    mutual = top_mask & top_mask.t()
    denom = top_mask.float().sum().clamp_min(1.0)
    stats[f"graph_prob_prior_monitor_neighbor_mutual_top{k}_ratio"] = _as_float(mutual.float().sum() / denom)
    hubness = top_mask.float().sum(dim=0)
    stats.update(tensor_stats("graph_prob_prior_monitor_neighbor_hubness", hubness))
    stats["graph_prob_prior_monitor_neighbor_hubness_gini"] = _gini(hubness)
    return stats


def prior_health_monitor(
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    logvar_min: Optional[float] = None,
    logvar_max: Optional[float] = None,
    topk: int = 5,
    overlap_margin: float = 0.0,
) -> Dict[str, float]:
    """
    Prior health：看每个类别的高斯 prior 是否塌缩、方差是否异常、不同类别分布是否重叠。
    overlap 使用球形半径近似，因此命名为 risk，不当作严格重叠概率。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(prior_mu) or not torch.is_tensor(prior_logvar):
        return stats
    if prior_mu.dim() != 2 or prior_logvar.shape != prior_mu.shape:
        return stats
    mu = prior_mu.detach().float()
    logvar = prior_logvar.detach().float()
    var = logvar.exp()
    class_count = int(mu.shape[0])
    stats.update(tensor_stats("graph_prob_prior_monitor_prior_mu_norm", mu.norm(dim=-1)))
    stats.update(tensor_stats("graph_prob_prior_monitor_prior_var", var))
    stats.update(tensor_stats("graph_prob_prior_monitor_prior_logvar", logvar))
    if logvar_min is not None:
        stats["graph_prob_prior_monitor_prior_logvar_clamp_min_ratio"] = _as_float((logvar <= float(logvar_min) + 1e-6).float().mean())
    if logvar_max is not None:
        stats["graph_prob_prior_monitor_prior_logvar_clamp_max_ratio"] = _as_float((logvar >= float(logvar_max) - 1e-6).float().mean())

    pair_dist = torch.cdist(mu, mu, p=2)
    off_pair_dist = _offdiag_values(pair_dist)
    stats.update(tensor_stats("graph_prob_prior_monitor_prior_pair_mu_dist", off_pair_dist))
    mu_norm = F.normalize(mu, p=2, dim=-1, eps=1e-12)
    cosine = mu_norm.matmul(mu_norm.t())
    off_cos = _offdiag_values(cosine)
    if off_cos.numel() > 0:
        stats["graph_prob_prior_monitor_prior_pair_cosine_max_offdiag"] = _as_float(off_cos.max())
        stats["graph_prob_prior_monitor_prior_pair_cosine_pairs_gt_0_9"] = _as_float((off_cos > 0.9).float().sum() / 2.0)
        stats["graph_prob_prior_monitor_prior_center_collapse_score"] = _as_float(off_cos.mean())

    symkl = _pairwise_symkl(mu, logvar)
    off_symkl = _offdiag_values(symkl)
    stats.update(tensor_stats("graph_prob_prior_monitor_prior_symkl", off_symkl))
    if off_symkl.numel() > 0:
        stats["graph_prob_prior_monitor_prior_symkl_min_offdiag"] = _as_float(off_symkl.min())

    radius = var.mean(dim=-1).clamp_min(1e-12).sqrt()
    risk = pair_dist < (radius[:, None] + radius[None, :] + float(overlap_margin))
    if class_count > 1:
        eye = torch.eye(class_count, dtype=torch.bool, device=mu.device)
        stats["graph_prob_prior_monitor_prior_overlap_risk_rate"] = _as_float(risk[~eye].float().mean())
    stats["graph_prob_prior_monitor_prior_effective_rank"] = _effective_rank(mu, center=True)
    return stats


def residual_anchor_prior_monitor(
    residual_attr: torch.Tensor,
    anchor: torch.Tensor,
    prior_mu: torch.Tensor,
    delta: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    graph: Optional[torch.Tensor] = None,
    positive_weight: Optional[torch.Tensor] = None,
    threshold: float = 0.9,
) -> Dict[str, float]:
    """
    residual-anchor prior 专用监测。

    这组指标回答三个问题：
    1. 312 维标准化属性残差是否数值健康；
    2. anchor_head 生成的类别锚点是否已经把类别拉开；
    3. delta_head 的小修正是否过强，是否又把空间推回自由 MLP。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(residual_attr) or not torch.is_tensor(anchor) or not torch.is_tensor(prior_mu):
        return stats
    residual = residual_attr.detach().float()
    anchor = anchor.detach().float()
    prior = prior_mu.detach().float()
    if residual.dim() != 2 or anchor.dim() != 2 or prior.dim() != 2:
        return stats

    # residual_attr_norm_*：看标准化后的 312 维残差有没有整体过大或过小。
    stats.update(tensor_stats("graph_prob_prior_monitor_residual_attr_norm", residual.norm(dim=-1)))

    # anchor_pair_cosine_*：只看 anchor_head 输出，不混入 delta，用来判断主锚点是否已经缓解 collapse。
    anchor_cos = F.normalize(anchor, p=2, dim=-1, eps=1e-12).matmul(
        F.normalize(anchor, p=2, dim=-1, eps=1e-12).t()
    )
    anchor_off = _offdiag_values(anchor_cos)
    if anchor_off.numel() > 0:
        stats.update(tensor_stats("graph_prob_prior_monitor_anchor_pair_cosine", anchor_off))
        stats["graph_prob_prior_monitor_anchor_pairs_gt_0_9"] = _as_float(
            (anchor_off > float(threshold)).float().sum() / 2.0
        )

    # prior_pair_cosine_*：看最终 prior_mu 是否仍然塌缩。prior_health_monitor 也会记一部分，
    # 这里补 q05/q50/q95，方便和 anchor_pair_cosine 直接对比。
    prior_cos = F.normalize(prior, p=2, dim=-1, eps=1e-12).matmul(
        F.normalize(prior, p=2, dim=-1, eps=1e-12).t()
    )
    prior_off = _offdiag_values(prior_cos)
    if prior_off.numel() > 0:
        stats.update(tensor_stats("graph_prob_prior_monitor_prior_pair_cosine", prior_off))

    # prior_anchor_cos_*：看最终 prior_mu 是否仍沿着 anchor 方向；过低说明 delta 修正覆盖了主锚点。
    if anchor.shape == prior.shape:
        anchor_n = F.normalize(anchor, p=2, dim=-1, eps=1e-12)
        prior_n = F.normalize(prior, p=2, dim=-1, eps=1e-12)
        stats.update(tensor_stats("graph_prob_prior_monitor_prior_anchor_cos", (anchor_n * prior_n).sum(dim=-1)))

    if delta is not None and torch.is_tensor(delta) and delta.shape == anchor.shape:
        d = delta.detach().float()
        delta_norm = d.norm(dim=-1)
        anchor_norm = anchor.norm(dim=-1).clamp_min(1e-12)
        # prior_delta_to_anchor_ratio：越大说明 correction 越像自由生成器；提示词期望它只是小修正。
        stats.update(tensor_stats("graph_prob_prior_monitor_prior_delta_norm", delta_norm))
        stats.update(tensor_stats("graph_prob_prior_monitor_prior_delta_to_anchor_ratio", delta_norm / anchor_norm))

    if context is not None and torch.is_tensor(context) and context.shape == anchor.shape:
        ctx = context.detach().float()
        stats.update(tensor_stats("graph_prob_prior_monitor_context_anchor_cos", (
            F.normalize(ctx, p=2, dim=-1, eps=1e-12) * F.normalize(anchor, p=2, dim=-1, eps=1e-12)
        ).sum(dim=-1)))

    if graph is not None and torch.is_tensor(graph) and graph.shape == prior_cos.shape:
        graph_det = graph.detach().float().to(device=prior.device)
        stats["graph_prob_prior_monitor_graph_pos_prior_relation_spearman"] = _spearman_corr(
            _offdiag_values(graph_det),
            _offdiag_values(prior_cos),
        )
        offdiag = ~torch.eye(int(graph_det.shape[0]), dtype=torch.bool, device=graph_det.device)
        false_high = (graph_det > float(threshold)) & offdiag
        if bool(false_high.any().item()):
            stats["graph_prob_prior_monitor_false_high_prior_relation_still_gt_0_9"] = _as_float(
                (prior_cos[false_high] > float(threshold)).float().sum()
            )

    if positive_weight is not None and torch.is_tensor(positive_weight) and positive_weight.shape == prior_cos.shape:
        pos_mask = positive_weight.detach().to(device=prior.device) > 0.0
        eye = torch.eye(int(pos_mask.shape[0]), dtype=torch.bool, device=prior.device)
        pos_mask = pos_mask & ~eye
        nonpos_mask = ~pos_mask & ~eye
        if bool(pos_mask.any().item()):
            stats["graph_prob_prior_monitor_true_neighbor_preservation_mean"] = _as_float(prior_cos[pos_mask].mean())
        if bool(nonpos_mask.any().item()):
            hard_values = prior_cos[nonpos_mask]
            stats.update(tensor_stats("graph_prob_prior_monitor_hardneg_prior_cos", hard_values))
            stats["graph_prob_prior_monitor_hardneg_violate_rate"] = _as_float(
                (hard_values > float(threshold)).float().mean()
            )
    return stats


def graph_prior_geometry_monitor(
    d_norm: torch.Tensor,
    dist_mu: torch.Tensor,
    radius: torch.Tensor,
    prior_mu: torch.Tensor,
    graph: Optional[torch.Tensor] = None,
    top_indices: Optional[torch.Tensor] = None,
    margin_min: float = 0.1,
) -> Dict[str, float]:
    """
    prior distribution geometry 监测。

    d_norm 是提示词里的 D_cd：均值距离除以类别分布半径之和。
    这些指标用来判断 prior mean 是否整体分散，以及 graph top-k 邻居是否被推得过远或过近。
    """
    stats: Dict[str, float] = {}
    if not all(torch.is_tensor(x) for x in (d_norm, dist_mu, radius, prior_mu)):
        return stats
    if d_norm.dim() != 2 or d_norm.shape[0] != d_norm.shape[1]:
        return stats
    n = int(d_norm.shape[0])
    eye = torch.eye(n, dtype=torch.bool, device=d_norm.device)
    off_d = d_norm.detach().float()[~eye]
    if off_d.numel() > 0:
        stats.update(tensor_stats("graph_prob_prior_monitor_geom_prior_dist", off_d))
        stats["graph_prob_prior_monitor_geom_pairs_lt_1_ratio"] = _as_float((off_d < 1.0).float().mean())
    stats.update(tensor_stats("graph_prob_prior_monitor_geom_mu_dist", dist_mu.detach().float()[~eye]))
    stats.update(tensor_stats("graph_prob_prior_monitor_geom_radius", radius.detach().float()))

    prior = prior_mu.detach().float()
    prior_cos = F.normalize(prior, p=2, dim=-1, eps=1e-12).matmul(
        F.normalize(prior, p=2, dim=-1, eps=1e-12).t()
    )
    off_cos = prior_cos[~eye]
    if off_cos.numel() > 0:
        stats.update(tensor_stats("graph_prob_prior_monitor_geom_prior_cos", off_cos))
        stats["graph_prob_prior_monitor_geom_pairs_cos_gt_0_9"] = _as_float((off_cos > 0.9).float().sum() / 2.0)

    if top_indices is not None and torch.is_tensor(top_indices):
        row = torch.arange(n, device=d_norm.device)[:, None]
        top_d = d_norm.detach().float()[row, top_indices.to(device=d_norm.device)]
        stats.update(tensor_stats("graph_prob_prior_monitor_geom_topk_dist", top_d))
        stats["graph_prob_prior_monitor_geom_topk_boundary_violate_ratio"] = _as_float(
            (top_d < (1.0 + float(margin_min))).float().mean()
        )
        if graph is not None and torch.is_tensor(graph) and graph.shape == d_norm.shape:
            graph_top = graph.detach().float().to(device=d_norm.device)[row, top_indices.to(device=d_norm.device)]
            stats["graph_prob_prior_monitor_geom_topk_graph_prior_corr"] = _spearman_corr(
                graph_top.reshape(-1),
                -top_d.reshape(-1),
            )
    return stats


def _index_tensor(ids, device: torch.device) -> torch.Tensor:
    """把 list/tuple/numpy/tensor 形式的类别 id 统一成 long tensor。"""
    if ids is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if torch.is_tensor(ids):
        out = ids.detach().to(device=device, dtype=torch.long).view(-1)
    else:
        out = torch.as_tensor(ids, device=device, dtype=torch.long).view(-1)
    return out[out >= 0]


def seen_unseen_prior_monitor(
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    seen_class_ids,
    unseen_class_ids,
) -> Dict[str, float]:
    """
    Seen/unseen prior 监测：检查 unseen prior 是否系统性更靠近 seen prior。
    只有调用方提供 seen/unseen split 时才会产出指标。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(prior_mu) or not torch.is_tensor(prior_logvar):
        return stats
    if prior_mu.dim() != 2 or prior_logvar.shape != prior_mu.shape:
        return stats
    device = prior_mu.device
    class_count = int(prior_mu.shape[0])
    seen = _index_tensor(seen_class_ids, device)
    unseen = _index_tensor(unseen_class_ids, device)
    seen = seen[(seen >= 0) & (seen < class_count)]
    unseen = unseen[(unseen >= 0) & (unseen < class_count)]
    if seen.numel() == 0 or unseen.numel() == 0:
        return stats

    mu = prior_mu.detach().float()
    var = prior_logvar.detach().float().exp()
    seen_mu = mu.index_select(0, seen)
    unseen_mu = mu.index_select(0, unseen)
    seen_var = var.index_select(0, seen)
    unseen_var = var.index_select(0, unseen)
    stats["graph_prob_prior_monitor_prior_seen_mu_norm_mean"] = _as_float(seen_mu.norm(dim=-1).mean())
    stats["graph_prob_prior_monitor_prior_unseen_mu_norm_mean"] = _as_float(unseen_mu.norm(dim=-1).mean())
    stats["graph_prob_prior_monitor_prior_seen_var_mean"] = _as_float(seen_var.mean())
    stats["graph_prob_prior_monitor_prior_unseen_var_mean"] = _as_float(unseen_var.mean())

    if seen.numel() > 1:
        stats["graph_prob_prior_monitor_prior_seen_seen_dist_mean"] = _as_float(
            _offdiag_values(torch.cdist(seen_mu, seen_mu, p=2)).mean()
        )
    if unseen.numel() > 1:
        unseen_unseen_dist = torch.cdist(unseen_mu, unseen_mu, p=2)
        stats["graph_prob_prior_monitor_prior_unseen_unseen_dist_mean"] = _as_float(_offdiag_values(unseen_unseen_dist).mean())
        nearest_unseen = unseen_unseen_dist.masked_fill(
            torch.eye(int(unseen.numel()), dtype=torch.bool, device=device),
            float("inf"),
        ).min(dim=-1).values
        stats["graph_prob_prior_monitor_unseen_nearest_unseen_distance_mean"] = _as_float(nearest_unseen.mean())
    seen_unseen_dist = torch.cdist(unseen_mu, seen_mu, p=2)
    stats["graph_prob_prior_monitor_prior_seen_unseen_dist_mean"] = _as_float(seen_unseen_dist.mean())
    stats["graph_prob_prior_monitor_unseen_nearest_seen_distance_mean"] = _as_float(seen_unseen_dist.min(dim=-1).values.mean())

    sim = F.normalize(mu, p=2, dim=-1, eps=1e-12).matmul(F.normalize(mu, p=2, dim=-1, eps=1e-12).t())
    unseen_to_seen = sim.index_select(0, unseen).index_select(1, seen).mean(dim=-1)
    if unseen.numel() > 1:
        unseen_to_unseen = sim.index_select(0, unseen).index_select(1, unseen)
        unseen_to_unseen = unseen_to_unseen.masked_fill(
            torch.eye(int(unseen.numel()), dtype=torch.bool, device=device),
            0.0,
        ).sum(dim=-1) / float(max(int(unseen.numel()) - 1, 1))
        stats["graph_prob_prior_monitor_unseen_to_seen_bias_risk_mean"] = _as_float(
            (unseen_to_seen - unseen_to_unseen).mean()
        )
    return stats


def gzsl_prior_risk_monitor(
    relation: torch.Tensor,
    seen_class_ids,
    unseen_class_ids,
    topk: int = 5,
    prefix: str = "graph_prob_prior_monitor_gzsl",
) -> Dict[str, float]:
    """
    GZSL relation 风险监测：检查 unseen 是否过度贴近 seen，以及 seen hub 是否吸附 unseen。
    relation 越大表示类别越接近，可传 graph、prior cosine 或 -symKL。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(relation) or relation.dim() != 2 or relation.shape[0] != relation.shape[1]:
        return stats
    rel = relation.detach().float()
    device = rel.device
    class_count = int(rel.shape[0])
    seen = _index_tensor(seen_class_ids, device)
    unseen = _index_tensor(unseen_class_ids, device)
    seen = seen[(seen >= 0) & (seen < class_count)]
    unseen = unseen[(unseen >= 0) & (unseen < class_count)]
    if seen.numel() == 0 or unseen.numel() == 0:
        return stats

    seen_seen = rel.index_select(0, seen).index_select(1, seen)
    unseen_unseen = rel.index_select(0, unseen).index_select(1, unseen)
    seen_unseen = rel.index_select(0, unseen).index_select(1, seen)
    if seen.numel() > 1:
        stats[f"{prefix}_seen_seen_relation_mean"] = _as_float(_offdiag_values(seen_seen).mean())
    if unseen.numel() > 1:
        stats[f"{prefix}_unseen_unseen_relation_mean"] = _as_float(_offdiag_values(unseen_unseen).mean())
    stats[f"{prefix}_seen_unseen_relation_mean"] = _as_float(seen_unseen.mean())
    if unseen.numel() > 1:
        local = unseen_unseen.masked_fill(
            torch.eye(int(unseen.numel()), dtype=torch.bool, device=device),
            0.0,
        ).sum(dim=-1) / float(max(int(unseen.numel()) - 1, 1))
        stats[f"{prefix}_unseen_to_seen_bias_risk_mean"] = _as_float((seen_unseen.mean(dim=-1) - local).mean())

    rel_for_rank = rel.clone()
    idx = torch.arange(class_count, device=device)
    rel_for_rank[idx, idx] = float("-inf")
    nearest = rel_for_rank.index_select(0, unseen).argmax(dim=-1)
    seen_mask = torch.zeros(class_count, dtype=torch.bool, device=device)
    seen_mask[seen] = True
    stats[f"{prefix}_unseen_nearest_is_seen_ratio"] = _as_float(seen_mask[nearest].float().mean())

    seen_scores = rel.index_select(0, unseen).index_select(1, seen)
    nearest_seen_score = seen_scores.max(dim=-1).values
    rank = 1 + (rel_for_rank.index_select(0, unseen) > nearest_seen_score[:, None]).float().sum(dim=-1)
    stats[f"{prefix}_unseen_nearest_seen_rank_mean"] = _as_float(rank.mean())

    k = max(1, min(int(topk), class_count - 1))
    top_idx = torch.topk(rel_for_rank.index_select(0, unseen), k=k, dim=-1).indices
    seen_hub = torch.zeros(class_count, device=device)
    seen_hub.scatter_add_(0, top_idx.reshape(-1), torch.ones(top_idx.numel(), device=device))
    seen_hub = seen_hub.index_select(0, seen)
    stats[f"{prefix}_seen_hub_for_unseen_count_max"] = _as_float(seen_hub.max())
    stats[f"{prefix}_seen_hub_for_unseen_count_mean"] = _as_float(seen_hub.mean())
    return stats


def false_high_pair_monitor(
    base_graph: torch.Tensor,
    new_relation: torch.Tensor,
    threshold: float = 0.9,
    true_neighbor_mask: Optional[torch.Tensor] = None,
    prefix: str = "graph_prob_prior_monitor_false_high",
) -> Dict[str, float]:
    """追踪 base_graph 中非对角高相似 pair 在 new_relation 中是否被压低。"""
    stats: Dict[str, float] = {}
    if not torch.is_tensor(base_graph) or not torch.is_tensor(new_relation):
        return stats
    if base_graph.shape != new_relation.shape or base_graph.dim() != 2 or base_graph.shape[0] != base_graph.shape[1]:
        return stats
    base = base_graph.detach().float()
    new = new_relation.detach().float().to(device=base.device)
    n = int(base.shape[0])
    offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=base.device)
    false_mask = (base > float(threshold)) & offdiag_mask
    stats[f"{prefix}_pair_count"] = _as_float(false_mask.float().sum())
    if bool(false_mask.any().item()):
        base_values = base[false_mask]
        new_values = new[false_mask]
        stats.update(tensor_stats(f"{prefix}_new_relation", new_values))
        stats[f"{prefix}_suppression_mean"] = _as_float((base_values - new_values).mean())
        stats[f"{prefix}_still_gt_0_9_count"] = _as_float((new_values > 0.9).float().sum())
    if true_neighbor_mask is not None and torch.is_tensor(true_neighbor_mask) and true_neighbor_mask.shape == base.shape:
        true_mask = true_neighbor_mask.to(device=base.device, dtype=torch.bool) & offdiag_mask
        if bool(true_mask.any().item()):
            true_mean = new[true_mask].mean()
            stats[f"{prefix}_true_neighbor_preservation_mean"] = _as_float(true_mean)
            if bool(false_mask.any().item()):
                stats[f"{prefix}_vs_true_neighbor_gap"] = _as_float(true_mean - new[false_mask].mean())
    return stats


def graph_prob_prior_grad_monitor(named_parameters) -> Dict[str, float]:
    """
    backward 之后调用的梯度健康监测。
    只读取参数梯度的 detach 值，判断 prior/dual/stats head 是否真的收到梯度。
    """
    groups = {
        "dual_alpha_head": ("dual_alpha_mu_head", "dual_alpha_anchor_head", "dual_alpha_delta_head"),
        "dual_beta_head": ("dual_beta_mu_head", "dual_beta_anchor_head", "dual_beta_delta_head"),
        "prior_head": (
            "prior_head",
            "factorized_semantic_prior_head",
            "residual_anchor_head",
            "residual_delta_head",
            "residual_logvar_head",
            "factorized_residual_anchor_head",
            "factorized_residual_delta_head",
            "factorized_residual_logvar_head",
            "dual_alpha_anchor_head",
            "dual_alpha_delta_head",
            "dual_beta_anchor_head",
            "dual_beta_delta_head",
        ),
        "stats_head": ("stats_head",),
    }
    sum_sq = {key: 0.0 for key in groups}
    counts = {key: 0 for key in groups}
    finite_num = 0
    finite_den = 0
    for name, param in named_parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        g = grad.detach().float()
        finite = torch.isfinite(g)
        finite_num += int(finite.float().sum().item())
        finite_den += int(g.numel())
        for key, needles in groups.items():
            if any(needle in name for needle in needles):
                finite_g = g[finite]
                if finite_g.numel() > 0:
                    sum_sq[key] += float(finite_g.pow(2).sum().item())
                    counts[key] += int(finite_g.numel())
    stats: Dict[str, float] = {}
    for key, value in sum_sq.items():
        stats[f"graph_prob_prior_monitor_grad_{key}_norm"] = math.sqrt(max(value, 0.0))
        stats[f"graph_prob_prior_monitor_grad_{key}_param_count"] = float(counts[key])
    denom = max(stats["graph_prob_prior_monitor_grad_dual_beta_head_norm"], 1e-12)
    stats["graph_prob_prior_monitor_grad_alpha_beta_ratio"] = (
        stats["graph_prob_prior_monitor_grad_dual_alpha_head_norm"] / denom
    )
    stats["graph_prob_prior_monitor_grad_finite_ratio"] = float(finite_num) / float(max(finite_den, 1))
    return stats


def posterior_prior_alignment_monitor(
    distance: torch.Tensor,
    latent_prob: Optional[torch.Tensor],
    targets_global: torch.Tensor,
    topk: int = 5,
) -> Dict[str, float]:
    """
    Posterior-prior alignment：看图像 posterior 是否真的更靠近正确类别 prior，
    而不是所有类别 prior 距离都差不多。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(distance) or not torch.is_tensor(targets_global) or distance.dim() != 2:
        return stats
    d = distance.detach().float()
    y = targets_global.detach().to(device=d.device, dtype=torch.long).view(-1)
    if y.numel() != d.shape[0]:
        return stats
    class_count = int(d.shape[1])
    true_d = d.gather(1, y[:, None]).squeeze(1)
    rank = 1 + (d < true_d[:, None]).float().sum(dim=-1)
    stats["graph_prob_prior_monitor_posterior_true_rank_top1"] = _as_float((rank <= 1).float().mean())
    stats["graph_prob_prior_monitor_posterior_true_rank_top5"] = _as_float((rank <= min(5, class_count)).float().mean())
    stats["graph_prob_prior_monitor_posterior_true_rank_top10"] = _as_float((rank <= min(10, class_count)).float().mean())
    stats.update(tensor_stats("graph_prob_prior_monitor_posterior_true_rank", rank))
    stats.update(tensor_stats("graph_prob_prior_monitor_posterior_kl_true", true_d))

    wrong = d.clone()
    wrong.scatter_(1, y[:, None], float("inf"))
    nearest_wrong = wrong.min(dim=-1).values
    margin = nearest_wrong - true_d
    stats.update(tensor_stats("graph_prob_prior_monitor_posterior_kl_nearest_wrong", nearest_wrong))
    stats.update(tensor_stats("graph_prob_prior_monitor_posterior_kl_margin", margin))
    stats["graph_prob_prior_monitor_posterior_kl_margin_positive_ratio"] = _as_float((margin > 0.0).float().mean())
    if latent_prob is not None and torch.is_tensor(latent_prob):
        stats.update(probability_stats("graph_prob_prior_monitor_latent_prob", latent_prob.detach(), true_index=y, topk=topk))
    return stats


def aggregate_moment_monitor(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    targets_global: torch.Tensor,
    aggregate_mu: torch.Tensor,
    aggregate_var: torch.Tensor,
    class_ids: torch.Tensor,
    prior_mu: Optional[torch.Tensor] = None,
    prior_logvar: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Aggregate moment：看同一类别的一批图像聚合后，均值和方差是否能对上类别 prior；
    同时拆开 E[var_i] 和 Var(mu_i)，判断聚合方差来自图像不确定性还是同类样本离散。
    """
    stats: Dict[str, float] = {"graph_prob_prior_monitor_agg_class_count": float(class_ids.numel())}
    if class_ids.numel() == 0:
        return stats
    counts = torch.stack([(targets_global == cid).float().sum() for cid in class_ids]).to(device=posterior_mu.device)
    stats.update(tensor_stats("graph_prob_prior_monitor_agg_samples_per_class", counts))
    stats["graph_prob_prior_monitor_agg_single_sample_class_ratio"] = _as_float((counts <= 1.0).float().mean())

    posterior_var = posterior_logvar.detach().float().exp()
    var_components = []
    spread_components = []
    for cid in class_ids:
        mask = targets_global == cid
        mu_c = posterior_mu.detach().float()[mask]
        var_c = posterior_var[mask]
        mean_c = mu_c.mean(dim=0)
        var_components.append(var_c.mean())
        spread_components.append((mu_c - mean_c).pow(2).mean())
    stats["graph_prob_prior_monitor_agg_posterior_var_component_mean"] = _as_float(torch.stack(var_components).mean())
    stats["graph_prob_prior_monitor_agg_mu_spread_component_mean"] = _as_float(torch.stack(spread_components).mean())
    stats["graph_prob_prior_monitor_agg_total_var_mean"] = _as_float(aggregate_var.detach().float().mean())
    if prior_mu is not None and torch.is_tensor(prior_mu):
        selected_mu = prior_mu.detach().float().index_select(0, class_ids.to(device=prior_mu.device))
        dist = (aggregate_mu.detach().float().to(selected_mu.device) - selected_mu).norm(dim=-1)
        stats.update(tensor_stats("graph_prob_prior_monitor_agg_mu_to_prior_dist", dist))
    if prior_logvar is not None and torch.is_tensor(prior_logvar):
        eps = 1e-12
        selected_logvar = prior_logvar.detach().float().index_select(0, class_ids.to(device=prior_logvar.device))
        mse = (aggregate_var.detach().float().to(selected_logvar.device).clamp_min(eps).log() - selected_logvar).pow(2).mean()
        stats["graph_prob_prior_monitor_agg_var_to_prior_logvar_mse"] = _as_float(mse)
    return stats


def factorized_health_monitor(
    semantic_mu: torch.Tensor,
    semantic_logvar: torch.Tensor,
    variation_mu: torch.Tensor,
    variation_logvar: torch.Tensor,
    targets_global: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Factorized latent：看 semantic factor 和 variation factor 是否各司其职，
    有没有互相串信息，variation 是否偷学类别语义。
    """
    stats: Dict[str, float] = {}
    if not all(torch.is_tensor(x) for x in (semantic_mu, semantic_logvar, variation_mu, variation_logvar)):
        return stats
    sm = semantic_mu.detach().float()
    vm = variation_mu.detach().float()
    stats.update(tensor_stats("graph_prob_prior_monitor_factorized_semantic_mu_norm", sm.norm(dim=-1)))
    stats.update(tensor_stats("graph_prob_prior_monitor_factorized_variation_mu_norm", vm.norm(dim=-1)))
    stats.update(tensor_stats("graph_prob_prior_monitor_factorized_semantic_var", semantic_logvar.detach().float().exp()))
    stats.update(tensor_stats("graph_prob_prior_monitor_factorized_variation_var", variation_logvar.detach().float().exp()))
    stats["graph_prob_prior_monitor_factorized_semantic_variation_norm_ratio"] = _as_float(
        sm.norm(dim=-1).mean() / vm.norm(dim=-1).mean().clamp_min(1e-12)
    )
    sm_centered = sm - sm.mean(dim=0, keepdim=True)
    vm_centered = vm - vm.mean(dim=0, keepdim=True)
    cross_cov = sm_centered.t().matmul(vm_centered) / float(max(sm.shape[0], 1))
    stats["graph_prob_prior_monitor_factorized_cross_cov_fro"] = _as_float(cross_cov.pow(2).sum().sqrt())
    stats["graph_prob_prior_monitor_factorized_semantic_batch_effective_rank"] = _effective_rank(sm, center=True)
    stats["graph_prob_prior_monitor_factorized_variation_batch_effective_rank"] = _effective_rank(vm, center=True)
    if targets_global is not None and torch.is_tensor(targets_global):
        y = targets_global.detach().to(device=vm.device, dtype=torch.long)
        class_ids = torch.unique(y, sorted=True)
        if class_ids.numel() > 1:
            centers = []
            within = []
            for cid in class_ids:
                vals = vm[y == cid]
                center = vals.mean(dim=0)
                centers.append(center)
                within.append((vals - center).norm(dim=-1).mean())
            centers = torch.stack(centers)
            between = _offdiag_values(torch.cdist(centers, centers, p=2)).mean()
            within_mean = torch.stack(within).mean()
            stats["graph_prob_prior_monitor_factorized_variation_between_class_mean_dist"] = _as_float(between)
            stats["graph_prob_prior_monitor_factorized_variation_within_class_mean_dist"] = _as_float(within_mean)
            stats["graph_prob_prior_monitor_factorized_variation_class_ratio"] = _as_float(between / within_mean.clamp_min(1e-12))
    return stats


def dual_sample_beta_monitor(
    beta_distance: torch.Tensor,
    targets_global: torch.Tensor,
    p_neg: Optional[torch.Tensor] = None,
    margin: Optional[torch.Tensor] = None,
    topk: int = 5,
) -> Dict[str, float]:
    """
    Dual sample beta：直接检查当前 beta loss 的样本级目标是否生效。
    class-level beta prior 分开不代表样本 posterior 已经分到正确类别。
    """
    stats = posterior_prior_alignment_monitor(beta_distance, None, targets_global, topk=topk)
    renamed = {}
    for key, value in stats.items():
        renamed[key.replace("graph_prob_prior_monitor_posterior_", "graph_prob_prior_monitor_dual_sample_beta_")] = value
    stats = renamed
    if p_neg is not None and torch.is_tensor(p_neg):
        d = beta_distance.detach().float()
        y = targets_global.detach().to(device=d.device, dtype=torch.long).view(-1)
        neg = p_neg.detach().float().to(device=d.device)
        if neg.shape == d.shape:
            true_d = d.gather(1, y[:, None])
            neg_mask = neg > 0.0
            weighted_margin = (d - true_d).mul(neg).sum(dim=-1)
            stats.update(tensor_stats("graph_prob_prior_monitor_dual_sample_beta_hardneg_weighted_margin", weighted_margin))
            masked_d = d.masked_fill(~neg_mask, float("inf"))
            nearest_hardneg = masked_d.min(dim=-1).values
            finite = nearest_hardneg[torch.isfinite(nearest_hardneg)]
            if finite.numel() > 0:
                stats.update(tensor_stats("graph_prob_prior_monitor_dual_sample_beta_nearest_hardneg_distance", finite))
            if margin is not None and torch.is_tensor(margin) and margin.shape == d.shape:
                violation = F.relu(true_d + margin.detach().float().to(device=d.device) - d)
                values = violation[neg_mask]
                if values.numel() > 0:
                    stats["graph_prob_prior_monitor_dual_sample_beta_hardneg_violation_rate"] = _as_float((values > 0.0).float().mean())
    return stats


def dual_metric_distribution_monitor(
    mu_alpha: torch.Tensor,
    logvar_alpha: Optional[torch.Tensor],
    mu_beta: torch.Tensor,
    logvar_beta: Optional[torch.Tensor],
    t_alpha: Optional[torch.Tensor] = None,
    p_pos: Optional[torch.Tensor] = None,
    p_neg: Optional[torch.Tensor] = None,
    bank: Optional[torch.Tensor] = None,
    topk: int = 5,
    margin: float = 0.0,
) -> Dict[str, float]:
    """
    Dual distribution：看 alpha 是否保留上下文关系，beta 是否把 hard negative 分开；
    理想状态是语义近邻可以相近，但类别分布不能交叠。
    """
    stats: Dict[str, float] = {}
    if p_pos is not None and torch.is_tensor(p_pos):
        stats.update(probability_stats("graph_prob_prior_monitor_dual_p_pos", p_pos.detach(), topk=topk))
        stats.update(_row_entropy_stats("graph_prob_prior_monitor_dual_p_pos", p_pos.detach()))
        stats["graph_prob_prior_monitor_dual_p_pos_nonzero_mean"] = _as_float((p_pos.detach() > 0).float().sum(dim=-1).mean())
        stats["graph_prob_prior_monitor_dual_p_pos_self_mass_mean"] = _as_float(torch.diagonal(p_pos.detach().float()).mean())
    if p_neg is not None and torch.is_tensor(p_neg):
        stats.update(probability_stats("graph_prob_prior_monitor_dual_p_neg", p_neg.detach(), topk=topk))
        stats.update(_row_entropy_stats("graph_prob_prior_monitor_dual_p_neg", p_neg.detach()))
        stats["graph_prob_prior_monitor_dual_p_neg_nonzero_mean"] = _as_float((p_neg.detach() > 0).float().sum(dim=-1).mean())
    if p_pos is not None and p_neg is not None and torch.is_tensor(p_pos) and torch.is_tensor(p_neg):
        pos = p_pos.detach().float()
        neg = p_neg.detach().float()
        pos_mask = pos > 0.0
        neg_mask = neg > 0.0
        inter = (pos_mask & neg_mask).float().sum(dim=-1)
        union = (pos_mask | neg_mask).float().sum(dim=-1).clamp_min(1.0)
        stats["graph_prob_prior_monitor_dual_pos_neg_topk_jaccard_mean"] = _as_float((inter / union).mean())
        stats["graph_prob_prior_monitor_dual_pos_neg_overlap_mass_mean"] = _as_float((neg * pos_mask.float()).sum(dim=-1).mean())
        stats["graph_prob_prior_monitor_dual_pos_neg_top1_same_ratio"] = _as_float((pos.argmax(dim=-1) == neg.argmax(dim=-1)).float().mean())
        p = pos / pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        q = neg / neg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        m = 0.5 * (p + q)
        js = 0.5 * (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1)
        js = js + 0.5 * (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1)
        stats["graph_prob_prior_monitor_dual_pos_neg_js_divergence_mean"] = _as_float(js.mean())

    if torch.is_tensor(mu_alpha) and torch.is_tensor(mu_beta):
        alpha_dist = torch.cdist(mu_alpha.detach().float(), mu_alpha.detach().float(), p=2).pow(2)
        alpha_sim = torch.exp(-alpha_dist / alpha_dist.detach().median().clamp_min(1e-12))
        if t_alpha is not None and torch.is_tensor(t_alpha):
            stats["graph_prob_prior_monitor_dual_alpha_relation_spearman_all"] = _spearman_corr(_offdiag_values(alpha_sim), _offdiag_values(t_alpha.detach().float()))
        if p_pos is not None and torch.is_tensor(p_pos):
            pos_mask = p_pos.detach() > 0.0
            nonpos_mask = ~pos_mask
            eye = torch.eye(pos_mask.shape[0], dtype=torch.bool, device=pos_mask.device)
            nonpos_mask = nonpos_mask & ~eye
            if bool(pos_mask.any().item()):
                stats.update(tensor_stats("graph_prob_prior_monitor_dual_alpha_pos_similarity", alpha_sim[pos_mask]))
            if bool(nonpos_mask.any().item()):
                nonpos_mean = alpha_sim[nonpos_mask].mean()
                stats["graph_prob_prior_monitor_dual_alpha_nonpos_similarity_mean"] = _as_float(nonpos_mean)
                if bool(pos_mask.any().item()):
                    stats["graph_prob_prior_monitor_dual_alpha_pos_nonpos_gap"] = _as_float(alpha_sim[pos_mask].mean() - nonpos_mean)

        beta_dist = torch.cdist(mu_beta.detach().float(), mu_beta.detach().float(), p=2)
        if logvar_beta is not None and torch.is_tensor(logvar_beta):
            radius = logvar_beta.detach().float().exp().mean(dim=-1).clamp_min(1e-12).sqrt()
        else:
            radius = beta_dist.new_zeros(beta_dist.shape[0])
        overlap_risk = beta_dist < (radius[:, None] + radius[None, :] + float(margin))
        eye = torch.eye(beta_dist.shape[0], dtype=torch.bool, device=beta_dist.device)
        stats["graph_prob_prior_monitor_dual_beta_all_overlap_risk_rate"] = _as_float(overlap_risk[~eye].float().mean())
        if p_neg is not None and torch.is_tensor(p_neg):
            neg_mask = p_neg.detach().to(device=beta_dist.device) > 0.0
            if bool(neg_mask.any().item()):
                hard_dist = beta_dist[neg_mask]
                stats.update(tensor_stats("graph_prob_prior_monitor_dual_beta_hardneg_distance", hard_dist))
                margin_value = beta_dist - radius[:, None] - radius[None, :] - float(margin)
                stats.update(tensor_stats("graph_prob_prior_monitor_dual_beta_hardneg_margin", margin_value[neg_mask]))
                stats["graph_prob_prior_monitor_dual_beta_hardneg_overlap_risk_rate"] = _as_float(overlap_risk[neg_mask].float().mean())
        if p_pos is not None and p_neg is not None and torch.is_tensor(p_pos) and torch.is_tensor(p_neg):
            pos_mask = p_pos.detach().to(device=beta_dist.device) > 0.0
            neg_mask = p_neg.detach().to(device=beta_dist.device) > 0.0
            if bool(pos_mask.any().item()) and bool(neg_mask.any().item()):
                stats["graph_prob_prior_monitor_dual_beta_hardneg_suppression_gap"] = _as_float(beta_dist[neg_mask].mean() - beta_dist[pos_mask].mean())
                alpha_high_beta_risk = pos_mask & overlap_risk
                stats["graph_prob_prior_monitor_dual_alpha_beta_conflict_rate"] = _as_float(alpha_high_beta_risk.float().sum() / pos_mask.float().sum().clamp_min(1.0))
                stats["graph_prob_prior_monitor_dual_alpha_high_beta_safe_rate"] = 1.0 - stats["graph_prob_prior_monitor_dual_alpha_beta_conflict_rate"]
        if bank is not None and torch.is_tensor(bank) and bank.shape == mu_alpha.shape:
            bank_n = F.normalize(bank.detach().float(), p=2, dim=-1, eps=1e-12)
            alpha_n = F.normalize(mu_alpha.detach().float(), p=2, dim=-1, eps=1e-12)
            cos = (alpha_n * bank_n).sum(dim=-1)
            stats["graph_prob_prior_monitor_dual_alpha_bank_cosine_mean"] = _as_float(cos.mean())
            stats["graph_prob_prior_monitor_dual_alpha_bank_drift_mean"] = _as_float((1.0 - cos).mean())
        if bank is not None and torch.is_tensor(bank) and bank.shape == mu_beta.shape:
            bank_n = F.normalize(bank.detach().float(), p=2, dim=-1, eps=1e-12)
            beta_n = F.normalize(mu_beta.detach().float(), p=2, dim=-1, eps=1e-12)
            cos = (beta_n * bank_n).sum(dim=-1)
            stats["graph_prob_prior_monitor_dual_beta_bank_cosine_mean"] = _as_float(cos.mean())
            stats["graph_prob_prior_monitor_dual_beta_bank_drift_mean"] = _as_float((1.0 - cos).mean())
    return stats


def loss_scale_monitor(
    main_loss: Optional[float] = None,
    graph_prob_prior_loss: Optional[float] = None,
    alpha_loss: Optional[float] = None,
    beta_lower_loss: Optional[float] = None,
    beta_upper_loss: Optional[float] = None,
    loss_weight: float = 1.0,
    alpha_weight: float = 1.0,
    beta_lower_weight: float = 1.0,
    beta_upper_weight: float = 0.0,
) -> Dict[str, float]:
    """监测 GraphProbPrior 各子项乘权重后的尺度，以及它相对主分类 loss 的强度。"""
    stats: Dict[str, float] = {}
    eps = 1e-12
    if graph_prob_prior_loss is not None:
        weighted = float(graph_prob_prior_loss) * float(loss_weight)
        stats["graph_prob_prior_monitor_loss_weighted_graph_prob_prior_loss"] = weighted
        if main_loss is not None:
            stats["graph_prob_prior_monitor_loss_weighted_gpp_to_main_loss_ratio"] = weighted / max(abs(float(main_loss)), eps)
    if alpha_loss is not None:
        stats["graph_prob_prior_monitor_loss_weighted_alpha_loss"] = float(alpha_loss) * float(alpha_weight) * float(loss_weight)
    if beta_lower_loss is not None:
        stats["graph_prob_prior_monitor_loss_weighted_beta_lower_loss"] = float(beta_lower_loss) * float(beta_lower_weight) * float(loss_weight)
    if beta_upper_loss is not None:
        stats["graph_prob_prior_monitor_loss_weighted_beta_upper_loss"] = float(beta_upper_loss) * float(beta_upper_weight) * float(loss_weight)
    if alpha_loss is not None and beta_lower_loss is not None:
        stats["graph_prob_prior_monitor_loss_alpha_to_beta_lower_loss_ratio"] = float(alpha_loss) / max(abs(float(beta_lower_loss)), eps)
    return stats


def aggregate_monitor_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """
    聚合短跑诊断脚本收集到的逐 batch 监测量。

    输入 rows 是若干个 batch 的 stats dict。
    输出对同名字段做简单平均，并跳过缺失值和非有限值。

    注意：
    - 这是为了快速得到一个“本次短跑的总体印象”；
    - 如果要看训练过程漂移，应直接看 CSV 中逐 batch 的曲线，而不是只看 summary。
    """
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                buckets.setdefault(key, []).append(float(value))
    summary: Dict[str, float] = {}
    for key, values in sorted(buckets.items()):
        if not values:
            continue
        mean_value = float(sum(values) / len(values))
        summary[key] = mean_value
        summary[f"{key}_first"] = float(values[0])
        summary[f"{key}_last"] = float(values[-1])
        if len(values) > 1:
            summary[f"{key}_slope"] = float((values[-1] - values[0]) / float(len(values) - 1))
        else:
            summary[f"{key}_slope"] = 0.0
    return summary
