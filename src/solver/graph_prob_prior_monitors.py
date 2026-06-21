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


def mmd_monitor(pair_dist: torch.Tensor, kernel: torch.Tensor) -> Dict[str, float]:
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
    return {key: float(sum(values) / len(values)) for key, values in sorted(buckets.items()) if values}
