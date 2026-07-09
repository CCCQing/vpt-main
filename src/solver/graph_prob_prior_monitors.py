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


def _membership_mask(values: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
    """
    判断 values 中每个类别 id 是否属于 members。

    这里不用 torch.isin，是为了兼容旧版 PyTorch。Graph-GP 的 support/pseudo
    监测需要按类别集合切分当前 batch，所以用广播比较构造布尔 mask。
    """
    if not torch.is_tensor(values) or not torch.is_tensor(members) or members.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    members = members.to(device=values.device, dtype=values.dtype)
    return (values[:, None] == members[None, :]).any(dim=1)


def _matrix_condition_number(x: torch.Tensor) -> float:
    """用奇异值估计条件数；只作为监测项，失败时返回 0。"""
    if not torch.is_tensor(x) or x.dim() != 2 or x.numel() == 0:
        return 0.0
    values = x.detach().to(device="cpu", dtype=torch.float64)
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    try:
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "svdvals"):
            singular = torch.linalg.svdvals(values)
        else:
            singular = torch.svd(values).S
    except RuntimeError:
        return 0.0
    singular = singular[torch.isfinite(singular)]
    if singular.numel() == 0:
        return 0.0
    max_s = singular.max()
    min_s = singular[singular > 1e-12].min() if bool((singular > 1e-12).any().item()) else singular.new_tensor(1e-12)
    return _as_float(max_s / min_s.clamp_min(1e-12))


def _row_probability_from_nonnegative(x: torch.Tensor) -> torch.Tensor:
    values = x.detach().float().clamp_min(0.0)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _abs_row_probability(x: torch.Tensor) -> torch.Tensor:
    values = x.detach().float().abs()
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _batch_class_center_stats(
    prefix: str,
    posterior_mu: torch.Tensor,
    prior_mu: torch.Tensor,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if not bool(sample_mask.any().item()):
        return stats
    mu = posterior_mu.detach().float()
    prior = prior_mu.detach().float().to(device=mu.device)
    cls = torch.unique(targets[sample_mask], sorted=True)
    centers = []
    proto = []
    for cid in cls:
        vals = mu[(targets == cid) & sample_mask]
        if vals.numel() == 0:
            continue
        centers.append(vals.mean(dim=0))
        proto.append(prior[int(cid.item())])
    if not centers:
        return stats
    centers_t = torch.stack(centers, dim=0)
    proto_t = torch.stack(proto, dim=0).to(device=centers_t.device)
    center_cos = F.cosine_similarity(centers_t, proto_t, dim=-1, eps=1e-12)
    center_mse = (centers_t - proto_t).pow(2).mean(dim=-1)
    stats.update(tensor_stats(f"{prefix}_center_cos", center_cos))
    stats.update(tensor_stats(f"{prefix}_center_mse", center_mse))
    return stats


def graph_gp_prototype_monitor(
    prior_mu: torch.Tensor,
    prior_logvar: Optional[torch.Tensor],
    support_ids: torch.Tensor,
    pseudo_unseen_ids: torch.Tensor,
    observed_support_ids: torch.Tensor,
    support_count: torch.Tensor,
    center_var: torch.Tensor,
    obs_noise: torch.Tensor,
    solve_residual: torch.Tensor,
    uncertainty_diag: torch.Tensor,
    system_diag_ratio: torch.Tensor,
    support_post_var: Optional[torch.Tensor] = None,
    support_visual_var: Optional[torch.Tensor] = None,
    visual_within_var: Optional[torch.Tensor] = None,
    proto_var_term: Optional[torch.Tensor] = None,
    visual_var_term: Optional[torch.Tensor] = None,
    prior_var_raw: Optional[torch.Tensor] = None,
    dynamic_prior_var: Optional[torch.Tensor] = None,
    kernel: Optional[torch.Tensor] = None,
    system: Optional[torch.Tensor] = None,
    k_all_s: Optional[torch.Tensor] = None,
    smoothing_coeff: Optional[torch.Tensor] = None,
    graph: Optional[torch.Tensor] = None,
    distance: Optional[torch.Tensor] = None,
    sample_energy_loss: Optional[torch.Tensor] = None,
    posterior_mu: Optional[torch.Tensor] = None,
    targets_global: Optional[torch.Tensor] = None,
    seen_class_ids=None,
    unseen_class_ids=None,
    topk: int = 5,
    prefix: str = "graph_prob_prior_monitor_graph_gp",
) -> Dict[str, float]:
    """
    Graph-GP prototype 推断专属监测量。

    这些统计不参与 loss，只回答四个问题：
    1. support-seen 中有多少类已经被 posterior_mu 观测到；
    2. V_support 的类内方差和观测噪声 R_s 是否异常；
    3. 线性方程求解是否稳定，Graph-GP predictive uncertainty 是否过大；
    4. 推断出的 M_star 是否仍然同向扎堆，以及 pseudo-unseen 样本能否在 energy CE 中找到真类 prototype。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(prior_mu):
        return stats

    device = prior_mu.device
    support_ids = support_ids.to(device=device, dtype=torch.long) if torch.is_tensor(support_ids) else torch.empty(0, device=device, dtype=torch.long)
    pseudo_unseen_ids = (
        pseudo_unseen_ids.to(device=device, dtype=torch.long)
        if torch.is_tensor(pseudo_unseen_ids)
        else torch.empty(0, device=device, dtype=torch.long)
    )
    observed_support_ids = (
        observed_support_ids.to(device=device, dtype=torch.long)
        if torch.is_tensor(observed_support_ids)
        else torch.empty(0, device=device, dtype=torch.long)
    )

    stats[f"{prefix}_support_class_count"] = float(support_ids.numel())
    stats[f"{prefix}_pseudo_unseen_class_count"] = float(pseudo_unseen_ids.numel())
    stats[f"{prefix}_support_observed_class_count"] = float(observed_support_ids.numel())
    stats[f"{prefix}_support_observed_ratio"] = float(observed_support_ids.numel()) / float(max(int(support_ids.numel()), 1))
    if support_ids.numel() > 0 and torch.is_tensor(support_count):
        selected_count = support_count.detach().float().to(device=device).index_select(0, support_ids)
        stats.update(tensor_stats(f"{prefix}_support_count", selected_count))

    if torch.is_tensor(center_var) and center_var.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_seen_center_var", center_var))
    if torch.is_tensor(obs_noise) and obs_noise.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_obs_noise", obs_noise))
        reliability = 1.0 / obs_noise.detach().float().clamp_min(1e-12)
        stats.update(tensor_stats(f"{prefix}_seen_center_reliability", reliability))

    if torch.is_tensor(solve_residual):
        stats[f"{prefix}_solve_residual"] = _as_float(solve_residual)
    if torch.is_tensor(system_diag_ratio):
        stats[f"{prefix}_kernel_diag_ratio"] = _as_float(system_diag_ratio)
    if torch.is_tensor(uncertainty_diag) and uncertainty_diag.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_uncertainty_diag", uncertainty_diag))
    if torch.is_tensor(support_post_var) and support_post_var.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_support_post_var", support_post_var))
    if torch.is_tensor(support_visual_var) and support_visual_var.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_support_visual_var", support_visual_var))
    if torch.is_tensor(visual_within_var) and visual_within_var.numel() > 0:
        visual_var = visual_within_var.detach().float()
        stats.update(tensor_stats(f"{prefix}_visual_within_var", visual_var))
    if torch.is_tensor(proto_var_term) and proto_var_term.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_prior_var_proto_term", proto_var_term.detach().float()))
    if torch.is_tensor(visual_var_term) and visual_var_term.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_prior_var_visual_term", visual_var_term.detach().float()))
    if torch.is_tensor(prior_var_raw) and prior_var_raw.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_prior_var_raw", prior_var_raw.detach().float()))
    if torch.is_tensor(dynamic_prior_var) and dynamic_prior_var.numel() > 0:
        dyn_var = dynamic_prior_var.detach().float()
        stats.update(tensor_stats(f"{prefix}_prior_var_dynamic", dyn_var))
        if torch.is_tensor(prior_var_raw) and tuple(prior_var_raw.shape) == tuple(dynamic_prior_var.shape):
            raw_var = prior_var_raw.detach().float().to(device=dyn_var.device)
            stats[f"{prefix}_prior_var_clamp_rate"] = _as_float((raw_var != dyn_var).float().mean())
            stats[f"{prefix}_prior_var_max_clamp_rate"] = _as_float((raw_var > dyn_var).float().mean())
            stats[f"{prefix}_prior_var_min_clamp_rate"] = _as_float((raw_var < dyn_var).float().mean())
        denom = dyn_var.clamp_min(1e-12)
        if torch.is_tensor(proto_var_term) and tuple(proto_var_term.shape) == tuple(dynamic_prior_var.shape):
            stats.update(tensor_stats(f"{prefix}_prior_var_proto_share", proto_var_term.detach().float().to(device=dyn_var.device) / denom))
        if torch.is_tensor(visual_var_term) and tuple(visual_var_term.shape) == tuple(dynamic_prior_var.shape):
            stats.update(tensor_stats(f"{prefix}_prior_var_visual_share", visual_var_term.detach().float().to(device=dyn_var.device) / denom))

    if torch.is_tensor(kernel) and kernel.dim() == 2 and kernel.shape[0] == kernel.shape[1]:
        k = kernel.detach().float().to(device=device)
        class_count = int(k.shape[0])
        eye = torch.eye(class_count, dtype=torch.bool, device=device)
        stats.update(tensor_stats(f"{prefix}_kernel_offdiag", k[~eye]))
        stats[f"{prefix}_kernel_effective_rank"] = _effective_rank(k, center=False)
        stats[f"{prefix}_kernel_condition"] = _matrix_condition_number(k)
        kernel_prob = _row_probability_from_nonnegative(k)
        stats.update(probability_stats(f"{prefix}_kernel_row", kernel_prob, topk=topk))

    if torch.is_tensor(system) and system.dim() == 2 and system.shape[0] == system.shape[1]:
        stats[f"{prefix}_system_condition"] = _matrix_condition_number(system)

    if torch.is_tensor(k_all_s) and k_all_s.dim() == 2 and k_all_s.numel() > 0:
        support_prob = _row_probability_from_nonnegative(k_all_s.to(device=device))
        stats.update(probability_stats(f"{prefix}_k_all_s_support", support_prob, topk=topk))
        if pseudo_unseen_ids.numel() > 0 and int(k_all_s.shape[0]) > int(pseudo_unseen_ids.max().item()):
            pseudo_support_prob = support_prob.index_select(0, pseudo_unseen_ids)
            stats.update(probability_stats(f"{prefix}_k_us_support", pseudo_support_prob, topk=topk))

    if torch.is_tensor(smoothing_coeff) and smoothing_coeff.dim() == 2 and smoothing_coeff.numel() > 0:
        coeff_prob = _abs_row_probability(smoothing_coeff.to(device=device))
        stats.update(probability_stats(f"{prefix}_smoothing_coeff_abs", coeff_prob, topk=topk))
        top1_mass = coeff_prob.max(dim=-1).values
        stats[f"{prefix}_smoothing_strength_mean"] = _as_float((1.0 - top1_mass).mean())
        stats.update(tensor_stats(f"{prefix}_smoothing_coeff_l1", smoothing_coeff.detach().float().abs().sum(dim=-1)))

    mu = prior_mu.detach().float()
    stats.update(tensor_stats(f"{prefix}_mstar_norm", mu.norm(dim=-1)))
    if mu.dim() == 2 and int(mu.shape[0]) > 1:
        class_count = int(mu.shape[0])
        eye = torch.eye(class_count, dtype=torch.bool, device=mu.device)
        stats[f"{prefix}_mstar_effective_rank"] = _effective_rank(mu, center=True)

        # pair cosine 用来看 M_star 是否全都朝同一个方向。offdiag 越高，越像一团同向云。
        mu_dir = F.normalize(mu, p=2, dim=-1, eps=1e-12)
        pair_cos = mu_dir.matmul(mu_dir.t())
        offdiag_cos = pair_cos[~eye]
        stats.update(tensor_stats(f"{prefix}_mstar_pair_cos_offdiag", offdiag_cos))
        stats[f"{prefix}_mstar_pair_cos_gt_0_9"] = _as_float((offdiag_cos > 0.9).float().mean())
        if torch.is_tensor(graph) and graph.shape == pair_cos.shape:
            stats[f"{prefix}_mstar_graph_prior_spearman"] = _spearman_corr(
                graph.detach().float().to(device=pair_cos.device)[~eye],
                offdiag_cos,
            )

        # 标准化欧氏距离用来看 prototype 在 latent 空间里实际隔多远，避免只看 cosine。
        dist = torch.cdist(mu, mu, p=2) / math.sqrt(float(max(int(mu.shape[1]), 1)))
        offdiag_dist = dist[~eye]
        stats.update(tensor_stats(f"{prefix}_mstar_dist_offdiag", offdiag_dist))
        if torch.is_tensor(prior_logvar) and prior_logvar.shape == prior_mu.shape:
            logvar = prior_logvar.detach().float().to(device=mu.device)
            symkl = _pairwise_symkl(mu, logvar)
            off_symkl = symkl[~eye]
            stats.update(tensor_stats(f"{prefix}_mstar_symkl", off_symkl))
            if off_symkl.numel() > 0:
                stats[f"{prefix}_mstar_symkl_min_offdiag"] = _as_float(off_symkl.min())
            radius = logvar.exp().mean(dim=-1).clamp_min(1e-12).sqrt()
            raw_dist = torch.cdist(mu, mu, p=2)
            overlap = raw_dist < (radius[:, None] + radius[None, :])
            stats[f"{prefix}_mstar_overlap_risk_rate"] = _as_float(overlap[~eye].float().mean())

    if torch.is_tensor(distance) and torch.is_tensor(targets_global):
        targets = targets_global.to(device=distance.device, dtype=torch.long)
        if distance.dim() == 2 and targets.dim() == 1 and int(distance.shape[0]) == int(targets.numel()):
            pred = distance.argmin(dim=-1)
            hit = (pred == targets).float()
            row = torch.arange(distance.shape[0], device=distance.device)
            true_dist = distance[row, targets]
            neg_distance = distance.clone()
            neg_distance[row, targets] = float("inf")
            best_neg = neg_distance.min(dim=-1).values
            margin = best_neg - true_dist

            pseudo_mask = _membership_mask(targets, pseudo_unseen_ids)
            support_mask = _membership_mask(targets, support_ids)
            if bool(pseudo_mask.any().item()):
                stats[f"{prefix}_pseudo_unseen_rank1"] = _as_float(hit[pseudo_mask].mean())
                stats[f"{prefix}_pseudo_unseen_acc"] = stats[f"{prefix}_pseudo_unseen_rank1"]
                stats[f"{prefix}_pseudo_unseen_true_kl_mean"] = _as_float(true_dist[pseudo_mask].mean())
                stats.update(tensor_stats(f"{prefix}_pseudo_unseen_margin", margin[pseudo_mask]))
                if torch.is_tensor(sample_energy_loss) and sample_energy_loss.numel() == targets.numel():
                    stats[f"{prefix}_pseudo_unseen_energy_loss"] = _as_float(
                        sample_energy_loss.detach().float().to(device=targets.device)[pseudo_mask].mean()
                    )
            if bool(support_mask.any().item()):
                stats[f"{prefix}_support_seen_rank1"] = _as_float(hit[support_mask].mean())
                stats[f"{prefix}_support_seen_acc"] = stats[f"{prefix}_support_seen_rank1"]
                stats[f"{prefix}_support_seen_true_kl_mean"] = _as_float(true_dist[support_mask].mean())
                stats.update(tensor_stats(f"{prefix}_support_seen_margin", margin[support_mask]))
                if torch.is_tensor(sample_energy_loss) and sample_energy_loss.numel() == targets.numel():
                    stats[f"{prefix}_support_seen_energy_loss"] = _as_float(
                        sample_energy_loss.detach().float().to(device=targets.device)[support_mask].mean()
                    )
            if bool(pseudo_mask.any().item()) and bool(support_mask.any().item()):
                stats[f"{prefix}_support_pseudo_rank1_gap"] = (
                    stats[f"{prefix}_support_seen_rank1"] - stats[f"{prefix}_pseudo_unseen_rank1"]
                )
                stats[f"{prefix}_support_pseudo_acc_gap"] = (
                    stats[f"{prefix}_support_seen_acc"] - stats[f"{prefix}_pseudo_unseen_acc"]
                )
                if f"{prefix}_support_seen_energy_loss" in stats and f"{prefix}_pseudo_unseen_energy_loss" in stats:
                    stats[f"{prefix}_support_pseudo_energy_loss_gap"] = (
                        stats[f"{prefix}_pseudo_unseen_energy_loss"] - stats[f"{prefix}_support_seen_energy_loss"]
                    )

            if torch.is_tensor(posterior_mu) and posterior_mu.dim() == 2 and posterior_mu.shape[0] == targets.numel():
                stats.update(
                    _batch_class_center_stats(
                        f"{prefix}_pseudo_unseen",
                        posterior_mu,
                        prior_mu,
                        targets,
                        pseudo_mask,
                    )
                )
                stats.update(
                    _batch_class_center_stats(
                        f"{prefix}_support_seen",
                        posterior_mu,
                        prior_mu,
                        targets,
                        support_mask,
                    )
                )

            seen = _index_tensor(seen_class_ids, distance.device)
            unseen = _index_tensor(unseen_class_ids, distance.device)
            seen = seen[(seen >= 0) & (seen < int(distance.shape[1]))]
            unseen = unseen[(unseen >= 0) & (unseen < int(distance.shape[1]))]
            if seen.numel() > 0 and unseen.numel() > 0:
                logits = -distance.detach().float()
                max_seen = logits.index_select(1, seen).max(dim=-1).values
                max_unseen = logits.index_select(1, unseen).max(dim=-1).values
                stats.update(tensor_stats(f"{prefix}_seen_unseen_logit_bias", max_seen - max_unseen))

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

    对应 GraphPriorInputBuilder.build_target() 的逻辑：
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
    values = x.detach()
    if values.numel() == 0:
        return 0.0
    # This rank metric is diagnostic only. CPU SVD avoids noisy CUDA MAGMA logs.
    values = values.to(device="cpu", dtype=torch.float64)
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    if center:
        values = values - values.mean(dim=0, keepdim=True)
    try:
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "svdvals"):
            singular = torch.linalg.svdvals(values)
        else:
            singular = torch.svd(values).S
    except RuntimeError:
        try:
            gram = values.matmul(values.t()) if values.shape[0] <= values.shape[1] else values.t().matmul(values)
            gram = 0.5 * (gram + gram.t())
            if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigvalsh"):
                eigvals = torch.linalg.eigvalsh(gram)
            else:
                eigvals = torch.symeig(gram, eigenvectors=False).eigenvalues
            singular = eigvals.clamp_min(0.0).sqrt()
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
    compute_effective_rank: bool = False,
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
    if bool(compute_effective_rank):
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
    nearest_seen = seen_unseen_dist.min(dim=-1).values
    stats["graph_prob_prior_monitor_prior_seen_unseen_dist_mean"] = _as_float(seen_unseen_dist.mean())
    stats["graph_prob_prior_monitor_unseen_nearest_seen_distance_mean"] = _as_float(nearest_seen.mean())
    if unseen.numel() > 1:
        stats["graph_prob_prior_monitor_unseen_nearest_seen_ratio_mean"] = _as_float(
            (nearest_seen / nearest_unseen.clamp_min(1e-12)).mean()
        )

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
    只读取参数梯度的 detach 值，判断 prior/stats head 是否真的收到梯度。
    """
    groups = {
        "prior_head": (
            "prior_head",
            "factorized_semantic_prior_head",
            "residual_anchor_head",
            "residual_delta_head",
            "residual_logvar_head",
            "factorized_residual_anchor_head",
            "factorized_residual_delta_head",
            "factorized_residual_logvar_head",
        ),
        "learnable_scalar": (
            "learnable_prior_mu_scale_raw",
            "learnable_prior_delta_scale_raw",
            "learnable_tau_graph_raw",
            "learnable_tau_latent_raw",
            "learnable_geom_tau_dist_raw",
            "learnable_geom_bound_weight_raw",
            "learnable_geom_ord_margin_scale_raw",
            "learnable_geom_ord_non_overlap_weight_raw",
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


def factorized_health_monitor(
    semantic_mu: torch.Tensor,
    semantic_logvar: torch.Tensor,
    variation_mu: torch.Tensor,
    variation_logvar: torch.Tensor,
    targets_global: Optional[torch.Tensor] = None,
    compute_effective_rank: bool = False,
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
    if bool(compute_effective_rank):
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


def loss_scale_monitor(
    main_loss: Optional[float] = None,
    graph_prob_prior_loss: Optional[float] = None,
    loss_weight: float = 1.0,
) -> Dict[str, float]:
    """监测 GraphProbPrior 乘权重后的尺度，以及它相对主分类 loss 的强度。"""
    stats: Dict[str, float] = {}
    eps = 1e-12
    if graph_prob_prior_loss is not None:
        weighted = float(graph_prob_prior_loss) * float(loss_weight)
        stats["graph_prob_prior_monitor_loss_weighted_graph_prob_prior_loss"] = weighted
        if main_loss is not None:
            stats["graph_prob_prior_monitor_loss_weighted_gpp_to_main_loss_ratio"] = weighted / max(abs(float(main_loss)), eps)
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
