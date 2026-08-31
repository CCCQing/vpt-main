#!/usr/bin/env python3
"""Graph-GP runtime statistics.

This module converts detached intermediate tensors and gradients to log scalars.
It does not participate in loss computation or alter gradients.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

def _as_float(value: torch.Tensor) -> float:
    """把 0 维 tensor 安全转成 Python float，便于写入 json/csv/log。"""
    return float(value.detach().float().cpu().item())

def tensor_stats(prefix: str, x: torch.Tensor) -> Dict[str, float]:
    """统计任意实值张量的有限比例、范围和分位数。"""
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

    这里不用 torch.isin，是为了兼容旧版 PyTorch。监测需要按类别集合筛选
    当前 batch，所以用广播比较构造布尔 mask。
    """
    if not torch.is_tensor(values) or not torch.is_tensor(members) or members.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    members = members.to(device=values.device, dtype=values.dtype)
    return (values[:, None] == members[None, :]).any(dim=1)

def _matrix_condition_number(x: torch.Tensor) -> float:
    """用奇异值估计条件数；只作为监测项，失败时返回 0。"""
    if not torch.is_tensor(x) or x.dim() != 2 or x.numel() == 0:
        return 0.0
    values = x.detach().to(device="cpu", dtype=torch.float32)
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
    values = values.to(device="cpu", dtype=torch.float32)
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

def _index_tensor(ids, device: torch.device) -> torch.Tensor:
    """把 list/tuple/numpy/tensor 形式的类别 id 统一成 long tensor。"""
    if ids is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if torch.is_tensor(ids):
        out = ids.detach().to(device=device, dtype=torch.long).view(-1)
    else:
        out = torch.as_tensor(ids, device=device, dtype=torch.long).view(-1)
    return out[out >= 0]

def graph_gp_prototype_monitor(
    prior_mu: torch.Tensor,
    prior_logvar: Optional[torch.Tensor],
    seen_ids: torch.Tensor,
    observed_seen_ids: torch.Tensor,
    seen_count: torch.Tensor,
    center_var: torch.Tensor,
    obs_noise: torch.Tensor,
    solve_residual: torch.Tensor,
    uncertainty_diag: torch.Tensor,
    system_diag_ratio: torch.Tensor,
    seen_post_var: Optional[torch.Tensor] = None,
    seen_visual_var: Optional[torch.Tensor] = None,
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
    1. 官方 Seen 类中有多少类已经被 posterior_mu 观测到；
    2. Seen 类中心的类内方差和观测噪声 R_s 是否异常；
    3. 线性方程求解是否稳定，Graph-GP predictive uncertainty 是否过大；
    4. 推断出的 M_star 是否仍然同向扎堆，以及 Seen 样本能否在 energy CE 中找到真类 prototype。
    """
    stats: Dict[str, float] = {}
    if not torch.is_tensor(prior_mu):
        return stats

    device = prior_mu.device
    seen_ids = seen_ids.to(device=device, dtype=torch.long) if torch.is_tensor(seen_ids) else torch.empty(0, device=device, dtype=torch.long)
    observed_seen_ids = (
        observed_seen_ids.to(device=device, dtype=torch.long)
        if torch.is_tensor(observed_seen_ids)
        else torch.empty(0, device=device, dtype=torch.long)
    )

    stats[f"{prefix}_seen_class_count"] = float(seen_ids.numel())
    stats[f"{prefix}_seen_observed_class_count"] = float(observed_seen_ids.numel())
    stats[f"{prefix}_seen_observed_ratio"] = float(observed_seen_ids.numel()) / float(max(int(seen_ids.numel()), 1))
    if seen_ids.numel() > 0 and torch.is_tensor(seen_count):
        selected_count = seen_count.detach().float().to(device=device).index_select(0, seen_ids)
        stats.update(tensor_stats(f"{prefix}_seen_count", selected_count))

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
    if torch.is_tensor(seen_post_var) and seen_post_var.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_seen_post_var", seen_post_var))
    if torch.is_tensor(seen_visual_var) and seen_visual_var.numel() > 0:
        stats.update(tensor_stats(f"{prefix}_seen_visual_var", seen_visual_var))
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
        seen_prob = _row_probability_from_nonnegative(k_all_s.to(device=device))
        stats.update(probability_stats(f"{prefix}_k_all_seen", seen_prob, topk=topk))

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

            seen_mask = _membership_mask(targets, seen_ids)
            if bool(seen_mask.any().item()):
                stats[f"{prefix}_seen_rank1"] = _as_float(hit[seen_mask].mean())
                stats[f"{prefix}_seen_acc"] = stats[f"{prefix}_seen_rank1"]
                stats[f"{prefix}_seen_true_kl_mean"] = _as_float(true_dist[seen_mask].mean())
                stats.update(tensor_stats(f"{prefix}_seen_margin", margin[seen_mask]))
                if torch.is_tensor(sample_energy_loss) and sample_energy_loss.numel() == targets.numel():
                    stats[f"{prefix}_seen_energy_loss"] = _as_float(
                        sample_energy_loss.detach().float().to(device=targets.device)[seen_mask].mean()
                    )

            if torch.is_tensor(posterior_mu) and posterior_mu.dim() == 2 and posterior_mu.shape[0] == targets.numel():
                stats.update(
                    _batch_class_center_stats(
                        f"{prefix}_seen",
                        posterior_mu,
                        prior_mu,
                        targets,
                        seen_mask,
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

def graph_health_monitor(
    graph: torch.Tensor,
    topk: int = 5,
    exclude_diag: bool = True,
) -> Dict[str, float]:
    """Summarize the raw Graph-GP kernel without an obsolete aggregation temperature."""
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

    neighbor = _row_probability_from_nonnegative(graph)
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
    """Report finite gradients and prompt-distributor stats-head gradient norm."""
    sum_sq = 0.0
    count = 0
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
        if "stats_head" in name:
            finite_g = g[finite]
            if finite_g.numel() > 0:
                sum_sq += float(finite_g.pow(2).sum().item())
                count += int(finite_g.numel())
    return {
        "graph_prob_prior_monitor_grad_stats_head_norm": math.sqrt(max(sum_sq, 0.0)),
        "graph_prob_prior_monitor_grad_stats_head_param_count": float(count),
        "graph_prob_prior_monitor_grad_finite_ratio": float(finite_num) / float(max(finite_den, 1)),
    }

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
