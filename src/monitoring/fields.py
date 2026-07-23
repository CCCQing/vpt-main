"""Field-level metadata and shared GraphProbPrior monitor aliases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class MetricFieldSpec:
    pattern: str
    description: str
    unit: str
    reducer: str
    required: bool = False


GPP_MONITOR_ALIASES = {
    "gEnt": "graph_prob_prior_monitor_neighbor_entropy_norm_mean",
    "gTop1": "graph_prob_prior_monitor_neighbor_top1_mean",
    "gTop5": "graph_prob_prior_monitor_neighbor_top5_mass_mean",
    "gHub": "graph_prob_prior_monitor_neighbor_hubness_gini",
    "gMean": "graph_prob_prior_monitor_graph_offdiag_mean",
    "gStd": "graph_prob_prior_monitor_graph_offdiag_std",
    "gQ95": "graph_prob_prior_monitor_graph_offdiag_q95",
    "g09": "graph_prob_prior_monitor_graph_pairs_gt_0_9_undirected",
    "g08": "graph_prob_prior_monitor_graph_pairs_gt_0_8_undirected",
    "gSelf": "graph_prob_prior_monitor_neighbor_self_mass_mean",
    "gMutual": "graph_prob_prior_monitor_neighbor_mutual_top5_ratio",
    "pMuMean": "graph_prob_prior_monitor_prior_mu_norm_mean",
    "pMuQ95": "graph_prob_prior_monitor_prior_mu_norm_q95",
    "pVarMean": "graph_prob_prior_monitor_prior_var_mean",
    "pVarQ95": "graph_prob_prior_monitor_prior_var_q95",
    "pDistQ50": "graph_prob_prior_monitor_prior_pair_mu_dist_q50",
    "pDistQ95": "graph_prob_prior_monitor_prior_pair_mu_dist_q95",
    "pCosMean": "graph_prob_prior_monitor_prior_center_collapse_score",
    "pCosMax": "graph_prob_prior_monitor_prior_pair_cosine_max_offdiag",
    "pCos09": "graph_prob_prior_monitor_prior_pair_cosine_pairs_gt_0_9",
    "pCollapse": "graph_prob_prior_monitor_prior_center_collapse_score",
    "pSymQ50": "graph_prob_prior_monitor_prior_symkl_q50",
    "pSymQ95": "graph_prob_prior_monitor_prior_symkl_q95",
    "pSymMin": "graph_prob_prior_monitor_prior_symkl_min_offdiag",
    "pOv": "graph_prob_prior_monitor_prior_overlap_risk_rate",
    "pRank": "graph_prob_prior_monitor_prior_effective_rank",
    "seenMu": "graph_prob_prior_monitor_prior_seen_mu_norm_mean",
    "unseenMu": "graph_prob_prior_monitor_prior_unseen_mu_norm_mean",
    "ssDist": "graph_prob_prior_monitor_prior_seen_seen_dist_mean",
    "uuDist": "graph_prob_prior_monitor_prior_unseen_unseen_dist_mean",
    "suDist": "graph_prob_prior_monitor_prior_seen_unseen_dist_mean",
    "unNearSeen": "graph_prob_prior_monitor_unseen_nearest_seen_distance_mean",
    "unNearRatio": "graph_prob_prior_monitor_unseen_nearest_seen_ratio_mean",
    "gpKRank": "graph_prob_prior_monitor_graph_gp_kernel_effective_rank",
    "gpKCond": "graph_prob_prior_monitor_graph_gp_kernel_condition",
    "gpSysCond": "graph_prob_prior_monitor_graph_gp_system_condition",
    "gpRowEnt": "graph_prob_prior_monitor_graph_gp_kernel_row_entropy_norm_mean",
    "gpKus1": "graph_prob_prior_monitor_graph_gp_k_us_support_top1_mean",
    "gpSmooth": "graph_prob_prior_monitor_graph_gp_smoothing_strength_mean",
    "gpMRank": "graph_prob_prior_monitor_graph_gp_mstar_effective_rank",
    "gpSp": "graph_prob_prior_monitor_graph_gp_mstar_graph_prior_spearman",
    "gpSolve": "graph_prob_prior_monitor_graph_gp_solve_residual",
    "gpObs": "graph_prob_prior_monitor_graph_gp_obs_noise_mean",
    "gpDynVar": "graph_prob_prior_monitor_graph_gp_prior_var_dynamic_mean",
    "gpClamp": "graph_prob_prior_monitor_graph_gp_prior_var_clamp_rate",
    "gpEce": "graph_prob_prior_graph_gp_energy_ce",
    "gpEacc": "graph_prob_prior_graph_gp_energy_acc",
    "gpEmar": "graph_prob_prior_graph_gp_energy_margin_mean",
    "gpPUloss": "graph_prob_prior_monitor_graph_gp_pseudo_unseen_energy_loss",
    "gpPUacc": "graph_prob_prior_monitor_graph_gp_pseudo_unseen_acc",
    "gpSupAcc": "graph_prob_prior_monitor_graph_gp_support_seen_acc",
    "gpGap": "graph_prob_prior_monitor_graph_gp_support_pseudo_acc_gap",
    "gpPUCos": "graph_prob_prior_monitor_graph_gp_pseudo_unseen_center_cos_mean",
    "gpPUMse": "graph_prob_prior_monitor_graph_gp_pseudo_unseen_center_mse_mean",
    "gpBias": "graph_prob_prior_monitor_graph_gp_seen_unseen_logit_bias_mean",
    "gzsl": "graph_prob_prior_monitor_prior_gzsl_unseen_to_seen_bias_risk_mean",
    "gGzsl": "graph_prob_prior_monitor_graph_gzsl_unseen_to_seen_bias_risk_mean",
    "fhG": "graph_prob_prior_monitor_false_high_graph_relation_still_gt_0_9_count",
    "fhP": "graph_prob_prior_monitor_false_high_prior_relation_still_gt_0_9_count",
    "wLoss": "graph_prob_prior_monitor_loss_weighted_graph_prob_prior_loss",
    "wRatio": "graph_prob_prior_monitor_loss_weighted_gpp_to_main_loss_ratio",
    "gStats": "graph_prob_prior_monitor_grad_stats_head_norm",
    "gFinite": "graph_prob_prior_monitor_grad_finite_ratio",
}
GPP_MONITOR_ALIAS_ITEMS = tuple(GPP_MONITOR_ALIASES.items())


MONITOR_FIELD_CATALOG: Dict[str, Tuple[MetricFieldSpec, ...]] = {
    "train": (
        MetricFieldSpec("loss", "当前训练 batch 的总损失", "loss", "sampled_batch", True),
        MetricFieldSpec("lr", "当前优化器学习率", "learning_rate", "sampled_batch", True),
    ),
    "train_epoch": (
        MetricFieldSpec("loss", "当前训练 epoch 的样本加权平均总损失", "loss", "sample_mean", True),
        MetricFieldSpec("lr", "当前训练 epoch 结束时的优化器学习率", "learning_rate", "last", True),
        MetricFieldSpec("batch_time_sec", "当前训练 epoch 的平均 batch 耗时", "seconds", "mean"),
        MetricFieldSpec("data_time_sec", "当前训练 epoch 的平均数据读取耗时", "seconds", "mean"),
    ),
    "train_debug": (
        MetricFieldSpec("ce_logits_std", "当前采样 batch 的 CE logits 总体标准差", "logit", "sampled_batch", True),
        MetricFieldSpec("ce_logits_abs_max", "当前采样 batch 的 CE logits 最大绝对值", "logit", "sampled_batch", True),
        MetricFieldSpec("ce_entropy", "当前采样 batch 的 CE softmax 平均熵", "nat", "sampled_batch", True),
        MetricFieldSpec("seen_only_top1", "当前采样 batch 在 seen-only 候选空间中的 top1", "ratio", "sampled_batch", True),
        MetricFieldSpec("effective_logit_scale", "分类头可学习 logit scale 的当前有效值", "scale", "sampled_batch"),
        MetricFieldSpec("ce_vs_raw_same_tensor", "CE logits 与模型 raw logits 是否为同一张量", "boolean", "one_time"),
        MetricFieldSpec("raw_logits_std", "仅路径不一致时记录的 raw logits 总体标准差", "logit", "one_time_if_path_diff"),
        MetricFieldSpec("raw_logits_abs_max", "仅路径不一致时记录的 raw logits 最大绝对值", "logit", "one_time_if_path_diff"),
        MetricFieldSpec("raw_entropy", "仅路径不一致时记录的 raw softmax 平均熵", "nat", "one_time_if_path_diff"),
    ),
    "numerical_guard": (
        MetricFieldSpec("*", "首个成功 step 或非有限数值失败证据", "event", "on_event", True),
    ),
    "optimizer_sanity": (
        MetricFieldSpec("phase", "optimizer 审计阶段", "event", "one_time", True),
        MetricFieldSpec("passed", "当前阶段最低检查是否通过", "boolean", "one_time", True),
        MetricFieldSpec("issue_count", "当前阶段异常或警告项总数", "count", "one_time", True),
        MetricFieldSpec("checks.*", "首个 step 的梯度、更新和冻结状态布尔检查", "boolean", "one_time"),
    ),
    "graph_prob_prior": (
        MetricFieldSpec("*", "GraphProbPrior loss stats 中所有有限标量", "scalar", "graph_prob_prior_forward"),
    ),
    "prompt_distribution": (
        MetricFieldSpec("{mu,logvar,std,prompt_tokens}_{mean,std,abs_mean,norm_mean}", "实例条件 prompt 分布摘要", "latent_value", "sampled_batch"),
    ),
    "prompt_parameter_health": (
        MetricFieldSpec("*", "静态 prompt 参数、梯度、更新和秩摘要", "scalar", "epoch"),
    ),
    "semantic_token_health": (
        MetricFieldSpec("*", "semantic token 尺度、槽多样性和输入输出关系", "scalar", "sampled_batch"),
    ),
    "attention_mediation": (
        MetricFieldSpec("*_mean", "跨 Transformer 层聚合的 Attention Mediation 标量", "scalar", "layer_mean"),
    ),
    "affinity_summary": (
        MetricFieldSpec("*.{mean,std,abs_mean}_layers_mean", "跨层聚合的 affinity 原始矩阵统计", "affinity", "layer_mean"),
    ),
    "auxiliary_loss_health": (
        MetricFieldSpec("*", "非主 CE 辅助损失的有限标量与相对强度", "loss", "sampled_batch"),
    ),
    "classification": (
        MetricFieldSpec("*", "Evaluator 在完整数据集上计算的分类指标", "metric", "dataset"),
    ),
    "prediction_health": (
        MetricFieldSpec("seen_unseen_logit_margin_mean", "最强 seen 与最强 unseen logit 的平均差", "logit", "dataset", True),
        MetricFieldSpec("seen_probability_mass_mean", "联合 softmax 中分配给 seen 类的平均概率质量", "probability", "dataset", True),
        MetricFieldSpec("wrong_domain_prediction_rate", "预测类别落入错误 seen/unseen 域的样本比例", "ratio", "dataset", True),
        MetricFieldSpec("true_class_margin_mean", "真实类别相对最强错误类别的平均 logit 余量", "logit", "dataset", True),
        MetricFieldSpec("true_class_rank_mean", "真实类别在联合候选空间中的平均排名", "rank", "dataset", True),
        MetricFieldSpec("entropy_mean", "联合 softmax 的平均熵", "nat", "dataset", True),
        MetricFieldSpec("confidence_incorrect", "错误预测样本的平均最大 softmax 概率", "probability", "dataset", True),
    ),
    "class_error": (
        MetricFieldSpec("bottom_k_class_mean", "准确率最低 10% 已观测类别的平均准确率", "ratio", "dataset", True),
        MetricFieldSpec("max_prediction_share", "预测次数最多类别占全部预测的比例", "ratio", "dataset", True),
    ),
    "calibration_profile": (
        MetricFieldSpec("ausuc", "固定 gamma 网格上 Seen-Unseen 曲线的诊断面积", "area", "dataset", True),
        MetricFieldSpec("raw_to_oracle_gain", "原始 H 到事后校准峰值 H 的差值", "ratio", "dataset", True),
        MetricFieldSpec("oracle_peak_gamma", "final_gzsl 诊断曲线的事后峰值位置", "gamma", "dataset", True),
    ),
    "checkpoint_selection_debug": (
        MetricFieldSpec("historical_independent_best_upper_bound", "不同 epoch 独立最佳 S/U 拼接的诊断上界", "metric", "history", True),
    ),
}


def graph_prob_prior_metric_group(field: str) -> str:
    name = str(field).lower()
    if "grad_" in name or "gradient" in name:
        return "gpp_gradient_health"
    if "graph_gp_" in name or "_energy_" in name:
        return "gpp_graph_gp"
    if "prior_seen" in name or "prior_unseen" in name or "gzsl" in name:
        return "gpp_seen_unseen_bias"
    if "prior_" in name or "overlap" in name or "effective_rank" in name:
        return "gpp_prior_geometry"
    if "graph_" in name and ("spearman" in name or "neighbor" in name or "false_high" in name):
        return "gpp_structure_transfer"
    if "graph_" in name or "neighbor_" in name:
        return "gpp_graph_health"
    if "loss" in name or "weighted" in name:
        return "gpp_loss_balance"
    return "gpp_other"


def field_catalog_manifest() -> Dict[str, object]:
    catalog: Dict[str, object] = {
        namespace: [asdict(field) for field in fields]
        for namespace, fields in MONITOR_FIELD_CATALOG.items()
    }
    catalog["graph_prob_prior_console_aliases"] = [
        {"alias": alias, "field": field}
        for alias, field in GPP_MONITOR_ALIAS_ITEMS
    ]
    catalog["graph_prob_prior_metric_groups"] = {
        field[len("graph_prob_prior_"):] if field.startswith("graph_prob_prior_") else field: {
            "metric_group": graph_prob_prior_metric_group(field),
            "entity_type": "runtime",
            "entity_id": "global",
        }
        for field in GPP_MONITOR_ALIASES.values()
    }
    return catalog
