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
    "gEnt": "graph_prob_prior_monitor_tau_graph_neighbor_entropy_norm_mean",
    "gTop1": "graph_prob_prior_monitor_tau_graph_neighbor_top1_mean",
    "gTop5": "graph_prob_prior_monitor_tau_graph_neighbor_top5_mass_mean",
    "gHub": "graph_prob_prior_monitor_neighbor_hubness_gini",
    "gMean": "graph_prob_prior_monitor_graph_offdiag_mean",
    "gStd": "graph_prob_prior_monitor_graph_offdiag_std",
    "gQ95": "graph_prob_prior_monitor_graph_offdiag_q95",
    "g09": "graph_prob_prior_monitor_graph_pairs_gt_0_9_undirected",
    "g08": "graph_prob_prior_monitor_graph_pairs_gt_0_8_undirected",
    "gSelf": "graph_prob_prior_monitor_neighbor_self_mass_mean",
    "gMutual": "graph_prob_prior_monitor_neighbor_mutual_top5_ratio",
    "tLogQ50": "graph_prob_prior_monitor_tau_acc_topk_logits_q50",
    "tSemEnt": "graph_prob_prior_monitor_tau_acc_target_sem_entropy_norm_mean",
    "tSemTop1": "graph_prob_prior_monitor_tau_acc_target_sem_top1_mean",
    "tSemTrue": "graph_prob_prior_monitor_tau_acc_target_sem_true_mean",
    "tEnt": "graph_prob_prior_monitor_tau_acc_target_entropy_norm_mean",
    "tTop1": "graph_prob_prior_monitor_tau_acc_target_top1_mean",
    "tTop5": "graph_prob_prior_monitor_tau_acc_target_top5_mass_mean",
    "tTrue": "graph_prob_prior_monitor_tau_acc_target_true_mean",
    "lEnt": "graph_prob_prior_monitor_tau_latent_entropy_norm_mean",
    "lTop1": "graph_prob_prior_monitor_tau_latent_top1_mean",
    "lTop5": "graph_prob_prior_monitor_tau_latent_top5_mass_mean",
    "lTrue": "graph_prob_prior_monitor_tau_latent_true_mean",
    "klQ50": "graph_prob_prior_monitor_tau_latent_distance_q50",
    "klQ95": "graph_prob_prior_monitor_tau_latent_distance_q95",
    "rank1": "graph_prob_prior_monitor_posterior_true_rank_top1",
    "rank5": "graph_prob_prior_monitor_posterior_true_rank_top5",
    "rank10": "graph_prob_prior_monitor_posterior_true_rank_top10",
    "rankQ50": "graph_prob_prior_monitor_posterior_true_rank_q50",
    "klTrueQ50": "graph_prob_prior_monitor_posterior_kl_true_q50",
    "klWrongQ50": "graph_prob_prior_monitor_posterior_kl_nearest_wrong_q50",
    "klMargin": "graph_prob_prior_monitor_posterior_kl_margin_mean",
    "marginPos": "graph_prob_prior_monitor_posterior_kl_margin_positive_ratio",
    "pMuMean": "graph_prob_prior_monitor_prior_mu_norm_mean",
    "pMuQ95": "graph_prob_prior_monitor_prior_mu_norm_q95",
    "pVarMean": "graph_prob_prior_monitor_prior_var_mean",
    "pVarQ95": "graph_prob_prior_monitor_prior_var_q95",
    "pDistQ50": "graph_prob_prior_monitor_prior_pair_mu_dist_q50",
    "pDistQ95": "graph_prob_prior_monitor_prior_pair_mu_dist_q95",
    "pCosMean": "graph_prob_prior_monitor_prior_pair_cosine_mean",
    "pCosQ95": "graph_prob_prior_monitor_prior_pair_cosine_q95",
    "pCos09": "graph_prob_prior_monitor_prior_pair_cosine_pairs_gt_0_9",
    "pCollapse": "graph_prob_prior_monitor_prior_center_collapse_score",
    "pSymQ50": "graph_prob_prior_monitor_prior_symkl_q50",
    "pSymQ95": "graph_prob_prior_monitor_prior_symkl_q95",
    "pSymMin": "graph_prob_prior_monitor_prior_symkl_min_offdiag",
    "pOv": "graph_prob_prior_monitor_prior_overlap_risk_rate",
    "pRank": "graph_prob_prior_monitor_prior_effective_rank",
    "pEnt": "graph_prob_prior_monitor_tau_prior_entropy_norm_mean",
    "rAttrMean": "graph_prob_prior_monitor_residual_attr_norm_mean",
    "rAttrQ95": "graph_prob_prior_monitor_residual_attr_norm_q95",
    "aCosMean": "graph_prob_prior_monitor_anchor_pair_cosine_mean",
    "aCosQ95": "graph_prob_prior_monitor_anchor_pair_cosine_q95",
    "aCos09": "graph_prob_prior_monitor_anchor_pairs_gt_0_9",
    "paCosMean": "graph_prob_prior_monitor_prior_anchor_cos_mean",
    "paCosQ50": "graph_prob_prior_monitor_prior_anchor_cos_q50",
    "dNormMean": "graph_prob_prior_monitor_prior_delta_norm_mean",
    "dNormQ95": "graph_prob_prior_monitor_prior_delta_norm_q95",
    "dRatioMean": "graph_prob_prior_monitor_prior_delta_to_anchor_ratio_mean",
    "dRatioQ95": "graph_prob_prior_monitor_prior_delta_to_anchor_ratio_q95",
    "ctxCosMean": "graph_prob_prior_monitor_context_anchor_cos_mean",
    "gPriorSp": "graph_prob_prior_monitor_graph_pos_prior_relation_spearman",
    "hardCosQ95": "graph_prob_prior_monitor_hardneg_prior_cos_q95",
    "hardV": "graph_prob_prior_monitor_hardneg_violate_rate",
    "neiPres": "graph_prob_prior_monitor_true_neighbor_preservation_mean",
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
    "gpKus5": "graph_prob_prior_monitor_graph_gp_k_us_support_top5_mass_mean",
    "gpSmooth": "graph_prob_prior_monitor_graph_gp_smoothing_strength_mean",
    "gpMRank": "graph_prob_prior_monitor_graph_gp_mstar_effective_rank",
    "gpSp": "graph_prob_prior_monitor_graph_gp_mstar_graph_prior_spearman",
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
    "mmdK": "graph_prob_prior_monitor_mmd_kernel_mean",
    "mmdKStd": "graph_prob_prior_monitor_mmd_kernel_std",
    "mmdLow": "graph_prob_prior_monitor_mmd_kernel_saturation_low_ratio",
    "mmdHigh": "graph_prob_prior_monitor_mmd_kernel_saturation_high_ratio",
    "mmdD50": "graph_prob_prior_monitor_mmd_pair_dist_q50",
    "mmdD95": "graph_prob_prior_monitor_mmd_pair_dist_q95",
    "agg1": "graph_prob_prior_monitor_agg_single_sample_class_ratio",
    "facCov": "graph_prob_prior_monitor_factorized_cross_cov_fro",
    "facRatio": "graph_prob_prior_monitor_factorized_semantic_variation_norm_ratio",
    "facSemRank": "graph_prob_prior_monitor_factorized_semantic_batch_effective_rank",
    "facVarRank": "graph_prob_prior_monitor_factorized_variation_batch_effective_rank",
    "facClass": "graph_prob_prior_monitor_factorized_variation_class_ratio",
    "gzsl": "graph_prob_prior_monitor_prior_gzsl_unseen_to_seen_bias_risk_mean",
    "gGzsl": "graph_prob_prior_monitor_graph_gzsl_unseen_to_seen_bias_risk_mean",
    "fhG": "graph_prob_prior_monitor_false_high_graph_relation_still_gt_0_9_count",
    "fhP": "graph_prob_prior_monitor_false_high_prior_relation_still_gt_0_9_count",
    "wLoss": "graph_prob_prior_monitor_loss_weighted_graph_prob_prior_loss",
    "wRatio": "graph_prob_prior_monitor_loss_weighted_gpp_to_main_loss_ratio",
    "muS": "graph_prob_prior_monitor_learnable_prior_mu_scale_value",
    "dS": "graph_prob_prior_monitor_learnable_prior_delta_scale_value",
    "tauG": "graph_prob_prior_monitor_learnable_tau_graph_value",
    "tauL": "graph_prob_prior_monitor_learnable_tau_latent_value",
    "gTauD": "graph_prob_prior_monitor_learnable_geom_tau_dist_value",
    "gBound": "graph_prob_prior_monitor_learnable_geom_bound_weight_value",
    "ordM": "graph_prob_prior_monitor_learnable_geom_ord_margin_scale_value",
    "ordO": "graph_prob_prior_monitor_learnable_geom_ord_non_overlap_weight_value",
    "gPrior": "graph_prob_prior_monitor_grad_prior_head_norm",
    "gAnchor": "graph_prob_prior_monitor_grad_anchor_head_norm",
    "gDelta": "graph_prob_prior_monitor_grad_delta_head_norm",
    "gScalar": "graph_prob_prior_monitor_grad_learnable_scalar_norm",
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
        MetricFieldSpec("*", "Trainer 与分类损失暴露的标量调试项", "scalar", "sampled_batch"),
    ),
    "numerical_guard": (
        MetricFieldSpec("*", "首个成功 step 或非有限数值失败证据", "event", "on_event", True),
    ),
    "optimizer_sanity": (
        MetricFieldSpec("*", "optimizer 参数归属、梯度和首次更新证据", "event", "one_time", True),
    ),
    "protocol_access": (
        MetricFieldSpec("*", "评测 split 的访问时机和用途", "event", "on_access", True),
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
        MetricFieldSpec("*", "联合类别空间中的 seen bias、margin、rank、entropy 和 confidence", "metric", "dataset"),
    ),
    "class_error": (
        MetricFieldSpec("*", "逐类错误 artifact 的标量摘要", "metric", "dataset"),
    ),
    "calibration_profile": (
        MetricFieldSpec("*", "固定 gamma 网格上的 final_gzsl 校准曲线摘要", "metric", "dataset"),
    ),
    "checkpoint_selection_debug": (
        MetricFieldSpec("historical_independent_best_upper_bound", "不同 epoch 独立最佳 S/U 拼接的诊断上界", "metric", "history", True),
    ),
}


def graph_prob_prior_metric_group(field: str) -> str:
    name = str(field).lower()
    if "grad_" in name or "gradient" in name:
        return "gpp_gradient_health"
    if "posterior" in name or "kl_true" in name or "kl_nearest_wrong" in name:
        return "gpp_posterior_quality"
    if "prior_seen" in name or "prior_unseen" in name or "gzsl" in name:
        return "gpp_seen_unseen_bias"
    if "anchor" in name or "delta" in name or "context_anchor" in name:
        return "gpp_anchor_delta"
    if "prior_" in name or "overlap" in name or "effective_rank" in name:
        return "gpp_prior_geometry"
    if "graph_" in name and ("spearman" in name or "neighbor" in name or "hardneg" in name or "false_high" in name):
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
