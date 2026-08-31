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
    "gpKSeen1": "graph_prob_prior_monitor_graph_gp_k_all_seen_top1_mean",
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
    "gpSeenAcc": "graph_prob_prior_monitor_graph_gp_seen_acc",
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
    "fixed_probe_runtime": (
        MetricFieldSpec("probe_total_time_sec", "固定Probe对应阶段的总墙钟耗时", "seconds", "sum"),
        MetricFieldSpec("probe_data_time_sec", "DataLoader创建迭代器及取得batch的累计等待时间", "seconds", "sum"),
        MetricFieldSpec("probe_compute_time_sec", "总耗时扣除DataLoader等待后的模型执行、数据传输与聚合时间", "seconds", "sum"),
        MetricFieldSpec("probe_data_time_ratio", "DataLoader等待时间占固定Probe总耗时的比例", "ratio", "derived"),
        MetricFieldSpec("batch_count", "固定Probe对应阶段实际消费的batch数", "count", "sum"),
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
        MetricFieldSpec("sampling_performed", "本次forward是否实际从分布生成采样Prompt", "boolean", "sampled_batch"),
    ),
    "deep_prompt_residual": (
        MetricFieldSpec("layer_*.gate", "逐层确定性均值修正门控", "scalar", "sampled_batch"),
        MetricFieldSpec("layer_*.layer_gate", "不含样本级门控与回放缩放的逐层基础门控", "scalar", "sampled_batch"),
        MetricFieldSpec("layer_*.sample_gate_{mean,std}", "样本自适应门控的批内均值与离散程度；未启用时固定为1", "scalar", "sampled_batch"),
        MetricFieldSpec("layer_*.sample_gate_{low,high}_saturation_rate", "样本门控接近关闭或完全打开的样本比例", "sample_ratio", "sampled_batch"),
        MetricFieldSpec("layer_*.runtime_scale", "checkpoint-only强度、分组或逐层回放施加的额外缩放", "scalar", "sampled_batch"),
        MetricFieldSpec("layer_*.{base_prompt_norm,raw_delta_norm,applied_delta_norm}", "逐层静态 Prompt 与直接均值修正的尺度", "feature_norm", "sampled_batch"),
        MetricFieldSpec("layer_*.applied_delta_to_base_ratio", "实际注入残差相对静态 Prompt 的尺度", "ratio", "sampled_batch"),
        MetricFieldSpec("layer_*.raw_delta_between_instance_variance", "同层残差在不同样本间是否真的变化", "variance", "sampled_batch"),
        MetricFieldSpec("layer_*.raw_delta_slot_variance", "同一图像的动态残差在不同Prompt槽之间是否形成内容差异；shared结构应为0", "variance", "sampled_batch"),
        MetricFieldSpec("layer_*.raw_delta_slot_effective_rank", "同一图像逐槽残差实际使用的等效方向数；shared非零结构应为1", "effective_count", "sampled_batch"),
        MetricFieldSpec("layers_{mean,max}.applied_delta_to_base_ratio", "跨层汇总的残差相对静态Prompt注入幅度", "ratio", "sampled_batch"),
        MetricFieldSpec("layers_{mean,max_abs}.gate", "跨层汇总的实际门控强度", "scalar", "sampled_batch"),
        MetricFieldSpec("layers_samples.sample_gate_{mean,std}", "跨层和样本汇总的自适应门控均值与离散程度", "scalar", "sampled_batch"),
    ),
    "deep_prompt_residual_experiments": (
        MetricFieldSpec("residual_static_cosine", "最终残差与同槽静态Prompt的方向相似度", "cosine", "checkpoint_probe_sample_layer_slot"),
        MetricFieldSpec("signed_parallel_projection", "最终残差沿同槽静态Prompt方向的有符号投影长度", "feature_norm", "checkpoint_probe_sample_layer_slot"),
        MetricFieldSpec("orthogonal_component_ratio", "残差中不沿静态Prompt原方向的新方向占比", "ratio", "checkpoint_probe_sample_layer_slot"),
        MetricFieldSpec("residual_static_norm_ratio", "最终残差长度相对同槽静态Prompt长度的比例", "ratio", "checkpoint_probe_sample_layer_slot"),
        MetricFieldSpec("slot_shift_effective_rank", "同层各Prompt槽动态残差实际形成的等效方向数", "effective_count", "checkpoint_probe_sample_layer"),
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
    "loss_component_trajectory": (
        MetricFieldSpec("*.raw", "当前损失分量未乘权重的epoch样本加权均值", "loss", "sample_mean"),
        MetricFieldSpec("*.weight", "当前损失分量实际使用的固定权重", "weight", "last"),
        MetricFieldSpec("*.weighted", "权重乘原始损失后的epoch样本加权均值", "loss", "sample_mean"),
        MetricFieldSpec("*.weighted_share", "加权分量相对总损失绝对值的比例", "ratio", "sample_mean"),
        MetricFieldSpec("total_loss", "所有损失分量相加后的epoch样本加权均值", "loss", "sample_mean", True),
    ),
    "multi_loss_gradient_audit": (
        MetricFieldSpec("*.grad_norm_raw", "未乘loss权重时，该loss对指定参数块的独立梯度范数", "gradient_norm", "milestone_batch"),
        MetricFieldSpec("*.grad_norm_weighted", "乘实际loss权重后，该loss对指定参数块的梯度力量", "gradient_norm", "milestone_batch"),
        MetricFieldSpec("*.weighted_grad_norm_ratio_vs_primary", "辅助loss加权梯度相对主CE梯度的力量比例", "ratio", "milestone_batch"),
        MetricFieldSpec("*.grad_cosine", "两个loss在同一参数块上的梯度方向余弦", "cosine", "milestone_batch"),
        MetricFieldSpec("*.gradient_cancellation_ratio", "合并梯度长度相对各分量梯度长度之和的比例", "ratio", "milestone_batch"),
        MetricFieldSpec("*.optimizer_descent_alignment", "实际参数更新与降低指定loss方向的余弦", "cosine", "milestone_batch"),
        MetricFieldSpec("*.status", "观测、无计算路径、零梯度、未请求或无效的明确状态", "status", "milestone_batch", True),
    ),
    "semantic_graph_reference": (
        MetricFieldSpec("projection_relation_spearman", "312维原始属性与768维投影语义的类间关系排序保留程度", "spearman", "fixed_probe_class_relation"),
        MetricFieldSpec("projection_neighbor_preservation_at_{1,5}", "投影前后每类最近语义邻居的保留比例", "ratio", "fixed_probe_class_relation"),
        MetricFieldSpec("projection_{new_false_high,dropped_high}_edge_rate", "投影新增错误强边或丢失原有强边的比例", "ratio", "fixed_probe_class_relation"),
        MetricFieldSpec("semantic_relation_normalized_effective_rank_{312,768}", "312/768语义关系矩阵去均值后的归一化有效秩", "ratio", "fixed_probe_class_relation"),
        MetricFieldSpec("semantic_visual_relation_spearman_{312,768}", "312/768语义类间关系与当前视觉类别中心关系的排序一致性", "spearman", "fixed_probe_split_class_relation"),
        MetricFieldSpec("semantic_visual_neighbor_overlap_at_5_{312,768}", "312/768语义近邻与视觉类别中心近邻的重合比例", "ratio", "fixed_probe_split_class_relation"),
        MetricFieldSpec("false_high_semantic_edge_rate_{312,768}", "语义认为高度相似但当前视觉类别中心并不相似的边比例", "ratio", "fixed_probe_split_class_relation"),
        MetricFieldSpec("*_delta_768_minus_312", "同口径下768维投影值减312维原始属性值；方向须结合具体指标判断", "delta", "derived"),
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
    "epoch_prediction_transition": (
        MetricFieldSpec("transition_available", "当前epoch是否存在可配对的上一epoch样本状态", "boolean", "dataset", True),
        MetricFieldSpec("prediction_flip_rate", "相邻epoch预测类别发生改变的样本比例", "ratio", "dataset"),
        MetricFieldSpec("correction_rate", "上一epoch错误且当前epoch正确的样本比例", "ratio", "dataset"),
        MetricFieldSpec("regression_rate", "上一epoch正确且当前epoch错误的样本比例", "ratio", "dataset"),
        MetricFieldSpec("net_correction_rate", "correction_rate减regression_rate", "ratio", "dataset"),
        MetricFieldSpec("persistent_correct_rate", "相邻两个epoch均预测正确的样本比例", "ratio", "dataset"),
        MetricFieldSpec("persistent_wrong_rate", "相邻两个epoch均预测错误的样本比例", "ratio", "dataset"),
        MetricFieldSpec("ever_correct_then_wrong_rate", "此前至少正确过但当前epoch错误的样本比例", "ratio", "dataset"),
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
