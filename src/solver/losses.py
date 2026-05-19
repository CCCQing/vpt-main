#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional


def _extract_logits_and_aux(pred_logits: Any, kwargs: Optional[Dict[str, Any]]):
    """
    统一解析模型输出。

    返回:
    - logits: 分类分数，通常形状为 [B, C]
    - aux: 可选辅助字典，主要保存 affinity / attention 监测量
    """
    aux = None
    logits = pred_logits

    if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
        logits = pred_logits[0]
        if len(pred_logits) > 1 and isinstance(pred_logits[1], Dict):
            aux = pred_logits[1]

    if isinstance(pred_logits, Dict) and "logits" in pred_logits:
        logits = pred_logits["logits"]
        aux = pred_logits

    if aux is None and kwargs is not None and isinstance(kwargs, Dict):
        aux = kwargs.get("aux")

    return logits, aux


def _effective_scale_from_model(model: Optional[nn.Module], logits: torch.Tensor) -> torch.Tensor:
    """
    从分类头读取当前实际使用的 logit scale。

    该值只用于诊断打印，不直接参与当前 loss 的数值计算。
    """
    scale = logits.new_tensor(1.0)
    if model is None:
        return scale
    r_head = model.r_similarity_head

    fixed_scale = float(r_head.fixed_logit_scale)
    if fixed_scale > 0:
        return logits.new_tensor(fixed_scale)

    logit_scale = r_head.logit_scale
    if logit_scale is not None:
        return logit_scale.exp().to(device=logits.device, dtype=logits.dtype)
    return scale


def _compute_cm_loss_from_rhead_cache(
    model: Optional[nn.Module],
    targets: Optional[torch.Tensor],
    logits: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    计算当前主线里的 CM loss。

    数学形式：
        L_cm = mean || normalize(v_i) - normalize(s_{y_i}) ||_2^2

    这里有两个关键约束：
    1. v_i 和 s_{y_i} 读取 `r_similarity_head` 在本次 forward 中缓存下来的：
         - `_loss_last_visual_repr`
         - `_loss_last_semantic_repr`
    2. targets 必须是“当前 active class space 下的局部标签”。
       因为 `semantic_bank` 的第 0..C-1 列，只对应这次 forward 实际参与分类的类空间。

    """
    if model is None or targets is None:
        return None
    r_head = model.r_similarity_head
    visual = r_head._loss_last_visual_repr
    semantic_bank = r_head._loss_last_semantic_repr
    if visual is None or semantic_bank is None:
        return None
    if (not torch.is_tensor(visual)) or (not torch.is_tensor(semantic_bank)):
        return None
    if visual.dim() != 2 or semantic_bank.dim() != 2:
        return None
    if visual.shape[0] != logits.shape[0]:
        return None

    y = targets.to(device=visual.device, dtype=torch.long)
    if y.shape[0] != visual.shape[0]:
        return None
    if y.min().item() < 0 or y.max().item() >= semantic_bank.shape[0]:
        return None

    s_pos = semantic_bank.index_select(0, y)
    v_n = F.normalize(visual, dim=-1)
    s_n = F.normalize(s_pos, dim=-1)
    return ((v_n - s_n) ** 2).sum(dim=-1).mean()


def _compute_consistency_loss_from_rhead_cache(model: Optional[nn.Module], dist_type: str = "cosine",) -> Optional[torch.Tensor]:
    """
    从分类头缓存中计算语义增量一致性损失。

    比较对象:
    - semantic branch 产生的语义增量 delta_sem
    - consistency_head 从原始属性预测出的目标方向
    """
    if model is None:
        return None
    r_head = model.r_similarity_head
    semantic_output = r_head._loss_last_semantic_final
    semantic_input = r_head._loss_last_semantic_anchor
    if torch.is_tensor(semantic_output) and torch.is_tensor(semantic_input) and semantic_output.shape == semantic_input.shape:
        delta_sem = semantic_output - semantic_input
    else:
        delta_sem = r_head._loss_last_semantic_delta
    target = r_head._loss_last_consistency_target
    if (not torch.is_tensor(delta_sem)) or (not torch.is_tensor(target)):
        return None
    if delta_sem.numel() == 0 or target.numel() == 0:
        return None
    if delta_sem.shape != target.shape:
        return None
    if str(dist_type).lower() == "l2":
        return ((delta_sem - target) ** 2).sum(dim=-1).mean()
    # cosine (default)
    d = F.normalize(delta_sem, dim=-1)
    t = F.normalize(target, dim=-1)
    return (1.0 - (d * t).sum(dim=-1)).mean()


def _normalize_affinity_for_aux_loss(x: torch.Tensor, norm_type: str, cfg_name: str) -> torch.Tensor:
    """
    对亲和矩阵做归一化，供 semantic-mediated loss 对齐使用。

    当前第一版只支持 softmax，避免多种归一化混用导致实验含义不清。
    """
    norm_type = str(norm_type).lower()
    if norm_type != "softmax":
        raise ValueError(f"Unsupported {cfg_name}.NORM='{norm_type}'. The first version only supports softmax.")
    return F.softmax(x, dim=-1)


def _semantic_mediated_distance(
    student: torch.Tensor,
    teacher: torch.Tensor,
    metric: str,
    cfg_name: str = "SOLVER.SEM_MED.METRIC",
) -> torch.Tensor:
    """
    计算两个归一化亲和矩阵之间的距离。

    支持 mse / cosine / kl 三种度量。
    """
    metric = str(metric).lower()
    if metric == "mse":
        return F.mse_loss(student, teacher)
    if metric == "cosine":
        student_flat = student.flatten(1)
        teacher_flat = teacher.flatten(1)
        return (1.0 - F.cosine_similarity(student_flat, teacher_flat, dim=-1)).mean()
    if metric == "kl":
        return F.kl_div(torch.log(student.clamp_min(1e-8)), teacher, reduction="batchmean")
    raise ValueError(f"Unsupported {cfg_name}='{metric}'. Expected mse / kl / cosine.")


def _compute_semantic_mediated_affinity_loss(aux: Optional[Dict[str, Any]], cfg) -> torch.Tensor:
    """
    L_sem_med aligns direct prompt-visual affinity with semantic-mediated affinity:
        A_psv = QsKp^T @ QsKv

    Shapes after trainer head-mean:
        QsKp: [B, S, P]
        QsKv: [B, S, V]
        Apv : [B, P, V]
    """
    if not isinstance(aux, Dict):
        raise RuntimeError("Semantic-mediated affinity loss requires affinity aux dict.")

    target = str(cfg.SOLVER.SEM_MED.TARGET)
    target_map = {
        "QpKv": "aff_qpkv",
        "QpQv": "aff_qpqv",
        "KpKv": "aff_kpkv",
    }
    if target not in target_map:
        raise ValueError(f"Unsupported SOLVER.SEM_MED.TARGET='{target}'. Expected QpKv / QpQv / KpKv.")

    required_keys = ["aff_qskp", "aff_qskv", target_map[target]]
    missing = [k for k in required_keys if k not in aux or not isinstance(aux[k], Dict)]
    if missing:
        raise RuntimeError(f"Semantic-mediated affinity loss missing aux keys: {missing}.")

    qskp_layers = aux["aff_qskp"]
    qskv_layers = aux["aff_qskv"]
    apv_layers = aux[target_map[target]]

    shared_layers = sorted(set(qskp_layers.keys()) & set(qskv_layers.keys()) & set(apv_layers.keys()))
    requested_layers = list(cfg.SOLVER.SEM_MED.LAYERS)
    if requested_layers:
        requested_layers = [int(x) for x in requested_layers]
        shared_layers = [x for x in shared_layers if x in requested_layers]
    if not shared_layers:
        raise RuntimeError("Semantic-mediated affinity loss found no shared layers.")

    metric = str(cfg.SOLVER.SEM_MED.METRIC).lower()
    norm_type = str(cfg.SOLVER.SEM_MED.NORM).lower()
    detach_mode = str(cfg.SOLVER.SEM_MED.DETACH).lower()
    if detach_mode not in {"mediated", "direct", "none"}:
        raise ValueError(f"Unsupported SOLVER.SEM_MED.DETACH='{detach_mode}'. Expected mediated / direct / none.")

    layer_losses = []
    for layer_idx in shared_layers:
        qskp = qskp_layers[layer_idx]
        qskv = qskv_layers[layer_idx]
        apv = apv_layers[layer_idx]
        if qskp.dim() != 3 or qskv.dim() != 3 or apv.dim() != 3:
            raise RuntimeError(
                "Semantic-mediated affinity loss expects [B,S,P], [B,S,V], [B,P,V], got {}, {}, {} at layer {}.".format(
                    tuple(qskp.shape),
                    tuple(qskv.shape),
                    tuple(apv.shape),
                    int(layer_idx),
                )
            )
        if qskp.shape[0] != qskv.shape[0] or qskp.shape[0] != apv.shape[0]:
            raise RuntimeError(f"Semantic-mediated affinity batch mismatch at layer {layer_idx}.")
        if qskp.shape[1] != qskv.shape[1]:
            raise RuntimeError(f"Semantic token count mismatch between QsKp and QsKv at layer {layer_idx}.")
        if qskp.shape[2] != apv.shape[1] or qskv.shape[2] != apv.shape[2]:
            raise RuntimeError(
                "Semantic-mediated affinity shape mismatch at layer {}: QsKp={}, QsKv={}, Apv={}.".format(
                    int(layer_idx),
                    tuple(qskp.shape),
                    tuple(qskv.shape),
                    tuple(apv.shape),
                )
            )

        mediated = torch.bmm(qskp.transpose(1, 2), qskv)
        mediated_norm = _normalize_affinity_for_aux_loss(mediated, norm_type, "SOLVER.SEM_MED")
        direct_norm = _normalize_affinity_for_aux_loss(apv, norm_type, "SOLVER.SEM_MED")

        if detach_mode == "mediated":
            loss_i = _semantic_mediated_distance(direct_norm, mediated_norm.detach(), metric, "SOLVER.SEM_MED.METRIC")
        elif detach_mode == "direct":
            loss_i = _semantic_mediated_distance(mediated_norm, direct_norm.detach(), metric, "SOLVER.SEM_MED.METRIC")
        else:
            if metric == "kl":
                loss_i = 0.5 * (
                    _semantic_mediated_distance(direct_norm, mediated_norm.detach(), metric, "SOLVER.SEM_MED.METRIC")
                    + _semantic_mediated_distance(mediated_norm, direct_norm.detach(), metric, "SOLVER.SEM_MED.METRIC")
                )
            else:
                loss_i = _semantic_mediated_distance(direct_norm, mediated_norm, metric, "SOLVER.SEM_MED.METRIC")
        layer_losses.append(loss_i)

    return torch.stack(layer_losses).mean()


def _compute_semantic_prompt_visual_cycle_loss(aux: Optional[Dict[str, Any]], cfg) -> torch.Tensor:
    """
    L_spv checks whether semantic -> prompt -> visual can reconstruct semantic -> visual.

    Compose modes:
        prob:
            P_s2v_via_p = softmax(QsKp) @ softmax(Apv)
        raw_then_norm:
            P_s2v_via_p = softmax(QsKp @ Apv)

    Shapes after trainer head-mean:
        QsKp: [B, S, P]
        QsKv: [B, S, V]
        Apv : [B, P, V]
    """
    if not isinstance(aux, Dict):
        raise RuntimeError("Semantic prompt-visual cycle loss requires affinity aux dict.")

    target = str(cfg.SOLVER.SPV.TARGET)
    target_map = {
        "QpKv": "aff_qpkv",
        "QpQv": "aff_qpqv",
        "KpKv": "aff_kpkv",
    }
    if target not in target_map:
        raise ValueError(f"Unsupported SOLVER.SPV.TARGET='{target}'. Expected QpKv / QpQv / KpKv.")

    required_keys = ["aff_qskp", "aff_qskv", target_map[target]]
    missing = [k for k in required_keys if k not in aux or not isinstance(aux[k], Dict)]
    if missing:
        raise RuntimeError(f"Semantic prompt-visual cycle loss missing aux keys: {missing}.")

    qskp_layers = aux["aff_qskp"]
    qskv_layers = aux["aff_qskv"]
    apv_layers = aux[target_map[target]]

    shared_layers = sorted(set(qskp_layers.keys()) & set(qskv_layers.keys()) & set(apv_layers.keys()))
    requested_layers = list(cfg.SOLVER.SPV.LAYERS)
    if requested_layers:
        requested_layers = [int(x) for x in requested_layers]
        shared_layers = [x for x in shared_layers if x in requested_layers]
    if not shared_layers:
        raise RuntimeError("Semantic prompt-visual cycle loss found no shared layers.")

    metric = str(cfg.SOLVER.SPV.METRIC).lower()
    norm_type = str(cfg.SOLVER.SPV.NORM).lower()
    compose_mode = str(cfg.SOLVER.SPV.COMPOSE).lower()
    detach_mode = str(cfg.SOLVER.SPV.DETACH).lower()
    if compose_mode not in {"prob", "raw_then_norm"}:
        raise ValueError(f"Unsupported SOLVER.SPV.COMPOSE='{compose_mode}'. Expected prob / raw_then_norm.")
    if detach_mode not in {"via_prompt", "direct", "none"}:
        raise ValueError(f"Unsupported SOLVER.SPV.DETACH='{detach_mode}'. Expected via_prompt / direct / none.")

    layer_losses = []
    for layer_idx in shared_layers:
        qskp = qskp_layers[layer_idx]
        qskv = qskv_layers[layer_idx]
        apv = apv_layers[layer_idx]
        if qskp.dim() != 3 or qskv.dim() != 3 or apv.dim() != 3:
            raise RuntimeError(
                "Semantic prompt-visual cycle loss expects [B,S,P], [B,S,V], [B,P,V], got {}, {}, {} at layer {}.".format(
                    tuple(qskp.shape),
                    tuple(qskv.shape),
                    tuple(apv.shape),
                    int(layer_idx),
                )
            )
        if qskp.shape[0] != qskv.shape[0] or qskp.shape[0] != apv.shape[0]:
            raise RuntimeError(f"Semantic prompt-visual cycle loss batch mismatch at layer {layer_idx}.")
        if qskp.shape[1] != qskv.shape[1]:
            raise RuntimeError(f"Semantic token count mismatch between QsKp and QsKv at layer {layer_idx}.")
        if qskp.shape[2] != apv.shape[1] or qskv.shape[2] != apv.shape[2]:
            raise RuntimeError(
                "Semantic prompt-visual cycle shape mismatch at layer {}: QsKp={}, QsKv={}, Apv={}.".format(
                    int(layer_idx),
                    tuple(qskp.shape),
                    tuple(qskv.shape),
                    tuple(apv.shape),
                )
            )

        if compose_mode == "prob":
            qskp_norm = _normalize_affinity_for_aux_loss(qskp, norm_type, "SOLVER.SPV")
            apv_norm = _normalize_affinity_for_aux_loss(apv, norm_type, "SOLVER.SPV")
            via_prompt = torch.bmm(qskp_norm, apv_norm)
        else:
            via_prompt_raw = torch.bmm(qskp, apv)
            via_prompt = _normalize_affinity_for_aux_loss(via_prompt_raw, norm_type, "SOLVER.SPV")
        direct_norm = _normalize_affinity_for_aux_loss(qskv, norm_type, "SOLVER.SPV")

        if detach_mode == "via_prompt":
            loss_i = _semantic_mediated_distance(direct_norm, via_prompt.detach(), metric, "SOLVER.SPV.METRIC")
        elif detach_mode == "direct":
            loss_i = _semantic_mediated_distance(via_prompt, direct_norm.detach(), metric, "SOLVER.SPV.METRIC")
        else:
            if metric == "kl":
                loss_i = 0.5 * (
                    _semantic_mediated_distance(direct_norm, via_prompt.detach(), metric, "SOLVER.SPV.METRIC")
                    + _semantic_mediated_distance(via_prompt, direct_norm.detach(), metric, "SOLVER.SPV.METRIC")
                )
            else:
                loss_i = _semantic_mediated_distance(via_prompt, direct_norm, metric, "SOLVER.SPV.METRIC")
        layer_losses.append(loss_i)

    return torch.stack(layer_losses).mean()


def _compute_route_teacher_student_loss(aux: Optional[Dict[str, Any]], cfg, route_side: str) -> torch.Tensor:
    """
    轻量 route teacher-student 对齐损失。

    该损失只约束当前 AFFINITY_EVOLUTION 实际使用的 teacher/student 路径：
    - prompt: direct(Apv) 与 mediated(QsKp^T @ QsKv)
    - semantic: direct(QsKv) 与 via_prompt(QsKp @ Apv)
    """
    if not isinstance(aux, Dict):
        raise RuntimeError("Route teacher-student loss requires affinity aux dict.")
    if not bool(cfg.MODEL.AFFINITY_EVOLUTION.ENABLE):
        raise RuntimeError("Route teacher-student loss requires MODEL.AFFINITY_EVOLUTION.ENABLE=True.")
    if route_side not in {"prompt", "semantic"}:
        raise ValueError(f"Unsupported route teacher-student side='{route_side}'. Expected prompt / semantic.")

    prompt_enable = route_side == "prompt"
    semantic_enable = route_side == "semantic"
    if prompt_enable and not bool(cfg.SOLVER.ROUTE_TS.PROMPT_ENABLE):
        raise ValueError("SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT > 0 requires SOLVER.ROUTE_TS.PROMPT_ENABLE=True.")
    if semantic_enable and not bool(cfg.SOLVER.ROUTE_TS.SEMANTIC_ENABLE):
        raise ValueError("SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT > 0 requires SOLVER.ROUTE_TS.SEMANTIC_ENABLE=True.")

    target_map = {
        "QpKv": "aff_qpkv",
        "QpQv": "aff_qpqv",
        "KpKv": "aff_kpkv",
    }
    prompt_target = str(cfg.MODEL.AFFINITY_EVOLUTION.PROMPT_TARGET)
    semantic_target = str(cfg.MODEL.AFFINITY_EVOLUTION.SEMANTIC_TARGET)

    prompt_detach = str(cfg.MODEL.AFFINITY_EVOLUTION.PROMPT_DETACH).lower()
    semantic_detach = str(cfg.MODEL.AFFINITY_EVOLUTION.SEMANTIC_DETACH).lower()
    compose_mode = str(cfg.MODEL.AFFINITY_EVOLUTION.SEMANTIC_COMPOSE).lower()
    if prompt_enable and prompt_detach not in {"mediated", "direct"}:
        raise ValueError("SOLVER.ROUTE_TS prompt loss requires AFFINITY_EVOLUTION.PROMPT_DETACH to be mediated or direct.")
    if semantic_enable and semantic_detach not in {"via_prompt", "direct"}:
        raise ValueError("SOLVER.ROUTE_TS semantic loss requires AFFINITY_EVOLUTION.SEMANTIC_DETACH to be via_prompt or direct.")
    if semantic_enable and compose_mode not in {"prob", "raw_then_norm"}:
        raise ValueError(f"Unsupported MODEL.AFFINITY_EVOLUTION.SEMANTIC_COMPOSE='{compose_mode}'. Expected prob / raw_then_norm.")

    required_keys = {"aff_qskp", "aff_qskv"}
    if prompt_enable:
        required_keys.add(target_map[prompt_target])
    if semantic_enable:
        required_keys.add(target_map[semantic_target])
    missing = [k for k in sorted(required_keys) if k not in aux or not isinstance(aux[k], Dict)]
    if missing:
        raise RuntimeError(f"Route teacher-student loss missing aux keys: {missing}.")

    layer_sets = [set(aux[k].keys()) for k in sorted(required_keys)]
    shared_layers = sorted(set.intersection(*layer_sets))
    requested_layers = list(cfg.SOLVER.ROUTE_TS.LAYERS)
    if requested_layers:
        requested_layers = [int(x) for x in requested_layers]
        shared_layers = [x for x in shared_layers if x in requested_layers]
    if not shared_layers:
        raise RuntimeError("Route teacher-student loss found no shared layers.")

    metric = str(cfg.SOLVER.ROUTE_TS.METRIC).lower()
    route_losses = []
    for layer_idx in shared_layers:
        qskp = aux["aff_qskp"][layer_idx]
        qskv = aux["aff_qskv"][layer_idx]
        if qskp.dim() != 3 or qskv.dim() != 3:
            raise RuntimeError(
                "Route teacher-student loss expects QsKp/QsKv shapes [B,S,P]/[B,S,V], got {}, {} at layer {}.".format(
                    tuple(qskp.shape),
                    tuple(qskv.shape),
                    int(layer_idx),
                )
            )
        if qskp.shape[0] != qskv.shape[0] or qskp.shape[1] != qskv.shape[1]:
            raise RuntimeError(f"Route teacher-student QsKp/QsKv shape mismatch at layer {layer_idx}.")

        if prompt_enable:
            apv_prompt = aux[target_map[prompt_target]][layer_idx]
            if apv_prompt.dim() != 3:
                raise RuntimeError(f"Route teacher-student prompt target must be [B,P,V] at layer {layer_idx}, got {tuple(apv_prompt.shape)}.")
            if qskp.shape[0] != apv_prompt.shape[0] or qskp.shape[2] != apv_prompt.shape[1] or qskv.shape[2] != apv_prompt.shape[2]:
                raise RuntimeError(
                    "Route teacher-student prompt shape mismatch at layer {}: QsKp={}, QsKv={}, Apv={}.".format(
                        int(layer_idx),
                        tuple(qskp.shape),
                        tuple(qskv.shape),
                        tuple(apv_prompt.shape),
                    )
                )
            mediated_prompt = _normalize_affinity_for_aux_loss(
                torch.bmm(qskp.transpose(1, 2), qskv),
                "softmax",
                "SOLVER.ROUTE_TS",
            )
            direct_prompt = _normalize_affinity_for_aux_loss(apv_prompt, "softmax", "SOLVER.ROUTE_TS")
            if prompt_detach == "mediated":
                student_prompt, teacher_prompt = direct_prompt, mediated_prompt
            else:
                student_prompt, teacher_prompt = mediated_prompt, direct_prompt
            # teacher 路径 stopgrad，只让 student 承担显式对齐梯度。
            route_losses.append(
                _semantic_mediated_distance(
                    student_prompt,
                    teacher_prompt.detach(),
                    metric,
                    "SOLVER.ROUTE_TS.METRIC",
                )
            )

        if semantic_enable:
            apv_semantic = aux[target_map[semantic_target]][layer_idx]
            if apv_semantic.dim() != 3:
                raise RuntimeError(f"Route teacher-student semantic target must be [B,P,V] at layer {layer_idx}, got {tuple(apv_semantic.shape)}.")
            if qskp.shape[0] != apv_semantic.shape[0] or qskp.shape[2] != apv_semantic.shape[1] or qskv.shape[2] != apv_semantic.shape[2]:
                raise RuntimeError(
                    "Route teacher-student semantic shape mismatch at layer {}: QsKp={}, QsKv={}, Apv={}.".format(
                        int(layer_idx),
                        tuple(qskp.shape),
                        tuple(qskv.shape),
                        tuple(apv_semantic.shape),
                    )
                )
            if compose_mode == "prob":
                via_prompt = torch.bmm(
                    _normalize_affinity_for_aux_loss(qskp, "softmax", "SOLVER.ROUTE_TS"),
                    _normalize_affinity_for_aux_loss(apv_semantic, "softmax", "SOLVER.ROUTE_TS"),
                )
            else:
                via_prompt = _normalize_affinity_for_aux_loss(
                    torch.bmm(qskp, apv_semantic),
                    "softmax",
                    "SOLVER.ROUTE_TS",
                )
            direct_semantic = _normalize_affinity_for_aux_loss(qskv, "softmax", "SOLVER.ROUTE_TS")
            if semantic_detach == "via_prompt":
                student_semantic, teacher_semantic = direct_semantic, via_prompt
            else:
                student_semantic, teacher_semantic = via_prompt, direct_semantic
            # teacher 路径 stopgrad，只让 student 承担显式对齐梯度。
            route_losses.append(
                _semantic_mediated_distance(
                    student_semantic,
                    teacher_semantic.detach(),
                    metric,
                    "SOLVER.ROUTE_TS.METRIC",
                )
            )

    return torch.stack(route_losses).mean()


class SoftmaxCMLoss(nn.Module):
    """
    当前主线使用的分类损失。

    数学形式：
        L = CE(logits, y) + lambda_cm * L_cm

    这里的设计意图是：
    - `CE(logits, y)` 负责“把样本分到正确类别”；
    - `CM` 负责“让视觉表示更贴近本样本对应类别的语义表示”。

    需要特别说明：
    - 历史上的 AM / additive margin 路径已经删除；
    - 当前这个类虽然是从旧的 `SoftmaxMarginCMLoss` 演化来的，
      但现在已经只保留 `CE + CM` 这条主线。

    因而它对分类头的要求也很明确：
    - 分类头需要输出 logits；
    - 同时还要在 forward 时缓存好 visual/semantic 的比较空间表示，
      供 CM 项直接复用。
    """
    def __init__(self, cfg=None):
        """读取 CE+CM 主损失及可选辅助正则的权重配置。"""
        super().__init__()

        self.lambda_cm = cfg.SOLVER.LOSS_CM_WEIGHT
        self.agr_res_weight = cfg.SOLVER.LOSS_AGR_RES_WEIGHT
        self.cons_weight = cfg.SOLVER.LOSS_CONS_WEIGHT
        self.anchor_cons_weight = cfg.SOLVER.LOSS_ANCHOR_CONS_WEIGHT
        self.free_kd_weight = cfg.SOLVER.LOSS_FREE_KD_WEIGHT
        self.consistency_dist = cfg.MODEL.CONSISTENCY.DIST.lower()
        self.diag_strict = cfg.SOLVER.DIAG.STRICT_CHECKS
        self.diag_print_wiring = cfg.SOLVER.DIAG.PRINT_LOSS_WIRING
        self._diag_printed = False
        self._last_loss_stats: Dict[str, float] = {}

    def is_single(self):
        """保持旧训练器接口兼容：当前 loss 返回单个标量。"""
        return True

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        这里是一次 batch 的真实损失计算入口。

        输入语义：
        - `pred_logits`
          模型前向返回的分类输出。允许是：
            1. 直接的 logits tensor
            2. (logits, aux) 形式
            3. {"logits": ..., ...} 形式
        - `targets`
          当前 active class space 下的局部标签
        - `per_cls_weights`
          类别权重，传给交叉熵
        - `kwargs`
          额外上下文，主要包括：
            - `model`
            - `raw_targets`
            - `epoch`

        整体流程：
        1. 统一抽取 logits / aux
        2. 计算交叉熵 CE
        3. 从分类头缓存里取 visual_repr / semantic_repr，计算 CM
        4. 叠加 role / consistency / AGR / free-KD 等附加项
        5. 把本次 batch 的关键损失值记录到 `_last_loss_stats`

        这里不再做 AM：
        - logits 不会再经过额外的正类 margin 扣减；
        - 交叉熵直接在原始 logits 上计算。
        """
        logits, _ = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise TypeError("SoftmaxCMLoss expects tensor logits.")

        model = kwargs.get("model", None) if isinstance(kwargs, Dict) else None
        raw_targets = kwargs.get("raw_targets", None) if isinstance(kwargs, Dict) else None
        curr_epoch = int(kwargs.get("epoch", 0)) if isinstance(kwargs, Dict) else 0
        scale = _effective_scale_from_model(model, logits)

        # CE 仍然保留按类权重的写法，这样和项目原有训练接口兼容。
        weight = torch.tensor(per_cls_weights, device=logits.device)
        ce = F.cross_entropy(logits, targets, weight, reduction="mean")

        # CM 不重新走一遍分类头，而是直接复用当前 forward 中缓存下来的比较空间表示。
        cm = _compute_cm_loss_from_rhead_cache(model=model, targets=targets, logits=logits)
        if self.diag_strict and self.lambda_cm > 0 and cm is None:
            raise RuntimeError("CM loss enabled but CM term is unavailable (cache/targets mismatch).")
        total = ce if (cm is None or self.lambda_cm <= 0) else (ce + self.lambda_cm * cm)
        self._last_loss_stats = {
            "ce_loss": float(ce.detach().item()),
        }
        if cm is not None:
            self._last_loss_stats["cm_loss"] = float(cm.detach().item())

        # AGR residual norm regularizer: keep semantic delta small.
        if self.agr_res_weight > 0 and model is not None:
            r_head = model.r_similarity_head
            delta_sem = r_head._loss_last_semantic_delta
            if torch.is_tensor(delta_sem) and delta_sem.numel() > 0:
                agr_res = (delta_sem ** 2).sum(dim=-1).mean()
                total = total + self.agr_res_weight * agr_res
                self._last_loss_stats["agr_res_loss"] = float(agr_res.detach().item())

        # AENet-style lightweight consistency on semantic increment only.
        if self.cons_weight > 0:
            cons = _compute_consistency_loss_from_rhead_cache(model=model, dist_type=self.consistency_dist)
            if cons is not None:
                total = total + self.cons_weight * cons
                self._last_loss_stats["consistency_loss"] = float(cons.detach().item())

        # Optional ablation 1: anchor-token consistency to class anchor h_y.
        if self.anchor_cons_weight > 0 and model is not None:
            r_head = model.r_similarity_head
            sem_state = r_head._runtime_semantic_state
            if isinstance(sem_state, dict):
                sem_vec = sem_state.get("sem_state")
                semantic_input = sem_state.get("semantic_input")
                if torch.is_tensor(sem_vec) and torch.is_tensor(semantic_input) and sem_vec.dim() == 2 and semantic_input.dim() == 2 and sem_vec.shape == semantic_input.shape:
                    h = F.normalize(semantic_input, dim=-1)
                    a = F.normalize(sem_vec, dim=-1)
                    anchor_cons = (1.0 - (a * h).sum(dim=-1)).mean()
                    total = total + self.anchor_cons_weight * anchor_cons
                    self._last_loss_stats["anchor_cons_loss"] = float(anchor_cons.detach().item())

        # Optional ablation 2: free-token KD to semantic increment direction.
        if self.free_kd_weight > 0 and model is not None:
            r_head = model.r_similarity_head
            sem_state = r_head._runtime_semantic_state
            if isinstance(sem_state, dict):
                sem_vec = sem_state.get("sem_state")
                delta_sem = sem_state.get("semantic_delta")
                if torch.is_tensor(sem_vec) and torch.is_tensor(delta_sem) and sem_vec.dim() == 2 and delta_sem.dim() == 2 and sem_vec.shape == delta_sem.shape:
                    t = F.normalize(delta_sem.detach(), dim=-1)
                    f = F.normalize(sem_vec, dim=-1)
                    free_kd = (1.0 - (f * t).sum(dim=-1)).mean()
                    total = total + self.free_kd_weight * free_kd
                    self._last_loss_stats["free_kd_loss"] = float(free_kd.detach().item())

        if self.diag_print_wiring and (not self._diag_printed):
            print(
                "[diag-loss] ce_targets[min,max]=({},{}) raw_targets[min,max]=({},{}) "
                "cm_enabled={} cm_available={} epoch={} scale={:.6f}".format(
                    int(targets.min().item()),
                    int(targets.max().item()),
                    int(raw_targets.min().item()) if torch.is_tensor(raw_targets) else -1,
                    int(raw_targets.max().item()) if torch.is_tensor(raw_targets) else -1,
                    bool(self.lambda_cm > 0),
                    bool(cm is not None),
                    int(curr_epoch),
                    float(scale.item()) if torch.is_tensor(scale) else float(scale),
                )
            )
            self._diag_printed = True
        return total

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """nn.Module 标准入口，转发到 loss() 执行实际计算。"""
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


class VSPCNBaselineLoss(nn.Module):
    """
    VSPCN-style baseline:
      L = CE(logits, y) + lambda_ar * mean(||cls_feat - proto_y||_2^2)
    """
    def __init__(self, cfg=None):
        """读取 VSPCN baseline 的 AR 辅助权重。"""
        super().__init__()
        self.lambda_ar = float(cfg.SOLVER.LOSS_VSPCN_AR_WEIGHT)
        self._last_loss_stats: Dict[str, float] = {}

    def is_single(self):
        """保持旧训练器接口兼容：该 loss 输出单个标量。"""
        return True

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        计算 VSPCN baseline 损失。

        组成:
        - CE(logits, y)
        - 可选 AR: CLS 特征对齐到对应类别 prototype
        """
        logits, _ = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise TypeError("VSPCNBaselineLoss expects tensor logits.")

        model = kwargs.get("model", None) if isinstance(kwargs, Dict) else None

        # Step 1. 基础交叉熵
        ce = F.cross_entropy(logits, targets, reduction="mean")
        total = ce

        self._last_loss_stats = {
            "baseline_ce_loss": float(ce.detach().item()),
        }

        # Step 2. AR loss
        if self.lambda_ar > 0 and model is not None:
            r_head = model.r_similarity_head
            cls_token = r_head._loss_last_visual_input
            proto_bank = r_head._loss_last_semantic_input
            if (
                torch.is_tensor(cls_token)
                and torch.is_tensor(proto_bank)
                and cls_token.dim() == 2
                and proto_bank.dim() == 2
                and cls_token.shape[0] == logits.shape[0]
            ):
                # targets 应该对应当前 proto_bank 的 local label
                y = targets.to(device=proto_bank.device, dtype=torch.long)
                if y.shape[0] == cls_token.shape[0] and y.min().item() >= 0 and y.max().item() < proto_bank.shape[0]:
                    # 取出每个样本真实类别对应的 prototype
                    pos_proto = proto_bank.index_select(0, y)
                    diff = cls_token - pos_proto
                    ar = diff.pow(2).sum(dim=-1).mean()
                    total = total + self.lambda_ar * ar
                    self._last_loss_stats["baseline_ar_loss"] = float(ar.detach().item())

        return total

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """nn.Module 标准入口，转发到 loss() 执行实际计算。"""
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


class SemanticMediatedAffinityAuxLoss(nn.Module):
    """
    语义中介 prompt-visual 亲和辅助损失。
    这个类只负责计算 L_sem_med，不再关心 CE / AR / CM 主损失。
    """
    def __init__(self, cfg=None):
        """读取 semantic-mediated affinity loss 的权重和配置。"""
        super().__init__()
        self.name = "sem_med_loss"
        self.requires_affinity_aux = True
        self.sem_med_weight = float(cfg.SOLVER.LOSS_SEM_MED_WEIGHT)
        self.cfg = cfg

    @property
    def weight(self) -> float:
        """返回当前辅助损失权重，供 CompositeLoss 判断是否启用。"""
        return self.sem_med_weight

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """从模型输出中取 aux，并计算 semantic-mediated affinity loss。"""
        _, aux = _extract_logits_and_aux(pred_logits, kwargs)
        return _compute_semantic_mediated_affinity_loss(aux, self.cfg)


class SemanticPromptVisualCycleAuxLoss(nn.Module):
    """
    语义-提示-视觉路径一致性辅助损失。
    只负责计算 L_spv，不参与分类主损失逻辑。
    """
    def __init__(self, cfg=None):
        """读取 semantic prompt-visual cycle loss 的权重和配置。"""
        super().__init__()
        self.name = "spv_loss"
        self.requires_affinity_aux = True
        self.spv_weight = float(cfg.SOLVER.LOSS_SPV_WEIGHT)
        self.cfg = cfg

    @property
    def weight(self) -> float:
        """返回当前辅助损失权重，供 CompositeLoss 判断是否启用。"""
        return self.spv_weight

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """从模型输出中取 aux，并计算 semantic -> prompt -> visual cycle loss。"""
        _, aux = _extract_logits_and_aux(pred_logits, kwargs)
        return _compute_semantic_prompt_visual_cycle_loss(aux, self.cfg)


class RouteTeacherStudentPromptAuxLoss(nn.Module):
    """
    prompt evolution route 的轻量 teacher-student 对齐损失。

    它复用 MODEL.AFFINITY_EVOLUTION 的 route 来源配置，
    只对 prompt route 额外提供显式对齐梯度。
    """
    def __init__(self, cfg=None):
        """读取 prompt route teacher-student loss 的权重和开关。"""
        super().__init__()
        self.name = "route_ts_prompt_loss"
        self.requires_affinity_aux = True
        self.route_ts_prompt_weight = float(cfg.SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT)
        self.cfg = cfg

    @property
    def weight(self) -> float:
        """返回当前辅助损失权重，供 CompositeLoss 判断是否启用。"""
        return self.route_ts_prompt_weight

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """从模型输出中取 aux，并计算 prompt route teacher-student loss。"""
        _, aux = _extract_logits_and_aux(pred_logits, kwargs)
        return _compute_route_teacher_student_loss(aux, self.cfg, "prompt")


class RouteTeacherStudentSemanticAuxLoss(nn.Module):
    """
    semantic evolution route 的轻量 teacher-student 对齐损失。

    它复用 MODEL.AFFINITY_EVOLUTION 的 route 来源配置，
    只对 semantic route 额外提供显式对齐梯度。
    """
    def __init__(self, cfg=None):
        """读取 semantic route teacher-student loss 的权重和开关。"""
        super().__init__()
        self.name = "route_ts_semantic_loss"
        self.requires_affinity_aux = True
        self.route_ts_semantic_weight = float(cfg.SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT)
        self.cfg = cfg

    @property
    def weight(self) -> float:
        """返回当前辅助损失权重，供 CompositeLoss 判断是否启用。"""
        return self.route_ts_semantic_weight

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """从模型输出中取 aux，并计算 semantic route teacher-student loss。"""
        _, aux = _extract_logits_and_aux(pred_logits, kwargs)
        return _compute_route_teacher_student_loss(aux, self.cfg, "semantic")


class CompositeLoss(nn.Module):
    """
    组合式 loss：main_loss 负责分类主线，aux_losses 只负责各自的辅助约束。
    后续新增辅助损失时，只需要新增 aux 类并在 build_loss 中组装，不再新增组合类。
    """
    def __init__(self, main_loss: nn.Module, aux_losses):
        """接收一个主损失和若干辅助损失，组成统一训练入口。"""
        super().__init__()
        self.main_loss = main_loss
        self.aux_losses = nn.ModuleList(aux_losses)
        self.requires_affinity_aux = any(
            bool(aux.requires_affinity_aux) and float(aux.weight) > 0
            for aux in self.aux_losses
        )
        self._last_loss_stats: Dict[str, float] = {}

    def is_single(self):
        """保持旧训练器接口兼容：当前 loss 返回单个标量。"""
        return True

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        先算主损失，再按权重叠加所有启用的辅助损失。

        同时合并各子损失的 `_last_loss_stats`，供 trainer 打印日志。
        """
        total = self.main_loss(pred_logits, targets, per_cls_weights, kwargs=kwargs)
        stats = dict(self.main_loss._last_loss_stats)
        for aux_loss in self.aux_losses:
            weight = float(aux_loss.weight)
            if weight <= 0:
                continue
            aux_value = aux_loss(pred_logits, targets, per_cls_weights, kwargs=kwargs)
            total = total + weight * aux_value
            stats[aux_loss.name] = float(aux_value.detach().item())
        self._last_loss_stats = stats
        return total


class RSimilarityLossV2(nn.Module):
    """
    RSimilarity v2 配套损失。

    设计目标：
    - 不动 baseline 现有实现；
    - 新建一条最小主干实验线；
    - 同一份头支持两种受控模式：
      1. baseline 等价模式：dot + AR
      2. 归一化 rsim 模式：cosine + CM

    当前类不强行把“score mode”和“align mode”绑定死，而是按配置决定：
    - `ALIGN_MODE = ar` 时：
        total = CE + w * AR
    - `ALIGN_MODE = cm` 时：
        total = CE + w * CM
    """

    def __init__(self, cfg=None):
        """读取 RSimilarity v2 的对齐模式和对齐权重。"""
        super().__init__()
        self.align_mode = str(cfg.SOLVER.RSIM_V2.ALIGN_MODE).lower()
        self.align_weight = float(cfg.SOLVER.RSIM_V2.ALIGN_WEIGHT)
        self.diag_strict = cfg.SOLVER.DIAG.STRICT_CHECKS
        self._last_loss_stats: Dict[str, float] = {}

    def is_single(self):
        """保持旧训练器接口兼容：该 loss 输出单个标量。"""
        return True

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        计算 RSimilarity v2 损失。

        组成:
        - CE(logits, y)
        - 可选 AR 或 CM，对齐方式由 SOLVER.RSIM_V2.ALIGN_MODE 决定
        """
        logits, _ = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise TypeError("RSimilarityLossV2 expects tensor logits.")

        model = kwargs.get("model", None) if isinstance(kwargs, Dict) else None
        weight = torch.tensor(per_cls_weights, device=logits.device)
        ce = F.cross_entropy(logits, targets, weight, reduction="mean")
        total = ce
        self._last_loss_stats = {"ce_loss": float(ce.detach().item())}

        if model is None or self.align_weight <= 0:
            return total

        r_head = model.r_similarity_head
        align_mode = self.align_mode
        align_term = None

        if align_mode == "ar":
            visual_input = r_head._loss_last_visual_input
            semantic_input = r_head._loss_last_semantic_input
            if (
                torch.is_tensor(visual_input)
                and torch.is_tensor(semantic_input)
                and visual_input.dim() == 2
                and semantic_input.dim() == 2
                and visual_input.shape[0] == logits.shape[0]
            ):
                y = targets.to(device=semantic_input.device, dtype=torch.long)
                if y.shape[0] == visual_input.shape[0] and y.min().item() >= 0 and y.max().item() < semantic_input.shape[0]:
                    pos_proto = semantic_input.index_select(0, y)
                    diff = visual_input - pos_proto
                    align_term = diff.pow(2).sum(dim=-1).mean()
                    self._last_loss_stats["ar_loss"] = float(align_term.detach().item())
        elif align_mode == "cm":
            align_term = _compute_cm_loss_from_rhead_cache(model=model, targets=targets, logits=logits)
            if align_term is not None:
                self._last_loss_stats["cm_loss"] = float(align_term.detach().item())
        else:
            raise ValueError(f"Unsupported SOLVER.RSIM_V2.ALIGN_MODE='{self.align_mode}'")

        if self.diag_strict and align_term is None:
            raise RuntimeError(f"RSimilarityLossV2 align_mode='{align_mode}' is unavailable for current batch.")

        if align_term is not None:
            total = total + self.align_weight * align_term
        return total

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """nn.Module 标准入口，转发到 loss() 执行实际计算。"""
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


def _resolve_main_loss_name(cfg) -> str:
    """
    主损失只由 SOLVER.MAIN_LOSS 决定。
    辅助损失由各自权重决定，不再通过组合 loss 名开启。
    """
    main_name = str(cfg.SOLVER.MAIN_LOSS).lower()
    if main_name == "":
        raise ValueError("SOLVER.MAIN_LOSS must be set to vspcn / rsim / rsim_v2.")
    return main_name


def _build_main_loss(cfg) -> nn.Module:
    """
    根据 SOLVER.MAIN_LOSS 构建主损失。

    只负责主损失，不处理任何辅助损失开关。
    """
    main_name = _resolve_main_loss_name(cfg)
    main_losses = {
        "vspcn": VSPCNBaselineLoss,
        "rsim": SoftmaxCMLoss,
        "rsim_v2": RSimilarityLossV2,
    }
    if main_name not in main_losses:
        raise ValueError(f"Unsupported SOLVER.MAIN_LOSS='{main_name}'. Expected vspcn / rsim / rsim_v2.")
    return main_losses[main_name](cfg)


def _sem_med_enabled(cfg) -> bool:
    """判断 semantic-mediated affinity 辅助损失是否启用。"""
    return float(cfg.SOLVER.LOSS_SEM_MED_WEIGHT) > 0


def _spv_enabled(cfg) -> bool:
    """判断 semantic prompt-visual cycle 辅助损失是否启用。"""
    return float(cfg.SOLVER.LOSS_SPV_WEIGHT) > 0


def _route_ts_prompt_enabled(cfg) -> bool:
    """判断 prompt route teacher-student 辅助损失是否启用。"""
    return float(cfg.SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT) > 0


def _route_ts_semantic_enabled(cfg) -> bool:
    """判断 semantic route teacher-student 辅助损失是否启用。"""
    return float(cfg.SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT) > 0


def _build_aux_losses(cfg):
    """
    根据各辅助损失权重构建辅助损失列表。

    当前只有 semantic-mediated affinity loss，后续新增辅助项也放这里。
    """
    aux_losses = []
    if _sem_med_enabled(cfg):
        aux_losses.append(SemanticMediatedAffinityAuxLoss(cfg))
    if _spv_enabled(cfg):
        aux_losses.append(SemanticPromptVisualCycleAuxLoss(cfg))
    if _route_ts_prompt_enabled(cfg):
        aux_losses.append(RouteTeacherStudentPromptAuxLoss(cfg))
    if _route_ts_semantic_enabled(cfg):
        aux_losses.append(RouteTeacherStudentSemanticAuxLoss(cfg))
    return aux_losses


def build_loss(cfg):
    """
    从配置构建最终训练使用的 loss。

    当前入口统一返回 CompositeLoss：
    - main_loss: CE/AR/CM 等分类主线
    - aux_losses: sem_med 等可插拔辅助约束
    """
    main_loss = _build_main_loss(cfg)
    aux_losses = _build_aux_losses(cfg)
    return CompositeLoss(main_loss, aux_losses)
