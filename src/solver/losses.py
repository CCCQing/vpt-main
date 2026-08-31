#!/usr/bin/env python3

import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..models.prompting.prompt_distribution import prompt_kl_loss
from .graph_prob_prior_monitors import loss_scale_monitor
from .semantic_graph_losses import GraphProbPriorLossComputer


@dataclass(frozen=True)
class LossTerm:
    """One graph-retaining scalar loss component exposed for sparse audits."""

    name: str
    raw_tensor: torch.Tensor
    weight: float
    role: str
    active: bool = True

    @property
    def weighted_tensor(self) -> torch.Tensor:
        return self.raw_tensor * float(self.weight)


def _capture_loss_terms(kwargs: Optional[Dict[str, Any]]) -> bool:
    return bool(
        isinstance(kwargs, Dict)
        and kwargs.get("capture_loss_terms", False)
    )


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
    if detach_mode not in {"mediated", "direct", "none"}:
        raise ValueError(f"Unsupported SOLVER.SPV.DETACH='{detach_mode}'. Expected mediated / direct / none.")

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

        if detach_mode == "mediated":
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


def _compute_attribute_reconstruction_loss(kwargs: Optional[Dict[str, Any]], cfg) -> torch.Tensor:
    """
    语义属性重建损失 L_attr。

    当前 orthogonal semantic tokenizer 会把输入属性 a_mean 编成 semantic tokens，
    这些 token 在 ViT 主序列中与 prompt / visual patch 交互后，再被固定 codebook 解码为:
        decoded_attributes = â_x, shape [B, 312]

    本损失用当前图像真实类别属性 a_y 监督 â_x:
        L_attr = MSE(â_x, a_y)

    设计边界:
    - 只读 model.get_runtime_semantic_state()["decoded_attributes"]；
    - 只支持 orthogonal tokenizer 这类能解码回属性空间的路径；
    - a_y 来自 dataloader 的 batch["attribute"]，不对目标属性反传梯度。
    """
    if not isinstance(kwargs, Dict):
        raise RuntimeError("Attribute reconstruction loss requires loss kwargs.")

    metric = str(cfg.SOLVER.ATTR.METRIC).lower()
    if metric != "mse":
        raise ValueError(f"Unsupported SOLVER.ATTR.METRIC='{metric}'. First version only supports mse.")

    if "model" not in kwargs:
        raise RuntimeError("Attribute reconstruction loss requires kwargs['model'].")
    if "target_attributes" not in kwargs:
        raise RuntimeError("Attribute reconstruction loss requires kwargs['target_attributes'].")

    model = kwargs["model"]
    target_attributes = kwargs["target_attributes"]
    if model is None:
        raise RuntimeError("Attribute reconstruction loss requires a non-None model reference.")
    if not torch.is_tensor(target_attributes):
        raise RuntimeError("Attribute reconstruction loss requires tensor target_attributes from batch['attribute'].")

    sem_state = model.get_runtime_semantic_state()
    if not isinstance(sem_state, Dict):
        raise RuntimeError("Attribute reconstruction loss requires runtime semantic state dict.")
    if "decoded_attributes" not in sem_state:
        raise RuntimeError(
            "Attribute reconstruction loss requires semantic_state['decoded_attributes']; "
            "use MODEL.SEMANTIC_TOKENS.TOKENIZER='orthogonal'."
        )

    decoded_attributes = sem_state["decoded_attributes"]
    if not torch.is_tensor(decoded_attributes):
        raise RuntimeError("semantic_state['decoded_attributes'] must be a tensor.")
    if decoded_attributes.dim() != 2 or target_attributes.dim() != 2:
        raise RuntimeError(
            "Attribute reconstruction loss expects decoded/target attributes as [B,A], got {} and {}.".format(
                tuple(decoded_attributes.shape),
                tuple(target_attributes.shape),
            )
        )
    if tuple(decoded_attributes.shape) != tuple(target_attributes.shape):
        raise RuntimeError(
            "Attribute reconstruction loss shape mismatch: decoded_attributes={} target_attributes={}.".format(
                tuple(decoded_attributes.shape),
                tuple(target_attributes.shape),
            )
        )

    target = target_attributes.to(
        device=decoded_attributes.device,
        dtype=decoded_attributes.dtype,
        non_blocking=True,
    ).detach()
    return F.mse_loss(decoded_attributes, target, reduction="mean")


class SemanticMediatedAffinityAuxLoss(nn.Module):
    """
    语义中介 prompt-visual 亲和辅助损失。
    这个类只负责计算 L_sem_med，不再关心 CE / AR / CM 主损失。
    """
    def __init__(self, cfg=None):
        """读取 semantic-mediated affinity loss 的权重和配置。"""
        super().__init__()
        self.name = "sem_med_loss"
        self.role = "transfer"
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
        self.role = "transfer"
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


class AttributeReconstructionAuxLoss(nn.Module):
    """
    ViT 交互后语义属性重建辅助损失。

    该损失不依赖 affinity aux，而是读取 runtime semantic state 中的
    decoded_attributes，把它约束到当前图像真实类别属性 a_y。
    """
    def __init__(self, cfg=None):
        """读取属性重建损失权重和度量方式。"""
        super().__init__()
        self.name = "attr_loss"
        self.role = "transfer"
        self.requires_affinity_aux = False
        self.attr_weight = float(cfg.SOLVER.LOSS_ATTR_WEIGHT)
        self.cfg = cfg

    @property
    def weight(self) -> float:
        """返回当前辅助损失权重，供 CompositeLoss 判断是否启用。"""
        return self.attr_weight

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        计算 L_attr = MSE(â_x, a_y)。

        pred_logits / targets / per_cls_weights 由 CompositeLoss 统一传入；
        这里真正使用的是 kwargs 中的 model runtime state 和 target_attributes。
        """
        return _compute_attribute_reconstruction_loss(kwargs, self.cfg)


def _compute_prompt_kl_aux_loss(kwargs: Optional[Dict[str, Any]]) -> torch.Tensor:
    """
    从模型 runtime cache 读取 prompt distribution stats 并计算 KL。

    数据路径:
        PromptedTransformer._last_prompt_distribution_stats
        -> ViT.get_runtime_prompt_distribution_stats()
        -> CompositeLoss / PromptKLAuxLoss

    这里不读取 label，只约束 q(z|x)=N(mu,std^2) 不要偏离 N(0,I) 太远。
    """
    if not isinstance(kwargs, Dict):
        raise RuntimeError("Prompt KL loss requires loss kwargs.")
    if "model" not in kwargs:
        raise RuntimeError("Prompt KL loss requires kwargs['model'].")
    model = kwargs["model"]
    if not hasattr(model, "get_runtime_prompt_distribution_stats"):
        raise RuntimeError("Prompt KL loss requires model.get_runtime_prompt_distribution_stats().")
    stats = model.get_runtime_prompt_distribution_stats()
    if not isinstance(stats, Dict):
        raise RuntimeError("Prompt KL loss requires runtime prompt distribution stats dict.")
    if "mu" not in stats or "logvar" not in stats:
        raise RuntimeError("Prompt KL loss requires stats['mu'] and stats['logvar'].")
    return prompt_kl_loss(stats["mu"], stats["logvar"], reduction="mean")


class PromptKLAuxLoss(nn.Module):
    """
    Gaussian prompt distribution 的 KL 辅助损失。

    权重由 SOLVER.LOSS_PROMPT_KL_WEIGHT 控制；权重为 0 时不会加入 CompositeLoss。
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.name = "prompt_kl_loss"
        self.role = "regularization"
        self.requires_affinity_aux = False
        self.prompt_kl_weight = float(cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT)

    @property
    def weight(self) -> float:
        return self.prompt_kl_weight

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        return _compute_prompt_kl_aux_loss(kwargs)


class B3ClassConsistencyAuxLoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.name = "b3_class_consistency_loss"
        self.role = "transfer"
        self.requires_affinity_aux = False
        self.cfg = cfg.SOLVER.B3_CLASS_CONSISTENCY
        self.intra_weight = float(self.cfg.INTRA_WEIGHT)
        self.inter_weight = float(self.cfg.INTER_WEIGHT)
        self._last_loss_stats: Dict[str, float] = {}
        self._last_loss_terms = ()
        if self.intra_weight < 0.0 or self.inter_weight < 0.0:
            raise ValueError("B3 class-consistency weights must be non-negative")
        if self.intra_weight + self.inter_weight <= 0.0:
            raise ValueError(
                "enabled B3 class consistency requires a positive loss weight"
            )
        if str(self.cfg.RELATION_MODE).lower() not in {
            "true_class",
            "sample_hash",
        }:
            raise ValueError(
                "B3 class-consistency RELATION_MODE must be true_class or sample_hash"
            )
        if int(self.cfg.SAMPLE_HASH_GROUPS) <= 1:
            raise ValueError("B3 SAMPLE_HASH_GROUPS must be greater than one")

    @property
    def weight(self) -> float:
        return 1.0

    def _group_ids(self, targets_global, sample_ids, device):
        mode = str(self.cfg.RELATION_MODE).lower()
        if mode == "true_class":
            if targets_global is None:
                raise RuntimeError(
                    "B3 true_class relation requires global training targets"
                )
            return torch.as_tensor(
                targets_global, device=device, dtype=torch.long
            ).reshape(-1)
        if sample_ids is None:
            raise RuntimeError(
                "B3 sample_hash relation control requires training sample_ids"
            )
        identifiers = [str(value) for value in list(sample_ids)]
        groups = []
        for identifier in identifiers:
            digest = hashlib.sha256(
                "{}|{}".format(
                    int(self.cfg.SAMPLE_HASH_SEED), identifier
                ).encode("utf-8")
            ).digest()
            groups.append(
                int.from_bytes(digest[:8], byteorder="little")
                % int(self.cfg.SAMPLE_HASH_GROUPS)
            )
        return torch.as_tensor(groups, device=device, dtype=torch.long)

    def forward(
        self,
        pred_logits,
        targets,
        per_cls_weights,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(kwargs, dict) or "model" not in kwargs:
            raise RuntimeError("B3 class consistency requires loss kwargs and model")
        logits, _ = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise RuntimeError("B3 class consistency requires tensor logits")
        if not bool(kwargs.get("is_train", self.training)):
            self._last_loss_stats = {"b3_class_consistency_eval_skipped": 1.0}
            return logits.new_zeros(())
        model = kwargs["model"]
        if not hasattr(model, "get_runtime_prompt_distribution_stats") or not hasattr(
            model, "compute_b3_class_consistency"
        ):
            raise RuntimeError("model does not expose B3 class-consistency state")
        stats = model.get_runtime_prompt_distribution_stats()
        if not isinstance(stats, dict) or not torch.is_tensor(stats.get("mu")):
            raise RuntimeError("B3 class consistency requires runtime mu")
        mu = stats["mu"]
        groups = self._group_ids(
            kwargs.get("targets_global"), kwargs.get("sample_ids"), mu.device
        )
        result = model.compute_b3_class_consistency(
            mu,
            groups,
            momentum=float(self.cfg.EMA_MOMENTUM),
            margin=float(self.cfg.INTER_MARGIN),
            update=True,
        )
        intra = result["intra_loss"]
        inter = result["inter_loss"]
        total = self.intra_weight * intra + self.inter_weight * inter
        if _capture_loss_terms(kwargs):
            nested_terms = []
            if self.intra_weight > 0.0:
                nested_terms.append(
                    LossTerm(
                        name="b3_intra_loss",
                        raw_tensor=intra,
                        weight=self.intra_weight,
                        role="transfer",
                        active=True,
                    )
                )
            if self.inter_weight > 0.0:
                nested_terms.append(
                    LossTerm(
                        name="b3_inter_loss",
                        raw_tensor=inter,
                        weight=self.inter_weight,
                        role="transfer",
                        active=True,
                    )
                )
            self._last_loss_terms = tuple(nested_terms)
        else:
            self._last_loss_terms = ()
        self._last_loss_stats = {
            "b3_intra_loss.raw": float(intra.detach().item()),
            "b3_intra_loss.weight": self.intra_weight,
            "b3_intra_loss.weighted": self.intra_weight
            * float(intra.detach().item()),
            "b3_inter_loss.raw": float(inter.detach().item()),
            "b3_inter_loss.weight": self.inter_weight,
            "b3_inter_loss.weighted": self.inter_weight
            * float(inter.detach().item()),
            "b3_positive_cosine": float(
                result["positive_cosine"].detach().item()
            ),
            "b3_hardest_negative_cosine": float(
                result["hardest_negative_cosine"].detach().item()
            ),
            "b3_present_group_count": float(
                result["present_group_count"].detach().item()
            ),
            "b3_initialized_group_count": float(
                result["initialized_group_count"].detach().item()
            ),
        }
        return total


class GraphProbPriorAuxLoss(nn.Module):
    """
    Graph-GP 辅助损失接入口。

    posterior 来自 prompt distributor 的完整 mu/logvar；Graph-GP 根据全部官方 Seen
    视觉中心与 external graph 推断全类 Gaussian prototype，再用 energy classification
    约束 posterior。评测阶段不更新 Seen 统计 buffer，因此明确跳过该训练期辅助损失。
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        # CompositeLoss 会用 name 作为 stats 里的键：
        # stats["graph_prob_prior_loss"] = 当前 batch 未乘权重的 aux loss 标量。
        self.name = "graph_prob_prior_loss"
        self.role = "graph"

        # GraphProbPrior 不需要 trainer 额外导出 attention affinity aux；
        # 它只依赖 prompt distributor runtime stats 和 trainer 传入的语义图资源。
        self.requires_affinity_aux = False

        # 这是 CompositeLoss 外层乘的辅助损失权重。
        # forward() 返回的是未乘 LOSS_WEIGHT 的原始 GraphProbPrior loss。
        self.graph_prob_prior_weight = float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT)

        # 标准 Prompt KL 约束的是 q(z|x) 接近 N(0,I)；
        # GraphProbPrior 使用 Graph-GP prototype 上的 energy classification。
        # 两者同时打开不是错误，但语义上是“双 prior”共同约束同一个 posterior。
        if self.graph_prob_prior_weight > 0.0 and float(cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT) > 0.0:
            print(
                "[graph-prob-prior] SOLVER.LOSS_PROMPT_KL_WEIGHT and "
                "MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT are both > 0; "
                "training will use both N(0,I) KL and Graph-GP energy classification."
            )

        # Graph-GP prior、energy loss 与监测都放在 GraphProbPriorLossComputer 中。
        self.computer = GraphProbPriorLossComputer(cfg)

        # 保存最近一次 forward 的细粒度诊断项，CompositeLoss 会把它们合并到总 stats。
        self._last_loss_stats: Dict[str, float] = {}

    @property
    def weight(self) -> float:
        # CompositeLoss 统一通过 aux_loss.weight 判断是否启用并决定外层加权。
        return self.graph_prob_prior_weight

    @staticmethod
    def _zero_loss_like(pred_logits, targets):
        if torch.is_tensor(pred_logits):
            return pred_logits.new_zeros(())
        if isinstance(pred_logits, (list, tuple)):
            for item in pred_logits:
                if torch.is_tensor(item):
                    return item.new_zeros(())
        if isinstance(pred_logits, dict):
            for item in pred_logits.values():
                if torch.is_tensor(item):
                    return item.new_zeros(())
        if torch.is_tensor(targets):
            return torch.zeros((), device=targets.device, dtype=torch.float32)
        return torch.tensor(0.0)

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        # pred_logits / targets / per_cls_weights 是 CompositeLoss 的统一接口参数。
        # GraphProbPrior 自己不直接使用分类 logits，也不使用 remap 后的 local targets；
        # 它需要的是 kwargs 中的模型运行时状态和全局类别语义资源。
        if not isinstance(kwargs, dict):
            raise RuntimeError("GraphProbPrior loss requires loss kwargs.")

        # trainer 在 forward_one_batch 中构造 loss_kwargs：
        #   model:              当前 ViT 模型引用，用于读取 runtime prompt stats；
        #   targets_global:     全局类别 id，不能用 local-output remap 后的 targets；
        #   class_attributes:   全类属性矩阵 A_conf；
        if "model" not in kwargs:
            raise RuntimeError("GraphProbPrior loss requires kwargs['model'].")
        if "targets_global" not in kwargs:
            raise RuntimeError("GraphProbPrior loss requires kwargs['targets_global'].")
        if "class_attributes" not in kwargs:
            raise RuntimeError("GraphProbPrior loss requires kwargs['class_attributes'].")

        model = kwargs["model"]
        # PromptedTransformer 在前向时会缓存 distributor 返回的 stats；
        # ViT model 通过 get_runtime_prompt_distribution_stats() 暴露给 loss 侧。
        if not hasattr(model, "get_runtime_prompt_distribution_stats"):
            raise RuntimeError("GraphProbPrior loss requires model.get_runtime_prompt_distribution_stats().")
        stats = model.get_runtime_prompt_distribution_stats()
        if not isinstance(stats, dict):
            raise RuntimeError("GraphProbPrior loss requires runtime prompt distribution stats dict.")

        is_train = bool(kwargs.get("is_train", self.training))
        if not is_train:
            self._last_loss_stats = {"graph_prob_prior_eval_skipped": 1.0}
            return self._zero_loss_like(pred_logits, targets)

        if "mu" not in stats or "logvar" not in stats:
            raise RuntimeError("Graph-GP energy classification requires stats['mu'] and stats['logvar'].")
        loss = self.computer(
            posterior_mu=stats["mu"],
            posterior_logvar=stats["logvar"],
            targets_global=kwargs["targets_global"],
            class_attributes=kwargs["class_attributes"],
            seen_class_ids=kwargs.get("seen_class_ids", None),
            unseen_class_ids=kwargs.get("unseen_class_ids", None),
            epoch=kwargs.get("epoch", None),
            is_train=is_train,
        )

        # GraphProbPriorLossComputer 内部会记录 energy、posterior/prior 和 Graph-GP 诊断项；
        # 这里复制出来给 CompositeLoss 合并。
        self._last_loss_stats = dict(self.computer._last_loss_stats)
        return loss


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
        self._last_loss_terms = ()

    def is_single(self):
        """保持旧训练器接口兼容：当前 loss 返回单个标量。"""
        return True

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        先算主损失，再按权重叠加所有启用的辅助损失。

        同时合并各子损失的 `_last_loss_stats`，供 trainer 打印日志。
        """
        capture_terms = _capture_loss_terms(kwargs)
        self._last_loss_terms = ()
        total = self.main_loss(pred_logits, targets, per_cls_weights, kwargs=kwargs)
        main_loss_value = float(total.detach().item())
        stats = dict(self.main_loss._last_loss_stats)
        loss_terms = list(
            getattr(self.main_loss, "_last_loss_terms", ())
            if capture_terms
            else ()
        )
        component_names = [
            name
            for name in ("ce_loss", "ar_loss", "cm_loss")
            if f"{name}.weighted" in stats
        ]
        for aux_loss in self.aux_losses:
            weight = float(aux_loss.weight)
            if weight <= 0:
                continue
            aux_value = aux_loss(pred_logits, targets, per_cls_weights, kwargs=kwargs)
            total = total + weight * aux_value
            raw_value = float(aux_value.detach().item())
            stats[aux_loss.name] = raw_value
            stats[f"{aux_loss.name}.raw"] = raw_value
            stats[f"{aux_loss.name}.weight"] = weight
            stats[f"{aux_loss.name}.weighted"] = weight * raw_value
            component_names.append(str(aux_loss.name))
            if capture_terms:
                nested_terms = tuple(getattr(aux_loss, "_last_loss_terms", ()))
                if nested_terms:
                    loss_terms.extend(nested_terms)
                    component_names.extend(str(term.name) for term in nested_terms)
                else:
                    loss_terms.append(
                        LossTerm(
                            name=str(aux_loss.name),
                            raw_tensor=aux_value,
                            weight=weight,
                            role=str(getattr(aux_loss, "role", "regularization")),
                            active=True,
                        )
                    )
            if hasattr(aux_loss, "_last_loss_stats"):
                stats.update(dict(aux_loss._last_loss_stats))
                component_names.extend(
                    key[:-len(".weighted")]
                    for key in aux_loss._last_loss_stats
                    if str(key).endswith(".weighted")
                )
            if isinstance(aux_loss, GraphProbPriorAuxLoss):
                stats.update(
                    loss_scale_monitor(
                        main_loss=main_loss_value,
                        graph_prob_prior_loss=float(aux_value.detach().item()),
                        loss_weight=weight,
                    )
                )
        total_value = float(total.detach().item())
        stats["total_loss"] = total_value
        denominator = max(abs(total_value), 1e-12)
        for name in dict.fromkeys(component_names):
            weighted = stats.get(f"{name}.weighted")
            if weighted is not None:
                stats[f"{name}.weighted_share"] = float(weighted) / denominator
        self._last_loss_stats = stats
        if capture_terms:
            names = [str(term.name) for term in loss_terms]
            if len(names) != len(set(names)):
                raise RuntimeError(
                    "LossTerm names must be unique within one forward: {}".format(names)
                )
            self._last_loss_terms = tuple(loss_terms)
        return total

    def get_last_loss_terms(self):
        """Return current-step loss tensors; callers must clear them promptly."""
        return tuple(self._last_loss_terms)

    def clear_last_loss_terms(self):
        """Release graph-retaining component references after sparse auditing."""
        self._last_loss_terms = ()
        if hasattr(self.main_loss, "_last_loss_terms"):
            self.main_loss._last_loss_terms = ()
        for aux_loss in self.aux_losses:
            if hasattr(aux_loss, "_last_loss_terms"):
                aux_loss._last_loss_terms = ()


class RSimilarityLoss(nn.Module):
    """
    RSimilarity 的统一损失入口。分类头的 dot/cosine 打分模式与
    none/ar/cm 对齐模式相互独立，由配置显式组合。
    """

    def __init__(self, cfg=None):
        """读取 RSimilarity 的对齐模式和对齐权重。"""
        super().__init__()
        self.align_mode = str(cfg.SOLVER.RSIM.ALIGN_MODE).lower()
        self.align_weight = float(cfg.SOLVER.RSIM.ALIGN_WEIGHT)
        if self.align_mode not in {"none", "ar", "cm"}:
            raise ValueError(
                f"Unsupported SOLVER.RSIM.ALIGN_MODE='{cfg.SOLVER.RSIM.ALIGN_MODE}'. "
                "Expected none, ar, or cm."
            )
        if self.align_weight < 0:
            raise ValueError("SOLVER.RSIM.ALIGN_WEIGHT must be non-negative.")
        self.diag_strict = cfg.SOLVER.DIAG.STRICT_CHECKS
        self._last_loss_stats: Dict[str, float] = {}
        self._last_loss_terms = ()

    def is_single(self):
        """保持旧训练器接口兼容：该 loss 输出单个标量。"""
        return True

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        计算 RSimilarity 损失。

        组成:
        - CE(logits, y)
        - 可选 AR 或 CM，对齐方式由 SOLVER.RSIM.ALIGN_MODE 决定
        """
        logits, _ = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise TypeError("RSimilarityLoss expects tensor logits.")

        model = kwargs.get("model", None) if isinstance(kwargs, Dict) else None
        weight = torch.tensor(per_cls_weights, device=logits.device)
        ce = F.cross_entropy(logits, targets, weight, reduction="mean")
        total = ce
        capture_terms = _capture_loss_terms(kwargs)
        self._last_loss_terms = (
            LossTerm(
                name="ce_loss",
                raw_tensor=ce,
                weight=1.0,
                role="primary",
                active=True,
            ),
        ) if capture_terms else ()
        ce_value = float(ce.detach().item())
        self._last_loss_stats = {
            "ce_loss": ce_value,
            "ce_loss.raw": ce_value,
            "ce_loss.weight": 1.0,
            "ce_loss.weighted": ce_value,
        }

        if self.align_mode == "none" or model is None or self.align_weight <= 0:
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
                    raw_align = float(align_term.detach().item())
                    self._last_loss_stats.update({
                        "ar_loss": raw_align,
                        "ar_loss.raw": raw_align,
                        "ar_loss.weight": self.align_weight,
                        "ar_loss.weighted": self.align_weight * raw_align,
                    })
        elif align_mode == "cm":
            align_term = _compute_cm_loss_from_rhead_cache(model=model, targets=targets, logits=logits)
            if align_term is not None:
                raw_align = float(align_term.detach().item())
                self._last_loss_stats.update({
                    "cm_loss": raw_align,
                    "cm_loss.raw": raw_align,
                    "cm_loss.weight": self.align_weight,
                    "cm_loss.weighted": self.align_weight * raw_align,
                })
        else:
            raise ValueError(f"Unsupported SOLVER.RSIM.ALIGN_MODE='{self.align_mode}'")

        if self.diag_strict and align_term is None:
            raise RuntimeError(f"RSimilarityLoss align_mode='{align_mode}' is unavailable for current batch.")

        if align_term is not None:
            total = total + self.align_weight * align_term
            if capture_terms:
                align_name = "ar_loss" if align_mode == "ar" else "cm_loss"
                self._last_loss_terms = self._last_loss_terms + (
                    LossTerm(
                        name=align_name,
                        raw_tensor=align_term,
                        weight=self.align_weight,
                        role="transfer",
                        active=True,
                    ),
                )
        return total

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """nn.Module 标准入口，转发到 loss() 执行实际计算。"""
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


def _sem_med_enabled(cfg) -> bool:
    """判断 semantic-mediated affinity 辅助损失是否启用。"""
    return float(cfg.SOLVER.LOSS_SEM_MED_WEIGHT) > 0


def _spv_enabled(cfg) -> bool:
    """判断 semantic prompt-visual cycle 辅助损失是否启用。"""
    return float(cfg.SOLVER.LOSS_SPV_WEIGHT) > 0


def _attr_reconstruction_enabled(cfg) -> bool:
    """判断 ViT 交互后语义属性重建损失是否启用。"""
    return float(cfg.SOLVER.LOSS_ATTR_WEIGHT) > 0


def _prompt_kl_enabled(cfg) -> bool:
    """判断 prompt distribution KL 是否启用。"""
    return float(cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT) > 0


def _graph_prob_prior_enabled(cfg) -> bool:
    """判断 GraphProbPrior 辅助损失是否启用。"""
    return (
        bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE)
        and float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT) > 0
    )


def _b3_class_consistency_enabled(cfg) -> bool:
    return bool(cfg.SOLVER.B3_CLASS_CONSISTENCY.ENABLE)


def _build_aux_losses(cfg):
    """
    根据各辅助损失权重构建辅助损失列表。

    affinity 类辅助损失会要求 trainer 导出逐层 affinity aux；
    属性重建损失只读取 runtime semantic state，不额外要求 affinity aux。
    """
    aux_losses = []
    if _sem_med_enabled(cfg):
        aux_losses.append(SemanticMediatedAffinityAuxLoss(cfg))
    if _spv_enabled(cfg):
        aux_losses.append(SemanticPromptVisualCycleAuxLoss(cfg))
    if _attr_reconstruction_enabled(cfg):
        aux_losses.append(AttributeReconstructionAuxLoss(cfg))
    if _prompt_kl_enabled(cfg):
        aux_losses.append(PromptKLAuxLoss(cfg))
    if _graph_prob_prior_enabled(cfg):
        aux_losses.append(GraphProbPriorAuxLoss(cfg))
    if _b3_class_consistency_enabled(cfg):
        aux_losses.append(B3ClassConsistencyAuxLoss(cfg))
    return aux_losses


def build_loss(cfg):
    """
    从配置构建最终训练使用的 loss。

    当前入口统一返回 CompositeLoss：
    - main_loss: CE/AR/CM 等分类主线
    - aux_losses: sem_med 等可插拔辅助约束
    """
    main_loss = RSimilarityLoss(cfg)
    aux_losses = _build_aux_losses(cfg)
    return CompositeLoss(main_loss, aux_losses)
