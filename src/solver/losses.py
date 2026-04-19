#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, Tuple


# ===========================
# 一些小工具函数
# ===========================
def norm1(u: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:

    denom = u.sum(dim=dim, keepdim=True) + eps
    return u / denom

def l_avg(attn_pv: torch.Tensor, attn_vs: torch.Tensor, eps: float = 1e-6, reduction: str = "mean",) -> torch.Tensor:

    pi = norm1(attn_vs.sum(dim=1), dim=-1, eps=eps)  # sum over patches

    r = torch.matmul(attn_vs, pi.unsqueeze(-1)).squeeze(-1)
    r = norm1(r, dim=-1, eps=eps)

    a_bar = attn_pv.mean(dim=1)

    per_sample = ((a_bar - r) ** 2).sum(dim=-1)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    # default: mean
    return per_sample.mean()


def l_avg_multi(attn_pv_dict: Dict[int, torch.Tensor], attn_vs_dict: Dict[int, torch.Tensor], layers: Any, eps: float = 1e-6, reduction: str = "mean",) -> torch.Tensor:

    losses = []
    for l in layers:
        losses.append(l_avg(attn_pv_dict[l], attn_vs_dict[l], eps=eps, reduction="none"))
    # losses: [num_layers, B] 鈫?鍦ㄥ眰涓婂钩鍧?鈫?[B]
    stacked = torch.stack(losses, dim=0).mean(dim=0)
    if reduction == "none":
        return stacked
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


def _extract_logits_and_aux(pred_logits: Any, kwargs: Optional[Dict[str, Any]]):
    """
    Unified extraction of:
      - logits: Tensor [B, C]
      - aux: optional dict containing attention tensors for align loss.
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
    Resolve effective classification scale used by r-similarity head.
    Fallback to 1.0 if unavailable.
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


def _apply_additive_margin(logits: torch.Tensor, targets: torch.Tensor, margin: float, scale: torch.Tensor) -> torch.Tensor:
    """
    Additive-margin in logit space:
      z_y = z_y - scale * m
    """
    if margin <= 0:
        return logits
    if logits.dim() != 2:
        return logits
    z = logits.clone()
    idx = torch.arange(z.shape[0], device=z.device)
    z[idx, targets.long()] = z[idx, targets.long()] - (scale * float(margin))
    return z


def _compute_cm_loss_from_rhead_cache(
    model: Optional[nn.Module],
    targets: Optional[torch.Tensor],
    logits: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    L_cm = mean || normalize(v_i) - normalize(s_{y_i}) ||_2^2
    using the same visual/semantic embeddings as r_similarity classification.
    """
    if model is None or targets is None:
        return None
    r_head = model.r_similarity_head
    visual = r_head._loss_last_cls_visual
    semantic_bank = r_head._loss_last_semantic
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


def _compute_role_migration_terms(
    aux: Optional[Dict[str, Any]],
    early_end: int,
    late_start: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Lightweight affinity-driven role losses from layer-wise aux:
      - early: encourage A_pv energy > A_ps energy
      - late:  encourage (A_ps + A_vs) > A_pv
      - avs_entropy: encourage sharper late A_vs
    """
    out: Dict[str, Optional[torch.Tensor]] = {
        "role_early": None,
        "role_late": None,
        "avs_entropy": None,
    }
    if aux is None or (not isinstance(aux, Dict)):
        return out
    attn_pv = aux.get("attn_pv")
    attn_vs = aux.get("attn_vs")
    attn_ps = aux.get("attn_ps")
    if not isinstance(attn_pv, Dict) or not isinstance(attn_vs, Dict):
        return out

    layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()))
    if len(layers) == 0:
        return out
    if isinstance(attn_ps, Dict):
        layers_ps = set(attn_ps.keys())
    else:
        layers_ps = set()

    early_layers = [l for l in layers if int(l) <= int(early_end)]
    late_layers = [l for l in layers if int(l) >= int(late_start)]
    if len(late_layers) == 0:
        late_layers = [layers[-1]]

    # Energy proxy: mean absolute attention value.
    def _energy(x: torch.Tensor) -> torch.Tensor:
        return x.float().abs().mean()

    if early_layers and len(layers_ps) > 0:
        e_pv = torch.stack([_energy(attn_pv[l]) for l in early_layers]).mean()
        e_ps = torch.stack([_energy(attn_ps[l]) for l in early_layers if l in attn_ps]).mean()
        out["role_early"] = F.relu(e_ps - e_pv)

    if late_layers and len(layers_ps) > 0:
        l_pv = torch.stack([_energy(attn_pv[l]) for l in late_layers]).mean()
        l_ps = torch.stack([_energy(attn_ps[l]) for l in late_layers if l in attn_ps]).mean()
        l_vs = torch.stack([_energy(attn_vs[l]) for l in late_layers]).mean()
        out["role_late"] = F.relu(l_pv - 0.5 * (l_ps + l_vs))

    # Late A_vs entropy (lower is sharper)
    if late_layers:
        ent = []
        for l in late_layers:
            x = attn_vs[l].float().clamp_min(1e-8)
            ent.append((-(x * x.log()).sum(dim=-1)).mean())
        out["avs_entropy"] = torch.stack(ent).mean()
    return out


def _compute_consistency_loss_from_rhead_cache(
    model: Optional[nn.Module],
    dist_type: str = "cosine",
) -> Optional[torch.Tensor]:
    if model is None:
        return None
    r_head = model.r_similarity_head
    mu_s_final = r_head._loss_last_mu_s_final
    h_y = r_head._loss_last_h_y
    if torch.is_tensor(mu_s_final) and torch.is_tensor(h_y) and mu_s_final.shape == h_y.shape:
        delta_sem = mu_s_final - h_y
    else:
        delta_sem = r_head._loss_last_delta_sem
    target = r_head._loss_last_cons_target
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


# ===========================
# 1. 损失名到类的映射
# ===========================
LOSS = {
    "softmax_margin_cm": None,
    "vspcn_baseline": None,
}


class SoftmaxMarginCMLoss(nn.Module):
    """
    L = L_cls_am + lambda_cm * L_cm
    """
    def __init__(self, cfg=None):
        super().__init__()

        self.margin = cfg.SOLVER.LOSS_MARGIN
        self.lambda_cm = cfg.SOLVER.LOSS_CM_WEIGHT
        self.hn_margin_enable = cfg.SOLVER.LOSS_HN_MARGIN_ENABLE
        self.hn_margin_weight = cfg.SOLVER.LOSS_HN_MARGIN_WEIGHT
        self.hn_margin_value = cfg.SOLVER.LOSS_HN_MARGIN_VALUE
        self.hn_margin_start_epoch = cfg.SOLVER.LOSS_HN_MARGIN_START_EPOCH
        self.hn_detach_neg = cfg.SOLVER.LOSS_HN_DETACH_NEG
        self.role_early_weight = cfg.SOLVER.LOSS_ROLE_EARLY_WEIGHT
        self.role_late_weight = cfg.SOLVER.LOSS_ROLE_LATE_WEIGHT
        self.agr_res_weight = cfg.SOLVER.LOSS_AGR_RES_WEIGHT
        self.avs_ent_weight = cfg.SOLVER.LOSS_AVS_ENT_WEIGHT
        self.cons_weight = cfg.SOLVER.LOSS_CONS_WEIGHT
        self.anchor_cons_weight = cfg.SOLVER.LOSS_ANCHOR_CONS_WEIGHT
        self.free_kd_weight = cfg.SOLVER.LOSS_FREE_KD_WEIGHT
        self.role_early_end = cfg.MODEL.ROLE_MIGRATION.EARLY_END
        self.role_late_start = cfg.MODEL.ROLE_MIGRATION.LATE_START
        self.consistency_dist = cfg.MODEL.CONSISTENCY.DIST.lower()
        self.diag_strict = cfg.SOLVER.DIAG.STRICT_CHECKS
        self.diag_print_wiring = cfg.SOLVER.DIAG.PRINT_LOSS_WIRING
        self._diag_printed = False
        self._last_hn_stats: Dict[str, float] = {}

    def is_single(self):
        return True

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        logits, aux = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise TypeError("SoftmaxMarginCMLoss expects tensor logits.")

        model = kwargs.get("model", None) if isinstance(kwargs, Dict) else None
        raw_targets = kwargs.get("raw_targets", None) if isinstance(kwargs, Dict) else None
        curr_epoch = int(kwargs.get("epoch", 0)) if isinstance(kwargs, Dict) else 0
        scale = _effective_scale_from_model(model, logits)
        logits_am = _apply_additive_margin(logits, targets, self.margin, scale)

        weight = torch.tensor(per_cls_weights, device=logits_am.device)
        ce = F.cross_entropy(logits_am, targets, weight, reduction="mean")
        cm = _compute_cm_loss_from_rhead_cache(model=model, targets=targets, logits=logits_am)
        if self.diag_strict and self.lambda_cm > 0 and cm is None:
            raise RuntimeError("CM loss enabled but CM term is unavailable (cache/targets mismatch).")
        total = ce if (cm is None or self.lambda_cm <= 0) else (ce + self.lambda_cm * cm)

        # HNMargin in the same seen-candidate subspace as the actually used logits/targets.
        # logits are scaled similarity; recover similarity scores by dividing scale.
        hn_term = None
        if self.hn_margin_enable and self.hn_margin_weight > 0 and curr_epoch >= self.hn_margin_start_epoch:
            scale_safe = scale.clamp_min(1e-12) if torch.is_tensor(scale) else max(float(scale), 1e-12)
            score = logits / scale_safe
            idx = torch.arange(score.shape[0], device=score.device)
            pos_score = score[idx, targets.long()]
            neg_score_mat = score.detach().clone() if self.hn_detach_neg else score.clone()
            neg_score_mat[idx, targets.long()] = -1e9
            hn_score, _ = neg_score_mat.max(dim=1)
            train_margin = pos_score - hn_score
            hn_term = F.relu(float(self.hn_margin_value) - train_margin).mean()
            total = total + self.hn_margin_weight * hn_term
            with torch.no_grad():
                self._last_hn_stats = {
                    "hn_margin_loss": float(hn_term.item()),
                    "pos_score_mean": float(pos_score.mean().item()),
                    "hn_score_mean": float(hn_score.mean().item()),
                    "train_margin_mean": float(train_margin.mean().item()),
                    "p_train_margin_lt_0": float((train_margin < 0).float().mean().item()),
                    "p_train_margin_lt_neg1": float((train_margin < -1).float().mean().item()),
                    "hn_detach_neg": bool(self.hn_detach_neg),
                }
        else:
            self._last_hn_stats = {}

        role_terms = _compute_role_migration_terms(
            aux=aux,
            early_end=self.role_early_end,
            late_start=self.role_late_start,
        )
        if self.role_early_weight > 0 and role_terms["role_early"] is not None:
            total = total + self.role_early_weight * role_terms["role_early"]
        if self.role_late_weight > 0 and role_terms["role_late"] is not None:
            total = total + self.role_late_weight * role_terms["role_late"]
        if self.avs_ent_weight > 0 and role_terms["avs_entropy"] is not None:
            total = total + self.avs_ent_weight * role_terms["avs_entropy"]

        # AGR residual norm regularizer: keep semantic delta small.
        if self.agr_res_weight > 0 and model is not None:
            r_head = model.r_similarity_head
            delta_sem = r_head._loss_last_delta_sem
            if torch.is_tensor(delta_sem) and delta_sem.numel() > 0:
                agr_res = (delta_sem ** 2).sum(dim=-1).mean()
                total = total + self.agr_res_weight * agr_res
                self._last_hn_stats["agr_res_loss"] = float(agr_res.detach().item())

        # AENet-style lightweight consistency on semantic increment only.
        if self.cons_weight > 0:
            cons = _compute_consistency_loss_from_rhead_cache(model=model, dist_type=self.consistency_dist)
            if cons is not None:
                total = total + self.cons_weight * cons
                self._last_hn_stats["consistency_loss"] = float(cons.detach().item())

        # Optional ablation 1: anchor-token consistency to class anchor h_y.
        if self.anchor_cons_weight > 0 and model is not None:
            r_head = model.r_similarity_head
            sem_state = r_head._runtime_semantic_state
            if isinstance(sem_state, dict):
                a_tok = sem_state.get("anchor_tokens")
                h_y = sem_state.get("h_y")
                if torch.is_tensor(a_tok) and torch.is_tensor(h_y) and a_tok.dim() == 3 and h_y.dim() == 2 and a_tok.shape[0] == h_y.shape[0]:
                    h = F.normalize(h_y, dim=-1).unsqueeze(1).expand_as(a_tok)
                    a = F.normalize(a_tok, dim=-1)
                    anchor_cons = (1.0 - (a * h).sum(dim=-1)).mean()
                    total = total + self.anchor_cons_weight * anchor_cons
                    self._last_hn_stats["anchor_cons_loss"] = float(anchor_cons.detach().item())

        # Optional ablation 2: free-token KD to semantic increment direction.
        if self.free_kd_weight > 0 and model is not None:
            r_head = model.r_similarity_head
            sem_state = r_head._runtime_semantic_state
            if isinstance(sem_state, dict):
                f_tok = sem_state.get("free_tokens")
                delta_sem = sem_state.get("delta_sem")
                if torch.is_tensor(f_tok) and torch.is_tensor(delta_sem) and f_tok.dim() == 3 and delta_sem.dim() == 2 and f_tok.shape[0] == delta_sem.shape[0] and f_tok.shape[1] > 0:
                    t = F.normalize(delta_sem.detach(), dim=-1).unsqueeze(1).expand_as(f_tok)
                    f = F.normalize(f_tok, dim=-1)
                    free_kd = (1.0 - (f * t).sum(dim=-1)).mean()
                    total = total + self.free_kd_weight * free_kd
                    self._last_hn_stats["free_kd_loss"] = float(free_kd.detach().item())

        if role_terms["role_early"] is not None:
            self._last_hn_stats["role_early_loss"] = float(role_terms["role_early"].detach().item())
        if role_terms["role_late"] is not None:
            self._last_hn_stats["role_late_loss"] = float(role_terms["role_late"].detach().item())
        if role_terms["avs_entropy"] is not None:
            self._last_hn_stats["avs_entropy"] = float(role_terms["avs_entropy"].detach().item())

        if self.diag_print_wiring and (not self._diag_printed):
            print(
                "[diag-loss] ce_targets[min,max]=({},{}) raw_targets[min,max]=({},{}) "
                "cm_enabled={} cm_available={} "
                "hn_enabled={} hn_start={} hn_detach_neg={} epoch={} scale={:.6f}".format(
                    int(targets.min().item()),
                    int(targets.max().item()),
                    int(raw_targets.min().item()) if torch.is_tensor(raw_targets) else -1,
                    int(raw_targets.max().item()) if torch.is_tensor(raw_targets) else -1,
                    bool(self.lambda_cm > 0),
                    bool(cm is not None),
                    bool(self.hn_margin_enable and self.hn_margin_weight > 0),
                    int(self.hn_margin_start_epoch),
                    bool(self.hn_detach_neg),
                    int(curr_epoch),
                    float(scale.item()) if torch.is_tensor(scale) else float(scale),
                )
            )
            self._diag_printed = True
        return total

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


LOSS["softmax_margin_cm"] = SoftmaxMarginCMLoss


class VSPCNBaselineLoss(nn.Module):
    """
    VSPCN-style baseline:
      L = CE(logits, y) + lambda_ar * mean(||cls_feat - proto_y||_2)
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.lambda_ar = float(cfg.SOLVER.LOSS_VSPCN_AR_WEIGHT)
        self._last_hn_stats: Dict[str, float] = {}

    def is_single(self):
        return True

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        logits, _ = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise TypeError("VSPCNBaselineLoss expects tensor logits.")

        model = kwargs.get("model", None) if isinstance(kwargs, Dict) else None

        # Step 1. 基础交叉熵
        ce = F.cross_entropy(logits, targets, reduction="mean")
        total = ce

        self._last_hn_stats = {
            "baseline_ce_loss": float(ce.detach().item()),
        }

        # Step 2. AR loss
        if self.lambda_ar > 0 and model is not None:
            r_head = model.r_similarity_head
            cls_token = r_head._loss_last_cls_token
            proto_bank = r_head._loss_last_projected_prototypes
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
                    # AR loss：
                    # 每个样本的图像特征与其正类 prototype 的 L2 距离
                    # 再对 batch 求平均
                    ar = torch.norm(cls_token - pos_proto, p=2, dim=-1).mean()
                    total = total + self.lambda_ar * ar
                    self._last_hn_stats["baseline_ar_loss"] = float(ar.detach().item())

        return total

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


LOSS["vspcn_baseline"] = VSPCNBaselineLoss

# ===========================
# 4. 基于共享语义原型的相似度分类头
# ===========================
class RSimilarityClassifier(nn.Module):
    """
    Use semantic prototypes (raw/refined/fused) for visual-semantic similarity classification.
    """

    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg,) -> None:
        super().__init__()
        proj_dim = cfg.MODEL.R_SIMILARITY.PROJ_DIM
        if proj_dim is None or proj_dim <= 0:
            proj_dim = hidden_size

        # 保存全局 class attribute bank
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]
        self.attr_dim = class_attr.shape[-1]
        self.hidden_size = hidden_size

        self.visual_proj_enabled = cfg.MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE
        out_dim = proj_dim if self.visual_proj_enabled else hidden_size
        self.visual_proj = nn.Linear(hidden_size, out_dim) if self.visual_proj_enabled else None
        self.semantic_proj = nn.Linear(hidden_size, out_dim)
        # Stable semantic anchor: h_c = Linear(s_raw_c)
        self.semantic_anchor = nn.Linear(self.attr_dim, hidden_size)

        # True  -> cosine similarity * scale
        # False -> 直接点积
        self.use_cosine = cfg.MODEL.R_SIMILARITY.USE_COSINE

        self.fixed_logit_scale = cfg.MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE
        self.shuffle_prototypes = cfg.SOLVER.DIAG.SHUFFLE_PROTOTYPES

        self.consistency_enable = cfg.MODEL.CONSISTENCY.ENABLE
        self.consistency_proj = cfg.MODEL.CONSISTENCY.PROJ.lower()
        self.consistency_dist = cfg.MODEL.CONSISTENCY.DIST.lower()

        if self.consistency_enable:
            if self.consistency_proj == "mlp":
                self.consistency_head = nn.Sequential(
                    nn.Linear(self.attr_dim, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_size, out_dim),
                )
            else:
                self.consistency_head = nn.Linear(self.attr_dim, out_dim)
        else:
            self.consistency_head = None

        # debug 开关
        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self.debug_shapes = cfg.SOLVER.DEBUG_SHAPES

        self._debug_sem_source_logged = False
        self._shape_debug_logged = False

        # loss 用缓存
        self._loss_last_cls_visual = None     # 当前 batch 视觉表示（已进入分类空间）
        self._loss_last_semantic = None       # 当前 batch 使用的类语义表示（已进入分类空间）
        self._loss_last_scale = None          # 当前 batch 实际使用的 logit scale
        self._loss_last_delta_sem = None      # 当前 batch 的语义增量 delta_sem
        self._loss_last_cons_target = None    # 当前 batch consistency target
        self._loss_last_mu_s_final = None     # 当前 batch 最终 refined semantic
        self._loss_last_h_y = None            # 当前 batch 对应类别的基础 semantic anchor
        self._last_bad_feat_rows = None
        self._last_bad_visual_rows = None
        self._last_bad_sim_rows = None
        self._last_semantic_all_finite = None

        self._runtime_targets = None          # 当前 batch 的 raw global targets
        self._runtime_token_sequence = None   # 当前 batch 的 token 序列（供 debug/vis）
        self._runtime_affinities = None       # 当前 batch 的 affinity（供 debug/vis）
        self._runtime_semantic_state = None   # 当前 batch 的 semantic state（供 consistency / AGR 等）

        if self.use_cosine:
            logit_scale_init = cfg.MODEL.R_SIMILARITY.LOGIT_SCALE_INIT
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(logit_scale_init, dtype=torch.float32)))
        else:
            self.logit_scale = None

    def _class_prototypes_raw(self) -> torch.Tensor:
        """
        构造“全局原始类别语义原型”。
        输入来源：- self.class_attr : [num_classes, attr_dim]
        处理：- 通过 semantic_anchor 映射到 hidden_size
        输出：- [num_classes, hidden_size]
        直观理解：
        - 每个类别 c 都有一份 class-level 原始语义 s_raw_c
        - 通过 semantic_anchor 得到 h_c
        - h_c 就是这个类别的“基础语义原型”

        注意：
        - 这是全局所有类的语义原型
        - 还没根据 class_ids 切 active class space
        - 也还没经过 semantic_proj 进入最终比较空间
        """
        attr = self.class_attr
        return self.semantic_anchor(attr)

    def _resolve_active_class_space(self, class_ids, device: torch.device):
        if class_ids is None:
            active_ids = torch.arange(self.num_classes, device=device, dtype=torch.long)
        elif torch.is_tensor(class_ids):
            active_ids = class_ids.to(device=device, dtype=torch.long).view(-1)
        else:
            active_ids = torch.as_tensor(list(class_ids), device=device, dtype=torch.long).view(-1)
        if active_ids.numel() == 0:
            raise ValueError("Active class space is empty.")
        mapping = torch.full((self.num_classes,), -1, device=device, dtype=torch.long)
        mapping[active_ids] = torch.arange(active_ids.numel(), device=device, dtype=torch.long)
        return active_ids, mapping

    def forward(self, cls_feat: torch.Tensor, class_ids=None) -> torch.Tensor:
        """
        输入：
        - cls_feat:
            [B, hidden_size]
            当前 batch 的视觉特征，通常来自 backbone 的 CLS token
        - class_ids:
            当前活动类空间的 global class ids
            若为 None，则默认所有类都参与分类

        输出：
        - logits:
            [B, num_active_classes]
            当前 batch 在当前活动类空间上的分类分数

        整体流程：
        ------------------------------------------------------------
        Step 1. 解析当前 active class space
        Step 2. 从全局语义原型中切出当前 active classes
        Step 3. 可选打乱 prototype 顺序（诊断用）
        Step 4. visual / semantic 各自映射到比较空间
        Step 5. 做 cosine similarity 或 dot product
        Step 6. 乘上温度 scale 得到 logits
        Step 7. 缓存中间量供 loss / trainer 使用
        Step 8. 若 runtime semantic state 存在，则缓存 refined semantic 相关信息
        ------------------------------------------------------------
        """
        active_class_ids, active_global_to_local = self._resolve_active_class_space(class_ids, device=cls_feat.device)
        proto_raw_full = self._class_prototypes_raw()
        proto_raw = proto_raw_full.index_select(0, active_class_ids)

        if self.shuffle_prototypes and proto_raw.shape[0] > 1:
            perm = torch.randperm(proto_raw.shape[0], device=proto_raw.device)
            proto_raw = proto_raw.index_select(0, perm)

        cls_visual = self.visual_proj(cls_feat) if self.visual_proj else cls_feat
        cls_semantic = self.semantic_proj(proto_raw)

        if self.debug_shapes and (not self._shape_debug_logged):
            print(
                "[SHAPE-DEBUG] RSimilarityClassifier.forward cls_feature={} raw_semantic_prototype={} "
                "visual_proj={} semantic_proj={} logits={} active_classes={}".format(
                    tuple(cls_feat.shape) if torch.is_tensor(cls_feat) else None,
                    tuple(proto_raw.shape) if torch.is_tensor(proto_raw) else None,
                    tuple(cls_visual.shape) if torch.is_tensor(cls_visual) else None,
                    tuple(cls_semantic.shape) if torch.is_tensor(cls_semantic) else None,
                    (int(cls_visual.shape[0]), int(cls_semantic.shape[0])) if (torch.is_tensor(cls_visual) and torch.is_tensor(cls_semantic)) else None,
                    int(active_class_ids.numel()),
                )
            )
            self._shape_debug_logged = True

        if self.use_cosine:
            cls_visual = F.normalize(cls_visual, dim=-1)
            cls_semantic = F.normalize(cls_semantic, dim=-1)
            raw_sim = cls_visual @ cls_semantic.t()
            scale = raw_sim.new_tensor(self.fixed_logit_scale) if self.fixed_logit_scale > 0 else self.logit_scale.exp()
            logits = raw_sim * scale
        else:
            logits = cls_visual @ cls_semantic.t()
            raw_sim = logits
            scale = logits.new_tensor(1.0)

        row_feat_ok = torch.isfinite(cls_feat).all(dim=1)
        row_visual_ok = torch.isfinite(cls_visual).all(dim=1)
        sem_ok = torch.isfinite(cls_semantic).all()
        row_sim_ok = torch.isfinite(raw_sim).all(dim=1)

        self._last_bad_feat_rows = None
        self._last_bad_visual_rows = None
        self._last_bad_sim_rows = None
        self._last_semantic_all_finite = bool(sem_ok.item())
        if not bool(row_feat_ok.all().item()):
            self._last_bad_feat_rows = (~row_feat_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        if not bool(row_visual_ok.all().item()):
            self._last_bad_visual_rows = (~row_visual_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        if not bool(row_sim_ok.all().item()):
            self._last_bad_sim_rows = (~row_sim_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()

        runtime_targets_global = self._runtime_targets if torch.is_tensor(self._runtime_targets) else None

        if self.debug_trace_once and (not self._debug_sem_source_logged):
            scale_scalar = float(scale.item()) if torch.is_tensor(scale) else float(scale)
            fixed_mode = bool(self.fixed_logit_scale > 0)
            print(
                "[trace] node=C.semantic_scoring semantic_score_mode={} classifier_semantic_source={} "
                "raw_semantic_shape={} classifier_semantic_shape={} raw_semantic_norm_mean={:.6f} "
                "classifier_semantic_norm_mean={:.6f} effective_logit_scale={:.6f} whether_fixed_logit_scale={}".format(
                    "global_raw",
                    "raw",
                    tuple(proto_raw.shape),
                    tuple(proto_raw.shape),
                    float(proto_raw.float().norm(dim=-1).mean().item()),
                    float(proto_raw.float().norm(dim=-1).mean().item()),
                    scale_scalar,
                    fixed_mode,
                )
            )
        self._debug_sem_source_logged = True

        self._loss_last_cls_visual = cls_visual
        self._loss_last_semantic = cls_semantic
        self._loss_last_scale = scale
        self._loss_last_cons_target = None
        self._loss_last_delta_sem = None
        self._loss_last_mu_s_final = None
        self._loss_last_h_y = None

        sem_state = self._runtime_semantic_state if isinstance(self._runtime_semantic_state, dict) else None
        if sem_state is not None:
            mu_s_final = sem_state.get("mu_s_final")
            h_y = sem_state.get("h_y")
            delta_sem = sem_state.get("delta_sem")
            if torch.is_tensor(mu_s_final) and torch.is_tensor(h_y):
                if (delta_sem is None) or (not torch.is_tensor(delta_sem)):
                    delta_sem = mu_s_final - h_y
                self._loss_last_mu_s_final = mu_s_final
                self._loss_last_h_y = h_y
                self._loss_last_delta_sem = delta_sem
                if self.consistency_head is not None and runtime_targets_global is not None and runtime_targets_global.numel() == delta_sem.shape[0]:
                    t = runtime_targets_global.to(delta_sem.device)
                    valid = active_global_to_local.to(delta_sem.device).index_select(0, t) >= 0
                    if valid.any():
                        attr_y = self.class_attr.index_select(0, t[valid]).to(delta_sem.device)
                        target = self.consistency_head(attr_y)
                        target = F.normalize(target, dim=-1) if self.use_cosine else target
                        self._loss_last_cons_target = target

        return logits


class VSPCNBaselineClassifier(nn.Module):
    """
    1. 每个类别都有一个原始语义属性向量 a_y
       例如 CUB 中每个类别是 312 维属性。

    2. 先通过一个线性层 W_d，把类别属性映射到视觉特征空间：
           \tilde{a}_y = a_y · W_d
       也就是：
           prototype_y = prototype_proj(a_y)

    3. 对输入图像，取 backbone 输出的 cls_feat 作为图像表征。

    4. 最后做最简单的分类打分：
           logits = cls_feat @ prototype_bank^T

    也就是说：
    - 图像侧：直接用 cls_feat
    - 语义侧：class_attr 经过一个线性映射得到类别原型
    - 分类：图像特征与类别原型做点积
    """

    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg,) -> None:
        super().__init__()
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]
        self.attr_dim = class_attr.shape[-1]
        self.hidden_size = int(hidden_size)

        # VSPCN Eq.(12): \tilde{a}_y = a_y · W_d
        self.prototype_proj = nn.Linear(self.attr_dim, self.hidden_size, bias=True)
        self.visual_proj = None

        self.use_cosine = False
        self.fixed_logit_scale = 0.0
        self.logit_scale = None

        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self.debug_shapes = cfg.SOLVER.DEBUG_SHAPES
        self._debug_logged = False
        self._shape_debug_logged = False

        self._loss_last_cls_token = None
        self._loss_last_projected_prototypes = None
        self._loss_last_scale = None

        self._runtime_targets = None
        self._runtime_token_sequence = None
        self._runtime_affinities = None
        self._runtime_semantic_state = None

    def _project_class_prototypes(self) -> torch.Tensor:
        return self.prototype_proj(self.class_attr)

    def _resolve_active_class_space(self, class_ids, device: torch.device):
        if class_ids is None:
            active_ids = torch.arange(self.num_classes, device=device, dtype=torch.long)
        elif torch.is_tensor(class_ids):
            active_ids = class_ids.to(device=device, dtype=torch.long).view(-1)
        else:
            active_ids = torch.as_tensor(list(class_ids), device=device, dtype=torch.long).view(-1)
        if active_ids.numel() == 0:
            raise ValueError("Active class space is empty.")
        return active_ids

    def forward(self, cls_feat: torch.Tensor, class_ids=None) -> torch.Tensor:
        """
        Step 1. 解析当前 active class space
        Step 2. 从全局类别原型库中切出当前 active class 的 prototype bank
        Step 3. 用 cls_feat 与 prototype bank 做点积分类
        Step 4. 打印 shape / trace debug（只一次）
        Step 5. 缓存中间量供 VSPCNBaselineLoss 使用"""
        active_class_ids = self._resolve_active_class_space(class_ids, device=cls_feat.device)
        proto_bank = self._project_class_prototypes().index_select(0, active_class_ids)
        logits = cls_feat @ proto_bank.t()

        if self.debug_shapes and (not self._shape_debug_logged):
            print(
                "[SHAPE-DEBUG] VSPCNBaselineClassifier.forward cls_feat={} proto_bank={} logits={}".format(
                    tuple(cls_feat.shape) if torch.is_tensor(cls_feat) else None,
                    tuple(proto_bank.shape) if torch.is_tensor(proto_bank) else None,
                    tuple(logits.shape) if torch.is_tensor(logits) else None,
                )
            )
            self._shape_debug_logged = True

        if self.debug_trace_once and (not self._debug_logged):
            print(
                "[trace] node=C.vspcn_baseline classifier=VSPCNBaselineClassifier score=cls_dot_attrWd logits_shape={} active_classes={}".format(
                    tuple(logits.shape),
                    int(active_class_ids.numel()),
                )
            )
            self._debug_logged = True

        self._loss_last_cls_token = cls_feat
        self._loss_last_projected_prototypes = proto_bank
        self._loss_last_scale = logits.new_tensor(1.0)

        return logits

def build_loss(cfg):
    """
    Build loss module from cfg.SOLVER.LOSS.
    """
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    return loss_fn(cfg)
