#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional


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


# ===========================
# 1. 损失名到类的映射
# ===========================
LOSS = {
    "softmax_cm": None,
    "r_similarity_v2": None,
    "vspcn_baseline": None,
}


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
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


LOSS["softmax_cm"] = SoftmaxCMLoss


class VSPCNBaselineLoss(nn.Module):
    """
    VSPCN-style baseline:
      L = CE(logits, y) + lambda_ar * mean(||cls_feat - proto_y||_2^2)
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.lambda_ar = float(cfg.SOLVER.LOSS_VSPCN_AR_WEIGHT)
        self._last_loss_stats: Dict[str, float] = {}

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
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


LOSS["vspcn_baseline"] = VSPCNBaselineLoss


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
        super().__init__()
        self.align_mode = str(cfg.SOLVER.RSIM_V2.ALIGN_MODE).lower()
        self.align_weight = float(cfg.SOLVER.RSIM_V2.ALIGN_WEIGHT)
        self.diag_strict = cfg.SOLVER.DIAG.STRICT_CHECKS
        self._last_loss_stats: Dict[str, float] = {}

    def is_single(self):
        return True

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
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
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


LOSS["r_similarity_v2"] = RSimilarityLossV2

# ===========================
# 4. 基于共享语义原型的相似度分类头
# ===========================
class RSimilarityClassifier(nn.Module):
    """
    当前主线使用的视觉-语义相似度分类头。

    这类的职责可以概括成 4 步：
    1. 从全局 `class_attr` 构造“当前活动类空间”的语义原型；
    2. 把视觉输入映射到比较空间；
    3. 把语义输入映射到比较空间；
    4. 在比较空间里做 cosine 或 dot-product 分类。

    当前统一后的命名约定如下：
    - `visual_input`
      指 backbone 直接输出给分类头的视觉输入，通常就是 CLS 特征。
    - `visual_repr`
      指真正参与相似度计算、也参与 CM loss 的视觉表示。
      若启用了 `visual_proj`，它是投影后的结果；否则与 `visual_input` 相同。
    - `semantic_input`
      指从 `class_attr` 经过 `prototype_proj` 得到的类别原型输入。
      这一步已经是“类级语义原型”，但还没进入最终比较空间。
    - `semantic_repr`
      指真正参与相似度计算、也参与 CM loss 的语义表示。
      它来自 `semantic_proj(semantic_input)`。

    这样命名后，你在看两套头时只需要记一件事：
    - `*_input` 是原料层；
    - `*_repr` 是分类/损失真正使用的表示层。
    """

    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg,) -> None:
        super().__init__()
        proj_dim = cfg.MODEL.R_SIMILARITY.PROJ_DIM
        if proj_dim is None or proj_dim <= 0:
            proj_dim = hidden_size

        # 保存全局 class-level semantic bank。
        # 注意这里存的是“原始类语义”，不是已经投影好的 prototype。
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]
        self.attr_dim = class_attr.shape[-1]
        self.hidden_size = hidden_size

        self.visual_proj_enabled = cfg.MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE
        out_dim = proj_dim if self.visual_proj_enabled else hidden_size
        # 视觉侧：
        # - 如果开 visual_proj，就先把视觉输入映射到比较空间；
        # - 否则 visual_repr 直接等于 visual_input。
        self.visual_proj = nn.Linear(hidden_size, out_dim) if self.visual_proj_enabled else None

        # 语义侧分成两层：
        # 1. prototype_proj: 原始属性 -> 类别原型输入
        # 2. semantic_proj : 类别原型输入 -> 最终比较空间语义表示
        #
        # 这种拆法的好处是：
        # - 和 baseline 头可以共享 `semantic_input` 这层语义；
        # - 主线又能保留自己原来的“再过一层 semantic_proj”的分类风格。
        self.semantic_proj = nn.Linear(hidden_size, out_dim)
        self.prototype_proj = nn.Linear(self.attr_dim, hidden_size)

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

        # loss / trainer / monitor 会从这些缓存里读当前 batch 的中间量。
        #
        # 统一约定：
        # - input: 原料层
        # - repr : 真正拿去分类和算 CM 的表示层
        self._loss_last_visual_input = None
        self._loss_last_visual_repr = None
        self._loss_last_semantic_input = None
        self._loss_last_semantic_repr = None
        self._loss_last_logit_scale = None
        self._loss_last_semantic_delta = None
        self._loss_last_consistency_target = None
        self._loss_last_semantic_final = None
        self._loss_last_semantic_anchor = None
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

    def _project_class_prototypes(self) -> torch.Tensor:
        """
        构造“全局类别语义原型输入”。

        输入来源：- self.class_attr : [num_classes, attr_dim]
        处理：- 通过 prototype_proj 映射到 hidden_size
        输出：- [num_classes, hidden_size]

        直观理解：
        - 每个类别 c 都有一份 class-level 原始语义 `s_raw_c`
        - `prototype_proj` 把它映射到 hidden_size
        - 得到的张量就是统一命名下的 `semantic_input`

        注意：
        - 这是全局所有类的语义原型
        - 还没根据 class_ids 切 active class space
        - 也还没经过 semantic_proj 进入最终比较空间
        """
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
        mapping = torch.full((self.num_classes,), -1, device=device, dtype=torch.long)
        mapping[active_ids] = torch.arange(active_ids.numel(), device=device, dtype=torch.long)
        return active_ids, mapping

    def forward(self, cls_feat: torch.Tensor, class_ids=None) -> torch.Tensor:
        """
        当前 batch 的分类前向。

        输入：
        - `cls_feat`
          形状 `[B, hidden_size]`，通常就是 backbone 给出的 CLS 特征。
        - `class_ids`
          当前 active class space 对应的 global class ids。
          如果是 None，表示默认所有类都参与分类。

        输出：
        - `logits`
          形状 `[B, num_active_classes]`，表示当前 batch 在当前活动类空间上的分类分数。

        详细流程：
        1. 先根据 `class_ids` 确定这次 forward 实际参与竞争的类空间；
        2. 从全局 `class_attr` 里取出这些类的原始语义，并投影成 `semantic_input`；
        3. 若启用 shuffle 诊断，则随机打乱当前 prototype 顺序；
        4. 视觉侧把 `cls_feat` 记成 `visual_input`，再可选经过 `visual_proj` 得到 `visual_repr`；
        5. 语义侧把 `semantic_input` 再经过 `semantic_proj` 得到 `semantic_repr`；
        6. 在 `visual_repr` 与 `semantic_repr` 之间做 cosine 或 dot-product；
        7. 把本次 forward 用到的关键张量缓存下来，供：
           - CE / CM loss
           - trainer monitor
           - debug / 可视化
           直接复用；
        8. 如果 semantic side branch 这次也产出了运行时语义状态，
           再额外把 refined semantic / delta_sem / consistency target 缓存下来。
        """
        active_class_ids, active_global_to_local = self._resolve_active_class_space(class_ids, device=cls_feat.device)
        semantic_input_full = self._project_class_prototypes()
        semantic_input = semantic_input_full.index_select(0, active_class_ids)

        # 这个 shuffle 只用于诊断实验，不是正常训练逻辑。
        if self.shuffle_prototypes and semantic_input.shape[0] > 1:
            perm = torch.randperm(semantic_input.shape[0], device=semantic_input.device)
            semantic_input = semantic_input.index_select(0, perm)

        visual_input = cls_feat
        visual_repr = self.visual_proj(visual_input) if self.visual_proj else visual_input
        semantic_repr = self.semantic_proj(semantic_input)

        if self.debug_shapes and (not self._shape_debug_logged):
            print(
                "[SHAPE-DEBUG] RSimilarityClassifier.forward visual_input={} semantic_input={} "
                "visual_repr={} semantic_repr={} logits={} active_classes={}".format(
                    tuple(cls_feat.shape) if torch.is_tensor(cls_feat) else None,
                    tuple(semantic_input.shape) if torch.is_tensor(semantic_input) else None,
                    tuple(visual_repr.shape) if torch.is_tensor(visual_repr) else None,
                    tuple(semantic_repr.shape) if torch.is_tensor(semantic_repr) else None,
                    (int(visual_repr.shape[0]), int(semantic_repr.shape[0])) if (torch.is_tensor(visual_repr) and torch.is_tensor(semantic_repr)) else None,
                    int(active_class_ids.numel()),
                )
            )
            self._shape_debug_logged = True

        # 分类分数有两种模式：
        # 1. cosine: 先归一化，再乘 logit scale
        # 2. dot    : 直接点积，scale 视为 1
        if self.use_cosine:
            visual_repr = F.normalize(visual_repr, dim=-1)
            semantic_repr = F.normalize(semantic_repr, dim=-1)
            raw_sim = visual_repr @ semantic_repr.t()
            scale = raw_sim.new_tensor(self.fixed_logit_scale) if self.fixed_logit_scale > 0 else self.logit_scale.exp()
            logits = raw_sim * scale
        else:
            logits = visual_repr @ semantic_repr.t()
            raw_sim = logits
            scale = logits.new_tensor(1.0)

        # 下面这组 finite 检查只服务于排查 NaN / inf。
        row_feat_ok = torch.isfinite(cls_feat).all(dim=1)
        row_visual_ok = torch.isfinite(visual_repr).all(dim=1)
        sem_ok = torch.isfinite(semantic_repr).all()
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
                "semantic_input_shape={} semantic_repr_shape={} semantic_input_norm_mean={:.6f} "
                "classifier_semantic_norm_mean={:.6f} effective_logit_scale={:.6f} whether_fixed_logit_scale={}".format(
                    "global_raw",
                    "prototype_proj",
                    tuple(semantic_input.shape),
                    tuple(semantic_repr.shape),
                    float(semantic_input.float().norm(dim=-1).mean().item()),
                    float(semantic_repr.float().norm(dim=-1).mean().item()),
                    scale_scalar,
                    fixed_mode,
                )
            )
        self._debug_sem_source_logged = True

        # 把“原料层”和“表示层”都缓存下来：
        # - loss 主要用 repr
        # - baseline 对照、debug、monitor 有时会直接看 input
        self._loss_last_visual_input = visual_input
        self._loss_last_visual_repr = visual_repr
        self._loss_last_semantic_input = semantic_input
        self._loss_last_semantic_repr = semantic_repr
        self._loss_last_logit_scale = scale
        self._loss_last_consistency_target = None
        self._loss_last_semantic_delta = None
        self._loss_last_semantic_final = None
        self._loss_last_semantic_anchor = None

        # 这部分是“分类头额外把 semantic side branch 的运行时结果转存成 loss 可读缓存”。
        # 主分类本身不依赖这里，但 consistency / AGR 等附加项会读这些量。
        sem_state = self._runtime_semantic_state if isinstance(self._runtime_semantic_state, dict) else None
        if sem_state is not None:
            semantic_output = sem_state.get("semantic_output")
            semantic_input = sem_state.get("semantic_input")
            semantic_delta = sem_state.get("semantic_delta")
            if torch.is_tensor(semantic_output) and torch.is_tensor(semantic_input):
                delta_sem = semantic_delta
                if (delta_sem is None) or (not torch.is_tensor(delta_sem)):
                    delta_sem = semantic_output - semantic_input
                self._loss_last_semantic_final = semantic_output
                self._loss_last_semantic_anchor = semantic_input
                self._loss_last_semantic_delta = delta_sem
                if self.consistency_head is not None and runtime_targets_global is not None and runtime_targets_global.numel() == delta_sem.shape[0]:
                    t = runtime_targets_global.to(delta_sem.device)
                    valid = active_global_to_local.to(delta_sem.device).index_select(0, t) >= 0
                    if valid.any():
                        attr_y = self.class_attr.index_select(0, t[valid]).to(delta_sem.device)
                        target = self.consistency_head(attr_y)
                        target = F.normalize(target, dim=-1) if self.use_cosine else target
                        self._loss_last_consistency_target = target
            else:
                mu_s_final = sem_state.get("mu_s_final")
                h_y = sem_state.get("h_y")
                delta_sem = sem_state.get("delta_sem")
                if torch.is_tensor(mu_s_final) and torch.is_tensor(h_y):
                    if (delta_sem is None) or (not torch.is_tensor(delta_sem)):
                        delta_sem = mu_s_final - h_y
                    self._loss_last_semantic_final = mu_s_final
                    self._loss_last_semantic_anchor = h_y
                    self._loss_last_semantic_delta = delta_sem
                    if self.consistency_head is not None and runtime_targets_global is not None and runtime_targets_global.numel() == delta_sem.shape[0]:
                        t = runtime_targets_global.to(delta_sem.device)
                        valid = active_global_to_local.to(delta_sem.device).index_select(0, t) >= 0
                        if valid.any():
                            attr_y = self.class_attr.index_select(0, t[valid]).to(delta_sem.device)
                            target = self.consistency_head(attr_y)
                            target = F.normalize(target, dim=-1) if self.use_cosine else target
                            self._loss_last_consistency_target = target

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
        self.semantic_proj = None

        self.use_cosine = False
        self.fixed_logit_scale = 0.0
        self.logit_scale = None

        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self.debug_shapes = cfg.SOLVER.DEBUG_SHAPES
        self._debug_logged = False
        self._shape_debug_logged = False

        self._loss_last_visual_input = None
        self._loss_last_visual_repr = None
        self._loss_last_semantic_input = None
        self._loss_last_semantic_repr = None
        self._loss_last_logit_scale = None

        self._runtime_targets = None
        self._runtime_token_sequence = None
        self._runtime_affinities = None
        self._runtime_semantic_state = None
        self._last_bad_feat_rows = None
        self._last_bad_visual_rows = None
        self._last_bad_sim_rows = None
        self._last_semantic_all_finite = None

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
        semantic_input = self._project_class_prototypes().index_select(0, active_class_ids)
        visual_input = cls_feat
        visual_repr = visual_input
        semantic_repr = semantic_input
        logits = visual_repr @ semantic_repr.t()
        row_feat_ok = torch.isfinite(visual_input).all(dim=1)
        row_visual_ok = torch.isfinite(visual_repr).all(dim=1)
        row_sim_ok = torch.isfinite(logits).all(dim=1)
        sem_ok = torch.isfinite(semantic_repr).all()

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

        if self.debug_shapes and (not self._shape_debug_logged):
            print(
                "[SHAPE-DEBUG] VSPCNBaselineClassifier.forward visual_input={} semantic_input={} logits={}".format(
                    tuple(visual_input.shape) if torch.is_tensor(visual_input) else None,
                    tuple(semantic_input.shape) if torch.is_tensor(semantic_input) else None,
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

        self._loss_last_visual_input = visual_input
        self._loss_last_visual_repr = visual_repr
        self._loss_last_semantic_input = semantic_input
        self._loss_last_semantic_repr = semantic_repr
        self._loss_last_logit_scale = logits.new_tensor(1.0)

        return logits


class RSimilarityClassifierV2(nn.Module):
    """
    最小化重构版 RSimilarity 分类头。

    共同主干只有两层原料：
    - visual_input   : backbone 输出的 cls_feat
    - semantic_input : prototype_proj(class_attr)

    然后根据 `SCORE_MODE` 切两种受控模式：
    - dot:
        visual_repr   = visual_input
        semantic_repr = semantic_input
        logits        = visual_repr @ semantic_repr^T
      这条可以用来验证是否与 baseline 等价。

    - cosine:
        visual_repr   = normalize(visual_input)
        semantic_repr = normalize(semantic_input)
        logits        = (visual_repr @ semantic_repr^T) * scale
      这条表示“在 baseline 主干上只加入归一化”的版本。
    """

    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg,) -> None:
        super().__init__()
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]
        self.attr_dim = class_attr.shape[-1]
        self.hidden_size = int(hidden_size)

        self.prototype_proj = nn.Linear(self.attr_dim, self.hidden_size, bias=True)
        self.visual_proj = None
        self.semantic_proj = None

        self.score_mode = str(cfg.MODEL.R_SIMILARITY_V2.SCORE_MODE).lower()
        self.learnable_scale = bool(cfg.MODEL.R_SIMILARITY_V2.LEARNABLE_SCALE)
        self.fixed_logit_scale = float(cfg.MODEL.R_SIMILARITY_V2.FIXED_LOGIT_SCALE)
        self.use_cosine = self.score_mode == "cosine"
        self.logit_scale = None
        if self.use_cosine and self.learnable_scale:
            logit_scale_init = float(cfg.MODEL.R_SIMILARITY_V2.LOGIT_SCALE_INIT)
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(logit_scale_init, dtype=torch.float32)))

        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self.debug_shapes = cfg.SOLVER.DEBUG_SHAPES
        self._debug_logged = False
        self._shape_debug_logged = False

        self._loss_last_visual_input = None
        self._loss_last_visual_repr = None
        self._loss_last_semantic_input = None
        self._loss_last_semantic_repr = None
        self._loss_last_logit_scale = None
        self._loss_last_semantic_delta = None
        self._loss_last_consistency_target = None
        self._loss_last_semantic_final = None
        self._loss_last_semantic_anchor = None

        self._runtime_targets = None
        self._runtime_token_sequence = None
        self._runtime_affinities = None
        self._runtime_semantic_state = None
        self._last_bad_feat_rows = None
        self._last_bad_visual_rows = None
        self._last_bad_sim_rows = None
        self._last_semantic_all_finite = None

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
        active_class_ids = self._resolve_active_class_space(class_ids, device=cls_feat.device)
        semantic_input = self._project_class_prototypes().index_select(0, active_class_ids)
        visual_input = cls_feat

        if self.score_mode == "dot":
            visual_repr = visual_input
            semantic_repr = semantic_input
            logits = visual_repr @ semantic_repr.t()
            scale = logits.new_tensor(1.0)
        elif self.score_mode == "cosine":
            visual_repr = F.normalize(visual_input, dim=-1)
            semantic_repr = F.normalize(semantic_input, dim=-1)
            raw_sim = visual_repr @ semantic_repr.t()
            if self.fixed_logit_scale > 0:
                scale = raw_sim.new_tensor(self.fixed_logit_scale)
            elif self.logit_scale is not None:
                scale = self.logit_scale.exp()
            else:
                scale = raw_sim.new_tensor(1.0)
            logits = raw_sim * scale
        else:
            raise ValueError(f"Unsupported MODEL.R_SIMILARITY_V2.SCORE_MODE='{self.score_mode}'")

        row_feat_ok = torch.isfinite(visual_input).all(dim=1)
        row_visual_ok = torch.isfinite(visual_repr).all(dim=1)
        row_sim_ok = torch.isfinite(logits).all(dim=1)
        sem_ok = torch.isfinite(semantic_repr).all()

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

        if self.debug_shapes and (not self._shape_debug_logged):
            print(
                "[SHAPE-DEBUG] RSimilarityClassifierV2.forward visual_input={} semantic_input={} visual_repr={} semantic_repr={} logits={} active_classes={}".format(
                    tuple(visual_input.shape) if torch.is_tensor(visual_input) else None,
                    tuple(semantic_input.shape) if torch.is_tensor(semantic_input) else None,
                    tuple(visual_repr.shape) if torch.is_tensor(visual_repr) else None,
                    tuple(semantic_repr.shape) if torch.is_tensor(semantic_repr) else None,
                    tuple(logits.shape) if torch.is_tensor(logits) else None,
                    int(active_class_ids.numel()),
                )
            )
            self._shape_debug_logged = True

        if self.debug_trace_once and (not self._debug_logged):
            print(
                "[trace] node=C.r_similarity_v2 classifier=RSimilarityClassifierV2 score_mode={} logits_shape={} active_classes={}".format(
                    self.score_mode,
                    tuple(logits.shape),
                    int(active_class_ids.numel()),
                )
            )
            self._debug_logged = True

        self._loss_last_visual_input = visual_input
        self._loss_last_visual_repr = visual_repr
        self._loss_last_semantic_input = semantic_input
        self._loss_last_semantic_repr = semantic_repr
        self._loss_last_logit_scale = scale
        self._loss_last_semantic_delta = None
        self._loss_last_consistency_target = None
        self._loss_last_semantic_final = None
        self._loss_last_semantic_anchor = None

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
