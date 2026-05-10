#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSimilarityClassifier(nn.Module):
    """
    视觉-语义相似度分类头。

    这个模块只负责把 backbone 的 CLS 表征和 class-level 语义原型做打分，
    同时缓存 loss / debug / visualization 需要读取的中间量。
    """

    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg) -> None:
        """
        初始化标准 RSimilarity 分类头。

        参数:
        - class_attr: 全部类别的原始语义属性矩阵
        - hidden_size: backbone 输出维度
        - cfg: 控制投影、打分方式、诊断开关的配置
        """
        super().__init__()
        proj_dim = cfg.MODEL.R_SIMILARITY.PROJ_DIM
        if proj_dim is None or proj_dim <= 0:
            proj_dim = hidden_size

        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]
        self.attr_dim = class_attr.shape[-1]
        self.hidden_size = hidden_size

        self.visual_proj_enabled = cfg.MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE
        out_dim = proj_dim if self.visual_proj_enabled else hidden_size
        self.visual_proj = nn.Linear(hidden_size, out_dim) if self.visual_proj_enabled else None
        self.semantic_proj = nn.Linear(hidden_size, out_dim)
        self.prototype_proj = nn.Linear(self.attr_dim, hidden_size)

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

        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self.debug_shapes = cfg.SOLVER.DEBUG_SHAPES
        self._debug_sem_source_logged = False
        self._shape_debug_logged = False

        self._init_runtime_cache()

        if self.use_cosine:
            logit_scale_init = cfg.MODEL.R_SIMILARITY.LOGIT_SCALE_INIT
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(logit_scale_init, dtype=torch.float32)))
        else:
            self.logit_scale = None

    def _init_runtime_cache(self):
        """
        初始化运行时缓存字段。

        这些字段会被 loss、trainer 诊断和可视化逻辑读取。
        """
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
        self._runtime_targets = None
        self._runtime_token_sequence = None
        self._runtime_affinities = None
        self._runtime_semantic_state = None

    def _project_class_prototypes(self) -> torch.Tensor:
        """将原始类别属性投影到 hidden_size，得到类别 prototype bank。"""
        return self.prototype_proj(self.class_attr)

    def _resolve_active_class_space(self, class_ids, device: torch.device):
        """
        解析当前 forward 使用的 active class space。

        返回 active class ids 以及 global label 到 local label 的映射。
        """
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

    def _cache_finite_checks(self, cls_feat, visual_repr, semantic_repr, sim):
        """
        缓存 NaN / inf 检查结果。

        只服务于训练诊断，不改变前向计算结果。
        """
        row_feat_ok = torch.isfinite(cls_feat).all(dim=1)
        row_visual_ok = torch.isfinite(visual_repr).all(dim=1)
        row_sim_ok = torch.isfinite(sim).all(dim=1)
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

    def _cache_semantic_branch_state(self, active_global_to_local):
        """
        把 semantic token 分支的运行时状态转存为 loss 可读缓存。

        主要供 consistency / AGR 等语义增量相关损失使用。
        """
        runtime_targets_global = self._runtime_targets if torch.is_tensor(self._runtime_targets) else None
        sem_state = self._runtime_semantic_state if isinstance(self._runtime_semantic_state, dict) else None
        if sem_state is None:
            return

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
        else:
            mu_s_final = sem_state.get("mu_s_final")
            h_y = sem_state.get("h_y")
            delta_sem = sem_state.get("delta_sem")
            if not (torch.is_tensor(mu_s_final) and torch.is_tensor(h_y)):
                return
            if (delta_sem is None) or (not torch.is_tensor(delta_sem)):
                delta_sem = mu_s_final - h_y
            self._loss_last_semantic_final = mu_s_final
            self._loss_last_semantic_anchor = h_y
            self._loss_last_semantic_delta = delta_sem

        if self.consistency_head is None or runtime_targets_global is None:
            return
        if runtime_targets_global.numel() != self._loss_last_semantic_delta.shape[0]:
            return
        t = runtime_targets_global.to(self._loss_last_semantic_delta.device)
        valid = active_global_to_local.to(self._loss_last_semantic_delta.device).index_select(0, t) >= 0
        if valid.any():
            attr_y = self.class_attr.index_select(0, t[valid]).to(self._loss_last_semantic_delta.device)
            target = self.consistency_head(attr_y)
            self._loss_last_consistency_target = F.normalize(target, dim=-1) if self.use_cosine else target

    def forward(self, cls_feat: torch.Tensor, class_ids=None) -> torch.Tensor:
        """
        根据 CLS 特征和 active class prototypes 计算分类 logits。

        同时缓存 visual/semantic input 与 repr，供后续 loss 复用。
        """
        active_class_ids, active_global_to_local = self._resolve_active_class_space(class_ids, device=cls_feat.device)
        semantic_input = self._project_class_prototypes().index_select(0, active_class_ids)

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
                    tuple(cls_feat.shape),
                    tuple(semantic_input.shape),
                    tuple(visual_repr.shape),
                    tuple(semantic_repr.shape),
                    (int(visual_repr.shape[0]), int(semantic_repr.shape[0])),
                    int(active_class_ids.numel()),
                )
            )
            self._shape_debug_logged = True

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

        self._cache_finite_checks(cls_feat, visual_repr, semantic_repr, raw_sim)

        if self.debug_trace_once and (not self._debug_sem_source_logged):
            scale_scalar = float(scale.item()) if torch.is_tensor(scale) else float(scale)
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
                    bool(self.fixed_logit_scale > 0),
                )
            )
        self._debug_sem_source_logged = True

        self._loss_last_visual_input = visual_input
        self._loss_last_visual_repr = visual_repr
        self._loss_last_semantic_input = semantic_input
        self._loss_last_semantic_repr = semantic_repr
        self._loss_last_logit_scale = scale
        self._loss_last_consistency_target = None
        self._loss_last_semantic_delta = None
        self._loss_last_semantic_final = None
        self._loss_last_semantic_anchor = None
        self._cache_semantic_branch_state(active_global_to_local)
        return logits


class VSPCNBaselineClassifier(nn.Module):
    """
    VSPCN baseline 分类头。

    做法:
    - 语义侧: class_attr -> prototype_proj -> prototype_bank
    - 视觉侧: 直接使用 backbone 输出的 CLS 特征
    - 分类: logits = cls_feat @ prototype_bank^T
    """

    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg) -> None:
        """初始化 VSPCN baseline 的类别 prototype 投影层和缓存字段。"""
        super().__init__()
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]
        self.attr_dim = class_attr.shape[-1]
        self.hidden_size = int(hidden_size)

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
        self._init_runtime_cache()

    def _init_runtime_cache(self):
        """初始化 loss / debug / visualization 需要读取的运行时缓存。"""
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
        """将原始类别属性投影为 VSPCN baseline 使用的 prototype bank。"""
        return self.prototype_proj(self.class_attr)

    def _resolve_active_class_space(self, class_ids, device: torch.device):
        """解析当前参与分类的类别集合，返回 active class ids。"""
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
        执行 VSPCN baseline 分类前向。

        输出 logits，并缓存 CLS 特征与 prototype，供 AR loss 使用。
        """
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
                    tuple(visual_input.shape),
                    tuple(semantic_input.shape),
                    tuple(logits.shape),
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
    最小化 RSimilarity 分类头。

    与 baseline 共用 prototype_proj 结构，只通过 SCORE_MODE 控制 dot / cosine 打分。
    """

    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg) -> None:
        """初始化 RSimilarity v2 的 prototype 投影、打分模式和缓存字段。"""
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
        self._init_runtime_cache()

    def _init_runtime_cache(self):
        """初始化 loss / debug / visualization 需要读取的运行时缓存。"""
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
        """将原始类别属性投影到 hidden_size，得到类别 prototype bank。"""
        return self.prototype_proj(self.class_attr)

    def _resolve_active_class_space(self, class_ids, device: torch.device):
        """解析当前参与分类的类别集合，返回 active class ids。"""
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
        执行 RSimilarity v2 分类前向。

        SCORE_MODE=dot 时做点积分类；SCORE_MODE=cosine 时做归一化相似度分类。
        """
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
                    tuple(visual_input.shape),
                    tuple(semantic_input.shape),
                    tuple(visual_repr.shape),
                    tuple(semantic_repr.shape),
                    tuple(logits.shape),
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
