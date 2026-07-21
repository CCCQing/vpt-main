#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSimilarityClassifier(nn.Module):
    def __init__(self, class_attr: torch.Tensor, hidden_size: int, cfg) -> None:
        super().__init__()
        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = int(class_attr.shape[0])
        self.attr_dim = int(class_attr.shape[-1])
        self.hidden_size = int(hidden_size)

        self.prototype_proj = nn.Linear(self.attr_dim, self.hidden_size, bias=True)

        rsim_cfg = cfg.MODEL.R_SIMILARITY
        self.score_mode = str(rsim_cfg.SCORE_MODE).lower()
        if self.score_mode not in {"dot", "cosine"}:
            raise ValueError(
                f"Unsupported MODEL.R_SIMILARITY.SCORE_MODE='{rsim_cfg.SCORE_MODE}'. "
                "Expected dot or cosine."
            )

        self.learnable_scale = bool(rsim_cfg.LEARNABLE_SCALE)
        self.fixed_logit_scale = float(rsim_cfg.FIXED_LOGIT_SCALE)
        self.use_cosine = self.score_mode == "cosine"
        self.logit_scale = None
        if self.use_cosine and self.learnable_scale:
            logit_scale_init = float(rsim_cfg.LOGIT_SCALE_INIT)
            if logit_scale_init <= 0:
                raise ValueError("MODEL.R_SIMILARITY.LOGIT_SCALE_INIT must be positive.")
            self.logit_scale = nn.Parameter(
                torch.log(torch.tensor(logit_scale_init, dtype=torch.float32))
            )

        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self.debug_shapes = cfg.SOLVER.DEBUG_SHAPES
        self._debug_logged = False
        self._shape_debug_logged = False
        self._init_runtime_cache()

    def _init_runtime_cache(self) -> None:
        self._loss_last_visual_input = None
        self._loss_last_visual_repr = None
        self._loss_last_semantic_input = None
        self._loss_last_semantic_repr = None
        self._loss_last_logit_scale = None
        self._last_bad_feat_rows = None
        self._last_bad_visual_rows = None
        self._last_bad_sim_rows = None
        self._last_semantic_all_finite = None

    def _project_class_prototypes(self) -> torch.Tensor:
        return self.prototype_proj(self.class_attr)

    def _resolve_active_class_space(self, class_ids, device: torch.device) -> torch.Tensor:
        if class_ids is None:
            active_ids = torch.arange(self.num_classes, device=device, dtype=torch.long)
        elif torch.is_tensor(class_ids):
            active_ids = class_ids.to(device=device, dtype=torch.long).view(-1)
        else:
            active_ids = torch.as_tensor(
                list(class_ids), device=device, dtype=torch.long
            ).view(-1)
        if active_ids.numel() == 0:
            raise ValueError("Active class space is empty.")
        return active_ids

    def _cache_finite_checks(
        self,
        visual_input: torch.Tensor,
        visual_repr: torch.Tensor,
        semantic_repr: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        row_feat_ok = torch.isfinite(visual_input).all(dim=1)
        row_visual_ok = torch.isfinite(visual_repr).all(dim=1)
        row_sim_ok = torch.isfinite(logits).all(dim=1)
        sem_ok = torch.isfinite(semantic_repr).all()

        self._last_bad_feat_rows = None
        self._last_bad_visual_rows = None
        self._last_bad_sim_rows = None
        self._last_semantic_all_finite = bool(sem_ok.item())
        if not bool(row_feat_ok.all().item()):
            self._last_bad_feat_rows = (
                (~row_feat_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            )
        if not bool(row_visual_ok.all().item()):
            self._last_bad_visual_rows = (
                (~row_visual_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            )
        if not bool(row_sim_ok.all().item()):
            self._last_bad_sim_rows = (
                (~row_sim_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            )

    def forward(
        self,
        cls_feat: torch.Tensor,
        class_ids=None,
        semantic_state=None,
        runtime_targets=None,
    ) -> torch.Tensor:
        active_class_ids = self._resolve_active_class_space(
            class_ids, device=cls_feat.device
        )
        semantic_input = self._project_class_prototypes().index_select(
            0, active_class_ids
        )
        visual_input = cls_feat

        if self.score_mode == "dot":
            visual_repr = visual_input
            semantic_repr = semantic_input
            logits = visual_repr @ semantic_repr.t()
            scale = logits.new_tensor(1.0)
        else:
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

        self._cache_finite_checks(
            visual_input, visual_repr, semantic_repr, logits
        )

        if self.debug_shapes and not self._shape_debug_logged:
            print(
                "[SHAPE-DEBUG] RSimilarityClassifier.forward visual_input={} "
                "semantic_input={} visual_repr={} semantic_repr={} logits={} "
                "active_classes={}".format(
                    tuple(visual_input.shape),
                    tuple(semantic_input.shape),
                    tuple(visual_repr.shape),
                    tuple(semantic_repr.shape),
                    tuple(logits.shape),
                    int(active_class_ids.numel()),
                )
            )
            self._shape_debug_logged = True

        if self.debug_trace_once and not self._debug_logged:
            print(
                "[trace] node=C.r_similarity classifier=RSimilarityClassifier "
                "score_mode={} logits_shape={} active_classes={}".format(
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
        return logits
