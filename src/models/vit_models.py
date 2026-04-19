#!/usr/bin/env python3

"""
ViT-related models ViT
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn
from .build_vit_backbone import (build_vit_sup_models)
from ..utils import logging
logger = logging.get_logger("visual_prompt")
from ..solver.losses import RSimilarityClassifier, VSPCNBaselineClassifier
from ..utils.param_logging import log_trainable_parameters


class ViT(nn.Module):
    """
    ViT-related model.
    """
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()
        self.cfg = cfg

        prompt_cfg = cfg.MODEL.PROMPT.clone()
        prompt_cfg.defrost()
        prompt_cfg.DEBUG_SHAPES = cfg.SOLVER.DEBUG_SHAPES
        prompt_cfg.SEMANTIC_BRANCH = cfg.MODEL.SEMANTIC_BRANCH.clone()
        prompt_cfg.AFFINITY = cfg.MODEL.AFFINITY.clone()
        prompt_cfg.freeze()

        adapter_cfg = None

        self.build_backbone(prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.r_similarity_head = None
        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self._debug_head_route_logged = False

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):

        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE,         # 棰勮缁冨悕绉帮紝濡?"imagenet21k_sup_vitb16"
            cfg.DATA.CROPSIZE,        # 杈撳叆瑁佸壀灏哄锛堝 224锛?
            prompt_cfg,               # prompt 瀛愰厤缃紙鍙兘涓?None锛?
            cfg.MODEL.MODEL_ROOT,     # 瀛樻斁棰勮缁冩潈閲嶇殑璺緞
            adapter_cfg,              # adapter 瀛愰厤缃紙鍙兘涓?None锛?
            load_pretrain,            # 鏄惁鍔犺浇棰勮缁冩潈閲?
            vis                       # 鍙鍖?璋冭瘯寮€鍏?
        )
        trainable_keys = []
        if cfg.MODEL.PROMPT.ENABLE:
            trainable_keys.extend([
                "prompt_update_layers",
                "prompt_init_provider",
            ])
        if cfg.MODEL.SEMANTIC_BRANCH.ENABLE:
            trainable_keys.append("semantic_side_branch")

        for k, p in self.enc.named_parameters():
            if not any(key in k for key in trainable_keys):
                p.requires_grad = False

        # 鍙€夛細鎵撳嵃鍙缁冨弬鏁扮粺璁★紝渚夸簬纭鍐荤粨绛栫暐鏄惁绗﹀悎棰勬湡
        if self.cfg.MODEL.LOG_TRAINABLE or self.cfg.SOLVER.DBG_TRAINABLE:
            self._log_trainable_parameters()

    def _log_trainable_parameters(self):
        log_trainable_parameters(self, logger, max_examples_per_group=10)

    def log_trainable_parameters(self):
        self._log_trainable_parameters()

    def attach_r_similarity_head(self, class_attributes):

        if not self.cfg.MODEL.R_SIMILARITY.ENABLE:
            raise ValueError("Current prompt-only mainline requires MODEL.R_SIMILARITY.ENABLE=True")

        if class_attributes is None:
            raise ValueError("class_attributes must be provided when R-similarity is enabled")

        if not isinstance(class_attributes, torch.Tensor):
            class_attributes = torch.from_numpy(class_attributes)

        class_attributes = class_attributes.float()

        device = next(self.parameters()).device
        class_attributes = class_attributes.to(device)
        classifier_name = str(self.cfg.MODEL.CLASSIFIER).lower()
        if classifier_name == "r_similarity":
            head_cls = RSimilarityClassifier
        elif classifier_name == "vspcn_baseline":
            head_cls = VSPCNBaselineClassifier
        else:
            raise ValueError(f"Unsupported MODEL.CLASSIFIER='{self.cfg.MODEL.CLASSIFIER}'")

        self.r_similarity_head = head_cls(
            class_attributes,
            hidden_size=self.feat_dim,
            cfg=self.cfg,
        ).to(device)

    def forward(self, x, return_feature=False, semantics=None, class_ids=None):

        x = self.enc(x, semantics=semantics)  # batch_size x self.feat_dim
        self._last_bad_enc_rows = None
        if torch.is_tensor(x) and x.dim() == 2:
            row_enc_ok = torch.isfinite(x).all(dim=1)
            if not bool(row_enc_ok.all().item()):
                self._last_bad_enc_rows = (~row_enc_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        feature_norm_mean = None
        if torch.is_tensor(x) and x.dim() == 2:
            with torch.no_grad():
                feature_norm_mean = float(x.float().norm(dim=-1).mean().item())

        if return_feature:
            return x, x

        if self.r_similarity_head is None:
            raise ValueError("r_similarity_head must be attached before ViT.forward is used.")

        self.r_similarity_head._runtime_token_sequence = None
        self.r_similarity_head._runtime_affinities = None
        transformer = getattr(self.enc, "transformer", None)
        self.r_similarity_head._runtime_semantic_state = (
            getattr(transformer, "_last_semantic_side_state", None) if transformer is not None else None
        )
        x = self.r_similarity_head(x, class_ids=class_ids)
        logits_source = "r_similarity_head"

        if self.debug_trace_once and not self._debug_head_route_logged:
            trace_id = getattr(self, "_debug_trace_id", "trace=NA")
            prompt_path_info = None
            prompt_role_stats = None
            transformer = getattr(self.enc, "transformer", None)
            if transformer is not None:
                prompt_path_info = getattr(transformer, "_last_prompt_path_info", None)
                prompt_role_stats = getattr(transformer, "_last_prompt_role_stats", None)
            logger.info(
                "[trace] %s node=C.vit_models.forward use_r_similarity_head=%s logits_source=%s logits_shape=%s "
                "final_cls_or_pooled_feature_norm=%s prompt_path_info=%s prompt_role_stats=%s",
                trace_id,
                True,
                logits_source,
                tuple(x.shape) if torch.is_tensor(x) else None,
                feature_norm_mean,
                prompt_path_info if isinstance(prompt_path_info, dict) else None,
                prompt_role_stats if isinstance(prompt_role_stats, dict) else None,
            )
            self._debug_head_route_logged = True

        return x
    
    def forward_cls_layerwise(self, x, semantics=None):

        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """
        get a (batch_size, self.feat_dim) feature
        """
        x = self.enc(x)  # batch_size x self.feat_dim
        return x

    def forward_with_affinity(self, x, affinity_config, semantics=None, vis=False, class_ids=None):
        """
        甯︿翰鍜岃緭鍑虹殑鍓嶅悜鎺ュ彛锛氫笌 enc/backbone 鐨?forward_with_affinity 骞宠銆?

        杩斿洖:
          - vis=False: logits, affinities
          - vis=True:  logits, attn_weights, affinities
        """
        if vis:
            feats, attn_weights, affinities = self.enc.forward_with_affinity(
                x, affinity_config, semantics=semantics, vis=vis
            )
        else:
            feats, affinities = self.enc.forward_with_affinity(
                x, affinity_config, semantics=semantics
            )
            attn_weights = None

        # Cache token-level runtime context for scoring heads that need late visual tokens / affinities.
        if self.r_similarity_head is None:
            raise ValueError("r_similarity_head must be attached before ViT.forward_with_affinity is used.")

        self.r_similarity_head._runtime_token_sequence = feats.detach() if torch.is_tensor(feats) else None
        self.r_similarity_head._runtime_affinities = affinities
        transformer = getattr(self.enc, "transformer", None)
        self.r_similarity_head._runtime_semantic_state = (
            getattr(transformer, "_last_semantic_side_state", None) if transformer is not None else None
        )

        # 涓?forward 瀵归綈锛歟nc 杈撳嚭鍙兘鏄?[B, 1+N, D] 鎴?[B, D]锛屽彇 CLS 鍚庢帴澶撮儴
        feats = feats[:, 0] if feats.dim() == 3 else feats
        logits = self.r_similarity_head(feats, class_ids=class_ids)

        if not vis:
            return logits, affinities
        return logits, attn_weights, affinities


