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
from .classifiers import RSimilarityClassifier
from ..utils.param_logging import log_trainable_parameters
from ..utils.reproducibility import derive_seed, isolated_torch_cpu_seed


class ViT(nn.Module):
    """
    ViT-related model.
    """
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()
        self.cfg = cfg

        use_plain_vit_backbone = (
            (not cfg.MODEL.PROMPT.ENABLE)
            and (not cfg.MODEL.SEMANTIC_TOKENS.ENABLE)
        )

        if use_plain_vit_backbone:
            prompt_cfg = None
        else:
            prompt_cfg = cfg.MODEL.PROMPT.clone()
            prompt_cfg.defrost()
            prompt_cfg.DEBUG_SHAPES = cfg.SOLVER.DEBUG_SHAPES
            prompt_cfg.SEMANTIC_TOKENS = cfg.MODEL.SEMANTIC_TOKENS.clone()
            prompt_cfg.AFFINITY = cfg.MODEL.AFFINITY.clone()
            prompt_cfg.ATTENTION_MEDIATION = cfg.MODEL.ATTENTION_MEDIATION.clone()
            prompt_cfg.freeze()

        adapter_cfg = None

        self.build_backbone(prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.r_similarity_head = None
        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self._debug_head_route_logged = False
        self.clear_runtime_state()

    def clear_runtime_state(self):
        """
        清空一次 forward 产生的模型级运行时状态。

        这些状态来自 ViT 主干 / semantic tokenizer / affinity monitor，
        不属于分类头内部计算，因此统一挂在 ViT model 上。
        """
        self._runtime_token_sequence = None
        self._runtime_affinities = None
        self._runtime_semantic_state = None
        self._runtime_prompt_distribution_stats = None
        self._runtime_attention_mediation_stats = None

    def get_runtime_semantic_state(self):
        """返回最近一次 forward 产生的 semantic runtime state。"""
        return self._runtime_semantic_state

    def get_runtime_affinities(self):
        """返回最近一次 forward_with_affinity 产生的逐层 affinity。"""
        return self._runtime_affinities

    def get_runtime_token_sequence(self):
        """返回最近一次 forward_with_affinity 的最终 token 序列。"""
        return self._runtime_token_sequence

    def get_runtime_prompt_distribution_stats(self):
        """
        返回最近一次 forward 产生的 prompt distribution 统计量。

        典型内容包括 mu/logvar/std/prompt_tokens。KL loss 和 GraphProbPrior loss
        都从这里读取 mu/logvar，避免把 label 传进 distributor.forward。
        """
        return self._runtime_prompt_distribution_stats

    def get_runtime_attention_mediation_stats(self):
        """Return detached scalar summaries from the latest attention-mediation forward."""
        return self._runtime_attention_mediation_stats

    def get_runtime_classifier_stats(self):
        if self.r_similarity_head is None:
            return None
        fields = {
            "visual_input": "_loss_last_visual_input",
            "visual_repr": "_loss_last_visual_repr",
            "semantic_input": "_loss_last_semantic_input",
            "semantic_repr": "_loss_last_semantic_repr",
        }
        stats = {
            name: getattr(self.r_similarity_head, attribute, None)
            for name, attribute in fields.items()
        }
        return stats if all(torch.is_tensor(value) for value in stats.values()) else None

    def set_runtime_prompt_distribution_override(self, mu, logvar, eps=None):
        transformer = self.enc.transformer
        if not hasattr(transformer, "set_runtime_prompt_distribution_override"):
            raise RuntimeError("Current backbone does not support external prompt distributions.")
        transformer.set_runtime_prompt_distribution_override(mu, logvar, eps=eps)

    def clear_runtime_prompt_distribution_override(self):
        transformer = self.enc.transformer
        if hasattr(transformer, "clear_runtime_prompt_distribution_override"):
            transformer.clear_runtime_prompt_distribution_override()

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):

        self.enc, self.feat_dim = build_vit_sup_models(
            model_type=cfg.DATA.FEATURE,
            crop_size=cfg.DATA.CROPSIZE,
            prompt_cfg=prompt_cfg,
            model_root=cfg.MODEL.MODEL_ROOT,
            adapter_cfg=adapter_cfg,
            load_pretrain=load_pretrain,
            vis=vis,
            prompt_init_seed=derive_seed(cfg.SEED, "prompt_init"),
        )
        trainable_keys = []
        if cfg.MODEL.PROMPT.ENABLE:
            prompt_backend = cfg.MODEL.PROMPT.BACKEND.lower()
            attention_mediation_enable = bool(cfg.MODEL.ATTENTION_MEDIATION.ENABLE)
            if prompt_backend == "dynamic":
                prompt_init_source = cfg.MODEL.PROMPT.INIT_SOURCE.lower()
                if prompt_init_source == "learned":
                    trainable_keys.extend(["prompt_embeddings", "prompt_proj"])
                elif prompt_init_source == "distributor_mean":
                    trainable_keys.extend(["prompt_init_provider", "prompt_proj"])
                else:
                    raise ValueError(
                        f"Unsupported MODEL.PROMPT.INIT_SOURCE='{cfg.MODEL.PROMPT.INIT_SOURCE}'"
                    )
                if attention_mediation_enable:
                    # ATTENTION_MEDIATION 的新增可训练量只有每层 P/S gamma gate。
                    # ViT block 的 Q/K/V/out/MLP 仍被冻结，不把主干 attention 参数加入 optimizer。
                    trainable_keys.append("attention_mediation")
            elif prompt_backend == "vpt_deep":
                prompt_init_source = cfg.MODEL.PROMPT.INIT_SOURCE.lower()
                if prompt_init_source == "learned":
                    trainable_keys.extend([
                        "prompt_embeddings",
                        "deep_prompt_embeddings",
                        "prompt_proj",
                    ])
                elif prompt_init_source == "distributor_mean":
                    trainable_keys.extend([
                        "prompt_init_provider",
                        "deep_prompt_embeddings",
                        "prompt_proj",
                    ])
                else:
                    raise ValueError(
                        f"Unsupported MODEL.PROMPT.INIT_SOURCE='{cfg.MODEL.PROMPT.INIT_SOURCE}'"
                    )
                if attention_mediation_enable:
                    # vpt_deep 下也允许 block 内 attention correction；
                    # 这里同样只解冻 attention_mediation_*gamma，不解冻 ViT 主干。
                    trainable_keys.append("attention_mediation")
            else:
                raise ValueError(f"Unsupported MODEL.PROMPT.BACKEND='{cfg.MODEL.PROMPT.BACKEND}'")
        if cfg.MODEL.SEMANTIC_TOKENS.ENABLE:
            trainable_keys.append("semantic_token_projector")

        for k, p in self.enc.named_parameters():
            if not any(key in k for key in trainable_keys):
                p.requires_grad = False

        # 鍙€夛細鎵撳嵃鍙缁冨弬鏁扮粺璁★紝渚夸簬纭鍐荤粨绛栫暐鏄惁绗﹀悎棰勬湡
    def _log_trainable_parameters(self):
        log_trainable_parameters(self, logger, max_examples_per_group=10)

    def log_trainable_parameters(self):
        self._log_trainable_parameters()

    def attach_r_similarity_head(self, class_attributes):

        if class_attributes is None:
            raise ValueError("class_attributes must be provided for r_similarity")

        if not isinstance(class_attributes, torch.Tensor):
            class_attributes = torch.from_numpy(class_attributes)

        class_attributes = class_attributes.float()

        device = next(self.parameters()).device
        class_attributes = class_attributes.to(device)
        classifier_name = self.cfg.MODEL.CLASSIFIER.lower()
        if classifier_name != "r_similarity":
            raise ValueError(f"Unsupported MODEL.CLASSIFIER='{self.cfg.MODEL.CLASSIFIER}'")

        classifier_init_seed = derive_seed(self.cfg.SEED, "classifier_init")
        with isolated_torch_cpu_seed(classifier_init_seed):
            head = RSimilarityClassifier(
                class_attributes,
                hidden_size=self.feat_dim,
                cfg=self.cfg,
            )
        self.r_similarity_head = head.to(device)
        if self.cfg.MODEL.LOG_TRAINABLE or self.cfg.SOLVER.DBG_TRAINABLE:
            self._log_trainable_parameters()

    def forward(self, x, return_feature=False, semantics=None, class_ids=None, prototype_class_ids=None, runtime_targets=None):

        self.clear_runtime_state()
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

        transformer = self.enc.transformer
        self._runtime_semantic_state = transformer._last_semantic_token_state
        # 缓存 prompt distributor stats，供 loss 侧读取；分类头仍只接收最终 CLS feature。
        self._runtime_prompt_distribution_stats = getattr(transformer, "_last_prompt_distribution_stats", None)
        self._runtime_attention_mediation_stats = getattr(transformer, "_last_attention_mediation_stats", None)
        x = self.r_similarity_head(
            x,
            class_ids=class_ids,
            prototype_class_ids=prototype_class_ids,
            semantic_state=self._runtime_semantic_state,
            runtime_targets=runtime_targets,
        )
        logits_source = "r_similarity_head"

        if self.debug_trace_once and not self._debug_head_route_logged:
            trace_id = self._debug_trace_id
            prompt_path_info = transformer._last_prompt_path_info
            logger.info(
                "[trace] %s node=C.vit_models.forward use_r_similarity_head=%s logits_source=%s logits_shape=%s "
                "final_cls_or_pooled_feature_norm=%s prompt_path_info=%s",
                trace_id,
                True,
                logits_source,
                tuple(x.shape) if torch.is_tensor(x) else None,
                feature_norm_mean,
                prompt_path_info if isinstance(prompt_path_info, dict) else None,
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

    def forward_with_affinity(self, x, affinity_config, semantics=None, vis=False, class_ids=None, prototype_class_ids=None, runtime_targets=None):
        """
        甯︿翰鍜岃緭鍑虹殑鍓嶅悜鎺ュ彛锛氫笌 enc/backbone 鐨?forward_with_affinity 骞宠銆?

        杩斿洖:
          - vis=False: logits, affinities
          - vis=True:  logits, attn_weights, affinities
        """
        self.clear_runtime_state()
        if vis:
            feats, attn_weights, affinities = self.enc.forward_with_affinity(
                x, affinity_config, semantics=semantics, vis=vis
            )
        else:
            feats, affinities = self.enc.forward_with_affinity(
                x, affinity_config, semantics=semantics
            )
            attn_weights = None

        # ViT 主模型缓存 token / affinity / semantic state，loss 和可视化从 model 明确接口读取。
        if self.r_similarity_head is None:
            raise ValueError("r_similarity_head must be attached before ViT.forward_with_affinity is used.")

        transformer = self.enc.transformer
        token_sequence = getattr(transformer, "_last_token_sequence", None)
        self._runtime_token_sequence = token_sequence.detach() if torch.is_tensor(token_sequence) else None
        self._runtime_affinities = affinities
        self._runtime_semantic_state = transformer._last_semantic_token_state
        # forward_with_affinity 路径同样缓存 stats，保证启用 affinity aux 时 KL/graph loss 仍可用。
        self._runtime_prompt_distribution_stats = getattr(transformer, "_last_prompt_distribution_stats", None)
        self._runtime_attention_mediation_stats = getattr(transformer, "_last_attention_mediation_stats", None)

        # 涓?forward 瀵归綈锛歟nc 杈撳嚭鍙兘鏄?[B, 1+N, D] 鎴?[B, D]锛屽彇 CLS 鍚庢帴澶撮儴
        feats = feats[:, 0] if feats.dim() == 3 else feats
        logits = self.r_similarity_head(
            feats,
            class_ids=class_ids,
            prototype_class_ids=prototype_class_ids,
            semantic_state=self._runtime_semantic_state,
            runtime_targets=runtime_targets,
        )

        if not vis:
            return logits, affinities
        return logits, attn_weights, affinities
