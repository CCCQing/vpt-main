#!/usr/bin/env python3
"""
trainer.py

这个文件实现的是整个工程里的“训练/评测总调度器（Trainer）”。

你可以把它理解成：
    训练主链的执行中心 + 调试与可视化中心

它本身不负责：
1. 定义数据协议（这已经在 dataset 层定义好了）
2. 定义模型结构（模型在 build_model / vit_models / losses 中定义）
3. 定义评测数学（singlelabel / evaluator 中定义）

它负责的是把这些东西真正串起来执行：

    dataset 协议
        -> model forward
        -> loss
        -> optimizer / scheduler
        -> evaluator
        -> visualization
        -> checkpoint / best epoch / early-stop

核心职责概览：
1. 构建优化器、学习率调度器、损失函数
2. 根据配置决定是否启用 affinity 分支、辅助损失、visualization
3. 管理 local-output / eval-local-output 的 target remap
4. 组织训练循环 train_classifier()
5. 组织单次 batch 前向 forward_one_batch()
6. 组织评测 eval_classifier()
7. 组织可视化导出
"""
import datetime
import time
import torch
import torch.nn as nn
import os
import numpy as np
import math
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

from ..utils.vis_pipeline import (
    as_long_path,
    ensure_dir,
    to_uint8_image,
    resize_map_torch,
    overlay_heatmap,
    save_overlay,
    save_panel,
    attention_rollout,
)

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    训练/评测主调度器。

    负责：
    1. 组织训练循环和评测流程
    2. 构建优化器、调度器和损失
    3. 管理 local-output / eval-local-output 的 target remap
    4. 导出注意力可视化结果
    """
    def __init__(self, cfg: CfgNode, model: nn.Module, evaluator: Evaluator, device: torch.device,) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        self.cls_criterion = build_loss(self.cfg)
        # loss 对象自己声明是否需要 affinity aux，trainer 不再硬编码具体辅助损失名。
        self.affinity_aux_needed = bool(self.cls_criterion.requires_affinity_aux)
        self._last_semantic_length = 0

        # 涓€涓负鐪熷嵆use_affinity
        self.use_affinity = cfg.MODEL.AFFINITY.ENABLE or self.affinity_aux_needed
        if self.use_affinity:
            self.affinity_cfg = {
                "prompt_length": cfg.MODEL.PROMPT.NUM_TOKENS,
                "semantic_length": cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS if cfg.MODEL.SEMANTIC_TOKENS.ENABLE else 0,
                "detach": cfg.MODEL.AFFINITY.DETACH,
            }
            self.affinity_vis = cfg.MODEL.AFFINITY.VIS
        else:
            self.affinity_cfg = None
            self.affinity_vis = False

        # solver related
        # ================== optimizer / scheduler / loss ==================
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)

        # ================== Checkpointer ==================
        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        # Optional pretrained checkpoint load.
        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.best_gzsl_seen_recorded = float("-inf")
        self.best_gzsl_unseen_recorded = float("-inf")
        self.cpu_device = torch.device("cpu")

        self.debug_grad_norm = cfg.SOLVER.DEBUG_GRAD_NORM
        self.debug_trace_once = cfg.SOLVER.DEBUG_TRACE_ONCE
        self.debug_shapes = cfg.SOLVER.DEBUG_SHAPES

        self._debug_batch_stats_logged = False
        self._debug_grad_logged = False
        self._debug_step_logged = False
        self._debug_forward_trace_logged = False
        self._debug_semantic_param_names_logged = False
        self._shape_debug_aux_logged = False
        self._shape_debug_loss_aux_logged = False
        self._named_param_cache = None

        self._last_train_debug = {}
        self._last_ce_logits = None
        self._last_raw_logits = None

        # BEGIN SEMANTIC_ABLATION_EXPERIMENT
        semantic_cfg = cfg.MODEL.SEMANTIC_TOKENS
        semantic_gen = torch.Generator(device="cpu")
        semantic_gen.manual_seed(int(semantic_cfg.RANDOM_SEED))
        self._semantic_random_fixed = (
            torch.randn(int(semantic_cfg.INPUT_DIM), generator=semantic_gen)
            * float(semantic_cfg.RANDOM_STD)
        )
        # END SEMANTIC_ABLATION_EXPERIMENT

        self._trace_epoch = -1
        self._trace_iter = -1
        self._trace_stage = "init"
        self._trace_global_step = 0
        self._trace_rank = int(cfg.DIST_RANK)

        diag_cfg = cfg.SOLVER.DIAG
        self.diag_shuffle_raw_targets = diag_cfg.SHUFFLE_RAW_TARGETS

        vis_cfg = cfg.SOLVER.VIS
        self.vis_enable = bool(vis_cfg.ENABLE)
        self.vis_final_only = bool(vis_cfg.FINAL_ONLY)
        self.vis_every_epoch = max(1, int(vis_cfg.EVERY_EPOCH))
        self.vis_epoch_list = self._parse_vis_epoch_list(vis_cfg.EPOCH_LIST)
        self.vis_splits = list(vis_cfg.SPLITS)
        self.vis_max_samples = max(1, int(vis_cfg.MAX_SAMPLES))
        self.vis_correct_samples = max(0, int(vis_cfg.CORRECT_SAMPLES))
        self.vis_wrong_samples = max(0, int(vis_cfg.WRONG_SAMPLES))
        self.vis_save_raw = bool(vis_cfg.SAVE_RAW)
        self.vis_save_images = bool(vis_cfg.SAVE_IMAGES)
        self.vis_rollout = bool(vis_cfg.ROLLOUT)
        # BEGIN SEMANTIC_ABLATION_EXPERIMENT
        self.semantic_ablation_enable = bool(vis_cfg.SEMANTIC_ABLATION.ENABLE)
        self.semantic_ablation_delta_asv = bool(vis_cfg.SEMANTIC_ABLATION.DELTA_ASV)
        self.semantic_ablation_attention_mass = bool(vis_cfg.SEMANTIC_ABLATION.ATTENTION_MASS)
        self.semantic_ablation_delta_reference_source = str(vis_cfg.SEMANTIC_ABLATION.DELTA_REFERENCE_SOURCE).lower()
        if self.semantic_ablation_delta_reference_source != "class_mean":
            raise ValueError("SOLVER.VIS.SEMANTIC_ABLATION.DELTA_REFERENCE_SOURCE currently supports only 'class_mean'.")
        # END SEMANTIC_ABLATION_EXPERIMENT
        self.vis_dir = os.path.join(self.cfg.OUTPUT_DIR, "visualization")
        self._last_attn_weights = None
        self._vis_processed = 0
        self._vis_correct_processed = 0
        self._vis_wrong_processed = 0

        if self.vis_enable:
            ensure_dir(self.vis_dir)
            # rollout needs per-layer attention weights
            self.affinity_vis = True
            logger.info(
                "[vis] enable=%s final_only=%s every_epoch=%d epoch_list=%s splits=%s max_samples=%d correct=%d wrong=%d save_raw=%s save_images=%s",
                bool(self.vis_enable),
                bool(self.vis_final_only),
                int(self.vis_every_epoch),
                self.vis_epoch_list,
                self.vis_splits,
                int(self.vis_max_samples),
                int(self.vis_correct_samples),
                int(self.vis_wrong_samples),
                bool(self.vis_save_raw),
                bool(self.vis_save_images),
            )

        if self.debug_grad_norm:
            self._log_optimizer_param_groups()

        if self.debug_trace_once:
            self._log_semantic_param_names_once()

    ### 1. =================================local-output / remap==========================================
    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            return outputs[0]
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        return outputs

    @staticmethod
    def _replace_logits(outputs, logits):
        """
         在保持 outputs 原有结构的前提下，把其中的 logits 替换成新的 logits。

         用途：
         - seen-only CE 时，把 full logits 换成 seen-only logits
         - 保持 tuple/list/dict 的其他辅助信息不丢失

         支持：
         1. tuple -> (new_logits, 其余保持不变)
         2. list  -> list[0] = new_logits
         3. dict  -> out["logits"] = new_logits
         4. 其他  -> 直接返回 logits
         """
        if isinstance(outputs, tuple):
            if len(outputs) == 0:
                return logits
            return (logits,) + tuple(outputs[1:])
        if isinstance(outputs, list):
            if len(outputs) == 0:
                return logits
            out = list(outputs)
            out[0] = logits
            return out
        if isinstance(outputs, dict) and "logits" in outputs:
            out = dict(outputs)
            out["logits"] = logits
            return out
        return logits

    @staticmethod
    def _dataset_space_meta(dataset, use_eval_space: bool):
        if dataset is None:
            raise ValueError("dataset is required for local-output remapping.")
        class_ids = dataset.eval_local_classes if use_eval_space else dataset.local_classes
        mapping = dataset.eval_global_to_local if use_eval_space else dataset.global_to_local
        class_ids = [int(x) for x in list(class_ids)]
        return class_ids, mapping

    def _slice_class_weights_to_space(self, class_ids):
        if self.cls_weights is None:
            return None
        w = np.asarray(self.cls_weights, dtype=np.float32)
        ids = np.asarray(class_ids, dtype=np.int64)
        return w[ids].tolist()

    def _prepare_dataset_local_targets(self, raw_targets, dataset, use_eval_space: bool):
        class_ids, mapping = self._dataset_space_meta(dataset, use_eval_space=use_eval_space)
        if len(class_ids) == 0:
            raise ValueError("Local-output class space is empty for split '{}'.".format(dataset.split_name))
        if not torch.is_tensor(mapping):
            mapping = torch.as_tensor(mapping, dtype=torch.long)
        mapping = mapping.to(device=raw_targets.device, dtype=torch.long)
        targets_local = mapping.index_select(0, raw_targets.long())
        if (targets_local < 0).any():
            bad = raw_targets[targets_local < 0][:8].detach().cpu().tolist()
            raise ValueError(
                "Found labels outside local-output space for split '{}' (e.g., {}).".format(
                    dataset.split_name,
                    bad,
                )
            )
        local_weights = self._slice_class_weights_to_space(class_ids)
        return class_ids, targets_local, local_weights

    @staticmethod
    def _model_ref(model):
        return model.module if hasattr(model, "module") else model

    ### 2. ==================================result / record==========================================
    def _pick_primary_metric(self, metric_dict):
        """
        Select the early-stop metric according to evaluator task type.
        Returns: (metric_name, metric_value) or (None, None) if unavailable.
        """
        if not isinstance(metric_dict, dict):
            return None, None

        task_type = self.evaluator.task_type
        if task_type == "gzsl":
            candidates = ["gzsl_h", "zsl_unseen", "top1"]
        elif task_type == "zsl":
            candidates = ["dev_unseen", "zsl_unseen", "gzsl_h", "top1"]
        else:
            raise ValueError("Unsupported evaluator.task_type '{}'".format(task_type))

        for key in candidates:
            val = metric_dict.get(key, None)
            if val is None:
                continue
            try:
                return key, float(val)
            except (TypeError, ValueError):
                continue
        return None, None

    def _resolve_eval_metric_key(self, dataset, prefix: str) -> str:
        protocol_mode = str(dataset.protocol_mode).lower()
        split = str(prefix).lower()

        if split == "val_unseen":
            if protocol_mode != "dev":
                raise ValueError("val_unseen is only valid under dev protocol, got '{}'".format(protocol_mode))
            return "dev_unseen"
        if split == "test_seen":
            if protocol_mode != "final_gzsl":
                raise ValueError("test_seen is only valid under final_gzsl protocol, got '{}'".format(protocol_mode))
            return "gzsl_seen"
        if split == "test_unseen":
            if protocol_mode == "final_zsl":
                return "zsl_unseen"
            if protocol_mode == "final_gzsl":
                return "gzsl_unseen"
            raise ValueError("test_unseen is only valid under final_zsl/final_gzsl, got '{}'".format(protocol_mode))
        raise ValueError("Unsupported eval split '{}' for metric-key resolution".format(prefix))

    def _update_gzsl_record_metrics(self, epoch: int, test_unseen_loader, seen_metrics, unseen_metrics):
        if not isinstance(seen_metrics, dict) or not isinstance(unseen_metrics, dict):
            return
        seen_val = seen_metrics.get("gzsl_seen", None)
        unseen_val = unseen_metrics.get("gzsl_unseen", None)
        if seen_val is None or unseen_val is None:
            return
        seen_val = float(seen_val)
        unseen_val = float(unseen_val)
        curr_h = 0.0 if (seen_val + unseen_val) <= 0 else float(2.0 * seen_val * unseen_val / (seen_val + unseen_val + 1e-8))
        self.best_gzsl_seen_recorded = max(self.best_gzsl_seen_recorded, seen_val)
        self.best_gzsl_unseen_recorded = max(self.best_gzsl_unseen_recorded, unseen_val)
        recorded_h = 0.0
        if (self.best_gzsl_seen_recorded + self.best_gzsl_unseen_recorded) > 0:
            recorded_h = float(
                2.0 * self.best_gzsl_seen_recorded * self.best_gzsl_unseen_recorded
                / (self.best_gzsl_seen_recorded + self.best_gzsl_unseen_recorded + 1e-8)
            )
        ds = test_unseen_loader.dataset
        eval_name = "test_gzsl_" + ds.name
        combined = {
            "gzsl_seen": seen_val,
            "gzsl_unseen": unseen_val,
            "gzsl_h": curr_h,
            "best_gzsl_seen_recorded": float(self.best_gzsl_seen_recorded),
            "best_gzsl_unseen_recorded": float(self.best_gzsl_unseen_recorded),
            "gzsl_h_recorded": recorded_h,
        }
        self.evaluator.update_result("classification", {eval_name: combined})
        logger.info(
            "[gzsl-record] epoch=%d gzsl_seen=%.4f gzsl_unseen=%.4f gzsl_h=%.4f "
            "best_seen_recorded=%.4f best_unseen_recorded=%.4f gzsl_h_recorded=%.4f",
            int(epoch + 1),
            seen_val,
            unseen_val,
            curr_h,
            float(self.best_gzsl_seen_recorded),
            float(self.best_gzsl_unseen_recorded),
            recorded_h,
        )

    ### =======================================3. debug=============================================
    @staticmethod
    def _get_output_type_name(outputs):
        if torch.is_tensor(outputs):
            return "tensor"
        if isinstance(outputs, tuple):
            return "tuple"
        if isinstance(outputs, list):
            return "list"
        if isinstance(outputs, dict):
            return "dict"
        return type(outputs).__name__

    @staticmethod
    def _shape_or_none(t):
        """obout Print log"""
        if torch.is_tensor(t):
            return tuple(t.shape)
        return None

    @staticmethod
    def _extract_logits_and_aux_for_debug(pred_logits):
        aux = None
        logits = pred_logits
        if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
            logits = pred_logits[0]
            if len(pred_logits) > 1 and isinstance(pred_logits[1], dict):
                aux = pred_logits[1]
        elif isinstance(pred_logits, dict):
            logits = pred_logits.get("logits", pred_logits)
            aux = pred_logits
        return logits, aux

    def _make_trace_id(self):
        """Assign a 'id' to the current trace:forward/current batch of data"""
        return "stage={}|rank={}|epoch={}|iter={}|gstep={}".format(
            self._trace_stage,
            self._trace_rank,
            self._trace_epoch,
            self._trace_iter,
            self._trace_global_step,
        )

    def _set_model_trace_context(self, trace_id):
        """print trace id in models context"""
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        setattr(model_ref, "_debug_trace_id", trace_id)
        setattr(model_ref, "_debug_shapes", bool(self.debug_shapes))
        enc = model_ref.enc
        setattr(enc, "_debug_trace_id", trace_id)
        setattr(enc, "_debug_shapes", bool(self.debug_shapes))
        transformer = enc.transformer
        setattr(transformer, "_debug_trace_id", trace_id)
        setattr(transformer, "_debug_shapes", bool(self.debug_shapes))
        setattr(self.model, "_debug_trace_id", trace_id)
        setattr(self.model, "_debug_shapes", bool(self.debug_shapes))

    def _named_params(self):
        """Cache the result of self.model.named_parameters() into a dictionary"""
        if self._named_param_cache is None:
            self._named_param_cache = dict(self.model.named_parameters())
        return self._named_param_cache

    def _find_param_by_name_contains(self, candidates):
        named = self._named_params()
        for needle in candidates:
            for name, param in named.items():
                if needle in name:
                    return name, param
        return None, None

    def _collect_debug_param_refs(self):
        """
        1. normal head
        2. posterior mu / logvar
        3. prompt generator
        4. r_similarity_head  visual / semantic proj
        5. semantic side branch
        6. prompt_update_layers """
        refs = {}
        refs["head.last_layer.weight"] = self._find_param_by_name_contains(
            ["head.last_layer.weight"]
        )
        refs["head.last_layer.bias"] = self._find_param_by_name_contains(
            ["head.last_layer.bias"]
        )
        refs["posterior.mu_head"] = self._find_param_by_name_contains(
            ["prompt_init_provider.posterior.mu_head.0.weight"]
        )
        refs["posterior.logvar_head"] = self._find_param_by_name_contains(
            ["prompt_init_provider.posterior.logvar_head.0.weight"]
        )
        refs["prompt_generator"] = self._find_param_by_name_contains(
            ["prompt_init_provider.prompt_generator.mlp.3.weight", "prompt_init_provider.prompt_generator"]
        )
        refs["r_head.visual_proj"] = self._find_param_by_name_contains(
            ["r_similarity_head.visual_proj.weight", "r_similarity_head.visual_proj"]
        )
        refs["r_head.semantic_proj"] = self._find_param_by_name_contains(
            ["r_similarity_head.semantic_proj.weight", "r_similarity_head.semantic_proj"]
        )
        refs["r_head.logit_scale"] = self._find_param_by_name_contains(
            ["r_similarity_head.logit_scale"]
        )
        # Track layer-wise prompt evolution with first/middle/last layers.
        prompt_layers = [
            name for name in self._named_params().keys()
            if "prompt_update_layers." in name and name.endswith(".weight")
        ]
        layer_ids = sorted(
            set(
                int(name.split("prompt_update_layers.")[1].split(".")[0])
                for name in prompt_layers
                if "prompt_update_layers." in name
            )
        )
        if layer_ids:
            first_id = layer_ids[0]
            mid_id = layer_ids[len(layer_ids) // 2]
            last_id = layer_ids[-1]
            refs["prompt_update.first"] = self._find_param_by_name_contains(
                [f"prompt_update_layers.{first_id}.weight"]
            )
            refs["prompt_update.mid"] = self._find_param_by_name_contains(
                [f"prompt_update_layers.{mid_id}.weight"]
            )
            refs["prompt_update.last"] = self._find_param_by_name_contains(
                [f"prompt_update_layers.{last_id}.weight"]
            )
        return refs

    def _log_semantic_param_names_once(self):
        if self._debug_semantic_param_names_logged:
            return
        names = [n for n, _ in self.model.named_parameters() if "semantic_token_projector" in n]
        logger.info(
            "[trace] semantic_token_projector param names (%d): %s",
            len(names),
            names if len(names) <= 40 else names[:40] + ["..."],
        )
        self._debug_semantic_param_names_logged = True

    def _semantic_source_for_stage(self, is_train: bool) -> str:
        semantic_cfg = self.cfg.MODEL.SEMANTIC_TOKENS
        source = semantic_cfg.TRAIN_SOURCE if is_train else semantic_cfg.EVAL_SOURCE
        source = str(source).lower()
        # BEGIN SEMANTIC_ABLATION_EXPERIMENT
        valid_sources = {"label", "class_mean", "none", "random_fixed", "label_shuffle", "learned_token"}
        # END SEMANTIC_ABLATION_EXPERIMENT
        if source not in valid_sources:
            raise ValueError(
                "Unsupported MODEL.SEMANTIC_TOKENS.{}_SOURCE='{}'; expected one of {}.".format(
                    "TRAIN" if is_train else "EVAL",
                    source,
                    "/".join(sorted(valid_sources)),
                )
            )
        return source

    # BEGIN SEMANTIC_ABLATION_EXPERIMENT
    def _class_mean_semantics(self, dataset, batch_size: int) -> torch.Tensor:
        if dataset is None or not hasattr(dataset, "class_attributes") or dataset.class_attributes is None:
            raise ValueError("MODEL.SEMANTIC_TOKENS source='class_mean' requires dataset.class_attributes.")
        class_attributes = dataset.class_attributes
        if not torch.is_tensor(class_attributes):
            class_attributes = torch.from_numpy(class_attributes)
        mean_attr = class_attributes.to(self.device, non_blocking=True).float().mean(dim=0)
        return mean_attr.unsqueeze(0).expand(int(batch_size), -1)

    def _label_semantics(self, attributes) -> torch.Tensor:
        if attributes is None:
            raise ValueError("MODEL.SEMANTIC_TOKENS source='label' requires batch['attribute'].")
        if not torch.is_tensor(attributes):
            attributes = torch.from_numpy(attributes)
        return attributes.to(self.device, non_blocking=True).float()
    # END SEMANTIC_ABLATION_EXPERIMENT

    def _prepare_semantics_for_stage(self, attributes, dataset, batch_size: int, is_train: bool):
        if not bool(self.cfg.MODEL.SEMANTIC_TOKENS.ENABLE):
            return None
        source = self._semantic_source_for_stage(is_train)
        if source == "none":
            return None
        if source == "label":
            return self._label_semantics(attributes)
        if source == "class_mean":
            return self._class_mean_semantics(dataset, batch_size)
        # BEGIN SEMANTIC_ABLATION_EXPERIMENT
        if source == "random_fixed":
            return self._semantic_random_fixed.to(self.device, non_blocking=True).float().unsqueeze(0).expand(int(batch_size), -1)
        if source == "label_shuffle":
            label_semantics = self._label_semantics(attributes)
            if int(label_semantics.shape[0]) != int(batch_size):
                raise ValueError(
                    f"label_shuffle requires attributes batch size {int(batch_size)}, got {int(label_semantics.shape[0])}."
                )
            if int(batch_size) < 2:
                raise ValueError("label_shuffle requires batch_size >= 2.")
            perm = torch.randperm(int(batch_size), device=label_semantics.device)
            if bool(torch.equal(perm, torch.arange(int(batch_size), device=label_semantics.device))):
                perm = torch.roll(perm, shifts=1, dims=0)
            return label_semantics.index_select(0, perm)
        if source == "learned_token":
            return torch.empty((int(batch_size), 0), device=self.device, dtype=torch.float32)
        # END SEMANTIC_ABLATION_EXPERIMENT
        raise ValueError(f"Unsupported MODEL.SEMANTIC_TOKENS source='{source}'.")

    def _log_optimizer_param_groups(self):
        named = self._named_params()
        id2name = {id(p): n for n, p in named.items()}
        logger.info("Optimizer param_groups summary:")
        for idx, group in enumerate(self.optimizer.param_groups):
            params = group.get("params", [])
            names = [id2name.get(id(p), "<unnamed>") for p in params]
            logger.info(
                "  group[%d]: lr=%s wd=%s params=%d (head=%s, r_head=%s, prompt_dist=%s, prompt_update=%s, semantic=%s)",
                idx,
                group.get("lr", None),
                group.get("weight_decay", None),
                len(params),
                any("head." in n for n in names),
                any("r_similarity_head" in n for n in names),
                any("prompt_init_provider" in n for n in names),
                any("prompt_update_layers" in n for n in names),
                any(("semantic_token_projector" in n) or ("prototype_proj" in n) for n in names),
            )

    def _log_batch_stats_once(self, logits, targets):
        """ - Mean/std/min/max of logits
            - Range and number of unique values of targets
            - Softmax entropy
            - Scale information of r_head"""
        if self._debug_batch_stats_logged or logits is None:
            return
        if not torch.is_tensor(logits):
            return
        with torch.no_grad():
            logger.info(
                "[debug] logits stats: shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
                tuple(logits.shape),
                float(logits.mean().item()),
                float(logits.std().item()),
                float(logits.min().item()),
                float(logits.max().item()),
            )
            logger.info(
                "[debug] targets stats: min=%d max=%d unique_count=%d",
                int(targets.min().item()),
                int(targets.max().item()),
                int(targets.unique().numel()),
            )
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
            max_entropy = float(np.log(max(int(logits.shape[-1]), 1)))
            logger.info(
                "[debug] softmax entropy: mean=%.6f max=%.6f classes=%d",
                float(entropy.item()),
                max_entropy,
                int(logits.shape[-1]),
            )
            model_ref = self._model_ref(self.model)
            r_head = model_ref.r_similarity_head
            if r_head is not None:
                fixed_scale = float(r_head.fixed_logit_scale)
                learnable_scale = None
                if r_head.logit_scale is not None:
                    learnable_scale = float(r_head.logit_scale.exp().item())
                effective_scale = fixed_scale if fixed_scale > 0 else learnable_scale
                logger.info(
                    "[debug] r_head scale: fixed=%.6f learnable_exp=%s effective=%.6f mode=%s",
                    fixed_scale,
                    "{:.6f}".format(learnable_scale) if learnable_scale is not None else "None",
                    float(effective_scale) if effective_scale is not None else float("nan"),
                    "fixed" if fixed_scale > 0 else "learnable",
                )
        self._debug_batch_stats_logged = True

    def _log_grad_norms_once(self, refs):
        """At the beginning, print the gradient norm of key parameters only once for self-checking"""
        if self._debug_grad_logged:
            return
        for alias, (name, param) in refs.items():
            if param is None:
                logger.info("[debug] grad %-24s : MISSING", alias)
                continue
            grad = param.grad
            if grad is None:
                logger.info("[debug] grad %-24s : None (%s)", alias, name)
            else:
                logger.info(
                    "[debug] grad %-24s : %.6e (%s)",
                    alias,
                    float(grad.norm().item()),
                    name,
                )
        self._debug_grad_logged = True

    def _capture_param_norms(self, refs):
        """Record the gradient norms of key parameters for comparison before and after"""
        norms = {}
        for alias, (_, param) in refs.items():
            if param is None:
                norms[alias] = None
            else:
                norms[alias] = float(param.data.norm().item())
        return norms

    def _log_update_once(self, before_norms, after_norms):
        """comparison"""
        if self._debug_step_logged:
            return
        for alias in before_norms.keys():
            b = before_norms[alias]
            a = after_norms[alias]
            if b is None or a is None:
                logger.info("[debug] step %-24s : unavailable", alias)
                continue
            logger.info(
                "[debug] step %-24s : before=%.6e after=%.6e delta=%.6e",
                alias,
                b,
                a,
                a - b,
            )
        self._debug_step_logged = True

    @staticmethod
    def _tensor_stats(t: torch.Tensor):
        """
        Return a dictionary of basic statistics of a tensor:
            - shape
            - mean
            - std
            - min
            - max"""
        if t is None or (not torch.is_tensor(t)):
            return None
        return {
            "shape": tuple(t.shape),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    def _capture_train_debug(self, loss_outputs, raw_outputs, loss_targets):
        """
        记录“当前训练 batch”的关键调试信息。

        这个函数不参与训练逻辑本身，它的职责是：
        1. 从训练时真正用于算 loss 的输出中提取 logits（ce_logits）
        2. 从模型原始输出中提取 logits（raw_logits）
        3. 统计：
            - logits 的形状/均值/方差/极值
            - softmax 熵
            - seen-only top1
            - CE logits 与 raw logits 是否是同一份张量
            - r_similarity_head 的一些评分统计
            - 当前 loss 的辅助统计
        4. 把这些信息放到 self._last_train_debug 里，
           供训练日志、overfit debug、NaN debug 等地方复用"""
        ce_logits = self._extract_logits(loss_outputs)
        raw_logits = self._extract_logits(raw_outputs)
        if not torch.is_tensor(ce_logits):
            return

        with torch.no_grad():
            self._last_ce_logits = ce_logits.detach()
            self._last_raw_logits = raw_logits.detach() if torch.is_tensor(raw_logits) else None
            ce_probs = torch.softmax(ce_logits.float(), dim=-1)
            ce_entropy = -(ce_probs * torch.log(ce_probs.clamp_min(1e-12))).sum(dim=-1).mean()
            top1 = (ce_logits.argmax(dim=1) == loss_targets).float().mean()

            same_tensor = (
                torch.is_tensor(raw_logits)
                and ce_logits.data_ptr() == raw_logits.data_ptr()
                and ce_logits.shape == raw_logits.shape
            )

            raw_entropy = None
            if torch.is_tensor(raw_logits):
                raw_probs = torch.softmax(raw_logits.float(), dim=-1)
                raw_entropy = float(
                    (-(raw_probs * torch.log(raw_probs.clamp_min(1e-12))).sum(dim=-1).mean()).item()
                )

            self._last_train_debug = {
                "ce_logits_stats": self._tensor_stats(ce_logits),
                "raw_logits_stats": self._tensor_stats(raw_logits) if torch.is_tensor(raw_logits) else None,
                "ce_entropy": float(ce_entropy.item()),
                "raw_entropy": raw_entropy,
                "entropy_from_ce_logits": True,
                "entropy_logits_is_ce_tensor": True,
                "ce_vs_raw_same_tensor": bool(same_tensor),
                "seen_only_top1": float(top1.item()),
                "ce_classes": int(ce_logits.shape[-1]),
            }
            model_ref = self._model_ref(self.model)
            r_head = model_ref.r_similarity_head
            if r_head is not None:
                fixed_scale = float(r_head.fixed_logit_scale)
                self._last_train_debug["whether_fixed_logit_scale"] = bool(fixed_scale > 0)
                scale_t = r_head._loss_last_logit_scale
                if torch.is_tensor(scale_t):
                    self._last_train_debug["effective_logit_scale"] = float(scale_t.detach().mean().item())
            loss_stats = self.cls_criterion._last_loss_stats
            if isinstance(loss_stats, dict) and len(loss_stats) > 0:
                self._last_train_debug.update(loss_stats)

    ##=========================== 4. affinity helpers=========================
    @staticmethod
    def _infer_grid(num_patches: int) -> int:
        g = int(round(math.sqrt(max(1, int(num_patches)))))
        return max(1, g)

    def _extract_alignment_aux(self, affinities):
        """
        从逐层 affinity 列表中抽取供 loss / debug 使用的辅助矩阵。

        约定：
        - aff_* : raw affinity / raw score

        语义 token 现在位于 ViT 主序列末尾；这里仅读取主干 Q/K 监测矩阵。
        """
        if affinities is None:
            return None

        aff_qpqv = {}
        aff_kpkv = {}
        aff_qpkv = {}
        aff_qskv = {}
        aff_qvks = {}
        aff_qskp = {}
        aff_qpks = {}

        raw_qpqv_shape = None
        raw_kpkv_shape = None
        raw_qpkv_shape = None
        raw_qskv_shape = None
        raw_qvks_shape = None
        raw_qskp_shape = None
        raw_qpks_shape = None

        for idx, affinity in enumerate(affinities):
            if not isinstance(affinity, dict):
                continue

            qpqv = affinity.get("QpQv_raw")
            if qpqv is not None:
                if raw_qpqv_shape is None and torch.is_tensor(qpqv):
                    raw_qpqv_shape = tuple(qpqv.shape)
                if qpqv.dim() == 4:
                    aff_qpqv[idx] = qpqv.mean(dim=1)
                elif qpqv.dim() == 3:
                    aff_qpqv[idx] = qpqv

            kpkv = affinity.get("KpKv_raw")
            if kpkv is not None:
                if raw_kpkv_shape is None and torch.is_tensor(kpkv):
                    raw_kpkv_shape = tuple(kpkv.shape)
                if kpkv.dim() == 4:
                    aff_kpkv[idx] = kpkv.mean(dim=1)
                elif kpkv.dim() == 3:
                    aff_kpkv[idx] = kpkv

            qpkv = affinity.get("QpKv_raw")
            if qpkv is not None:
                if raw_qpkv_shape is None and torch.is_tensor(qpkv):
                    raw_qpkv_shape = tuple(qpkv.shape)
                if qpkv.dim() == 4:
                    aff_qpkv[idx] = qpkv.mean(dim=1)
                elif qpkv.dim() == 3:
                    aff_qpkv[idx] = qpkv

            qskv = affinity.get("QsKv_raw")
            if qskv is not None:
                if raw_qskv_shape is None and torch.is_tensor(qskv):
                    raw_qskv_shape = tuple(qskv.shape)
                if qskv.dim() == 4:
                    aff_qskv[idx] = qskv.mean(dim=1)
                elif qskv.dim() == 3:
                    aff_qskv[idx] = qskv

            qvks = affinity.get("QvKs_raw")
            if qvks is not None:
                if raw_qvks_shape is None and torch.is_tensor(qvks):
                    raw_qvks_shape = tuple(qvks.shape)
                if qvks.dim() == 4:
                    aff_qvks[idx] = qvks.mean(dim=1)
                elif qvks.dim() == 3:
                    aff_qvks[idx] = qvks

            qskp = affinity.get("QsKp_raw")
            if qskp is not None:
                if raw_qskp_shape is None and torch.is_tensor(qskp):
                    raw_qskp_shape = tuple(qskp.shape)
                if qskp.dim() == 4:
                    aff_qskp[idx] = qskp.mean(dim=1)
                elif qskp.dim() == 3:
                    aff_qskp[idx] = qskp

            qpks = affinity.get("QpKs_raw")
            if qpks is not None:
                if raw_qpks_shape is None and torch.is_tensor(qpks):
                    raw_qpks_shape = tuple(qpks.shape)
                if qpks.dim() == 4:
                    aff_qpks[idx] = qpks.mean(dim=1)
                elif qpks.dim() == 3:
                    aff_qpks[idx] = qpks

        if not aff_qpqv and not aff_kpkv and not aff_qpkv and not aff_qskv and not aff_qvks and not aff_qskp and not aff_qpks:
            return None

        out = {}
        if aff_qpqv:
            out["aff_qpqv"] = aff_qpqv
        if aff_kpkv:
            out["aff_kpkv"] = aff_kpkv
        if aff_qpkv:
            out["aff_qpkv"] = aff_qpkv
        if aff_qskv:
            out["aff_qskv"] = aff_qskv
        if aff_qvks:
            out["aff_qvks"] = aff_qvks
        if aff_qskp:
            out["aff_qskp"] = aff_qskp
        if aff_qpks:
            out["aff_qpks"] = aff_qpks

        if self.debug_shapes and (not self._shape_debug_aux_logged):
            layer_keys = set()
            for d in (aff_qpqv, aff_kpkv, aff_qpkv, aff_qskv, aff_qvks, aff_qskp, aff_qpks):
                layer_keys.update(d.keys())
            sample_layer = sorted(layer_keys)[0] if len(layer_keys) > 0 else None
            qpqv_after = tuple(aff_qpqv[sample_layer].shape) if sample_layer is not None and sample_layer in aff_qpqv else None
            kpkv_after = tuple(aff_kpkv[sample_layer].shape) if sample_layer is not None and sample_layer in aff_kpkv else None
            qpkv_after = tuple(aff_qpkv[sample_layer].shape) if sample_layer is not None and sample_layer in aff_qpkv else None
            qskv_after = tuple(aff_qskv[sample_layer].shape) if sample_layer is not None and sample_layer in aff_qskv else None
            qvks_after = tuple(aff_qvks[sample_layer].shape) if sample_layer is not None and sample_layer in aff_qvks else None
            qskp_after = tuple(aff_qskp[sample_layer].shape) if sample_layer is not None and sample_layer in aff_qskp else None
            qpks_after = tuple(aff_qpks[sample_layer].shape) if sample_layer is not None and sample_layer in aff_qpks else None
            print(
                "[SHAPE-DEBUG] trainer._extract_alignment_aux raw QpQv={} KpKv={} QpKv={} QsKv={} QvKs={} QsKp={} QpKs={} "
                "head_avg aff_qpqv={} aff_kpkv={} aff_qpkv={} aff_qskv={} aff_qvks={} aff_qskp={} aff_qpks={} layer={}".format(
                    raw_qpqv_shape,
                    raw_kpkv_shape,
                    raw_qpkv_shape,
                    raw_qskv_shape,
                    raw_qvks_shape,
                    raw_qskp_shape,
                    raw_qpks_shape,
                    qpqv_after,
                    kpkv_after,
                    qpkv_after,
                    qskv_after,
                    qvks_after,
                    qskp_after,
                    qpks_after,
                    sample_layer,
                )
            )
            self._shape_debug_aux_logged = True

        return out

    ## =======================================5. vis================================================
    @staticmethod
    def _parse_vis_epoch_list(v) -> list:
        """init"""
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return sorted(set([x for x in out if x > 0]))
        if isinstance(v, str):
            s = v.strip()
            if len(s) == 0:
                return []
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple)):
                out = []
                for x in parsed:
                    try:
                        out.append(int(x))
                    except Exception:
                        continue
                return sorted(set([x for x in out if x > 0]))
        return []

    def _vis_split_enabled(self, split: str) -> bool:
        if not self.vis_enable:
            return False
        epoch_1based = int(self._trace_epoch + 1)
        if self.vis_final_only and epoch_1based != int(self.cfg.SOLVER.TOTAL_EPOCH):
            return False
        if len(self.vis_epoch_list) > 0:
            if epoch_1based not in self.vis_epoch_list:
                return False
        else:
            if (epoch_1based % self.vis_every_epoch) != 0:
                return False
        allowed = {str(x).lower() for x in self.vis_splits}
        split = str(split).lower()
        return split in allowed

    def _vis_init_epoch(self, split: str):
        self._vis_processed = 0
        self._vis_correct_processed = 0
        self._vis_wrong_processed = 0

    def _vis_should_collect_case(self, is_correct: bool) -> bool:
        if self._vis_processed >= self.vis_max_samples:
            return False
        if is_correct:
            return self._vis_correct_processed < self.vis_correct_samples
        return self._vis_wrong_processed < self.vis_wrong_samples

    def _vis_finalize_case(self, is_correct: bool) -> None:
        self._vis_processed += 1
        if is_correct:
            self._vis_correct_processed += 1
        else:
            self._vis_wrong_processed += 1

    def _vis_extract_cls_patch_layer_maps(self, local_idx: int) -> list:
        if not isinstance(self._last_attn_weights, list) or len(self._last_attn_weights) == 0:
            return []
        prompt_len = int(self.cfg.MODEL.PROMPT.NUM_TOKENS) if bool(self.cfg.MODEL.PROMPT.ENABLE) else 0
        semantic_len = int(self._last_semantic_length)
        out = []
        for li, weights in enumerate(self._last_attn_weights):
            if (not torch.is_tensor(weights)) or weights.dim() != 4 or local_idx >= weights.shape[0]:
                continue
            sample = weights[local_idx].detach().cpu()
            mean_attn = sample.float().mean(dim=0)
            start_patch = 1 + prompt_len
            end_patch = int(mean_attn.shape[-1] - semantic_len)
            n_patch = int(end_patch - start_patch)
            if n_patch <= 0:
                continue
            g = self._infer_grid(n_patch)
            out.append({
                "layer": int(li),
                "grid_map": mean_attn[0, start_patch:end_patch].view(g, g).numpy(),
                "head_raw": sample[:, 0, start_patch:end_patch].numpy(),
            })
        return out

    def _vis_extract_prompt_visual_qk_layer_maps(self, local_idx: int, affinities: list, raw_key: str, vis_key: str) -> list:
        if not isinstance(affinities, list) or len(affinities) == 0:
            return []
        out = []
        for li, aff in enumerate(affinities):
            if not isinstance(aff, dict):
                continue
            raw = aff.get(raw_key, None)
            vis = aff.get(vis_key, None)
            if (not torch.is_tensor(raw)) or raw.dim() != 4 or local_idx >= raw.shape[0]:
                continue
            if (not torch.is_tensor(vis)) or vis.dim() != 4 or local_idx >= vis.shape[0]:
                continue
            raw_sample = raw[local_idx].detach().cpu()   # [H,P,V]
            vis_sample = vis[local_idx].detach().cpu()   # [H,P,V]
            mean_map = vis_sample.float().mean(dim=0).mean(dim=0)  # [V]
            n_patch = int(mean_map.shape[-1])
            if n_patch <= 0:
                continue
            g = self._infer_grid(n_patch)
            out.append({
                "layer": int(li),
                "grid_map": mean_map.view(g, g).numpy(),
                "head_raw": raw_sample.numpy(),
                "head_vis": vis_sample.numpy(),
            })
        return out

    def _vis_extract_prompt_visual_matrix_maps(self, local_idx: int, affinities: list, raw_key: str, vis_key: str) -> list:
        if not isinstance(affinities, list) or len(affinities) == 0:
            return []
        out = []
        for li, aff in enumerate(affinities):
            if not isinstance(aff, dict):
                continue
            raw = aff.get(raw_key, None)
            vis = aff.get(vis_key, None)
            if (not torch.is_tensor(raw)) or raw.dim() != 4 or local_idx >= raw.shape[0]:
                continue
            if (not torch.is_tensor(vis)) or vis.dim() != 4 or local_idx >= vis.shape[0]:
                continue
            raw_sample = raw[local_idx].detach().cpu()   # [H,P,V]
            vis_sample = vis[local_idx].detach().cpu()   # [H,P,V]
            out.append({
                "layer": int(li),
                "matrix_raw": raw_sample.float().mean(dim=0).numpy(),
                "matrix_vis": vis_sample.float().mean(dim=0).numpy(),
                "head_raw": raw_sample.numpy(),
                "head_vis": vis_sample.numpy(),
            })
        return out

    def _vis_extract_affinity_to_visual_maps(self, local_idx: int, affinities: list, raw_key: str, vis_key: str, visual_axis: int) -> list:
        if not isinstance(affinities, list) or len(affinities) == 0:
            return []
        out = []
        for li, aff in enumerate(affinities):
            if not isinstance(aff, dict):
                continue
            raw = aff.get(raw_key, None)
            vis = aff.get(vis_key, None)
            if (not torch.is_tensor(raw)) or raw.dim() != 4 or local_idx >= raw.shape[0]:
                continue
            if (not torch.is_tensor(vis)) or vis.dim() != 4 or local_idx >= vis.shape[0]:
                continue
            raw_sample = raw[local_idx].detach().cpu()
            vis_sample = vis[local_idx].detach().cpu()
            if visual_axis == 2:
                mean_map = vis_sample.float().mean(dim=0).mean(dim=0)
            elif visual_axis == 1:
                mean_map = vis_sample.float().mean(dim=0).mean(dim=-1)
            else:
                raise ValueError(f"Unsupported visual_axis={visual_axis}")
            n_patch = int(mean_map.shape[-1])
            if n_patch <= 0:
                continue
            g = self._infer_grid(n_patch)
            out.append({
                "layer": int(li),
                "grid_map": mean_map.view(g, g).numpy(),
                "head_raw": raw_sample.numpy(),
                "head_vis": vis_sample.numpy(),
            })
        return out

    def _vis_extract_prompt_response_maps(self, local_idx: int, affinities: list, raw_key: str, vis_key: str, prompt_axis: int) -> list:
        if not isinstance(affinities, list) or len(affinities) == 0:
            return []
        out = []
        for li, aff in enumerate(affinities):
            if not isinstance(aff, dict):
                continue
            raw = aff.get(raw_key, None)
            vis = aff.get(vis_key, None)
            if (not torch.is_tensor(raw)) or raw.dim() != 4 or local_idx >= raw.shape[0]:
                continue
            if (not torch.is_tensor(vis)) or vis.dim() != 4 or local_idx >= vis.shape[0]:
                continue
            raw_sample = raw[local_idx].detach().cpu()
            vis_sample = vis[local_idx].detach().cpu()
            if prompt_axis == 2:
                prompt_raw = raw_sample.float().mean(dim=0).mean(dim=0)
                prompt_vis = vis_sample.float().mean(dim=0).mean(dim=0)
            elif prompt_axis == 1:
                prompt_raw = raw_sample.float().mean(dim=0).mean(dim=-1)
                prompt_vis = vis_sample.float().mean(dim=0).mean(dim=-1)
            else:
                raise ValueError(f"Unsupported prompt_axis={prompt_axis}")
            out.append({
                "layer": int(li),
                "prompt_raw_mean": prompt_raw.numpy(),
                "prompt_vis_mean": prompt_vis.numpy(),
                "head_raw": raw_sample.numpy(),
                "head_vis": vis_sample.numpy(),
            })
        return out

    def _vis_save_layer_panel(self, path: str, image_u8: np.ndarray, layer_maps: list, title_prefix: str) -> None:
        if (not self.vis_save_images) or len(layer_maps) == 0:
            return
        images = []
        titles = []
        h, w = image_u8.shape[:2]
        for item in layer_maps:
            heat_up = resize_map_torch(item["grid_map"], (h, w))
            images.append(overlay_heatmap(image_u8, heat_up))
            titles.append(f"{title_prefix} L{int(item['layer']):02d}")
        save_panel(path, images, titles=titles, ncols=min(4, len(images)))

    def _vis_save_layer_raw(self, path: str, layer_maps: list) -> None:
        if (not self.vis_save_raw) or len(layer_maps) == 0:
            return
        ensure_dir(os.path.dirname(path))
        arrays = {}
        for item in layer_maps:
            li = int(item["layer"])
            arrays[f"layer_{li:02d}_grid"] = np.asarray(item["grid_map"], dtype=np.float32)
            arrays[f"layer_{li:02d}_heads"] = np.asarray(item["head_raw"], dtype=np.float32)
            if "head_vis" in item:
                arrays[f"layer_{li:02d}_heads_vis"] = np.asarray(item["head_vis"], dtype=np.float32)
        np.savez_compressed(as_long_path(path), **arrays)

    def _vis_save_prompt_matrix(self, path: str, prompt_maps: list, use_vis: bool) -> None:
        if (not self.vis_save_images) or len(prompt_maps) == 0:
            return
        ensure_dir(os.path.dirname(path))
        layers = [int(item["layer"]) for item in prompt_maps]
        values = []
        for item in prompt_maps:
            key = "prompt_vis_mean" if use_vis else "prompt_raw_mean"
            values.append(np.asarray(item[key], dtype=np.float32))
        matrix = np.stack(values, axis=0)
        fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.4), max(3, matrix.shape[0] * 0.45)))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xlabel("Prompt Index")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{li:02d}" for li in layers])
        ax.set_title("semantic/prompt affinity mean" + (" (vis)" if use_vis else " (raw mean)"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(as_long_path(path), dpi=200)
        plt.close(fig)

    def _vis_save_overlay_group_panel(self, path: str, image_u8: np.ndarray, groups: list) -> None:
        if (not self.vis_save_images) or len(groups) == 0:
            return
        valid_groups = [(name, maps) for name, maps in groups if len(maps) > 0]
        if len(valid_groups) == 0:
            return
        max_cols = max(len(maps) for _, maps in valid_groups)
        nrows = len(valid_groups)
        h, w = image_u8.shape[:2]
        fig, axes = plt.subplots(nrows, max_cols, figsize=(max(4, max_cols * 3.0), max(3, nrows * 2.8)))
        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)
        if max_cols == 1:
            axes = np.expand_dims(axes, axis=1)
        for r, (name, maps) in enumerate(valid_groups):
            for c in range(max_cols):
                ax = axes[r, c]
                ax.axis("off")
                if c >= len(maps):
                    continue
                item = maps[c]
                heat_up = resize_map_torch(item["grid_map"], (h, w))
                ax.imshow(overlay_heatmap(image_u8, heat_up))
                ax.set_title(f"{name} L{int(item['layer']):02d}", fontsize=9)
        fig.tight_layout()
        ensure_dir(os.path.dirname(path))
        fig.savefig(as_long_path(path), dpi=200)
        plt.close(fig)

    def _vis_save_matrix_group_panel(self, path: str, groups: list) -> None:
        if (not self.vis_save_images) or len(groups) == 0:
            return
        valid_groups = [(name, maps) for name, maps in groups if len(maps) > 0]
        if len(valid_groups) == 0:
            return
        max_cols = max(len(maps) for _, maps in valid_groups)
        nrows = len(valid_groups)
        fig, axes = plt.subplots(nrows, max_cols, figsize=(max(4, max_cols * 3.0), max(3, nrows * 2.8)))
        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)
        if max_cols == 1:
            axes = np.expand_dims(axes, axis=1)
        for r, (name, maps) in enumerate(valid_groups):
            for c in range(max_cols):
                ax = axes[r, c]
                ax.axis("off")
                if c >= len(maps):
                    continue
                item = maps[c]
                ax.imshow(item["matrix_vis"], aspect="auto", cmap="viridis")
                ax.set_title(f"{name} L{int(item['layer']):02d}", fontsize=9)
        fig.tight_layout()
        ensure_dir(os.path.dirname(path))
        fig.savefig(as_long_path(path), dpi=200)
        plt.close(fig)

    def _vis_save_overlay_group_raw(self, path: str, groups: list) -> None:
        if (not self.vis_save_raw) or len(groups) == 0:
            return
        arrays = {}
        for name, maps in groups:
            for item in maps:
                li = int(item["layer"])
                prefix = f"{name}_layer_{li:02d}"
                arrays[f"{prefix}_grid"] = np.asarray(item["grid_map"], dtype=np.float32)
                arrays[f"{prefix}_heads_raw"] = np.asarray(item["head_raw"], dtype=np.float32)
                if "head_vis" in item:
                    arrays[f"{prefix}_heads_vis"] = np.asarray(item["head_vis"], dtype=np.float32)
        if len(arrays) == 0:
            return
        ensure_dir(os.path.dirname(path))
        np.savez_compressed(as_long_path(path), **arrays)

    def _vis_save_matrix_group_raw(self, path: str, groups: list) -> None:
        if (not self.vis_save_raw) or len(groups) == 0:
            return
        arrays = {}
        for name, maps in groups:
            for item in maps:
                li = int(item["layer"])
                prefix = f"{name}_layer_{li:02d}"
                arrays[f"{prefix}_matrix_raw"] = np.asarray(item["matrix_raw"], dtype=np.float32)
                arrays[f"{prefix}_matrix_vis"] = np.asarray(item["matrix_vis"], dtype=np.float32)
                arrays[f"{prefix}_heads_raw"] = np.asarray(item["head_raw"], dtype=np.float32)
                arrays[f"{prefix}_heads_vis"] = np.asarray(item["head_vis"], dtype=np.float32)
        if len(arrays) == 0:
            return
        ensure_dir(os.path.dirname(path))
        np.savez_compressed(as_long_path(path), **arrays)

    def _vis_save_prompt_raw(self, path: str, prompt_maps: list) -> None:
        if (not self.vis_save_raw) or len(prompt_maps) == 0:
            return
        ensure_dir(os.path.dirname(path))
        arrays = {}
        for item in prompt_maps:
            li = int(item["layer"])
            arrays[f"layer_{li:02d}_prompt_raw_mean"] = np.asarray(item["prompt_raw_mean"], dtype=np.float32)
            arrays[f"layer_{li:02d}_prompt_vis_mean"] = np.asarray(item["prompt_vis_mean"], dtype=np.float32)
            arrays[f"layer_{li:02d}_prompt_heads_raw"] = np.asarray(item["head_raw"], dtype=np.float32)
            arrays[f"layer_{li:02d}_prompt_heads_vis"] = np.asarray(item["head_vis"], dtype=np.float32)
        np.savez_compressed(as_long_path(path), **arrays)

    # BEGIN SEMANTIC_ABLATION_EXPERIMENT
    def _attention_mass_rows_and_arrays(self, split: str, sample_idx: int, target: int, pred: int, is_correct: bool,
        local_idx: int, attn_weights: list,):
        if not isinstance(attn_weights, list) or len(attn_weights) == 0:
            raise ValueError("attention mass requires per-layer attention weights.")
        prompt_len = int(self.cfg.MODEL.PROMPT.NUM_TOKENS) if bool(self.cfg.MODEL.PROMPT.ENABLE) else 0
        semantic_len = int(self._last_semantic_length)
        if prompt_len <= 0:
            raise ValueError("attention mass prompt-to-S requires MODEL.PROMPT.ENABLE=True and NUM_TOKENS>0.")
        if semantic_len <= 0:
            raise ValueError("attention mass requires active semantic tokens.")

        rows = []
        arrays = {}
        for li, weights in enumerate(attn_weights):
            if (not torch.is_tensor(weights)) or weights.dim() != 4 or local_idx >= weights.shape[0]:
                raise ValueError(f"Invalid attention weights at layer {li}: expected [B,H,T,T].")
            sample = weights[local_idx].detach().cpu().float()
            num_heads = int(sample.shape[0])
            start_patch = 1 + prompt_len
            end_patch = int(sample.shape[-1] - semantic_len)
            if end_patch <= start_patch:
                raise ValueError(f"Layer {li} has no visual patch segment for attention mass.")
            sem_slice = slice(end_patch, end_patch + semantic_len)
            prompt_slice = slice(1, 1 + prompt_len)
            visual_slice = slice(start_patch, end_patch)

            s_to_cls = sample[:, sem_slice, 0].mean(dim=1)
            cls_to_s = sample[:, 0, sem_slice].sum(dim=1)
            v_to_s_tokens = sample[:, visual_slice, sem_slice].sum(dim=-1)
            p_to_s_tokens = sample[:, prompt_slice, sem_slice].sum(dim=-1)
            s_to_v = sample[:, sem_slice, visual_slice].mean(dim=1)
            cls_to_v = sample[:, 0, visual_slice]

            s_centered = s_to_v - s_to_v.mean(dim=1, keepdim=True)
            c_centered = cls_to_v - cls_to_v.mean(dim=1, keepdim=True)
            numerator = (s_centered * c_centered).sum(dim=1)
            denominator = torch.sqrt((s_centered.square().sum(dim=1)) * (c_centered.square().sum(dim=1)))
            corr = torch.where(
                denominator > 0,
                numerator / denominator,
                torch.full_like(numerator, float("nan")),
            )

            arrays[f"layer_{li:02d}_v_to_s_token_mass"] = v_to_s_tokens.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_p_to_s_token_mass"] = p_to_s_tokens.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_s_to_v_attention"] = s_to_v.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_cls_to_v_attention"] = cls_to_v.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_s_to_cls"] = s_to_cls.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_cls_to_s"] = cls_to_s.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_corr_s_cls_to_v"] = corr.numpy().astype(np.float32)

            for hi in range(num_heads):
                rows.append({
                    "sample_id": int(sample_idx),
                    "split": str(split),
                    "epoch": int(self._trace_epoch + 1),
                    "is_correct": int(bool(is_correct)),
                    "y": int(target),
                    "pred": int(pred),
                    "layer": int(li),
                    "head": int(hi),
                    "mass_s_to_cls": float(s_to_cls[hi].item()),
                    "mass_cls_to_s": float(cls_to_s[hi].item()),
                    "mass_v_to_s_mean": float(v_to_s_tokens[hi].mean().item()),
                    "mass_v_to_s_sum": float(v_to_s_tokens[hi].sum().item()),
                    "mass_v_to_s_max": float(v_to_s_tokens[hi].max().item()),
                    "mass_p_to_s_mean": float(p_to_s_tokens[hi].mean().item()),
                    "mass_p_to_s_sum": float(p_to_s_tokens[hi].sum().item()),
                    "mass_p_to_s_max": float(p_to_s_tokens[hi].max().item()),
                    "corr_s_cls_to_v": float(corr[hi].item()),
                })
            finite_corr = corr[torch.isfinite(corr)]
            corr_mean = float(finite_corr.mean().item()) if finite_corr.numel() > 0 else float("nan")
            rows.append({
                "sample_id": int(sample_idx),
                "split": str(split),
                "epoch": int(self._trace_epoch + 1),
                "is_correct": int(bool(is_correct)),
                "y": int(target),
                "pred": int(pred),
                "layer": int(li),
                "head": -1,
                "mass_s_to_cls": float(s_to_cls.mean().item()),
                "mass_cls_to_s": float(cls_to_s.mean().item()),
                "mass_v_to_s_mean": float(v_to_s_tokens.mean().item()),
                "mass_v_to_s_sum": float(v_to_s_tokens.sum(dim=1).mean().item()),
                "mass_v_to_s_max": float(v_to_s_tokens.max(dim=1).values.mean().item()),
                "mass_p_to_s_mean": float(p_to_s_tokens.mean().item()),
                "mass_p_to_s_sum": float(p_to_s_tokens.sum(dim=1).mean().item()),
                "mass_p_to_s_max": float(p_to_s_tokens.max(dim=1).values.mean().item()),
                "corr_s_cls_to_v": corr_mean,
            })
        return rows, arrays

    def _vis_save_attention_mass(self, sample_dir: str, split: str, sample_idx: int, target: int, pred: int,
        is_correct: bool, local_idx: int, attn_weights: list,) -> None:
        rows, arrays = self._attention_mass_rows_and_arrays(
            split=split,
            sample_idx=sample_idx,
            target=target,
            pred=pred,
            is_correct=is_correct,
            local_idx=local_idx,
            attn_weights=attn_weights,
        )
        if self.vis_save_raw:
            np.savez_compressed(as_long_path(os.path.join(sample_dir, "attention_mass_layers.npz")), **arrays)
        csv_path = os.path.join(sample_dir, "attention_mass_layers.csv")
        header = [
            "sample_id", "split", "epoch", "is_correct", "y", "pred", "layer", "head",
            "mass_s_to_cls", "mass_cls_to_s", "mass_v_to_s_mean", "mass_v_to_s_sum",
            "mass_v_to_s_max", "mass_p_to_s_mean", "mass_p_to_s_sum", "mass_p_to_s_max",
            "corr_s_cls_to_v",
        ]
        with open(as_long_path(csv_path), "w", encoding="utf-8", newline="") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(str(row[key]) for key in header) + "\n")

    def _qskv_prob_by_layer(self, local_idx: int, affinities: list) -> dict:
        if not isinstance(affinities, list) or len(affinities) == 0:
            raise ValueError("Delta-Asv requires affinity outputs.")
        out = {}
        for li, aff in enumerate(affinities):
            if not isinstance(aff, dict):
                raise ValueError(f"Invalid affinity entry at layer {li}.")
            raw = aff.get("QsKv_raw", None)
            if (not torch.is_tensor(raw)) or raw.dim() != 4 or local_idx >= raw.shape[0]:
                raise ValueError(f"Delta-Asv requires QsKv_raw [B,H,S,V] at layer {li}.")
            raw_sample = raw[local_idx].detach().cpu().float()
            prob_sample = torch.softmax(raw_sample, dim=-1)
            out[int(li)] = {
                "raw": raw_sample,
                "prob": prob_sample,
            }
        return out

    def _class_mean_reference_affinities(self, image_chw: torch.Tensor, dataset, global_class_ids):
        if self.affinity_cfg is None:
            raise ValueError("Delta-Asv requires affinity configuration.")
        ref_input = image_chw.unsqueeze(0).to(self.device, non_blocking=True)
        ref_semantics = self._class_mean_semantics(dataset, batch_size=1)
        ref_affinity_cfg = dict(self.affinity_cfg)
        ref_affinity_cfg["semantic_length"] = int(self.cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS)
        _, ref_affinities = self.model.forward_with_affinity(
            ref_input,
            ref_affinity_cfg,
            semantics=ref_semantics,
            vis=False,
            class_ids=global_class_ids,
        )
        return ref_affinities

    def _vis_save_delta_asv(self, sample_dir: str, image_u8: np.ndarray, current_affinities: list,
        reference_affinities: list, current_local_idx: int,) -> None:
        current = self._qskv_prob_by_layer(current_local_idx, current_affinities)
        reference = self._qskv_prob_by_layer(0, reference_affinities)
        shared_layers = sorted(set(current.keys()) & set(reference.keys()))
        if len(shared_layers) == 0:
            raise ValueError("Delta-Asv found no shared QsKv layers.")

        arrays = {}
        panel_images = []
        panel_titles = []
        h, w = image_u8.shape[:2]
        for li in shared_layers:
            cur_prob = current[li]["prob"]
            ref_prob = reference[li]["prob"]
            if tuple(cur_prob.shape) != tuple(ref_prob.shape):
                raise ValueError(
                    f"Delta-Asv shape mismatch at layer {li}: current={tuple(cur_prob.shape)} reference={tuple(ref_prob.shape)}"
                )
            delta_prob = cur_prob - ref_prob
            abs_delta_map = delta_prob.abs().mean(dim=(0, 1))
            n_patch = int(abs_delta_map.shape[-1])
            g = self._infer_grid(n_patch)
            grid = abs_delta_map.view(g, g).numpy()

            arrays[f"layer_{li:02d}_asv_current_raw"] = current[li]["raw"].numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_asv_reference_raw"] = reference[li]["raw"].numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_asv_current_prob"] = cur_prob.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_asv_reference_prob"] = ref_prob.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_delta_asv_prob"] = delta_prob.numpy().astype(np.float32)
            arrays[f"layer_{li:02d}_abs_delta_asv_grid"] = grid.astype(np.float32)

            if self.vis_save_images:
                panel_images.append(overlay_heatmap(image_u8, resize_map_torch(grid, (h, w))))
                panel_titles.append(f"abs Delta Asv L{li:02d}")

        if self.vis_save_images and len(panel_images) > 0:
            save_panel(
                os.path.join(sample_dir, "delta_asv_layers.png"),
                panel_images,
                titles=panel_titles,
                ncols=min(4, len(panel_images)),
            )
        if self.vis_save_raw:
            np.savez_compressed(as_long_path(os.path.join(sample_dir, "delta_asv_layers.npz")), **arrays)
    # END SEMANTIC_ABLATION_EXPERIMENT

    def _vis_collect_sample(self, split: str, local_idx: int, sample_idx: int, image_chw: torch.Tensor, logits: torch.Tensor, target: int, pred: int, is_correct: bool, global_class_ids, model_ref, dataset, current_affinities: list, current_attn_weights: list,):
        if not self._vis_should_collect_case(is_correct):
            return
        r_head = model_ref.r_similarity_head
        if r_head is None:
            return
        affinities = current_affinities
        if not isinstance(affinities, list):
            return

        epoch = int(self._trace_epoch + 1)
        split = str(split).lower()
        case_tag = "correct" if is_correct else "wrong"
        case_name = f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_{case_tag}_y{int(target)}_p{int(pred)}"
        sample_dir = os.path.join(self.vis_dir, split, f"epoch_{epoch:03d}", case_name)
        ensure_dir(sample_dir)
        img_u8 = to_uint8_image(image_chw)
        h, w = img_u8.shape[:2]

        self_maps = self._vis_extract_cls_patch_layer_maps(local_idx)
        prompt_enable = bool(self.cfg.MODEL.PROMPT.ENABLE)
        qpqv_overlay_maps = self._vis_extract_prompt_visual_qk_layer_maps(local_idx, affinities, "QpQv_raw", "QpQv_vis") if prompt_enable else []
        kpkv_overlay_maps = self._vis_extract_prompt_visual_qk_layer_maps(local_idx, affinities, "KpKv_raw", "KpKv_vis") if prompt_enable else []
        qpkv_overlay_maps = self._vis_extract_prompt_visual_qk_layer_maps(local_idx, affinities, "QpKv_raw", "QpKv_vis") if prompt_enable else []
        qpqv_matrix_maps = self._vis_extract_prompt_visual_matrix_maps(local_idx, affinities, "QpQv_raw", "QpQv_vis") if prompt_enable else []
        kpkv_matrix_maps = self._vis_extract_prompt_visual_matrix_maps(local_idx, affinities, "KpKv_raw", "KpKv_vis") if prompt_enable else []
        qpkv_matrix_maps = self._vis_extract_prompt_visual_matrix_maps(local_idx, affinities, "QpKv_raw", "QpKv_vis") if prompt_enable else []
        semantic_enable = int(self._last_semantic_length) > 0
        qskv_overlay_maps = self._vis_extract_affinity_to_visual_maps(local_idx, affinities, "QsKv_raw", "QsKv_vis", visual_axis=2) if semantic_enable else []
        qvks_overlay_maps = self._vis_extract_affinity_to_visual_maps(local_idx, affinities, "QvKs_raw", "QvKs_vis", visual_axis=1) if semantic_enable else []
        qskv_matrix_maps = self._vis_extract_prompt_visual_matrix_maps(local_idx, affinities, "QsKv_raw", "QsKv_vis") if semantic_enable else []
        qvks_matrix_maps = self._vis_extract_prompt_visual_matrix_maps(local_idx, affinities, "QvKs_raw", "QvKs_vis") if semantic_enable else []
        qskp_matrix_maps = self._vis_extract_prompt_visual_matrix_maps(local_idx, affinities, "QsKp_raw", "QsKp_vis") if semantic_enable and prompt_enable else []
        qpks_matrix_maps = self._vis_extract_prompt_visual_matrix_maps(local_idx, affinities, "QpKs_raw", "QpKs_vis") if semantic_enable and prompt_enable else []
        qskp_prompt_maps = self._vis_extract_prompt_response_maps(local_idx, affinities, "QsKp_raw", "QsKp_vis", prompt_axis=2) if semantic_enable and prompt_enable else []
        qpks_prompt_maps = self._vis_extract_prompt_response_maps(local_idx, affinities, "QpKs_raw", "QpKs_vis", prompt_axis=1) if semantic_enable and prompt_enable else []

        overlay_groups = [("CLS", self_maps)]
        matrix_groups = []
        if prompt_enable:
            overlay_groups.extend([
                ("QpQv", qpqv_overlay_maps),
                ("KpKv", kpkv_overlay_maps),
                ("QpKv", qpkv_overlay_maps),
            ])
            matrix_groups.extend([
                ("QpQv", qpqv_matrix_maps),
                ("KpKv", kpkv_matrix_maps),
                ("QpKv", qpkv_matrix_maps),
            ])
        if semantic_enable:
            overlay_groups.extend([
                ("QsKv", qskv_overlay_maps),
                ("QvKs", qvks_overlay_maps),
            ])
            matrix_groups.extend([
                ("QsKv", qskv_matrix_maps),
                ("QvKs", qvks_matrix_maps),
                ("QsKp", qskp_matrix_maps),
                ("QpKs", qpks_matrix_maps),
            ])
        self._vis_save_overlay_group_panel(
            os.path.join(sample_dir, "overlay_layers.png"),
            img_u8,
            overlay_groups,
        )
        self._vis_save_overlay_group_raw(
            os.path.join(sample_dir, "overlay_layers.npz"),
            overlay_groups,
        )
        if prompt_enable:
            self._vis_save_matrix_group_panel(
                os.path.join(sample_dir, "affinity_matrix_layers.png"),
                matrix_groups,
            )
            self._vis_save_matrix_group_raw(
                os.path.join(sample_dir, "affinity_matrix_layers.npz"),
                matrix_groups,
            )
            self._vis_save_prompt_matrix(
                os.path.join(sample_dir, "qskp_prompt_layers.png"),
                qskp_prompt_maps,
                use_vis=True,
            )
            self._vis_save_prompt_raw(
                os.path.join(sample_dir, "qskp_prompt_layers.npz"),
                qskp_prompt_maps,
            )
            self._vis_save_prompt_matrix(
                os.path.join(sample_dir, "qpks_prompt_layers.png"),
                qpks_prompt_maps,
                use_vis=True,
            )
            self._vis_save_prompt_raw(
                os.path.join(sample_dir, "qpks_prompt_layers.npz"),
                qpks_prompt_maps,
            )

        # BEGIN SEMANTIC_ABLATION_EXPERIMENT
        if self.semantic_ablation_enable and self.semantic_ablation_attention_mass:
            self._vis_save_attention_mass(
                sample_dir=sample_dir,
                split=split,
                sample_idx=sample_idx,
                target=target,
                pred=pred,
                is_correct=is_correct,
                local_idx=local_idx,
                attn_weights=current_attn_weights,
            )
        if self.semantic_ablation_enable and self.semantic_ablation_delta_asv:
            if not semantic_enable:
                raise ValueError("Delta-Asv visualization requires active semantic tokens.")
            reference_affinities = self._class_mean_reference_affinities(
                image_chw=image_chw,
                dataset=dataset,
                global_class_ids=global_class_ids,
            )
            self._vis_save_delta_asv(
                sample_dir=sample_dir,
                image_u8=img_u8,
                current_affinities=affinities,
                reference_affinities=reference_affinities,
                current_local_idx=local_idx,
            )
        # END SEMANTIC_ABLATION_EXPERIMENT

        # Group 2: rollout maps (CLS + prompt tokens).
        if self.vis_rollout and isinstance(current_attn_weights, list) and len(current_attn_weights) > 0:
            roll = attention_rollout(current_attn_weights)
            if torch.is_tensor(roll) and local_idx < roll.shape[0]:
                r = roll[local_idx].detach().cpu()
                p_len = int(self.cfg.MODEL.PROMPT.NUM_TOKENS) if bool(self.cfg.MODEL.PROMPT.ENABLE) else 0
                s_len = int(self._last_semantic_length)
                start_patch = 1 + p_len
                end_patch = int(r.shape[0] - s_len)
                n_patch = int(max(0, end_patch - start_patch))
                if n_patch > 0:
                    g = self._infer_grid(n_patch)
                    cls_map = r[0, start_patch:end_patch].view(g, g).numpy()
                    cls_up = resize_map_torch(cls_map, (h, w))
                    if self.vis_save_images:
                        save_overlay(
                            os.path.join(sample_dir, "rollout_cls.png"),
                            img_u8,
                            cls_up,
                            title="CLS rollout",
                        )
                    if p_len > 0:
                        shallow_idx = 1
                        deep_idx = 1 + max(0, p_len - 1)
                        for name, sid in [("shallow_prompt", shallow_idx), ("deep_prompt", deep_idx)]:
                            if sid < r.shape[0]:
                                pm = r[sid, start_patch:end_patch].view(g, g).numpy()
                                pm_up = resize_map_torch(pm, (h, w))
                                if self.vis_save_images:
                                    save_overlay(
                                        os.path.join(sample_dir, f"rollout_{name}.png"),
                                        img_u8,
                                        pm_up,
                                            title=f"{name} rollout",
                                    )
                    s_to_v_up = None
                    if s_len > 0:
                        s_start = end_patch
                        s_rows = r[s_start:s_start + s_len, start_patch:end_patch]
                        if s_rows.numel() > 0:
                            s_map = s_rows.float().mean(dim=0).view(g, g).numpy()
                            s_to_v_up = resize_map_torch(s_map, (h, w))
                            if self.vis_save_images:
                                save_overlay(
                                    os.path.join(sample_dir, "rollout_s_to_v.png"),
                                    img_u8,
                                    s_to_v_up,
                                    title="S-to-V rollout",
                                )
                    if self.vis_save_raw:
                        arrays = {"cls": cls_up}
                        if s_to_v_up is not None:
                            arrays["s_to_v"] = s_to_v_up
                        np.savez_compressed(as_long_path(os.path.join(sample_dir, "rollout_maps.npz")), **arrays)

        self._vis_finalize_case(is_correct)

    def get_input(self, data):
        """
        从 dataloader 返回的数据字典中取出：
        - image
        - label
        - attribute（若存在）

        兼容 numpy / torch.Tensor 两种输入格式。

        返回：
        - inputs: float32 图像张量
        - labels: 标签张量
        - attributes: 属性张量或 None
        """
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]

        attributes = data.get("attribute") if isinstance(data, dict) else None
        if attributes is not None and not isinstance(attributes, torch.Tensor):
            attributes = torch.from_numpy(attributes)
        return inputs, labels, attributes

    ## 7. ================================main entry===============================
    def forward_one_batch(self, inputs, targets, is_train, attributes=None, dataset=None):
        """Train a single (full) epoch on the model using the given data loader.
       这是 Trainer 最核心的单 batch 执行函数。

        它完成的事情非常多：

        1. 把输入搬到 device
        2. 设置 trace 上下文
        3. 根据当前 protocol，从 dataset 中准备 local class_ids / local targets
        4. 调用 model forward 或 forward_with_affinity
        5. 若需要，从 affinity 中抽取对齐损失所需辅助量
        6. 根据损失类型决定：
            - 直接用 local-output 算 loss
            - 或先 remap seen-only
            - 或先 remap 到 eval-local 空间
        7. 检查 loss 是否 NaN / inf
        8. 如果 is_train=True：
            - backward
            - optimizer.step
            - 梯度/参数更新调试

        返回：
        - loss
        - outputs        """

        # =============================== 1. 输入搬到设备 ======================================
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )
        semantics = self._prepare_semantics_for_stage(
            attributes,
            dataset,
            batch_size=int(inputs.shape[0]),
            is_train=is_train,
        )
        self._last_semantic_length = (
            int(self.cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS)
            if torch.is_tensor(semantics)
            else 0
        )
        affinity_cfg = None
        if self.affinity_cfg is not None:
            affinity_cfg = dict(self.affinity_cfg)
            affinity_cfg["semantic_length"] = int(self._last_semantic_length)

        trace_id = self._make_trace_id()
        self._set_model_trace_context(trace_id)

        # =============================== 2. 前向推理======================================
        debug_logits = None
        with torch.set_grad_enabled(is_train):
            effective_targets = targets

            # 若启用 SHUFFLE_RAW_TARGETS，则训练时先打乱 raw targets
            if is_train and self.diag_shuffle_raw_targets and targets.numel() > 1:
                perm = torch.randperm(targets.shape[0], device=targets.device)
                effective_targets = targets.index_select(0, perm)

            # 根据 dataset 协议，准备当前活动类空间 local_class_ids 训练用 split-native local space 评测用 eval-local space
            local_class_ids = None
            if dataset is not None:
                local_class_ids, _, _ = self._prepare_dataset_local_targets(
                    effective_targets,
                    dataset,
                    use_eval_space=(not is_train),
                )

            # 把 runtime targets 暂存到 r_head，供某些动态评分模式使用
            model_ref_for_runtime = self._model_ref(self.model)
            r_head_runtime = model_ref_for_runtime.r_similarity_head
            if r_head_runtime is None:
                raise ValueError("Current ViT XLSA pipeline requires r_similarity_head to be attached.")
            r_head_runtime._runtime_targets = effective_targets.detach() if is_train else None

            if self.use_affinity:
                self._last_attn_weights = None
                if self.affinity_vis:
                    outputs, attn_weights, affinities = self.model.forward_with_affinity(
                        inputs, affinity_cfg, semantics=semantics, vis=True, class_ids=local_class_ids
                    )
                    self._last_attn_weights = attn_weights
                else:
                    outputs, affinities = self.model.forward_with_affinity(
                        inputs, affinity_cfg, semantics=semantics, class_ids=local_class_ids
                    )

                # 如果当前损失需要 affinity 辅助量，则从逐层 affinities 中抽取统一监测量。
                if self.affinity_aux_needed:
                    aux = self._extract_alignment_aux(affinities)
                    outputs = (outputs if not isinstance(outputs, tuple) else outputs[0], aux)
                else:
                    outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
            else:
                self._last_attn_weights = None
                outputs = self.model(inputs, semantics=semantics, class_ids=local_class_ids)

            r_head_runtime._runtime_targets = None

            # 3. 准备用于 loss 的 outputs / targets / weights
            loss_outputs = outputs
            loss_targets = effective_targets
            loss_weights = self.cls_weights

            # 当前主线由 trainer 统一完成 dataset local/eval-local remap，再交给分类损失。
            if dataset is not None:
                _, loss_targets, loss_weights = self._prepare_dataset_local_targets(
                    effective_targets,
                    dataset,
                    use_eval_space=(not is_train),
                )

            # 首次 trace 打印
            if (self.debug_trace_once or self.debug_grad_norm) and not self._debug_forward_trace_logged:
                logits_full = self._extract_logits(outputs)
                logits_loss = self._extract_logits(loss_outputs)
                with torch.no_grad():
                    logger.info(
                        "[trace] %s node=A.forward_one_batch inputs=%s targets[min,max,uniq]=(%d,%d,%d) "
                        "outputs_type=%s logits_full=%s logits_loss=%s seen_only=%s targets_seen_min=%s",
                        trace_id,
                        tuple(inputs.shape),
                        int(targets.min().item()),
                        int(targets.max().item()),
                        int(targets.unique().numel()),
                        self._get_output_type_name(outputs),
                        self._shape_or_none(logits_full),
                        self._shape_or_none(logits_loss),
                        False,
                        int(loss_targets.min().item()) if torch.is_tensor(loss_targets) else "NA",
                    )
                self._debug_forward_trace_logged = True

            # 训练时抓一份调试信息
            if is_train:
                self._capture_train_debug(loss_outputs, outputs, loss_targets)

            debug_logits = self._extract_logits(loss_outputs)
            if self.debug_grad_norm:
                self._log_batch_stats_once(debug_logits, loss_targets)

            # 打印对齐辅助量的 shape
            if self.debug_shapes and (not self._shape_debug_loss_aux_logged):
                _, aux_dbg = self._extract_logits_and_aux_for_debug(loss_outputs)
                if isinstance(aux_dbg, dict):
                    aff_qpqv_dbg = aux_dbg.get("aff_qpqv")
                    aff_kpkv_dbg = aux_dbg.get("aff_kpkv")
                    aff_qpkv_dbg = aux_dbg.get("aff_qpkv")
                    aff_qskv_dbg = aux_dbg.get("aff_qskv")
                    aff_qvks_dbg = aux_dbg.get("aff_qvks")
                    aff_qskp_dbg = aux_dbg.get("aff_qskp")
                    aff_qpks_dbg = aux_dbg.get("aff_qpks")
                    sample_layer = None
                    layer_keys = set()
                    for d in (aff_qpqv_dbg, aff_kpkv_dbg, aff_qpkv_dbg, aff_qskv_dbg, aff_qvks_dbg, aff_qskp_dbg, aff_qpks_dbg):
                        if isinstance(d, dict):
                            layer_keys.update(d.keys())
                    if len(layer_keys) > 0:
                        sample_layer = sorted(layer_keys)[0]
                    print(
                        "[SHAPE-DEBUG] trainer.loss_inputs aff_qpqv={} aff_kpkv={} aff_qpkv={} aff_qskv={} aff_qvks={} aff_qskp={} aff_qpks={} layer={}".format(
                            tuple(aff_qpqv_dbg[sample_layer].shape) if isinstance(aff_qpqv_dbg, dict) and sample_layer in aff_qpqv_dbg else (tuple(aff_qpqv_dbg.shape) if torch.is_tensor(aff_qpqv_dbg) else None),
                            tuple(aff_kpkv_dbg[sample_layer].shape) if isinstance(aff_kpkv_dbg, dict) and sample_layer in aff_kpkv_dbg else (tuple(aff_kpkv_dbg.shape) if torch.is_tensor(aff_kpkv_dbg) else None),
                            tuple(aff_qpkv_dbg[sample_layer].shape) if isinstance(aff_qpkv_dbg, dict) and sample_layer in aff_qpkv_dbg else (tuple(aff_qpkv_dbg.shape) if torch.is_tensor(aff_qpkv_dbg) else None),
                            tuple(aff_qskv_dbg[sample_layer].shape) if isinstance(aff_qskv_dbg, dict) and sample_layer in aff_qskv_dbg else (tuple(aff_qskv_dbg.shape) if torch.is_tensor(aff_qskv_dbg) else None),
                            tuple(aff_qvks_dbg[sample_layer].shape) if isinstance(aff_qvks_dbg, dict) and sample_layer in aff_qvks_dbg else (tuple(aff_qvks_dbg.shape) if torch.is_tensor(aff_qvks_dbg) else None),
                            tuple(aff_qskp_dbg[sample_layer].shape) if isinstance(aff_qskp_dbg, dict) and sample_layer in aff_qskp_dbg else (tuple(aff_qskp_dbg.shape) if torch.is_tensor(aff_qskp_dbg) else None),
                            tuple(aff_qpks_dbg[sample_layer].shape) if isinstance(aff_qpks_dbg, dict) and sample_layer in aff_qpks_dbg else (tuple(aff_qpks_dbg.shape) if torch.is_tensor(aff_qpks_dbg) else None),
                            sample_layer,
                        )
                    )
                    self._shape_debug_loss_aux_logged = True

            # ================== 3. compute loss ==================
            model_ref = self._model_ref(self.model)
            loss_kwargs = {
                "model": model_ref,
                "raw_targets": targets,
                "epoch": int(self._trace_epoch + 1),
            }
            # 常规分类损失（如 SoftmaxLoss），只需 outputs / targets / class_weights。
            loss = self.cls_criterion(
                loss_outputs, loss_targets, loss_weights, kwargs=loss_kwargs)

            # ========== 4. NaN / inf 防御==========
            if loss == float('inf'):
                raise FloatingPointError("encountered infinite loss during forward_one_batch")
            elif torch.isnan(loss).any():
                logits_dbg = debug_logits if torch.is_tensor(debug_logits) else self._extract_logits(loss_outputs)
                finite_ratio = float("nan")
                logit_min = float("nan")
                logit_max = float("nan")
                if torch.is_tensor(logits_dbg):
                    finite_mask = torch.isfinite(logits_dbg)
                    finite_ratio = float(finite_mask.float().mean().item())
                    if finite_mask.any():
                        finite_logits = logits_dbg[finite_mask]
                        logit_min = float(finite_logits.min().item())
                        logit_max = float(finite_logits.max().item())
                bad_rows = None
                if torch.is_tensor(logits_dbg) and logits_dbg.dim() == 2:
                    row_ok = torch.isfinite(logits_dbg).all(dim=1)
                    if not bool(row_ok.all().item()):
                        bad_rows = (~row_ok).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()

                # scale 可帮助判断 logits 的有效温度是否异常
                scale_dbg = None
                model_ref_dbg = self._model_ref(self.model)
                r_head_dbg = model_ref_dbg.r_similarity_head
                if r_head_dbg is not None:
                    scale_dbg = r_head_dbg._loss_last_logit_scale
                scale_str = (
                    str(float(scale_dbg.detach().item()))
                    if torch.is_tensor(scale_dbg) and scale_dbg.numel() == 1
                    else str(scale_dbg)
                )

                if torch.is_tensor(logits_dbg) and finite_ratio < 1.0:
                    reason = "non-finite logits"
                elif torch.is_tensor(scale_dbg) and (not torch.isfinite(scale_dbg).all()):
                    reason = "non-finite logit scale"
                else:
                    reason = "loss became NaN after logits were formed"

                if bad_rows is not None:
                    logger.info("[nan-debug] bad_rows=%s", bad_rows)
                if model_ref_dbg is not None:
                    bad_enc_rows = model_ref_dbg._last_bad_enc_rows
                    if bad_enc_rows is not None:
                        logger.info("[nan-debug] bad_enc_rows=%s", bad_enc_rows)
                if r_head_dbg is not None:
                    bad_feat_rows = r_head_dbg._last_bad_feat_rows
                    bad_visual_rows = r_head_dbg._last_bad_visual_rows
                    bad_sim_rows = r_head_dbg._last_bad_sim_rows
                    sem_all_finite = r_head_dbg._last_semantic_all_finite
                    if bad_feat_rows is not None:
                        logger.info("[nan-debug] bad_feat_rows=%s", bad_feat_rows)
                    if bad_visual_rows is not None:
                        logger.info("[nan-debug] bad_visual_rows=%s", bad_visual_rows)
                    if sem_all_finite is not None:
                        logger.info("[nan-debug] semantic_all_finite=%s", bool(sem_all_finite))
                    if bad_sim_rows is not None:
                        logger.info("[nan-debug] bad_sim_rows=%s", bad_sim_rows)

                raise FloatingPointError(
                    "encountered nan loss during forward_one_batch: "
                    f"reason={reason}, logits_finite_ratio={finite_ratio:.6f}, "
                    f"logits_min={logit_min:.6f}, logits_max={logit_max:.6f}, scale={scale_str}"
                )

        # =======backward and optim step only if in training phase... =========
        # ========== 5. 训练阶段执行 backward + step ==========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            refs = None
            before_norms = None
            if self.debug_grad_norm:
                refs = self._collect_debug_param_refs()
                self._log_grad_norms_once(refs)
                before_norms = self._capture_param_norms(refs)
            self.optimizer.step()
            if self.debug_grad_norm and refs is not None:
                after_norms = self._capture_param_norms(refs)
                self._log_update_once(before_norms, after_norms)

        return loss, outputs

    def _run_train_epoch(self, epoch, effective_total_epoch, total_data, train_loader, log_interval, losses, batch_time, data_time):
        # 只负责共享的单个训练 epoch
        losses.reset()
        batch_time.reset()
        data_time.reset()

        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
        logger.info("Training {} / {} epoch, with learning rate {}".format(epoch + 1, effective_total_epoch, lr))

        self.model.train()
        end = time.time()

        for idx, input_data in enumerate(train_loader):
            self._trace_stage = "train"
            self._trace_epoch = int(epoch)
            self._trace_iter = int(idx)
            self._trace_global_step += 1

            X, targets, attributes = self.get_input(input_data)
            data_time.update(time.time() - end)

            train_loss, _ = self.forward_one_batch(
                X,
                targets,
                True,
                attributes=attributes,
                dataset=train_loader.dataset,
            )

            losses.update(train_loss.item(), X.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % log_interval == 0:
                seconds_per_batch = batch_time.val
                eta = datetime.timedelta(
                    seconds=int(
                        seconds_per_batch * (total_data - idx - 1)
                        + seconds_per_batch * total_data * (effective_total_epoch - epoch - 1)
                    )
                )
                logger.info(
                    "\tTraining {}/{}. train loss: {:.4f},".format(
                        idx + 1,
                        total_data,
                        train_loss
                    )
                    + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                        seconds_per_batch,
                        data_time.val,
                        str(eta),
                    )
                    + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                )

        logger.info(
            "Epoch %d/%d train: loss=%.4f batch=%.4fs data=%.2es",
            epoch + 1,
            effective_total_epoch,
            float(losses.avg),
            float(batch_time.avg),
            float(data_time.avg),
        )

        if self.scheduler is not None:
            self.scheduler.step()

    def _train_classifier_dev(self, train_loader, val_loader, test_seen_loader, test_unseen_loader):
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        best_epoch = -1
        best_metric = float("-inf")
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        self.cls_weights = train_loader.dataset.get_class_weights(self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        patience = 0
        for epoch in range(total_epoch):
            self._run_train_epoch(epoch, total_epoch, total_data, train_loader, log_interval, losses, batch_time, data_time)

            self.model.eval()
            self.evaluator.update_iteration(epoch)

            self.eval_classifier(val_loader, "val_unseen")
            t_name = "val_unseen_" + val_loader.dataset.name
            metrics_this_epoch = (
                self.evaluator.results
                .get(f"epoch_{epoch}", {})
                .get("classification", {})
                .get(t_name, {})
            )
            metric_name, curr_acc = self._pick_primary_metric(metrics_this_epoch)
            if metric_name is None:
                logger.warning(
                    "No usable validation metric found for %s at epoch %d. Available keys: %s",
                    t_name,
                    epoch + 1,
                    sorted(list(metrics_this_epoch.keys())) if isinstance(metrics_this_epoch, dict) else [],
                )
                patience += 1
                if patience >= self.cfg.SOLVER.PATIENCE:
                    logger.info("No improvement. Breaking out of loop.")
                    break
                continue

            improved = curr_acc > best_metric
            logger.info(
                "[save-gate] epoch=%d metric=%s curr=%.6f best=%.6f improved=%s",
                epoch + 1,
                str(metric_name),
                float(curr_acc),
                float(best_metric),
                bool(improved),
            )

            if test_seen_loader is not None and str(self.evaluator.task_type).lower() == "gzsl":
                seen_metrics = self.eval_classifier(test_seen_loader, "test_seen")
            else:
                seen_metrics = None

            if test_unseen_loader is not None:
                unseen_metrics = self.eval_classifier(test_unseen_loader, "test_unseen")
            else:
                unseen_metrics = None

            if str(self.evaluator.task_type).lower() == "gzsl":
                self._update_gzsl_record_metrics(epoch, test_unseen_loader, seen_metrics, unseen_metrics)

            if improved:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    "Best epoch %d: best %s = %.3f",
                    best_epoch,
                    metric_name,
                    best_metric,
                )
                patience = 0
            else:
                patience += 1

            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

    def _train_classifier_final(self, train_loader, test_seen_loader, test_unseen_loader):
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        self.cls_weights = train_loader.dataset.get_class_weights(self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        for epoch in range(total_epoch):
            self._run_train_epoch(epoch, total_epoch, total_data, train_loader, log_interval, losses, batch_time, data_time)

            self.model.eval()
            self.evaluator.update_iteration(epoch)

            logger.info(
                "[final-protocol] epoch=%d no dev validation; early-stop disabled",
                epoch + 1,
            )

            if test_seen_loader is not None and str(self.evaluator.task_type).lower() == "gzsl":
                seen_metrics = self.eval_classifier(test_seen_loader, "test_seen")
            else:
                seen_metrics = None

            if test_unseen_loader is not None:
                unseen_metrics = self.eval_classifier(test_unseen_loader, "test_unseen")
            else:
                unseen_metrics = None

            if str(self.evaluator.task_type).lower() == "gzsl":
                self._update_gzsl_record_metrics(epoch, test_unseen_loader, seen_metrics, unseen_metrics)

    def train_classifier(self, train_loader, val_loader, test_seen_loader, test_unseen_loader):
        if val_loader is not None:
            return self._train_classifier_dev(train_loader, val_loader, test_seen_loader, test_unseen_loader)
        return self._train_classifier_final(train_loader, test_seen_loader, test_unseen_loader)

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix):
        """
        在一个 data_loader 上执行完整评测。

        它负责：
        1. 遍历整个验证/测试集
        2. 调用 forward_one_batch(..., is_train=False)
        3. 统一把 raw global targets remap 到 dataset 定义的 eval-local space
        4. 聚合 joint_logits 与 total_targets
        5. 调用 evaluator.classify()

        特别重要：
        - 这里的 total_targets 不再是 raw global labels
        - 而是 dataset 提供的 eval_global_to_local 现算出来的 targets_eval_local
        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        dataset = data_loader.dataset

        metric_key = self._resolve_eval_metric_key(dataset, prefix)
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        eval_class_ids, eval_map = self._dataset_space_meta(dataset, use_eval_space=True)
        eval_map = eval_map.to(dtype=torch.long)

        # initialize features and target
        total_logits = []
        total_targets = []
        model_ref = self._model_ref(self.model)
        r_head_eval = model_ref.r_similarity_head

        # 清理
        if r_head_eval is not None:
            r_head_eval._runtime_targets = None
        if self._vis_split_enabled(prefix):
            self._vis_init_epoch(prefix)

        # ========== 遍历整个验证/测试集==========
        for idx, input_data in enumerate(data_loader):
            self._trace_stage = f"eval_{prefix}"
            self._trace_iter = int(idx)
            self._trace_global_step += 1
            end = time.time()
            X, targets, attributes = self.get_input(input_data)

            data_time.update(time.time() - end)

            # 评测阶段：is_train=False
            loss, outputs = self.forward_one_batch(
                X,
                targets,
                False,
                attributes=attributes,
                dataset=data_loader.dataset,
            )

            losses.update(loss, X.shape[0])

            batch_time.update(time.time() - end)

            # periodic eval log
            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # raw global targets -> eval-local targets
            eval_map_cpu = eval_map.to(device=targets.device)
            targets_eval_local = eval_map_cpu.index_select(0, targets.long())
            if (targets_eval_local < 0).any():
                bad = targets[targets_eval_local < 0][:8].detach().cpu().tolist()
                raise ValueError(
                    "Found targets outside eval local space for split '{}' (e.g., {}).".format(
                        prefix, bad
                    )
                )

            total_targets.extend(list(targets_eval_local.detach().cpu().numpy()))

            # 提取 logits
            logits = outputs
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                logits = outputs[0]
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]

            total_logits.append(logits)

            # visualization：当前 split 若启用，就积累 trend 并保存若干样本图
            if self._vis_split_enabled(prefix):
                r_head_vis = model_ref.r_similarity_head
                affinities_vis = r_head_vis._runtime_affinities if r_head_vis is not None else None
                attn_weights_vis = self._last_attn_weights
                if torch.is_tensor(logits):
                    bsz = int(logits.shape[0])
                    for bi in range(bsz):
                        if self._vis_processed >= self.vis_max_samples:
                            break
                        pred_local = int(logits[bi].detach().argmax().item())
                        target_local = int(targets_eval_local[bi].item())
                        is_correct = bool(pred_local == target_local)
                        if not self._vis_should_collect_case(is_correct):
                            continue
                        gidx = int(idx * bsz + bi)
                        self._vis_collect_sample(
                            split=prefix,
                            local_idx=bi,
                            sample_idx=gidx,
                            image_chw=X[bi],
                            logits=logits.detach(),
                            target=target_local,
                            pred=pred_local,
                            is_correct=is_correct,
                            global_class_ids=eval_class_ids,
                            model_ref=model_ref,
                            dataset=data_loader.dataset,
                            current_affinities=affinities_vis,
                            current_attn_weights=attn_weights_vis,
                        )

        # 一个 split 跑完后的整体日志
        # 把所有 batch 的 logits 拼成全量矩阵
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()

        # 调 evaluator 做正式指标计算 这里已经是在 dataset-defined eval local space 上了
        raw_metrics = self.evaluator.classify(joint_logits, total_targets)
        metrics = {
            "top1": raw_metrics["top1"],
            metric_key: raw_metrics["per_class"],
        }
        log_results = {k: np.around(v * 100, decimals=2) for k, v in metrics.items()}
        logger.info(
            "Eval %s: loss=%.4f batch=%.4fs top1=%.2f %s=%.2f",
            test_name,
            float(losses.avg),
            float(batch_time.avg),
            float(log_results["top1"]),
            metric_key,
            float(log_results[metric_key]),
        )
        self.evaluator.log_and_update(log_results, metrics, test_name)
        return metrics




