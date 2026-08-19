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
import hashlib
import json
import re
from typing import Dict
from collections import defaultdict
from contextlib import nullcontext
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
from ..utils import distributed as du
from ..utils.train_utils import AverageMeter, gpu_mem_usage
from ..utils.runtime_progress import TrainingProgressController
from ..monitoring import (
    DiagnosticManager,
    MonitorManager,
    NumericalGuard,
    OptimizerSanity,
    PromptParameterTracker,
)
from ..monitoring.fields import GPP_MONITOR_ALIAS_ITEMS
from ..monitoring.comparability import build_comparability_identity
from ..monitoring.adapters import (
    auxiliary_loss_metrics,
    affinity_metrics,
    attention_mediation_metrics,
    deep_prompt_residual_metrics,
    graph_prob_prior_metrics,
    loss_component_metrics,
    prompt_distribution_metrics,
    semantic_token_metrics,
    train_debug_metrics,
)
from ..monitoring.module_effect import (
    attribute_concept_prompt_patch_block_intervention,
    attention_mediation_gamma_zero_intervention,
    checkpoint_sha256,
    both_prompt_zero_intervention,
    domain_prompt_zero_intervention,
    deep_prompt_residual_swap_intervention,
    deep_prompt_residual_zero_intervention,
    instance_prompt_swap_intervention,
    instance_prompt_zero_intervention,
    layer_prompt_read_block_intervention,
    PairedModuleEffectAccumulator,
    patch_prompt_uniform_intervention,
    prompt_patch_uniform_intervention,
    prompt_patch_value_globalize_intervention,
    prompt_value_zero_intervention,
    relevance_edge_delete_intervention,
    prompt_context_swap_intervention,
    prompt_read_block_intervention,
    random_prompt_patch_block_intervention,
    transport_prompt_patch_block_intervention,
    transport_random_patch_block_intervention,
    prompt_zero_intervention,
    prompt_write_block_intervention,
)
from ..monitoring.prompt_analysis import (
    PairedFlipAccumulator,
    PromptSourceDecompositionAccumulator,
    build_relevance_deletion_masks,
    summarize_deletion_curves,
)
from ..monitoring.bayesian_object_selection import (
    BayesianHierarchyTraceAccumulator,
    StaticPromptPerturbation,
    build_candidate_registry,
    build_object_selection_report,
    normalized_latent_direction,
    numeric_leaf_metrics,
    static_prompt_vector,
)
from ..monitoring.probe import (
    FixedProbeDataset,
    ProbeAttentionAffinityAccumulator,
    StreamingFixedProbeAccumulator,
    StreamingPatchDiversityAccumulator,
    StreamingTokenViewAccumulator,
    TargetRelevanceAccumulator,
    build_probe_manifest,
    validate_probe_manifest,
)
from ..data.transforms import get_transforms

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


class _StreamingProbeMetricRows:
    def __init__(self, diagnostic_manager, flush_rows=16384):
        self.diagnostic_manager = diagnostic_manager
        self.flush_rows = max(1, int(flush_rows))
        self.buffer = []
        self.total_count = 0

    def extend(self, rows):
        for row in rows:
            self.buffer.append(row)
            self.total_count += 1
            if len(self.buffer) >= self.flush_rows:
                self.flush()

    def flush(self):
        if not self.buffer:
            return
        self.diagnostic_manager.append_probe_metrics(self.buffer)
        self.buffer.clear()

    def __len__(self):
        return int(self.total_count)


class _TimedProbeLoader:
    def __init__(self, loader):
        self.loader = loader
        self.data_time_sec = 0.0
        self.batch_count = 0

    def __iter__(self):
        started = time.perf_counter()
        iterator = iter(self.loader)
        self.data_time_sec += time.perf_counter() - started
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                self.data_time_sec += time.perf_counter() - started
                return
            self.data_time_sec += time.perf_counter() - started
            self.batch_count += 1
            yield batch

    def __len__(self):
        return len(self.loader)


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

        self.cls_criterion = build_loss(self.cfg).to(self.device)
        # loss 对象自己声明是否需要 affinity aux，trainer 不再硬编码具体辅助损失名。
        self.affinity_aux_needed = bool(self.cls_criterion.requires_affinity_aux)
        if self.affinity_aux_needed and bool(cfg.MODEL.AFFINITY.DETACH):
            raise ValueError("Affinity auxiliary losses require MODEL.AFFINITY.DETACH=False.")
        self._last_semantic_length = 0
        self.affinity_monitor_requested = bool(cfg.MONITOR.ENABLE) and bool(cfg.MONITOR.AFFINITY.ENABLE)
        self.use_affinity = cfg.MODEL.AFFINITY.ENABLE or self.affinity_aux_needed or self.affinity_monitor_requested
        if self.use_affinity:
            self.affinity_cfg = {
                "prompt_length": cfg.MODEL.PROMPT.NUM_TOKENS,
                "semantic_length": cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS if cfg.MODEL.SEMANTIC_TOKENS.ENABLE else 0,
                "detach": cfg.MODEL.AFFINITY.DETACH,
                "block_s_to_cls": cfg.MODEL.SEMANTIC_TOKENS.BLOCK_S_TO_CLS,
            }
            self.affinity_vis = cfg.MODEL.AFFINITY.VIS
        else:
            self.affinity_cfg = None
            self.affinity_vis = False

        # solver related
        # ================== optimizer / scheduler / loss ==================
        # GraphProbPrior 的 prior head / 可学习温度等参数也挂在 loss module 内，需要交给 optimizer。
        self.optimizer = make_optimizer([self.model, self.cls_criterion], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)

        # ================== Checkpointer ==================
        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True,
            cls_criterion=self.cls_criterion,
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
        self._train_debug_contract_pending = True

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
        self._graph_prob_prior_forward = 0
        self._runtime_progress = TrainingProgressController(cfg, logger)
        self.graph_prob_prior_loss_active = (
            bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE)
            and float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT) > 0.0
        )
        self._trace_rank = int(du.get_rank())
        is_monitor_writer = du.get_rank() == 0
        self.monitor_manager = MonitorManager(cfg, is_writer=is_monitor_writer)
        self.diagnostic_manager = DiagnosticManager(
            cfg,
            self.monitor_manager,
            is_writer=is_monitor_writer,
        )
        guarded_modules = (("model", self._model_ref(self.model)), ("cls_criterion", self.cls_criterion))
        self.numerical_guard = NumericalGuard(guarded_modules)
        self._optimizer_sanity_enabled = bool(
            self.monitor_manager.monitor_groups["optimizer_sanity"]["effective"]
        )
        self.optimizer_sanity = (
            OptimizerSanity(guarded_modules, self.optimizer)
            if self._optimizer_sanity_enabled
            else None
        )
        self.prompt_parameter_tracker = PromptParameterTracker(self._model_ref(self.model))
        self._loss_component_epoch_sums = {}
        self._loss_component_epoch_counts = {}
        self._optimizer_sanity_first_step_done = False
        self._optimizer_sanity_payload = None
        if self.optimizer_sanity is not None:
            initialization_report = self.optimizer_sanity.initialization_report()
            self._optimizer_sanity_payload = {
                "initialization": initialization_report,
                "first_step": None,
            }
            self.monitor_manager.write_evidence("optimizer_sanity.json", self._optimizer_sanity_payload)
            self.monitor_manager.record_event(
                "optimizer_sanity",
                self.optimizer_sanity.event_summary(initialization_report),
            )
        self._probe_manifests = {}
        self._final_trainable_checkpoint_path = None
        self._fixed_probe_checkpoint_source = None
        self._milestone_probe_records = []
        self._milestone_probe_epochs = self._resolve_milestone_probe_epochs(
            int(cfg.SOLVER.TOTAL_EPOCH)
        )
        self.monitor_manager.record_event(
            "monitor_initialized",
            {
                "protocol_mode": str(cfg.DATA.XLSA.PROTOCOL_MODE),
                "task_type": str(self.evaluator.task_type),
            },
        )

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
        if split == "train_eval_seen":
            if protocol_mode != "final_gzsl":
                raise ValueError(
                    "train_eval_seen is only valid under final_gzsl, got '{}'".format(
                        protocol_mode
                    )
                )
            return "train_eval_seen"
        raise ValueError("Unsupported eval split '{}' for metric-key resolution".format(prefix))

    def _update_gzsl_record_metrics(
        self,
        epoch: int,
        test_seen_loader,
        test_unseen_loader,
        seen_metrics,
        unseen_metrics,
    ):
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
        }
        selection_debug = {
            "best_gzsl_seen_recorded": float(self.best_gzsl_seen_recorded),
            "best_gzsl_unseen_recorded": float(self.best_gzsl_unseen_recorded),
            "historical_independent_best_upper_bound": recorded_h,
        }
        self.evaluator.update_result("classification", {eval_name: combined})
        self.evaluator.update_result("checkpoint_selection_debug", {eval_name: selection_debug})
        self.monitor_manager.set_context(
            stage="eval",
            epoch=int(epoch + 1),
            global_step=int(self._trace_global_step),
            graph_prob_prior_forward=int(self._graph_prob_prior_forward),
        )
        self.monitor_manager.record_epoch(
            "test_gzsl",
            "classification",
            combined,
            reducer="dataset",
            n=len(test_seen_loader.dataset) + len(test_unseen_loader.dataset),
        )
        self.monitor_manager.record_epoch(
            "test_gzsl",
            "checkpoint_selection_debug",
            selection_debug,
            reducer="history",
            n=1,
        )
        self.diagnostic_manager.record_calibration(int(epoch + 1))
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
        4. r_similarity_head prototype projection / logit scale
        5. semantic side branch """
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
        refs["r_head.prototype_proj"] = self._find_param_by_name_contains(
            ["r_similarity_head.prototype_proj.weight", "r_similarity_head.prototype_proj"]
        )
        refs["r_head.logit_scale"] = self._find_param_by_name_contains(
            ["r_similarity_head.logit_scale"]
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
        if str(semantic_cfg.TOKENIZER).lower() == "orthogonal" and source != "class_mean":
            raise ValueError("MODEL.SEMANTIC_TOKENS.TOKENIZER='orthogonal' requires TRAIN_SOURCE=class_mean and EVAL_SOURCE=class_mean.")
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
                "  group[%d]: lr=%s wd=%s params=%d (head=%s, r_head=%s, prompt_dist=%s, semantic=%s)",
                idx,
                group.get("lr", None),
                group.get("weight_decay", None),
                len(params),
                any("head." in n for n in names),
                any("r_similarity_head" in n for n in names),
                any("prompt_init_provider" in n for n in names),
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

    def _sync_cls_criterion_grads(self) -> None:
        if int(getattr(self.cfg, "NUM_GPUS", 1)) <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        grads = [
            param.grad
            for param in self.cls_criterion.parameters()
            if param.requires_grad and param.grad is not None
        ]
        if grads:
            du.scaled_all_reduce(self.cfg, grads)

    def _format_graph_prob_prior_monitor_log(self) -> str:
        if not bool(self.cfg.MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE):
            return ""
        stats = getattr(self.cls_criterion, "_last_loss_stats", None)
        if not isinstance(stats, dict):
            return ""
        keys = GPP_MONITOR_ALIAS_ITEMS
        parts = []
        for label, key in keys:
            value = stats.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                parts.append(f"{label}={float(value):.4g}")
        if not parts:
            return ""
        return "\n\t[graph-prob-prior-monitor] " + " ".join(parts)

    def _log_train_loader_summary(self, protocol_name: str, train_loader, total_data: int, log_interval: int) -> None:
        dataset = getattr(train_loader, "dataset", None)
        try:
            dataset_size = len(dataset) if dataset is not None else None
        except TypeError:
            dataset_size = None
        loader_batch_size = getattr(train_loader, "batch_size", None)
        logger.info(
            "[train-loader] protocol=%s batches_per_epoch=%d dataset_size=%s cfg_batch_size=%s loader_batch_size=%s log_every_n=%d",
            str(protocol_name),
            int(total_data),
            str(dataset_size) if dataset_size is not None else "unknown",
            str(self.cfg.DATA.BATCH_SIZE),
            str(loader_batch_size) if loader_batch_size is not None else "unknown",
            int(log_interval),
        )

    def _capture_train_debug(self, loss_outputs, raw_outputs, loss_targets):
        ce_logits = self._extract_logits(loss_outputs)
        raw_logits = self._extract_logits(raw_outputs)
        if not torch.is_tensor(ce_logits):
            self._last_train_debug = {}
            return

        with torch.no_grad():
            ce_value = ce_logits.detach().float()
            ce_probs = torch.softmax(ce_value, dim=-1)
            ce_entropy = -(ce_probs * torch.log(ce_probs.clamp_min(1e-12))).sum(dim=-1).mean()
            top1 = (ce_logits.argmax(dim=1) == loss_targets).float().mean()

            self._last_train_debug = {
                "ce_logits_std": float(ce_value.std(unbiased=False).item()),
                "ce_logits_abs_max": float(ce_value.abs().max().item()),
                "ce_entropy": float(ce_entropy.item()),
                "seen_only_top1": float(top1.item()),
            }
            model_ref = self._model_ref(self.model)
            r_head = model_ref.r_similarity_head
            if r_head is not None:
                learnable_scale = getattr(r_head, "logit_scale", None)
                scale_t = getattr(r_head, "_loss_last_logit_scale", None)
                if (
                    torch.is_tensor(learnable_scale)
                    and learnable_scale.requires_grad
                    and torch.is_tensor(scale_t)
                ):
                    self._last_train_debug["effective_logit_scale"] = float(scale_t.detach().mean().item())
            if self._train_debug_contract_pending:
                same_tensor = (
                    torch.is_tensor(raw_logits)
                    and ce_logits.shape == raw_logits.shape
                    and ce_logits.data_ptr() == raw_logits.data_ptr()
                )
                self._last_train_debug["ce_vs_raw_same_tensor"] = bool(same_tensor)
                if torch.is_tensor(raw_logits) and not same_tensor:
                    raw_value = raw_logits.detach().float()
                    raw_probs = torch.softmax(raw_value, dim=-1)
                    raw_entropy = -(
                        raw_probs * torch.log(raw_probs.clamp_min(1e-12))
                    ).sum(dim=-1).mean()
                    self._last_train_debug.update(
                        {
                            "raw_logits_std": float(raw_value.std(unbiased=False).item()),
                            "raw_logits_abs_max": float(raw_value.abs().max().item()),
                            "raw_entropy": float(raw_entropy.item()),
                        }
                    )

    def _record_train_step_monitors(self, train_loss):
        epoch = int(self._trace_epoch + 1) if self._trace_epoch >= 0 else None
        self.monitor_manager.set_context(
            stage="train",
            epoch=epoch,
            global_step=int(self._trace_global_step),
            graph_prob_prior_forward=int(self._graph_prob_prior_forward),
        )
        loss_stats = getattr(self.cls_criterion, "_last_loss_stats", None)
        if self.monitor_manager.should_sample_step("graph_prob_prior"):
            self.monitor_manager.record_step(
                "graph_prob_prior",
                graph_prob_prior_metrics(loss_stats),
            )
        if not self.monitor_manager.should_sample_step("train"):
            return
        loss_value = float(train_loss.detach().item()) if torch.is_tensor(train_loss) else float(train_loss)
        self.monitor_manager.record_step(
            "train",
            {
                "loss": loss_value,
                "lr": float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else None,
            },
        )
        train_debug = train_debug_metrics(self._last_train_debug)
        self.monitor_manager.record_step("train_debug", train_debug)
        if "ce_vs_raw_same_tensor" in train_debug:
            self._train_debug_contract_pending = False
        model_ref = self._model_ref(self.model)
        self.monitor_manager.record_step(
            "prompt_distribution",
            prompt_distribution_metrics(model_ref.get_runtime_prompt_distribution_stats()),
        )
        self.monitor_manager.record_step(
            "deep_prompt_residual",
            deep_prompt_residual_metrics(
                model_ref.get_runtime_deep_prompt_residual_trace()
            ),
        )
        self.monitor_manager.record_step(
            "semantic_token_health",
            semantic_token_metrics(model_ref.get_runtime_semantic_state()),
        )
        self.monitor_manager.record_step(
            "attention_mediation",
            attention_mediation_metrics(model_ref.get_runtime_attention_mediation_stats()),
        )
        self.monitor_manager.record_step(
            "affinity_summary",
            affinity_metrics(model_ref.get_runtime_affinities()),
        )
        self.monitor_manager.record_step(
            "auxiliary_loss_health",
            auxiliary_loss_metrics(loss_stats),
        )

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

            # consistency 等分类头内部缓存需要 raw global target 时，由 forward 显式传入。
            # 不再通过外部写 r_similarity_head._runtime_targets 暂存状态。
            runtime_targets = effective_targets.detach() if is_train else None

            use_affinity_for_batch = self.use_affinity and (
                bool(is_train) or bool(self.cfg.MODEL.AFFINITY.ENABLE) or self.affinity_aux_needed
            )
            if use_affinity_for_batch:
                self._last_attn_weights = None
                if self.affinity_vis:
                    outputs, attn_weights, affinities = self.model.forward_with_affinity(
                        inputs,
                        affinity_cfg,
                        semantics=semantics,
                        vis=True,
                        class_ids=local_class_ids,
                        runtime_targets=runtime_targets,
                    )
                    self._last_attn_weights = attn_weights
                else:
                    outputs, affinities = self.model.forward_with_affinity(
                        inputs,
                        affinity_cfg,
                        semantics=semantics,
                        class_ids=local_class_ids,
                        runtime_targets=runtime_targets,
                    )

                # 如果当前损失需要 affinity 辅助量，则从逐层 affinities 中抽取统一监测量。
                if self.affinity_aux_needed:
                    aux = self._extract_alignment_aux(affinities)
                    outputs = (outputs if not isinstance(outputs, tuple) else outputs[0], aux)
                else:
                    outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
            else:
                self._last_attn_weights = None
                outputs = self.model(
                    inputs,
                    semantics=semantics,
                    class_ids=local_class_ids,
                    runtime_targets=runtime_targets,
                )

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
            dataset_class_attributes = None
            if dataset is not None and hasattr(dataset, "class_attributes"):
                # XLSA/CUB 下这里通常是 [200,312] 的全局类别属性矩阵。
                # GraphProbPrior 直接复用 dataloader 已加载的全类属性矩阵；
                # 不再提供路径兜底，避免和语义分支的数据来源分叉。
                dataset_class_attributes = dataset.class_attributes
            dataset_seen_classes = getattr(dataset, "seen_classes", None) if dataset is not None else None
            dataset_unseen_classes = getattr(dataset, "unseen_classes", None) if dataset is not None else None
            loss_kwargs = {
                "model": model_ref,
                "raw_targets": targets,
                # GraphProbPrior 使用全局类别 id；不能使用 local-output remap 后的 loss_targets。
                "targets_global": effective_targets.detach(),
                "class_attributes": dataset_class_attributes,
                "seen_class_ids": dataset_seen_classes,
                "unseen_class_ids": dataset_unseen_classes,
                "epoch": int(self._trace_epoch + 1),
                "is_train": bool(is_train),
                # 属性重建辅助损失使用 batch 真实类别属性 a_y 作为监督目标。
                # attributes 来自 xlsa_dataset.__getitem__ 返回的 class_attributes[label]。
                "target_attributes": attributes.to(self.device, non_blocking=True).float() if torch.is_tensor(attributes) else None,
            }
            # 常规分类损失（如 SoftmaxLoss），只需 outputs / targets / class_weights。
            loss = self.cls_criterion(
                loss_outputs, loss_targets, loss_weights, kwargs=loss_kwargs)

            # ========== 4. NaN / inf 防御==========
            numerical_failure = self.numerical_guard.check_forward(loss, debug_logits)
            if numerical_failure is not None:
                self.monitor_manager.set_context(
                    stage=str(self._trace_stage),
                    epoch=int(self._trace_epoch + 1) if self._trace_epoch >= 0 else None,
                    global_step=int(self._trace_global_step),
                    graph_prob_prior_forward=int(self._graph_prob_prior_forward),
                )
                self.monitor_manager.record_event("numerical_guard", numerical_failure)
                raise FloatingPointError(
                    "numerical_guard rejected forward: {}".format(numerical_failure["failures"][:8])
                )
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
                    bad_enc_rows = getattr(model_ref_dbg, "_last_bad_enc_rows", None)
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
            self._sync_cls_criterion_grads()
            self.prompt_parameter_tracker.observe_gradients()
            numerical_failure = self.numerical_guard.check_gradients()
            if numerical_failure is not None:
                self.monitor_manager.set_context(
                    stage="train",
                    epoch=int(self._trace_epoch + 1) if self._trace_epoch >= 0 else None,
                    global_step=int(self._trace_global_step),
                    graph_prob_prior_forward=int(self._graph_prob_prior_forward),
                )
                self.monitor_manager.record_event("numerical_guard", numerical_failure)
                raise FloatingPointError(
                    "numerical_guard rejected backward: {}".format(numerical_failure["failures"][:8])
                )
            if (
                self.graph_prob_prior_loss_active
                and bool(self.cfg.MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE)
            ):
                monitor_every = max(1, int(self.cfg.MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N))
                graph_prob_prior_forward = int(self._graph_prob_prior_forward) + 1
                if graph_prob_prior_forward % monitor_every == 0:
                    from ..solver.graph_prob_prior_monitors import graph_prob_prior_grad_monitor

                    loss_stats = getattr(self.cls_criterion, "_last_loss_stats", None)
                    if isinstance(loss_stats, dict):
                        named_parameters = list(self.model.named_parameters())
                        named_parameters.extend(
                            (f"cls_criterion.{name}", param)
                            for name, param in self.cls_criterion.named_parameters()
                        )
                        loss_stats.update(graph_prob_prior_grad_monitor(named_parameters))
            refs = None
            before_norms = None
            if self.debug_grad_norm:
                refs = self._collect_debug_param_refs()
                self._log_grad_norms_once(refs)
                before_norms = self._capture_param_norms(refs)
            self.optimizer.step()
            numerical_failure = self.numerical_guard.check_parameters()
            if numerical_failure is not None:
                self.monitor_manager.set_context(
                    stage="train",
                    epoch=int(self._trace_epoch + 1) if self._trace_epoch >= 0 else None,
                    global_step=int(self._trace_global_step),
                    graph_prob_prior_forward=int(self._graph_prob_prior_forward),
                )
                self.monitor_manager.record_event("numerical_guard", numerical_failure)
                raise FloatingPointError(
                    "numerical_guard rejected optimizer step: {}".format(numerical_failure["failures"][:8])
                )
            if self.optimizer_sanity is not None and not self._optimizer_sanity_first_step_done:
                self._optimizer_sanity_first_step_done = True
                first_step_report = self.optimizer_sanity.first_step_report()
                self._optimizer_sanity_payload["first_step"] = first_step_report
                self.monitor_manager.set_context(
                    stage="train",
                    epoch=int(self._trace_epoch + 1) if self._trace_epoch >= 0 else None,
                    global_step=int(self._trace_global_step),
                    graph_prob_prior_forward=int(self._graph_prob_prior_forward),
                )
                self.monitor_manager.write_evidence("optimizer_sanity.json", self._optimizer_sanity_payload)
                self.monitor_manager.record_event(
                    "optimizer_sanity",
                    self.optimizer_sanity.event_summary(first_step_report),
                )
                self.monitor_manager.record_event(
                    "numerical_guard",
                    {
                        "phase": "first_backward_optimizer_step",
                        "loss_is_finite": True,
                        "logits_are_finite": True,
                        "grads_are_finite": True,
                        "params_are_finite_after_step": True,
                    },
                )
            if self.debug_grad_norm and refs is not None:
                after_norms = self._capture_param_norms(refs)
                self._log_update_once(before_norms, after_norms)

        return loss, outputs

    def _aggregate_train_epoch_metrics(self, losses, batch_time, data_time):
        loss_sum = float(losses.sum)
        sample_count = int(losses.count)
        batch_time_sec = float(batch_time.avg)
        data_time_sec = float(data_time.avg)
        if du.get_world_size() > 1:
            loss_stats = torch.tensor(
                [loss_sum, float(sample_count)],
                dtype=torch.float32,
                device=self.device,
            )
            torch.distributed.all_reduce(loss_stats, op=torch.distributed.ReduceOp.SUM)
            loss_sum = float(loss_stats[0].item())
            sample_count = int(round(float(loss_stats[1].item())))
            timing_stats = torch.tensor(
                [batch_time_sec, data_time_sec],
                dtype=torch.float32,
                device=self.device,
            )
            torch.distributed.all_reduce(timing_stats, op=torch.distributed.ReduceOp.MAX)
            batch_time_sec = float(timing_stats[0].item())
            data_time_sec = float(timing_stats[1].item())
        loss = loss_sum / sample_count if sample_count > 0 else 0.0
        return {
            "loss": float(loss),
            "batch_time_sec": batch_time_sec,
            "data_time_sec": data_time_sec,
            "sample_count": sample_count,
        }

    def _reset_loss_component_epoch_stats(self):
        self._loss_component_epoch_sums = {}
        self._loss_component_epoch_counts = {}

    def _observe_loss_component_epoch_stats(self, sample_count):
        if not bool(self.cfg.MONITOR.LOSS_COMPONENT_TRAJECTORY.ENABLE):
            return
        metrics = loss_component_metrics(
            getattr(self.cls_criterion, "_last_loss_stats", None)
        )
        count = max(0, int(sample_count))
        for name, value in metrics.items():
            self._loss_component_epoch_sums[name] = (
                self._loss_component_epoch_sums.get(name, 0.0)
                + float(value) * count
            )
            self._loss_component_epoch_counts[name] = (
                self._loss_component_epoch_counts.get(name, 0) + count
            )

    def _finalize_loss_component_epoch_stats(self):
        result = {}
        for name in sorted(self._loss_component_epoch_sums):
            total = float(self._loss_component_epoch_sums[name])
            count = int(self._loss_component_epoch_counts.get(name, 0))
            if du.get_world_size() > 1:
                state = torch.tensor(
                    [total, float(count)], dtype=torch.float64, device=self.device
                )
                torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.SUM)
                total = float(state[0].item())
                count = int(round(float(state[1].item())))
            if count > 0:
                result[name] = total / count
        return result

    def _build_train_progress(self, epoch, effective_total_epoch, total_data, train_loader):
        return self._runtime_progress.build_train_progress(
            epoch,
            effective_total_epoch,
            total_data,
            train_loader,
        )

    def _write_progress_state(self, force=False, **updates):
        self._runtime_progress.write_state(force=force, **updates)

    def _begin_progress_epoch(self, epoch, total_epochs, total_batches):
        self._runtime_progress.begin_epoch(epoch, total_epochs, total_batches)

    def _record_progress_batch(self, phase, epoch, total_epochs, batch, total_batches, batch_time_seconds):
        self._runtime_progress.record_batch(
            phase,
            epoch,
            total_epochs,
            batch,
            total_batches,
            batch_time_seconds,
        )

    def _finish_progress_train_phase(self, epoch, total_epochs, total_batches):
        self._runtime_progress.finish_train_phase(epoch, total_epochs, total_batches)

    def _finish_progress_epoch(self, epoch, total_epochs):
        self._runtime_progress.finish_epoch(epoch, total_epochs)

    def _finalize_progress_state(self, status):
        self._runtime_progress.finalize(status)

    def _run_train_epoch(self, epoch, effective_total_epoch, total_data, train_loader, log_interval, losses, batch_time, data_time):
        # 只负责共享的单个训练 epoch
        losses.reset()
        batch_time.reset()
        data_time.reset()
        self.prompt_parameter_tracker.reset_epoch_gradient_stats()
        self._reset_loss_component_epoch_stats()

        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))

        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
        logger.info("Training {} / {} epoch, with learning rate {}".format(epoch + 1, effective_total_epoch, lr))

        self.model.train()
        end = time.time()
        self._begin_progress_epoch(epoch, effective_total_epoch, total_data)
        progress = self._build_train_progress(
            epoch,
            effective_total_epoch,
            total_data,
            train_loader,
        )
        iterator = progress if progress is not None else train_loader
        world_size = max(1, int(du.get_world_size()))

        try:
            for idx, input_data in enumerate(iterator):
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
                self._observe_loss_component_epoch_stats(X.shape[0])
                if self.graph_prob_prior_loss_active:
                    self._graph_prob_prior_forward += 1
                self._record_train_step_monitors(train_loss)
                batch_time.update(time.time() - end)
                end = time.time()
                self._record_progress_batch(
                    "train",
                    epoch,
                    effective_total_epoch,
                    idx + 1,
                    total_data,
                    batch_time.val,
                )

                if progress is not None:
                    loss_label = "loss" if world_size == 1 else "loss_r0"
                    mem_label = "mem" if world_size == 1 else "mem_r0"
                    progress.set_postfix_str(
                        f"{loss_label}={train_loss.item():.4f}, "
                        f"lr={lr:.2e}, {mem_label}={gpu_mem_usage():.1f}G",
                        refresh=False,
                    )

                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(
                        seconds=int(
                            seconds_per_batch * (total_data - idx - 1)
                            + seconds_per_batch * total_data * (effective_total_epoch - epoch - 1)
                        )
                    )
                    if progress is not None:
                        progress.clear()
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
                        + self._format_graph_prob_prior_monitor_log()
                    )
                    if progress is not None:
                        progress.refresh()
        finally:
            if progress is not None:
                progress.refresh()
                progress.close()
        self._finish_progress_train_phase(epoch, effective_total_epoch, total_data)

        epoch_metrics = self._aggregate_train_epoch_metrics(losses, batch_time, data_time)
        logger.info(
            "Epoch %d/%d train: loss=%.4f batch=%.4fs data=%.2es",
            epoch + 1,
            effective_total_epoch,
            epoch_metrics["loss"],
            epoch_metrics["batch_time_sec"],
            epoch_metrics["data_time_sec"],
        )
        self.monitor_manager.set_context(
            stage="train",
            epoch=int(epoch + 1),
            global_step=int(self._trace_global_step),
            graph_prob_prior_forward=int(self._graph_prob_prior_forward),
        )
        self.monitor_manager.record_epoch(
            "train",
            "train_epoch",
            {
                "loss": epoch_metrics["loss"],
                "batch_time_sec": epoch_metrics["batch_time_sec"],
                "data_time_sec": epoch_metrics["data_time_sec"],
                "lr": float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else None,
            },
            reducer={
                "loss": "sample_mean",
                "lr": "last",
                "batch_time_sec": "mean",
                "data_time_sec": "mean",
            },
            n=epoch_metrics["sample_count"],
        )
        prompt_parameter_metrics = self.prompt_parameter_tracker.metrics()
        for layer_index, layer_metrics in self.prompt_parameter_tracker.layer_metrics().items():
            prompt_parameter_metrics.update({
                f"layer_{layer_index}.{name}": value
                for name, value in layer_metrics.items()
            })
        self.monitor_manager.record_epoch(
            "train",
            "prompt_parameter_health",
            prompt_parameter_metrics,
            reducer="last",
            n=1,
        )
        self.prompt_parameter_tracker.commit_epoch_snapshot()
        if bool(self.cfg.MONITOR.LOSS_COMPONENT_TRAJECTORY.ENABLE):
            loss_component_epoch_metrics = self._finalize_loss_component_epoch_stats()
            self.monitor_manager.record_epoch(
                "train",
                "loss_component_trajectory",
                loss_component_epoch_metrics,
                reducer={
                    name: ("last" if name.endswith(".weight") else "sample_mean")
                    for name in loss_component_epoch_metrics
                },
                n=epoch_metrics["sample_count"],
            )

        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def _new_equivalence_state():
        return {
            "logit_abs_sum": 0.0,
            "logit_element_count": 0,
            "logit_max_abs_diff": 0.0,
            "true_margin_abs_sum": 0.0,
            "sample_count": 0,
            "true_margin_abs_diff": 0.0,
            "prediction_flip_count": 0,
        }

    @classmethod
    def _update_equivalence_state(cls, state, normal_logits, changed_logits, targets):
        normal = np.asarray(torch.as_tensor(normal_logits).detach().cpu(), dtype=np.float32)
        changed = np.asarray(torch.as_tensor(changed_logits).detach().cpu(), dtype=np.float32)
        target = np.asarray(targets, dtype=np.int64).reshape(-1)
        difference = np.abs(normal - changed)
        normal_margin = cls._true_margin_numpy(normal, target)
        changed_margin = cls._true_margin_numpy(changed, target)
        margin_difference = np.abs(normal_margin - changed_margin)
        state["logit_abs_sum"] += float(difference.sum())
        state["logit_element_count"] += int(difference.size)
        state["logit_max_abs_diff"] = max(state["logit_max_abs_diff"], float(difference.max(initial=0.0)))
        state["true_margin_abs_sum"] += float(margin_difference.sum())
        state["true_margin_abs_diff"] = max(state["true_margin_abs_diff"], float(margin_difference.max(initial=0.0)))
        state["prediction_flip_count"] += int((normal.argmax(axis=1) != changed.argmax(axis=1)).sum())
        state["sample_count"] += int(target.size)

    @staticmethod
    def _finalize_equivalence_state(state):
        return {
            "logit_max_abs_diff": float(state["logit_max_abs_diff"]),
            "logit_mean_abs_diff": float(state["logit_abs_sum"] / max(1, state["logit_element_count"])),
            "true_margin_abs_diff": float(state["true_margin_abs_diff"]),
            "true_margin_mean_abs_diff": float(state["true_margin_abs_sum"] / max(1, state["sample_count"])),
            "prediction_flip_rate": float(state["prediction_flip_count"] / max(1, state["sample_count"])),
        }

    @staticmethod
    def _probe_metric_delta(normal_metrics, intervention_metrics):
        result = {}
        for name in sorted(set(normal_metrics).intersection(intervention_metrics)):
            normal_value = normal_metrics[name]
            intervention_value = intervention_metrics[name]
            if not isinstance(normal_value, (int, float, np.integer, np.floating)):
                continue
            if not isinstance(intervention_value, (int, float, np.integer, np.floating)):
                continue
            delta = float(intervention_value) - float(normal_value)
            if math.isfinite(delta):
                result[f"delta_{name}"] = delta
        return result

    @classmethod
    def _probe_layer_metric_delta(cls, normal_layers, intervention_layers):
        return {
            int(layer_index): cls._probe_metric_delta(
                normal_layers[layer_index], intervention_layers[layer_index]
            )
            for layer_index in sorted(set(normal_layers).intersection(intervention_layers))
        }

    def _semantic_permutation(self, candidate_class_ids, split):
        seed = int(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.PERMUTATION_SEED)
        split_offset = int(hashlib.sha256(str(split).encode("utf-8")).hexdigest()[:8], 16)
        effective_seed = int((seed + split_offset) % (2 ** 32))
        generator = np.random.RandomState(effective_seed)
        permutation = generator.permutation(len(candidate_class_ids)).astype(np.int64)
        if permutation.size > 1 and np.array_equal(permutation, np.arange(permutation.size)):
            permutation = np.roll(permutation, 1)
        inverse = np.argsort(permutation)
        encoded = json.dumps(permutation.tolist(), separators=(",", ":")).encode("utf-8")
        return permutation, inverse, effective_seed, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _attribute_concept_reference(model_ref, raw_semantic_reference):
        if raw_semantic_reference is None:
            return {
                "available": False,
                "reason": "class_attribute_reference_unavailable",
            }
        head = getattr(model_ref, "r_similarity_head", None)
        projector = getattr(head, "prototype_proj", None)
        if not isinstance(projector, torch.nn.Linear):
            return {
                "available": False,
                "reason": "linear_prototype_projection_unavailable",
            }
        raw = torch.as_tensor(raw_semantic_reference).detach().float()
        weight = projector.weight.detach()
        if raw.dim() != 2 or weight.dim() != 2 or raw.shape[1] != weight.shape[1]:
            return {
                "available": False,
                "reason": "attribute_projection_dimension_mismatch",
            }
        return {
            "available": True,
            "reason": None,
            "directions": weight.transpose(0, 1),
            "attribute_count": int(raw.shape[1]),
            "score_mode": str(getattr(head, "score_mode", "unknown")).lower(),
            "margin_reference": (
                "additive_exact"
                if str(getattr(head, "score_mode", "unknown")).lower() == "dot"
                else "projected_direction_reference"
            ),
        }

    def _module_effect_intervention_specs(
        self,
        prompt_length,
        *,
        attribute_concept_available=False,
        attribute_concept_reason="attribute_concept_reference_unavailable",
        patch_semantic_transport_available=False,
        patch_semantic_transport_reason="patch_semantic_transport_reference_unavailable",
    ):
        if not bool(self.cfg.MONITOR.MODULE_EFFECT.ENABLE):
            return []
        prompt_length = int(prompt_length)
        prompt_available = prompt_length > 0
        prompt_parameter_available = bool(self.prompt_parameter_tracker.active)
        distributor_cfg = self.cfg.MODEL.PROMPT.DISTRIBUTOR
        distributor_available = bool(
            prompt_available
            and distributor_cfg.ENABLE
            and str(self.cfg.MODEL.PROMPT.INIT_SOURCE).lower()
            == "distributor_mean"
        )
        deep_residual_available = bool(
            prompt_available
            and distributor_cfg.ENABLE
            and distributor_cfg.DEEP_RESIDUAL.ENABLE
        )
        instance_prompt_length = (
            int(distributor_cfg.INSTANCE_TOKENS) if distributor_available else 0
        )
        domain_prompt_length = (
            int(distributor_cfg.DOMAIN_TOKENS) if distributor_available else 0
        )
        path_intervention_available = bool(
            prompt_available
            and not self.cfg.MODEL.ATTENTION_MEDIATION.ENABLE
        )
        path_not_applicable_reason = (
            "no_prompt_tokens"
            if not prompt_available
            else "attention_mediation_active"
        )
        specs = []

        def add(
            requested,
            name,
            factory,
            applicable,
            reason,
            semantics,
            diagnostic_chain,
            dynamic_context=None,
        ):
            if bool(requested):
                specs.append({
                    "name": str(name),
                    "factory": factory,
                    "applicable": bool(applicable),
                    "not_applicable_reason": None if applicable else str(reason),
                    "intervention_semantics": dict(semantics),
                    "diagnostic_chain": bool(diagnostic_chain),
                    "dynamic_context": dynamic_context,
                })

        add(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_ZERO,
            "prompt_zeroed",
            prompt_zero_intervention,
            prompt_parameter_available,
            "no_trainable_prompt_parameters",
            {
                "zeroed_object": "trainable_prompt_parameters",
                "prompt_slots_removed": False,
                "attention_route_retained": True,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_ZERO,
            "deep_prompt_residual_zeroed",
            deep_prompt_residual_zero_intervention,
            deep_residual_available,
            "deep_prompt_residual_not_active",
            {
                "changed_object": "applied_delta_prompt_all_layers",
                "replacement": "zero",
                "static_deep_prompt_preserved": True,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_SWAP,
            "deep_prompt_residual_swapped",
            deep_prompt_residual_swap_intervention,
            deep_residual_available,
            "deep_prompt_residual_not_active",
            {
                "changed_object": "shared_distributor_mu",
                "pairing": "deterministic_in_batch_derangement",
                "pairing_identity": "deep_residual_mean_swap_seed_and_sample_id",
                "static_deep_prompt_preserved": True,
                "self_pair_allowed": False,
                "swap_seed": int(
                    self.cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_SWAP_SEED
                ),
            },
            True,
            "deep_residual_swap",
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_ZERO,
            "instance_prompt_zeroed",
            instance_prompt_zero_intervention,
            bool(distributor_available and instance_prompt_length > 0),
            (
                "prompt_distributor_not_active"
                if not distributor_available
                else "no_instance_prompt_tokens"
            ),
            {
                "changed_object": "raw_instance_prompt_output",
                "replacement": "zero",
                "domain_prompt_preserved": True,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.DOMAIN_PROMPT_ZERO,
            "domain_prompt_zeroed",
            domain_prompt_zero_intervention,
            bool(distributor_available and domain_prompt_length > 0),
            (
                "prompt_distributor_not_active"
                if not distributor_available
                else "no_domain_prompt_tokens"
            ),
            {
                "changed_object": "raw_domain_prompt_output",
                "replacement": "zero",
                "instance_prompt_preserved": True,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.BOTH_PROMPT_ZERO,
            "both_prompt_zeroed",
            both_prompt_zero_intervention,
            bool(
                distributor_available
                and instance_prompt_length > 0
                and domain_prompt_length > 0
            ),
            (
                "prompt_distributor_not_active"
                if not distributor_available
                else "instance_and_domain_prompt_tokens_required"
            ),
            {
                "changed_object": "raw_instance_and_domain_prompt_output",
                "replacement": "zero",
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_SWAP,
            "instance_prompt_swapped",
            instance_prompt_swap_intervention,
            bool(distributor_available and instance_prompt_length > 0),
            (
                "prompt_distributor_not_active"
                if not distributor_available
                else "no_instance_prompt_tokens"
            ),
            {
                "changed_object": "raw_instance_prompt_output",
                "pairing": "deterministic_in_batch_derangement",
                "pairing_identity": "instance_prompt_swap_seed_and_sample_id",
                "domain_prompt_preserved": True,
                "self_pair_allowed": False,
                "swap_seed": int(
                    self.cfg.MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_SWAP_SEED
                ),
            },
            True,
            "instance_prompt_swap",
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_READ_BLOCK,
            "prompt_read_blocked",
            prompt_read_block_intervention,
            path_intervention_available,
            path_not_applicable_reason,
            {
                "blocked_edges": "prompt_query_to_patch_key",
                "affected_rows_renormalized": True,
                "prompt_patch_mass_preserved": False,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_WRITE_BLOCK,
            "prompt_write_blocked",
            prompt_write_block_intervention,
            path_intervention_available,
            path_not_applicable_reason,
            {
                "blocked_edges": "non_prompt_query_to_prompt_key",
                "non_prompt_scope": "cls_patch_and_semantic_queries",
                "affected_rows_renormalized": True,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_SELECTION_UNIFORM,
            "prompt_patch_selection_uniform",
            prompt_patch_uniform_intervention,
            bool(
                path_intervention_available
                and self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
            ),
            (
                path_not_applicable_reason
                if not path_intervention_available
                else "affinity_probe_disabled"
            ),
            {
                "changed_edges": "prompt_query_to_patch_key",
                "patch_distribution": "uniform_within_patch_subset",
                "mass_preservation_scope": "per_sample_layer_head_prompt_row",
                "prompt_patch_mass_preserved": True,
                "randomized": False,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.PATCH_PROMPT_SELECTION_UNIFORM,
            "patch_prompt_selection_uniform",
            patch_prompt_uniform_intervention,
            bool(
                path_intervention_available
                and self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
            ),
            (
                path_not_applicable_reason
                if not path_intervention_available
                else "affinity_probe_disabled"
            ),
            {
                "changed_edges": "patch_query_to_prompt_key",
                "prompt_distribution": "uniform_within_prompt_subset",
                "mass_preservation_scope": "per_sample_layer_head_patch_row",
                "patch_prompt_mass_preserved": True,
                "randomized": False,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_VALUE_GLOBALIZE,
            "prompt_patch_value_globalized",
            prompt_patch_value_globalize_intervention,
            bool(
                path_intervention_available
                and self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
            ),
            (
                path_not_applicable_reason
                if not path_intervention_available
                else "affinity_probe_disabled"
            ),
            {
                "changed_object": "patch_value_read_by_prompt_queries",
                "value_replacement": "per_sample_per_head_patch_mean",
                "attention_weights_preserved": True,
                "prompt_patch_mass_preserved": True,
                "other_query_rows_preserved": True,
            },
            True,
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_VALUE_ZERO,
            "prompt_value_zeroed",
            prompt_value_zero_intervention,
            path_intervention_available,
            path_not_applicable_reason,
            {
                "changed_object": "prompt_value_contribution_to_all_queries",
                "attention_weights_preserved": True,
                "prompt_key_competition_preserved": True,
                "value_replacement": "zero_contribution",
            },
            True,
        )
        selected_probe_layers = [
            int(item) for item in self.cfg.MONITOR.PROBE.LAYERS
        ]
        final_selected_layer = (
            max(selected_probe_layers) if selected_probe_layers else -1
        )
        requested_read_layers = list(dict.fromkeys(
            int(item)
            for item in self.cfg.MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_LAYERS
        ))
        for source_layer in requested_read_layers:
            layer_valid = bool(
                source_layer in selected_probe_layers
                and source_layer >= 0
                and source_layer < final_selected_layer
            )
            layer_reason = (
                path_not_applicable_reason
                if not path_intervention_available
                else "affinity_probe_disabled"
                if not self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
                else "source_layer_must_be_selected_and_precede_final_selected_layer"
            )

            def layer_factory(model, layer=source_layer):
                return layer_prompt_read_block_intervention(
                    model,
                    target_layer=int(layer),
                )

            add(
                self.cfg.MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_BLOCK,
                f"prompt_read_blocked_layer_{source_layer}",
                layer_factory,
                bool(
                    path_intervention_available
                    and self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
                    and layer_valid
                ),
                layer_reason,
                {
                    "blocked_edges": "prompt_query_to_patch_key",
                    "source_layer": int(source_layer),
                    "affected_rows_renormalized": True,
                    "evidence_chain": (
                        "source_prompt_read_to_downstream_prompt_state_"
                        "to_later_cls_consumption_to_final_alignment"
                    ),
                    "a1_interpretation": "persistent_prompt_memory_handoff",
                    "a2_interpretation": "layer_output_replacement_negative_control",
                },
                True,
            )
        swap_layer = int(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_LAYER
        )
        shallow_prompt = bool(
            prompt_available and not self.cfg.MODEL.PROMPT.DEEP
        )
        swap_layer_valid = bool(
            swap_layer in selected_probe_layers
            and swap_layer > 0
            and swap_layer <= final_selected_layer
        )
        if not prompt_available:
            swap_reason = "no_prompt_tokens"
        elif self.cfg.MODEL.PROMPT.DEEP:
            swap_reason = "deep_prompt_layerwise_replacement"
        elif self.cfg.MODEL.ATTENTION_MEDIATION.ENABLE:
            swap_reason = "attention_mediation_active"
        elif not self.cfg.MONITOR.PROBE.AFFINITY_ENABLE:
            swap_reason = "affinity_probe_disabled"
        else:
            swap_reason = "swap_layer_must_be_a_selected_layer_after_layer_zero"
        add(
            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP,
            "prompt_context_swapped",
            prompt_context_swap_intervention,
            bool(
                shallow_prompt
                and path_intervention_available
                and self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
                and swap_layer_valid
            ),
            swap_reason,
            {
                "changed_object": "contextualized_shallow_prompt_state",
                "swap_layer": int(swap_layer),
                "pairing": "deterministic_in_batch_derangement",
                "pairing_identity": "prompt_context_swap_seed_and_sample_id",
                "self_pair_allowed": False,
                "a1_only": True,
                "a2_applicability": "not_applicable_layerwise_replaced_prompt",
                "swap_seed": int(
                    self.cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_SEED
                ),
            },
            True,
            "prompt_swap",
        )
        concept_path_available = bool(
            path_intervention_available
            and self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
            and self.cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_ENABLE
            and attribute_concept_available
        )
        if not path_intervention_available:
            concept_not_applicable_reason = path_not_applicable_reason
        elif not self.cfg.MONITOR.PROBE.AFFINITY_ENABLE:
            concept_not_applicable_reason = "affinity_probe_disabled"
        elif not self.cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_ENABLE:
            concept_not_applicable_reason = "attribute_concept_probe_disabled"
        else:
            concept_not_applicable_reason = str(attribute_concept_reason)
        concept_semantics = {
            "changed_edges": "prompt_query_to_attribute_concept_patch_key",
            "selection_reference": "true_vs_hard_negative_attribute_margin",
            "selected_patch_ratio": float(
                self.cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_PATCH_RATIO
            ),
            "mass_preservation_scope": "per_sample_layer_head_prompt_row",
            "prompt_patch_mass_preserved": True,
            "selected_mass_redistribution": "uniform_over_unselected_patches",
        }
        add(
            self.cfg.MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_PATCH_BLOCK,
            "attribute_concept_prompt_patch_blocked",
            attribute_concept_prompt_patch_block_intervention,
            concept_path_available,
            concept_not_applicable_reason,
            {**concept_semantics, "selection_mode": "highest_concept_support"},
            True,
            "attribute_concept",
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_PATCH_BLOCK,
            "attribute_concept_random_patch_blocked",
            random_prompt_patch_block_intervention,
            concept_path_available,
            concept_not_applicable_reason,
            {
                **concept_semantics,
                "changed_edges": "prompt_query_to_random_patch_key",
                "selection_mode": "equal_count_random_control",
                "random_seed": int(
                    self.cfg.MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_RANDOM_SEED
                ),
            },
            True,
            "attribute_concept",
        )
        transport_path_available = bool(
            path_intervention_available
            and self.cfg.MONITOR.PROBE.AFFINITY_ENABLE
            and self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.ENABLE
            and patch_semantic_transport_available
        )
        if not path_intervention_available:
            transport_not_applicable_reason = path_not_applicable_reason
        elif not self.cfg.MONITOR.PROBE.AFFINITY_ENABLE:
            transport_not_applicable_reason = "affinity_probe_disabled"
        elif not self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.ENABLE:
            transport_not_applicable_reason = "patch_semantic_transport_probe_disabled"
        else:
            transport_not_applicable_reason = str(
                patch_semantic_transport_reason
            )
        transport_semantics = {
            "changed_edges": "prompt_query_to_transport_selected_patch_key",
            "selection_reference": "normal_final_layer_prediction_weighted_patch_semantic_transport",
            "cost_purpose": "local_visual_to_class_semantic_geometric_reference",
            "cost_name": str(
                self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.COST
            ).lower(),
            "selected_patch_ratio": float(
                self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO
            ),
            "selection_scope": "normal_final_layer_spatial_patch_indices_reused_across_prompt_layers",
            "mass_preservation_scope": "per_sample_layer_head_prompt_row",
            "prompt_patch_mass_preserved": True,
            "selected_mass_redistribution": "uniform_over_unselected_patches",
        }
        add(
            self.cfg.MONITOR.MODULE_EFFECT.TRANSPORT_PATCH_BLOCK,
            "transport_targeted_prompt_patch_blocked",
            transport_prompt_patch_block_intervention,
            transport_path_available,
            transport_not_applicable_reason,
            {**transport_semantics, "selection_mode": "highest_transport_support"},
            True,
            "transport",
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.TRANSPORT_PATCH_BLOCK,
            "transport_random_patch_blocked",
            transport_random_patch_block_intervention,
            transport_path_available,
            transport_not_applicable_reason,
            {
                **transport_semantics,
                "changed_edges": "prompt_query_to_equal_budget_random_patch_key",
                "selection_mode": "equal_count_random_control",
                "random_seed": int(
                    self.cfg.MONITOR.MODULE_EFFECT.TRANSPORT_RANDOM_SEED
                ),
            },
            True,
            "transport",
        )
        add(
            self.cfg.MONITOR.MODULE_EFFECT.ATTENTION_MEDIATION_GAMMA_ZERO,
            "attention_mediation_gamma_zero",
            attention_mediation_gamma_zero_intervention,
            bool(self.cfg.MODEL.ATTENTION_MEDIATION.ENABLE),
            "attention_mediation_disabled",
            {"zeroed_object": "attention_mediation_gamma_parameters"},
            False,
        )
        return specs

    def _execute_target_relevance_probe(
        self,
        probe_loader,
        source_dataset,
        candidate_class_ids,
        *,
        split,
        include_explanation_validity=True,
    ):
        model_ref = self._model_ref(self.model)
        candidate_class_ids = [int(item) for item in candidate_class_ids]
        if len(candidate_class_ids) < 2:
            raise ValueError("target relevance requires at least two candidate classes")
        global_to_local = {
            global_id: local_id for local_id, global_id in enumerate(candidate_class_ids)
        }
        prompt_length = (
            int(self.cfg.MODEL.PROMPT.NUM_TOKENS)
            if bool(self.cfg.MODEL.PROMPT.ENABLE) else 0
        )
        semantic_length = (
            int(self.cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS)
            if bool(self.cfg.MODEL.SEMANTIC_TOKENS.ENABLE) else 0
        )
        selected_layers = [int(item) for item in self.cfg.MONITOR.PROBE.LAYERS]
        explanation_cfg = self.cfg.MONITOR.PROBE.EXPLANATION_VALIDITY
        explanation_requested = bool(
            explanation_cfg.ENABLE and include_explanation_validity
        )
        explanation_layers = [int(item) for item in explanation_cfg.LAYERS]
        explanation_conditions = [
            str(item).lower() for item in explanation_cfg.CONDITIONS
        ]
        explanation_fractions = [
            float(item) for item in explanation_cfg.K_FRACTIONS
        ]
        explanation_paths = [str(item) for item in explanation_cfg.PATHS]
        if explanation_requested:
            if prompt_length <= 0:
                raise ValueError(
                    "MONITOR.PROBE.EXPLANATION_VALIDITY requires visual Prompt tokens"
                )
            if not bool(self.cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE):
                raise ValueError(
                    "EXPLANATION_VALIDITY requires TARGET_RELEVANCE.ENABLE"
                )
            if not explanation_layers:
                raise ValueError(
                    "EXPLANATION_VALIDITY.LAYERS must be explicitly declared"
                )
            if not set(explanation_layers).issubset(set(selected_layers)):
                raise ValueError(
                    "EXPLANATION_VALIDITY.LAYERS must be a subset of MONITOR.PROBE.LAYERS"
                )
            if not explanation_conditions or not explanation_fractions:
                raise ValueError(
                    "EXPLANATION_VALIDITY requires conditions and k fractions"
                )
            supported_conditions = {
                "positive", "negative", "absolute", "low", "random"
            }
            unknown_conditions = sorted(
                set(explanation_conditions).difference(supported_conditions)
            )
            if unknown_conditions:
                raise ValueError(
                    "unsupported EXPLANATION_VALIDITY conditions: "
                    + ",".join(unknown_conditions)
                )
            if not {"positive", "negative", "random"}.issubset(
                set(explanation_conditions)
            ):
                raise ValueError(
                    "EXPLANATION_VALIDITY.CONDITIONS must include positive, "
                    "negative, and random equal-budget controls"
                )
            supported_paths = {
                "cls_to_prompt",
                "prompt_to_cls",
                "prompt_to_patch",
                "patch_to_prompt",
            }
            unknown_paths = sorted(
                set(explanation_paths).difference(supported_paths)
            )
            if not explanation_paths or unknown_paths:
                raise ValueError(
                    "EXPLANATION_VALIDITY.PATHS must contain only supported "
                    "Prompt-related paths"
                )
            if len(set(explanation_layers)) != len(explanation_layers):
                raise ValueError(
                    "EXPLANATION_VALIDITY.LAYERS must not contain duplicates"
                )
            if len(set(explanation_fractions)) != len(
                explanation_fractions
            ):
                raise ValueError(
                    "EXPLANATION_VALIDITY.K_FRACTIONS must not contain duplicates"
                )
            if any(not 0.0 < value < 1.0 for value in explanation_fractions):
                raise ValueError(
                    "EXPLANATION_VALIDITY.K_FRACTIONS must be between zero and one"
                )
        explanation_accumulators: Dict[
            tuple[int, str, float], PairedModuleEffectAccumulator
        ] = {}
        explanation_metadata: Dict[tuple[int, str, float], Dict[str, float]] = (
            defaultdict(lambda: defaultdict(float))
        )
        true_accumulator = TargetRelevanceAccumulator(
            prompt_length=prompt_length,
            semantic_length=semantic_length,
            selected_layers=selected_layers,
        )
        predicted_accumulator = TargetRelevanceAccumulator(
            prompt_length=prompt_length,
            semantic_length=semantic_length,
            selected_layers=selected_layers,
            margin_metric_name="predicted_margin_mean",
        )
        equivalence_state = self._new_equivalence_state()
        available_layers = set()
        true_missing_layers = set()
        predicted_missing_layers = set()
        affinity_cfg = {
            "prompt_length": prompt_length,
            "semantic_length": semantic_length,
            "detach": True,
            "retain_attention_for_relevance": True,
            "selected_layers": selected_layers,
            "include_visual_normalizations": False,
            "block_s_to_cls": bool(self.cfg.MODEL.SEMANTIC_TOKENS.BLOCK_S_TO_CLS),
        }

        for batch_index, input_data in enumerate(probe_loader):
            inputs, targets_global, attributes = self.get_input(input_data)
            inputs = inputs.to(self.device, non_blocking=True)
            targets_global = targets_global.to(self.device, non_blocking=True)
            target_global_list = [
                int(item) for item in targets_global.detach().cpu().tolist()
            ]
            target_local = torch.as_tensor(
                [global_to_local[item] for item in target_global_list],
                device=self.device,
                dtype=torch.long,
            )
            semantics = self._prepare_semantics_for_stage(
                attributes,
                source_dataset,
                batch_size=int(inputs.shape[0]),
                is_train=False,
            )
            with torch.no_grad():
                reference_logits = model_ref(
                    inputs,
                    semantics=semantics,
                    class_ids=candidate_class_ids,
                    runtime_targets=None,
                )
            gradient_inputs = inputs.detach().requires_grad_(True)
            with torch.enable_grad():
                relevance_output = model_ref.forward_with_affinity(
                    gradient_inputs,
                    affinity_cfg,
                    semantics=semantics,
                    vis=True,
                    class_ids=candidate_class_ids,
                    runtime_targets=None,
                )
                relevance_logits, attention_layers, affinities = relevance_output
                retained_attention_layers = []
                for affinity in affinities or []:
                    retained_attention_layers.append(
                        affinity.pop("_target_relevance_attention", None)
                        if isinstance(affinity, dict)
                        else None
                    )
                if any(
                    torch.is_tensor(attention)
                    for attention in retained_attention_layers
                ):
                    attention_layers = retained_attention_layers
                self._update_equivalence_state(
                    equivalence_state,
                    reference_logits,
                    relevance_logits,
                    target_local.detach().cpu().numpy(),
                )
                reference_prediction = reference_logits.detach().argmax(dim=1)
                correct = reference_prediction.eq(target_local)
                wrong = ~correct
                reference_other = reference_logits.detach().clone()
                reference_other.scatter_(1, target_local[:, None], float("-inf"))
                competitor_local = reference_other.argmax(dim=1)
                target_score = relevance_logits.gather(1, target_local[:, None]).squeeze(1)
                competitor_score = relevance_logits.gather(
                    1, competitor_local[:, None]
                ).squeeze(1)
                true_margins = target_score - competitor_score
                true_accumulator.update_target(true_margins, correct)

                predicted_other = reference_logits.detach().clone()
                predicted_other.scatter_(
                    1, reference_prediction[:, None], float("-inf")
                )
                predicted_competitor = predicted_other.argmax(dim=1)
                predicted_score = relevance_logits.gather(
                    1, reference_prediction[:, None]
                ).squeeze(1)
                predicted_competitor_score = relevance_logits.gather(
                    1, predicted_competitor[:, None]
                ).squeeze(1)
                predicted_margins = predicted_score - predicted_competitor_score
                if bool(wrong.any()):
                    predicted_accumulator.update_target(
                        predicted_margins[wrong], correct[wrong]
                    )

                available_layers.update(range(len(attention_layers or [])))
                requested_layers = (
                    selected_layers if selected_layers
                    else list(range(len(attention_layers or [])))
                )
                gradient_inputs_by_layer = []
                true_relevance_by_layer = {}
                for layer_index in requested_layers:
                    if layer_index < 0 or layer_index >= len(attention_layers or []):
                        true_missing_layers.add(int(layer_index))
                        if bool(wrong.any()):
                            predicted_missing_layers.add(int(layer_index))
                        continue
                    attention = attention_layers[layer_index]
                    if not torch.is_tensor(attention) or not attention.requires_grad:
                        true_missing_layers.add(int(layer_index))
                        if bool(wrong.any()):
                            predicted_missing_layers.add(int(layer_index))
                        continue
                    gradient_inputs_by_layer.append((int(layer_index), attention))
                if gradient_inputs_by_layer:
                    true_gradients = torch.autograd.grad(
                        true_margins.sum(),
                        [attention for _, attention in gradient_inputs_by_layer],
                        retain_graph=bool(wrong.any()),
                        create_graph=False,
                        allow_unused=True,
                    )
                    for (layer_index, attention), gradient in zip(
                        gradient_inputs_by_layer, true_gradients
                    ):
                        if gradient is None:
                            true_missing_layers.add(int(layer_index))
                            continue
                        if not true_accumulator.update_layer(
                            layer_index, attention, gradient, correct
                        ):
                            true_missing_layers.add(int(layer_index))
                        if (
                            explanation_requested
                            and int(layer_index) in explanation_layers
                        ):
                            true_relevance_by_layer[int(layer_index)] = (
                                attention.detach().float()
                                * gradient.detach().float()
                            ).to(device="cpu")
                    if bool(wrong.any()):
                        predicted_gradients = torch.autograd.grad(
                            predicted_margins[wrong].sum(),
                            [attention for _, attention in gradient_inputs_by_layer],
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )
                        for (layer_index, attention), gradient in zip(
                            gradient_inputs_by_layer, predicted_gradients
                        ):
                            if gradient is None or not predicted_accumulator.update_layer(
                                layer_index,
                                attention[wrong],
                                gradient[wrong],
                                correct[wrong],
                            ):
                                predicted_missing_layers.add(int(layer_index))
                if explanation_requested and true_relevance_by_layer:
                    reference_logits_cpu = reference_logits.detach().cpu()
                    for layer_index, relevance in sorted(
                        true_relevance_by_layer.items()
                    ):
                        relevance_device = relevance.to(
                            device=self.device, non_blocking=True
                        )
                        for fraction_index, fraction in enumerate(
                            explanation_fractions
                        ):
                            masks, mask_metadata = build_relevance_deletion_masks(
                                relevance_device,
                                prompt_length=prompt_length,
                                semantic_length=semantic_length,
                                paths=explanation_paths,
                                conditions=explanation_conditions,
                                fraction=fraction,
                                random_seed=int(
                                    explanation_cfg.RANDOM_SEED
                                    + 1000003 * batch_index
                                    + 1009 * layer_index
                                    + fraction_index
                                ),
                            )
                            for condition, deletion_mask in masks.items():
                                metadata = mask_metadata.get(condition, {})
                                key = (
                                    int(layer_index),
                                    str(condition),
                                    float(fraction),
                                )
                                accumulator = explanation_accumulators.setdefault(
                                    key,
                                    PairedModuleEffectAccumulator(
                                        candidate_class_ids,
                                        source_dataset.seen_classes,
                                    ),
                                )
                                for name, value in metadata.items():
                                    explanation_metadata[key][name] += float(value)
                                if metadata.get("selected_edge_count", 0.0) <= 0.0:
                                    accumulator.update(
                                        reference_logits_cpu,
                                        reference_logits_cpu,
                                        target_local.detach().cpu().numpy(),
                                    )
                                    explanation_metadata[key][
                                        "runtime_sample_count"
                                    ] += float(reference_logits_cpu.shape[0])
                                    continue
                                explanation_metadata[key][
                                    "expected_runtime_sample_count"
                                ] += float(reference_logits_cpu.shape[0])
                                with relevance_edge_delete_intervention(
                                    model_ref,
                                    target_layer=layer_index,
                                    deletion_mask=deletion_mask,
                                ):
                                    with torch.no_grad():
                                        changed_logits = model_ref(
                                            inputs,
                                            semantics=semantics,
                                            class_ids=candidate_class_ids,
                                            runtime_targets=None,
                                        )
                                runtime_sample_count = 0
                                for module in model_ref.modules():
                                    runtime_stats = getattr(
                                        module,
                                        "_last_prompt_path_intervention_stats",
                                        None,
                                    )
                                    if not isinstance(runtime_stats, dict):
                                        continue
                                    applied = runtime_stats.get(
                                        "relevance_delete_applied"
                                    )
                                    if not torch.is_tensor(applied):
                                        continue
                                    runtime_sample_count = max(
                                        runtime_sample_count,
                                        int(applied.numel()),
                                    )
                                    for name in (
                                        "relevance_delete_mass",
                                        "relevance_delete_edge_ratio",
                                        "relevance_delete_row_mass_abs_error",
                                    ):
                                        values = runtime_stats.get(name)
                                        if torch.is_tensor(values):
                                            explanation_metadata[key][
                                                f"{name}_sum"
                                            ] += float(
                                                values.detach().float().sum().item()
                                            )
                                explanation_metadata[key][
                                    "runtime_sample_count"
                                ] += float(runtime_sample_count)
                                accumulator.update(
                                    reference_logits_cpu,
                                    changed_logits.detach().cpu(),
                                    target_local.detach().cpu().numpy(),
                                )
                                del changed_logits, deletion_mask
                            del masks, mask_metadata
                        del relevance_device
                    del reference_logits_cpu, true_relevance_by_layer

            model_ref.clear_runtime_state()
            del reference_logits, relevance_output, relevance_logits
            del attention_layers, affinities, gradient_inputs, inputs
            del targets_global, target_local, semantics, true_margins, correct
            del predicted_margins, reference_prediction, wrong

        equivalence = self._finalize_equivalence_state(equivalence_state)
        equivalence_pass = bool(
            equivalence["logit_max_abs_diff"]
            <= float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL)
            and equivalence["true_margin_abs_diff"]
            <= float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_MARGIN_ATOL)
            and equivalence["prediction_flip_rate"] == 0.0
        )
        equivalence["target_relevance_forward_equivalence_pass"] = float(
            equivalence_pass
        )
        required_layers = set(selected_layers) if selected_layers else available_layers
        true_missing_layers.update(
            required_layers.difference(true_accumulator.observed_layers)
        )
        predicted_target_summary = predicted_accumulator.finalize_target()
        predicted_applicable = bool(
            predicted_target_summary.get("all.sample_count", 0.0) > 0.0
        )
        if predicted_applicable:
            predicted_missing_layers.update(
                required_layers.difference(predicted_accumulator.observed_layers)
            )
        true_failure_reasons = []
        if not equivalence_pass:
            true_failure_reasons.append(
                "gradient-enabled affinity forward is not equivalent to ordinary forward"
            )
        if true_missing_layers:
            true_failure_reasons.append(
                "target relevance was not observed for layers: "
                + ",".join(str(item) for item in sorted(true_missing_layers))
            )
        if not true_accumulator.observed_layers:
            true_failure_reasons.append("no target-relevance layer was observed")
        true_valid = bool(
            equivalence_pass
            and true_accumulator.observed_layers
            and not true_missing_layers
        )
        predicted_failure_reasons = []
        if predicted_applicable and not equivalence_pass:
            predicted_failure_reasons.append(
                "gradient-enabled affinity forward is not equivalent to ordinary forward"
            )
        if predicted_applicable and predicted_missing_layers:
            predicted_failure_reasons.append(
                "predicted-target relevance was not observed for layers: "
                + ",".join(str(item) for item in sorted(predicted_missing_layers))
            )
        if predicted_applicable and not predicted_accumulator.observed_layers:
            predicted_failure_reasons.append(
                "no predicted-target relevance layer was observed for wrong samples"
            )
        predicted_valid = bool(
            predicted_applicable
            and equivalence_pass
            and predicted_accumulator.observed_layers
            and not predicted_missing_layers
        )
        valid = bool(true_valid and (predicted_valid or not predicted_applicable))
        failure_reasons = list(true_failure_reasons)
        failure_reasons.extend(predicted_failure_reasons)
        true_result = {
            "applicability": "applicable",
            "sample_scope": "all_reference_samples",
            "valid": true_valid,
            "failure_reasons": true_failure_reasons,
            "observed_layers": sorted(true_accumulator.observed_layers),
            "missing_layers": sorted(true_missing_layers),
            "target_summary": true_accumulator.finalize_target(),
            "by_layer": true_accumulator.finalize_by_layer(),
        }
        predicted_result = {
            "applicability": (
                "applicable" if predicted_applicable else "not_applicable_no_wrong_samples"
            ),
            "sample_scope": "reference_wrong_samples_only",
            "valid": predicted_valid if predicted_applicable else None,
            "failure_reasons": predicted_failure_reasons,
            "observed_layers": sorted(predicted_accumulator.observed_layers),
            "missing_layers": (
                sorted(predicted_missing_layers) if predicted_applicable else []
            ),
            "target_summary": predicted_target_summary,
            "by_layer": predicted_accumulator.finalize_by_layer(),
        }
        explanation_effects = {}
        for key, accumulator in sorted(explanation_accumulators.items()):
            effect = accumulator.finalize()
            metadata = explanation_metadata.get(key, {})
            selected_count = float(metadata.get("selected_edge_count", 0.0))
            eligible_count = float(metadata.get("eligible_edge_count", 0.0))
            effect["summary"].update({
                "selected_edge_count": selected_count,
                "eligible_edge_count": eligible_count,
                "actual_deletion_ratio": float(
                    selected_count / max(eligible_count, 1.0)
                ),
            })
            runtime_sample_count = float(
                metadata.get("runtime_sample_count", 0.0)
            )
            if runtime_sample_count > 0.0:
                for name in (
                    "relevance_delete_mass",
                    "relevance_delete_edge_ratio",
                    "relevance_delete_row_mass_abs_error",
                ):
                    effect["summary"][name] = float(
                        metadata.get(f"{name}_sum", 0.0)
                        / runtime_sample_count
                    )
            explanation_effects[key] = effect
        explanation_curves = summarize_deletion_curves(explanation_effects)
        expected_explanation_keys = {
            (int(layer), str(condition), float(fraction))
            for layer in explanation_layers
            for condition in explanation_conditions
            for fraction in explanation_fractions
        }
        missing_budget_keys = {
                key
                for key, metadata in explanation_metadata.items()
                if float(metadata.get("selected_edge_count", 0.0)) <= 0.0
        }
        runtime_missing_keys = {
            key
            for key, metadata in explanation_metadata.items()
            if float(metadata.get("expected_runtime_sample_count", 0.0)) > 0.0
            and float(metadata.get("runtime_sample_count", 0.0))
            < float(metadata.get("expected_runtime_sample_count", 0.0))
        }
        unobserved_explanation_keys = expected_explanation_keys.difference(
            explanation_effects
        )
        missing_explanation_keys = sorted(
            unobserved_explanation_keys
            .union(missing_budget_keys)
            .union(runtime_missing_keys)
        )
        explanation_failure_reasons = []
        if explanation_requested and not true_valid:
            explanation_failure_reasons.append(
                "target relevance reference was invalid"
            )
        if explanation_requested and (
            unobserved_explanation_keys or missing_budget_keys
        ):
            explanation_failure_reasons.append(
                "some declared layer/condition/fraction cells had no sign-valid deletion budget"
            )
        if explanation_requested and runtime_missing_keys:
            explanation_failure_reasons.append(
                "relevance deletion did not execute at the target Attention layer for all expected samples"
            )
        directional_checks = {}
        for layer, comparison in explanation_curves[
            "comparisons_by_layer"
        ].items():
            if "positive_minus_random_margin_drop_auc" in comparison:
                directional_checks[
                    f"layer_{layer}_positive_more_harmful_than_random"
                ] = bool(
                    comparison["positive_minus_random_margin_drop_auc"] > 0.0
                )
            if "negative_minus_random_margin_drop_auc" in comparison:
                directional_checks[
                    f"layer_{layer}_negative_less_harmful_than_random"
                ] = bool(
                    comparison["negative_minus_random_margin_drop_auc"] < 0.0
                )
        explanation_valid = bool(
            explanation_requested
            and true_valid
            and explanation_effects
            and not missing_explanation_keys
        )
        explanation_validity = {
            **explanation_curves,
            "requested": explanation_requested,
            "applicability": (
                "applicable" if explanation_requested else "not_requested"
            ),
            "valid": explanation_valid if explanation_requested else None,
            "directional_support_pass": (
                bool(directional_checks)
                and all(directional_checks.values())
                if explanation_requested
                else None
            ),
            "directional_checks": directional_checks,
            "failure_reasons": explanation_failure_reasons,
            "layers": explanation_layers,
            "conditions": explanation_conditions,
            "k_fractions": explanation_fractions,
            "paths": explanation_paths,
            "missing_cells": [
                {
                    "layer": layer,
                    "condition": condition,
                    "fraction": fraction,
                }
                for layer, condition, fraction in missing_explanation_keys
            ],
            "runtime_missing_cells": [
                {
                    "layer": layer,
                    "condition": condition,
                    "fraction": fraction,
                }
                for layer, condition, fraction in sorted(runtime_missing_keys)
            ],
            "deletion_semantics": "post_softmax_zero_then_row_renormalize",
        }
        return {
            "format": "target_relevance_reference_v3",
            "split": str(split),
            "valid": valid,
            "failure_reasons": failure_reasons,
            "objective": {
                "name": "true_class_margin",
                "competitor": "strongest_non_target_from_reference_forward",
                "signed_relevance": True,
                "method": "attention_probability_times_target_gradient",
                "cross_layer_rollout": False,
            },
            "objectives": {
                "true_class_margin": {
                    "competitor": "strongest_non_true_class_from_reference_forward",
                    "sample_scope": "all_reference_samples",
                },
                "predicted_class_margin": {
                    "competitor": "strongest_non_predicted_class_from_reference_forward",
                    "sample_scope": "reference_wrong_samples_only",
                },
            },
            "execution": {
                "model_eval": True,
                "torch_grad_enabled": True,
                "input_requires_grad": True,
                "parameter_grad_accumulation": False,
                "attention_capture": "graph_connected_post_softmax_affinity_private_key",
                "attention_capture_independent_of_visualization": True,
                "available_layers": sorted(available_layers),
                "aggregation_dtype": "float32",
                "batch_size": int(self.cfg.MONITOR.PROBE.TARGET_RELEVANCE.BATCH_SIZE),
            },
            "selected_layers": selected_layers,
            "observed_layers": true_result["observed_layers"],
            "missing_layers": true_result["missing_layers"],
            "equivalence": equivalence,
            "target_summary": true_result["target_summary"],
            "by_layer": true_result["by_layer"],
            "objective_results": {
                "true_class_margin": true_result,
                "predicted_class_margin": predicted_result,
            },
            "explanation_validity": explanation_validity,
            "storage_mode": "aggregate_only",
        }

    @torch.no_grad()
    def _execute_bayesian_object_selection_probe(
        self,
        probe_loader,
        source_dataset,
        candidate_class_ids,
        *,
        split,
    ):
        model_ref = self._model_ref(self.model)
        cfg = self.cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION
        candidate_class_ids = [int(item) for item in candidate_class_ids]
        prompt_enabled = bool(self.cfg.MODEL.PROMPT.ENABLE)
        prompt_length = (
            int(self.cfg.MODEL.PROMPT.NUM_TOKENS) if prompt_enabled else 0
        )
        distributor_active = bool(
            prompt_enabled
            and self.cfg.MODEL.PROMPT.DISTRIBUTOR.ENABLE
            and str(self.cfg.MODEL.PROMPT.INIT_SOURCE).lower()
            == "distributor_mean"
        )
        selected_layers = [int(item) for item in cfg.LAYERS]
        if not selected_layers:
            selected_layers = (
                [int(item) for item in self.cfg.MONITOR.PROBE.LAYERS]
                if bool(self.cfg.MODEL.PROMPT.DEEP)
                else [0]
            )
        selected_layers = list(dict.fromkeys(selected_layers))
        registered_objects = (
            model_ref.get_bayesian_candidate_registry()
            if hasattr(model_ref, "get_bayesian_candidate_registry")
            else []
        )
        registry = build_candidate_registry(
            requested_candidates=list(cfg.CANDIDATE_SPACES),
            requested_auxiliary_views=list(cfg.AUXILIARY_VIEWS),
            prompt_enabled=prompt_enabled,
            distributor_active=distributor_active,
            prompt_deep=bool(self.cfg.MODEL.PROMPT.DEEP),
            registered_objects=registered_objects,
        )
        base_result = {
            "format": "bayesian_object_selection_probe_v1",
            "requested": True,
            "split": str(split),
            "registry": registry,
            "execution_contract": {
                "same_checkpoint": True,
                "same_probe_manifest": True,
                "same_sample_id_order": True,
                "same_candidate_class_ids": True,
                "reference": str(cfg.HIERARCHY_REFERENCE),
                "variant_source": str(cfg.HIERARCHY_VARIANT_SOURCE),
                "upstream_only_perturbation": True,
                "posterior_interpretation_allowed": False,
                "sample_vectors_persisted": False,
                "variant_storage_device": "cpu",
                "gpu_live_variant_policy": "reference_plus_current_variant",
            },
        }
        if not prompt_enabled:
            return {
                **base_result,
                "applicability": "not_applicable_prompt_not_enabled",
                "observed": False,
                "valid": None,
                "failure_reason": "prompt_not_enabled",
            }
        if str(cfg.PERTURBATION_MODE).lower() != "normalized_direction":
            raise ValueError(
                "BAYESIAN_OBJECT_SELECTION currently requires PERTURBATION_MODE='normalized_direction'"
            )
        if str(cfg.HIERARCHY_VARIANT_SOURCE).lower() != "controlled_perturbation":
            raise ValueError(
                "posterior_sample is unavailable before a Bayesian object and posterior are defined"
            )
        if str(cfg.HIERARCHY_REFERENCE).lower() != "unperturbed_same_checkpoint":
            raise ValueError(
                "BAYESIAN_OBJECT_SELECTION requires HIERARCHY_REFERENCE='unperturbed_same_checkpoint'"
            )
        if not bool(cfg.HIERARCHY_TRACE_ENABLE):
            raise ValueError(
                "BAYESIAN_OBJECT_SELECTION v1 requires HIERARCHY_TRACE_ENABLE=true when the master switch is enabled"
            )
        if bool(cfg.EXPORT_SAMPLE_VECTORS):
            raise ValueError(
                "BAYESIAN_OBJECT_SELECTION sample-vector export is not enabled in aggregate-only fixed probes"
            )
        scales = [float(item) for item in cfg.PERTURBATION_SCALES]
        if not scales or any((not math.isfinite(item)) or item <= 0.0 for item in scales):
            raise ValueError(
                "BAYESIAN_OBJECT_SELECTION.PERTURBATION_SCALES must contain positive finite values"
            )
        direction_count = int(cfg.DIRECTION_COUNT)
        if direction_count <= 0:
            raise ValueError(
                "BAYESIAN_OBJECT_SELECTION.DIRECTION_COUNT must be positive when enabled"
            )
        base_variant_specs = [
            {
                "variant_id": f"direction_{direction_id:03d}_scale_{scale:.8g}",
                "perturbation_id": f"normalized_direction/{direction_id}/{scale:.8g}",
                "direction_id": int(direction_id),
                "scale": float(scale),
                "perturbation_seed": int(cfg.RANDOM_SEED),
            }
            for direction_id in range(direction_count)
            for scale in scales
        ]
        variant_limit = int(cfg.HIERARCHY_VARIANT_COUNT)
        if variant_limit > 0:
            base_variant_specs = base_variant_specs[:variant_limit]
        if distributor_active:
            variant_specs = list(base_variant_specs)
        else:
            variant_specs = [
                {
                    **spec,
                    "variant_id": f"layer_{layer_id}/{spec['variant_id']}",
                    "perturbation_id": f"layer_{layer_id}/{spec['perturbation_id']}",
                    "target_layer": int(layer_id),
                }
                for layer_id in selected_layers
                for spec in base_variant_specs
            ]
        if not variant_specs:
            raise ValueError("BAYESIAN_OBJECT_SELECTION produced no controlled variants")
        use_layer_interface_map = (
            not distributor_active
            and bool(self.cfg.MODEL.PROMPT.DEEP)
            and hasattr(model_ref, "get_runtime_layer_prompt_trace")
        )
        accumulator = BayesianHierarchyTraceAccumulator(
            candidate_class_ids,
            bootstrap_samples=int(cfg.BOOTSTRAP_SAMPLES),
            random_seed=int(cfg.RANDOM_SEED),
            collapse_relative_threshold=float(cfg.COLLAPSE_RELATIVE_THRESHOLD),
            distance_eps=float(cfg.DISTANCE_EPS),
            # Cross-layer variants share one unperturbed checkpoint reference.
            # Reference-only distances avoid quadratic comparisons between
            # interventions on different layer identities.
            distance_pairing_mode=(
                "reference_only" if use_layer_interface_map else "all_pairs"
            ),
        )
        layer_accumulators = {
            int(layer_id): BayesianHierarchyTraceAccumulator(
                candidate_class_ids,
                bootstrap_samples=int(cfg.BOOTSTRAP_SAMPLES),
                random_seed=int(cfg.RANDOM_SEED) + int(layer_id),
                collapse_relative_threshold=float(cfg.COLLAPSE_RELATIVE_THRESHOLD),
                distance_eps=float(cfg.DISTANCE_EPS),
            )
            for layer_id in selected_layers
        } if use_layer_interface_map else {}
        global_to_local = {
            global_id: local_id
            for local_id, global_id in enumerate(candidate_class_ids)
        }
        source_distance_values = defaultdict(list)
        semantic_view_requested = (
            "semantic_aligned_contextualized_prompt"
            in {str(item) for item in cfg.AUXILIARY_VIEWS}
        )

        def snapshot(inputs, semantics):
            logits = model_ref(
                inputs,
                semantics=semantics,
                class_ids=candidate_class_ids,
                runtime_targets=None,
            )
            classifier_stats = model_ref.get_runtime_classifier_stats()
            if not isinstance(classifier_stats, dict):
                raise RuntimeError(
                    "Bayesian object selection requires runtime classifier statistics"
                )
            cls_tensor = classifier_stats.get("visual_input")
            semantic_tensor = classifier_stats.get("semantic_repr")
            injected = model_ref.get_runtime_injected_prompt_tokens()
            token_sequence = model_ref.get_runtime_token_sequence()
            runtime_layer_trace = (
                model_ref.get_runtime_layer_prompt_trace()
                if hasattr(model_ref, "get_runtime_layer_prompt_trace")
                else None
            )
            if not all(
                torch.is_tensor(item)
                for item in (cls_tensor, injected, token_sequence)
            ):
                raise RuntimeError(
                    "Bayesian object selection requires injected Prompt, token sequence, and CLS runtime tensors"
                )
            if int(injected.shape[1]) != prompt_length:
                raise RuntimeError("Injected Prompt length does not match configured identity")
            layer_prompt_trace = {}
            if isinstance(runtime_layer_trace, (list, tuple)):
                for item in runtime_layer_trace:
                    if not isinstance(item, dict):
                        continue
                    layer_id = int(item.get("layer_id", -1))
                    layer_injected = item.get("injected_prompt")
                    layer_contextualized = item.get("contextualized_prompt")
                    if layer_id < 0 or not all(
                        torch.is_tensor(value)
                        for value in (layer_injected, layer_contextualized)
                    ):
                        continue
                    layer_prompt_trace[layer_id] = {
                        "injected_prompt": layer_injected,
                        "contextualized_prompt": layer_contextualized,
                    }
            effective_layers = [
                layer_id
                for layer_id in selected_layers
                if layer_id in layer_prompt_trace
            ]
            if effective_layers:
                injected = torch.cat(
                    [
                        layer_prompt_trace[layer_id]["injected_prompt"]
                        for layer_id in effective_layers
                    ],
                    dim=1,
                )
                contextualized = torch.cat(
                    [
                        layer_prompt_trace[layer_id]["contextualized_prompt"]
                        for layer_id in effective_layers
                    ],
                    dim=1,
                )
            else:
                contextualized = token_sequence[:, 1 : 1 + prompt_length, :]
            semantic_aligned = None
            pooled_prompt = contextualized.float().mean(dim=1)
            if semantic_view_requested and torch.is_tensor(semantic_tensor):
                semantic_float = semantic_tensor.float()
                pooled_normalized = torch.nn.functional.normalize(
                    pooled_prompt, dim=-1
                )
                if semantic_float.dim() == 2 and int(semantic_float.shape[-1]) == int(
                    pooled_prompt.shape[-1]
                ):
                    semantic_aligned = pooled_normalized @ torch.nn.functional.normalize(
                        semantic_float, dim=-1
                    ).transpose(0, 1)
                elif (
                    semantic_float.dim() == 3
                    and int(semantic_float.shape[0]) == int(pooled_prompt.shape[0])
                    and int(semantic_float.shape[-1]) == int(pooled_prompt.shape[-1])
                ):
                    semantic_aligned = torch.einsum(
                        "bd,bcd->bc",
                        pooled_normalized,
                        torch.nn.functional.normalize(semantic_float, dim=-1),
                    )
            prompt_stats = model_ref.get_runtime_prompt_distribution_stats()
            return {
                "logits": logits.detach(),
                "injected_prompt": injected.detach(),
                "contextualized_prompt": contextualized.detach(),
                "semantic_aligned_contextualized_prompt": (
                    semantic_aligned.detach()
                    if torch.is_tensor(semantic_aligned)
                    else None
                ),
                "cls": cls_tensor.detach(),
                "prompt_distribution_stats": (
                    prompt_stats if isinstance(prompt_stats, dict) else {}
                ),
                "layer_prompt_trace": layer_prompt_trace,
                "observed_prompt_layers": effective_layers,
            }

        def detached_cpu(value):
            return (
                value.detach().to(device="cpu")
                if torch.is_tensor(value)
                else None
            )

        for input_data in probe_loader:
            inputs, targets_global, attributes = self.get_input(input_data)
            inputs = inputs.to(self.device, non_blocking=True)
            targets_global = targets_global.to(self.device, non_blocking=True)
            sample_ids = input_data.get("sample_id")
            if sample_ids is None:
                raise RuntimeError(
                    "Bayesian object selection requires fixed-probe sample_id"
                )
            sample_ids = [str(item) for item in list(sample_ids)]
            target_global_list = [
                int(item) for item in targets_global.detach().cpu().tolist()
            ]
            targets_local = [global_to_local[item] for item in target_global_list]
            semantics = self._prepare_semantics_for_stage(
                attributes,
                source_dataset,
                batch_size=int(inputs.shape[0]),
                is_train=False,
            )
            batch_size = int(inputs.shape[0])
            if distributor_active:
                discovery = snapshot(inputs, semantics)
                discovery_stats = discovery["prompt_distribution_stats"]
                reference_mu = discovery_stats.get("mu")
                reference_logvar = discovery_stats.get("logvar")
                if not torch.is_tensor(reference_mu) or not torch.is_tensor(
                    reference_logvar
                ):
                    raise RuntimeError(
                        "Active Prompt Distributor did not expose aligned mu/logvar"
                    )
                reference_mu = reference_mu.detach().clone()
                reference_logvar = reference_logvar.detach().clone()
                instance_tokens = int(
                    self.cfg.MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS
                )
                fixed_eps = reference_mu.new_zeros(
                    (batch_size, instance_tokens, int(reference_mu.shape[-1]))
                )
                model_ref.set_runtime_prompt_distribution_override(
                    reference_mu, reference_logvar, eps=fixed_eps
                )
                reference = snapshot(inputs, semantics)
                reference_source = reference_mu
                del discovery, discovery_stats
            else:
                reference_source = static_prompt_vector(
                    model_ref, selected_layers
                ).unsqueeze(0).expand(batch_size, -1)
                reference = snapshot(inputs, semantics)
            layer_variants = {}
            layer_reference_sources = {}
            if layer_accumulators:
                for layer_id in selected_layers:
                    layer_trace = reference["layer_prompt_trace"].get(layer_id)
                    if not isinstance(layer_trace, dict):
                        raise RuntimeError(
                            f"Bayesian object selection did not observe Prompt layer {layer_id}"
                        )
                    layer_source = static_prompt_vector(
                        model_ref, [layer_id]
                    ).unsqueeze(0).expand(batch_size, -1)
                    layer_reference_sources[layer_id] = layer_source
                    layer_variants[layer_id] = [
                        {
                            "variant_id": "reference",
                            "sample_ids": sample_ids,
                            "source_object": detached_cpu(layer_source),
                            "injected_prompt": detached_cpu(
                                layer_trace["injected_prompt"]
                            ),
                            "contextualized_prompt": detached_cpu(
                                layer_trace["contextualized_prompt"]
                            ),
                            "cls_effect": torch.zeros_like(
                                reference["cls"], device="cpu"
                            ),
                            "logit_effect": torch.zeros_like(
                                reference["logits"], device="cpu"
                            ),
                            "logits": detached_cpu(reference["logits"]),
                        }
                    ]
            variants = [
                {
                    "variant_id": "reference",
                    "sample_ids": sample_ids,
                    "source_object": detached_cpu(reference_source),
                    "injected_prompt": detached_cpu(
                        reference["injected_prompt"]
                    ),
                    "contextualized_prompt": detached_cpu(
                        reference["contextualized_prompt"]
                    ),
                    "semantic_aligned_contextualized_prompt": detached_cpu(
                        reference[
                            "semantic_aligned_contextualized_prompt"
                        ]
                    ),
                    "cls_effect": torch.zeros_like(
                        reference["cls"], device="cpu"
                    ),
                    "logit_effect": torch.zeros_like(
                        reference["logits"], device="cpu"
                    ),
                    "logits": detached_cpu(reference["logits"]),
                }
            ]
            for spec in variant_specs:
                if distributor_active:
                    changed_mu = normalized_latent_direction(
                        reference_mu,
                        direction_id=spec["direction_id"],
                        scale=spec["scale"],
                        seed=spec["perturbation_seed"],
                    )
                    model_ref.set_runtime_prompt_distribution_override(
                        changed_mu, reference_logvar, eps=fixed_eps
                    )
                    changed = snapshot(inputs, semantics)
                    changed_source = changed_mu
                else:
                    target_layer = int(spec["target_layer"])
                    with StaticPromptPerturbation(
                        model_ref,
                        direction_id=spec["direction_id"],
                        scale=spec["scale"],
                        seed=spec["perturbation_seed"],
                        selected_layers=[target_layer],
                    ) as intervention:
                        changed_source = static_prompt_vector(
                            model_ref, selected_layers
                        ).unsqueeze(0).expand(batch_size, -1)
                        changed_layer_source = static_prompt_vector(
                            model_ref, [target_layer]
                        ).unsqueeze(0).expand(batch_size, -1)
                        changed = snapshot(inputs, semantics)
                source_rms = (
                    (changed_source.detach().float() - reference_source.detach().float())
                    .reshape(batch_size, -1)
                    .pow(2)
                    .mean(dim=-1)
                    .sqrt()
                )
                if not bool(
                    (source_rms > float(cfg.DISTANCE_EPS)).all().item()
                ):
                    raise RuntimeError(
                        f"Controlled Prompt variant {spec['variant_id']} did not change its source object"
                    )
                source_distance_values[spec["variant_id"]].extend(
                    float(item) for item in source_rms.detach().cpu().tolist()
                )
                variants.append(
                    {
                        "variant_id": spec["variant_id"],
                        "sample_ids": sample_ids,
                        "source_object": detached_cpu(changed_source),
                        "injected_prompt": detached_cpu(
                            changed["injected_prompt"]
                        ),
                        "contextualized_prompt": detached_cpu(
                            changed["contextualized_prompt"]
                        ),
                        "semantic_aligned_contextualized_prompt": detached_cpu(
                            changed[
                                "semantic_aligned_contextualized_prompt"
                            ]
                        ),
                        "cls_effect": detached_cpu(
                            changed["cls"] - reference["cls"]
                        ),
                        "logit_effect": detached_cpu(
                            changed["logits"] - reference["logits"]
                        ),
                        "logits": detached_cpu(changed["logits"]),
                    }
                )
                if layer_accumulators:
                    changed_layer_trace = changed["layer_prompt_trace"].get(
                        target_layer
                    )
                    if not isinstance(changed_layer_trace, dict):
                        raise RuntimeError(
                            f"Changed forward did not observe Prompt layer {target_layer}"
                        )
                    layer_variants[target_layer].append(
                        {
                            "variant_id": spec["variant_id"],
                            "sample_ids": sample_ids,
                            "source_object": detached_cpu(changed_layer_source),
                            "injected_prompt": detached_cpu(
                                changed_layer_trace["injected_prompt"]
                            ),
                            "contextualized_prompt": detached_cpu(
                                changed_layer_trace["contextualized_prompt"]
                            ),
                            "cls_effect": detached_cpu(
                                changed["cls"] - reference["cls"]
                            ),
                            "logit_effect": detached_cpu(
                                changed["logits"] - reference["logits"]
                            ),
                            "logits": detached_cpu(changed["logits"]),
                        }
                    )
                model_ref.clear_runtime_state()
                del changed, changed_source
                if distributor_active:
                    del changed_mu
            accumulator.update(
                sample_ids=sample_ids,
                targets_local=targets_local,
                variants=variants,
            )
            for layer_id, layer_accumulator in layer_accumulators.items():
                layer_accumulator.update(
                    sample_ids=sample_ids,
                    targets_local=targets_local,
                    variants=layer_variants[layer_id],
                )
            model_ref.clear_runtime_prompt_distribution_override()
            model_ref.clear_runtime_state()
            del inputs, targets_global, semantics, variants, reference
            del reference_source
            if distributor_active:
                del reference_mu, reference_logvar, fixed_eps

        hierarchy_trace = accumulator.finalize()
        layer_interface_map = {
            f"layer_{layer_id}": {
                **layer_accumulator.finalize(),
                "layer_id": int(layer_id),
                "source_object_name": "static_prompt_parameter",
            }
            for layer_id, layer_accumulator in layer_accumulators.items()
        }
        distance_correspondence = hierarchy_trace.get(
            "distance_correspondence", {}
        )
        if distributor_active:
            for alias, canonical in {
                "latent_to_injected_prompt_distance_spearman": "source_to_injected_distance_spearman",
                "latent_to_contextual_prompt_distance_spearman": "source_to_contextualized_distance_spearman",
                "latent_to_cls_delta_distance_spearman": "source_to_cls_effect_distance_spearman",
                "latent_to_logit_delta_distance_spearman": "source_to_logit_effect_distance_spearman",
            }.items():
                if canonical in distance_correspondence:
                    distance_correspondence[alias] = dict(
                        distance_correspondence[canonical]
                    )
        contextual_alias = "contextualized_prompt_to_logit_delta_distance_spearman"
        if contextual_alias in distance_correspondence:
            distance_correspondence[
                "contextual_prompt_to_logit_delta_distance_spearman"
            ] = dict(distance_correspondence[contextual_alias])
        hierarchy_trace["variant_protocol"] = variant_specs
        hierarchy_trace["source_object_name"] = (
            "raw_latent" if distributor_active else "static_prompt_parameter"
        )
        hierarchy_trace["prompt_type_contract"] = (
            "instance_and_domain" if distributor_active else "visual"
        )
        hierarchy_trace["selected_layers"] = selected_layers
        hierarchy_trace["layer_interface_map"] = layer_interface_map
        hierarchy_trace["source_distance_by_variant"] = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": int(len(values)),
            }
            for name, values in sorted(source_distance_values.items())
            if values
        }
        observed_stages = set(hierarchy_trace.get("observed_stages", []))
        stage_by_candidate = {
            "raw_latent": "source_object" if distributor_active else None,
            "injected_prompt": "injected_prompt",
            "contextualized_prompt": "contextualized_prompt",
            "cls_effect": "cls_effect",
            "logit_effect": "logit_effect",
        }
        for name, state in registry["candidate_spaces"].items():
            stage = stage_by_candidate.get(name)
            if stage is None and "/layer_" in name:
                prefix, layer_text = name.rsplit("/layer_", 1)
                try:
                    layer_key = f"layer_{int(layer_text)}"
                except ValueError:
                    layer_key = None
                layer_trace = layer_interface_map.get(layer_key, {})
                layer_observed = set(layer_trace.get("observed_stages", []))
                stage = (
                    "injected_prompt"
                    if prefix in {"injected_prompt", "delta_prompt"}
                    else "contextualized_prompt"
                    if prefix == "contextualized_prompt"
                    else None
                )
                observed_here = bool(stage and stage in layer_observed)
            else:
                observed_here = bool(stage in observed_stages)
            if state["requested"] and state["applicable"]:
                state["observed"] = observed_here
                state["valid"] = observed_here
                state["failure_reason"] = (
                    None if state["observed"] else "runtime_stage_not_observed"
                )
        auxiliary_stage = "semantic_aligned_contextualized_prompt"
        semantic_state = registry["auxiliary_views"][auxiliary_stage]
        if semantic_state["requested"] and semantic_state["applicable"]:
            semantic_state["observed"] = auxiliary_stage in observed_stages
            semantic_state["valid"] = semantic_state["observed"]
            semantic_state["failure_reason"] = (
                None
                if semantic_state["observed"]
                else "not_applicable_no_shared_semantic_space"
            )
        decision_state = registry["auxiliary_views"]["decision_margin_effect"]
        if decision_state["requested"] and decision_state["applicable"]:
            decision_count = hierarchy_trace.get("prediction_effect", {}).get(
                "true_vs_hard_negative_margin_effect", {}
            ).get("count", 0)
            decision_state["observed"] = int(decision_count) > 0
            decision_state["valid"] = decision_state["observed"]
            decision_state["failure_reason"] = (
                None if decision_state["observed"] else "margin_effect_not_observed"
            )
        functional_geometry = {
            "format": "prompt_functional_geometry_v1",
            "valid": bool(hierarchy_trace.get("valid", False)),
            "perturbation_mode": str(cfg.PERTURBATION_MODE),
            "reference": str(cfg.HIERARCHY_REFERENCE),
            "distance_normalization": "root_mean_square_per_dimension",
            "distance_pairing_mode": hierarchy_trace.get(
                "distance_pairing_mode", "all_pairs"
            ),
            "bootstrap_unit": "sample_variant_distance_pair",
            "source_object_name": hierarchy_trace["source_object_name"],
            "distance_correspondence": hierarchy_trace.get(
                "distance_correspondence", {}
            ),
            "layer_interface_map": layer_interface_map,
            "class_structure": hierarchy_trace.get("class_structure", {}),
            "scale_curve": {
                "scales": scales,
                "directions": direction_count,
                "variant_count": len(variant_specs),
                "source_distance_by_variant": hierarchy_trace[
                    "source_distance_by_variant"
                ],
            },
            "direction_effect_norm": {
                "cls_effect": hierarchy_trace.get(
                    "reference_distance_by_stage", {}
                ).get("cls_effect", {}),
                "logit_effect": hierarchy_trace.get(
                    "reference_distance_by_stage", {}
                ).get("logit_effect", {}),
            },
            "functional_null_direction": hierarchy_trace.get(
                "perturbation_functional_null_direction_ratio", {}
            ),
            "local_jacobian": {
                "status": "deferred_high_cost",
                "reason": "run_only_after_shortlist_geometry_supports_a_candidate",
            },
            "posterior_interpretation_allowed": False,
        }
        source_retention = {
            "format": "prompt_source_information_retention_v1",
            "applicability": (
                "applicable" if distributor_active else "not_applicable_prompt_generator_not_active"
            ),
            "observed": bool(distributor_active),
            "interface_class_structure": (
                hierarchy_trace.get("class_structure", {})
                if distributor_active
                else {}
            ),
            "heldout_linear_probe": {
                "status": "deferred_high_cost",
                "reason": "requires_predeclared_fit_and_heldout_subsets_after_shortlist",
            },
        }
        report = build_object_selection_report(registry, hierarchy_trace)
        return {
            **base_result,
            "applicability": "applicable",
            "observed": True,
            "valid": bool(hierarchy_trace.get("valid", False)),
            "failure_reason": (
                None if hierarchy_trace.get("valid", False) else "incomplete_hierarchy_trace"
            ),
            "registry": registry,
            "functional_geometry": functional_geometry,
            "hierarchy_trace": hierarchy_trace,
            "source_retention": source_retention,
            "object_selection_report": report,
        }

    @staticmethod
    def _fixed_probe_execution_profile(value):
        profile = str(value or "final_full").strip().lower()
        supported = {"final_full", "milestone_core", "robustness_core"}
        if profile not in supported:
            raise ValueError(
                "unsupported fixed-Probe execution profile: {} (expected one of {})".format(
                    profile, ", ".join(sorted(supported))
                )
            )
        return profile

    @staticmethod
    def _fixed_probe_profile_specs(specs, execution_profile):
        if execution_profile == "final_full":
            return list(specs)
        core_names = {
            "prompt_zeroed",
            "deep_prompt_residual_zeroed",
            "deep_prompt_residual_swapped",
            "prompt_read_blocked",
            "prompt_write_blocked",
            "prompt_patch_selection_uniform",
            "patch_prompt_selection_uniform",
            "prompt_patch_value_globalized",
            "prompt_context_swapped",
        }
        return [
            spec
            for spec in specs
            if spec["name"] in core_names
            or spec["name"].startswith("prompt_read_blocked_layer_")
        ]

    def _fixed_probe_loader_settings(self):
        num_workers = int(self.cfg.MONITOR.PROBE.NUM_WORKERS)
        if num_workers < 0:
            raise ValueError("MONITOR.PROBE.NUM_WORKERS must be non-negative")
        return {
            "num_workers": num_workers,
            "pin_memory": bool(self.cfg.MONITOR.PROBE.PIN_MEMORY),
        }

    def _build_fixed_probe_loader(self, dataset, batch_size):
        settings = self._fixed_probe_loader_settings()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=settings["num_workers"],
            pin_memory=settings["pin_memory"],
            drop_last=False,
        )

    def _synchronize_probe_device(self):
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _execute_timed_probe_stage(self, loader, execute):
        timed_loader = _TimedProbeLoader(loader)
        self._synchronize_probe_device()
        started = time.perf_counter()
        result = execute(timed_loader)
        self._synchronize_probe_device()
        total_time_sec = max(0.0, time.perf_counter() - started)
        data_time_sec = min(total_time_sec, max(0.0, timed_loader.data_time_sec))
        compute_time_sec = max(0.0, total_time_sec - data_time_sec)
        return result, {
            "probe_total_time_sec": total_time_sec,
            "probe_data_time_sec": data_time_sec,
            "probe_compute_time_sec": compute_time_sec,
            "probe_data_time_ratio": (
                data_time_sec / total_time_sec if total_time_sec > 0.0 else 0.0
            ),
            "batch_count": int(timed_loader.batch_count),
        }

    @staticmethod
    def _aggregate_probe_stage_timings(timing_by_split):
        stage_timings = [
            timing
            for split_timings in timing_by_split.values()
            for timing in split_timings.values()
        ]
        total_time_sec = sum(
            float(item.get("probe_total_time_sec", 0.0)) for item in stage_timings
        )
        data_time_sec = sum(
            float(item.get("probe_data_time_sec", 0.0)) for item in stage_timings
        )
        compute_time_sec = sum(
            float(item.get("probe_compute_time_sec", 0.0)) for item in stage_timings
        )
        return {
            "probe_total_time_sec": total_time_sec,
            "probe_data_time_sec": data_time_sec,
            "probe_compute_time_sec": compute_time_sec,
            "probe_data_time_ratio": (
                data_time_sec / total_time_sec if total_time_sec > 0.0 else 0.0
            ),
            "batch_count": sum(
                int(item.get("batch_count", 0)) for item in stage_timings
            ),
            "stage_count": len(stage_timings),
        }

    @torch.no_grad()
    def _execute_fixed_probe_bundle(
        self,
        probe_loader,
        source_dataset,
        candidate_class_ids,
        *,
        split,
        execution_profile="final_full",
    ):
        execution_profile = self._fixed_probe_execution_profile(execution_profile)
        core_profile = execution_profile != "final_full"
        model_ref = self._model_ref(self.model)
        candidate_class_ids = [int(item) for item in candidate_class_ids]
        global_to_local = {global_id: local_id for local_id, global_id in enumerate(candidate_class_ids)}
        prompt_length = int(self.cfg.MODEL.PROMPT.NUM_TOKENS) if bool(self.cfg.MODEL.PROMPT.ENABLE) else 0
        semantic_length = int(self.cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS) if bool(self.cfg.MODEL.SEMANTIC_TOKENS.ENABLE) else 0
        prompt_analysis_cfg = self.cfg.MONITOR.PROBE.PROMPT_ANALYSIS
        prompt_analysis_requested = bool(
            prompt_analysis_cfg.ENABLE and not core_profile
        )
        prompt_analysis_enabled = bool(
            prompt_analysis_requested and prompt_length > 0
        )
        distributor_cfg = self.cfg.MODEL.PROMPT.DISTRIBUTOR
        distributor_active = bool(
            prompt_length > 0
            and distributor_cfg.ENABLE
            and str(self.cfg.MODEL.PROMPT.INIT_SOURCE).lower()
            == "distributor_mean"
        )
        instance_prompt_length = (
            int(distributor_cfg.INSTANCE_TOKENS) if distributor_active else 0
        )
        domain_prompt_length = (
            int(distributor_cfg.DOMAIN_TOKENS) if distributor_active else 0
        )
        source_accumulator = (
            PromptSourceDecompositionAccumulator(
                instance_tokens=instance_prompt_length,
                domain_tokens=domain_prompt_length,
                contextualized_domain_applicable=not bool(
                    self.cfg.MODEL.PROMPT.DEEP
                ),
            )
            if prompt_analysis_enabled
            and bool(prompt_analysis_cfg.SOURCE_DECOMPOSITION_ENABLE)
            and distributor_active
            else None
        )
        label_guard_requested = bool(
            prompt_analysis_enabled
            and prompt_analysis_cfg.LABEL_DEPENDENCY_GUARD_ENABLE
        )
        label_guard = {
            "requested": label_guard_requested,
            "applicability": (
                "applicable"
                if label_guard_requested and distributor_active
                else "not_applicable_prompt_distributor_not_active"
                if label_guard_requested
                else "not_requested"
            ),
            "observed": False,
            "valid": None,
            "metrics": {},
            "failure_reasons": [],
        }
        normal_accumulator = StreamingFixedProbeAccumulator(candidate_class_ids, track_geometry=True)
        semantic_enabled = bool(
            self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE
            and not core_profile
        )
        permutation = inverse = None
        permuted_ids = None
        synchronized_accumulator = mismatched_accumulator = mismatched_effect = None
        synchronized_state = self._new_equivalence_state()
        if semantic_enabled:
            permutation, inverse, permutation_seed, permutation_sha256 = self._semantic_permutation(candidate_class_ids, split)
            permuted_ids = [candidate_class_ids[int(index)] for index in permutation]
            synchronized_accumulator = StreamingFixedProbeAccumulator(permuted_ids, track_geometry=False)
            mismatched_accumulator = StreamingFixedProbeAccumulator(candidate_class_ids, track_geometry=False)
            mismatched_effect = PairedModuleEffectAccumulator(candidate_class_ids, source_dataset.seen_classes)
            synchronized_global_to_local = {global_id: local_id for local_id, global_id in enumerate(permuted_ids)}

        affinity_enabled = bool(self.cfg.MONITOR.PROBE.AFFINITY_ENABLE)
        use_attention = affinity_enabled and bool(self.cfg.MONITOR.PROBE.ATTENTION_ENABLE)
        selected_probe_layers = [int(item) for item in self.cfg.MONITOR.PROBE.LAYERS]
        raw_candidate_semantics = None
        class_attributes = getattr(source_dataset, "class_attributes", None)
        if class_attributes is not None:
            class_attributes = torch.as_tensor(class_attributes).detach().cpu()
            candidate_index = torch.as_tensor(candidate_class_ids, dtype=torch.long)
            if (
                class_attributes.dim() == 2
                and candidate_index.numel() > 0
                and candidate_index.numel() == len(candidate_class_ids)
                and int(candidate_index.min().item()) >= 0
                and int(candidate_index.max().item()) < class_attributes.shape[0]
            ):
                raw_candidate_semantics = class_attributes.index_select(
                    0, candidate_index
                )
        attribute_concept_reference = self._attribute_concept_reference(
            model_ref, raw_candidate_semantics
        )
        prompt_mechanism_mode = (
            "layerwise_replaced"
            if prompt_length > 0 and bool(self.cfg.MODEL.PROMPT.DEEP)
            else "persistent_contextualized"
            if prompt_length > 0
            else "not_applicable"
        )
        attribute_concept_enabled = bool(
            self.cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_ENABLE
            and not core_profile
            and attribute_concept_reference.get("available", False)
            and prompt_length > 0
        )
        semantic_granularity_requested = bool(
            prompt_analysis_enabled
            and prompt_analysis_cfg.SEMANTIC_GRANULARITY_ENABLE
        )
        local_attribute_indices = [
            int(item) for item in prompt_analysis_cfg.LOCAL_ATTRIBUTE_INDICES
        ]
        global_attribute_indices = [
            int(item) for item in prompt_analysis_cfg.GLOBAL_ATTRIBUTE_INDICES
        ]
        semantic_granularity_applicable = bool(
            semantic_granularity_requested
            and attribute_concept_enabled
            and local_attribute_indices
            and global_attribute_indices
        )
        if semantic_granularity_requested:
            overlap = set(local_attribute_indices).intersection(
                global_attribute_indices
            )
            if overlap:
                raise ValueError(
                    "Prompt semantic local/global attribute groups must be disjoint"
                )
            attribute_count = int(
                attribute_concept_reference.get("attribute_count", 0)
            )
            invalid_indices = [
                item
                for item in local_attribute_indices + global_attribute_indices
                if item < 0 or item >= attribute_count
            ]
            if invalid_indices:
                raise ValueError(
                    "Prompt semantic attribute group contains out-of-range indices: "
                    + ",".join(str(item) for item in sorted(set(invalid_indices)))
                )
        patch_semantic_transport_enabled = bool(
            affinity_enabled
            and not core_profile
            and self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.ENABLE
        )
        patch_semantic_transport_reference = {
            "available": bool(
                patch_semantic_transport_enabled
                and getattr(model_ref, "r_similarity_head", None) is not None
            ),
            "reason": (
                None
                if patch_semantic_transport_enabled
                and getattr(model_ref, "r_similarity_head", None) is not None
                else "patch_semantic_transport_probe_disabled"
                if not patch_semantic_transport_enabled
                else "r_similarity_head_unavailable"
            ),
            "purpose": "local_visual_to_class_semantic_geometric_reference",
            "cost_name": str(
                self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.COST
            ).lower(),
            "cost_formula": "one_minus_cosine_similarity",
            "cost_formula_replaceable": True,
            "temperature": float(
                self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.TEMPERATURE
            ),
            "selected_patch_ratio": float(
                self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO
            ),
            "classifier_patch_logit": False,
            "class_weight_mode": "normal_prediction_probability",
            "layer_scope": "final_transformer_layer",
            "storage_mode": "aggregate_only",
        }
        affinity_state = self._new_equivalence_state()
        affinity_metric_accumulator = None
        token_accumulator = None
        affinity_cfg = {
            "prompt_length": prompt_length,
            "semantic_length": semantic_length,
            "detach": True,
            "offload_diagnostics_to_cpu": True,
            "selected_layers": selected_probe_layers,
            "include_visual_normalizations": False,
            "block_s_to_cls": bool(self.cfg.MODEL.SEMANTIC_TOKENS.BLOCK_S_TO_CLS),
            "attribute_concept_enable": attribute_concept_enabled,
            "attribute_concept_selected_layers": selected_probe_layers,
            "attribute_concept_patch_ratio": float(
                self.cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_PATCH_RATIO
            ),
            "local_attribute_indices": (
                local_attribute_indices if semantic_granularity_applicable else []
            ),
            "global_attribute_indices": (
                global_attribute_indices if semantic_granularity_applicable else []
            ),
        }
        prompt_content_enabled = bool(
            prompt_analysis_enabled
            and prompt_analysis_cfg.CONTENT_ENABLE
            and affinity_enabled
        )
        role_profile_export_enabled = bool(
            prompt_analysis_enabled
            and prompt_analysis_cfg.ROLE_PROFILE_EXPORT_ENABLE
            and affinity_enabled
        )
        flip_requested = bool(
            prompt_analysis_enabled and prompt_analysis_cfg.FLIP_ENABLE
        )
        flip_accumulator = (
            PairedFlipAccumulator(
                prompt_length=prompt_length,
                semantic_length=semantic_length,
                topk=int(prompt_analysis_cfg.FLIP_TOPK),
                selected_layers=selected_probe_layers,
            )
            if flip_requested and affinity_enabled
            else None
        )
        flip_pair_hash = hashlib.sha256()
        flip_pair_sample_count = 0
        prompt_accumulator_kwargs = {
            "prompt_content_enable": prompt_content_enabled,
            "instance_prompt_length": instance_prompt_length,
            "domain_prompt_length": domain_prompt_length,
            "content_redundancy_cosine": float(
                prompt_analysis_cfg.CONTENT_REDUNDANCY_COSINE
            ),
            "content_opposition_cosine": float(
                prompt_analysis_cfg.CONTENT_OPPOSITION_COSINE
            ),
            "content_cancellation_ratio": float(
                prompt_analysis_cfg.CONTENT_CANCELLATION_RATIO
            ),
            "low_usage_fraction": float(
                prompt_analysis_cfg.LOW_USAGE_FRACTION
            ),
            "low_function_fraction": float(
                prompt_analysis_cfg.LOW_FUNCTION_FRACTION
            ),
            "low_role_coverage": float(
                prompt_analysis_cfg.LOW_ROLE_COVERAGE
            ),
            "role_profile_export_enable": role_profile_export_enabled,
        }
        if affinity_enabled:
            affinity_metric_accumulator = ProbeAttentionAffinityAccumulator(
                prompt_length=prompt_length,
                semantic_length=semantic_length,
                selected_layers=selected_probe_layers,
                class_count=len(candidate_class_ids),
                raw_semantic_reference=raw_candidate_semantics,
                attribute_count=int(
                    attribute_concept_reference.get("attribute_count", 0)
                ),
                attribute_concept_enable=attribute_concept_enabled,
                patch_semantic_transport_enable=(
                    patch_semantic_transport_reference["available"]
                ),
                patch_semantic_transport_cost=str(
                    self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.COST
                ),
                patch_semantic_transport_temperature=float(
                    self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.TEMPERATURE
                ),
                patch_semantic_transport_patch_ratio=float(
                    self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO
                ),
                prompt_mode=prompt_mechanism_mode,
                score_mode=str(
                    attribute_concept_reference.get("score_mode", "unknown")
                ),
                temperature=float(self.cfg.MONITOR.PROBE.AFFINITY_TEMPERATURE),
                saturation_threshold=float(self.cfg.MONITOR.PROBE.AFFINITY_SATURATION_THRESHOLD),
                **prompt_accumulator_kwargs,
            )
            token_accumulator = StreamingTokenViewAccumulator(len(candidate_class_ids), prompt_length, semantic_length)

        intervention_specs = self._fixed_probe_profile_specs(
            self._module_effect_intervention_specs(
                prompt_length,
                attribute_concept_available=bool(
                    attribute_concept_reference.get("available", False)
                ),
                attribute_concept_reason=str(
                    attribute_concept_reference.get(
                        "reason", "attribute_concept_reference_unavailable"
                    )
                ),
                patch_semantic_transport_available=bool(
                    patch_semantic_transport_reference["available"]
                ),
                patch_semantic_transport_reason=str(
                    patch_semantic_transport_reference.get(
                        "reason", "patch_semantic_transport_reference_unavailable"
                    )
                ),
            ),
            execution_profile,
        )
        effective_intervention_specs = [
            spec for spec in intervention_specs if spec["applicable"]
        ]
        intervention_factories = [
            (spec["name"], spec["factory"], spec["dynamic_context"])
            for spec in effective_intervention_specs
        ]
        intervention_semantics = {
            spec["name"]: spec["intervention_semantics"]
            for spec in effective_intervention_specs
        }
        module_accumulators = {
            name: PairedModuleEffectAccumulator(candidate_class_ids, source_dataset.seen_classes)
            for name, _, _ in intervention_factories
        }
        intervention_runtime_contracts = {
            name: {
                "sample_count": 0,
                "swapped_sample_count": 0,
                "same_class_pair_count": 0,
                "singleton_batch_count": 0,
                "batch_count": 0,
                "pairing_hash": hashlib.sha256(),
            }
            for name, _, dynamic_context in intervention_factories
            if dynamic_context in {"prompt_swap", "instance_prompt_swap", "deep_residual_swap"}
        }
        prompt_output_intervention_states = {
            name: {
                "sums": defaultdict(float),
                "counts": defaultdict(int),
            }
            for name, _, _ in intervention_factories
            if name in {
                "instance_prompt_zeroed",
                "domain_prompt_zeroed",
                "both_prompt_zeroed",
                "instance_prompt_swapped",
            }
        }
        diagnostic_intervention_names = {
            spec["name"]
            for spec in effective_intervention_specs
            if spec["diagnostic_chain"]
        }
        intervention_probe_accumulators = {
            name: StreamingFixedProbeAccumulator(
                candidate_class_ids,
                track_geometry=False,
            )
            for name in diagnostic_intervention_names
        }
        intervention_affinity_accumulators = {}
        intervention_patch_diversity_accumulators = {}
        intervention_affinity_states = {
            name: self._new_equivalence_state()
            for name in diagnostic_intervention_names
        }
        if affinity_enabled:
            intervention_affinity_accumulators = {
                name: ProbeAttentionAffinityAccumulator(
                    prompt_length=prompt_length,
                    semantic_length=semantic_length,
                    selected_layers=selected_probe_layers,
                    class_count=len(candidate_class_ids),
                    raw_semantic_reference=raw_candidate_semantics,
                    attribute_count=int(
                        attribute_concept_reference.get("attribute_count", 0)
                    ),
                    attribute_concept_enable=attribute_concept_enabled,
                    patch_semantic_transport_enable=(
                        patch_semantic_transport_reference["available"]
                    ),
                    patch_semantic_transport_cost=str(
                        self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.COST
                    ),
                    patch_semantic_transport_temperature=float(
                        self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.TEMPERATURE
                    ),
                    patch_semantic_transport_patch_ratio=float(
                        self.cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO
                    ),
                    prompt_mode=prompt_mechanism_mode,
                    score_mode=str(
                        attribute_concept_reference.get("score_mode", "unknown")
                    ),
                    temperature=float(self.cfg.MONITOR.PROBE.AFFINITY_TEMPERATURE),
                    saturation_threshold=float(self.cfg.MONITOR.PROBE.AFFINITY_SATURATION_THRESHOLD),
                    **prompt_accumulator_kwargs,
                )
                for name in diagnostic_intervention_names
            }
            intervention_patch_diversity_accumulators = {
                name: StreamingPatchDiversityAccumulator(
                    len(candidate_class_ids),
                    prompt_length,
                    semantic_length,
                )
                for name in diagnostic_intervention_names
            }

        for batch_index, input_data in enumerate(probe_loader):
            inputs, targets_global, attributes = self.get_input(input_data)
            inputs = inputs.to(self.device, non_blocking=True)
            targets_global = targets_global.to(self.device, non_blocking=True)
            target_global_list = [int(item) for item in targets_global.detach().cpu().tolist()]
            target_local = np.asarray([global_to_local[item] for item in target_global_list], dtype=np.int64)
            semantics = self._prepare_semantics_for_stage(
                attributes, source_dataset, batch_size=int(inputs.shape[0]), is_train=False
            )
            normal_logits = model_ref(
                inputs, semantics=semantics, class_ids=candidate_class_ids, runtime_targets=None
            )
            normal_stats = model_ref.get_runtime_classifier_stats()
            if not isinstance(normal_stats, dict):
                raise RuntimeError("fixed probe requires runtime classifier statistics")
            normal_visual_input = normal_stats.get("visual_input")
            normal_visual = normal_stats.get("visual_repr")
            normal_semantic_input = normal_stats.get("semantic_input")
            normal_semantic = normal_stats.get("semantic_repr")
            if not all(
                torch.is_tensor(value)
                for value in (
                    normal_visual_input,
                    normal_visual,
                    normal_semantic_input,
                    normal_semantic,
                )
            ):
                raise RuntimeError("fixed probe classifier statistics are incomplete")
            prompt_distribution_stats_getter = getattr(
                model_ref,
                "get_runtime_prompt_distribution_stats",
                None,
            )
            normal_prompt_distribution_stats = (
                prompt_distribution_stats_getter()
                if callable(prompt_distribution_stats_getter)
                else {}
            )
            source_true_margin = None
            if source_accumulator is not None:
                source_target = torch.as_tensor(
                    target_local,
                    device=normal_logits.device,
                    dtype=torch.long,
                )
                source_other = normal_logits.detach().float().clone()
                source_other.scatter_(
                    1,
                    source_target.unsqueeze(1),
                    torch.finfo(source_other.dtype).min,
                )
                source_true_margin = (
                    normal_logits.detach().float().gather(
                        1, source_target.unsqueeze(1)
                    ).squeeze(1)
                    - source_other.max(dim=1).values
                )
                source_accumulator.update_raw(
                    normal_prompt_distribution_stats,
                    target_local,
                    cls_repr=normal_visual,
                    true_margin=source_true_margin,
                )
            if (
                label_guard_requested
                and distributor_active
                and not label_guard["observed"]
            ):
                if int(inputs.shape[0]) < 2:
                    label_guard["applicability"] = (
                        "not_applicable_singleton_first_batch"
                    )
                else:
                    permutation = torch.roll(
                        torch.arange(inputs.shape[0], device=inputs.device),
                        shifts=1,
                    )
                    permuted_semantics = semantics
                    if (
                        torch.is_tensor(semantics)
                        and semantics.dim() > 0
                        and int(semantics.shape[0]) == int(inputs.shape[0])
                    ):
                        permuted_semantics = semantics.index_select(0, permutation)
                    elif (
                        torch.is_tensor(semantics)
                        and semantics.dim() > 0
                        and int(semantics.shape[0])
                        == len(candidate_class_ids)
                    ):
                        semantic_permutation = torch.roll(
                            torch.arange(
                                semantics.shape[0], device=semantics.device
                            ),
                            shifts=1,
                        )
                        permuted_semantics = semantics.index_select(
                            0, semantic_permutation
                        )
                    with torch.no_grad():
                        model_ref(
                            inputs,
                            semantics=permuted_semantics,
                            class_ids=candidate_class_ids,
                            runtime_targets=targets_global.index_select(
                                0, permutation
                            ),
                        )
                    permuted_prompt_stats = (
                        prompt_distribution_stats_getter()
                        if callable(prompt_distribution_stats_getter)
                        else {}
                    )
                    max_diffs = {}
                    for field in ("mu", "logvar", "prompt_tokens"):
                        reference = (
                            normal_prompt_distribution_stats.get(field)
                            if isinstance(normal_prompt_distribution_stats, dict)
                            else None
                        )
                        changed = (
                            permuted_prompt_stats.get(field)
                            if isinstance(permuted_prompt_stats, dict)
                            else None
                        )
                        if (
                            not torch.is_tensor(reference)
                            or not torch.is_tensor(changed)
                            or reference.shape != changed.shape
                        ):
                            label_guard["failure_reasons"].append(
                                f"{field} was unavailable or changed shape"
                            )
                            continue
                        max_diffs[f"{field}_permuted_metadata_max_abs_diff"] = float(
                            (reference.detach() - changed.detach()).abs().max().item()
                        )
                    tolerance = float(
                        self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
                    )
                    valid = bool(
                        len(max_diffs) == 3
                        and all(value <= tolerance for value in max_diffs.values())
                    )
                    if not valid and not label_guard["failure_reasons"]:
                        label_guard["failure_reasons"].append(
                            "Prompt Distributor output changed after label/class-attribute metadata permutation"
                        )
                    label_guard.update({
                        "observed": True,
                        "valid": valid,
                        "metrics": {
                            **max_diffs,
                            "label_dependency_guard_pass": float(valid),
                            "static_prompt_provider_receives_label": 0.0,
                            "static_prompt_provider_receives_class_attribute": 0.0,
                        },
                    })
            normal_logits_cpu = normal_logits.detach().cpu()
            normal_visual_cpu = normal_visual.detach().cpu()
            normal_accumulator.update(normal_logits_cpu, target_local, normal_visual_cpu, normal_semantic)

            attribute_concept_batch = None
            if attribute_concept_enabled:
                raw_reference_device = raw_candidate_semantics.to(
                    device=normal_logits.device, dtype=torch.float32
                )
                local_target_tensor = torch.as_tensor(
                    target_local, device=normal_logits.device, dtype=torch.long
                )
                hard_negative_logits = normal_logits.detach().float().clone()
                hard_negative_logits.scatter_(
                    1,
                    local_target_tensor.unsqueeze(1),
                    torch.finfo(hard_negative_logits.dtype).min,
                )
                hard_negative = hard_negative_logits.argmax(dim=1)
                true_attribute_weights = raw_reference_device.index_select(
                    0, local_target_tensor
                )
                hard_attribute_weights = raw_reference_device.index_select(
                    0, hard_negative
                )
                margin_attribute_weights = (
                    true_attribute_weights - hard_attribute_weights
                )
                attribute_directions = attribute_concept_reference[
                    "directions"
                ].to(device=normal_logits.device, dtype=torch.float32)
                attribute_concept_batch = {
                    "attribute_directions": attribute_directions,
                    "true_weights": true_attribute_weights,
                    "margin_weights": margin_attribute_weights,
                    "patch_ratio": float(
                        self.cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_PATCH_RATIO
                    ),
                    "random_seed": int(
                        self.cfg.MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_RANDOM_SEED
                    ) + int(batch_index),
                }
                affinity_cfg.update({
                    "attribute_concept_directions": attribute_directions,
                    "attribute_concept_true_weights": true_attribute_weights,
                    "attribute_concept_margin_weights": margin_attribute_weights,
                })

            if semantic_enabled:
                synchronized_logits = model_ref.r_similarity_head(
                    normal_visual_input,
                    class_ids=permuted_ids,
                    prototype_class_ids=permuted_ids,
                    runtime_targets=None,
                )
                synchronized_stats = model_ref.get_runtime_classifier_stats()
                synchronized_target = np.asarray(
                    [synchronized_global_to_local[item] for item in target_global_list], dtype=np.int64
                )
                synchronized_accumulator.update(
                    synchronized_logits, synchronized_target,
                    synchronized_stats["visual_repr"], synchronized_stats["semantic_repr"],
                )
                recovered_logits = synchronized_logits.index_select(
                    1, torch.as_tensor(inverse, device=synchronized_logits.device, dtype=torch.long)
                )
                self._update_equivalence_state(
                    synchronized_state, normal_logits_cpu, recovered_logits, target_local
                )
                mismatched_logits = model_ref.r_similarity_head(
                    normal_visual_input,
                    class_ids=candidate_class_ids,
                    prototype_class_ids=permuted_ids,
                    runtime_targets=None,
                )
                mismatched_stats = model_ref.get_runtime_classifier_stats()
                mismatched_accumulator.update(
                    mismatched_logits, target_local,
                    mismatched_stats["visual_repr"], mismatched_stats["semantic_repr"],
                )
                mismatched_effect.update(normal_logits_cpu, mismatched_logits, target_local)
                del synchronized_logits, recovered_logits, mismatched_logits

            transport_batch_reference = None
            if affinity_enabled:
                affinity_output = model_ref.forward_with_affinity(
                    inputs,
                    affinity_cfg,
                    semantics=semantics,
                    vis=use_attention,
                    class_ids=candidate_class_ids,
                    runtime_targets=None,
                )
                if use_attention:
                    affinity_logits, attention_layers, affinities = affinity_output
                else:
                    affinity_logits, affinities = affinity_output
                    attention_layers = []
                token_sequence = model_ref.get_runtime_token_sequence()
                transport_batch_reference = affinity_metric_accumulator.update(
                    list(attention_layers or []),
                    list(affinities or []),
                    predictions=affinity_logits.detach().argmax(dim=1).cpu().tolist(),
                    targets=target_local.tolist(),
                    projected_semantic_reference=normal_semantic,
                    transport_semantic_reference=normal_semantic_input,
                    token_sequence=token_sequence,
                    logits=affinity_logits,
                )
                if torch.is_tensor(token_sequence):
                    token_accumulator.update(
                        token_sequence,
                        target_local,
                        semantic_prototypes=normal_semantic,
                    )
                    if source_accumulator is not None:
                        source_accumulator.update_contextualized(
                            token_sequence,
                            target_local,
                            prompt_length=prompt_length,
                            semantic_length=semantic_length,
                            cls_repr=normal_visual,
                            true_margin=source_true_margin,
                        )
                self._update_equivalence_state(
                    affinity_state, normal_logits_cpu, affinity_logits, target_local
                )
                if flip_accumulator is not None:
                    flipped_inputs = torch.flip(inputs, dims=[-1])
                    flipped_output = model_ref.forward_with_affinity(
                        flipped_inputs,
                        affinity_cfg,
                        semantics=semantics,
                        vis=use_attention,
                        class_ids=candidate_class_ids,
                        runtime_targets=None,
                    )
                    if use_attention:
                        (
                            flipped_logits,
                            flipped_attention_layers,
                            flipped_affinities,
                        ) = flipped_output
                    else:
                        flipped_logits, flipped_affinities = flipped_output
                        flipped_attention_layers = []
                    flip_accumulator.update(
                        list(attention_layers or []),
                        list(affinities or []),
                        list(flipped_attention_layers or []),
                        list(flipped_affinities or []),
                        normal_predictions=affinity_logits.detach().argmax(dim=1),
                        flipped_predictions=flipped_logits.detach().argmax(dim=1),
                    )
                    sample_ids = input_data.get("sample_id")
                    if sample_ids is None:
                        raise RuntimeError(
                            "paired horizontal flip requires fixed-probe sample_id"
                        )
                    sample_ids = [str(item) for item in list(sample_ids)]
                    flip_pair_hash.update("|".join(sample_ids).encode("utf-8"))
                    flip_pair_sample_count += len(sample_ids)
                    del flipped_output, flipped_logits, flipped_attention_layers
                    del flipped_affinities, flipped_inputs
                del affinity_output, affinity_logits, attention_layers, affinities, token_sequence

            transport_intervention_batch = None
            if patch_semantic_transport_reference["available"]:
                transport_specs_requested = any(
                    dynamic_context == "transport"
                    for _, _, dynamic_context in intervention_factories
                )
                if transport_batch_reference is None:
                    if transport_specs_requested:
                        raise RuntimeError(
                            "Transport intervention lacks a normal final-layer transport reference"
                        )
                else:
                    targeted_indices = transport_batch_reference[
                        "selected_patch_indices"
                    ]
                    reference_scores = transport_batch_reference[
                        "reference_patch_scores"
                    ]
                    random_generator = torch.Generator(
                        device=reference_scores.device
                    )
                    random_generator.manual_seed(
                        int(
                            (
                                int(
                                    self.cfg.MONITOR.MODULE_EFFECT.TRANSPORT_RANDOM_SEED
                                )
                                + int(batch_index)
                            )
                            % (2 ** 63 - 1)
                        )
                    )
                    random_scores = torch.rand(
                        reference_scores.shape,
                        generator=random_generator,
                        device=reference_scores.device,
                        dtype=torch.float32,
                    )
                    random_indices = random_scores.topk(
                        int(targeted_indices.shape[1]), dim=-1, largest=True
                    ).indices
                    transport_intervention_batch = {
                        "targeted_indices": targeted_indices,
                        "random_indices": random_indices,
                        "reference_scores": reference_scores,
                    }

            for intervention_name, factory, dynamic_context in intervention_factories:
                if dynamic_context == "attribute_concept":
                    if attribute_concept_batch is None:
                        raise RuntimeError(
                            "Dynamic attribute-concept intervention lacks batch reference"
                        )
                    context = factory(
                        model_ref,
                        attribute_directions=attribute_concept_batch[
                            "attribute_directions"
                        ],
                        margin_weights=attribute_concept_batch["margin_weights"],
                        patch_ratio=attribute_concept_batch["patch_ratio"],
                        random_seed=attribute_concept_batch["random_seed"],
                    )
                elif dynamic_context == "transport":
                    if transport_intervention_batch is None:
                        raise RuntimeError(
                            "Dynamic transport intervention lacks batch reference"
                        )
                    selected_indices = (
                        transport_intervention_batch["targeted_indices"]
                        if intervention_name
                        == "transport_targeted_prompt_patch_blocked"
                        else transport_intervention_batch["random_indices"]
                    )
                    context = factory(
                        model_ref,
                        selected_patch_indices=selected_indices,
                        reference_patch_scores=transport_intervention_batch[
                            "reference_scores"
                        ],
                    )
                elif dynamic_context in {"prompt_swap", "instance_prompt_swap", "deep_residual_swap"}:
                    sample_ids = input_data.get("sample_id")
                    if sample_ids is None:
                        raise RuntimeError(
                            f"{dynamic_context} requires fixed-probe sample_id"
                        )
                    sample_ids = [str(item) for item in list(sample_ids)]
                    batch_size = len(sample_ids)
                    if batch_size != int(inputs.shape[0]):
                        raise RuntimeError(
                            "Prompt context swap sample identity count does not match batch size"
                        )
                    if dynamic_context == "prompt_swap":
                        swap_seed = int(
                            self.cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_SEED
                        )
                    elif dynamic_context == "instance_prompt_swap":
                        swap_seed = int(
                            self.cfg.MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_SWAP_SEED
                        )
                    else:
                        swap_seed = int(
                            self.cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_SWAP_SEED
                        )
                    order = sorted(
                        range(batch_size),
                        key=lambda index: (
                            hashlib.sha256(
                                f"{swap_seed}|{sample_ids[index]}".encode("utf-8")
                            ).hexdigest(),
                            sample_ids[index],
                        ),
                    )
                    permutation_values = list(range(batch_size))
                    if batch_size > 1:
                        for position, destination in enumerate(order):
                            permutation_values[destination] = order[
                                (position + 1) % batch_size
                            ]
                    permutation = torch.as_tensor(
                        permutation_values,
                        device=inputs.device,
                        dtype=torch.long,
                    )
                    runtime_contract = intervention_runtime_contracts[
                        intervention_name
                    ]
                    runtime_contract["sample_count"] += batch_size
                    runtime_contract["swapped_sample_count"] += (
                        batch_size if batch_size > 1 else 0
                    )
                    runtime_contract["singleton_batch_count"] += int(
                        batch_size == 1
                    )
                    runtime_contract["batch_count"] += 1
                    partner_targets = targets_global.index_select(0, permutation)
                    runtime_contract["same_class_pair_count"] += int(
                        (partner_targets == targets_global).sum().item()
                    )
                    runtime_contract["pairing_hash"].update(
                        "|".join(
                            f"{sample_ids[index]}->{sample_ids[permutation_values[index]]}"
                            for index in range(batch_size)
                        ).encode("utf-8")
                    )
                    if dynamic_context == "prompt_swap":
                        context = factory(
                            model_ref,
                            target_layer=int(
                                self.cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_LAYER
                            ),
                            permutation=permutation,
                        )
                    else:
                        context = factory(model_ref, permutation=permutation)
                elif dynamic_context:
                    raise RuntimeError(
                        f"Unsupported dynamic intervention context: {dynamic_context}"
                    )
                else:
                    context = factory(model_ref)
                with context:
                    changed_logits = model_ref(
                        inputs, semantics=semantics, class_ids=candidate_class_ids, runtime_targets=None
                    )
                    changed_stats = model_ref.get_runtime_classifier_stats()
                    if intervention_name in prompt_output_intervention_states:
                        prompt_output_stats = (
                            prompt_distribution_stats_getter()
                            if callable(prompt_distribution_stats_getter)
                            else {}
                        )
                        state = prompt_output_intervention_states[
                            intervention_name
                        ]
                        if isinstance(prompt_output_stats, dict):
                            for name, values in prompt_output_stats.items():
                                if name not in {
                                    "prompt_output_intervention_applied",
                                    "instance_prompt_norm_before",
                                    "instance_prompt_norm_after",
                                    "domain_prompt_norm_before",
                                    "domain_prompt_norm_after",
                                    "instance_prompt_swap_fixed_point_ratio",
                                }:
                                    continue
                                if not torch.is_tensor(values):
                                    continue
                                values = values.detach().float().reshape(-1)
                                finite = values[torch.isfinite(values)]
                                if finite.numel() <= 0:
                                    continue
                                state["sums"][name] += float(
                                    finite.sum().item()
                                )
                                state["counts"][name] += int(
                                    finite.numel()
                                )
                    changed_visual = changed_stats.get("visual_repr") if isinstance(changed_stats, dict) else None
                    changed_semantic_input = (
                        changed_stats.get("semantic_input")
                        if isinstance(changed_stats, dict)
                        else None
                    )
                    changed_semantic = changed_stats.get("semantic_repr") if isinstance(changed_stats, dict) else None
                    changed_logits_cpu = changed_logits.detach().cpu()
                    changed_visual_cpu = (
                        changed_visual.detach().cpu() if torch.is_tensor(changed_visual) else None
                    )
                    module_accumulators[intervention_name].update(
                        normal_logits_cpu,
                        changed_logits_cpu,
                        target_local,
                        normal_features=normal_visual_cpu,
                        intervention_features=changed_visual_cpu,
                    )
                    if intervention_name in diagnostic_intervention_names:
                        if not torch.is_tensor(changed_visual) or not torch.is_tensor(changed_semantic):
                            raise RuntimeError(
                                f"{intervention_name} fixed-probe visual/semantic representations are unavailable"
                            )
                        intervention_probe_accumulators[intervention_name].update(
                            changed_logits_cpu,
                            target_local,
                            changed_visual_cpu,
                            changed_semantic,
                        )
                        if affinity_enabled:
                            intervention_affinity_output = model_ref.forward_with_affinity(
                                inputs,
                                affinity_cfg,
                                semantics=semantics,
                                vis=use_attention,
                                class_ids=candidate_class_ids,
                                runtime_targets=None,
                            )
                            if use_attention:
                                intervention_affinity_logits, intervention_attention_layers, intervention_affinities = (
                                    intervention_affinity_output
                                )
                            else:
                                intervention_affinity_logits, intervention_affinities = intervention_affinity_output
                                intervention_attention_layers = []
                            intervention_affinity_accumulators[intervention_name].update(
                                list(intervention_attention_layers or []),
                                list(intervention_affinities or []),
                                predictions=intervention_affinity_logits.detach().argmax(dim=1).cpu().tolist(),
                                targets=target_local.tolist(),
                                projected_semantic_reference=changed_semantic,
                                transport_semantic_reference=changed_semantic_input,
                                token_sequence=model_ref.get_runtime_token_sequence(),
                                logits=intervention_affinity_logits,
                            )
                            intervention_token_sequence = (
                                model_ref.get_runtime_token_sequence()
                            )
                            if torch.is_tensor(intervention_token_sequence):
                                intervention_patch_diversity_accumulators[
                                    intervention_name
                                ].update(
                                    intervention_token_sequence,
                                    target_local,
                                )
                            self._update_equivalence_state(
                                intervention_affinity_states[intervention_name],
                                changed_logits_cpu,
                                intervention_affinity_logits,
                                target_local,
                            )
                            del intervention_affinity_output, intervention_affinity_logits
                            del intervention_attention_layers, intervention_affinities
                            del intervention_token_sequence
                del changed_logits, changed_logits_cpu, changed_stats
                del changed_visual, changed_visual_cpu, changed_semantic
                del changed_semantic_input
            if source_accumulator is not None:
                del source_target, source_other, source_true_margin
            model_ref.clear_runtime_state()
            del normal_logits, normal_logits_cpu, normal_visual_input, normal_visual, normal_visual_cpu
            del normal_semantic_input, normal_semantic
            del inputs, targets_global, semantics
            if attribute_concept_batch is not None:
                for key in (
                    "attribute_concept_directions",
                    "attribute_concept_true_weights",
                    "attribute_concept_margin_weights",
                ):
                    affinity_cfg.pop(key, None)
                del attribute_concept_batch
                del raw_reference_device, local_target_tensor
                del hard_negative_logits, hard_negative
                del true_attribute_weights, hard_attribute_weights
                del margin_attribute_weights, attribute_directions
            if transport_intervention_batch is not None:
                del transport_intervention_batch
            if transport_batch_reference is not None:
                del transport_batch_reference

        normal = normal_accumulator.finalize()
        normal_accumulator.representation.release_covariance()
        result = {
            "execution_profile": execution_profile,
            "normal": normal,
            "normal_accumulator": normal_accumulator,
            "conditions": {"normal": normal},
            "module_effects": {name: accumulator.finalize() for name, accumulator in module_accumulators.items()},
            "prompt_length": prompt_length,
            "semantic_length": semantic_length,
        }
        source_decomposition = (
            source_accumulator.finalize()
            if source_accumulator is not None
            else {
                "format": "prompt_source_decomposition_v1",
                "applicability": (
                    "not_applicable_prompt_distributor_not_active"
                    if prompt_analysis_requested
                    and bool(prompt_analysis_cfg.SOURCE_DECOMPOSITION_ENABLE)
                    else "not_requested"
                ),
                "metrics": {},
            }
        )
        paired_flip = (
            flip_accumulator.finalize()
            if flip_accumulator is not None
            else {
                "format": "paired_prompt_horizontal_flip_v1",
                "valid": False,
                "applicability": (
                    "not_applicable_affinity_probe_disabled"
                    if flip_requested and not affinity_enabled
                    else "not_applicable_no_prompt"
                    if bool(prompt_analysis_cfg.FLIP_ENABLE) and prompt_length <= 0
                    else "not_requested"
                ),
                "metrics": {},
                "by_layer": {},
            }
        )
        paired_flip.update({
            "pair_sample_count": int(flip_pair_sample_count),
            "pairing_sha256": (
                flip_pair_hash.hexdigest() if flip_pair_sample_count > 0 else None
            ),
            "transform": "horizontal_flip_tensor_last_dimension",
        })
        result["prompt_analysis"] = {
            "format": "prompt_analysis_bundle_v1",
            "requested": prompt_analysis_requested,
            "applicability": (
                "applicable"
                if prompt_analysis_enabled
                else "not_applicable_no_prompt"
                if prompt_analysis_requested
                else "not_requested"
            ),
            "source_decomposition": source_decomposition,
            "label_dependency_guard": label_guard,
            "semantic_granularity": {
                "requested": semantic_granularity_requested,
                "applicability": (
                    "applicable"
                    if semantic_granularity_applicable
                    else "not_applicable_missing_attribute_groups_or_reference"
                    if semantic_granularity_requested
                    else "not_requested"
                ),
                "local_attribute_indices": local_attribute_indices,
                "global_attribute_indices": global_attribute_indices,
            },
            "paired_flip": paired_flip,
        }
        for intervention_name, runtime_contract in intervention_runtime_contracts.items():
            sample_count = int(runtime_contract["sample_count"])
            effect = result["module_effects"][intervention_name]
            metric_prefix = (
                "prompt_context_swap"
                if intervention_name == "prompt_context_swapped"
                else "deep_prompt_residual_swap"
                if intervention_name == "deep_prompt_residual_swapped"
                else "instance_prompt_swap"
            )
            effect["summary"].update({
                f"{metric_prefix}_coverage_ratio": float(
                    runtime_contract["swapped_sample_count"]
                    / max(1, sample_count)
                ),
                f"{metric_prefix}_same_class_pair_ratio": float(
                    runtime_contract["same_class_pair_count"]
                    / max(1, sample_count)
                ),
                f"{metric_prefix}_singleton_batch_ratio": float(
                    runtime_contract["singleton_batch_count"]
                    / max(1, int(runtime_contract["batch_count"]))
                ),
            })
            effect["runtime_contract"] = {
                "sample_count": sample_count,
                "swapped_sample_count": int(
                    runtime_contract["swapped_sample_count"]
                ),
                "same_class_pair_count": int(
                    runtime_contract["same_class_pair_count"]
                ),
                "singleton_batch_count": int(
                    runtime_contract["singleton_batch_count"]
                ),
                "batch_count": int(runtime_contract["batch_count"]),
                "pairing_sha256": runtime_contract["pairing_hash"].hexdigest(),
                "pairing_storage": "hash_only",
            }
        for intervention_name, state in prompt_output_intervention_states.items():
            effect = result["module_effects"][intervention_name]
            effect["summary"].update({
                name: float(total / max(1, state["counts"][name]))
                for name, total in sorted(state["sums"].items())
                if state["counts"][name] > 0
            })
        if affinity_enabled:
            affinity_metrics = affinity_metric_accumulator.finalize()
            affinity_equivalence = self._finalize_equivalence_state(affinity_state)
            logit_atol = float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL)
            margin_atol = float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_MARGIN_ATOL)
            affinity_equivalence["affinity_forward_equivalence_pass"] = float(
                affinity_equivalence["logit_max_abs_diff"] <= logit_atol
                and affinity_equivalence["true_margin_abs_diff"] <= margin_atol
                and affinity_equivalence["prediction_flip_rate"] == 0.0
            )
            result["affinity"] = {
                "equivalence": affinity_equivalence,
                "attention_flow_metrics": affinity_metrics["attention_flow"],
                "attention_flow_by_layer": affinity_metrics["attention_flow_by_layer"],
                "affinity_health_metrics": affinity_metrics["affinity_health"],
                "affinity_health_by_layer": affinity_metrics[
                    "affinity_health_by_layer"
                ],
                "prompt_layer_mechanism_by_layer": affinity_metrics[
                    "prompt_layer_mechanism_by_layer"
                ],
                "prompt_patch_bridge_metrics": affinity_metrics[
                    "prompt_patch_bridge"
                ],
                "prompt_patch_bridge_by_layer": affinity_metrics[
                    "prompt_patch_bridge_by_layer"
                ],
                "prompt_content_metrics": affinity_metrics[
                    "prompt_content_metrics"
                ],
                "prompt_content_by_layer": affinity_metrics[
                    "prompt_content_by_layer"
                ],
                "prompt_content_by_layer_and_head": affinity_metrics[
                    "prompt_content_by_layer_and_head"
                ],
                "prompt_content_by_layer_and_prompt": affinity_metrics[
                    "prompt_content_by_layer_and_prompt"
                ],
                "prompt_semantic_role_metrics": affinity_metrics[
                    "prompt_semantic_role"
                ],
                "prompt_semantic_role_by_layer": affinity_metrics[
                    "prompt_semantic_role_by_layer"
                ],
                "prompt_semantic_role_by_layer_and_prompt": affinity_metrics[
                    "prompt_semantic_role_by_layer_and_prompt"
                ],
                "prompt_role_profiles": affinity_metrics["prompt_role_profiles"],
                "attribute_concept_grounding_metrics": affinity_metrics[
                    "attribute_concept_grounding"
                ],
                "attribute_concept_grounding_by_layer": affinity_metrics[
                    "attribute_concept_grounding_by_layer"
                ],
                "attribute_concept_grounding_by_layer_and_prompt": (
                    affinity_metrics[
                        "attribute_concept_grounding_by_layer_and_prompt"
                    ]
                ),
                "cross_layer_concept_continuity_by_pair": affinity_metrics[
                    "cross_layer_concept_continuity_by_pair"
                ],
                "patch_semantic_transport_metrics": affinity_metrics[
                    "patch_semantic_transport"
                ],
                "patch_semantic_transport_by_prompt": affinity_metrics[
                    "patch_semantic_transport_by_prompt"
                ],
                "patch_semantic_transport_reference": (
                    patch_semantic_transport_reference
                ),
                "attribute_concept_reference": {
                    key: value
                    for key, value in attribute_concept_reference.items()
                    if key != "directions"
                },
                "prompt_mechanism_mode": prompt_mechanism_mode,
                "token_metrics": token_accumulator.finalize(),
            }
        intervention_diagnostics = {}
        for intervention_name in sorted(diagnostic_intervention_names):
            intervention_result = intervention_probe_accumulators[
                intervention_name
            ].finalize()
            intervention_probe_accumulators[
                intervention_name
            ].representation.release_covariance()
            alignment_delta = self._probe_metric_delta(
                normal["visual_semantic_alignment"],
                intervention_result["visual_semantic_alignment"],
            )
            effect = result["module_effects"][intervention_name]
            effect["summary"].update(alignment_delta)
            effect["intervention_semantics"] = intervention_semantics[
                intervention_name
            ]
            effect["valid"] = True
            effect["failure_reasons"] = []
            diagnostics = {
                "alignment_metrics": intervention_result[
                    "visual_semantic_alignment"
                ],
                "alignment_delta": alignment_delta,
            }
            if affinity_enabled:
                intervention_affinity_metrics = intervention_affinity_accumulators[
                    intervention_name
                ].finalize()
                intervention_patch_diversity = (
                    intervention_patch_diversity_accumulators[
                        intervention_name
                    ].finalize()
                )
                normal_patch_diversity = result["affinity"]["token_metrics"][
                    "representation_geometry"
                ].get("within_image_patch_diversity", {})
                intervention_equivalence = self._finalize_equivalence_state(
                    intervention_affinity_states[intervention_name]
                )
                logit_atol = float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL)
                margin_atol = float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_MARGIN_ATOL)
                intervention_equivalence_pass = bool(
                    intervention_equivalence["logit_max_abs_diff"] <= logit_atol
                    and intervention_equivalence["true_margin_abs_diff"] <= margin_atol
                    and intervention_equivalence["prediction_flip_rate"] == 0.0
                )
                intervention_equivalence[
                    "affinity_forward_equivalence_pass"
                ] = float(
                    intervention_equivalence_pass
                )
                normal_equivalence_pass = bool(
                    result["affinity"]["equivalence"].get(
                        "affinity_forward_equivalence_pass", 0.0
                    )
                )
                paired_attention_valid = bool(
                    normal_equivalence_pass and intervention_equivalence_pass
                )
                diagnostics["affinity"] = {
                    "equivalence": intervention_equivalence,
                    "attention_flow_metrics": intervention_affinity_metrics[
                        "attention_flow"
                    ],
                    "attention_flow_by_layer": intervention_affinity_metrics[
                        "attention_flow_by_layer"
                    ],
                    "paired_attention_valid": paired_attention_valid,
                    "attention_delta": {},
                    "attention_delta_by_layer": {},
                    "prompt_layer_mechanism_by_layer": (
                        intervention_affinity_metrics[
                            "prompt_layer_mechanism_by_layer"
                        ]
                    ),
                    "prompt_layer_mechanism_delta_by_layer": {},
                    "prompt_patch_bridge_metrics": intervention_affinity_metrics[
                        "prompt_patch_bridge"
                    ],
                    "prompt_patch_bridge_by_layer": intervention_affinity_metrics[
                        "prompt_patch_bridge_by_layer"
                    ],
                    "prompt_patch_bridge_delta": {},
                    "prompt_patch_bridge_delta_by_layer": {},
                    "prompt_semantic_role_metrics": intervention_affinity_metrics[
                        "prompt_semantic_role"
                    ],
                    "prompt_semantic_role_by_layer": intervention_affinity_metrics[
                        "prompt_semantic_role_by_layer"
                    ],
                    "prompt_semantic_role_by_layer_and_prompt": (
                        intervention_affinity_metrics[
                            "prompt_semantic_role_by_layer_and_prompt"
                        ]
                    ),
                    "prompt_semantic_role_delta": {},
                    "prompt_semantic_role_delta_by_layer": {},
                    "attribute_concept_grounding_metrics": (
                        intervention_affinity_metrics[
                            "attribute_concept_grounding"
                        ]
                    ),
                    "attribute_concept_grounding_by_layer": (
                        intervention_affinity_metrics[
                            "attribute_concept_grounding_by_layer"
                        ]
                    ),
                    "attribute_concept_grounding_by_layer_and_prompt": (
                        intervention_affinity_metrics[
                            "attribute_concept_grounding_by_layer_and_prompt"
                        ]
                    ),
                    "cross_layer_concept_continuity_by_pair": (
                        intervention_affinity_metrics[
                            "cross_layer_concept_continuity_by_pair"
                        ]
                    ),
                    "attribute_concept_grounding_delta": {},
                    "attribute_concept_grounding_delta_by_layer": {},
                    "patch_semantic_transport_metrics": (
                        intervention_affinity_metrics[
                            "patch_semantic_transport"
                        ]
                    ),
                    "patch_semantic_transport_by_prompt": (
                        intervention_affinity_metrics[
                            "patch_semantic_transport_by_prompt"
                        ]
                    ),
                    "patch_semantic_transport_delta": {},
                }
                diagnostics["representation_geometry"] = {
                    "within_image_patch_diversity": intervention_patch_diversity,
                    "within_image_patch_diversity_delta": {},
                }
                selection_contract = {
                    "prompt_patch_selection_uniform": {
                        "delta_metric": "delta_prompt_to_patch_mass",
                        "algorithm_metric": "prompt_patch_uniform_mass_abs_error",
                        "applied_metric": "prompt_patch_uniform_applied",
                        "pass_metric": "prompt_patch_mass_preservation_pass",
                        "failure_reason": (
                            "Local Prompt-to-Patch mass preservation was not observed or failed"
                        ),
                    },
                    "patch_prompt_selection_uniform": {
                        "delta_metric": "delta_patch_to_prompt_mass",
                        "algorithm_metric": "patch_prompt_uniform_mass_abs_error",
                        "applied_metric": "patch_prompt_uniform_applied",
                        "pass_metric": "patch_prompt_mass_preservation_pass",
                        "failure_reason": (
                            "Local Patch-to-Prompt mass preservation was not observed or failed"
                        ),
                    },
                    "attribute_concept_prompt_patch_blocked": {
                        "delta_metric": "delta_prompt_to_patch_mass",
                        "pass_metric": (
                            "attribute_concept_prompt_patch_mass_preservation_pass"
                        ),
                        "failure_reason": (
                            "Attribute-concept Prompt-to-Patch mass preservation was not observed"
                        ),
                        "concept_mode": "targeted",
                    },
                    "attribute_concept_random_patch_blocked": {
                        "delta_metric": "delta_prompt_to_patch_mass",
                        "pass_metric": (
                            "attribute_concept_random_patch_mass_preservation_pass"
                        ),
                        "failure_reason": (
                            "Random-control Prompt-to-Patch mass preservation was not observed"
                        ),
                        "concept_mode": "random_control",
                    },
                    "transport_targeted_prompt_patch_blocked": {
                        "delta_metric": "delta_prompt_to_patch_mass",
                        "pass_metric": (
                            "transport_targeted_prompt_patch_mass_preservation_pass"
                        ),
                        "failure_reason": (
                            "Transport-targeted Prompt-to-Patch mass preservation was not observed"
                        ),
                        "transport_mode": "targeted",
                    },
                    "transport_random_patch_blocked": {
                        "delta_metric": "delta_prompt_to_patch_mass",
                        "pass_metric": (
                            "transport_random_prompt_patch_mass_preservation_pass"
                        ),
                        "failure_reason": (
                            "Transport random-control Prompt-to-Patch mass preservation was not observed"
                        ),
                        "transport_mode": "random_control",
                    },
                }.get(intervention_name)
                if paired_attention_valid:
                    if normal_patch_diversity and intervention_patch_diversity:
                        patch_diversity_delta = self._probe_metric_delta(
                            normal_patch_diversity,
                            intervention_patch_diversity,
                        )
                        diagnostics["representation_geometry"][
                            "within_image_patch_diversity_delta"
                        ] = patch_diversity_delta
                        effect["summary"].update({
                            "delta_within_image_patch_{}".format(
                                metric_name[len("delta_"):]
                            ): value
                            for metric_name, value in patch_diversity_delta.items()
                            if metric_name.startswith("delta_")
                        })
                    diagnostics["affinity"]["attention_delta"] = (
                        self._probe_metric_delta(
                            result["affinity"]["attention_flow_metrics"],
                            intervention_affinity_metrics["attention_flow"],
                        )
                    )
                    diagnostics["affinity"]["attention_delta_by_layer"] = (
                        self._probe_layer_metric_delta(
                            result["affinity"]["attention_flow_by_layer"],
                            intervention_affinity_metrics[
                                "attention_flow_by_layer"
                            ],
                        )
                    )
                    diagnostics["affinity"][
                        "prompt_layer_mechanism_delta_by_layer"
                    ] = self._probe_layer_metric_delta(
                        result["affinity"][
                            "prompt_layer_mechanism_by_layer"
                        ],
                        intervention_affinity_metrics[
                            "prompt_layer_mechanism_by_layer"
                        ],
                    )
                    diagnostics["affinity"]["prompt_patch_bridge_delta"] = (
                        self._probe_metric_delta(
                            result["affinity"]["prompt_patch_bridge_metrics"],
                            intervention_affinity_metrics[
                                "prompt_patch_bridge"
                            ],
                        )
                    )
                    diagnostics["affinity"][
                        "prompt_patch_bridge_delta_by_layer"
                    ] = self._probe_layer_metric_delta(
                        result["affinity"]["prompt_patch_bridge_by_layer"],
                        intervention_affinity_metrics[
                            "prompt_patch_bridge_by_layer"
                        ],
                    )
                    diagnostics["affinity"]["prompt_semantic_role_delta"] = (
                        self._probe_metric_delta(
                            result["affinity"]["prompt_semantic_role_metrics"],
                            intervention_affinity_metrics[
                                "prompt_semantic_role"
                            ],
                        )
                    )
                    diagnostics["affinity"][
                        "prompt_semantic_role_delta_by_layer"
                    ] = self._probe_layer_metric_delta(
                        result["affinity"]["prompt_semantic_role_by_layer"],
                        intervention_affinity_metrics[
                            "prompt_semantic_role_by_layer"
                        ],
                    )
                    diagnostics["affinity"][
                        "attribute_concept_grounding_delta"
                    ] = self._probe_metric_delta(
                        result["affinity"][
                            "attribute_concept_grounding_metrics"
                        ],
                        intervention_affinity_metrics[
                            "attribute_concept_grounding"
                        ],
                    )
                    diagnostics["affinity"][
                        "attribute_concept_grounding_delta_by_layer"
                    ] = self._probe_layer_metric_delta(
                        result["affinity"][
                            "attribute_concept_grounding_by_layer"
                        ],
                        intervention_affinity_metrics[
                            "attribute_concept_grounding_by_layer"
                        ],
                    )
                    diagnostics["affinity"][
                        "patch_semantic_transport_delta"
                    ] = self._probe_metric_delta(
                        result["affinity"][
                            "patch_semantic_transport_metrics"
                        ],
                        intervention_affinity_metrics[
                            "patch_semantic_transport"
                        ],
                    )
                    if selection_contract is not None:
                        downstream_mass_delta = diagnostics["affinity"][
                            "attention_delta"
                        ].get(selection_contract["delta_metric"])
                        mass_tolerance = float(
                            self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
                        )
                        algorithm_mass_error = None
                        applied = None
                        algorithm_metric = selection_contract.get(
                            "algorithm_metric"
                        )
                        applied_metric = selection_contract.get(
                            "applied_metric"
                        )
                        if algorithm_metric is not None:
                            algorithm_mass_error = intervention_affinity_metrics[
                                "attention_flow"
                            ].get(algorithm_metric)
                        if applied_metric is not None:
                            applied = intervention_affinity_metrics[
                                "attention_flow"
                            ].get(applied_metric)
                        concept_mode = selection_contract.get("concept_mode")
                        if concept_mode is not None:
                            concept_metrics = intervention_affinity_metrics[
                                "attribute_concept_grounding"
                            ]
                            concept_mass_error = concept_metrics.get(
                                "all.concept_intervention_prompt_patch_mass_abs_error"
                            )
                            applied = concept_metrics.get(
                                "all.concept_intervention_applied"
                            )
                            algorithm_mass_error = concept_mass_error
                            selected_mass_after = concept_metrics.get(
                                "all.concept_intervention_selected_prompt_attention_mass_after"
                            )
                            selection_score_gap = concept_metrics.get(
                                "all.concept_intervention_selection_score_gap"
                            )
                            algorithm_mass_pass = bool(
                                concept_mass_error is not None
                                and abs(float(concept_mass_error)) <= mass_tolerance
                            )
                            selected_path_removed_pass = bool(
                                selected_mass_after is not None
                                and abs(float(selected_mass_after)) <= mass_tolerance
                            )
                            targeted_selection_pass = bool(
                                concept_mode != "targeted"
                                or (
                                    selection_score_gap is not None
                                    and float(selection_score_gap) > 0.0
                                )
                            )
                            diagnostics["concept_intervention_contract"] = {
                                "mode": concept_mode,
                                "algorithm_mass_error": concept_mass_error,
                                "algorithm_mass_pass": algorithm_mass_pass,
                                "selected_mass_after": selected_mass_after,
                                "selected_path_removed_pass": (
                                    selected_path_removed_pass
                                ),
                                "selection_score_gap": selection_score_gap,
                                "targeted_selection_pass": targeted_selection_pass,
                            }
                            effect["summary"].update({
                                "concept_intervention_algorithm_mass_pass": float(
                                    algorithm_mass_pass
                                ),
                                "concept_intervention_selected_path_removed_pass": float(
                                    selected_path_removed_pass
                                ),
                                "concept_intervention_targeted_selection_pass": float(
                                    targeted_selection_pass
                                ),
                            })
                        transport_mode = selection_contract.get(
                            "transport_mode"
                        )
                        if transport_mode is not None:
                            transport_metrics = intervention_affinity_metrics[
                                "patch_semantic_transport"
                            ]
                            transport_mass_error = transport_metrics.get(
                                "all.transport_intervention_prompt_patch_mass_abs_error"
                            )
                            applied = transport_metrics.get(
                                "all.transport_intervention_applied"
                            )
                            algorithm_mass_error = transport_mass_error
                            selected_mass_after = transport_metrics.get(
                                "all.transport_intervention_selected_prompt_attention_mass_after"
                            )
                            selection_score_gap = transport_metrics.get(
                                "all.transport_intervention_selection_score_gap"
                            )
                            algorithm_mass_pass = bool(
                                transport_mass_error is not None
                                and abs(float(transport_mass_error))
                                <= mass_tolerance
                            )
                            selected_path_removed_pass = bool(
                                selected_mass_after is not None
                                and abs(float(selected_mass_after))
                                <= mass_tolerance
                            )
                            targeted_selection_pass = bool(
                                transport_mode != "targeted"
                                or (
                                    selection_score_gap is not None
                                    and float(selection_score_gap) > 0.0
                                )
                            )
                            diagnostics["transport_intervention_contract"] = {
                                "mode": transport_mode,
                                "algorithm_mass_error": transport_mass_error,
                                "algorithm_mass_pass": algorithm_mass_pass,
                                "selected_mass_after": selected_mass_after,
                                "selected_path_removed_pass": (
                                    selected_path_removed_pass
                                ),
                                "selection_score_gap": selection_score_gap,
                                "targeted_selection_pass": targeted_selection_pass,
                            }
                            effect["summary"].update({
                                "transport_intervention_algorithm_mass_pass": float(
                                    algorithm_mass_pass
                                ),
                                "transport_intervention_selected_path_removed_pass": float(
                                    selected_path_removed_pass
                                ),
                                "transport_intervention_targeted_selection_pass": float(
                                    targeted_selection_pass
                                ),
                            })
                        algorithm_mass_pass = bool(
                            algorithm_mass_error is not None
                            and abs(float(algorithm_mass_error)) <= mass_tolerance
                        )
                        applied_pass = bool(
                            applied is not None
                            and float(applied) >= 1.0 - mass_tolerance
                        )
                        mass_preservation_pass = bool(
                            algorithm_mass_pass and applied_pass
                        )
                        diagnostics["selection_mass_preservation"] = {
                            selection_contract["delta_metric"]: downstream_mass_delta,
                            "downstream_mass_delta_is_descriptive": True,
                            "contract_basis": "local_intervention_mass_error",
                            "local_mass_abs_error": algorithm_mass_error,
                            "applied": applied,
                            "applied_pass": applied_pass,
                            "tolerance": mass_tolerance,
                            "pass": mass_preservation_pass,
                        }
                        effect["summary"][selection_contract["pass_metric"]] = float(
                            mass_preservation_pass
                        )
                if intervention_name == "prompt_patch_value_globalized":
                    value_metrics = diagnostics["affinity"][
                        "attention_flow_metrics"
                    ]
                    applied = value_metrics.get(
                        "prompt_value_globalize_applied"
                    )
                    mass_error = value_metrics.get(
                        "prompt_value_globalize_attention_mass_abs_error"
                    )
                    dispersion_after = value_metrics.get(
                        "prompt_value_globalize_patch_value_dispersion_after"
                    )
                    tolerance = float(
                        self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
                    )
                    applied_pass = bool(
                        applied is not None and float(applied) >= 1.0 - tolerance
                    )
                    attention_preserved_pass = bool(
                        mass_error is not None
                        and abs(float(mass_error)) <= tolerance
                    )
                    value_collapsed_pass = bool(
                        dispersion_after is not None
                        and abs(float(dispersion_after)) <= tolerance
                    )
                    value_contract = {
                        "applied": applied,
                        "applied_pass": applied_pass,
                        "attention_mass_abs_error": mass_error,
                        "attention_preserved_pass": attention_preserved_pass,
                        "patch_value_dispersion_after": dispersion_after,
                        "value_collapsed_pass": value_collapsed_pass,
                    }
                    diagnostics["value_globalization_contract"] = value_contract
                    effect["summary"].update({
                        "prompt_value_globalize_applied_pass": float(applied_pass),
                        "prompt_value_globalize_attention_preserved_pass": float(
                            attention_preserved_pass
                        ),
                        "prompt_value_globalize_value_collapsed_pass": float(
                            value_collapsed_pass
                        ),
                    })
                    effect["valid"] = bool(
                        diagnostics["affinity"]["paired_attention_valid"]
                        and applied_pass
                        and attention_preserved_pass
                        and value_collapsed_pass
                    )
                    if not diagnostics["affinity"]["paired_attention_valid"]:
                        effect["failure_reasons"].append(
                            "normal/intervention affinity forward equivalence was not jointly valid"
                        )
                    if not applied_pass:
                        effect["failure_reasons"].append(
                            "Prompt Patch-Value globalization was not applied on all selected layers"
                        )
                    if not attention_preserved_pass:
                        effect["failure_reasons"].append(
                            "Prompt Patch-Value globalization changed its source Attention mass"
                        )
                    if not value_collapsed_pass:
                        effect["failure_reasons"].append(
                            "Prompt Patch-Value globalization did not collapse local Patch Value dispersion"
                        )
                if intervention_name == "prompt_value_zeroed":
                    value_metrics = diagnostics["affinity"][
                        "attention_flow_metrics"
                    ]
                    applied = value_metrics.get("prompt_value_zero_applied")
                    mass_error = value_metrics.get(
                        "prompt_value_zero_attention_mass_abs_error"
                    )
                    context_delta = value_metrics.get(
                        "prompt_value_zero_context_delta_norm"
                    )
                    tolerance = float(
                        self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
                    )
                    applied_pass = bool(
                        applied is not None
                        and float(applied) >= 1.0 - tolerance
                    )
                    attention_preserved_pass = bool(
                        mass_error is not None
                        and abs(float(mass_error)) <= tolerance
                    )
                    diagnostics["prompt_value_zero_contract"] = {
                        "applied": applied,
                        "applied_pass": applied_pass,
                        "attention_mass_abs_error": mass_error,
                        "attention_preserved_pass": attention_preserved_pass,
                        "context_delta_norm": context_delta,
                    }
                    effect["summary"].update({
                        "prompt_value_zero_applied_pass": float(
                            applied_pass
                        ),
                        "prompt_value_zero_attention_preserved_pass": float(
                            attention_preserved_pass
                        ),
                    })
                    effect["valid"] = bool(
                        diagnostics["affinity"]["paired_attention_valid"]
                        and applied_pass
                        and attention_preserved_pass
                    )
                    if not diagnostics["affinity"][
                        "paired_attention_valid"
                    ]:
                        effect["failure_reasons"].append(
                            "normal/intervention affinity forward equivalence was not jointly valid"
                        )
                    if not applied_pass:
                        effect["failure_reasons"].append(
                            "Prompt Value removal was not applied on all selected layers"
                        )
                    if not attention_preserved_pass:
                        effect["failure_reasons"].append(
                            "Prompt Value removal changed Attention probabilities"
                        )
                if intervention_name.startswith("prompt_read_blocked_layer_"):
                    source_layer = int(intervention_name.rsplit("_", 1)[-1])
                    source_metrics = diagnostics["affinity"][
                        "attention_flow_by_layer"
                    ].get(source_layer, {})
                    source_delta = diagnostics["affinity"][
                        "attention_delta_by_layer"
                    ].get(source_layer, {})
                    prompt_state_delta = diagnostics["affinity"][
                        "prompt_layer_mechanism_delta_by_layer"
                    ]
                    source_prompt_state_delta = prompt_state_delta.get(
                        source_layer, {}
                    )
                    applied = source_metrics.get("prompt_read_block_applied")
                    mass_after = source_metrics.get(
                        "prompt_read_block_patch_mass_after"
                    )
                    tolerance = float(
                        self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
                    )
                    applied_pass = bool(
                        applied is not None and float(applied) >= 1.0 - tolerance
                    )
                    source_removed_pass = bool(
                        mass_after is not None
                        and abs(float(mass_after)) <= tolerance
                    )
                    downstream_layers = sorted(
                        layer_index
                        for layer_index in diagnostics["affinity"][
                            "attention_delta_by_layer"
                        ]
                        if int(layer_index) > source_layer
                    )

                    def downstream_mean(metric_name):
                        values = [
                            diagnostics["affinity"][
                                "attention_delta_by_layer"
                            ][layer_index].get(metric_name)
                            for layer_index in downstream_layers
                        ]
                        values = [
                            float(value)
                            for value in values
                            if isinstance(value, (int, float))
                        ]
                        return float(np.mean(values)) if values else None

                    def downstream_prompt_state_mean(metric_name):
                        values = [
                            prompt_state_delta.get(layer_index, {}).get(
                                metric_name
                            )
                            for layer_index in downstream_layers
                        ]
                        values = [
                            float(value)
                            for value in values
                            if isinstance(value, (int, float))
                        ]
                        return float(np.mean(values)) if values else None

                    chain_summary = {
                        "source_layer": source_layer,
                        "source_delta_prompt_to_patch_mass": source_delta.get(
                            "delta_prompt_to_patch_mass"
                        ),
                        "source_delta_prompt_patch_value_contribution_norm": (
                            source_delta.get(
                                "delta_prompt_patch_value_contribution_norm"
                            )
                        ),
                        "source_delta_prompt_layer_change_norm": (
                            source_prompt_state_delta.get(
                                "delta_prompt_layer_change_norm"
                            )
                        ),
                        "source_delta_prompt_layer_output_norm": (
                            source_prompt_state_delta.get(
                                "delta_prompt_layer_output_norm"
                            )
                        ),
                        "source_delta_prompt_layer_output_instance_variance": (
                            source_prompt_state_delta.get(
                                "delta_prompt_layer_output_instance_variance"
                            )
                        ),
                        "downstream_delta_prompt_handoff_gap_mean": (
                            downstream_prompt_state_mean(
                                "delta_prompt_previous_output_to_current_input_gap_norm"
                            )
                        ),
                        "downstream_delta_prompt_layer_output_norm_mean": (
                            downstream_prompt_state_mean(
                                "delta_prompt_layer_output_norm"
                            )
                        ),
                        "downstream_delta_cls_to_prompt_mass_mean": downstream_mean(
                            "delta_cls_to_prompt_mass"
                        ),
                        "downstream_delta_cls_prompt_value_contribution_norm_mean": (
                            downstream_mean(
                                "delta_cls_prompt_value_contribution_norm"
                            )
                        ),
                        "downstream_selected_layers": downstream_layers,
                        "final_delta_semantic_margin": effect["summary"].get(
                            "delta_semantic_margin"
                        ),
                        "final_delta_logits_norm": effect["summary"].get(
                            "delta_logits_norm"
                        ),
                    }
                    diagnostics["read_save_consume_contract"] = {
                        "applied": applied,
                        "applied_pass": applied_pass,
                        "source_prompt_patch_mass_after": mass_after,
                        "source_removed_pass": source_removed_pass,
                        "chain_summary": chain_summary,
                    }
                    effect["summary"].update({
                        "layer_prompt_read_applied_pass": float(applied_pass),
                        "layer_prompt_read_source_removed_pass": float(
                            source_removed_pass
                        ),
                        **{
                            key: value
                            for key, value in chain_summary.items()
                            if isinstance(value, (int, float))
                        },
                    })
                    effect["valid"] = bool(
                        diagnostics["affinity"]["paired_attention_valid"]
                        and applied_pass
                        and source_removed_pass
                    )
                    if not diagnostics["affinity"]["paired_attention_valid"]:
                        effect["failure_reasons"].append(
                            "normal/intervention affinity forward equivalence was not jointly valid"
                        )
                    if not applied_pass:
                        effect["failure_reasons"].append(
                            "Layer-specific Prompt read block was not applied at the declared source layer"
                        )
                    if not source_removed_pass:
                        effect["failure_reasons"].append(
                            "Layer-specific Prompt read block did not remove source-layer Prompt-to-Patch mass"
                        )
                if intervention_name == "prompt_context_swapped":
                    swap_layer = int(
                        self.cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_LAYER
                    )
                    swap_metrics = diagnostics["affinity"][
                        "prompt_layer_mechanism_by_layer"
                    ].get(swap_layer, {})
                    tolerance = float(
                        self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
                    )
                    applied = swap_metrics.get("prompt_context_swap_applied")
                    fixed_point_ratio = swap_metrics.get(
                        "prompt_context_swap_fixed_point_ratio"
                    )
                    coverage = effect["summary"].get(
                        "prompt_context_swap_coverage_ratio"
                    )
                    applied_pass = bool(
                        applied is not None and float(applied) >= 1.0 - tolerance
                    )
                    no_self_pair_pass = bool(
                        fixed_point_ratio is not None
                        and abs(float(fixed_point_ratio)) <= tolerance
                    )
                    full_coverage_pass = bool(
                        coverage is not None
                        and float(coverage) >= 1.0 - tolerance
                    )
                    diagnostics["prompt_context_swap_contract"] = {
                        "swap_layer": swap_layer,
                        "applied": applied,
                        "applied_pass": applied_pass,
                        "fixed_point_ratio": fixed_point_ratio,
                        "no_self_pair_pass": no_self_pair_pass,
                        "coverage_ratio": coverage,
                        "full_coverage_pass": full_coverage_pass,
                        "runtime_contract": effect.get("runtime_contract"),
                    }
                    effect["summary"].update({
                        "prompt_context_swap_applied_pass": float(applied_pass),
                        "prompt_context_swap_no_self_pair_pass": float(
                            no_self_pair_pass
                        ),
                        "prompt_context_swap_full_coverage_pass": float(
                            full_coverage_pass
                        ),
                    })
                    effect["valid"] = bool(
                        diagnostics["affinity"]["paired_attention_valid"]
                        and applied_pass
                        and no_self_pair_pass
                        and full_coverage_pass
                    )
                    if not diagnostics["affinity"]["paired_attention_valid"]:
                        effect["failure_reasons"].append(
                            "normal/intervention affinity forward equivalence was not jointly valid"
                        )
                    if not applied_pass:
                        effect["failure_reasons"].append(
                            "A1 contextualized Prompt swap was not applied at the declared layer"
                        )
                    if not no_self_pair_pass:
                        effect["failure_reasons"].append(
                            "A1 contextualized Prompt swap contained self-pairs"
                        )
                    if not full_coverage_pass:
                        effect["failure_reasons"].append(
                            "A1 contextualized Prompt swap did not cover every probe sample"
                        )
                if selection_contract is not None:
                    selection_check = diagnostics.get(
                        "selection_mass_preservation", {}
                    )
                    selection_valid = bool(
                        diagnostics["affinity"]["paired_attention_valid"]
                        and selection_check.get("pass", False)
                    )
                    concept_contract = diagnostics.get(
                        "concept_intervention_contract"
                    )
                    if concept_contract is not None:
                        selection_valid = bool(
                            selection_valid
                            and concept_contract.get("algorithm_mass_pass", False)
                            and concept_contract.get(
                                "selected_path_removed_pass", False
                            )
                            and concept_contract.get(
                                "targeted_selection_pass", False
                            )
                        )
                    transport_contract = diagnostics.get(
                        "transport_intervention_contract"
                    )
                    if transport_contract is not None:
                        selection_valid = bool(
                            selection_valid
                            and transport_contract.get(
                                "algorithm_mass_pass", False
                            )
                            and transport_contract.get(
                                "selected_path_removed_pass", False
                            )
                            and transport_contract.get(
                                "targeted_selection_pass", False
                            )
                        )
                    effect["valid"] = selection_valid
                    if not diagnostics["affinity"]["paired_attention_valid"]:
                        effect["failure_reasons"].append(
                            "normal/intervention affinity forward equivalence was not jointly valid"
                        )
                    if not selection_check.get("pass", False):
                        effect["failure_reasons"].append(
                            selection_contract["failure_reason"]
                        )
                    if concept_contract is not None:
                        if not concept_contract.get("algorithm_mass_pass", False):
                            effect["failure_reasons"].append(
                                "Concept intervention did not preserve Prompt-to-Patch mass internally"
                            )
                        if not concept_contract.get(
                            "selected_path_removed_pass", False
                        ):
                            effect["failure_reasons"].append(
                                "Selected concept Patch paths were not fully removed"
                            )
                        if not concept_contract.get(
                            "targeted_selection_pass", False
                        ):
                            effect["failure_reasons"].append(
                                "Targeted concept selection did not separate selected and unselected Patch scores"
                            )
                    if transport_contract is not None:
                        if not transport_contract.get(
                            "algorithm_mass_pass", False
                        ):
                            effect["failure_reasons"].append(
                                "Transport intervention did not preserve Prompt-to-Patch mass internally"
                            )
                        if not transport_contract.get(
                            "selected_path_removed_pass", False
                        ):
                            effect["failure_reasons"].append(
                                "Selected transport Patch paths were not fully removed"
                            )
                        if not transport_contract.get(
                            "targeted_selection_pass", False
                        ):
                            effect["failure_reasons"].append(
                                "Targeted transport selection did not separate selected and unselected Patch scores"
                            )
            if intervention_name in prompt_output_intervention_states:
                tolerance = float(
                    self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL
                )
                applied = effect["summary"].get(
                    "prompt_output_intervention_applied"
                )
                applied_pass = bool(
                    applied is not None
                    and float(applied) >= 1.0 - tolerance
                )
                required_zero_fields = {
                    "instance_prompt_zeroed": (
                        "instance_prompt_norm_after",
                    ),
                    "domain_prompt_zeroed": (
                        "domain_prompt_norm_after",
                    ),
                    "both_prompt_zeroed": (
                        "instance_prompt_norm_after",
                        "domain_prompt_norm_after",
                    ),
                }.get(intervention_name, ())
                zero_pass = bool(
                    all(
                        field in effect["summary"]
                        and abs(float(effect["summary"][field])) <= tolerance
                        for field in required_zero_fields
                    )
                )
                fixed_point = effect["summary"].get(
                    "instance_prompt_swap_fixed_point_ratio"
                )
                no_self_pair_pass = bool(
                    intervention_name != "instance_prompt_swapped"
                    or (
                        fixed_point is not None
                        and abs(float(fixed_point)) <= tolerance
                    )
                )
                paired_attention_valid = bool(
                    diagnostics.get("affinity", {}).get(
                        "paired_attention_valid", True
                    )
                )
                diagnostics["prompt_output_intervention_contract"] = {
                    "applied": applied,
                    "applied_pass": applied_pass,
                    "required_zero_fields": list(required_zero_fields),
                    "zero_pass": zero_pass,
                    "fixed_point_ratio": fixed_point,
                    "no_self_pair_pass": no_self_pair_pass,
                    "paired_attention_valid": paired_attention_valid,
                }
                effect["summary"].update({
                    "prompt_output_intervention_applied_pass": float(
                        applied_pass
                    ),
                    "prompt_output_intervention_zero_pass": float(
                        zero_pass
                    ),
                    "instance_prompt_swap_no_self_pair_pass": float(
                        no_self_pair_pass
                    ),
                })
                effect["valid"] = bool(
                    effect.get("valid", True)
                    and applied_pass
                    and zero_pass
                    and no_self_pair_pass
                    and paired_attention_valid
                )
                if not applied_pass:
                    effect["failure_reasons"].append(
                        "Prompt Distributor output intervention was not observed"
                    )
                if not zero_pass:
                    effect["failure_reasons"].append(
                        "Prompt Distributor zero intervention left a non-zero target component"
                    )
                if not no_self_pair_pass:
                    effect["failure_reasons"].append(
                        "Instance Prompt swap contained self-pairs"
                    )
                if not paired_attention_valid:
                    effect["failure_reasons"].append(
                        "normal/intervention affinity forward equivalence was not jointly valid"
                    )
            effect["diagnostic_chain"] = diagnostics
            intervention_diagnostics[intervention_name] = diagnostics
        concept_targeted_name = "attribute_concept_prompt_patch_blocked"
        concept_random_name = "attribute_concept_random_patch_blocked"
        if (
            concept_targeted_name in result["module_effects"]
            and concept_random_name in result["module_effects"]
            and concept_targeted_name in intervention_diagnostics
            and concept_random_name in intervention_diagnostics
        ):
            targeted_effect = result["module_effects"][concept_targeted_name]
            random_effect = result["module_effects"][concept_random_name]
            comparison = {}
            for metric_name in (
                "delta_logits_norm",
                "delta_true_margin",
                "prediction_flip_rate",
                "harmful_flip_rate",
                "delta_entropy",
                "delta_semantic_margin",
            ):
                targeted_value = targeted_effect["summary"].get(metric_name)
                random_value = random_effect["summary"].get(metric_name)
                if isinstance(targeted_value, (int, float)) and isinstance(
                    random_value, (int, float)
                ):
                    comparison[
                        f"targeted_minus_random_{metric_name}"
                    ] = float(targeted_value) - float(random_value)
            targeted_grounding = intervention_diagnostics[concept_targeted_name][
                "affinity"
            ]["attribute_concept_grounding_metrics"]
            random_grounding = intervention_diagnostics[concept_random_name][
                "affinity"
            ]["attribute_concept_grounding_metrics"]
            targeted_score = targeted_grounding.get(
                "all.concept_intervention_selected_score_mean"
            )
            random_score = random_grounding.get(
                "all.concept_intervention_selected_score_mean"
            )
            selection_advantage = (
                float(targeted_score) - float(random_score)
                if isinstance(targeted_score, (int, float))
                and isinstance(random_score, (int, float))
                else None
            )
            selection_separation_pass = bool(
                selection_advantage is not None and selection_advantage > 0.0
            )
            comparison.update({
                "concept_targeted_selection_score_advantage": (
                    selection_advantage
                ),
                "concept_targeted_vs_random_selection_pass": float(
                    selection_separation_pass
                ),
            })
            targeted_effect["summary"].update({
                key: value
                for key, value in comparison.items()
                if value is not None
            })
            if not selection_separation_pass:
                targeted_effect["valid"] = False
                targeted_effect["failure_reasons"].append(
                    "Targeted concept Patch selection did not exceed the equal-count random control"
                )
            intervention_diagnostics[concept_targeted_name][
                "random_control_comparison"
            ] = comparison
            result["attribute_concept_intervention_comparison"] = comparison
        transport_targeted_name = "transport_targeted_prompt_patch_blocked"
        transport_random_name = "transport_random_patch_blocked"
        if (
            transport_targeted_name in result["module_effects"]
            and transport_random_name in result["module_effects"]
            and transport_targeted_name in intervention_diagnostics
            and transport_random_name in intervention_diagnostics
        ):
            targeted_effect = result["module_effects"][transport_targeted_name]
            random_effect = result["module_effects"][transport_random_name]
            comparison = {}
            for metric_name in (
                "delta_logits_norm",
                "delta_true_margin",
                "prediction_flip_rate",
                "harmful_flip_rate",
                "delta_entropy",
                "delta_semantic_margin",
            ):
                targeted_value = targeted_effect["summary"].get(metric_name)
                random_value = random_effect["summary"].get(metric_name)
                if isinstance(targeted_value, (int, float)) and isinstance(
                    random_value, (int, float)
                ):
                    comparison[
                        f"targeted_minus_random_{metric_name}"
                    ] = float(targeted_value) - float(random_value)
            targeted_transport = intervention_diagnostics[
                transport_targeted_name
            ]["affinity"]["patch_semantic_transport_metrics"]
            random_transport = intervention_diagnostics[
                transport_random_name
            ]["affinity"]["patch_semantic_transport_metrics"]
            targeted_score = targeted_transport.get(
                "all.transport_intervention_selected_score_mean"
            )
            random_score = random_transport.get(
                "all.transport_intervention_selected_score_mean"
            )
            selection_advantage = (
                float(targeted_score) - float(random_score)
                if isinstance(targeted_score, (int, float))
                and isinstance(random_score, (int, float))
                else None
            )
            selection_separation_pass = bool(
                selection_advantage is not None and selection_advantage > 0.0
            )
            comparison.update({
                "transport_targeted_selection_score_advantage": (
                    selection_advantage
                ),
                "transport_targeted_vs_random_selection_pass": float(
                    selection_separation_pass
                ),
            })
            targeted_effect["summary"].update({
                key: value
                for key, value in comparison.items()
                if value is not None
            })
            if not selection_separation_pass:
                targeted_effect["valid"] = False
                targeted_effect["failure_reasons"].append(
                    "Targeted transport Patch selection did not exceed the equal-count random control"
                )
            intervention_diagnostics[transport_targeted_name][
                "random_control_comparison"
            ] = comparison
            result["transport_intervention_comparison"] = comparison
        if intervention_diagnostics:
            result["intervention_diagnostics"] = intervention_diagnostics
        if "prompt_zeroed" in intervention_diagnostics:
            result["prompt_zero_diagnostics"] = intervention_diagnostics[
                "prompt_zeroed"
            ]
        for intervention_name, runtime_contract in (
            intervention_runtime_contracts.items()
        ):
            if int(runtime_contract["singleton_batch_count"]) <= 0:
                continue
            effect = result["module_effects"][intervention_name]
            effect["valid"] = False
            reasons = effect.setdefault("failure_reasons", [])
            reason = (
                "fixed-probe swap contained a singleton batch and could not "
                "maintain the no-self-pair contract"
            )
            if reason not in reasons:
                reasons.append(reason)
        component_names = (
            "instance_prompt_zeroed",
            "domain_prompt_zeroed",
            "both_prompt_zeroed",
        )
        if all(
            name in result["module_effects"]
            and bool(result["module_effects"][name].get("valid", True))
            for name in component_names
        ):
            instance_summary = result["module_effects"][
                "instance_prompt_zeroed"
            ]["summary"]
            domain_summary = result["module_effects"][
                "domain_prompt_zeroed"
            ]["summary"]
            both_summary = result["module_effects"]["both_prompt_zeroed"][
                "summary"
            ]
            interaction = {}
            for metric_name in sorted(
                set(instance_summary).intersection(domain_summary, both_summary)
            ):
                values = (
                    instance_summary[metric_name],
                    domain_summary[metric_name],
                    both_summary[metric_name],
                )
                if all(
                    isinstance(value, (int, float, np.integer, np.floating))
                    and math.isfinite(float(value))
                    for value in values
                ):
                    interaction[f"{metric_name}_interaction"] = float(
                        values[2] - values[1] - values[0]
                    )
            result["prompt_component_synergy"] = {
                "formula": "delta_both_zero_minus_delta_domain_zero_minus_delta_instance_zero",
                "metrics": interaction,
            }
        if semantic_enabled:
            synchronized = synchronized_accumulator.finalize()
            mismatched = mismatched_accumulator.finalize()
            synchronized_equivalence = self._finalize_equivalence_state(synchronized_state)
            sync_logit_atol = float(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.SYNC_LOGIT_ATOL)
            sync_margin_atol = float(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.SYNC_MARGIN_ATOL)
            sync_pass = bool(
                synchronized_equivalence["logit_max_abs_diff"] <= sync_logit_atol
                and synchronized_equivalence["true_margin_abs_diff"] <= sync_margin_atol
                and synchronized_equivalence["prediction_flip_rate"] == 0.0
            )
            mismatch_effect = mismatched_effect.finalize()
            normal_classification = normal["classification"]
            mismatch_classification = mismatched["classification"]
            normal_alignment = normal["visual_semantic_alignment"]
            mismatch_alignment = mismatched["visual_semantic_alignment"]
            mismatch_summary = dict(mismatch_effect["summary"])
            mismatch_summary.update({
                "top1_accuracy_drop": float(normal_classification.get("top1", 0.0) - mismatch_classification.get("top1", 0.0)),
                "per_class_accuracy_drop": float(normal_classification.get("per_class", 0.0) - mismatch_classification.get("per_class", 0.0)),
                "semantic_margin_drop": float(normal_alignment.get("semantic_margin", 0.0) - mismatch_alignment.get("semantic_margin", 0.0)),
                "true_prototype_rank_increase": float(mismatch_alignment.get("true_prototype_rank", 0.0) - normal_alignment.get("true_prototype_rank", 0.0)),
                "visual_semantic_structure_drop": float(normal_alignment.get("visual_semantic_structure_spearman", 0.0) - mismatch_alignment.get("visual_semantic_structure_spearman", 0.0)),
            })
            thresholds = {
                "accuracy_drop": float(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.MISMATCH_MIN_ACCURACY_DROP),
                "margin_drop": float(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.MISMATCH_MIN_MARGIN_DROP),
                "rank_increase": float(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.MISMATCH_MIN_RANK_INCREASE),
            }
            checks = {
                "accuracy_drop": mismatch_summary["per_class_accuracy_drop"] > thresholds["accuracy_drop"],
                "margin_drop": mismatch_summary["semantic_margin_drop"] > thresholds["margin_drop"],
                "rank_increase": mismatch_summary["true_prototype_rank_increase"] > thresholds["rank_increase"],
                "structure_drop": mismatch_summary["visual_semantic_structure_drop"] > 0.0,
            }
            mismatch_pass = bool(all(checks.values()))
            failure_reasons = []
            if not sync_pass:
                failure_reasons.append("synchronized permutation was not equivalent after inverse column recovery")
            failure_reasons.extend(
                f"mismatched semantic {name} did not exceed its predeclared threshold"
                for name, passed in checks.items() if not passed
            )
            synchronized_equivalence["synchronized_equivalence_pass"] = float(sync_pass)
            mismatch_summary["mismatched_semantic_effect_pass"] = float(mismatch_pass)
            result["conditions"].update({
                "synchronized_class_permutation": synchronized,
                "mismatched_semantic_permutation": mismatched,
            })
            result["semantic_intervention"] = {
                "format": "semantic_prototype_intervention_v1",
                "conditions": ["normal", "synchronized_class_permutation", "mismatched_semantic_permutation"],
                "permutation_seed": int(permutation_seed),
                "permutation": permutation.tolist(),
                "inverse_permutation": inverse.tolist(),
                "permutation_sha256": permutation_sha256,
                "thresholds": thresholds,
                "synchronized": synchronized_equivalence,
                "mismatched": mismatch_summary,
                "mismatched_per_class": mismatch_effect["per_class"],
                "synchronized_equivalence_pass": sync_pass,
                "mismatched_semantic_effect_pass": mismatch_pass,
                "overall_pass": bool(sync_pass and mismatch_pass),
                "failure_reasons": failure_reasons,
            }
        return result

    def _probe_metric_rows(
        self,
        metrics,
        *,
        checkpoint_id,
        probe_id,
        split,
        domain,
        condition="normal",
        entity_type="split",
        entity_id="all",
        selection_seed=None,
        probe_manifest_sha256=None,
    ):
        return [
            {
                "run_id": self.monitor_manager.run_id,
                "session_id": self.monitor_manager.session_id,
                "checkpoint_id": str(checkpoint_id),
                "probe_id": str(probe_id),
                "selection_seed": None if selection_seed is None else int(selection_seed),
                "probe_manifest_sha256": probe_manifest_sha256,
                "split": str(split),
                "condition": str(condition),
                "domain": str(domain),
                "entity_type": str(entity_type),
                "entity_id": str(entity_id),
                "metric": str(name),
                "value": float(value),
            }
            for name, value in sorted(metrics.items())
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))
        ]

    def _prompt_role_metric_rows(
        self,
        metrics,
        *,
        checkpoint_id,
        probe_id,
        split,
        layer_index,
        prompt_index,
        condition="normal",
        selection_seed=None,
        probe_manifest_sha256=None,
    ):
        class_id_suffix = "_class_local_id"
        continuous_metrics = {
            name: value
            for name, value in metrics.items()
            if not str(name).endswith(class_id_suffix)
        }
        rows = self._probe_metric_rows(
            continuous_metrics,
            checkpoint_id=checkpoint_id,
            probe_id=probe_id,
            split=split,
            condition=condition,
            domain="prompt_semantic_role_reference",
            entity_type="layer_prompt",
            entity_id=f"layer_{layer_index}/prompt_{prompt_index}",
            selection_seed=selection_seed,
            probe_manifest_sha256=probe_manifest_sha256,
        )
        for metric_name, class_id in sorted(metrics.items()):
            match = re.fullmatch(
                r"(cls_consumption|patch_collection)_top(\d+)_class_local_id",
                str(metric_name),
            )
            if match is None:
                continue
            role_name, rank = match.groups()
            share_name = f"{role_name}_top{rank}_class_share"
            share = metrics.get(share_name)
            if share is None:
                continue
            rows.extend(self._probe_metric_rows(
                {"class_share": share},
                checkpoint_id=checkpoint_id,
                probe_id=probe_id,
                split=split,
                condition=condition,
                domain="prompt_semantic_role_reference",
                entity_type="layer_prompt_class",
                entity_id=(
                    f"layer_{layer_index}/prompt_{prompt_index}/{role_name}/"
                    f"top{rank}/class_local_{int(class_id)}"
                ),
                selection_seed=selection_seed,
                probe_manifest_sha256=probe_manifest_sha256,
            ))
        return rows

    def _attribute_concept_metric_rows(
        self,
        metrics,
        *,
        checkpoint_id,
        probe_id,
        split,
        layer_index,
        prompt_index,
        condition="normal",
        selection_seed=None,
        probe_manifest_sha256=None,
    ):
        continuous_metrics = {
            name: value
            for name, value in metrics.items()
            if not str(name).endswith("_attribute_id")
        }
        rows = self._probe_metric_rows(
            continuous_metrics,
            checkpoint_id=checkpoint_id,
            probe_id=probe_id,
            split=split,
            condition=condition,
            domain="attribute_concept_grounding_reference",
            entity_type="layer_prompt",
            entity_id=f"layer_{layer_index}/prompt_{prompt_index}",
            selection_seed=selection_seed,
            probe_manifest_sha256=probe_manifest_sha256,
        )
        for metric_name, attribute_id in sorted(metrics.items()):
            match = re.fullmatch(
                r"(prompt_true|prompt_margin|collection_true|collection_margin)_top(\d+)_attribute_id",
                str(metric_name),
            )
            if match is None:
                continue
            profile_name, rank = match.groups()
            share = metrics.get(
                f"{profile_name}_top{rank}_attribute_share"
            )
            if share is None:
                continue
            rows.extend(self._probe_metric_rows(
                {"attribute_share": share},
                checkpoint_id=checkpoint_id,
                probe_id=probe_id,
                split=split,
                condition=condition,
                domain="attribute_concept_grounding_reference",
                entity_type="layer_prompt_attribute",
                entity_id=(
                    f"layer_{layer_index}/prompt_{prompt_index}/{profile_name}/"
                    f"top{rank}/attribute_{int(attribute_id)}"
                ),
                selection_seed=selection_seed,
                probe_manifest_sha256=probe_manifest_sha256,
            ))
        return rows

    def _patch_semantic_transport_metric_rows(
        self,
        metrics,
        *,
        checkpoint_id,
        probe_id,
        split,
        prompt_index,
        condition="normal",
        selection_seed=None,
        probe_manifest_sha256=None,
    ):
        continuous_metrics = {
            name: value
            for name, value in metrics.items()
            if not str(name).endswith("_class_local_id")
            and not str(name).endswith("_attribute_id")
        }
        rows = self._probe_metric_rows(
            continuous_metrics,
            checkpoint_id=checkpoint_id,
            probe_id=probe_id,
            split=split,
            condition=condition,
            domain="patch_semantic_transport_reference",
            entity_type="prompt_transport_profile",
            entity_id=f"final_layer/prompt_{prompt_index}",
            selection_seed=selection_seed,
            probe_manifest_sha256=probe_manifest_sha256,
        )
        identity_specs = (
            (
                r"(attention|av)_top(\d+)_class_local_id",
                "class_share",
                "prompt_transport_class",
                "class_local",
            ),
            (
                r"(attention|av)_top(\d+)_attribute_id",
                "attribute_share",
                "prompt_transport_attribute",
                "attribute",
            ),
        )
        for metric_name, identity in sorted(metrics.items()):
            for pattern, share_metric, entity_type, identity_prefix in identity_specs:
                match = re.fullmatch(pattern, str(metric_name))
                if match is None:
                    continue
                profile_name, rank = match.groups()
                share = metrics.get(
                    f"{profile_name}_top{rank}_{share_metric}"
                )
                if share is None:
                    break
                rows.extend(self._probe_metric_rows(
                    {share_metric: share},
                    checkpoint_id=checkpoint_id,
                    probe_id=probe_id,
                    split=split,
                    condition=condition,
                    domain="patch_semantic_transport_reference",
                    entity_type=entity_type,
                    entity_id=(
                        f"final_layer/prompt_{prompt_index}/{profile_name}/"
                        f"top{rank}/{identity_prefix}_{int(identity)}"
                    ),
                    selection_seed=selection_seed,
                    probe_manifest_sha256=probe_manifest_sha256,
                ))
                break
        return rows

    @staticmethod
    def _true_margin_numpy(logits, targets):
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)
        true = logits[np.arange(targets.size), targets]
        other = logits.copy()
        other[np.arange(targets.size), targets] = -np.inf
        return true - other.max(axis=1)

    def _record_probe_module_effect(
        self,
        *,
        checkpoint_id,
        checkpoint_manifest,
        split,
        manifest,
        intervention_name,
        effect,
        artifact_prefix="",
    ):
        pair_id = f"{checkpoint_id}:{manifest['probe_id']}:{intervention_name}"
        relative_path = f"module_effect/{checkpoint_id}/{split}_{intervention_name}_summary.json"
        if artifact_prefix:
            relative_path = f"{str(artifact_prefix).strip('/')}/{relative_path}"
        self.diagnostic_manager.record_module_effect_artifact(
            relative_path,
            {
                "format": "baseline_module_effect_summary_v9",
                "pair_id": pair_id,
                "checkpoint": checkpoint_manifest,
                "probe_id": manifest["probe_id"],
                "probe_manifest_sha256": manifest["manifest_sha256"],
                "condition": str(intervention_name),
                "runtime_overrides": {"intervention": str(intervention_name)},
                "model_eval": True,
                "torch_no_grad": True,
                "dtype": str(next(self._model_ref(self.model).parameters()).dtype),
                "device": str(self.device),
                "intervention_seed": int(manifest["selection_seed"]),
                "shared_randomness_id": manifest["manifest_sha256"],
                "comparison_tolerance": float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL),
                "summary": effect["summary"],
                "per_class": effect["per_class"],
                "intervention_semantics": effect.get("intervention_semantics", {}),
                "runtime_contract": effect.get("runtime_contract"),
                "diagnostic_chain": effect.get("diagnostic_chain", {}),
                "valid": bool(effect.get("valid", True)),
                "failure_reasons": list(effect.get("failure_reasons", [])),
            },
        )

    def _run_fixed_probes(
        self,
        train_loader,
        test_seen_loader,
        test_unseen_loader,
        *,
        checkpoint_epoch,
        selection_seed=None,
        artifact_prefix="",
        execution_profile=None,
    ):
        if not self.diagnostic_manager.enabled or not bool(self.cfg.MONITOR.PROBE.ENABLE) or du.get_rank() != 0:
            return
        execution_profile = self._fixed_probe_execution_profile(
            execution_profile
            if execution_profile is not None
            else self.cfg.MONITOR.PROBE.EXECUTION_PROFILE
        )
        if selection_seed is None:
            selection_seeds = [int(self.cfg.MONITOR.PROBE.SELECTION_SEED)]
            selection_seeds.extend(int(item) for item in self.cfg.MONITOR.PROBE.ROBUSTNESS_SELECTION_SEEDS)
            selection_seeds = list(dict.fromkeys(selection_seeds))
            if any(seed < 0 for seed in selection_seeds):
                raise ValueError("fixed-probe selection seeds must be non-negative")
            executions = []
            base_prefix = str(artifact_prefix or "").strip("/")
            for index, current_seed in enumerate(selection_seeds):
                suffix = "" if index == 0 else f"fixed_probe_robustness/selection_seed_{current_seed}"
                current_profile = (
                    "robustness_core"
                    if index > 0 and execution_profile == "final_full"
                    else execution_profile
                )
                current_prefix = "/".join(
                    item for item in (base_prefix, suffix) if item
                )
                executions.append(self._run_fixed_probes(
                    train_loader,
                    test_seen_loader,
                    test_unseen_loader,
                    checkpoint_epoch=checkpoint_epoch,
                    selection_seed=current_seed,
                    artifact_prefix=current_prefix,
                    execution_profile=current_profile,
                ))
            self.diagnostic_manager.record_probe_artifact(
                "/".join(
                    item for item in (base_prefix, "probe_robustness_manifest.json") if item
                ),
                {
                    "format": "baseline_fixed_probe_robustness_collection_v1",
                    "primary_selection_seed": selection_seeds[0],
                    "selection_seeds": selection_seeds,
                    "execution_count": len(executions),
                    "executions": executions,
                    "primary_execution_profile": execution_profile,
                    "execution_profiles": [
                        execution.get("execution_profile")
                        for execution in executions
                        if isinstance(execution, dict)
                    ],
                    "storage_mode": "aggregate_only",
                },
            )
            return
        selection_seed = int(selection_seed)
        artifact_prefix = str(artifact_prefix or "").strip("/")
        if execution_profile == "final_full" and bool(
            self.cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE
        ) and not bool(self.cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE):
            raise ValueError(
                "MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE requires "
                "MONITOR.PROBE.TARGET_RELEVANCE.ENABLE"
            )

        def artifact_path(relative_path):
            relative_path = str(relative_path).lstrip("/")
            return f"{artifact_prefix}/{relative_path}" if artifact_prefix else relative_path

        model_ref = self._model_ref(self.model)
        was_training = bool(model_ref.training)
        model_ref.eval()
        checkpoint_source = dict(self._fixed_probe_checkpoint_source or {})
        checkpoint_id = str(checkpoint_source.get(
            "checkpoint_id", f"final_epoch_{int(checkpoint_epoch):04d}"
        ))
        checkpoint_path = checkpoint_source.get(
            "checkpoint_path", self._final_trainable_checkpoint_path
        )
        checkpoint_manifest = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_epoch": int(checkpoint_epoch),
            "checkpoint_global_step": int(
                checkpoint_source.get("checkpoint_global_step", self._trace_global_step)
            ),
            "checkpoint_selection_rule": checkpoint_source.get(
                "checkpoint_selection_rule", "predeclared_final_epoch"
            ),
            "source_run_id": checkpoint_source.get(
                "source_run_id", self.monitor_manager.run_id
            ),
            "source_session_id": checkpoint_source.get(
                "source_session_id", self.monitor_manager.session_id
            ),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": (
                checkpoint_sha256(checkpoint_path)
                if checkpoint_path and os.path.isfile(checkpoint_path)
                else None
            ),
        }
        if checkpoint_source:
            checkpoint_manifest.update({
                "diagnostic_replay": True,
                "diagnostic_execution_run_id": self.monitor_manager.run_id,
                "diagnostic_execution_session_id": self.monitor_manager.session_id,
            })
            expected_sha256 = checkpoint_source.get("checkpoint_sha256")
            if (
                expected_sha256 is not None
                and checkpoint_manifest["checkpoint_sha256"] != expected_sha256
            ):
                raise RuntimeError(
                    "fixed-probe replay checkpoint hash mismatch: expected={} actual={}".format(
                        expected_sha256, checkpoint_manifest["checkpoint_sha256"]
                    )
                )
        split_sources = {
            "probe_train_seen": train_loader.dataset if train_loader is not None else None,
            "probe_test_seen": test_seen_loader.dataset if test_seen_loader is not None else None,
            "probe_test_unseen": test_unseen_loader.dataset if test_unseen_loader is not None else None,
        }
        if execution_profile in {"milestone_core", "robustness_core"}:
            split_sources = {
                "probe_test_unseen": split_sources["probe_test_unseen"]
            }
        output_root_reference = "../" * (1 + len([item for item in artifact_prefix.split("/") if item]))
        combined_manifest = {
            "format": "baseline_fixed_probe_collection_v11",
            "run_id": self.monitor_manager.run_id,
            "session_id": self.monitor_manager.session_id,
            "selection_seed": selection_seed,
            "execution_profile": execution_profile,
            "analysis_role": (
                "final_full_evidence"
                if execution_profile == "final_full"
                else "diagnostic_mechanism_core"
            ),
            "required_splits": [
                split for split, dataset in split_sources.items() if dataset is not None
            ],
            "checkpoint": checkpoint_manifest,
            "storage_contract": {
                "mode": "aggregate_only",
                "sample_logits_persisted": False,
                "sample_vectors_persisted": False,
                "batch_state_released_after_update": True,
            },
            "identity_references": {
                "resolved_config": f"{output_root_reference}resolved_config.yaml",
                "reproducibility_manifest": f"{output_root_reference}reproducibility_manifest.json",
                "dataset_manifest": f"{output_root_reference}dataset_manifest.json",
                "comparability": "comparability.json",
            },
            "probes": {},
            "validity": {},
            "semantic_interventions": {},
            "target_relevance": {},
            "explanation_validity": {},
            "prompt_analysis": {},
            "bayesian_object_selection": {},
            "patch_semantic_transport": {},
        }
        normal_results = {}
        semantic_results = {}
        class_aggregates = {}
        intervention_diagnostic_status = {}
        target_relevance_status = {}
        bayesian_object_selection_status = {}
        bayesian_object_selection_reports = {}
        all_rows = _StreamingProbeMetricRows(self.diagnostic_manager)
        probe_loader_settings = self._fixed_probe_loader_settings()
        probe_timing_by_split = {}
        deterministic_transform = get_transforms("test_seen", self.cfg.DATA.CROPSIZE)
        prepared_probes = []
        for split, dataset in split_sources.items():
            if dataset is None:
                continue
            candidate_class_ids = (
                list(dataset.seen_classes) + list(dataset.unseen_classes)
                if str(self.cfg.DATA.XLSA.PROTOCOL_MODE).lower() == "final_gzsl"
                else list(dataset.eval_local_classes)
            )
            manifest = build_probe_manifest(
                dataset,
                split=split,
                per_class=int(self.cfg.MONITOR.PROBE.PER_CLASS),
                max_samples=int(self.cfg.MONITOR.PROBE.MAX_SAMPLES),
                selection_seed=selection_seed,
                candidate_class_ids=candidate_class_ids,
            )
            validity = validate_probe_manifest(
                manifest,
                require_full_class_coverage=bool(self.cfg.MONITOR.PROBE.REQUIRE_FULL_CLASS_COVERAGE),
                require_per_class_quota=bool(self.cfg.MONITOR.PROBE.REQUIRE_PER_CLASS_QUOTA),
                allow_max_samples_truncation=bool(self.cfg.MONITOR.PROBE.ALLOW_MAX_SAMPLES_TRUNCATION),
            )
            combined_manifest["probes"][split] = manifest
            combined_manifest["validity"][split] = validity
            if not artifact_prefix:
                self._probe_manifests[split] = manifest
            self.diagnostic_manager.record_probe_artifact(
                artifact_path(f"probe_manifests/{split}.json"), manifest
            )
            self.diagnostic_manager.record_probe_artifact(
                artifact_path(f"probe_validity/{split}.json"), validity
            )
            prepared_probes.append((split, dataset, candidate_class_ids, manifest))

        overall_probe_valid = bool(prepared_probes) and all(
            bool(payload["valid"]) for payload in combined_manifest["validity"].values()
        )
        validity_collection = {
            "format": "baseline_fixed_probe_validity_collection_v1",
            "selection_seed": selection_seed,
            "policy": {
                "require_full_class_coverage": bool(self.cfg.MONITOR.PROBE.REQUIRE_FULL_CLASS_COVERAGE),
                "require_per_class_quota": bool(self.cfg.MONITOR.PROBE.REQUIRE_PER_CLASS_QUOTA),
                "allow_max_samples_truncation": bool(self.cfg.MONITOR.PROBE.ALLOW_MAX_SAMPLES_TRUNCATION),
            },
            "splits": combined_manifest["validity"],
            "valid": overall_probe_valid,
            "failure_reasons": [
                f"{split}: {reason}"
                for split, payload in combined_manifest["validity"].items()
                for reason in payload["failure_reasons"]
            ],
        }
        self.diagnostic_manager.record_probe_artifact(
            artifact_path("probe_validity.json"), validity_collection
        )
        if not overall_probe_valid:
            self.diagnostic_manager.record_probe_artifact(
                artifact_path("probe_manifest.json"), combined_manifest
            )
            self.diagnostic_manager.record_probe_artifact(
                artifact_path("probe_runtime_summary.json"),
                {
                    "status": "invalid_probe_manifest",
                    "selection_seed": selection_seed,
                    "execution_profile": execution_profile,
                    "checkpoint": checkpoint_manifest,
                    "probe_count": int(len(combined_manifest["probes"])),
                    "required_splits": list(combined_manifest["required_splits"]),
                    "metric_row_count": 0,
                    "storage_mode": "aggregate_only",
                    "probe_loader": probe_loader_settings,
                    "failure_reasons": validity_collection["failure_reasons"],
                },
            )
            if was_training:
                model_ref.train()
            raise RuntimeError(
                "fixed-probe manifest validation failed: "
                + "; ".join(validity_collection["failure_reasons"])
            )

        for split, dataset, candidate_class_ids, manifest in prepared_probes:
            probe_timing_by_split[split] = {}
            probe_loader = self._build_fixed_probe_loader(
                FixedProbeDataset(dataset, manifest, deterministic_transform),
                self.cfg.MONITOR.PROBE.BATCH_SIZE,
            )
            bundle, stage_timing = self._execute_timed_probe_stage(
                probe_loader,
                lambda timed_loader: self._execute_fixed_probe_bundle(
                    timed_loader,
                    dataset,
                    candidate_class_ids,
                    split=split,
                    execution_profile=execution_profile,
                ),
            )
            probe_timing_by_split[split]["fixed_probe_bundle"] = stage_timing
            if execution_profile == "final_full" and bool(
                self.cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION.ENABLE
            ):
                object_loader = self._build_fixed_probe_loader(
                    FixedProbeDataset(dataset, manifest, deterministic_transform),
                    self.cfg.MONITOR.PROBE.BATCH_SIZE,
                )
                object_result, stage_timing = self._execute_timed_probe_stage(
                    object_loader,
                    lambda timed_loader: self._execute_bayesian_object_selection_probe(
                        timed_loader,
                        dataset,
                        candidate_class_ids,
                        split=split,
                    ),
                )
                probe_timing_by_split[split][
                    "bayesian_object_selection"
                ] = stage_timing
                object_root = f"bayesian_object_selection/{checkpoint_id}"
                object_artifacts = {}
                for artifact_name, payload_name in (
                    ("candidate_registry", "registry"),
                    ("functional_geometry", "functional_geometry"),
                    ("hierarchy_trace", "hierarchy_trace"),
                    ("source_retention", "source_retention"),
                    ("candidate_report", "object_selection_report"),
                ):
                    payload = object_result.get(payload_name)
                    if not isinstance(payload, dict):
                        continue
                    path = artifact_path(
                        f"{object_root}/{split}_{artifact_name}.json"
                    )
                    self.diagnostic_manager.record_probe_artifact(
                        path,
                        {
                            **payload,
                            "checkpoint": checkpoint_manifest,
                            "probe_id": manifest["probe_id"],
                            "probe_manifest_sha256": manifest[
                                "manifest_sha256"
                            ],
                            "selection_seed": selection_seed,
                            "split": split,
                            "candidate_class_ids": [
                                int(item) for item in candidate_class_ids
                            ],
                        },
                    )
                    object_artifacts[artifact_name] = path
                status = {
                    "requested": True,
                    "applicability": object_result.get("applicability"),
                    "observed": bool(object_result.get("observed", False)),
                    "valid": object_result.get("valid"),
                    "failure_reason": object_result.get("failure_reason"),
                    "artifacts": object_artifacts,
                    "posterior_interpretation_allowed": False,
                }
                bayesian_object_selection_status[split] = status
                combined_manifest["bayesian_object_selection"][split] = dict(
                    status
                )
                object_report = object_result.get("object_selection_report")
                if isinstance(object_report, dict):
                    bayesian_object_selection_reports[split] = object_report
                hierarchy_trace = object_result.get("hierarchy_trace", {})
                all_rows.extend(self._probe_metric_rows(
                    numeric_leaf_metrics(
                        {
                            "distance_correspondence": hierarchy_trace.get(
                                "distance_correspondence", {}
                            ),
                            "normalized_trace_variance": hierarchy_trace.get(
                                "normalized_trace_variance", {}
                            ),
                            "propagation": hierarchy_trace.get(
                                "propagation", {}
                            ),
                            "collapse": hierarchy_trace.get("collapse", {}),
                            "functional_null": hierarchy_trace.get(
                                "perturbation_functional_null_direction_ratio", {}
                            ),
                            "reference_distance_by_stage": hierarchy_trace.get(
                                "reference_distance_by_stage", {}
                            ),
                            "prediction_effect": hierarchy_trace.get(
                                "prediction_effect", {}
                            ),
                        }
                    ),
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="controlled_perturbation",
                    domain="bayesian_object_selection",
                    entity_type="hierarchy_trace",
                    entity_id="source_to_prediction",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
            if bool(self.cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE):
                relevance_loader = self._build_fixed_probe_loader(
                    FixedProbeDataset(dataset, manifest, deterministic_transform),
                    self.cfg.MONITOR.PROBE.TARGET_RELEVANCE.BATCH_SIZE,
                )
                target_relevance, stage_timing = self._execute_timed_probe_stage(
                    relevance_loader,
                    lambda timed_loader: self._execute_target_relevance_probe(
                        timed_loader,
                        dataset,
                        candidate_class_ids,
                        split=split,
                        include_explanation_validity=(
                            execution_profile == "final_full"
                        ),
                    ),
                )
                probe_timing_by_split[split]["target_relevance"] = stage_timing
                relevance_path = artifact_path(
                    f"target_relevance/{checkpoint_id}/{split}_summary.json"
                )
                self.diagnostic_manager.record_probe_artifact(
                    relevance_path,
                    {
                        **target_relevance,
                        "checkpoint": checkpoint_manifest,
                        "probe_id": manifest["probe_id"],
                        "probe_manifest_sha256": manifest["manifest_sha256"],
                        "selection_seed": selection_seed,
                    },
                )
                target_relevance_status[split] = {
                    "requested": True,
                    "valid": bool(target_relevance["valid"]),
                    "artifact_path": relevance_path,
                    "observed_layers": target_relevance["observed_layers"],
                    "missing_layers": target_relevance["missing_layers"],
                    "failure_reasons": target_relevance["failure_reasons"],
                    "objectives": {
                        name: {
                            "applicability": result["applicability"],
                            "valid": result["valid"],
                            "observed_layers": result["observed_layers"],
                            "missing_layers": result["missing_layers"],
                            "failure_reasons": result["failure_reasons"],
                        }
                        for name, result in target_relevance[
                            "objective_results"
                        ].items()
                    },
                }
                combined_manifest["target_relevance"][split] = dict(
                    target_relevance_status[split]
                )
                explanation_validity = target_relevance.get(
                    "explanation_validity", {}
                )
                if explanation_validity.get("requested", False):
                    explanation_path = artifact_path(
                        f"explanation_validity/{checkpoint_id}/{split}_summary.json"
                    )
                    self.diagnostic_manager.record_probe_artifact(
                        explanation_path,
                        {
                            **explanation_validity,
                            "checkpoint": checkpoint_manifest,
                            "probe_id": manifest["probe_id"],
                            "probe_manifest_sha256": manifest[
                                "manifest_sha256"
                            ],
                            "selection_seed": selection_seed,
                        },
                    )
                    combined_manifest["explanation_validity"][split] = {
                        "requested": True,
                        "valid": explanation_validity.get("valid"),
                        "directional_support_pass": explanation_validity.get(
                            "directional_support_pass"
                        ),
                        "artifact_path": explanation_path,
                        "failure_reasons": explanation_validity.get(
                            "failure_reasons", []
                        ),
                    }
                    for layer_index, conditions in sorted(
                        explanation_validity.get(
                            "by_layer_condition", {}
                        ).items()
                    ):
                        for condition, curve in sorted(conditions.items()):
                            all_rows.extend(self._probe_metric_rows(
                                {
                                    "margin_drop_auc": curve[
                                        "margin_drop_auc"
                                    ],
                                    "accuracy_drop_auc": curve[
                                        "accuracy_drop_auc"
                                    ],
                                },
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=f"{condition}_relevance_deleted",
                                domain="explanation_validity",
                                entity_type="layer_curve",
                                entity_id=f"layer_{layer_index}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest[
                                    "manifest_sha256"
                                ],
                            ))
                            for point in curve.get("points", []):
                                fraction = float(point["fraction"])
                                all_rows.extend(self._probe_metric_rows(
                                    {
                                        name: value
                                        for name, value in point.items()
                                        if name != "fraction"
                                    },
                                    checkpoint_id=checkpoint_id,
                                    probe_id=manifest["probe_id"],
                                    split=split,
                                    condition=(
                                        f"{condition}_relevance_deleted"
                                    ),
                                    domain="explanation_validity",
                                    entity_type="layer_fraction",
                                    entity_id=(
                                        f"layer_{layer_index}/fraction_{fraction:g}"
                                    ),
                                    selection_seed=selection_seed,
                                    probe_manifest_sha256=manifest[
                                        "manifest_sha256"
                                    ],
                                ))
                    for layer_index, metrics in sorted(
                        explanation_validity.get(
                            "comparisons_by_layer", {}
                        ).items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition="relevance_ranked_vs_random",
                            domain="explanation_validity",
                            entity_type="layer_control",
                            entity_id=f"layer_{layer_index}",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest[
                                "manifest_sha256"
                            ],
                        ))
                all_rows.extend(self._probe_metric_rows(
                    target_relevance["equivalence"],
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="gradient_forward",
                    domain="target_relevance_forward_equivalence",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
                true_objective = target_relevance["objective_results"][
                    "true_class_margin"
                ]
                if true_objective["valid"]:
                    all_rows.extend(self._probe_metric_rows(
                        true_objective["target_summary"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="target_relevance_reference",
                        entity_type="objective",
                        entity_id="true_class_margin",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    for layer_index, path_metrics in sorted(
                        true_objective["by_layer"].items()
                    ):
                        for path_name, metrics in sorted(path_metrics.items()):
                            all_rows.extend(self._probe_metric_rows(
                                metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                domain="target_relevance_reference",
                                entity_type="layer_path",
                                entity_id=f"layer_{layer_index}/{path_name}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                predicted_objective = target_relevance["objective_results"][
                    "predicted_class_margin"
                ]
                if predicted_objective["valid"]:
                    all_rows.extend(self._probe_metric_rows(
                        predicted_objective["target_summary"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="target_relevance_reference",
                        entity_type="objective",
                        entity_id="predicted_class_margin",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    for layer_index, path_metrics in sorted(
                        predicted_objective["by_layer"].items()
                    ):
                        for path_name, metrics in sorted(path_metrics.items()):
                            all_rows.extend(self._probe_metric_rows(
                                metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                domain="target_relevance_reference",
                                entity_type="objective_layer_path",
                                entity_id=(
                                    f"predicted_class_margin/layer_{layer_index}/"
                                    f"{path_name}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                if not target_relevance["valid"]:
                    logger.warning(
                        "[fixed-probe] target relevance incomplete for split=%s: %s",
                        split,
                        "; ".join(target_relevance["failure_reasons"]),
                    )
            normal_results[split] = {
                **bundle["normal"],
                "accumulator": bundle["normal_accumulator"],
                "prompt_response": {},
            }
            class_aggregates[split] = {
                "probe_id": manifest["probe_id"],
                "probe_manifest_sha256": manifest["manifest_sha256"],
                "selection_seed": selection_seed,
                "normal": bundle["normal_accumulator"].class_aggregates(),
            }
            prompt_analysis = bundle.get("prompt_analysis", {})
            if prompt_analysis.get("requested", False):
                prompt_analysis_status = {
                    "format": prompt_analysis.get(
                        "format", "prompt_analysis_bundle_v1"
                    ),
                    "requested": True,
                    "applicability": prompt_analysis.get("applicability"),
                    "source_decomposition": {},
                    "label_dependency_guard": {},
                    "semantic_granularity": prompt_analysis.get(
                        "semantic_granularity", {}
                    ),
                    "paired_flip": {},
                    "content": {
                        "requested": bool(
                            self.cfg.MONITOR.PROBE.PROMPT_ANALYSIS.CONTENT_ENABLE
                        ),
                        "observed": False,
                        "valid": None,
                    },
                    "role_profile": {
                        "requested": bool(
                            self.cfg.MONITOR.PROBE.PROMPT_ANALYSIS.ROLE_PROFILE_EXPORT_ENABLE
                        ),
                        "observed": False,
                        "valid": None,
                    },
                }
                prompt_affinity = bundle.get("affinity")
                if prompt_affinity is not None:
                    prompt_equivalence_pass = bool(
                        prompt_affinity.get("equivalence", {}).get(
                            "affinity_forward_equivalence_pass", 0.0
                        )
                    )
                    if prompt_analysis_status["content"]["requested"]:
                        content_observed = bool(
                            prompt_affinity.get(
                                "prompt_content_by_layer", {}
                            )
                        )
                        prompt_analysis_status["content"].update({
                            "observed": content_observed,
                            "valid": bool(
                                prompt_equivalence_pass and content_observed
                            ),
                            "reason": (
                                None
                                if prompt_equivalence_pass and content_observed
                                else "affinity_forward_equivalence_failed"
                                if not prompt_equivalence_pass
                                else "prompt_content_not_observed"
                            ),
                        })
                    if prompt_analysis_status["role_profile"]["requested"]:
                        profile_observed = bool(
                            prompt_affinity.get("prompt_role_profiles", {})
                        )
                        prompt_analysis_status["role_profile"].update({
                            "observed": profile_observed,
                            "valid": bool(
                                prompt_equivalence_pass and profile_observed
                            ),
                            "reason": (
                                None
                                if prompt_equivalence_pass and profile_observed
                                else "affinity_forward_equivalence_failed"
                                if not prompt_equivalence_pass
                                else "prompt_role_profile_not_observed"
                            ),
                        })
                source_decomposition = prompt_analysis.get(
                    "source_decomposition", {}
                )
                if bool(
                    self.cfg.MONITOR.PROBE.PROMPT_ANALYSIS.SOURCE_DECOMPOSITION_ENABLE
                ):
                    source_path = artifact_path(
                        f"prompt_source/{checkpoint_id}/{split}_summary.json"
                    )
                    self.diagnostic_manager.record_probe_artifact(
                        source_path,
                        {
                            **source_decomposition,
                            "checkpoint": checkpoint_manifest,
                            "probe_id": manifest["probe_id"],
                            "probe_manifest_sha256": manifest[
                                "manifest_sha256"
                            ],
                            "selection_seed": selection_seed,
                            "split": split,
                        },
                    )
                    prompt_analysis_status["source_decomposition"] = {
                        "applicability": source_decomposition.get(
                            "applicability"
                        ),
                        "raw_observed": source_decomposition.get(
                            "raw_observed", False
                        ),
                        "contextualized_domain_observed": (
                            source_decomposition.get(
                                "contextualized_domain_observed", False
                            )
                        ),
                        "artifact_path": source_path,
                    }
                    all_rows.extend(self._probe_metric_rows(
                        source_decomposition.get("metrics", {}),
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="prompt_distribution",
                        entity_type="source_decomposition",
                        entity_id="instance_domain_sources",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest[
                            "manifest_sha256"
                        ],
                    ))
                label_guard = prompt_analysis.get(
                    "label_dependency_guard", {}
                )
                if bool(
                    self.cfg.MONITOR.PROBE.PROMPT_ANALYSIS.LABEL_DEPENDENCY_GUARD_ENABLE
                ):
                    label_guard_path = artifact_path(
                        f"prompt_label_dependency/{checkpoint_id}/"
                        f"{split}_summary.json"
                    )
                    self.diagnostic_manager.record_probe_artifact(
                        label_guard_path,
                        {
                            **label_guard,
                            "checkpoint": checkpoint_manifest,
                            "probe_id": manifest["probe_id"],
                            "probe_manifest_sha256": manifest[
                                "manifest_sha256"
                            ],
                            "selection_seed": selection_seed,
                            "split": split,
                        },
                    )
                    prompt_analysis_status["label_dependency_guard"] = {
                        "applicability": label_guard.get("applicability"),
                        "observed": label_guard.get("observed", False),
                        "valid": label_guard.get("valid"),
                        "artifact_path": label_guard_path,
                        "failure_reasons": label_guard.get(
                            "failure_reasons", []
                        ),
                    }
                    all_rows.extend(self._probe_metric_rows(
                        label_guard.get("metrics", {}),
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="prompt_distribution",
                        entity_type="runtime_guard",
                        entity_id="label_dependency",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest[
                            "manifest_sha256"
                        ],
                    ))
                paired_flip = prompt_analysis.get("paired_flip", {})
                if bool(
                    self.cfg.MONITOR.PROBE.PROMPT_ANALYSIS.FLIP_ENABLE
                ):
                    flip_path = artifact_path(
                        f"paired_flip/{checkpoint_id}/{split}_summary.json"
                    )
                    self.diagnostic_manager.record_probe_artifact(
                        flip_path,
                        {
                            **paired_flip,
                            "checkpoint": checkpoint_manifest,
                            "probe_id": manifest["probe_id"],
                            "probe_manifest_sha256": manifest[
                                "manifest_sha256"
                            ],
                            "selection_seed": selection_seed,
                            "split": split,
                        },
                    )
                    prompt_analysis_status["paired_flip"] = {
                        "applicability": paired_flip.get("applicability"),
                        "valid": paired_flip.get("valid", False),
                        "observed_layers": paired_flip.get(
                            "observed_layers", []
                        ),
                        "artifact_path": flip_path,
                    }
                    spatial_metrics = {
                        name: value
                        for name, value in paired_flip.get(
                            "metrics", {}
                        ).items()
                        if name.startswith(("prompt_patch_", "prompt_av_"))
                    }
                    assignment_metrics = {
                        name: value
                        for name, value in paired_flip.get(
                            "metrics", {}
                        ).items()
                        if name.startswith("assignment_")
                    }
                    prediction_metrics = {
                        name: value
                        for name, value in paired_flip.get(
                            "metrics", {}
                        ).items()
                        if name.startswith("prediction_")
                    }
                    for metrics, domain, entity_id in (
                        (
                            spatial_metrics,
                            "attention_flow_reference",
                            "paired_horizontal_flip_spatial",
                        ),
                        (
                            assignment_metrics,
                            "prompt_semantic_role_reference",
                            "paired_horizontal_flip_assignment",
                        ),
                        (
                            prediction_metrics,
                            "prediction_health",
                            "paired_horizontal_flip_prediction",
                        ),
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition="horizontal_flip_paired",
                            domain=domain,
                            entity_type="paired_transform",
                            entity_id=entity_id,
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest[
                                "manifest_sha256"
                            ],
                        ))
                    for layer_index, metrics in sorted(
                        paired_flip.get("by_layer", {}).items()
                    ):
                        for prefix, domain, entity_suffix in (
                            (
                                ("prompt_patch_", "prompt_av_"),
                                "attention_flow_reference",
                                "spatial",
                            ),
                            (
                                ("assignment_",),
                                "prompt_semantic_role_reference",
                                "assignment",
                            ),
                        ):
                            selected = {
                                name: value
                                for name, value in metrics.items()
                                if name.startswith(prefix)
                            }
                            all_rows.extend(self._probe_metric_rows(
                                selected,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition="horizontal_flip_paired",
                                domain=domain,
                                entity_type="layer_paired_transform",
                                entity_id=(
                                    f"layer_{layer_index}/{entity_suffix}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest[
                                    "manifest_sha256"
                                ],
                            ))
                combined_manifest["prompt_analysis"][split] = (
                    prompt_analysis_status
                )
            for condition, condition_metrics in bundle["conditions"].items():
                for domain in (
                    "classification", "representation_geometry",
                    "visual_semantic_alignment", "semantic_graph_reference",
                ):
                    entity_type = "graph" if domain == "semantic_graph_reference" else "split"
                    entity_id = "visual_consistency" if domain == "semantic_graph_reference" else "all"
                    all_rows.extend(self._probe_metric_rows(
                        condition_metrics.get(domain, {}),
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        condition=condition,
                        domain=domain,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
            semantic_intervention = bundle.get("semantic_intervention")
            if semantic_intervention is not None:
                semantic_results[split] = semantic_intervention
                combined_manifest["semantic_interventions"][split] = {
                    "format": semantic_intervention["format"],
                    "conditions": semantic_intervention["conditions"],
                    "permutation_seed": semantic_intervention["permutation_seed"],
                    "permutation": semantic_intervention["permutation"],
                    "inverse_permutation": semantic_intervention["inverse_permutation"],
                    "permutation_sha256": semantic_intervention["permutation_sha256"],
                    "thresholds": semantic_intervention["thresholds"],
                    "sync_logit_atol": float(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.SYNC_LOGIT_ATOL),
                    "sync_margin_atol": float(self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.SYNC_MARGIN_ATOL),
                }
                all_rows.extend(self._probe_metric_rows(
                    semantic_intervention["synchronized"],
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="synchronized_class_permutation",
                    domain="semantic_prototype_intervention",
                    entity_type="intervention",
                    entity_id="inverse_recovery",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
                all_rows.extend(self._probe_metric_rows(
                    semantic_intervention["mismatched"],
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="mismatched_semantic_permutation",
                    domain="semantic_prototype_intervention",
                    entity_type="intervention",
                    entity_id="semantic_mismatch",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
            affinity = bundle.get("affinity")
            if affinity is not None:
                combined_manifest["patch_semantic_transport"][split] = dict(
                    affinity["patch_semantic_transport_reference"]
                )
                equivalence = affinity["equivalence"]
                equivalence_pass = bool(equivalence.get("affinity_forward_equivalence_pass", 0.0))
                self.diagnostic_manager.record_probe_artifact(
                    artifact_path(f"probe_equivalence/{checkpoint_id}_{split}.json"),
                    {
                        "format": "affinity_forward_equivalence_v2",
                        "checkpoint_id": checkpoint_id,
                        "probe_id": manifest["probe_id"],
                        "split": split,
                        "metrics": equivalence,
                        "valid": equivalence_pass,
                    },
                )
                all_rows.extend(self._probe_metric_rows(
                    equivalence,
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="affinity_forward",
                    domain="affinity_forward_equivalence",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
                if equivalence_pass:
                    all_rows.extend(self._probe_metric_rows(
                        affinity["prompt_content_metrics"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="attention_flow_reference",
                        entity_type="prompt_content",
                        entity_id="all_layers",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest[
                            "manifest_sha256"
                        ],
                    ))
                    for layer_index, layer_metrics in sorted(
                        affinity["prompt_content_by_layer"].items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            layer_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="attention_flow_reference",
                            entity_type="layer_prompt_content",
                            entity_id=f"layer_{layer_index}",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest[
                                "manifest_sha256"
                            ],
                        ))
                    for layer_index, head_metrics in sorted(
                        affinity[
                            "prompt_content_by_layer_and_head"
                        ].items()
                    ):
                        for head_index, metrics in sorted(
                            head_metrics.items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                domain="attention_flow_reference",
                                entity_type="layer_head_prompt_content",
                                entity_id=(
                                    f"layer_{layer_index}/head_{head_index}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest[
                                    "manifest_sha256"
                                ],
                            ))
                    for layer_index, prompt_metrics in sorted(
                        affinity[
                            "prompt_content_by_layer_and_prompt"
                        ].items()
                    ):
                        for prompt_index, metrics in sorted(
                            prompt_metrics.items()
                        ):
                            numeric_metrics = {
                                name: value
                                for name, value in metrics.items()
                                if name != "prompt_type"
                            }
                            all_rows.extend(self._probe_metric_rows(
                                numeric_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                domain="prompt_semantic_role_reference",
                                entity_type="layer_prompt_slot_health",
                                entity_id=(
                                    f"layer_{layer_index}/prompt_{prompt_index}/"
                                    f"type_{metrics.get('prompt_type', 'visual')}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest[
                                    "manifest_sha256"
                                ],
                            ))
                    role_profiles = affinity.get("prompt_role_profiles", {})
                    if role_profiles:
                        distributor_active = bool(
                            self.cfg.MODEL.PROMPT.ENABLE
                            and self.cfg.MODEL.PROMPT.DISTRIBUTOR.ENABLE
                            and str(
                                self.cfg.MODEL.PROMPT.INIT_SOURCE
                            ).lower() == "distributor_mean"
                        )
                        instance_tokens = (
                            int(
                                self.cfg.MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS
                            )
                            if distributor_active
                            else 0
                        )
                        domain_tokens = (
                            int(
                                self.cfg.MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS
                            )
                            if distributor_active
                            else 0
                        )
                        prompt_types = [
                            (
                                "instance"
                                if prompt_index < instance_tokens
                                else "domain"
                                if prompt_index
                                < instance_tokens + domain_tokens
                                else "visual"
                            )
                            for prompt_index in range(bundle["prompt_length"])
                        ]
                        role_profile_path = artifact_path(
                            f"prompt_role_profiles/{checkpoint_id}/"
                            f"{split}.json"
                        )
                        self.diagnostic_manager.record_probe_artifact(
                            role_profile_path,
                            {
                                "format": "prompt_role_profile_v1",
                                "checkpoint": checkpoint_manifest,
                                "probe_id": manifest["probe_id"],
                                "probe_manifest_sha256": manifest[
                                    "manifest_sha256"
                                ],
                                "selection_seed": selection_seed,
                                "split": split,
                                "prompt_mode": (
                                    "distributor_instance_domain"
                                    if distributor_active
                                    else "static_visual_prompt"
                                ),
                                "prompt_types": prompt_types,
                                "layers": role_profiles,
                            },
                        )
                        combined_manifest["prompt_analysis"].setdefault(
                            split,
                            {
                                "requested": True,
                                "applicability": "applicable",
                            },
                        ).setdefault("role_profile", {}).update({
                            "requested": True,
                            "observed": True,
                            "valid": True,
                            "artifact_path": role_profile_path,
                            "layer_count": len(role_profiles),
                            "reason": None,
                        })
                    all_rows.extend(self._probe_metric_rows(
                        affinity["attention_flow_metrics"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="attention_flow_reference",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    for layer_index, layer_metrics in sorted(
                        affinity["attention_flow_by_layer"].items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            layer_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="attention_flow_reference",
                            entity_type="layer",
                            entity_id=f"layer_{layer_index}",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    for layer_index, layer_metrics in sorted(
                        affinity["prompt_layer_mechanism_by_layer"].items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            layer_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="prompt_layer_mechanism",
                            entity_type="layer",
                            entity_id=f"layer_{layer_index}",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    all_rows.extend(self._probe_metric_rows(
                        affinity["prompt_patch_bridge_metrics"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="prompt_patch_bridge_reference",
                        entity_type="bridge",
                        entity_id="prompt_patch",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    for layer_index, layer_metrics in sorted(
                        affinity["prompt_patch_bridge_by_layer"].items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            layer_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="prompt_patch_bridge_reference",
                            entity_type="layer",
                            entity_id=f"layer_{layer_index}",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    all_rows.extend(self._probe_metric_rows(
                        affinity["prompt_semantic_role_metrics"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="prompt_semantic_role_reference",
                        entity_type="profile",
                        entity_id="all_layers",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    for layer_index, layer_metrics in sorted(
                        affinity["prompt_semantic_role_by_layer"].items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            layer_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="prompt_semantic_role_reference",
                            entity_type="layer_profile",
                            entity_id=f"layer_{layer_index}",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    for layer_index, prompt_metrics in sorted(
                        affinity[
                            "prompt_semantic_role_by_layer_and_prompt"
                        ].items()
                    ):
                        for prompt_index, metrics in sorted(
                            prompt_metrics.items()
                        ):
                            all_rows.extend(self._prompt_role_metric_rows(
                                metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                layer_index=layer_index,
                                prompt_index=prompt_index,
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                    all_rows.extend(self._probe_metric_rows(
                        affinity["attribute_concept_grounding_metrics"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="attribute_concept_grounding_reference",
                        entity_type="profile",
                        entity_id="all_layers",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    for layer_index, layer_metrics in sorted(
                        affinity["attribute_concept_grounding_by_layer"].items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            layer_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="attribute_concept_grounding_reference",
                            entity_type="layer_profile",
                            entity_id=f"layer_{layer_index}",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    for layer_index, prompt_metrics in sorted(
                        affinity[
                            "attribute_concept_grounding_by_layer_and_prompt"
                        ].items()
                    ):
                        for prompt_index, metrics in sorted(
                            prompt_metrics.items()
                        ):
                            all_rows.extend(self._attribute_concept_metric_rows(
                                metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                layer_index=layer_index,
                                prompt_index=prompt_index,
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                    for layer_pair, pair_metrics in sorted(
                        affinity["cross_layer_concept_continuity_by_pair"].items()
                    ):
                        all_rows.extend(self._probe_metric_rows(
                            pair_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="attribute_concept_grounding_reference",
                            entity_type="layer_pair",
                            entity_id=layer_pair,
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    all_rows.extend(self._probe_metric_rows(
                        affinity["patch_semantic_transport_metrics"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="patch_semantic_transport_reference",
                        entity_type="transport",
                        entity_id="final_layer_bidirectional",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    for prompt_index, prompt_metrics in sorted(
                        affinity["patch_semantic_transport_by_prompt"].items()
                    ):
                        all_rows.extend(self._patch_semantic_transport_metric_rows(
                            prompt_metrics,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            prompt_index=prompt_index,
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    for metric_name, value in sorted(affinity["affinity_health_metrics"].items()):
                        relation, _, metric = metric_name.partition(".")
                        all_rows.extend(self._probe_metric_rows(
                            {metric or relation: value},
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="affinity_health",
                            entity_type="relation",
                            entity_id=relation if metric else "global",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    for layer_index, layer_metrics in sorted(
                        affinity["affinity_health_by_layer"].items()
                    ):
                        for metric_name, value in sorted(layer_metrics.items()):
                            relation, _, metric = metric_name.partition(".")
                            all_rows.extend(self._probe_metric_rows(
                                {metric: value},
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                domain="affinity_health",
                                entity_type="layer_relation",
                                entity_id=f"layer_{layer_index}/{relation}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                    token_metrics = affinity["token_metrics"]
                    for entity_id, values in token_metrics["representation_geometry"].items():
                        all_rows.extend(self._probe_metric_rows(
                            values,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="representation_geometry",
                            entity_type="representation",
                            entity_id=entity_id,
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                    prompt_response = token_metrics["prompt_parameter_health"]
                    normal_results[split]["prompt_response"] = prompt_response
                    all_rows.extend(self._probe_metric_rows(
                        prompt_response,
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="prompt_parameter_health",
                        entity_type="representation",
                        entity_id="contextualized_prompt",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    all_rows.extend(self._probe_metric_rows(
                        token_metrics["relation_stability"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="relation_stability",
                        entity_type="relationship",
                        entity_id="prompt_cls_patch",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    all_rows.extend(self._probe_metric_rows(
                        token_metrics["prompt_semantic_role_reference"],
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="prompt_semantic_role_reference",
                        entity_type="representation",
                        entity_id="final_contextualized_prompt",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                else:
                    logger.warning(
                        "[fixed-probe] affinity diagnostics skipped for split=%s because forward equivalence failed",
                        split,
                    )
            for intervention_name, diagnostics in sorted(
                bundle.get("intervention_diagnostics", {}).items()
            ):
                condition_status = {
                    "alignment_observed": bool(diagnostics["alignment_metrics"]),
                    "affinity_requested": bool(diagnostics.get("affinity") is not None),
                    "affinity_forward_equivalence_pass": None,
                    "paired_attention_valid": False,
                    "layer_count": 0,
                }
                intervention_diagnostic_status.setdefault(
                    intervention_name,
                    {},
                )[split] = condition_status
                all_rows.extend(self._probe_metric_rows(
                    diagnostics["alignment_metrics"],
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition=intervention_name,
                    domain="visual_semantic_alignment",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
                intervention_affinity = diagnostics.get("affinity")
                if intervention_affinity is not None:
                    intervention_equivalence = intervention_affinity["equivalence"]
                    intervention_equivalence_pass = bool(
                        intervention_equivalence.get(
                            "affinity_forward_equivalence_pass", 0.0
                        )
                    )
                    condition_status.update({
                        "affinity_forward_equivalence_pass": intervention_equivalence_pass,
                        "paired_attention_valid": bool(
                            intervention_affinity["paired_attention_valid"]
                        ),
                        "layer_count": int(
                            len(intervention_affinity["attention_flow_by_layer"])
                        ),
                    })
                    self.diagnostic_manager.record_probe_artifact(
                        artifact_path(
                            f"probe_equivalence/{checkpoint_id}_{split}_{intervention_name}.json"
                        ),
                        {
                            "format": "affinity_forward_equivalence_v2",
                            "checkpoint_id": checkpoint_id,
                            "probe_id": manifest["probe_id"],
                            "split": split,
                            "condition": intervention_name,
                            "metrics": intervention_equivalence,
                            "valid": intervention_equivalence_pass,
                        },
                    )
                    all_rows.extend(self._probe_metric_rows(
                        intervention_equivalence,
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        condition=f"{intervention_name}_affinity_forward",
                        domain="affinity_forward_equivalence",
                        selection_seed=selection_seed,
                        probe_manifest_sha256=manifest["manifest_sha256"],
                    ))
                    if intervention_equivalence_pass:
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity["attention_flow_metrics"],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="attention_flow_reference",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "attention_flow_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="attention_flow_reference",
                                entity_type="layer",
                                entity_id=f"layer_{layer_index}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "prompt_layer_mechanism_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="prompt_layer_mechanism",
                                entity_type="layer",
                                entity_id=f"layer_{layer_index}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "prompt_patch_bridge_metrics"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="prompt_patch_bridge_reference",
                            entity_type="bridge",
                            entity_id="prompt_patch",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "prompt_patch_bridge_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="prompt_patch_bridge_reference",
                                entity_type="layer",
                                entity_id=f"layer_{layer_index}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "prompt_semantic_role_metrics"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="prompt_semantic_role_reference",
                            entity_type="profile",
                            entity_id="all_layers",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "prompt_semantic_role_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="prompt_semantic_role_reference",
                                entity_type="layer_profile",
                                entity_id=f"layer_{layer_index}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        for layer_index, prompt_metrics in sorted(
                            intervention_affinity[
                                "prompt_semantic_role_by_layer_and_prompt"
                            ].items()
                        ):
                            for prompt_index, metrics in sorted(
                                prompt_metrics.items()
                            ):
                                all_rows.extend(self._prompt_role_metric_rows(
                                    metrics,
                                    checkpoint_id=checkpoint_id,
                                    probe_id=manifest["probe_id"],
                                    split=split,
                                    condition=intervention_name,
                                    layer_index=layer_index,
                                    prompt_index=prompt_index,
                                    selection_seed=selection_seed,
                                    probe_manifest_sha256=manifest["manifest_sha256"],
                                ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "attribute_concept_grounding_metrics"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="attribute_concept_grounding_reference",
                            entity_type="profile",
                            entity_id="all_layers",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "attribute_concept_grounding_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="attribute_concept_grounding_reference",
                                entity_type="layer_profile",
                                entity_id=f"layer_{layer_index}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        for layer_index, prompt_metrics in sorted(
                            intervention_affinity[
                                "attribute_concept_grounding_by_layer_and_prompt"
                            ].items()
                        ):
                            for prompt_index, metrics in sorted(
                                prompt_metrics.items()
                            ):
                                all_rows.extend(self._attribute_concept_metric_rows(
                                    metrics,
                                    checkpoint_id=checkpoint_id,
                                    probe_id=manifest["probe_id"],
                                    split=split,
                                    condition=intervention_name,
                                    layer_index=layer_index,
                                    prompt_index=prompt_index,
                                    selection_seed=selection_seed,
                                    probe_manifest_sha256=manifest["manifest_sha256"],
                                ))
                        for layer_pair, pair_metrics in sorted(
                            intervention_affinity[
                                "cross_layer_concept_continuity_by_pair"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                pair_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="attribute_concept_grounding_reference",
                                entity_type="layer_pair",
                                entity_id=layer_pair,
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "patch_semantic_transport_metrics"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="patch_semantic_transport_reference",
                            entity_type="transport",
                            entity_id="final_layer_bidirectional",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for prompt_index, prompt_metrics in sorted(
                            intervention_affinity[
                                "patch_semantic_transport_by_prompt"
                            ].items()
                        ):
                            all_rows.extend(self._patch_semantic_transport_metric_rows(
                                prompt_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                prompt_index=prompt_index,
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                    if intervention_affinity["paired_attention_valid"]:
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity["attention_delta"],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="module_effect",
                            entity_type="intervention",
                            entity_id=f"{intervention_name}_attention",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "attention_delta_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="module_effect",
                                entity_type="layer",
                                entity_id=f"layer_{layer_index}",
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "prompt_layer_mechanism_delta_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="module_effect",
                                entity_type="layer",
                                entity_id=(
                                    f"{intervention_name}_prompt_state/"
                                    f"layer_{layer_index}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "patch_semantic_transport_delta"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="module_effect",
                            entity_type="intervention",
                            entity_id=f"{intervention_name}_transport",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "prompt_patch_bridge_delta"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="module_effect",
                            entity_type="intervention",
                            entity_id=f"{intervention_name}_bridge",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "prompt_patch_bridge_delta_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="module_effect",
                                entity_type="layer",
                                entity_id=(
                                    f"{intervention_name}_bridge/layer_{layer_index}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "prompt_semantic_role_delta"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="module_effect",
                            entity_type="intervention",
                            entity_id=f"{intervention_name}_prompt_semantic_role",
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "prompt_semantic_role_delta_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="module_effect",
                                entity_type="layer",
                                entity_id=(
                                    f"{intervention_name}_prompt_semantic_role/"
                                    f"layer_{layer_index}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
                        all_rows.extend(self._probe_metric_rows(
                            intervention_affinity[
                                "attribute_concept_grounding_delta"
                            ],
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            condition=intervention_name,
                            domain="module_effect",
                            entity_type="intervention",
                            entity_id=(
                                f"{intervention_name}_attribute_concept"
                            ),
                            selection_seed=selection_seed,
                            probe_manifest_sha256=manifest["manifest_sha256"],
                        ))
                        for layer_index, layer_metrics in sorted(
                            intervention_affinity[
                                "attribute_concept_grounding_delta_by_layer"
                            ].items()
                        ):
                            all_rows.extend(self._probe_metric_rows(
                                layer_metrics,
                                checkpoint_id=checkpoint_id,
                                probe_id=manifest["probe_id"],
                                split=split,
                                condition=intervention_name,
                                domain="module_effect",
                                entity_type="layer",
                                entity_id=(
                                    f"{intervention_name}_attribute_concept/"
                                    f"layer_{layer_index}"
                                ),
                                selection_seed=selection_seed,
                                probe_manifest_sha256=manifest["manifest_sha256"],
                            ))
            concept_comparison = bundle.get(
                "attribute_concept_intervention_comparison"
            )
            if concept_comparison:
                all_rows.extend(self._probe_metric_rows(
                    concept_comparison,
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="attribute_concept_prompt_patch_blocked",
                    domain="module_effect",
                    entity_type="intervention_control",
                    entity_id="attribute_concept_targeted_vs_random",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
            transport_comparison = bundle.get(
                "transport_intervention_comparison"
            )
            if transport_comparison:
                all_rows.extend(self._probe_metric_rows(
                    transport_comparison,
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="transport_targeted_prompt_patch_blocked",
                    domain="module_effect",
                    entity_type="intervention_control",
                    entity_id="transport_targeted_vs_random",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
            for intervention_name, effect in bundle["module_effects"].items():
                self._record_probe_module_effect(
                    checkpoint_id=checkpoint_id,
                    checkpoint_manifest=checkpoint_manifest,
                    split=split,
                    manifest=manifest,
                    intervention_name=intervention_name,
                    effect=effect,
                    artifact_prefix=artifact_prefix,
                )
                all_rows.extend(self._probe_metric_rows(
                    effect["summary"],
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition=intervention_name,
                    domain="module_effect",
                    entity_type="intervention",
                    entity_id=intervention_name,
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest["manifest_sha256"],
                ))
            component_synergy = bundle.get("prompt_component_synergy")
            if component_synergy:
                synergy_path = artifact_path(
                    f"module_effect/{checkpoint_id}/{split}/"
                    "instance_domain_synergy.json"
                )
                self.diagnostic_manager.record_module_effect_artifact(
                    synergy_path,
                    {
                        "format": "prompt_component_synergy_v1",
                        "checkpoint": checkpoint_manifest,
                        "probe_id": manifest["probe_id"],
                        "probe_manifest_sha256": manifest[
                            "manifest_sha256"
                        ],
                        "selection_seed": selection_seed,
                        "split": split,
                        **component_synergy,
                    },
                )
                all_rows.extend(self._probe_metric_rows(
                    component_synergy.get("metrics", {}),
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    condition="instance_domain_component_zero",
                    domain="module_effect",
                    entity_type="component_interaction",
                    entity_id="instance_domain_synergy",
                    selection_seed=selection_seed,
                    probe_manifest_sha256=manifest[
                        "manifest_sha256"
                    ],
                ))

        if "probe_test_seen" in normal_results and "probe_test_unseen" in normal_results:
            seen_result = normal_results["probe_test_seen"]
            unseen_result = normal_results["probe_test_unseen"]
            cross_split_manifest_sha256 = hashlib.sha256(
                "|".join((
                    combined_manifest["probes"]["probe_test_seen"]["manifest_sha256"],
                    combined_manifest["probes"]["probe_test_unseen"]["manifest_sha256"],
                )).encode("utf-8")
            ).hexdigest()
            seen_alignment = seen_result["visual_semantic_alignment"]
            unseen_alignment = unseen_result["visual_semantic_alignment"]
            transfer = {
                f"{key}_transfer_gap": float(seen_alignment[key] - unseen_alignment[key])
                for key in sorted(set(seen_alignment).intersection(unseen_alignment))
            }
            combined = StreamingFixedProbeAccumulator(
                seen_result["accumulator"].candidate, track_geometry=False
            )
            combined.merge_from(seen_result["accumulator"])
            combined.merge_from(unseen_result["accumulator"])
            transfer.update({
                f"combined_{name}": value
                for name, value in combined.finalize()["semantic_graph_reference"].items()
            })
            seen_prompt = seen_result.get("prompt_response", {})
            unseen_prompt = unseen_result.get("prompt_response", {})
            if "contextualized_prompt_norm" in seen_prompt and "contextualized_prompt_norm" in unseen_prompt:
                transfer["prompt_seen_unseen_response_gap"] = float(
                    seen_prompt["contextualized_prompt_norm"] - unseen_prompt["contextualized_prompt_norm"]
                )
            all_rows.extend(self._probe_metric_rows(
                transfer,
                checkpoint_id=checkpoint_id,
                probe_id=f"paired_test_seen_unseen-seed{selection_seed}",
                split="test_seen_vs_test_unseen",
                domain="visual_semantic_alignment",
                entity_type="cross_split",
                entity_id="seen_minus_unseen",
                selection_seed=selection_seed,
                probe_manifest_sha256=cross_split_manifest_sha256,
            ))
        if "probe_train_seen" in normal_results and "probe_test_seen" in normal_results:
            train_metric = normal_results["probe_train_seen"]["classification"]
            test_metric = normal_results["probe_test_seen"]["classification"]
            cross_split_manifest_sha256 = hashlib.sha256(
                "|".join((
                    combined_manifest["probes"]["probe_train_seen"]["manifest_sha256"],
                    combined_manifest["probes"]["probe_test_seen"]["manifest_sha256"],
                )).encode("utf-8")
            ).hexdigest()
            all_rows.extend(self._probe_metric_rows(
                {"train_joint_to_test_seen_gap": float(train_metric["per_class"] - test_metric["per_class"])},
                checkpoint_id=checkpoint_id,
                probe_id=f"paired_train_test_seen-seed{selection_seed}",
                split="train_seen_vs_test_seen",
                domain="probe_context",
                entity_type="cross_split",
                entity_id="train_minus_test_seen",
                selection_seed=selection_seed,
                probe_manifest_sha256=cross_split_manifest_sha256,
            ))

        probe_manifest_path = artifact_path("probe_manifest.json")
        comparability_path = artifact_path("comparability.json")
        self.diagnostic_manager.record_probe_artifact(probe_manifest_path, combined_manifest)
        comparability = build_comparability_identity(
            self.cfg,
            run_id=self.monitor_manager.run_id,
            session_id=self.monitor_manager.session_id,
            checkpoint_manifest=checkpoint_manifest,
            probe_manifest=combined_manifest,
        )
        self.diagnostic_manager.record_probe_artifact(comparability_path, comparability)
        if bayesian_object_selection_reports:
            self.diagnostic_manager.record_probe_artifact(
                artifact_path(
                    "bayesian_object_selection/object_selection_report.json"
                ),
                {
                    "format": "bayesian_object_selection_collection_v2",
                    "checkpoint": checkpoint_manifest,
                    "selection_seed": selection_seed,
                    "probe_manifest_path": probe_manifest_path,
                    "recommended_candidate": None,
                    "recommended_stochastic_root": None,
                    "recommended_transfer_space": None,
                    "predictive_validation_space": "logit_effect",
                    "selection_status": "insufficient_evidence",
                    "reason": (
                        "controlled perturbation evidence is available, but held-out "
                        "and cross-training-seed gates are not completed"
                    ),
                    "reports_by_split": bayesian_object_selection_reports,
                    "automatic_composite_score_used": False,
                    "test_unseen_used_for_selection": False,
                    "posterior_metrics_deferred": True,
                },
            )
        self.diagnostic_manager.record_probe_artifact(
            artifact_path("fixed_probe_class_aggregates.json"),
            {
                "format": "baseline_fixed_probe_class_aggregates_v1",
                "analysis_role": "probe_context_only",
                "formal_task_result_source": "complete_test_epoch_metrics",
                "checkpoint": checkpoint_manifest,
                "selection_seed": selection_seed,
                "probe_manifest_path": probe_manifest_path,
                "storage_mode": "aggregate_only",
                "splits": class_aggregates,
            },
        )
        if semantic_results:
            test_unseen_semantic = semantic_results.get("probe_test_unseen")
            semantic_overall_pass = bool(test_unseen_semantic and test_unseen_semantic["overall_pass"])
            self.diagnostic_manager.record_probe_artifact(
                artifact_path("semantic_prototype_intervention_summary.json"),
                {
                    "format": "semantic_prototype_intervention_collection_v1",
                    "checkpoint": checkpoint_manifest,
                    "selection_seed": selection_seed,
                    "probe_manifest_path": probe_manifest_path,
                    "comparability_path": comparability_path,
                    "storage_contract": combined_manifest["storage_contract"],
                    "splits": semantic_results,
                    "required_split": "probe_test_unseen",
                    "overall_pass": semantic_overall_pass,
                    "failure_reasons": (
                        test_unseen_semantic.get("failure_reasons", [])
                        if test_unseen_semantic else ["probe_test_unseen was not available"]
                    ),
                },
            )
        else:
            semantic_overall_pass = None
            test_unseen_semantic = None
        if bool(self.cfg.MONITOR.MODULE_EFFECT.ENABLE):
            prompt_length = (
                int(self.cfg.MODEL.PROMPT.NUM_TOKENS)
                if bool(self.cfg.MODEL.PROMPT.ENABLE)
                else 0
            )
            manifest_attribute_reference = {
                "available": False,
                "reason": "class_attribute_reference_unavailable",
            }
            for dataset in split_sources.values():
                dataset_attributes = getattr(dataset, "class_attributes", None)
                if dataset_attributes is None:
                    continue
                manifest_attribute_reference = self._attribute_concept_reference(
                    model_ref, dataset_attributes
                )
                if manifest_attribute_reference.get("available", False):
                    break
            module_effect_specs = self._fixed_probe_profile_specs(
                self._module_effect_intervention_specs(
                    prompt_length,
                    attribute_concept_available=bool(
                        manifest_attribute_reference.get("available", False)
                    ),
                    attribute_concept_reason=str(
                        manifest_attribute_reference.get(
                            "reason", "attribute_concept_reference_unavailable"
                        )
                    ),
                ),
                execution_profile,
            )
            requested_conditions = [
                spec["name"] for spec in module_effect_specs
            ]
            configured_conditions = [
                spec["name"]
                for spec in module_effect_specs
                if spec["applicable"]
            ]
            not_applicable_conditions = {
                spec["name"]: spec["not_applicable_reason"]
                for spec in module_effect_specs
                if not spec["applicable"]
            }
            condition_semantics = {
                spec["name"]: spec["intervention_semantics"]
                for spec in module_effect_specs
            }
            module_effect_manifest_path = artifact_path("module_effect_manifest.json")
            self.diagnostic_manager.record_module_effect_artifact(
                module_effect_manifest_path,
                {
                    "format": "baseline_module_effect_v9",
                    "checkpoint": checkpoint_manifest,
                    "selection_seed": selection_seed,
                    "execution_profile": execution_profile,
                    "probe_manifest_path": probe_manifest_path,
                    "probe_manifest_sha256_by_split": {
                        split: manifest["manifest_sha256"] for split, manifest in combined_manifest["probes"].items()
                    },
                    "conditions": ["normal"] + configured_conditions,
                    "requested_conditions": requested_conditions,
                    "not_applicable_conditions": not_applicable_conditions,
                    "applicability": "applicable" if configured_conditions else "not_applicable",
                    "intervention_semantics": condition_semantics,
                    "prompt_zero_semantics": (
                        condition_semantics["prompt_zeroed"]
                        if "prompt_zeroed" in configured_conditions else None
                    ),
                    "intervention_diagnostics_by_condition": intervention_diagnostic_status,
                    "prompt_zero_diagnostics_by_split": intervention_diagnostic_status.get(
                        "prompt_zeroed", {}
                    ),
                    "storage_mode": "aggregate_only",
                    "model_eval": True,
                    "torch_no_grad": True,
                },
            )
        all_rows.flush()
        self.diagnostic_manager.record_probe_artifact(
            artifact_path("probe_runtime_summary.json"),
            {
                "status": "completed",
                "selection_seed": selection_seed,
                "execution_profile": execution_profile,
                "normal_task_output_role": "probe_context_only",
                "checkpoint": checkpoint_manifest,
                "probe_count": int(len(combined_manifest["probes"])),
                "required_splits": list(combined_manifest["required_splits"]),
                "metric_row_count": int(len(all_rows)),
                "storage_mode": "aggregate_only",
                "probe_loader": probe_loader_settings,
                "probe_runtime_timing": {
                    "definition": {
                        "probe_data_time_sec": (
                            "DataLoader iterator creation and next-batch wait time"
                        ),
                        "probe_compute_time_sec": (
                            "remaining wall time including device transfer, model "
                            "execution, intervention, and aggregation; not pure GPU time"
                        ),
                        "probe_data_time_ratio": (
                            "probe_data_time_sec divided by probe_total_time_sec"
                        ),
                    },
                    "by_split": probe_timing_by_split,
                    "aggregate": self._aggregate_probe_stage_timings(
                        probe_timing_by_split
                    ),
                },
                "intervention_diagnostics_by_condition": intervention_diagnostic_status,
                "prompt_zero_diagnostics_by_split": intervention_diagnostic_status.get(
                    "prompt_zeroed", {}
                ),
                "target_relevance_enabled": bool(
                    self.cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE
                ),
                "target_relevance_by_split": target_relevance_status,
                "explanation_validity_enabled": bool(
                    self.cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE
                    and execution_profile == "final_full"
                ),
                "explanation_validity_by_split": combined_manifest[
                    "explanation_validity"
                ],
                "prompt_analysis_enabled": bool(
                    self.cfg.MONITOR.PROBE.PROMPT_ANALYSIS.ENABLE
                    and execution_profile == "final_full"
                ),
                "prompt_analysis_by_split": combined_manifest[
                    "prompt_analysis"
                ],
                "bayesian_object_selection_enabled": bool(
                    self.cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION.ENABLE
                    and execution_profile == "final_full"
                ),
                "bayesian_object_selection_by_split": (
                    bayesian_object_selection_status
                ),
                "semantic_intervention_enabled": bool(
                    self.cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE
                    and execution_profile == "final_full"
                ),
                "semantic_intervention_pass": semantic_overall_pass,
                "semantic_intervention_failure_reasons": (
                    test_unseen_semantic.get("failure_reasons", [])
                    if test_unseen_semantic else []
                ),
            },
        )
        normal_results.clear()
        if was_training:
            model_ref.train()
        return {
            "selection_seed": selection_seed,
            "artifact_prefix": artifact_prefix,
            "execution_profile": execution_profile,
            "valid": True,
            "probe_manifest_path": probe_manifest_path,
            "comparability_path": comparability_path,
            "probe_manifest_sha256_by_split": {
                split: manifest["manifest_sha256"]
                for split, manifest in combined_manifest["probes"].items()
            },
            "metric_row_count": int(len(all_rows)),
            "probe_loader": probe_loader_settings,
            "probe_runtime_timing": {
                "by_split": probe_timing_by_split,
                "aggregate": self._aggregate_probe_stage_timings(
                    probe_timing_by_split
                ),
            },
        }

    def _train_classifier_dev(self, train_loader, val_loader, test_seen_loader, test_unseen_loader):
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        self._log_train_loader_summary("dev", train_loader, total_data, log_interval)

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
                self._finish_progress_epoch(epoch, total_epoch)
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
                self._update_gzsl_record_metrics(epoch, test_seen_loader, test_unseen_loader, seen_metrics, unseen_metrics)

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

            self._finish_progress_epoch(epoch, total_epoch)
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

    def _train_classifier_final(
        self,
        train_loader,
        test_seen_loader,
        test_unseen_loader,
        *,
        train_eval_loader=None,
    ):
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        self._log_train_loader_summary("final", train_loader, total_data, log_interval)

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        self.cls_weights = train_loader.dataset.get_class_weights(self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        if total_epoch == 0:
            self.model.eval()
            self.evaluator.update_iteration(0)
            seen_metrics = (
                self.eval_classifier(test_seen_loader, "test_seen")
                if test_seen_loader is not None and str(self.evaluator.task_type).lower() == "gzsl"
                else None
            )
            unseen_metrics = (
                self.eval_classifier(test_unseen_loader, "test_unseen")
                if test_unseen_loader is not None
                else None
            )
            if str(self.evaluator.task_type).lower() == "gzsl":
                self._update_gzsl_record_metrics(-1, test_seen_loader, test_unseen_loader, seen_metrics, unseen_metrics)

        for epoch in range(total_epoch):
            self._run_train_epoch(epoch, total_epoch, total_data, train_loader, log_interval, losses, batch_time, data_time)

            self.model.eval()
            self.evaluator.update_iteration(epoch)

            logger.info(
                "[final-protocol] epoch=%d no dev validation; early-stop disabled",
                epoch + 1,
            )

            train_eval_every = max(1, int(self.cfg.MONITOR.TRAIN_EVAL.EVERY_N))
            if train_eval_loader is not None and (
                (epoch + 1) % train_eval_every == 0 or epoch + 1 == total_epoch
            ):
                self.eval_classifier(train_eval_loader, "train_eval_seen")

            if test_seen_loader is not None and str(self.evaluator.task_type).lower() == "gzsl":
                seen_metrics = self.eval_classifier(test_seen_loader, "test_seen")
            else:
                seen_metrics = None

            if test_unseen_loader is not None:
                unseen_metrics = self.eval_classifier(test_unseen_loader, "test_unseen")
            else:
                unseen_metrics = None

            if str(self.evaluator.task_type).lower() == "gzsl":
                self._update_gzsl_record_metrics(epoch, test_seen_loader, test_unseen_loader, seen_metrics, unseen_metrics)
            self._run_milestone_probe_if_due(
                epoch=epoch + 1,
                total_epoch=total_epoch,
                train_loader=train_loader,
                test_seen_loader=test_seen_loader,
                test_unseen_loader=test_unseen_loader,
            )
            self._finish_progress_epoch(epoch, total_epoch)

        if bool(self.cfg.MONITOR.MODULE_EFFECT.ENABLE) and not bool(
            self.cfg.SOLVER.SAVE_TRAINABLE_FINAL_CHECKPOINT
        ):
            raise ValueError(
                "MONITOR.MODULE_EFFECT.ENABLE requires SOLVER.SAVE_TRAINABLE_FINAL_CHECKPOINT=True"
            )
        if du.get_rank() == 0:
            if bool(self.cfg.SOLVER.SAVE_TRAINABLE_FINAL_CHECKPOINT):
                self._final_trainable_checkpoint_path = self._save_trainable_final_checkpoint(total_epoch)
            self._run_fixed_probes(
                train_loader,
                test_seen_loader,
                test_unseen_loader,
                checkpoint_epoch=total_epoch,
            )
            if self._milestone_probe_epochs:
                self.diagnostic_manager.record_probe_artifact(
                    "milestone_probe_manifest.json",
                    {
                        "format": "predeclared_milestone_probe_collection_v1",
                        "analysis_role": "diagnostic_only",
                        "checkpoint_selection_allowed": False,
                        "configured_nonfinal_epochs": self._milestone_probe_epochs,
                        "final_epoch": int(total_epoch),
                        "records": list(self._milestone_probe_records),
                        "final_probe_artifact_prefix": "",
                        "selection_rule": "warmup_end_and_fixed_training_fractions",
                    },
                )
        if du.get_world_size() > 1:
            torch.distributed.barrier()

    def _resolve_milestone_probe_epochs(self, total_epoch):
        cfg = self.cfg.MONITOR.MILESTONE_PROBE
        if not bool(cfg.ENABLE):
            return []
        if str(self.cfg.DATA.XLSA.PROTOCOL_MODE).lower() != "final_gzsl":
            raise ValueError("MONITOR.MILESTONE_PROBE is only supported for final_gzsl")
        if int(self.cfg.NUM_GPUS) != 1 or int(self.cfg.NUM_SHARDS) != 1:
            raise ValueError(
                "MONITOR.MILESTONE_PROBE currently requires single-GPU, single-shard execution"
            )
        if bool(cfg.RUN_FIXED_PROBE) and not bool(self.cfg.MONITOR.PROBE.ENABLE):
            raise ValueError("MILESTONE_PROBE.RUN_FIXED_PROBE requires MONITOR.PROBE.ENABLE")
        if bool(cfg.RUN_FIXED_PROBE) and not bool(cfg.SAVE_CHECKPOINTS):
            raise ValueError("milestone fixed probes require SAVE_CHECKPOINTS for checkpoint identity")
        if not bool(cfg.SAVE_CHECKPOINTS) and not bool(cfg.RUN_FIXED_PROBE):
            raise ValueError("milestone probe requires SAVE_CHECKPOINTS or RUN_FIXED_PROBE")
        epochs = []
        if bool(cfg.INCLUDE_WARMUP_END):
            warmup = int(self.cfg.SOLVER.WARMUP_EPOCH)
            if 0 < warmup < int(total_epoch):
                epochs.append(warmup)
        for raw_fraction in list(cfg.FRACTIONS):
            fraction = float(raw_fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError("MONITOR.MILESTONE_PROBE.FRACTIONS must be inside (0, 1)")
            epoch = int(round(float(total_epoch) * fraction))
            if 0 < epoch < int(total_epoch):
                epochs.append(epoch)
        return sorted(set(epochs))

    def _trainable_model_state(self):
        model_ref = self._model_ref(self.model)
        trainable_names = [name for name, param in model_ref.named_parameters() if param.requires_grad]
        state = model_ref.state_dict()
        trainable_state = {
            name: state[name].detach().cpu()
            for name in trainable_names
            if name in state
        }
        missing = sorted(set(trainable_names).difference(trainable_state))
        if missing:
            raise RuntimeError("Trainable checkpoint is missing model state keys: {}".format(missing[:20]))
        return trainable_names, trainable_state

    def _save_trainable_final_checkpoint(self, total_epoch):
        trainable_names, trainable_state = self._trainable_model_state()

        checkpoint_name = str(self.cfg.SOLVER.TRAINABLE_FINAL_CHECKPOINT_NAME).strip()
        if not checkpoint_name or os.path.basename(checkpoint_name) != checkpoint_name:
            raise ValueError("SOLVER.TRAINABLE_FINAL_CHECKPOINT_NAME must be a file name without directories.")
        checkpoint_path = os.path.join(str(self.cfg.OUTPUT_DIR), checkpoint_name)
        os.makedirs(str(self.cfg.OUTPUT_DIR), exist_ok=True)
        torch.save(
            {
                "format": "vpt_trainable_v1",
                "model_state": trainable_state,
                "trainable_parameter_names": trainable_names,
                "seed": int(self.cfg.SEED) if self.cfg.SEED is not None else None,
                "cell_id": str(self.cfg.SOLVER.STAGE2_CHECKPOINT_CELL_ID),
                "protocol_mode": str(self.cfg.DATA.XLSA.PROTOCOL_MODE),
                "total_epoch": int(total_epoch),
                "config": str(self.cfg),
            },
            checkpoint_path,
        )
        logger.info(
            "Saved trainable-only final checkpoint: %s tensors=%d parameters=%d",
            checkpoint_path,
            len(trainable_state),
            sum(int(t.numel()) for t in trainable_state.values()),
        )
        return checkpoint_path

    def _save_trainable_milestone_checkpoint(self, checkpoint_epoch, total_epoch):
        trainable_names, trainable_state = self._trainable_model_state()
        checkpoint_dir = os.path.join(str(self.cfg.OUTPUT_DIR), "milestone_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_trainable_epoch_{int(checkpoint_epoch):04d}.pth"
        )
        torch.save(
            {
                "format": "vpt_trainable_milestone_v1",
                "model_state": trainable_state,
                "trainable_parameter_names": trainable_names,
                "seed": int(self.cfg.SEED) if self.cfg.SEED is not None else None,
                "cell_id": str(self.cfg.SOLVER.STAGE2_CHECKPOINT_CELL_ID),
                "protocol_mode": str(self.cfg.DATA.XLSA.PROTOCOL_MODE),
                "checkpoint_epoch": int(checkpoint_epoch),
                "planned_total_epoch": int(total_epoch),
                "checkpoint_selection_rule": "predeclared_training_milestone",
                "config": str(self.cfg),
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _run_milestone_probe_if_due(
        self,
        *,
        epoch,
        total_epoch,
        train_loader,
        test_seen_loader,
        test_unseen_loader,
    ):
        if int(epoch) not in self._milestone_probe_epochs:
            return
        if du.get_rank() == 0:
            checkpoint_path = (
                self._save_trainable_milestone_checkpoint(epoch, total_epoch)
                if bool(self.cfg.MONITOR.MILESTONE_PROBE.SAVE_CHECKPOINTS)
                else None
            )
            checkpoint_identity = {
                "checkpoint_id": f"milestone_epoch_{int(epoch):04d}",
                "checkpoint_epoch": int(epoch),
                "checkpoint_global_step": int(self._trace_global_step),
                "checkpoint_selection_rule": "predeclared_training_milestone",
                "source_run_id": self.monitor_manager.run_id,
                "source_session_id": self.monitor_manager.session_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": (
                    checkpoint_sha256(checkpoint_path) if checkpoint_path else None
                ),
            }
            artifact_prefix = f"milestone_probes/epoch_{int(epoch):04d}"
            self._fixed_probe_checkpoint_source = checkpoint_identity
            try:
                if bool(self.cfg.MONITOR.MILESTONE_PROBE.RUN_FIXED_PROBE):
                    self._run_fixed_probes(
                        train_loader,
                        test_seen_loader,
                        test_unseen_loader,
                        checkpoint_epoch=int(epoch),
                        artifact_prefix=artifact_prefix,
                        execution_profile=str(
                            self.cfg.MONITOR.MILESTONE_PROBE.PROFILE
                        ),
                    )
            finally:
                self._fixed_probe_checkpoint_source = None
            self._milestone_probe_records.append({
                **checkpoint_identity,
                "artifact_prefix": artifact_prefix,
                "fixed_probe_executed": bool(
                    self.cfg.MONITOR.MILESTONE_PROBE.RUN_FIXED_PROBE
                ),
                "execution_profile": str(
                    self.cfg.MONITOR.MILESTONE_PROBE.PROFILE
                ),
            })
        if du.get_world_size() > 1:
            torch.distributed.barrier()

    def train_classifier(
        self,
        train_loader,
        val_loader,
        test_seen_loader,
        test_unseen_loader,
        *,
        train_eval_loader=None,
    ):
        completed = False
        self._write_progress_state(
            force=True,
            status="running",
            phase="initializing",
            epoch=0,
            completed_epochs=0,
            total_epochs=int(self.cfg.SOLVER.TOTAL_EPOCH),
        )
        try:
            self.diagnostic_manager.record_static_semantic_graph(train_loader.dataset)
            if val_loader is not None:
                result = self._train_classifier_dev(
                    train_loader,
                    val_loader,
                    test_seen_loader,
                    test_unseen_loader,
                )
            else:
                result = self._train_classifier_final(
                    train_loader,
                    test_seen_loader,
                    test_unseen_loader,
                    train_eval_loader=train_eval_loader,
                )
            completed = True
            return result
        finally:
            self._finalize_progress_state("completed" if completed else "interrupted")
            self.diagnostic_manager.finalize(
                status="completed" if completed else "interrupted"
            )
            self.monitor_manager.finalize(
                status="completed" if completed else "interrupted"
            )

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
        eval_epoch = int(self._trace_epoch + 1) if self._trace_epoch >= 0 else 0
        self._write_progress_state(
            force=True,
            status="running",
            phase=str(prefix),
            epoch=eval_epoch,
            completed_epochs=max(0, eval_epoch - 1),
            total_epochs=int(self.cfg.SOLVER.TOTAL_EPOCH),
            batch=0,
            total_batches=int(total),
        )
        self.monitor_manager.set_context(
            stage=f"eval_{prefix}",
            epoch=eval_epoch,
            global_step=int(self._trace_global_step),
            graph_prob_prior_forward=int(self._graph_prob_prior_forward),
        )
        eval_class_ids, eval_map = self._dataset_space_meta(dataset, use_eval_space=True)
        eval_map = eval_map.to(dtype=torch.long)

        # initialize features and target
        total_logits = []
        total_targets = []
        total_sample_ids = []
        model_ref = self._model_ref(self.model)
        model_ref.clear_runtime_state()
        if self._vis_split_enabled(prefix):
            self._vis_init_epoch(prefix)

        # ========== 遍历整个验证/测试集==========
        for idx, input_data in enumerate(data_loader):
            self._trace_stage = f"eval_{prefix}"
            self._trace_iter = int(idx)
            end = time.time()
            X, targets, attributes = self.get_input(input_data)
            batch_sample_ids = input_data.get("sample_id") if isinstance(input_data, dict) else None
            if batch_sample_ids is not None:
                if isinstance(batch_sample_ids, str):
                    batch_sample_ids = [batch_sample_ids]
                total_sample_ids.extend(str(item) for item in list(batch_sample_ids))

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
            self._record_progress_batch(
                str(prefix),
                int(self._trace_epoch),
                int(self.cfg.SOLVER.TOTAL_EPOCH),
                idx + 1,
                total,
                batch_time.val,
            )

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

            total_logits.append(logits.detach().to(device="cpu"))

            # visualization：当前 split 若启用，就积累 trend 并保存若干样本图
            if self._vis_split_enabled(prefix):
                affinities_vis = model_ref.get_runtime_affinities()
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
        joint_logits = torch.cat(total_logits, dim=0).numpy()

        # 调 evaluator 做正式指标计算 这里已经是在 dataset-defined eval local space 上了
        raw_metrics = self.evaluator.classify(joint_logits, total_targets)
        metrics = {
            "top1": raw_metrics["top1"],
            "top5": raw_metrics["top5"],
            "nll": raw_metrics["nll"],
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
        self.monitor_manager.set_context(
            stage="eval",
            epoch=eval_epoch,
            global_step=int(self._trace_global_step),
            graph_prob_prior_forward=int(self._graph_prob_prior_forward),
        )
        self.monitor_manager.record_epoch(
            str(prefix),
            "classification",
            metrics,
            reducer="dataset",
            n=len(dataset),
        )
        self.diagnostic_manager.record_eval(
            epoch=eval_epoch,
            split=str(prefix),
            scores=joint_logits,
            targets_local=np.asarray(total_targets, dtype=np.int64),
            dataset=dataset,
            sample_ids=(
                total_sample_ids
                if len(total_sample_ids) == len(total_targets)
                else None
            ),
        )
        epoch_elapsed = self._runtime_progress.current_epoch_elapsed()
        self._write_progress_state(
            force=True,
            status="running",
            phase=f"{prefix}_complete",
            epoch=eval_epoch,
            completed_epochs=max(0, eval_epoch - 1),
            total_epochs=int(self.cfg.SOLVER.TOTAL_EPOCH),
            batch=int(total),
            total_batches=int(total),
            epoch_elapsed_seconds=float(epoch_elapsed),
        )
        return metrics
