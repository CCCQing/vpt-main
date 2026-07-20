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
from ..monitoring import (
    DiagnosticManager,
    MonitorManager,
    NumericalGuard,
    OptimizerSanity,
    PromptParameterTracker,
)
from ..monitoring.fields import GPP_MONITOR_ALIAS_ITEMS
from ..monitoring.adapters import (
    auxiliary_loss_metrics,
    affinity_metrics,
    attention_mediation_metrics,
    graph_prob_prior_metrics,
    prompt_distribution_metrics,
    semantic_token_metrics,
    train_debug_metrics,
)
from ..monitoring.eval_metrics import (
    classification_metrics,
    representation_geometry_metrics,
    semantic_visual_graph_metrics,
    visual_semantic_alignment_metrics,
)
from ..monitoring.module_effect import (
    attention_mediation_gamma_zero_intervention,
    checkpoint_sha256,
    paired_module_effect_metrics,
    prompt_zero_intervention,
)
from ..monitoring.probe import (
    FixedProbeDataset,
    affinity_health_metrics,
    attention_flow_metrics,
    build_probe_manifest,
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


def _load_graph_attr_name_embeddings(cfg: CfgNode) -> torch.Tensor:
    """
    Trainer 侧一次性读取属性名文本 embedding。

    这样 GraphProbPriorLossComputer 只消费张量，不再读路径；
    后续如果 dataset 自带 attr_name_embeddings，也可以在 batch 侧优先使用 dataset 的版本。
    """
    graph_cfg = cfg.MODEL.GRAPH_INPUT
    path = str(graph_cfg.ATTR_NAME_EMBED_PATH)
    if not path:
        raise ValueError("MODEL.GRAPH_INPUT.ATTR_NAME_EMBED_PATH must be set when GraphProbPrior is enabled.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MODEL.GRAPH_INPUT.ATTR_NAME_EMBED_PATH not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        payload = np.load(path)
    else:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        for key in ("embeddings", "tensor"):
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, np.ndarray):
        payload = torch.from_numpy(payload)
    if not torch.is_tensor(payload):
        raise TypeError(f"ATTR_NAME_EMBED_PATH must load as tensor/ndarray/dict tensor, got {type(payload)} from {path}")

    embeddings = payload.float().contiguous()
    expected_shape = (int(graph_cfg.ATTR_DIM), int(graph_cfg.TEXT_DIM))
    if tuple(embeddings.shape) != expected_shape:
        raise ValueError(f"Expected attr_name_embeddings shape {expected_shape}, got {tuple(embeddings.shape)} from {path}")
    return embeddings


def _graph_prior_inputs_enabled(cfg: CfgNode) -> bool:
    """Trainer 侧判断是否需要预加载属性名文本 embedding。"""
    return (
        bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE)
        and float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT) > 0
        and str(cfg.MODEL.GRAPH_PROB_PRIOR.PRIOR_MEAN_MODE).lower() != "graph_gp_conditioned"
    )


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
        self.graph_attr_name_embeddings = (
            _load_graph_attr_name_embeddings(cfg)
            if _graph_prior_inputs_enabled(cfg)
            else None
        )

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
        4. r_similarity_head  visual / semantic proj
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
        refs["r_head.visual_proj"] = self._find_param_by_name_contains(
            ["r_similarity_head.visual_proj.weight", "r_similarity_head.visual_proj"]
        )
        refs["r_head.semantic_proj"] = self._find_param_by_name_contains(
            ["r_similarity_head.semantic_proj.weight", "r_similarity_head.semantic_proj"]
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
            dataset_attr_name_embeddings = None
            if dataset is not None and hasattr(dataset, "attr_name_embeddings"):
                dataset_attr_name_embeddings = dataset.attr_name_embeddings
            attr_name_embeddings = (
                dataset_attr_name_embeddings
                if dataset_attr_name_embeddings is not None
                else self.graph_attr_name_embeddings
            )
            dataset_seen_classes = getattr(dataset, "seen_classes", None) if dataset is not None else None
            dataset_unseen_classes = getattr(dataset, "unseen_classes", None) if dataset is not None else None
            loss_kwargs = {
                "model": model_ref,
                "raw_targets": targets,
                # GraphProbPrior 使用全局类别 id；不能使用 local-output remap 后的 loss_targets。
                "targets_global": effective_targets.detach(),
                "class_attributes": dataset_class_attributes,
                "attr_name_embeddings": attr_name_embeddings,
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
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(loss_stats, op=torch.distributed.ReduceOp.SUM)
            loss_sum = float(loss_stats[0].item())
            sample_count = int(round(float(loss_stats[1].item())))
            timing_stats = torch.tensor(
                [batch_time_sec, data_time_sec],
                dtype=torch.float64,
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

    def _run_train_epoch(self, epoch, effective_total_epoch, total_data, train_loader, log_interval, losses, batch_time, data_time):
        # 只负责共享的单个训练 epoch
        losses.reset()
        batch_time.reset()
        data_time.reset()

        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))

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
            if self.graph_prob_prior_loss_active:
                self._graph_prob_prior_forward += 1
            self._record_train_step_monitors(train_loss)
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
                    + self._format_graph_prob_prior_monitor_log()
                )

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
        self.monitor_manager.record_epoch(
            "train",
            "prompt_parameter_health",
            self.prompt_parameter_tracker.metrics(),
            reducer="last",
            n=1,
        )

        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def _merge_probe_layer_tensors(batch_layers):
        if not batch_layers:
            return []
        layer_count = max(len(item) for item in batch_layers)
        merged = []
        for layer_index in range(layer_count):
            tensors = []
            for layers in batch_layers:
                if layer_index < len(layers) and torch.is_tensor(layers[layer_index]):
                    tensors.append(layers[layer_index].detach().cpu())
            merged.append(torch.cat(tensors, dim=0) if tensors else None)
        return merged

    @staticmethod
    def _merge_probe_affinities(batch_affinities):
        if not batch_affinities:
            return []
        layer_count = max(len(item) for item in batch_affinities)
        merged = []
        for layer_index in range(layer_count):
            keys = set()
            for layers in batch_affinities:
                if layer_index < len(layers) and isinstance(layers[layer_index], dict):
                    keys.update(layers[layer_index].keys())
            layer = {}
            for key in sorted(keys):
                tensors = []
                for layers in batch_affinities:
                    if layer_index >= len(layers) or not isinstance(layers[layer_index], dict):
                        continue
                    value = layers[layer_index].get(key)
                    if torch.is_tensor(value):
                        tensors.append(value.detach().cpu())
                if tensors:
                    layer[key] = torch.cat(tensors, dim=0)
            merged.append(layer)
        return merged

    @torch.no_grad()
    def _execute_fixed_probe_condition(
        self,
        probe_loader,
        source_dataset,
        candidate_class_ids,
        *,
        affinity_forward=False,
    ):
        model_ref = self._model_ref(self.model)
        candidate_class_ids = [int(item) for item in candidate_class_ids]
        global_to_local = {global_id: local_id for local_id, global_id in enumerate(candidate_class_ids)}
        logits_rows = []
        target_local_rows = []
        target_global_rows = []
        sample_ids = []
        visual_rows = []
        semantic_prototypes = None
        token_rows = []
        attention_batches = []
        affinity_batches = []
        prompt_length = int(self.cfg.MODEL.PROMPT.NUM_TOKENS) if bool(self.cfg.MODEL.PROMPT.ENABLE) else 0
        semantic_length = (
            int(self.cfg.MODEL.SEMANTIC_TOKENS.NUM_TOKENS)
            if bool(self.cfg.MODEL.SEMANTIC_TOKENS.ENABLE)
            else 0
        )
        affinity_cfg = {
            "prompt_length": prompt_length,
            "semantic_length": semantic_length,
            "detach": True,
            "block_s_to_cls": bool(self.cfg.MODEL.SEMANTIC_TOKENS.BLOCK_S_TO_CLS),
        }
        for input_data in probe_loader:
            inputs, targets_global, attributes = self.get_input(input_data)
            inputs = inputs.to(self.device, non_blocking=True)
            targets_global = targets_global.to(self.device, non_blocking=True)
            semantics = self._prepare_semantics_for_stage(
                attributes,
                source_dataset,
                batch_size=int(inputs.shape[0]),
                is_train=False,
            )
            if affinity_forward:
                use_attention = bool(self.cfg.MONITOR.PROBE.ATTENTION_ENABLE)
                output = model_ref.forward_with_affinity(
                    inputs,
                    affinity_cfg,
                    semantics=semantics,
                    vis=use_attention,
                    class_ids=candidate_class_ids,
                    runtime_targets=None,
                )
                if use_attention:
                    logits, attention_layers, affinities = output
                    attention_batches.append(list(attention_layers or []))
                else:
                    logits, affinities = output
                affinity_batches.append(list(affinities or []))
                tokens = model_ref.get_runtime_token_sequence()
                if torch.is_tensor(tokens):
                    token_rows.append(tokens.detach().cpu())
            else:
                logits = model_ref(
                    inputs,
                    semantics=semantics,
                    class_ids=candidate_class_ids,
                    runtime_targets=None,
                )
            stats = model_ref.get_runtime_classifier_stats()
            if isinstance(stats, dict):
                visual = stats.get("visual_repr")
                semantic = stats.get("semantic_repr")
                if torch.is_tensor(visual):
                    visual_rows.append(visual.detach().cpu())
                if semantic_prototypes is None and torch.is_tensor(semantic):
                    semantic_prototypes = semantic.detach().cpu()
            logits_rows.append(logits.detach().cpu())
            target_global_list = [int(item) for item in targets_global.detach().cpu().tolist()]
            target_global_rows.extend(target_global_list)
            target_local_rows.extend([global_to_local[item] for item in target_global_list])
            batch_ids = input_data.get("sample_id", [])
            if isinstance(batch_ids, str):
                batch_ids = [batch_ids]
            sample_ids.extend([str(item) for item in list(batch_ids)])
        return {
            "logits": torch.cat(logits_rows, dim=0).numpy() if logits_rows else np.empty((0, len(candidate_class_ids))),
            "targets_local": np.asarray(target_local_rows, dtype=np.int64),
            "targets_global": np.asarray(target_global_rows, dtype=np.int64),
            "sample_ids": np.asarray(sample_ids),
            "visual_features": torch.cat(visual_rows, dim=0).numpy() if visual_rows else None,
            "semantic_prototypes": semantic_prototypes.numpy() if semantic_prototypes is not None else None,
            "token_sequence": torch.cat(token_rows, dim=0).numpy() if token_rows else None,
            "attention_layers": self._merge_probe_layer_tensors(attention_batches),
            "affinities": self._merge_probe_affinities(affinity_batches),
            "candidate_class_ids": np.asarray(candidate_class_ids, dtype=np.int64),
            "prompt_length": prompt_length,
            "semantic_length": semantic_length,
        }

    def _probe_metric_rows(self, metrics, *, checkpoint_id, probe_id, split, domain, entity_type="split", entity_id="all"):
        return [
            {
                "run_id": self.monitor_manager.run_id,
                "session_id": self.monitor_manager.session_id,
                "checkpoint_id": str(checkpoint_id),
                "probe_id": str(probe_id),
                "split": str(split),
                "domain": str(domain),
                "entity_type": str(entity_type),
                "entity_id": str(entity_id),
                "metric": str(name),
                "value": float(value),
            }
            for name, value in sorted(metrics.items())
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))
        ]

    @staticmethod
    def _true_margin_numpy(logits, targets):
        logits = np.asarray(logits, dtype=np.float64)
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
        normal,
        intervention_name,
        intervention,
        seen_global_ids,
    ):
        pair_id = f"{checkpoint_id}:{manifest['probe_id']}:{intervention_name}"
        effect = paired_module_effect_metrics(
            normal["logits"],
            intervention["logits"],
            normal["targets_local"],
            normal["candidate_class_ids"],
            seen_global_ids,
            normal_features=normal.get("visual_features"),
            intervention_features=intervention.get("visual_features"),
        )
        arrays_path = self.diagnostic_manager.record_module_effect_artifact(
            f"module_effect/{checkpoint_id}/{split}_{intervention_name}.npz",
            effect["arrays"],
            is_array=True,
        )
        sample_rows = []
        normal_margin = self._true_margin_numpy(normal["logits"], normal["targets_local"])
        changed_margin = self._true_margin_numpy(intervention["logits"], intervention["targets_local"])
        for index, sample_id in enumerate(normal["sample_ids"].tolist()):
            sample_rows.append({
                "pair_id": pair_id,
                "sample_id": str(sample_id),
                "condition": "normal",
                "target_local": int(normal["targets_local"][index]),
                "target_global": int(normal["targets_global"][index]),
                "prediction": int(np.argmax(normal["logits"][index])),
                "true_class_margin": float(normal_margin[index]),
                "logits": normal["logits"][index].tolist(),
            })
            sample_rows.append({
                "pair_id": pair_id,
                "sample_id": str(sample_id),
                "condition": str(intervention_name),
                "target_local": int(intervention["targets_local"][index]),
                "target_global": int(intervention["targets_global"][index]),
                "prediction": int(np.argmax(intervention["logits"][index])),
                "true_class_margin": float(changed_margin[index]),
                "logits": intervention["logits"][index].tolist(),
            })
        results_path = self.diagnostic_manager.record_module_effect_jsonl(
            f"module_effect/{checkpoint_id}/{split}_{intervention_name}.jsonl",
            sample_rows,
        )
        self.diagnostic_manager.record_module_effect_artifact(
            f"module_effect/{checkpoint_id}/{split}_{intervention_name}_summary.json",
            {
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
                "intervention_seed": int(self.cfg.MONITOR.PROBE.SELECTION_SEED),
                "shared_randomness_id": manifest["manifest_sha256"],
                "comparison_tolerance": float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL),
                "summary": effect["summary"],
                "per_class": effect["per_class"],
                "arrays_path": str(arrays_path),
                "results_path": str(results_path),
            },
        )

    def _run_fixed_probes(self, train_loader, test_seen_loader, test_unseen_loader, *, checkpoint_epoch):
        if not self.diagnostic_manager.enabled or not bool(self.cfg.MONITOR.PROBE.ENABLE):
            return
        if du.get_rank() != 0:
            return
        model_ref = self._model_ref(self.model)
        was_training = bool(model_ref.training)
        model_ref.eval()
        checkpoint_id = f"final_epoch_{int(checkpoint_epoch):04d}"
        checkpoint_manifest = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_epoch": int(checkpoint_epoch),
            "checkpoint_global_step": int(self._trace_global_step),
            "checkpoint_selection_rule": "predeclared_final_epoch",
            "source_run_id": self.monitor_manager.run_id,
            "source_session_id": self.monitor_manager.session_id,
            "checkpoint_path": self._final_trainable_checkpoint_path,
            "checkpoint_sha256": (
                checkpoint_sha256(self._final_trainable_checkpoint_path)
                if self._final_trainable_checkpoint_path and os.path.isfile(self._final_trainable_checkpoint_path)
                else None
            ),
        }
        split_sources = {
            "probe_train_seen": train_loader.dataset if train_loader is not None else None,
            "probe_test_seen": test_seen_loader.dataset if test_seen_loader is not None else None,
            "probe_test_unseen": test_unseen_loader.dataset if test_unseen_loader is not None else None,
        }
        combined_manifest = {
            "format": "baseline_fixed_probe_collection_v1",
            "run_id": self.monitor_manager.run_id,
            "session_id": self.monitor_manager.session_id,
            "checkpoint": checkpoint_manifest,
            "probes": {},
        }
        normal_results = {}
        all_rows = []
        deterministic_transform = get_transforms("test_seen", self.cfg.DATA.CROPSIZE)
        for split, dataset in split_sources.items():
            if dataset is None:
                continue
            if str(self.cfg.DATA.XLSA.PROTOCOL_MODE).lower() == "final_gzsl":
                candidate_class_ids = list(dataset.seen_classes) + list(dataset.unseen_classes)
            else:
                candidate_class_ids = list(dataset.eval_local_classes)
            manifest = build_probe_manifest(
                dataset,
                split=split,
                per_class=int(self.cfg.MONITOR.PROBE.PER_CLASS),
                max_samples=int(self.cfg.MONITOR.PROBE.MAX_SAMPLES),
                selection_seed=int(self.cfg.MONITOR.PROBE.SELECTION_SEED),
                candidate_class_ids=candidate_class_ids,
            )
            combined_manifest["probes"][split] = manifest
            self._probe_manifests[split] = manifest
            self.diagnostic_manager.record_probe_artifact(f"probe_manifests/{split}.json", manifest)
            probe_dataset = FixedProbeDataset(dataset, manifest, deterministic_transform)
            probe_loader = torch.utils.data.DataLoader(
                probe_dataset,
                batch_size=max(1, int(self.cfg.MONITOR.PROBE.BATCH_SIZE)),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
            normal = self._execute_fixed_probe_condition(
                probe_loader,
                dataset,
                candidate_class_ids,
                affinity_forward=False,
            )
            normal_results[split] = normal
            geometry = representation_geometry_metrics(normal.get("visual_features"), normal["targets_local"])
            alignment = visual_semantic_alignment_metrics(
                normal.get("visual_features"),
                normal.get("semantic_prototypes"),
                normal["targets_local"],
            )
            semantic_visual_graph = semantic_visual_graph_metrics(
                normal.get("visual_features"),
                normal.get("semantic_prototypes"),
                normal["targets_local"],
                logits=normal["logits"],
                neighbor_k=int(self.cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.NEIGHBOR_K),
            )
            all_rows.extend(self._probe_metric_rows(
                geometry,
                checkpoint_id=checkpoint_id,
                probe_id=manifest["probe_id"],
                split=split,
                domain="representation_geometry",
            ))
            all_rows.extend(self._probe_metric_rows(
                alignment,
                checkpoint_id=checkpoint_id,
                probe_id=manifest["probe_id"],
                split=split,
                domain="visual_semantic_alignment",
            ))
            all_rows.extend(self._probe_metric_rows(
                semantic_visual_graph,
                checkpoint_id=checkpoint_id,
                probe_id=manifest["probe_id"],
                split=split,
                domain="semantic_graph_reference",
                entity_type="graph",
                entity_id="visual_consistency",
            ))
            vector_payload = {
                "sample_ids": normal["sample_ids"],
                "targets_local": normal["targets_local"],
                "targets_global": normal["targets_global"],
                "candidate_class_ids": normal["candidate_class_ids"],
                "logits": normal["logits"],
            }
            if normal.get("visual_features") is not None:
                vector_payload["visual_features"] = normal["visual_features"]
            if normal.get("semantic_prototypes") is not None:
                vector_payload["semantic_prototypes"] = normal["semantic_prototypes"]
            self.diagnostic_manager.record_probe_artifact(
                f"probe_vectors/representation/{checkpoint_id}_{split}.npz",
                vector_payload,
                is_array=True,
            )

            if bool(self.cfg.MONITOR.PROBE.AFFINITY_ENABLE):
                affinity_result = self._execute_fixed_probe_condition(
                    probe_loader,
                    dataset,
                    candidate_class_ids,
                    affinity_forward=True,
                )
                normal_logits = np.asarray(normal["logits"])
                affinity_logits = np.asarray(affinity_result["logits"])
                normal_margin = self._true_margin_numpy(normal_logits, normal["targets_local"])
                affinity_margin = self._true_margin_numpy(affinity_logits, normal["targets_local"])
                logit_atol = float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL)
                margin_atol = float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_MARGIN_ATOL)
                equivalence = {
                    "logit_max_abs_diff": float(np.max(np.abs(normal_logits - affinity_logits))),
                    "true_margin_abs_diff": float(np.max(np.abs(normal_margin - affinity_margin))),
                    "prediction_flip_rate": float((normal_logits.argmax(axis=1) != affinity_logits.argmax(axis=1)).mean()),
                }
                equivalence_pass = bool(
                    equivalence["logit_max_abs_diff"] <= logit_atol
                    and equivalence["true_margin_abs_diff"] <= margin_atol
                    and equivalence["prediction_flip_rate"] == 0.0
                )
                equivalence["affinity_forward_equivalence_pass"] = float(equivalence_pass)
                self.diagnostic_manager.record_probe_artifact(
                    f"probe_equivalence/{checkpoint_id}_{split}.json",
                    {
                        "format": "affinity_forward_equivalence_v1",
                        "checkpoint_id": checkpoint_id,
                        "probe_id": manifest["probe_id"],
                        "split": split,
                        "logit_atol": logit_atol,
                        "margin_atol": margin_atol,
                        "required_prediction_flip_rate": 0.0,
                        "metrics": equivalence,
                        "valid": equivalence_pass,
                    },
                )
                all_rows.extend(self._probe_metric_rows(
                    equivalence,
                    checkpoint_id=checkpoint_id,
                    probe_id=manifest["probe_id"],
                    split=split,
                    domain="affinity_forward_equivalence",
                ))
                if equivalence_pass:
                    selected_layers = {int(item) for item in self.cfg.MONITOR.PROBE.LAYERS}
                    attention_layers = [
                        value for index, value in enumerate(affinity_result["attention_layers"])
                        if value is not None and (not selected_layers or index in selected_layers)
                    ]
                    affinity_layers = [
                        value for index, value in enumerate(affinity_result["affinities"])
                        if value and (not selected_layers or index in selected_layers)
                    ]
                    attention = attention_flow_metrics(
                        attention_layers,
                        prompt_length=int(affinity_result["prompt_length"]),
                        semantic_length=int(affinity_result["semantic_length"]),
                        affinity_layers=affinity_layers,
                        predictions=affinity_logits.argmax(axis=1),
                        targets=normal["targets_local"],
                    )
                    affinity_health = affinity_health_metrics(
                        affinity_layers,
                        temperature=float(self.cfg.MONITOR.PROBE.AFFINITY_TEMPERATURE),
                        saturation_threshold=float(self.cfg.MONITOR.PROBE.AFFINITY_SATURATION_THRESHOLD),
                    )
                    all_rows.extend(self._probe_metric_rows(
                        attention,
                        checkpoint_id=checkpoint_id,
                        probe_id=manifest["probe_id"],
                        split=split,
                        domain="attention_flow_reference",
                    ))
                    for metric_name, value in sorted(affinity_health.items()):
                        relation, _, metric = metric_name.partition(".")
                        all_rows.extend(self._probe_metric_rows(
                            {metric or relation: value},
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="affinity_health",
                            entity_type="relation",
                            entity_id=relation if metric else "global",
                        ))
                else:
                    logger.warning(
                        "[fixed-probe] affinity diagnostics skipped for split=%s because forward equivalence failed "
                        "(logit=%.3e, margin=%.3e, flip_rate=%.3e)",
                        split,
                        equivalence["logit_max_abs_diff"],
                        equivalence["true_margin_abs_diff"],
                        equivalence["prediction_flip_rate"],
                    )
                token_sequence = affinity_result.get("token_sequence")
                if equivalence_pass and token_sequence is not None:
                    normal_results[split]["token_sequence"] = token_sequence
                    prompt_length = int(affinity_result["prompt_length"])
                    semantic_length = int(affinity_result["semantic_length"])
                    patch_start = 1 + prompt_length
                    patch_end = int(token_sequence.shape[1]) - semantic_length
                    token_views = {
                        "cls": token_sequence[:, 0, :],
                        "pooled_patch": token_sequence[:, patch_start:patch_end, :].mean(axis=1),
                    }
                    if prompt_length > 0:
                        token_views["contextualized_prompt"] = token_sequence[:, 1:patch_start, :].mean(axis=1)
                    if semantic_length > 0:
                        token_views["semantic_token"] = token_sequence[:, patch_end:, :].mean(axis=1)
                    for entity_id, values in token_views.items():
                        token_geometry = representation_geometry_metrics(values, normal["targets_local"])
                        all_rows.extend(self._probe_metric_rows(
                            token_geometry,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="representation_geometry",
                            entity_type="representation",
                            entity_id=entity_id,
                        ))
                    if prompt_length > 0:
                        prompt_output = token_views["contextualized_prompt"]
                        cls_output = token_views["cls"]
                        patch_output = token_views["pooled_patch"]
                        prompt_norm = np.linalg.norm(prompt_output, axis=1)
                        prompt_normalized = prompt_output / np.maximum(prompt_norm[:, None], 1e-12)
                        cls_normalized = cls_output / np.maximum(np.linalg.norm(cls_output, axis=1, keepdims=True), 1e-12)
                        patch_normalized = patch_output / np.maximum(np.linalg.norm(patch_output, axis=1, keepdims=True), 1e-12)
                        prompt_response = {
                            "contextualized_prompt_instance_variance": float(np.var(prompt_output, axis=0).mean()),
                            "prompt_cls_cosine": float(np.sum(prompt_normalized * cls_normalized, axis=1).mean()),
                            "prompt_patch_cosine": float(np.sum(prompt_normalized * patch_normalized, axis=1).mean()),
                            "contextualized_prompt_norm": float(prompt_norm.mean()),
                        }
                        normal_results[split]["prompt_response"] = prompt_response
                        all_rows.extend(self._probe_metric_rows(
                            prompt_response,
                            checkpoint_id=checkpoint_id,
                            probe_id=manifest["probe_id"],
                            split=split,
                            domain="prompt_parameter_health",
                            entity_type="representation",
                            entity_id="contextualized_prompt",
                        ))
            if bool(self.cfg.MONITOR.MODULE_EFFECT.ENABLE):
                interventions = []
                if bool(self.cfg.MONITOR.MODULE_EFFECT.PROMPT_ZERO) and self.prompt_parameter_tracker.active:
                    interventions.append(("prompt_zeroed", prompt_zero_intervention(model_ref)))
                if (
                    bool(self.cfg.MONITOR.MODULE_EFFECT.ATTENTION_MEDIATION_GAMMA_ZERO)
                    and bool(self.cfg.MODEL.ATTENTION_MEDIATION.ENABLE)
                ):
                    interventions.append(("attention_mediation_gamma_zero", attention_mediation_gamma_zero_intervention(model_ref)))
                for intervention_name, context in interventions:
                    with context:
                        changed = self._execute_fixed_probe_condition(
                            probe_loader,
                            dataset,
                            candidate_class_ids,
                            affinity_forward=False,
                        )
                    self._record_probe_module_effect(
                        checkpoint_id=checkpoint_id,
                        checkpoint_manifest=checkpoint_manifest,
                        split=split,
                        manifest=manifest,
                        normal=normal,
                        intervention_name=intervention_name,
                        intervention=changed,
                        seen_global_ids=dataset.seen_classes,
                    )

        self.diagnostic_manager.record_probe_artifact("probe_manifest.json", combined_manifest)
        if bool(self.cfg.MONITOR.MODULE_EFFECT.ENABLE):
            configured_conditions = []
            if bool(self.cfg.MONITOR.MODULE_EFFECT.PROMPT_ZERO) and self.prompt_parameter_tracker.active:
                configured_conditions.append("prompt_zeroed")
            if (
                bool(self.cfg.MONITOR.MODULE_EFFECT.ATTENTION_MEDIATION_GAMMA_ZERO)
                and bool(self.cfg.MODEL.ATTENTION_MEDIATION.ENABLE)
            ):
                configured_conditions.append("attention_mediation_gamma_zero")
            self.diagnostic_manager.record_module_effect_artifact(
                "module_effect_manifest.json",
                {
                    "format": "baseline_module_effect_v1",
                    "checkpoint": checkpoint_manifest,
                    "probe_manifest_path": "probe_manifest.json",
                    "probe_manifest_sha256_by_split": {
                        split: manifest["manifest_sha256"]
                        for split, manifest in combined_manifest["probes"].items()
                    },
                    "conditions": ["normal"] + configured_conditions,
                    "model_eval": True,
                    "torch_no_grad": True,
                    "intervention_seed": int(self.cfg.MONITOR.PROBE.SELECTION_SEED),
                    "comparison_tolerance": {
                        "logit_atol": float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_ATOL),
                        "margin_atol": float(self.cfg.MONITOR.PROBE.FORWARD_EQUIVALENCE_MARGIN_ATOL),
                        "prediction_flip_rate": 0.0,
                    },
                },
            )
        self.diagnostic_manager.append_probe_metrics(all_rows)
        if "probe_test_seen" in normal_results and "probe_test_unseen" in normal_results:
            seen_result = normal_results["probe_test_seen"]
            unseen_result = normal_results["probe_test_unseen"]
            seen_alignment = visual_semantic_alignment_metrics(
                seen_result.get("visual_features"), seen_result.get("semantic_prototypes"), seen_result["targets_local"]
            )
            unseen_alignment = visual_semantic_alignment_metrics(
                unseen_result.get("visual_features"), unseen_result.get("semantic_prototypes"), unseen_result["targets_local"]
            )
            transfer = {
                f"{key}_transfer_gap": float(seen_alignment[key] - unseen_alignment[key])
                for key in sorted(set(seen_alignment).intersection(unseen_alignment))
            }
            combined_visual_graph = semantic_visual_graph_metrics(
                np.concatenate([seen_result["visual_features"], unseen_result["visual_features"]], axis=0)
                if seen_result.get("visual_features") is not None and unseen_result.get("visual_features") is not None
                else None,
                seen_result.get("semantic_prototypes"),
                np.concatenate([seen_result["targets_local"], unseen_result["targets_local"]], axis=0),
                logits=np.concatenate([seen_result["logits"], unseen_result["logits"]], axis=0),
                neighbor_k=int(self.cfg.MONITOR.SEMANTIC_GRAPH_REFERENCE.NEIGHBOR_K),
            )
            transfer.update({f"combined_{name}": value for name, value in combined_visual_graph.items()})
            seen_prompt = seen_result.get("prompt_response", {})
            unseen_prompt = unseen_result.get("prompt_response", {})
            if "contextualized_prompt_norm" in seen_prompt and "contextualized_prompt_norm" in unseen_prompt:
                transfer["prompt_seen_unseen_response_gap"] = float(
                    seen_prompt["contextualized_prompt_norm"] - unseen_prompt["contextualized_prompt_norm"]
                )
            all_transfer_rows = self._probe_metric_rows(
                transfer,
                checkpoint_id=checkpoint_id,
                probe_id="paired_test_seen_unseen",
                split="test_seen_vs_test_unseen",
                domain="visual_semantic_alignment",
                entity_type="cross_split",
                entity_id="seen_minus_unseen",
            )
            self.diagnostic_manager.append_probe_metrics(all_transfer_rows)
        if "probe_train_seen" in normal_results and "probe_test_seen" in normal_results:
            train_result = normal_results["probe_train_seen"]
            test_result = normal_results["probe_test_seen"]
            train_metric = classification_metrics(train_result["logits"], train_result["targets_local"])
            test_metric = classification_metrics(test_result["logits"], test_result["targets_local"])
            gap_rows = self._probe_metric_rows(
                {
                    "train_joint_to_test_seen_gap": float(train_metric["per_class"] - test_metric["per_class"]),
                },
                checkpoint_id=checkpoint_id,
                probe_id="paired_train_test_seen",
                split="train_seen_vs_test_seen",
                domain="prediction_health",
                entity_type="cross_split",
                entity_id="train_minus_test_seen",
            )
            self.diagnostic_manager.append_probe_metrics(gap_rows)
        self.diagnostic_manager.record_probe_artifact(
            "probe_runtime_summary.json",
            {
                "status": "completed",
                "checkpoint": checkpoint_manifest,
                "probe_count": int(len(combined_manifest["probes"])),
                "metric_row_count": int(len(all_rows)),
            },
        )
        if was_training:
            model_ref.train()

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

            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

    def _train_classifier_final(self, train_loader, test_seen_loader, test_unseen_loader):
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
        if du.get_world_size() > 1:
            torch.distributed.barrier()

    def _save_trainable_final_checkpoint(self, total_epoch):
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

    def train_classifier(self, train_loader, val_loader, test_seen_loader, test_unseen_loader):
        completed = False
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
                )
            completed = True
            return result
        finally:
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
        total_global_targets = []
        total_sample_ids = []
        total_visual_features = []
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
            total_global_targets.extend(list(targets.detach().cpu().numpy()))
            batch_sample_ids = input_data.get("sample_id") if isinstance(input_data, dict) else None
            if batch_sample_ids is None:
                batch_sample_ids = [f"{prefix}:{idx}:{item}" for item in range(int(targets.shape[0]))]
            elif isinstance(batch_sample_ids, str):
                batch_sample_ids = [batch_sample_ids]
            elif torch.is_tensor(batch_sample_ids):
                batch_sample_ids = batch_sample_ids.detach().cpu().tolist()
            total_sample_ids.extend([str(item) for item in list(batch_sample_ids)])

            # 提取 logits
            logits = outputs
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                logits = outputs[0]
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]

            total_logits.append(logits)
            classifier_stats = model_ref.get_runtime_classifier_stats()
            if isinstance(classifier_stats, dict):
                visual_repr = classifier_stats.get("visual_repr")
                if torch.is_tensor(visual_repr) and int(visual_repr.shape[0]) == int(targets.shape[0]):
                    total_visual_features.append(visual_repr.detach().cpu())

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
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()

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
        visual_features = None
        if total_visual_features:
            candidate_visual = torch.cat(total_visual_features, dim=0)
            if int(candidate_visual.shape[0]) == int(joint_logits.shape[0]):
                visual_features = candidate_visual.numpy()
        self.diagnostic_manager.record_eval(
            epoch=eval_epoch,
            split=str(prefix),
            scores=joint_logits,
            targets_local=np.asarray(total_targets, dtype=np.int64),
            targets_global=np.asarray(total_global_targets, dtype=np.int64),
            sample_ids=total_sample_ids,
            dataset=dataset,
            visual_features=visual_features,
        )
        return metrics
