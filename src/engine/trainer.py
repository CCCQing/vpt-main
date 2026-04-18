#!/usr/bin/env python3
"""
trainer.py

这个文件实现的是整个工程里的“训练/评测总调度器（Trainer）”。

你可以把它理解成：
    训练主链的执行中心 + 调试与可视化中心 + monitor 诊断中心

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
        -> visualization / monitor
        -> checkpoint / best epoch / early-stop

核心职责概览：
1. 构建优化器、学习率调度器、损失函数
2. 根据配置决定是否启用 affinity 分支、辅助损失、visualization、monitor
3. 管理 local-output / eval-local-output 的 target remap
4. 组织训练循环 train_classifier()
5. 组织单次 batch 前向 forward_one_batch()
6. 组织评测 eval_classifier()
7. 组织 monitor 指标与可视化导出
"""
import datetime
import time
import torch
import torch.nn as nn
import os
import numpy as np
import json
import csv
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
    ensure_dir,
    to_uint8_image,
    resize_map_torch,
    overlay_heatmap,
    save_overlay,
    save_panel,
    attention_rollout,
    entropy_lastdim,
    append_csv_row,
    save_json,
)

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    璁粌涓婚摼鐨勬墽琛屼腑蹇?+ 璋冭瘯涓庡彲瑙嗗寲涓績 + monitor 璇婃柇涓績
    瀹冩湰韬笉璐熻矗锛?
    1. 瀹氫箟鏁版嵁鍗忚锛堣繖宸茬粡鍦?dataset 灞傚畾涔夊ソ浜嗭級
    2. 瀹氫箟妯″瀷缁撴瀯锛堟ā鍨嬪湪 build_model / vit_models / losses 涓畾涔夛級
    3. 瀹氫箟璇勬祴鏁板锛坰inglelabel / evaluator 涓畾涔夛級

    a trainer with below logics: 璁粌鍣紙Trainer锛変富瑕侀€昏緫
    1. 鏋勫缓浼樺寲鍣ㄣ€佸涔犵巼璋冨害鍣ㄣ€佹崯澶卞嚱鏁?
    2. 鏍规嵁閰嶇疆鍐冲畾鏄惁鍚敤 affinity 鍒嗘敮銆佽緟鍔╂崯澶便€乿isualization銆乵onitor
    3. 绠＄悊 local-output / eval-local-output 鐨?target remap
    4. 缁勭粐璁粌寰幆 train_classifier()
    5. 缁勭粐鍗曟 batch 鍓嶅悜 forward_one_batch()
    6. 缁勭粐璇勬祴 eval_classifier()
    7. 缁勭粐 monitor 鎸囨爣涓庡彲瑙嗗寲瀵煎嚭
    """
    def __init__(self, cfg: CfgNode, model: nn.Module, evaluator: Evaluator, device: torch.device,) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # Affinity aux is only enabled for the current role/entropy scoring path.
        self.affinity_aux_needed = (
            cfg.SOLVER.LOSS_ROLE_EARLY_WEIGHT > 0
            or cfg.SOLVER.LOSS_ROLE_LATE_WEIGHT > 0
            or cfg.SOLVER.LOSS_AVS_ENT_WEIGHT > 0
        )

        # 涓€涓负鐪熷嵆use_affinity
        self.use_affinity = cfg.MODEL.AFFINITY.ENABLE or self.affinity_aux_needed
        if self.use_affinity:
            self.affinity_cfg = {
                "prompt_length": cfg.MODEL.PROMPT.NUM_TOKENS,
                "return_cross": cfg.MODEL.AFFINITY.RETURN_CROSS or self.affinity_aux_needed,
                "normalize": cfg.MODEL.AFFINITY.NORMALIZE,
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
        self.cls_criterion = build_loss(self.cfg)

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

        self.patch_compete_balance_weight = cfg.MODEL.AFFINITY.PATCH_COMPETE_BALANCE_WEIGHT

        self._trace_epoch = -1
        self._trace_iter = -1
        self._trace_stage = "init"
        self._trace_global_step = 0
        self._trace_rank = int(getattr(cfg, "DIST_RANK", 0))

        diag_cfg = cfg.SOLVER.DIAG
        self.diag_shuffle_raw_targets = diag_cfg.SHUFFLE_RAW_TARGETS

        # ================== monitor ==================
        mon_cfg = cfg.SOLVER.MONITOR
        self.monitor_enable = mon_cfg.ENABLE
        self.monitor_every_epoch = max(1, mon_cfg.EVERY_EPOCH)
        self.monitor_max_samples = max(1, mon_cfg.MAX_SAMPLES)
        self.monitor_save_json = mon_cfg.SAVE_JSON
        self.monitor_save_csv = mon_cfg.SAVE_CSV
        self.monitor_save_heatmap = mon_cfg.SAVE_HEATMAP
        self.monitor_heatmap_topk = max(1, int(mon_cfg.HEATMAP_TOPK))

        self.token_patch_stats_enable = bool(mon_cfg.TOKEN_PATCH_STATS_ENABLE)
        self.token_patch_source = str(mon_cfg.TOKEN_PATCH_SOURCE).lower()
        self.token_patch_head_mode = str(mon_cfg.TOKEN_PATCH_HEAD_MODE).lower()
        self.token_patch_toprho = float(mon_cfg.TOKEN_PATCH_TOPRHO)
        self.token_patch_save_maps = bool(mon_cfg.TOKEN_PATCH_SAVE_MAPS)
        self.token_patch_max_samples = max(1, int(mon_cfg.TOKEN_PATCH_MAX_SAMPLES))

        self.affinity_summary_enable = bool(mon_cfg.AFFINITY_SUMMARY_ENABLE)
        self.affinity_save_raw_dump = bool(mon_cfg.AFFINITY_SAVE_RAW_DUMP)
        self.affinity_keynode_viz_enable = bool(mon_cfg.AFFINITY_KEYNODE_VIZ_ENABLE)
        self.affinity_keynode_splits = [str(x).lower() for x in list(mon_cfg.AFFINITY_KEYNODE_SPLITS)]
        self.affinity_keynode_max_figs = max(0, int(mon_cfg.AFFINITY_KEYNODE_MAX_FIGS))
        self.affinity_keynode_layer_policy = str(mon_cfg.AFFINITY_KEYNODE_LAYER_POLICY).lower()
        self.monitor_dir = os.path.join(self.cfg.OUTPUT_DIR, "monitor")
        self._monitor_csv_path = os.path.join(self.monitor_dir, "summary.csv")
        self._monitor_warned_no_refined = False

        vis_cfg = cfg.SOLVER.VIS
        self.vis_enable = bool(vis_cfg.ENABLE)
        self.vis_every_epoch = max(1, int(vis_cfg.EVERY_EPOCH))
        self.vis_epoch_list = self._parse_vis_epoch_list(vis_cfg.EPOCH_LIST)
        self.vis_splits = list(vis_cfg.SPLITS)
        self.vis_max_samples = max(1, int(vis_cfg.MAX_SAMPLES))
        self.vis_save_raw = bool(vis_cfg.SAVE_RAW)
        self.vis_save_images = bool(vis_cfg.SAVE_IMAGES)
        self.vis_local_control = bool(vis_cfg.LOCAL_CONTROL)
        self.vis_rollout = bool(vis_cfg.ROLLOUT)
        self.vis_gt_hn = bool(vis_cfg.GT_HN_COMPARE)
        self.vis_trends = bool(vis_cfg.TRENDS)
        self.vis_dir = os.path.join(self.cfg.OUTPUT_DIR, "visualization")
        self._last_attn_weights = None
        self._vis_trend_buf = None
        self._vis_processed = 0

        if self.vis_enable:
            ensure_dir(self.vis_dir)
            # rollout needs per-layer attention weights
            self.affinity_vis = True
            logger.info(
                "[vis] enable=%s every_epoch=%d epoch_list=%s splits=%s max_samples=%d save_raw=%s save_images=%s",
                bool(self.vis_enable),
                int(self.vis_every_epoch),
                self.vis_epoch_list,
                self.vis_splits,
                int(self.vis_max_samples),
                bool(self.vis_save_raw),
                bool(self.vis_save_images),
            )

        if self.debug_grad_norm:
            self._log_optimizer_param_groups()

        if self.debug_trace_once:
            self._log_semantic_param_names_once()

        # output monitor
        if self.monitor_enable:
            os.makedirs(self.monitor_dir, exist_ok=True)
            logger.info(
                "[monitor] enable=%s every_epoch=%d max_samples=%d save_json=%s save_csv=%s save_heatmap=%s",
                bool(self.monitor_enable),
                int(self.monitor_every_epoch),
                int(self.monitor_max_samples),
                bool(self.monitor_save_json),
                bool(self.monitor_save_csv),
                bool(self.monitor_save_heatmap),
            )
            logger.info(
                "[monitor-token-patch] enable=%s source=%s head_mode=%s toprho=%.3f save_maps=%s max_samples=%d",
                bool(self.token_patch_stats_enable),
                self.token_patch_source,
                self.token_patch_head_mode,
                float(self.token_patch_toprho),
                bool(self.token_patch_save_maps),
                int(self.token_patch_max_samples),
            )
            logger.info(
                "[monitor-affinity] summary=%s raw_dump=%s keynode_viz=%s keynode_splits=%s keynode_max_figs=%d keynode_layer_policy=%s",
                bool(self.affinity_summary_enable),
                bool(self.affinity_save_raw_dump),
                bool(self.affinity_keynode_viz_enable),
                self.affinity_keynode_splits,
                int(self.affinity_keynode_max_figs),
                self.affinity_keynode_layer_policy,
            )

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
        refs["semantic_branch.anchor_slot"] = self._find_param_by_name_contains(
            ["semantic_side_branch.anchor_slot_embed", "semantic_side_branch.anchor_token_init"]
        )
        refs["semantic_branch.readout"] = self._find_param_by_name_contains(
            ["semantic_side_branch.readout_gate.weight", "semantic_side_branch.readout_norm.weight"]
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
        names = [n for n, _ in self.model.named_parameters() if "semantic_side_branch" in n]
        logger.info(
            "[trace] semantic_side_branch param names (%d): %s",
            len(names),
            names if len(names) <= 40 else names[:40] + ["..."],
        )
        self._debug_semantic_param_names_logged = True

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
                any(("semantic_side_branch" in n) or ("semantic_anchor" in n) for n in names),
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
                fixed_scale = float(getattr(r_head, "fixed_logit_scale", 0.0))
                learnable_scale = None
                if getattr(r_head, "logit_scale", None) is not None:
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
            - HN margin loss 的辅助统计
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
                fixed_scale = float(getattr(r_head, "fixed_logit_scale", 0.0))
                self._last_train_debug["whether_fixed_logit_scale"] = bool(fixed_scale > 0)
                scale_t = getattr(r_head, "_loss_last_scale", None)
                if torch.is_tensor(scale_t):
                    self._last_train_debug["effective_logit_scale"] = float(scale_t.detach().mean().item())
            hn_stats = getattr(self.cls_criterion, "_last_hn_stats", None)
            if isinstance(hn_stats, dict) and len(hn_stats) > 0:
                self._last_train_debug.update(hn_stats)

    ##=========================== 4. affinity / token-patch helpers=========================
    @staticmethod
    def _infer_grid(num_patches: int) -> int:
        g = int(round(math.sqrt(max(1, int(num_patches)))))
        return max(1, g)

    @staticmethod
    def _entropy_np(x: torch.Tensor) -> float:
        """Calculate the average entropy of a tensor along the last dimension"""
        if (not torch.is_tensor(x)) or x.numel() == 0:
            return float("nan")
        return float(entropy_lastdim(x).mean().item())

    @staticmethod
    def _gini_np(x: np.ndarray) -> float:
        """计算一维非负向量的 Gini 系数。

        用途：
        - 衡量 token usage 是否集中
        - 衡量 monopoly 程度
        - 衡量分配是否不均匀"""
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        arr = np.maximum(arr, 0.0)
        s = float(arr.sum())
        if s <= 0:
            return float("nan")
        arr = np.sort(arr)
        n = arr.size
        idx = np.arange(1, n + 1, dtype=np.float64)
        g = (2.0 * np.sum(idx * arr) / (n * s)) - (n + 1.0) / n
        return float(g)

    @staticmethod
    def _matrix_head_mean(x: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(x)) or x.numel() == 0:
            return None
        t = x.detach().float()
        if t.dim() == 4:
            t = t.mean(dim=1)  # [B,Q,K]
        if t.dim() != 3:
            return None
        return t

    @staticmethod
    def _matrix_stats_from_bqk(a: torch.Tensor) -> dict:
        """从 [B, Q, K] affinity 矩阵中提取一批统计量。

        统计包括：
        1. 基础统计：
           - n_samples
           - q_len / k_len
           - mean / std / min / max

        2. row-wise 分布统计：
           - row_entropy_mean / p50 / p90
           - row_monopoly_mean / p50 / p90
           - row_gini_mean / p50 / p90

        3. 谱统计：
           - effective_rank_mean
           - sv_top1/2/3_cum_mean"""
        if (not torch.is_tensor(a)) or a.dim() != 3 or a.numel() == 0:
            return {}
        out = {}
        b, q, k = a.shape
        x = a.detach().float()
        flat = x.reshape(-1)
        out["n_samples"] = int(b)
        out["q_len"] = int(q)
        out["k_len"] = int(k)
        out["mean"] = float(flat.mean().item())
        out["std"] = float(flat.std().item())
        out["min"] = float(flat.min().item())
        out["max"] = float(flat.max().item())

        # Row-wise distribution stats after safe renorm.
        p = x.clamp_min(0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        row_ent = (-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=-1)).reshape(-1).cpu().numpy()
        if row_ent.size > 0:
            out["row_entropy_mean"] = float(np.mean(row_ent))
            out["row_entropy_p50"] = float(np.percentile(row_ent, 50))
            out["row_entropy_p90"] = float(np.percentile(row_ent, 90))

        row_mon = p.max(dim=-1).values.reshape(-1).cpu().numpy()
        if row_mon.size > 0:
            out["row_monopoly_mean"] = float(np.mean(row_mon))
            out["row_monopoly_p50"] = float(np.percentile(row_mon, 50))
            out["row_monopoly_p90"] = float(np.percentile(row_mon, 90))

        row_g = []
        p_np = p.cpu().numpy().reshape(-1, k)
        for r in p_np:
            g = Trainer._gini_np(r)
            if np.isfinite(g):
                row_g.append(g)
        if len(row_g) > 0:
            g_arr = np.asarray(row_g, dtype=np.float64)
            out["row_gini_mean"] = float(np.mean(g_arr))
            out["row_gini_p50"] = float(np.percentile(g_arr, 50))
            out["row_gini_p90"] = float(np.percentile(g_arr, 90))

        # Effective rank / singular spectrum (batch mean).
        er_vals, c1_vals, c2_vals, c3_vals = [], [], [], []
        for bi in range(min(int(b), 8)):  # cap for speed
            try:
                s = torch.linalg.svdvals(p[bi])
            except Exception:
                continue
            if s.numel() == 0:
                continue
            ps = s / s.sum().clamp_min(1e-12)
            er = torch.exp(-(ps * ps.clamp_min(1e-12).log()).sum())
            er_vals.append(float(er.item()))
            c1_vals.append(float(ps[:1].sum().item()))
            c2_vals.append(float(ps[:2].sum().item()))
            c3_vals.append(float(ps[:3].sum().item()))
        if len(er_vals) > 0:
            out["effective_rank_mean"] = float(np.mean(er_vals))
            out["sv_top1_cum_mean"] = float(np.mean(c1_vals))
            out["sv_top2_cum_mean"] = float(np.mean(c2_vals))
            out["sv_top3_cum_mean"] = float(np.mean(c3_vals))
        return out

    @staticmethod
    def _matrix_sources_meta() -> dict:
        return {
            "vv": {
                "raw_key": "Avv",
                "source_path_or_source_name": "vit_backbones.vit.Attention.compute_affinity",
                "softmax_dim_name": "key_patch(last_dim)",
            },
            "pp": {
                "raw_key": "App",
                "source_path_or_source_name": "vit_backbones.vit.Attention.compute_affinity",
                "softmax_dim_name": "key_prompt(last_dim)",
            },
            "pv": {
                "raw_key": "Apv",
                "source_path_or_source_name": "vit_backbones.vit.Attention.compute_affinity",
                "softmax_dim_name": "key_patch(last_dim)",
            },
            "vs": {
                "raw_key": "Avs",
                "source_path_or_source_name": "vit_prompt.vit.LateSemanticSideBranch.step",
                "softmax_dim_name": "key_semantic(last_dim)",
            },
            "ps": {
                "raw_key": "Aps",
                "source_path_or_source_name": "vit_prompt.vit.LateSemanticSideBranch.step",
                "softmax_dim_name": "key_semantic(last_dim)",
            },
            "sp": {
                "raw_key": "Asp",
                "source_path_or_source_name": "vit_prompt.vit.LateSemanticSideBranch.step",
                "softmax_dim_name": "key_prompt(last_dim)",
            },
            "sv": {
                "raw_key": "Asv",
                "source_path_or_source_name": "vit_prompt.vit.LateSemanticSideBranch.step",
                "softmax_dim_name": "key_visual(last_dim)",
            },
        }

    def _token_patch_from_affinity(self, aff: dict) -> torch.Tensor:
        if not isinstance(aff, dict):
            return None
        source = str(self.token_patch_source).lower()
        x = None
        if source in {"asv", "avs"}:
            x = aff.get("Asv")
            if (not torch.is_tensor(x)) and source == "avs":
                x = aff.get("Avs")
        elif source in {"patch_compete", "compete", "patch_compete_map"}:
            x = aff.get("PatchCompeteMap")
            if torch.is_tensor(x) and x.dim() == 3:
                a = x.detach().float()  # already [B,T,P]
                a = a.clamp_min(0.0)
                a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                return a
        if not torch.is_tensor(x) or x.numel() == 0:
            return None
        a = x.detach().float()
        # [B,H,T,P] or [B,T,P] -> [B,T,P]
        if a.dim() == 4:
            if self.token_patch_head_mode == "head0":
                a = a[:, 0, :, :]
            else:
                a = a.mean(dim=1)
        if a.dim() != 3:
            return None
        a = a.clamp_min(0.0)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return a

    @staticmethod
    def _upper_tri_values(x: torch.Tensor) -> torch.Tensor:
        """Extract the upper triangular elements of a symmetric matrix"""
        if (not torch.is_tensor(x)) or x.dim() < 2 or x.shape[-1] < 2:
            return None
        n = x.shape[-1]
        idx = torch.triu_indices(n, n, offset=1, device=x.device)
        if x.dim() == 2:
            return x[idx[0], idx[1]]
        if x.dim() == 3:
            return x[:, idx[0], idx[1]].reshape(-1)
        return None

    def _token_patch_batch_stats(self, a_t_p: torch.Tensor) -> dict:
        """        输出统计包括：
        1. per-token patch distribution 的 entropy / monopoly / gini
        2. token-token overlap：
           - cosine
           - top-rho IoU
        3. 奇异值谱统计：
           - effective rank
           - sv top1/top2/top3 累积占比
           - sv mean spectrum"""
        if (not torch.is_tensor(a_t_p)) or a_t_p.dim() != 3 or a_t_p.numel() == 0:
            return {}
        out = {}
        b, t, p = a_t_p.shape
        p_float = float(max(p, 1))

        # per-token patch-distribution stats
        ent = (-(a_t_p.clamp_min(1e-12) * a_t_p.clamp_min(1e-12).log()).sum(dim=-1))  # [B,T]
        mon = a_t_p.max(dim=-1).values  # [B,T]

        def _summ(v: torch.Tensor, prefix: str):
            if (not torch.is_tensor(v)) or v.numel() == 0:
                return
            out[f"{prefix}_mean"] = float(v.mean().item())
            out[f"{prefix}_std"] = float(v.std().item())
            out[f"{prefix}_min"] = float(v.min().item())
            out[f"{prefix}_max"] = float(v.max().item())

        _summ(ent, "patchdist_entropy")
        _summ(mon, "patchdist_monopoly")

        g_vals = []
        a_np = a_t_p.detach().cpu().numpy().reshape(-1, p)
        for row in a_np:
            g = self._gini_np(row)
            if np.isfinite(g):
                g_vals.append(g)
        if len(g_vals) > 0:
            g_arr = np.asarray(g_vals, dtype=np.float64)
            out["patchdist_gini_mean"] = float(np.mean(g_arr))
            out["patchdist_gini_std"] = float(np.std(g_arr))
            out["patchdist_gini_min"] = float(np.min(g_arr))
            out["patchdist_gini_max"] = float(np.max(g_arr))

        # overlap: cosine
        a_n = torch.nn.functional.normalize(a_t_p, dim=-1)
        cos = torch.einsum("btp,bup->btu", a_n, a_n)
        ut_cos = self._upper_tri_values(cos)
        if torch.is_tensor(ut_cos) and ut_cos.numel() > 0:
            out["patchdist_overlap_cos_mean"] = float(ut_cos.mean().item())
            out["patchdist_overlap_cos_max"] = float(ut_cos.max().item())

        # overlap: top-rho IoU
        k = max(1, int(round(float(self.token_patch_toprho) * p_float)))
        topk_idx = torch.topk(a_t_p, k=k, dim=-1).indices
        mask = torch.zeros_like(a_t_p, dtype=torch.float32)
        mask.scatter_(dim=-1, index=topk_idx, value=1.0)
        inter = torch.einsum("btp,bup->btu", mask, mask)
        msum = mask.sum(dim=-1, keepdim=True)
        union = msum + msum.transpose(1, 2) - inter
        iou = inter / union.clamp_min(1e-12)
        ut_iou = self._upper_tri_values(iou)
        if torch.is_tensor(ut_iou) and ut_iou.numel() > 0:
            out["patchdist_overlap_iou_mean"] = float(ut_iou.mean().item())
            out["patchdist_overlap_iou_max"] = float(ut_iou.max().item())

        # effective rank + singular spectrum
        er_vals = []
        c1_vals = []
        c2_vals = []
        c3_vals = []
        sv_list = []
        for bi in range(b):
            try:
                s = torch.linalg.svdvals(a_t_p[bi])  # [min(T,P)]
            except Exception:
                continue
            if s.numel() == 0:
                continue
            sv_list.append(s.detach().cpu())
            ps = s / s.sum().clamp_min(1e-12)
            er = torch.exp(-(ps * ps.clamp_min(1e-12).log()).sum())
            er_vals.append(float(er.item()))
            c1_vals.append(float(ps[:1].sum().item()))
            c2_vals.append(float(ps[:2].sum().item()))
            c3_vals.append(float(ps[:3].sum().item()))
        if len(er_vals) > 0:
            out["patchdist_effective_rank_mean"] = float(np.mean(er_vals))
            out["patchdist_sv_top1_cum_mean"] = float(np.mean(c1_vals))
            out["patchdist_sv_top2_cum_mean"] = float(np.mean(c2_vals))
            out["patchdist_sv_top3_cum_mean"] = float(np.mean(c3_vals))
        if len(sv_list) > 0:
            max_r = max(int(x.numel()) for x in sv_list)
            sv_mat = np.full((len(sv_list), max_r), np.nan, dtype=np.float64)
            for i, s in enumerate(sv_list):
                n = int(s.numel())
                sv_mat[i, :n] = s.numpy()
            out["patchdist_sv_mean"] = np.nanmean(sv_mat, axis=0).tolist()

        return out

    @staticmethod
    def _safe_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x.float(), dim=-1)
        return x @ x.t()

    @staticmethod
    def _upper_tri_flat(m: torch.Tensor) -> torch.Tensor:
        if m.dim() != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
            return torch.empty(0, device=m.device)
        idx = torch.triu_indices(m.shape[0], m.shape[1], offset=1, device=m.device)
        return m[idx[0], idx[1]]

    @staticmethod
    def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.float().reshape(-1)
        y = y.float().reshape(-1)
        if x.numel() != y.numel() or x.numel() < 2:
            return x.new_tensor(float("nan"))
        x = x - x.mean()
        y = y - y.mean()
        denom = (x.norm(p=2) * y.norm(p=2)).clamp_min(1e-12)
        return (x * y).sum() / denom

    def _select_key_layers(self, n_layers: int) -> list:
        if n_layers <= 0:
            return []
        if self.affinity_keynode_layer_policy == "all":
            return list(range(n_layers))
        # default: first / middle / last
        ids = [0, n_layers // 2, n_layers - 1]
        return sorted(set([int(x) for x in ids if 0 <= int(x) < n_layers]))

    @staticmethod
    def _key_nodes_for_matrix(m: np.ndarray, matrix_name: str) -> dict:
        """
        针对一个二维 affinity 矩阵，抽取“关键节点”。

        不同矩阵类型对应不同语义：
        - vv / pp:
            sink_like_node
            focused_node
            diffuse_node
        - pv / ps / sp:
            focused_prompt
            diffuse_prompt
        - vs / sv:
            confident_patch
            ambiguous_patch

        这些 key nodes 主要用于可视化标注。
        """
        if m.ndim != 2 or m.size == 0:
            return {}
        eps = 1e-12
        p = np.clip(m, 0.0, None)
        p = p / np.clip(np.sum(p, axis=1, keepdims=True), eps, None)
        row_ent = -np.sum(p * np.log(np.clip(p, eps, None)), axis=1)
        out = {}
        if matrix_name in {"vv", "pp"}:
            col_sum = np.sum(np.clip(m, 0.0, None), axis=0)
            out["sink_like_node"] = int(np.argmax(col_sum))
            out["focused_node"] = int(np.argmin(row_ent))
            out["diffuse_node"] = int(np.argmax(row_ent))
        elif matrix_name in {"pv", "ps", "sp"}:
            out["focused_prompt"] = int(np.argmin(row_ent))
            out["diffuse_prompt"] = int(np.argmax(row_ent))
        elif matrix_name in {"vs", "sv"}:
            out["confident_patch"] = int(np.argmin(row_ent))
            out["ambiguous_patch"] = int(np.argmax(row_ent))
        return out

    def _maybe_export_affinity_keynodes(self, split: str, epoch: int, sample_idx: int, affinities: list, base: str):
        if (not self.affinity_keynode_viz_enable) or (str(split).lower() not in set(self.affinity_keynode_splits)):
            return
        if self._affinity_keynode_saved >= self.affinity_keynode_max_figs:
            return
        if (not isinstance(affinities, list)) or len(affinities) == 0:
            return

        key_layers = self._select_key_layers(len(affinities))
        matrix_map = [
            ("vv", "Avv"),
            ("pp", "App"),
            ("pv", "Apv"),
            ("vs", "Avs"),
            ("ps", "Aps"),
            ("sv", "Asv"),
            ("sp", "Asp"),
        ]
        rows = []
        fig_rows = len(key_layers)
        fig_cols = len(matrix_map)
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3.0 * fig_cols, 2.5 * max(1, fig_rows)), squeeze=False)

        for r_i, li in enumerate(key_layers):
            aff = affinities[li] if (0 <= li < len(affinities) and isinstance(affinities[li], dict)) else {}
            for c_i, (mname, raw_key) in enumerate(matrix_map):
                ax = axes[r_i][c_i]
                x = aff.get(raw_key, None) if isinstance(aff, dict) else None
                if not torch.is_tensor(x):
                    ax.set_axis_off()
                    ax.set_title(f"L{li} {mname} (missing)")
                    rows.append({
                        "epoch": int(epoch), "split": str(split), "sample_idx": int(sample_idx),
                        "layer": int(li), "matrix_name": mname, "exists_flag": 0,
                    })
                    continue
                a = self._matrix_head_mean(x)
                if (not torch.is_tensor(a)) or a.dim() != 3 or a.shape[0] <= 0:
                    ax.set_axis_off()
                    ax.set_title(f"L{li} {mname} (invalid)")
                    continue
                m = a[0].detach().cpu().numpy()
                knd = self._key_nodes_for_matrix(m, mname)
                im = ax.imshow(m, aspect="auto", cmap="viridis")
                ax.set_title(f"L{li} {mname}")
                ax.set_xticks([])
                ax.set_yticks([])
                # draw key rows when row-like node is selected
                for kk in ["focused_node", "diffuse_node", "focused_prompt", "diffuse_prompt", "confident_patch", "ambiguous_patch"]:
                    if kk in knd:
                        y = int(knd[kk])
                        ax.axhline(y=y, color="w", linewidth=0.5, alpha=0.8)
                rows.append({
                    "epoch": int(epoch), "split": str(split), "sample_idx": int(sample_idx),
                    "layer": int(li), "matrix_name": mname, "exists_flag": 1,
                    **knd,
                })

        plt.tight_layout()
        fig_path = os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_affinity_keynodes.png")
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        save_json(
            os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_affinity_keynodes.json"),
            {"rows": rows, "layer_policy": self.affinity_keynode_layer_policy},
        )
        self._affinity_keynode_saved += 1

    def _extract_alignment_aux(self, affinities):
        """
        ?forward_with_affinity 鐨勪翰鍜屽垪琛ㄤ腑鎻愬彇瀵归綈鎹熷け闇€瑕佺殑 attn_pv / attn_vs?        鍏煎澶氬眰锛氭瀯?{layer_idx: tensor} 鐨勫瓧鍏革紝缂哄け鏃惰繑?None銆傦紙妯″瀷鍓嶅悜杩斿洖鎵€鏈夊眰鐨勪翰鍜岀煩闃碉紝鏂逛究?loss 渚х伒娲婚€夋嫨浣跨敤鍝竴灞傛垨澶氬眰銆傦級
        """
        if affinities is None:
            return None

        attn_pv = {}
        attn_vs = {}
        attn_ps = {}
        attn_sp = {}
        attn_sv = {}

        raw_apv_shape = None
        raw_asv_shape = None
        raw_asp_shape = None

        for idx, affinity in enumerate(affinities):
            if not isinstance(affinity, dict):
                continue

            apv = affinity.get("Apv")
            if apv is not None:
                if raw_apv_shape is None and torch.is_tensor(apv):
                    raw_apv_shape = tuple(apv.shape)
                if apv.dim() == 4:
                    attn_pv[idx] = apv.mean(dim=1)
                elif apv.dim() == 3:
                    attn_pv[idx] = apv

            avs = affinity.get("Avs")
            if avs is not None:
                if raw_asv_shape is None and torch.is_tensor(avs):
                    raw_asv_shape = tuple(avs.shape)
                if avs.dim() == 4:
                    # [B,1,N,M] -> [B,N,M]
                    attn_vs[idx] = avs.mean(dim=1)
                elif avs.dim() == 3:
                    attn_vs[idx] = avs
            else:
                asv = affinity.get("Asv")
                if asv is not None:
                    if raw_asv_shape is None and torch.is_tensor(asv):
                        raw_asv_shape = tuple(asv.shape)
                    if asv.dim() == 4:
                        attn_vs[idx] = asv.mean(dim=1).transpose(1, 2).contiguous()
                    elif asv.dim() == 3:
                        attn_vs[idx] = asv.transpose(1, 2).contiguous()

            aps = affinity.get("Aps")
            if aps is not None:
                if raw_asp_shape is None and torch.is_tensor(aps):
                    raw_asp_shape = tuple(aps.shape)
                if aps.dim() == 4:
                    # [B,1,P,M] -> [B,P,M]
                    attn_ps[idx] = aps.mean(dim=1)
                elif aps.dim() == 3:
                    attn_ps[idx] = aps
            else:
                asp = affinity.get("Asp")
                if asp is not None:
                    if raw_asp_shape is None and torch.is_tensor(asp):
                        raw_asp_shape = tuple(asp.shape)
                    if asp.dim() == 4:
                        attn_sp[idx] = asp.mean(dim=1)
                    elif asp.dim() == 3:
                        attn_sp[idx] = asp

            asv = affinity.get("Asv")
            if asv is not None:
                if asv.dim() == 4:
                    attn_sv[idx] = asv.mean(dim=1)
                elif asv.dim() == 3:
                    attn_sv[idx] = asv

        if not attn_pv and not attn_vs and not attn_ps and not attn_sp and not attn_sv:
            return None

        out = {}
        if attn_pv:
            out["attn_pv"] = attn_pv
        if attn_vs:
            out["attn_vs"] = attn_vs
        if attn_ps:
            out["attn_ps"] = attn_ps
        if attn_sp:
            out["attn_sp"] = attn_sp
        if attn_sv:
            out["attn_sv"] = attn_sv

        if self.debug_shapes and (not self._shape_debug_aux_logged):
            sample_layer = sorted(attn_pv.keys())[0] if len(attn_pv) > 0 else None
            apv_after = tuple(attn_pv[sample_layer].shape) if sample_layer is not None else None
            avs_after = tuple(attn_vs[sample_layer].shape) if sample_layer is not None and sample_layer in attn_vs else None
            aps_after = tuple(attn_ps[sample_layer].shape) if sample_layer is not None and sample_layer in attn_ps else None
            asp_after = tuple(attn_sp[sample_layer].shape) if sample_layer is not None and sample_layer in attn_sp else None
            asv_after = tuple(attn_sv[sample_layer].shape) if sample_layer is not None and sample_layer in attn_sv else None
            print(
                "[SHAPE-DEBUG] trainer._extract_alignment_aux affinity_raw Apv={} Asv={} Asp={} "
                "head_avg Apv={} Avs->attn_vs={} Aps->attn_ps={} Asp->attn_sp={} Asv->attn_sv={} layer={}".format(
                    raw_apv_shape,
                    raw_asv_shape,
                    raw_asp_shape,
                    apv_after,
                    avs_after,
                    aps_after,
                    asp_after,
                    asv_after,
                    sample_layer,
                )
            )
            self._shape_debug_aux_logged = True

        # Role-migration diagnostics (lightweight scalar summaries).
        role_cfg = bool(self.cfg.MODEL.ROLE_MIGRATION.ENABLE)
        if bool(role_cfg):
            def _energy(x):
                return float(x.float().abs().mean().item())
            def _entropy(x):
                p = x.float().clamp_min(1e-8)
                return float((-(p * p.log()).sum(dim=-1).mean()).item())

            layer_ids = sorted(set(attn_pv.keys()) | set(attn_vs.keys()) | set(attn_ps.keys()))
            early_end = int(self.cfg.MODEL.ROLE_MIGRATION.EARLY_END)
            late_start = int(self.cfg.MODEL.ROLE_MIGRATION.LATE_START)
            early_layers = [l for l in layer_ids if l <= early_end]
            late_layers = [l for l in layer_ids if l >= late_start]
            if len(late_layers) == 0 and len(layer_ids) > 0:
                late_layers = [layer_ids[-1]]

            def _mean(vals):
                return float(sum(vals) / max(1, len(vals))) if len(vals) > 0 else None

            out["role_summary"] = {
                "layer_ids": layer_ids,
                "early_layers": early_layers,
                "late_layers": late_layers,
                "early_apv_energy": _mean([_energy(attn_pv[l]) for l in early_layers if l in attn_pv]),
                "early_aps_energy": _mean([_energy(attn_ps[l]) for l in early_layers if l in attn_ps]),
                "early_avs_energy": _mean([_energy(attn_vs[l]) for l in early_layers if l in attn_vs]),
                "late_apv_energy": _mean([_energy(attn_pv[l]) for l in late_layers if l in attn_pv]),
                "late_aps_energy": _mean([_energy(attn_ps[l]) for l in late_layers if l in attn_ps]),
                "late_avs_energy": _mean([_energy(attn_vs[l]) for l in late_layers if l in attn_vs]),
                "late_apv_entropy": _mean([_entropy(attn_pv[l]) for l in late_layers if l in attn_pv]),
            }
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
        """
        self._vis_trend_buf 会在一个 epoch 内累计：
        - Aps/Avs entropy
        - token specialization
        - token_pairwise_cos
        - gt_hn_gap
        - token norm 统计
        - gini / monopoly
        - affinity_rows 等
        """
        self._vis_processed = 0
        self._vis_trend_buf = {
            "split": str(split),
            "epoch": int(self._trace_epoch + 1),
            "layers": {},
            "token_specialization": [],
            "token_pairwise_cos": [],
            "gt_hn_gap": [],
            "token_norm_mean": [],
            "token_norm_std": [],
            "token_norm_max": [],
            "token_norm_p95": [],
            "token_norm_p99": [],
            "token_norm_outlier_ratio": [],
            "token_usage_gini": [],
            "token_monopoly_index": [],
            "anchor_usage_gini": [],
            "free_usage_gini": [],
            "anchor_monopoly": [],
            "free_monopoly": [],
            "affinity_rows": [],
        }
        self._affinity_keynode_saved = 0

    def _vis_update_trend(self, affinities, logits, targets, sem_tokens, visual_tokens=None, sem_state=None):
        """累计：
        1. 每层 Aps / Avs 的 entropy
        2. token specialization
        3. token usage 的 gini / monopoly
        4. anchor/free token 的 usage concentration
        5. affinity summary rows
        6. token-patch diagnostics
        7. visual token norm 分布
        8. semantic token pairwise cosine
        9. GT vs hardest-negative 的 margin gap
"""
        if not self.vis_trends:
            return
        if not isinstance(self._vis_trend_buf, dict):
            return
        if isinstance(affinities, list):
            for li, aff in enumerate(affinities):
                if not isinstance(aff, dict):
                    continue
                aps = aff.get("Aps")
                avs = aff.get("Avs")
                if (not torch.is_tensor(aps)) and torch.is_tensor(aff.get("Asp")):
                    aps = aff.get("Asp").transpose(-1, -2).contiguous()
                if (not torch.is_tensor(avs)) and torch.is_tensor(aff.get("Asv")):
                    avs = aff.get("Asv").transpose(-1, -2).contiguous()
                if li not in self._vis_trend_buf["layers"]:
                    self._vis_trend_buf["layers"][li] = {
                        "aps_entropy": [],
                        "avs_entropy": [],
                        "patchdist_entropy_mean": [],
                        "patchdist_entropy_std": [],
                        "patchdist_entropy_min": [],
                        "patchdist_entropy_max": [],
                        "patchdist_gini_mean": [],
                        "patchdist_gini_std": [],
                        "patchdist_gini_min": [],
                        "patchdist_gini_max": [],
                        "patchdist_monopoly_mean": [],
                        "patchdist_monopoly_std": [],
                        "patchdist_monopoly_min": [],
                        "patchdist_monopoly_max": [],
                        "patchdist_overlap_cos_mean": [],
                        "patchdist_overlap_cos_max": [],
                        "patchdist_overlap_iou_mean": [],
                        "patchdist_overlap_iou_max": [],
                        "patchdist_effective_rank_mean": [],
                        "patchdist_sv_top1_cum_mean": [],
                        "patchdist_sv_top2_cum_mean": [],
                        "patchdist_sv_top3_cum_mean": [],
                        "patchdist_sv_mean": [],
                    }
                if torch.is_tensor(aps) and aps.numel() > 0:
                    self._vis_trend_buf["layers"][li]["aps_entropy"].append(self._entropy_np(aps.detach().cpu()))
                if torch.is_tensor(avs) and avs.numel() > 0:
                    self._vis_trend_buf["layers"][li]["avs_entropy"].append(self._entropy_np(avs.detach().cpu()))
                    avs2 = avs.detach().float()
                    if avs2.dim() == 4:
                        avs2 = avs2.mean(dim=1)  # [B,N,M]
                    if avs2.dim() == 3 and avs2.numel() > 0:
                        asv2 = avs2.transpose(1, 2).contiguous()  # [B,M,N]
                        spec = float(asv2.max(dim=-1).values.mean().item())
                        self._vis_trend_buf["token_specialization"].append(spec)
                        usage = asv2.sum(dim=-1)  # [B, M]
                        usage_np = usage.detach().cpu().numpy()
                        if usage_np.size > 0:
                            g_all = np.array([self._gini_np(u) for u in usage_np], dtype=np.float64)
                            mono_all = usage_np.max(axis=1) / np.clip(usage_np.sum(axis=1), 1e-12, None)
                            g_all = g_all[np.isfinite(g_all)]
                            mono_all = mono_all[np.isfinite(mono_all)]
                            if g_all.size > 0:
                                self._vis_trend_buf["token_usage_gini"].append(float(np.mean(g_all)))
                            if mono_all.size > 0:
                                self._vis_trend_buf["token_monopoly_index"].append(float(np.mean(mono_all)))
                            if isinstance(sem_state, dict):
                                a_tok = sem_state.get("anchor_tokens")
                                f_tok = sem_state.get("free_tokens")
                                a_n = int(a_tok.shape[1]) if torch.is_tensor(a_tok) and a_tok.dim() == 3 else 0
                                f_n = int(f_tok.shape[1]) if torch.is_tensor(f_tok) and f_tok.dim() == 3 else 0
                                m_tot = int(usage_np.shape[1])
                                if a_n > 0 and (a_n + f_n) <= m_tot:
                                    a_usage = usage_np[:, :a_n]
                                    a_g = np.array([self._gini_np(u) for u in a_usage], dtype=np.float64)
                                    a_m = a_usage.max(axis=1) / np.clip(a_usage.sum(axis=1), 1e-12, None)
                                    a_g = a_g[np.isfinite(a_g)]
                                    a_m = a_m[np.isfinite(a_m)]
                                    if a_g.size > 0:
                                        self._vis_trend_buf["anchor_usage_gini"].append(float(np.mean(a_g)))
                                    if a_m.size > 0:
                                        self._vis_trend_buf["anchor_monopoly"].append(float(np.mean(a_m)))
                                if f_n > 0 and (a_n + f_n) <= m_tot:
                                    f_usage = usage_np[:, a_n:a_n + f_n]
                                    f_g = np.array([self._gini_np(u) for u in f_usage], dtype=np.float64)
                                    f_m = f_usage.max(axis=1) / np.clip(f_usage.sum(axis=1), 1e-12, None)
                                    f_g = f_g[np.isfinite(f_g)]
                                    f_m = f_m[np.isfinite(f_m)]
                                    if f_g.size > 0:
                                        self._vis_trend_buf["free_usage_gini"].append(float(np.mean(f_g)))
                                    if f_m.size > 0:
                                        self._vis_trend_buf["free_monopoly"].append(float(np.mean(f_m)))
                if self.affinity_summary_enable:
                    meta = self._matrix_sources_meta()
                    for mname, mmeta in meta.items():
                        raw_key = mmeta["raw_key"]
                        x = aff.get(raw_key) if isinstance(aff, dict) else None
                        if torch.is_tensor(x):
                            a = self._matrix_head_mean(x)
                            st = self._matrix_stats_from_bqk(a) if torch.is_tensor(a) else {}
                            row = {
                                "epoch": int(self._trace_epoch + 1),
                                "split": str(self._vis_trend_buf.get("split", "unknown")),
                                "layer": int(li),
                                "matrix_name": str(mname),
                                "exists_flag": 1,
                                "source_path_or_source_name": mmeta["source_path_or_source_name"],
                                "q_len": int(st.get("q_len", a.shape[1] if torch.is_tensor(a) else -1)),
                                "k_len": int(st.get("k_len", a.shape[2] if torch.is_tensor(a) else -1)),
                                "softmax_dim_name": mmeta["softmax_dim_name"],
                            }
                            for k in [
                                "n_samples", "mean", "std", "min", "max",
                                "row_entropy_mean", "row_entropy_p50", "row_entropy_p90",
                                "row_gini_mean", "row_gini_p50", "row_gini_p90",
                                "row_monopoly_mean", "row_monopoly_p50", "row_monopoly_p90",
                                "effective_rank_mean", "sv_top1_cum_mean", "sv_top2_cum_mean", "sv_top3_cum_mean",
                            ]:
                                row[k] = st.get(k, None)
                            self._vis_trend_buf["affinity_rows"].append(row)
                        else:
                            self._vis_trend_buf["affinity_rows"].append({
                                "epoch": int(self._trace_epoch + 1),
                                "split": str(self._vis_trend_buf.get("split", "unknown")),
                                "layer": int(li),
                                "matrix_name": str(mname),
                                "exists_flag": 0,
                                "source_path_or_source_name": mmeta["source_path_or_source_name"],
                                "q_len": None,
                                "k_len": None,
                                "softmax_dim_name": mmeta["softmax_dim_name"],
                                "n_samples": 0,
                                "mean": None,
                                "std": None,
                                "min": None,
                                "max": None,
                                "row_entropy_mean": None,
                                "row_entropy_p50": None,
                                "row_entropy_p90": None,
                                "row_gini_mean": None,
                                "row_gini_p50": None,
                                "row_gini_p90": None,
                                "row_monopoly_mean": None,
                                "row_monopoly_p50": None,
                                "row_monopoly_p90": None,
                                "effective_rank_mean": None,
                                "sv_top1_cum_mean": None,
                                "sv_top2_cum_mean": None,
                                "sv_top3_cum_mean": None,
                            })
                if self.token_patch_stats_enable:
                    a_t_p = self._token_patch_from_affinity(aff)
                    stats = self._token_patch_batch_stats(a_t_p) if torch.is_tensor(a_t_p) else {}
                    if isinstance(stats, dict) and len(stats) > 0:
                        for k, v in stats.items():
                            if k == "patchdist_sv_mean":
                                if isinstance(v, list) and len(v) > 0:
                                    self._vis_trend_buf["layers"][li]["patchdist_sv_mean"].append(v)
                            elif k in self._vis_trend_buf["layers"][li] and isinstance(v, (float, int)) and np.isfinite(v):
                                self._vis_trend_buf["layers"][li][k].append(float(v))
        if torch.is_tensor(visual_tokens) and visual_tokens.dim() == 3 and visual_tokens.numel() > 0:
            nrm = visual_tokens.detach().float().norm(dim=-1).reshape(-1)
            if nrm.numel() > 0:
                nrm_np = nrm.cpu().numpy()
                mu = float(np.mean(nrm_np))
                sd = float(np.std(nrm_np))
                self._vis_trend_buf["token_norm_mean"].append(mu)
                self._vis_trend_buf["token_norm_std"].append(sd)
                self._vis_trend_buf["token_norm_max"].append(float(np.max(nrm_np)))
                self._vis_trend_buf["token_norm_p95"].append(float(np.percentile(nrm_np, 95)))
                self._vis_trend_buf["token_norm_p99"].append(float(np.percentile(nrm_np, 99)))
                thr = mu + 2.0 * sd
                self._vis_trend_buf["token_norm_outlier_ratio"].append(float(np.mean(nrm_np > thr)))
        if torch.is_tensor(sem_tokens) and sem_tokens.dim() == 3 and sem_tokens.shape[1] >= 2:
            t = torch.nn.functional.normalize(sem_tokens.detach().float(), dim=-1)
            sim = torch.einsum("bmd,bnd->bmn", t, t)
            b, m, _ = sim.shape
            mask = ~torch.eye(m, device=sim.device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
            if mask.any():
                self._vis_trend_buf["token_pairwise_cos"].append(float(sim[mask].mean().item()))
        if torch.is_tensor(logits) and torch.is_tensor(targets):
            y = targets.to(logits.device, dtype=torch.long)
            valid = (y >= 0) & (y < logits.shape[1])
            if valid.any():
                ridx = torch.arange(logits.shape[0], device=logits.device)[valid]
                ly = logits[ridx, y[valid]]
                hn = logits[ridx].clone()
                hn.scatter_(1, y[valid].view(-1, 1), -1e9)
                lhn = hn.max(dim=1).values
                self._vis_trend_buf["gt_hn_gap"].append(float((ly - lhn).mean().item()))

    def _vis_export_trend(self, split: str):
        if not self.vis_trends or (not isinstance(self._vis_trend_buf, dict)):
            return
        epoch = int(self._trace_epoch + 1)
        split = str(split).lower()
        trend_dir = os.path.join(self.vis_dir, "trends", split)
        ensure_dir(trend_dir)
        layers = self._vis_trend_buf.get("layers", {})
        layer_ids = sorted(list(layers.keys()))
        avs_curve = []
        aps_curve = []
        for li in layer_ids:
            aps_vals = [x for x in layers[li]["aps_entropy"] if np.isfinite(x)]
            avs_vals = [x for x in layers[li]["avs_entropy"] if np.isfinite(x)]
            aps_curve.append(float(np.mean(aps_vals)) if len(aps_vals) > 0 else float("nan"))
            avs_curve.append(float(np.mean(avs_vals)) if len(avs_vals) > 0 else float("nan"))

        out_json = {
            "split": split,
            "epoch": epoch,
            "layers": layer_ids,
            "aps_entropy_curve": aps_curve,
            "avs_entropy_curve": avs_curve,
            "token_specialization": float(np.mean(self._vis_trend_buf["token_specialization"])) if len(self._vis_trend_buf["token_specialization"]) > 0 else None,
            "token_pairwise_cos": float(np.mean(self._vis_trend_buf["token_pairwise_cos"])) if len(self._vis_trend_buf["token_pairwise_cos"]) > 0 else None,
            "gt_hn_gap": float(np.mean(self._vis_trend_buf["gt_hn_gap"])) if len(self._vis_trend_buf["gt_hn_gap"]) > 0 else None,
            "token_norm_mean": float(np.mean(self._vis_trend_buf["token_norm_mean"])) if len(self._vis_trend_buf["token_norm_mean"]) > 0 else None,
            "token_norm_std": float(np.mean(self._vis_trend_buf["token_norm_std"])) if len(self._vis_trend_buf["token_norm_std"]) > 0 else None,
            "token_norm_max": float(np.mean(self._vis_trend_buf["token_norm_max"])) if len(self._vis_trend_buf["token_norm_max"]) > 0 else None,
            "token_norm_p95": float(np.mean(self._vis_trend_buf["token_norm_p95"])) if len(self._vis_trend_buf["token_norm_p95"]) > 0 else None,
            "token_norm_p99": float(np.mean(self._vis_trend_buf["token_norm_p99"])) if len(self._vis_trend_buf["token_norm_p99"]) > 0 else None,
            "token_norm_outlier_ratio": float(np.mean(self._vis_trend_buf["token_norm_outlier_ratio"])) if len(self._vis_trend_buf["token_norm_outlier_ratio"]) > 0 else None,
            "token_usage_gini": float(np.mean(self._vis_trend_buf["token_usage_gini"])) if len(self._vis_trend_buf["token_usage_gini"]) > 0 else None,
            "token_monopoly_index": float(np.mean(self._vis_trend_buf["token_monopoly_index"])) if len(self._vis_trend_buf["token_monopoly_index"]) > 0 else None,
            "anchor_usage_gini": float(np.mean(self._vis_trend_buf["anchor_usage_gini"])) if len(self._vis_trend_buf["anchor_usage_gini"]) > 0 else None,
            "free_usage_gini": float(np.mean(self._vis_trend_buf["free_usage_gini"])) if len(self._vis_trend_buf["free_usage_gini"]) > 0 else None,
            "anchor_monopoly": float(np.mean(self._vis_trend_buf["anchor_monopoly"])) if len(self._vis_trend_buf["anchor_monopoly"]) > 0 else None,
            "free_monopoly": float(np.mean(self._vis_trend_buf["free_monopoly"])) if len(self._vis_trend_buf["free_monopoly"]) > 0 else None,
        }
        save_json(os.path.join(trend_dir, f"epoch_{epoch:03d}_trend.json"), out_json)

        # Affinity matrix summaries for vv/pp/pv/vs/ps.
        if self.affinity_summary_enable:
            rows = self._vis_trend_buf.get("affinity_rows", [])
            grouped = {}
            for r in rows:
                key = (r.get("epoch"), r.get("split"), r.get("layer"), r.get("matrix_name"))
                grouped.setdefault(key, []).append(r)
            agg_rows = []
            numeric_keys = [
                "n_samples", "mean", "std", "min", "max",
                "row_entropy_mean", "row_entropy_p50", "row_entropy_p90",
                "row_gini_mean", "row_gini_p50", "row_gini_p90",
                "row_monopoly_mean", "row_monopoly_p50", "row_monopoly_p90",
                "effective_rank_mean", "sv_top1_cum_mean", "sv_top2_cum_mean", "sv_top3_cum_mean",
            ]
            for key, items in grouped.items():
                base = {
                    "epoch": int(key[0]),
                    "split": str(key[1]),
                    "layer": int(key[2]),
                    "matrix_name": str(key[3]),
                    "exists_flag": int(max(int(x.get("exists_flag", 0)) for x in items)),
                    "source_path_or_source_name": str(items[0].get("source_path_or_source_name", "")),
                    "q_len": items[0].get("q_len", None),
                    "k_len": items[0].get("k_len", None),
                    "softmax_dim_name": str(items[0].get("softmax_dim_name", "")),
                }
                for nk in numeric_keys:
                    vals = [x.get(nk, None) for x in items]
                    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                    if nk == "n_samples":
                        base[nk] = int(np.sum(vals)) if len(vals) > 0 else 0
                    else:
                        base[nk] = float(np.mean(vals)) if len(vals) > 0 else None
                agg_rows.append(base)

            agg_rows = sorted(agg_rows, key=lambda x: (x["layer"], x["matrix_name"]))
            save_json(
                os.path.join(trend_dir, f"epoch_{epoch:03d}_affinity_epoch_summary.json"),
                {"rows": agg_rows},
            )
            save_json(
                os.path.join(trend_dir, "affinity_epoch_summary.json"),
                {"epoch": int(epoch), "split": str(split), "rows": agg_rows},
            )
            field_order = [
                "epoch", "split", "layer", "matrix_name", "exists_flag",
                "source_path_or_source_name", "q_len", "k_len", "softmax_dim_name",
                "n_samples", "mean", "std", "min", "max",
                "row_entropy_mean", "row_entropy_p50", "row_entropy_p90",
                "row_gini_mean", "row_gini_p50", "row_gini_p90",
                "row_monopoly_mean", "row_monopoly_p50", "row_monopoly_p90",
                "effective_rank_mean", "sv_top1_cum_mean", "sv_top2_cum_mean", "sv_top3_cum_mean",
            ]
            for row in agg_rows:
                append_csv_row(os.path.join(trend_dir, "affinity_epoch_summary.csv"), row, field_order=field_order)
                append_csv_row(os.path.join(trend_dir, "affinity_trend_summary.csv"), row, field_order=field_order)

        # token-patch per-layer summaries (A[T,P]-based diagnostics)
        if self.token_patch_stats_enable and len(layer_ids) > 0:
            tp_rows = []
            sv_rows = []
            for li in layer_ids:
                lbuf = layers.get(li, {})
                row = {"epoch": epoch, "split": split, "layer": int(li)}
                metric_keys = [
                    "patchdist_entropy_mean", "patchdist_entropy_std", "patchdist_entropy_min", "patchdist_entropy_max",
                    "patchdist_gini_mean", "patchdist_gini_std", "patchdist_gini_min", "patchdist_gini_max",
                    "patchdist_monopoly_mean", "patchdist_monopoly_std", "patchdist_monopoly_min", "patchdist_monopoly_max",
                    "patchdist_overlap_cos_mean", "patchdist_overlap_cos_max",
                    "patchdist_overlap_iou_mean", "patchdist_overlap_iou_max",
                    "patchdist_effective_rank_mean",
                    "patchdist_sv_top1_cum_mean", "patchdist_sv_top2_cum_mean", "patchdist_sv_top3_cum_mean",
                ]
                for mk in metric_keys:
                    vals = [x for x in lbuf.get(mk, []) if np.isfinite(x)]
                    row[mk] = float(np.mean(vals)) if len(vals) > 0 else None

                sv_entries = lbuf.get("patchdist_sv_mean", [])
                if isinstance(sv_entries, list) and len(sv_entries) > 0:
                    max_r = max(len(v) for v in sv_entries if isinstance(v, list))
                    if max_r > 0:
                        sv_mat = np.full((len(sv_entries), max_r), np.nan, dtype=np.float64)
                        for si, vec in enumerate(sv_entries):
                            if not isinstance(vec, list):
                                continue
                            n = min(len(vec), max_r)
                            sv_mat[si, :n] = np.asarray(vec[:n], dtype=np.float64)
                        sv_mean = np.nanmean(sv_mat, axis=0)
                        row["patchdist_sv_mean"] = [float(x) for x in sv_mean if np.isfinite(x)]
                        ssum = float(np.nansum(sv_mean))
                        if ssum > 0:
                            c = np.cumsum(np.nan_to_num(sv_mean, nan=0.0)) / ssum
                            row["patchdist_sv_top1_cum_mean"] = float(c[0]) if c.size > 0 else row.get("patchdist_sv_top1_cum_mean")
                            row["patchdist_sv_top2_cum_mean"] = float(c[min(1, c.size - 1)]) if c.size > 0 else row.get("patchdist_sv_top2_cum_mean")
                            row["patchdist_sv_top3_cum_mean"] = float(c[min(2, c.size - 1)]) if c.size > 0 else row.get("patchdist_sv_top3_cum_mean")
                        for r_idx, sv in enumerate(sv_mean, start=1):
                            if np.isfinite(sv):
                                sv_rows.append({
                                    "epoch": epoch, "split": split, "layer": int(li),
                                    "sv_idx": int(r_idx), "sv_mean": float(sv)
                                })
                tp_rows.append(row)

            save_json(os.path.join(trend_dir, f"epoch_{epoch:03d}_token_patch_layer_summary.json"), {"rows": tp_rows})
            tp_field_order = [
                "epoch", "split", "layer",
                "patchdist_entropy_mean", "patchdist_entropy_std", "patchdist_entropy_min", "patchdist_entropy_max",
                "patchdist_gini_mean", "patchdist_gini_std", "patchdist_gini_min", "patchdist_gini_max",
                "patchdist_monopoly_mean", "patchdist_monopoly_std", "patchdist_monopoly_min", "patchdist_monopoly_max",
                "patchdist_overlap_cos_mean", "patchdist_overlap_cos_max",
                "patchdist_overlap_iou_mean", "patchdist_overlap_iou_max",
                "patchdist_effective_rank_mean",
                "patchdist_sv_top1_cum_mean", "patchdist_sv_top2_cum_mean", "patchdist_sv_top3_cum_mean",
            ]
            for row in tp_rows:
                append_csv_row(
                    os.path.join(trend_dir, "token_patch_layer_summary.csv"),
                    row,
                    field_order=tp_field_order,
                )
            if len(sv_rows) > 0:
                for row in sv_rows:
                    append_csv_row(
                        os.path.join(trend_dir, "token_patch_sv_spectrum.csv"),
                        row,
                        field_order=["epoch", "split", "layer", "sv_idx", "sv_mean"],
                    )
        append_csv_row(
            os.path.join(trend_dir, "trend_summary.csv"),
            {
                "epoch": epoch,
                "split": split,
                "token_specialization": out_json["token_specialization"],
                "token_pairwise_cos": out_json["token_pairwise_cos"],
                "gt_hn_gap": out_json["gt_hn_gap"],
                "avs_entropy_mean": float(np.nanmean(avs_curve)) if len(avs_curve) > 0 else None,
                "aps_entropy_mean": float(np.nanmean(aps_curve)) if len(aps_curve) > 0 else None,
                "token_norm_mean": out_json["token_norm_mean"],
                "token_norm_std": out_json["token_norm_std"],
                "token_norm_max": out_json["token_norm_max"],
                "token_norm_p95": out_json["token_norm_p95"],
                "token_norm_p99": out_json["token_norm_p99"],
                "token_norm_outlier_ratio": out_json["token_norm_outlier_ratio"],
                "token_usage_gini": out_json["token_usage_gini"],
                "token_monopoly_index": out_json["token_monopoly_index"],
                "anchor_usage_gini": out_json["anchor_usage_gini"],
                "free_usage_gini": out_json["free_usage_gini"],
                "anchor_monopoly": out_json["anchor_monopoly"],
                "free_monopoly": out_json["free_monopoly"],
                "patchdist_entropy_mean_last": float(np.mean([x for x in layers.get(layer_ids[-1], {}).get("patchdist_entropy_mean", []) if np.isfinite(x)])) if self.token_patch_stats_enable and len(layer_ids) > 0 else None,
                "patchdist_gini_mean_last": float(np.mean([x for x in layers.get(layer_ids[-1], {}).get("patchdist_gini_mean", []) if np.isfinite(x)])) if self.token_patch_stats_enable and len(layer_ids) > 0 else None,
                "patchdist_monopoly_mean_last": float(np.mean([x for x in layers.get(layer_ids[-1], {}).get("patchdist_monopoly_mean", []) if np.isfinite(x)])) if self.token_patch_stats_enable and len(layer_ids) > 0 else None,
                "patchdist_effective_rank_mean_last": float(np.mean([x for x in layers.get(layer_ids[-1], {}).get("patchdist_effective_rank_mean", []) if np.isfinite(x)])) if self.token_patch_stats_enable and len(layer_ids) > 0 else None,
            },
            field_order=[
                "epoch", "split", "token_specialization", "token_pairwise_cos",
                "gt_hn_gap", "avs_entropy_mean", "aps_entropy_mean",
                "token_norm_mean", "token_norm_std", "token_norm_max",
                "token_norm_p95", "token_norm_p99", "token_norm_outlier_ratio",
                "token_usage_gini", "token_monopoly_index",
                "anchor_usage_gini", "free_usage_gini",
                "anchor_monopoly", "free_monopoly",
                "patchdist_entropy_mean_last", "patchdist_gini_mean_last",
                "patchdist_monopoly_mean_last", "patchdist_effective_rank_mean_last",
            ],
        )
        logger.info(
            "[vis-trend] split=%s epoch=%d token_norm_mean=%.4f token_norm_p95=%.4f gini=%.4f monopoly=%.4f",
            split,
            epoch,
            float(out_json["token_norm_mean"]) if out_json["token_norm_mean"] is not None else float("nan"),
            float(out_json["token_norm_p95"]) if out_json["token_norm_p95"] is not None else float("nan"),
            float(out_json["token_usage_gini"]) if out_json["token_usage_gini"] is not None else float("nan"),
            float(out_json["token_monopoly_index"]) if out_json["token_monopoly_index"] is not None else float("nan"),
        )
        if self.token_patch_stats_enable and len(layer_ids) > 0:
            li_last = int(layer_ids[-1])
            lbuf = layers.get(li_last, {})
            pe = [x for x in lbuf.get("patchdist_entropy_mean", []) if np.isfinite(x)]
            pg = [x for x in lbuf.get("patchdist_gini_mean", []) if np.isfinite(x)]
            pm = [x for x in lbuf.get("patchdist_monopoly_mean", []) if np.isfinite(x)]
            logger.info(
                "[vis-patchdist] split=%s epoch=%d layer=%d entropy=%.4f gini=%.4f monopoly=%.4f",
                split,
                epoch,
                li_last,
                float(np.mean(pe)) if len(pe) > 0 else float("nan"),
                float(np.mean(pg)) if len(pg) > 0 else float("nan"),
                float(np.mean(pm)) if len(pm) > 0 else float("nan"),
            )
        if self.vis_save_images and len(layer_ids) > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(layer_ids, avs_curve, marker="o", label="Avs entropy")
            plt.plot(layer_ids, aps_curve, marker="o", label="Aps entropy")
            plt.xlabel("Layer")
            plt.ylabel("Entropy")
            plt.title(f"{split} epoch {epoch} entropy curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(trend_dir, f"epoch_{epoch:03d}_entropy_curve.png"), dpi=160)
            plt.close()

    def _vis_collect_sample(self, split: str, local_idx: int, sample_idx: int, image_chw: torch.Tensor, logits: torch.Tensor, target: int, global_class_ids, model_ref,):
        if self._vis_processed >= self.vis_max_samples:
            return
        r_head = getattr(model_ref, "r_similarity_head", None)
        if r_head is None:
            return
        affinities = getattr(r_head, "_runtime_affinities", None)
        token_seq = getattr(r_head, "_runtime_token_sequence", None)
        if not isinstance(affinities, list):
            return

        epoch = int(self._trace_epoch + 1)
        split = str(split).lower()
        base = os.path.join(self.vis_dir, split, f"epoch_{epoch:03d}")
        ensure_dir(base)
        img_u8 = to_uint8_image(image_chw)
        h, w = img_u8.shape[:2]

        self._maybe_export_affinity_keynodes(
            split=split,
            epoch=epoch,
            sample_idx=sample_idx,
            affinities=affinities,
            base=base,
        )

        # Group 1: token-patch diagnostics from A[T,P] (summary-first, raw dump optional).
        if self.vis_local_control:
            if self._vis_processed < int(self.token_patch_max_samples):
                for li, aff in enumerate(affinities):
                    a_b = self._token_patch_from_affinity(aff) if self.token_patch_stats_enable else None
                    if (not torch.is_tensor(a_b)) or a_b.dim() != 3 or local_idx >= a_b.shape[0]:
                        continue
                    a = a_b[local_idx].detach().cpu().numpy()  # [T,P]
                    t_num, p_num = int(a.shape[0]), int(a.shape[1])
                    g = self._infer_grid(p_num)
                    a_bar = a.mean(axis=0)  # [P]
                    residual = a - a_bar[None, :]  # [T,P]
                    mean_abs = np.mean(np.abs(residual), axis=1)
                    max_abs = np.max(np.abs(residual), axis=1)
                    l2 = np.sqrt(np.sum(residual * residual, axis=1))
                    # overlap
                    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
                    ov_cos = a_n @ a_n.T
                    k = max(1, int(round(float(self.token_patch_toprho) * float(p_num))))
                    topk_idx = np.argpartition(-a, kth=k - 1, axis=1)[:, :k]
                    mask = np.zeros_like(a, dtype=np.float32)
                    for ti in range(t_num):
                        mask[ti, topk_idx[ti]] = 1.0
                    inter = mask @ mask.T
                    msum = mask.sum(axis=1, keepdims=True)
                    union = msum + msum.T - inter
                    ov_iou = inter / np.clip(union, 1e-12, None)
                    # singular values
                    try:
                        sv = np.linalg.svd(a, full_matrices=False, compute_uv=False)
                    except Exception:
                        sv = np.zeros((min(t_num, p_num),), dtype=np.float64)
                    ssum = float(np.sum(sv))
                    if ssum > 0:
                        sv_ratio = sv / ssum
                        er = float(np.exp(-np.sum(sv_ratio * np.log(np.clip(sv_ratio, 1e-12, None)))))
                        cum = np.cumsum(sv_ratio)
                    else:
                        er = float("nan")
                        cum = np.zeros_like(sv)
                    stats_obj = {
                        "split": split,
                        "epoch": int(epoch),
                        "sample_idx": int(sample_idx),
                        "layer": int(li),
                        "source": self.token_patch_source,
                        "head_mode": self.token_patch_head_mode,
                        "toprho": float(self.token_patch_toprho),
                        "patchdist_entropy_mean": float(np.mean(-np.sum(a * np.log(np.clip(a, 1e-12, None)), axis=1))),
                        "patchdist_gini_mean": float(np.mean([self._gini_np(x) for x in a])) if a.shape[0] > 0 else None,
                        "patchdist_monopoly_mean": float(np.mean(np.max(a, axis=1))) if a.shape[0] > 0 else None,
                        "patchdist_effective_rank": er,
                        "patchdist_sv_top1_cum": float(cum[0]) if cum.size > 0 else None,
                        "patchdist_sv_top2_cum": float(cum[min(1, cum.size - 1)]) if cum.size > 0 else None,
                        "patchdist_sv_top3_cum": float(cum[min(2, cum.size - 1)]) if cum.size > 0 else None,
                        "residual_mean_abs": mean_abs.tolist(),
                        "residual_max_abs": max_abs.tolist(),
                        "residual_l2": l2.tolist(),
                    }
                    save_json(
                        os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_layer{li:02d}_tokenpatch_summary.json"),
                        stats_obj,
                    )
                    if self.affinity_save_raw_dump and self.vis_save_raw:
                        np.save(os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_layer{li:02d}_a_bar.npy"), a_bar)
                        np.save(os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_layer{li:02d}_residual.npy"), residual)
                        np.save(os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_layer{li:02d}_overlap_cos.npy"), ov_cos)
                        np.save(os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_layer{li:02d}_overlap_iou.npy"), ov_iou)
                        np.save(os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_layer{li:02d}_sv.npy"), sv)
                    # Only one total map overlay; no per-token small maps.
                    if self.token_patch_save_maps and self.vis_save_images:
                        bar_up = resize_map_torch(a_bar.reshape(g, g), (h, w))
                        save_overlay(
                            os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_layer{li:02d}_abar.png"),
                            img_u8,
                            bar_up,
                            title=f"a_bar layer{li}",
                        )

        # Group 2: rollout maps (CLS + prompt tokens).
        if self.vis_rollout and isinstance(self._last_attn_weights, list) and len(self._last_attn_weights) > 0:
            roll = attention_rollout(self._last_attn_weights)
            if torch.is_tensor(roll) and local_idx < roll.shape[0]:
                r = roll[local_idx].detach().cpu()
                p_len = int(self.cfg.MODEL.PROMPT.NUM_TOKENS)
                start_patch = 1 + p_len
                n_patch = int(max(0, r.shape[0] - start_patch))
                if n_patch > 0:
                    g = self._infer_grid(n_patch)
                    cls_map = r[0, start_patch:].view(g, g).numpy()
                    cls_up = resize_map_torch(cls_map, (h, w))
                    if self.vis_save_images:
                        save_overlay(
                            os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_rollout_cls.png"),
                            img_u8,
                            cls_up,
                            title="CLS rollout",
                        )
                    if p_len > 0:
                        shallow_idx = 1
                        deep_idx = 1 + max(0, p_len - 1)
                        for name, sid in [("shallow_prompt", shallow_idx), ("deep_prompt", deep_idx)]:
                            if sid < r.shape[0]:
                                pm = r[sid, start_patch:].view(g, g).numpy()
                                pm_up = resize_map_torch(pm, (h, w))
                                if self.vis_save_images:
                                    save_overlay(
                                        os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_rollout_{name}.png"),
                                        img_u8,
                                        pm_up,
                                        title=f"{name} rollout",
                                    )
                    if self.affinity_save_raw_dump and self.vis_save_raw:
                        np.savez_compressed(
                            os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_rollout_maps.npz"),
                            cls=cls_up,
                        )

        # Group 3: GT vs hardest-negative semantic comparison.
        if self.vis_gt_hn and torch.is_tensor(token_seq) and torch.is_tensor(logits):
            if token_seq.dim() == 3 and local_idx < token_seq.shape[0]:
                p_len = int(self.cfg.MODEL.PROMPT.NUM_TOKENS)
                patch_tokens = token_seq[local_idx:local_idx + 1, 1 + p_len:, :]
                if patch_tokens.numel() > 0:
                    v_patch = r_head.visual_proj(patch_tokens) if getattr(r_head, "visual_proj", None) is not None else patch_tokens
                    if bool(getattr(r_head, "use_cosine", True)):
                        v_patch = torch.nn.functional.normalize(v_patch, dim=-1)
                    sem = getattr(r_head, "_loss_last_semantic", None)
                    if torch.is_tensor(sem):
                        sem = sem.detach()
                        if bool(getattr(r_head, "use_cosine", True)):
                            sem = torch.nn.functional.normalize(sem, dim=-1)
                        score_patch_cls = torch.einsum("bnd,cd->bnc", v_patch, sem)[0]  # [N,C]
                        y = int(target)
                        if 0 <= y < logits.shape[1]:
                            l = logits[local_idx].detach().clone()
                            l[y] = -1e9
                            hn = int(l.argmax().item())
                            if global_class_ids is not None:
                                global_ids_t = torch.as_tensor(global_class_ids, device=score_patch_cls.device, dtype=torch.long)
                                y_global = int(global_ids_t[y].item())
                                hn_global = int(global_ids_t[hn].item())
                            else:
                                y_global = y
                                hn_global = hn
                            n_patch = int(score_patch_cls.shape[0])
                            g = self._infer_grid(n_patch)
                            gt_map = score_patch_cls[:, y].view(g, g).cpu().numpy()
                            hn_map = score_patch_cls[:, hn].view(g, g).cpu().numpy()
                            gt_up = resize_map_torch(gt_map, (h, w))
                            hn_up = resize_map_torch(hn_map, (h, w))
                            diff_up = gt_up - hn_up
                            if self.vis_save_images:
                                imgs = [
                                    overlay_heatmap(img_u8, gt_up),
                                    overlay_heatmap(img_u8, hn_up),
                                    overlay_heatmap(img_u8, diff_up),
                                ]
                                titles = [f"GT={y_global}", f"HN={hn_global}", "GT-HN"]
                                save_panel(
                                    os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_gt_hn_compare.png"),
                                    imgs, titles=titles, ncols=3
                                )
                            if self.vis_save_raw:
                                np.savez_compressed(
                                    os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_gt_hn_maps.npz"),
                                    gt=gt_up, hn=hn_up, diff=diff_up, gt_id=y_global, hn_id=hn_global
                                )

        self._vis_processed += 1

    ## 6. =================================monitor================================
    def _should_run_monitor(self, epoch: int) -> bool:
        if not self.monitor_enable:
            return False
        return ((epoch + 1) % self.monitor_every_epoch) == 0

    @torch.no_grad()
    def _collect_monitor_samples(self, data_loader, split: str):
        model_ref = self._model_ref(self.model)
        dataset = data_loader.dataset
        class_attr = dataset.class_attributes

        max_samples = int(self.monitor_max_samples)
        feats_all, labels_all = [], []
        raw_all, refined_all = [], []
        total = 0

        for input_data in data_loader:
            X, targets, attributes = self.get_input(input_data)
            X = X.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if attributes is not None:
                attributes = attributes.to(self.device, non_blocking=True)

            feats = model_ref.enc(X, semantics=attributes)
            feats = feats[:, 0] if torch.is_tensor(feats) and feats.dim() == 3 else feats
            if not torch.is_tensor(feats):
                break

            transformer = model_ref.enc.transformer
            raw_sem = getattr(transformer, "_monitor_last_raw_semantics", None)
            refined_sem = getattr(transformer, "_monitor_last_refined_semantics", None)
            if (raw_sem is None) and (attributes is not None):
                raw_sem = attributes

            bsz = int(feats.shape[0])
            remain = max_samples - total
            if remain <= 0:
                break
            keep = min(remain, bsz)

            feats_all.append(feats[:keep].detach().cpu())
            labels_all.append(targets[:keep].detach().cpu())
            if torch.is_tensor(raw_sem):
                raw_all.append(raw_sem[:keep].detach().cpu())
            if torch.is_tensor(refined_sem):
                refined_all.append(refined_sem[:keep].detach().cpu())

            total += keep
            if total >= max_samples:
                break

        if total == 0:
            return None

        feats = torch.cat(feats_all, dim=0).float().to(self.device)
        labels = torch.cat(labels_all, dim=0).long().to(self.device)
        raw_sem = torch.cat(raw_all, dim=0).float().to(self.device) if len(raw_all) == len(feats_all) and len(raw_all) > 0 else None
        refined_sem = torch.cat(refined_all, dim=0).float().to(self.device) if len(refined_all) == len(feats_all) and len(refined_all) > 0 else None

        return {
            "split": str(split),
            "dataset": dataset,
            "features": feats,
            "labels": labels,
            "raw_sem_batch": raw_sem,
            "refined_sem_batch": refined_sem,
            "num_samples": int(total),
        }

    @torch.no_grad()
    def _build_monitor_semantic_banks(self, dataset, candidate_ids: np.ndarray, r_head):
        class_attr = dataset.class_attributes
        class_attr = class_attr.to(self.device).float() if torch.is_tensor(class_attr) else torch.tensor(class_attr, device=self.device).float()
        cids = torch.as_tensor(candidate_ids, device=self.device, dtype=torch.long)
        raw_attr_cand = class_attr.index_select(0, cids)

        bank = {
            "candidate_ids": cids,
            "raw_attr": raw_attr_cand,
            "semantic_proj": None,
        }

        if r_head is not None:
            raw_embed = r_head.semantic_anchor(raw_attr_cand)
            bank["semantic_proj"] = r_head.semantic_proj(raw_embed)

        return bank

    @torch.no_grad()
    def _compute_monitor_metrics(self, sample_pack):
        split = sample_pack["split"]
        dataset = sample_pack["dataset"]
        feats = sample_pack["features"]
        labels = sample_pack["labels"]
        raw_sem_batch = sample_pack["raw_sem_batch"]
        refined_sem_batch = sample_pack["refined_sem_batch"]
        num_samples = int(sample_pack["num_samples"])

        model_ref = self._model_ref(self.model)
        r_head = getattr(model_ref, "r_similarity_head", None)
        if r_head is None:
            logger.warning("[monitor] split=%s skipped: r_similarity_head is missing", split)
            return None

        use_eval_space = str(split).lower() != "train"
        candidate_ids, candidate_map = self._dataset_space_meta(dataset, use_eval_space=use_eval_space)
        candidate_ids_np = np.asarray(candidate_ids, dtype=np.int64)
        if candidate_ids_np.size == 0:
            logger.warning("[monitor] split=%s skipped: empty candidate_ids", split)
            return None
        local_index = {int(cid): idx for idx, cid in enumerate(candidate_ids_np.tolist())}

        banks = self._build_monitor_semantic_banks(dataset, candidate_ids_np, r_head)
        if banks is None:
            return None

        cids = banks["candidate_ids"]
        if not torch.is_tensor(candidate_map):
            candidate_map = torch.as_tensor(candidate_map, dtype=torch.long)
        candidate_map = candidate_map.to(device=self.device, dtype=torch.long)
        y_local_all = candidate_map.index_select(0, labels.long())
        keep_mask = y_local_all >= 0
        if keep_mask.sum() == 0:
            logger.warning("[monitor] split=%s skipped: no labels in candidate_ids", split)
            return None

        feats = feats[keep_mask]
        labels = labels[keep_mask]
        y_local = y_local_all[keep_mask]
        if raw_sem_batch is not None:
            raw_sem_batch = raw_sem_batch[keep_mask]
        if refined_sem_batch is not None:
            refined_sem_batch = refined_sem_batch[keep_mask]

        v_proj = r_head.visual_proj(feats) if getattr(r_head, "visual_proj", None) is not None else feats
        v_norm = torch.nn.functional.normalize(v_proj.float(), dim=-1)

        metrics = {
            "epoch": int(self._trace_epoch + 1),
            "split": split,
            "num_samples": int(feats.shape[0]),
            "candidate_count": int(cids.numel()),
            "candidate_head": [int(x) for x in cids[:10].detach().cpu().tolist()],
        }

        # Layer 1: single-modality separability in visual space.
        uniq = torch.unique(labels)
        centers = []
        intra_vals = []
        for cls in uniq:
            cls_feat = feats[labels == cls]
            if cls_feat.shape[0] == 0:
                continue
            center = cls_feat.mean(dim=0)
            centers.append(center)
            intra_vals.append(torch.norm(cls_feat - center.unsqueeze(0), dim=-1).mean())
        if len(intra_vals) > 0:
            metrics["visual_intra_l2"] = float(torch.stack(intra_vals).mean().item())
        if len(centers) >= 2:
            center_t = torch.stack(centers, dim=0)
            dmat = torch.cdist(center_t, center_t, p=2)
            tri = self._upper_tri_flat(dmat)
            if tri.numel() > 0:
                metrics["visual_inter_l2"] = float(tri.mean().item())

        semantic_proj = banks.get("semantic_proj")
        if torch.is_tensor(semantic_proj):
            sem_sim = self._safe_cosine_matrix(semantic_proj)
            tri = self._upper_tri_flat(sem_sim)
            if tri.numel() > 0:
                metrics["semantic_sep_cos_dissim"] = float((1.0 - tri).mean().item())

        # Layer 2: cross-modal alignment.
        if torch.is_tensor(semantic_proj):
            sem_norm = torch.nn.functional.normalize(semantic_proj.float(), dim=-1)
            sim = v_norm @ sem_norm.t()
            pos = sim.gather(1, y_local.view(-1, 1)).squeeze(1)
            neg = sim.clone()
            neg.scatter_(1, y_local.view(-1, 1), -1e9)
            hard_neg = neg.max(dim=1).values
            margin = pos - hard_neg
            metrics["pos_sim_mean"] = float(pos.mean().item())
            metrics["pos_sim_std"] = float(pos.std().item())
            metrics["hard_neg_sim_mean"] = float(hard_neg.mean().item())
            metrics["hard_neg_sim_std"] = float(hard_neg.std().item())
            metrics["margin_mean"] = float(margin.mean().item())
            metrics["margin_std"] = float(margin.std().item())

            seen_ids = set(int(x) for x in list(dataset.seen_classes))
            unseen_ids = set(int(x) for x in list(dataset.unseen_classes))
            if len(seen_ids) > 0:
                m_seen = torch.tensor([int(y.item()) in seen_ids for y in labels], device=self.device, dtype=torch.bool)
                if m_seen.any():
                    metrics["margin_seen_mean"] = float(margin[m_seen].mean().item())
            if len(unseen_ids) > 0:
                m_unseen = torch.tensor([int(y.item()) in unseen_ids for y in labels], device=self.device, dtype=torch.bool)
                if m_unseen.any():
                    metrics["margin_unseen_mean"] = float(margin[m_unseen].mean().item())

            # Class-center to semantic prototype matrix.
            if len(centers) > 0:
                center_cls_ids = uniq.detach().cpu().tolist()
                center_t = torch.stack(centers, dim=0)
                center_n = torch.nn.functional.normalize(center_t.float(), dim=-1)
                m_v2s = center_n @ sem_norm.t()
                diag_vals = []
                top1 = []
                for row_idx, cid in enumerate(center_cls_ids):
                    if int(cid) in local_index:
                        col = local_index[int(cid)]
                        diag_vals.append(m_v2s[row_idx, col])
                        top1.append(int(m_v2s[row_idx].argmax().item() == col))
                if len(diag_vals) > 0:
                    diag_t = torch.stack(diag_vals)
                    metrics["v2s_diag_mean"] = float(diag_t.mean().item())
                    metrics["v2s_diag_top1_rate"] = float(np.mean(top1))
                off_mean = float(m_v2s.mean().item())
                if len(diag_vals) > 0:
                    metrics["v2s_offdiag_mean"] = off_mean - float(np.mean([x.item() for x in diag_vals])) / max(m_v2s.shape[1] - 1, 1)
                metrics["_m_v2s"] = m_v2s.detach().cpu().numpy()
                metrics["_m_v2s_rows"] = [int(x) for x in center_cls_ids]
                metrics["_m_v2s_cols"] = [int(x) for x in cids.detach().cpu().tolist()]

        # Layer 3: S^# specific monitoring.
        if torch.is_tensor(refined_sem_batch):
            if raw_sem_batch is not None:
                raw_map = r_head.semantic_anchor(raw_sem_batch)
                faith = torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(refined_sem_batch.float(), dim=-1),
                    torch.nn.functional.normalize(raw_map.float(), dim=-1),
                    dim=-1,
                )
                metrics["sref_faith_mean"] = float(faith.mean().item())
                metrics["sref_faith_min"] = float(faith.min().item())
                metrics["sref_faith_max"] = float(faith.max().item())

            # Intra-class stability of instance-conditioned refined semantics.
            st_vals = []
            for cls in torch.unique(labels):
                s_cls = refined_sem_batch[labels == cls]
                if s_cls.shape[0] <= 1:
                    continue
                c_s = s_cls.mean(dim=0, keepdim=True)
                st_vals.append(torch.norm(s_cls - c_s, dim=-1).mean())
            if len(st_vals) > 0:
                metrics["sref_intra_l2"] = float(torch.stack(st_vals).mean().item())

        if torch.is_tensor(semantic_proj) and len(centers) >= 2:
            center_t = torch.stack(centers, dim=0)
            mv = self._safe_cosine_matrix(center_t)
            center_cls_ids = [int(x) for x in uniq.detach().cpu().tolist()]
            cols = [local_index[cid] for cid in center_cls_ids if cid in local_index]
            if len(cols) >= 2:
                sem_sub = sem_norm.index_select(0, torch.tensor(cols, device=self.device, dtype=torch.long))
                ms = self._safe_cosine_matrix(sem_sub)
                v_flat = self._upper_tri_flat(mv)
                s_flat = self._upper_tri_flat(ms)
                if v_flat.numel() > 1:
                    corr = self._pearson_corr(v_flat, s_flat)
                    metrics["struct_corr_v_s"] = float(corr.item())

        metrics["num_samples"] = num_samples
        return metrics

    def _write_monitor_outputs(self, metrics: dict):
        if metrics is None:
            return
        os.makedirs(self.monitor_dir, exist_ok=True)
        epoch = int(metrics.get("epoch", self._trace_epoch + 1))
        split = str(metrics.get("split", "na"))

        logger.info(
            "[monitor] epoch=%d split=%s n=%s cand=%s margin=%.4f pos=%.4f hard_neg=%.4f sem_sep=%.4f faith=%.4f",
            epoch,
            split,
            metrics.get("num_samples", "NA"),
            metrics.get("candidate_count", "NA"),
            float(metrics.get("margin_mean", float("nan"))),
            float(metrics.get("pos_sim_mean", float("nan"))),
            float(metrics.get("hard_neg_sim_mean", float("nan"))),
            float(metrics.get("semantic_sep_cos_dissim", float("nan"))),
            float(metrics.get("sref_faith_mean", float("nan"))),
        )

        if self.monitor_save_json:
            out = {}
            for k, v in metrics.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (np.floating, np.integer)):
                    out[k] = v.item()
                else:
                    out[k] = v
            p = os.path.join(self.monitor_dir, "epoch_{:04d}_{}.json".format(epoch, split))
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        if self.monitor_save_csv:
            csv_fields = [
                "epoch", "split", "num_samples", "candidate_count",
                "margin_mean", "margin_std", "pos_sim_mean", "hard_neg_sim_mean",
                "sref_faith_mean", "sref_intra_l2",
                "visual_intra_l2", "visual_inter_l2",
                "semantic_sep_cos_dissim", "struct_corr_v_s",
                "v2s_diag_mean", "v2s_diag_top1_rate",
                "margin_seen_mean", "margin_unseen_mean",
            ]
            row = {k: metrics.get(k, "") for k in csv_fields}
            csv_exists = os.path.exists(self._monitor_csv_path)
            with open(self._monitor_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                if not csv_exists:
                    writer.writeheader()
                writer.writerow(row)

        if self.monitor_save_heatmap and ("_m_v2s" in metrics):
            try:
                import matplotlib.pyplot as plt
                m = metrics["_m_v2s"]
                r = min(m.shape[0], self.monitor_heatmap_topk)
                c = min(m.shape[1], self.monitor_heatmap_topk)
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                im = ax.imshow(m[:r, :c], aspect="auto", cmap="viridis")
                ax.set_title("V-center vs S-prototype (epoch {} {})".format(epoch, split))
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                heat_dir = os.path.join(self.monitor_dir, "heatmap")
                os.makedirs(heat_dir, exist_ok=True)
                fig.savefig(os.path.join(heat_dir, "epoch_{:04d}_{}_v2s.png".format(epoch, split)), dpi=160, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logger.warning("[monitor] heatmap save failed: %s", str(e))

    @torch.no_grad()
    def _run_monitor_epoch(self, epoch: int, train_loader, val_loader, test_seen_loader, test_unseen_loader):
        model_was_training = self.model.training
        self.model.eval()
        task_type = self.evaluator.task_type
        split_loaders = [
            ("train", train_loader),
            ("val_unseen", val_loader),
            ("test_unseen", test_unseen_loader),
        ]
        if task_type == "gzsl":
            split_loaders.insert(2, ("test_seen", test_seen_loader))
        for split, loader in split_loaders:
            if loader is None:
                continue
            pack = self._collect_monitor_samples(loader, split=split)
            if pack is None:
                continue
            metrics = self._compute_monitor_metrics(pack)
            if metrics is None:
                continue
            metrics["epoch"] = int(epoch + 1)
            self._write_monitor_outputs(metrics)
        if model_was_training:
            self.model.train()

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
        if attributes is None:
            raise ValueError("XLSA pipeline requires batch['attribute'] for semantic supervision.")
        attributes = attributes.to(self.device, non_blocking=True)

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
                        inputs, self.affinity_cfg, semantics=attributes, vis=True, class_ids=local_class_ids
                    )
                    self._last_attn_weights = attn_weights
                else:
                    outputs, affinities = self.model.forward_with_affinity(
                        inputs, self.affinity_cfg, semantics=attributes, class_ids=local_class_ids
                    )

                # 如果当前损失需要 affinity 辅助量，则从逐层 affinities 中抽取 attn_pv / attn_vs
                if self.affinity_aux_needed:
                    aux = self._extract_alignment_aux(affinities)
                    outputs = (outputs if not isinstance(outputs, tuple) else outputs[0], aux)
                else:
                    outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
            else:
                self._last_attn_weights = None
                outputs = self.model(inputs, semantics=attributes, class_ids=local_class_ids)

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
                    attn_pv_dbg = aux_dbg.get("attn_pv")
                    attn_vs_dbg = aux_dbg.get("attn_vs")
                    attn_ps_dbg = aux_dbg.get("attn_ps")
                    attn_sp_dbg = aux_dbg.get("attn_sp")
                    attn_sv_dbg = aux_dbg.get("attn_sv")
                    sample_layer = None
                    if isinstance(attn_pv_dbg, dict) and len(attn_pv_dbg) > 0:
                        sample_layer = sorted(attn_pv_dbg.keys())[0]
                    print(
                        "[SHAPE-DEBUG] trainer.loss_inputs attn_pv={} attn_vs={} attn_ps={} attn_sp={} attn_sv={} layer={}".format(
                            tuple(attn_pv_dbg[sample_layer].shape) if isinstance(attn_pv_dbg, dict) and sample_layer in attn_pv_dbg else (tuple(attn_pv_dbg.shape) if torch.is_tensor(attn_pv_dbg) else None),
                            tuple(attn_vs_dbg[sample_layer].shape) if isinstance(attn_vs_dbg, dict) and sample_layer in attn_vs_dbg else (tuple(attn_vs_dbg.shape) if torch.is_tensor(attn_vs_dbg) else None),
                            tuple(attn_ps_dbg[sample_layer].shape) if isinstance(attn_ps_dbg, dict) and sample_layer in attn_ps_dbg else (tuple(attn_ps_dbg.shape) if torch.is_tensor(attn_ps_dbg) else None),
                            tuple(attn_sp_dbg[sample_layer].shape) if isinstance(attn_sp_dbg, dict) and sample_layer in attn_sp_dbg else (tuple(attn_sp_dbg.shape) if torch.is_tensor(attn_sp_dbg) else None),
                            tuple(attn_sv_dbg[sample_layer].shape) if isinstance(attn_sv_dbg, dict) and sample_layer in attn_sv_dbg else (tuple(attn_sv_dbg.shape) if torch.is_tensor(attn_sv_dbg) else None),
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

            # patch_compete_balance_loss
            if is_train and self.patch_compete_balance_weight > 0.0:
                sem_state = None
                enc_ref = getattr(model_ref, "enc", None)
                transformer_ref = getattr(enc_ref, "transformer", None) if enc_ref is not None else None
                if transformer_ref is not None:
                    sem_state = getattr(transformer_ref, "_last_semantic_side_state", None)
                balance_term = sem_state.get("patch_compete_balance_loss") if isinstance(sem_state, dict) else None
                if torch.is_tensor(balance_term):
                    balance_term = balance_term.float().mean()
                    loss = loss + self.patch_compete_balance_weight * balance_term
                    self._last_train_debug["patch_compete_balance_loss"] = float(balance_term.detach().item())

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

                # scale可防止softmax 太平以及影响 margin loss 的实际强度
                scale_dbg = None
                model_ref_dbg = self._model_ref(self.model)
                r_head_dbg = model_ref_dbg.r_similarity_head
                if r_head_dbg is not None:
                    scale_dbg = getattr(r_head_dbg, "_loss_last_scale", None)
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

            if self._should_run_monitor(epoch):
                self._run_monitor_epoch(epoch, train_loader, val_loader, test_seen_loader, test_unseen_loader)

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

            if self._should_run_monitor(epoch):
                self._run_monitor_epoch(epoch, train_loader, None, test_seen_loader, test_unseen_loader)

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
                sem_tokens = None
                vis_tokens = None
                r_head_vis = getattr(model_ref, "r_similarity_head", None)
                affinities_vis = getattr(r_head_vis, "_runtime_affinities", None) if r_head_vis is not None else None
                sem_state_vis = getattr(r_head_vis, "_runtime_semantic_state", None) if r_head_vis is not None else None
                if isinstance(sem_state_vis, dict):
                    sem_tokens = sem_state_vis.get("sem_tokens", None)
                token_seq_vis = getattr(r_head_vis, "_runtime_token_sequence", None) if r_head_vis is not None else None
                if torch.is_tensor(token_seq_vis) and token_seq_vis.dim() == 3:
                    p_len = int(self.cfg.MODEL.PROMPT.NUM_TOKENS)
                    p_eff = p_len if token_seq_vis.shape[1] > (1 + p_len) else 0
                    vis_tokens = token_seq_vis[:, 1 + p_eff:, :]
                if torch.is_tensor(logits):
                    self._vis_update_trend(
                        affinities=affinities_vis,
                        logits=logits.detach(),
                        targets=targets_eval_local,
                        sem_tokens=sem_tokens,
                        visual_tokens=vis_tokens,
                        sem_state=sem_state_vis,
                    )
                    bsz = int(logits.shape[0])
                    for bi in range(bsz):
                        if self._vis_processed >= self.vis_max_samples:
                            break
                        gidx = int(idx * bsz + bi)
                        self._vis_collect_sample(
                            split=prefix,
                            local_idx=bi,
                            sample_idx=gidx,
                            image_chw=X[bi],
                            logits=logits.detach(),
                            target=int(targets_eval_local[bi].item()),
                            global_class_ids=eval_class_ids,
                            model_ref=model_ref,
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
        # 导出本轮可视化趋势
        if self._vis_split_enabled(prefix):
            self._vis_export_trend(prefix)
        return metrics




