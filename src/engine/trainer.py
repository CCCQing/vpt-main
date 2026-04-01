#!/usr/bin/env python3
"""
a trainer class
涓€涓€氱敤鐨勫垎绫昏缁冨櫒锛圱rainer锛?
鑱岃矗姒傝锛?1) 鏋勫缓浼樺寲鍣ㄤ笌瀛︿範鐜囪皟搴﹀櫒
2) 锛堝彲閫夛級浠庣粰瀹氳矾寰勫姞杞芥潈閲?3) 浠?epoch 涓虹矑搴﹁繘琛岃缁冧笌璇勬祴锛坴al/test锛?4) 璁板綍涓庢墦鍗拌缁?楠岃瘉鎸囨爣锛屽苟鏀寔鏃╁仠锛坧atience锛?"""
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

from ..tools.tsne_vis import extract_features, run_tsne, plot_tsne
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
    a trainer with below logics: 璁粌鍣紙Trainer锛変富瑕侀€昏緫锛?
    1. Build optimizer, scheduler 鏋勫缓浼樺寲鍣ㄥ拰瀛︿範鐜囪皟搴﹀櫒
    2. Load checkpoints if provided 锛堝彲閫夛級鍔犺浇 checkpoint锛堢敤浜庣户缁缁冩垨鍥哄畾鍒濆鍖栵級
    3. Train and eval at each epoch 姣忎釜 epoch锛氳缁?鈫?楠岃瘉锛堚啋 娴嬭瘯锛夆啋 璁板綍鏈€浣虫寚鏍?鈫?鏃╁仠
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # prompt 对齐损失是否启用（legacy）
        self.align_loss_enabled = (
                cfg.SOLVER.LOSS == "softmax_prompt_align" and getattr(cfg.SOLVER, "LOSS_ALPHA", 0.0) > 0
        )
        # Affinity aux may be needed by multiple new losses/modes.
        self.affinity_aux_needed = bool(
            self.align_loss_enabled
            or float(getattr(cfg.SOLVER, "LOSS_SEM_ROUTE_WEIGHT", 0.0)) > 0
            or float(getattr(cfg.SOLVER, "LOSS_ROLE_EARLY_WEIGHT", 0.0)) > 0
            or float(getattr(cfg.SOLVER, "LOSS_ROLE_LATE_WEIGHT", 0.0)) > 0
            or float(getattr(cfg.SOLVER, "LOSS_AVS_ENT_WEIGHT", 0.0)) > 0
            or str(getattr(cfg.MODEL, "SEMANTIC_SCORE_MODE", "global_refined")).lower() in {"affinity_role_migration", "agr_c2f"}
        )

        # affinity branch configuration锛氭寜闇€璧?forward_with_affinity 鍒嗘敮
        self.use_affinity = cfg.MODEL.AFFINITY.ENABLE or self.affinity_aux_needed
        if self.use_affinity:
            prompt_length = cfg.MODEL.AFFINITY.PROMPT_LENGTH
            if prompt_length <= 0:
                prompt_length = cfg.MODEL.PROMPT.NUM_TOKENS
            self.affinity_cfg = {
                "prompt_length": prompt_length,
                # 瀵归綈鎹熷け闇€瑕?prompt鈫抪atch 浜插拰锛岀‘淇?return_cross 鎵撳紑
                "return_cross": cfg.MODEL.AFFINITY.RETURN_CROSS or self.affinity_aux_needed,
                "normalize": cfg.MODEL.AFFINITY.NORMALIZE,
                "detach": cfg.MODEL.AFFINITY.DETACH,
            }
            self.affinity_vis = cfg.MODEL.AFFINITY.VIS
        else:
            self.affinity_cfg = None
            self.affinity_vis = False

        # solver related
        # ================== 浼樺寲鍣?/ 瀛︿範鐜囪皟搴﹀櫒 / 鎹熷け鍑芥暟 ==================
        logger.info("\tSetting up the optimizer...")
        # 杩欓噷浼犲叆 [self.model] 鏄负浜嗗吋瀹光€滃妯″瀷鑱斿悎浼樺寲鈥濈殑鎯呭喌
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        # ================== Checkpointer锛氱粺涓€绠＄悊淇濆瓨/鍔犺浇 ==================
        # Checkpointer 浼氳嚜鍔ㄥ鐞?state_dict 鐨勪繚瀛樹笌鍔犺浇
        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )
        # Optional pretrained checkpoint load.
        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments 浠呯敤浜?VTAB in-domain 瀹為獙
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.debug_grad_norm = bool(getattr(cfg.SOLVER, "DEBUG_GRAD_NORM", False))
        self.debug_trace_once = bool(getattr(cfg.SOLVER, "DEBUG_TRACE_ONCE", False))
        self.debug_shapes = bool(getattr(cfg.SOLVER, "DEBUG_SHAPES", False))
        self.overfit_one_batch_steps = int(getattr(cfg.SOLVER, "OVERFIT_ONE_BATCH_STEPS", 0))
        self._debug_batch_stats_logged = False
        self._debug_grad_logged = False
        self._debug_step_logged = False
        self._debug_forward_trace_logged = False
        self._debug_semantic_param_names_logged = False
        self._shape_debug_aux_logged = False
        self._shape_debug_loss_aux_logged = False
        self._overfit_cached_batch = None
        self._named_param_cache = None
        self.use_seen_only_train_ce = False
        self.train_seen_ids = None
        self.train_seen_ids_tensor = None
        self.train_seen_remap = None
        self.cls_weights_seen = None
        self._last_train_debug = {}
        self._last_ce_logits = None
        self._last_raw_logits = None
        self.patch_compete_balance_weight = float(getattr(cfg.MODEL.AFFINITY, "PATCH_COMPETE_BALANCE_WEIGHT", 0.0))
        self.overfit_disable_prompt_sampling = bool(
            getattr(cfg.SOLVER, "OVERFIT_DISABLE_PROMPT_SAMPLING", False)
        )
        self._trace_epoch = -1
        self._trace_iter = -1
        self._trace_stage = "init"
        self._trace_global_step = 0
        self._trace_rank = int(getattr(cfg, "DIST_RANK", 0))
        diag_cfg = getattr(cfg.SOLVER, "DIAG", None)
        self.diag_shuffle_raw_targets = bool(getattr(diag_cfg, "SHUFFLE_RAW_TARGETS", False)) if diag_cfg is not None else False
        mon_cfg = getattr(cfg.SOLVER, "MONITOR", None)
        self.monitor_enable = bool(getattr(mon_cfg, "ENABLE", False)) if mon_cfg is not None else False
        self.monitor_every_epoch = max(1, int(getattr(mon_cfg, "EVERY_EPOCH", 1))) if mon_cfg is not None else 1
        self.monitor_max_samples = max(1, int(getattr(mon_cfg, "MAX_SAMPLES", 512))) if mon_cfg is not None else 512
        self.monitor_save_json = bool(getattr(mon_cfg, "SAVE_JSON", True)) if mon_cfg is not None else True
        self.monitor_save_csv = bool(getattr(mon_cfg, "SAVE_CSV", True)) if mon_cfg is not None else True
        self.monitor_save_heatmap = bool(getattr(mon_cfg, "SAVE_HEATMAP", False)) if mon_cfg is not None else False
        self.monitor_heatmap_topk = max(1, int(getattr(mon_cfg, "HEATMAP_TOPK", 50))) if mon_cfg is not None else 50
        self.token_patch_stats_enable = bool(getattr(mon_cfg, "TOKEN_PATCH_STATS_ENABLE", False)) if mon_cfg is not None else False
        self.token_patch_source = str(getattr(mon_cfg, "TOKEN_PATCH_SOURCE", "avs")).lower() if mon_cfg is not None else "avs"
        self.token_patch_head_mode = str(getattr(mon_cfg, "TOKEN_PATCH_HEAD_MODE", "head_avg")).lower() if mon_cfg is not None else "head_avg"
        self.token_patch_toprho = float(getattr(mon_cfg, "TOKEN_PATCH_TOPRHO", 0.2)) if mon_cfg is not None else 0.2
        self.token_patch_save_maps = bool(getattr(mon_cfg, "TOKEN_PATCH_SAVE_MAPS", False)) if mon_cfg is not None else False
        self.token_patch_max_samples = max(1, int(getattr(mon_cfg, "TOKEN_PATCH_MAX_SAMPLES", 8))) if mon_cfg is not None else 8
        self.affinity_summary_enable = bool(getattr(mon_cfg, "AFFINITY_SUMMARY_ENABLE", True)) if mon_cfg is not None else True
        self.affinity_save_raw_dump = bool(getattr(mon_cfg, "AFFINITY_SAVE_RAW_DUMP", False)) if mon_cfg is not None else False
        self.affinity_keynode_viz_enable = bool(getattr(mon_cfg, "AFFINITY_KEYNODE_VIZ_ENABLE", True)) if mon_cfg is not None else True
        self.affinity_keynode_splits = [str(x).lower() for x in list(getattr(mon_cfg, "AFFINITY_KEYNODE_SPLITS", ["test"]))] if mon_cfg is not None else ["test"]
        self.affinity_keynode_max_figs = max(0, int(getattr(mon_cfg, "AFFINITY_KEYNODE_MAX_FIGS", 1))) if mon_cfg is not None else 1
        self.affinity_keynode_layer_policy = str(getattr(mon_cfg, "AFFINITY_KEYNODE_LAYER_POLICY", "first_middle_last")).lower() if mon_cfg is not None else "first_middle_last"
        self.monitor_dir = os.path.join(self.cfg.OUTPUT_DIR, "monitor")
        self._monitor_csv_path = os.path.join(self.monitor_dir, "summary.csv")
        self._monitor_warned_no_refined = False
        vis_cfg = getattr(cfg.SOLVER, "VIS", None)
        self.vis_enable = bool(getattr(vis_cfg, "ENABLE", False)) if vis_cfg is not None else False
        self.vis_every_epoch = max(1, int(getattr(vis_cfg, "EVERY_EPOCH", 1))) if vis_cfg is not None else 1
        self.vis_epoch_list = self._parse_vis_epoch_list(getattr(vis_cfg, "EPOCH_LIST", [])) if vis_cfg is not None else []
        self.vis_splits = list(getattr(vis_cfg, "SPLITS", ["val", "test"])) if vis_cfg is not None else ["val", "test"]
        self.vis_max_samples = max(1, int(getattr(vis_cfg, "MAX_SAMPLES", 8))) if vis_cfg is not None else 8
        self.vis_save_raw = bool(getattr(vis_cfg, "SAVE_RAW", True)) if vis_cfg is not None else True
        self.vis_save_images = bool(getattr(vis_cfg, "SAVE_IMAGES", True)) if vis_cfg is not None else True
        self.vis_local_control = bool(getattr(vis_cfg, "LOCAL_CONTROL", True)) if vis_cfg is not None else True
        self.vis_rollout = bool(getattr(vis_cfg, "ROLLOUT", True)) if vis_cfg is not None else True
        self.vis_gt_hn = bool(getattr(vis_cfg, "GT_HN_COMPARE", True)) if vis_cfg is not None else True
        self.vis_trends = bool(getattr(vis_cfg, "TRENDS", True)) if vis_cfg is not None else True
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
        return "stage={}|rank={}|epoch={}|iter={}|gstep={}".format(
            self._trace_stage,
            self._trace_rank,
            self._trace_epoch,
            self._trace_iter,
            self._trace_global_step,
        )

    def _set_model_trace_context(self, trace_id):
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        setattr(model_ref, "_debug_trace_id", trace_id)
        setattr(model_ref, "_debug_shapes", bool(self.debug_shapes))
        enc = getattr(model_ref, "enc", None)
        if enc is not None:
            setattr(enc, "_debug_trace_id", trace_id)
            setattr(enc, "_debug_shapes", bool(self.debug_shapes))
            transformer = getattr(enc, "transformer", None)
            if transformer is not None:
                setattr(transformer, "_debug_trace_id", trace_id)
                setattr(transformer, "_debug_shapes", bool(self.debug_shapes))
        setattr(self.model, "_debug_trace_id", trace_id)
        setattr(self.model, "_debug_shapes", bool(self.debug_shapes))

    def _named_params(self):
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
            r_head = getattr(self.model, "r_similarity_head", None)
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
        norms = {}
        for alias, (_, param) in refs.items():
            if param is None:
                norms[alias] = None
            else:
                norms[alias] = float(param.data.norm().item())
        return norms

    def _log_update_once(self, before_norms, after_norms):
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
        if t is None or (not torch.is_tensor(t)):
            return None
        return {
            "shape": tuple(t.shape),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

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
        return str(split).lower() in allowed

    @staticmethod
    def _parse_vis_epoch_list(v) -> list:
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

    @staticmethod
    def _infer_grid(num_patches: int) -> int:
        g = int(round(math.sqrt(max(1, int(num_patches)))))
        return max(1, g)

    @staticmethod
    def _entropy_np(x: torch.Tensor) -> float:
        if (not torch.is_tensor(x)) or x.numel() == 0:
            return float("nan")
        return float(entropy_lastdim(x).mean().item())

    def _vis_init_epoch(self, split: str):
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

    @staticmethod
    def _gini_np(x: np.ndarray) -> float:
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
                "source_path_or_source_name": "vit_backbones.vit.Block._compute_semantic_affinity",
                "softmax_dim_name": "key_semantic(last_dim)",
            },
            "ps": {
                "raw_key": "Aps",
                "source_path_or_source_name": "vit_backbones.vit.Block._compute_semantic_affinity",
                "softmax_dim_name": "key_semantic(last_dim)",
            },
        }

    def _token_patch_from_affinity(self, aff: dict) -> torch.Tensor:
        if not isinstance(aff, dict):
            return None
        source = str(self.token_patch_source).lower()
        x = None
        if source == "avs":
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
        # [B,H,N,T] or [B,N,T] -> [B,T,P]
        if a.dim() == 4:
            if self.token_patch_head_mode == "head0":
                a = a[:, 0, :, :]
            else:
                a = a.mean(dim=1)
        if a.dim() != 3:
            return None
        a = a.transpose(1, 2).contiguous()  # [B,T,P]
        a = a.clamp_min(0.0)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return a

    @staticmethod
    def _upper_tri_values(x: torch.Tensor) -> torch.Tensor:
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

    def _vis_update_trend(self, affinities, logits, targets, sem_tokens, visual_tokens=None, sem_state=None):
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
                        spec = float(avs2.max(dim=-1).values.mean().item())
                        self._vis_trend_buf["token_specialization"].append(spec)
                        usage = avs2.sum(dim=1)  # [B, M]
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
        elif matrix_name in {"pv", "ps"}:
            out["focused_prompt"] = int(np.argmin(row_ent))
            out["diffuse_prompt"] = int(np.argmax(row_ent))
        elif matrix_name == "vs":
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
        matrix_map = [("vv", "Avv"), ("pp", "App"), ("pv", "Apv"), ("vs", "Avs"), ("ps", "Aps")]
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

    def _vis_collect_sample(
        self,
        split: str,
        local_idx: int,
        sample_idx: int,
        image_chw: torch.Tensor,
        logits: torch.Tensor,
        target: int,
        model_ref,
    ):
        if self._vis_processed >= self.vis_max_samples:
            return
        r_head = getattr(model_ref, "r_similarity_head", None)
        if r_head is None:
            return
        affinities = getattr(r_head, "_runtime_affinities", None)
        token_seq = getattr(r_head, "_runtime_token_sequence", None)
        sem_state = getattr(r_head, "_runtime_semantic_state", None)
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
                if bool(getattr(self.cfg.MODEL.PROMPT, "NOOP_KEEP_PARAMS", False)):
                    p_len = 0
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
                if bool(getattr(self.cfg.MODEL.PROMPT, "NOOP_KEEP_PARAMS", False)):
                    p_len = 0
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
                                titles = [f"GT={y}", f"HN={hn}", "GT-HN"]
                                save_panel(
                                    os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_gt_hn_compare.png"),
                                    imgs, titles=titles, ncols=3
                                )
                            if self.vis_save_raw:
                                np.savez_compressed(
                                    os.path.join(base, f"{split}_ep{epoch:03d}_idx{sample_idx:05d}_gt_hn_maps.npz"),
                                    gt=gt_up, hn=hn_up, diff=diff_up, gt_id=y, hn_id=hn
                                )

        self._vis_processed += 1

    def _capture_train_debug(self, loss_outputs, raw_outputs, loss_targets):
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
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            r_head = getattr(model_ref, "r_similarity_head", None)
            if r_head is not None:
                fixed_scale = float(getattr(r_head, "fixed_logit_scale", 0.0))
                self._last_train_debug["whether_fixed_logit_scale"] = bool(fixed_scale > 0)
                scale_t = getattr(r_head, "_loss_last_scale", None)
                if torch.is_tensor(scale_t):
                    self._last_train_debug["effective_logit_scale"] = float(scale_t.detach().mean().item())
                score_stats = getattr(r_head, "_last_score_stats", None)
                if isinstance(score_stats, dict) and len(score_stats) > 0:
                    for k, v in score_stats.items():
                        if isinstance(v, (int, float, bool, str)) or v is None:
                            self._last_train_debug[k] = v
                        elif isinstance(v, dict):
                            for kk, vv in v.items():
                                if isinstance(vv, (int, float, bool, str)) or vv is None:
                                    self._last_train_debug[f"{k}.{kk}"] = vv
            hn_stats = getattr(self.cls_criterion, "_last_hn_stats", None)
            if isinstance(hn_stats, dict) and len(hn_stats) > 0:
                self._last_train_debug.update(hn_stats)

    def _set_prompt_sampling_mode(self, disable_sampling: bool):
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        provider = getattr(
            getattr(getattr(model_ref, "enc", None), "transformer", None),
            "prompt_init_provider",
            None,
        )
        if provider is None:
            return
        if hasattr(provider, "disable_sampling"):
            provider.disable_sampling = bool(disable_sampling)
            logger.info(
                "[debug] prompt provider sampling mode: disable_sampling=%s (overfit=%s)",
                bool(disable_sampling),
                self.overfit_one_batch_steps > 0,
            )
        else:
            logger.warning(
                "[debug] prompt provider does not expose disable_sampling switch; cannot force z=mu."
            )

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            return outputs[0]
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        return outputs

    @staticmethod
    def _replace_logits(outputs, logits):
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

    def _configure_seen_only_train_ce(self, train_loader):
        self.use_seen_only_train_ce = False
        self.train_seen_ids = None
        self.train_seen_ids_tensor = None
        self.train_seen_remap = None
        self.cls_weights_seen = None

        xlsa_cfg = getattr(self.cfg.DATA, "XLSA", None)
        xlsa_enabled = bool(getattr(xlsa_cfg, "ENABLED", False)) if xlsa_cfg is not None else False
        if not xlsa_enabled:
            return

        dataset = getattr(train_loader, "dataset", None)
        seen = getattr(dataset, "seen_classes", None)
        if seen is None:
            raise ValueError("XLSA training requires train_loader.dataset.seen_classes.")

        seen_ids = sorted(set(int(x) for x in list(seen)))
        if len(seen_ids) == 0:
            raise ValueError("XLSA training requires non-empty seen_classes.")

        total_classes = len(self.cls_weights)
        bad_ids = [cid for cid in seen_ids if cid < 0 or cid >= total_classes]
        if bad_ids:
            raise ValueError(
                "seen_classes contain out-of-range ids for class count {} (e.g., {}).".format(
                    total_classes, bad_ids[:10]
                )
            )

        seen_tensor = torch.tensor(seen_ids, dtype=torch.long, device=self.device)
        remap = torch.full((total_classes,), -1, dtype=torch.long, device=self.device)
        remap[seen_tensor] = torch.arange(len(seen_ids), dtype=torch.long, device=self.device)

        weights_np = np.asarray(self.cls_weights, dtype=np.float32)
        self.cls_weights_seen = weights_np[seen_ids].tolist()
        self.train_seen_ids = seen_ids
        self.train_seen_ids_tensor = seen_tensor
        self.train_seen_remap = remap
        self.use_seen_only_train_ce = True

        unseen = getattr(dataset, "unseen_classes", None)
        unseen_count = len(unseen) if unseen is not None else -1
        logger.info(
            "Seen-only train CE enabled: seen=%d unseen=%d ln(S)=%.6f seen_head=%s",
            len(seen_ids),
            unseen_count,
            float(np.log(max(len(seen_ids), 1))),
            seen_ids[:10],
        )

    def _prepare_seen_only_loss(self, outputs, targets):
        if not self.use_seen_only_train_ce:
            return outputs, targets, self.cls_weights

        logits = self._extract_logits(outputs)
        if not torch.is_tensor(logits):
            raise TypeError("Expected tensor logits for seen-only CE, got {}".format(type(logits)))

        if logits.dim() != 2:
            raise ValueError("Expected 2D logits [B, C], got shape {}".format(tuple(logits.shape)))

        max_t = int(targets.max().item())
        if max_t >= self.train_seen_remap.numel():
            raise ValueError(
                "Target id {} exceeds remap size {}.".format(max_t, self.train_seen_remap.numel())
            )

        mapped_targets = self.train_seen_remap[targets]
        if (mapped_targets < 0).any():
            bad = targets[mapped_targets < 0][:8].detach().cpu().tolist()
            raise ValueError(
                "Train batch contains non-seen target ids under seen-only CE (e.g., {}).".format(bad)
            )

        logits_seen = logits.index_select(dim=1, index=self.train_seen_ids_tensor)
        outputs_seen = self._replace_logits(outputs, logits_seen)
        return outputs_seen, mapped_targets, self.cls_weights_seen

    @staticmethod
    def _model_ref(model):
        return model.module if hasattr(model, "module") else model

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

    def _should_run_monitor(self, epoch: int) -> bool:
        if not self.monitor_enable:
            return False
        return ((epoch + 1) % self.monitor_every_epoch) == 0

    def _resolve_candidate_ids(self, split: str, dataset, targets_np: np.ndarray, num_classes: int) -> np.ndarray:
        split = str(split).lower()
        if split == "train":
            source = getattr(dataset, "seen_classes", None)
            if source is None:
                source = np.unique(targets_np)
        elif split == "val":
            source = np.unique(targets_np)
        else:
            source = getattr(dataset, "unseen_classes", None)
            if source is None:
                source = np.unique(targets_np)
        ids = np.asarray(list(source), dtype=np.int64).reshape(-1)
        valid = (ids >= 0) & (ids < int(num_classes))
        return np.unique(ids[valid])

    @torch.no_grad()
    def _collect_monitor_samples(self, data_loader, split: str):
        model_ref = self._model_ref(self.model)
        dataset = getattr(data_loader, "dataset", None)
        class_attr = getattr(dataset, "class_attributes", None)
        if class_attr is None:
            logger.warning("[monitor] split=%s skipped: dataset.class_attributes is missing", split)
            return None

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

            transformer = getattr(getattr(model_ref, "enc", None), "transformer", None)
            raw_sem = getattr(transformer, "_monitor_last_raw_semantics", None) if transformer is not None else None
            refined_sem = getattr(transformer, "_monitor_last_refined_semantics", None) if transformer is not None else None
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
        class_attr = getattr(dataset, "class_attributes", None)
        if class_attr is None:
            return None
        class_attr = class_attr.to(self.device).float() if torch.is_tensor(class_attr) else torch.tensor(class_attr, device=self.device).float()
        cids = torch.as_tensor(candidate_ids, device=self.device, dtype=torch.long)
        raw_attr_cand = class_attr.index_select(0, cids)

        bank = {
            "candidate_ids": cids,
            "raw_attr": raw_attr_cand,
            "orig_proj": None,
            "refined_proj": None,
        }

        if r_head is not None and getattr(r_head, "semantic_anchor", None) is not None:
            raw_embed = r_head.semantic_anchor(raw_attr_cand)
            bank["orig_proj"] = r_head.semantic_proj(raw_embed)

        if r_head is not None:
            refined_all = r_head._class_prototypes_refined()
            refined_cand = refined_all.index_select(0, cids)
            bank["refined_proj"] = r_head.semantic_proj(refined_cand)

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

        num_classes = int(getattr(r_head, "num_classes", feats.shape[-1]))
        targets_np = labels.detach().cpu().numpy()
        candidate_ids_np = self._resolve_candidate_ids(split, dataset, targets_np, num_classes)
        if candidate_ids_np.size == 0:
            logger.warning("[monitor] split=%s skipped: empty candidate_ids", split)
            return None

        banks = self._build_monitor_semantic_banks(dataset, candidate_ids_np, r_head)
        if banks is None:
            return None

        cids = banks["candidate_ids"]
        cid_set = set(int(x) for x in cids.detach().cpu().tolist())
        keep_mask = torch.tensor([int(y.item()) in cid_set for y in labels], device=self.device, dtype=torch.bool)
        if keep_mask.sum() == 0:
            logger.warning("[monitor] split=%s skipped: no labels in candidate_ids", split)
            return None

        feats = feats[keep_mask]
        labels = labels[keep_mask]
        if raw_sem_batch is not None:
            raw_sem_batch = raw_sem_batch[keep_mask]
        if refined_sem_batch is not None:
            refined_sem_batch = refined_sem_batch[keep_mask]

        v_proj = r_head.visual_proj(feats) if getattr(r_head, "visual_proj", None) is not None else feats
        v_norm = torch.nn.functional.normalize(v_proj.float(), dim=-1)

        local_index = {int(cid): i for i, cid in enumerate(cids.detach().cpu().tolist())}
        y_local = torch.tensor([local_index[int(y.item())] for y in labels], device=self.device, dtype=torch.long)

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

        refined_proj = banks.get("refined_proj")
        orig_proj = banks.get("orig_proj")
        if torch.is_tensor(refined_proj):
            sem_sim = self._safe_cosine_matrix(refined_proj)
            tri = self._upper_tri_flat(sem_sim)
            if tri.numel() > 0:
                metrics["semantic_refined_sep_cos_dissim"] = float((1.0 - tri).mean().item())
        if torch.is_tensor(orig_proj):
            sem_sim_o = self._safe_cosine_matrix(orig_proj)
            tri_o = self._upper_tri_flat(sem_sim_o)
            if tri_o.numel() > 0:
                metrics["semantic_orig_sep_cos_dissim"] = float((1.0 - tri_o).mean().item())

        # Layer 2: cross-modal alignment (refined as main).
        if torch.is_tensor(refined_proj):
            s_ref = torch.nn.functional.normalize(refined_proj.float(), dim=-1)
            sim_ref = v_norm @ s_ref.t()
            pos_ref = sim_ref.gather(1, y_local.view(-1, 1)).squeeze(1)
            neg_ref = sim_ref.clone()
            neg_ref.scatter_(1, y_local.view(-1, 1), -1e9)
            hard_ref = neg_ref.max(dim=1).values
            margin_ref = pos_ref - hard_ref
            metrics["pos_sim_mean"] = float(pos_ref.mean().item())
            metrics["pos_sim_std"] = float(pos_ref.std().item())
            metrics["hard_neg_sim_mean"] = float(hard_ref.mean().item())
            metrics["hard_neg_sim_std"] = float(hard_ref.std().item())
            metrics["margin_mean"] = float(margin_ref.mean().item())
            metrics["margin_std"] = float(margin_ref.std().item())

            seen_ids = set(int(x) for x in list(getattr(dataset, "seen_classes", []) or []))
            unseen_ids = set(int(x) for x in list(getattr(dataset, "unseen_classes", []) or []))
            if len(seen_ids) > 0:
                m_seen = torch.tensor([int(y.item()) in seen_ids for y in labels], device=self.device, dtype=torch.bool)
                if m_seen.any():
                    metrics["margin_seen_mean"] = float(margin_ref[m_seen].mean().item())
            if len(unseen_ids) > 0:
                m_unseen = torch.tensor([int(y.item()) in unseen_ids for y in labels], device=self.device, dtype=torch.bool)
                if m_unseen.any():
                    metrics["margin_unseen_mean"] = float(margin_ref[m_unseen].mean().item())

            # Class-center to semantic prototype matrix.
            if len(centers) > 0:
                center_cls_ids = uniq.detach().cpu().tolist()
                center_t = torch.stack(centers, dim=0)
                center_n = torch.nn.functional.normalize(center_t.float(), dim=-1)
                m_v2s = center_n @ s_ref.t()
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
            if (raw_sem_batch is not None) and (getattr(r_head, "semantic_anchor", None) is not None):
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

        if torch.is_tensor(orig_proj) and torch.is_tensor(refined_proj):
            s_org = torch.nn.functional.normalize(orig_proj.float(), dim=-1)
            s_ref = torch.nn.functional.normalize(refined_proj.float(), dim=-1)
            sim_org = v_norm @ s_org.t()
            sim_ref = v_norm @ s_ref.t()
            pos_org = sim_org.gather(1, y_local.view(-1, 1)).squeeze(1)
            pos_ref = sim_ref.gather(1, y_local.view(-1, 1)).squeeze(1)
            neg_org = sim_org.clone()
            neg_ref = sim_ref.clone()
            neg_org.scatter_(1, y_local.view(-1, 1), -1e9)
            neg_ref.scatter_(1, y_local.view(-1, 1), -1e9)
            margin_org = pos_org - neg_org.max(dim=1).values
            margin_ref = pos_ref - neg_ref.max(dim=1).values
            gain = margin_ref - margin_org
            metrics["sref_margin_gain_mean"] = float(gain.mean().item())

            # Structural consistency corr(M^V, M^S) vs corr(M^V, M^S#).
            if len(centers) >= 2:
                center_t = torch.stack(centers, dim=0)
                mv = self._safe_cosine_matrix(center_t)
                # Align semantic matrices to classes available in centers.
                center_cls_ids = [int(x) for x in uniq.detach().cpu().tolist()]
                cols = [local_index[cid] for cid in center_cls_ids if cid in local_index]
                if len(cols) >= 2:
                    so = s_org.index_select(0, torch.tensor(cols, device=self.device, dtype=torch.long))
                    sr = s_ref.index_select(0, torch.tensor(cols, device=self.device, dtype=torch.long))
                    ms_o = self._safe_cosine_matrix(so)
                    ms_r = self._safe_cosine_matrix(sr)
                    v_flat = self._upper_tri_flat(mv)
                    o_flat = self._upper_tri_flat(ms_o)
                    r_flat = self._upper_tri_flat(ms_r)
                    if v_flat.numel() > 1:
                        corr_o = self._pearson_corr(v_flat, o_flat)
                        corr_r = self._pearson_corr(v_flat, r_flat)
                        metrics["struct_corr_v_s_orig"] = float(corr_o.item())
                        metrics["struct_corr_v_s_refined"] = float(corr_r.item())

        metrics["num_samples"] = num_samples
        return metrics

    def _write_monitor_outputs(self, metrics: dict):
        if metrics is None:
            return
        os.makedirs(self.monitor_dir, exist_ok=True)
        epoch = int(metrics.get("epoch", self._trace_epoch + 1))
        split = str(metrics.get("split", "na"))

        logger.info(
            "[monitor] epoch=%d split=%s n=%s cand=%s margin=%.4f pos=%.4f hard_neg=%.4f gain=%.4f faith=%.4f",
            epoch,
            split,
            metrics.get("num_samples", "NA"),
            metrics.get("candidate_count", "NA"),
            float(metrics.get("margin_mean", float("nan"))),
            float(metrics.get("pos_sim_mean", float("nan"))),
            float(metrics.get("hard_neg_sim_mean", float("nan"))),
            float(metrics.get("sref_margin_gain_mean", float("nan"))),
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
                "sref_margin_gain_mean", "sref_faith_mean", "sref_intra_l2",
                "visual_intra_l2", "visual_inter_l2",
                "semantic_orig_sep_cos_dissim", "semantic_refined_sep_cos_dissim",
                "struct_corr_v_s_orig", "struct_corr_v_s_refined",
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
    def _run_monitor_epoch(self, epoch: int, train_loader, val_loader, test_loader):
        model_was_training = self.model.training
        self.model.eval()
        split_loaders = [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]
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

    def forward_one_batch(self, inputs, targets, is_train, attributes=None):
        """Train a single (full) epoch on the model using the given data loader.
        瀵逛竴涓?batch 鍋氬墠鍚戯紙鍙€夊弽鍚戯級璁＄畻銆?
        鍙傛暟锛?            inputs: 杈撳叆寮犻噺锛堜竴鑸舰鐘朵负 [B, C, H, W] 鎴?[B, D]锛?            targets: 鏍囩寮犻噺锛堜竴鑸舰鐘朵负 [B]锛?            is_train: bool锛岃缁冮樁娈典负 True锛岄獙璇?娴嬭瘯闃舵涓?False

        杩斿洖锛?            loss: 鏍囬噺鎹熷け锛堣缁冮樁娈碉級鎴栧崰浣嶆崯澶憋紙鏌愪簺鐗规畩鎯呭喌锛?            outputs: 妯″瀷杈撳嚭 logits锛屽舰鐘?[B, num_classes]
        """

        # ========== 1. 鎶婃暟鎹惉鍒版寚瀹氳澶?==========
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )
        if attributes is not None:
            attributes = attributes.to(self.device, non_blocking=True)

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        trace_id = self._make_trace_id()
        self._set_model_trace_context(trace_id)

        # ========== 2. 鍓嶅悜鎺ㄧ悊锛堣缁冩椂寮€鍚搴︼紝楠岃瘉/娴嬭瘯绂佺敤姊害锛?==========
        debug_logits = None
        with torch.set_grad_enabled(is_train):
            effective_targets = targets
            if is_train and self.diag_shuffle_raw_targets and targets.numel() > 1:
                perm = torch.randperm(targets.shape[0], device=targets.device)
                effective_targets = targets.index_select(0, perm)
            model_ref_for_runtime = self.model.module if hasattr(self.model, "module") else self.model
            r_head_runtime = getattr(model_ref_for_runtime, "r_similarity_head", None)
            if r_head_runtime is not None:
                r_head_runtime._runtime_targets = effective_targets.detach() if is_train else None
            if self.use_affinity:
                self._last_attn_weights = None
                if attributes is not None:
                    if self.affinity_vis:
                        outputs, attn_weights, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg, semantics=attributes, vis=True
                        )
                        self._last_attn_weights = attn_weights
                    else:
                        outputs, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg, semantics=attributes
                        )
                else:
                    if self.affinity_vis:
                        outputs, attn_weights, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg, vis=True
                        )
                        self._last_attn_weights = attn_weights
                    else:
                        outputs, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg
                        )

                if self.affinity_aux_needed: # forward_with_affinity 杩斿洖閫愬眰浜插拰鐭╅樀锛屾寜灞傛彁鍙栧榻愭崯澶辨墍闇€鐨?attn_pv/attn_vs
                    aux = self._extract_alignment_aux(affinities) # 鍙栦翰鍜屽苟鍋氬ご骞冲潎
                    outputs = (outputs if not isinstance(outputs, tuple) else outputs[0], aux)
                else:
                    outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
            else:
                self._last_attn_weights = None
                outputs = self.model(inputs, semantics=attributes)
            if r_head_runtime is not None:
                r_head_runtime._runtime_targets = None

            loss_outputs = outputs
            loss_targets = effective_targets
            loss_weights = self.cls_weights
            if is_train and self.use_seen_only_train_ce and not self.cls_criterion.is_local():
                loss_outputs, loss_targets, loss_weights = self._prepare_seen_only_loss(
                    outputs, targets
                )
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
                        bool(self.use_seen_only_train_ce),
                        int(loss_targets.min().item()) if torch.is_tensor(loss_targets) else "NA",
                    )
                self._debug_forward_trace_logged = True
            if is_train:
                self._capture_train_debug(loss_outputs, outputs, loss_targets)

            debug_logits = self._extract_logits(loss_outputs)
            if self.debug_grad_norm:
                self._log_batch_stats_once(debug_logits, loss_targets)

            if self.cfg.DBG:
                _logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        _logits.shape, targets.shape))

            if self.debug_shapes and (not self._shape_debug_loss_aux_logged):
                _, aux_dbg = self._extract_logits_and_aux_for_debug(loss_outputs)
                if isinstance(aux_dbg, dict):
                    attn_pv_dbg = aux_dbg.get("attn_pv")
                    attn_vs_dbg = aux_dbg.get("attn_vs")
                    attn_ps_dbg = aux_dbg.get("attn_ps")
                    sample_layer = None
                    if isinstance(attn_pv_dbg, dict) and len(attn_pv_dbg) > 0:
                        sample_layer = sorted(attn_pv_dbg.keys())[0]
                    print(
                        "[SHAPE-DEBUG] trainer.loss_inputs attn_pv={} attn_vs={} attn_ps={} layer={}".format(
                            tuple(attn_pv_dbg[sample_layer].shape) if isinstance(attn_pv_dbg, dict) and sample_layer in attn_pv_dbg else (tuple(attn_pv_dbg.shape) if torch.is_tensor(attn_pv_dbg) else None),
                            tuple(attn_vs_dbg[sample_layer].shape) if isinstance(attn_vs_dbg, dict) and sample_layer in attn_vs_dbg else (tuple(attn_vs_dbg.shape) if torch.is_tensor(attn_vs_dbg) else None),
                            tuple(attn_ps_dbg[sample_layer].shape) if isinstance(attn_ps_dbg, dict) and sample_layer in attn_ps_dbg else (tuple(attn_ps_dbg.shape) if torch.is_tensor(attn_ps_dbg) else None),
                            sample_layer,
                        )
                    )
                    self._shape_debug_loss_aux_logged = True

            # ================== 3. compute loss ==================
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            loss_kwargs = {
                "model": model_ref,
                "raw_targets": targets,
                "epoch": int(self._trace_epoch + 1),
            }
            if self.train_seen_ids_tensor is not None:
                loss_kwargs["seen_ids"] = self.train_seen_ids_tensor
            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                # 甯歌鍒嗙被鎹熷け锛堝 SoftmaxLoss锛夛紝鍙渶瑕?outputs / targets / class_weights
                loss = self.cls_criterion(
                    loss_outputs, loss_targets, loss_weights, kwargs=loss_kwargs)

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

            # ========== 4. 妫€鏌ユ崯澶辨槸鍚﹀紓甯革紙inf 鎴?NaN锛?==========
            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                try:
                    logits_dbg = debug_logits if torch.is_tensor(debug_logits) else self._extract_logits(loss_outputs)
                except Exception:
                    logits_dbg = None
                if torch.is_tensor(logits_dbg):
                    finite_ratio = float(torch.isfinite(logits_dbg).float().mean().item())
                    logit_min = float(torch.nan_to_num(logits_dbg, nan=0.0, posinf=0.0, neginf=0.0).min().item())
                    logit_max = float(torch.nan_to_num(logits_dbg, nan=0.0, posinf=0.0, neginf=0.0).max().item())
                else:
                    finite_ratio = float("nan")
                    logit_min = float("nan")
                    logit_max = float("nan")
                scale_dbg = None
                model_ref_dbg = self.model.module if hasattr(self.model, "module") else self.model
                r_head_dbg = getattr(model_ref_dbg, "r_similarity_head", None)
                if r_head_dbg is not None:
                    scale_dbg = getattr(r_head_dbg, "_loss_last_scale", None)
                logger.info(
                    "[nan-debug] loss=NaN logits_finite_ratio=%.6f logits[min,max]=[%.6f, %.6f] scale=%s last_train_debug=%s",
                    finite_ratio,
                    logit_min,
                    logit_max,
                    float(scale_dbg.detach().item()) if torch.is_tensor(scale_dbg) and scale_dbg.numel() == 1 else str(scale_dbg),
                    self._last_train_debug if isinstance(self._last_train_debug, dict) else {},
                )
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        # ========== 5. 鑻ュ浜庤缁冮樁娈碉紝鍒欐墽琛屽弽鍚戜笌鍙傛暟鏇存柊 ==========
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

    def _extract_alignment_aux(self, affinities):
        """
        浠?forward_with_affinity 鐨勪翰鍜屽垪琛ㄤ腑鎻愬彇瀵归綈鎹熷け闇€瑕佺殑 attn_pv / attn_vs銆?        鍏煎澶氬眰锛氭瀯閫?{layer_idx: tensor} 鐨勫瓧鍏革紝缂哄け鏃惰繑鍥?None銆傦紙妯″瀷鍓嶅悜杩斿洖鎵€鏈夊眰鐨勪翰鍜岀煩闃碉紝鏂逛究鍦?loss 渚х伒娲婚€夋嫨浣跨敤鍝竴灞傛垨澶氬眰銆傦級
        """
        if affinities is None:
            return None

        attn_pv = {}
        attn_vs = {}
        attn_ps = {}

        raw_apv_shape = None
        raw_avs_shape = None
        raw_aps_shape = None

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
                if raw_avs_shape is None and torch.is_tensor(avs):
                    raw_avs_shape = tuple(avs.shape)
                if avs.dim() == 4:
                    attn_vs[idx] = avs.mean(dim=1)
                elif avs.dim() == 3:
                    attn_vs[idx] = avs

            aps = affinity.get("Aps")
            if aps is not None:
                if raw_aps_shape is None and torch.is_tensor(aps):
                    raw_aps_shape = tuple(aps.shape)
                if aps.dim() == 4:
                    attn_ps[idx] = aps.mean(dim=1)
                elif aps.dim() == 3:
                    attn_ps[idx] = aps

        if not attn_pv or not attn_vs:
            return None

        out = {"attn_pv": attn_pv, "attn_vs": attn_vs}
        if attn_ps:
            out["attn_ps"] = attn_ps

        if self.debug_shapes and (not self._shape_debug_aux_logged):
            sample_layer = sorted(attn_pv.keys())[0] if len(attn_pv) > 0 else None
            apv_after = tuple(attn_pv[sample_layer].shape) if sample_layer is not None else None
            avs_after = tuple(attn_vs[sample_layer].shape) if sample_layer is not None and sample_layer in attn_vs else None
            aps_after = tuple(attn_ps[sample_layer].shape) if sample_layer is not None and sample_layer in attn_ps else None
            print(
                "[SHAPE-DEBUG] trainer._extract_alignment_aux affinity_raw Apv={} Avs={} Aps={} "
                "head_avg Apv={} Avs={} Aps={} layer={}".format(
                    raw_apv_shape,
                    raw_avs_shape,
                    raw_aps_shape,
                    apv_after,
                    avs_after,
                    aps_after,
                    sample_layer,
                )
            )
            self._shape_debug_aux_logged = True

        # Role-migration diagnostics (lightweight scalar summaries).
        role_cfg = getattr(getattr(self.cfg.MODEL, "ROLE_MIGRATION", None), "ENABLE", False)
        if bool(role_cfg):
            def _energy(x):
                return float(x.float().abs().mean().item())
            def _entropy(x):
                p = x.float().clamp_min(1e-8)
                return float((-(p * p.log()).sum(dim=-1).mean()).item())

            layer_ids = sorted(set(attn_pv.keys()) | set(attn_vs.keys()) | set(attn_ps.keys()))
            early_end = int(getattr(getattr(self.cfg.MODEL, "ROLE_MIGRATION", None), "EARLY_END", 3))
            late_start = int(getattr(getattr(self.cfg.MODEL, "ROLE_MIGRATION", None), "LATE_START", 9))
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

    def get_input(self, data):
        """
        浠?DataLoader 杩斿洖鐨?data 瀛楀吀涓彁鍙栬緭鍏ヤ笌鏍囩銆?
        棰勬湡 data 鐨勭粨鏋勶細
            data["image"]: np.ndarray 鎴?torch.Tensor
            data["label"]: np.ndarray 鎴?torch.Tensor

        杩斿洖锛?            inputs: float32 鐨勫浘鍍忓紶閲?            labels: 鏍囩寮犻噺锛堥€氬父涓?long 绫诲瀷锛?        """
        # 濡傛灉 dataloader 杩斿洖鐨勬槸 numpy锛屽垯缁熶竴杞垚 torch.Tensor
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()  # 淇濊瘉 float锛堟ā鍨嬩竴鑸湡鏈?float锛?
        labels = data["label"]

        attributes = data.get("attribute") if isinstance(data, dict) else None
        if attributes is not None and not isinstance(attributes, torch.Tensor):
            attributes = torch.from_numpy(attributes)
        return inputs, labels, attributes

    def _pick_primary_metric(self, metric_dict):
        """
        Select the early-stop metric according to evaluator task type.
        Returns: (metric_name, metric_value) or (None, None) if unavailable.
        """
        if not isinstance(metric_dict, dict):
            return None, None

        task_type = str(getattr(self.evaluator, "task_type", "standard") or "standard").lower()
        if task_type == "gzsl":
            candidates = ["gzsl_h", "zsl_unseen", "top1", "rocauc"]
        elif task_type == "zsl":
            candidates = ["zsl_unseen", "gzsl_h", "top1", "rocauc"]
        else:
            candidates = ["top1", "rocauc", "top5"]

        for key in candidates:
            val = metric_dict.get(key, None)
            if val is None:
                continue
            try:
                return key, float(val)
            except (TypeError, ValueError):
                continue
        return None, None

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        浠?epoch 涓哄崟浣嶈缁冨垎绫诲櫒锛屽苟鍦ㄦ瘡涓?epoch 鍚庤繘琛岄獙璇佸拰锛堝彲閫夛級娴嬭瘯銆?
        鍙傛暟锛?            train_loader: 璁粌闆?DataLoader
            val_loader:   楠岃瘉闆?DataLoader
            test_loader:  娴嬭瘯闆?DataLoader锛堝彲涓?None锛?        """

        # ================== 0. optional prompt snapshot before training ==================
        self.model.eval()
        self.save_prompt(0)

        # ================== 1. 涓€浜涜缁冭秴鍙傛暟涓庣姸鎬佸彉閲?==================
        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        effective_total_epoch = total_epoch
        if self.overfit_one_batch_steps > 0:
            effective_total_epoch = 1
            total_data = self.overfit_one_batch_steps
            if self._overfit_cached_batch is None:
                self._overfit_cached_batch = next(iter(train_loader))
            logger.info(
                "[debug] OVERFIT_ONE_BATCH_STEPS enabled: repeat one batch for %d steps",
                self.overfit_one_batch_steps,
            )
        best_epoch = -1                                 # 褰撳墠鏈€浼?epoch
        best_metric = float("-inf")                     # ensure first valid epoch can trigger improved/save
        logger.info("Best metric initialized to -inf for first-epoch save compatibility.")
        log_interval = self.cfg.SOLVER.LOG_EVERY_N      # 姣忓灏戜釜 batch 鎵撲竴娆℃棩蹇?
        # meters for per-epoch logging
        losses = AverageMeter('Loss', ':.4e')
        seen_top1_meter = AverageMeter('SeenTop1', ':.4e')
        hn_margin_meter = AverageMeter('HNMargin', ':.4e')
        pos_score_meter = AverageMeter('PosScore', ':.4e')
        hn_score_meter = AverageMeter('HNScore', ':.4e')
        train_margin_meter = AverageMeter('TrainMargin', ':.4e')
        p_margin_lt0_meter = AverageMeter('PMarginLT0', ':.4e')
        p_margin_ltneg1_meter = AverageMeter('PMarginLTNeg1', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        # class weights from dataset
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        self._configure_seen_only_train_ce(train_loader)
        if self.overfit_one_batch_steps > 0:
            self._set_prompt_sampling_mode(self.overfit_disable_prompt_sampling)
        # logger.info(f"class weights: {self.cls_weights}")

        patience = 0
        # ================== 2. 涓昏缁冨惊鐜紙鎸?epoch锛?==================
        for epoch in range(effective_total_epoch):
            # reset averagemeters to measure per-epoch results 姣忎釜 epoch 寮€濮嬪墠锛岄噸缃粺璁￠噺
            losses.reset()
            seen_top1_meter.reset()
            hn_margin_meter.reset()
            pos_score_meter.reset()
            hn_score_meter.reset()
            train_margin_meter.reset()
            p_margin_lt0_meter.reset()
            p_margin_ltneg1_meter.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, effective_total_epoch, lr
                )
            )

            # Enable training mode 鍒囨崲鍒拌缁冩ā寮忥紙鍚敤 Dropout / 鏇存柊 BN 缁熻绛夛級
            self.model.train()

            end = time.time()

            # ---------- 閬嶅巻涓€涓?epoch 鐨勬墍鏈?batch ----------
            if self.overfit_one_batch_steps > 0:
                batch_iter = ((i, self._overfit_cached_batch) for i in range(self.overfit_one_batch_steps))
            else:
                batch_iter = enumerate(train_loader)

            for idx, input_data in batch_iter:
                self._trace_stage = "train"
                self._trace_epoch = int(epoch)
                self._trace_iter = int(idx)
                self._trace_global_step += 1
                if self.cfg.DBG and idx == 20:
                    break
                
                X, targets, attributes = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                # 缁熻鏁版嵁鍔犺浇鏃堕棿
                data_time.update(time.time() - end)

                # 鍓嶅悜 + 锛堣嫢 is_train=True锛夊弽鍚戜笌浼樺寲
                train_loss, _ = self.forward_one_batch(X, targets, True, attributes=attributes)

                if train_loss == -1:
                    return None

                losses.update(train_loss.item(), X.shape[0])
                if isinstance(self._last_train_debug, dict) and "seen_only_top1" in self._last_train_debug:
                    seen_top1_meter.update(float(self._last_train_debug["seen_only_top1"]), X.shape[0])
                if isinstance(self._last_train_debug, dict):
                    if "hn_margin_loss" in self._last_train_debug:
                        hn_margin_meter.update(float(self._last_train_debug["hn_margin_loss"]), X.shape[0])
                    if "pos_score_mean" in self._last_train_debug:
                        pos_score_meter.update(float(self._last_train_debug["pos_score_mean"]), X.shape[0])
                    if "hn_score_mean" in self._last_train_debug:
                        hn_score_meter.update(float(self._last_train_debug["hn_score_mean"]), X.shape[0])
                    if "train_margin_mean" in self._last_train_debug:
                        train_margin_meter.update(float(self._last_train_debug["train_margin_mean"]), X.shape[0])
                    if "p_train_margin_lt_0" in self._last_train_debug:
                        p_margin_lt0_meter.update(float(self._last_train_debug["p_train_margin_lt_0"]), X.shape[0])
                    if "p_train_margin_lt_neg1" in self._last_train_debug:
                        p_margin_ltneg1_meter.update(float(self._last_train_debug["p_train_margin_lt_neg1"]), X.shape[0])

                # measure elapsed time 缁熻 batch 澶勭悊鏃堕棿
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
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
                    if hn_margin_meter.count > 0:
                        logger.info(
                            "[hn-margin] hn_margin_loss=%.6f pos_score_mean=%.6f hn_score_mean=%.6f "
                            "train_margin_mean=%.6f p_train_margin_lt_0=%.4f p_train_margin_lt_neg1=%.4f hn_detach_neg=%s",
                            float(hn_margin_meter.val),
                            float(pos_score_meter.val),
                            float(hn_score_meter.val),
                            float(train_margin_meter.val),
                            float(p_margin_lt0_meter.val),
                            float(p_margin_ltneg1_meter.val),
                            bool(getattr(self.cls_criterion, "hn_detach_neg", False)),
                        )
                    if isinstance(self._last_train_debug, dict) and self._last_train_debug.get("semantic_score_mode") in {"coarse_to_fine", "affinity_role_migration"}:
                        logger.info(
                            "[score-debug] mode=%s topk=%s alpha=%s train_include_gt=%s "
                            "coarse_topk_contains_gt_rate(pre_union)=%.4f coarse_topk_contains_gt_rate(post_union)=%s "
                            "agr_delta_norm=%s early_apv=%s late_aps=%s late_avs=%s",
                            str(self._last_train_debug.get("semantic_score_mode")),
                            str(self._last_train_debug.get("semantic_score_topk")),
                            str(self._last_train_debug.get("semantic_score_alpha")),
                            str(self._last_train_debug.get("semantic_score_train_include_gt")),
                            float(self._last_train_debug.get("coarse_topk_contains_gt_rate", float("nan"))),
                            str(self._last_train_debug.get("coarse_topk_contains_gt_rate_postunion")),
                            str(self._last_train_debug.get("agr_delta_norm_mean")),
                            str(self._last_train_debug.get("affinity_early_apv_energy")),
                            str(self._last_train_debug.get("affinity_late_aps_energy")),
                            str(self._last_train_debug.get("affinity_late_avs_energy")),
                        )
                    if self.overfit_one_batch_steps > 0 and self._last_train_debug:
                        dbg = self._last_train_debug
                        ce_stats = dbg.get("ce_logits_stats")
                        raw_stats = dbg.get("raw_logits_stats")
                        logger.info(
                            "[overfit-debug] loss=%.6f seen_top1=%.4f ce_classes=%d "
                            "ce_logits(mean/std/min/max)=%.6f/%.6f/%.6f/%.6f "
                            "entropy(ce)=%.6f entropy_from_ce=%s ce_vs_raw_same_tensor=%s",
                            float(train_loss),
                            float(dbg.get("seen_only_top1", 0.0)),
                            int(dbg.get("ce_classes", -1)),
                            float(ce_stats["mean"]) if ce_stats else float("nan"),
                            float(ce_stats["std"]) if ce_stats else float("nan"),
                            float(ce_stats["min"]) if ce_stats else float("nan"),
                            float(ce_stats["max"]) if ce_stats else float("nan"),
                            float(dbg.get("ce_entropy", float("nan"))),
                            bool(dbg.get("entropy_from_ce_logits", False)),
                            bool(dbg.get("ce_vs_raw_same_tensor", False)),
                        )
                        if raw_stats is not None:
                            logger.info(
                                "[overfit-debug] raw_logits(mean/std/min/max)=%.6f/%.6f/%.6f/%.6f entropy(raw)=%.6f",
                                float(raw_stats["mean"]),
                                float(raw_stats["std"]),
                                float(raw_stats["min"]),
                                float(raw_stats["max"]),
                                float(dbg.get("raw_entropy", float("nan"))),
                            )

                        r_head = getattr(self.model, "r_similarity_head", None)
                        if r_head is not None:
                            raw_sim = getattr(r_head, "_debug_last_raw_sim", None)
                            scaled_logits = getattr(r_head, "_debug_last_scaled_logits", None)
                            raw_sim_stats = self._tensor_stats(raw_sim)
                            scaled_stats = self._tensor_stats(scaled_logits)
                            if raw_sim_stats is not None and scaled_stats is not None:
                                logger.info(
                                    "[overfit-debug] r_head raw_sim(mean/std/min/max)=%.6f/%.6f/%.6f/%.6f "
                                    "scaled(mean/std/min/max)=%.6f/%.6f/%.6f/%.6f",
                                    float(raw_sim_stats["mean"]),
                                    float(raw_sim_stats["std"]),
                                    float(raw_sim_stats["min"]),
                                    float(raw_sim_stats["max"]),
                                    float(scaled_stats["mean"]),
                                    float(scaled_stats["std"]),
                                    float(scaled_stats["min"]),
                                    float(scaled_stats["max"]),
                                )
                                ce_logits = self._last_ce_logits
                                ce_from_scaled = False
                                if torch.is_tensor(ce_logits):
                                    if (
                                        ce_logits.shape == scaled_logits.shape
                                        and torch.allclose(ce_logits, scaled_logits, rtol=1e-5, atol=1e-6)
                                    ):
                                        ce_from_scaled = True
                                    elif (
                                        self.use_seen_only_train_ce
                                        and self.train_seen_ids_tensor is not None
                                        and ce_logits.shape[0] == scaled_logits.shape[0]
                                        and ce_logits.shape[1] == int(self.train_seen_ids_tensor.numel())
                                    ):
                                        scaled_seen = scaled_logits.index_select(
                                            dim=1, index=self.train_seen_ids_tensor
                                        )
                                        ce_from_scaled = bool(
                                            torch.allclose(ce_logits, scaled_seen, rtol=1e-5, atol=1e-6)
                                        )
                                logger.info(
                                    "[overfit-debug] CE logits sourced from scaled_logits=%s (seen_only=%s)",
                                    bool(ce_from_scaled),
                                    bool(self.use_seen_only_train_ce),
                                )
            # One-epoch summary
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, effective_total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}, train_seen_top1: {:.4f}".format(
                    losses.avg, seen_top1_meter.avg
                ))
            if hn_margin_meter.count > 0:
                logger.info(
                    "Epoch {} HN summary: hn_margin_loss={:.6f}, pos_score_mean={:.6f}, hn_score_mean={:.6f}, "
                    "train_margin_mean={:.6f}, p_train_margin_lt_0={:.4f}, p_train_margin_lt_neg1={:.4f}, hn_detach_neg={}".format(
                        epoch + 1,
                        hn_margin_meter.avg,
                        pos_score_meter.avg,
                        hn_score_meter.avg,
                        train_margin_meter.avg,
                        p_margin_lt0_meter.avg,
                        p_margin_ltneg1_meter.avg,
                        bool(getattr(self.cls_criterion, "hn_detach_neg", False)),
                    )
                )
            if isinstance(self._last_train_debug, dict) and self._last_train_debug.get("semantic_score_mode") in {"coarse_to_fine", "affinity_role_migration"}:
                logger.info(
                    "Epoch {} score summary: mode={} topk={} alpha={} train_include_gt={} "
                    "coarse_topk_contains_gt_rate(pre_union)={:.4f} coarse_topk_contains_gt_rate(post_union)={} agr_delta_norm={}".format(
                        epoch + 1,
                        self._last_train_debug.get("semantic_score_mode"),
                        self._last_train_debug.get("semantic_score_topk"),
                        self._last_train_debug.get("semantic_score_alpha"),
                        self._last_train_debug.get("semantic_score_train_include_gt"),
                        float(self._last_train_debug.get("coarse_topk_contains_gt_rate", float("nan"))),
                        self._last_train_debug.get("coarse_topk_contains_gt_rate_postunion"),
                        self._last_train_debug.get("agr_delta_norm_mean"),
                    )
                )
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
             # 鎸夊畼鏂瑰缓璁細scheduler.step() 搴斿湪 optimizer.step() 涔嬪悗璋冪敤
            if self.scheduler is not None:
                self.scheduler.step()

            # ================== 3. 楠岃瘉 / 娴嬭瘯闃舵 ==================
            # 鍒囨崲鍒?eval 妯″紡
            self.model.eval()
            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training # 鏇存柊 evaluator 鐨?epoch 鍙凤紝璇勬祴鏃剁敤浜庣粨鏋滃綊妗ｅ埌 epoch_k 涓?            # self.evaluator.update_iteration(epoch)
            # self.eval_classifier(val_loader, "val", epoch == total_epoch - 1) # 楠岃瘉闆嗚瘎娴嬶紙prefix="val"锛?
            # 20250902鏀瑰姩锛氬彲瑙嗗寲瀹炵幇锛岀‘淇濇嬁鍒版渶浣?checkpoint 鐨勫浘
            # -------- 鍏堝湪 val 涓婅瘎娴?--------
            self.evaluator.update_iteration(epoch)
            # save=False锛氶獙璇侀樁娈典笉闇€瑕佺珛鍗充繚瀛?logits
            self.eval_classifier(val_loader, "val", save=False)

            # 璇诲彇鏈疆 val 鐨勪富鎸囨爣锛坰tandard: top1; zsl: zsl_unseen; gzsl: gzsl_h锛?
            t_name = "val_" + val_loader.dataset.name
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

            # -------- 濡傛灉鎻愪緵浜?test_loader锛屽垯鍦?test 涓婁篃鍋氳瘎娴?--------
            # 鍙湁鍒锋柊鏈€浣虫椂锛屾墠鍦?test 涓婅Е鍙?save=True
            # 璁?eval_classifier 鍐呴儴淇濆瓨 logits / CLS 鐗瑰緛绛夌紦瀛橈紝鐢ㄤ簬鍚庣画鍙鍖栥€?
            if test_loader is not None:
                self.eval_classifier(test_loader, "test", save=improved)

            if self._should_run_monitor(epoch):
                self._run_monitor_epoch(epoch, train_loader, val_loader, test_loader)

            # 鍘熸潵鐨勪唬鐮佸仛鐨勬槸鍙繚瀛樻渶鍚庝竴杞紝杩欐牱瀹规槗鍙楁棭鍋滅殑褰卞搷
            # if test_loader is not None:                                       # 娴嬭瘯闆嗚瘎娴嬶紙濡傛彁渚涗簡 test_loader锛?            #     self.eval_classifier(test_loader, "test", epoch == total_epoch - 1)

            # check the patience ---------- 鏃╁仠閫昏緫锛氭牴鎹獙璇侀泦 top1 ----------
            # t_name = "val_" + val_loader.dataset.name
            # try:
            #     curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            # except KeyError: # 鑻ヨ瘎娴嬫寚鏍囩己澶憋紙渚嬪鏁版嵁/娴佺▼闂锛夛紝鍒欑洿鎺ヨ繑鍥?            #     return
            # --- 鏃╁仠涓庢渶浣宠褰曪紙淇濇寔浣犵殑鍘熼€昏緫涓嶅彉锛屼絾鍙敤鍒氬垰閭ｄ竴娆?curr_acc锛?--

            # ================== 4. 鏃╁仠閫昏緫锛堝熀浜庨獙璇侀泦 top1锛?==================
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

        # save the last checkpoints                 鍙€夛細淇濆瓨鏈€鍚庢ā鍨?        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        """
        灏嗗綋鍓嶆ā鍨嬩腑鐨?prompt embeddings 淇濆瓨鍒扮鐩橈紙鍙湪浣跨敤 ViT + prompt 鏃剁敓鏁堬級銆?
        鏉′欢锛?            - cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH 涓?True
            - cfg.MODEL.TYPE == "vit"
            - "prompt" in cfg.MODEL.TRANSFER_TYPE锛堝嵆鍚敤浜?prompt tuning锛?
        淇濆瓨鍐呭锛?            - "shallow_prompt": 娴呭眰 prompt锛堝墠缃?prompt锛夛紝褰㈢姸 [1, P, D 鎴?d]
            - "deep_prompt": 鑻?PROMPT.DEEP=True锛屽垯鍐嶄繚瀛樻瘡灞?deep prompt锛屽舰鐘?[L-1, P, D 鎴?d]

        鏂囦欢鍚嶏細
            OUTPUT_DIR/prompt_ep{epoch}.pth
        """
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                logger.warning(
                    "save_prompt skipped: static prompt embeddings were removed; "
                    "current prompt path is provider-driven."
                )

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """
        鍦ㄧ粰瀹?data_loader锛堥獙璇?娴嬭瘯闆嗭級涓婅瘎浼板垎绫绘€ц兘銆?
        鍙傛暟锛?            data_loader: DataLoader锛坴al 鎴?test锛?            prefix: 瀛楃涓插墠缂€锛岀敤浜庢爣璇嗗綋鍓嶈瘎娴嬬被鍨嬶紙"val" 鎴?"test"锛?            save: 鑻?True 涓?cfg.MODEL.SAVE_CKPT=True锛屽垯淇濆瓨 logits 涓?targets锛?                  骞跺湪 test 闃舵棰濆缂撳瓨 CLS 鐗瑰緛鐢ㄤ簬 t-SNE 鍙鍖栥€?        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target 鑱氬悎鍏ㄩ噺 logits 涓?targets锛岃瘎娴嬬粨鏉熶竴娆℃€ц绠楁寚鏍?
        total_logits = []
        total_targets = []
        total_sim_true_raw = []
        total_sim_true_ref = []
        total_sim_hn_raw = []
        total_sim_hn_ref = []
        total_sem_source = []
        total_sim_raw_all = []
        total_sim_ref_all = []
        coarse_topk_contains_gt_vals = []
        coarse_gap_mean_vals = []
        coarse_gap_median_vals = []
        coarse_recall_k_vals = []
        candidate_delta_gap_vals = []
        candidate_delta_gap_avail_vals = []
        coarse_topk_contains_gt_post_vals = []
        agr_delta_norm_vals = []
        affinity_early_apv_vals = []
        affinity_late_aps_vals = []
        affinity_late_avs_vals = []
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        r_head_eval = getattr(model_ref, "r_similarity_head", None)
        eval_override = str(getattr(self.cfg.MODEL, "SEMANTIC_SCORE_EVAL_OVERRIDE", "") or "").strip().lower()
        valid_override = {"global_raw", "global_refined", "coarse_to_fine", "affinity_role_migration"}
        override_applied = False
        original_score_mode = None
        if (
            r_head_eval is not None
            and prefix in {"val", "test"}
            and eval_override in valid_override
            and eval_override != str(getattr(r_head_eval, "semantic_score_mode", "")).lower()
        ):
            original_score_mode = str(getattr(r_head_eval, "semantic_score_mode", "global_refined"))
            r_head_eval.semantic_score_mode = eval_override
            override_applied = True
            test_name = f"{test_name}_{eval_override}"
            logger.info(
                "[eval-override] split=%s semantic_score_mode: %s -> %s",
                prefix,
                original_score_mode,
                eval_override,
            )
        if r_head_eval is not None:
            r_head_eval._runtime_targets = None
        if self._vis_split_enabled(prefix):
            self._vis_init_epoch(prefix)

        # ========== 閬嶅巻鏁翠釜鏁版嵁闆?==========
        for idx, input_data in enumerate(data_loader):
            self._trace_stage = f"eval_{prefix}"
            self._trace_iter = int(idx)
            self._trace_global_step += 1
            end = time.time()
            X, targets, attributes = self.get_input(input_data)

            # 缁熻鏁版嵁鍔犺浇鏃堕棿
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))

            # 璇勬祴闃舵锛歩s_train=False 鈫?forward_one_batch 鍙仛鍓嶅悜涓?loss 璁＄畻
            loss, outputs = self.forward_one_batch(X, targets, False, attributes=attributes)
            if loss == -1:                # 鍑虹幇 inf / NaN 鏃讹紝鐩存帴鍋滄
                if override_applied and r_head_eval is not None and original_score_mode is not None:
                    r_head_eval.semantic_score_mode = original_score_mode
                return
            losses.update(loss, X.shape[0])

            # 缁熻 batch 鏃堕棿
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

            # targets: Tensor 鈫?Python list[int]
            total_targets.extend(list(targets.numpy()))
            # outputs: logits Tensor锛屽厛鏀堕泦锛屾渶鍚庡啀 cat
            # outputs 鍙兘涓?logits Tensor / (logits, aux) / {"logits": ...}
            # 缁熶竴鎻愬彇 logits 浠ヤ究鍚庣画 cat
            logits = outputs
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                logits = outputs[0]
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]

            total_logits.append(logits)
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
                        targets=targets,
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
                            target=int(targets[bi].item()),
                            model_ref=model_ref,
                        )

            # Optional gain-cache signals for analyze_confusion.
            r_head = getattr(model_ref, "r_similarity_head", None)
            if r_head is not None:
                sim_raw_all = getattr(r_head, "_debug_last_raw_sim_raw", None)
                sim_ref_all = getattr(r_head, "_debug_last_raw_sim_ref", None)
                if torch.is_tensor(sim_raw_all) and torch.is_tensor(sim_ref_all):
                    total_sim_raw_all.append(sim_raw_all.detach().cpu())
                    total_sim_ref_all.append(sim_ref_all.detach().cpu())
                sst = getattr(r_head, "_last_score_stats", None)
                if isinstance(sst, dict):
                    c = sst.get("coarse_topk_contains_gt_rate", None)
                    if c is not None:
                        coarse_topk_contains_gt_vals.append(float(c))
                    c = sst.get("coarse_gap_mean", None)
                    if c is not None:
                        coarse_gap_mean_vals.append(float(c))
                    c = sst.get("coarse_gap_median", None)
                    if c is not None:
                        coarse_gap_median_vals.append(float(c))
                    c = sst.get("coarse_recall_at_k", None)
                    if c is not None:
                        coarse_recall_k_vals.append(float(c))
                    c = sst.get("candidate_delta_gap", None)
                    if c is not None:
                        candidate_delta_gap_vals.append(float(c))
                    c = sst.get("candidate_delta_gap_available_ratio", None)
                    if c is not None:
                        candidate_delta_gap_avail_vals.append(float(c))
                    c = sst.get("coarse_topk_contains_gt_rate_postunion", None)
                    if c is not None:
                        coarse_topk_contains_gt_post_vals.append(float(c))
                    c = sst.get("agr_delta_norm_mean", None)
                    if c is not None:
                        agr_delta_norm_vals.append(float(c))
                    c = sst.get("affinity_early_apv_energy", None)
                    if c is not None:
                        affinity_early_apv_vals.append(float(c))
                    c = sst.get("affinity_late_aps_energy", None)
                    if c is not None:
                        affinity_late_aps_vals.append(float(c))
                    c = sst.get("affinity_late_avs_energy", None)
                    if c is not None:
                        affinity_late_avs_vals.append(float(c))
                v = getattr(r_head, "_loss_last_visual", None)
                s_raw = getattr(r_head, "_loss_last_semantic_raw", None)
                s_ref = getattr(r_head, "_loss_last_semantic_ref", None)
                if torch.is_tensor(v) and torch.is_tensor(s_raw) and torch.is_tensor(s_ref):
                    if v.dim() == 2 and s_raw.dim() == 2 and s_ref.dim() == 2 and v.shape[0] == logits.shape[0]:
                        with torch.no_grad():
                            y = targets.to(device=v.device, dtype=torch.long)
                            sim_raw = v @ s_raw.t()
                            sim_ref = v @ s_ref.t()
                            pos_raw = sim_raw.gather(1, y.view(-1, 1)).squeeze(1)
                            pos_ref = sim_ref.gather(1, y.view(-1, 1)).squeeze(1)
                            logits_det = logits.detach()
                            hn = logits_det.clone()
                            hn.scatter_(1, y.view(-1, 1), -1e9)
                            hn_idx = hn.argmax(dim=1)
                            hn_raw = sim_raw.gather(1, hn_idx.view(-1, 1)).squeeze(1)
                            hn_ref = sim_ref.gather(1, hn_idx.view(-1, 1)).squeeze(1)
                            total_sim_true_raw.append(pos_raw.detach().cpu())
                            total_sim_true_ref.append(pos_ref.detach().cpu())
                            total_sim_hn_raw.append(hn_raw.detach().cpu())
                            total_sim_hn_ref.append(hn_ref.detach().cpu())
                            total_sem_source.extend([str(getattr(r_head, "_loss_last_source", "unknown"))] * int(v.shape[0]))

        # 鏁翠綋璇勬祴鏃ュ織
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))

        # 鑻ユā鍨嬩娇鐢ㄤ簡 side-tuning 鍒嗘敮锛岄澶栨墦鍗拌瀺鍚堢郴鏁?alpha
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))

        # 鎷兼帴寰楀埌 (num_samples, num_classes) 鐨?logits 鐭╅樀
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()

        # 璋冪敤 evaluator 璁＄畻鍒嗙被鎸囨爣锛堝唴閮ㄤ細鏍规嵁 DATA.MULTILABEL 澶勭悊鍗?澶氭爣绛惧満鏅級
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # ========== 鑻ラ渶瑕侊紝鍒欎繚瀛?logits 涓?targets 鍒版枃浠朵腑 ==========
        if save and self.cfg.MODEL.SAVE_CKPT:
            # 1) 宸叉湁
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out["semantic_score_mode"] = str(getattr(getattr(model_ref, "r_similarity_head", None), "semantic_score_mode", "unknown"))
            out["semantic_score_topk"] = int(getattr(getattr(model_ref, "r_similarity_head", None), "semantic_score_topk", 0))
            out["semantic_score_alpha"] = float(getattr(getattr(model_ref, "r_similarity_head", None), "semantic_score_alpha", 0.0))
            out["semantic_score_train_include_gt"] = bool(getattr(getattr(model_ref, "r_similarity_head", None), "semantic_score_train_include_gt", False))
            out["candidate_source"] = "raw_topk"
            if len(total_sim_true_raw) > 0 and len(total_sim_true_ref) > 0 and len(total_sim_hn_raw) > 0 and len(total_sim_hn_ref) > 0:
                out["sim_true_raw"] = torch.cat(total_sim_true_raw, dim=0).numpy()
                out["sim_true_ref"] = torch.cat(total_sim_true_ref, dim=0).numpy()
                out["sim_hn_raw"] = torch.cat(total_sim_hn_raw, dim=0).numpy()
                out["sim_hn_ref"] = torch.cat(total_sim_hn_ref, dim=0).numpy()
                if len(total_sem_source) == len(total_targets):
                    out["classifier_semantic_source"] = total_sem_source
            if len(total_sim_raw_all) > 0 and len(total_sim_ref_all) > 0:
                out["sim_raw_all"] = torch.cat(total_sim_raw_all, dim=0).numpy()
                out["sim_ref_all"] = torch.cat(total_sim_ref_all, dim=0).numpy()
            if len(coarse_topk_contains_gt_vals) > 0:
                out["coarse_topk_contains_gt_rate_preunion_mean"] = float(np.mean(coarse_topk_contains_gt_vals))
            if len(coarse_topk_contains_gt_post_vals) > 0:
                out["coarse_topk_contains_gt_rate_postunion_mean"] = float(np.mean(coarse_topk_contains_gt_post_vals))
            if len(coarse_recall_k_vals) > 0:
                out["coarse_recall_at_k_preunion_mean"] = float(np.mean(coarse_recall_k_vals))
            if len(coarse_gap_mean_vals) > 0:
                out["coarse_gap_mean_batches"] = float(np.mean(coarse_gap_mean_vals))
            if len(coarse_gap_median_vals) > 0:
                out["coarse_gap_median_batches"] = float(np.mean(coarse_gap_median_vals))
            if len(candidate_delta_gap_vals) > 0:
                out["candidate_delta_gap_batches"] = float(np.mean(candidate_delta_gap_vals))
            if len(candidate_delta_gap_avail_vals) > 0:
                out["candidate_delta_gap_available_ratio_batches"] = float(np.mean(candidate_delta_gap_avail_vals))
            if len(agr_delta_norm_vals) > 0:
                out["agr_delta_norm_batches"] = float(np.mean(agr_delta_norm_vals))
            if len(affinity_early_apv_vals) > 0:
                out["affinity_early_apv_energy_batches"] = float(np.mean(affinity_early_apv_vals))
            if len(affinity_late_aps_vals) > 0:
                out["affinity_late_aps_energy_batches"] = float(np.mean(affinity_late_aps_vals))
            if len(affinity_late_avs_vals) > 0:
                out["affinity_late_avs_energy_batches"] = float(np.mean(affinity_late_avs_vals))
            class_names = getattr(data_loader.dataset, "classes", None)
            if class_names is None:
                class_names = getattr(data_loader.dataset, "class_names", None)
            if class_names is not None:
                out["class_names"] = [str(x) for x in list(class_names)]
            out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(f"Saved logits and targets for {test_name} at {out_path}")

        # ========== 鑻ユ槸 test 闃舵涓?save=True锛屽垯棰濆缂撳瓨 CLS 鐗瑰緛鐢ㄤ簬 t-SNE ==========
        if save and prefix == "test":
            os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "cache"), exist_ok=True)
            cache_dir = os.path.join(self.cfg.OUTPUT_DIR, "cache")

            # 1) extract CLS features (up to 100 per class)
            X_cls, y = extract_features(
                self.model, data_loader, self.device,
                feat_type="cls", max_per_class=100
            )
            np.savez_compressed(
                os.path.join(cache_dir, f"{test_name}_cls.npz"),
                X=X_cls.astype("float32"),      # CLS 鐗瑰緛
                y=y,                            # 瀵瑰簲鏍囩
                meta=dict(type="cls")           # 鍏冧俊鎭?
            )

            logger.info(f"[t-SNE cache] saved CLS features to {cache_dir}")
        if override_applied and r_head_eval is not None and original_score_mode is not None:
            r_head_eval.semantic_score_mode = original_score_mode
        if self._vis_split_enabled(prefix):
            self._vis_export_trend(prefix)
        # === eval_classifier 缁撴潫 ===
