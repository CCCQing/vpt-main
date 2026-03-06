#!/usr/bin/env python3
"""
a trainer class
一个通用的分类训练器（Trainer）

职责概览：
1) 构建优化器与学习率调度器
2) （可选）从给定路径加载权重
3) 以 epoch 为粒度进行训练与评测（val/test）
4) 记录与打印训练/验证指标，并支持早停（patience）
"""
import datetime
import time
import torch
import torch.nn as nn
import os
import numpy as np

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

from ..tools.tsne_vis import extract_features, run_tsne, plot_tsne

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics: 训练器（Trainer）主要逻辑：

    1. Build optimizer, scheduler 构建优化器和学习率调度器
    2. Load checkpoints if provided （可选）加载 checkpoint（用于继续训练或固定初始化）
    3. Train and eval at each epoch 每个 epoch：训练 → 验证（→ 测试）→ 记录最佳指标 → 早停
    """
    def __init__(
        self,
        cfg: CfgNode,           # 全局配置（SOLVER / DATA / MODEL 等）
        model: nn.Module,       # 待训练模型（已由外部 build 完成）
        evaluator: Evaluator,   # 评测器（负责聚合并打印指标）
        device: torch.device,   # 训练设备（cuda / cpu）
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # prompt 对齐损失是否启用（仅在选择 softmax_prompt_align 且 alpha>0 时开启，对应 L_avg 正则）
        self.align_loss_enabled = (
                cfg.SOLVER.LOSS == "softmax_prompt_align" and getattr(cfg.SOLVER, "LOSS_ALPHA", 0.0) > 0
        )

        # affinity branch configuration：按需走 forward_with_affinity 分支
        self.use_affinity = cfg.MODEL.AFFINITY.ENABLE or self.align_loss_enabled
        if self.use_affinity:
            prompt_length = cfg.MODEL.AFFINITY.PROMPT_LENGTH
            if prompt_length <= 0:
                prompt_length = cfg.MODEL.PROMPT.NUM_TOKENS
            self.affinity_cfg = {
                "prompt_length": prompt_length,
                # 对齐损失需要 prompt→patch 亲和，确保 return_cross 打开
                "return_cross": cfg.MODEL.AFFINITY.RETURN_CROSS or self.align_loss_enabled,
                "normalize": cfg.MODEL.AFFINITY.NORMALIZE,
                "detach": cfg.MODEL.AFFINITY.DETACH,
            }
            self.affinity_vis = cfg.MODEL.AFFINITY.VIS
        else:
            self.affinity_cfg = None
            self.affinity_vis = False

        # solver related
        # ================== 优化器 / 学习率调度器 / 损失函数 ==================
        logger.info("\tSetting up the optimizer...")
        # 这里传入 [self.model] 是为了兼容“多模型联合优化”的情况
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # 根据 cfg.SOLVER.LOSS 构建分类损失（默认 softmax cross-entropy）
        self.cls_criterion = build_loss(self.cfg)

        # ================== Checkpointer：统一管理保存/加载 ==================
        # Checkpointer 会自动处理 state_dict 的保存与加载
        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,    # checkpoint 的保存路径
            save_to_disk=True
        )
        # 若指定了 MODEL.WEIGHT_PATH，则先加载指定权重
        # 注：这里排除了分类头最后一层（head.last_layer.*），常用于“线性探针/下游任务”场景
        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments 仅用于 VTAB in-domain 实验
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.debug_grad_norm = bool(getattr(cfg.SOLVER, "DEBUG_GRAD_NORM", False))
        self.debug_trace_once = bool(getattr(cfg.SOLVER, "DEBUG_TRACE_ONCE", False))
        self.overfit_one_batch_steps = int(getattr(cfg.SOLVER, "OVERFIT_ONE_BATCH_STEPS", 0))
        self._debug_batch_stats_logged = False
        self._debug_grad_logged = False
        self._debug_step_logged = False
        self._debug_forward_trace_logged = False
        self._debug_semantic_param_names_logged = False
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
        self.overfit_disable_prompt_sampling = bool(
            getattr(cfg.SOLVER, "OVERFIT_DISABLE_PROMPT_SAMPLING", False)
        )
        self._trace_epoch = -1
        self._trace_iter = -1
        self._trace_stage = "init"
        self._trace_global_step = 0
        self._trace_rank = int(getattr(cfg, "DIST_RANK", 0))

        if self.debug_grad_norm:
            self._log_optimizer_param_groups()
        if self.debug_trace_once:
            self._log_semantic_param_names_once()

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
        enc = getattr(model_ref, "enc", None)
        if enc is not None:
            setattr(enc, "_debug_trace_id", trace_id)
            transformer = getattr(enc, "transformer", None)
            if transformer is not None:
                setattr(transformer, "_debug_trace_id", trace_id)
        setattr(self.model, "_debug_trace_id", trace_id)

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
        refs["semantic_concept"] = self._find_param_by_name_contains(
            ["semantic_concept.concept_slots", "semantic_cross_attn", "semantic_concept"]
        )
        refs["semantic_concept.proj"] = self._find_param_by_name_contains(
            [
                "semantic_concept.query_semantic.weight",
                "semantic_concept.semantic_proj.weight",
                "semantic_concept.key_slots.weight",
            ]
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
        names = [n for n, _ in self.model.named_parameters() if "semantic_concept" in n]
        logger.info(
            "[trace] semantic_concept param names (%d): %s",
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
                any(("semantic_concept" in n) or ("semantic_cross_attn" in n) for n in names),
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

    def forward_one_batch(self, inputs, targets, is_train, attributes=None):
        """Train a single (full) epoch on the model using the given data loader.
        对一个 batch 做前向（可选反向）计算。

        参数：
            inputs: 输入张量（一般形状为 [B, C, H, W] 或 [B, D]）
            targets: 标签张量（一般形状为 [B]）
            is_train: bool，训练阶段为 True，验证/测试阶段为 False

        返回：
            loss: 标量损失（训练阶段）或占位损失（某些特殊情况）
            outputs: 模型输出 logits，形状 [B, num_classes]
        """

        # ========== 1. 把数据搬到指定设备 ==========
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )
        if attributes is not None:
            attributes = attributes.to(self.device, non_blocking=True)

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        trace_id = self._make_trace_id()
        self._set_model_trace_context(trace_id)

        # ========== 2. 前向推理（训练时开启梯度，验证/测试禁用梯度） ==========
        debug_logits = None
        with torch.set_grad_enabled(is_train):
            if self.use_affinity:
                if attributes is not None:
                    if self.affinity_vis:
                        outputs, attn_weights, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg, semantics=attributes, vis=True
                        )
                    else:
                        outputs, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg, semantics=attributes
                        )
                else:
                    if self.affinity_vis:
                        outputs, attn_weights, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg, vis=True
                        )
                    else:
                        outputs, affinities = self.model.forward_with_affinity(
                            inputs, self.affinity_cfg
                        )

                if self.align_loss_enabled: # forward_with_affinity 返回逐层亲和矩阵，按层提取对齐损失所需的 attn_pv/attn_vs
                    aux = self._extract_alignment_aux(affinities) # 取亲和并做头平均
                    outputs = (outputs if not isinstance(outputs, tuple) else outputs[0], aux)
                else:
                    outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
            else:
                outputs = self.model(inputs, semantics=attributes)

            loss_outputs = outputs
            loss_targets = targets
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

            # ================== 3. 计算损失 ==================
            # 一些“局部损失”（is_local=True）需要拿到 model / inputs 参与计算
            # 例如某些正则或提示学习的约束，且为了稳定会在 eval() 下运行
            if self.cls_criterion.is_local() and is_train:
                # 把模型暂时切到 eval，避免 BN/Dropout 带来随机性
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                # 评测阶段且是“局部损失”时，当前实现直接返回一个占位 loss（值=1）
                # 这里只是为了接口统一，真正评测时一般只关心 logits。
                return torch.tensor(1), outputs
            else:
                # 常规分类损失（如 SoftmaxLoss），只需要 outputs / targets / class_weights
                loss = self.cls_criterion(
                    loss_outputs, loss_targets, loss_weights)

            # ========== 4. 检查损失是否异常（inf 或 NaN） ==========
            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        # ========== 5. 若处于训练阶段，则执行反向与参数更新 ==========
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
        从 forward_with_affinity 的亲和列表中提取对齐损失需要的 attn_pv / attn_vs。
        兼容多层：构造 {layer_idx: tensor} 的字典，缺失时返回 None。（模型前向返回所有层的亲和矩阵，方便在 loss 侧灵活选择使用哪一层或多层。）
        """
        if affinities is None:
            return None

        attn_pv = {}
        attn_vs = {}

        for idx, affinity in enumerate(affinities):
            if not isinstance(affinity, dict):
                continue

            apv = affinity.get("Apv")
            if apv is not None:
                if apv.dim() == 4:
                    attn_pv[idx] = apv.mean(dim=1)
                elif apv.dim() == 3:
                    attn_pv[idx] = apv

            avs = affinity.get("Avs")
            if avs is not None:
                if avs.dim() == 4:
                    attn_vs[idx] = avs.mean(dim=1)
                elif avs.dim() == 3:
                    attn_vs[idx] = avs

        if not attn_pv or not attn_vs:
            return None

        return {"attn_pv": attn_pv, "attn_vs": attn_vs}

    def get_input(self, data):
        """
        从 DataLoader 返回的 data 字典中提取输入与标签。

        预期 data 的结构：
            data["image"]: np.ndarray 或 torch.Tensor
            data["label"]: np.ndarray 或 torch.Tensor

        返回：
            inputs: float32 的图像张量
            labels: 标签张量（通常为 long 类型）
        """
        # 如果 dataloader 返回的是 numpy，则统一转成 torch.Tensor
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()  # 保证 float（模型一般期望 float）
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
        以 epoch 为单位训练分类器，并在每个 epoch 后进行验证和（可选）测试。

        参数：
            train_loader: 训练集 DataLoader
            val_loader:   验证集 DataLoader
            test_loader:  测试集 DataLoader（可为 None）
        """

        # ================== 0. 在训练开始前可选地保存一次 prompt（VPT） ==================
        # save the model prompt if required before training 如需保存 “prompt embedding”，在训练开始前保存一次“epoch 0 前”的状态
        self.model.eval()
        self.save_prompt(0)

        # ================== 1. 一些训练超参数与状态变量 ==================
        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH       # 总 epoch 数
        total_data = len(train_loader)                  # 每个 epoch 的 batch 数
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
        best_epoch = -1                                 # 当前最优 epoch
        best_metric = 0                                 # 最优指标（比如 top1）
        log_interval = self.cfg.SOLVER.LOG_EVERY_N      # 每多少个 batch 打一次日志

        # 若干计量器（统计平均损失/时间等，便于打印）
        losses = AverageMeter('Loss', ':.4e')           # - losses: 每个 epoch 内的平均训练损失
        seen_top1_meter = AverageMeter('SeenTop1', ':.4e')  # - seen_top1_meter: 训练口径 top1
        batch_time = AverageMeter('Time', ':6.3f')      # - batch_time: 每个 batch 的时间
        data_time = AverageMeter('Data', ':6.3f')       # - data_time: 数据加载时间

        # 从训练集 Dataset 获取类别权重，传给损失函数（应对类分布不平衡）
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        self._configure_seen_only_train_ce(train_loader)
        if self.overfit_one_batch_steps > 0:
            self._set_prompt_sampling_mode(self.overfit_disable_prompt_sampling)
        # logger.info(f"class weights: {self.cls_weights}")

        # 早停用 patience：若验证集 metric 连续若干次不提升就停止
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training 早停计数器；若超过 cfg.SOLVER.PATIENCE 则停止训练

        # ================== 2. 主训练循环（按 epoch） ==================
        for epoch in range(effective_total_epoch):
            # reset averagemeters to measure per-epoch results 每个 epoch 开始前，重置统计量
            losses.reset()
            seen_top1_meter.reset()
            batch_time.reset()
            data_time.reset()

            # 当前学习率（假设 scheduler 里第一组 lr 代表全局 lr）
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, effective_total_epoch, lr
                )
            )

            # Enable training mode 切换到训练模式（启用 Dropout / 更新 BN 统计等）
            self.model.train()

            end = time.time()

            # ---------- 遍历一个 epoch 的所有 batch ----------
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
                    # if debugging, only need to see the first few iterations # 调试模式：仅跑前 20 个 batch 以加速
                    break
                
                X, targets, attributes = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                # 统计数据加载时间
                data_time.update(time.time() - end)

                # 前向 + （若 is_train=True）反向与优化
                train_loss, _ = self.forward_one_batch(X, targets, True, attributes=attributes)

                # 若 forward 返回 -1，说明出现 inf / NaN，直接停止训练
                if train_loss == -1:
                    return None

                # 更新本 epoch 的平均损失
                losses.update(train_loss.item(), X.shape[0])
                if isinstance(self._last_train_debug, dict) and "seen_only_top1" in self._last_train_debug:
                    seen_top1_meter.update(float(self._last_train_debug["seen_only_top1"]), X.shape[0])

                # measure elapsed time 统计 batch 处理时间
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch 每隔 log_interval 个 batch 打印一次训练日志
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    # 估算剩余时间（本 epoch 剩余 + 后续 epoch）
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(effective_total_epoch-epoch-1)))
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
            # 一个 epoch 的汇总日志
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, effective_total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}, train_seen_top1: {:.4f}".format(
                    losses.avg, seen_top1_meter.avg
                ))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
             # 按官方建议：scheduler.step() 应在 optimizer.step() 之后调用
            if self.scheduler is not None:
                self.scheduler.step()

            # ================== 3. 验证 / 测试阶段 ==================
            # 切换到 eval 模式
            self.model.eval()
            # 保存当下 epoch 的 prompt embeddings（如需求与配置指定）
            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training # 更新 evaluator 的 epoch 号，评测时用于结果归档到 epoch_k 下
            # self.evaluator.update_iteration(epoch)
            # self.eval_classifier(val_loader, "val", epoch == total_epoch - 1) # 验证集评测（prefix="val"）

            # 20250902改动：可视化实现，确保拿到最佳 checkpoint 的图
            # -------- 先在 val 上评测 --------
            self.evaluator.update_iteration(epoch)
            # save=False：验证阶段不需要立即保存 logits
            self.eval_classifier(val_loader, "val", save=False)

            # 读取本轮 val 的主指标（standard: top1; zsl: zsl_unseen; gzsl: gzsl_h）
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

            # -------- 如果提供了 test_loader，则在 test 上也做评测 --------
            # 只有刷新最佳时，才在 test 上触发 save=True
            # 让 eval_classifier 内部保存 logits / CLS 特征等缓存，用于后续可视化。
            if test_loader is not None:
                self.eval_classifier(test_loader, "test", save=improved)

            # 原来的代码做的是只保存最后一轮，这样容易受早停的影响
            # if test_loader is not None:                                       # 测试集评测（如提供了 test_loader）
            #     self.eval_classifier(test_loader, "test", epoch == total_epoch - 1)

            # check the patience ---------- 早停逻辑：根据验证集 top1 ----------
            # t_name = "val_" + val_loader.dataset.name
            # try:
            #     curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            # except KeyError: # 若评测指标缺失（例如数据/流程问题），则直接返回
            #     return
            # --- 早停与最佳记录（保持你的原逻辑不变，但只用刚刚那一次 curr_acc）---

            # ================== 4. 早停逻辑（基于验证集 top1） ==================
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

        # save the last checkpoints                 可选：保存最后模型
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        """
        将当前模型中的 prompt embeddings 保存到磁盘（只在使用 ViT + prompt 时生效）。

        条件：
            - cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH 为 True
            - cfg.MODEL.TYPE == "vit"
            - "prompt" in cfg.MODEL.TRANSFER_TYPE（即启用了 prompt tuning）

        保存内容：
            - "shallow_prompt": 浅层 prompt（前置 prompt），形状 [1, P, D 或 d]
            - "deep_prompt": 若 PROMPT.DEEP=True，则再保存每层 deep prompt，形状 [L-1, P, D 或 d]

        文件名：
            OUTPUT_DIR/prompt_ep{epoch}.pth
        """
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_module = self.model.enc.transformer
                out = {}

                prompt_embds = getattr(prompt_module, "prompt_embeddings", None)
                if prompt_embds is not None:
                    out["shallow_prompt"] = prompt_embds.cpu().numpy()

                if self.cfg.MODEL.PROMPT.DEEP and hasattr(prompt_module, "deep_prompt_embeddings"):
                    deep_embds = prompt_module.deep_prompt_embeddings
                    if deep_embds is not None:
                        out["deep_prompt"] = deep_embds.cpu().numpy()

                if not out:
                    logger.warning("No prompt parameters to save for this epoch; skipping dump.")
                    return

                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """
        在给定 data_loader（验证/测试集）上评估分类性能。

        参数：
            data_loader: DataLoader（val 或 test）
            prefix: 字符串前缀，用于标识当前评测类型（"val" 或 "test"）
            save: 若 True 且 cfg.MODEL.SAVE_CKPT=True，则保存 logits 与 targets，
                  并在 test 阶段额外缓存 CLS 特征用于 t-SNE 可视化。
        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target 聚合全量 logits 与 targets，评测结束一次性计算指标
        total_logits = []
        total_targets = []

        # ========== 遍历整个数据集 ==========
        for idx, input_data in enumerate(data_loader):
            self._trace_stage = f"eval_{prefix}"
            self._trace_iter = int(idx)
            self._trace_global_step += 1
            end = time.time()
            X, targets, attributes = self.get_input(input_data)

            # 统计数据加载时间
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))

            # 评测阶段：is_train=False → forward_one_batch 只做前向与 loss 计算
            loss, outputs = self.forward_one_batch(X, targets, False, attributes=attributes)
            if loss == -1:                # 出现 inf / NaN 时，直接停止
                return
            losses.update(loss, X.shape[0])

            # 统计 batch 时间
            batch_time.update(time.time() - end)

            # 周期性打印测试过程日志
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

            # targets: Tensor → Python list[int]
            total_targets.extend(list(targets.numpy()))
            # outputs: logits Tensor，先收集，最后再 cat
            # outputs 可能为 logits Tensor / (logits, aux) / {"logits": ...}
            # 统一提取 logits 以便后续 cat
            logits = outputs
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                logits = outputs[0]
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]

            total_logits.append(logits)

        # 整体评测日志
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))

        # 若模型使用了 side-tuning 分支，额外打印融合系数 alpha
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))

        # 拼接得到 (num_samples, num_classes) 的 logits 矩阵
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()

        # 调用 evaluator 计算分类指标（内部会根据 DATA.MULTILABEL 处理单/多标签场景）
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # ========== 若需要，则保存 logits 与 targets 到文件中 ==========
        if save and self.cfg.MODEL.SAVE_CKPT:
            # 1) 已有
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(f"Saved logits and targets for {test_name} at {out_path}")

        # ========== 若是 test 阶段且 save=True，则额外缓存 CLS 特征用于 t-SNE ==========
        if save and prefix == "test":
            os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "cache"), exist_ok=True)
            cache_dir = os.path.join(self.cfg.OUTPUT_DIR, "cache")

            # 1) 提取 CLS 特征（每类最多取 100 个样本，以免太大）
            X_cls, y = extract_features(
                self.model, data_loader, self.device,
                feat_type="cls", max_per_class=100
            )
            np.savez_compressed(
                os.path.join(cache_dir, f"{test_name}_cls.npz"),
                X=X_cls.astype("float32"),      # CLS 特征
                y=y,                            # 对应标签
                meta=dict(type="cls")           # 元信息
            )

            logger.info(f"[t-SNE cache] saved CLS features to {cache_dir}")
        # === eval_classifier 结束 ===
