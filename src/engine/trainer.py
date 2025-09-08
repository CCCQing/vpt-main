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

        # solver related
        # ---------- 优化器 / 学习率调度器 / 损失函数 ----------
        logger.info("\tSetting up the optimizer...")
        # 这里传入 [self.model] 以兼容可能的多模型场景
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # 分类损失构建器（可返回 CE / Focal / LDAM 等，视配置而定）
        self.cls_criterion = build_loss(self.cfg)
        # ---------- Checkpointer（保存/加载权重的统一入口） ----------
        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
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

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given data loader.
           前向一次（一个 batch），并在训练阶段执行反向与优化步
        Args:
            X: input dict
            targets
            is_train: bool，训练阶段为 True，验证/测试为 False
        Returns:
            loss
            outputs: output logits（形状 [B, num_cls]）
        """
        # move data to device ---------- 数据搬运到设备 ----------
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward ---------- 前向 ---------- set_grad_enabled：仅在训练阶段计算梯度，评测阶段节省显存与算力
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))
            # 一些“局部损失”（is_local=True）需要拿到 model / inputs 参与计算
            # 例如某些正则或提示学习的约束，且为了稳定会在 eval() 下运行
            if self.cls_criterion.is_local() and is_train:
                self.model.eval() # 临时切 eval，避免 BN/Dropout 带来随机性
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local(): # 评测阶段且为“局部损失”，此处直接返回占位 loss=1（无实际意义）
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(      # 常规损失（如交叉熵）：只需 (outputs, targets, class_weights)
                    outputs, targets, self.cls_weights)
            # ---------- 异常值检测 ----------
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
        # ---------- 训练阶段：反向与优化 ----------
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        """
        将 dataloader 返回的数据 dict 统一转为 Tensor，并拆分出输入与标签。
        期望输入字典结构：
            data["image"] : np.ndarray 或 Tensor
            data["label"] : np.ndarray 或 Tensor
        """
        if not isinstance(data["image"], torch.Tensor): # 若是 numpy，则统一转为 torch
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()  # 保证 float（模型一般期望 float）
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch 以 epoch 为单位进行分类器训练与评测。
        """
        # save the model prompt if required before training 如需保存 “prompt embedding”，在训练开始前保存一次“epoch 0 前”的状态
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params ---------- 训练超参 ----------
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        # 若干计量器（统计平均损失/时间等，便于打印）
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        # 类别权重（用于不均衡数据的损失加权）
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training 早停计数器；若超过 cfg.SOLVER.PATIENCE 则停止训练

        # ================== 主训练循环（按 epoch） ==================
        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results 每个 epoch 重置平均器
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode 切训练态（启用 Dropout / 更新 BN 统计等）
            self.model.train()

            end = time.time()
            # ---------- 遍历一个 epoch 的 batch ----------
            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations # 调试模式：仅跑前 20 个 batch 以加速
                    break
                
                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end) # 统计数据加载时间（I/O 与预处理）

                train_loss, _ = self.forward_one_batch(X, targets, True)    # 前向 + 反向（训练阶段）

                if train_loss == -1:
                    # continue  若出现 nan/inf，被 forward_one_batch 拦截，此处直接返回
                    return None

                losses.update(train_loss.item(), X.shape[0])    # 更新平均损失

                # measure elapsed time
                batch_time.update(time.time() - end)            # 统计一个 batch 的时间
                end = time.time()

                # log during one batch 每隔 log_interval 个 batch 打印一次训练日志
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val          # 估算剩余时间（本 epoch 剩余 + 后续 epoch）
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
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
            # 一个 epoch 的汇总日志
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
             # 按官方建议：scheduler.step() 应在 optimizer.step() 之后调用
            self.scheduler.step()

            # Enable eval mode ---------- 验证 / 测试 ----------
            self.model.eval()
            # 保存当下 epoch 的 prompt embeddings（如需求与配置指定）
            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training # 更新 evaluator 的 epoch 号，评测时用于结果归档到 epoch_k 下
            # self.evaluator.update_iteration(epoch)
            # self.eval_classifier(val_loader, "val", epoch == total_epoch - 1) # 验证集评测（prefix="val"）

            # 20250902改动：可视化实现，确保拿到最佳 checkpoint 的图
            # 先评 val（不保存）
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", save=False)

            # 读取本轮 val 的 top1，判断是否刷新最佳
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return
            improved = curr_acc > best_metric
            # 只有刷新最佳时，才在 test 上触发 save=True（从而在 eval_classifier 内部落盘 logits/CLS 等缓存）
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
            if improved:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
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
        按配置保存（VPT）prompt embeddings，便于后续分析或可视化。
        保存条件： - cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH 为 True
                 - 模型类型为 ViT（cfg.MODEL.TYPE == "vit"）
                 - 迁移方式包含 "prompt"（即使用了提示调优）
        保存内容： - shallow_prompt : 形状 [1, P, D 或 d]
                 - deep_prompt    : 若开启 DEEP，则还包含 [L-1, P, D 或 d]
        保存路径： OUTPUT_DIR/prompt_ep{epoch}.pth
        """
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """
        evaluate classifier
        在给定数据集上进行评测（不计算梯度）。
        参数：data_loader : 验证/测试集的 DataLoader
             prefix      : "val" 或 "test"（用于日志前缀与结果键名）
             save        : 若为 True 且开启 SAVE_CKPT，则保存 logits 与 targets"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target 聚合全量 logits 与 targets，评测结束一次性计算指标
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time 统计数据时间
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            # 前向（评测阶段）：不做反向
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time 统计 batch 时间
            batch_time.update(time.time() - end)
            # 间隔打印
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

            # targets: List[int] 收集标签与 logits，targets 为 Tensor，这里转为 python 列表（int）
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        # 评测阶段总体日志
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        # 若模型使用了 sidetune 分支，额外打印融合系数 alpha
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes 拼接得到 (num_samples, num_classes) 的 logits
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        # 调用 evaluator 计算分类指标（会根据 cfg.DATA.MULTILABEL 走单/多标签分支）
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets （可选）保存 logits 与 targets 便于后处理/分析
        if save and self.cfg.MODEL.SAVE_CKPT:
            # 1) 已有
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(f"Saved logits and targets for {test_name} at {out_path}")

        if save and prefix == "test":
            os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "cache"), exist_ok=True)
            cache_dir = os.path.join(self.cfg.OUTPUT_DIR, "cache")

            # 1) CLS 特征（每类最多 100，避免样本过多）
            X_cls, y = extract_features(
                self.model, data_loader, self.device,
                feat_type="cls", max_per_class=100
            )
            np.savez_compressed(
                os.path.join(cache_dir, f"{test_name}_cls.npz"),
                X=X_cls.astype("float32"), y=y, meta=dict(type="cls")
            )

            logger.info(f"[t-SNE cache] saved CLS features to {cache_dir}")
        # === 结束 ===

