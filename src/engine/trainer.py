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

        # ========== 2. 前向推理（训练时开启梯度，验证/测试禁用梯度） ==========
        with torch.set_grad_enabled(is_train):
            if attributes is not None:
                outputs = self.model(inputs, semantics=attributes)  # (batchsize, num_cls)
            else:
                outputs = self.model(inputs)  # (batchsize, num_cls)

            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

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
                    outputs, targets, self.cls_weights)

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
            self.optimizer.step()

        return loss, outputs

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
        best_epoch = -1                                 # 当前最优 epoch
        best_metric = 0                                 # 最优指标（比如 top1）
        log_interval = self.cfg.SOLVER.LOG_EVERY_N      # 每多少个 batch 打一次日志

        # 若干计量器（统计平均损失/时间等，便于打印）
        losses = AverageMeter('Loss', ':.4e')           # - losses: 每个 epoch 内的平均训练损失
        batch_time = AverageMeter('Time', ':6.3f')      # - batch_time: 每个 batch 的时间
        data_time = AverageMeter('Data', ':6.3f')       # - data_time: 数据加载时间

        # 从训练集 Dataset 获取类别权重，传给损失函数（应对类分布不平衡）
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")

        # 早停用 patience：若验证集 metric 连续若干次不提升就停止
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training 早停计数器；若超过 cfg.SOLVER.PATIENCE 则停止训练

        # ================== 2. 主训练循环（按 epoch） ==================
        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results 每个 epoch 开始前，重置统计量
            losses.reset()
            batch_time.reset()
            data_time.reset()

            # 当前学习率（假设 scheduler 里第一组 lr 代表全局 lr）
            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode 切换到训练模式（启用 Dropout / 更新 BN 统计等）
            self.model.train()

            end = time.time()

            # ---------- 遍历一个 epoch 的所有 batch ----------
            for idx, input_data in enumerate(train_loader):
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

                # measure elapsed time 统计 batch 处理时间
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch 每隔 log_interval 个 batch 打印一次训练日志
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    # 估算剩余时间（本 epoch 剩余 + 后续 epoch）
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

            # 读取本轮 val 的 top1，判断是否刷新最佳
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                # 若指标缺失（可能是评测流程问题），直接返回
                return

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
            total_logits.append(outputs)

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

