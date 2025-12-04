#!/usr/bin/env python3
import numpy as np

from collections import defaultdict
from typing import List, Union

# 引入同目录下的评测实现：
# - singlelabel：单标签（每个样本只有 1 个类别）评测（Top-1/Top-5、ROC-AUC 等）
# - multilabel：多标签（每个样本可属于多个类别）评测（AP、mAP、F1 等）
from .eval import multilabel
from .eval import singlelabel
from ..utils import logging
logger = logging.get_logger("visual_prompt")


class Evaluator():
    """
    An evaluator with below logics:1. find which eval module to use.2. store the eval results, pretty print it in log file as well.
    评测器（Evaluator）
    ==================
    负责两件事：
      1) 选择并调用合适的评测流程（单标签 / 多标签）。
      2) 组织与存储评测结果，并以易读格式写入日志。
    结果组织形式：
      self.results 是一个两层字典：
        {
          "epoch_0": {"classification": {...不同 eval_type 的结果...}},
          "epoch_1": {...},
          ...
          "final": {...}  # 若未设置迭代/epoch，则全部记到 final 下
        }
    """

    def __init__(
            self,
            seen_classes: np.ndarray = None,
            unseen_classes: np.ndarray = None,
            task_type: str = "standard",
    ) -> None:
        # 用 defaultdict(dict) 便于后续直接按 key 累加子项
        self.results = defaultdict(dict)
        # 当前迭代/轮次（外部可在每个 epoch 开头更新）；默认 -1 表示“尚未设定”，落到 'final'
        self.iteration = -1
        # 多标签评测中，搜索“最佳 F1 阈值”的上界（闭区间的右端点），默认 0.5
        # 例如从 0.0 ~ 0.5 的阈值范围内搜索能使 F1 最优的分割点
        self.threshold_end = 0.5
        # ZSL / GZSL 评测所需的类划分与任务类型（standard / zsl / gzsl）
        self.task_type = task_type
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes

    def update_iteration(self, iteration: int) -> None:
        """update iteration info
        由训练循环在合适时机调用，更新当前迭代/epoch 信息。"""
        self.iteration = iteration

    def update_result(self, metric: str, value: Union[float, dict]) -> None:
        """
        将某项评测结果写入结果字典。
        参数     metric : 结果类别名（如 "classification"）
                value  : 可为标量（float）或字典（dict）
                         - 若为 float，直接以 {metric: value} 写入
                         - 若为 dict，按子键进行 merge（便于多次追加不同 eval_type）
        """
        # 根据是否设置了 iteration，决定存到哪一层 key
        if self.iteration > -1:
            key_name = "epoch_" + str(self.iteration)
        else:
            key_name = "final"
        if isinstance(value, float):
            # 直接覆盖/写入一个标量指标
            self.results[key_name].update({metric: value})
        else:
            # dict 情况需要 merge，避免覆盖其它 eval_type 的子结果
            if metric in self.results[key_name]:
                self.results[key_name][metric].update(value)
            else:
                self.results[key_name].update({metric: value})

    def classify(self, probs, targets, test_data, multilabel=False):
        """
        Evaluate classification result. 分类评测入口（单/多标签统一入口）
        Args: 参数
            probs: np.ndarray for num_data x num_class, predicted probabilities
                   np.ndarray，形状 [num_samples, num_classes]
                    - 单标签：通常是每类的 softmax/sigmoid 分数（或 logits 也可，内部会按需要处理）
                    - 多标签：通常是每类的 sigmoid 概率
            targets: np.ndarray for multilabel, list of integers for single label
                    - 单标签：List[int]，每个样本的类别 id
                    - 多标签：List[List[int]] 或等价结构（每个样本多个类别 id）
            test_labels:  map test image ids to a list of class labels
            test_data : str 或其它可打印对象，用作“评测类型名 / 数据分割名”的标记（例如 "val"、"test"、"trainval"）
            multilabel : bool，是否走多标签评测逻辑 根据 multilabel 分支到不同的评测函数。
        """
        if not targets:
            raise ValueError(
                "When evaluating classification, need at least give targets")

        if multilabel:
            self._eval_multilabel(probs, targets, test_data)
        else:
            if self.task_type in ["zsl", "gzsl"]:
                self._eval_zsl_gzsl_singlelabel(probs, targets, test_data)
            else:
                self._eval_singlelabel(probs, targets, test_data)

    def _eval_zsl_gzsl_singlelabel(self, scores: np.ndarray, targets: List[int], eval_type: str) -> None:
        """
        ZSL / GZSL 评测：
          - ZSL：仅 unseen 类别上的 per-class top1。
          - GZSL：seen/unseen 上的 per-class top1 及调和平均 H。
        """
        if self.seen_classes is None or self.unseen_classes is None:
            logger.warning("ZSL/GZSL evaluation requires seen_classes and unseen_classes. Falling back to standard metrics.")
            self._eval_singlelabel(scores, targets, eval_type)
            return

        metrics = singlelabel.compute_zsl_gzsl_metrics(
            scores,
            np.asarray(targets, dtype=np.int64),
            np.asarray(self.seen_classes, dtype=np.int64),
            np.asarray(self.unseen_classes, dtype=np.int64),
        )

        # 日志用百分比显示、保留两位小数；存盘保持原始小数（0~1）
        log_results = {k: np.around(v * 100, decimals=2) for k, v in metrics.items()}
        save_results = metrics

        self.log_and_update(log_results, save_results, eval_type)

    def _eval_singlelabel(self,scores: np.ndarray,targets: List[int],eval_type: str) -> None:
        """
        if number of labels > 2:
            top1 and topk (5 by default) accuracy 返回 top1 / topk（默认 top5）准确率
        if number of labels == 2:
            top1 and rocauc 返回 top1 与 ROC-AUC
        scores :[N,C]，每类分数 ；targets:[N]，整型类别id ；eval_type:评测类型名（如 "val", "test"），用于日志打印与结果归档
        """
        acc_dict = singlelabel.compute_acc_auc(scores, targets) # 返回一个字典，如 {"top1":..., "top5":...} 或 {"top1":..., "auc":...}
        # 日志用百分比显示、保留两位小数；存盘保持原始小数（0~1）
        log_results = {
            k: np.around(v * 100, decimals=2) for k, v in acc_dict.items()
        }
        save_results = acc_dict

        self.log_and_update(log_results, save_results, eval_type)

    def _eval_multilabel(self,scores: np.ndarray,targets: np.ndarray,eval_type: str) -> None:
        """
        多标签评测：典型指标：
            - AP（per-class Average Precision）、mAP（mean AP）
            - AR（per-class Average Recall）、mAR（mean AR）
            - F1：在给定阈值范围内寻找使 F1 最优的阈值（per-class 与 macro）
        """
        num_labels = scores.shape[-1]
        # 将 targets 转成 multi-hot 矩阵，形状 [N, C]，便于逐类/逐样本对齐评测
        targets = multilabel.multihot(targets, num_labels)

        log_results = {}
        # 计算 AP/AR 及其均值（mAP/mAR）
        ap, ar, mAP, mAR = multilabel.compute_map(scores, targets)
        # 在阈值区间 [0.0, self.threshold_end] 内搜索最佳 F1（可返回 macro/micro 或 per-class，视实现而定）
        f1_dict = multilabel.get_best_f1_scores(
            targets, scores, self.threshold_end)
        # 日志以百分比显示
        log_results["mAP"] = np.around(mAP * 100, decimals=2)
        log_results["mAR"] = np.around(mAR * 100, decimals=2)
        log_results.update({
            k: np.around(v * 100, decimals=2) for k, v in f1_dict.items()})
        # 完整结果（用于保存/记录）：包含 per-class 的 ap/ar、整体的 mAP/mAR，以及 F1 指标集合
        save_results = {
            "ap": ap, "ar": ar, "mAP": mAP, "mAR": mAR, "f1": f1_dict
        }
        self.log_and_update(log_results, save_results, eval_type)

    def log_and_update(self, log_results, save_results, eval_type):
        """
        将评测结果写到日志，并同步存入 self.results。
        参数
        log_results  : 已转成便于阅读的结果（百分比字符串/数值），用于打印
        save_results : 保持原始数值的结果（0~1 浮点、ndarray 等），用于结果归档
        eval_type    : 评测类型名（如 "val"、"test"），会作为 classification 的子键
        """
        log_str = ""
        for k, result in log_results.items():
            # 若是标量，按 %.2f 输出；若是数组/列表，直接打印列表形式
            if not isinstance(result, np.ndarray):
                log_str += f"{k}: {result:.2f}\t"
            else:
                log_str += f"{k}: {list(result)}\t"
        logger.info(f"Classification results with {eval_type}: {log_str}")
        # save everything
        # 保存到结果字典中，结构为：
        # self.results[epoch_or_final]["classification"][eval_type] = save_results
        self.update_result("classification", {eval_type: save_results})
