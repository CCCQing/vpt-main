#!/usr/bin/env python3
import numpy as np

from collections import defaultdict
from typing import Union

from .eval import singlelabel
from ..utils import logging

# singlelabel.py:
# 当前 evaluator 的底层数学计算器
# 里面负责：
# - compute_top1
# - compute_per_class_top1
#
# 注意：
# 现在 evaluator 本身已经不再负责 seen/unseen candidate set 推断，
# 而只消费已经准备好的 scores / targets。
logger = logging.get_logger("visual_prompt")


class Evaluator:
    """
    Evaluator：负责三件事：

    1. 接收 trainer 传进来的：
       - probs / scores
       - targets
       - eval_name
       - protocol_role

    2. 调用 singlelabel 中的基础指标函数计算：
       - top1
       - per-class top1

    所以你可以把它理解成：
        “评测结果记账器 + 日志打印器”
    而不是“协议解释器”。
    """
    def __init__(self, task_type: str) -> None:
        self.results = defaultdict(dict)
        self.iteration = -1
        self.task_type = str(task_type).lower()
        if self.task_type not in {"zsl", "gzsl"}:
            raise ValueError("Evaluator task_type must be 'zsl' or 'gzsl', got '{}'".format(task_type))

    def update_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def update_result(self, metric: str, value: Union[float, dict]) -> None:
        key_name = f"epoch_{self.iteration}" if self.iteration > -1 else "final"
        if isinstance(value, float):
            self.results[key_name].update({metric: value})
        else:
            if metric in self.results[key_name]:
                self.results[key_name][metric].update(value)
            else:
                self.results[key_name].update({metric: value})

    def classify(self, probs, targets):
        """输入：
        - probs:
            模型输出的分数矩阵
            形状通常是 [N, C]
            注意这里虽然变量名叫 probs，
            但实际上可以是 logits / similarity scores，
            只要最后是“每个样本对每个类别的分数”即可。

        - targets:
            对应的标签
            注意：
            当前重构后，这里传进来的 targets 已经应该是：
                local / eval-local 空间下的标签
            而不是 raw global labels。"""
        scores = np.asarray(probs)
        targets_np = np.asarray(targets, dtype=np.int64)

        top1 = singlelabel.compute_top1(scores, targets_np)["top1"]
        per_class = singlelabel.compute_per_class_top1(scores, targets_np)
        return {"top1": top1, "per_class": per_class}

    def log_and_update(self, log_results, save_results, eval_name):
        self.update_result("classification", {eval_name: save_results})
