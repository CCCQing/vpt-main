#!/usr/bin/env python3
"""
GraphProbPrior 温度系数短跑诊断脚本。

这个脚本服务于“版本一”的离线诊断：
    不改正式训练日志节奏，单独跑若干个 batch 或 1 个 epoch，
    把 GraphProbPrior 里各个温度系数处理后的数值分布导出成 CSV/JSON。

它主要回答这类问题：
1. TAU_GRAPH 作用前后的 graph 相似度和 neighbor softmax 是否过尖/过平。
2. TAU_LATENT 作用前后的 latent distance softmax 是否过尖/过平。
3. TAU_PRIOR relation regularization 的 prior 间关系是否过尖/过平。
4. class_aggregate_mmd 模式下 RBF kernel 的距离和核值是否落在合理范围。
5. 每个 batch 的 loss 与上述监测量是否一起出现异常。

典型使用方式：
    python src/tools/diagnose_graph_prob_prior_temperatures.py \
        --config-file configs/prompt/cub.yaml \
        --max-batches 300 \
        MODEL.GRAPH_PROB_PRIOR.ENABLE True \
        MODEL.GRAPH_PROB_PRIOR.MODE graph_conditioned_semantic_prior \
        MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT 0.001

输出文件：
1. graph_prob_prior_temperature_diagnosis.csv
   - 每一行对应一个 batch。
   - 保存该 batch 的 GraphProbPrior loss stats 和 monitor stats。

2. graph_prob_prior_temperature_diagnosis.json
   - 保存整体汇总统计，例如各监测量的均值、分位数等。
   - 同时记录本次使用的温度配置，方便之后对比实验。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # 允许从仓库根目录外直接运行本脚本时，也能 import src.* 和 train.py。
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import src.utils.logging as logging
from src.utils import distributed as du
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.solver.graph_prob_prior_monitors import aggregate_monitor_rows
from train import _merge_local_path_cfg_if_exists, _sync_xlsa_protocol


GRAPH_PROB_PRIOR_MODES = [
    "true_class_kl",
    "graph_conditioned_semantic_prior",
    "class_aggregate_moment",
    "class_aggregate_mmd",
    "factorized_latent",
    "dual_metric_semantic_distribution",
]


def parse_args():
    """
    解析命令行参数。

    这里把脚本分成两类输入：
    1. 脚本自己的控制参数：
       - --config-file：基础实验配置。
       - --max-batches：最多采集多少个训练 batch。
       - --output-dir：诊断结果保存目录。
       - --no-train-step：只前向和算 loss，不更新模型。

    2. 透传给 cfg 的配置覆盖：
       - opts：末尾的 KEY VALUE KEY VALUE。
       - 例如 MODEL.GRAPH_PROB_PRIOR.TAU_LATENT 0.5。
    """
    parser = argparse.ArgumentParser(description="Run a short GraphProbPrior temperature diagnosis.")
    parser.add_argument("--config-file", required=True, help="Path to experiment yaml.")
    parser.add_argument("--max-batches", type=int, default=300, help="Number of train batches to run; <=0 means one full epoch.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for diagnosis outputs. Defaults to <cfg.OUTPUT_DIR>/graph_prob_prior_temperature_diagnosis.",
    )
    parser.add_argument(
        "--no-train-step",
        action="store_true",
        help="Only run forward/loss collection without optimizer update. This keeps model weights fixed.",
    )
    parser.add_argument(
        "--modes",
        default="",
        help="Comma-separated GraphProbPrior modes to run, or 'all'. Empty means use cfg.MODEL.GRAPH_PROB_PRIOR.MODE.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional config overrides in KEY VALUE form.")
    return parser.parse_args()


def parse_modes(raw: str) -> List[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    if raw.lower() == "all":
        return list(GRAPH_PROB_PRIOR_MODES)
    modes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    bad = [mode for mode in modes if mode not in GRAPH_PROB_PRIOR_MODES]
    if bad:
        raise ValueError(f"Unsupported --modes values: {bad}. Expected one of {GRAPH_PROB_PRIOR_MODES} or all.")
    return modes


def run_multiple_modes(args, modes: List[str]) -> None:
    base_cfg = setup_cfg(args)
    base_output = Path(args.output_dir) if args.output_dir else Path(base_cfg.OUTPUT_DIR) / "all_modes"
    base_output.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in modes:
        mode_output = base_output / mode
        child_opts = list(args.opts) + [
            "MODEL.GRAPH_PROB_PRIOR.MODE",
            mode,
        ]
        if mode in {"factorized_latent", "dual_metric_semantic_distribution"}:
            child_opts.extend([
                "MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_ENABLE",
                "True",
            ])

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config-file",
            str(args.config_file),
            "--max-batches",
            str(args.max_batches),
            "--output-dir",
            str(mode_output),
        ]
        if args.no_train_step:
            cmd.append("--no-train-step")
        cmd.extend(child_opts)

        print(f"[multi-mode] running {mode}: {' '.join(cmd)}", flush=True)
        started = subprocess.run(cmd, cwd=str(ROOT))
        results.append({
            "mode": mode,
            "output_dir": str(mode_output),
            "returncode": int(started.returncode),
        })
        if started.returncode != 0:
            break

    summary_path = base_output / "all_modes_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"modes": results}, f, indent=2, sort_keys=True)
    failed = [item for item in results if item["returncode"] != 0]
    if failed:
        raise SystemExit(f"At least one mode failed. See {summary_path}")
    print(f"[multi-mode] wrote summary: {summary_path}", flush=True)


def setup_cfg(args):
    """
    构建诊断用 cfg。

    和正式训练相比，这里额外做三件事：
    1. 读取本地路径配置 local_path.yaml：
       - 保证数据路径、缓存路径仍然沿用当前机器的本地设置。

    2. 强制打开 GraphProbPrior monitor：
       - MONITOR_ENABLE=True 表示 GraphProbPrior forward 时会生成监测量。
       - MONITOR_EVERY_N=1 表示每个 batch 都采集，不跳 batch。

    3. 检查 GraphProbPrior 是否真的启用：
       - 如果 ENABLE=False 或 LOSS_WEIGHT<=0，本脚本没有诊断对象，直接报错。
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    _merge_local_path_cfg_if_exists(cfg)
    cfg.merge_from_list(args.opts)
    _sync_xlsa_protocol(cfg)

    # 诊断脚本需要每个 batch 都拿到温度相关监测量，所以这里覆盖正式训练里的监测频率。
    cfg.MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE = True
    cfg.MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N = 1
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    else:
        # 默认把诊断输出放到当前实验输出目录下面，避免污染正式训练根目录。
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "graph_prob_prior_temperature_diagnosis")

    if not bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE):
        raise ValueError("MODEL.GRAPH_PROB_PRIOR.ENABLE must be True for temperature diagnosis.")
    if float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT) <= 0.0:
        raise ValueError("MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT must be > 0 for temperature diagnosis.")
    return cfg


def build_train_loader(cfg):
    """
    根据 XLSA 协议选择训练 loader。

    dev 协议：
        使用 train split，训练后通常还会用 val_unseen 做验证。

    final_zsl / final_gzsl 协议：
        使用 trainval split，因为 final 协议通常没有 dev validation，
        训练时会把 train 和 val 合起来作为最终训练集。
    """
    protocol_mode = str(cfg.DATA.XLSA.PROTOCOL_MODE).lower()
    if protocol_mode == "dev":
        return data_loader.construct_train_loader(cfg)
    if protocol_mode in {"final_zsl", "final_gzsl"}:
        return data_loader.construct_trainval_loader(cfg)
    raise ValueError(f"Unsupported DATA.XLSA.PROTOCOL_MODE='{cfg.DATA.XLSA.PROTOCOL_MODE}'.")


def _to_percent(value: float) -> float:
    return float(value) * 100.0


def _harmonic_mean(a: float, b: float) -> float:
    a = float(a)
    b = float(b)
    if a + b <= 0.0:
        return 0.0
    return float(2.0 * a * b / (a + b + 1e-8))


def evaluate_after_diagnosis(cfg, trainer: Trainer, logger) -> Dict[str, float]:
    """
    在短跑训练结束后补一次正式评测。

    这个函数对应“方案 1”的核心：
    - 前面仍然只跑 max_batches 个训练 batch，用来收集 GraphProbPrior 监测量；
    - 训练截断后，不再继续完整 epoch，而是直接跑一次当前模型的评测；
    - final_gzsl 协议下同时评测 test_seen / test_unseen，并计算 H。

    返回值里的准确率统一使用百分数，例如 52.3 表示 52.3%。
    这样生成的字段和之前 grid summary 里的 gzsl_seen_last/gzsl_unseen_last/gzsl_h_last 口径一致。
    """
    protocol_mode = str(cfg.DATA.XLSA.PROTOCOL_MODE).lower()
    trainer.model.eval()
    trainer.evaluator.update_iteration(0)

    performance: Dict[str, float] = {
        "eval_ran": 1.0,
    }

    if protocol_mode == "final_gzsl":
        logger.info("Running post-diagnosis GZSL eval: test_seen + test_unseen")
        test_seen_loader = data_loader.construct_test_seen_loader(cfg)
        test_unseen_loader = data_loader.construct_test_unseen_loader(cfg)
        seen_metrics = trainer.eval_classifier(test_seen_loader, "test_seen")
        unseen_metrics = trainer.eval_classifier(test_unseen_loader, "test_unseen")

        seen = float(seen_metrics["gzsl_seen"])
        unseen = float(unseen_metrics["gzsl_unseen"])
        h = _harmonic_mean(seen, unseen)
        trainer._update_gzsl_record_metrics(0, test_unseen_loader, seen_metrics, unseen_metrics)

        performance.update(
            {
                "gzsl_seen_last": _to_percent(seen),
                "gzsl_unseen_last": _to_percent(unseen),
                "gzsl_h_last": _to_percent(h),
                "gzsl_seen_best": _to_percent(seen),
                "gzsl_unseen_best": _to_percent(unseen),
                "gzsl_h_best": _to_percent(h),
                "eval_seen_top1": _to_percent(float(seen_metrics["top1"])),
                "eval_unseen_top1": _to_percent(float(unseen_metrics["top1"])),
            }
        )
        logger.info(
            "Post-diagnosis GZSL eval: seen=%.2f unseen=%.2f H=%.2f",
            performance["gzsl_seen_last"],
            performance["gzsl_unseen_last"],
            performance["gzsl_h_last"],
        )
        return performance

    if protocol_mode == "final_zsl":
        logger.info("Running post-diagnosis ZSL eval: test_unseen")
        test_unseen_loader = data_loader.construct_test_unseen_loader(cfg)
        unseen_metrics = trainer.eval_classifier(test_unseen_loader, "test_unseen")
        zsl_unseen = _to_percent(float(unseen_metrics["zsl_unseen"]))
        performance.update(
            {
                "zsl_unseen_last": zsl_unseen,
                "zsl_unseen_best": zsl_unseen,
                "eval_unseen_top1": _to_percent(float(unseen_metrics["top1"])),
            }
        )
        logger.info("Post-diagnosis ZSL eval: unseen=%.2f", zsl_unseen)
        return performance

    if protocol_mode == "dev":
        logger.info("Running post-diagnosis dev eval: val_unseen")
        val_loader = data_loader.construct_val_loader(cfg)
        val_metrics = trainer.eval_classifier(val_loader, "val_unseen")
        dev_unseen = _to_percent(float(val_metrics["dev_unseen"]))
        performance.update(
            {
                "dev_unseen_last": dev_unseen,
                "dev_unseen_best": dev_unseen,
                "eval_unseen_top1": _to_percent(float(val_metrics["top1"])),
            }
        )
        logger.info("Post-diagnosis dev eval: val_unseen=%.2f", dev_unseen)
        return performance

    raise ValueError(f"Unsupported DATA.XLSA.PROTOCOL_MODE='{cfg.DATA.XLSA.PROTOCOL_MODE}'.")


def collect_graph_prob_prior_stats(stats: Dict[str, float]) -> Dict[str, float]:
    """
    从 loss 的 _last_loss_stats 中筛出本脚本关心的字段。

    CompositeLoss / GraphProbPriorAuxLoss 会把很多诊断量放到 _last_loss_stats。
    这里不把所有字段都写进 CSV，只保留：
    1. graph_prob_prior_*：
       - GraphProbPrior loss 本身。
       - GraphProbPrior monitor 产生的温度统计量。

    2. ce_loss：
       - 主分类交叉熵，用来观察温度异常是否伴随主任务 loss 异常。
    """
    keep_prefixes = (
        "graph_prob_prior_",
        "ce_loss",
    )
    row = {}
    for key, value in stats.items():
        if key.startswith(keep_prefixes) and isinstance(value, (int, float)):
            row[key] = float(value)
    return row


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    """
    把逐 batch 监测结果写成 CSV。

    rows 中每个 batch 可能拥有不同字段：
        例如 graph_conditioned_semantic_prior 有 TAU_LATENT 监测，
        class_aggregate_mmd 有 MMD kernel 监测，
        true_class_kl 可能只有真类 KL 监测。

    因此这里先合并所有 row 的 key，再统一作为 CSV 表头。
    缺失的字段会在对应 batch 行里留空。
    """
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    """
    脚本主流程。

    整体顺序：
    1. 解析命令行并构建 cfg。
    2. 构建训练 loader。
    3. 构建模型、Evaluator、Trainer。
    4. 逐 batch 跑 forward/loss。
    5. 从 cls_criterion._last_loss_stats 抽取 GraphProbPrior 监测量。
    6. 写出逐 batch CSV 和整体 JSON 汇总。

    注意：
        这里复用 Trainer.forward_one_batch，而不是手写模型 forward。
        这样可以确保诊断路径和正式训练路径尽量一致。
    """
    args = parse_args()
    modes = parse_modes(args.modes)
    if modes:
        run_multiple_modes(args, modes)
        return

    cfg = setup_cfg(args)
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.setup_logging(cfg.NUM_GPUS, cfg.NUM_SHARDS, output=str(output_dir), name="visual_prompt")
    logger = logging.get_logger("visual_prompt")

    if cfg.SEED is not None:
        # 固定随机种子，方便多次诊断同一配置时比较数值变化。
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    train_loader = build_train_loader(cfg)
    if len(train_loader) <= 0:
        raise RuntimeError("Train loader is empty; cannot run GraphProbPrior diagnosis.")
    logger.info("Constructing model for GraphProbPrior temperature diagnosis...")
    model, cur_device = build_model(cfg)
    class_attr = train_loader.dataset.class_attributes
    if class_attr is None:
        raise ValueError("Train dataset must expose class_attributes for GraphProbPrior diagnosis.")
    if hasattr(model, "attach_r_similarity_head"):
        # GraphProbPrior / 语义图相关 loss 依赖类别属性向量，
        # 这里和正式训练一样把 class_attributes 挂到模型的 r_similarity head。
        model.attach_r_similarity_head(class_attr)

    evaluator = Evaluator(task_type=str(cfg.SOLVER.EVAL_MODE).lower())
    trainer = Trainer(cfg, model, evaluator, cur_device)
    trainer.cls_criterion.to(cur_device)
    trainer.cls_weights = train_loader.dataset.get_class_weights(cfg.DATA.CLASS_WEIGHTS_TYPE)
    trainer.model.train()

    rows: List[Dict[str, float]] = []
    max_batches = int(args.max_batches)
    logger.info(
        "Running GraphProbPrior temperature diagnosis: mode=%s max_batches=%s train_step=%s output=%s",
        str(cfg.MODEL.GRAPH_PROB_PRIOR.MODE),
        "full_epoch" if max_batches <= 0 else str(max_batches),
        str(not bool(args.no_train_step)),
        str(output_dir),
    )

    target_batches = len(train_loader) if max_batches <= 0 else max_batches
    epoch_idx = 0
    while len(rows) < target_batches:
        if epoch_idx > 0:
            data_loader.shuffle(train_loader, epoch_idx)
        for iter_idx, input_data in enumerate(train_loader):
            if len(rows) >= target_batches:
                break
            # 这些 trace 字段主要给 Trainer 内部的调试/日志逻辑使用。
            # 现在 max_batches 可以跨过一个完整 loader，因此这里记录真实的短跑 epoch / iter。
            trainer._trace_epoch = int(epoch_idx)
            trainer._trace_iter = int(iter_idx)
            trainer._trace_global_step += 1

            x, targets, attributes = trainer.get_input(input_data)
            if args.no_train_step:
                # 只采集当前模型状态下的温度监测量，不更新参数。
                # 适合想看“初始化/已有 checkpoint 的静态分布”的情况。
                trainer.model.train()
                with torch.enable_grad():
                    loss, _ = trainer.forward_one_batch(
                        x,
                        targets,
                        False,
                        attributes=attributes,
                        dataset=train_loader.dataset,
                    )
            else:
                # 默认走一次完整训练 step：
                # forward -> loss -> backward -> optimizer step。
                # 这样采集到的是“模型边训练边变化”的温度统计。
                loss, _ = trainer.forward_one_batch(
                    x,
                    targets,
                    True,
                    attributes=attributes,
                    dataset=train_loader.dataset,
                )

            batch_number = len(rows) + 1
            # GraphProbPriorLossComputer 每次 forward 后会把监测量合并到
            # cls_criterion._last_loss_stats；这里把它抽出来形成一行 CSV。
            stats = collect_graph_prob_prior_stats(getattr(trainer.cls_criterion, "_last_loss_stats", {}))
            stats.update(
                {
                    "batch": float(batch_number),
                    "epoch": float(epoch_idx + 1),
                    "iter": float(iter_idx + 1),
                    "loss": float(loss.detach().item()),
                }
            )
            rows.append(stats)
            if batch_number % max(1, int(cfg.SOLVER.LOG_EVERY_N)) == 0:
                logger.info(
                    "diagnosis batch %d/%d epoch=%d iter=%d loss=%.6f monitor_keys=%d",
                    batch_number,
                    target_batches,
                    epoch_idx + 1,
                    iter_idx + 1,
                    float(loss.detach().item()),
                    len(stats),
                )
        epoch_idx += 1

    if not rows:
        raise RuntimeError("No diagnosis rows collected. Check the train loader and max-batches.")

    performance = evaluate_after_diagnosis(cfg, trainer, logger)

    if not du.is_master_process(cfg.NUM_GPUS):
        return

    csv_path = output_dir / "graph_prob_prior_temperature_diagnosis.csv"
    json_path = output_dir / "graph_prob_prior_temperature_diagnosis.json"
    write_csv(csv_path, rows)

    # aggregate_monitor_rows 会对每个数值字段做整体统计，
    # 方便不打开 CSV 时也能快速看全局均值、分位数、范围等。
    summary = aggregate_monitor_rows(rows)
    summary.update(performance)
    payload = {
        "config_file": args.config_file,
        "opts": list(args.opts),
        "num_batches": len(rows),
        "mode": str(cfg.MODEL.GRAPH_PROB_PRIOR.MODE),
        "performance": performance,
        "temperatures": {
            # 记录本次温度配置，避免之后只看 JSON 不知道当时跑的是哪组温度。
            "TAU_GRAPH": float(cfg.MODEL.GRAPH_PROB_PRIOR.TAU_GRAPH),
            "TAU_LATENT": float(cfg.MODEL.GRAPH_PROB_PRIOR.TAU_LATENT),
            "TAU_PRIOR": float(cfg.MODEL.GRAPH_PROB_PRIOR.TAU_PRIOR),
            "MMD_SIGMA": float(cfg.MODEL.GRAPH_PROB_PRIOR.MMD_SIGMA),
            "DUAL_TAU_NEG": float(cfg.MODEL.GRAPH_PROB_PRIOR.DUAL_TAU_NEG),
            "DUAL_NEG_TOPK": int(cfg.MODEL.GRAPH_PROB_PRIOR.DUAL_NEG_TOPK),
            "TAU_ACC": float(cfg.MODEL.SEMANTIC_GRAPH.TAU_ACC),
            "TARGET_MIX_ALPHA": float(cfg.MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA),
        },
        "csv_path": str(csv_path),
        "summary": summary,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    logger.info("Wrote diagnosis CSV: %s", csv_path)
    logger.info("Wrote diagnosis JSON: %s", json_path)


if __name__ == "__main__":
    main()
