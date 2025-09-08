#!/usr/bin/env python3

"""Logging.
本模块封装了工程统一的日志系统（基于 Python 标准库 logging），
支持：
- 仅主进程输出到控制台（多进程/分布式时屏蔽非主进程的 print/log）
- 同时输出到文件并缓存文件句柄，避免重复打开
- 彩色日志（依赖 termcolor），便于在终端快速分辨不同级别
- 统一的 JSON 统计输出（训练/测试 epoch 级别指标）"""

import builtins
import decimal
import functools
import logging
import simplejson
import sys
import os
from termcolor import colored

from .distributed import is_master_process
from .file_io import PathManager

# Show filename and line number in logs 自定义日志格式：包含级别、文件名、行号、消息
_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"


def _suppress_print():
    """Suppresses printing from the current process.
    屏蔽当前进程的 print 输出。
    在多进程/分布式训练中，常只希望“主进程”打印，其他进程静默以防止日志刷屏。
    这里通过重写 builtins.print 为 no-op 实现。"""

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        # 什么都不做，直接吞掉输出
        pass

    builtins.print = print_pass


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
# 通过 lru_cache 缓存已打开的同一路径文件对象，避免重复打开导致的句柄泄露或竞争。
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers  # noqa
def setup_logging(
    num_gpu, num_shards, output="", name="visual_prompt", color=True):
    """Sets up the logging.
    初始化并返回一个命名 logger。
    参数：
        num_gpu: 当前节点的 GPU 数量，用于判断主进程（单机多卡时 rank=0 的进程）
        num_shards: 分布式场景下的“节点/分片”数，用于进一步判断全局主进程
        output: 日志文件输出目录或具体文件路径（.txt/.log），为空则不写文件
        name: logger 的名称（根名），用于区分不同子模块
        color: 终端是否启用彩色输出
    设计要点：
        - 仅“主进程”往 stdout 打印，非主进程屏蔽 print
        - 全局主进程（num_gpu * num_shards 的语义）再决定是否写入日志文件
        - 使用自定义 formatter，彩色/非彩色两套"""
    # Enable logging only for the master process
    if is_master_process(num_gpu):
        # Clear the root logger to prevent any existing logging config
        # (e.g. set by another module) from messing with our setup
        logging.root.handlers = []
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format=_FORMAT, stream=sys.stdout
        )
    else:
        _suppress_print()

    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    # remove any lingering handler
    logger.handlers.clear()

    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(name),
        )
    else:
        formatter = plain_formatter

    if is_master_process(num_gpu):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if is_master_process(num_gpu * num_shards):
        if len(output) > 0:
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, "logs.txt")

            PathManager.mkdirs(os.path.dirname(filename))

            fh = logging.StreamHandler(_cached_log_stream(filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)
    return logger


def setup_single_logging(name, output=""):
    """Sets up the logging.非分布式/单进程的简化版日志初始化（始终在 stdout 打印）。
    参数：
        name: logger 名称
        output: 可选的日志文件路径或目录
    与 setup_logging 的区别：
        - 不考虑主进程判断；总是配置 stdout 的输出
        - 不处理彩色开关参数（这里默认使用彩色）"""
    # Enable logging only for the master process
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )

    if len(name) == 0:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name=name,
        abbrev_name=str(name),
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if len(output) > 0:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs.txt")

        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


def log_json_stats(stats, sort_keys=True):
    """
    Logs json stats.将统计信息（字典）以 JSON 格式写日志。
    典型用法：训练/测试周期结束后，将指标（loss、acc 等）统一记录成一行 JSON，
    方便后处理脚本（grep/jq/pandas）解析与可视化。
    细节：
    - Python 3.6+ 中 `json.encoder.FLOAT_REPR` 已无效；为保证定长小数位，
      这里用 decimal + 字符串格式化，将浮点数转成固定 6 位小数的 Decimal。
    - simplejson.dumps(sort_keys=True, use_decimal=True) 以保证键排序和 Decimal 正确序列化。
    - 若 _type 是 "test_epoch"/"train_epoch"，额外加上 "json_stats:" 前缀，便于 grep。
    """
    # It seems that in Python >= 3.6 json.encoder.FLOAT_REPR has no effect
    # Use decimal+string as a workaround for having fixed length values in logs
    logger = get_logger(__name__)
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    if stats["_type"] == "test_epoch" or stats["_type"] == "train_epoch":
        logger.info("json_stats: {:s}".format(json_stats))
    else:
        logger.info("{:s}".format(json_stats))


class _ColorfulFormatter(logging.Formatter):
    # from detectron2
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log
