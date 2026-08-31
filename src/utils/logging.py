#!/usr/bin/env python3

"""Logging.
本模块封装了工程统一的日志系统（基于 Python 标准库 logging），
支持：
- 仅主进程输出到控制台（多进程/分布式时屏蔽非主进程的 print/log）
- 同时输出到文件并缓存文件句柄，避免重复打开
- 彩色日志（依赖 termcolor），便于在终端快速分辨不同级别"""

import builtins
import functools
import logging
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


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


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
