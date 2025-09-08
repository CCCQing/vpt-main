#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object 全局配置对象
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False  # 禁用调试模式
_C.OUTPUT_DIR = "./output"  # 指定训练结果输出目录
_C.RUN_N_TIMES = 5  # 定义训练的重复次数
# Perform benchmarking to select the fastest CUDNN algorithms to use 进行基准测试以选择最快的 CUDNN 算法来使用
# Note that this may increase the memory usage and will likely not result 注意：当输入尺寸可变（例如 COCO 训练）时，这可能会增加内存使用，
# in overall speedups when variable size inputs are used (e.g. COCO training) 并且通常不会带来总体速度提升。
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing) 要使用的 GPU 数量（适用于训练和测试）
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries 注意：由于 GPU 操作库中某些算子的非确定性实现，仍可能存在非确定性
_C.SEED = None

# ----------------------------------------------------------------------
# Model options 模型选项
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
# 迁移学习类型
_C.MODEL.TRANSFER_TYPE = "prompt"  # one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias 可选
_C.MODEL.WEIGHT_PATH = ""  # if resume from some checkpoint file 如果从某个 checkpoint 文件恢复
_C.MODEL.SAVE_CKPT = False

_C.MODEL.MODEL_ROOT = "D:/postgraduate1/project/vpt-main/weights/official"  # root folder for pretrained model weights  预训练模型权重的根目录

_C.MODEL.TYPE = "vit"   # 视觉骨干网络
_C.MODEL.MLP_NUM = 0

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()
_C.MODEL.PROMPT.NUM_TOKENS = 5      # Prompt 长度
_C.MODEL.PROMPT.LOCATION = "prepend"
# prompt initalizatioin:
#    (1) default "random"
#    (2) "final-cls" use aggregated final [cls] embeddings from training dataset
#    (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
#    (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
# prompt 初始化:
#    (1) 默认 "random"
#    (2) "final-cls" 使用来自训练数据集的聚合最终 [cls] 嵌入
#    (3) "cls-nolastl": 对于 deep prompt，使用前 12 个 cls 嵌入（不包含最终输出）
#    (4) "cls-nofirstl": 使用最后 12 个 cls 嵌入（不包含输入到第一层的部分）
_C.MODEL.PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""
_C.MODEL.PROMPT.CLSEMB_PATH = ""
_C.MODEL.PROMPT.PROJECT = -1  # "projection mlp hidden dim""projection mlp 隐藏维度"
# Prompt 类型 （深度或浅层）指定
_C.MODEL.PROMPT.DEEP = True # True# False # "whether do deep prompt or not, only for prepend location"是否进行深度 prompt，仅在 prepend 位置支持

# 在前x层插提示，若为shallow则忽略
_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning 若设为 int，则进行部分深度 prompt 调优
_C.MODEL.PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer 是否只更新最后 n 层，而不是输入层
_C.MODEL.PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb 若为真，则所有深层将使用相同的 prompt 嵌入
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt 若为真，则不会为没有 prompt 的层扩展输入序列
# how to get the output emb for cls head: 如何获取用于 cls 头的输出嵌入:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
#    original: 遵循原始 backbone 的选择，
#    img_pool: 仅对图像 patch 做池化
#    prompt_pool: 仅对 prompt 嵌入做池化
#    imgprompt_pool: 对除了 cls token 外的所有内容做池化
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_C.MODEL.PROMPT.DROPOUT = 0.0       # 对 prompt 嵌入做随机置零，用于正则化，缓解提示过拟合
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False
# ----------------------------------------------------------------------
# adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"

# ----------------------------------------------------------------------
# Solver options（优化器/训练）选项
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001         # 权重衰减
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300


_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.01    # 学习率
_C.SOLVER.BIAS_MULTIPLIER = 1.              # for prompt + bias

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000


_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params 若为 True，将打印可训练参数的名称

# ----------------------------------------------------------------------
# Dataset options 数据集选项
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""
_C.DATA.DATAPATH = ""
_C.DATA.FEATURE = ""  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process 每个训练进程的数据加载器 worker 数量
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory 将数据加载到固定内存（pinned host memory）
_C.DATA.PIN_MEMORY = True

_C.DIST_BACKEND = "gloo"      # Linux默认是 nccl，win默认是gloo
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config. 获取默认配置的副本。
    """
    return _C.clone()
