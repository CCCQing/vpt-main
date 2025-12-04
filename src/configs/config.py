#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object 全局配置对象
_C = CfgNode()

_C.DBG = False            # 调试用开关
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
# 是否采用广义零样本模式（影响划分 / 评估）
_C.GZSL = False

_C.USE_TRAINVAL = False
# JSON / XLSA 数据调试相关的开关
# 这些只是为了让 cub.yaml 中的顶层字段能顺利 merge，不会报 key 不存在。
_C.MODE = "xlsa"            # "json" / "xlsa"
_C.SKIP_DUMMY = False
_C.SPLIT = "train"          # "train" / "val" / "test"

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
_C.MODEL.PROMPT.NUM_TOKENS = 50      # Prompt 长度
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
# 是否在前向时切断 prompt 参数的梯度（包括初始 prompt 与深层 prompt），
# 便于只训练提示生成/融合模块而冻结提示表本身
_C.MODEL.PROMPT.DETACH_PROMPT_GRAD = False
_C.MODEL.PROMPT.DISTRIBUTION_ONLY = True  # True: 完全依赖提示分布/运行时 provider，不创建可训练 prompt 参数
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False
# 是否在模型构建后打印可训练参数统计，便于确认冻结策略和提示/语义模块是否参与训练
_C.MODEL.LOG_TRAINABLE = True

_C.MODEL.PROMPT.FREEZE_EMBEDDINGS = True

# 共享概念基（语义-视觉对齐）
_C.MODEL.PROMPT.SEMANTIC_CONCEPT = CfgNode()
_C.MODEL.PROMPT.SEMANTIC_CONCEPT.ENABLE = False
_C.MODEL.PROMPT.SEMANTIC_CONCEPT.NUM_SLOTS = 4        # K，概念槽数量
_C.MODEL.PROMPT.SEMANTIC_CONCEPT.NUM_HEADS = 4        # 注意力头数（需能整除 hidden_size）
_C.MODEL.PROMPT.SEMANTIC_CONCEPT.DROPOUT = 0.0
_C.MODEL.PROMPT.SEMANTIC_CONCEPT.LAMBDA_INIT = 1.0    # 融合残差的初始缩放
_C.MODEL.PROMPT.SEMANTIC_CONCEPT.USE_LAYER_NORM = True
_C.MODEL.PROMPT.SEMANTIC_CONCEPT.PROJ_NORM = True

# 提示分布生成器（Pre-ViT prompt distribution）。在 DISTRIBUTION_ONLY=True 时，需要
# 提供 runtime provider 才能生成第 0 层 prompt；此配置用于构建该 provider。
_C.MODEL.PROMPT.DISTRIBUTOR = CfgNode()
_C.MODEL.PROMPT.DISTRIBUTOR.ENABLE = False     # 是否启用提示分布模块
_C.MODEL.PROMPT.DISTRIBUTOR.LATENT_DIM = 256   # 提示生成的潜变量维度
_C.MODEL.PROMPT.DISTRIBUTOR.HIDDEN_DIM = 512   # 后验头/生成器隐藏层维度
_C.MODEL.PROMPT.DISTRIBUTOR.POOL = "gap"        # 视觉统计池化方式
_C.MODEL.PROMPT.DISTRIBUTOR.SEMANTIC_DIM = 0   # 语义维度（0 表示不使用语义条件）
_C.MODEL.PROMPT.DISTRIBUTOR.SEMANTIC_PROJ_DIM = 0  # 语义投影维度（0 表示不投影）

# R-similarity classification head (CLS ↔ R-space prototypes)
_C.MODEL.R_SIMILARITY = CfgNode()
_C.MODEL.R_SIMILARITY.ENABLE = True               # 用r_similarity_head分类头
_C.MODEL.R_SIMILARITY.PROJ_DIM = -1
_C.MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE = False  # 是否对 CLS 额外做线性投影（默认直接用 CLS）
_C.MODEL.R_SIMILARITY.USE_COSINE = True
_C.MODEL.R_SIMILARITY.LOGIT_SCALE_INIT = 10.0
# ----------------------------------------------------------------------
# ===== 动态提示与多层融合设置 ===== 10.27 加入
# ----------------------------------------------------------------------
_C.MODEL.PROMPT_FUSION = CfgNode()
_C.MODEL.PROMPT_FUSION.ENABLED = False  # 是否启用动态提示/多层融合模块
_C.MODEL.PROMPT_FUSION.LATENT_DIM = 256  # 提示潜在空间维度
_C.MODEL.PROMPT_FUSION.LAYER_IDS = [0, 6, 11]  # 在哪些 Transformer 层执行提示更新
_C.MODEL.PROMPT_FUSION.KL_WEIGHT = 1e-4  # KL 正则权重（训练时可读）
_C.MODEL.PROMPT_FUSION.DROPOUT = 0.0  # 采样后的提示向量 dropout
_C.MODEL.PROMPT_FUSION.USE_SEMANTICS = False  # 是否拼接额外语义向量到提示生成器
_C.MODEL.PROMPT_FUSION.SEMANTIC_DIM = 0  # 语义向量维度（USE_SEMANTICS=True 时生效）
_C.MODEL.PROMPT_FUSION.AFFINITY_BIAS = True  # 在亲和矩阵上是否学习额外偏置
_C.MODEL.PROMPT_FUSION.RETURN_AUX = False  # 默认推理是否返回辅助 loss/亲和信息
# ----------------------------------------------------------------------
# Affinity options（是否走亲和分支及其计算配置）
# ----------------------------------------------------------------------
_C.MODEL.AFFINITY = CfgNode()
_C.MODEL.AFFINITY.ENABLE = True           # 是否启用 forward_with_affinity 分支
_C.MODEL.AFFINITY.PROMPT_LENGTH = 0        # prompt 长度（缺省时由 NUM_TOKENS 填充）
_C.MODEL.AFFINITY.RETURN_CROSS = False     # 是否返回跨模态亲和
_C.MODEL.AFFINITY.NORMALIZE = True         # 亲和矩阵是否归一化
_C.MODEL.AFFINITY.DETACH = True            # 计算亲和时是否分离梯度
_C.MODEL.AFFINITY.VIS = False              # 是否同时返回注意力权重（vis 模式）

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
_C.SOLVER.LOSS = "softmax_prompt_align"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001         # 权重衰减
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300        # 早停

# Zero-shot evaluation mode: "zsl" (default unseen-only) or "gzsl" (include seen + unseen)
_C.SOLVER.EVAL_MODE = "zsl"
# Backwards-compatible toggle for configs that directly set SOLVER.GZSL or top-level GZSL flags.
_C.SOLVER.GZSL = False

_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.25     # 学习率
_C.SOLVER.BIAS_MULTIPLIER = 1.              # for prompt + bias

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000


_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params 若为 True，将打印可训练参数的名称

# ----------------------------------------------------------------------
# Dataset options 数据集选项
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

# Optional dataset root for demos or external loaders that expect a top-level path.
_C.DATA_ROOT = ""


_C.DATA.NAME = "CUB_200_2011"
_C.DATA.DATAPATH = "D:/postgraduate1/project/datasets/CUB/CUB_200_2011"
_C.DATA.FEATURE = "sup_vitb16_224"  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = 200
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process 每个训练进程的数据加载器 worker 数量
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory 将数据加载到固定内存（pinned host memory）
_C.DATA.PIN_MEMORY = True

# -------------------------11.15新增：读入语义模态信息---------------------------------
_C.DATA.XLSA = CfgNode()
_C.DATA.XLSA.ENABLED = True


_C.DATA.XLSA.RES101_PATH = "D:/postgraduate1/project/datasets/xlsa17/xlsa17/data/CUB/res101.mat"
_C.DATA.XLSA.SPLIT_PATH = "D:/postgraduate1/project/datasets/xlsa17/xlsa17/data/CUB/att_splits.mat"
_C.DATA.XLSA.TRAIN_KEY = "train_loc"
_C.DATA.XLSA.VAL_KEY = "val_loc"
_C.DATA.XLSA.TRAINVAL_KEY = "trainval_loc"
_C.DATA.XLSA.TRAIN_USE_TRAINVAL = False     # 决定在 split="train" 时用 train_loc 还是 trainval_loc：
_C.DATA.XLSA.TEST_KEY = "test_unseen_loc"
_C.DATA.XLSA.TEST_SEEN_KEY = "test_seen_loc"
_C.DATA.XLSA.TEST_INCLUDE_SEEN = False      # 是否把 seen 测试样本也拼到 test split 里。
# -------------------------11.15新增结束---------------------------------

_C.DIST_BACKEND = "gloo"      # Linux默认是 nccl，win默认是gloo
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config. 获取默认配置的副本。
    """
    return _C.clone()
