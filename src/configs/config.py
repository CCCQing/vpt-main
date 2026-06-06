#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode

_C = CfgNode()
# ==========================================================
# A. 全局运行配置
# ==========================================================
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 5
_C.CUDNN_BENCHMARK = False
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1
_C.SEED = None
_C.DIST_RANK = 0

# ==========================================================
# B. MODEL：模型主配置
# ==========================================================
_C.MODEL = CfgNode()
_C.MODEL.WEIGHT_PATH = ""
_C.MODEL.MODEL_ROOT = "weights/official"
_C.MODEL.TYPE = "vit"
_C.MODEL.CLASSIFIER = "r_similarity"        # r_similarity r_similarity_v2 vspcn_baseline

_C.MODEL.PROMPT = CfgNode()
_C.MODEL.PROMPT.ENABLE = True
_C.MODEL.PROMPT.BACKEND = "dynamic"   # dynamic / vpt_deep
_C.MODEL.PROMPT.INIT_SOURCE = "learned"   # learned / distributor_mean
_C.MODEL.PROMPT.NUM_TOKENS = 50
_C.MODEL.PROMPT.DEEP = True
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.DEBUG_SHAPES = False
_C.MODEL.PROMPT.EVOLVE_INIT_MODE = "identity"
_C.MODEL.LOG_TRAINABLE = True

_C.MODEL.PROMPT.DISTRIBUTOR = CfgNode()
_C.MODEL.PROMPT.DISTRIBUTOR.ENABLE = True                  # 是否启用 prompt_init_provider；通常配合 INIT_SOURCE="distributor_mean" 使用
_C.MODEL.PROMPT.DISTRIBUTOR.SOURCE = "token_mlp"           # 视觉统计来源：token_mlp / vit_cls_prepass / cnn_torchvision / clip_frozen / dinov2_small
_C.MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM = 64          # stats head 中间维度 H：视觉输入先降到 H，再输出 mu/logvar
_C.MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS = 25           # 图像条件 instance prompt 数量；与 DOMAIN_TOKENS 之和必须等于 NUM_TOKENS
_C.MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS = 25             # 任务/数据集级 learnable domain prompt 数量
_C.MODEL.PROMPT.DISTRIBUTOR.OUTPUT_PARAM = "logvar"        # 分布参数输出形式；当前只支持 logvar，即 stats_out=[mu, logvar]
_C.MODEL.PROMPT.DISTRIBUTOR.LOGVAR_MIN = -10.0             # logvar clamp 下界，防止 std 过小导致数值异常
_C.MODEL.PROMPT.DISTRIBUTOR.LOGVAR_MAX = 5.0               # logvar clamp 上界，防止 std 爆炸
_C.MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE = "mean"      # 评测采样：mean 使用 eps=0；fixed_eps 使用固定噪声 buffer
_C.MODEL.PROMPT.DISTRIBUTOR.FIXED_EPS_SEED = 0             # fixed_eps buffer 的独立随机种子；不影响全局 torch 随机状态
_C.MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED = False         # 是否给每个 instance prompt 加可学习槽位编码
_C.MODEL.PROMPT.DISTRIBUTOR.CNN_NAME = "efficientnet_b0"   # cnn_torchvision 候选：efficientnet_b0 / mobilenet_v3_small
_C.MODEL.PROMPT.DISTRIBUTOR.CLIP_NAME = "mobileclip_s0"    # clip_frozen 候选：mobileclip_s0 / tinyclip_vit8m；需本地权重
_C.MODEL.PROMPT.DISTRIBUTOR.CLIP_LOCAL_DIR = ""            # clip_frozen 本地权重目录，期望存在 {CLIP_NAME}.pt
_C.MODEL.PROMPT.DISTRIBUTOR.DINO_LOCAL_DIR = ""            # dinov2_small 本地权重目录，期望存在 dinov2_small.pt
_C.MODEL.PROMPT.DISTRIBUTOR.EXTERNAL_ALLOW_DOWNLOAD = False # 是否允许 torchvision 自动下载 CNN 权重；默认禁止
_C.MODEL.PROMPT.DISTRIBUTOR.DEBUG_DISTRIBUTOR_SHAPES = False # 打印一次 distributor 输入、mu/logvar、prompt 形状

_C.MODEL.R_SIMILARITY = CfgNode()
_C.MODEL.R_SIMILARITY.ENABLE = True
_C.MODEL.R_SIMILARITY.PROJ_DIM = -1             # v/s投影到新维度
_C.MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE = False
_C.MODEL.R_SIMILARITY.USE_COSINE = True
_C.MODEL.R_SIMILARITY.LOGIT_SCALE_INIT = 10.0
_C.MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE = 0.0

_C.MODEL.R_SIMILARITY_V2 = CfgNode()
_C.MODEL.R_SIMILARITY_V2.SCORE_MODE = "dot"          # dot / cosine
_C.MODEL.R_SIMILARITY_V2.LEARNABLE_SCALE = False
_C.MODEL.R_SIMILARITY_V2.LOGIT_SCALE_INIT = 10.0
_C.MODEL.R_SIMILARITY_V2.FIXED_LOGIT_SCALE = 0.0

_C.MODEL.SEMANTIC_TOKENS = CfgNode()
_C.MODEL.SEMANTIC_TOKENS.ENABLE = True
_C.MODEL.SEMANTIC_TOKENS.TOKENIZER = "linear"      # linear / orthogonal
_C.MODEL.SEMANTIC_TOKENS.NUM_TOKENS = 1
_C.MODEL.SEMANTIC_TOKENS.INPUT_DIM = 312
_C.MODEL.SEMANTIC_TOKENS.TRAIN_SOURCE = "label"       # label / class_mean / none / random_fixed / label_shuffle / learned_token
_C.MODEL.SEMANTIC_TOKENS.EVAL_SOURCE = "none"         # label / class_mean / none / random_fixed / label_shuffle / learned_token
# BEGIN SEMANTIC_ABLATION_EXPERIMENT                    # 语义替换实验+att mass统计  ΔAsv
_C.MODEL.SEMANTIC_TOKENS.RANDOM_SEED = 0
_C.MODEL.SEMANTIC_TOKENS.RANDOM_STD = 1.0
_C.MODEL.SEMANTIC_TOKENS.LEARNED_INIT_STD = 0.02
# END SEMANTIC_ABLATION_EXPERIMENT
_C.MODEL.SEMANTIC_TOKENS.BLOCK_S_TO_CLS = False
_C.MODEL.SEMANTIC_TOKENS.ORTHO = CfgNode()
_C.MODEL.SEMANTIC_TOKENS.ORTHO.GROUP_MODE = "manual_cub8"  # manual_cub8 / equal / prefix
_C.MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_MODE = "none"          # none / null_residual / text_init_codebook

_C.MODEL.SEMANTIC_TOKENS.ORTHO.ATTRIBUTES_PATH = "datasets/CUB/attributes.txt"
_C.MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_EMBED_PATH = "datasets/xlsa17/xlsa17/data/CUB/cub_attributes_sbert_all_mpnet_base_v2.pt"
_C.MODEL.SEMANTIC_TOKENS.ORTHO.TEXT_GATE_INIT = 0.0
_C.MODEL.SEMANTIC_TOKENS.ORTHO.CODEBOOK_TRAINABLE = False
_C.MODEL.SEMANTIC_TOKENS.ORTHO.CODEBOOK_SEED = 0
_C.MODEL.SEMANTIC_TOKENS.ORTHO.DEBUG = False

_C.MODEL.CONSISTENCY = CfgNode()
_C.MODEL.CONSISTENCY.ENABLE = False
_C.MODEL.CONSISTENCY.PROJ = "linear"
_C.MODEL.CONSISTENCY.DIST = "cosine"

_C.MODEL.SEMANTIC_GRAPH = CfgNode()
_C.MODEL.SEMANTIC_GRAPH.ENABLE = False                  # 是否启用 prompt distribution 语义图辅助约束
_C.MODEL.SEMANTIC_GRAPH.ATTR_NAME_EMBED_PATH = "datasets/xlsa17/xlsa17/data/CUB/cub_attributes_sbert_all_mpnet_base_v2.pt" # 属性名文本 embedding 路径；默认与 ORTHO.TEXT_EMBED_PATH 指向同一缓存
_C.MODEL.SEMANTIC_GRAPH.NUM_CLASSES = 200               # CUB 全局类别数；语义图 G 的尺寸为 [NUM_CLASSES, NUM_CLASSES]
_C.MODEL.SEMANTIC_GRAPH.ATTR_DIM = 312                  # CUB 属性维度
_C.MODEL.SEMANTIC_GRAPH.TEXT_DIM = 768                  # 属性名文本 embedding 维度，也对应 prompt mu 维度
_C.MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE = "fuse"           # 语义图来源：acc=属性置信图；acssc=属性文本语义图；fuse=rho 融合 acc / acssc / fuse
_C.MODEL.SEMANTIC_GRAPH.RHO = 0.5                       # fuse 图中 Acc 的融合权重；G=rho*Acc+(1-rho)*Acssc
_C.MODEL.SEMANTIC_GRAPH.TOPK = 20                       # 从 G[y] 中保留的语义相近类别数
_C.MODEL.SEMANTIC_GRAPH.TAU_ACC = 0.07                  # 构造语义 target T_y 时的 softmax 温度
_C.MODEL.SEMANTIC_GRAPH.TAU_SEM = 0.07                  # rel_kl 中 batch 语义关系 R_s 的 softmax 温度
_C.MODEL.SEMANTIC_GRAPH.TAU_PROMPT = 0.07               # prompt 关系/原型 logits 的初始温度；若 PROMPT_SCALE_LEARNABLE=True，仅作为初始化
_C.MODEL.SEMANTIC_GRAPH.PROMPT_SCALE_LEARNABLE = True   # 是否学习 prompt_logit_scale；True 时 scale 初始为 1/TAU_PROMPT
_C.MODEL.SEMANTIC_GRAPH.PROMPT_STAT_SOURCE = "mu"        # semantic graph 约束使用的 prompt 统计量：mu / instance_mean
_C.MODEL.SEMANTIC_GRAPH.TARGET_MIX_ALPHA = 0.1          # semantic target 与 one-hot 的混合比例；0=纯 one-hot，1=纯语义近邻分布
_C.MODEL.SEMANTIC_GRAPH.LOSS_TYPE = "none"              # 候选：none / acc_hidden / rel_kl / rel_all / ot / gw / fgw
_C.MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT = 0.0               # 语义图辅助损失总权重；0 表示不参与训练
_C.MODEL.SEMANTIC_GRAPH.OT_EPS = 0.05                   # Sinkhorn 熵正则系数；越大运输计划越平滑
_C.MODEL.SEMANTIC_GRAPH.OT_ITERS = 20                   # Sinkhorn 迭代次数
_C.MODEL.SEMANTIC_GRAPH.OT_PRIOR_ETA = 1.0              # FGW 中语义先验强度；作用于 prior_cost=M-eta*log(T+delta)
_C.MODEL.SEMANTIC_GRAPH.OT_ALPHA = 0.5                  # FGW 中结构项权重；loss=(1-alpha)*node+alpha*gw
_C.MODEL.SEMANTIC_GRAPH.OT_DELTA = 1e-8                 # OT/KL 概率归一化与 log 的数值稳定下界
_C.MODEL.SEMANTIC_GRAPH.OT_BALANCED_MODE = "batch_semantic_mean" # OT 目标边界 b 的构造方式；当前支持 batch_semantic_mean
_C.MODEL.SEMANTIC_GRAPH.OT_DETACH_PLAN = True           # 是否停止 Sinkhorn plan 的梯度；True 时只让 cost/GW 项回传到 mu
_C.MODEL.SEMANTIC_GRAPH.DEBUG = False                   # 打印一次 A_conf/E_attr/G/T/OT plan 等调试形状

_C.MODEL.AFFINITY = CfgNode()
_C.MODEL.AFFINITY.ENABLE = False
_C.MODEL.AFFINITY.DETACH = True
_C.MODEL.AFFINITY.VIS = True

_C.MODEL.AFFINITY_EVOLUTION = CfgNode()
_C.MODEL.AFFINITY_EVOLUTION.ENABLE = False
_C.MODEL.AFFINITY_EVOLUTION.PROMPT_ENABLE = True
_C.MODEL.AFFINITY_EVOLUTION.SEMANTIC_ENABLE = True
_C.MODEL.AFFINITY_EVOLUTION.PROMPT_TARGET = "QpQv"      # QpKv / QpQv / KpKv
_C.MODEL.AFFINITY_EVOLUTION.SEMANTIC_TARGET = "QpKv"    # QpKv / QpQv / KpKv
_C.MODEL.AFFINITY_EVOLUTION.PROMPT_LAMBDA = 0.0     # teacher correction 强度，范围 [0, 1]
_C.MODEL.AFFINITY_EVOLUTION.SEMANTIC_LAMBDA = 0.0   # teacher correction 强度，范围 [0, 1]
_C.MODEL.AFFINITY_EVOLUTION.PROMPT_GAMMA_INIT = 0.0
_C.MODEL.AFFINITY_EVOLUTION.SEMANTIC_GAMMA_INIT = 0.0
_C.MODEL.AFFINITY_EVOLUTION.PROMPT_DETACH = "none"       # mediated / direct / none；非 none 时表示 teacher 选择
_C.MODEL.AFFINITY_EVOLUTION.SEMANTIC_DETACH = "none"     # via_prompt / direct / none；非 none 时表示 teacher 选择
_C.MODEL.AFFINITY_EVOLUTION.SEMANTIC_COMPOSE = "prob"    # prob / raw_then_norm

_C.MODEL.ATTENTION_MEDIATION = CfgNode()
# ATTENTION_MEDIATION 是新增的 block 内 mediated attention correction 分支。
# 默认关闭；开启时会在每层 ViT self-attention 内部额外构造 mediated route，
# 只对 prompt/semantic token 产生 correction，不替换原始 ViT attention 主路径。
# SOURCE:
#   probs  - 从 full softmax 后的真实 attention 概率子块构造 route。
#   scores - 从 softmax 前 QK logits 子块构造局部条件 route，再用 log-ratio bias 回到 full softmax。
# EXECUTION_MODE:
#   attention_parallel 表示只复制 attention 级别计算，不复制完整 Transformer block。
# MLP_POLICY:
#   enter_mlp 表示 correction 在当前层 MLP 前合并；skip_mlp 表示 block 后只写回 P/S residual。
# ROUTE_SCOPE/MASS_MODE:
#   visual_block 只改 P->V/S->V 的 visual 子块；full_row 直接构造完整 attention row。
#   row_preserve 保持每个 P/S token 的 visual mass；block_redistribute 允许组内重分配 visual mass。
# PROMPT_ROUTE/SEMANTIC_ROUTE:
#   分别控制 prompt-mediated 和 semantic-mediated 路径的方向。
# *_GAMMA_INIT:
#   每层可学习 gate 的初始值；0 表示初始等价于不开启 correction。
_C.MODEL.ATTENTION_MEDIATION.ENABLE = False
_C.MODEL.ATTENTION_MEDIATION.SOURCE = "probs"             # probs / scores；从 full attention 概率或 logits 构造 mediated route
_C.MODEL.ATTENTION_MEDIATION.EXECUTION_MODE = "attention_parallel"  # attention_parallel / block_parallel
_C.MODEL.ATTENTION_MEDIATION.MLP_POLICY = "enter_mlp"     # enter_mlp / skip_mlp
_C.MODEL.ATTENTION_MEDIATION.ROUTE_SCOPE = "visual_block" # visual_block / full_row
_C.MODEL.ATTENTION_MEDIATION.PROMPT_ROUTE = "S_to_P_and_V"    # S_to_P_and_V / P_to_S_to_V
_C.MODEL.ATTENTION_MEDIATION.SEMANTIC_ROUTE = "S_to_P_to_V"   # S_to_P_to_V / P_to_S_and_V
_C.MODEL.ATTENTION_MEDIATION.MASS_MODE = "row_preserve"   # row_preserve / block_redistribute
_C.MODEL.ATTENTION_MEDIATION.BETA_PROMPT_MASS = 0.0
_C.MODEL.ATTENTION_MEDIATION.BETA_SEMANTIC_MASS = 0.0
_C.MODEL.ATTENTION_MEDIATION.PROMPT_GAMMA_INIT = 0.0
_C.MODEL.ATTENTION_MEDIATION.SEMANTIC_GAMMA_INIT = 0.0
# 约束：
#   block_parallel 必须配合 MLP_POLICY="enter_mlp"。
#   full_row 已经构造完整 attention row，因此不能配合 block_redistribute。

_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"

_C.SOLVER = CfgNode()
_C.SOLVER.MAIN_LOSS = "vspcn"                          # vspcn / rsim / rsim_v2
_C.SOLVER.LOSS_CM_WEIGHT = 0.05
_C.SOLVER.LOSS_VSPCN_AR_WEIGHT = 0.0005
_C.SOLVER.LOSS_SEM_MED_WEIGHT = 0.0
_C.SOLVER.LOSS_SPV_WEIGHT = 0.0
_C.SOLVER.LOSS_ROUTE_TS_PROMPT_WEIGHT = 0.0
_C.SOLVER.LOSS_ROUTE_TS_SEMANTIC_WEIGHT = 0.0
_C.SOLVER.LOSS_ATTR_WEIGHT = 0.0
_C.SOLVER.LOSS_PROMPT_KL_WEIGHT = 0.0              # prompt distribution KL 权重；只约束 instance prompt 的 mu/logvar

_C.SOLVER.SEM_MED = CfgNode()
_C.SOLVER.SEM_MED.TARGET = "KpKv"                 # QpKv / QpQv / KpKv
_C.SOLVER.SEM_MED.METRIC = "cosine"                  # mse / kl / cosine
_C.SOLVER.SEM_MED.NORM = "softmax"                # first version only supports softmax
_C.SOLVER.SEM_MED.DETACH = "mediated"             # mediated / direct / none
_C.SOLVER.SEM_MED.LAYERS = []                     # empty means all shared layers

_C.SOLVER.SPV = CfgNode()
_C.SOLVER.SPV.COMPOSE = "prob"                    # prob (softmax(QsKp) @ softmax(Apv) 对齐 softmax(QsKv))/ raw_then_norm(softmax(QsKp @ Apv) 对齐 softmax(QsKv))
_C.SOLVER.SPV.TARGET = "QpKv"                     # QpKv / QpQv / KpKv
_C.SOLVER.SPV.METRIC = "kl"                       # mse / kl / cosine
_C.SOLVER.SPV.NORM = "softmax"                    # "none"raw affinity 直接相乘
_C.SOLVER.SPV.DETACH = "none"                     # mediated / direct / none
_C.SOLVER.SPV.LAYERS = []                         # empty means all shared layers

_C.SOLVER.ROUTE_TS = CfgNode()
_C.SOLVER.ROUTE_TS.PROMPT_ENABLE = True           # 对 prompt evolution 的 student/teacher route 做显式对齐
_C.SOLVER.ROUTE_TS.SEMANTIC_ENABLE = True         # 对 semantic evolution 的 student/teacher route 做显式对齐
_C.SOLVER.ROUTE_TS.METRIC = "kl"                  # mse / kl / cosine
_C.SOLVER.ROUTE_TS.LAYERS = []                    # empty means all shared layers

_C.SOLVER.ATTR = CfgNode()
_C.SOLVER.ATTR.METRIC = "mse"                     # 第一版只支持 mse：约束 ViT 交互后的 decoded attributes 接近真实类别属性

_C.SOLVER.RSIM_V2 = CfgNode()
_C.SOLVER.RSIM_V2.ALIGN_MODE = "ar"                  # ar / cm
_C.SOLVER.RSIM_V2.ALIGN_WEIGHT = 0.02
_C.SOLVER.LOSS_AGR_RES_WEIGHT = 0.0
_C.SOLVER.LOSS_CONS_WEIGHT = 0.0

_C.SOLVER.DIAG = CfgNode()
_C.SOLVER.DIAG.SHUFFLE_RAW_TARGETS = False
_C.SOLVER.DIAG.SHUFFLE_PROTOTYPES = False
_C.SOLVER.DIAG.STRICT_CHECKS = False
_C.SOLVER.DIAG.PRINT_LOSS_WIRING = False
_C.SOLVER.OPTIMIZER = "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.PATIENCE = 300
_C.SOLVER.SCHEDULER = "cosine"          # 学习率随 epoch 怎么变化
_C.SOLVER.BASE_LR = 0.0005
_C.SOLVER.BIAS_MULTIPLIER = 1.
_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000
_C.SOLVER.DEBUG_GRAD_NORM = False
_C.SOLVER.DEBUG_TRACE_ONCE = False
_C.SOLVER.DEBUG_SHAPES = False
_C.SOLVER.EVAL_MODE = "zsl"


_C.SOLVER.VIS = CfgNode()
_C.SOLVER.VIS.ENABLE = False
_C.SOLVER.VIS.FINAL_ONLY = True
_C.SOLVER.VIS.EVERY_EPOCH = 1
_C.SOLVER.VIS.EPOCH_LIST = []
_C.SOLVER.VIS.SPLITS = ["test_unseen"]
_C.SOLVER.VIS.MAX_SAMPLES = 5
_C.SOLVER.VIS.CORRECT_SAMPLES = 3
_C.SOLVER.VIS.WRONG_SAMPLES = 2
_C.SOLVER.VIS.SAVE_RAW = True
_C.SOLVER.VIS.SAVE_IMAGES = True
_C.SOLVER.VIS.ROLLOUT = True
# BEGIN SEMANTIC_ABLATION_EXPERIMENT
_C.SOLVER.VIS.SEMANTIC_ABLATION = CfgNode()
_C.SOLVER.VIS.SEMANTIC_ABLATION.ENABLE = False
_C.SOLVER.VIS.SEMANTIC_ABLATION.DELTA_ASV = False
_C.SOLVER.VIS.SEMANTIC_ABLATION.ATTENTION_MASS = False
_C.SOLVER.VIS.SEMANTIC_ABLATION.DELTA_REFERENCE_SOURCE = "class_mean"
# END SEMANTIC_ABLATION_EXPERIMENT
_C.SOLVER.DBG_TRAINABLE = False

_C.DATA = CfgNode()
_C.DATA.NAME = "CUB"
_C.DATA.DATAPATH = "datasets/CUB/CUB_200_2011"
_C.DATA.FEATURE = "sup_vitb16_224"
_C.DATA.NUMBER_CLASSES = 200
_C.DATA.CLASS_WEIGHTS_TYPE = "none"
_C.DATA.CROPSIZE = 224
_C.DATA.BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 4
_C.DATA.PIN_MEMORY = True

_C.DATA.XLSA = CfgNode()
_C.DATA.XLSA.ENABLED = True
_C.DATA.XLSA.RES101_PATH = "datasets/xlsa17/xlsa17/data/CUB/res101.mat"
_C.DATA.XLSA.SPLIT_PATH = "datasets/xlsa17/xlsa17/data/CUB/att_splits.mat"
_C.DATA.XLSA.PROTOCOL_MODE = "dev"     #dev / final_zsl / final_gzsl
_C.DIST_INIT_PATH = "env://"


def get_cfg():
    """Get a copy of the default config."""
    return _C.clone()
