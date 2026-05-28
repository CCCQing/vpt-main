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
_C.MODEL.PROMPT.DISTRIBUTOR.DISABLE_SAMPLING = False       # 旧键：已废弃；新逻辑用 EVAL_SAMPLE_MODE 控制评测采样
_C.MODEL.PROMPT.DISTRIBUTOR.LATENT_DIM = 256               # 旧键：旧 PromptGenerator latent 维度；ViaPT-style 新分支不再使用
_C.MODEL.PROMPT.DISTRIBUTOR.HIDDEN_DIM = 512               # 旧键：旧 PromptGenerator hidden 维度；ViaPT-style 新分支不再使用
_C.MODEL.PROMPT.DISTRIBUTOR.POOL = "gap"                   # 旧键：旧视觉池化方式；新分支按 SOURCE 各自处理
_C.MODEL.PROMPT.DISTRIBUTOR.SOURCE = "token_mlp"           # 视觉统计来源：token_mlp / vit_cls_prepass / cnn_torchvision / clip_frozen / dinov2_small
_C.MODEL.PROMPT.DISTRIBUTOR.STATS_HIDDEN_DIM = 64          # stats head 中间维度 H：视觉输入先降到 H，再输出 mu/logvar
_C.MODEL.PROMPT.DISTRIBUTOR.INSTANCE_TOKENS = 25           # 图像条件 instance prompt 数量；与 DOMAIN_TOKENS 之和必须等于 NUM_TOKENS
_C.MODEL.PROMPT.DISTRIBUTOR.DOMAIN_TOKENS = 25             # 任务/数据集级 learnable domain prompt 数量
_C.MODEL.PROMPT.DISTRIBUTOR.OUTPUT_PARAM = "logvar"        # 分布参数输出形式；当前只支持 logvar，即 stats_out=[mu, logvar]
_C.MODEL.PROMPT.DISTRIBUTOR.LOGVAR_MIN = -10.0             # logvar clamp 下界，防止 std 过小导致数值异常
_C.MODEL.PROMPT.DISTRIBUTOR.LOGVAR_MAX = 5.0               # logvar clamp 上界，防止 std 爆炸
_C.MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE = "mean"      # 评测采样：mean 使用 eps=0；fixed_eps 使用固定噪声 buffer
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
_C.MODEL.SEMANTIC_GRAPH.GRAPH_SOURCE = "fuse"           # 语义图来源：acc=属性置信图；acssc=属性文本语义图；fuse=rho 融合
_C.MODEL.SEMANTIC_GRAPH.TOPK = 20                       # 从 G[y] 中保留的语义相近类别数
_C.MODEL.SEMANTIC_GRAPH.TAU_ACC = 0.07                  # 构造语义 target T_y 时的 softmax 温度
_C.MODEL.SEMANTIC_GRAPH.LOSS_TYPE = "none"              # 候选：none / acc_hidden / rel_kl / rel_all / ot / gw / fgw
_C.MODEL.SEMANTIC_GRAPH.LOSS_WEIGHT = 0.0               # 语义图辅助损失总权重；0 表示不参与训练
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
_C.SOLVER.SPV.DETACH = "none"                     # via_prompt / direct / none
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
_C.SOLVER.LOSS_ANCHOR_CONS_WEIGHT = 0.0
_C.SOLVER.LOSS_FREE_KD_WEIGHT = 0.0

_C.SOLVER.DIAG = CfgNode()
_C.SOLVER.DIAG.SHUFFLE_RAW_TARGETS = False
_C.SOLVER.DIAG.SHUFFLE_PROTOTYPES = False
_C.SOLVER.DIAG.STRICT_CHECKS = False
_C.SOLVER.DIAG.PRINT_LOSS_WIRING = False
_C.SOLVER.OPTIMIZER = "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.ADAM_BETA1 = 0.9
_C.SOLVER.ADAM_BETA2 = 0.999
_C.SOLVER.ADAM_EPS = 1e-8
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
_C.DATA_ROOT = ""
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
_C.DIST_BACKEND = "gloo"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """Get a copy of the default config."""
    return _C.clone()
