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
_C.MODEL.MODEL_ROOT = "D:\\postgraduate1\\project\\vpt-main\\weights\\official"
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
_C.MODEL.PROMPT.DISTRIBUTOR.ENABLE = True
_C.MODEL.PROMPT.DISTRIBUTOR.DISABLE_SAMPLING = False
_C.MODEL.PROMPT.DISTRIBUTOR.LATENT_DIM = 256
_C.MODEL.PROMPT.DISTRIBUTOR.HIDDEN_DIM = 512
_C.MODEL.PROMPT.DISTRIBUTOR.POOL = "gap"

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

_C.MODEL.CONSISTENCY = CfgNode()
_C.MODEL.CONSISTENCY.ENABLE = False
_C.MODEL.CONSISTENCY.PROJ = "linear"
_C.MODEL.CONSISTENCY.DIST = "cosine"

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
_C.DATA.DATAPATH = "D:\\postgraduate1\\project\\datasets\\CUB\\CUB_200_2011"
_C.DATA.FEATURE = "sup_vitb16_224"
_C.DATA.NUMBER_CLASSES = 200
_C.DATA.CLASS_WEIGHTS_TYPE = "none"
_C.DATA.CROPSIZE = 224
_C.DATA.BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 4
_C.DATA.PIN_MEMORY = True

_C.DATA.XLSA = CfgNode()
_C.DATA.XLSA.ENABLED = True
_C.DATA.XLSA.RES101_PATH = "D:\\postgraduate1\\project\\datasets\\xlsa17\\xlsa17\\data\\CUB\\res101.mat"
_C.DATA.XLSA.SPLIT_PATH = "D:\\postgraduate1\\project\\datasets\\xlsa17\\xlsa17\\data\\CUB\\att_splits.mat"
_C.DATA.XLSA.PROTOCOL_MODE = "dev"     #dev / final_zsl / final_gzsl
_C.DIST_BACKEND = "gloo"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """Get a copy of the default config."""
    return _C.clone()
