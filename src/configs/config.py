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
_C.MODEL.PROMPT.DEEP = False
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.DEBUG_SHAPES = False
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
_C.MODEL.PROMPT.DISTRIBUTOR.EVAL_SAMPLE_MODE = "fixed_eps"      # 评测采样：mean 使用 eps=0；fixed_eps 使用固定噪声 buffer
_C.MODEL.PROMPT.DISTRIBUTOR.FIXED_EPS_SEED = 0             # fixed_eps buffer 的独立随机种子；不影响全局 torch 随机状态
_C.MODEL.PROMPT.DISTRIBUTOR.USE_SLOT_EMBED = True         # 是否给每个 instance prompt 加可学习槽位编码

_C.MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_ENABLE = False      # 是否启用 split-latent prompt：把 posterior 切成 semantic/variation 两个因子
_C.MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_SEMANTIC_DIM = 384  # semantic factor 维度；第一版默认 384
_C.MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_VARIATION_DIM = 384 # variation factor 维度；第一版默认 384，二者之和必须等于 768
_C.MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_VARIATION_GATE_INIT = 0.0 # variation prompt 注入强度初值；0 表示初始先不让 variation 强扰动 prompt

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
_C.MODEL.SEMANTIC_TOKENS.ORTHO.CODEBOOK_TRAINABLE = False   # 让 semantic token 更像确定的属性编码，而不是又变成一组自由可学习 prompt
_C.MODEL.SEMANTIC_TOKENS.ORTHO.CODEBOOK_SEED = 0
_C.MODEL.SEMANTIC_TOKENS.ORTHO.DEBUG = False

_C.MODEL.CONSISTENCY = CfgNode()
_C.MODEL.CONSISTENCY.ENABLE = False
_C.MODEL.CONSISTENCY.PROJ = "linear"
_C.MODEL.CONSISTENCY.DIST = "cosine"

_C.MODEL.GRAPH_INPUT = CfgNode()
_C.MODEL.GRAPH_INPUT.ATTR_NAME_EMBED_PATH = "datasets/xlsa17/xlsa17/data/CUB/cub_attributes_sbert_all_mpnet_base_v2.pt" # 属性名文本 embedding 路径；默认与 ORTHO.TEXT_EMBED_PATH 指向同一缓存
_C.MODEL.GRAPH_INPUT.NUM_CLASSES = 200               # CUB 全局类别数；类别关系图 G 的尺寸为 [NUM_CLASSES, NUM_CLASSES]
_C.MODEL.GRAPH_INPUT.ATTR_DIM = 312                  # CUB 属性维度
_C.MODEL.GRAPH_INPUT.TEXT_DIM = 768                  # 属性名文本 embedding 维度，也对应 prompt latent 维度
_C.MODEL.GRAPH_INPUT.GRAPH_SOURCE = "fuse"           # 图来源：acc/acssc/fuse；或外部矩阵 key，如 method1_diff/method2_diff/method3_diff；external 兼容读取 key=graph
_C.MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_PATH = ""        # GRAPH_SOURCE 指向外部 key 时读取的 [C,C] 类别关系矩阵文件；支持 .npz/.npy/.pt/.pth
_C.MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_SYMMETRIZE = True # 是否强制 external graph 对称化
_C.MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_CLAMP = True     # 是否把 external graph 截断到 [0,1]
_C.MODEL.GRAPH_INPUT.EXTERNAL_GRAPH_DIAG_VALUE = 1.0 # external graph 对角线值；负数表示不改对角线
_C.MODEL.GRAPH_INPUT.RHO = 0.0                       # fuse 图中 Acc 的融合权重；G=rho*Acc+(1-rho)*Acssc
_C.MODEL.GRAPH_INPUT.TOPK = 16                       # 从 G[y] 中保留的语义相近类别数
_C.MODEL.GRAPH_INPUT.TAU_ACC = 0.07                  # 构造语义 target T_y 时的 softmax 温度
_C.MODEL.GRAPH_INPUT.TARGET_MIX_ALPHA = 0.1          # semantic target 与 one-hot 的混合比例；0=纯 one-hot，1=纯语义近邻分布
_C.MODEL.GRAPH_INPUT.EPS = 1e-8                      # GraphProbPrior 概率归一化与 log 的数值稳定下界

_C.MODEL.GRAPH_PROB_PRIOR = CfgNode()
_C.MODEL.GRAPH_PROB_PRIOR.ENABLE = True                # 是否启用 GraphProbPrior；可单独替代标准 Prompt KL
_C.MODEL.GRAPH_PROB_PRIOR.MODE = "graph_conditioned_semantic_prior" # 候选：graph_conditioned_semantic_prior / class_aggregate_mmd / factorized_latent
_C.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT = 0.001             # GraphProbPrior 辅助损失权重；用于替代标准 N(0,I) KL 时单独开启
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_MEAN_MODE = "residual_anchor" # prior mean 构造方式：learned=旧 prior_head；residual_anchor=312维属性残差锚点+小修正；graph_gp_conditioned=用 support-seen 视觉中心经 Graph-GP 条件推断全类 prototype
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_VAR_MODE = "unit"       # prior 方差策略：learned=MLP预测；unit=logvar=0；constant=固定 PRIOR_LOGVAR_CONST
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_LOGVAR_CONST = 0.0      # PRIOR_VAR_MODE=constant 时使用的固定 logvar
_C.MODEL.GRAPH_PROB_PRIOR.RESIDUAL_SIGMA_MIN = 0.05     # 312维属性残差标准化时的 std 下界，防止低方差属性被放大
_C.MODEL.GRAPH_PROB_PRIOR.RESIDUAL_CLIP = 3.0           # 标准化属性残差的截断范围 [-clip, clip]
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_DELTA_SCALE = 0.1       # small correction 强度：prior_mu 由 anchor + scale*tanh(delta) 得到
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_MU_SCALE = 2.0          # prior_mu 的全局基础半径，控制 Gaussian KL/几何距离的整体尺度
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_PRIOR_MU_SCALE = False
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_MU_SCALE_MIN = 0.5
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_MU_SCALE_MAX = 5.0
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_PRIOR_DELTA_SCALE = False
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_DELTA_SCALE_MIN = 0.0
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_DELTA_SCALE_MAX = 0.8
_C.MODEL.GRAPH_PROB_PRIOR.PRIOR_RADIUS_MODE = "residual_norm" # fixed=所有类别同半径；residual_norm=属性残差越大，类别 prior 半径越大
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SUPPORT_RATIO = 0.8  # graph_gp_conditioned 中 seen 类划为 support-seen 的比例；剩余 seen 类作为 pseudo-unseen 诊断外推能力
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SPLIT_EVERY_EPOCH = 1 # 每多少个 epoch 重新划分一次 support-seen / pseudo-unseen；1 表示每个 epoch 换一次
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_SPLIT_SEED = 2027    # Graph-GP 类别 split 的随机种子；保证 support/pseudo 划分可复现
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_CENTER_SOURCE = "posterior_mu" # Graph-GP 视觉中心来源；第一版只支持 posterior_mu，保证和 KL 对齐空间一致
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_DETACH_CENTERS = True # 统计 V_support 时截断 posterior_mu 梯度，避免 prior target 和 posterior 相互追逐
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MODE = "class_var_over_count" # R_s 观测噪声：constant / class_var_over_count
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_CONST = 0.05 # constant 模式下每个 support center 的观测噪声标量
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MIN = 1e-4 # R_s 下界；防止 K_ss + R_s 对角线过小导致 solve 不稳定
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBS_NOISE_MAX = 1.0  # R_s 上界；防止某些类被过大噪声完全忽略
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_RIDGE = 1e-4         # 加到 K_ss + R_s 对角线上的数值稳定项；只服务线性方程求解
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_SYMMETRIZE = True # 条件推断前是否对 graph kernel 做对称化
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_CLAMP = True  # 条件推断前是否把 graph kernel 裁到非负，避免负边直接进入协方差
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_KERNEL_NORMALIZE = "diag" # Graph-GP kernel 归一化：diag 让对角线尺度接近 1；none 保留原始尺度
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_SOURCE = "unit" # Graph-GP prior_logvar 来源：unit / constant / current_prior_var_mode / dynamic_uncertainty
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_FLOOR = 0.05    # dynamic_uncertainty 的最低 prior variance，避免 KL 因 prior 太窄而爆炸
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_PROTO_WEIGHT = 1.0  # dynamic_uncertainty 中 GP predictive uncertainty 的权重
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_VISUAL_WEIGHT = 1.0 # dynamic_uncertainty 中传播后的类内视觉方差权重
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_MIN = 1e-4      # dynamic_uncertainty 最终 prior variance 下界
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PRIOR_VAR_MAX = 10.0      # dynamic_uncertainty 最终 prior variance 上界
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_USE_PSEUDO_UNSEEN = True # 训练期是否在 seen 类内部划 pseudo-unseen，用来模拟 ZSL 外推
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_MATCH_DETACH_PRIOR = True # posterior 对齐 M_star 时是否 detach；第一版应保持 True
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_OBJECTIVE = "energy_classification" # Graph-GP 自身训练目标；当前只支持 energy_classification
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_ENERGY_TAU = 1.0          # Graph-GP Energy Classification 的 softmax 温度
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_ENERGY_CLASS_SPACE = "seen" # energy classification 的类别空间；当前只支持 seen
_C.MODEL.GRAPH_PROB_PRIOR.GRAPH_GP_PSEUDO_WEIGHT = 1.0       # pseudo-unseen 样本在 energy CE 中的样本权重；1 表示不额外加权
_C.MODEL.GRAPH_PROB_PRIOR.TAU_GRAPH = 0.07              # 用 G[c] 构造 graph top-k context / 旧 neighbor_bank 时的 softmax 温度
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_TAU_GRAPH = False
_C.MODEL.GRAPH_PROB_PRIOR.TAU_GRAPH_MIN = 0.02
_C.MODEL.GRAPH_PROB_PRIOR.TAU_GRAPH_MAX = 0.40
_C.MODEL.GRAPH_PROB_PRIOR.TAU_LATENT = 1.0              # softmax(-KL(q||p_c)/tau) 的温度，控制 latent matching 分布尖锐程度
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_TAU_LATENT = False
_C.MODEL.GRAPH_PROB_PRIOR.TAU_LATENT_MIN = 0.01
_C.MODEL.GRAPH_PROB_PRIOR.TAU_LATENT_MAX = 0.30
_C.MODEL.GRAPH_PROB_PRIOR.REL_WEIGHT = 0.0              # class_aggregate_* 专用全类 prior 关系正则权重；0 表示关闭
_C.MODEL.GRAPH_PROB_PRIOR.TAU_PRIOR = 0.07              # prior Gaussian symKL 关系分布 softmax 温度，仅 REL_WEIGHT>0 时生效
_C.MODEL.GRAPH_PROB_PRIOR.MMD_SAMPLES = 1               # class_aggregate_mmd 中每个 posterior/prior 高斯采样次数
_C.MODEL.GRAPH_PROB_PRIOR.MMD_SIGMA = 1.0               # class_aggregate_mmd 的 RBF kernel sigma
_C.MODEL.GRAPH_PROB_PRIOR.FACTORIZED_VARIATION_WEIGHT = 0.0 # factorized_latent 中 variation aggregate matching 权重；第一版默认关闭
_C.MODEL.GRAPH_PROB_PRIOR.FACTORIZED_DECOUPLE_WEIGHT = 0.0  # factorized_latent 中 semantic/variation 去相关权重；0 表示不启用
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_ENABLE = False      # 是否启用 prior_mu 几何校准正则；默认关闭
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_TYPE = "soft_distribution_matching" # soft_distribution_matching / graph_ordinal_ranking
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_LOSS_WEIGHT = 1e-4       # geometry loss 加到 GraphProbPrior 内部的权重
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_TOPK = 5                 # geometry loss 使用的 graph top-k 邻居数
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_MARGIN_MIN = 0.1         # geometry 边界项的最小非交叠安全间隔 m0
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_SIGMA_PRIOR = 0.2        # 固定方差模式下 geometry 距离使用的 prior 半径
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_DETACH_RADIUS = True     # geometry 距离中是否截断 radius 梯度，防止用方差逃避约束
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_BARRIER = 0.1        # geometry soft boundary 的 softplus 平滑温度
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_GRAPH_DIST = 0.1     # soft_distribution_matching 的 graph target softmax 温度
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_DIST = 0.1           # soft_distribution_matching 中 softmax(-D/tau) 的距离温度
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_GEOM_TAU_DIST = False
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_DIST_MIN = 0.01
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_TAU_DIST_MAX = 0.30
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_BOUND_WEIGHT = 0.1       # soft_distribution_matching 可选边界项权重
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_GEOM_BOUND_WEIGHT = False
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_BOUND_WEIGHT_MIN = 0.0
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_BOUND_WEIGHT_MAX = 0.50
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_DISTANCE_TYPE = "clearance" # graph_ordinal_ranking 使用的距离：clearance复用分布半径距离；cosine/euclidean为诊断备选
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_MARGIN_BASE = 0.0    # ordinal ranking 的基础排序间隔
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_MARGIN_SCALE = 0.1   # graph 相似度差距越大，额外排序间隔越大
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_GEOM_ORD_MARGIN_SCALE = False
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_MARGIN_SCALE_MIN = 0.0
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_MARGIN_SCALE_MAX = 0.50
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_GRAPH_GAP_EPS = 1e-6 # graph 相似度差距小于该值时不构造排序对
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_WEIGHT_BY_GAP = True # 是否按归一化 graph gap 加权排序对
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_EPS = 1e-8           # ordinal ranking 内部归一化数值稳定项
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_NON_OVERLAP_WEIGHT = 0.0 # 可选 non-overlap 弱边界权重；默认关闭
_C.MODEL.GRAPH_PROB_PRIOR.LEARN_GEOM_ORD_NON_OVERLAP_WEIGHT = False
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_NON_OVERLAP_WEIGHT_MIN = 0.0
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_NON_OVERLAP_WEIGHT_MAX = 0.08
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_NON_OVERLAP_MIN_DIST = 0.05 # d_norm 小于该值时认为分布间隔过小
_C.MODEL.GRAPH_PROB_PRIOR.GEOM_ORD_NON_OVERLAP_SCOPE = "topk" # non-overlap 作用范围：topk / all
_C.MODEL.GRAPH_PROB_PRIOR.MONITOR_ENABLE = False        # 是否记录 GraphProbPrior 温度/距离尺度监测量；默认关闭避免日常日志过长
_C.MODEL.GRAPH_PROB_PRIOR.MONITOR_INACTIVE = False      # 是否额外计算当前 MODE 未使用的温度位置；默认关闭以避免额外开销
_C.MODEL.GRAPH_PROB_PRIOR.MONITOR_TOPK = 5              # 监测 top-k mass 时使用的 k
_C.MODEL.GRAPH_PROB_PRIOR.MONITOR_EVERY_N = 37           # 每多少次 GraphProbPrior forward 记录一次监测量；1 表示每次都记录
_C.MODEL.GRAPH_PROB_PRIOR.MONITOR_EFFECTIVE_RANK = False # 是否计算 effective-rank 监测；关闭可避免 SVD/MAGMA 日志
_C.MODEL.GRAPH_PROB_PRIOR.DEBUG = False                 # 打印一次 GraphProbPrior 的关键 shape 和 loss 标量

_C.MODEL.AFFINITY = CfgNode()
_C.MODEL.AFFINITY.ENABLE = False
_C.MODEL.AFFINITY.DETACH = True
_C.MODEL.AFFINITY.VIS = True

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
_C.SOLVER.LOG_EVERY_N = 111
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
