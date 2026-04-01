#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object 鍏ㄥ眬閰嶇疆瀵硅薄
_C = CfgNode()

# -------------------------------
# 鍩虹杩愯閰嶇疆
# -------------------------------
_C.DBG = False                         # 鍏ㄥ眬璋冭瘯寮€鍏?
_C.OUTPUT_DIR = "./output"             # 璁粌杈撳嚭鐩綍
_C.RUN_N_TIMES = 5                     # 閲嶅杩愯娆℃暟锛堢敤浜庡娆″疄楠屽彇骞冲潎锛?
# CUDNN benchmark锛?# 褰撹緭鍏ュ昂瀵稿浐瀹氭椂鍙兘鍔犻€燂紱褰撹緭鍏ュ昂瀵稿彲鍙樻椂锛屽彲鑳藉鍔犳樉瀛樺崰鐢ㄤ笖鏀剁泭鏈夐檺
_C.CUDNN_BENCHMARK = False

# 鍒嗗竷寮?/ 闅忔満鎬?
_C.NUM_GPUS = 1                        # 浣跨敤 GPU 鏁伴噺锛堣缁?娴嬭瘯閫氱敤锛?
_C.NUM_SHARDS = 1                      # 鍒嗙墖鏁帮紙澶氭満璁粌鏃朵娇鐢級
_C.SEED = None                         # 闅忔満绉嶅瓙锛涙敞鎰?GPU 绠楀瓙浠嶅彲鑳藉瓨鍦ㄩ潪纭畾鎬?
# 鍏煎鏃ч厤缃殑椤跺眰寮€鍏?
_C.GZSL = False                        # 鏃х増鍏煎锛氭槸鍚﹂噰鐢?GZSL 妯″紡锛堝疄闄呬互 SOLVER.EVAL_MODE 涓哄噯锛?
_C.USE_TRAINVAL = False                # 鏃х増鍏煎锛氭槸鍚﹀湪璁粌涓娇鐢?trainval锛堝疄闄呭缓璁湅 DATA.XLSA.TRAIN_USE_TRAINVAL锛?
# 这些只是为了让 cub.yaml 中的顶层字段能顺利 merge，不会报 key 不存在。
_C.MODE = "xlsa"            # "json" / "xlsa"
_C.SKIP_DUMMY = False
_C.SPLIT = "train"          # "train" / "val" / "test"
# ----------------------------------------------------------------------
# Model options 妯″瀷閫夐」
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
# 杩佺Щ瀛︿範绫诲瀷
_C.MODEL.TRANSFER_TYPE = "prompt"  # one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias 鍙€?
_C.MODEL.WEIGHT_PATH = ""              # 鑻ヤ粠 checkpoint 鎭㈠锛屽垯鎸囧畾璺緞
_C.MODEL.SAVE_CKPT = False             # 鏄惁淇濆瓨 checkpoint
_C.MODEL.MODEL_ROOT = "weights/official"  # 棰勮缁冩潈閲嶆牴鐩綍

_C.MODEL.TYPE = "vit"                  # 涓诲共绫诲瀷
_C.MODEL.MLP_NUM = 0                   # 澶囩敤瀛楁

# -----------------------------------------------------------------------------
# Linear head options
# -----------------------------------------------------------------------------
_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []         # 绾挎€?MLP 澶寸殑闅愯棌灞傞厤缃?_C.MODEL.LINEAR.DROPOUT = 0.1          # 绾挎€?MLP 澶?dropout

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()

# Prompt 鍩烘湰璁剧疆
_C.MODEL.PROMPT.NUM_TOKENS = 50        # Prompt token 鏁伴噺
_C.MODEL.PROMPT.LOCATION = "prepend"   # Prompt 鎻掑叆浣嶇疆锛岀洰鍓嶄富瑕佷娇鐢?prepend

# prompt 鍒濆鍖?
#    (1) 榛樿 "random"
#    (2) "final-cls" 浣跨敤鏉ヨ嚜璁粌鏁版嵁闆嗙殑鑱氬悎鏈€缁?[cls] 宓屽叆
#    (3) "cls-nolastl": 瀵逛簬 deep prompt锛屼娇鐢ㄥ墠 12 涓?cls 宓屽叆锛堜笉鍖呭惈鏈€缁堣緭鍑猴級
#    (4) "cls-nofirstl": 浣跨敤鏈€鍚?12 涓?cls 宓屽叆锛堜笉鍖呭惈杈撳叆鍒扮涓€灞傜殑閮ㄥ垎锛?
_C.MODEL.PROMPT.INITIATION = "random"
# _C.MODEL.PROMPT.CLSEMB_FOLDER = ""     # cls embedding 鎵€鍦ㄧ洰褰曪紙鑻ヤ娇鐢ㄧ壒寰佸垵濮嬪寲锛?
_C.MODEL.PROMPT.CLSEMB_PATH = ""       # cls embedding 鏂囦欢璺緞锛堣嫢鐩存帴鎸囧畾鏂囦欢锛?
# Deep prompt 璁剧疆
_C.MODEL.PROMPT.DEEP = True            # 鏄惁鍚敤 deep prompt锛堜粎 prepend 妯″紡鏈夋晥锛?
_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None # 鑻ヨ涓?int锛屽垯鍙湪閮ㄥ垎灞備娇鐢?deep prompt
_C.MODEL.PROMPT.REVERSE_DEEP = False   # True 琛ㄧず鍙湪鏈€鍚?n 灞備娇鐢?deep prompt
_C.MODEL.PROMPT.DEEP_SHARED = False    # True 琛ㄧず鎵€鏈?deep layers 鍏变韩鍚屼竴缁?prompt 鍙傛暟
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False
# 鑻ヤ负 True锛屽垯娌℃湁 prompt 鐨勫眰涓嶆墿灞曡緭鍏ュ簭鍒?
# ViT 鏈€缁堣緭鍑哄浣曟睜鍖栫敤浜庝笅娓稿垎绫?#   original       : 淇濇寔 backbone 鍘熷鍋氭硶
#   img_pool       : 浠呮睜鍖栧浘鍍?patch
#   prompt_pool    : 浠呮睜鍖?prompt token
#   imgprompt_pool : 姹犲寲闄?CLS 澶栫殑鍏ㄩ儴 token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"

# Prompt 璁粌鐩稿叧
_C.MODEL.PROMPT.DROPOUT = 0.0       # 瀵?prompt 宓屽叆鍋氶殢鏈虹疆闆讹紝鐢ㄤ簬姝ｅ垯鍖栵紝缂撹В鎻愮ず杩囨嫙鍚?_
_C.MODEL.PROMPT.DETACH_PROMPT_GRAD = True     # 鏄惁鍒囨柇 prompt 鍙傛暟姊害
_C.MODEL.PROMPT.DISTRIBUTION_ONLY = True


_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False    # 鏄惁姣忎釜 epoch 淇濆瓨 prompt
_C.MODEL.PROMPT.DEBUG_FLOW = False             # 鏄惁鎵撳嵃 prompt 鍓嶅悜璋冭瘯淇℃伅
_C.MODEL.PROMPT.DEBUG_SHAPES = False
# Layer-wise prompt evolution init mode:
# - identity: near no-op residual at start (legacy behavior)
# - zero: minimize initial perturbation to protect frozen backbone
_C.MODEL.PROMPT.EVOLVE_ZERO_INIT = True
_C.MODEL.PROMPT.EVOLVE_INIT_MODE = "zero"   # identity | zero
_C.MODEL.PROMPT.NOOP_KEEP_PARAMS = False
# True: 淇濈暀 prompt 妯″潡鍜屽弬鏁帮紝浣嗕笉灏?prompt token 鐪熸娉ㄥ叆 backbone 搴忓垪
_C.MODEL.LOG_TRAINABLE = True                  # 鏋勫缓妯″瀷鍚庢墦鍗板彲璁粌鍙傛暟缁熻
_C.MODEL.PROMPT.FREEZE_EMBEDDINGS = True       # 鏄惁鍐荤粨鍘熷 embedding

# -----------------------------------------------------------------------------
# Semantic concept module锛堟棫鐗堝叡浜蹇?璇箟-瑙嗚瀵归綈妯″潡锛?# 璇存槑锛?
#   杩欐槸浣犳棫妗嗘灦涓殑鈥淪haredConceptAligner / 鍏变韩姒傚康妲解€濈浉鍏宠缃€?
#   濡傛灉鍚庣画瀹屽叏鍒囧埌 AGR / affinity role migration锛岃繖鍧楀彲浠ラ€愭寮卞寲銆?
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Prompt distributor锛圥re-ViT prompt distribution generator锛?
# 璇存槑锛?#   褰?PROMPT.DISTRIBUTION_ONLY=True 鏃讹紝闇€瑕?runtime provider 鐢熸垚绗?0 灞?prompt銆?
#   鏈ā鍧楀嵆鐢ㄤ簬鏋勫缓璇?provider銆?# -----------------------------------------------------------------------------
_C.MODEL.PROMPT.DISTRIBUTOR = CfgNode()
_C.MODEL.PROMPT.DISTRIBUTOR.ENABLE = False
_C.MODEL.PROMPT.DISTRIBUTOR.LATENT_DIM = 256        # Prompt latent 缁村害
_C.MODEL.PROMPT.DISTRIBUTOR.HIDDEN_DIM = 512        # 鐢熸垚鍣?/ 鍚庨獙澶撮殣钘忓眰缁村害
_C.MODEL.PROMPT.DISTRIBUTOR.POOL = "gap"            # 鍥惧儚鍏ㄥ眬缁熻姹犲寲鏂瑰紡
_C.MODEL.PROMPT.DISTRIBUTOR.SEMANTIC_DIM = 0        # 璇箟鏉′欢缁村害锛? 琛ㄧず涓嶄娇鐢ㄨ涔夋潯浠讹級
_C.MODEL.PROMPT.DISTRIBUTOR.SEMANTIC_PROJ_DIM = 0   # 璇箟鎶曞奖缁村害锛? 琛ㄧず涓嶆姇褰憋級

# -----------------------------------------------------------------------------
# R-similarity classification head
# 璇存槑锛?#   浣跨敤瑙嗚琛ㄧず涓庤涔夊師鍨嬶紙raw / refined / top-k refined锛夊仛鐩镐技搴﹀垎绫汇€?
# -----------------------------------------------------------------------------
_C.MODEL.R_SIMILARITY = CfgNode()
_C.MODEL.R_SIMILARITY.ENABLE = True
_C.MODEL.R_SIMILARITY.PROJ_DIM = -1
_C.MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE = True
_C.MODEL.R_SIMILARITY.USE_COSINE = True
_C.MODEL.R_SIMILARITY.LOGIT_SCALE_INIT = 10.0
_C.MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE = 0.0
_C.MODEL.R_SIMILARITY.USE_PROTO_PER_SAMPLE = False
# -----------------------------------------------------------------------------
# Semantic scoring / refinement mode
# 璇存槑锛?#   鎺у埗鏈€缁堜娇鐢ㄥ摢绉嶈涔夊師鍨嬪弬涓庢墦鍒嗭細
#   - global_raw                : 浠呭師濮嬭涔夐敋鐐?#   - global_refined            : 鍏ㄧ被 refined semantics
#   - coarse_to_fine            : 鍏?raw 绮楁帓锛屽啀鍊欓€夊唴 refined
#   - affinity_role_migration   : 浜插拰椹卞姩鑱岃矗杩佺Щ鐗堟湰锛堝綋鍓嶄富绾匡級
# -----------------------------------------------------------------------------
_C.MODEL.SEMANTIC_SCORE_SOURCE = "auto"  # auto | raw | refined | fused
_C.MODEL.SEMANTIC_SCORE_MODE = "affinity_role_migration"  # global_raw | global_refined | coarse_to_fine | affinity_role_migration
_C.MODEL.SEMANTIC_SCORE_TOPK = 5
_C.MODEL.SEMANTIC_SCORE_ALPHA = 1.0
_C.MODEL.SEMANTIC_SCORE_TRAIN_INCLUDE_GT = True
_C.MODEL.SEMANTIC_SCORE_EVAL_OVERRIDE = ""  # "" | global_raw | global_refined | coarse_to_fine | affinity_role_migration

# -----------------------------------------------------------------------------
# Role migration锛堜翰鍜岄┍鍔ㄨ亴璐ｈ縼绉伙級
# 璇存槑锛?#   鐢ㄤ簬瀹氫箟鈥滄祬灞傚亸瑙嗚 / 娣卞眰鍋忚涔夆€濈殑灞傝寖鍥淬€?
# -----------------------------------------------------------------------------
_C.MODEL.ROLE_MIGRATION = CfgNode()
_C.MODEL.ROLE_MIGRATION.ENABLE = True
_C.MODEL.ROLE_MIGRATION.EARLY_END = 3     # 娴呭眰闃舵缁撴潫灞傦紙鍚級
_C.MODEL.ROLE_MIGRATION.LATE_START = 9    # 娣卞眰闃舵璧峰灞傦紙鍚級

# -----------------------------------------------------------------------------
# AGR: Affinity-Gated Residual
# 璇存槑锛?#   杞婚噺鐧界洅璇箟琛ュ厖璺緞锛?#   raw semantic anchor + affinity-weighted visual residual -> refined semantic
# -----------------------------------------------------------------------------
_C.MODEL.AGR = CfgNode()
_C.MODEL.AGR.ENABLE = True
_C.MODEL.AGR.TOPK = 5                      # 鍦?top-k 鍊欓€夌被鍐呭仛 AGR
_C.MODEL.AGR.ALPHA = 0.2                   # 娈嬪樊娉ㄥ叆寮哄害
_C.MODEL.AGR.FUSE_ALPHA = 0.5              # raw/refined 铻嶅悎绯绘暟
_C.MODEL.AGR.TRAIN_INCLUDE_GT = True       # 璁粌鏃舵槸鍚﹀己鍒舵妸 GT 骞跺叆鍊欓€?
# -----------------------------------------------------------------------------
# Lightweight semantic side branch (new mainline semantic evolution path)
# -----------------------------------------------------------------------------
_C.MODEL.SEMANTIC_BRANCH = CfgNode()
_C.MODEL.SEMANTIC_BRANCH.ENABLE = True
_C.MODEL.SEMANTIC_BRANCH.NUM_TOKENS = 4
_C.MODEL.SEMANTIC_BRANCH.USE_ANCHOR_FREE = False
_C.MODEL.SEMANTIC_BRANCH.ANCHOR_TOKENS = 8
_C.MODEL.SEMANTIC_BRANCH.FREE_TOKENS = 2
_C.MODEL.SEMANTIC_BRANCH.FREE_COMPETE_LAMBDA = 0.5
_C.MODEL.SEMANTIC_BRANCH.GAMMA_ANCHOR_SCALE = 1.0
_C.MODEL.SEMANTIC_BRANCH.GAMMA_FREE_SCALE = 1.0
_C.MODEL.SEMANTIC_BRANCH.START_LAYER = 0
_C.MODEL.SEMANTIC_BRANCH.END_LAYER = -1
_C.MODEL.SEMANTIC_BRANCH.GAMMA_MIN = 0.05
_C.MODEL.SEMANTIC_BRANCH.GAMMA_MAX = 1.0
# -----------------------------------------------------------------------------
# Consistency loss锛圓ENet 椋庢牸鐨勪竴鑷存€ф€濊矾锛屼絾涓嶇瓑浠蜂簬 AENet 缁撴瀯锛?# 璇存槑锛?
#   寤鸿鐢ㄤ簬绾︽潫鈥滄渶缁堣涔夊閲?/ 娣卞眰璇箟鍧囧€煎亸绉烩€濈殑鏂瑰悜涓€鑷存€с€?
# -----------------------------------------------------------------------------
_C.MODEL.CONSISTENCY = CfgNode()
_C.MODEL.CONSISTENCY.ENABLE = False
_C.MODEL.CONSISTENCY.PROJ = "linear"  # linear | mlp
_C.MODEL.CONSISTENCY.DIST = "cosine"  # cosine | l2

# -----------------------------------------------------------------------------
# Prompt fusion锛堟棫鐗堝灞傚姩鎬佹彁绀鸿瀺鍚堬級
# 璇存槑锛?#   杩欐槸 10.27 鍔犲叆鐨勫姩鎬佹彁绀?澶氬眰铻嶅悎妯″潡銆?#   濡傛灉鍚庣画浠?DISTRIBUTOR + ROLE_MIGRATION 涓轰富绾匡紝杩欏潡鍙鎯呭喌寮卞寲銆?
# -----------------------------------------------------------------------------
_C.MODEL.PROMPT_FUSION = CfgNode()
_C.MODEL.PROMPT_FUSION.ENABLED = False  # 鏄惁鍚敤鍔ㄦ€佹彁绀?澶氬眰铻嶅悎妯″潡
_C.MODEL.PROMPT_FUSION.LATENT_DIM = 256  # 鎻愮ず娼滃湪绌洪棿缁村害
_C.MODEL.PROMPT_FUSION.LAYER_IDS = [0, 6, 11]  # 鍦ㄥ摢浜?Transformer 灞傛墽琛屾彁绀烘洿鏂?
_C.MODEL.PROMPT_FUSION.KL_WEIGHT = 1e-4  # KL 姝ｅ垯鏉冮噸锛堣缁冩椂鍙锛?
_C.MODEL.PROMPT_FUSION.DROPOUT = 0.0  # 閲囨牱鍚庣殑鎻愮ず鍚戦噺 dropout
_C.MODEL.PROMPT_FUSION.USE_SEMANTICS = False  # 鏄惁鎷兼帴棰濆璇箟鍚戦噺鍒版彁绀虹敓鎴愬櫒
_C.MODEL.PROMPT_FUSION.SEMANTIC_DIM = 0  # 璇箟鍚戦噺缁村害锛圲SE_SEMANTICS=True 鏃剁敓鏁堬級
_C.MODEL.PROMPT_FUSION.AFFINITY_BIAS = True  # 鍦ㄤ翰鍜岀煩闃典笂鏄惁瀛︿範棰濆鍋忕疆
_C.MODEL.PROMPT_FUSION.RETURN_AUX = False  # 榛樿鎺ㄧ悊鏄惁杩斿洖杈呭姪 loss/浜插拰淇℃伅

# -----------------------------------------------------------------------------
# Affinity options
# 璇存槑锛?#   鎺у埗鏄惁璧?affinity 鍒嗘敮锛屼互鍙婁翰鍜岀煩闃垫槸鍚﹀綊涓€鍖栥€佹槸鍚?detach 绛夈€?
# -----------------------------------------------------------------------------
_C.MODEL.AFFINITY = CfgNode()
_C.MODEL.AFFINITY.ENABLE = True            # 鏄惁鍚敤 forward_with_affinity 鍒嗘敮
_C.MODEL.AFFINITY.PROMPT_LENGTH = 0        # prompt 闀垮害锛堢己鐪佹椂鐢?NUM_TOKENS 濉厖锛?
_C.MODEL.AFFINITY.RETURN_CROSS = False     # 鏄惁杩斿洖璺ㄦā鎬佷翰鍜?
_C.MODEL.AFFINITY.NORMALIZE = True         # 浜插拰鐭╅樀鏄惁褰掍竴鍖?
_C.MODEL.AFFINITY.DETACH = True            # 璁＄畻浜插拰鏃舵槸鍚﹀垎绂绘搴?
_C.MODEL.AFFINITY.VIS = False              # 鏄惁鍚屾椂杩斿洖娉ㄦ剰鍔涙潈閲嶏紙vis 妯″紡锛?
_C.MODEL.AFFINITY.PATCH_COMPETE_ENABLE = True
_C.MODEL.AFFINITY.PATCH_COMPETE_LAYERS = [-1]  # default: last layer only
_C.MODEL.AFFINITY.PATCH_COMPETE_TEMPERATURE = 1.0
_C.MODEL.AFFINITY.PATCH_COMPETE_MODE = "token_softmax"  # token_softmax | anchor_then_free
_C.MODEL.AFFINITY.PATCH_COMPETE_BALANCE_WEIGHT = 0.0
_C.MODEL.AFFINITY.PATCH_COMPETE_USE_NULL_TOKEN = False
# ----------------------------------------------------------------------
# adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"

# ----------------------------------------------------------------------
# Solver options锛堜紭鍖栧櫒/璁粌锛夐€夐」
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
# 涓绘崯澶辩被鍨?
_C.SOLVER.LOSS = "softmax_prompt_align"

# 閫氱敤鎹熷け瓒呭弬
_C.SOLVER.LOSS_ALPHA = 0.01
_C.SOLVER.LOSS_MARGIN = 0.05
_C.SOLVER.LOSS_CM_WEIGHT = 0.05

# 璇箟鍒嗗竷璺濈鎹熷け
_C.SOLVER.LOSS_SEM_DIST_WEIGHT = 0.01
_C.SOLVER.LOSS_SEM_DIST_TYPE = "cosine"  # cosine | kl | jsd
_C.SOLVER.LOSS_SEM_DIST_TEMP = 1.0
_C.SOLVER.LOSS_SEM_DIST_START_EPOCH = 10

# 璇箟璺敱鎹熷け
_C.SOLVER.LOSS_SEM_ROUTE_WEIGHT = 0.01
_C.SOLVER.LOSS_SEM_ROUTE_MASK_TYPE = "hard_topk"  # all_ones | hard_topk | soft_topk | confidence_weighted
_C.SOLVER.LOSS_SEM_ROUTE_TOPK = 8
_C.SOLVER.LOSS_SEM_ROUTE_GAMMA_IND = 0.0
_C.SOLVER.LOSS_SEM_ROUTE_GAMMA_DIR = 1.0
_C.SOLVER.LOSS_SEM_ROUTE_START_EPOCH = 10

# Hard negative margin
_C.SOLVER.LOSS_HN_MARGIN_ENABLE = False
_C.SOLVER.LOSS_HN_MARGIN_WEIGHT = 0.05
_C.SOLVER.LOSS_HN_MARGIN_VALUE = 0.1
_C.SOLVER.LOSS_HN_MARGIN_START_EPOCH = 0
_C.SOLVER.LOSS_HN_DETACH_NEG = True

# 鏂颁富绾跨浉鍏虫崯澶憋紙褰撳墠榛樿鍧囧叧闂紝鐢卞疄楠岄€愭鎵撳紑锛?
_C.SOLVER.LOSS_ROLE_EARLY_WEIGHT = 0.0     # 娴呭眰瑙掕壊杩佺Щ鎹熷け鏉冮噸
_C.SOLVER.LOSS_ROLE_LATE_WEIGHT = 0.0      # 娣卞眰瑙掕壊杩佺Щ鎹熷け鏉冮噸
_C.SOLVER.LOSS_AGR_RES_WEIGHT = 0.0        # AGR 娈嬪樊鑼冩暟绾︽潫鏉冮噸
_C.SOLVER.LOSS_AVS_ENT_WEIGHT = 0.0        # A_vs 鐔?灏栭攼搴︾害鏉熸潈閲?
_C.SOLVER.LOSS_CONS_WEIGHT = 0.0           # 涓€鑷存€ф崯澶辨潈閲?
_C.SOLVER.LOSS_ANCHOR_CONS_WEIGHT = 0.0
_C.SOLVER.LOSS_FREE_KD_WEIGHT = 0.0
_C.SOLVER.DIAG = CfgNode()
_C.SOLVER.DIAG.SHUFFLE_RAW_TARGETS = False
_C.SOLVER.DIAG.SHUFFLE_PROTOTYPES = False
_C.SOLVER.DIAG.STRICT_CHECKS = False
_C.SOLVER.DIAG.PRINT_LOSS_WIRING = False

# 浼樺寲鍣?
_C.SOLVER.OPTIMIZER = "adamw"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001         # 鏉冮噸琛板噺
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.ADAM_BETA1 = 0.9
_C.SOLVER.ADAM_BETA2 = 0.999
_C.SOLVER.ADAM_EPS = 1e-8

# 璁粌鎺у埗
_C.SOLVER.PATIENCE = 300        # 鏃╁仠
_C.SOLVER.SCHEDULER = "cosine"
_C.SOLVER.BASE_LR = 0.25     # 瀛︿範鐜?
_C.SOLVER.BIAS_MULTIPLIER = 1.               # prompt / bias 鐨?lr 鍊嶇巼
_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000

# 璋冭瘯璁粌琛屼负
_C.SOLVER.DEBUG_GRAD_NORM = False
_C.SOLVER.DEBUG_TRACE_ONCE = True
_C.SOLVER.DEBUG_SHAPES = False
_C.SOLVER.OVERFIT_ONE_BATCH_STEPS = 0
_C.SOLVER.OVERFIT_DISABLE_PROMPT_SAMPLING = False

# 闆舵牱鏈瘎浼版ā寮?
_C.SOLVER.EVAL_MODE = "zsl"
_C.SOLVER.GZSL = True  # 鏃х増鍏煎寮€鍏筹紝寤鸿瀹為檯浠?EVAL_MODE 涓哄噯

_C.SOLVER.MONITOR = CfgNode()
_C.SOLVER.c.ENABLE = False
_C.SOLVER.MONITOR.EVERY_EPOCH = 1
_C.SOLVER.MONITOR.MAX_SAMPLES = 512
_C.SOLVER.MONITOR.SAVE_JSON = True
_C.SOLVER.MONITOR.SAVE_CSV = True
_C.SOLVER.MONITOR.SAVE_HEATMAP = False
_C.SOLVER.MONITOR.HEATMAP_TOPK = 50
_C.SOLVER.MONITOR.TOKEN_PATCH_STATS_ENABLE = True
_C.SOLVER.MONITOR.TOKEN_PATCH_SOURCE = "avs"        # avs
_C.SOLVER.MONITOR.TOKEN_PATCH_HEAD_MODE = "head_avg"  # head_avg | head0
_C.SOLVER.MONITOR.TOKEN_PATCH_TOPRHO = 0.2
_C.SOLVER.MONITOR.TOKEN_PATCH_SAVE_MAPS = False
_C.SOLVER.MONITOR.TOKEN_PATCH_MAX_SAMPLES = 8

# -----------------------------------------------------------------------------
# Visualization pipeline (off by default)
# -----------------------------------------------------------------------------
_C.SOLVER.VIS = CfgNode()
_C.SOLVER.VIS.ENABLE = False
_C.SOLVER.VIS.EVERY_EPOCH = 1
# If non-empty, use explicit 1-based epoch list and ignore EVERY_EPOCH.
_C.SOLVER.VIS.EPOCH_LIST = []
_C.SOLVER.VIS.SPLITS = ["val", "test"]
_C.SOLVER.VIS.MAX_SAMPLES = 8
_C.SOLVER.VIS.SAVE_RAW = True
_C.SOLVER.VIS.SAVE_IMAGES = True
_C.SOLVER.VIS.LOCAL_CONTROL = True
_C.SOLVER.VIS.ROLLOUT = True
_C.SOLVER.VIS.GT_HN_COMPARE = True
_C.SOLVER.VIS.TRENDS = True

_C.SOLVER.DBG_TRAINABLE = False # 鑻ヤ负 True锛屽皢鎵撳嵃鍙缁冨弬鏁扮殑鍚嶇О

# ----------------------------------------------------------------------
# Dataset options 鏁版嵁闆嗛€夐」
# ----------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA_ROOT = ""                          # 椤跺眰鏁版嵁鏍圭洰褰曪紙渚?demo / 澶栭儴 loader 浣跨敤锛?
_C.DATA.NAME = "CUB_200_2011"
_C.DATA.DATAPATH = "datasets/CUB/CUB_200_2011"
_C.DATA.FEATURE = "sup_vitb16_224"         # 棰勬彁鍙栫壒寰佸悕绉版垨 backbone 鏍囪瘑

_C.DATA.PERCENTAGE = 1.0                   # 浣跨敤鏁版嵁姣斾緥
_C.DATA.NUMBER_CLASSES = 200
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"        # none / inv / inv_sqrt 绛?
_C.DATA.CROPSIZE = 224  # or 384
_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 4                     # 姣忎釜璁粌杩涚▼鐨勬暟鎹姞杞藉櫒 worker 鏁伴噺
_C.DATA.PIN_MEMORY = True                   # 灏嗘暟鎹姞杞藉埌鍥哄畾鍐呭瓨锛坧inned host memory锛?
# -----------------------------------------------------------------------------
# XLSA / semantic split options
# 璇存槑锛?#   杩欐槸褰撳墠闆舵牱鏈?骞夸箟闆舵牱鏈疄楠屾渶鍏抽敭鐨勬暟鎹垏鍒嗛厤缃€?
# -----------------------------------------------------------------------------
_C.DATA.XLSA = CfgNode()
_C.DATA.XLSA.ENABLED = True
_C.DATA.XLSA.RES101_PATH = "datasets/xlsa17/xlsa17/data/CUB/res101.mat"
_C.DATA.XLSA.SPLIT_PATH = "datasets/xlsa17/xlsa17/data/CUB/att_splits.mat"

# 鍚?split 瀵瑰簲鐨?key
_C.DATA.XLSA.TRAIN_KEY = "train_loc"
_C.DATA.XLSA.VAL_KEY = "val_loc"
_C.DATA.XLSA.TRAINVAL_KEY = "trainval_loc"
_C.DATA.XLSA.TEST_KEY = "test_unseen_loc"
_C.DATA.XLSA.TEST_SEEN_KEY = "test_seen_loc"

# 璁粌闃舵浣跨敤 train_loc 杩樻槸 trainval_loc
_C.DATA.XLSA.TRAIN_USE_TRAINVAL = False     # 鍐冲畾鍦?split="train" 鏃剁敤 train_loc 杩樻槸 trainval_loc锛?
# 娴嬭瘯 split 涓槸鍚︽妸 seen test 鏍锋湰鎷艰繘鍘?# True 鏃舵洿鎺ヨ繎 GZSL 娴嬭瘯锛汧alse 鏃舵洿鎺ヨ繎绾?unseen 娴嬭瘯
_C.DATA.XLSA.TEST_INCLUDE_SEEN = True

# -----------------------------------------------------------------------------
# Distributed backend
# -----------------------------------------------------------------------------
_C.DIST_BACKEND = "gloo"      # Linux榛樿鏄?nccl锛寃in榛樿鏄痝loo
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""

# ----------------------------------------------------------------------
# Defensive repair for accidentally merged comment+assignment lines.
# Keeps backward compatibility when some keys were not actually created.
# ----------------------------------------------------------------------
# Default mainline cleanup: old semantic branches disabled by default.
_C.SOLVER.LOSS = "softmax_margin_cm"
_C.SOLVER.LOSS_ALPHA = 0.0
_C.SOLVER.LOSS_SEM_DIST_WEIGHT = 0.0
_C.SOLVER.LOSS_SEM_ROUTE_WEIGHT = 0.0
_C.MODEL.SEMANTIC_SCORE_SOURCE = "raw"
_C.MODEL.R_SIMILARITY.USE_PROTO_PER_SAMPLE = False

def get_cfg():
    """
    Get a copy of the default config. 鑾峰彇榛樿閰嶇疆鐨勫壇鏈€?    """
    return _C.clone()
