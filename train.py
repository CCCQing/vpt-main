#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager
from src.utils.param_logging import log_trainable_parameters

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")   # 灞忚斀绗笁鏂瑰簱鐨勪竴浜涢潪鍏抽敭璀﹀憡锛岄伩鍏嶆棩蹇楀櫔澹?


def _sync_zero_shot_mode(cfg):
    """Ensure all ZSL/GZSL toggles are aligned before freezing the config."""

    # Read desired evaluation mode from solver / legacy top-level flags.
    eval_mode = str(getattr(cfg.SOLVER, "EVAL_MODE", "zsl") or "zsl").lower()
    gzsl_requested = any(
        [
            getattr(cfg, "GZSL", False),
            getattr(cfg.SOLVER, "GZSL", False),
            eval_mode == "gzsl",
        ]
    )

    if gzsl_requested:
        eval_mode = "gzsl"

    cfg.GZSL = gzsl_requested
    cfg.SOLVER.GZSL = gzsl_requested
    cfg.SOLVER.EVAL_MODE = eval_mode

    # Keep DATA.XLSA aligned with the evaluation choice so Dataset picks correct splits.
    xlsa_cfg = getattr(cfg.DATA, "XLSA", None)
    if xlsa_cfg is not None:
        xlsa_cfg.TEST_INCLUDE_SEEN = bool(gzsl_requested)


def _merge_local_path_cfg_if_exists(cfg):
    """Merge src/configs/local_path.yaml if present (machine-local path overrides)."""
    local_cfg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src",
        "configs",
        "local_path.yaml",
    )
    if os.path.isfile(local_cfg):
        cfg.merge_from_file(local_cfg)
        print(f"[config] merged local path overrides: {local_cfg}")

def setup(args):
    """
    Create configs and perform basic setups.鍒涘缓閰嶇疆骞跺仛鍩虹鐜璁剧疆锛堝寘鍚細鍚堝苟閰嶇疆銆佸垎甯冨紡鍦板潃銆佽緭鍑虹洰褰曠瓑锛?
    """
    cfg = get_cfg()                             # 鎷垮埌榛樿閰嶇疆锛圕onfigNode 鐨勫厠闅嗭級
    cfg.merge_from_file(args.config_file)       # 浠庡懡浠よ鎸囧畾鐨?yaml 鏂囦欢鍚堝苟閰嶇疆
    _merge_local_path_cfg_if_exists(cfg)        # optional: machine-local path override
    cfg.merge_from_list(args.opts)              # 浠庡懡浠よ鐨?KEY VALUE 褰㈠紡琛ュ厖/瑕嗙洊閰嶇疆
    _sync_zero_shot_mode(cfg)  # 缁熶竴 ZSL / GZSL 閰嶇疆锛岄伩鍏?KeyError

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])
    # 8.21鏃ヤ慨鏀筼s.environ璁块棶鐜鍙橀噺鍦ㄦ湰鍦?闈?SLURM 鐜涓嬭繖涓彉閲忎笉瀛樺湪

    # -------------------- 鍒嗗竷寮忓垵濮嬪寲鍦板潃璁剧疆 --------------------
    # 鍘熷瀹炵幇锛氬湪 SLURM 鐜涓嬩粠鐜鍙橀噺鍙栬妭鐐瑰悕骞剁粍瑁?tcp://<node>:12399
    # 浣嗗湪鏈湴/闈?SLURM 鐜涓嬩笉瀛樺湪璇ュ彉閲忥紝浼氳Е鍙?KeyError銆?
    # 鍥犳杩欓噷鍋氫簡鍋ュ．鍖栧鐞嗭細鑻ョ幆澧冨彉閲忓瓨鍦ㄥ垯瑕嗙洊 DIST_INIT_PATH锛屽惁鍒欎繚鎸侀厤缃粯璁ゅ€笺€?
    node = os.environ.get("SLURMD_NODENAME")
    if node:
        # 鍦?SLURM 闆嗙兢鑺傜偣涓婏紝鏄惧紡鐢?tcp://<鑺傜偣鍚?:12399锛屼繚璇佸鏈鸿兘鑱旈€?
        cfg.DIST_INIT_PATH = f"tcp://{node}:12399"
    # 鍚﹀垯浠€涔堜篃涓嶅仛锛屼繚鐣欓厤缃噷鐨勯粯璁ゅ€煎湪鏈湴 8.21淇敼缁撴潫

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1

    # -------------------- 杈撳嚭鐩綍缁勭粐瑙勫垯 --------------------
    # 鏈熸湜鐩綍缁撴瀯锛?
    # <OUTPUT_DIR>/<DATA.NAME>/<DATA.FEATURE>/lr<lr>_wd<wd>/run<count>
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    # 涓轰簡閬垮厤澶氳繘绋?骞跺彂鎻愪氦鏃朵簰鐩歌鐩栵紝鍚屼竴缁勮秴鍙傛渶澶氳繍琛?RUN_N_TIMES 娆?
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        # 闅忔満鐫＄湢 3~30 绉掞紝闄嶄綆骞跺彂鍐茬獊姒傜巼锛堜緥濡傚涓綔涓氬悓鏃跺垱寤哄悓鍚嶇洰褰曪級
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)     # 浠呭綋璺緞涓嶅瓨鍦ㄦ椂鍒涘缓
            cfg.OUTPUT_DIR = output_path        # 灏嗘渶缁堣緭鍑虹洰褰曞啓鍥?cfg锛屼緵鍚庣画鏃ュ織/ckpt 浣跨敤
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        # 淇濇姢锛氳嫢鍚岄厤缃凡缁忚窇婊?RUN_N_TIMES 娆★紝鍒欑洿鎺ユ姤閿欓€€鍑?
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()                               # 鍐荤粨閰嶇疆锛岄槻姝㈠悗缁唬鐮佹剰澶栦慨鏀?
    return cfg


def get_loaders(cfg, logger):
    """
        鏍规嵁閰嶇疆鏋勫缓 DataLoader锛坱rain / val / test锛?
        - VTAB 浠诲姟锛氭寜鐓у畼鏂圭害瀹氫娇鐢?train+val锛坱rainval锛変綔涓烘渶缁堣缁冮泦
        - 鍏朵粬浠诲姟锛氭瀯寤烘爣鍑嗙殑 train / val / test
    """
    logger.info("Loading training data (final training data for vtab)...")

    if cfg.DATA.NAME.startswith("vtab-"):
        # VTAB 鏈€缁堣缁冿細浣跨敤 800/200 璋冨弬鍚庯紝鍚堝苟 train+val 浣滀负鏈€缁堣缁冮泦
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    # 娉細VTAB 鏈€缁堣繍琛岄樁娈甸€氬父涓嶉渶瑕?val锛屼絾杩欓噷淇濇寔缁熶竴鎺ュ彛
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        # 鏌愪簺鍦烘櫙涓嶆彁渚涙祴璇曢泦锛堟垨鍙仛璁粌/楠岃瘉锛夛紝姝ゆ椂杩斿洖 None
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def _extract_class_ids_from_dataset(dataset) -> set:
    """Best-effort extraction of class ids from a dataset instance."""
    if dataset is None:
        return set()

    for attr in ["_class_ids", "class_ids", "class_id_list", "classes"]:
        value = getattr(dataset, attr, None)
        if value is not None:
            return set(int(x) for x in list(value))

    for attr in ["_targets", "targets"]:
        value = getattr(dataset, attr, None)
        if value is not None:
            return set(int(x) for x in np.asarray(value).reshape(-1))

    imdb = getattr(dataset, "_imdb", None)
    if imdb is not None:
        return set(int(item.get("class")) for item in imdb if "class" in item)

    return set()


def _infer_seen_unseen_classes(train_loader, test_loader):
    """Read seen/unseen classes from dataset-provided fields (XLSA authority)."""
    train_dataset = getattr(train_loader, "dataset", None)
    test_dataset = getattr(test_loader, "dataset", None)

    seen_classes = getattr(train_dataset, "seen_classes", None)
    unseen_classes = getattr(train_dataset, "unseen_classes", None)
    if seen_classes is None and test_dataset is not None:
        seen_classes = getattr(test_dataset, "seen_classes", None)
    if unseen_classes is None and test_dataset is not None:
        unseen_classes = getattr(test_dataset, "unseen_classes", None)

    return (
        sorted(int(x) for x in list(seen_classes)) if seen_classes is not None else None,
        sorted(int(x) for x in list(unseen_classes)) if unseen_classes is not None else None,
    )



def train(cfg, args):
    """
        璁粌涓庤瘎浼颁富娴佺▼锛?
        - 娓呯悊鏄惧瓨缂撳瓨
        - 鍥哄畾闅忔満绉嶅瓙锛堣嫢閰嶇疆鎻愪緵锛?
        - 鍒濆鍖栨棩蹇椾笌鐜
        - 鏋勫缓鏁版嵁鍔犺浇銆佹ā鍨嬨€佽瘎浼板櫒涓庤缁冨櫒
        - 鎵ц璁粌锛堟垨浠呰瘎浼帮級
    """
    # clear up residual cache from previous runs    娓呯悊涓婁竴娆¤繍琛岄仐鐣欑殑 GPU 鏄惧瓨缂撳瓨锛堣嫢鍙敤 GPU锛?
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here
    # ---------- 澶嶇幇瀹為獙锛氬浐瀹氶殢鏈虹瀛?----------
    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)    # 娉ㄦ剰锛氳繖閲屽 Python 鑷甫鐨?random 鍥哄畾涓?0锛岃€岄潪 cfg.SEED 鑻ラ渶瑕佸畬鍏ㄤ竴鑷寸殑澶嶇幇锛屽彲鏀逛负 random.seed(cfg.SEED)
        random.seed(0)
    # -------------------------------------------

    # ---------- 鏃ュ織/鐜鍒濆鍖?----------
    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")
    # -----------------------------------

    # ---------- 鏋勫缓鏁版嵁鍔犺浇鍣?----------
    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    # -----------------------------------

    # ---------- 鏋勫缓妯″瀷 ----------------
    logger.info("Constructing models...")   # 鎵撲竴鏉?INFO 绾у埆鐨勬棩蹇?
    model, cur_device = build_model(cfg)    # 鏍规嵁prompt/linear/adapter 绛変笌楠ㄥ共缃戠粶绫诲瀷锛堝 ViT/Swin锛夋潵瀹炰緥鍖栧搴旂殑妯″瀷涓庢斁缃澶?
    # -----------------------------------

    # 濡傞渶璋冭瘯鍐荤粨绛栫暐锛屽彲涓诲姩鎵撳嵃褰撳墠鍙缁冨弬鏁板垪琛?
    if cfg.MODEL.LOG_TRAINABLE or cfg.SOLVER.DBG_TRAINABLE:
        log_trainable_parameters(model, logger, max_examples_per_group=10)

    if cfg.MODEL.R_SIMILARITY.ENABLE:
        # 浠庢暟鎹泦鑾峰彇绫诲埆绾у睘鎬х煩闃?
        class_attr = getattr(getattr(train_loader, "dataset", None), "class_attributes", None)
        if class_attr is None:
            raise ValueError("R-similarity head enabled but no class_attributes provided by dataset")
        if hasattr(model, "attach_r_similarity_head"):
            # 鍦ㄨ缁冨墠瀹夎R-鐩镐技搴﹀垎绫诲櫒
            model.attach_r_similarity_head(class_attr)

    # ---------- 璇勪及鍣ㄤ笌璁粌鍣?----------
    logger.info("Setting up Evalutator...")
    seen_classes, unseen_classes = _infer_seen_unseen_classes(train_loader, test_loader)
    task_type = str(getattr(cfg.SOLVER, "EVAL_MODE", "standard") or "standard").lower()
    xlsa_cfg = getattr(cfg.DATA, "XLSA", None)
    xlsa_enabled = bool(getattr(xlsa_cfg, "ENABLED", False)) if xlsa_cfg is not None else False
    if xlsa_enabled:
        if task_type not in {"zsl", "gzsl"}:
            raise ValueError(
                f"XLSA mode requires SOLVER.EVAL_MODE in {{'zsl', 'gzsl'}}, got '{task_type}'."
            )
        if not seen_classes or not unseen_classes:
            raise ValueError(
                "XLSA mode requires non-empty dataset.seen_classes and dataset.unseen_classes."
            )
        overlap = set(seen_classes).intersection(set(unseen_classes))
        if overlap:
            raise ValueError(
                "Invalid XLSA split: seen/unseen overlap detected (e.g., {}).".format(
                    sorted(list(overlap))[:10]
                )
            )
        logger.info(
            "XLSA class split ready: seen=%d unseen=%d; seen_head=%s; unseen_head=%s",
            len(seen_classes),
            len(unseen_classes),
            seen_classes[:10],
            unseen_classes[:10],
        )
        test_dataset = getattr(test_loader, "dataset", None)
        if test_dataset is not None:
            test_seen = getattr(test_dataset, "seen_classes", None)
            test_unseen = getattr(test_dataset, "unseen_classes", None)
            if test_seen is not None and sorted(int(x) for x in list(test_seen)) != seen_classes:
                raise ValueError("Mismatch between train/test dataset seen_classes in XLSA mode.")
            if test_unseen is not None and sorted(int(x) for x in list(test_unseen)) != unseen_classes:
                raise ValueError("Mismatch between train/test dataset unseen_classes in XLSA mode.")
    if not seen_classes or not unseen_classes:
        # 鑻ユ棤娉曟帹鏂?ZSL/GZSL 绫诲垝鍒嗭紝鍒欓€€鍖栦负鏍囧噯璇勬祴
        task_type = "standard"
    evaluator = Evaluator(  # 缁勭粐璇勪及鎸囨爣涓庤瘎娴嬮€昏緫
        seen_classes=np.asarray(seen_classes) if seen_classes else None,
        unseen_classes=np.asarray(unseen_classes) if unseen_classes else None,
        task_type=task_type,
        test_include_seen=bool(getattr(getattr(cfg, "DATA", None), "XLSA", None).TEST_INCLUDE_SEEN) if getattr(getattr(cfg, "DATA", None), "XLSA", None) is not None else False,
    )
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)    # Trainer 灏佽浜嗚缁?楠岃瘉/娴嬭瘯鐨勫惊鐜笌淇濆瓨閫昏緫
    # -----------------------------------

    # ---------- 璁粌鎴栭€€鍑?----------
    if train_loader:
        # 甯歌璁粌娴佺▼锛氬湪 train 涓婅缁冿紝鍛ㄦ湡鎬у湪 val/test 涓婅瘎浼?
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")    # 鏋佸皯鏁板満鏅紙渚嬪浠呭仛鎺ㄧ悊/鍒嗘瀽锛夊彲鑳芥病鏈?train_loader
    # -----------------------------------

    # ---------- 浠呰瘎浼版ā寮?----------
    if cfg.SOLVER.TOTAL_EPOCH == 0:         # 鑻?TOTAL_EPOCH 璁句负 0锛屽彲璺宠繃璁粌锛岀洿鎺ュ湪 test 涓婅瘎浼颁竴娆?
        trainer.eval_classifier(test_loader, "test", bool(cfg.MODEL.SAVE_CKPT))
    # -----------------------------------

def main(args):
    """main function to call from workflow."""

    # set up cfg and args 1) 瑙ｆ瀽骞跺悎骞堕厤缃?
    cfg = setup(args)

    # Perform training. 2) 鎵ц璁粌/璇勪及
    train(cfg, args)


if __name__ == '__main__':
    # 瑙ｆ瀽鍛戒护琛屽弬鏁帮紙鏀寔 --config-file 涓庡悗缁殑 KEY VALUE 瑕嗙洊锛?
    args = default_argument_parser().parse_args()
    main(args)
