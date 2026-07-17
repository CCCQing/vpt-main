#!/usr/bin/env python3

import os
import torch
import warnings

import numpy as np
import random

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model, log_model_info, wrap_distributed_model
from src.utils.file_io import PathManager
from src.utils.dataset_manifest import write_xlsa_dataset_manifest
from src.utils.run_artifacts import (
    collect_distributed_runtime_checks,
    write_reproducibility_manifest,
    write_resolved_config,
    write_trainable_parameter_manifest,
)
from src.utils.reproducibility import apply_rank_runtime_seed, seed_streams
from src.utils import distributed as du

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def _sync_xlsa_protocol(cfg):
    """Use DATA.XLSA.PROTOCOL_MODE as the single protocol truth source."""

    protocol_mode = str(cfg.DATA.XLSA.PROTOCOL_MODE or "dev").lower()
    protocol_to_eval = {
        "dev": "zsl",
        "final_zsl": "zsl",
        "final_gzsl": "gzsl",
    }
    if protocol_mode not in protocol_to_eval:
        raise ValueError(
            "Unsupported DATA.XLSA.PROTOCOL_MODE='{}', expected one of ['dev', 'final_zsl', 'final_gzsl'].".format(
                cfg.DATA.XLSA.PROTOCOL_MODE
            )
        )
    cfg.DATA.XLSA.PROTOCOL_MODE = protocol_mode
    cfg.SOLVER.EVAL_MODE = protocol_to_eval[protocol_mode]


def _merge_local_path_cfg_if_exists(cfg):
    """Merge src/configs/local_path.yaml if present (machine-local path overrides)."""
    local_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "configs", "local_path.yaml",)
    if os.path.isfile(local_cfg):
        cfg.merge_from_file(local_cfg)
        print(f"[config] merged local path overrides: {local_cfg}")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    _merge_local_path_cfg_if_exists(cfg)
    cfg.merge_from_list(args.opts)
    _sync_xlsa_protocol(cfg)

    node = os.environ.get("SLURMD_NODENAME")
    if node:
        cfg.DIST_INIT_PATH = f"tcp://{node}:12399"

    # setup output dir
    # <OUTPUT_DIR>/<DATA.NAME>/<DATA.FEATURE>/lr<lr>_wd<wd>/run<count>
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    """Build loaders according to the Xian protocol stage."""
    protocol_mode = str(cfg.DATA.XLSA.PROTOCOL_MODE).lower()
    val_loader = None
    test_seen_loader = None
    test_unseen_loader = None

    if protocol_mode == "dev":
        logger.info("Loading dev train data (train_loc)...")
        train_loader = data_loader.construct_train_loader(cfg)
        logger.info("Loading dev unseen validation data (val_loc)...")
        val_loader = data_loader.construct_val_loader(cfg)
    elif protocol_mode == "final_zsl":
        logger.info("Loading final train data (trainval_loc)...")
        train_loader = data_loader.construct_trainval_loader(cfg)
        logger.info("Loading final ZSL test data (test_unseen_loc)...")
        test_unseen_loader = data_loader.construct_test_unseen_loader(cfg)
    elif protocol_mode == "final_gzsl":
        logger.info("Loading final train data (trainval_loc)...")
        train_loader = data_loader.construct_trainval_loader(cfg)
        logger.info("Loading final GZSL seen test data (test_seen_loc)...")
        test_seen_loader = data_loader.construct_test_seen_loader(cfg)
        logger.info("Loading final GZSL unseen test data (test_unseen_loc)...")
        test_unseen_loader = data_loader.construct_test_unseen_loader(cfg)
    else:
        raise ValueError("Unsupported DATA.XLSA.PROTOCOL_MODE='{}'".format(cfg.DATA.XLSA.PROTOCOL_MODE))
    return train_loader, val_loader, test_seen_loader, test_unseen_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here
    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")
    streams = seed_streams(cfg.SEED)
    logger.info(
        "[reproducibility] master_seed=%s classifier_init_seed=%s prompt_init_seed=%s data_order_seed=%s",
        str(cfg.SEED),
        str(streams["classifier_init"]),
        str(streams["prompt_init"]),
        str(streams["data_order"]),
    )
    resolved_config_path = write_resolved_config(cfg)
    if resolved_config_path is not None:
        logger.info("Wrote complete resolved config: %s", resolved_config_path)
    logger.info(
        "XLSA protocol mode=%s eval_mode=%s",
        str(cfg.DATA.XLSA.PROTOCOL_MODE),
        str(cfg.SOLVER.EVAL_MODE),
    )

    train_loader, val_loader, test_seen_loader, test_unseen_loader = get_loaders(cfg, logger)
    manifest_path = write_xlsa_dataset_manifest(
        cfg,
        {
            "train": train_loader.dataset if train_loader is not None else None,
            "val_unseen": val_loader.dataset if val_loader is not None else None,
            "test_seen": test_seen_loader.dataset if test_seen_loader is not None else None,
            "test_unseen": test_unseen_loader.dataset if test_unseen_loader is not None else None,
        },
    )
    if manifest_path is not None:
        logger.info("Wrote XLSA dataset manifest: %s", manifest_path)

    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    if not cfg.MODEL.R_SIMILARITY.ENABLE:
        raise ValueError("Current prompt-only XLSA mainline requires MODEL.R_SIMILARITY.ENABLE=True.")
    class_attr = train_loader.dataset.class_attributes
    if class_attr is None:
        raise ValueError("R-similarity head enabled but no class_attributes provided by dataset")
    if hasattr(model, "attach_r_similarity_head"):
        model.attach_r_similarity_head(class_attr)
    trainable_manifest_path = write_trainable_parameter_manifest(cfg, model)
    if trainable_manifest_path is not None:
        logger.info("Wrote effective trainable-parameter manifest: %s", trainable_manifest_path)
    log_model_info(model, verbose=False, label="Effective model before DDP wrapping")
    model = wrap_distributed_model(model, cfg)

    # ------------------------------------
    train_dataset = train_loader.dataset
    test_seen_dataset = test_seen_loader.dataset if test_seen_loader is not None else None
    test_unseen_dataset = test_unseen_loader.dataset if test_unseen_loader is not None else None

    task_type = str(cfg.SOLVER.EVAL_MODE).lower()
    if not bool(cfg.DATA.XLSA.ENABLED):
        raise ValueError("Current training entry expects DATA.XLSA.ENABLED=True.")

    seen_classes = sorted(int(x) for x in list(train_dataset.seen_classes))
    unseen_classes = sorted(int(x) for x in list(train_dataset.unseen_classes))

    if task_type not in {"zsl", "gzsl"}:
        raise ValueError(f"XLSA mode requires SOLVER.EVAL_MODE in {{'zsl', 'gzsl'}}, got '{task_type}'.")

    if not seen_classes or not unseen_classes:
        raise ValueError("XLSA mode requires non-empty dataset.seen_classes and dataset.unseen_classes.")

    overlap = set(seen_classes).intersection(set(unseen_classes))
    if overlap:
        raise ValueError("Invalid XLSA split: seen/unseen overlap detected (e.g., {}).".format(
                sorted(list(overlap))[:10]))
    if test_seen_dataset is not None and test_unseen_dataset is not None:
        test_seen = sorted(int(x) for x in list(test_seen_dataset.seen_classes))
        test_unseen = sorted(int(x) for x in list(test_unseen_dataset.unseen_classes))
        if test_seen != seen_classes:
            raise ValueError("Mismatch between train/test dataset seen_classes in XLSA mode.")
        if test_unseen != unseen_classes:
            raise ValueError("Mismatch between train/test dataset unseen_classes in XLSA mode.")
    evaluator = Evaluator(task_type=task_type)

    # ------------------------------------
    trainer = Trainer(cfg, model, evaluator, cur_device)
    runtime_seed = apply_rank_runtime_seed(cfg.SEED, du.get_rank())
    distributed_checks = collect_distributed_runtime_checks(
        model,
        trainer.cls_criterion,
        train_loader,
        runtime_seed,
    )
    if bool(distributed_checks.get("enabled")):
        ddp_failures = []
        if not bool(distributed_checks["initial_parameter_fingerprint"]["all_ranks_equal"]):
            ddp_failures.append("initial model/loss parameters differ across ranks")
        if not bool(distributed_checks["rank_runtime_seeds_unique"]):
            ddp_failures.append("rank runtime seeds are not unique")
        if not bool(distributed_checks["train_sampler_partition_configured"]):
            ddp_failures.append("DistributedSampler rank/replica/seed configuration is invalid")
        if ddp_failures:
            raise RuntimeError("DDP reproducibility gate failed: {}".format("; ".join(ddp_failures)))
    reproducibility_manifest_path = write_reproducibility_manifest(
        cfg,
        {
            "train": train_loader,
            "val_unseen": val_loader,
            "test_seen": test_seen_loader,
            "test_unseen": test_unseen_loader,
        },
        distributed_checks=distributed_checks,
    )
    if reproducibility_manifest_path is not None:
        logger.info("Wrote reproducibility manifest: %s", reproducibility_manifest_path)
    logger.info(
        "[reproducibility] rank_runtime_seed=%s rank=%d world_size=%d",
        str(runtime_seed),
        du.get_rank(),
        du.get_world_size(),
    )

    # -----------------------------------
    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_seen_loader, test_unseen_loader)
    else:
        print("No train loader presented. Exit")

def main(args):
    """main function to call from workflow."""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)



