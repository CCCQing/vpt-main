#!/usr/bin/env python3

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
import warnings

import numpy as np
import random

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager
from src.utils.distributed import get_rank, get_world_size

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(repo_root):
    def run_git(*args):
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_status": status.splitlines(),
    }


def _dataset_manifest(dataset):
    digest = hashlib.sha256()
    records = list(getattr(dataset, "_imdb", []) or [])
    for record in records:
        image_path = str(record.get("im_path", "")).replace("\\", "/")
        class_id = int(record.get("class", -1))
        digest.update(f"{image_path}\t{class_id}\n".encode("utf-8"))
    seen = sorted(int(value) for value in list(getattr(dataset, "seen_classes", []) or []))
    unseen = sorted(int(value) for value in list(getattr(dataset, "unseen_classes", []) or []))
    return {
        "name": str(getattr(dataset, "name", "")),
        "sample_count": len(records),
        "manifest_sha256": digest.hexdigest(),
        "seen_classes": seen,
        "unseen_classes": unseen,
    }


def _write_audit_artifacts(cfg, args, split_loaders):
    if get_rank() != 0:
        return

    output_dir = Path(str(cfg.OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_text = cfg.dump() if hasattr(cfg, "dump") else str(cfg)
    resolved_path = output_dir / "resolved_config.yaml"
    resolved_path.write_text(resolved_text, encoding="utf-8")

    repo_root = Path(__file__).resolve().parent
    graph_raw = str(cfg.MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_PATH or "")
    graph_path = Path(graph_raw)
    if graph_raw and not graph_path.is_absolute():
        graph_path = repo_root / graph_path
    graph_exists = bool(graph_raw) and graph_path.is_file()

    split_manifests = {}
    for split_name, data_loader in split_loaders.items():
        if data_loader is not None:
            split_manifests[split_name] = _dataset_manifest(data_loader.dataset)

    identity = {
        "schema_version": 1,
        "experiment_family": "c_gpp_t0009_factorial",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_identity(repo_root),
        "config_file": str(getattr(args, "config_file", "")),
        "command_opts": list(getattr(args, "opts", []) or []),
        "resolved_config_sha256": _sha256_file(resolved_path),
        "seed": None if cfg.SEED is None else int(cfg.SEED),
        "world_size": int(get_world_size()),
        "num_gpus_config": int(cfg.NUM_GPUS),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "graph": {
            "path": str(graph_path),
            "key": str(cfg.MODEL.SEMANTIC_GRAPH.EXTERNAL_GRAPH_KEY),
            "exists": graph_exists,
            "sha256": _sha256_file(graph_path) if graph_exists else "",
        },
        "switches": {
            "graph_prob_prior_enable": bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE),
            "graph_prob_prior_loss_weight": float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT),
            "attention_mediation_enable": bool(cfg.MODEL.ATTENTION_MEDIATION.ENABLE),
        },
        "splits": split_manifests,
        "training_performed": True,
        "optimizer_created": True,
    }
    (output_dir / "audit_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
        seed = int(cfg.SEED)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")
    logger.info(
        "XLSA protocol mode=%s eval_mode=%s",
        str(cfg.DATA.XLSA.PROTOCOL_MODE),
        str(cfg.SOLVER.EVAL_MODE),
    )

    train_loader, val_loader, test_seen_loader, test_unseen_loader = get_loaders(cfg, logger)

    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    if not cfg.MODEL.R_SIMILARITY.ENABLE:
        raise ValueError("Current prompt-only XLSA mainline requires MODEL.R_SIMILARITY.ENABLE=True.")
    class_attr = train_loader.dataset.class_attributes
    if class_attr is None:
        raise ValueError("R-similarity head enabled but no class_attributes provided by dataset")
    if hasattr(model, "attach_r_similarity_head"):
        model.attach_r_similarity_head(class_attr)

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

    _write_audit_artifacts(
        cfg,
        args,
        {
            "train": train_loader,
            "val": val_loader,
            "test_seen": test_seen_loader,
            "test_unseen": test_unseen_loader,
        },
    )

    # -----------------------------------
    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_seen_loader, test_unseen_loader)
    else:
        print("No train loader presented. Exit")

    # -------------------------------------
    if cfg.SOLVER.TOTAL_EPOCH == 0:
        if test_seen_loader is not None and str(cfg.SOLVER.EVAL_MODE).lower() == "gzsl":
            trainer.eval_classifier(test_seen_loader, "test_seen")
        if test_unseen_loader is not None:
            trainer.eval_classifier(test_unseen_loader, "test_unseen")

def main(args):
    """main function to call from workflow."""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)



