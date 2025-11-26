#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""JSONDataset / XLSA 调试脚本.

本脚本帮助快速验证 ``src.data.datasets.json_dataset.JSONDataset`` 中
对 JSON 标注和 xlsa17 (res101.mat + att_splits.mat) 的解析是否符合预期。

新增 YAML 配置：默认在 ``configs/TEST/run_json_dataset_demo.yaml`` 中记录所有
脚本参数，避免每次都通过命令行传递。该文件本质上是 ``CfgNode``
导出的 YAML，因此既可以直接编辑，也可以用 ``--init-settings`` 重新
生成模板。命令行选项仍然可用，会在读取 YAML 后作为**覆盖**值。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import scipy.io as sio

import numpy as np
import yaml
from PIL import Image

from src.configs.config import get_cfg
from src.configs.config_node import CfgNode  # 用于构造脚本自己的 settings YAML
from src.data.datasets.json_dataset import CUB200Dataset


# ---------------------------------------------------------------------------
# JSON 模式相关的辅助函数（生成极简 dummy 数据集）
# ---------------------------------------------------------------------------
def _write_image(path: Path, rgb: Iterable[int]) -> None:
    """生成 256x256 纯色 JPEG，用于 dummy 数据集。

    若目标 path 已存在，则直接返回，不重复写入。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[..., 0] = rgb[0]
    arr[..., 1] = rgb[1]
    arr[..., 2] = rgb[2]
    Image.fromarray(arr).save(path)


def _prepare_dummy_dataset(root: Path) -> None:
    """在 ``root`` 下构造极简 JSON 数据与彩色图片。"""
    images_root = root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    # rel_path -> (class_name, class_id, color)
    samples: Dict[str, Tuple[str, int, Tuple[int, int, int]]] = {
        "001.DummyA/img_0001.jpg": ("001.DummyA", 0, (255, 140, 0)),
        "001.DummyA/img_0002.jpg": ("001.DummyA", 0, (240, 128, 128)),
        "002.DummyB/img_0001.jpg": ("002.DummyB", 1, (65, 105, 225)),
        "002.DummyB/img_0002.jpg": ("002.DummyB", 1, (50, 205, 50)),
    }

    for rel_path, (_, _, color) in samples.items():
        _write_image(images_root / rel_path, color)

    splits = {
        "train": ["001.DummyA/img_0001.jpg", "002.DummyB/img_0001.jpg"],
        "val": ["001.DummyA/img_0002.jpg"],
        "test": ["002.DummyB/img_0002.jpg"],
    }

    for split_name, rel_paths in splits.items():
        anno = {rel_path: samples[rel_path][1] for rel_path in rel_paths}
        with open(root / f"{split_name}.json", "w", encoding="utf-8") as handle:
            json.dump(anno, handle)


# ---------------------------------------------------------------------------
# YAML settings（只管这个 demo 脚本自身的参数，不是模型大 cfg）
# ---------------------------------------------------------------------------

# ⚠️ 这里我简化了一下路径，假设 run_json_dataset_demo.py 和 configs 在同一个工程根目录下
DEFAULT_SETTINGS_PATH = (
    Path(__file__).resolve().parent / "configs" / "TEST" / "run_json_dataset_demo.yaml"
)


def _str2bool(value: str) -> bool:
    """把命令行字符串安全地转成 bool，用于覆盖 YAML 设置。"""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Cannot interpret boolean value from '{value}'"
    )


def _build_default_settings() -> CfgNode:
    """构造一份本脚本用的默认设置（不是训练用 cfg）。"""
    cfg = CfgNode()
    cfg.MODE = "xlsa"          # "json" / "xlsa"
    cfg.SPLIT = "train"        # "train" / "val" / "test"
    cfg.DATA_ROOT = os.fspath(Path("tmp/cub_dummy"))
    cfg.SKIP_DUMMY = False
    cfg.XLSA_ROOT = ""
    cfg.GZSL = False
    cfg.USE_TRAINVAL = False
    return cfg


def _write_settings_template(path: Path) -> None:
    """写出默认 settings YAML 模板到指定路径。"""
    template = _build_default_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(template.dump())
    print(f"[INFO] 默认配置已写入 {path}")


def _load_settings(path: Optional[Path]) -> CfgNode:
    """从 YAML 文件加载 settings；若不存在则返回默认并提示。

    ⚠️ 这里我们自己用 UTF-8 读取 YAML，避免 fvcore / yacs 在 Windows 下用 GBK 读文件。
    """
    cfg = _build_default_settings()
    if path is None:
        return cfg
    if not path.exists():
        print(f"[WARN] settings 文件 {path} 不存在，已创建模板供后续编辑。")
        _write_settings_template(path)
        return cfg

    # 1) 用 UTF-8 手动读取 YAML → dict
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"settings 文件 {path} 的 YAML 顶层必须是字典(mapping)，当前类型: {type(data)}"
        )

    # 2) 用 merge_from_other_cfg 合并，完全绕开 merge_from_file(内部会用 GBK)
    cfg.merge_from_other_cfg(CfgNode(data))
    print(f"[INFO] 从 {path} 读取设置")
    return cfg


# ---------------------------------------------------------------------------
# 主流程：根据 mode/json/xlsa 配置 cfg 并运行数据集
# ---------------------------------------------------------------------------
def _configure_common(cfg, data_root: Path) -> None:
    """配置不区分 JSON / XLSA 的公共部分。"""
    cfg.DATA.NAME = "CUB"
    cfg.DATA.DATAPATH = os.fspath(data_root)
    cfg.DATA.NUMBER_CLASSES = 200
    cfg.DATA.CROPSIZE = 224
    cfg.DATA.XLSA.ENABLED = False
    cfg.DATA.ATTS_PATH = ""


def _configure_for_json(cfg, data_root: Path, skip_dummy: bool) -> None:
    """针对 JSON 模式的配置与数据准备。"""
    if not skip_dummy:
        _prepare_dummy_dataset(data_root)
    print(f"[INFO] JSON 模式，数据根目录: {data_root}")


def _configure_for_xlsa(cfg, data_root: Path, xlsa_root: Path, gzsl: bool, use_trainval: bool) -> None:
    """针对 XLSA 模式（xlsa17）的配置。"""
    if xlsa_root is None:
        raise ValueError("XLSA 模式下必须提供 xlsa_root (含 res101.mat / att_splits.mat)")

    res101_path = xlsa_root / "res101.mat"
    split_path = xlsa_root / "att_splits.mat"
    if not res101_path.exists():
        raise FileNotFoundError(f"res101.mat not found: {res101_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"att_splits.mat not found: {split_path}")

    cfg.DATA.XLSA.ENABLED = True
    cfg.DATA.XLSA.RES101_PATH = os.fspath(res101_path)
    cfg.DATA.XLSA.SPLIT_PATH = os.fspath(split_path)

    cfg.DATA.XLSA.TRAIN_KEY = "train_loc"
    cfg.DATA.XLSA.VAL_KEY = "val_loc"
    cfg.DATA.XLSA.TRAINVAL_KEY = "trainval_loc"
    cfg.DATA.XLSA.TRAIN_USE_TRAINVAL = bool(use_trainval)
    cfg.DATA.XLSA.TEST_KEY = "test_unseen_loc"
    cfg.DATA.XLSA.TEST_SEEN_KEY = "test_seen_loc"
    cfg.DATA.XLSA.TEST_INCLUDE_SEEN = bool(gzsl)

    cfg.SOLVER.EVAL_MODE = "gzsl" if gzsl else "zsl"
    cfg.SOLVER.GZSL = bool(gzsl)
    cfg.GZSL = bool(gzsl)


    cfg.DATA.ATTS_PATH = os.fspath(split_path)
    cfg.DATA.ATTS_KEY = "att"

    print(
        "[INFO] XLSA 模式：data_root=%s, xlsa_root=%s, use_trainval=%s, GZSL=%s"
        % (data_root, xlsa_root, use_trainval, gzsl)
    )


def _describe_dataset(ds: CUB200Dataset, split: str) -> None:
    """打印当前数据集和 XLSA 源文件的一些关键信息，用于 sanity check。"""

    # ----------------- 1. 先看 Dataset 自身 -----------------
    cfg = ds.cfg  # JSONDataset 在 __init__ 里保存了 self.cfg = cfg

    print(f"====== CUB {split} split (Dataset view) ======")
    # 样本总数，方便你对照 xlsa17 各个 *_loc 的长度
    print("num_images (len(ds)):", len(ds))
    # cfg 中的类别数（cfg.DATA.NUMBER_CLASSES）
    print("num_classes (cfg):   ", ds.get_class_num())

    # 打印前 5 个样本的路径和类别（来自 _imdb）
    print("\n[Dataset] 前 5 个样本的路径和类别：")
    for i in range(min(5, len(ds))):
        rec = ds._imdb[i]
        im_path = rec["im_path"]
        cls_id = rec["class"]
        print(
            f"  idx={i}, path={im_path}, class={cls_id}, "
            f"exists={os.path.exists(im_path)}"
        )

    # 取第一个样本，检查 transform 后的张量形状、标签、属性
    sample = ds[0]
    print("\n[Dataset] sample[0] 结构：")
    print("  image shape:", tuple(sample["image"].shape))  # 一般是 (3, 224, 224)
    print("  label:      ", sample["label"])
    if "attribute" in sample:
        print("  attribute shape:", tuple(sample["attribute"].shape))
    else:
        print("  attribute:      <missing>  (sample[0] 中没有 attribute 字段)")

    # ----------------- 2. 再看 XLSA 的 att_splits.mat -----------------
    xlsa_cfg = getattr(cfg.DATA, "XLSA", None)
    if xlsa_cfg is not None:
        split_path = xlsa_cfg.SPLIT_PATH or cfg.DATA.ATTS_PATH
    else:
        split_path = cfg.DATA.ATTS_PATH

    if split_path and os.path.isfile(split_path):
        print(f"\n====== att_splits.mat ({split_path}) ======")
        split_mat = sio.loadmat(split_path)

        # 关心的几个典型 split：train / val / trainval / test_seen / test_unseen
        for key in ["train_loc", "val_loc", "trainval_loc", "test_seen_loc", "test_unseen_loc"]:
            if key not in split_mat:
                continue
            arr = np.asarray(split_mat[key]).squeeze()
            if arr.size == 0:
                continue
            # 通常是 1-based 索引，后面会减 1
            print(
                f"  {key:15s}: shape={arr.shape}, "
                f"min={int(arr.min())}, max={int(arr.max())}, "
                f"unique={len(np.unique(arr))}"
            )
    else:
        print("\n[WARN] 未找到 att_splits.mat / ATTS_PATH，跳过 *_loc 信息打印。")

    # ----------------- 3. 最后看 XLSA 的 res101.mat -----------------
    res_path = ""
    if xlsa_cfg is not None:
        res_path = xlsa_cfg.RES101_PATH

    if res_path and os.path.isfile(res_path):
        print(f"\n====== res101.mat ({res_path}) ======")
        res_mat = sio.loadmat(res_path)

        image_files_raw = res_mat.get("image_files")
        labels_raw = res_mat.get("labels")

        if image_files_raw is None or labels_raw is None:
            print("  [ERROR] res101.mat 中没有 'image_files' 或 'labels' 字段。")
        else:
            print("  image_files_raw shape:", image_files_raw.shape)
            print("  labels_raw shape:     ", labels_raw.shape)

            # 将 MATLAB cell array 转成 Python list[str]（这里只取前几个做展示）
            def _matlab_cell_to_list_head(cell, n_head=5):
                flat = np.atleast_1d(cell.squeeze())
                out = []
                for x in flat[:n_head]:
                    v = x
                    if isinstance(v, np.ndarray):
                        v = v.squeeze()
                        if v.size == 1:
                            v = v.item()
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    out.append(str(v))
                return out

            image_list_head = _matlab_cell_to_list_head(image_files_raw, n_head=5)
            labels_all = labels_raw.squeeze().astype(np.int64)

            print("  num_images in res101:", labels_all.shape[0])
            print("  labels min / max:    ", int(labels_all.min()), int(labels_all.max()))

            print("  前 5 个 image_files / labels：")
            for i, (p, y) in enumerate(zip(image_list_head, labels_all[:len(image_list_head)])):
                print(f"    res[{i}]: path={p}, label={int(y)}")
    else:
        print("\n[WARN] 未找到 res101.mat (DATA.XLSA.RES101_PATH)，跳过原始 image_files/labels 打印。")





def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check JSONDataset with JSON/XLSA inputs (支持 YAML settings)"
    )
    parser.add_argument("--mode", choices=["json", "xlsa"], default=None, help="覆盖 YAML 中的 mode")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="覆盖 YAML 中的 split")
    parser.add_argument(
        "--data-root", type=Path, default=None, help="覆盖 YAML：JSON/XLSA 数据根目录"
    )
    parser.add_argument(
        "--skip-dummy", type=_str2bool, default=None, help="覆盖 YAML：JSON 模式是否跳过 dummy 数据"
    )
    parser.add_argument(
        "--xlsa-root", type=Path, default=None, help="覆盖 YAML：xlsa17 CUB 目录 (含 res101.mat / att_splits.mat)"
    )
    parser.add_argument(
        "--gzsl", type=_str2bool, default=None, help="覆盖 YAML：XLSA 模式 test split 是否合并 test_seen_loc"
    )
    parser.add_argument(
        "--use-trainval", type=_str2bool, default=None, help="覆盖 YAML：train split 是否改用 trainval_loc"
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="用于读取/保存脚本参数的 YAML 文件路径",
    )
    parser.add_argument(
        "--init-settings",
        action="store_true",
        help="仅写出默认 YAML 模板到 --settings-file 然后退出",
    )
    args = parser.parse_args()

    # ① 只初始化 YAML 模板的情况
    if args.init_settings:
        target = args.settings_file or DEFAULT_SETTINGS_PATH
        _write_settings_template(target)
        return

    # ② 先从 YAML 里读一份 settings（此处已强制 UTF-8）
    settings = _load_settings(args.settings_file)

    # ③ 再用命令行覆盖（若提供）
    if args.mode is not None:
        settings.MODE = args.mode
    if args.split is not None:
        settings.SPLIT = args.split
    if args.data_root is not None:
        settings.DATA_ROOT = os.fspath(args.data_root)
    if args.skip_dummy is not None:
        settings.SKIP_DUMMY = bool(args.skip_dummy)
    if args.xlsa_root is not None:
        settings.XLSA_ROOT = os.fspath(args.xlsa_root)
    if args.gzsl is not None:
        settings.GZSL = bool(args.gzsl)
    if args.use_trainval is not None:
        settings.USE_TRAINVAL = bool(args.use_trainval)

    # ④ 展开 settings
    data_root = Path(settings.DATA_ROOT)
    xlsa_root = Path(settings.XLSA_ROOT) if settings.XLSA_ROOT else None
    skip_dummy = bool(settings.SKIP_DUMMY)
    gzsl = bool(settings.GZSL)
    use_trainval = bool(settings.USE_TRAINVAL)
    split = settings.SPLIT
    mode = settings.MODE

    # ⑤ 构造项目大 cfg，并填入通用字段
    cfg = get_cfg()
    _configure_common(cfg, data_root)

    # ⑥ 根据 mode 选择 JSON / XLSA 流程
    if mode == "json":
        _configure_for_json(cfg, data_root, skip_dummy)
    else:
        _configure_for_xlsa(cfg, data_root, xlsa_root, gzsl, use_trainval)

    # ⑦ 实例化 CUB200Dataset，并打印信息
    dataset = CUB200Dataset(cfg, split=split)
    _describe_dataset(dataset, split)


if __name__ == "__main__":
    main()
