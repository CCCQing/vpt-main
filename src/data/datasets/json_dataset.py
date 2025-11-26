#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars
使用 JSON 标注文件的通用数据集定义，支持 CUB、NABirds、Flower、Dogs、Cars 等多个数据集。"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter
# ------------ 11.14 新增：XLSA / 属性加载相关 -------------
import scipy.io as sio                          # 新增：用于读取 .mat 语义属性文件
from typing import Dict, Iterable, List, Optional, Sequence     # 新增：用于类型注解 List / Optional等
# ---------------------------------
from ..transforms import get_transforms         # 图像增广与预处理
from ...utils import logging
from ...utils.io_utils import read_json
logger = logging.get_logger("visual_prompt")

class AttributeLoaderMixin:
    """
    AttributeLoaderMixin：封装“类级语义属性（attributes）”的通用加载逻辑。

    设计目的：
    - 一些经典 ZSL 数据集（CUB/AWA2/SUN）会提供类级属性矩阵 att（att_splits.mat 里的 att）；
    - 不同 Dataset 子类（CUB、AWA2、SUN 等）可以复用本 mixin 提供的加载 / 对齐逻辑；
    - JSONDataset 只需要继承本 mixin，就具备属性加载能力，无需在每个数据集中重复写代码。
    """

    # 这三个是“类属性”的类型标注（typing hint），实际使用时在 __init__ 里会按实例级重新赋值。
    has_attributes: bool = False                        # 是否已经成功加载过类级属性
    class_attributes: Optional[torch.Tensor] = None     # [num_classes, att_dim]，每类一行属性向量
    _class_has_attribute: Optional[torch.Tensor] = None # [num_classes]，True 表示该类在属性矩阵中有定义

    def _load_attributes_if_available(self, cfg):
        """
        从 cfg.DATA.ATTS_PATH 指定的 .mat 文件中加载类级属性向量（通常也是 att_splits.mat），
        作为“通用 fallback 入口”：

        - 如果之前已经通过 XLSA split（_load_attributes_from_split_dict）加载过属性，
          self.has_attributes 会被置 True，此时直接 return，避免重复读取。
        - 否则：
          1）检查 DATA.ATTS_PATH 是否配置，且文件是否存在；
          2）从 .mat 中读取 key=ATTS_KEY（默认 'att'）；
          3）调用 _populate_class_attributes 统一完成 numpy → torch → class 映射。
        """
        # 若已有属性（例如已在 XLSA 构建阶段加载），则不再重复加载
        if self.has_attributes:
            return

        # 从配置中取属性文件路径（例如 D:/.../CUB/att_splits.mat）
        atts_path = getattr(cfg.DATA, "ATTS_PATH", None)
        if not atts_path:
            logger.info("No DATA.ATTS_PATH specified, skip loading attributes.")
            return
        if not os.path.exists(atts_path):
            logger.warning("Attribute file %s not found, skip.", atts_path)
            return

        logger.info("Loading class-level attributes from %s ...", atts_path)
        # 读取 .mat 文件为字典形式（key -> numpy 数组）
        atts_mat = sio.loadmat(atts_path)
        # 属性矩阵在 .mat 中对应的 key 名（默认 'att'）
        att_key = getattr(cfg.DATA, "ATTS_KEY", "att")
        if att_key not in atts_mat:
            logger.warning(
                "Key '%s' not found in %s, cannot load attributes.", att_key, atts_path
            )
            return

        # 将原始 numpy 数组交给统一的处理函数，source_name 仅用于日志打印
        self._populate_class_attributes(
            np.asarray(atts_mat[att_key]),
            cfg,
            source_name=atts_path,
        )

    def _load_attributes_from_split_dict(self, cfg, split_mat, source_path: str):
        """
        直接从“已经加载好的 att_splits.mat 字典 split_mat”中注入属性。

        典型场景：
        - 在 XLSA 模式下，_construct_imdb_from_xlsa 已经通过
          split_mat = sio.loadmat(att_splits.mat) 得到一个字典；
        - 这里就可以复用这个 split_mat，从其中的 att 字段直接构造类级属性，
          避免再额外打开一次磁盘。
        """
        # 若已经加载过属性，则不需要再次注入
        if self.has_attributes:
            return

        # 从配置中拿属性矩阵在 .mat 中的 key 名（默认 'att'）
        att_key = getattr(cfg.DATA, "ATTS_KEY", "att")
        if att_key not in split_mat:
            # 某些数据集可能没有 att 字段，此时直接跳过
            return

        # 取出 numpy 数组，交由 _populate_class_attributes 处理
        self._populate_class_attributes(
            np.asarray(split_mat[att_key]),
            cfg,
            # source_name 记录为 “att_splits.mat::att”，方便之后在日志中区分来源
            source_name="{}::{}".format(source_path, att_key) if source_path else att_key,
        )

    def _populate_class_attributes(self, atts_np: np.ndarray, cfg, source_name: str):
        """
        通用的 numpy -> torch 属性写入逻辑，供不同数据来源（XLSA split / ATTS_PATH）复用。

        功能：
        1）检查当前数据集是否在支持列表（CUB/AWA/AWA2/SUN）中；
        2）将 att 矩阵规整成 [num_classes, att_dim] 形状；
        3）按 self._class_ids（一般为 0..C-1）对每个类别取出一行属性；
        4）填充 class_attributes（属性矩阵）与 _class_has_attribute（可用性掩码）；
        5）将 has_attributes 置为 True，并打印日志说明加载情况。
        """
        # 只在经典 ZSL 数据集上启用属性逻辑，避免其它数据集误用
        dataset_key = cfg.DATA.NAME.lower()
        supported = {"cub", "awa", "awa2", "sun"}
        if dataset_key not in supported:
            logger.info(
                "Attribute loading is currently only implemented for CUB/AWA/SUN. "
                "Dataset '%s' will skip attribute injection.",
                cfg.DATA.NAME,
            )
            return

        # 至少保证是二维矩阵：[num_rows, att_dim]
        if atts_np.ndim < 2:
            atts_np = np.atleast_2d(atts_np)

        # 典型 att_splits.mat 中 att 的形状为 [att_dim, num_classes]，
        # 若行数 != 类数、但列数 == 类数，则说明需要转置为 [num_classes, att_dim]
        if atts_np.shape[0] != len(self._class_ids) and atts_np.shape[1] == len(self._class_ids):
            atts_np = atts_np.T

        # 一些数据集可能用 1-based id 存储类索引，用 ATTS_ID_SHIFT 做纠偏（一般为 0）
        id_shift = getattr(cfg.DATA, "ATTS_ID_SHIFT", 0)
        total_rows = atts_np.shape[0]   # 属性矩阵行数（原始类 id 数量）
        att_dim = atts_np.shape[1]      # 属性维度（例如 CUB 为 312）

        attr_tensor = torch.from_numpy(atts_np).float()
        # 初始化一个 [num_classes, att_dim] 的零矩阵，对应连续类 id（0..C-1）
        class_attributes = torch.zeros((len(self._class_ids), att_dim), dtype=torch.float32)
        has_attr_mask = torch.zeros(len(self._class_ids), dtype=torch.bool)
        missing_original_ids = []

        # 遍历每个连续类 id（cont_id）及其原始 id（orig_id）
        # 对于 XLSA 场景，orig_id 和 cont_id 通常相同（0..C-1）
        for cont_id, orig_id in enumerate(self._class_ids):
            # 根据 shift 后的原始 id 确定在属性矩阵中的行索引
            row_idx = int(orig_id) + id_shift
            if row_idx < 0 or row_idx >= total_rows:
                # 若越界，说明该类在属性矩阵中缺失（例如某些类没有标注 att）
                missing_original_ids.append(orig_id)
                continue
            # 将该行属性写入连续类 id 对应位置
            class_attributes[cont_id] = attr_tensor[row_idx]
            has_attr_mask[cont_id] = True

        available = int(has_attr_mask.sum().item())
        if available == 0:
            logger.warning(
                "No matching attributes found for classes in split '%s'.", self._split
            )
            return

        if missing_original_ids:
            logger.warning(
                "Attributes missing for %d/%d classes (e.g., %s). Samples from these "
                "classes will not include attribute vectors.",
                len(missing_original_ids),
                len(self._class_ids),
                missing_original_ids[:5],
            )

        # 将最终结果挂到实例上，供 __getitem__ 或 Trainer 使用
        self.class_attributes = class_attributes          # [num_classes, att_dim]
        self._class_has_attribute = has_attr_mask         # [num_classes]
        self.has_attributes = True                        # 标记：已经有可用属性

        logger.info(
            "Loaded attributes with shape %s from %s; available for %d/%d classes.",
            tuple(class_attributes.shape),
            source_name,
            available,
            len(self._class_ids),
        )

class JSONDataset(AttributeLoaderMixin, torch.utils.data.Dataset):
    """
    一个通用的图像数据集基类：通过 JSON 标注构建样本列表，再交给 DataLoader 使用。

    支持两种构建 imdb 的方式：
    1）纯 JSON 模式：
        - 读取 <DATAPATH>/<split>.json（格式：{"xxx.jpg": class_id, ...}）
        - 子类通过 get_imagedir() 告诉“图像根目录”在哪（例如 CUB 的 images/）
        - 拼成形如 {"im_path": "/abs/path/to/xxx.jpg", "class": cont_id} 的 imdb 列表

    2）XLSA 模式（xlsa17 协议）：
        - 当 cfg.DATA.XLSA.ENABLED = True 时：
          从 res101.mat + att_splits.mat 中读取：
            * image_files：原始图像相对路径；
            * labels      ：类标签（1-based）；
            * trainval_loc / test_seen_loc / test_unseen_loc 等样本索引；
          然后用这些 index 去索引“原图路径 + 标签”，构造 imdb。
        - 属性 att 则从 att_splits.mat 中读取，用于 ZSL / GZSL 的语义对齐。
    """
    def __init__(self, cfg, split):
        """
        :param cfg: 全局配置对象（yacs CfgNode），包含数据路径、类别数、裁剪尺寸等。
        :param split: 数据划分，支持 {"train", "val", "test", "trainval"}。

        整体流程：
        1. 保存基本配置（路径、名称、split 类型等）；
        2. 调用 _construct_imdb(cfg) 构建 imdb：
           - 若启用 XLSA，则优先尝试 _construct_imdb_from_xlsa；
           - 否则退回到纯 JSON 的构建方式；
        3. 根据 split 设置图像增广（transform）；
        4. 尝试从 .mat 文件中加载类级属性向量 att（用于 ZSL / GZSL）。
        """
        assert split in {
            "train",
            "val",
            "test",
            "trainval",
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))

        self.cfg = cfg           # 保存配置
        self._split = split      # 当前使用的数据划分（train/val/test）
        self.name = cfg.DATA.NAME
        self.data_dir = cfg.DATA.DATAPATH        # 数据集根目录
        self.data_percentage = cfg.DATA.PERCENTAGE  # 训练阶段是否按比例采样子集。若 <1.0，表示只用一部分训练数据做子集

        # AttributeLoaderMixin 中声明的是“类级属性的类型 hint”，这里按实例级初始化：
        # 当前这个 Dataset 实例是否已经加载了类级属性
        self.has_attributes = False
        # 保存每个类的属性向量，[num_classes, att_dim]
        self.class_attributes = None
        # 标记哪些类有属性，哪些类缺失
        self._class_has_attribute = None

        # 1) 按当前配置构建 imdb：
        #    - 若启用 XLSA，则优先从 res101.mat + att_splits.mat 构建 imdb；
        #    - 若未启用或失败，则退回到 JSON 标注模式。
        self._construct_imdb(cfg)

        # 2) 为当前 split 准备图像 transform（增广 / 预处理），例如：
        #    - train: 随机裁剪 + 水平翻转 + 归一化；
        #    - val/test: center crop + 归一化。
        self.transform = get_transforms(split, cfg.DATA.CROPSIZE)

        # 3) 从 DATA.ATTS_PATH 中尝试加载属性（如果配置了 ATTS_PATH 且尚未加载属性）；
        #    - 若已经在 _construct_imdb_from_xlsa 中通过 XLSA split 成功加载，has_attributes=True，
        #      此时 _load_attributes_if_available 会直接返回，避免重复 IO；
        #    - 若尚未加载属性，则此处作为“通用 fallback”，从单独的 att_splits.mat / 属性文件中读取。
        self._load_attributes_if_available(cfg)


    def get_anno(self):
        """
        读取当前 split 对应的 JSON 标注文件。
        默认路径：
            <data_dir>/<split>.json
        当 split 包含 "train" 且 data_percentage < 1.0 时：
            <data_dir>/<split>_<percentage>.json
        返回：
            dict, 形如 {"001.jpg": class_id, "002.jpg": class_id, ...}
        """
        # 默认标注路径：eg. /path/to/CUB/train.json
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))

        # 若为 train，并且设置了只用部分数据，则改用 train_0.1.json 之类的文件
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        # 使用项目自带的 read_json 工具函数读取并返回
        return read_json(anno_path)

    def get_imagedir(self):
        """
        返回该数据集的图像存放目录。

        不同数据集的目录结构不一样：
        - CUB: <DATAPATH>/images
        - AWA2: <DATAPATH>/JPEGImages
        - SUN: <DATAPATH>/images
        因此此函数由子类重写。
        """
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """
        构建 imdb（image database）：

        imdb 形如：
            [
              {"im_path": "/abs/path/to/img1.jpg", "class": 0},
              {"im_path": "/abs/path/to/img2.jpg", "class": 5},
              ...
            ]

        优先策略：
        1. 若启用 XLSA 且配置正确，则调用 _construct_imdb_from_xlsa；
           返回 True 则说明 imdb 已构建完毕。
        2. 否则使用传统 JSON 标注方式构建 imdb。
        """
        # ---------------- 11.15 新增：优先尝试 XLSA 模式 ----------------
        if self._construct_imdb_from_xlsa(cfg):
            # 成功使用 xlsa17 构建 imdb，则不再走 JSON 路径
            return
        # -----------------------------------------------------------------

        # 否则：使用原有 JSON 标注方式
        # 由子类提供图像根目录（如 CUB: <data_dir>/images）
        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        # 读取 JSON 标注：eg. {"001.jpg": 10, "002.jpg": 3, ...}
        anno = self.get_anno()

        # Map class ids to contiguous ids
        # anno 中的 class_id 可能不是从 0 连续编号的
        # 因此先取所有 class_id，排序后形成 self._class_ids
        self._class_ids = sorted(list(set(anno.values())))

        # 建立“原始 class_id -> 连续 id(0..C-1)” 的映射
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        # 正式构建 imdb 列表
        self._imdb = []
        for img_name, cls_id in anno.items():
            # 将原始 class_id 映射到连续 id
            cont_id = self._class_id_cont_id[cls_id]
            # 拼接出图像的完整路径
            im_path = os.path.join(img_dir, img_name)
            # 存入 imdb
            self._imdb.append({"im_path": im_path, "class": cont_id})

        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    # ------------------------------------------------------------------
    # 11.15新增：xlsa17 helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _matlab_cell_to_list(cell_array: np.ndarray) -> List[str]:
        """
        将 matlab 中的 cell 数组(字符串)转为 Python 的 list[str]。

        xlsa17 的 res101.mat 里，image_files 通常是 (N,1) cell：
        - 每个 cell 可能是 bytes 或嵌套的 numpy.ndarray；
        - 这里统一展开（squeeze + item）并 decode 为 str。
        """
        # squeeze 掉多余维度，再确保至少一维
        flattened = np.atleast_1d(cell_array.squeeze())
        result: List[str] = []
        for entry in flattened:
            value = entry
            # matlab 的 cell -> numpy 通常会套一层 ndarray，这里先展开
            if isinstance(value, np.ndarray):
                value = value.squeeze()
                if value.size == 1:
                    value = value.item()
            # 有些会是 bytes，需要解码
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            result.append(str(value))
        return result

    @staticmethod
    def _strip_image_prefix(path: str) -> str:
        """
        去掉路径中多余的前缀，只保留相对 images/ 或 JPEGImages/ 的部分。

        例如：
        - "images/001.xxx/xxx.jpg"    -> "001.xxx/xxx.jpg"
        - "JPEGImages/antelope_1.jpg" -> "antelope_1.jpg"

        这样在不同数据集（CUB / AWA2 / SUN）中，只要 get_imagedir()
        返回对应 images/ 或 JPEGImages/ 根目录，就能统一拼出绝对路径。
        """
        norm = path.replace("\\", "/")
        for marker in ("images/", "JPEGImages/"):
            if marker in norm:
                return norm.split(marker, 1)[1]
        # 若没有匹配到上述前缀，则原样返回
        return norm

    @staticmethod
    def _select_split_indices(
            split_mat: Dict[str, np.ndarray],
            keys: Sequence[str],
            description: str,
    ) -> np.ndarray:
        """
        从 att_splits.mat 中按优先级选择 split 索引数组（*_loc）
        参数
        ----
        split_mat : 从 att_splits.mat 读出的 dict
        keys      : 可能的字段名列表，如 ("train_loc", "trainval_loc")
        description : 日志用途，描述当前在选哪个 split（train / val / test）

        返回
        ----
        indices: np.ndarray，已经减去 1，从 matlab 的 1-based 变成 0-based。
        """
        for key in keys:
            if key in split_mat:
                idx = np.asarray(split_mat[key]).squeeze()
                if idx.size == 0:
                    continue
                logger.info("Using xlsa17 split '%s' for %s", key, description)
                # xlsa17 中 *_loc 是 matlab 索引，这里统一转成 int64 并减 1
                return idx.astype(np.int64) - 1
        # 若所有 key 都找不到非空数组，则抛异常
        raise KeyError(
            "None of the keys {} were found with non-empty content in att_splits.mat".format(
                keys
            )
        )

    def _construct_imdb_from_xlsa(self, cfg) -> bool:
        """
        尝试根据 xlsa17 协议（res101.mat + att_splits.mat）构建 imdb。

        目标：
        - 使用 xlsa17 的官方划分（trainval_loc / test_unseen_loc / test_seen_loc 等）
          来决定“哪些样本属于 train / val / test”；
        - 用 res101.mat 中的 image_files + labels 定位“原图路径 + 类别标签”，
          但不使用其中的 ResNet 特征（features）；
        - 使得你的 ViT / VPT 可以在“官方 ZSL / GZSL 划分”下直接读原图。

        返回：
        - True  : 已成功构建 imdb（上层不需要再走 JSON）；
        - False : 未启用或配置不完整，需回退到 JSON 模式。
        """
        # 从配置中读取 DATA.XLSA 子配置
        xlsa_cfg = getattr(cfg.DATA, "XLSA", None)
        enabled = bool(getattr(xlsa_cfg, "ENABLED", False)) if xlsa_cfg is not None else False
        if not enabled:
            # 未启用 xlsa17，直接返回 False
            return False

        # res101.mat 路径：包含 image_files / labels / features（features 此处不用）
        res_path = getattr(xlsa_cfg, "RES101_PATH", "") if xlsa_cfg is not None else ""
        # att_splits.mat 路径：包含 att / trainval_loc / test_seen_loc / test_unseen_loc
        split_path = getattr(xlsa_cfg, "SPLIT_PATH", "") if xlsa_cfg is not None else ""

        # 若未单独指定 SPLIT_PATH，则默认沿用 DATA.ATTS_PATH
        if not split_path:
            split_path = getattr(cfg.DATA, "ATTS_PATH", "")

        # 路径检查，不存在就写 warning 并回退 JSON
        if not res_path or not os.path.exists(res_path):
            logger.warning(
                "DATA.XLSA.ENABLED is True but RES101_PATH '%s' is missing. Falling back to JSON.",
                res_path,
            )
            return False
        if not split_path or not os.path.exists(split_path):
            logger.warning(
                "DATA.XLSA.ENABLED is True but SPLIT_PATH '%s' is missing. Falling back to JSON.",
                split_path,
            )
            return False

        # 图像根目录仍然由数据集子类提供（images/ 或 JPEGImages/）
        img_dir = self.get_imagedir()
        if not os.path.exists(img_dir):
            raise FileNotFoundError("{} dir not found".format(img_dir))

        logger.info("Constructing imdb from xlsa17 files: %s + %s", res_path, split_path)

        res_mat = sio.loadmat(res_path)                 # -> res101.mat
        split_mat = sio.loadmat(split_path)             # -> att_splits.mat

        # 从 res101.mat 中取得 image_files 和 labels
        image_files_raw = res_mat.get("image_files")
        labels_raw = res_mat.get("labels")

        if image_files_raw is None or labels_raw is None:
            logger.warning(
                "res101.mat must contain 'image_files' and 'labels'. Falling back to JSON annotations."
            )
            return False

        # image_files: 转成 Python list[str]，并去掉 images/ 或 JPEGImages/ 前缀
        image_list = [self._strip_image_prefix(p) for p in self._matlab_cell_to_list(image_files_raw)]
        # labels: [num_images]，从 1..C 转为 0..C-1
        labels_all = np.asarray(labels_raw).squeeze().astype(np.int64) - 1
        assert labels_all.shape[0] == len(image_list), (
            "labels and image_files have mismatched lengths: {} vs {}".format(
                labels_all.shape[0],
                len(image_list),
            )
        )

        # 根据 labels 推断类数：max(label) + 1；若配置中指定了 NUMBER_CLASSES，
        # 则取二者较大值，以避免某些类在当前 split 中未出现。
        max_label = int(labels_all.max()) if labels_all.size > 0 else -1
        num_classes = max(max_label + 1, getattr(cfg.DATA, "NUMBER_CLASSES", -1))
        if num_classes <= 0:
            raise ValueError("Unable to infer NUMBER_CLASSES from xlsa metadata; please set it explicitly.")

        # 在 xlsa 模式下，直接认为类 id 为 0..num_classes-1，一一对应
        self._class_ids = list(range(num_classes))
        self._class_id_cont_id = {i: i for i in self._class_ids}

        # 优先尝试从已经加载的 split 字典中抽取语义属性，避免重复读取 .mat
        # - split_mat = sio.loadmat(att_splits.mat) 已经作为字典在当前函数中存在；
        # - 其中通常包含 att 矩阵（key 为 cfg.DATA.ATTS_KEY，默认 'att'）；
        # - 本调用会尝试直接从 split_mat 中读出 att，并映射到 self.class_attributes 上。
        # 注意：如果读取成功，self.has_attributes 会被置 True，
        #       后续 __init__ 里的 _load_attributes_if_available 将不会再从 ATTS_PATH 读一次。
        self._load_attributes_from_split_dict(cfg, split_mat, source_path=split_path)

        # 小工具：确保配置项可以既接受单个字符串，也接受列表
        def ensure_tuple(value: Iterable[str]) -> Sequence[str]:
            if isinstance(value, (list, tuple)):
                return value
            return (value,)

        # 下面这些 key 对应 xlsa17 的标准字段命名：
        # train_loc / val_loc / trainval_loc / test_unseen_loc / test_seen_loc
        train_keys = ensure_tuple(getattr(xlsa_cfg, "TRAIN_KEY", "train_loc"))
        val_keys = ensure_tuple(getattr(xlsa_cfg, "VAL_KEY", "val_loc"))
        trainval_keys = ensure_tuple(getattr(xlsa_cfg, "TRAINVAL_KEY", "trainval_loc"))

        # 测试集默认只用 unseen（ZSL），若 TEST_INCLUDE_SEEN=True，则拼接 seen 部分（GZSL）
        test_keys = list(ensure_tuple(getattr(xlsa_cfg, "TEST_KEY", "test_unseen_loc")))
        if getattr(xlsa_cfg, "TEST_INCLUDE_SEEN", False):
            seen_key = getattr(xlsa_cfg, "TEST_SEEN_KEY", "test_seen_loc")
            test_keys.append(seen_key)

        # ---------------- 根据当前 split 选择使用哪个 *_loc ----------------
        if self._split == "train":
            preferred = train_keys
            # 若 TRAIN_USE_TRAINVAL=True，则训练直接使用 trainval_loc
            if getattr(xlsa_cfg, "TRAIN_USE_TRAINVAL", False):
                preferred = trainval_keys
            try:
                split_indices = self._select_split_indices(split_mat, preferred, "train")
            except KeyError:
                # 若 preferred 不存在，则 fallback 到 trainval_loc
                split_indices = self._select_split_indices(split_mat, trainval_keys, "train (fallback trainval)")
        elif self._split == "val":
            split_indices = self._select_split_indices(split_mat, val_keys, "val")
        elif self._split == "trainval":
            split_indices = self._select_split_indices(split_mat, trainval_keys, "trainval")
        else:  # test
            # 测试集可能需要拼接多个 loc（如 unseen + seen）
            collected: List[np.ndarray] = []
            for key in test_keys:
                if key not in split_mat:
                    logger.warning("Split key '%s' not found in att_splits.mat; skipping.", key)
                    continue
                arr = np.asarray(split_mat[key]).squeeze()
                if arr.size == 0:
                    continue
                logger.info("Appending xlsa17 split '%s' for test", key)
                # 这里已经 -1，转为 0-based
                collected.append(arr.astype(np.int64) - 1)
            if not collected:
                raise KeyError("No valid *_loc arrays found for the requested test split: {}".format(test_keys))
            split_indices = np.concatenate(collected, axis=0)

        split_indices = split_indices.astype(np.int64)

        # ------------------- 根据 split_indices 构造 imdb -------------------
        self._imdb = []
        missing_files: List[str] = []
        for idx in split_indices:
            # xlsa17 给出的是“res101.features 的列号”，我们用它来索引 image_files / labels
            rel_path = image_list[idx]                  # 相对路径，例如 "001.xxx/xxx.jpg"
            abs_path = os.path.join(img_dir, rel_path)  # 真实磁盘路径
            if not os.path.exists(abs_path):
                # 记录一下找不到的文件，用于调试路径不一致问题
                missing_files.append(abs_path)
            label = int(labels_all[idx])                # 0..num_classes-1
            if label < 0 or label >= num_classes:
                raise ValueError("Label {} out of range for NUM_CLASSES {}".format(label, num_classes))
            self._imdb.append({"im_path": abs_path, "class": label})

        if missing_files:
            # 若有缺失文件，通常说明 CUB_200_2011 的路径和 xlsa17 的路径约定不一致
            logger.warning(
                "%d files listed in res101.mat were not found under %s. Example: %s",
                len(missing_files),
                img_dir,
                missing_files[0],
            )

        logger.info(
            "Constructed imdb using xlsa17 split '%s': %d images (%d unique classes).",
            self._split,
            len(self._imdb),
            len({item["class"] for item in self._imdb}),
        )
        return True

    # mixin already provides _load_attributes_* helpers

    def get_info(self):
        """
        返回两项信息：
        - num_imgs: 当前 imdb 中样本总数；
        - get_class_num(): 类别数（由 cfg.DATA.NUMBER_CLASSES 给定）。
        """
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        """
        返回类别数。
        此处使用 cfg.DATA.NUMBER_CLASSES，是为了与外部配置保持一致。
        若不依赖配置，也可以改成 return len(self._class_ids)。
        """
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """
        get a list of class weight, return a list float
        根据类别在训练集中的出现频次，计算类别权重（用于重加权 loss）。
        :param weight_type: "none" / "inv" / "inv_sqrt"
            - "none": 所有类别权重为 1
            - "inv":  权重与类别频次的 -1 次幂成正比，即 1/f
            - "inv_sqrt": 权重与类别频次的 -0.5 次幂成正比，即 1/sqrt(f)
        :return: 长度为 num_classes 的权重列表
        """
        if "train" not in self._split:
            # 只在训练集上统计类别分布，因此 val/test 不允许调用
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            # 不做重加权：所有类别权重为 1
            return [1.0] * cls_num

        # -------------------------11.15删除---------------------------------
        # Counter(self._class_ids) 的含义：
        # 这里有一点 subtle：self._class_ids 是“出现过的原始类别 id 列表”
        # 如果只是简单 Counter(self._class_ids)，则每个 id 只出现 1 次，
        # 对于真正要统计样本数的场景，通常应该遍历 imdb。
        # 但这里可能假设每个类别样本数相同，或另有约定。
        # id2counts = Counter(self._class_ids)
        # assert len(id2counts) == cls_num

        # 将不同类别的频次按 self._class_ids 的顺序排成数组
        # num_per_cls = np.array([id2counts[i] for i in self._class_ids])
        # -------------------------11.15删除结束---------------------------------
        # -------------------------11.15新增---------------------------------
        # 统计当前 imdb 中每个类的样本数（使用连续类 id）
        counts = Counter(item["class"] for item in self._imdb)
        num_per_cls = np.zeros(cls_num, dtype=np.float32)
        for orig_id in self._class_ids:
            cont_id = self._class_id_cont_id[orig_id]
            num_per_cls[cont_id] = counts.get(cont_id, 0)

        if np.any(num_per_cls == 0):
            logger.warning("Some classes have zero samples when computing class weights.")
        # -------------------------11.15新增结束---------------------------------

        # 根据设定选择指数
        if weight_type == 'inv':
            mu = -1.0       # 1 / f
        elif weight_type == 'inv_sqrt':
            mu = -0.5       # 1 / sqrt(f)
        # 按频次的 mu 次幂得到原始权重
        weight_list = num_per_cls ** mu
        # 再归一化到 L1 范数为 cls_num（方便数值尺度控制）
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        """
        按索引返回一条样本字典：

        返回内容：
            {
              "image": Tensor[C, H, W],   # 经过 transform 后的图像
              "label": int,               # 连续类 id (0..C-1)
              "attribute": Tensor[att_dim] (可选，若成功加载了 att)
            }
        """
        # 读取图像路径  使用 torchvision 默认 loader 读成 PIL.Image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        # 获取连续化后的类别 id
        label = self._imdb[index]["class"]
        # 做图像增广 / 预处理
        im = self.transform(im)

        # -------------------------11.15删除---------------------------------
        # 下面这段 index/id 逻辑目前没有实际使用（id 字段被注释掉了）
        # if self._split == "train":
        #     index = index
        # else:
            # 给 val/test 的样本加前缀，表示其属于哪个 split
        #     index = f"{self._split}{index}"
        # -------------------------11.15删除结束---------------------------------

        sample = {
            "image": im,
            "label": label,
            # "id": index    # 如需样本 id，可解开此行
        }

        # -------------------------11.15新增---------------------------------
        # 若已经加载了类级属性，则在 sample 中额外返回 "attribute"
        if self.has_attributes and self.class_attributes is not None:
            # 部分类可能在 att 矩阵中缺失，这里用 mask 做一次检查
            if self._class_has_attribute is None or self._class_has_attribute[label]:
                sample["attribute"] = self.class_attributes[label]
        # -------------------------11.15新增结束---------------------------------

        return sample

    def __len__(self):
        """
        返回数据集中样本数量（即 imdb 的长度）。
        """
        return len(self._imdb)


class CUB200Dataset(JSONDataset):
    """
    CUB_200_2011 数据集的适配类。

    只需重写 get_imagedir() 指定图像目录：
    - xlsa17 + CUB 的约定是：原始图片在 <DATAPATH>/images 下。
    """

    def __init__(self, cfg, split):
        super(CUB200Dataset, self).__init__(cfg, split)

    def get_imagedir(self):
        """
        CUB_200_2011 的图像一般放在:
        <DATAPATH>/images 下，因此这里返回该路径。
        """
        return os.path.join(self.data_dir, "images")


class AWA2Dataset(JSONDataset):

    def __init__(self, cfg, split):
        super(AWA2Dataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "JPEGImages")


class SUNAttributeDataset(JSONDataset):

    def __init__(self, cfg, split):
        super(SUNAttributeDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")



class FlowersDataset(JSONDataset):
    """
    flowers dataset.
    Flowers 数据集：
    一般图像就放在 DATAPATH 根目录下。
    """
    def __init__(self, cfg, split):
        super(FlowersDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        """
        Flowers 数据集的图像目录：
        直接使用 cfg.DATA.DATAPATH。
        """
        return self.data_dir




