#!/usr/bin/env python3

"""Convert the .mat protocol files of XLSA17 / xlsa17 (res101.mat att_splits.mat)
into Dataset objects that can be directly used by the current training pipeline."""

import os
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data
import torchvision as tv

from ..transforms import get_transforms
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class XLSAAttributeMixin:
    """Read class-level attributes from att_splits.mat,
    and align them to the global class ID space of the current dataset"""

    class_attributes: Optional[torch.Tensor] = None

    def _load_attributes_from_split_dict(self, split_mat, source_path: str) -> None:
        """load attributes from att_splits.mat"""
        att_key = "att"
        if att_key not in split_mat:
            raise KeyError("XLSA split file '{}' does not contain attribute key '{}'".format(source_path, att_key))
        self._populate_class_attributes(np.asarray(split_mat[att_key]), source_name="{}::{}".format(source_path, att_key))

    def _populate_class_attributes(self, atts_np: np.ndarray, source_name: str) -> None:
        """load attributes from att_splits.mat"""
        num_classes = int(self.num_classes)
        if atts_np.ndim != 2:
            raise ValueError("Expected 2D class-attribute matrix from {}, got shape {}".format(source_name,tuple(atts_np.shape),))
        if atts_np.shape[0] != num_classes and atts_np.shape[1] == num_classes:
            atts_np = atts_np.T

        total_rows = int(atts_np.shape[0])
        att_dim = int(atts_np.shape[1])

        attr_tensor = torch.from_numpy(atts_np).float()

        class_attributes = torch.zeros((num_classes, att_dim), dtype=torch.float32)

        missing_ids: List[int] = []

        for class_id in range(num_classes):
            row_idx = int(class_id)
            if row_idx < 0 or row_idx >= total_rows:
                missing_ids.append(int(class_id))
                continue
            class_attributes[class_id] = attr_tensor[row_idx]

        if missing_ids:
            raise ValueError(
                "Attributes missing for {}/{} classes in {} (e.g. {})".format(
                    len(missing_ids),
                    num_classes,
                    source_name,
                    missing_ids[:5],)
            )

        self.class_attributes = class_attributes
        logger.info(
            "Loaded attributes %s from %s for %d/%d classes",
            tuple(class_attributes.shape),
            source_name,
            num_classes,
            num_classes,
        )


class XLSADataset(XLSAAttributeMixin, torch.utils.data.Dataset):
    """Dataset wrapper that only supports XLSA protocol splits."""

    VALID_SPLITS = {"train", "val_unseen", "test_seen", "test_unseen", "trainval"}

    def __init__(self, cfg, split: str):
        xlsa_cfg = cfg.DATA.XLSA
        if not bool(xlsa_cfg.ENABLED):
            raise ValueError("XLSADataset requires DATA.XLSA.ENABLED=True.")

        split = str(split)
        if split not in self.VALID_SPLITS:
            raise ValueError("Split '{}' not supported under XLSA protocol".format(split))

        self.cfg = cfg
        self._split = split
        self.split_name = split
        self.name = cfg.DATA.NAME
        self.data_dir = cfg.DATA.DATAPATH
        self.transform = get_transforms(split, cfg.DATA.CROPSIZE)
        self.num_classes = int(cfg.DATA.NUMBER_CLASSES)

        self.class_attributes = None

        self.seen_classes = None
        self.unseen_classes = None
        self.seen_classnames = None
        self.unseen_classnames = None

        self.protocol_mode = None
        self.split_source_keys = []

        self.split_classes = None
        self.local_classes = None
        self.global_to_local = None
        self.labels_local = None

        self.eval_local_classes = None
        self.eval_global_to_local = None

        self._construct_from_xlsa(cfg)

    def get_imagedir(self):
        """Tell which directory the current dataset images are actually stored in"""
        raise NotImplementedError()

    @staticmethod
    def _matlab_cell_to_list(cell_array: np.ndarray) -> List[str]:
        """Convert MATLAB cell array to Python list[str]"""
        flattened = np.atleast_1d(cell_array.squeeze())
        result: List[str] = []
        for entry in flattened:
            value = entry
            if isinstance(value, np.ndarray):
                value = value.squeeze()
                if value.size == 1:
                    value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            result.append(str(value))
        return result

    @staticmethod
    def _strip_image_prefix(path: str) -> str:
        """Remove the possible path prefix that comes with image_files in res101.mat"""
        norm = str(path).replace("\\", "/")
        for marker in ("images/", "JPEGImages/"):
            if marker in norm:
                return norm.split(marker, 1)[1]
        return norm

    @staticmethod
    def _select_split_indices(split_mat: Dict[str, np.ndarray], keys: Sequence[str], description: str) -> np.ndarray:
        """Select a non-empty split key from att_splits.mat and return its sample indices"""
        for key in keys:
            if key in split_mat:
                idx = np.asarray(split_mat[key]).squeeze()
                if idx.size == 0:
                    continue
                return idx.astype(np.int64) - 1
        raise KeyError("None of the keys {} were found with non-empty content in att_splits.mat".format(tuple(keys)))

    @staticmethod
    def _build_global_to_local(num_classes: int, class_ids: Sequence[int]) -> torch.Tensor:
        """Construct a global -> local mapping table"""
        mapping = torch.full((int(num_classes),), -1, dtype=torch.long)
        if len(class_ids) == 0:
            return mapping
        ids = torch.as_tensor(list(class_ids), dtype=torch.long)
        mapping[ids] = torch.arange(ids.numel(), dtype=torch.long)
        return mapping

    def _finalize_protocol_meta(self, *, num_classes: int, split_labels: np.ndarray, split_source_keys: Sequence[str], local_classes: Sequence[int], eval_local_classes: Sequence[int],) -> None:
        """
        Organize all protocol metadata of the current split at once and attach it to the dataset instance
        参数
        num_classes:
            全局类别数
        split_labels:
            当前 split 样本池对应的 global labels
        split_source_keys:
            这个 split 实际消费了 att_splits.mat 里的哪些 key
        local_classes:
            当前 split 自身的 local space 使用哪些 global class ids
        eval_local_classes:
            正式评测时使用哪套 local eval class space
            （例如 strict GZSL 下 seen+unseen 的统一评测空间）

        最终保存的字段
        - split_source_keys
        - split_classes
        - local_classes
        - global_to_local
        - labels_local
        - eval_local_classes
        - eval_global_to_local
        """
        split_labels = np.asarray(split_labels, dtype=np.int64).reshape(-1)
        split_classes = sorted(int(x) for x in np.unique(split_labels).tolist())
        self.split_source_keys = [str(x) for x in list(split_source_keys)]
        self.split_classes = split_classes
        self.local_classes = [int(x) for x in list(local_classes)]
        self.global_to_local = self._build_global_to_local(num_classes, self.local_classes)
        self.labels_local = self.global_to_local.index_select(0, torch.as_tensor(split_labels.tolist(), dtype=torch.long))
        self.eval_local_classes = [int(x) for x in list(eval_local_classes)]
        self.eval_global_to_local = self._build_global_to_local(num_classes, self.eval_local_classes)

    def _construct_from_xlsa(self, cfg) -> None:

        # 1. Load res101.mat / att_splits.mat
        xlsa_cfg = cfg.DATA.XLSA
        res_path = str(xlsa_cfg.RES101_PATH or "")
        split_path = str(xlsa_cfg.SPLIT_PATH or "")

        if not res_path or not os.path.exists(res_path):
            raise FileNotFoundError("DATA.XLSA.RES101_PATH not found: {}".format(res_path))
        if not split_path or not os.path.exists(split_path):
            raise FileNotFoundError("DATA.XLSA.SPLIT_PATH not found: {}".format(split_path))

        img_dir = self.get_imagedir()
        if not os.path.exists(img_dir):
            raise FileNotFoundError("image dir not found: {}".format(img_dir))

        res_mat = sio.loadmat(res_path)
        split_mat = sio.loadmat(split_path)

        image_files_raw = res_mat.get("image_files")
        labels_raw = res_mat.get("labels")
        if image_files_raw is None or labels_raw is None:
            raise KeyError("res101.mat must contain 'image_files' and 'labels'")

        # 2. Build the full image list and the full global label list for the entire dataset.
        #    image_list: relative image paths for all samples in the dataset
        #    labels_all: global class ids for all samples, converted from MATLAB 1-based ids to Python 0-based ids
        image_list = [self._strip_image_prefix(p) for p in self._matlab_cell_to_list(image_files_raw)]
        labels_all = np.asarray(labels_raw).squeeze().astype(np.int64) - 1

        if labels_all.shape[0] != len(image_list):
            raise ValueError("labels and image_files have mismatched lengths: {} vs {}".format(labels_all.shape[0], len(image_list)))

        max_label = int(labels_all.max()) if labels_all.size > 0 else -1
        num_classes = max(max_label + 1, int(cfg.DATA.NUMBER_CLASSES))
        if num_classes <= 0:
            raise ValueError("Unable to infer NUMBER_CLASSES from XLSA metadata")

        self.num_classes = int(num_classes)

        # 3. Load class-level attributes from att_splits.mat
        self._load_attributes_from_split_dict(split_mat, source_path=split_path)

        # 4. Define
        train_keys = ("train_loc",)
        val_keys = ("val_loc",)
        trainval_keys = ("trainval_loc",)
        test_unseen_keys = ("test_unseen_loc",)
        test_seen_keys = ("test_seen_loc",)
        protocol_mode = str(xlsa_cfg.PROTOCOL_MODE).lower()
        if protocol_mode not in {"dev", "final_zsl", "final_gzsl"}:
            raise ValueError("Unsupported DATA.XLSA.PROTOCOL_MODE='{}'".format(xlsa_cfg.PROTOCOL_MODE))
        self.protocol_mode = protocol_mode

        if protocol_mode == "dev":
            seen_keys = train_keys
            unseen_keys = val_keys
            eval_mode = "zsl"
            allowed_splits = {"train", "val_unseen"}
        elif protocol_mode == "final_zsl":
            seen_keys = trainval_keys
            unseen_keys = test_unseen_keys
            eval_mode = "zsl"
            allowed_splits = {"trainval", "test_unseen"}
        else:
            seen_keys = trainval_keys
            unseen_keys = test_unseen_keys
            eval_mode = "gzsl"
            allowed_splits = {"trainval", "test_seen", "test_unseen"}

        if self._split not in allowed_splits:
            raise ValueError(
                "split='{}' is incompatible with DATA.XLSA.PROTOCOL_MODE='{}'.".format(
                    self._split, protocol_mode
                )
            )

        seen_indices = self._select_split_indices(split_mat, seen_keys, "seen-classes")
        unseen_indices = self._select_split_indices(split_mat, unseen_keys, "unseen-classes")

        self.seen_classes = sorted(int(x) for x in np.unique(labels_all[seen_indices]).tolist())
        self.unseen_classes = sorted(int(x) for x in np.unique(labels_all[unseen_indices]).tolist())

        # 5. Load all class names from att_splits.mat and attach human-readable names
        if "allclasses_names" not in split_mat:
            raise KeyError("att_splits.mat must contain 'allclasses_names'")
        all_class_names = self._matlab_cell_to_list(np.asarray(split_mat["allclasses_names"]))
        if len(all_class_names) < num_classes:
            raise ValueError("allclasses_names length {} is smaller than num_classes {}".format(
                    len(all_class_names), num_classes))
        self.seen_classnames = [all_class_names[i] for i in self.seen_classes]
        self.unseen_classnames = [all_class_names[i] for i in self.unseen_classes]

        # 6. Determine the current split's sample pool under the selected protocol mode.
        if protocol_mode == "dev":
            if self._split == "train":
                split_indices = self._select_split_indices(split_mat, train_keys, "train")
                split_source_keys = train_keys
            else:
                split_indices = self._select_split_indices(split_mat, val_keys, "val_unseen")
                split_source_keys = val_keys
        elif protocol_mode == "final_zsl":
            if self._split == "trainval":
                split_indices = self._select_split_indices(split_mat, trainval_keys, "trainval")
                split_source_keys = trainval_keys
            else:
                split_indices = self._select_split_indices(split_mat, test_unseen_keys, "test_unseen")
                split_source_keys = test_unseen_keys
        else:
            if self._split == "trainval":
                split_indices = self._select_split_indices(split_mat, trainval_keys, "trainval")
                split_source_keys = trainval_keys
            elif self._split == "test_seen":
                split_indices = self._select_split_indices(split_mat, test_seen_keys, "test_seen")
                split_source_keys = test_seen_keys
            else:
                split_indices = self._select_split_indices(split_mat, test_unseen_keys, "test_unseen")
                split_source_keys = test_unseen_keys

        split_indices = split_indices.astype(np.int64)
        split_labels = labels_all[split_indices]

        # 8. Build the split-native local class space and the formal evaluation class space.
        # local_classes:the native local-label space of the current split itself
        # eval_local_classes:the formal evaluation space used by evaluator/monitor
        if self._split in {"train", "trainval"}:
            local_classes = list(self.seen_classes)
            eval_local_classes = list(local_classes)
        elif self._split == "val_unseen":
            local_classes = sorted(int(x) for x in np.unique(split_labels).tolist())
            eval_local_classes = list(self.unseen_classes)
        elif self._split == "test_seen":
            local_classes = sorted(int(x) for x in np.unique(split_labels).tolist())
            eval_local_classes = list(self.seen_classes) + list(self.unseen_classes)
        else:
            local_classes = sorted(int(x) for x in np.unique(split_labels).tolist())
            eval_local_classes = list(self.seen_classes) + list(self.unseen_classes) if eval_mode == "gzsl" else list(self.unseen_classes)

        if protocol_mode == "dev" and self._split == "val_unseen" and local_classes != list(self.unseen_classes):
            raise ValueError("dev protocol requires val_unseen local_classes to match unseen_classes")
        if protocol_mode == "final_zsl" and self._split == "test_unseen" and local_classes != list(self.unseen_classes):
            raise ValueError("final_zsl protocol requires test_unseen local_classes to match unseen_classes")
        if protocol_mode == "final_gzsl":
            if self._split == "test_seen" and local_classes != list(self.seen_classes):
                raise ValueError("final_gzsl protocol requires test_seen local_classes to match seen_classes")
            if self._split == "test_unseen" and local_classes != list(self.unseen_classes):
                raise ValueError("final_gzsl protocol requires test_unseen local_classes to match unseen_classes")

        self._finalize_protocol_meta(
            num_classes=num_classes,
            split_labels=split_labels,
            split_source_keys=split_source_keys,
            local_classes=local_classes,
            eval_local_classes=eval_local_classes,
        )

        self._imdb = []
        missing_files: List[str] = []
        for idx in split_indices:
            rel_path = image_list[int(idx)]
            abs_path = os.path.join(img_dir, rel_path)
            if not os.path.exists(abs_path):
                missing_files.append(abs_path)
            label = int(labels_all[int(idx)])
            if label < 0 or label >= num_classes:
                raise ValueError("Label {} out of range for NUMBER_CLASSES {}".format(label, num_classes))
            self._imdb.append({"im_path": abs_path, "class": label})

        if missing_files:
            raise FileNotFoundError(
                "{} files listed in res101.mat were not found under {}. Example: {}".format(
                    len(missing_files),
                    img_dir,
                    missing_files[0],
                )
            )

        logger.info(
            "XLSA split=%s images=%d classes=%d sources=%s",
            self._split,
            len(self._imdb),
            len(self.split_classes),
            ",".join(self.split_source_keys),
        )

    def get_info(self):
        """Return the number of samples in the current dataset and the number of classes in the configuration"""
        return len(self._imdb), self.get_class_num()

    def get_class_num(self):
        """Return the total number of categories in the configuration"""
        return int(self.num_classes)

    def get_class_weights(self, weight_type):
        if "train" not in self._split:
            raise ValueError("only training splits support class weight estimation, got '{}'".format(self._split))
        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        counts = Counter(item["class"] for item in self._imdb)
        num_per_cls = np.zeros(cls_num, dtype=np.float32)
        for class_id in range(cls_num):
            num_per_cls[class_id] = counts.get(class_id, 0)

        valid_mask = num_per_cls > 0
        num_per_cls_safe = num_per_cls.copy()
        num_per_cls_safe[~valid_mask] = 1.0
        if weight_type == "inv":
            mu = -1.0
        elif weight_type == "inv_sqrt":
            mu = -0.5
        else:
            raise ValueError("Unsupported class weight type '{}'".format(weight_type))

        weight_list = num_per_cls_safe ** mu
        weight_list[~valid_mask] = 0.0
        denom = float(np.linalg.norm(weight_list, 1))
        if denom <= 0:
            return [1.0] * cls_num
        weight_list = np.divide(weight_list, denom) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        """
        Take a single sample
        """
        record = self._imdb[index]
        image = tv.datasets.folder.default_loader(record["im_path"])
        image = self.transform(image)
        label = int(record["class"])
        return {
            "image": image,
            "label": label,
            "attribute": self.class_attributes[label],
        }

    def __len__(self):
        return len(self._imdb)


class CUB200Dataset(XLSADataset):
    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class AWA2Dataset(XLSADataset):
    def get_imagedir(self):
        return os.path.join(self.data_dir, "JPEGImages")


class SUNAttributeDataset(XLSADataset):
    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")
