#!/usr/bin/env python3

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional

import torch

from .distributed import get_rank


MANIFEST_SCHEMA_VERSION = "xlsa_dataset_manifest_v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str) -> Dict[str, object]:
    absolute_path = os.path.abspath(path)
    if not os.path.isfile(absolute_path):
        raise FileNotFoundError("Manifest source file does not exist: {}".format(absolute_path))
    return {
        "path": absolute_path,
        "size_bytes": int(os.path.getsize(absolute_path)),
        "sha256": _sha256_file(absolute_path),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": [int(x) for x in tensor.shape]},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(header + b"\n" + tensor.numpy().tobytes())


def _normalized_relative_path(path: str, image_root: str) -> str:
    relative = os.path.relpath(path, image_root)
    return relative.replace("\\", "/")


def _image_records_sha256(dataset) -> str:
    image_root = dataset.get_imagedir()
    digest = hashlib.sha256()
    for record in dataset._imdb:
        relative_path = _normalized_relative_path(record["im_path"], image_root)
        line = "{}\t{}\n".format(int(record["class"]), relative_path)
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _int_list(values) -> list:
    return [int(value) for value in list(values)]


def _dataset_record(dataset) -> Dict[str, object]:
    class_attributes = getattr(dataset, "class_attributes", None)
    all_classnames = getattr(dataset, "all_classnames", None)
    if class_attributes is None:
        raise ValueError("Dataset manifest requires class_attributes.")
    if all_classnames is None or len(all_classnames) != int(dataset.num_classes):
        raise ValueError("Dataset manifest requires complete all_classnames in global-ID order.")
    return {
        "dataset_name": str(dataset.name),
        "protocol_mode": str(dataset.protocol_mode),
        "split": str(dataset.split_name),
        "split_source_keys": [str(key) for key in dataset.split_source_keys],
        "image_root": os.path.abspath(dataset.get_imagedir()),
        "image_count": int(len(dataset)),
        "image_records_sha256": _image_records_sha256(dataset),
        "split_class_ids": _int_list(dataset.split_classes),
        "local_class_ids": _int_list(dataset.local_classes),
        "eval_local_class_ids": _int_list(dataset.eval_local_classes),
        "seen_class_ids": _int_list(dataset.seen_classes),
        "unseen_class_ids": _int_list(dataset.unseen_classes),
        "class_attributes": {
            "dtype": str(class_attributes.dtype),
            "shape": [int(x) for x in class_attributes.shape],
            "sha256": _tensor_sha256(class_attributes),
        },
    }


def _assert_shared_global_space(dataset_records: Mapping[str, Dict[str, object]]) -> None:
    reference = None
    for record in dataset_records.values():
        attributes = record["class_attributes"]
        key = (
            record["dataset_name"],
            record["protocol_mode"],
            tuple(attributes["shape"]),
            attributes["sha256"],
            tuple(record["seen_class_ids"]),
            tuple(record["unseen_class_ids"]),
        )
        if reference is None:
            reference = key
        elif key != reference:
            raise ValueError("Datasets in one run do not share the same global class/attribute protocol.")


def _global_class_mapping(dataset) -> list:
    return [
        {
            "global_id": int(class_id),
            "class_name": str(dataset.all_classnames[class_id]),
            "attribute_row": int(class_id),
        }
        for class_id in range(int(dataset.num_classes))
    ]


def _atomic_json_dump(path: str, payload: Dict[str, object]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def write_xlsa_dataset_manifest(cfg, datasets: Mapping[str, Optional[object]]) -> Optional[str]:
    if get_rank() != 0:
        return None

    active_datasets = {
        str(role): dataset
        for role, dataset in datasets.items()
        if dataset is not None
    }
    if not active_datasets:
        raise ValueError("Dataset manifest requires at least one active dataset.")

    records = {
        role: _dataset_record(dataset)
        for role, dataset in active_datasets.items()
    }
    _assert_shared_global_space(records)
    reference_dataset = next(iter(active_datasets.values()))
    config_text = cfg.dump()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_config_sha256": _sha256_bytes(config_text.encode("utf-8")),
        "source_files": {
            "res101_mat": _file_record(str(cfg.DATA.XLSA.RES101_PATH)),
            "att_splits_mat": _file_record(str(cfg.DATA.XLSA.SPLIT_PATH)),
        },
        "global_class_mapping": _global_class_mapping(reference_dataset),
        "datasets": records,
    }
    manifest_path = os.path.join(str(cfg.OUTPUT_DIR), "dataset_manifest.json")
    _atomic_json_dump(manifest_path, manifest)
    return manifest_path
