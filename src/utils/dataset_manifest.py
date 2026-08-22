#!/usr/bin/env python3

import hashlib
import json
import os
from typing import Dict, Mapping, Optional

import torch

from .distributed import get_rank


MANIFEST_SCHEMA_VERSION = "xlsa_dataset_manifest_v2"


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
    return {"sha256": _sha256_file(absolute_path)}


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
    if class_attributes is None:
        raise ValueError("Dataset manifest requires class_attributes.")
    record = {
        "dataset_name": str(dataset.name),
        "protocol_mode": str(dataset.protocol_mode),
        "split": str(dataset.split_name),
        "image_count": int(len(dataset)),
        "image_records_sha256": _image_records_sha256(dataset),
        "eval_local_class_ids": _int_list(dataset.eval_local_classes),
        "seen_class_ids": _int_list(dataset.seen_classes),
        "unseen_class_ids": _int_list(dataset.unseen_classes),
        "class_attributes": {
            "shape": [int(x) for x in class_attributes.shape],
            "sha256": _tensor_sha256(class_attributes),
        },
    }
    b3_path = getattr(dataset, "b3_pseudo_manifest_path", None)
    if b3_path:
        record["b3_pseudo_manifest"] = {
            "sha256": str(dataset.b3_pseudo_manifest_sha256),
        }
    return record


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


def _atomic_json_dump(path: str, payload: Dict[str, object]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def _atomic_copy_file(source: str, target: str) -> None:
    with open(source, "rb") as handle:
        data = handle.read()
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path = "{}.tmp.{}".format(target, os.getpid())
    with open(temporary_path, "wb") as handle:
        handle.write(data)
    os.replace(temporary_path, target)


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
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_files": {
            "res101_mat": _file_record(str(cfg.DATA.XLSA.RES101_PATH)),
            "att_splits_mat": _file_record(str(cfg.DATA.XLSA.SPLIT_PATH)),
        },
        "datasets": records,
    }
    b3_manifest_path = str(cfg.DATA.XLSA.B3_PSEUDO_MANIFEST).strip()
    if b3_manifest_path:
        manifest["source_files"]["b3_pseudo_manifest"] = _file_record(
            b3_manifest_path
        )
        portable_path = os.path.join(
            str(cfg.OUTPUT_DIR), "b3_pseudo_manifest.json"
        )
        _atomic_copy_file(b3_manifest_path, portable_path)
        if _sha256_file(portable_path) != _sha256_file(b3_manifest_path):
            raise RuntimeError("portable B3 manifest copy failed hash verification")
        manifest["portable_b3_pseudo_manifest"] = {
            "path": "b3_pseudo_manifest.json",
            "sha256": _sha256_file(portable_path),
        }
    manifest_path = os.path.join(str(cfg.OUTPUT_DIR), "dataset_manifest.json")
    _atomic_json_dump(manifest_path, manifest)
    return manifest_path
