#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _seeds(raw: str) -> list[int]:
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if len(values) != 3 or len(set(values)) != 3 or min(values) < 0:
        raise ValueError("--class-seeds requires exactly three unique non-negative seeds")
    return values


def _attribute_difficulty(attributes, train_classes, pseudo_classes):
    values = np.asarray(attributes, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("att must be a 2D matrix")
    total_classes = max(max(train_classes), max(pseudo_classes)) + 1
    if values.shape[0] != total_classes and values.shape[1] >= total_classes:
        values = values.T
    if values.shape[0] < total_classes:
        raise ValueError("att does not cover the selected class ids")
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)
    similarity = values[pseudo_classes] @ values[train_classes].T
    nearest = similarity.max(axis=1)
    return {
        "pseudo_to_train_nearest_cosine_mean": float(nearest.mean()),
        "pseudo_to_train_nearest_cosine_min": float(nearest.min()),
        "pseudo_to_train_nearest_cosine_max": float(nearest.max()),
    }


def _build_manifest(
    *,
    dataset: str,
    labels: np.ndarray,
    source_indices: np.ndarray,
    attributes: np.ndarray,
    class_seed: int,
    train_class_ratio: float,
    train_image_ratio: float,
    res101_path: Path,
    split_path: Path,
):
    source_classes = sorted(int(value) for value in np.unique(labels[source_indices]))
    class_rng = np.random.RandomState(int(class_seed))
    shuffled = np.asarray(source_classes, dtype=np.int64)
    class_rng.shuffle(shuffled)
    train_count = int(round(len(source_classes) * float(train_class_ratio)))
    if train_count <= 0 or train_count >= len(source_classes):
        raise ValueError("train class ratio creates an empty partition")
    train_classes = sorted(int(value) for value in shuffled[:train_count].tolist())
    pseudo_classes = sorted(int(value) for value in shuffled[train_count:].tolist())
    train_set = set(train_classes)
    pseudo_set = set(pseudo_classes)

    train_indices = []
    seen_eval_indices = []
    pseudo_indices = []
    per_class_counts = {}
    for class_id in source_classes:
        class_indices = source_indices[labels[source_indices] == int(class_id)].copy()
        if int(class_id) in pseudo_set:
            pseudo_indices.extend(int(value) for value in class_indices.tolist())
            per_class_counts[str(class_id)] = {
                "role": "pseudo_unseen",
                "all": int(class_indices.size),
            }
            continue
        sample_seed = int.from_bytes(
            hashlib.sha256(
                "{}|{}".format(int(class_seed), int(class_id)).encode("utf-8")
            ).digest()[:4],
            byteorder="little",
        )
        sample_rng = np.random.RandomState(sample_seed)
        sample_rng.shuffle(class_indices)
        train_sample_count = int(round(class_indices.size * float(train_image_ratio)))
        train_sample_count = min(max(1, train_sample_count), int(class_indices.size) - 1)
        current_train = class_indices[:train_sample_count]
        current_eval = class_indices[train_sample_count:]
        train_indices.extend(int(value) for value in current_train.tolist())
        seen_eval_indices.extend(int(value) for value in current_eval.tolist())
        per_class_counts[str(class_id)] = {
            "role": "train_seen",
            "all": int(class_indices.size),
            "train": int(current_train.size),
            "seen_eval": int(current_eval.size),
        }

    partition = train_indices + seen_eval_indices + pseudo_indices
    if len(set(partition)) != len(partition) or set(partition) != set(
        int(value) for value in source_indices.tolist()
    ):
        raise RuntimeError("generated B3 sample partition is not disjoint and complete")
    if set(train_classes).intersection(pseudo_classes) or set(
        train_classes + pseudo_classes
    ) != set(source_classes):
        raise RuntimeError("generated B3 class partition is invalid")

    return {
        "format": "b3_class_disjoint_manifest_v1",
        "dataset": str(dataset),
        "class_seed": int(class_seed),
        "sample_split_rule": "per_class_deterministic_shuffle",
        "train_class_ratio": float(train_class_ratio),
        "train_image_ratio": float(train_image_ratio),
        "source_protocol_key": "trainval_loc",
        "source_class_ids": source_classes,
        "train_class_ids": train_classes,
        "pseudo_unseen_class_ids": pseudo_classes,
        "train_source_indices": sorted(train_indices),
        "seen_eval_source_indices": sorted(seen_eval_indices),
        "pseudo_unseen_source_indices": sorted(pseudo_indices),
        "source_files": {
            "res101_path": str(res101_path.resolve()),
            "res101_sha256": _sha256(res101_path),
            "split_path": str(split_path.resolve()),
            "split_sha256": _sha256(split_path),
        },
        "counts": {
            "source_classes": len(source_classes),
            "train_classes": len(train_classes),
            "pseudo_unseen_classes": len(pseudo_classes),
            "train_samples": len(train_indices),
            "seen_eval_samples": len(seen_eval_indices),
            "pseudo_unseen_samples": len(pseudo_indices),
        },
        "semantic_difficulty": _attribute_difficulty(
            attributes, train_classes, pseudo_classes
        ),
        "per_class_counts": per_class_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate three fixed class-disjoint B3 pseudo-GZSL manifests."
    )
    parser.add_argument("--res101-path", required=True, type=Path)
    parser.add_argument("--split-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="CUB")
    parser.add_argument("--class-seeds", default="31001,31002,31003")
    parser.add_argument("--train-class-ratio", type=float, default=0.8)
    parser.add_argument("--train-image-ratio", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.0 < args.train_class_ratio < 1.0:
        raise SystemExit("--train-class-ratio must lie inside (0, 1)")
    if not 0.0 < args.train_image_ratio < 1.0:
        raise SystemExit("--train-image-ratio must lie inside (0, 1)")
    res101_path = args.res101_path.resolve()
    split_path = args.split_path.resolve()
    if not res101_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("XLSA source files are missing")
    res = sio.loadmat(str(res101_path))
    split = sio.loadmat(str(split_path))
    labels = np.asarray(res["labels"]).reshape(-1).astype(np.int64) - 1
    source_indices = np.asarray(split["trainval_loc"]).reshape(-1).astype(np.int64) - 1
    attributes = np.asarray(split["att"])
    records = []
    for class_seed in _seeds(args.class_seeds):
        payload = _build_manifest(
            dataset=args.dataset,
            labels=labels,
            source_indices=source_indices,
            attributes=attributes,
            class_seed=class_seed,
            train_class_ratio=args.train_class_ratio,
            train_image_ratio=args.train_image_ratio,
            res101_path=res101_path,
            split_path=split_path,
        )
        path = args.output_dir.resolve() / "b3_pseudo_split_seed{}.json".format(
            class_seed
        )
        _atomic_json(path, payload)
        records.append(
            {
                "class_seed": class_seed,
                # Keep the suite portable between the local workspace and the
                # experiment server; the loader resolves this beside the suite.
                "path": path.name,
                "sha256": _sha256(path),
                "counts": payload["counts"],
                "semantic_difficulty": payload["semantic_difficulty"],
            }
        )
    _atomic_json(
        args.output_dir.resolve() / "b3_pseudo_split_suite.json",
        {
            "format": "b3_class_disjoint_suite_v1",
            "selection_locked_before_results": True,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
