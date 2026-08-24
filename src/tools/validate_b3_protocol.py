#!/usr/bin/env python3
"""Focused validation for the isolated B3 class-disjoint protocol."""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from src.configs.config import get_cfg
from src.data.datasets.xlsa_dataset import CUB200Dataset
from src.utils.dataset_manifest import write_xlsa_dataset_manifest
from src.tools.search_plans.b_series.generate_b3_class_disjoint_manifests import (
    _atomic_json,
    _build_manifest,
    _sha256,
)
from src.tools.search_plans.b_series.run_b3_series_training import (
    _build_jobs,
    _load_splits,
)
from src.tools.search_plans.b_series.run_b3_deferred_probe_queue import (
    _compact_valid_replay,
    _deferred_monitor_overrides,
)
from src.tools.search_plans.b_series.run_b_series_replays import _parse_run


ROOT = Path(__file__).resolve().parents[2]


def _source_paths():
    cfg = get_cfg()
    cfg.merge_from_file(str(ROOT / "configs" / "baseline_rebuild" / "A-04-A2-vpt-deep-ce.yaml"))
    local_path = ROOT / "src" / "configs" / "local_path.yaml"
    if local_path.is_file():
        cfg.merge_from_file(str(local_path))
    return cfg, Path(str(cfg.DATA.XLSA.RES101_PATH)), Path(str(cfg.DATA.XLSA.SPLIT_PATH))


def _generate_suite(output_dir: Path):
    cfg, res_path, split_path = _source_paths()
    res = sio.loadmat(str(res_path))
    split = sio.loadmat(str(split_path))
    labels = np.asarray(res["labels"]).reshape(-1).astype(np.int64) - 1
    source_indices = (
        np.asarray(split["trainval_loc"]).reshape(-1).astype(np.int64) - 1
    )
    records = []
    for class_seed in (31001, 31002, 31003):
        payload = _build_manifest(
            dataset=str(cfg.DATA.NAME),
            labels=labels,
            source_indices=source_indices,
            attributes=np.asarray(split["att"]),
            class_seed=class_seed,
            train_class_ratio=0.8,
            train_image_ratio=0.8,
            res101_path=res_path,
            split_path=split_path,
        )
        path = output_dir / "b3_pseudo_split_seed{}.json".format(class_seed)
        _atomic_json(path, payload)
        records.append(
            {
                "class_seed": class_seed,
                "path": path.name,
                "sha256": _sha256(path),
                "counts": payload["counts"],
                "semantic_difficulty": payload["semantic_difficulty"],
            }
        )
    suite_path = output_dir / "b3_pseudo_split_suite.json"
    _atomic_json(
        suite_path,
        {
            "format": "b3_class_disjoint_suite_v1",
            "selection_locked_before_results": True,
            "records": records,
        },
    )
    return cfg, suite_path


def validate_b3_dataset_and_runner_contract() -> None:
    with tempfile.TemporaryDirectory(dir=".") as temporary:
        cfg, suite_path = _generate_suite(Path(temporary).resolve())
        splits = _load_splits("b3_pseudo_gzsl", suite_path)
        assert len(splits) == 3 and len({split.sha256 for split in splits}) == 3

        cfg.defrost()
        cfg.DATA.XLSA.PROTOCOL_MODE = "b3_pseudo_gzsl"
        cfg.DATA.XLSA.B3_PSEUDO_MANIFEST = str(splits[0].manifest)
        cfg.OUTPUT_DIR = str(Path(temporary).resolve() / "manifest_output")
        cfg.freeze()
        train = CUB200Dataset(cfg, "trainval")
        seen = CUB200Dataset(cfg, "test_seen")
        unseen = CUB200Dataset(cfg, "test_unseen")
        source_sets = [
            {int(row["source_index"]) for row in dataset._imdb}
            for dataset in (train, seen, unseen)
        ]
        assert not source_sets[0].intersection(source_sets[1])
        assert not source_sets[0].intersection(source_sets[2])
        assert not source_sets[1].intersection(source_sets[2])
        assert set(train.seen_classes) == set(seen.local_classes)
        assert set(unseen.unseen_classes) == set(unseen.local_classes)
        assert train.b3_pseudo_manifest_sha256 == splits[0].sha256
        dataset_manifest_path = write_xlsa_dataset_manifest(
            cfg,
            {"trainval": train, "test_seen": seen, "test_unseen": unseen},
        )
        assert Path(dataset_manifest_path).is_file()
        portable_manifest = Path(cfg.OUTPUT_DIR) / "b3_pseudo_manifest.json"
        assert portable_manifest.is_file()
        assert _sha256(portable_manifest) == splits[0].sha256

        jobs = _build_jobs(
            "P0",
            splits,
            ratios=(0.25, 0.50),
            out_root=Path(temporary).resolve() / "outputs",
            require_checkpoints=False,
        )
        assert len(jobs) == 9
        assert {job.seed for job in jobs} == {0, 1, 2}
        assert all(job.checkpoint is None for job in jobs)
        pilot_jobs = _build_jobs(
            "R2",
            splits,
            ratios=(0.25,),
            out_root=Path(temporary).resolve() / "outputs",
            require_checkpoints=False,
            pilot_weights=((0.01, 0.002), (0.05, 0.01)),
        )
        assert len(pilot_jobs) == 2
        assert {job.seed for job in pilot_jobs} == {0}
        assert all(job.method.startswith("B3-R2P-") for job in pilot_jobs)
        assert _parse_run("B3-R1I-R025:2") == ("B3-R1I-R025", 2)

        command_cfg = get_cfg()
        command_cfg.merge_from_list(_deferred_monitor_overrides())
        assert command_cfg.MONITOR.PROBE.FINAL_EXECUTION_MODE == "deferred"
        assert command_cfg.MONITOR.PROBE.CACHE_TRANSFORMED_IMAGES is True
        assert command_cfg.MONITOR.PROBE.CACHE_VIT_CLS_PREPASS is True

        replay_run = Path(temporary).resolve() / "valid_replay"
        diagnostics = replay_run / "diagnostics"
        diagnostics.mkdir(parents=True)
        _atomic_json(
            replay_run / "probe_robustness_replay_summary.json",
            {"valid": True},
        )
        metrics_path = diagnostics / "probe_metrics.csv"
        metrics_path.write_text("domain,value\nmechanism,1.0\n", encoding="utf-8")
        compaction = _compact_valid_replay(replay_run)
        assert compaction["status"] == "compacted"
        assert not metrics_path.exists()
        with gzip.open(str(metrics_path) + ".gz", "rt", encoding="utf-8") as handle:
            assert handle.read() == "domain,value\nmechanism,1.0\n"


def main() -> None:
    validate_b3_dataset_and_runner_contract()
    print("[PASS] validate_b3_dataset_and_runner_contract")


if __name__ == "__main__":
    main()
