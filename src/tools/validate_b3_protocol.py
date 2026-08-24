#!/usr/bin/env python3
"""Focused validation for the active B3 normal Seen/Unseen protocol."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

from src.configs.config import get_cfg
from src.data.datasets.xlsa_dataset import CUB200Dataset
from src.utils.dataset_manifest import write_xlsa_dataset_manifest
from src.tools.search_plans.b_series.run_b3_series_training import (
    _build_jobs,
    _command,
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


def _atomic_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_b3_final_dataset_and_runner_contract() -> None:
    with tempfile.TemporaryDirectory(dir=".") as temporary:
        cfg, _, _ = _source_paths()
        splits = _load_splits("final_gzsl")
        assert len(splits) == 1
        assert splits[0].name == "final_gzsl"
        assert splits[0].manifest is None and splits[0].sha256 is None

        cfg.defrost()
        cfg.DATA.XLSA.PROTOCOL_MODE = "final_gzsl"
        cfg.DATA.XLSA.B3_PSEUDO_MANIFEST = ""
        cfg.OUTPUT_DIR = str(Path(temporary).resolve() / "manifest_output")
        cfg.freeze()
        train = CUB200Dataset(cfg, "trainval")
        seen = CUB200Dataset(cfg, "test_seen")
        unseen = CUB200Dataset(cfg, "test_unseen")
        assert set(train.seen_classes) == set(seen.local_classes)
        assert set(unseen.unseen_classes) == set(unseen.local_classes)
        assert not set(train.seen_classes).intersection(unseen.unseen_classes)
        dataset_manifest_path = write_xlsa_dataset_manifest(
            cfg,
            {"trainval": train, "test_seen": seen, "test_unseen": unseen},
        )
        assert Path(dataset_manifest_path).is_file()

        jobs = _build_jobs(
            "P0",
            splits,
            ratios=(0.25, 0.50),
            out_root=Path(temporary).resolve() / "outputs",
            require_checkpoints=False,
        )
        assert len(jobs) == 3
        assert {job.seed for job in jobs} == {0, 1, 2}
        assert all(job.checkpoint is None for job in jobs)
        command = _command(jobs[0], sys.executable, "final_gzsl")
        assert "DATA.XLSA.B3_PSEUDO_MANIFEST" not in command
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
    validate_b3_final_dataset_and_runner_contract()
    print("[PASS] validate_b3_final_dataset_and_runner_contract")


if __name__ == "__main__":
    main()
