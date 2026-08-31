#!/usr/bin/env python3
"""Synthetic behavior checks for shared search-plan helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .search_plans.common import (
    atomic_write_json,
    build_single_gpu_train_command,
    directory_has_contents,
    parse_gpu_worker_slots,
    read_json,
    write_json,
)
from .search_plans.a_series.run_baseline_series import (
    Job as BaselineJob,
    _command as baseline_command,
)
from .search_plans.b_series.run_b3_series_training import (
    _gpu_groups as b3_gpu_groups,
)
from .search_plans.b_series.run_prompt_distribution_series import (
    Job as PromptDistributionJob,
    _command as prompt_distribution_command,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact = root / "nested" / "artifact.json"
        payload = {"unicode": "类别", "value": 3, "items": [1, 2]}
        write_json(artifact, payload)
        assert read_json(artifact) == payload
        assert artifact.read_text(encoding="utf-8").endswith("\n")

        replacement = {"status": "complete", "finite": 1.25}
        atomic_write_json(artifact, replacement, allow_nan=False, sort_keys=True)
        assert json.loads(artifact.read_text(encoding="utf-8")) == replacement
        assert not list(artifact.parent.glob(".*.tmp"))

        empty = root / "empty"
        empty.mkdir()
        assert not directory_has_contents(empty)
        (empty / "value.txt").write_text("ok", encoding="utf-8")
        assert directory_has_contents(empty)

        assert parse_gpu_worker_slots("0; 3;7") == ["0", "3", "7"]
        try:
            parse_gpu_worker_slots("0,1")
        except ValueError:
            pass
        else:
            raise AssertionError("multi-GPU worker slot must be rejected")

        command = build_single_gpu_train_command(
            root,
            "python",
            root / "config.yaml",
            2,
            root / "output",
        )
        assert command[:4] == [
            "python",
            str(root / "train.py"),
            "--config-file",
            str(root / "config.yaml"),
        ]
        assert command[4:] == [
            "SEED",
            "2",
            "OUTPUT_DIR",
            str(root / "output"),
            "RUN_N_TIMES",
            "1",
            "NUM_GPUS",
            "1",
            "NUM_SHARDS",
            "1",
        ]

        config_file = root / "config.yaml"
        output_root = root / "output"
        baseline_job = BaselineJob(
            model="A2",
            seed=2,
            config_file=config_file,
            output_root=output_root,
            log_path=root / "a2.log",
        )
        prompt_job = PromptDistributionJob(
            method="B1",
            seed=2,
            config_file=config_file,
            output_root=output_root,
            log_path=root / "b1.log",
        )
        expected_project_command = build_single_gpu_train_command(
            PROJECT_ROOT,
            "python",
            config_file,
            2,
            output_root,
        )
        assert baseline_command("python", baseline_job) == expected_project_command
        assert (
            prompt_distribution_command("python", prompt_job)
            == expected_project_command
        )
        assert b3_gpu_groups("0;3;7") == ["0", "3", "7"]
    print("search-plan common helper validation passed")


if __name__ == "__main__":
    main()
