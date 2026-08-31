"""Queue helpers whose behavior is identical across experiment series."""

from __future__ import annotations

from pathlib import Path
from typing import List


def directory_has_contents(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and next(candidate.iterdir(), None) is not None


def parse_gpu_worker_slots(raw: str) -> List[str]:
    """Parse semicolon-separated single-GPU worker slots."""
    values = [value.strip() for value in str(raw).split(";") if value.strip()]
    if not values or any("," in value for value in values):
        raise ValueError("GPU worker slots must be single cards separated by semicolons")
    return values


def build_single_gpu_train_command(
    project_root: str | Path,
    python_bin: str,
    config_file: str | Path,
    seed: int,
    output_root: str | Path,
) -> List[str]:
    """Build the common one-seed, one-GPU train.py command."""
    root = Path(project_root)
    return [
        str(python_bin),
        str(root / "train.py"),
        "--config-file",
        str(config_file),
        "SEED",
        str(int(seed)),
        "OUTPUT_DIR",
        str(output_root),
        "RUN_N_TIMES",
        "1",
        "NUM_GPUS",
        "1",
        "NUM_SHARDS",
        "1",
    ]
