"""Small JSON helpers shared by A/B/C experiment tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON artifact without changing its value types."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(
    path: str | Path,
    payload: Any,
    *,
    allow_nan: bool = True,
    sort_keys: bool = False,
) -> None:
    """Write the project's standard indented UTF-8 JSON representation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=bool(allow_nan),
            sort_keys=bool(sort_keys),
        )
        + "\n",
        encoding="utf-8",
    )


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any] | Any,
    *,
    allow_nan: bool = True,
    sort_keys: bool = False,
) -> None:
    """Write JSON beside its destination and atomically replace the target."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.{}.tmp".format(destination.name, os.getpid())
    )
    try:
        write_json(
            temporary,
            payload,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
        )
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
